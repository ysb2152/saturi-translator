"""STT용 스트리밍 전처리 — 큰 오디오 zip을 통째로 풀지 않고 처리.

AI Hub 원천(음성) zip은 하나가 17~28GB라, 압축을 다 풀면(중복 용량) 로컬 34GB에 안 들어간다.
그래서 zip 안에서 WAV를 **하나씩** 꺼내(temp) → 라벨의 발화 구간으로 잘라 작은 클립 저장 →
temp WAV 삭제, 를 반복한다. 피크 디스크 = (zip 크기) + (temp WAV 1개) + (누적 클립).
--max-clips 로 개수를 제한하면 포트폴리오용 소량만 뽑아 무료 드라이브에 올릴 수 있다.

라벨(전사 JSON)은 먼저 따로 받아 풀어둔 경로(--labels)를 쓴다(작아서 부담 없음).

  python data/preprocess_streaming.py \
      --zip "(비식별화완료)경상도_1.zip" \
      --labels data/raw_label \
      --out data/processed --max-clips 20000
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
import zipfile
from bisect import bisect_right
from hashlib import md5
from pathlib import Path

from aihub_schema import extract_utterance, get_utterance_list
from wav_utils import slice_wav

AUDIO_SUFFIXES = (".wav", ".WAV")


class _ConcatParts(io.RawIOBase):
    """AI Hub가 tar 안에 `xxx.zip.part<offset>` 로 1GB씩 쪼개 담은 zip을,
    재구성(디스크 복사) 없이 하나의 seekable 스트림처럼 제공한다. 실제 바이트는
    tar 파일에서 직접 읽어, zipfile이 그대로 멤버를 추출할 수 있게 한다."""

    def __init__(self, tar_path: str):
        self._f = open(tar_path, "rb")
        parts = []  # (zip_offset, tar_data_offset, size)
        with tarfile.open(tar_path) as t:
            for m in t.getmembers():
                if ".zip.part" in m.name:
                    off = int(m.name.rsplit(".part", 1)[1])
                    parts.append((off, m.offset_data, m.size))
        parts.sort()
        self._starts = [p[0] for p in parts]
        self._parts = parts
        self._size = parts[-1][0] + parts[-1][2] if parts else 0
        self._pos = 0

    def seekable(self):
        return True

    def seek(self, pos, whence=0):
        self._pos = pos if whence == 0 else (self._pos + pos if whence == 1 else self._size + pos)
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        out = bytearray()
        while n > 0 and self._pos < self._size:
            i = bisect_right(self._starts, self._pos) - 1
            zoff, toff, size = self._parts[i]
            within = self._pos - zoff
            take = min(n, size - within)
            self._f.seek(toff + within)
            chunk = self._f.read(take)
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_audio_zip(path: str) -> zipfile.ZipFile:
    """일반 zip 이면 그대로, tar(내부 .zip.partN)면 가상 스트림으로 연다."""
    if tarfile.is_tarfile(path):
        print("tar(분할 zip) 감지 → 가상 스트림으로 처리(재구성 없이)")
        return zipfile.ZipFile(_ConcatParts(path))
    return zipfile.ZipFile(path)


def build_label_index(labels_root: Path) -> dict[str, Path]:
    """라벨 stem(예: DKCI20000001) → JSON 경로."""
    idx = {}
    for p in labels_root.rglob("*.json"):
        idx.setdefault(p.stem.lower(), p)
    return idx


def split_of(stem: str, val_ratio: float) -> str:
    """세션 stem 해시로 결정적 train/val 분할(청크를 나눠 돌려도 일관)."""
    h = int(md5(stem.encode("utf-8")).hexdigest(), 16) % 10000
    return "val" if h < val_ratio * 10000 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="AI Hub 원천(음성) zip 경로")
    ap.add_argument("--labels", required=True, help="라벨 JSON 들이 풀려있는 루트")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--max-clips", type=int, default=0, help="누적 클립 상한(0=무제한)")
    ap.add_argument("--append", action="store_true", help="기존 매니페스트에 이어쓰기(여러 zip 연속 처리)")
    args = ap.parse_args()

    out_root = Path(args.out)
    clip_dir = out_root / "stt" / "audio"
    clip_dir.mkdir(parents=True, exist_ok=True)
    man = {sp: out_root / "stt" / f"{sp}.jsonl" for sp in ("train", "val")}
    mode = "a" if args.append else "w"
    fhs = {sp: open(man[sp], mode, encoding="utf-8") for sp in ("train", "val")}

    label_index = build_label_index(Path(args.labels))
    print(f"라벨 인덱스 {len(label_index)}개")
    if not label_index:
        raise SystemExit("라벨 JSON을 찾지 못했습니다(--labels 확인).")

    stats = {"wav_members": 0, "matched": 0, "no_label": 0, "clips": 0,
             "skipped_dur": 0, "skipped_slice": 0}
    tmp_wav = clip_dir / "_stream_tmp.wav"

    with open_audio_zip(args.zip) as zf:
        members = [m for m in zf.namelist() if m.endswith(AUDIO_SUFFIXES)]
        print(f"zip 내 WAV {len(members)}개 — 스트리밍 처리 시작")
        for mi, member in enumerate(members):
            stem = Path(member).stem
            stats["wav_members"] += 1
            label_path = label_index.get(stem.lower())
            if label_path is None:
                stats["no_label"] += 1
                continue
            stats["matched"] += 1

            # zip 멤버 하나만 temp 로 스트리밍 추출(메모리 저부하)
            with zf.open(member) as src, open(tmp_wav, "wb") as dst:
                shutil.copyfileobj(src, dst)

            try:
                obj = json.loads(label_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            split = split_of(stem, args.val_ratio)

            for i, raw_utt in enumerate(get_utterance_list(obj)):
                u = extract_utterance(raw_utt)
                if u is None or u.start is None or u.end is None:
                    continue
                clip = clip_dir / f"{stem}_{i:04d}.wav"
                dur = slice_wav(tmp_wav, clip, u.start, u.end)
                if not dur:
                    stats["skipped_slice"] += 1
                    continue
                if not (args.min_dur <= dur <= args.max_dur):
                    clip.unlink(missing_ok=True)
                    stats["skipped_dur"] += 1
                    continue
                fhs[split].write(json.dumps(
                    {"audio_filepath": str(clip.as_posix()), "text": u.dialect,
                     "duration": dur}, ensure_ascii=False) + "\n")
                stats["clips"] += 1

            if (mi + 1) % 50 == 0:
                print(f"  {mi+1}/{len(members)} 세션, 누적 클립 {stats['clips']:,}")
            if args.max_clips and stats["clips"] >= args.max_clips:
                print(f"--max-clips {args.max_clips} 도달 → 중단")
                break

    tmp_wav.unlink(missing_ok=True)
    for fh in fhs.values():
        fh.close()
    (out_root / "stt_stream_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== 완료 ===")
    print(json.dumps(stats, ensure_ascii=False))
    print(f"클립: {clip_dir}  매니페스트: {man['train']}, {man['val']}")


if __name__ == "__main__":
    main()
