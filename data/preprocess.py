"""AI Hub 방언 발화 raw 데이터 → 학습용 산출물 전처리.

산출물 2종:
  1) STT 매니페스트 (Whisper 파인튜닝용):  {"audio_filepath","text","duration"}
     - text = 방언(사투리) 전사. 표준 Whisper 대비 개선을 측정할 핵심 데이터.
  2) 변환 문장쌍 (KoBART/T5용):            {"dialect","standard"}

입력 가정:
  raw/
    labels/*.json      (또는 raw 전체를 재귀 탐색; --labels-glob 로 조정)
    audio/*.wav        (세션 오디오; 발화 start/end 로 슬라이스)
  * 오디오가 이미 발화 단위로 잘려 있으면 --no-slice 로 원본 참조.

사용:
  python data/preprocess.py --raw data/sample_raw --out data/processed
  python data/preprocess.py --raw <AIHub_추출경로> --out data/processed --val-ratio 0.05
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aihub_schema import (extract_utterance, get_session_audio_name,
                          get_utterance_list, parse_txt_label)
from wav_utils import slice_wav, wav_duration

AUDIO_EXTS = [".wav", ".WAV"]


def build_audio_index(raw_root: Path) -> dict[str, Path]:
    """raw 전체를 '한 번만' 훑어 오디오 basename(소문자) → 경로 맵을 만든다.

    라벨 파일마다 rglob 하면 O(n^2)라 파일이 많을 때 매우 느리다. 인덱스로 dict 조회.
    """
    idx: dict[str, Path] = {}
    for ext in ("*.wav", "*.WAV"):
        for p in raw_root.rglob(ext):
            idx.setdefault(p.name.lower(), p)
    return idx


def find_session_audio(label_path: Path, obj: dict, audio_index: dict[str, Path]) -> Path | None:
    """라벨에 대응하는 세션 오디오를 인덱스에서 찾는다(basename 기준)."""
    if not audio_index:
        return None
    # 1) 메타데이터에 명시된 파일명
    name = get_session_audio_name(obj)
    if name and Path(name).name.lower() in audio_index:
        return audio_index[Path(name).name.lower()]
    # 2) 라벨 파일명과 같은 basename
    for ext in AUDIO_EXTS:
        key = (label_path.stem + ext).lower()
        if key in audio_index:
            return audio_index[key]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="AI Hub 추출 루트(라벨+오디오)")
    ap.add_argument("--out", default="data/processed", help="산출물 출력 루트")
    ap.add_argument("--labels-glob", default="**/*.json", help="라벨 JSON glob")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="검증셋 비율(세션 단위 분할)")
    ap.add_argument("--min-dur", type=float, default=0.3, help="이 길이(초) 미만 발화 제외")
    ap.add_argument("--max-dur", type=float, default=30.0, help="이 길이(초) 초과 발화 제외(Whisper 30s)")
    ap.add_argument("--no-slice", action="store_true", help="오디오가 이미 발화 단위면 슬라이스 생략")
    ap.add_argument("--limit", type=int, default=0, help="라벨 파일 수 제한(디버그)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    clip_dir = out_root / "stt" / "audio"
    (out_root / "stt").mkdir(parents=True, exist_ok=True)
    (out_root / "mt").mkdir(parents=True, exist_ok=True)

    json_files = sorted(raw_root.glob(args.labels_glob))
    # JSON이 없는 세션은 .txt(이중전사)로 대체 — 지역에 따라 .txt만 제공됨
    json_stems = {f.stem.lower() for f in json_files}
    txt_files = [f for f in raw_root.rglob("*.txt") if f.stem.lower() not in json_stems]
    label_files = json_files + txt_files
    if args.limit:
        label_files = label_files[: args.limit]
    if not label_files:
        raise SystemExit(f"라벨(.json/.txt)을 찾지 못했습니다: {raw_root}")
    print(f"라벨: JSON {len(json_files)}개 + TXT {len(txt_files)}개")

    # 오디오 인덱스를 한 번만 구축(라벨만 있으면 비어 있음 → STT는 자연히 skip)
    audio_index = build_audio_index(raw_root)
    print(f"라벨 {len(label_files)}개, 오디오 {len(audio_index)}개 인덱싱 완료")

    rng = random.Random(args.seed)
    # 세션(라벨 파일) 단위로 train/val 분할 → 화자/세션 누수 방지
    val_set = set(f for f in label_files if rng.random() < args.val_ratio)

    stt = {"train": [], "val": []}
    mt = {"train": [], "val": []}
    stats = {
        "label_files": len(label_files), "utterances": 0, "kept": 0,
        "skipped_no_audio": 0, "skipped_dur": 0, "skipped_empty": 0,
        "identical_pairs": 0, "parse_errors": 0,
    }

    for lf in label_files:
        split = "val" if lf in val_set else "train"
        try:
            if lf.suffix.lower() == ".json":
                obj = json.loads(lf.read_text(encoding="utf-8"))
                utts = [extract_utterance(u) for u in get_utterance_list(obj)]
            else:  # .txt (이중전사)
                obj = {}
                utts = parse_txt_label(lf.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["parse_errors"] += 1
            continue

        session_audio = None if args.no_slice else find_session_audio(lf, obj, audio_index)

        for i, u in enumerate(utts):
            stats["utterances"] += 1
            if u is None:
                stats["skipped_empty"] += 1
                continue

            # 변환 문장쌍(오디오 불필요)
            mt[split].append({"dialect": u.dialect, "standard": u.standard})
            if u.dialect == u.standard:
                stats["identical_pairs"] += 1

            # STT 매니페스트(오디오 필요)
            audio_path, dur = None, u.duration
            if args.no_slice:
                # 발화 단위 오디오: 라벨 stem 또는 utterance id 로 매칭 시도
                cand = find_session_audio(lf, obj, audio_index)
                if cand:
                    audio_path = cand
                    dur = dur or wav_duration(cand)
            elif session_audio and u.start is not None and u.end is not None:
                clip = clip_dir / f"{lf.stem}_{i:04d}.wav"
                got = slice_wav(session_audio, clip, u.start, u.end)
                if got:
                    audio_path, dur = clip, got

            if audio_path is None:
                stats["skipped_no_audio"] += 1
                continue
            if dur is not None and not (args.min_dur <= dur <= args.max_dur):
                stats["skipped_dur"] += 1
                continue

            stt[split].append({
                "audio_filepath": str(Path(audio_path).as_posix()),
                "text": u.dialect,
                "duration": dur,
            })
            stats["kept"] += 1

    # 저장
    def dump(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for sp in ("train", "val"):
        dump(stt[sp], out_root / "stt" / f"{sp}.jsonl")
        dump(mt[sp], out_root / "mt" / f"{sp}.jsonl")
    (out_root / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 전처리 완료 ===")
    print(f"STT  train={len(stt['train'])}  val={len(stt['val'])}")
    print(f"MT   train={len(mt['train'])}  val={len(mt['val'])}")
    print(f"stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"출력: {out_root}")


if __name__ == "__main__":
    main()
