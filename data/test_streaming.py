"""스트리밍 STT 전처리 스모크 테스트(실데이터 불필요).

  python data/test_streaming.py
합성 샘플(오디오+라벨) 생성 → 오디오를 zip으로 묶음 → 스트리밍 전처리 → 산출물 검증.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_streaming_end_to_end():
    # 1) 합성 샘플 생성(sample_raw/audio, sample_raw/labels)
    subprocess.run([sys.executable, str(HERE / "make_sample.py")], check=True,
                   stdout=subprocess.DEVNULL)
    sample = HERE / "sample_raw"
    wavs = list((sample / "audio").glob("*.wav"))
    assert wavs, "샘플 오디오가 없음"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zip_path = td / "audio.zip"
        # 2) 오디오를 zip으로(멤버명 = 파일명, 라벨 stem과 일치)
        with zipfile.ZipFile(zip_path, "w") as zf:
            for w in wavs:
                zf.write(w, arcname=w.name)

        out = td / "processed"
        # 3) 스트리밍 전처리 실행
        r = subprocess.run(
            [sys.executable, str(HERE / "preprocess_streaming.py"),
             "--zip", str(zip_path), "--labels", str(sample / "labels"),
             "--out", str(out), "--val-ratio", "0.5"],
            check=True, capture_output=True, text=True, encoding="utf-8")
        print("   " + r.stdout.strip().replace("\n", "\n   "))

        # 4) 검증
        recs = []
        for sp in ("train", "val"):
            p = out / "stt" / f"{sp}.jsonl"
            if p.exists():
                recs += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert recs, "STT 매니페스트가 비어있음"
        r0 = recs[0]
        assert {"audio_filepath", "text", "duration"} <= r0.keys()
        assert Path(r0["audio_filepath"]).exists(), "스트리밍으로 자른 클립이 실제로 없음"
        assert r0["duration"] and r0["duration"] > 0

        # temp WAV 가 남지 않았는지(정리 확인)
        assert not (out / "stt" / "audio" / "_stream_tmp.wav").exists(), "temp WAV 미삭제"

        stats = json.loads((out / "stt_stream_stats.json").read_text(encoding="utf-8"))
        assert stats["clips"] == len(recs)
        print(f"  [ok] streaming end_to_end (클립 {stats['clips']}개, temp 정리 확인)")


if __name__ == "__main__":
    print("스트리밍 전처리 테스트 시작")
    test_streaming_end_to_end()
    print("통과 ✅")
