"""전처리 파이프라인 스모크 테스트(실제 데이터 불필요).

  python data/test_preprocess.py
합성 샘플 생성 → 전처리 실행 → 산출물 검증까지 한 번에 돈다.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 정제 로직 단위 검사
from aihub_schema import clean_text, extract_utterance


def test_clean_text():
    assert clean_text("가가 (억수로)/(굉장히) 머라카노", "dialect") == "가가 억수로 머라카노"
    assert clean_text("가가 (억수로)/(굉장히) 머라카노", "standard") == "가가 굉장히 머라카노"
    assert clean_text("밥 문나b/ {laughing}", "dialect") == "밥 문나"
    assert clean_text("(())", "dialect") == ""
    print("  [ok] clean_text")


def test_extract():
    u = extract_utterance({
        "dialect_form": "가가 억수로 머라카노",
        "standard_form": "그 아이가 굉장히 뭐라고 하니",
        "start": 1.0, "end": 3.5,
    })
    assert u is not None
    assert u.dialect == "가가 억수로 머라카노"
    assert u.standard == "그 아이가 굉장히 뭐라고 하니"
    assert u.duration == 2.5
    # 표준 필드 없으면 방언으로 대체
    u2 = extract_utterance({"dialect_form": "밥 문나"})
    assert u2.standard == "밥 문나"
    print("  [ok] extract_utterance")


def test_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "processed"
        # 1) 샘플 생성
        subprocess.run([sys.executable, str(HERE / "make_sample.py")], check=True,
                       stdout=subprocess.DEVNULL)
        # 2) 전처리 실행
        r = subprocess.run(
            [sys.executable, str(HERE / "preprocess.py"),
             "--raw", str(HERE / "sample_raw"), "--out", str(out), "--val-ratio", "0.5"],
            check=True, capture_output=True, text=True, encoding="utf-8")
        print("   " + r.stdout.strip().replace("\n", "\n   "))

        # 3) 산출물 검증
        stt_train = (out / "stt" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        mt_train = (out / "mt" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        assert stt_train or (out / "stt" / "val.jsonl").read_text(encoding="utf-8"), "STT 비어있음"

        rec = json.loads(stt_train[0]) if stt_train else json.loads(
            (out / "stt" / "val.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert {"audio_filepath", "text", "duration"} <= rec.keys()
        assert Path(rec["audio_filepath"]).exists(), "슬라이스된 클립이 실제로 없음"
        assert rec["duration"] and rec["duration"] > 0

        mrec = json.loads(mt_train[0]) if mt_train else None
        if mrec:
            assert {"dialect", "standard"} <= mrec.keys()

        stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
        assert stats["kept"] > 0
        print(f"  [ok] end_to_end (kept={stats['kept']} utterances, 클립 슬라이스 확인)")


if __name__ == "__main__":
    print("전처리 테스트 시작")
    test_clean_text()
    test_extract()
    test_end_to_end()
    print("전체 통과 ✅")
