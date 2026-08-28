"""AI Hub 방언 라벨 구조를 흉내 낸 합성 샘플 생성(전처리 파이프라인 검증용).

실제 데이터가 없어도 preprocess.py 가 도는지 확인하기 위한 것이다.
data/sample_raw/{audio,labels} 에 세션 WAV + 라벨 JSON 을 만든다.
"""
from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

# (방언, 표준) 예시 문장쌍
PAIRS = [
    ("가가 억수로 머라카노", "그 아이가 굉장히 뭐라고 하니"),
    ("밥 문나", "밥 먹었니"),
    ("여 앉아 있그라", "여기 앉아 있어라"),
    ("와 이래 쌌노", "왜 이렇게 그러니"),
    ("퍼뜩 온나", "빨리 오너라"),
    ("고마 치아라", "그만 치워라"),
]

SR = 16000


def write_tone_wav(path: Path, seconds: float, freq: float = 220.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(SR * seconds)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        for i in range(n):
            val = int(2500 * math.sin(2 * math.pi * freq * i / SR))
            w.writeframes(struct.pack("<h", val))


def make_session(idx: int, root: Path, pairs):
    """한 세션 = 여러 발화. 세션 오디오 1개 + 라벨 JSON 1개."""
    audio_name = f"session_{idx:02d}.wav"
    audio_path = root / "audio" / audio_name

    utterances, t = [], 0.0
    for j, (dia, std) in enumerate(pairs):
        dur = 1.2 + 0.3 * j          # 발화별 길이 조금씩 다르게
        start, end = t, t + dur
        t = end + 0.2                # 발화 사이 간격
        utterances.append({
            "id": f"u{j+1}",
            "form": dia,
            "dialect_form": dia,
            "standard_form": std,
            "eojeolList": [
                {"eojeol": w, "standard": w, "isDialect": False} for w in dia.split()
            ],
            "start": round(start, 3),
            "end": round(end, 3),
        })

    write_tone_wav(audio_path, seconds=t + 0.5, freq=200 + 40 * idx)

    label = {
        "metadata": {"audioPath": f"audio/{audio_name}", "region": "gyeongsang", "session": idx},
        "utterance": utterances,
    }
    label_path = root / "labels" / f"session_{idx:02d}.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent / "sample_raw"
    make_session(1, root, PAIRS)
    make_session(2, root, list(reversed(PAIRS)))
    print(f"샘플 생성 완료: {root}")


if __name__ == "__main__":
    main()
