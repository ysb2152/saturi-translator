"""표준 라이브러리(wave)만으로 PCM WAV를 다루는 최소 유틸.

AI Hub 방언 오디오는 대체로 PCM WAV(16kHz mono)라 wave 모듈로 충분하다.
다른 포맷(mp3/flac 등)이면 slice가 불가하므로 원본을 그대로 참조하고 duration만 비운다.
"""
from __future__ import annotations

import contextlib
import wave
from pathlib import Path


def wav_duration(path: str | Path) -> float | None:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            return round(w.getnframes() / float(w.getframerate()), 3)
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return None


def slice_wav(src: str | Path, dst: str | Path, start: float, end: float) -> float | None:
    """src의 [start, end] 구간을 잘라 dst로 저장. 저장한 클립 길이(초)를 반환."""
    try:
        with contextlib.closing(wave.open(str(src), "rb")) as w:
            fr = w.getframerate()
            nframes = w.getnframes()
            s = max(0, int(start * fr))
            e = min(nframes, int(end * fr))
            if e <= s:
                return None
            w.setpos(s)
            frames = w.readframes(e - s)
            params = w.getparams()
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return None

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(wave.open(str(dst), "wb")) as out:
        out.setnchannels(params.nchannels)
        out.setsampwidth(params.sampwidth)
        out.setframerate(params.framerate)
        out.writeframes(frames)
    return round((e - s) / float(fr), 3)
