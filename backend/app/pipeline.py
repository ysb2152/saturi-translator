"""사투리 음성 → 표준어 텍스트 파이프라인.

1단계 STT:  사투리 음성 → 사투리 텍스트   (faster-whisper)
2단계 변환: 사투리 텍스트 → 표준어 텍스트  (지금은 규칙기반 스텁, 이후 KoBART로 교체)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from . import config

logger = logging.getLogger("pipeline")


@dataclass
class TranslationResult:
    dialect_text: str      # STT가 뽑은 사투리 텍스트
    standard_text: str     # 표준어로 변환된 텍스트
    language: str
    duration: float        # 오디오 길이(초)


class Transcriber:
    """faster-whisper 래퍼. 모델은 첫 사용 시 지연 로딩."""

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        model_path = config.WHISPER_MODEL_DIR or config.WHISPER_MODEL
        logger.info(
            "Whisper 모델 로딩: %s (device=%s, compute=%s)",
            model_path, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE,
        )
        self._model = WhisperModel(
            model_path,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        return self._model

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        """오디오 파일 경로를 받아 (텍스트, 오디오길이초)를 반환."""
        model = self._load()
        segments, info = model.transcribe(
            audio_path,
            language=config.LANGUAGE,
            beam_size=5,
            vad_filter=True,  # 무음 구간 제거로 환각(hallucination) 감소
        )
        text = "".join(seg.text for seg in segments).strip()
        return text, float(info.duration)


class DialectConverter:
    """사투리 텍스트 → 표준어 텍스트.

    지금은 파인튜닝 모델이 없으므로 아주 얕은 규칙 사전으로 동작하는 '스텁'.
    E2E 파이프라인이 도는 것을 확인하는 용도이며, 2단계에서 KoBART/T5 seq2seq
    모델로 이 클래스만 교체하면 된다(convert 시그니처 유지).
    """

    # 데모용 최소 사전. 실제 개선은 AI Hub 데이터로 학습한 모델이 담당.
    _RULES = {
        "머라카노": "뭐라고 하니",
        "머라꼬": "뭐라고",
        "와이라노": "왜 이러니",
        "우야노": "어떻게 하니",
        "가가": "그 아이가",
        "댕겨왔다": "다녀왔다",
        "억수로": "굉장히",
        "쫌": "좀",
        "맞나": "맞니",
        "뭐라는겨": "뭐라는 거야",
    }

    def convert(self, dialect_text: str) -> str:
        text = dialect_text
        for dia, std in self._RULES.items():
            text = text.replace(dia, std)
        return text


class Pipeline:
    def __init__(self) -> None:
        self.transcriber = Transcriber()
        self.converter = DialectConverter()

    def warmup(self) -> None:
        """서버 기동 시 모델을 미리 로딩(첫 요청 지연 방지)."""
        self.transcriber._load()

    def run(self, audio_path: str) -> TranslationResult:
        dialect_text, duration = self.transcriber.transcribe(audio_path)
        standard_text = self.converter.convert(dialect_text)
        return TranslationResult(
            dialect_text=dialect_text,
            standard_text=standard_text,
            language=config.LANGUAGE,
            duration=duration,
        )


# 애플리케이션 전역 싱글턴
pipeline = Pipeline()
