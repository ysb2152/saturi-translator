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
    """STT 래퍼. config.WHISPER_FT_DIR 이 있으면 파인튜닝 Whisper(transformers)로,
    없으면 faster-whisper로 인식. 모델은 첫 사용 시 지연 로딩."""

    def __init__(self) -> None:
        self._model = None
        self._ft = None      # 파인튜닝 Whisper(transformers)
        self._proc = None

    def _load_ft(self):
        if self._ft is not None:
            return
        import torch  # noqa: F401
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("파인튜닝 Whisper 로딩: %s", config.WHISPER_FT_DIR)
        self._proc = WhisperProcessor.from_pretrained(
            config.WHISPER_FT_DIR, language="korean", task="transcribe")
        self._ft = WhisperForConditionalGeneration.from_pretrained(
            config.WHISPER_FT_DIR).to(config.WHISPER_DEVICE).eval()

    def _transcribe_ft(self, audio_path: str) -> tuple[str, float]:
        import torch
        from faster_whisper.audio import decode_audio  # av 기반 디코딩 재사용

        arr = decode_audio(audio_path, sampling_rate=16000)
        dur = len(arr) / 16000.0
        feats = self._proc(arr, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(config.WHISPER_DEVICE)
        with torch.no_grad():
            gen = self._ft.generate(feats, max_new_tokens=128,
                                    no_repeat_ngram_size=3)  # 반복 환각 억제
        text = self._proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return text, float(dur)

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
        if config.WHISPER_FT_DIR:
            self._load_ft()
            return self._transcribe_ft(audio_path)
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

    config.CONVERTER_MODEL_DIR 이 설정돼 있으면 파인튜닝한 KoBART seq2seq 모델로 추론하고,
    없으면 얕은 규칙 사전(스텁)으로 폴백한다. 모델은 첫 사용 시 지연 로딩.
    학습은 notebooks/kobart_finetune.ipynb 참고.
    """

    # 규칙 폴백용 최소 사전(모델 없을 때만 사용).
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

    def __init__(self) -> None:
        self._model = None
        self._tok = None

    def _load(self):
        """KoBART 모델/토크나이저 지연 로딩. 실패하면 규칙 폴백."""
        if self._model is not None or not config.CONVERTER_MODEL_DIR:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info("변환 모델 로딩: %s", config.CONVERTER_MODEL_DIR)
            self._tok = AutoTokenizer.from_pretrained(config.CONVERTER_MODEL_DIR)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(config.CONVERTER_MODEL_DIR)
            self._model.to(config.CONVERTER_DEVICE).eval()
        except Exception as e:  # 로딩 실패 → 규칙 폴백
            logger.warning("변환 모델 로딩 실패(규칙 스텁으로 폴백): %s", e)
            self._model = None

    def _rule_convert(self, text: str) -> str:
        for dia, std in self._RULES.items():
            text = text.replace(dia, std)
        return text

    def convert(self, dialect_text: str) -> str:
        if not dialect_text:
            return dialect_text
        self._load()
        if self._model is None:  # 모델 없음/실패 → 규칙
            return self._rule_convert(dialect_text)
        import torch

        # KoBART 토크나이저의 token_type_ids는 BART generate가 받지 않으므로 제외
        enc = self._tok(dialect_text, return_tensors="pt", truncation=True,
                        max_length=128, return_token_type_ids=False).to(config.CONVERTER_DEVICE)
        with torch.no_grad():
            gen = self._model.generate(
                **enc, max_new_tokens=64, num_beams=4,
                no_repeat_ngram_size=3, repetition_penalty=1.3, early_stopping=True)
        return self._tok.batch_decode(gen, skip_special_tokens=True)[0].strip()


class Pipeline:
    def __init__(self) -> None:
        self.transcriber = Transcriber()
        self.converter = DialectConverter()

    def warmup(self) -> None:
        """서버 기동 시 모델을 미리 로딩(첫 요청 지연 방지)."""
        if config.WHISPER_FT_DIR:
            self.transcriber._load_ft()
        else:
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
