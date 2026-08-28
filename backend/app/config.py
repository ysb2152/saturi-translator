"""서버 설정. 환경 변수로 덮어쓸 수 있음."""
import os
from pathlib import Path

# STT 모델 크기: tiny / base / small / medium / large-v3
# 뼈대 단계에서는 CPU에서 빠른 base로 시작. 파인튜닝 후에는 학습된 모델 경로로 교체.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# 파인튜닝된 STT 모델을 CTranslate2로 변환해 둔 디렉터리 경로(있으면 WHISPER_MODEL 대신 사용).
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "").strip()

# 추론 장치: cpu / cuda
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

# 연산 타입: cpu면 int8, GPU면 float16 권장
WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE", "int8" if WHISPER_DEVICE == "cpu" else "float16"
)

# 인식 언어(한국어 고정)
LANGUAGE = os.getenv("LANGUAGE", "ko")

# 업로드 임시 파일 저장 위치
TMP_DIR = Path(os.getenv("TMP_DIR", Path(__file__).resolve().parent.parent / "tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# 허용 오디오 확장자
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".aac"}

# 업로드 최대 크기(바이트). 기본 25MB.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
