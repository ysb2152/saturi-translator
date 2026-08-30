"""FastAPI 서버: 사투리 음성을 받아 표준어 텍스트로 변환.

실행:  uvicorn app.main:app --reload   (backend/ 디렉터리에서)
문서:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .pipeline import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 모델 미리 로딩(선택). 느린 첫 요청을 피함.
    if os.getenv("WARMUP", "1") == "1":
        try:
            pipeline.warmup()
            logger.info("모델 워밍업 완료")
        except Exception as e:  # 모델 로딩 실패해도 서버는 뜨게 함
            logger.warning("워밍업 실패(첫 요청 때 재시도): %s", e)
    yield


app = FastAPI(title="사투리→표준어 음성 번역 API", version="0.1.0", lifespan=lifespan)

# 개발 중 앱/웹에서 호출 가능하도록 CORS 개방(배포 시 도메인 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": config.WHISPER_MODEL_DIR or config.WHISPER_MODEL}


# 녹음 없이 앱에서 파이프라인을 검증하기 위한 데모: 서버의 사투리 샘플을 무작위로 처리.
import random
from pathlib import Path as _Path
_SAMPLES = _Path(__file__).resolve().parent.parent / "samples"


@app.get("/demo")
def demo():
    clips = list(_SAMPLES.glob("*.wav"))
    if not clips:
        raise HTTPException(status_code=404, detail="샘플 클립이 없습니다.")
    clip = random.choice(clips)
    result = pipeline.run(str(clip))
    return {
        "dialect_text": result.dialect_text,
        "standard_text": result.standard_text,
        "language": result.language,
        "duration": round(result.duration, 2),
        "sample": clip.name,
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 확장자 검증
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in config.ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=415,
            detail=f"지원하지 않는 형식: {ext or '알수없음'}. 허용: {sorted(config.ALLOWED_AUDIO_EXT)}",
        )

    # 임시 파일로 저장(크기 제한 확인하며 스트리밍)
    tmp_path = config.TMP_DIR / f"{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with open(tmp_path, "wb") as f:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다.")
                f.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        result = pipeline.run(str(tmp_path))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("변환 실패")
        raise HTTPException(status_code=500, detail=f"변환 중 오류: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "dialect_text": result.dialect_text,
        "standard_text": result.standard_text,
        "language": result.language,
        "duration": round(result.duration, 2),
    }
