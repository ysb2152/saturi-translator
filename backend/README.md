# backend — 사투리→표준어 음성 번역 API

FastAPI 서버. 오디오 파일을 받아 STT(faster-whisper)로 사투리 텍스트를 뽑고,
2단계 변환기로 표준어 텍스트를 만들어 반환한다.

현재 단계: **E2E 뼈대**. STT는 기본 Whisper(`base`), 2단계 변환은 규칙기반 스텁.
이후 AI Hub 경상도 방언 데이터로 STT 파인튜닝 + KoBART 변환기 학습으로 교체 예정.

## 설치 & 실행 (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

첫 실행 시 Whisper `base` 모델(약 140MB)을 자동 다운로드한다.

- API 문서(스웨거): http://127.0.0.1:8000/docs
- 헬스체크: http://127.0.0.1:8000/health

## 엔드포인트

### `POST /transcribe`
`multipart/form-data`로 `audio` 파일 업로드.

```powershell
curl.exe -X POST http://127.0.0.1:8000/transcribe -F "audio=@sample.wav"
```

응답:
```json
{
  "dialect_text": "가가 억수로 머라카노",
  "standard_text": "그 아이가 굉장히 뭐라고 하니",
  "language": "ko",
  "duration": 2.3
}
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `WHISPER_MODEL` | `base` | 모델 크기(tiny/base/small/medium/large-v3) |
| `WHISPER_MODEL_DIR` | (없음) | 파인튜닝된 CTranslate2 모델 경로. 있으면 우선 사용 |
| `WHISPER_DEVICE` | `cpu` | `cpu` 또는 `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8`/`float16` | 연산 타입 |
| `WARMUP` | `1` | 기동 시 모델 미리 로딩 |

## 다음 단계

1. AI Hub 경상도 방언 데이터 다운로드 → 전처리(`data/` 스크립트, 예정)
2. Whisper 파인튜닝(Colab) → CTranslate2 변환 → `WHISPER_MODEL_DIR`로 연결
3. `DialectConverter`를 KoBART seq2seq 추론으로 교체
4. React Native(Expo) 앱에서 `/transcribe` 호출
