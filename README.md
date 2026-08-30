# 사투리 → 표준어 음성 번역기

경상도 사투리 음성을 녹음하면 **표준어 텍스트로 변환**하는 안드로이드 앱. 표준 음성/언어 모델이 잘 못하는 **사투리 영역을 AI Hub 방언 데이터로 파인튜닝해 개선**하는 것이 핵심이다.

> 취업 포트폴리오 프로젝트. "모델을 직접 학습시킨 경험"과 "앱을 배포한 경험"을 함께 채우는 것을 목표로 한다. (임시 저장소명: `translate`)

---

## 무엇을 푸는가

부모·조부모 세대의 사투리를 잘 못 알아듣는 자녀 세대, 사투리 콘텐츠 자막화·고객센터 등에서 쓸 수 있는 시나리오. 표준 Whisper는 사투리 발음/억양에서, 규칙 기반 변환은 다양한 방언 표현에서 한계가 있다. 이 격차를 **데이터로 좁힌다.**

## 파이프라인

```
음성 ──[① STT: Whisper]──▶ 사투리 텍스트 ──[② 변환: KoBART]──▶ 표준어 텍스트
       (사투리 음성 인식)                    (사투리→표준어 재작성)
```

- **① STT** — Whisper. 표준 Whisper를 사투리 음성으로 파인튜닝해 인식률 개선(whisper-small **CER 14.6%→6.7%**). 서빙은 파인튜닝 모델을 transformers로 직접.
- **② 변환** — `KoBART` seq2seq를 사투리→표준어 문장쌍으로 파인튜닝(진행 중).
- 앱은 얇게: 녹음 → 서버(FastAPI) 전송 → 결과 표시.

## 핵심 서사 — 데이터로 개선

AI Hub 경상도 방언 라벨을 전처리해 **208만 문장쌍**을 확보했고, 분석 결과:

| 지표 | 값 |
|------|-----|
| 총 문장쌍 | 2,088,334 |
| 실제 방언 변형(사투리≠표준) | 336,121 (16.1%) |
| 가장 흔한 치환 | `쫌`→`조금` (139,427회), `이케`→`이렇게`, `그니까`→`그러니까`, `니`→`너`, `걍`→`그냥` … |

→ 규칙 사전으로는 다 담기 어려운 이 다양한 변형을 **모델이 데이터로 학습**한다. KoBART 변환 모델은 '그대로 두기(copy)' 기준선 대비 크게 개선됐다:
- **경상도 단일**: CER 4.62% → 1.01%, 정확일치 29.6% → 82.2%
- **4지역(경상·전라·충청·강원)**: CER 5.79% → 1.63%, 정확일치 26.5% → 77.7%

그리고 **STT(Whisper)** 도 표준 대비 사투리 인식 CER을 **14.6% → 6.7%(54% 개선, whisper-small)** 로 낮췄다. 즉 음성 인식·표준어 변환 **두 단계 모두** "표준 모델이 못하는 것을 데이터로 개선"을 정량 증명했다.

전체 분석은 [data/analysis.md](data/analysis.md), 파인튜닝 설계·근거는 [why_finetune.md](why_finetune.md) 참고.

## 기술 스택

- **앱**: React Native (Expo SDK 57), `expo-audio` 녹음, `expo-file-system` 업로드
- **백엔드**: FastAPI, `faster-whisper`(STT), KoBART(변환)
- **학습**: HuggingFace `transformers`, Google Colab 무료 GPU
- **데이터**: AI Hub 한국어 방언 발화(경상도), 표준 라이브러리 기반 전처리

## 저장소 구조

```
backend/      FastAPI 서버 (STT + 변환 파이프라인)
mobile/       React Native(Expo) 앱
data/         AI Hub 전처리·분석 스크립트 (원본 데이터는 미포함)
notebooks/    Colab 노트북 (데이터 준비, KoBART 파인튜닝)
DEVELOPMENT_JOURNEY.md   개발 기록(설계·의사결정·문제해결)
why_finetune.md          파인튜닝 모델·방식·하이퍼파라미터 근거
```

## 실행

**백엔드**
```bash
cd backend
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**앱** (에뮬레이터 또는 Expo Go)
```bash
cd mobile
npm install
npm run android
```
자세한 실행·서버 연결은 [backend/README.md](backend/README.md), [mobile/README.md](mobile/README.md).

**데이터 전처리** (AI Hub 데이터는 한국 IP에서 직접 다운로드)
```bash
python data/preprocess.py --raw <AIHub_추출경로> --out data/processed
python data/build_mt_dataset.py       # 변환 학습셋(균형)
```

## 현재 상태

- [x] 백엔드 E2E 파이프라인 (STT + 변환 API)
- [x] 앱 뼈대 + "한지×클린" 디자인, 녹음→변환 왕복 확인
- [x] AI Hub 전처리 + 208만 문장쌍 + 데이터 분석
- [x] KoBART 변환 모델 파인튜닝 — 경상 단일 **CER 4.6%→1.0%**, 4지역 통합 **CER 5.8%→1.6%** (copy 대비) + 백엔드 연동
- [x] Whisper STT 파인튜닝 — 표준 대비 **CER 14.6%→6.7%**(54% 개선, whisper-small), 로컬 GPU 학습
- [ ] 플레이스토어 배포

진행 상세는 [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md).
