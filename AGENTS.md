# 프로젝트 컨텍스트 — 사투리 → 표준어 음성 번역

AI 코딩 툴이 이 저장소의 맥락을 빠르게 파악하기 위한 요약. 상세 진행은 [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md), 정량 평가는 [EVALUATION.md](EVALUATION.md), 파인튜닝 근거는 [why_finetune.md](why_finetune.md) 참고.

## 개요
사투리 음성을 녹음하면 표준어 텍스트로 변환하는 안드로이드 앱. **2단계 파이프라인**:
`음성 → ① STT(Whisper) → 사투리 텍스트 → ② 변환(KoBART) → 표준어 텍스트`
핵심: 표준 모델이 약한 사투리를 **AI Hub 방언 데이터로 파인튜닝해 개선**.

## 저장소 구조
- `backend/` — FastAPI 서버. `app/pipeline.py`(STT+변환 추론), `app/main.py`(`/transcribe`,`/health`,`/demo`), `app/config.py`(모델 경로 자동감지). 모델은 `backend/models/`(gitignore).
- `mobile/` — React Native(Expo SDK 57) 앱. 녹음→서버 전송→결과. `mobile/AGENTS.md`: Expo 57 버전 문서 먼저 확인.
- `data/` — AI Hub 전처리. `preprocess_streaming.py`(대용량 음성 zip 스트리밍), `aihub_schema.py`(라벨 파싱), `build_mt_dataset.py`(변환 학습셋).
- `training/` — 학습·평가 스크립트(아래 재현 참고).
- `notebooks/` — Colab 노트북.

## 현재 상태 (핵심 지표)
- **STT**: whisper-small을 4지역(경상·전라·충청·강원) 방언으로 **LoRA 파인튜닝 + on-the-fly 소음 증강**. 어댑터 14MB. held-out 전체 val CER **표준 20.3% → 11.7%**. beam search 디코딩으로 추가 −0.72%p. 실제 폰+소음 조건 26.7%→20.8%.
- **변환(KoBART)**: 4지역, copy 대비 CER 5.79%→1.63%. 완벽STT 가정 시 0.5%로 거의 무손실.
- **End-to-End(음성→표준어)**: CER **9.49%**(사용자 체감 지표). 병목은 STT(변환기는 최대치).
- **서빙 모델**: `backend/models/whisper-dialect`(=최종 증강 LoRA), `backend/models/kobart-dialect`.

## 아키텍처/기술 결정
- 2단계를 교체 가능한 인터페이스로 분리(먼저 파이프라인 관통, 이후 학습 모델로 교체).
- STT 파인튜닝은 **full 대신 LoRA**(표준어 능력 보존 + 작은 어댑터 → 온디바이스·지역확장 유리).
- 대용량 학습은 **청크 나눠 어댑터 이어학습**(`--resume-adapter`) — 디스크 캐시 축소로 크래시 회피.
- 앱은 얇게(서버 추론). 온디바이스(ONNX/TFLite·whisper.cpp 양자화)는 최종 목표.

## 데이터·라이선스 (중요)
- 데이터: **AI 허브 「한국어 방언 발화」**(경상 119·전라 120·충청 122·강원 121).
- AI 허브 공식 확인: **파인튜닝 학습 결과물(모델)은 영리·비영리 배포 가능**. 단 **원본/추출 데이터(음성·전사·클립)는 재배포 금지** → 저장소·앱에 미포함(gitignore). **출처 "AI 허브" 표기 필요**(README에 있음, 앱에도 추가 예정).
- 저장소엔 **기술 내용만**(면접·포트폴리오성 서술 금지).

## 학습/평가 재현 (training/)
- 학습: `train_whisper.py --lora [--augment] [--resume-adapter <경로>]` (whisper-small, LoRA, 소음 증강, 이어학습).
- 평가: `eval_stt.py`(지역별 CER, --baseline/--examples), `eval_robust.py`(소음 강건성), `eval_e2e.py`(+`build_e2e_val.py`, 음성→표준어 E2E), `eval_beams.py`(greedy vs beam).
- 데이터 산출물(`data/processed*`, 모델)은 gitignore — 스크립트로 재생성.

## 다음 할 일 (우선순위)
1. 변환기(KoBART)를 STT 오류에 강하게(STT 출력→표준 재학습) — E2E 추가 개선.
2. 충청 데이터 보강(제일 약한 지역), 필요시 whisper-medium.
3. 온디바이스 전환(양자화) + 릴리스 준비(package id·eas.json·앱 출처표기·개인정보처리방침).
