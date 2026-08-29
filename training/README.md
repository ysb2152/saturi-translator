# training — 로컬 GPU 학습

Colab 없이 로컬 GPU(RTX 3080 등)에서 파인튜닝한다. 사용량 제한·지오블록·업로드 없이
데이터가 있는 자리에서 바로 학습.

## 환경 (설치 완료)
- `training/.venv` — CUDA용 PyTorch(cu121) + transformers + datasets + jiwer + accelerate
- 확인: `training/.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"` → `True`

## KoBART 변환 모델 학습

```powershell
training\.venv\Scripts\python.exe training\train_kobart.py `
    --data data\processed\mt_balanced `
    --out backend\models\kobart-dialect `
    --epochs 2
```
- 끝나면 `backend/models/kobart-dialect`에 모델+토크나이저 저장 → 백엔드가 바로 로드
- 학습 후 자동으로 **copy vs KoBART CER** + 예시 출력
- 빠른 실험: `--max-train 100000 --epochs 1`
- 배치 조정: `--batch 64` (3080 10GB면 여유)

## 참고
- KoBART 토크나이저는 EOS를 자동으로 안 붙여서, 라벨 끝에 `eos_token_id`를 추가해야
  생성이 정상 종료된다(무한 반복 방지). 스크립트에 반영돼 있음.
- 생성은 반복 억제(`no_repeat_ngram_size`, `repetition_penalty`, `early_stopping`) 적용.

## Whisper STT 학습

표준 Whisper를 사투리 음성으로 파인튜닝해 인식률을 개선한다. 입력은
`data/preprocess_streaming.py`(또는 `preprocess.py`)가 만든 STT 매니페스트
(`data/processed/stt/{train,val}.jsonl`).

```powershell
training\.venv\Scripts\python.exe training\train_whisper.py `
    --data data\processed\stt --model openai/whisper-base --epochs 2
```
- 학습 후 val **CER** 출력(표준 Whisper 대비 비교용). 저장: `backend/models/whisper-dialect`
- 모델 크기: `openai/whisper-base`(빠름) / `openai/whisper-small`(정확, 느림)
- 빠른 실험: `--max-train 3000 --epochs 1`
- 오디오 로딩은 `soundfile`+`librosa`로 직접 처리(datasets Audio의 torchcodec 의존 회피).

> ⚠️ 데이터 주의: 음성 파일명 stem이 라벨 stem과 **일치해야** 매칭된다. AI Hub 경상도의
> add-on 음성(경상도_8)은 학습데이터_1 라벨과 겹치지 않으니, 라벨과 짝인 음성을 써야 함.
