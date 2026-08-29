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

## 다음: Whisper STT 학습(예정)
음성 클립(`data/preprocess_streaming.py` 산출물)이 준비되면 같은 환경에서 Whisper 파인튜닝
스크립트를 추가한다. GPU가 로컬에 있으니 Colab 불필요.
