# backend — 모델 export · 양자화 · 평가 (개발용)

앱은 **온디바이스**로 동작하므로 이 디렉터리는 **런타임 서버가 아니라 개발 도구**다. 파인튜닝된 모델을 앱에 넣을 포맷으로 변환·양자화하고, 정확도를 검증한다. (초기 개발 땐 FastAPI 서버로 STT+변환을 서빙하며 앱과 연동했고, 그 코드도 `app/`에 남아 있다.)

## 주요 스크립트

| 파일 | 역할 |
|---|---|
| `export_kobart_pte.py` | KoBART 변환기 → ExecuTorch `.pte`(encoder/decoder). `QUANT=1`이면 torchao int8 weight-only |
| `debug_pte_gen.py` | `.pte`를 executorch 런타임으로 greedy 실행해 PyTorch와 생성 정합성 비교(온디바이스 빈출력 버그 진단에 사용) |
| `onnx_quant_parity.py` | (초기 경로) KoBART ONNX int8 export + PyTorch 대비 정합성 검증 |
| `app/` | FastAPI 서버(개발 중 STT+변환 서빙에 사용) |

STT GGUF 변환·q5 양자화는 whisper.cpp 도구(`convert-h5-to-ggml.py`, 프리빌트 `whisper-quantize`)를 사용했다. 자세한 과정은 [docs/ondevice-plan.md](../docs/ondevice-plan.md).

## 환경

`.pte` export는 별도 venv(executorch + torch)에서 수행한다(학습 venv 미오염). 모델 산출물(`backend/models/`)은 AI Hub 이용정책에 따라 `.gitignore`로 제외된다.

```powershell
# 예: 변환기 int8 .pte export
C:/et/Scripts/python.exe backend/export_kobart_pte.py   # QUANT=1 환경변수로 int8
```

정량 평가 결과는 [EVALUATION.md](../EVALUATION.md), STT/변환 평가 스크립트는 [training/](../training/).
