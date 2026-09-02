# 온디바이스 통합 계획 (STT + 변환기)

배포 방식으로 **온디바이스**를 택했다(근거·트레이드오프: DEVELOPMENT_JOURNEY B-15). 서버 없이 앱 안에서 음성→표준어 전체를 돌린다. 양자화 실현가능성(정확도 유지·총 ~370MB)은 이미 검증됐다(EVALUATION 온디바이스 양자화 섹션).

> ⚠️ 착수 첫걸음: 이 프로젝트는 Expo SDK 57 / RN 0.86 / React 19다. 네이티브 모듈 버전 호환을 [Expo 버전별 문서](https://docs.expo.dev/versions/v57.0.0/)와 각 패키지 릴리스로 **먼저 확인**한 뒤 설치한다(mobile/AGENTS.md 지침).

## 1. 구조

앱 안에 네이티브 추론 엔진 두 개를 넣는다. 둘 다 Expo Go로는 안 되고 **prebuild + 커스텀 dev client + EAS Build**가 필요하다.

| 컴포넌트 | 엔진 | 모델 포맷 | 상태 |
|---|---|---|---|
| STT (Whisper-small, 4지역 LoRA 병합) | [`whisper.rn`](https://github.com/mybigday/whisper.rn) (whisper.cpp 바인딩, New Arch 지원, v0.5.x) | GGUF(ggml) q5 | 경로 명확 |
| 변환기 (KoBART) | [`onnxruntime-react-native`](https://onnxruntime.ai/docs/get-started/with-javascript/react-native.html) (v1.24.x) | ONNX int8 | 리스크 있음(아래) |

## 2. STT — whisper.rn + GGUF

> ✅ **ggml 변환은 이미 검증 완료**(2026-09-02, 앱 없이 dev PC). 파인튜닝 병합 모델(`backend/models/whisper-dialect`)을 whisper.cpp `convert-h5-to-ggml.py`로 변환 → **`ggml-model-f16.bin` 487MB**, ggml 매직(`0x6c6d6767`) 유효 확인. 우리 모델이 whisper.cpp 포맷으로 깨끗이 변환됨. 재현: 모델의 `vocab.json`·`added_tokens.json`·`config.json` + openai/whisper의 `whisper/assets/mel_filters.npz`만 있으면 됨 → `python convert-h5-to-ggml.py <model_dir> <whisper_assets_dir> <out_dir>`. **q5 양자화는 미완**(whisper.cpp `quantize` 바이너리 = C 빌드 필요, 이 PC에 cmake/컴파일러 없음). q5 목표 ~190MB는 EVALUATION에 기산정됨.

1. 파인튜닝 병합 모델(`backend/models/whisper-dialect`)을 **GGUF로 변환** ✅ (f16 완료) — 후속 `quantize`로 q5_0(≈190MB) 또는 q8_0.
2. 앱에서 `initWhisper({ filePath })`로 로드, 녹음 파일 경로를 `transcribe`에 전달. 언어=한국어 고정.
3. 디코딩은 서빙과 맞추되(현재 beam=5) 온디바이스 지연을 보며 beam 축소(3 등) 절충 — EVALUATION에 이미 기록한 지연 트레이드오프.
4. 토크나이저는 whisper.rn(whisper.cpp)이 내부 처리하므로 별도 작업 없음.

## 3. 변환기 — onnxruntime-react-native + ONNX (핵심 리스크)

KoBART는 encoder-decoder(BART) seq2seq라 STT보다 손이 많이 간다.

> ✅ **모델 export·양자화·정합성은 이미 검증 완료**(2026-09-02, 앱 없이 dev PC에서). `optimum-cli export onnx --task text2text-generation-with-past`로 encoder/decoder/with-past 분해 export → onnxruntime 동적 int8 양자화. **변환기 int8 합계 268MB**(encoder 67 + decoder 104 + with_past 97), **PyTorch 대비 생성 정합성 5/5 완전 일치**(서빙 디코딩 파라미터 동일). 재현: [`backend/onnx_quant_parity.py`](../backend/onnx_quant_parity.py). 주의: merged 디코더는 If-노드 서브그래프라 동적 양자화가 안 되므로 **비-merged(decoder + decoder_with_past)를 양자화해 사용**한다. 이로써 변환기 리스크는 "모델 export"가 아니라 **앱 쪽 JS 토크나이저 + 생성 루프**로 좁혀졌다(tokenizer.json이 export에 포함돼 재사용 가능).

1. **ONNX export**: `optimum-cli export onnx` 또는 optimum으로 KoBART를 encoder / decoder(+with-past)로 분해 export, int8 양자화(EVALUATION에서 int8 무손실 확인). ✅ 완료
2. **토크나이저 온디바이스**: KoBART 토크나이저(SentencePiece/BPE 계열)를 JS/네이티브에서 재현해야 한다 — 가장 불확실한 부분. 후보: SentencePiece 모델을 번들해 JS 포트로 인코딩/디코딩, 또는 어휘·머지 규칙을 JS로 이식.
3. **생성 루프**: onnxruntime-react-native로 encoder 1회 → decoder 자기회귀(greedy 또는 beam) 루프를 **JS에서 구현**(현 서빙: `max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.3`). EOS 처리 주의(과거 KoBART EOS 미부착 버그 이력, B-9).

## 4. 모델 전달

- 합계 ~370MB는 APK/AAB에 담기 부적절(Play 기본 용량 제한). **첫 실행 시 다운로드** 방식(설치 ~50MB) — EVALUATION에 이미 명시한 계획.
- 호스팅: GitHub Releases 또는 CDN. 다운로드 무결성 체크(체크섬) + 실패 재시도 + 진행률 UI.

## 5. 빌드 설정 (EAS)

1. `expo-dev-client` 추가, `npx expo prebuild`로 네이티브 프로젝트 생성.
2. `whisper.rn`·`onnxruntime-react-native` 설치 + (있으면) config plugin 등록, 없으면 prebuild 후 네이티브 설정.
3. `eas.json`의 `development` 프로파일(이미 있음, `developmentClient: true`)로 dev client 빌드 → 실기 테스트. 이후 `preview`(APK) → `production`(AAB).

## 6. 리스크와 폴백

- **최대 리스크는 변환기(KoBART) 온디바이스** — 토크나이저 이식 + JS 생성 루프. 여기서 막히면 폴백:
  - (A) **하이브리드**: STT는 온디바이스, 변환기만 경량 서버 — 음성은 기기를 안 벗어나고(프라이버시 이득 유지) 텍스트만 서버로. 서버 부담이 작아 저가 호스팅 가능.
  - (B) **STT 온디바이스 우선 출시** 후 변환기 온디바이스는 후속 업데이트.
  - (C) 변환기를 더 가벼운 규칙+소형 모델로 대체(정확도 손해 감수).
- RN 0.86 + React 19 + New Architecture는 매우 최신이라, 두 네이티브 모듈의 실제 빌드 통과를 **가장 먼저** 확인(작은 스파이크).

### 6-1. 빌드(dev client) 실패 시 폴백 사다리

첫 스파이크(§7-1)에서 네이티브 모듈이 RN 0.86/React 19/New Arch에서 빌드가 안 되더라도 길은 많다. 이건 "온디바이스가 가능한가"가 아니라 "어떤 버전 조합이냐"의 문제다 — 안드로이드 온디바이스 Whisper·seq2seq는 검증된 기술이고, 걸리는 건 최신 버전을 모듈이 아직 못 따라오는 것뿐. 선호 순으로:

1. **Expo SDK/RN 버전 정렬 (가장 확실)** — 네이티브 모듈이 검증된 SDK(예: 53/54)로 앱 스캐폴드를 낮춘다. 이 앱은 화면 하나(App.js+api.js)라 다운그레이드 비용이 거의 없다. 최신을 고집할 이유가 없으므로 1순위.
2. **New Architecture 끄기** — 모듈이 구 아키텍처만 지원하면 `app.json`의 `newArchEnabled: false`로 레거시 모드 시도(무료 시도).
3. **런타임 통합** — 한쪽 모듈만 빌드되면 두 모델을 그 런타임 하나로 몰아 호환 표면을 최소화(예: whisper.rn이 안 되면 Whisper도 ONNX로 → onnxruntime 단일 의존).
4. **대체 런타임** — `react-native-executorch`(PyTorch 온디바이스, RN 공식 지원) 또는 TFLite로 동일 모델 실행.
5. **로컬 prebuild + patch-package** — gradle/podspec 한두 줄이 걸리는 흔한 경우를 패치로 통과.
6. **최후 안전망 — 서버 방식** — 정 안 되면 현재 서버 구조 그대로 출시(항상 배포 가능한 앱이 손에 있으므로 "아무것도 못 냄" 시나리오는 없음). 온디바이스는 후속 업데이트로.

첫 스파이크를 맨 앞에 두는 이유가 이 판단을 30분 안에 내리기 위함이다 — 통과하면 그대로, 실패하면 즉시 1번으로 튼다.

## 7. 실행 순서 (다음 세션)

1. 버전 호환 확인(SDK 57 문서 + 두 패키지) → `expo-dev-client` + prebuild 스파이크로 **빈 dev client 빌드 통과** 확인.
2. whisper.rn 단독 통합 — GGUF 변환한 STT 모델로 온디바이스 인식 동작 확인(가장 확실한 파트 먼저).
3. onnxruntime-react-native 스파이크 — KoBART ONNX + 토크나이저/생성 루프. 막히면 §6 폴백 결정.
4. 첫 실행 모델 다운로드 + 무결성/진행률.
5. preview(APK) 실기 → production(AAB) → 스토어 등록(개발자 계정·개인정보처리방침 URL 호스팅).

각 단계는 독립적으로 커밋·검증하고, 결과·판단을 개발일지에 남긴다.
