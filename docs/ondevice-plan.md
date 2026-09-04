# 온디바이스 통합 — 구현 회고

서버 없이 앱 안에서 음성→표준어 전체를 돌리는 것을 목표로 했고, **완료**했다. 이 문서는 최종 아키텍처와, 그 과정에서 가장 어려웠던 문제들을 어떻게 풀었는지 남긴다. 시간순 상세는 [DEVELOPMENT_JOURNEY.md](../DEVELOPMENT_JOURNEY.md), 정량 지표는 [EVALUATION.md](../EVALUATION.md).

## 최종 아키텍처

| 단계 | 엔진 | 모델 | 크기 |
|---|---|---|---|
| STT | [`whisper.rn`](https://github.com/mybigday/whisper.rn) (whisper.cpp) | Whisper-small 4지역 LoRA → GGUF **q5_0** | 175MB |
| 변환 | [`react-native-executorch`](https://github.com/software-mansion/react-native-executorch) (ExecuTorch) | KoBART → **.pte int8** (encoder + decoder) | 314MB |
| 녹음 | `@siteed/expo-audio-studio` | 16kHz mono PCM WAV | — |

모델은 APK에 넣지 않고 **첫 실행 시 GitHub Releases에서 다운로드**(~490MB) → 이후 오프라인. 파이프라인은 [`mobile/src/ondevicePipeline.js`](../mobile/src/ondevicePipeline.js).

## 핵심 난관과 해결

1. **변환기 런타임 — onnxruntime 실패 → ExecuTorch 전환.** 첫 계획은 onnxruntime-react-native였으나 이 스택(Expo SDK57/RN0.86/gradle9)에서 `SoftwareComponent 'release' not found`로 빌드 불가(라이브러리 모듈 publishing 컨벤션 문제, patch로 해결 안 됨). `react-native-executorch`로 전환해 빌드 통과. **모델 export는 문제없었고 막힌 건 RN 네이티브 런타임**이라는 점을 분리해 판단한 게 핵심.

2. **KoBART를 .pte로 — 커스텀 seq2seq 온디바이스.** encoder + decoder(단일스텝)를 `torch.export → to_edge → to_executorch`로 변환. 토크나이저(`TokenizerModule`) + encoder 1회 + decoder greedy 루프를 **앱에서 직접 구현**([`mobile/src/ondeviceConverter.js`](../mobile/src/ondeviceConverter.js)).

3. **빈 출력 버그.** 변환 결과가 계속 비어 나옴. Python executorch 런타임으로 .pte greedy를 재현해 **모델·로직은 PyTorch와 완전 일치**함을 확인([`backend/debug_pte_gen.py`](../backend/debug_pte_gen.py)) → 버그를 JS 레이어로 국한 → decoder 출력 `logits.dataPtr`가 `Float32Array`가 아니라 `ArrayBuffer`여서 argmax가 0 고정되던 것. `new Float32Array()`로 감싸 해결(겸사 int32 입력으로 재-export해 BigInt64Array 회피).

4. **int8 양자화 — 임베딩 함정.** PT2E(XNNPACKQuantizer)는 정수 임베딩 입력까지 양자화하려다 실패. **torchao weight-only int8**(`Int8WeightOnlyConfig`)로 우회 — Linear 가중치만 int8, 입력·activation 불변이라 임베딩 이슈를 구조적으로 회피. 587→314MB, 정확도 완전 유지.

5. **오디오 포맷.** whisper.rn은 16kHz mono PCM WAV만 받는데 expo-audio의 Android 출력은 AAC/m4a뿐 → `@siteed/expo-audio-studio`로 원시 PCM 캡처. 파일경로 로딩이 실패해 base64 data URI로 우회.

6. **STT q5 — C 컴파일러 없이.** whisper.cpp 공식 프리빌트 `whisper-quantize`(ggml-org 릴리스)로 f16 487→q5_0 175MB. 동일 val 120파일 비교로 f16 대비 CER 저하 없음 검증.

## 결과

실기기(Galaxy S24)에서 신규 설치 → 다운로드 → **완전 오프라인** 녹음·인식·번역 동작, 처리 7~8초. 총 1.2GB→490MB(양자화), CER 저하 없음.
