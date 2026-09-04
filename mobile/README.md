# mobile — 알아 묵나? 앱 (React Native / Expo, 온디바이스)

녹음 → **기기 안에서** STT(사투리 인식) → 변환(표준어) → 결과 표시. 서버 없이 완전 오프라인으로 동작한다(첫 실행 시 모델만 1회 다운로드).

## 핵심 파일

| 파일 | 역할 |
|---|---|
| `App.js` | 메인 화면(녹음/취소, 결과 카드, 다운로드 진행률·재시도) |
| `src/ondevicePipeline.js` | `loadPipeline()` + `runPipeline(audio) → {dialect, standard}` |
| `src/ondeviceConverter.js` | ExecuTorch 변환기(토크나이저 + encoder + decoder greedy 루프) |
| `src/modelDownload.js` | 첫 실행 모델 다운로드(무결성 검증·재시도) |
| `src/etfetcher/` | executorch 리소스 페처 vendoring(metro resolve 우회) |
| `plugins/withCleanPermissions.js` | 미사용 민감 권한 제거(config plugin) |
| `scripts/make_icon.py`, `make_store_assets.py` | 아이콘·스토어 자산 생성(Pillow) |

## 실행 (개발)

네이티브 모듈(whisper.rn, react-native-executorch)을 쓰므로 **Expo Go로는 안 되고** prebuild + 개발 빌드가 필요하다.

```powershell
cd mobile
npm install
npx expo run:android      # 실기기 또는 에뮬레이터. 첫 실행 시 모델(~490MB) 다운로드
```

> 실사용 성능은 실기기(arm)에서 확인한다(에뮬 x86은 느림). 녹음은 `@siteed/expo-audio-studio`로 16kHz mono PCM WAV.

## 릴리스 빌드 · 서명

업로드 keystore는 `mobile/credentials/`(gitignore). 서명된 AAB:

```powershell
cd mobile/android
./gradlew.bat bundleRelease -PMYAPP_UPLOAD_STORE_FILE=<절대경로> -PMYAPP_UPLOAD_STORE_PASSWORD=<pw> -PMYAPP_UPLOAD_KEY_ALIAS=upload -PMYAPP_UPLOAD_KEY_PASSWORD=<pw>
```

플레이스토어 준비 전체는 [docs/release-checklist.md](../docs/release-checklist.md), 온디바이스 통합 회고는 [docs/ondevice-plan.md](../docs/ondevice-plan.md).

## 환경 메모

- JDK 17 / Android SDK / 에뮬레이터(AVD `saturi_pixel6`, API 34) 설치됨. `ANDROID_HOME` 사용자 범위 등록.
- 마이크 권한 문구는 `app.json`의 `expo-audio` 플러그인 설정.
- 앱 이름은 "알아 묵나?"(런처/스토어), 앱 안 큰 제목은 "사투리 번역"(기능 명확성 위해 유지).
