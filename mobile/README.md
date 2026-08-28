# mobile — 사투리 번역 앱 (React Native / Expo)

녹음 → 백엔드(`/transcribe`) 전송 → 사투리 인식/표준어 변환 결과 표시.

- `App.js` — 메인 화면(녹음 버튼, 서버 주소, 결과 카드)
- `src/api.js` — 백엔드 호출 + 서버 주소 기본값

## 로컬 개발환경(설치 완료 상태)

- JDK 17, Android SDK(cmdline-tools), 에뮬레이터, 시스템이미지(android-34)까지
  설치됨. `ANDROID_HOME` 등 환경변수도 사용자 범위로 등록됨(새 터미널부터 적용).
- 하드웨어 가속: Windows Hypervisor Platform(WHPX) 이미 활성 — 추가 설정 불필요.
- 생성된 가상기기(AVD) 이름: **`saturi_pixel6`** (Pixel 6 / API 34)

에뮬레이터 켜기(새 터미널이면 환경변수 자동 적용):
```powershell
emulator -avd saturi_pixel6
```

## 실행

먼저 백엔드를 띄운다(다른 터미널):
```powershell
cd ..\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0
```
> 실제 기기에서 접속하려면 `--host 0.0.0.0`으로 떠야 PC 밖에서 보인다.

앱 실행:
```powershell
cd mobile
npm run android      # Android 에뮬레이터 or 연결된 기기
# 또는
npm start            # QR코드 → 폰의 Expo Go 앱으로 스캔
```

## 서버 주소 맞추기 (중요)

앱 화면 상단 **'서버 주소'** 칸에서 백엔드 위치를 지정한다.

| 실행 방식 | 서버 주소 |
|-----------|-----------|
| Android 에뮬레이터 | `http://10.0.2.2:8000` (기본값) |
| 실제 폰 + Expo Go | `http://<PC의 LAN IP>:8000` — PC에서 `ipconfig`로 IPv4 확인 (예: `http://192.168.0.10:8000`) |

폰과 PC가 **같은 Wi-Fi**에 있어야 하고, Windows 방화벽에서 8000 포트 인바운드가 막히면 허용해야 한다.

## 참고

- 녹음은 `expo-audio`(SDK 57)의 `HIGH_QUALITY` 프리셋 → Android에서 `.m4a`.
- 마이크 권한 문구는 `app.json`의 `expo-audio` 플러그인 설정에 있음.
- Expo Go로 대부분 테스트 가능. 플레이스토어 출시용 APK/AAB는 나중에 EAS Build로.
