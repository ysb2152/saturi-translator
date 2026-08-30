// 백엔드(FastAPI) 호출 모듈.
//
// 녹음된 로컬 파일(file:// URI)을 서버로 업로드해 사투리→표준어 변환 결과를 받는다.
// Expo SDK 57에서는 전역 fetch + { uri, name, type } FormData 파트가 지원되지 않으므로
// (에러: "unsupported FormDataPart implementation"), expo-file-system의 File.upload를 사용.
// expo-file-system은 Expo Go에 기본 내장되어 별도 빌드 없이 동작한다.
import { File, UploadType } from 'expo-file-system';

// 서버 주소 기본값:
// - Android 에뮬레이터: http://10.0.2.2:8000  (에뮬레이터가 PC의 localhost를 가리키는 특수 IP)
// - 실제 안드로이드 기기(Expo Go): PC의 LAN IP로 바꿔야 함. 예) http://192.168.0.10:8000
export const DEFAULT_SERVER_URL = 'http://10.0.2.2:8000';

export async function transcribe(serverUrl, uri) {
  const name = uri.split('/').pop() || 'audio.m4a';
  const ext = (name.split('.').pop() || 'm4a').toLowerCase();
  const url = `${serverUrl.replace(/\/+$/, '')}/transcribe`;

  const file = new File(uri);
  const res = await file.upload(url, {
    httpMethod: 'POST',
    uploadType: UploadType.MULTIPART,
    fieldName: 'audio',        // 백엔드 UploadFile 필드명과 일치
    mimeType: `audio/${ext}`,
    headers: { Accept: 'application/json' },
  });

  if (res.status < 200 || res.status >= 300) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = JSON.parse(res.body);
      if (j.detail) detail = j.detail;
    } catch (_) {}
    throw new Error(detail);
  }

  return JSON.parse(res.body); // { dialect_text, standard_text, language, duration }
}

// 녹음 없이 검증: 서버의 사투리 샘플을 파이프라인에 태워 결과만 받아온다.
export async function demoTranscribe(serverUrl) {
  const res = await fetch(`${serverUrl.replace(/\/+$/, '')}/demo`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j.detail) detail = j.detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}
