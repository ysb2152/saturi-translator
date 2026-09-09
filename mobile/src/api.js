// 백엔드(FastAPI)를 호출하는 모듈. 녹음한 로컬 파일(file:// URI)을 서버에 올려서 변환 결과를 받는다.
// 처음엔 전역 fetch + { uri, name, type } FormData로 올리려 했는데 Expo SDK 57에서
// "unsupported FormDataPart implementation" 에러가 나서, expo-file-system의 File.upload로 바꿨다.
// expo-file-system은 Expo Go에 이미 들어 있어서 따로 빌드할 필요가 없다.
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
