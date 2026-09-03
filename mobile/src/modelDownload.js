// 첫 실행 모델 다운로드 — APK엔 코드만, 모델(~490MB)은 앱이 받아 documentDirectory에 저장.
// 이후 실행부턴 이미 있으니 바로 로드. 진행률 콜백 + 재개(resumable) + 크기 검증.

import {
  documentDirectory,
  getInfoAsync,
  createDownloadResumable,
} from 'expo-file-system/legacy';

// 모델 호스팅 베이스(깃헙 릴리스). 파일명은 앱이 로드하는 이름과 동일.
export const MODELS_BASE_URL =
  'https://github.com/ysb2152/translate/releases/download/models-v1';

// name = 저장/로드 파일명, bytes = 대략 크기(존재·완결성 확인용)
const FILES = [
  { name: 'ggml-model-q5_0.bin', bytes: 175_000_000 },
  { name: 'encoder.pte', bytes: 138_000_000 },
  { name: 'decoder.pte', bytes: 176_000_000 },
  { name: 'tokenizer.json', bytes: 1_500_000 },
];

async function present(name, minBytes) {
  try {
    const info = await getInfoAsync(`${documentDirectory}${name}`);
    return info.exists && (info.size ?? 0) >= minBytes * 0.9;
  } catch (_) {
    return false;
  }
}

/** 모델이 모두 있으면 즉시 반환. 없으면 다운로드. onProgress(0..1) 호출. */
export async function ensureModels(onProgress) {
  const missing = [];
  for (const f of FILES) {
    if (!(await present(f.name, f.bytes))) missing.push(f);
  }
  if (missing.length === 0) return false; // 이미 준비됨(다운로드 안 함)

  const total = missing.reduce((s, f) => s + f.bytes, 0);
  let base = 0;
  for (const f of missing) {
    const dl = createDownloadResumable(
      `${MODELS_BASE_URL}/${f.name}`,
      `${documentDirectory}${f.name}`,
      {},
      (p) => {
        const cur = base + p.totalBytesWritten;
        onProgress && onProgress(Math.min(0.999, cur / total));
      }
    );
    const res = await dl.downloadAsync();
    if (!res || (res.status && res.status >= 400)) {
      throw new Error(`다운로드 실패(${f.name}, HTTP ${res && res.status})`);
    }
    base += f.bytes;
    onProgress && onProgress(Math.min(0.999, base / total));
  }
  onProgress && onProgress(1);
  return true; // 다운로드함
}
