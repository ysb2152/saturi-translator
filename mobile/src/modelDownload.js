// 첫 실행 모델 다운로드 — APK엔 코드만, 모델(~490MB)은 앱이 받아 documentDirectory에 저장.
// 견고성: 임시파일(.download)로 받고 → 정확 바이트 크기 검증 → 통과 시에만 최종 경로로 이동.
// 부분/손상 파일이 "완결"로 오인되지 않게 하고, 실패 시 임시파일을 정리한다.

import {
  documentDirectory,
  getInfoAsync,
  createDownloadResumable,
  moveAsync,
  deleteAsync,
} from 'expo-file-system/legacy';

// 모델 호스팅 베이스(깃헙 릴리스). 파일명은 앱이 로드하는 이름과 동일.
export const MODELS_BASE_URL =
  'https://github.com/ysb2152/saturi-translator/releases/download/models-v1';

// name = 저장/로드 파일명, bytes = 릴리스 에셋의 정확한 바이트 크기(무결성 검증용)
const FILES = [
  { name: 'ggml-model-q5_0.bin', bytes: 175209680 },
  { name: 'encoder.pte', bytes: 138325016 },
  { name: 'decoder.pte', bytes: 176043048 },
  { name: 'tokenizer.json', bytes: 1513021 },
];

async function sizeOf(uri) {
  try {
    const info = await getInfoAsync(uri);
    return info.exists ? (info.size ?? 0) : -1;
  } catch (_) {
    return -1;
  }
}

// 최종 파일이 정확한 크기로 존재하면 완결
async function isComplete(name, bytes) {
  return (await sizeOf(`${documentDirectory}${name}`)) === bytes;
}

async function rm(uri) {
  try { await deleteAsync(uri, { idempotent: true }); } catch (_) {}
}

/** 모델이 모두 완결돼 있으면 즉시 반환(false). 없으면 다운로드(true). onProgress(0..1). */
export async function ensureModels(onProgress) {
  const missing = [];
  for (const f of FILES) {
    if (!(await isComplete(f.name, f.bytes))) missing.push(f);
  }
  if (missing.length === 0) return false; // 이미 준비됨

  const total = missing.reduce((s, f) => s + f.bytes, 0);
  let base = 0;
  for (const f of missing) {
    const finalUri = `${documentDirectory}${f.name}`;
    const partUri = `${finalUri}.download`;
    await rm(partUri); // 이전 부분 파일 정리(이어받기 토큰은 보관하지 않음)

    const dl = createDownloadResumable(
      `${MODELS_BASE_URL}/${f.name}`,
      partUri,
      {},
      (p) => {
        const cur = base + p.totalBytesWritten;
        onProgress && onProgress(Math.min(0.999, cur / total));
      }
    );

    let res;
    try {
      res = await dl.downloadAsync();
    } catch (e) {
      await rm(partUri);
      throw new Error(`다운로드 중 오류(${f.name}): ${String((e && e.message) || e)}`);
    }
    if (!res || (res.status && res.status >= 400)) {
      await rm(partUri);
      throw new Error(`다운로드 실패(${f.name}, HTTP ${res && res.status})`);
    }

    // 무결성: 정확한 바이트 크기 확인(손상/절단 감지)
    const got = await sizeOf(partUri);
    if (got !== f.bytes) {
      await rm(partUri);
      throw new Error(`파일이 손상됐어요(${f.name}). 다시 시도해 주세요.`);
    }

    // 완결 시에만 최종 경로로 이동
    await rm(finalUri);
    await moveAsync({ from: partUri, to: finalUri });

    base += f.bytes;
    onProgress && onProgress(Math.min(0.999, base / total));
  }
  onProgress && onProgress(1);
  return true;
}
