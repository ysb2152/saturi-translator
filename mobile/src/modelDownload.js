// 첫 실행 모델 다운로드 — APK엔 코드만, 모델(~490MB)은 앱이 받아 documentDirectory에 저장.
//
// 견고성:
//  - 임시파일(.download)로 받고 → 정확 바이트 크기 검증 → 통과 시에만 최종 경로로 이동.
//  - 이어받기: 진행 중 savable() 상태를 .resume(JSON)에 저장하고, 다음 실행에서 그게 있으면
//    resumeAsync()로 이어받는다(앱이 중간에 죽어도 처음부터 다시 받지 않음). 이어받기 실패 시 새로 받기로 폴백.
//  - 버전 매니페스트: models.version 마커로 현재 모델 세트를 식별. MODELS_VERSION을 올리면(models-v2 등)
//    이전 모델을 정리하고 재다운로드한다. 마커가 없고 파일이 이미 완결이면 기존 사용자로 보고 재다운로드하지 않는다.

import {
  documentDirectory,
  getInfoAsync,
  createDownloadResumable,
  moveAsync,
  deleteAsync,
  readAsStringAsync,
  writeAsStringAsync,
} from 'expo-file-system/legacy';

// 모델 세트 버전 마커. 변환기(.pte)를 새로 올릴 때마다 올린다(→ 앱이 자동 갱신).
// 주의: 재학습해도 int8 .pte는 '바이트 크기가 동일'하고 내용만 다르다. 따라서 크기 검증만으론
//       업데이트를 감지할 수 없어, 이 마커가 바뀌면 volatile(변환기) 파일을 강제 재다운로드한다.
export const MODELS_VERSION = 'v2';

const RELEASE_BASE = 'https://github.com/ysb2152/saturi-translator/releases/download';
const VERSION_FILE = 'models.version';

// name=저장/로드 파일명, bytes=정확한 바이트(무결성), tag=호스팅 릴리스, volatile=모델 변경 시 갱신 대상.
// STT·토크나이저는 안 바뀌므로 models-v1 그대로 재사용(재업로드 불필요). 변환기 .pte만 models-v2.
const FILES = [
  { name: 'ggml-model-q5_0.bin', bytes: 175209680, tag: 'models-v1' },
  { name: 'tokenizer.json', bytes: 1513021, tag: 'models-v1' },
  { name: 'encoder.pte', bytes: 138325016, tag: 'models-v2', volatile: true },
  { name: 'decoder.pte', bytes: 176043048, tag: 'models-v2', volatile: true },
];

function fileUrl(f) {
  return `${RELEASE_BASE}/${f.tag}/${f.name}`;
}

// 하위호환: 이전 코드/문서가 참조하던 이름
export const MODELS_BASE_URL = `${RELEASE_BASE}/models-v2`;

const RESUME_SAVE_EVERY = 8 * 1024 * 1024; // 이어받기 상태 저장 간격(~8MB)

async function sizeOf(uri) {
  try {
    const info = await getInfoAsync(uri);
    return info.exists ? (info.size ?? 0) : -1;
  } catch (_) {
    return -1;
  }
}

async function isComplete(name, bytes) {
  return (await sizeOf(`${documentDirectory}${name}`)) === bytes;
}

async function rm(uri) {
  try { await deleteAsync(uri, { idempotent: true }); } catch (_) {}
}

async function readText(uri) {
  try {
    if ((await sizeOf(uri)) < 0) return null;
    return await readAsStringAsync(uri);
  } catch (_) {
    return null;
  }
}

async function writeText(uri, text) {
  try { await writeAsStringAsync(uri, text); } catch (_) {}
}

// 버전 변경 시 변환기(volatile) 파일만 강제 삭제 → 재다운로드 유도.
// int8 .pte는 재학습해도 바이트 크기가 같아 크기 검증만으론 갱신을 못 잡으므로 강제 삭제가 필요.
// STT·토크나이저(stable)는 건드리지 않아 업데이트 시 .pte(~314MB)만 다시 받는다.
async function forceRedownloadVolatile() {
  for (const f of FILES) {
    if (!f.volatile) continue;
    await rm(`${documentDirectory}${f.name}`);
    await rm(`${documentDirectory}${f.name}.download`);
    await rm(`${documentDirectory}${f.name}.resume`);
  }
}

// 파일 하나 다운로드(이어받기 지원). 성공 시 최종 경로로 이동.
async function downloadOne(f, partUri, finalUri, resumeUri, onTick) {
  // 이어받기 상태가 있으면 재구성해 resumeAsync, 없으면 새로 downloadAsync
  const saved = await readText(resumeUri);
  let savedState = null;
  if (saved) {
    try { savedState = JSON.parse(saved); } catch (_) { savedState = null; }
  }

  let lastSaved = 0;
  const cb = (p) => {
    onTick(p.totalBytesWritten);
    // 진행 상태를 주기적으로 저장 → 다음 실행에서 이어받기
    if (p.totalBytesWritten - lastSaved >= RESUME_SAVE_EVERY) {
      lastSaved = p.totalBytesWritten;
      try { writeText(resumeUri, JSON.stringify(dl.savable())); } catch (_) {}
    }
  };

  let dl;
  let res;
  if (savedState && savedState.resumeData && (await sizeOf(partUri)) > 0) {
    dl = createDownloadResumable(
      savedState.url || fileUrl(f),
      partUri,
      savedState.options || {},
      cb,
      savedState.resumeData
    );
    try {
      res = await dl.resumeAsync();
    } catch (_) {
      // 이어받기 실패 → 처음부터 새로 받기
      await rm(partUri);
      await rm(resumeUri);
      dl = createDownloadResumable(fileUrl(f), partUri, {}, cb);
      res = await dl.downloadAsync();
    }
  } else {
    await rm(partUri); // 이어받을 근거 없으면 부분파일 정리 후 새로
    dl = createDownloadResumable(fileUrl(f), partUri, {}, cb);
    res = await dl.downloadAsync();
  }

  if (!res || (res.status && res.status >= 400)) {
    // 실패: 이어받기 상태 저장(있으면) 후 부분파일은 남겨 다음 재시도에 활용
    try { await writeText(resumeUri, JSON.stringify(dl.savable())); } catch (_) {}
    throw new Error(`다운로드 실패(${f.name}, HTTP ${res && res.status})`);
  }

  // 무결성: 정확한 바이트 크기 확인(손상/절단 감지)
  const got = await sizeOf(partUri);
  if (got !== f.bytes) {
    await rm(partUri);
    await rm(resumeUri);
    throw new Error(`파일이 손상됐어요(${f.name}). 다시 시도해 주세요.`);
  }

  // 완결 시에만 최종 경로로 이동 + 이어받기 상태 정리
  await rm(finalUri);
  await moveAsync({ from: partUri, to: finalUri });
  await rm(resumeUri);
}

/** 다운로드가 필요한지 미리 확인(네트워크 사용 없음). 셀룰러 경고 게이트 판단용. */
export async function needsDownload() {
  const stored = await readText(`${documentDirectory}${VERSION_FILE}`);
  if (stored !== MODELS_VERSION) return true; // 버전 불일치(마커 없음=구버전 사용자 포함) → 재다운로드 필요
  for (const f of FILES) {
    if (!(await isComplete(f.name, f.bytes))) return true;
  }
  return false;
}

/** 모델이 모두 완결돼 있으면 즉시 반환(false). 없으면 다운로드(true). onProgress(0..1). */
export async function ensureModels(onProgress) {
  const versionUri = `${documentDirectory}${VERSION_FILE}`;
  const storedVersion = await readText(versionUri);

  // 마커가 현재 버전과 다르면(마커 없음=구버전 사용자 포함) 변환기(volatile)만 강제 재다운로드.
  // STT·토크나이저는 크기 검증으로 완결이면 유지 → 업데이트는 .pte(~314MB)만 받는다.
  if (storedVersion !== MODELS_VERSION) {
    await forceRedownloadVolatile();
  }

  const missing = [];
  for (const f of FILES) {
    if (!(await isComplete(f.name, f.bytes))) missing.push(f);
  }
  if (missing.length === 0) {
    await writeText(versionUri, MODELS_VERSION); // 마커만 없던 경우 기록
    return false;
  }

  const total = missing.reduce((s, f) => s + f.bytes, 0);
  let base = 0;
  for (const f of missing) {
    const finalUri = `${documentDirectory}${f.name}`;
    const partUri = `${finalUri}.download`;
    const resumeUri = `${finalUri}.resume`;

    try {
      await downloadOne(f, partUri, finalUri, resumeUri, (written) => {
        const cur = base + written;
        onProgress && onProgress(Math.min(0.999, cur / total));
      });
    } catch (e) {
      throw new Error(`다운로드 중 오류(${f.name}): ${String((e && e.message) || e)}`);
    }

    base += f.bytes;
    onProgress && onProgress(Math.min(0.999, base / total));
  }

  await writeText(versionUri, MODELS_VERSION);
  onProgress && onProgress(1);
  return true;
}
