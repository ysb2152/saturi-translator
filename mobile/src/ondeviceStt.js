// whisper.rn으로 사투리 음성을 기기에서 바로 인식한다(서버 없음).
// 0.7.4 API는 initWhisper({filePath}) → ctx.transcribe(path) → { promise } 형태.
// ggml 모델은 용량이 커서 앱에 넣지 않고, 첫 실행 때 받아둔 파일 경로로 로드한다.

import { initWhisper } from 'whisper.rn';

let _ctx = null;
let _loadingPath = null;

/** 모델을 한 번 로드해 재사용(컨텍스트 캐시). modelPath는 기기 절대경로. */
export async function loadWhisper(modelPath) {
  if (_ctx && _loadingPath === modelPath) return _ctx;
  if (_ctx) {
    try { await _ctx.release(); } catch (_) {}
    _ctx = null;
  }
  _ctx = await initWhisper({ filePath: modelPath });
  _loadingPath = modelPath;
  return _ctx;
}

/**
 * 기기에서 오디오 파일을 표준어가 아닌 '사투리 전사'로 인식.
 * @param {string} modelPath ggml 모델 절대경로
 * @param {string} audioPath 16kHz wav 등 오디오 절대경로
 * @returns {Promise<{text: string, ms: number}>}
 */
export async function transcribeOnDevice(modelPath, audioPath) {
  const ctx = await loadWhisper(modelPath);
  const t0 = Date.now();
  const { promise } = ctx.transcribe(audioPath, {
    language: 'ko',
    maxThreads: 4,
    // 서빙과 맞추려면 beamSize 지정(지연 보며 조정). 기본은 greedy.
  });
  const res = await promise;
  return { text: (res && res.result ? res.result : '').trim(), ms: Date.now() - t0 };
}

export async function releaseWhisper() {
  if (_ctx) {
    try { await _ctx.release(); } catch (_) {}
    _ctx = null;
    _loadingPath = null;
  }
}
