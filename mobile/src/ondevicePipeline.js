// 온디바이스 파이프라인: 음성 → (whisper.rn STT) 사투리 → (executorch 변환) 표준어.
// 서버 없이 기기에서 전체 수행. 모델 파일은 앱 파일시스템 경로로 로드.

import { initWhisper } from 'whisper.rn';
import { loadConverter, convertOnDevice } from './ondeviceConverter';

let whisperCtx = null;
let ready = false;

/**
 * 모델 로드(1회). paths는 기기 절대경로.
 * @param {{ sttModel, encoder, decoder, tokenizer }} paths
 */
export async function loadPipeline({ sttModel, encoder, decoder, tokenizer }) {
  if (ready) return;
  whisperCtx = await initWhisper({ filePath: sttModel });
  await loadConverter({ tokenizerSource: tokenizer, encoderSource: encoder, decoderSource: decoder });
  ready = true;
}

export function isPipelineReady() {
  return ready;
}

/**
 * 오디오 파일(16kHz mono PCM WAV) → { dialect, standard, ms }.
 * @param {string} audioPath 기기 오디오 경로
 */
export async function runPipeline(audioPath) {
  if (!ready) throw new Error('pipeline not loaded');
  const t0 = Date.now();
  const { promise } = whisperCtx.transcribe(audioPath, { language: 'ko', maxThreads: 4 });
  const stt = await promise;
  const dialect = (stt && stt.result ? stt.result : '').trim();
  const { text: standard } = await convertOnDevice(dialect);
  return { dialect, standard, ms: Date.now() - t0 };
}
