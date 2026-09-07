// 온디바이스 파이프라인: 음성 → (whisper.rn STT) 사투리 → (executorch 변환) 표준어.
// 서버 없이 기기에서 전체 수행. 모델 파일은 앱 파일시스템 경로로 로드.

import { initWhisper } from 'whisper.rn';
import { readAsStringAsync } from 'expo-file-system/legacy';
import { loadConverter, convertOnDevice } from './ondeviceConverter';
import { applyDialectPostfix } from './dialectPostfix';

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
  // whisper.rn의 파일경로 WAV 로딩이 실패해, base64 data URI 경로로 전달(다른 처리 경로)
  // fileUri가 file:/... 또는 file:///... 또는 raw 경로로 올 수 있어 정규화(이중 접두 방지)
  const raw = audioPath.replace(/^file:\/+/, '/');       // → /data/.../xxx.wav
  const b64 = await readAsStringAsync(`file://${raw}`, { encoding: 'base64' });
  const { promise } = whisperCtx.transcribe(`data:audio/wav;base64,${b64}`, {
    language: 'ko',
    maxThreads: 4,
    beamSize: 3,     // 빔서치: 짧고 애매한 발화의 오인식·환각을 greedy보다 안정적으로 줄임
    temperature: 0,  // 결정적 디코딩 — 무음 패딩에서 오는 환각 억제
    prompt: '다음은 한국어 사투리 대화입니다.', // 도메인 프라이밍(한국어 사투리 쪽으로 유도)
  });
  const stt = await promise;
  const dialect = (stt && stt.result ? stt.result : '').trim();
  const { text: converted } = await convertOnDevice(dialect);
  // 하이브리드 후처리: 변환기가 놓친 잔여 사투리 표현을 규칙으로 보정(학습 데이터 커버리지 공백 대응)
  const standard = applyDialectPostfix(converted);
  return { dialect, standard, ms: Date.now() - t0 };
}
