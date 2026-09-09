// 기기 안에서 음성 → 사투리(whisper.rn STT) → 표준어(executorch 변환)를 한 번에 돌린다.
// 서버를 안 거치고, 모델 파일은 기기 파일시스템 경로로 로드한다.

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
  // whisper.rn이 파일경로 WAV를 제대로 못 읽어서, base64 data URI로 넘긴다(내부의 다른 경로를 타게 됨).
  // fileUri가 file:/..., file:///..., raw 경로로 제각각 와서 접두사를 정규화한다(이중 접두 방지).
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

/**
 * 텍스트 전용: 사투리 텍스트 → 표준어. STT 없이 변환기+후처리만 사용(가장 정확한 경로).
 * @param {string} dialectText 사투리 텍스트
 */
export async function convertText(dialectText) {
  if (!ready) throw new Error('pipeline not loaded');
  const t0 = Date.now();
  const { text: converted } = await convertOnDevice(dialectText);
  const standard = applyDialectPostfix(converted);
  return { standard, ms: Date.now() - t0 };
}
