// 온디바이스 변환기(KoBART, ExecuTorch) — 사투리 텍스트 → 표준어.
// react-native-executorch: TokenizerModule(tokenizer.json) + ExecutorchModule(encoder/decoder .pte)
// encoder 1회 → decoder 자기회귀(단일스텝, KV캐시 없음) greedy 루프.

import { ExecutorchModule, TokenizerModule, initExecutorch } from 'react-native-executorch';
// NOTE: react-native-executorch-expo-resource-fetcher 0.9.1의 package.json "exports"가
// default/require 조건이 없어(그리고 exports 캡슐화로 내부 상대 import 차단) metro가 resolve 실패한다.
// 해결: patch-package로 그 패키지의 "exports" 필드를 제거(또는 default 조건 추가). 그 뒤 아래 import가 동작.
import { ExpoResourceFetcher } from 'react-native-executorch-expo-resource-fetcher';

let _initialized = false;

const INT = 3;   // ScalarType.INT (int32) — .pte를 int32 입력으로 export(BigInt64Array 회피)
const DEC_START = 1; // KoBART decoder_start_token_id
const EOS = 1;       // KoBART eos_token_id

let tok = null;
let enc = null;
let dec = null;

/** encoder.pte / decoder.pte / tokenizer.json 경로(기기 절대경로 또는 require) */
export async function loadConverter({ tokenizerSource, encoderSource, decoderSource }) {
  if (!_initialized) {
    initExecutorch({ resourceFetcher: ExpoResourceFetcher });
    _initialized = true;
  }
  tok = new TokenizerModule();
  await tok.load({ tokenizerSource });
  enc = new ExecutorchModule();
  await enc.load(encoderSource);
  dec = new ExecutorchModule();
  await dec.load(decoderSource);
}

// number[] → int32 텐서(TensorPtr)
function intTensor(arr) {
  return {
    dataPtr: Int32Array.from(arr),
    sizes: [1, arr.length],
    scalarType: INT,
  };
}

/** 사투리 텍스트 → 표준어(greedy). {maxNew} 최대 생성 토큰. */
export async function convertOnDevice(dialectText, { maxNew = 64 } = {}) {
  const t0 = Date.now();
  const ids = await tok.encode(dialectText);
  const mask = ids.map(() => 1);
  const [encHidden] = await enc.forward([intTensor(ids), intTensor(mask)]);
  const maskT = intTensor(mask);

  const out = [DEC_START];
  for (let step = 0; step < maxNew; step++) {
    const [logits] = await dec.forward([intTensor(out), encHidden, maskT]);
    const vocab = logits.sizes[logits.sizes.length - 1];
    const buf = logits.dataPtr; // Float32Array, [1, out.length, vocab]
    const base = (out.length - 1) * vocab;
    let best = 0;
    let bestVal = -Infinity;
    for (let v = 0; v < vocab; v++) {
      const x = buf[base + v];
      if (x > bestVal) { bestVal = x; best = v; }
    }
    if (best === EOS) break;
    out.push(best);
  }
  const text = (await tok.decode(out.slice(1), true)).trim();
  return { text, ms: Date.now() - t0 };
}

export async function releaseConverter() {
  try { await enc?.forward; enc = null; } catch (_) {}
  dec = null; tok = null;
}
