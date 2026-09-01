"""양자화(int8) 정확도 영향 빠른 검증 — fp32 vs torch 동적 int8 CER 비교.

whisper.cpp GGUF(q5/q8)가 실제 온디바이스 경로지만 C 빌드 필요 → 여기선 torch
dynamic int8 로 근사(보통 GGUF q5가 더 좋음, 즉 보수적 신호). 크기도 함께 추정.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import torch, jiwer, numpy as np, soundfile as sf, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000
REGIONS = [("data/processed/stt/val.jsonl", "경상"), ("data/processed_jl/stt/val.jsonl", "전라"),
           ("data/processed_cc/stt/val.jsonl", "충청"), ("data/processed_gw/stt/val.jsonl", "강원")]


def load_audio(p):
    arr, sr = sf.read(p)
    if getattr(arr, "ndim", 1) > 1: arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if sr != SR: arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return arr


def cer_of(model, proc, items, dev, batch=8):
    refs, preds = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        feats = proc.feature_extractor([load_audio(x["audio_filepath"]) for x in chunk],
                                       sampling_rate=SR, return_tensors="pt").input_features.to(dev)
        with torch.no_grad():
            g = model.generate(feats, max_new_tokens=128, no_repeat_ngram_size=3)
        preds += proc.tokenizer.batch_decode(g, skip_special_tokens=True)
        refs += [x["text"] for x in chunk]
    return jiwer.cer(refs, preds)


def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="backend/models/whisper-dialect")
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(args.model, language="korean", task="transcribe")

    def prep(m):
        m.generation_config.language = "korean"; m.generation_config.task = "transcribe"
        m.generation_config.forced_decoder_ids = None
        return m.eval()

    # fp32 (GPU 있으면 GPU)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m32 = prep(WhisperForConditionalGeneration.from_pretrained(args.model)).to(dev)
    sz32 = param_bytes(m32) / 1e6

    # int8 동적 양자화 (CPU 전용)
    m8 = prep(WhisperForConditionalGeneration.from_pretrained(args.model))
    m8 = torch.quantization.quantize_dynamic(m8, {torch.nn.Linear}, dtype=torch.qint8)
    sz8 = param_bytes(m8) / 1e6  # 주: 동적양자화는 런타임 int8, 여기 크기는 근사

    print(f"fp32 파라미터 ~{sz32:.0f} MB | int8(동적) Linear 양자화\n")
    print(f"{'지역':<6}{'fp32(GPU)':>12}{'int8(CPU)':>12}{'변화':>9}")
    a32, a8 = [], []
    for path, name in REGIONS:
        p = Path(path)
        if not p.exists(): continue
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()][:args.n]
        c32 = cer_of(m32, proc, items, dev)
        c8 = cer_of(m8, proc, items, "cpu")
        a32.append((c32, len(items))); a8.append((c8, len(items)))
        print(f"{name:<6}{c32*100:>11.2f}%{c8*100:>11.2f}%{(c8-c32)*100:>+8.2f}%p")
    w32 = sum(v*n for v, n in a32)/sum(n for _, n in a32)
    w8 = sum(v*n for v, n in a8)/sum(n for _, n in a8)
    print(f"\n{'가중평균':<6}{w32*100:>11.2f}%{w8*100:>11.2f}%{(w8-w32)*100:>+8.2f}%p")
    print(f"\n크기: fp32 ~{sz32:.0f}MB → int8 이론상 ~{sz32/4:.0f}MB (whisper.cpp q5 ≈ 유사/더 작음)")


if __name__ == "__main__":
    main()
