"""실환경 강건성 테스트 — held-out 클립에 소음/코덱을 입혀 CER 변화 측정(녹음 불필요).

  training/.venv/Scripts/python.exe training/eval_robust.py --model backend/models/whisper-dialect-lora --n 200
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np, torch, jiwer, soundfile as sf, librosa
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


def add_noise(arr, snr_db, rng):
    sig_p = np.mean(arr ** 2) + 1e-9
    noise = rng.standard_normal(len(arr)).astype(np.float32)
    noise_p = np.mean(noise ** 2)
    scale = np.sqrt(sig_p / (10 ** (snr_db / 10)) / noise_p)
    return arr + noise * scale


def phone_narrow(arr):
    # 16k → 8k → 16k (전화 협대역 시뮬레이션: 고역 손실)
    return librosa.resample(librosa.resample(arr, orig_sr=SR, target_sr=8000), orig_sr=8000, target_sr=SR)


def degrade(arr, cond, rng):
    if cond == "clean": return arr
    if cond == "SNR15": return add_noise(arr, 15, rng)
    if cond == "SNR5": return add_noise(arr, 5, rng)
    if cond == "phone8k": return phone_narrow(arr)
    if cond == "phone+SNR10": return add_noise(phone_narrow(arr), 10, rng)
    return arr


def cer_cond(model, proc, items, cond, dev, rng, batch=8):
    refs, preds = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        arrs = [degrade(load_audio(x["audio_filepath"]), cond, rng) for x in chunk]
        feats = proc.feature_extractor(arrs, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
        with torch.no_grad():
            g = model.generate(feats, max_new_tokens=128, no_repeat_ngram_size=3)
        preds += proc.tokenizer.batch_decode(g, skip_special_tokens=True)
        refs += [x["text"] for x in chunk]
    return jiwer.cer(refs, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="backend/models/whisper-dialect-lora")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    proc = WhisperProcessor.from_pretrained(args.model, language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(dev).eval()
    model.generation_config.language = "korean"; model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    CONDS = ["clean", "SNR15", "SNR5", "phone8k", "phone+SNR10"]
    print(f"장치 {dev} | 지역별 {args.n} | 조건: {CONDS}\n")
    header = f"{'지역':<6}" + "".join(f"{c:>12}" for c in CONDS)
    print(header)
    agg = {c: [] for c in CONDS}
    for path, name in REGIONS:
        p = Path(path)
        if not p.exists(): continue
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()][:args.n]
        row = f"{name:<6}"
        for c in CONDS:
            v = cer_cond(model, proc, items, c, dev, rng)
            agg[c].append((v, len(items)))
            row += f"{v*100:>11.2f}%"
        print(row)
    print(f"\n{'평균':<6}" + "".join(
        f"{sum(v*n for v,n in agg[c])/sum(n for _,n in agg[c])*100:>11.2f}%" for c in CONDS))


if __name__ == "__main__":
    main()
