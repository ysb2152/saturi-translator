"""실환경 E2E: 소음 입힌 음성 → STT(beam) → 변환(KoBART) → 표준어 CER.

조건: clean, phone+SNR10(실제 폰+생활소음). val_e2e.jsonl(표준 정답) 사용.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, torch, jiwer, soundfile as sf, librosa
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          AutoTokenizer, AutoModelForSeq2SeqLM)

SR = 16000
REGIONS = [("data/processed", "경상"), ("data/processed_jl", "전라"),
           ("data/processed_cc", "충청"), ("data/processed_gw", "강원")]


def load_audio(p):
    arr, sr = sf.read(p)
    if getattr(arr, "ndim", 1) > 1: arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if sr != SR: arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return arr


def add_noise(arr, snr_db, rng):
    sp = np.mean(arr ** 2) + 1e-9
    n = rng.standard_normal(len(arr)).astype(np.float32)
    return arr + n * np.sqrt(sp / (10 ** (snr_db / 10)) / np.mean(n ** 2))


def phone(arr):
    return librosa.resample(librosa.resample(arr, orig_sr=SR, target_sr=8000), orig_sr=8000, target_sr=SR)


def degrade(arr, cond, rng):
    if cond == "clean": return arr
    if cond == "phone+SNR10": return add_noise(phone(arr), 10, rng)
    if cond == "SNR5": return add_noise(arr, 5, rng)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", default="backend/models/whisper-dialect")
    ap.add_argument("--conv", default="backend/models/kobart-dialect")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    CONDS = ["clean", "phone+SNR10"]

    proc = WhisperProcessor.from_pretrained(args.stt, language="korean", task="transcribe")
    stt = WhisperForConditionalGeneration.from_pretrained(args.stt).to(dev).eval()
    stt.generation_config.language = "korean"; stt.generation_config.task = "transcribe"
    stt.generation_config.forced_decoder_ids = None
    tok = AutoTokenizer.from_pretrained(args.conv)
    conv = AutoModelForSeq2SeqLM.from_pretrained(args.conv).to(dev).eval()

    def run(items, cond):
        refs = [x["standard"].strip() for x in items]; preds = []
        for i in range(0, len(items), args.batch):
            ch = items[i:i+args.batch]
            arrs = [degrade(load_audio(x["audio_filepath"]), cond, rng) for x in ch]
            feats = proc.feature_extractor(arrs, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
            with torch.no_grad():
                g = stt.generate(feats, max_new_tokens=128, num_beams=5, no_repeat_ngram_size=3)
            dia = proc.tokenizer.batch_decode(g, skip_special_tokens=True)
            enc = tok(dia, return_tensors="pt", padding=True, truncation=True, max_length=128,
                      return_token_type_ids=False).to(dev)
            with torch.no_grad():
                g2 = conv.generate(**enc, max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3,
                                  repetition_penalty=1.3, early_stopping=True)
            preds += tok.batch_decode(g2, skip_special_tokens=True)
        return jiwer.cer(refs, preds)

    print(f"장치 {dev} | 지역별 {args.n} | 조건 {CONDS}\n")
    print(f"{'지역':<6}" + "".join(f"{c:>14}" for c in CONDS))
    agg = {c: [] for c in CONDS}
    for rdir, name in REGIONS:
        vp = Path(rdir) / "stt" / "val_e2e.jsonl"
        if not vp.exists(): print(f"{name} (val_e2e 없음)"); continue
        items = [json.loads(l) for l in vp.read_text(encoding="utf-8").splitlines()][:args.n]
        row = f"{name:<6}"
        for c in CONDS:
            v = run(items, c); agg[c].append((v, len(items))); row += f"{v*100:>13.2f}%"
        print(row)
    print(f"\n{'가중평균':<6}" + "".join(
        f"{sum(v*n for v,n in agg[c])/sum(n for _,n in agg[c])*100:>13.2f}%" for c in CONDS))


if __name__ == "__main__":
    main()
