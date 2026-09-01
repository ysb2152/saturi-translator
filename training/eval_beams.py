"""STT 디코딩 비교: greedy vs beam search (지역별 CER). 재학습 없이 디코딩만 변경."""
from __future__ import annotations
import argparse, json, sys
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
    if sr != SR: arr = librosa.resample(arr, orig_sr=SR if False else sr, target_sr=SR)
    return arr


def cer_beams(model, proc, items, dev, beams, batch=8):
    refs, preds = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        feats = proc.feature_extractor([load_audio(x["audio_filepath"]) for x in chunk],
                                       sampling_rate=SR, return_tensors="pt").input_features.to(dev)
        kw = dict(max_new_tokens=128, no_repeat_ngram_size=3)
        if beams > 1:
            kw["num_beams"] = beams
        with torch.no_grad():
            g = model.generate(feats, **kw)
        preds += proc.tokenizer.batch_decode(g, skip_special_tokens=True)
        refs += [x["text"] for x in chunk]
    return jiwer.cer(refs, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="backend/models/whisper-dialect")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--beams", type=int, default=5)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(args.model, language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(dev).eval()
    model.generation_config.language = "korean"; model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print(f"장치 {dev} | {args.model} | 지역별 {args.n}\n")
    print(f"{'지역':<6}{'greedy':>10}{'beam'+str(args.beams):>10}{'변화':>9}")
    agg = {1: [], args.beams: []}
    for path, name in REGIONS:
        p = Path(path)
        if not p.exists(): continue
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()][:args.n]
        g = cer_beams(model, proc, items, dev, 1)
        b = cer_beams(model, proc, items, dev, args.beams)
        agg[1].append((g, len(items))); agg[args.beams].append((b, len(items)))
        print(f"{name:<6}{g*100:>9.2f}%{b*100:>9.2f}%{(b-g)*100:>+8.2f}%p")
    wg = sum(v*n for v, n in agg[1])/sum(n for _, n in agg[1])
    wb = sum(v*n for v, n in agg[args.beams])/sum(n for _, n in agg[args.beams])
    print(f"\n{'가중평균':<6}{wg*100:>9.2f}%{wb*100:>9.2f}%{(wb-wg)*100:>+8.2f}%p")


if __name__ == "__main__":
    main()
