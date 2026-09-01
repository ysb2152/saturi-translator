"""변환기(KoBART) 양자화 정확도 검증 — fp32 vs torch 동적 int8, MT val 사투리→표준어 CER."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import torch, jiwer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

REGIONS = [("data/processed/mt/val.jsonl", "경상"), ("data/processed_jl/mt/val.jsonl", "전라"),
           ("data/processed_cc/mt/val.jsonl", "충청"), ("data/processed_gw/mt/val.jsonl", "강원")]


def convert(model, tok, texts, dev, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=128, return_token_type_ids=False).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3,
                              repetition_penalty=1.3, early_stopping=True)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out


def pbytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="backend/models/kobart-dialect")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m32 = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(dev).eval()
    sz = pbytes(m32)
    m8 = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    m8 = torch.quantization.quantize_dynamic(m8, {torch.nn.Linear}, dtype=torch.qint8)

    print(f"fp32 파라미터 ~{sz:.0f} MB | int8 동적 양자화\n")
    print(f"{'지역':<6}{'fp32(GPU)':>12}{'int8(CPU)':>12}{'변화':>9}")
    a32, a8 = [], []
    for path, name in REGIONS:
        p = Path(path)
        if not p.exists(): continue
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()][:args.n]
        dia = [x["dialect"] for x in items]; gt = [x["standard"] for x in items]
        c32 = jiwer.cer(gt, convert(m32, tok, dia, dev))
        c8 = jiwer.cer(gt, convert(m8, tok, dia, "cpu"))
        a32.append((c32, len(items))); a8.append((c8, len(items)))
        print(f"{name:<6}{c32*100:>11.2f}%{c8*100:>11.2f}%{(c8-c32)*100:>+8.2f}%p")
    w32 = sum(v*n for v, n in a32)/sum(n for _, n in a32)
    w8 = sum(v*n for v, n in a8)/sum(n for _, n in a8)
    print(f"\n{'가중평균':<6}{w32*100:>11.2f}%{w8*100:>11.2f}%{(w8-w32)*100:>+8.2f}%p")
    print(f"\n크기: fp32 ~{sz:.0f}MB → int8 이론상 ~{sz/4:.0f}MB")


if __name__ == "__main__":
    main()
