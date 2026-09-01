"""End-to-end 파이프라인 평가: 음성 → STT → 변환(KoBART) → 표준어, 최종 표준어 CER.

표준어 정답은 STT val의 사투리 텍스트를 MT({dialect,standard}) 데이터로 역참조해 복구한다.

  training/.venv/Scripts/python.exe training/eval_e2e.py --n 300
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import torch, jiwer, numpy as np, soundfile as sf, librosa
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


def build_std_lookup(region_dir):
    """MT train+val 의 {dialect:standard} 사전."""
    d = {}
    for sp in ("train", "val"):
        p = Path(region_dir) / "mt" / f"{sp}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    o = json.loads(line)
                    d.setdefault(o["dialect"].strip(), o["standard"].strip())
                except Exception:
                    pass
    return d


def stt_batch(model, proc, arrs, dev):
    feats = proc.feature_extractor(arrs, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
    with torch.no_grad():
        # 서빙과 동일하게 beam search(5)
        g = model.generate(feats, max_new_tokens=128, num_beams=5, no_repeat_ngram_size=3)
    return proc.tokenizer.batch_decode(g, skip_special_tokens=True)


def convert_batch(conv, tok, texts, dev):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=128, return_token_type_ids=False).to(dev)
    with torch.no_grad():
        g = conv.generate(**enc, max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3,
                          repetition_penalty=1.3, early_stopping=True)
    return tok.batch_decode(g, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", default="backend/models/whisper-dialect")
    ap.add_argument("--conv", default="backend/models/kobart-dialect")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    proc = WhisperProcessor.from_pretrained(args.stt, language="korean", task="transcribe")
    stt = WhisperForConditionalGeneration.from_pretrained(args.stt).to(dev).eval()
    stt.generation_config.language = "korean"; stt.generation_config.task = "transcribe"
    stt.generation_config.forced_decoder_ids = None
    tok = AutoTokenizer.from_pretrained(args.conv)
    conv = AutoModelForSeq2SeqLM.from_pretrained(args.conv).to(dev).eval()

    print(f"장치 {dev} | STT={args.stt} | 변환={args.conv} | 지역별 {args.n}\n")
    print(f"{'지역':<6}{'커버리지':>9}{'변환기만':>10}{'E2E(음성→표준)':>16}")
    tot = []
    for rdir, name in REGIONS:
        vp = Path(rdir) / "stt" / "val_e2e.jsonl"  # build_e2e_val.py 생성(표준 정답 포함)
        if not vp.exists():
            print(f"{name:<6} (val_e2e.jsonl 없음 — build_e2e_val.py 먼저 실행)"); continue
        items = [json.loads(l) for l in vp.read_text(encoding="utf-8").splitlines()][:args.n]
        gts = [x["standard"].strip() for x in items]
        dia_gt = [x["dialect"].strip() for x in items]

        e2e_pred, ideal_pred = [], []
        for i in range(0, len(items), args.batch):
            arrs = [load_audio(x["audio_filepath"]) for x in items[i:i+args.batch]]
            dia_pred = stt_batch(stt, proc, arrs, dev)          # 음성→사투리
            e2e_pred += convert_batch(conv, tok, dia_pred, dev)  # →표준(파이프라인)
        for i in range(0, len(items), args.batch):
            ideal_pred += convert_batch(conv, tok, dia_gt[i:i+args.batch], dev)  # 완벽STT 가정

        e2e = jiwer.cer(gts, e2e_pred)
        ideal = jiwer.cer(gts, ideal_pred)
        cov = len(items)
        print(f"{name:<6}{cov:>9}{ideal*100:>9.2f}%{e2e*100:>15.2f}%")
        tot.append((e2e, ideal, len(items)))
    if tot:
        w_e2e = sum(e*n for e, _, n in tot)/sum(n for _, _, n in tot)
        w_id = sum(i*n for _, i, n in tot)/sum(n for _, _, n in tot)
        print(f"\n{'가중평균':<6}{'':>9}{w_id*100:>9.2f}%{w_e2e*100:>15.2f}%")
        print("\n변환기만=완벽STT가정(음성오류 없음) / E2E=실제 음성부터. 둘 차이가 STT 오류의 영향.")


if __name__ == "__main__":
    main()
