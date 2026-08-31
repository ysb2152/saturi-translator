"""학습된 STT 모델을 지역별 held-out val로 평가(진짜 CER).

  training/.venv/Scripts/python.exe training/eval_stt.py --model backend/models/whisper-dialect-lora --n 400
  (--baseline 로 표준 whisper-small 도 함께 측정)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import torch, jiwer, numpy as np, soundfile as sf, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000
REGIONS = [("data/processed/stt/val.jsonl", "경상"),
           ("data/processed_jl/stt/val.jsonl", "전라"),
           ("data/processed_cc/stt/val.jsonl", "충청"),
           ("data/processed_gw/stt/val.jsonl", "강원")]


def load_audio(p):
    arr, sr = sf.read(p)
    if getattr(arr, "ndim", 1) > 1: arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if sr != SR: arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return arr


def cer_on(model, proc, items, dev, batch=8):
    refs, preds = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        feats = proc.feature_extractor([load_audio(x["audio_filepath"]) for x in chunk],
                                       sampling_rate=SR, return_tensors="pt").input_features.to(dev)
        with torch.no_grad():
            g = model.generate(feats, max_new_tokens=128, no_repeat_ngram_size=3)
        preds += proc.tokenizer.batch_decode(g, skip_special_tokens=True)
        refs += [x["text"] for x in chunk]
    return jiwer.cer(refs, preds), refs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="backend/models/whisper-dialect-lora")
    ap.add_argument("--baseline", action="store_true", help="표준 whisper-small 도 측정")
    ap.add_argument("--n", type=int, default=400, help="지역별 평가 샘플 수")
    ap.add_argument("--examples", type=int, default=0, help="지역별 정답/예측 예시 출력 개수")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    proc = WhisperProcessor.from_pretrained(args.model, language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(dev).eval()
    model.generation_config.language = "korean"; model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    base = None
    if args.baseline:
        base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(dev).eval()
        base.generation_config.language = "korean"; base.generation_config.task = "transcribe"
        base.generation_config.forced_decoder_ids = None

    print(f"장치 {dev} | 모델 {args.model} | 지역별 {args.n}개\n")
    print(f"{'지역':<6}{'표준' if base else '':<10}{'학습모델':<10}")
    all_ref = []
    tot_ft = []
    for path, name in REGIONS:
        p = Path(path)
        if not p.exists():
            print(f"{name:<6} (val 없음)"); continue
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()][:args.n]
        ft, refs, preds = cer_on(model, proc, items, dev)
        row = f"{name:<6}"
        if base is not None:
            b, _, _ = cer_on(base, proc, items, dev)
            row += f"{b*100:>7.2f}%  "
        row += f"{ft*100:>7.2f}%"
        print(row)
        tot_ft.append((name, ft, len(items)))
        if args.examples:
            # 정확일치 + 근접오류 섞어서 보여주기
            exact = sum(1 for r, q in zip(refs, preds) if r.strip() == q.strip())
            print(f"   └ 정확일치 {exact}/{len(refs)} ({exact/len(refs)*100:.0f}%)")
            for r, q in list(zip(refs, preds))[:args.examples]:
                mark = "✓" if r.strip() == q.strip() else "≈"
                print(f"     {mark} 정답: {r}")
                print(f"       예측: {q}")
    # 가중 평균
    if tot_ft:
        w = sum(f*n for _, f, n in tot_ft) / sum(n for _, _, n in tot_ft)
        print(f"\n전체(가중평균) 학습모델 CER: {w*100:.2f}%")


if __name__ == "__main__":
    main()
