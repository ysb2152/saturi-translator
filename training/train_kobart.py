"""사투리→표준어 변환(KoBART) 로컬 GPU 파인튜닝.

Colab 없이 로컬 GPU(예: RTX 3080)에서 학습한다. Colab 노트북과 동일한 로직
(EOS 라벨 부착 + 반복 억제 생성). 학습 후 CER 평가 + 모델 저장.

  training/.venv/Scripts/python.exe training/train_kobart.py \
      --data data/processed/mt_balanced --out backend/models/kobart-dialect --epochs 2

빠른 실험: --max-train 100000 --epochs 1
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Windows 콘솔 한글 출력 안전
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

MODEL_NAME = "gogamza/kobart-base-v2"
MAXLEN = 128
GEN = dict(max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3,
           repetition_penalty=1.3, early_stopping=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/mt_balanced")
    ap.add_argument("--out", default="backend/models/kobart-dialect")
    ap.add_argument("--run-dir", default="training/runs")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--max-train", type=int, default=0, help="train 개수 제한(0=전체, 빠른 테스트용)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {dev}", f"({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")

    data = Path(args.data)
    ds = load_dataset("json", data_files={
        "train": str(data / "train.jsonl"),
        "validation": str(data / "val.jsonl")})
    if args.max_train:
        ds["train"] = ds["train"].select(range(min(args.max_train, len(ds["train"]))))
    print(ds)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    eos = tok.eos_token_id

    def prep(batch):
        x = tok(batch["dialect"], max_length=MAXLEN, truncation=True)
        labels = tok(text_target=batch["standard"], max_length=MAXLEN - 1,
                     truncation=True)["input_ids"]
        # KoBART 토크나이저는 EOS를 자동으로 안 붙임 → 디코더가 '멈춤'을 배우도록 추가
        x["labels"] = [ids + [eos] for ids in labels]
        return x

    tokenized = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    collator = DataCollatorForSeq2Seq(tok, model=model)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.run_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=(dev == "cuda"),
        logging_steps=200,
        save_strategy="no",       # 로컬: 중간 체크포인트 저장 안 함(끝에 한 번만)
        report_to="none",
    )
    trainer = Seq2SeqTrainer(model=model, args=targs,
                             train_dataset=tokenized["train"],
                             eval_dataset=tokenized["validation"],
                             data_collator=collator)
    trainer.train()

    # 저장(백엔드가 로드하는 폴더)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"모델 저장: {out}")

    # 평가: copy 기준선 vs KoBART
    import jiwer
    model.eval().to(dev)
    sub = ds["validation"].select(range(min(args.eval_n, len(ds["validation"]))))
    inps, refs = list(sub["dialect"]), list(sub["standard"])
    preds = []
    for i in range(0, len(inps), args.batch):
        ch = inps[i:i + args.batch]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN, return_token_type_ids=False).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, **GEN)
        preds += tok.batch_decode(g, skip_special_tokens=True)
    em = lambda a, b: sum(x.strip() == y.strip() for x, y in zip(a, b)) / len(a)
    print("\n=== 평가 ===")
    print(f"CER  copy   : {jiwer.cer(refs, inps):.4f}")
    print(f"CER  KoBART : {jiwer.cer(refs, preds):.4f}")
    print(f"정확일치 copy   : {em(inps, refs):.3f}")
    print(f"정확일치 KoBART : {em(preds, refs):.3f}")
    print("\n--- 예시 ---")
    shown = 0
    for d, p, r in zip(inps, preds, refs):
        if d != r:
            print(f"방언: {d}\n예측: {p}\n정답: {r}\n")
            shown += 1
            if shown >= 6:
                break


if __name__ == "__main__":
    main()
