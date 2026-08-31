"""사투리 음성 → 사투리 텍스트 (Whisper) 로컬 GPU 파인튜닝.

표준 Whisper가 사투리 발음에서 약한 것을 방언 음성으로 파인튜닝해 개선한다.
입력: data/preprocess_streaming.py(또는 preprocess.py)가 만든 STT 매니페스트
      data/processed/stt/{train,val}.jsonl  ({audio_filepath, text, duration})

  training/.venv/Scripts/python.exe training/train_whisper.py \
      --data data/processed/stt --model openai/whisper-base --epochs 2

빠른 실험: --max-train 3000 --epochs 1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from datasets import Audio, load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

SR = 16000


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any

    def __call__(self, features):
        inp = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inp, return_tensors="pt")
        lab = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(lab, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/stt")
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--out", default="backend/models/whisper-dialect")
    ap.add_argument("--run-dir", default="training/runs_whisper")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1, help="그래디언트 누적(유효배치=batch*accum)")
    ap.add_argument("--lora", action="store_true",
                    help="LoRA 파인튜닝(원본 표준어 능력 보존, 어댑터 작음). 저장 시 병합해 일반 모델로 서빙")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--resume-adapter", default="",
                    help="기존 LoRA 어댑터 경로. 지정하면 새로 만들지 않고 이어서 학습(나눠 학습용)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="학습 전 CER 측정 생략(이어학습에서 시간 절약)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"장치: {dev}", torch.cuda.get_device_name(0) if dev == "cuda" else "")

    data = Path(args.data)
    ds = load_dataset("json", data_files={
        "train": str(data / "train.jsonl"),
        "validation": str(data / "val.jsonl")})
    if args.max_train:
        ds["train"] = ds["train"].select(range(min(args.max_train, len(ds["train"]))))
    print(ds)

    processor = WhisperProcessor.from_pretrained(args.model, language="korean", task="transcribe")

    import soundfile as sf
    import librosa
    import numpy as np

    def load_audio(path):
        arr, sr = sf.read(path)
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)  # 모노
        arr = arr.astype(np.float32)
        if sr != SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
        return arr

    def prep(b):
        arr = load_audio(b["audio_filepath"])
        b["input_features"] = processor.feature_extractor(
            arr, sampling_rate=SR).input_features[0]
        b["labels"] = processor.tokenizer(b["text"]).input_ids
        return b

    ds = ds.map(prep, remove_columns=ds["train"].column_names, num_proc=1)

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "korean"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.to(dev)

    import jiwer

    def eval_cer(m, tag, n=300):
        """val 일부에서 CER 측정(baseline/파인튜닝 공통)."""
        m.eval()
        val = ds["validation"].select(range(min(n, len(ds["validation"]))))
        refs, preds = [], []
        for i in range(0, len(val), args.batch):
            feats = torch.tensor(val[i:i + args.batch]["input_features"]).to(dev)
            with torch.no_grad():
                g = m.generate(feats, max_new_tokens=128)
            preds += processor.tokenizer.batch_decode(g, skip_special_tokens=True)
            refs += processor.tokenizer.batch_decode(
                val[i:i + args.batch]["labels"], skip_special_tokens=True)
        c = jiwer.cer(refs, preds)
        print(f"[{tag}] CER: {c:.4f}")
        return c, refs, preds

    if args.lora:
        if args.resume_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
            print(f"기존 LoRA 어댑터 이어서 학습: {args.resume_adapter}")
        else:
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                              target_modules=["q_proj", "v_proj"],
                              lora_dropout=args.lora_dropout, bias="none")
            model = get_peft_model(model, lcfg)
        print("LoRA (원본 동결, 어댑터만 학습):")
        model.print_trainable_parameters()

    base_cer = None
    if not args.no_baseline:
        print("\n=== 학습 전 CER 측정 ===")
        base_cer, _, _ = eval_cer(model, "before")

    collator = DataCollatorSpeechSeq2Seq(processor)

    def metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": jiwer.cer(ref_str, pred_str)}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.run_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        fp16=(dev == "cuda"),
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
    )
    trainer = Seq2SeqTrainer(model=model, args=targs,
                             train_dataset=ds["train"], eval_dataset=ds["validation"],
                             data_collator=collator, compute_metrics=metrics,
                             processing_class=processor)
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.lora:
        adir = out / "adapter_lora"
        model.save_pretrained(str(adir))  # 어댑터만(수 MB) — 로드맵(온디바이스/지역별) 재사용
        print(f"LoRA 어댑터 저장: {adir}")
        model = model.merge_and_unload()  # 서빙용 병합 → 백엔드는 일반 모델로 로드(변경 불필요)
        model.save_pretrained(str(out))
    else:
        trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    print(f"모델 저장: {out}")

    print("\n=== 파인튜닝 후 ===")
    ft_cer, refs, preds = eval_cer(model, "파인튜닝")
    print(f"\n★ CER  표준 {base_cer:.4f} → 파인튜닝 {ft_cer:.4f}  "
          f"({(base_cer - ft_cer) / base_cer * 100:.0f}% 개선)" if base_cer else "")
    print("--- 예시 (정답 / 예측) ---")
    for r, p in list(zip(refs, preds))[:6]:
        print(f"정답: {r}\n예측: {p}\n")


if __name__ == "__main__":
    main()
