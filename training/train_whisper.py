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


def _load_audio_np(path):
    import soundfile as sf, librosa, numpy as np
    arr, sr = sf.read(path)
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if sr != SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return arr


_NOISE_FILES = None


def _noise_files():
    """MUSAN 실제 소음 wav 목록(환경변수 MUSAN_NOISE_DIR, 기본 data/noise/musan/noise)."""
    global _NOISE_FILES
    if _NOISE_FILES is None:
        import os, glob
        d = os.getenv("MUSAN_NOISE_DIR", "data/noise/musan/noise")
        _NOISE_FILES = glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)
        if _NOISE_FILES:
            print(f"[augment] MUSAN 실제 소음 {len(_NOISE_FILES)}개 사용")
        else:
            print("[augment] MUSAN 없음 → 가우시안 소음 폴백")
    return _NOISE_FILES


def _real_noise(n, rng):
    """MUSAN에서 랜덤 소음 조각을 길이 n 으로 반환(없으면 None → 가우시안 폴백)."""
    import numpy as np, soundfile as sf, librosa
    files = _noise_files()
    if not files:
        return None
    for _ in range(3):
        try:
            a, sr = sf.read(files[rng.integers(len(files))])
            if getattr(a, "ndim", 1) > 1:
                a = a.mean(axis=1)
            a = a.astype(np.float32)
            if sr != SR:
                a = librosa.resample(a, orig_sr=sr, target_sr=SR)
            if len(a) == 0:
                continue
            if len(a) < n:
                a = np.tile(a, int(np.ceil(n / len(a))))
            st = int(rng.integers(0, len(a) - n + 1)) if len(a) > n else 0
            return a[st:st + n]
        except Exception:
            continue
    return None


def _reverb(arr, rng):
    """합성 잔향(exponential-decay RIR convolution). 실측 RIR 없이 잔향 강건성 부여."""
    import numpy as np
    rt60 = rng.uniform(0.1, 0.6)
    n = max(8, int(rt60 * SR))
    t = np.arange(n)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / n)
    ir[0] = 1.0
    peak = float(np.max(np.abs(arr))) + 1e-9
    out = np.convolve(arr, ir)[:len(arr)]
    out *= peak / (float(np.max(np.abs(out))) + 1e-9)
    return out.astype(np.float32)


def _augment(arr, rng):
    """on-the-fly 증강: 잔향 + 전화 협대역 + harder SNR 소음(MUSAN 실제소음 우선)."""
    import numpy as np, librosa
    if rng.random() < 0.30:
        arr = _reverb(arr, rng)
    if rng.random() < 0.25:
        arr = librosa.resample(librosa.resample(arr, orig_sr=SR, target_sr=8000),
                               orig_sr=8000, target_sr=SR)
    if rng.random() < 0.80:
        snr = rng.uniform(-5, 20)  # harder: 신호보다 소음이 큰 경우까지
        sig = float(np.mean(arr ** 2)) + 1e-9
        noise = _real_noise(len(arr), rng)
        if noise is None:
            noise = rng.standard_normal(len(arr)).astype(np.float32)
        scale = np.sqrt(sig / (10 ** (snr / 10)) / (float(np.mean(noise ** 2)) + 1e-9))
        arr = arr + noise * scale
    return arr.astype(np.float32)


def _spec_augment(feats, rng):
    """SpecAugment: mel 특징에 시간/주파수 마스킹(배치 텐서 in-place)."""
    b, M, T = feats.shape
    for i in range(b):
        for _ in range(2):
            f = int(rng.integers(0, 16))
            if f > 0:
                f0 = int(rng.integers(0, max(1, M - f)))
                feats[i, f0:f0 + f, :] = 0.0
        for _ in range(2):
            tt = int(rng.integers(0, 70))
            if tt > 0:
                t0 = int(rng.integers(0, max(1, T - tt)))
                feats[i, :, t0:t0 + tt] = 0.0
    return feats


@dataclass
class AugmentCollator:
    """학습 배치마다 오디오를 로드→증강→특징추출(캐시 없음, 매 에폭 다른 소음)."""
    processor: Any

    def __post_init__(self):
        import numpy as np
        self._rng = np.random.default_rng()

    def __call__(self, features):
        arrs = [_augment(_load_audio_np(f["audio_filepath"]), self._rng) for f in features]
        inp = self.processor.feature_extractor(arrs, sampling_rate=SR, return_tensors="pt")
        feats = _spec_augment(inp.input_features, self._rng)  # SpecAugment(시간/주파수 마스킹)
        batch = {"input_features": feats}
        lab = [{"input_ids": f["labels"]} for f in features]
        lb = self.processor.tokenizer.pad(lab, return_tensors="pt")
        labels = lb["input_ids"].masked_fill(lb.attention_mask.ne(1), -100)
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
    ap.add_argument("--augment", action="store_true",
                    help="학습 배치에 on-the-fly 소음/코덱 증강(평가는 clean). map 캐시 없음")
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

    if args.augment:
        # 증강 모드: 특징은 콜레이터가 배치마다 계산 → map은 라벨만(경로 유지, 캐시 최소)
        def prep(b):
            b["labels"] = processor.tokenizer(b["text"]).input_ids
            return b
        ds = ds.map(prep, remove_columns=["text", "duration"], num_proc=1)
    else:
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
        has_feat = "input_features" in val.column_names
        for i in range(0, len(val), args.batch):
            b = val[i:i + args.batch]
            if has_feat:
                feats = torch.tensor(b["input_features"]).to(dev)
            else:  # 증강 모드: 평가는 clean 오디오로 특징 계산
                arrs = [_load_audio_np(p) for p in b["audio_filepath"]]
                feats = processor.feature_extractor(
                    arrs, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
            with torch.no_grad():
                g = m.generate(feats, max_new_tokens=128)
            preds += processor.tokenizer.batch_decode(g, skip_special_tokens=True)
            refs += processor.tokenizer.batch_decode(b["labels"], skip_special_tokens=True)
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

    collator = AugmentCollator(processor) if args.augment else DataCollatorSpeechSeq2Seq(processor)
    if args.augment:
        print("소음/코덱 on-the-fly 증강 학습 (평가는 clean)")

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
        remove_unused_columns=not args.augment,  # 증강 모드는 콜레이터가 audio_filepath 필요
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
