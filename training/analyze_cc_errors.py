"""충청 val 오류 유형 분석 — 재학습 없이 잔여 오류의 성격 진단.
서빙과 동일 디코딩(beam=5). 오류를 유형별로 분류하고 반복 치환 패턴을 집계.

  training/.venv/Scripts/python.exe training/analyze_cc_errors.py --n 300
"""
from __future__ import annotations
import argparse, json, sys, difflib
from collections import Counter
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import torch, jiwer, numpy as np, soundfile as sf, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000
VAL = "data/processed_cc/stt/val.jsonl"


def load_audio(p):
    arr, sr = sf.read(p)
    if getattr(arr, "ndim", 1) > 1: arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if sr != SR: arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return arr


def lev(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify(r, q):
    r, q = r.strip(), q.strip()
    if r == q: return "정확일치"
    rd, qd = r.replace(" ", ""), q.replace(" ", "")
    if rd == qd: return "띄어쓰기만"
    d = lev(rd, qd); L = max(1, len(rd))
    if d <= 2: return "근소(1~2자: 조사·어미·유사음)"
    if d / L <= 0.15: return "경미(부분 치환)"
    return "중대(어휘·구조)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default="backend/models/whisper-dialect")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    proc = WhisperProcessor.from_pretrained(args.model, language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(dev).eval()
    model.generation_config.language = "korean"; model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    items = [json.loads(l) for l in Path(VAL).read_text(encoding="utf-8").splitlines()][:args.n]
    refs, preds = [], []
    B = 8
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        feats = proc.feature_extractor([load_audio(x["audio_filepath"]) for x in chunk],
                                       sampling_rate=SR, return_tensors="pt").input_features.to(dev)
        with torch.no_grad():
            g = model.generate(feats, max_new_tokens=128, num_beams=5, no_repeat_ngram_size=3)
        preds += proc.tokenizer.batch_decode(g, skip_special_tokens=True)
        refs += [x["text"] for x in chunk]

    overall = jiwer.cer(refs, preds) * 100
    cats = Counter(classify(r, q) for r, q in zip(refs, preds))

    # 반복 치환 패턴(단어 단위 replace op)
    sub = Counter()
    for r, q in zip(refs, preds):
        if r.strip() == q.strip(): continue
        rw, qw = r.split(), q.split()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, rw, qw).get_opcodes():
            if tag == "replace":
                sub[(" ".join(rw[i1:i2]), " ".join(qw[j1:j2]))] += 1

    n = len(refs)
    print(f"\n=== 충청 val 오류 분석 (n={n}, beam=5, 서빙모델) ===")
    print(f"전체 CER: {overall:.2f}%\n")
    print(f"{'유형':<28}{'건수':>6}{'비율':>8}")
    order = ["정확일치", "띄어쓰기만", "근소(1~2자: 조사·어미·유사음)", "경미(부분 치환)", "중대(어휘·구조)"]
    for k in order:
        c = cats.get(k, 0)
        print(f"{k:<28}{c:>6}{c/n*100:>7.1f}%")
    err = n - cats.get("정확일치", 0)
    minor = cats.get("띄어쓰기만", 0) + cats.get("근소(1~2자: 조사·어미·유사음)", 0)
    print(f"\n오류 {err}건 중 경미(띄어쓰기+근소) 비중: {minor}/{err} = {minor/max(1,err)*100:.0f}%")

    print(f"\n=== 반복 치환 패턴 (2회 이상) ===")
    rep = [(k, v) for k, v in sub.most_common() if v >= 2]
    if not rep:
        print("2회 이상 반복되는 치환 없음 → 오류가 산발적(특정 어휘 집중 아님)")
    else:
        print(f"{'정답 → 예측':<40}{'횟수':>5}")
        for (rr, qq), v in rep[:20]:
            print(f"{(rr + ' → ' + qq):<40}{v:>5}")
    uniq = sum(1 for _, v in sub.items() if v == 1)
    print(f"\n치환 종류: 총 {len(sub)}종, 1회성 {uniq}종({uniq/max(1,len(sub))*100:.0f}%), 반복 {len(rep)}종")


if __name__ == "__main__":
    main()
