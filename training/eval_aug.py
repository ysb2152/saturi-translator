# 증강이 효과가 있었는지 held-out으로 확인한다. val(학습에 안 쓴) 표준문장에 어미·어휘 변형을 걸어서
# 기존 모델과 증강 모델이 사투리를 표준으로 얼마나 잘 되돌리는지 비교한다.
#
# 사용: training/.venv/Scripts/python.exe training/eval_aug.py

import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "data")
from augment_endings import transform as end_transform  # (region, dialect) or None
from augment_lexical import LEX, strip_punct

import torch
import jiwer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

VAL = "data/processed/mt_balanced/val.jsonl"
OLD = "backend/models/kobart-dialect"
NEW = "backend/models/kobart-dialect-aug"
MAXLEN = 128
GEN = dict(max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3,
           repetition_penalty=1.3, early_stopping=True)


def lex_transform(std):
    toks = std.split(" ")
    new = list(toks)
    hit = False
    for i, tok in enumerate(toks):
        pre, core, post = strip_punct(tok)
        if core in LEX:
            new[i] = pre + LEX[core] + post
            hit = True
    return " ".join(new) if hit else None


def build_cases(n_each=250, seed=0):
    random.seed(seed)
    stds = []
    with open(VAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                s = (o.get("standard") or "").strip()
                if 6 <= len(s) <= 60:
                    stds.append(s)
    random.shuffle(stds)
    end_cases, lex_cases = [], []
    for s in stds:
        if len(end_cases) < n_each:
            t = end_transform(s)
            if t and t[1] != s:
                end_cases.append((t[1], s))
                continue
        if len(lex_cases) < n_each:
            d = lex_transform(s)
            if d and d != s:
                lex_cases.append((d, s))
        if len(end_cases) >= n_each and len(lex_cases) >= n_each:
            break
    return end_cases, lex_cases


def run(model_dir, inputs, dev, batch=32):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).eval().to(dev)
    preds = []
    for i in range(0, len(inputs), batch):
        ch = inputs[i:i + batch]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN, return_token_type_ids=False).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, **GEN)
        preds += tok.batch_decode(g, skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return preds


def report(name, cases, dev, out):
    dia = [d for d, s in cases]
    ref = [s for d, s in cases]
    old = run(OLD, dia, dev)
    new = run(NEW, dia, dev)
    em = lambda a, b: sum(x.strip() == y.strip() for x, y in zip(a, b)) / len(a)
    out.append(f"## {name} (n={len(cases)}, held-out)")
    out.append("| 모델 | CER | 정확일치 |")
    out.append("|---|---:|---:|")
    out.append(f"| copy(원 사투리) | {jiwer.cer(ref, dia):.4f} | {em(dia, ref):.3f} |")
    out.append(f"| 기존 모델 | {jiwer.cer(ref, old):.4f} | {em(old, ref):.3f} |")
    out.append(f"| **증강 모델** | **{jiwer.cer(ref, new):.4f}** | **{em(new, ref):.3f}** |")
    out.append("")
    out.append("예시 (사투리 → 기존 / 증강 / 정답):")
    shown = 0
    for (d, s), o, nw in zip(cases, old, new):
        if o != nw:
            out.append(f"- `{d}`")
            out.append(f"    - 기존: {o}")
            out.append(f"    - 증강: {nw}")
            out.append(f"    - 정답: {s}")
            shown += 1
            if shown >= 8:
                break
    out.append("")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    end_cases, lex_cases = build_cases()
    out = [f"# 증강 효과 검증 (held-out from val)", ""]
    report("종결어미 변형", end_cases, dev, out)
    report("어휘 변형", lex_cases, dev, out)
    text = "\n".join(out) + "\n"
    Path("training/eval_aug.md").write_text(text, encoding="utf-8")
    print("[written] training/eval_aug.md")


if __name__ == "__main__":
    main()
