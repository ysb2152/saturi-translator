"""변환(사투리→표준어) 학습셋 구성.

preprocess.py 가 만든 mt/{train,val}.jsonl 은 중복이 많고 동일쌍(방언=표준)이 ~84%다.
그대로 학습하면 모델이 '그대로 베끼기'만 배우기 쉬우므로:
  1) 완전 중복(같은 (방언,표준)) 제거
  2) 동일쌍을 목표 비율까지 다운샘플링(변형 학습에 집중)
  3) 길이 필터 + 셔플 + (선택) 상한
해서 mt_balanced/{train,val}.jsonl 로 저장한다.

  python data/build_mt_dataset.py --in-dir data/processed/mt --out-dir data/processed/mt_balanced
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(paths):
    """여러 폴더의 {split}.jsonl 을 모두 순회(다지역 통합)."""
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def build(records, identity_ratio: float, min_len: int, max_len: int,
          seed: int, max_n: int):
    rng = random.Random(seed)
    seen = set()
    ident, diff = [], []
    for r in records:
        d = (r.get("dialect") or "").strip()
        s = (r.get("standard") or "").strip()
        if not d or not s:
            continue
        if not (min_len <= len(d) <= max_len and len(s) <= max_len):
            continue
        key = (d, s)
        if key in seen:
            continue
        seen.add(key)
        (ident if d == s else diff).append({"dialect": d, "standard": s})

    # 동일쌍 개수 = 변형쌍 * ratio/(1-ratio)  → 최종 동일쌍 비율이 identity_ratio가 됨
    if identity_ratio <= 0:
        keep_ident = []
    elif identity_ratio >= 1:
        keep_ident = ident
    else:
        target = int(len(diff) * identity_ratio / (1 - identity_ratio))
        rng.shuffle(ident)
        keep_ident = ident[:target]

    out = diff + keep_ident
    rng.shuffle(out)
    if max_n and len(out) > max_n:
        out = out[:max_n]
    return out, len(diff), len(keep_ident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", nargs="+", default=["data/processed/mt"],
                    help="변환쌍 폴더(들). 여러 지역을 합치려면 공백으로 나열")
    ap.add_argument("--out-dir", default="data/processed/mt_balanced")
    ap.add_argument("--identity-ratio", type=float, default=0.3,
                    help="최종 세트에서 동일쌍(방언=표준) 목표 비율")
    ap.add_argument("--min-len", type=int, default=2, help="방언 문장 최소 글자수")
    ap.add_argument("--max-len", type=int, default=150, help="문장 최대 글자수")
    ap.add_argument("--max-train", type=int, default=0, help="train 상한(0=무제한)")
    ap.add_argument("--max-val", type=int, default=5000, help="val 상한")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dirs = [Path(d) for d in args.in_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("입력 폴더:", ", ".join(str(d) for d in in_dirs))

    for split, cap in (("train", args.max_train), ("val", args.max_val)):
        recs = load([d / f"{split}.jsonl" for d in in_dirs])
        out, n_diff, n_ident = build(recs, args.identity_ratio,
                                     args.min_len, args.max_len, args.seed, cap)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{split}] 총 {len(out)}쌍 (변형 {n_diff}, 동일 {n_ident}) → {out_dir/f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
