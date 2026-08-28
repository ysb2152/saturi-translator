"""변환 문장쌍(mt/*.jsonl) 탐색적 분석(EDA).

'표준 모델이 못하는 영역을 데이터로 개선한다'는 서사를 숫자로 뒷받침하기 위해:
  - 동일/변형 쌍 비율
  - 문장 길이 분포
  - 가장 흔한 사투리 어절 → 표준 어절 치환(Top-N)
  - 사투리 종결어미 패턴
을 집계해 콘솔 + data/analysis.md 로 남긴다. GPU 불필요, CPU 수 초.

  python data/analyze.py --in-dir data/processed/mt --out data/analysis.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_pairs(in_dir: Path):
    for name in ("train.jsonl", "val.jsonl"):
        p = in_dir / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    yield r.get("dialect", ""), r.get("standard", "")


def length_buckets(lengths):
    buckets = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 10**9)]
    labels = ["1-10", "11-20", "21-30", "31-50", "51-100", "100+"]
    counts = [0] * len(buckets)
    for L in lengths:
        for i, (lo, hi) in enumerate(buckets):
            if lo < L <= hi:
                counts[i] += 1
                break
    return labels, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/processed/mt")
    ap.add_argument("--out", default="data/analysis.md")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    total = same = diff = 0
    lengths = []
    subs = Counter()            # (사투리어절, 표준어절) 치환 빈도
    dialect_end = Counter()     # 사투리 문장 마지막 어절 끝 2글자(종결 패턴 근사)

    for d, s in load_pairs(Path(args.in_dir)):
        if not d or not s:
            continue
        total += 1
        lengths.append(len(d))
        if d == s:
            same += 1
            continue
        diff += 1
        dw, sw = d.split(), s.split()
        if len(dw) == len(sw):  # 위치 정렬되는 경우만 어절 치환 집계
            for a, b in zip(dw, sw):
                if a != b:
                    subs[(a, b)] += 1
        if dw:
            dialect_end[dw[-1][-2:]] += 1

    labels, counts = length_buckets(lengths)
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    top_subs = subs.most_common(args.top)

    # ---- 마크다운 리포트 ----
    lines = []
    lines.append("# 데이터 분석 (경상도 방언 → 표준어 문장쌍)\n")
    lines.append("`data/analyze.py` 자동 생성. 소스: AI Hub 한국어 방언 발화(경상도) Training 라벨.\n")
    lines.append("## 1. 규모와 변형 비율\n")
    lines.append(f"- 총 문장쌍: **{total:,}**")
    lines.append(f"- 변형 있음(사투리≠표준): **{diff:,}** ({diff/total*100:.1f}%)")
    lines.append(f"- 동일(사투리=표준): **{same:,}** ({same/total*100:.1f}%)")
    lines.append("")
    lines.append("> 대부분의 어절은 표준과 같고 일부만 바뀐다. 그래서 학습셋은 동일쌍을 "
                 "다운샘플링(→30%)해 '변형'에 집중시켰다(build_mt_dataset.py).\n")
    lines.append("## 2. 문장 길이 분포 (사투리 글자 수)\n")
    lines.append(f"- 평균 {avg_len:.1f}자\n")
    lines.append("| 글자 수 | 문장 수 | 비율 |")
    lines.append("|---|---|---|")
    for lb, c in zip(labels, counts):
        lines.append(f"| {lb} | {c:,} | {c/total*100:.1f}% |")
    lines.append("\n> 대화 발화라 대체로 짧다 → 토큰 max_length 128로 충분(why_finetune.md 참고).\n")
    lines.append(f"## 3. 가장 흔한 사투리 → 표준 어절 치환 (Top {args.top})\n")
    lines.append("| 순위 | 사투리 | 표준 | 빈도 |")
    lines.append("|---|---|---|---|")
    for i, ((a, b), c) in enumerate(top_subs, 1):
        lines.append(f"| {i} | {a} | {b} | {c:,} |")
    lines.append("\n> 규칙 스텁 사전 대신 이런 다양한 치환을 **데이터로 학습**하는 것이 KoBART의 역할이다.\n")
    lines.append("## 4. 사투리 문장 종결 패턴 (끝 2글자 Top 15)\n")
    lines.append("| 끝음절 | 빈도 |")
    lines.append("|---|---|")
    for suf, c in dialect_end.most_common(15):
        lines.append(f"| …{suf} | {c:,} |")
    lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")

    # 콘솔 요약
    print(f"총 {total:,}쌍 | 변형 {diff:,}({diff/total*100:.1f}%) | 동일 {same:,}")
    print(f"평균 길이 {avg_len:.1f}자")
    print("Top 10 치환:")
    for (a, b), c in top_subs[:10]:
        print(f"  {a} → {b}  ({c:,})")
    print(f"리포트 저장: {args.out}")


if __name__ == "__main__":
    main()
