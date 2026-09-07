# 어휘 사투리 합성 증강 — 어미가 아니라 '단어 자체'가 바뀌는 사투리를 보강.
#
# 계기: 어휘째 사투리인 케이스(맡기다→매끼다 등 동사·부사·대명사가 통째로 바뀜)는
#        어미 증강(augment_endings.py)으로 안 잡힌다는 비공개 테스트 피드백.
# 방법: 실제 '표준어' 문장의 어절 중, 사전에 있는 표준 표면형을 사투리 표면형으로 치환해
#        (합성 사투리, 표준어) 쌍 생성. 활용 오류를 피하려 '어절 정확일치'만 사용(중의적 항목 배제).
#
# 사용:
#   <python> data/augment_lexical.py --src data/processed/mt/train.jsonl \
#       --out data/processed/aug/lexical.jsonl --cap 3000 --samples 40

import argparse
import json
import random
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 표준 어절 → 사투리 어절 (표면형 정확일치). 고신뢰·비중의 항목만.
# 중의적/표준과 겹치는 형태(그렇게→그래, 나→내 등)는 넣지 않는다.
LEX = {
    # ── 경상: 테스터 피드백 계열(맡기다=매끼다) ──
    "맡겨": "매껴", "맡겨라": "매끼라", "맡기고": "매끼고", "맡겼어": "매꼈어", "맡길": "매낄",
    # ── 경상: 부사·감탄 ──
    "매우": "억수로", "굉장히": "억수로", "엄청": "억수로", "되게": "억수로",
    "빨리": "퍼뜩", "조금": "쫌", "그냥": "걍",
    # ── 경상: 의문·지시 ──
    "왜": "와", "무엇": "머", "뭐": "머", "뭐를": "머를", "무엇을": "머를",
    "어디": "어데", "어디야": "어덴노",
    # ── 경상: 호칭 ──
    "할머니": "할매", "할아버지": "할배", "아버지": "아부지", "어머니": "어무이",
    "아니야": "아이다", "괜찮아": "안 카나",
    # ── 전라 ──
    "그런데": "근디", "무슨": "뭔", "여기": "여그", "저기": "저짝",
    "할머니가": "할무니가", "빨리빨리": "허벌나게",
    # ── 충청 ──
    "무엇여": "뭐여", "그러니까": "그러니께", "이렇게": "이릏게",
    # ── 강원(어휘는 약해 소수만) ──
    "빨리요": "빨리유", "많아요": "많드래요",
}

PUNCT = ".?!~,…\"')]}"
LPUNCT = "\"'([{"


def strip_punct(tok):
    a = 0
    while a < len(tok) and tok[a] in LPUNCT:
        a += 1
    b = len(tok)
    while b > a and tok[b - 1] in PUNCT:
        b -= 1
    return tok[:a], tok[a:b], tok[b:]  # (앞부호, 핵심, 뒤부호)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/mt/train.jsonl")
    ap.add_argument("--out", default="data/processed/aug/lexical.jsonl")
    ap.add_argument("--cap", type=int, default=3000, help="사전 항목당 최대 사용 수")
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    used = defaultdict(int)   # 사전 항목별 사용 횟수(cap)
    out_pairs = []
    scanned = 0

    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            std = (o.get("standard") or "").strip()
            if not (args.min_len <= len(std) <= args.max_len):
                continue
            scanned += 1

            toks = std.split(" ")
            new_toks = list(toks)
            hit_keys = []
            for i, tok in enumerate(toks):
                pre, core, post = strip_punct(tok)
                if core in LEX and used[core] < args.cap:
                    new_toks[i] = pre + LEX[core] + post
                    hit_keys.append(core)
            if not hit_keys:
                continue
            dialect = " ".join(new_toks)
            if dialect == std:
                continue
            for k in hit_keys:
                used[k] += 1
            out_pairs.append({"dialect": dialect, "standard": std})

    random.shuffle(out_pairs)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in out_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"스캔 표준문장(길이필터 후): {scanned:,}")
    print(f"생성 쌍: {len(out_pairs):,}  → {args.out}\n")
    print("사전 항목별 사용 수(상위 30):")
    for k, v in sorted(used.items(), key=lambda x: -x[1])[:30]:
        print(f"  {k} → {LEX[k]} : {v:,}")

    print(f"\n샘플 {args.samples}개:")
    for p in out_pairs[: args.samples]:
        print(f"  [사투리] {p['dialect']}   ←  [표준] {p['standard']}")


if __name__ == "__main__":
    main()
