# 학습 데이터에 거의 없는 지역 대표 종결어미를 인공으로 만들어 보강한다.
#
# ending_coverage.md로 보니 랑께·능교·드래·임다 같은 지역 상징 어미가 학습쌍에 사실상 0건이었다.
# 그래서 진짜 표준어 문장(데이터의 standard 쪽)에 믿을 만한 어미 치환을 거꾸로 적용해서
# (합성 사투리, 표준어) 쌍을 만든다. 표준측은 실제 문장이라 자연스럽고 사투리측은 어미만 바뀐다.
#
# 오생성을 막으려고 몇 가지를 지킨다: 문장 끝(마지막 어절)에서만, 표면형이 정확히 맞을 때만 바꾸고,
# 규칙당 상한(--cap)으로 균형을 맞추고, 한 문장엔 처음 걸린 규칙 하나만 적용한다.
# 치환도 형태론적으로 안전한 접미사 + 축약형 사전 몇 개만 쓴다(무리한 활용 변형은 안 함).
#
# 사용:
#   <python> data/augment_endings.py --src data/processed/mt/train.jsonl \
#       --out data/processed/aug/endings.jsonl --cap 8000 --samples 40

import argparse
import json
import random
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (지역, 표준 접미사, 사투리 접미사) — 문장 끝에서 표준→사투리로 치환.
# 접미사 매칭은 문장부호를 뗀 마지막 어절의 끝 표면형 기준. 위에서부터 첫 매칭 하나만 적용.
RULES = [
    # ── 경상: 존대 종결(데이터 공백의 핵심) ──
    ("경상", "습니다", "심더"),      # 먹었습니다 → 먹었심더
    ("경상", "입니다", "임더"),      # 학생입니다 → 학생임더
    ("경상", "습니까", "능교"),      # 먹었습니까 → 먹었능교
    ("경상", "합니까", "하능교"),    # 뭐 합니까 → 뭐 하능교
    ("경상", "습니꺼", "심꺼"),
    ("경상", "지요", "지예"),        # 그렇지요 → 그렇지예
    ("경상", "네요", "네예"),        # 좋네요 → 좋네예
    ("경상", "세요", "이소"),        # 하세요 → 하이소
    # ── 전라: 축약형은 사전으로 안전 처리 ──
    ("전라", "그러니까", "그랑께"),
    ("전라", "그니까", "그랑께"),
    ("전라", "니까", "당께"),        # 하니까 → 하당께
    ("전라", "버렸어", "부렀어"),    # 해버렸어 → 해부렀어
    ("전라", "버렸다", "부렀다"),
    ("전라", "그래", "그려"),        # 문장 끝 그래 → 그려
    # ── 충청: 요/야 계열(규칙적) ──
    ("충청", "지요", "쥬"),          # 그렇지요 → 그렇쥬
    ("충청", "이에요", "이에유"),
    ("충청", "예요", "예유"),
    ("충청", "어요", "어유"),        # 먹어요 → 먹어유
    ("충청", "아요", "아유"),        # 좋아요 → 좋아유
    ("충청", "해요", "해유"),
    ("충청", "세요", "세유"),
    ("충청", "이야", "이여"),        # 그거야 → 그거여 (받침 뒤 이야)
    ("충청", "야", "여"),            # 뭐야 → 뭐여
    # ── 강원: 반말/존대 보고체 ──
    ("강원", "었어요", "었드래요"),  # 그랬어요 → 그랬드래요
    ("강원", "았어요", "았드래요"),
    ("강원", "었어", "었드래"),      # 그랬어 → 그랬드래
    ("강원", "았어", "았드래"),
]

PUNCT = ".?!~,…"


def split_punct(s):
    s = s.rstrip()
    i = len(s)
    while i > 0 and s[i - 1] in PUNCT:
        i -= 1
    return s[:i], s[i:]


def transform(standard):
    """표준 문장 → (지역, 합성 사투리) 목록. 첫 매칭 규칙 하나만."""
    core, punct = split_punct(standard)
    if not core:
        return None
    for region, sfx, rep in RULES:
        if core.endswith(sfx) and len(core) > len(sfx):
            dialect = core[: -len(sfx)] + rep + punct
            return region, dialect
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/mt/train.jsonl")
    ap.add_argument("--out", default="data/processed/aug/endings.jsonl")
    ap.add_argument("--cap", type=int, default=8000, help="규칙당 최대 생성 수")
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    # 규칙별 후보 수집(중복 표준문장 제거)
    per_rule = defaultdict(list)  # (region, sfx, rep) -> [(dialect, standard)]
    seen = set()
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
            core, punct = split_punct(std)
            if not core:
                continue
            for region, sfx, rep in RULES:
                if core.endswith(sfx) and len(core) > len(sfx):
                    key = (std, sfx)
                    if key in seen:
                        break
                    seen.add(key)
                    dialect = core[: -len(sfx)] + rep + punct
                    if dialect == std:
                        break
                    per_rule[(region, sfx, rep)].append((dialect, std))
                    break

    # 규칙당 cap 샘플링, 셔플
    out_pairs = []
    counts = []
    for (region, sfx, rep), pairs in RULES_ORDER(per_rule):
        random.shuffle(pairs)
        take = pairs[: args.cap]
        counts.append((region, f"{sfx}→{rep}", len(pairs), len(take)))
        for d, s in take:
            out_pairs.append({"dialect": d, "standard": s})
    random.shuffle(out_pairs)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in out_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"스캔 표준문장(길이필터 후): {scanned:,}")
    print(f"생성 쌍: {len(out_pairs):,}  → {args.out}\n")
    print("규칙별 (지역 | 치환 | 후보 | 채택)")
    for region, rule, cand, took in counts:
        print(f"  {region} | {rule} | {cand:,} | {took:,}")

    print(f"\n샘플 {args.samples}개:")
    for p in out_pairs[: args.samples]:
        print(f"  [사투리] {p['dialect']}   ←  [표준] {p['standard']}")


def RULES_ORDER(per_rule):
    """RULES 정의 순서대로 (key, pairs) 반환."""
    for region, sfx, rep in RULES:
        yield (region, sfx, rep), per_rule.get((region, sfx, rep), [])


if __name__ == "__main__":
    main()
