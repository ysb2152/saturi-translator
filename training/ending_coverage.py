# 변환(MT) 학습 데이터에서 어떤 종결어미가 실제로 부족한지 정량화하는 진단 스크립트.
#
# 두 가지를 본다.
# 하나는, 큐레이션한 지역 종결어미들이 방언 측에 얼마나 나오고 그중 실제 변형된(방언≠표준) 쌍이 몇 개인지.
# 등장도 변형도 적으면 모델이 그 어미를 표준으로 바꾸는 법을 못 배운 것이다(커버리지 공백).
# 다른 하나는, 실제 변형된 쌍에서 마지막 어절 치환 Top — 뭐가 잘 커버되는지 보는 대조군.
#
# 사용: <python> training/ending_coverage.py [jsonl ...]
#   인자 없으면 data/processed/mt/train.jsonl(4지역 병합)을 본다.

import json
import sys
from collections import Counter

# 큐레이션: 지역별 대표 구어 종결어미(어절 끝 표면형). 완전하진 않지만 진단 신호로 충분.
CURATED = {
    "경상": ["임다", "임더", "임니더", "니더", "니껴", "능교", "능기요", "겠능교",
             "나", "노", "데이", "카이", "아이가", "지예", "라예", "다카이"],
    "전라": ["랑께", "쟤", "브렀", "부렀", "잉", "제", "당께", "그려", "께라", "드랑께"],
    "충청": ["유", "겨", "vw유", "슈", "쟈", "댜", "여", "혀"],
    "강원": ["드래요", "래요", "드래", "야드래", "잖소", "우야", "드라야"],
}

def last_eojeol(s):
    s = s.strip()
    if not s:
        return ""
    return s.split()[-1].strip(".?!~,")

def main(paths, out=None):
    curated_hits = {r: {e: [0, 0] for e in es} for r, es in CURATED.items()}  # [등장, 변형]
    tail_change = Counter()  # (방언마지막어절 -> 표준마지막어절)
    n = 0
    n_diff = 0

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                d = o.get("dialect", "")
                s = o.get("standard", "")
                if not d:
                    continue
                n += 1
                diff = d != s
                if diff:
                    n_diff += 1
                    dl, sl = last_eojeol(d), last_eojeol(s)
                    if dl != sl:
                        tail_change[f"{dl} → {sl}"] += 1
                # 큐레이션 어미: 방언 문장이 그 어미로 끝나는지(어절 끝 표면형 포함)
                dtail = last_eojeol(d)
                for r, es in CURATED.items():
                    for e in es:
                        if dtail.endswith(e):
                            curated_hits[r][e][0] += 1
                            if diff:
                                curated_hits[r][e][1] += 1

    lines = []
    def w(s=""):
        lines.append(s)

    w("# 종결어미 커버리지 진단")
    w(f"- 소스: {', '.join(paths)}")
    w(f"- 총 쌍: {n:,} / 변형(방언≠표준): {n_diff:,} ({100*n_diff/max(n,1):.1f}%)")
    w()
    w("## 1. 큐레이션 어미별 등장·변형 (문장 끝 기준)")
    w("- 판정: 등장<30 = [공백], <300 = [희소]")
    w("| 지역 | 어미 | 방언끝 등장 | 그중 변형 | 변형비 | 판정 |")
    w("|---|---|---:|---:|---:|---|")
    for r, es in CURATED.items():
        for e in es:
            occ, chg = curated_hits[r][e]
            ratio = f"{100*chg/occ:.0f}%" if occ else "-"
            flag = "[공백]" if occ < 30 else ("[희소]" if occ < 300 else "")
            w(f"| {r} | {e} | {occ:,} | {chg:,} | {ratio} | {flag} |")
        w("| | | | | | |")
    w()
    w("## 2. 대조군 — 실제 변형된 마지막 어절 치환 Top 30")
    w("| 방언끝 → 표준끝 | 빈도 |")
    w("|---|---:|")
    for k, v in tail_change.most_common(30):
        w(f"| {k} | {v:,} |")

    text = "\n".join(lines) + "\n"
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[written] {out}  ({n:,} pairs scanned)")
    else:
        print(text)

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["data/processed/mt/train.jsonl"]
    main(args, out="training/ending_coverage.md")
