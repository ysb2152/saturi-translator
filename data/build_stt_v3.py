"""STT v3 학습셋을 병합한다. 확대한 경상(processed_gs_big)에 기존 3지역(각 2500)을 합쳐서
train_whisper.py가 읽는 폴더(data/processed_stt_v3/stt)로 만든다.

  training/.venv/Scripts/python.exe data/build_stt_v3.py

val은 확대 경상의 held-out 세션(새 화자·표현) 위주에 기존 지역 val을 조금 섞었다.
audio_filepath는 리포 루트 기준 상대경로라, 루트에서 학습하면 그대로 로드된다."""
import json, os, random, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

random.seed(0)
OUT = "data/processed_stt_v3/stt"
os.makedirs(OUT, exist_ok=True)


def load(p):
    return [l.strip() for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


# train: 확대 경상 + 기존 충청/강원/전라(각 2500)
train = load("data/processed_gs_big/stt/train.jsonl")
n_gs = len(train)
for d in ["processed_cc", "processed_gw", "processed_jl"]:
    train += load(f"data/{d}/stt/train.jsonl")
train = [x for x in train if x]
random.shuffle(train)

# val: 확대 경상 held-out(새 세션) + 기존 3지역 val
val = load("data/processed_gs_big/stt/val.jsonl")
for d in ["processed_cc", "processed_gw", "processed_jl"]:
    val += load(f"data/{d}/stt/val.jsonl")
val = [x for x in val if x]

open(f"{OUT}/train.jsonl", "w", encoding="utf-8").write("\n".join(train) + "\n")
open(f"{OUT}/val.jsonl", "w", encoding="utf-8").write("\n".join(val) + "\n")

print(f"경상 확대 train: {n_gs:,}")
print(f"전체 train: {len(train):,}  (경상 {n_gs:,} + 기존 3지역 7,500)")
print(f"전체 val  : {len(val):,}")
print(f"→ {OUT}/  (학습: --data {OUT})")
