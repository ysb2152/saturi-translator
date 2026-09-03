"""실제 온디바이스 STT 모델 양자화 검증: f16 GGUF vs q5_0 GGUF, 같은 파일로 CER 비교.
whisper-cli.exe(공식 프리빌트)로 두 모델을 각각 돌려 지역별/전체 CER과 델타를 낸다.
"""
import json, subprocess, sys, os, random
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import jiwer

ROOT = Path(r"C:/Users/ysb21/saturi-translator")
CLI = Path(r"C:/Users/ysb21/AppData/Local/Temp/claude/C--Users-ysb21-saturi-translator/8949ad4b-d0be-4dac-9233-d93afedeb500/scratchpad/wcpp/Release/whisper-cli.exe")
GG = ROOT / "backend/models/whisper-dialect-gguf"
F16 = GG / "ggml-model-f16.bin"
Q5 = GG / "ggml-model-q5_0.bin"
REGIONS = [("data/processed/stt/val.jsonl","경상"),("data/processed_jl/stt/val.jsonl","전라"),
           ("data/processed_cc/stt/val.jsonl","충청"),("data/processed_gw/stt/val.jsonl","강원")]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
random.seed(42)

def transcribe(model, wav):
    r = subprocess.run([str(CLI),"-m",str(model),"-f",str(wav),"-l","ko","-nt","-np"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return " ".join(l.strip() for l in r.stdout.splitlines() if l.strip())

def norm(s):  # 공백/구두점 정규화 (CER 공정 비교)
    import re
    return re.sub(r"\s+"," ", re.sub(r"[.,!?…]"," ", s)).strip()

items = []
for path, name in REGIONS:
    lines = [json.loads(x) for x in open(ROOT/path, encoding="utf-8")]
    random.shuffle(lines)
    for it in lines[:N]:
        it["_region"] = name
        it["audio_filepath"] = str(ROOT / it["audio_filepath"])
        items.append(it)
print(f"총 {len(items)}개 파일 × 2모델 평가 시작 (지역당 {N})", flush=True)

rows = {name: {"ref":[], "f16":[], "q5":[]} for _,name in REGIONS}
for i, it in enumerate(items):
    ref = norm(it["text"]); r = it["_region"]
    pf = norm(transcribe(F16, it["audio_filepath"]))
    pq = norm(transcribe(Q5, it["audio_filepath"]))
    rows[r]["ref"].append(ref); rows[r]["f16"].append(pf); rows[r]["q5"].append(pq)
    if (i+1) % 10 == 0: print(f"  {i+1}/{len(items)}", flush=True)

print("\n지역 | f16 CER | q5 CER | 델타(q5-f16)")
print("-"*46)
allref, allf16, allq5 = [], [], []
for _, name in REGIONS:
    d = rows[name];
    if not d["ref"]: continue
    cf = jiwer.cer(d["ref"], d["f16"]); cq = jiwer.cer(d["ref"], d["q5"])
    allref += d["ref"]; allf16 += d["f16"]; allq5 += d["q5"]
    print(f"{name} | {cf*100:6.2f}% | {cq*100:6.2f}% | {(cq-cf)*100:+.2f}%p")
CF = jiwer.cer(allref, allf16); CQ = jiwer.cer(allref, allq5)
print("-"*46)
print(f"전체 | {CF*100:6.2f}% | {CQ*100:6.2f}% | {(CQ-CF)*100:+.2f}%p")
# f16과 q5 예측이 완전히 동일한 비율
same = sum(1 for a,b in zip(allf16, allq5) if a==b)
print(f"\nf16==q5 완전 일치: {same}/{len(allf16)} ({same/len(allf16)*100:.1f}%)")
json.dump({"f16_cer":CF,"q5_cer":CQ,"delta":CQ-CF,"same":same,"total":len(allf16),
           "per_region":{n:{"f16":jiwer.cer(rows[n]["ref"],rows[n]["f16"]),
                            "q5":jiwer.cer(rows[n]["ref"],rows[n]["q5"])} for _,n in REGIONS if rows[n]["ref"]}},
          open(ROOT/"backend/q5_vs_f16_result.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n저장: backend/q5_vs_f16_result.json")
