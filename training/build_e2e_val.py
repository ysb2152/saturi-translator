"""기존 STT val 클립에 표준어 정답을 붙여 E2E 평가용 매니페스트 생성.

라벨(Downloads/download*.tar)에서 stem별 발화 리스트를 읽어, 클립 파일명
{stem}_{index}.wav 의 index 발화의 standard_form 을 정답으로 매칭한다.
출력: data/processed_*/stt/val_e2e.jsonl  ({audio_filepath, dialect, standard})
"""
from __future__ import annotations
import sys, os, glob, json, tarfile
from pathlib import Path
sys.path.insert(0, r'C:\Users\ysb21\뭐라는겨\data')
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from preprocess_streaming import open_audio_zip
from aihub_schema import get_utterance_list, extract_utterance

DL = r'C:\Users\ysb21\Downloads'
REGIONS = [("data/processed", "경상"), ("data/processed_jl", "전라"),
           ("data/processed_cc", "충청"), ("data/processed_gw", "강원")]

# 모든 라벨 tar에서 stem(소문자) → 발화별 standard 리스트 전역 사전
print("라벨 읽는 중...")
stem2std = {}
for t in glob.glob(os.path.join(DL, "download*.tar")):
    try:
        if os.path.getsize(t) > 200 * 1024 * 1024:  # 라벨은 ~40MB — 대용량(음성) 파일 스킵
            continue
        if not tarfile.is_tarfile(t):
            continue
        z = open_audio_zip(t)
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            stem = os.path.splitext(os.path.basename(n))[0].lower()
            if stem in stem2std:
                continue
            try:
                obj = json.loads(z.read(n).decode("utf-8", "ignore"))
            except Exception:
                continue
            stds = []
            for u in get_utterance_list(obj):
                eu = extract_utterance(u)
                stds.append(eu.standard if eu else None)
            stem2std[stem] = stds
    except Exception:
        pass
print(f"라벨 stem {len(stem2std)}개 로드")

for rdir, name in REGIONS:
    vp = Path(rdir) / "stt" / "val.jsonl"
    if not vp.exists():
        print(f"{name}: val 없음"); continue
    out = Path(rdir) / "stt" / "val_e2e.jsonl"
    n_ok = n_miss = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in vp.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            base = os.path.splitext(os.path.basename(x["audio_filepath"]))[0]
            stem, _, idx = base.rpartition("_")
            stds = stem2std.get(stem.lower())
            if stds is None or not idx.isdigit() or int(idx) >= len(stds) or stds[int(idx)] is None:
                n_miss += 1; continue
            std = stds[int(idx)].strip()
            if not std:
                n_miss += 1; continue
            f.write(json.dumps({"audio_filepath": x["audio_filepath"],
                                "dialect": x["text"], "standard": std}, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"{name}: 매칭 {n_ok} / 실패 {n_miss}  → {out}")
