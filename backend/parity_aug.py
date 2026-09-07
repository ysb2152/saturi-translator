"""증강 .pte 정합성 — 실제 encoder/decoder.pte(int8) greedy vs PyTorch, 증강 케이스 유지 확인.
실행: MODEL=... PTE=... C:/et/Scripts/python.exe backend/parity_aug.py
결과는 backend/parity_aug.md 로 저장."""
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer
from executorch.runtime import Runtime

MODEL = os.getenv("MODEL", "backend/models/kobart-dialect-aug")
PTE = os.getenv("PTE", "backend/models/kobart-pte-aug")

TESTS = [
    "업체에 매껴 놓고 알아서 준비해주세요",   # 매껴→맡겨 (테스터 피드백)
    "어데 갈 때는 여기다 매껴 놓고 가나",
    "끓여 먹기 솔직히 귀찮으당께",             # 당께→니까
    "그 사람도 그 가운데 한명이었드래",         # 드래→어
    "네 저는 대학교까지 다 졸업을 했심더",      # 심더→습니다
    "요리왕 비룡이라고 혹시 봤능교",           # 능교→습니까
    "그 최근 또 보고 있어유",                  # 유→요
    "우리 어무이 취미가 음식 만드는 거야",       # 어무이→어머니
]

tok = AutoTokenizer.from_pretrained(MODEL)
pt = BartForConditionalGeneration.from_pretrained(MODEL).eval()
rt = Runtime.get()
enc_m = rt.load_program(f"{PTE}/encoder.pte").load_method("forward")
dec_m = rt.load_program(f"{PTE}/decoder.pte").load_method("forward")
DEC_START, EOS = pt.config.decoder_start_token_id, pt.config.eos_token_id


def pt_greedy(text):
    ein = tok(text, return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        g = pt.generate(**ein, max_new_tokens=64, num_beams=1)
    return tok.decode(g[0], skip_special_tokens=True)


def pte_greedy(text):
    ein = tok(text, return_tensors="pt", return_token_type_ids=False)
    ids32, mask32 = ein["input_ids"].int(), ein["attention_mask"].int()
    enc_hidden = enc_m.execute([ids32, mask32])[0]
    dec = [DEC_START]
    for _ in range(64):
        dt = torch.tensor([dec], dtype=torch.int32)
        logits = dec_m.execute([dt, enc_hidden, mask32])[0]
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        dec.append(nxt)
    return tok.decode(dec[1:], skip_special_tokens=True)


out = ["# 증강 .pte 정합성 (int8 ExecuTorch vs PyTorch, greedy)", "",
       f"- MODEL={MODEL}", f"- PTE={PTE}", "",
       "| 입력(사투리) | PyTorch | .pte(int8) | 일치 |", "|---|---|---|:--:|"]
match = 0
for t in TESTS:
    a, b = pt_greedy(t), pte_greedy(t)
    ok = a.strip() == b.strip()
    match += ok
    out.append(f"| {t} | {a} | {b} | {'✓' if ok else '✗'} |")
out.append("")
out.append(f"**일치: {match}/{len(TESTS)}**")
open("backend/parity_aug.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"[written] backend/parity_aug.md  ({match}/{len(TESTS)} match)")
