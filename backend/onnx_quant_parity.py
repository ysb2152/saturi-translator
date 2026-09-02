"""KoBART ONNX 온디바이스 검증: int8 양자화(용량) + PyTorch 대비 생성 정합성.
실행: backend/.venv/Scripts/python.exe backend/onnx_quant_parity.py
"""
import os, sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ONNX_DIR = Path("backend/models/kobart-onnx")
PT_DIR = "backend/models/kobart-dialect"
GEN = dict(max_new_tokens=64, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.3)

SAMPLES = [
    "밥 문나 아직 안 무따",
    "내 퍼뜩 갈끼니까 쪼매만 기다리라",
    "거시기 그 냥반이 뭐라 카든가",
    "언능 온나 비 올락 말락 한다",
    "그카지 말고 밥 무라 안카나",
]

def mb(p): return os.path.getsize(p) / 1e6

# 1) int8 동적 양자화 (encoder + merged decoder)
from onnxruntime.quantization import quantize_dynamic, QuantType
# 비-merged 그래프를 양자화(merged는 If-노드 서브그래프라 동적 양자화가 못 들어감)
targets = ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]
print("=== int8 동적 양자화 ===")
for t in targets:
    src = ONNX_DIR / t
    dst = ONNX_DIR / t.replace(".onnx", "_int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"{t}: {mb(src):.0f}MB → {mb(dst):.0f}MB (int8)")

enc8 = mb(ONNX_DIR / "encoder_model_int8.onnx")
dec8 = mb(ONNX_DIR / "decoder_model_int8.onnx")
past8 = mb(ONNX_DIR / "decoder_with_past_model_int8.onnx")
print(f"변환기 int8 합계(encoder+decoder+with_past): {enc8+dec8+past8:.0f}MB")

# 2) 정합성: PyTorch vs ONNX(fp32) 생성 비교
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained(PT_DIR)
pt = BartForConditionalGeneration.from_pretrained(PT_DIR).eval()
ort = ORTModelForSeq2SeqLM.from_pretrained(str(ONNX_DIR))

def gen(model, text):
    ids = tok(text, return_tensors="pt", return_token_type_ids=False)
    with torch.no_grad():
        out = model.generate(**ids, **GEN)
    return tok.decode(out[0], skip_special_tokens=True)

print("\n=== PyTorch vs ONNX(fp32) 생성 정합성 ===")
match = 0
for s in SAMPLES:
    a = gen(pt, s); b = gen(ort, s)
    ok = a.strip() == b.strip()
    match += ok
    print(f"[{'일치' if ok else '불일치'}] 입력: {s}")
    print(f"   PT : {a}")
    if not ok: print(f"   ONNX: {b}")
print(f"\n정합성: {match}/{len(SAMPLES)} 일치")
