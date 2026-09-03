"""KoBART(변환기)를 ExecuTorch .pte로 export — react-native-executorch 온디바이스용.
1단계: encoder를 .pte로. (decoder/with-past는 후속)
실행: C:/et/Scripts/python.exe backend/export_kobart_pte.py
"""
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import torch
from torch.export import export, Dim
from transformers import BartForConditionalGeneration, AutoTokenizer
from executorch.exir import to_edge_transform_and_lower, to_edge
QUANT = os.getenv("QUANT", "0") == "1"          # int8 양자화(XNNPACK 델리게이트로 실행)
USE_XNNPACK = QUANT or os.getenv("XNN", "0") == "1"
if USE_XNNPACK:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

MODEL = "backend/models/kobart-dialect"
OUT = "backend/models/kobart-pte"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = BartForConditionalGeneration.from_pretrained(MODEL).eval()


class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc = m.get_encoder()

    def forward(self, input_ids, attention_mask):
        # int32 입력을 받아 내부에서 long으로(JS에서 Int32Array 사용 → BigInt64Array 회피)
        return self.enc(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state


def save_pte(prog, path):
    if hasattr(prog, "write_to_file"):
        with open(path, "wb") as f:
            prog.write_to_file(f)
    else:
        with open(path, "wb") as f:
            f.write(prog.buffer)
    return os.path.getsize(path) / 1e6


class Decoder(torch.nn.Module):
    """단일 스텝(KV캐시 없음): (decoder_input_ids, enc_hidden, enc_mask) → logits."""
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
        out = self.model(
            decoder_input_ids=decoder_input_ids.long(),
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=encoder_attention_mask.long(),
        )
        return out.logits


def lower(ep):
    if USE_XNNPACK:
        return to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    return to_edge(ep).to_executorch()


def _quantize(module, example, dynamic_shapes):
    """PT2E int8 양자화(XNNPACKQuantizer, per-channel 대칭).
    주의(미완): 전역 config가 정수 임베딩 입력까지 양자화하려다 실패
    (embedding indices에 float). 임베딩/토큰입력을 제외하는 quantizer 주석 설정이 필요 — 후속 과제.
    현재 동작 경로는 QUANT 미사용(XNNPACK fp32 또는 portable)."""
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    captured = export(module, example, dynamic_shapes=dynamic_shapes).module()
    prepared = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        prepared(*example)  # 보정(calibration)
    return convert_pt2e(prepared)


def export_and_lower(module, example, dynamic_shapes):
    m = _quantize(module, example, dynamic_shapes) if QUANT else module
    with torch.no_grad():
        ep = export(m, example, dynamic_shapes=dynamic_shapes)
    return lower(ep)


STATIC = torch.export.Dim.STATIC
ex = tok("밥 문나 아직 안 무따", return_tensors="pt", return_token_type_ids=False)
ii = ex["input_ids"].int()        # int32 (JS Int32Array)
am = ex["attention_mask"].int()   # int32

print(f"모드: {'int8 양자화(XNNPACK)' if QUANT else ('XNNPACK fp32' if USE_XNNPACK else 'portable fp32')}")

# ── encoder ──
enc = Encoder(model).eval()
eseq = Dim("eseq", min=2, max=512)
print("[encoder] export ...")
mb = save_pte(export_and_lower(enc, (ii, am),
              ({0: STATIC, 1: eseq}, {0: STATIC, 1: eseq})), os.path.join(OUT, "encoder.pte"))
print(f"✅ encoder.pte {mb:.1f} MB")

# ── decoder (단일 스텝) ──
with torch.no_grad():
    enc_hidden = enc(ii, am)
dec_ids = torch.tensor([[model.config.decoder_start_token_id, 100]], dtype=torch.int32)
dec = Decoder(model).eval()
eseq2 = Dim("eseq2", min=2, max=512)
dseq = Dim("dseq", min=1, max=64)
print("[decoder] export ...")
mb = save_pte(export_and_lower(dec, (dec_ids, enc_hidden, am),
              ({0: STATIC, 1: dseq}, {0: STATIC, 1: eseq2, 2: STATIC}, {0: STATIC, 1: eseq2})),
              os.path.join(OUT, "decoder.pte"))
print(f"✅ decoder.pte {mb:.1f} MB")
print("완료: encoder.pte + decoder.pte")
