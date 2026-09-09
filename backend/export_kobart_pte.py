"""KoBART 변환기를 ExecuTorch .pte로 뽑는다(react-native-executorch 온디바이스용).
encoder / decoder를 각각 .pte로 export한다.
실행: C:/et/Scripts/python.exe backend/export_kobart_pte.py
"""
import os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import torch
from torch.export import export, Dim
from transformers import BartForConditionalGeneration, AutoTokenizer
from executorch.exir import to_edge_transform_and_lower, to_edge
QUANT = os.getenv("QUANT", "0") == "1"          # int8 weight-only 양자화(torchao, portable)
USE_XNNPACK = os.getenv("XNN", "0") == "1"
if USE_XNNPACK:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

MODEL = os.getenv("MODEL", "backend/models/kobart-dialect")
OUT = os.getenv("OUT", "backend/models/kobart-pte")
os.makedirs(OUT, exist_ok=True)
print(f"MODEL={MODEL}  OUT={OUT}")

tok = AutoTokenizer.from_pretrained(MODEL)
model = BartForConditionalGeneration.from_pretrained(MODEL).eval()


class Encoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc = m.get_encoder()

    def forward(self, input_ids, attention_mask):
        # int32로 받아 내부에서 long으로 캐스팅(JS에서 Int32Array 쓰려는 것 — BigInt64Array 피하려고)
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
    """torchao weight-only int8. Linear 가중치만 int8로 바꾸고 activation·입력은 그대로 둔다.
    처음엔 PT2E(XNNPACKQuantizer)를 썼는데 정수 임베딩 입력까지 양자화하려다 실패해서 weight-only로 돌아섰다.
    이러면 portable로 export돼서 flatc/XNNPACK도 필요 없다. 임베딩은 fp32 유지."""
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    quantize_(module, Int8WeightOnlyConfig())
    return module


def export_and_lower(module, example, dynamic_shapes):
    m = _quantize(module, example, dynamic_shapes) if QUANT else module
    with torch.no_grad():
        ep = export(m, example, dynamic_shapes=dynamic_shapes)
    return lower(ep)


STATIC = torch.export.Dim.STATIC
ex = tok("밥 문나 아직 안 무따", return_tensors="pt", return_token_type_ids=False)
ii = ex["input_ids"].int()        # int32 (JS Int32Array)
am = ex["attention_mask"].int()   # int32

print(f"모드: {'int8 weight-only(portable)' if QUANT else ('XNNPACK fp32' if USE_XNNPACK else 'portable fp32')}")

# ── encoder ──
enc = Encoder(model).eval()
eseq = Dim("eseq", min=2, max=512)
print("[encoder] export ...")
mb = save_pte(export_and_lower(enc, (ii, am),
              ({0: STATIC, 1: eseq}, {0: STATIC, 1: eseq})), os.path.join(OUT, "encoder.pte"))
print(f"encoder.pte {mb:.1f} MB")

# ── decoder (단일 스텝) ── 앞에서 quantize_가 인코더를 in-place로 바꿔놔서 fp32 모델을 새로 로드한다
model2 = BartForConditionalGeneration.from_pretrained(MODEL).eval()
with torch.no_grad():
    enc_hidden = model2.get_encoder()(input_ids=ii.long(), attention_mask=am.long()).last_hidden_state
dec_ids = torch.tensor([[model2.config.decoder_start_token_id, 100]], dtype=torch.int32)
dec = Decoder(model2).eval()
eseq2 = Dim("eseq2", min=2, max=512)
dseq = Dim("dseq", min=1, max=64)
print("[decoder] export ...")
mb = save_pte(export_and_lower(dec, (dec_ids, enc_hidden, am),
              ({0: STATIC, 1: dseq}, {0: STATIC, 1: eseq2, 2: STATIC}, {0: STATIC, 1: eseq2})),
              os.path.join(OUT, "decoder.pte"))
print(f"decoder.pte {mb:.1f} MB")
print("완료: encoder.pte + decoder.pte")
