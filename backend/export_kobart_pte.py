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
USE_XNNPACK = os.getenv("XNN", "0") == "1"
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
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


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
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=encoder_attention_mask,
        )
        return out.logits


def lower(ep):
    if USE_XNNPACK:
        return to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    return to_edge(ep).to_executorch()


STATIC = torch.export.Dim.STATIC
ex = tok("밥 문나 아직 안 무따", return_tensors="pt", return_token_type_ids=False)

# ── encoder ──
enc = Encoder(model).eval()
eseq = Dim("eseq", min=2, max=512)
print("[encoder] export ...")
with torch.no_grad():
    ep = export(enc, (ex["input_ids"], ex["attention_mask"]),
                dynamic_shapes=({0: STATIC, 1: eseq}, {0: STATIC, 1: eseq}))
mb = save_pte(lower(ep), os.path.join(OUT, "encoder.pte"))
print(f"✅ encoder.pte {mb:.1f} MB")

# ── decoder (단일 스텝) ──
with torch.no_grad():
    enc_hidden = enc(ex["input_ids"], ex["attention_mask"])
dec_ids = torch.tensor([[model.config.decoder_start_token_id, 100]], dtype=torch.long)
dec = Decoder(model).eval()
eseq2 = Dim("eseq2", min=2, max=512)
dseq = Dim("dseq", min=1, max=64)
print("[decoder] export ...")
with torch.no_grad():
    dp = export(dec, (dec_ids, enc_hidden, ex["attention_mask"]),
                dynamic_shapes=({0: STATIC, 1: dseq}, {0: STATIC, 1: eseq2, 2: STATIC}, {0: STATIC, 1: eseq2}))
mb = save_pte(lower(dp), os.path.join(OUT, "decoder.pte"))
print(f"✅ decoder.pte {mb:.1f} MB")
print("완료: encoder.pte + decoder.pte")
