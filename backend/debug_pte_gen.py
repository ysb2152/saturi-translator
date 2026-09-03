"""빈 출력 디버깅: .pte(encoder/decoder)로 greedy 생성을 Python에서 재현하고 PyTorch와 비교.
실행: C:/et/Scripts/python.exe backend/debug_pte_gen.py
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer
from executorch.runtime import Runtime

MODEL = "backend/models/kobart-dialect"
PTE = "backend/models/kobart-pte"
tok = AutoTokenizer.from_pretrained(MODEL)
pt = BartForConditionalGeneration.from_pretrained(MODEL).eval()

text = "밥 문나 아직 안 무따"
ein = tok(text, return_tensors="pt", return_token_type_ids=False)
ids, mask = ein["input_ids"], ein["attention_mask"]
print("input_ids:", ids.tolist()[0])
print("tokens   :", tok.convert_ids_to_tokens(ids[0].tolist()))
print("decoder_start:", pt.config.decoder_start_token_id, "eos:", pt.config.eos_token_id, "bos:", pt.config.bos_token_id, "pad:", pt.config.pad_token_id)

with torch.no_grad():
    g = pt.generate(**ein, max_new_tokens=64, num_beams=1)
print("PT greedy ids   :", g[0].tolist())
print("PT greedy decode:", tok.decode(g[0], skip_special_tokens=True))

rt = Runtime.get()
enc_m = rt.load_program(f"{PTE}/encoder.pte").load_method("forward")
dec_m = rt.load_program(f"{PTE}/decoder.pte").load_method("forward")

enc_hidden = enc_m.execute([ids, mask])[0]
print("\nenc_hidden:", tuple(enc_hidden.shape), enc_hidden.dtype, "mean", float(enc_hidden.float().mean()))

# PyTorch encoder 비교(정합성)
with torch.no_grad():
    pt_hidden = pt.get_encoder()(input_ids=ids, attention_mask=mask).last_hidden_state
print("pt_hidden vs pte max diff:", float((pt_hidden - enc_hidden).abs().max()))

DEC_START, EOS = pt.config.decoder_start_token_id, pt.config.eos_token_id
dec = [DEC_START]
for step in range(20):
    dt = torch.tensor([dec], dtype=torch.long)
    logits = dec_m.execute([dt, enc_hidden, mask])[0]  # [1, len, vocab]
    last = logits[0, -1]
    nxt = int(last.argmax())
    top5 = torch.topk(last, 5)
    if step < 5:
        print(f"step{step} dec={dec} -> next={nxt}({tok.convert_ids_to_tokens([nxt])[0]}) top5={[(int(i),round(float(v),2)) for v,i in zip(top5.values,top5.indices)]}")
    if nxt == EOS:
        print("EOS at step", step); break
    dec.append(nxt)
print("ET greedy ids   :", dec)
print("ET greedy decode:", tok.decode(dec[1:], skip_special_tokens=True))
