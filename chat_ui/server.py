"""Lokal chat UI serveri: base Qwen2.5-0.5B va 1.5B modellari hamda ularning LoRA adapterlari.

Ishga tushirish:  python server.py   ->   http://127.0.0.1:8472
0.5B ishga tushganda yuklanadi; 1.5B esa uning modellaridan biri birinchi marta tanlanganda (~6 GB xotira).
"""
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

HERE = Path(__file__).parent
BASES = {"0.5b": HERE.parent / "Qwen2.5-0.5B", "1.5b": HERE.parent / "Qwen2.5-1.5B"}
ADAPTERS = HERE / "adapters"   # adapters/<id>/adapter_config.json bo'lsa, model ro'yxatda yonadi

# Dropdown tartibi. adapter=None — o'qitilmagan base model; qolganlari base + adapters/<id> dagi LoRA adapter.
# Ball — 68 ta test savolidagi avtomatik tekshiruv natijasi (results/per_task_scores.md).
MODELS = [
    {"id": "base",           "base": "0.5b", "group": "v1 · Qwen2.5-0.5B", "label": "Base 0.5B (Pretrained)",
     "desc": "Faqat keyingi so‘zni bashorat qiladi · 5/68"},
    {"id": "sft-lora",       "base": "0.5b", "group": "v1 · Qwen2.5-0.5B", "label": "v1 · SFT + LoRA",
     "desc": "CPU'da o‘qitilgan, v1 data · 32/68"},
    {"id": "sft-qlora",      "base": "0.5b", "group": "v1 · Qwen2.5-0.5B", "label": "v1 · SFT + QLoRA",
     "desc": "4-bit base ustida LoRA, v1 data · 32/68"},
    {"id": "base-1.5b",      "base": "1.5b", "group": "v2 · Qwen2.5-1.5B, v1 data", "label": "Base 1.5B (Pretrained)",
     "desc": "Faqat keyingi so‘zni bashorat qiladi · 15/68"},
    {"id": "sft-lora-1.5b",  "base": "1.5b", "group": "v2 · Qwen2.5-1.5B, v1 data", "label": "v2 · SFT + LoRA",
     "desc": "1.5B, v1 data · 44/68"},
    {"id": "sft-qlora-1.5b", "base": "1.5b", "group": "v2 · Qwen2.5-1.5B, v1 data", "label": "v2 · SFT + QLoRA",
     "desc": "1.5B, v1 data · 42/68"},
    {"id": "sft-lora-v3",    "base": "1.5b", "group": "v3 · Qwen2.5-1.5B, yangi data", "label": "v3 · SFT + LoRA",
     "desc": "1.5B, v3 data · 41/68"},
    {"id": "sft-qlora-v3",   "base": "1.5b", "group": "v3 · Qwen2.5-1.5B, yangi data", "label": "v3 · SFT + QLoRA",
     "desc": "1.5B, v3 data · 41/68"},
    {"id": "sft-lora-rlhf",  "base": "1.5b", "group": "Preference optimisation", "label": "SFT + LoRA + RLHF",
     "desc": "LoRA SFT, so‘ng RLHF"},
    {"id": "sft-lora-dpo",   "base": "1.5b", "group": "Preference optimisation", "label": "v3 · SFT + LoRA + DPO",
     "desc": "v3 LoRA ustida DPO"},
    {"id": "sft-lora-dpo-v2", "base": "1.5b", "group": "Preference optimisation", "label": "v3 · SFT + LoRA + DPO v2",
     "desc": "Toza rad javoblari, uzunlik muvozanati, SFT qismi"},
    {"id": "sft-qlora-rlhf", "base": "1.5b", "group": "Preference optimisation", "label": "SFT + QLoRA + RLHF",
     "desc": "QLoRA SFT, so‘ng RLHF"},
    {"id": "sft-qlora-dpo",  "base": "1.5b", "group": "Preference optimisation", "label": "SFT + QLoRA + DPO",
     "desc": "QLoRA SFT, so‘ng DPO"},
]
BY_ID = {m["id"]: m for m in MODELS}
PRETRAINED = {"base", "base-1.5b"}

cuda = torch.cuda.is_available()
dtype = torch.float16 if cuda else torch.float32   # CPU'da float16 ishlamaydi


class Base:
    """Bitta base model, uning tokenizeri va ustiga yuklangan adapterlar (birinchi so'rovda yaratiladi)."""

    def __init__(self, path):
        print("Model yuklanmoqda:", path)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).eval()
        if cuda:
            self.model.cuda()
        self.peft = None
        print("Tayyor:", path.name, "| qurilma:", self.model.device)

    def use_adapter(self, model_id):
        from peft import PeftModel
        name = model_id.replace(".", "_")   # peft adapter nomida nuqta bo'lmasligi kerak ("sft-lora-1.5b")
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.model, ADAPTERS / model_id, adapter_name=name).eval()
        elif name not in self.peft.peft_config:
            self.peft.load_adapter(ADAPTERS / model_id, adapter_name=name)
        self.peft.set_adapter(name)
        return self.peft


bases = {"0.5b": Base(BASES["0.5b"])}
gen_lock = threading.Lock()  # bir vaqtda bitta generatsiya (va bitta model yuklash)


def get_base(key):
    if key not in bases:
        bases[key] = Base(BASES[key])
    return bases[key]


def is_available(model_id):
    m = BY_ID[model_id]
    if not (BASES[m["base"]] / "model.safetensors").exists():   # 1.5B hali yuklab olinmagan bo'lishi mumkin
        return False
    return model_id in PRETRAINED or (ADAPTERS / model_id / "adapter_config.json").exists()


class StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = "base"
    messages: list[Message]
    max_new_tokens: int = 100


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/models")
def models():
    return [{**m, "pretrained": m["id"] in PRETRAINED, "available": is_available(m["id"])} for m in MODELS]


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model not in BY_ID or not req.messages:
        raise HTTPException(400, "Noto‘g‘ri so‘rov")
    if not is_available(req.model):
        raise HTTPException(409, "Bu model hali o‘qitilmagan")
    pretrained = req.model in PRETRAINED
    base_key = BY_ID[req.model]["base"]
    tok = (bases.get(base_key) or bases["0.5b"]).tok   # ikkala modelda tokenizer bir xil

    if pretrained:
        # Base model chat formatini bilmaydi: 1-darsdagidek xom matn, faqat oxirgi xabar.
        prompt = req.messages[-1].content
    else:
        # Faqat oxirgi savol: modellar bitta savol -> bitta javob ko'rinishida o'qitilgan. Suhbat tarixi berilsa,
        # ular oldingi javobni ko'chirib qo'yadi (masalan, imtihon savolini rad etish o'rniga).
        prompt = tok.apply_chat_template([req.messages[-1].model_dump()], tokenize=False, add_generation_prompt=True)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    stop = threading.Event()
    stop_ids = [tok.eos_token_id]
    if not pretrained:
        stop_ids.append(tok.convert_tokens_to_ids("<|im_end|>"))   # o'qitilgan model javobni shu token bilan tugatadi
    kwargs = dict(max_new_tokens=max(1, min(req.max_new_tokens, 400)), do_sample=False,
                  repetition_penalty=1.1, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids, streamer=streamer,
                  stopping_criteria=StoppingCriteriaList([StopOnEvent(stop)]))

    def run():
        with gen_lock:
            try:
                b = get_base(base_key)   # 1.5B birinchi marta shu yerda yuklanadi
                inp = b.tok(prompt, return_tensors="pt").to(b.model.device)
                with torch.no_grad():
                    if not pretrained:
                        b.use_adapter(req.model).generate(**inp, **kwargs)
                    elif b.peft is not None:
                        with b.peft.disable_adapter():
                            b.peft.generate(**inp, **kwargs)
                    else:
                        b.model.generate(**inp, **kwargs)
            except Exception as e:
                print("Generatsiya xatosi:", repr(e))
                streamer.end()

    threading.Thread(target=run, daemon=True).start()

    def stream():
        try:
            yield from streamer
        finally:
            stop.set()   # brauzer "to'xtatish"ni bossa, generatsiya ham to'xtaydi

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8472)
