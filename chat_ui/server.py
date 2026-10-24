
import os
import threading
from functools import lru_cache
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

HERE = Path(__file__).resolve().parent
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="LLM Post-Training Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

generation_lock = threading.Lock()


@lru_cache(maxsize=1)
def load_model():
    print(f"Loading model: {MODEL_ID} on {DEVICE}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE).eval()

    print("Model loaded successfully", flush=True)
    return tokenizer, model


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = "base"
    messages: list[Message]
    max_new_tokens: int = 100


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def models():
    return [{
        "id": "base",
        "base": "0.5b",
        "group": "Qwen2.5-0.5B",
        "label": "Base Qwen2.5-0.5B",
        "desc": "Pretrained base model",
        "pretrained": True,
        "available": True,
    }]


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not req.messages:
        raise HTTPException(400, "Xabar bo'sh")

    if req.model != "base":
        raise HTTPException(
            400,
            "Hozircha faqat Base Qwen2.5-0.5B modeli mavjud",
        )

    prompt = req.messages[-1].content.strip()

    if not prompt:
        raise HTTPException(400, "Savol bo'sh")

    max_tokens = max(1, min(req.max_new_tokens, 200))

    def generate_stream():
        with generation_lock:
            try:
                tokenizer, model = load_model()

                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    timeout=180,
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                ).to(DEVICE)

                kwargs = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "pad_token_id": tokenizer.pad_token_id,
                    "streamer": streamer,
                }

                worker = threading.Thread(
                    target=model.generate,
                    kwargs=kwargs,
                    daemon=True,
                )
                worker.start()

                yield from streamer

            except Exception as exc:
                print(f"Generation error: {exc!r}", flush=True)
                yield "\n[ERROR] Model javobini yaratishda xatolik yuz berdi."

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain; charset=utf-8",
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8472")),
    )
