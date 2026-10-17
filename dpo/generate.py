"""DPO 2-qadam: SFT modeldan har bir promptga bir nechta javob olish -> dpo/data/generations.jsonl.

Kaggle'da (GPU):  python dpo/generate.py             ->  dpo/data_v2/generations.jsonl (8 ta javob)
CPU'da sinov:     python dpo/generate.py --base Qwen2.5-0.5B --adapter sft-lora --limit 4

Javoblar sampling bilan olinadi (har safar boshqacha), shunda ular orasidan yaxshisi va yomonini tanlash mumkin.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
ADAPTERS = ROOT / "chat_ui" / "adapters"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen2.5-1.5B")
    p.add_argument("--adapter", default="sft-lora-v3", help="DPO boshlanadigan SFT adapter")
    p.add_argument("--n", type=int, default=8, help="har bir promptga nechta javob (v1: 4)")
    p.add_argument("--out", default="data_v2", help="dpo/ ichidagi natija papkasi (v1: data)")
    p.add_argument("--batch", type=int, default=16, help="bir vaqtda nechta prompt")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    prompts = [json.loads(line) for line in (HERE / "data" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    prompts = prompts[:args.limit]

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(ROOT / args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"   # batch generatsiyada javob hamma qatorda bir joydan boshlanishi uchun
    model = AutoModelForCausalLM.from_pretrained(ROOT / args.base, dtype=torch.float16 if cuda else torch.float32)
    # Adapter base og'irliklarga qo'shib yuboriladi (merge): generatsiya tezroq ishlaydi.
    model = PeftModel.from_pretrained(model, ADAPTERS / args.adapter).merge_and_unload().eval()
    if cuda:
        model.cuda()
    stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

    out_path = HERE / args.out / "generations.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    start, done = time.time(), 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": c["prompt"]}], tokenize=False,
                                             add_generation_prompt=True) for c in chunk]
            inp = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, do_sample=True, temperature=args.temperature, top_p=0.95,
                                     num_return_sequences=args.n, max_new_tokens=args.max_new_tokens,
                                     repetition_penalty=1.1, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
            new = out[:, inp["input_ids"].shape[1]:]
            for j, c in enumerate(chunk):
                answers = []
                for k in range(args.n):
                    ids = new[j * args.n + k].tolist()
                    # finished=False: model to'xtamay, token chegarasiga yetib qolgan (bu ham yomon javob belgisi).
                    finished = any(t in stop_ids for t in ids)
                    answers.append({"text": tok.decode(ids, skip_special_tokens=True).strip(), "finished": finished})
                f.write(json.dumps({**c, "answers": answers}, ensure_ascii=False) + "\n")
            done += len(chunk)
            elapsed = time.time() - start
            print(f"{done}/{len(prompts)} prompt | o'tdi {elapsed / 60:.1f} daq | "
                  f"qoldi ~{elapsed / done * (len(prompts) - done) / 60:.0f} daq", flush=True)
    print("Saqlandi:", out_path)


if __name__ == "__main__":
    main()
