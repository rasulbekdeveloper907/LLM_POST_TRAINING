"""SFT v2: Qwen2.5-1.5B base modelini LoRA yoki QLoRA bilan train.jsonl da o'qitadi. Kaggle GPU uchun.

Kaggle'da sft_v2/sft_v2_kaggle.ipynb notebook'i shu skriptni ishga tushiradi:
    python sft_v2/train.py --method lora  --epochs 5 --rank 64   ->  chat_ui/adapters/sft-lora-1.5b/
    python sft_v2/train.py --method qlora --epochs 5 --rank 64   ->  chat_ui/adapters/sft-qlora-1.5b/
    python sft_v2/train.py --method lora --max-steps 3           ->  tezlikni o'lchash uchun qisqa sinov
Keyin:
    python eval/run_eval.py --base Qwen2.5-1.5B --model sft-lora-1.5b

v1 (sft/lora_cpu, sft/qlora_kaggle) dan farqi: model 1.5B, ikkala usul bitta skriptda va bir xil
sozlamalar bilan, shuning uchun LoRA va QLoRA natijalarini to'g'ridan-to'g'ri solishtirish mumkin.
Data hozircha v1 niki (sft/train.jsonl): 0.5B -> 1.5B farqi faqat model hajmidan keladi.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).parent.parent
ADAPTERS = ROOT / "chat_ui" / "adapters"   # chat UI va run_eval.py shu papkadan o'qiydi

MAX_TOKENS = 384   # bundan uzun namuna qirqiladi
IGNORE = -100      # loss hisoblanmaydigan tokenlar belgisi


def load(data_dir, name):
    return [json.loads(line) for line in (data_dir / name).read_text(encoding="utf-8").splitlines()]


def encode(tok, messages):
    """Bitta namuna -> (input_ids, labels). Loss faqat assistant javobiga qo'yiladi.

    Prompt: <|im_start|>system ... <|im_start|>user savol <|im_end|> <|im_start|>assistant
    Javob:  javob matni <|endoftext|>   <- model shu yerda to'xtashni o'rganadi

    Nega <|im_end|> emas: base modelda <|im_end|> o'qitilmagan (batafsil: sft/lora_cpu/train_lora.py).
    """
    prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt)["input_ids"]
    answer_ids = tok(messages[1]["content"])["input_ids"] + [tok.eos_token_id]
    ids = (prompt_ids + answer_ids)[:MAX_TOKENS]
    labels = ([IGNORE] * len(prompt_ids) + answer_ids)[:MAX_TOKENS]
    return ids, labels


def batches(examples, batch_size, pad_id, shuffle):
    """Namunalarni batchlarga yig'adi; har bir batch o'zidagi eng uzun namunagacha to'ldiriladi (padding)."""
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        yield (torch.tensor([ids + [pad_id] * (width - len(ids)) for ids, _ in chunk]),
               torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in chunk]),
               torch.tensor([labels + [IGNORE] * (width - len(labels)) for _, labels in chunk]))


@torch.no_grad()
def evaluate(model, examples, batch_size, pad_id, device):
    """Validation loss: model ko'rmagan namunalardagi o'rtacha xato."""
    model.eval()
    total, n = 0.0, 0
    for ids, mask, labels in batches(examples, batch_size, pad_id, shuffle=False):
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
        total += out.loss.item() * len(ids)
        n += len(ids)
    model.train()
    return total / n


def load_base(method, base_path):
    """LoRA va QLoRA o'rtasidagi yagona farq: base model qanday yuklanadi."""
    if method == "lora":
        # LoRA: base model to'liq aniqlikda (32 bit). 1.5B uchun bu ~6 GB.
        base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32, device_map={"": 0})
        print(f"LoRA: base model to'liq aniqlikda | xotirada {base.get_memory_footprint() / 1e6:.0f} MB")
        # Gradient checkpointing: activationlar saqlanmay, qayta hisoblanadi. Sekinroq, lekin 1.5B
        # 16 GB GPU'ga sig'adi. QLoRA'da buni prepare_model_for_kbit_training o'zi yoqadi.
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        return base

    # QLoRA: base og'irliklar 4-bitga siqib yuklanadi, LoRA adapter esa oddiy aniqlikda o'qitiladi.
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=quant, device_map={"": 0})
    # Tekshiruv: qatlamlar haqiqatan 4-bit bo'lganmi? Bo'lmasa, bu QLoRA emas, oddiy LoRA bo'lib qoladi.
    import bitsandbytes as bnb
    n_4bit = sum(isinstance(m, bnb.nn.Linear4bit) for m in base.modules())
    if n_4bit == 0:
        raise SystemExit("XATO: model 4-bit ko'rinishda yuklanmadi, QLoRA ishlamaydi.")
    print(f"QLoRA: {n_4bit} ta qatlam 4-bit | base model xotirada {base.get_memory_footprint() / 1e6:.0f} MB")
    return prepare_model_for_kbit_training(base)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["lora", "qlora"], required=True)
    p.add_argument("--base", default="Qwen2.5-1.5B", help="loyiha papkasidagi base model papkasi")
    p.add_argument("--data", default="sft", help="train.jsonl va val.jsonl turgan papka")
    p.add_argument("--name", help="adapter papkasi nomi (standart: sft-<method>-1.5b)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU topilmadi. 1.5B model uchun Kaggle GPU kerak (QLoRA esa CPU'da umuman ishlamaydi).")
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda"
    base_path, data_dir = ROOT / args.base, ROOT / args.data
    name = args.name or f"sft-{args.method}-{args.base.split('-')[-1].lower()}"

    tok = AutoTokenizer.from_pretrained(base_path)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = load_base(args.method, base_path)

    # LoRA: base og'irliklar muzlatiladi, attention va MLP qatlamlariga kichik matritsalar qo'shiladi.
    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train = [encode(tok, r["messages"]) for r in load(data_dir, "train.jsonl")]
    val = [encode(tok, r["messages"]) for r in load(data_dir, "val.jsonl")]
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 20)
    print(f"{args.method} | {args.base} | data: {args.data} | train {len(train)}, val {len(val)} | {total_steps} qadam "
          f"({steps_per_epoch} qadam/epoch) | o'rtacha uzunlik {sum(len(i) for i, _ in train) // len(train)} token")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    # Learning rate: avval asta ko'tariladi (warmup), keyin nolgacha chiziqli pasayadi.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    log, step, start, best = [], 0, time.time(), float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        for ids, mask, labels in batches(train, args.batch_size, tok.pad_token_id, shuffle=True):
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch, "train_loss": round(loss.item(), 4)})
            if step <= 3 or step % 10 == 0 or step == total_steps:
                elapsed = time.time() - start
                left = elapsed / step * (total_steps - step)
                print(f"qadam {step}/{total_steps} | epoch {epoch} | loss {loss.item():.3f} | "
                      f"o'tdi {elapsed / 60:.1f} daq | qoldi ~{left / 60:.0f} daq", flush=True)
            if step == total_steps:
                break
        if args.max_steps:
            break
        val_loss = evaluate(model, val, args.batch_size, tok.pad_token_id, device)
        log.append({"step": step, "epoch": epoch, "val_loss": round(val_loss, 4)})
        # Faqat validation loss yaxshilangan epoch saqlanadi: overfitting boshlansa, eng yaxshi adapter qoladi.
        saved = val_loss < best
        if saved:
            best = val_loss
            model.save_pretrained(ADAPTERS / name)
        print(f"=== epoch {epoch} tugadi | validation loss {val_loss:.3f}" + (" | saqlandi" if saved else " | yaxshilanmadi"),
              flush=True)

    print(f"GPU xotira cho'qqisi: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    if args.max_steps:
        print("Sinov tugadi, adapter saqlanmadi.")
        return
    (ADAPTERS / name / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Tayyor: {ADAPTERS / name} | eng yaxshi validation loss {best:.3f} | jami {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()
