"""SFT + LoRA: Qwen2.5-0.5B base modelini train.jsonl da o'qitadi. CPU'da ishlaydi.

Ishga tushirish (loyiha papkasidan):
    python sft/lora_cpu/train_lora.py                 ->  chat_ui/adapters/sft-lora/
    python sft/lora_cpu/train_lora.py --max-steps 3   ->  tezlikni o'lchash uchun qisqa sinov
Keyin:
    python eval/run_eval.py --model sft-lora
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT = Path(__file__).parent.parent          # sft/ papkasi: train.jsonl va val.jsonl shu yerda
ROOT = SFT.parent
BASE_PATH = ROOT / "Qwen2.5-0.5B"
ADAPTERS = ROOT / "chat_ui" / "adapters"   # chat UI va run_eval.py shu papkadan o'qiydi

MAX_TOKENS = 384   # bundan uzun namuna qirqiladi
IGNORE = -100      # loss hisoblanmaydigan tokenlar belgisi


def load(name):
    return [json.loads(line) for line in (SFT / name).read_text(encoding="utf-8").splitlines()]


def encode(tok, messages):
    """Bitta namuna -> (input_ids, labels). Loss faqat assistant javobiga qo'yiladi.

    Prompt: <|im_start|>system ... <|im_start|>user savol <|im_end|> <|im_start|>assistant
    Javob:  javob matni <|endoftext|>   <- model shu yerda to'xtashni o'rganadi

    Nega <|im_end|> emas: base modelda <|im_end|> o'qitilmagan, uning embeddingi 5000 ga yaqin boshqa
    ishlatilmagan token bilan bir xil. LoRA embeddinglarni o'zgartirmaydi, shuning uchun model uni
    boshqalardan ajrata olmaydi va javob oxirida tasodifiy token chiqaradi. <|endoftext|> esa
    pretraining'da o'qitilgan va yagona.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sft-lora", help="adapter papkasi nomi")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name

    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    tok.pad_token = tok.pad_token or tok.eos_token
    # Base model to'liq aniqlikda (32 bit) yuklanadi.
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=torch.float32).to(device)
    print(f"LoRA: base model to'liq aniqlikda | xotirada {base.get_memory_footprint() / 1e6:.0f} MB")

    # LoRA: base og'irliklar muzlatiladi, attention va MLP qatlamlariga kichik matritsalar qo'shiladi.
    model = get_peft_model(base, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train = [encode(tok, r["messages"]) for r in load("train.jsonl")]
    val = [encode(tok, r["messages"]) for r in load("val.jsonl")]
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 20)
    print(f"Qurilma: {device} | train {len(train)}, val {len(val)} | {total_steps} qadam "
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

    if args.max_steps:
        print("Sinov tugadi, adapter saqlanmadi.")
        return
    (ADAPTERS / name / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Tayyor: {ADAPTERS / name} | eng yaxshi validation loss {best:.3f} | jami {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()
