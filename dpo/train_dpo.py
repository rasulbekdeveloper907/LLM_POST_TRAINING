"""DPO 4-qadam: SFT adapterni chosen/rejected juftliklarida DPO bilan o'qitish -> chat_ui/adapters/sft-lora-dpo/.

Kaggle'da (GPU):  python dpo/train_dpo.py                 ->  v2: dpo/data_v2, chat_ui/adapters/sft-lora-dpo-v2/
CPU'da sinov:     python dpo/train_dpo.py --base Qwen2.5-0.5B --sft-adapter sft-lora --max-steps 2

Qanday ishlaydi:
  - Policy: base model + SFT v3 LoRA adapter. DPO shu adapterni o'qitishni davom ettiradi.
  - Reference: o'qitish boshlanishidagi o'sha model (ya'ni SFT model). Uning javob ehtimolliklari o'qitishdan
    oldin bir marta hisoblab qo'yiladi, shuning uchun xotirada faqat bitta model turadi.
  - Loss: -log sigmoid(beta * [(policy - ref) chosen uchun  -  (policy - ref) rejected uchun])
    Model chosen javob ehtimolligini oshiradi, rejected nikini kamaytiradi — reference'ga nisbatan.
  - v2: loss'ga chosen javob uchun SFT loss ham qo'shiladi (--sft-weight). v1 da chosen ehtimolligi ham pasayib
    ketgan edi (chosen reward -0.7): model ikkala javobni ham pastga itargan. SFT qismi chosen'ni ushlab turadi.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).parent
ROOT = HERE.parent
ADAPTERS = ROOT / "chat_ui" / "adapters"

MAX_TOKENS = 384   # bundan uzun namuna qirqiladi
IGNORE = -100      # loss hisoblanmaydigan tokenlar belgisi


def load(folder, name):
    return [json.loads(line) for line in (HERE / folder / name).read_text(encoding="utf-8").splitlines()]


def encode(tok, prompt, answer):
    """SFT'dagi bilan bir xil format: prompt chat template'da, javob oxirida <|endoftext|>."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(text)["input_ids"]
    answer_ids = tok(answer)["input_ids"] + [tok.eos_token_id]
    return (prompt_ids + answer_ids)[:MAX_TOKENS], ([IGNORE] * len(prompt_ids) + answer_ids)[:MAX_TOKENS]


def answer_logprob(model, seqs, pad_id, device):
    """Har bir ketma-ketlik uchun javob tokenlari log-ehtimolliklarining yig'indisi: log p(javob | prompt)."""
    width = max(len(ids) for ids, _ in seqs)
    ids = torch.tensor([i + [pad_id] * (width - len(i)) for i, _ in seqs], device=device)
    mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i, _ in seqs], device=device)
    labels = torch.tensor([l + [IGNORE] * (width - len(l)) for _, l in seqs], device=device)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    # cross_entropy = -log p(token); prompt va padding tokenlarida 0 bo'ladi.
    token_logp = -F.cross_entropy(logits.transpose(1, 2), labels[:, 1:], ignore_index=IGNORE, reduction="none")
    return token_logp.sum(-1), (labels[:, 1:] != IGNORE).sum(-1)


def pair_logprobs(model, pairs, pad_id, device):
    """Bir batch juftlik: chosen va rejected bitta forward'da -> (chosen_logp, rejected_logp, chosen_tokenlar_soni)."""
    lp, n = answer_logprob(model, [c for c, _ in pairs] + [r for _, r in pairs], pad_id, device)
    return lp[:len(pairs)], lp[len(pairs):], n[:len(pairs)]


def dpo_stats(beta, pc, pr, rc, rr, nc, sft_weight):
    """Implicit reward = beta * log(policy / reference). Loss chosen reward rejected'dan katta bo'lishini talab qiladi."""
    chosen_r, rejected_r = beta * (pc - rc), beta * (pr - rr)
    dpo = -F.logsigmoid(chosen_r - rejected_r).mean()
    sft = (-pc / nc).mean()   # chosen javobning har bir tokeniga o'rtacha -log p (oddiy SFT loss)
    loss = dpo + sft_weight * sft
    return loss, {"loss": dpo.item(), "sft": sft.item(), "acc": (chosen_r > rejected_r).float().mean().item(),
                  "margin": (chosen_r - rejected_r).mean().item(),
                  "chosen_r": chosen_r.mean().item(), "rejected_r": rejected_r.mean().item()}


def load_model(method, base_path, adapter_path, device):
    if method == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=quant, device_map={"": 0})
        base = prepare_model_for_kbit_training(base)
    else:
        base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32).to(device)
        base.gradient_checkpointing_enable()   # chosen + rejected: SFT'dagidan ikki barobar ko'p ketma-ketlik
        base.enable_input_require_grads()
    # is_trainable=True: SFT adapter muzlatilmaydi, DPO uni o'qitishni davom ettiradi.
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    # Dropout o'chiriladi: aks holda o'qitishning birinchi qadamidayoq policy reference'dan tasodifan farq qiladi.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return model


@torch.no_grad()
def reference_logprobs(model, data, batch_size, pad_id, device):
    model.eval()
    out = [pair_logprobs(model, data[i:i + batch_size], pad_id, device) for i in range(0, len(data), batch_size)]
    model.train()
    return torch.cat([c for c, _, _ in out]), torch.cat([r for _, r, _ in out])


@torch.no_grad()
def evaluate(model, data, ref, batch_size, beta, sft_weight, pad_id, device):
    model.eval()
    totals, n = {}, 0
    for i in range(0, len(data), batch_size):
        pc, pr, nc = pair_logprobs(model, data[i:i + batch_size], pad_id, device)
        _, s = dpo_stats(beta, pc, pr, ref[0][i:i + batch_size], ref[1][i:i + batch_size], nc, sft_weight)
        k = len(pc)
        totals = {key: totals.get(key, 0) + v * k for key, v in s.items()}
        n += k
    model.train()
    return {key: v / n for key, v in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["lora", "qlora"], default="lora")
    p.add_argument("--base", default="Qwen2.5-1.5B")
    p.add_argument("--sft-adapter", default="sft-lora-v3", help="DPO boshlanadigan SFT adapter")
    p.add_argument("--data", default="data_v2", help="dpo/ ichidagi juftliklar papkasi (v1: data)")
    p.add_argument("--name", default="sft-lora-dpo-v2", help="natija adapter papkasi nomi (v1: sft-lora-dpo)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4, help="juftliklar soni (ketma-ketliklar 2 barobar)")
    p.add_argument("--lr", type=float, default=5e-6, help="SFT'dagidan (2e-4) ancha kichik (v1: 2e-5)")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft-weight", type=float, default=1.0, help="chosen uchun SFT loss og'irligi (v1: 0)")
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.method == "qlora" and device != "cuda":
        raise SystemExit("QLoRA uchun NVIDIA GPU kerak. Kaggle'da ishga tushiring.")

    tok = AutoTokenizer.from_pretrained(ROOT / args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = load_model(args.method, ROOT / args.base, ADAPTERS / args.sft_adapter, device)
    model.print_trainable_parameters()

    train = [(encode(tok, r["prompt"], r["chosen"]), encode(tok, r["prompt"], r["rejected"])) for r in load(args.data, "train.jsonl")]
    val = [(encode(tok, r["prompt"], r["chosen"]), encode(tok, r["prompt"], r["rejected"])) for r in load(args.data, "val.jsonl")]
    if args.max_steps:
        train, val = train[:args.max_steps * args.batch_size], val[:args.batch_size]
    pad = tok.pad_token_id

    print(f"Reference ehtimolliklari hisoblanmoqda: {len(train)} train, {len(val)} val juftlik...", flush=True)
    order = list(range(len(train)))
    ref_train = reference_logprobs(model, train, args.batch_size, pad, device)
    ref_val = reference_logprobs(model, val, args.batch_size, pad, device)

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 10)
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    # O'qitishdan oldin policy == reference, shuning uchun loss = log 2 = 0.693, acc = 0.
    s0 = evaluate(model, val, ref_val, args.batch_size, args.beta, args.sft_weight, pad, device)
    print(f"Boshlanishida: val loss {s0['loss']:.3f} (kutilgan 0.693)", flush=True)
    log, step, start, best = [{"step": 0, "val": s0}], 0, time.time(), float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            pc, pr, nc = pair_logprobs(model, [train[j] for j in idx], pad, device)
            loss, s = dpo_stats(args.beta, pc, pr, ref_train[0][idx], ref_train[1][idx], nc, args.sft_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            log.append({"step": step, "epoch": epoch, **{k: round(v, 4) for k, v in s.items()}})
            if step <= 3 or step % 10 == 0 or step == total_steps:
                elapsed = time.time() - start
                print(f"qadam {step}/{total_steps} | loss {s['loss']:.3f} | acc {s['acc']:.2f} | margin {s['margin']:.3f} | "
                      f"chosen {s['chosen_r']:+.3f} rejected {s['rejected_r']:+.3f} | sft {s['sft']:.3f} | "
                      f"qoldi ~{elapsed / step * (total_steps - step) / 60:.0f} daq", flush=True)
            if step == total_steps:
                break
        sv = evaluate(model, val, ref_val, args.batch_size, args.beta, args.sft_weight, pad, device)
        log.append({"step": step, "epoch": epoch, "val": sv})
        # Eng yaxshi epoch DPO + SFT loss bo'yicha tanlanadi: v1 da faqat DPO loss yaxshilangan, xulq esa buzilgan.
        total = sv["loss"] + args.sft_weight * sv["sft"]
        saved = total < best and not args.max_steps
        if saved:
            best = total
            model.save_pretrained(ADAPTERS / args.name)
        print(f"=== epoch {epoch} | val loss {sv['loss']:.3f} | val acc {sv['acc']:.2f} | val margin {sv['margin']:.3f} "
              f"| chosen {sv['chosen_r']:+.3f}"
              + (" | saqlandi" if saved else ""), flush=True)
        if step == total_steps:
            break

    if args.max_steps:
        print("Sinov tugadi, adapter saqlanmadi.")
        return
    (ADAPTERS / args.name / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Tayyor: {ADAPTERS / args.name} | eng yaxshi val loss (DPO + SFT) {best:.3f} | jami {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()
