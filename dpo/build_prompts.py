"""DPO 1-qadam: promptlar. sft_v3/data dagi savollar + yangi arifmetika -> dpo/data/prompts.jsonl.

Ishga tushirish (loyiha papkasidan):  python dpo/build_prompts.py

Har bir qator: {"id", "task", "prompt", "reference", ...}
  reference — SFT datadagi to'g'ri javob: hamma model javoblari yomon bo'lsa, "chosen" sifatida ishlatiladi.
  arifmetikada "number" (to'g'ri natija), ro'yxatda "items" (so'ralgan bandlar soni) ham bor.
Test savollari (eval/eval_set.json) kirmaydi: SFT data ularsiz yig'ilgan, yangi arifmetika ham tekshiriladi.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "sft"))
import build_sft  # noqa: E402

# Har bir vazifadan nechta prompt olinadi (None — hammasi). Rad etish va "Bilmayman" hammasi olinadi:
# DPO aynan shu xulqni tuzatishi kerak.
QUOTA = {"harmful": None, "cheating": None, "unknown": None, "exam_prep": None, "chat": None, "list": None,
         "factual": None, "explain": 150, "translate": 100, "arithmetic": 60}
N_NEW_ARITHMETIC = 100
rng = random.Random(7)


def main():
    rows = [json.loads(line) for name in ("train.jsonl", "val.jsonl")
            for line in (ROOT / "sft_v3" / "data" / name).read_text(encoding="utf-8").splitlines()]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    prompts = []
    for task, quota in QUOTA.items():
        items = by_task.get(task, [])
        if quota is not None and len(items) > quota:
            items = rng.sample(items, quota)
        for r in items:
            q, a = r["messages"][0]["content"], r["messages"][1]["content"]
            p = {"task": task, "prompt": q, "reference": a}
            if task == "arithmetic":
                p["number"] = int(re.search(r"=\s*(\d+)", a).group(1))
            if task == "list":
                p["items"] = int(re.search(r"\d+", q).group())
            prompts.append(p)

    # Model SFT'da ko'rmagan arifmetika: yodlab olingan javob emas, haqiqiy hisoblash tekshiriladi.
    eval_items = json.loads(build_sft.EVAL_SET.read_text(encoding="utf-8"))
    eval_squeezed = [build_sft.squeeze(i["messages"][0]["content"]) for i in eval_items]
    seen = {build_sft.squeeze(p["prompt"]) for p in prompts}
    build_sft.rng = random.Random(7)   # SFT'dagidan boshqa misollar chiqishi uchun boshqa seed
    new = 0
    while new < N_NEW_ARITHMETIC:
        key, uz_q, uz_a, en_q, en_a = build_sft.arithmetic_example()
        k = build_sft.squeeze(key)
        if k in seen or any(k in e for e in eval_squeezed) or any(k in s for s in seen):
            continue
        seen.add(k)
        q, a = (en_q, en_a) if rng.random() < 0.2 else (uz_q, uz_a)
        prompts.append({"task": "arithmetic", "prompt": q, "reference": a,
                        "number": int(re.search(r"=\s*(\d+)", a).group(1))})
        new += 1

    rng.shuffle(prompts)
    for i, p in enumerate(prompts):
        p["id"] = f"p{i:04d}"
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    (out / "prompts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts),
                                       encoding="utf-8")
    for task, n in Counter(p["task"] for p in prompts).most_common():
        print(f"  {task:<12} {n}")
    print(f"  {'jami':<12} {len(prompts)}  ->  {out / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
