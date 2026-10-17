"""Ochiq datasetlardan SFT namunalari: yuklab oladi, filtrlaydi va seeds.txt formatida yozadi.

Ishga tushirish:  python fetch_external.py   ->  external_oasst_en.txt, external_alpaca_uz.txt
Keyin:            python build_sft.py        ->  train.jsonl, val.jsonl

Manbalar:
  OpenAssistant/oasst1         (ingliz tili, Apache-2.0)  odamlar yozgan savol va javoblar
  bizb0630/alpaca-cleaned_uz   (o'zbek tili, CC-BY-4.0)   Alpaca datasetining mashina tarjimasi
"""
import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
RAW = HERE / "external" / "raw"   # yuklab olingan xom fayllar (gitga kirmaydi)

MAX_PER_SOURCE = 150              # har bir manbadan ko'pi bilan shuncha namuna
Q_CHARS = (10, 200)               # savol uzunligi chegarasi
A_CHARS = (30, 400)               # javob uzunligi chegarasi: spec'dagi "1–5 gap" ga mos
MAX_SENTENCES = 5
rng = random.Random(42)

# Kurs mavzusiga (va unga kerakli asoslarga) oid savollargina olinadi (MODEL_SPEC.md, 1-bo'lim).
TOPIC_EN = r"machine learning|deep learning|neural net|artificial intelligence|\bAI\b|language model|\bLLM|chatbot|" \
           r"\bGPT|transformer|training data|dataset|algorithm|\bpython\b|programming|data scien|" \
           r"supervised|regression|classification|gradient|fine-tun|embedding|overfit|\bGPU\b|" \
           r"statistic|probabilit|matrix|matrices|vector|linear algebra|derivative|database|\bSQL\b|\bAPI\b"
TOPIC_UZ = r"mashinani o'rganish|mashinali o'qitish|mashina o'rganish|mashina o'qitish|chuqur o'rganish|" \
           r"chuqur o'qitish|neyron|sun'iy intellekt|sun'iy aql|\bAI\b|til modeli|chatbot|\bGPT|algoritm|" \
           r"\bpython\b|dasturlash|ma'lumotlar to'plami|ma'lumotlar fani|regressiya|klassifikatsiya|gradient|" \
           r"\bGPU\b|statistika|ehtimollik|matritsa|vektor|hosila|ma'lumotlar bazasi|\bSQL\b|\bAPI\b"

# Faqat bilim so'ragan savollar olinadi; "yozib ber", "sarlavha o'ylab top" kabi ijodiy topshiriqlar olinmaydi.
ASKS_EN = r"^(what|how|why|when|which|who|explain|describe|define|can you explain|is |are |does |do )\b"
ASKS_UZ = r"\bnima|qanday|qaysi|nega\b|nechta|tushuntir|ta'rifla|izohla|farq|sanab|nomlang|ayting"
CREATIVE = r"\b(write|generate|create|compose|rewrite|pretend|imagine|recommend|suggest|you are)\b|" \
           r"yarating|yarat\b|yozing|yozib|taklif qil|sarlavha|tvit|she'r|hikoya|shior|qayta yoz|tasavvur qil|" \
           r"\b(latest|newest|current|do you think|do you know|companies)\b|eng yangi|^[\"“”«']"   # eskiradigan va fikr so'ragan savollar

# Sifat filtri: havola, kod, markdown va modelning o'zi haqidagi gaplar spec'dagi uslubga to'g'ri kelmaydi.
BAD_ANSWER = r"```|`|https?://|www\.|\*\*|^#|^\s*[-*>] |^ {4}|" \
             r"\b(I am|I'm|I was|I cannot|I can't|I don't|my training|as an AI|as a language model)\b|" \
             r"LAION|Open.?Assistant|sun'iy intellekt sifatida|til modeli sifatida|AI sifatida|men sun'iy"
CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def plain(text):
    return re.sub(r"[‘’ʻʼ]", "'", text)


def fits(q, a):
    """Savol va javob spec'dagi uslubga mosmi?"""
    q, a = q.strip(), a.strip()
    if "\n" in q or not Q_CHARS[0] <= len(q) <= Q_CHARS[1] or not A_CHARS[0] <= len(a) <= A_CHARS[1]:
        return False
    if re.search(BAD_ANSWER, plain(a), flags=re.M | re.I) or re.search(r"^(Q: |## )", a, flags=re.M):
        return False
    if re.search(CREATIVE, plain(q), flags=re.I):
        return False
    return len(re.findall(r"[.!?](\s|$)", a)) <= MAX_SENTENCES


def uz_apostrophes(text):
    """Tarjimadagi to'g'ri apostrofni spec'dagi shaklga keltiradi: o' g' -> o‘ g‘, qolgani -> ’."""
    text = re.sub(r"([oOgG])['`ʻʼ‘’]", "\\1‘", text)
    return re.sub(r"(?<=\w)['`ʻʼ](?=\w)", "’", text)


def oasst_en():
    """Birinchi savol + eng yuqori baholangan javob (rank 0), faqat ingliz tili va tekshiruvdan o'tganlari."""
    path = hf_hub_download("OpenAssistant/oasst1", "data/train-00000-of-00001-b42a775f407cee45.parquet",
                           repo_type="dataset", local_dir=RAW / "oasst1")
    df = pd.read_parquet(path, columns=["message_id", "parent_id", "text", "role", "lang", "review_result",
                                        "deleted", "rank"])
    ok = df[(df.lang == "en") & (df.review_result == True) & (df.deleted == False)]   # noqa: E712
    questions = ok[(ok.role == "prompter") & ok.parent_id.isna()].set_index("message_id").text
    best = ok[(ok.role == "assistant") & (ok["rank"] == 0)]
    stats = {"savol-javob juftligi": 0, "uslubga mos": 0, "kurs mavzusida": 0}
    out = []
    for parent, a in zip(best.parent_id, best.text):
        if parent not in questions.index:
            continue
        q = questions[parent]
        stats["savol-javob juftligi"] += 1
        if not fits(q, a):
            continue
        stats["uslubga mos"] += 1
        if re.search(TOPIC_EN, q, flags=re.I) and re.search(ASKS_EN, q.strip(), flags=re.I):
            stats["kurs mavzusida"] += 1
            out.append((q.strip(), a.strip()))
    return out, stats


def alpaca_uz():
    """Qo'shimcha "input" maydoni bo'sh bo'lgan, lotin yozuvidagi savol-javoblar."""
    path = hf_hub_download("bizb0630/alpaca-cleaned_uz", "alpaca-cleaned_uz.json",
                           repo_type="dataset", local_dir=RAW / "alpaca-cleaned_uz")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    stats = {"savol-javob juftligi": 0, "uslubga mos": 0, "kurs mavzusida": 0}
    out = []
    for r in rows:
        q, a = r.get("instruction") or "", r.get("output") or ""
        if (r.get("input") or "").strip() or CYRILLIC.search(q + a):
            continue
        stats["savol-javob juftligi"] += 1
        if not fits(q, a):
            continue
        stats["uslubga mos"] += 1
        if re.search(TOPIC_UZ, plain(q), flags=re.I) and re.search(ASKS_UZ, plain(q), flags=re.I):
            stats["kurs mavzusida"] += 1
            out.append((uz_apostrophes(q.strip()), uz_apostrophes(a.strip())))
    return out, stats


def write(name, header, pairs):
    pairs = sorted(set(pairs))
    rng.shuffle(pairs)
    pairs = pairs[:MAX_PER_SOURCE]
    body = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in pairs)
    (HERE / name).write_text(f"{header}\n## external\n\n{body}", encoding="utf-8")
    return len(pairs)


def main():
    for name, load, header in (
        ("external_oasst_en.txt", oasst_en,
         "# Manba: OpenAssistant/oasst1 (https://huggingface.co/datasets/OpenAssistant/oasst1), litsenziya Apache-2.0.\n"
         "# fetch_external.py yasagan. Qo'lda tahrirlamang: o'zgarishlar keyingi yuklashda yo'qoladi."),
        ("external_alpaca_uz.txt", alpaca_uz,
         "# Manba: bizb0630/alpaca-cleaned_uz (https://huggingface.co/datasets/bizb0630/alpaca-cleaned_uz), litsenziya CC-BY-4.0.\n"
         "# Alpaca datasetining o'zbek tiliga mashina tarjimasi. fetch_external.py yasagan, qo'lda tahrirlamang."),
    ):
        pairs, stats = load()
        kept = write(name, header, pairs)
        print(name)
        for step, n in stats.items():
            print(f"  {step:<22} {n}")
        print(f"  {'faylga yozildi':<22} {kept}")


if __name__ == "__main__":
    main()
