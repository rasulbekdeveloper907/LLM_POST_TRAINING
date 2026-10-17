"""DPO 3-qadam: javoblarni baholab, chosen/rejected juftliklarini tuzish -> dpo/data_v2/train.jsonl, val.jsonl.

Ishga tushirish:  python dpo/label.py              (v2 qoidalari, dpo/data_v2/generations.jsonl dan)
                  python dpo/label.py --dir data   (v1 javoblari ustida v2 qoidalarini sinash)

Rad etish va "Bilmayman" (harmful, cheating, unknown):
  chosen — doim toza, qat'iy jumla (rad etishda) yoki SFT datadagi javob ("Bilmayman"da), model javobi emas:
  v1 da modelning buzuq rad javoblari ("...ulashishga yordam bera olmayman") chosen bo'lib qolgan edi.
  rejected — modelning qabul qilib bo'lmaydigan javobi (rad etmagan, buzuq yoki ma'nosiz qo'shimchali).
Qolgan vazifalar:
  - arifmetika, ro'yxat, tarjima: 1 — to'g'ri, aks holda past ball;
  - tushuntirish, faktlar va h.k.: SFT datadagi javobga so'zlar bo'yicha o'xshashlik (0..1);
  - jarimalar: keraksiz "Kechirasiz", to'xtamaslik, takrorlanish, boshqa tilda javob.
  Eng yaxshi javob — chosen (hammasi yomon bo'lsa, SFT datadagi javob). Rejected — ancha yomon javoblar ichidan
  uzunligi chosen'ga eng yaqini. chosen va rejected uzunligi juda farq qilsa, juftlik tashlanadi:
  v1 da 63% juftlikda chosen qisqaroq edi va model "qisqa — yaxshi" deb o'rganib qoldi.
"""
import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
VAL_SHARE = 0.05
LENGTH_RATIO = (0.6, 1.7)   # len(chosen) / len(rejected) shu oraliqda bo'lishi kerak (rad etishdan tashqari)
rng = random.Random(42)

REFUSAL_OK = re.compile(r"yordam bera olmayman|can't help|cannot help", re.I)
REFUSAL_ANY = re.compile(r"kechirasiz|olmayman|\bsorry\b|can't help", re.I)
UNKNOWN_OK = re.compile(r"^(bilmayman|i don't know)\b", re.I)
UZ_MARK = re.compile(r"[‘’ʻʼ]|\b(va|bu|uchun|bilan|nima|emas|ham|bo'l|qil|bor)\w*", re.I)
EN_MARK = re.compile(r"\b(what|how|why|which|who|where|the|is|are|give|list|name|translate|tell|my|i)\b", re.I)
APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʻʼ`"})

# Rad etish uchun qat'iy jumlalar: bitta shakl — model shakllarni aralashtirib buzib yubormaydi.
FIXED = {("harmful", "uz"): "Kechirasiz, bunga yordam bera olmayman.",
         ("harmful", "en"): "Sorry, I can't help with that.",
         ("cheating", "uz"): "Kechirasiz, imtihon paytida yordam bera olmayman.",
         ("cheating", "en"): "Sorry, I can't help with answers during an exam."}


def words(text):
    return re.findall(r"[\w']+", text.translate(APOSTROPHES).lower())


def f1(a, b):
    """So'zlar bo'yicha o'xshashlik: 1 — bir xil so'zlar, 0 — umumiy so'z yo'q."""
    wa, wb = Counter(words(a)), Counter(words(b))
    common = sum((wa & wb).values())
    if not common:
        return 0.0
    p, r = common / sum(wa.values()), common / sum(wb.values())
    return 2 * p * r / (p + r)


def language(text):
    """'uz', 'en' yoki None (masalan, faqat raqamlardan iborat "83 - 6 * 9 = ?": v1 da bu xato 'en' deb olingan)."""
    if UZ_MARK.search(text):
        return "uz"
    if EN_MARK.search(text):
        return "en"
    return None


OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
       ast.Div: lambda a, b: a // b}


def calc(expr):
    """Faqat butun sonlar va + - * / ni hisoblaydi (eval o'rniga: model matnidagi "9**9**9" kabi narsa ishlamaydi)."""
    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](walk(n.left), walk(n.right))
        raise ValueError("ruxsat etilmagan ifoda")
    return walk(ast.parse(expr.strip(), mode="eval").body)


def arithmetic_ok(text, number):
    """Birinchi natija to'g'ri va matndagi har bir "a op b = c" tenglik ham to'g'ri bo'lishi kerak."""
    first = re.search(r"=\s*(\d+)", text)
    if not first or int(first.group(1)) != number:
        return False
    for lhs, rhs in re.findall(r"([\d\s+\-*/()]*\d[\d\s+\-*/()]*)=\s*(\d+)", text):
        try:
            if calc(lhs) != int(rhs):
                return False
        except (SyntaxError, ValueError, ZeroDivisionError):
            pass   # "Avval ... 4 / 2" kabi qismlar to'liq ifoda emas: tekshirilmaydi
    return True


# ---------- rad etish va "Bilmayman" ----------

def acceptable_refusal(item, text, chosen):
    if item["task"] == "unknown":
        return bool(UNKNOWN_OK.match(text)) and not REFUSAL_ANY.search(text) and (
            len(words(text)) <= 2 or f1(text, item["reference"]) >= 0.5)
    return bool(REFUSAL_OK.search(text)) and (f1(text, chosen) >= 0.7 or f1(text, item["reference"]) >= 0.7)


def refusal_pair(item):
    task = item["task"]
    chosen = item["reference"] if task == "unknown" else FIXED[(task, language(item["prompt"]) or "uz")]
    bad = [a["text"] for a in item["answers"] if a["text"] and not acceptable_refusal(item, a["text"], chosen)]
    if not bad:
        return None   # model hamma safar to'g'ri javob bergan: o'rgatadigan narsa yo'q
    # Avval umuman rad etmagan javoblar (eng xavflisi), keyin buzuq rad javoblari.
    worst = sorted(bad, key=lambda t: (bool(REFUSAL_OK.search(t) or UNKNOWN_OK.match(t)), abs(len(t) - len(chosen))))[0]
    return chosen, worst, "fixed" if task != "unknown" else "reference", []


# ---------- qolgan vazifalar ----------

def task_score(item, text):
    task = item["task"]
    if task == "arithmetic":
        return 1.0 if arithmetic_ok(text, item["number"]) else 0.0
    if task == "list":
        found = re.findall(r"^\s*\d+[.)]\s+(.+)$", text, flags=re.M)
        # Bitta so'zli bandlar ("1. Menda.") o'tmaydi; "Label encoding" kabi nomlar esa o'tadi.
        ok = len(found) == item["items"] and all(len(words(x)) >= 2 for x in found)
        return 1.0 if ok and not REFUSAL_ANY.search(text) else 0.0
    if task == "translate":
        # So'zlar o'xshashligi yetmaydi: "hot" o'rniga "cold" deyilsa ham o'xshashlik 0.8 chiqadi.
        ref, ans = set(words(item["reference"])), words(text)
        if ref <= set(ans) and len(ans) <= 1.5 * len(ref) + 1:
            return 1.0
        return 0.6 * f1(text, item["reference"])
    return f1(text, item["reference"])


def penalty(item, answer):
    text, why = answer["text"], []
    if REFUSAL_ANY.search(text):
        why.append("keraksiz rad")
    if not answer["finished"]:
        why.append("to'xtamadi")
    sentences = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 10]
    if len(sentences) != len(set(sentences)):
        why.append("takror")
    if item["task"] != "translate":
        q = language(item["prompt"])
        if q == "uz" and len(words(text)) >= 6 and not UZ_MARK.search(text):
            why.append("inglizcha javob")
        if q == "en" and re.search(r"[‘’ʻʼ]", text):
            why.append("o'zbekcha javob")
    return why


def score(item, answer):
    if not answer["text"]:
        return -1.0, ["bo'sh"]
    why = penalty(item, answer)
    return task_score(item, answer["text"]) - 0.5 * len(why), why


def normal_pair(item):
    scored = [(score(item, a), a["text"]) for a in item["answers"]]
    good = 1.0 if item["task"] in ("arithmetic", "list", "translate") else 0.5   # "yaxshi javob" chegarasi
    (best, _), best_text = max(scored, key=lambda x: (x[0][0], -len(x[1])))
    if best >= good:
        chosen, chosen_score, source = best_text, best, "model"
    elif best < good / 2:
        chosen, chosen_score, source = item["reference"], good, "reference"   # model umuman yaxshi javob bermagan
    else:
        return None
    worse = [(s, why, t) for (s, why), t in scored if s <= chosen_score - 0.5 and t != chosen]
    if not worse:
        return None
    # Rejected: yomon javoblar ichidan uzunligi chosen'ga eng yaqini — farq uzunlikda emas, sifatda bo'lsin.
    _, why, rejected = min(worse, key=lambda x: abs(len(x[2]) - len(chosen)))
    if item["task"] not in ("arithmetic", "translate") and f1(chosen, rejected) >= 0.85:
        # Deyarli bir xil javoblar (faqat tinish belgisi farq qiladi): o'rgatadigan farq yo'q. Arifmetika va
        # tarjimada esa bitta son yoki so'z farqi — aynan muhim farq, shuning uchun ular tashlanmaydi.
        return None
    ratio = len(chosen) / max(1, len(rejected))
    if not LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1]:
        return None
    return chosen, rejected, source, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data_v2", help="dpo/ ichidagi papka: generations.jsonl shu yerdan o'qiladi")
    args = p.parse_args()
    folder = HERE / args.dir
    items = [json.loads(line) for line in (folder / "generations.jsonl").read_text(encoding="utf-8").splitlines()]

    pairs, skipped = [], Counter()
    for item in items:
        pair = refusal_pair(item) if item["task"] in ("harmful", "cheating", "unknown") else normal_pair(item)
        if pair is None:
            skipped[item["task"]] += 1
            continue
        chosen, rejected, source, why = pair
        pairs.append({"id": item["id"], "task": item["task"], "source": source, "rejected_why": why,
                      "prompt": item["prompt"], "chosen": chosen, "rejected": rejected})

    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * VAL_SHARE))
    for name, rows in (("val.jsonl", pairs[:n_val]), ("train.jsonl", pairs[n_val:])):
        (folder / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    print(f"{'vazifa':<12} {'juftlik':>8} {'model':>6} {'ref/fixed':>10} {'tashlandi':>10}")
    for task in sorted({i["task"] for i in items}):
        mine = [p for p in pairs if p["task"] == task]
        print(f"{task:<12} {len(mine):>8} {sum(p['source'] == 'model' for p in mine):>6} "
              f"{sum(p['source'] != 'model' for p in mine):>10} {skipped[task]:>10}")
    shorter = sum(len(p["chosen"]) < len(p["rejected"]) for p in pairs)
    print(f"{'jami':<12} {len(pairs):>8}  (train {len(pairs) - n_val}, val {n_val})")
    print(f"chosen rejected'dan qisqa: {shorter} ({100 * shorter / max(1, len(pairs)):.0f}%) | o'rtacha uzunlik: "
          f"chosen {sum(len(p['chosen']) for p in pairs) / max(1, len(pairs)):.0f}, "
          f"rejected {sum(len(p['rejected']) for p in pairs) / max(1, len(pairs)):.0f} belgi")
    print("Rejected javoblardagi jarimalar:", dict(Counter(w for p in pairs for w in p["rejected_why"]).most_common()))

    print("\nTekshirish uchun misollar (har vazifadan bittadan):")
    for task in sorted({p["task"] for p in pairs}):
        p = next(p for p in pairs if p["task"] == task)
        print(f"\n[{task}, {p['source']}] {p['prompt']}\n  + {p['chosen'][:150]}\n  - {p['rejected'][:150]}")


if __name__ == "__main__":
    main()
