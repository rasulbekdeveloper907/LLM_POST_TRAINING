"""SFT v3 dataset: v1 manbalari + sft_v3/new_samples.txt -> sft_v3/data/train.jsonl, val.jsonl.

Ishga tushirish (loyiha papkasidan):  python sft_v3/build_v3.py

v1 dan farqi:
  - qo'shildi:  new_samples.txt (kam uchragan tushunchalar, ro'yxatlar, imtihon misollari, tarjima juftliklari)
  - olib tashlandi:  external_alpaca_uz.txt (buzuq mashina tarjimasi) va extra_general.txt
    (umumiy bilim va kod: MODEL_SPEC.md 7-bo'limiga ko'ra bu versiyada kerak emas)
  - arifmetika 100 emas, 200 ta
  - test savoliga deyarli bir xil savollar ham tashlanadi (v1 da faqat aynan bir xillari tashlangan)
Tozalash, tekshiruv va arifmetika/tarjima generatsiyasi sft/build_sft.py dan olinadi.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sft"))
import build_sft  # noqa: E402

SOURCES = [
    ("seeds.txt", None),
    ("sft_500_uzbek_samples.txt", "explain"),
    (HERE / "new_samples.txt", None),   # build_sft HERE / Path -> shu Path'ning o'zi
]

if __name__ == "__main__":
    build_sft.N_ARITHMETIC = 200
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    # 0.98: faqat deyarli bir xil savollar (apostrof, tinish belgisi farqi). Pastroq chegara "LoRA qisqartmasi
    # nimani anglatadi?" kabi faqat shabloni testga o'xshash, lekin javobi boshqa savollarni ham tashlab yuboradi.
    build_sft.main(sources=SOURCES, out_dir=out, fuzzy_cutoff=0.98)
