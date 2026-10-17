# SFT v3: new data, Qwen2.5-1.5B, LoRA and QLoRA (Kaggle GPU)

Same model and training script as v2 (`sft_v2/train.py`). Only the data changes, so the v2 → v3
difference shows the effect of the data.

| | |
|---|---|
| Build the data | `python sft_v3/build_v3.py` → `sft_v3/data/train.jsonl`, `val.jsonl` |
| New examples | `new_samples.txt` (165 kept after cleaning) |
| Notebook | `sft_v3_kaggle.ipynb` (import it on Kaggle, set Accelerator to GPU and Internet to On) |
| Settings | 3 epochs (v2 overfit after epoch 2), rank 64, batch size 8, learning rate 2e-4; best epoch saved |
| Adapters | `chat_ui/adapters/sft-lora-v3/`, `chat_ui/adapters/sft-qlora-v3/` (not in git) |
| Test answers | `eval/results/sft-lora-v3.json`, `sft-qlora-v3.json` |

## What changed in the data

| | v1 / v2 | v3 |
|---|---|---|
| Examples (train + val) | 1,252 | 1,583 |
| Alpaca-uz (machine-translated) | 49 | removed |
| General knowledge and code (`extra_general.txt`) | 42 | removed (not in scope, MODEL_SPEC.md §7) |
| Arithmetic | 100 | 200 |
| Translation | 160 | 317 (80 new pairs, many with opposite adjectives) |
| New course concepts (gradient clipping, KV cache, top-p, MoE…) | — | ~70, in Uzbek and English |
| Lists of practical tips | 64 | 95 (against refusals leaking into lists) |
| Exam cheating with a real question inside | 20 | 33 |
| Exam preparation (should be answered) | 12 | 25 |
| Test-question check | exact match | exact match + near-identical (similarity ≥ 0.98) |

New harmful-request examples were not added; the 27 refusals from `seeds.txt` are kept.
Refusals and "Bilmayman" make up 7% of the data.

The validation set is new, so v3 validation loss cannot be compared with v2; compare the test scores.

## Results (Kaggle, 2026-10-16)

| | LoRA v3 | QLoRA v3 | LoRA v2 | QLoRA v2 |
|---|---|---|---|---|
| Validation loss per epoch | 1.35, **1.08**, 1.14 | 1.39, **1.11**, 1.18 | see `sft_v2/README.md` | |
| Test score (automatic check) | 41/68 | 41/68 | 44/68 | 42/68 |
| "Kechirasiz" inside non-refusal answers | 1 | 0 | 4 | 4 |

The total did not improve; a 1–3 point change on 68 questions is within noise.
What the new data fixed: refusal text no longer leaks into lists, and exam questions asked
during an exam are no longer answered (cheat-01). What it did not fix: "Bugun havo issiq" is
still translated as "cold", facts are still often wrong, and some new refusals are garbled
("Kechirasiz, imtihondan keyin bo‘lmaydi"), which the automatic check counts as a fail.
