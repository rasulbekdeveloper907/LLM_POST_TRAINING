# SFT v2: Qwen2.5-1.5B, LoRA and QLoRA (Kaggle GPU)

The same SFT as v1, on a model three times larger. LoRA and QLoRA run with identical settings,
so this time the two can be compared directly.

| | |
|---|---|
| Script | `train.py --method lora` or `train.py --method qlora` |
| Notebook | `sft_v2_kaggle.ipynb` (import it on Kaggle, set Accelerator to GPU and Internet to On) |
| Base model | Qwen2.5-1.5B (downloaded in the notebook, not in git) |
| Data | Same as v1: `sft/train.jsonl`, `sft/val.jsonl` (change with `--data`) |
| Settings | 5 epochs, rank 64, batch size 8, learning rate 2e-4; only the best epoch is saved |
| LoRA | Base model in 32 bit, gradient checkpointing on so it fits a 16 GB GPU |
| QLoRA | Base model in 4 bit |
| Adapters | `chat_ui/adapters/sft-lora-1.5b/`, `chat_ui/adapters/sft-qlora-1.5b/` (not in git) |
| Test answers | `eval/results/base-1.5b.json`, `sft-lora-1.5b.json`, `sft-qlora-1.5b.json` |

What each comparison shows:

| Compare | Shows the effect of |
|---|---|
| v1 0.5B vs v2 1.5B (same data) | Model size |
| LoRA 1.5B vs QLoRA 1.5B (same settings) | 4-bit loading |
| Base 1.5B vs SFT 1.5B | SFT itself |

## Results (Kaggle, 2026-10-15)

| | LoRA 1.5B | QLoRA 1.5B |
|---|---|---|
| Validation loss per epoch | 1.44, **1.19**, 1.23, 1.35, 1.46 | 1.49, **1.22**, 1.23, 1.37, 1.46 |
| Best epoch (saved) | 2 | 2 |
| Test score (base 1.5B: 15/68) | **44/68** | 42/68 |

Both runs overfit after epoch 2, so 2–3 epochs are enough on this data.
Full per-task table: `results/per_task_scores.md`.

Evaluate a 1.5B adapter with `python eval/run_eval.py --base Qwen2.5-1.5B --model sft-lora-1.5b`.
