# SFT + LoRA (trained on this PC, CPU)

The base model stays at full precision and frozen; only a small LoRA adapter is trained.

| | |
|---|---|
| Script | `train_lora.py` |
| Run from the project folder | `python sft/lora_cpu/train_lora.py` |
| Hardware | Laptop CPU, no GPU |
| Settings | 3 epochs, rank 16, batch size 8, learning rate 2e-4 |
| Trainable parameters | 8.8 million of 503 million (1.75%) |
| Training time | 151 minutes |
| Validation loss per epoch | 2.09, 1.69, 1.62 |
| Adapter | `chat_ui/adapters/sft-lora/` (not in git) |
| Test score | 32/68 (base model: 5/68), answers in `eval/results/sft-lora.json` |
| Loss at every step | `training_log.json` |

Evaluate with `python eval/run_eval.py --model sft-lora`.
