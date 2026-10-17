# SFT + QLoRA (trained on a Kaggle GPU)

The base model is loaded in 4-bit and frozen; the same kind of LoRA adapter is trained on top.
The only code difference from `sft/lora_cpu/train_lora.py` is the block that loads the model in 4-bit.

| | |
|---|---|
| Script | `train_qlora.py` |
| Notebook | `sft_qlora.ipynb` (import it on Kaggle, set Accelerator to GPU and Internet to On) |
| Hardware | Kaggle free GPU (QLoRA does not run on a CPU) |
| Settings | 10 epochs, rank 64, batch size 8, learning rate 2e-4; only the best epoch is saved |
| Validation loss per epoch | 2.00, 1.56, 1.48 (best, saved), then rising to 1.91 by epoch 10 |
| Adapter | `chat_ui/adapters/sft-qlora/` (not in git) |
| Test score | 32/68 (base model: 5/68), answers in `eval/results/sft-qlora.json` |
| Loss at every step | `training_log.json` |

The adapter was downloaded from Kaggle and the test questions were run on the laptop CPU:
`python eval/run_eval.py --model sft-qlora`.

Note: this run used a higher rank and more epochs than the CPU LoRA run, so the two scores
compare two setups, not LoRA against QLoRA alone.
