# DPO on top of SFT v3 LoRA (Qwen2.5-1.5B, Kaggle GPU)

Starting model: SFT v3 LoRA (41/68). It has the cleanest behaviour of the SFT models; what is left
(garbled refusals, refusing harmless exam questions, "Bilmayman" followed by nonsense, not stopping)
is behaviour, which DPO targets. DPO is not expected to fix wrong facts.

| Step | Script | Where | Output |
|---|---|---|---|
| 1. Prompts | `build_prompts.py` | laptop (done) | `data/prompts.jsonl` — 776 prompts from the SFT v3 data + 100 new arithmetic |
| 2. Answers | `generate.py` | Kaggle GPU | `data/generations.jsonl` — 4 sampled answers per prompt |
| 3. Pairs | `label.py` | Kaggle (or laptop) | `data/train.jsonl`, `data/val.jsonl` — chosen / rejected |
| 4. Training | `train_dpo.py` | Kaggle GPU | `chat_ui/adapters/sft-lora-dpo/` |
| 5. Evaluation | `eval/run_eval.py --base Qwen2.5-1.5B --model sft-lora-dpo` | Kaggle | `eval/results/sft-lora-dpo.json` |

Everything from step 2 runs in `dpo_kaggle.ipynb`. Before running it, upload
`sft_v3/sft_v3_adapters.zip` to Kaggle as a dataset (Add Input → Upload); otherwise the notebook
retrains SFT v3 LoRA first.

## How the pairs are labelled (`label.py`)

| Task | Good answer (score 1) |
|---|---|
| Harmful request, exam cheating | contains "yordam bera olmayman" / "can't help", short |
| Unknown | starts with "Bilmayman" / "I don't know", no refusal, close to the clean reference |
| Arithmetic | first result correct and every "a op b = c" in the text correct |
| List | exactly the requested number of items, no refusal text |
| Translation | contains every word of the correct translation (word overlap alone accepts "cold" for "hot") |
| Explanation, facts, exam prep, chat | word overlap with the reference answer (0–1) |

Penalties (−0.5 each): refusal text in a normal answer, did not stop, repeated sentences, far too long,
wrong language. The best answer becomes *chosen*, the worst *rejected*. If no model answer is good,
the reference answer from the SFT data is *chosen*. Prompts without a clear winner are skipped.

## Training (`train_dpo.py`)

- The policy is the base model with the SFT v3 adapter, trained further (`is_trainable=True`).
- The reference model is the same model before training. Its log-probabilities are computed once at the
  start, so only one model is in memory.
- Loss: `-log σ(β · [(log π − log π_ref)(chosen) − (log π − log π_ref)(rejected)])`, β = 0.1.
- Learning rate 2e-5 (SFT used 2e-4), 2 epochs, 4 pairs per step, dropout off.
- Before training the validation loss is exactly 0.693 (ln 2), because policy and reference are identical.

## Results: DPO v1 (Kaggle, 2026-10-16)

313 pairs, 2 epochs. Validation: loss 0.693 → 0.311, accuracy 0 → 0.75, so DPO learned its pairs.
Test score 40/68 (SFT v3: 41/68).

Fixed: refusal text in normal answers (1 → 0), refusing a harmless exam question ("Bilmayman" now), one arithmetic question.
Worse: one harmful request no longer refused, refusals cut short ("Kechirasiz, imtihondan keyin."), answers much shorter (87 → 51 characters).

Causes found in the data and the training log:
- Length bias: chosen was shorter than rejected in 63% of pairs, so the model learned "shorter is better".
- The chosen reward also fell (−0.7, rejected −5.2): the model pushed both answers down.
- Chosen refusals were often garbled; the rules only checked for "yordam bera olmayman".
- Only 313 pairs, and a bug penalised Uzbek answers to digits-only arithmetic questions as "wrong language".

## DPO v2: fixes for what went wrong in v1

Notebook: `dpo_v2_kaggle.ipynb` (same Kaggle setup; the `sft-v3-adapters` dataset can be reused).
Output: `dpo/data_v2/`, `chat_ui/adapters/sft-lora-dpo-v2/`, `eval/results/sft-lora-dpo-v2.json`.

| Problem in v1 | Fix in v2 |
|---|---|
| Garbled model refusals became *chosen* | Refusals: *chosen* is always one fixed clean sentence; "Bilmayman": the SFT reference answer |
| Length bias (chosen shorter in 63% of pairs) | *Rejected* is the bad answer closest in length to *chosen*; pairs with length ratio outside 0.6–1.7 are skipped (refusals excepted) |
| Chosen probability fell during training | SFT loss on the chosen answer added to the DPO loss (`--sft-weight 1.0`); best epoch chosen by DPO + SFT loss |
| Too aggressive training | Learning rate 5e-6 instead of 2e-5 |
| Too few pairs | 8 answers per prompt instead of 4 |
| One-word list items passed ("1. Menda tayyor.") | Every list item needs at least 2 words |
| Digits-only arithmetic prompts treated as English | Language check skipped when the prompt has no language markers |
| Near-identical pairs (only punctuation differs) | Skipped, except arithmetic and translation, where one number or word is the real difference |

Tested on the v1 answers (4 per prompt): 293 pairs, chosen shorter in 49% of them (v1: 63%).

## Results: DPO v2 (Kaggle, 2026-10-16)

357 pairs (chosen shorter in 47%), 2 epochs. Validation: loss 0.693 → 0.571, accuracy 0.78.
The chosen reward rose to +0.22 (v1: fell to −0.7), so the SFT term did its job.

| | SFT v3 | DPO v1 | DPO v2 |
|---|---|---|---|
| Test score | 41/68 | 40/68 | **44/68** |
| Refuses harmful | 4/4 | 3/4 | **4/4** |
| Refuses cheating | 3/4 | 2/4 | **4/4** |
| "Bilmayman" | 4/5 | 5/5 | **5/5** |
| "Kechirasiz" inside normal answers | 1 | 0 | **0** |
| Average answer length | 87 | 51 | 76 |

Against SFT v3, DPO v2 changed three test answers, all for the better and none for the worse:
the exam question is now refused cleanly, a harmless exam question gets "Bilmayman" instead of a refusal,
and spam detection is now called classification. Refusals are now the one clean sentence from the pairs.
Not fixed, as expected from DPO: wrong facts and "Bugun havo issiq" → "cold".
