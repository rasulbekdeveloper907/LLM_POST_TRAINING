# LLM Post-Training — Qwen2.5-0.5B

Ushbu loyiha **Qwen2.5-0.5B** katta til modelini (LLM) **post-training** qilish jarayonini o‘rganish va amaliy tarzda bajarish uchun yaratilgan.

Loyihaning asosiy maqsadi — tayyor **base model**ni maxsus dataset yordamida kerakli vazifalarni yaxshiroq bajarishga moslashtirish.

---

## 🎯 Loyiha maqsadi

Ushbu loyiha davomida:

* Qwen2.5-0.5B base model bilan ishlash
* LLM post-training tushunchasini o‘rganish
* Dataset tayyorlash
* Supervised Fine-Tuning (SFT)
* Training va validation jarayonlarini tashkil qilish
* Model natijalarini baholash
* Base model va fine-tuned model natijalarini taqqoslash
* Tayyor modelni inference qilish

amalga oshiriladi.

---

## 🧠 Post-Training nima?

**Post-training** — oldindan katta hajmdagi datasetda pre-training qilingan tayyor modelni ma'lum vazifalar, instruktsiyalar yoki inson preferensiyalariga moslashtirish jarayonidir.

Oddiy ko‘rinishda:

```text
Pre-trained Base Model
        ↓
   Post-Training
        ↓
Instruction / Task Dataset
        ↓
   Fine-Tuned Model
        ↓
Better Task Performance
```

---

## 🤖 Model

Ushbu loyihada:

**Qwen2.5-0.5B** modeli ishlatiladi.

Model:

* Architecture: Transformer
* Parameters: ~0.5B
* Model family: Qwen2.5
* Type: Causal Language Model
* Base model: Qwen2.5-0.5B

Model kichik hajmli bo‘lgani sababli tajriba va o‘quv loyihalari uchun qulay.

---

## 🔧 Post-Training usuli

Loyihaning asosiy training usuli:

### Supervised Fine-Tuning (SFT)

SFT jarayonida modelga:

```text
Instruction
     +
Input
     ↓
Expected Output
```

ko‘rinishidagi misollar beriladi.

Masalan:

```json
{
  "instruction": "Python'da listni qanday sort qilish mumkin?",
  "input": "",
  "output": "Python'da sorted() funksiyasidan foydalanish mumkin."
}
```

Model ushbu misollar orqali foydalanuvchi instructionlariga mos javob berishni o‘rganadi.

---

## 📊 Training Pipeline

Loyihaning umumiy pipeline'i:

```text
Dataset
   ↓
Data Cleaning
   ↓
Data Formatting
   ↓
Train / Validation Split
   ↓
Tokenizer
   ↓
Qwen2.5-0.5B
   ↓
Supervised Fine-Tuning
   ↓
Evaluation
   ↓
Fine-Tuned Model
   ↓
Inference
```

---

## 📁 Project Structure

```text
LLM_Post_Training/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   └── val/
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── configs/
│   └── training_config.yaml
│
├── outputs/
│   ├── checkpoints/
│   └── final_model/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ O‘rnatish

Repository'ni clone qilish:

```bash
git clone https://github.com/rasulbekdeveloper907/LLM_POST_TRAINING.git
cd LLM_POST_TRAINING
```

Virtual environment yaratish:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Kerakli kutubxonalarni o‘rnatish:

```bash
pip install -r requirements.txt
```

---

## 📦 Asosiy kutubxonalar

Loyihada quyidagi texnologiyalardan foydalaniladi:

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* Accelerate
* TRL
* Qwen2.5-0.5B

---

## 🚀 Training

Training jarayonida Qwen2.5-0.5B modeli datasetdagi instruction-response juftliklari yordamida fine-tune qilinadi.

Umumiy buyruq:

```bash
python train.py
```

Training parametrlari `configs/training_config.yaml` faylida saqlanishi mumkin.

Asosiy parametrlar:

```text
model_name
learning_rate
batch_size
gradient_accumulation_steps
num_train_epochs
max_seq_length
warmup_ratio
weight_decay
```

---

## 📈 Evaluation

Trainingdan keyin model validation dataset yordamida baholanadi.

Baholashda:

* Training Loss
* Validation Loss
* Perplexity
* Response Quality
* Instruction Following

kabi ko‘rsatkichlardan foydalanish mumkin.

Shuningdek, **base model** va **fine-tuned model** bir xil promptlar yordamida taqqoslanadi.

---

## 💬 Inference

Fine-tuned modelni test qilish:

```text
User Prompt
     ↓
Tokenizer
     ↓
Qwen2.5-0.5B Fine-Tuned
     ↓
Generated Response
```

Masalan:

```text
Prompt:
Python'da dictionary nima?

Response:
Dictionary — Python'da key-value juftliklarida ma'lumot
saqlash uchun ishlatiladigan ma'lumotlar strukturasi.
```

---

## 🔬 Base Model vs Fine-Tuned Model

Post-training natijasini ko‘rish uchun ikkita model taqqoslanadi:

| Model                   | Holati                         |
| ----------------------- | ------------------------------ |
| Qwen2.5-0.5B Base       | Post-training qilinmagan       |
| Qwen2.5-0.5B Fine-Tuned | Dataset yordamida SFT qilingan |

Maqsad — fine-tuningdan keyin modelning kerakli vazifalarni bajarish qobiliyati yaxshilanganini ko‘rsatish.

---

## 🧩 Keyingi bosqichlar

Loyiha keyinchalik quyidagi yo‘nalishlarda kengaytirilishi mumkin:

* LoRA / QLoRA
* DPO
* RLHF
* Preference Dataset
* Human Evaluation
* Automatic Evaluation
* Quantization
* Model Deployment
* Web UI
* API

---

## 📚 O‘rganiladigan asosiy tushunchalar

Ushbu loyiha orqali quyidagi tushunchalar amaliy o‘rganiladi:

```text
LLM
 │
 ├── Pre-Training
 │
 └── Post-Training
       │
       ├── SFT
       │
       ├── RLHF
       │
       └── DPO
```

Shuningdek:

* Base Model
* Instruct Model
* Tokenization
* Fine-Tuning
* LoRA
* PEFT
* Dataset Formatting
* Evaluation
* Inference

---

## 👨‍💻 Author

**Rasulbek**

GitHub:

https://github.com/rasulbekdeveloper907

---

## ⭐ Project

Agar loyiha foydali bo‘lsa, repository'ga ⭐ star qo‘yish mumkin.

```text
Qwen2.5-0.5B
      ↓
   Dataset
      ↓
     SFT
      ↓
Fine-Tuned LLM
```

**Goal:** kichik hajmdagi LLM yordamida post-training jarayonini noldan amaliy o‘rganish.
