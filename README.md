# 🩺 AI Doctor — Medical LLM Fine-Tuning Project

Fine-tuning **DeepSeek-R1-Distill-Llama-8B** to act as an AI-powered medical assistant that answers clinical questions with step-by-step reasoning using Chain-of-Thought (CoT) methodology.

---

## 🎯 Project Goal

Train an open-source LLM to reason like a medical expert — analyzing patient symptoms, applying clinical knowledge, and providing structured diagnostic answers using **LoRA fine-tuning** on a medical reasoning dataset.

---

## ✨ Features

- **Medical instruction tuning** using real clinical Q&A data
- **Chain-of-Thought (CoT) reasoning** — model thinks step by step before answering
- **Parameter-efficient fine-tuning** with LoRA (only 0.52% of parameters trained!)
- **4-bit quantization** for memory-efficient GPU training
- **GPU-optimized training** with Unsloth (2x faster than standard training)
- **WandB integration** for real-time training monitoring
- **Google Drive integration** for model persistence across sessions
- **Pre-training inference** + **post-training inference** comparison

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **DeepSeek-R1-Distill-Llama-8B** | Base pretrained model (8 Billion parameters) |
| **Unsloth** | 2x faster fine-tuning & memory optimization |
| **HuggingFace Transformers** | Model loading and tokenization |
| **TRL (SFTTrainer)** | Supervised fine-tuning trainer |
| **PEFT (LoRA)** | Parameter-efficient adaptation |
| **PyTorch** | Deep learning backend |
| **WandB** | Training metrics tracking |
| **Google Colab (T4 GPU)** | Cloud GPU training environment |
| **ModelScope** | Alternative model download source |

---

## 📂 Project Structure

```
AI-Doctor-Finetuning-Project/
├── notebooks/
│   └── AI_Doctor.ipynb        # Main training & inference notebook
├── datasets/                   # Training datasets
├── scripts/                    # Utility scripts
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔄 Project Pipeline

```
Environment Setup
      ↓
Install Dependencies (Unsloth, TRL, PEFT, ModelScope)
      ↓
Load Pretrained Model (DeepSeek-R1 8B, 4-bit quantized)
      ↓
Pre-training Inference (test base model)
      ↓
Load Medical Dataset (500 examples)
      ↓
Data Preprocessing (format with CoT prompt template)
      ↓
Apply LoRA Adapters (only 41.9M / 8B parameters)
      ↓
Fine-tune with SFTTrainer (60 steps)
      ↓
Post-training Inference (test improved model)
      ↓
Save Model to Google Drive
```

---

## 📊 Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | DeepSeek-R1-Distill-Llama-8B | Strong reasoning capability |
| Quantization | 4-bit (QLoRA) | Fits in 15GB T4 GPU |
| LoRA rank (r) | 16 | Balance of quality vs speed |
| LoRA alpha | 16 | Scaling factor = 1.0 |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Batch size | 2 (× 4 accumulation = 8 effective) | Memory efficient |
| Training steps | 60 | Quick experimentation |
| Trainable params | 41.9M / 8.07B (0.52%) | Efficient fine-tuning |

---

## 📋 Dataset

**FreedomIntelligence/medical-o1-reasoning-SFT**

- 500 medical Q&A examples (English)
- Each example has 3 fields:
  - `Question` — Clinical scenario
  - `Complex_CoT` — Step-by-step reasoning
  - `Response` — Final medical answer

---

## 🧠 Prompt Template

```
### Instruction:
You are a medical expert specializing in clinical reasoning,
diagnostics, and treatment planning.

### Question:
{clinical question}

### Response:
<think>
{step-by-step reasoning}
</think>
{final answer}
```

---

## 📈 Training Results

| Step | Training Loss |
|------|--------------|
| 10 | 11.41 |
| 20 | 4.20 |
| 30 | 3.14 |
| 40 | 3.57 |
| 50 | 3.11 |
| 60 | ~3.0 |

Loss decreased from **11.41 → ~3.0** showing the model successfully learned medical reasoning patterns.

---

## 🚀 How to Run

### Prerequisites
- Google Colab account (T4 GPU)
- HuggingFace account + API token
- WandB account + API token
- Google Drive (for model storage)

### Steps

**1. Set up Colab Secrets:**
```
HF_TOKEN = your_huggingface_token
WANDB_API_TOKEN = your_wandb_token
```

**2. Run notebook cells in order:**
```python
# Cell 1: Environment setup
import os
os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'
os.environ["LD_LIBRARY_PATH"] = (
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

# Cell 2-3: Install dependencies
!pip install modelscope -q
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

# Cell 4: Import libraries
# Cell 5: HuggingFace login
# Cell 6: GPU check
# Cell 7: Load model
# Cell 8-10: Google Drive setup
# Cell 11-12: Pre-training inference
# Cell 13-19: Dataset preparation
# Cell 20-25: LoRA setup + Trainer
# Cell 26-28: Training + WandB
# Cell 29: Post-training inference
```

---

## 💡 Key Technical Concepts

### LoRA (Low-Rank Adaptation)
Instead of training all 8 billion parameters, LoRA adds small trainable matrices to specific layers. Only **0.52% of parameters** are trained, saving memory and time.

### 4-bit Quantization (QLoRA)
Model weights compressed from 32-bit to 4-bit, reducing memory from ~32GB to ~4GB, making it possible to train on a free T4 GPU.

### Chain-of-Thought (CoT) Reasoning
The model is trained to show its thinking process inside `<think>...</think>` tags before giving the final answer, improving accuracy on complex medical questions.

### Unsloth Optimization
Unsloth uses custom CUDA kernels to make training 2x faster with 70% less memory compared to standard HuggingFace training.

---

## ⚙️ Requirements

```
torch>=2.0.0
unsloth
transformers
trl
peft
datasets
wandb
huggingface_hub
modelscope
accelerate
bitsandbytes
```

---

## 👨‍💻 Author

**Prashant Kumar**

*AI/ML Engineer | LLM Fine-tuning | Medical AI*

---
