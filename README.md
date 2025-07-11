# GPT-From-Scratch 🚀  
A minimal yet powerful implementation of a **Generative Pretrained Transformer (GPT)** model inspired by the seminal paper **“Attention is All You Need”** and modern transformer architectures 

---

## 🎯 Project Aim

This project aims to build a compact, efficient, and interpretable transformer model from scratch using only core PyTorch components. Inspired by nanoGPT, our implementation targets ~**0.2 million parameters**, making it lightweight enough to train on a single GPU or CPU.

The model is trained on the **Tiny Shakespeare** dataset and optimized to generate coherent, relevant, and stylistically consistent text.

---

## 📚 Dataset

- **Name**: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)  
- **Size**: ~1MB (character-level text corpus)  
- **Format**: Plain text (.txt)

---

## 🧠 Model Overview

- Architecture: Transformer Decoder (GPT-style)
- Components:
  - Multi-Head Self-Attention
  - Positional Encoding
  - Layer Normalization
  - Feedforward Layers
- Parameters: ~0.2M
- Training Objective: Next-token prediction using **causal language modeling**
- Optimizer: AdamW
- Loss Function: Cross Entropy Loss

---

## 🏗️ Architecture Inspiration

- **"Attention is All You Need"** (Vaswani et al.)
- **GPT-2/GPT-3** by OpenAI
- **nanoGPT** by Andrej Karpathy

Our implementation mimics the core transformer components, but is scaled down to accommodate smaller compute environments.

---

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/GPT-From-Scratch.git
cd GPT-From-Scratch
```

2. **Create virtual environment (optional)**
```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🧪 Training

```bash
python train.py --epochs 20 --batch_size 64 --lr 3e-4 --block_size 128
```

- `train.py` handles tokenization, model training, checkpoint saving
- Uses PyTorch and runs on CPU or GPU (if available)

---

## 📈 Evaluation & Generation

After training completes:

```bash
python generate.py --checkpoint checkpoints/model.pt --start "ROMEO:" --length 200
```

Output:
```
ROMEO:
What light through yonder window breaks? It is the east...
```

---

## 🗂️ Directory Structure

```
GPT-From-Scratch/
├── data/
│   └── tinyshakespeare.txt
├── model/
│   ├── gpt.py           # Transformer model
│   └── config.py        # Model config
├── train.py             # Training script
├── generate.py          # Text generation script
├── utils.py             # Helper functions
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

Edit `config.py` to adjust model parameters:

```python
vocab_size = 65          # Number of unique tokens
n_embed = 128            # Embedding dimension
n_heads = 4              # Attention heads
n_layers = 4             # Transformer layers
block_size = 128         # Context length
dropout = 0.1
```

---

## 📌 Future Work

- Add training visualization with TensorBoard
- Experiment with larger datasets
- Implement mixed-precision training
- Add tokenizer support for subword units

---

