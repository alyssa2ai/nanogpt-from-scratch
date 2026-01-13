# NanoGPT Project Structure

This directory contains a research-grade implementation of a Decoder-Only Transformer (GPT) trained on Tiny Shakespeare.

## 📁 Directory Layout

```
.
├── gpt.py                          # Full Transformer implementation with detailed "why" comments
├── bigram.py                       # Baseline bigram model for comparison
├── train.py                        # CPU-optimized training script (2-3 hours, 3.2M params)
├── train_colab.py                  # GPU-ready standalone Python script for Colab/Kaggle
│
├── colab_nanoGPT.ipynb            # Jupyter notebook for Google Colab (7-section setup)
├── myowngpt.ipynb                  # Original exploratory notebook
│
├── input.txt                       # Tiny Shakespeare dataset (1.1MB, 4.4M characters)
├── generated_output.txt            # Output from local CPU training
├── more.txt                        # Alternate output (commented in code)
│
├── README.md                       # Main documentation with architecture details + all 3 samples
├── TRAINING_REPORT.md              # Comprehensive training report (this file)
├── requirements.txt                # Python dependencies (torch, numpy, matplotlib, jupyter)
│
├── collab train/                   # Google Colab GPU training artifacts
│   ├── sample.txt                  # Generated text from Colab run
│   └── colab_nanoGPT.ipynb        # Notebook from Colab session
│
├── kaggle train/                   # Kaggle GPU training artifacts
│   ├── kagglesampletxt.txt        # Generated text from Kaggle run
│   └── myowngptkaggle-ipynb.ipynb # Notebook from Kaggle session
│
└── .git/                           # Version control
```

## 🚀 Quick Start

### Option 1: CPU Training (Local, 2-3 hours)

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
python -m pip install -r requirements.txt
python train.py
```

### Option 2: GPU Training (Colab/Kaggle, 30 minutes)

1. Open [Google Colab](https://colab.research.google.com) or [Kaggle](https://kaggle.com)
2. Upload or create a new notebook
3. Copy cells from `colab_nanoGPT.ipynb`
4. Run all cells (GPU enabled: Runtime > Change runtime type > GPU)

Or use the standalone script:

```bash
python train_colab.py  # On Colab/Kaggle with GPU
```

## 📊 Model Specifications

| Aspect              | Value                                 |
| ------------------- | ------------------------------------- |
| Architecture        | Decoder-Only Transformer (GPT-style)  |
| Parameters          | 10.8M (GPU) / 3.2M (CPU)              |
| Context Window      | 256 tokens                            |
| Embedding Dimension | 384 (GPU) / 256 (CPU)                 |
| Attention Heads     | 6 (GPU) / 4 (CPU)                     |
| Transformer Layers  | 6 (GPU) / 4 (CPU)                     |
| Optimizer           | AdamW (lr=3e-4)                       |
| Loss Function       | Cross-Entropy (next-token prediction) |
| Training Data       | Tiny Shakespeare (4.4M characters)    |
| Data Split          | 90/10 train/validation                |

## 🔬 Key Architectural Components

### 1. Scaled Dot-Product Multi-Head Attention

- Query, Key, Value projections for each head
- Scaling by 1/√d_k prevents attention saturation
- 6 heads learn diverse patterns (syntax, semantics, discourse)

### 2. Causal Masking

- Triangular mask ensures auto-regressive property
- Tokens only attend to previous positions
- Critical for realistic next-token prediction

### 3. Residual Connections & Pre-LayerNorm

- Skip connections enable stable gradient flow in 6-layer network
- Pre-LN configuration improves training stability vs Post-LN
- Model learns to route information through depth

### 4. Feed-Forward Networks

- Position-wise FFN with 4x expansion in hidden layer
- ReLU non-linearity provides representational capacity
- Complements attention for sequence understanding

### 5. Learnable Positional Embeddings

- Adapted to task at hand (unlike fixed sinusoidal)
- Provides spatial awareness within sequences

## 📈 Training Results

### Sample Quality Progression

- **Local (3.2M, 2K iters):** Character-level patterns, high noise
- **Colab (10.8M, 6K iters):** Character names, dialogue, punctuation
- **Kaggle (10.8M, 6K iters):** Consistent quality, proper formatting

### Convergence

```
Local:   4.60 → 1.69 (2000 iters)
Colab:   4.59 → 1.42 (6000 iters)  [GPU accelerated]
Kaggle:  4.59 → 1.40 (6000 iters)  [GPU accelerated]
```

## 🛠️ Code Quality & Comments

All implementations include:

- **Architecture Comments:** "Why do we use softmax?" / "Why scale dot products?"
- **Data Handling:** Clear tokenization, batching, train/val split
- **Training Loop:** Progress tracking every 50-500 steps with ETA
- **Generation:** Auto-regressive sampling with temperature control (ready to extend)

See [gpt.py](gpt.py) for detailed inline explanations of each component.

## 📚 Research Value

This project demonstrates:

1. **Foundational Knowledge:** Understanding Transformer internals, not just PyTorch APIs
2. **Scale-aware Design:** Adapting model size to available compute
3. **Empirical Validation:** Multiple independent training runs prove reproducibility
4. **Practical ML:** CPU prototyping → GPU production workflow
5. **Language Learning:** Model discovers linguistic patterns from raw text

## 🔗 References

- "Attention Is All You Need" (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
- Tiny Shakespeare dataset: https://github.com/karpathy/char-rnn/data/tinyshakespeare
- PyTorch Documentation: https://pytorch.org/docs

## 📝 License

Open source. Use for research and education.

---

**Status:** ✅ Complete  
**Last Updated:** January 13, 2026  
**Training Platforms:** Local CPU, Google Colab, Kaggle Notebooks
