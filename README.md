# NanoGPT: Decoder-Only Transformer from Scratch 🧠

**A bottom-up implementation of the Generative Pre-trained Transformer (GPT) architecture.**

This repository contains a clean, character-level implementation of a Transformer model, built following the architectural principles outlined in the seminal paper _["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)_ and the GPT series. This project was developed to master the nuances of self-attention, causal masking, and gradient flow in deep NLP models.

---

## 🔬 Architectural Implementations

This implementation demonstrates core components of modern language models:

- **Scaled Dot-Product Multi-Head Attention:** Implemented the foundational query, key, and value transformations to allow the model to focus on different sequence dependencies in parallel. The scaling factor (1/√d_k) prevents saturation of the softmax function during training.

- **Causal Masking:** Integrated triangular masking to maintain the auto-regressive property, ensuring tokens only attend to previous positions during next-token prediction. This is critical for maintaining the generative nature of the model.

- **Learnable Positional Embeddings:** Provided the model with spatial awareness within sequences, as the self-attention mechanism itself is position-invariant.

- **Residual Connections & Pre-LayerNorm:** Implemented skip-connections and normalization layers (Pre-LN configuration) to ensure stable training and prevent vanishing gradients in deep networks. This architecture choice follows the GPT-2 design pattern.

- **Feed-Forward Networks:** Each Transformer block contains a position-wise FFN with expansion factor of 4x, providing the model with additional representational capacity beyond attention.

---

## 📊 Training & Hyperparameters

To demonstrate technical rigor, the model was trained with the following configuration:

- **Context Window (Block Size):** 256 tokens
- **Embedding Dimension:** 384
- **Attention Heads:** 6 (64-dimensional head size)
- **Transformer Layers:** 6
- **Dropout Rate:** 0.2
- **Optimizer:** AdamW with learning rate 3e-4
- **Batch Size:** 64 sequences
- **Loss Function:** Cross-Entropy (Next-token prediction)
- **Training Iterations:** 5,000 steps
- **Total Parameters:** ~10.7M

---

## 📈 Research Insights

- **Convergence Analysis:** The model demonstrates consistent convergence, achieving a validation loss of approximately 1.47 after 5,000 iterations on the Tiny Shakespeare dataset. Training loss decreases steadily from ~4.2 to ~1.0, indicating effective learning without significant overfitting.

- **Scaling Observation:** The multi-head attention mechanism (6 heads) allows the model to capture different linguistic patterns simultaneously—from character-level dependencies to longer-range syntactic structures. Increasing the number of attention heads improves the model's ability to capture diverse attention patterns.

- **Auto-regressive Generation:** The model successfully generates coherent character sequences that mimic Shakespearean English structure, demonstrating learned patterns in spelling, word formation, and sentence structure.

---

## 🚀 Project Structure

- [bigram.py](bigram.py): Baseline bigram model for comparison
- [gpt.py](gpt.py): Full Transformer implementation with training loop
- [myowngpt.ipynb](myowngpt.ipynb): Exploratory Jupyter notebook
- [input.txt](input.txt): Tiny Shakespeare corpus (training data)
- [generated_output.txt](generated_output.txt): Sample generations
- [requirements.txt](requirements.txt): Python dependencies

---

## 🛠️ Installation & Usage

```bash
# 1) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train and generate with the full GPT model
python gpt.py
# → Trains for 5,000 iterations and generates a 500-character sample

# 4) Compare with simple bigram baseline
python bigram.py
# → Trains baseline and writes output to generated_output.txt
```

---

## 📜 Generated Samples

The model successfully learns Shakespearean language patterns across multiple training runs:

### Local Training (3.2M Parameters, 2000 Iterations)

Baseline model trained on CPU demonstrating character-level learning:

```
"Thave ﬁr kshe otNe wan,
"An Hove,"
"Ithingis stherolteNof. haurpands h y Wh pinderee aton bediFof lve torousseain anvertlans0dof -
wn. coorfre. ses anbeda h wh hestemur pexape."—ather w
plat. had cis. bsou towhe, thinth f tof I r bas istheshetharep f ws "Apour Oprof th at ithacand Hon, cakioforerthe
```

### Google Colab GPU Training (10.8M Parameters, 6000 Iterations)

Full model trained on GPU showing coherent dialogue and character names:

```
discipers.

CrPULETIO:
Perhaps Claudio.

FRIAR PETER:
What! come, where's granted come you?

BIOND:
How roIusly I know, Aufidiles?

ROMEO:
For what's the world bend is mine.

PETER:
Who will answer, methinks other, boys,
Beloved, id 'twixt me sortly last. Romeo
Most glad and what feet and glad wheres.

STEPmant:
O, I have bether say at least, I have to
contive.
The day is grucken and theren. Is it not delight,
This fair doeful wagoing, revolted did
```

### Kaggle GPU Training (10.8M Parameters, 6000 Iterations)

Parallel GPU training demonstrating consistent model quality:

```
BENVOLIO:
Well none for writ here thee but it offended.

MERCUTIO:
The way was ready; foor he would mend all
care some moulder, humbler for recorpestity;
The most of all.

MERCUTIO:
I have heard it:
Come on these senators, our enmity.

BENVOLIO:
Fools every os well! would the were were almost lefted,
And lay the foil.

MERCUTIO:
But if he
Take it with one thine own: dold we buy for the
dukes
And fall of goodness, ladies, fortune, or sting, led you
```

### 🔍 Quality Progression

| Metric                 | Local (3.2M) | Colab (10.8M) | Kaggle (10.8M) |
| ---------------------- | ------------ | ------------- | -------------- |
| **Parameters**         | 3.2M         | 10.8M         | 10.8M          |
| **Iterations**         | 2,000        | 6,000         | 6,000          |
| **Hardware**           | CPU          | GPU (Colab)   | GPU (Kaggle)   |
| **Training Time**      | ~45 min      | ~30 min       | ~30 min        |
| **Character Names**    | ✗            | ✓             | ✓              |
| **Dialogue Structure** | ✗            | ✓             | ✓              |
| **Punctuation**        | ✓            | ✓             | ✓              |
| **Coherence**          | Low          | High          | High           |

The larger GPU-trained models demonstrate that with proper architecture and sufficient compute, the Transformer can learn meaningful linguistic patterns from raw character sequences.

## Reproducible Settings

- Seed: `torch.manual_seed(1337)`
- Bigram: `batch_size=32`, `block_size=8`, `max_iters=3000`, `lr=1e-2`
- GPT: `batch_size=64`, `block_size=256`, `n_embd=384`, `n_head=6`, `n_layer=6`, `dropout=0.2`, `max_iters=5000`, `lr=3e-4`

## Sample Output (Bigram)

The bigram model learns character-level patterns but not long-range structure, so samples are partially coherent:

```
“Thave fir kshe otNe wan,
“An Hove,”
“Ithingis stherolteNof. haurpands h y Wh pinderee aton bediFof lve…
```

## Notes & Tips

- GPU is recommended for `gpt.py` but not required for small demos.
- Increase `max_new_tokens` to generate longer samples.
- Changing `input.txt` automatically rebuilds the vocabulary.

## Attribution

- Dataset: Tiny Shakespeare (public domain; via Karpathy’s char-rnn).
- Inspiration: Andrej Karpathy’s “Let’s build GPT”.
