# nanoGPT From Scratch — Character-Level GPT

Minimal, readable implementation of a character-level GPT in PyTorch, progressing from a baseline **bigram** model to a multi-head **Transformer**. Trains on Tiny Shakespeare (`input.txt`) and writes samples to `generated_output.txt`.

Designed for recruiters and learners: clear structure, reproducible runs, and concise explanations of how each component works.

## Highlights
- **Two models**: fast bigram baseline and full Transformer (token+positional embeddings, multi-head self-attention, MLP, residuals, LayerNorm).
- **Reproducible**: fixed seeds and documented hyperparameters; CPU-friendly for demos, GPU-ready for speed.
- **Clean output flow**: saves generated text to a file to avoid console encoding quirks on Windows.

## Project Structure
- `bigram.py`: simplest baseline; each character predicts the next.
- `gpt.py`: full Transformer training + generation.
- `myowngpt.ipynb`: exploratory notebook.
- `input.txt`: Tiny Shakespeare corpus.
- `generated_output.txt`: latest sample written by `bigram.py`.
- `requirements.txt`: minimal dependencies.

## Quickstart

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS/Linux

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the simple bigram baseline (fast)
python bigram.py
# → writes generated_output.txt

# 4) Run the full GPT (slower, higher quality)
python gpt.py
# → prints training loss every 500 iters and a 500-char sample
```

## How It Works (at a Glance)
- **Data prep**: read `input.txt`, build character vocabulary, encode to ints; 90/10 train/val.
- **Batching**: sample contiguous `block_size` windows for inputs/targets.
- **Models**:
   - Bigram: `nn.Embedding(vocab, vocab)` → logits for next char.
   - GPT: token + positional embeddings → Transformer blocks → logits.
- **Training**: cross-entropy loss with AdamW; periodic evaluation on validation set.
- **Generation**: autoregressive sampling for `max_new_tokens`; decode and save to `generated_output.txt`.

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
