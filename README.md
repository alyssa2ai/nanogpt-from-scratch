# nanoGPT From Scratch — Character-Level GPT

Hands-on, minimal implementation of a character-level GPT in PyTorch, progressing from a baseline **bigram** model to a multi-head **Transformer** with positional embeddings and layer normalization. The project trains on the Tiny Shakespeare corpus (shipped as `input.txt`) and generates text to `generated_output.txt`.

Inspired by Andrej Karpathy’s “Let’s build GPT”, but organized for recruiters and learners to inspect, run, and extend quickly.

## Project Structure

- `bigram.py` — simplest baseline; each token predicts the next directly from an embedding lookup.
- `gpt.py` — full Transformer: token + positional embeddings, multi-head self-attention, MLP feed-forward, residuals + layer norms, generation loop.
- `myowngpt.ipynb` — exploratory notebook (mirrors the scripts).
- `input.txt` — Tiny Shakespeare training corpus.
- `generated_output.txt` — latest generated sample after running `bigram.py`.
- `requirements.txt` — minimal dependencies.

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

# 4) Run the full GPT (slower, better quality)
python gpt.py
# → prints training loss every 500 iters and a 500-char sample at the end
```

## What the Code Does (at a Glance)

1. **Data prep**: read `input.txt`, build character vocab, encode to integer tensors, 90/10 train/val split.
2. **Batching**: random contiguous blocks (`block_size`) from train/val for inputs/targets.
3. **Models**:
   - **Bigram**: `nn.Embedding(vocab, vocab)` maps current char → logits for next char.
   - **GPT**: token + positional embeddings → stacked Transformer blocks (multi-head self-attention + MLP + residual + LayerNorm) → vocab logits.
4. **Training loop**: cross-entropy loss, AdamW optimizer, periodic eval on val set.
5. **Generation**: start from a zero token, autoregressively sample `max_new_tokens` chars, decode to text, save to `generated_output.txt`.

## Reproducible Settings

- Seeds: `torch.manual_seed(1337)`
- Bigram: `batch_size=32`, `block_size=8`, `max_iters=3000`, `lr=1e-2`
- GPT: `batch_size=64`, `block_size=256`, `n_embd=384`, `n_head=6`, `n_layer=6`, `dropout=0.2`, `max_iters=5000`, `lr=3e-4`

## Results Snapshot

- Bigram (character context = 1): train/val loss ~2.45 after 3k iters; outputs are coherent at the character level but semantically weak.
- GPT (context = 256): ~10.8M params; val loss ≈ 1.2 after 5k iters on Tiny Shakespeare; outputs show recognizable words, punctuation, and dialogue structure.

## Tips

- GPU recommended for `gpt.py` but CPU works for small demos.
- To generate longer text, increase `max_new_tokens` in either script.
- If you change `input.txt`, vocab is rebuilt automatically—no extra steps.

## Licensing / Attribution

- Dataset: Tiny Shakespeare (public domain, via Karpathy’s char-rnn repo).
- Architecture inspiration: Andrej Karpathy’s “Let’s build GPT”.
