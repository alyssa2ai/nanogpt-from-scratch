import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters - Optimized for 2-3 hour high-quality CPU training
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000  # Extended training for better convergence
eval_interval = 500  # Show progress every 500 steps
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200  # More thorough evaluation
n_embd = 384  # Full embedding dimension for better quality
n_head = 6    # More attention heads for richer representations
n_layer = 6   # Deeper network for better learning
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        # WHY SCALING: We scale by 1/√d_k to prevent the dot products from growing too large,
        # which would push the softmax into regions with extremely small gradients (saturation).
        # This scaling maintains stable gradients during backpropagation.
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # WHY CAUSAL MASKING: We apply triangular masking to prevent tokens from attending to future positions.
        # This preserves the auto-regressive property: each token can only use information from previous tokens.
        # The mask sets future positions to -inf, which become 0 after softmax.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # WHY SOFTMAX: Converts raw attention scores into a probability distribution (sums to 1).
        # This creates a weighted average where the model learns which tokens are most relevant.
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel 
    
    WHY MULTI-HEAD: Different attention heads can learn to focus on different aspects
    of the input (e.g., syntax vs. semantics, local vs. long-range dependencies).
    This provides the model with richer representational capacity than single-head attention.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # WHY PROJECTION: After concatenating multi-head outputs, we project back to n_embd dimension
        # to mix information across heads and maintain consistent dimensionality.
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ Position-wise Feed-Forward Network
    
    WHY FFN: While attention handles token interactions, the FFN provides additional
    representational capacity for each position independently. The expansion to 4x dimension
    allows the model to learn complex non-linear transformations of the attended information.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand to 4x (standard in Transformers)
            nn.ReLU(),  # Non-linearity for learning complex patterns
            nn.Linear(4 * n_embd, n_embd),  # Project back to original dimension
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation 
    
    WHY THIS ARCHITECTURE:
    - Pre-LayerNorm (normalize before attention/FFN): Improves training stability
    - Residual connections (x + ...): Allows gradients to flow directly through the network,
      preventing vanishing gradients in deep models. The model learns residual functions
      (what to ADD to x) rather than learning the full transformation from scratch.
    """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LN architecture: normalize first, then apply transformation with residual
        x = x + self.sa(self.ln1(x))    # Attention block with residual
        x = x + self.ffwd(self.ln2(x))  # FFN block with residual
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token embeddings: map each character to a learned dense vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # WHY POSITIONAL EMBEDDINGS: Self-attention is permutation-invariant (order-agnostic).
        # We add positional information so the model knows where each token appears in the sequence.
        # Learned embeddings (vs. sinusoidal) allow the model to adapt position encoding to the task.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Project to vocabulary for next-token prediction

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

import time
start_time = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | {elapsed/60:.1f} min elapsed")
    
    # Show quick progress every 50 steps (without expensive eval)
    elif iter % 50 == 0:
        elapsed = time.time() - start_time
        steps_per_sec = iter / elapsed if elapsed > 0 else 0
        eta_seconds = (max_iters - iter) / steps_per_sec if steps_per_sec > 0 else 0
        print(f"step {iter}/{max_iters} | {steps_per_sec:.2f} steps/sec | ETA: {eta_seconds/60:.1f} min")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))