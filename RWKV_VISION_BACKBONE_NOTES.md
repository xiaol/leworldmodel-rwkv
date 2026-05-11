# RWKV Vision Backbone Notes

## Current LeWM Setup

The current LeWM architecture uses a ViT encoder and a latent dynamics
predictor:

```text
pixels -> ViT-tiny encoder -> latent embeddings -> predictor
```

The RWKV experiments so far replace only the predictor:

```text
pixels -> ViT-tiny encoder -> latent embeddings -> RWKV predictor
```

Replacing the ViT backbone would be a separate architecture experiment:

```text
pixels -> RWKV vision encoder -> latent embeddings -> predictor
```

## ViT Attention vs RWKV

The transformer inside ViT is not causal. It uses full bidirectional
self-attention over image patch tokens:

```text
patch tokens -> full self-attention
```

That means every patch can attend to every other patch in the same layer.

Plain RWKV is naturally sequential and causal. In a left-to-right patch scan,
patch `i` only sees previous patches. To approximate ViT-style global patch
mixing with RWKV, we need a vision-specific design.

## Ways To Give RWKV Global Image Context

### 1. Bidirectional RWKV

Run RWKV in both directions over patch tokens and combine the outputs:

```text
forward scan:  patch i sees patches <= i
backward scan: patch i sees patches >= i
combine:       patch i gets both directions
```

This is the simplest way to make every patch receive information from all
patches after one bidirectional block.

### 2. Multi-Direction 2D Scans

For images, scan patches in multiple spatial orders:

```text
left -> right
right -> left
top -> bottom
bottom -> top
```

Then fuse the scan outputs. This better respects the 2D structure of images
than a single 1D patch order.

### 3. Global Memory Tokens

Add a small number of learned global tokens. These act as fixed-size global
memory for the image.

A useful two-pass causal RWKV pattern is:

```text
[patches, globals] -> globals read patches
[globals', patches] -> patches read globals
```

This lets global tokens collect image-wide information first, then lets patches
read the updated global memory.

### 4. Row/Column RWKV

Alternate row and column scans:

```text
row scan mixes horizontal context
column scan mixes vertical context
stacked layers spread global context
```

This is efficient and image-aware, but global communication may take multiple
layers.

### 5. Hybrid Conv + RWKV

Use convolution for local spatial mixing and RWKV for long-range/global mixing.
This is often more practical than replacing ViT attention with a naive RWKV
scan.

## Global Memory Tokens

Global tokens are trainable parameters, similar to ViT class tokens:

```python
self.global_tokens = nn.Parameter(torch.empty(1, num_global_tokens, dim))
nn.init.trunc_normal_(self.global_tokens, std=0.02)
```

At runtime they are expanded per batch:

```python
globals = self.global_tokens.expand(batch_size, -1, -1)
```

The tokens are learned by backpropagation together with the rest of the model.
Gradients flow from the LeWM prediction loss into the encoder, RWKV blocks, and
the global token parameters.

## Choosing The Number Of Global Tokens

For the current LeWM ViT setup:

```text
image size: 224 x 224
patch size: 14 x 14
patch grid: 16 x 16
patch tokens: 256
hidden dim: 192
```

The number of global tokens `M` controls the global information bottleneck.

| Global tokens | Typical use |
| ---: | --- |
| 1 | CLS-token-like, very compressed |
| 4 | tiny global memory |
| 8 | good first baseline |
| 16 | stronger global memory |
| 32 | less bottleneck, more compute |
| 64 | likely too large for first run |

A practical first sweep:

```text
M = 4, 8, 16, 32
```

The parameter cost is tiny:

```text
params = M * hidden_dim
```

With `hidden_dim = 192`:

| M | Extra params |
| ---: | ---: |
| 8 | 1,536 |
| 16 | 3,072 |
| 32 | 6,144 |
| 64 | 12,288 |

The main cost is compute from the extra tokens, not parameter count.

## Initialization

Recommended initialization:

```python
self.global_tokens = nn.Parameter(torch.empty(1, M, D))
nn.init.trunc_normal_(self.global_tokens, std=0.02)
```

Zero initialization is possible:

```python
self.global_tokens = nn.Parameter(torch.zeros(1, M, D))
```

But with multiple global tokens, random or truncated-normal initialization is
usually better because tokens can specialize earlier.

## Suggested Experiment Matrix

To separate predictor effects from backbone effects, compare:

| Encoder | Predictor | Purpose |
| --- | --- | --- |
| ViT | Transformer | original baseline |
| ViT | RWKV | predictor-only test |
| RWKV vision encoder | Transformer | backbone-only test |
| RWKV vision encoder | RWKV | full RWKV stack |

This avoids confusing predictor improvements with encoder changes.
