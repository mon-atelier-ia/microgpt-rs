# Architecture — microgpt-rs

> **LLM instruction**: Use this document as a reference for project structure, file roles, and data flow. When modifying code, verify your changes are consistent with the architecture described here. If you add files or change responsibilities, update this document.

## Overview

microgpt-rs is a **library crate** — a minimal GPT engine in pure Rust with zero dependencies, faithful to [Karpathy's microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). Scalar autograd engine, forward pass, Adam optimizer, character-level tokenizer — all in `std` only.

Two design principles:
1. **Karpathy-faithful**: architecture, hyperparameters, and numerical behavior match the gist (see `gist-conformance.md`)
2. **Zero dependencies**: the entire GPT is visible, no external crates

## Module Map

```
src/
├── lib.rs           # Crate root — public and internal module declarations
├── value.rs         # Scalar autograd engine (Value + backward)
├── config.rs        # ModelConfig (Copy) + TrainConfig
├── model.rs         # StateDict, LayerWeights, Model (Value matrices + Adam)
├── forward.rs       # Forward pass with KV cache (pub(crate))
├── train.rs         # train_step() — forward → backward → Adam
├── inference.rs     # generate() — autoregressive sampling with temperature
├── data.rs          # Vocab, build_vocab(), tokenize()
├── rng.rs           # Xorshift64 PRNG, gaussian, categorical
├── ops.rs           # linear, softmax, rmsnorm on Vec<Value> (pub(crate))
└── bin/
    └── demo.rs      # CLI demo using the library's public API

tests/
└── smoke.rs         # 5 smoke tests
```

### Visibility Convention

| Visibility | Modules | Rule |
|-----------|---------|------|
| `pub` (library API) | `value`, `config`, `model`, `data`, `train`, `inference`, `rng` | Stable, public-facing |
| `pub(crate)` (internal) | `forward`, `ops` | Free to refactor without breaking consumers |

## Forward Pass

```
token_id ──→ wte[tok]  ─┐
                         ├──→ x = tok_emb + pos_emb
pos_id   ──→ wpe[pos]  ─┘
                │
         x = RMSNorm(x)
                │
      ┌────────▼─────────────────────────────┐
      │       Transformer Block ×1           │
      │                                      │
      │  x_res = x                           │
      │  x = RMSNorm(x)                      │
      │  Q = x @ Wq                          │
      │  K = x @ Wk  ──→ append to KV cache  │
      │  V = x @ Wv  ──→ append to KV cache  │
      │                                      │
      │  ┌─ For each of 4 heads: ──────────┐ │
      │  │  scores = Q·K^T / √d_head       │ │
      │  │  weights = softmax(scores)       │ │
      │  │  head_out = weights · V          │ │
      │  └─────────────────────────────────┘ │
      │                                      │
      │  attn_out = concat(heads) @ Wo       │
      │  x = x_res + attn_out               │
      │  x_res = x                           │
      │  x = RMSNorm(x)                      │
      │  x = x @ Wfc1 → relu → x @ Wfc2     │
      │  x = x_res + x                       │
      └──────────────────────────────────────┘
                │
      logits = x @ lm_head  (separate matrix, no weight tying)
                │
      probs  = softmax(logits)
                │
      loss   = -log(probs[target]) / seq_len
```

## Key Design Choices (matching Karpathy's gist)

- **Scalar autograd** — `Value` nodes with `Rc<RefCell<>>`, reverse-mode `backward()`
- **RMSNorm** instead of LayerNorm (no bias, no learnable scale)
- **ReLU** activation in MLP
- **Separate lm_head** — no weight tying with token embeddings
- **All weights at std=0.08** — no special zero-init
- **BOS only** — same token for start and end of sequence
- **f64 throughout** — matches Python's float precision
- **`ModelConfig` is `Copy`** — avoids borrow conflicts when `&mut Model` is in use

## Data Flow

```
Training:
  docs (string[]) → build_vocab() → Vocab
  doc (string) → tokenize(doc, vocab, block_size) → token_ids
  train_step(&mut model, &tokens, step, &config) → loss: f64
    → forward (builds Value graph with KV cache)
    → backward (reverse-mode autograd through graph)
    → Adam update (all params with lr decay)

Inference:
  generate(&state_dict, &vocab, &mut rng, &config, n, temp, prefix) → Vec<String>
    → loop: compute logits → softmax(logits/temp) → sample → append
    → stop on BOS token or max length
```

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| n_embd | 16 | Embedding dimension |
| n_head | 4 | Attention heads |
| n_layer | 1 | Transformer blocks |
| block_size | 16 | Max sequence length |
| lr | 0.01 | Learning rate (linear decay to 0) |
| beta1/beta2 | 0.85/0.99 | Adam moments |
| std_init | 0.08 | Weight init std |
| temperature | 0.5 | Inference temperature |

All values match Karpathy's gist. Override via `ModelConfig` and `TrainConfig` structs.

## Origin & Conformance

Rebased on [blackopsrepl's microgpt.rs](https://gist.github.com/blackopsrepl/bf7838f8f365c77e36075ca301db298e) — 8/8 conformant with Karpathy's gist. Previous base (zeroclawgpt) archived on `archive/zeroclawgpt` branch (0/8 conformant, analytical gradients, 4580x faster but unfaithful). See `gist-conformance.md` for the full audit.
