# microgpt-rs

> Minimal GPT **library** in pure Rust — zero dependencies, based on Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

A complete, from-scratch GPT restructured as a reusable Rust crate. Forward pass, analytical backward pass, Adam optimizer, character-level tokenizer — all in `std` only. Trains on 92 baby names and generates real human names in under a second.

```
microgpt-rs  vocab=27  params=3632  layers=1  embd=16  heads=4
Training 5000 steps

step     0  loss=3.2943  t=0.00s  | m<BOS>kv<BOS>tpl  kygocwfv  vydcdhlm
step  2500  loss=2.1965  t=0.16s  | logan  leo  lagan
step  4999  loss=0.6869  t=0.69s  | naomi  eleanora  ryan

Done in 0.689s
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Library Usage](#library-usage)
- [Architecture](#architecture)
- [Module Map](#module-map)
- [Hyperparameters](#hyperparameters)
- [Performance](#performance)
- [Origin & Bug Fixes](#origin--bug-fixes)
- [Roadmap](#roadmap)
- [License](#license)

---

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.56+ for edition 2021)

### Run the demo

```bash
git clone https://github.com/mon-atelier-ia/microgpt-rs.git
cd microgpt-rs
cargo run --release --bin demo
```

### Run the tests

```bash
cargo test --release
```

---

## Library Usage

microgpt-rs is a library crate — import it in your own project:

```toml
# Cargo.toml
[dependencies]
microgpt-rs = { git = "https://github.com/mon-atelier-ia/microgpt-rs.git" }
```

```rust
use microgpt_rs::config::{ModelConfig, TrainConfig};
use microgpt_rs::data::{build_vocab, tokenize};
use microgpt_rs::inference::generate;
use microgpt_rs::model::Model;
use microgpt_rs::rng::Rng;
use microgpt_rs::train::train_step;

fn main() {
    let mc = ModelConfig::default();
    let tc = TrainConfig::default();
    let mut rng = Rng::new(42);

    let docs = vec!["emma", "olivia", "liam", "noah", "luna"];
    let vocab = build_vocab(&docs);
    let mut model = Model::new(vocab.size(), &mut rng, mc);

    // Train
    for step in 0..tc.n_steps {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        let loss = train_step(&mut model, &tokens, step, &tc);
        if step % 1000 == 0 {
            println!("step {step}  loss={loss:.4}");
        }
    }

    // Generate
    let samples = generate(&model.w, &vocab, &mut rng, &mc, 5);
    println!("{}", samples.join("  "));
}
```

### Public API

| Module | Key exports | Purpose |
|--------|------------|---------|
| `config` | `ModelConfig`, `TrainConfig` | All hyperparameters as configurable structs |
| `model` | `Model`, `Params`, `LayerParams` | Weight storage, gradient buffers, Adam optimizer |
| `train` | `train_step()` | Atomic training step: forward + backward + Adam |
| `inference` | `generate()` | Autoregressive name generation |
| `data` | `Vocab`, `build_vocab()`, `tokenize()` | Character-level tokenizer |
| `rng` | `Rng` | xoshiro128+ PRNG with gaussian & categorical sampling |

---

## Architecture

```
token_id ──→ wte[tok]  ─┐
                         ├──→ x = tok_emb + pos_emb
pos_id   ──→ wpe[pos]  ─┘
                │
      ┌────────▼─────────────────────────────┐
      │       Transformer Block ×1           │
      │                                      │
      │  x_res = x                           │
      │  x = RMSNorm(x)                      │
      │                                      │
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
      │                                      │
      │  x_res = x                           │
      │  x = RMSNorm(x)                      │
      │  x = x @ Wfc1                        │
      │  x = squared_relu(x)                 │
      │  x = x @ Wfc2                        │
      │  x = x_res + x                       │
      └──────────────────────────────────────┘
                │
      logits = x @ Wte^T        (weight-tied with token embeddings)
                │
      probs  = softmax(logits)
                │
      loss   = -log(probs[target]) / seq_len
```

Key design choices (matching Karpathy's implementation):
- **RMSNorm** instead of LayerNorm (no bias, no learnable scale)
- **Squared ReLU** activation (`max(0, x)^2`)
- **Weight tying** — output projection reuses the token embedding matrix
- **Zero-initialized** output projections (Wo, Wfc2) — residual stream starts as identity
- **Analytical gradients** — no autograd graph, hand-derived matrix gradients (~4,580x faster than Python's scalar autograd)

---

## Module Map

```
src/
├── lib.rs           # Crate root — public and internal module declarations
├── config.rs        # ModelConfig (Copy) + TrainConfig
├── model.rs         # Params, LayerParams, Model (weights + grad + Adam state)
├── forward.rs       # Forward pass with KV cache, PosCache, AttnCtx
├── backward.rs      # Analytical gradients through every operation
├── train.rs         # train_step() — atomic forward → backward → Adam
├── inference.rs     # generate() — autoregressive sampling
├── data.rs          # Vocab, build_vocab(), tokenize()
├── rng.rs           # xoshiro128+ PRNG, gaussian, categorical
├── ops.rs           # linear, softmax, rmsnorm + backward ops
└── bin/
    └── demo.rs      # CLI demo using the library's public API

tests/
└── smoke.rs         # Param count, loss decrease, generation output
```

### Internal vs Public

| Visibility | Modules |
|-----------|---------|
| `pub` (library API) | `config`, `model`, `data`, `train`, `inference`, `rng` |
| `pub(crate)` (internal) | `forward`, `backward`, `ops` |

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `n_embd` | 16 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `head_dim` | 4 | Per-head dimension (`n_embd / n_head`) |
| `n_layer` | 1 | Number of transformer blocks |
| `block_size` | 8 | Maximum sequence length |
| `n_steps` | 5000 | Training iterations |
| `lr` | 1e-2 | Initial learning rate |
| `beta1` | 0.9 | Adam first moment decay |
| `beta2` | 0.95 | Adam second moment decay |
| LR schedule | linear decay | `lr * (1 - step/n_steps)` |

All values match Karpathy's defaults. Override via `ModelConfig` and `TrainConfig` structs.

---

## Performance

| Metric | Karpathy Python | microgpt-rs |
|---|---|---|
| 1000-step training time | 297.7s | **0.065s** |
| Full 5000-step run | ~25 min | **< 1s** |
| Speedup | 1x | **~4,580x** |
| Parameters | 3,632 | 3,632 |
| Final loss (5000 steps) | ~2.4 (1000 steps) | **0.69** |

The speedup comes from **analytical matrix-level gradients** replacing Python's scalar autograd. No heap-allocated computation graph — just direct gradient formulas.

---

## Origin & Bug Fixes

This project is a fork of [rustystack/zeroclawgpt](https://github.com/rustystack/zeroclawgpt), restructured from a single 474-line `main.rs` into a modular library crate.

The original zeroclawgpt identified and fixed 5 differences vs Karpathy's gist:

1. **KV Cache — Real Causal Attention** (critical) — v1 processed positions independently; v2 accumulates a growing KV cache so each position attends to all previous ones
2. **Adam beta2: 0.999 → 0.95** — faster adaptation for tiny models
3. **Linear LR Decay** — prevents overshooting near convergence
4. **Zero-Init Output Projections** — Wo and Wfc2 start at zero (GPT-2 technique)
5. **Loss Normalization** — scale gradients by `1/(seq_len-1)` before backward

Full details in [docs/implementation-notes.md](docs/implementation-notes.md).

---

## Roadmap

This crate is the Rust core for the [mon-atelier-ia](https://github.com/mon-atelier-ia) educational GPT ecosystem:

| Target | Approach | Status |
|--------|----------|--------|
| Library restructuring | Split `main.rs` into modules | Done |
| WASM bindings | `wasm-bindgen` / `wasm-pack` sub-crate | Planned |
| Android bindings | UniFFI or JNI sub-crate | Planned |
| French datasets | Port from microgpt-ts-fr | Planned |
| Integration with [microgpt-ts-fr](https://github.com/mon-atelier-ia/microgpt-ts-fr) | WASM replaces TS engine | Planned |
| Integration with [microgpt-visualizer-fr](https://github.com/mon-atelier-ia/microgpt-visualizer-fr) | WASM for fast forward pass | Planned |

See [docs/roadmap.md](docs/roadmap.md) for the full plan.

---

## License

[MIT](LICENSE)

---

<p align="center">
  Fork of <a href="https://github.com/rustystack/zeroclawgpt">rustystack/zeroclawgpt</a>, restructured by <a href="https://github.com/mon-atelier-ia">mon-atelier-ia</a>
</p>
