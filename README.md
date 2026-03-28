# microgpt-rs

> Minimal GPT **library** in pure Rust — zero dependencies, faithful port of Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) with scalar autograd.

A complete, from-scratch GPT restructured as a reusable Rust crate. Scalar autograd engine, forward pass, Adam optimizer, character-level tokenizer — all in `std` only.

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
    let mut model = Model::new(vocab.size(), &mut rng, mc, &tc);

    // Train
    for step in 0..tc.n_steps {
        let doc = docs[step % docs.len()];
        let tokens = tokenize(doc, &vocab, mc.block_size);
        let loss = train_step(&mut model, &tokens, step, &tc);
        if step % 100 == 0 {
            println!("step {step}  loss={loss:.4}");
        }
    }

    // Generate
    let samples = generate(&model.sd, &vocab, &mut rng, &mc, 5, tc.temperature, "");
    println!("{}", samples.join("  "));
}
```

### Public API

| Module | Key exports | Purpose |
|--------|------------|---------|
| `value` | `Value` | Scalar autograd engine (computation graph + backward) |
| `config` | `ModelConfig`, `TrainConfig` | All hyperparameters as configurable structs |
| `model` | `Model`, `StateDict`, `LayerWeights` | Weight matrices of `Value` nodes, Adam optimizer |
| `train` | `train_step()` | Atomic training step: forward + backward + Adam |
| `inference` | `generate()` | Autoregressive generation with temperature |
| `data` | `Vocab`, `build_vocab()`, `tokenize()` | Character-level tokenizer |
| `rng` | `Rng` | Xorshift64 PRNG with gaussian & categorical sampling |

---

## Architecture

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
      │  x = relu(x)                         │
      │  x = x @ Wfc2                        │
      │  x = x_res + x                       │
      └──────────────────────────────────────┘
                │
      logits = x @ lm_head                   (separate matrix, no weight tying)
                │
      probs  = softmax(logits)
                │
      loss   = -log(probs[target]) / seq_len
```

Key design choices (matching Karpathy's gist):
- **Scalar autograd** — `Value` nodes with `Rc<RefCell<>>`, reverse-mode backward()
- **RMSNorm** instead of LayerNorm (no bias, no learnable scale)
- **ReLU** activation in MLP
- **Separate lm_head** — no weight tying with token embeddings
- **All weights at std=0.08** — no special zero-init
- **BOS only** — same token for start and end of sequence

---

## Module Map

```
src/
├── lib.rs           # Crate root — public and internal module declarations
├── value.rs         # Scalar autograd engine (Value + backward)
├── config.rs        # ModelConfig (Copy) + TrainConfig
├── model.rs         # StateDict, LayerWeights, Model (Value matrices + Adam)
├── forward.rs       # Forward pass with KV cache
├── train.rs         # train_step() — forward → backward → Adam
├── inference.rs     # generate() — autoregressive sampling with temperature
├── data.rs          # Vocab, build_vocab(), tokenize()
├── rng.rs           # Xorshift64 PRNG, gaussian, categorical
├── ops.rs           # linear, softmax, rmsnorm on Vec<Value>
└── bin/
    └── demo.rs      # CLI demo using the library's public API

tests/
└── smoke.rs         # 5 smoke tests
```

### Internal vs Public

| Visibility | Modules |
|-----------|---------|
| `pub` (library API) | `value`, `config`, `model`, `data`, `train`, `inference`, `rng` |
| `pub(crate)` (internal) | `forward`, `ops` |

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `n_embd` | 16 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `head_dim` | 4 | Per-head dimension (`n_embd / n_head`) |
| `n_layer` | 1 | Number of transformer blocks |
| `block_size` | 16 | Maximum sequence length |
| `n_steps` | 1000 | Training iterations |
| `lr` | 0.01 | Initial learning rate |
| `beta1` | 0.85 | Adam first moment decay |
| `beta2` | 0.99 | Adam second moment decay |
| `std_init` | 0.08 | Weight initialization std |
| `temperature` | 0.5 | Inference temperature |
| `n_samples` | 5 | Number of samples to generate |
| LR schedule | linear decay | `lr * (1 - step/n_steps)` |

All values match Karpathy's gist. Override via `ModelConfig` and `TrainConfig` structs.

---

## Performance

| Metric | Karpathy Python | microgpt-rs |
|---|---|---|
| Gradient approach | Scalar autograd (`Value`) | Scalar autograd (`Value`) |
| Speedup vs Python | 1x | **~8x** |
| Dependencies | 0 | 0 |

The ~8x speedup comes from Rust's compiled performance vs Python's interpreter overhead, with the same algorithmic approach (scalar autograd, `Rc<RefCell<>>` computation graph).

---

## Origin

Faithful Rust port of Karpathy's [microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), restructured as a modular library crate. See [docs/gist-conformance.md](docs/gist-conformance.md) for the conformance audit.

---

## Roadmap

This crate is the Rust core for the [mon-atelier-ia](https://github.com/mon-atelier-ia) educational GPT ecosystem:

| Project | Language | Role | Live |
|---------|----------|------|------|
| [microgpt-rs](https://github.com/mon-atelier-ia/microgpt-rs) | Rust | Core engine (this repo) | -- |
| [microgpt-ts-fr](https://github.com/mon-atelier-ia/microgpt-ts-fr) | TypeScript | Training playground, French datasets | [Demo](https://microgpt-ts-fr.vercel.app) |
| [microgpt-visualizer-fr](https://github.com/mon-atelier-ia/microgpt-visualizer-fr) | TypeScript | Transformer pipeline visualizer | [Demo](https://microgpt-visualizer-fr.vercel.app) |
| [microgpt-lab](https://github.com/mon-atelier-ia/microgpt-lab) | TypeScript | Interactive lab environment | [Demo](https://microgpt-lab.vercel.app) |
| [microgpt-anime](https://github.com/mon-atelier-ia/microgpt-anime) | TypeScript | Animated explainer | -- |

| Target | Approach | Status |
|--------|----------|--------|
| Library restructuring | Modular crate from gist | Done |
| Gist conformance | Scalar autograd, faithful to gist | Done |
| WASM bindings | `wasm-bindgen` / `wasm-pack` sub-crate | Planned |
| Android bindings | UniFFI or JNI sub-crate | Planned |
| French datasets | Port from microgpt-ts-fr | Planned |
| Integration with [microgpt-ts-fr](https://github.com/mon-atelier-ia/microgpt-ts-fr) | WASM replaces TS engine | Planned |
| Integration with [microgpt-visualizer-fr](https://github.com/mon-atelier-ia/microgpt-visualizer-fr) | WASM replaces TS engine | Planned |

See [docs/roadmap.md](docs/roadmap.md) for the full plan.

---

## License

[MIT](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/mon-atelier-ia">mon-atelier-ia</a> — based on <a href="https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95">Karpathy's microgpt.py</a>
</p>
