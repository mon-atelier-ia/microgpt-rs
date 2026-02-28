# Roadmap

> Plan d'integration de microgpt-rs dans l'ecosysteme mon-atelier-ia.

## Phase 0 — Library Foundation (Done)

- [x] Evaluate Rust ports of Karpathy's gist (see `gist-conformance.md`)
- [x] Initial restructuring from zeroclawgpt (analytical gradients, fast but unfaithful)
- [x] Migrate to blackopsrepl base (scalar autograd, faithful to gist)
- [x] Archive zeroclawgpt code on `archive/zeroclawgpt` branch
- [x] Modular library: value.rs, config.rs, model.rs, forward.rs, train.rs, inference.rs, data.rs, rng.rs, ops.rs
- [x] Public API: `train_step()`, `generate()`, `build_vocab()`, `tokenize()`, `Value`
- [x] Smoke tests (vocab, tokenizer, lm_head, loss decrease, generation, prefix)
- [x] Dev tooling: rustfmt, clippy, git hooks, rust-analyzer
- [x] Push to mon-atelier-ia/microgpt-rs
- [x] Quick wins: `prefix` param in `generate()`, `n_samples` in `TrainConfig`

## Phase 0.5 — Naming & Architecture Decision (Done)

**Decision**: the final product is **microgpt animé** (`microgpt-anime`).

### Context

| Project | Origin | Role |
|---|---|---|
| `microgpt-rs` | Original (blackopsrepl base) | Rust ML engine (library crate) |
| `microgpt-visualizer-fr` | Fork FR of enescang/microgpt-visualizer | UI source (React/Vite/TS) |
| `microgpt-anime` | **New project** | Rust engine + visualizer UI = standalone product |

### Why a new project?

- `microgpt-rs` = engine only (no UI, reusable library)
- `microgpt-visualizer-fr` = fork localisé (not an original project)
- `microgpt-anime` = original creation combining both — not a fork of anything
- Naming convention: `-fr` suffix = French localization of upstream. This project is not a localization.

### What goes where

| Concern | Lives in |
|---|---|
| Autograd engine, GPT model, training, inference | `microgpt-rs` (this repo) |
| WASM bindings (`wasm-bindgen`) | `microgpt-anime` (depends on `microgpt-rs`) |
| Visualizer UI (React pages, components) | `microgpt-anime` (ported from `microgpt-visualizer-fr`) |
| French datasets | `microgpt-anime` (imported from both TS projects) |
| Android bindings (future) | `microgpt-rs` or `microgpt-anime` TBD |

## Phase 1 — microgpt-anime: WASM + UI

**Goal:** Create `microgpt-anime` — Rust engine compiled to WASM, powering the visualizer UI.

### Steps

1. Create `C:\Dev\microgpt-anime\` project
   - Cargo workspace: WASM crate depends on `microgpt-rs`
   - Vite/React frontend (ported from `microgpt-visualizer-fr`)
2. WASM bindings (`wasm-bindgen`):
   - `create_model(vocab_size, config_json) -> ModelHandle`
   - `train_step(handle, tokens) -> f64` (returns loss)
   - `generate(handle, n_samples, temperature, prefix) -> Vec<String>`
   - `get_activations(handle) -> JSON` (for visualizer pages)
3. Wire UI to WASM:
   - Replace TS engine calls with WASM calls
   - Keep all React pages/components from visualizer
4. French datasets embedded in WASM crate

### Technical Notes

- Zero crate deps in `microgpt-rs` core = trivial WASM compilation
- `Rc<RefCell<>>` compiles to WASM (no GC, reference counting in linear memory)
- `wasm-bindgen` only in the WASM crate, not in `microgpt-rs`
- UI params already covered: see `ui-parameters-audit.md`

### Expected Outcome

- Training speed: ~8x faster than current TS engine in browser
- Same pedagogical UI (tokenizer, embeddings, forward pass, training, inference pages)
- Autograd `Value` graph available for visualization

## Phase 2 — Android (Future)

**Goal:** Expose the Rust core to Kotlin for an Android app.

- UniFFI or JNI bindings
- Jetpack Compose UI
- Depends on Phase 1 proving the architecture

## Architecture Target

```
microgpt-rs/                    # This repo — engine library
├── src/                        # Core library (zero deps)
│   ├── lib.rs, value.rs, config.rs, model.rs
│   ├── forward.rs, train.rs, inference.rs
│   ├── data.rs, rng.rs, ops.rs
│   └── bin/demo.rs
└── tests/smoke.rs

microgpt-anime/                 # New repo — standalone product
├── crates/
│   └── wasm/                   # wasm-bindgen, depends on microgpt-rs
│       ├── Cargo.toml
│       └── src/lib.rs
├── src/                        # React/Vite frontend
│   ├── components/             # From microgpt-visualizer-fr
│   ├── pages/                  # Tokenizer, Embeddings, ForwardPass, Training, Inference
│   └── App.tsx
├── datasets/                   # French + English datasets
├── package.json
└── vite.config.ts
```

## References

- [Karpathy's microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [blackopsrepl's microgpt.rs](https://gist.github.com/blackopsrepl/bf7838f8f365c77e36075ca301db298e)
- [dubzdubz/microgpt-ts](https://github.com/dubzdubz/microgpt-ts) — upstream of microgpt-ts-fr
- [enescang/microgpt-visualizer](https://github.com/enescang/microgpt-visualizer) — upstream of microgpt-visualizer-fr
- [wasm-pack](https://rustwasm.github.io/wasm-pack/)
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/)
