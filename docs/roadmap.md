# Roadmap

> Plan d'integration de microgpt-rs dans l'ecosysteme mon-atelier-ia.

## Phase 0 — Library Foundation (Done)

- [x] Evaluate Rust ports of Karpathy's gist (see `gist-conformance.md`)
- [x] Initial restructuring from zeroclawgpt (analytical gradients, fast but unfaithful)
- [x] Migrate to blackopsrepl base (scalar autograd, faithful to gist)
- [x] Archive zeroclawgpt code on `archive/zeroclawgpt` branch
- [x] Modular library: value.rs, config.rs, model.rs, forward.rs, train.rs, inference.rs, data.rs, rng.rs, ops.rs
- [x] Public API: `train_step()`, `generate()`, `build_vocab()`, `tokenize()`, `Value`
- [x] Smoke tests (vocab, tokenizer, lm_head, loss decrease, generation)
- [x] Dev tooling: rustfmt, clippy, git hooks, rust-analyzer
- [x] Push to mon-atelier-ia/microgpt-rs

## Phase 1 — WASM Bindings

**Goal:** Replace the TypeScript ML engine in microgpt-ts-fr with a WASM module compiled from this crate.

### Steps

1. Create `crates/wasm/` sub-crate with `wasm-bindgen` dependency
2. Expose key API functions:
   - `create_model(vocab_size, config_json) -> ModelHandle`
   - `train_step(handle, tokens) -> f64` (returns loss)
   - `generate(handle, n_samples, temperature) -> Vec<String>`
   - `get_activations(handle) -> JSON` (for visualizer)
3. Build with `wasm-pack build --target web`
4. Integrate into microgpt-ts-fr's existing Web Worker architecture
   - Workers currently call TS engine — switch to calling WASM
   - Keep the React UI unchanged

### Technical Notes

- Zero crate deps in core = trivial WASM compilation
- `Rc<RefCell<>>` compiles to WASM (no GC, reference counting in linear memory)
- `wasm-bindgen` only needed in the `crates/wasm/` sub-crate
- Shared types via `serde_json` (or manual JSON serialization to stay minimal)

### Expected Outcome

- Training speed: TS engine → WASM should be ~8x faster in browser
- Same UI, same worker architecture, just faster engine
- Autograd `Value` graph is available for visualization

## Phase 2 — Android Bindings (UniFFI)

**Goal:** Expose the Rust core to Kotlin for an Android app.

### Steps

1. Create `crates/android/` sub-crate with UniFFI
2. Define UDL interface file (UniFFI's IDL)
3. Generate Kotlin bindings: `cargo uniffi-bindgen generate`
4. Cross-compile:
   - `aarch64-linux-android` (ARM64 devices)
   - `armv7-linux-androideabi` (older ARM)
   - `x86_64-linux-android` (emulator)
5. Build Android app shell with Jetpack Compose

### Technical Notes

- UniFFI (Mozilla) is production-ready — used in Firefox for Android
- `cargo-ndk` handles the Android NDK toolchain
- No C/C++ layer — UniFFI generates direct Kotlin <-> Rust bindings
- Same core crate, different binding layer

## Phase 3 — French Datasets

**Goal:** Port the French datasets from microgpt-ts-fr to Rust.

### Steps

1. Add a `datasets/` directory with embedded `.txt` files (or `include_str!`)
2. Port from microgpt-ts-fr:
   - `prenoms-simple` (50 entries) — quick demo
   - `prenoms` (1,000 entries) — full training
   - `dinosaures` (1,530 entries)
   - `pokemon-fr` (1,022 entries)
3. Also available from microgpt-visualizer-fr:
   - `prenoms-insee` (33,235 entries) — comprehensive
4. Adapt demo binary with a `--dataset` flag

### Technical Notes

- Character-level tokenizer already supports Unicode — no code changes needed
- Larger vocab = more embedding parameters = slightly slower training
- Datasets embedded at compile time via `include_str!` (still zero deps)

## Phase 4 — Visualizer Integration

**Goal:** Feed activation data and autograd graph from the Rust engine to microgpt-visualizer-fr.

### Steps

1. Add `pub fn get_layer_activations()` to the library API
2. Export attention weights, intermediate embeddings, gradient norms
3. Expose autograd `Value` graph structure (children, gradients) for educational visualization
4. WASM binding: `get_activations()` returns a JS-accessible object

### Note

The migration to blackopsrepl's autograd approach resolves the earlier open question: the `Value` computation graph is now available in Rust, matching the TS visualizer's architecture. Both projects can share the same autograd-based visualization.

## Architecture Target

```
microgpt-rs/
├── Cargo.toml              # Workspace root
├── src/                    # Core library (zero deps)
│   ├── lib.rs
│   ├── value.rs
│   ├── config.rs
│   ├── model.rs
│   ├── forward.rs
│   ├── train.rs
│   ├── inference.rs
│   ├── data.rs
│   ├── rng.rs
│   └── ops.rs
├── crates/
│   ├── wasm/               # wasm-bindgen bindings
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   └── android/            # UniFFI bindings
│       ├── Cargo.toml
│       ├── src/lib.rs
│       └── src/model.udl
├── datasets/               # Embedded French/English data
├── src/bin/demo.rs         # CLI demo
└── tests/smoke.rs          # Smoke tests
```

## Priority Order

1. **WASM** — highest impact, enables browser speedup for both TS projects
2. **French datasets** — can be done in parallel with WASM
3. **Android** — requires more tooling (NDK, UniFFI, Compose), do after web is proven
4. **Visualizer** — autograd graph now available, integration straightforward

## References

- [wasm-pack](https://rustwasm.github.io/wasm-pack/)
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/)
- [UniFFI](https://github.com/mozilla/uniffi-rs)
- [cargo-ndk](https://github.com/nickelc/cargo-ndk)
- [Karpathy's microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [blackopsrepl's microgpt.rs](https://gist.github.com/blackopsrepl/bf7838f8f365c77e36075ca301db298e)
