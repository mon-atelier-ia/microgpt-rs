# Roadmap

> Plan d'integration de microgpt-rs dans l'ecosysteme mon-atelier-ia.

## Phase 0 — Library Foundation (Done)

- [x] Fork zeroclawgpt and restructure into `lib.rs` + modules
- [x] Typed parameter structs (replace `HashMap<String, Vec<f32>>`)
- [x] Configurable `ModelConfig` / `TrainConfig` (no more hardcoded constants)
- [x] Public API: `train_step()`, `generate()`, `build_vocab()`, `tokenize()`
- [x] Smoke tests (param count, loss decrease, generation)
- [x] Dev tooling: rustfmt, clippy, git hooks, rust-analyzer
- [x] Push to mon-atelier-ia/microgpt-rs

## Phase 1 — WASM Bindings

**Goal:** Replace the TypeScript ML engine in microgpt-ts-fr with a WASM module compiled from this crate.

### Steps

1. Create `crates/wasm/` sub-crate with `wasm-bindgen` dependency
2. Expose key API functions:
   - `create_model(vocab_size, config_json) -> ModelHandle`
   - `train_step(handle, tokens) -> f32` (returns loss)
   - `generate(handle, n_samples) -> Vec<String>`
   - `get_activations(handle) -> JSON` (for visualizer)
3. Build with `wasm-pack build --target web`
4. Integrate into microgpt-ts-fr's existing Web Worker architecture
   - Workers currently call TS engine — switch to calling WASM
   - Keep the React UI unchanged

### Technical Notes

- Zero crate deps in core = trivial WASM compilation
- `wasm-bindgen` only needed in the `crates/wasm/` sub-crate
- Memory: WASM linear memory, no GC overhead
- Shared types via `serde_json` (or manual JSON serialization to stay minimal)

### Expected Outcome

- Training speed: TS engine → WASM should be 10-50x faster in browser
- Same UI, same worker architecture, just faster engine

## Phase 2 — Android Bindings (UniFFI)

**Goal:** Expose the Rust core to Kotlin for an Android app.

### Steps

1. Create `crates/android/` sub-crate with UniFFI
2. Define UDL interface file (UniFFI's IDL):
   ```
   interface Model {
     constructor(u32 vocab_size, ModelConfig config);
     f32 train_step(sequence<u32> tokens, u32 step);
     sequence<string> generate(u32 n_samples);
   };
   ```
3. Generate Kotlin bindings: `cargo uniffi-bindgen generate`
4. Cross-compile:
   - `aarch64-linux-android` (ARM64 devices)
   - `armv7-linux-androideabi` (older ARM)
   - `x86_64-linux-android` (emulator)
5. Build Android app shell with Jetpack Compose

### Technical Notes

- UniFFI (Mozilla) is production-ready — used in Firefox for Android
- `cargo-ndk` handles the Android NDK toolchain
- No C/C++ layer — UniFFI generates direct Kotlin ↔ Rust bindings
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
3. Adapt `build_vocab()` for accented characters (e, e, c expand vocab from 27 to ~40+)
4. Update demo binary with a `--dataset` flag

### Technical Notes

- Character-level tokenizer already supports Unicode — no code changes needed
- Larger vocab = more embedding parameters = slightly slower training
- Datasets embedded at compile time via `include_str!` (still zero deps)

## Phase 4 — Visualizer Integration

**Goal:** Feed activation data from the Rust engine to microgpt-visualizer-fr.

### Steps

1. Add `pub fn get_layer_activations()` to the library API
2. Export attention weights, intermediate embeddings, gradient norms
3. Serialize as JSON (or binary) for the visualizer's React frontend
4. WASM binding: `get_activations()` returns a JS-accessible object

### Open Question

The visualizer currently shows the autograd computation graph (Value nodes, backward closures). microgpt-rs uses analytical gradients — there is no graph to show. Options:

- **Option A:** Visualize activations/attention only (not the autograd graph)
- **Option B:** Add an optional `Value`-based autograd mode (like blackopsrepl's approach) for educational purposes, at the cost of performance
- **Option C:** Keep the TS engine in the visualizer for graph visualization, use Rust only in microgpt-ts-fr

Decision deferred until Phase 2 is complete.

## Architecture Target

```
microgpt-rs/
├── Cargo.toml              # Workspace root
├── src/                    # Core library (zero deps)
│   ├── lib.rs
│   ├── config.rs
│   ├── model.rs
│   ├── forward.rs
│   ├── backward.rs
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

1. **WASM** — highest impact, enables browser speedup for microgpt-ts-fr immediately
2. **French datasets** — can be done in parallel with WASM
3. **Android** — requires more tooling (NDK, UniFFI, Compose), do after web is proven
4. **Visualizer** — depends on architecture decision (Option A/B/C)

## References

- [wasm-pack](https://rustwasm.github.io/wasm-pack/)
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/)
- [UniFFI](https://github.com/mozilla/uniffi-rs)
- [cargo-ndk](https://github.com/nickelc/cargo-ndk)
- [Research: Rust ports of Karpathy's gist](../docs-ts-fr/rust-port-research.md) (in microgpt-ts-fr)
