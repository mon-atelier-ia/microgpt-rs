# Rust Port Research

> **Context**: Research into existing Rust ports of Karpathy's [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) gist — the same 243-line, zero-dependency Python GPT that our TypeScript fork is based on. Goal: evaluate candidates for a Rust core engine that compiles to both WASM (web) and Android (NDK/Kotlin).

**Date**: 2026-02-27

## Source gist

- **Author**: Andrej Karpathy
- **Gist**: <https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95>
- **What it is**: `microgpt.py` — complete GPT in ~243 lines of pure Python, no dependencies
- **Key identifiers**: `Value` (autograd), `gpt()` (forward), `linear()`, `softmax()`, `rmsnorm()`, `state_dict`, Adam optimizer
- **Architecture**: character-level tokenizer, scalar autograd, multi-head attention with KV cache, RMSNorm, squared ReLU, weight tying, linear LR decay
- **Licence**: MIT

---

## Rust ports found

### 1. rustystack/zeroclawgpt — Faithful port, analytical gradients

| | |
|---|---|
| **Repo** | <https://github.com/rustystack/zeroclawgpt> |
| **Stars** | 6 (created 2026-02-21) |
| **Licence** | MIT |
| **Dependencies** | **Zero** (`std` only) |
| **Lines** | 475 (single file `src/main.rs`) |
| **Gradient approach** | **Analytical** (hand-derived matrix gradients, no autograd graph) |
| **Speedup vs Python** | **~4,580x** (5000 steps in 0.69s vs ~25min) |
| **Parameters** | 3,632 (identical to gist) |
| **Architecture match** | Exact — RMSNorm, squared ReLU, weight tying, KV cache, Adam with linear LR decay |
| **WASM support** | Not implemented, but trivially possible (zero deps) |
| **Android support** | Not implemented, but trivially possible (zero deps) |

**Key technical details:**
- PRNG: xoshiro128+ in 15 lines with Box-Muller gaussian sampling
- Matrix ops: row-major matmul, element-wise ops, all inline
- Forward: processes one token position at a time, accumulates KV cache
- Backward: analytical gradients through every operation (`linear_bwd_w`, `linear_bwd_x`, `rmsnorm_bwd`, softmax-CE shortcut `d_logits = probs - one_hot`)
- Optimizer: Adam with bias correction, linear LR decay
- Dataset: 92 baby names embedded in source

**5 bugs found vs original gist:**
1. KV cache causal attention (critical — was processing positions independently)
2. Adam beta2: 0.999 → 0.95
3. Linear LR decay (was constant)
4. Zero-init output projections (Wo, Wfc2)
5. Loss normalization before backward pass

**Strengths:**
- Fastest port by far
- Zero dependencies = compiles everywhere
- Single file, readable in 20 min
- Bug fixes are documented and explained

**Limitations:**
- No autograd — adding a new layer type requires hand-deriving gradients
- Not structured as a library (single `main.rs`)
- No test suite

---

### 2. blackopsrepl/microgpt.rs — Faithful port WITH autograd

| | |
|---|---|
| **Gist** | <https://gist.github.com/blackopsrepl/bf7838f8f365c77e36075ca301db298e> |
| **Format** | Cargo script (`cargo +nightly -Zscript`) |
| **Dependencies** | **Zero** |
| **Lines** | ~700+ |
| **Gradient approach** | **Scalar autograd** (`Value` struct + `Rc<RefCell<>>` + topological sort backprop) |
| **Speedup vs Python** | **~8x** |
| **Architecture match** | Exact — same `Value`, `linear()`, `softmax()`, `rmsnorm()`, `gpt()` functions |
| **WASM support** | Not mentioned |
| **Android support** | Not mentioned |

**Key technical details:**
- `Value` struct: data, gradient, children refs via `Rc<RefCell<>>`
- `backward()`: topological sort + gradient accumulation (same algorithm as Python/TS)
- Custom xorshift64 RNG with Box-Muller
- KV cache management passed through function parameters
- Adam with manual buffer management

**Strengths:**
- **Closest to our TS architecture** — same `Value`-based autograd pattern
- Extensible: new operations only need forward + `_backward`
- Good candidate for the visualizer (graph is inspectable)

**Limitations:**
- "Only" 8x faster (Rc/RefCell overhead for graph nodes)
- Single gist, not a structured project
- No tests
- `Rc<RefCell<>>` pattern compiles to WASM but with GC overhead

---

### 3. keyvank/femtoGPT — Most mature, but different source

| | |
|---|---|
| **Repo** | <https://github.com/keyvank/femtoGPT> |
| **Stars** | **936** (created 2023-05-28) |
| **Licence** | MIT |
| **Dependencies** | `rand`, `rayon`, `serde`, `bincode`, optional `ocl` (GPU) |
| **Gradient approach** | Custom tensors + analytical gradients |
| **GPU** | Yes (OpenCL, works on NVIDIA and AMD) |
| **Inspired by** | Karpathy's **nanoGPT video lecture**, NOT the `microgpt.py` gist |
| **WASM support** | No (`rayon` and `ocl` don't compile to WASM easily) |
| **Android support** | Complex (multiple deps with native code) |

**File structure:**
```
src/
  funcs/        # activation functions
  gpt.rs        # GPT architecture
  graph/        # computation graph
  lib.rs        # library root
  main.rs       # CLI entry
  optimizer.rs  # Adam optimizer
  tensor/       # tensor operations
  tokenizer/    # tokenizer
```

**Strengths:**
- Most battle-tested (936 stars, 67 forks, 2+ years old)
- GPU support via OpenCL
- Modular architecture
- Trained on Shakespeare dataset (larger scale)

**Limitations:**
- **Not based on the same gist** — different architecture choices
- Multiple dependencies make cross-compilation harder
- rayon (parallelism) incompatible with WASM
- Not a 1:1 mapping to our TS code

---

## Comparison matrix

| Criteria | zeroclawgpt | blackopsrepl | femtoGPT |
|----------|:-----------:|:------------:|:--------:|
| Based on same gist | **exact** | **exact** | nanoGPT lecture |
| Autograd like our TS | no | **yes** | no |
| Raw performance | **4,580x** | 8x | good (+ GPU) |
| Zero dependencies | **yes** | **yes** | no |
| Compilable to WASM | **trivial** | yes* | hard |
| Compilable to Android/NDK | **trivial** | yes* | complex |
| Extensibility | low | **good** | **good** |
| Community maturity | low | gist only | **strong** |
| Structured as library | no | no | **yes** |

*`Rc<RefCell<>>` compiles to WASM but with overhead.*

---

## Recommendations

### For `microgpt-ts-fr` (training + generation playground)

**Start from zeroclawgpt.** Rationale:
- Zero deps = compiles to `wasm32-unknown-unknown` and `aarch64-linux-android` with zero configuration
- 4,580x speedup enables real-time training on mobile
- Faithful to the gist = 1:1 mapping with our TS model code
- Trade-off (no autograd) is acceptable: our model architecture is fixed

**Work needed:**
1. Restructure from single `main.rs` into `lib.rs` + modules
2. Add `#[wasm_bindgen]` bindings in a `crates/wasm/` sub-crate
3. Add UniFFI or JNI bindings in a `crates/android/` sub-crate
4. Port our French datasets and dynamic tokenizer
5. Add test suite (port our Vitest cases)

### For `microgpt-visualizer-fr` (educational visualization)

**Consider blackopsrepl's autograd approach** for the Rust core, since the visualizer needs to inspect the computation graph (Value nodes, gradients, topology). The 8x speedup is still significant for the visualizer's use case (small models, emphasis on transparency).

**Alternative:** Keep the visualizer in pure TS/WASM (zeroclawgpt compiled to WASM is fast enough for small models) and only visualize the forward pass / activations rather than the autograd graph itself.

### Rust → Web (WASM) pathway

- **Toolchain**: `wasm-pack` + `wasm-bindgen`
- **Integration**: WASM module imported by React/Next.js, replaces current TS engine
- **Workers**: existing Web Worker architecture stays, just calls WASM instead of TS

### Rust → Android (Kotlin) pathway

- **Toolchain**: `cargo-ndk` + Android NDK
- **Bindings**: [UniFFI](https://github.com/mozilla/uniffi-rs) (Mozilla, production-ready) generates Kotlin bindings automatically
- **Integration**: Kotlin/Jetpack Compose UI calls Rust core via generated bindings
- **Targets**: `aarch64-linux-android`, `armv7-linux-androideabi`, `x86_64-linux-android`

### Reference: cross-platform Rust architecture

```
microgpt-core-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Public API
│   ├── value.rs        # (optional) Autograd engine
│   ├── model.rs        # GPT model
│   ├── train.rs        # Training loop + Adam
│   └── utils.rs        # Softmax, sampling, matmul
├── crates/
│   ├── wasm/           # wasm-bindgen bindings
│   │   └── src/lib.rs
│   └── android/        # UniFFI bindings
│       └── src/lib.rs
```

---

## References

- [zeroclawgpt](https://github.com/rustystack/zeroclawgpt) — faithful Rust port, 4,580x faster
- [blackopsrepl's microgpt.rs](https://gist.github.com/blackopsrepl/bf7838f8f365c77e36075ca301db298e) — Rust port with autograd
- [femtoGPT](https://github.com/keyvank/femtoGPT) — mature Rust GPT (different source)
- [UniFFI](https://github.com/mozilla/uniffi-rs) — Mozilla's Rust → Kotlin/Swift bindings
- [Building WASM, Android and iOS with single Rust core](https://dev.to/h_ajsf/building-wasm-android-and-ios-app-with-singlecommon-rust-core-code-3ja4)
- [Practical Client-side Rust (Mux)](https://www.mux.com/blog/practical-client-side-rust-for-android-ios-and-web)
- [Karpathy's microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [Karpathy's microgpt blog post](http://karpathy.github.io/2026/02/12/microgpt/)
