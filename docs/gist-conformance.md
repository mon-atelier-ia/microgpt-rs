# Gist Conformance Audit

> Comparison of all three implementations against the source of truth:
> [Karpathy's microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

**Date**: 2026-02-28

## Source of Truth — Karpathy's Gist

| Parameter | Gist Value |
|---|---|
| Activation | ReLU (`xi.relu()`) |
| block_size | 16 |
| n_embd | 16 |
| n_head | 4 |
| n_layer | 1 |
| beta1 | 0.85 |
| beta2 | 0.99 |
| eps | 1e-8 |
| lr | 0.01, linear decay to 0 |
| std init | 0.08, uniform across all matrices |
| BOS/EOS | BOS only — same token for start and end |
| Weight tying | No — separate `wte` and `lm_head` |
| Zero-init | No — all matrices at std=0.08 |
| RMSNorm eps | 1e-5 |
| Loss normalization | Before backward: `loss * (1/n)` |
| Tokenizer | Character-level, dynamic vocab from dataset |
| Autograd | Yes — scalar `Value` class with backward() |

## Conformance Matrix

| Parameter | Gist | zeroclawgpt (microgpt-rs) | blackopsrepl | microgpt-ts-fr | microgpt-visualizer-fr |
|---|---|---|---|---|---|
| Activation | ReLU | **squared ReLU** | ReLU | ReLU | ReLU |
| block_size | 16 | **8** | 16 | 16 | 16 |
| beta1 | 0.85 | **0.9** | 0.85 | **0.9** | 0.85 |
| beta2 | 0.99 | **0.95** | 0.99 | **0.95** | 0.99 |
| std init | 0.08 | **0.02** | 0.08 | ? | ? |
| BOS/EOS | BOS only | **BOS + EOS** | BOS only | BOS only | BOS only |
| Weight tying | No (lm_head separate) | **Yes (wte reused)** | No | ? | ? |
| Zero-init wo/fc2 | No | **Yes** | No | ? | ? |
| Autograd | Yes (Value) | **No (analytical)** | Yes (Value) | Yes (Value) | Yes (Value) |
| **Score** | — | **0/8** | **8/8** | **6/8** | **8/8** |

Bold = diverges from the gist.

## Rust Ports Comparison

### zeroclawgpt (current microgpt-rs base)

- **Repo**: <https://github.com/rustystack/zeroclawgpt>
- **Conformance**: 0/8 — diverges on every checked parameter
- **Speed**: ~4,580x vs Python
- **Gradient approach**: Analytical (hand-derived matrix gradients)
- **Dependencies**: Zero
- **Structure**: Restructured as library (by us)

The "5 bug fixes" documented in zeroclawgpt's README were not corrections toward the gist — they were modifications that moved AWAY from the gist's actual values. Possible explanations:
1. The gist was updated after zeroclawgpt was written
2. zeroclawgpt compared against a different reference
3. Intentional modifications presented as corrections

### blackopsrepl

- **Gist**: <https://gist.github.com/blackopsrepl/bf7838f8f365c77e36075ca301db298e>
- **Conformance**: 8/8 — faithful on all checked parameters
- **Speed**: ~8x vs Python
- **Gradient approach**: Scalar autograd (`Value` + `Rc<RefCell<>>`)
- **Dependencies**: Zero
- **Structure**: Single gist file, not structured as library

### Trade-offs

| | zeroclawgpt | blackopsrepl |
|---|---|---|
| Gist faithfulness | 0/8 | **8/8** |
| Performance | **4,580x** | 8x |
| Autograd (matches TS) | No | **Yes** |
| WASM feasibility | Trivial | Yes (Rc compiles to WASM) |
| Library structure | **Done** (by us) | Needs restructuring |
| Code extensibility | Low (manual gradients) | **Good** (just add ops) |

## Impact on microgpt-rs

### Option A: Correct zeroclawgpt (8 changes)

Keep current microgpt-rs base, fix all divergences:
1. Activation: squared ReLU → ReLU (forward.rs + backward.rs)
2. block_size default: 8 → 16 (config.rs)
3. beta1: 0.9 → 0.85 (config.rs)
4. beta2: 0.95 → 0.99 (config.rs)
5. std init: 0.02 → 0.08 (model.rs)
6. BOS/EOS: remove EOS, BOS serves as both (data.rs)
7. Weight tying: add separate lm_head to Params (model.rs + forward.rs + backward.rs)
8. Zero-init: remove, all matrices at std=0.08 (model.rs)
9. Tests: update param_count (lm_head adds parameters), adjust smoke.rs

**Pro**: Keep library structure, keep analytical speed
**Con**: Still no autograd, doesn't match TS architecture

### Option B: Rebase on blackopsrepl

Start from blackopsrepl, restructure into library:
1. Already faithful to gist (0 fixes needed)
2. Has autograd (Value) — mirrors TS engine
3. Needs library restructuring (same work we did for zeroclawgpt)
4. Slower (~8x vs ~4,580x), but still much faster than TS

**Pro**: Faithful, autograd matches TS, extensible
**Con**: Slower, restructuring work to redo

### Option C: Hybrid

Keep zeroclawgpt's library structure + analytical speed, add autograd as optional feature:
1. Fix the 8 conformance issues
2. Add a `value.rs` module with autograd (port from blackopsrepl)
3. Feature flag: `--features autograd` for Value-based path, default for analytical

**Pro**: Best of both worlds
**Con**: Two code paths to maintain

## Decision

Pending — awaiting user input.

## References

- [Karpathy's microgpt.py gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [zeroclawgpt](https://github.com/rustystack/zeroclawgpt)
- [blackopsrepl's microgpt.rs](https://gist.github.com/blackopsrepl/bf7838f8f365c77e36075ca301db298e)
