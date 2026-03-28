## Agent Behavior

- Be proactive: when you learn something important (decisions, conventions, pitfalls), save it to `CLAUDE.md` or the relevant file in `docs/`.
- **Always document findings** in `docs/` before acting on them. Never keep research results only in the conversation.
- For non-trivial tasks, create a plan doc in `docs/` using a descriptive filename.

## Documentation

Project documentation lives in `docs/`. Key files:
- `docs/gist-conformance.md` — audit of all implementations vs Karpathy's gist
- `docs/roadmap.md` — WASM/Android integration plan, next steps
- `docs/implementation-notes.md` — archived implementation analysis
- `docs/ui-parameters-audit.md` — UI params coverage vs TS projects

## Stack

| Tool | Purpose |
|------|---------|
| Rust (edition 2021) | Language |
| Cargo | Build system & package manager |
| rustfmt | Code formatting |
| clippy | Linting |
| rust-analyzer | IDE support (VS Code) |

No external crate dependencies. Everything is `std` only.

## Project Structure

```
src/
├── lib.rs           # Crate root (pub vs pub(crate) modules)
├── value.rs         # Scalar autograd engine (Value + backward)
├── config.rs        # ModelConfig (Copy) + TrainConfig
├── model.rs         # StateDict, LayerWeights, Model (Adam state)
├── forward.rs       # Forward pass + KV cache (internal)
├── train.rs         # train_step() — public training API
├── inference.rs     # generate() — public inference API
├── data.rs          # Vocab, build_vocab(), tokenize()
├── rng.rs           # Xorshift64 PRNG
├── ops.rs           # linear, softmax, rmsnorm on Vec<Value> (internal)
└── bin/
    └── demo.rs      # CLI demo

tests/
└── smoke.rs         # 5 smoke tests
```

### Visibility Convention

- `pub` modules (`value`, `config`, `model`, `data`, `train`, `inference`, `rng`): stable library API
- `pub(crate)` modules (`forward`, `ops`): internal implementation, free to refactor

## Scripts

```bash
cargo build --release           # Build
cargo run --release --bin demo  # Run demo
cargo test --release            # Run tests
cargo fmt --check               # Check formatting
cargo clippy -- -D warnings     # Lint
```

## Key Rules

- **Zero dependencies**: no external crates. The entire GPT is visible.
- **Karpathy-faithful**: architecture, hyperparameters, and numerical behavior match the [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). See `docs/gist-conformance.md`.
- **Scalar autograd**: `Value(Rc<RefCell<>>)` computation graph with `backward()`. No analytical gradients.
- **`ModelConfig` is `Copy`**: avoids borrow conflicts when `&mut Model` is in use.
- **BOS only**: same token for start and end of sequence. No EOS token.
- **Separate lm_head**: no weight tying with wte.
- **f64 throughout**: matches Python's float precision.
- Run `cargo fmt` and `cargo clippy -- -D warnings` before committing.

## Git Rules — STRICT

- NEVER run `git push` without explicit user request
- NEVER run destructive git commands (force push, reset --hard, etc.)
- Work is LOCAL ONLY: edits, commits, lint. User decides when to push.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `style:`, `chore:`, `docs:`

## Git Hooks

Pre-commit (`.githooks/pre-commit`):
```bash
cargo fmt --check && cargo clippy -- -D warnings
```

Pre-push (`.githooks/pre-push`):
```bash
cargo build --release && cargo test --release
```

Activate hooks: `git config core.hooksPath .githooks`

## Environment

- Windows 11, VS Code
- Rust toolchain via rustup (stable, MSVC target)
- PATH: `~/.cargo/bin` must be in PATH for `cargo`, `rustfmt`, `clippy`
- No venv, no nvm — Rust manages its own toolchain via rustup

## Ecosystem Context

This crate is part of the [mon-atelier-ia](https://github.com/mon-atelier-ia) educational GPT ecosystem:

| Project | Language | Role |
|---------|----------|------|
| [microgpt-rs](https://github.com/mon-atelier-ia/microgpt-rs) | Rust | Core engine (this repo) |
| [microgpt-ts-fr](https://github.com/mon-atelier-ia/microgpt-ts-fr) | TypeScript | Training playground, French datasets |
| [microgpt-visualizer-fr](https://github.com/mon-atelier-ia/microgpt-visualizer-fr) | TypeScript | Transformer pipeline visualizer |

The Rust core will compile to WASM (web) and Android (NDK/UniFFI) to replace the TS engine in both sibling projects.
