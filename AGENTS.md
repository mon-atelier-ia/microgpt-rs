> **Note:** `CLAUDE.md` is a symlink to this file. `AGENTS.md` is the source of truth — edit this file, not `CLAUDE.md`.

## Agent Behavior

- Be proactive: when you learn something important (decisions, conventions, pitfalls), save it to `AGENTS.md` or the relevant file in `docs/`.
- For non-trivial tasks, create a plan doc in `docs/` using a descriptive filename.

## Documentation

Project documentation lives in `docs/`. Key files:
- `docs/implementation-notes.md` — original zeroclawgpt analysis, bug fixes, test plan
- `docs/roadmap.md` — WASM/Android integration plan, next steps

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
├── config.rs        # ModelConfig (Copy) + TrainConfig
├── model.rs         # Params, LayerParams, Model
├── forward.rs       # Forward pass + KV cache (internal)
├── backward.rs      # Analytical gradients (internal)
├── train.rs         # train_step() — public training API
├── inference.rs     # generate() — public inference API
├── data.rs          # Vocab, build_vocab(), tokenize()
├── rng.rs           # xoshiro128+ PRNG
├── ops.rs           # Linear algebra primitives (internal)
└── bin/
    └── demo.rs      # CLI demo

tests/
└── smoke.rs         # 3 smoke tests
```

### Visibility Convention

- `pub` modules (`config`, `model`, `data`, `train`, `inference`, `rng`): stable library API, consumed by external crates and future WASM/Android bindings
- `pub(crate)` modules (`forward`, `backward`, `ops`): internal implementation, free to refactor

## Scripts

```bash
cargo build --release        # Build
cargo run --release --bin demo  # Run demo
cargo test --release         # Run tests
cargo fmt --check            # Check formatting
cargo clippy -- -D warnings  # Lint
```

## Key Rules

- **Zero dependencies**: no external crates. This is the point — the entire GPT is visible.
- **Karpathy-faithful**: architecture, hyperparameters, and numerical behavior match the [original gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).
- **`ModelConfig` is `Copy`**: avoids borrow conflicts when `&mut Model` is in use. Never change it to non-Copy.
- **Split borrows**: `forward()` takes `&Params`, `backward()` takes `(&Params, &mut Params)` — this is how we avoid borrow checker fights on `Model`.
- **Numerical code style**: `#[allow(clippy::needless_range_loop)]` is acceptable in `forward.rs` and `backward.rs` where multi-array indexed loops are clearer than iterator chains.
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

The Rust core will eventually compile to WASM (web) and Android (NDK/UniFFI) to replace the TS engine in the sibling projects.
