//! Minimal GPT library in pure Rust — faithful port of Karpathy's microgpt.py
//! with scalar autograd, zero dependencies.

#![warn(missing_docs)]
#![deny(unsafe_code)]

/// GPT model architecture and training hyperparameters.
pub mod config;
/// Character-level vocabulary and tokenization.
pub mod data;
/// Autoregressive text generation with temperature sampling.
pub mod inference;
/// Weight matrices, state dict, and Adam optimizer.
pub mod model;
/// Xorshift64 PRNG with gaussian and categorical sampling.
pub mod rng;
/// Single training step: forward, backward, Adam update.
pub mod train;
/// Scalar autograd engine (computation graph with reverse-mode backward).
pub mod value;

mod forward;
mod ops;
