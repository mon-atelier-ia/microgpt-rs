//! Minimal GPT library in pure Rust — faithful port of Karpathy's microgpt.py
//! with scalar autograd, zero dependencies.

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod config;
pub mod data;
pub mod inference;
pub mod model;
pub mod rng;
pub mod train;
pub mod value;

mod forward;
mod ops;
