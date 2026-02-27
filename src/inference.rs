use crate::config::ModelConfig;
use crate::model::Params;
use crate::forward::{forward, new_kv_cache};
use crate::data::Vocab;
use crate::rng::Rng;

/// Generate `n_samples` names autoregressively.
pub fn generate(
    w: &Params,
    vocab: &Vocab,
    rng: &mut Rng,
    cfg: &ModelConfig,
    n_samples: usize,
) -> Vec<String> {
    let bos = vocab.bos();
    let eos = vocab.eos();
    let mut results = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut kv = new_kv_cache(cfg);
        let mut tok = bos;
        let mut name = String::new();
        for pos in 0..cfg.block_size {
            let c = forward(w, tok, pos, &mut kv, cfg);
            tok = rng.categorical(&c.probs);
            if tok == eos { break; }
            name.push_str(&vocab.tokens[tok]);
        }
        results.push(name);
    }

    results
}
