use crate::config::ModelConfig;
use crate::data::Vocab;
use crate::forward::{forward, new_kv_cache};
use crate::model::StateDict;
use crate::ops::softmax;
use crate::rng::Rng;
use crate::value::Value;

/// Generate `n_samples` names autoregressively with temperature scaling.
pub fn generate(
    sd: &StateDict,
    vocab: &Vocab,
    rng: &mut Rng,
    cfg: &ModelConfig,
    n_samples: usize,
    temperature: f64,
) -> Vec<String> {
    let bos = vocab.bos();
    let mut results = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let (mut keys, mut vals) = new_kv_cache(cfg);
        let mut token_id = bos;
        let mut name = String::new();

        for pos_id in 0..cfg.block_size {
            let logits = forward(token_id, pos_id, &mut keys, &mut vals, sd, cfg);
            let scaled: Vec<Value> = logits
                .iter()
                .map(|l| l.mul_f64(1.0 / temperature))
                .collect();
            let probs = softmax(&scaled);
            let weights: Vec<f64> = probs.iter().map(|p| p.data()).collect();
            token_id = rng.categorical(&weights);
            if token_id == bos {
                break;
            }
            name.push_str(&vocab.tokens[token_id]);
        }

        results.push(name);
    }

    results
}
