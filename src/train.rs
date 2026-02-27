use crate::backward::backward;
use crate::config::TrainConfig;
use crate::forward::{forward, new_kv_cache, DKvCache};
use crate::model::Model;
use crate::ops::zeros;

/// Run one training step: forward → backward → Adam update.
/// Returns the average cross-entropy loss for this sequence.
pub fn train_step(model: &mut Model, tokens: &[usize], step: usize, tc: &TrainConfig) -> f32 {
    let cfg = model.config; // Copy — avoids borrow conflict with &mut self
    let e = cfg.n_embd;
    let n_pred = tokens.len().saturating_sub(1);
    if n_pred == 0 {
        return 0.0;
    }

    model.zero_grad();

    // Forward all positions (building KV cache)
    let mut kv = new_kv_cache(&cfg);
    let mut caches = Vec::with_capacity(n_pred);
    let mut loss_sum = 0.0f32;

    for pos in 0..n_pred {
        let cache = forward(&model.w, tokens[pos], pos, &mut kv, &cfg);
        loss_sum -= cache.probs[tokens[pos + 1]].ln();
        caches.push(cache);
    }

    // Backward in reverse — d_kv_cache accumulates cross-position gradients
    let mut d_kv: DKvCache = (0..cfg.n_layer)
        .map(|_| {
            let dk: Vec<Vec<f32>> = (0..n_pred).map(|_| zeros(e)).collect();
            let dv: Vec<Vec<f32>> = (0..n_pred).map(|_| zeros(e)).collect();
            (dk, dv)
        })
        .collect();

    for pos in (0..n_pred).rev() {
        backward(
            &model.w,
            &mut model.g,
            &caches[pos],
            tokens[pos + 1],
            tokens.len(),
            &mut d_kv,
            &cfg,
        );
    }

    model.adam_step(step + 1, tc); // 1-indexed for Adam bias correction

    loss_sum / n_pred as f32
}
