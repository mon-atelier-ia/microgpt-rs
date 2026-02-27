use crate::config::ModelConfig;
use crate::model::Params;
use crate::forward::PosCache;
use crate::ops::{zeros, linear_bwd_w, linear_bwd_x, rmsnorm_bwd};

pub(crate) fn backward(
    w: &Params,
    g: &mut Params,
    c: &PosCache,
    target: usize,
    seq_len: usize,
    d_kv_cache: &mut Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>,
    cfg: &ModelConfig,
) {
    let e  = cfg.n_embd;
    let hd = cfg.head_dim();
    let nh = cfg.n_head;
    let bs = cfg.block_size;
    let vs = w.wte.len() / e;
    let norm = 1.0 / (seq_len - 1) as f32;

    // Cross-entropy gradient (scaled by normalization)
    let mut d_logits = c.probs.clone();
    d_logits[target] -= 1.0;
    d_logits.iter_mut().for_each(|v| *v *= norm);

    // lm_head (weight-tied wte)
    let last_x = if cfg.n_layer > 0 { &c.layers[cfg.n_layer - 1].x_post_mlp } else { &c.x0 };
    linear_bwd_w(&mut g.wte, &d_logits, last_x, vs, e);
    let mut dx = zeros(e);
    linear_bwd_x(&mut dx, &d_logits, &w.wte, vs, e);

    for l in (0..cfg.n_layer).rev() {
        let lw = &w.layers[l];
        let lg = &mut g.layers[l];
        let lc = &c.layers[l];
        let ac = &c.attn_ctx[l];
        let t_len = ac.all_k.len();
        let cur_t = t_len - 1;

        // ── MLP residual ────────────────────────────────────────────────────
        let dx_s = dx.clone();
        let mut d_h1a = zeros(4 * e);
        linear_bwd_w(&mut lg.fc2, &dx, &lc.h1a, e, 4 * e);
        linear_bwd_x(&mut d_h1a, &dx, &lw.fc2, e, 4 * e);
        let d_h1: Vec<f32> = d_h1a.iter().zip(&lc.h1)
            .map(|(da, h)| if *h > 0.0 { da * 2.0 * h } else { 0.0 }).collect();
        let mut d_xn_mlp = zeros(e);
        linear_bwd_w(&mut lg.fc1, &d_h1, &lc.xn_mlp, 4 * e, e);
        linear_bwd_x(&mut d_xn_mlp, &d_h1, &lw.fc1, 4 * e, e);
        let d_rn_m = rmsnorm_bwd(&d_xn_mlp, &lc.x_post_attn, lc.rms_m);
        dx = dx_s.iter().zip(&d_rn_m).map(|(a, b)| a + b).collect();

        // ── Attention residual ──────────────────────────────────────────────
        let dx_s = dx.clone();
        let mut d_ho = zeros(e);
        linear_bwd_w(&mut lg.wo, &dx, &ac.ho, e, e);
        linear_bwd_x(&mut d_ho, &dx, &lw.wo, e, e);

        let mut d_q = zeros(e);
        for h in 0..nh {
            let aw_h  = &ac.aw[h];
            let scale = (hd as f32).sqrt();

            // gradient into value vectors
            for t in 0..t_len {
                let dv = &mut d_kv_cache[l].1[t];
                for i in 0..hd { dv[h * hd + i] += aw_h[t] * d_ho[h * hd + i]; }
            }

            // gradient into attention weights
            let d_aw_logits_raw: Vec<f32> = (0..t_len).map(|t| {
                (0..hd).map(|i| d_ho[h * hd + i] * ac.all_v[t][h * hd + i]).sum::<f32>()
            }).collect();
            let dot_aw: f32 = aw_h.iter().zip(&d_aw_logits_raw).map(|(a, b)| a * b).sum();
            let d_attn_logits: Vec<f32> = aw_h.iter().zip(&d_aw_logits_raw)
                .map(|(a, d)| a * (d - dot_aw)).collect();

            // d_q
            for t in 0..t_len {
                let ks = &ac.all_k[t][h * hd..(h + 1) * hd];
                for i in 0..hd {
                    d_q[h * hd + i] += d_attn_logits[t] * ks[i] / scale;
                }
            }
            // d_k
            for t in 0..t_len {
                let dk = &mut d_kv_cache[l].0[t];
                let qs = &c.layers[l].q[h * hd..(h + 1) * hd];
                for i in 0..hd {
                    dk[h * hd + i] += d_attn_logits[t] * qs[i] / scale;
                }
            }
        }

        // Project d_q, d_k_cur, d_v_cur back through wq/wk/wv
        let dk_cur = &d_kv_cache[l].0[cur_t];
        let dv_cur = &d_kv_cache[l].1[cur_t];
        let mut d_xn_attn = zeros(e);
        linear_bwd_w(&mut lg.wq, &d_q,   &lc.xn_attn, e, e);
        linear_bwd_x(&mut d_xn_attn, &d_q,   &lw.wq, e, e);
        linear_bwd_w(&mut lg.wk, dk_cur, &lc.xn_attn, e, e);
        linear_bwd_x(&mut d_xn_attn, dk_cur, &lw.wk, e, e);
        linear_bwd_w(&mut lg.wv, dv_cur, &lc.xn_attn, e, e);
        linear_bwd_x(&mut d_xn_attn, dv_cur, &lw.wv, e, e);

        let d_rn_a = rmsnorm_bwd(&d_xn_attn, &lc.x_pre_attn, lc.rms_a);
        dx = dx_s.iter().zip(&d_rn_a).map(|(a, b)| a + b).collect();
    }

    // Embedding gradients
    for i in 0..e { g.wte[c.tok_id * e + i] += dx[i]; }
    for i in 0..e { g.wpe[(c.pos_id % bs) * e + i] += dx[i]; }
}
