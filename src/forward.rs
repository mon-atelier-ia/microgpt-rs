use crate::config::ModelConfig;
use crate::model::Params;
use crate::ops::{zeros, linear, softmax, rmsnorm};

pub(crate) struct AttnCtx {
    pub all_k: Vec<Vec<f32>>,
    pub all_v: Vec<Vec<f32>>,
    pub aw: Vec<Vec<f32>>,
    pub ho: Vec<f32>,
}

pub(crate) struct LCache {
    pub x_pre_attn:  Vec<f32>,
    pub xn_attn:     Vec<f32>,
    pub rms_a:       f32,
    pub q:           Vec<f32>,
    pub attn_out:    Vec<f32>,
    pub x_post_attn: Vec<f32>,
    pub xn_mlp:      Vec<f32>,
    pub rms_m:       f32,
    pub h1:          Vec<f32>,
    pub h1a:         Vec<f32>,
    pub mlp_out:     Vec<f32>,
    pub x_post_mlp:  Vec<f32>,
}

pub(crate) struct PosCache {
    pub tok_id:   usize,
    pub pos_id:   usize,
    pub x0:       Vec<f32>,
    pub layers:   Vec<LCache>,
    pub probs:    Vec<f32>,
    pub attn_ctx: Vec<AttnCtx>,
}

pub type KvCache = Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>;

pub(crate) fn new_kv_cache(cfg: &ModelConfig) -> KvCache {
    vec![(vec![], vec![]); cfg.n_layer]
}

pub(crate) fn forward(
    w: &Params,
    tok: usize,
    pos: usize,
    kv_cache: &mut KvCache,
    cfg: &ModelConfig,
) -> PosCache {
    let vs = w.wte.len() / cfg.n_embd;
    let e  = cfg.n_embd;
    let hd = cfg.head_dim();
    let nh = cfg.n_head;
    let bs = cfg.block_size;

    // Embeddings
    let te = &w.wte[tok * e..(tok + 1) * e];
    let pe = &w.wpe[(pos % bs) * e..(pos % bs + 1) * e];
    let mut x: Vec<f32> = te.iter().zip(pe).map(|(a, b)| a + b).collect();
    let x0 = x.clone();

    let mut layers   = Vec::with_capacity(cfg.n_layer);
    let mut attn_ctx = Vec::with_capacity(cfg.n_layer);

    for l in 0..cfg.n_layer {
        let lw = &w.layers[l];
        let x_pre_attn = x.clone();
        let (xn_attn, rms_a) = rmsnorm(&x);

        let q = linear(&xn_attn, &lw.wq, e, e);
        let k = linear(&xn_attn, &lw.wk, e, e);
        let v = linear(&xn_attn, &lw.wv, e, e);

        // Accumulate KV cache (real causal attention)
        kv_cache[l].0.push(k);
        kv_cache[l].1.push(v);
        let all_k = &kv_cache[l].0;
        let all_v = &kv_cache[l].1;
        let t_len = all_k.len();
        let scale = (hd as f32).sqrt();

        let mut aw_all: Vec<Vec<f32>> = Vec::with_capacity(nh);
        let mut ho = zeros(e);

        for h in 0..nh {
            let qs = &q[h * hd..(h + 1) * hd];
            let logits: Vec<f32> = (0..t_len).map(|t| {
                let ks = &all_k[t][h * hd..(h + 1) * hd];
                qs.iter().zip(ks).map(|(a, b)| a * b).sum::<f32>() / scale
            }).collect();
            let aw_h = softmax(&logits);
            for i in 0..hd {
                let out_i: f32 = (0..t_len).map(|t| aw_h[t] * all_v[t][h * hd + i]).sum();
                ho[h * hd + i] = out_i;
            }
            aw_all.push(aw_h);
        }

        let attn_out = linear(&ho, &lw.wo, e, e);
        let x_post_attn: Vec<f32> = x_pre_attn.iter().zip(&attn_out).map(|(a, b)| a + b).collect();
        x = x_post_attn.clone();

        let (xn_mlp, rms_m) = rmsnorm(&x);
        let h1: Vec<f32>  = linear(&xn_mlp, &lw.fc1, 4 * e, e);
        let h1a: Vec<f32> = h1.iter().map(|v| if *v > 0.0 { v * v } else { 0.0 }).collect();
        let mlp_out       = linear(&h1a, &lw.fc2, e, 4 * e);
        let x_post_mlp: Vec<f32> = x.iter().zip(&mlp_out).map(|(a, b)| a + b).collect();
        x = x_post_mlp.clone();

        attn_ctx.push(AttnCtx {
            all_k: all_k.clone(), all_v: all_v.clone(), aw: aw_all, ho,
        });
        layers.push(LCache {
            x_pre_attn, xn_attn, rms_a, q, attn_out, x_post_attn,
            xn_mlp, rms_m, h1, h1a, mlp_out, x_post_mlp,
        });
    }

    let logits = linear(&x, &w.wte, vs, e);
    let probs  = softmax(&logits);
    PosCache { tok_id: tok, pos_id: pos, x0, layers, probs, attn_ctx }
}
