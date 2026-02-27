use crate::config::{ModelConfig, TrainConfig};
use crate::ops::{mat_rand, zeros};
use crate::rng::Rng;

#[derive(Clone)]
pub struct LayerParams {
    pub wq:  Vec<f32>, // [n_embd, n_embd]
    pub wk:  Vec<f32>,
    pub wv:  Vec<f32>,
    pub wo:  Vec<f32>,
    pub fc1: Vec<f32>, // [4*n_embd, n_embd]
    pub fc2: Vec<f32>, // [n_embd, 4*n_embd]
}

impl LayerParams {
    fn new_random(rng: &mut Rng, cfg: &ModelConfig) -> Self {
        let e = cfg.n_embd;
        Self {
            wq:  mat_rand(e, e, rng, 0.02),
            wk:  mat_rand(e, e, rng, 0.02),
            wv:  mat_rand(e, e, rng, 0.02),
            wo:  zeros(e * e),                   // zero-init (matches Karpathy)
            fc1: mat_rand(4 * e, e, rng, 0.02),
            fc2: zeros(e * 4 * e),               // zero-init (matches Karpathy)
        }
    }

    fn new_zeros(cfg: &ModelConfig) -> Self {
        let e = cfg.n_embd;
        Self {
            wq: zeros(e * e), wk: zeros(e * e), wv: zeros(e * e), wo: zeros(e * e),
            fc1: zeros(4 * e * e), fc2: zeros(e * 4 * e),
        }
    }

    fn zero_fill(&mut self) {
        for v in [&mut self.wq, &mut self.wk, &mut self.wv, &mut self.wo, &mut self.fc1, &mut self.fc2] {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}

#[derive(Clone)]
pub struct Params {
    pub wte: Vec<f32>, // [vocab_size, n_embd]
    pub wpe: Vec<f32>, // [block_size, n_embd]
    pub layers: Vec<LayerParams>,
}

impl Params {
    fn new_random(vocab_size: usize, rng: &mut Rng, cfg: &ModelConfig) -> Self {
        Self {
            wte: mat_rand(vocab_size, cfg.n_embd, rng, 0.02),
            wpe: mat_rand(cfg.block_size, cfg.n_embd, rng, 0.02),
            layers: (0..cfg.n_layer).map(|_| LayerParams::new_random(rng, cfg)).collect(),
        }
    }

    fn new_zeros(vocab_size: usize, cfg: &ModelConfig) -> Self {
        Self {
            wte: zeros(vocab_size * cfg.n_embd),
            wpe: zeros(cfg.block_size * cfg.n_embd),
            layers: (0..cfg.n_layer).map(|_| LayerParams::new_zeros(cfg)).collect(),
        }
    }

    pub(crate) fn zero_fill(&mut self) {
        self.wte.iter_mut().for_each(|x| *x = 0.0);
        self.wpe.iter_mut().for_each(|x| *x = 0.0);
        for layer in &mut self.layers {
            layer.zero_fill();
        }
    }
}

pub struct Model {
    pub config: ModelConfig,
    pub vocab_size: usize,
    pub w: Params,
    pub g: Params,
    pub m: Params,
    pub v: Params,
}

impl Model {
    pub fn new(vocab_size: usize, rng: &mut Rng, config: ModelConfig) -> Self {
        Self {
            w: Params::new_random(vocab_size, rng, &config),
            g: Params::new_zeros(vocab_size, &config),
            m: Params::new_zeros(vocab_size, &config),
            v: Params::new_zeros(vocab_size, &config),
            vocab_size,
            config,
        }
    }

    pub fn zero_grad(&mut self) {
        self.g.zero_fill();
    }

    pub fn param_count(&self) -> usize {
        fn count_layer(l: &LayerParams) -> usize {
            l.wq.len() + l.wk.len() + l.wv.len() + l.wo.len() + l.fc1.len() + l.fc2.len()
        }
        self.w.wte.len() + self.w.wpe.len() + self.w.layers.iter().map(count_layer).sum::<usize>()
    }

    pub fn adam_step(&mut self, step: usize, tc: &TrainConfig) {
        let decay = 1.0 - step as f32 / tc.n_steps as f32;
        let lr_t = tc.lr * decay
            * (1.0 - tc.beta2.powi(step as i32)).sqrt()
            / (1.0 - tc.beta1.powi(step as i32));

        fn update(w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr_t: f32, tc: &TrainConfig) {
            for i in 0..w.len() {
                m[i] = tc.beta1 * m[i] + (1.0 - tc.beta1) * g[i];
                v[i] = tc.beta2 * v[i] + (1.0 - tc.beta2) * g[i] * g[i];
                w[i] -= lr_t * m[i] / (v[i].sqrt() + tc.eps);
            }
        }

        let Self { ref mut w, ref g, ref mut m, ref mut v, .. } = *self;

        update(&mut w.wte, &g.wte, &mut m.wte, &mut v.wte, lr_t, tc);
        update(&mut w.wpe, &g.wpe, &mut m.wpe, &mut v.wpe, lr_t, tc);
        for l in 0..w.layers.len() {
            update(&mut w.layers[l].wq,  &g.layers[l].wq,  &mut m.layers[l].wq,  &mut v.layers[l].wq,  lr_t, tc);
            update(&mut w.layers[l].wk,  &g.layers[l].wk,  &mut m.layers[l].wk,  &mut v.layers[l].wk,  lr_t, tc);
            update(&mut w.layers[l].wv,  &g.layers[l].wv,  &mut m.layers[l].wv,  &mut v.layers[l].wv,  lr_t, tc);
            update(&mut w.layers[l].wo,  &g.layers[l].wo,  &mut m.layers[l].wo,  &mut v.layers[l].wo,  lr_t, tc);
            update(&mut w.layers[l].fc1, &g.layers[l].fc1, &mut m.layers[l].fc1, &mut v.layers[l].fc1, lr_t, tc);
            update(&mut w.layers[l].fc2, &g.layers[l].fc2, &mut m.layers[l].fc2, &mut v.layers[l].fc2, lr_t, tc);
        }
    }
}
