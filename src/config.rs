/// GPT model architecture parameters.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub block_size: usize,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_embd: 16,
            n_head: 4,
            n_layer: 1,
            block_size: 16,
        }
    }
}

/// Training and inference hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub n_steps: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub std_init: f64,
    pub temperature: f64,
    pub n_samples: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_steps: 1000,
            lr: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            eps: 1e-8,
            std_init: 0.08,
            temperature: 0.5,
            n_samples: 5,
        }
    }
}
