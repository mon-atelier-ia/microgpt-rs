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
            block_size: 8,
        }
    }
}

/// Training hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub n_steps: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_steps: 5000,
            lr: 1e-2,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
        }
    }
}
