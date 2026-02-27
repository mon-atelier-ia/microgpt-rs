/// Tiny PRNG (xoshiro128+), zero dependencies.
pub struct Rng {
    s: [u32; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let lo = seed as u32;
        let hi = (seed >> 32) as u32;
        Self { s: [lo ^ 0xdeadbeef, hi ^ 0xcafebabe, lo.wrapping_add(1), hi.wrapping_add(1)] }
    }

    pub fn next_u32(&mut self) -> u32 {
        let r = self.s[0].wrapping_add(self.s[3]);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0]; self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2]; self.s[0] ^= self.s[3];
        self.s[2] ^= t; self.s[3] = self.s[3].rotate_left(11);
        r
    }

    pub fn gauss(&mut self, std: f32) -> f32 {
        let u1 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let u2 = self.next_u32() as f32 / u32::MAX as f32;
        std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn categorical(&mut self, probs: &[f32]) -> usize {
        let mut dart = (self.next_u32() as f32 / u32::MAX as f32) * probs.iter().sum::<f32>();
        for (i, &p) in probs.iter().enumerate() {
            dart -= p;
            if dart <= 0.0 { return i; }
        }
        probs.len() - 1
    }
}
