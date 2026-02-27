use crate::rng::Rng;

pub(crate) fn mat_rand(rows: usize, cols: usize, rng: &mut Rng, std: f32) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.gauss(std)).collect()
}

pub(crate) fn zeros(n: usize) -> Vec<f32> {
    vec![0.0f32; n]
}

/// y = x @ W^T   x:[inp]  W:[out, inp]  → y:[out]
pub(crate) fn linear(x: &[f32], w: &[f32], out: usize, inp: usize) -> Vec<f32> {
    (0..out).map(|o| x.iter().zip(&w[o * inp..(o + 1) * inp]).map(|(a, b)| a * b).sum()).collect()
}

pub(crate) fn linear_bwd_w(dw: &mut [f32], dy: &[f32], x: &[f32], out: usize, inp: usize) {
    for o in 0..out {
        for i in 0..inp {
            dw[o * inp + i] += dy[o] * x[i];
        }
    }
}

pub(crate) fn linear_bwd_x(dx: &mut [f32], dy: &[f32], w: &[f32], out: usize, inp: usize) {
    for i in 0..inp {
        dx[i] += (0..out).map(|o| dy[o] * w[o * inp + i]).sum::<f32>();
    }
}

pub(crate) fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = ex.iter().sum();
    ex.iter().map(|v| v / s).collect()
}

/// Returns (normed_x, rms_inv).
pub(crate) fn rmsnorm(x: &[f32]) -> (Vec<f32>, f32) {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let ri = (ms + 1e-5_f32).sqrt().recip();
    (x.iter().map(|v| v * ri).collect(), ri)
}

pub(crate) fn rmsnorm_bwd(dy: &[f32], x: &[f32], ri: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let dot: f32 = dy.iter().zip(x).map(|(a, b)| a * b).sum();
    dy.iter().zip(x).map(|(dy_i, x_i)| ri * dy_i - (ri * ri * ri / n) * dot * x_i).collect()
}
