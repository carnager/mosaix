//! Wavelet noise reduction on the demosaiced camera image ("à trous" B3
//! spline, soft thresholding). Works on square-rooted values, which makes the
//! sensor's signal-dependent (shot) noise roughly uniform, and separately on
//! luminance and two colour differences. The noise level is measured from
//! the finest wavelet scale, so the sliders mean the same on every camera
//! and ISO.

use mosaix_core::raw::NoiseReduction;
use rayon::prelude::*;

use crate::CameraImage;

/// Noise of each à-trous scale relative to the finest (B3 spline, white noise).
const SCALE_NOISE: [f32; 6] = [1.0, 0.225, 0.097, 0.046, 0.022, 0.011];

/// Pixels with a channel at or above this are left alone (clipping must stay
/// recognisable for highlight reconstruction).
const CLIP: f32 = 0.98;

/// B3 spline blur with holes `step` apart, edges clamped.
fn blur(src: &[f32], w: usize, h: usize, step: usize) -> Vec<f32> {
    const K: [f32; 3] = [6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..(y + 1) * w];
        let at = |x: isize| s[x.clamp(0, w as isize - 1) as usize];
        let (s1, s2) = (step as isize, 2 * step as isize);
        let (lo, hi) = ((2 * step).min(w), w.saturating_sub(2 * step).max((2 * step).min(w)));
        for x in (0..lo).chain(hi..w) {
            let xi = x as isize;
            row[x] = K[0] * s[x] + K[1] * (at(xi - s1) + at(xi + s1)) + K[2] * (at(xi - s2) + at(xi + s2));
        }
        // Interior: plain slices (vectorises).
        if hi > lo {
            let n = hi - lo;
            let c = &s[lo..hi];
            let (l1, r1) = (&s[lo - step..lo - step + n], &s[lo + step..lo + step + n]);
            let (l2, r2) = (&s[lo - 2 * step..lo - 2 * step + n], &s[lo + 2 * step..lo + 2 * step + n]);
            for (i, o) in row[lo..hi].iter_mut().enumerate() {
                *o = K[0] * c[i] + K[1] * (l1[i] + r1[i]) + K[2] * (l2[i] + r2[i]);
            }
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let r = |d: isize| {
            let yy = (y as isize + d * step as isize).clamp(0, h as isize - 1) as usize;
            &tmp[yy * w..(yy + 1) * w]
        };
        let (m2, m1, c, p1, p2) = (r(-2), r(-1), r(0), r(1), r(2));
        for x in 0..w {
            row[x] = K[0] * c[x] + K[1] * (m1[x] + p1[x]) + K[2] * (m2[x] + p2[x]);
        }
    });
    out
}

/// Half-size copy (2×2 box).
fn half(p: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0.0f32; hw * hh];
    out.par_chunks_mut(hw).enumerate().for_each(|(y, row)| {
        let (y0, y1) = (2 * y, (2 * y + 1).min(h - 1));
        for (x, o) in row.iter_mut().enumerate() {
            let (x0, x1) = (2 * x, (2 * x + 1).min(w - 1));
            *o = 0.25 * (p[y0 * w + x0] + p[y0 * w + x1] + p[y1 * w + x0] + p[y1 * w + x1]);
        }
    });
    (out, hw, hh)
}

/// Bilinear sample of a half-size plane at full-size pixel (x, y).
fn upsample_at(p: &[f32], hw: usize, hh: usize, x: usize, y: usize) -> f32 {
    let fx = ((x as f32 - 0.5) * 0.5).clamp(0.0, (hw - 1) as f32);
    let fy = ((y as f32 - 0.5) * 0.5).clamp(0.0, (hh - 1) as f32);
    let (x0, y0) = (fx as usize, fy as usize);
    let (x1, y1) = ((x0 + 1).min(hw - 1), (y0 + 1).min(hh - 1));
    let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
    let top = p[y0 * hw + x0] + (p[y0 * hw + x1] - p[y0 * hw + x0]) * ax;
    let bot = p[y1 * hw + x0] + (p[y1 * hw + x1] - p[y1 * hw + x0]) * ax;
    top + (bot - top) * ay
}

/// Colour planes: the finest scale is thresholded at full size, the rest
/// is denoised at half size (colour blotches are coarse; 4× less work).
fn denoise_chroma(p: &mut [f32], w: usize, h: usize, k: f32) {
    if k <= 0.0 {
        return;
    }
    let coarse = blur(p, w, h, 1);
    let mut fine: Vec<f32> = p.iter().zip(&coarse).map(|(a, b)| a - b).collect();
    let sigma0 = mad(&fine) / 0.6745;
    let t = k * sigma0;
    fine.par_iter_mut().for_each(|d| *d = d.signum() * (d.abs() - t).max(0.0));
    let (mut small, hw, hh) = half(&coarse, w, h);
    // At half size the noise is halved by the box; the levels continue at 2× scale.
    denoise_plane_sigma(&mut small, hw, hh, k, 5, sigma0 * SCALE_NOISE[1] / 0.5);
    p.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = upsample_at(&small, hw, hh, x, y) + fine[y * w + x];
        }
    });
}

/// Median absolute value of a sample of `v` (robust noise estimate).
fn mad(v: &[f32]) -> f32 {
    let step = (v.len() / 100_000).max(1);
    let mut s: Vec<f32> = v.iter().step_by(step).map(|x| x.abs()).collect();
    if s.is_empty() {
        return 0.0;
    }
    let mid = s.len() / 2;
    *s.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
}

/// Denoises one plane in place; `k` is the threshold in noise sigmas.
fn denoise_plane(p: &mut [f32], w: usize, h: usize, k: f32, levels: usize) {
    denoise_plane_sigma(p, w, h, k, levels, 0.0);
}

/// Like [`denoise_plane`]; `sigma0` is the finest scale's noise (0 = measure it).
fn denoise_plane_sigma(p: &mut [f32], w: usize, h: usize, k: f32, levels: usize, mut sigma0: f32) {
    if k <= 0.0 {
        return;
    }
    let mut cur = p.to_vec();
    let mut acc = vec![0.0f32; w * h];
    for (l, rel) in SCALE_NOISE.iter().enumerate().take(levels) {
        let next = blur(&cur, w, h, 1 << l);
        let mut detail: Vec<f32> = cur.iter().zip(&next).map(|(a, b)| a - b).collect();
        if l == 0 && sigma0 <= 0.0 {
            sigma0 = mad(&detail) / 0.6745;
        }
        let t = k * sigma0 * rel;
        detail.par_iter_mut().zip(acc.par_iter_mut()).for_each(|(d, a)| {
            // Soft threshold.
            *a += d.signum() * (d.abs() - t).max(0.0);
        });
        cur = next;
    }
    p.par_iter_mut().zip(acc.par_iter().zip(&cur)).for_each(|(o, (a, c))| *o = a + c);
}

/// Reduces noise in a camera image with `channels` colour channels.
pub fn denoise(cam: &mut CameraImage, n: &NoiseReduction, channels: usize) {
    if n.is_off() || cam.width < 8 || cam.height < 8 {
        return;
    }
    let (w, h) = (cam.width, cam.height);
    // Strength 0…100 → threshold 0…3 sigmas (colour noise is blotchier:
    // more scales, a little stronger).
    let kl = n.luminance.clamp(0.0, 100.0) / 100.0 * 3.0;
    let kc = n.color.clamp(0.0, 100.0) / 100.0 * 4.0;
    let sq = |v: f32| v.max(0.0).sqrt();
    if channels < 3 || cam.data.iter().take(1000).all(|p| p[0] == p[1] && p[1] == p[2]) {
        // Monochrome: luminance only.
        let mut y: Vec<f32> = cam.data.par_iter().map(|p| sq(p[0])).collect();
        denoise_plane(&mut y, w, h, kl, 5);
        cam.data.par_iter_mut().zip(&y).for_each(|(p, v)| {
            if p[0] < CLIP {
                let v = v.max(0.0) * v.max(0.0);
                *p = [v, v, v, p[3]];
            }
        });
        return;
    }
    // Luminance and colour differences of the square-rooted values.
    let mut y: Vec<f32> = cam.data.par_iter().map(|p| (sq(p[0]) + 2.0 * sq(p[1]) + sq(p[2])) / 4.0).collect();
    let mut cr: Vec<f32> = cam.data.par_iter().map(|p| sq(p[0]) - sq(p[1])).collect();
    let mut cb: Vec<f32> = cam.data.par_iter().map(|p| sq(p[2]) - sq(p[1])).collect();
    denoise_plane(&mut y, w, h, kl, 5);
    denoise_chroma(&mut cr, w, h, kc);
    denoise_chroma(&mut cb, w, h, kc);
    cam.data.par_iter_mut().enumerate().for_each(|(i, p)| {
        if p[0] >= CLIP || p[1] >= CLIP || p[2] >= CLIP {
            return;
        }
        let g = y[i] - (cr[i] + cb[i]) / 4.0;
        let r = cr[i] + g;
        let b = cb[i] + g;
        let back = |v: f32| v.max(0.0) * v.max(0.0);
        p[0] = back(r);
        p[1] = back(g);
        p[2] = back(b);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_noise_keeps_edges() {
        let (w, h) = (128, 128);
        let mut seed = 1u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        // Left half dark, right half bright, plus noise.
        let clean = |x: usize| if x < w / 2 { 0.1 } else { 0.5 };
        let data: Vec<[f32; 4]> = (0..w * h)
            .map(|i| {
                let c = clean(i % w);
                [c + 0.04 * rnd(), c + 0.04 * rnd(), c + 0.04 * rnd(), 0.0]
            })
            .collect();
        let err = |d: &[[f32; 4]]| {
            d.iter().enumerate().map(|(i, p)| (0..3).map(|k| (p[k] - clean(i % w)).powi(2)).sum::<f32>()).sum::<f32>()
                / (3 * w * h) as f32
        };
        let mut cam = CameraImage { width: w, height: h, data: data.clone() };
        denoise(&mut cam, &NoiseReduction { luminance: 60.0, color: 60.0 }, 3);
        let (before, after) = (err(&data), err(&cam.data));
        assert!(after < before * 0.4, "{before} → {after}");
        // The edge stays sharp: pixels two away from it keep their level.
        let row = &cam.data[64 * w..65 * w];
        assert!(row[w / 2 - 3][1] < 0.2 && row[w / 2 + 2][1] > 0.4, "{:?} {:?}", row[w / 2 - 3], row[w / 2 + 2]);
    }
}
