//! Built-in textured brush tips, generated procedurally at start-up into a
//! mipmapped R8 texture array (layer = tip index − 1). Index 0 (round) is
//! computed analytically in the shader and has no texture.

use mosaix_core::TIP_NAMES;

pub const TIP_SIZE: u32 = 256;
pub const TIP_MIPS: u32 = 9;
/// Layers of the tip texture: the built-in tips, then room for custom ones.
pub const TIP_LAYERS: u32 = 64;
/// Tip index of the first custom tip (`tip` 1…5 are built in, 0 is round).
pub const FIRST_CUSTOM_TIP: u32 = 6;
/// How many custom tips fit.
pub const MAX_CUSTOM_TIPS: usize = (TIP_LAYERS - FIRST_CUSTOM_TIP + 1) as usize;

/// The mip chain of one TIP_SIZE² tip (2×2 averaging).
pub fn mips(base: &[u8]) -> Vec<Vec<u8>> {
    let mut levels = vec![base.to_vec()];
    for mip in 1..TIP_MIPS {
        let size = (TIP_SIZE >> mip) as usize;
        let ps = size * 2;
        let p = &levels[mip as usize - 1];
        let next = (0..size * size)
            .map(|k| {
                let (x, y) = (k % size * 2, k / size * 2);
                let sum = p[y * ps + x] as u32 + p[y * ps + x + 1] as u32 + p[(y + 1) * ps + x] as u32 + p[(y + 1) * ps + x + 1] as u32;
                ((sum + 2) / 4) as u8
            })
            .collect();
        levels.push(next);
    }
    levels
}

impl super::Engine {
    /// Loads custom tips (TIP_SIZE² coverage each) as tips FIRST_CUSTOM_TIP…;
    /// extra ones beyond MAX_CUSTOM_TIPS are ignored.
    pub fn set_custom_tips(&mut self, tips: &[Vec<u8>]) {
        for (i, base) in tips.iter().take(MAX_CUSTOM_TIPS).enumerate() {
            if base.len() != (TIP_SIZE * TIP_SIZE) as usize {
                continue;
            }
            let layer = FIRST_CUSTOM_TIP - 1 + i as u32;
            for (mip, data) in mips(base).iter().enumerate() {
                let size = TIP_SIZE >> mip;
                self.queue.write_texture(
                    crate::pool::texel_copy(&self.tips_tex, mip as u32, layer),
                    data,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size), rows_per_image: Some(size) },
                    wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                );
            }
        }
    }
}

/// Smooth value noise in [0, 1] with integer lattice spacing `cell`.
fn value_noise(x: f32, y: f32, cell: f32, seed: u32) -> f32 {
    let hash = |ix: i32, iy: i32| {
        let mut h =
            (ix as u32).wrapping_mul(0x27d4_eb2d) ^ (iy as u32).wrapping_mul(0x1656_67b1) ^ seed;
        h ^= h >> 15;
        h = h.wrapping_mul(0x85eb_ca6b);
        h ^= h >> 13;
        (h & 0xffff) as f32 / 65535.0
    };
    let (fx, fy) = (x / cell, y / cell);
    let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(tx), s(ty));
    let a = hash(ix, iy) + (hash(ix + 1, iy) - hash(ix, iy)) * sx;
    let b = hash(ix, iy + 1) + (hash(ix + 1, iy + 1) - hash(ix, iy + 1)) * sx;
    a + (b - a) * sy
}

fn fbm(x: f32, y: f32, cell: f32, seed: u32) -> f32 {
    (value_noise(x, y, cell, seed) * 0.6
        + value_noise(x, y, cell * 0.5, seed ^ 0xabc) * 0.3
        + value_noise(x, y, cell * 0.25, seed ^ 0x123) * 0.1)
        .clamp(0.0, 1.0)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn rng(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state >> 8) as f32 / (1u32 << 24) as f32
}

/// Coverage (0…1) of one tip at pixel (x, y) with centre-relative coordinates in [-1, 1].
fn tip(index: usize, dots: &[(f32, f32, f32, f32)], x: f32, y: f32) -> f32 {
    let (u, v) = (
        (x + 0.5) / TIP_SIZE as f32 * 2.0 - 1.0,
        (y + 0.5) / TIP_SIZE as f32 * 2.0 - 1.0,
    );
    let r = (u * u + v * v).sqrt();
    match index {
        // Chalk: grainy disc.
        1 => {
            let disc = smoothstep(1.0, 0.8, r);
            let grain = fbm(x, y, 9.0, 7);
            disc * smoothstep(0.35, 0.65, grain * (1.15 - r * 0.4))
        }
        // Bristles, spray: soft dots.
        2 | 3 => dots
            .iter()
            .map(|&(cx, cy, cr, a)| {
                let d = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt();
                a * smoothstep(cr, cr * 0.4, d)
            })
            .fold(0.0, f32::max),
        // Rough: solid disc with a ragged edge.
        4 => {
            let edge = 0.78 + 0.2 * (fbm(u * 60.0 + 100.0, v * 60.0 + 100.0, 6.0, 3) - 0.5) * 2.0;
            smoothstep(edge, edge - 0.04, r)
        }
        // Charcoal: streaks along x, fading towards the ends.
        5 => {
            let streak = fbm(x * 0.15, y * 3.0, 6.0, 11);
            let shape = smoothstep(1.0, 0.75, u.abs()) * smoothstep(0.7, 0.4, v.abs());
            shape * smoothstep(0.3, 0.6, streak)
        }
        _ => 0.0,
    }
}

/// All textured tips with their mip chains: `levels[mip][layer]` row-major R8.
pub fn generate() -> Vec<Vec<Vec<u8>>> {
    let n = TIP_NAMES.len() - 1;
    let mut seed = 0x1234_5678u32;
    let mut bristles = Vec::new();
    for i in 0..24 {
        let a = i as f32 / 24.0 * std::f32::consts::TAU + rng(&mut seed) * 0.3;
        let d = 0.25 + 0.55 * rng(&mut seed);
        bristles.push((
            a.cos() * d,
            a.sin() * d * 0.7,
            0.07 + 0.06 * rng(&mut seed),
            0.6 + 0.4 * rng(&mut seed),
        ));
    }
    let mut spray = Vec::new();
    while spray.len() < 140 {
        let (x, y) = (rng(&mut seed) * 2.0 - 1.0, rng(&mut seed) * 2.0 - 1.0);
        let r = (x * x + y * y).sqrt();
        if r < 0.95 && rng(&mut seed) > r * 0.8 {
            spray.push((
                x,
                y,
                0.035 + 0.04 * rng(&mut seed),
                0.6 + 0.4 * rng(&mut seed),
            ));
        }
    }
    let base: Vec<Vec<u8>> = (1..=n)
        .map(|i| {
            let dots: &[(f32, f32, f32, f32)] = match i {
                2 => &bristles,
                3 => &spray,
                _ => &[],
            };
            (0..TIP_SIZE * TIP_SIZE)
                .map(|k| {
                    let (x, y) = ((k % TIP_SIZE) as f32, (k / TIP_SIZE) as f32);
                    (tip(i, dots, x, y) * 255.0).round() as u8
                })
                .collect()
        })
        .collect();
    let mut levels = vec![base];
    for mip in 1..TIP_MIPS {
        let size = (TIP_SIZE >> mip) as usize;
        let prev = &levels[mip as usize - 1];
        let next = prev
            .iter()
            .map(|p| {
                let ps = size * 2;
                (0..size * size)
                    .map(|k| {
                        let (x, y) = (k % size * 2, k / size * 2);
                        let sum = p[y * ps + x] as u32
                            + p[y * ps + x + 1] as u32
                            + p[(y + 1) * ps + x] as u32
                            + p[(y + 1) * ps + x + 1] as u32;
                        ((sum + 2) / 4) as u8
                    })
                    .collect()
            })
            .collect();
        levels.push(next);
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tips_have_content_and_fade_out_at_the_border() {
        let levels = generate();
        assert_eq!(levels.len(), TIP_MIPS as usize);
        for (i, t) in levels[0].iter().enumerate() {
            let sum: u64 = t.iter().map(|&v| v as u64).sum();
            let avg = sum as f64 / t.len() as f64;
            assert!(avg > 5.0, "tip {} nearly empty: {avg}", i + 1);
            let corner =
                t[0] as u32 + t[255] as u32 + t[255 * 256] as u32 + t[256 * 256 - 1] as u32;
            assert_eq!(corner, 0, "tip {} touches the corners", i + 1);
        }
        assert_eq!(levels[8][0].len(), 1);
    }
}
