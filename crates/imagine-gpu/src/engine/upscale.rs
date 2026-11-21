//! Enlarging the image with a computed upscaler (AI): split like the
//! content-aware fill so the slow part runs on a worker thread:
//! [`Engine::upscale_job`] reads every pixel layer (GPU),
//! [`UpscaleJob::run_with`] enlarges them (any thread),
//! [`Engine::apply_upscale`] scales the document and puts the new pixels in
//! (one undo step).

use std::collections::HashMap;
use std::sync::Arc;

use glam::DAffine2;
use half::f16;
use imagine_core::color::{linear_to_srgb, srgb_to_linear};
use imagine_core::doc::Content;
use imagine_core::{LayerId, TILE_SIZE, TileCoord};
use rayon::prelude::*;

use super::Engine;
use super::transform::Interp;

/// One pixel layer, read: straight sRGB and alpha over its tile bounds.
struct Plane {
    id: LayerId,
    /// Tile-aligned origin (document px).
    x0: i64,
    y0: i64,
    w: usize,
    h: usize,
    rgb: Vec<[f32; 3]>,
    alpha: Vec<f32>,
}

/// Everything the upscale needs, detached from the GPU.
pub struct UpscaleJob {
    factor: u32,
    size: (u32, u32),
    /// Every pixel layer's tile map when the job started.
    bases: Vec<(LayerId, usize)>,
    planes: Vec<Plane>,
}

/// The enlarged layers, ready for [`Engine::apply_upscale`].
pub struct UpscaleResult {
    factor: u32,
    size: (u32, u32),
    bases: Vec<(LayerId, usize)>,
    layers: Vec<(LayerId, HashMap<TileCoord, Vec<f16>>)>,
    pub ms: f64,
}

impl Engine {
    fn pixel_bases(&self) -> Vec<(LayerId, usize)> {
        let mut v = Vec::new();
        self.doc.for_each(&mut |l, _| {
            if let Content::Pixels(map) = &l.content {
                v.push((l.id, Arc::as_ptr(map) as usize));
            }
        });
        v
    }

    /// Reads every pixel layer for enlarging by `factor` (2…8).
    pub fn upscale_job(&mut self, factor: u32) -> Result<UpscaleJob, &'static str> {
        if !(2..=8).contains(&factor) {
            return Err(imagine_core::tr("the factor must be 2 to 8"));
        }
        let (w, h) = self.size();
        if (w as u64 * factor as u64) > 300_000 || (h as u64 * factor as u64) > 300_000 {
            return Err(imagine_core::tr("invalid image size"));
        }
        if self.stroke.is_some() {
            return Err(imagine_core::tr("not possible during a brush stroke"));
        }
        let mut layers = Vec::new();
        self.doc.for_each(&mut |l, _| {
            if let Content::Pixels(map) = &l.content
                && !map.is_empty()
            {
                layers.push((l.id, map.clone()));
            }
        });
        let ts = TILE_SIZE as i64;
        let mut planes = Vec::new();
        for (id, map) in layers {
            let (mut tx0, mut ty0, mut tx1, mut ty1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for c in map.keys() {
                (tx0, ty0, tx1, ty1) = (tx0.min(c.x), ty0.min(c.y), tx1.max(c.x), ty1.max(c.y));
            }
            let coords: Vec<TileCoord> = map.keys().copied().collect();
            let handles: Vec<_> = coords.iter().map(|c| map[c].clone()).collect();
            let data = self.read_layer_tiles(&handles);
            let (pw, ph) = (((tx1 - tx0 + 1) as i64 * ts) as usize, ((ty1 - ty0 + 1) as i64 * ts) as usize);
            let mut rgb = vec![[0.0f32; 3]; pw * ph];
            let mut alpha = vec![0.0f32; pw * ph];
            for (c, px) in coords.iter().zip(&data) {
                let (ox, oy) = (((c.x - tx0) as i64 * ts) as usize, ((c.y - ty0) as i64 * ts) as usize);
                for y in 0..ts as usize {
                    for x in 0..ts as usize {
                        let s = (y * ts as usize + x) * 4;
                        let a = px[s + 3].to_f32().clamp(0.0, 1.0);
                        let i = (oy + y) * pw + ox + x;
                        alpha[i] = a;
                        if a > 1e-6 {
                            rgb[i] = [0, 1, 2].map(|k| linear_to_srgb((px[s + k].to_f32() / a).clamp(0.0, 1.0)));
                        }
                    }
                }
            }
            bleed(&mut rgb, &alpha, pw, ph);
            planes.push(Plane { id, x0: tx0 as i64 * ts, y0: ty0 as i64 * ts, w: pw, h: ph, rgb, alpha });
        }
        Ok(UpscaleJob { factor, size: (w, h), bases: self.pixel_bases(), planes })
    }

    /// Scales the document by the job's factor (vectors, masks and the rest
    /// as Image Size does) with the computed pixels in the pixel layers.
    /// Err if the document changed meanwhile.
    pub fn apply_upscale(&mut self, r: UpscaleResult, label: &str) -> Result<(), &'static str> {
        if self.pixel_bases() != r.bases || self.size() != r.size {
            return Err(imagine_core::tr("the image has changed in the meantime"));
        }
        let (w, h) = (r.size.0 * r.factor, r.size.1 * r.factor);
        let mut out = self.canvas_doc(w, h, DAffine2::from_scale(glam::DVec2::splat(r.factor as f64)), Interp::Bilinear)?;
        for (id, tiles) in r.layers {
            let mut map = HashMap::with_capacity(tiles.len());
            for (c, px) in tiles {
                let t = self.write_new_tile(&px).ok_or(imagine_core::tr("out of GPU memory"))?;
                map.insert(c, t);
            }
            if let Some(l) = out.find_mut(id) {
                l.content = Content::Pixels(Arc::new(map));
            }
        }
        self.edit(label, |d| {
            *d = out;
            true
        });
        Ok(())
    }
}

/// Gives fully transparent pixels the colour of their opaque neighbours, so
/// enlarging does not pull dark fringes into edges.
fn bleed(rgb: &mut [[f32; 3]], alpha: &[f32], w: usize, h: usize) {
    let mut known: Vec<bool> = alpha.iter().map(|&a| a > 1e-6).collect();
    if known.iter().all(|&k| k) || !known.iter().any(|&k| k) {
        return;
    }
    for _ in 0..8 {
        let mut next = known.clone();
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if known[i] {
                    continue;
                }
                let (mut s, mut n) = ([0.0f32; 3], 0.0);
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h && known[yy as usize * w + xx as usize] {
                        let p = rgb[yy as usize * w + xx as usize];
                        s = [s[0] + p[0], s[1] + p[1], s[2] + p[2]];
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    rgb[i] = s.map(|v| v / n);
                    next[i] = true;
                    changed = true;
                }
            }
        }
        known = next;
        if !changed {
            break;
        }
    }
}

impl UpscaleJob {
    pub fn factor(&self) -> u32 {
        self.factor
    }

    /// Pixels to compute (for time estimates).
    pub fn pixels(&self) -> usize {
        self.planes.iter().map(|p| p.w * p.h).sum()
    }

    /// Enlarges with `f(w, h, rgb, alpha) -> (scale, rgb × scale)`; a model
    /// scale above the factor is reduced (box filter), alpha is enlarged
    /// bilinearly.
    pub fn run_with<E>(
        self,
        mut f: impl FnMut(usize, usize, &[[f32; 3]], &[f32]) -> Result<(usize, Vec<[f32; 3]>), E>,
        mut too_small: impl FnMut() -> E,
    ) -> Result<UpscaleResult, E> {
        let t0 = std::time::Instant::now();
        let k = self.factor as usize;
        let ts = TILE_SIZE as usize;
        let mut layers = Vec::new();
        for p in self.planes {
            let (m, big) = f(p.w, p.h, &p.rgb, &p.alpha)?;
            if m < k || m % k != 0 {
                return Err(too_small());
            }
            let r = m / k;
            let (ow, oh) = (p.w * k, p.h * k);
            let (bw, _) = (p.w * m, p.h * m);
            let (gw, gh) = (ow / ts, oh / ts);
            let (ox, oy) = ((p.x0 * k as i64 / ts as i64) as i32, (p.y0 * k as i64 / ts as i64) as i32);
            let alpha_at = |x: usize, y: usize| -> f32 {
                // Bilinear in source pixels.
                let fx = ((x as f32 + 0.5) / k as f32 - 0.5).clamp(0.0, (p.w - 1) as f32);
                let fy = ((y as f32 + 0.5) / k as f32 - 0.5).clamp(0.0, (p.h - 1) as f32);
                let (x0, y0, ax, ay) = (fx as usize, fy as usize, fx.fract(), fy.fract());
                let (x1, y1) = ((x0 + 1).min(p.w - 1), (y0 + 1).min(p.h - 1));
                let a = |x: usize, y: usize| p.alpha[y * p.w + x];
                let top = a(x0, y0) + (a(x1, y0) - a(x0, y0)) * ax;
                let bot = a(x0, y1) + (a(x1, y1) - a(x0, y1)) * ax;
                top + (bot - top) * ay
            };
            let tiles: HashMap<TileCoord, Vec<f16>> = (0..gw * gh)
                .into_par_iter()
                .filter_map(|i| {
                    let (gx, gy) = (i % gw, i / gw);
                    let mut px = vec![f16::ZERO; ts * ts * 4];
                    let mut any = false;
                    for y in 0..ts {
                        for x in 0..ts {
                            let (qx, qy) = (gx * ts + x, gy * ts + y);
                            let a = alpha_at(qx, qy);
                            if a <= 1.0 / 1024.0 {
                                continue;
                            }
                            any = true;
                            let mut c = [0.0f32; 3];
                            for yy in 0..r {
                                for xx in 0..r {
                                    let q = big[(qy * r + yy) * bw + qx * r + xx];
                                    c = [c[0] + q[0], c[1] + q[1], c[2] + q[2]];
                                }
                            }
                            let n = (r * r) as f32;
                            let s = (y * ts + x) * 4;
                            for kk in 0..3 {
                                px[s + kk] = f16::from_f32(srgb_to_linear(c[kk] / n) * a);
                            }
                            px[s + 3] = f16::from_f32(a.min(1.0));
                        }
                    }
                    any.then(|| (TileCoord::new(ox + gx as i32, oy + gy as i32), px))
                })
                .collect();
            layers.push((p.id, tiles));
        }
        Ok(UpscaleResult { factor: self.factor, size: self.size, bases: self.bases, layers, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }
}
