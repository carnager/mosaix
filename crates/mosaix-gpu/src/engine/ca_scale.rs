//! Edit › Content-Aware Scale, split so the slow part can run on a worker:
//! [`Engine::ca_scale_job`] reads the layer (GPU), [`CaScaleJob::run`] carves
//! it (any thread), [`Engine::apply_ca_scale`] puts it back (one undo step).

use std::collections::HashMap;
use std::sync::Arc;

use half::f16;
use mosaix_core::doc::Content;
use mosaix_core::{LayerId, TILE_SIZE, TileCoord};
use rayon::prelude::*;

use super::Engine;

/// Most output pixels.
const MAX_PIXELS: u64 = 250_000_000;

/// The layer's pixels, detached from the GPU.
pub struct CaScaleJob {
    layer: LayerId,
    base: usize,
    /// Content rectangle (document px).
    x0: i64,
    y0: i64,
    w: usize,
    h: usize,
    px: Vec<[f32; 4]>,
    protect: Option<Vec<u8>>,
    nw: usize,
    nh: usize,
    amount: f32,
}

/// The scaled layer, ready for [`Engine::apply_ca_scale`].
pub struct CaScaleResult {
    layer: LayerId,
    base: usize,
    tiles: Vec<(TileCoord, Vec<f16>)>,
    /// New content size.
    pub size: (u32, u32),
}

impl Engine {
    /// Reads the pixel layer's content. `target` turns its size into the new
    /// one; `amount` is 0 (plain resize) to 1 (fully content-aware);
    /// `protect` keeps seams out of the current selection.
    pub fn ca_scale_job(
        &mut self,
        layer: LayerId,
        target: impl FnOnce(u32, u32) -> (u32, u32),
        amount: f32,
        protect: bool,
    ) -> Result<CaScaleJob, &'static str> {
        use mosaix_core::tr;
        if self.stroke.is_some() {
            return Err(tr("not possible during a brush stroke"));
        }
        let l = self.doc.find(layer).ok_or(tr("Works on pixel layers"))?;
        let Content::Pixels(map) = &l.content else {
            return Err(tr("Works on pixel layers"));
        };
        if l.lock.pixels || l.lock.position {
            return Err(tr("The layer is locked"));
        }
        if l.mask.is_some() {
            return Err(tr("Apply or delete the layer mask first"));
        }
        if protect && !self.has_selection() {
            return Err(tr("Needs a selection"));
        }
        let map = map.clone();
        let base = Arc::as_ptr(&map) as usize;
        let coords: Vec<TileCoord> = map.keys().copied().collect();
        let handles: Vec<_> = coords.iter().map(|c| map[c].clone()).collect();
        let data = self.read_layer_tiles(&handles);

        let ts = TILE_SIZE as usize;
        let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for (c, px) in coords.iter().zip(&data) {
            let (ox, oy) = c.origin();
            for y in 0..ts {
                for x in 0..ts {
                    if px[(y * ts + x) * 4 + 3].to_f32() > 1.0 / 512.0 {
                        (x0, y0) = (x0.min(ox + x as i64), y0.min(oy + y as i64));
                        (x1, y1) = (x1.max(ox + x as i64 + 1), y1.max(oy + y as i64 + 1));
                    }
                }
            }
        }
        if x0 >= x1 {
            return Err(tr("Nothing to scale: the layer is empty."));
        }
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let (nw, nh) = target(w as u32, h as u32);
        if nw == 0 || nh == 0 || nw > 300_000 || nh > 300_000 || nw as u64 * nh as u64 > MAX_PIXELS || w as u64 * h as u64 > MAX_PIXELS {
            return Err(tr("invalid image size"));
        }

        let mut px = vec![[0.0f32; 4]; w * h];
        for (c, tile) in coords.iter().zip(&data) {
            let (ox, oy) = c.origin();
            for y in 0..ts {
                let dy = oy + y as i64 - y0;
                if dy < 0 || dy >= h as i64 {
                    continue;
                }
                for x in 0..ts {
                    let dx = ox + x as i64 - x0;
                    if dx < 0 || dx >= w as i64 {
                        continue;
                    }
                    let s = (y * ts + x) * 4;
                    px[dy as usize * w + dx as usize] = [0, 1, 2, 3].map(|k| tile[s + k].to_f32());
                }
            }
        }
        let protect = protect.then(|| self.selection_cpu()).flatten().map(|sel| {
            let mut m = vec![0u8; w * h];
            m.par_chunks_mut(w).enumerate().for_each(|(dy, row)| {
                let y = y0 + dy as i64;
                let mut x = x0;
                while x < x1 {
                    let c = TileCoord::containing(x as f64, y as f64);
                    let (ox, oy) = c.origin();
                    let end = (ox + ts as i64).min(x1);
                    let tile = sel.tile(c);
                    for q in x..end {
                        row[(q - x0) as usize] = tile.get((q - ox) as usize, (y - oy) as usize);
                    }
                    x = end;
                }
            });
            m
        });
        Ok(CaScaleJob { layer, base, x0, y0, w, h, px, protect, nw: nw as usize, nh: nh as usize, amount })
    }

    /// Puts the scaled pixels in. Err if the layer changed meanwhile.
    pub fn apply_ca_scale(&mut self, r: CaScaleResult, label: &str) -> Result<(), &'static str> {
        use mosaix_core::tr;
        let Some(Content::Pixels(map)) = self.doc.find(r.layer).map(|l| &l.content) else {
            return Err(tr("the image has changed in the meantime"));
        };
        if Arc::as_ptr(map) as usize != r.base {
            return Err(tr("the image has changed in the meantime"));
        }
        let new: HashMap<TileCoord, Vec<f16>> = r.tiles.into_iter().collect();
        let mut edits: Vec<(TileCoord, Option<Vec<f16>>)> = map.keys().filter(|c| !new.contains_key(c)).map(|c| (*c, None)).collect();
        edits.extend(new.into_iter().map(|(c, px)| (c, Some(px))));
        if self.replace_tiles(r.layer, edits, label) { Ok(()) } else { Err(tr("out of GPU memory")) }
    }
}

impl CaScaleJob {
    /// Carves the pixels; the result stays centred on the old content.
    pub fn run(self) -> CaScaleResult {
        let out = mosaix_core::seam::scale(self.w, self.h, &self.px, self.protect.as_deref(), self.nw, self.nh, self.amount);
        let ox = self.x0 + (self.w as i64 - self.nw as i64).div_euclid(2);
        let oy = self.y0 + (self.h as i64 - self.nh as i64).div_euclid(2);
        let ts = TILE_SIZE as i64;
        let (tx0, ty0) = (ox.div_euclid(ts), oy.div_euclid(ts));
        let (tx1, ty1) = ((ox + self.nw as i64 - 1).div_euclid(ts), (oy + self.nh as i64 - 1).div_euclid(ts));
        let coords: Vec<(i64, i64)> = (ty0..=ty1).flat_map(|y| (tx0..=tx1).map(move |x| (x, y))).collect();
        let (nw, nh) = (self.nw as i64, self.nh as i64);
        let tiles = coords
            .into_par_iter()
            .filter_map(|(tx, ty)| {
                let mut px = vec![f16::ZERO; (ts * ts * 4) as usize];
                let mut any = false;
                for y in 0..ts {
                    let dy = ty * ts + y - oy;
                    if dy < 0 || dy >= nh {
                        continue;
                    }
                    for x in 0..ts {
                        let dx = tx * ts + x - ox;
                        if dx < 0 || dx >= nw {
                            continue;
                        }
                        let p = out[(dy * nw + dx) as usize];
                        if p[3] > 1.0 / 2048.0 {
                            any = true;
                            let s = ((y * ts + x) * 4) as usize;
                            for k in 0..4 {
                                px[s + k] = f16::from_f32(p[k]);
                            }
                        }
                    }
                }
                any.then(|| (TileCoord::new(tx as i32, ty as i32), px))
            })
            .collect();
        CaScaleResult { layer: self.layer, base: self.base, tiles, size: (self.nw as u32, self.nh as u32) }
    }
}
