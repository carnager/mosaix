//! Content-aware fill of the selection on the active layer. Split in three
//! so the expensive part can run on a worker thread:
//! [`Engine::fill_job`] reads the area (GPU), [`FillJob::run`] computes
//! (CPU, `Send`), [`Engine::apply_fill`] writes the result as one undo step.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use half::f16;
use imagine_core::color::{linear_to_srgb, srgb_to_linear};
use imagine_core::fill::{LevelSolver, content_aware_fill_with};
use imagine_core::{LayerId, MaskTile, TILE_SIZE, TileCoord};

use super::{Engine, StrokeTarget};
use crate::pool::Tile;

type Px = [f32; 4];

/// Everything the fill needs, detached from the GPU.
pub struct FillJob {
    layer: LayerId,
    /// The layer's tile map when the job started (to detect later edits).
    base: usize,
    x0: i64,
    y0: i64,
    w: usize,
    h: usize,
    /// Premultiplied linear pixels of the layer (destination).
    dest: Vec<Px>,
    /// The pixels to learn from (layer or flattened image).
    source: Vec<Px>,
    /// Selection 0…1.
    mask: Vec<f32>,
    tiles: HashMap<TileCoord, Vec<f16>>,
    seed: u64,
    /// Large levels run on the GPU (None: CPU only).
    gpu: Option<super::cafill_gpu::FillGpu>,
}

/// The computed fill, ready for [`Engine::apply_fill`].
pub struct FillResult {
    layer: LayerId,
    base: usize,
    tiles: Vec<(TileCoord, Option<Vec<f16>>)>,
    pub ms: f64,
}

fn encode(p: Px) -> Px {
    let a = p[3];
    if a <= 1e-6 {
        return [0.0; 4];
    }
    [
        linear_to_srgb((p[0] / a).clamp(0.0, 1.0)),
        linear_to_srgb((p[1] / a).clamp(0.0, 1.0)),
        linear_to_srgb((p[2] / a).clamp(0.0, 1.0)),
        a,
    ]
}

fn decode(p: Px) -> Px {
    let a = p[3].clamp(0.0, 1.0);
    [
        srgb_to_linear(p[0]) * a,
        srgb_to_linear(p[1]) * a,
        srgb_to_linear(p[2]) * a,
        a,
    ]
}

impl Engine {
    /// Collects what the fill of the current selection needs.
    pub fn fill_job(&mut self, all_layers: bool) -> Result<FillJob, &'static str> {
        self.fill_job_sized(all_layers, 0, 0)
    }

    /// Like [`Engine::fill_job`], with at least `min_side` pixels of context
    /// per side where the document allows (AI inpainting sees 512²), the
    /// selection grown by `grow` pixels.
    pub fn fill_job_sized(
        &mut self,
        all_layers: bool,
        min_side: i64,
        grow: usize,
    ) -> Result<FillJob, &'static str> {
        let layer = self.check_target(StrokeTarget::Pixels)?;
        let mask_tiles = self.selection_tiles()?;
        self.fill_job_tiles(layer, mask_tiles, all_layers, min_side, grow)
    }

    /// A fill job for a painted area (`[x, y, radius]` dabs) on `layer`.
    pub fn fill_job_dabs(
        &mut self,
        layer: LayerId,
        dabs: &[[f32; 3]],
        all_layers: bool,
        min_side: i64,
    ) -> Result<FillJob, &'static str> {
        if self.doc.find(layer).and_then(|l| l.tiles()).is_none() {
            return Err(imagine_core::tr("the active layer has no pixels to edit"));
        }
        let ts = TILE_SIZE as i64;
        let (dw, dh) = (self.doc.width as i64, self.doc.height as i64);
        let mut mask_tiles: HashMap<TileCoord, Vec<u8>> = HashMap::new();
        for d in dabs {
            let (cx, cy, r) = (d[0], d[1], d[2].max(0.5));
            let x0 = ((cx - r - 1.0).floor() as i64).max(0);
            let y0 = ((cy - r - 1.0).floor() as i64).max(0);
            let x1 = ((cx + r + 1.0).ceil() as i64).min(dw);
            let y1 = ((cy + r + 1.0).ceil() as i64).min(dh);
            for y in y0..y1 {
                for x in x0..x1 {
                    let dist = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
                    let v = ((r - dist + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                    if v == 0 {
                        continue;
                    }
                    let c = TileCoord::new(x.div_euclid(ts) as i32, y.div_euclid(ts) as i32);
                    let t = mask_tiles.entry(c).or_insert_with(|| vec![0; (TILE_SIZE * TILE_SIZE) as usize]);
                    let i = (y.rem_euclid(ts) * ts + x.rem_euclid(ts)) as usize;
                    t[i] = t[i].max(v);
                }
            }
        }
        if mask_tiles.is_empty() {
            return Err(imagine_core::tr("the selection is empty"));
        }
        self.fill_job_tiles(layer, mask_tiles, all_layers, min_side, 0)
    }

    /// The selection's values per selected tile.
    fn selection_tiles(&mut self) -> Result<HashMap<TileCoord, Vec<u8>>, &'static str> {
        let sel = self
            .doc
            .selection
            .clone()
            .ok_or(imagine_core::tr("first select the area to fill"))?;
        if sel.default > 0 {
            return Err(imagine_core::tr(
                "the selection extends to infinity – please select an area",
            ));
        }
        let selected: Vec<TileCoord> = sel
            .tiles
            .iter()
            .filter(|(_, t)| !matches!(t, MaskTile::Solid(0)))
            .map(|(c, _)| *c)
            .filter(|c| self.in_document(*c))
            .collect();
        if selected.is_empty() {
            return Err(imagine_core::tr("the selection is empty"));
        }
        let pix: Vec<(TileCoord, Tile)> = selected
            .iter()
            .filter_map(|c| match &sel.tiles[c] {
                MaskTile::Pixels(t) => Some((*c, t.clone())),
                MaskTile::Solid(_) => None,
            })
            .collect();
        let data = self.read_mask_tiles(&pix.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>());
        let mut mask_tiles: HashMap<TileCoord, Vec<u8>> =
            pix.iter().map(|(c, _)| *c).zip(data).collect();
        for c in &selected {
            if let MaskTile::Solid(v) = sel.tiles[c] {
                mask_tiles.insert(*c, vec![v; (TILE_SIZE * TILE_SIZE) as usize]);
            }
        }
        Ok(mask_tiles)
    }

    fn fill_job_tiles(
        &mut self,
        layer: LayerId,
        mask_tiles: HashMap<TileCoord, Vec<u8>>,
        all_layers: bool,
        min_side: i64,
        grow: usize,
    ) -> Result<FillJob, &'static str> {
        let ts = TILE_SIZE as i64;
        // Bounding box of the selected pixels.
        let (mut bx0, mut by0, mut bx1, mut by1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for (c, m) in &mask_tiles {
            for (i, &v) in m.iter().enumerate() {
                if v > 0 {
                    let (x, y) = (
                        c.x as i64 * ts + (i as i64 % ts),
                        c.y as i64 * ts + (i as i64 / ts),
                    );
                    (bx0, by0, bx1, by1) = (bx0.min(x), by0.min(y), bx1.max(x), by1.max(y));
                }
            }
        }
        if bx0 > bx1 {
            return Err(imagine_core::tr("the selection is empty"));
        }
        // Learn from a margin around the hole (larger holes look further).
        let size = (bx1 - bx0).max(by1 - by0) + 1;
        let margin = (size * 3 / 4).max(64);
        let (dw, dh) = (self.doc.width as i64, self.doc.height as i64);
        let mx = margin.max((min_side - (bx1 - bx0 + 1) + 1) / 2);
        let my = margin.max((min_side - (by1 - by0 + 1) + 1) / 2);
        let x0 = (bx0 - mx).max(0);
        let y0 = (by0 - my).max(0);
        let x1 = (bx1 + 1 + mx).min(dw);
        let y1 = (by1 + 1 + my).min(dh);
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let (dest, tiles) = self.read_region(Some(layer), x0, y0, w, h);
        let source = if all_layers {
            self.read_region(None, x0, y0, w, h).0
        } else {
            dest.clone()
        };
        let mut mask = vec![0.0f32; w * h];
        for (c, m) in &mask_tiles {
            let (ox, oy) = (c.x as i64 * ts - x0, c.y as i64 * ts - y0);
            for (i, &v) in m.iter().enumerate() {
                let (x, y) = (ox + i as i64 % ts, oy + i as i64 / ts);
                if v > 0 && x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    mask[y as usize * w + x as usize] = v as f32 / 255.0;
                }
            }
        }
        if grow > 0 {
            dilate(&mut mask, w, h, grow);
        }
        let base = self
            .doc
            .find(layer)
            .and_then(|l| l.tiles())
            .map_or(0, |t| Arc::as_ptr(t) as usize);
        let seed = (x0 as u64) << 32 ^ y0 as u64 ^ (w as u64) << 16;
        let gpu = self
            .fill_gpu
            .get_or_insert_with(|| super::cafill_gpu::FillGpu::new(&self.device, &self.queue))
            .clone();
        Ok(FillJob {
            layer,
            base,
            x0,
            y0,
            w,
            h,
            dest,
            source,
            mask,
            tiles,
            seed,
            gpu: Some(gpu),
        })
    }

    /// Writes a computed fill (one undo step). False if the layer changed meanwhile.
    pub fn apply_fill(&mut self, r: FillResult, label: &str) -> bool {
        let now = self
            .doc
            .find(r.layer)
            .and_then(|l| l.tiles())
            .map_or(0, |t| Arc::as_ptr(t) as usize);
        if now != r.base {
            tracing::warn!(
                "{}",
                imagine_core::tr!(
                    "{label}: discarded, the layer has changed in the meantime",
                    label = label
                )
            );
            return false;
        }
        self.replace_tiles(r.layer, r.tiles, label)
    }

    /// Content-aware fill of the selection, synchronously (agents, tests).
    pub fn content_aware_fill(
        &mut self,
        all_layers: bool,
        label: &str,
    ) -> Result<f64, &'static str> {
        let r = self.fill_job(all_layers)?.run()?;
        let ms = r.ms;
        if !self.apply_fill(r, label) {
            return Err(imagine_core::tr("the layer has changed in the meantime"));
        }
        Ok(ms)
    }
}

impl FillJob {
    /// Computes on the CPU only (comparison, tests).
    pub fn cpu_only(mut self) -> Self {
        self.gpu = None;
        self
    }

    /// The fill itself (CPU plus GPU for large levels; any thread).
    pub fn run(self) -> Result<FillResult, &'static str> {
        let t0 = std::time::Instant::now();
        let FillJob {
            layer,
            base,
            x0,
            y0,
            w,
            h,
            dest,
            source,
            mask,
            tiles,
            seed,
            mut gpu,
        } = self;
        let hole: Vec<bool> = mask.iter().map(|&m| m > 0.0).collect();
        let allowed = vec![true; w * h];
        let mut img: Vec<Px> = source.iter().map(|&p| encode(p)).collect();
        let solver = gpu.as_mut().map(|g| g as &mut dyn LevelSolver);
        if !content_aware_fill_with(w, h, &mut img, &hole, &allowed, seed, solver) {
            return Err(imagine_core::tr(
                "there is too little image around the selection to fill from",
            ));
        }
        Ok(write_back(
            layer, base, x0, y0, w, &dest, &mask, &img, tiles, t0,
        ))
    }

    /// The fill computed by `f` (any thread): it gets the area as straight
    /// sRGB (`w × h`) and the hole, and replaces the hole's pixels.
    pub fn run_with<E>(
        self,
        f: impl FnOnce(usize, usize, &mut [[f32; 3]], &[bool]) -> Result<(), E>,
    ) -> Result<FillResult, E> {
        let t0 = std::time::Instant::now();
        let FillJob {
            layer,
            base,
            x0,
            y0,
            w,
            h,
            dest,
            source,
            mask,
            tiles,
            ..
        } = self;
        let hole: Vec<bool> = mask.iter().map(|&m| m > 0.0).collect();
        let enc: Vec<Px> = source.iter().map(|&p| encode(p)).collect();
        let mut rgb: Vec<[f32; 3]> = enc.iter().map(|p| [p[0], p[1], p[2]]).collect();
        f(w, h, &mut rgb, &hole)?;
        // The filled pixels are opaque.
        let img: Vec<Px> = rgb
            .iter()
            .zip(&enc)
            .zip(&hole)
            .map(|((c, e), &hl)| [c[0], c[1], c[2], if hl { 1.0 } else { e[3] }])
            .collect();
        Ok(write_back(
            layer, base, x0, y0, w, &dest, &mask, &img, tiles, t0,
        ))
    }
}

/// Grows a mask by `r` pixels (round, max of the values within reach).
fn dilate(mask: &mut [f32], w: usize, h: usize, r: usize) {
    use rayon::prelude::*;
    let src = mask.to_vec();
    let r = r as i64;
    mask.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y = y as i64;
        for x in 0..w as i64 {
            let mut m = 0.0f32;
            for dy in -r..=r {
                let yy = y + dy;
                if yy < 0 || yy >= h as i64 {
                    continue;
                }
                let span = ((r * r - dy * dy) as f64).sqrt() as i64;
                for xx in (x - span).max(0)..=(x + span).min(w as i64 - 1) {
                    m = m.max(src[yy as usize * w + xx as usize]);
                }
            }
            row[x as usize] = m;
        }
    });
}

/// Blends the filled pixels into the layer's tiles by the selection.
#[allow(clippy::too_many_arguments)]
fn write_back(
    layer: LayerId,
    base: usize,
    x0: i64,
    y0: i64,
    w: usize,
    dest: &[Px],
    mask: &[f32],
    img: &[Px],
    mut tiles: HashMap<TileCoord, Vec<f16>>,
    t0: std::time::Instant,
) -> FillResult {
    let h = mask.len() / w;
    let ts = TILE_SIZE as i64;
    let mut touched = BTreeSet::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let m = mask[i];
            if m <= 0.0 {
                continue;
            }
            let f = decode(img[i]);
            let o = [0, 1, 2, 3].map(|c| dest[i][c] + (f[c] - dest[i][c]) * m);
            let (gx, gy) = (x0 + x as i64, y0 + y as i64);
            let c = TileCoord::new(gx.div_euclid(ts) as i32, gy.div_euclid(ts) as i32);
            let tile = tiles
                .entry(c)
                .or_insert_with(|| vec![f16::ZERO; (TILE_SIZE * TILE_SIZE * 4) as usize]);
            let s = ((gy.rem_euclid(ts) * ts + gx.rem_euclid(ts)) * 4) as usize;
            for k in 0..4 {
                tile[s + k] = f16::from_f32(o[k]);
            }
            touched.insert(c);
        }
    }
    let tiles = touched.into_iter().map(|c| (c, tiles.remove(&c))).collect();
    FillResult {
        layer,
        base,
        tiles,
        ms: t0.elapsed().as_secs_f64() * 1e3,
    }
}
