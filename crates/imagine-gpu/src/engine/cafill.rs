//! Content-aware fill of the selection on the active layer. Split in three
//! so the expensive part can run on a worker thread:
//! [`Engine::fill_job`] reads the area (GPU), [`FillJob::run`] computes
//! (CPU, `Send`), [`Engine::apply_fill`] writes the result as one undo step.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use half::f16;
use imagine_core::color::{linear_to_srgb, srgb_to_linear};
use imagine_core::fill::content_aware_fill;
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
    [linear_to_srgb((p[0] / a).clamp(0.0, 1.0)), linear_to_srgb((p[1] / a).clamp(0.0, 1.0)), linear_to_srgb((p[2] / a).clamp(0.0, 1.0)), a]
}

fn decode(p: Px) -> Px {
    let a = p[3].clamp(0.0, 1.0);
    [srgb_to_linear(p[0]) * a, srgb_to_linear(p[1]) * a, srgb_to_linear(p[2]) * a, a]
}

impl Engine {
    /// Collects what the fill of the current selection needs.
    pub fn fill_job(&mut self, all_layers: bool) -> Result<FillJob, &'static str> {
        let layer = self.check_target(StrokeTarget::Pixels)?;
        let sel = self.doc.selection.clone().ok_or("erst den Bereich auswählen, der gefüllt werden soll")?;
        if sel.default > 0 {
            return Err("die Auswahl reicht bis ins Unendliche – bitte einen Bereich auswählen");
        }
        let selected: Vec<TileCoord> =
            sel.tiles.iter().filter(|(_, t)| !matches!(t, MaskTile::Solid(0))).map(|(c, _)| *c).filter(|c| self.in_document(*c)).collect();
        if selected.is_empty() {
            return Err("die Auswahl ist leer");
        }
        // Selection values of the selected tiles.
        let ts = TILE_SIZE as i64;
        let pix: Vec<(TileCoord, Tile)> = selected
            .iter()
            .filter_map(|c| match &sel.tiles[c] {
                MaskTile::Pixels(t) => Some((*c, t.clone())),
                MaskTile::Solid(_) => None,
            })
            .collect();
        let data = self.read_mask_tiles(&pix.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>());
        let mut mask_tiles: HashMap<TileCoord, Vec<u8>> = pix.iter().map(|(c, _)| *c).zip(data).collect();
        for c in &selected {
            if let MaskTile::Solid(v) = sel.tiles[c] {
                mask_tiles.insert(*c, vec![v; (TILE_SIZE * TILE_SIZE) as usize]);
            }
        }
        // Bounding box of the selected pixels.
        let (mut bx0, mut by0, mut bx1, mut by1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for (c, m) in &mask_tiles {
            for (i, &v) in m.iter().enumerate() {
                if v > 0 {
                    let (x, y) = (c.x as i64 * ts + (i as i64 % ts), c.y as i64 * ts + (i as i64 / ts));
                    (bx0, by0, bx1, by1) = (bx0.min(x), by0.min(y), bx1.max(x), by1.max(y));
                }
            }
        }
        if bx0 > bx1 {
            return Err("die Auswahl ist leer");
        }
        // Learn from a margin around the hole (larger holes look further).
        let size = (bx1 - bx0).max(by1 - by0) + 1;
        let margin = (size * 3 / 4).max(64);
        let (dw, dh) = (self.doc.width as i64, self.doc.height as i64);
        let x0 = (bx0 - margin).max(0);
        let y0 = (by0 - margin).max(0);
        let x1 = (bx1 + 1 + margin).min(dw);
        let y1 = (by1 + 1 + margin).min(dh);
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let (dest, tiles) = self.read_region(Some(layer), x0, y0, w, h);
        let source = if all_layers { self.read_region(None, x0, y0, w, h).0 } else { dest.clone() };
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
        let base = self.doc.find(layer).and_then(|l| l.tiles()).map_or(0, |t| Arc::as_ptr(t) as usize);
        let seed = (x0 as u64) << 32 ^ y0 as u64 ^ (w as u64) << 16;
        Ok(FillJob { layer, base, x0, y0, w, h, dest, source, mask, tiles, seed })
    }

    /// Writes a computed fill (one undo step). False if the layer changed meanwhile.
    pub fn apply_fill(&mut self, r: FillResult, label: &str) -> bool {
        let now = self.doc.find(r.layer).and_then(|l| l.tiles()).map_or(0, |t| Arc::as_ptr(t) as usize);
        if now != r.base {
            tracing::warn!("{label}: verworfen, die Ebene wurde inzwischen verändert");
            return false;
        }
        self.replace_tiles(r.layer, r.tiles, label)
    }

    /// Content-aware fill of the selection, synchronously (agents, tests).
    pub fn content_aware_fill(&mut self, all_layers: bool, label: &str) -> Result<f64, &'static str> {
        let r = self.fill_job(all_layers)?.run()?;
        let ms = r.ms;
        if !self.apply_fill(r, label) {
            return Err("die Ebene wurde inzwischen verändert");
        }
        Ok(ms)
    }
}

impl FillJob {
    /// The fill itself (CPU, any thread).
    pub fn run(self) -> Result<FillResult, &'static str> {
        let t0 = std::time::Instant::now();
        let FillJob { layer, base, x0, y0, w, h, dest, source, mask, mut tiles, seed } = self;
        let hole: Vec<bool> = mask.iter().map(|&m| m > 0.0).collect();
        let allowed = vec![true; w * h];
        let mut img: Vec<Px> = source.iter().map(|&p| encode(p)).collect();
        if !content_aware_fill(w, h, &mut img, &hole, &allowed, seed) {
            return Err("um die Auswahl ist zu wenig Bild, aus dem gefüllt werden kann");
        }
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
                let tile = tiles.entry(c).or_insert_with(|| vec![f16::ZERO; (TILE_SIZE * TILE_SIZE * 4) as usize]);
                let s = ((gy.rem_euclid(ts) * ts + gx.rem_euclid(ts)) * 4) as usize;
                for k in 0..4 {
                    tile[s + k] = f16::from_f32(o[k]);
                }
                touched.insert(c);
            }
        }
        let tiles = touched.into_iter().map(|c| (c, tiles.remove(&c))).collect();
        Ok(FillResult { layer, base, tiles, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }
}
