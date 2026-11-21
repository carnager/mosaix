//! Layer styles in the render document: the layer's composited pixels (after
//! smart filters / vector rasterisation) are read back, the effects are
//! computed on the CPU (mosaix_core::style) and uploaded again. Effects
//! outside the shape become extra layers right under the styled layer, with
//! their own blend mode; the cached result is reused until the layer's
//! pixels or its effects change.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use half::f16;
use mosaix_core::style::{Effect, Region, apply_at, reach};
use mosaix_core::doc::{Mask, MaskTile};
use mosaix_core::{BlendMode, LayerId, TILE_SIZE, TileCoord};

use super::Engine;
use crate::pool::Tile;

type TileSet = HashMap<TileCoord, Tile>;

/// Larger styled areas are skipped (CPU time).
const MAX_PIXELS: usize = 24 << 20;

/// What a layer style is computed from.
pub(super) struct StyleSource {
    pub id: LayerId,
    /// The layer's pixels (after smart filters / rasterisation).
    pub src: Arc<TileSet>,
    pub effects: Vec<Effect>,
    pub fill: f32,
    /// The layer mask (switched on): like Photoshop, the effects follow the
    /// masked content, and the mask is applied here instead of afterwards.
    pub mask: Option<Mask<Tile>>,
}

pub(super) struct StyleEntry {
    key: u64,
    /// The layer mask is already applied to the result.
    pub masked: bool,
    _src: Arc<TileSet>,
    pub below: Vec<(BlendMode, Arc<TileSet>)>,
    pub content: Arc<TileSet>,
}

impl Engine {
    /// Brings the styles of all layers up to date; `sources` maps layer →
    /// the pixels the style is computed from. True if anything changed.
    pub(super) fn refresh_styles(
        &mut self,
        sources: &[StyleSource],
    ) -> bool {
        let before = self.styles.len();
        self.styles
            .retain(|id, _| sources.iter().any(|s| s.id == *id));
        let mut changed = self.styles.len() != before;
        for StyleSource { id, src, effects, fill, mask } in sources {
            let mut h = DefaultHasher::new();
            (Arc::as_ptr(src) as usize).hash(&mut h);
            format!("{effects:?}").hash(&mut h);
            fill.to_bits().hash(&mut h);
            if let Some(m) = mask {
                (Arc::as_ptr(&m.tiles) as usize).hash(&mut h);
                m.default.hash(&mut h);
            }
            let key = h.finish();
            if self.styles.get(id).is_some_and(|e| e.key == key) {
                continue;
            }
            let t0 = std::time::Instant::now();
            match self.compute_style(src, effects, *fill, mask.as_ref()) {
                Some((below, content)) => {
                    self.styles.insert(
                        *id,
                        StyleEntry {
                            key,
                            masked: mask.is_some(),
                            _src: src.clone(),
                            below,
                            content,
                        },
                    );
                }
                None => {
                    self.styles.remove(id);
                }
            }
            tracing::debug!(
                "{}",
                mosaix_core::tr!(
                    "Layer style {id}: {ms} ms",
                    id = id,
                    ms = format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3)
                )
            );
            changed = true;
        }
        changed
    }

    #[allow(clippy::type_complexity)]
    fn compute_style(
        &mut self,
        src: &Arc<TileSet>,
        effects: &[Effect],
        fill: f32,
        mask: Option<&Mask<Tile>>,
    ) -> Option<(Vec<(BlendMode, Arc<TileSet>)>, Arc<TileSet>)> {
        if src.is_empty() {
            return None;
        }
        let ts = TILE_SIZE as i64;
        let pad = (reach(effects).ceil() as i64).max(1) + 1;
        // The source pixels: from the CPU copy of a vector layer, else read from the GPU.
        let coords: Vec<TileCoord> = src.keys().copied().collect();
        let data: Vec<Vec<f16>> = match self.vector_cpu.get(&(Arc::as_ptr(src) as usize)) {
            Some(cpu) => coords.iter().map(|c| cpu[c].clone()).collect(),
            None => {
                let handles: Vec<Tile> = coords.iter().map(|c| src[c].clone()).collect();
                self.read_layer_tiles(&handles)
            }
        };
        // Work only on the area with pixels (plus the effects' reach), not the whole tiles.
        let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for (c, px) in coords.iter().zip(&data) {
            let (ox, oy) = (c.x as i64 * ts, c.y as i64 * ts);
            for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
                if p[3].to_f32() > 0.0 {
                    let (x, y) = (ox + i as i64 % ts, oy + i as i64 / ts);
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                }
            }
        }
        if x0 >= x1 {
            return None;
        }
        let (x0, y0, x1, y1) = (x0 - pad, y0 - pad, x1 + pad, y1 + pad);
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        if w * h > MAX_PIXELS {
            tracing::warn!(
                "{}",
                mosaix_core::tr!(
                    "Layer style skipped: layer too large ({mp} MP)",
                    mp = w * h / 1_000_000
                )
            );
            return None;
        }
        let mut region = Region::new(w, h);
        for (c, px) in coords.iter().zip(&data) {
            let (ox, oy) = (c.x as i64 * ts - x0, c.y as i64 * ts - y0);
            for ly in 0..ts {
                let y = oy + ly;
                if y < 0 || y >= h as i64 {
                    continue;
                }
                let row = y as usize * w;
                for lx in 0..ts {
                    let x = ox + lx;
                    if x < 0 || x >= w as i64 {
                        continue;
                    }
                    let s = ((ly * ts + lx) * 4) as usize;
                    region.px[row + x as usize] = [
                        px[s].to_f32(),
                        px[s + 1].to_f32(),
                        px[s + 2].to_f32(),
                        px[s + 3].to_f32(),
                    ];
                }
            }
        }
        if let Some(m) = mask {
            self.apply_mask(&mut region, (x0, y0), m);
        }
        let styled = apply_at(effects, &region, fill, (x0, y0));
        let mut upload = |r: &Region| -> Option<Arc<TileSet>> {
            let mut map = TileSet::new();
            for ty in y0.div_euclid(ts)..=(y1 - 1).div_euclid(ts) {
                for tx in x0.div_euclid(ts)..=(x1 - 1).div_euclid(ts) {
                    let mut px = vec![f16::ZERO; (TILE_SIZE * TILE_SIZE * 4) as usize];
                    let mut any = false;
                    for ly in 0..ts {
                        let y = ty * ts + ly - y0;
                        if y < 0 || y >= h as i64 {
                            continue;
                        }
                        for lx in 0..ts {
                            let x = tx * ts + lx - x0;
                            if x < 0 || x >= w as i64 {
                                continue;
                            }
                            let p = r.px[y as usize * w + x as usize];
                            if p[3] <= 1.0 / 1024.0 {
                                continue;
                            }
                            any = true;
                            let o = ((ly * ts + lx) * 4) as usize;
                            for k in 0..4 {
                                px[o + k] = f16::from_f32(p[k]);
                            }
                        }
                    }
                    if any {
                        map.insert(
                            TileCoord::new(tx as i32, ty as i32),
                            self.write_new_tile(&px)?,
                        );
                    }
                }
            }
            Some(Arc::new(map))
        };
        let content = upload(&styled.content)?;
        let mut below = Vec::new();
        for (mode, r) in &styled.below {
            below.push((*mode, upload(r)?));
        }
        Some((below, content))
    }
}

impl Engine {
    /// Multiplies the region (origin `o` in document px) by the layer mask.
    fn apply_mask(&self, region: &mut Region, o: (i64, i64), m: &Mask<Tile>) {
        let ts = TILE_SIZE as i64;
        let (w, h) = (region.w as i64, region.h as i64);
        let coords: Vec<TileCoord> = (o.1.div_euclid(ts)..=(o.1 + h - 1).div_euclid(ts))
            .flat_map(|ty| (o.0.div_euclid(ts)..=(o.0 + w - 1).div_euclid(ts)).map(move |tx| TileCoord::new(tx as i32, ty as i32)))
            .collect();
        // Pixel tiles are read back in one go.
        let wanted: Vec<(TileCoord, Tile)> = coords
            .iter()
            .filter_map(|c| match m.tiles.get(c) {
                Some(MaskTile::Pixels(t)) => Some((*c, t.clone())),
                _ => None,
            })
            .collect();
        let data = self.read_mask_tiles(&wanted.iter().map(|w| w.1.clone()).collect::<Vec<_>>());
        let pixels: HashMap<TileCoord, &Vec<u8>> = wanted.iter().map(|w| w.0).zip(&data).collect();
        for c in coords {
            let (tx, ty) = (c.x as i64 * ts - o.0, c.y as i64 * ts - o.1);
            let solid = match m.tiles.get(&c) {
                Some(MaskTile::Solid(v)) => Some(*v),
                Some(MaskTile::Pixels(_)) => None,
                None => Some(m.default),
            };
            if solid == Some(255) {
                continue;
            }
            for ly in 0..ts {
                let y = ty + ly;
                if y < 0 || y >= h {
                    continue;
                }
                for lx in 0..ts {
                    let x = tx + lx;
                    if x < 0 || x >= w {
                        continue;
                    }
                    let v = match solid {
                        Some(v) => v,
                        None => pixels[&c][(ly * ts + lx) as usize],
                    } as f32
                        / 255.0;
                    let p = &mut region.px[(y * w + x) as usize];
                    *p = p.map(|c| c * v);
                }
            }
        }
    }
}
