//! Healing brush: while painting it behaves like the clone stamp; when the
//! stroke ends, the copied pixels are blended into their surroundings with
//! the membrane solver (`imagine_core::heal`), so texture comes from the
//! source while colour and brightness come from around the stroke.

use std::collections::{BTreeSet, HashMap};

use half::f16;
use imagine_core::heal::membrane;
use imagine_core::{LayerId, TILE_SIZE, TileCoord};

use super::{CloneSource, Engine};

type Px = [f32; 4];

impl Engine {
    /// Reads a rectangle (document px) of the active layer or the composite
    /// as straight f32 RGBA (premultiplied, as stored), plus the full tiles it
    /// came from (layer source only).
    fn read_region(&mut self, layer: Option<LayerId>, x0: i64, y0: i64, w: usize, h: usize) -> (Vec<Px>, HashMap<TileCoord, Vec<f16>>) {
        let ts = TILE_SIZE as i64;
        let coords: Vec<TileCoord> = (y0.div_euclid(ts)..=(y0 + h as i64 - 1).div_euclid(ts))
            .flat_map(|ty| (x0.div_euclid(ts)..=(x0 + w as i64 - 1).div_euclid(ts)).map(move |tx| TileCoord::new(tx as i32, ty as i32)))
            .collect();
        let tiles: HashMap<TileCoord, Vec<f16>> = match layer {
            Some(id) => {
                let map = self.doc.find(id).and_then(|l| l.tiles()).cloned().unwrap_or_default();
                let present: Vec<TileCoord> = coords.iter().copied().filter(|c| map.contains_key(c)).collect();
                let handles: Vec<_> = present.iter().map(|c| map[c].clone()).collect();
                present.into_iter().zip(self.read_layer_tiles(&handles)).collect()
            }
            None => {
                self.flush();
                let inside: Vec<TileCoord> = coords.iter().copied().filter(|c| self.in_document(*c)).collect();
                inside.iter().copied().zip(self.read_composite(&inside)).collect()
            }
        };
        let mut out = vec![[0.0f32; 4]; w * h];
        for (c, px) in &tiles {
            let (ox, oy) = (c.x as i64 * ts, c.y as i64 * ts);
            for ly in 0..ts {
                let y = oy + ly - y0;
                if y < 0 || y >= h as i64 {
                    continue;
                }
                for lx in 0..ts {
                    let x = ox + lx - x0;
                    if x < 0 || x >= w as i64 {
                        continue;
                    }
                    let s = ((ly * ts + lx) * 4) as usize;
                    out[y as usize * w + x as usize] = [px[s].to_f32(), px[s + 1].to_f32(), px[s + 2].to_f32(), px[s + 3].to_f32()];
                }
            }
        }
        (out, tiles)
    }

    /// Heals the area covered by `dabs` (centre x, y, radius) on `layer`.
    pub(super) fn heal_stroke(&mut self, layer: LayerId, src: CloneSource, dabs: &[[f32; 3]], opacity: f32, label: &str) {
        if dabs.is_empty() {
            return;
        }
        let t0 = std::time::Instant::now();
        let (dw, dh) = (self.doc.width as i64, self.doc.height as i64);
        let (mut a, mut b) = ((f32::MAX, f32::MAX), (f32::MIN, f32::MIN));
        for d in dabs {
            a = (a.0.min(d[0] - d[2]), a.1.min(d[1] - d[2]));
            b = (b.0.max(d[0] + d[2]), b.1.max(d[1] + d[2]));
        }
        let x0 = (a.0.floor() as i64 - 2).max(0);
        let y0 = (a.1.floor() as i64 - 2).max(0);
        let x1 = (b.0.ceil() as i64 + 2).min(dw);
        let y1 = (b.1.ceil() as i64 + 2).min(dh);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        // The region: union of the dab discs (pixel centres), off the border ring.
        let mut inside = vec![false; w * h];
        for d in dabs {
            let r = d[2];
            let (cx, cy) = (d[0] - x0 as f32, d[1] - y0 as f32);
            let ya = ((cy - r).floor().max(1.0)) as usize;
            let yb = ((cy + r).ceil() as usize).min(h.saturating_sub(1));
            let xa = ((cx - r).floor().max(1.0)) as usize;
            let xb = ((cx + r).ceil() as usize).min(w.saturating_sub(1));
            for y in ya..yb {
                for x in xa..xb {
                    if (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) < r {
                        inside[y * w + x] = true;
                    }
                }
            }
        }
        // Pixels outside the canvas cannot be healed from; keep the canvas edge as border.
        let (dest, mut tiles) = self.read_region(Some(layer), x0, y0, w, h);
        let [ox, oy] = src.offset;
        let (source, _) = self.read_region(if src.all_layers { None } else { Some(layer) }, x0 + ox as i64, y0 + oy as i64, w, h);
        let mut v: Vec<Px> = dest.iter().zip(&source).map(|(d, s)| [0, 1, 2, 3].map(|c| d[c] - s[c])).collect();
        membrane(w, h, &inside, &mut v);
        // Write the result into the layer's tiles.
        let ts = TILE_SIZE as i64;
        let mut touched = BTreeSet::new();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if !inside[i] {
                    continue;
                }
                let healed = [0, 1, 2, 3].map(|c| source[i][c] + v[i][c]);
                let o = [0, 1, 2, 3].map(|c| dest[i][c] + (healed[c] - dest[i][c]) * opacity);
                let alpha = o[3].clamp(0.0, 1.0);
                let px = [o[0].max(0.0), o[1].max(0.0), o[2].max(0.0), alpha];
                let (gx, gy) = (x0 + x as i64, y0 + y as i64);
                let c = TileCoord::new(gx.div_euclid(ts) as i32, gy.div_euclid(ts) as i32);
                let tile = tiles.entry(c).or_insert_with(|| vec![f16::ZERO; (TILE_SIZE * TILE_SIZE * 4) as usize]);
                let s = ((gy.rem_euclid(ts) * ts + gx.rem_euclid(ts)) * 4) as usize;
                for k in 0..4 {
                    tile[s + k] = f16::from_f32(px[k]);
                }
                touched.insert(c);
            }
        }
        let changes: Vec<(TileCoord, Option<Vec<f16>>)> = touched.into_iter().map(|c| (c, tiles.remove(&c))).collect();
        self.replace_tiles(layer, changes, label);
        tracing::debug!("Reparieren {w}×{h}: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
}
