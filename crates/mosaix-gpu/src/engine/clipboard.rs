//! Copying pixels for the clipboard: the active layer or the flattened
//! image, inside the selection (soft edges become alpha) or whole.

use std::collections::HashMap;

use mosaix_core::{MaskTile, TILE_SIZE, TileCoord};

use half::f16;
use mosaix_core::LayerId;

use super::{Engine, StrokeTarget};
use crate::pool::Tile;

/// A copied rectangle: document position, size and premultiplied linear pixels.
pub struct CopiedPixels {
    pub x: i64,
    pub y: i64,
    pub w: usize,
    pub h: usize,
    pub px: Vec<[f32; 4]>,
}

impl Engine {
    /// The selection as 0…1 values per tile (only tiles that are not empty).
    fn selection_values(&mut self) -> Option<HashMap<TileCoord, Vec<f32>>> {
        let sel = self.doc.selection.clone()?;
        let n = (TILE_SIZE * TILE_SIZE) as usize;
        let mut out = HashMap::new();
        let coords: Vec<TileCoord> = if sel.default > 0 {
            self.all_doc_coords()
        } else {
            sel.tiles.keys().copied().collect()
        };
        let pix: Vec<(TileCoord, Tile)> = coords
            .iter()
            .filter_map(|c| match sel.tiles.get(c) {
                Some(MaskTile::Pixels(t)) => Some((*c, t.clone())),
                _ => None,
            })
            .collect();
        let data = self.read_mask_tiles(&pix.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>());
        for ((c, _), d) in pix.iter().zip(data) {
            out.insert(*c, d.iter().map(|&v| v as f32 / 255.0).collect());
        }
        for c in coords {
            let v = match sel.tiles.get(&c) {
                Some(MaskTile::Solid(v)) => *v,
                None => sel.default,
                _ => continue,
            };
            if v > 0 {
                out.insert(c, vec![v as f32 / 255.0; n]);
            }
        }
        Some(out)
    }

    /// Copies the active layer (`merged`: the flattened image) inside the
    /// selection, cropped to what is there.
    pub fn copy_pixels(&mut self, merged: bool) -> Result<CopiedPixels, &'static str> {
        let layer = if merged {
            None
        } else {
            Some(self.check_target(StrokeTarget::Pixels)?)
        };
        let ts = TILE_SIZE as i64;
        let sel = self.selection_values();
        // Area: the selected tiles, else the layer's tiles (or the canvas).
        let tiles: Vec<TileCoord> = match (&sel, layer) {
            (Some(s), _) => s.keys().copied().collect(),
            (None, Some(id)) => self
                .doc
                .find(id)
                .and_then(|l| l.tiles())
                .map(|t| t.keys().copied().collect())
                .unwrap_or_default(),
            (None, None) => self.all_doc_coords(),
        };
        if tiles.is_empty() {
            return Err(mosaix_core::tr("nothing to copy"));
        }
        let (x0, y0) = (
            tiles.iter().map(|c| c.x).min().unwrap() as i64 * ts,
            tiles.iter().map(|c| c.y).min().unwrap() as i64 * ts,
        );
        let (x1, y1) = (
            (tiles.iter().map(|c| c.x).max().unwrap() as i64 + 1) * ts,
            (tiles.iter().map(|c| c.y).max().unwrap() as i64 + 1) * ts,
        );
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let (mut px, _) = self.read_region(layer, x0, y0, w, h);
        if let Some(s) = &sel {
            for (i, p) in px.iter_mut().enumerate() {
                let (x, y) = (x0 + (i % w) as i64, y0 + (i / w) as i64);
                let c = TileCoord::new(x.div_euclid(ts) as i32, y.div_euclid(ts) as i32);
                let m = s.get(&c).map_or(0.0, |v| {
                    v[(y.rem_euclid(ts) * ts + x.rem_euclid(ts)) as usize]
                });
                *p = p.map(|v| v * m);
            }
        }
        // Crop to the pixels that are there.
        let (mut a, mut b, mut c, mut d) = (usize::MAX, usize::MAX, 0, 0);
        for (i, p) in px.iter().enumerate() {
            if p[3] > 1.0 / 512.0 {
                let (x, y) = (i % w, i / w);
                (a, b, c, d) = (a.min(x), b.min(y), c.max(x + 1), d.max(y + 1));
            }
        }
        if a == usize::MAX {
            return Err(mosaix_core::tr("the selection contains no pixels"));
        }
        let (cw, ch) = (c - a, d - b);
        let mut out = Vec::with_capacity(cw * ch);
        for y in b..d {
            out.extend_from_slice(&px[y * w + a..y * w + c]);
        }
        Ok(CopiedPixels {
            x: x0 + a as i64,
            y: y0 + b as i64,
            w: cw,
            h: ch,
            px: out,
        })
    }
}

impl Engine {
    /// Layer › Layer via Copy / Cut: the active layer's pixels inside the
    /// selection (soft edges become alpha) onto a new layer above it, in
    /// place; `cut` also takes them out of the layer. One undo step.
    pub fn layer_via(&mut self, cut: bool, label: &str) -> Result<LayerId, &'static str> {
        let layer = self.check_target(StrokeTarget::Pixels)?;
        let sel = self.selection_values().ok_or_else(|| mosaix_core::tr("nothing is selected"))?;
        let src = self.doc.find(layer).and_then(|l| l.tiles()).cloned().unwrap_or_default();
        let coords: Vec<TileCoord> = sel.keys().filter(|c| src.contains_key(c)).copied().collect();
        let data = self.read_layer_tiles(&coords.iter().map(|c| src[c].clone()).collect::<Vec<_>>());
        let (mut copied, mut rest) = (Vec::new(), Vec::new());
        for (c, px) in coords.into_iter().zip(data) {
            let m = &sel[&c];
            let mut inside = px.clone();
            let mut outside = px;
            let mut any = false;
            for (i, &v) in m.iter().enumerate() {
                for k in 0..4 {
                    let p = inside[i * 4 + k].to_f32();
                    inside[i * 4 + k] = f16::from_f32(p * v);
                    outside[i * 4 + k] = f16::from_f32(p * (1.0 - v));
                }
                any |= v > 0.0 && inside[i * 4 + 3].to_f32() > 0.0;
            }
            if any {
                let empty = outside.chunks(4).all(|p| p[3].to_f32() <= 0.0);
                copied.push((c, inside));
                rest.push((c, (!empty).then_some(outside)));
            }
        }
        if copied.is_empty() {
            return Err(mosaix_core::tr("the selection contains no pixels"));
        }
        let mut new = HashMap::new();
        for (c, px) in copied {
            new.insert(c, self.write_new_tile(&px).ok_or_else(|| mosaix_core::tr("out of GPU memory"))?);
        }
        let mut left = Vec::new();
        if cut {
            for (c, px) in rest {
                let t = match px {
                    Some(px) => Some(self.write_new_tile(&px).ok_or_else(|| mosaix_core::tr("out of GPU memory"))?),
                    None => None,
                };
                left.push((c, t));
            }
        }
        let coords: Vec<TileCoord> = new.keys().copied().collect();
        let name = self.doc.find(layer).map(|l| l.name.clone()).unwrap_or_default();
        let mut id = 0;
        self.edit(label, |d| {
            if let Some(map) = d.tiles_mut(layer) {
                for (c, t) in left {
                    match t {
                        Some(t) => map.insert(c, t),
                        None => map.remove(&c),
                    };
                }
            }
            id = d.alloc_id();
            let mut l = mosaix_core::Layer::pixels(id, &mosaix_core::tr!("{layer} copy", layer = name));
            l.content = mosaix_core::Content::Pixels(std::sync::Arc::new(new));
            d.active = layer;
            d.insert_above_active(l);
            true
        });
        self.dirty.extend(coords);
        Ok(id)
    }
}
