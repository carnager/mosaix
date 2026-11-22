//! Selection and layer masks. Shape rasterisation, combining and feathering
//! run on the CPU (`mosaix_core::mask`); results are uploaded as sparse R8
//! tiles. Uniform tiles stay `Solid` and cost no GPU memory.

use std::collections::HashMap;
use std::sync::Arc;

use mosaix_core::mask::{CombineOp, CpuMask, MaskData, Shape, combine, feather, rasterize};
use mosaix_core::{LayerId, Mask, MaskTile, TileCoord};

use super::Engine;
use super::readback::PoolKind;
use crate::pool::Tile;

impl Engine {
    /// Downloads a GPU mask to the CPU.
    fn mask_to_cpu(&self, m: &Mask<Tile>) -> CpuMask {
        let pixel_tiles: Vec<(TileCoord, Tile)> = m
            .tiles
            .iter()
            .filter_map(|(c, t)| match t {
                MaskTile::Pixels(p) => Some((*c, p.clone())),
                MaskTile::Solid(_) => None,
            })
            .collect();
        let handles: Vec<Tile> = pixel_tiles.iter().map(|(_, t)| t.clone()).collect();
        let data = self.read_tiles_raw(PoolKind::Mask, &handles);
        let mut tiles: HashMap<TileCoord, MaskData> = m
            .tiles
            .iter()
            .filter_map(|(c, t)| match t {
                MaskTile::Solid(v) => Some((*c, MaskData::Solid(*v))),
                MaskTile::Pixels(_) => None,
            })
            .collect();
        tiles.extend(
            pixel_tiles
                .into_iter()
                .map(|(c, _)| c)
                .zip(data.into_iter().map(MaskData::Pixels)),
        );
        CpuMask {
            tiles,
            default: m.default,
        }
    }

    /// Uploads a CPU mask (compacted first). `None` if GPU memory ran out.
    pub(super) fn mask_to_gpu(&mut self, mut m: CpuMask) -> Option<Mask<Tile>> {
        m.compact();
        let mut tiles = HashMap::with_capacity(m.tiles.len());
        for (c, t) in m.tiles {
            let t = match t {
                MaskData::Solid(v) => MaskTile::Solid(v),
                MaskData::Pixels(p) => MaskTile::Pixels(self.write_new_mask_tile(&p)?),
            };
            tiles.insert(c, t);
        }
        Some(Mask {
            tiles: Arc::new(tiles),
            default: m.default,
        })
    }

    /// Identity of the current selection (changes whenever it is replaced);
    /// used to discard background results computed from an older one.
    pub fn selection_id(&self) -> usize {
        self.doc
            .selection
            .as_ref()
            .map_or(0, |s| Arc::as_ptr(&s.tiles) as usize)
    }

    pub fn has_selection(&self) -> bool {
        self.doc.selection.is_some()
    }

    /// The selection on the CPU (`None` = nothing selected).
    pub fn selection_cpu(&self) -> Option<CpuMask> {
        self.doc.selection.as_ref().map(|m| self.mask_to_cpu(m))
    }

    /// Replaces the selection as one undo step. An empty mask deselects.
    pub fn set_selection(
        &mut self,
        mask: Option<CpuMask>,
        label: &str,
    ) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        let gpu = match mask.filter(|m| !m.is_empty()) {
            Some(m) => Some(
                self.mask_to_gpu(m)
                    .ok_or(mosaix_core::tr("GPU memory full"))?,
            ),
            None => None,
        };
        let had = self.doc.selection.is_some();
        if !had && gpu.is_none() {
            return Ok(());
        }
        self.edit(label, |d| {
            d.selection = gpu;
            true
        });
        Ok(())
    }

    /// Adds/subtracts/intersects a shape with the selection.
    pub fn select_shape(
        &mut self,
        shape: &Shape,
        op: CombineOp,
        label: &str,
    ) -> Result<(), &'static str> {
        let (w, h) = self.size();
        let new = rasterize(shape, w, h);
        let result = match self.selection_cpu() {
            Some(old) => combine(&old, &new, op),
            // Without a selection, subtracting leaves nothing; everything else is the shape.
            None if op == CombineOp::Subtract => CpuMask::filled(0),
            None => new,
        };
        self.set_selection(Some(result), label)
    }

    pub fn select_all(&mut self, label: &str) -> Result<(), &'static str> {
        self.set_selection(Some(CpuMask::filled(255)), label)
    }

    pub fn deselect(&mut self, label: &str) -> Result<(), &'static str> {
        self.set_selection(None, label)
    }

    pub fn invert_selection(&mut self, label: &str) -> Result<(), &'static str> {
        let sel = self.selection_cpu().unwrap_or(CpuMask::filled(0));
        self.set_selection(Some(sel.invert()), label)
    }

    pub fn feather_selection(&mut self, radius: f64, label: &str) -> Result<(), &'static str> {
        let Some(sel) = self.selection_cpu() else {
            return Err(mosaix_core::tr("no selection"));
        };
        let (w, h) = self.size();
        self.set_selection(Some(feather(&sel, radius, w, h)), label)
    }

    /// Select › Modify: expand, contract, border or smooth.
    pub fn modify_selection(&mut self, op: mosaix_core::mask::Modify, label: &str) -> Result<(), &'static str> {
        let Some(sel) = self.selection_cpu() else {
            return Err(mosaix_core::tr("no selection"));
        };
        let (w, h) = self.size();
        self.set_selection(Some(mosaix_core::mask::modify(&sel, op, w, h)), label)
    }

    // ---- saved selections (channels) ---------------------------------------------

    /// Saves the selection as a new channel (or replaces the one with this
    /// name). Returns its index.
    pub fn save_selection(&mut self, name: &str, label: &str) -> Result<usize, &'static str> {
        let Some(sel) = self.doc.selection.clone() else {
            return Err(mosaix_core::tr("no selection"));
        };
        let name = name.trim();
        if name.is_empty() {
            return Err(mosaix_core::tr("the name is empty"));
        }
        let index = self.doc.channels.iter().position(|c| c.name == name).unwrap_or(self.doc.channels.len());
        let name = name.to_string();
        self.edit(label, |d| {
            let c = mosaix_core::Channel { name, mask: sel };
            match d.channels.get_mut(index) {
                Some(old) => *old = c,
                None => d.channels.push(c),
            }
            true
        });
        Ok(index)
    }

    /// Loads a channel as the selection, combined with the current one.
    pub fn load_channel(&mut self, index: usize, op: CombineOp, invert: bool, label: &str) -> Result<(), &'static str> {
        let m = self.doc.channels.get(index).ok_or(mosaix_core::tr("no such channel"))?.mask.clone();
        let mut new = self.mask_to_cpu(&m);
        if invert {
            new = new.invert();
        }
        let result = match self.selection_cpu() {
            Some(old) if op != CombineOp::Replace => combine(&old, &new, op),
            None if op == CombineOp::Subtract => CpuMask::filled(0),
            None if op == CombineOp::Intersect => CpuMask::filled(0),
            _ => new,
        };
        self.set_selection(Some(result), label)
    }

    /// Selects a layer's pixels by their opacity (Ctrl+click on the
    /// thumbnail), combined with the selection by `op`. Text, shapes and RAW
    /// count as they are drawn.
    pub fn select_layer_pixels(&mut self, layer: LayerId, op: CombineOp, label: &str) -> Result<(), &'static str> {
        let l = self.doc.find(layer).ok_or(mosaix_core::tr("no such layer"))?;
        let tiles = if l.is_rendered() {
            self.refresh_smart();
            self.vectors.get(&layer).map(|(_, t)| t.clone())
        } else {
            l.tiles().cloned()
        }
        .ok_or(mosaix_core::tr("this layer has no pixels"))?;
        let coords: Vec<TileCoord> = tiles.keys().copied().filter(|c| self.in_document(*c)).collect();
        let data = self.read_layer_tiles(&coords.iter().map(|c| tiles[c].clone()).collect::<Vec<_>>());
        let mut new = CpuMask::filled(0);
        for (c, px) in coords.into_iter().zip(data) {
            let a: Vec<u8> = px.chunks(4).map(|p| (p[3].to_f32().clamp(0.0, 1.0) * 255.0).round() as u8).collect();
            if a.iter().any(|&v| v > 0) {
                new.tiles.insert(c, MaskData::Pixels(a).normalized());
            }
        }
        let result = match self.selection_cpu() {
            Some(old) if op != CombineOp::Replace => combine(&old, &new, op),
            None if op == CombineOp::Subtract || op == CombineOp::Intersect => CpuMask::filled(0),
            _ => new,
        };
        self.set_selection(Some(result), label)
    }

    /// Select › Select and Mask: the edge follows the image within `radius`
    /// px (hair, fur), then `contrast` (0…1), `shift` (px, + grows) and
    /// `feather` (px). Result: the selection, or with `to_mask` the active
    /// layer's mask (the selection then goes).
    pub fn select_and_mask(
        &mut self,
        radius: f64,
        contrast: f32,
        shift: f64,
        feather_px: f64,
        to_mask: bool,
        label: &str,
    ) -> Result<(), &'static str> {
        use mosaix_core::color::linear_to_srgb;
        use mosaix_core::mask::{Modify, modify, refine_edge};
        let sel = self.selection_cpu().ok_or(mosaix_core::tr("nothing is selected"))?;
        let (w, h) = self.size();
        let mut m = if radius >= 1.0 {
            refine_edge(
                &sel,
                radius,
                contrast,
                |x0, y0, rw, rh| {
                    let (px, _) = self.read_region(None, x0, y0, rw, rh);
                    px.iter()
                        .map(|p| {
                            if p[3] <= 1e-6 {
                                return 0.0;
                            }
                            linear_to_srgb((0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) / p[3])
                        })
                        .collect()
                },
                w,
                h,
            )
        } else {
            sel
        };
        if shift >= 0.5 {
            m = modify(&m, Modify::Expand(shift), w, h);
        } else if shift <= -0.5 {
            m = modify(&m, Modify::Contract(-shift), w, h);
        }
        if feather_px >= 0.5 {
            m = feather(&m, feather_px, w, h);
        }
        if to_mask {
            let layer = self.doc.active;
            let n = self.history.undo_len();
            let before = self.doc.clone();
            self.set_layer_mask(layer, Some(m), label)?;
            self.edit(label, |d| {
                d.selection = None;
                true
            });
            self.history.squash_since(n, before, label);
            Ok(())
        } else {
            self.set_selection(Some(m), label)
        }
    }

    /// Pixel bounds of the selected area (x0, y0, x1, y1; exclusive).
    pub fn selection_bounds(&self) -> Option<(i64, i64, i64, i64)> {
        let sel = self.selection_cpu()?;
        let ts = mosaix_core::TILE_SIZE as i64;
        if sel.default > 0 {
            return Some((0, 0, self.doc.width as i64, self.doc.height as i64));
        }
        let mut b = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for (c, m) in &sel.tiles {
            let (ox, oy) = (c.x as i64 * ts, c.y as i64 * ts);
            match m {
                MaskData::Solid(0) => {}
                MaskData::Solid(_) => b = (b.0.min(ox), b.1.min(oy), b.2.max(ox + ts), b.3.max(oy + ts)),
                MaskData::Pixels(p) => {
                    for (i, &v) in p.iter().enumerate() {
                        if v > 0 {
                            let (x, y) = (ox + i as i64 % ts, oy + i as i64 / ts);
                            b = (b.0.min(x), b.1.min(y), b.2.max(x + 1), b.3.max(y + 1));
                        }
                    }
                }
            }
        }
        (b.0 < b.2).then_some(b)
    }

    /// Select › Transform Selection: moves, scales, rotates or distorts the
    /// selection outline only (the pixels stay). One undo step.
    pub fn transform_selection(&mut self, fwd: impl Into<super::Warp>, label: &str) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        let sel = self.doc.selection.clone().ok_or(mosaix_core::tr("nothing is selected"))?;
        let new = self.warp_mask(&sel, fwd.into(), super::Interp::Bilinear)?;
        self.edit(label, |d| {
            d.selection = Some(new);
            true
        });
        Ok(())
    }

    /// Quick Mask: the selection becomes a red, half-opaque layer on top
    /// (red = not selected) that every painting tool can edit; the
    /// selection itself is lifted so painting is not clipped. Returns the
    /// layer; [`Self::leave_quick_mask`] turns it back into a selection.
    pub fn enter_quick_mask(&mut self, label: &str) -> Result<LayerId, &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        let sel = self.selection_cpu();
        let ts = mosaix_core::TILE_SIZE as usize;
        let mut tiles = HashMap::new();
        for c in self.all_doc_coords() {
            let m = sel.as_ref().map(|s| s.tiles.get(&c).cloned().unwrap_or(MaskData::Solid(s.default)));
            if matches!(m, Some(MaskData::Solid(255))) {
                continue;
            }
            let mut px = vec![half::f16::ZERO; ts * ts * 4];
            for i in 0..ts * ts {
                let v = m.as_ref().map_or(0, |m| m.get(i % ts, i / ts));
                let a = 1.0 - v as f32 / 255.0;
                px[i * 4] = half::f16::from_f32(a);
                px[i * 4 + 3] = half::f16::from_f32(a);
            }
            if sel.is_some() {
                tiles.insert(c, self.write_new_tile(&px).ok_or(mosaix_core::tr("GPU memory full"))?);
            }
        }
        let coords: Vec<TileCoord> = tiles.keys().copied().collect();
        let mut id = 0;
        self.edit(label, |d| {
            id = d.alloc_id();
            let mut l = mosaix_core::Layer::pixels(id, mosaix_core::tr("Quick Mask"));
            l.content = mosaix_core::Content::Pixels(Arc::new(tiles));
            l.opacity = 0.5;
            d.layers.push(l);
            d.active = id;
            d.picked.clear();
            d.selection = None;
            true
        });
        self.dirty.extend(coords);
        Ok(id)
    }

    /// Ends Quick Mask: the layer's opacity (inverted) becomes the
    /// selection, the layer goes away and `back` is active again.
    pub fn leave_quick_mask(&mut self, layer: LayerId, back: LayerId, label: &str) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        let Some(tiles) = self.doc.find(layer).and_then(|l| l.tiles()).cloned() else {
            return Err(mosaix_core::tr("the Quick Mask layer is gone"));
        };
        let coords: Vec<TileCoord> = tiles.keys().copied().filter(|c| self.in_document(*c)).collect();
        let data = self.read_layer_tiles(&coords.iter().map(|c| tiles[c].clone()).collect::<Vec<_>>());
        // Where the layer has no tile nothing is masked: selected.
        let mut sel = CpuMask::filled(255);
        for (c, px) in coords.into_iter().zip(data) {
            let v: Vec<u8> = px.chunks(4).map(|p| ((1.0 - p[3].to_f32().clamp(0.0, 1.0)) * 255.0).round() as u8).collect();
            sel.tiles.insert(c, MaskData::Pixels(v).normalized());
        }
        sel.compact();
        let everything = sel.tiles.is_empty() && sel.default == 255;
        let gpu = if everything { None } else { Some(self.mask_to_gpu(sel).ok_or(mosaix_core::tr("GPU memory full"))?) };
        self.edit(label, |d| {
            d.remove_layers(&[layer]);
            if d.find(back).is_some() {
                d.active = back;
            }
            d.selection = gpu;
            true
        });
        Ok(())
    }

    pub fn delete_channel(&mut self, index: usize, label: &str) -> Result<(), &'static str> {
        if index >= self.doc.channels.len() {
            return Err(mosaix_core::tr("no such channel"));
        }
        self.edit(label, |d| {
            d.channels.remove(index);
            true
        });
        Ok(())
    }

    pub fn rename_channel(&mut self, index: usize, name: &str, label: &str) -> Result<(), &'static str> {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(mosaix_core::tr("the name is empty"));
        }
        if index >= self.doc.channels.len() {
            return Err(mosaix_core::tr("no such channel"));
        }
        self.edit(label, |d| {
            let changed = d.channels[index].name != name;
            d.channels[index].name = name;
            changed
        });
        Ok(())
    }

    // ---- guides --------------------------------------------------------------------

    /// Replaces the guides as one undo step (nothing happens if they are equal).
    pub fn set_guides(&mut self, guides: Vec<mosaix_core::Guide>, label: &str) {
        if self.doc.guides == guides {
            return;
        }
        self.edit(label, |d| {
            d.guides = guides;
            true
        });
    }

    // ---- layer masks --------------------------------------------------------------

    pub fn layer_mask_cpu(&self, layer: LayerId) -> Option<CpuMask> {
        let m = self.doc.find(layer)?.mask.as_ref()?;
        Some(self.mask_to_cpu(m))
    }

    /// Sets (or removes) a layer's mask as one undo step.
    pub fn set_layer_mask(
        &mut self,
        layer: LayerId,
        mask: Option<CpuMask>,
        label: &str,
    ) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        if self.doc.find(layer).is_none() {
            return Err(mosaix_core::tr("layer does not exist"));
        }
        let gpu = match mask {
            Some(m) => Some(
                self.mask_to_gpu(m)
                    .ok_or(mosaix_core::tr("GPU memory full"))?,
            ),
            None => None,
        };
        self.edit(label, |d| {
            let l = d.find_mut(layer).unwrap();
            l.mask = gpu;
            l.mask_enabled = true;
            true
        });
        Ok(())
    }

    /// Adds an adjustment layer above the active one; with a selection, the
    /// selection becomes its mask (shared tiles, no copy).
    pub fn add_adjustment(
        &mut self,
        adj: mosaix_core::Adjustment,
        label: &str,
    ) -> Option<LayerId> {
        let mut id = None;
        self.edit(label, |d| {
            let new = d.add_adjustment(adj);
            // Like Photoshop: always a mask to paint into (free while uniform).
            let mask = d.selection.clone().unwrap_or_else(|| Mask {
                tiles: Arc::new(HashMap::new()),
                default: 255,
            });
            let l = d.find_mut(new).unwrap();
            l.mask = Some(mask);
            l.mask_enabled = true;
            id = Some(new);
            true
        });
        id
    }

    /// Adds a mask to the active layer: from the selection, or revealing everything.
    pub fn add_layer_mask(&mut self, label: &str) -> Result<(), &'static str> {
        let layer = self.doc.active;
        if self.doc.find(layer).is_some_and(|l| l.mask.is_some()) {
            return Err(mosaix_core::tr("the layer already has a mask"));
        }
        let mask = self.selection_cpu().unwrap_or(CpuMask::filled(255));
        self.set_layer_mask(layer, Some(mask), label)
    }

    /// Loads a mask from a file without creating an undo step.
    pub fn load_mask(
        &mut self,
        slot: mosaix_core::MaskSlot,
        mask: CpuMask,
        enabled: bool,
    ) -> Option<Vec<(TileCoord, Tile)>> {
        let gpu = self.mask_to_gpu(mask)?;
        let uploaded: Vec<(TileCoord, Tile)> = gpu
            .tiles
            .iter()
            .filter_map(|(c, t)| match t {
                MaskTile::Pixels(p) => Some((*c, p.clone())),
                MaskTile::Solid(_) => None,
            })
            .collect();
        match slot {
            mosaix_core::MaskSlot::Layer(id) => {
                let l = self.doc.find_mut(id)?;
                l.mask = Some(gpu);
                l.mask_enabled = enabled;
                self.recheck_all = true;
            }
            mosaix_core::MaskSlot::Selection => self.doc.selection = Some(gpu),
            mosaix_core::MaskSlot::Channel(i) => self.doc.channels.get_mut(i)?.mask = gpu,
        }
        Some(uploaded)
    }
}
