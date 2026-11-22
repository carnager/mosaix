//! Merge Down, Merge Visible and Flatten Image: layers composited on the GPU
//! into one pixel layer, as they look (text, shapes, smart filters, styles,
//! masks, blend modes, adjustments baked in). Content outside the canvas is
//! kept.

use std::collections::HashMap;
use std::sync::Arc;

use mosaix_core::{Content, Layer, LayerId, MergeKind, TILE_SIZE, TileCoord};

use super::frame::FlattenBatch;
use super::{Engine, mask_slot};
use crate::pool::{Tile, texel_copy};

/// Tiles composited per GPU submission.
const BATCH: usize = 256;

impl Engine {
    /// Merges layers as `kind` says; returns the new layer.
    pub fn merge(&mut self, kind: MergeKind, label: &str) -> Result<LayerId, &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("finish the stroke first"));
        }
        let ids = self.doc.merge_set(kind)?;
        // The render document has text, shapes, filters and styles as pixels.
        self.flush();
        let shown = self.render.clone().unwrap_or_else(|| self.doc.clone());
        let (list, _) = shown.siblings(ids[0]).ok_or_else(|| mosaix_core::tr("no such layer"))?;
        let ours = |id: LayerId| ids.contains(&id) || ids.iter().any(|&i| (0..16).any(|k| super::filter::effect_layer_id(i, k) == id));
        let mut alone = shown.clone();
        alone.layers = list.iter().filter(|l| ours(l.id)).cloned().collect();
        alone.selection = None;

        let mut coords: Vec<TileCoord> = alone.all_tile_coords().into_iter().collect();
        if alone.layers.iter().any(|l| l.adjustment().is_some()) || coords.is_empty() {
            // Adjustments alone have no tiles; they act on the canvas.
            coords.extend(self.all_doc_coords());
        }
        coords.sort();
        coords.dedup();

        let tiles = self.composite_alone(&alone, &coords)?;

        let name = match kind {
            MergeKind::Down => shown.find(ids[0]).map_or_else(|| mosaix_core::tr("Merged").to_string(), |l| l.name.clone()),
            MergeKind::Visible | MergeKind::Selected => mosaix_core::tr("Merged").to_string(),
            MergeKind::Flatten => mosaix_core::tr("Background").to_string(),
        };
        let coords: Vec<TileCoord> = tiles.keys().copied().collect();
        let mut new_id = 0;
        let ok = self.edit(label, |d| {
            let id = d.alloc_id();
            let mut l = Layer::pixels(id, &name);
            l.content = Content::Pixels(Arc::new(tiles));
            new_id = id;
            if kind == MergeKind::Flatten {
                d.layers.clear();
                d.layers.push(l);
                d.active = id;
                true
            } else {
                d.replace_merged(&ids, l)
            }
        });
        if !ok {
            return Err(mosaix_core::tr("could not merge"));
        }
        self.dirty.extend(coords);
        Ok(new_id)
    }

    /// Composites `alone` (a document holding just the layers wanted) into
    /// new layer tiles at `coords`, on the GPU.
    pub(super) fn composite_alone(&mut self, alone: &mosaix_core::Document<Tile>, coords: &[TileCoord]) -> Result<HashMap<TileCoord, Tile>, &'static str> {
        let mut tiles: HashMap<TileCoord, Tile> = HashMap::new();
        for chunk in coords.chunks(BATCH) {
            let mut b = FlattenBatch::default();
            let mut done: Vec<(TileCoord, Tile)> = Vec::new();
            for &c in chunk {
                let program = alone.program(c, None);
                if program.is_empty() {
                    continue;
                }
                let tmp = self.comp_pool.alloc().ok_or_else(|| mosaix_core::tr("out of GPU memory"))?;
                b.push(&program, None, mask_slot(None), tmp.slot());
                done.push((c, tmp));
            }
            if done.is_empty() {
                continue;
            }
            let mut enc = self.device.create_command_encoder(&Default::default());
            self.dispatch_flatten(&mut enc, b);
            let mut keep = Vec::with_capacity(done.len());
            for (c, tmp) in &done {
                let dst = self.layer_pool.alloc().ok_or_else(|| mosaix_core::tr("out of GPU memory"))?;
                let (s, d) = (tmp.slot(), dst.slot());
                enc.copy_texture_to_texture(
                    texel_copy(&self.comp_pool.chunk(s).texture, 0, s.layer as u32),
                    texel_copy(&self.layer_pool.chunk(d).texture, 0, d.layer as u32),
                    wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
                );
                keep.push((*c, dst));
            }
            self.params.upload(&self.queue);
            self.queue.submit([enc.finish()]);
            tiles.extend(keep);
        }

        Ok(tiles)
    }

    /// Shows the active pixel layer with `a` applied (inside the selection)
    /// without changing anything; `None` ends the preview.
    pub fn set_adjust_preview(&mut self, a: Option<&mosaix_core::Adjustment>) -> Result<(), &'static str> {
        match a {
            None => self.set_filter_preview(None),
            Some(a) => {
                let kind = super::transform::PreviewKind::Adjust(a.clone());
                if self.preview.as_ref().is_some_and(|p| p.kind == kind) {
                    return Ok(());
                }
                self.compute_preview(kind)
            }
        }
    }

    /// Image › Adjustments: applies `a` to the active pixel layer's own
    /// pixels, inside the selection (soft edges blend). One undo step.
    pub fn apply_adjustment(&mut self, a: &mosaix_core::Adjustment, label: &str) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("finish the stroke first"));
        }
        let layer = self.check_target(super::StrokeTarget::Pixels)?;
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        let src = self.doc.find(layer).ok_or_else(|| mosaix_core::tr("no such layer"))?;
        let mut plain = src.clone();
        plain.opacity = 1.0;
        plain.fill_opacity = 1.0;
        plain.blend = mosaix_core::BlendMode::Normal;
        plain.mask = None;
        plain.effects.clear();
        plain.filters.clear();
        plain.clip = false;
        plain.visible = true;
        let mut alone = self.doc.clone();
        alone.selection = None;
        let id = alone.alloc_id();
        let mut adj = Layer::pixels(id, "");
        adj.content = Content::Adjustment(a.clone());
        adj.mask = self.doc.selection.clone();
        adj.mask_enabled = true;
        alone.layers = vec![plain, adj];
        let own = src.tiles().cloned().unwrap_or_default();
        // Tiles the selection leaves out keep their pixels exactly.
        let sel = self.doc.selection.as_ref();
        let coords: Vec<TileCoord> = own
            .keys()
            .copied()
            .filter(|c| !matches!(sel.map(|s| s.at(*c)), Some(mosaix_core::MaskRef::Solid(0))))
            .collect();
        if coords.is_empty() {
            return Err(mosaix_core::tr("nothing to change in the selection"));
        }
        let fresh = self.composite_alone(&alone, &coords)?;
        let changed: Vec<TileCoord> = coords.clone();
        let mut tiles = (*own).clone();
        for c in coords {
            match fresh.get(&c) {
                Some(t) => tiles.insert(c, t.clone()),
                None => tiles.remove(&c),
            };
        }
        self.edit(label, |d| {
            if let Some(l) = d.find_mut(layer) {
                l.content = Content::Pixels(Arc::new(tiles));
            }
            true
        });
        self.dirty.extend(changed);
        Ok(())
    }
}
