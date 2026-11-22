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
            MergeKind::Down | MergeKind::Group => shown.find(ids[0]).map_or_else(|| mosaix_core::tr("Merged").to_string(), |l| l.name.clone()),
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

    /// Aligns the selected layers' contents (or one layer to the canvas):
    /// `edge` is left, center_x, right, top, center_y or bottom. Each layer
    /// moves by whole pixels; one undo step.
    pub fn align_layers(&mut self, edge: Align, label: &str) -> Result<(), &'static str> {
        let ids = self.doc.selected_layers();
        let mut boxes = Vec::new();
        for id in &ids {
            if let Some(b) = self.content_bounds(*id) {
                boxes.push((*id, b));
            }
        }
        if boxes.is_empty() {
            return Err(mosaix_core::tr("nothing to align"));
        }
        // Several layers align to their joint bounds, a single one to the canvas.
        let all = if boxes.len() > 1 {
            boxes.iter().fold((i64::MAX, i64::MAX, i64::MIN, i64::MIN), |a, (_, b)| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)))
        } else {
            (0, 0, self.doc.width as i64, self.doc.height as i64)
        };
        let moves: Vec<(LayerId, i64, i64)> = boxes
            .iter()
            .map(|&(id, (x0, y0, x1, y1))| {
                let (dx, dy) = match edge {
                    Align::Left => (all.0 - x0, 0),
                    Align::Right => (all.2 - x1, 0),
                    Align::CenterX => ((all.0 + all.2 - x0 - x1) / 2, 0),
                    Align::Top => (0, all.1 - y0),
                    Align::Bottom => (0, all.3 - y1),
                    Align::CenterY => (0, (all.1 + all.3 - y0 - y1) / 2),
                };
                (id, dx, dy)
            })
            .collect();
        self.move_layers(&moves, label)
    }

    /// Spaces the centres of three or more selected layers evenly between
    /// the outermost ones (`horizontal` or vertically). One undo step.
    pub fn distribute_layers(&mut self, horizontal: bool, label: &str) -> Result<(), &'static str> {
        let ids = self.doc.selected_layers();
        let mut c: Vec<(LayerId, f64)> = Vec::new();
        for id in &ids {
            if let Some((x0, y0, x1, y1)) = self.content_bounds(*id) {
                c.push((*id, if horizontal { (x0 + x1) as f64 / 2.0 } else { (y0 + y1) as f64 / 2.0 }));
            }
        }
        if c.len() < 3 {
            return Err(mosaix_core::tr("select at least three layers to distribute"));
        }
        c.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (first, last) = (c[0].1, c[c.len() - 1].1);
        let step = (last - first) / (c.len() - 1) as f64;
        let moves: Vec<(LayerId, i64, i64)> = c
            .iter()
            .enumerate()
            .map(|(i, &(id, at))| {
                let d = (first + step * i as f64 - at).round() as i64;
                if horizontal { (id, d, 0) } else { (id, 0, d) }
            })
            .collect();
        self.move_layers(&moves, label)
    }

    /// Moves layers by their own whole-pixel offsets, as one undo step.
    fn move_layers(&mut self, moves: &[(LayerId, i64, i64)], label: &str) -> Result<(), &'static str> {
        if moves.iter().any(|(id, ..)| self.doc.find(*id).is_some_and(|l| l.lock.position)) {
            return Err(mosaix_core::tr("the layer's position is locked"));
        }
        let (active, picked) = (self.doc.active, self.doc.picked.clone());
        let before = self.doc.clone();
        let n = self.history.undo_len();
        let mut result = Ok(());
        for &(id, dx, dy) in moves {
            if dx == 0 && dy == 0 {
                continue;
            }
            self.doc.active = id;
            self.doc.picked.clear();
            let fwd = glam::DAffine2::from_translation(glam::DVec2::new(dx as f64, dy as f64));
            if let Err(e) = self.transform_layer(fwd, super::Interp::Nearest, false, label) {
                result = Err(e);
                break;
            }
        }
        self.doc.active = active;
        self.doc.picked = picked;
        if result.is_err() {
            // Undo what was done so far.
            while self.history.undo_len() > n {
                self.undo();
            }
            self.doc.active = active;
            self.doc.picked = before.picked.clone();
            return result;
        }
        self.history.squash_since(n, before, label);
        Ok(())
    }

    /// Image › Adjustments › Auto Tone / Contrast / Color: Levels worked out
    /// from the active layer's histogram, applied to its pixels.
    pub fn auto_adjust(&mut self, mode: mosaix_core::adjust::AutoMode, label: &str) -> Result<mosaix_core::Adjustment, &'static str> {
        let layer = self.check_target(super::StrokeTarget::Pixels)?;
        let h = self.layer_histogram(layer).ok_or_else(|| mosaix_core::tr("no such layer"))?;
        let a = mosaix_core::adjust::auto_levels(&h.bins, mode);
        self.apply_adjustment(&a, label)?;
        Ok(a)
    }
}

/// Edges and centres for [`Engine::align_layers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    CenterX,
    Right,
    Top,
    CenterY,
    Bottom,
}
