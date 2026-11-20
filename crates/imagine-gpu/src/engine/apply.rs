//! Operations that produce new tiles via the apply kernel: committing a
//! stroke, filling, gradients, clearing, moving a layer and baking its mask.
//! Each produces fresh tiles (old ones stay in the undo history) and
//! records exactly one undo step.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use imagine_core::doc::{Content, MaskTile};
use imagine_core::{LayerId, MaskRef, TILE_SIZE, TileCoord};

use super::util::color_attachment;
use super::{Engine, NONE, SCRATCH_LAYERS, StrokeTarget, mask_slot};
use crate::pool::{Tile, texel_copy};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ApplyJob {
    color0: [f32; 4],
    color1: [f32; 4],
    g0: [f32; 2],
    g1: [f32; 2],
    origin: [f32; 2],
    shift: [f32; 2],
    src_slots: [u32; 4],
    src_consts: [f32; 4],
    old_slot: u32,
    old_const: f32,
    src_kind: u32,
    mode: u32,
    opacity: f32,
    sel_slot: u32,
    sel_const: f32,
    space: u32,
}

const KIND_STROKE: u32 = 0;
const KIND_COLOR: u32 = 1;
const KIND_LINEAR: u32 = 2;
const KIND_RADIAL: u32 = 3;
const KIND_TRANSLATE: u32 = 4;
const KIND_BAKE_MASK: u32 = 5;
const MODE_PAINT: u32 = 1;
const MODE_ERASE: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientKind {
    Linear,
    Radial,
}

impl Engine {
    fn job(&self, c: TileCoord) -> ApplyJob {
        let (x, y) = c.origin();
        let (sel_slot, sel_const) = self.sel_slot(c);
        ApplyJob {
            origin: [x as f32, y as f32],
            old_slot: NONE,
            src_slots: [NONE; 4],
            mode: MODE_PAINT,
            opacity: 1.0,
            sel_slot,
            sel_const,
            space: self.doc.blend_space as u32,
            ..Zeroable::zeroed()
        }
    }

    /// Sets `old_slot`/`old_const` from the target (layer pixels or layer mask) at `c`.
    fn with_old(&self, mut j: ApplyJob, layer: LayerId, target: StrokeTarget, c: TileCoord) -> ApplyJob {
        let l = self.doc.find(layer).expect("layer exists");
        match target {
            StrokeTarget::Pixels => {
                j.old_slot = l.tiles().and_then(|t| t.get(&c)).map_or(NONE, |t| t.slot().packed());
            }
            StrokeTarget::Mask => {
                let (slot, val) = mask_slot(l.mask.as_ref().map(|m| m.at(c)));
                j.old_slot = slot;
                j.old_const = val;
            }
        }
        j
    }

    /// Renders one new tile per job into the target pool. `None` if out of memory.
    fn run_apply(&mut self, target: StrokeTarget, jobs: &[ApplyJob]) -> Option<Vec<Tile>> {
        let mut out = Vec::with_capacity(jobs.len());
        for _ in jobs {
            out.push(match target {
                StrokeTarget::Pixels => self.layer_pool.alloc()?,
                StrokeTarget::Mask => self.mask_pool.alloc()?,
            });
        }
        self.rebuild_arrays_bg();
        for (b, batch) in jobs.chunks(SCRATCH_LAYERS as usize).enumerate() {
            self.apply_buf.write(&self.device, &self.queue, bytemuck::cast_slice(batch));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apply"),
                layout: &self.apply_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.apply_buf.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.params.binding() },
                ],
            });
            let (pipeline, views, scratch) = match target {
                StrokeTarget::Pixels => (&self.apply_rgba_pipeline, &self.scratch_rgba_views, &self.scratch_rgba),
                StrokeTarget::Mask => (&self.apply_mask_pipeline, &self.scratch_mask_views, &self.scratch_mask),
            };
            let mut enc = self.device.create_command_encoder(&Default::default());
            for (i, view) in views.iter().enumerate().take(batch.len()) {
                let off = self.params.push([i as u32, 0, 0, 0]);
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("apply"),
                    color_attachments: &[Some(color_attachment(view, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)))],
                    ..Default::default()
                });
                rp.set_pipeline(pipeline);
                rp.set_bind_group(0, self.arrays_bg.as_ref().unwrap(), &[]);
                rp.set_bind_group(1, &bg, &[off]);
                rp.draw(0..3, 0..1);
            }
            let pool = match target {
                StrokeTarget::Pixels => &self.layer_pool,
                StrokeTarget::Mask => &self.mask_pool,
            };
            for (i, tile) in out[b * SCRATCH_LAYERS as usize..][..batch.len()].iter().enumerate() {
                let s = tile.slot();
                enc.copy_texture_to_texture(
                    texel_copy(scratch, 0, i as u32),
                    texel_copy(&pool.chunk(s).texture, 0, s.layer as u32),
                    wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
                );
            }
            self.params.upload(&self.queue);
            self.queue.submit([enc.finish()]);
        }
        Some(out)
    }

    /// Writes new tiles into the layer (or its mask) as one undo step.
    fn commit(
        &mut self,
        layer: LayerId,
        target: StrokeTarget,
        tiles: Vec<(TileCoord, Option<Tile>)>,
        label: &str,
    ) -> bool {
        let coords: Vec<TileCoord> = tiles.iter().map(|(c, _)| *c).collect();
        let changed = self.edit(label, |d| {
            let l = d.find_mut(layer).expect("layer exists");
            match target {
                StrokeTarget::Pixels => {
                    let Content::Pixels(map) = &mut l.content else { return false };
                    let map = Arc::make_mut(map);
                    for (c, t) in tiles {
                        match t {
                            Some(t) => map.insert(c, t),
                            None => map.remove(&c),
                        };
                    }
                }
                StrokeTarget::Mask => {
                    let Some(mask) = &mut l.mask else { return false };
                    let map = Arc::make_mut(&mut mask.tiles);
                    for (c, t) in tiles {
                        match t {
                            Some(t) => map.insert(c, MaskTile::Pixels(t)),
                            None => map.remove(&c),
                        };
                    }
                }
            }
            true
        });
        self.recheck.extend(coords);
        changed
    }

    /// Bakes the stroke into new tiles and records an undo step.
    pub fn end_stroke(&mut self, label: &str) {
        if self.stroke.is_none() {
            return;
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut stats = Default::default();
        self.encode_paint(&mut enc, &mut stats);
        self.queue.submit([enc.finish()]);
        let stroke = self.stroke.take().unwrap();
        let target = stroke.params.target;
        let layer = stroke.layer;

        let mut coords: Vec<TileCoord> = stroke.tiles.iter().filter(|(_, (_, unused))| !unused).map(|(c, _)| *c).collect();
        coords.sort();
        let mut jobs = Vec::new();
        let mut kept = Vec::new();
        for c in coords {
            let mut j = self.with_old(self.job(c), layer, target, c);
            if target == StrokeTarget::Pixels && stroke.params.erase && j.old_slot == NONE {
                continue; // erasing nothing
            }
            j.src_kind = KIND_STROKE;
            j.src_slots[0] = stroke.tiles[&c].0.slot().packed();
            j.mode = if stroke.params.erase { MODE_ERASE } else { MODE_PAINT };
            j.opacity = stroke.params.opacity;
            jobs.push(j);
            kept.push(c);
        }
        if jobs.is_empty() {
            self.recheck.extend(stroke.tiles.keys().copied());
            return;
        }
        let Some(tiles) = self.run_apply(target, &jobs) else {
            tracing::warn!("GPU-Speicher voll – Strich verworfen");
            self.recheck.extend(stroke.tiles.keys().copied());
            return;
        };
        self.recheck.extend(stroke.tiles.keys().copied());
        drop(stroke); // stroke tiles return to the pool
        self.commit(layer, target, kept.into_iter().zip(tiles.into_iter().map(Some)).collect(), label);
    }

    /// Tiles that a fill-like operation must touch: everything selected.
    fn selected_coords(&self) -> Vec<TileCoord> {
        match &self.doc.selection {
            None => self.all_doc_coords(),
            Some(sel) if sel.default > 0 => {
                self.all_doc_coords().into_iter().filter(|c| !matches!(sel.at(*c), MaskRef::Solid(0))).collect()
            }
            Some(sel) => {
                let mut v: Vec<TileCoord> = sel
                    .tiles
                    .iter()
                    .filter(|(c, t)| !matches!(t, MaskTile::Solid(0)) && self.in_document(**c))
                    .map(|(c, _)| *c)
                    .collect();
                v.sort();
                v
            }
        }
    }

    fn check_target(&self, target: StrokeTarget) -> Result<LayerId, &'static str> {
        if self.stroke.is_some() {
            return Err("während eines Pinselstrichs nicht möglich");
        }
        let l = self.doc.active_layer().ok_or("keine aktive Ebene")?;
        if l.is_group() {
            return Err("Gruppen haben keine Pixel");
        }
        if target == StrokeTarget::Mask && l.mask.is_none() {
            return Err("die Ebene hat keine Maske");
        }
        Ok(l.id)
    }

    /// Fills the selection (or everything) of the active layer / its mask.
    /// `color` is premultiplied linear RGBA; for masks only red and alpha matter.
    pub fn fill(&mut self, target: StrokeTarget, color: [f32; 4], opacity: f32, label: &str) -> Result<usize, &'static str> {
        let layer = self.check_target(target)?;
        let coords = self.selected_coords();
        let jobs: Vec<ApplyJob> = coords
            .iter()
            .map(|&c| {
                let mut j = self.with_old(self.job(c), layer, target, c);
                j.src_kind = KIND_COLOR;
                j.color0 = color;
                j.opacity = opacity;
                j
            })
            .collect();
        self.finish(layer, target, coords, jobs, label)
    }

    /// Draws a gradient from `p0` (colour `c0`) to `p1` (`c1`) into the selection.
    #[allow(clippy::too_many_arguments)]
    pub fn gradient(
        &mut self,
        target: StrokeTarget,
        kind: GradientKind,
        p0: (f64, f64),
        p1: (f64, f64),
        c0: [f32; 4],
        c1: [f32; 4],
        opacity: f32,
        label: &str,
    ) -> Result<usize, &'static str> {
        let layer = self.check_target(target)?;
        let coords = self.selected_coords();
        let jobs: Vec<ApplyJob> = coords
            .iter()
            .map(|&c| {
                let mut j = self.with_old(self.job(c), layer, target, c);
                j.src_kind = if kind == GradientKind::Linear { KIND_LINEAR } else { KIND_RADIAL };
                j.color0 = c0;
                j.color1 = c1;
                j.g0 = [p0.0 as f32, p0.1 as f32];
                j.g1 = [p1.0 as f32, p1.1 as f32];
                j.opacity = opacity;
                j
            })
            .collect();
        self.finish(layer, target, coords, jobs, label)
    }

    /// Makes the selected pixels of the active layer transparent.
    pub fn clear_selected(&mut self, label: &str) -> Result<usize, &'static str> {
        let layer = self.check_target(StrokeTarget::Pixels)?;
        let existing: BTreeSet<TileCoord> =
            self.doc.find(layer).and_then(|l| l.tiles()).map(|t| t.keys().copied().collect()).unwrap_or_default();
        if self.doc.selection.is_none() {
            // Everything: just drop the tiles.
            let n = existing.len();
            self.commit(layer, StrokeTarget::Pixels, existing.into_iter().map(|c| (c, None)).collect(), label);
            return Ok(n);
        }
        let coords: Vec<TileCoord> = self.selected_coords().into_iter().filter(|c| existing.contains(c)).collect();
        let jobs: Vec<ApplyJob> = coords
            .iter()
            .map(|&c| {
                let mut j = self.with_old(self.job(c), layer, StrokeTarget::Pixels, c);
                j.src_kind = KIND_COLOR;
                j.color0 = [0.0, 0.0, 0.0, 1.0];
                j.mode = MODE_ERASE;
                j
            })
            .collect();
        self.finish(layer, StrokeTarget::Pixels, coords, jobs, label)
    }

    fn finish(
        &mut self,
        layer: LayerId,
        target: StrokeTarget,
        coords: Vec<TileCoord>,
        jobs: Vec<ApplyJob>,
        label: &str,
    ) -> Result<usize, &'static str> {
        if jobs.is_empty() {
            return Ok(0);
        }
        let tiles = self.run_apply(target, &jobs).ok_or("GPU-Speicher voll")?;
        let n = tiles.len();
        self.commit(layer, target, coords.into_iter().zip(tiles.into_iter().map(Some)).collect(), label);
        Ok(n)
    }

    /// Moves the active layer (pixels and mask) by whole pixels. Content moved
    /// outside the document is cut off.
    pub fn translate_active(&mut self, dx: i64, dy: i64, label: &str) -> Result<(), &'static str> {
        let layer = self.check_target(StrokeTarget::Pixels)?;
        if dx == 0 && dy == 0 {
            return Ok(());
        }
        let ts = TILE_SIZE as i64;
        let l = self.doc.find(layer).unwrap();
        let src: HashMap<TileCoord, Tile> = l.tiles().map(|t| (**t).clone()).unwrap_or_default();
        let mask = l.mask.clone();

        // Target tiles = shifted footprint of the source tiles.
        let (gw, gh) = imagine_core::tile::grid_size(self.doc.width, self.doc.height);
        let in_doc = |t: TileCoord| t.x >= 0 && t.y >= 0 && (t.x as u32) < gw && (t.y as u32) < gh;
        let shifted = |keys: Vec<TileCoord>| -> BTreeSet<TileCoord> {
            let mut out = BTreeSet::new();
            for c in keys {
                let (ox, oy) = c.origin();
                for ty in (oy + dy).div_euclid(ts)..=(oy + dy + ts - 1).div_euclid(ts) {
                    for tx in (ox + dx).div_euclid(ts)..=(ox + dx + ts - 1).div_euclid(ts) {
                        let t = TileCoord::new(tx as i32, ty as i32);
                        if in_doc(t) {
                            out.insert(t);
                        }
                    }
                }
            }
            out
        };
        // Source block for a target tile: base tile and offset into the 2×2 block.
        let block = |t: TileCoord| {
            let (ox, oy) = t.origin();
            let (sx, sy) = (ox - dx, oy - dy);
            let (bx, by) = (sx.div_euclid(ts), sy.div_euclid(ts));
            let shift = [(sx - bx * ts) as f32, (sy - by * ts) as f32];
            let srcs = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(i, j)| TileCoord::new((bx + i) as i32, (by + j) as i32));
            (srcs, shift)
        };

        let pixel_targets: Vec<TileCoord> = shifted(src.keys().copied().collect()).into_iter().collect();
        let mut pixel_jobs = Vec::new();
        for &t in &pixel_targets {
            let (srcs, shift) = block(t);
            let mut j = self.job(t);
            j.src_kind = KIND_TRANSLATE;
            j.shift = shift;
            j.src_slots = srcs.map(|s| src.get(&s).map_or(NONE, |t| t.slot().packed()));
            pixel_jobs.push(j);
        }
        let new_pixels = self.run_apply(StrokeTarget::Pixels, &pixel_jobs).ok_or("GPU-Speicher voll")?;

        let mut new_mask = None;
        if let Some(m) = &mask {
            let keys: Vec<TileCoord> = m.tiles.keys().copied().collect();
            let targets: Vec<TileCoord> = shifted(keys).into_iter().collect();
            let mut jobs = Vec::new();
            for &t in &targets {
                let (srcs, shift) = block(t);
                let mut j = self.job(t);
                j.src_kind = KIND_TRANSLATE;
                j.shift = shift;
                for (k, s) in srcs.iter().enumerate() {
                    let (slot, val) = mask_slot(Some(m.at(*s)));
                    j.src_slots[k] = slot;
                    j.src_consts[k] = val;
                }
                jobs.push(j);
            }
            let tiles = self.run_apply(StrokeTarget::Mask, &jobs).ok_or("GPU-Speicher voll")?;
            new_mask = Some((targets, tiles, m.default));
        }

        self.edit(label, |d| {
            let l = d.find_mut(layer).unwrap();
            l.content = Content::Pixels(Arc::new(pixel_targets.into_iter().zip(new_pixels).collect()));
            if let Some((targets, tiles, default)) = new_mask {
                l.mask = Some(imagine_core::Mask {
                    tiles: Arc::new(targets.into_iter().zip(tiles.into_iter().map(MaskTile::Pixels)).collect()),
                    default,
                });
            }
            true
        });
        Ok(())
    }

    /// Multiplies the active layer's pixels by its mask and removes the mask.
    pub fn apply_layer_mask(&mut self, label: &str) -> Result<(), &'static str> {
        let layer = self.check_target(StrokeTarget::Mask)?;
        let l = self.doc.find(layer).unwrap();
        let mask = l.mask.clone().unwrap();
        let tiles: Vec<(TileCoord, Tile)> =
            l.tiles().map(|t| t.iter().map(|(c, t)| (*c, t.clone())).collect()).unwrap_or_default();
        let mut keep = Vec::new();
        let mut jobs = Vec::new();
        let mut job_coords = Vec::new();
        for (c, t) in tiles {
            match mask.at(c) {
                MaskRef::Solid(255) => keep.push((c, t)),
                MaskRef::Solid(0) => {}
                m => {
                    let mut j = self.job(c);
                    j.src_kind = KIND_BAKE_MASK;
                    j.old_slot = t.slot().packed();
                    let (slot, val) = mask_slot(Some(m));
                    j.src_slots[0] = slot;
                    j.src_consts[0] = val;
                    jobs.push(j);
                    job_coords.push(c);
                }
            }
        }
        let baked = self.run_apply(StrokeTarget::Pixels, &jobs).ok_or("GPU-Speicher voll")?;
        self.edit(label, |d| {
            let l = d.find_mut(layer).unwrap();
            let map = keep.into_iter().chain(job_coords.into_iter().zip(baked)).collect();
            l.content = Content::Pixels(Arc::new(map));
            l.mask = None;
            true
        });
        Ok(())
    }
}
