//! Operations that produce new tiles via the apply kernel: committing a
//! stroke, filling, gradients, clearing, moving a layer and baking its mask.
//! Each produces fresh tiles (old ones stay in the undo history) and
//! records exactly one undo step.

use std::collections::BTreeSet;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use imagine_core::doc::{Content, MaskTile};
use imagine_core::{LayerId, MaskRef, TILE_SIZE, TileCoord};

use super::util::color_attachment;
use super::{Engine, NONE, SCRATCH_LAYERS, StrokeTarget, mask_slot};
use super::filter::FilterJob;
use super::magic::SimilarJob;
use super::transform::WarpJob;
use crate::pool::{Tile, texel_copy};

#[derive(Clone, Copy)]
enum Kernel {
    Apply,
    Warp,
    Similar,
    Filter,
    Liquify,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ApplyJob {
    color0: [f32; 4],
    color1: [f32; 4],
    g0: [f32; 2],
    g1: [f32; 2],
    origin: [f32; 2],
    _shift: [f32; 2],
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
const KIND_BAKE_MASK: u32 = 5;
const MODE_PAINT: u32 = 1;
const MODE_ERASE: u32 = 2;
const MODE_REPLACE: u32 = 3;

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
        self.run_tiles(target, Kernel::Apply, bytemuck::cast_slice(jobs), size_of::<ApplyJob>())
    }

    /// Like [`Self::run_apply`] for the warp kernel (grids must be uploaded).
    pub(super) fn run_warp(&mut self, target: StrokeTarget, jobs: &[WarpJob]) -> Option<Vec<Tile>> {
        self.run_tiles(target, Kernel::Warp, bytemuck::cast_slice(jobs), size_of::<WarpJob>())
    }

    /// Colour similarity tiles (mask pool) for the magic wand.
    pub(super) fn run_similar(&mut self, jobs: &[SimilarJob]) -> Option<Vec<Tile>> {
        self.run_tiles(StrokeTarget::Mask, Kernel::Similar, bytemuck::cast_slice(jobs), size_of::<SimilarJob>())
    }

    /// Liquify tiles (grids and the field must be uploaded).
    pub(super) fn run_liquify(&mut self, jobs: &[super::liquify::LiquifyJob]) -> Option<Vec<Tile>> {
        self.run_tiles(StrokeTarget::Pixels, Kernel::Liquify, bytemuck::cast_slice(jobs), size_of::<super::liquify::LiquifyJob>())
    }

    pub(super) fn run_filter(&mut self, jobs: &[FilterJob]) -> Option<Vec<Tile>> {
        self.run_tiles(StrokeTarget::Pixels, Kernel::Filter, bytemuck::cast_slice(jobs), size_of::<FilterJob>())
    }

    /// Job mixing `src_slot` over the layer's tile at `c` through the selection.
    pub(super) fn replace_job(&self, layer: LayerId, c: TileCoord, src_slot: u32) -> ApplyJob {
        let mut j = self.with_old(self.job(c), layer, StrokeTarget::Pixels, c);
        j.src_kind = KIND_STROKE;
        j.src_slots[0] = src_slot;
        j.mode = MODE_REPLACE;
        j
    }

    pub(super) fn run_mix(&mut self, jobs: &[ApplyJob]) -> Option<Vec<Tile>> {
        self.run_apply(StrokeTarget::Pixels, jobs)
    }

    fn run_tiles(&mut self, target: StrokeTarget, kernel: Kernel, jobs: &[u8], stride: usize) -> Option<Vec<Tile>> {
        let count = jobs.len() / stride;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(match target {
                StrokeTarget::Pixels => self.layer_pool.alloc()?,
                StrokeTarget::Mask => self.mask_pool.alloc()?,
            });
        }
        if count == 0 {
            return Some(out);
        }
        self.rebuild_arrays_bg();
        let batch_bytes = SCRATCH_LAYERS as usize * stride;
        for (b, batch) in jobs.chunks(batch_bytes).enumerate() {
            let n = batch.len() / stride;
            self.apply_buf.write(&self.device, &self.queue, batch);
            let mut extra = None;
            let bg = match kernel {
                Kernel::Apply | Kernel::Similar => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("apply"),
                    layout: &self.apply_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.apply_buf.buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.params.binding() },
                    ],
                }),
                Kernel::Warp | Kernel::Filter | Kernel::Liquify => self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("warp"),
                    layout: &self.warp_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.apply_buf.buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.params.binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.grid_buf.buf.as_entire_binding() },
                    ],
                }),
            };
            if matches!(kernel, Kernel::Similar) {
                let comp = super::frame::chunk_views(&self.comp_pool, super::MAX_COMPOSITE_CHUNKS);
                extra = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("composite array"),
                    layout: &self.comp_array_bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureViewArray(&comp) }],
                }));
            }
            if matches!(kernel, Kernel::Liquify) {
                let buf = &self.liquify.as_ref()?.buf;
                extra = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("liquify field"),
                    layout: &self.liquify_bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
                }));
            }
            let pipeline = match (kernel, target) {
                (Kernel::Apply, StrokeTarget::Pixels) => &self.apply_rgba_pipeline,
                (Kernel::Apply, StrokeTarget::Mask) => &self.apply_mask_pipeline,
                (Kernel::Warp, StrokeTarget::Pixels) => &self.warp_rgba_pipeline,
                (Kernel::Warp, StrokeTarget::Mask) => &self.warp_mask_pipeline,
                (Kernel::Similar, _) => &self.similar_pipeline,
                (Kernel::Filter, _) => &self.filter_pipeline,
                (Kernel::Liquify, _) => &self.liquify_pipeline,
            };
            let (views, scratch) = match target {
                StrokeTarget::Pixels => (&self.scratch_rgba_views, &self.scratch_rgba),
                StrokeTarget::Mask => (&self.scratch_mask_views, &self.scratch_mask),
            };
            let mut enc = self.device.create_command_encoder(&Default::default());
            for (i, view) in views.iter().enumerate().take(n) {
                let off = self.params.push([i as u32, 0, 0, 0]);
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("apply"),
                    color_attachments: &[Some(color_attachment(view, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)))],
                    ..Default::default()
                });
                rp.set_pipeline(pipeline);
                rp.set_bind_group(0, self.arrays_bg.as_ref().unwrap(), &[]);
                rp.set_bind_group(1, &bg, &[off]);
                if let Some(extra) = &extra {
                    rp.set_bind_group(2, extra, &[]);
                }
                rp.draw(0..3, 0..1);
            }
            let pool = match target {
                StrokeTarget::Pixels => &self.layer_pool,
                StrokeTarget::Mask => &self.mask_pool,
            };
            for (i, tile) in out[b * SCRATCH_LAYERS as usize..][..n].iter().enumerate() {
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
        self.params.upload(&self.queue);
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
            j.mode = match (stroke.params.smudge, stroke.params.erase) {
                (true, _) => MODE_REPLACE,
                (false, true) => MODE_ERASE,
                (false, false) => MODE_PAINT,
            };
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

    pub(super) fn check_target(&self, target: StrokeTarget) -> Result<LayerId, &'static str> {
        if self.stroke.is_some() {
            return Err("während eines Pinselstrichs nicht möglich");
        }
        let l = self.doc.active_layer().ok_or("keine aktive Ebene")?;
        if target == StrokeTarget::Pixels && !l.is_pixels() {
            return Err(if l.is_group() {
                "Gruppen haben keine Pixel"
            } else {
                "Einstellungsebenen haben keine Pixel – in die Maske malen (\\)"
            });
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

    /// Moves the whole active layer (pixels and mask) by whole pixels,
    /// ignoring the selection. Content outside the canvas is kept.
    pub fn translate_active(&mut self, dx: i64, dy: i64, label: &str) -> Result<(), &'static str> {
        let fwd = glam::DAffine2::from_translation(glam::DVec2::new(dx as f64, dy as f64));
        self.transform_layer(fwd, super::Interp::Nearest, false, label)
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
