//! Per-frame GPU work: painting dabs into stroke tiles, deciding which
//! composite tiles are stale (signatures), re-flattening them and rebuilding
//! their mips.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use glam::DVec2;
use imagine_core::adjust::{GpuAdjust, KIND_LUT};
use imagine_core::doc::hash_program;
use imagine_core::tile::tiles_in_rect;
use imagine_core::{LayerId, MaskRef, Op, TILE_MIP_LEVELS, TILE_SIZE, TileCoord};

use super::util::color_attachment;
use super::{
    CompositeEntry, Engine, MAX_GROUP_DEPTH, MAX_LAYER_CHUNKS, MAX_MASK_CHUNKS, NONE, RenderStats,
    StrokeParams, StrokeTarget, UNIFORM_STRIDE, make_origin_buf, mask_slot,
};
use crate::pool::{Pool, SlotId, Tile};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOp {
    kind: u32,
    slot: u32,
    blend: u32,
    opacity: f32,
    stroke_slot: u32,
    stroke_opacity: f32,
    stroke_mode: u32,
    mask_slot: u32,
    mask_const: f32,
    sel_slot: u32,
    sel_const: f32,
    /// Adjustments: blend mode of the adjusted result onto what is below.
    adj_blend: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuJob {
    dst_layer: u32,
    op_start: u32,
    op_count: u32,
    _pad: u32,
}

/// Flatten jobs collected for one dispatch.
#[derive(Default)]
pub(super) struct FlattenBatch {
    ops: Vec<GpuOp>,
    jobs: BTreeMap<u16, Vec<GpuJob>>,
    /// Adjustment parameters, once per adjustment layer: id → (offset, kind).
    adjust: Vec<[f32; 4]>,
    adjust_at: HashMap<LayerId, (u32, u32)>,
    deep_warned: bool,
}

impl FlattenBatch {
    /// Adds the program of one tile, rendered into the composite slot `dst`.
    pub(super) fn push(
        &mut self,
        program: &[Op<'_, Tile>],
        stroke: Option<(LayerId, SlotId, StrokeParams)>,
        (sel_slot, sel_const): (u32, f32),
        dst: SlotId,
    ) {
        if self.adjust.is_empty() {
            self.adjust.push([0.0; 4]);
        }
        let start = self.ops.len() as u32;
        let mut depth = 0usize;
        for op in program {
            let gpu = match *op {
                Op::Layer { id, tile, opacity, blend, mask } => {
                    let (mask_slot, mask_const) = mask_slot(mask);
                    let st = stroke.filter(|s| s.0 == id);
                    let stroke_mode = st.map_or(0, |s| match (s.2.target, s.2.erase) {
                        _ if s.2.working_copy() => 5,
                        (StrokeTarget::Pixels, false) => 1,
                        (StrokeTarget::Pixels, true) => 2,
                        (StrokeTarget::Mask, false) => 3,
                        (StrokeTarget::Mask, true) => 4,
                    });
                    GpuOp {
                        kind: 0,
                        slot: tile.map_or(NONE, |t| t.slot().packed()),
                        blend: blend as u32,
                        opacity,
                        stroke_slot: st.map_or(NONE, |s| s.1.packed()),
                        stroke_opacity: st.map_or(0.0, |s| s.2.opacity),
                        stroke_mode,
                        mask_slot,
                        mask_const,
                        sel_slot,
                        sel_const,
                        adj_blend: 0,
                    }
                }
                Op::Adjust { id, adj, opacity, blend, mask } => {
                    let adjust = &mut self.adjust;
                    let (off, kind) = *self.adjust_at.entry(id).or_insert_with(|| {
                        let off = adjust.len() as u32;
                        match adj.gpu() {
                            GpuAdjust::Lut(lut) => {
                                adjust.extend(lut.iter().map(|c| [c[0], c[1], c[2], 0.0]));
                                (off, KIND_LUT)
                            }
                            GpuAdjust::Shader(kind, p) => {
                                adjust.push([p[0], p[1], p[2], p[3]]);
                                adjust.push([p[4], p[5], p[6], p[7]]);
                                (off, kind)
                            }
                        }
                    });
                    let (mask_slot, mask_const) = mask_slot(mask);
                    // A stroke into this adjustment's mask (preview).
                    let st = stroke.filter(|s| s.0 == id && s.2.target == StrokeTarget::Mask);
                    GpuOp {
                        kind: 3,
                        slot: off,
                        blend: kind,
                        opacity,
                        mask_slot,
                        mask_const,
                        stroke_slot: st.map_or(NONE, |s| s.1.packed()),
                        stroke_opacity: st.map_or(0.0, |s| s.2.opacity),
                        stroke_mode: st.map_or(0, |s| if s.2.erase { 4 } else { 3 }),
                        sel_slot,
                        sel_const,
                        adj_blend: blend as u32,
                    }
                }
                Op::Push => {
                    depth += 1;
                    if depth > MAX_GROUP_DEPTH && !self.deep_warned {
                        tracing::warn!("Gruppen tiefer als {MAX_GROUP_DEPTH} Ebenen werden falsch dargestellt");
                        self.deep_warned = true;
                    }
                    GpuOp { kind: 1, ..Zeroable::zeroed() }
                }
                Op::Pop { opacity, blend, mask } => {
                    depth = depth.saturating_sub(1);
                    let (mask_slot, mask_const) = mask_slot(mask);
                    GpuOp { kind: 2, blend: blend as u32, opacity, mask_slot, mask_const, ..Zeroable::zeroed() }
                }
            };
            self.ops.push(gpu);
        }
        self.jobs.entry(dst.chunk).or_default().push(GpuJob {
            dst_layer: dst.layer as u32,
            op_start: start,
            op_count: self.ops.len() as u32 - start,
            _pad: 0,
        });
    }
}

/// Binding-array views of all chunks of a pool, padded to `n` entries.
pub(super) fn chunk_views(pool: &Pool, n: u32) -> Vec<&wgpu::TextureView> {
    let chunks = &pool.chunks;
    (0..n as usize).map(|i| &chunks.get(i).unwrap_or(&chunks[0]).view).collect()
}

impl Engine {
    /// The binding arrays over all layer and mask chunks; rebuilt when a chunk is added.
    pub(super) fn rebuild_arrays_bg(&mut self) {
        let generation = (self.layer_pool.generation, self.mask_pool.generation);
        if self.arrays_bg.is_some() && self.arrays_gen == generation {
            return;
        }
        let layers = chunk_views(&self.layer_pool, MAX_LAYER_CHUNKS);
        let masks = chunk_views(&self.mask_pool, MAX_MASK_CHUNKS);
        self.arrays_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile arrays"),
            layout: &self.arrays_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureViewArray(&layers) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureViewArray(&masks) },
            ],
        }));
        self.arrays_gen = generation;
    }

    /// Selection value at a tile as shader slot/constant (no selection = 1).
    pub(super) fn sel_slot(&self, c: TileCoord) -> (u32, f32) {
        mask_slot(self.doc.selection.as_ref().map(|s| s.at(c)))
    }

    /// Whether the selection allows any change in this tile.
    pub(super) fn selected_any(&self, c: TileCoord) -> bool {
        !matches!(self.doc.selection.as_ref().map(|s| s.at(c)), Some(MaskRef::Solid(0)))
    }

    pub(super) fn encode_paint(&mut self, enc: &mut wgpu::CommandEncoder, stats: &mut RenderStats) {
        if self.pending_dabs.is_empty() {
            return;
        }
        let dabs = std::mem::take(&mut self.pending_dabs);
        let Some(stroke) = &self.stroke else { return };
        let (w, h) = (self.doc.width, self.doc.height);

        let mut touched = BTreeSet::new();
        for d in &dabs {
            let (min, max) = d.bounds();
            touched.extend(tiles_in_rect(min, max, w, h));
        }
        // Tiles outside the selection cannot change.
        touched.retain(|c| self.selected_any(*c));
        let mut passes = Vec::new();
        let stroke_tiles_missing: Vec<TileCoord> =
            touched.iter().copied().filter(|c| !stroke.tiles.contains_key(c)).collect();
        for c in stroke_tiles_missing {
            let Some(t) = self.layer_pool.alloc() else {
                tracing::warn!("Ebenen-Speicher voll");
                continue;
            };
            self.stroke.as_mut().unwrap().tiles.insert(c, (t, true));
            self.recheck.insert(c);
        }
        let stroke = self.stroke.as_mut().unwrap();
        let smudge = stroke.params.working_copy();
        let clone = stroke.params.clone.filter(|c| !c.spot);
        let mut fresh = Vec::new();
        for c in touched {
            if let Some((tile, needs_clear)) = stroke.tiles.get_mut(&c) {
                let clear = std::mem::replace(needs_clear, false);
                if clear {
                    fresh.push(c);
                }
                passes.push((c, tile.slot(), clear));
                self.dirty.insert(c);
            }
        }
        if smudge {
            self.init_smudge_tiles(enc, &fresh);
            stats.dabs = match clone {
                Some(src) => self.encode_clone(enc, &dabs, src),
                None => self.encode_smudge(enc, &dabs),
            };
            stats.paint_passes = passes.len() as u32;
            return;
        }

        self.dab_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&dabs));
        if passes.len() as u64 > self.origin_cap {
            self.origin_cap = (passes.len() as u64).next_power_of_two();
            (self.origin_buf, self.origin_bg) = make_origin_buf(&self.device, &self.brush_bgl, self.origin_cap);
        }
        let mut origins = vec![0u8; passes.len() * UNIFORM_STRIDE as usize];
        for (i, (c, _, _)) in passes.iter().enumerate() {
            let (x, y) = c.origin();
            let v: [f32; 4] = [x as f32, y as f32, 0.0, 0.0];
            origins[i * UNIFORM_STRIDE as usize..][..16].copy_from_slice(bytemuck::bytes_of(&v));
        }
        self.queue.write_buffer(&self.origin_buf, 0, &origins);

        for (i, &(_, slot, clear)) in passes.iter().enumerate() {
            let target = self.layer_pool.target(slot).clone();
            let load = if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load };
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paint"),
                color_attachments: &[Some(color_attachment(&target, load))],
                ..Default::default()
            });
            rp.set_pipeline(&self.brush_pipeline);
            rp.set_bind_group(0, &self.origin_bg, &[(i as u64 * UNIFORM_STRIDE) as u32]);
            rp.set_bind_group(1, &self.tips_bg, &[]);
            rp.set_vertex_buffer(0, self.dab_buf.buf.slice(..));
            rp.draw(0..4, 0..dabs.len() as u32);
        }
        stats.dabs = dabs.len() as u32;
        stats.paint_passes = passes.len() as u32;
    }

    fn stroke_at(&self, c: TileCoord) -> Option<(LayerId, &Tile, StrokeParams)> {
        let s = self.stroke.as_ref()?;
        s.tiles.get(&c).map(|(t, _)| (s.layer, t, s.params))
    }

    fn signature(&self, c: TileCoord) -> Option<u64> {
        let stroke = self.stroke_at(c);
        let ops = self.shown().program(c, stroke.map(|s| s.0));
        if ops.is_empty() {
            return None;
        }
        let mut h = DefaultHasher::new();
        hash_program(&ops, |t: &Tile| t.serial(), &mut h);
        (self.shown().blend_space as u32).hash(&mut h);
        if let Some((_, t, p)) = stroke {
            (t.serial(), p.opacity.to_bits(), p.erase, p.target, p.working_copy()).hash(&mut h);
            let (slot, val) = self.sel_slot(c);
            (slot, val.to_bits()).hash(&mut h);
        }
        Some(h.finish())
    }

    pub(super) fn update_signatures(&mut self) {
        self.refresh_smart();
        let coords: Vec<TileCoord> = if self.recheck_all {
            let mut s = self.shown().all_tile_coords();
            s.extend(self.comp.keys().copied());
            if let Some(st) = &self.stroke {
                s.extend(st.tiles.keys().copied());
            }
            self.recheck.clear();
            s.into_iter().collect()
        } else {
            self.recheck.drain().collect()
        };
        self.recheck_all = false;
        for c in coords {
            // Layers may hold content outside the canvas; it is not composited.
            let sig = if self.in_document(c) { self.signature(c) } else { None };
            match sig {
                None => {
                    self.comp.remove(&c);
                    self.dirty.remove(&c);
                }
                Some(sig) => {
                    let e = self.comp.entry(c).or_insert(CompositeEntry { tile: None, sig: !sig });
                    if e.sig != sig {
                        e.sig = sig;
                        self.dirty.insert(c);
                    }
                }
            }
        }
    }

    pub(super) fn encode_flatten(&mut self, enc: &mut wgpu::CommandEncoder, budget: Option<usize>, stats: &mut RenderStats) {
        if self.dirty.is_empty() {
            return;
        }
        let mut list: Vec<TileCoord> = self.dirty.iter().copied().collect();
        if let Some(budget) = budget
            && list.len() > budget
        {
            let (min, max) = self.visible;
            let center = (min + max) * 0.5;
            let ts = TILE_SIZE as f64;
            let key = |c: &TileCoord| {
                let p = DVec2::new((c.x as f64 + 0.5) * ts, (c.y as f64 + 0.5) * ts);
                let visible = p.x + ts > min.x && p.x - ts < max.x && p.y + ts > min.y && p.y - ts < max.y;
                (if visible { 0.0 } else { 1e18 }) + p.distance_squared(center)
            };
            list.sort_by(|a, b| key(a).total_cmp(&key(b)));
            list.truncate(budget);
        }

        let mut b = FlattenBatch::default();
        for c in list {
            self.dirty.remove(&c);
            let stroke = self.stroke_at(c).map(|(l, t, p)| (l, t.slot(), p));
            let sel = self.sel_slot(c);
            let shown = match (&self.render, &self.preview) {
                (Some(r), _) => r,
                (None, Some(p)) => &p.doc,
                (None, None) => &self.doc,
            };
            let program = shown.program(c, stroke.map(|s| s.0));
            if program.is_empty() {
                continue;
            }
            let Some(entry) = self.comp.get_mut(&c) else { continue };
            if entry.tile.is_none() {
                entry.tile = self.comp_pool.alloc();
            }
            let Some(tile) = &entry.tile else {
                tracing::warn!("Composite-Speicher voll");
                continue;
            };
            let slot = tile.slot();
            b.push(&program, stroke, sel, slot);
            self.pending_mips.push(slot);
            stats.flattened += 1;
        }
        self.dispatch_flatten(enc, b);
    }

    /// Records the flatten dispatches of a batch (buffers are written now).
    pub(super) fn dispatch_flatten(&mut self, enc: &mut wgpu::CommandEncoder, b: FlattenBatch) {
        if b.jobs.is_empty() {
            return;
        }
        let FlattenBatch { ops, jobs, adjust, .. } = b;
        let mut all_jobs = Vec::new();
        let mut ranges = Vec::new();
        for (chunk, js) in jobs {
            ranges.push((chunk, all_jobs.len() as u32, js.len() as u32));
            all_jobs.extend(js);
        }
        self.ops_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&ops));
        self.jobs_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&all_jobs));
        self.adjust_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&adjust));
        self.rebuild_arrays_bg();
        let space = self.doc.blend_space as u32;
        let mut dispatches = Vec::new();
        for (chunk, base, count) in ranges {
            let off = self.params.push([base, count, space, 0]);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flatten"),
                layout: &self.flatten_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.ops_buf.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.jobs_buf.buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.params.binding() },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.comp_pool.chunks[chunk as usize].mip_views[0]),
                    },
                    wgpu::BindGroupEntry { binding: 4, resource: self.adjust_buf.buf.as_entire_binding() },
                ],
            });
            dispatches.push((bg, off, count));
        }
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("flatten"), timestamp_writes: None });
        cp.set_pipeline(&self.flatten_pipeline);
        cp.set_bind_group(0, self.arrays_bg.as_ref().unwrap(), &[]);
        for (bg, off, count) in &dispatches {
            cp.set_bind_group(1, bg, &[*off]);
            cp.dispatch_workgroups(TILE_SIZE / 8, TILE_SIZE / 8, *count);
        }
    }

    pub(super) fn encode_mips(&mut self, enc: &mut wgpu::CommandEncoder) {
        if self.pending_mips.is_empty() {
            return;
        }
        let mut slots = std::mem::take(&mut self.pending_mips);
        slots.sort();
        slots.dedup();
        let mut ranges: Vec<(u16, u32, u32)> = Vec::new();
        let list: Vec<u32> = slots.iter().map(|s| s.layer as u32).collect();
        for (i, s) in slots.iter().enumerate() {
            match ranges.last_mut() {
                Some(r) if r.0 == s.chunk => r.2 += 1,
                _ => ranges.push((s.chunk, i as u32, 1)),
            }
        }
        self.list_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&list));
        let mut dispatches = Vec::new();
        for (chunk, base, count) in ranges {
            let off = self.params.push([base, count, 0, 0]);
            let ch = &self.comp_pool.chunks[chunk as usize];
            for mip in 1..TILE_MIP_LEVELS {
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mip"),
                    layout: &self.mip_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&ch.mip_views[mip as usize - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&ch.mip_views[mip as usize]),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: self.list_buf.buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.params.binding() },
                    ],
                });
                let s = TILE_SIZE >> mip;
                dispatches.push((bg, off, s.div_ceil(8), count));
            }
        }
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mips"), timestamp_writes: None });
        cp.set_pipeline(&self.mip_pipeline);
        for (bg, off, groups, count) in &dispatches {
            cp.set_bind_group(0, bg, &[*off]);
            cp.dispatch_workgroups(*groups, *groups, *count);
        }
    }

    /// True while there is work left that needs further frames.
    pub fn busy(&self) -> bool {
        !self.dirty.is_empty() || !self.pending_dabs.is_empty() || !self.recheck.is_empty() || self.recheck_all
    }

    /// Processes all pending work without a budget (for export, saving, tests).
    pub fn flush(&mut self) -> RenderStats {
        let mut stats = RenderStats::default();
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode_paint(&mut enc, &mut stats);
        self.update_signatures();
        self.encode_flatten(&mut enc, None, &mut stats);
        self.encode_mips(&mut enc);
        self.params.upload(&self.queue);
        self.queue.submit([enc.finish()]);
        stats
    }

    /// Coordinates that currently have composited content.
    pub fn composite_coords(&self) -> Vec<TileCoord> {
        let mut v: Vec<_> = self.comp.iter().filter(|(_, e)| e.tile.is_some()).map(|(c, _)| *c).collect();
        v.sort();
        v
    }
}
