//! Smudge strokes. The stroke tiles start as a copy of the layer; each dab
//! first picks up colour from the canvas into a small buffer (mixing with
//! what it already carries), then stamps that buffer through the brush tip.
//! Dabs depend on each other, so they are processed one by one.

use std::collections::BTreeSet;

use bytemuck::{Pod, Zeroable};
use imagine_core::tile::tiles_in_rect;
use imagine_core::{Dab, TILE_SIZE, TileCoord};

use super::util::color_attachment;
use super::{Engine, NONE};
use crate::pool::{TILE_FORMAT, texel_copy};

/// Resolution of the pick-up buffer (covers the dab's bounding square).
pub(super) const BUF: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SmudgeJob {
    center: [f32; 2],
    radius: f32,
    hardness: f32,
    shape: [f32; 4],
    tip: u32,
    rate: f32,
    primed: u32,
    stroke_slot: u32,
    origin: [f32; 2],
    grid_x0: i32,
    grid_y0: i32,
    grid_w: u32,
    grid_h: u32,
    /// Clone stamp: source = destination + offset.
    offset: [i32; 2],
}

/// Two ping-pong pick-up buffers and their bind groups (index = buffer read).
pub(super) struct SmudgeBuffers {
    views: [wgpu::TextureView; 2],
    bgs: [wgpu::BindGroup; 2],
}

impl SmudgeBuffers {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        tips: &wgpu::TextureView,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("smudge"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("smudge tips"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let views = [0, 1].map(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("smudge buffer"),
                    size: wgpu::Extent3d {
                        width: BUF,
                        height: BUF,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: TILE_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });
        let bgs = [0, 1].map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smudge"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(tips),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&tip_sampler),
                    },
                ],
            })
        });
        Self { views, bgs }
    }
}

impl Engine {
    /// Copies the layer into freshly allocated smudge stroke tiles (or clears them).
    pub(super) fn init_smudge_tiles(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        fresh: &[TileCoord],
    ) {
        let Some(stroke) = &self.stroke else { return };
        let layer = self.doc.find(stroke.layer).and_then(|l| l.tiles());
        for c in fresh {
            let (dst, _) = &stroke.tiles[c];
            let ds = dst.slot();
            match layer.and_then(|t| t.get(c)) {
                Some(src) => {
                    let ss = src.slot();
                    enc.copy_texture_to_texture(
                        texel_copy(&self.layer_pool.chunk(ss).texture, 0, ss.layer as u32),
                        texel_copy(&self.layer_pool.chunk(ds).texture, 0, ds.layer as u32),
                        wgpu::Extent3d {
                            width: TILE_SIZE,
                            height: TILE_SIZE,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                None => {
                    let target = self.layer_pool.target(ds).clone();
                    let _ = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("smudge clear"),
                        color_attachments: &[Some(color_attachment(
                            &target,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        ))],
                        ..Default::default()
                    });
                }
            }
        }
    }

    /// Smudges dab by dab into the stroke tiles.
    pub(super) fn encode_smudge(&mut self, enc: &mut wgpu::CommandEncoder, dabs: &[Dab]) -> u32 {
        if self.smudge_buffers.is_none() {
            self.smudge_buffers = Some(SmudgeBuffers::new(
                &self.device,
                &self.smudge_tex_bgl,
                &self.tips_view,
            ));
        }
        let Some(stroke) = &self.stroke else { return 0 };
        let (w, h) = (self.doc.width, self.doc.height);

        // Lookup grid over all stroke tiles (for picking up canvas colour).
        let entries: Vec<(TileCoord, u32)> = stroke
            .tiles
            .iter()
            .map(|(c, (t, _))| (*c, t.slot().packed()))
            .collect();
        let (gx0, gy0) = (
            entries.iter().map(|e| e.0.x).min().unwrap_or(0),
            entries.iter().map(|e| e.0.y).min().unwrap_or(0),
        );
        let gw = entries
            .iter()
            .map(|e| e.0.x)
            .max()
            .map_or(0, |m| (m - gx0 + 1) as u32);
        let gh = entries
            .iter()
            .map(|e| e.0.y)
            .max()
            .map_or(0, |m| (m - gy0 + 1) as u32);
        let mut grid = vec![NONE; (gw * gh).max(1) as usize];
        for (c, slot) in &entries {
            grid[((c.y - gy0) as u32 * gw + (c.x - gx0) as u32) as usize] = *slot;
        }

        // Jobs: per dab one pick-up pass, then one write pass per touched tile.
        enum Pass {
            Pickup {
                job: u32,
            },
            Write {
                job: u32,
                scratch: u32,
                coord: TileCoord,
            },
        }
        let mut jobs = Vec::new();
        let mut passes = Vec::new();
        let mut scratch = 0u32;
        for d in dabs {
            let base = SmudgeJob {
                center: d.center,
                radius: d.radius,
                hardness: d.hardness,
                shape: [d.angle.cos(), d.angle.sin(), d.roundness, 0.0],
                tip: d.tip,
                rate: 1.0 - d.color[3],
                primed: self.smudge_primed as u32,
                stroke_slot: NONE,
                origin: [0.0; 2],
                grid_x0: gx0,
                grid_y0: gy0,
                grid_w: gw,
                grid_h: gh,
                offset: [0; 2],
            };
            self.smudge_primed = true;
            passes.push(Pass::Pickup {
                job: jobs.len() as u32,
            });
            jobs.push(base);
            let (min, max) = d.bounds();
            for c in tiles_in_rect(min, max, w, h) {
                let Some((tile, _)) = stroke.tiles.get(&c) else {
                    continue;
                };
                let (ox, oy) = c.origin();
                passes.push(Pass::Write {
                    job: jobs.len() as u32,
                    scratch,
                    coord: c,
                });
                jobs.push(SmudgeJob {
                    stroke_slot: tile.slot().packed(),
                    origin: [ox as f32, oy as f32],
                    ..base
                });
                scratch = (scratch + 1) % super::SCRATCH_LAYERS;
            }
        }
        self.grid_buf
            .write(&self.device, &self.queue, bytemuck::cast_slice(&grid));
        self.smudge_jobs
            .write(&self.device, &self.queue, bytemuck::cast_slice(&jobs));
        self.rebuild_arrays_bg();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smudge jobs"),
            layout: &self.warp_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.smudge_jobs.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.grid_buf.buf.as_entire_binding(),
                },
            ],
        });
        let bufs = self.smudge_buffers.as_ref().unwrap();
        let stroke = self.stroke.as_ref().unwrap();
        let mut cur = self.smudge_cur;
        for pass in passes {
            match pass {
                Pass::Pickup { job } => {
                    let next = 1 - cur;
                    let off = self.params.push([job, 0, 0, 0]);
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("smudge pickup"),
                        color_attachments: &[Some(color_attachment(
                            &bufs.views[next],
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        ))],
                        ..Default::default()
                    });
                    rp.set_pipeline(&self.smudge_pickup_pipeline);
                    rp.set_bind_group(0, self.arrays_bg.as_ref().unwrap(), &[]);
                    rp.set_bind_group(1, &bg, &[off]);
                    rp.set_bind_group(2, &bufs.bgs[cur], &[]);
                    rp.draw(0..3, 0..1);
                    cur = next;
                }
                Pass::Write {
                    job,
                    scratch,
                    coord,
                } => {
                    let off = self.params.push([job, 0, 0, 0]);
                    {
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("smudge write"),
                            color_attachments: &[Some(color_attachment(
                                &self.scratch_rgba_views[scratch as usize],
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            ))],
                            ..Default::default()
                        });
                        rp.set_pipeline(&self.smudge_write_pipeline);
                        rp.set_bind_group(0, self.arrays_bg.as_ref().unwrap(), &[]);
                        rp.set_bind_group(1, &bg, &[off]);
                        rp.set_bind_group(2, &bufs.bgs[cur], &[]);
                        rp.draw(0..3, 0..1);
                    }
                    let s = stroke.tiles[&coord].0.slot();
                    enc.copy_texture_to_texture(
                        texel_copy(&self.scratch_rgba, 0, scratch),
                        texel_copy(&self.layer_pool.chunk(s).texture, 0, s.layer as u32),
                        wgpu::Extent3d {
                            width: TILE_SIZE,
                            height: TILE_SIZE,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }
        self.smudge_cur = cur;
        dabs.len() as u32
    }
}

impl Engine {
    /// Clone stamp: per dab and touched stroke tile, the source pixels
    /// (`offset` away, exact integer copy) are painted through the tip.
    pub(super) fn encode_clone(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        dabs: &[Dab],
        src: super::CloneSource,
    ) -> u32 {
        if self.smudge_buffers.is_none() {
            self.smudge_buffers = Some(SmudgeBuffers::new(
                &self.device,
                &self.smudge_tex_bgl,
                &self.tips_view,
            ));
        }
        if self.stroke.is_none() {
            return 0;
        }
        let (w, h) = (self.doc.width, self.doc.height);
        let [ox, oy] = src.offset;
        // Source tiles: the layer as it was when the stroke started, or the flattened image.
        let source: Vec<(TileCoord, u32)> = if src.all_layers {
            let mut need = BTreeSet::new();
            for d in dabs {
                let (min, max) = d.bounds();
                let (fx, fy) = (ox as f64, oy as f64);
                need.extend(tiles_in_rect(
                    (min.0 + fx, min.1 + fy),
                    (max.0 + fx, max.1 + fy),
                    w,
                    h,
                ));
            }
            for c in need {
                if self.clone_cache.contains_key(&c) {
                    continue;
                }
                let copy = self
                    .comp
                    .get(&c)
                    .and_then(|e| e.tile.clone())
                    .and_then(|ct| {
                        let t = self.layer_pool.alloc()?;
                        let (cs, ts) = (ct.slot(), t.slot());
                        enc.copy_texture_to_texture(
                            texel_copy(&self.comp_pool.chunk(cs).texture, 0, cs.layer as u32),
                            texel_copy(&self.layer_pool.chunk(ts).texture, 0, ts.layer as u32),
                            wgpu::Extent3d {
                                width: TILE_SIZE,
                                height: TILE_SIZE,
                                depth_or_array_layers: 1,
                            },
                        );
                        Some(t)
                    });
                self.clone_cache.insert(c, copy);
            }
            self.clone_cache
                .iter()
                .filter_map(|(c, t)| t.as_ref().map(|t| (*c, t.slot().packed())))
                .collect()
        } else {
            let stroke = self.stroke.as_ref().unwrap();
            self.doc
                .find(stroke.layer)
                .and_then(|l| l.tiles())
                .map_or(Vec::new(), |t| {
                    t.iter().map(|(c, t)| (*c, t.slot().packed())).collect()
                })
        };
        let stroke = self.stroke.as_ref().unwrap();
        let (gx0, gy0) = (
            source.iter().map(|e| e.0.x).min().unwrap_or(0),
            source.iter().map(|e| e.0.y).min().unwrap_or(0),
        );
        let gw = source
            .iter()
            .map(|e| e.0.x)
            .max()
            .map_or(0, |m| (m - gx0 + 1) as u32);
        let gh = source
            .iter()
            .map(|e| e.0.y)
            .max()
            .map_or(0, |m| (m - gy0 + 1) as u32);
        let mut grid = vec![NONE; (gw * gh).max(1) as usize];
        for (c, slot) in &source {
            grid[((c.y - gy0) as u32 * gw + (c.x - gx0) as u32) as usize] = *slot;
        }
        let mut jobs = Vec::new();
        let mut passes = Vec::new();
        let mut scratch = 0u32;
        for d in dabs {
            let (min, max) = d.bounds();
            for c in tiles_in_rect(min, max, w, h) {
                let Some((tile, _)) = stroke.tiles.get(&c) else {
                    continue;
                };
                let (tx, ty) = c.origin();
                passes.push((jobs.len() as u32, scratch, c));
                jobs.push(SmudgeJob {
                    center: d.center,
                    radius: d.radius,
                    hardness: d.hardness,
                    shape: [d.angle.cos(), d.angle.sin(), d.roundness, 0.0],
                    tip: d.tip,
                    rate: d.color[3],
                    primed: 1,
                    stroke_slot: tile.slot().packed(),
                    origin: [tx as f32, ty as f32],
                    grid_x0: gx0,
                    grid_y0: gy0,
                    grid_w: gw,
                    grid_h: gh,
                    offset: [ox, oy],
                });
                scratch = (scratch + 1) % super::SCRATCH_LAYERS;
            }
        }
        if jobs.is_empty() {
            return dabs.len() as u32;
        }
        self.grid_buf
            .write(&self.device, &self.queue, bytemuck::cast_slice(&grid));
        self.smudge_jobs
            .write(&self.device, &self.queue, bytemuck::cast_slice(&jobs));
        self.rebuild_arrays_bg();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clone jobs"),
            layout: &self.warp_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.smudge_jobs.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params.binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.grid_buf.buf.as_entire_binding(),
                },
            ],
        });
        let bufs = self.smudge_buffers.as_ref().unwrap();
        let stroke = self.stroke.as_ref().unwrap();
        for (job, scratch, coord) in passes {
            let off = self.params.push([job, 0, 0, 0]);
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("clone"),
                    color_attachments: &[Some(color_attachment(
                        &self.scratch_rgba_views[scratch as usize],
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    ))],
                    ..Default::default()
                });
                rp.set_pipeline(&self.clone_pipeline);
                rp.set_bind_group(0, self.arrays_bg.as_ref().unwrap(), &[]);
                rp.set_bind_group(1, &bg, &[off]);
                rp.set_bind_group(2, &bufs.bgs[0], &[]);
                rp.draw(0..3, 0..1);
            }
            let s = stroke.tiles[&coord].0.slot();
            enc.copy_texture_to_texture(
                texel_copy(&self.scratch_rgba, 0, scratch),
                texel_copy(&self.layer_pool.chunk(s).texture, 0, s.layer as u32),
                wgpu::Extent3d {
                    width: TILE_SIZE,
                    height: TILE_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }
        dabs.len() as u32
    }
}
