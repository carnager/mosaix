//! Drawing a frame: composite tiles, selection (marching ants / quick mask),
//! overlay lines and the brush cursor.

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use imagine_core::tile::tiles_in_rect;
use imagine_core::{MaskTile, TileCoord, View};

use super::util::color_attachment;
use super::{
    CompositeEntry, Engine, FLATTEN_BUDGET, Globals, NONE, Overlay, RenderStats, ScreenMask,
};
use crate::pool::MASK_FORMAT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileInstance {
    origin: [f32; 2],
    layer: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SelInstance {
    origin: [f32; 2],
    slot: u32,
    value: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    pos: [f32; 2],
    dist: f32,
    side: f32,
}

const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.035,
    g: 0.037,
    b: 0.042,
    a: 1.0,
};

fn line_vertices(overlay: &Overlay) -> Vec<LineVertex> {
    let mut v = Vec::new();
    for line in &overlay.lines {
        let mut pts = line.points.clone();
        if line.closed && pts.len() > 2 {
            pts.push(pts[0]);
        }
        let mut dist = 0.0f32;
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let d = b - a;
            let len = d.length() as f32;
            if len < 1e-3 {
                continue;
            }
            let n = glam::DVec2::new(-d.y, d.x) / d.length() * 1.25;
            let corner = |p: glam::DVec2, side: f32, dd: f32| LineVertex {
                pos: [
                    (p.x + n.x * side as f64) as f32,
                    (p.y + n.y * side as f64) as f32,
                ],
                dist: dd,
                side,
            };
            let (a0, a1) = (corner(a, -1.0, dist), corner(a, 1.0, dist));
            let (b0, b1) = (corner(b, -1.0, dist + len), corner(b, 1.0, dist + len));
            v.extend([a0, a1, b0, b0, a1, b1]);
            dist += len;
        }
    }
    v
}

impl Engine {
    pub(super) fn rebuild_display_bgs(&mut self) {
        if self.display_gen == self.comp_pool.generation {
            return;
        }
        self.display_bgs = self
            .comp_pool
            .chunks
            .iter()
            .map(|ch| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("display"),
                    layout: &self.display_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&ch.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler_nearest),
                        },
                    ],
                })
            })
            .collect();
        self.display_gen = self.comp_pool.generation;
    }

    fn ensure_screen_mask(&mut self, w: u32, h: u32) {
        if self.screen_mask.as_ref().is_some_and(|s| s.size == (w, h)) {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screen selection mask"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MASK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("screen mask"),
            layout: &self.screen_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        self.screen_mask = Some(ScreenMask {
            view,
            bind_group,
            size: (w, h),
        });
    }

    /// Applies pending paint, re-flattens (budgeted, visible first) and draws.
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        view: &View,
        overlay: &Overlay,
    ) -> RenderStats {
        self.visible = view.visible_doc_rect();
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut stats = RenderStats::default();
        self.refresh_preview();
        self.encode_paint(&mut enc, &mut stats);
        self.update_signatures();
        self.encode_flatten(&mut enc, Some(FLATTEN_BUDGET), &mut stats);
        self.encode_mips(&mut enc);
        self.rebuild_display_bgs();
        stats.dirty_left = self.dirty.len() as u32;

        let c = view.doc_to_clip().to_cols_array_2d();
        let nearest = view.zoom >= 1.0 && view.is_axis_aligned();
        let (cursor, on) = match overlay.cursor {
            Some((p, r)) => ([p.x as f32, p.y as f32, r], 1.0),
            None => ([0.0; 3], 0.0),
        };
        let globals = Globals {
            m: [
                [c[0][0], c[0][1], c[0][2], 0.0],
                [c[1][0], c[1][1], c[1][2], 0.0],
                [c[2][0], c[2][1], c[2][2], 0.0],
            ],
            doc_view: [
                self.doc.width as f32,
                self.doc.height as f32,
                view.viewport.x as f32,
                view.viewport.y as f32,
            ],
            cursor: [cursor[0], cursor[1], cursor[2], on],
            params: [
                if nearest { 1.0 } else { 0.0 },
                overlay.checker_px.max(1.0),
                overlay.time,
                if overlay.quick_mask { 1.0 } else { 0.0 },
            ],
        };
        self.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        let (min, max) = self.visible;
        let visible: Vec<TileCoord> = tiles_in_rect(
            (min.x, min.y),
            (max.x, max.y),
            self.doc.width,
            self.doc.height,
        )
        .collect();

        // Visible composite tiles, grouped by chunk.
        let mut by_chunk: BTreeMap<u16, Vec<TileInstance>> = BTreeMap::new();
        for &c in &visible {
            if let Some(CompositeEntry { tile: Some(t), .. }) = self.comp.get(&c) {
                let (x, y) = c.origin();
                let s = t.slot();
                by_chunk.entry(s.chunk).or_default().push(TileInstance {
                    origin: [x as f32, y as f32],
                    layer: s.layer as u32,
                    _pad: 0,
                });
            }
        }
        let mut instances = Vec::new();
        let mut draws = Vec::new();
        for (chunk, list) in by_chunk {
            draws.push((chunk, instances.len() as u32, list.len() as u32));
            instances.extend(list);
        }
        stats.visible_tiles = instances.len() as u32;
        self.instance_buf
            .write(&self.device, &self.queue, bytemuck::cast_slice(&instances));

        // Selection → screen-space mask for the ants pass.
        let show_sel =
            self.shown().selection.is_some() && (overlay.show_selection || overlay.quick_mask);
        if show_sel {
            let (vw, vh) = (
                view.viewport.x.max(1.0) as u32,
                view.viewport.y.max(1.0) as u32,
            );
            self.ensure_screen_mask(vw, vh);
            self.rebuild_arrays_bg();
            let sel = self.shown().selection.as_ref().unwrap();
            let sel_instances: Vec<SelInstance> = visible
                .iter()
                .filter_map(|&c| {
                    let (x, y) = c.origin();
                    let origin = [x as f32, y as f32];
                    let solid = |v: u8| {
                        (v > 0).then_some(SelInstance {
                            origin,
                            slot: NONE,
                            value: v as f32 / 255.0,
                        })
                    };
                    match sel.tiles.get(&c) {
                        Some(MaskTile::Pixels(t)) => Some(SelInstance {
                            origin,
                            slot: t.slot().packed(),
                            value: 0.0,
                        }),
                        Some(MaskTile::Solid(v)) => solid(*v),
                        None => solid(sel.default),
                    }
                })
                .collect();
            self.sel_instance_buf.write(
                &self.device,
                &self.queue,
                bytemuck::cast_slice(&sel_instances),
            );
            let sm = self.screen_mask.as_ref().unwrap();
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("selection mask"),
                color_attachments: &[Some(color_attachment(
                    &sm.view,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                ))],
                ..Default::default()
            });
            if !sel_instances.is_empty() {
                rp.set_pipeline(&self.sel_pipeline);
                rp.set_bind_group(0, &self.globals_bg, &[]);
                rp.set_bind_group(1, self.arrays_bg.as_ref().unwrap(), &[]);
                rp.set_vertex_buffer(0, self.sel_instance_buf.buf.slice(..));
                rp.draw(0..4, 0..sel_instances.len() as u32);
            }
        }

        let lines = line_vertices(overlay);
        self.line_buf
            .write(&self.device, &self.queue, bytemuck::cast_slice(&lines));
        self.params.upload(&self.queue);

        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display"),
                color_attachments: &[Some(color_attachment(
                    target,
                    wgpu::LoadOp::Clear(BACKGROUND),
                ))],
                ..Default::default()
            });
            rp.set_bind_group(0, &self.globals_bg, &[]);
            rp.set_bind_group(1, &self.display_bgs[0], &[]);
            rp.set_pipeline(&self.checker_pipeline);
            rp.draw(0..4, 0..1);
            if !draws.is_empty() {
                rp.set_pipeline(&self.tile_pipeline);
                rp.set_vertex_buffer(0, self.instance_buf.buf.slice(..));
                for (chunk, start, count) in draws {
                    rp.set_bind_group(1, &self.display_bgs[chunk as usize], &[]);
                    rp.draw(0..4, start..start + count);
                }
            }
            if show_sel {
                rp.set_pipeline(&self.ants_pipeline);
                rp.set_bind_group(1, &self.screen_mask.as_ref().unwrap().bind_group, &[]);
                rp.draw(0..3, 0..1);
            }
            if !lines.is_empty() {
                rp.set_pipeline(&self.line_pipeline);
                rp.set_vertex_buffer(0, self.line_buf.buf.slice(..));
                rp.draw(0..lines.len() as u32, 0..1);
            }
            if overlay.cursor.is_some() {
                rp.set_pipeline(&self.cursor_pipeline);
                rp.set_bind_group(1, &self.display_bgs[0], &[]);
                rp.draw(0..4, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        stats
    }
}
