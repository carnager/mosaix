//! Sparse tiled canvas living on the GPU.
//!
//! All tiles share one `Rgba16Float` 2D-array texture (one layer per tile,
//! full mip chain per tile). The array grows on demand by doubling. Painting
//! renders dab quads straight into the tile layers with hardware blending;
//! a compute pass then rebuilds the mips of every touched tile, and the
//! display pass draws only the visible, allocated tiles.

use std::collections::{BTreeSet, HashMap};

use bytemuck::{Pod, Zeroable};
use glam::DVec2;
use half::f16;
use imagine_core::{Dab, TILE_MIP_LEVELS, TILE_SIZE, TileCoord, View, tile::tiles_in_rect};
use wgpu::util::DeviceExt;

const TILE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BYTES_PER_PIXEL: u32 = 8;
const INITIAL_CAPACITY: u32 = 64;
const UNIFORM_STRIDE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileInstance {
    origin: [f32; 2],
    layer: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    m: [[f32; 4]; 3],
    doc_view: [f32; 4],
    cursor: [f32; 4],
    params: [f32; 4],
}

/// Things drawn on top of the document.
#[derive(Debug, Clone, Copy, Default)]
pub struct Overlay {
    /// Brush ring: screen position (physical px) and radius in screen px.
    pub cursor: Option<(DVec2, f32)>,
    /// Size of checkerboard squares in physical px.
    pub checker_px: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub visible_tiles: u32,
    pub dabs: u32,
    pub tile_paint_passes: u32,
    pub mip_updates: u32,
}

/// The texture array plus everything that must be rebuilt when it grows.
struct Pool {
    texture: wgpu::Texture,
    capacity: u32,
    display_bg: wgpu::BindGroup,
    mip_bgs: Vec<wgpu::BindGroup>,
    layer_targets: Vec<Option<wgpu::TextureView>>,
    instance_buf: wgpu::Buffer,
    layer_list_buf: wgpu::Buffer,
}

struct DabBatch {
    eraser: bool,
    dabs: Vec<Dab>,
}

pub struct Canvas {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    max_layers: u32,

    tiles: HashMap<TileCoord, u32>,
    next_layer: u32,
    pool: Pool,

    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    tiles_bgl: wgpu::BindGroupLayout,
    mip_bgl: wgpu::BindGroupLayout,
    sampler_linear: wgpu::Sampler,
    sampler_nearest: wgpu::Sampler,

    checker_pipeline: wgpu::RenderPipeline,
    tile_pipeline: wgpu::RenderPipeline,
    cursor_pipeline: wgpu::RenderPipeline,
    brush_pipeline: wgpu::RenderPipeline,
    eraser_pipeline: wgpu::RenderPipeline,
    mip_pipeline: wgpu::ComputePipeline,

    brush_bgl: wgpu::BindGroupLayout,
    origin_buf: wgpu::Buffer,
    origin_bg: wgpu::BindGroup,
    origin_cap: u64,
    dab_buf: wgpu::Buffer,
    dab_cap: u64,

    pending: Vec<DabBatch>,
    dirty_mips: BTreeSet<u32>,
}

impl Canvas {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let max_layers = device.limits().max_texture_array_layers;

        let display_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/display.wgsl"));
        let brush_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/brush.wgsl"));
        let mip_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/mip.wgsl"));

        // ---- display ----------------------------------------------------
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let tiles_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tiles"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                sampler_entry(1),
                sampler_entry(2),
            ],
        });
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() }],
        });
        let display_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display"),
            bind_group_layouts: &[Some(&globals_bgl), Some(&tiles_bgl)],
            immediate_size: 0,
        });

        let tile_attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32];
        let tile_vb = wgpu::VertexBufferLayout {
            array_stride: size_of::<TileInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &tile_attrs,
        };
        let premul = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let checker_pipeline = quad_pipeline(
            device,
            "checker",
            &display_layout,
            &display_shader,
            ("vs_checker", "fs_checker"),
            &[],
            surface_format,
            None,
        );
        let tile_pipeline = quad_pipeline(
            device,
            "tiles",
            &display_layout,
            &display_shader,
            ("vs_tile", "fs_tile"),
            &[Some(tile_vb)],
            surface_format,
            premul,
        );
        let cursor_pipeline = quad_pipeline(
            device,
            "cursor",
            &display_layout,
            &display_shader,
            ("vs_cursor", "fs_cursor"),
            &[],
            surface_format,
            premul,
        );

        // ---- brush ------------------------------------------------------
        let brush_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brush"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        });
        let brush_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush"),
            bind_group_layouts: &[Some(&brush_bgl)],
            immediate_size: 0,
        });
        let dab_attrs = wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32, 2 => Float32, 3 => Float32x4
        ];
        let dab_vb = wgpu::VertexBufferLayout {
            array_stride: size_of::<Dab>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &dab_attrs,
        };
        let over = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
        let erase_comp = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let erase = wgpu::BlendState { color: erase_comp, alpha: erase_comp };
        let brush_pipeline = quad_pipeline(
            device,
            "brush",
            &brush_layout,
            &brush_shader,
            ("vs", "fs"),
            &[Some(dab_vb.clone())],
            TILE_FORMAT,
            Some(over),
        );
        let eraser_pipeline = quad_pipeline(
            device,
            "eraser",
            &brush_layout,
            &brush_shader,
            ("vs", "fs"),
            &[Some(dab_vb)],
            TILE_FORMAT,
            Some(erase),
        );
        let origin_cap = 64;
        let (origin_buf, origin_bg) = make_origin_buf(device, &brush_bgl, origin_cap);
        let dab_cap = 1024;
        let dab_buf = make_vertex_buf(device, "dabs", dab_cap * size_of::<Dab>() as u64);

        // ---- mips -------------------------------------------------------
        let mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mip"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TILE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let mip_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mip"),
            bind_group_layouts: &[Some(&mip_bgl)],
            immediate_size: 0,
        });
        let mip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mip"),
            layout: Some(&mip_layout),
            module: &mip_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let pool = Pool::new(
            device,
            &tiles_bgl,
            &mip_bgl,
            &sampler_linear,
            &sampler_nearest,
            INITIAL_CAPACITY.min(max_layers),
        );

        Self {
            device: device.clone(),
            queue: queue.clone(),
            width,
            height,
            max_layers,
            tiles: HashMap::new(),
            next_layer: 0,
            pool,
            globals_buf,
            globals_bg,
            tiles_bgl,
            mip_bgl,
            sampler_linear,
            sampler_nearest,
            checker_pipeline,
            tile_pipeline,
            cursor_pipeline,
            brush_pipeline,
            eraser_pipeline,
            mip_pipeline,
            brush_bgl,
            origin_buf,
            origin_bg,
            origin_cap,
            dab_buf,
            dab_cap,
            pending: Vec::new(),
            dirty_mips: BTreeSet::new(),
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Maximum number of tiles this device can hold (one array layer each).
    pub fn max_tiles(&self) -> u32 {
        self.max_layers
    }

    /// GPU memory used by the tile array in bytes.
    pub fn gpu_bytes(&self) -> u64 {
        let per_tile: u64 = (0..TILE_MIP_LEVELS)
            .map(|m| ((TILE_SIZE >> m) as u64).pow(2) * BYTES_PER_PIXEL as u64)
            .sum();
        per_tile * self.pool.capacity as u64
    }

    /// Makes sure the array can hold `n` tiles in total without growing again.
    pub fn reserve(&mut self, n: u32) {
        let need = n.min(self.max_layers);
        if need > self.pool.capacity {
            self.grow_to(need);
        }
    }

    fn grow_to(&mut self, new_cap: u32) {
        let new_cap = new_cap.min(self.max_layers);
        let old = self.pool.capacity;
        tracing::debug!(old, new_cap, "growing tile array");
        let pool = Pool::new(
            &self.device,
            &self.tiles_bgl,
            &self.mip_bgl,
            &self.sampler_linear,
            &self.sampler_nearest,
            new_cap,
        );
        if self.next_layer > 0 {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for mip in 0..TILE_MIP_LEVELS {
                let s = TILE_SIZE >> mip;
                enc.copy_texture_to_texture(
                    texel_copy(&self.pool.texture, mip, 0),
                    texel_copy(&pool.texture, mip, 0),
                    wgpu::Extent3d { width: s, height: s, depth_or_array_layers: self.next_layer },
                );
            }
            self.queue.submit([enc.finish()]);
        }
        self.pool = pool;
    }

    /// Returns the array layer of a tile, allocating it if needed.
    /// `None` if the tile lies outside the document or the device is full.
    pub fn ensure_tile(&mut self, c: TileCoord) -> Option<u32> {
        if let Some(&l) = self.tiles.get(&c) {
            return Some(l);
        }
        let (gw, gh) = imagine_core::tile::grid_size(self.width, self.height);
        if c.x < 0 || c.y < 0 || c.x as u32 >= gw || c.y as u32 >= gh {
            return None;
        }
        if self.next_layer >= self.pool.capacity {
            if self.pool.capacity >= self.max_layers {
                tracing::warn!("tile array full ({} tiles)", self.max_layers);
                return None;
            }
            self.grow_to(self.pool.capacity * 2);
        }
        let l = self.next_layer;
        self.next_layer += 1;
        self.tiles.insert(c, l);
        Some(l)
    }

    /// Uploads premultiplied linear RGBA f16 pixels (TILE_SIZE² × 4) for one tile.
    pub fn upload_tile(&mut self, c: TileCoord, pixels: &[f16]) -> bool {
        assert_eq!(pixels.len(), (TILE_SIZE * TILE_SIZE * 4) as usize);
        let Some(layer) = self.ensure_tile(c) else { return false };
        self.queue.write_texture(
            texel_copy(&self.pool.texture, 0, layer),
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TILE_SIZE * BYTES_PER_PIXEL),
                rows_per_image: Some(TILE_SIZE),
            },
            wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
        );
        self.dirty_mips.insert(layer);
        true
    }

    /// Queues dabs to be painted with the next `render`/`flush`.
    pub fn paint(&mut self, dabs: &[Dab], eraser: bool) {
        if dabs.is_empty() {
            return;
        }
        match self.pending.last_mut() {
            Some(b) if b.eraser == eraser => b.dabs.extend_from_slice(dabs),
            _ => self.pending.push(DabBatch { eraser, dabs: dabs.to_vec() }),
        }
    }

    pub fn has_pending_work(&self) -> bool {
        !self.pending.is_empty() || !self.dirty_mips.is_empty()
    }

    /// Encodes pending paint + mip work and submits it without drawing.
    pub fn flush(&mut self) -> RenderStats {
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut stats = RenderStats::default();
        self.encode_paint(&mut enc, &mut stats);
        self.encode_mips(&mut enc, &mut stats);
        self.queue.submit([enc.finish()]);
        stats
    }

    fn encode_paint(&mut self, enc: &mut wgpu::CommandEncoder, stats: &mut RenderStats) {
        if self.pending.is_empty() {
            return;
        }
        let batches = std::mem::take(&mut self.pending);

        // Work out (batch, tile) passes first; this may allocate tiles.
        let mut passes: Vec<(usize, u32, TileCoord)> = Vec::new();
        for (bi, batch) in batches.iter().enumerate() {
            let mut touched = BTreeSet::new();
            for d in &batch.dabs {
                let (min, max) = d.bounds();
                touched.extend(tiles_in_rect(min, max, self.width, self.height));
            }
            for c in touched {
                let layer = if batch.eraser {
                    // Erasing an unallocated tile is a no-op.
                    self.tiles.get(&c).copied()
                } else {
                    self.ensure_tile(c)
                };
                if let Some(l) = layer {
                    passes.push((bi, l, c));
                }
            }
        }

        // Upload dabs (all batches back to back) and per-pass tile origins.
        let all: Vec<Dab> = batches.iter().flat_map(|b| b.dabs.iter().copied()).collect();
        if all.len() as u64 > self.dab_cap {
            self.dab_cap = (all.len() as u64).next_power_of_two();
            self.dab_buf =
                make_vertex_buf(&self.device, "dabs", self.dab_cap * size_of::<Dab>() as u64);
        }
        self.queue.write_buffer(&self.dab_buf, 0, bytemuck::cast_slice(&all));

        if passes.len() as u64 > self.origin_cap {
            self.origin_cap = (passes.len() as u64).next_power_of_two();
            (self.origin_buf, self.origin_bg) =
                make_origin_buf(&self.device, &self.brush_bgl, self.origin_cap);
        }
        let mut origins = vec![0u8; passes.len() * UNIFORM_STRIDE as usize];
        for (i, (_, _, c)) in passes.iter().enumerate() {
            let (x, y) = c.origin();
            let v: [f32; 4] = [x as f32, y as f32, 0.0, 0.0];
            origins[i * UNIFORM_STRIDE as usize..][..16].copy_from_slice(bytemuck::bytes_of(&v));
        }
        self.queue.write_buffer(&self.origin_buf, 0, &origins);

        let mut starts = Vec::with_capacity(batches.len());
        let mut acc = 0u32;
        for b in &batches {
            starts.push(acc);
            acc += b.dabs.len() as u32;
        }

        for (i, &(bi, layer, _)) in passes.iter().enumerate() {
            let target = self.pool.layer_target(layer).clone();
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            let batch = &batches[bi];
            rp.set_pipeline(if batch.eraser { &self.eraser_pipeline } else { &self.brush_pipeline });
            rp.set_bind_group(0, &self.origin_bg, &[(i as u64 * UNIFORM_STRIDE) as u32]);
            rp.set_vertex_buffer(0, self.dab_buf.slice(..));
            let s = starts[bi];
            rp.draw(0..4, s..s + batch.dabs.len() as u32);
            self.dirty_mips.insert(layer);
        }
        stats.dabs = all.len() as u32;
        stats.tile_paint_passes = passes.len() as u32;
    }

    fn encode_mips(&mut self, enc: &mut wgpu::CommandEncoder, stats: &mut RenderStats) {
        if self.dirty_mips.is_empty() {
            return;
        }
        let layers: Vec<u32> = std::mem::take(&mut self.dirty_mips).into_iter().collect();
        self.queue.write_buffer(&self.pool.layer_list_buf, 0, bytemuck::cast_slice(&layers));
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mips"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&self.mip_pipeline);
        for mip in 1..TILE_MIP_LEVELS {
            let s = TILE_SIZE >> mip;
            cp.set_bind_group(0, &self.pool.mip_bgs[mip as usize - 1], &[]);
            cp.dispatch_workgroups(s.div_ceil(8), s.div_ceil(8), layers.len() as u32);
        }
        stats.mip_updates = layers.len() as u32;
    }

    /// Applies pending paint, rebuilds dirty mips and draws the view into `target`.
    pub fn render(&mut self, target: &wgpu::TextureView, view: &View, overlay: &Overlay) -> RenderStats {
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut stats = RenderStats::default();
        self.encode_paint(&mut enc, &mut stats);
        self.encode_mips(&mut enc, &mut stats);

        let m = view.doc_to_clip();
        let c = m.to_cols_array_2d();
        let nearest = view.zoom >= 1.0 && view.is_axis_aligned();
        let (cursor, cursor_on) = match overlay.cursor {
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
                self.width as f32,
                self.height as f32,
                view.viewport.x as f32,
                view.viewport.y as f32,
            ],
            cursor: [cursor[0], cursor[1], cursor[2], cursor_on],
            params: [if nearest { 1.0 } else { 0.0 }, overlay.checker_px.max(1.0), 0.0, 0.0],
        };
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        // Cull to the visible part of the document.
        let (min, max) = view.visible_doc_rect();
        let instances: Vec<TileInstance> =
            tiles_in_rect((min.x, min.y), (max.x, max.y), self.width, self.height)
                .filter_map(|c| {
                    self.tiles.get(&c).map(|&layer| {
                        let (x, y) = c.origin();
                        TileInstance { origin: [x as f32, y as f32], layer, _pad: 0 }
                    })
                })
                .collect();
        self.queue.write_buffer(&self.pool.instance_buf, 0, bytemuck::cast_slice(&instances));
        stats.visible_tiles = instances.len() as u32;

        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("display"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.035, g: 0.037, b: 0.042, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_bind_group(0, &self.globals_bg, &[]);
            rp.set_bind_group(1, &self.pool.display_bg, &[]);
            rp.set_pipeline(&self.checker_pipeline);
            rp.draw(0..4, 0..1);
            if !instances.is_empty() {
                rp.set_pipeline(&self.tile_pipeline);
                rp.set_vertex_buffer(0, self.pool.instance_buf.slice(..));
                rp.draw(0..4, 0..instances.len() as u32);
            }
            if overlay.cursor.is_some() {
                rp.set_pipeline(&self.cursor_pipeline);
                rp.draw(0..4, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        stats
    }

    /// Reads back mip 0 of a tile (blocking). Unallocated tiles read as transparent.
    pub fn read_tile(&mut self, c: TileCoord) -> Vec<f16> {
        let n = (TILE_SIZE * TILE_SIZE * 4) as usize;
        let Some(&layer) = self.tiles.get(&c) else { return vec![f16::ZERO; n] };
        self.flush();
        let size = (TILE_SIZE * TILE_SIZE * BYTES_PER_PIXEL) as u64;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            texel_copy(&self.pool.texture, 0, layer),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(TILE_SIZE * BYTES_PER_PIXEL),
                    rows_per_image: Some(TILE_SIZE),
                },
            },
            wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map tile"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = buf.slice(..).get_mapped_range().expect("mapped range");
        bytemuck::cast_slice::<u8, f16>(&data).to_vec()
    }
}

impl Pool {
    fn new(
        device: &wgpu::Device,
        tiles_bgl: &wgpu::BindGroupLayout,
        mip_bgl: &wgpu::BindGroupLayout,
        linear: &wgpu::Sampler,
        nearest: &wgpu::Sampler,
        capacity: u32,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tile array"),
            size: wgpu::Extent3d {
                width: TILE_SIZE,
                height: TILE_SIZE,
                depth_or_array_layers: capacity,
            },
            mip_level_count: TILE_MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TILE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let array_view = |base_mip: u32, count: Option<u32>| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: base_mip,
                mip_level_count: count,
                ..Default::default()
            })
        };
        let display_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiles"),
            layout: tiles_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&array_view(0, None)),
                },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(linear) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(nearest) },
            ],
        });
        let layer_list_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dirty layers"),
            size: capacity as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mip_bgs = (1..TILE_MIP_LEVELS)
            .map(|mip| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mip"),
                    layout: mip_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&array_view(mip - 1, Some(1))),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&array_view(mip, Some(1))),
                        },
                        wgpu::BindGroupEntry { binding: 2, resource: layer_list_buf.as_entire_binding() },
                    ],
                })
            })
            .collect();
        let instance_buf =
            make_vertex_buf(device, "tile instances", capacity as u64 * size_of::<TileInstance>() as u64);
        Self {
            texture,
            capacity,
            display_bg,
            mip_bgs,
            layer_targets: vec![None; capacity as usize],
            instance_buf,
            layer_list_buf,
        }
    }

    /// Render-target view of mip 0 of one layer (cached).
    fn layer_target(&mut self, layer: u32) -> &wgpu::TextureView {
        let texture = &self.texture;
        self.layer_targets[layer as usize].get_or_insert_with(|| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("tile target"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
    }
}

fn texel_copy(texture: &wgpu::Texture, mip: u32, layer: u32) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: mip,
        origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
        aspect: wgpu::TextureAspect::All,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn make_vertex_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_origin_buf(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    slots: u64,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tile origins"),
        contents: &vec![0u8; (slots * UNIFORM_STRIDE) as usize],
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile origins"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buf,
                offset: 0,
                size: wgpu::BufferSize::new(16),
            }),
        }],
    });
    (buf, bg)
}

#[allow(clippy::too_many_arguments)]
fn quad_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    (vs, fs): (&str, &str),
    buffers: &[Option<wgpu::VertexBufferLayout>],
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
