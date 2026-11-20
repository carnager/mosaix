//! The document engine: layer pixels, undo history, strokes, compositing and
//! display, all on the GPU.
//!
//! Data flow per frame:
//! 1. Brush dabs are rendered into *stroke tiles* (one per touched tile),
//!    never directly into layer tiles.
//! 2. Every document tile has a compositing program (its layer stack). Its
//!    hash ("signature") identifies the result; when it changes, or a stroke
//!    tile under it was painted, the tile is re-flattened by a compute kernel
//!    into the *composite cache* – visible tiles first, within a budget.
//! 3. Mips of re-flattened composite tiles are rebuilt; the display pass
//!    draws visible composite tiles only.
//!
//! When a stroke ends it is merged into *new* layer tiles. Old tiles stay
//! alive as long as an undo snapshot references them, so undo/redo is just
//! swapping documents.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use bytemuck::{Pod, Zeroable};
use glam::DVec2;
use half::f16;
use imagine_core::doc::hash_program;
use imagine_core::tile::tiles_in_rect;
use imagine_core::{Dab, Document, History, LayerId, Op, TILE_MIP_LEVELS, TILE_SIZE, TileCoord, View};

use crate::pool::{BYTES_PER_PIXEL, Pool, PoolConfig, SlotId, TILE_FORMAT, Tile, texel_copy};

/// Size of the layer-texture binding array (must match the WGSL shaders).
pub const MAX_LAYER_CHUNKS: u32 = 64;
const CHUNK_LAYERS: u32 = 256;
const MAX_COMPOSITE_CHUNKS: u32 = 64;
const NONE: u32 = u32::MAX;
const UNIFORM_STRIDE: u64 = 256;
const PARAM_SLOTS: usize = 4096;
const MERGE_BATCH: u32 = 64;
/// Composite tiles re-flattened per frame (≈ 50 MP); the rest follows next frames.
const FLATTEN_BUDGET: usize = 768;
const HISTORY_LIMIT: usize = 100;
const MAX_GROUP_DEPTH: usize = 7;

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
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuJob {
    dst_layer: u32,
    op_start: u32,
    op_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MergeJob {
    old_slot: u32,
    stroke_slot: u32,
    mode: u32,
    opacity: f32,
}

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
    pub paint_passes: u32,
    pub flattened: u32,
    pub dirty_left: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeParams {
    pub opacity: f32,
    pub erase: bool,
}

struct ActiveStroke {
    layer: LayerId,
    params: StrokeParams,
    /// Stroke tile and whether it still has to be cleared before first use.
    tiles: HashMap<TileCoord, (Tile, bool)>,
}

struct CompositeEntry {
    tile: Option<Tile>,
    sig: u64,
}

/// Per-dispatch parameters in a dynamic-offset uniform buffer.
struct Params {
    buf: wgpu::Buffer,
    data: Vec<[u32; 4]>,
}

impl Params {
    fn push(&mut self, v: [u32; 4]) -> u32 {
        assert!(self.data.len() < PARAM_SLOTS, "too many dispatches in one submission");
        self.data.push(v);
        ((self.data.len() - 1) as u64 * UNIFORM_STRIDE) as u32
    }

    fn upload(&mut self, queue: &wgpu::Queue) {
        if self.data.is_empty() {
            return;
        }
        let mut bytes = vec![0u8; self.data.len() * UNIFORM_STRIDE as usize];
        for (i, v) in self.data.iter().enumerate() {
            bytes[i * UNIFORM_STRIDE as usize..][..16].copy_from_slice(bytemuck::bytes_of(v));
        }
        queue.write_buffer(&self.buf, 0, &bytes);
        self.data.clear();
    }

    fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buf,
            offset: 0,
            size: wgpu::BufferSize::new(16),
        })
    }
}

/// A growable GPU buffer.
struct DynBuf {
    buf: wgpu::Buffer,
    label: &'static str,
    usage: wgpu::BufferUsages,
}

impl DynBuf {
    fn new(device: &wgpu::Device, label: &'static str, usage: wgpu::BufferUsages, size: u64) -> Self {
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(256), usage, mapped_at_creation: false });
        Self { buf, label, usage }
    }

    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        if data.len() as u64 > self.buf.size() {
            let size = (data.len() as u64).next_power_of_two();
            self.buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size,
                usage: self.usage,
                mapped_at_creation: false,
            });
        }
        if !data.is_empty() {
            queue.write_buffer(&self.buf, 0, data);
        }
    }
}

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,

    doc: Document<Tile>,
    history: History<Document<Tile>>,
    layer_pool: Pool,
    comp_pool: Pool,
    comp: HashMap<TileCoord, CompositeEntry>,
    dirty: HashSet<TileCoord>,
    recheck: HashSet<TileCoord>,
    recheck_all: bool,
    stroke: Option<ActiveStroke>,
    pending_dabs: Vec<Dab>,
    pending_mips: Vec<SlotId>,
    visible: (DVec2, DVec2),

    params: Params,
    ops_buf: DynBuf,
    jobs_buf: DynBuf,
    merge_buf: DynBuf,
    list_buf: DynBuf,
    instance_buf: DynBuf,
    dab_buf: DynBuf,
    origin_buf: wgpu::Buffer,
    origin_bg: wgpu::BindGroup,
    origin_cap: u64,
    scratch: wgpu::Texture,
    scratch_view: wgpu::TextureView,

    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    display_bgl: wgpu::BindGroupLayout,
    display_bgs: Vec<wgpu::BindGroup>,
    display_gen: u64,
    sampler_linear: wgpu::Sampler,
    sampler_nearest: wgpu::Sampler,
    brush_bgl: wgpu::BindGroupLayout,
    layers_bgl: wgpu::BindGroupLayout,
    layers_bg: Option<wgpu::BindGroup>,
    layers_gen: u64,
    flatten_bgl: wgpu::BindGroupLayout,
    merge_bgl: wgpu::BindGroupLayout,
    mip_bgl: wgpu::BindGroupLayout,

    checker_pipeline: wgpu::RenderPipeline,
    tile_pipeline: wgpu::RenderPipeline,
    cursor_pipeline: wgpu::RenderPipeline,
    brush_pipeline: wgpu::RenderPipeline,
    flatten_pipeline: wgpu::ComputePipeline,
    merge_pipeline: wgpu::ComputePipeline,
    mip_pipeline: wgpu::ComputePipeline,
}

/// Features the engine needs on top of WebGPU core.
pub fn required_features() -> wgpu::Features {
    wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
}

impl Engine {
    /// Creates an engine for a document structure without pixel data
    /// (tiles are added with [`Engine::upload_tile`]).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        structure: &Document<()>,
    ) -> Self {
        let mut doc = structure.map_tiles(&mut |_| -> Tile { unreachable!("structure has no tiles") });
        doc.fix_next_id();

        let serial = Arc::new(AtomicU64::new(1));
        let layer_pool = Pool::new(
            device,
            PoolConfig {
                label: "layer tiles",
                mip_levels: 1,
                chunk_layers: CHUNK_LAYERS,
                max_chunks: MAX_LAYER_CHUNKS,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
            },
            serial.clone(),
        );
        let comp_pool = Pool::new(
            device,
            PoolConfig {
                label: "composite tiles",
                mip_levels: TILE_MIP_LEVELS,
                chunk_layers: CHUNK_LAYERS,
                max_chunks: MAX_COMPOSITE_CHUNKS,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            },
            serial,
        );

        let blend_src = include_str!("shaders/blend.wgsl");
        let shader = |label: &str, src: String| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let display_shader = shader("display", include_str!("shaders/display.wgsl").into());
        let brush_shader = shader("brush", include_str!("shaders/brush.wgsl").into());
        let flatten_shader = shader("flatten", format!("enable wgpu_binding_array;\n{blend_src}\n{}", include_str!("shaders/flatten.wgsl")));
        let merge_shader = shader("merge", include_str!("shaders/merge.wgsl").into());
        let mip_shader = shader("mip", include_str!("shaders/mip.wgsl").into());

        use wgpu::ShaderStages as S;
        let uniform = |binding, vis, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        };
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: S::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture = |binding, vis, filterable, count| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count,
        };
        let storage_tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: S::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: TILE_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        };
        let sampler = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: S::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let bgl = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
        };
        let layer_array = NonZeroU32::new(MAX_LAYER_CHUNKS);

        let globals_bgl = bgl("globals", &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: S::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }]);
        let display_bgl = bgl("display", &[texture(0, S::FRAGMENT, true, None), sampler(1), sampler(2)]);
        let brush_bgl = bgl("brush", &[uniform(0, S::VERTEX, true)]);
        let layers_bgl = bgl("layer tiles", &[texture(0, S::COMPUTE, false, layer_array)]);
        let flatten_bgl = bgl("flatten", &[storage(0), storage(1), uniform(2, S::COMPUTE, true), storage_tex(3)]);
        let merge_bgl = bgl("merge", &[storage(0), uniform(1, S::COMPUTE, true), storage_tex(2)]);
        let mip_bgl = bgl(
            "mip",
            &[texture(0, S::COMPUTE, false, None), storage_tex(1), storage(2), uniform(3, S::COMPUTE, true)],
        );

        let layout = |label, bgls: &[&wgpu::BindGroupLayout]| {
            let bgls: Vec<_> = bgls.iter().map(|b| Some(*b)).collect();
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            })
        };
        let display_layout = layout("display", &[&globals_bgl, &display_bgl]);
        let brush_layout = layout("brush", &[&brush_bgl]);
        let compute = |label, l: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(l),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let flatten_pipeline = compute("flatten", &layout("flatten", &[&layers_bgl, &flatten_bgl]), &flatten_shader, "main");
        let merge_pipeline = compute("merge", &layout("merge", &[&layers_bgl, &merge_bgl]), &merge_shader, "merge");
        let mip_pipeline = compute("mip", &layout("mip", &[&mip_bgl]), &mip_shader, "main");

        let tile_attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32];
        let tile_vb = wgpu::VertexBufferLayout {
            array_stride: size_of::<TileInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &tile_attrs,
        };
        let dab_attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Float32, 3 => Float32x4];
        let dab_vb = wgpu::VertexBufferLayout {
            array_stride: size_of::<Dab>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &dab_attrs,
        };
        let premul = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let q = |label, l, m, e, b: &[Option<wgpu::VertexBufferLayout>], f, bl| {
            quad_pipeline(device, label, l, m, e, b, f, bl)
        };
        let checker_pipeline =
            q("checker", &display_layout, &display_shader, ("vs_checker", "fs_checker"), &[], surface_format, None);
        let tile_pipeline =
            q("tiles", &display_layout, &display_shader, ("vs_tile", "fs_tile"), &[Some(tile_vb)], surface_format, premul);
        let cursor_pipeline =
            q("cursor", &display_layout, &display_shader, ("vs_cursor", "fs_cursor"), &[], surface_format, premul);
        let brush_pipeline =
            q("brush", &brush_layout, &brush_shader, ("vs", "fs"), &[Some(dab_vb)], TILE_FORMAT, premul);

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
        let params = Params {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dispatch params"),
                size: PARAM_SLOTS as u64 * UNIFORM_STRIDE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            data: Vec::new(),
        };
        let origin_cap = 64;
        let (origin_buf, origin_bg) = make_origin_buf(device, &brush_bgl, origin_cap);
        let scratch = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("merge scratch"),
            size: wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: MERGE_BATCH },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TILE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scratch_view = scratch.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        use wgpu::BufferUsages as B;

        let mut e = Self {
            device: device.clone(),
            queue: queue.clone(),
            doc,
            history: History::new(HISTORY_LIMIT),
            layer_pool,
            comp_pool,
            comp: HashMap::new(),
            dirty: HashSet::new(),
            recheck: HashSet::new(),
            recheck_all: true,
            stroke: None,
            pending_dabs: Vec::new(),
            pending_mips: Vec::new(),
            visible: (DVec2::ZERO, DVec2::ZERO),
            params,
            ops_buf: DynBuf::new(device, "ops", B::STORAGE, 64 << 10),
            jobs_buf: DynBuf::new(device, "jobs", B::STORAGE, 16 << 10),
            merge_buf: DynBuf::new(device, "merge jobs", B::STORAGE, 16 << 10),
            list_buf: DynBuf::new(device, "mip layers", B::STORAGE, 16 << 10),
            instance_buf: DynBuf::new(device, "tile instances", B::VERTEX, 64 << 10),
            dab_buf: DynBuf::new(device, "dabs", B::VERTEX, 32 << 10),
            origin_buf,
            origin_bg,
            origin_cap,
            scratch,
            scratch_view,
            globals_buf,
            globals_bg,
            display_bgl,
            display_bgs: Vec::new(),
            display_gen: 0,
            sampler_linear,
            sampler_nearest,
            brush_bgl,
            layers_bgl,
            layers_bg: None,
            layers_gen: 0,
            flatten_bgl,
            merge_bgl,
            mip_bgl,
            checker_pipeline,
            tile_pipeline,
            cursor_pipeline,
            brush_pipeline,
            flatten_pipeline,
            merge_pipeline,
            mip_pipeline,
        };
        e.rebuild_display_bgs();
        e
    }

    // ---- document access ------------------------------------------------------

    pub fn doc(&self) -> &Document<Tile> {
        &self.doc
    }

    pub fn size(&self) -> (u32, u32) {
        (self.doc.width, self.doc.height)
    }

    /// Applies an undoable edit. `f` returns whether anything changed.
    pub fn edit(&mut self, label: &str, f: impl FnOnce(&mut Document<Tile>) -> bool) -> bool {
        if self.stroke.is_some() {
            return false;
        }
        let before = self.doc.clone();
        if f(&mut self.doc) {
            self.history.push(before, label);
            self.recheck_all = true;
            true
        } else {
            false
        }
    }

    /// Changes that are not part of the history (layer selection).
    pub fn select(&mut self, f: impl FnOnce(&mut Document<Tile>)) {
        if self.stroke.is_none() {
            f(&mut self.doc);
        }
    }

    pub fn undo(&mut self) -> Option<String> {
        if self.stroke.is_some() {
            return None;
        }
        let cur = std::mem::replace(&mut self.doc, Document::empty(0, 0));
        match self.history.undo(cur) {
            Ok((prev, label)) => {
                self.doc = prev;
                self.recheck_all = true;
                Some(label)
            }
            Err(cur) => {
                self.doc = cur;
                None
            }
        }
    }

    pub fn redo(&mut self) -> Option<String> {
        if self.stroke.is_some() {
            return None;
        }
        let cur = std::mem::replace(&mut self.doc, Document::empty(0, 0));
        match self.history.redo(cur) {
            Ok((next, label)) => {
                self.doc = next;
                self.recheck_all = true;
                Some(label)
            }
            Err(cur) => {
                self.doc = cur;
                None
            }
        }
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn history_len(&self) -> (usize, usize) {
        (self.history.undo_len(), self.history.redo_len())
    }

    /// Writes premultiplied linear RGBA f16 pixels (TILE_SIZE² × 4) into a
    /// pixel layer, outside of the undo history (used for loading).
    pub fn upload_tile(&mut self, layer: LayerId, c: TileCoord, pixels: &[f16]) -> Option<Tile> {
        assert_eq!(pixels.len(), (TILE_SIZE * TILE_SIZE * 4) as usize);
        if !self.in_document(c) {
            return None;
        }
        let tile = self.layer_pool.alloc()?;
        let slot = tile.slot();
        self.queue.write_texture(
            texel_copy(&self.layer_pool.chunk(slot).texture, 0, slot.layer as u32),
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TILE_SIZE * BYTES_PER_PIXEL),
                rows_per_image: Some(TILE_SIZE),
            },
            wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
        );
        self.doc.tiles_mut(layer)?.insert(c, tile.clone());
        self.recheck.insert(c);
        self.dirty.insert(c); // content may change under an identical signature
        Some(tile)
    }

    fn in_document(&self, c: TileCoord) -> bool {
        let (gw, gh) = imagine_core::tile::grid_size(self.doc.width, self.doc.height);
        c.x >= 0 && c.y >= 0 && (c.x as u32) < gw && (c.y as u32) < gh
    }

    // ---- strokes ----------------------------------------------------------------

    /// Starts a stroke on the active layer. Fails for groups and hidden layers.
    pub fn begin_stroke(&mut self, params: StrokeParams) -> Result<(), &'static str> {
        let l = self.doc.active_layer().ok_or("keine aktive Ebene")?;
        if l.is_group() {
            return Err("auf Gruppen kann nicht gemalt werden");
        }
        if !l.visible {
            return Err("die Ebene ist ausgeblendet");
        }
        self.stroke = Some(ActiveStroke { layer: l.id, params, tiles: HashMap::new() });
        Ok(())
    }

    pub fn stroke_active(&self) -> bool {
        self.stroke.is_some()
    }

    pub fn stroke_dabs(&mut self, dabs: &[Dab]) {
        if self.stroke.is_some() {
            self.pending_dabs.extend_from_slice(dabs);
        }
    }

    /// Bakes the stroke into new layer tiles and records an undo step.
    pub fn end_stroke(&mut self, label: &str) {
        if self.stroke.is_none() {
            return;
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut stats = RenderStats::default();
        self.encode_paint(&mut enc, &mut stats);
        let stroke = self.stroke.take().unwrap();
        let Some(old_tiles) = self.doc.find(stroke.layer).and_then(|l| l.tiles()).cloned() else {
            self.queue.submit([enc.finish()]);
            return;
        };

        let mut jobs = Vec::new();
        let mut targets = Vec::new();
        let mut coords: Vec<_> = stroke.tiles.keys().copied().collect();
        coords.sort();
        for c in coords {
            let (stroke_tile, never_painted) = &stroke.tiles[&c];
            let old = old_tiles.get(&c);
            if *never_painted || (stroke.params.erase && old.is_none()) {
                continue;
            }
            let Some(new) = self.layer_pool.alloc() else {
                tracing::warn!("Ebenen-Speicher voll – Strich unvollständig übernommen");
                break;
            };
            jobs.push(MergeJob {
                old_slot: old.map_or(NONE, |t| t.slot().packed()),
                stroke_slot: stroke_tile.slot().packed(),
                mode: if stroke.params.erase { 2 } else { 1 },
                opacity: stroke.params.opacity,
            });
            targets.push((c, new));
        }

        if !jobs.is_empty() {
            self.merge_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&jobs));
            self.rebuild_layers_bg();
            for (b, batch) in targets.chunks(MERGE_BATCH as usize).enumerate() {
                let base = b as u32 * MERGE_BATCH;
                let off = self.params.push([base, batch.len() as u32, 0, 0]);
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("merge"),
                    layout: &self.merge_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.merge_buf.buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.params.binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.scratch_view) },
                    ],
                });
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    cp.set_pipeline(&self.merge_pipeline);
                    cp.set_bind_group(0, self.layers_bg.as_ref().unwrap(), &[]);
                    cp.set_bind_group(1, &bg, &[off]);
                    cp.dispatch_workgroups(TILE_SIZE / 8, TILE_SIZE / 8, batch.len() as u32);
                }
                for (i, (_, new)) in batch.iter().enumerate() {
                    let slot = new.slot();
                    enc.copy_texture_to_texture(
                        texel_copy(&self.scratch, 0, i as u32),
                        texel_copy(&self.layer_pool.chunk(slot).texture, 0, slot.layer as u32),
                        wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
                    );
                }
            }
        }
        self.params.upload(&self.queue);
        self.queue.submit([enc.finish()]);

        if !targets.is_empty() {
            let before = self.doc.clone();
            if let Some(map) = self.doc.tiles_mut(stroke.layer) {
                for (c, new) in targets {
                    map.insert(c, new);
                }
            }
            self.history.push(before, label);
        }
        self.recheck.extend(stroke.tiles.keys().copied());
        // `stroke` (and its scratch tiles) is dropped here; slots return to the pool.
    }


    // ---- frame work ---------------------------------------------------------------

    fn encode_paint(&mut self, enc: &mut wgpu::CommandEncoder, stats: &mut RenderStats) {
        if self.pending_dabs.is_empty() {
            return;
        }
        let dabs = std::mem::take(&mut self.pending_dabs);
        let Some(stroke) = &mut self.stroke else { return };
        let (w, h) = (self.doc.width, self.doc.height);

        let mut touched = std::collections::BTreeSet::new();
        for d in &dabs {
            let (min, max) = d.bounds();
            touched.extend(tiles_in_rect(min, max, w, h));
        }
        let mut passes = Vec::new();
        for c in touched {
            if let std::collections::hash_map::Entry::Vacant(e) = stroke.tiles.entry(c) {
                let Some(t) = self.layer_pool.alloc() else {
                    tracing::warn!("Ebenen-Speicher voll");
                    continue;
                };
                e.insert((t, true));
                self.recheck.insert(c);
            }
            let (tile, needs_clear) = stroke.tiles.get_mut(&c).unwrap();
            passes.push((c, tile.slot(), std::mem::replace(needs_clear, false)));
            self.dirty.insert(c);
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
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_pipeline(&self.brush_pipeline);
            rp.set_bind_group(0, &self.origin_bg, &[(i as u64 * UNIFORM_STRIDE) as u32]);
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
        let ops = self.doc.program(c, stroke.map(|s| s.0));
        if ops.is_empty() {
            return None;
        }
        let mut h = DefaultHasher::new();
        hash_program(&ops, |t: &Tile| t.serial(), &mut h);
        (self.doc.blend_space as u32).hash(&mut h);
        if let Some((_, t, p)) = stroke {
            (t.serial(), p.opacity.to_bits(), p.erase).hash(&mut h);
        }
        Some(h.finish())
    }

    fn update_signatures(&mut self) {
        let coords: Vec<TileCoord> = if self.recheck_all {
            let mut s = self.doc.all_tile_coords();
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
            match self.signature(c) {
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

    fn encode_flatten(&mut self, enc: &mut wgpu::CommandEncoder, budget: Option<usize>, stats: &mut RenderStats) {
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

        let mut ops: Vec<GpuOp> = Vec::new();
        let mut jobs: BTreeMap<u16, Vec<GpuJob>> = BTreeMap::new();
        let mut deep_warned = false;
        for c in list {
            self.dirty.remove(&c);
            let stroke = self.stroke.as_ref().and_then(|s| s.tiles.get(&c).map(|(t, _)| (s.layer, t.slot(), s.params)));
            let program = self.doc.program(c, stroke.map(|s| s.0));
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
            let start = ops.len() as u32;
            let mut depth = 0usize;
            for op in &program {
                ops.push(match *op {
                    Op::Layer { id, tile, opacity, blend } => {
                        let st = stroke.filter(|s| s.0 == id);
                        GpuOp {
                            kind: 0,
                            slot: tile.map_or(NONE, |t| t.slot().packed()),
                            blend: blend as u32,
                            opacity,
                            stroke_slot: st.map_or(NONE, |s| s.1.packed()),
                            stroke_opacity: st.map_or(0.0, |s| s.2.opacity),
                            stroke_mode: st.map_or(0, |s| if s.2.erase { 2 } else { 1 }),
                            _pad: 0,
                        }
                    }
                    Op::Push => {
                        depth += 1;
                        if depth > MAX_GROUP_DEPTH && !deep_warned {
                            tracing::warn!("Gruppen tiefer als {MAX_GROUP_DEPTH} Ebenen werden falsch dargestellt");
                            deep_warned = true;
                        }
                        GpuOp { kind: 1, ..Zeroable::zeroed() }
                    }
                    Op::Pop { opacity, blend } => {
                        depth = depth.saturating_sub(1);
                        GpuOp { kind: 2, blend: blend as u32, opacity, ..Zeroable::zeroed() }
                    }
                });
            }
            let slot = tile.slot();
            jobs.entry(slot.chunk).or_default().push(GpuJob {
                dst_layer: slot.layer as u32,
                op_start: start,
                op_count: ops.len() as u32 - start,
                _pad: 0,
            });
            self.pending_mips.push(slot);
            stats.flattened += 1;
        }
        if jobs.is_empty() {
            return;
        }

        let mut all_jobs = Vec::new();
        let mut ranges = Vec::new();
        for (chunk, js) in jobs {
            ranges.push((chunk, all_jobs.len() as u32, js.len() as u32));
            all_jobs.extend(js);
        }
        self.ops_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&ops));
        self.jobs_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&all_jobs));
        self.rebuild_layers_bg();
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
                ],
            });
            dispatches.push((bg, off, count));
        }
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("flatten"), timestamp_writes: None });
        cp.set_pipeline(&self.flatten_pipeline);
        cp.set_bind_group(0, self.layers_bg.as_ref().unwrap(), &[]);
        for (bg, off, count) in &dispatches {
            cp.set_bind_group(1, bg, &[*off]);
            cp.dispatch_workgroups(TILE_SIZE / 8, TILE_SIZE / 8, *count);
        }
    }

    fn encode_mips(&mut self, enc: &mut wgpu::CommandEncoder) {
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

    /// The binding array over all layer chunks; rebuilt when a chunk is added.
    fn rebuild_layers_bg(&mut self) {
        if self.layers_bg.is_some() && self.layers_gen == self.layer_pool.generation {
            return;
        }
        let views = layer_views(&self.layer_pool);
        self.layers_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer tiles"),
            layout: &self.layers_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureViewArray(&views) }],
        }));
        self.layers_gen = self.layer_pool.generation;
    }

    fn rebuild_display_bgs(&mut self) {
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
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&ch.view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler_linear) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler_nearest) },
                    ],
                })
            })
            .collect();
        self.display_gen = self.comp_pool.generation;
    }

    /// True while there is work left that needs further frames.
    pub fn busy(&self) -> bool {
        !self.dirty.is_empty() || !self.pending_dabs.is_empty() || !self.recheck.is_empty() || self.recheck_all
    }

    /// Applies pending paint, re-flattens (budgeted, visible first) and draws.
    pub fn render(&mut self, target: &wgpu::TextureView, view: &View, overlay: &Overlay) -> RenderStats {
        self.visible = view.visible_doc_rect();
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut stats = RenderStats::default();
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
            doc_view: [self.doc.width as f32, self.doc.height as f32, view.viewport.x as f32, view.viewport.y as f32],
            cursor: [cursor[0], cursor[1], cursor[2], on],
            params: [if nearest { 1.0 } else { 0.0 }, overlay.checker_px.max(1.0), 0.0, 0.0],
        };
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        // Visible composite tiles, grouped by chunk.
        let (min, max) = self.visible;
        let mut by_chunk: BTreeMap<u16, Vec<TileInstance>> = BTreeMap::new();
        for c in tiles_in_rect((min.x, min.y), (max.x, max.y), self.doc.width, self.doc.height) {
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
        self.instance_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&instances));
        self.params.upload(&self.queue);

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
            if overlay.cursor.is_some() {
                rp.set_pipeline(&self.cursor_pipeline);
                rp.draw(0..4, 0..1);
            }
        }
        self.queue.submit([enc.finish()]);
        stats
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

    // ---- readback & stats ---------------------------------------------------------

    /// Reads mip 0 of layer tiles (blocking).
    pub fn read_layer_tiles(&self, tiles: &[Tile]) -> Vec<Vec<f16>> {
        let slots: Vec<SlotId> = tiles.iter().map(Tile::slot).collect();
        read_slots(&self.device, &self.queue, &self.layer_pool, &slots)
    }

    /// Coordinates that currently have composited content.
    pub fn composite_coords(&self) -> Vec<TileCoord> {
        let mut v: Vec<_> = self.comp.iter().filter(|(_, e)| e.tile.is_some()).map(|(c, _)| *c).collect();
        v.sort();
        v
    }

    /// Reads the flattened image of the given tiles (call [`Engine::flush`] first).
    pub fn read_composite(&self, coords: &[TileCoord]) -> Vec<Vec<f16>> {
        let n = (TILE_SIZE * TILE_SIZE * 4) as usize;
        let slots: Vec<Option<SlotId>> =
            coords.iter().map(|c| self.comp.get(c).and_then(|e| e.tile.as_ref()).map(Tile::slot)).collect();
        let present: Vec<SlotId> = slots.iter().flatten().copied().collect();
        let mut data = read_slots(&self.device, &self.queue, &self.comp_pool, &present).into_iter();
        slots.iter().map(|s| if s.is_some() { data.next().unwrap() } else { vec![f16::ZERO; n] }).collect()
    }

    pub fn layer_tiles_live(&self) -> usize {
        self.layer_pool.live()
    }

    pub fn composite_tiles_live(&self) -> usize {
        self.comp_pool.live()
    }

    pub fn gpu_bytes(&self) -> u64 {
        self.layer_pool.allocated_bytes() + self.comp_pool.allocated_bytes()
    }

    pub fn max_layer_tiles(&self) -> u64 {
        self.layer_pool.capacity_tiles()
    }
}

/// All layer chunk views, padded to the binding array size.
fn layer_views(pool: &Pool) -> Vec<&wgpu::TextureView> {
    let chunks = &pool.chunks;
    (0..MAX_LAYER_CHUNKS as usize).map(|i| &chunks.get(i).unwrap_or(&chunks[0]).view).collect()
}

/// Copies mip 0 of the given slots to the CPU (batched, blocking).
fn read_slots(device: &wgpu::Device, queue: &wgpu::Queue, pool: &Pool, slots: &[SlotId]) -> Vec<Vec<f16>> {
    let tile_bytes = (TILE_SIZE * TILE_SIZE * BYTES_PER_PIXEL) as u64;
    let mut out = Vec::with_capacity(slots.len());
    for batch in slots.chunks(256) {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: tile_bytes * batch.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        for (i, s) in batch.iter().enumerate() {
            enc.copy_texture_to_buffer(
                texel_copy(&pool.chunk(*s).texture, 0, s.layer as u32),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: i as u64 * tile_bytes,
                        bytes_per_row: Some(TILE_SIZE * BYTES_PER_PIXEL),
                        rows_per_image: Some(TILE_SIZE),
                    },
                },
                wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
            );
        }
        queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = buf.slice(..).get_mapped_range().expect("mapped range");
        for chunk in data.chunks_exact(tile_bytes as usize) {
            out.push(bytemuck::cast_slice::<u8, f16>(chunk).to_vec());
        }
    }
    out
}

fn make_origin_buf(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, slots: u64) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile origins"),
        size: slots * UNIFORM_STRIDE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
        vertex: wgpu::VertexState { module, entry_point: Some(vs), compilation_options: Default::default(), buffers },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
