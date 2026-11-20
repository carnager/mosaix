//! The document engine: layer pixels, masks, selection, undo history,
//! strokes, compositing and display – all on the GPU.
//!
//! Data flow per frame:
//! 1. Brush dabs are rendered into *stroke tiles* (one per touched tile),
//!    never directly into layer or mask tiles.
//! 2. Every document tile has a compositing program (its layer stack with
//!    masks). Its hash ("signature") identifies the result; when it changes, or
//!    a stroke tile under it was painted, the tile is re-flattened into the
//!    *composite cache* – visible tiles first, within a budget.
//! 3. Mips of re-flattened composite tiles are rebuilt; the display pass
//!    draws visible composite tiles, then selection ants and overlays.
//!
//! Operations that produce new pixels (committing a stroke, fill, gradient,
//! clear, move, baking a mask) go through the *apply* kernel, which renders
//! *new* tiles. Old tiles stay alive as long as an undo snapshot references
//! them, so undo/redo is just swapping documents.
//!
//! Modules: `frame` (paint, signatures, flatten, mips), `apply` (tile
//! producing operations), `selection` (masks ↔ CPU), `display` (drawing),
//! `readback` (GPU → CPU).

mod apply;
mod display;
mod frame;
mod magic;
mod readback;
mod selection;
mod smudge;
mod thumbs;
mod tips;
mod transform;
mod util;

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use bytemuck::{Pod, Zeroable};
use glam::DVec2;
use half::f16;
use imagine_core::{Dab, Document, History, LayerId, MaskRef, TILE_MIP_LEVELS, TILE_SIZE, TileCoord};

pub use apply::GradientKind;
pub use magic::SampleFrom;
pub use readback::{PoolKind, Readback};
pub use thumbs::THUMB;
pub use transform::{Interp, is_integer_translation};

use crate::pool::{BYTES_PER_PIXEL, MASK_FORMAT, Pool, PoolConfig, TILE_FORMAT, Tile, texel_copy};
use util::{DynBuf, Params, PipelineSpec, compute_pipeline, make_origin_buf, render_pipeline};

/// Size of the layer/mask texture binding arrays (must match the WGSL shaders).
pub const MAX_LAYER_CHUNKS: u32 = 64;
pub const MAX_MASK_CHUNKS: u32 = 64;
const CHUNK_LAYERS: u32 = 256;
pub(super) const MAX_COMPOSITE_CHUNKS: u32 = 64;
const NONE: u32 = u32::MAX;
const UNIFORM_STRIDE: u64 = 256;
const PARAM_SLOTS: usize = 4096;
/// Tiles produced per apply batch (size of the scratch arrays).
const SCRATCH_LAYERS: u32 = 64;
/// Composite tiles re-flattened per frame (≈ 50 MP); the rest follows next frames.
const FLATTEN_BUDGET: usize = 768;
const HISTORY_LIMIT: usize = 100;
const MAX_GROUP_DEPTH: usize = 7;

/// Things drawn on top of the document.
#[derive(Debug, Clone, Default)]
pub struct Overlay {
    /// Brush ring: screen position (physical px) and radius in screen px.
    pub cursor: Option<(DVec2, f32)>,
    /// Size of checkerboard squares in physical px.
    pub checker_px: f32,
    /// Dashed lines in screen px (selection rubber bands, lasso, gradient).
    pub lines: Vec<OverlayLine>,
    /// Draw marching ants around the selection.
    pub show_selection: bool,
    /// Tint unselected areas red.
    pub quick_mask: bool,
    /// Seconds, drives the ant animation.
    pub time: f32,
}

#[derive(Debug, Clone, Default)]
pub struct OverlayLine {
    pub points: Vec<DVec2>,
    pub closed: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub visible_tiles: u32,
    pub dabs: u32,
    pub paint_passes: u32,
    pub flattened: u32,
    pub dirty_left: u32,
}

/// What a stroke (or fill) modifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StrokeTarget {
    #[default]
    Pixels,
    /// The active layer's mask.
    Mask,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeParams {
    pub opacity: f32,
    pub erase: bool,
    pub target: StrokeTarget,
    /// Smudge (drag colour) instead of painting; dab alpha = strength.
    pub smudge: bool,
}

impl Default for StrokeParams {
    fn default() -> Self {
        Self { opacity: 1.0, erase: false, target: StrokeTarget::Pixels, smudge: false }
    }
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    m: [[f32; 4]; 3],
    doc_view: [f32; 4],
    cursor: [f32; 4],
    params: [f32; 4],
}

/// Viewport-sized mask the selection is rendered into for the ants pass.
struct ScreenMask {
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,

    doc: Document<Tile>,
    history: History<Document<Tile>>,
    layer_pool: Pool,
    mask_pool: Pool,
    comp_pool: Pool,
    comp: HashMap<TileCoord, CompositeEntry>,
    dirty: HashSet<TileCoord>,
    recheck: HashSet<TileCoord>,
    recheck_all: bool,
    stroke: Option<ActiveStroke>,
    preview: Option<transform::Preview>,
    merge_key: Option<String>,
    thumbs: Option<thumbs::Thumbs>,
    pending_dabs: Vec<Dab>,
    pending_mips: Vec<crate::pool::SlotId>,
    visible: (DVec2, DVec2),

    params: Params,
    ops_buf: DynBuf,
    jobs_buf: DynBuf,
    apply_buf: DynBuf,
    grid_buf: DynBuf,
    bounds_jobs: DynBuf,
    list_buf: DynBuf,
    instance_buf: DynBuf,
    sel_instance_buf: DynBuf,
    line_buf: DynBuf,
    dab_buf: DynBuf,
    origin_buf: wgpu::Buffer,
    origin_bg: wgpu::BindGroup,
    origin_cap: u64,
    scratch_rgba: wgpu::Texture,
    scratch_rgba_views: Vec<wgpu::TextureView>,
    scratch_mask: wgpu::Texture,
    scratch_mask_views: Vec<wgpu::TextureView>,
    screen_mask: Option<ScreenMask>,

    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    display_bgl: wgpu::BindGroupLayout,
    display_bgs: Vec<wgpu::BindGroup>,
    display_gen: u64,
    sampler_linear: wgpu::Sampler,
    sampler_nearest: wgpu::Sampler,
    brush_bgl: wgpu::BindGroupLayout,
    tips_bg: wgpu::BindGroup,
    tips_view: wgpu::TextureView,
    smudge_tex_bgl: wgpu::BindGroupLayout,
    smudge_buffers: Option<smudge::SmudgeBuffers>,
    smudge_jobs: DynBuf,
    smudge_primed: bool,
    smudge_cur: usize,
    smudge_pickup_pipeline: wgpu::RenderPipeline,
    smudge_write_pipeline: wgpu::RenderPipeline,
    arrays_bgl: wgpu::BindGroupLayout,
    arrays_bg: Option<wgpu::BindGroup>,
    arrays_gen: (u64, u64),
    flatten_bgl: wgpu::BindGroupLayout,
    apply_bgl: wgpu::BindGroupLayout,
    warp_bgl: wgpu::BindGroupLayout,
    comp_array_bgl: wgpu::BindGroupLayout,
    bounds_bgl: wgpu::BindGroupLayout,
    mip_bgl: wgpu::BindGroupLayout,
    screen_bgl: wgpu::BindGroupLayout,

    checker_pipeline: wgpu::RenderPipeline,
    tile_pipeline: wgpu::RenderPipeline,
    cursor_pipeline: wgpu::RenderPipeline,
    brush_pipeline: wgpu::RenderPipeline,
    sel_pipeline: wgpu::RenderPipeline,
    ants_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    apply_rgba_pipeline: wgpu::RenderPipeline,
    apply_mask_pipeline: wgpu::RenderPipeline,
    warp_rgba_pipeline: wgpu::RenderPipeline,
    warp_mask_pipeline: wgpu::RenderPipeline,
    bounds_pipeline: wgpu::ComputePipeline,
    similar_pipeline: wgpu::RenderPipeline,
    flatten_pipeline: wgpu::ComputePipeline,
    mip_pipeline: wgpu::ComputePipeline,
}

/// Features the engine needs on top of WebGPU core.
pub fn required_features() -> wgpu::Features {
    wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
}

/// Shader-side reference to a mask value: a tile slot, or a constant.
fn mask_slot(m: Option<MaskRef<'_, Tile>>) -> (u32, f32) {
    match m {
        None => (NONE, 1.0),
        Some(MaskRef::Solid(v)) => (NONE, v as f32 / 255.0),
        Some(MaskRef::Tile(t)) => (t.slot().packed(), 0.0),
    }
}

impl Engine {
    /// Creates an engine for a document structure without pixel data
    /// (tiles are added with [`Engine::upload_tile`] / [`Engine::load_mask`]).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        structure: &Document<()>,
    ) -> Self {
        let mut doc = structure.map_tiles(&mut |_| -> Tile { unreachable!("structure has no tiles") });
        doc.fix_next_id();

        use wgpu::TextureUsages as U;
        let serial = Arc::new(AtomicU64::new(1));
        let pool = |label, format, mip_levels, max_chunks, usage| {
            Pool::new(
                device,
                PoolConfig { label, format, mip_levels, chunk_layers: CHUNK_LAYERS, max_chunks, usage },
                serial.clone(),
            )
        };
        let layer_pool = pool(
            "layer tiles",
            TILE_FORMAT,
            1,
            MAX_LAYER_CHUNKS,
            U::TEXTURE_BINDING | U::RENDER_ATTACHMENT | U::COPY_SRC | U::COPY_DST,
        );
        let mask_pool = pool(
            "mask tiles",
            MASK_FORMAT,
            1,
            MAX_MASK_CHUNKS,
            U::TEXTURE_BINDING | U::RENDER_ATTACHMENT | U::COPY_SRC | U::COPY_DST,
        );
        let comp_pool = pool(
            "composite tiles",
            TILE_FORMAT,
            TILE_MIP_LEVELS,
            MAX_COMPOSITE_CHUNKS,
            U::TEXTURE_BINDING | U::STORAGE_BINDING | U::COPY_SRC,
        );

        let blend_src = include_str!("../shaders/blend.wgsl");
        let shader = |label: &str, src: String| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let display_shader = shader("display", include_str!("../shaders/display.wgsl").into());
        let overlay_shader = shader("overlay", include_str!("../shaders/overlay.wgsl").into());
        let tip_src = include_str!("../shaders/tip.wgsl");
        let brush_shader = shader("brush", format!("{}\n{tip_src}", include_str!("../shaders/brush.wgsl")));
        let flatten_shader = shader(
            "flatten",
            format!("enable wgpu_binding_array;\n{blend_src}\n{}", include_str!("../shaders/flatten.wgsl")),
        );
        let apply_shader = shader("apply", include_str!("../shaders/apply.wgsl").into());
        let mip_shader = shader("mip", include_str!("../shaders/mip.wgsl").into());
        let smudge_shader = shader("smudge", format!("{}\n{tip_src}", include_str!("../shaders/smudge.wgsl")));
        let warp_shader = shader("warp", include_str!("../shaders/warp.wgsl").into());
        let bounds_shader = shader("bounds", include_str!("../shaders/bounds.wgsl").into());
        let similar_shader = shader("similar", include_str!("../shaders/similar.wgsl").into());

        // ---- bind group layouts ---------------------------------------------
        use wgpu::ShaderStages as S;
        let buffer = |binding, vis, ty, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: dynamic,
                min_binding_size: if dynamic { wgpu::BufferSize::new(16) } else { None },
            },
            count: None,
        };
        let uniform = wgpu::BufferBindingType::Uniform;
        let storage = wgpu::BufferBindingType::Storage { read_only: true };
        let texture = |binding, vis, filterable, dim, count| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: dim,
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
        use wgpu::TextureViewDimension as D;

        let globals_bgl = bgl("globals", &[buffer(0, S::VERTEX_FRAGMENT, uniform, false)]);
        let display_bgl = bgl("display", &[texture(0, S::FRAGMENT, true, D::D2Array, None), sampler(1), sampler(2)]);
        let brush_bgl = bgl("brush", &[buffer(0, S::VERTEX, uniform, true)]);
        let arrays_bgl = bgl(
            "tile arrays",
            &[
                texture(0, S::COMPUTE | S::FRAGMENT, false, D::D2Array, NonZeroU32::new(MAX_LAYER_CHUNKS)),
                texture(1, S::COMPUTE | S::FRAGMENT, false, D::D2Array, NonZeroU32::new(MAX_MASK_CHUNKS)),
            ],
        );
        let flatten_bgl = bgl(
            "flatten",
            &[
                buffer(0, S::COMPUTE, storage, false),
                buffer(1, S::COMPUTE, storage, false),
                buffer(2, S::COMPUTE, uniform, true),
                storage_tex(3),
            ],
        );
        let apply_bgl = bgl("apply", &[buffer(0, S::FRAGMENT, storage, false), buffer(1, S::FRAGMENT, uniform, true)]);
        let warp_bgl = bgl(
            "warp",
            &[
                buffer(0, S::FRAGMENT, storage, false),
                buffer(1, S::FRAGMENT, uniform, true),
                buffer(2, S::FRAGMENT, storage, false),
            ],
        );
        // Binding arrays cannot share a group with dynamic offsets: the
        // composite array gets its own group.
        let comp_array_bgl =
            bgl("composite array", &[texture(0, S::FRAGMENT, false, D::D2Array, NonZeroU32::new(MAX_COMPOSITE_CHUNKS))]);
        let storage_rw = wgpu::BufferBindingType::Storage { read_only: false };
        let bounds_bgl = bgl(
            "bounds",
            &[
                buffer(0, S::COMPUTE, storage, false),
                buffer(1, S::COMPUTE, storage_rw, false),
                buffer(2, S::COMPUTE, uniform, true),
            ],
        );
        let mip_bgl = bgl(
            "mip",
            &[
                texture(0, S::COMPUTE, false, D::D2Array, None),
                storage_tex(1),
                buffer(2, S::COMPUTE, storage, false),
                buffer(3, S::COMPUTE, uniform, true),
            ],
        );
        let screen_bgl = bgl("screen mask", &[texture(2, S::FRAGMENT, false, D::D2, None)]);

        let layout = |label, bgls: &[&wgpu::BindGroupLayout]| {
            let bgls: Vec<_> = bgls.iter().map(|b| Some(*b)).collect();
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &bgls,
                immediate_size: 0,
            })
        };
        let display_layout = layout("display", &[&globals_bgl, &display_bgl]);
        let sel_layout = layout("selection", &[&globals_bgl, &arrays_bgl]);
        let ants_layout = layout("ants", &[&globals_bgl, &screen_bgl]);
        let line_layout = layout("lines", &[&globals_bgl]);
        let tips_bgl = bgl(
            "brush tips",
            &[
                texture(0, S::FRAGMENT, true, D::D2Array, None),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: S::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        );
        let brush_layout = layout("brush", &[&brush_bgl, &tips_bgl]);
        let smudge_tex_bgl = bgl(
            "smudge textures",
            &[
                texture(0, S::FRAGMENT, true, D::D2, None),
                sampler(1),
                texture(2, S::FRAGMENT, true, D::D2Array, None),
                sampler(3),
            ],
        );
        let apply_layout = layout("apply", &[&arrays_bgl, &apply_bgl]);
        let warp_layout = layout("warp", &[&arrays_bgl, &warp_bgl]);
        let smudge_layout = layout("smudge", &[&arrays_bgl, &warp_bgl, &smudge_tex_bgl]);
        let similar_layout = layout("similar", &[&arrays_bgl, &apply_bgl, &comp_array_bgl]);

        // ---- pipelines ------------------------------------------------------
        let tile_attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32];
        let tile_vb = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &tile_attrs,
        };
        let sel_attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32, 2 => Float32];
        let sel_vb = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &sel_attrs,
        };
        let line_attrs = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32, 2 => Float32];
        let line_vb = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &line_attrs,
        };
        let dab_attrs = wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32, 2 => Float32, 3 => Float32x4, 4 => Float32, 5 => Float32, 6 => Uint32
        ];
        let dab_vb = wgpu::VertexBufferLayout {
            array_stride: size_of::<Dab>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &dab_attrs,
        };
        let premul = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        use wgpu::PrimitiveTopology::{TriangleList, TriangleStrip};
        let rp = |label, layout, module, (vs, fs), buffers, format, blend, topology| {
            render_pipeline(device, PipelineSpec { label, layout, module, vs, fs, buffers, format, blend, topology })
        };
        let f = surface_format;
        let (tile_vbs, sel_vbs, line_vbs, dab_vbs) = ([Some(tile_vb)], [Some(sel_vb)], [Some(line_vb)], [Some(dab_vb)]);
        let checker_pipeline =
            rp("checker", &display_layout, &display_shader, ("vs_checker", "fs_checker"), &[], f, None, TriangleStrip);
        let tile_pipeline =
            rp("tiles", &display_layout, &display_shader, ("vs_tile", "fs_tile"), &tile_vbs, f, premul, TriangleStrip);
        let cursor_pipeline =
            rp("cursor", &display_layout, &display_shader, ("vs_cursor", "fs_cursor"), &[], f, premul, TriangleStrip);
        let sel_pipeline = rp(
            "selection",
            &sel_layout,
            &overlay_shader,
            ("vs_sel", "fs_sel"),
            &sel_vbs,
            MASK_FORMAT,
            None,
            TriangleStrip,
        );
        let ants_pipeline = rp("ants", &ants_layout, &overlay_shader, ("vs_full", "fs_ants"), &[], f, premul, TriangleList);
        let line_pipeline =
            rp("lines", &line_layout, &overlay_shader, ("vs_line", "fs_line"), &line_vbs, f, premul, TriangleList);
        let brush_pipeline =
            rp("brush", &brush_layout, &brush_shader, ("vs", "fs"), &dab_vbs, TILE_FORMAT, premul, TriangleStrip);
        let apply_rgba_pipeline =
            rp("apply rgba", &apply_layout, &apply_shader, ("vs", "fs_rgba"), &[], TILE_FORMAT, None, TriangleList);
        let apply_mask_pipeline =
            rp("apply mask", &apply_layout, &apply_shader, ("vs", "fs_mask"), &[], MASK_FORMAT, None, TriangleList);
        let flatten_pipeline = compute_pipeline(
            device,
            "flatten",
            &layout("flatten", &[&arrays_bgl, &flatten_bgl]),
            &flatten_shader,
            "main",
        );
        let mip_pipeline = compute_pipeline(device, "mip", &layout("mip", &[&mip_bgl]), &mip_shader, "main");
        let warp_rgba_pipeline =
            rp("warp rgba", &warp_layout, &warp_shader, ("vs", "fs_rgba"), &[], TILE_FORMAT, None, TriangleList);
        let warp_mask_pipeline =
            rp("warp mask", &warp_layout, &warp_shader, ("vs", "fs_mask"), &[], MASK_FORMAT, None, TriangleList);
        let smudge_pickup_pipeline =
            rp("smudge pickup", &smudge_layout, &smudge_shader, ("vs", "fs_pickup"), &[], TILE_FORMAT, None, TriangleList);
        let smudge_write_pipeline =
            rp("smudge write", &smudge_layout, &smudge_shader, ("vs", "fs_write"), &[], TILE_FORMAT, None, TriangleList);
        let similar_pipeline =
            rp("similar", &similar_layout, &similar_shader, ("vs", "fs"), &[], MASK_FORMAT, None, TriangleList);
        let bounds_pipeline = compute_pipeline(
            device,
            "bounds",
            &layout("bounds", &[&arrays_bgl, &bounds_bgl]),
            &bounds_shader,
            "main",
        );

        // ---- resources ------------------------------------------------------
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
        let make_sampler = |label, mag| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                mag_filter: mag,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            })
        };
        let sampler_linear = make_sampler("linear", wgpu::FilterMode::Linear);
        let sampler_nearest = make_sampler("nearest", wgpu::FilterMode::Nearest);
        let scratch = |label, format| {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: SCRATCH_LAYERS },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: U::RENDER_ATTACHMENT | U::COPY_SRC,
                view_formats: &[],
            });
            let views = (0..SCRATCH_LAYERS)
                .map(|l| {
                    t.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(D::D2),
                        base_array_layer: l,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            (t, views)
        };
        let (scratch_rgba, scratch_rgba_views) = scratch("apply scratch rgba", TILE_FORMAT);
        let (scratch_mask, scratch_mask_views) = scratch("apply scratch mask", MASK_FORMAT);
        // Brush tips.
        let tip_levels = tips::generate();
        let tips_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brush tips"),
            size: wgpu::Extent3d {
                width: tips::TIP_SIZE,
                height: tips::TIP_SIZE,
                depth_or_array_layers: tip_levels[0].len() as u32,
            },
            mip_level_count: tips::TIP_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: U::TEXTURE_BINDING | U::COPY_DST,
            view_formats: &[],
        });
        for (mip, layers) in tip_levels.iter().enumerate() {
            let size = tips::TIP_SIZE >> mip;
            for (layer, data) in layers.iter().enumerate() {
                queue.write_texture(
                    texel_copy(&tips_tex, mip as u32, layer as u32),
                    data,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size), rows_per_image: Some(size) },
                    wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                );
            }
        }
        let tips_view = tips_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(D::D2Array),
            ..Default::default()
        });
        let tips_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brush tips"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let tips_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush tips"),
            layout: &tips_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tips_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&tips_sampler) },
            ],
        });
        let origin_cap = 64;
        let (origin_buf, origin_bg) = make_origin_buf(device, &brush_bgl, origin_cap);
        use wgpu::BufferUsages as B;

        let mut e = Self {
            device: device.clone(),
            queue: queue.clone(),
            doc,
            history: History::new(HISTORY_LIMIT),
            layer_pool,
            mask_pool,
            comp_pool,
            comp: HashMap::new(),
            dirty: HashSet::new(),
            recheck: HashSet::new(),
            recheck_all: true,
            stroke: None,
            preview: None,
            merge_key: None,
            thumbs: None,
            pending_dabs: Vec::new(),
            pending_mips: Vec::new(),
            visible: (DVec2::ZERO, DVec2::ZERO),
            params: Params::new(device),
            ops_buf: DynBuf::new(device, "ops", B::STORAGE, 64 << 10),
            jobs_buf: DynBuf::new(device, "jobs", B::STORAGE, 16 << 10),
            apply_buf: DynBuf::new(device, "apply jobs", B::STORAGE, 16 << 10),
            grid_buf: DynBuf::new(device, "warp grids", B::STORAGE, 16 << 10),
            bounds_jobs: DynBuf::new(device, "bounds jobs", B::STORAGE, 16 << 10),
            list_buf: DynBuf::new(device, "mip layers", B::STORAGE, 16 << 10),
            instance_buf: DynBuf::new(device, "tile instances", B::VERTEX, 64 << 10),
            sel_instance_buf: DynBuf::new(device, "selection instances", B::VERTEX, 16 << 10),
            line_buf: DynBuf::new(device, "overlay lines", B::VERTEX, 16 << 10),
            dab_buf: DynBuf::new(device, "dabs", B::VERTEX, 32 << 10),
            origin_buf,
            origin_bg,
            origin_cap,
            scratch_rgba,
            scratch_rgba_views,
            scratch_mask,
            scratch_mask_views,
            screen_mask: None,
            globals_buf,
            globals_bg,
            display_bgl,
            display_bgs: Vec::new(),
            display_gen: 0,
            sampler_linear,
            sampler_nearest,
            brush_bgl,
            tips_bg,
            tips_view,
            smudge_tex_bgl,
            smudge_buffers: None,
            smudge_jobs: DynBuf::new(device, "smudge jobs", B::STORAGE, 16 << 10),
            smudge_primed: false,
            smudge_cur: 0,
            smudge_pickup_pipeline,
            smudge_write_pipeline,
            arrays_bgl,
            arrays_bg: None,
            arrays_gen: (0, 0),
            flatten_bgl,
            apply_bgl,
            warp_bgl,
            comp_array_bgl,
            bounds_bgl,
            mip_bgl,
            screen_bgl,
            checker_pipeline,
            tile_pipeline,
            cursor_pipeline,
            brush_pipeline,
            sel_pipeline,
            ants_pipeline,
            line_pipeline,
            apply_rgba_pipeline,
            apply_mask_pipeline,
            warp_rgba_pipeline,
            warp_mask_pipeline,
            bounds_pipeline,
            similar_pipeline,
            flatten_pipeline,
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
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        self.merge_key = None;
        let before = self.doc.clone();
        if f(&mut self.doc) {
            self.history.push(before, label);
            self.recheck_all = true;
            true
        } else {
            false
        }
    }

    /// Like [`Self::edit`], but consecutive calls with the same `key` (e.g.
    /// while dragging an opacity slider) form a single undo step.
    pub fn edit_merged(&mut self, label: &str, key: &str, f: impl FnOnce(&mut Document<Tile>) -> bool) -> bool {
        if self.merge_key.as_deref() == Some(key) && self.stroke.is_none() {
            if f(&mut self.doc) {
                self.recheck_all = true;
                return true;
            }
            return false;
        }
        let changed = self.edit(label, f);
        if changed {
            self.merge_key = Some(key.to_string());
        }
        changed
    }

    /// Ends merging of [`Self::edit_merged`] steps (slider released).
    pub fn end_merge(&mut self) {
        self.merge_key = None;
    }

    /// Labels of all history steps (oldest first) and how many are done.
    pub fn history_labels(&self) -> (Vec<String>, usize) {
        (self.history.labels().into_iter().map(String::from).collect(), self.history.undo_len())
    }

    /// Undoes/redoes until exactly `done` steps are applied.
    pub fn jump_history(&mut self, done: usize) -> bool {
        let mut moved = false;
        while self.history.undo_len() > done && self.undo().is_some() {
            moved = true;
        }
        while self.history.undo_len() < done && self.redo().is_some() {
            moved = true;
        }
        moved
    }

    /// Changes that are not part of the history (layer selection).
    pub fn select(&mut self, f: impl FnOnce(&mut Document<Tile>)) {
        if self.stroke.is_none() {
            f(&mut self.doc);
        }
    }

    pub fn undo(&mut self) -> Option<String> {
        self.step_history(true)
    }

    pub fn redo(&mut self) -> Option<String> {
        self.step_history(false)
    }

    fn step_history(&mut self, back: bool) -> Option<String> {
        if self.stroke.is_some() {
            return None;
        }
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        self.merge_key = None;
        let cur = std::mem::replace(&mut self.doc, Document::empty(0, 0));
        let r = if back { self.history.undo(cur) } else { self.history.redo(cur) };
        match r {
            Ok((doc, label)) => {
                self.doc = doc;
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
    /// pixel layer, outside of the undo history (used for loading). Tiles
    /// outside the canvas are kept but not shown.
    pub fn upload_tile(&mut self, layer: LayerId, c: TileCoord, pixels: &[f16]) -> Option<Tile> {
        let tile = self.write_new_tile(pixels)?;
        self.doc.tiles_mut(layer)?.insert(c, tile.clone());
        self.recheck.insert(c);
        self.dirty.insert(c); // content may change under an identical signature
        Some(tile)
    }

    /// Replaces (or removes, with `None`) tiles of a pixel layer as one undoable step.
    /// Returns false if the layer is not a pixel layer, a stroke is active, or memory ran out.
    pub fn replace_tiles(&mut self, layer: LayerId, tiles: Vec<(TileCoord, Option<Vec<f16>>)>, label: &str) -> bool {
        if self.stroke.is_some() || self.doc.find(layer).is_none_or(|l| l.is_group()) {
            return false;
        }
        let mut new = Vec::with_capacity(tiles.len());
        for (c, px) in tiles {
            match px {
                Some(px) => match self.write_new_tile(&px) {
                    Some(t) => new.push((c, Some(t))),
                    None => return false,
                },
                None => new.push((c, None)),
            }
        }
        let coords: Vec<TileCoord> = new.iter().map(|(c, _)| *c).collect();
        let changed = self.edit(label, |d| {
            let map = d.tiles_mut(layer).expect("checked above");
            for (c, t) in new {
                match t {
                    Some(t) => map.insert(c, t),
                    None => map.remove(&c),
                };
            }
            true
        });
        self.dirty.extend(coords);
        changed
    }

    fn write_new_tile(&mut self, pixels: &[f16]) -> Option<Tile> {
        assert_eq!(pixels.len(), (TILE_SIZE * TILE_SIZE * 4) as usize);
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
        Some(tile)
    }

    fn write_new_mask_tile(&mut self, values: &[u8]) -> Option<Tile> {
        assert_eq!(values.len(), (TILE_SIZE * TILE_SIZE) as usize);
        let tile = self.mask_pool.alloc()?;
        let slot = tile.slot();
        self.queue.write_texture(
            texel_copy(&self.mask_pool.chunk(slot).texture, 0, slot.layer as u32),
            values,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(TILE_SIZE), rows_per_image: Some(TILE_SIZE) },
            wgpu::Extent3d { width: TILE_SIZE, height: TILE_SIZE, depth_or_array_layers: 1 },
        );
        Some(tile)
    }

    /// Pixels of a layer tile (`None` if the layer has no tile there).
    pub fn read_layer_tile(&self, layer: LayerId, c: TileCoord) -> Option<Vec<f16>> {
        let t = self.doc.find(layer)?.tiles()?.get(&c)?.clone();
        Some(self.read_layer_tiles(&[t]).remove(0))
    }

    fn in_document(&self, c: TileCoord) -> bool {
        let (gw, gh) = imagine_core::tile::grid_size(self.doc.width, self.doc.height);
        c.x >= 0 && c.y >= 0 && (c.x as u32) < gw && (c.y as u32) < gh
    }

    fn all_doc_coords(&self) -> Vec<TileCoord> {
        let (gw, gh) = imagine_core::tile::grid_size(self.doc.width, self.doc.height);
        (0..gh as i32).flat_map(|y| (0..gw as i32).map(move |x| TileCoord::new(x, y))).collect()
    }

    // ---- strokes ----------------------------------------------------------------

    /// Starts a stroke on the active layer (or its mask).
    pub fn begin_stroke(&mut self, params: StrokeParams) -> Result<(), &'static str> {
        let l = self.doc.active_layer().ok_or("keine aktive Ebene")?;
        if l.is_group() {
            return Err("auf Gruppen kann nicht gemalt werden");
        }
        if !l.visible {
            return Err("die Ebene ist ausgeblendet");
        }
        if params.target == StrokeTarget::Mask && l.mask.is_none() {
            return Err("die Ebene hat keine Maske");
        }
        if params.smudge && (params.target == StrokeTarget::Mask || params.erase) {
            return Err("Verwischen geht nur auf Pixeln");
        }
        self.smudge_primed = false;
        if self.doc.selection.as_ref().is_some_and(|s| s.tiles.is_empty() && s.default == 0) {
            return Err("die Auswahl ist leer");
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

    // ---- stats --------------------------------------------------------------------

    pub fn layer_tiles_live(&self) -> usize {
        self.layer_pool.live()
    }

    pub fn mask_tiles_live(&self) -> usize {
        self.mask_pool.live()
    }

    pub fn composite_tiles_live(&self) -> usize {
        self.comp_pool.live()
    }

    pub fn gpu_bytes(&self) -> u64 {
        self.layer_pool.allocated_bytes() + self.mask_pool.allocated_bytes() + self.comp_pool.allocated_bytes()
    }

    pub fn max_layer_tiles(&self) -> u64 {
        self.layer_pool.capacity_tiles()
    }
}
