//! The agent command set: one JSON-serialisable command per operation,
//! executed against a [`Session`]. Shared by the MCP server, `imagine run`
//! scripts and the live socket of a running window.
//!
//! Conventions (also stated in the tool descriptions):
//! - coordinates are document pixels, origin top-left;
//! - colours are sRGB: `"#rrggbb"`, `"#rrggbbaa"` or `[r, g, b(, a)]` in 0–255;
//! - `layer` defaults to the active layer; layer ids are stable.

use std::path::PathBuf;

use anyhow::{Context, bail, ensure};
use glam::{DAffine2, DVec2};
use half::f16;
use image::RgbaImage;
use imagine_core::color::{Pixels, extract_region, linear_to_srgb, srgb_to_linear};
use imagine_core::doc::Content;
use imagine_core::mask::{CombineOp, Shape};
use imagine_core::tile::tiles_in_rect;
use imagine_core::tr;
use imagine_core::{
    BlendMode, BlendSpace, BrushSettings, Document, InputSample, LayerId, Stroke, TILE_SIZE,
    TIP_NAMES, TileCoord, tip_from_name,
};
use imagine_gpu::{GpuContext, GradientKind, Interp, SampleFrom, StrokeParams, StrokeTarget, Tile};
use imagine_io::export::blit_region;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::loader::Source;
use crate::session::{self, Session};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ColorSpec {
    Hex(String),
    Rgb(Vec<f32>),
}

impl ColorSpec {
    /// Linear RGB + straight alpha.
    fn linear(&self) -> anyhow::Result<([f32; 3], f32)> {
        let v: Vec<f32> = match self {
            ColorSpec::Hex(h) => {
                let h = h.trim_start_matches('#');
                ensure!(
                    h.len() == 6 || h.len() == 8,
                    "colour must be #rrggbb or #rrggbbaa, got {h:?}"
                );
                (0..h.len() / 2)
                    .map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).map(f32::from))
                    .collect::<Result<_, _>>()
                    .context("invalid hex colour")?
            }
            ColorSpec::Rgb(v) => v.clone(),
        };
        ensure!(
            v.len() == 3 || v.len() == 4,
            "colour needs 3 or 4 components"
        );
        ensure!(
            v.iter().all(|c| (0.0..=255.0).contains(c)),
            "colour components must be 0–255"
        );
        let rgb = [0, 1, 2].map(|i| srgb_to_linear(v[i] / 255.0));
        Ok((rgb, v.get(3).map_or(1.0, |a| a / 255.0)))
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    DocumentNew {
        width: u32,
        height: u32,
    },
    DocumentOpen {
        path: PathBuf,
    },
    DocumentInfo,
    LayerAdd {
        name: Option<String>,
    },
    LayerDuplicate {
        layer: Option<LayerId>,
    },
    LayerDelete {
        layer: Option<LayerId>,
    },
    LayerSet {
        layer: Option<LayerId>,
        name: Option<String>,
        opacity: Option<f32>,
        blend: Option<String>,
        visible: Option<bool>,
        /// Clipping mask: show only where the layers below have pixels.
        clip: Option<bool>,
        /// Groups: Pass Through (children blend with what is below).
        pass_through: Option<bool>,
    },
    LayerSelect {
        layer: LayerId,
    },
    LayerMove {
        layer: Option<LayerId>,
        steps: i32,
    },
    LayerGroup {
        layer: Option<LayerId>,
    },
    LayerUngroup {
        layer: Option<LayerId>,
    },
    SetBlendSpace {
        space: String,
    },
    PaintStroke {
        layer: Option<LayerId>,
        points: Vec<Vec<f64>>,
        color: Option<ColorSpec>,
        size: f32,
        hardness: Option<f32>,
        opacity: Option<f32>,
        flow: Option<f32>,
        spacing: Option<f32>,
        #[serde(default)]
        eraser: bool,
        target: Option<String>,
        tip: Option<String>,
        angle: Option<f32>,
        roundness: Option<f32>,
        size_jitter: Option<f32>,
        angle_jitter: Option<f32>,
        scatter: Option<f32>,
        flow_jitter: Option<f32>,
        follow_direction: Option<bool>,
        smudge: Option<f32>,
        /// Clone stamp: source = point + clone_offset.
        clone_offset: Option<[i32; 2]>,
        #[serde(default)]
        clone_all_layers: bool,
        /// Healing brush (with clone_offset): blend into the surroundings.
        #[serde(default)]
        heal: bool,
        /// Spot healing: no source needed, it is found automatically.
        #[serde(default)]
        spot_heal: bool,
    },
    Fill {
        layer: Option<LayerId>,
        color: ColorSpec,
        opacity: Option<f32>,
        target: Option<String>,
    },
    Clear {
        layer: Option<LayerId>,
    },
    Gradient {
        layer: Option<LayerId>,
        kind: Option<String>,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        color0: ColorSpec,
        color1: Option<ColorSpec>,
        opacity: Option<f32>,
        target: Option<String>,
    },
    SelectRect {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        mode: Option<String>,
    },
    SelectEllipse {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        mode: Option<String>,
    },
    SelectPolygon {
        points: Vec<Vec<f64>>,
        mode: Option<String>,
    },
    SelectAll,
    SelectNone,
    SelectInvert,
    SelectFeather {
        radius: f64,
    },
    SelectMagicWand {
        x: u32,
        y: u32,
        tolerance: Option<f32>,
        contiguous: Option<bool>,
        antialias: Option<bool>,
        sample: Option<String>,
        mode: Option<String>,
    },
    SelectColorRange {
        color: Option<ColorSpec>,
        x: Option<u32>,
        y: Option<u32>,
        fuzziness: Option<f32>,
        sample: Option<String>,
        mode: Option<String>,
    },
    LayerMaskAdd {
        layer: Option<LayerId>,
        from_selection: Option<bool>,
    },
    LayerMaskDelete {
        layer: Option<LayerId>,
    },
    LayerMaskSet {
        layer: Option<LayerId>,
        enabled: bool,
    },
    LayerMaskApply {
        layer: Option<LayerId>,
    },
    LayerTranslate {
        layer: Option<LayerId>,
        dx: i64,
        dy: i64,
    },
    LayerTransform {
        layer: Option<LayerId>,
        scale: Option<f64>,
        scale_x: Option<f64>,
        scale_y: Option<f64>,
        rotate: Option<f64>,
        dx: Option<f64>,
        dy: Option<f64>,
        pivot_x: Option<f64>,
        pivot_y: Option<f64>,
        interp: Option<String>,
        use_selection: Option<bool>,
    },
    LayerBounds {
        layer: Option<LayerId>,
    },
    Histogram {
        below: Option<LayerId>,
        bins: Option<u32>,
    },
    FilterApply {
        layer: Option<LayerId>,
        filter: imagine_core::Filter,
        #[serde(default)]
        smart: bool,
    },
    Liquify {
        layer: Option<LayerId>,
        strokes: Vec<LiquifyStroke>,
    },
    ContentAwareFill {
        layer: Option<LayerId>,
        #[serde(default)]
        all_layers: bool,
    },
    TextAdd {
        text: String,
        x: f64,
        y: f64,
        size: f32,
        font: Option<String>,
        color: Option<ColorSpec>,
        weight: Option<f32>,
        #[serde(default)]
        italic: bool,
        align: Option<imagine_core::vector::TextAlign>,
        line_height: Option<f32>,
        letter_spacing: Option<f32>,
        width: Option<f32>,
        name: Option<String>,
    },
    ShapeAdd {
        shape: String,
        x: Option<f64>,
        y: Option<f64>,
        width: Option<f64>,
        height: Option<f64>,
        radius: Option<f64>,
        points: Option<Vec<[f64; 2]>>,
        d: Option<String>,
        fill: Option<ColorSpec>,
        #[serde(default)]
        no_fill: bool,
        stroke_color: Option<ColorSpec>,
        stroke_width: Option<f32>,
        name: Option<String>,
    },
    VectorSet {
        layer: Option<LayerId>,
        changes: Value,
    },
    LayerRasterize {
        layer: Option<LayerId>,
    },
    LayerEffectsSet {
        layer: Option<LayerId>,
        effects: Vec<imagine_core::style::Effect>,
    },
    SmartFilterSet {
        layer: Option<LayerId>,
        index: Option<usize>,
        filter: Option<imagine_core::Filter>,
        enabled: Option<bool>,
        #[serde(default)]
        remove: bool,
        #[serde(default)]
        apply: bool,
    },
    AdjustmentAdd {
        adjustment: imagine_core::Adjustment,
        name: Option<String>,
    },
    AdjustmentSet {
        layer: Option<LayerId>,
        adjustment: imagine_core::Adjustment,
    },
    Crop {
        x: i64,
        y: i64,
        width: u32,
        height: u32,
    },
    CanvasSize {
        width: u32,
        height: u32,
        anchor: Option<String>,
    },
    ImageResize {
        width: Option<u32>,
        height: Option<u32>,
        interp: Option<String>,
    },
    FillRect {
        layer: Option<LayerId>,
        x: i64,
        y: i64,
        width: i64,
        height: i64,
        color: ColorSpec,
        opacity: Option<f32>,
    },
    ClearRect {
        layer: Option<LayerId>,
        x: i64,
        y: i64,
        width: i64,
        height: i64,
    },
    PlaceImage {
        path: PathBuf,
        x: Option<i64>,
        y: Option<i64>,
        name: Option<String>,
    },
    SampleColor {
        x: f64,
        y: f64,
    },
    RenderPreview {
        max_size: Option<u32>,
        x: Option<i64>,
        y: Option<i64>,
        width: Option<i64>,
        height: Option<i64>,
        #[serde(default = "default_true")]
        checkerboard: bool,
        /// Tint unselected areas red (like quick mask).
        #[serde(default)]
        show_selection: bool,
        /// Also write the PNG to this path.
        output: Option<PathBuf>,
    },
    Undo,
    Redo,
    Save {
        path: Option<PathBuf>,
    },
    Export {
        path: PathBuf,
        color_space: Option<imagine_io::export::ColorSpace>,
        bit_depth: Option<u8>,
        quality: Option<u8>,
    },
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::DocumentNew { .. } => "document_new",
            Self::DocumentOpen { .. } => "document_open",
            Self::DocumentInfo => "document_info",
            Self::LayerAdd { .. } => "layer_add",
            Self::LayerDuplicate { .. } => "layer_duplicate",
            Self::LayerDelete { .. } => "layer_delete",
            Self::LayerSet { .. } => "layer_set",
            Self::LayerSelect { .. } => "layer_select",
            Self::LayerMove { .. } => "layer_move",
            Self::LayerGroup { .. } => "layer_group",
            Self::LayerUngroup { .. } => "layer_ungroup",
            Self::SetBlendSpace { .. } => "set_blend_space",
            Self::PaintStroke { .. } => "paint_stroke",
            Self::Fill { .. } => "fill",
            Self::Clear { .. } => "clear",
            Self::Gradient { .. } => "gradient",
            Self::SelectRect { .. } => "select_rect",
            Self::SelectEllipse { .. } => "select_ellipse",
            Self::SelectPolygon { .. } => "select_polygon",
            Self::SelectAll => "select_all",
            Self::SelectNone => "select_none",
            Self::SelectInvert => "select_invert",
            Self::SelectFeather { .. } => "select_feather",
            Self::LayerMaskAdd { .. } => "layer_mask_add",
            Self::LayerMaskDelete { .. } => "layer_mask_delete",
            Self::LayerMaskSet { .. } => "layer_mask_set",
            Self::LayerMaskApply { .. } => "layer_mask_apply",
            Self::LayerTranslate { .. } => "layer_translate",
            Self::LayerTransform { .. } => "layer_transform",
            Self::SelectMagicWand { .. } => "select_magic_wand",
            Self::SelectColorRange { .. } => "select_color_range",
            Self::LayerBounds { .. } => "layer_bounds",
            Self::Histogram { .. } => "histogram",
            Self::AdjustmentAdd { .. } => "adjustment_add",
            Self::FilterApply { .. } => "filter_apply",
            Self::SmartFilterSet { .. } => "smart_filter_set",
            Self::Liquify { .. } => "liquify",
            Self::ContentAwareFill { .. } => "content_aware_fill",
            Self::TextAdd { .. } => "text_add",
            Self::ShapeAdd { .. } => "shape_add",
            Self::VectorSet { .. } => "vector_set",
            Self::LayerRasterize { .. } => "layer_rasterize",
            Self::LayerEffectsSet { .. } => "layer_effects_set",
            Self::AdjustmentSet { .. } => "adjustment_set",
            Self::Crop { .. } => "crop",
            Self::CanvasSize { .. } => "canvas_size",
            Self::ImageResize { .. } => "image_resize",
            Self::FillRect { .. } => "fill_rect",
            Self::ClearRect { .. } => "clear_rect",
            Self::PlaceImage { .. } => "place_image",
            Self::SampleColor { .. } => "sample_color",
            Self::RenderPreview { .. } => "render_preview",
            Self::Undo => "undo",
            Self::Redo => "redo",
            Self::Save { .. } => "save",
            Self::Export { .. } => "export",
        }
    }

    /// Whether the command changes the document (for "modified" tracking).
    pub fn mutates(&self) -> bool {
        !matches!(
            self,
            Self::DocumentInfo
                | Self::SampleColor { .. }
                | Self::RenderPreview { .. }
                | Self::Save { .. }
                | Self::Export { .. }
        )
    }

    pub fn parse(v: Value) -> anyhow::Result<Self> {
        serde_json::from_value(v).map_err(|e| anyhow::anyhow!("invalid command: {e}"))
    }
}

#[derive(Debug, Default)]
pub struct Output {
    pub result: Value,
    /// PNG bytes (render_preview).
    pub image_png: Option<Vec<u8>>,
    /// The session was replaced (document_new / document_open).
    pub replaced_document: bool,
}

impl Output {
    fn json(result: Value) -> Self {
        Self {
            result,
            ..Default::default()
        }
    }
}

pub struct Ctx<'a> {
    pub gpu: &'a GpuContext,
    pub format: wgpu::TextureFormat,
    pub session: &'a mut Option<Session>,
}

fn session(s: &mut Option<Session>) -> anyhow::Result<&mut Session> {
    s.as_mut()
        .context("no document open – use document_new or document_open first")
}

/// Resolves `layer` (default: active), checks it exists and makes it active.
fn target_layer(s: &mut Session, layer: Option<LayerId>) -> anyhow::Result<LayerId> {
    let id = layer.unwrap_or(s.engine.doc().active);
    ensure!(
        s.engine.doc().find(id).is_some(),
        "layer {id} does not exist"
    );
    s.engine.select(|d| d.active = id);
    Ok(id)
}

fn pixel_or_group(s: &mut Session, layer: Option<LayerId>) -> anyhow::Result<LayerId> {
    target_layer(s, layer)
}

/// Selection summary for agents.
fn selection_info(s: &Session) -> Value {
    match &s.engine.doc().selection {
        None => json!({ "selection": null }),
        Some(m) => {
            let partial = m
                .tiles
                .values()
                .filter(|t| matches!(t, imagine_core::MaskTile::Pixels(_)))
                .count();
            json!({ "selection": { "default": m.default, "tiles": m.tiles.len(), "edge_tiles": partial } })
        }
    }
}

fn pixel_layer(s: &mut Session, layer: Option<LayerId>) -> anyhow::Result<LayerId> {
    let id = target_layer(s, layer)?;
    ensure!(
        !s.engine.doc().find(id).unwrap().is_group(),
        "layer {id} is a group – pick a pixel layer"
    );
    Ok(id)
}

fn parse_target(t: &Option<String>) -> anyhow::Result<StrokeTarget> {
    match t.as_deref() {
        None | Some("pixels") => Ok(StrokeTarget::Pixels),
        Some("mask") => Ok(StrokeTarget::Mask),
        Some(o) => bail!("target must be \"pixels\" or \"mask\", got {o:?}"),
    }
}

fn parse_mode(m: &Option<String>) -> anyhow::Result<CombineOp> {
    match m.as_deref() {
        None => Ok(CombineOp::Replace),
        Some(k) => CombineOp::from_key(k).with_context(|| {
            format!("mode must be replace, add, subtract or intersect, got {k:?}")
        }),
    }
}

/// Premultiplied linear colour; for mask targets the grey value is the sRGB luminance.
fn premul(c: &ColorSpec, target: StrokeTarget) -> anyhow::Result<[f32; 4]> {
    let (rgb, a) = c.linear()?;
    Ok(match target {
        StrokeTarget::Pixels => [rgb[0] * a, rgb[1] * a, rgb[2] * a, a],
        StrokeTarget::Mask => {
            let l = linear_to_srgb(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]);
            [l * a, l * a, l * a, a]
        }
    })
}

fn parse_sample(v: &Option<String>) -> anyhow::Result<SampleFrom> {
    match v.as_deref() {
        None | Some("layer") => Ok(SampleFrom::Layer),
        Some("image") => Ok(SampleFrom::Image),
        Some(o) => bail!("sample must be \"layer\" or \"image\", got {o:?}"),
    }
}

fn parse_interp(i: &Option<String>) -> anyhow::Result<Interp> {
    match i.as_deref() {
        None => Ok(Interp::Bicubic),
        Some(k) => Interp::from_key(k)
            .with_context(|| format!("interp must be nearest, bilinear or bicubic, got {k:?}")),
    }
}

/// Fractions (0 = left/top, 1 = right/bottom) for a canvas anchor.
fn parse_anchor(a: Option<&str>) -> anyhow::Result<(f64, f64)> {
    let a = a.unwrap_or("center");
    let x = if a.contains("left") {
        0.0
    } else if a.contains("right") {
        1.0
    } else {
        0.5
    };
    let y = if a.starts_with("top") {
        0.0
    } else if a.starts_with("bottom") {
        1.0
    } else {
        0.5
    };
    let known = [
        "center",
        "top",
        "bottom",
        "left",
        "right",
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ];
    ensure!(known.contains(&a), "anchor must be one of {known:?}");
    Ok((x, y))
}

fn bounds_json(b: (i64, i64, i64, i64)) -> Value {
    json!({ "x": b.0, "y": b.1, "width": b.2 - b.0, "height": b.3 - b.1 })
}

fn engine_err(e: &str) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

fn not_painting(s: &Session) -> anyhow::Result<()> {
    ensure!(
        !s.engine.stroke_active(),
        "the user is painting right now – retry in a moment"
    );
    Ok(())
}

pub fn execute(ctx: &mut Ctx, cmd: Command) -> anyhow::Result<Output> {
    match cmd {
        Command::DocumentNew { width, height } => {
            ensure!(
                (1..=200_000).contains(&width) && (1..=200_000).contains(&height),
                "size out of range"
            );
            *ctx.session = Some(session::load(
                ctx.gpu,
                ctx.format,
                Source::Blank(width, height),
            )?);
            let mut out = Output::json(info(session(ctx.session)?));
            out.replaced_document = true;
            Ok(out)
        }
        Command::DocumentOpen { path } => {
            ensure!(path.exists(), "{} does not exist", path.display());
            *ctx.session = Some(session::load(ctx.gpu, ctx.format, Source::File(path))?);
            let mut out = Output::json(info(session(ctx.session)?));
            out.replaced_document = true;
            Ok(out)
        }
        cmd => {
            let s = session(ctx.session)?;
            execute_on(s, cmd)
        }
    }
}

fn edit(
    s: &mut Session,
    label: &str,
    f: impl FnOnce(&mut Document<Tile>) -> bool,
) -> anyhow::Result<()> {
    not_painting(s)?;
    ensure!(s.engine.edit(label, f), "{label}: nothing changed");
    Ok(())
}

fn execute_on(s: &mut Session, cmd: Command) -> anyhow::Result<Output> {
    let ok = |s: &Session| {
        Ok(Output::json(
            json!({ "ok": true, "active_layer": s.engine.doc().active }),
        ))
    };
    match cmd {
        Command::DocumentNew { .. } | Command::DocumentOpen { .. } => unreachable!(),
        Command::DocumentInfo => Ok(Output::json(info(s))),
        Command::LayerAdd { name } => {
            not_painting(s)?;
            let mut id = 0;
            s.engine.edit(tr("New Layer"), |d| {
                id = d.add_pixel_layer();
                if let Some(n) = name {
                    d.find_mut(id).unwrap().name = n;
                }
                true
            });
            Ok(Output::json(json!({ "layer": id })))
        }
        Command::LayerDuplicate { layer } => {
            target_layer(s, layer)?;
            let mut id = None;
            edit(s, tr("Duplicate Layer"), |d| {
                id = d.duplicate_active();
                id.is_some()
            })?;
            Ok(Output::json(json!({ "layer": id })))
        }
        Command::LayerDelete { layer } => {
            target_layer(s, layer)?;
            edit(s, tr("Delete Layer"), |d| d.remove_active())
                .context("cannot delete the last layer")?;
            ok(s)
        }
        Command::LayerSet {
            layer,
            name,
            opacity,
            blend,
            visible,
            clip,
            pass_through,
        } => {
            let id = target_layer(s, layer)?;
            let blend = blend
                .map(|b| {
                    BlendMode::from_key(&b).with_context(|| {
                        let keys: Vec<_> = BlendMode::ALL.iter().map(|m| m.key()).collect();
                        format!("unknown blend mode {b:?}; valid: {}", keys.join(", "))
                    })
                })
                .transpose()?;
            if let Some(o) = opacity {
                ensure!((0.0..=1.0).contains(&o), "opacity must be 0.0–1.0");
            }
            edit(s, tr("Layer Properties"), |d| {
                let l = d.find_mut(id).unwrap();
                if let Some(n) = name {
                    l.name = n;
                }
                if let Some(o) = opacity {
                    l.opacity = o;
                }
                if let Some(b) = blend {
                    l.blend = b;
                }
                if let Some(c) = clip {
                    l.clip = c;
                }
                if let Some(p) = pass_through {
                    l.pass_through = p && l.is_group();
                }
                if let Some(v) = visible {
                    l.visible = v;
                }
                true
            })?;
            ok(s)
        }
        Command::LayerSelect { layer } => {
            target_layer(s, Some(layer))?;
            ok(s)
        }
        Command::LayerMove { layer, steps } => {
            target_layer(s, layer)?;
            not_painting(s)?;
            let mut moved = 0;
            s.engine.edit(tr("Move Layer"), |d| {
                for _ in 0..steps.unsigned_abs() {
                    if !d.move_active(steps.signum()) {
                        break;
                    }
                    moved += steps.signum();
                }
                moved != 0
            });
            Ok(Output::json(json!({ "moved": moved })))
        }
        Command::LayerGroup { layer } => {
            target_layer(s, layer)?;
            let mut id = None;
            edit(s, tr("Group Layers"), |d| {
                id = d.group_active();
                id.is_some()
            })?;
            Ok(Output::json(json!({ "group": id })))
        }
        Command::LayerUngroup { layer } => {
            target_layer(s, layer)?;
            edit(s, tr("Ungroup Layers"), |d| d.ungroup_active())
                .context("layer is not a group")?;
            ok(s)
        }
        Command::SetBlendSpace { space } => {
            let space =
                BlendSpace::from_key(&space).context("space must be \"srgb\" or \"linear\"")?;
            edit(s, tr("Blend Color Space"), |d| {
                let changed = d.blend_space != space;
                d.blend_space = space;
                changed
            })?;
            ok(s)
        }
        Command::PaintStroke {
            layer,
            points,
            color,
            size,
            hardness,
            opacity,
            flow,
            spacing,
            eraser,
            target,
            tip,
            angle,
            roundness,
            size_jitter,
            angle_jitter,
            scatter,
            flow_jitter,
            follow_direction,
            smudge,
            clone_offset,
            clone_all_layers,
            heal,
            spot_heal,
        } => {
            pixel_layer(s, layer)?;
            not_painting(s)?;
            let target = parse_target(&target)?;
            ensure!(!points.is_empty(), "points must not be empty");
            ensure!(size > 0.0 && size <= 5000.0, "size must be 0–5000 px");
            let (mut color, alpha) = match &color {
                Some(c) => c.linear()?,
                None if eraser || smudge.is_some() || clone_offset.is_some() || spot_heal => {
                    ([0.35; 3], 1.0)
                }
                None => bail!("color is required unless eraser, smudge or clone_offset is set"),
            };
            if target == StrokeTarget::Mask {
                let l = linear_to_srgb(0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]);
                color = [l; 3];
            }
            let settings = BrushSettings {
                size,
                hardness: hardness.unwrap_or(0.8).clamp(0.0, 1.0),
                flow: flow.unwrap_or(1.0).clamp(0.0, 1.0),
                opacity: (opacity.unwrap_or(1.0) * alpha).clamp(0.0, 1.0),
                spacing: spacing.unwrap_or(0.1).clamp(0.01, 10.0),
                min_size: 0.15,
                smoothing: 0.0,
                color,
                eraser,
                tip: match &tip {
                    None => 0,
                    Some(t) => tip_from_name(t)
                        .with_context(|| format!("unknown tip {t:?}, use one of {TIP_NAMES:?}"))?,
                },
                angle: angle.unwrap_or(0.0),
                roundness: roundness.unwrap_or(1.0).clamp(0.05, 1.0),
                size_jitter: size_jitter.unwrap_or(0.0).clamp(0.0, 1.0),
                angle_jitter: angle_jitter.unwrap_or(0.0).clamp(0.0, 180.0),
                scatter: scatter.unwrap_or(0.0).clamp(0.0, 5.0),
                flow_jitter: flow_jitter.unwrap_or(0.0).clamp(0.0, 1.0),
                follow_direction: follow_direction.unwrap_or(false),
                smudge: smudge.map(|v| v.clamp(0.0, 1.0)),
            };
            let samples = points
                .iter()
                .map(|p| {
                    ensure!(
                        p.len() == 2 || p.len() == 3,
                        "each point is [x, y] or [x, y, pressure]"
                    );
                    Ok(InputSample {
                        pos: DVec2::new(p[0], p[1]),
                        pressure: p.get(2).map_or(1.0, |&v| v as f32),
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            s.engine
                .begin_stroke(StrokeParams {
                    opacity: settings.opacity,
                    erase: eraser,
                    target,
                    smudge: smudge.is_some() && clone_offset.is_none(),
                    clone: if spot_heal {
                        Some(imagine_gpu::CloneSource {
                            offset: [0, 0],
                            all_layers: clone_all_layers,
                            heal: true,
                            spot: true,
                        })
                    } else {
                        clone_offset.map(|offset| imagine_gpu::CloneSource {
                            offset,
                            all_layers: clone_all_layers,
                            heal,
                            spot: false,
                        })
                    },
                })
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut stroke = Stroke::new(settings);
            let mut dabs = 0;
            for p in &samples {
                let d = stroke.add(*p);
                dabs += d.len();
                s.engine.stroke_dabs(&d);
            }
            let d = stroke.finish(*samples.last().unwrap());
            dabs += d.len();
            s.engine.stroke_dabs(&d);
            s.engine.end_stroke(if spot_heal {
                tr("Spot Healing Brush")
            } else if heal && clone_offset.is_some() {
                tr("Healing Brush")
            } else if clone_offset.is_some() {
                tr("Clone Stamp")
            } else if smudge.is_some() {
                tr("Smudge")
            } else if eraser {
                tr("Erase")
            } else {
                tr("Brush Stroke")
            });
            Ok(Output::json(json!({ "ok": true, "dabs": dabs })))
        }
        Command::FillRect {
            layer,
            x,
            y,
            width,
            height,
            color,
            opacity,
        } => {
            let id = pixel_layer(s, layer)?;
            let (rgb, a) = color.linear()?;
            let a = a * opacity.unwrap_or(1.0).clamp(0.0, 1.0);
            modify_rect(s, id, (x, y, width, height), tr("Fill Area"), |px| {
                for k in 0..3 {
                    px[k] = rgb[k] * a + px[k] * (1.0 - a);
                }
                px[3] = a + px[3] * (1.0 - a);
            })
        }
        Command::Fill {
            layer,
            color,
            opacity,
            target,
        } => {
            pixel_layer(s, layer)?;
            let target = parse_target(&target)?;
            let n = s
                .engine
                .fill(
                    target,
                    premul(&color, target)?,
                    opacity.unwrap_or(1.0).clamp(0.0, 1.0),
                    tr("Fill"),
                )
                .map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "tiles": n })))
        }
        Command::Clear { layer } => {
            pixel_layer(s, layer)?;
            let n = s.engine.clear_selected(tr("Clear")).map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "tiles": n })))
        }
        Command::Gradient {
            layer,
            kind,
            x0,
            y0,
            x1,
            y1,
            color0,
            color1,
            opacity,
            target,
        } => {
            pixel_layer(s, layer)?;
            let target = parse_target(&target)?;
            let kind = match kind.as_deref() {
                None | Some("linear") => GradientKind::Linear,
                Some("radial") => GradientKind::Radial,
                Some(k) => bail!("kind must be \"linear\" or \"radial\", got {k:?}"),
            };
            let c0 = premul(&color0, target)?;
            let c1 = match &color1 {
                Some(c) => premul(c, target)?,
                None => [0.0; 4],
            };
            let n = s
                .engine
                .gradient(
                    target,
                    kind,
                    (x0, y0),
                    (x1, y1),
                    c0,
                    c1,
                    opacity.unwrap_or(1.0).clamp(0.0, 1.0),
                    tr("Gradient"),
                )
                .map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "tiles": n })))
        }
        Command::SelectRect {
            x,
            y,
            width,
            height,
            mode,
        } => {
            ensure!(width > 0.0 && height > 0.0, "width and height must be > 0");
            let op = parse_mode(&mode)?;
            s.engine
                .select_shape(
                    &Shape::Rect {
                        x,
                        y,
                        w: width,
                        h: height,
                    },
                    op,
                    tr("Selection"),
                )
                .map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectEllipse {
            x,
            y,
            width,
            height,
            mode,
        } => {
            ensure!(width > 0.0 && height > 0.0, "width and height must be > 0");
            let op = parse_mode(&mode)?;
            let shape = Shape::Ellipse {
                cx: x + width / 2.0,
                cy: y + height / 2.0,
                rx: width / 2.0,
                ry: height / 2.0,
            };
            s.engine
                .select_shape(&shape, op, tr("Selection"))
                .map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectPolygon { points, mode } => {
            ensure!(points.len() >= 3, "a polygon needs at least 3 points");
            ensure!(points.iter().all(|p| p.len() == 2), "each point is [x, y]");
            let op = parse_mode(&mode)?;
            let poly = points.iter().map(|p| (p[0], p[1])).collect();
            s.engine
                .select_shape(&Shape::Polygon(poly), op, tr("Lasso Selection"))
                .map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectAll => {
            s.engine.select_all(tr("Select All")).map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectNone => {
            s.engine.deselect(tr("Deselect")).map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectInvert => {
            s.engine
                .invert_selection(tr("Inverse"))
                .map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectFeather { radius } => {
            ensure!((0.0..=1000.0).contains(&radius), "radius must be 0–1000 px");
            s.engine
                .feather_selection(radius, tr("Feather Selection"))
                .map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectMagicWand {
            x,
            y,
            tolerance,
            contiguous,
            antialias,
            sample,
            mode,
        } => {
            let tolerance = tolerance.unwrap_or(32.0);
            ensure!(
                (0.0..=255.0).contains(&tolerance),
                "tolerance must be 0–255"
            );
            s.engine
                .magic_wand(
                    x,
                    y,
                    tolerance,
                    contiguous.unwrap_or(true),
                    antialias.unwrap_or(true),
                    parse_sample(&sample)?,
                    parse_mode(&mode)?,
                    tr("Magic Wand"),
                )
                .map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectColorRange {
            color,
            x,
            y,
            fuzziness,
            sample,
            mode,
        } => {
            let from = parse_sample(&sample)?;
            let reference = match (color, x, y) {
                (Some(c), None, None) => {
                    let (rgb, a) = c.linear()?;
                    [
                        linear_to_srgb(rgb[0]),
                        linear_to_srgb(rgb[1]),
                        linear_to_srgb(rgb[2]),
                        a,
                    ]
                }
                (None, Some(x), Some(y)) => {
                    s.engine.sample_straight(x, y, from).map_err(engine_err)?
                }
                _ => bail!("give either color or x and y"),
            };
            let fuzziness = fuzziness.unwrap_or(40.0);
            ensure!(
                (0.0..=200.0).contains(&fuzziness),
                "fuzziness must be 0–200"
            );
            s.engine
                .select_color_range(
                    reference,
                    fuzziness,
                    from,
                    parse_mode(&mode)?,
                    tr("Color Range"),
                )
                .map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::LayerMaskAdd {
            layer,
            from_selection,
        } => {
            let id = pixel_or_group(s, layer)?;
            let use_sel = from_selection.unwrap_or(true);
            let mask = match (use_sel, s.engine.selection_cpu()) {
                (true, Some(sel)) => sel,
                _ => imagine_core::mask::CpuMask::filled(255),
            };
            ensure!(
                s.engine.doc().find(id).unwrap().mask.is_none(),
                "layer {id} already has a mask"
            );
            s.engine
                .set_layer_mask(id, Some(mask), tr("Layer Mask"))
                .map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "layer": id })))
        }
        Command::LayerMaskDelete { layer } => {
            let id = pixel_or_group(s, layer)?;
            ensure!(
                s.engine.doc().find(id).unwrap().mask.is_some(),
                "layer {id} has no mask"
            );
            s.engine
                .set_layer_mask(id, None, tr("Delete Layer Mask"))
                .map_err(engine_err)?;
            ok(s)
        }
        Command::LayerMaskSet { layer, enabled } => {
            let id = pixel_or_group(s, layer)?;
            ensure!(
                s.engine.doc().find(id).unwrap().mask.is_some(),
                "layer {id} has no mask"
            );
            edit(s, tr("Toggle Layer Mask"), |d| {
                let l = d.find_mut(id).unwrap();
                let changed = l.mask_enabled != enabled;
                l.mask_enabled = enabled;
                changed
            })?;
            ok(s)
        }
        Command::LayerMaskApply { layer } => {
            pixel_layer(s, layer)?;
            s.engine
                .apply_layer_mask(tr("Apply Layer Mask"))
                .map_err(engine_err)?;
            ok(s)
        }
        Command::LayerTranslate { layer, dx, dy } => {
            pixel_layer(s, layer)?;
            s.engine
                .translate_active(dx, dy, tr("Move"))
                .map_err(engine_err)?;
            ok(s)
        }
        Command::LayerTransform {
            layer,
            scale,
            scale_x,
            scale_y,
            rotate,
            dx,
            dy,
            pivot_x,
            pivot_y,
            interp,
            use_selection,
        } => {
            let id = pixel_layer(s, layer)?;
            let interp = parse_interp(&interp)?;
            let use_selection = use_selection.unwrap_or(true);
            let k = DVec2::new(
                scale_x.or(scale).unwrap_or(1.0),
                scale_y.or(scale).unwrap_or(1.0),
            );
            ensure!(k.x != 0.0 && k.y != 0.0, "scale must not be 0");
            let pivot = match (pivot_x, pivot_y) {
                (Some(x), Some(y)) => DVec2::new(x, y),
                (None, None) => {
                    let b = s
                        .engine
                        .content_bounds(id)
                        .context("the layer (or selection) is empty")?;
                    DVec2::new((b.0 + b.2) as f64 * 0.5, (b.1 + b.3) as f64 * 0.5)
                }
                _ => bail!("give both pivot_x and pivot_y, or neither"),
            };
            let fwd = DAffine2::from_translation(
                DVec2::new(dx.unwrap_or(0.0), dy.unwrap_or(0.0)) + pivot,
            ) * DAffine2::from_angle(rotate.unwrap_or(0.0).to_radians())
                * DAffine2::from_scale(k)
                * DAffine2::from_translation(-pivot);
            s.engine
                .transform_layer(fwd, interp, use_selection, tr("Transform"))
                .map_err(engine_err)?;
            let bounds = s.engine.content_bounds(id).map(bounds_json);
            Ok(Output::json(
                json!({ "ok": true, "active_layer": id, "bounds": bounds }),
            ))
        }
        Command::FilterApply {
            layer,
            filter,
            smart,
        } => {
            pixel_layer(s, layer)?;
            let label = filter.name();
            if smart {
                s.engine
                    .add_smart_filter(&filter, label)
                    .map_err(engine_err)?;
            } else {
                s.engine.apply_filter(&filter, label).map_err(engine_err)?;
            }
            ok(s)
        }
        Command::TextAdd {
            text,
            x,
            y,
            size,
            font,
            color,
            weight,
            italic,
            align,
            line_height,
            letter_spacing,
            width,
            name,
        } => {
            ensure!(size > 0.0, "size must be positive");
            let (rgb, a) = match &color {
                Some(c) => c.linear()?,
                None => ([0.0; 3], 1.0),
            };
            let mut t =
                imagine_core::vector::TextSpec::new(&text, x, y, size, [rgb[0], rgb[1], rgb[2], a]);
            if let Some(f) = font {
                t.font = f;
            }
            t.weight = weight.unwrap_or(t.weight).clamp(100.0, 900.0);
            t.italic = italic;
            t.align = align.unwrap_or_default();
            t.line_height = line_height.unwrap_or(t.line_height);
            t.letter_spacing = letter_spacing.unwrap_or(0.0);
            t.width = width;
            let id = s
                .engine
                .add_vector(imagine_core::VectorLayer::text(t), tr("Text"))
                .context("not possible right now (painting?)")?;
            if let Some(n) = name {
                s.engine.select(|d| d.find_mut(id).unwrap().name = n);
            }
            vector_ok(s, id)
        }
        Command::ShapeAdd {
            shape,
            x,
            y,
            width,
            height,
            radius,
            points,
            d,
            fill,
            no_fill,
            stroke_color,
            stroke_width,
            name,
        } => {
            use imagine_core::vector::{Geometry, ShapeSpec, StrokeStyle};
            let need = |v: Option<f64>, n: &str| v.with_context(|| format!("{shape} needs {n}"));
            let geometry = match shape.as_str() {
                "rect" => Geometry::Rect {
                    x: need(x, "x")?,
                    y: need(y, "y")?,
                    w: need(width, "width")?,
                    h: need(height, "height")?,
                    radius: radius.unwrap_or(0.0),
                },
                "ellipse" => {
                    let (w, h) = (need(width, "width")?, need(height, "height")?);
                    Geometry::Ellipse {
                        cx: need(x, "x")? + w / 2.0,
                        cy: need(y, "y")? + h / 2.0,
                        rx: w / 2.0,
                        ry: h / 2.0,
                    }
                }
                "polygon" => Geometry::Polygon {
                    points: points.context("polygon needs points")?,
                    closed: true,
                },
                "path" => Geometry::Path {
                    d: d.context("path needs d (SVG path data)")?,
                },
                o => bail!("shape must be rect, ellipse, polygon or path, got {o:?}"),
            };
            let lin = |c: &ColorSpec| -> anyhow::Result<[f32; 4]> {
                let (rgb, a) = c.linear()?;
                Ok([rgb[0], rgb[1], rgb[2], a])
            };
            let fill = match (&fill, no_fill) {
                (_, true) => None,
                (Some(c), false) => Some(lin(c)?),
                (None, false) => Some([0.0, 0.0, 0.0, 1.0]),
            };
            let stroke = match (&stroke_color, stroke_width) {
                (None, None) => None,
                (c, w) => Some(StrokeStyle {
                    color: c
                        .as_ref()
                        .map(lin)
                        .transpose()?
                        .unwrap_or([0.0, 0.0, 0.0, 1.0]),
                    width: w.unwrap_or(2.0).max(0.1),
                }),
            };
            ensure!(
                fill.is_some() || stroke.is_some(),
                "a shape needs a fill or a stroke"
            );
            let v = imagine_core::VectorLayer::shape(ShapeSpec {
                geometry,
                fill,
                stroke,
            });
            ensure!(
                imagine_vector::bounds(&v).is_some(),
                "the shape is empty or the path data is invalid"
            );
            let id = s
                .engine
                .add_vector(v, tr("Shape"))
                .context("not possible right now (painting?)")?;
            if let Some(n) = name {
                s.engine.select(|d| d.find_mut(id).unwrap().name = n);
            }
            vector_ok(s, id)
        }
        Command::VectorSet { layer, changes } => {
            let id = target_layer(s, layer)?;
            let current = s
                .engine
                .doc()
                .find(id)
                .and_then(|l| l.vector().cloned())
                .with_context(|| format!("layer {id} is not a text or shape layer"))?;
            let mut v = serde_json::to_value(&current)?;
            merge_json(&mut v, changes)?;
            let new: imagine_core::VectorLayer =
                serde_json::from_value(v).context("invalid change")?;
            ensure!(
                std::mem::discriminant(&new.kind) == std::mem::discriminant(&current.kind),
                "cannot turn text into a shape or back"
            );
            s.engine
                .set_vector(id, new, tr("Edit Vector"), None)
                .map_err(engine_err)?;
            vector_ok(s, id)
        }
        Command::LayerEffectsSet { layer, effects } => {
            let id = target_layer(s, layer)?;
            let l = s.engine.doc().find(id).unwrap();
            ensure!(
                l.is_pixels() || l.vector().is_some(),
                "layer {id} has no pixels (group or adjustment)"
            );
            s.engine.edit(tr("Layer Style"), |d| {
                d.find_mut(id).unwrap().effects = effects;
                true
            });
            ok(s)
        }
        Command::LayerRasterize { layer } => {
            let id = target_layer(s, layer)?;
            s.engine
                .rasterize(id, tr("Rasterize"))
                .map_err(engine_err)?;
            ok(s)
        }
        Command::ContentAwareFill { layer, all_layers } => {
            pixel_layer(s, layer)?;
            let ms = s
                .engine
                .content_aware_fill(all_layers, tr("Content-Aware Fill"))
                .map_err(engine_err)?;
            let mut v = ok(s)?;
            v.result["ms"] = json!(ms.round());
            Ok(v)
        }
        Command::Liquify { layer, strokes } => {
            pixel_layer(s, layer)?;
            ensure!(
                !s.engine.liquify_active(),
                "liquify is open in the window – apply or cancel it there first"
            );
            ensure!(!strokes.is_empty(), "no strokes");
            s.engine.liquify_begin().map_err(engine_err)?;
            for st in &strokes {
                let pts: Vec<DVec2> = st.points.iter().map(|p| DVec2::new(p[0], p[1])).collect();
                if pts.is_empty() {
                    s.engine.liquify_cancel();
                    bail!("a stroke needs points");
                }
                let r = st.radius.max(1.0);
                let k = st.strength.unwrap_or(0.7).clamp(0.0, 1.0);
                s.engine.liquify_stroke_start();
                if st.tool == imagine_gpu::LiquifyTool::Push {
                    for w in pts.windows(2) {
                        let d = w[1] - w[0];
                        let n = (d.length() / (r * 0.1).max(1.0)).ceil().max(1.0) as usize;
                        for i in 1..=n {
                            s.engine.liquify_dab(
                                st.tool,
                                w[0] + d * (i as f64 / n as f64),
                                d / n as f64,
                                r,
                                k,
                                1.0,
                            );
                        }
                    }
                } else {
                    let steps = st.steps.unwrap_or(10).min(1000);
                    for p in &pts {
                        for _ in 0..steps {
                            s.engine.liquify_dab(st.tool, *p, DVec2::ZERO, r, k, 1.0);
                        }
                    }
                }
            }
            s.engine.liquify_commit(tr("Liquify")).map_err(engine_err)?;
            ok(s)
        }
        Command::SmartFilterSet {
            layer,
            index,
            filter,
            enabled,
            remove,
            apply,
        } => {
            let id = pixel_layer(s, layer)?;
            let n = s.engine.doc().find(id).unwrap().filters.len();
            ensure!(n > 0, "layer {id} has no smart filters");
            if apply {
                s.engine
                    .bake_smart_filters(id, tr("Apply Smart Filters"))
                    .map_err(engine_err)?;
                return ok(s);
            }
            let i = index.context("index (0 = first filter, applied first) is required")?;
            ensure!(
                i < n,
                "index {i} out of range (layer {id} has {n} smart filters)"
            );
            if remove {
                s.engine.edit(tr("Remove Smart Filter"), |d| {
                    d.find_mut(id).unwrap().filters.remove(i);
                    true
                });
                return ok(s);
            }
            if let Some(f) = filter {
                ensure!(
                    std::mem::discriminant(&f)
                        == std::mem::discriminant(
                            &s.engine.doc().find(id).unwrap().filters[i].filter
                        ),
                    "filter kind differs – remove it and add a new one"
                );
                s.engine
                    .set_smart_filter(id, i, &f, tr("Edit Smart Filter"))
                    .map_err(engine_err)?;
            }
            if let Some(on) = enabled {
                s.engine.edit(tr("Toggle Smart Filter"), |d| {
                    let f = &mut d.find_mut(id).unwrap().filters[i];
                    let changed = f.enabled != on;
                    f.enabled = on;
                    changed
                });
            }
            ok(s)
        }
        Command::AdjustmentAdd { adjustment, name } => {
            let label = tr!("Adjustment Layer {name}", name = adjustment.name());
            let id = s
                .engine
                .add_adjustment(adjustment, &label)
                .context("not possible right now (painting?)")?;
            if let Some(n) = name {
                s.engine.select(|d| d.find_mut(id).unwrap().name = n);
            }
            ok(s)
        }
        Command::AdjustmentSet { layer, adjustment } => {
            let id = target_layer(s, layer)?;
            let l = s.engine.doc().find(id).unwrap();
            let old = l
                .adjustment()
                .with_context(|| format!("layer {id} is not an adjustment layer"))?;
            ensure!(
                old.key() == adjustment.key(),
                "layer {id} is a {} adjustment; delete it and add a new one to change the kind",
                old.key()
            );
            s.engine.edit(tr("Edit Adjustment"), |d| {
                d.find_mut(id).unwrap().content = Content::Adjustment(adjustment);
                true
            });
            ok(s)
        }
        Command::LayerBounds { layer } => {
            let id = target_layer(s, layer)?;
            let b = s.engine.content_bounds(id).map(bounds_json);
            Ok(Output::json(json!({ "layer": id, "bounds": b })))
        }
        Command::Histogram { below, bins } => {
            let n = bins.unwrap_or(32);
            if n == 0 || n > 256 || 256 % n != 0 {
                anyhow::bail!("bins: 1, 2, 4 … 256");
            }
            if let Some(id) = below
                && s.engine.doc().find(id).is_none()
            {
                anyhow::bail!("layer {id} does not exist");
            }
            let h = s.engine.histogram(below);
            let channel = |bins: &[f32; 256]| {
                let bins: Vec<f64> = bins.iter().map(|&b| b as f64).collect();
                let total: f64 = bins.iter().sum::<f64>().max(1e-9);
                let r4 = |v: f64| (v * 1e4).round() / 1e4;
                let quantile = |q: f64| {
                    let mut acc = 0.0;
                    bins.iter()
                        .position(|b| {
                            acc += b;
                            acc >= q * total
                        })
                        .unwrap_or(255)
                };
                let mean = bins
                    .iter()
                    .enumerate()
                    .map(|(i, b)| i as f64 * b)
                    .sum::<f64>()
                    / total;
                let grouped: Vec<f64> = bins
                    .chunks(256 / n as usize)
                    .map(|c| r4(c.iter().sum::<f64>() / total))
                    .collect();
                json!({
                    "mean": (mean * 10.0).round() / 10.0,
                    "median": quantile(0.5), "p1": quantile(0.01), "p99": quantile(0.99),
                    "clipped_black": r4(bins[0] / total),
                    "clipped_white": r4(bins[255] / total),
                    "bins": grouped,
                })
            };
            Ok(Output::json(json!({
                "below": below,
                "scale": "sRGB-encoded 0–255, fractions of all (alpha-weighted) pixels",
                "rgb": channel(&h.bins[0]), "red": channel(&h.bins[1]),
                "green": channel(&h.bins[2]), "blue": channel(&h.bins[3]),
            })))
        }
        Command::Crop {
            x,
            y,
            width,
            height,
        } => {
            s.engine
                .crop(x, y, width, height, tr("Crop"))
                .map_err(engine_err)?;
            Ok(Output::json(
                json!({ "ok": true, "width": width, "height": height }),
            ))
        }
        Command::CanvasSize {
            width,
            height,
            anchor,
        } => {
            let (w, h) = s.engine.size();
            let (ax, ay) = parse_anchor(anchor.as_deref())?;
            let x = ((w as f64 - width as f64) * ax).round() as i64;
            let y = ((h as f64 - height as f64) * ay).round() as i64;
            s.engine
                .crop(x, y, width, height, tr("Canvas Size"))
                .map_err(engine_err)?;
            Ok(Output::json(
                json!({ "ok": true, "width": width, "height": height, "offset": [-x, -y] }),
            ))
        }
        Command::ImageResize {
            width,
            height,
            interp,
        } => {
            let (w, h) = s.engine.size();
            let (nw, nh) = match (width, height) {
                (Some(a), Some(b)) => (a, b),
                (Some(a), None) => (a, ((h as f64 * a as f64 / w as f64).round() as u32).max(1)),
                (None, Some(b)) => (((w as f64 * b as f64 / h as f64).round() as u32).max(1), b),
                (None, None) => bail!("give width and/or height"),
            };
            s.engine
                .resize_image(nw, nh, parse_interp(&interp)?, tr("Image Size"))
                .map_err(engine_err)?;
            Ok(Output::json(
                json!({ "ok": true, "width": nw, "height": nh }),
            ))
        }
        Command::ClearRect {
            layer,
            x,
            y,
            width,
            height,
        } => {
            let id = pixel_layer(s, layer)?;
            modify_rect(s, id, (x, y, width, height), tr("Clear Area"), |px| {
                *px = [0.0; 4]
            })
        }
        Command::PlaceImage { path, x, y, name } => {
            place_image(s, &path, x.unwrap_or(0), y.unwrap_or(0), name)
        }
        Command::SampleColor { x, y } => sample(s, x, y),
        Command::RenderPreview {
            max_size,
            x,
            y,
            width,
            height,
            checkerboard,
            show_selection,
            output,
        } => {
            let mut out = preview(
                s,
                max_size.unwrap_or(1024),
                (x, y, width, height),
                checkerboard,
                show_selection,
            )?;
            if let (Some(path), Some(png)) = (&output, &out.image_png) {
                std::fs::write(path, png)
                    .with_context(|| format!("cannot write {}", path.display()))?;
                out.result["output"] = json!(path);
            }
            Ok(out)
        }
        Command::Undo => {
            not_painting(s)?;
            Ok(Output::json(json!({ "undone": s.engine.undo() })))
        }
        Command::Redo => {
            not_painting(s)?;
            Ok(Output::json(json!({ "redone": s.engine.redo() })))
        }
        Command::Save { path } => {
            let r = s.save(path.as_deref())?;
            Ok(Output::json(json!({
                "path": r.path, "tiles_written": r.tiles_written, "bytes": r.bytes, "incremental": r.incremental
            })))
        }
        Command::Export {
            path,
            color_space,
            bit_depth,
            quality,
        } => {
            let d = imagine_io::export::ExportOptions::default();
            let opts = imagine_io::export::ExportOptions {
                space: color_space.unwrap_or(d.space),
                sixteen_bit: match bit_depth {
                    None | Some(8) => false,
                    Some(16) => true,
                    Some(b) => anyhow::bail!("bit_depth {b}: must be 8 or 16"),
                },
                quality: quality.unwrap_or(d.quality).clamp(1, 100),
            };
            let sixteen = opts.sixteen_bit && imagine_io::export::supports_16bit(&path);
            let ms = s.export(&path, &opts)?;
            Ok(Output::json(
                json!({ "path": path, "ms": ms.round(), "color_space": opts.space, "bit_depth": if sixteen { 16 } else { 8 } }),
            ))
        }
    }
}

/// One liquify stroke for the `liquify` command.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiquifyStroke {
    #[serde(default)]
    pub tool: imagine_gpu::LiquifyTool,
    /// Document px; push follows the path, the other tools act at each point.
    pub points: Vec<[f64; 2]>,
    pub radius: f64,
    /// 0…1 (default 0.7).
    pub strength: Option<f64>,
    /// Continuous tools: dabs per point (default 10; ≈ frames held).
    pub steps: Option<u32>,
}

/// Result for a vector command: the layer and where its content ended up.
fn vector_ok(s: &mut Session, id: LayerId) -> anyhow::Result<Output> {
    let mut v = Output::json(json!({ "ok": true, "active_layer": s.engine.doc().active }));
    v.result["layer"] = json!(id);
    if let Some(b) = s
        .engine
        .doc()
        .find(id)
        .and_then(|l| l.vector())
        .and_then(imagine_vector::bounds)
    {
        v.result["bounds"] = json!(b.map(|x| x.round()));
    }
    Ok(v)
}

/// Merges `patch` into `target` (objects recursively; "#rrggbb" colours
/// become linear [r, g, b, a] arrays like the stored format).
fn merge_json(target: &mut Value, patch: Value) -> anyhow::Result<()> {
    match patch {
        Value::Object(p) => {
            if !target.is_object() {
                *target = json!({});
            }
            let t = target.as_object_mut().unwrap();
            for (k, mut v) in p {
                if matches!(k.as_str(), "color" | "fill")
                    && let Value::String(hex) = &v
                {
                    let (rgb, a) = ColorSpec::Hex(hex.clone()).linear()?;
                    v = json!([rgb[0], rgb[1], rgb[2], a]);
                }
                match t.get_mut(&k) {
                    Some(old) if v.is_object() => merge_json(old, v)?,
                    _ => {
                        t.insert(k, v);
                    }
                }
            }
        }
        v => *target = v,
    }
    Ok(())
}

/// Document summary for agents.
pub fn info(s: &Session) -> Value {
    fn layers(ls: &[imagine_core::Layer<Tile>]) -> Vec<Value> {
        ls.iter()
            .rev() // top → bottom, like a layers panel
            .map(|l| {
                let mut v = json!({
                    "id": l.id,
                    "name": l.name,
                    "blend": l.blend.key(),
                    "opacity": l.opacity,
                    "visible": l.visible,
                });
                if let Some(m) = &l.mask {
                    v["mask"] = json!({ "enabled": l.mask_enabled, "default": m.default, "tiles": m.tiles.len() });
                }
                if !l.filters.is_empty() {
                    v["smart_filters"] = serde_json::to_value(&l.filters).unwrap_or_default();
                }
                if !l.effects.is_empty() {
                    v["effects"] = serde_json::to_value(&l.effects).unwrap_or_default();
                }
                match &l.content {
                    Content::Pixels(t) => {
                        v["type"] = "pixels".into();
                        v["tiles"] = t.len().into();
                        if let (Some(x0), Some(y0), Some(x1), Some(y1)) = (
                            t.keys().map(|c| c.x).min(),
                            t.keys().map(|c| c.y).min(),
                            t.keys().map(|c| c.x).max(),
                            t.keys().map(|c| c.y).max(),
                        ) {
                            let ts = TILE_SIZE as i64;
                            v["tile_bounds"] = json!([
                                x0 as i64 * ts,
                                y0 as i64 * ts,
                                (x1 - x0 + 1) as i64 * ts,
                                (y1 - y0 + 1) as i64 * ts
                            ]);
                        }
                    }
                    Content::Group(c) => {
                        v["type"] = "group".into();
                        v["children"] = layers(c).into();
                    }
                    Content::Vector(vl) => {
                        v["type"] = "vector".into();
                        v["vector"] = serde_json::to_value(vl).unwrap_or_default();
                        if let Some(b) = imagine_vector::bounds(vl) {
                            v["bounds"] = json!(b.map(|x| x.round()));
                        }
                    }
                    Content::Adjustment(a) => {
                        v["type"] = "adjustment".into();
                        v["adjustment"] = serde_json::to_value(a).unwrap_or_default();
                    }
                }
                v
            })
            .collect()
    }
    let d = s.engine.doc();
    let (undo, redo) = s.engine.history_len();
    json!({
        "name": s.name,
        "width": d.width,
        "height": d.height,
        "blend_space": d.blend_space.key(),
        "active_layer": d.active,
        "layers_top_to_bottom": layers(&d.layers),
        "history": { "undo_steps": undo, "redo_steps": redo },
        "selection": selection_info(s)["selection"],
        "source": s.source,
        "save_path": s.default_save_path(),
    })
}

/// Applies `f` to every pixel (premultiplied linear RGBA) inside the rectangle, as one undo step.
fn modify_rect(
    s: &mut Session,
    layer: LayerId,
    (x, y, w, h): (i64, i64, i64, i64),
    label: &str,
    f: impl Fn(&mut [f32; 4]),
) -> anyhow::Result<Output> {
    not_painting(s)?;
    ensure!(w > 0 && h > 0, "width and height must be > 0");
    let (dw, dh) = s.engine.size();
    let (x0, y0) = (x.max(0), y.max(0));
    let (x1, y1) = ((x + w).min(dw as i64), (y + h).min(dh as i64));
    ensure!(x0 < x1 && y0 < y1, "rectangle lies outside the document");
    let ts = TILE_SIZE as i64;
    let coords: Vec<TileCoord> =
        tiles_in_rect((x0 as f64, y0 as f64), (x1 as f64, y1 as f64), dw, dh).collect();
    let mut tiles = Vec::with_capacity(coords.len());
    for c in coords {
        let mut px = s
            .engine
            .read_layer_tile(layer, c)
            .unwrap_or_else(|| vec![f16::ZERO; (ts * ts * 4) as usize]);
        let (ox, oy) = c.origin();
        for py in (y0 - oy).max(0)..(y1 - oy).min(ts) {
            for pxx in (x0 - ox).max(0)..(x1 - ox).min(ts) {
                let i = ((py * ts + pxx) * 4) as usize;
                let mut v = [px[i], px[i + 1], px[i + 2], px[i + 3]].map(f16::to_f32);
                f(&mut v);
                for k in 0..4 {
                    px[i + k] = f16::from_f32(v[k]);
                }
            }
        }
        let empty = px.iter().skip(3).step_by(4).all(|a| a.to_f32() == 0.0);
        tiles.push((c, (!empty).then_some(px)));
    }
    let n = tiles.len();
    ensure!(
        s.engine.replace_tiles(layer, tiles, label),
        "{label} failed (GPU memory full?)"
    );
    Ok(Output::json(json!({ "ok": true, "tiles": n })))
}

fn place_image(
    s: &mut Session,
    path: &std::path::Path,
    x: i64,
    y: i64,
    name: Option<String>,
) -> anyhow::Result<Output> {
    not_painting(s)?;
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    reader.no_limits();
    let img = reader
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))?
        .into_rgba8();
    let name = name.unwrap_or_else(|| {
        path.file_stem()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    });
    let (id, n) = place_rgba(s, &img, x, y, &name, tr("Place Image"))?;
    Ok(Output::json(
        json!({ "layer": id, "width": img.width(), "height": img.height(), "tiles": n }),
    ))
}

/// Puts an sRGB image on a new layer at (x, y) (one undo step). Returns the
/// layer and the number of tiles.
pub(crate) fn place_rgba(
    s: &mut Session,
    img: &image::RgbaImage,
    x: i64,
    y: i64,
    name: &str,
    label: &str,
) -> anyhow::Result<(LayerId, usize)> {
    let (iw, ih) = img.dimensions();
    let (dw, dh) = s.engine.size();
    let px = Pixels::Srgb8(img);
    let coords: Vec<TileCoord> = tiles_in_rect(
        (x as f64, y as f64),
        ((x + iw as i64) as f64, (y + ih as i64) as f64),
        dw,
        dh,
    )
    .collect();
    let tiles: Vec<(TileCoord, Option<Vec<f16>>)> = coords
        .into_iter()
        .filter_map(|c| {
            let (ox, oy) = c.origin();
            extract_region(&px, iw, ih, ox - x, oy - y).map(|t| (c, Some(t)))
        })
        .collect();
    ensure!(!tiles.is_empty(), "image does not overlap the document");
    let mut id = 0;
    let name = name.to_string();
    s.engine.edit(tr("New Layer"), |d| {
        id = d.add_pixel_layer();
        d.find_mut(id).unwrap().name = name;
        true
    });
    let n = tiles.len();
    ensure!(
        s.engine.replace_tiles(id, tiles, label),
        "placing failed (GPU memory full?)"
    );
    Ok((id, n))
}

fn to_srgb8(px: [f32; 4]) -> ([u8; 3], f32) {
    let a = px[3].clamp(0.0, 1.0);
    let c = if a > 0.0 {
        [0, 1, 2].map(|i| (linear_to_srgb((px[i] / a).clamp(0.0, 1.0)) * 255.0).round() as u8)
    } else {
        [0; 3]
    };
    (c, a)
}

fn sample(s: &mut Session, x: f64, y: f64) -> anyhow::Result<Output> {
    let (dw, dh) = s.engine.size();
    ensure!(
        x >= 0.0 && y >= 0.0 && x < dw as f64 && y < dh as f64,
        "point outside the document"
    );
    s.engine.flush();
    let c = TileCoord::containing(x, y);
    let tile = s.engine.read_composite(&[c]).remove(0);
    let (ox, oy) = c.origin();
    let i = (((y as i64 - oy) * TILE_SIZE as i64 + (x as i64 - ox)) * 4) as usize;
    let (rgb, a) = to_srgb8([tile[i], tile[i + 1], tile[i + 2], tile[i + 3]].map(f16::to_f32));
    Ok(Output::json(json!({
        "hex": format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2]),
        "rgb": rgb,
        "alpha": (a * 1000.0).round() / 1000.0,
    })))
}

/// Flattened image (or a region of it) as PNG, at most `max_size` px on its longer side.
fn preview(
    s: &mut Session,
    max_size: u32,
    region: (Option<i64>, Option<i64>, Option<i64>, Option<i64>),
    checkerboard: bool,
    show_selection: bool,
) -> anyhow::Result<Output> {
    let max_size = max_size.clamp(16, 4096);
    let (dw, dh) = s.engine.size();
    let rx = region.0.unwrap_or(0).clamp(0, dw as i64 - 1);
    let ry = region.1.unwrap_or(0).clamp(0, dh as i64 - 1);
    let rw = region.2.unwrap_or(dw as i64 - rx).clamp(1, dw as i64 - rx);
    let rh = region.3.unwrap_or(dh as i64 - ry).clamp(1, dh as i64 - ry);

    // Read the composite at the mip level closest to the requested size.
    let longest = rw.max(rh) as f64;
    let mip = (longest / max_size as f64).log2().floor().clamp(0.0, 8.0) as u32;
    let scale = 1i64 << mip;
    s.engine.flush();
    let coords: Vec<TileCoord> = tiles_in_rect(
        (rx as f64, ry as f64),
        ((rx + rw) as f64, (ry + rh) as f64),
        dw,
        dh,
    )
    .collect();
    let present: Vec<TileCoord> = {
        let have: std::collections::HashSet<_> = s.engine.composite_coords().into_iter().collect();
        coords.into_iter().filter(|c| have.contains(c)).collect()
    };
    let (iw, ih) = (
        ((rw + scale - 1) / scale) as u32,
        ((rh + scale - 1) / scale) as u32,
    );
    let mut img = RgbaImage::new(iw.max(1), ih.max(1));
    for batch in present.chunks(256) {
        let px = s.engine.read_composite_mip(batch, mip);
        let tiles: Vec<_> = batch.iter().copied().zip(px).collect();
        blit_region(&mut img, &tiles, TILE_SIZE >> mip, (rx / scale, ry / scale));
    }
    if img.width().max(img.height()) > max_size {
        let f = max_size as f64 / img.width().max(img.height()) as f64;
        let (w, h) = (
            ((img.width() as f64 * f).round() as u32).max(1),
            ((img.height() as f64 * f).round() as u32).max(1),
        );
        img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
    }
    if checkerboard {
        for (x, y, p) in img.enumerate_pixels_mut() {
            let bg = if ((x / 8) + (y / 8)) % 2 == 0 {
                204u32
            } else {
                153
            };
            let a = p[3] as u32;
            let mix = |c: u8| ((c as u32 * a + bg * (255 - a)) / 255) as u8;
            *p = image::Rgba([mix(p[0]), mix(p[1]), mix(p[2]), 255]);
        }
    }
    if show_selection && let Some(sel) = s.engine.selection_cpu() {
        // Quick-mask style: unselected areas tinted red.
        let (w, h) = (img.width() as f64, img.height() as f64);
        for (px, py, p) in img.enumerate_pixels_mut() {
            let dx = rx + ((px as f64 + 0.5) / w * rw as f64) as i64;
            let dy = ry + ((py as f64 + 0.5) / h * rh as f64) as i64;
            let a = (255 - sel.value_at(dx, dy)) as u32 / 2;
            for (i, tint) in [204u32, 0, 0].iter().enumerate() {
                p[i] = ((p[i] as u32 * (255 - a) + tint * a) / 255) as u8;
            }
        }
    }
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    Ok(Output {
        result: json!({
            "width": img.width(),
            "height": img.height(),
            "region": [rx, ry, rw, rh],
            "scale": rw as f64 / img.width() as f64,
        }),
        image_png: Some(png),
        replaced_document: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_commands_and_rejects_typos() {
        let c = Command::parse(json!({"cmd": "fill_rect", "x": 1, "y": 2, "width": 3, "height": 4, "color": "#ff0000"})).unwrap();
        assert_eq!(c.name(), "fill_rect");
        assert!(Command::parse(json!({"cmd": "fill_rect", "x": 1, "y": 2, "widht": 3, "height": 4, "color": "#f00"})).is_err());
        assert!(Command::parse(json!({"cmd": "explode"})).is_err());
        let c = Command::parse(json!({"cmd": "paint_stroke", "points": [[0, 0], [10, 5, 0.5]], "size": 4, "color": [255, 0, 0]})).unwrap();
        assert!(matches!(c, Command::PaintStroke { eraser: false, .. }));
        assert!(
            !Command::parse(json!({"cmd": "document_info"}))
                .unwrap()
                .mutates()
        );
    }

    #[test]
    fn colours() {
        let (rgb, a) = ColorSpec::Hex("#ffffff80".into()).linear().unwrap();
        assert_eq!(rgb, [1.0; 3]);
        assert!((a - 128.0 / 255.0).abs() < 1e-6);
        assert!(ColorSpec::Hex("#12".into()).linear().is_err());
        assert!(ColorSpec::Rgb(vec![300.0, 0.0, 0.0]).linear().is_err());
        assert_eq!(ColorSpec::Rgb(vec![0.0, 0.0, 0.0]).linear().unwrap().1, 1.0);
    }
}
