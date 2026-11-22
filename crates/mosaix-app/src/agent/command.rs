//! The agent command set: one JSON-serialisable command per operation,
//! executed against a [`Session`]. Shared by the MCP server, `mosaix run`
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
use mosaix_core::color::{Pixels, extract_region, linear_to_srgb, srgb_to_linear};
use mosaix_core::doc::Content;
use mosaix_core::mask::{CombineOp, CpuMask, Shape};
use mosaix_core::tile::tiles_in_rect;
use mosaix_core::tr;
use mosaix_core::{
    BlendMode, BlendSpace, BrushSettings, Document, InputSample, LayerId, Stroke, TILE_SIZE,
    TIP_NAMES, TileCoord, tip_from_name,
};
use mosaix_gpu::{GpuContext, GradientKind, Interp, SampleFrom, StrokeParams, StrokeTarget, Tile};
use mosaix_io::export::blit_region;
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

/// A Puppet Warp pin: from where (document px, on the layer) to where.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PuppetPin {
    pub from: [f64; 2],
    pub to: [f64; 2],
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
        /// How to develop a camera RAW file (default: as shot, no exposure change).
        #[serde(default)]
        raw: Option<mosaix_io::raw::Develop>,
    },
    /// Selects the main subject of the picture (local AI).
    SelectSubject {
        mode: Option<String>,
    },
    /// Selects the object(s) at points (local AI); `negative` points mark
    /// what does not belong to it.
    SelectObject {
        points: Vec<Vec<f64>>,
        #[serde(default)]
        negative: Vec<Vec<f64>>,
        mode: Option<String>,
    },
    /// Masks everything but the main subject on a layer (local AI).
    RemoveBackground {
        layer: Option<LayerId>,
    },
    /// Enlarges the image with local AI (every pixel layer; the rest as image_resize).
    ImageUpscale {
        #[serde(default = "default_upscale")]
        factor: u32,
    },
    /// Removes the selected area: generated from its surroundings (local AI).
    RemoveArea {
        layer: Option<LayerId>,
        #[serde(default)]
        all_layers: bool,
        /// Grow the selection by this many pixels first (default 4, at most 48).
        expand: Option<u32>,
    },
    /// Downloads the local AI models (the user agreed to the download).
    AiModelDownload {
        model: String,
    },
    /// Actions (macros) and scripts in the actions folder.
    ActionList,
    /// Plays a macro on the open document (each step one undo step).
    ActionRun {
        name: String,
    },
    /// Filter plugins: names, parameters, errors.
    FilterPluginList,
    /// Checks a WGSL filter plugin and (unless `check_only`) saves it to the plugin folder.
    FilterPluginInstall {
        source: String,
        #[serde(default)]
        check_only: bool,
    },
    /// Camera, shot data and as-shot white balance of a RAW file.
    RawInfo {
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
        /// Fill opacity: the layer's own pixels, not its style.
        fill_opacity: Option<f32>,
        /// Locks: {transparency, pixels, position}.
        lock: Option<mosaix_core::doc::Lock>,
        /// Colour label: none, red, orange, yellow, green, blue, violet, gray.
        label: Option<String>,
    },
    /// Layer › Delete Hidden Layers.
    LayerDeleteHidden,
    /// Edit › Transform › Warp: 16 offsets (row-major, top row first) for
    /// the 4 × 4 Bézier grid over the layer's content.
    LayerWarp {
        offsets: Vec<[f64; 2]>,
    },
    /// Edit › Puppet Warp: pins at `from` move to `to` (equal: holds that
    /// place); the layer bends as rigidly as possible.
    LayerPuppetWarp {
        pins: Vec<PuppetPin>,
    },
    /// Turn the whole image by an angle (degrees, clockwise) about its centre.
    ImageRotate {
        angle: f64,
    },
    LayerSelect {
        layer: LayerId,
        /// Further layers selected along with it (delete, group, merge,
        /// duplicate and move then act on all of them).
        #[serde(default)]
        also: Vec<LayerId>,
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
        /// Pencil: hard pixels without anti-aliasing.
        #[serde(default)]
        pencil: bool,
        /// History Brush: paint the layer back as it was at the history
        /// source (see `history_source`).
        #[serde(default)]
        history: bool,
        /// Paint in this blend mode (multiply, screen, color …).
        blend: Option<String>,
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
        /// Retouching: dodge, burn, sponge_saturate, sponge_desaturate, blur, sharpen.
        tone: Option<String>,
        /// Dodge/burn: shadows, midtones (default), highlights.
        range: Option<String>,
        /// Exposure / strength of the retouching brush, 0…1 (default 0.5).
        amount: Option<f32>,
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
    SelectModify {
        op: String,
        amount: f64,
    },
    SelectSave {
        name: String,
    },
    SelectLoad {
        channel: Option<usize>,
        name: Option<String>,
        mode: Option<String>,
        #[serde(default)]
        invert: bool,
    },
    ChannelList,
    GuideAdd {
        orientation: String,
        position: f64,
    },
    GuideList,
    GenerativeFill {
        layer: Option<LayerId>,
        prompt: String,
        /// Result on a new layer (default: as set in the dialog, normally yes).
        new_layer: Option<bool>,
    },
    LayerPlace {
        path: PathBuf,
    },
    LayerConvertSmart {
        layer: Option<LayerId>,
    },
    /// The selected part of the active layer onto a new layer (Ctrl+J /
    /// Ctrl+Shift+J with a selection).
    LayerVia {
        #[serde(default)]
        cut: bool,
    },
    /// A selection from a layer's opacity (Ctrl+click on the thumbnail).
    SelectLayerPixels {
        layer: Option<LayerId>,
        mode: Option<String>,
    },
    /// Paint Bucket: fill the region of similar colour at (x, y).
    BucketFill {
        x: u32,
        y: u32,
        color: ColorSpec,
        opacity: Option<f32>,
        tolerance: Option<f32>,
        contiguous: Option<bool>,
        antialias: Option<bool>,
        sample: Option<String>,
    },
    /// Select › Select and Mask.
    SelectRefine {
        radius: Option<f64>,
        contrast: Option<f32>,
        shift: Option<f64>,
        feather: Option<f64>,
        output: Option<String>,
    },
    /// Which history state the History Brush paints from (0 = as opened).
    HistorySource {
        position: usize,
    },
    /// Select › Transform Selection: the outline only, about its centre.
    SelectTransform {
        scale: Option<f64>,
        scale_x: Option<f64>,
        scale_y: Option<f64>,
        rotate: Option<f64>,
        dx: Option<f64>,
        dy: Option<f64>,
    },
    /// Align the selected layers (or one layer to the canvas).
    LayerAlign {
        edge: String,
    },
    /// Space three or more selected layers evenly.
    LayerDistribute {
        direction: String,
    },
    /// Select › Selection to Shape Layer: the outline as an editable shape.
    SelectToShape {
        color: Option<ColorSpec>,
        /// Simplification in px (default 1).
        tolerance: Option<f64>,
    },
    /// Image › Adjustments › Auto Tone / Contrast / Color.
    AutoAdjust {
        mode: String,
    },
    /// Edit › Stroke: a line along the selection's edge.
    StrokeSelection {
        width: f64,
        position: Option<String>,
        color: ColorSpec,
        opacity: Option<f32>,
    },
    /// Image › Adjustments: an adjustment applied to the layer's own pixels.
    AdjustmentApply {
        layer: Option<LayerId>,
        adjustment: mosaix_core::Adjustment,
    },
    /// "down" (active onto the one below), "visible" or "flatten".
    LayerMerge {
        /// Default: "selected" when several layers are selected, else "down".
        mode: Option<String>,
    },
    SmartReplace {
        layer: Option<LayerId>,
        path: PathBuf,
    },
    GuideDelete {
        index: usize,
    },
    GuideClear,
    ChannelDelete {
        channel: usize,
    },
    ChannelRename {
        channel: usize,
        name: String,
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
        skew_x: Option<f64>,
        skew_y: Option<f64>,
        dx: Option<f64>,
        dy: Option<f64>,
        pivot_x: Option<f64>,
        pivot_y: Option<f64>,
        /// Distort / perspective: where the corners of the content bounds go.
        corners: Option<[[f64; 2]; 4]>,
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
        filter: mosaix_core::Filter,
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
        #[serde(default)]
        underline: bool,
        #[serde(default)]
        strikethrough: bool,
        align: Option<mosaix_core::vector::TextAlign>,
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
        /// Polygon corners / star points.
        sides: Option<u32>,
        /// Star: inner radius as a share of the outer one.
        inset: Option<f64>,
        fill: Option<ColorSpec>,
        #[serde(default)]
        no_fill: bool,
        stroke_color: Option<ColorSpec>,
        stroke_width: Option<f32>,
        /// Closed paths: inside, center (default) or outside.
        stroke_align: Option<mosaix_core::vector::StrokeAlign>,
        /// Fill with a gradient through these colours (evenly spaced).
        gradient: Option<Vec<ColorSpec>>,
        gradient_style: Option<mosaix_core::gradient::GradientStyle>,
        /// Degrees, 90 (default) = from bottom to top.
        gradient_angle: Option<f32>,
        /// Fill with a pattern from the library (by name).
        pattern: Option<String>,
        pattern_scale: Option<f32>,
        name: Option<String>,
        /// Add to the active shape layer with this path operation (unite,
        /// subtract, intersect, exclude) instead of making a new layer.
        combine: Option<String>,
    },
    /// Layer › Combine Shapes: the selected shape layers become one (the
    /// bottom one keeps its fill and stroke).
    ShapeCombine {
        op: String,
    },
    VectorSet {
        layer: Option<LayerId>,
        changes: Value,
    },
    /// Makes every node of a pen path smooth (or a corner).
    PathSmooth {
        layer: Option<LayerId>,
        #[serde(default = "yes")]
        smooth: bool,
    },
    LayerRasterize {
        layer: Option<LayerId>,
    },
    /// Re-develops a RAW layer.
    RawSet {
        layer: Option<LayerId>,
        white_balance: Option<mosaix_core::raw::WhiteBalance>,
        exposure: Option<f32>,
        highlights: Option<mosaix_core::raw::Highlights>,
        noise: Option<mosaix_core::raw::NoiseReduction>,
        lens: Option<bool>,
        /// Fields given replace the current ones.
        tone: Option<Value>,
    },
    LayerEffectsSet {
        layer: Option<LayerId>,
        effects: Vec<mosaix_core::style::Effect>,
    },
    /// The pattern library (built-ins and the user's).
    PatternList,
    /// Adds an image or GIMP .pat file to the pattern library.
    PatternImport {
        path: PathBuf,
    },
    /// The visible pixels in the selection (or the image) become a pattern.
    PatternDefine {
        name: Option<String>,
    },
    SmartFilterSet {
        layer: Option<LayerId>,
        index: Option<usize>,
        filter: Option<mosaix_core::Filter>,
        enabled: Option<bool>,
        #[serde(default)]
        remove: bool,
        #[serde(default)]
        apply: bool,
    },
    AdjustmentAdd {
        adjustment: mosaix_core::Adjustment,
        name: Option<String>,
    },
    AdjustmentSet {
        layer: Option<LayerId>,
        adjustment: mosaix_core::Adjustment,
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
        color_space: Option<mosaix_io::export::ColorSpace>,
        bit_depth: Option<u8>,
        quality: Option<u8>,
    },
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::DocumentNew { .. } => "document_new",
            Self::DocumentOpen { .. } => "document_open",
            Self::RawInfo { .. } => "raw_info",
            Self::FilterPluginList => "filter_plugin_list",
            Self::ActionList => "action_list",
            Self::SelectSubject { .. } => "select_subject",
            Self::SelectObject { .. } => "select_object",
            Self::RemoveBackground { .. } => "remove_background",
            Self::RemoveArea { .. } => "remove_area",
            Self::ImageUpscale { .. } => "image_upscale",
            Self::AiModelDownload { .. } => "ai_model_download",
            Self::ActionRun { .. } => "action_run",
            Self::FilterPluginInstall { .. } => "filter_plugin_install",
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
            Self::SelectModify { .. } => "select_modify",
            Self::SelectSave { .. } => "select_save",
            Self::SelectLoad { .. } => "select_load",
            Self::ChannelList => "channel_list",
            Self::GuideAdd { .. } => "guide_add",
            Self::GuideList => "guide_list",
            Self::GenerativeFill { .. } => "generative_fill",
            Self::LayerPlace { .. } => "layer_place",
            Self::LayerConvertSmart { .. } => "layer_convert_smart",
            Self::LayerMerge { .. } => "layer_merge",
            Self::LayerVia { .. } => "layer_via",
            Self::LayerDeleteHidden => "layer_delete_hidden",
            Self::ImageRotate { .. } => "image_rotate",
            Self::LayerWarp { .. } => "layer_warp",
            Self::LayerPuppetWarp { .. } => "layer_puppet_warp",
            Self::SelectLayerPixels { .. } => "select_layer_pixels",
            Self::BucketFill { .. } => "bucket_fill",
            Self::StrokeSelection { .. } => "stroke_selection",
            Self::AutoAdjust { .. } => "auto_adjust",
            Self::SelectToShape { .. } => "select_to_shape",
            Self::LayerAlign { .. } => "layer_align",
            Self::SelectTransform { .. } => "select_transform",
            Self::HistorySource { .. } => "history_source",
            Self::SelectRefine { .. } => "select_refine",
            Self::LayerDistribute { .. } => "layer_distribute",
            Self::AdjustmentApply { .. } => "adjustment_apply",
            Self::SmartReplace { .. } => "smart_replace",
            Self::GuideDelete { .. } => "guide_delete",
            Self::GuideClear => "guide_clear",
            Self::ChannelDelete { .. } => "channel_delete",
            Self::ChannelRename { .. } => "channel_rename",
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
            Self::ShapeCombine { .. } => "shape_combine",
            Self::VectorSet { .. } => "vector_set",
            Self::PathSmooth { .. } => "path_smooth",
            Self::LayerRasterize { .. } => "layer_rasterize",
            Self::RawSet { .. } => "raw_set",
            Self::PatternList => "pattern_list",
            Self::PatternImport { .. } => "pattern_import",
            Self::PatternDefine { .. } => "pattern_define",
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
                | Self::GuideList
                | Self::ChannelList
                | Self::SampleColor { .. }
                | Self::RenderPreview { .. }
                | Self::Save { .. }
                | Self::Export { .. }
        )
    }

    pub fn parse(mut v: Value) -> anyhow::Result<Self> {
        resolve_patterns(&mut v)?;
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

/// Pattern overlays may name their pattern ("pattern": "Dots"): replaced
/// by the pattern from the library.
fn resolve_patterns(v: &mut Value) -> anyhow::Result<()> {
    if v["cmd"] == "vector_set" {
        let p = &mut v["changes"]["paint"];
        if p["type"] == "pattern"
            && let Value::String(name) = &p["pattern"]
        {
            let lib = crate::app::patterns::load_library(&crate::app::patterns::pattern_dir());
            let found = crate::app::patterns::find(&lib, name).with_context(|| {
                let names: Vec<&str> = lib.iter().map(|p| p.name.as_str()).collect();
                format!("unknown pattern {name:?}; available: {}", names.join(", "))
            })?;
            p["pattern"] = serde_json::to_value(found)?;
        }
        return Ok(());
    }
    if v["cmd"] != "layer_effects_set" {
        return Ok(());
    }
    let Some(list) = v["effects"].as_array_mut() else { return Ok(()) };
    let mut lib: Option<Vec<mosaix_core::pattern::Pattern>> = None;
    for e in list {
        if e["effect"] != "pattern_overlay" {
            continue;
        }
        let name = match &e["pattern"] {
            Value::String(n) => n.clone(),
            Value::Null => String::new(),
            _ => continue,
        };
        let lib = lib.get_or_insert_with(|| crate::app::patterns::load_library(&crate::app::patterns::pattern_dir()));
        let p = if name.is_empty() {
            lib.first().cloned()
        } else {
            crate::app::patterns::find(lib, &name)
        };
        let p = p.with_context(|| {
            let names: Vec<&str> = lib.iter().map(|p| p.name.as_str()).collect();
            format!("unknown pattern {name:?}; available: {}", names.join(", "))
        })?;
        e["pattern"] = serde_json::to_value(p)?;
    }
    Ok(())
}

fn session(s: &mut Option<Session>) -> anyhow::Result<&mut Session> {
    s.as_mut()
        .context("no document open – use document_new or document_open first")
}

/// Resolves `layer` (default: active), checks it exists and makes it active.
/// A plugin filter named without its source gets it from the library (and
/// default values for parameters not given).
fn resolve_plugin(f: mosaix_core::Filter) -> anyhow::Result<mosaix_core::Filter> {
    use mosaix_core::Filter;
    let Filter::Plugin { name, source, values } = f else { return Ok(f) };
    if !source.is_empty() {
        return Ok(Filter::Plugin { name, source, values });
    }
    let lib = crate::app::filters::load_library(&crate::app::filters::filter_dir());
    let Some(Filter::Plugin { name, source, values: defaults }) = crate::app::filters::find(&lib, &name) else {
        bail!("no filter plugin named {name} (see filter_plugin_list)");
    };
    let spec = mosaix_core::plugin::parse(&source).map_err(|e| anyhow::anyhow!(e))?;
    for k in values.keys() {
        ensure!(spec.params.iter().any(|p| p.id == *k), "{name} has no parameter {k}");
    }
    let mut v = defaults;
    v.extend(values);
    Ok(Filter::Plugin { name, source, values: v })
}

/// Combines a mask with the selection and reports how much is selected.
fn apply_selection(s: &mut Session, mask: CpuMask, op: CombineOp, label: &str) -> anyhow::Result<Output> {
    let new = match s.engine.selection_cpu() {
        Some(old) if op != CombineOp::Replace => mosaix_core::mask::combine(&old, &mask, op),
        None if matches!(op, CombineOp::Subtract | CombineOp::Intersect) => CpuMask::filled(0),
        _ => mask,
    };
    let empty = new.is_empty();
    s.engine.set_selection(Some(new), label).map_err(engine_err)?;
    Ok(Output::json(json!({ "ok": true, "empty": empty })))
}

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
                .filter(|t| matches!(t, mosaix_core::MaskTile::Pixels(_)))
                .count();
            json!({ "selection": { "default": m.default, "tiles": m.tiles.len(), "edge_tiles": partial } })
        }
    }
}

fn channel_names(s: &Session) -> Vec<String> {
    s.engine.doc().channels.iter().map(|c| c.name.clone()).collect()
}

fn yes() -> bool {
    true
}

fn default_upscale() -> u32 {
    2
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

/// Shear by angles in degrees: `x` slants vertical lines, `y` horizontal ones.
fn skew(x: f64, y: f64) -> anyhow::Result<DAffine2> {
    ensure!(x.abs() < 89.0 && y.abs() < 89.0, "skew angles must be between −89° and 89°");
    Ok(DAffine2::from_cols_array(&[1.0, y.to_radians().tan(), x.to_radians().tan(), 1.0, 0.0, 0.0]))
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
        Command::DocumentOpen { path, raw } => {
            ensure!(path.exists(), "{} does not exist", path.display());
            let source = match raw {
                Some(d) => {
                    ensure!(mosaix_io::raw::is_raw(&path), "{} is not a camera RAW file", path.display());
                    Source::Raw(path, d)
                }
                None => Source::File(path),
            };
            *ctx.session = Some(session::load(ctx.gpu, ctx.format, source)?);
            let mut out = Output::json(info(session(ctx.session)?));
            out.replaced_document = true;
            Ok(out)
        }
        Command::AiModelDownload { model } => {
            let m = match model.as_str() {
                "subject" | "birefnet-lite" => &mosaix_ai::SUBJECT,
                "object" | "sam" | "sam2.1-small" => &mosaix_ai::SAM,
                "remove" | "lama" => &mosaix_ai::INPAINT,
                "upscale" | "realesrgan" | "realesrgan-x4v3" => &mosaix_ai::UPSCALE,
                other => bail!("unknown model {other} (subject, object, remove or upscale)"),
            };
            let t0 = std::time::Instant::now();
            let mut last = 0;
            let dir = mosaix_ai::models::download(m, &mut |done, total| {
                let pct = done * 10 / total.max(1);
                if pct != last {
                    last = pct;
                    tracing::info!("{}: {} %", m.name, pct * 10);
                }
            })?;
            Ok(Output::json(json!({ "ok": true, "path": dir, "seconds": t0.elapsed().as_secs() })))
        }
        Command::ActionList => {
            use crate::app::actions;
            let dir = actions::actions_dir();
            Ok(Output::json(json!({
                "folder": dir,
                "actions": actions::list(&dir).iter().map(|e| json!({
                    "name": e.name,
                    "kind": match e.kind { actions::Kind::Macro => "macro", actions::Kind::Script => "script" },
                    "steps": (e.kind == actions::Kind::Macro).then(|| actions::read_macro(&e.path).map(|s| s.len()).ok()).flatten(),
                })).collect::<Vec<_>>(),
                "format": "macro = NAME.jsonl in the folder, one agent command per line (no layer ids: steps act on the active layer); script = executable file, started by the window with MOSAIX_SOCKET set",
            })))
        }
        Command::ActionRun { name } => {
            use crate::app::actions;
            let e = actions::find(&actions::actions_dir(), &name).with_context(|| format!("no action named {name} (see action_list)"))?;
            ensure!(e.kind == actions::Kind::Macro, "{name} is a script; scripts run from the window (File › Actions)");
            let steps = actions::read_macro(&e.path)?;
            for (i, step) in steps.iter().enumerate() {
                let c = Command::parse(step.clone()).with_context(|| format!("step {}", i + 1))?;
                ensure!(!matches!(c, Command::ActionRun { .. }), "step {}: actions cannot run actions", i + 1);
                execute(ctx, c).with_context(|| format!("{name}, step {} ({})", i + 1, step["cmd"].as_str().unwrap_or("?")))?;
            }
            let mut out = Output::json(json!({ "ok": true, "steps": steps.len() }));
            out.replaced_document = steps.iter().any(|s| matches!(s["cmd"].as_str(), Some("document_open" | "document_new")));
            Ok(out)
        }
        Command::FilterPluginList => {
            let lib = crate::app::filters::load_library(&crate::app::filters::filter_dir());
            Ok(Output::json(json!({
                "folder": crate::app::filters::filter_dir(),
                "plugins": lib.iter().map(|p| match &p.result {
                    Ok(mosaix_core::Filter::Plugin { name, source, .. }) => {
                        let spec = mosaix_core::plugin::parse(source).ok();
                        json!({
                            "name": name, "file": p.file, "builtin": p.builtin,
                            "description": spec.as_ref().map(|s| s.description.clone()),
                            "params": spec.map(|s| s.params).unwrap_or_default(),
                        })
                    }
                    Ok(_) => json!({ "file": p.file }),
                    Err(e) => json!({ "file": p.file, "builtin": p.builtin, "error": e }),
                }).collect::<Vec<_>>(),
                "format": format!("WGSL with a header: // @name …, // @description …, // @param id \"Label\" min max default [unit] [log] | bool [true] | color [#rrggbb], // @reach px|param, // @canvas; and fn process(p: vec2<i32>) -> vec4<f32> returning premultiplied linear RGBA. Parameters are functions of their id. Helpers:\n{}", mosaix_core::plugin::PRELUDE_DOC),
            })))
        }
        Command::FilterPluginInstall { source, check_only } => {
            let spec = mosaix_gpu::engine::validate_plugin(&source).map_err(|e| anyhow::anyhow!(e))?;
            if check_only {
                return Ok(Output::json(json!({ "ok": true, "name": spec.name, "params": spec.params })));
            }
            let dir = crate::app::filters::filter_dir();
            std::fs::create_dir_all(&dir)?;
            let stem: String = spec.name.chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
            let path = dir.join(format!("{stem}.wgsl"));
            std::fs::write(&path, &source).with_context(|| path.display().to_string())?;
            Ok(Output::json(json!({ "ok": true, "name": spec.name, "path": path })))
        }
        Command::RawInfo { path } => {
            ensure!(mosaix_io::raw::is_raw(&path), "{} is not a camera RAW file", path.display());
            let f = mosaix_io::raw::RawFile::open(&path)?;
            let i = &f.info;
            Ok(Output::json(json!({
                "camera": i.camera, "lens": i.lens, "iso": i.iso, "exposure_time": i.exposure_time,
                "aperture": i.aperture, "focal_length": i.focal_length, "lens_profile": i.lens_profile,
                "as_shot": i.as_shot.map(|(t, tint)| json!({ "temperature": t.round(), "tint": tint.round() })),
            })))
        }
        Command::PatternList => {
            let lib = crate::app::patterns::load_library(&crate::app::patterns::pattern_dir());
            let builtin = mosaix_core::pattern::Pattern::builtin().len();
            Ok(Output::json(json!({
                "patterns": lib.iter().enumerate().map(|(i, p)| json!({
                    "name": p.name, "width": p.width, "height": p.height, "builtin": i < builtin,
                })).collect::<Vec<_>>(),
                "folder": crate::app::patterns::pattern_dir(),
            })))
        }
        Command::PatternImport { path } => {
            ensure!(path.exists(), "{} does not exist", path.display());
            let names = crate::app::patterns::import(&path, &crate::app::patterns::pattern_dir())?;
            Ok(Output::json(json!({ "ok": true, "patterns": names })))
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
        Command::LayerDuplicate { layer: None } if s.engine.doc().selected_layers().len() > 1 => {
            let ids = s.engine.doc().selected_layers();
            let mut copies = Vec::new();
            edit(s, tr("Duplicate Layers"), |d| {
                copies = d.duplicate_layers(&ids);
                !copies.is_empty()
            })?;
            Ok(Output::json(json!({ "layers": copies })))
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
        Command::LayerDelete { layer: None } if s.engine.doc().selected_layers().len() > 1 => {
            let ids = s.engine.doc().selected_layers();
            edit(s, tr("Delete Layers"), |d| d.remove_layers(&ids)).context("cannot delete the last layer")?;
            ok(s)
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
            fill_opacity,
            lock,
            label,
        } => {
            let id = target_layer(s, layer)?;
            let label = label
                .map(|l| {
                    mosaix_core::doc::LABELS
                        .iter()
                        .position(|n| *n == l)
                        .map(|i| i as u8)
                        .with_context(|| format!("unknown label {l:?}; valid: {}", mosaix_core::doc::LABELS.join(", ")))
                })
                .transpose()?;
            let blend = blend
                .map(|b| {
                    BlendMode::from_key(&b).with_context(|| {
                        let keys: Vec<_> = BlendMode::ALL.iter().map(|m| m.key()).collect();
                        format!("unknown blend mode {b:?}; valid: {}", keys.join(", "))
                    })
                })
                .transpose()?;
            for o in [opacity, fill_opacity].into_iter().flatten() {
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
                if let Some(f) = fill_opacity {
                    l.fill_opacity = f;
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
                if let Some(k) = lock {
                    l.lock = k;
                }
                if let Some(c) = label {
                    l.label = c;
                }
                true
            })?;
            ok(s)
        }
        Command::LayerSelect { layer, also } => {
            target_layer(s, Some(layer))?;
            for id in &also {
                ensure!(s.engine.doc().find(*id).is_some(), "no layer {id}");
            }
            s.engine.select(|d| d.picked = also.iter().copied().filter(|id| *id != layer).collect());
            let sel = s.engine.doc().selected_layers();
            let mut v = ok(s)?;
            v.result["selected_layers"] = json!(sel);
            Ok(v)
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
        Command::LayerGroup { layer: None } if s.engine.doc().selected_layers().len() > 1 => {
            let ids = s.engine.doc().selected_layers();
            let mut id = None;
            edit(s, tr("Group Layers"), |d| {
                id = d.group_layers(&ids);
                id.is_some()
            })?;
            Ok(Output::json(json!({ "group": id })))
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
            pencil,
            history,
            blend,
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
            tone,
            range,
            amount,
        } => {
            pixel_layer(s, layer)?;
            let tone = match tone.as_deref() {
                None => None,
                Some(k) => {
                    use mosaix_gpu::ToneKind as T;
                    let kind = match k {
                        "dodge" => T::Dodge,
                        "burn" => T::Burn,
                        "sponge_saturate" => T::Saturate,
                        "sponge_desaturate" => T::Desaturate,
                        "blur" => T::Blur,
                        "sharpen" => T::Sharpen,
                        "background_erase" => T::BackgroundErase,
                        _ => bail!("tone must be dodge, burn, sponge_saturate, sponge_desaturate, blur, sharpen or background_erase"),
                    };
                    let range = match range.as_deref().unwrap_or(if kind == T::BackgroundErase { "continuous" } else { "midtones" }) {
                        "continuous" if kind == T::BackgroundErase => 0,
                        "once" if kind == T::BackgroundErase => 1,
                        r if kind == T::BackgroundErase => bail!("range for background_erase must be continuous or once, got {r:?}"),
                        "shadows" => 0,
                        "midtones" => 1,
                        "highlights" => 2,
                        r => bail!("range must be shadows, midtones or highlights, got {r:?}"),
                    };
                    ensure!(target.as_deref().is_none_or(|t| t == "pixels"), "retouching brushes work on pixels");
                    Some(mosaix_gpu::Tone { kind, range, amount: amount.unwrap_or(0.5).clamp(0.0, 1.0) })
                }
            };
            not_painting(s)?;
            let target = parse_target(&target)?;
            ensure!(!points.is_empty(), "points must not be empty");
            ensure!(size > 0.0 && size <= 5000.0, "size must be 0–5000 px");
            let (mut color, alpha) = match &color {
                Some(c) => c.linear()?,
                None if eraser || smudge.is_some() || clone_offset.is_some() || spot_heal || tone.is_some() => {
                    ([0.35; 3], 1.0)
                }
                None => bail!("color is required unless eraser, smudge, tone or clone_offset is set"),
            };
            if target == StrokeTarget::Mask {
                let l = linear_to_srgb(0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]);
                color = [l; 3];
            }
            let settings = BrushSettings {
                size,
                hardness: if pencil { 2.0 } else { hardness.unwrap_or(0.8).clamp(0.0, 1.0) },
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
                        Some(mosaix_gpu::CloneSource {
                            offset: [0, 0],
                            all_layers: clone_all_layers,
                            heal: true,
                            spot: true,
                        })
                    } else if history {
                        Some(mosaix_gpu::CloneSource { offset: [0, 0], all_layers: false, heal: false, spot: false })
                    } else {
                        clone_offset.map(|offset| mosaix_gpu::CloneSource {
                            offset,
                            all_layers: clone_all_layers,
                            heal,
                            spot: false,
                        })
                    },
                    tone,
                    keep_alpha: false,
                    history,
                    blend: blend
                        .map(|b| BlendMode::from_key(&b).with_context(|| format!("unknown blend mode {b:?}")))
                        .transpose()?
                        .unwrap_or(BlendMode::Normal),
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
        Command::SelectModify { op, amount } => {
            use mosaix_core::mask::Modify;
            ensure!((0.0..=500.0).contains(&amount), "amount must be 0–500 px");
            let (m, label) = match op.as_str() {
                "expand" => (Modify::Expand(amount), tr("Expand Selection")),
                "contract" => (Modify::Contract(amount), tr("Contract Selection")),
                "border" => (Modify::Border(amount), tr("Border Selection")),
                "smooth" => (Modify::Smooth(amount), tr("Smooth Selection")),
                "feather" => {
                    s.engine.feather_selection(amount, tr("Feather Selection")).map_err(engine_err)?;
                    return Ok(Output::json(selection_info(s)));
                }
                _ => bail!("op must be expand, contract, border, smooth or feather"),
            };
            s.engine.modify_selection(m, label).map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::SelectSave { name } => {
            let index = s.engine.save_selection(&name, tr("Save Selection")).map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "channel": index, "channels": channel_names(s) })))
        }
        Command::SelectLoad { channel, name, mode, invert } => {
            let names = channel_names(s);
            let index = match (channel, name) {
                (Some(i), None) => i,
                (None, Some(n)) => names.iter().position(|c| *c == n).with_context(|| format!("no channel named {n:?}; there are {names:?}"))?,
                _ => bail!("give channel (index) or name"),
            };
            s.engine.load_channel(index, parse_mode(&mode)?, invert, tr("Load Selection")).map_err(engine_err)?;
            Ok(Output::json(selection_info(s)))
        }
        Command::ChannelList => Ok(Output::json(json!({ "channels": channel_names(s) }))),
        Command::GuideAdd { orientation, position } => {
            let vertical = match orientation.as_str() {
                "vertical" => true,
                "horizontal" => false,
                _ => bail!("orientation must be vertical or horizontal"),
            };
            ensure!(position.is_finite() && position.abs() < 1e7, "position out of range");
            let mut g = s.engine.doc().guides.clone();
            g.push(mosaix_core::Guide { vertical, pos: position });
            s.engine.set_guides(g, tr("New Guide"));
            Ok(Output::json(json!({ "guides": s.engine.doc().guides })))
        }
        Command::GuideList => Ok(Output::json(json!({ "guides": s.engine.doc().guides }))),
        Command::GenerativeFill { layer, prompt, new_layer } => {
            let settings = crate::settings::Settings::load_from(&crate::settings::config_path()).generative;
            let onto_new = new_layer.unwrap_or(settings.new_layer) && layer.is_none();
            if !onto_new {
                pixel_layer(s, layer)?;
            }
            ensure!(s.engine.has_selection(), "generative fill needs a selection");
            let provider = settings.provider().map_err(|e| anyhow::anyhow!(e))?;
            let job = if onto_new {
                s.engine.fill_job_new_layer(&crate::ai::generative_layer_name(&prompt), crate::ai::GENERATIVE_CONTEXT, crate::ai::REMOVE_GROW)
            } else {
                s.engine.fill_job_sized(true, crate::ai::GENERATIVE_CONTEXT, crate::ai::REMOVE_GROW)
            }
            .map_err(engine_err)?;
            let r = crate::ai::generative_fill(job, &provider, &prompt)?;
            let ms = r.ms;
            ensure!(s.engine.apply_fill(r, tr("Generative Fill")), "the layer has changed in the meantime");
            let mut v = ok(s)?;
            v.result["ms"] = json!(ms.round());
            Ok(v)
        }
        Command::LayerPlace { path } => {
            let bytes = std::fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let id = s.engine.place_smart(bytes, &name, tr("Place Embedded")).map_err(engine_err)?;
            let bounds = s.engine.content_bounds(id).map(bounds_json);
            Ok(Output::json(json!({ "ok": true, "active_layer": id, "bounds": bounds })))
        }
        Command::LayerConvertSmart { layer } => {
            let id = pixel_layer(s, layer)?;
            s.engine.convert_to_smart(id, tr("Convert to Smart Object")).map_err(engine_err)?;
            ok(s)
        }
        Command::LayerPuppetWarp { pins } => {
            ensure!(!pins.is_empty(), "pins must not be empty");
            let id = pixel_layer(s, None)?;
            not_painting(s)?;
            let mut p = crate::puppet::Puppet::begin(&mut s.engine).map_err(|e| anyhow::anyhow!(e))?;
            for (k, pin) in pins.iter().enumerate() {
                let from = DVec2::new(pin.from[0], pin.from[1]);
                let Some(i) = p.add_pin(from) else {
                    s.engine.liquify_cancel();
                    bail!("pin {k} at {:?} is not on the layer's pixels", pin.from);
                };
                p.move_pin(i, DVec2::new(pin.to[0], pin.to[1]));
            }
            p.update(&mut s.engine);
            s.engine.liquify_commit(tr("Puppet Warp")).map_err(engine_err)?;
            let bounds = s.engine.content_bounds(id).map(bounds_json);
            Ok(Output::json(json!({ "ok": true, "bounds": bounds })))
        }
        Command::LayerWarp { offsets } => {
            ensure!(offsets.len() == 16, "offsets needs 16 [dx, dy] pairs (4 × 4 grid, row by row)");
            let id = pixel_layer(s, None)?;
            let (x0, y0, x1, y1) = s.engine.content_bounds(id).context("the layer is empty")?;
            let m = 8.0;
            let mut p = mosaix_core::warp::Patch::identity(DVec2::new(x0 as f64 - m, y0 as f64 - m), DVec2::new(x1 as f64 + m, y1 as f64 + m));
            for (k, d) in offsets.iter().enumerate() {
                p.pts[k / 4][k % 4] += DVec2::new(d[0], d[1]);
            }
            s.engine.liquify_begin().map_err(engine_err)?;
            s.engine.liquify_set_warp(&p);
            s.engine.liquify_commit(tr("Warp")).map_err(engine_err)?;
            let bounds = s.engine.content_bounds(id).map(bounds_json);
            Ok(Output::json(json!({ "ok": true, "bounds": bounds })))
        }
        Command::ImageRotate { angle } => {
            ensure!(angle.is_finite() && angle.abs() <= 360.0, "angle must be −360…360");
            not_painting(s)?;
            s.engine.rotate_all(angle, tr("Straighten")).map_err(engine_err)?;
            ok(s)
        }
        Command::LayerDeleteHidden => {
            let ids = s.engine.doc().hidden_layers();
            ensure!(!ids.is_empty(), "no hidden layers");
            edit(s, tr("Delete Hidden Layers"), |d| d.remove_layers(&ids)).context("cannot delete the last layer")?;
            let mut v = ok(s)?;
            v.result["deleted"] = json!(ids.len());
            Ok(v)
        }
        Command::LayerVia { cut } => {
            pixel_layer(s, None)?;
            let label = if cut { tr("Layer via Cut") } else { tr("Layer via Copy") };
            s.engine.layer_via(cut, label).map_err(engine_err)?;
            ok(s)
        }
        Command::SelectLayerPixels { layer, mode } => {
            let id = target_layer(s, layer)?;
            s.engine.select_layer_pixels(id, parse_mode(&mode)?, tr("Select Layer Pixels")).map_err(engine_err)?;
            ok(s)
        }
        Command::BucketFill { x, y, color, opacity, tolerance, contiguous, antialias, sample } => {
            pixel_layer(s, None)?;
            let tolerance = tolerance.unwrap_or(32.0);
            ensure!((0.0..=255.0).contains(&tolerance), "tolerance must be 0–255");
            let n = s
                .engine
                .bucket_fill(
                    x,
                    y,
                    tolerance,
                    contiguous.unwrap_or(true),
                    antialias.unwrap_or(true),
                    parse_sample(&sample)?,
                    premul(&color, StrokeTarget::Pixels)?,
                    opacity.unwrap_or(1.0).clamp(0.0, 1.0),
                    tr("Paint Bucket"),
                )
                .map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "tiles": n })))
        }
        Command::SelectRefine { radius, contrast, shift, feather, output } => {
            let to_mask = match output.as_deref() {
                None | Some("selection") => false,
                Some("mask") => true,
                Some(o) => bail!("unknown output {o:?} (selection, mask)"),
            };
            if to_mask {
                target_layer(s, None)?;
            }
            s.engine
                .select_and_mask(
                    radius.unwrap_or(10.0).clamp(0.0, 500.0),
                    contrast.unwrap_or(0.0).clamp(0.0, 1.0),
                    shift.unwrap_or(0.0).clamp(-500.0, 500.0),
                    feather.unwrap_or(0.0).clamp(0.0, 500.0),
                    to_mask,
                    tr("Select and Mask"),
                )
                .map_err(engine_err)?;
            ok(s)
        }
        Command::HistorySource { position } => {
            let (labels, _) = s.engine.history_labels();
            ensure!(position <= labels.len(), "no history position {position}; there are {} steps", labels.len());
            s.engine.set_history_source(position);
            ok(s)
        }
        Command::SelectTransform { scale, scale_x, scale_y, rotate, dx, dy } => {
            let b = s.engine.selection_bounds().context("nothing is selected")?;
            let pivot = DVec2::new((b.0 + b.2) as f64 * 0.5, (b.1 + b.3) as f64 * 0.5);
            let k = DVec2::new(scale_x.or(scale).unwrap_or(1.0), scale_y.or(scale).unwrap_or(1.0));
            ensure!(k.x != 0.0 && k.y != 0.0, "scale must not be 0");
            let fwd = DAffine2::from_translation(DVec2::new(dx.unwrap_or(0.0), dy.unwrap_or(0.0)) + pivot)
                * DAffine2::from_angle(rotate.unwrap_or(0.0).to_radians())
                * DAffine2::from_scale(k)
                * DAffine2::from_translation(-pivot);
            s.engine.transform_selection(fwd, tr("Transform Selection")).map_err(engine_err)?;
            ok(s)
        }
        Command::LayerAlign { edge } => {
            use mosaix_gpu::Align as A;
            let edge = match edge.as_str() {
                "left" => A::Left,
                "center_x" | "centre_x" => A::CenterX,
                "right" => A::Right,
                "top" => A::Top,
                "center_y" | "centre_y" => A::CenterY,
                "bottom" => A::Bottom,
                e => bail!("unknown edge {e:?} (left, center_x, right, top, center_y, bottom)"),
            };
            not_painting(s)?;
            s.engine.align_layers(edge, tr("Align Layers")).map_err(engine_err)?;
            ok(s)
        }
        Command::LayerDistribute { direction } => {
            let horizontal = match direction.as_str() {
                "horizontal" => true,
                "vertical" => false,
                d => bail!("unknown direction {d:?} (horizontal, vertical)"),
            };
            not_painting(s)?;
            s.engine.distribute_layers(horizontal, tr("Distribute Layers")).map_err(engine_err)?;
            ok(s)
        }
        Command::SelectToShape { color, tolerance } => {
            let sel = s.engine.selection_cpu().context("nothing is selected")?;
            let (w, h) = s.engine.size();
            let loops = mosaix_core::mask::trace(&sel, tolerance.unwrap_or(1.0).clamp(0.1, 50.0), w, h);
            ensure!(!loops.is_empty(), "the selection has no outline");
            let mut d = String::new();
            for l in &loops {
                for (i, p) in l.iter().enumerate() {
                    d += &format!("{}{} {} ", if i == 0 { "M" } else { "L" }, p[0], p[1]);
                }
                d += "Z ";
            }
            let fill = match &color {
                Some(c) => {
                    let (rgb, a) = c.linear()?;
                    [rgb[0], rgb[1], rgb[2], a]
                }
                None => [0.0, 0.0, 0.0, 1.0],
            };
            let v = mosaix_core::VectorLayer::shape(mosaix_core::vector::ShapeSpec {
                geometry: mosaix_core::vector::Geometry::Path { d: d.trim_end().to_string() },
                fill: Some(fill),
                stroke: None,
                paint: None,
                parts: Vec::new(),
            });
            let id = s.engine.add_vector(v, tr("Shape from Selection")).context("not possible right now")?;
            Ok(Output::json(json!({ "ok": true, "layer": id, "outlines": loops.len() })))
        }
        Command::AutoAdjust { mode } => {
            use mosaix_core::adjust::AutoMode as M;
            let (m, label) = match mode.as_str() {
                "tone" => (M::Tone, tr("Auto Tone")),
                "contrast" => (M::Contrast, tr("Auto Contrast")),
                "color" | "colour" => (M::Color, tr("Auto Color")),
                m => bail!("unknown mode {m:?} (tone, contrast, color)"),
            };
            pixel_layer(s, None)?;
            let a = s.engine.auto_adjust(m, label).map_err(engine_err)?;
            let mut v = ok(s)?;
            v.result["levels"] = serde_json::to_value(&a).unwrap_or_default();
            Ok(v)
        }
        Command::StrokeSelection { width, position, color, opacity } => {
            pixel_layer(s, None)?;
            use mosaix_core::style::StrokePosition as P;
            let position = match position.as_deref() {
                None | Some("outside") => P::Outside,
                Some("inside") => P::Inside,
                Some("center") | Some("centre") => P::Center,
                Some(p) => bail!("unknown position {p:?} (inside, center, outside)"),
            };
            let n = s
                .engine
                .stroke_selection(width, position, premul(&color, StrokeTarget::Pixels)?, opacity.unwrap_or(1.0).clamp(0.0, 1.0), tr("Stroke"))
                .map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "tiles": n })))
        }
        Command::AdjustmentApply { layer, mut adjustment } => {
            adjustment.load_files().map_err(|e| anyhow::anyhow!(e))?;
            pixel_layer(s, layer)?;
            let label = tr!("Adjust: {name}", name = adjustment.name());
            s.engine.apply_adjustment(&adjustment, &label).map_err(engine_err)?;
            ok(s)
        }
        Command::LayerMerge { mode } => {
            let mode = mode.unwrap_or_else(|| {
                if s.engine.doc().selected_layers().len() > 1 {
                    "selected".into()
                } else if s.engine.doc().active_layer().is_some_and(|l| l.is_group()) {
                    "group".into()
                } else {
                    "down".into()
                }
            });
            let kind = match mode.as_str() {
                "selected" => mosaix_core::MergeKind::Selected,
                "group" => mosaix_core::MergeKind::Group,
                "down" => mosaix_core::MergeKind::Down,
                "visible" => mosaix_core::MergeKind::Visible,
                "flatten" => mosaix_core::MergeKind::Flatten,
                m => bail!("unknown merge mode {m:?} (down, selected, group, visible, flatten)"),
            };
            let label = match kind {
                mosaix_core::MergeKind::Down => tr("Merge Down"),
                mosaix_core::MergeKind::Selected => tr("Merge Layers"),
                mosaix_core::MergeKind::Group => tr("Merge Group"),
                mosaix_core::MergeKind::Visible => tr("Merge Visible"),
                mosaix_core::MergeKind::Flatten => tr("Flatten Image"),
            };
            s.engine.merge(kind, label).map_err(engine_err)?;
            ok(s)
        }
        Command::SmartReplace { layer, path } => {
            let id = target_layer(s, layer)?;
            let bytes = std::fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            s.engine.replace_smart(id, bytes, &name, tr("Replace Contents")).map_err(engine_err)?;
            ok(s)
        }
        Command::GuideDelete { index } => {
            let mut g = s.engine.doc().guides.clone();
            ensure!(index < g.len(), "no guide {index}; there are {}", g.len());
            g.remove(index);
            s.engine.set_guides(g, tr("Delete Guide"));
            Ok(Output::json(json!({ "guides": s.engine.doc().guides })))
        }
        Command::GuideClear => {
            s.engine.set_guides(Vec::new(), tr("Clear Guides"));
            Ok(Output::json(json!({ "guides": [] })))
        }
        Command::ChannelDelete { channel } => {
            s.engine.delete_channel(channel, tr("Delete Channel")).map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "channels": channel_names(s) })))
        }
        Command::ChannelRename { channel, name } => {
            s.engine.rename_channel(channel, &name, tr("Rename Channel")).map_err(engine_err)?;
            Ok(Output::json(json!({ "ok": true, "channels": channel_names(s) })))
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
                _ => mosaix_core::mask::CpuMask::filled(255),
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
            skew_x,
            skew_y,
            dx,
            dy,
            pivot_x,
            pivot_y,
            corners,
            interp,
            use_selection,
        } => {
            let id = pixel_layer(s, layer)?;
            let interp = parse_interp(&interp)?;
            let use_selection = use_selection.unwrap_or(true);
            if let Some(c) = corners {
                ensure!(
                    scale.is_none() && scale_x.is_none() && scale_y.is_none() && rotate.is_none() && skew_x.is_none() && skew_y.is_none() && dx.is_none() && dy.is_none(),
                    "corners replaces scale, rotate, skew and dx/dy: give one or the other"
                );
                let b = s.engine.content_bounds(id).context("the layer (or selection) is empty")?;
                let quad = c.map(|[x, y]| DVec2::new(x, y));
                let w = mosaix_gpu::Warp::from_quad(DVec2::new(b.0 as f64, b.1 as f64), DVec2::new(b.2 as f64, b.3 as f64), quad)
                    .context("the corners must form a convex shape, clockwise from the top left")?;
                s.engine.transform_layer(w, interp, use_selection, tr("Distort")).map_err(engine_err)?;
                let bounds = s.engine.content_bounds(id).map(bounds_json);
                return Ok(Output::json(json!({ "ok": true, "active_layer": id, "bounds": bounds })));
            }
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
                * skew(skew_x.unwrap_or(0.0), skew_y.unwrap_or(0.0))?
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
            let filter = resolve_plugin(filter)?;
            if let mosaix_core::Filter::Plugin { source, .. } = &filter {
                s.engine.check_plugin(source).map_err(|e| anyhow::anyhow!("filter plugin: {e}"))?;
            }
            let label = filter.name();
            if smart {
                s.engine
                    .add_smart_filter(&filter, &label)
                    .map_err(engine_err)?;
            } else {
                s.engine.apply_filter(&filter, &label).map_err(engine_err)?;
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
            underline,
            strikethrough,
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
                mosaix_core::vector::TextSpec::new(&text, x, y, size, [rgb[0], rgb[1], rgb[2], a]);
            if let Some(f) = font {
                t.font = f;
            }
            t.weight = weight.unwrap_or(t.weight).clamp(100.0, 900.0);
            t.italic = italic;
            t.underline = underline;
            t.strikethrough = strikethrough;
            t.align = align.unwrap_or_default();
            t.line_height = line_height.unwrap_or(t.line_height);
            t.letter_spacing = letter_spacing.unwrap_or(0.0);
            t.width = width;
            let id = s
                .engine
                .add_vector(mosaix_core::VectorLayer::text(t), tr("Text"))
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
            stroke_align,
            gradient,
            gradient_style,
            gradient_angle,
            pattern,
            pattern_scale,
            name,
            sides,
            inset,
            combine,
        } => {
            use mosaix_core::vector::{Geometry, Paint, ShapeSpec, StrokeStyle};
            let combine = match combine.as_deref() {
                None => None,
                Some(k) => Some(mosaix_core::vector::PathOp::from_key(k).with_context(|| format!("combine must be unite, subtract, intersect or exclude, got {k:?}"))?),
            };
            // A line is an open polygon with a stroke (and no fill unless asked).
            let is_line = shape == "line";
            let no_fill = no_fill || (is_line && fill.is_none());
            let stroke_width = if is_line { stroke_width.or(Some(4.0)) } else { stroke_width };
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
                "line" => {
                    let p = points.context("line needs points [[x0, y0], [x1, y1]]")?;
                    ensure!(p.len() >= 2, "line needs two points");
                    Geometry::Polygon { points: p, closed: false }
                }
                "regular_polygon" | "star" => Geometry::Polygon {
                    points: mosaix_core::vector::regular_polygon(
                        need(x, "x")?,
                        need(y, "y")?,
                        need(width, "width")?,
                        need(height, "height")?,
                        sides.unwrap_or(5),
                        (shape == "star").then(|| inset.unwrap_or(0.5)),
                    ),
                    closed: true,
                },
                o if mosaix_core::vector::SHAPE_PRESETS.contains(&o) => Geometry::Polygon {
                    points: mosaix_core::vector::preset_shape(o, need(x, "x")?, need(y, "y")?, need(width, "width")?, need(height, "height")?).unwrap(),
                    closed: true,
                },
                o => bail!(
                    "shape must be rect, ellipse, polygon, path, line, regular_polygon, star or one of {}, got {o:?}",
                    mosaix_core::vector::SHAPE_PRESETS.join(", ")
                ),
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
                    align: stroke_align.unwrap_or_default(),
                    ..StrokeStyle::new(
                        c.as_ref().map(lin).transpose()?.unwrap_or([0.0, 0.0, 0.0, 1.0]),
                        w.unwrap_or(2.0).max(0.1),
                    )
                }),
            };
            ensure!(
                fill.is_some() || stroke.is_some(),
                "a shape needs a fill or a stroke"
            );
            ensure!(gradient.is_none() || pattern.is_none(), "either a gradient or a pattern");
            ensure!(!(no_fill && (gradient.is_some() || pattern.is_some())), "no_fill and a gradient or pattern exclude each other");
            let paint = match (gradient, pattern) {
                (Some(colors), _) => {
                    ensure!(colors.len() >= 2, "a gradient needs at least two colours");
                    let n = (colors.len() - 1) as f32;
                    let mut g = mosaix_core::gradient::Gradient { colors: Vec::new(), alphas: Vec::new(), smoothness: 0.0 };
                    for (i, c) in colors.iter().enumerate() {
                        let (rgb, a) = c.linear()?;
                        let at = i as f32 / n;
                        g.colors.push(mosaix_core::gradient::ColorStop { at, color: rgb, mid: 0.5 });
                        g.alphas.push(mosaix_core::gradient::AlphaStop { at, opacity: a, mid: 0.5 });
                    }
                    if g.alphas.iter().all(|a| a.opacity >= 1.0) {
                        g.alphas.clear();
                    }
                    let mut p = Paint::gradient(g);
                    if let Paint::Gradient { style, angle, .. } = &mut p {
                        *style = gradient_style.unwrap_or_default();
                        *angle = gradient_angle.unwrap_or(90.0);
                    }
                    Some(p)
                }
                (None, Some(name)) => {
                    let lib = crate::app::patterns::load_library(&crate::app::patterns::pattern_dir());
                    let p = crate::app::patterns::find(&lib, &name).with_context(|| {
                        let names: Vec<&str> = lib.iter().map(|p| p.name.as_str()).collect();
                        format!("unknown pattern {name:?}; available: {}", names.join(", "))
                    })?;
                    Some(Paint::Pattern { pattern: p, scale: pattern_scale.unwrap_or(1.0).max(0.01), offset: [0.0; 2] })
                }
                (None, None) => None,
            };
            if let Some(op) = combine {
                let id = s.engine.doc().active;
                let mut v = s
                    .engine
                    .doc()
                    .find(id)
                    .and_then(|l| l.vector().cloned())
                    .filter(|v| matches!(v.kind, mosaix_core::vector::VectorKind::Shape(_)))
                    .context("combine needs a shape layer as the active layer")?;
                ensure!(mosaix_vector::add_shape_part(&mut v, &geometry, op), "the shape is empty or the path data is invalid");
                s.engine.set_vector(id, v, op.name(), None).map_err(engine_err)?;
                return vector_ok(s, id);
            }
            let v = mosaix_core::VectorLayer::shape(ShapeSpec {
                geometry,
                fill,
                stroke,
                paint,
                parts: Vec::new(),
            });
            ensure!(
                mosaix_vector::bounds(&v).is_some(),
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
        Command::ShapeCombine { op } => {
            let op = mosaix_core::vector::PathOp::from_key(&op).with_context(|| format!("op must be unite, subtract, intersect or exclude, got {op:?}"))?;
            not_painting(s)?;
            let doc = s.engine.doc();
            let ids: Vec<LayerId> = doc
                .selected_layers()
                .into_iter()
                .filter(|id| doc.find(*id).and_then(|l| l.vector()).is_some_and(|v| matches!(v.kind, mosaix_core::vector::VectorKind::Shape(_))))
                .collect();
            ensure!(ids.len() >= 2, "select two or more shape layers (layer_select with also)");
            let base = ids[0];
            let layers: Vec<mosaix_core::VectorLayer> = ids.iter().map(|id| doc.find(*id).unwrap().vector().unwrap().clone()).collect();
            let others: Vec<&mosaix_core::VectorLayer> = layers[1..].iter().collect();
            let v = mosaix_vector::combine_shapes(&layers[0], &others, op).context("only shape layers can be combined")?;
            let rest = ids[1..].to_vec();
            s.engine.edit(op.name(), |d| {
                d.find_mut(base).unwrap().content = mosaix_core::doc::Content::Vector(v);
                d.remove_layers(&rest);
                d.active = base;
                d.picked.clear();
                true
            });
            vector_ok(s, base)
        }
        Command::PathSmooth { layer, smooth } => {
            let id = target_layer(s, layer)?;
            let mut vl = s
                .engine
                .doc()
                .find(id)
                .and_then(|l| l.vector().cloned())
                .with_context(|| format!("layer {id} is not a shape layer"))?;
            let mosaix_core::vector::VectorKind::Shape(spec) = &mut vl.kind else {
                bail!("layer {id} is not a shape layer");
            };
            ensure!(
                mosaix_core::vector::smooth_geometry(&mut spec.geometry, smooth),
                "layer {id} is not a pen path or polygon"
            );
            s.engine.set_vector(id, vl, tr("Edit Path"), None).map_err(engine_err)?;
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
            let new: mosaix_core::VectorLayer =
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
        Command::RawSet { layer, white_balance, exposure, highlights, noise, lens, tone } => {
            let id = target_layer(s, layer)?;
            let mut d = s
                .engine
                .doc()
                .find(id)
                .and_then(|l| l.raw().map(|r| r.develop))
                .with_context(|| format!("layer {id} is not a RAW layer"))?;
            if let Some(wb) = white_balance {
                d.white_balance = wb;
            }
            if let Some(e) = exposure {
                ensure!((-10.0..=10.0).contains(&e), "exposure must be −10…10 stops");
                d.exposure = e;
            }
            if let Some(h) = highlights {
                d.highlights = h;
            }
            if let Some(l) = lens {
                d.lens = l;
            }
            if let Some(t) = tone {
                let mut v = serde_json::to_value(d.tone)?;
                merge_json(&mut v, t)?;
                let t: mosaix_core::raw::Tone = serde_json::from_value(v).context("invalid tone")?;
                ensure!(
                    [t.contrast, t.highlights, t.shadows, t.vibrance, t.saturation].iter().all(|v| (-100.0..=100.0).contains(v)),
                    "tone values must be −100…100"
                );
                d.tone = t;
            }
            if let Some(n) = noise {
                ensure!(
                    (0.0..=100.0).contains(&n.luminance) && (0.0..=100.0).contains(&n.color),
                    "noise strengths must be 0–100"
                );
                d.noise = n;
            }
            s.engine.set_raw(id, d, tr("Develop RAW"), None).map_err(engine_err)?;
            s.engine.flush();
            let sum = s.engine.raw_summary(id);
            Ok(Output::json(json!({
                "layer": id,
                "develop": d,
                "temperature": sum.as_ref().map(|m| m.temperature(d.white_balance).0.round()),
                "tint": sum.as_ref().map(|m| m.temperature(d.white_balance).1.round()),
                "lens_profile": sum.as_ref().and_then(|m| m.info.lens_profile.clone()),
            })))
        }
        Command::SelectSubject { mode } => {
            let op = parse_mode(&mode)?;
            let snap = crate::ai::snapshot(s);
            let mask = crate::ai::subject(&snap)?;
            apply_selection(s, mask, op, tr("Select Subject"))
        }
        Command::SelectObject { points, negative, mode } => {
            let op = parse_mode(&mode)?;
            ensure!(!points.is_empty(), "give at least one point");
            let mut pts = Vec::new();
            for (list, pos) in [(&points, true), (&negative, false)] {
                for p in list {
                    ensure!(p.len() == 2, "points are [x, y]");
                    pts.push((p[0], p[1], pos));
                }
            }
            let snap = crate::ai::snapshot(s);
            let mask = crate::ai::object(&snap, &pts)?;
            apply_selection(s, mask, op, tr("Select Object"))
        }
        Command::RemoveBackground { layer } => {
            let id = target_layer(s, layer)?;
            let snap = crate::ai::snapshot(s);
            let mask = crate::ai::subject(&snap)?;
            ensure!(!mask.is_empty(), "no subject found");
            s.engine.set_layer_mask(id, Some(mask), tr("Remove Background")).map_err(engine_err)?;
            ok(s)
        }
        Command::ImageUpscale { factor } => {
            ensure!(factor == 2 || factor == 4, "factor is 2 or 4");
            let job = s.engine.upscale_job(factor).map_err(engine_err)?;
            let r = crate::ai::upscale(job)?;
            let ms = r.ms;
            s.engine.apply_upscale(r, tr("Upscale (AI)")).map_err(engine_err)?;
            let (w, h) = s.engine.size();
            Ok(Output::json(json!({ "ok": true, "width": w, "height": h, "ms": ms.round() })))
        }
        Command::RemoveArea { layer, all_layers, expand } => {
            pixel_layer(s, layer)?;
            let grow = expand.map_or(crate::ai::REMOVE_GROW, |e| e.min(48) as usize);
            let job = s.engine.fill_job_sized(all_layers, crate::ai::INPAINT_CONTEXT, grow).map_err(engine_err)?;
            let r = crate::ai::inpaint(job)?;
            let ms = r.ms;
            ensure!(s.engine.apply_fill(r, tr("Remove")), "the layer has changed in the meantime");
            let mut v = ok(s)?;
            v.result["ms"] = json!(ms.round());
            Ok(v)
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
        Command::PatternList
        | Command::PatternImport { .. }
        | Command::RawInfo { .. }
        | Command::FilterPluginList
        | Command::FilterPluginInstall { .. }
        | Command::ActionList
        | Command::ActionRun { .. }
        | Command::AiModelDownload { .. } => unreachable!(),
        Command::PatternDefine { name } => {
            let c = s.engine.copy_pixels(true).map_err(engine_err)?;
            let img = crate::app::clipboard::to_rgba8(c.w, c.h, &c.px);
            let name = crate::app::patterns::save(&crate::app::patterns::pattern_dir(), name.as_deref().unwrap_or("Pattern"), img)?;
            Ok(Output::json(json!({ "ok": true, "pattern": name, "width": c.w, "height": c.h })))
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
                if st.tool == mosaix_gpu::LiquifyTool::Push {
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
        Command::AdjustmentAdd { mut adjustment, name } => {
            adjustment.load_files().map_err(|e| anyhow::anyhow!(e))?;
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
        Command::AdjustmentSet { layer, mut adjustment } => {
            adjustment.load_files().map_err(|e| anyhow::anyhow!(e))?;
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
            let d = mosaix_io::export::ExportOptions::default();
            let opts = mosaix_io::export::ExportOptions {
                space: color_space.unwrap_or(d.space),
                sixteen_bit: match bit_depth {
                    None | Some(8) => false,
                    Some(16) => true,
                    Some(b) => anyhow::bail!("bit_depth {b}: must be 8 or 16"),
                },
                quality: quality.unwrap_or(d.quality).clamp(1, 100),
            };
            let sixteen = opts.sixteen_bit && mosaix_io::export::supports_16bit(&path);
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
    pub tool: mosaix_gpu::LiquifyTool,
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
        .and_then(mosaix_vector::bounds)
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
    fn layers(ls: &[mosaix_core::Layer<Tile>]) -> Vec<Value> {
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
                if l.fill_opacity < 1.0 {
                    v["fill_opacity"] = json!(l.fill_opacity);
                }
                if l.label > 0 {
                    v["label"] = json!(mosaix_core::doc::LABELS[l.label.min(7) as usize]);
                }
                if l.lock.any() {
                    v["lock"] = serde_json::to_value(l.lock).unwrap_or_default();
                }
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
                    Content::Raw(r) => {
                        v["type"] = "raw".into();
                        v["raw"] = json!({ "file": r.data.name, "develop": r.develop, "transform": r.transform });
                    }
                    Content::Vector(vl) => {
                        v["type"] = "vector".into();
                        v["vector"] = serde_json::to_value(vl).unwrap_or_default();
                        if let Some(b) = mosaix_vector::bounds(vl) {
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
    if mosaix_io::raw::is_raw(path) {
        // A RAW file stays a RAW layer (developed as shot, editable with raw_set).
        let fname = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let data = mosaix_core::raw::RawData::new(
            std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?,
            fname,
        );
        let mut r = mosaix_core::raw::RawLayer::new(data, Default::default());
        r.transform[4] = x as f64;
        r.transform[5] = y as f64;
        let name = name.unwrap_or_else(|| path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        let id = s.engine.add_raw(r, &name, tr("Place Image")).context("cannot add the layer")?;
        s.engine.flush();
        ensure!(s.engine.raw_summary(id).is_some(), "cannot develop {}", path.display());
        return Ok(Output::json(json!({ "layer": id })));
    }
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
/// The whole visible image as sRGB RGBA, its longest side between
/// `max / 2` and `max` (read at a mip level, no further scaling).
pub(crate) fn read_composite(s: &mut Session, max: u32) -> RgbaImage {
    let (dw, dh) = s.engine.size();
    let longest = dw.max(dh) as f64;
    let mip = (longest / max as f64).log2().ceil().clamp(0.0, 8.0) as u32;
    let scale = 1i64 << mip;
    s.engine.flush();
    let have: std::collections::HashSet<_> = s.engine.composite_coords().into_iter().collect();
    let coords: Vec<TileCoord> = tiles_in_rect((0.0, 0.0), (dw as f64, dh as f64), dw, dh)
        .filter(|c| have.contains(c))
        .collect();
    let (iw, ih) = ((dw as i64 + scale - 1) / scale, (dh as i64 + scale - 1) / scale);
    let mut img = RgbaImage::new(iw.max(1) as u32, ih.max(1) as u32);
    for batch in coords.chunks(256) {
        let px = s.engine.read_composite_mip(batch, mip);
        let tiles: Vec<_> = batch.iter().copied().zip(px).collect();
        blit_region(&mut img, &tiles, TILE_SIZE >> mip, (0, 0));
    }
    img
}

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

#[cfg(test)]
mod pattern_command_tests {
    use super::*;

    #[test]
    fn pattern_overlay_by_name() {
        let name = mosaix_core::pattern::Pattern::builtin()[1].name.clone();
        let c = Command::parse(json!({
            "cmd": "layer_effects_set",
            "effects": [{"effect": "pattern_overlay", "pattern": name.to_uppercase(), "opacity": 0.5}]
        }))
        .unwrap();
        let Command::LayerEffectsSet { effects, .. } = c else { panic!() };
        let mosaix_core::style::EffectKind::PatternOverlay { pattern, scale, .. } = &effects[0].kind else { panic!() };
        assert_eq!((pattern.name.as_str(), *scale), (name.as_str(), 1.0));
        let err = Command::parse(json!({"cmd": "layer_effects_set", "effects": [{"effect": "pattern_overlay", "pattern": "nope", "opacity": 1}]}));
        assert!(err.unwrap_err().to_string().contains("available"));
    }
}

