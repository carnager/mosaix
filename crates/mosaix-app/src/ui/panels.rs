//! Panel layout. `build` reads a [`Model`] (snapshot of the app) and returns
//! [`Action`]s; it never touches the engine directly.

use egui::{Align, Color32, Layout, RichText, Sense, Stroke, Vec2};
use mosaix_core::{BlendMode, LayerId, TIP_NAMES, tr};

use super::icons::{self, Icon};
use super::palette::{self, COMMANDS, Cmd, MENUS};

/// Tools as shown in the toolbar: (id, name, shortcut); English, translated
/// when shown.
pub const TOOLS: [(ToolId, &str, &str); 20] = [
    (ToolId::Move, "Move", "V"),
    (ToolId::Marquee, "Rectangular Marquee", "M"),
    (ToolId::Ellipse, "Elliptical Marquee", "M"),
    (ToolId::Lasso, "Lasso", "L"),
    (ToolId::Wand, "Magic Wand", "W"),
    (ToolId::ObjectSelect, "Object Selection (AI)", "Shift+W"),
    (ToolId::Crop, "Crop", "C"),
    (ToolId::Brush, "Brush", "B"),
    (ToolId::Smudge, "Smudge", "U"),
    (ToolId::Clone, "Clone Stamp", "S"),
    (ToolId::SpotHeal, "Spot Healing Brush", "J"),
    (ToolId::Remove, "Remove (AI)", "J"),
    (ToolId::Heal, "Healing Brush", "J"),
    (ToolId::Eraser, "Eraser", "E"),
    (ToolId::Text, "Text", "T"),
    (ToolId::Shape, "Shape (Rectangle / Ellipse)", "Shift+U"),
    (ToolId::Pen, "Pen", "P"),
    (ToolId::Eyedropper, "Eyedropper", "I"),
    (ToolId::Gradient, "Linear Gradient", "G"),
    (ToolId::Radial, "Radial Gradient", "G"),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolId {
    Move,
    Marquee,
    Ellipse,
    Lasso,
    Wand,
    Crop,
    Brush,
    Smudge,
    Clone,
    Heal,
    SpotHeal,
    Text,
    Shape,
    Pen,
    Eyedropper,
    Eraser,
    Gradient,
    Radial,
    ObjectSelect,
    Remove,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerRow {
    pub id: LayerId,
    pub depth: usize,
    pub name: String,
    pub visible: bool,
    pub group: bool,
    pub has_mask: bool,
    pub mask_enabled: bool,
    /// Thumbnail texture and the used part of it (u, v).
    pub thumb: Option<(egui::TextureId, (f32, f32))>,
    pub adjustment: bool,
    pub filters: Vec<mosaix_core::SmartFilter>,
    /// Text (true) or shape (false) layer.
    pub vector: Option<bool>,
    /// Has layer styles switched on.
    pub fx: bool,
    /// Clipped to the layer below (clipping mask).
    pub clip: bool,
    /// Mask thumbnail (grey) and the used part of it.
    pub mask_thumb: Option<(egui::TextureId, (f32, f32))>,
}

/// A path layer offered for text on a path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathChoice {
    pub name: String,
    pub nodes: Vec<mosaix_core::vector::PathNode>,
    pub closed: bool,
}

/// Side of a thumbnail in pixels (matches the engine's THUMB).
pub const THUMB_PX: usize = 64;

/// Caret, selection and frame of the text edited on the canvas, as quads
/// in screen points.
#[derive(Debug, Clone, PartialEq)]
pub struct TextOverlay {
    pub selection: Vec<[[f32; 2]; 4]>,
    pub caret: [[f32; 2]; 4],
    pub frame: [[f32; 2]; 4],
}

/// The pen's path on screen: curve, nodes (position, selected, smooth),
/// handles (node → handle), rubber band to the cursor, closing hint.
#[derive(Debug, Clone, PartialEq)]
pub struct PathOverlay {
    pub curve: Vec<[f32; 2]>,
    pub nodes: Vec<([f32; 2], bool, bool)>,
    pub handles: Vec<([f32; 2], [f32; 2])>,
    pub rubber: Option<([f32; 2], [f32; 2])>,
    pub closing: bool,
}

/// The filter dialog: a new filter (destructive or smart) or an existing
/// smart filter being edited.
pub struct FilterDialog {
    pub filter: mosaix_core::Filter,
    preview: bool,
    smart: bool,
    /// Editing smart filter `index` of `layer`.
    edit: Option<(LayerId, usize)>,
}

/// What the canvas previews for the filter dialog.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPreview {
    Pixels(mosaix_core::Filter),
    /// Smart filter `Some(index)` replaced, or one added.
    Smart(Option<usize>, mosaix_core::Filter),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushModel {
    pub size: f32,
    pub hardness: f32,
    pub flow: f32,
    pub opacity: f32,
    pub spacing: f32,
    pub smoothing: f32,
    pub tip: u32,
    pub angle: f32,
    pub roundness: f32,
    pub size_jitter: f32,
    pub angle_jitter: f32,
    pub scatter: f32,
    pub flow_jitter: f32,
    pub follow_direction: bool,
}

/// What the panels show; rebuilt every frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub doc_size: (u32, u32),
    pub name: String,
    pub modified: bool,
    pub zoom: f64,
    pub rotation_deg: f64,
    pub cursor_doc: Option<(f64, f64)>,
    pub tool: ToolId,
    pub color: [f32; 3],
    pub brush: BrushModel,
    pub smudge_strength: f32,
    /// Clone stamp: sample all layers, aligned, source set.
    pub clone_all: bool,
    pub clone_aligned: bool,
    pub clone_source: bool,
    pub wand_tolerance: f32,
    pub wand_contiguous: bool,
    pub bg_color: [f32; 3],
    /// Names of the custom brush tips (tip numbers FIRST_CUSTOM_TIP, …).
    pub custom_tips: Vec<String>,
    /// Pattern library (built-ins first).
    pub patterns: std::sync::Arc<Vec<mosaix_core::pattern::Pattern>>,
    pub filter_plugins: std::sync::Arc<Vec<crate::app::filters::Plugin>>,
    pub actions: std::sync::Arc<Vec<crate::app::actions::Entry>>,
    /// Steps recorded so far, while an action is being recorded.
    pub recording: Option<usize>,
    pub eyedrop_radius: i64,
    pub eyedrop_all: bool,
    pub boxed: Option<&'static str>,
    /// Free transform: current angle (degrees).
    pub box_angle: Option<f64>,
    /// Free transform: what the handles do.
    pub box_mode: Option<crate::boxedit::Mode>,
    /// Liquify session: tool, brush diameter, strength.
    pub liquify: Option<(mosaix_gpu::LiquifyTool, f32, f32)>,
    /// Text edited on the canvas: caret, selection, frame (screen points).
    pub text_overlay: Option<TextOverlay>,
    /// Pen: the path being drawn or edited (screen points).
    pub path_overlay: Option<PathOverlay>,
    pub mask_edit: bool,
    pub layers: Vec<LayerRow>,
    pub active: LayerId,
    pub active_opacity: f32,
    /// Fill opacity of the active layer (its pixels only, not its style).
    pub active_fill: f32,
    pub active_blend: BlendMode,
    /// The user's keyboard shortcuts.
    pub keymap: std::sync::Arc<palette::Keymap>,
    /// Active layer is a group: whether it is Pass Through.
    pub active_pass_through: Option<bool>,
    pub history: Vec<String>,
    pub history_done: usize,
    pub has_selection: bool,
    /// Names of the saved selections.
    pub channels: Vec<String>,
    pub status: String,
    pub busy: bool,
    pub recent: Vec<std::path::PathBuf>,
    pub active_adjustment: Option<mosaix_core::Adjustment>,
    pub active_raw: Option<super::raw_ui::ActiveRaw>,
    pub active_vector: Option<mosaix_core::VectorLayer>,
    /// Layer styles of the active layer (`None` for groups and adjustments).
    pub active_effects: Option<Vec<mosaix_core::style::Effect>>,
    /// Installed font families (loaded when first needed).
    pub fonts: std::sync::Arc<Vec<mosaix_vector::FontGroup>>,
    /// Path layers a text can follow: name, nodes in document coordinates, closed.
    pub paths: Vec<PathChoice>,
    pub favourite_fonts: Vec<String>,
    pub recent_fonts: Vec<String>,
    /// Shape tool: ellipse instead of rectangle.
    pub shape_ellipse: bool,
    /// Histogram of what is below the active adjustment (levels and curves).
    pub histogram: Option<std::sync::Arc<mosaix_gpu::Histogram>>,
}

#[derive(Debug, Clone)]
pub enum Action {
    /// An agent command (JSON), run in the window and recorded as is.
    Agent(serde_json::Value),
    /// Select › Modify (`op` as in the `select_modify` command).
    ModifySelection { op: &'static str, amount: f64 },
    Command(Cmd),
    /// Open a camera RAW file developed with these settings.
    OpenRaw(std::path::PathBuf, mosaix_io::raw::Develop),
    Tool(ToolId),
    Color([f32; 3]),
    BgColor([f32; 3]),
    /// Add the tips of an .abr file or an image to the brush library.
    ImportBrushes(std::path::PathBuf),
    ImportPatterns(std::path::PathBuf),
    /// Read the filter plugin folder again.
    ReloadFilterPlugins,
    /// Play a macro or start a script.
    Run(crate::app::actions::Entry),
    StartRecording,
    /// Stop recording; save under this name (`None`: discard).
    StopRecording(Option<String>),
    ReloadActions,
    /// Download a local AI model (by id) after the user agreed.
    DownloadModel(String),
    /// Keyboard shortcut of a command: `None` removes it, `reset` restores the default.
    Shortcut { id: String, combo: Option<palette::Combo>, reset: bool },
    ResetShortcuts,
    /// Groups: Pass Through on/off (choosing a blend mode turns it off).
    PassThrough(bool),
    SwapColors,
    ResetColors,
    Eyedrop {
        radius: i64,
        all: bool,
    },
    Brush(BrushModel),
    SmudgeStrength(f32),
    CloneOptions {
        all: bool,
        aligned: bool,
    },
    SetVector {
        v: mosaix_core::VectorLayer,
        done: bool,
    },
    /// Re-develop the active RAW layer.
    SetRaw {
        d: mosaix_core::raw::Develop,
        done: bool,
    },
    FontUsed(String),
    ToggleFavouriteFont(String),
    SetEffects {
        effects: Vec<mosaix_core::style::Effect>,
        done: bool,
    },
    ShapeEllipse(bool),
    WandTolerance(f32),
    WandContiguous(bool),
    SelectLayer(LayerId),
    ToggleVisible(LayerId),
    ToggleMaskEdit(LayerId),
    RenameLayer(LayerId, String),
    MoveLayer {
        id: LayerId,
        target: LayerId,
        above: bool,
        into: bool,
    },
    /// `done` = slider released (ends the merged undo step).
    LayerOpacity {
        value: f32,
        done: bool,
        /// Fill opacity instead of opacity.
        fill: bool,
    },
    LayerBlend(BlendMode),
    DeleteLayer,
    HistoryJump(usize),
    CommitBox,
    CancelBox,
    /// Free transform: turn by (degrees), set the angle, mirror.
    BoxRotate(f64),
    BoxAngle(f64),
    /// Skew / distort / perspective (opens free transform if needed).
    BoxMode(crate::boxedit::Mode),
    BoxFlip {
        x: bool,
        y: bool,
    },
    CanvasSize {
        w: u32,
        h: u32,
        anchor: (f64, f64),
    },
    ImageSize {
        w: u32,
        h: u32,
    },
    /// Image › Upscale (AI): the factor.
    Upscale(u32),
    OpenFile(std::path::PathBuf),
    SaveAs(std::path::PathBuf),
    ExportAs(std::path::PathBuf, mosaix_io::export::ExportOptions),
    NewDocument {
        w: u32,
        h: u32,
        white: bool,
    },
    Unsaved(super::files::Answer),
    AddAdjustment(mosaix_core::Adjustment),
    ApplyFilter(mosaix_core::Filter),
    AddSmartFilter(mosaix_core::Filter),
    SetSmartFilter {
        layer: LayerId,
        index: usize,
        filter: mosaix_core::Filter,
    },
    ToggleSmartFilter {
        layer: LayerId,
        index: usize,
    },
    RemoveSmartFilter {
        layer: LayerId,
        index: usize,
    },
    BakeSmartFilters(LayerId),
    Liquify {
        tool: mosaix_gpu::LiquifyTool,
        size: f32,
        strength: f32,
    },
    LiquifyCommit,
    LiquifyCancel,
    LiquifyUndo,
    SetAdjustment {
        adj: mosaix_core::Adjustment,
        done: bool,
    },
}

/// UI state that lives across frames.
#[derive(Default)]
pub struct State {
    pub hidden: bool,
    /// The start screen (shown over the canvas).
    pub home: Option<super::home::Home>,
    /// The tool each tool-bar group shows (index into [`TOOL_GROUPS`]).
    pub group_tool: std::collections::HashMap<usize, ToolId>,
    /// UI snapshots: a menu drawn open (English menu name, horizontal offset).
    pub snapshot_menu: Option<(&'static str, f32)>,
    /// UI snapshots: a tool group's list drawn open.
    pub snapshot_flyout: Option<usize>,
    /// Area left for the canvas (points); `None` = whole window.
    pub canvas: Option<egui::Rect>,
    pub palette: Option<(String, usize)>,
    /// The keyboard-shortcuts dialog: search text and the command whose new
    /// keys are being recorded.
    pub shortcuts: Option<ShortcutsDialog>,
    rename: Option<(LayerId, String)>,
    dialog: Option<Dialog>,
    show_log: bool,
    /// Picker state: HSV and the linear colour it produced.
    hsv: Option<([f32; 3], [f32; 3])>,
    hex: Option<String>,
    pub file: Option<super::files::FileDialog>,
    pub new_doc: Option<super::files::NewDialog>,
    /// Naming the recorded action.
    pub record_name: Option<String>,
    /// Asking to download a local AI model (its id).
    pub ai_download: Option<&'static str>,
    /// Developing a camera RAW file before it opens.
    pub raw: Option<super::raw_ui::RawDialog>,
    /// Unsaved-changes question: what happens if the user discards.
    pub confirm: Option<String>,
    adj_editor: super::adjust_ui::EditorState,
    pub vec_editor: super::vector_ui::EditorState,
    /// Expanded effect in the layer-styles section.
    fx_open: Option<usize>,
    filter: Option<FilterDialog>,
    /// New filters are added as smart filters (remembered while running).
    smart_filters: bool,
    /// What the canvas should preview right now (set by the dialog).
    pub filter_preview: Option<FilterPreview>,
    /// Whether the active layer had properties last frame (shows that panel).
    had_props: bool,
    /// The dock layout (panels, tabs, floating windows).
    pub dock: Option<egui_dock::DockState<super::dock::Pane>>,
    /// A panel to bring to the front on the next pass.
    pub focus_pane: Option<super::dock::Pane>,
    /// Layout as last saved / loaded; a changed layout lands in `layout_changed`.
    pub saved_layout: Option<serde_json::Value>,
    pub layout_changed: Option<serde_json::Value>,
    /// The colour panel edits the background colour (else the brush colour).
    edit_bg: bool,
}

enum Dialog {
    Canvas { w: u32, h: u32, anchor: (u8, u8) },
    Image { w: u32, h: u32, keep: bool },
    /// Select › Modify (op as in `select_modify`).
    Modify { op: &'static str, amount: f64 },
    ColorRange { color: [f32; 3], fuzziness: f32, image: bool, mode: &'static str },
    SaveSelection { name: String },
}

/// Select › Modify: agent op, menu label, default amount.
const MODIFY: [(&str, &str, f64); 5] = [
    ("expand", "Expand …", 5.0),
    ("contract", "Contract …", 5.0),
    ("border", "Border …", 10.0),
    ("smooth", "Smooth …", 5.0),
    ("feather", "Feather …", 10.0),
];

const ACCENT: Color32 = Color32::from_rgb(90, 150, 255);

impl State {
    /// Expands an effect in the layer-styles section.
    pub fn open_effect(&mut self, i: usize) {
        self.fx_open = Some(i);
        self.focus_pane = Some(super::dock::Pane::Styles);
    }

    /// Opens a Select › Modify dialog (`op` as in `select_modify`).
    pub fn open_modify(&mut self, op: &str) {
        if let Some((op, _, amount)) = MODIFY.iter().find(|x| x.0 == op) {
            self.dialog = Some(Dialog::Modify { op, amount: *amount });
        }
    }

    pub fn open_filter(&mut self, f: mosaix_core::Filter) {
        self.filter = Some(FilterDialog {
            filter: f,
            preview: true,
            smart: self.smart_filters,
            edit: None,
        });
    }

    /// A palette, dialog or question is open: keys go to the interface.
    pub fn captures_keyboard(&self) -> bool {
        self.palette.is_some()
            || self.confirm.is_some()
            || self.file.is_some()
            || self.new_doc.is_some()
            || self.raw.is_some()
            || self.record_name.is_some()
            || self.ai_download.is_some()
            || self.dialog.is_some()
            || self.filter.is_some()
            || self.vec_editor.fonts.open
    }
}

#[derive(Default)]
pub struct ShortcutsDialog {
    pub query: String,
    pub capturing: Option<String>,
}

pub fn style(ctx: &egui::Context) {
    use egui::{CornerRadius, Stroke as S};
    // "Graphite": panels a step lighter than the canvas surround, readable
    // secondary text, one blue accent for selection and active controls.
    let gray = Color32::from_gray;
    let mut v = egui::Visuals::dark();
    v.panel_fill = Color32::from_rgb(40, 42, 47);
    v.window_fill = Color32::from_rgb(46, 48, 54);
    v.window_stroke = S::new(1.0, Color32::from_rgb(62, 65, 72));
    v.extreme_bg_color = Color32::from_rgb(27, 28, 32);
    v.text_edit_bg_color = Some(Color32::from_rgb(30, 31, 35));
    v.faint_bg_color = Color32::from_rgb(47, 49, 55);
    v.code_bg_color = Color32::from_rgb(30, 31, 35);
    v.weak_text_color = Some(gray(150));
    v.hyperlink_color = ACCENT;
    v.selection.bg_fill = Color32::from_rgb(52, 98, 196);
    v.selection.stroke = S::new(1.0, Color32::WHITE);
    v.slider_trailing_fill = true;
    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(6);
    let r = CornerRadius::same(4);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = v.panel_fill;
    w.noninteractive.weak_bg_fill = v.panel_fill;
    w.noninteractive.bg_stroke = S::new(1.0, Color32::from_rgb(58, 61, 68));
    w.noninteractive.fg_stroke = S::new(1.0, gray(205));
    w.noninteractive.corner_radius = r;
    w.inactive.bg_fill = Color32::from_rgb(58, 61, 69);
    w.inactive.weak_bg_fill = Color32::from_rgb(54, 57, 64);
    w.inactive.bg_stroke = S::NONE;
    w.inactive.fg_stroke = S::new(1.0, gray(215));
    w.inactive.corner_radius = r;
    w.hovered.bg_fill = Color32::from_rgb(70, 74, 84);
    w.hovered.weak_bg_fill = Color32::from_rgb(66, 70, 80);
    w.hovered.bg_stroke = S::new(1.0, Color32::from_rgb(92, 97, 110));
    w.hovered.fg_stroke = S::new(1.5, gray(240));
    w.hovered.corner_radius = r;
    w.active.bg_fill = Color32::from_rgb(80, 85, 97);
    w.active.weak_bg_fill = Color32::from_rgb(76, 81, 93);
    w.active.bg_stroke = S::new(1.0, ACCENT);
    w.active.fg_stroke = S::new(2.0, Color32::WHITE);
    w.active.corner_radius = r;
    w.open = w.hovered;
    ctx.set_visuals(v);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = Vec2::new(6.0, 5.0);
        s.spacing.button_padding = Vec2::new(7.0, 3.0);
        s.spacing.slider_width = 150.0;
    });
}

pub fn build(ctx: &egui::Context, m: &Model, st: &mut State) -> Vec<Action> {
    let mut out = Vec::new();
    if let Some(po) = &m.path_overlay {
        let p = ctx.layer_painter(egui::LayerId::background());
        let pt = |v: [f32; 2]| egui::pos2(v[0], v[1]);
        let line = Color32::from_rgb(70, 150, 255);
        if po.curve.len() > 1 {
            p.add(egui::Shape::line(
                po.curve.iter().map(|v| pt(*v)).collect(),
                Stroke::new(1.5, line),
            ));
        }
        if let Some((a, b)) = po.rubber {
            p.add(egui::Shape::dashed_line(
                &[pt(a), pt(b)],
                Stroke::new(1.0, line),
                4.0,
                3.0,
            ));
        }
        for (n, h) in &po.handles {
            p.line_segment([pt(*n), pt(*h)], Stroke::new(1.0, Color32::from_gray(200)));
            p.circle(
                pt(*h),
                3.5,
                Color32::from_gray(30),
                Stroke::new(1.2, Color32::WHITE),
            );
        }
        for (i, (n, sel, smooth)) in po.nodes.iter().enumerate() {
            let r = egui::Rect::from_center_size(pt(*n), egui::vec2(7.0, 7.0));
            let fill = if *sel { line } else { Color32::WHITE };
            if *smooth {
                p.circle(pt(*n), 4.0, fill, Stroke::new(1.0, Color32::BLACK));
            } else {
                p.rect(
                    r,
                    0.0,
                    fill,
                    Stroke::new(1.0, Color32::BLACK),
                    egui::StrokeKind::Middle,
                );
            }
            if i == 0 && po.closing {
                p.circle_stroke(pt(*n), 8.0, Stroke::new(1.5, line));
            }
        }
    }
    if let Some(t) = &m.text_overlay {
        // Under the panels, over the canvas.
        let p = ctx.layer_painter(egui::LayerId::background());
        let pts = |q: &[[f32; 2]; 4]| q.iter().map(|v| egui::pos2(v[0], v[1])).collect::<Vec<_>>();
        let mut frame = pts(&t.frame);
        frame.push(frame[0]);
        p.add(egui::Shape::dashed_line(
            &frame,
            Stroke::new(1.0, Color32::from_white_alpha(140)),
            4.0,
            3.0,
        ));
        for q in &t.selection {
            p.add(egui::Shape::convex_polygon(
                pts(q),
                Color32::from_rgba_unmultiplied(70, 130, 255, 90),
                Stroke::NONE,
            ));
        }
        p.add(egui::Shape::convex_polygon(
            pts(&t.caret),
            Color32::from_rgb(70, 140, 255),
            Stroke::new(0.5, Color32::WHITE),
        ));
    }
    if let Some(what) = &st.confirm
        && let Some(a) = super::files::confirm_unsaved(ctx, &m.name, what)
    {
        st.confirm = None;
        out.push(Action::Unsaved(a));
    }
    if let Some(d) = &mut st.file
        && !d.show(ctx, &m.recent, &mut out)
    {
        st.file = None;
    }
    if let Some(d) = &mut st.new_doc
        && !d.show(ctx, &mut out)
    {
        st.new_doc = None;
    }
    if let Some(d) = &mut st.raw
        && !d.show(ctx, &mut out)
    {
        st.raw = None;
    }
    filter_dialog(ctx, st, &mut out);
    record_name_dialog(ctx, st, &mut out);
    ai_download_dialog(ctx, st, &mut out);
    shortcuts_dialog(ctx, m, st, &mut out);
    if let Some((query, sel)) = &mut st.palette {
        let mut close = false;
        command_palette(ctx, &m.keymap, query, sel, &mut out, &mut close);
        if close {
            st.palette = None;
        }
    }
    if st.hidden {
        st.canvas = None;
        return out;
    }
    ctx.root_ui(|ui| {
        egui::Panel::top("menu")
            .exact_size(26.0)
            .show(ui, |ui| menu_bar(ui, m, st, &mut out));
        egui::Panel::bottom("status")
            .exact_size(24.0)
            .show(ui, |ui| status_bar(ui, m, st));
        egui::Panel::top("options")
            .exact_size(34.0)
            .show(ui, |ui| options_bar(ui, m, &mut out));
        egui::Panel::left("tools")
            .exact_size(46.0)
            .resizable(false)
            .show(ui, |ui| toolbar(ui, m, st, &mut out));
        // A newly selected text, shape or adjustment layer shows its properties.
        let has_props = m.active_vector.is_some() || m.active_adjustment.is_some() || m.active_raw.is_some();
        if has_props && !st.had_props {
            st.focus_pane = Some(super::dock::Pane::Properties);
        }
        st.had_props = has_props;
        super::dock::show(ui, m, st, &mut out);
    });
    if let (Some(h), Some(rect)) = (&mut st.home, st.canvas)
        && !super::home::show(h, ctx, rect, m, &mut st.new_doc, &mut out)
    {
        st.home = None;
    }
    dialogs(ctx, m, st, &mut out);
    if st.show_log {
        let mut open = true;
        egui::Window::new(tr("Messages"))
            .open(&mut open)
            .default_size([520.0, 300.0])
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for msg in super::log::recent(200) {
                            ui.label(
                                RichText::new(&msg.text)
                                    .color(level_color(msg.level))
                                    .monospace(),
                            );
                        }
                    });
            });
        st.show_log = open;
    }
    if let Some((menu, x)) = st.snapshot_menu {
        egui::Area::new(egui::Id::unique("snapshot menu"))
            .fixed_pos(egui::pos2(x, 26.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::containers::menu::menu_style(ui.style_mut());
                egui::Frame::menu(ui.style()).show(ui, |ui| {
                    ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
                        menu_contents(ui, menu, m, st, &mut Vec::new())
                    })
                });
            });
    }
    out
}

fn level_color(l: tracing::Level) -> Color32 {
    match l {
        tracing::Level::ERROR => Color32::from_rgb(255, 110, 100),
        tracing::Level::WARN => Color32::from_rgb(255, 200, 90),
        _ => Color32::from_gray(200),
    }
}

/// File › Actions: macros and scripts, recording.
fn actions_menu(ui: &mut egui::Ui, icon: egui::Atom<'static>, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    let label = match m.recording {
        Some(n) => tr!("Actions (recording, {n} steps)", n = n),
        None => tr("Actions").to_string(),
    };
    ui.menu_button((icon, label), |ui| {
        for e in m.actions.iter() {
            let (text, hint) = match e.kind {
                crate::app::actions::Kind::Macro => (e.name.clone(), tr("Play on the open image")),
                crate::app::actions::Kind::Script => (format!("{}  ({})", e.name, tr("script")), tr("Script: runs with this window")),
            };
            if ui.button(text).on_hover_text(hint).clicked() {
                out.push(Action::Run(e.clone()));
                ui.close();
            }
        }
        if m.actions.is_empty() {
            ui.label(RichText::new(tr("No actions yet")).weak());
        }
        ui.separator();
        if m.recording.is_some() {
            if ui.button(tr("Stop Recording …")).clicked() {
                st.record_name = Some(String::new());
                ui.close();
            }
        } else if ui
            .button(tr("Start Recording"))
            .on_hover_text(tr("Filters, adjustments, layer styles, layer and selection commands become an action"))
            .clicked()
        {
            out.push(Action::StartRecording);
            ui.close();
        }
        if ui
            .button(tr("Reload Actions"))
            .on_hover_text(crate::app::actions::actions_dir().display().to_string())
            .clicked()
        {
            out.push(Action::ReloadActions);
            ui.close();
        }
    });
}

/// Asks before downloading a local AI model.
fn ai_download_dialog(ctx: &egui::Context, st: &mut State, out: &mut Vec<Action>) {
    let Some(id) = st.ai_download else { return };
    let Some(m) = mosaix_ai::models::by_id(id) else { return };
    let mut close = false;
    egui::Window::new(tr("Download AI Model"))
        .id(egui::Id::unique("ai download"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            ui.set_max_width(380.0);
            ui.label(tr!(
                "This feature needs the model {name} ({mb} MB, license {license}). It is downloaded once from huggingface.co and then runs on this computer – your images are not sent anywhere.",
                name = m.name,
                mb = m.size() / 1_000_000,
                license = m.license
            ));
            ui.label(RichText::new(mosaix_ai::models::root().join(m.id).display().to_string()).weak().small());
            ui.horizontal(|ui| {
                if ui.button(tr("Download")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    out.push(Action::DownloadModel(id.to_string()));
                    close = true;
                }
                if ui.button(tr("Cancel")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    close = true;
                }
            });
        });
    if close {
        st.ai_download = None;
    }
}

/// Name for the recorded action.
fn record_name_dialog(ctx: &egui::Context, st: &mut State, out: &mut Vec<Action>) {
    let Some(name) = &mut st.record_name else { return };
    let mut close = false;
    egui::Window::new(tr("Save Action"))
        .id(egui::Id::unique("save action"))
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(tr("Name"));
                let r = ui.text_edit_singleline(name);
                r.request_focus();
            });
            ui.horizontal(|ui| {
                let ok = !name.trim().is_empty();
                if ui.add_enabled(ok, egui::Button::new(tr("Save"))).clicked() || (ok && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    out.push(Action::StopRecording(Some(name.trim().to_string())));
                    close = true;
                }
                if ui.button(tr("Discard")).clicked() {
                    out.push(Action::StopRecording(None));
                    close = true;
                }
                if ui.button(tr("Continue Recording")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    close = true;
                }
            });
        });
    if close {
        st.record_name = None;
    }
}

fn menu_bar(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    egui::MenuBar::new().ui(ui, |ui| {
        for menu in MENUS {
            ui.menu_button(palette::menu_text(menu), |ui| menu_contents(ui, menu, m, st, out));
        }
        ui.menu_button(tr("Window"), |ui| super::dock::window_menu(ui, st));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let text = RichText::new(format!("{}   Ctrl+Shift+P", tr("Search commands …")))
                .color(Color32::from_gray(150));
            let b = egui::Button::new(text)
                .min_size(Vec2::new(240.0, 20.0))
                .fill(Color32::from_rgb(30, 31, 35));
            if ui.add(b).on_hover_text(tr("Command Palette")).clicked() {
                st.palette = Some((String::new(), 0));
            }
        });
    });
}

/// The entries of one menu, in the order and groups of [`super::menus`].
pub(super) fn menu_contents(ui: &mut egui::Ui, menu: &str, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    use super::menus::{Item, atom};
    ui.set_min_width(220.0);
    for item in super::menus::layout(menu) {
        match *item {
            Item::Sep => {
                ui.separator();
            }
            Item::Cmd(label, icon) => {
                let Some(c) = COMMANDS.iter().find(|c| c.menu == menu && c.label == label) else {
                    continue;
                };
                let b = egui::Button::new((atom(icon), c.label_text()))
                    .shortcut_text(palette::shortcut_text(c, &m.keymap));
                let why = unavailable(label, m);
                if ui.add_enabled(why.is_none(), b).on_disabled_hover_text(why.unwrap_or_default()).clicked() {
                    out.push(Action::Command(*c));
                    ui.close();
                }
            }
            Item::Special(sp, icon) => special_entry(ui, sp, atom(icon), m, st, out),
        }
    }
}

/// Why a menu command cannot run right now (`None`: it can). Only clear
/// cases; anything that depends on more than the model stays enabled.
fn unavailable(label: &str, m: &Model) -> Option<&'static str> {
    let active = m.layers.iter().find(|r| r.id == m.active);
    let pixels = active.is_some_and(|r| !r.group && !r.adjustment && r.vector.is_none());
    let need = |ok: bool, why: &'static str| (!ok).then_some(why);
    match label {
        "Undo" => need(m.history_done > 0, tr("Nothing to undo")),
        "Redo" => need(m.history_done < m.history.len(), tr("Nothing to redo")),
        "Delete Selection (Pixels)" | "Content-Aware Fill" | "Content-Aware Fill (All Layers)" | "Remove (AI)"
        | "Deselect" | "Inverse" | "Feather …" => {
            need(m.has_selection, tr("Needs a selection"))
        }
        "Ungroup Layers" => need(active.is_some_and(|r| r.group), tr("The active layer is not a group")),
        "Add Layer Mask" => need(active.is_some_and(|r| !r.has_mask), tr("The layer already has a mask")),
        "Toggle Layer Mask" | "Apply Layer Mask" | "Paint Mask / Pixels" => {
            need(active.is_some_and(|r| r.has_mask), tr("The layer has no mask"))
        }
        "Remove Background (AI)" => need(pixels, tr("Works on pixel layers")),
        "Rasterize (Text/Shape/RAW to Pixels)" => need(
            active.is_some_and(|r| r.vector.is_some()) || m.active_raw.is_some(),
            tr("Only text, shape and RAW layers"),
        ),
        _ => None,
    }
}

fn special_entry(
    ui: &mut egui::Ui,
    sp: super::menus::Special,
    icon: egui::Atom<'static>,
    m: &Model,
    st: &mut State,
    out: &mut Vec<Action>,
) {
    use super::menus::Special;
    match sp {
        Special::Recent => {
            ui.add_enabled_ui(!m.recent.is_empty(), |ui| {
                ui.menu_button((icon, tr("Open Recent")), |ui| {
                    for p in &m.recent {
                        let name = p
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        if ui.button(name).on_hover_text(p.display().to_string()).clicked() {
                            out.push(Action::OpenFile(p.clone()));
                            ui.close();
                        }
                    }
                });
            });
        }
        Special::Actions => actions_menu(ui, icon, m, st, out),
        Special::ColorRange => {
            if ui.button((icon, tr("Color Range …"))).clicked() {
                st.dialog = Some(Dialog::ColorRange { color: m.color, fuzziness: 40.0, image: false, mode: "replace" });
                ui.close();
            }
        }
        Special::Modify => {
            ui.add_enabled_ui(m.has_selection, |ui| {
                ui.menu_button((icon, tr("Modify")), |ui| {
                    for (op, label, _) in MODIFY.iter().filter(|x| x.0 != "feather") {
                        if ui.button(tr(label)).clicked() {
                            st.open_modify(op);
                            ui.close();
                        }
                    }
                });
            })
            .response
            .on_disabled_hover_text(tr("Needs a selection"));
        }
        Special::SaveSelection => {
            let r = ui.add_enabled(m.has_selection, egui::Button::new((icon, tr("Save Selection …"))));
            if r.on_disabled_hover_text(tr("Needs a selection")).clicked() {
                let n = m.channels.len() + 1;
                st.dialog = Some(Dialog::SaveSelection { name: tr!("Selection {n}", n = n).to_string() });
                ui.close();
            }
        }
        Special::LoadSelection => {
            ui.add_enabled_ui(!m.channels.is_empty(), |ui| {
                ui.menu_button((icon, tr("Load Selection")), |ui| {
                    let mods = ui.input(|i| i.modifiers);
                    for (i, name) in m.channels.iter().enumerate() {
                        let r = ui.button(name).on_hover_text(tr("Shift: add · Alt: subtract · Shift+Alt: intersect"));
                        if r.clicked() {
                            let mode = match (mods.shift, mods.alt) {
                                (true, true) => "intersect",
                                (true, false) => "add",
                                (false, true) => "subtract",
                                _ => "replace",
                            };
                            out.push(Action::Agent(serde_json::json!({ "cmd": "select_load", "channel": i, "mode": mode })));
                            ui.close();
                        }
                    }
                    ui.separator();
                    ui.menu_button(tr("Delete"), |ui| {
                        for (i, name) in m.channels.iter().enumerate() {
                            if ui.button(name).clicked() {
                                out.push(Action::Agent(serde_json::json!({ "cmd": "channel_delete", "channel": i })));
                                ui.close();
                            }
                        }
                    });
                });
            })
            .response
            .on_disabled_hover_text(tr("No saved selections yet"));
        }
        Special::Transform => {
            use crate::boxedit::Mode;
            let active = m.layers.iter().find(|r| r.id == m.active);
            let any = active.is_some_and(|r| !r.adjustment);
            let pixels = any && active.is_some_and(|r| r.vector.is_none()) && m.active_raw.is_none();
            ui.add_enabled_ui(any, |ui| {
                ui.menu_button((icon, tr("Transform")), |ui| {
                    for mo in Mode::ALL {
                        let ok = pixels || matches!(mo, Mode::Scale | Mode::Skew);
                        let r = ui
                            .add_enabled(ok, egui::Button::new(tr(mo.name())))
                            .on_disabled_hover_text(tr("Text, shape and RAW layers: rasterize first"));
                        if r.clicked() {
                            out.push(Action::BoxMode(mo));
                            ui.close();
                        }
                    }
                });
            })
            .response
            .on_disabled_hover_text(tr("Select a layer to transform"));
        }
        Special::Filters => {
            ui.menu_button((icon, tr("Filter")), |ui| {
                for f in mosaix_core::Filter::defaults() {
                    if ui.button(format!("{} …", f.name())).clicked() {
                        st.open_filter(f);
                        ui.close();
                    }
                }
                ui.separator();
                ui.menu_button((super::menus::atom(super::menus::Icon::Glyph(super::glyphs::PUZZLE_PIECE)), tr("Plugins")), |ui| {
                    for p in m.filter_plugins.iter() {
                        match &p.result {
                            Ok(f) => {
                                let r = ui.button(format!("{} …", f.name()));
                                let d = p.description();
                                let r = if d.is_empty() { r } else { r.on_hover_text(d) };
                                if r.clicked() {
                                    st.open_filter(f.clone());
                                    ui.close();
                                }
                            }
                            Err(e) => {
                                ui.add_enabled(false, egui::Button::new(&p.file)).on_disabled_hover_text(e);
                            }
                        }
                    }
                    if m.filter_plugins.is_empty() {
                        ui.label(RichText::new(tr("No plugins yet")).weak());
                    }
                    ui.separator();
                    if ui
                        .button(tr("Reload Plugins"))
                        .on_hover_text(crate::app::filters::filter_dir().display().to_string())
                        .clicked()
                    {
                        out.push(Action::ReloadFilterPlugins);
                        ui.close();
                    }
                });
            });
        }
        Special::Adjustments => {
            ui.menu_button((icon, tr("New Adjustment Layer")), |ui| {
                for a in mosaix_core::Adjustment::defaults() {
                    if ui.button(a.name()).clicked() {
                        out.push(Action::AddAdjustment(a));
                        ui.close();
                    }
                }
            });
        }
        Special::CanvasSize => {
            if ui.button((icon, tr("Canvas Size …"))).clicked() {
                st.dialog = Some(Dialog::Canvas {
                    w: m.doc_size.0,
                    h: m.doc_size.1,
                    anchor: (1, 1),
                });
                ui.close();
            }
        }
        Special::ImageSize => {
            if ui.button((icon, tr("Image Size …"))).clicked() {
                st.dialog = Some(Dialog::Image {
                    w: m.doc_size.0,
                    h: m.doc_size.1,
                    keep: true,
                });
                ui.close();
            }
        }
        Special::Upscale => {
            ui.menu_button((icon, tr("Upscale (AI)")), |ui| {
                let (w, h) = m.doc_size;
                for k in [2u32, 4] {
                    let secs = crate::ai::upscale_estimate(w as usize * h as usize);
                    let b = ui.button(format!("{k}×  ({} × {} px)", w * k, h * k)).on_hover_text(tr!(
                        "Enlarges every pixel layer with local AI (Real-ESRGAN) – adds plausible detail instead of blur. About {s} s here.",
                        s = format!("{:.0}", secs.max(1.0))
                    ));
                    if b.clicked() {
                        out.push(Action::Upscale(k));
                        ui.close();
                    }
                }
            });
        }
        Special::Messages => {
            if ui.button((icon, tr("Messages"))).clicked() {
                st.show_log = true;
                ui.close();
            }
        }
    }
}

fn status_bar(ui: &mut egui::Ui, m: &Model, st: &mut State) {
    ui.horizontal_centered(|ui| {
        ui.label(format!("{:.1} %", m.zoom * 100.0));
        if m.rotation_deg.abs() > 0.05 {
            ui.label(format!("{:.0}°", m.rotation_deg));
        }
        ui.separator();
        ui.label(format!("{} × {} px", m.doc_size.0, m.doc_size.1));
        if let Some((x, y)) = m.cursor_doc {
            ui.separator();
            ui.label(RichText::new(format!("x {:.0}  y {:.0}", x.floor(), y.floor())).monospace());
        }
        if m.has_selection {
            ui.separator();
            ui.label(tr("Selection active"));
        }
        if m.busy {
            ui.separator();
            ui.spinner();
            ui.label(tr("working …"));
        }
        ui.separator();
        // Latest message, fading out after a while; click for the log.
        if let Some(msg) = super::log::recent(1).pop() {
            let age = msg.at.elapsed().as_secs_f32();
            let alpha = if msg.level <= tracing::Level::WARN {
                1.0
            } else {
                (1.0 - (age - 6.0) / 2.0).clamp(0.0, 1.0)
            };
            if alpha > 0.0 {
                let r = ui.add(
                    egui::Label::new(
                        RichText::new(&msg.text)
                            .color(level_color(msg.level).gamma_multiply(alpha)),
                    )
                    .truncate()
                    .sense(Sense::click()),
                );
                if r.on_hover_text(tr("Show all messages")).clicked() {
                    st.show_log = true;
                }
                if alpha < 1.0 || age < 8.0 {
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
            }
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(&m.status).weak());
        });
    });
}

/// Tools that share a place in the tool bar (like Photoshop): the place shows
/// the last one used; right-click or press and hold lists them all.
pub const TOOL_GROUPS: &[&[ToolId]] = &[
    &[ToolId::Move],
    &[ToolId::Marquee, ToolId::Ellipse],
    &[ToolId::Lasso],
    &[ToolId::ObjectSelect, ToolId::Wand],
    &[ToolId::Crop],
    &[ToolId::Eyedropper],
    &[ToolId::SpotHeal, ToolId::Remove, ToolId::Heal],
    &[ToolId::Brush],
    &[ToolId::Clone],
    &[ToolId::Eraser],
    &[ToolId::Gradient, ToolId::Radial],
    &[ToolId::Smudge],
    &[ToolId::Pen],
    &[ToolId::Text],
    &[ToolId::Shape],
];

/// How long a press on a tool group opens its list.
const HOLD_SECS: f64 = 0.35;

fn tool_info(id: ToolId) -> (&'static str, &'static str) {
    TOOLS.iter().find(|t| t.0 == id).map(|t| (t.1, t.2)).unwrap_or(("", ""))
}

fn tool_hover(id: ToolId) -> String {
    let (name, key) = tool_info(id);
    format!("{} ({})", tr(name), key.replace("Shift", tr("Shift")))
}

fn toolbar(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    ui.add_space(4.0);
    ui.vertical_centered(|ui| {
        for (gi, group) in TOOL_GROUPS.iter().enumerate() {
            if group.contains(&m.tool) {
                st.group_tool.insert(gi, m.tool);
            }
            let shown = st.group_tool.get(&gi).copied().filter(|t| group.contains(t)).unwrap_or(group[0]);
            let active = group.contains(&m.tool);
            let resp = icons::button(ui, Icon::Tool(shown), 34.0, active);
            let several = group.len() > 1;
            if several {
                // The small corner triangle that says "more tools here".
                let r = resp.rect;
                let c = if active { Color32::WHITE } else { ui.visuals().weak_text_color() };
                ui.painter().add(egui::Shape::convex_polygon(
                    vec![r.right_bottom() + egui::vec2(-3.0, -3.0), r.right_bottom() + egui::vec2(-3.0, -8.0), r.right_bottom() + egui::vec2(-8.0, -3.0)],
                    c,
                    egui::Stroke::NONE,
                ));
            }
            let id = egui::Id::unique(("tool group", gi));
            let held = several
                && resp.is_pointer_button_down_on()
                && ui.input(|i| i.pointer.press_start_time().is_some_and(|t| i.time - t >= HOLD_SECS));
            if several && resp.is_pointer_button_down_on() {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));
            }
            if several && (resp.secondary_clicked() || held || st.snapshot_flyout == Some(gi)) {
                egui::Popup::open_id(ui.ctx(), id);
            }
            let resp = if egui::Popup::is_id_open(ui.ctx(), id) {
                resp
            } else {
                let more = if several { format!("\n{}", tr("Right-click or hold: more tools")) } else { String::new() };
                resp.on_hover_text(format!("{}{more}", tool_hover(shown)))
            };
            if resp.clicked() {
                out.push(Action::Tool(shown));
            }
            egui::Popup::from_response(&resp)
                .id(id)
                .align(egui::RectAlign::RIGHT_START)
                .open_memory(None)
                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                .show(|ui| {
                    egui::containers::menu::menu_style(ui.style_mut());
                    ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                        for &t in group.iter() {
                            let (name, key) = tool_info(t);
                            let b = egui::Button::new((super::menus::atom(super::menus::Icon::Tool(t)), tr(name)))
                                .shortcut_text(key.replace("Shift", tr("Shift")))
                                .selected(t == m.tool);
                            if ui.add(b).clicked() {
                                out.push(Action::Tool(t));
                                egui::Popup::close_id(ui.ctx(), id);
                            }
                        }
                    });
                });
        }
        ui.add_space(10.0);
        color_chips(ui, m, st, out);
    });
}

fn linear_color(c: [f32; 3]) -> Color32 {
    egui::Rgba::from_rgb(c[0], c[1], c[2]).into()
}

/// The bar under the menu: the active tool and its options (or the
/// options of a running free transform / liquify).
fn options_bar(ui: &mut egui::Ui, m: &Model, out: &mut Vec<Action>) {
    ui.horizontal_centered(|ui| {
        ui.spacing_mut().slider_width = 96.0;
        ui.spacing_mut().item_spacing.x = 6.0;
        if let Some((tool, size, strength)) = m.liquify {
            ui.label(RichText::new(tr("Liquify")).strong());
            ui.separator();
            let (mut t, mut sz, mut st) = (tool, size, strength);
            for (lt, key) in mosaix_gpu::LiquifyTool::ALL
                .into_iter()
                .zip(["W", "R", "C", "Alt+C", "S", "B"])
            {
                if ui.selectable_label(t == lt, lt.name()).on_hover_text(key).clicked() {
                    t = lt;
                }
            }
            ui.separator();
            ui.label(tr("Size"));
            ui.add(egui::Slider::new(&mut sz, 4.0..=5000.0).logarithmic(true).suffix(" px").max_decimals(0));
            let mut pct = st * 100.0;
            ui.label(tr("Strength"));
            ui.add(egui::Slider::new(&mut pct, 1.0..=100.0).suffix(" %").max_decimals(0));
            st = pct / 100.0;
            if (t, sz, st) != (tool, size, strength) {
                out.push(Action::Liquify { tool: t, size: sz, strength: st });
            }
            ui.separator();
            if ui.button(tr("Apply")).on_hover_text("Enter").clicked() {
                out.push(Action::LiquifyCommit);
            }
            if ui.button(tr("Step Back")).on_hover_text(tr("Ctrl+Z undoes the last stroke")).clicked() {
                out.push(Action::LiquifyUndo);
            }
            if ui.button(tr("Cancel")).on_hover_text("Esc").clicked() {
                out.push(Action::LiquifyCancel);
            }
            return;
        }
        if let Some(kind) = m.boxed {
            ui.label(RichText::new(tr(kind)).strong());
            ui.separator();
            if let Some(angle) = m.box_angle {
                ui.label(tr("Angle"));
                let mut a = angle;
                if ui.add(egui::DragValue::new(&mut a).range(-180.0..=180.0).suffix("°").speed(0.5).max_decimals(1)).changed() {
                    out.push(Action::BoxAngle(a));
                }
                for (label, deg, tip) in [("90° left", -90.0, "90° counterclockwise"), ("90° right", 90.0, "90° clockwise"), ("180°", 180.0, "half turn")] {
                    if ui.small_button(tr(label)).on_hover_text(tr(tip)).clicked() {
                        out.push(Action::BoxRotate(deg));
                    }
                }
                ui.separator();
                ui.label(tr("Flip"));
                if ui.small_button(super::glyphs::FLIP_HORIZONTAL).on_hover_text(tr("horizontal")).clicked() {
                    out.push(Action::BoxFlip { x: true, y: false });
                }
                if ui.small_button(super::glyphs::FLIP_VERTICAL).on_hover_text(tr("vertical")).clicked() {
                    out.push(Action::BoxFlip { x: false, y: true });
                }
                ui.separator();
            }
            if let Some(mode) = m.box_mode {
                for mo in crate::boxedit::Mode::ALL {
                    let tip = match mo {
                        crate::boxedit::Mode::Scale => tr("Handles scale"),
                        crate::boxedit::Mode::Skew => tr("Edges slide along themselves"),
                        crate::boxedit::Mode::Distort => tr("Corners move freely (also: hold Ctrl)"),
                        crate::boxedit::Mode::Perspective => tr("Opposite corners move together"),
                    };
                    if ui.selectable_label(mode == mo, tr(mo.name())).on_hover_text(tip).clicked() && mode != mo {
                        out.push(Action::BoxMode(mo));
                    }
                }
                ui.separator();
            }
            if ui.button(tr("Apply")).on_hover_text("Enter").clicked() {
                out.push(Action::CommitBox);
            }
            if ui.button(tr("Cancel")).on_hover_text("Esc").clicked() {
                out.push(Action::CancelBox);
            }
            if m.box_angle.is_some() {
                hint(ui, tr("Drag corners: scale (Shift: free) · Ctrl: distort · drag outside: rotate (Shift: 15°) · Alt: around the center"));
            }
            return;
        }
        if let Some((id, name, _)) = TOOLS.iter().find(|t| t.0 == m.tool) {
            let (r, _) = ui.allocate_exact_size(Vec2::splat(20.0), Sense::hover());
            icons::paint(ui.painter(), r, Icon::Tool(*id), ui.visuals().strong_text_color());
            ui.label(RichText::new(tr(name)).strong());
            ui.separator();
        }
        tool_section(ui, m, out);
    });
}

/// A short explanation at the end of the options bar (cut off when tight,
/// complete on hover).
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.separator();
    ui.add(egui::Label::new(RichText::new(text).weak()).truncate())
        .on_hover_text(text);
}

pub(super) fn styles_tab(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    match &m.active_effects {
        Some(fx) => {
            if let Some((effects, done)) = super::style_ui::editor(ui, fx, &mut st.fx_open, &m.patterns) {
                out.push(Action::SetEffects { effects, done });
            }
        }
        None => {
            ui.label(RichText::new(tr("Select a layer to add effects.")).weak());
        }
    }
}

pub(super) fn properties_tab(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    if let Some(v) = &m.active_vector {
        let e = super::vector_ui::editor(
            ui,
            v,
            &m.paths,
            &m.fonts,
            &m.favourite_fonts,
            &m.recent_fonts,
            m.color,
            &m.patterns,
            &mut st.vec_editor,
        );
        if let Some((v, done)) = e.change {
            out.push(Action::SetVector { v, done });
        }
        if let Some(f) = e.font_used {
            out.push(Action::FontUsed(f));
        }
        if let Some(f) = e.toggle_favourite {
            out.push(Action::ToggleFavouriteFont(f));
        }
    } else if let Some(r) = &m.active_raw {
        if let Some((d, done)) = super::raw_ui::layer_editor(ui, r) {
            out.push(Action::SetRaw { d, done });
        }
    } else if let Some(adj) = &m.active_adjustment {
        if let Some((adj, done)) =
            super::adjust_ui::editor(ui, adj, &mut st.adj_editor, m.histogram.as_deref())
        {
            out.push(Action::SetAdjustment { adj, done });
        }
    } else {
        ui.label(
            RichText::new(tr(
                "Pixel layer – text, shape, RAW and adjustment layers show their settings here.",
            ))
            .weak(),
        );
    }
}

const SWATCHES: [[u8; 3]; 12] = [
    [0, 0, 0],
    [255, 255, 255],
    [128, 128, 128],
    [230, 57, 70],
    [244, 162, 97],
    [233, 196, 106],
    [42, 157, 143],
    [38, 70, 83],
    [69, 123, 157],
    [106, 76, 147],
    [255, 105, 180],
    [139, 94, 60],
];

/// Photoshop's colour squares: brush colour over background colour, a swap
/// arrow (X) and a reset to black/white (D).
fn color_chips(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(38.0), Sense::hover());
    let o = rect.min;
    let r = |x: f32, y: f32, w: f32| egui::Rect::from_min_size(o + Vec2::new(x, y), Vec2::splat(w));
    let (fg_r, bg_r, swap_r, reset_r) = (
        r(2.0, 2.0, 22.0),
        r(14.0, 14.0, 22.0),
        r(27.0, 1.0, 10.0),
        r(1.0, 27.0, 10.0),
    );
    let chip = |ui: &mut egui::Ui, rect: egui::Rect, c: [f32; 3], id: &str, active: bool| {
        let resp = ui.interact(rect, ui.scope_id().with(id), Sense::click());
        let p = ui.painter();
        p.rect_filled(rect, 2.0, linear_color(c));
        p.rect_stroke(
            rect,
            2.0,
            Stroke::new(1.0, Color32::from_gray(20)),
            egui::StrokeKind::Inside,
        );
        let ring = if active {
            ACCENT
        } else {
            Color32::from_gray(170)
        };
        p.rect_stroke(rect, 2.0, Stroke::new(1.0, ring), egui::StrokeKind::Outside);
        resp
    };
    // Background first so the brush colour lies on top.
    if chip(ui, bg_r, m.bg_color, "bg chip", st.edit_bg)
        .on_hover_text(tr("Background color – click to edit"))
        .clicked()
    {
        st.edit_bg = true;
        st.focus_pane = Some(super::dock::Pane::Color);
    }
    if chip(ui, fg_r, m.color, "fg chip", !st.edit_bg)
        .on_hover_text(tr("Brush color – click to edit"))
        .clicked()
    {
        st.edit_bg = false;
        st.focus_pane = Some(super::dock::Pane::Color);
    }
    let p = ui.painter();
    let line = Stroke::new(1.2, Color32::from_gray(200));
    // Swap: a bent double arrow.
    let (a, b, c) = (
        swap_r.left_top() + Vec2::new(2.0, 2.0),
        swap_r.right_top() + Vec2::new(-2.0, 2.0),
        swap_r.right_bottom() + Vec2::new(-2.0, -1.0),
    );
    p.line_segment([a, b], line);
    p.line_segment([b, c], line);
    p.line_segment([a, a + Vec2::new(2.5, -2.0)], line);
    p.line_segment([a, a + Vec2::new(2.5, 2.0)], line);
    p.line_segment([c, c + Vec2::new(-2.0, -2.5)], line);
    p.line_segment([c, c + Vec2::new(2.0, -2.5)], line);
    if ui
        .interact(swap_r, ui.scope_id().with("swap"), Sense::click())
        .on_hover_text(tr("Swap colors (X)"))
        .clicked()
    {
        out.push(Action::SwapColors);
    }
    // Reset: small black over white.
    let p = ui.painter();
    p.rect_filled(
        egui::Rect::from_min_size(reset_r.min + Vec2::new(3.0, 3.0), Vec2::splat(6.0)),
        0.0,
        Color32::WHITE,
    );
    p.rect_filled(
        egui::Rect::from_min_size(reset_r.min, Vec2::splat(6.0)),
        0.0,
        Color32::BLACK,
    );
    p.rect_stroke(
        egui::Rect::from_min_size(reset_r.min, Vec2::splat(6.0)),
        0.0,
        Stroke::new(0.8, Color32::from_gray(200)),
        egui::StrokeKind::Outside,
    );
    if ui
        .interact(reset_r, ui.scope_id().with("reset"), Sense::click())
        .on_hover_text(tr("Default colors: black/white (D)"))
        .clicked()
    {
        out.push(Action::ResetColors);
    }
}

pub(super) fn color_section(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    use egui::ecolor::Hsva;
    let target = if st.edit_bg { m.bg_color } else { m.color };
    // Which colour is edited, plus swap / reset / eyedropper.
    ui.horizontal(|ui| {
        for (bg, c, label) in [
            (false, m.color, tr("Brush")),
            (true, m.bg_color, tr("Background")),
        ] {
            let active = st.edit_bg == bg;
            let (rect, resp) = ui.allocate_exact_size(Vec2::new(28.0, 20.0), Sense::click());
            ui.painter().rect_filled(rect, 3.0, linear_color(c));
            ui.painter().rect_stroke(
                rect,
                3.0,
                Stroke::new(
                    if active { 2.0 } else { 1.0 },
                    if active {
                        ACCENT
                    } else {
                        Color32::from_gray(110)
                    },
                ),
                egui::StrokeKind::Outside,
            );
            if resp.on_hover_text(label).clicked() {
                st.edit_bg = bg;
            }
            if ui.selectable_label(active, label).clicked() {
                st.edit_bg = bg;
            }
        }
    });
    ui.horizontal(|ui| {
        if icons::button(
            ui,
            Icon::Tool(ToolId::Eyedropper),
            22.0,
            m.tool == ToolId::Eyedropper,
        )
        .on_hover_text(tr("Eyedropper (I) – or Alt+click while painting"))
        .clicked()
        {
            out.push(Action::Tool(ToolId::Eyedropper));
        }
        if ui
            .small_button(tr("Swap"))
            .on_hover_text(tr("Swap colors (X)"))
            .clicked()
        {
            out.push(Action::SwapColors);
        }
        if ui
            .small_button(tr("Reset"))
            .on_hover_text(tr("Default colors: black/white (D)"))
            .clicked()
        {
            out.push(Action::ResetColors);
        }
    });
    ui.add_space(4.0);
    let current: Color32 = linear_color(target);
    // Keep our own HSV so hue survives black/grey colours.
    let mut hsv = match st.hsv {
        Some((hsv, seen)) if seen == target => hsv,
        _ => {
            let h = Hsva::from(current);
            [h.h, h.s, h.v]
        }
    };
    let width = ui.available_width().clamp(120.0, 320.0);
    let (sv_rect, sv) =
        ui.allocate_exact_size(Vec2::new(width, width * 0.62), Sense::click_and_drag());
    let (hue_rect, hue) = ui.allocate_exact_size(Vec2::new(width, 14.0), Sense::click_and_drag());
    let p = ui.painter();
    let n = 16;
    let mut mesh = egui::Mesh::default();
    for j in 0..=n {
        for i in 0..=n {
            let (s, v) = (i as f32 / n as f32, 1.0 - j as f32 / n as f32);
            let pos = sv_rect.min + Vec2::new(s * sv_rect.width(), (1.0 - v) * sv_rect.height());
            mesh.colored_vertex(pos, Hsva::new(hsv[0], s, v, 1.0).into());
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = (j * (n + 1) + i) as u32;
            mesh.add_triangle(a, a + 1, a + n as u32 + 2);
            mesh.add_triangle(a, a + n as u32 + 2, a + n as u32 + 1);
        }
    }
    p.add(egui::Shape::mesh(mesh));
    let mut hue_mesh = egui::Mesh::default();
    for i in 0..=36 {
        let t = i as f32 / 36.0;
        let x = hue_rect.min.x + t * hue_rect.width();
        let c: Color32 = Hsva::new(t, 1.0, 1.0, 1.0).into();
        hue_mesh.colored_vertex(egui::pos2(x, hue_rect.min.y), c);
        hue_mesh.colored_vertex(egui::pos2(x, hue_rect.max.y), c);
        if i > 0 {
            let k = (i * 2) as u32;
            hue_mesh.add_triangle(k - 2, k - 1, k);
            hue_mesh.add_triangle(k - 1, k + 1, k);
        }
    }
    p.add(egui::Shape::mesh(hue_mesh));
    let mut changed = false;
    if let Some(pos) = sv
        .interact_pointer_pos()
        .filter(|_| sv.is_pointer_button_down_on() || sv.clicked())
    {
        hsv[1] = ((pos.x - sv_rect.min.x) / sv_rect.width()).clamp(0.0, 1.0);
        hsv[2] = 1.0 - ((pos.y - sv_rect.min.y) / sv_rect.height()).clamp(0.0, 1.0);
        changed = true;
    }
    if let Some(pos) = hue
        .interact_pointer_pos()
        .filter(|_| hue.is_pointer_button_down_on() || hue.clicked())
    {
        hsv[0] = ((pos.x - hue_rect.min.x) / hue_rect.width()).clamp(0.0, 0.9999);
        changed = true;
    }
    let mark = sv_rect.min + Vec2::new(hsv[1] * sv_rect.width(), (1.0 - hsv[2]) * sv_rect.height());
    p.circle_stroke(
        mark,
        5.0,
        Stroke::new(
            2.0,
            if hsv[2] > 0.5 {
                Color32::BLACK
            } else {
                Color32::WHITE
            },
        ),
    );
    let hx = hue_rect.min.x + hsv[0] * hue_rect.width();
    p.rect_stroke(
        egui::Rect::from_min_max(
            egui::pos2(hx - 2.0, hue_rect.min.y - 1.0),
            egui::pos2(hx + 2.0, hue_rect.max.y + 1.0),
        ),
        1.0,
        Stroke::new(1.5, Color32::WHITE),
        egui::StrokeKind::Outside,
    );
    let edit_bg = st.edit_bg;
    let mut emit = |c: Color32, hsv: [f32; 3], st: &mut State| {
        let r = egui::Rgba::from(c);
        let lin = [r.r(), r.g(), r.b()];
        st.hsv = Some((hsv, lin));
        out.push(if edit_bg {
            Action::BgColor(lin)
        } else {
            Action::Color(lin)
        });
    };
    // Numeric fields: HSB and 8-bit sRGB, plus hex.
    ui.add_space(4.0);
    let [r8, g8, b8, _] = current.to_array();
    let (mut h, mut sat, mut bri) = (hsv[0] * 360.0, hsv[1] * 100.0, hsv[2] * 100.0);
    let mut rgb = [r8, g8, b8];
    let mut hsv_changed = false;
    let mut rgb_changed = false;
    egui::Grid::new("color fields")
        .num_columns(4)
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            let dv = |ui: &mut egui::Ui, v: &mut f32, max: f32, suffix: &str| {
                ui.add(
                    egui::DragValue::new(v)
                        .range(0.0..=max)
                        .max_decimals(0)
                        .suffix(suffix)
                        .speed(0.5),
                )
                .changed()
            };
            ui.label("H");
            hsv_changed |= dv(ui, &mut h, 359.0, "°");
            ui.label("R");
            rgb_changed |= ui
                .add(egui::DragValue::new(&mut rgb[0]).range(0..=255))
                .changed();
            ui.end_row();
            ui.label("S");
            hsv_changed |= dv(ui, &mut sat, 100.0, " %");
            ui.label("G");
            rgb_changed |= ui
                .add(egui::DragValue::new(&mut rgb[1]).range(0..=255))
                .changed();
            ui.end_row();
            ui.label("B");
            hsv_changed |= dv(ui, &mut bri, 100.0, " %");
            ui.label("B");
            rgb_changed |= ui
                .add(egui::DragValue::new(&mut rgb[2]).range(0..=255))
                .changed();
            ui.end_row();
        });
    if hsv_changed {
        hsv = [(h / 360.0).clamp(0.0, 0.9999), sat / 100.0, bri / 100.0];
        changed = true;
    }
    if changed {
        emit(Hsva::new(hsv[0], hsv[1], hsv[2], 1.0).into(), hsv, st);
    } else if rgb_changed {
        let col = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        let hs = Hsva::from(col);
        emit(col, [hs.h, hs.s, hs.v], st);
    }
    let mut hex = st
        .hex
        .clone()
        .unwrap_or_else(|| format!("#{r8:02x}{g8:02x}{b8:02x}"));
    ui.horizontal(|ui| {
        ui.label("#");
        let resp = ui.add(
            egui::TextEdit::singleline(&mut hex)
                .desired_width(80.0)
                .font(egui::TextStyle::Monospace),
        );
        if resp.has_focus() {
            st.hex = Some(hex.clone());
        }
        if resp.lost_focus() {
            st.hex = None;
            let h = hex.trim().trim_start_matches('#');
            let h: String = if h.len() == 3 {
                h.chars().flat_map(|c| [c, c]).collect()
            } else {
                h.to_string()
            };
            if h.len() == 6
                && let Ok(v) = u32::from_str_radix(&h, 16)
            {
                let col = Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8);
                let hs = Hsva::from(col);
                emit(col, [hs.h, hs.s, hs.v], st);
            }
        }
    });
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        for s in SWATCHES {
            let col = Color32::from_rgb(s[0], s[1], s[2]);
            let (rect, resp) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::click());
            ui.painter().rect_filled(rect, 3.0, col);
            if resp.hovered() {
                ui.painter().rect_stroke(
                    rect,
                    3.0,
                    Stroke::new(1.0, Color32::WHITE),
                    egui::StrokeKind::Outside,
                );
            }
            if resp.clicked() {
                let h = Hsva::from(col);
                emit(col, [h.h, h.s, h.v], st);
            }
        }
    });
}

fn slider(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    pct: bool,
) -> bool {
    ui.label(label);
    let s = egui::Slider::new(v, range).clamping(egui::SliderClamping::Always);
    let s = if pct {
        s.custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
            .custom_parser(|t| {
                t.trim_end_matches('%')
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .map(|v| v / 100.0)
            })
    } else {
        s
    };
    ui.add(s).changed()
}

/// Options of the active tool, laid out in a row (the options bar).
fn tool_section(ui: &mut egui::Ui, m: &Model, out: &mut Vec<Action>) {
    match m.tool {
        ToolId::ObjectSelect => hint(
            ui,
            tr(
                "Click an object to select it (Shift adds, Alt subtracts). Local AI (SAM 2.1): the first click on an image takes a moment, further ones are quick. Select › Subject (Ctrl+Alt+S) finds the main subject.",
            ),
        ),
        ToolId::Text => hint(
            ui,
            tr(
                "Click in the image: new text. Click on a text: edit it. Font, size and color are under Properties; new text uses the brush color.",
            ),
        ),
        ToolId::Pen => hint(
            ui,
            tr(
                "Click: corner · Drag: curve · Start point: close (filled) · Enter/Esc: finish open (line). On a path: drag points and handles, Alt+click corner ↔ curve, Alt+drag moves one handle, Del deletes the point.",
            ),
        ),
        ToolId::Shape => {
            if ui
                .selectable_label(!m.shape_ellipse, tr("Rectangle"))
                .clicked()
            {
                out.push(Action::ShapeEllipse(false));
            }
            if ui.selectable_label(m.shape_ellipse, "Ellipse").clicked() {
                out.push(Action::ShapeEllipse(true));
            }
            hint(
                ui,
                tr(
                    "Drag out (Shift: square/circle, Alt: from the center). Fill in the brush color; stroke and corners under Properties.",
                ),
            );
        }
        ToolId::Brush
        | ToolId::Eraser
        | ToolId::Smudge
        | ToolId::Clone
        | ToolId::Heal
        | ToolId::SpotHeal
        | ToolId::Remove => {
            let mut b = m.brush;
            let mut changed = false;
            ui.label(tr("Size"));
            changed |= ui
                .add(
                    egui::Slider::new(&mut b.size, 1.0..=2000.0)
                        .logarithmic(true)
                        .suffix(" px")
                        .max_decimals(0),
                )
                .changed();
            let first = mosaix_gpu::FIRST_CUSTOM_TIP;
            let tip_name = |t: u32| -> String {
                if t >= first {
                    m.custom_tips.get((t - first) as usize).cloned().unwrap_or_else(|| "?".into())
                } else {
                    tr(TIP_NAMES.get(t as usize).copied().unwrap_or("round")).to_string()
                }
            };
            egui::ComboBox::from_id_salt("tip")
                .selected_text(tip_name(b.tip))
                .width(110.0)
                .height(400.0)
                .show_ui(ui, |ui| {
                    for (i, n) in TIP_NAMES.iter().enumerate() {
                        changed |= ui.selectable_value(&mut b.tip, i as u32, tr(n)).changed();
                    }
                    if !m.custom_tips.is_empty() {
                        ui.separator();
                        for (i, n) in m.custom_tips.iter().enumerate() {
                            changed |= ui.selectable_value(&mut b.tip, first + i as u32, n).changed();
                        }
                    }
                    ui.separator();
                    if ui.button(tr("Import Brushes …")).on_hover_text(tr("Photoshop (.abr) or GIMP brushes (.gbr, .gih), or an image (dark or opaque = paint)")).clicked() {
                        out.push(Action::Command(palette::find("Import Brushes …")));
                    }
                })
                .response
                .on_hover_text(tr("Tip"));
            if b.tip == 0 {
                changed |= slider(ui, tr("Hardness"), &mut b.hardness, 0.0..=1.0, true);
            }
            if m.tool == ToolId::Smudge {
                let mut s = m.smudge_strength;
                if slider(ui, tr("Strength"), &mut s, 0.0..=1.0, true) {
                    out.push(Action::SmudgeStrength(s));
                }
            } else if !matches!(m.tool, ToolId::SpotHeal | ToolId::Remove) {
                changed |= slider(ui, tr("Opacity"), &mut b.opacity, 0.0..=1.0, true);
                changed |= slider(ui, tr("Flow"), &mut b.flow, 0.01..=1.0, true);
            }
            ui.menu_button(tr("Brush Settings …"), |ui| {
                egui::Grid::new("brush settings")
                    .num_columns(2)
                    .spacing([10.0, 6.0])
                    .show(ui, |ui| {
                        changed |= slider(ui, tr("Spacing"), &mut b.spacing, 0.02..=2.0, true);
                        ui.end_row();
                        changed |= slider(ui, tr("Smoothing"), &mut b.smoothing, 0.0..=0.95, true);
                        ui.end_row();
                        changed |= slider(ui, tr("Angle"), &mut b.angle, -180.0..=180.0, false);
                        ui.end_row();
                        changed |= slider(ui, tr("Roundness"), &mut b.roundness, 0.05..=1.0, true);
                        ui.end_row();
                        ui.label("");
                        changed |= ui
                            .checkbox(&mut b.follow_direction, tr("Follow direction"))
                            .changed();
                        ui.end_row();
                        changed |= slider(ui, tr("Size ±"), &mut b.size_jitter, 0.0..=1.0, true);
                        ui.end_row();
                        changed |=
                            slider(ui, tr("Angle ±"), &mut b.angle_jitter, 0.0..=180.0, false);
                        ui.end_row();
                        changed |= slider(ui, tr("Scatter"), &mut b.scatter, 0.0..=5.0, false);
                        ui.end_row();
                        changed |= slider(ui, tr("Flow ±"), &mut b.flow_jitter, 0.0..=1.0, true);
                        ui.end_row();
                    });
            });
            if changed {
                out.push(Action::Brush(b));
            }
            if matches!(m.tool, ToolId::Clone | ToolId::Heal | ToolId::SpotHeal | ToolId::Remove) {
                ui.separator();
                let (mut all, mut aligned) = (m.clone_all, m.clone_aligned);
                let tip = if matches!(m.tool, ToolId::SpotHeal | ToolId::Remove) {
                    tr("Heal on an empty layer")
                } else {
                    tr("The source is the whole image – for stamping onto an empty layer")
                };
                let mut c = ui
                    .checkbox(&mut all, tr("Sample All Layers"))
                    .on_hover_text(tip)
                    .changed();
                if !matches!(m.tool, ToolId::SpotHeal | ToolId::Remove) {
                    c |= ui
                        .checkbox(&mut aligned, tr("Aligned"))
                        .on_hover_text(tr(
                            "The offset to the source stays the same across all strokes",
                        ))
                        .changed();
                }
                if c {
                    out.push(Action::CloneOptions { all, aligned });
                }
                if m.tool == ToolId::SpotHeal {
                    hint(
                        ui,
                        tr("Paint over the spot – the source is found automatically"),
                    );
                } else if m.tool == ToolId::Remove {
                    hint(
                        ui,
                        tr("Paint over what should go – local AI (LaMa) fills it in. A selection: Shift+Alt+Backspace."),
                    );
                } else if !m.clone_source {
                    ui.separator();
                    ui.label(
                        RichText::new(tr("Alt+click sets the source"))
                            .color(Color32::from_rgb(255, 200, 120)),
                    );
                }
            }
        }
        ToolId::Wand => {
            let mut t = m.wand_tolerance;
            ui.label(tr("Tolerance"));
            if ui
                .add(egui::Slider::new(&mut t, 0.0..=255.0).max_decimals(0))
                .changed()
            {
                out.push(Action::WandTolerance(t));
            }
            let mut c = m.wand_contiguous;
            if ui
                .checkbox(&mut c, tr("Contiguous"))
                .on_hover_text(tr("only connected pixels (W)"))
                .changed()
            {
                out.push(Action::WandContiguous(c));
            }
            hint(ui, tr("Samples all layers. Shift = add, Alt = subtract."));
        }
        ToolId::Eyedropper => {
            let (mut radius, mut all) = (m.eyedrop_radius, m.eyedrop_all);
            ui.label(tr("Sample Size"));
            egui::ComboBox::from_id_salt("sample size")
                .selected_text(match radius {
                    0 => tr("Point Sample").to_string(),
                    r => format!("{n} × {n}", n = 2 * r + 1),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut radius, 0, tr("Point Sample"));
                    for r in [1, 2, 5, 15] {
                        ui.selectable_value(
                            &mut radius,
                            r,
                            format!("{n} × {n} {avg}", n = 2 * r + 1, avg = tr("Average")),
                        );
                    }
                });
            ui.checkbox(&mut all, tr("Sample All Layers"));
            if (radius, all) != (m.eyedrop_radius, m.eyedrop_all) {
                out.push(Action::Eyedrop { radius, all });
            }
            hint(
                ui,
                tr(
                    "Click or drag: picks the brush color. Alt+click does the same with the brush, eraser, gradient, shape and text tools.",
                ),
            );
        }
        ToolId::Marquee | ToolId::Ellipse | ToolId::Lasso => {
            hint(ui, tr("Shift = add, Alt = subtract, both = intersect"))
        }
        ToolId::Move => {
            if ui.button(tr("Free Transform (Ctrl+T)")).clicked() {
                out.push(Action::Command(palette::find("Free Transform")));
            }
            hint(
                ui,
                tr(
                    "Drag or arrow keys (Shift: 10 px). With a selection, only the selected pixels.",
                ),
            );
        }
        ToolId::Crop => hint(
            ui,
            tr("Drag a frame, adjust the handles. Dragging past the edge enlarges the canvas."),
        ),
        ToolId::Gradient | ToolId::Radial => {
            let mut o = m.brush;
            if slider(ui, tr("Opacity"), &mut o.opacity, 0.0..=1.0, true) {
                out.push(Action::Brush(o));
            }
            hint(ui, tr("Drag: from the brush color to transparent"));
        }
    }
}

pub(super) fn layer_section(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        let pass = m.active_pass_through == Some(true);
        egui::ComboBox::from_id_salt("blend")
            .width(130.0)
            .selected_text(if pass {
                tr("Pass Through")
            } else {
                m.active_blend.name()
            })
            .show_ui(ui, |ui| {
                if m.active_pass_through.is_some() {
                    if ui
                        .selectable_label(pass, tr("Pass Through"))
                        .on_hover_text(tr(
                            "The group's layers blend directly with what is below the group",
                        ))
                        .clicked()
                    {
                        out.push(Action::PassThrough(true));
                    }
                    ui.separator();
                }
                for b in BlendMode::ALL {
                    if ui
                        .selectable_label(!pass && b == m.active_blend, b.name())
                        .clicked()
                    {
                        out.push(Action::LayerBlend(b));
                    }
                }
            });
        let mut percent = |ui: &mut egui::Ui, value: f32, fill: bool| {
            let mut o = value;
            let r = ui.add(
                egui::DragValue::new(&mut o)
                    .range(0.0..=1.0)
                    .speed(0.005)
                    .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                    .custom_parser(|t| {
                        t.trim_end_matches('%')
                            .trim()
                            .parse::<f64>()
                            .ok()
                            .map(|v| v / 100.0)
                    }),
            );
            if r.changed() {
                out.push(Action::LayerOpacity { value: o, done: false, fill });
            }
            if r.drag_stopped() || r.lost_focus() {
                out.push(Action::LayerOpacity { value: o, done: true, fill });
            }
            r
        };
        percent(ui, m.active_opacity, false).on_hover_text(tr("Opacity (drag or type)"));
        ui.label(mosaix_core::i18n::tr_ctx("layer", "Fill"));
        percent(ui, m.active_fill, true)
            .on_hover_text(tr("Fill: opacity of the layer's own pixels; its layer style keeps full strength"));
    });
    ui.add_space(2.0);

    // Buttons stay at the bottom; the list scrolls in the space above.
    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
        ui.horizontal(|ui| {
            let cmd = palette::find;
            for (icon, label) in [
                (Icon::Plus, "New Layer"),
                (Icon::Folder, "Group Layers"),
                (Icon::Mask, "Add Layer Mask"),
                (Icon::Duplicate, "Duplicate Layer"),
            ] {
                let c = cmd(label);
                if icons::button(ui, icon, 24.0, false)
                    .on_hover_text(format!("{} ({})", c.label_text(), palette::shortcut_text(&c, &m.keymap)))
                    .clicked()
                {
                    out.push(Action::Command(c));
                }
            }
            let r = icons::button(ui, Icon::Adjust, 24.0, false)
                .on_hover_text(tr("New Adjustment Layer"));
            egui::Popup::menu(&r).show(|ui| {
                for a in mosaix_core::Adjustment::defaults() {
                    if ui.button(a.name()).clicked() {
                        out.push(Action::AddAdjustment(a));
                    }
                }
            });
            if icons::button(ui, Icon::Trash, 24.0, false)
                .on_hover_text(tr("Delete Layer"))
                .clicked()
            {
                out.push(Action::DeleteLayer);
            }
        });
        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .max_height((ui.available_height() - 8.0).max(0.0))
                .show(ui, |ui| {
                    let row_h = 34.0;
                    for row in &m.layers {
                        let active = row.id == m.active;
                        let (rect, resp) = ui.allocate_exact_size(
                            Vec2::new(ui.available_width(), row_h),
                            Sense::click_and_drag(),
                        );
                        let painter = ui.painter_at(rect);
                        if active {
                            painter.rect_filled(rect, 3.0, ACCENT.linear_multiply(0.35));
                        } else if resp.hovered() {
                            painter.rect_filled(rect, 3.0, Color32::from_white_alpha(10));
                        }
                        // Eye.
                        let eye = egui::Rect::from_center_size(
                            egui::pos2(rect.min.x + 12.0, rect.center().y),
                            Vec2::splat(18.0),
                        );
                        let eye_resp = ui.interact(eye, resp.id.with("eye"), Sense::click());
                        icons::paint(
                            &painter,
                            eye.shrink(2.0),
                            if row.visible {
                                Icon::Eye
                            } else {
                                Icon::EyeClosed
                            },
                            if row.visible {
                                Color32::from_gray(220)
                            } else {
                                Color32::from_gray(110)
                            },
                        );
                        if eye_resp.on_hover_text(tr("Show/hide")).clicked() {
                            out.push(Action::ToggleVisible(row.id));
                        }
                        let mut x = eye.max.x + 6.0 + row.depth as f32 * 14.0;
                        if row.clip {
                            // Photoshop's clipping arrow: indented, pointing down to the base.
                            let ax = x + 6.0;
                            let (top, mid) = (rect.min.y + 9.0, rect.center().y + 6.0);
                            let s = Stroke::new(1.4, Color32::from_gray(190));
                            painter.line_segment([egui::pos2(ax, top), egui::pos2(ax, mid)], s);
                            painter.line_segment(
                                [egui::pos2(ax - 3.5, mid - 4.0), egui::pos2(ax, mid)],
                                s,
                            );
                            painter.line_segment(
                                [egui::pos2(ax + 3.5, mid - 4.0), egui::pos2(ax, mid)],
                                s,
                            );
                            x += 16.0;
                        }
                        // Adjustment icon, or thumbnail on a checkerboard.
                        if row.adjustment {
                            let t = egui::Rect::from_min_size(
                                egui::pos2(x, rect.min.y + 3.0),
                                Vec2::splat(row_h - 6.0),
                            );
                            icons::paint(
                                &painter,
                                t.shrink(4.0),
                                Icon::Adjust,
                                Color32::from_gray(210),
                            );
                            x = t.max.x + 8.0;
                        } else if let Some(text) = row.vector {
                            let t = egui::Rect::from_min_size(
                                egui::pos2(x, rect.min.y + 3.0),
                                Vec2::splat(row_h - 6.0),
                            );
                            painter.rect_stroke(
                                t,
                                2.0,
                                Stroke::new(1.0, Color32::from_gray(90)),
                                egui::StrokeKind::Inside,
                            );
                            icons::paint(
                                &painter,
                                t.shrink(5.0),
                                Icon::Tool(if text { ToolId::Text } else { ToolId::Shape }),
                                Color32::from_gray(220),
                            );
                            x = t.max.x + 8.0;
                        } else {
                            let t = egui::Rect::from_min_size(
                                egui::pos2(x, rect.min.y + 3.0),
                                Vec2::splat(row_h - 6.0),
                            );
                            // A group shows its layers' thumbnails stacked (bottom first).
                            let stack: Vec<(egui::TextureId, (f32, f32))> = if row.group {
                                let i = m.layers.iter().position(|r| r.id == row.id).unwrap_or(0);
                                let mut v: Vec<_> = m.layers[i + 1..]
                                    .iter()
                                    .take_while(|r| r.depth > row.depth)
                                    .filter(|r| r.visible)
                                    .filter_map(|r| r.thumb)
                                    .collect();
                                v.reverse();
                                v
                            } else {
                                row.thumb.into_iter().collect()
                            };
                            let n = 4;
                            let c = t.width() / n as f32;
                            for j in 0..n {
                                for i in 0..n {
                                    let g = if (i + j) % 2 == 0 { 190 } else { 140 };
                                    let r = egui::Rect::from_min_size(
                                        t.min + Vec2::new(i as f32 * c, j as f32 * c),
                                        Vec2::splat(c),
                                    );
                                    painter.rect_filled(r, 0.0, Color32::from_gray(g));
                                }
                            }
                            for (tex, (u, v)) in stack {
                                let size = Vec2::new(t.width() * u, t.height() * v);
                                let img = egui::Rect::from_center_size(t.center(), size);
                                painter.image(
                                    tex,
                                    img,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(u, v),
                                    ),
                                    Color32::WHITE,
                                );
                            }
                            painter.rect_stroke(
                                t,
                                0.0,
                                Stroke::new(1.0, Color32::from_gray(70)),
                                egui::StrokeKind::Outside,
                            );
                            x = t.max.x + 8.0;
                        }
                        // Mask badge.
                        let mask_w = row_h - 6.0;
                        let name_end = rect.max.x
                            - if row.has_mask { mask_w + 10.0 } else { 4.0 }
                            - if row.fx { 20.0 } else { 0.0 };
                        if row.fx {
                            let x = rect.max.x - if row.has_mask { mask_w + 12.0 } else { 6.0 } - 8.0;
                            painter.text(
                                egui::pos2(x, rect.center().y),
                                egui::Align2::CENTER_CENTER,
                                "fx",
                                egui::FontId::proportional(11.0),
                                Color32::from_gray(190),
                            );
                        }
                        if row.has_mask {
                            let badge = egui::Rect::from_center_size(
                                egui::pos2(rect.max.x - 6.0 - mask_w * 0.5, rect.center().y),
                                Vec2::splat(mask_w),
                            );
                            let b_resp = ui.interact(badge, resp.id.with("mask"), Sense::click());
                            let editing = active && m.mask_edit;
                            match row.mask_thumb {
                                Some((tex, (u, v))) => {
                                    painter.rect_filled(badge, 0.0, Color32::from_gray(60));
                                    let img = egui::Rect::from_center_size(badge.center(), Vec2::new(badge.width() * u, badge.height() * v));
                                    painter.image(tex, img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(u, v)), Color32::WHITE);
                                }
                                None => {
                                    painter.rect_filled(badge, 0.0, Color32::from_gray(200));
                                    painter.circle_filled(badge.center(), 5.0, Color32::from_gray(30));
                                }
                            }
                            painter.rect_stroke(
                                badge,
                                0.0,
                                Stroke::new(if editing { 2.0 } else { 1.0 }, if editing { ACCENT } else { Color32::from_gray(70) }),
                                egui::StrokeKind::Outside,
                            );
                            if !row.mask_enabled {
                                // Like Photoshop: a red cross over a disabled mask.
                                let s = Stroke::new(2.0, Color32::from_rgb(220, 40, 40));
                                painter.line_segment([badge.left_top(), badge.right_bottom()], s);
                                painter.line_segment([badge.right_top(), badge.left_bottom()], s);
                            }
                            if b_resp
                .on_hover_text(tr(
                    "Mask: click to switch between painting the mask and the pixels",
                ))
                .clicked()
            {
                out.push(Action::ToggleMaskEdit(row.id));
            }
                        }
                        // Name (double-click: rename).
                        if st.rename.as_ref().is_some_and(|(id, _)| *id == row.id) {
                            let (_, text) = st.rename.as_mut().unwrap();
                            let edit_rect = egui::Rect::from_min_max(
                                egui::pos2(x, rect.min.y + 1.0),
                                egui::pos2(name_end, rect.max.y - 1.0),
                            );
                            let r = ui.put(edit_rect, egui::TextEdit::singleline(text));
                            // Enter (or clicking elsewhere) renames, Escape cancels.
                            // The focus is only requested while editing: asking
                            // every frame took it back right after Enter.
                            let (enter, escape) = ui.input(|i| {
                                (
                                    i.key_pressed(egui::Key::Enter),
                                    i.key_pressed(egui::Key::Escape),
                                )
                            });
                            if enter || escape || r.lost_focus() || r.clicked_elsewhere() {
                                let (id, name) = st.rename.take().unwrap();
                                if !escape && !name.trim().is_empty() {
                                    out.push(Action::RenameLayer(id, name.trim().to_string()));
                                }
                            } else if !r.has_focus() {
                                r.request_focus();
                            }
                        } else {
                            if row.group {
                                let r = egui::Rect::from_center_size(
                                    egui::pos2(x + 7.0, rect.center().y),
                                    Vec2::splat(14.0),
                                );
                                icons::paint(&painter, r, Icon::Folder, Color32::from_gray(190));
                                x += 18.0;
                            }
                            let galley = painter.layout_no_wrap(
                                row.name.clone(),
                                egui::FontId::proportional(13.5),
                                if row.visible {
                                    Color32::from_gray(230)
                                } else {
                                    Color32::from_gray(130)
                                },
                            );
                            painter
                                .with_clip_rect(egui::Rect::from_min_max(
                                    rect.min,
                                    egui::pos2(name_end, rect.max.y),
                                ))
                                .galley(
                                    egui::pos2(x, rect.center().y - galley.size().y / 2.0),
                                    galley,
                                    Color32::WHITE,
                                );
                        }
                        if resp.clicked() {
                            out.push(Action::SelectLayer(row.id));
                        }
                        if resp.double_clicked() {
                            st.rename = Some((row.id, row.name.clone()));
                        }
                        // Drag & drop: drop on the upper/lower third = above/below, middle of a group = into.
                        resp.dnd_set_drag_payload(row.id);
                        if let Some(pointer) = ui.ctx().pointer_interact_pos()
                            && rect.contains(pointer)
                            && let Some(payload) = resp.dnd_hover_payload::<LayerId>()
                            && *payload != row.id
                        {
                            let t = (pointer.y - rect.min.y) / rect.height();
                            let into = row.group && (0.3..0.7).contains(&t);
                            let stroke = Stroke::new(2.0, ACCENT);
                            if into {
                                painter.rect_stroke(
                                    rect.shrink(1.0),
                                    3.0,
                                    stroke,
                                    egui::StrokeKind::Inside,
                                );
                            } else {
                                let y = if t < 0.5 { rect.min.y } else { rect.max.y };
                                painter.hline(rect.x_range(), y, stroke);
                            }
                            if let Some(payload) = resp.dnd_release_payload::<LayerId>() {
                                out.push(Action::MoveLayer {
                                    id: *payload,
                                    target: row.id,
                                    above: t < 0.5,
                                    into,
                                });
                            }
                        }
                        if !row.filters.is_empty() {
                            smart_filter_rows(ui, row, row.depth as f32 * 14.0 + 30.0, st, out);
                        }
                    }
                });
        });
    });
}

pub(super) fn history_section(ui: &mut egui::Ui, m: &Model, out: &mut Vec<Action>) {
    let mut entry = |ui: &mut egui::Ui, i: usize, label: &str| {
        let done = i <= m.history_done;
        let text = RichText::new(label).color(if done {
            Color32::from_gray(220)
        } else {
            Color32::from_gray(110)
        });
        if ui.selectable_label(i == m.history_done, text).clicked() {
            out.push(Action::HistoryJump(i));
        }
    };
    egui::ScrollArea::vertical()
        .max_height(220.0)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            entry(ui, 0, tr("Initial State"));
            for (i, l) in m.history.iter().enumerate() {
                entry(ui, i + 1, l);
            }
        });
}

fn dialogs(ctx: &egui::Context, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    let Some(d) = &mut st.dialog else { return };
    let mut open = true;
    let mut done = false;
    match d {
        Dialog::Canvas { w, h, anchor } => {
            egui::Window::new(tr("Canvas Size"))
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    size_fields(ui, w, h);
                    ui.label(tr("Anchor"));
                    egui::Grid::new("anchor").show(ui, |ui| {
                        for y in 0..3u8 {
                            for x in 0..3u8 {
                                if ui
                                    .add_sized(
                                        [22.0, 22.0],
                                        egui::Button::new(if *anchor == (x, y) { "X" } else { "" })
                                            .selected(*anchor == (x, y)),
                                    )
                                    .clicked()
                                {
                                    *anchor = (x, y);
                                }
                            }
                            ui.end_row();
                        }
                    });
                    ui.label(RichText::new(tr("Nothing is deleted: shrinking only hides.")).weak());
                    if ui.button("OK").clicked() {
                        out.push(Action::CanvasSize {
                            w: *w,
                            h: *h,
                            anchor: (anchor.0 as f64 / 2.0, anchor.1 as f64 / 2.0),
                        });
                        done = true;
                    }
                });
        }
        Dialog::Image { w, h, keep } => {
            egui::Window::new(tr("Image Size"))
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    let (ow, oh) = m.doc_size;
                    let (pw, ph) = (*w, *h);
                    size_fields(ui, w, h);
                    ui.checkbox(keep, tr("Constrain Proportions"));
                    if *keep {
                        if *w != pw {
                            *h = ((*w as f64 * oh as f64 / ow as f64).round() as u32).max(1);
                        } else if *h != ph {
                            *w = ((*h as f64 * ow as f64 / oh as f64).round() as u32).max(1);
                        }
                    }
                    ui.label(
                        RichText::new(tr!(
                            "{pct} % of the current size",
                            pct = format!("{:.0}", *w as f64 / ow as f64 * 100.0)
                        ))
                        .weak(),
                    );
                    if ui.button("OK").clicked() {
                        out.push(Action::ImageSize { w: *w, h: *h });
                        done = true;
                    }
                });
        }
        Dialog::Modify { op, amount } => {
            let title = MODIFY.iter().find(|x| x.0 == *op).map_or("", |x| x.1).trim_end_matches(" …");
            egui::Window::new(tr(title))
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(if *op == "border" { tr("Width") } else { tr("Radius") });
                        let r = ui.add(egui::DragValue::new(amount).range(0.0..=500.0).speed(0.25).max_decimals(1).suffix(" px"));
                        r.request_focus();
                    });
                    if ui.button("OK").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        out.push(Action::ModifySelection { op, amount: *amount });
                        done = true;
                    }
                });
        }
        Dialog::ColorRange { color, fuzziness, image, mode } => {
            egui::Window::new(tr("Color Range"))
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    egui::Grid::new("color range").num_columns(2).show(ui, |ui| {
                        ui.label(tr("Color"));
                        ui.horizontal(|ui| {
                            ui.color_edit_button_rgb(color);
                            if ui.small_button(tr("Foreground")).clicked() {
                                *color = m.color;
                            }
                        });
                        ui.end_row();
                        ui.label(tr("Fuzziness"));
                        ui.add(egui::Slider::new(fuzziness, 0.0..=200.0).max_decimals(0));
                        ui.end_row();
                        ui.label(tr("Sample"));
                        ui.horizontal(|ui| {
                            ui.selectable_value(image, false, tr("Layer"));
                            ui.selectable_value(image, true, tr("Image"));
                        });
                        ui.end_row();
                        ui.label(tr("Selection"));
                        ui.horizontal(|ui| {
                            for (k, l) in [("replace", "New"), ("add", "Add"), ("subtract", "Subtract"), ("intersect", "Intersect")] {
                                ui.selectable_value(mode, k, tr(l));
                            }
                        });
                        ui.end_row();
                    });
                    ui.label(RichText::new(tr("Selects this color everywhere, softly by fuzziness.")).weak());
                    if ui.button("OK").clicked() {
                        let hex = format!(
                            "#{:02x}{:02x}{:02x}",
                            (color[0] * 255.0).round() as u8,
                            (color[1] * 255.0).round() as u8,
                            (color[2] * 255.0).round() as u8
                        );
                        out.push(Action::Agent(serde_json::json!({
                            "cmd": "select_color_range", "color": hex, "fuzziness": *fuzziness,
                            "sample": if *image { "image" } else { "layer" }, "mode": *mode,
                        })));
                        done = true;
                    }
                });
        }
        Dialog::SaveSelection { name } => {
            egui::Window::new(tr("Save Selection"))
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(tr("Name"));
                        ui.text_edit_singleline(name).request_focus();
                    });
                    if m.channels.iter().any(|c| c == name.trim()) {
                        ui.label(RichText::new(tr("Replaces the saved selection of that name.")).weak());
                    }
                    let ok = !name.trim().is_empty();
                    if ui.add_enabled(ok, egui::Button::new("OK")).clicked() || (ok && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                        out.push(Action::Agent(serde_json::json!({ "cmd": "select_save", "name": name.trim() })));
                        done = true;
                    }
                });
        }
    }
    if !open || done {
        st.dialog = None;
    }
}

fn size_fields(ui: &mut egui::Ui, w: &mut u32, h: &mut u32) {
    egui::Grid::new("size").show(ui, |ui| {
        ui.label(tr("Width"));
        ui.add(egui::DragValue::new(w).range(1..=300_000).suffix(" px"));
        ui.end_row();
        ui.label(tr("Height"));
        ui.add(egui::DragValue::new(h).range(1..=300_000).suffix(" px"));
        ui.end_row();
    });
}

fn command_palette(
    ctx: &egui::Context,
    keymap: &palette::Keymap,
    query: &mut String,
    sel: &mut usize,
    out: &mut Vec<Action>,
    close: &mut bool,
) {
    let hits = palette::search(query);
    *sel = (*sel).min(hits.len().saturating_sub(1));
    let (up, down, enter, esc) = ctx.input(|i| {
        (
            i.key_pressed(egui::Key::ArrowUp),
            i.key_pressed(egui::Key::ArrowDown),
            i.key_pressed(egui::Key::Enter),
            i.key_pressed(egui::Key::Escape),
        )
    });
    if up {
        *sel = sel.saturating_sub(1);
    }
    if down && *sel + 1 < hits.len() {
        *sel += 1;
    }
    if esc {
        *close = true;
        return;
    }
    if enter {
        if let Some(c) = hits.get(*sel) {
            out.push(Action::Command(**c));
        }
        *close = true;
        return;
    }
    let screen = ctx.content_rect();
    let width = (screen.width() - 80.0).clamp(320.0, 640.0);
    let list_h = (screen.height() * 0.55).clamp(160.0, 520.0);
    egui::Window::new(tr("Command"))
        .title_bar(false)
        .anchor(egui::Align2::CENTER_TOP, Vec2::new(0.0, 70.0))
        .fixed_size(Vec2::new(width, list_h + 60.0))
        .frame(egui::Frame::window(&ctx.global_style()).inner_margin(10.0))
        .show(ctx, |ui| {
            let r = ui.add(
                egui::TextEdit::singleline(query)
                    .hint_text(tr("Search commands …"))
                    .font(egui::FontId::proportional(17.0))
                    .margin(Vec2::new(8.0, 6.0))
                    .desired_width(f32::INFINITY),
            );
            r.request_focus();
            ui.add_space(6.0);
            egui::ScrollArea::vertical()
                .max_height(list_h)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if hits.is_empty() {
                        ui.label(RichText::new(tr("No matching command")).weak());
                    }
                    let w = ui.available_width();
                    for (i, c) in hits.iter().enumerate() {
                        let active = i == *sel;
                        let (rect, resp) =
                            ui.allocate_exact_size(Vec2::new(w, 28.0), Sense::click());
                        let p = ui.painter();
                        if active || resp.hovered() {
                            p.rect_filled(
                                rect,
                                4.0,
                                if active {
                                    Color32::from_rgb(52, 98, 196)
                                } else {
                                    Color32::from_gray(58)
                                },
                            );
                        }
                        let y = rect.center().y;
                        p.text(
                            egui::pos2(rect.left() + 10.0, y),
                            egui::Align2::LEFT_CENTER,
                            c.label_text(),
                            egui::FontId::proportional(14.5),
                            Color32::from_gray(240),
                        );
                        let menu_col = if active {
                            Color32::from_gray(215)
                        } else {
                            Color32::from_gray(150)
                        };
                        let shortcut = palette::shortcut_text(c, keymap);
                        let right = if shortcut.is_empty() {
                            c.menu_text().to_string()
                        } else {
                            format!("{}    {}", c.menu_text(), shortcut)
                        };
                        p.text(
                            egui::pos2(rect.right() - 10.0, y),
                            egui::Align2::RIGHT_CENTER,
                            right,
                            egui::FontId::proportional(12.5),
                            menu_col,
                        );
                        if active && (up || down) {
                            resp.scroll_to_me(None);
                        }
                        if resp.hovered() && ui.ctx().input(|i| i.pointer.delta() != Vec2::ZERO) {
                            *sel = i;
                        }
                        if resp.clicked() {
                            out.push(Action::Command(**c));
                            *close = true;
                        }
                    }
                });
            ui.add_space(4.0);
            ui.label(
                RichText::new(tr("↑↓ choose · Enter run · Esc close"))
                    .weak()
                    .small(),
            );
        });
}

/// Rows of the layer panel (top → bottom).
pub fn layer_rows<T: Clone>(doc: &mosaix_core::Document<T>) -> Vec<LayerRow> {
    doc.panel_rows()
        .into_iter()
        .filter_map(|(id, depth)| {
            let l = doc.find(id)?;
            Some(LayerRow {
                id,
                depth,
                name: l.name.clone(),
                visible: l.visible,
                group: l.is_group(),
                has_mask: l.mask.is_some(),
                mask_enabled: l.mask_enabled,
                thumb: None,
                adjustment: l.adjustment().is_some(),
                filters: l.filters.clone(),
                vector: l
                    .vector()
                    .map(|v| matches!(v.kind, mosaix_core::vector::VectorKind::Text(_))),
                fx: l.effects.iter().any(|e| e.enabled),
                clip: l.clip,
                mask_thumb: None,
            })
        })
        .collect()
}

fn filter_dialog(ctx: &egui::Context, st: &mut State, out: &mut Vec<Action>) {
    use mosaix_core::Filter;
    let Some(d) = &mut st.filter else {
        st.filter_preview = None;
        return;
    };
    let mut open = true;
    let mut apply = false;
    let mut cancel = false;
    let title = if d.edit.is_some() {
        tr!("{name} (Smart Filter)", name = d.filter.name())
    } else {
        d.filter.name().to_string()
    };
    egui::Window::new(title)
        .id(egui::Id::unique("filter dialog"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_pos(ctx.content_rect().center() - Vec2::new(160.0, 120.0))
        .show(ctx, |ui| {
            let s = |ui: &mut egui::Ui, label: &str, v: &mut f32, r: std::ops::RangeInclusive<f32>, log: bool, suffix: &str| {
                ui.horizontal(|ui| {
                    ui.add_sized([90.0, 18.0], egui::Label::new(label));
                    ui.add(egui::Slider::new(v, r).logarithmic(log).suffix(suffix).max_decimals(1));
                });
            };
            match &mut d.filter {
                Filter::GaussianBlur { radius } => s(ui, "Radius", radius, 0.1..=1000.0, true, " px"),
                Filter::UnsharpMask { amount, radius, threshold } => {
                    s(ui, tr("Amount"), amount, 1.0..=500.0, false, " %");
                    s(ui, "Radius", radius, 0.1..=250.0, true, " px");
                    s(ui, tr("Threshold"), threshold, 0.0..=255.0, false, "");
                }
                Filter::HighPass { radius } => s(ui, "Radius", radius, 0.1..=1000.0, true, " px"),
                Filter::AddNoise { amount, monochrome, gaussian, seed } => {
                    s(ui, tr("Amount"), amount, 0.0..=100.0, false, " %");
                    ui.checkbox(monochrome, tr("Monochromatic"));
                    ui.checkbox(gaussian, tr("Gaussian"));
                    let mut sd = *seed as f32;
                    s(ui, tr("Seed"), &mut sd, 0.0..=100.0, false, "");
                    *seed = sd as u32;
                }
                Filter::Plugin { source, values, .. } => match mosaix_core::plugin::parse(source) {
                    Err(e) => {
                        ui.colored_label(ui.visuals().error_fg_color, e);
                    }
                    Ok(spec) => {
                        if !spec.description.is_empty() {
                            ui.label(RichText::new(&spec.description).weak());
                        }
                        use mosaix_core::plugin::ParamKind;
                        for p in &spec.params {
                            let v = values.entry(p.id.clone()).or_insert_with(|| p.default.clone());
                            if v.len() != p.width() {
                                *v = p.default.clone();
                            }
                            match &p.kind {
                                ParamKind::Number { min, max, unit, log } => {
                                    let suffix = if unit.is_empty() { String::new() } else { format!(" {unit}") };
                                    s(ui, &p.label, &mut v[0], *min..=*max, *log, &suffix);
                                }
                                ParamKind::Bool => {
                                    let mut b = v[0] > 0.5;
                                    if ui.checkbox(&mut b, &p.label).changed() {
                                        v[0] = b as u8 as f32;
                                    }
                                }
                                ParamKind::Color => {
                                    ui.horizontal(|ui| {
                                        ui.add_sized([90.0, 18.0], egui::Label::new(&p.label));
                                        let mut rgb = egui::Rgba::from_rgb(v[0], v[1], v[2]);
                                        if egui::color_picker::color_edit_button_rgba(ui, &mut rgb, egui::color_picker::Alpha::Opaque).changed() {
                                            *v = vec![rgb.r(), rgb.g(), rgb.b()];
                                        }
                                    });
                                }
                            }
                        }
                    }
                },
            }
            ui.checkbox(&mut d.preview, tr("Preview"));
            if d.edit.is_none() {
                ui.checkbox(&mut d.smart, tr("As Smart Filter"))
                    .on_hover_text(tr("Non-destructive: stays on the layer, can be changed or switched off at any time (affects the whole layer)"));
            }
            ui.horizontal(|ui| {
                let label = if d.edit.is_some() { tr("Apply") } else { tr("Apply Filter") };
                if ui.button(label).clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    apply = true;
                }
                if ui.button(tr("Cancel")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    cancel = true;
                }
            });
        });
    if apply {
        let f = d.filter.clone();
        out.push(match d.edit {
            Some((layer, index)) => Action::SetSmartFilter {
                layer,
                index,
                filter: f,
            },
            None if d.smart => Action::AddSmartFilter(f),
            None => Action::ApplyFilter(f),
        });
    }
    if d.edit.is_none() {
        st.smart_filters = d.smart;
    }
    if apply || cancel || !open {
        st.filter = None;
        st.filter_preview = None;
    } else {
        st.filter_preview = d.preview.then(|| match d.edit {
            Some((_, i)) => FilterPreview::Smart(Some(i), d.filter.clone()),
            None if d.smart => FilterPreview::Smart(None, d.filter.clone()),
            None => FilterPreview::Pixels(d.filter.clone()),
        });
    }
}

/// The smart filters under a layer row: a header with "apply" and one row
/// per filter (eye, name: click to edit, remove).
fn smart_filter_rows(
    ui: &mut egui::Ui,
    row: &LayerRow,
    indent: f32,
    st: &mut State,
    out: &mut Vec<Action>,
) {
    let h = 22.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), h), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.text(
        egui::pos2(rect.min.x + indent, rect.center().y),
        egui::Align2::LEFT_CENTER,
        tr("Smart Filters"),
        egui::FontId::proportional(12.0),
        Color32::from_gray(150),
    );
    let bake = egui::Rect::from_min_max(
        egui::pos2(rect.max.x - 74.0, rect.min.y + 2.0),
        egui::pos2(rect.max.x - 4.0, rect.max.y - 2.0),
    );
    if ui
        .put(
            bake,
            egui::Button::new(RichText::new(tr("Apply All")).small()),
        )
        .on_hover_text(tr(
            "Apply the filters to the layer's pixels and remove them",
        ))
        .clicked()
    {
        out.push(Action::BakeSmartFilters(row.id));
    }
    for (index, sf) in row.filters.iter().enumerate() {
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), h), Sense::click());
        let painter = ui.painter_at(rect);
        if resp.hovered() {
            painter.rect_filled(rect, 3.0, Color32::from_white_alpha(8));
        }
        let eye = egui::Rect::from_center_size(
            egui::pos2(rect.min.x + indent + 7.0, rect.center().y),
            Vec2::splat(14.0),
        );
        let eye_resp = ui.interact(eye, resp.id.with("eye"), Sense::click());
        icons::paint(
            &painter,
            eye,
            if sf.enabled {
                Icon::Eye
            } else {
                Icon::EyeClosed
            },
            if sf.enabled {
                Color32::from_gray(200)
            } else {
                Color32::from_gray(100)
            },
        );
        if eye_resp.on_hover_text(tr("Turn filter on/off")).clicked() {
            out.push(Action::ToggleSmartFilter {
                layer: row.id,
                index,
            });
        }
        painter.text(
            egui::pos2(eye.max.x + 8.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            sf.filter.name(),
            egui::FontId::proportional(12.5),
            if sf.enabled {
                Color32::from_gray(215)
            } else {
                Color32::from_gray(120)
            },
        );
        let del = egui::Rect::from_center_size(
            egui::pos2(rect.max.x - 12.0, rect.center().y),
            Vec2::splat(16.0),
        );
        let del_resp = ui.interact(del, resp.id.with("del"), Sense::click());
        if del_resp.hovered() || resp.hovered() {
            let c = if del_resp.hovered() {
                Color32::from_rgb(255, 120, 100)
            } else {
                Color32::from_gray(140)
            };
            let r = del.shrink(4.0);
            painter.line_segment([r.left_top(), r.right_bottom()], Stroke::new(1.5, c));
            painter.line_segment([r.right_top(), r.left_bottom()], Stroke::new(1.5, c));
        }
        if del_resp.on_hover_text(tr("Remove smart filter")).clicked() {
            out.push(Action::RemoveSmartFilter {
                layer: row.id,
                index,
            });
        } else if resp.on_hover_text(tr("Click to edit")).clicked() {
            out.push(Action::SelectLayer(row.id));
            st.filter = Some(FilterDialog {
                filter: sf.filter.clone(),
                preview: true,
                smart: true,
                edit: Some((row.id, index)),
            });
        }
    }
}

/// Edit › Keyboard Shortcuts: every menu command with its shortcut; click
/// one and press the new keys.
fn shortcuts_dialog(ctx: &egui::Context, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    let Some(d) = &mut st.shortcuts else { return };
    let mut open = true;
    egui::Window::new(tr("Keyboard Shortcuts"))
        .open(&mut open)
        .default_size([520.0, 560.0])
        .default_pos(ctx.content_rect().center() - Vec2::new(260.0, 300.0))
        .collapsible(false)
        .show(ctx, |ui| {
            ui.add(egui::TextEdit::singleline(&mut d.query).hint_text(tr("Search commands …")).desired_width(f32::INFINITY));
            ui.label(
                RichText::new(tr("Click a shortcut, then press the new keys. Backspace removes it, Esc cancels. Keys taken from another command are removed there."))
                    .weak()
                    .small(),
            );
            ui.separator();
            let q = d.query.to_lowercase();
            egui::ScrollArea::vertical().auto_shrink([false, false]).max_height(460.0).show(ui, |ui| {
                egui::Grid::new("shortcuts").num_columns(3).striped(true).spacing([12.0, 4.0]).show(ui, |ui| {
                    for menu in MENUS {
                        let cmds: Vec<&palette::Cmd> = COMMANDS
                            .iter()
                            .filter(|c| c.menu == menu)
                            .filter(|c| q.is_empty() || c.label_text().to_lowercase().contains(&q) || c.label.to_lowercase().contains(&q))
                            .collect();
                        if cmds.is_empty() {
                            continue;
                        }
                        ui.label(RichText::new(palette::menu_text(menu)).strong());
                        ui.end_row();
                        for c in cmds {
                            let id = c.id();
                            ui.label(c.label_text());
                            let capturing = d.capturing.as_deref() == Some(id.as_str());
                            let text = if capturing {
                                tr("Press keys …").to_string()
                            } else {
                                let s = palette::shortcut_text(c, &m.keymap);
                                if s.is_empty() { "—".to_string() } else { s }
                            };
                            if ui.add(egui::Button::selectable(capturing, text).min_size(Vec2::new(150.0, 0.0))).clicked() {
                                d.capturing = if capturing { None } else { Some(id.clone()) };
                            }
                            if m.keymap.contains_key(&id) {
                                if ui.small_button(tr("Default")).on_hover_text(format!("{} {}", tr("Default:"), palette::Combo::of(c).text())).clicked() {
                                    out.push(Action::Shortcut { id, combo: None, reset: true });
                                }
                            } else {
                                ui.label("");
                            }
                            ui.end_row();
                        }
                    }
                });
            });
            ui.separator();
            if ui.add_enabled(!m.keymap.is_empty(), egui::Button::new(tr("Reset All"))).clicked() {
                out.push(Action::ResetShortcuts);
            }
        });
    if !open {
        st.shortcuts = None;
    }
}
