//! Panel layout. `build` reads a [`Model`] (snapshot of the app) and returns
//! [`Action`]s; it never touches the engine directly.

use egui::{Align, Color32, Layout, RichText, Sense, Stroke, Vec2};
use imagine_core::{BlendMode, LayerId, TIP_NAMES, tr};

use super::icons::{self, Icon};
use super::palette::{self, COMMANDS, Cmd, MENUS};

/// Tools as shown in the toolbar: (id, name, shortcut); English, translated
/// when shown.
pub const TOOLS: [(ToolId, &str, &str); 17] = [
    (ToolId::Move, "Move", "V"),
    (ToolId::Marquee, "Rectangular Marquee", "M"),
    (ToolId::Ellipse, "Elliptical Marquee", "M"),
    (ToolId::Lasso, "Lasso", "L"),
    (ToolId::Wand, "Magic Wand", "W"),
    (ToolId::Crop, "Crop", "C"),
    (ToolId::Brush, "Brush", "B"),
    (ToolId::Smudge, "Smudge", "U"),
    (ToolId::Clone, "Clone Stamp", "S"),
    (ToolId::SpotHeal, "Spot Healing Brush", "J"),
    (ToolId::Heal, "Healing Brush", "J"),
    (ToolId::Eraser, "Eraser", "E"),
    (ToolId::Text, "Text", "T"),
    (ToolId::Shape, "Shape (Rectangle / Ellipse)", "Shift+U"),
    (ToolId::Pen, "Pen", "P"),
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
    Eraser,
    Gradient,
    Radial,
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
    pub filters: Vec<imagine_core::SmartFilter>,
    /// Text (true) or shape (false) layer.
    pub vector: Option<bool>,
    /// Has layer styles switched on.
    pub fx: bool,
}

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
    pub filter: imagine_core::Filter,
    preview: bool,
    smart: bool,
    /// Editing smart filter `index` of `layer`.
    edit: Option<(LayerId, usize)>,
}

/// What the canvas previews for the filter dialog.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPreview {
    Pixels(imagine_core::Filter),
    /// Smart filter `Some(index)` replaced, or one added.
    Smart(Option<usize>, imagine_core::Filter),
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
    pub boxed: Option<&'static str>,
    /// Free transform: current angle (degrees).
    pub box_angle: Option<f64>,
    /// Liquify session: tool, brush diameter, strength.
    pub liquify: Option<(imagine_gpu::LiquifyTool, f32, f32)>,
    /// Text edited on the canvas: caret, selection, frame (screen points).
    pub text_overlay: Option<TextOverlay>,
    /// Pen: the path being drawn or edited (screen points).
    pub path_overlay: Option<PathOverlay>,
    pub mask_edit: bool,
    pub layers: Vec<LayerRow>,
    pub active: LayerId,
    pub active_opacity: f32,
    pub active_blend: BlendMode,
    pub history: Vec<String>,
    pub history_done: usize,
    pub has_selection: bool,
    pub status: String,
    pub busy: bool,
    pub recent: Vec<std::path::PathBuf>,
    pub active_adjustment: Option<imagine_core::Adjustment>,
    pub active_vector: Option<imagine_core::VectorLayer>,
    /// Layer styles of the active layer (`None` for groups and adjustments).
    pub active_effects: Option<Vec<imagine_core::style::Effect>>,
    /// Installed font families (loaded when first needed).
    pub fonts: std::sync::Arc<Vec<imagine_vector::FontGroup>>,
    pub favourite_fonts: Vec<String>,
    pub recent_fonts: Vec<String>,
    /// Shape tool: ellipse instead of rectangle.
    pub shape_ellipse: bool,
    /// Histogram of what is below the active adjustment (levels and curves).
    pub histogram: Option<std::sync::Arc<imagine_gpu::Histogram>>,
}

#[derive(Debug, Clone)]
pub enum Action {
    Command(Cmd),
    Tool(ToolId),
    Color([f32; 3]),
    Brush(BrushModel),
    SmudgeStrength(f32),
    CloneOptions {
        all: bool,
        aligned: bool,
    },
    SetVector {
        v: imagine_core::VectorLayer,
        done: bool,
    },
    FontUsed(String),
    ToggleFavouriteFont(String),
    SetEffects {
        effects: Vec<imagine_core::style::Effect>,
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
    },
    LayerBlend(BlendMode),
    DeleteLayer,
    HistoryJump(usize),
    CommitBox,
    CancelBox,
    /// Free transform: turn by (degrees), set the angle, mirror.
    BoxRotate(f64),
    BoxAngle(f64),
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
    OpenFile(std::path::PathBuf),
    SaveAs(std::path::PathBuf),
    ExportAs(std::path::PathBuf, imagine_io::export::ExportOptions),
    NewDocument {
        w: u32,
        h: u32,
        white: bool,
    },
    Unsaved(super::files::Answer),
    AddAdjustment(imagine_core::Adjustment),
    ApplyFilter(imagine_core::Filter),
    AddSmartFilter(imagine_core::Filter),
    SetSmartFilter {
        layer: LayerId,
        index: usize,
        filter: imagine_core::Filter,
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
        tool: imagine_gpu::LiquifyTool,
        size: f32,
        strength: f32,
    },
    LiquifyCommit,
    LiquifyCancel,
    LiquifyUndo,
    SetAdjustment {
        adj: imagine_core::Adjustment,
        done: bool,
    },
}

/// UI state that lives across frames.
#[derive(Default)]
pub struct State {
    pub hidden: bool,
    /// Area left for the canvas (points); `None` = whole window.
    pub canvas: Option<egui::Rect>,
    pub palette: Option<(String, usize)>,
    rename: Option<(LayerId, String)>,
    dialog: Option<Dialog>,
    show_log: bool,
    /// Picker state: HSV and the linear colour it produced.
    hsv: Option<([f32; 3], [f32; 3])>,
    hex: Option<String>,
    pub file: Option<super::files::FileDialog>,
    pub new_doc: Option<super::files::NewDialog>,
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
    /// Upper tab of the right dock.
    pub dock_tab: DockTab,
    /// Whether the active layer had properties last frame (switches the tab).
    had_props: bool,
}

/// Tabs above the layers in the right dock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DockTab {
    #[default]
    Properties,
    Styles,
    Color,
    History,
}

enum Dialog {
    Canvas { w: u32, h: u32, anchor: (u8, u8) },
    Image { w: u32, h: u32, keep: bool },
}

const ACCENT: Color32 = Color32::from_rgb(90, 150, 255);

impl State {
    /// Expands an effect in the layer-styles section.
    pub fn open_effect(&mut self, i: usize) {
        self.fx_open = Some(i);
        self.dock_tab = DockTab::Styles;
    }

    pub fn open_filter(&mut self, f: imagine_core::Filter) {
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
            || self.dialog.is_some()
            || self.filter.is_some()
            || self.vec_editor.fonts.open
    }
}

pub fn style(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::dark());
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = Vec2::new(6.0, 4.0);
        s.spacing.slider_width = 150.0;
        s.visuals.selection.bg_fill = ACCENT.linear_multiply(0.6);
        s.visuals.panel_fill = Color32::from_rgb(30, 31, 35);
        s.visuals.window_fill = Color32::from_rgb(36, 37, 42);
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
    filter_dialog(ctx, st, &mut out);
    if let Some((query, sel)) = &mut st.palette {
        let mut close = false;
        command_palette(ctx, query, sel, &mut out, &mut close);
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
        egui::Panel::right("side")
            .default_size(300.0)
            .min_size(240.0)
            .show(ui, |ui| dock(ui, m, st, &mut out));
        st.canvas = Some(ui.available_rect_before_wrap());
    });
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
    out
}

fn level_color(l: tracing::Level) -> Color32 {
    match l {
        tracing::Level::ERROR => Color32::from_rgb(255, 110, 100),
        tracing::Level::WARN => Color32::from_rgb(255, 200, 90),
        _ => Color32::from_gray(200),
    }
}

fn menu_bar(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    egui::MenuBar::new().ui(ui, |ui| {
        for menu in MENUS {
            ui.menu_button(palette::menu_text(menu), |ui| {
                for c in COMMANDS.iter().filter(|c| c.menu == menu) {
                    if ui
                        .add(egui::Button::new(c.label_text()).shortcut_text(c.shortcut()))
                        .clicked()
                    {
                        out.push(Action::Command(*c));
                        ui.close();
                    }
                }
                if menu == "File" && !m.recent.is_empty() {
                    ui.separator();
                    ui.menu_button(tr("Open Recent"), |ui| {
                        for p in &m.recent {
                            let name = p
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            if ui
                                .button(name)
                                .on_hover_text(p.display().to_string())
                                .clicked()
                            {
                                out.push(Action::OpenFile(p.clone()));
                                ui.close();
                            }
                        }
                    });
                }
                if menu == "Image" {
                    ui.separator();
                    ui.menu_button("Filter", |ui| {
                        for f in imagine_core::Filter::defaults() {
                            if ui.button(format!("{} …", f.name())).clicked() {
                                st.open_filter(f);
                                ui.close();
                            }
                        }
                    });
                }
                if menu == "Layer" {
                    ui.separator();
                    ui.menu_button(tr("New Adjustment Layer"), |ui| {
                        for a in imagine_core::Adjustment::defaults() {
                            if ui.button(a.name()).clicked() {
                                out.push(Action::AddAdjustment(a));
                                ui.close();
                            }
                        }
                    });
                }
                if menu == "Image" {
                    ui.separator();
                    if ui.button(tr("Canvas Size …")).clicked() {
                        st.dialog = Some(Dialog::Canvas {
                            w: m.doc_size.0,
                            h: m.doc_size.1,
                            anchor: (1, 1),
                        });
                        ui.close();
                    }
                    if ui.button(tr("Image Size …")).clicked() {
                        st.dialog = Some(Dialog::Image {
                            w: m.doc_size.0,
                            h: m.doc_size.1,
                            keep: true,
                        });
                        ui.close();
                    }
                }
                if menu == "View" {
                    ui.separator();
                    if ui.button(tr("Messages")).clicked() {
                        st.show_log = true;
                        ui.close();
                    }
                }
            });
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{}{}", if m.modified { "• " } else { "" }, m.name)).strong(),
            );
        });
    });
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

fn toolbar(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    ui.add_space(4.0);
    ui.vertical_centered(|ui| {
        for (id, name, key) in TOOLS {
            if icons::button(ui, Icon::Tool(id), 34.0, m.tool == id)
                .on_hover_text(format!(
                    "{} ({})",
                    tr(name),
                    key.replace("Shift", tr("Shift"))
                ))
                .clicked()
            {
                out.push(Action::Tool(id));
            }
        }
        ui.add_space(10.0);
        // Current colour; click for the picker in the side panel.
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(30.0), Sense::click());
        ui.painter().rect_filled(rect, 4.0, linear_color(m.color));
        ui.painter().rect_stroke(
            rect,
            4.0,
            Stroke::new(1.0, Color32::from_gray(160)),
            egui::StrokeKind::Outside,
        );
        if resp
            .on_hover_text(tr("Brush color (X: black/white)"))
            .clicked()
        {
            st.dock_tab = DockTab::Color;
        }
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
            for (lt, key) in imagine_gpu::LiquifyTool::ALL
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
                if ui.small_button("↔").on_hover_text(tr("horizontal")).clicked() {
                    out.push(Action::BoxFlip { x: true, y: false });
                }
                if ui.small_button("↕").on_hover_text(tr("vertical")).clicked() {
                    out.push(Action::BoxFlip { x: false, y: true });
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
                hint(ui, tr("Drag corners: scale (Shift: free) · drag outside: rotate (Shift: 15°) · Alt: around the center"));
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

/// The right dock: tabs (properties, styles, colour, history) above the
/// layers, which always stay visible; the divider can be dragged.
fn dock(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    // A newly selected text, shape or adjustment layer shows its properties.
    let has_props = m.active_vector.is_some() || m.active_adjustment.is_some();
    if has_props && !st.had_props {
        st.dock_tab = DockTab::Properties;
    }
    st.had_props = has_props;
    egui::Panel::bottom("layers dock")
        .resizable(true)
        .default_size(340.0)
        .min_size(140.0)
        .max_size((ui.available_height() - 120.0).max(140.0))
        .show(ui, |ui| {
            ui.add_space(4.0);
            ui.label(RichText::new(tr("Layers")).strong());
            layer_section(ui, m, st, out);
        });
    let fx = m.active_effects.as_ref().map_or(0, Vec::len);
    let styles = if fx > 0 {
        tr!("Styles ({n})", n = fx)
    } else {
        tr("Styles").to_string()
    };
    ui.horizontal(|ui| {
        for (tab, label) in [
            (DockTab::Properties, tr("Properties").to_string()),
            (DockTab::Styles, styles),
            (DockTab::Color, tr("Color").to_string()),
            (DockTab::History, tr("History").to_string()),
        ] {
            if ui.selectable_label(st.dock_tab == tab, label).clicked() {
                st.dock_tab = tab;
            }
        }
    });
    ui.separator();
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| match st.dock_tab {
            DockTab::Properties => properties_tab(ui, m, st, out),
            DockTab::Styles => match &m.active_effects {
                Some(fx) => {
                    if let Some((effects, done)) = super::style_ui::editor(ui, fx, &mut st.fx_open)
                    {
                        out.push(Action::SetEffects { effects, done });
                    }
                }
                None => {
                    ui.label(RichText::new(tr("Select a layer to add effects.")).weak());
                }
            },
            DockTab::Color => color_section(ui, m, st, out),
            DockTab::History => history_section(ui, m, out),
        });
}

fn properties_tab(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    if let Some(v) = &m.active_vector {
        let e = super::vector_ui::editor(
            ui,
            v,
            &m.fonts,
            &m.favourite_fonts,
            &m.recent_fonts,
            m.color,
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
    } else if let Some(adj) = &m.active_adjustment {
        if let Some((adj, done)) =
            super::adjust_ui::editor(ui, adj, &mut st.adj_editor, m.histogram.as_deref())
        {
            out.push(Action::SetAdjustment { adj, done });
        }
    } else {
        ui.label(
            RichText::new(tr(
                "Pixel layer – text, shape and adjustment layers show their settings here.",
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

fn color_section(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    use egui::ecolor::Hsva;
    let current: Color32 = linear_color(m.color);
    // Keep our own HSV so hue survives black/grey colours.
    let hsv = match st.hsv {
        Some((hsv, seen)) if seen == m.color => hsv,
        _ => {
            let h = Hsva::from(current);
            [h.h, h.s, h.v]
        }
    };
    let mut hsv = hsv;
    let width = ui.available_width().min(260.0);
    let (sv_rect, sv) =
        ui.allocate_exact_size(Vec2::new(width, width * 0.62), Sense::click_and_drag());
    let (hue_rect, hue) = ui.allocate_exact_size(Vec2::new(width, 14.0), Sense::click_and_drag());
    let p = ui.painter();
    // Saturation/value field: a grid of vertex-coloured quads.
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
    let mut emit = |c: Color32, hsv: [f32; 3], st: &mut State| {
        let r = egui::Rgba::from(c);
        let lin = [r.r(), r.g(), r.b()];
        st.hsv = Some((hsv, lin));
        out.push(Action::Color(lin));
    };
    if changed {
        emit(Hsva::new(hsv[0], hsv[1], hsv[2], 1.0).into(), hsv, st);
    }
    ui.add_space(2.0);
    ui.horizontal_wrapped(|ui| {
        for s in SWATCHES {
            let col = Color32::from_rgb(s[0], s[1], s[2]);
            let (rect, resp) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::click());
            ui.painter().rect_filled(rect, 3.0, col);
            if resp.clicked() {
                let h = Hsva::from(col);
                emit(col, [h.h, h.s, h.v], st);
            }
        }
    });
    // Hex field.
    let [r, g, b, _] = current.to_array();
    let mut hex = st
        .hex
        .clone()
        .unwrap_or_else(|| format!("#{r:02x}{g:02x}{b:02x}"));
    ui.horizontal(|ui| {
        let (sw, _) = ui.allocate_exact_size(Vec2::new(28.0, 18.0), Sense::hover());
        ui.painter().rect_filled(sw, 3.0, current);
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
            if h.len() == 6
                && let Ok(v) = u32::from_str_radix(h, 16)
            {
                let col = Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8);
                let hs = Hsva::from(col);
                emit(col, [hs.h, hs.s, hs.v], st);
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
        | ToolId::SpotHeal => {
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
            egui::ComboBox::from_id_salt("tip")
                .selected_text(tr(TIP_NAMES[b.tip as usize]))
                .width(80.0)
                .show_ui(ui, |ui| {
                    for (i, n) in TIP_NAMES.iter().enumerate() {
                        changed |= ui.selectable_value(&mut b.tip, i as u32, tr(n)).changed();
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
            } else if m.tool != ToolId::SpotHeal {
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
            if matches!(m.tool, ToolId::Clone | ToolId::Heal | ToolId::SpotHeal) {
                ui.separator();
                let (mut all, mut aligned) = (m.clone_all, m.clone_aligned);
                let tip = if m.tool == ToolId::SpotHeal {
                    tr("Heal on an empty layer")
                } else {
                    tr("The source is the whole image – for stamping onto an empty layer")
                };
                let mut c = ui
                    .checkbox(&mut all, tr("Sample All Layers"))
                    .on_hover_text(tip)
                    .changed();
                if m.tool != ToolId::SpotHeal {
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

fn layer_section(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("blend")
            .width(130.0)
            .selected_text(m.active_blend.name())
            .show_ui(ui, |ui| {
                for b in BlendMode::ALL {
                    if ui.selectable_label(b == m.active_blend, b.name()).clicked() {
                        out.push(Action::LayerBlend(b));
                    }
                }
            });
        let mut o = m.active_opacity;
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
            out.push(Action::LayerOpacity {
                value: o,
                done: false,
            });
        }
        if r.drag_stopped() || r.lost_focus() {
            out.push(Action::LayerOpacity {
                value: o,
                done: true,
            });
        }
        r.on_hover_text(tr("Opacity (drag or type)"));
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
                    .on_hover_text(format!("{} ({})", c.label_text(), c.shortcut()))
                    .clicked()
                {
                    out.push(Action::Command(c));
                }
            }
            let r = icons::button(ui, Icon::Adjust, 24.0, false)
                .on_hover_text(tr("New Adjustment Layer"));
            egui::Popup::menu(&r).show(|ui| {
                for a in imagine_core::Adjustment::defaults() {
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
                .auto_shrink([false; 2])
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
                        } else if !row.group {
                            let t = egui::Rect::from_min_size(
                                egui::pos2(x, rect.min.y + 3.0),
                                Vec2::splat(row_h - 6.0),
                            );
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
                            if let Some((tex, (u, v))) = row.thumb {
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
                        let name_end = rect.max.x
                            - if row.has_mask { 28.0 } else { 4.0 }
                            - if row.fx { 20.0 } else { 0.0 };
                        if row.fx {
                            let x = rect.max.x - if row.has_mask { 30.0 } else { 6.0 } - 8.0;
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
                                egui::pos2(rect.max.x - 15.0, rect.center().y),
                                Vec2::new(22.0, 16.0),
                            );
                            let b_resp = ui.interact(badge, resp.id.with("mask"), Sense::click());
                            let editing = active && m.mask_edit;
                            painter.rect(
                                badge,
                                3.0,
                                if row.mask_enabled {
                                    Color32::from_gray(200)
                                } else {
                                    Color32::from_gray(80)
                                },
                                Stroke::new(if editing { 2.0 } else { 0.0 }, ACCENT),
                                egui::StrokeKind::Outside,
                            );
                            painter.circle_filled(badge.center(), 4.0, Color32::from_gray(30));
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
                            r.request_focus();
                            if r.lost_focus() {
                                let (id, name) = st.rename.take().unwrap();
                                if ui.input(|i| !i.key_pressed(egui::Key::Escape))
                                    && !name.trim().is_empty()
                                {
                                    out.push(Action::RenameLayer(id, name.trim().to_string()));
                                }
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

fn history_section(ui: &mut egui::Ui, m: &Model, out: &mut Vec<Action>) {
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
    egui::Window::new(tr("Command"))
        .title_bar(false)
        .anchor(egui::Align2::CENTER_TOP, Vec2::new(0.0, 60.0))
        .fixed_size(Vec2::new(460.0, 0.0))
        .show(ctx, |ui| {
            let r = ui.add(
                egui::TextEdit::singleline(query)
                    .hint_text(tr("Search commands …"))
                    .desired_width(f32::INFINITY),
            );
            r.request_focus();
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height(360.0)
                .show(ui, |ui| {
                    for (i, c) in hits.iter().enumerate() {
                        let text = format!("{} › {}", c.menu_text(), c.label_text());
                        let resp = ui.add(
                            egui::Button::new(text)
                                .shortcut_text(c.shortcut())
                                .selected(i == *sel)
                                .frame(i == *sel),
                        );
                        if i == *sel {
                            resp.scroll_to_me(None);
                        }
                        if resp.clicked() {
                            out.push(Action::Command(**c));
                            *close = true;
                        }
                    }
                });
        });
}

/// Rows of the layer panel (top → bottom).
pub fn layer_rows<T: Clone>(doc: &imagine_core::Document<T>) -> Vec<LayerRow> {
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
                    .map(|v| matches!(v.kind, imagine_core::vector::VectorKind::Text(_))),
                fx: l.effects.iter().any(|e| e.enabled),
            })
        })
        .collect()
}

fn filter_dialog(ctx: &egui::Context, st: &mut State, out: &mut Vec<Action>) {
    use imagine_core::Filter;
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
