//! Panel layout. `build` reads a [`Model`] (snapshot of the app) and returns
//! [`Action`]s; it never touches the engine directly.

use egui::{Align, Color32, Layout, RichText, Sense, Stroke, Vec2};
use imagine_core::{BlendMode, LayerId, TIP_NAMES};

use super::icons::{self, Icon};
use super::palette::{self, Cmd, COMMANDS, MENUS};

/// Tools as shown in the toolbar: (id, name, shortcut).
pub const TOOLS: [(ToolId, &str, &str); 11] = [
    (ToolId::Move, "Verschieben", "V"),
    (ToolId::Marquee, "Auswahlrechteck", "M"),
    (ToolId::Ellipse, "Auswahlellipse", "M"),
    (ToolId::Lasso, "Lasso", "L"),
    (ToolId::Wand, "Zauberstab", "W"),
    (ToolId::Crop, "Zuschneiden", "C"),
    (ToolId::Brush, "Pinsel", "B"),
    (ToolId::Smudge, "Wischfinger", "U"),
    (ToolId::Eraser, "Radierer", "E"),
    (ToolId::Gradient, "Verlauf linear", "G"),
    (ToolId::Radial, "Verlauf radial", "G"),
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
    pub wand_tolerance: f32,
    pub wand_contiguous: bool,
    pub boxed: Option<&'static str>,
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
    WandTolerance(f32),
    WandContiguous(bool),
    SelectLayer(LayerId),
    ToggleVisible(LayerId),
    ToggleMaskEdit(LayerId),
    RenameLayer(LayerId, String),
    MoveLayer { id: LayerId, target: LayerId, above: bool, into: bool },
    /// `done` = slider released (ends the merged undo step).
    LayerOpacity { value: f32, done: bool },
    LayerBlend(BlendMode),
    DeleteLayer,
    HistoryJump(usize),
    CommitBox,
    CancelBox,
    CanvasSize { w: u32, h: u32, anchor: (f64, f64) },
    ImageSize { w: u32, h: u32 },
    OpenFile(std::path::PathBuf),
    SaveAs(std::path::PathBuf),
    ExportAs(std::path::PathBuf),
    NewDocument { w: u32, h: u32, white: bool },
    Unsaved(super::files::Answer),
    AddAdjustment(imagine_core::Adjustment),
    ApplyFilter(imagine_core::Filter),
    SetAdjustment { adj: imagine_core::Adjustment, done: bool },
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
    /// Open filter dialog: the filter and whether to preview it.
    filter: Option<(imagine_core::Filter, bool)>,
    /// What the canvas should preview right now (set by the dialog).
    pub filter_preview: Option<imagine_core::Filter>,
}

enum Dialog {
    Canvas { w: u32, h: u32, anchor: (u8, u8) },
    Image { w: u32, h: u32, keep: bool },
}

const ACCENT: Color32 = Color32::from_rgb(90, 150, 255);

impl State {
    pub fn open_filter(&mut self, f: imagine_core::Filter) {
        self.filter = Some((f, true));
    }

    /// A palette, dialog or question is open: keys go to the interface.
    pub fn captures_keyboard(&self) -> bool {
        self.palette.is_some()
            || self.confirm.is_some()
            || self.file.is_some()
            || self.new_doc.is_some()
            || self.dialog.is_some()
            || self.filter.is_some()
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
        egui::Panel::top("menu").exact_size(26.0).show(ui, |ui| menu_bar(ui, m, st, &mut out));
        egui::Panel::bottom("status").exact_size(24.0).show(ui, |ui| status_bar(ui, m, st));
        egui::Panel::left("tools").exact_size(46.0).resizable(false).show(ui, |ui| toolbar(ui, m, &mut out));
        egui::Panel::right("side").default_size(280.0).min_size(220.0).show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| side_panel(ui, m, st, &mut out));
        });
        st.canvas = Some(ui.available_rect_before_wrap());
    });
    dialogs(ctx, m, st, &mut out);
    if st.show_log {
        let mut open = true;
        egui::Window::new("Meldungen").open(&mut open).default_size([520.0, 300.0]).show(ctx, |ui| {
            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for msg in super::log::recent(200) {
                    ui.label(RichText::new(&msg.text).color(level_color(msg.level)).monospace());
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
            ui.menu_button(menu, |ui| {
                for c in COMMANDS.iter().filter(|c| c.menu == menu) {
                    if ui.add(egui::Button::new(c.label).shortcut_text(c.shortcut())).clicked() {
                        out.push(Action::Command(*c));
                        ui.close();
                    }
                }
                if menu == "Datei" && !m.recent.is_empty() {
                    ui.separator();
                    ui.menu_button("Zuletzt geöffnet", |ui| {
                        for p in &m.recent {
                            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            if ui.button(name).on_hover_text(p.display().to_string()).clicked() {
                                out.push(Action::OpenFile(p.clone()));
                                ui.close();
                            }
                        }
                    });
                }
                if menu == "Bild" {
                    ui.separator();
                    ui.menu_button("Filter", |ui| {
                        for f in imagine_core::Filter::defaults() {
                            if ui.button(format!("{} …", f.name())).clicked() {
                                st.filter = Some((f, true));
                                ui.close();
                            }
                        }
                    });
                }
                if menu == "Ebene" {
                    ui.separator();
                    ui.menu_button("Neue Einstellungsebene", |ui| {
                        for a in imagine_core::Adjustment::defaults() {
                            if ui.button(a.name()).clicked() {
                                out.push(Action::AddAdjustment(a));
                                ui.close();
                            }
                        }
                    });
                }
                if menu == "Bild" {
                    ui.separator();
                    if ui.button("Arbeitsfläche …").clicked() {
                        st.dialog = Some(Dialog::Canvas { w: m.doc_size.0, h: m.doc_size.1, anchor: (1, 1) });
                        ui.close();
                    }
                    if ui.button("Bildgröße …").clicked() {
                        st.dialog = Some(Dialog::Image { w: m.doc_size.0, h: m.doc_size.1, keep: true });
                        ui.close();
                    }
                }
                if menu == "Ansicht" {
                    ui.separator();
                    if ui.button("Meldungen").clicked() {
                        st.show_log = true;
                        ui.close();
                    }
                }
            });
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(format!("{}{}", if m.modified { "• " } else { "" }, m.name)).strong());
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
            ui.label("Auswahl aktiv");
        }
        if m.busy {
            ui.separator();
            ui.spinner();
            ui.label("rechnet …");
        }
        ui.separator();
        // Latest message, fading out after a while; click for the log.
        if let Some(msg) = super::log::recent(1).pop() {
            let age = msg.at.elapsed().as_secs_f32();
            let alpha = if msg.level <= tracing::Level::WARN { 1.0 } else { (1.0 - (age - 6.0) / 2.0).clamp(0.0, 1.0) };
            if alpha > 0.0 {
                let r = ui.add(
                    egui::Label::new(RichText::new(&msg.text).color(level_color(msg.level).gamma_multiply(alpha)))
                        .truncate()
                        .sense(Sense::click()),
                );
                if r.on_hover_text("Alle Meldungen anzeigen").clicked() {
                    st.show_log = true;
                }
                if alpha < 1.0 || age < 8.0 {
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
                }
            }
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new(&m.status).weak());
        });
    });
}

fn toolbar(ui: &mut egui::Ui, m: &Model, out: &mut Vec<Action>) {
    ui.add_space(4.0);
    ui.vertical_centered(|ui| {
        for (id, name, key) in TOOLS {
            if icons::button(ui, Icon::Tool(id), 34.0, m.tool == id).on_hover_text(format!("{name} ({key})")).clicked() {
                out.push(Action::Tool(id));
            }
        }
        ui.add_space(10.0);
        // Current colour; click for the picker in the side panel.
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(30.0), Sense::click());
        ui.painter().rect_filled(rect, 4.0, linear_color(m.color));
        ui.painter().rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_gray(160)), egui::StrokeKind::Outside);
        resp.on_hover_text("Pinselfarbe (X: Schwarz/Weiß)");
    });
}

fn linear_color(c: [f32; 3]) -> Color32 {
    egui::Rgba::from_rgb(c[0], c[1], c[2]).into()
}

fn side_panel(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    if let Some(kind) = m.boxed {
        ui.group(|ui| {
            ui.label(RichText::new(kind).strong());
            ui.label(RichText::new("Enter übernimmt, Esc bricht ab").weak());
            ui.horizontal(|ui| {
                if ui.button("Übernehmen").clicked() {
                    out.push(Action::CommitBox);
                }
                if ui.button("Abbrechen").clicked() {
                    out.push(Action::CancelBox);
                }
            });
        });
    }
    egui::CollapsingHeader::new("Farbe").default_open(true).show(ui, |ui| color_section(ui, m, st, out));
    egui::CollapsingHeader::new("Werkzeug").default_open(true).show(ui, |ui| tool_section(ui, m, out));
    if let Some(adj) = &m.active_adjustment {
        egui::CollapsingHeader::new("Eigenschaften").default_open(true).show(ui, |ui| {
            if let Some((adj, done)) = super::adjust_ui::editor(ui, adj, &mut st.adj_editor, m.histogram.as_deref()) {
                out.push(Action::SetAdjustment { adj, done });
            }
        });
    }
    egui::CollapsingHeader::new("Ebenen").default_open(true).show(ui, |ui| layer_section(ui, m, st, out));
    egui::CollapsingHeader::new("Verlauf").default_open(false).show(ui, |ui| history_section(ui, m, out));
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
    let (sv_rect, sv) = ui.allocate_exact_size(Vec2::new(width, width * 0.62), Sense::click_and_drag());
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
    if let Some(pos) = sv.interact_pointer_pos().filter(|_| sv.is_pointer_button_down_on() || sv.clicked()) {
        hsv[1] = ((pos.x - sv_rect.min.x) / sv_rect.width()).clamp(0.0, 1.0);
        hsv[2] = 1.0 - ((pos.y - sv_rect.min.y) / sv_rect.height()).clamp(0.0, 1.0);
        changed = true;
    }
    if let Some(pos) = hue.interact_pointer_pos().filter(|_| hue.is_pointer_button_down_on() || hue.clicked()) {
        hsv[0] = ((pos.x - hue_rect.min.x) / hue_rect.width()).clamp(0.0, 0.9999);
        changed = true;
    }
    let mark = sv_rect.min + Vec2::new(hsv[1] * sv_rect.width(), (1.0 - hsv[2]) * sv_rect.height());
    p.circle_stroke(mark, 5.0, Stroke::new(2.0, if hsv[2] > 0.5 { Color32::BLACK } else { Color32::WHITE }));
    let hx = hue_rect.min.x + hsv[0] * hue_rect.width();
    p.rect_stroke(
        egui::Rect::from_min_max(egui::pos2(hx - 2.0, hue_rect.min.y - 1.0), egui::pos2(hx + 2.0, hue_rect.max.y + 1.0)),
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
    let mut hex = st.hex.clone().unwrap_or_else(|| format!("#{r:02x}{g:02x}{b:02x}"));
    ui.horizontal(|ui| {
        let (sw, _) = ui.allocate_exact_size(Vec2::new(28.0, 18.0), Sense::hover());
        ui.painter().rect_filled(sw, 3.0, current);
        let resp = ui.add(egui::TextEdit::singleline(&mut hex).desired_width(80.0).font(egui::TextStyle::Monospace));
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

fn slider(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>, pct: bool) -> bool {
    ui.horizontal(|ui| {
        ui.add_sized([86.0, 18.0], egui::Label::new(label).truncate());
        let s = egui::Slider::new(v, range).clamping(egui::SliderClamping::Always);
        let s = if pct { s.custom_formatter(|v, _| format!("{:.0} %", v * 100.0)).custom_parser(|t| t.trim_end_matches('%').trim().parse::<f64>().ok().map(|v| v / 100.0)) } else { s };
        ui.add(s).changed()
    })
    .inner
}

fn tool_section(ui: &mut egui::Ui, m: &Model, out: &mut Vec<Action>) {
    match m.tool {
        ToolId::Brush | ToolId::Eraser | ToolId::Smudge => {
            let mut b = m.brush;
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Größe"));
                changed |= ui
                    .add(egui::Slider::new(&mut b.size, 1.0..=2000.0).logarithmic(true).suffix(" px").max_decimals(0))
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Spitze"));
                egui::ComboBox::from_id_salt("tip").selected_text(TIP_NAMES[b.tip as usize]).show_ui(ui, |ui| {
                    for (i, n) in TIP_NAMES.iter().enumerate() {
                        changed |= ui.selectable_value(&mut b.tip, i as u32, *n).changed();
                    }
                });
            });
            if b.tip == 0 {
                changed |= slider(ui, "Härte", &mut b.hardness, 0.0..=1.0, true);
            }
            if m.tool == ToolId::Smudge {
                let mut s = m.smudge_strength;
                if slider(ui, "Stärke", &mut s, 0.0..=1.0, true) {
                    out.push(Action::SmudgeStrength(s));
                }
            } else {
                changed |= slider(ui, "Deckkraft", &mut b.opacity, 0.0..=1.0, true);
                changed |= slider(ui, "Fluss", &mut b.flow, 0.01..=1.0, true);
            }
            changed |= slider(ui, "Abstand", &mut b.spacing, 0.02..=2.0, true);
            changed |= slider(ui, "Glättung", &mut b.smoothing, 0.0..=0.95, true);
            egui::CollapsingHeader::new("Form & Dynamik").default_open(false).show(ui, |ui| {
                changed |= slider(ui, "Winkel", &mut b.angle, -180.0..=180.0, false);
                changed |= slider(ui, "Rundheit", &mut b.roundness, 0.05..=1.0, true);
                changed |= ui.checkbox(&mut b.follow_direction, "Richtung folgen").changed();
                changed |= slider(ui, "Größe ±", &mut b.size_jitter, 0.0..=1.0, true);
                changed |= slider(ui, "Winkel ±", &mut b.angle_jitter, 0.0..=180.0, false);
                changed |= slider(ui, "Streuung", &mut b.scatter, 0.0..=5.0, false);
                changed |= slider(ui, "Fluss ±", &mut b.flow_jitter, 0.0..=1.0, true);
            });
            if changed {
                out.push(Action::Brush(b));
            }
        }
        ToolId::Wand => {
            let mut t = m.wand_tolerance;
            ui.horizontal(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Toleranz"));
                if ui.add(egui::Slider::new(&mut t, 0.0..=255.0).max_decimals(0)).changed() {
                    out.push(Action::WandTolerance(t));
                }
            });
            let mut c = m.wand_contiguous;
            if ui.checkbox(&mut c, "Benachbart").on_hover_text("nur zusammenhängende Pixel (W)").changed() {
                out.push(Action::WandContiguous(c));
            }
            ui.label(RichText::new("Nimmt alle Ebenen auf. Umschalt = hinzufügen, Alt = abziehen.").weak());
        }
        ToolId::Marquee | ToolId::Ellipse | ToolId::Lasso => {
            ui.label(RichText::new("Umschalt = hinzufügen, Alt = abziehen, beides = Schnittmenge").weak());
        }
        ToolId::Move => {
            ui.label(RichText::new("Ziehen oder Pfeiltasten (Umschalt: 10 px). Mit Auswahl nur die ausgewählten Pixel.").weak());
            if ui.button("Frei transformieren (Strg+T)").clicked() {
                out.push(Action::Command(*COMMANDS.iter().find(|c| c.label == "Frei transformieren").unwrap()));
            }
        }
        ToolId::Crop => {
            ui.label(RichText::new("Rahmen ziehen, Anfasser anpassen. Über den Rand hinaus vergrößert die Arbeitsfläche.").weak());
        }
        ToolId::Gradient | ToolId::Radial => {
            ui.label(RichText::new("Ziehen: von der Pinselfarbe zu transparent").weak());
            let mut o = m.brush;
            if slider(ui, "Deckkraft", &mut o.opacity, 0.0..=1.0, true) {
                out.push(Action::Brush(o));
            }
        }
    }
}

fn layer_section(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("blend").width(130.0).selected_text(m.active_blend.name()).show_ui(ui, |ui| {
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
                .custom_parser(|t| t.trim_end_matches('%').trim().parse::<f64>().ok().map(|v| v / 100.0)),
        );
        if r.changed() {
            out.push(Action::LayerOpacity { value: o, done: false });
        }
        if r.drag_stopped() || r.lost_focus() {
            out.push(Action::LayerOpacity { value: o, done: true });
        }
        r.on_hover_text("Deckkraft (ziehen oder tippen)");
    });
    ui.add_space(2.0);

    let row_h = 34.0;
    for row in &m.layers {
        let active = row.id == m.active;
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), row_h), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        if active {
            painter.rect_filled(rect, 3.0, ACCENT.linear_multiply(0.35));
        } else if resp.hovered() {
            painter.rect_filled(rect, 3.0, Color32::from_white_alpha(10));
        }
        // Eye.
        let eye = egui::Rect::from_center_size(egui::pos2(rect.min.x + 12.0, rect.center().y), Vec2::splat(18.0));
        let eye_resp = ui.interact(eye, resp.id.with("eye"), Sense::click());
        icons::paint(
            &painter,
            eye.shrink(2.0),
            if row.visible { Icon::Eye } else { Icon::EyeClosed },
            if row.visible { Color32::from_gray(220) } else { Color32::from_gray(110) },
        );
        if eye_resp.on_hover_text("ein-/ausblenden").clicked() {
            out.push(Action::ToggleVisible(row.id));
        }
        let mut x = eye.max.x + 6.0 + row.depth as f32 * 14.0;
        // Adjustment icon, or thumbnail on a checkerboard.
        if row.adjustment {
            let t = egui::Rect::from_min_size(egui::pos2(x, rect.min.y + 3.0), Vec2::splat(row_h - 6.0));
            icons::paint(&painter, t.shrink(4.0), Icon::Adjust, Color32::from_gray(210));
            x = t.max.x + 8.0;
        } else if !row.group {
            let t = egui::Rect::from_min_size(egui::pos2(x, rect.min.y + 3.0), Vec2::splat(row_h - 6.0));
            let n = 4;
            let c = t.width() / n as f32;
            for j in 0..n {
                for i in 0..n {
                    let g = if (i + j) % 2 == 0 { 190 } else { 140 };
                    let r = egui::Rect::from_min_size(t.min + Vec2::new(i as f32 * c, j as f32 * c), Vec2::splat(c));
                    painter.rect_filled(r, 0.0, Color32::from_gray(g));
                }
            }
            if let Some((tex, (u, v))) = row.thumb {
                let size = Vec2::new(t.width() * u, t.height() * v);
                let img = egui::Rect::from_center_size(t.center(), size);
                painter.image(tex, img, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(u, v)), Color32::WHITE);
            }
            painter.rect_stroke(t, 0.0, Stroke::new(1.0, Color32::from_gray(70)), egui::StrokeKind::Outside);
            x = t.max.x + 8.0;
        }
        // Mask badge.
        let name_end = rect.max.x - if row.has_mask { 28.0 } else { 4.0 };
        if row.has_mask {
            let badge = egui::Rect::from_center_size(egui::pos2(rect.max.x - 15.0, rect.center().y), Vec2::new(22.0, 16.0));
            let b_resp = ui.interact(badge, resp.id.with("mask"), Sense::click());
            let editing = active && m.mask_edit;
            painter.rect(
                badge,
                3.0,
                if row.mask_enabled { Color32::from_gray(200) } else { Color32::from_gray(80) },
                Stroke::new(if editing { 2.0 } else { 0.0 }, ACCENT),
                egui::StrokeKind::Outside,
            );
            painter.circle_filled(badge.center(), 4.0, Color32::from_gray(30));
            if b_resp.on_hover_text("Maske: klicken wechselt zwischen Maske bemalen und Pixel").clicked() {
                out.push(Action::ToggleMaskEdit(row.id));
            }
        }
        // Name (double-click: rename).
        if st.rename.as_ref().is_some_and(|(id, _)| *id == row.id) {
            let (_, text) = st.rename.as_mut().unwrap();
            let edit_rect = egui::Rect::from_min_max(egui::pos2(x, rect.min.y + 1.0), egui::pos2(name_end, rect.max.y - 1.0));
            let r = ui.put(edit_rect, egui::TextEdit::singleline(text));
            r.request_focus();
            if r.lost_focus() {
                let (id, name) = st.rename.take().unwrap();
                if ui.input(|i| !i.key_pressed(egui::Key::Escape)) && !name.trim().is_empty() {
                    out.push(Action::RenameLayer(id, name.trim().to_string()));
                }
            }
        } else {
            if row.group {
                let r = egui::Rect::from_center_size(egui::pos2(x + 7.0, rect.center().y), Vec2::splat(14.0));
                icons::paint(&painter, r, Icon::Folder, Color32::from_gray(190));
                x += 18.0;
            }
            let galley = painter.layout_no_wrap(
                row.name.clone(),
                egui::FontId::proportional(13.5),
                if row.visible { Color32::from_gray(230) } else { Color32::from_gray(130) },
            );
            painter.with_clip_rect(egui::Rect::from_min_max(rect.min, egui::pos2(name_end, rect.max.y))).galley(
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
                painter.rect_stroke(rect.shrink(1.0), 3.0, stroke, egui::StrokeKind::Inside);
            } else {
                let y = if t < 0.5 { rect.min.y } else { rect.max.y };
                painter.hline(rect.x_range(), y, stroke);
            }
            if let Some(payload) = resp.dnd_release_payload::<LayerId>() {
                out.push(Action::MoveLayer { id: *payload, target: row.id, above: t < 0.5, into });
            }
        }
    }
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let cmd = |label: &str| *COMMANDS.iter().find(|c| c.label == label).unwrap();
        for (icon, label) in [
            (Icon::Plus, "Neue Ebene"),
            (Icon::Folder, "Gruppieren"),
            (Icon::Mask, "Ebenenmaske hinzufügen"),
            (Icon::Duplicate, "Ebene duplizieren"),
        ] {
            let c = cmd(label);
            if icons::button(ui, icon, 24.0, false).on_hover_text(format!("{label} ({})", c.shortcut())).clicked() {
                out.push(Action::Command(c));
            }
        }
        let r = icons::button(ui, Icon::Adjust, 24.0, false).on_hover_text("Neue Einstellungsebene");
        egui::Popup::menu(&r).show(|ui| {
            for a in imagine_core::Adjustment::defaults() {
                if ui.button(a.name()).clicked() {
                    out.push(Action::AddAdjustment(a));
                }
            }
        });
        if icons::button(ui, Icon::Trash, 24.0, false).on_hover_text("Ebene löschen").clicked() {
            out.push(Action::DeleteLayer);
        }
    });
}

fn history_section(ui: &mut egui::Ui, m: &Model, out: &mut Vec<Action>) {
    let mut entry = |ui: &mut egui::Ui, i: usize, label: &str| {
        let done = i <= m.history_done;
        let text = RichText::new(label).color(if done { Color32::from_gray(220) } else { Color32::from_gray(110) });
        if ui.selectable_label(i == m.history_done, text).clicked() {
            out.push(Action::HistoryJump(i));
        }
    };
    egui::ScrollArea::vertical().max_height(220.0).stick_to_bottom(true).show(ui, |ui| {
        entry(ui, 0, "Ausgangszustand");
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
            egui::Window::new("Arbeitsfläche").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
                size_fields(ui, w, h);
                ui.label("Anker");
                egui::Grid::new("anchor").show(ui, |ui| {
                    for y in 0..3u8 {
                        for x in 0..3u8 {
                            if ui.add_sized([22.0, 22.0], egui::Button::new(if *anchor == (x, y) { "X" } else { "" }).selected(*anchor == (x, y))).clicked() {
                                *anchor = (x, y);
                            }
                        }
                        ui.end_row();
                    }
                });
                ui.label(RichText::new("Nichts wird gelöscht: verkleinern verbirgt nur.").weak());
                if ui.button("OK").clicked() {
                    out.push(Action::CanvasSize { w: *w, h: *h, anchor: (anchor.0 as f64 / 2.0, anchor.1 as f64 / 2.0) });
                    done = true;
                }
            });
        }
        Dialog::Image { w, h, keep } => {
            egui::Window::new("Bildgröße").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
                let (ow, oh) = m.doc_size;
                let (pw, ph) = (*w, *h);
                size_fields(ui, w, h);
                ui.checkbox(keep, "Seitenverhältnis beibehalten");
                if *keep {
                    if *w != pw {
                        *h = ((*w as f64 * oh as f64 / ow as f64).round() as u32).max(1);
                    } else if *h != ph {
                        *w = ((*h as f64 * ow as f64 / oh as f64).round() as u32).max(1);
                    }
                }
                ui.label(RichText::new(format!("{:.0} % der jetzigen Größe", *w as f64 / ow as f64 * 100.0)).weak());
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
        ui.label("Breite");
        ui.add(egui::DragValue::new(w).range(1..=300_000).suffix(" px"));
        ui.end_row();
        ui.label("Höhe");
        ui.add(egui::DragValue::new(h).range(1..=300_000).suffix(" px"));
        ui.end_row();
    });
}

fn command_palette(ctx: &egui::Context, query: &mut String, sel: &mut usize, out: &mut Vec<Action>, close: &mut bool) {
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
    egui::Window::new("Befehl")
        .title_bar(false)
        .anchor(egui::Align2::CENTER_TOP, Vec2::new(0.0, 60.0))
        .fixed_size(Vec2::new(460.0, 0.0))
        .show(ctx, |ui| {
            let r = ui.add(egui::TextEdit::singleline(query).hint_text("Befehl suchen …").desired_width(f32::INFINITY));
            r.request_focus();
            ui.separator();
            egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                for (i, c) in hits.iter().enumerate() {
                    let text = format!("{} › {}", c.menu, c.label);
                    let resp = ui.add(egui::Button::new(text).shortcut_text(c.shortcut()).selected(i == *sel).frame(i == *sel));
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
            })
        })
        .collect()
}

fn filter_dialog(ctx: &egui::Context, st: &mut State, out: &mut Vec<Action>) {
    use imagine_core::Filter;
    let Some((f, preview)) = &mut st.filter else {
        st.filter_preview = None;
        return;
    };
    let mut open = true;
    let mut apply = false;
    let mut cancel = false;
    egui::Window::new(f.name())
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
            match f {
                Filter::GaussianBlur { radius } => s(ui, "Radius", radius, 0.1..=1000.0, true, " px"),
                Filter::UnsharpMask { amount, radius, threshold } => {
                    s(ui, "Stärke", amount, 1.0..=500.0, false, " %");
                    s(ui, "Radius", radius, 0.1..=250.0, true, " px");
                    s(ui, "Schwellenwert", threshold, 0.0..=255.0, false, "");
                }
                Filter::HighPass { radius } => s(ui, "Radius", radius, 0.1..=1000.0, true, " px"),
                Filter::AddNoise { amount, monochrome, gaussian, seed } => {
                    s(ui, "Stärke", amount, 0.0..=100.0, false, " %");
                    ui.checkbox(monochrome, "Monochrom");
                    ui.checkbox(gaussian, "Gaußverteilt");
                    let mut sd = *seed as f32;
                    s(ui, "Variante", &mut sd, 0.0..=100.0, false, "");
                    *seed = sd as u32;
                }
            }
            ui.checkbox(preview, "Vorschau");
            ui.horizontal(|ui| {
                if ui.button("Anwenden").clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    apply = true;
                }
                if ui.button("Abbrechen").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    cancel = true;
                }
            });
        });
    if apply {
        out.push(Action::ApplyFilter(f.clone()));
    }
    if apply || cancel || !open {
        st.filter = None;
        st.filter_preview = None;
    } else {
        st.filter_preview = preview.then(|| f.clone());
    }
}
