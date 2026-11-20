//! Properties of text and shape layers. Returns the changed description and
//! whether the edit is finished (so typing or a drag is one undo step).

use egui::RichText;
use imagine_core::VectorLayer;
use imagine_core::vector::{Geometry, StrokeStyle, TextAlign, VectorKind};

#[derive(Default)]
pub struct EditorState {
    /// Give the text field the keyboard on the next pass (new text layer).
    pub focus_text: bool,
    /// Font list open, its search text and the highlighted row.
    font_open: bool,
    font_query: String,
    font_sel: usize,
    focus_query: bool,
    /// Fonts picked recently (newest first).
    recent: Vec<String>,
    /// Rendered previews per family (`None`: the font has no glyphs for it).
    previews: std::collections::HashMap<String, Option<egui::TextureHandle>>,
}

impl EditorState {
    /// Opens the font list (for snapshots).
    pub fn open_font_list(&mut self, query: &str) {
        self.font_open = true;
        self.font_query = query.into();
        self.focus_query = true;
    }
}

const PREVIEW_TEXT: &str = "Imagine 123";
const ROW_H: f32 = 26.0;

/// Font picker: the current font (shown in itself); click opens a search
/// field and a list where every family is previewed in its own face.
/// Returns the chosen family.
fn font_picker(ui: &mut egui::Ui, current: &str, fonts: &[String], st: &mut EditorState) -> Option<String> {
    let mut chosen = None;
    // Closed: a button with the current font's preview.
    let r = ui.add_sized([ui.available_width().min(220.0), 22.0], egui::Button::new(current).truncate()).on_hover_text("Schrift wählen");
    if r.clicked() {
        st.font_open = !st.font_open;
        st.font_query.clear();
        st.font_sel = 0;
    }
    if !st.font_open {
        return None;
    }
    let q = ui.add(egui::TextEdit::singleline(&mut st.font_query).hint_text("Schrift suchen …").desired_width(f32::INFINITY));
    if r.clicked() || std::mem::take(&mut st.focus_query) {
        q.request_focus();
    }
    if q.changed() {
        st.font_sel = 0;
    }
    // Entries: recent and generic first (without a query), then all matches.
    let query = st.font_query.to_lowercase();
    let mut list: Vec<(String, bool)> = Vec::new();
    if query.is_empty() {
        list.extend(st.recent.iter().map(|f| (f.clone(), true)));
        list.extend(["sans-serif", "serif", "monospace"].iter().map(|f| (f.to_string(), false)));
    }
    list.extend(fonts.iter().filter(|f| query.is_empty() || f.to_lowercase().contains(&query)).map(|f| (f.clone(), false)));
    // Keyboard: arrows move, Enter picks, Escape closes.
    let (up, down, enter, esc) = ui.input(|i| {
        (i.key_pressed(egui::Key::ArrowUp), i.key_pressed(egui::Key::ArrowDown), i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Escape))
    });
    if !list.is_empty() {
        if down {
            st.font_sel = (st.font_sel + 1).min(list.len() - 1);
        }
        if up {
            st.font_sel = st.font_sel.saturating_sub(1);
        }
        if enter {
            chosen = Some(list[st.font_sel.min(list.len() - 1)].0.clone());
        }
    }
    if esc {
        st.font_open = false;
        return None;
    }
    let mut rendered = 0;
    let ctx = ui.ctx().clone();
    egui::ScrollArea::vertical().id_salt("font list").max_height(ROW_H * 10.0).auto_shrink([false, true]).show_rows(ui, ROW_H, list.len(), |ui, rows| {
        for i in rows {
            let (name, recent) = &list[i];
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_H), egui::Sense::click());
            let selected = i == st.font_sel || name == current;
            if i == st.font_sel {
                ui.painter().rect_filled(rect, 3.0, egui::Color32::from_rgb(60, 90, 140));
                if up || down {
                    ui.scroll_to_rect(rect, None);
                }
            } else if resp.hovered() {
                ui.painter().rect_filled(rect, 3.0, egui::Color32::from_white_alpha(12));
            }
            let text_col = if selected { egui::Color32::WHITE } else { egui::Color32::from_gray(200) };
            ui.painter().text(
                rect.left_center() + egui::vec2(6.0, 0.0),
                egui::Align2::LEFT_CENTER,
                if *recent { format!("{name}  ·") } else { name.clone() },
                egui::FontId::proportional(12.5),
                text_col,
            );
            // Preview in the font itself (rendered once, a few per frame).
            if !st.previews.contains_key(name) && rendered < 12 {
                rendered += 1;
                let tex = imagine_vector::font_preview(name, PREVIEW_TEXT, 17.0, 150).map(|(w, h, px)| {
                    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
                    ctx.load_texture(format!("font:{name}"), img, egui::TextureOptions::LINEAR)
                });
                st.previews.insert(name.clone(), tex);
            }
            match st.previews.get(name) {
                Some(Some(tex)) => {
                    let size = tex.size_vec2();
                    let scale = ((ROW_H - 4.0) / size.y).min(1.0);
                    let s = size * scale;
                    let r = egui::Rect::from_min_size(egui::pos2(rect.max.x - s.x - 6.0, rect.center().y - s.y / 2.0), s);
                    ui.painter().image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), text_col);
                }
                Some(None) => {}
                None => ctx.request_repaint(),
            }
            if resp.clicked() {
                chosen = Some(name.clone());
            }
        }
    });
    if list.is_empty() {
        ui.label(RichText::new("keine passende Schrift").weak());
    }
    if let Some(f) = &chosen {
        st.recent.retain(|r| r != f);
        st.recent.insert(0, f.clone());
        st.recent.truncate(6);
        st.font_open = false;
    }
    chosen
}

#[derive(Default)]
struct Track {
    changed: bool,
    done: bool,
}

impl Track {
    fn add(&mut self, r: &egui::Response) {
        if r.changed() {
            self.changed = true;
            if !r.dragged() && !r.has_focus() {
                self.done = true;
            }
        }
        if r.drag_stopped() || r.lost_focus() {
            self.done = true;
        }
    }
}

fn row(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> egui::Response, t: &mut Track) {
    ui.horizontal(|ui| {
        ui.add_sized([86.0, 18.0], egui::Label::new(label));
        let r = add(ui);
        t.add(&r);
    });
}

/// Linear RGBA colour button (alpha included).
fn color_button(ui: &mut egui::Ui, c: &mut [f32; 4]) -> egui::Response {
    let mut rgba = egui::Rgba::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
    let r = egui::color_picker::color_edit_button_rgba(ui, &mut rgba, egui::color_picker::Alpha::OnlyBlend);
    if r.changed() {
        let [cr, cg, cb, ca] = rgba.to_rgba_unmultiplied();
        *c = [cr, cg, cb, ca];
    }
    r
}

pub fn editor(ui: &mut egui::Ui, v: &VectorLayer, fonts: &[String], brush: [f32; 3], st: &mut EditorState) -> Option<(VectorLayer, bool)> {
    let mut out = v.clone();
    let mut t = Track::default();
    match &mut out.kind {
        VectorKind::Text(s) => {
            ui.label(RichText::new("Text").strong());
            let r = ui.add(egui::TextEdit::multiline(&mut s.text).desired_rows(3).desired_width(f32::INFINITY));
            if std::mem::take(&mut st.focus_text) {
                r.request_focus();
            }
            t.add(&r);
            ui.horizontal_top(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Schrift"));
                ui.vertical(|ui| {
                    if let Some(f) = font_picker(ui, &s.font.clone(), fonts, st) {
                        s.font = f;
                        t.changed = true;
                        t.done = true;
                    }
                });
            });
            row(ui, "Größe", |ui| ui.add(egui::Slider::new(&mut s.size, 4.0..=2000.0).logarithmic(true).suffix(" px").max_decimals(0)), &mut t);
            ui.horizontal(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Schnitt"));
                let mut bold = s.weight >= 600.0;
                let r = ui.toggle_value(&mut bold, RichText::new("F").strong());
                if r.changed() {
                    s.weight = if bold { 700.0 } else { 400.0 };
                    t.changed = true;
                    t.done = true;
                }
                let r = ui.toggle_value(&mut s.italic, RichText::new("K").italics());
                t.add(&r);
                ui.add_space(8.0);
                for a in TextAlign::ALL {
                    let label = match a {
                        TextAlign::Left => "links",
                        TextAlign::Center => "Mitte",
                        TextAlign::Right => "rechts",
                        TextAlign::Justify => "Block",
                    };
                    if ui.selectable_label(s.align == a, label).on_hover_text(a.name()).clicked() && s.align != a {
                        s.align = a;
                        t.changed = true;
                        t.done = true;
                    }
                }
            });
            row(ui, "Farbe", |ui| {
                let mut r = color_button(ui, &mut s.color);
                if ui.small_button("Pinselfarbe").clicked() {
                    s.color = [brush[0], brush[1], brush[2], s.color[3]];
                    r.mark_changed();
                }
                r
            }, &mut t);
            row(ui, "Zeilenabstand", |ui| ui.add(egui::Slider::new(&mut s.line_height, 0.6..=3.0).max_decimals(2)), &mut t);
            row(ui, "Laufweite", |ui| ui.add(egui::Slider::new(&mut s.letter_spacing, -20.0..=100.0).suffix(" px").max_decimals(1)), &mut t);
            ui.horizontal(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Absatz"));
                let mut wrap = s.width.is_some();
                if ui.checkbox(&mut wrap, "").on_hover_text("Zeilen bei einer Breite umbrechen").changed() {
                    s.width = wrap.then_some(s.size * 12.0);
                    t.changed = true;
                    t.done = true;
                }
                if let Some(w) = &mut s.width {
                    let r = ui.add(egui::Slider::new(w, 20.0..=10000.0).logarithmic(true).suffix(" px").max_decimals(0));
                    t.add(&r);
                }
            });
        }
        VectorKind::Shape(s) => {
            ui.label(RichText::new("Form").strong());
            if let Geometry::Rect { radius, w, h, .. } = &mut s.geometry {
                let max = (w.abs().min(h.abs()) * 0.5).max(1.0);
                row(ui, "Ecken", |ui| ui.add(egui::Slider::new(radius, 0.0..=max).suffix(" px").max_decimals(0)), &mut t);
            }
            ui.horizontal(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Fläche"));
                let mut on = s.fill.is_some();
                if ui.checkbox(&mut on, "").changed() {
                    s.fill = on.then_some([brush[0], brush[1], brush[2], 1.0]);
                    t.changed = true;
                    t.done = true;
                }
                if let Some(f) = &mut s.fill {
                    let r = color_button(ui, f);
                    t.add(&r);
                }
            });
            ui.horizontal(|ui| {
                ui.add_sized([86.0, 18.0], egui::Label::new("Kontur"));
                let mut on = s.stroke.is_some();
                if ui.checkbox(&mut on, "").changed() {
                    s.stroke = on.then_some(StrokeStyle { color: [brush[0], brush[1], brush[2], 1.0], width: 4.0 });
                    t.changed = true;
                    t.done = true;
                }
                if let Some(st) = &mut s.stroke {
                    let r = color_button(ui, &mut st.color);
                    t.add(&r);
                    let r = ui.add(egui::DragValue::new(&mut st.width).range(0.5..=500.0).suffix(" px").speed(0.2));
                    t.add(&r);
                }
            });
        }
    }
    ui.label(RichText::new("Strg+T transformiert verlustfrei, Ebene › Rastern macht Pixel daraus").weak().small());
    (t.changed || t.done).then_some((out, t.done))
}
