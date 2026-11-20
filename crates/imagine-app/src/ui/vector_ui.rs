//! Properties of text and shape layers. Returns the changed description and
//! whether the edit is finished (so typing or a drag is one undo step).

use egui::RichText;
use imagine_core::VectorLayer;
use imagine_core::vector::{Geometry, StrokeStyle, TextAlign, VectorKind};

#[derive(Default)]
pub struct EditorState {
    /// Give the text field the keyboard on the next pass (new text layer).
    pub focus_text: bool,
    font_filter: String,
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
            row(ui, "Schrift", |ui| {
                let mut chosen = None;
                let r = egui::ComboBox::from_id_salt("font").selected_text(&s.font).width(170.0).height(320.0).show_ui(ui, |ui| {
                    ui.add(egui::TextEdit::singleline(&mut st.font_filter).hint_text("suchen …"));
                    let f = st.font_filter.to_lowercase();
                    for name in ["sans-serif", "serif", "monospace"].iter().map(|s| s.to_string()).chain(fonts.iter().cloned()) {
                        if !f.is_empty() && !name.to_lowercase().contains(&f) {
                            continue;
                        }
                        if ui.selectable_label(s.font == name, &name).clicked() {
                            chosen = Some(name);
                        }
                    }
                });
                let mut resp = r.response;
                if let Some(c) = chosen {
                    s.font = c;
                    resp.mark_changed();
                }
                resp
            }, &mut t);
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
