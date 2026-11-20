//! Properties of text and shape layers. Returns the changed description and
//! whether the edit is finished (so typing or a drag is one undo step).

use egui::RichText;
use imagine_core::vector::{Geometry, StrokeStyle, TextAlign, VectorKind};
use imagine_core::{VectorLayer, tr};

#[derive(Default)]
pub struct EditorState {
    /// Give the text field the keyboard on the next pass (new text layer).
    pub focus_text: bool,
    /// The font window (its live preview is read by the app).
    pub fonts: super::font_ui::FontWindow,
    /// OpenType features of the current font (font, weight, italic → tags).
    feature_cache: Option<((String, u32, bool), std::collections::BTreeSet<String>)>,
}

/// What the properties editor changed.
#[derive(Default)]
pub struct VectorEdit {
    /// New description and whether the edit is finished.
    pub change: Option<(VectorLayer, bool)>,
    pub font_used: Option<String>,
    pub toggle_favourite: Option<String>,
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

fn row(
    ui: &mut egui::Ui,
    label: &str,
    add: impl FnOnce(&mut egui::Ui) -> egui::Response,
    t: &mut Track,
) {
    ui.label(label);
    ui.horizontal(|ui| {
        let r = add(ui);
        t.add(&r);
    });
    ui.end_row();
}

/// Linear RGBA colour button (alpha included).
fn color_button(ui: &mut egui::Ui, c: &mut [f32; 4]) -> egui::Response {
    let mut rgba = egui::Rgba::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
    let r = egui::color_picker::color_edit_button_rgba(
        ui,
        &mut rgba,
        egui::color_picker::Alpha::OnlyBlend,
    );
    if r.changed() {
        let [cr, cg, cb, ca] = rgba.to_rgba_unmultiplied();
        *c = [cr, cg, cb, ca];
    }
    r
}

#[allow(clippy::too_many_arguments)]
pub fn editor(
    ui: &mut egui::Ui,
    v: &VectorLayer,
    paths: &[super::panels::PathChoice],
    fonts: &[imagine_vector::FontGroup],
    favourites: &[String],
    recent: &[String],
    brush: [f32; 3],
    st: &mut EditorState,
) -> VectorEdit {
    let mut edit = VectorEdit::default();
    let ev = super::font_ui::show(ui.ctx(), &mut st.fonts, v, fonts, favourites, recent);
    edit.font_used = ev.used;
    edit.toggle_favourite = ev.toggle_favourite;
    let mut out = v.clone();
    let mut t = Track::default();
    match &mut out.kind {
        VectorKind::Text(s) => {
            ui.label(RichText::new("Text").strong());
            let r = ui.add(
                egui::TextEdit::multiline(&mut s.text)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY),
            );
            if std::mem::take(&mut st.focus_text) {
                r.request_focus();
            }
            t.add(&r);
            egui::Grid::new("text props")
                .num_columns(2)
                .spacing([10.0, 6.0])
                .show(ui, |ui| {
                    ui.label(tr("Font"));
                    ui.horizontal(|ui| {
                        // The family name (the face is chosen below).
                        let shown = fonts
                            .iter()
                            .find(|g| g.faces.iter().any(|f| f.family == s.font))
                            .map_or(s.font.clone(), |g| g.name.clone());
                        let r = ui
                            .add_sized(
                                [ui.available_width().min(220.0), 22.0],
                                egui::Button::new(shown).truncate(),
                            )
                            .on_hover_text(tr("Choose font"));
                        if r.clicked() {
                            if st.fonts.open {
                                st.fonts.open = false;
                                st.fonts.preview = None;
                            } else {
                                st.fonts.open_for(v, fonts, "");
                            }
                        }
                    });
                    ui.end_row();
                    if let Some(f) = ev.chosen {
                        s.font = f.family;
                        s.weight = f.weight;
                        s.italic = f.italic;
                        t.changed = true;
                        t.done = true;
                    }
                    row(
                        ui,
                        tr("Size"),
                        |ui| {
                            ui.add(
                                egui::Slider::new(&mut s.size, 4.0..=2000.0)
                                    .logarithmic(true)
                                    .suffix(" px")
                                    .max_decimals(0),
                            )
                        },
                        &mut t,
                    );
                    // Faces of the family (all weights and italics), else bold/italic toggles.
                    let group = fonts
                        .iter()
                        .find(|g| g.faces.iter().any(|f| f.family == s.font))
                        .filter(|g| g.faces.len() > 1);
                    ui.label(tr("Style"));
                    ui.horizontal(|ui| match group {
                        Some(g) => {
                            let cur = g
                                .face(&s.font, s.weight, s.italic)
                                .map(|f| f.label.clone())
                                .unwrap_or_default();
                            egui::ComboBox::from_id_salt("face")
                                .selected_text(cur.clone())
                                .width(150.0)
                                .height(400.0)
                                .show_ui(ui, |ui| {
                                    for f in &g.faces {
                                        if ui.selectable_label(f.label == cur, &f.label).clicked() {
                                            s.font = f.family.clone();
                                            s.weight = f.weight;
                                            s.italic = f.italic;
                                            t.changed = true;
                                            t.done = true;
                                        }
                                    }
                                });
                        }
                        None => {
                            let mut bold = s.weight >= 600.0;
                            let r = ui
                                .toggle_value(&mut bold, RichText::new(tr("B")).strong())
                                .on_hover_text(tr(
                                    "bold (synthetic if the font has no bold style)",
                                ));
                            if r.changed() {
                                s.weight = if bold { 700.0 } else { 400.0 };
                                t.changed = true;
                                t.done = true;
                            }
                            let r =
                                ui.toggle_value(&mut s.italic, RichText::new(tr("I")).italics());
                            t.add(&r);
                        }
                    });
                    ui.end_row();
                    ui.label(tr("Align"));
                    ui.horizontal(|ui| {
                        for a in TextAlign::ALL {
                            let label = match a {
                                TextAlign::Left => tr("Left"),
                                TextAlign::Center => tr("Middle"),
                                TextAlign::Right => tr("Right"),
                                TextAlign::Justify => tr("Justified"),
                            };
                            if ui
                                .selectable_label(s.align == a, label)
                                .on_hover_text(a.name())
                                .clicked()
                                && s.align != a
                            {
                                s.align = a;
                                t.changed = true;
                                t.done = true;
                            }
                        }
                    });
                    ui.end_row();
                    row(
                        ui,
                        tr("Color"),
                        |ui| {
                            let mut r = color_button(ui, &mut s.color);
                            if ui.small_button(tr("Brush Color")).clicked() {
                                s.color = [brush[0], brush[1], brush[2], s.color[3]];
                                r.mark_changed();
                            }
                            r
                        },
                        &mut t,
                    );
                    row(
                        ui,
                        tr("Line height"),
                        |ui| {
                            ui.add(egui::Slider::new(&mut s.line_height, 0.6..=3.0).max_decimals(2))
                        },
                        &mut t,
                    );
                    row(
                        ui,
                        tr("Tracking"),
                        |ui| {
                            ui.add(
                                egui::Slider::new(&mut s.letter_spacing, -20.0..=100.0)
                                    .suffix(" px")
                                    .max_decimals(1),
                            )
                        },
                        &mut t,
                    );
                    // Text on a path: follow one of the document's path layers.
                    ui.label(tr("Path"));
                    ui.horizontal(|ui| {
                        let current = if s.path.is_some() { tr("On path").to_string() } else { tr("None").to_string() };
                        egui::ComboBox::from_id_salt("text path")
                            .selected_text(current)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(s.path.is_none(), tr("None")).clicked() && s.path.is_some() {
                                    s.path = None;
                                    t.changed = true;
                                    t.done = true;
                                }
                                if paths.is_empty() {
                                    ui.label(RichText::new(tr("Draw a path with the pen (P) first.")).weak());
                                }
                                for p in paths {
                                    if ui.selectable_label(false, &p.name).clicked() {
                                        // Into this layer's own coordinates.
                                        let inv = glam::DAffine2::from_cols_array(&v.transform).inverse();
                                        let t2 = |q: [f64; 2]| {
                                            let r = inv.transform_point2(glam::DVec2::new(q[0], q[1]));
                                            [r.x, r.y]
                                        };
                                        let nodes = p.nodes.iter().map(|n| imagine_core::vector::PathNode { p: t2(n.p), h_in: t2(n.h_in), h_out: t2(n.h_out) }).collect();
                                        let keep = s.path.as_ref().map_or((0.0, false), |o| (o.offset, o.flip));
                                        s.path = Some(imagine_core::vector::TextPath { nodes, closed: p.closed, offset: keep.0, flip: keep.1 });
                                        t.changed = true;
                                        t.done = true;
                                    }
                                }
                            });
                    });
                    ui.end_row();
                    if let Some(p) = &mut s.path {
                        ui.label(tr("Path Start"));
                        ui.horizontal(|ui| {
                            let mut pct = p.offset * 100.0;
                            let r = ui
                                .add(egui::Slider::new(&mut pct, 0.0..=100.0).suffix(" %").max_decimals(0))
                                .on_hover_text(tr("Where the text starts on the path"));
                            if r.changed() {
                                p.offset = pct / 100.0;
                            }
                            t.add(&r);
                            if ui
                                .checkbox(&mut p.flip, tr("Flip"))
                                .on_hover_text(tr("Run along the other side of the path"))
                                .changed()
                            {
                                t.changed = true;
                                t.done = true;
                            }
                        });
                        ui.end_row();
                    }
                    ui.label(tr("Paragraph"));
                    ui.horizontal(|ui| {
                        let mut wrap = s.width.is_some();
                        if ui
                            .checkbox(&mut wrap, "")
                            .on_hover_text(tr("Wrap lines at a width"))
                            .changed()
                        {
                            s.width = wrap.then_some(s.size * 12.0);
                            t.changed = true;
                            t.done = true;
                        }
                        if let Some(w) = &mut s.width {
                            let r = ui.add(
                                egui::Slider::new(w, 20.0..=10000.0)
                                    .logarithmic(true)
                                    .suffix(" px")
                                    .max_decimals(0),
                            );
                            t.add(&r);
                        }
                    });
                    ui.end_row();
                });
            // OpenType features (unsupported ones greyed out).
            let key = (s.font.clone(), s.weight as u32, s.italic);
            if st.feature_cache.as_ref().is_none_or(|(k, _)| *k != key) {
                st.feature_cache = Some((key, imagine_vector::font_feature_tags(s)));
            }
            let tags = &st.feature_cache.as_ref().unwrap().1;
            let changed_count = s.features.len();
            egui::CollapsingHeader::new(if changed_count > 0 { format!("OpenType ({changed_count})") } else { "OpenType".to_string() })
                .id_salt("opentype")
                .show(ui, |ui| {
                    egui::Grid::new("ot features").num_columns(2).spacing([16.0, 2.0]).show(ui, |ui| {
                        for (i, (tag, label, _)) in imagine_core::vector::FEATURES.iter().enumerate() {
                            let mut on = s.feature(tag);
                            let has = tags.contains(*tag);
                            let r = ui
                                .add_enabled(has, egui::Checkbox::new(&mut on, tr(label)))
                                .on_hover_text(if has { format!("OpenType “{tag}”") } else { tr("This font does not have it").to_string() })
                                .on_disabled_hover_text(tr("This font does not have it"));
                            if r.changed() {
                                s.set_feature(tag, on);
                                t.changed = true;
                                t.done = true;
                            }
                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
                });
        }
        VectorKind::Shape(s) => {
            ui.label(RichText::new(tr("Shape")).strong());
            egui::Grid::new("shape props")
                .num_columns(2)
                .spacing([10.0, 6.0])
                .show(ui, |ui| {
                    if let Geometry::Rect { radius, w, h, .. } = &mut s.geometry {
                        let max = (w.abs().min(h.abs()) * 0.5).max(1.0);
                        row(
                            ui,
                            tr("Corners"),
                            |ui| {
                                ui.add(
                                    egui::Slider::new(radius, 0.0..=max)
                                        .suffix(" px")
                                        .max_decimals(0),
                                )
                            },
                            &mut t,
                        );
                    }
                    ui.label(tr("Fill Color"));
                    ui.horizontal(|ui| {
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
                    ui.end_row();
                    ui.label(tr("Stroke"));
                    ui.horizontal(|ui| {
                        let mut on = s.stroke.is_some();
                        if ui.checkbox(&mut on, "").changed() {
                            s.stroke = on.then_some(StrokeStyle {
                                color: [brush[0], brush[1], brush[2], 1.0],
                                width: 4.0,
                            });
                            t.changed = true;
                            t.done = true;
                        }
                        if let Some(st) = &mut s.stroke {
                            let r = color_button(ui, &mut st.color);
                            t.add(&r);
                            let r = ui.add(
                                egui::DragValue::new(&mut st.width)
                                    .range(0.5..=500.0)
                                    .suffix(" px")
                                    .speed(0.2),
                            );
                            t.add(&r);
                        }
                    });
                    ui.end_row();
                });
        }
    }
    ui.label(
        RichText::new(tr(
            "Ctrl+T transforms losslessly, Layer › Rasterize turns it into pixels",
        ))
        .weak()
        .small(),
    );
    edit.change = (t.changed || t.done).then_some((out, t.done));
    edit
}
