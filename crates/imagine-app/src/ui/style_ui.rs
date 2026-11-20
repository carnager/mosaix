//! The layer-styles section: list of effects with on/off, settings, add and
//! remove. Returns the new list and whether the edit is finished.

use egui::RichText;
use imagine_core::style::{BevelStyle, Effect, EffectKind, StrokePosition};
use imagine_core::tr;

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

    fn click(&mut self) {
        self.changed = true;
        self.done = true;
    }
}

fn slider(
    ui: &mut egui::Ui,
    t: &mut Track,
    label: &str,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
) {
    ui.label(tr(label));
    let r = ui.add(
        egui::Slider::new(v, range)
            .suffix(suffix)
            .max_decimals(0)
            .clamping(egui::SliderClamping::Never),
    );
    t.add(&r);
    ui.end_row();
}

fn percent(ui: &mut egui::Ui, t: &mut Track, label: &str, v: &mut f32) {
    let mut p = *v * 100.0;
    slider(ui, t, label, &mut p, 0.0..=100.0, " %");
    *v = (p / 100.0).clamp(0.0, 1.0);
}

fn colour(ui: &mut egui::Ui, t: &mut Track, label: &str, c: &mut [f32; 3]) {
    ui.label(tr(label));
    let mut rgb = egui::Rgba::from_rgb(c[0], c[1], c[2]);
    let r =
        egui::color_picker::color_edit_button_rgba(ui, &mut rgb, egui::color_picker::Alpha::Opaque);
    if r.changed() {
        *c = [rgb.r(), rgb.g(), rgb.b()];
    }
    t.add(&r);
    ui.end_row();
}

fn angle(ui: &mut egui::Ui, t: &mut Track, v: &mut f32) {
    ui.label(tr("Light Angle"));
    ui.horizontal(|ui| {
        let r = ui.add(
            egui::DragValue::new(v)
                .range(-180.0..=180.0)
                .suffix("°")
                .speed(1.0),
        );
        t.add(&r);
        ui.label(RichText::new(tr("90° = from above")).weak().small());
    });
    ui.end_row();
}

fn settings(ui: &mut egui::Ui, k: &mut EffectKind, t: &mut Track) {
    match k {
        EffectKind::Stroke {
            size,
            position,
            color,
            opacity,
        } => {
            slider(ui, t, "Size", size, 1.0..=100.0, " px");
            ui.label(tr("Position"));
            ui.horizontal(|ui| {
                for (p, n) in [
                    (StrokePosition::Outside, tr("Outside")),
                    (StrokePosition::Center, tr("Centered")),
                    (StrokePosition::Inside, tr("Inside")),
                ] {
                    if ui.selectable_label(*position == p, n).clicked() && *position != p {
                        *position = p;
                        t.click();
                    }
                }
            });
            ui.end_row();
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
        }
        EffectKind::DropShadow {
            color,
            opacity,
            angle: a,
            distance,
            size,
            spread,
        } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
            angle(ui, t, a);
            slider(ui, t, "Distance", distance, 0.0..=200.0, " px");
            slider(ui, t, "Size", size, 0.0..=150.0, " px");
            slider(ui, t, "Spread", spread, 0.0..=50.0, " px");
        }
        EffectKind::InnerShadow {
            color,
            opacity,
            angle: a,
            distance,
            size,
        } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
            angle(ui, t, a);
            slider(ui, t, "Distance", distance, 0.0..=100.0, " px");
            slider(ui, t, "Size", size, 0.0..=100.0, " px");
        }
        EffectKind::OuterGlow {
            color,
            opacity,
            size,
            spread,
        } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
            slider(ui, t, "Size", size, 0.0..=150.0, " px");
            slider(ui, t, "Spread", spread, 0.0..=50.0, " px");
        }
        EffectKind::InnerGlow {
            color,
            opacity,
            size,
        } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
            slider(ui, t, "Size", size, 0.0..=100.0, " px");
        }
        EffectKind::ColorOverlay { color, opacity } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
        }
        EffectKind::Bevel {
            style,
            size,
            depth,
            soften,
            angle: a,
            altitude,
            highlight,
            highlight_opacity,
            shadow,
            shadow_opacity,
        } => {
            ui.label(tr("Type"));
            ui.horizontal(|ui| {
                for (s, n) in [
                    (BevelStyle::Inner, tr("Inner Bevel")),
                    (BevelStyle::Emboss, tr("Emboss")),
                    (BevelStyle::Pillow, tr("Pillow Emboss")),
                ] {
                    if ui.selectable_label(*style == s, n).clicked() && *style != s {
                        *style = s;
                        t.click();
                    }
                }
            });
            ui.end_row();
            slider(ui, t, "Size", size, 1.0..=100.0, " px");
            let mut d = *depth * 100.0;
            slider(ui, t, "Depth", &mut d, 1.0..=500.0, " %");
            *depth = d / 100.0;
            slider(ui, t, "Soften", soften, 0.0..=30.0, " px");
            angle(ui, t, a);
            slider(ui, t, "Altitude", altitude, 5.0..=85.0, "°");
            colour(ui, t, "Highlight", highlight);
            percent(ui, t, "Highlight Opacity", highlight_opacity);
            colour(ui, t, "Shadow", shadow);
            percent(ui, t, "Shadow Opacity", shadow_opacity);
        }
    }
}

/// Colour and a short description for the effect's row (so two strokes
/// can be told apart without opening them).
fn summary(k: &EffectKind) -> ([f32; 3], String) {
    let px = |v: f32| format!("{v:.0} px");
    match k {
        EffectKind::Stroke {
            size,
            position,
            color,
            ..
        } => {
            let pos = match position {
                StrokePosition::Outside => tr("Outside"),
                StrokePosition::Center => tr("Centered"),
                StrokePosition::Inside => tr("Inside"),
            };
            (*color, format!("{} · {}", px(*size), pos))
        }
        EffectKind::DropShadow {
            color,
            distance,
            size,
            ..
        } => (*color, format!("{} · {}", px(*distance), px(*size))),
        EffectKind::InnerShadow {
            color,
            distance,
            size,
            ..
        } => (*color, format!("{} · {}", px(*distance), px(*size))),
        EffectKind::OuterGlow { color, size, .. } | EffectKind::InnerGlow { color, size, .. } => {
            (*color, px(*size))
        }
        EffectKind::ColorOverlay { color, opacity } => {
            (*color, format!("{:.0} %", opacity * 100.0))
        }
        EffectKind::Bevel {
            highlight, size, ..
        } => (*highlight, px(*size)),
    }
}

/// Shows the effects of the active layer; `Some((new list, done))` on change.
pub fn editor(
    ui: &mut egui::Ui,
    effects: &[Effect],
    open: &mut Option<usize>,
) -> Option<(Vec<Effect>, bool)> {
    let mut list = effects.to_vec();
    let mut t = Track::default();
    let mut remove = None;
    if list.is_empty() {
        ui.label(RichText::new(tr("No effects yet – stroke, shadows, glow, bevel …")).weak());
    }
    for (i, e) in list.iter_mut().enumerate() {
        let expanded = *open == Some(i);
        let (colour, info) = summary(&e.kind);
        ui.horizontal(|ui| {
            if ui
                .checkbox(&mut e.enabled, "")
                .on_hover_text(tr("On/Off"))
                .changed()
            {
                t.click();
            }
            let (r, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
            ui.painter().rect_filled(
                r,
                2.0,
                egui::Rgba::from_rgb(colour[0], colour[1], colour[2]),
            );
            ui.painter().rect_stroke(
                r,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(120)),
                egui::StrokeKind::Outside,
            );
            let strong = if e.enabled {
                ui.visuals().strong_text_color()
            } else {
                ui.visuals().weak_text_color()
            };
            let mut job = egui::text::LayoutJob::default();
            job.append(
                e.kind.name(),
                0.0,
                egui::TextFormat {
                    color: strong,
                    ..Default::default()
                },
            );
            job.append(
                &format!("   {info}"),
                0.0,
                egui::TextFormat {
                    color: ui.visuals().weak_text_color(),
                    ..Default::default()
                },
            );
            let w = (ui.available_width() - 30.0).max(40.0);
            let (rect, r) = ui.allocate_exact_size(egui::vec2(w, 22.0), egui::Sense::click());
            if expanded || r.hovered() {
                let fill = if expanded {
                    egui::Color32::from_gray(58)
                } else {
                    egui::Color32::from_gray(44)
                };
                ui.painter().rect_filled(rect, 3.0, fill);
            }
            let galley = ui.painter().layout_job(job);
            ui.painter().galley(
                egui::pos2(rect.left() + 6.0, rect.center().y - galley.size().y / 2.0),
                galley,
                strong,
            );
            if r.on_hover_text(tr("Show settings")).clicked() {
                *open = if expanded { None } else { Some(i) };
            }
            if super::icons::button(ui, super::icons::Icon::Trash, 20.0, false)
                .on_hover_text(tr("Remove"))
                .clicked()
            {
                remove = Some(i);
            }
        });
        if expanded {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                egui::Grid::new(("fx", i))
                    .num_columns(2)
                    .spacing([10.0, 6.0])
                    .show(ui, |ui| settings(ui, &mut e.kind, &mut t));
            });
        }
    }
    if let Some(i) = remove {
        list.remove(i);
        *open = None;
        t.click();
    }
    ui.add_space(4.0);
    let mut add = None;
    ui.menu_button(format!("+ {}", tr("Add effect …")), |ui| {
        for k in EffectKind::defaults() {
            if ui.button(k.name()).clicked() {
                add = Some(k);
            }
        }
    });
    if let Some(kind) = add {
        list.push(Effect {
            enabled: true,
            kind,
        });
        *open = Some(list.len() - 1);
        t.click();
    }
    (t.changed || t.done).then_some((list, t.done))
}
