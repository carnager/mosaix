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
    ui.horizontal(|ui| {
        ui.add_sized([80.0, 18.0], egui::Label::new(tr(label)));
        let r = ui.add(
            egui::Slider::new(v, range)
                .suffix(suffix)
                .max_decimals(0)
                .clamping(egui::SliderClamping::Never),
        );
        t.add(&r);
    });
}

fn percent(ui: &mut egui::Ui, t: &mut Track, label: &str, v: &mut f32) {
    let mut p = *v * 100.0;
    slider(ui, t, label, &mut p, 0.0..=100.0, " %");
    *v = (p / 100.0).clamp(0.0, 1.0);
}

fn colour(ui: &mut egui::Ui, t: &mut Track, label: &str, c: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        ui.add_sized([80.0, 18.0], egui::Label::new(tr(label)));
        let mut rgb = egui::Rgba::from_rgb(c[0], c[1], c[2]);
        let r = egui::color_picker::color_edit_button_rgba(
            ui,
            &mut rgb,
            egui::color_picker::Alpha::Opaque,
        );
        if r.changed() {
            *c = [rgb.r(), rgb.g(), rgb.b()];
        }
        t.add(&r);
    });
}

fn angle(ui: &mut egui::Ui, t: &mut Track, v: &mut f32) {
    ui.horizontal(|ui| {
        ui.add_sized([80.0, 18.0], egui::Label::new(tr("Light Angle")));
        let r = ui.add(
            egui::DragValue::new(v)
                .range(-180.0..=180.0)
                .suffix("°")
                .speed(1.0),
        );
        t.add(&r);
        ui.label(RichText::new(tr("90° = from above")).weak().small());
    });
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
            ui.horizontal(|ui| {
                ui.add_sized([80.0, 18.0], egui::Label::new(tr("Position")));
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
            ui.horizontal(|ui| {
                ui.add_sized([80.0, 18.0], egui::Label::new(tr("Type")));
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

/// Shows the effects of the active layer; `Some((new list, done))` on change.
pub fn editor(
    ui: &mut egui::Ui,
    effects: &[Effect],
    open: &mut Option<usize>,
) -> Option<(Vec<Effect>, bool)> {
    let mut list = effects.to_vec();
    let mut t = Track::default();
    let mut remove = None;
    for (i, e) in list.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            if ui
                .checkbox(&mut e.enabled, "")
                .on_hover_text(tr("On/Off"))
                .changed()
            {
                t.click();
            }
            let expanded = *open == Some(i);
            if ui.selectable_label(expanded, e.kind.name()).clicked() {
                *open = if expanded { None } else { Some(i) };
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button(tr("Remove")).clicked() {
                    remove = Some(i);
                }
            });
        });
        if *open == Some(i) {
            ui.indent(("fx", i), |ui| settings(ui, &mut e.kind, &mut t));
        }
    }
    if let Some(i) = remove {
        list.remove(i);
        *open = None;
        t.click();
    }
    let mut add = None;
    egui::ComboBox::from_id_salt("add effect")
        .selected_text(tr("Add effect …"))
        .show_ui(ui, |ui| {
            for k in EffectKind::defaults() {
                if ui.selectable_label(false, k.name()).clicked() {
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
