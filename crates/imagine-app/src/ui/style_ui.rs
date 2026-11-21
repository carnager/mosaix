//! The layer-styles section: list of effects with on/off, settings, add and
//! remove. Returns the new list and whether the edit is finished.

use egui::RichText;
use imagine_core::BlendMode;
use imagine_core::pattern::Pattern;
use imagine_core::style::{BevelStyle, Effect, EffectKind, GradientStyle, StrokePosition};
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

/// Blend mode and (shadows, bevel) the global light switch.
fn common(ui: &mut egui::Ui, e: &mut Effect, t: &mut Track) {
    if matches!(e.kind, EffectKind::Bevel { .. }) {
        // Bevels use Screen/Multiply for highlight and shadow.
    } else {
        ui.label(tr("Blend Mode"));
        let current = e.mode();
        egui::ComboBox::from_id_salt(("fx blend", e.kind.name()))
            .selected_text(current.name())
            .show_ui(ui, |ui| {
                for b in BlendMode::ALL {
                    if ui.selectable_label(b == current, b.name()).clicked() && b != current {
                        e.blend = (b != e.kind.default_blend()).then_some(b);
                        t.click();
                    }
                }
            });
        ui.end_row();
    }
    if e.kind.angle().is_some() {
        ui.label("");
        if ui
            .checkbox(&mut e.global_light, tr("Use Global Light"))
            .on_hover_text(tr("Share the light angle with all other effects that use the global light"))
            .changed()
        {
            t.click();
        }
        ui.end_row();
    }
}

/// A pattern as a UI texture (cached by name and size).
fn pattern_texture(ui: &egui::Ui, p: &Pattern) -> egui::TextureHandle {
    let id = egui::Id::unique(("pattern", &p.name, p.width, p.height, p.pixels.len(), p.pixels.get(..64).map(<[u8]>::to_vec)));
    if let Some(t) = ui.ctx().data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return t;
    }
    let img = egui::ColorImage::from_rgba_unmultiplied([p.width as usize, p.height as usize], &p.pixels);
    let t = ui.ctx().load_texture(format!("pattern {}", p.name), img, egui::TextureOptions::NEAREST_REPEAT);
    ui.ctx().data_mut(|d| d.insert_temp(id, t.clone()));
    t
}

/// A square swatch showing the pattern repeating.
fn pattern_swatch(ui: &mut egui::Ui, p: &Pattern, size: f32, selected: bool) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click());
    let tex = pattern_texture(ui, p);
    // Show about 48 pattern pixels across, repeated.
    let reps = egui::vec2(48.0 / p.width as f32, 48.0 / p.height as f32).max(egui::vec2(1.0, 1.0));
    ui.painter().image(tex.id(), rect, egui::Rect::from_min_size(egui::Pos2::ZERO, reps), egui::Color32::WHITE);
    let stroke = if selected {
        egui::Stroke::new(2.0, ui.visuals().selection.stroke.color)
    } else {
        egui::Stroke::new(1.0, egui::Color32::from_gray(90))
    };
    ui.painter().rect_stroke(rect, 2.0, stroke, egui::StrokeKind::Outside);
    r.on_hover_text(&p.name)
}

/// Swatches of the library's patterns (plus `pattern` if it is not one of
/// them); true if another was chosen.
pub(super) fn pattern_picker(ui: &mut egui::Ui, pattern: &mut Pattern, patterns: &[Pattern]) -> bool {
    let mut changed = false;
            // Rows of swatches (explicit rows: a wrapped layout inside the
            // grid reports its height a frame late).
            let mut all: Vec<Option<&Pattern>> = patterns.iter().map(Some).collect();
            let foreign = !patterns.iter().any(|p| p == pattern);
            if foreign {
                // A pattern from a file that is not in the library.
                all.push(None);
            }
            let per_row = ((ui.available_width() + 4.0) / 34.0).floor().max(1.0) as usize;
            let current = pattern.clone();
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                for row in all.chunks(per_row) {
                    ui.horizontal(|ui| {
                        for p in row {
                            match p {
                                Some(p) => {
                                    let sel = !foreign && *p == &current;
                                    if pattern_swatch(ui, p, 30.0, sel).clicked() && !sel {
                                        *pattern = (*p).clone();
                                        changed = true;
                                    }
                                }
                                None => {
                                    pattern_swatch(ui, &current, 30.0, true);
                                }
                            }
                        }
                    });
                }
            });
    changed
}

fn settings(ui: &mut egui::Ui, k: &mut EffectKind, t: &mut Track, patterns: &[Pattern]) {
    match k {
        EffectKind::PatternOverlay { pattern, opacity, scale, offset } => {
            ui.label(tr("Pattern"));
            if pattern_picker(ui, pattern, patterns) {
                t.click();
            }
            ui.end_row();
            percent(ui, t, "Opacity", opacity);
            let mut pct = *scale * 100.0;
            slider(ui, t, "Scale", &mut pct, 10.0..=400.0, " %");
            *scale = (pct / 100.0).max(0.01);
            ui.label(tr("Offset"));
            ui.horizontal(|ui| {
                for v in offset.iter_mut() {
                    let r = ui.add(egui::DragValue::new(v).speed(0.5).suffix(" px"));
                    t.add(&r);
                }
            });
            ui.end_row();
        }
        EffectKind::GradientOverlay {
            start,
            end,
            opacity,
            angle: a,
            style,
            scale,
            reverse,
            gradient,
        } => {
            ui.label(tr("Gradient"));
            ui.vertical(|ui| {
                let mut g = gradient.clone().unwrap_or_else(|| imagine_core::gradient::Gradient::two(*start, *end));
                let o = super::gradient_ui::editor(ui, &mut g, "overlay");
                if o.changed {
                    t.changed = true;
                    (*start, *end) = g.ends();
                    let simple = g.colors.len() <= 2 && g.alphas.is_empty() && g.smoothness == 0.0 && g.colors.iter().all(|c| c.mid == 0.5);
                    *gradient = (!simple).then_some(g);
                }
                t.done |= o.done;
            });
            ui.end_row();
            percent(ui, t, "Opacity", opacity);
            ui.label(tr("Style"));
            ui.horizontal(|ui| {
                if super::gradient_ui::style_combo(ui, style, "overlay") {
                    t.click();
                }
                if ui.checkbox(reverse, tr("Reverse")).changed() {
                    t.click();
                }
            });
            ui.end_row();
            ui.label(tr("Angle"));
            let r = ui.add(egui::DragValue::new(a).range(-180.0..=180.0).suffix("°").speed(1.0));
            t.add(&r);
            ui.end_row();
            let mut pct = *scale * 100.0;
            slider(ui, t, "Scale", &mut pct, 10.0..=150.0, " %");
            *scale = pct / 100.0;
        }
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
        EffectKind::Satin { color, opacity, angle: a, distance, size, invert } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
            ui.label(tr("Angle"));
            let r = ui.add(egui::DragValue::new(a).range(-180.0..=180.0).suffix("°").speed(1.0));
            t.add(&r);
            ui.end_row();
            slider(ui, t, "Distance", distance, 1.0..=250.0, " px");
            slider(ui, t, "Size", size, 0.0..=250.0, " px");
            ui.label("");
            if ui.checkbox(invert, tr("Invert")).changed() {
                t.click();
            }
            ui.end_row();
        }
        EffectKind::OuterGlow {
            color,
            opacity,
            size,
            spread,
            range,
        } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
            slider(ui, t, "Size", size, 0.0..=150.0, " px");
            slider(ui, t, "Spread", spread, 0.0..=50.0, " px");
            percent(ui, t, "Range", range);
        }
        EffectKind::InnerGlow {
            color,
            opacity,
            size,
            range,
        } => {
            colour(ui, t, "Color", color);
            percent(ui, t, "Opacity", opacity);
            slider(ui, t, "Size", size, 0.0..=100.0, " px");
            percent(ui, t, "Range", range);
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
        EffectKind::Satin { color, distance, size, .. } => (*color, format!("{} · {}", px(*distance), px(*size))),
        EffectKind::ColorOverlay { color, opacity } => {
            (*color, format!("{:.0} %", opacity * 100.0))
        }
        EffectKind::GradientOverlay { start, style, angle, .. } => (
            *start,
            match style {
                GradientStyle::Radial => style.label().to_string(),
                s => format!("{} · {angle:.0}°", s.label()),
            },
        ),
        EffectKind::Bevel {
            highlight, size, ..
        } => (*highlight, px(*size)),
        EffectKind::PatternOverlay { pattern, scale, .. } => {
            // Swatch: the pattern's average colour.
            let n = (pattern.pixels.len() / 4).max(1) as f32;
            let avg: [f32; 3] = std::array::from_fn(|k| {
                let mean = pattern.pixels.iter().skip(k).step_by(4).map(|&v| v as f32).sum::<f32>() / n / 255.0;
                imagine_core::color::srgb_to_linear(mean)
            });
            (avg, format!("{} · {:.0} %", pattern.name, scale * 100.0))
        }
    }
}

/// Shows the effects of the active layer; `Some((new list, done))` on change.
pub fn editor(
    ui: &mut egui::Ui,
    effects: &[Effect],
    open: &mut Option<usize>,
    patterns: &[Pattern],
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
                    .show(ui, |ui| {
                        common(ui, e, &mut t);
                        settings(ui, &mut e.kind, &mut t, patterns)
                    });
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
        // New shadows and bevels take the global light angle of the others.
        let mut e = Effect::new(kind);
        if let Some(g) = list.iter().find(|x| x.global_light).and_then(|x| x.kind.angle())
            && let Some((a, _)) = e.kind.light_mut()
        {
            *a = g;
        }
        list.push(e);
        *open = Some(list.len() - 1);
        t.click();
    }
    // An effect using the global light that changed its angle takes the
    // others along (the app does the same across layers).
    if t.changed {
        let before: Vec<Option<f32>> = effects.iter().map(|e| e.global_light.then(|| e.kind.angle()).flatten()).collect();
        let changed = list.iter().enumerate().find_map(|(i, e)| {
            let now = e.global_light.then(|| e.kind.angle()).flatten()?;
            (before.get(i).copied().flatten() != Some(now)).then_some(now)
        });
        if let Some(g) = changed {
            for e in list.iter_mut().filter(|e| e.global_light) {
                if let Some((a, _)) = e.kind.light_mut() {
                    *a = g;
                }
            }
        }
    }
    (t.changed || t.done).then_some((list, t.done))
}
