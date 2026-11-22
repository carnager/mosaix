//! Property editors for adjustment layers, including an interactive curves
//! graph. Each editor returns the changed adjustment and whether the edit
//! is finished (slider released), so a drag becomes one undo step.

use egui::{Color32, Pos2, RichText, Sense, Stroke, Vec2, pos2};
use mosaix_core::adjust::{Levels, eval_curve};
use mosaix_core::{Adjustment, tr};
use mosaix_gpu::Histogram;

#[derive(Default)]
pub struct EditorState {
    /// 0 = RGB, 1–3 = red, green, blue.
    channel: usize,
    dragging: Option<usize>,
    /// Hue/Saturation: 0 = Master, 1–6 = Reds … Magentas.
    hue_range: usize,
    /// Color balance: 0 = shadows, 1 = midtones, 2 = highlights.
    tone: Option<usize>,
    /// Channel mixer: output channel 0–2.
    output: usize,
    /// Selective color: range 0–8.
    sel_range: usize,
    /// Color lookup: the user asked to load a .cube file.
    pub pick_lut: bool,
}

const CHANNELS: [&str; 4] = ["RGB", "Red", "Green", "Blue"];
const CHANNEL_COLORS: [Color32; 4] = [
    Color32::from_gray(230),
    Color32::from_rgb(240, 90, 80),
    Color32::from_rgb(90, 200, 100),
    Color32::from_rgb(90, 140, 250),
];

/// Tracks whether any widget changed and whether the interaction ended.
#[derive(Default)]
struct Track {
    changed: bool,
    done: bool,
}

impl Track {
    fn add(&mut self, r: &egui::Response) {
        if r.changed() {
            self.changed = true;
            if !r.dragged() {
                self.done = true;
            }
        }
        if r.drag_stopped() || r.lost_focus() {
            self.done = true;
        }
    }
}

fn row<'a>(ui: &mut egui::Ui, label: &str, w: impl egui::Widget + 'a, t: &mut Track) {
    ui.horizontal(|ui| {
        ui.add_sized([96.0, 18.0], egui::Label::new(tr(label)).truncate());
        let r = ui.add(w);
        t.add(&r);
    });
}

fn slider(v: &mut f32, range: std::ops::RangeInclusive<f32>) -> egui::Slider<'_> {
    egui::Slider::new(v, range)
        .clamping(egui::SliderClamping::Always)
        .max_decimals(1)
}

fn channel_pick(ui: &mut egui::Ui, st: &mut EditorState) {
    ui.horizontal(|ui| {
        for (i, n) in CHANNELS.iter().enumerate() {
            if ui
                .selectable_label(
                    st.channel == i,
                    RichText::new(tr(n)).color(CHANNEL_COLORS[i]),
                )
                .clicked()
            {
                st.channel = i;
                st.dragging = None;
            }
        }
    });
}

/// Shows the editor; `Some((new, done))` when something changed or finished.
pub fn editor(
    ui: &mut egui::Ui,
    adj: &Adjustment,
    st: &mut EditorState,
    hist: Option<&Histogram>,
) -> Option<(Adjustment, bool)> {
    let mut a = adj.clone();
    let mut t = Track::default();
    ui.label(RichText::new(adj.name()).strong());
    match &mut a {
        Adjustment::BrightnessContrast {
            brightness,
            contrast,
            legacy,
        } => {
            row(ui, "Brightness", slider(brightness, -150.0..=150.0), &mut t);
            row(ui, "Contrast", slider(contrast, -50.0..=100.0), &mut t);
            row(
                ui,
                "Use Legacy",
                egui::Checkbox::without_text(legacy),
                &mut t,
            );
        }
        Adjustment::Levels {
            master,
            red,
            green,
            blue,
        } => {
            channel_pick(ui, st);
            let l: &mut Levels = match st.channel {
                1 => red,
                2 => green,
                3 => blue,
                _ => master,
            };
            levels_graph(ui, l, hist, st, &mut t);
            let mut v = [
                l.in_black * 255.0,
                l.in_white * 255.0,
                l.out_black * 255.0,
                l.out_white * 255.0,
            ];
            row(
                ui,
                "Shadows",
                slider(&mut v[0], 0.0..=253.0).max_decimals(0),
                &mut t,
            );
            row(
                ui,
                "Gamma",
                slider(&mut l.gamma, 0.1..=9.99)
                    .logarithmic(true)
                    .max_decimals(2),
                &mut t,
            );
            row(
                ui,
                "Highlights",
                slider(&mut v[1], 2.0..=255.0).max_decimals(0),
                &mut t,
            );
            row(
                ui,
                "Output Shadows",
                slider(&mut v[2], 0.0..=255.0).max_decimals(0),
                &mut t,
            );
            row(
                ui,
                "Output Highlights",
                slider(&mut v[3], 0.0..=255.0).max_decimals(0),
                &mut t,
            );
            l.in_black = v[0] / 255.0;
            l.in_white = (v[1] / 255.0).max(l.in_black + 2.0 / 255.0);
            l.out_black = v[2] / 255.0;
            l.out_white = v[3] / 255.0;
        }
        Adjustment::Curves {
            master,
            red,
            green,
            blue,
        } => {
            channel_pick(ui, st);
            let ch = st.channel;
            let curve = match ch {
                1 => red,
                2 => green,
                3 => blue,
                _ => master,
            };
            curve_graph(ui, curve, hist, st, &mut t);
            ui.label(
                RichText::new(tr(
                    "Click adds a point, drag moves it, drag it out of the graph to remove it.",
                ))
                .weak()
                .small(),
            );
        }
        Adjustment::HueSaturation {
            hue,
            saturation,
            lightness,
            colorize,
            ranges,
        } => {
            const RANGES: [&str; 7] = ["Master", "Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"];
            if *colorize {
                st.hue_range = 0;
            }
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!*colorize, |ui| {
                    egui::ComboBox::from_id_salt("hue range")
                        .selected_text(tr(RANGES[st.hue_range]))
                        .show_ui(ui, |ui| {
                            for (i, n) in RANGES.iter().enumerate() {
                                let edited = i > 0 && ranges.get(i - 1).is_some_and(|r| !r.is_identity());
                                let label = if edited { format!("{} •", tr(n)) } else { tr(n).to_string() };
                                ui.selectable_value(&mut st.hue_range, i, label);
                            }
                        })
                        .response
                        .on_hover_text(tr("Edit all colors, or only one color range"));
                });
                let r = ui.checkbox(colorize, tr("Colorize"));
                t.add(&r);
            });
            if st.hue_range == 0 {
                row(ui, "Hue", slider(hue, -180.0..=180.0).suffix("°"), &mut t);
                row(ui, "Saturation", slider(saturation, -100.0..=100.0), &mut t);
                row(ui, "Lightness", slider(lightness, -100.0..=100.0), &mut t);
            } else {
                if ranges.len() < 6 {
                    *ranges = mosaix_core::adjust::HueRange::defaults();
                }
                let r = &mut ranges[st.hue_range - 1];
                row(ui, "Hue", slider(&mut r.hue, -180.0..=180.0).suffix("°"), &mut t);
                row(ui, "Saturation", slider(&mut r.saturation, -100.0..=100.0), &mut t);
                row(ui, "Lightness", slider(&mut r.lightness, -100.0..=100.0), &mut t);
                // Where the range lies on the hue wheel (fade-in, full, fade-out).
                hue_range_bar(ui, r.range);
                if ranges.iter().all(|r| r.is_identity()) {
                    ranges.clear();
                }
            }
        }
        Adjustment::Vibrance {
            vibrance,
            saturation,
        } => {
            row(ui, "Vibrance", slider(vibrance, -100.0..=100.0), &mut t);
            row(ui, "Saturation", slider(saturation, -100.0..=100.0), &mut t);
        }
        Adjustment::BlackWhite { weights } => {
            for (w, n) in weights
                .iter_mut()
                .zip(["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"])
            {
                row(
                    ui,
                    n,
                    slider(w, -200.0..=300.0).suffix(" %").max_decimals(0),
                    &mut t,
                );
            }
        }
        Adjustment::Exposure {
            exposure,
            offset,
            gamma,
        } => {
            row(
                ui,
                "Exposure",
                slider(exposure, -10.0..=10.0).max_decimals(2),
                &mut t,
            );
            row(
                ui,
                "Offset",
                slider(offset, -0.5..=0.5).max_decimals(3),
                &mut t,
            );
            row(
                ui,
                "Gamma",
                slider(gamma, 0.1..=9.99).logarithmic(true).max_decimals(2),
                &mut t,
            );
        }
        Adjustment::Invert => {
            ui.label(RichText::new(tr("No settings")).weak());
        }
        Adjustment::Threshold { level } => row(
            ui,
            "Threshold",
            slider(level, 1.0..=255.0).max_decimals(0),
            &mut t,
        ),
        Adjustment::Posterize { levels } => row(
            ui,
            "Steps",
            slider(levels, 2.0..=255.0).max_decimals(0),
            &mut t,
        ),
        Adjustment::ColorBalance {
            shadows,
            midtones,
            highlights,
            preserve_luminosity,
        } => {
            let tone = st.tone.get_or_insert(1);
            ui.horizontal(|ui| {
                for (i, n) in ["Shadows", "Midtones", "Highlights"].iter().enumerate() {
                    ui.selectable_value(tone, i, tr(n));
                }
            });
            let v = match *tone {
                0 => shadows,
                2 => highlights,
                _ => midtones,
            };
            for (k, (l, r)) in [("Cyan", "Red"), ("Magenta", "Green"), ("Yellow", "Blue")].into_iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.add_sized([56.0, 18.0], egui::Label::new(tr(l)));
                    let resp = ui.add(slider(&mut v[k], -100.0..=100.0).max_decimals(0));
                    t.add(&resp);
                    ui.label(tr(r));
                });
            }
            row(ui, "Preserve Luminosity", egui::Checkbox::without_text(preserve_luminosity), &mut t);
        }
        Adjustment::ChannelMixer {
            red,
            green,
            blue,
            gray,
            monochrome,
        } => {
            row(ui, "Monochrome", egui::Checkbox::without_text(monochrome), &mut t);
            let r = if *monochrome {
                ui.label(RichText::new(tr("Output: Gray")).weak());
                gray
            } else {
                ui.horizontal(|ui| {
                    for (i, n) in ["Red", "Green", "Blue"].iter().enumerate() {
                        if ui.selectable_label(st.output == i, RichText::new(tr(n)).color(CHANNEL_COLORS[i + 1])).clicked() {
                            st.output = i;
                        }
                    }
                });
                match st.output {
                    1 => green,
                    2 => blue,
                    _ => red,
                }
            };
            for (k, n) in ["Red", "Green", "Blue"].iter().enumerate() {
                row(ui, n, slider(&mut r[k], -200.0..=200.0).suffix(" %").max_decimals(0), &mut t);
            }
            let total = r[0] + r[1] + r[2];
            ui.label(
                RichText::new(tr!("Total {pct} %", pct = format!("{total:+.0}")))
                    .color(if total > 100.0 { Color32::from_rgb(240, 170, 60) } else { ui.visuals().weak_text_color() }),
            );
            row(ui, "Constant", slider(&mut r[3], -200.0..=200.0).suffix(" %").max_decimals(0), &mut t);
        }
        Adjustment::GradientMap { gradient, reverse } => {
            let o = super::gradient_ui::editor(ui, gradient, "gradient map");
            if o.changed {
                t.changed = true;
            }
            if o.done {
                t.done = true;
            }
            row(ui, "Reverse", egui::Checkbox::without_text(reverse), &mut t);
        }
        Adjustment::PhotoFilter { color, density, preserve_luminosity } => {
            // Photoshop's filters (sRGB).
            const PRESETS: [(&str, [u8; 3]); 6] = [
                ("Warming (85)", [236, 138, 0]),
                ("Warming (81)", [235, 177, 19]),
                ("Cooling (80)", [0, 109, 255]),
                ("Cooling (82)", [0, 181, 255]),
                ("Sepia", [172, 122, 51]),
                ("Underwater", [0, 194, 177]),
            ];
            ui.horizontal_wrapped(|ui| {
                for (n, c) in PRESETS {
                    let lin = c.map(|v| mosaix_core::color::srgb_to_linear(v as f32 / 255.0));
                    let fill = Color32::from_rgb(c[0], c[1], c[2]);
                    if ui.add(egui::Button::new("").fill(fill).min_size(Vec2::new(18.0, 18.0))).on_hover_text(tr(n)).clicked() {
                        *color = lin;
                        t.changed = true;
                        t.done = true;
                    }
                }
                let mut rgb = egui::Rgba::from_rgb(color[0], color[1], color[2]);
                let r = egui::color_picker::color_edit_button_rgba(ui, &mut rgb, egui::color_picker::Alpha::Opaque);
                if r.changed() {
                    *color = [rgb.r(), rgb.g(), rgb.b()];
                }
                t.add(&r);
            });
            row(ui, "Density", slider(density, 0.0..=100.0).max_decimals(0).suffix(" %"), &mut t);
            row(ui, "Preserve Luminosity", egui::Checkbox::without_text(preserve_luminosity), &mut t);
        }
        Adjustment::SelectiveColor { ranges, relative } => {
            egui::ComboBox::from_id_salt("selective range")
                .selected_text(tr(mosaix_core::adjust::SELECTIVE_RANGES[st.sel_range]))
                .show_ui(ui, |ui| {
                    for (i, n) in mosaix_core::adjust::SELECTIVE_RANGES.iter().enumerate() {
                        ui.selectable_value(&mut st.sel_range, i, tr(n));
                    }
                });
            let r = &mut ranges[st.sel_range.min(8)];
            for (k, n) in ["Cyan", "Magenta", "Yellow", "Black"].iter().enumerate() {
                row(ui, n, slider(&mut r[k], -100.0..=100.0).max_decimals(0).suffix(" %"), &mut t);
            }
            ui.horizontal(|ui| {
                let a = ui.selectable_value(relative, true, tr("Relative"));
                let b = ui.selectable_value(relative, false, tr("Absolute"));
                t.add(&a);
                t.add(&b);
            });
        }
        Adjustment::ColorLookup { name, size, amount, .. } => {
            ui.horizontal(|ui| {
                if *size == 0 {
                    ui.label(RichText::new(tr("No table loaded")).weak());
                } else {
                    ui.label(format!("{name} ({size}³)"));
                }
                if ui.button(tr("Load .cube …")).clicked() {
                    st.pick_lut = true;
                }
            });
            row(ui, "Amount", slider(amount, 0.0..=100.0).max_decimals(0).suffix(" %"), &mut t);
        }
    }
    if ui.small_button(tr("Reset")).clicked()
        && let Some(d) = Adjustment::defaults()
            .into_iter()
            .find(|d| d.key() == adj.key())
    {
        return Some((d, true));
    }
    (t.changed || t.done).then_some((a, t.done))
}

/// Interactive curve: points in 0…1, drawn in a square graph.
fn curve_graph(
    ui: &mut egui::Ui,
    curve: &mut Vec<[f32; 2]>,
    hist: Option<&Histogram>,
    st: &mut EditorState,
    t: &mut Track,
) {
    let color = CHANNEL_COLORS[st.channel];
    let side = ui.available_width().min(260.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(side), Sense::click_and_drag());
    let to_screen = |p: [f32; 2]| {
        pos2(
            rect.min.x + p[0] * rect.width(),
            rect.max.y - p[1] * rect.height(),
        )
    };
    let to_curve = |s: Pos2| {
        [
            ((s.x - rect.min.x) / rect.width()),
            ((rect.max.y - s.y) / rect.height()),
        ]
    };
    let painter = ui.painter_at(rect.expand(6.0));
    painter.rect_filled(rect, 2.0, Color32::from_gray(22));
    if let Some(h) = hist {
        histogram_plot(&painter, rect, h, st.channel, Color32::from_gray(52));
    }
    for i in 1..4 {
        let f = i as f32 / 4.0;
        let g = Stroke::new(1.0, Color32::from_gray(50));
        painter.line_segment(
            [
                pos2(rect.min.x + f * rect.width(), rect.min.y),
                pos2(rect.min.x + f * rect.width(), rect.max.y),
            ],
            g,
        );
        painter.line_segment(
            [
                pos2(rect.min.x, rect.min.y + f * rect.height()),
                pos2(rect.max.x, rect.min.y + f * rect.height()),
            ],
            g,
        );
    }
    painter.line_segment(
        [rect.left_bottom(), rect.right_top()],
        Stroke::new(1.0, Color32::from_gray(70)),
    );

    // Interaction.
    curve.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let grab = 9.0;
    if resp.drag_started()
        && let Some(p) = resp.interact_pointer_pos()
    {
        let near = curve
            .iter()
            .enumerate()
            .map(|(i, c)| (i, to_screen(*c).distance(p)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        st.dragging = match near {
            Some((i, d)) if d <= grab => Some(i),
            _ => {
                let c = to_curve(p);
                let y = eval_curve(curve, c[0].clamp(0.0, 1.0));
                curve.push([c[0].clamp(0.0, 1.0), y]);
                curve.sort_by(|a, b| a[0].total_cmp(&b[0]));
                t.changed = true;
                curve.iter().position(|q| q[0] == c[0].clamp(0.0, 1.0))
            }
        };
    }
    if resp.clicked()
        && let Some(p) = resp.interact_pointer_pos()
        && !curve.iter().any(|c| to_screen(*c).distance(p) <= grab)
    {
        let c = to_curve(p);
        curve.push([c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0)]);
        curve.sort_by(|a, b| a[0].total_cmp(&b[0]));
        t.changed = true;
        t.done = true;
    }
    if resp.dragged()
        && let (Some(i), Some(p)) = (st.dragging, resp.interact_pointer_pos())
        && i < curve.len()
    {
        let c = to_curve(p);
        let outside = c[1] < -0.12 || c[1] > 1.12 || c[0] < -0.12 || c[0] > 1.12;
        if outside && curve.len() > 2 {
            curve.remove(i);
            st.dragging = None;
        } else {
            // Keep the order: a point cannot pass its neighbours.
            let lo = if i > 0 { curve[i - 1][0] + 0.01 } else { 0.0 };
            let hi = if i + 1 < curve.len() {
                curve[i + 1][0] - 0.01
            } else {
                1.0
            };
            curve[i] = [c[0].clamp(lo, hi.max(lo)), c[1].clamp(0.0, 1.0)];
        }
        t.changed = true;
    }
    if resp.drag_stopped() {
        st.dragging = None;
        t.done = true;
    }

    // Curve and points.
    let pts: Vec<Pos2> = (0..=128)
        .map(|i| i as f32 / 128.0)
        .map(|x| to_screen([x, eval_curve(curve, x)]))
        .collect();
    painter.add(egui::Shape::line(pts, Stroke::new(2.0, color)));
    for (i, c) in curve.iter().enumerate() {
        let p = to_screen(*c);
        let active = st.dragging == Some(i);
        painter.circle(
            p,
            if active { 5.0 } else { 4.0 },
            if active {
                color
            } else {
                Color32::from_gray(22)
            },
            Stroke::new(1.5, color),
        );
    }
    if let Some(p) = resp.hover_pos() {
        let c = to_curve(p);
        painter.text(
            rect.left_top() + Vec2::new(4.0, 2.0),
            egui::Align2::LEFT_TOP,
            tr!(
                "in {input}  out {output}",
                input = format!("{:.0}", c[0].clamp(0.0, 1.0) * 255.0),
                output = format!("{:.0}", eval_curve(curve, c[0].clamp(0.0, 1.0)) * 255.0)
            ),
            egui::FontId::monospace(11.0),
            Color32::from_gray(160),
        );
    }
}

/// Filled histogram of one channel, scaled to its peak.
fn histogram_plot(
    painter: &egui::Painter,
    rect: egui::Rect,
    h: &Histogram,
    ch: usize,
    color: Color32,
) {
    let peak = h.peak(ch);
    let w = rect.width() / 256.0;
    for (i, v) in h.bins[ch].iter().enumerate() {
        let f = (v / peak).min(1.0);
        if f <= 0.0 {
            continue;
        }
        let x = rect.min.x + i as f32 * w;
        painter.rect_filled(
            egui::Rect::from_min_max(
                pos2(x, rect.max.y - f * rect.height()),
                pos2(x + w + 0.5, rect.max.y),
            ),
            0.0,
            color,
        );
    }
}

/// Position of the gamma marker (input value mapped to 50 % grey).
fn gamma_pos(l: &Levels) -> f32 {
    l.in_black + (l.in_white - l.in_black) * 0.5f32.powf(l.gamma)
}

/// Histogram with draggable black, gamma and white input markers.
fn levels_graph(
    ui: &mut egui::Ui,
    l: &mut Levels,
    hist: Option<&Histogram>,
    st: &mut EditorState,
    t: &mut Track,
) {
    let width = ui.available_width().min(260.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, 104.0), Sense::drag());
    let plot = egui::Rect::from_min_max(rect.min, pos2(rect.max.x, rect.max.y - 14.0));
    let painter = ui.painter_at(rect.expand(6.0));
    painter.rect_filled(plot, 2.0, Color32::from_gray(22));
    match hist {
        Some(h) => histogram_plot(
            &painter,
            plot,
            h,
            st.channel,
            Color32::from_gray(22).lerp_to_gamma(CHANNEL_COLORS[st.channel], 0.7),
        ),
        None => {
            painter.text(
                plot.center(),
                egui::Align2::CENTER_CENTER,
                tr("no histogram"),
                egui::FontId::proportional(11.0),
                Color32::from_gray(90),
            );
        }
    }
    let to_x = |v: f32| plot.min.x + v.clamp(0.0, 1.0) * plot.width();
    let to_v = |x: f32| ((x - plot.min.x) / plot.width()).clamp(0.0, 1.0);
    let markers = [l.in_black, gamma_pos(l), l.in_white];
    if resp.drag_started()
        && let Some(p) = resp.interact_pointer_pos()
    {
        st.dragging = markers
            .iter()
            .enumerate()
            .map(|(i, m)| (i, (to_x(*m) - p.x).abs()))
            .filter(|(_, d)| *d <= 10.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i);
    }
    if resp.dragged()
        && let (Some(i), Some(p)) = (st.dragging, resp.interact_pointer_pos())
    {
        let v = to_v(p.x);
        let gap = 2.0 / 255.0;
        match i {
            0 => l.in_black = v.min(l.in_white - gap).max(0.0),
            2 => l.in_white = v.max(l.in_black + gap).min(1.0),
            _ => {
                let f = ((v - l.in_black) / (l.in_white - l.in_black).max(1e-4)).clamp(0.01, 0.99);
                l.gamma = (f.ln() / 0.5f32.ln()).clamp(0.1, 9.99);
            }
        }
        t.changed = true;
    }
    if resp.drag_stopped() {
        st.dragging = None;
        t.done = true;
    }
    // Markers: triangles under the graph.
    let fills = [Color32::BLACK, Color32::from_gray(128), Color32::WHITE];
    for (i, m) in [l.in_black, gamma_pos(l), l.in_white].iter().enumerate() {
        let x = to_x(*m);
        let y = plot.max.y + 2.0;
        let active = st.dragging == Some(i);
        painter.add(egui::Shape::convex_polygon(
            vec![pos2(x, y), pos2(x + 6.0, y + 11.0), pos2(x - 6.0, y + 11.0)],
            fills[i],
            Stroke::new(
                1.0,
                if active {
                    Color32::from_rgb(90, 140, 250)
                } else {
                    Color32::from_gray(150)
                },
            ),
        ));
    }
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }
}

/// Whether the editor for this adjustment shows a histogram.
pub fn wants_histogram(adj: Option<&Adjustment>) -> bool {
    matches!(
        adj,
        Some(Adjustment::Levels { .. } | Adjustment::Curves { .. })
    )
}

/// A hue strip with the range marked: full strength bright, fades dimmer.
fn hue_range_bar(ui: &mut egui::Ui, r: [f32; 4]) {
    let w = ui.available_width().min(260.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 14.0), egui::Sense::hover());
    let p = ui.painter();
    let steps = 72;
    for i in 0..steps {
        let h0 = i as f32 / steps as f32;
        let x0 = rect.left() + h0 * w;
        let deg = h0 * 360.0;
        // Weight of this hue in the range (as Photoshop's trapezoid).
        let rel = |v: f32| (v - r[0]).rem_euclid(360.0);
        let (c, ls, cs, rs) = (rel(deg), rel(r[1]), rel(r[2]), rel(r[3]));
        let m = if c >= ls && c <= cs {
            1.0
        } else if c < ls {
            c / ls.max(1e-3)
        } else if c <= rs {
            (rs - c) / (rs - cs).max(1e-3)
        } else {
            0.0
        };
        let col: egui::Color32 = egui::ecolor::Hsva::new(h0, 1.0, 0.25 + 0.75 * m, 1.0).into();
        p.rect_filled(egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x0 + w / steps as f32 + 0.5, rect.bottom())), 0.0, col);
    }
}
