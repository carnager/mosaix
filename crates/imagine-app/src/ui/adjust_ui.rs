//! Property editors for adjustment layers, including an interactive curves
//! graph. Each editor returns the changed adjustment and whether the edit
//! is finished (slider released), so a drag becomes one undo step.

use egui::{Color32, Pos2, RichText, Sense, Stroke, Vec2, pos2};
use imagine_core::adjust::{Levels, eval_curve};
use imagine_core::{Adjustment, tr};
use imagine_gpu::Histogram;

#[derive(Default)]
pub struct EditorState {
    /// 0 = RGB, 1–3 = red, green, blue.
    channel: usize,
    dragging: Option<usize>,
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
        } => {
            let r = ui.checkbox(colorize, tr("Colorize"));
            t.add(&r);
            row(ui, "Hue", slider(hue, -180.0..=180.0).suffix("°"), &mut t);
            row(ui, "Saturation", slider(saturation, -100.0..=100.0), &mut t);
            row(ui, "Lightness", slider(lightness, -100.0..=100.0), &mut t);
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
