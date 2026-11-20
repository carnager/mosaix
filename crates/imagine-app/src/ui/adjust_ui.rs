//! Property editors for adjustment layers, including an interactive curves
//! graph. Each editor returns the changed adjustment and whether the edit
//! is finished (slider released), so a drag becomes one undo step.

use egui::{Color32, Pos2, RichText, Sense, Stroke, Vec2, pos2};
use imagine_core::Adjustment;
use imagine_core::adjust::{Levels, eval_curve};

#[derive(Default)]
pub struct EditorState {
    /// 0 = RGB, 1–3 = red, green, blue.
    channel: usize,
    dragging: Option<usize>,
}

const CHANNELS: [&str; 4] = ["RGB", "Rot", "Grün", "Blau"];
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
        ui.add_sized([96.0, 18.0], egui::Label::new(label).truncate());
        let r = ui.add(w);
        t.add(&r);
    });
}

fn slider(v: &mut f32, range: std::ops::RangeInclusive<f32>) -> egui::Slider<'_> {
    egui::Slider::new(v, range).clamping(egui::SliderClamping::Always).max_decimals(1)
}

fn channel_pick(ui: &mut egui::Ui, st: &mut EditorState) {
    ui.horizontal(|ui| {
        for (i, n) in CHANNELS.iter().enumerate() {
            if ui.selectable_label(st.channel == i, RichText::new(*n).color(CHANNEL_COLORS[i])).clicked() {
                st.channel = i;
                st.dragging = None;
            }
        }
    });
}

/// Shows the editor; `Some((new, done))` when something changed or finished.
pub fn editor(ui: &mut egui::Ui, adj: &Adjustment, st: &mut EditorState) -> Option<(Adjustment, bool)> {
    let mut a = adj.clone();
    let mut t = Track::default();
    ui.label(RichText::new(adj.name()).strong());
    match &mut a {
        Adjustment::BrightnessContrast { brightness, contrast } => {
            row(ui, "Helligkeit", slider(brightness, -150.0..=150.0), &mut t);
            row(ui, "Kontrast", slider(contrast, -50.0..=100.0), &mut t);
        }
        Adjustment::Levels { master, red, green, blue } => {
            channel_pick(ui, st);
            let l: &mut Levels = match st.channel {
                1 => red,
                2 => green,
                3 => blue,
                _ => master,
            };
            let mut v = [l.in_black * 255.0, l.in_white * 255.0, l.out_black * 255.0, l.out_white * 255.0];
            row(ui, "Tiefen", slider(&mut v[0], 0.0..=253.0).max_decimals(0), &mut t);
            row(ui, "Gamma", slider(&mut l.gamma, 0.1..=9.99).logarithmic(true).max_decimals(2), &mut t);
            row(ui, "Lichter", slider(&mut v[1], 2.0..=255.0).max_decimals(0), &mut t);
            row(ui, "Ausgabe Tiefen", slider(&mut v[2], 0.0..=255.0).max_decimals(0), &mut t);
            row(ui, "Ausgabe Lichter", slider(&mut v[3], 0.0..=255.0).max_decimals(0), &mut t);
            l.in_black = v[0] / 255.0;
            l.in_white = (v[1] / 255.0).max(l.in_black + 2.0 / 255.0);
            l.out_black = v[2] / 255.0;
            l.out_white = v[3] / 255.0;
        }
        Adjustment::Curves { master, red, green, blue } => {
            channel_pick(ui, st);
            let ch = st.channel;
            let curve = match ch {
                1 => red,
                2 => green,
                3 => blue,
                _ => master,
            };
            curve_graph(ui, curve, CHANNEL_COLORS[ch], st, &mut t);
            ui.label(RichText::new("Klick setzt einen Punkt, Ziehen verschiebt ihn, aus dem Feld ziehen entfernt ihn.").weak().small());
        }
        Adjustment::HueSaturation { hue, saturation, lightness, colorize } => {
            let r = ui.checkbox(colorize, "Färben");
            t.add(&r);
            row(ui, "Farbton", slider(hue, -180.0..=180.0).suffix("°"), &mut t);
            row(ui, "Sättigung", slider(saturation, -100.0..=100.0), &mut t);
            row(ui, "Helligkeit", slider(lightness, -100.0..=100.0), &mut t);
        }
        Adjustment::Vibrance { vibrance, saturation } => {
            row(ui, "Dynamik", slider(vibrance, -100.0..=100.0), &mut t);
            row(ui, "Sättigung", slider(saturation, -100.0..=100.0), &mut t);
        }
        Adjustment::BlackWhite { weights } => {
            for (w, n) in weights.iter_mut().zip(["Rottöne", "Gelbtöne", "Grüntöne", "Cyantöne", "Blautöne", "Magentatöne"]) {
                row(ui, n, slider(w, -200.0..=300.0).suffix(" %").max_decimals(0), &mut t);
            }
        }
        Adjustment::Exposure { exposure, offset, gamma } => {
            row(ui, "Belichtung", slider(exposure, -10.0..=10.0).max_decimals(2), &mut t);
            row(ui, "Versatz", slider(offset, -0.5..=0.5).max_decimals(3), &mut t);
            row(ui, "Gamma", slider(gamma, 0.1..=9.99).logarithmic(true).max_decimals(2), &mut t);
        }
        Adjustment::Invert => {
            ui.label(RichText::new("Keine Einstellungen").weak());
        }
        Adjustment::Threshold { level } => row(ui, "Schwellenwert", slider(level, 1.0..=255.0).max_decimals(0), &mut t),
        Adjustment::Posterize { levels } => row(ui, "Stufen", slider(levels, 2.0..=255.0).max_decimals(0), &mut t),
    }
    if ui.small_button("Zurücksetzen").clicked()
        && let Some(d) = Adjustment::defaults().into_iter().find(|d| d.key() == adj.key())
    {
        return Some((d, true));
    }
    (t.changed || t.done).then_some((a, t.done))
}

/// Interactive curve: points in 0…1, drawn in a square graph.
fn curve_graph(ui: &mut egui::Ui, curve: &mut Vec<[f32; 2]>, color: Color32, st: &mut EditorState, t: &mut Track) {
    let side = ui.available_width().min(260.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(side), Sense::click_and_drag());
    let to_screen = |p: [f32; 2]| pos2(rect.min.x + p[0] * rect.width(), rect.max.y - p[1] * rect.height());
    let to_curve = |s: Pos2| [((s.x - rect.min.x) / rect.width()), ((rect.max.y - s.y) / rect.height())];
    let painter = ui.painter_at(rect.expand(6.0));
    painter.rect_filled(rect, 2.0, Color32::from_gray(22));
    for i in 1..4 {
        let f = i as f32 / 4.0;
        let g = Stroke::new(1.0, Color32::from_gray(50));
        painter.line_segment([pos2(rect.min.x + f * rect.width(), rect.min.y), pos2(rect.min.x + f * rect.width(), rect.max.y)], g);
        painter.line_segment([pos2(rect.min.x, rect.min.y + f * rect.height()), pos2(rect.max.x, rect.min.y + f * rect.height())], g);
    }
    painter.line_segment([rect.left_bottom(), rect.right_top()], Stroke::new(1.0, Color32::from_gray(70)));

    // Interaction.
    curve.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let grab = 9.0;
    if resp.drag_started()
        && let Some(p) = resp.interact_pointer_pos()
    {
        let near = curve.iter().enumerate().map(|(i, c)| (i, to_screen(*c).distance(p))).min_by(|a, b| a.1.total_cmp(&b.1));
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
            let hi = if i + 1 < curve.len() { curve[i + 1][0] - 0.01 } else { 1.0 };
            curve[i] = [c[0].clamp(lo, hi.max(lo)), c[1].clamp(0.0, 1.0)];
        }
        t.changed = true;
    }
    if resp.drag_stopped() {
        st.dragging = None;
        t.done = true;
    }

    // Curve and points.
    let pts: Vec<Pos2> = (0..=128).map(|i| i as f32 / 128.0).map(|x| to_screen([x, eval_curve(curve, x)])).collect();
    painter.add(egui::Shape::line(pts, Stroke::new(2.0, color)));
    for (i, c) in curve.iter().enumerate() {
        let p = to_screen(*c);
        let active = st.dragging == Some(i);
        painter.circle(p, if active { 5.0 } else { 4.0 }, if active { color } else { Color32::from_gray(22) }, Stroke::new(1.5, color));
    }
    if let Some(p) = resp.hover_pos() {
        let c = to_curve(p);
        painter.text(
            rect.left_top() + Vec2::new(4.0, 2.0),
            egui::Align2::LEFT_TOP,
            format!("ein {:.0}  aus {:.0}", c[0].clamp(0.0, 1.0) * 255.0, eval_curve(curve, c[0].clamp(0.0, 1.0)) * 255.0),
            egui::FontId::monospace(11.0),
            Color32::from_gray(160),
        );
    }
}
