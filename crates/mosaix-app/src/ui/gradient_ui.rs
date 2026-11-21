//! A gradient editor like Photoshop's: the gradient as a bar, opacity
//! stops above it, colour stops below. Click a stop to select it, drag it to
//! move it, click next to the bar to add one, drag it off the bar (or press
//! the delete button) to remove it.

use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, vec2};
use mosaix_core::gradient::{AlphaStop, ColorStop, Gradient, GradientStyle};
use mosaix_core::tr;

/// What an edit did: something changed / the edit is finished (one undo step).
#[derive(Default, Clone, Copy)]
pub struct Out {
    pub changed: bool,
    pub done: bool,
}

impl Out {
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

#[derive(Clone, Copy, PartialEq)]
enum Sel {
    Color(usize),
    Alpha(usize),
}

const BAR_H: f32 = 18.0;
const MARK: f32 = 9.0;

fn checker(p: &egui::Painter, r: Rect) {
    p.rect_filled(r, 0.0, Color32::from_gray(200));
    let s = 6.0;
    let (nx, ny) = ((r.width() / s).ceil() as i32, (r.height() / s).ceil() as i32);
    for y in 0..ny {
        for x in (y % 2..nx).step_by(2) {
            let min = r.min + vec2(x as f32 * s, y as f32 * s);
            let c = Rect::from_min_max(min, (min + vec2(s, s)).min(r.max));
            p.rect_filled(c, 0.0, Color32::from_gray(255));
        }
    }
}

fn rgba(c: [f32; 4]) -> Color32 {
    egui::Rgba::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]).into()
}

fn paint_bar(p: &egui::Painter, r: Rect, g: &Gradient) {
    checker(p, r);
    // One mesh with a colour per column (separate rectangles would be
    // anti-aliased against each other and let the checkerboard through).
    let lut = g.lut();
    let n = (r.width() as usize).clamp(2, 512);
    let mut mesh = egui::Mesh::default();
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let c = rgba(lut.at(t));
        let x = r.min.x + t * r.width();
        mesh.colored_vertex(Pos2::new(x, r.min.y), c);
        mesh.colored_vertex(Pos2::new(x, r.max.y), c);
        if i > 0 {
            let k = (2 * i) as u32;
            mesh.add_triangle(k - 2, k - 1, k);
            mesh.add_triangle(k - 1, k + 1, k);
        }
    }
    p.add(Shape::mesh(mesh));
    p.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::from_gray(90)), egui::StrokeKind::Inside);
}

/// Edits `g`; `salt` keeps several editors apart.
pub fn editor(ui: &mut egui::Ui, g: &mut Gradient, salt: &str) -> Out {
    let mut out = Out::default();
    let id = ui.scope_id().with(("gradient editor", salt));
    let mut sel: Sel = ui.data(|d| d.get_temp(id)).unwrap_or(Sel::Color(0));
    // Opacity stops are optional (none = opaque); edit two opaque ones then.
    let had_alphas = !g.alphas.is_empty();
    if !had_alphas {
        g.alphas = vec![AlphaStop { at: 0.0, opacity: 1.0, mid: 0.5 }, AlphaStop { at: 1.0, opacity: 1.0, mid: 0.5 }];
    }
    let width = (ui.available_width() - 4.0).clamp(120.0, 280.0);
    let (full, _) = ui.allocate_exact_size(vec2(width, BAR_H + 2.0 * MARK + 4.0), Sense::hover());
    let bar = Rect::from_min_size(full.min + vec2(MARK * 0.5, MARK + 2.0), vec2(width - MARK, BAR_H));
    let p = ui.painter().clone();
    paint_bar(&p, bar, g);
    let x_of = |t: f32| bar.min.x + t * bar.width();
    let t_of = |x: f32| ((x - bar.min.x) / bar.width()).clamp(0.0, 1.0);
    // Adding: a click in a marker row.
    let top = Rect::from_min_max(Pos2::new(bar.min.x, full.min.y), Pos2::new(bar.max.x, bar.min.y));
    let bottom = Rect::from_min_max(Pos2::new(bar.min.x, bar.max.y), Pos2::new(bar.max.x, full.max.y));
    let add_top = ui.interact(top, id.with("add alpha"), Sense::click());
    let add_bottom = ui.interact(bottom, id.with("add color"), Sense::click());
    let mut remove: Option<Sel> = None;
    let selected = Color32::from_rgb(40, 120, 230);
    // Colour stops.
    for i in 0..g.colors.len() {
        let s = g.colors[i];
        let x = x_of(s.at);
        let r = Rect::from_center_size(Pos2::new(x, bar.max.y + MARK * 0.5 + 1.0), vec2(MARK, MARK + 2.0));
        let resp = ui.interact(r, id.with(("c", i)), Sense::click_and_drag());
        if resp.clicked() || resp.drag_started() {
            sel = Sel::Color(i);
        }
        if resp.dragged()
            && let Some(pos) = resp.interact_pointer_pos()
        {
            g.colors[i].at = t_of(pos.x);
            out.changed = true;
        }
        if resp.drag_stopped() {
            out.done = true;
            if resp.interact_pointer_pos().is_some_and(|p| p.y > full.max.y + 24.0) && g.colors.len() > 2 {
                remove = Some(Sel::Color(i));
            }
        }
        let tip = Pos2::new(x, bar.max.y + 1.0);
        let edge = if sel == Sel::Color(i) { selected } else { Color32::from_gray(60) };
        p.add(Shape::convex_polygon(
            vec![tip, Pos2::new(x + MARK * 0.5, tip.y + MARK * 0.5), Pos2::new(x + MARK * 0.5, tip.y + MARK), Pos2::new(x - MARK * 0.5, tip.y + MARK), Pos2::new(x - MARK * 0.5, tip.y + MARK * 0.5)],
            rgba([s.color[0], s.color[1], s.color[2], 1.0]),
            Stroke::new(if sel == Sel::Color(i) { 2.0 } else { 1.0 }, edge),
        ));
    }
    // Opacity stops.
    for i in 0..g.alphas.len() {
        let s = g.alphas[i];
        let x = x_of(s.at);
        let r = Rect::from_center_size(Pos2::new(x, bar.min.y - MARK * 0.5 - 1.0), vec2(MARK, MARK + 2.0));
        let resp = ui.interact(r, id.with(("a", i)), Sense::click_and_drag());
        if resp.clicked() || resp.drag_started() {
            sel = Sel::Alpha(i);
        }
        if resp.dragged()
            && let Some(pos) = resp.interact_pointer_pos()
        {
            g.alphas[i].at = t_of(pos.x);
            out.changed = true;
        }
        if resp.drag_stopped() {
            out.done = true;
            if resp.interact_pointer_pos().is_some_and(|p| p.y < full.min.y - 24.0) && g.alphas.len() > 2 {
                remove = Some(Sel::Alpha(i));
            }
        }
        let tip = Pos2::new(x, bar.min.y - 1.0);
        let edge = if sel == Sel::Alpha(i) { selected } else { Color32::from_gray(60) };
        let v = (s.opacity.clamp(0.0, 1.0) * 255.0) as u8;
        p.add(Shape::convex_polygon(
            vec![Pos2::new(x - MARK * 0.5, tip.y - MARK), Pos2::new(x + MARK * 0.5, tip.y - MARK), Pos2::new(x + MARK * 0.5, tip.y - MARK * 0.5), tip, Pos2::new(x - MARK * 0.5, tip.y - MARK * 0.5)],
            Color32::from_gray(255 - v),
            Stroke::new(if sel == Sel::Alpha(i) { 2.0 } else { 1.0 }, edge),
        ));
    }
    if add_bottom.clicked()
        && let Some(pos) = add_bottom.interact_pointer_pos()
    {
        let t = t_of(pos.x);
        let c = g.eval(t);
        g.colors.push(ColorStop { at: t, color: [c[0], c[1], c[2]], mid: 0.5 });
        sel = Sel::Color(g.colors.len() - 1);
        out.click();
    }
    if add_top.clicked()
        && let Some(pos) = add_top.interact_pointer_pos()
    {
        let t = t_of(pos.x);
        g.alphas.push(AlphaStop { at: t, opacity: g.eval(t)[3], mid: 0.5 });
        sel = Sel::Alpha(g.alphas.len() - 1);
        out.click();
    }
    // The selected stop.
    sel = match sel {
        Sel::Color(i) if i >= g.colors.len() => Sel::Color(0),
        Sel::Alpha(i) if i >= g.alphas.len() => Sel::Alpha(0),
        s => s,
    };
    ui.horizontal(|ui| {
        let loc = match sel {
            Sel::Color(i) => {
                let c = &mut g.colors[i].color;
                let mut rgb = egui::Rgba::from_rgb(c[0], c[1], c[2]);
                let r = egui::color_picker::color_edit_button_rgba(ui, &mut rgb, egui::color_picker::Alpha::Opaque);
                if r.changed() {
                    *c = [rgb.r(), rgb.g(), rgb.b()];
                }
                out.add(&r);
                &mut g.colors[i].at
            }
            Sel::Alpha(i) => {
                let mut pct = g.alphas[i].opacity * 100.0;
                let r = ui.add(egui::DragValue::new(&mut pct).range(0.0..=100.0).suffix(" %").speed(0.5));
                g.alphas[i].opacity = pct / 100.0;
                out.add(&r);
                &mut g.alphas[i].at
            }
        };
        let mut pct = *loc * 100.0;
        let r = ui.add(egui::DragValue::new(&mut pct).range(0.0..=100.0).suffix(" %").speed(0.5)).on_hover_text(tr("Location"));
        *loc = pct / 100.0;
        out.add(&r);
        let can = match sel {
            Sel::Color(_) => g.colors.len() > 2,
            Sel::Alpha(_) => g.alphas.len() > 2,
        };
        let r = ui
            .add_enabled_ui(can, |ui| super::icons::button(ui, super::icons::Icon::Trash, 20.0, false))
            .inner
            .on_hover_text(tr("Delete Stop"));
        if r.clicked() {
            remove = Some(sel);
        }
    });
    match remove {
        Some(Sel::Color(i)) => {
            g.colors.remove(i);
            sel = Sel::Color(0);
            out.click();
        }
        Some(Sel::Alpha(i)) => {
            g.alphas.remove(i);
            sel = Sel::Alpha(0);
            out.click();
        }
        None => {}
    }
    ui.horizontal(|ui| {
        ui.label(tr("Smoothness"));
        ui.spacing_mut().slider_width = (width - 140.0).max(40.0);
        let mut pct = g.smoothness * 100.0;
        let r = ui.add(egui::Slider::new(&mut pct, 0.0..=100.0).suffix(" %").max_decimals(0));
        g.smoothness = pct / 100.0;
        out.add(&r);
    });
    if g.alphas.iter().all(|a| a.opacity >= 0.999) {
        g.alphas.clear();
    }
    ui.data_mut(|d| d.insert_temp(id, sel));
    out
}

/// A combo box for the gradient style.
pub fn style_combo(ui: &mut egui::Ui, style: &mut GradientStyle, salt: &str) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(("gradient style", salt)).selected_text(style.label()).show_ui(ui, |ui| {
        for s in GradientStyle::ALL {
            if ui.selectable_label(*style == s, s.label()).clicked() && *style != s {
                *style = s;
                changed = true;
            }
        }
    });
    changed
}
