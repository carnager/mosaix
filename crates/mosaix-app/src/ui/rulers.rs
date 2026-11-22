//! Rulers along the canvas, guides and the grid. Guides are pulled out of a
//! ruler, moved with the Move tool (or with Ctrl held) and deleted by
//! dragging them back onto a ruler.

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, pos2, vec2};
use glam::DVec2;
use mosaix_core::Guide;

use super::panels::{Action, Model, ToolId};
use crate::settings::ViewOpts;

/// Ruler thickness in points.
pub const RULER: f32 = 18.0;
const GUIDE: Color32 = Color32::from_rgb(0, 200, 230);
const HIT: f32 = 4.0;

/// A guide being dragged: a new one (`index` None) or an existing one.
#[derive(Debug, Clone, Copy)]
pub struct GuideDrag {
    pub index: Option<usize>,
    pub vertical: bool,
}

/// Document ↔ points (egui) on the canvas.
struct Map {
    view: mosaix_core::View,
    ppp: f64,
}

impl Map {
    fn to_pt(&self, d: DVec2) -> Pos2 {
        let s = self.view.doc_to_screen(d) / self.ppp;
        pos2(s.x as f32, s.y as f32)
    }
    fn to_doc(&self, p: Pos2) -> DVec2 {
        self.view.screen_to_doc(DVec2::new(p.x as f64, p.y as f64) * self.ppp)
    }
    /// Points per document px.
    fn scale(&self) -> f64 {
        self.view.zoom / self.ppp
    }
}

/// A "nice" step (1, 2, 5 × 10ⁿ) of at least `min` document px.
fn nice_step(min: f64) -> f64 {
    let e = 10f64.powf(min.max(1e-9).log10().floor());
    [1.0, 2.0, 5.0, 10.0].into_iter().map(|k| k * e).find(|s| *s >= min).unwrap_or(10.0 * e)
}

/// Draws grid, guides and rulers over the canvas `rect` and handles guide
/// dragging. `drag` is the guide being moved, kept between frames.
pub fn show(ctx: &egui::Context, rect: Rect, m: &Model, drag: &mut Option<GuideDrag>, out: &mut Vec<Action>) {
    let opts = m.view_opts;
    let map = Map { view: m.view, ppp: ctx.pixels_per_point() as f64 };
    let painter = ctx.layer_painter(egui::LayerId::background()).with_clip_rect(rect);
    let (vmin, vmax) = {
        let c = [rect.min, rect.right_top(), rect.max, rect.left_bottom()].map(|p| map.to_doc(p));
        let min = c.iter().fold(DVec2::INFINITY, |a, p| a.min(*p));
        let max = c.iter().fold(DVec2::NEG_INFINITY, |a, p| a.max(*p));
        (min, max)
    };
    if opts.grid {
        grid(&painter, &map, opts, (vmin, vmax), m.doc_size);
    }
    let pointer = ctx.pointer_latest_pos();
    let canvas_inner = if opts.rulers { Rect::from_min_max(rect.min + vec2(RULER, RULER), rect.max) } else { rect };
    // Guides (the dragged one at the pointer).
    if opts.guides || drag.is_some() {
        let line = |vertical: bool, pos: f64, color: Color32| {
            let (a, b) = if vertical {
                (DVec2::new(pos, vmin.y), DVec2::new(pos, vmax.y))
            } else {
                (DVec2::new(vmin.x, pos), DVec2::new(vmax.x, pos))
            };
            painter.line_segment([map.to_pt(a), map.to_pt(b)], Stroke::new(1.0, color));
        };
        for (i, g) in m.guides.iter().enumerate() {
            if drag.is_some_and(|d| d.index == Some(i)) {
                continue;
            }
            if opts.guides {
                line(g.vertical, g.pos, GUIDE);
            }
        }
        if let (Some(d), Some(p)) = (*drag, pointer) {
            let pos = guide_pos(&map, d.vertical, p, ctx.input(|i| i.modifiers.shift));
            let over = canvas_inner.contains(p);
            line(d.vertical, pos, if over { GUIDE } else { GUIDE.gamma_multiply(0.4) });
            if ctx.input(|i| i.pointer.any_released()) {
                let mut guides = m.guides.clone();
                match (d.index, over) {
                    (Some(i), true) if i < guides.len() => guides[i].pos = pos,
                    (Some(i), false) if i < guides.len() => {
                        guides.remove(i);
                    }
                    (None, true) => guides.push(Guide { vertical: d.vertical, pos }),
                    _ => {}
                }
                out.push(Action::SetGuides(guides));
                *drag = None;
            }
        }
    }
    // Existing guides can be grabbed with the Move tool or with Ctrl.
    let ctrl = ctx.input(|i| i.modifiers.command);
    if opts.guides && drag.is_none() && (m.tool == ToolId::Move || ctrl) {
        for (i, g) in m.guides.iter().enumerate() {
            let a = map.to_pt(if g.vertical { DVec2::new(g.pos, vmin.y) } else { DVec2::new(vmin.x, g.pos) });
            let b = map.to_pt(if g.vertical { DVec2::new(g.pos, vmax.y) } else { DVec2::new(vmax.x, g.pos) });
            // Only axis-aligned views give the guide a straight strip to grab.
            let strip = Rect::from_two_pos(a, b).expand(HIT).intersect(canvas_inner);
            if !m.view.is_axis_aligned() || strip.width() <= 0.0 || strip.height() <= 0.0 {
                continue;
            }
            let id = egui::Id::unique(("guide", i));
            let r = egui::Area::new(id)
                .fixed_pos(strip.min)
                .order(egui::Order::Middle)
                .show(ctx, |ui| ui.allocate_response(strip.size(), Sense::drag()))
                .inner;
            let vertical_on_screen = (a.x - b.x).abs() < (a.y - b.y).abs();
            if r.hovered() || r.dragged() {
                ctx.set_cursor_icon(if vertical_on_screen { egui::CursorIcon::ResizeHorizontal } else { egui::CursorIcon::ResizeVertical });
            }
            if r.drag_started() {
                *drag = Some(GuideDrag { index: Some(i), vertical: g.vertical });
            }
        }
    }
    if opts.rulers {
        rulers(ctx, rect, &map, m, pointer, drag);
    }
}

/// Where a dragged guide goes: the pointer, on whole pixels when zoomed in,
/// on ruler ticks with Shift.
fn guide_pos(map: &Map, vertical: bool, p: Pos2, shift: bool) -> f64 {
    let d = map.to_doc(p);
    let v = if vertical { d.x } else { d.y };
    if shift {
        let step = nice_step(8.0 / map.scale());
        (v / step).round() * step
    } else if map.scale() >= 1.0 {
        v.round()
    } else {
        // About a screen point's precision.
        let step = nice_step(1.0 / map.scale());
        (v / step).round() * step
    }
}

fn grid(painter: &egui::Painter, map: &Map, opts: ViewOpts, (vmin, vmax): (DVec2, DVec2), doc: (u32, u32)) {
    let step = opts.grid_step.max(1.0);
    let sub = step / opts.grid_div.max(1) as f64;
    // Bluish, visible on light and dark pictures.
    let major = Color32::from_rgba_unmultiplied(90, 130, 255, 150);
    let minor = Color32::from_rgba_unmultiplied(90, 130, 255, 60);
    let (x0, x1) = (vmin.x.max(0.0), vmax.x.min(doc.0 as f64));
    let (y0, y1) = (vmin.y.max(0.0), vmax.y.min(doc.1 as f64));
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    for (s, color) in [(sub, minor), (step, major)] {
        if s * map.scale() < 6.0 || (s == sub && opts.grid_div <= 1) {
            continue;
        }
        let mut x = (x0 / s).ceil() * s;
        while x <= x1 {
            painter.line_segment([map.to_pt(DVec2::new(x, y0)), map.to_pt(DVec2::new(x, y1))], Stroke::new(1.0, color));
            x += s;
        }
        let mut y = (y0 / s).ceil() * s;
        while y <= y1 {
            painter.line_segment([map.to_pt(DVec2::new(x0, y)), map.to_pt(DVec2::new(x1, y))], Stroke::new(1.0, color));
            y += s;
        }
    }
}

fn rulers(ctx: &egui::Context, rect: Rect, map: &Map, m: &Model, pointer: Option<Pos2>, drag: &mut Option<GuideDrag>) {
    let style = ctx.global_style();
    let bg = style.visuals.panel_fill;
    let fg = style.visuals.weak_text_color();
    let top = Rect::from_min_max(pos2(rect.min.x + RULER, rect.min.y), pos2(rect.max.x, rect.min.y + RULER));
    let left = Rect::from_min_max(pos2(rect.min.x, rect.min.y + RULER), pos2(rect.min.x + RULER, rect.max.y));
    let corner = Rect::from_min_size(rect.min, vec2(RULER, RULER));
    // Ticks: about one label every 60 points.
    let step = nice_step(60.0 / map.scale());
    let minor = step / 10.0;
    let aligned = m.view.is_axis_aligned();
    for (horizontal, r) in [(true, top), (false, left)] {
        let resp = egui::Area::new(egui::Id::unique(("ruler", horizontal)))
            .fixed_pos(r.min)
            .order(egui::Order::Middle)
            .show(ctx, |ui| {
                let resp = ui.allocate_response(r.size(), Sense::drag());
                let p = ui.painter();
                p.rect_filled(r, 0.0, bg);
                p.line_segment(
                    if horizontal { [r.left_bottom(), r.right_bottom()] } else { [r.right_top(), r.right_bottom()] },
                    Stroke::new(1.0, style.visuals.widgets.noninteractive.bg_stroke.color),
                );
                if aligned {
                    // Which document axis this ruler shows (the view may be turned by 90°).
                    let probe = |t: f32| if horizontal { pos2(t, r.center().y) } else { pos2(r.center().x, t) };
                    let (from, to) = if horizontal { (r.min.x, r.max.x) } else { (r.min.y, r.max.y) };
                    let (a, b) = (map.to_doc(probe(from)), map.to_doc(probe(to)));
                    let along_x = (b.x - a.x).abs() > (b.y - a.y).abs();
                    let (va, vb) = if along_x { (a.x, b.x) } else { (a.y, b.y) };
                    let (lo, hi) = (va.min(vb), va.max(vb));
                    let at = |v: f64| from + ((v - va) / (vb - va)) as f32 * (to - from);
                    let mut v = (lo / minor).floor() * minor;
                    while v <= hi {
                        let t = at(v);
                        let is_major = ((v / step).round() * step - v).abs() < minor * 0.01;
                        let len = if is_major { RULER } else if ((v / (step / 2.0)).round() * (step / 2.0) - v).abs() < minor * 0.01 { RULER * 0.45 } else { RULER * 0.25 };
                        let (p0, p1) = if horizontal { (pos2(t, r.max.y - len), pos2(t, r.max.y)) } else { (pos2(r.max.x - len, t), pos2(r.max.x, t)) };
                        p.line_segment([p0, p1], Stroke::new(1.0, fg));
                        if is_major {
                            let label = format!("{}", (v * 100.0).round() / 100.0);
                            if horizontal {
                                p.text(pos2(t + 3.0, r.min.y + 1.0), Align2::LEFT_TOP, label, FontId::proportional(10.0), fg);
                            } else {
                                // Digits stacked, like most editors (no rotated text in egui).
                                let s: String = label.chars().flat_map(|c| [c, '\n']).collect();
                                p.text(pos2(r.min.x + 3.0, t + 2.0), Align2::LEFT_TOP, s.trim_end(), FontId::proportional(9.0), fg);
                            }
                        }
                        v += minor;
                    }
                    // The pointer's position.
                    if let Some(ptr) = pointer.filter(|q| rect.contains(*q)) {
                        let t = if horizontal { ptr.x } else { ptr.y };
                        let (p0, p1) = if horizontal { (pos2(t, r.min.y), pos2(t, r.max.y)) } else { (pos2(r.min.x, t), pos2(r.max.x, t)) };
                        p.line_segment([p0, p1], Stroke::new(1.0, style.visuals.selection.bg_fill));
                    }
                }
                resp
            })
            .inner;
        if resp.hovered() {
            ctx.set_cursor_icon(if horizontal { egui::CursorIcon::ResizeVertical } else { egui::CursorIcon::ResizeHorizontal });
        }
        if resp.drag_started() && aligned && drag.is_none() {
            // The top ruler gives a horizontal guide (on screen); find out which
            // document axis that is.
            let a = map.to_doc(r.center());
            let b = map.to_doc(r.center() + vec2(10.0, 0.0));
            let screen_horizontal_is_doc_x = (b.x - a.x).abs() > (b.y - a.y).abs();
            let vertical = if horizontal { !screen_horizontal_is_doc_x } else { screen_horizontal_is_doc_x };
            *drag = Some(GuideDrag { index: None, vertical });
        }
    }
    egui::Area::new(egui::Id::unique("ruler corner"))
        .fixed_pos(corner.min)
        .order(egui::Order::Middle)
        .show(ctx, |ui| {
            let (r, _) = ui.allocate_exact_size(corner.size(), Sense::hover());
            ui.painter().rect_filled(r, 0.0, bg);
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_steps() {
        assert_eq!(nice_step(7.0), 10.0);
        assert_eq!(nice_step(13.0), 20.0);
        assert_eq!(nice_step(40.0), 50.0);
        assert_eq!(nice_step(0.3), 0.5);
        assert_eq!(nice_step(100.0), 100.0);
    }
}
