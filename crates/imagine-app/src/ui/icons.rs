//! Vector icons painted with egui shapes (no icon font needed).

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, Vec2, vec2};

use super::panels::ToolId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Tool(ToolId),
    Eye,
    EyeClosed,
    Folder,
    Plus,
    Mask,
    Duplicate,
    Trash,
    Adjust,
}

fn dashed_poly(p: &Painter, pts: &[Pos2], stroke: Stroke) {
    let mut closed = pts.to_vec();
    closed.push(pts[0]);
    p.extend(Shape::dashed_line(&closed, stroke, 3.0, 2.0));
}

fn circle_pts(c: Pos2, r: Vec2, n: usize) -> Vec<Pos2> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            c + vec2(r.x * t.cos(), r.y * t.sin())
        })
        .collect()
}

/// Paints `icon` into `rect` (square) in `color`.
pub fn paint(p: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let s = rect.width().min(rect.height());
    let c = rect.center();
    let u = s / 20.0; // design grid: 20 × 20
    let at = |x: f32, y: f32| c + vec2((x - 10.0) * u, (y - 10.0) * u);
    let thin = Stroke::new(1.3 * u.max(0.8), color);
    let mid = Stroke::new(1.8 * u.max(0.8), color);
    match icon {
        Icon::Tool(ToolId::Move) => {
            p.line_segment([at(10.0, 2.5), at(10.0, 17.5)], mid);
            p.line_segment([at(2.5, 10.0), at(17.5, 10.0)], mid);
            for (tip, a, b) in [
                ((10.0, 2.0), (7.5, 4.8), (12.5, 4.8)),
                ((10.0, 18.0), (7.5, 15.2), (12.5, 15.2)),
                ((2.0, 10.0), (4.8, 7.5), (4.8, 12.5)),
                ((18.0, 10.0), (15.2, 7.5), (15.2, 12.5)),
            ] {
                p.add(Shape::convex_polygon(
                    vec![at(tip.0, tip.1), at(a.0, a.1), at(b.0, b.1)],
                    color,
                    Stroke::NONE,
                ));
            }
        }
        Icon::Tool(ToolId::Marquee) => {
            dashed_poly(
                p,
                &[at(3.0, 4.0), at(17.0, 4.0), at(17.0, 16.0), at(3.0, 16.0)],
                mid,
            );
        }
        Icon::Tool(ToolId::Ellipse) => {
            dashed_poly(
                p,
                &circle_pts(at(10.0, 10.0), vec2(7.5 * u, 6.0 * u), 28),
                mid,
            );
        }
        Icon::Tool(ToolId::Lasso) => {
            let mut loop_pts = circle_pts(at(11.0, 8.0), vec2(6.5 * u, 4.5 * u), 24);
            loop_pts.push(loop_pts[0]);
            p.add(Shape::line(loop_pts, mid));
            p.add(Shape::line(
                vec![at(6.5, 11.0), at(6.0, 14.5), at(8.5, 16.0), at(7.0, 18.5)],
                mid,
            ));
        }
        Icon::Tool(ToolId::ObjectSelect) => {
            // A dashed frame around a figure, with a spark (AI).
            for (a, b) in [((2.5, 4.5), (6.0, 4.5)), ((9.0, 4.5), (12.0, 4.5)), ((2.5, 4.5), (2.5, 8.0)), ((2.5, 11.0), (2.5, 14.5)),
                ((2.5, 17.5), (6.0, 17.5)), ((9.0, 17.5), (12.5, 17.5)), ((15.5, 17.5), (17.5, 17.5)), ((17.5, 14.5), (17.5, 11.0))] {
                p.line_segment([at(a.0, a.1), at(b.0, b.1)], thin);
            }
            p.circle_filled(at(10.0, 8.5), 2.0 * u, color);
            p.add(Shape::convex_polygon(vec![at(6.5, 16.0), at(7.5, 12.0), at(12.5, 12.0), at(13.5, 16.0)], color, Stroke::NONE));
            let star = at(16.0, 4.5);
            for k in 0..4 {
                let a = k as f32 * std::f32::consts::FRAC_PI_4 * 2.0;
                let d = vec2(a.cos(), a.sin()) * 2.8 * u;
                p.line_segment([star - d, star + d], thin);
            }
        }
        Icon::Tool(ToolId::Wand) => {
            p.line_segment([at(3.0, 17.0), at(12.0, 8.0)], Stroke::new(2.4 * u, color));
            let star = at(14.5, 5.5);
            for k in 0..4 {
                let a = k as f32 * std::f32::consts::FRAC_PI_4 * 2.0;
                let d = vec2(a.cos(), a.sin()) * 4.0 * u;
                p.line_segment([star - d, star + d], thin);
            }
        }
        Icon::Tool(ToolId::Crop) => {
            p.add(Shape::line(
                vec![at(6.0, 2.0), at(6.0, 14.0), at(18.0, 14.0)],
                mid,
            ));
            p.add(Shape::line(
                vec![at(2.0, 6.0), at(14.0, 6.0), at(14.0, 18.0)],
                mid,
            ));
        }
        Icon::Tool(ToolId::Brush) => {
            p.line_segment([at(17.0, 3.0), at(9.0, 11.0)], Stroke::new(2.2 * u, color));
            p.add(Shape::convex_polygon(
                vec![
                    at(8.0, 10.0),
                    at(10.0, 12.0),
                    at(8.0, 15.5),
                    at(3.0, 17.0),
                    at(4.5, 12.0),
                ],
                color,
                Stroke::NONE,
            ));
        }
        Icon::Tool(ToolId::Smudge) => {
            p.add(Shape::line(
                vec![at(3.0, 15.0), at(7.0, 12.5), at(11.0, 12.0), at(15.0, 9.0)],
                Stroke::new(3.0 * u, color.gamma_multiply(0.5)),
            ));
            p.circle_filled(at(15.5, 7.0), 2.6 * u, color);
            p.line_segment([at(15.5, 7.0), at(17.5, 2.5)], Stroke::new(2.6 * u, color));
        }
        Icon::Tool(ToolId::Clone) => {
            // A stamp: handle, neck and base.
            p.circle_filled(at(10.0, 4.5), 2.8 * u, color);
            p.add(Shape::convex_polygon(
                vec![at(8.5, 7.0), at(11.5, 7.0), at(12.5, 11.5), at(7.5, 11.5)],
                color,
                Stroke::NONE,
            ));
            p.add(Shape::convex_polygon(
                vec![at(3.5, 12.0), at(16.5, 12.0), at(16.5, 15.0), at(3.5, 15.0)],
                color,
                Stroke::NONE,
            ));
            p.line_segment([at(4.0, 17.5), at(16.0, 17.5)], thin);
        }
        Icon::Tool(ToolId::Remove) => {
            // A painted-over figure, dashed (going away), with a spark (AI).
            dashed_poly(p, &circle_pts(at(8.0, 7.0), vec2(2.4 * u, 2.4 * u), 10), thin);
            dashed_poly(p, &[at(4.0, 18.0), at(5.0, 12.0), at(11.0, 12.0), at(12.0, 18.0)], thin);
            p.line_segment([at(3.0, 15.5), at(13.5, 9.5)], Stroke::new(3.2 * u, color.gamma_multiply(0.6)));
            let star = at(16.0, 4.5);
            for k in 0..4 {
                let a = k as f32 * std::f32::consts::FRAC_PI_4 * 2.0;
                let d = vec2(a.cos(), a.sin()) * 2.8 * u;
                p.line_segment([star - d, star + d], thin);
            }
        }
        Icon::Tool(ToolId::SpotHeal) => {
            // A plaster with a dotted spot next to it.
            let band = [at(6.0, 15.0), at(15.0, 6.0), at(18.0, 9.0), at(9.0, 18.0)];
            p.add(Shape::convex_polygon(
                band.to_vec(),
                color.gamma_multiply(0.55),
                Stroke::NONE,
            ));
            p.add(Shape::convex_polygon(
                vec![at(9.5, 11.5), at(11.5, 9.5), at(14.5, 12.5), at(12.5, 14.5)],
                color,
                Stroke::NONE,
            ));
            dashed_poly(
                p,
                &circle_pts(at(5.5, 5.5), vec2(3.2 * u, 3.2 * u), 10),
                thin,
            );
        }
        Icon::Tool(ToolId::Heal) => {
            // A plaster: rotated band with a pad.
            let band = [at(4.0, 13.0), at(13.0, 4.0), at(16.0, 7.0), at(7.0, 16.0)];
            p.add(Shape::convex_polygon(
                band.to_vec(),
                color.gamma_multiply(0.55),
                Stroke::NONE,
            ));
            p.add(Shape::convex_polygon(
                vec![at(7.5, 9.5), at(9.5, 7.5), at(12.5, 10.5), at(10.5, 12.5)],
                color,
                Stroke::NONE,
            ));
        }
        Icon::Tool(ToolId::Eraser) => {
            let pts = [at(3.0, 13.0), at(10.0, 6.0), at(16.0, 12.0), at(9.0, 19.0)];
            p.add(Shape::convex_polygon(
                pts.to_vec(),
                Color32::TRANSPARENT,
                mid,
            ));
            p.add(Shape::convex_polygon(
                vec![pts[0], at(6.5, 9.5), at(12.5, 15.5), pts[3]],
                color,
                Stroke::NONE,
            ));
            p.line_segment([at(9.0, 19.0), at(18.0, 19.0)], thin);
        }
        Icon::Tool(ToolId::Text) => {
            p.line_segment([at(4.0, 4.0), at(16.0, 4.0)], Stroke::new(2.4 * u, color));
            p.line_segment([at(10.0, 4.0), at(10.0, 17.0)], Stroke::new(2.4 * u, color));
            p.line_segment([at(7.5, 17.0), at(12.5, 17.0)], Stroke::new(1.6 * u, color));
        }
        Icon::Tool(ToolId::Eyedropper) => {
            // A pipette: bulb top right, glass tube down to the tip.
            p.circle_filled(at(14.5, 5.5), 2.6 * u, color);
            p.line_segment([at(12.5, 7.5), at(5.0, 15.0)], Stroke::new(2.2 * u, color));
            p.line_segment([at(5.0, 15.0), at(3.5, 16.5)], Stroke::new(1.2 * u, color));
            p.line_segment([at(11.0, 4.5), at(15.5, 9.0)], Stroke::new(1.4 * u, color));
        }
        Icon::Tool(ToolId::Pen) => {
            // A fountain-pen nib.
            p.add(Shape::convex_polygon(
                vec![at(10.0, 3.0), at(15.0, 10.0), at(10.0, 18.0), at(5.0, 10.0)],
                color,
                Stroke::NONE,
            ));
            p.circle_filled(at(10.0, 10.5), 1.4 * u, Color32::from_gray(30));
            p.line_segment(
                [at(10.0, 12.0), at(10.0, 17.0)],
                Stroke::new(0.9 * u, Color32::from_gray(30)),
            );
        }
        Icon::Tool(ToolId::Shape) => {
            p.rect_stroke(
                Rect::from_min_max(at(3.0, 3.0), at(12.0, 12.0)),
                1.0,
                Stroke::new(1.6 * u, color),
                egui::StrokeKind::Middle,
            );
            p.circle_filled(at(13.0, 13.0), 4.5 * u, color);
        }
        Icon::Tool(ToolId::Gradient) => {
            let r = Rect::from_min_max(at(3.0, 4.0), at(17.0, 16.0));
            let mut mesh = egui::Mesh::default();
            let dark = color.gamma_multiply(0.1);
            mesh.colored_vertex(r.left_top(), color);
            mesh.colored_vertex(r.right_top(), dark);
            mesh.colored_vertex(r.right_bottom(), dark);
            mesh.colored_vertex(r.left_bottom(), color);
            mesh.add_triangle(0, 1, 2);
            mesh.add_triangle(0, 2, 3);
            p.add(Shape::mesh(mesh));
            p.rect_stroke(r, 1.0, thin, egui::StrokeKind::Inside);
        }
        Icon::Tool(ToolId::Radial) => {
            for (r, a) in [(7.5, 0.35), (5.0, 0.65), (2.5, 1.0)] {
                p.circle_filled(at(10.0, 10.0), r * u, color.gamma_multiply(a));
            }
        }
        Icon::Eye => {
            let mut lid = Vec::new();
            for i in 0..=12 {
                let t = i as f32 / 12.0;
                lid.push(at(
                    2.0 + 16.0 * t,
                    10.0 - 6.0 * (t * std::f32::consts::PI).sin(),
                ));
            }
            for i in (0..=12).rev() {
                let t = i as f32 / 12.0;
                lid.push(at(
                    2.0 + 16.0 * t,
                    10.0 + 6.0 * (t * std::f32::consts::PI).sin(),
                ));
            }
            p.add(Shape::closed_line(lid, thin));
            p.circle_filled(at(10.0, 10.0), 2.8 * u, color);
        }
        Icon::EyeClosed => {
            let pts: Vec<Pos2> = (0..=12)
                .map(|i| i as f32 / 12.0)
                .map(|t| at(2.0 + 16.0 * t, 9.0 + 4.0 * (t * std::f32::consts::PI).sin()))
                .collect();
            p.add(Shape::line(pts, thin));
        }
        Icon::Folder => {
            p.add(Shape::closed_line(
                vec![
                    at(2.0, 5.0),
                    at(8.0, 5.0),
                    at(10.0, 7.0),
                    at(18.0, 7.0),
                    at(18.0, 16.0),
                    at(2.0, 16.0),
                ],
                thin,
            ));
        }
        Icon::Plus => {
            p.line_segment([at(10.0, 3.0), at(10.0, 17.0)], mid);
            p.line_segment([at(3.0, 10.0), at(17.0, 10.0)], mid);
        }
        Icon::Mask => {
            p.rect_stroke(
                Rect::from_min_max(at(2.5, 4.0), at(17.5, 16.0)),
                1.5,
                thin,
                egui::StrokeKind::Inside,
            );
            p.circle_filled(at(10.0, 10.0), 3.8 * u, color);
        }
        Icon::Duplicate => {
            p.rect_stroke(
                Rect::from_min_max(at(2.5, 6.0), at(13.0, 17.0)),
                1.5,
                thin,
                egui::StrokeKind::Inside,
            );
            p.add(Shape::line(
                vec![
                    at(7.0, 6.0),
                    at(7.0, 2.5),
                    at(17.5, 2.5),
                    at(17.5, 13.0),
                    at(13.0, 13.0),
                ],
                thin,
            ));
        }
        Icon::Adjust => {
            let c0 = at(10.0, 10.0);
            p.circle_stroke(c0, 7.0 * u, thin);
            let half: Vec<Pos2> = (0..=16)
                .map(|i| {
                    let a = -std::f32::consts::FRAC_PI_2 + i as f32 / 16.0 * std::f32::consts::PI;
                    c0 + vec2(a.cos(), a.sin()) * 7.0 * u
                })
                .collect();
            p.add(Shape::convex_polygon(half, color, Stroke::NONE));
        }
        Icon::Trash => {
            p.line_segment([at(3.0, 5.0), at(17.0, 5.0)], thin);
            p.add(Shape::line(
                vec![at(7.5, 5.0), at(8.0, 2.5), at(12.0, 2.5), at(12.5, 5.0)],
                thin,
            ));
            p.add(Shape::closed_line(
                vec![at(4.5, 5.0), at(15.5, 5.0), at(14.5, 18.0), at(5.5, 18.0)],
                thin,
            ));
            p.line_segment([at(8.5, 8.0), at(8.5, 15.0)], thin);
            p.line_segment([at(11.5, 8.0), at(11.5, 15.0)], thin);
        }
    }
}

/// A clickable square icon button.
pub fn button(ui: &mut egui::Ui, icon: Icon, size: f32, selected: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::click());
    let visuals = ui.style().interact_selectable(&resp, selected);
    if selected || resp.hovered() {
        ui.painter().rect(
            rect,
            4.0,
            visuals.weak_bg_fill,
            visuals.bg_stroke,
            egui::StrokeKind::Inside,
        );
    }
    let fg = if selected {
        Color32::WHITE
    } else {
        visuals.fg_stroke.color
    };
    paint(ui.painter(), rect.shrink(size * 0.2), icon, fg);
    resp
}
