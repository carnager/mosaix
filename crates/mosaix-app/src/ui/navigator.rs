//! The Navigator (the whole picture small, the visible part framed, click or
//! drag to look elsewhere, zoom) and Info (histogram, colour and position
//! under the pointer) panels.

use egui::{Color32, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2};
use glam::DVec2;
use mosaix_core::tr;

use super::panels::{Action, Model, State};

/// Side of the navigator texture in pixels.
pub const NAV: u32 = 256;

/// The view used to draw the navigator texture: the document fitted.
pub fn nav_view(doc: (u32, u32)) -> mosaix_core::View {
    let (w, h) = (doc.0.max(1) as f64, doc.1.max(1) as f64);
    mosaix_core::View {
        center: DVec2::new(w * 0.5, h * 0.5),
        zoom: (NAV as f64 / w).min(NAV as f64 / h),
        rotation: 0.0,
        viewport: DVec2::splat(NAV as f64),
    }
}

/// Data the two panels need from the app.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PanelData {
    pub navigator: Option<egui::TextureId>,
    pub histogram: Option<std::sync::Arc<mosaix_gpu::Histogram>>,
    /// Straight sRGB + alpha under the pointer.
    pub color: Option<[f32; 4]>,
}

pub fn navigator_tab(ui: &mut egui::Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    st.panels_shown.navigator = true;
    let (dw, dh) = (m.doc_size.0.max(1) as f32, m.doc_size.1.max(1) as f32);
    let avail = ui.available_width().min(320.0);
    let k = (avail / dw).min(avail / dh);
    let size = vec2(dw * k, dh * k);
    let (rect, resp) = ui.allocate_exact_size(vec2(avail, size.y), Sense::click_and_drag());
    let img = Rect::from_center_size(rect.center(), size);
    let p = ui.painter_at(rect.expand(1.0));
    match m.panels.navigator {
        Some(tex) => {
            // The document's part of the square texture.
            let z = nav_view(m.doc_size).zoom as f32;
            let (uw, uh) = (dw * z / NAV as f32, dh * z / NAV as f32);
            let uv = Rect::from_center_size(pos2(0.5, 0.5), vec2(uw, uh));
            p.image(tex, img, uv, Color32::WHITE);
        }
        None => {
            p.rect_filled(img, 0.0, ui.visuals().extreme_bg_color);
        }
    }
    p.rect_stroke(img, 0.0, Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color), egui::StrokeKind::Outside);
    let doc_to_nav = |d: DVec2| pos2(img.min.x + d.x as f32 * k, img.min.y + d.y as f32 * k);
    // The visible part of the canvas.
    if let Some(canvas) = st.canvas {
        let ppp = ui.ctx().pixels_per_point() as f64;
        let corners = [canvas.min, canvas.right_top(), canvas.max, canvas.left_bottom()]
            .map(|c| doc_to_nav(m.view.screen_to_doc(DVec2::new(c.x as f64, c.y as f64) * ppp)));
        let mut pts = corners.to_vec();
        pts.push(corners[0]);
        p.add(Shape::line(pts, Stroke::new(1.5, Color32::from_rgb(230, 70, 60))));
    }
    if (resp.clicked() || resp.dragged())
        && let Some(pos) = resp.interact_pointer_pos()
    {
        let d = ((pos - img.min) / k).to_pos2();
        out.push(Action::ViewCenter(d.x as f64, d.y as f64));
    }
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let mut pct = (m.zoom * 100.0) as f32;
        if ui.small_button("−").clicked() {
            out.push(Action::ViewZoom(m.zoom / 1.25));
        }
        let r = ui.add(egui::Slider::new(&mut pct, 1.0..=3200.0).logarithmic(true).suffix(" %").max_decimals(0));
        if r.changed() {
            out.push(Action::ViewZoom(pct as f64 / 100.0));
        }
        if ui.small_button("+").clicked() {
            out.push(Action::ViewZoom(m.zoom * 1.25));
        }
    });
}

pub fn info_tab(ui: &mut egui::Ui, m: &Model, st: &mut State) {
    st.panels_shown.info = true;
    let w = ui.available_width().min(320.0);
    let (rect, _) = ui.allocate_exact_size(vec2(w, 90.0), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    if let Some(h) = &m.panels.histogram {
        let x_at = |i: usize| rect.min.x + (i as f32 + 0.5) / 256.0 * rect.width();
        let curve = |ch: usize| -> Vec<Pos2> {
            let peak = h.peak(ch).max(1e-6);
            (0..256).map(|i| pos2(x_at(i), rect.max.y - (h.bins[ch][i] / peak).min(1.0) * (rect.height() - 4.0))).collect()
        };
        // Luminance-ish (all channels) filled, the channels as lines.
        let all = curve(0);
        for (i, q) in all.iter().enumerate() {
            let _ = i;
            p.line_segment([*q, pos2(q.x, rect.max.y)], Stroke::new(rect.width() / 256.0 + 0.3, Color32::from_gray(110)));
        }
        for (ch, c) in [(1, Color32::from_rgb(230, 80, 70)), (2, Color32::from_rgb(80, 200, 100)), (3, Color32::from_rgb(90, 140, 250))] {
            p.add(Shape::line(curve(ch), Stroke::new(1.0, c.gamma_multiply(0.8))));
        }
    }
    ui.add_space(6.0);
    egui::Grid::new("info").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        ui.label(RichText::new("X / Y").weak());
        ui.label(match m.cursor_doc {
            Some((x, y)) => format!("{:.0} / {:.0} px", x.floor(), y.floor()),
            None => "–".into(),
        });
        ui.end_row();
        ui.label(RichText::new(tr("Color")).weak());
        match m.panels.color.filter(|_| m.cursor_doc.is_some()) {
            Some(c) => {
                let b = c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                    ui.painter().rect_filled(r, 2.0, Color32::from_rgb(b[0], b[1], b[2]));
                    ui.label(RichText::new(format!("#{:02x}{:02x}{:02x}", b[0], b[1], b[2])).monospace());
                });
                ui.end_row();
                ui.label(RichText::new("R G B A").weak());
                ui.label(RichText::new(format!("{} {} {} {}", b[0], b[1], b[2], b[3])).monospace());
            }
            None => {
                ui.label("–");
            }
        }
        ui.end_row();
        ui.label(RichText::new(tr("Document")).weak());
        ui.label(format!("{} × {} px", m.doc_size.0, m.doc_size.1));
        ui.end_row();
    });
}
