//! Canvas view transform: pan, zoom and rotation.
//!
//! Screen space is in physical pixels with the origin at the top-left and y
//! pointing down. Document space is in document pixels.
//!
//! `screen = R(rotation) · (doc − center) · zoom + viewport / 2`

use glam::{DMat2, DMat3, DVec2, Mat3};

pub const MIN_ZOOM: f64 = 1.0 / 256.0;
pub const MAX_ZOOM: f64 = 256.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    /// Document point shown at the centre of the viewport.
    pub center: DVec2,
    /// Screen pixels per document pixel.
    pub zoom: f64,
    /// Rotation in radians (positive = clockwise on screen).
    pub rotation: f64,
    /// Viewport size in physical pixels.
    pub viewport: DVec2,
}

impl Default for View {
    fn default() -> Self {
        Self {
            center: DVec2::ZERO,
            zoom: 1.0,
            rotation: 0.0,
            viewport: DVec2::new(1.0, 1.0),
        }
    }
}

impl View {
    fn rot(&self) -> DMat2 {
        DMat2::from_angle(self.rotation)
    }

    fn half_viewport(&self) -> DVec2 {
        self.viewport * 0.5
    }

    pub fn doc_to_screen(&self, doc: DVec2) -> DVec2 {
        self.rot() * ((doc - self.center) * self.zoom) + self.half_viewport()
    }

    pub fn screen_to_doc(&self, screen: DVec2) -> DVec2 {
        self.rot().transpose() * (screen - self.half_viewport()) / self.zoom + self.center
    }

    /// Re-centres so that `doc` lands on `screen` with the current zoom/rotation.
    fn pin(&mut self, doc: DVec2, screen: DVec2) {
        self.center = doc - self.rot().transpose() * (screen - self.half_viewport()) / self.zoom;
    }

    pub fn pan(&mut self, screen_delta: DVec2) {
        self.center -= self.rot().transpose() * screen_delta / self.zoom;
    }

    /// Multiplies the zoom by `factor`, keeping the document point under `anchor` fixed.
    pub fn zoom_at(&mut self, anchor: DVec2, factor: f64) {
        let doc = self.screen_to_doc(anchor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.pin(doc, anchor);
    }

    /// Sets an absolute zoom, keeping the document point under `anchor` fixed.
    pub fn set_zoom_at(&mut self, anchor: DVec2, zoom: f64) {
        self.zoom_at(anchor, zoom / self.zoom);
    }

    /// Rotates by `delta` radians around the screen point `anchor`.
    pub fn rotate_at(&mut self, anchor: DVec2, delta: f64) {
        let doc = self.screen_to_doc(anchor);
        self.rotation = (self.rotation + delta).rem_euclid(std::f64::consts::TAU);
        self.pin(doc, anchor);
    }

    /// Fits the whole document into the viewport and resets rotation.
    pub fn fit(&mut self, doc_w: u32, doc_h: u32) {
        let doc = DVec2::new(doc_w.max(1) as f64, doc_h.max(1) as f64);
        let z = (self.viewport / doc).min_element() * 0.92;
        self.zoom = z.clamp(MIN_ZOOM, MAX_ZOOM);
        self.rotation = 0.0;
        self.center = doc * 0.5;
    }

    /// Fits the document into a sub-rectangle of the viewport (screen px),
    /// e.g. the area left free by panels.
    pub fn fit_in(&mut self, doc_w: u32, doc_h: u32, min: DVec2, max: DVec2) {
        let doc = DVec2::new(doc_w.max(1) as f64, doc_h.max(1) as f64);
        let area = (max - min).max(DVec2::ONE);
        self.zoom = ((area / doc).min_element() * 0.92).clamp(MIN_ZOOM, MAX_ZOOM);
        self.rotation = 0.0;
        // The document centre goes to the centre of the area.
        let offset = (min + max) * 0.5 - self.half_viewport();
        self.center = doc * 0.5 - offset / self.zoom;
    }

    /// Whether the view is axis aligned (rotation is a multiple of 90°).
    pub fn is_axis_aligned(&self) -> bool {
        let q = self.rotation / std::f64::consts::FRAC_PI_2;
        (q - q.round()).abs() < 1e-6
    }

    /// Doc → clip-space matrix for the GPU (column-major, acts on (x, y, 1)).
    pub fn doc_to_clip(&self) -> Mat3 {
        let vp = self.viewport.max(DVec2::ONE);
        let to_clip = DMat3::from_cols_array(&[
            2.0 / vp.x,
            0.0,
            0.0, //
            0.0,
            -2.0 / vp.y,
            0.0, //
            -1.0,
            1.0,
            1.0,
        ]);
        let m = to_clip
            * DMat3::from_translation(self.half_viewport())
            * DMat3::from_mat2(self.rot())
            * DMat3::from_scale(DVec2::splat(self.zoom))
            * DMat3::from_translation(-self.center);
        m.as_mat3()
    }

    /// Axis-aligned bounding box (min, max) of the visible area in document space.
    pub fn visible_doc_rect(&self) -> (DVec2, DVec2) {
        let corners = [
            DVec2::ZERO,
            DVec2::new(self.viewport.x, 0.0),
            DVec2::new(0.0, self.viewport.y),
            self.viewport,
        ]
        .map(|c| self.screen_to_doc(c));
        let min = corners.iter().copied().reduce(DVec2::min).unwrap();
        let max = corners.iter().copied().reduce(DVec2::max).unwrap();
        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_in_centres_the_document_in_the_area() {
        let mut v = View {
            viewport: DVec2::new(1000.0, 800.0),
            ..Default::default()
        };
        v.fit_in(400, 400, DVec2::new(100.0, 0.0), DVec2::new(700.0, 800.0));
        let c = v.doc_to_screen(DVec2::new(200.0, 200.0));
        assert!((c - DVec2::new(400.0, 400.0)).length() < 1e-9, "{c}");
        assert!((v.zoom - 600.0 / 400.0 * 0.92).abs() < 1e-9);
    }

    fn view() -> View {
        View {
            center: DVec2::new(500.0, 300.0),
            zoom: 1.7,
            rotation: 0.6,
            viewport: DVec2::new(1920.0, 1080.0),
        }
    }

    fn close(a: DVec2, b: DVec2) -> bool {
        (a - b).length() < 1e-6
    }

    #[test]
    fn roundtrip() {
        let v = view();
        let p = DVec2::new(123.4, -56.7);
        assert!(close(v.screen_to_doc(v.doc_to_screen(p)), p));
    }

    #[test]
    fn center_maps_to_viewport_center() {
        let v = view();
        assert!(close(v.doc_to_screen(v.center), v.viewport * 0.5));
    }

    #[test]
    fn zoom_and_rotate_keep_anchor_fixed() {
        let mut v = view();
        let anchor = DVec2::new(300.0, 900.0);
        let doc = v.screen_to_doc(anchor);
        v.zoom_at(anchor, 3.0);
        assert!(close(v.doc_to_screen(doc), anchor));
        v.rotate_at(anchor, 1.1);
        assert!(close(v.doc_to_screen(doc), anchor));
    }

    #[test]
    fn pan_moves_content_with_cursor() {
        let mut v = view();
        let p = DVec2::new(10.0, 20.0);
        let before = v.doc_to_screen(p);
        v.pan(DVec2::new(5.0, -7.0));
        assert!(close(v.doc_to_screen(p), before + DVec2::new(5.0, -7.0)));
    }

    #[test]
    fn clip_matrix_matches_screen_transform() {
        let v = view();
        let m = v.doc_to_clip().as_dmat3();
        let p = DVec2::new(77.0, 412.0);
        let s = v.doc_to_screen(p);
        let c = m * p.extend(1.0);
        let expected = DVec2::new(
            s.x / v.viewport.x * 2.0 - 1.0,
            1.0 - s.y / v.viewport.y * 2.0,
        );
        assert!((c.truncate() - expected).length() < 1e-4);
    }

    #[test]
    fn fit_shows_whole_document() {
        let mut v = view();
        v.fit(8000, 2000);
        let (min, max) = v.visible_doc_rect();
        assert!(min.x <= 0.0 && min.y <= 0.0 && max.x >= 8000.0 && max.y >= 2000.0);
    }
}
