//! An editable box with handles, used by free transform (move / scale /
//! rotate an affine transform) and by the crop tool (axis-aligned rectangle).
//! Pure geometry: all positions in document pixels unless noted.

use glam::{DAffine2, DVec2};
use imagine_core::View;
use imagine_gpu::OverlayLine;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Handle {
    Move,
    Rotate,
    /// Side per axis: -1 = min edge, 1 = max edge, 0 = this axis is not dragged.
    Scale {
        sx: i8,
        sy: i8,
    },
    /// Draw a new rectangle (crop: click outside the box).
    Create,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Modifiers {
    /// Photoshop: toggles proportional scaling, snaps rotation to 15°, constrains moves.
    pub shift: bool,
    /// Scale around the centre.
    pub alt: bool,
}

#[derive(Debug, Clone)]
struct Drag {
    handle: Handle,
    start: DVec2,
    start_fwd: DAffine2,
}

#[derive(Debug, Clone)]
pub struct BoxEdit {
    /// The untransformed box (min, max).
    pub base: (DVec2, DVec2),
    /// Current transform of the box (and of the content inside it).
    pub fwd: DAffine2,
    /// Free transform: rotation allowed, corners scale proportionally by default.
    /// Crop: axis-aligned, free aspect ratio, clicking outside draws a new box.
    pub free: bool,
    /// Transform the content already had when the box was opened (vector
    /// layers keep their rotation): the box starts there, `delta` is what
    /// gets applied.
    pub origin: DAffine2,
    drag: Option<Drag>,
}

const HANDLE_PX: f64 = 5.0;
const HIT_PX: f64 = 12.0;

impl BoxEdit {
    pub fn new(min: DVec2, max: DVec2, free: bool) -> Self {
        Self {
            base: (min, max),
            fwd: DAffine2::IDENTITY,
            free,
            origin: DAffine2::IDENTITY,
            drag: None,
        }
    }

    /// Starts the box in the content's own frame (e.g. a rotated text):
    /// `min`/`max` are then the untransformed bounds.
    pub fn with_origin(mut self, m: DAffine2) -> Self {
        self.fwd = m;
        self.origin = m;
        self
    }

    /// The change made in this box (what to apply to the content).
    pub fn delta(&self) -> DAffine2 {
        self.fwd * self.origin.inverse()
    }

    fn local_point(&self, sx: i8, sy: i8) -> DVec2 {
        let (a, b) = self.base;
        let pick = |s: i8, lo: f64, hi: f64| match s {
            -1 => lo,
            1 => hi,
            _ => (lo + hi) * 0.5,
        };
        DVec2::new(pick(sx, a.x, b.x), pick(sy, a.y, b.y))
    }

    /// Centre of the transformed box (document px).
    pub fn centre(&self) -> DVec2 {
        self.fwd.transform_point2(self.local_point(0, 0))
    }

    /// Current rotation in degrees (−180 … 180).
    pub fn angle(&self) -> f64 {
        let x = self.fwd.matrix2.x_axis;
        let a = x.y.atan2(x.x).to_degrees();
        // Mirrored boxes: report the angle of the unmirrored axis.
        if self.fwd.matrix2.determinant() < 0.0 {
            -a
        } else {
            a
        }
    }

    /// Turns the box (and its content) about its centre.
    pub fn rotate_by(&mut self, degrees: f64) {
        let c = self.centre();
        let r = DAffine2::from_angle(degrees.to_radians());
        self.fwd = DAffine2::from_translation(c) * r * DAffine2::from_translation(-c) * self.fwd;
    }

    /// Mirrors the box about its centre (horizontally and/or vertically).
    pub fn flip(&mut self, x: bool, y: bool) {
        let c = self.centre();
        let f = DAffine2::from_scale(DVec2::new(
            if x { -1.0 } else { 1.0 },
            if y { -1.0 } else { 1.0 },
        ));
        self.fwd = DAffine2::from_translation(c) * f * DAffine2::from_translation(-c) * self.fwd;
    }

    /// Corners in document space, clockwise from the top left.
    pub fn corners(&self) -> [DVec2; 4] {
        [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            .map(|(x, y)| self.fwd.transform_point2(self.local_point(x, y)))
    }

    /// The axis-aligned rectangle (crop), rounded to whole pixels: (x, y, w, h).
    pub fn rect(&self) -> (i64, i64, i64, i64) {
        let c = self.corners();
        let min = c.iter().fold(DVec2::INFINITY, |a, p| a.min(*p)).round();
        let max = c.iter().fold(DVec2::NEG_INFINITY, |a, p| a.max(*p)).round();
        (
            min.x as i64,
            min.y as i64,
            (max.x - min.x) as i64,
            (max.y - min.y) as i64,
        )
    }

    fn handles(&self) -> Vec<(Handle, DVec2)> {
        let mut v = Vec::new();
        for sy in -1..=1i8 {
            for sx in -1..=1i8 {
                if sx != 0 || sy != 0 {
                    v.push((
                        Handle::Scale { sx, sy },
                        self.fwd.transform_point2(self.local_point(sx, sy)),
                    ));
                }
            }
        }
        v
    }

    fn contains(&self, doc: DVec2) -> bool {
        let inv = self.fwd.inverse();
        let u = inv.transform_point2(doc);
        let (a, b) = self.base;
        u.x >= a.x.min(b.x) && u.x <= a.x.max(b.x) && u.y >= a.y.min(b.y) && u.y <= a.y.max(b.y)
    }

    /// What a press at `screen` (physical px) would grab.
    pub fn hit(&self, view: &View, screen: DVec2, scale: f64) -> Handle {
        let best = self
            .handles()
            .into_iter()
            .map(|(h, p)| (h, view.doc_to_screen(p).distance(screen)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((h, d)) = best
            && d <= HIT_PX * scale
        {
            return h;
        }
        if self.contains(view.screen_to_doc(screen)) {
            Handle::Move
        } else if self.free {
            Handle::Rotate
        } else {
            Handle::Create
        }
    }

    pub fn begin(&mut self, handle: Handle, doc: DVec2) {
        if handle == Handle::Create {
            self.base = (doc, doc);
            self.fwd = DAffine2::IDENTITY;
        }
        self.drag = Some(Drag {
            handle,
            start: doc,
            start_fwd: self.fwd,
        });
    }

    pub fn update(&mut self, doc: DVec2, m: Modifiers) {
        let Some(d) = self.drag.clone() else { return };
        self.fwd = match d.handle {
            Handle::Create => {
                let mut end = doc;
                if m.shift {
                    // Square.
                    let delta = doc - d.start;
                    let s = delta.x.abs().max(delta.y.abs());
                    end = d.start + DVec2::new(s.copysign(delta.x), s.copysign(delta.y));
                }
                self.base = (d.start.min(end), d.start.max(end));
                DAffine2::IDENTITY
            }
            Handle::Move => {
                let mut delta = doc - d.start;
                if m.shift {
                    if delta.x.abs() > delta.y.abs() {
                        delta.y = 0.0;
                    } else {
                        delta.x = 0.0;
                    }
                }
                DAffine2::from_translation(delta) * d.start_fwd
            }
            Handle::Rotate => {
                let c = d.start_fwd.transform_point2(self.local_point(0, 0));
                let (a, b) = (d.start - c, doc - c);
                let mut angle = b.y.atan2(b.x) - a.y.atan2(a.x);
                if m.shift {
                    let step = 15f64.to_radians();
                    angle = (angle / step).round() * step;
                }
                DAffine2::from_translation(c)
                    * DAffine2::from_angle(angle)
                    * DAffine2::from_translation(-c)
                    * d.start_fwd
            }
            Handle::Scale { sx, sy } => {
                let u = d.start_fwd.inverse().transform_point2(doc);
                let h = self.local_point(sx, sy);
                let anchor = if m.alt {
                    self.local_point(0, 0)
                } else {
                    self.local_point(-sx, -sy)
                };
                let ratio = |num: f64, den: f64| if den.abs() < 1e-9 { 1.0 } else { num / den };
                let mut k = DVec2::new(
                    if sx != 0 {
                        ratio(u.x - anchor.x, h.x - anchor.x)
                    } else {
                        1.0
                    },
                    if sy != 0 {
                        ratio(u.y - anchor.y, h.y - anchor.y)
                    } else {
                        1.0
                    },
                );
                // Corners keep the aspect ratio in free transform; Shift toggles.
                let proportional = sx != 0 && sy != 0 && (self.free != m.shift);
                if proportional {
                    let dir = h - anchor;
                    let s = (u - anchor).dot(dir) / dir.length_squared().max(1e-9);
                    k = DVec2::splat(s);
                }
                let k = DVec2::new(nonzero(k.x), nonzero(k.y));
                d.start_fwd
                    * DAffine2::from_translation(anchor)
                    * DAffine2::from_scale(k)
                    * DAffine2::from_translation(-anchor)
            }
        };
    }

    pub fn end(&mut self) {
        self.drag = None;
    }

    /// Outline and handles in screen px.
    pub fn lines(&self, view: &View, scale: f64) -> Vec<OverlayLine> {
        let mut out = vec![OverlayLine {
            points: self.corners().map(|p| view.doc_to_screen(p)).to_vec(),
            closed: true,
        }];
        let r = HANDLE_PX * scale;
        for (_, p) in self.handles() {
            let s = view.doc_to_screen(p);
            let points = [(-r, -r), (r, -r), (r, r), (-r, r)]
                .map(|(x, y)| s + DVec2::new(x, y))
                .to_vec();
            out.push(OverlayLine {
                points,
                closed: true,
            });
        }
        out
    }
}

fn nonzero(v: f64) -> f64 {
    if v.abs() < 1e-3 {
        1e-3f64.copysign(v)
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_frame_scales_along_the_rotated_axes() {
        // A box already turned by 30°: dragging the right edge stretches
        // along that axis and the angle stays.
        let m = DAffine2::from_angle(30f64.to_radians());
        let mut b = BoxEdit::new(DVec2::ZERO, DVec2::new(100.0, 50.0), true).with_origin(m);
        assert!((b.angle() - 30.0).abs() < 1e-9);
        assert_eq!(b.delta(), DAffine2::IDENTITY);
        let edge = m.transform_point2(DVec2::new(100.0, 25.0));
        b.begin(Handle::Scale { sx: 1, sy: 0 }, edge);
        b.update(
            m.transform_point2(DVec2::new(200.0, 25.0)),
            Modifiers::default(),
        );
        assert!((b.angle() - 30.0).abs() < 1e-9, "{}", b.angle());
        // The delta doubles the width along the rotated x axis.
        let d = b.delta();
        let p = d.transform_point2(m.transform_point2(DVec2::new(100.0, 0.0)));
        let want = m.transform_point2(DVec2::new(200.0, 0.0));
        assert!(p.distance(want) < 1e-6, "{p} vs {want}");
    }

    const NONE: Modifiers = Modifiers {
        shift: false,
        alt: false,
    };

    fn edit() -> BoxEdit {
        BoxEdit::new(DVec2::new(100.0, 100.0), DVec2::new(300.0, 200.0), true)
    }

    #[test]
    fn corner_scales_proportionally_from_opposite_corner() {
        let mut b = edit();
        b.begin(Handle::Scale { sx: 1, sy: 1 }, DVec2::new(300.0, 200.0));
        b.update(DVec2::new(500.0, 230.0), NONE);
        let c = b.corners();
        assert!(
            c[0].distance(DVec2::new(100.0, 100.0)) < 1e-9,
            "anchor stays"
        );
        // Projection onto the diagonal: (400, 130)·(200, 100) / |(200, 100)|² = 1.86
        let k = (400.0 * 200.0 + 130.0 * 100.0) / (200.0f64 * 200.0 + 100.0 * 100.0);
        assert!(c[2].distance(DVec2::new(100.0 + 200.0 * k, 100.0 + 100.0 * k)) < 1e-9);
        // Shift: free aspect ratio.
        b.update(
            DVec2::new(500.0, 230.0),
            Modifiers {
                shift: true,
                alt: false,
            },
        );
        assert!(b.corners()[2].distance(DVec2::new(500.0, 230.0)) < 1e-9);
    }

    #[test]
    fn edge_scales_one_axis_and_alt_scales_around_centre() {
        let mut b = edit();
        b.begin(Handle::Scale { sx: 1, sy: 0 }, DVec2::new(300.0, 150.0));
        b.update(
            DVec2::new(400.0, 999.0),
            Modifiers {
                shift: false,
                alt: true,
            },
        );
        let c = b.corners();
        assert!(c[0].distance(DVec2::new(0.0, 100.0)) < 1e-9, "{c:?}");
        assert!(c[2].distance(DVec2::new(400.0, 200.0)) < 1e-9, "{c:?}");
    }

    #[test]
    fn rotate_snaps_with_shift_and_move_constrains() {
        let mut b = edit();
        let centre = DVec2::new(200.0, 150.0);
        b.begin(Handle::Rotate, centre + DVec2::new(200.0, 0.0));
        b.update(
            centre + DVec2::new(200.0, 210.0),
            Modifiers {
                shift: true,
                alt: false,
            },
        );
        // atan(210/200) ≈ 46.4° → 45°.
        let angle = b
            .fwd
            .matrix2
            .x_axis
            .y
            .atan2(b.fwd.matrix2.x_axis.x)
            .to_degrees();
        assert!((angle - 45.0).abs() < 1e-9, "{angle}");
        assert!(
            b.fwd.transform_point2(centre).distance(centre) < 1e-9,
            "rotates around the centre"
        );
        b.end();

        let mut b = edit();
        b.begin(Handle::Move, DVec2::new(150.0, 150.0));
        b.update(
            DVec2::new(250.0, 160.0),
            Modifiers {
                shift: true,
                alt: false,
            },
        );
        assert!(b.fwd.translation.distance(DVec2::new(100.0, 0.0)) < 1e-9);
    }

    #[test]
    fn crop_box_is_created_and_hit_tested() {
        let mut b = BoxEdit::new(DVec2::ZERO, DVec2::new(1000.0, 800.0), false);
        let view = View {
            viewport: DVec2::new(1000.0, 800.0),
            center: DVec2::new(500.0, 400.0),
            ..Default::default()
        };
        assert_eq!(
            b.hit(&view, DVec2::new(1000.0, 800.0), 1.0),
            Handle::Scale { sx: 1, sy: 1 }
        );
        assert_eq!(b.hit(&view, DVec2::new(500.0, 400.0), 1.0), Handle::Move);
        assert_eq!(b.hit(&view, DVec2::new(1200.0, 400.0), 1.0), Handle::Create);
        b.begin(Handle::Create, DVec2::new(300.0, 200.0));
        b.update(DVec2::new(100.0, 50.0), NONE);
        b.end();
        assert_eq!(b.rect(), (100, 50, 200, 150));
        // Crop corners scale freely.
        b.begin(Handle::Scale { sx: 1, sy: 1 }, DVec2::new(300.0, 200.0));
        b.update(DVec2::new(400.0, 210.0), NONE);
        assert_eq!(b.rect(), (100, 50, 300, 160));
    }
}
