//! Warp (Edit › Transform › Warp): a bicubic Bézier patch over a box. Its
//! 4 × 4 control points start evenly spaced (no change); moving them bends
//! the content. Rendering needs the inverse: for a point on the canvas,
//! where in the box it comes from.

use glam::DVec2;

/// 4 × 4 control points, row-major (row = v, column = u).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Patch {
    pub min: DVec2,
    pub max: DVec2,
    pub pts: [[DVec2; 4]; 4],
}

fn bern(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t]
}

fn dbern(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [-3.0 * s * s, 3.0 * s * s - 6.0 * s * t, 6.0 * s * t - 3.0 * t * t, 3.0 * t * t]
}

impl Patch {
    /// The unchanged patch over the box.
    pub fn identity(min: DVec2, max: DVec2) -> Self {
        let mut pts = [[DVec2::ZERO; 4]; 4];
        for (j, row) in pts.iter_mut().enumerate() {
            for (i, p) in row.iter_mut().enumerate() {
                *p = min + (max - min) * DVec2::new(i as f64 / 3.0, j as f64 / 3.0);
            }
        }
        Self { min, max, pts }
    }

    /// Where (u, v) ∈ [0, 1]² of the box ends up.
    pub fn eval(&self, u: f64, v: f64) -> DVec2 {
        let (bu, bv) = (bern(u), bern(v));
        let mut p = DVec2::ZERO;
        for j in 0..4 {
            for i in 0..4 {
                p += self.pts[j][i] * (bu[i] * bv[j]);
            }
        }
        p
    }

    fn jacobian(&self, u: f64, v: f64) -> (DVec2, DVec2) {
        let (bu, bv, du, dv) = (bern(u), bern(v), dbern(u), dbern(v));
        let (mut pu, mut pv) = (DVec2::ZERO, DVec2::ZERO);
        for j in 0..4 {
            for i in 0..4 {
                pu += self.pts[j][i] * (du[i] * bv[j]);
                pv += self.pts[j][i] * (bu[i] * dv[j]);
            }
        }
        (pu, pv)
    }

    /// (u, v) landing on `x`, also a little beyond the patch (the surface
    /// continued past its edges, up to a quarter of it); None if not found.
    pub fn inverse_uv(&self, x: DVec2, guess: Option<DVec2>) -> Option<DVec2> {
        let size = (self.max - self.min).max(DVec2::splat(1e-9));
        let mut uv = guess.unwrap_or((x - self.min) / size).clamp(DVec2::splat(-0.25), DVec2::splat(1.25));
        for _ in 0..40 {
            let f = self.eval(uv.x, uv.y) - x;
            if f.length_squared() < 1e-6 {
                return Some(uv);
            }
            let (pu, pv) = self.jacobian(uv.x, uv.y);
            let det = pu.x * pv.y - pu.y * pv.x;
            if det.abs() < 1e-12 {
                return None;
            }
            let step = DVec2::new((f.x * pv.y - f.y * pv.x) / det, (pu.x * f.y - pu.y * f.x) / det);
            uv = (uv - step).clamp(DVec2::splat(-0.25), DVec2::splat(1.25));
        }
        ((self.eval(uv.x, uv.y) - x).length() < 0.25).then_some(uv)
    }

    /// The point of the original box that lands on `x` (None: outside the
    /// warped patch). `guess`: a nearby solution in (u, v), if known.
    pub fn inverse(&self, x: DVec2, guess: Option<DVec2>) -> Option<DVec2> {
        let size = (self.max - self.min).max(DVec2::splat(1e-9));
        let mut uv = guess.unwrap_or((x - self.min) / size).clamp(DVec2::ZERO, DVec2::ONE);
        for _ in 0..30 {
            let f = self.eval(uv.x, uv.y) - x;
            if f.length_squared() < 1e-6 {
                break;
            }
            let (pu, pv) = self.jacobian(uv.x, uv.y);
            let det = pu.x * pv.y - pu.y * pv.x;
            if det.abs() < 1e-12 {
                return None;
            }
            let step = DVec2::new((f.x * pv.y - f.y * pv.x) / det, (pu.x * f.y - pu.y * f.x) / det);
            // Damped, kept a little beyond the patch so the edge converges.
            uv = (uv - step).clamp(DVec2::splat(-0.25), DVec2::splat(1.25));
        }
        if (self.eval(uv.x, uv.y) - x).length() > 0.25 {
            return None;
        }
        let eps = 1e-6;
        (uv.x >= -eps && uv.x <= 1.0 + eps && uv.y >= -eps && uv.y <= 1.0 + eps).then(|| self.min + uv.clamp(DVec2::ZERO, DVec2::ONE) * size)
    }

    /// Outline and inner grid lines, as polylines (for drawing the warp).
    pub fn grid_lines(&self, steps: usize) -> Vec<Vec<DVec2>> {
        let mut out = Vec::new();
        for k in 0..=3 {
            let t = k as f64 / 3.0;
            out.push((0..=steps).map(|s| self.eval(s as f64 / steps as f64, t)).collect());
            out.push((0..=steps).map(|s| self.eval(t, s as f64 / steps as f64)).collect());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_inverse() {
        let p = Patch::identity(DVec2::new(10.0, 20.0), DVec2::new(110.0, 70.0));
        assert!((p.eval(0.5, 0.5) - DVec2::new(60.0, 45.0)).length() < 1e-9);
        let x = DVec2::new(37.0, 51.0);
        assert!((p.inverse(x, None).unwrap() - x).length() < 1e-6);
        // Bend: the middle top points pulled up 20 px.
        let mut b = p;
        b.pts[0][1].y -= 20.0;
        b.pts[0][2].y -= 20.0;
        let top = b.eval(0.5, 0.0);
        assert!(top.y < 20.0 - 10.0, "bulges up: {top}");
        // A point on the bent top edge comes from the original top edge.
        let src = b.inverse(top, None).unwrap();
        assert!((src - DVec2::new(60.0, 20.0)).length() < 1e-3, "{src}");
        assert!(b.inverse(DVec2::new(60.0, 200.0), None).is_none(), "outside");
    }
}
