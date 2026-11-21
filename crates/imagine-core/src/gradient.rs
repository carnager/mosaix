//! Multi-stop gradients (as in Photoshop's gradient editor) and the
//! geometry of the gradient styles, shared by gradient fill layers and the
//! gradient overlay effect.

use serde::{Deserialize, Serialize};

use crate::color::{linear_to_srgb, srgb_to_linear};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientStyle {
    #[default]
    Linear,
    Radial,
    /// Sweeps around the centre.
    Angle,
    /// Linear, mirrored at the centre.
    Reflected,
    Diamond,
}

impl GradientStyle {
    pub const ALL: [GradientStyle; 5] = [Self::Linear, Self::Radial, Self::Angle, Self::Reflected, Self::Diamond];

    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => crate::tr("Linear"),
            Self::Radial => crate::tr("Radial"),
            Self::Angle => crate::i18n::tr_ctx("gradient", "Angle"),
            Self::Reflected => crate::tr("Reflected"),
            Self::Diamond => crate::tr("Diamond"),
        }
    }
}

fn half() -> f32 {
    0.5
}

fn is_half(v: &f32) -> bool {
    *v == 0.5
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorStop {
    /// Position, 0–1.
    pub at: f32,
    /// Linear RGB.
    pub color: [f32; 3],
    /// Where between this stop and the next both colours mix half and half
    /// (0–1 of the way, 0.5 = in the middle).
    #[serde(default = "half", skip_serializing_if = "is_half")]
    pub mid: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlphaStop {
    pub at: f32,
    pub opacity: f32,
    #[serde(default = "half", skip_serializing_if = "is_half")]
    pub mid: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gradient {
    pub colors: Vec<ColorStop>,
    /// Opacity stops; none = opaque.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alphas: Vec<AlphaStop>,
    /// Photoshop's smoothness (0–1): eases the transitions at the stops.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub smoothness: f32,
}

fn is_zero(v: &f32) -> bool {
    *v == 0.0
}

impl Default for Gradient {
    fn default() -> Self {
        Self::two([0.0; 3], [1.0; 3])
    }
}

/// `u` (0–1 between two stops) bent so that `mid` maps to ½.
fn bend(u: f32, mid: f32) -> f32 {
    let m = mid.clamp(0.01, 0.99);
    if u < m { 0.5 * u / m } else { 0.5 + 0.5 * (u - m) / (1.0 - m) }
}

/// The segment of `stops` (sorted by position) around `t`: indices and the
/// bent fraction between them.
fn segment(pos: impl Iterator<Item = (f32, f32)> + Clone, t: f32) -> Option<(usize, usize, f32)> {
    let n = pos.clone().count();
    if n == 0 {
        return None;
    }
    let mut prev: Option<(usize, f32, f32)> = None;
    for (i, (at, mid)) in pos.enumerate() {
        if t <= at {
            return Some(match prev {
                None => (i, i, 0.0),
                Some((j, pa, pm)) => {
                    let u = if at > pa { (t - pa) / (at - pa) } else { 1.0 };
                    (j, i, bend(u, pm))
                }
            });
        }
        prev = Some((i, at, mid));
    }
    Some((n - 1, n - 1, 0.0))
}

impl Gradient {
    /// From `start` to `end` (linear RGB), opaque.
    pub fn two(start: [f32; 3], end: [f32; 3]) -> Self {
        Self {
            colors: vec![
                ColorStop { at: 0.0, color: start, mid: 0.5 },
                ColorStop { at: 1.0, color: end, mid: 0.5 },
            ],
            alphas: Vec::new(),
            smoothness: 0.0,
        }
    }

    /// Stops sorted by position (as evaluation expects).
    pub fn sorted(mut self) -> Self {
        self.colors.sort_by(|a, b| a.at.total_cmp(&b.at));
        self.alphas.sort_by(|a, b| a.at.total_cmp(&b.at));
        self
    }

    /// Prepared for evaluating many positions.
    pub fn lut(&self) -> GradientLut {
        let g = self.clone().sorted();
        let colors: Vec<ColorStop> =
            g.colors.iter().map(|s| ColorStop { color: s.color.map(linear_to_srgb), ..*s }).collect();
        // Smoothness: halfway to smoothstep at 100 % (measured on Photoshop's output).
        let k = 0.5 * g.smoothness.clamp(0.0, 1.0);
        let ease = |u: f32| u + k * (u * u * (3.0 - 2.0 * u) - u);
        GradientLut {
            table: (0..LUT_SIZE)
                .map(|i| {
                    let t = i as f32 / (LUT_SIZE - 1) as f32;
                    let c = match segment(colors.iter().map(|s| (s.at, s.mid)), t) {
                        // Interpolated like Photoshop: in sRGB-encoded values.
                        Some((a, b, u)) => {
                            let u = ease(u);
                            let (a, b) = (colors[a].color, colors[b].color);
                            std::array::from_fn(|k| srgb_to_linear(a[k] + (b[k] - a[k]) * u))
                        }
                        None => [0.0; 3],
                    };
                    let alpha = match segment(g.alphas.iter().map(|s| (s.at, s.mid)), t) {
                        Some((a, b, u)) => {
                            let u = ease(u);
                            let (a, b) = (g.alphas[a].opacity, g.alphas[b].opacity);
                            a + (b - a) * u
                        }
                        None => 1.0,
                    };
                    [c[0], c[1], c[2], alpha]
                })
                .collect(),
        }
    }

    /// Straight linear RGB + alpha at `t` (0–1).
    pub fn eval(&self, t: f32) -> [f32; 4] {
        self.lut().at(t)
    }

    /// The first and last colour.
    pub fn ends(&self) -> ([f32; 3], [f32; 3]) {
        let g = self.clone().sorted();
        let f = |s: Option<&ColorStop>| s.map_or([0.0; 3], |s| s.color);
        (f(g.colors.first()), f(g.colors.last()))
    }
}

const LUT_SIZE: usize = 1024;

/// A gradient sampled into a table.
#[derive(Debug, Clone)]
pub struct GradientLut {
    table: Vec<[f32; 4]>,
}

impl GradientLut {
    /// Straight linear RGB + alpha at `t` (0–1, clamped).
    pub fn at(&self, t: f32) -> [f32; 4] {
        let f = t.clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
        let i = (f as usize).min(LUT_SIZE - 2);
        let u = f - i as f32;
        let (a, b) = (self.table[i], self.table[i + 1]);
        std::array::from_fn(|k| a[k] + (b[k] - a[k]) * u)
    }
}

/// Where a gradient runs: its style and direction across a reference box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientGeometry {
    pub style: GradientStyle,
    /// Degrees; 0 = left → right, 90 = bottom → top.
    pub angle: f32,
    /// Length relative to the box (1 = edge to edge).
    pub scale: f32,
    pub reverse: bool,
    /// Reference box `[x0, y0, x1, y1]`.
    pub bounds: [f32; 4],
    /// Shift of the centre in px.
    pub offset: [f32; 2],
}

/// A [`GradientGeometry`] prepared for evaluating many points.
#[derive(Debug, Clone, Copy)]
pub struct Positions {
    style: GradientStyle,
    reverse: bool,
    c: (f32, f32),
    dir: (f32, f32),
    half: f32,
    angle: f32,
}

impl GradientGeometry {
    pub fn positions(&self) -> Positions {
        let [x0, y0, x1, y1] = self.bounds;
        let (bw, bh) = ((x1 - x0).max(0.0), (y1 - y0).max(0.0));
        let scale = self.scale.max(0.01);
        let a = self.angle.to_radians();
        // Screen y points down: 90° runs from the bottom up.
        let dir = (a.cos(), -a.sin());
        // As Photoshop: from the centre to where the line at the angle
        // leaves the box (also the radius of the round styles).
        let reach = |extent: f32, d: f32| if d.abs() > 1e-6 { 0.5 * extent / d.abs() } else { f32::INFINITY };
        let half = reach(bw, dir.0).min(reach(bh, dir.1));
        Positions {
            style: self.style,
            reverse: self.reverse,
            c: ((x0 + x1) * 0.5 + self.offset[0], (y0 + y1) * 0.5 + self.offset[1]),
            dir,
            half: (half * scale).max(1e-3),
            angle: a,
        }
    }
}

impl Positions {
    /// The gradient position (0–1) at point `(x, y)`.
    pub fn at(&self, x: f32, y: f32) -> f32 {
        let (x, y) = (x - self.c.0, y - self.c.1);
        let p = x * self.dir.0 + y * self.dir.1;
        let t = match self.style {
            GradientStyle::Linear => 0.5 + p / (2.0 * self.half),
            GradientStyle::Radial => x.hypot(y) / self.half,
            GradientStyle::Reflected => p.abs() / self.half,
            GradientStyle::Diamond => {
                let q = -x * self.dir.1 + y * self.dir.0;
                (p.abs() + q.abs()) / self.half
            }
            GradientStyle::Angle => {
                // Clockwise on screen, from the direction of the angle.
                let t = self.angle - (-y).atan2(x);
                t.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU
            }
        }
        .clamp(0.0, 1.0);
        if self.reverse { 1.0 - t } else { t }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_and_midpoints() {
        let mut g = Gradient::two([0.0; 3], [1.0; 3]);
        assert_eq!(g.eval(0.0)[0], 0.0);
        assert!((g.eval(1.0)[0] - 1.0).abs() < 1e-5);
        // Halfway in sRGB values.
        assert!((linear_to_srgb(g.eval(0.5)[0]) - 0.5).abs() < 0.01);
        // A midpoint at ¼ reaches the half-way colour there.
        g.colors[0].mid = 0.25;
        assert!((linear_to_srgb(g.eval(0.25)[0]) - 0.5).abs() < 0.01);
        // Three stops, unsorted, with opacity.
        let g = Gradient {
            colors: vec![
                ColorStop { at: 1.0, color: [0.0, 0.0, 1.0], mid: 0.5 },
                ColorStop { at: 0.0, color: [1.0, 0.0, 0.0], mid: 0.5 },
                ColorStop { at: 0.5, color: [0.0, 1.0, 0.0], mid: 0.5 },
            ],
            alphas: vec![AlphaStop { at: 0.0, opacity: 1.0, mid: 0.5 }, AlphaStop { at: 1.0, opacity: 0.0, mid: 0.5 }],
            smoothness: 0.0,
        };
        let m = g.eval(0.5);
        assert!(m[1] > 0.99 && m[0] < 0.01 && (m[3] - 0.5).abs() < 0.01);
        assert!(g.eval(0.0)[0] > 0.99 && g.eval(1.0)[2] > 0.99);
    }

    #[test]
    fn styles() {
        let geo = |style| GradientGeometry { style, angle: 0.0, scale: 1.0, reverse: false, bounds: [0.0, 0.0, 100.0, 100.0], offset: [0.0; 2] }.positions();
        let l = geo(GradientStyle::Linear);
        assert!(l.at(0.0, 50.0) < 0.01 && l.at(100.0, 50.0) > 0.99);
        let r = geo(GradientStyle::Reflected);
        assert!(r.at(50.0, 50.0) < 0.01 && r.at(0.0, 50.0) > 0.99 && r.at(100.0, 50.0) > 0.99);
        let d = geo(GradientStyle::Diamond);
        assert!(d.at(50.0, 50.0) < 0.01);
        let a = geo(GradientStyle::Angle);
        assert!(a.at(60.0, 51.0) < a.at(50.0, 60.0));
    }
}
