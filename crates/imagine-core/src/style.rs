//! Layer styles (effects): stroke, shadows, glows, colour overlay, bevel &
//! emboss. Computed from the layer's alpha on a pixel region:
//! - exact Euclidean distance fields (Felzenszwalb & Huttenlocher) for
//!   strokes, glow spread and the bevel height map,
//! - box blurs (three passes ≈ Gaussian) for shadows and glows,
//! - a lit height field for bevel & emboss.
//!
//! Effects outside the shape (drop shadow, outer glow, outside stroke) come
//! back as separate layers to place under the layer with their own blend
//! mode; effects inside are baked into the layer's pixels.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::blend::BlendMode;

type Px = [f32; 4];

fn yes() -> bool {
    true
}

/// One effect on a layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Effect {
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(flatten)]
    pub kind: EffectKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrokePosition {
    #[default]
    Outside,
    Inside,
    Center,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BevelStyle {
    /// Raised inside the shape.
    #[default]
    Inner,
    /// Raised inside, sunk outside: the shape stands out of the surface.
    Emboss,
    /// The edge sinks into the surface (pillow).
    Pillow,
}

/// Colours are linear RGB; opacities 0…1; angles in degrees (light from
/// that direction, 0 = from the right, 90 = from the top); sizes in px.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "effect", rename_all = "snake_case")]
pub enum EffectKind {
    Stroke {
        size: f32,
        #[serde(default)]
        position: StrokePosition,
        color: [f32; 3],
        #[serde(default = "one")]
        opacity: f32,
    },
    DropShadow {
        color: [f32; 3],
        opacity: f32,
        angle: f32,
        distance: f32,
        size: f32,
        #[serde(default)]
        spread: f32,
    },
    InnerShadow {
        color: [f32; 3],
        opacity: f32,
        angle: f32,
        distance: f32,
        size: f32,
    },
    OuterGlow {
        color: [f32; 3],
        opacity: f32,
        size: f32,
        #[serde(default)]
        spread: f32,
    },
    InnerGlow {
        color: [f32; 3],
        opacity: f32,
        size: f32,
    },
    ColorOverlay {
        color: [f32; 3],
        opacity: f32,
    },
    Bevel {
        #[serde(default)]
        style: BevelStyle,
        /// Width of the slope, px.
        size: f32,
        /// Strength of the relief (1 = normal).
        #[serde(default = "one")]
        depth: f32,
        /// Extra smoothing of the slope, px.
        #[serde(default)]
        soften: f32,
        angle: f32,
        /// Height of the light above the surface (degrees).
        #[serde(default = "thirty")]
        altitude: f32,
        #[serde(default = "white")]
        highlight: [f32; 3],
        #[serde(default = "three_quarters")]
        highlight_opacity: f32,
        #[serde(default = "black")]
        shadow: [f32; 3],
        #[serde(default = "three_quarters")]
        shadow_opacity: f32,
    },
}

fn one() -> f32 {
    1.0
}
fn thirty() -> f32 {
    30.0
}
fn three_quarters() -> f32 {
    0.75
}
fn white() -> [f32; 3] {
    [1.0; 3]
}
fn black() -> [f32; 3] {
    [0.0; 3]
}

impl EffectKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Stroke { .. } => "Kontur",
            Self::DropShadow { .. } => "Schlagschatten",
            Self::InnerShadow { .. } => "Schatten nach innen",
            Self::OuterGlow { .. } => "Schein nach außen",
            Self::InnerGlow { .. } => "Schein nach innen",
            Self::ColorOverlay { .. } => "Farbüberlagerung",
            Self::Bevel { .. } => "Abgeflachte Kante & Relief",
        }
    }

    /// One of each, with sensible settings (for menus).
    pub fn defaults() -> Vec<EffectKind> {
        vec![
            Self::Stroke { size: 3.0, position: StrokePosition::Outside, color: [0.0; 3], opacity: 1.0 },
            Self::DropShadow { color: [0.0; 3], opacity: 0.6, angle: 120.0, distance: 6.0, size: 8.0, spread: 0.0 },
            Self::InnerShadow { color: [0.0; 3], opacity: 0.6, angle: 120.0, distance: 4.0, size: 6.0 },
            Self::OuterGlow { color: [1.0, 0.85, 0.4], opacity: 0.75, size: 12.0, spread: 0.0 },
            Self::InnerGlow { color: [1.0, 0.85, 0.4], opacity: 0.75, size: 8.0 },
            Self::ColorOverlay { color: [0.8, 0.1, 0.1], opacity: 1.0 },
            Self::Bevel {
                style: BevelStyle::Inner,
                size: 6.0,
                depth: 1.0,
                soften: 0.0,
                angle: 120.0,
                altitude: 30.0,
                highlight: white(),
                highlight_opacity: 0.75,
                shadow: black(),
                shadow_opacity: 0.75,
            },
        ]
    }

    /// How far the effect can reach outside the shape (px).
    pub fn reach(&self) -> f32 {
        match self {
            Self::Stroke { size, position: StrokePosition::Outside, .. } => *size + 1.0,
            Self::Stroke { size, position: StrokePosition::Center, .. } => *size * 0.5 + 1.0,
            Self::DropShadow { distance, size, spread, .. } => distance + size * 1.5 + spread + 2.0,
            Self::OuterGlow { size, spread, .. } => size * 1.5 + spread + 2.0,
            Self::Bevel { style: BevelStyle::Emboss | BevelStyle::Pillow, size, soften, .. } => size + soften * 1.5 + 2.0,
            _ => 0.0,
        }
    }
}

/// How far a list of effects reaches outside the shape (px).
pub fn reach(effects: &[Effect]) -> f32 {
    effects.iter().filter(|e| e.enabled).map(|e| e.kind.reach()).fold(0.0, f32::max)
}

/// A rectangle of premultiplied linear RGBA pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub w: usize,
    pub h: usize,
    pub px: Vec<Px>,
}

impl Region {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, px: vec![[0.0; 4]; w * h] }
    }

    fn alpha(&self) -> Vec<f32> {
        self.px.iter().map(|p| p[3]).collect()
    }
}

/// The result: layers to put underneath (bottom first) and the layer itself.
pub struct Styled {
    pub below: Vec<(BlendMode, Region)>,
    pub content: Region,
}

// ---- building blocks ----------------------------------------------------------

/// 1D squared-distance transform (Felzenszwalb & Huttenlocher).
fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    let mut k = 0;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        loop {
            let p = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * (q as f32 - p as f32));
            // z[0] = −∞, so k never drops below 0.
            if s <= z[k] {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }
    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        *dq = (q as f32 - p as f32).powi(2) + f[p];
    }
}

/// Euclidean distance from every pixel to the nearest `seed` pixel.
#[cfg_attr(not(test), allow(dead_code))]
fn distance(w: usize, h: usize, seed: &[bool]) -> Vec<f32> {
    const INF: f32 = 1e20;
    let mut g: Vec<f32> = seed.iter().map(|&s| if s { 0.0 } else { INF }).collect();
    // Columns.
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let f: Vec<f32> = (0..h).map(|y| g[y * w + x]).collect();
            let mut d = vec![0.0; h];
            let (mut v, mut z) = (vec![0usize; h], vec![0.0f32; h + 1]);
            edt_1d(&f, &mut d, &mut v, &mut z);
            d
        })
        .collect();
    for (x, c) in cols.iter().enumerate() {
        for y in 0..h {
            g[y * w + x] = c[y];
        }
    }
    // Rows.
    g.par_chunks_mut(w).for_each(|row| {
        let f = row.to_vec();
        let (mut v, mut z) = (vec![0usize; w], vec![0.0f32; w + 1]);
        edt_1d(&f, row, &mut v, &mut z);
    });
    g.par_iter_mut().for_each(|d| *d = d.sqrt());
    g
}

/// Signed distance to the shape edge: negative inside, positive outside
/// (half-pixel accurate at antialiased edges).
fn signed_distance(w: usize, h: usize, alpha: &[f32]) -> Vec<f32> {
    // Edge points with sub-pixel position: in every pixel on the boundary
    // the 0.5-contour of the alpha lies (a − 0.5) from the centre, against
    // the alpha gradient. Every pixel then takes the distance to the nearest
    // edge point (two-pass propagation of the point, Danielsson-style), so
    // antialiased edges stay smooth instead of following pixel steps.
    let at = |x: isize, y: isize| -> f32 {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize { 0.0 } else { alpha[y as usize * w + x as usize].clamp(0.0, 1.0) }
    };
    const NONE: (f32, f32) = (f32::NAN, f32::NAN);
    let mut seed: Vec<(f32, f32)> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            let a = at(x, y);
            let inside = a >= 0.5;
            let partial = a > 0.02 && a < 0.98;
            // Hard edge (no antialiasing): full pixel next to an empty one.
            let hard = !partial
                && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
                    let b = at(x + dx, y + dy);
                    if inside { b <= 0.02 } else { b >= 0.98 }
                });
            let boundary = partial || hard;
            if !boundary {
                return NONE;
            }
            // Gradient (Sobel) points into the shape.
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            let len = (gx * gx + gy * gy).sqrt();
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            if len < 1e-6 {
                return (cx, cy);
            }
            let off = (a - 0.5).clamp(-0.5, 0.5);
            (cx - gx / len * off, cy - gy / len * off)
        })
        .collect();
    let d2 = |i: usize, s: (f32, f32)| {
        let (px, py) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
        (px - s.0).powi(2) + (py - s.1).powi(2)
    };
    let mut best: Vec<f32> = seed.iter().enumerate().map(|(i, s)| if s.0.is_nan() { f32::INFINITY } else { d2(i, *s) }).collect();
    let mut relax = |i: usize, j: usize, seed: &mut Vec<(f32, f32)>, best: &mut Vec<f32>| {
        let s = seed[j];
        if s.0.is_nan() {
            return;
        }
        let d = d2(i, s);
        if d < best[i] {
            best[i] = d;
            seed[i] = s;
        }
    };
    // Forward: from left/top neighbours; backward: from right/bottom.
    for _ in 0..2 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x > 0 {
                    relax(i, i - 1, &mut seed, &mut best);
                }
                if y > 0 {
                    relax(i, i - w, &mut seed, &mut best);
                    if x > 0 {
                        relax(i, i - w - 1, &mut seed, &mut best);
                    }
                    if x + 1 < w {
                        relax(i, i - w + 1, &mut seed, &mut best);
                    }
                }
            }
            for x in (0..w.saturating_sub(1)).rev() {
                let i = y * w + x;
                relax(i, i + 1, &mut seed, &mut best);
            }
        }
        for y in (0..h).rev() {
            for x in (0..w).rev() {
                let i = y * w + x;
                if x + 1 < w {
                    relax(i, i + 1, &mut seed, &mut best);
                }
                if y + 1 < h {
                    relax(i, i + w, &mut seed, &mut best);
                    if x + 1 < w {
                        relax(i, i + w + 1, &mut seed, &mut best);
                    }
                    if x > 0 {
                        relax(i, i + w - 1, &mut seed, &mut best);
                    }
                }
            }
            for x in 1..w {
                let i = y * w + x;
                relax(i, i - 1, &mut seed, &mut best);
            }
        }
    }
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let d = best[i].sqrt();
            let d = if d.is_finite() { d } else { 1e6 };
            if alpha[i] >= 0.5 { -d } else { d }
        })
        .collect()
}

/// One horizontal + vertical box blur pass of radius r.
fn box_blur(w: usize, h: usize, src: &[f32], r: usize) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    let norm = 1.0 / (2 * r + 1) as f32;
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &src[y * w..(y + 1) * w];
        let get = |i: isize| if i < 0 || i >= w as isize { 0.0 } else { row[i as usize] };
        let mut acc: f32 = (-(r as isize)..=r as isize).map(get).sum();
        for (x, o) in out.iter_mut().enumerate() {
            *o = acc * norm;
            acc += get(x as isize + r as isize + 1) - get(x as isize - r as isize);
        }
    });
    let mut out = vec![0.0f32; w * h];
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let get = |i: isize| if i < 0 || i >= h as isize { 0.0 } else { tmp[i as usize * w + x] };
            let mut acc: f32 = (-(r as isize)..=r as isize).map(get).sum();
            (0..h)
                .map(|y| {
                    let v = acc * norm;
                    acc += get(y as isize + r as isize + 1) - get(y as isize - r as isize);
                    v
                })
                .collect()
        })
        .collect();
    for (x, c) in cols.iter().enumerate() {
        for (y, v) in c.iter().enumerate() {
            out[y * w + x] = *v;
        }
    }
    out
}

/// ≈ Gaussian blur where `size` is the visible spread (≈ 2σ), three box passes.
fn blur(w: usize, h: usize, src: &[f32], size: f32) -> Vec<f32> {
    let sigma = (size * 0.5).max(0.0);
    if sigma < 0.3 {
        return src.to_vec();
    }
    // Box radius for three passes with the same variance.
    let r = (((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) * 0.5).round().max(1.0) as usize;
    let a = box_blur(w, h, src, r);
    let b = box_blur(w, h, &a, r);
    box_blur(w, h, &b, r)
}

/// Shifts a field by (dx, dy) px (nearest), zero outside.
fn shift(w: usize, h: usize, src: &[f32], dx: f32, dy: f32, fill: f32) -> Vec<f32> {
    let (sx, sy) = (dx.round() as isize, dy.round() as isize);
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i % w) as isize - sx, (i / w) as isize - sy);
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize { fill } else { src[y as usize * w + x as usize] }
        })
        .collect()
}

/// Direction the light comes from → offset of a shadow (away from the light).
fn away(angle: f32, distance: f32) -> (f32, f32) {
    let a = angle.to_radians();
    // Screen y points down: light from the top (90°) throws shadows downwards.
    (-a.cos() * distance, a.sin() * distance)
}

fn solid(alpha: &[f32], color: [f32; 3], opacity: f32, w: usize, h: usize) -> Region {
    Region { w, h, px: alpha.iter().map(|&a| a.clamp(0.0, 1.0) * opacity).map(|a| [color[0] * a, color[1] * a, color[2] * a, a]).collect() }
}

/// `src` over `dst` with a blend mode, both premultiplied linear (simple
/// per-pixel version of the compositor, for effects inside the shape).
fn blend_into(dst: &mut [Px], src_rgb: [f32; 3], src_alpha: &[f32], mode: BlendMode) {
    dst.par_iter_mut().zip(src_alpha.par_iter()).for_each(|(d, &sa)| {
        let sa = sa.clamp(0.0, 1.0);
        if sa <= 0.0 {
            return;
        }
        let s = [src_rgb[0] * sa, src_rgb[1] * sa, src_rgb[2] * sa, sa];
        *d = crate::blend::composite(*d, s, mode, crate::blend::BlendSpace::Srgb);
    });
}

// ---- the effects ----------------------------------------------------------------

/// Applies `effects` to `content` (premultiplied linear). The region must
/// already include `reach(effects)` of transparent margin.
pub fn apply(effects: &[Effect], content: &Region) -> Styled {
    let (w, h) = (content.w, content.h);
    let alpha = content.alpha();
    let active: Vec<&EffectKind> = effects.iter().filter(|e| e.enabled).map(|e| &e.kind).collect();
    let needs_sdf = active.iter().any(|k| match k {
        EffectKind::Stroke { .. } | EffectKind::Bevel { .. } => true,
        EffectKind::OuterGlow { spread, .. } | EffectKind::DropShadow { spread, .. } => *spread > 0.0,
        _ => false,
    });
    let sdf = needs_sdf.then(|| signed_distance(w, h, &alpha));
    // Shape grown by `spread` px (for shadows and glows).
    let grown = |spread: f32| -> Vec<f32> {
        match (&sdf, spread > 0.0) {
            (Some(d), true) => d.iter().zip(&alpha).map(|(&d, &a)| a.max((spread + 0.5 - d).clamp(0.0, 1.0))).collect(),
            _ => alpha.clone(),
        }
    };

    let mut below: Vec<(BlendMode, Region)> = Vec::new();
    let mut out = content.clone();
    // Underneath (in Photoshop's order, bottom first): drop shadow, outer glow, outside stroke.
    for k in &active {
        match **k {
            EffectKind::DropShadow { color, opacity, angle, distance, size, spread } => {
                let (dx, dy) = away(angle, distance);
                let a = blur(w, h, &shift(w, h, &grown(spread), dx, dy, 0.0), size);
                below.push((BlendMode::Multiply, solid(&a, color, opacity, w, h)));
            }
            EffectKind::OuterGlow { color, opacity, size, spread } => {
                let a = blur(w, h, &grown(spread), size);
                below.push((BlendMode::Screen, solid(&a, color, opacity, w, h)));
            }
            _ => {}
        }
    }
    // Inside the shape: overlay, glows, shadows, bevel, then inside/centre strokes on top.
    for k in &active {
        match **k {
            EffectKind::ColorOverlay { color, opacity } => {
                // Normal blend clipped to the shape: the alpha stays.
                out.px.par_iter_mut().for_each(|p| {
                    for c in 0..3 {
                        p[c] = p[c] * (1.0 - opacity) + color[c] * p[3] * opacity;
                    }
                });
            }
            EffectKind::InnerGlow { color, opacity, size } => {
                // Bright at the edge, fading inwards.
                let inv: Vec<f32> = alpha.iter().map(|a| 1.0 - a).collect();
                let b = blur(w, h, &inv, size * 2.0);
                let a: Vec<f32> = b.iter().zip(&alpha).map(|(g, a)| (g * 2.0).min(1.0) * a * opacity).collect();
                blend_into(&mut out.px, color, &a, BlendMode::Screen);
            }
            EffectKind::InnerShadow { color, opacity, angle, distance, size } => {
                let (dx, dy) = away(angle, distance);
                let inv: Vec<f32> = alpha.iter().map(|a| 1.0 - a).collect();
                let b = blur(w, h, &shift(w, h, &inv, dx, dy, 1.0), size);
                let a: Vec<f32> = b.iter().zip(&alpha).map(|(s, a)| s * a * opacity).collect();
                blend_into(&mut out.px, color, &a, BlendMode::Multiply);
            }
            _ => {}
        }
    }
    for k in &active {
        if let EffectKind::Bevel { style, size, depth, soften, angle, altitude, highlight, highlight_opacity, shadow, shadow_opacity } = **k {
            let d = sdf.as_ref().unwrap();
            let size = size.max(0.5);
            // Height: 0 at the edge rising to 1 at `size` inside (emboss: also falling outside).
            let mut height: Vec<f32> = d
                .iter()
                .map(|&d| {
                    let t = (-d / size).clamp(-1.0, 1.0);
                    let s = |t: f32| t * t * (3.0 - 2.0 * t);
                    match style {
                        BevelStyle::Inner => s(t.max(0.0)),
                        BevelStyle::Emboss => 0.5 + 0.5 * if t >= 0.0 { s(t) } else { -s(-t) },
                        BevelStyle::Pillow => 1.0 - s(t.abs()),
                    }
                })
                .collect();
            if soften > 0.0 {
                height = blur(w, h, &height, soften * 2.0);
            }
            let (az, alt) = (angle.to_radians(), altitude.to_radians());
            let light = [az.cos() * alt.cos(), -az.sin() * alt.cos(), alt.sin()];
            let flat = light[2];
            let k = depth * 2.0;
            let mut hi = vec![0.0f32; w * h];
            let mut lo = vec![0.0f32; w * h];
            let reach_out = !matches!(style, BevelStyle::Inner);
            hi.par_iter_mut().zip(lo.par_iter_mut()).enumerate().for_each(|(i, (hv, lv))| {
                let (x, y) = (i % w, i / w);
                let hx = |xx: usize| height[y * w + xx];
                let hy = |yy: usize| height[yy * w + x];
                let gx = (hx((x + 1).min(w - 1)) - hx(x.saturating_sub(1))) * 0.5;
                let gy = (hy((y + 1).min(h - 1)) - hy(y.saturating_sub(1))) * 0.5;
                let n = [-gx * k * size * 0.5, -gy * k * size * 0.5, 1.0];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                let s = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) / len;
                // Where the relief is visible: inside the shape, or around it for emboss.
                let cover = if reach_out { 1.0 } else { alpha[i] };
                *hv = ((s - flat).max(0.0) / (1.0 - flat).max(1e-3)).min(1.0) * cover;
                *lv = ((flat - s).max(0.0) / flat.max(1e-3)).min(1.0) * cover;
            });
            if reach_out {
                // Outside the shape the relief shades what is below: separate layers.
                let mut outer_hi = vec![0.0; w * h];
                let mut outer_lo = vec![0.0; w * h];
                for i in 0..w * h {
                    let o = 1.0 - alpha[i];
                    outer_hi[i] = hi[i] * o * highlight_opacity;
                    outer_lo[i] = lo[i] * o * shadow_opacity;
                    hi[i] *= alpha[i];
                    lo[i] *= alpha[i];
                }
                below.push((BlendMode::Multiply, solid(&outer_lo, shadow, 1.0, w, h)));
                below.push((BlendMode::Screen, solid(&outer_hi, highlight, 1.0, w, h)));
            }
            let hi: Vec<f32> = hi.iter().map(|v| v * highlight_opacity).collect();
            let lo: Vec<f32> = lo.iter().map(|v| v * shadow_opacity).collect();
            blend_into(&mut out.px, highlight, &hi, BlendMode::Screen);
            blend_into(&mut out.px, shadow, &lo, BlendMode::Multiply);
        }
    }
    for k in &active {
        if let EffectKind::Stroke { size, position, color, opacity } = **k {
            let d = sdf.as_ref().unwrap();
            let (inner, outer) = match position {
                StrokePosition::Outside => (0.0, size),
                StrokePosition::Inside => (-size, 0.0),
                StrokePosition::Center => (-size * 0.5, size * 0.5),
            };
            // Coverage of the band inner < d < outer (antialiased).
            let band: Vec<f32> = d.iter().map(|&d| ((outer - d + 0.5).clamp(0.0, 1.0) * (d - inner + 0.5).clamp(0.0, 1.0)) * opacity).collect();
            if position == StrokePosition::Outside {
                below.push((BlendMode::Normal, solid(&band, color, 1.0, w, h)));
            } else {
                // Inside the shape: on top of the content; the outer half of a centre stroke goes under it.
                let inside: Vec<f32> = band.iter().zip(&alpha).map(|(b, a)| b * a).collect();
                blend_into(&mut out.px, color, &inside, BlendMode::Normal);
                if position == StrokePosition::Center {
                    let outside: Vec<f32> = band.iter().zip(&alpha).map(|(b, a)| b * (1.0 - a)).collect();
                    below.push((BlendMode::Normal, solid(&outside, color, 1.0, w, h)));
                }
            }
        }
    }
    Styled { below, content: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A white filled disc of radius r in the middle of an n × n region.
    fn disc(n: usize, r: f32) -> Region {
        let c = n as f32 / 2.0;
        let mut reg = Region::new(n, n);
        for (i, p) in reg.px.iter_mut().enumerate() {
            let d = ((i % n) as f32 + 0.5 - c).hypot((i / n) as f32 + 0.5 - c);
            let a = (r - d + 0.5).clamp(0.0, 1.0);
            *p = [a, a, a, a];
        }
        reg
    }

    fn fx(kind: EffectKind) -> Vec<Effect> {
        vec![Effect { enabled: true, kind }]
    }

    #[test]
    fn distance_is_euclidean() {
        let (w, h) = (50, 40);
        let mut seed = vec![false; w * h];
        seed[10 * w + 10] = true;
        let d = distance(w, h, &seed);
        assert!((d[13 * w + 14] - 5.0).abs() < 1e-4);
        assert!((d[10 * w + 40] - 30.0).abs() < 1e-4);
    }

    #[test]
    fn outside_stroke_is_a_ring_of_the_right_width() {
        let reg = disc(100, 20.0);
        let s = apply(&fx(EffectKind::Stroke { size: 5.0, position: StrokePosition::Outside, color: [1.0, 0.0, 0.0], opacity: 1.0 }), &reg);
        assert_eq!(s.below.len(), 1);
        let ring = &s.below[0].1;
        let at = |x: usize| ring.px[50 * 100 + x];
        // Disc edge at x = 70; ring from 70 to 75.
        assert!(at(72)[0] > 0.99 && at(72)[3] > 0.99, "{:?}", at(72));
        assert!(at(77)[3] < 0.01, "{:?}", at(77));
        assert!(at(60)[3] < 0.01, "not inside the disc: {:?}", at(60));
        assert_eq!(s.content, reg, "the content stays");
    }

    #[test]
    fn drop_shadow_moves_away_from_the_light_and_glow_spreads() {
        let reg = disc(120, 20.0);
        // Light from the top-left (135°) → shadow to the bottom-right.
        let s = apply(&fx(EffectKind::DropShadow { color: [0.0; 3], opacity: 1.0, angle: 135.0, distance: 10.0, size: 4.0, spread: 0.0 }), &reg);
        let sh = &s.below[0];
        assert_eq!(sh.0, BlendMode::Multiply);
        let a = |x: usize, y: usize| sh.1.px[y * 120 + x][3];
        assert!(a(60 + 18, 60 + 18) > a(60 - 18, 60 - 18) + 0.3, "{} vs {}", a(78, 78), a(42, 42));
        let g = apply(&fx(EffectKind::OuterGlow { color: [1.0; 3], opacity: 1.0, size: 10.0, spread: 0.0 }), &reg);
        let glow = &g.below[0].1;
        assert!(glow.px[60 * 120 + 60 + 24][3] > 0.05, "glow outside the edge");
    }

    #[test]
    fn bevel_lights_one_side_and_darkens_the_other() {
        let reg = disc(100, 30.0);
        let s = apply(
            &fx(EffectKind::Bevel {
                style: BevelStyle::Inner,
                size: 8.0,
                depth: 1.0,
                soften: 0.0,
                angle: 135.0,
                altitude: 30.0,
                highlight: [1.0; 3],
                highlight_opacity: 1.0,
                shadow: [0.0; 3],
                shadow_opacity: 1.0,
            }),
            &disc(100, 30.0),
        );
        let lum = |x: usize, y: usize| s.content.px[y * 100 + x][1];
        // Grey content would be needed to see both; white: the shadow side darkens.
        let lit = lum(50 - 18, 50 - 18);
        let dark = lum(50 + 18, 50 + 18);
        assert!(dark < 0.6 && lit > dark + 0.3, "lit {lit}, dark {dark}");
        assert!((lum(50, 50) - 1.0).abs() < 0.05, "flat top unchanged");
        let _ = reg;
    }

    #[test]
    fn overlay_and_reach() {
        let reg = disc(60, 10.0);
        let s = apply(&fx(EffectKind::ColorOverlay { color: [0.0, 1.0, 0.0], opacity: 1.0 }), &reg);
        let p = s.content.px[30 * 60 + 30];
        assert!(p[1] > 0.99 && p[0] < 0.01 && (p[3] - 1.0).abs() < 1e-6);
        assert_eq!(s.content.px[0], [0.0; 4]);
        let all: Vec<Effect> = EffectKind::defaults().into_iter().map(|kind| Effect { enabled: true, kind }).collect();
        assert!(reach(&all) > 15.0);
        let json = serde_json::to_string(&all).unwrap();
        assert_eq!(serde_json::from_str::<Vec<Effect>>(&json).unwrap(), all);
    }

    #[test]
    fn stroke_edge_is_smooth_not_stepped() {
        // Outer edge of a 6 px stroke around a disc of radius 30.3 must follow
        // the circle 36.3 to within a fraction of a pixel everywhere.
        let n = 120;
        let r = 30.3;
        let s = apply(&fx(EffectKind::Stroke { size: 6.0, position: StrokePosition::Outside, color: [1.0; 3], opacity: 1.0 }), &disc(n, r));
        let ring = &s.below[0].1;
        let c = n as f32 / 2.0;
        let mut worst = 0.0f32;
        for (i, p) in ring.px.iter().enumerate() {
            let d = ((i % n) as f32 + 0.5 - c).hypot((i / n) as f32 + 0.5 - c);
            if d > r + 3.0 {
                let want = (r + 6.0 - d + 0.5).clamp(0.0, 1.0);
                worst = worst.max((p[3] - want).abs());
            }
        }
        assert!(worst < 0.1, "largest coverage error at the outer edge {worst}");
    }
}
