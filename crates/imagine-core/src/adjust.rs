//! Adjustment layers: non-destructive colour corrections applied to
//! everything below them. Most work on sRGB-encoded values like in
//! Photoshop; per-channel ones are baked into a lookup table, the others
//! (hue/saturation, vibrance, black & white, threshold, exposure) are
//! evaluated in the shader. [`Adjustment::apply`] is the CPU reference.

use serde::{Deserialize, Serialize};

use crate::color::{linear_to_srgb, srgb_to_linear};

/// Entries per channel in a lookup table.
pub const LUT_SIZE: usize = 256;

/// Input/output range and gamma of one channel (all 0…1 except gamma).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Levels {
    pub in_black: f32,
    pub in_white: f32,
    pub gamma: f32,
    pub out_black: f32,
    pub out_white: f32,
}

impl Default for Levels {
    fn default() -> Self {
        Self { in_black: 0.0, in_white: 1.0, gamma: 1.0, out_black: 0.0, out_white: 1.0 }
    }
}

impl Levels {
    pub fn map(&self, x: f32) -> f32 {
        let span = (self.in_white - self.in_black).max(1e-4);
        let t = ((x - self.in_black) / span).clamp(0.0, 1.0);
        let t = t.powf(1.0 / self.gamma.clamp(0.1, 10.0));
        self.out_black + t * (self.out_white - self.out_black)
    }
}

/// A tone curve through control points (x, y in 0…1, sorted by x).
pub type Curve = Vec<[f32; 2]>;

fn identity_curve() -> Curve {
    vec![[0.0, 0.0], [1.0, 1.0]]
}

/// Monotone cubic interpolation (Fritsch–Carlson): smooth, no overshoot.
pub fn eval_curve(points: &[[f32; 2]], x: f32) -> f32 {
    let mut p: Vec<[f32; 2]> = points.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    p.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-6);
    match p.len() {
        0 => return x,
        1 => return p[0][1].clamp(0.0, 1.0),
        _ => {}
    }
    if x <= p[0][0] {
        return p[0][1].clamp(0.0, 1.0);
    }
    if x >= p[p.len() - 1][0] {
        return p[p.len() - 1][1].clamp(0.0, 1.0);
    }
    let n = p.len();
    let d: Vec<f32> = (0..n - 1).map(|i| (p[i + 1][1] - p[i][1]) / (p[i + 1][0] - p[i][0])).collect();
    let mut m = vec![0.0f32; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for i in 1..n - 1 {
        m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) / 2.0 };
    }
    for i in 0..n - 1 {
        if d[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let (a, b) = (m[i] / d[i], m[i + 1] / d[i]);
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * d[i];
                m[i + 1] = t * b * d[i];
            }
        }
    }
    let i = p.windows(2).position(|w| x >= w[0][0] && x <= w[1][0]).unwrap_or(n - 2);
    let h = p[i + 1][0] - p[i][0];
    let t = (x - p[i][0]) / h;
    let (t2, t3) = (t * t, t * t * t);
    let y = (2.0 * t3 - 3.0 * t2 + 1.0) * p[i][1]
        + (t3 - 2.0 * t2 + t) * h * m[i]
        + (-2.0 * t3 + 3.0 * t2) * p[i + 1][1]
        + (t3 - t2) * h * m[i + 1];
    y.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Adjustment {
    /// brightness −150…150, contrast −50…100 (Photoshop ranges).
    BrightnessContrast {
        #[serde(default)]
        brightness: f32,
        #[serde(default)]
        contrast: f32,
    },
    Levels {
        #[serde(default)]
        master: Levels,
        #[serde(default)]
        red: Levels,
        #[serde(default)]
        green: Levels,
        #[serde(default)]
        blue: Levels,
    },
    Curves {
        #[serde(default = "identity_curve")]
        master: Curve,
        #[serde(default = "identity_curve")]
        red: Curve,
        #[serde(default = "identity_curve")]
        green: Curve,
        #[serde(default = "identity_curve")]
        blue: Curve,
    },
    /// hue −180…180°, saturation and lightness −100…100.
    HueSaturation {
        #[serde(default)]
        hue: f32,
        #[serde(default)]
        saturation: f32,
        #[serde(default)]
        lightness: f32,
        #[serde(default)]
        colorize: bool,
    },
    /// −100…100 each.
    Vibrance {
        #[serde(default)]
        vibrance: f32,
        #[serde(default)]
        saturation: f32,
    },
    /// Percent weights per hue sector (Photoshop defaults 40/60/40/60/20/80).
    BlackWhite {
        #[serde(default = "bw_default")]
        weights: [f32; 6],
    },
    /// exposure in stops, offset −0.5…0.5, gamma 0.1…10 (linear light).
    Exposure {
        #[serde(default)]
        exposure: f32,
        #[serde(default)]
        offset: f32,
        #[serde(default = "one")]
        gamma: f32,
    },
    Invert,
    /// Luminance level 1…255.
    Threshold {
        #[serde(default = "half_level")]
        level: f32,
    },
    /// 2…255 levels per channel.
    Posterize {
        #[serde(default = "four")]
        levels: f32,
    },
}

fn bw_default() -> [f32; 6] {
    [40.0, 60.0, 40.0, 60.0, 20.0, 80.0]
}
fn one() -> f32 {
    1.0
}
fn half_level() -> f32 {
    128.0
}
fn four() -> f32 {
    4.0
}

/// How the GPU evaluates an adjustment.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuAdjust {
    /// Per-channel table on sRGB values: `LUT_SIZE` entries of [r, g, b].
    Lut(Vec<[f32; 3]>),
    /// Kind (see `adjust.wgsl`) and up to 8 parameters.
    Shader(u32, [f32; 8]),
}

pub const KIND_LUT: u32 = 0;
pub const KIND_HSL: u32 = 1;
pub const KIND_VIBRANCE: u32 = 2;
pub const KIND_BW: u32 = 3;
pub const KIND_EXPOSURE: u32 = 4;
pub const KIND_THRESHOLD: u32 = 5;

impl Adjustment {
    /// All kinds with neutral settings, for menus.
    pub fn defaults() -> Vec<Adjustment> {
        vec![
            Adjustment::BrightnessContrast { brightness: 0.0, contrast: 0.0 },
            Adjustment::Levels {
                master: Levels::default(),
                red: Levels::default(),
                green: Levels::default(),
                blue: Levels::default(),
            },
            Adjustment::Curves { master: identity_curve(), red: identity_curve(), green: identity_curve(), blue: identity_curve() },
            Adjustment::Exposure { exposure: 0.0, offset: 0.0, gamma: 1.0 },
            Adjustment::Vibrance { vibrance: 0.0, saturation: 0.0 },
            Adjustment::HueSaturation { hue: 0.0, saturation: 0.0, lightness: 0.0, colorize: false },
            Adjustment::BlackWhite { weights: bw_default() },
            Adjustment::Invert,
            Adjustment::Posterize { levels: 4.0 },
            Adjustment::Threshold { level: 128.0 },
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Adjustment::BrightnessContrast { .. } => "Helligkeit/Kontrast",
            Adjustment::Levels { .. } => "Tonwertkorrektur",
            Adjustment::Curves { .. } => "Gradationskurven",
            Adjustment::HueSaturation { .. } => "Farbton/Sättigung",
            Adjustment::Vibrance { .. } => "Dynamik",
            Adjustment::BlackWhite { .. } => "Schwarzweiß",
            Adjustment::Exposure { .. } => "Belichtung",
            Adjustment::Invert => "Umkehren",
            Adjustment::Threshold { .. } => "Schwellenwert",
            Adjustment::Posterize { .. } => "Tontrennung",
        }
    }

    pub fn key(&self) -> &'static str {
        match self {
            Adjustment::BrightnessContrast { .. } => "brightness_contrast",
            Adjustment::Levels { .. } => "levels",
            Adjustment::Curves { .. } => "curves",
            Adjustment::HueSaturation { .. } => "hue_saturation",
            Adjustment::Vibrance { .. } => "vibrance",
            Adjustment::BlackWhite { .. } => "black_white",
            Adjustment::Exposure { .. } => "exposure",
            Adjustment::Invert => "invert",
            Adjustment::Threshold { .. } => "threshold",
            Adjustment::Posterize { .. } => "posterize",
        }
    }

    /// Per-channel mapping on sRGB values, if this adjustment is one.
    fn channel_map(&self) -> Option<Box<dyn Fn(usize, f32) -> f32 + '_>> {
        Some(match self {
            Adjustment::BrightnessContrast { brightness, contrast } => {
                // Brightness bends the midtones (gamma), contrast is an S-curve around 0.5.
                let g = 2f32.powf(-brightness.clamp(-150.0, 150.0) / 100.0);
                let c = contrast.clamp(-50.0, 100.0) / 100.0;
                let k = if c >= 0.0 { 1.0 / (1.0 - 0.7 * c) } else { 1.0 + c };
                Box::new(move |_, x| {
                    let x = x.clamp(0.0, 1.0).powf(g);
                    (0.5 + (x - 0.5) * k).clamp(0.0, 1.0)
                })
            }
            Adjustment::Levels { master, red, green, blue } => {
                let ch = [*red, *green, *blue];
                let m = *master;
                Box::new(move |c, x| ch[c].map(m.map(x)))
            }
            Adjustment::Curves { master, red, green, blue } => {
                let ch = [red, green, blue];
                Box::new(move |c, x| eval_curve(ch[c], eval_curve(master, x)))
            }
            Adjustment::Invert => Box::new(|_, x| 1.0 - x.clamp(0.0, 1.0)),
            Adjustment::Posterize { levels } => {
                let n = levels.clamp(2.0, 255.0).round();
                Box::new(move |_, x| ((x.clamp(0.0, 1.0) * n).floor().min(n - 1.0)) / (n - 1.0))
            }
            _ => return None,
        })
    }

    pub fn gpu(&self) -> GpuAdjust {
        if let Some(f) = self.channel_map() {
            return GpuAdjust::Lut(
                (0..LUT_SIZE)
                    .map(|i| {
                        let x = i as f32 / (LUT_SIZE - 1) as f32;
                        [f(0, x), f(1, x), f(2, x)]
                    })
                    .collect(),
            );
        }
        let mut p = [0.0f32; 8];
        let kind = match self {
            Adjustment::HueSaturation { hue, saturation, lightness, colorize } => {
                p[..4].copy_from_slice(&[*hue / 360.0, *saturation / 100.0, *lightness / 100.0, *colorize as u8 as f32]);
                KIND_HSL
            }
            Adjustment::Vibrance { vibrance, saturation } => {
                p[..2].copy_from_slice(&[*vibrance / 100.0, *saturation / 100.0]);
                KIND_VIBRANCE
            }
            Adjustment::BlackWhite { weights } => {
                for (i, w) in weights.iter().enumerate() {
                    p[i] = w / 100.0;
                }
                KIND_BW
            }
            Adjustment::Exposure { exposure, offset, gamma } => {
                p[..3].copy_from_slice(&[2f32.powf(*exposure), *offset, 1.0 / gamma.clamp(0.1, 10.0)]);
                KIND_EXPOSURE
            }
            Adjustment::Threshold { level } => {
                p[0] = level.clamp(1.0, 255.0) / 255.0;
                KIND_THRESHOLD
            }
            _ => unreachable!("handled by channel_map"),
        };
        GpuAdjust::Shader(kind, p)
    }

    /// CPU reference on one straight-alpha linear RGB colour (the shader does
    /// the same on premultiplied tiles).
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let enc = rgb.map(|v| linear_to_srgb(v.max(0.0)));
        let out = match self.gpu() {
            GpuAdjust::Lut(lut) => {
                let look = |c: usize, x: f32| {
                    let t = x.clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
                    let i = (t.floor() as usize).min(LUT_SIZE - 2);
                    let f = t - i as f32;
                    lut[i][c] * (1.0 - f) + lut[i + 1][c] * f
                };
                [look(0, enc[0]), look(1, enc[1]), look(2, enc[2])]
            }
            GpuAdjust::Shader(kind, p) => match kind {
                KIND_EXPOSURE => {
                    return rgb.map(|v| ((v * p[0] + p[1]).max(0.0)).powf(p[2]));
                }
                KIND_THRESHOLD => {
                    let l = 0.299 * enc[0] + 0.587 * enc[1] + 0.114 * enc[2];
                    [if l >= p[0] { 1.0 } else { 0.0 }; 3]
                }
                KIND_HSL => hue_sat(enc, p),
                KIND_VIBRANCE => vibrance(enc, p),
                KIND_BW => {
                    let g = black_white(enc, p);
                    [g; 3]
                }
                _ => enc,
            },
        };
        out.map(srgb_to_linear)
    }
}

fn rgb_to_hsl(c: [f32; 3]) -> [f32; 3] {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let l = (max + min) / 2.0;
    let d = max - min;
    if d < 1e-6 {
        return [0.0, 0.0, l];
    }
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    let h = if max == c[0] {
        (c[1] - c[2]) / d + if c[1] < c[2] { 6.0 } else { 0.0 }
    } else if max == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    [h / 6.0, s, l]
}

fn hsl_to_rgb(h: [f32; 3]) -> [f32; 3] {
    let (hue, s, l) = (h[0], h[1], h[2]);
    if s <= 0.0 {
        return [l; 3];
    }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let f = |t: f32| {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [f(hue + 1.0 / 3.0), f(hue), f(hue - 1.0 / 3.0)]
}

/// Hue/saturation/lightness on sRGB values (mirrors `adjust.wgsl`).
fn hue_sat(c: [f32; 3], p: [f32; 8]) -> [f32; 3] {
    let mut hsl = rgb_to_hsl(c.map(|v| v.clamp(0.0, 1.0)));
    if p[3] > 0.5 {
        // Colorize: fixed hue and saturation (p[1] = 0 means 25 %).
        hsl[0] = p[0].rem_euclid(1.0);
        hsl[1] = (0.25 + p[1] * 0.75).clamp(0.0, 1.0);
    } else {
        hsl[0] = (hsl[0] + p[0]).rem_euclid(1.0);
        hsl[1] = if p[1] >= 0.0 { hsl[1] + (1.0 - hsl[1]) * p[1] * hsl[1].sqrt() } else { hsl[1] * (1.0 + p[1]) };
    }
    let mut rgb = hsl_to_rgb(hsl);
    if p[2] > 0.0 {
        rgb = rgb.map(|v| v + (1.0 - v) * p[2]);
    } else {
        rgb = rgb.map(|v| v * (1.0 + p[2]));
    }
    rgb
}

fn vibrance(c: [f32; 3], p: [f32; 8]) -> [f32; 3] {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let sat = max - min;
    let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    // Vibrance boosts dull colours more than saturated ones.
    let k = 1.0 + p[1] + p[0] * (1.0 - sat) * if p[0] > 0.0 { 1.0 } else { 0.5 };
    c.map(|v| (l + (v - l) * k).clamp(0.0, 1.0))
}

fn black_white(c: [f32; 3], p: [f32; 8]) -> f32 {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let h = rgb_to_hsl(c)[0] * 6.0; // 0 red, 1 yellow, 2 green, 3 cyan, 4 blue, 5 magenta
    let i = h.floor() as usize % 6;
    let f = h - h.floor();
    let w = p[i] * (1.0 - f) + p[(i + 1) % 6] * f;
    (min + (max - min) * w).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= eps)
    }

    #[test]
    fn neutral_settings_change_nothing() {
        let c = [0.2, 0.5, 0.05];
        for a in Adjustment::defaults() {
            match a {
                Adjustment::Invert
                | Adjustment::Posterize { .. }
                | Adjustment::Threshold { .. }
                | Adjustment::BlackWhite { .. } => continue,
                _ => {}
            }
            assert!(close(a.apply(c), c, 0.01), "{}: {:?}", a.name(), a.apply(c));
        }
    }

    #[test]
    fn curves_are_monotone_and_hit_their_points() {
        let pts = vec![[0.0, 0.0], [0.25, 0.4], [0.75, 0.6], [1.0, 1.0]];
        assert!((eval_curve(&pts, 0.25) - 0.4).abs() < 1e-5);
        let mut last = -1.0;
        for i in 0..=100 {
            let y = eval_curve(&pts, i as f32 / 100.0);
            assert!(y >= last - 1e-6, "monotone");
            last = y;
        }
        assert_eq!(eval_curve(&[[0.0, 1.0], [1.0, 0.0]], 0.0), 1.0, "inverted curve");
    }

    #[test]
    fn levels_invert_threshold_and_bw() {
        let l = Levels { in_black: 0.2, in_white: 0.8, ..Default::default() };
        assert_eq!(l.map(0.1), 0.0);
        assert_eq!(l.map(0.9), 1.0);
        assert!((l.map(0.5) - 0.5).abs() < 1e-6);
        let inv = Adjustment::Invert.apply([1.0, 0.0, srgb_to_linear(0.5)]);
        assert!(close(inv, [0.0, 1.0, srgb_to_linear(0.5)], 1e-3));
        let t = Adjustment::Threshold { level: 128.0 };
        assert_eq!(t.apply([0.9, 0.9, 0.9]), [1.0; 3]);
        assert_eq!(t.apply([0.1, 0.1, 0.1]), [0.0; 3]);
        // Black & white: pure red with the red weight 40 % → about 40 % grey.
        let g = Adjustment::BlackWhite { weights: bw_default() }.apply([1.0, 0.0, 0.0]);
        assert!((linear_to_srgb(g[0]) - 0.4).abs() < 0.01 && g[0] == g[1] && g[1] == g[2]);
    }

    #[test]
    fn hue_rotation_and_desaturation() {
        let h = Adjustment::HueSaturation { hue: 120.0, saturation: 0.0, lightness: 0.0, colorize: false };
        assert!(close(h.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], 1e-3), "red → green");
        let d = Adjustment::HueSaturation { hue: 0.0, saturation: -100.0, lightness: 0.0, colorize: false };
        let g = d.apply([1.0, 0.0, 0.0]);
        assert!((g[0] - g[1]).abs() < 1e-4 && (g[1] - g[2]).abs() < 1e-4);
        let e = Adjustment::Exposure { exposure: 1.0, offset: 0.0, gamma: 1.0 };
        assert!(close(e.apply([0.25, 0.1, 0.0]), [0.5, 0.2, 0.0], 1e-6));
    }

    #[test]
    fn serde_roundtrip_with_defaults() {
        let a: Adjustment = serde_json::from_str(r#"{"kind": "curves", "master": [[0,0],[0.5,0.7],[1,1]]}"#).unwrap();
        let Adjustment::Curves { master, red, .. } = &a else { panic!() };
        assert_eq!(master.len(), 3);
        assert_eq!(red, &identity_curve());
        let s = serde_json::to_string(&a).unwrap();
        assert_eq!(serde_json::from_str::<Adjustment>(&s).unwrap(), a);
        let b: Adjustment = serde_json::from_str(r#"{"kind": "black_white"}"#).unwrap();
        assert_eq!(b, Adjustment::BlackWhite { weights: bw_default() });
    }
}
