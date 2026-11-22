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
        Self {
            in_black: 0.0,
            in_white: 1.0,
            gamma: 1.0,
            out_black: 0.0,
            out_white: 1.0,
        }
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
/// Value of a curve through `points` (input, output) at `x`, like
/// Photoshop: a natural cubic spline through the points (it may overshoot,
/// the result is clamped), constant outside the first and last point.
pub fn eval_curve(points: &[[f32; 2]], x: f32) -> f32 {
    let mut p: Vec<[f32; 2]> = points.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    p.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-6);
    match p.len() {
        0 => return x,
        1 => return p[0][1].clamp(0.0, 1.0),
        _ => {}
    }
    let xs: Vec<f64> = p.iter().map(|q| q[0] as f64).collect();
    let ys: Vec<f64> = p.iter().map(|q| q[1] as f64).collect();
    natural_spline(&xs, &ys, (x as f64).clamp(xs[0], xs[xs.len() - 1])).clamp(0.0, 1.0) as f32
}

/// Natural cubic spline through (xs, ys) evaluated at `x` (xs ascending).
pub fn natural_spline(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 2 {
        let t = (x - xs[0]) / (xs[1] - xs[0]);
        return ys[0] + (ys[1] - ys[0]) * t;
    }
    // Second derivatives (tridiagonal system, zero at the ends).
    let mut m = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        let (h0, h1) = (xs[i] - xs[i - 1], xs[i + 1] - xs[i]);
        let a = h0;
        let b = 2.0 * (h0 + h1);
        let cc = h1;
        let r = 6.0 * ((ys[i + 1] - ys[i]) / h1 - (ys[i] - ys[i - 1]) / h0);
        let w = b - a * c[i - 1];
        c[i] = cc / w;
        d[i] = (r - a * d[i - 1]) / w;
    }
    for i in (1..n - 1).rev() {
        m[i] = d[i] - c[i] * m[i + 1];
    }
    let k = xs.partition_point(|&v| v <= x).clamp(1, n - 1);
    let (x0, x1) = (xs[k - 1], xs[k]);
    let h = x1 - x0;
    let (a, b) = ((x1 - x) / h, (x - x0) / h);
    a * ys[k - 1] + b * ys[k] + ((a * a * a - a) * m[k - 1] + (b * b * b - b) * m[k]) * h * h / 6.0
}

/// Photoshop's brightness/contrast curve (the modern one), as reverse
/// engineered by psd-tools: a contrast S-spline plus a brightness bump,
/// sheared 45°. Returns the output for `x` (sRGB 0…1).
fn brightness_contrast_curve(brightness: f32, contrast: f32) -> impl Fn(f32) -> f32 {
    const N: usize = 1024;
    let b = (brightness.clamp(-150.0, 150.0) / 150.0) as f64;
    let c = (contrast.clamp(-50.0, 100.0) / 100.0) as f64;
    let cx = [0.0, 63.0 / 255.0, 191.0 / 255.0, 1.0];
    let delta = 25.0 / 255.0;
    let cy = [0.0, cx[1] - c * delta, cx[2] + c * delta, 1.0];
    let (a, r) = ([1.65, -1.0, 1.96, 1.0, 1.0], [0.35, 10.0, 0.4, 4.0, 1.25]);
    let pol = |i: usize, t: f64| a[i] * t.powf(r[i]);
    let mut xr = Vec::with_capacity(N);
    let mut yr = Vec::with_capacity(N);
    for i in 0..N {
        let t = i as f64 / (N - 1) as f64;
        let h = 0.5
            * (b.abs() * (pol(0, t) + pol(1, t))
                + (1.0 - b.abs()) * (pol(2, t) + pol(3, t))
                + pol(4, t));
        let bump = b * t * (1.0 - t) * h;
        xr.push(t - bump);
        yr.push(natural_spline(&cx, &cy, t) + bump);
    }
    move |x: f32| {
        let x = x as f64;
        // Linear interpolation in the (sorted) sheared samples.
        let k = xr.partition_point(|&v| v <= x).clamp(1, N - 1);
        let (x0, x1) = (xr[k - 1], xr[k]);
        let t = if x1 > x0 {
            ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let y = if x <= xr[0] {
            yr[0]
        } else if x >= xr[N - 1] {
            yr[N - 1]
        } else {
            yr[k - 1] + (yr[k] - yr[k - 1]) * t
        };
        y.clamp(0.0, 1.0) as f32
    }
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
        /// Photoshop's "Use Legacy": a straight line through mid-grey.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        legacy: bool,
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
        /// Edits for single colour ranges (Photoshop's Reds, Yellows …).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        ranges: Vec<HueRange>,
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
    /// Shifts per tonal range: cyan–red, magenta–green, yellow–blue, each
    /// −100…100.
    ColorBalance {
        #[serde(default)]
        shadows: [f32; 3],
        #[serde(default)]
        midtones: [f32; 3],
        #[serde(default)]
        highlights: [f32; 3],
        #[serde(default = "yes")]
        preserve_luminosity: bool,
    },
    /// Each output channel as a mix of red, green and blue in % (−200…200)
    /// plus a constant (−200…200). Monochrome uses `gray` for all three.
    ChannelMixer {
        #[serde(default = "mix_red")]
        red: [f32; 4],
        #[serde(default = "mix_green")]
        green: [f32; 4],
        #[serde(default = "mix_blue")]
        blue: [f32; 4],
        #[serde(default = "mix_gray")]
        gray: [f32; 4],
        #[serde(default)]
        monochrome: bool,
    },
    /// The luminance mapped onto a gradient (dark → its start).
    GradientMap {
        #[serde(default)]
        gradient: crate::gradient::Gradient,
        #[serde(default)]
        reverse: bool,
    },
    /// Like a coloured filter in front of the lens: `color` linear RGB,
    /// density 0…100 %.
    PhotoFilter {
        #[serde(default = "warming")]
        color: [f32; 3],
        #[serde(default = "twenty_five")]
        density: f32,
        #[serde(default = "yes")]
        preserve_luminosity: bool,
    },
    /// Cyan, magenta, yellow, black (−100…100 %) per colour range: reds,
    /// yellows, greens, cyans, blues, magentas, whites, neutrals, blacks.
    /// `relative` scales by the ink already there (Photoshop's Relative).
    SelectiveColor {
        #[serde(default)]
        ranges: [[f32; 4]; 9],
        #[serde(default = "yes")]
        relative: bool,
    },
    /// A 3D colour lookup table (.cube), applied to sRGB values; `amount`
    /// 0…100 %. `path` is read into `data` when given (agents).
    ColorLookup {
        #[serde(default)]
        name: String,
        #[serde(default)]
        size: u32,
        /// size³ RGB triples, red fastest (as in .cube files).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        data: Vec<[f32; 3]>,
        #[serde(default = "hundred")]
        amount: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<std::path::PathBuf>,
    },
}

/// Names of the Selective Color ranges, in order.
pub const SELECTIVE_RANGES: [&str; 9] = ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas", "Whites", "Neutrals", "Blacks"];

fn warming() -> [f32; 3] {
    // Photoshop's Warming Filter (85): #ec8a00.
    [srgb_to_linear(236.0 / 255.0), srgb_to_linear(138.0 / 255.0), 0.0]
}
fn twenty_five() -> f32 {
    25.0
}
fn hundred() -> f32 {
    100.0
}

/// Reads a .cube 3D LUT: (size, size³ RGB triples, red fastest).
/// A .cube file for a table (`size`³ RGB triples, red fastest).
pub fn write_cube(title: &str, size: u32, data: &[[f32; 3]]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let title = title.replace('"', "'");
    let _ = writeln!(s, "TITLE \"{}\"", if title.is_empty() { "mosaix" } else { &title });
    let _ = writeln!(s, "LUT_3D_SIZE {size}");
    for c in data {
        let _ = writeln!(s, "{:.6} {:.6} {:.6}", c[0], c[1], c[2]);
    }
    s
}

pub fn parse_cube(text: &str) -> Result<(u32, Vec<[f32; 3]>), String> {
    let mut size = 0u32;
    let (mut lo, mut hi) = ([0.0f32; 3], [1.0f32; 3]);
    let mut data = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        let mut it = l.split_whitespace();
        let first = it.next().unwrap_or("");
        match first {
            "TITLE" => {}
            "LUT_3D_SIZE" => size = it.next().and_then(|v| v.parse().ok()).ok_or("bad LUT_3D_SIZE")?,
            "LUT_1D_SIZE" => return Err("1D LUTs are not supported, only 3D".into()),
            "DOMAIN_MIN" | "DOMAIN_MAX" => {
                let v: Vec<f32> = it.filter_map(|v| v.parse().ok()).collect();
                if v.len() == 3 {
                    let t = [v[0], v[1], v[2]];
                    if first == "DOMAIN_MIN" { lo = t } else { hi = t }
                }
            }
            _ if first.parse::<f32>().is_ok() => {
                let v: Vec<f32> = l.split_whitespace().filter_map(|v| v.parse().ok()).collect();
                if v.len() != 3 {
                    return Err(format!("bad line: {l}"));
                }
                data.push([0, 1, 2].map(|k| (v[k] - lo[k]) / (hi[k] - lo[k]).max(1e-6)));
            }
            _ => {}
        }
    }
    if !(2..=65).contains(&size) {
        return Err("LUT_3D_SIZE missing or out of range (2…65)".into());
    }
    if data.len() != (size * size * size) as usize {
        return Err(format!("expected {} entries, found {}", size * size * size, data.len()));
    }
    Ok((size, data))
}

/// Trilinear lookup in a 3D LUT (sRGB in and out).
fn lut3d(size: u32, data: &[[f32; 3]], c: [f32; 3]) -> [f32; 3] {
    let n = size as usize;
    let at = |r: usize, g: usize, b: usize| data[(b * n + g) * n + r];
    let pos = c.map(|v| v.clamp(0.0, 1.0) * (n - 1) as f32);
    let i0 = pos.map(|p| (p.floor() as usize).min(n - 2));
    let f: [f32; 3] = std::array::from_fn(|k| pos[k] - i0[k] as f32);
    let mut out = [0.0; 3];
    for (dz, wz) in [(0, 1.0 - f[2]), (1, f[2])] {
        for (dy, wy) in [(0, 1.0 - f[1]), (1, f[1])] {
            for (dx, wx) in [(0, 1.0 - f[0]), (1, f[0])] {
                let v = at(i0[0] + dx, i0[1] + dy, i0[2] + dz);
                for k in 0..3 {
                    out[k] += v[k] * wx * wy * wz;
                }
            }
        }
    }
    out
}

/// Selective Color on sRGB values (mirrors `adjust.wgsl`).
fn selective_color(c: [f32; 3], relative: bool, ranges: &[[f32; 4]]) -> [f32; 3] {
    let c = c.map(|v| v.clamp(0.0, 1.0));
    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let mid = c[0] + c[1] + c[2] - mx - mn;
    let w = [
        if c[0] >= mx { mx - mid } else { 0.0 },
        if c[2] <= mn { mid - mn } else { 0.0 },
        if c[1] >= mx && c[0] < mx { mx - mid } else { 0.0 },
        if c[0] <= mn && c[2] > mn { mid - mn } else { 0.0 },
        if c[2] >= mx && c[0] < mx && c[1] < mx { mx - mid } else { 0.0 },
        if c[1] <= mn && c[0] > mn && c[2] > mn { mid - mn } else { 0.0 },
        ((mn - 0.5) * 2.0).max(0.0),
        (1.0 - ((mx - 0.5).abs() + (mn - 0.5).abs())).max(0.0),
        ((0.5 - mx) * 2.0).max(0.0),
    ];
    let mut out = c;
    for (r, &wr) in ranges.iter().zip(&w) {
        if wr <= 0.0 {
            continue;
        }
        for k in 0..3 {
            // More of an ink (and of black) takes away its channel.
            let a = r[k] + r[3];
            let scale = if relative { 1.0 - c[k] } else { 1.0 };
            out[k] -= wr * a * scale;
        }
    }
    out.map(|v| v.clamp(0.0, 1.0))
}

/// Auto Tone / Auto Contrast / Auto Color from a histogram (bins on sRGB
/// 0…255; 0 = all channels, 1–3 = red, green, blue): Levels that clip 0.1 %
/// at each end – per channel, for all channels alike, or per channel with
/// neutral midtones.
pub fn auto_levels(bins: &[[f32; 256]; 4], mode: AutoMode) -> Adjustment {
    let ends = |h: &[f32; 256]| {
        let total: f32 = h.iter().sum();
        let clip = total * 0.001;
        let (mut acc, mut lo) = (0.0, 0);
        for (i, v) in h.iter().enumerate() {
            acc += v;
            if acc > clip {
                lo = i;
                break;
            }
        }
        let (mut acc, mut hi) = (0.0, 255);
        for (i, v) in h.iter().enumerate().rev() {
            acc += v;
            if acc > clip {
                hi = i;
                break;
            }
        }
        let mean = if total > 0.0 { h.iter().enumerate().map(|(i, v)| i as f32 * v).sum::<f32>() / total / 255.0 } else { 0.5 };
        (lo as f32 / 255.0, (hi.max(lo + 1)) as f32 / 255.0, mean)
    };
    let lv = |(b, w, _): (f32, f32, f32)| Levels { in_black: b, in_white: w, ..Default::default() };
    let identity = Levels::default();
    match mode {
        AutoMode::Contrast => {
            let all = ends(&bins[0]);
            Adjustment::Levels { master: lv(all), red: identity, green: identity, blue: identity }
        }
        AutoMode::Tone | AutoMode::Color => {
            let ch: [(f32, f32, f32); 3] = [ends(&bins[1]), ends(&bins[2]), ends(&bins[3])];
            let mut out = ch.map(lv);
            if mode == AutoMode::Color {
                // Midtones to the common grey: each channel's stretched mean to the average.
                let stretched: Vec<f32> = ch.iter().map(|(b, w, m)| ((m - b) / (w - b).max(1e-4)).clamp(0.02, 0.98)).collect();
                let target = stretched.iter().sum::<f32>() / 3.0;
                for (l, t) in out.iter_mut().zip(&stretched) {
                    l.gamma = (t.ln() / target.ln()).clamp(0.3, 3.0);
                }
            }
            let [red, green, blue] = out;
            Adjustment::Levels { master: identity, red, green, blue }
        }
    }
}

/// Image › Adjustments › Auto …
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoMode {
    Tone,
    Contrast,
    Color,
}

impl Adjustment {
    /// Reads files an adjustment refers to (a .cube `path`).
    pub fn load_files(&mut self) -> Result<(), String> {
        if let Adjustment::ColorLookup { name, size, data, path, .. } = self
            && let Some(p) = path.take()
        {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let (n, d) = parse_cube(&text)?;
            (*size, *data) = (n, d);
            if name.is_empty() {
                *name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            }
        }
        Ok(())
    }
}

fn yes() -> bool {
    true
}
fn mix_red() -> [f32; 4] {
    [100.0, 0.0, 0.0, 0.0]
}
fn mix_green() -> [f32; 4] {
    [0.0, 100.0, 0.0, 0.0]
}
fn mix_blue() -> [f32; 4] {
    [0.0, 0.0, 100.0, 0.0]
}
fn mix_gray() -> [f32; 4] {
    [40.0, 40.0, 20.0, 0.0]
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
/// A colour range of Hue/Saturation: four hue points in degrees (start of
/// the fade-in, full range from … to, end of the fade-out) and its edits
/// (hue −180…180, saturation and lightness −100…100).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HueRange {
    pub range: [f32; 4],
    #[serde(default)]
    pub hue: f32,
    #[serde(default)]
    pub saturation: f32,
    #[serde(default)]
    pub lightness: f32,
}

impl HueRange {
    /// Photoshop's six default ranges.
    pub const DEFAULTS: [[f32; 4]; 6] = [
        [315.0, 345.0, 15.0, 45.0],
        [15.0, 45.0, 75.0, 105.0],
        [75.0, 105.0, 135.0, 165.0],
        [135.0, 165.0, 195.0, 225.0],
        [195.0, 225.0, 255.0, 285.0],
        [255.0, 285.0, 315.0, 345.0],
    ];

    pub fn defaults() -> Vec<HueRange> {
        Self::DEFAULTS.iter().map(|r| HueRange { range: *r, hue: 0.0, saturation: 0.0, lightness: 0.0 }).collect()
    }

    pub fn is_identity(&self) -> bool {
        self.hue == 0.0 && self.saturation == 0.0 && self.lightness == 0.0
    }
}

/// How Photoshop combines saturation edits of overlapping ranges (measured by
/// psd-tools): 21 × 21 values over (accumulated, contribution) in −1…1.
pub const HUE_SAT_GRID: [f32; 441] = [-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.89,-0.76,-0.58,-0.35,-0.01,0.32,0.56,0.74,0.88,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.66,-0.48,-0.25,0.09,0.36,0.58,0.75,0.88,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.89,-0.80,-0.69,-0.56,-0.38,-0.15,0.17,0.41,0.59,0.76,0.88,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.59,-0.46,-0.28,-0.05,0.22,0.44,0.61,0.76,0.89,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.89,-0.79,-0.70,-0.60,-0.49,-0.36,-0.18,0.05,0.28,0.47,0.62,0.77,0.89,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.60,-0.49,-0.39,-0.26,-0.08,0.13,0.33,0.49,0.64,0.77,0.89,1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.60,-0.50,-0.40,-0.29,-0.16,0.02,0.21,0.37,0.52,0.65,0.78,0.89,1.00,-1.00,-1.00,-1.00,-0.99,-0.89,-0.79,-0.70,-0.59,-0.50,-0.40,-0.30,-0.19,-0.06,0.11,0.26,0.41,0.54,0.66,0.78,0.89,1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.60,-0.50,-0.40,-0.30,-0.20,-0.09,0.04,0.19,0.31,0.44,0.56,0.67,0.79,0.89,1.00,-1.00,-0.99,-0.89,-0.79,-0.70,-0.60,-0.50,-0.40,-0.30,-0.19,-0.10,0.01,0.13,0.24,0.36,0.47,0.58,0.68,0.79,0.89,1.00,-1.00,-0.90,-0.80,-0.70,-0.60,-0.49,-0.40,-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,-0.89,-0.79,-0.69,-0.59,-0.49,-0.39,-0.29,-0.19,-0.09,0.01,0.10,0.17,0.26,0.35,0.43,0.52,0.61,0.70,0.80,0.90,1.00,-0.76,-0.66,-0.56,-0.46,-0.36,-0.26,-0.16,-0.06,0.04,0.13,0.20,0.26,0.33,0.39,0.47,0.55,0.63,0.71,0.81,0.90,1.00,-0.58,-0.48,-0.38,-0.28,-0.18,-0.08,0.02,0.11,0.19,0.24,0.30,0.35,0.39,0.46,0.52,0.59,0.65,0.73,0.81,0.90,1.00,-0.35,-0.25,-0.15,-0.05,0.05,0.13,0.21,0.26,0.31,0.36,0.40,0.43,0.47,0.52,0.57,0.62,0.68,0.74,0.82,0.90,1.00,-0.01,0.09,0.17,0.22,0.28,0.33,0.37,0.41,0.44,0.47,0.50,0.52,0.55,0.59,0.62,0.66,0.71,0.77,0.83,0.90,1.00,0.32,0.36,0.41,0.44,0.47,0.49,0.52,0.54,0.56,0.58,0.60,0.61,0.63,0.65,0.68,0.71,0.75,0.79,0.84,0.91,1.00,0.56,0.58,0.59,0.61,0.62,0.64,0.65,0.66,0.67,0.68,0.70,0.70,0.71,0.73,0.74,0.77,0.79,0.82,0.86,0.91,1.00,0.74,0.75,0.76,0.76,0.77,0.77,0.78,0.78,0.79,0.79,0.80,0.80,0.81,0.81,0.82,0.83,0.84,0.86,0.89,0.92,1.00,0.88,0.88,0.88,0.89,0.89,0.89,0.89,0.89,0.89,0.89,0.90,0.90,0.90,0.90,0.90,0.90,0.91,0.91,0.92,0.94,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00];

fn hue_sat_grid(a: f32, b: f32) -> f32 {
    let (x, y) = ((a.clamp(-1.0, 1.0) + 1.0) * 10.0, (b.clamp(-1.0, 1.0) + 1.0) * 10.0);
    let (i, j) = ((x.floor() as usize).min(19), (y.floor() as usize).min(19));
    let (fx, fy) = (x - i as f32, y - j as f32);
    let g = |i: usize, j: usize| HUE_SAT_GRID[i * 21 + j];
    let top = g(i, j) * (1.0 - fy) + g(i, j + 1) * fy;
    let bot = g(i + 1, j) * (1.0 - fy) + g(i + 1, j + 1) * fy;
    top * (1.0 - fx) + bot * fx
}

/// Trapezoid weight of `hue` (0…1) in a range given in degrees.
fn range_mask(hue: f32, r: [f32; 4]) -> f32 {
    let [bl, tl, tr, br] = r.map(|v| v / 360.0);
    let c = (hue - bl).rem_euclid(1.0);
    let (ls, cs, rs) = ((tl - bl).rem_euclid(1.0), (tr - bl).rem_euclid(1.0), (br - bl).rem_euclid(1.0));
    if c >= ls && c <= cs {
        1.0
    } else if c < ls {
        if ls > 1e-9 { c / ls } else { 0.0 }
    } else if c <= rs {
        if rs - cs > 1e-9 { (rs - c) / (rs - cs) } else { 0.0 }
    } else {
        0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuAdjust {
    /// Per-channel table on sRGB values: `LUT_SIZE` entries of [r, g, b].
    Lut(Vec<[f32; 3]>),
    /// Kind (see `adjust.wgsl`), up to 8 parameters and more (hue ranges).
    Shader(u32, [f32; 8], Vec<[f32; 4]>),
}

pub const KIND_LUT: u32 = 0;
pub const KIND_HSL: u32 = 1;
pub const KIND_VIBRANCE: u32 = 2;
pub const KIND_BW: u32 = 3;
pub const KIND_EXPOSURE: u32 = 4;
pub const KIND_THRESHOLD: u32 = 5;
pub const KIND_COLOR_BALANCE: u32 = 6;
pub const KIND_CHANNEL_MIXER: u32 = 7;
pub const KIND_GRADIENT_MAP: u32 = 8;
pub const KIND_PHOTO_FILTER: u32 = 9;
pub const KIND_SELECTIVE: u32 = 10;
pub const KIND_LUT3D: u32 = 11;

impl Adjustment {
    /// All kinds with neutral settings, for menus.
    pub fn defaults() -> Vec<Adjustment> {
        vec![
            Adjustment::BrightnessContrast {
                brightness: 0.0,
                contrast: 0.0,
                legacy: false,
            },
            Adjustment::Levels {
                master: Levels::default(),
                red: Levels::default(),
                green: Levels::default(),
                blue: Levels::default(),
            },
            Adjustment::Curves {
                master: identity_curve(),
                red: identity_curve(),
                green: identity_curve(),
                blue: identity_curve(),
            },
            Adjustment::Exposure {
                exposure: 0.0,
                offset: 0.0,
                gamma: 1.0,
            },
            Adjustment::Vibrance {
                vibrance: 0.0,
                saturation: 0.0,
            },
            Adjustment::HueSaturation {
                hue: 0.0,
                saturation: 0.0,
                lightness: 0.0,
                colorize: false,
                ranges: Vec::new(),
            },
            Adjustment::BlackWhite {
                weights: bw_default(),
            },
            Adjustment::ColorBalance {
                shadows: [0.0; 3],
                midtones: [0.0; 3],
                highlights: [0.0; 3],
                preserve_luminosity: true,
            },
            Adjustment::ChannelMixer {
                red: mix_red(),
                green: mix_green(),
                blue: mix_blue(),
                gray: mix_gray(),
                monochrome: false,
            },
            Adjustment::GradientMap {
                gradient: crate::gradient::Gradient::two([0.0; 3], [1.0; 3]),
                reverse: false,
            },
            Adjustment::PhotoFilter { color: warming(), density: 25.0, preserve_luminosity: true },
            Adjustment::SelectiveColor { ranges: [[0.0; 4]; 9], relative: true },
            Adjustment::ColorLookup { name: String::new(), size: 0, data: Vec::new(), amount: 100.0, path: None },
            Adjustment::Invert,
            Adjustment::Posterize { levels: 4.0 },
            Adjustment::Threshold { level: 128.0 },
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Adjustment::BrightnessContrast { .. } => crate::tr("Brightness/Contrast"),
            Adjustment::Levels { .. } => crate::tr("Levels"),
            Adjustment::Curves { .. } => crate::tr("Curves"),
            Adjustment::HueSaturation { .. } => crate::tr("Hue/Saturation"),
            Adjustment::Vibrance { .. } => crate::tr("Vibrance"),
            Adjustment::BlackWhite { .. } => crate::tr("Black & White"),
            Adjustment::Exposure { .. } => crate::tr("Exposure"),
            Adjustment::Invert => crate::tr("Invert"),
            Adjustment::Threshold { .. } => crate::tr("Threshold"),
            Adjustment::Posterize { .. } => crate::tr("Posterize"),
            Adjustment::ColorBalance { .. } => crate::tr("Color Balance"),
            Adjustment::ChannelMixer { .. } => crate::tr("Channel Mixer"),
            Adjustment::GradientMap { .. } => crate::tr("Gradient Map"),
            Adjustment::PhotoFilter { .. } => crate::tr("Photo Filter"),
            Adjustment::SelectiveColor { .. } => crate::tr("Selective Color"),
            Adjustment::ColorLookup { .. } => crate::tr("Color Lookup"),
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
            Adjustment::ColorBalance { .. } => "color_balance",
            Adjustment::ChannelMixer { .. } => "channel_mixer",
            Adjustment::GradientMap { .. } => "gradient_map",
            Adjustment::PhotoFilter { .. } => "photo_filter",
            Adjustment::SelectiveColor { .. } => "selective_color",
            Adjustment::ColorLookup { .. } => "color_lookup",
        }
    }

    /// Per-channel mapping on sRGB values, if this adjustment is one.
    fn channel_map(&self) -> Option<Box<dyn Fn(usize, f32) -> f32 + '_>> {
        Some(match self {
            Adjustment::BrightnessContrast {
                brightness,
                contrast,
                legacy: true,
            } => {
                // Photoshop's legacy mode: shift, then a line through mid-grey.
                let (shift, c) = (brightness / 255.0, *contrast);
                let pivot = 127.0 / 255.0;
                Box::new(move |_, x| {
                    let y = if c > 0.0 {
                        (x + shift - pivot) * (100.0 / (100.0 - c).max(1e-6)) + pivot
                    } else {
                        (x - pivot) * ((100.0 + c) / 100.0) + pivot + shift
                    };
                    y.clamp(0.0, 1.0)
                })
            }
            Adjustment::BrightnessContrast {
                brightness,
                contrast,
                legacy: false,
            } => {
                let f = brightness_contrast_curve(*brightness, *contrast);
                Box::new(move |_, x| f(x.clamp(0.0, 1.0)))
            }
            Adjustment::Levels {
                master,
                red,
                green,
                blue,
            } => {
                let ch = [*red, *green, *blue];
                let m = *master;
                // Like Photoshop: the channel first, then the composite.
                Box::new(move |c, x| m.map(ch[c].map(x)))
            }
            Adjustment::Curves {
                master,
                red,
                green,
                blue,
            } => {
                let ch = [red, green, blue];
                Box::new(move |c, x| eval_curve(master, eval_curve(ch[c], x)))
            }
            Adjustment::Invert => Box::new(|_, x| 1.0 - x.clamp(0.0, 1.0)),
            Adjustment::Posterize { levels } => {
                let n = levels.clamp(2.0, 255.0).round();
                // Photoshop scales by 255/256 before flooring.
                Box::new(move |_, x| {
                    ((x.clamp(0.0, 1.0) * n * (255.0 / 256.0))
                        .floor()
                        .min(n - 1.0))
                        / (n - 1.0)
                })
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
        let mut extra = Vec::new();
        let kind = match self {
            Adjustment::HueSaturation {
                hue,
                saturation,
                lightness,
                colorize,
                ranges,
            } => {
                p[..4].copy_from_slice(&[
                    *hue / 360.0,
                    *saturation / 100.0,
                    *lightness / 100.0,
                    *colorize as u8 as f32,
                ]);
                // Ranges follow: [bl, tl, tr, br] (degrees), [h, s, l, 0] (normalised).
                let used: Vec<&HueRange> = ranges.iter().filter(|r| !r.is_identity()).take(6).collect();
                p[4] = used.len() as f32;
                for r in used {
                    extra.push(r.range);
                    extra.push([r.hue / 360.0, r.saturation / 100.0, r.lightness / 100.0, 0.0]);
                }
                KIND_HSL
            }
            Adjustment::Vibrance {
                vibrance,
                saturation,
            } => {
                p[..2].copy_from_slice(&[*vibrance / 100.0, *saturation / 100.0]);
                KIND_VIBRANCE
            }
            Adjustment::BlackWhite { weights } => {
                for (i, w) in weights.iter().enumerate() {
                    p[i] = w / 100.0;
                }
                KIND_BW
            }
            Adjustment::Exposure {
                exposure,
                offset,
                gamma,
            } => {
                p[..3].copy_from_slice(&[
                    2f32.powf(*exposure),
                    *offset,
                    1.0 / gamma.clamp(0.1, 10.0),
                ]);
                KIND_EXPOSURE
            }
            Adjustment::Threshold { level } => {
                p[0] = level.clamp(1.0, 255.0) / 255.0;
                KIND_THRESHOLD
            }
            Adjustment::ColorBalance {
                shadows,
                midtones,
                highlights,
                preserve_luminosity,
            } => {
                let n = |v: &[f32; 3]| v.map(|x| x.clamp(-100.0, 100.0) / 100.0);
                let (sh, mi, hi) = (n(shadows), n(midtones), n(highlights));
                p = [sh[0], sh[1], sh[2], *preserve_luminosity as u8 as f32, mi[0], mi[1], mi[2], 0.0];
                extra.push([hi[0], hi[1], hi[2], 0.0]);
                KIND_COLOR_BALANCE
            }
            Adjustment::ChannelMixer {
                red,
                green,
                blue,
                gray,
                monochrome,
            } => {
                let n = |v: &[f32; 4]| v.map(|x| x.clamp(-200.0, 200.0) / 100.0);
                let rows = if *monochrome { [n(gray); 3] } else { [n(red), n(green), n(blue)] };
                p[..4].copy_from_slice(&rows[0]);
                p[4..].copy_from_slice(&rows[1]);
                extra.push(rows[2]);
                KIND_CHANNEL_MIXER
            }
            Adjustment::GradientMap { gradient, reverse } => {
                p[0] = *reverse as u8 as f32;
                // The gradient as a table of sRGB-encoded colours.
                let lut = gradient.lut();
                extra.extend((0..LUT_SIZE).map(|i| {
                    let c = lut.at(i as f32 / (LUT_SIZE - 1) as f32);
                    [linear_to_srgb(c[0]), linear_to_srgb(c[1]), linear_to_srgb(c[2]), 0.0]
                }));
                KIND_GRADIENT_MAP
            }
            Adjustment::PhotoFilter { color, density, preserve_luminosity } => {
                let f = color.map(|v| linear_to_srgb(v.clamp(0.0, 1.0)));
                p[..4].copy_from_slice(&[f[0], f[1], f[2], density.clamp(0.0, 100.0) / 100.0]);
                p[4] = *preserve_luminosity as u8 as f32;
                KIND_PHOTO_FILTER
            }
            Adjustment::SelectiveColor { ranges, relative } => {
                p[0] = *relative as u8 as f32;
                extra.extend(ranges.iter().map(|r| r.map(|v| v.clamp(-100.0, 100.0) / 100.0)));
                KIND_SELECTIVE
            }
            Adjustment::ColorLookup { size, data, amount, .. } => {
                if *size < 2 || data.len() != (*size as usize).pow(3) {
                    // No table loaded: no change.
                    p[1] = 0.0;
                } else {
                    p[0] = *size as f32;
                    p[1] = amount.clamp(0.0, 100.0) / 100.0;
                    extra.extend(data.iter().map(|c| [c[0], c[1], c[2], 0.0]));
                }
                KIND_LUT3D
            }
            _ => unreachable!("handled by channel_map"),
        };
        GpuAdjust::Shader(kind, p, extra)
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
            GpuAdjust::Shader(kind, p, extra) => match kind {
                KIND_EXPOSURE => {
                    return rgb.map(|v| ((v * p[0] + p[1]).max(0.0)).powf(p[2]));
                }
                KIND_THRESHOLD => {
                    // Photoshop: luminance in 8-bit steps.
                    let l =
                        ((0.3 * enc[0] + 0.59 * enc[1] + 0.11 * enc[2]) * 255.0).round() / 255.0;
                    [if l >= p[0] - 0.5 / 255.0 { 1.0 } else { 0.0 }; 3]
                }
                KIND_HSL => hue_sat(enc, p, &extra),
                KIND_COLOR_BALANCE => color_balance(enc, p, &extra),
                KIND_CHANNEL_MIXER => {
                    let row = |r: &[f32]| (r[0] * enc[0] + r[1] * enc[1] + r[2] * enc[2] + r[3]).clamp(0.0, 1.0);
                    [row(&p[..4]), row(&p[4..]), row(&extra[0])]
                }
                KIND_GRADIENT_MAP => {
                    let mut l = (0.3 * enc[0] + 0.59 * enc[1] + 0.11 * enc[2]).clamp(0.0, 1.0);
                    if p[0] > 0.5 {
                        l = 1.0 - l;
                    }
                    let t = l * (LUT_SIZE - 1) as f32;
                    let i = (t.floor() as usize).min(LUT_SIZE - 2);
                    let f = t - i as f32;
                    std::array::from_fn(|k| extra[i][k] * (1.0 - f) + extra[i + 1][k] * f)
                }
                KIND_VIBRANCE => vibrance(enc, p),
                KIND_PHOTO_FILTER => photo_filter(enc, p),
                KIND_SELECTIVE => selective_color(enc, p[0] > 0.5, &extra),
                KIND_LUT3D => {
                    if p[1] <= 0.0 {
                        enc
                    } else {
                        let n = p[0] as u32;
                        let data: Vec<[f32; 3]> = extra.iter().map(|v| [v[0], v[1], v[2]]).collect();
                        let o = lut3d(n, &data, enc);
                        std::array::from_fn(|k| enc[k] + (o[k] - enc[k]) * p[1])
                    }
                }
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

/// Photo filter on sRGB values (mirrors `adjust.wgsl`): the colour
/// multiplied in by the density, luminance kept if asked.
fn photo_filter(c: [f32; 3], p: [f32; 8]) -> [f32; 3] {
    let luma = |v: [f32; 3]| 0.3 * v[0] + 0.59 * v[1] + 0.11 * v[2];
    let c = c.map(|v| v.clamp(0.0, 1.0));
    let m: [f32; 3] = std::array::from_fn(|k| c[k] * p[k]);
    let mut o: [f32; 3] = std::array::from_fn(|k| c[k] + (m[k] - c[k]) * p[3]);
    if p[4] > 0.5 {
        let (l0, l1) = (luma(c), luma(o));
        if l1 > 1e-5 {
            o = o.map(|v| v * l0 / l1);
        }
    }
    o.map(|v| v.clamp(0.0, 1.0))
}

/// Color balance on sRGB values (mirrors `adjust.wgsl`; GIMP's weighting of
/// the tonal ranges by lightness).
fn color_balance(c: [f32; 3], p: [f32; 8], extra: &[[f32; 4]]) -> [f32; 3] {
    let c = c.map(|v| v.clamp(0.0, 1.0));
    let l = rgb_to_hsl(c)[2];
    let (a, b, scale) = (0.25f32, 0.333f32, 0.7f32);
    let ws = ((l - b) / -a + 0.5).clamp(0.0, 1.0) * scale;
    let wm = ((l - b) / a + 0.5).clamp(0.0, 1.0) * ((l + b - 1.0) / -a + 0.5).clamp(0.0, 1.0) * scale;
    let wh = ((l + b - 1.0) / a + 0.5).clamp(0.0, 1.0) * scale;
    let mut o: [f32; 3] = std::array::from_fn(|k| (c[k] + p[k] * ws + p[4 + k] * wm + extra[0][k] * wh).clamp(0.0, 1.0));
    if p[3] > 0.5 {
        let mut h = rgb_to_hsl(o);
        h[2] = l;
        o = hsl_to_rgb(h);
    }
    o
}

fn rgb_to_hsl(c: [f32; 3]) -> [f32; 3] {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let l = (max + min) / 2.0;
    let d = max - min;
    if d < 1e-6 {
        return [0.0, 0.0, l];
    }
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
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
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
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
fn hue_sat(c: [f32; 3], p: [f32; 8], extra: &[[f32; 4]]) -> [f32; 3] {
    // Photoshop (as measured by psd-tools): lightness first on RGB, then the
    // colour ranges (lightness in HSV), then hue and saturation in HSL;
    // colorize sets hue and saturation and lightens in HSL.
    let light = |v: f32, l: f32| if l > 0.0 { v * (1.0 - l) + l } else { v * (1.0 + l) };
    let sat = |s: f32, k: f32| if k > 0.0 { s / (1.0 - k + 0.01) } else { s * (1.0 + k) }.clamp(0.0, 1.0);
    let c = c.map(|v| v.clamp(0.0, 1.0));
    if p[3] > 0.5 {
        let mut hsl = rgb_to_hsl(c);
        hsl[0] = p[0].rem_euclid(1.0);
        hsl[1] = (0.25 + p[1] * 0.75).clamp(0.0, 1.0);
        hsl[2] = light(hsl[2], p[2]).clamp(0.0, 1.0);
        return hsl_to_rgb(hsl);
    }
    let mut hsl = rgb_to_hsl(c.map(|v| light(v, p[2]).clamp(0.0, 1.0)));
    let (mut rh, mut rs, mut rl) = (0.0f32, 0.0f32, 0.0f32);
    for r in extra.chunks(2).take(p[4] as usize) {
        let (range, e) = (r[0], r[1]);
        let m = range_mask(hsl[0], range);
        rh += e[0] * m;
        rl = (rl + e[2] * m).clamp(-1.0, 1.0);
        let k = e[1];
        let contrib = if k > 0.0 { k * (1.0 - (1.0 - m).powf(1.5 / (1.0 - k.powi(4) + 0.05))) } else { k * m };
        rs = hue_sat_grid(rs, contrib);
    }
    if rl != 0.0 {
        // Range lightness in HSV.
        let (s_l, l) = (hsl[1], hsl[2]);
        let v = l + s_l * l.min(1.0 - l);
        let s_v = if v > 0.0 { 2.0 * (1.0 - l / v) } else { 0.0 };
        let v2 = if rl >= 0.0 { v } else { (v * (1.0 + rl * s_v)).clamp(0.0, 1.0) };
        let s2 = if rl >= 0.0 {
            s_v * (1.0 - rl)
        } else {
            let div = 1.0 / (s_v + 1e-3);
            1.0 + (div - 1.0) / (rl.abs() - div + 1e-3)
        }
        .clamp(0.0, 1.0);
        let l2 = v2 * (1.0 - s2 / 2.0);
        let s3 = if l2 > 0.0 && l2 < 1.0 { (v2 - l2) / l2.min(1.0 - l2) } else { 0.0 };
        hsl[1] = s3.clamp(0.0, 1.0);
        hsl[2] = l2.clamp(0.0, 1.0);
    }
    hsl[0] = (hsl[0] + rh + p[0]).rem_euclid(1.0);
    hsl[1] = sat(sat(hsl[1], p[1]), rs);
    hsl_to_rgb(hsl)
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
    fn photo_filter_selective_color_lookup_and_auto() {
        // Photo filter: warmer, luminance kept.
        let pf = Adjustment::PhotoFilter { color: warming(), density: 50.0, preserve_luminosity: true };
        let grey = [0.2f32; 3];
        let o = pf.apply(grey);
        assert!(o[0] > o[2] + 0.02, "warmer: {o:?}");
        let l = |c: [f32; 3]| {
            let e = c.map(linear_to_srgb);
            0.3 * e[0] + 0.59 * e[1] + 0.11 * e[2]
        };
        assert!((l(o) - l(grey)).abs() < 0.01);
        // Selective color: more cyan in the reds lowers red, leaves blue alone.
        let mut ranges = [[0.0; 4]; 9];
        ranges[0][0] = 50.0;
        let sc = Adjustment::SelectiveColor { ranges, relative: false };
        let red = [0.8, 0.05, 0.05];
        let o = sc.apply(red);
        assert!(o[0] < red[0] - 0.05 && (o[2] - red[2]).abs() < 1e-4, "{o:?}");
        assert!(close(sc.apply([0.05, 0.05, 0.8]), [0.05, 0.05, 0.8], 1e-4), "blues untouched");
        // An identity .cube changes nothing; one that swaps red and blue swaps them.
        let mut cube = String::from("LUT_3D_SIZE 2\n");
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    cube += &format!("{b} {g} {r}\n");
                }
            }
        }
        let (size, data) = parse_cube(&cube).unwrap();
        let lut = Adjustment::ColorLookup { name: "swap".into(), size, data, amount: 100.0, path: None };
        let o = lut.apply([0.6, 0.1, 0.02]);
        assert!(close(o, [0.02, 0.1, 0.6], 0.01), "{o:?}");
        assert!(parse_cube("LUT_3D_SIZE 2\n0 0 0\n").is_err());
        // Auto contrast stretches a narrow histogram to the full range.
        let mut bins = [[0.0f32; 256]; 4];
        for b in bins.iter_mut() {
            for v in &mut b[60..=180] {
                *v = 1.0;
            }
        }
        let Adjustment::Levels { master, .. } = auto_levels(&bins, AutoMode::Contrast) else { panic!() };
        assert!((master.in_black - 60.0 / 255.0).abs() < 0.01 && (master.in_white - 180.0 / 255.0).abs() < 0.01, "{master:?}");
    }

    #[test]
    fn neutral_settings_change_nothing() {
        let c = [0.2, 0.5, 0.05];
        for a in Adjustment::defaults() {
            match a {
                Adjustment::Invert
                | Adjustment::Posterize { .. }
                | Adjustment::Threshold { .. }
                | Adjustment::GradientMap { .. }
                | Adjustment::PhotoFilter { .. }
                | Adjustment::BlackWhite { .. } => continue,
                _ => {}
            }
            assert!(close(a.apply(c), c, 0.01), "{}: {:?}", a.name(), a.apply(c));
        }
    }

    #[test]
    fn color_balance_mixer_and_gradient_map() {
        // More red in the midtones: mid grey turns reddish, keeps its lightness.
        let cb = Adjustment::ColorBalance { shadows: [0.0; 3], midtones: [50.0, 0.0, 0.0], highlights: [0.0; 3], preserve_luminosity: true };
        let grey = [srgb_to_linear(0.5); 3];
        let o = cb.apply(grey).map(linear_to_srgb);
        assert!(o[0] > o[1] + 0.05 && (o[1] - o[2]).abs() < 1e-3, "{o:?}");
        assert!((rgb_to_hsl(o)[2] - 0.5).abs() < 0.01);
        // Black stays black with a midtone shift.
        assert!(cb.apply([0.0; 3])[0] < 0.01);
        // Channel mixer: swap red and blue.
        let m = Adjustment::ChannelMixer { red: [0.0, 0.0, 100.0, 0.0], green: mix_green(), blue: [100.0, 0.0, 0.0, 0.0], gray: mix_gray(), monochrome: false };
        let c = [srgb_to_linear(0.8), srgb_to_linear(0.4), srgb_to_linear(0.1)];
        let o = m.apply(c).map(linear_to_srgb);
        assert!(close(o, [0.1, 0.4, 0.8], 1e-3), "{o:?}");
        // Monochrome: 40/40/20 grey.
        let m = Adjustment::ChannelMixer { red: mix_red(), green: mix_green(), blue: mix_blue(), gray: mix_gray(), monochrome: true };
        let o = m.apply(c).map(linear_to_srgb);
        assert!(close(o, [0.5; 3], 1e-3), "{o:?}");
        // Gradient map black → red: white becomes red, black stays black.
        let g = Adjustment::GradientMap { gradient: crate::gradient::Gradient::two([0.0; 3], [1.0, 0.0, 0.0]), reverse: false };
        assert!(close(g.apply([1.0; 3]), [1.0, 0.0, 0.0], 1e-3));
        assert!(close(g.apply([0.0; 3]), [0.0; 3], 1e-3));
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
        assert_eq!(
            eval_curve(&[[0.0, 1.0], [1.0, 0.0]], 0.0),
            1.0,
            "inverted curve"
        );
    }

    #[test]
    fn levels_invert_threshold_and_bw() {
        let l = Levels {
            in_black: 0.2,
            in_white: 0.8,
            ..Default::default()
        };
        assert_eq!(l.map(0.1), 0.0);
        assert_eq!(l.map(0.9), 1.0);
        assert!((l.map(0.5) - 0.5).abs() < 1e-6);
        let inv = Adjustment::Invert.apply([1.0, 0.0, srgb_to_linear(0.5)]);
        assert!(close(inv, [0.0, 1.0, srgb_to_linear(0.5)], 1e-3));
        let t = Adjustment::Threshold { level: 128.0 };
        assert_eq!(t.apply([0.9, 0.9, 0.9]), [1.0; 3]);
        assert_eq!(t.apply([0.1, 0.1, 0.1]), [0.0; 3]);
        // Black & white: pure red with the red weight 40 % → about 40 % grey.
        let g = Adjustment::BlackWhite {
            weights: bw_default(),
        }
        .apply([1.0, 0.0, 0.0]);
        assert!((linear_to_srgb(g[0]) - 0.4).abs() < 0.01 && g[0] == g[1] && g[1] == g[2]);
    }

    #[test]
    fn hue_rotation_and_desaturation() {
        let h = Adjustment::HueSaturation {
            hue: 120.0,
            saturation: 0.0,
            lightness: 0.0,
            colorize: false,
            ranges: Vec::new(),
        };
        assert!(
            close(h.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], 1e-3),
            "red → green"
        );
        let d = Adjustment::HueSaturation {
            hue: 0.0,
            saturation: -100.0,
            lightness: 0.0,
            colorize: false,
            ranges: Vec::new(),
        };
        let g = d.apply([1.0, 0.0, 0.0]);
        assert!((g[0] - g[1]).abs() < 1e-4 && (g[1] - g[2]).abs() < 1e-4);
        let e = Adjustment::Exposure {
            exposure: 1.0,
            offset: 0.0,
            gamma: 1.0,
        };
        assert!(close(e.apply([0.25, 0.1, 0.0]), [0.5, 0.2, 0.0], 1e-6));
    }

    #[test]
    fn serde_roundtrip_with_defaults() {
        let a: Adjustment =
            serde_json::from_str(r#"{"kind": "curves", "master": [[0,0],[0.5,0.7],[1,1]]}"#)
                .unwrap();
        let Adjustment::Curves { master, red, .. } = &a else {
            panic!()
        };
        assert_eq!(master.len(), 3);
        assert_eq!(red, &identity_curve());
        let s = serde_json::to_string(&a).unwrap();
        assert_eq!(serde_json::from_str::<Adjustment>(&s).unwrap(), a);
        let b: Adjustment = serde_json::from_str(r#"{"kind": "black_white"}"#).unwrap();
        assert_eq!(
            b,
            Adjustment::BlackWhite {
                weights: bw_default()
            }
        );
    }
}
