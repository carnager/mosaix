//! Camera RAW development: decoding and demosaicing via `rawler`, then our
//! own colour step – white balance (as shot, automatic, daylight or a colour
//! temperature), the camera matrix to linear sRGB and exposure. The result is
//! scene-linear and keeps colours outside sRGB (negative components), as the
//! rest of the pipeline does; only blown highlights are clipped to neutral.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rawler::decoders::RawDecodeParams;
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use rawler::imgop::xyz::Illuminant;
use rawler::rawsource::RawSource;
use rawler::{Orientation, RawImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// File extensions treated as camera RAW.
pub const EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "cr2", "cr3", "crw", "dcr", "dcs", "dng", "erf", "iiq", "kdc", "mef", "mos", "mrw", "nef",
    "nrw", "orf", "ori", "pef", "raf", "raw", "rw2", "rwl", "srw", "x3f",
];

pub fn is_raw(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum WhiteBalance {
    /// What the camera recorded.
    AsShot,
    /// Grey world over the unclipped pixels.
    Auto,
    /// Neutral under D65.
    Daylight,
    /// A colour temperature in kelvin; positive tint is more magenta.
    Custom { temperature: f32, tint: f32 },
}

/// How to develop a RAW file.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Develop {
    pub white_balance: WhiteBalance,
    /// In stops.
    pub exposure: f32,
}

impl Default for Develop {
    fn default() -> Self {
        Self { white_balance: WhiteBalance::AsShot, exposure: 0.0 }
    }
}

/// What the file says about the shot.
#[derive(Debug, Clone, Default)]
pub struct Info {
    pub camera: String,
    pub lens: Option<String>,
    pub iso: Option<u32>,
    pub exposure_time: Option<f32>,
    pub aperture: Option<f32>,
    pub focal_length: Option<f32>,
    /// The as-shot white balance as temperature and tint, if the camera stored one.
    pub as_shot: Option<(f32, f32)>,
}

impl Info {
    /// "1/250 s · f/5.6 · ISO 200 · 50 mm"
    pub fn shot(&self) -> String {
        let mut parts = Vec::new();
        if let Some(t) = self.exposure_time {
            parts.push(if t < 0.5 { format!("1/{:.0} s", 1.0 / t) } else { format!("{t:.1} s") });
        }
        if let Some(f) = self.aperture {
            parts.push(format!("f/{f:.1}"));
        }
        if let Some(i) = self.iso {
            parts.push(format!("ISO {i}"));
        }
        if let Some(f) = self.focal_length {
            parts.push(format!("{f:.0} mm"));
        }
        parts.join(" · ")
    }
}

/// Demosaiced sensor data in camera colour space (black and white level
/// applied, cropped, not yet oriented).
#[derive(Clone)]
pub struct CameraImage {
    pub width: usize,
    pub height: usize,
    /// Three or four colour channels per pixel (the fourth is 0 for RGB sensors).
    pub data: Vec<[f32; 4]>,
}

impl CameraImage {
    /// A box-filtered copy whose longer side is at most `max`.
    pub fn downscale(&self, max: usize) -> CameraImage {
        let f = self.width.max(self.height).div_ceil(max.max(1)).max(1);
        if f == 1 {
            return self.clone();
        }
        let (w, h) = (self.width / f, self.height / f);
        let data = (0..w * h)
            .into_par_iter()
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let mut s = [0.0f32; 4];
                for yy in y * f..(y + 1) * f {
                    for p in &self.data[yy * self.width + x * f..yy * self.width + (x + 1) * f] {
                        for c in 0..4 {
                            s[c] += p[c];
                        }
                    }
                }
                s.map(|v| v / (f * f) as f32)
            })
            .collect();
        CameraImage { width: w, height: h, data }
    }
}

/// An opened RAW file.
pub struct RawFile {
    raw: RawImage,
    pub info: Info,
    /// Camera colour channels (3 or 4).
    channels: usize,
    /// Linear sRGB → camera (unnormalised).
    rgb_to_cam: [[f32; 3]; 4],
    /// Camera → linear sRGB.
    cam_to_rgb: [[f32; 4]; 3],
}

const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.072175],
    [0.0193339, 0.119192, 0.9503041],
];
const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.969266, 1.8760108, 0.041556],
    [0.0556434, -0.2040259, 1.0572252],
];

fn mul3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

/// Moore–Penrose inverse of an n×3 matrix (n = 3 or 4): (AᵀA)⁻¹Aᵀ.
fn pseudo_inverse(a: &[[f32; 3]; 4], n: usize) -> [[f32; 4]; 3] {
    let mut ata = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ata[i][j] = (0..n).map(|k| a[k][i] as f64 * a[k][j] as f64).sum();
        }
    }
    let inv = invert3(ata);
    let mut out = [[0.0f32; 4]; 3];
    for i in 0..3 {
        for k in 0..n {
            out[i][k] = (0..3).map(|j| inv[i][j] * a[k][j] as f64).sum::<f64>() as f32;
        }
    }
    out
}

fn invert3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = if det.abs() < 1e-12 { 1e-12 } else { det };
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (a, b) = ((j + 1) % 3, (j + 2) % 3);
            let (c, e) = ((i + 1) % 3, (i + 2) % 3);
            (m[a][c] * m[b][e] - m[a][e] * m[b][c]) / d
        })
    })
}

/// CIE xy of a black body (Kim et al. cubic fit, 1667–25000 K).
fn planckian_xy(t: f32) -> (f32, f32) {
    let t = t.clamp(1667.0, 25000.0) as f64;
    let (t2, t3) = (t * t, t * t * t);
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    (x as f32, y as f32)
}

/// Tint is a factor on the green multiplier: +100 halves it… roughly
/// (2^(−tint/300)); good enough to match the feel of other raw converters.
const TINT_SCALE: f32 = 300.0;

impl RawFile {
    pub fn open(path: &Path) -> Result<Self> {
        let src = RawSource::new(path).with_context(|| path.display().to_string())?;
        let decoder = rawler::get_decoder(&src).map_err(|e| anyhow!("{e}"))?;
        let params = RawDecodeParams::default();
        let raw = decoder.raw_image(&src, &params, false).map_err(|e| anyhow!("{e}"))?;
        let md = decoder.raw_metadata(&src, &params).ok();
        let channels = if raw.cpp == 1 {
            match &raw.photometric {
                rawler::rawimage::RawPhotometricInterpretation::Cfa(c) if c.cfa.unique_colors() == 4 => 4,
                _ => 3,
            }
        } else {
            raw.cpp.clamp(3, 4)
        };
        // XYZ → camera, adapted to D65 when the file only has another illuminant.
        let (illu, m) = raw
            .color_matrix_find_first([
                Illuminant::D65,
                Illuminant::D55,
                Illuminant::D50,
                Illuminant::D75,
                Illuminant::Daylight,
                Illuminant::A,
                Illuminant::B,
                Illuminant::C,
                Illuminant::Flash,
            ])
            .ok_or_else(|| anyhow!("no colour matrix for {} {}", raw.clean_make, raw.clean_model))?;
        let mut xyz_to_cam = [[0.0f32; 3]; 4];
        for (i, row) in m.as_chunks::<3>().0.iter().take(4).enumerate() {
            xyz_to_cam[i] = [row[0], row[1], row[2]];
        }
        if illu != Illuminant::D65 && m.len() == 9 {
            let a = rawler::imgop::chromatic_adaption::adapt_bradford(
                &illu,
                &Illuminant::D65,
                &[xyz_to_cam[0], xyz_to_cam[1], xyz_to_cam[2]],
            );
            xyz_to_cam[..3].copy_from_slice(&a);
        }
        // Linear sRGB → camera; its row sums are the camera's response to D65.
        let mut rgb_to_cam = [[0.0f32; 3]; 4];
        for i in 0..channels {
            for j in 0..3 {
                rgb_to_cam[i][j] = (0..3).map(|k| xyz_to_cam[i][k] * SRGB_TO_XYZ[k][j]).sum();
            }
        }
        // Rows normalised so that white maps to 1 (white balanced data).
        let mut norm = rgb_to_cam;
        for row in norm.iter_mut().take(channels) {
            let s: f32 = row.iter().sum();
            if s.abs() > 1e-9 {
                *row = row.map(|v| v / s);
            }
        }
        let cam_to_rgb = pseudo_inverse(&norm, channels);
        let mut f = Self { raw, info: Info::default(), channels, rgb_to_cam, cam_to_rgb };
        let r = |v: &rawler::formats::tiff::Rational| if v.d == 0 { None } else { Some(v.n as f32 / v.d as f32) };
        f.info = Info {
            camera: format!("{} {}", f.raw.clean_make, f.raw.clean_model).trim().to_string(),
            lens: md.as_ref().and_then(|m| {
                m.lens.as_ref().map(|l| format!("{} {}", l.lens_make, l.lens_model).trim().to_string())
                    .or_else(|| m.exif.lens_model.clone())
            }),
            iso: md.as_ref().and_then(|m| m.exif.iso_speed_ratings.map(u32::from).or(m.exif.iso_speed)),
            exposure_time: md.as_ref().and_then(|m| m.exif.exposure_time.as_ref().and_then(r)),
            aperture: md.as_ref().and_then(|m| m.exif.fnumber.as_ref().and_then(r)),
            focal_length: md.as_ref().and_then(|m| m.exif.focal_length.as_ref().and_then(r)),
            as_shot: None,
        };
        f.info.as_shot = f.as_shot_wb().map(|wb| f.temperature_of(wb));
        Ok(f)
    }

    fn swaps_axes(&self) -> bool {
        matches!(
            self.raw.orientation,
            Orientation::Transpose | Orientation::Rotate90 | Orientation::Transverse | Orientation::Rotate270
        )
    }

    fn as_shot_wb(&self) -> Option<[f32; 4]> {
        let wb = self.raw.wb_coeffs;
        (!wb[0].is_nan() && wb[1] > 0.0 && wb[..self.channels].iter().all(|v| *v > 0.0)).then_some(wb)
    }

    /// White balance multipliers (green = 1) neutralising light whose linear
    /// sRGB colour is `rgb`.
    fn multipliers_for(&self, rgb: [f32; 3]) -> [f32; 4] {
        let mut wb = [1.0f32; 4];
        for (c, w) in wb.iter_mut().enumerate().take(self.channels) {
            let cam: f32 = (0..3).map(|j| self.rgb_to_cam[c][j] * rgb[j]).sum();
            *w = if cam > 1e-6 { 1.0 / cam } else { 1.0 };
        }
        let g = wb[1];
        wb.map(|v| v / g)
    }

    fn temperature_rgb(t: f32) -> [f32; 3] {
        let (x, y) = planckian_xy(t);
        mul3(&XYZ_TO_SRGB, [x / y, 1.0, (1.0 - x - y) / y])
    }

    /// Multipliers for a colour temperature and tint.
    pub fn multipliers_for_temperature(&self, temperature: f32, tint: f32) -> [f32; 4] {
        let mut wb = self.multipliers_for(Self::temperature_rgb(temperature));
        // The green channel(s): index 1, and 3 for RGBG-style four-colour sensors.
        wb[1] *= (-tint / TINT_SCALE).exp2();
        wb
    }

    /// Temperature and tint whose multipliers come closest to `wb`.
    pub fn temperature_of(&self, wb: [f32; 4]) -> (f32, f32) {
        let target = (wb[0] / wb[2]).ln();
        // Red/blue ratio falls monotonically with temperature: bisect in log space.
        let (mut lo, mut hi) = (2000f32.ln(), 25000f32.ln());
        let ratio = |t: f32| {
            let m = self.multipliers_for(Self::temperature_rgb(t));
            (m[0] / m[2]).ln()
        };
        let rising = ratio(25000.0) > ratio(2000.0);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if (ratio(mid.exp()) < target) == rising {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (0.5 * (lo + hi)).exp();
        let m = self.multipliers_for(Self::temperature_rgb(t));
        // Compare green against the mean of red and blue (they matched as a ratio).
        let k = ((wb[0] * wb[2]) / (m[0] * m[2])).sqrt();
        let tint = -TINT_SCALE * (wb[1] / (m[1] * k)).log2();
        (t, tint)
    }

    /// Demosaics the sensor data (the slow part; cache the result).
    pub fn camera_image(&self) -> Result<CameraImage> {
        let steps = [
            ProcessingStep::Rescale,
            ProcessingStep::Demosaic,
            ProcessingStep::FujiRotate,
            ProcessingStep::CropActiveArea,
            ProcessingStep::CropDefault,
        ];
        let im = RawDevelop::new_with(&steps).develop_intermediate(&self.raw).map_err(|e| anyhow!("{e}"))?;
        Ok(match im {
            Intermediate::Monochrome(p) => CameraImage {
                width: p.width,
                height: p.height,
                data: p.data.iter().map(|v| [*v, *v, *v, 0.0]).collect(),
            },
            Intermediate::ThreeColor(p) => CameraImage {
                width: p.width,
                height: p.height,
                data: p.data.iter().map(|v| [v[0], v[1], v[2], 0.0]).collect(),
            },
            Intermediate::FourColor(p) => CameraImage { width: p.width, height: p.height, data: p.data },
        })
    }

    /// The multipliers a setting stands for, normalised so the smallest is 1
    /// (then every channel clips at 1 and blown highlights stay neutral).
    pub fn multipliers(&self, wb: WhiteBalance, cam: &CameraImage) -> [f32; 4] {
        let m = match wb {
            WhiteBalance::AsShot => self.as_shot_wb().unwrap_or_else(|| self.multipliers_for([1.0; 3])),
            WhiteBalance::Daylight => self.multipliers_for([1.0; 3]),
            WhiteBalance::Custom { temperature, tint } => self.multipliers_for_temperature(temperature, tint),
            WhiteBalance::Auto => {
                // Grey world over pixels without a clipped channel.
                let step = (cam.data.len() / 200_000).max(1);
                let mut sum = [0.0f64; 4];
                for p in cam.data.iter().step_by(step) {
                    if p[..self.channels].iter().all(|v| *v < 0.95 && *v > 0.002) {
                        for c in 0..4 {
                            sum[c] += p[c] as f64;
                        }
                    }
                }
                let mut m = [1.0f32; 4];
                for c in 0..self.channels {
                    if sum[c] > 0.0 {
                        m[c] = (sum[1] / sum[c]) as f32;
                    }
                }
                m
            }
        };
        let min = m[..self.channels].iter().cloned().fold(f32::MAX, f32::min).max(1e-6);
        m.map(|v| v / min)
    }

    /// Develops a camera image to straight RGBA, linear sRGB (extended range),
    /// oriented. Returns (width, height, pixels).
    pub fn develop(&self, cam: &CameraImage, d: &Develop) -> (u32, u32, Vec<f32>) {
        let wb = self.multipliers(d.white_balance, cam);
        let gain = d.exposure.exp2();
        let n = self.channels;
        let m = self.cam_to_rgb;
        let (w, h) = (cam.width, cam.height);
        let swap = self.swaps_axes();
        let (ow, oh) = if swap { (h, w) } else { (w, h) };
        let orient = self.raw.orientation;
        let mut out = vec![0.0f32; ow * oh * 4];
        out.par_chunks_mut(ow * 4).enumerate().for_each(|(oy, row)| {
            for ox in 0..ow {
                // Output pixel → source pixel.
                let (sx, sy) = match orient {
                    Orientation::HorizontalFlip => (w - 1 - ox, oy),
                    Orientation::Rotate180 => (w - 1 - ox, h - 1 - oy),
                    Orientation::VerticalFlip => (ox, h - 1 - oy),
                    Orientation::Transpose => (oy, ox),
                    Orientation::Rotate90 => (oy, h - 1 - ox),
                    Orientation::Transverse => (w - 1 - oy, h - 1 - ox),
                    Orientation::Rotate270 => (w - 1 - oy, ox),
                    _ => (ox, oy),
                };
                let p = cam.data[sy * w + sx];
                let mut c = [0.0f32; 4];
                for k in 0..n {
                    c[k] = (p[k] * wb[k]).min(1.0);
                }
                let o = &mut row[ox * 4..ox * 4 + 4];
                for r in 0..3 {
                    o[r] = (0..n).map(|k| m[r][k] * c[k]).sum::<f32>() * gain;
                }
                o[3] = 1.0;
            }
        });
        (ow as u32, oh as u32, out)
    }
}

/// Opens and develops a RAW file in one go.
pub fn load(path: &Path, d: &Develop) -> Result<(u32, u32, Vec<f32>, Info)> {
    let f = RawFile::open(path)?;
    let cam = f.camera_image()?;
    let (w, h, px) = f.develop(&cam, d);
    Ok((w, h, px, f.info.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices() {
        let a = [[0.8, 0.3, -0.1], [0.1, 1.0, -0.1], [0.0, 0.2, 0.8], [0.0; 3]];
        let p = pseudo_inverse(&a, 3);
        for (i, pi) in p.iter().enumerate() {
            for j in 0..3 {
                let v: f32 = pi.iter().zip(&a).map(|(p, r)| p * r[j]).sum();
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-4, "{i}{j} {v}");
            }
        }
        // Temperatures: warm light is red, D65-ish is neutral.
        let warm = RawFile::temperature_rgb(2800.0);
        assert!(warm[0] > warm[2] * 2.0);
        let d = RawFile::temperature_rgb(6504.0);
        assert!((d[0] / d[2] - 1.0).abs() < 0.1, "{d:?}");
        assert!(is_raw(Path::new("a/B.CR3")) && !is_raw(Path::new("x.png")));
    }
}
