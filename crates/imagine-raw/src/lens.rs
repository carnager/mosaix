//! Lens correction from the lensfun database (read from the system, e.g.
//! /usr/share/lensfun): distortion, transverse chromatic aberration (colour
//! fringes) and vignetting, interpolated for the shot's focal length and
//! aperture.
//!
//! Coordinates follow lensfun: distortion and TCA radii are normalised to
//! half the short side of the calibration sensor, vignetting radii to half
//! its diagonal. A camera with another crop factor sees a part of that.

use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distortion {
    /// Rd = Ru·(1 − k1 + k1·Ru²)
    Poly3(f32),
    /// Rd = Ru·(1 + k1·Ru² + k2·Ru⁴)
    Poly5(f32, f32),
    /// Rd = Ru·(a·Ru³ + b·Ru² + c·Ru + 1 − a − b − c)
    PtLens(f32, f32, f32),
}

impl Distortion {
    /// Distorted radius / undistorted radius.
    fn factor(&self, r: f32) -> f32 {
        match *self {
            Distortion::Poly3(k1) => 1.0 - k1 + k1 * r * r,
            Distortion::Poly5(k1, k2) => 1.0 + k1 * r * r + k2 * r.powi(4),
            Distortion::PtLens(a, b, c) => a * r.powi(3) + b * r * r + c * r + 1.0 - a - b - c,
        }
    }

    fn params(&self) -> [f32; 3] {
        match *self {
            Distortion::Poly3(k) => [k, 0.0, 0.0],
            Distortion::Poly5(a, b) => [a, b, 0.0],
            Distortion::PtLens(a, b, c) => [a, b, c],
        }
    }

    fn with(&self, p: [f32; 3]) -> Self {
        match self {
            Distortion::Poly3(_) => Distortion::Poly3(p[0]),
            Distortion::Poly5(..) => Distortion::Poly5(p[0], p[1]),
            Distortion::PtLens(..) => Distortion::PtLens(p[0], p[1], p[2]),
        }
    }
}

/// Red and blue: Rd = Ru·(b·Ru² + c·Ru + v) (linear: b = c = 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tca {
    pub red: [f32; 3],
    pub blue: [f32; 3],
}

#[derive(Debug, Clone)]
struct Vignetting {
    focal: f32,
    aperture: f32,
    distance: f32,
    k: [f32; 3],
}

#[derive(Debug, Clone)]
struct Lens {
    maker: String,
    models: Vec<String>,
    mounts: Vec<String>,
    crop: f32,
    /// Long / short side of the calibration sensor.
    aspect: f32,
    distortion: Vec<(f32, Distortion)>,
    tca: Vec<(f32, Tca)>,
    vignetting: Vec<Vignetting>,
}

#[derive(Debug, Clone)]
struct Camera {
    maker: String,
    models: Vec<String>,
    mount: String,
    crop: f32,
}

#[derive(Default)]
struct Db {
    cameras: Vec<Camera>,
    lenses: Vec<Lens>,
}

fn dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(d) = std::env::var("IMAGINE_LENSFUN_DIR") {
        v.push(PathBuf::from(d));
    }
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
    if let Some(d) = data {
        // lensfun-update-data puts newer data here.
        v.push(d.join("lensfun/updates/version_1"));
        v.push(d.join("lensfun/version_1"));
    }
    v.push("/var/lib/lensfun-updates/version_1".into());
    v.push("/usr/share/lensfun/version_1".into());
    v.push("/usr/local/share/lensfun/version_1".into());
    v
}

fn norm(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

fn attr(n: &roxmltree::Node, a: &str) -> Option<f32> {
    n.attribute(a).and_then(|v| v.parse().ok())
}

fn text(n: &roxmltree::Node, tag: &str) -> Option<String> {
    n.children().find(|c| c.has_tag_name(tag)).and_then(|c| c.text()).map(|t| t.trim().to_string())
}

fn parse(xml: &str, db: &mut Db) {
    let Ok(doc) = roxmltree::Document::parse(xml) else { return };
    for n in doc.root_element().children().filter(|n| n.is_element()) {
        let models: Vec<String> = n
            .children()
            .filter(|c| c.has_tag_name("model"))
            .filter_map(|c| c.text().map(|t| t.trim().to_string()))
            .collect();
        let maker = text(&n, "maker").unwrap_or_default();
        let crop = text(&n, "cropfactor").and_then(|t| t.parse().ok()).unwrap_or(1.0);
        if n.has_tag_name("camera") {
            db.cameras.push(Camera { maker, models, mount: text(&n, "mount").unwrap_or_default(), crop });
        } else if n.has_tag_name("lens") {
            let aspect = text(&n, "aspect-ratio")
                .and_then(|a| {
                    let (x, y) = a.split_once(':')?;
                    Some(x.parse::<f32>().ok()? / y.parse::<f32>().ok()?)
                })
                .map(|a| if a < 1.0 { 1.0 / a } else { a })
                .unwrap_or(1.5);
            let mut lens = Lens {
                maker,
                models,
                mounts: n
                    .children()
                    .filter(|c| c.has_tag_name("mount"))
                    .filter_map(|c| c.text().map(|t| t.trim().to_string()))
                    .collect(),
                crop,
                aspect,
                distortion: Vec::new(),
                tca: Vec::new(),
                vignetting: Vec::new(),
            };
            for cal in n.children().filter(|c| c.has_tag_name("calibration")) {
                for e in cal.children().filter(|c| c.is_element()) {
                    let Some(focal) = attr(&e, "focal") else { continue };
                    let model = e.attribute("model").unwrap_or("");
                    let a = |k: &str| attr(&e, k).unwrap_or(0.0);
                    match (e.tag_name().name(), model) {
                        ("distortion", "poly3") => lens.distortion.push((focal, Distortion::Poly3(a("k1")))),
                        ("distortion", "poly5") => lens.distortion.push((focal, Distortion::Poly5(a("k1"), a("k2")))),
                        ("distortion", "ptlens") => {
                            lens.distortion.push((focal, Distortion::PtLens(a("a"), a("b"), a("c"))))
                        }
                        ("tca", "linear" | "poly3") => {
                            let v = |k: &str| attr(&e, k).unwrap_or(1.0);
                            lens.tca.push((
                                focal,
                                Tca { red: [a("br"), a("cr"), v("vr")], blue: [a("bb"), a("cb"), v("vb")] },
                            ))
                        }
                        ("vignetting", "pa") => lens.vignetting.push(Vignetting {
                            focal,
                            aperture: attr(&e, "aperture").unwrap_or(8.0),
                            distance: attr(&e, "distance").unwrap_or(1000.0),
                            k: [a("k1"), a("k2"), a("k3")],
                        }),
                        _ => {}
                    }
                }
            }
            lens.distortion.sort_by(|a, b| a.0.total_cmp(&b.0));
            lens.tca.sort_by(|a, b| a.0.total_cmp(&b.0));
            db.lenses.push(lens);
        }
    }
}

fn db() -> &'static Db {
    static DB: OnceLock<Db> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = Db::default();
        // The first directory with data wins (updates before the system copy).
        for d in dirs() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            let mut files: Vec<PathBuf> =
                rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "xml")).collect();
            if files.is_empty() {
                continue;
            }
            files.sort();
            for f in files {
                if let Ok(x) = std::fs::read_to_string(&f) {
                    parse(&x, &mut db);
                }
            }
            break;
        }
        db
    })
}

/// Whether a lens database was found at all.
pub fn available() -> bool {
    !db().lenses.is_empty()
}

/// Linear interpolation of per-focal-length entries.
fn interp<T: Copy>(v: &[(f32, T)], f: f32, mix: impl Fn(T, T, f32) -> T) -> Option<T> {
    let i = v.iter().position(|e| e.0 >= f);
    Some(match i {
        None => v.last()?.1,
        Some(0) => v[0].1,
        Some(i) => {
            let (a, b) = (v[i - 1], v[i]);
            mix(a.1, b.1, (f - a.0) / (b.0 - a.0).max(1e-6))
        }
    })
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// The corrections for one shot.
#[derive(Debug, Clone, PartialEq)]
pub struct LensProfile {
    /// The matched lens, e.g. "Sony FE 16-35mm f/2.8 GM".
    pub name: String,
    pub distortion: Option<Distortion>,
    pub tca: Option<Tca>,
    pub vignetting: Option<[f32; 3]>,
    /// Calibration sensor size relative to this camera's (lens crop / camera crop is
    /// its inverse), and the calibration aspect ratio.
    crop_ratio: f32,
    aspect: f32,
}

/// Finds the profile for a camera and lens at a focal length and aperture.
pub fn find(camera_make: &str, camera_model: &str, lens: &str, focal: Option<f32>, aperture: Option<f32>) -> Option<LensProfile> {
    let db = db();
    let q = norm(lens);
    if q.len() < 4 {
        return None;
    }
    let cm = norm(camera_model);
    let camera = db
        .cameras
        .iter()
        .find(|c| c.models.iter().any(|m| norm(m) == cm) && norm(camera_make).contains(&norm(&c.maker).chars().take(4).collect::<String>()));
    // Best lens: exact name (with or without maker) first, then the closest
    // containing match; a lens for the camera's mount wins ties.
    let mut best: Option<(i32, &Lens, &String)> = None;
    for l in &db.lenses {
        if l.distortion.is_empty() && l.tca.is_empty() && l.vignetting.is_empty() {
            continue;
        }
        for m in &l.models {
            let a = norm(m);
            let b = norm(&format!("{} {}", l.maker, m));
            let mut score = if a == q || b == q {
                1000
            } else if a.len() >= 8 && (q.contains(&a) || a.contains(&q) || b.contains(&q) || q.contains(&b)) {
                500 - (a.len() as i32 - q.len() as i32).abs()
            } else {
                continue;
            };
            if camera.is_some_and(|c| l.mounts.contains(&c.mount)) {
                score += 50;
            }
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, l, m));
            }
        }
    }
    let (_, l, model) = best?;
    let cam_crop = camera.map_or(l.crop, |c| c.crop);
    let f = focal.unwrap_or_else(|| l.distortion.first().map_or(50.0, |d| d.0));
    let distortion = interp(&l.distortion, f, |a, b, t| {
        if std::mem::discriminant(&a) == std::mem::discriminant(&b) { a.with(lerp3(a.params(), b.params(), t)) } else { a }
    });
    let tca = interp(&l.tca, f, |a, b, t| Tca { red: lerp3(a.red, b.red, t), blue: lerp3(a.blue, b.blue, t) });
    let vignetting = vignetting_at(&l.vignetting, f, aperture.unwrap_or(8.0));
    let name = if model.to_lowercase().starts_with(&l.maker.to_lowercase()) { model.clone() } else { format!("{} {}", l.maker, model) };
    Some(LensProfile { name, distortion, tca, vignetting, crop_ratio: l.crop / cam_crop, aspect: l.aspect })
}

/// Vignetting at the farthest calibrated distance, interpolated in focal
/// length and (log) aperture.
fn vignetting_at(v: &[Vignetting], focal: f32, aperture: f32) -> Option<[f32; 3]> {
    let far = v.iter().map(|e| e.distance).fold(f32::MIN, f32::max);
    let far: Vec<&Vignetting> = v.iter().filter(|e| e.distance == far).collect();
    let mut focals: Vec<f32> = far.iter().map(|e| e.focal).collect();
    focals.sort_by(|a, b| a.total_cmp(b));
    focals.dedup();
    let at_focal = |fl: f32| -> Option<[f32; 3]> {
        let mut pts: Vec<(f32, [f32; 3])> =
            far.iter().filter(|e| e.focal == fl).map(|e| (e.aperture.max(0.5).log2(), e.k)).collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        interp(&pts, aperture.max(0.5).log2(), lerp3)
    };
    let pts: Vec<(f32, [f32; 3])> = focals.iter().filter_map(|f| at_focal(*f).map(|k| (*f, k))).collect();
    interp(&pts, focal, lerp3)
}

/// Lens correction set up for an image of `w × h` pixels (unrotated sensor
/// orientation).
pub struct Corrector {
    p: LensProfile,
    cx: f32,
    cy: f32,
    /// Pixels → distortion units, and → vignetting units.
    to_d: f32,
    to_v: f32,
    /// Zoom that keeps the corrected image free of empty borders.
    pub zoom: f32,
    pub distortion: bool,
    pub tca: bool,
    pub vignetting: bool,
}

impl Corrector {
    pub fn new(p: LensProfile, w: usize, h: usize) -> Self {
        // Half diagonal of this image in pixels = half diagonal of this
        // sensor; the calibration sensor's half diagonal is crop_ratio⁻¹ times… its
        // half short side follows from its aspect ratio.
        let half_diag = 0.5 * ((w * w + h * h) as f32).sqrt();
        let cal_half_diag = half_diag / p.crop_ratio;
        let cal_half_short = cal_half_diag / (1.0 + p.aspect * p.aspect).sqrt();
        let mut c = Corrector {
            cx: w as f32 * 0.5,
            cy: h as f32 * 0.5,
            to_d: 1.0 / cal_half_short,
            to_v: 1.0 / cal_half_diag,
            zoom: 1.0,
            distortion: p.distortion.is_some(),
            tca: p.tca.is_some(),
            vignetting: p.vignetting.is_some(),
            p,
        };
        c.zoom = c.auto_zoom(w as f32, h as f32);
        c
    }

    pub fn name(&self) -> &str {
        &self.p.name
    }

    /// Smallest zoom at which every border pixel of the output samples
    /// inside the image (all three colours).
    fn auto_zoom(&self, w: f32, h: f32) -> f32 {
        if !self.distortion && !self.tca {
            return 1.0;
        }
        let mut border = Vec::new();
        for i in 0..=32 {
            let t = i as f32 / 32.0;
            border.extend([(t * w, 0.0), (t * w, h), (0.0, t * h), (w, t * h)]);
        }
        let inside = |z: f32| {
            border.iter().all(|&(x, y)| {
                (0..3).all(|ch| {
                    let (sx, sy) = self.source(x, y, ch, z);
                    (-0.5..=w + 0.5).contains(&sx) && (-0.5..=h + 0.5).contains(&sy)
                })
            })
        };
        let (mut lo, mut hi) = (0.5f32, 2.0f32);
        if !inside(hi) {
            return 1.0;
        }
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if inside(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// Where the corrected pixel position (x, y) comes from in the image,
    /// for colour channel `ch` (0 red, 1 green, 2 blue).
    pub fn source(&self, x: f32, y: f32, ch: usize, zoom: f32) -> (f32, f32) {
        let (dx, dy) = ((x - self.cx) / zoom, (y - self.cy) / zoom);
        let ru = (dx * dx + dy * dy).sqrt() * self.to_d;
        let mut f = if self.distortion { self.p.distortion.map_or(1.0, |d| d.factor(ru)) } else { 1.0 };
        if self.tca
            && let Some(t) = self.p.tca
            && ch != 1
        {
            let k = if ch == 0 { t.red } else { t.blue };
            let rd = ru * f;
            f *= k[0] * rd * rd + k[1] * rd + k[2];
        }
        (self.cx + dx * f, self.cy + dy * f)
    }

    /// Brightness factor that undoes vignetting at image position (x, y).
    pub fn vignetting_gain(&self, x: f32, y: f32) -> f32 {
        if !self.vignetting {
            return 1.0;
        }
        let Some(k) = self.p.vignetting else { return 1.0 };
        let (dx, dy) = (x - self.cx, y - self.cy);
        let r2 = (dx * dx + dy * dy) * self.to_v * self.to_v;
        let v = 1.0 + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 * r2 * r2;
        1.0 / v.max(0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn database_and_models() {
        let d = Distortion::PtLens(0.01, -0.04, 0.03);
        assert!((d.factor(1.0) - 1.0).abs() < 1e-6, "ptlens keeps r = 1");
        assert!((Distortion::Poly3(-0.02).factor(0.0) - 1.02).abs() < 1e-6);
        if !available() {
            return;
        }
        let p = find("Sony", "ILCE-7M3", "Sony FE 16-35mm F2.8 GM", Some(16.0), Some(4.0)).expect("lens found");
        assert!(p.name.contains("16-35"), "{}", p.name);
        assert!(p.distortion.is_some() && p.vignetting.is_some());
        let c = Corrector::new(p, 6000, 4000);
        // Wide angle at 16 mm: corners darker (gain > 1) and the centre untouched.
        assert!(c.vignetting_gain(0.0, 0.0) > 1.05);
        assert!((c.vignetting_gain(3000.0, 2000.0) - 1.0).abs() < 1e-6);
        assert!((0.8..1.25).contains(&c.zoom), "{}", c.zoom);
        assert!(find("Sony", "ILCE-7M3", "No Such Lens 12mm", None, None).is_none());
    }
}
