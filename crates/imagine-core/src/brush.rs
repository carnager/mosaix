//! Turns raw input samples (mouse / stylus) into evenly spaced brush dabs,
//! with brush tips (round or textured), shape dynamics (size / angle /
//! flow jitter, scatter) and an optional smudge mode.

use bytemuck::{Pod, Zeroable};
use glam::DVec2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushSettings {
    /// Diameter in document pixels at full pressure.
    pub size: f32,
    /// 0 = soft, 1 = hard edge.
    pub hardness: f32,
    /// Alpha deposited per dab at full pressure (builds up within a stroke).
    pub flow: f32,
    /// Maximum coverage of one stroke (applied when the stroke is composited).
    pub opacity: f32,
    /// Dab distance as a fraction of the current diameter.
    pub spacing: f32,
    /// Diameter at zero pressure, as a fraction of `size`.
    pub min_size: f32,
    /// 0 = raw input, towards 1 = heavy smoothing.
    pub smoothing: f32,
    /// Linear-light RGB, straight alpha.
    pub color: [f32; 3],
    pub eraser: bool,
    /// 0 = round (uses `hardness`), 1… = textured tip (see [`TIP_NAMES`]).
    pub tip: u32,
    /// Tip rotation in degrees.
    pub angle: f32,
    /// 1 = circular, smaller = flattened ellipse.
    pub roundness: f32,
    /// Random size reduction per dab, 0…1.
    pub size_jitter: f32,
    /// Random rotation per dab, degrees (0…180).
    pub angle_jitter: f32,
    /// Random offset across the stroke, in diameters (0…5).
    pub scatter: f32,
    /// Random flow reduction per dab, 0…1.
    pub flow_jitter: f32,
    /// Rotate the tip along the stroke direction.
    pub follow_direction: bool,
    /// Smudge instead of paint: `Some(strength)` 0…1 (1 = drag colour indefinitely).
    pub smudge: Option<f32>,
}

/// Identifiers of the built-in tips, index = `tip` (see [`tip_label`] for display).
pub const TIP_NAMES: [&str; 6] = ["round", "chalk", "bristles", "spray", "rough", "charcoal"];

/// Display name of a tip, translated.
pub fn tip_label(tip: u32) -> &'static str {
    match tip {
        0 => crate::tr("round"),
        1 => crate::tr("chalk"),
        2 => crate::tr("bristles"),
        3 => crate::tr("spray"),
        4 => crate::tr("rough"),
        _ => crate::tr("charcoal"),
    }
}

/// Index of a tip by name (English or German).
pub fn tip_from_name(name: &str) -> Option<u32> {
    let de = ["rund", "kreide", "borsten", "spray", "rau", "kohle"];
    TIP_NAMES
        .iter()
        .position(|n| *n == name)
        .or_else(|| de.iter().position(|n| *n == name))
        .map(|i| i as u32)
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            size: 40.0,
            hardness: 0.8,
            flow: 0.25,
            opacity: 1.0,
            spacing: 0.1,
            min_size: 0.15,
            smoothing: 0.4,
            color: [0.02, 0.02, 0.025],
            eraser: false,
            tip: 0,
            angle: 0.0,
            roundness: 1.0,
            size_jitter: 0.0,
            angle_jitter: 0.0,
            scatter: 0.0,
            flow_jitter: 0.0,
            follow_direction: false,
            smudge: None,
        }
    }
}

/// One input sample in document space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InputSample {
    pub pos: DVec2,
    /// 0..=1. Mice report 1.
    pub pressure: f32,
}

/// A single brush stamp, laid out for direct upload to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Dab {
    pub center: [f32; 2],
    pub radius: f32,
    pub hardness: f32,
    /// Linear RGB + alpha (alpha = flow at this dab; smudge: strength).
    pub color: [f32; 4],
    /// Rotation in radians.
    pub angle: f32,
    pub roundness: f32,
    /// 0 = round, else textured tip index.
    pub tip: u32,
    pub _pad: f32,
}

impl Dab {
    /// Bounding box (min, max) in document pixels.
    pub fn bounds(&self) -> ((f64, f64), (f64, f64)) {
        let r = self.radius as f64 + 1.0;
        let (x, y) = (self.center[0] as f64, self.center[1] as f64);
        ((x - r, y - r), (x + r, y + r))
    }
}

/// An in-progress stroke.
#[derive(Debug, Clone)]
pub struct Stroke {
    pub settings: BrushSettings,
    /// Smoothed position and pressure of the last processed point.
    last: Option<(DVec2, f32)>,
    /// Distance along the path until the next dab is due.
    to_next: f64,
    rng: u32,
}

impl Stroke {
    pub fn new(settings: BrushSettings) -> Self {
        Self {
            settings,
            last: None,
            to_next: 0.0,
            rng: 0x9e37_79b9,
        }
    }

    /// Uniform random number in [0, 1) (xorshift; deterministic per stroke).
    fn random(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    fn diameter(&self, pressure: f32) -> f32 {
        let s = &self.settings;
        s.size * (s.min_size + (1.0 - s.min_size) * pressure.clamp(0.0, 1.0))
    }

    fn dab(&mut self, pos: DVec2, pressure: f32, direction: f64) -> Dab {
        let s = self.settings;
        let p = pressure.clamp(0.0, 1.0);
        let mut radius = self.diameter(p) * 0.5;
        let mut angle = s.angle.to_radians()
            + if s.follow_direction {
                direction as f32
            } else {
                0.0
            };
        let mut flow = s.flow * (0.3 + 0.7 * p);
        let mut pos = pos;
        if s.size_jitter > 0.0 {
            radius *= 1.0 - s.size_jitter.clamp(0.0, 1.0) * self.random();
        }
        if s.angle_jitter > 0.0 {
            angle += (self.random() * 2.0 - 1.0) * s.angle_jitter.to_radians();
        }
        if s.flow_jitter > 0.0 {
            flow *= 1.0 - s.flow_jitter.clamp(0.0, 1.0) * self.random();
        }
        if s.scatter > 0.0 {
            // Across the stroke direction, up to ±scatter diameters.
            let off = (self.random() * 2.0 - 1.0) * s.scatter * self.diameter(p);
            pos += DVec2::new(-direction.sin(), direction.cos()) * off as f64;
        }
        let alpha = match s.smudge {
            Some(strength) => strength.clamp(0.0, 1.0),
            None => flow,
        };
        Dab {
            center: [pos.x as f32, pos.y as f32],
            radius: radius.max(0.5),
            hardness: s.hardness,
            color: [s.color[0], s.color[1], s.color[2], alpha],
            angle,
            roundness: s.roundness.clamp(0.05, 1.0),
            tip: s.tip,
            _pad: 0.0,
        }
    }

    /// Feeds a sample and returns the dabs it produced.
    pub fn add(&mut self, sample: InputSample) -> Vec<Dab> {
        let target = match self.last {
            None => (sample.pos, sample.pressure),
            Some((p, pr)) => {
                let k = 1.0 - self.settings.smoothing.clamp(0.0, 0.95) as f64;
                (
                    p + (sample.pos - p) * k,
                    pr + (sample.pressure - pr) * k as f32,
                )
            }
        };
        self.advance_to(target.0, target.1)
    }

    /// Ends the stroke, catching up with the raw end position that smoothing lagged behind.
    pub fn finish(&mut self, end: InputSample) -> Vec<Dab> {
        if self.last.is_none() {
            return Vec::new();
        }
        self.advance_to(end.pos, end.pressure)
    }

    fn advance_to(&mut self, pos: DVec2, pressure: f32) -> Vec<Dab> {
        let mut dabs = Vec::new();
        let Some((from, from_pr)) = self.last else {
            dabs.push(self.dab(pos, pressure, 0.0));
            self.last = Some((pos, pressure));
            self.to_next = self.spacing_px(pressure);
            return dabs;
        };
        let seg = pos - from;
        let len = seg.length();
        let direction = seg.y.atan2(seg.x);
        let mut d = self.to_next;
        while d <= len && len > 0.0 {
            let t = d / len;
            let pr = from_pr + (pressure - from_pr) * t as f32;
            let dab = self.dab(from + seg * t, pr, direction);
            dabs.push(dab);
            d += self.spacing_px(pr);
        }
        self.to_next = d - len;
        self.last = Some((pos, pressure));
        dabs
    }

    fn spacing_px(&self, pressure: f32) -> f64 {
        (self.diameter(pressure) * self.settings.spacing).max(0.5) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw() -> BrushSettings {
        BrushSettings {
            smoothing: 0.0,
            size: 10.0,
            spacing: 0.1,
            min_size: 1.0,
            ..Default::default()
        }
    }

    fn s(x: f64, y: f64) -> InputSample {
        InputSample {
            pos: DVec2::new(x, y),
            pressure: 1.0,
        }
    }

    #[test]
    fn first_sample_places_a_dab() {
        let mut st = Stroke::new(raw());
        let d = st.add(s(5.0, 5.0));
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].center, [5.0, 5.0]);
        assert_eq!(d[0].radius, 5.0);
    }

    #[test]
    fn dabs_are_evenly_spaced_across_samples() {
        let mut st = Stroke::new(raw());
        let mut dabs = st.add(s(0.0, 0.0));
        for i in 1..=20 {
            dabs.extend(st.add(s(i as f64 * 0.73, 0.0)));
        }
        // 14.6 px at 1 px spacing → dabs at 0..=14.
        assert_eq!(dabs.len(), 15);
        for w in dabs.windows(2) {
            assert!(((w[1].center[0] - w[0].center[0]) - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn smoothing_lags_but_finish_catches_up() {
        let mut st = Stroke::new(BrushSettings {
            smoothing: 0.8,
            ..raw()
        });
        st.add(s(0.0, 0.0));
        let mid = st.add(s(100.0, 0.0));
        assert!(mid.last().unwrap().center[0] < 50.0);
        let end = st.finish(s(100.0, 0.0));
        assert!(end.last().unwrap().center[0] > 99.0);
    }

    #[test]
    fn jitter_varies_dabs_within_limits_and_is_deterministic() {
        let settings = BrushSettings {
            size_jitter: 0.5,
            angle_jitter: 90.0,
            scatter: 1.0,
            flow_jitter: 0.5,
            ..raw()
        };
        let run = || {
            let mut st = Stroke::new(settings);
            let mut d = st.add(s(0.0, 0.0));
            d.extend(st.add(s(50.0, 0.0)));
            d
        };
        let a = run();
        assert_eq!(a, run(), "same input, same dabs");
        assert!(a.iter().all(|d| d.radius <= 5.0 && d.radius >= 2.5));
        assert!(
            a.iter().all(|d| d.center[1].abs() <= 10.0),
            "scatter across the stroke"
        );
        assert!(a.iter().any(|d| d.center[1].abs() > 1.0));
        assert!(a.iter().map(|d| d.angle).any(|x| x.abs() > 0.1));
    }

    #[test]
    fn follow_direction_and_tip_names() {
        let mut st = Stroke::new(BrushSettings {
            follow_direction: true,
            tip: 2,
            ..raw()
        });
        st.add(s(0.0, 0.0));
        let d = st.add(s(0.0, 10.0));
        assert!((d[0].angle - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert_eq!(d[0].tip, 2);
        assert_eq!(tip_from_name("spray"), Some(3));
        assert_eq!(tip_from_name("kohle"), Some(5));
        assert_eq!(tip_from_name("nope"), None);
    }

    #[test]
    fn pressure_scales_size() {
        let mut st = Stroke::new(BrushSettings {
            min_size: 0.2,
            ..raw()
        });
        let d = st.add(InputSample {
            pos: DVec2::ZERO,
            pressure: 0.0,
        });
        assert!((d[0].radius - 1.0).abs() < 1e-6);
    }
}
