//! Turns raw input samples (mouse / stylus) into evenly spaced brush dabs.

use bytemuck::{Pod, Zeroable};
use glam::DVec2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrushSettings {
    /// Diameter in document pixels at full pressure.
    pub size: f32,
    /// 0 = soft, 1 = hard edge.
    pub hardness: f32,
    /// Alpha deposited per dab at full pressure.
    pub flow: f32,
    /// Dab distance as a fraction of the current diameter.
    pub spacing: f32,
    /// Diameter at zero pressure, as a fraction of `size`.
    pub min_size: f32,
    /// 0 = raw input, towards 1 = heavy smoothing.
    pub smoothing: f32,
    /// Linear-light RGB, straight alpha.
    pub color: [f32; 3],
    pub eraser: bool,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            size: 40.0,
            hardness: 0.8,
            flow: 0.25,
            spacing: 0.1,
            min_size: 0.15,
            smoothing: 0.4,
            color: [0.02, 0.02, 0.025],
            eraser: false,
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
    /// Linear RGB + alpha (alpha = flow at this dab).
    pub color: [f32; 4],
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
}

impl Stroke {
    pub fn new(settings: BrushSettings) -> Self {
        Self { settings, last: None, to_next: 0.0 }
    }

    fn diameter(&self, pressure: f32) -> f32 {
        let s = &self.settings;
        s.size * (s.min_size + (1.0 - s.min_size) * pressure.clamp(0.0, 1.0))
    }

    fn dab(&self, pos: DVec2, pressure: f32) -> Dab {
        let s = &self.settings;
        let p = pressure.clamp(0.0, 1.0);
        Dab {
            center: [pos.x as f32, pos.y as f32],
            radius: (self.diameter(p) * 0.5).max(0.5),
            hardness: s.hardness,
            color: [s.color[0], s.color[1], s.color[2], s.flow * (0.3 + 0.7 * p)],
        }
    }

    /// Feeds a sample and returns the dabs it produced.
    pub fn add(&mut self, sample: InputSample) -> Vec<Dab> {
        let target = match self.last {
            None => (sample.pos, sample.pressure),
            Some((p, pr)) => {
                let k = 1.0 - self.settings.smoothing.clamp(0.0, 0.95) as f64;
                (p + (sample.pos - p) * k, pr + (sample.pressure - pr) * k as f32)
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
            dabs.push(self.dab(pos, pressure));
            self.last = Some((pos, pressure));
            self.to_next = self.spacing_px(pressure);
            return dabs;
        };
        let seg = pos - from;
        let len = seg.length();
        let mut d = self.to_next;
        while d <= len && len > 0.0 {
            let t = d / len;
            let pr = from_pr + (pressure - from_pr) * t as f32;
            dabs.push(self.dab(from + seg * t, pr));
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
        BrushSettings { smoothing: 0.0, size: 10.0, spacing: 0.1, min_size: 1.0, ..Default::default() }
    }

    fn s(x: f64, y: f64) -> InputSample {
        InputSample { pos: DVec2::new(x, y), pressure: 1.0 }
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
        let mut st = Stroke::new(BrushSettings { smoothing: 0.8, ..raw() });
        st.add(s(0.0, 0.0));
        let mid = st.add(s(100.0, 0.0));
        assert!(mid.last().unwrap().center[0] < 50.0);
        let end = st.finish(s(100.0, 0.0));
        assert!(end.last().unwrap().center[0] > 99.0);
    }

    #[test]
    fn pressure_scales_size() {
        let mut st = Stroke::new(BrushSettings { min_size: 0.2, ..raw() });
        let d = st.add(InputSample { pos: DVec2::ZERO, pressure: 0.0 });
        assert!((d[0].radius - 1.0).abs() < 1e-6);
    }
}
