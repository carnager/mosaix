//! Basic tone and colour on scene-linear sRGB (after exposure): contrast
//! and shadows in stops around middle grey, highlights on the largest
//! channel (so compressed colours stay in range), vibrance and saturation
//! around luminance.

use imagine_core::raw::Tone;

const GREY: f32 = 0.18;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Precomputed settings (strengths −1…1).
pub struct ToneOp {
    contrast: f32,
    highlights: f32,
    shadows: f32,
    vibrance: f32,
    saturation: f32,
}

impl ToneOp {
    pub fn new(t: &Tone) -> Option<Self> {
        (!t.is_neutral()).then(|| {
            let s = |v: f32| v.clamp(-100.0, 100.0) / 100.0;
            ToneOp {
                contrast: s(t.contrast),
                highlights: s(t.highlights),
                shadows: s(t.shadows),
                vibrance: s(t.vibrance),
                saturation: s(t.saturation),
            }
        })
    }

    /// Brightness curve on a norm (max of the channels): returns the factor.
    fn curve(&self, n: f32) -> f32 {
        if n <= 1e-6 {
            return 1.0;
        }
        // In stops relative to middle grey.
        let mut e = (n / GREY).log2();
        if self.contrast != 0.0 {
            // Steeper/flatter around grey, easing off beyond ±4 stops.
            e += self.contrast * 0.35 * e * (-(e * e) / 32.0).exp();
        }
        if self.shadows != 0.0 {
            // Up to ±1.5 stops three stops below grey and darker, nothing from
            // grey up (slope ≤ 0.75, so the curve stays monotonic).
            let t = ((e + 3.0) / 3.0).clamp(0.0, 1.0);
            e += self.shadows * 1.5 * (1.0 - t * t * (3.0 - 2.0 * t));
        }
        if self.highlights > 0.0 {
            e += self.highlights * 0.8 * sigmoid(2.0 * (e - 1.0));
        }
        let mut m = GREY * e.exp2();
        if self.highlights < 0.0 {
            // Soft shoulder: from the knee up, approach 1 instead of clipping.
            let k = 1.0 - 0.6 * -self.highlights;
            if m > k {
                m = k + (1.0 - k) * (1.0 - (-(m - k) / (1.0 - k)).exp());
            }
        }
        m / n
    }

    pub fn apply(&self, rgb: &mut [f32]) {
        let n = rgb[0].max(rgb[1]).max(rgb[2]);
        let f = self.curve(n);
        for v in rgb.iter_mut().take(3) {
            *v *= f;
        }
        if self.saturation != 0.0 || self.vibrance != 0.0 {
            let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            let mx = rgb[0].max(rgb[1]).max(rgb[2]);
            let mn = rgb[0].min(rgb[1]).min(rgb[2]);
            // Vibrance acts mostly on muted colours.
            let sat = if mx > 1e-6 { ((mx - mn) / mx).clamp(0.0, 1.0) } else { 0.0 };
            let k = 1.0 + self.saturation + self.vibrance * (1.0 - sat) * (1.0 - sat);
            for v in rgb.iter_mut().take(3) {
                *v = y + (*v - y) * k.max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves() {
        let op = |t: Tone| ToneOp::new(&t).unwrap();
        // Grey stays put under contrast and shadows.
        let c = op(Tone { contrast: 80.0, shadows: 60.0, ..Default::default() });
        assert!((c.curve(GREY) - 1.0).abs() < 0.03);
        // Contrast darkens below grey and brightens above.
        let c = op(Tone { contrast: 80.0, ..Default::default() });
        assert!(c.curve(0.05) < 1.0 && c.curve(0.5) > 1.0);
        // Negative highlights bring 4.0 below 1 and keep the order.
        let h = op(Tone { highlights: -100.0, ..Default::default() });
        let (a, b) = (0.9 * h.curve(0.9), 4.0 * h.curve(4.0));
        assert!(b < 1.0 && a < b, "{a} {b}");
        assert!((0.1 * h.curve(0.1) - 0.1).abs() < 1e-5);
        // Shadows lift the dark end.
        let s = op(Tone { shadows: 100.0, ..Default::default() });
        assert!(s.curve(0.01) > 1.5);
        // Saturation 0 → −100 makes grey.
        let g = op(Tone { saturation: -100.0, ..Default::default() });
        let mut px = [0.6, 0.2, 0.1];
        g.apply(&mut px);
        assert!((px[0] - px[2]).abs() < 1e-5);
        assert!(ToneOp::new(&Tone::default()).is_none());
    }
}
