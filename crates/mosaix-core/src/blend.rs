//! Blend modes and the CPU reference compositor.
//!
//! The GPU implementation lives in `mosaix-gpu/src/shaders/flatten.wgsl` and
//! must match [`composite`] (tests compare both). Formulas follow the W3C
//! "Compositing and Blending Level 1" spec, which matches Photoshop's modes.

use crate::color::{linear_to_srgb, srgb_to_linear};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendMode {
    #[default]
    Normal = 0,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Add,
    Subtract,
    Divide,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl BlendMode {
    pub const ALL: [BlendMode; 19] = [
        Self::Normal,
        Self::Multiply,
        Self::Screen,
        Self::Overlay,
        Self::Darken,
        Self::Lighten,
        Self::ColorDodge,
        Self::ColorBurn,
        Self::HardLight,
        Self::SoftLight,
        Self::Difference,
        Self::Exclusion,
        Self::Add,
        Self::Subtract,
        Self::Divide,
        Self::Hue,
        Self::Saturation,
        Self::Color,
        Self::Luminosity,
    ];

    /// Display name (Photoshop's terms, translated).
    pub fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Multiply => crate::tr("Multiply"),
            Self::Screen => crate::tr("Screen"),
            Self::Overlay => crate::tr("Overlay"),
            Self::Darken => crate::tr("Darken"),
            Self::Lighten => crate::tr("Lighten"),
            Self::ColorDodge => crate::tr("Color Dodge"),
            Self::ColorBurn => crate::tr("Color Burn"),
            Self::HardLight => crate::tr("Hard Light"),
            Self::SoftLight => crate::tr("Soft Light"),
            Self::Difference => crate::tr("Difference"),
            Self::Exclusion => crate::tr("Exclusion"),
            Self::Add => crate::tr("Linear Dodge (Add)"),
            Self::Subtract => crate::tr("Subtract"),
            Self::Divide => crate::tr("Divide"),
            Self::Hue => crate::tr("Hue"),
            Self::Saturation => crate::tr("Saturation"),
            Self::Color => crate::tr("Color"),
            Self::Luminosity => crate::tr("Luminosity"),
        }
    }

    /// Stable identifier for files.
    pub fn key(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Multiply => "multiply",
            Self::Screen => "screen",
            Self::Overlay => "overlay",
            Self::Darken => "darken",
            Self::Lighten => "lighten",
            Self::ColorDodge => "color-dodge",
            Self::ColorBurn => "color-burn",
            Self::HardLight => "hard-light",
            Self::SoftLight => "soft-light",
            Self::Difference => "difference",
            Self::Exclusion => "exclusion",
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Divide => "divide",
            Self::Hue => "hue",
            Self::Saturation => "saturation",
            Self::Color => "color",
            Self::Luminosity => "luminosity",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key() == key)
    }

    pub fn cycle(self, step: i32) -> Self {
        let n = Self::ALL.len() as i32;
        Self::ALL[((self as i32 + step).rem_euclid(n)) as usize]
    }
}

/// Colour space in which blend formulas are evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendSpace {
    /// sRGB-encoded values, like Photoshop in 8/16 bit. What users expect.
    #[default]
    Srgb = 0,
    /// Linear light: physically correct, softer transitions.
    Linear = 1,
}

impl BlendSpace {
    pub fn name(self) -> &'static str {
        match self {
            Self::Srgb => crate::tr("sRGB (Photoshop-compatible)"),
            Self::Linear => "linear",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Srgb => "srgb",
            Self::Linear => "linear",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        match key {
            "srgb" => Some(Self::Srgb),
            "linear" => Some(Self::Linear),
            _ => None,
        }
    }
}

type Rgb = [f32; 3];

fn lum(c: Rgb) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(c: Rgb) -> Rgb {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut c = c;
    if n < 0.0 {
        c = c.map(|v| l + (v - l) * l / (l - n));
    }
    if x > 1.0 {
        c = c.map(|v| l + (v - l) * (1.0 - l) / (x - l));
    }
    c
}

fn set_lum(c: Rgb, l: f32) -> Rgb {
    let d = l - lum(c);
    clip_color(c.map(|v| v + d))
}

fn sat(c: Rgb) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: Rgb, s: f32) -> Rgb {
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    if x > n {
        c.map(|v| (v - n) * s / (x - n))
    } else {
        [0.0; 3]
    }
}

fn separable(b: f32, s: f32, mode: BlendMode) -> f32 {
    use BlendMode::*;
    let hard_light = |b: f32, s: f32| {
        if s <= 0.5 {
            b * 2.0 * s
        } else {
            let s2 = 2.0 * s - 1.0;
            b + s2 - b * s2
        }
    };
    match mode {
        Normal => s,
        Multiply => b * s,
        Screen => b + s - b * s,
        Overlay => hard_light(s, b),
        Darken => b.min(s),
        Lighten => b.max(s),
        ColorDodge => {
            if b <= 0.0 {
                0.0
            } else if s >= 1.0 {
                1.0
            } else {
                (b / (1.0 - s)).min(1.0)
            }
        }
        ColorBurn => {
            if b >= 1.0 {
                1.0
            } else if s <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - b) / s).min(1.0)
            }
        }
        HardLight => hard_light(b, s),
        SoftLight => {
            if s <= 0.5 {
                b - (1.0 - 2.0 * s) * b * (1.0 - b)
            } else {
                let d = if b <= 0.25 {
                    ((16.0 * b - 12.0) * b + 4.0) * b
                } else {
                    b.max(0.0).sqrt()
                };
                b + (2.0 * s - 1.0) * (d - b)
            }
        }
        Difference => (b - s).abs(),
        Exclusion => b + s - 2.0 * b * s,
        Add => b + s,
        Subtract => (b - s).max(0.0),
        Divide => (b / s.max(1e-5)).min(1.0),
        Hue | Saturation | Color | Luminosity => unreachable!(),
    }
}

/// The blend function B(Cb, Cs) on unpremultiplied colours.
pub fn blend_rgb(b: Rgb, s: Rgb, mode: BlendMode) -> Rgb {
    use BlendMode::*;
    match mode {
        Hue => set_lum(set_sat(s, sat(b)), lum(b)),
        Saturation => set_lum(set_sat(b, sat(s)), lum(b)),
        Color => set_lum(s, lum(b)),
        Luminosity => set_lum(b, lum(s)),
        _ => [0, 1, 2].map(|i| separable(b[i], s[i], mode)),
    }
}

fn encode(v: f32, space: BlendSpace) -> f32 {
    match space {
        BlendSpace::Srgb => linear_to_srgb(v.max(0.0)),
        BlendSpace::Linear => v,
    }
}

fn decode(v: f32, space: BlendSpace) -> f32 {
    match space {
        BlendSpace::Srgb => srgb_to_linear(v.max(0.0)),
        BlendSpace::Linear => v,
    }
}

/// Composites premultiplied linear `src` (already multiplied by layer opacity)
/// onto premultiplied linear `dst`. Returns premultiplied linear.
pub fn composite(dst: [f32; 4], src: [f32; 4], mode: BlendMode, space: BlendSpace) -> [f32; 4] {
    let (ab, a_s) = (dst[3], src[3]);
    if a_s <= 0.0 {
        return dst;
    }
    let ao = a_s + ab * (1.0 - a_s);
    if mode == BlendMode::Normal && space == BlendSpace::Linear {
        return [0, 1, 2]
            .map(|i| src[i] + dst[i] * (1.0 - a_s))
            .into_rgba(ao);
    }
    let unpre = |c: [f32; 4], a: f32| -> Rgb {
        if a > 0.0 {
            [0, 1, 2].map(|i| encode(c[i] / a, space))
        } else {
            [0.0; 3]
        }
    };
    let cb = unpre(dst, ab);
    let cs = unpre(src, a_s);
    let mixed = blend_rgb(cb, cs, mode);
    // Porter-Duff source-over with the blend result, in the blend space.
    let co: Rgb = [0, 1, 2]
        .map(|i| cs[i] * a_s * (1.0 - ab) + cb[i] * ab * (1.0 - a_s) + a_s * ab * mixed[i]);
    let rgb = [0, 1, 2].map(|i| decode(co[i] / ao, space) * ao);
    rgb.into_rgba(ao)
}

trait IntoRgba {
    fn into_rgba(self, a: f32) -> [f32; 4];
}

impl IntoRgba for Rgb {
    fn into_rgba(self, a: f32) -> [f32; 4] {
        [self[0], self[1], self[2], a]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn keys_roundtrip_and_cycle() {
        for m in BlendMode::ALL {
            assert_eq!(BlendMode::from_key(m.key()), Some(m));
        }
        assert_eq!(BlendMode::Normal.cycle(-1), BlendMode::Luminosity);
        assert_eq!(BlendMode::Luminosity.cycle(1), BlendMode::Normal);
    }

    #[test]
    fn normal_is_source_over_in_both_spaces() {
        let dst = [0.2, 0.3, 0.4, 1.0];
        let src = [0.25, 0.0, 0.0, 0.5];
        let lin = composite(dst, src, BlendMode::Normal, BlendSpace::Linear);
        assert!(close(lin, [0.35, 0.15, 0.2, 1.0]));
        // Fully opaque source replaces the destination in any space.
        let opaque = [0.7, 0.1, 0.05, 1.0];
        assert!(close(
            composite(dst, opaque, BlendMode::Normal, BlendSpace::Srgb),
            opaque
        ));
    }

    #[test]
    fn empty_backdrop_shows_source_for_every_mode() {
        let src = [0.3, 0.2, 0.1, 0.6];
        for m in BlendMode::ALL {
            for space in [BlendSpace::Srgb, BlendSpace::Linear] {
                let out = composite([0.0; 4], src, m, space);
                assert!(close(out, src), "{m:?} {space:?}: {out:?}");
            }
        }
    }

    #[test]
    fn known_values() {
        let b = [0.5, 0.5, 0.5];
        assert_eq!(blend_rgb(b, [0.5; 3], BlendMode::Multiply), [0.25; 3]);
        assert_eq!(blend_rgb(b, [0.5; 3], BlendMode::Screen), [0.75; 3]);
        assert_eq!(blend_rgb(b, [0.2; 3], BlendMode::Difference)[0], 0.3);
        // Luminosity of grey onto a colour keeps hue, takes luminance.
        let c = blend_rgb([1.0, 0.0, 0.0], [0.5; 3], BlendMode::Luminosity);
        assert!((lum(c) - 0.5).abs() < 1e-6);
    }
}
