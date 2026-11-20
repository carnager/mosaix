//! Patterns for the pattern overlay: small sRGB tiles that repeat. A layer
//! style carries its pattern's pixels, so documents stay self-contained.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A repeating tile: straight (not premultiplied) sRGB-encoded RGBA, 8 bit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    pub name: String,
    pub width: u32,
    pub height: u32,
    #[serde(with = "b64")]
    pub pixels: Vec<u8>,
}

mod b64 {
    use super::*;
    use base64::Engine;

    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(v))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let t = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD.decode(t).map_err(serde::de::Error::custom)
    }
}

/// Largest pattern side (bigger images are scaled down on import).
pub const MAX_PATTERN: u32 = 1024;

impl Pattern {
    /// A pattern from straight RGBA pixels; `None` if the sizes do not match.
    pub fn new(name: &str, width: u32, height: u32, pixels: Vec<u8>) -> Option<Self> {
        (width > 0 && height > 0 && pixels.len() == width as usize * height as usize * 4)
            .then(|| Self { name: name.into(), width, height, pixels })
    }

    /// The pixel at (x, y), repeating: straight sRGB-encoded RGBA 0…1.
    pub fn at(&self, x: i64, y: i64) -> [f32; 4] {
        let (w, h) = (self.width as i64, self.height as i64);
        let i = (y.rem_euclid(h) * w + x.rem_euclid(w)) as usize * 4;
        let p = &self.pixels[i..i + 4];
        [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0]
    }

    /// Bilinear sample at a continuous position (pattern pixels), repeating.
    pub fn sample(&self, x: f32, y: f32) -> [f32; 4] {
        let (x, y) = (x - 0.5, y - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let (a, b, c, d) = (self.at(x0, y0), self.at(x0 + 1, y0), self.at(x0, y0 + 1), self.at(x0 + 1, y0 + 1));
        std::array::from_fn(|k| {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bot = c[k] + (d[k] - c[k]) * fx;
            top + (bot - top) * fy
        })
    }

    /// Patterns that come with imagine (generated).
    pub fn builtin() -> Vec<Pattern> {
        let make = |name: &str, n: u32, f: &dyn Fn(u32, u32) -> [u8; 4]| {
            let mut px = Vec::with_capacity((n * n * 4) as usize);
            for y in 0..n {
                for x in 0..n {
                    px.extend(f(x, y));
                }
            }
            Pattern::new(name, n, n, px).unwrap()
        };
        let grey = |v: u8| [v, v, v, 255];
        // Smooth edges: coverage of a distance (px) to an edge.
        let cov = |d: f32| (0.5 - d).clamp(0.0, 1.0);
        let mix = |a: u8, b: u8, t: f32| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
        // Deterministic value noise.
        let hash = |x: u32, y: u32| {
            let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
            h ^= h >> 15;
            h = h.wrapping_mul(0x2C1B_3C6D);
            h ^= h >> 12;
            (h & 0xff) as u8
        };
        vec![
            make(crate::tr("Checkerboard"), 32, &|x, y| grey(if (x / 16 + y / 16) % 2 == 0 { 230 } else { 120 })),
            make(crate::tr("Dots"), 24, &|x, y| {
                let d = ((x as f32 + 0.5 - 12.0).hypot(y as f32 + 0.5 - 12.0)) - 6.0;
                grey(mix(235, 60, cov(d)))
            }),
            make(crate::tr("Diagonal Lines"), 16, &|x, y| {
                // Lines 4 px wide every 16 px along x + y, seamless.
                let t = ((x + y) % 16) as f32 + 0.5;
                let d = (t - 8.0).abs() - 2.0;
                grey(mix(230, 70, cov(d)))
            }),
            make(crate::tr("Grid"), 32, &|x, y| {
                let d = (x.min(31 - x).min(y).min(31 - y)) as f32;
                grey(if d < 1.0 { 90 } else { 235 })
            }),
            make(crate::tr("Noise"), 128, &|x, y| grey(96 + hash(x, y) / 2)),
            make(crate::i18n::tr_ctx("pattern", "Canvas"), 64, &|x, y| {
                // Woven threads: alternating over/under with some grain.
                let warp = ((x % 4) as f32 - 1.5).abs() / 1.5;
                let weft = ((y % 4) as f32 - 1.5).abs() / 1.5;
                let over = (x / 4 + y / 4) % 2 == 0;
                let shade = if over { 1.0 - warp * 0.35 } else { 1.0 - weft * 0.35 };
                let g = (205.0 * shade) as i32 + hash(x, y) as i32 / 12 - 10;
                let g = g.clamp(0, 255) as u8;
                [g, (g as f32 * 0.97) as u8, (g as f32 * 0.9) as u8, 255]
            }),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_repeat_and_serialise() {
        for p in Pattern::builtin() {
            assert_eq!(p.pixels.len(), (p.width * p.height * 4) as usize);
            assert_eq!(p.at(-1, -1), p.at(p.width as i64 - 1, p.height as i64 - 1));
            let j = serde_json::to_string(&p).unwrap();
            assert!(j.len() < p.pixels.len() * 2, "compact (base64)");
            assert_eq!(serde_json::from_str::<Pattern>(&j).unwrap(), p);
        }
    }
}
