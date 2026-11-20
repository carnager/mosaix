//! Colour conversion into the working format: premultiplied, linear-light
//! RGBA in 16-bit float.

use half::f16;
use std::sync::OnceLock;

use crate::tile::TILE_SIZE;

pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

fn srgb8_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

/// Pixel source for tile extraction.
pub enum Pixels<'a> {
    /// sRGB-encoded 8-bit RGBA, straight alpha.
    Srgb8(&'a [u8]),
    /// sRGB-encoded 16-bit RGBA, straight alpha.
    Srgb16(&'a [u16]),
    /// Linear 32-bit float RGBA, straight alpha (e.g. EXR).
    Linear32(&'a [f32]),
    /// 16-bit RGBA in any colour space (straight alpha); `convert` turns a
    /// row of RGB triples (0…1, encoded) into linear sRGB in place – values
    /// outside 0…1 (wide gamut) are kept.
    Converted16 { data: &'a [u16], convert: &'a (dyn Fn(&mut [f32]) + Sync) },
}

/// Extracts one tile from an image of `width × height` pixels, converted to
/// premultiplied linear f16. Pixels outside the image are transparent.
/// Returns `None` if the tile is fully transparent (it can stay unallocated).
pub fn extract_tile(src: &Pixels, width: u32, height: u32, tx: u32, ty: u32) -> Option<Vec<f16>> {
    extract_region(src, width, height, (tx * TILE_SIZE) as i64, (ty * TILE_SIZE) as i64)
}

/// Like [`extract_tile`] for a tile whose top-left corner lies at image
/// coordinates `(x0, y0)` (may be negative or unaligned – used for placing
/// images at arbitrary positions).
pub fn extract_region(src: &Pixels, width: u32, height: u32, x0: i64, y0: i64) -> Option<Vec<f16>> {
    let ts = TILE_SIZE as i64;
    let mut out = vec![f16::ZERO; (ts * ts * 4) as usize];
    let lut = srgb8_lut();
    let mut any = false;
    let (w, h) = (width as i64, height as i64);
    let (col0, col1) = ((-x0).clamp(0, ts), (w - x0).clamp(0, ts));
    if col0 >= col1 {
        return None;
    }
    let mut rgb_row: Vec<f32> = Vec::new();
    for row in 0..ts {
        let y = y0 + row;
        if y < 0 || y >= h {
            continue;
        }
        if let Pixels::Converted16 { data, convert } = src {
            rgb_row.clear();
            for col in col0..col1 {
                let s = ((y * w + x0 + col) * 4) as usize;
                rgb_row.extend([data[s], data[s + 1], data[s + 2]].map(|v| v as f32 / 65535.0));
            }
            convert(&mut rgb_row);
        }
        for col in col0..col1 {
            let s = ((y * w + x0 + col) * 4) as usize;
            let (r, g, b, a) = match src {
                Pixels::Srgb8(p) => (
                    lut[p[s] as usize],
                    lut[p[s + 1] as usize],
                    lut[p[s + 2] as usize],
                    p[s + 3] as f32 / 255.0,
                ),
                Pixels::Srgb16(p) => (
                    srgb_to_linear(p[s] as f32 / 65535.0),
                    srgb_to_linear(p[s + 1] as f32 / 65535.0),
                    srgb_to_linear(p[s + 2] as f32 / 65535.0),
                    p[s + 3] as f32 / 65535.0,
                ),
                Pixels::Linear32(p) => (p[s], p[s + 1], p[s + 2], p[s + 3].clamp(0.0, 1.0)),
                Pixels::Converted16 { data, .. } => {
                    let i = ((col - col0) * 3) as usize;
                    (rgb_row[i], rgb_row[i + 1], rgb_row[i + 2], data[s + 3] as f32 / 65535.0)
                }
            };
            any |= a > 0.0;
            let d = ((row * ts + col) * 4) as usize;
            out[d] = f16::from_f32(r * a);
            out[d + 1] = f16::from_f32(g * a);
            out[d + 2] = f16::from_f32(b * a);
            out[d + 3] = f16::from_f32(a);
        }
    }
    any.then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_roundtrip() {
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
    }

    #[test]
    fn extract_premultiplies_and_pads() {
        // 2×1 image: opaque white, half-transparent sRGB 50% grey.
        let img = [255u8, 255, 255, 255, 128, 128, 128, 128];
        let t = extract_tile(&Pixels::Srgb8(&img), 2, 1, 0, 0).unwrap();
        assert_eq!(t[0].to_f32(), 1.0);
        let a = 128.0 / 255.0;
        assert!((t[4].to_f32() - srgb_to_linear(a) * a).abs() < 1e-3);
        assert!((t[7].to_f32() - a).abs() < 1e-3);
        // Outside the image: transparent.
        assert_eq!(t[8].to_f32(), 0.0);
        assert_eq!(t[TILE_SIZE as usize * 4 + 3].to_f32(), 0.0);
    }

    #[test]
    fn region_with_negative_offset() {
        // 2×1 opaque white image placed so its first pixel lands at tile pixel (3, 5).
        let img = [255u8; 8];
        let t = extract_region(&Pixels::Srgb8(&img), 2, 1, -3, -5).unwrap();
        let at = |x: usize, y: usize| t[(y * TILE_SIZE as usize + x) * 4 + 3].to_f32();
        assert_eq!(at(3, 5), 1.0);
        assert_eq!(at(4, 5), 1.0);
        assert_eq!(at(5, 5), 0.0);
        assert_eq!(at(3, 4), 0.0);
        assert!(extract_region(&Pixels::Srgb8(&img), 2, 1, 10, 0).is_none());
    }

    #[test]
    fn transparent_tile_is_skipped() {
        let img = [0u8; 16];
        assert!(extract_tile(&Pixels::Srgb8(&img), 2, 2, 0, 0).is_none());
    }
}
