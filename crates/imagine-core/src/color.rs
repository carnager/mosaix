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
}

/// Extracts one tile from an image of `width × height` pixels, converted to
/// premultiplied linear f16. Pixels outside the image are transparent.
/// Returns `None` if the tile is fully transparent (it can stay unallocated).
pub fn extract_tile(src: &Pixels, width: u32, height: u32, tx: u32, ty: u32) -> Option<Vec<f16>> {
    let ts = TILE_SIZE as usize;
    let mut out = vec![f16::ZERO; ts * ts * 4];
    let x0 = (tx * TILE_SIZE) as usize;
    let y0 = (ty * TILE_SIZE) as usize;
    let w = (width as usize).saturating_sub(x0).min(ts);
    let h = (height as usize).saturating_sub(y0).min(ts);
    let lut = srgb8_lut();
    let mut any = false;

    for row in 0..h {
        let src_off = ((y0 + row) * width as usize + x0) * 4;
        let dst = &mut out[row * ts * 4..][..w * 4];
        for (i, px) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let s = src_off + i * 4;
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
            };
            any |= a > 0.0;
            px[0] = f16::from_f32(r * a);
            px[1] = f16::from_f32(g * a);
            px[2] = f16::from_f32(b * a);
            px[3] = f16::from_f32(a);
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
    fn transparent_tile_is_skipped() {
        let img = [0u8; 16];
        assert!(extract_tile(&Pixels::Srgb8(&img), 2, 2, 0, 0).is_none());
    }
}
