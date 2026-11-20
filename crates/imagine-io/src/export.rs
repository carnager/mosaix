//! Export of the flattened image to common formats.

use std::path::Path;

use anyhow::{Context, bail};
use half::f16;
use image::{ImageFormat, RgbImage, RgbaImage};
use imagine_core::color::linear_to_srgb;
use imagine_core::{TILE_SIZE, TileCoord};
use rayon::prelude::*;

/// Converts premultiplied linear f16 tiles into an sRGB 8-bit image.
/// Missing tiles are transparent.
pub fn tiles_to_rgba8(width: u32, height: u32, tiles: &[(TileCoord, Vec<f16>)]) -> RgbaImage {
    let mut img = RgbaImage::new(width, height);
    blit(&mut img, tiles);
    img
}

/// Converts tiles and writes them into `img` (tiles may be fed in batches).
pub fn blit(img: &mut RgbaImage, tiles: &[(TileCoord, Vec<f16>)]) {
    blit_region(img, tiles, TILE_SIZE, (0, 0));
}

/// Like [`blit`] for tiles of `tile_px` pixels (a mip level), where `img`'s
/// top-left corresponds to `origin` in that scaled pixel space.
pub fn blit_region(img: &mut RgbaImage, tiles: &[(TileCoord, Vec<f16>)], tile_px: u32, origin: (i64, i64)) {
    let (width, height) = img.dimensions();
    let lut: Vec<u8> = (0..=4096).map(|i| (linear_to_srgb(i as f32 / 4096.0) * 255.0).round() as u8).collect();
    let to8 = |v: f32| lut[(v.clamp(0.0, 1.0) * 4096.0).round() as usize];
    let ts = tile_px as usize;
    let converted: Vec<(TileCoord, Vec<u8>)> = tiles
        .par_iter()
        .map(|(c, px)| {
            let mut out = vec![0u8; ts * ts * 4];
            for (o, p) in out.as_chunks_mut::<4>().0.iter_mut().zip(px.as_chunks::<4>().0) {
                let a = p[3].to_f32().clamp(0.0, 1.0);
                if a > 0.0 {
                    o[0] = to8(p[0].to_f32() / a);
                    o[1] = to8(p[1].to_f32() / a);
                    o[2] = to8(p[2].to_f32() / a);
                    o[3] = (a * 255.0).round() as u8;
                }
            }
            (*c, out)
        })
        .collect();

    let stride = width as usize * 4;
    let buf: &mut [u8] = img;
    for (c, data) in converted {
        let x0 = c.x as i64 * ts as i64 - origin.0;
        let y0 = c.y as i64 * ts as i64 - origin.1;
        for row in 0..ts as i64 {
            let y = y0 + row;
            if y < 0 || y >= height as i64 {
                continue;
            }
            let xa = x0.max(0);
            let xb = (x0 + ts as i64).min(width as i64);
            if xa >= xb {
                continue;
            }
            let src = (row as usize * ts + (xa - x0) as usize) * 4;
            let dst = y as usize * stride + xa as usize * 4;
            let n = (xb - xa) as usize * 4;
            buf[dst..dst + n].copy_from_slice(&data[src..src + n]);
        }
    }
}

/// Saves by file extension: png, jpg/jpeg (flattened onto white), webp, tif/tiff.
pub fn save(img: &RgbaImage, path: &Path) -> anyhow::Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let format = ImageFormat::from_extension(&ext).with_context(|| format!("unbekanntes Format „{ext}“"))?;
    match format {
        ImageFormat::Jpeg => {
            let mut rgb = RgbImage::new(img.width(), img.height());
            for (o, p) in rgb.pixels_mut().zip(img.pixels()) {
                let a = p[3] as u32;
                *o = image::Rgb([0, 1, 2].map(|i| ((p[i] as u32 * a + 255 * (255 - a)) / 255) as u8));
            }
            let file = std::io::BufWriter::new(std::fs::File::create(path)?);
            let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(file, 92);
            rgb.write_with_encoder(enc)?;
        }
        ImageFormat::Png | ImageFormat::WebP | ImageFormat::Tiff => img.save_with_format(path, format)?,
        _ => bail!("Export nach „{ext}“ wird nicht unterstützt (png, jpg, webp, tiff)"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_and_crops() {
        let ts = TILE_SIZE as usize;
        let mut px = vec![f16::ZERO; ts * ts * 4];
        // Pixel (0,0): 50 % alpha, linear 0.5 premultiplied → straight 1.0 → 255
        px[0] = f16::from_f32(0.5);
        px[3] = f16::from_f32(0.5);
        let img = tiles_to_rgba8(300, 10, &[(TileCoord::new(1, 0), px)]);
        assert_eq!(img.dimensions(), (300, 10));
        assert_eq!(img.get_pixel(256, 0).0, [255, 0, 0, 128]);
        assert_eq!(img.get_pixel(0, 0).0, [0, 0, 0, 0]);
    }

    #[test]
    fn jpeg_flattens_on_white() {
        let dir = std::env::temp_dir().join(format!("imagine-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.jpg");
        save(&RgbaImage::new(8, 8), &p).unwrap();
        let back = image::open(&p).unwrap().to_rgb8();
        assert!(back.get_pixel(4, 4).0.iter().all(|&v| v > 250));
        assert!(save(&RgbaImage::new(1, 1), &dir.join("t.xyz")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
