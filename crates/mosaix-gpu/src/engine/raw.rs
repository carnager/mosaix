//! Camera RAW layers: developed (crate mosaix-raw) when composited, like
//! vector layers. The demosaiced image of the last developed file is kept, so
//! changing white balance or exposure only redoes the cheap colour step.

use std::sync::Arc;

use half::f16;
use mosaix_core::raw::{Develop, NoiseReduction, RawLayer, WhiteBalance};
use mosaix_core::vector::Affine;
use mosaix_core::{LayerId, TILE_SIZE, TileCoord};
use mosaix_raw::{CameraImage, Info, RawFile};

use super::Engine;

pub(super) struct RawCache {
    id: u64,
    /// The noise reduction applied to `cam`.
    noise: NoiseReduction,
    file: Arc<RawFile>,
    cam: Arc<CameraImage>,
}

/// What the interface shows about a RAW layer.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSummary {
    pub info: Info,
    /// Size of the developed image.
    pub size: (u32, u32),
    /// Temperature and tint of "As Shot", "Auto" and "Daylight".
    pub modes: [(f32, f32); 3],
}

impl RawSummary {
    /// The temperature and tint a setting stands for.
    pub fn temperature(&self, wb: WhiteBalance) -> (f32, f32) {
        match wb {
            WhiteBalance::AsShot => self.modes[0],
            WhiteBalance::Auto => self.modes[1],
            WhiteBalance::Daylight => self.modes[2],
            WhiteBalance::Custom { temperature, tint } => (temperature, tint),
        }
    }
}

/// A smart object's image, decoded: straight-alpha linear RGBA.
pub struct SmartImage {
    pub w: u32,
    pub h: u32,
    pub px: Vec<f32>,
}

impl SmartImage {
    /// Box-filtered copy `f` times smaller (for strong downscaling).
    fn shrink(&self, f: usize) -> SmartImage {
        let (w, h) = ((self.w as usize).div_ceil(f), (self.h as usize).div_ceil(f));
        let mut px = vec![0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0f32; 4];
                let mut n = 0.0;
                for sy in y * f..((y + 1) * f).min(self.h as usize) {
                    for sx in x * f..((x + 1) * f).min(self.w as usize) {
                        let s = (sy * self.w as usize + sx) * 4;
                        let a = self.px[s + 3];
                        for k in 0..3 {
                            acc[k] += self.px[s + k] * a;
                        }
                        acc[3] += a;
                        n += 1.0;
                    }
                }
                let d = (y * w + x) * 4;
                let a = acc[3] / n;
                for k in 0..3 {
                    px[d + k] = if acc[3] > 0.0 { acc[k] / acc[3] } else { 0.0 };
                }
                px[d + 3] = a;
            }
        }
        SmartImage { w: w as u32, h: h as u32, px }
    }
}

/// Decodes an ordinary image file into straight-alpha linear RGBA (sRGB
/// assumed; colour profiles are not applied yet).
pub fn decode_image(bytes: &[u8]) -> anyhow::Result<SmartImage> {
    use image::ImageDecoder;
    // The embedded colour profile, if any (otherwise sRGB).
    let icc = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .ok()
        .and_then(|r| r.into_decoder().ok())
        .and_then(|mut d| d.icc_profile().ok().flatten());
    let convert = icc.as_deref().and_then(crate::color::color_transform);
    let img = image::load_from_memory(bytes)?;
    let rgba = img.to_rgba32f();
    let (w, h) = rgba.dimensions();
    let mut px = rgba.into_raw();
    match convert {
        Some((_, conv)) => {
            use rayon::prelude::*;
            px.par_chunks_mut(4 * w.max(1) as usize).for_each(|row| {
                let mut rgb: Vec<f32> = row.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
                conv(&mut rgb);
                for (p, c) in row.chunks_mut(4).zip(rgb.chunks(3)) {
                    p[..3].copy_from_slice(c);
                }
            });
        }
        None => {
            for p in px.as_chunks_mut::<4>().0 {
                for v in &mut p[..3] {
                    *v = mosaix_core::color::srgb_to_linear(v.clamp(0.0, 1.0));
                }
            }
        }
    }
    Ok(SmartImage { w, h, px })
}

impl Engine {
    /// A smart object's decoded image (cached).
    pub(super) fn smart_image(&mut self, data: &mosaix_core::raw::RawData) -> Option<Arc<SmartImage>> {
        if let Some(i) = self.smart_cache.get(&data.id) {
            return Some(i.clone());
        }
        match decode_image(&data.bytes) {
            Ok(i) => {
                let i = Arc::new(i);
                // A few images at most stay decoded.
                if self.smart_cache.len() >= 8 {
                    self.smart_cache.clear();
                }
                self.smart_cache.insert(data.id, i.clone());
                Some(i)
            }
            Err(e) => {
                tracing::warn!("{}", mosaix_core::tr!("Smart object {name}: {error}", name = data.name.clone(), error = format!("{e:#}")));
                None
            }
        }
    }

    /// Size of a smart object's image.
    pub fn smart_size(&mut self, data: &mosaix_core::raw::RawData) -> Option<(u32, u32)> {
        self.smart_image(data).map(|i| (i.w, i.h))
    }

    /// Decodes, demosaics and denoises a RAW file (or takes it from the cache).
    fn raw_source(&mut self, r: &RawLayer) -> anyhow::Result<(Arc<RawFile>, Arc<CameraImage>)> {
        let noise = r.develop.noise;
        let file = match &self.raw_cache {
            Some(c) if c.id == r.data.id && c.noise == noise => return Ok((c.file.clone(), c.cam.clone())),
            // Same file, other noise reduction: demosaic again from the decoded file.
            Some(c) if c.id == r.data.id => c.file.clone(),
            _ => Arc::new(RawFile::from_bytes(r.data.bytes.clone(), &r.data.name)?),
        };
        let t0 = std::time::Instant::now();
        let mut cam = file.camera_image()?;
        file.denoise(&mut cam, &noise);
        tracing::debug!(
            "{}",
            mosaix_core::tr!(
                "RAW decoded in {ms} ms",
                ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3)
            )
        );
        let cam = Arc::new(cam);
        self.prime_raw(r.data.id, noise, file.clone(), cam.clone());
        Ok((file, cam))
    }

    /// Hands over a RAW file decoded elsewhere (the loader; `cam` denoised
    /// with `noise`), so opening does not decode it twice.
    pub fn prime_raw(&mut self, id: u64, noise: NoiseReduction, file: Arc<RawFile>, cam: Arc<CameraImage>) {
        self.raw_summaries.entry(id).or_insert_with(|| {
            let modes = [WhiteBalance::AsShot, WhiteBalance::Auto, WhiteBalance::Daylight]
                .map(|wb| file.temperature_of(file.multipliers(wb, &cam)));
            RawSummary { info: file.info.clone(), size: file.output_size(&cam), modes }
        });
        self.raw_cache = Some(RawCache { id, noise, file, cam });
    }

    /// Develops a RAW layer into tiles.
    pub(super) fn render_raw(&mut self, id: LayerId, r: &RawLayer) -> Vec<(TileCoord, Vec<f16>)> {
        if r.data.is_image() {
            let Some(img) = self.smart_image(&r.data) else { return Vec::new() };
            let t = r.transform;
            let k = (t[0] * t[3] - t[1] * t[2]).abs().sqrt();
            let f = if k > 0.0 { (0.75 / k).floor().max(1.0) as usize } else { 1 };
            if f > 1 {
                let small = img.shrink(f);
                let (sx, sy) = (img.w as f64 / small.w as f64, img.h as f64 / small.h as f64);
                let t2 = mosaix_core::vector::compose(t, [sx, 0.0, 0.0, sy, 0.0, 0.0]);
                return place(small.w, small.h, &small.px, t2);
            }
            return place(img.w, img.h, &img.px, t);
        }
        let (file, cam) = match self.raw_source(r) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("{}", mosaix_core::tr!("RAW layer {id}: {error}", id = id, error = format!("{e:#}")));
                return Vec::new();
            }
        };
        // Shrunk a lot: develop a box-filtered copy (no aliasing, faster).
        let t = r.transform;
        let k = (t[0] * t[3] - t[1] * t[2]).abs().sqrt();
        let f = if k > 0.0 { (0.75 / k).floor().max(1.0) as usize } else { 1 };
        if f > 1 {
            let small = cam.shrink(f);
            // The copy's pixel (x, y) covers the original's (f·x, f·y); its
            // orientation flips relative to the cropped size, so scale about
            // the origin of the developed image.
            let (w, h, px) = file.develop(&small, &r.develop);
            let (fw, fh) = file.output_size(&cam);
            let sx = fw as f64 / w as f64;
            let sy = fh as f64 / h as f64;
            let t2 = mosaix_core::vector::compose(t, [sx, 0.0, 0.0, sy, 0.0, 0.0]);
            return place(w, h, &px, t2);
        }
        let (w, h, px) = file.develop(&cam, &r.develop);
        place(w, h, &px, t)
    }

    /// Facts about a RAW layer (after it has been developed once).
    pub fn raw_summary(&self, layer: LayerId) -> Option<RawSummary> {
        let r = self.doc.find(layer)?.raw()?;
        self.raw_summaries.get(&r.data.id).cloned()
    }

    /// Changes how a RAW layer is developed. With a `merge` key, repeated
    /// changes (dragging a slider) form one undo step.
    pub fn set_raw(&mut self, layer: LayerId, d: Develop, label: &str, merge: Option<&str>) -> Result<(), &'static str> {
        if self.doc.find(layer).is_none_or(|l| l.raw().is_none()) {
            return Err(mosaix_core::tr("not a RAW layer"));
        }
        let f = |doc: &mut mosaix_core::Document<crate::pool::Tile>| {
            let Some(mosaix_core::doc::Content::Raw(r)) = doc.find_mut(layer).map(|l| &mut l.content) else {
                return false;
            };
            let changed = r.develop != d;
            r.develop = d;
            changed
        };
        match merge {
            Some(key) => self.edit_merged(label, key, f),
            None => self.edit(label, f),
        };
        Ok(())
    }

    /// Places an image file as a smart object above the active layer:
    /// centred, scaled down to fit if larger than the canvas.
    pub fn place_smart(&mut self, bytes: Vec<u8>, name: &str, label: &str) -> Result<LayerId, &'static str> {
        let data = mosaix_core::raw::RawData::new(bytes, name.to_string());
        if !data.is_image() {
            return Err(mosaix_core::tr("not an image this can embed (PNG, JPEG, WebP, TIFF, AVIF, EXR)"));
        }
        let (w, h) = self.smart_size(&data).ok_or(mosaix_core::tr("the image cannot be read"))?;
        let (cw, ch) = (self.doc.width as f64, self.doc.height as f64);
        let k = (cw / w as f64).min(ch / h as f64).min(1.0);
        let (pw, ph) = (w as f64 * k, h as f64 * k);
        let t = [k, 0.0, 0.0, k, ((cw - pw) * 0.5).round(), ((ch - ph) * 0.5).round()];
        let mut r = RawLayer::new(data, Develop::default());
        r.transform = t;
        let stem = std::path::Path::new(name).file_stem().map_or(name.to_string(), |s| s.to_string_lossy().into_owned());
        self.add_raw(r, &stem, label).ok_or(mosaix_core::tr("layer does not exist"))
    }

    /// Turns a pixel layer into a smart object (its pixels embedded as a
    /// 16-bit PNG), so transforms stay lossless from now on.
    pub fn convert_to_smart(&mut self, layer: LayerId, label: &str) -> Result<(), &'static str> {
        let l = self.doc.find(layer).ok_or(mosaix_core::tr("layer does not exist"))?;
        let Some(tiles) = l.tiles().cloned() else {
            return Err(mosaix_core::tr("only pixel layers can become smart objects"));
        };
        let name = l.name.clone();
        let (x0, y0, x1, y1) = {
            // Bounds of the whole layer (not just the selection).
            let sel = self.doc.selection.take();
            let b = self.content_bounds(layer);
            self.doc.selection = sel;
            b.ok_or(mosaix_core::tr("the layer is empty"))?
        };
        let (w, h) = ((x1 - x0) as u32, (y1 - y0) as u32);
        if w == 0 || h == 0 || (w as u64) * (h as u64) > 400_000_000 {
            return Err(mosaix_core::tr("the layer is too large"));
        }
        let ts = TILE_SIZE as i64;
        let coords: Vec<TileCoord> = tiles.keys().copied().collect();
        let list: Vec<crate::pool::Tile> = coords.iter().map(|c| tiles[c].clone()).collect();
        let raw = self.read_tiles_raw(super::PoolKind::Layer, &list);
        let mut buf = vec![0u16; w as usize * h as usize * 4];
        for (c, bytes) in coords.iter().zip(raw) {
            let px: &[f16] = bytemuck::cast_slice(&bytes);
            let (ox, oy) = c.origin();
            for row in 0..ts {
                let y = oy + row - y0;
                if y < 0 || y >= h as i64 {
                    continue;
                }
                for col in 0..ts {
                    let x = ox + col - x0;
                    if x < 0 || x >= w as i64 {
                        continue;
                    }
                    let s = ((row * ts + col) * 4) as usize;
                    let a = px[s + 3].to_f32().clamp(0.0, 1.0);
                    let d = (y as usize * w as usize + x as usize) * 4;
                    for k in 0..3 {
                        let v = if a > 0.0 { px[s + k].to_f32() / a } else { 0.0 };
                        buf[d + k] = (mosaix_core::color::linear_to_srgb(v.clamp(0.0, 1.0)) * 65535.0).round() as u16;
                    }
                    buf[d + 3] = (a * 65535.0).round() as u16;
                }
            }
        }
        let img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(w, h, buf).ok_or(mosaix_core::tr("the layer is too large"))?;
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba16(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .map_err(|_| mosaix_core::tr("the layer could not be encoded"))?;
        let mut r = RawLayer::new(mosaix_core::raw::RawData::new(png, format!("{name}.png")), Develop::default());
        r.transform = [1.0, 0.0, 0.0, 1.0, x0 as f64, y0 as f64];
        self.edit(label, |d| {
            d.find_mut(layer).unwrap().content = mosaix_core::doc::Content::Raw(r);
            true
        });
        Ok(())
    }

    /// Replaces what a smart object shows (the placement stays).
    pub fn replace_smart(&mut self, layer: LayerId, bytes: Vec<u8>, name: &str, label: &str) -> Result<(), &'static str> {
        let data = mosaix_core::raw::RawData::new(bytes, name.to_string());
        if !data.is_image() || self.smart_size(&data).is_none() {
            return Err(mosaix_core::tr("the image cannot be read"));
        }
        if self.doc.find(layer).and_then(|l| l.raw()).is_none_or(|r| !r.data.is_image()) {
            return Err(mosaix_core::tr("not a smart object"));
        }
        self.edit(label, |d| {
            if let Some(mosaix_core::doc::Content::Raw(r)) = d.find_mut(layer).map(|l| &mut l.content) {
                r.data = data;
            }
            true
        });
        Ok(())
    }

    /// Adds a RAW layer above the active one (one undo step).
    pub fn add_raw(&mut self, r: RawLayer, name: &str, label: &str) -> Option<LayerId> {
        let mut id = None;
        self.edit(label, |d| {
            id = Some(d.add_raw(r, name));
            true
        });
        id
    }
}

/// Places an image (straight-alpha linear RGBA) into tiles through `t`.
/// Whole-pixel translations copy; anything else samples bilinearly.
pub fn place(w: u32, h: u32, px: &[f32], t: Affine) -> Vec<(TileCoord, Vec<f16>)> {
    use rayon::prelude::*;
    let ts = TILE_SIZE as i64;
    let integer = t[0] == 1.0 && t[1] == 0.0 && t[2] == 0.0 && t[3] == 1.0 && t[4].fract() == 0.0 && t[5].fract() == 0.0;
    // Bounding box of the placed image.
    let corners = [(0.0, 0.0), (w as f64, 0.0), (0.0, h as f64), (w as f64, h as f64)]
        .map(|(x, y)| (t[0] * x + t[2] * y + t[4], t[1] * x + t[3] * y + t[5]));
    let x0 = corners.iter().map(|c| c.0).fold(f64::MAX, f64::min).floor() as i64;
    let y0 = corners.iter().map(|c| c.1).fold(f64::MAX, f64::min).floor() as i64;
    let x1 = corners.iter().map(|c| c.0).fold(f64::MIN, f64::max).ceil() as i64;
    let y1 = corners.iter().map(|c| c.1).fold(f64::MIN, f64::max).ceil() as i64;
    if x1 - x0 > 300_000 || y1 - y0 > 300_000 {
        return Vec::new();
    }
    let coords: Vec<TileCoord> = (y0.div_euclid(ts)..=(y1 - 1).div_euclid(ts))
        .flat_map(|ty| (x0.div_euclid(ts)..=(x1 - 1).div_euclid(ts)).map(move |tx| TileCoord::new(tx as i32, ty as i32)))
        .collect();
    let src = mosaix_core::color::Pixels::Linear32(px);
    if integer {
        let (dx, dy) = (t[4] as i64, t[5] as i64);
        return coords
            .into_par_iter()
            .filter_map(|c| {
                mosaix_core::color::extract_region(&src, w, h, c.x as i64 * ts - dx, c.y as i64 * ts - dy)
                    .map(|p| (c, p))
            })
            .collect();
    }
    // Inverse transform: output pixel centre → source position.
    let det = t[0] * t[3] - t[1] * t[2];
    if det.abs() < 1e-12 {
        return Vec::new();
    }
    let inv = [t[3] / det, -t[1] / det, -t[2] / det, t[0] / det];
    let (iw, ih) = (w as i64, h as i64);
    // Sampling clamps at the border; the edge itself is anti-aliased by distance.
    let at = |x: i64, y: i64| -> [f32; 4] {
        let s = ((y.clamp(0, ih - 1) * iw + x.clamp(0, iw - 1)) * 4) as usize;
        let a = px[s + 3].clamp(0.0, 1.0);
        [px[s] * a, px[s + 1] * a, px[s + 2] * a, a]
    };
    // Output pixels per source pixel.
    let k = det.abs().sqrt();
    coords
        .into_par_iter()
        .filter_map(|c| {
            let mut out = vec![f16::ZERO; (ts * ts * 4) as usize];
            let mut any = false;
            for row in 0..ts {
                for col in 0..ts {
                    let ox = (c.x as i64 * ts + col) as f64 + 0.5 - t[4];
                    let oy = (c.y as i64 * ts + row) as f64 + 0.5 - t[5];
                    let (ux, uy) = (inv[0] * ox + inv[2] * oy, inv[1] * ox + inv[3] * oy);
                    let edge = ux.min(w as f64 - ux).min(uy).min(h as f64 - uy) * k + 0.5;
                    if edge <= 0.0 {
                        continue;
                    }
                    let cover = edge.min(1.0) as f32;
                    let (sx, sy) = (ux - 0.5, uy - 0.5);
                    let (fx, fy) = (sx.floor(), sy.floor());
                    let (ax, ay) = ((sx - fx) as f32, (sy - fy) as f32);
                    let (ix, iy) = (fx as i64, fy as i64);
                    let (p00, p10, p01, p11) = (at(ix, iy), at(ix + 1, iy), at(ix, iy + 1), at(ix + 1, iy + 1));
                    let d = ((row * ts + col) * 4) as usize;
                    for k in 0..4 {
                        let top = p00[k] + (p10[k] - p00[k]) * ax;
                        let bot = p01[k] + (p11[k] - p01[k]) * ax;
                        out[d + k] = f16::from_f32((top + (bot - top) * ay) * cover);
                    }
                    any |= out[d + 3] > f16::ZERO;
                }
            }
            any.then_some((c, out))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosaix_core::vector::IDENTITY;

    #[test]
    fn placing() {
        // 3×2 opaque image, shifted by a whole tile plus one pixel.
        let px: Vec<f32> = (0..6).flat_map(|i| [i as f32 / 10.0, 0.0, 0.0, 1.0]).collect();
        let t = [1.0, 0.0, 0.0, 1.0, TILE_SIZE as f64 + 1.0, 0.0];
        let tiles = place(3, 2, &px, t);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].0, TileCoord::new(1, 0));
        let p = &tiles[0].1;
        assert_eq!(p[4 + 3].to_f32(), 1.0);
        assert!((p[8].to_f32() - 0.1).abs() < 1e-3);
        // Scaling by 2 keeps the corners of the image.
        let tiles = place(3, 2, &px, [2.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        let p = &tiles[0].1;
        assert!(p[3].to_f32() > 0.99);
        assert_eq!(place(3, 2, &px, IDENTITY).len(), 1);
    }
}

#[cfg(test)]
mod icc_tests {
    #[test]
    fn smart_object_images_keep_their_colour_profile() {
        use image::ImageEncoder;
        // Pure Display P3 red is more saturated than sRGB can show.
        let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let mut png = Vec::new();
        let mut enc = image::codecs::png::PngEncoder::new(&mut png);
        enc.set_icc_profile(icc).unwrap();
        enc.write_image(&[255, 0, 0, 255], 1, 1, image::ExtendedColorType::Rgba8).unwrap();
        let img = super::decode_image(&png).unwrap();
        assert!(img.px[0] > 1.1 && img.px[1] < 0.0, "P3 red in linear sRGB: {:?}", &img.px[..3]);
        // Without a profile: plain sRGB.
        let mut plain = Vec::new();
        image::codecs::png::PngEncoder::new(&mut plain).write_image(&[255, 0, 0, 255], 1, 1, image::ExtendedColorType::Rgba8).unwrap();
        let img = super::decode_image(&plain).unwrap();
        assert!((img.px[0] - 1.0).abs() < 1e-4 && img.px[1].abs() < 1e-4);
    }
}
