//! Export of the flattened image to common formats.

use std::path::Path;

use std::io::BufWriter;

use anyhow::{Context, anyhow, bail};
use half::f16;
use image::{ExtendedColorType, ImageBuffer, ImageEncoder, ImageFormat, RgbImage, Rgba, RgbaImage};
use mosaix_core::color::linear_to_srgb;
use mosaix_core::{TILE_SIZE, TileCoord};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Colour space of an exported file. Anything but sRGB embeds its ICC profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    #[default]
    Srgb,
    DisplayP3,
    AdobeRgb,
    Rec2020,
}

impl ColorSpace {
    pub const ALL: [Self; 4] = [Self::Srgb, Self::DisplayP3, Self::AdobeRgb, Self::Rec2020];

    pub fn name(self) -> &'static str {
        match self {
            Self::Srgb => "sRGB",
            Self::DisplayP3 => "Display P3",
            Self::AdobeRgb => "Adobe RGB (1998)",
            Self::Rec2020 => "Rec. 2020",
        }
    }

    /// CICP colour primaries and transfer characteristics (ITU-T H.273),
    /// for formats that signal the space by code; Adobe RGB has none.
    pub fn cicp(self) -> Option<(u16, u16)> {
        match self {
            Self::Srgb => Some((1, 13)),
            Self::DisplayP3 => Some((12, 13)),
            Self::Rec2020 => Some((9, 1)),
            Self::AdobeRgb => None,
        }
    }

    /// The colour encoding libjxl signals for this space.
    fn jxl_encoding(self) -> jpegxl_sys::color::color_encoding::JxlColorEncoding {
        use jpegxl_sys::color::color_encoding::*;
        let mut e = JxlColorEncoding {
            color_space: JxlColorSpace::Rgb,
            white_point: JxlWhitePoint::D65,
            white_point_xy: [0.3127, 0.3290],
            primaries: JxlPrimaries::SRgb,
            primaries_red_xy: [0.0; 2],
            primaries_green_xy: [0.0; 2],
            primaries_blue_xy: [0.0; 2],
            transfer_function: JxlTransferFunction::SRGB,
            gamma: 0.0,
            rendering_intent: JxlRenderingIntent::Relative,
        };
        match self {
            Self::Srgb => {}
            Self::DisplayP3 => e.primaries = JxlPrimaries::P3,
            Self::Rec2020 => {
                e.primaries = JxlPrimaries::Rec2100;
                e.transfer_function = JxlTransferFunction::BT709;
            }
            Self::AdobeRgb => {
                e.primaries = JxlPrimaries::Custom;
                e.primaries_red_xy = [0.64, 0.33];
                e.primaries_green_xy = [0.21, 0.71];
                e.primaries_blue_xy = [0.15, 0.06];
                e.transfer_function = JxlTransferFunction::Gamma;
                // libjxl's gamma is the encoding exponent.
                e.gamma = 1.0 / 2.19921875;
            }
        }
        e
    }

    pub(crate) fn profile(self) -> moxcms::ColorProfile {
        use moxcms::ColorProfile as P;
        match self {
            Self::Srgb => P::new_srgb(),
            Self::DisplayP3 => P::new_display_p3(),
            Self::AdobeRgb => P::new_adobe_rgb(),
            Self::Rec2020 => P::new_bt2020(),
        }
    }

    /// The ICC profile to embed (`None` for sRGB, the assumed default).
    pub fn icc(self) -> anyhow::Result<Option<Vec<u8>>> {
        if self == Self::Srgb {
            return Ok(None);
        }
        let mut p = self.profile();
        p.description = Some(moxcms::ProfileText::PlainString(self.name().into()));
        p.encode().map(Some).map_err(|e| {
            anyhow!(mosaix_core::tr!(
                "ICC profile {name}: {err}",
                name = self.name(),
                err = format!("{e:?}")
            ))
        })
    }

    /// Converter: straight linear sRGB triples (extended range) → encoded
    /// values of this space, clamped to 0…1.
    pub fn converter(self) -> anyhow::Result<Converter> {
        if self == Self::Srgb {
            return Ok(Box::new(|rgb: &mut [f32]| {
                rgb.iter_mut()
                    .for_each(|v| *v = linear_to_srgb(v.clamp(0.0, 1.0)))
            }));
        }
        use moxcms::{ColorProfile, Layout, ToneReprCurve, TransformOptions};
        let mut linear = ColorProfile::new_srgb();
        let identity = ToneReprCurve::Lut(vec![]);
        linear.red_trc = Some(identity.clone());
        linear.green_trc = Some(identity.clone());
        linear.blue_trc = Some(identity);
        linear.cicp = None;
        let opts = TransformOptions {
            prefer_fixed_point: false,
            allow_extended_range_rgb_xyz: true,
            ..Default::default()
        };
        let t = linear
            .create_transform_f32(Layout::Rgb, &self.profile(), Layout::Rgb, opts)
            .map_err(|e| {
                anyhow!(mosaix_core::tr!(
                    "Conversion to {name}: {err}",
                    name = self.name(),
                    err = format!("{e:?}")
                ))
            })?;
        Ok(Box::new(move |rgb: &mut [f32]| {
            // The target's curve is not defined below zero: clip first.
            let input: Vec<f32> = rgb.to_vec();
            if t.transform(&input, rgb).is_err() {
                rgb.fill(0.0);
            }
            rgb.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
        }))
    }
}

/// Converts a row of RGB triples in place.
pub type Converter = Box<dyn Fn(&mut [f32]) + Send + Sync>;

/// How an image is exported.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportOptions {
    pub space: ColorSpace,
    /// 16 bits per channel (PNG, TIFF and PSD; other formats use 8).
    pub sixteen_bit: bool,
    /// JPEG quality 1–100.
    pub quality: u8,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            space: ColorSpace::Srgb,
            sixteen_bit: false,
            quality: 92,
        }
    }
}

/// Whether a file name's format can hold 16 bits per channel.
pub fn supports_16bit(path: &Path) -> bool {
    matches!(
        ext_of(path).as_str(),
        "png" | "tif" | "tiff" | "psd" | "psb" | "jxl"
    )
}

fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Converts premultiplied linear tiles and writes the file `path` with the
/// given options (format by extension: png, jpg/jpeg, webp, tif/tiff).
pub fn write(
    path: &Path,
    (width, height): (u32, u32),
    tiles: &[(TileCoord, Vec<f16>)],
    opts: &ExportOptions,
) -> anyhow::Result<()> {
    let ext = ext_of(path);
    if ext == "jxl" {
        return write_jxl(path, (width, height), tiles, opts);
    }
    let format = ImageFormat::from_extension(&ext)
        .with_context(|| mosaix_core::tr!("unknown format “{ext}”", ext = ext))?;
    if !matches!(
        format,
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Tiff | ImageFormat::Avif
    ) {
        bail!(
            "{}",
            mosaix_core::tr!(
                "Export to “{ext}” is not supported (png, jpg, webp, tiff, avif, jxl)",
                ext = ext
            )
        );
    }
    if format == ImageFormat::Avif && opts.space.cicp().is_none() {
        bail!("{}", mosaix_core::tr!("AVIF cannot be saved in {space} (sRGB, Display P3 and Rec. 2020 can)", space = opts.space.name()));
    }
    let icc = opts.space.icc()?;
    if opts.space == ColorSpace::Srgb && !(opts.sixteen_bit && supports_16bit(path)) {
        // Fast path: lookup table.
        let img = tiles_to_rgba8(width, height, tiles);
        return encode8(&img, path, format, opts.quality, icc);
    }
    let convert = opts.space.converter()?;
    let converted = convert_tiles(tiles, &convert);
    if format == ImageFormat::Avif && opts.space != ColorSpace::Srgb {
        let mut img = RgbaImage::new(width, height);
        place(&mut img, (width, height), &converted, |v| ((v as u32 * 255 + 32767) / 65535) as u8);
        return write_avif_wide(path, (width, height), img.as_raw(), opts.quality, opts.space);
    }
    if opts.sixteen_bit && supports_16bit(path) {
        let mut img: ImageBuffer<Rgba<u16>, Vec<u16>> = ImageBuffer::new(width, height);
        place(&mut img, (width, height), &converted, |v| v);
        drop(converted);
        let bytes: &[u8] = bytemuck::cast_slice(img.as_raw());
        encode(
            path,
            format,
            (width, height),
            bytes,
            ExtendedColorType::Rgba16,
            100,
            icc,
        )
    } else {
        let mut img = RgbaImage::new(width, height);
        place(&mut img, (width, height), &converted, |v| {
            ((v as u32 * 255 + 32767) / 65535) as u8
        });
        drop(converted);
        encode8(&img, path, format, opts.quality, icc)
    }
}

/// AVIF in a wide-gamut space: encoded by ravif, then the container is
/// written again with the space's CICP codes in an "nclx" colour box
/// (ravif always describes sRGB).
fn write_avif_wide(path: &Path, (width, height): (u32, u32), rgba: &[u8], quality: u8, space: ColorSpace) -> anyhow::Result<()> {
    use avif_serialize::constants::{ColorPrimaries, MatrixCoefficients, TransferCharacteristics};
    let px: Vec<ravif::RGBA8> = rgba.as_chunks::<4>().0.iter().map(|p| ravif::RGBA8::new(p[0], p[1], p[2], p[3])).collect();
    let q = quality.clamp(1, 100) as f32;
    let enc = ravif::Encoder::new()
        .with_quality(q)
        .with_alpha_quality(q)
        .with_speed(6)
        .with_bit_depth(ravif::BitDepth::Ten)
        .with_alpha_color_mode(ravif::AlphaColorMode::UnassociatedClean);
    let r = enc
        .encode_rgba(ravif::Img::new(&px[..], width as usize, height as usize))
        .map_err(|e| anyhow!("AVIF: {e}"))?;
    // The AV1 data: colour, then alpha, at the end of the file ("mdat").
    let f = &r.avif_file;
    let n = r.color_byte_size + r.alpha_byte_size;
    anyhow::ensure!(f.len() >= n && f.windows(4).any(|w| w == b"mdat"), "AVIF: unexpected layout");
    let data = &f[f.len() - n..];
    let (color, alpha) = data.split_at(r.color_byte_size);
    let (cp, tc) = match space {
        ColorSpace::DisplayP3 => (ColorPrimaries::DisplayP3, TransferCharacteristics::Srgb),
        ColorSpace::Rec2020 => (ColorPrimaries::Bt2020, TransferCharacteristics::Bt709),
        _ => (ColorPrimaries::Bt709, TransferCharacteristics::Srgb),
    };
    let file = avif_serialize::Aviffy::new()
        .set_color_primaries(cp)
        .set_transfer_characteristics(tc)
        .set_matrix_coefficients(MatrixCoefficients::Bt601)
        .set_full_color_range(true)
        .to_vec(color, (!alpha.is_empty()).then_some(alpha), width, height, 10);
    std::fs::write(path, file).with_context(|| mosaix_core::tr!("creating {path}", path = path.display()))?;
    Ok(())
}

/// The ICC profile for an AVIF file's "nclx" colour box, if it names a
/// space other than sRGB that mosaix knows (the decoder ignores the box).
pub fn avif_nclx_icc(bytes: &[u8]) -> Option<Vec<u8>> {
    let i = bytes.windows(8).position(|w| w == b"colrnclx")?;
    let r = |o: usize| Some(u16::from_be_bytes([*bytes.get(i + o)?, *bytes.get(i + o + 1)?]));
    let (p, t) = (r(8)?, r(10)?);
    let space = ColorSpace::ALL.into_iter().find(|s| s.cicp() == Some((p, t)) || (s.cicp().map(|c| c.0) == Some(p) && p != 1))?;
    space.icc().ok().flatten()
}

/// Premultiplied linear tiles → straight RGBA, 16-bit, encoded by `convert`.
fn convert_tiles(tiles: &[(TileCoord, Vec<f16>)], convert: &Converter) -> Vec<(TileCoord, Vec<u16>)> {
    let ts = TILE_SIZE as usize;
    // Straight, encoded, 16-bit per tile.
    tiles
        .par_iter()
        .map(|(c, px)| {
            let mut rgb = vec![0.0f32; ts * ts * 3];
            let mut alpha = vec![0.0f32; ts * ts];
            for ((o, a), p) in rgb
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(&mut alpha)
                .zip(px.as_chunks::<4>().0)
            {
                *a = p[3].to_f32().clamp(0.0, 1.0);
                if *a > 0.0 {
                    for i in 0..3 {
                        o[i] = p[i].to_f32() / *a;
                    }
                }
            }
            for row in rgb.chunks_mut(ts * 3) {
                convert(row);
            }
            let q = |v: f32| (v * 65535.0).round() as u16;
            let mut out = vec![0u16; ts * ts * 4];
            for ((o, c), a) in out
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(rgb.as_chunks::<3>().0)
                .zip(&alpha)
            {
                if *a > 0.0 {
                    o.copy_from_slice(&[q(c[0]), q(c[1]), q(c[2]), q(*a)]);
                }
            }
            (*c, out)
        })
        .collect()
}

/// Copies converted tiles into an image, cropping at its edges.
fn place<P: Copy>(
    buf: &mut [P],
    (width, height): (u32, u32),
    tiles: &[(TileCoord, Vec<u16>)],
    q: impl Fn(u16) -> P,
) {
    let ts = TILE_SIZE as i64;
    let stride = width as usize * 4;
    for (c, data) in tiles {
        let (x0, y0) = (c.x as i64 * ts, c.y as i64 * ts);
        for row in 0..ts {
            let y = y0 + row;
            if y < 0 || y >= height as i64 {
                continue;
            }
            let xa = x0.max(0);
            let xb = (x0 + ts).min(width as i64);
            if xa >= xb {
                continue;
            }
            let src = (row * ts + (xa - x0)) as usize * 4;
            let dst = y as usize * stride + xa as usize * 4;
            let n = (xb - xa) as usize * 4;
            for (d, s) in buf[dst..dst + n].iter_mut().zip(&data[src..src + n]) {
                *d = q(*s);
            }
        }
    }
}

fn encode8(
    img: &RgbaImage,
    path: &Path,
    format: ImageFormat,
    quality: u8,
    icc: Option<Vec<u8>>,
) -> anyhow::Result<()> {
    let size = img.dimensions();
    if format == ImageFormat::Jpeg {
        // No alpha in JPEG: flatten onto white (in the encoded space).
        let mut rgb = RgbImage::new(size.0, size.1);
        for (o, p) in rgb.pixels_mut().zip(img.pixels()) {
            let a = p[3] as u32;
            *o = image::Rgb([0, 1, 2].map(|i| ((p[i] as u32 * a + 255 * (255 - a)) / 255) as u8));
        }
        return encode(
            path,
            format,
            size,
            rgb.as_raw(),
            ExtendedColorType::Rgb8,
            quality,
            icc,
        );
    }
    encode(
        path,
        format,
        size,
        img.as_raw(),
        ExtendedColorType::Rgba8,
        quality,
        icc,
    )
}

fn encode(
    path: &Path,
    format: ImageFormat,
    (w, h): (u32, u32),
    bytes: &[u8],
    color: ExtendedColorType,
    quality: u8,
    icc: Option<Vec<u8>>,
) -> anyhow::Result<()> {
    let file = BufWriter::new(
        std::fs::File::create(path)
            .with_context(|| mosaix_core::tr!("creating {path}", path = path.display()))?,
    );
    fn go(
        mut e: impl ImageEncoder,
        icc: Option<Vec<u8>>,
        bytes: &[u8],
        w: u32,
        h: u32,
        color: ExtendedColorType,
    ) -> anyhow::Result<()> {
        if let Some(icc) = icc {
            e.set_icc_profile(icc).map_err(|e| {
                anyhow!(mosaix_core::tr!("embedding color profile: {err}", err = e))
            })?;
        }
        e.write_image(bytes, w, h, color)?;
        Ok(())
    }
    match format {
        ImageFormat::Png => go(
            image::codecs::png::PngEncoder::new(file),
            icc,
            bytes,
            w,
            h,
            color,
        ),
        ImageFormat::Jpeg => go(
            image::codecs::jpeg::JpegEncoder::new_with_quality(file, quality.clamp(1, 100)),
            icc,
            bytes,
            w,
            h,
            color,
        ),
        ImageFormat::WebP => go(
            image::codecs::webp::WebPEncoder::new_lossless(file),
            icc,
            bytes,
            w,
            h,
            color,
        ),
        ImageFormat::Tiff => go(
            image::codecs::tiff::TiffEncoder::new(file),
            icc,
            bytes,
            w,
            h,
            color,
        ),
        ImageFormat::Avif => go(
            image::codecs::avif::AvifEncoder::new_with_speed_quality(file, 6, quality.clamp(1, 100)),
            None,
            bytes,
            w,
            h,
            color,
        ),
        _ => unreachable!(),
    }
}

/// JPEG XL through libjxl: 8 or 16 bit with alpha in any of the export
/// colour spaces (signalled as a colour encoding); quality 100 is lossless,
/// below that the JPEG-like quality scale.
fn write_jxl(path: &Path, (width, height): (u32, u32), tiles: &[(TileCoord, Vec<f16>)], opts: &ExportOptions) -> anyhow::Result<()> {
    use jpegxl_rs::encode::{EncoderFrame, encoder_builder};
    let lossless = opts.quality >= 100;
    let mut enc = encoder_builder().has_alpha(true).lossless(lossless).build().map_err(|e| anyhow!("JPEG XL: {e}"))?;
    if opts.space != ColorSpace::Srgb {
        enc.color_encoding = Some(jpegxl_rs::encode::ColorEncoding::Custom(opts.space.jxl_encoding()));
    }
    if lossless {
        enc.uses_original_profile = true;
    } else {
        enc.quality = jxl_distance(opts.quality);
    }
    let data = if opts.space != ColorSpace::Srgb {
        let conv = convert_tiles(tiles, &opts.space.converter()?);
        let mut img16 = vec![0u16; width as usize * height as usize * 4];
        place(&mut img16, (width, height), &conv, |v| v);
        if opts.sixteen_bit {
            enc.encode_frame(&EncoderFrame::new(&img16).num_channels(4), width, height)
        } else {
            let img8: Vec<u8> = img16.iter().map(|&v| ((v as u32 * 255 + 32767) / 65535) as u8).collect();
            enc.encode_frame(&EncoderFrame::new(&img8).num_channels(4), width, height)
        }
    } else if opts.sixteen_bit {
        let img16 = to_srgb16(width, height, tiles);
        enc.encode_frame(&EncoderFrame::new(&img16).num_channels(4), width, height)
    } else {
        let img = tiles_to_rgba8(width, height, tiles);
        enc.encode_frame(&EncoderFrame::new(img.as_raw()).num_channels(4), width, height)
    }
    .map_err(|e| anyhow!("JPEG XL: {e}"))?;
    std::fs::write(path, data).with_context(|| mosaix_core::tr!("creating {path}", path = path.display()))?;
    Ok(())
}

/// libjxl's distance for a JPEG-style quality (1–99), as `cjxl` maps it.
fn jxl_distance(q: u8) -> f32 {
    let q = q.clamp(1, 99) as f32;
    if q >= 30.0 { 0.1 + (100.0 - q) * 0.09 } else { 6.4 + 2.5f32.powf((30.0 - q) / 5.0) / 6.25 }
}

/// Premultiplied linear tiles → straight sRGB 16-bit RGBA (whole image).
fn to_srgb16(width: u32, height: u32, tiles: &[(TileCoord, Vec<f16>)]) -> Vec<u16> {
    let ts = TILE_SIZE as usize;
    let conv: Vec<(TileCoord, Vec<u16>)> = tiles
        .par_iter()
        .map(|(c, px)| {
            let mut out = vec![0u16; ts * ts * 4];
            for (o, p) in out.chunks_mut(4).zip(px.chunks(4)) {
                let a = p[3].to_f32().clamp(0.0, 1.0);
                if a > 0.0 {
                    for k in 0..3 {
                        o[k] = (linear_to_srgb((p[k].to_f32() / a).clamp(0.0, 1.0)) * 65535.0).round() as u16;
                    }
                    o[3] = (a * 65535.0).round() as u16;
                }
            }
            (*c, out)
        })
        .collect();
    let mut buf = vec![0u16; width as usize * height as usize * 4];
    place(&mut buf, (width, height), &conv, |v| v);
    buf
}

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
pub fn blit_region(
    img: &mut RgbaImage,
    tiles: &[(TileCoord, Vec<f16>)],
    tile_px: u32,
    origin: (i64, i64),
) {
    let (width, height) = img.dimensions();
    let lut: Vec<u8> = (0..=4096)
        .map(|i| (linear_to_srgb(i as f32 / 4096.0) * 255.0).round() as u8)
        .collect();
    let to8 = |v: f32| lut[(v.clamp(0.0, 1.0) * 4096.0).round() as usize];
    let ts = tile_px as usize;
    let converted: Vec<(TileCoord, Vec<u8>)> = tiles
        .par_iter()
        .map(|(c, px)| {
            let mut out = vec![0u8; ts * ts * 4];
            for (o, p) in out
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(px.as_chunks::<4>().0)
            {
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

/// Saves an sRGB image by file extension: png, jpg/jpeg (flattened onto
/// white), webp, tif/tiff.
pub fn save(img: &RgbaImage, path: &Path) -> anyhow::Result<()> {
    let ext = ext_of(path);
    let format = ImageFormat::from_extension(&ext)
        .with_context(|| mosaix_core::tr!("unknown format “{ext}”", ext = ext))?;
    if !matches!(
        format,
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Tiff
    ) {
        bail!(
            "{}",
            mosaix_core::tr!(
                "Export to “{ext}” is not supported (png, jpg, webp, tiff)",
                ext = ext
            )
        );
    }
    encode8(img, path, format, ExportOptions::default().quality, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_avif_and_jpeg_xl() {
        let ts = TILE_SIZE as usize;
        let mut t = vec![f16::ZERO; ts * ts * 4];
        for p in t.chunks_mut(4) {
            p.copy_from_slice(&[0.2, 0.4, 0.1, 1.0].map(f16::from_f32));
        }
        let tiles = vec![(TileCoord::new(0, 0), t)];
        let dir = tempfile::tempdir().unwrap();
        for (name, q, sixteen) in [("a.avif", 80, false), ("a.jxl", 90, false), ("l.jxl", 100, true)] {
            let p = dir.path().join(name);
            write(&p, (64, 48), &tiles, &ExportOptions { quality: q, sixteen_bit: sixteen, ..Default::default() }).unwrap();
            let b = std::fs::read(&p).unwrap();
            if name.ends_with("avif") {
                assert_eq!(&b[4..12], b"ftypavif", "AVIF container");
                let img = image::open(&p).unwrap().into_rgba8();
                assert_eq!(img.dimensions(), (64, 48));
            } else {
                assert!(b.starts_with(&[0xff, 0x0a]) || b.starts_with(b"\0\0\0\x0cJXL "), "JPEG XL signature");
            }
        }
    }

    #[test]
    fn wide_gamut_avif_and_jpeg_xl() {
        let ts = TILE_SIZE as usize;
        // Pure sRGB red: inside the wider spaces it is less saturated.
        let mut t = vec![f16::ZERO; ts * ts * 4];
        for p in t.chunks_mut(4) {
            p.copy_from_slice(&[1.0, 0.0, 0.0, 1.0].map(f16::from_f32));
        }
        let tiles = vec![(TileCoord::new(0, 0), t)];
        let dir = tempfile::tempdir().unwrap();
        for space in [ColorSpace::DisplayP3, ColorSpace::Rec2020, ColorSpace::AdobeRgb] {
            // Expected encoded values of sRGB red in that space.
            let mut want = [1.0f32, 0.0, 0.0];
            space.converter().unwrap()(&mut want);
            let near = |px: &[u8], tol: f32| (0..3).all(|k| (px[k] as f32 / 255.0 - want[k]).abs() < tol);
            // JPEG XL (lossless, so exact), with the colour encoding signalled.
            let p = dir.path().join("w.jxl");
            let opts = ExportOptions { space, quality: 100, ..Default::default() };
            write(&p, (64, 48), &tiles, &opts).unwrap();
            let dec = jpegxl_rs::decoder_builder().icc_profile(true).build().unwrap();
            let (meta, px) = dec.decode_with::<u8>(&std::fs::read(&p).unwrap()).unwrap();
            assert!(near(&px, 1.5 / 255.0), "{space:?} JXL {:?} vs {want:?}", &px[..4]);
            let icc = meta.icc_profile.expect("colour profile");
            let prof = moxcms::ColorProfile::new_from_slice(&icc).unwrap();
            // Wider than sRGB: the green primary differs (Adobe RGB shares red and blue).
            let xy = |c: moxcms::Xyzd| (c.x / (c.x + c.y + c.z), c.y / (c.x + c.y + c.z));
            let (g, s) = (xy(prof.green_colorant), xy(moxcms::ColorProfile::new_srgb().green_colorant));
            assert!((g.0 - s.0).hypot(g.1 - s.1) > 0.05, "{space:?}: not sRGB's primaries");
            // AVIF: only spaces with a CICP code.
            let p = dir.path().join("w.avif");
            let r = write(&p, (64, 48), &tiles, &ExportOptions { space, quality: 95, ..Default::default() });
            if space.cicp().is_none() {
                assert!(r.is_err());
                continue;
            }
            r.unwrap();
            let b = std::fs::read(&p).unwrap();
            let img = image::open(&p).unwrap().into_rgba8();
            assert!(near(img.get_pixel(10, 10).0.as_slice(), 0.03), "{space:?} AVIF {:?} vs {want:?}", img.get_pixel(10, 10));
            assert_eq!(avif_nclx_icc(&b), space.icc().unwrap(), "{space:?} signalled");
        }
    }

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
        let dir = std::env::temp_dir().join(format!("mosaix-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.jpg");
        save(&RgbaImage::new(8, 8), &p).unwrap();
        let back = image::open(&p).unwrap().to_rgb8();
        assert!(back.get_pixel(4, 4).0.iter().all(|&v| v > 250));
        assert!(save(&RgbaImage::new(1, 1), &dir.join("t.xyz")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn red_tile() -> Vec<f16> {
        let ts = TILE_SIZE as usize;
        let mut px = vec![f16::ZERO; ts * ts * 4];
        for p in px.as_chunks_mut::<4>().0 {
            p.copy_from_slice(&[f16::ONE, f16::ZERO, f16::ZERO, f16::ONE]);
        }
        px
    }

    fn read_back(p: &Path) -> (image::DynamicImage, Option<Vec<u8>>) {
        use image::ImageDecoder;
        let mut dec = image::ImageReader::open(p)
            .unwrap()
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap();
        let icc = dec.icc_profile().unwrap();
        (image::DynamicImage::from_decoder(dec).unwrap(), icc)
    }

    #[test]
    fn wide_gamut_export_embeds_profile() {
        let dir = tempfile::tempdir().unwrap();
        let tiles = [(TileCoord::new(0, 0), red_tile())];
        // sRGB red expressed in Display P3 ≈ (0.9175, 0.2003, 0.1386).
        let opts = ExportOptions {
            space: ColorSpace::DisplayP3,
            sixteen_bit: true,
            quality: 90,
        };
        for name in ["p3.png", "p3.tif"] {
            let p = dir.path().join(name);
            write(&p, (100, 50), &tiles, &opts).unwrap();
            let (img, icc) = read_back(&p);
            let img = img.to_rgba16();
            assert_eq!(img.dimensions(), (100, 50));
            let px = img.get_pixel(10, 10).0.map(|v| v as f32 / 65535.0);
            for (got, want) in px.iter().zip([0.9175, 0.2003, 0.1386, 1.0]) {
                assert!((got - want).abs() < 0.003, "{name}: {px:?}");
            }
            let prof = moxcms::ColorProfile::new_from_slice(&icc.expect("ICC embedded")).unwrap();
            assert_eq!(
                prof.red_colorant,
                moxcms::ColorProfile::new_display_p3().red_colorant
            );
        }
        // 8-bit formats get the profile too; sRGB stays untagged.
        for name in ["p3.jpg", "p3.webp"] {
            let p = dir.path().join(name);
            write(&p, (100, 50), &tiles, &opts).unwrap();
            let (img, icc) = read_back(&p);
            assert!(icc.is_some(), "{name}");
            let r = img.to_rgb8().get_pixel(10, 10).0;
            assert!(
                (r[0] as i32 - 234).abs() <= 2 && (r[1] as i32 - 51).abs() <= 3,
                "{name}: {r:?}"
            );
        }
        let p = dir.path().join("s.png");
        write(&p, (100, 50), &tiles, &ExportOptions::default()).unwrap();
        let (img, icc) = read_back(&p);
        assert!(icc.is_none());
        assert_eq!(img.to_rgba8().get_pixel(1, 1).0, [255, 0, 0, 255]);
        // sRGB 16-bit.
        let p = dir.path().join("s16.png");
        write(
            &p,
            (100, 50),
            &tiles,
            &ExportOptions {
                sixteen_bit: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            read_back(&p).0.to_rgba16().get_pixel(1, 1).0,
            [65535, 0, 0, 65535]
        );
    }

    #[test]
    #[ignore = "benchmark"]
    fn wide_gamut_export_speed() {
        let dir = tempfile::tempdir().unwrap();
        let (w, h) = (6000u32, 4000u32);
        let tiles: Vec<_> = (0..h.div_ceil(TILE_SIZE) as i32)
            .flat_map(|y| {
                (0..w.div_ceil(TILE_SIZE) as i32).map(move |x| (TileCoord::new(x, y), red_tile()))
            })
            .collect();
        for (name, opts) in [
            ("s.tif", ExportOptions::default()),
            (
                "p3.tif",
                ExportOptions {
                    space: ColorSpace::DisplayP3,
                    ..Default::default()
                },
            ),
            (
                "p3-16.tif",
                ExportOptions {
                    space: ColorSpace::DisplayP3,
                    sixteen_bit: true,
                    ..Default::default()
                },
            ),
        ] {
            let t = std::time::Instant::now();
            write(&dir.path().join(name), (w, h), &tiles, &opts).unwrap();
            eprintln!("{name}: {:?}", t.elapsed());
        }
    }
}
