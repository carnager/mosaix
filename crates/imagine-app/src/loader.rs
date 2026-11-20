//! Background document loading. Tiles are converted in parallel and streamed
//! to the UI thread centre-first, so the image appears progressively instead
//! of after one long freeze.

use std::path::{Path, PathBuf};
use std::time::Instant;

use half::f16;
use image::{DynamicImage, ImageReader};
use imagine_core::color::{Pixels, extract_tile};
use imagine_core::tile::grid_size;
use imagine_core::{Document, LayerId, TILE_SIZE, TileCoord};
use imagine_io::BlobRef;
use imagine_core::mask::MaskData;
use imagine_core::MaskTile;
use imagine_io::native::{Reader, decompress_mask, decompress_tile};
use std::collections::HashMap;
use rayon::prelude::*;

pub enum Source {
    File(PathBuf),
    /// Procedural test image.
    Demo(u32, u32),
    /// Empty transparent document.
    Blank(u32, u32),
}

pub type TileData = (TileCoord, Vec<f16>, Option<BlobRef>);

/// A layer mask (`layer: Some`) or the selection (`None`) read from a file.
pub struct MaskLoad {
    pub layer: Option<LayerId>,
    pub default: u8,
    pub enabled: bool,
    pub tiles: HashMap<TileCoord, MaskData>,
    pub blobs: HashMap<TileCoord, BlobRef>,
}

pub enum LoaderMsg {
    /// Document structure (no pixels yet). Sent before decoding starts.
    Structure {
        doc: Document<()>,
        name: String,
        source: Option<PathBuf>,
        /// Set when the source is a native file: its length (for incremental saves).
        native_len: Option<u64>,
        expected_tiles: u64,
    },
    Tiles { layer: LayerId, tiles: Vec<TileData> },
    Mask(MaskLoad),
    Done { decode_ms: f64, total_ms: f64 },
    Error(String),
}

/// Where loader messages go (calloop channel for the window, std channel headless).
pub trait Sink: Send + 'static {
    fn send(&self, m: LoaderMsg) -> bool;
}

impl Sink for smithay_client_toolkit::reexports::calloop::channel::Sender<LoaderMsg> {
    fn send(&self, m: LoaderMsg) -> bool {
        smithay_client_toolkit::reexports::calloop::channel::Sender::send(self, m).is_ok()
    }
}

impl Sink for std::sync::mpsc::Sender<LoaderMsg> {
    fn send(&self, m: LoaderMsg) -> bool {
        std::sync::mpsc::Sender::send(self, m).is_ok()
    }
}

const CHUNK: usize = 48;

pub fn spawn(source: Source, tx: impl Sink) {
    std::thread::Builder::new()
        .name("loader".into())
        .spawn(move || run(source, &tx))
        .expect("spawn loader thread");
}

pub fn run(source: Source, tx: &impl Sink) {
    if let Err(e) = load(source, tx) {
        tx.send(LoaderMsg::Error(format!("{e:#}")));
    }
}

fn is_native(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("imagine"))
}

fn load(source: Source, tx: &impl Sink) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let send = |m| if tx.send(m) { Ok(()) } else { Err(anyhow::anyhow!("UI beendet")) };
    let single_layer = |w, h, name: String, source: Option<PathBuf>| {
        let doc = Document::<()>::new(w, h);
        let layer = doc.active;
        let (gw, gh) = grid_size(w, h);
        let msg = LoaderMsg::Structure { doc, name, source, native_len: None, expected_tiles: gw as u64 * gh as u64 };
        (msg, layer)
    };
    match source {
        Source::Blank(w, h) => {
            let (mut msg, _) = single_layer(w, h, "Unbenannt".into(), None);
            if let LoaderMsg::Structure { expected_tiles, .. } = &mut msg {
                *expected_tiles = 0;
            }
            send(msg)?;
            send(LoaderMsg::Done { decode_ms: 0.0, total_ms: 0.0 })?;
        }
        Source::Demo(w, h) => {
            let (msg, layer) = single_layer(w, h, format!("Demo {w}×{h}"), None);
            send(msg)?;
            stream_tiles(w, h, layer, tx, |x, y| Some(demo_tile(x, y, w, h)))?;
            send(LoaderMsg::Done { decode_ms: 0.0, total_ms: ms(t0) })?;
        }
        Source::File(path) if is_native(&path) => {
            let name = file_name(&path);
            let (reader, doc) = Reader::open(&path)?;
            let mut jobs: Vec<(LayerId, TileCoord, BlobRef)> = Vec::new();
            doc.for_each(&mut |l, _| {
                if let Some(t) = l.tiles() {
                    jobs.extend(t.iter().map(|(c, b)| (l.id, *c, *b)));
                }
            });
            // Structure without pixel data and masks (those follow as messages).
            let mut empty = doc.map_tiles(&mut |_| ());
            let ids = empty.display_order();
            for id in &ids {
                if let Some(t) = empty.tiles_mut(*id) {
                    t.clear();
                }
                empty.find_mut(*id).unwrap().mask = None;
            }
            empty.selection = None;
            send(LoaderMsg::Structure {
                doc: empty,
                name,
                source: Some(path.clone()),
                native_len: Some(reader.len),
                expected_tiles: jobs.len() as u64,
            })?;
            jobs.sort_by_key(|j| j.0);
            for layer_jobs in jobs.chunk_by(|a, b| a.0 == b.0) {
                let layer = layer_jobs[0].0;
                for chunk in layer_jobs.chunks(CHUNK) {
                    let tiles = chunk
                        .par_iter()
                        .map(|&(_, c, b)| Ok((c, decompress_tile(&reader.read_blob(b)?)?, Some(b))))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    send(LoaderMsg::Tiles { layer, tiles })?;
                }
            }
            let mut masks: Vec<(Option<LayerId>, imagine_core::Mask<BlobRef>, bool)> = Vec::new();
            doc.for_each(&mut |l, _| {
                if let Some(m) = &l.mask {
                    masks.push((Some(l.id), m.clone(), l.mask_enabled));
                }
            });
            if let Some(sel) = &doc.selection {
                masks.push((None, sel.clone(), true));
            }
            for (layer, m, enabled) in masks {
                let entries: Vec<_> = m.tiles.iter().map(|(c, t)| (*c, t.clone())).collect();
                let decoded = entries
                    .par_iter()
                    .map(|(c, t)| {
                        Ok(match t {
                            MaskTile::Solid(v) => (*c, MaskData::Solid(*v), None),
                            MaskTile::Pixels(b) => (*c, MaskData::Pixels(decompress_mask(&reader.read_blob(*b)?)?), Some(*b)),
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let blobs = decoded.iter().filter_map(|(c, _, b)| b.map(|b| (*c, b))).collect();
                let tiles = decoded.into_iter().map(|(c, d, _)| (c, d)).collect();
                send(LoaderMsg::Mask(MaskLoad { layer, default: m.default, enabled, tiles, blobs }))?;
            }
            send(LoaderMsg::Done { decode_ms: 0.0, total_ms: ms(t0) })?;
        }
        Source::File(path) => {
            let name = file_name(&path);
            let (w, h) = image::image_dimensions(&path)?;
            let (msg, layer) = single_layer(w, h, name, Some(path.clone()));
            send(msg)?;

            let mut reader = ImageReader::open(&path)?.with_guessed_format()?;
            reader.no_limits();
            let mut decoder = reader.into_decoder()?;
            let icc = image::ImageDecoder::icc_profile(&mut decoder).ok().flatten();
            let img = DynamicImage::from_decoder(decoder)?;
            let decode_ms = ms(t0);
            tracing::info!("dekodiert {w}×{h} {:?} in {decode_ms:.0} ms", img.color());

            match icc.as_deref().and_then(color_transform) {
                Some((name, convert)) => {
                    tracing::info!("Farbprofil „{name}“ wird nach sRGB (linear, erweiterter Farbraum) umgerechnet");
                    let buf = img.into_rgba16();
                    let px = Pixels::Converted16 { data: &buf, convert: &*convert };
                    stream_tiles(w, h, layer, tx, |x, y| extract_tile(&px, w, h, x, y))?;
                }
                None => match img {
                    DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_) => {
                        let buf = img.into_rgba32f();
                        let px = Pixels::Linear32(&buf);
                        stream_tiles(w, h, layer, tx, |x, y| extract_tile(&px, w, h, x, y))?;
                    }
                    DynamicImage::ImageLuma16(_)
                    | DynamicImage::ImageLumaA16(_)
                    | DynamicImage::ImageRgb16(_)
                    | DynamicImage::ImageRgba16(_) => {
                        let buf = img.into_rgba16();
                        let px = Pixels::Srgb16(&buf);
                        stream_tiles(w, h, layer, tx, |x, y| extract_tile(&px, w, h, x, y))?;
                    }
                    _ => {
                        let buf = img.into_rgba8();
                        let px = Pixels::Srgb8(&buf);
                        stream_tiles(w, h, layer, tx, |x, y| extract_tile(&px, w, h, x, y))?;
                    }
                },
            }
            send(LoaderMsg::Done { decode_ms, total_ms: ms(t0) })?;
        }
    }
    Ok(())
}

/// Converter from an embedded ICC profile to linear sRGB, unless the
/// profile already is sRGB (or cannot be used, then the image is taken as sRGB).
pub fn color_transform(icc: &[u8]) -> Option<(String, Box<dyn Fn(&mut [f32]) + Send + Sync>)> {
    use moxcms::{ColorProfile, Layout, ToneReprCurve, TransformOptions};
    let src = match ColorProfile::new_from_slice(icc) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("Eingebettetes Farbprofil unlesbar ({e:?}), nehme sRGB an");
            return None;
        }
    };
    let name = match &src.description {
        Some(moxcms::ProfileText::Localizable(v)) => v.first().map(|s| s.value.clone()).unwrap_or_default(),
        Some(moxcms::ProfileText::PlainString(s)) => s.clone(),
        _ => String::new(),
    };
    let srgb = ColorProfile::new_srgb();
    let same_primaries = src.red_colorant == srgb.red_colorant
        && src.green_colorant == srgb.green_colorant
        && src.blue_colorant == srgb.blue_colorant;
    if name.to_lowercase().contains("srgb") || (same_primaries && src.red_trc == srgb.red_trc) {
        return None;
    }
    let mut linear = ColorProfile::new_srgb();
    let identity = ToneReprCurve::Lut(vec![]);
    linear.red_trc = Some(identity.clone());
    linear.green_trc = Some(identity.clone());
    linear.blue_trc = Some(identity);
    linear.cicp = None;
    let opts = TransformOptions { prefer_fixed_point: false, allow_extended_range_rgb_xyz: true, ..Default::default() };
    match src.create_transform_f32(Layout::Rgb, &linear, Layout::Rgb, opts) {
        Ok(t) => Some((
            if name.is_empty() { "unbenannt".into() } else { name },
            Box::new(move |rgb: &mut [f32]| {
                let input = rgb.to_vec();
                if t.transform(&input, rgb).is_err() {
                    rgb.iter_mut().for_each(|v| *v = imagine_core::color::srgb_to_linear(*v));
                }
            }),
        )),
        Err(e) => {
            tracing::warn!("Farbprofil „{name}“ nicht umrechenbar ({e:?}), nehme sRGB an");
            None
        }
    }
}

fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Converts all tiles in parallel, centre-first, and sends them in chunks.
fn stream_tiles(
    w: u32,
    h: u32,
    layer: LayerId,
    tx: &impl Sink,
    make: impl Fn(u32, u32) -> Option<Vec<f16>> + Sync,
) -> anyhow::Result<()> {
    let (gw, gh) = grid_size(w, h);
    let (cx, cy) = (gw as f64 / 2.0, gh as f64 / 2.0);
    let mut coords: Vec<(u32, u32)> = (0..gh).flat_map(|y| (0..gw).map(move |x| (x, y))).collect();
    coords.sort_by(|a, b| {
        let d = |(x, y): (u32, u32)| (x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2);
        d(*a).total_cmp(&d(*b))
    });
    for chunk in coords.chunks(CHUNK) {
        let tiles: Vec<_> = chunk
            .par_iter()
            .filter_map(|&(x, y)| make(x, y).map(|t| (TileCoord::new(x as i32, y as i32), t, None)))
            .collect();
        if !tiles.is_empty() && !tx.send(LoaderMsg::Tiles { layer, tiles }) {
            anyhow::bail!("UI beendet");
        }
    }
    Ok(())
}

/// A colourful procedural test pattern (linear, premultiplied, opaque).
fn demo_tile(tx: u32, ty: u32, w: u32, h: u32) -> Vec<f16> {
    let ts = TILE_SIZE as usize;
    let mut out = vec![f16::ZERO; ts * ts * 4];
    let (fw, fh) = (w as f32, h as f32);
    for row in 0..ts {
        let y = (ty as usize * ts + row) as f32;
        if y >= fh {
            break;
        }
        for col in 0..ts {
            let x = (tx as usize * ts + col) as f32;
            if x >= fw {
                break;
            }
            let (u, v) = (x / fw, y / fh);
            let r = ((x - fw * 0.5).hypot(y - fh * 0.5) / 40.0).sin() * 0.5 + 0.5;
            let grid = if (x as u32 % 512) < 2 || (y as u32 % 512) < 2 { 0.0 } else { 1.0 };
            let i = (row * ts + col) * 4;
            out[i] = f16::from_f32(u * u * grid);
            out[i + 1] = f16::from_f32(r * 0.6 * grid);
            out[i + 2] = f16::from_f32(v * grid);
            out[i + 3] = f16::ONE;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_with_profile(path: &Path, icc: Option<Vec<u8>>, px: [u8; 4]) {
        use image::ImageEncoder;
        let file = std::fs::File::create(path).unwrap();
        let mut enc = image::codecs::png::PngEncoder::new(file);
        if let Some(icc) = icc {
            enc.set_icc_profile(icc).unwrap();
        }
        let data: Vec<u8> = px.repeat(4);
        enc.write_image(&data, 2, 2, image::ExtendedColorType::Rgba8).unwrap();
    }

    fn first_pixel(path: &Path) -> [f32; 4] {
        let (tx, rx) = std::sync::mpsc::channel();
        run(Source::File(path.to_path_buf()), &tx);
        for m in rx {
            if let LoaderMsg::Tiles { tiles, .. } = m {
                return [0, 1, 2, 3].map(|i| tiles[0].1[i].to_f32());
            }
        }
        panic!("no tiles");
    }

    #[test]
    fn display_p3_is_converted_and_srgb_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let p3 = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let path = dir.path().join("p3.png");
        png_with_profile(&path, Some(p3), [255, 0, 0, 255]);
        let px = first_pixel(&path);
        // P3 red lies outside sRGB: more than 1 in red, negative green/blue.
        assert!((px[0] - 1.2249).abs() < 0.01, "{px:?}");
        assert!(px[1] < -0.02 && px[2] < -0.01, "{px:?}");

        let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
        let path = dir.path().join("srgb.png");
        png_with_profile(&path, Some(srgb), [255, 0, 0, 255]);
        assert_eq!(first_pixel(&path)[..3], [1.0, 0.0, 0.0]);
        let path = dir.path().join("plain.png");
        png_with_profile(&path, None, [128, 0, 0, 255]);
        assert!((first_pixel(&path)[0] - imagine_core::color::srgb_to_linear(128.0 / 255.0)).abs() < 1e-3);
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// `cargo test --release -p imagine-app icc_load_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn icc_load_speed() {
        use image::ImageEncoder;
        let dir = tempfile::tempdir().unwrap();
        let (w, h) = (6000u32, 4000u32);
        let data: Vec<u8> = (0..w * h).flat_map(|i| [(i % 251) as u8, (i % 241) as u8, (i % 239) as u8]).collect();
        for (name, icc) in [("srgb", None), ("p3", Some(moxcms::ColorProfile::new_display_p3().encode().unwrap()))] {
            let path = dir.path().join(format!("{name}.png"));
            let mut enc = image::codecs::png::PngEncoder::new_with_quality(
                std::fs::File::create(&path).unwrap(),
                image::codecs::png::CompressionType::Fast,
                image::codecs::png::FilterType::NoFilter,
            );
            if let Some(icc) = icc {
                enc.set_icc_profile(icc).unwrap();
            }
            enc.write_image(&data, w, h, image::ExtendedColorType::Rgb8).unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let t0 = Instant::now();
            run(Source::File(path), &tx);
            drop(tx);
            let n: usize = rx.iter().map(|m| if let LoaderMsg::Tiles { tiles, .. } = m { tiles.len() } else { 0 }).sum();
            println!("{name}: {n} Kacheln in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
    }
}
