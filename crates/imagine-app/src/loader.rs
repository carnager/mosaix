//! Background document loading. Tiles are converted in parallel and streamed
//! to the UI thread centre-first, so the image appears progressively instead
//! of after one long freeze.

use std::path::PathBuf;
use std::time::Instant;

use half::f16;
use image::{DynamicImage, ImageReader};
use imagine_core::color::{Pixels, extract_tile};
use imagine_core::tile::grid_size;
use imagine_core::{TILE_SIZE, TileCoord};
use rayon::prelude::*;
use smithay_client_toolkit::reexports::calloop::channel::Sender;

pub enum Source {
    File(PathBuf),
    /// Procedural test image.
    Demo(u32, u32),
    /// Empty transparent document.
    Blank(u32, u32),
}

pub enum LoaderMsg {
    /// Document size is known (sent before decoding starts).
    Size { width: u32, height: u32, name: String, expected_tiles: u32 },
    Tiles(Vec<(TileCoord, Vec<f16>)>),
    Done { decode_ms: f64, total_ms: f64 },
    Error(String),
}

const CHUNK: usize = 48;

pub fn spawn(source: Source, tx: Sender<LoaderMsg>) {
    std::thread::Builder::new()
        .name("loader".into())
        .spawn(move || {
            if let Err(e) = run(source, &tx) {
                let _ = tx.send(LoaderMsg::Error(format!("{e:#}")));
            }
        })
        .expect("spawn loader thread");
}

fn run(source: Source, tx: &Sender<LoaderMsg>) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let send = |m| tx.send(m).map_err(|_| anyhow::anyhow!("UI gone"));
    match source {
        Source::Blank(w, h) => {
            send(LoaderMsg::Size { width: w, height: h, name: "Unbenannt".into(), expected_tiles: 0 })?;
            send(LoaderMsg::Done { decode_ms: 0.0, total_ms: 0.0 })?;
        }
        Source::Demo(w, h) => {
            let (gw, gh) = grid_size(w, h);
            send(LoaderMsg::Size { width: w, height: h, name: format!("Demo {w}×{h}"), expected_tiles: gw * gh })?;
            stream_tiles(w, h, tx, |tx_, ty_| Some(demo_tile(tx_, ty_, w, h)))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            send(LoaderMsg::Done { decode_ms: 0.0, total_ms: ms })?;
        }
        Source::File(path) => {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let (w, h) = image::image_dimensions(&path)?;
            let (gw, gh) = grid_size(w, h);
            send(LoaderMsg::Size { width: w, height: h, name, expected_tiles: gw * gh })?;

            let mut reader = ImageReader::open(&path)?.with_guessed_format()?;
            reader.no_limits();
            let img = reader.decode()?;
            let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
            tracing::info!("decoded {}×{} {:?} in {decode_ms:.0} ms", w, h, img.color());

            match img {
                DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_) => {
                    let buf = img.into_rgba32f();
                    let px = Pixels::Linear32(&buf);
                    stream_tiles(w, h, tx, |x, y| extract_tile(&px, w, h, x, y))?;
                }
                DynamicImage::ImageLuma16(_)
                | DynamicImage::ImageLumaA16(_)
                | DynamicImage::ImageRgb16(_)
                | DynamicImage::ImageRgba16(_) => {
                    let buf = img.into_rgba16();
                    let px = Pixels::Srgb16(&buf);
                    stream_tiles(w, h, tx, |x, y| extract_tile(&px, w, h, x, y))?;
                }
                _ => {
                    let buf = img.into_rgba8();
                    let px = Pixels::Srgb8(&buf);
                    stream_tiles(w, h, tx, |x, y| extract_tile(&px, w, h, x, y))?;
                }
            }
            let total_ms = t0.elapsed().as_secs_f64() * 1e3;
            send(LoaderMsg::Done { decode_ms, total_ms })?;
        }
    }
    Ok(())
}

/// Converts all tiles in parallel, centre-first, and sends them in chunks.
fn stream_tiles(
    w: u32,
    h: u32,
    tx: &Sender<LoaderMsg>,
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
            .filter_map(|&(x, y)| make(x, y).map(|t| (TileCoord::new(x as i32, y as i32), t)))
            .collect();
        if !tiles.is_empty() {
            tx.send(LoaderMsg::Tiles(tiles)).map_err(|_| anyhow::anyhow!("UI gone"))?;
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
