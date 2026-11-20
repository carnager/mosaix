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
use imagine_io::native::{Reader, decompress_tile};
use rayon::prelude::*;

pub enum Source {
    File(PathBuf),
    /// Procedural test image.
    Demo(u32, u32),
    /// Empty transparent document.
    Blank(u32, u32),
}

pub type TileData = (TileCoord, Vec<f16>, Option<BlobRef>);

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
            let structure = doc.map_tiles(&mut |_| ());
            // Structure without tiles: strip the tile maps.
            let mut empty = structure.clone();
            let ids = empty.display_order();
            for id in ids {
                if let Some(t) = empty.tiles_mut(id) {
                    t.clear();
                }
            }
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
            send(LoaderMsg::Done { decode_ms: 0.0, total_ms: ms(t0) })?;
        }
        Source::File(path) => {
            let name = file_name(&path);
            let (w, h) = image::image_dimensions(&path)?;
            let (msg, layer) = single_layer(w, h, name, Some(path.clone()));
            send(msg)?;

            let mut reader = ImageReader::open(&path)?.with_guessed_format()?;
            reader.no_limits();
            let img = reader.decode()?;
            let decode_ms = ms(t0);
            tracing::info!("dekodiert {w}×{h} {:?} in {decode_ms:.0} ms", img.color());

            match img {
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
            }
            send(LoaderMsg::Done { decode_ms, total_ms: ms(t0) })?;
        }
    }
    Ok(())
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
