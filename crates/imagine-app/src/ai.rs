//! Local AI selections: the visible image is read (scaled to ~1–2k px),
//! the model runs on the CPU (crate imagine-ai), and its mask is scaled
//! back up to a selection or layer mask. Models stay loaded; the SAM
//! encoding of the last image is kept, so further clicks take ~0.1 s.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use anyhow::{Result, anyhow, bail};
use imagine_ai::{Embedding, Image, Mask, Model, Point, Sam, Subject};
use imagine_core::mask::{CpuMask, MaskData};
use imagine_core::{TILE_SIZE, TileCoord, tr};
use rayon::prelude::*;

use crate::session::Session;

/// The visible image, scaled down, and the document size.
pub struct Snapshot {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub doc: (u32, u32),
    key: u64,
}

/// Longest side read for the models (they work at 1024).
const READ_MAX: u32 = 2048;

/// Reads the composite (main thread; the GPU is involved).
pub fn snapshot(s: &mut Session) -> Snapshot {
    let img = crate::agent::command::read_composite(s, READ_MAX);
    let (w, h) = (img.width() as usize, img.height() as usize);
    // Transparent areas count as white.
    let rgb: Vec<u8> = img
        .pixels()
        .flat_map(|p| {
            let a = p[3] as u32;
            [0, 1, 2].map(|i| ((p[i] as u32 * a + 255 * (255 - a)) / 255) as u8)
        })
        .collect();
    let mut hsh = std::collections::hash_map::DefaultHasher::new();
    rgb.hash(&mut hsh);
    (w, h).hash(&mut hsh);
    Snapshot { rgb, width: w, height: h, doc: s.engine.size(), key: hsh.finish() }
}

impl Snapshot {
    fn image(&self) -> Image<'_> {
        Image { width: self.width, height: self.height, rgb: &self.rgb }
    }
}

#[derive(Default)]
struct Loaded {
    sam: Option<Sam>,
    embedding: Option<(u64, Embedding)>,
    subject: Option<Subject>,
}

fn loaded() -> &'static Mutex<Loaded> {
    static L: std::sync::OnceLock<Mutex<Loaded>> = std::sync::OnceLock::new();
    L.get_or_init(Default::default)
}

/// Whether a model can be used now (`Err` explains: no runtime, not downloaded).
pub fn ready(m: &Model) -> Result<()> {
    imagine_ai::available()?;
    if imagine_ai::models::path(m).is_none() {
        bail!(
            "{}",
            tr!(
                "The model {name} ({mb} MB) is not downloaded yet",
                name = m.name,
                mb = m.size() / 1_000_000
            )
        );
    }
    Ok(())
}

/// The main subject as a mask at document size (worker thread).
pub fn subject(snap: &Snapshot) -> Result<CpuMask> {
    ready(&imagine_ai::SUBJECT)?;
    let mut l = loaded().lock().map_err(|_| anyhow!("model lock poisoned"))?;
    if l.subject.is_none() {
        l.subject = Some(Subject::load()?);
    }
    let m = l.subject.as_mut().unwrap().mask(&snap.image())?;
    Ok(to_cpu(&m, snap.doc))
}

/// The object(s) at document points (`true` = belongs to it), at document size.
pub fn object(snap: &Snapshot, points: &[(f64, f64, bool)]) -> Result<CpuMask> {
    ready(&imagine_ai::SAM)?;
    let mut l = loaded().lock().map_err(|_| anyhow!("model lock poisoned"))?;
    if l.sam.is_none() {
        l.sam = Some(Sam::load()?);
    }
    if l.embedding.as_ref().is_none_or(|(k, _)| *k != snap.key) {
        let t0 = std::time::Instant::now();
        let e = l.sam.as_mut().unwrap().encode(&snap.image())?;
        tracing::info!("{}", tr!("Image analysed in {ms} ms", ms = t0.elapsed().as_millis()));
        l.embedding = Some((snap.key, e));
    }
    let (kx, ky) = (snap.width as f64 / snap.doc.0 as f64, snap.height as f64 / snap.doc.1 as f64);
    let pts: Vec<Point> = points
        .iter()
        .map(|&(x, y, positive)| Point { x: (x * kx) as f32, y: (y * ky) as f32, positive })
        .collect();
    let Loaded { sam, embedding, .. } = &mut *l;
    let m = sam.as_mut().unwrap().segment(&embedding.as_ref().unwrap().1, &pts)?;
    Ok(to_cpu(&m, snap.doc))
}

/// A model mask scaled (bilinear) to a document-size mask.
pub fn to_cpu(m: &Mask, (dw, dh): (u32, u32)) -> CpuMask {
    let ts = TILE_SIZE as usize;
    let (gw, gh) = (dw.div_ceil(TILE_SIZE), dh.div_ceil(TILE_SIZE));
    let (sx, sy) = (m.width as f64 / dw as f64, m.height as f64 / dh as f64);
    let at = |x: usize, y: usize| m.data[y.min(m.height - 1) * m.width + x.min(m.width - 1)] as f32;
    let tiles: HashMap<TileCoord, MaskData> = (0..gw * gh)
        .into_par_iter()
        .map(|i| {
            let c = TileCoord::new((i % gw) as i32, (i / gw) as i32);
            let (ox, oy) = c.origin();
            let mut px = vec![0u8; ts * ts];
            for y in 0..ts {
                let gy = oy + y as i64;
                if gy >= dh as i64 {
                    break;
                }
                let fy = ((gy as f64 + 0.5) * sy - 0.5).max(0.0);
                let (y0, ay) = (fy as usize, (fy.fract()) as f32);
                for x in 0..ts {
                    let gx = ox + x as i64;
                    if gx >= dw as i64 {
                        break;
                    }
                    let fx = ((gx as f64 + 0.5) * sx - 0.5).max(0.0);
                    let (x0, ax) = (fx as usize, (fx.fract()) as f32);
                    let top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * ax;
                    let bot = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * ax;
                    px[y * ts + x] = (top + (bot - top) * ay + 0.5) as u8;
                }
            }
            (c, MaskData::Pixels(px).normalized())
        })
        .collect();
    let mut cpu = CpuMask { tiles, default: 0 };
    cpu.compact();
    cpu
}
