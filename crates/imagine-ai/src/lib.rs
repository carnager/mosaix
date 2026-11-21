//! Local AI models, run with ONNX Runtime from the system (loaded at
//! runtime, so imagine works without it; the features then report why
//! they are unavailable). Models are downloaded once on request into
//! `$XDG_DATA_HOME/imagine/models`.
//!
//! - [`Sam`]: SAM 2.1 (Meta, Apache-2.0) – select an object by clicking.
//! - [`Subject`]: BiRefNet lite (MIT) – the main subject of a picture.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array3, Array4};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

pub mod models;

pub use models::{Model, SAM, SUBJECT};

/// Where ONNX Runtime is looked for (`ORT_DYLIB_PATH` first).
fn dylib_candidates() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from).into_iter().collect();
    for d in ["/usr/lib", "/usr/lib64", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"] {
        v.push(Path::new(d).join("libonnxruntime.so"));
    }
    v
}

fn env() -> Result<&'static ort::environment::Environment> {
    static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        let Some(lib) = dylib_candidates().into_iter().find(|p| p.exists()) else {
            return Err("ONNX Runtime not found (install onnxruntime, e.g. the Arch package onnxruntime-cpu, or set ORT_DYLIB_PATH)".into());
        };
        ort::init_from(&lib)
            .map_err(|e| format!("{}: {e}", lib.display()))?
            .with_name("imagine")
            .build()
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| anyhow!("{e}"))
}

/// Whether models can run here (`Err` says why not).
pub fn available() -> Result<()> {
    env().map(|_| ())
}

fn session(path: &Path) -> Result<Session> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    Session::builder(env()?)?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .with_context(|| path.display().to_string())
}

/// An sRGB image, 8-bit RGB.
pub struct Image<'a> {
    pub width: usize,
    pub height: usize,
    pub rgb: &'a [u8],
}

/// A mask at the image's size: 0…255 per pixel.
pub struct Mask {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Bilinear resize to `n × n`, ImageNet normalisation, NCHW.
fn to_tensor(img: &Image, n: usize) -> Array4<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let (w, h) = (img.width, img.height);
    let planes: Vec<Vec<f32>> = (0..3)
        .into_par_iter()
        .map(|c| {
            let mut out = vec![0.0f32; n * n];
            for y in 0..n {
                let fy = ((y as f32 + 0.5) * h as f32 / n as f32 - 0.5).clamp(0.0, (h - 1) as f32);
                let (y0, ay) = (fy as usize, fy.fract());
                let y1 = (y0 + 1).min(h - 1);
                for x in 0..n {
                    let fx = ((x as f32 + 0.5) * w as f32 / n as f32 - 0.5).clamp(0.0, (w - 1) as f32);
                    let (x0, ax) = (fx as usize, fx.fract());
                    let x1 = (x0 + 1).min(w - 1);
                    let p = |x: usize, y: usize| img.rgb[(y * w + x) * 3 + c] as f32 / 255.0;
                    let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * ax;
                    let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * ax;
                    out[y * n + x] = (top + (bot - top) * ay - MEAN[c]) / STD[c];
                }
            }
            out
        })
        .collect();
    let data: Vec<f32> = planes.concat();
    Array4::from_shape_vec((1, 3, n, n), data).expect("shape")
}

/// A probability map (`n × m`, values via `f`) scaled bilinearly to `w × h`.
fn resize_map(src: &[f32], sw: usize, sh: usize, w: usize, h: usize, f: impl Fn(f32) -> f32 + Sync) -> Mask {
    let mut data = vec![0u8; w * h];
    data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) * sh as f32 / h as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
        let (y0, ay) = (fy as usize, fy.fract());
        let y1 = (y0 + 1).min(sh - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) * sw as f32 / w as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let (x0, ax) = (fx as usize, fx.fract());
            let x1 = (x0 + 1).min(sw - 1);
            let p = |x: usize, y: usize| src[y * sw + x];
            let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * ax;
            let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * ax;
            *o = (f(top + (bot - top) * ay).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    });
    Mask { width: w, height: h, data }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---- SAM 2.1 -----------------------------------------------------------------

/// SAM 2.1: encode an image once, then masks for clicks in milliseconds.
pub struct Sam {
    encoder: Session,
    decoder: Session,
}

/// The encoded image (reused for every click on it).
pub struct Embedding {
    levels: [Array4<f32>; 3],
    width: usize,
    height: usize,
}

/// A click: position in image pixels; `true` = belongs to the object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub positive: bool,
}

const SAM_SIZE: usize = 1024;

impl Sam {
    pub fn load() -> Result<Self> {
        let dir = models::path(&SAM).ok_or_else(|| anyhow!("model {} is not installed", SAM.id))?;
        Ok(Sam { encoder: session(&dir.join("vision_encoder.onnx"))?, decoder: session(&dir.join("prompt_encoder_mask_decoder.onnx"))? })
    }

    pub fn encode(&mut self, img: &Image) -> Result<Embedding> {
        let input = Tensor::from_array(to_tensor(img, SAM_SIZE))?;
        let out = self.encoder.run(ort::inputs!["pixel_values" => input])?;
        let level = |name: &str| -> Result<Array4<f32>> {
            let (shape, data) = out[name].try_extract_tensor::<f32>()?;
            let d: Vec<usize> = shape.iter().map(|v| *v as usize).collect();
            Ok(Array4::from_shape_vec((d[0], d[1], d[2], d[3]), data.to_vec())?)
        };
        Ok(Embedding {
            levels: [level("image_embeddings.0")?, level("image_embeddings.1")?, level("image_embeddings.2")?],
            width: img.width,
            height: img.height,
        })
    }

    /// The mask for some clicks (the best of SAM's three proposals for one
    /// click; the first proposal for several, as SAM recommends).
    pub fn segment(&mut self, e: &Embedding, points: &[Point]) -> Result<Mask> {
        if points.is_empty() {
            bail!("no points");
        }
        let (sx, sy) = (SAM_SIZE as f32 / e.width as f32, SAM_SIZE as f32 / e.height as f32);
        let n = points.len();
        let pts = Array4::from_shape_vec((1, 1, n, 2), points.iter().flat_map(|p| [p.x * sx, p.y * sy]).collect())?;
        let labels = Array3::from_shape_vec((1, 1, n), points.iter().map(|p| p.positive as i64).collect())?;
        let boxes = Array3::<f32>::zeros((1, 0, 4));
        let out = self.decoder.run(ort::inputs![
            "input_points" => Tensor::from_array(pts)?,
            "input_labels" => Tensor::from_array(labels)?,
            "input_boxes" => Tensor::from_array(boxes)?,
            "image_embeddings.0" => Tensor::from_array(e.levels[0].clone())?,
            "image_embeddings.1" => Tensor::from_array(e.levels[1].clone())?,
            "image_embeddings.2" => Tensor::from_array(e.levels[2].clone())?,
        ])?;
        let (_, iou) = out["iou_scores"].try_extract_tensor::<f32>()?;
        let (shape, masks) = out["pred_masks"].try_extract_tensor::<f32>()?;
        let (k, mh, mw) = (shape[2] as usize, shape[3] as usize, shape[4] as usize);
        let best = if n == 1 {
            (0..k).max_by(|a, b| iou[*a].total_cmp(&iou[*b])).unwrap_or(0)
        } else {
            0
        };
        let m = &masks[best * mh * mw..(best + 1) * mh * mw];
        // Logits, inside > 0 (as SAM thresholds them); a narrow ramp
        // anti-aliases the edge.
        Ok(resize_map(m, mw, mh, e.width, e.height, |v| v * 2.0 + 0.5))
    }
}

// ---- subject (BiRefNet) ------------------------------------------------------

/// The main subject of a picture (BiRefNet lite).
pub struct Subject {
    s: Session,
}

impl Subject {
    pub fn load() -> Result<Self> {
        let dir = models::path(&SUBJECT).ok_or_else(|| anyhow!("model {} is not installed", SUBJECT.id))?;
        Ok(Subject { s: session(&dir.join("model.onnx"))? })
    }

    pub fn mask(&mut self, img: &Image) -> Result<Mask> {
        let input = Tensor::from_array(to_tensor(img, 1024))?;
        let out = self.s.run(ort::inputs!["input_image" => input])?;
        let (shape, data) = out["output_image"].try_extract_tensor::<f32>()?;
        let (h, w) = (shape[2] as usize, shape[3] as usize);
        // The export ends in logits.
        Ok(resize_map(data, w, h, img.width, img.height, sigmoid))
    }
}
