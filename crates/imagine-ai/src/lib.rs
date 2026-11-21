//! Local AI models, run with ONNX Runtime from the system (loaded at
//! runtime, so imagine works without it; the features then report why
//! they are unavailable). Models are downloaded once on request into
//! `$XDG_DATA_HOME/imagine/models`.
//!
//! - [`Sam`]: SAM 2.1 (Meta, Apache-2.0) – select an object by clicking.
//! - [`Subject`]: BiRefNet lite (MIT) – the main subject of a picture.
//! - [`Inpaint`]: LaMa (Apache-2.0) – fills a hole plausibly (Remove).
//! - [`Upscale`]: Real-ESRGAN (BSD-3-Clause) – enlarges with detail.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array3, Array4};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

pub mod models;

pub use models::{INPAINT, Model, SAM, SUBJECT, UPSCALE};

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

    /// The mask for some clicks (the best rated of SAM's three proposals).
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
        // This export only has SAM's three proposals (not its single-mask
        // output for several clicks): the best-rated one, also for several.
        let best = (0..k).max_by(|a, b| iou[*a].total_cmp(&iou[*b])).unwrap_or(0);
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

// ---- inpainting (LaMa) -------------------------------------------------------

/// Fills holes plausibly from their surroundings (LaMa, works at 512²).
pub struct Inpaint {
    s: Session,
}

const LAMA_SIZE: usize = 512;

/// A straight sRGB pixel, 0…1.
pub type Rgb = [f32; 3];

/// Box-filtered resize (down) / bilinear (up) of an RGB image.
fn resize_rgb(src: &[Rgb], sw: usize, sh: usize, w: usize, h: usize) -> Vec<Rgb> {
    let mut out = vec![[0.0; 3]; w * h];
    let (kx, ky) = (sw as f32 / w as f32, sh as f32 / h as f32);
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            if kx > 1.0 || ky > 1.0 {
                // Average of the covered source pixels.
                let (x0, x1) = ((x as f32 * kx) as usize, (((x + 1) as f32 * kx).ceil() as usize).clamp(1, sw));
                let (y0, y1) = ((y as f32 * ky) as usize, (((y + 1) as f32 * ky).ceil() as usize).clamp(1, sh));
                let (x0, y0) = (x0.min(x1 - 1), y0.min(y1 - 1));
                let mut s = [0.0f32; 3];
                for yy in y0..y1 {
                    for p in &src[yy * sw + x0..yy * sw + x1] {
                        for c in 0..3 {
                            s[c] += p[c];
                        }
                    }
                }
                let n = ((x1 - x0) * (y1 - y0)) as f32;
                *o = s.map(|v| v / n);
            } else {
                let fx = ((x as f32 + 0.5) * kx - 0.5).clamp(0.0, (sw - 1) as f32);
                let fy = ((y as f32 + 0.5) * ky - 0.5).clamp(0.0, (sh - 1) as f32);
                let (x0, y0, ax, ay) = (fx as usize, fy as usize, fx.fract(), fy.fract());
                let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
                let p = |x: usize, y: usize| src[y * sw + x];
                *o = [0, 1, 2].map(|c| {
                    let top = p(x0, y0)[c] + (p(x1, y0)[c] - p(x0, y0)[c]) * ax;
                    let bot = p(x0, y1)[c] + (p(x1, y1)[c] - p(x0, y1)[c]) * ax;
                    top + (bot - top) * ay
                });
            }
        }
    });
    out
}

impl Inpaint {
    pub fn load() -> Result<Self> {
        let dir = models::path(&INPAINT).ok_or_else(|| anyhow!("model {} is not installed", INPAINT.id))?;
        Ok(Inpaint { s: session(&dir.join("lama_fp32.onnx"))? })
    }

    /// Replaces the hole pixels of `img` (`w × h`, straight sRGB 0…1) with
    /// generated ones. Larger areas are scaled down to 512 for the model,
    /// and only the hole takes the (scaled up) result.
    pub fn fill(&mut self, w: usize, h: usize, img: &mut [Rgb], hole: &[bool]) -> Result<()> {
        let n = LAMA_SIZE;
        let k = (n as f32 / w.max(h) as f32).min(1.0);
        let (sw, sh) = (((w as f32 * k).round() as usize).clamp(1, n), ((h as f32 * k).round() as usize).clamp(1, n));
        let small = resize_rgb(img, w, h, sw, sh);
        // The hole at model size: any hole pixel in the covered area, grown
        // by a few pixels (LaMa wants the object fully covered).
        let mut m = vec![false; sw * sh];
        for y in 0..h {
            for x in 0..w {
                if hole[y * w + x] {
                    let (sx, sy) = (((x as f32 + 0.5) * k) as usize, ((y as f32 + 0.5) * k) as usize);
                    m[sy.min(sh - 1) * sw + sx.min(sw - 1)] = true;
                }
            }
        }
        const GROW: i64 = 3;
        let mut grown = vec![0.0f32; n * n];
        for y in 0..sh as i64 {
            for x in 0..sw as i64 {
                let hit = (-GROW..=GROW).any(|dy| {
                    (-GROW..=GROW).any(|dx| {
                        let (xx, yy) = (x + dx, y + dy);
                        xx >= 0 && yy >= 0 && xx < sw as i64 && yy < sh as i64 && m[(yy * sw as i64 + xx) as usize]
                    })
                });
                grown[y as usize * n + x as usize] = hit as u8 as f32;
            }
        }
        // NCHW, the area beyond the image repeats its edge.
        let mut t = vec![0.0f32; 3 * n * n];
        for y in 0..n {
            for x in 0..n {
                let p = small[y.min(sh - 1) * sw + x.min(sw - 1)];
                for c in 0..3 {
                    t[c * n * n + y * n + x] = p[c];
                }
            }
        }
        let image = Tensor::from_array(Array4::from_shape_vec((1, 3, n, n), t)?)?;
        let mask = Tensor::from_array(Array4::from_shape_vec((1, 1, n, n), grown)?)?;
        let out = self.s.run(ort::inputs!["image" => image, "mask" => mask])?;
        let (_, data) = out["output"].try_extract_tensor::<f32>()?;
        // The export returns 0…255.
        let mut res = vec![[0.0f32; 3]; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                res[y * sw + x] = [0, 1, 2].map(|c| (data[c * n * n + y * n + x] / 255.0).clamp(0.0, 1.0));
            }
        }
        let big = if (sw, sh) == (w, h) { res } else { resize_rgb(&res, sw, sh, w, h) };
        for (i, p) in img.iter_mut().enumerate() {
            if hole[i] {
                *p = big[i];
            }
        }
        Ok(())
    }
}

// ---- upscaling (Real-ESRGAN) -------------------------------------------------

/// Enlarges pictures with detail (Real-ESRGAN, BSD-3-Clause). The model has a
/// fixed tile size; the image is processed in overlapping tiles.
pub struct Upscale {
    s: Session,
    tile: usize,
    scale: usize,
    input: String,
    output: String,
}

/// Context around each tile (input pixels) so tile seams do not show.
const UPSCALE_OVERLAP: usize = 12;

impl Upscale {
    pub fn load() -> Result<Self> {
        let dir = models::path(&UPSCALE).ok_or_else(|| anyhow!("model {} is not installed", UPSCALE.id))?;
        Self::load_file(&dir.join("model.onnx"))
    }

    pub fn load_file(path: &Path) -> Result<Self> {
        let s = session(path)?;
        let input = s.inputs().first().ok_or_else(|| anyhow!("model without input"))?;
        let output = s.outputs().first().ok_or_else(|| anyhow!("model without output"))?;
        let dims = |t: &ort::value::ValueType| -> Vec<i64> {
            match t {
                ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
                _ => Vec::new(),
            }
        };
        let (i, o) = (dims(input.dtype()), dims(output.dtype()));
        let tile = i.get(2).copied().filter(|v| *v > 0).unwrap_or(128) as usize;
        let scale = o.get(2).copied().filter(|v| *v > 0).map_or(4, |v| v as usize / tile);
        Ok(Upscale { tile, scale, input: input.name().to_string(), output: output.name().to_string(), s })
    }

    pub fn scale(&self) -> usize {
        self.scale
    }

    /// The image (`w × h`, straight sRGB 0…1) enlarged by [`Upscale::scale`];
    /// with `alpha`, fully transparent parts are skipped (left black).
    /// `progress(done, total)` per tile.
    pub fn run(&mut self, w: usize, h: usize, img: &[Rgb], alpha: Option<&[f32]>, progress: &mut dyn FnMut(usize, usize)) -> Result<Vec<Rgb>> {
        let (t, k) = (self.tile, self.scale);
        let step = t - 2 * UPSCALE_OVERLAP;
        let (nx, ny) = (w.div_ceil(step), h.div_ceil(step));
        let (ow, oh) = (w * k, h * k);
        let mut out = vec![[0.0f32; 3]; ow * oh];
        let mut buf = vec![0.0f32; 3 * t * t];
        for ty in 0..ny {
            for tx in 0..nx {
                // The tile's source window: the step area plus the overlap,
                // clamped at the edges (repeating the border).
                let (sx, sy) = ((tx * step) as i64 - UPSCALE_OVERLAP as i64, (ty * step) as i64 - UPSCALE_OVERLAP as i64);
                if let Some(a) = alpha {
                    let (x0, y0) = (sx.max(0) as usize, sy.max(0) as usize);
                    let (x1, y1) = (((sx + t as i64) as usize).min(w), ((sy + t as i64) as usize).min(h));
                    if (y0..y1).all(|y| a[y * w + x0..y * w + x1].iter().all(|&v| v <= 0.0)) {
                        progress(ty * nx + tx + 1, nx * ny);
                        continue;
                    }
                }
                for y in 0..t {
                    let yy = (sy + y as i64).clamp(0, h as i64 - 1) as usize;
                    for x in 0..t {
                        let xx = (sx + x as i64).clamp(0, w as i64 - 1) as usize;
                        let p = img[yy * w + xx];
                        for c in 0..3 {
                            buf[c * t * t + y * t + x] = p[c];
                        }
                    }
                }
                let input = Tensor::from_array(Array4::from_shape_vec((1, 3, t, t), buf.clone())?)?;
                let res = self.s.run(ort::inputs![self.input.as_str() => input])?;
                let (_, data) = res[self.output.as_str()].try_extract_tensor::<f32>()?;
                let tk = t * k;
                // Keep only the step area of the tile.
                let (x0, y0) = (tx * step, ty * step);
                for y in 0..(step.min(h - y0)) * k {
                    for x in 0..(step.min(w - x0)) * k {
                        let (ix, iy) = (x + UPSCALE_OVERLAP * k, y + UPSCALE_OVERLAP * k);
                        out[(y0 * k + y) * ow + x0 * k + x] =
                            [0, 1, 2].map(|c| data[c * tk * tk + iy * tk + ix].clamp(0.0, 1.0));
                    }
                }
                progress(ty * nx + tx + 1, nx * ny);
            }
        }
        Ok(out)
    }
}
