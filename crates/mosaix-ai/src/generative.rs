//! Generative fill: a hole in an image is painted from a text prompt by a
//! provider – a hosted model (OpenAI's image edits) or any program the user
//! names (a local Stable Diffusion / ComfyUI script, …).
//!
//! The caller decides whether sending pixels to a hosted service is allowed;
//! nothing here asks.

use std::io::Read;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use base64::Engine as _;
use image::{GrayImage, Luma, Rgba, RgbaImage, imageops::FilterType};

/// Who paints.
#[derive(Debug, Clone, PartialEq)]
pub enum Provider {
    /// OpenAI image edits (`/v1/images/edits`).
    OpenAi { model: String, key: String },
    /// `program image.png mask.png prompt out.png` (mask: white = fill);
    /// the prompt is also in `MOSAIX_PROMPT`.
    Command { program: String },
}

impl Provider {
    /// Whether the pixels leave this computer.
    pub fn is_remote(&self) -> bool {
        matches!(self, Provider::OpenAi { .. })
    }
}

/// Fills the `hole` of a `w × h` area (straight sRGB 0…1, row-major) from
/// `prompt`. Only hole pixels are replaced.
pub fn fill(p: &Provider, w: usize, h: usize, rgb: &mut [[f32; 3]], hole: &[bool], prompt: &str) -> Result<()> {
    ensure!(w > 0 && h > 0 && rgb.len() == w * h && hole.len() == w * h, "bad area");
    ensure!(!prompt.trim().is_empty(), "the prompt is empty");
    let img = RgbaImage::from_fn(w as u32, h as u32, |x, y| {
        let c = rgb[y as usize * w + x as usize].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        Rgba([c[0], c[1], c[2], 255])
    });
    let out = match p {
        Provider::OpenAi { model, key } => openai(model, key, &img, hole, prompt)?,
        Provider::Command { program } => command(program, &img, hole, prompt)?,
    };
    let out = if out.dimensions() == (w as u32, h as u32) {
        out
    } else {
        image::imageops::resize(&out, w as u32, h as u32, FilterType::CatmullRom)
    };
    for (i, c) in rgb.iter_mut().enumerate() {
        if hole[i] {
            let p = out.get_pixel((i % w) as u32, (i / w) as u32);
            *c = [p[0], p[1], p[2]].map(|v| v as f32 / 255.0);
        }
    }
    Ok(())
}

fn png(img: &image::DynamicImage) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut v), image::ImageFormat::Png)?;
    Ok(v)
}

/// The sizes the image models take, by aspect ratio.
fn model_size(w: usize, h: usize) -> (u32, u32) {
    let a = w as f64 / h as f64;
    if a > 1.25 {
        (1536, 1024)
    } else if a < 0.8 {
        (1024, 1536)
    } else {
        (1024, 1024)
    }
}

fn openai(model: &str, key: &str, img: &RgbaImage, hole: &[bool], prompt: &str) -> Result<RgbaImage> {
    ensure!(!key.trim().is_empty(), "no OpenAI API key (set OPENAI_API_KEY)");
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (mw, mh) = model_size(w, h);
    let scaled = image::imageops::resize(img, mw, mh, FilterType::CatmullRom);
    // The mask: transparent where the model paints.
    let mut mask = RgbaImage::from_pixel(w as u32, h as u32, Rgba([0, 0, 0, 255]));
    for (i, &hl) in hole.iter().enumerate() {
        if hl {
            mask.put_pixel((i % w) as u32, (i / w) as u32, Rgba([0, 0, 0, 0]));
        }
    }
    let mask = image::imageops::resize(&mask, mw, mh, FilterType::Nearest);
    let boundary = "----mosaix-generative-fill";
    let mut body = Vec::new();
    let mut field = |name: &str, value: &[u8], file: Option<&str>| {
        body.extend(format!("--{boundary}\r\n").as_bytes());
        match file {
            Some(f) => body.extend(
                format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\nContent-Type: image/png\r\n\r\n").as_bytes(),
            ),
            None => body.extend(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes()),
        }
        body.extend(value);
        body.extend(b"\r\n");
    };
    field("model", model.as_bytes(), None);
    field("prompt", prompt.as_bytes(), None);
    field("size", format!("{mw}x{mh}").as_bytes(), None);
    field("image", &png(&image::DynamicImage::ImageRgba8(scaled))?, Some("image.png"));
    field("mask", &png(&image::DynamicImage::ImageRgba8(mask))?, Some("mask.png"));
    body.extend(format!("--{boundary}--\r\n").as_bytes());
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(300)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut resp = agent
        .post("https://api.openai.com/v1/images/edits")
        .header("Authorization", &format!("Bearer {}", key.trim()))
        .header("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
        .send(&body[..])
        .context("OpenAI")?;
    let status = resp.status();
    let mut text = String::new();
    resp.body_mut().as_reader().read_to_string(&mut text)?;
    let v: serde_json::Value = serde_json::from_str(&text).with_context(|| format!("OpenAI answered {status}"))?;
    if !status.is_success() {
        let msg = v["error"]["message"].as_str().unwrap_or(&text);
        bail!("OpenAI: {msg}");
    }
    let b64 = v["data"][0]["b64_json"].as_str().context("OpenAI sent no image")?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).context("OpenAI image")?;
    Ok(image::load_from_memory(&bytes)?.to_rgba8())
}

fn command(program: &str, img: &RgbaImage, hole: &[bool], prompt: &str) -> Result<RgbaImage> {
    ensure!(!program.trim().is_empty(), "no generative fill program set");
    let dir = std::env::temp_dir().join(format!("mosaix-fill-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let (ip, mp, op) = (dir.join("image.png"), dir.join("mask.png"), dir.join("out.png"));
    let _ = std::fs::remove_file(&op);
    img.save(&ip)?;
    let w = img.width() as usize;
    GrayImage::from_fn(img.width(), img.height(), |x, y| Luma([if hole[y as usize * w + x as usize] { 255 } else { 0 }])).save(&mp)?;
    let out = std::process::Command::new(program.trim())
        .arg(&ip)
        .arg(&mp)
        .arg(prompt)
        .arg(&op)
        .env("MOSAIX_PROMPT", prompt)
        .output()
        .with_context(|| format!("cannot run {program}"))?;
    if !out.status.success() {
        bail!("{program} failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let res = load(&op).with_context(|| format!("{program} wrote no {}", op.display()));
    let _ = std::fs::remove_dir_all(&dir);
    res
}

fn load(p: &Path) -> Result<RgbaImage> {
    Ok(image::open(p)?.to_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_follow_the_aspect() {
        assert_eq!(model_size(800, 400), (1536, 1024));
        assert_eq!(model_size(300, 600), (1024, 1536));
        assert_eq!(model_size(500, 450), (1024, 1024));
    }

    #[test]
    fn a_program_fills_only_the_hole() {
        // A tiny "model": paints the whole picture red.
        let dir = std::env::temp_dir().join(format!("mosaix-fill-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("red.sh");
        std::fs::write(&script, "#!/bin/sh\nconvert \"$1\" -fill red -colorize 100 \"$4\" 2>/dev/null || python3 -c \"from PIL import Image; Image.new('RGB', Image.open('$1').size, 'red').save('$4')\"\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let (w, h) = (8, 4);
        let mut rgb = vec![[0.0, 0.0, 1.0]; w * h];
        let hole: Vec<bool> = (0..w * h).map(|i| i % w < 4).collect();
        let p = Provider::Command { program: script.display().to_string() };
        fill(&p, w, h, &mut rgb, &hole, "something red").unwrap();
        assert!((rgb[0][0] - 1.0).abs() < 0.02 && rgb[0][2] < 0.02, "hole painted: {:?}", rgb[0]);
        assert_eq!(rgb[5], [0.0, 0.0, 1.0], "the rest stays");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
