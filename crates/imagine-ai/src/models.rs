//! The models imagine uses, where they come from and where they live.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

pub struct File {
    pub name: &'static str,
    pub url: &'static str,
    pub size: u64,
}

pub struct Model {
    pub id: &'static str,
    pub name: &'static str,
    pub license: &'static str,
    pub files: &'static [File],
}

impl Model {
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

pub const SAM: Model = Model {
    id: "sam2.1-small",
    name: "SAM 2.1 small (Meta)",
    license: "Apache-2.0",
    files: &[
        File { name: "vision_encoder.onnx", url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/vision_encoder.onnx", size: 467_440 },
        File { name: "vision_encoder.onnx_data", url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/vision_encoder.onnx_data", size: 162_476_288 },
        File {
            name: "prompt_encoder_mask_decoder.onnx",
            url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
            size: 213_114,
        },
        File {
            name: "prompt_encoder_mask_decoder.onnx_data",
            url: "https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data",
            size: 20_958_208,
        },
    ],
};

pub const SUBJECT: Model = Model {
    id: "birefnet-lite",
    name: "BiRefNet lite",
    license: "MIT",
    files: &[File {
        name: "model.onnx",
        url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx",
        size: 224_005_088,
    }],
};

/// Folder of all models: `$IMAGINE_MODELS`, else `$XDG_DATA_HOME/imagine/models`.
pub fn root() -> PathBuf {
    if let Some(d) = std::env::var_os("IMAGINE_MODELS") {
        return d.into();
    }
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| ".".into())
        .join("imagine/models")
}

/// The model's folder, if all its files are there.
pub fn path(m: &Model) -> Option<PathBuf> {
    let dir = root().join(m.id);
    m.files.iter().all(|f| std::fs::metadata(dir.join(f.name)).is_ok_and(|md| md.len() == f.size)).then_some(dir)
}

/// Downloads a model (files already complete are kept). `progress(done, total)`.
pub fn download(m: &Model, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    let dir = root().join(m.id);
    std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
    let total = m.size();
    let mut done = 0u64;
    for f in m.files {
        let target = dir.join(f.name);
        if std::fs::metadata(&target).is_ok_and(|md| md.len() == f.size) {
            done += f.size;
            progress(done, total);
            continue;
        }
        let part = dir.join(format!("{}.part", f.name));
        let resp = ureq::get(f.url).call().with_context(|| f.url.to_string())?;
        let mut body = resp.into_body().into_reader();
        let mut out = std::fs::File::create(&part).with_context(|| part.display().to_string())?;
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        loop {
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            got += n as u64;
            progress(done + got, total);
        }
        out.sync_all()?;
        if got != f.size {
            let _ = std::fs::remove_file(&part);
            bail!("{}: got {got} bytes, expected {}", f.url, f.size);
        }
        std::fs::rename(&part, &target)?;
        done += f.size;
    }
    Ok(dir)
}
