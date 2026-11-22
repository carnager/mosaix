//! The system file chooser (XDG desktop portal, e.g. the GTK/GNOME dialog).
//! Runs on its own thread; the answer comes back as a [`Picked`]. Without a
//! portal the caller falls back to the built-in dialog.

use std::path::{Path, PathBuf};

use ashpd::WindowIdentifier;
use ashpd::desktop::file_chooser::{Choice, FileFilter, SelectedFiles};
use mosaix_core::tr;
use mosaix_io::export::{ColorSpace, ExportOptions};
use wayland_client::protocol::wl_surface::WlSurface;

use crate::ui::files::FileMode;

/// What the user chose.
pub enum Picked {
    File(PathBuf, ExportOptions),
    Cancelled,
    /// No portal (or it failed): use the built-in dialog.
    Unavailable(String),
}

/// What to show: the mode, start folder, suggested name and export defaults.
pub struct Request {
    pub mode: FileMode,
    pub folder: PathBuf,
    pub name: String,
    pub export: ExportOptions,
}

const QUALITIES: [u8; 8] = [100, 95, 90, 85, 80, 70, 60, 50];

fn filters(mode: FileMode) -> Vec<FileFilter> {
    let one = |label: &str, exts: &[&str]| {
        exts.iter().fold(FileFilter::new(label), |f, e| {
            f.glob(&format!("*.{e}")).glob(&format!("*.{}", e.to_uppercase()))
        })
    };
    match mode {
        FileMode::Open => {
            let mut all: Vec<&str> = FileMode::Open.extensions().to_vec();
            all.extend(mosaix_io::raw::EXTENSIONS);
            vec![
                one(tr("All supported images"), &all),
                one("Mosaix", &["mosaix", "imagine"]),
                one("Photoshop", &["psd", "psb"]),
                one(tr("Camera RAW"), mosaix_io::raw::EXTENSIONS),
            ]
        }
        FileMode::SaveAs => vec![one("Mosaix", &["mosaix"])],
        FileMode::ExportAs => vec![
            one("PNG", &["png"]),
            one("JPEG", &["jpg", "jpeg"]),
            one("WebP", &["webp"]),
            one("AVIF", &["avif"]),
            one("JPEG XL", &["jxl"]),
            one("TIFF", &["tif", "tiff"]),
            one("Photoshop", &["psd", "psb"]),
        ],
        FileMode::ImportBrushes => vec![one(tr("Brushes and images"), FileMode::ImportBrushes.extensions())],
        FileMode::ImportPatterns => vec![one(tr("Patterns and images"), FileMode::ImportPatterns.extensions())],
        FileMode::PlaceEmbedded | FileMode::ReplaceContents | FileMode::OpenAsLayer => vec![one(tr("Images"), FileMode::PlaceEmbedded.extensions())],
    }
}

/// Export options as choices in the dialog (combo boxes / check box).
fn choices(o: &ExportOptions) -> Vec<Choice> {
    let space = ColorSpace::ALL
        .iter()
        .enumerate()
        .fold(Choice::new("space", tr("Color space"), &ColorSpace::ALL.iter().position(|s| *s == o.space).unwrap_or(0).to_string()), |c, (i, s)| {
            c.insert(&i.to_string(), s.name())
        });
    let q = QUALITIES.iter().min_by_key(|q| (**q as i32 - o.quality as i32).abs()).copied().unwrap_or(90);
    let quality = QUALITIES
        .iter()
        .fold(Choice::new("quality", tr("JPEG quality"), &q.to_string()), |c, v| c.insert(&v.to_string(), &format!("{v} %")));
    vec![space, Choice::boolean("bits", tr("16-bit (PNG, TIFF, PSD)"), o.sixteen_bit), quality]
}

fn read_choices(mut o: ExportOptions, picked: &[(String, String)]) -> ExportOptions {
    for (k, v) in picked {
        match k.as_str() {
            "space" => {
                if let Some(s) = v.parse::<usize>().ok().and_then(|i| ColorSpace::ALL.get(i)) {
                    o.space = *s;
                }
            }
            "bits" => o.sixteen_bit = v == "true",
            "quality" => {
                if let Ok(q) = v.parse() {
                    o.quality = q;
                }
            }
            _ => {}
        }
    }
    o
}

/// `file:///a%20b` → `/a b`.
fn uri_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    let b = rest.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).ok()?, 16)
        {
            out.push(v);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    use std::os::unix::ffi::OsStringExt;
    Some(std::ffi::OsString::from_vec(out).into())
}

async fn ask(req: &Request, surface: Option<WlSurface>) -> ashpd::Result<Option<(PathBuf, ExportOptions)>> {
    // Lets the compositor put the dialog over our window (xdg-foreign).
    let parent = match &surface {
        Some(s) => WindowIdentifier::from_wayland(s).await,
        None => None,
    };
    let title = req.mode.title();
    let folder = Some(req.folder.as_path()).filter(|p| p.is_dir());
    let response = match req.mode {
        FileMode::Open
        | FileMode::ImportBrushes
        | FileMode::ImportPatterns
        | FileMode::PlaceEmbedded
        | FileMode::ReplaceContents
        | FileMode::OpenAsLayer => {
            let mut r = SelectedFiles::open_file()
                .identifier(parent)
                .title(title)
                .modal(true)
                .multiple(false)
                .filters(filters(req.mode));
            if let Some(f) = folder {
                r = r.current_folder(f)?;
            }
            r.send().await?.response()
        }
        FileMode::SaveAs | FileMode::ExportAs => {
            let mut r = SelectedFiles::save_file()
                .identifier(parent)
                .title(title)
                .modal(true)
                .current_name(req.name.as_str())
                .filters(filters(req.mode));
            if req.mode == FileMode::ExportAs {
                let ext = Path::new(&req.name).extension().and_then(|e| e.to_str()).unwrap_or("png").to_ascii_lowercase();
                let fs = filters(req.mode);
                let cur = fs.into_iter().find(|f| f.pattern_filters().iter().any(|p| p.trim_start_matches("*.") == ext));
                r = r.current_filter(cur).choices(choices(&req.export));
            }
            if let Some(f) = folder {
                r = r.current_folder(f)?;
            }
            r.send().await?.response()
        }
    };
    match response {
        Ok(files) => {
            let Some(path) = files.uris().first().and_then(|u| uri_path(u.as_str())) else {
                return Ok(None);
            };
            Ok(Some((path, read_choices(req.export, files.choices()))))
        }
        Err(ashpd::Error::Response(ashpd::desktop::ResponseError::Cancelled)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Shows the system dialog; blocks the calling thread until it is answered.
pub fn pick(req: &Request, surface: Option<WlSurface>) -> Picked {
    match pollster::block_on(ask(req, surface)) {
        Ok(Some((path, o))) => Picked::File(path, o),
        Ok(None) => Picked::Cancelled,
        Err(e) => Picked::Unavailable(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_uris() {
        assert_eq!(uri_path("file:///home/a%20b/%C3%BC.png"), Some(PathBuf::from("/home/a b/ü.png")));
        assert_eq!(uri_path("file:///x/100%"), Some(PathBuf::from("/x/100%")));
        assert_eq!(uri_path("https://x"), None);
    }

    #[test]
    fn export_choices_round_trip() {
        let o = ExportOptions { space: ColorSpace::DisplayP3, quality: 85, sixteen_bit: false };
        let picked = [("space".to_string(), "2".to_string()), ("bits".into(), "true".into()), ("quality".into(), "70".into())];
        let r = read_choices(o, &picked);
        assert_eq!((r.space, r.sixteen_bit, r.quality), (ColorSpace::ALL[2], true, 70));
    }
}
