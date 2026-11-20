//! Custom brush tips: a folder of grey PNGs (coverage, white = paint) in
//! `$XDG_DATA_HOME/imagine/brushes`, filled by importing Photoshop brushes
//! (.abr) or any image. Loaded into the engine as tips 6, 7, …

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use imagine_gpu::{FIRST_CUSTOM_TIP, MAX_CUSTOM_TIPS, TIP_SIZE};

use super::App;

pub fn brush_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("imagine/brushes")
}

/// The library: names and TIP_SIZE² coverage, sorted by file name.
pub fn load_library(dir: &Path) -> (Vec<String>, Vec<Vec<u8>>) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"))).collect())
        .unwrap_or_default();
    // By name ("Tip" before "Tip 2", which a plain path sort gets wrong).
    files.sort_by_key(|p| p.file_stem().map(|s| s.to_string_lossy().to_lowercase()));
    let mut names = Vec::new();
    let mut tips = Vec::new();
    for p in files.into_iter().take(MAX_CUSTOM_TIPS) {
        let Ok(img) = image::open(&p) else { continue };
        let g = image::imageops::resize(&img.into_luma8(), TIP_SIZE, TIP_SIZE, image::imageops::FilterType::Triangle);
        names.push(p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
        tips.push(g.into_raw());
    }
    (names, tips)
}

/// Coverage of an image used as a brush: its alpha if it has transparency,
/// else dark = paint (like Photoshop's "Define Brush Preset").
fn coverage(img: &image::DynamicImage) -> imagine_io::abr::AbrTip {
    let rgba = img.to_rgba8();
    let transparent = rgba.pixels().any(|p| p[3] < 250);
    let data = rgba
        .pixels()
        .map(|p| {
            if transparent {
                p[3]
            } else {
                255 - ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8
            }
        })
        .collect();
    imagine_io::abr::AbrTip { name: String::new(), width: rgba.width(), height: rgba.height(), data }
}

/// A file name for a tip that does not overwrite others.
fn unique(dir: &Path, name: &str) -> PathBuf {
    let clean: String = name.chars().map(|c| if c.is_alphanumeric() || " -_".contains(c) { c } else { '_' }).collect();
    let clean = if clean.trim().is_empty() { "tip".to_string() } else { clean.trim().to_string() };
    let mut p = dir.join(format!("{clean}.png"));
    let mut n = 2;
    while p.exists() {
        p = dir.join(format!("{clean} {n}.png"));
        n += 1;
    }
    p
}

/// Adds the tips of an .abr file or an image to the library. Returns their names.
pub fn import(path: &Path, dir: &Path) -> Result<Vec<String>> {
    std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "tip".into());
    let tips = if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("abr")) {
        imagine_io::abr::read(&std::fs::read(path).with_context(|| path.display().to_string())?, &stem)?
    } else {
        let mut t = coverage(&image::open(path).with_context(|| path.display().to_string())?);
        t.name = stem.clone();
        vec![t]
    };
    let mut names = Vec::new();
    for t in tips {
        let px = imagine_io::abr::fit(&t, TIP_SIZE);
        let p = unique(dir, &t.name);
        image::GrayImage::from_raw(TIP_SIZE, TIP_SIZE, px).context("tip size")?.save(&p)?;
        names.push(p.file_stem().unwrap().to_string_lossy().into_owned());
    }
    Ok(names)
}

impl App {
    /// (Re)loads the tip library into the engine.
    pub(super) fn load_custom_tips(&mut self) {
        let (names, tips) = load_library(&brush_dir());
        if let Some(e) = self.engine() {
            e.set_custom_tips(&tips);
        }
        self.custom_tips = names;
        // A tip that no longer exists falls back to round.
        if self.brush.tip >= FIRST_CUSTOM_TIP + self.custom_tips.len() as u32 {
            self.brush.tip = 0;
        }
    }

    pub(super) fn import_brushes(&mut self, path: &Path) {
        match import(path, &brush_dir()) {
            Ok(names) => {
                self.load_custom_tips();
                if let Some(i) = self.custom_tips.iter().position(|n| Some(n) == names.first()) {
                    self.brush.tip = FIRST_CUSTOM_TIP + i as u32;
                }
                tracing::info!("{}", imagine_core::tr!("{n} brush tips imported", n = names.len()));
                if self.custom_tips.len() >= MAX_CUSTOM_TIPS {
                    tracing::warn!("{}", imagine_core::tr!("Only the first {n} custom tips are used", n = MAX_CUSTOM_TIPS));
                }
            }
            Err(e) => tracing::warn!("{}", imagine_core::tr!("Import brushes: {error}", error = format!("{e:#}"))),
        }
        self.needs_redraw = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_becomes_a_tip_in_the_library() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("Square.png");
        // Black square on white: dark = paint.
        let img = image::RgbImage::from_fn(40, 40, |x, y| if (10..30).contains(&x) && (10..30).contains(&y) { image::Rgb([0, 0, 0]) } else { image::Rgb([255, 255, 255]) });
        img.save(&src).unwrap();
        let lib = dir.path().join("lib");
        assert_eq!(import(&src, &lib).unwrap(), vec!["Square".to_string()]);
        // Importing again does not overwrite.
        assert_eq!(import(&src, &lib).unwrap(), vec!["Square 2".to_string()]);
        let (names, tips) = load_library(&lib);
        assert_eq!(names, ["Square", "Square 2"]);
        let n = TIP_SIZE as usize;
        assert_eq!(tips[0][n / 2 * n + n / 2], 255, "centre painted");
        assert_eq!(tips[0][2 * n + 2], 0, "corner empty");
    }
}
