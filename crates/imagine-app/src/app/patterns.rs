//! The pattern library: imagine's built-in patterns plus PNGs in
//! `$XDG_DATA_HOME/imagine/patterns` (filled by Edit › Define Pattern and
//! Edit › Import Patterns: images and GIMP .pat files).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use imagine_core::pattern::{MAX_PATTERN, Pattern};
use imagine_core::{tr, tr as trm};

use super::App;

pub fn pattern_dir() -> PathBuf {
    super::brushes::brush_dir().with_file_name("patterns")
}

/// Built-ins, then the library (sorted by name).
pub fn load_library(dir: &Path) -> Vec<Pattern> {
    let mut out = Pattern::builtin();
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"))).collect())
        .unwrap_or_default();
    files.sort_by_key(|p| p.file_stem().map(|s| s.to_string_lossy().to_lowercase()));
    for p in files {
        let Ok(img) = image::open(&p) else { continue };
        let img = img.into_rgba8();
        let name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(pat) = Pattern::new(&name, img.width(), img.height(), img.into_raw()) {
            out.push(pat);
        }
    }
    out
}

/// A file name that does not overwrite others.
fn unique(dir: &Path, name: &str) -> PathBuf {
    let clean: String = name.chars().map(|c| if c.is_alphanumeric() || " -_".contains(c) { c } else { '_' }).collect();
    let clean = if clean.trim().is_empty() { "pattern".to_string() } else { clean.trim().to_string() };
    let mut p = dir.join(format!("{clean}.png"));
    let mut n = 2;
    while p.exists() {
        p = dir.join(format!("{clean} {n}.png"));
        n += 1;
    }
    p
}

/// Saves straight RGBA pixels as a library pattern (scaled down if huge).
pub fn save(dir: &Path, name: &str, img: image::RgbaImage) -> Result<String> {
    std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    let img = if img.width().max(img.height()) > MAX_PATTERN {
        let k = MAX_PATTERN as f32 / img.width().max(img.height()) as f32;
        image::imageops::resize(&img, ((img.width() as f32 * k) as u32).max(1), ((img.height() as f32 * k) as u32).max(1), image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let p = unique(dir, name);
    img.save(&p)?;
    Ok(p.file_stem().unwrap().to_string_lossy().into_owned())
}

/// Adds the pattern(s) of a GIMP .pat file or an image to the library.
pub fn import(path: &Path, dir: &Path) -> Result<Vec<String>> {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "pattern".into());
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let (name, img) = if ext == "pat" {
        let (name, w, h, px) = imagine_io::gpat::read(&std::fs::read(path).with_context(|| path.display().to_string())?)?;
        let name = if name.is_empty() { stem } else { name };
        (name, image::RgbaImage::from_raw(w, h, px).context("pattern size")?)
    } else {
        (stem, image::open(path).with_context(|| path.display().to_string())?.into_rgba8())
    };
    Ok(vec![save(dir, &name, img)?])
}

/// Finds a pattern by name (case-insensitive): built-in or library.
pub fn find(library: &[Pattern], name: &str) -> Option<Pattern> {
    library.iter().find(|p| p.name.eq_ignore_ascii_case(name)).cloned()
}

impl App {
    pub(super) fn load_patterns(&mut self) {
        self.patterns = Arc::new(load_library(&pattern_dir()));
    }

    pub(super) fn import_patterns(&mut self, path: &Path) {
        match import(path, &pattern_dir()) {
            Ok(names) => {
                self.load_patterns();
                tracing::info!("{}", trm!("{n} patterns imported", n = names.len()));
            }
            Err(e) => tracing::warn!("{}", trm!("Import patterns: {error}", error = format!("{e:#}"))),
        }
        self.needs_redraw = true;
    }

    /// Edit › Define Pattern: the visible pixels in the selection (or the
    /// whole image) become a library pattern.
    pub(super) fn define_pattern(&mut self) {
        let Some(e) = self.engine() else { return };
        let copied = match e.copy_pixels(true) {
            Ok(c) => c,
            Err(err) => return tracing::warn!("{}", trm!("Define pattern: {error}", error = tr(err))),
        };
        let img = super::clipboard::to_rgba8(copied.w, copied.h, &copied.px);
        let n = self.patterns.len() - Pattern::builtin().len() + 1;
        match save(&pattern_dir(), &trm!("Pattern {n}", n = n), img) {
            Ok(name) => {
                self.load_patterns();
                tracing::info!("{}", trm!("Pattern “{name}” defined", name = name));
            }
            Err(err) => tracing::warn!("{}", trm!("Define pattern: {error}", error = format!("{err:#}"))),
        }
        self.needs_redraw = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_has_builtins_and_imports() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("Bricks.png");
        image::RgbaImage::from_pixel(8, 4, image::Rgba([200, 80, 40, 255])).save(&src).unwrap();
        let lib = dir.path().join("lib");
        assert_eq!(import(&src, &lib).unwrap(), vec!["Bricks".to_string()]);
        let all = load_library(&lib);
        assert_eq!(all.len(), Pattern::builtin().len() + 1);
        let b = find(&all, "bricks").unwrap();
        assert_eq!((b.width, b.height, &b.pixels[..4]), (8, 4, &[200u8, 80, 40, 255][..]));
        assert!(find(&all, &Pattern::builtin()[1].name).is_some());
    }
}
