//! The start screen: new image, open, and the recently opened files with
//! thumbnails. Shown when mosaix starts without a file.
//!
//! Thumbnails live in `$XDG_CACHE_HOME/mosaix/thumbs`: written from the
//! document itself when it is saved or left, or decoded from the file for
//! plain images.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use egui::{Align, Color32, CornerRadius, Layout, RichText, Sense, Vec2};
use mosaix_core::tr;

use super::glyphs as g;
use super::panels::{Action, Model};

const THUMB: u32 = 320;
const CARD: Vec2 = Vec2::new(184.0, 176.0);

enum Thumb {
    Loading,
    Ready(egui::ColorImage),
    Missing,
}

#[derive(Default)]
pub struct Home {
    thumbs: Arc<Mutex<HashMap<PathBuf, Thumb>>>,
    textures: HashMap<PathBuf, egui::TextureHandle>,
}

fn cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("mosaix/thumbs")
}

/// FNV-1a: stable across builds (the std hasher is not).
fn cache_path(file: &Path) -> PathBuf {
    let abs = std::fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in abs.as_os_str().as_encoded_bytes() {
        h = (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    cache_dir().join(format!("{h:016x}.png"))
}

/// Stores the thumbnail of `file` (sRGB RGBA, any size) in the background.
pub fn store_thumbnail(file: &Path, img: image::RgbaImage) {
    let out = cache_path(file);
    std::thread::spawn(move || {
        let img = if img.width().max(img.height()) > THUMB {
            image::DynamicImage::ImageRgba8(img).thumbnail(THUMB, THUMB).into_rgba8()
        } else {
            img
        };
        let _ = std::fs::create_dir_all(out.parent().unwrap());
        let tmp = out.with_extension("tmp");
        if img.save_with_format(&tmp, image::ImageFormat::Png).is_ok() {
            let _ = std::fs::rename(&tmp, &out);
        }
    });
}

fn modified(p: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// A plain image the `image` crate decodes quickly enough for a thumbnail.
fn decodable(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "webp" | "tif" | "tiff"))
}

fn load(file: &Path) -> Option<egui::ColorImage> {
    let cached = cache_path(file);
    // A cached picture of a plain image is redone when the file is newer; for
    // documents an older picture is better than none.
    let fresh = match (modified(&cached), modified(file)) {
        (Some(c), Some(f)) => c >= f || !decodable(file),
        (Some(_), None) => true,
        _ => false,
    };
    let img = if fresh {
        image::open(&cached).ok()?.into_rgba8()
    } else if decodable(file) {
        let img = image::open(file).ok()?.thumbnail(THUMB, THUMB).into_rgba8();
        let _ = std::fs::create_dir_all(cached.parent()?);
        let _ = img.save_with_format(&cached, image::ImageFormat::Png);
        img
    } else {
        return None;
    };
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [img.width() as usize, img.height() as usize],
        img.as_raw(),
    ))
}

impl Home {
    /// Starts loading thumbnails that are not known yet.
    fn request(&mut self, files: &[PathBuf]) {
        let mut todo = Vec::new();
        {
            let mut t = self.thumbs.lock().unwrap();
            for f in files {
                if !t.contains_key(f) {
                    t.insert(f.clone(), Thumb::Loading);
                    todo.push(f.clone());
                }
            }
        }
        if todo.is_empty() {
            return;
        }
        let thumbs = self.thumbs.clone();
        std::thread::Builder::new()
            .name("thumbnails".into())
            .spawn(move || {
                for f in todo {
                    let t = load(&f).map_or(Thumb::Missing, Thumb::Ready);
                    thumbs.lock().unwrap().insert(f, t);
                }
            })
            .expect("spawn thumbnail thread");
    }

    /// Waits for all thumbnails (UI snapshots).
    pub fn load_now(&mut self, files: &[PathBuf]) {
        let mut t = self.thumbs.lock().unwrap();
        for f in files {
            t.insert(f.clone(), load(f).map_or(Thumb::Missing, Thumb::Ready));
        }
    }

    /// Moves finished thumbnails into textures; true while some still load.
    fn upload(&mut self, ctx: &egui::Context) -> bool {
        let mut t = self.thumbs.lock().unwrap();
        let mut loading = false;
        for (path, thumb) in t.iter_mut() {
            match thumb {
                Thumb::Loading => loading = true,
                Thumb::Ready(_) => {
                    let Thumb::Ready(img) = std::mem::replace(thumb, Thumb::Missing) else { unreachable!() };
                    let tex = ctx.load_texture(path.display().to_string(), img, egui::TextureOptions::LINEAR);
                    self.textures.insert(path.clone(), tex);
                }
                Thumb::Missing => {}
            }
        }
        loading
    }
}

const PRESETS: [(&str, u32, u32); 4] = [("Full HD", 1920, 1080), ("4K", 3840, 2160), ("A4, 300 dpi", 2480, 3508), ("Square 2048", 2048, 2048)];

fn big_button(ui: &mut egui::Ui, icon: &'static str, text: &str) -> egui::Response {
    ui.add_sized(
        [220.0, 36.0],
        egui::Button::new((
            super::menus::atom(super::menus::Icon::Glyph(icon)),
            RichText::new(text).size(15.0),
        )),
    )
}

/// Draws the start screen over `rect` (the canvas area). Returns false when
/// it should close.
pub fn show(home: &mut Home, ctx: &egui::Context, rect: egui::Rect, m: &Model, st_new: &mut Option<super::files::NewDialog>, out: &mut Vec<Action>) -> bool {
    let mut keep = !ctx.input(|i| i.key_pressed(egui::Key::Escape));
    home.request(&m.recent);
    if home.upload(ctx) {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
    let bg = ctx.global_style().visuals.panel_fill;
    // A little over the edges: the canvas must not peek out.
    let rect = rect.expand(3.0);
    egui::Area::new(egui::Id::unique("home"))
        .fixed_pos(rect.min)
        .constrain(false)
        .order(egui::Order::Middle)
        .show(ctx, |ui| {
            ui.set_min_size(rect.size());
            ui.set_max_size(rect.size());
            ui.painter().rect_filled(rect, CornerRadius::ZERO, bg);
            egui::Frame::new().inner_margin(32.0).show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(240.0);
                        ui.label(RichText::new("Mosaix").size(28.0).strong());
                        ui.add_space(20.0);
                        if big_button(ui, g::FILE_PLUS, tr("New Image …")).clicked() {
                            *st_new = Some(Default::default());
                        }
                        if big_button(ui, g::FOLDER_OPEN, tr("Open …")).clicked()
                            && let Some(c) = super::palette::COMMANDS.iter().find(|c| c.label == "Open …")
                        {
                            out.push(Action::Command(*c));
                        }
                        if big_button(ui, g::IMAGE_SQUARE, tr("Empty Canvas")).on_hover_text(tr!("{w} × {h} px, transparent", w = m.doc_size.0, h = m.doc_size.1)).clicked() {
                            keep = false;
                        }
                        ui.add_space(20.0);
                        ui.label(RichText::new(tr("Quick start")).weak());
                        for (name, w, h) in PRESETS {
                            let b = egui::Button::new(format!("{}   {w} × {h}", tr(name))).frame(false);
                            if ui.add(b).clicked() {
                                out.push(Action::NewDocument { w, h, white: true });
                            }
                        }
                    });
                    ui.add_space(24.0);
                    ui.separator();
                    ui.add_space(24.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new(tr("Recently opened")).size(18.0));
                        ui.add_space(12.0);
                        if m.recent.is_empty() {
                            ui.label(RichText::new(tr("Files you open show up here.")).weak());
                            return;
                        }
                        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = Vec2::splat(14.0);
                                for p in &m.recent {
                                    if card(ui, p, home.textures.get(p)).clicked() {
                                        out.push(Action::OpenFile(p.clone()));
                                    }
                                }
                            });
                        });
                    });
                });
            });
        });
    keep
}

fn card(ui: &mut egui::Ui, path: &Path, tex: Option<&egui::TextureHandle>) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(CARD, Sense::click());
    let v = ui.visuals().clone();
    let exists = path.exists();
    let fill = if resp.hovered() && exists { v.widgets.hovered.weak_bg_fill } else { v.extreme_bg_color };
    let p = ui.painter();
    p.rect_filled(rect, 6.0, fill);
    let img_rect = egui::Rect::from_min_size(rect.min + Vec2::splat(8.0), Vec2::new(CARD.x - 16.0, 116.0));
    match tex {
        Some(t) => {
            let s = t.size_vec2();
            let f = (img_rect.width() / s.x).min(img_rect.height() / s.y);
            let r = egui::Rect::from_center_size(img_rect.center(), s * f);
            super::gradient_ui::checker(p, r);
            p.image(t.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        None => {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_uppercase();
            p.text(img_rect.center() - Vec2::new(0.0, 8.0), egui::Align2::CENTER_CENTER, g::FILE_IMAGE, egui::FontId::proportional(34.0), v.weak_text_color());
            p.text(img_rect.center() + Vec2::new(0.0, 24.0), egui::Align2::CENTER_CENTER, ext, egui::FontId::proportional(11.0), v.weak_text_color());
        }
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let dir = path.parent().map(|d| d.display().to_string()).unwrap_or_default();
    let text_rect = egui::Rect::from_min_max(egui::pos2(rect.min.x + 8.0, img_rect.max.y + 6.0), rect.max - Vec2::splat(6.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(text_rect).layout(Layout::top_down(Align::Min)));
    child.add(egui::Label::new(RichText::new(&name).color(if exists { v.strong_text_color() } else { v.weak_text_color() })).truncate());
    let sub = if exists { dir.clone() } else { tr("not found").to_string() };
    child.add(egui::Label::new(RichText::new(sub).small().weak()).truncate());
    if exists { resp.on_hover_text(path.display().to_string()) } else { resp }
}
