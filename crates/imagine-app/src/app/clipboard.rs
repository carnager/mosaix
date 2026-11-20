//! The system clipboard (Wayland data device): copy, cut and paste text
//! (while editing text, in UI fields) and pixels (as PNG; pasted images
//! become a new layer). Our own clipboard content is used directly, so
//! copying and pasting within imagine keeps the position.

use std::io::{Read, Write};

use image::ImageEncoder;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use smithay_client_toolkit::data_device_manager::data_device::{DataDevice, DataDeviceHandler};
use smithay_client_toolkit::data_device_manager::data_offer::{DataOfferHandler, DragOffer};
use smithay_client_toolkit::data_device_manager::data_source::{CopyPasteSource, DataSourceHandler};
use smithay_client_toolkit::data_device_manager::{DataDeviceManagerState, WritePipe};
use smithay_client_toolkit::reexports::client::protocol::{wl_data_device::WlDataDevice, wl_data_device_manager::DndAction, wl_data_source::WlDataSource, wl_surface::WlSurface};
use smithay_client_toolkit::reexports::client::{Connection, QueueHandle};

use super::{App, BgDone};

const TEXT_MIMES: [&str; 6] = ["text/plain;charset=utf-8", "UTF8_STRING", "text/plain", "STRING", "TEXT", "text/plain;charset=UTF-8"];
const PNG: &str = "image/png";

#[derive(Clone)]
pub(super) enum Clip {
    Text(String),
    /// An image and where it came from (pasting in imagine puts it back there).
    Image { png: Arc<Vec<u8>>, img: Arc<image::RgbaImage>, x: i64, y: i64 },
}

#[derive(Default)]
pub(super) struct Clipboard {
    pub manager: Option<DataDeviceManagerState>,
    pub device: Option<DataDevice>,
    source: Option<CopyPasteSource>,
    content: Option<Clip>,
    /// Serial of the last key press (needed to set the selection).
    pub serial: u32,
    /// Where a paste goes once it has been read.
    pending: Option<PasteTarget>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PasteTarget {
    /// The text being edited on the canvas.
    Text,
    /// A text field of the UI.
    Ui,
    /// The document: image → new layer, text → new text layer.
    Document,
}

/// Premultiplied linear pixels → straight sRGB 8 bit.
fn to_rgba8(w: usize, h: usize, px: &[[f32; 4]]) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for (o, p) in img.pixels_mut().zip(px) {
        let a = p[3].clamp(0.0, 1.0);
        if a > 0.0 {
            let c = |v: f32| (imagine_core::color::linear_to_srgb((v / a).clamp(0.0, 1.0)) * 255.0).round() as u8;
            *o = image::Rgba([c(p[0]), c(p[1]), c(p[2]), (a * 255.0).round() as u8]);
        }
    }
    img
}

impl Clipboard {
    pub fn new(manager: Option<DataDeviceManagerState>) -> Self {
        Self { manager, ..Default::default() }
    }
}

impl App {
    /// Offers `clip` on the system clipboard.
    pub(super) fn clip_set(&mut self, clip: Clip) {
        let cb = &mut self.clipboard;
        let (Some(manager), Some(device)) = (&cb.manager, &cb.device) else {
            tracing::warn!("Zwischenablage nicht verfügbar");
            return;
        };
        let mimes: Vec<&str> = match &clip {
            Clip::Text(_) => TEXT_MIMES.to_vec(),
            Clip::Image { .. } => vec![PNG],
        };
        let source = manager.create_copy_paste_source(&self.qh, mimes);
        source.set_selection(device, cb.serial);
        cb.source = Some(source);
        cb.content = Some(clip);
    }

    /// Copies text (while editing), else the layer's (or with `merged`
    /// the image's) pixels in the selection. `cut` also removes them.
    pub(super) fn copy(&mut self, cut: bool, merged: bool) {
        if let Some(text) = self.text_edit_selected() {
            if !text.is_empty() {
                self.clip_set(Clip::Text(text));
                if cut {
                    self.text_edit_delete_selection();
                }
            }
            return;
        }
        let Some(e) = self.engine() else { return };
        let copied = match e.copy_pixels(merged) {
            Ok(c) => c,
            Err(err) => return tracing::warn!("Kopieren: {err}"),
        };
        let img = to_rgba8(copied.w, copied.h, &copied.px);
        let mut png = Vec::new();
        if let Err(err) = image::codecs::png::PngEncoder::new_with_quality(&mut png, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::Adaptive)
            .write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgba8)
        {
            return tracing::warn!("Kopieren: {err}");
        }
        tracing::info!("{} {}×{} px", if cut { "Ausgeschnitten:" } else { "Kopiert:" }, copied.w, copied.h);
        self.clip_set(Clip::Image { png: Arc::new(png), img: Arc::new(img), x: copied.x, y: copied.y });
        if cut && !merged {
            self.engine_op(|e| e.clear_selected("Ausschneiden").map(|_| ()));
        }
    }

    /// Pastes into `target`: our own content directly, else read from the
    /// clipboard owner on a worker thread.
    pub(super) fn paste(&mut self, target: PasteTarget) {
        if let Some(clip) = self.clipboard.source.as_ref().and(self.clipboard.content.clone()) {
            self.paste_clip(clip, target);
            return;
        }
        let Some(offer) = self.clipboard.device.as_ref().and_then(|d| d.data().selection_offer()) else {
            return tracing::info!("Die Zwischenablage ist leer");
        };
        let want_image = target == PasteTarget::Document;
        let mime = offer.with_mime_types(|m| {
            let png = m.iter().find(|t| t.as_str() == PNG).cloned();
            let text = TEXT_MIMES.iter().find_map(|t| m.iter().find(|x| x.as_str() == *t).cloned());
            if want_image { png.or(text) } else { text }
        });
        let Some(mime) = mime else {
            return tracing::info!("Nichts Einfügbares in der Zwischenablage");
        };
        let pipe = match offer.receive(mime.clone()) {
            Ok(p) => p,
            Err(err) => return tracing::warn!("Einfügen: {err:?}"),
        };
        self.clipboard.pending = Some(target);
        self.spawn_bg("paste", move || {
            let mut f = std::fs::File::from(OwnedFd::from(pipe));
            let mut data = Vec::new();
            let result = f.read_to_end(&mut data).map(|_| (mime, data)).map_err(|e| e.to_string());
            BgDone::Pasted(result)
        });
    }

    /// The data from another program has arrived.
    pub(super) fn on_pasted(&mut self, result: Result<(String, Vec<u8>), String>) {
        let Some(target) = self.clipboard.pending.take() else { return };
        let clip = match result {
            Ok((mime, data)) if mime == PNG => match image::load_from_memory(&data) {
                Ok(img) => {
                    let img = img.into_rgba8();
                    // From elsewhere: centred in the view.
                    let c = self.view.screen_to_doc(self.view.viewport * 0.5);
                    let (x, y) = ((c.x - img.width() as f64 / 2.0).round() as i64, (c.y - img.height() as f64 / 2.0).round() as i64);
                    Clip::Image { png: Arc::new(data), img: Arc::new(img), x, y }
                }
                Err(err) => return tracing::warn!("Einfügen: Bild unlesbar ({err})"),
            },
            Ok((_, data)) => Clip::Text(String::from_utf8_lossy(&data).replace("\r\n", "\n")),
            Err(err) => return tracing::warn!("Einfügen: {err}"),
        };
        self.paste_clip(clip, target);
    }

    fn paste_clip(&mut self, clip: Clip, target: PasteTarget) {
        match (clip, target) {
            (Clip::Text(t), PasteTarget::Text) => self.text_edit_insert(&t),
            (Clip::Text(t), PasteTarget::Ui) => {
                self.gui.paste(&t);
                self.needs_redraw = true;
            }
            (Clip::Text(t), PasteTarget::Document) => {
                // A new text layer in the middle of the view.
                let c = self.view.screen_to_doc(self.view.viewport * 0.5);
                let size = (24.0 / self.view.zoom as f32).clamp(8.0, 2000.0).round();
                let color = self.brush.color;
                let spec = imagine_core::vector::TextSpec::new(&t, c.x, c.y, size, [color[0], color[1], color[2], 1.0]);
                if let Some(e) = self.engine()
                    && e.add_vector(imagine_core::VectorLayer::text(spec), "Text einfügen").is_some()
                {
                    self.touch();
                }
            }
            (Clip::Image { img, x, y, .. }, PasteTarget::Document) => {
                let Some(s) = &mut self.session else { return };
                match crate::agent::command::place_rgba(s, &img, x, y, "Eingefügt", "Einfügen") {
                    Ok(_) => {
                        tracing::info!("Eingefügt als neue Ebene ({}×{} px)", img.width(), img.height());
                        self.touch();
                    }
                    Err(err) => tracing::warn!("Einfügen: {err:#}"),
                }
            }
            (Clip::Image { .. }, _) => tracing::info!("Ein Bild lässt sich nicht in Text einfügen"),
        }
        self.needs_redraw = true;
    }

    /// Text the UI copied (egui text fields).
    pub(super) fn ui_copied(&mut self, text: String) {
        if !text.is_empty() {
            self.clip_set(Clip::Text(text));
        }
    }
}

impl DataDeviceHandler for App {
    fn enter(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice, _: f64, _: f64, _: &WlSurface) {}
    fn leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
    fn motion(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice, _: f64, _: f64) {}
    fn selection(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
    fn drop_performed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
}

impl DataOfferHandler for App {
    fn source_actions(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &mut DragOffer, _: DndAction) {}
    fn selected_action(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &mut DragOffer, _: DndAction) {}
}

impl DataSourceHandler for App {
    fn accept_mime(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource, _: Option<String>) {}

    fn send_request(&mut self, _: &Connection, _: &QueueHandle<Self>, source: &WlDataSource, mime: String, fd: WritePipe) {
        let ours = self.clipboard.source.as_ref().is_some_and(|s| s.inner() == source);
        let data: Option<Arc<Vec<u8>>> = match (&self.clipboard.content, ours) {
            (Some(Clip::Text(t)), true) if TEXT_MIMES.contains(&mime.as_str()) => Some(Arc::new(t.clone().into_bytes())),
            (Some(Clip::Image { png, .. }), true) if mime == PNG => Some(png.clone()),
            _ => None,
        };
        let Some(data) = data else { return };
        // Large images: write on a thread so the window keeps running.
        std::thread::spawn(move || {
            let mut f = std::fs::File::from(OwnedFd::from(fd));
            let _ = f.write_all(&data);
        });
    }

    fn cancelled(&mut self, _: &Connection, _: &QueueHandle<Self>, source: &WlDataSource) {
        // Another program owns the clipboard now.
        if self.clipboard.source.as_ref().is_some_and(|s| s.inner() == source) {
            self.clipboard.source = None;
            self.clipboard.content = None;
        }
        source.destroy();
    }

    fn dnd_dropped(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
    fn dnd_finished(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
    fn action(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource, _: DndAction) {}
}
