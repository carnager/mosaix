//! The RAW development dialog shown when a camera RAW file is opened: a
//! preview with white balance and exposure, then "Open" develops the full
//! image.

use std::path::PathBuf;
use std::sync::mpsc;

use egui::{RichText, Vec2};
use imagine_core::i18n::tr_ctx;
use imagine_core::tr;
use imagine_io::raw::{CameraImage, Develop, RawFile, WhiteBalance};

use super::panels::Action;

/// Longer side of the preview's camera image.
const PREVIEW: usize = 1400;
const VIEW: Vec2 = Vec2::new(640.0, 460.0);

struct Loaded {
    file: RawFile,
    cam: CameraImage,
}

pub struct RawDialog {
    pub path: PathBuf,
    rx: Option<mpsc::Receiver<Result<Loaded, String>>>,
    loaded: Option<Loaded>,
    error: Option<String>,
    pub develop: Develop,
    tex: Option<egui::TextureHandle>,
    dirty: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    AsShot,
    Auto,
    Daylight,
    Custom,
}

impl Mode {
    const ALL: [Mode; 4] = [Mode::AsShot, Mode::Auto, Mode::Daylight, Mode::Custom];

    fn of(wb: WhiteBalance) -> Mode {
        match wb {
            WhiteBalance::AsShot => Mode::AsShot,
            WhiteBalance::Auto => Mode::Auto,
            WhiteBalance::Daylight => Mode::Daylight,
            WhiteBalance::Custom { .. } => Mode::Custom,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::AsShot => tr("As Shot"),
            Mode::Auto => tr_ctx("white balance", "Auto"),
            Mode::Daylight => tr("Daylight"),
            Mode::Custom => tr_ctx("white balance", "Custom"),
        }
    }
}

impl RawDialog {
    pub fn new(path: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel();
        let p = path.clone();
        std::thread::Builder::new()
            .name("raw preview".into())
            .spawn(move || {
                let r = RawFile::open(&p)
                    .and_then(|file| Ok(Loaded { cam: file.camera_image()?.downscale(PREVIEW), file }))
                    .map_err(|e| format!("{e:#}"));
                let _ = tx.send(r);
            })
            .expect("spawn raw preview thread");
        Self { path, rx: Some(rx), loaded: None, error: None, develop: Develop::default(), tex: None, dirty: true }
    }

    /// Blocks until the preview is decoded (snapshots).
    pub fn wait(&mut self) {
        if let Some(rx) = self.rx.take() {
            match rx.recv() {
                Ok(Ok(l)) => self.loaded = Some(l),
                Ok(Err(e)) => self.error = Some(e),
                Err(_) => {}
            }
        }
    }

    /// Shows the dialog; false once it is closed.
    pub fn show(&mut self, ctx: &egui::Context, out: &mut Vec<Action>) -> bool {
        if let Some(rx) = &self.rx {
            match rx.try_recv() {
                Ok(Ok(l)) => {
                    self.loaded = Some(l);
                    self.rx = None;
                }
                Ok(Err(e)) => {
                    self.error = Some(e);
                    self.rx = None;
                }
                Err(mpsc::TryRecvError::Empty) => ctx.request_repaint_after(std::time::Duration::from_millis(40)),
                Err(mpsc::TryRecvError::Disconnected) => self.rx = None,
            }
        }
        if self.dirty
            && let Some(l) = &self.loaded
        {
            self.dirty = false;
            let (w, h, px) = l.file.develop(&l.cam, &self.develop);
            let rgba: Vec<u8> = px
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    let e = |v: f32| (imagine_core::color::linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                    [e(p[0]), e(p[1]), e(p[2]), 255]
                })
                .collect();
            let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
            match &mut self.tex {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => self.tex = Some(ctx.load_texture("raw preview", img, egui::TextureOptions::LINEAR)),
            }
        }
        let mut open = true;
        let mut done = false;
        let name = self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        egui::Window::new(tr("Develop RAW"))
            .id(egui::Id::unique("raw dialog"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    let (rect, _) = ui.allocate_exact_size(VIEW, egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, egui::Color32::from_gray(28));
                    match (&self.tex, &self.error) {
                        (_, Some(e)) => {
                            ui.painter().text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                e,
                                egui::FontId::proportional(13.0),
                                ui.visuals().error_fg_color,
                            );
                        }
                        (Some(t), None) => {
                            let s = t.size_vec2();
                            let k = (VIEW.x / s.x).min(VIEW.y / s.y);
                            let r = egui::Rect::from_center_size(rect.center(), s * k);
                            ui.painter().image(
                                t.id(),
                                r,
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                        }
                        (None, None) => {
                            ui.put(
                                egui::Rect::from_center_size(rect.center(), Vec2::new(200.0, 40.0)),
                                egui::Spinner::new().size(24.0),
                            );
                        }
                    }
                    ui.vertical(|ui| {
                        ui.set_width(250.0);
                        self.controls(ui, &name);
                        ui.add_space(12.0);
                        ui.horizontal(|ui| {
                            let ready = self.loaded.is_some();
                            if ui.add_enabled(ready, egui::Button::new(tr("Open"))).clicked()
                                || (ready && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                            {
                                out.push(Action::OpenRaw(self.path.clone(), self.develop));
                                done = true;
                            }
                            if ui.button(tr("Cancel")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                done = true;
                            }
                        });
                    });
                });
            });
        open && !done
    }

    fn controls(&mut self, ui: &mut egui::Ui, name: &str) {
        ui.label(RichText::new(name).strong());
        if let Some(l) = &self.loaded {
            let i = &l.file.info;
            ui.label(RichText::new(&i.camera).weak());
            if let Some(lens) = &i.lens {
                ui.label(RichText::new(lens).weak());
            }
            ui.label(RichText::new(i.shot()).weak());
        }
        ui.add_space(8.0);
        let Some(l) = &self.loaded else {
            return;
        };
        let wb = self.develop.white_balance;
        let mut mode = Mode::of(wb);
        // What the current setting is as temperature and tint.
        let (mut temp, mut tint) = match wb {
            WhiteBalance::Custom { temperature, tint } => (temperature, tint),
            _ => l.file.temperature_of(l.file.multipliers(wb, &l.cam)),
        };
        // Sliders round what they show; only a real move switches to Custom.
        let (temp0, tint0) = (temp, tint);
        let mut changed = false;
        egui::Grid::new("raw controls").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label(tr("White Balance"));
            egui::ComboBox::from_id_salt("raw wb").selected_text(mode.label()).show_ui(ui, |ui| {
                for m in Mode::ALL {
                    if ui.selectable_label(mode == m, m.label()).clicked() && mode != m {
                        mode = m;
                        changed = true;
                    }
                }
            });
            ui.end_row();
            ui.label(tr("Temperature"));
            ui.add(egui::Slider::new(&mut temp, 2000.0..=25000.0).logarithmic(true).suffix(" K").max_decimals(0));
            if (temp - temp0).abs() > 1.0 {
                mode = Mode::Custom;
                changed = true;
            }
            ui.end_row();
            ui.label(tr("Tint"));
            ui.add(egui::Slider::new(&mut tint, -150.0..=150.0).max_decimals(0));
            if (tint - tint0).abs() > 0.5 {
                mode = Mode::Custom;
                changed = true;
            }
            ui.end_row();
            ui.label(tr("Exposure"));
            if ui
                .add(egui::Slider::new(&mut self.develop.exposure, -5.0..=5.0).suffix(" EV").max_decimals(2))
                .changed()
            {
                self.dirty = true;
            }
            ui.end_row();
        });
        if changed {
            self.develop.white_balance = match mode {
                Mode::AsShot => WhiteBalance::AsShot,
                Mode::Auto => WhiteBalance::Auto,
                Mode::Daylight => WhiteBalance::Daylight,
                Mode::Custom => WhiteBalance::Custom { temperature: temp, tint },
            };
            self.dirty = true;
        }
        if ui.small_button(tr("Reset")).clicked() {
            self.develop = Develop::default();
            self.dirty = true;
        }
    }
}
