//! The RAW development dialog shown when a camera RAW file is opened: a
//! preview with white balance and exposure, then "Open" develops the full
//! image.

use std::path::PathBuf;
use std::sync::mpsc;

use egui::{RichText, Vec2};
use mosaix_core::i18n::tr_ctx;
use mosaix_core::tr;
use mosaix_io::raw::{CameraImage, Develop, Highlights, RawFile, WhiteBalance};

use super::panels::Action;

/// Longer side of the preview's camera image.
const PREVIEW: usize = 1400;
const VIEW: Vec2 = Vec2::new(640.0, 460.0);

struct Loaded {
    file: RawFile,
    cam: CameraImage,
    /// `cam` with noise reduction, and the settings used.
    denoised: Option<(mosaix_io::raw::NoiseReduction, CameraImage)>,
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
                    .and_then(|file| Ok(Loaded { cam: file.camera_image()?.downscale(PREVIEW), file, denoised: None }))
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
            && let Some(l) = &mut self.loaded
        {
            self.dirty = false;
            let n = self.develop.noise;
            if !n.is_off() && l.denoised.as_ref().is_none_or(|d| d.0 != n) {
                let mut c = l.cam.clone();
                l.file.denoise(&mut c, &n);
                l.denoised = Some((n, c));
            }
            let cam = if n.is_off() { &l.cam } else { &l.denoised.as_ref().unwrap().1 };
            let (w, h, px) = l.file.develop(cam, &self.develop);
            let rgba: Vec<u8> = px
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    let e = |v: f32| (mosaix_core::color::linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
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
        let Some(l) = &self.loaded else {
            return;
        };
        info_lines(ui, &l.file.info);
        ui.add_space(8.0);
        let (file, cam) = (&l.file, &l.cam);
        let e = develop_controls(ui, &mut self.develop, "raw dialog", file.info.lens_profile.as_deref(), |wb| file.temperature_of(file.multipliers(wb, cam)));
        self.dirty |= e.changed;
    }
}

/// Camera, lens and shot data.
pub fn info_lines(ui: &mut egui::Ui, i: &mosaix_io::raw::Info) {
    ui.label(RichText::new(&i.camera).weak());
    if let Some(lens) = &i.lens {
        ui.label(RichText::new(lens).weak());
    }
    ui.label(RichText::new(i.shot()).weak());
}

/// White balance, temperature, tint and exposure. `temperature_of` gives
/// what the non-custom settings stand for.
pub fn develop_controls(
    ui: &mut egui::Ui,
    develop: &mut Develop,
    salt: &str,
    lens_profile: Option<&str>,
    temperature_of: impl Fn(WhiteBalance) -> (f32, f32),
) -> super::gradient_ui::Out {
    let mut out = super::gradient_ui::Out::default();
    let wb = develop.white_balance;
    let mut mode = Mode::of(wb);
    let (mut temp, mut tint) = temperature_of(wb);
    // Sliders round what they show; only a real move switches to Custom.
    let (temp0, tint0) = (temp, tint);
    let mut wb_changed = false;
    let track = |out: &mut super::gradient_ui::Out, r: &egui::Response| {
        if r.drag_stopped() || r.lost_focus() || (r.changed() && !r.dragged() && !r.has_focus()) {
            out.done = true;
        }
    };
    egui::Grid::new(("raw controls", salt)).num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
        ui.label(tr("White Balance"));
        egui::ComboBox::from_id_salt(("raw wb", salt)).selected_text(mode.label()).show_ui(ui, |ui| {
            for m in Mode::ALL {
                if ui.selectable_label(mode == m, m.label()).clicked() && mode != m {
                    mode = m;
                    wb_changed = true;
                    out.done = true;
                }
            }
        });
        ui.end_row();
        ui.label(tr("Temperature"));
        let r = ui.add(egui::Slider::new(&mut temp, 2000.0..=25000.0).logarithmic(true).suffix(" K").max_decimals(0));
        if (temp - temp0).abs() > 1.0 {
            mode = Mode::Custom;
            wb_changed = true;
        }
        track(&mut out, &r);
        ui.end_row();
        ui.label(tr("Tint"));
        let r = ui.add(egui::Slider::new(&mut tint, -150.0..=150.0).max_decimals(0));
        if (tint - tint0).abs() > 0.5 {
            mode = Mode::Custom;
            wb_changed = true;
        }
        track(&mut out, &r);
        ui.end_row();
        ui.label(tr("Exposure"));
        let r = ui.add(egui::Slider::new(&mut develop.exposure, -5.0..=5.0).suffix(" EV").max_decimals(2));
        out.changed |= r.changed();
        track(&mut out, &r);
        ui.end_row();
        // Tone and colour: fast, applied live.
        let live = |ui: &mut egui::Ui, out: &mut super::gradient_ui::Out, name: &str, v: &mut f32| {
            ui.label(name);
            let r = ui.add(egui::Slider::new(v, -100.0..=100.0).max_decimals(0));
            out.changed |= r.changed();
            track(out, &r);
            if r.double_clicked() {
                *v = 0.0;
                out.changed = true;
                out.done = true;
            }
            ui.end_row();
        };
        live(ui, &mut out, tr("Contrast"), &mut develop.tone.contrast);
        live(ui, &mut out, tr("Highlights"), &mut develop.tone.highlights);
        live(ui, &mut out, tr("Shadows"), &mut develop.tone.shadows);
        live(ui, &mut out, tr("Vibrance"), &mut develop.tone.vibrance);
        live(ui, &mut out, tr("Saturation"), &mut develop.tone.saturation);
        ui.label(tr("Clipped Highlights"));
        let label = |h: Highlights| match h {
            Highlights::Clip => tr_ctx("highlights", "Clip"),
            Highlights::Reconstruct => tr("Reconstruct"),
        };
        egui::ComboBox::from_id_salt(("raw highlights", salt)).selected_text(label(develop.highlights)).show_ui(ui, |ui| {
            for h in [Highlights::Reconstruct, Highlights::Clip] {
                if ui.selectable_label(develop.highlights == h, label(h)).clicked() && develop.highlights != h {
                    develop.highlights = h;
                    out.changed = true;
                    out.done = true;
                }
            }
        });
        ui.end_row();
        ui.label(RichText::new(tr("Noise Reduction")).strong());
        ui.end_row();
        // Denoising takes a moment: the value applies when the slider is let go.
        let mut slow = |ui: &mut egui::Ui, v: &mut f32, which: &str| {
            let id = ui.scope_id().with(("raw nr", salt, which));
            let mut x: f32 = ui.data(|d| d.get_temp(id)).unwrap_or(*v);
            let r = ui.add(egui::Slider::new(&mut x, 0.0..=100.0).max_decimals(0));
            if r.dragged() || r.has_focus() {
                ui.data_mut(|d| d.insert_temp(id, x));
            } else {
                ui.data_mut(|d| d.remove::<f32>(id));
                if (x - *v).abs() > 0.01 {
                    *v = x;
                    out.changed = true;
                    out.done = true;
                }
            }
        };
        ui.label(tr("Luminance"));
        slow(ui, &mut develop.noise.luminance, "luminance");
        ui.end_row();
        ui.label(tr_ctx("noise", "Color"));
        slow(ui, &mut develop.noise.color, "color");
        ui.end_row();
    });
    ui.add_space(4.0);
    match lens_profile {
        Some(name) => {
            if ui
                .checkbox(&mut develop.lens, tr("Lens Correction"))
                .on_hover_text(tr("Distortion, color fringes and vignetting from the lensfun profile"))
                .changed()
            {
                out.changed = true;
                out.done = true;
            }
            ui.label(RichText::new(name).weak().small());
        }
        None => {
            ui.label(RichText::new(tr("No lens profile found")).weak().small());
        }
    }
    if wb_changed {
        develop.white_balance = match mode {
            Mode::AsShot => WhiteBalance::AsShot,
            Mode::Auto => WhiteBalance::Auto,
            Mode::Daylight => WhiteBalance::Daylight,
            Mode::Custom => WhiteBalance::Custom { temperature: temp, tint },
        };
        out.changed = true;
    }
    if ui.small_button(tr("Reset")).clicked() && *develop != Develop::default() {
        *develop = Develop::default();
        out.changed = true;
        out.done = true;
    }
    out
}

/// The Properties panel of a RAW layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveRaw {
    pub file: String,
    pub develop: Develop,
    pub summary: Option<mosaix_gpu::engine::raw::RawSummary>,
}

/// Returns the new settings and whether the edit is finished.
pub fn layer_editor(ui: &mut egui::Ui, a: &ActiveRaw) -> Option<(Develop, bool)> {
    ui.label(RichText::new(&a.file).strong());
    let Some(sum) = &a.summary else {
        ui.add(egui::Spinner::new());
        return None;
    };
    info_lines(ui, &sum.info);
    ui.label(RichText::new(format!("{} × {} px", sum.size.0, sum.size.1)).weak());
    ui.add_space(8.0);
    let mut d = a.develop;
    let e = develop_controls(ui, &mut d, "raw layer", sum.info.lens_profile.as_deref(), |wb| sum.temperature(wb));
    ui.add_space(6.0);
    ui.label(RichText::new(tr("Ctrl+T transforms losslessly, Layer › Rasterize turns it into pixels")).weak().small());
    (e.changed || e.done).then_some((d, e.done))
}
