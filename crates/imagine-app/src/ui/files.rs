//! File dialogs drawn with egui (open / save as / export as), the "new
//! document" dialog and the unsaved-changes question.

use std::path::{Path, PathBuf};

use egui::{Color32, RichText, Vec2};
use imagine_core::tr;

use imagine_io::export::{ColorSpace, ExportOptions};

use super::panels::Action;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileMode {
    Open,
    SaveAs,
    ExportAs,
}

impl FileMode {
    fn title(self) -> &'static str {
        match self {
            FileMode::Open => tr("Open"),
            FileMode::SaveAs => tr("Save As"),
            FileMode::ExportAs => tr("Export As"),
        }
    }

    fn extensions(self) -> &'static [&'static str] {
        match self {
            FileMode::Open => &[
                "imagine", "psd", "psb", "png", "jpg", "jpeg", "webp", "tif", "tiff", "exr",
            ],
            FileMode::SaveAs => &["imagine"],
            FileMode::ExportAs => &["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        }
    }
}

struct Entry {
    name: String,
    dir: bool,
    size: u64,
}

pub struct FileDialog {
    pub mode: FileMode,
    dir: PathBuf,
    path_text: String,
    name: String,
    entries: Vec<Entry>,
    error: Option<String>,
    selected: Option<usize>,
    /// Export options (export mode).
    pub export: ExportOptions,
}

fn list(dir: &Path, mode: FileMode) -> Result<Vec<Entry>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let exts = mode.extensions();
    let mut v: Vec<Entry> = rd
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                return None;
            }
            let md = e.metadata().ok()?;
            let dir = md.is_dir();
            let ext_ok = Path::new(&name)
                .extension()
                .is_some_and(|x| exts.iter().any(|e| x.eq_ignore_ascii_case(e)));
            (dir || ext_ok).then_some(Entry {
                name,
                dir,
                size: md.len(),
            })
        })
        .collect();
    v.sort_by(|a, b| {
        b.dir
            .cmp(&a.dir)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(v)
}

fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / (1u64 << 20) as f64),
        b => format!("{} kB", b.div_ceil(1024)),
    }
}

impl FileDialog {
    /// `start`: folder to show; `name`: suggested file name (save/export).
    pub fn new(mode: FileMode, start: PathBuf, name: String) -> Self {
        let mut d = Self {
            mode,
            dir: PathBuf::new(),
            path_text: String::new(),
            name,
            entries: Vec::new(),
            error: None,
            selected: None,
            export: ExportOptions::default(),
        };
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| "/".into());
        d.go(if start.is_dir() { start } else { home });
        d
    }

    fn go(&mut self, dir: PathBuf) {
        match list(&dir, self.mode) {
            Ok(e) => {
                self.entries = e;
                self.dir = dir;
                self.error = None;
                self.selected = None;
            }
            Err(e) => self.error = Some(e),
        }
        self.path_text = self.dir.display().to_string();
    }

    fn finish(&self, out: &mut Vec<Action>) -> bool {
        let name = self.name.trim();
        if name.is_empty() {
            return false;
        }
        let mut path = if Path::new(name).is_absolute() {
            PathBuf::from(name)
        } else {
            self.dir.join(name)
        };
        match self.mode {
            FileMode::Open => out.push(Action::OpenFile(path)),
            FileMode::SaveAs => {
                if path
                    .extension()
                    .is_none_or(|e| !e.eq_ignore_ascii_case("imagine"))
                {
                    path.set_extension("imagine");
                }
                out.push(Action::SaveAs(path));
            }
            FileMode::ExportAs => {
                if path.extension().is_none() {
                    path.set_extension("png");
                }
                out.push(Action::ExportAs(path, self.export));
            }
        }
        true
    }

    fn export_options(&mut self, ui: &mut egui::Ui) {
        let name = Path::new(self.name.trim());
        let ext = name
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("png")
            .to_ascii_lowercase();
        let o = &mut self.export;
        ui.horizontal(|ui| {
            ui.label(tr("Color space"));
            egui::ComboBox::from_id_salt("export space")
                .selected_text(o.space.name())
                .show_ui(ui, |ui| {
                    for sp in ColorSpace::ALL {
                        ui.selectable_value(&mut o.space, sp, sp.name());
                    }
                });
            ui.add_space(12.0);
            let deep = imagine_io::export::supports_16bit(Path::new(&format!("x.{ext}")));
            ui.add_enabled_ui(deep, |ui| {
                let mut v = o.sixteen_bit && deep;
                if ui
                    .checkbox(&mut v, tr("16-bit"))
                    .on_disabled_hover_text(tr("PNG and TIFF only"))
                    .changed()
                {
                    o.sixteen_bit = v;
                }
            });
            if matches!(ext.as_str(), "jpg" | "jpeg") {
                ui.add_space(12.0);
                ui.label(tr("Quality"));
                ui.add(egui::Slider::new(&mut o.quality, 1..=100));
            }
        });
        if o.space == ColorSpace::Rec2020
            && !(o.sixteen_bit && imagine_io::export::supports_16bit(name))
        {
            ui.label(
                RichText::new(tr("Rec. 2020 at 8 bits may show banding in gradients."))
                    .weak()
                    .small(),
            );
        }
    }

    /// Shows the dialog; returns false once it is closed.
    pub fn show(&mut self, ctx: &egui::Context, recent: &[PathBuf], out: &mut Vec<Action>) -> bool {
        let mut open = true;
        let mut done = false;
        egui::Window::new(self.mode.title())
            .open(&mut open)
            .collapsible(false)
            .default_size([720.0, 480.0])
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("..").on_hover_text(tr("Parent folder")).clicked()
                        && let Some(p) = self.dir.parent()
                    {
                        self.go(p.to_path_buf());
                    }
                    if ui.button("~").on_hover_text(tr("Home folder")).clicked()
                        && let Some(h) = std::env::var_os("HOME")
                    {
                        self.go(h.into());
                    }
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.path_text)
                            .desired_width(f32::INFINITY),
                    );
                    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        self.go(PathBuf::from(self.path_text.trim()));
                    }
                });
                if let Some(e) = &self.error {
                    ui.colored_label(Color32::from_rgb(255, 120, 100), e);
                }
                ui.separator();
                let footer = if self.mode == FileMode::ExportAs {
                    96.0
                } else {
                    64.0
                };
                ui.horizontal_top(|ui| {
                    if self.mode == FileMode::Open && !recent.is_empty() {
                        ui.vertical(|ui| {
                            ui.set_width(200.0);
                            ui.label(RichText::new(tr("Recently opened")).strong());
                            egui::ScrollArea::vertical()
                                .id_salt("recent")
                                .max_height(ui.available_height() - footer)
                                .show(ui, |ui| {
                                    for p in recent {
                                        let name = p
                                            .file_name()
                                            .map(|n| n.to_string_lossy().into_owned())
                                            .unwrap_or_default();
                                        let r =
                                            ui.add(egui::Button::new(name).frame(false).truncate());
                                        if r.on_hover_text(p.display().to_string()).clicked() {
                                            out.push(Action::OpenFile(p.clone()));
                                            done = true;
                                        }
                                    }
                                });
                        });
                        ui.separator();
                    }
                    ui.vertical(|ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("files")
                            .max_height(ui.available_height() - footer)
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                let mut go_to = None;
                                for (i, e) in self.entries.iter().enumerate() {
                                    let label = if e.dir {
                                        format!("{}/", e.name)
                                    } else {
                                        e.name.clone()
                                    };
                                    let resp = ui
                                        .horizontal(|ui| {
                                            let r = ui
                                                .selectable_label(self.selected == Some(i), label);
                                            if !e.dir {
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        ui.label(
                                                            RichText::new(human(e.size)).weak(),
                                                        );
                                                    },
                                                );
                                            }
                                            r
                                        })
                                        .inner;
                                    if resp.clicked() {
                                        self.selected = Some(i);
                                        if !e.dir {
                                            self.name = e.name.clone();
                                        }
                                    }
                                    if resp.double_clicked() {
                                        if e.dir {
                                            go_to = Some(self.dir.join(&e.name));
                                        } else {
                                            self.name = e.name.clone();
                                            done = self.finish(out);
                                        }
                                    }
                                }
                                if let Some(d) = go_to {
                                    self.go(d);
                                }
                                if self.entries.is_empty() {
                                    ui.label(RichText::new(tr("(no matching files)")).weak());
                                }
                            });
                    });
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(tr("File"));
                    let r = ui.add(
                        egui::TextEdit::singleline(&mut self.name)
                            .desired_width(ui.available_width() - 180.0),
                    );
                    let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button(self.mode.title()).clicked() || enter {
                        done = self.finish(out);
                    }
                });
                if self.mode == FileMode::ExportAs {
                    self.export_options(ui);
                }
                ui.label(
                    RichText::new(self.mode.extensions().join(", "))
                        .weak()
                        .small(),
                );
            });
        open && !done
    }
}

/// "New document" dialog state.
pub struct NewDialog {
    w: u32,
    h: u32,
    white: bool,
}

impl Default for NewDialog {
    fn default() -> Self {
        Self {
            w: 3000,
            h: 2000,
            white: true,
        }
    }
}

const PRESETS: [(&str, u32, u32); 6] = [
    ("Full HD", 1920, 1080),
    ("4K", 3840, 2160),
    ("A4, 300 dpi", 2480, 3508),
    ("Square 2048", 2048, 2048),
    ("Photo 6000×4000", 6000, 4000),
    ("Huge 20000²", 20000, 20000),
];

impl NewDialog {
    pub fn show(&mut self, ctx: &egui::Context, out: &mut Vec<Action>) -> bool {
        let mut open = true;
        let mut done = false;
        egui::Window::new(tr("New Image"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (name, w, h) in PRESETS {
                        if ui
                            .selectable_label(self.w == w && self.h == h, tr(name))
                            .clicked()
                        {
                            (self.w, self.h) = (w, h);
                        }
                    }
                });
                egui::Grid::new("new size").show(ui, |ui| {
                    ui.label(tr("Width"));
                    ui.add(
                        egui::DragValue::new(&mut self.w)
                            .range(1..=300_000)
                            .suffix(" px"),
                    );
                    ui.end_row();
                    ui.label(tr("Height"));
                    ui.add(
                        egui::DragValue::new(&mut self.h)
                            .range(1..=300_000)
                            .suffix(" px"),
                    );
                    ui.end_row();
                    ui.label(tr("Background"));
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.white, true, tr("White"));
                        ui.radio_value(&mut self.white, false, tr("Transparent"));
                    });
                    ui.end_row();
                });
                ui.label(
                    RichText::new(tr!(
                        "{mp} megapixels",
                        mp = format!("{:.1}", self.w as f64 * self.h as f64 / 1e6)
                    ))
                    .weak(),
                );
                if ui.button(tr("Create")).clicked()
                    || ui.input(|i| i.key_pressed(egui::Key::Enter))
                {
                    out.push(Action::NewDocument {
                        w: self.w,
                        h: self.h,
                        white: self.white,
                    });
                    done = true;
                }
            });
        open && !done
    }
}

/// What to do after the user answered the unsaved-changes question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answer {
    Save,
    Discard,
    Cancel,
}

/// Modal question; returns the answer once given.
pub fn confirm_unsaved(ctx: &egui::Context, name: &str, what: &str) -> Option<Answer> {
    let mut answer = None;
    egui::Modal::new(egui::Id::unique("unsaved")).show(ctx, |ui| {
        ui.set_width(380.0);
        ui.heading(tr("Unsaved Changes"));
        ui.label(tr!(
            "“{name}” has unsaved changes. {what}",
            name = name,
            what = what
        ));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(tr("Save")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                answer = Some(Answer::Save);
            }
            if ui.button(tr("Discard")).clicked() {
                answer = Some(Answer::Discard);
            }
            if ui.button(tr("Cancel")).clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                answer = Some(Answer::Cancel);
            }
        });
    });
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listing_filters_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        for f in ["b.png", "A.JPG", "notes.txt", ".hidden.png", "x.imagine"] {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        std::fs::create_dir(dir.path().join("zdir")).unwrap();
        let names = |m| {
            list(dir.path(), m)
                .unwrap()
                .into_iter()
                .map(|e| e.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            names(FileMode::Open),
            vec!["zdir", "A.JPG", "b.png", "x.imagine"]
        );
        assert_eq!(names(FileMode::SaveAs), vec!["zdir", "x.imagine"]);
    }

    #[test]
    fn save_as_adds_the_extension() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = FileDialog::new(FileMode::SaveAs, dir.path().to_path_buf(), "bild".into());
        let mut out = Vec::new();
        assert!(d.finish(&mut out));
        assert!(matches!(&out[0], Action::SaveAs(p) if p.ends_with("bild.imagine")));
        d.mode = FileMode::ExportAs;
        d.name = "x".into();
        d.finish(&mut out);
        assert!(matches!(&out[1], Action::ExportAs(p, _) if p.ends_with("x.png")));
    }
}
