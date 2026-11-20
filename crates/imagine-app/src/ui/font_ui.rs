//! The font window: search, filter by kind, favourites and recently used
//! fonts, every family previewed with the layer's own text. Browsing shows
//! the font live on the canvas (`preview`); Enter / double click applies,
//! Escape restores.

use std::collections::HashMap;

use egui::{Color32, RichText, Vec2, pos2, vec2};
use imagine_core::VectorLayer;
use imagine_core::vector::{TextSpec, VectorKind};
use imagine_vector::{FontFace, FontGroup, FontKind};

const ROW_H: f32 = 44.0;
const PREVIEW_PX: f32 = 26.0;

#[derive(Default)]
pub struct FontWindow {
    pub open: bool,
    query: String,
    focus_query: bool,
    /// Highlighted family (by name).
    sel: Option<String>,
    scroll_to_sel: bool,
    kind: Option<FontKind>,
    favourites_only: bool,
    all_scripts: bool,
    /// The family being tried out (shown live on the canvas).
    trying: Option<String>,
    /// What the canvas should show while browsing (read by the app); always
    /// built from the current layer, so other edits stay visible.
    pub preview: Option<VectorLayer>,
    /// Rendered previews: (family, text) → texture.
    previews: HashMap<(String, String), Option<egui::TextureHandle>>,
}

/// What happened in the window.
#[derive(Default)]
pub struct FontEvents {
    /// A face to apply (window closed).
    pub chosen: Option<FontFace>,
    /// A family was used (for the recent list).
    pub used: Option<String>,
    pub toggle_favourite: Option<String>,
}

impl FontWindow {
    /// Opens the window for a text layer.
    pub fn open_for(&mut self, layer: &VectorLayer, fonts: &[FontGroup], query: &str) {
        self.open = true;
        self.query = query.into();
        self.focus_query = true;
        self.trying = None;
        self.preview = None;
        if let VectorKind::Text(t) = &layer.kind {
            self.sel = fonts.iter().find(|g| g.faces.iter().any(|f| f.family == t.font)).map(|g| g.name.clone());
            self.scroll_to_sel = true;
        }
    }

    /// Tries a family as if it had been clicked (snapshots).
    pub fn try_family(&mut self, name: &str) {
        self.sel = Some(name.into());
        self.trying = Some(name.into());
    }

    fn close(&mut self) {
        self.open = false;
        self.preview = None;
        self.trying = None;
    }
}

fn generic(name: &str) -> FontGroup {
    FontGroup {
        name: name.into(),
        kind: FontKind::Sans,
        latin: true,
        faces: vec![FontFace { family: name.into(), weight: 400.0, italic: false, label: "Regular".into() }],
    }
}

/// The layer with `group` applied (keeping weight and slope as far as the family has them).
fn with_family(layer: &VectorLayer, group: &FontGroup) -> (VectorLayer, Option<FontFace>) {
    let mut v = layer.clone();
    let mut face = None;
    if let VectorKind::Text(t) = &mut v.kind {
        face = group.face("", t.weight, t.italic).or(group.regular()).cloned();
        if let Some(f) = &face {
            t.font = f.family.clone();
            t.weight = f.weight;
            t.italic = f.italic;
        }
    }
    (v, face)
}

/// A five-pointed star (filled = favourite), drawn from convex pieces.
fn star(p: &egui::Painter, c: egui::Pos2, r: f32, filled: bool, color: Color32) {
    let pt = |i: usize, rr: f32| {
        let a = -std::f32::consts::FRAC_PI_2 + i as f32 * std::f32::consts::PI / 5.0;
        c + vec2(a.cos(), a.sin()) * rr
    };
    let outer: Vec<egui::Pos2> = (0..10).map(|i| pt(i, if i % 2 == 0 { r } else { r * 0.45 })).collect();
    if filled {
        let inner: Vec<egui::Pos2> = (0..5).map(|i| outer[i * 2 + 1]).collect();
        p.add(egui::Shape::convex_polygon(inner.clone(), color, egui::Stroke::NONE));
        for i in 0..5 {
            p.add(egui::Shape::convex_polygon(vec![outer[i * 2], outer[(i * 2 + 1) % 10], outer[(i * 2 + 9) % 10]], color, egui::Stroke::NONE));
        }
    } else {
        let mut closed = outer.clone();
        closed.push(outer[0]);
        p.add(egui::Shape::line(closed, egui::Stroke::new(1.2, color)));
    }
}

/// Shows the window (if open) for the text layer `layer`.
pub fn show(ctx: &egui::Context, w: &mut FontWindow, layer: &VectorLayer, fonts: &[FontGroup], favourites: &[String], recent: &[String]) -> FontEvents {
    let mut ev = FontEvents::default();
    if !w.open {
        return ev;
    }
    let VectorKind::Text(spec) = &layer.kind else {
        w.close();
        return ev;
    };
    let base = layer.clone();
    let sample = preview_text(spec);
    let mut open = true;
    let mut apply = false;
    let mut cancel = false;
    egui::Window::new("Schrift")
        .id(egui::Id::unique("font window"))
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_size([560.0, 620.0])
        .default_pos(ctx.content_rect().right_top() + vec2(-940.0, 60.0))
        .show(ctx, |ui| {
            let q = ui.add(egui::TextEdit::singleline(&mut w.query).hint_text("Schrift suchen …").desired_width(f32::INFINITY));
            if std::mem::take(&mut w.focus_query) {
                q.request_focus();
            }
            ui.horizontal_wrapped(|ui| {
                if ui.selectable_label(w.kind.is_none() && !w.favourites_only, "Alle").clicked() {
                    w.kind = None;
                    w.favourites_only = false;
                }
                if ui.selectable_label(w.favourites_only, "Favoriten").clicked() {
                    w.favourites_only = !w.favourites_only;
                }
                for k in FontKind::ALL {
                    if ui.selectable_label(w.kind == Some(k), k.name()).clicked() {
                        w.kind = if w.kind == Some(k) { None } else { Some(k) };
                    }
                }
                ui.checkbox(&mut w.all_scripts, "andere Schriftsysteme");
            });
            ui.separator();
            // The list: recent first (without a query or filter), then all matches.
            let query = w.query.to_lowercase();
            let filtered = |g: &&FontGroup| {
                (query.is_empty() || g.name.to_lowercase().contains(&query))
                    && w.kind.is_none_or(|k| g.kind == k)
                    && (!w.favourites_only || favourites.contains(&g.name))
                    && (w.all_scripts || g.latin || w.kind == Some(FontKind::Symbol))
                    && (w.kind == Some(FontKind::Symbol) || g.kind != FontKind::Symbol || !query.is_empty())
            };
            let mut list: Vec<(FontGroup, bool)> = Vec::new();
            if query.is_empty() && w.kind.is_none() && !w.favourites_only {
                list.extend(recent.iter().filter_map(|r| fonts.iter().find(|g| &g.name == r)).map(|g| (g.clone(), true)));
                list.extend(["sans-serif", "serif", "monospace"].iter().map(|n| (generic(n), false)));
            }
            list.extend(fonts.iter().filter(filtered).map(|g| (g.clone(), false)));
            // Keyboard: arrows move (and preview), Enter applies, Escape cancels.
            let (up, down, enter, esc) = ui.input(|i| {
                (i.key_pressed(egui::Key::ArrowUp), i.key_pressed(egui::Key::ArrowDown), i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Escape))
            });
            let pos = w.sel.as_ref().and_then(|s| list.iter().position(|(g, _)| &g.name == s));
            if (up || down) && !list.is_empty() {
                let i = match (pos, down) {
                    (None, _) => 0,
                    (Some(i), true) => (i + 1).min(list.len() - 1),
                    (Some(i), false) => i.saturating_sub(1),
                };
                w.sel = Some(list[i].0.name.clone());
                w.scroll_to_sel = true;
                w.trying = w.sel.clone();
            }
            if enter && w.sel.is_some() {
                apply = true;
            }
            if esc {
                cancel = true;
            }
            let mut rendered = 0;
            let footer = 34.0;
            egui::ScrollArea::vertical().id_salt("fonts").max_height(ui.available_height() - footer).auto_shrink([false, false]).show_rows(
                ui,
                ROW_H,
                list.len(),
                |ui, rows| {
                    for i in rows {
                        let (group, is_recent) = &list[i];
                        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_H), egui::Sense::click());
                        let p = ui.painter_at(rect);
                        let selected = w.sel.as_ref() == Some(&group.name);
                        if selected {
                            p.rect_filled(rect, 4.0, Color32::from_rgb(55, 85, 135));
                            if std::mem::take(&mut w.scroll_to_sel) {
                                ui.scroll_to_rect(rect, Some(egui::Align::Center));
                            }
                        } else if resp.hovered() {
                            p.rect_filled(rect, 4.0, Color32::from_white_alpha(10));
                        }
                        // Star: favourite on/off.
                        let star_rect = egui::Rect::from_center_size(pos2(rect.min.x + 16.0, rect.center().y), Vec2::splat(20.0));
                        let star_resp = ui.interact(star_rect, resp.id.with("star"), egui::Sense::click());
                        let fav = favourites.contains(&group.name);
                        let col = if fav { Color32::from_rgb(250, 200, 70) } else if star_resp.hovered() { Color32::from_gray(200) } else { Color32::from_gray(110) };
                        star(&p, star_rect.center(), 7.5, fav, col);
                        if star_resp.on_hover_text("Favorit").clicked() {
                            ev.toggle_favourite = Some(group.name.clone());
                        }
                        // Name and details (top), preview (bottom).
                        let text_col = if selected { Color32::WHITE } else { Color32::from_gray(205) };
                        let detail = format!(
                            "{}{}{}",
                            group.kind.name(),
                            if group.faces.len() > 1 { format!(" · {} Schnitte", group.faces.len()) } else { String::new() },
                            if *is_recent { " · zuletzt" } else { "" }
                        );
                        p.text(pos2(rect.min.x + 32.0, rect.min.y + 4.0), egui::Align2::LEFT_TOP, &group.name, egui::FontId::proportional(12.5), text_col);
                        p.text(pos2(rect.max.x - 8.0, rect.min.y + 4.0), egui::Align2::RIGHT_TOP, detail, egui::FontId::proportional(11.0), Color32::from_gray(130));
                        let key = (group.name.clone(), sample.clone());
                        if !w.previews.contains_key(&key) && rendered < 10 {
                            rendered += 1;
                            let tex = group.regular().and_then(|f| imagine_vector::font_preview(&f.family, f.weight, f.italic, &sample, PREVIEW_PX, 900)).map(|(pw, ph, px)| {
                                let img = egui::ColorImage::from_rgba_unmultiplied([pw as usize, ph as usize], &px);
                                ui.ctx().load_texture(format!("font:{}:{}", group.name, sample), img, egui::TextureOptions::LINEAR)
                            });
                            w.previews.insert(key.clone(), tex);
                        }
                        match w.previews.get(&key) {
                            Some(Some(tex)) => {
                                let size = tex.size_vec2();
                                let avail = vec2(rect.width() - 40.0, ROW_H - 20.0);
                                let s = size * (avail.y / size.y).min(avail.x / size.x).min(1.0);
                                let r = egui::Rect::from_min_size(pos2(rect.min.x + 32.0, rect.max.y - s.y - 3.0), s);
                                p.image(tex.id(), r, egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), text_col);
                            }
                            Some(None) => {
                                p.text(pos2(rect.min.x + 32.0, rect.max.y - 6.0), egui::Align2::LEFT_BOTTOM, "(keine Zeichen für diesen Text)", egui::FontId::proportional(11.0), Color32::from_gray(110));
                            }
                            None => ui.ctx().request_repaint(),
                        }
                        if resp.clicked() {
                            w.sel = Some(group.name.clone());
                            w.trying = w.sel.clone();
                        }
                        if resp.double_clicked() {
                            w.sel = Some(group.name.clone());
                            apply = true;
                        }
                    }
                },
            );
            if list.is_empty() {
                ui.label(RichText::new("keine passende Schrift").weak());
            }
            ui.horizontal(|ui| {
                if ui.button("Übernehmen").clicked() {
                    apply = true;
                }
                if ui.button("Abbrechen").clicked() {
                    cancel = true;
                }
                ui.label(RichText::new("Pfeiltasten: blättern mit Vorschau im Bild · Enter: übernehmen · Esc: zurück").weak().small());
            });
            if apply && let Some(g) = w.sel.as_ref().and_then(|s| list.iter().find(|(g, _)| &g.name == s)) {
                ev.chosen = with_family(&base, &g.0).1;
                ev.used = Some(g.0.name.clone()).filter(|n| fonts.iter().any(|f| &f.name == n));
            }
        });
    if apply || cancel || !open {
        w.close();
    } else {
        // The live preview: the current layer with the family being tried.
        w.preview = w.trying.as_ref().and_then(|n| {
            let g = fonts.iter().find(|g| &g.name == n).cloned().or_else(|| ["sans-serif", "serif", "monospace"].contains(&n.as_str()).then(|| generic(n)))?;
            Some(with_family(&base, &g).0)
        });
    }
    // Keep the cache bounded (previews of an old sample text).
    if w.previews.len() > 600 {
        w.previews.retain(|(_, t), _| *t == sample);
    }
    ev
}

/// The first line of the text (shortened), or a sample.
fn preview_text(t: &TextSpec) -> String {
    let first = t.text.lines().next().unwrap_or("").trim();
    if first.is_empty() {
        return "Imagine 123".into();
    }
    let mut s: String = first.chars().take(28).collect();
    if first.chars().count() > 28 {
        s.push('…');
    }
    s
}
