//! Editing text directly on the canvas: caret, selection with mouse and
//! keyboard, typing. Changes go to the layer at once (one undo step per
//! editing session); the caret and selection are drawn by the UI.

use std::time::Instant;

use glam::{DAffine2, DVec2};
use imagine_core::vector::{TextSpec, VectorKind};
use imagine_core::{LayerId, tr};
use imagine_vector::{ClickKind, TextMove, TextSelection};
use smithay_client_toolkit::seat::keyboard::{KeyEvent, Keysym};

use super::{App, Drag};
use crate::ui::panels::TextOverlay;

pub(super) struct TextEdit {
    pub layer: LayerId,
    pub sel: TextSelection,
    /// For double / triple clicks: time, position (screen) and count.
    last_click: Option<(Instant, DVec2, u8)>,
}

/// The selection, or a caret at the end if the text became shorter (undo).
fn valid(t: &TextSpec, sel: &TextSelection) -> TextSelection {
    let r = sel.text_range();
    let n = t.text.len();
    if r.end > n
        || r.start > n
        || !t.text.is_char_boundary(r.start)
        || !t.text.is_char_boundary(r.end)
    {
        imagine_vector::text_range(t, n, n)
    } else {
        *sel
    }
}

fn affine(m: &[f64; 6]) -> DAffine2 {
    DAffine2::from_cols_array(m)
}

impl App {
    /// The text being edited and the layer transform.
    fn edit_spec(&mut self) -> Option<(TextSpec, DAffine2)> {
        let id = self.text_edit.as_ref()?.layer;
        let v = self
            .session
            .as_ref()?
            .engine
            .doc()
            .find(id)?
            .vector()?
            .clone();
        match v.kind {
            VectorKind::Text(t) => Some((t, affine(&v.transform))),
            _ => None,
        }
    }

    /// Document point → text-box coordinates of `t`.
    fn to_box(t: &TextSpec, m: DAffine2, doc: DVec2) -> (f32, f32) {
        let p = m.inverse().transform_point2(doc);
        ((p.x - t.x) as f32, (p.y - t.y) as f32)
    }

    /// Starts editing a text layer; `at` places the caret (else all is selected).
    pub(super) fn begin_text_edit(&mut self, layer: LayerId, at: Option<DVec2>) {
        self.end_text_edit();
        let Some(v) = self
            .session
            .as_ref()
            .and_then(|s| s.engine.doc().find(layer))
            .and_then(|l| l.vector().cloned())
        else {
            return;
        };
        let VectorKind::Text(t) = &v.kind else { return };
        let sel = match at {
            Some(p) => {
                let (x, y) = Self::to_box(t, affine(&v.transform), p);
                imagine_vector::text_select_at(t, x, y, ClickKind::Caret, None)
            }
            None => imagine_vector::text_range(t, 0, t.text.len()),
        };
        if let Some(e) = self.engine() {
            e.select(|d| d.active = layer);
        }
        self.text_edit = Some(TextEdit {
            layer,
            sel,
            last_click: None,
        });
        self.needs_redraw = true;
    }

    /// Ends editing (an emptied text layer is removed, like in Photoshop).
    pub(super) fn end_text_edit(&mut self) {
        let Some(te) = self.text_edit.take() else {
            return;
        };
        if let Some(e) = self.engine() {
            e.end_merge();
            let empty = e.doc().find(te.layer).and_then(|l| l.vector()).is_some_and(
                |v| matches!(&v.kind, VectorKind::Text(t) if t.text.trim().is_empty()),
            );
            if empty {
                e.edit(tr("Delete Empty Text"), |d| {
                    d.active = te.layer;
                    d.remove_active()
                });
            }
        }
        self.needs_redraw = true;
    }

    /// Mouse down with the text tool while editing: caret, word, line, or
    /// Shift extends. False if the click is outside the edited text.
    pub(super) fn text_edit_press(&mut self, screen: DVec2, doc: DVec2) -> bool {
        let Some((t, m)) = self.edit_spec() else {
            return false;
        };
        let v_bounds = self
            .session
            .as_ref()
            .and_then(|s| s.engine.doc().find(self.text_edit.as_ref().unwrap().layer))
            .and_then(|l| l.vector())
            .and_then(imagine_vector::bounds);
        if v_bounds.is_none_or(|b| doc.x < b[0] || doc.x > b[2] || doc.y < b[1] || doc.y > b[3]) {
            return false;
        }
        let (x, y) = Self::to_box(&t, m, doc);
        let shift = self.mods.shift;
        let te = self.text_edit.as_mut().unwrap();
        let count = match te.last_click {
            Some((when, p, n)) if when.elapsed().as_millis() < 450 && p.distance(screen) < 5.0 => {
                n % 3 + 1
            }
            _ => 1,
        };
        te.last_click = Some((Instant::now(), screen, count));
        te.sel = if shift {
            imagine_vector::text_select_at(&t, x, y, ClickKind::Caret, Some(&valid(&t, &te.sel)))
        } else {
            let kind = match count {
                2 => ClickKind::Word,
                3 => ClickKind::Line,
                _ => ClickKind::Caret,
            };
            imagine_vector::text_select_at(&t, x, y, kind, None)
        };
        self.drag = Drag::TextSelect;
        self.needs_redraw = true;
        true
    }

    /// Dragging extends the selection.
    pub(super) fn text_edit_drag(&mut self, doc: DVec2) {
        let Some((t, m)) = self.edit_spec() else {
            return;
        };
        let (x, y) = Self::to_box(&t, m, doc);
        if let Some(te) = &mut self.text_edit {
            te.sel = imagine_vector::text_select_at(
                &t,
                x,
                y,
                ClickKind::Caret,
                Some(&valid(&t, &te.sel)),
            );
        }
        self.needs_redraw = true;
    }

    /// Replaces the selection with `s` (typing, deleting with "").
    fn text_replace(&mut self, range: std::ops::Range<usize>, s: &str) {
        let Some(te) = &self.text_edit else { return };
        let id = te.layer;
        let Some(mut v) = self
            .session
            .as_ref()
            .and_then(|x| x.engine.doc().find(id))
            .and_then(|l| l.vector().cloned())
        else {
            return;
        };
        let VectorKind::Text(t) = &mut v.kind else {
            return;
        };
        let (a, b) = (range.start.min(t.text.len()), range.end.min(t.text.len()));
        t.text.replace_range(a..b, s);
        let caret = a + s.len();
        let spec = t.clone();
        if let Some(e) = self.engine() {
            let _ = e.set_vector(id, v, tr("Typing"), Some(&format!("typing-{id}")));
        }
        if let Some(te) = &mut self.text_edit {
            te.sel = imagine_vector::text_range(&spec, caret, caret);
        }
        self.touch();
    }

    /// Keyboard input while editing text. True if the key was used.
    pub(super) fn text_edit_key(&mut self, ev: &KeyEvent) -> bool {
        let Some((t, _)) = self.edit_spec() else {
            self.text_edit = None;
            return false;
        };
        let (ctrl, shift) = (self.mods.ctrl, self.mods.shift);
        let sel = valid(&t, &self.text_edit.as_ref().unwrap().sel);
        let range = sel.text_range();
        let mv = |m: TextMove| imagine_vector::text_move(&t, &sel, m, shift);
        let moved = match ev.keysym {
            Keysym::Left => Some(mv(if ctrl {
                TextMove::WordLeft
            } else {
                TextMove::Left
            })),
            Keysym::Right => Some(mv(if ctrl {
                TextMove::WordRight
            } else {
                TextMove::Right
            })),
            Keysym::Up => Some(mv(TextMove::Up)),
            Keysym::Down => Some(mv(TextMove::Down)),
            Keysym::Home => Some(mv(if ctrl {
                TextMove::Start
            } else {
                TextMove::LineStart
            })),
            Keysym::End => Some(mv(if ctrl {
                TextMove::End
            } else {
                TextMove::LineEnd
            })),
            Keysym::a | Keysym::A if ctrl => Some(imagine_vector::text_range(&t, 0, t.text.len())),
            _ => None,
        };
        if let Some(s) = moved {
            self.text_edit.as_mut().unwrap().sel = s;
            self.needs_redraw = true;
            return true;
        }
        match ev.keysym {
            // Leave editing (the text stays).
            Keysym::Escape => self.end_text_edit(),
            Keysym::Return | Keysym::KP_Enter if ctrl => self.end_text_edit(),
            Keysym::Return | Keysym::KP_Enter => self.text_replace(range, "\n"),
            Keysym::BackSpace => {
                let r = if !range.is_empty() {
                    range
                } else {
                    let i = range.start;
                    let before = if ctrl {
                        imagine_vector::text_move(&t, &sel, TextMove::WordLeft, false)
                            .focus()
                            .index()
                    } else {
                        t.text[..i].char_indices().next_back().map_or(0, |(j, _)| j)
                    };
                    before..i
                };
                self.text_replace(r, "");
            }
            Keysym::Delete => {
                let r = if !range.is_empty() {
                    range
                } else {
                    let i = range.start;
                    let after = if ctrl {
                        imagine_vector::text_move(&t, &sel, TextMove::WordRight, false)
                            .focus()
                            .index()
                    } else {
                        t.text[i..].chars().next().map_or(i, |c| i + c.len_utf8())
                    };
                    i..after
                };
                self.text_replace(r, "");
            }
            Keysym::c | Keysym::C if ctrl => self.copy(false, false),
            Keysym::x | Keysym::X if ctrl => self.copy(true, false),
            Keysym::v | Keysym::V if ctrl => self.paste(super::clipboard::PasteTarget::Text),
            // Other shortcuts with Ctrl/Alt (undo, save …) work as usual.
            _ if ctrl || self.mods.alt => return false,
            _ => match ev
                .utf8
                .as_deref()
                .filter(|s| !s.is_empty() && s.chars().all(|c| !c.is_control() || c == '\t'))
            {
                Some(s) => self.text_replace(range, s),
                None => return false,
            },
        }
        true
    }

    /// The selected text while editing (`None` when not editing).
    pub(super) fn text_edit_selected(&mut self) -> Option<String> {
        let (t, _) = self.edit_spec()?;
        let r = valid(&t, &self.text_edit.as_ref()?.sel).text_range();
        Some(t.text[r].to_string())
    }

    pub(super) fn text_edit_delete_selection(&mut self) {
        if let Some((t, _)) = self.edit_spec() {
            let r = valid(&t, &self.text_edit.as_ref().unwrap().sel).text_range();
            self.text_replace(r, "");
        }
    }

    /// Inserts text at the caret (replacing the selection).
    pub(super) fn text_edit_insert(&mut self, s: &str) {
        if let Some((t, _)) = self.edit_spec() {
            let r = valid(&t, &self.text_edit.as_ref().unwrap().sel).text_range();
            self.text_replace(r, s);
        }
    }

    /// Caret, selection and box of the edited text in screen points (for the UI).
    pub(super) fn text_overlay(&self) -> Option<TextOverlay> {
        let te = self.text_edit.as_ref()?;
        let v = self
            .session
            .as_ref()?
            .engine
            .doc()
            .find(te.layer)?
            .vector()?;
        overlay(v, &te.sel, &self.view, self.scale)
    }
}

/// Caret, selection and frame of a text layer in screen points.
pub fn overlay(
    v: &imagine_core::VectorLayer,
    sel: &TextSelection,
    view: &imagine_core::View,
    scale: f64,
) -> Option<TextOverlay> {
    let VectorKind::Text(t) = &v.kind else {
        return None;
    };
    let m = affine(&v.transform);
    let (rects, caret) = imagine_vector::text_geometry(t, &valid(t, sel));
    let k = 1.0 / scale as f32;
    let quad = |r: [f32; 4]| -> [[f32; 2]; 4] {
        [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]].map(|[x, y]| {
            let d = m.transform_point2(DVec2::new(t.x + x as f64, t.y + y as f64));
            let s = view.doc_to_screen(d);
            [s.x as f32 * k, s.y as f32 * k]
        })
    };
    let (w, h) = imagine_vector::text_box_size(t);
    Some(TextOverlay {
        selection: rects.into_iter().map(quad).collect(),
        caret: quad(caret),
        frame: quad([-4.0, -2.0, w + 4.0, h + 2.0]),
    })
}
