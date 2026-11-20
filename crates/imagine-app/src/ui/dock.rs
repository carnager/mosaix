//! The dockable workspace: the canvas is a fixed tab, the panels
//! (properties, styles, colour, history, layers) can be dragged into other
//! places, stacked as tabs, split, detached as floating windows and closed
//! (Window menu brings them back). The layout is kept in the settings.

use egui::{Id, Ui, WidgetText};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use imagine_core::tr;
use serde::{Deserialize, Serialize};

use super::panels::{Action, Model, State};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pane {
    Canvas,
    Properties,
    Styles,
    Color,
    History,
    Layers,
}

impl Pane {
    /// The panels (everything but the canvas), in menu order.
    pub const PANELS: [Pane; 5] = [
        Pane::Properties,
        Pane::Styles,
        Pane::Color,
        Pane::History,
        Pane::Layers,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Pane::Canvas => tr("Canvas"),
            Pane::Properties => tr("Properties"),
            Pane::Styles => tr("Styles"),
            Pane::Color => tr("Color"),
            Pane::History => tr("History"),
            Pane::Layers => tr("Layers"),
        }
    }
}

/// Canvas on the left, a column on the right: tabs above, layers below.
pub fn default_layout() -> DockState<Pane> {
    let mut d = DockState::new(vec![Pane::Canvas]);
    let s = d.main_surface_mut();
    let [_, right] = s.split_right(
        NodeIndex::root(),
        0.79,
        vec![Pane::Properties, Pane::Styles, Pane::Color, Pane::History],
    );
    s.split_below(right, 0.55, vec![Pane::Layers]);
    d
}

/// A saved layout, if it is valid (has exactly one canvas).
pub fn layout_from_json(v: &serde_json::Value) -> Option<DockState<Pane>> {
    // Areas not laid out yet have NaN coordinates, which JSON stores as null.
    fn fix(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, x) in m.iter_mut() {
                    if (k == "x" || k == "y") && x.is_null() {
                        *x = 0.0.into();
                    } else {
                        fix(x);
                    }
                }
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(fix),
            _ => {}
        }
    }
    let mut v = v.clone();
    fix(&mut v);
    let d: DockState<Pane> = serde_json::from_value(v).ok()?;
    (d.iter_all_tabs()
        .filter(|(_, t)| **t == Pane::Canvas)
        .count()
        == 1)
        .then_some(d)
}

fn translations() -> egui_dock::Translations {
    let mut t = egui_dock::Translations::english();
    t.tab_context_menu.close_button = tr("Close").into();
    t.tab_context_menu.eject_button = tr("Detach").into();
    t.tab_context_menu.hide_tab_bar_button = tr("Hide tab bar").into();
    t.tab_context_menu.show_tab_bar_button = tr("Show tab bar").into();
    t.leaf.close_button_disabled_tooltip = tr("The canvas cannot be closed").into();
    t.leaf.close_all_button = tr("Close all").into();
    t
}

struct Viewer<'a> {
    m: &'a Model,
    st: &'a mut State,
    out: &'a mut Vec<Action>,
}

impl TabViewer for Viewer<'_> {
    type Tab = Pane;

    fn id(&mut self, tab: &mut Pane) -> Id {
        Id::unique(("pane", *tab))
    }

    fn title(&mut self, tab: &mut Pane) -> WidgetText {
        match tab {
            Pane::Canvas => {
                format!("{}{}", if self.m.modified { "• " } else { "" }, self.m.name).into()
            }
            Pane::Styles => match self.m.active_effects.as_ref().map_or(0, Vec::len) {
                0 => tr("Styles").into(),
                n => imagine_core::tr!("Styles ({n})", n = n).into(),
            },
            p => p.name().into(),
        }
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Pane) {
        let (m, st, out) = (self.m, &mut *self.st, &mut *self.out);
        match tab {
            Pane::Canvas => {
                let r = ui.max_rect();
                st.canvas = Some(r);
                ui.allocate_rect(r, egui::Sense::hover());
            }
            Pane::Properties => super::panels::properties_tab(ui, m, st, out),
            Pane::Styles => super::panels::styles_tab(ui, m, st, out),
            Pane::Color => super::panels::color_section(ui, m, st, out),
            Pane::History => super::panels::history_section(ui, m, out),
            Pane::Layers => super::panels::layer_section(ui, m, st, out),
        }
    }

    fn is_closeable(&self, tab: &Pane) -> bool {
        *tab != Pane::Canvas
    }

    fn allowed_in_windows(&self, tab: &mut Pane) -> bool {
        *tab != Pane::Canvas
    }

    fn clear_background(&self, tab: &Pane) -> bool {
        // The GPU canvas is drawn under the interface.
        *tab != Pane::Canvas
    }

    fn scroll_bars(&self, tab: &Pane) -> [bool; 2] {
        match tab {
            // The layer list scrolls by itself (buttons stay at the bottom).
            Pane::Canvas | Pane::Layers => [false, false],
            _ => [false, true],
        }
    }
}

/// Shows the dock in the remaining area.
pub fn show(ui: &mut Ui, m: &Model, st: &mut State, out: &mut Vec<Action>) {
    let mut dock = st.dock.take().unwrap_or_else(default_layout);
    dock.translations = translations();
    if let Some(p) = st.focus_pane.take() {
        focus(&mut dock, p);
    }
    let mut style = Style::from_egui(ui.style());
    style.tab_bar.bg_fill = egui::Color32::from_rgb(33, 35, 39);
    style.tab_bar.height = 26.0;
    style.separator.width = 2.0;
    style.separator.color_idle = egui::Color32::from_rgb(28, 29, 33);
    style.main_surface_border_stroke = egui::Stroke::NONE;
    {
        let mut viewer = Viewer { m, st, out };
        DockArea::new(&mut dock)
            .id(Id::unique("dock"))
            .style(style)
            .show_close_buttons(false)
            .show_leaf_collapse_buttons(false)
            .show_leaf_close_all_buttons(false)
            .show_inside(ui, &mut viewer);
    }
    // The canvas may not be removed in any way (e.g. a broken saved layout).
    if dock.find_tab(&Pane::Canvas).is_none() {
        dock = default_layout();
    }
    // Remember a changed layout once the mouse is released.
    if !ui.ctx().input(|i| i.pointer.any_down())
        && let Ok(v) = serde_json::to_value(&dock)
        && st.saved_layout.as_ref() != Some(&v)
    {
        st.saved_layout = Some(v.clone());
        st.layout_changed = Some(v);
    }
    st.dock = Some(dock);
}

/// Makes `p` the visible tab, reopening it next to the properties (or at the
/// right edge) if it was closed.
pub fn focus(dock: &mut DockState<Pane>, p: Pane) {
    if dock.find_tab(&p).is_none() {
        let anchor = Pane::PANELS
            .iter()
            .filter(|q| **q != p)
            .find_map(|q| dock.find_tab(q));
        match anchor {
            Some(path) => {
                if let Ok(leaf) = dock.leaf_mut(path.node_path()) {
                    leaf.append_tab(p);
                }
            }
            None => {
                let s = dock.main_surface_mut();
                s.split_right(NodeIndex::root(), 0.79, vec![p]);
            }
        }
    }
    if let Some(path) = dock.find_tab(&p) {
        let _ = dock.set_active_tab(path);
    }
}

/// Window menu: show or hide each panel, reset the layout.
pub fn window_menu(ui: &mut Ui, st: &mut State) {
    let dock = st.dock.get_or_insert_with(default_layout);
    for p in Pane::PANELS {
        let mut open = dock.find_tab(&p).is_some();
        if ui.checkbox(&mut open, p.name()).clicked() {
            if open {
                focus(dock, p);
            } else if let Some(path) = dock.find_tab(&p) {
                dock.remove_tab(path);
            }
            ui.close();
        }
    }
    ui.separator();
    if ui.button(tr("Reset Panel Layout")).clicked() {
        *dock = default_layout();
        ui.close();
    }
    ui.label(
        egui::RichText::new(tr("Drag tabs to move panels; drag a tab out to float it."))
            .weak()
            .small(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_roundtrip_and_reopen() {
        let mut d = default_layout();
        let v = serde_json::to_value(&d).unwrap();
        assert!(layout_from_json(&v).is_some());
        // Closing and focusing a panel brings it back.
        let path = d.find_tab(&Pane::History).unwrap();
        d.remove_tab(path);
        assert!(d.find_tab(&Pane::History).is_none());
        focus(&mut d, Pane::History);
        assert!(d.find_tab(&Pane::History).is_some());
        // A layout without canvas is rejected.
        let no_canvas = DockState::new(vec![Pane::Layers]);
        assert!(layout_from_json(&serde_json::to_value(&no_canvas).unwrap()).is_none());
    }
}
