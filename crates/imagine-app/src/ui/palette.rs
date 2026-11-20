//! Every command reachable from the menus and the command palette, with
//! its keyboard shortcut. Executing a command replays its shortcut, so
//! menus, palette and keyboard can never disagree.

use imagine_core::tr;
use smithay_client_toolkit::seat::keyboard::Keysym as K;

#[derive(Debug, Clone, Copy)]
pub struct Cmd {
    pub menu: &'static str,
    pub label: &'static str,
    pub key: K,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

const fn c(
    menu: &'static str,
    label: &'static str,
    key: K,
    ctrl: bool,
    shift: bool,
    alt: bool,
) -> Cmd {
    Cmd {
        menu,
        label,
        key,
        ctrl,
        shift,
        alt,
    }
}

pub const MENUS: [&str; 7] = ["File", "Edit", "Image", "Layer", "Select", "Tool", "View"];

pub const COMMANDS: &[Cmd] = &[
    c("File", "New …", K::n, true, false, false),
    c("File", "Open …", K::o, true, false, false),
    c("File", "Save", K::s, true, false, false),
    c("File", "Save As …", K::S, true, true, false),
    c("File", "Export As …", K::E, true, true, true),
    c("File", "Export as PNG", K::E, true, true, false),
    c("File", "Quit", K::q, true, false, false),
    c("Edit", "Undo", K::z, true, false, false),
    c("Edit", "Redo", K::Z, true, true, false),
    c("Edit", "Copy", K::c, true, false, false),
    c("Edit", "Copy Merged", K::C, true, true, false),
    c("Edit", "Cut", K::x, true, false, false),
    c("Edit", "Paste", K::v, true, false, false),
    c("Edit", "Free Transform", K::t, true, false, false),
    c("Edit", "Fill Area", K::BackSpace, false, false, true),
    c(
        "Edit",
        "Content-Aware Fill",
        K::BackSpace,
        false,
        true,
        false,
    ),
    c(
        "Edit",
        "Content-Aware Fill (All Layers)",
        K::BackSpace,
        true,
        true,
        false,
    ),
    c(
        "Edit",
        "Delete Selection (Pixels)",
        K::Delete,
        false,
        false,
        false,
    ),
    c("Image", "Crop Image", K::c, false, false, false),
    c(
        "Image",
        "Rotate Image 90° Clockwise",
        K::Right,
        true,
        true,
        true,
    ),
    c(
        "Image",
        "Rotate Image 90° Counter-Clockwise",
        K::Left,
        true,
        true,
        true,
    ),
    c("Image", "Rotate Image 180°", K::Down, true, true, true),
    c("Image", "Flip Image Horizontal", K::H, true, true, true),
    c("Image", "Flip Image Vertical", K::V, true, true, true),
    c("Image", "Liquify …", K::X, true, true, false),
    c(
        "Image",
        "Blend Space sRGB / Linear",
        K::F7,
        false,
        false,
        false,
    ),
    c("Layer", "New Layer", K::N, true, true, false),
    c("Layer", "Duplicate Layer", K::j, true, false, false),
    c("Layer", "Group Layers", K::g, true, false, false),
    c("Layer", "Ungroup Layers", K::G, true, true, false),
    c(
        "Layer",
        "Rasterize (Text/Shape to Pixels)",
        K::r,
        true,
        false,
        true,
    ),
    c(
        "Layer",
        "Rotate Layer 90° Clockwise",
        K::Right,
        true,
        false,
        true,
    ),
    c(
        "Layer",
        "Rotate Layer 90° Counter-Clockwise",
        K::Left,
        true,
        false,
        true,
    ),
    c("Layer", "Rotate Layer 180°", K::Down, true, false, true),
    c("Layer", "Flip Layer Horizontal", K::h, true, false, true),
    c("Layer", "Flip Layer Vertical", K::v, true, false, true),
    c("Layer", "Free Transform / Rotate", K::t, true, false, false),
    c("Layer", "Add Layer Mask", K::M, true, true, false),
    c("Layer", "Toggle Layer Mask", K::m, true, false, true),
    c("Layer", "Apply Layer Mask", K::M, true, true, true),
    c(
        "Layer",
        "Paint Mask / Pixels",
        K::backslash,
        false,
        false,
        false,
    ),
    c(
        "Layer",
        "Toggle Layer Visibility",
        K::comma,
        true,
        false,
        false,
    ),
    c(
        "Layer",
        "Bring Layer Forward",
        K::bracketright,
        true,
        false,
        false,
    ),
    c(
        "Layer",
        "Send Layer Backward",
        K::bracketleft,
        true,
        false,
        false,
    ),
    c("Layer", "Next Blend Mode", K::plus, false, true, false),
    c("Select", "Select All", K::a, true, false, false),
    c("Select", "Deselect", K::d, true, false, false),
    c("Select", "Inverse", K::I, true, true, false),
    c("Select", "Feather (10 px)", K::F6, false, true, false),
    c("Select", "Quick Mask View", K::q, false, false, false),
    c("Tool", "Brush", K::b, false, false, false),
    c("Tool", "Text", K::t, false, false, false),
    c("Tool", "Pen", K::p, false, false, false),
    c("Tool", "Next Palette Color", K::P, false, true, false),
    c(
        "Tool",
        "Shape (Rectangle / Ellipse)",
        K::U,
        false,
        true,
        false,
    ),
    c("Tool", "Smudge", K::u, false, false, false),
    c("Tool", "Clone Stamp", K::s, false, false, false),
    c(
        "Tool",
        "Spot Healing / Healing Brush",
        K::j,
        false,
        false,
        false,
    ),
    c("Tool", "Eraser", K::e, false, false, false),
    c(
        "Tool",
        "Rectangular / Elliptical Marquee",
        K::m,
        false,
        false,
        false,
    ),
    c("Tool", "Lasso", K::l, false, false, false),
    c("Tool", "Magic Wand", K::w, false, false, false),
    c("Tool", "Gradient", K::g, false, false, false),
    c("Tool", "Move", K::v, false, false, false),
    c("Tool", "Crop Tool", K::c, false, false, false),
    c("Tool", "Switch Brush Tip", K::B, false, true, false),
    c("Tool", "Toggle Brush Dynamics", K::J, false, true, false),
    c("Tool", "Black / White", K::x, false, false, false),
    c("View", "Fit on Screen", K::_0, true, false, false),
    c("View", "100 %", K::_1, true, false, false),
    c("View", "Reset Rotation", K::R, false, true, false),
    c("View", "Toggle Interface", K::Tab, false, false, false),
    c("View", "Command Palette", K::p, true, true, false),
    c("View", "Benchmark", K::F12, false, false, false),
];

impl Cmd {
    /// The menu name as shown (translated). `menu` itself stays the English key.
    pub fn menu_text(&self) -> &'static str {
        tr(self.menu)
    }

    /// The label as shown (translated). `label` itself stays the English key.
    pub fn label_text(&self) -> &'static str {
        tr(self.label)
    }

    /// Human-readable shortcut, e.g. "Ctrl+Shift+N".
    pub fn shortcut(&self) -> String {
        let ctrl = tr("Ctrl");
        let shift = tr("Shift");
        let mut s = String::new();
        if self.ctrl {
            s += ctrl;
            s += "+";
        }
        if self.alt {
            s += "Alt+";
        }
        let name = match self.key {
            K::BackSpace => tr("Backspace").to_string(),
            K::Delete => tr("Del").to_string(),
            K::Tab => "Tab".to_string(),
            K::backslash => "\\".to_string(),
            K::comma => ",".to_string(),
            K::plus => "+".to_string(),
            K::bracketleft => "[".to_string(),
            K::bracketright => "]".to_string(),
            K::Left => tr("Left Arrow").to_string(),
            K::Right => tr("Right Arrow").to_string(),
            K::Up => tr("Up Arrow").to_string(),
            K::Down => tr("Down Arrow").to_string(),
            k => {
                let n = k
                    .name()
                    .unwrap_or("?")
                    .trim_start_matches("XK_")
                    .to_string();
                let n = n.trim_start_matches('_').to_string();
                // Upper-case letters stand for Shift+letter.
                if n.len() == 1 && n.chars().all(|c| c.is_ascii_uppercase()) && !self.shift {
                    format!("{shift}+{n}")
                } else {
                    n.to_uppercase()
                }
            }
        };
        if self.shift {
            s += shift;
            s += "+";
        }
        s + &name
    }
}

/// Translated name of a menu key from [`MENUS`].
pub fn menu_text(menu: &'static str) -> &'static str {
    tr(menu)
}

/// The command with the given (English) label.
pub fn find(label: &str) -> Cmd {
    *COMMANDS
        .iter()
        .find(|c| c.label == label)
        .unwrap_or_else(|| panic!("no command {label:?}"))
}

/// Commands matching a query: all words must occur (case-insensitive) in
/// "menu label" (shown text, or the English original); prefix matches of
/// the label rank first.
pub fn search(query: &str) -> Vec<&'static Cmd> {
    let q = query.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();
    if words.is_empty() {
        return COMMANDS.iter().collect();
    }
    // How well one query word fits a label: 0 = the label starts with it,
    // 1 = a word of the label does, 2 = somewhere inside; menu name: 3.
    fn fit(w: &str, label: &str, menu: &str) -> Option<u32> {
        let label = label.to_lowercase();
        if label.starts_with(w) {
            Some(0)
        } else if label
            .split(|c: char| !c.is_alphanumeric())
            .any(|x| x.starts_with(w))
        {
            Some(1)
        } else if label.contains(w) {
            Some(2)
        } else if menu.to_lowercase().contains(w) {
            Some(3)
        } else {
            None
        }
    }
    let score = |label: &str, menu: &str| -> Option<u32> {
        words.iter().map(|w| fit(w, label, menu)).sum()
    };
    let mut hits: Vec<(&Cmd, u32, usize)> = COMMANDS
        .iter()
        .filter_map(|c| {
            // Translated text or the English original, whichever fits better.
            let s = match (score(c.label_text(), c.menu_text()), score(c.label, c.menu)) {
                (Some(a), Some(b)) => a.min(b),
                (a, b) => a.or(b)?,
            };
            Some((c, s, c.label_text().chars().count()))
        })
        .collect();
    hits.sort_by_key(|(_, s, len)| (*s, *len));
    hits.into_iter().map(|(c, ..)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortcuts_read_like_photoshop() {
        let find = |l: &str| COMMANDS.iter().find(|c| c.label == l).unwrap().shortcut();
        assert_eq!(find("New Layer"), "Ctrl+Shift+N");
        assert_eq!(find("Save"), "Ctrl+S");
        assert_eq!(find("Fill Area"), "Alt+Backspace");
        assert_eq!(find("Apply Layer Mask"), "Ctrl+Alt+Shift+M");
        assert_eq!(find("Fit on Screen"), "Ctrl+0");
        assert_eq!(find("Switch Brush Tip"), "Shift+B");
    }

    #[test]
    fn search_finds_by_words_and_ranks_prefixes() {
        let r = search("mask");
        assert!(r.len() >= 3);
        assert!(search("layer new").iter().any(|c| c.label == "New Layer"));
        assert_eq!(search("mag")[0].label, "Magic Wand");
        assert!(search("xyzzy").is_empty());
        assert_eq!(search("").len(), COMMANDS.len());
    }

    #[test]
    fn every_menu_has_commands() {
        for m in MENUS {
            assert!(COMMANDS.iter().any(|c| c.menu == m), "{m}");
        }
        assert!(COMMANDS.iter().all(|c| MENUS.contains(&c.menu)));
    }
}
