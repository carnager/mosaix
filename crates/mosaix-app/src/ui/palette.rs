//! Every command reachable from the menus and the command palette, with
//! its keyboard shortcut. Executing a command replays its shortcut, so
//! menus, palette and keyboard can never disagree.

use std::collections::BTreeMap;

use mosaix_core::tr;
use serde::{Deserialize, Serialize};
use smithay_client_toolkit::seat::keyboard::Keysym as K;

/// A key combination, normalised (an upper-case letter is the lower-case
/// letter with Shift).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Combo {
    pub key: u32,
    #[serde(default)]
    pub ctrl: bool,
    #[serde(default)]
    pub shift: bool,
    #[serde(default)]
    pub alt: bool,
}

impl Combo {
    pub fn new(key: K, ctrl: bool, shift: bool, alt: bool) -> Self {
        let raw = key.raw();
        let (key, shift) = if (0x41..=0x5a).contains(&raw) { (raw + 0x20, true) } else { (raw, shift) };
        Self { key, ctrl, shift, alt }
    }

    pub fn of(c: &Cmd) -> Self {
        Self::new(c.key, c.ctrl, c.shift, c.alt)
    }

    pub fn text(&self) -> String {
        Cmd { menu: "", label: "", key: K::new(self.key), ctrl: self.ctrl, shift: self.shift, alt: self.alt }.shortcut()
    }
}

/// The user's shortcuts: command id → combination (`None` = no shortcut).
/// Commands not listed keep their default.
pub type Keymap = BTreeMap<String, Option<Combo>>;

/// The shortcut a command has with the user's keymap.
pub fn effective(c: &Cmd, km: &Keymap) -> Option<Combo> {
    match km.get(&c.id()) {
        Some(o) => *o,
        None => Some(Combo::of(c)),
    }
}

/// Text of a command's shortcut ("" if it has none).
pub fn shortcut_text(c: &Cmd, km: &Keymap) -> String {
    effective(c, km).map(|k| k.text()).unwrap_or_default()
}

/// What a key combination means with the user's keymap.
pub enum Lookup {
    /// Bound by the user to this command.
    Command(Cmd),
    /// A default shortcut whose command was given another one (or none).
    Freed,
    /// Not affected by the keymap: handle as usual.
    Default,
}

pub fn lookup(combo: Combo, km: &Keymap) -> Lookup {
    if km.is_empty() {
        return Lookup::Default;
    }
    if let Some(c) = COMMANDS.iter().find(|c| km.get(&c.id()) == Some(&Some(combo))) {
        return Lookup::Command(*c);
    }
    let defaults = |overridden: bool| COMMANDS.iter().any(|c| km.contains_key(&c.id()) == overridden && Combo::of(c) == combo);
    if defaults(false) {
        Lookup::Default
    } else if defaults(true) {
        Lookup::Freed
    } else {
        Lookup::Default
    }
}

/// Sets a command's shortcut (`None` = remove; `reset` = back to the
/// default). Another command using the same keys loses them, like in Photoshop.
pub fn rebind(km: &mut Keymap, id: &str, combo: Option<Combo>, reset: bool) {
    let Some(cmd) = COMMANDS.iter().find(|c| c.id() == id) else { return };
    let new = if reset { Some(Combo::of(cmd)) } else { combo };
    if let Some(k) = new {
        for other in COMMANDS.iter().filter(|c| c.id() != id) {
            if effective(other, km) == Some(k) {
                km.insert(other.id(), None);
            }
        }
    }
    if reset {
        km.remove(id);
    } else {
        km.insert(id.to_string(), combo);
    }
    // Entries equal to their default are not needed.
    km.retain(|id, v| COMMANDS.iter().find(|c| c.id() == *id).is_some_and(|c| *v != Some(Combo::of(c))));
}

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
    c("File", "Close", K::w, true, false, false),
    c("File", "Save", K::s, true, false, false),
    c("File", "Save As …", K::S, true, true, false),
    c("File", "Export As …", K::E, true, true, true),
    c("File", "Export as PNG", K::E, true, true, false),
    c("File", "Quit", K::q, true, false, false),
    c("Edit", "Undo", K::z, true, false, false),
    c("Edit", "Redo", K::Z, true, true, false),
    c("Edit", "Keyboard Shortcuts …", K::k, true, true, true),
    c("Edit", "Import Brushes …", K::b, true, true, true),
    c("Edit", "Define Pattern", K::D, true, true, true),
    c("Edit", "Import Patterns …", K::Y, true, true, true),
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
    c("Edit", "Remove (AI)", K::BackSpace, false, true, true),
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
    c("Layer", "Layer via Cut", K::J, true, true, false),
    c("Layer", "Merge Down", K::e, true, false, false),
    c("Layer", "Merge Visible", K::e, true, false, true),
    c("Layer", "Flatten Image", K::F, true, true, true),
    c(
        "Layer",
        "Create/Release Clipping Mask",
        K::g,
        true,
        false,
        true,
    ),
    c("Layer", "Group Layers", K::g, true, false, false),
    c("Layer", "Ungroup Layers", K::G, true, true, false),
    c(
        "Layer",
        "Rasterize (Text/Shape/RAW to Pixels)",
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
    c("Layer", "Remove Background (AI)", K::b, true, false, true),
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
    c("Image", "Invert", K::i, true, false, false),
    c("Image", "Desaturate", K::U, true, true, false),
    c("Image", "Hue/Saturation …", K::u, true, false, false),
    c("Image", "Levels …", K::l, true, false, false),
    c("Image", "Curves …", K::m, true, false, false),
    c("Image", "Auto Tone", K::L, true, true, false),
    c("Image", "Auto Contrast", K::L, true, true, true),
    c("Image", "Auto Color", K::B, true, true, false),
    c("Select", "Select All", K::a, true, false, false),
    c("Select", "Deselect", K::d, true, false, false),
    c("Select", "Inverse", K::I, true, true, false),
    c("Select", "Feather …", K::F6, false, true, false),
    c("Select", "Quick Mask", K::q, false, false, false),
    c("Select", "Subject (AI)", K::s, true, false, true),
    c("Tool", "Object Selection (AI)", K::W, false, true, false),
    c("Tool", "Brush / Pencil", K::b, false, false, false),
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
    c("Tool", "Dodge / Burn / Sponge", K::o, false, false, false),
    c("Tool", "Clone Stamp", K::s, false, false, false),
    c("Tool", "History Brush", K::y, false, false, false),
    c(
        "Tool",
        "Spot Healing / Remove (AI) / Healing Brush",
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
    c("Tool", "Lasso / Polygonal Lasso", K::l, false, false, false),
    c("Tool", "Magic Wand", K::w, false, false, false),
    c("Tool", "Gradient / Paint Bucket", K::g, false, false, false),
    c("Tool", "Move", K::v, false, false, false),
    c("Tool", "Crop Tool", K::c, false, false, false),
    c("Tool", "Switch Brush Tip", K::B, false, true, false),
    c("Tool", "Toggle Brush Dynamics", K::J, false, true, false),
    c("Tool", "Black / White", K::x, false, false, false),
    c("View", "Fit on Screen", K::_0, true, false, false),
    c("View", "100 %", K::_1, true, false, false),
    c("View", "Reset Rotation", K::R, false, true, false),
    c("View", "Toggle Interface", K::Tab, false, false, false),
    c("View", "Next Document", K::Tab, true, false, false),
    c("View", "Previous Document", K::ISO_Left_Tab, true, true, false),
    c("View", "Rulers", K::r, true, false, false),
    c("View", "Guides", K::semicolon, true, false, false),
    c("View", "Grid", K::apostrophe, true, false, false),
    c("View", "Snap", K::colon, true, true, false),
    c("View", "Clear Guides", K::semicolon, true, false, true),
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
    /// Stable id (English menu and label) for the keymap.
    pub fn id(&self) -> String {
        format!("{}/{}", self.menu, self.label)
    }

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
            K::Tab | K::ISO_Left_Tab => "Tab".to_string(),
            K::backslash => "\\".to_string(),
            K::comma => ",".to_string(),
            K::plus => "+".to_string(),
            K::semicolon => ";".to_string(),
            // Shift+; on most layouts; Shift is shown already.
            K::colon => ";".to_string(),
            K::apostrophe => "'".to_string(),
            K::bracketleft => "[".to_string(),
            K::bracketright => "]".to_string(),
            K::Left => super::glyphs::ARROW_LEFT.to_string(),
            K::Right => super::glyphs::ARROW_RIGHT.to_string(),
            K::Up => super::glyphs::ARROW_UP.to_string(),
            K::Down => super::glyphs::ARROW_DOWN.to_string(),
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
#[cfg(test)]
pub fn search(query: &str) -> Vec<&'static Cmd> {
    rank(query, COMMANDS.iter().map(|c| (c, [c.label_text(), c.menu_text(), c.label, c.menu])))
}

/// Ranks items by `query`: each item comes with its shown label and menu
/// and their English originals. Without a query all items, in order.
pub fn rank<'a, T>(query: &str, items: impl IntoIterator<Item = (T, [&'a str; 4])>) -> Vec<T> {
    let q = query.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();
    if words.is_empty() {
        return items.into_iter().map(|(t, _)| t).collect();
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
    let mut hits: Vec<(T, u32, usize)> = items
        .into_iter()
        .filter_map(|(t, [label, menu, label_en, menu_en])| {
            // Translated text or the English original, whichever fits better.
            let s = match (score(label, menu), score(label_en, menu_en)) {
                (Some(a), Some(b)) => a.min(b),
                (a, b) => a.or(b)?,
            };
            Some((t, s, label.chars().count()))
        })
        .collect();
    hits.sort_by_key(|(_, s, len)| (*s, *len));
    hits.into_iter().map(|(c, ..)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keymap_rebinds_frees_and_resolves_conflicts() {
        let mut km = Keymap::new();
        let copy = find("Copy");
        let paste = find("Paste");
        let ctrl_k = Combo::new(K::k, true, false, false);
        // Copy → Ctrl+K: runs Copy; Ctrl+C is free now.
        rebind(&mut km, &copy.id(), Some(ctrl_k), false);
        assert!(matches!(lookup(ctrl_k, &km), Lookup::Command(c) if c.label == "Copy"));
        assert!(matches!(lookup(Combo::of(&copy), &km), Lookup::Freed));
        assert_eq!(shortcut_text(&copy, &km), ctrl_k.text());
        // Paste takes Ctrl+K: Copy loses it (no shortcut).
        rebind(&mut km, &paste.id(), Some(ctrl_k), false);
        assert_eq!(effective(&copy, &km), None);
        assert!(matches!(lookup(ctrl_k, &km), Lookup::Command(c) if c.label == "Paste"));
        // Untouched defaults keep working.
        assert!(matches!(lookup(Combo::of(&find("Cut")), &km), Lookup::Default));
        // Reset: back to defaults, nothing stored.
        rebind(&mut km, &paste.id(), None, true);
        rebind(&mut km, &copy.id(), None, true);
        assert!(km.is_empty(), "{km:?}");
        // Upper-case keysym = Shift + letter.
        assert_eq!(Combo::new(K::C, true, false, false), Combo::new(K::c, true, true, false));
    }

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
