//! Every command reachable from the menus and the command palette, with
//! its keyboard shortcut. Executing a command replays its shortcut, so
//! menus, palette and keyboard can never disagree.

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

const fn c(menu: &'static str, label: &'static str, key: K, ctrl: bool, shift: bool, alt: bool) -> Cmd {
    Cmd { menu, label, key, ctrl, shift, alt }
}

pub const MENUS: [&str; 7] = ["Datei", "Bearbeiten", "Bild", "Ebene", "Auswahl", "Werkzeug", "Ansicht"];

pub const COMMANDS: &[Cmd] = &[
    c("Datei", "Neu …", K::n, true, false, false),
    c("Datei", "Öffnen …", K::o, true, false, false),
    c("Datei", "Speichern", K::s, true, false, false),
    c("Datei", "Speichern unter …", K::S, true, true, false),
    c("Datei", "Exportieren unter …", K::E, true, true, true),
    c("Datei", "Als PNG exportieren", K::E, true, true, false),
    c("Datei", "Beenden", K::q, true, false, false),
    c("Bearbeiten", "Rückgängig", K::z, true, false, false),
    c("Bearbeiten", "Wiederholen", K::Z, true, true, false),
    c("Bearbeiten", "Frei transformieren", K::t, true, false, false),
    c("Bearbeiten", "Fläche füllen", K::BackSpace, false, false, true),
    c("Bearbeiten", "Auswahl löschen (Pixel)", K::Delete, false, false, false),
    c("Bild", "Zuschneiden", K::c, false, false, false),
    c("Bild", "Verflüssigen …", K::X, true, true, false),
    c("Bild", "Blend-Raum sRGB / linear", K::F7, false, false, false),
    c("Ebene", "Neue Ebene", K::N, true, true, false),
    c("Ebene", "Ebene duplizieren", K::j, true, false, false),
    c("Ebene", "Gruppieren", K::g, true, false, false),
    c("Ebene", "Gruppierung aufheben", K::G, true, true, false),
    c("Ebene", "Ebenenmaske hinzufügen", K::M, true, true, false),
    c("Ebene", "Ebenenmaske ein/aus", K::m, true, false, true),
    c("Ebene", "Ebenenmaske anwenden", K::M, true, true, true),
    c("Ebene", "Maske bemalen / Pixel", K::backslash, false, false, false),
    c("Ebene", "Ebene sichtbar/unsichtbar", K::comma, true, false, false),
    c("Ebene", "Ebene nach oben", K::bracketright, true, false, false),
    c("Ebene", "Ebene nach unten", K::bracketleft, true, false, false),
    c("Ebene", "Nächste Füllmethode", K::plus, false, true, false),
    c("Auswahl", "Alles auswählen", K::a, true, false, false),
    c("Auswahl", "Auswahl aufheben", K::d, true, false, false),
    c("Auswahl", "Auswahl umkehren", K::I, true, true, false),
    c("Auswahl", "Weiche Kante (10 px)", K::F6, false, true, false),
    c("Auswahl", "Maskierungsmodus-Ansicht", K::q, false, false, false),
    c("Werkzeug", "Pinsel", K::b, false, false, false),
    c("Werkzeug", "Wischfinger", K::u, false, false, false),
    c("Werkzeug", "Kopierstempel", K::s, false, false, false),
    c("Werkzeug", "Radierer", K::e, false, false, false),
    c("Werkzeug", "Auswahlrechteck / -ellipse", K::m, false, false, false),
    c("Werkzeug", "Lasso", K::l, false, false, false),
    c("Werkzeug", "Zauberstab", K::w, false, false, false),
    c("Werkzeug", "Verlauf", K::g, false, false, false),
    c("Werkzeug", "Verschieben", K::v, false, false, false),
    c("Werkzeug", "Zuschneiden", K::c, false, false, false),
    c("Werkzeug", "Pinselspitze wechseln", K::B, false, true, false),
    c("Werkzeug", "Pinsel-Dynamik an/aus", K::j, false, false, false),
    c("Werkzeug", "Schwarz / Weiß", K::x, false, false, false),
    c("Ansicht", "Einpassen", K::_0, true, false, false),
    c("Ansicht", "100 %", K::_1, true, false, false),
    c("Ansicht", "Drehung zurücksetzen", K::R, false, true, false),
    c("Ansicht", "Oberfläche ein/aus", K::Tab, false, false, false),
    c("Ansicht", "Befehlspalette", K::p, true, true, false),
    c("Ansicht", "Benchmark", K::F12, false, false, false),
];

impl Cmd {
    /// Human-readable shortcut, e.g. "Strg+Umschalt+N".
    pub fn shortcut(&self) -> String {
        let mut s = String::new();
        if self.ctrl {
            s += "Strg+";
        }
        if self.alt {
            s += "Alt+";
        }
        let name = match self.key {
            K::BackSpace => "Rücktaste".to_string(),
            K::Delete => "Entf".to_string(),
            K::Tab => "Tab".to_string(),
            K::backslash => "\\".to_string(),
            K::comma => ",".to_string(),
            K::plus => "+".to_string(),
            K::bracketleft => "[".to_string(),
            K::bracketright => "]".to_string(),
            k => {
                let n = k.name().unwrap_or("?").trim_start_matches("XK_").to_string();
                let n = n.trim_start_matches('_').to_string();
                // Upper-case letters stand for Shift+letter.
                if n.len() == 1 && n.chars().all(|c| c.is_ascii_uppercase()) && !self.shift {
                    format!("Umschalt+{n}")
                } else {
                    n.to_uppercase()
                }
            }
        };
        if self.shift {
            s += "Umschalt+";
        }
        s + &name
    }
}

/// Commands matching a query: all words must occur (case-insensitive) in
/// "menu label"; prefix matches of the label rank first.
pub fn search(query: &str) -> Vec<&'static Cmd> {
    let q = query.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();
    let mut hits: Vec<(&Cmd, bool)> = COMMANDS
        .iter()
        .filter_map(|c| {
            let hay = format!("{} {}", c.menu, c.label).to_lowercase();
            words.iter().all(|w| hay.contains(w)).then(|| (c, c.label.to_lowercase().starts_with(&q)))
        })
        .collect();
    hits.sort_by_key(|(_, prefix)| !*prefix);
    hits.into_iter().map(|(c, _)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortcuts_read_like_photoshop() {
        let find = |l: &str| COMMANDS.iter().find(|c| c.label == l).unwrap().shortcut();
        assert_eq!(find("Neue Ebene"), "Strg+Umschalt+N");
        assert_eq!(find("Speichern"), "Strg+S");
        assert_eq!(find("Fläche füllen"), "Alt+Rücktaste");
        assert_eq!(find("Ebenenmaske anwenden"), "Strg+Alt+Umschalt+M");
        assert_eq!(find("Einpassen"), "Strg+0");
        assert_eq!(find("Pinselspitze wechseln"), "Umschalt+B");
    }

    #[test]
    fn search_finds_by_words_and_ranks_prefixes() {
        let r = search("maske");
        assert!(r.len() >= 3);
        assert!(search("ebene neu").iter().any(|c| c.label == "Neue Ebene"));
        assert_eq!(search("zau")[0].label, "Zauberstab");
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
