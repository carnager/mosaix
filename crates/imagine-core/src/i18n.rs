//! Translations. Texts are written in English in the code; other languages
//! come from gettext PO catalogs embedded at build time
//! (`crates/imagine-core/i18n/<lang>.po`). The language follows the usual
//! Linux variables (LANGUAGE, LC_ALL, LC_MESSAGES, LANG), English for C/POSIX.
//!
//! - `tr("Open")` translates a literal or a runtime string (e.g. an undo
//!   label or an engine error that was written in English).
//! - `tr!("Exported {path}", path = p)` translates and fills `{name}` placeholders.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Catalogs compiled into the program: (language, PO text).
const CATALOGS: &[(&str, &str)] = &[("de", include_str!("../i18n/de.po"))];

static ACTIVE: OnceLock<Option<HashMap<String, String>>> = OnceLock::new();

/// Parses the msgid / msgstr pairs of a PO file (single and multi-line strings).
pub fn parse_po(text: &str) -> HashMap<String, String> {
    fn unquote(s: &str) -> Option<String> {
        let s = s.trim();
        let s = s.strip_prefix('"')?.strip_suffix('"')?;
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(o) => {
                        out.push('\\');
                        out.push(o);
                    }
                    None => out.push('\\'),
                }
            } else {
                out.push(c);
            }
        }
        Some(out)
    }
    let mut map = HashMap::new();
    let (mut id, mut st) = (None::<String>, None::<String>);
    // msgctxt: the same English text with different meanings ("ctx\u{4}text").
    let mut ctx: Option<String> = None;
    // Which string continuation lines belong to: 1 = msgid, 2 = msgstr, 3 = msgctxt.
    let mut which = 0;
    let mut flush = |id: &mut Option<String>, st: &mut Option<String>, ctx: &mut Option<String>| {
        if let (Some(i), Some(t)) = (id.take(), st.take())
            && !i.is_empty()
            && !t.is_empty()
        {
            match ctx.take() {
                Some(c) => map.insert(format!("{c}\u{4}{i}"), t),
                None => map.insert(i, t),
            };
        }
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("msgctxt ") {
            flush(&mut id, &mut st, &mut ctx);
            ctx = unquote(rest);
            which = 3;
        } else if let Some(rest) = line.strip_prefix("msgid ") {
            if which != 3 {
                flush(&mut id, &mut st, &mut ctx);
            }
            id = unquote(rest);
            which = 1;
        } else if let Some(rest) = line.strip_prefix("msgstr ") {
            st = unquote(rest);
            which = 2;
        } else if let Some(more) = unquote(line) {
            match which {
                1 => id.get_or_insert_with(String::new).push_str(&more),
                2 => st.get_or_insert_with(String::new).push_str(&more),
                3 => ctx.get_or_insert_with(String::new).push_str(&more),
                _ => {}
            }
        }
    }
    flush(&mut id, &mut st, &mut ctx);
    map
}

/// The language the environment asks for ("de", "en", …).
pub fn language_from_env() -> String {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let posix = |v: &str| v == "C" || v == "POSIX" || v.starts_with("C.");
    // LC_ALL wins over LC_MESSAGES over LANG; LANGUAGE (a priority list)
    // is honoured unless the locale is C/POSIX, like gettext does.
    let locale = get("LC_ALL")
        .or_else(|| get("LC_MESSAGES"))
        .or_else(|| get("LANG"))
        .unwrap_or_default();
    if posix(&locale) || locale.is_empty() {
        return "en".into();
    }
    let candidates = get("LANGUAGE")
        .map(|l| l.split(':').map(str::to_string).collect::<Vec<_>>())
        .unwrap_or_default();
    for c in candidates.iter().chain(std::iter::once(&locale)) {
        let code: String = c.split(['_', '.', '@']).next().unwrap_or("").to_lowercase();
        if code == "en" || CATALOGS.iter().any(|(l, _)| *l == code) {
            return code;
        }
    }
    "en".into()
}

/// Chooses the language once at startup ("auto" = from the environment).
/// Later calls have no effect.
pub fn init(choice: &str) {
    let lang = if choice.is_empty() || choice == "auto" {
        language_from_env()
    } else {
        choice.to_lowercase()
    };
    let _ = ACTIVE.set(
        CATALOGS
            .iter()
            .find(|(l, _)| *l == lang)
            .map(|(_, po)| parse_po(po)),
    );
}

/// The active language code.
pub fn language() -> &'static str {
    match ACTIVE.get() {
        Some(Some(_)) => "de",
        _ => "en",
    }
}

/// Translates `s` (returned unchanged if there is no translation).
pub fn tr(s: &str) -> &str {
    match ACTIVE.get_or_init(|| None) {
        Some(map) => map.get(s).map_or(s, |t| t.as_str()),
        None => s,
    }
}

/// Translates `s` in a context, for English words with several meanings
/// ("Fill" the command vs. "Fill" the layer property): the catalog entry
/// with `msgctxt "ctx"`, else `s` unchanged.
pub fn tr_ctx<'a>(ctx: &str, s: &'a str) -> &'a str {
    match ACTIVE.get_or_init(|| None) {
        Some(map) => map.get(&format!("{ctx}\u{4}{s}")).map_or(s, |t| t.as_str()),
        None => s,
    }
}

/// `tr!("text")` or `tr!("text with {name}", name = value, …)`.
#[macro_export]
macro_rules! tr {
    ($s:literal) => {
        $crate::i18n::tr($s)
    };
    ($s:literal, $($k:ident = $v:expr),+ $(,)?) => {{
        let mut out = $crate::i18n::tr($s).to_string();
        $( out = out.replace(concat!("{", stringify!($k), "}"), &format!("{}", $v)); )+
        out
    }};
}

#[cfg(test)]
mod tests {

    #[test]
    fn context_entries_are_separate() {
        let m = super::parse_po("msgid \"Fill\"\nmsgstr \"Füllen\"\n\nmsgctxt \"layer\"\nmsgid \"Fill\"\nmsgstr \"Fläche\"\n");
        assert_eq!(m.get("Fill").map(String::as_str), Some("Füllen"));
        assert_eq!(m.get("layer\u{4}Fill").map(String::as_str), Some("Fläche"));
    }

    use super::*;

    #[test]
    fn parses_po_and_formats() {
        let po = "# comment\nmsgid \"\"\nmsgstr \"\"\n\"Content-Type: text/plain\\n\"\n\nmsgid \"Open\"\nmsgstr \"Öffnen\"\n\nmsgid \"Two\\nlines\"\nmsgstr \"\"\n\"Zwei\\n\"\n\"Zeilen\"\n";
        let m = parse_po(po);
        assert_eq!(m.get("Open").map(String::as_str), Some("Öffnen"));
        assert_eq!(
            m.get("Two\nlines").map(String::as_str),
            Some("Zwei\nZeilen")
        );
        assert!(!m.contains_key(""), "the header is not a translation");
    }

    #[test]
    fn the_german_catalog_is_valid() {
        let m = parse_po(CATALOGS[0].1);
        // Placeholders must survive translation.
        for (k, v) in &m {
            for part in k.split('{').skip(1) {
                if let Some(name) = part
                    .split('}')
                    .next()
                    .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_'))
                {
                    assert!(
                        v.contains(&format!("{{{name}}}")),
                        "placeholder {{{name}}} missing in translation of {k:?}"
                    );
                }
            }
        }
    }
}
