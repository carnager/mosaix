//! Filter plugins: imagine's examples plus `.wgsl` files in
//! `$XDG_DATA_HOME/imagine/filters` (see imagine_core::plugin for the format).

use std::path::{Path, PathBuf};

use imagine_core::Filter;

pub const BUILTIN: [(&str, &str); 4] = [
    ("vignette.wgsl", include_str!("../../plugins/vignette.wgsl")),
    ("pixelate.wgsl", include_str!("../../plugins/pixelate.wgsl")),
    ("chromatic.wgsl", include_str!("../../plugins/chromatic.wgsl")),
    ("clouds.wgsl", include_str!("../../plugins/clouds.wgsl")),
];

pub fn filter_dir() -> PathBuf {
    super::brushes::brush_dir().with_file_name("filters")
}

/// A plugin that loaded, or why not.
#[derive(Debug, Clone, PartialEq)]
pub struct Plugin {
    pub file: String,
    pub builtin: bool,
    pub result: Result<Filter, String>,
}

impl Plugin {
    pub fn description(&self) -> String {
        match &self.result {
            Ok(Filter::Plugin { source, .. }) => imagine_core::plugin::parse(source).map(|s| s.description).unwrap_or_default(),
            _ => String::new(),
        }
    }
}

fn load(file: String, builtin: bool, source: &str) -> Plugin {
    let result = imagine_gpu::engine::validate_plugin(source).and_then(|_| Filter::plugin(source));
    Plugin { file, builtin, result }
}

/// Built-ins, then the folder (sorted by file name; a file with a built-in's
/// name replaces it).
pub fn load_library(dir: &Path) -> Vec<Plugin> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wgsl"))).collect())
        .unwrap_or_default();
    files.sort();
    let names: Vec<String> = files.iter().filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).collect();
    let mut out: Vec<Plugin> = BUILTIN
        .iter()
        .filter(|(f, _)| !names.iter().any(|n| n == f))
        .map(|(f, s)| load(f.to_string(), true, s))
        .collect();
    for p in files {
        let file = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match std::fs::read_to_string(&p) {
            Ok(s) => out.push(load(file, false, &s)),
            Err(e) => out.push(Plugin { file, builtin: false, result: Err(e.to_string()) }),
        }
    }
    out
}

/// The plugin filter with this name (case-insensitive), with default values.
pub fn find(lib: &[Plugin], name: &str) -> Option<Filter> {
    lib.iter().filter_map(|p| p.result.as_ref().ok()).find(|f| f.name().eq_ignore_ascii_case(name)).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_compile() {
        for (f, s) in BUILTIN {
            let p = load(f.into(), true, s);
            assert!(p.result.is_ok(), "{f}: {:?}", p.result);
        }
    }
}
