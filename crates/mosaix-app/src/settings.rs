//! Preferences that survive restarts: brush, colour, tool options, recent
//! files and the last folder. Stored as JSON in
//! `$XDG_CONFIG_HOME/mosaix/settings.json` (default `~/.config/…`).

use std::path::{Path, PathBuf};

use mosaix_core::BrushSettings;
use serde::{Deserialize, Serialize};

const RECENT: usize = 12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Brush {
    pub size: f32,
    pub hardness: f32,
    pub flow: f32,
    pub opacity: f32,
    pub spacing: f32,
    pub smoothing: f32,
    pub tip: u32,
    pub angle: f32,
    pub roundness: f32,
    pub size_jitter: f32,
    pub angle_jitter: f32,
    pub scatter: f32,
    pub flow_jitter: f32,
    pub follow_direction: bool,
    pub color: [f32; 3],
}

impl Default for Brush {
    fn default() -> Self {
        Self::from(&BrushSettings::default())
    }
}

impl From<&BrushSettings> for Brush {
    fn from(b: &BrushSettings) -> Self {
        Self {
            size: b.size,
            hardness: b.hardness,
            flow: b.flow,
            opacity: b.opacity,
            spacing: b.spacing,
            smoothing: b.smoothing,
            tip: b.tip,
            angle: b.angle,
            roundness: b.roundness,
            size_jitter: b.size_jitter,
            angle_jitter: b.angle_jitter,
            scatter: b.scatter,
            flow_jitter: b.flow_jitter,
            follow_direction: b.follow_direction,
            color: b.color,
        }
    }
}

impl Brush {
    /// Applies the stored values (clamped, so a hand-edited file cannot break anything).
    pub fn apply(&self, b: &mut BrushSettings) {
        b.size = self.size.clamp(1.0, 5000.0);
        b.hardness = self.hardness.clamp(0.0, 1.0);
        b.flow = self.flow.clamp(0.01, 1.0);
        b.opacity = self.opacity.clamp(0.0, 1.0);
        b.spacing = self.spacing.clamp(0.02, 2.0);
        b.smoothing = self.smoothing.clamp(0.0, 0.95);
        b.tip = self.tip.min(mosaix_core::TIP_NAMES.len() as u32 - 1);
        b.angle = self.angle;
        b.roundness = self.roundness.clamp(0.05, 1.0);
        b.size_jitter = self.size_jitter.clamp(0.0, 1.0);
        b.angle_jitter = self.angle_jitter.clamp(0.0, 180.0);
        b.scatter = self.scatter.clamp(0.0, 5.0);
        b.flow_jitter = self.flow_jitter.clamp(0.0, 1.0);
        b.follow_direction = self.follow_direction;
        b.color = self.color.map(|c| c.clamp(0.0, 1.0));
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub brush: Brush,
    pub tool: String,
    pub wand_tolerance: f32,
    pub wand_contiguous: bool,
    pub bg_color: [f32; 3],
    pub eyedrop_radius: i64,
    pub eyedrop_all: bool,
    pub smudge_strength: f32,
    pub recent: Vec<PathBuf>,
    pub last_dir: Option<PathBuf>,
    pub export: mosaix_io::export::ExportOptions,
    /// Interface language: "auto" (from LANGUAGE / LC_ALL / LC_MESSAGES / LANG), "en", "de".
    pub language: String,
    pub favourite_fonts: Vec<String>,
    /// Panel layout (egui_dock state); `None` = default.
    pub dock_layout: Option<serde_json::Value>,
    /// The user's keyboard shortcuts (only those that differ from the defaults).
    pub keymap: crate::ui::palette::Keymap,
    pub recent_fonts: Vec<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            brush: Brush::default(),
            tool: "brush".into(),
            wand_tolerance: 32.0,
            wand_contiguous: true,
            bg_color: [1.0; 3],
            eyedrop_radius: 0,
            eyedrop_all: true,
            smudge_strength: 0.7,
            recent: Vec::new(),
            last_dir: None,
            export: Default::default(),
            language: "auto".into(),
            favourite_fonts: Vec::new(),
            dock_layout: None,
            keymap: Default::default(),
            recent_fonts: Vec::new(),
        }
    }
}

pub fn config_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(std::env::temp_dir);
    mosaix_core::paths::app_dir(&base).join("settings.json")
}

impl Settings {
    /// Loads the settings; missing or broken files give the defaults.
    pub fn load_from(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                // Plain English, no tr(): this runs before the language is initialised
                // (tr() would lock the catalog to English).
                tracing::warn!(
                    "Settings {} unreadable ({e}), using defaults",
                    path.display()
                );
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Writes atomically (temp file + rename).
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    /// Puts a file at the top of the recent list.
    pub fn add_recent(&mut self, path: &Path) {
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.recent.retain(|p| *p != path);
        self.recent.insert(0, path.clone());
        self.recent.truncate(RECENT);
        self.last_dir = path.parent().map(Path::to_path_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_recent_and_robust_loading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/settings.json");
        let mut s = Settings::default();
        s.brush.size = 123.0;
        for i in 0..20 {
            s.add_recent(&dir.path().join(format!("f{i}.png")));
        }
        s.add_recent(&dir.path().join("f5.png"));
        assert_eq!(s.recent.len(), RECENT);
        assert!(s.recent[0].ends_with("f5.png"));
        assert_eq!(s.recent.iter().filter(|p| p.ends_with("f5.png")).count(), 1);
        s.save_to(&path).unwrap();
        assert_eq!(Settings::load_from(&path), s);

        // Unknown and missing fields are fine; garbage gives defaults.
        std::fs::write(&path, r#"{"tool": "wand", "future_option": 1}"#).unwrap();
        let l = Settings::load_from(&path);
        assert_eq!(l.tool, "wand");
        assert_eq!(l.wand_tolerance, 32.0);
        std::fs::write(&path, "nonsense").unwrap();
        assert_eq!(Settings::load_from(&path), Settings::default());

        let mut b = BrushSettings::default();
        Brush {
            size: 1e9,
            hardness: -3.0,
            tip: 99,
            ..Brush::default()
        }
        .apply(&mut b);
        assert_eq!((b.size, b.hardness), (5000.0, 0.0));
        assert!(b.tip < mosaix_core::TIP_NAMES.len() as u32);
    }
}
