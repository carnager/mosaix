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

/// Generative fill: who paints. The OpenAI key is never in this file (see
/// [`openai_key`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenSettings {
    /// "openai" or "command".
    pub provider: String,
    pub model: String,
    /// The program for "command": `program image.png mask.png prompt out.png`.
    pub command: String,
    /// The user agreed that pixels go to the hosted provider.
    pub remote_ok: bool,
}

impl Default for GenSettings {
    fn default() -> Self {
        Self { provider: "openai".into(), model: "gpt-image-1".into(), command: String::new(), remote_ok: false }
    }
}

impl GenSettings {
    /// The provider to use, or why there is none.
    pub fn provider(&self) -> Result<mosaix_ai::generative::Provider, String> {
        use mosaix_ai::generative::Provider;
        match self.provider.as_str() {
            "command" if self.command.trim().is_empty() => Err(mosaix_core::tr("Set the program for generative fill first").into()),
            "command" => Ok(Provider::Command { program: self.command.clone() }),
            _ => {
                if !self.remote_ok {
                    return Err(mosaix_core::tr("Generative fill with OpenAI needs your consent first (Edit › Generative Fill …)").into());
                }
                let Some(key) = openai_key() else {
                    return Err(mosaix_core::tr("No OpenAI API key: enter one in Edit › Generative Fill … or set OPENAI_API_KEY").into());
                };
                Ok(Provider::OpenAi { model: self.model.clone(), key })
            }
        }
    }
}

/// Dodge, burn, sponge, blur and sharpen.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToneOpts {
    /// Dodge/burn: 0 shadows, 1 midtones, 2 highlights.
    pub range: u8,
    /// Dodge/burn exposure, 0…1.
    pub exposure: f32,
    pub sponge_saturate: bool,
    /// Sponge flow, 0…1.
    pub sponge_flow: f32,
    /// Blur/sharpen strength, 0…1.
    pub strength: f32,
}

impl Default for ToneOpts {
    fn default() -> Self {
        Self { range: 1, exposure: 0.5, sponge_saturate: false, sponge_flow: 0.5, strength: 0.5 }
    }
}

/// View › Rulers, Guides, Grid, Snap.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewOpts {
    pub rulers: bool,
    pub guides: bool,
    pub grid: bool,
    pub snap: bool,
    /// Grid line distance in document px, and subdivisions between lines.
    pub grid_step: f64,
    pub grid_div: u32,
}

impl Default for ViewOpts {
    fn default() -> Self {
        Self { rulers: false, guides: true, grid: false, snap: true, grid_step: 100.0, grid_div: 4 }
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
    pub view: ViewOpts,
    pub tone: ToneOpts,
    pub generative: GenSettings,
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
            view: ViewOpts::default(),
            tone: ToneOpts::default(),
            generative: GenSettings::default(),
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

/// The saved OpenAI key: its own file next to settings.json (readable only
/// by the user), so settings.json can be shared or kept in dotfiles.
fn key_path() -> PathBuf {
    config_path().with_file_name("openai-key")
}

fn saved_key() -> Option<String> {
    let k = std::fs::read_to_string(key_path()).ok()?;
    let k = k.trim();
    (!k.is_empty()).then(|| k.to_string())
}

/// The OpenAI key: the one saved in the app, else `MOSAIX_OPENAI_KEY` or
/// `OPENAI_API_KEY`.
pub fn openai_key() -> Option<String> {
    saved_key().or_else(|| {
        std::env::var("MOSAIX_OPENAI_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .ok()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
    })
}

/// Saves (`Some`) or forgets (`None`) the OpenAI key.
pub fn store_openai_key(key: Option<&str>) -> anyhow::Result<()> {
    store_key_at(&key_path(), key)
}

fn store_key_at(path: &Path, key: Option<&str>) -> anyhow::Result<()> {
    let path = path.to_path_buf();
    match key.map(str::trim).filter(|k| !k.is_empty()) {
        None => match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        },
        Some(k) => {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let tmp = path.with_extension("tmp");
            let _ = std::fs::remove_file(&tmp);
            let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
            f.write_all(k.as_bytes())?;
            f.sync_all()?;
            std::fs::rename(tmp, path)?;
            Ok(())
        }
    }
}

/// Where the key comes from, for the dialog: the saved key's last four
/// characters, or that the environment has one.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum KeyState {
    #[default]
    None,
    Saved(String),
    Environment,
}

pub fn key_state() -> KeyState {
    match saved_key() {
        Some(k) => KeyState::Saved(k.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect()),
        None if openai_key().is_some() => KeyState::Environment,
        None => KeyState::None,
    }
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
mod key_tests {
    use super::*;

    #[test]
    fn saved_key_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("mosaix-key-{}", std::process::id()));
        let path = dir.join("openai-key");
        store_key_at(&path, Some("  sk-test-1234 \n")).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "sk-test-1234");
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        store_key_at(&path, Some("sk-other")).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "sk-other");
        store_key_at(&path, None).unwrap();
        assert!(!path.exists());
        store_key_at(&path, None).unwrap();
        let _ = std::fs::remove_dir_all(dir);
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
