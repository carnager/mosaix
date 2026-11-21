//! The app's folder inside an XDG base directory (config, data, state).

use std::path::{Path, PathBuf};

/// `base/mosaix`. The project used to be called imagine: if only
/// `base/imagine` exists it is moved over once (settings, brushes, models …
/// keep working); if that fails the old folder is used as it is.
pub fn app_dir(base: &Path) -> PathBuf {
    let dir = base.join("mosaix");
    let old = base.join("imagine");
    if !dir.exists() && old.is_dir() && std::fs::rename(&old, &dir).is_err() {
        return old;
    }
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_the_old_folder() {
        let base = std::env::temp_dir().join(format!("mosaix-paths-{}", std::process::id()));
        std::fs::create_dir_all(base.join("imagine")).unwrap();
        std::fs::write(base.join("imagine/settings.json"), "{}").unwrap();
        let d = app_dir(&base);
        assert_eq!(d, base.join("mosaix"));
        assert!(d.join("settings.json").exists() && !base.join("imagine").exists());
        // Once moved (or for new users) nothing happens.
        assert_eq!(app_dir(&base), base.join("mosaix"));
        std::fs::remove_dir_all(&base).unwrap();
    }
}
