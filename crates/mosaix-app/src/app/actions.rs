//! Actions (macros) and scripts in `$XDG_DATA_HOME/mosaix/actions`:
//! - `NAME.jsonl`: agent commands, one per line (recorded with File ›
//!   Actions › Start Recording, or written by hand); played back on the open
//!   document, each step one undo step, stopping at the first error.
//! - executable files: scripts in any language, started with
//!   `MOSAIX_SOCKET` pointing at this window (talk to it with
//!   `mosaix run --attach -` or the socket directly).

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mosaix_core::{tr, tr as trm};
use serde_json::{Value, json};

use super::App;

pub fn actions_dir() -> PathBuf {
    super::brushes::brush_dir().with_file_name("actions")
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Macro,
    Script,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub name: String,
    pub path: PathBuf,
    pub kind: Kind,
}

/// Macros and scripts in the folder, sorted by name.
pub fn list(dir: &Path) -> Vec<Entry> {
    let mut out: Vec<Entry> = std::fs::read_dir(dir)
        .map(|r| {
            r.flatten()
                .filter_map(|e| {
                    let path = e.path();
                    let md = e.metadata().ok()?;
                    if !md.is_file() {
                        return None;
                    }
                    let kind = if path.extension().is_some_and(|x| x == "jsonl") {
                        Kind::Macro
                    } else if md.permissions().mode() & 0o111 != 0 {
                        Kind::Script
                    } else {
                        return None;
                    };
                    let name = path.file_stem()?.to_string_lossy().into_owned();
                    Some(Entry { name, path, kind })
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by_key(|e| e.name.to_lowercase());
    out
}

pub fn find(dir: &Path, name: &str) -> Option<Entry> {
    list(dir).into_iter().find(|e| e.name.eq_ignore_ascii_case(name))
}

/// The steps of a macro (blank lines and `#` / `//` comments skipped).
pub fn read_macro(path: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim();
            !(t.is_empty() || t.starts_with('#') || t.starts_with("//"))
        })
        .map(|(n, l)| serde_json::from_str(l.trim()).with_context(|| format!("{}: line {}", path.display(), n + 1)))
        .collect()
}

/// Saves recorded steps as `NAME.jsonl` (an existing file gets a number).
pub fn save_macro(dir: &Path, name: &str, steps: &[Value]) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    let clean: String = name.chars().map(|c| if c.is_alphanumeric() || " -_".contains(c) { c } else { '_' }).collect();
    let clean = if clean.trim().is_empty() { "action".to_string() } else { clean.trim().to_string() };
    let mut p = dir.join(format!("{clean}.jsonl"));
    let mut n = 2;
    while p.exists() {
        p = dir.join(format!("{clean} {n}.jsonl"));
        n += 1;
    }
    let mut text = format!("# mosaix action “{clean}” – one agent command per line (see docs/agents.md)\n");
    for s in steps {
        text += &serde_json::to_string(s)?;
        text.push('\n');
    }
    std::fs::write(&p, text).with_context(|| p.display().to_string())?;
    Ok(p)
}

/// sRGB hex of a linear colour (for recorded fills).
pub fn hex(c: [f32; 3]) -> String {
    let e = |v: f32| (mosaix_core::color::linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", e(c[0]), e(c[1]), e(c[2]))
}

impl App {
    /// Adds a step to the recording (if one is running).
    pub(super) fn record(&mut self, step: Value) {
        if let Some(r) = &mut self.recording {
            tracing::info!("{}", trm!("Recorded: {step}", step = step["cmd"].as_str().unwrap_or("?")));
            r.push(step);
        }
    }

    pub(super) fn start_recording(&mut self) {
        self.recording = Some(Vec::new());
        tracing::info!("{}", tr("Recording an action – filters, adjustments, layer and selection commands are recorded"));
        self.needs_redraw = true;
    }

    /// Stops recording; with a name, saves the action.
    pub(super) fn stop_recording(&mut self, name: Option<String>) {
        let Some(steps) = self.recording.take() else { return };
        self.needs_redraw = true;
        let Some(name) = name else {
            tracing::info!("{}", tr("Recording discarded"));
            return;
        };
        if steps.is_empty() {
            tracing::warn!("{}", tr("Nothing was recorded"));
            return;
        }
        match save_macro(&actions_dir(), &name, &steps) {
            Ok(p) => tracing::info!("{}", trm!("Action saved: {path} ({n} steps)", path = p.display(), n = steps.len())),
            Err(e) => tracing::error!("{}", trm!("Action not saved: {error}", error = format!("{e:#}"))),
        }
    }

    /// Plays a macro on the open document or starts a script.
    pub(super) fn run_action(&mut self, e: &Entry) {
        match e.kind {
            Kind::Macro => {
                let steps = match read_macro(&e.path) {
                    Ok(s) => s,
                    Err(err) => {
                        tracing::error!("{err:#}");
                        return;
                    }
                };
                // Not recorded again while recording (the steps would double).
                let rec = self.recording.take();
                let t0 = std::time::Instant::now();
                for (i, s) in steps.iter().enumerate() {
                    let reply = self.agent_line(s.to_string().as_bytes());
                    let v: Value = serde_json::from_str(&reply).unwrap_or(Value::Null);
                    if v["ok"] != json!(true) {
                        tracing::error!(
                            "{}",
                            trm!(
                                "Action “{name}” stopped at step {n} ({cmd}): {error}",
                                name = e.name,
                                n = i + 1,
                                cmd = s["cmd"].as_str().unwrap_or("?"),
                                error = v["error"].as_str().unwrap_or("?")
                            )
                        );
                        self.recording = rec;
                        return;
                    }
                }
                self.recording = rec;
                if let Some(r) = &mut self.recording {
                    // Recording a macro inside a macro: the steps themselves.
                    r.extend(steps.iter().cloned());
                }
                tracing::info!(
                    "{}",
                    trm!("Action “{name}”: {n} steps in {ms} ms", name = e.name, n = steps.len(), ms = t0.elapsed().as_millis())
                );
            }
            Kind::Script => {
                let socket = self.socket.as_ref().map(|s| s.path().to_path_buf());
                let mut cmd = std::process::Command::new(&e.path);
                if let Some(s) = &socket {
                    cmd.env("MOSAIX_SOCKET", s);
                }
                if let Ok(exe) = std::env::current_exe() {
                    cmd.env("MOSAIX", exe);
                }
                cmd.current_dir(e.path.parent().unwrap_or(Path::new(".")))
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped());
                match cmd.spawn() {
                    Ok(mut child) => {
                        tracing::info!("{}", trm!("Script “{name}” started", name = e.name));
                        let name = e.name.clone();
                        // Its output goes to the log.
                        for (pipe, err) in [
                            (child.stdout.take().map(|p| Box::new(p) as Box<dyn std::io::Read + Send>), false),
                            (child.stderr.take().map(|p| Box::new(p) as Box<dyn std::io::Read + Send>), true),
                        ] {
                            let Some(pipe) = pipe else { continue };
                            let name = name.clone();
                            std::thread::spawn(move || {
                                use std::io::BufRead;
                                for line in std::io::BufReader::new(pipe).lines().map_while(Result::ok) {
                                    if err {
                                        tracing::warn!("{name}: {line}");
                                    } else {
                                        tracing::info!("{name}: {line}");
                                    }
                                }
                            });
                        }
                        std::thread::spawn(move || match child.wait() {
                            Ok(s) if s.success() => tracing::info!("{}", trm!("Script “{name}” finished", name = name)),
                            Ok(s) => tracing::warn!("{}", trm!("Script “{name}” failed ({status})", name = name, status = s)),
                            Err(e) => tracing::warn!("{name}: {e}"),
                        });
                    }
                    Err(err) => tracing::error!("{}", trm!("Script “{name}” could not start: {error}", name = e.name, error = err)),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_list_read() {
        let dir = tempfile::tempdir().unwrap();
        let steps = vec![json!({"cmd": "select_all"}), json!({"cmd": "filter_apply", "filter": {"kind": "gaussian_blur", "radius": 2}})];
        let p = save_macro(dir.path(), "Soft/blur", &steps).unwrap();
        assert_eq!(p.file_name().unwrap(), "Soft_blur.jsonl");
        let p2 = save_macro(dir.path(), "Soft/blur", &steps).unwrap();
        assert_eq!(p2.file_name().unwrap(), "Soft_blur 2.jsonl");
        std::fs::write(dir.path().join("hello.sh"), "#!/bin/sh\necho hi\n").unwrap();
        std::fs::set_permissions(dir.path().join("hello.sh"), std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(dir.path().join("notes.txt"), "x").unwrap();
        let l = list(dir.path());
        let names: Vec<_> = l.iter().map(|e| (e.name.as_str(), e.kind.clone())).collect();
        assert_eq!(names, vec![("hello", Kind::Script), ("Soft_blur", Kind::Macro), ("Soft_blur 2", Kind::Macro)]);
        assert_eq!(read_macro(&p).unwrap(), steps);
        assert_eq!(hex([1.0, 0.0, 0.21404114]), "#ff0080");
    }
}
