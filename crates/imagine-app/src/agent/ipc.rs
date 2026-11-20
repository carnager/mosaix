//! Live control of a running imagine window over a Unix socket.
//!
//! Each window listens on `$XDG_RUNTIME_DIR/imagine/<pid>.sock` (a per-user,
//! mode-0700 directory). Protocol: one JSON command per line
//! (`{"id": 1, "cmd": "fill_rect", …}`), one JSON response per line
//! (`{"id": 1, "ok": true, "result": …, "image_png": "<base64>"}` or
//! `{"id": 1, "ok": false, "error": "…"}`).

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use base64::Engine as _;
use serde_json::{Value, json};

use super::command::Output;
use super::{Backend, CallOutput};

pub fn socket_dir() -> PathBuf {
    match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(d) => PathBuf::from(d).join("imagine"),
        None => std::env::temp_dir().join(format!(
            "imagine-{}",
            std::env::var("USER").unwrap_or_default()
        )),
    }
}

/// Removes the socket file when the window closes.
pub struct SocketGuard(PathBuf);

impl SocketGuard {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Removes sockets of windows that are gone (crashed or killed).
fn remove_stale(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "sock") && UnixStream::connect(&p).is_err() {
            let _ = std::fs::remove_file(&p);
        }
    }
}

pub fn bind() -> anyhow::Result<(UnixListener, SocketGuard)> {
    let dir = socket_dir();
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)?;
    remove_stale(&dir);
    let path = dir.join(format!("{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let listener =
        UnixListener::bind(&path).with_context(|| format!("cannot bind {}", path.display()))?;
    listener.set_nonblocking(true)?;
    Ok((listener, SocketGuard(path)))
}

/// Connects to `explicit`, `$IMAGINE_SOCKET`, or the most recently started window.
pub fn connect(explicit: Option<&Path>) -> anyhow::Result<UnixStream> {
    if let Some(p) = explicit
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("IMAGINE_SOCKET").map(PathBuf::from))
    {
        return UnixStream::connect(&p)
            .with_context(|| format!("cannot connect to {}", p.display()));
    }
    let mut socks: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(socket_dir())
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "sock"))
                .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
                .collect()
        })
        .unwrap_or_default();
    socks.sort_by_key(|s| std::cmp::Reverse(s.0));
    for (_, p) in socks {
        match UnixStream::connect(&p) {
            Ok(s) => return Ok(s),
            Err(_) => {
                let _ = std::fs::remove_file(&p); // stale socket of a crashed window
            }
        }
    }
    bail!("no running imagine window found (start one with `imagine`)")
}

/// Formats the response line for one executed request.
pub fn response(id: Value, result: anyhow::Result<Output>) -> String {
    let v = match result {
        Ok(out) => {
            let mut v = json!({ "id": id, "ok": true, "result": out.result });
            if let Some(png) = out.image_png {
                v["image_png"] = base64::engine::general_purpose::STANDARD.encode(png).into();
            }
            v
        }
        Err(e) => json!({ "id": id, "ok": false, "error": format!("{e:#}") }),
    };
    let mut s = v.to_string();
    s.push('\n');
    s
}

/// Client side: forwards commands to a running window.
pub struct Client {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    next_id: u64,
}

impl Client {
    pub fn connect(explicit: Option<&Path>) -> anyhow::Result<Self> {
        let s = connect(explicit)?;
        Ok(Self {
            reader: BufReader::new(s.try_clone()?),
            writer: s,
            next_id: 1,
        })
    }
}

impl Backend for Client {
    fn call(&mut self, mut cmd: Value) -> anyhow::Result<CallOutput> {
        let id = self.next_id;
        self.next_id += 1;
        cmd["id"] = id.into();
        let mut line = cmd.to_string();
        line.push('\n');
        self.writer
            .write_all(line.as_bytes())
            .context("window closed")?;
        let mut resp = String::new();
        if self.reader.read_line(&mut resp)? == 0 {
            bail!("window closed");
        }
        let v: Value = serde_json::from_str(&resp)?;
        if v["ok"] != true {
            bail!("{}", v["error"].as_str().unwrap_or("unknown error"));
        }
        let image_png = match v["image_png"].as_str() {
            Some(b) => Some(base64::engine::general_purpose::STANDARD.decode(b)?),
            None => None,
        };
        Ok(CallOutput {
            result: v["result"].clone(),
            image_png,
        })
    }
}
