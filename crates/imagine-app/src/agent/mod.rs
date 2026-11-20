//! Agent interface: the command set plus three ways to reach it –
//! MCP over stdio (`imagine mcp`), JSON-lines scripts (`imagine run`) and a
//! live socket into a running window (`--attach`).

pub mod command;
pub mod ipc;
pub mod mcp;

use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::Context;
use imagine_gpu::GpuContext;
use serde_json::{Value, json};

use crate::session::Session;
use command::{Command, Ctx};

/// Result of a call, independent of where it was executed.
pub struct CallOutput {
    pub result: Value,
    pub image_png: Option<Vec<u8>>,
}

/// Executes commands: locally (own headless engine) or in a running window.
pub trait Backend {
    fn call(&mut self, cmd: Value) -> anyhow::Result<CallOutput>;
}

pub struct Local {
    pub gpu: GpuContext,
    pub session: Option<Session>,
}

impl Local {
    pub fn new(open: Option<&Path>) -> anyhow::Result<Self> {
        let gpu = pollster::block_on(GpuContext::new(GpuContext::create_instance(), None))?;
        let mut me = Self { gpu, session: None };
        if let Some(p) = open {
            me.call(json!({ "cmd": "document_open", "path": p }))?;
        }
        Ok(me)
    }
}

impl Backend for Local {
    fn call(&mut self, cmd: Value) -> anyhow::Result<CallOutput> {
        let cmd = Command::parse(cmd)?;
        let mut ctx = Ctx { gpu: &self.gpu, format: wgpu::TextureFormat::Rgba8UnormSrgb, session: &mut self.session };
        let out = command::execute(&mut ctx, cmd)?;
        Ok(CallOutput { result: out.result, image_png: out.image_png })
    }
}

/// `imagine run`: executes a JSON-lines script (`-` = stdin), printing one
/// result line per command. Stops at the first error (exit code 1).
pub fn run_script(backend: &mut dyn Backend, script: &Path) -> anyhow::Result<bool> {
    let input: Box<dyn BufRead> = if script == Path::new("-") {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(std::io::BufReader::new(
            std::fs::File::open(script).with_context(|| format!("cannot open {}", script.display()))?,
        ))
    };
    let mut stdout = std::io::stdout().lock();
    for (n, line) in input.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }
        let cmd: Value = serde_json::from_str(trimmed).with_context(|| format!("line {}: invalid JSON", n + 1))?;
        let name = cmd.get("cmd").cloned().unwrap_or(Value::Null);
        let out = match backend.call(cmd) {
            Ok(o) => json!({ "line": n + 1, "cmd": name, "ok": true, "result": o.result }),
            Err(e) => {
                writeln!(stdout, "{}", json!({ "line": n + 1, "cmd": name, "ok": false, "error": format!("{e:#}") }))?;
                return Ok(false);
            }
        };
        writeln!(stdout, "{out}")?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(b: &mut Local, v: Value) -> CallOutput {
        let name = v["cmd"].clone();
        b.call(v).unwrap_or_else(|e| panic!("{name}: {e:#}"))
    }

    /// A complete agent session against the real engine.
    #[test]
    fn agent_workflow() {
        let mut b = match Local::new(None) {
            Ok(b) => b,
            Err(e) => return eprintln!("skipping: {e}"),
        };
        assert!(b.call(json!({"cmd": "document_info"})).is_err(), "no document yet");
        call(&mut b, json!({"cmd": "document_new", "width": 600, "height": 400}));

        let id = call(&mut b, json!({"cmd": "layer_add", "name": "Rot"})).result["layer"].as_u64().unwrap();
        call(&mut b, json!({"cmd": "fill_rect", "x": 100, "y": 100, "width": 300, "height": 200, "color": "#ff0000"}));
        let c = call(&mut b, json!({"cmd": "sample_color", "x": 250, "y": 200})).result;
        assert_eq!(c["hex"], "#ff0000");
        assert_eq!(c["alpha"], 1.0);
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 50, "y": 50})).result["alpha"], 0.0);

        // Multiply a blue stroke over the red: red × blue = black where both are opaque.
        let blue = call(&mut b, json!({"cmd": "layer_add", "name": "Blau"})).result["layer"].as_u64().unwrap();
        call(&mut b, json!({"cmd": "paint_stroke", "points": [[150, 200], [350, 200]], "size": 40, "hardness": 1, "color": "#0000ff"}));
        call(&mut b, json!({"cmd": "layer_set", "layer": blue, "blend": "multiply"}));
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 250, "y": 200})).result["hex"], "#000000");
        assert!(b.call(json!({"cmd": "layer_set", "blend": "sparkle"})).is_err());

        // Undo the blend change → plain blue on top.
        assert_eq!(call(&mut b, json!({"cmd": "undo"})).result["undone"], "Ebeneneigenschaften");
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 250, "y": 200})).result["hex"], "#0000ff");

        // Preview: whole document scaled to max 200 px, as PNG.
        let p = call(&mut b, json!({"cmd": "render_preview", "max_size": 200}));
        let img = image::load_from_memory(p.image_png.as_ref().unwrap()).unwrap();
        assert_eq!((img.width(), img.height()), (200, 133));
        let px = img.to_rgba8().get_pixel(83, 66).0; // doc (250, 200) → blue
        assert!(px[2] > 200 && px[0] < 50, "{px:?}");

        // Place an image, erase part of it, check the layer tree.
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("dot.png");
        image::RgbaImage::from_pixel(20, 10, image::Rgba([0, 255, 0, 255])).save(&png).unwrap();
        let placed = call(&mut b, json!({"cmd": "place_image", "path": png, "x": -5, "y": 390})).result;
        assert_eq!(placed["width"], 20);
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 0, "y": 395})).result["hex"], "#00ff00");
        call(&mut b, json!({"cmd": "clear_rect", "x": 0, "y": 390, "width": 5, "height": 10}));
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 2, "y": 395})).result["alpha"], 0.0);

        let info = call(&mut b, json!({"cmd": "document_info"})).result;
        let layers = info["layers_top_to_bottom"].as_array().unwrap();
        let names: Vec<_> = layers.iter().map(|l| l["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["dot", "Blau", "Rot", "Hintergrund"]);
        assert_eq!(layers[2]["id"], id);

        // Save, reopen, verify.
        let file = dir.path().join("agent.imagine");
        call(&mut b, json!({"cmd": "save", "path": file}));
        call(&mut b, json!({"cmd": "document_open", "path": file}));
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 250, "y": 200})).result["hex"], "#0000ff");
        assert_eq!(call(&mut b, json!({"cmd": "document_info"})).result["layers_top_to_bottom"].as_array().unwrap().len(), 4);
    }

    /// Selections, masks, gradients – and that all of it survives save/load.
    #[test]
    fn agent_selection_and_mask_workflow() {
        let mut b = match Local::new(None) {
            Ok(b) => b,
            Err(e) => return eprintln!("skipping: {e}"),
        };
        call(&mut b, json!({"cmd": "document_new", "width": 800, "height": 400}));
        call(&mut b, json!({"cmd": "fill", "color": "#ffffff"}));
        let top = call(&mut b, json!({"cmd": "layer_add", "name": "Verlauf"})).result["layer"].clone();
        call(&mut b, json!({"cmd": "gradient", "x0": 0, "y0": 0, "x1": 800, "y1": 0, "color0": "#ff0000", "color1": "#0000ff"}));
        let rgb = call(&mut b, json!({"cmd": "sample_color", "x": 1, "y": 200})).result["rgb"].clone();
        assert!(rgb[0].as_u64().unwrap() >= 250 && rgb[2].as_u64().unwrap() <= 5, "gradient starts red: {rgb}");

        // Mask: only an ellipse of the gradient layer stays visible.
        let sel = call(&mut b, json!({"cmd": "select_ellipse", "x": 200, "y": 100, "width": 400, "height": 200})).result;
        assert!(sel["selection"]["edge_tiles"].as_u64().unwrap() > 0);
        call(&mut b, json!({"cmd": "layer_mask_add", "layer": top}));
        call(&mut b, json!({"cmd": "select_none"}));
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 10, "y": 10})).result["hex"], "#ffffff", "masked out");
        assert_ne!(call(&mut b, json!({"cmd": "sample_color", "x": 400, "y": 200})).result["hex"], "#ffffff", "inside ellipse");

        // Paint black into the mask at the centre → hidden there.
        call(&mut b, json!({"cmd": "paint_stroke", "points": [[400, 200]], "size": 40, "hardness": 1, "color": "#000000", "target": "mask"}));
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 400, "y": 200})).result["hex"], "#ffffff");

        // Selection-limited fill + preview with selection tint.
        call(&mut b, json!({"cmd": "select_rect", "x": 0, "y": 0, "width": 100, "height": 100}));
        call(&mut b, json!({"cmd": "select_rect", "x": 50, "y": 50, "width": 100, "height": 100, "mode": "add"}));
        let bg = call(&mut b, json!({"cmd": "document_info"})).result["layers_top_to_bottom"][1]["id"].clone();
        call(&mut b, json!({"cmd": "fill", "layer": bg, "color": "#00ff00"}));
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 120, "y": 120})).result["hex"], "#00ff00");
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 120, "y": 20})).result["hex"], "#ffffff");
        let p = call(&mut b, json!({"cmd": "render_preview", "max_size": 200, "show_selection": true}));
        let img = image::load_from_memory(p.image_png.as_ref().unwrap()).unwrap().to_rgba8();
        let outside = img.get_pixel(150, 90).0; // doc (600, 360): unselected → reddish
        assert!(outside[0] > outside[1] + 50, "tinted: {outside:?}");

        // Save with mask + selection, reopen, verify.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("masken.imagine");
        call(&mut b, json!({"cmd": "save", "path": file}));
        call(&mut b, json!({"cmd": "document_open", "path": file}));
        let info = call(&mut b, json!({"cmd": "document_info"})).result;
        assert_eq!(info["layers_top_to_bottom"][0]["mask"]["enabled"], true);
        assert!(info["selection"]["tiles"].as_u64().unwrap() > 0, "selection restored");
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 400, "y": 200})).result["hex"], "#ffffff");
        assert_eq!(call(&mut b, json!({"cmd": "sample_color", "x": 10, "y": 10})).result["hex"], "#00ff00");

        // Move the masked layer and bake its mask.
        call(&mut b, json!({"cmd": "layer_translate", "layer": top, "dx": 100, "dy": 0}));
        call(&mut b, json!({"cmd": "layer_mask_apply", "layer": top}));
        let info = call(&mut b, json!({"cmd": "document_info"})).result;
        assert!(info["layers_top_to_bottom"][0].get("mask").is_none());
        assert!(b.call(json!({"cmd": "fill", "color": "#000", "target": "mask"})).is_err(), "no mask anymore");
    }
}
