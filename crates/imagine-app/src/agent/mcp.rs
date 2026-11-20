//! Model Context Protocol server over stdio (newline-delimited JSON-RPC 2.0).
//!
//! `imagine mcp` runs its own headless engine; `imagine mcp --attach`
//! forwards every tool call to a running imagine window, so the user watches
//! the agent work live and can undo its steps.

use std::io::{BufRead, Write};

use base64::Engine as _;
use serde_json::{Map, Value, json};

use super::Backend;

const PROTOCOL_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "\
imagine is a GPU-accelerated layered image editor. Conventions: coordinates are document \
pixels with the origin at the top-left; colours are sRGB hex (\"#rrggbb\" or \"#rrggbbaa\") \
or [r, g, b(, a)] in 0–255; `layer` parameters default to the active layer and layer ids \
are stable. Typical workflow: document_open or document_new → document_info → edit \
(layer_add, paint_stroke, fill_rect, place_image, layer_set …) → render_preview to look at \
the result → save (.imagine keeps layers) or export (flattened png/jpg/webp/tiff). Every \
edit is a single undo step (undo/redo). When attached to a running window, the user sees \
every change immediately and may be editing too.";

fn layer_param() -> Value {
    json!({ "type": "integer", "description": "Layer id (default: active layer)" })
}

fn color_param(desc: &str) -> Value {
    json!({
        "description": desc,
        "oneOf": [
            { "type": "string", "pattern": "^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$" },
            { "type": "array", "items": { "type": "number", "minimum": 0, "maximum": 255 }, "minItems": 3, "maxItems": 4 }
        ]
    })
}

fn target_param() -> Value {
    json!({ "type": "string", "enum": ["pixels", "mask"], "description": "Edit the layer's pixels (default) or its mask (white reveals, black hides)" })
}

fn mode_param() -> Value {
    json!({ "type": "string", "enum": ["replace", "add", "subtract", "intersect"], "description": "How to combine with the current selection (default replace)" })
}

fn rect_props() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("x".into(), json!({ "type": "integer" }));
    m.insert("y".into(), json!({ "type": "integer" }));
    m.insert("width".into(), json!({ "type": "integer", "minimum": 1 }));
    m.insert("height".into(), json!({ "type": "integer", "minimum": 1 }));
    m
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str], read_only: bool) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        },
        "annotations": { "readOnlyHint": read_only }
    })
}

/// Tool catalogue. Names equal the command names of [`super::command::Command`].
pub fn tools() -> Vec<Value> {
    let blend_keys: Vec<&str> = imagine_core::BlendMode::ALL.iter().map(|m| m.key()).collect();
    let mut fill = rect_props();
    fill.insert("layer".into(), layer_param());
    fill.insert("color".into(), color_param("Fill colour (alpha component = coverage)"));
    fill.insert("opacity".into(), json!({ "type": "number", "minimum": 0, "maximum": 1 }));
    let mut clear = rect_props();
    clear.insert("layer".into(), layer_param());
    let mut prev = rect_props();
    prev.insert("max_size".into(), json!({ "type": "integer", "minimum": 16, "maximum": 4096, "description": "Longest side of the returned image in px (default 1024)" }));
    prev.insert("checkerboard".into(), json!({ "type": "boolean", "description": "Show transparency as a checkerboard (default true)" }));
    prev.insert("show_selection".into(), json!({ "type": "boolean", "description": "Tint unselected areas red so you can see the selection (default false)" }));
    let sel_rect = |what: &str| {
        let mut m = Map::new();
        for k in ["x", "y", "width", "height"] {
            m.insert(k.into(), json!({ "type": "number" }));
        }
        m.insert("mode".into(), mode_param());
        let _ = what;
        Value::Object(m)
    };
    prev.insert("output".into(), json!({ "type": "string", "description": "Optional path to also write the PNG to" }));
    vec![
        tool("document_new", "Create a new empty document (replaces the open one).",
            json!({ "width": { "type": "integer", "minimum": 1 }, "height": { "type": "integer", "minimum": 1 } }),
            &["width", "height"], false),
        tool("document_open", "Open an image (png, jpg, webp, tiff, exr) or .imagine file (replaces the open document).",
            json!({ "path": { "type": "string" } }), &["path"], false),
        tool("document_info", "Size, blend space, layer tree (top to bottom, with ids, blend modes, opacity, visibility, coarse tile bounds) and undo history.",
            json!({}), &[], true),
        tool("layer_add", "Add an empty pixel layer above the active layer and make it active. Returns its id.",
            json!({ "name": { "type": "string" } }), &[], false),
        tool("layer_duplicate", "Duplicate a layer (instant, shares pixel data). Returns the new id.",
            json!({ "layer": layer_param() }), &[], false),
        tool("layer_delete", "Delete a layer or group (the last layer cannot be deleted).",
            json!({ "layer": layer_param() }), &[], false),
        tool("layer_set", "Change layer properties.", json!({
            "layer": layer_param(),
            "name": { "type": "string" },
            "opacity": { "type": "number", "minimum": 0, "maximum": 1 },
            "blend": { "type": "string", "enum": blend_keys, "description": "Blend mode (Photoshop semantics)" },
            "visible": { "type": "boolean" }
        }), &[], false),
        tool("layer_select", "Make a layer active.", json!({ "layer": { "type": "integer" } }), &["layer"], false),
        tool("layer_move", "Move a layer up (positive steps) or down (negative) within its group.",
            json!({ "layer": layer_param(), "steps": { "type": "integer" } }), &["steps"], false),
        tool("layer_group", "Wrap a layer in a new group. Returns the group id.", json!({ "layer": layer_param() }), &[], false),
        tool("layer_ungroup", "Dissolve a group into its parent.", json!({ "layer": layer_param() }), &[], false),
        tool("set_blend_space", "Colour space for blend modes: \"srgb\" (Photoshop-like, default) or \"linear\".",
            json!({ "space": { "type": "string", "enum": ["srgb", "linear"] } }), &["space"], false),
        tool("paint_stroke",
            "Paint (or erase) one brush stroke along a polyline on a pixel layer. Dabs are spaced evenly along the path, \
             so a few points suffice for straight segments; use many points for curves.",
            json!({
                "layer": layer_param(),
                "points": { "type": "array", "minItems": 1, "description": "[[x, y] or [x, y, pressure 0–1], …]",
                            "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 3 } },
                "color": color_param("Brush colour (required unless eraser)"),
                "size": { "type": "number", "exclusiveMinimum": 0, "maximum": 5000, "description": "Diameter in px at pressure 1" },
                "hardness": { "type": "number", "minimum": 0, "maximum": 1, "description": "Edge hardness (default 0.8)" },
                "opacity": { "type": "number", "minimum": 0, "maximum": 1, "description": "Maximum coverage of the stroke (default 1)" },
                "flow": { "type": "number", "minimum": 0, "maximum": 1, "description": "Coverage per dab, builds up to opacity (default 1)" },
                "spacing": { "type": "number", "minimum": 0.01, "description": "Dab spacing as fraction of diameter (default 0.1)" },
                "eraser": { "type": "boolean" },
                "target": target_param(),
                "tip": { "type": "string", "enum": ["round", "chalk", "bristles", "spray", "rough", "charcoal"],
                         "description": "Brush tip (default round; hardness only applies to round)" },
                "angle": { "type": "number", "description": "Tip rotation in degrees" },
                "roundness": { "type": "number", "minimum": 0.05, "maximum": 1, "description": "1 = circular, smaller = flat" },
                "size_jitter": { "type": "number", "minimum": 0, "maximum": 1 },
                "angle_jitter": { "type": "number", "minimum": 0, "maximum": 180, "description": "degrees" },
                "scatter": { "type": "number", "minimum": 0, "maximum": 5, "description": "random offset across the stroke, in diameters" },
                "flow_jitter": { "type": "number", "minimum": 0, "maximum": 1 },
                "follow_direction": { "type": "boolean", "description": "rotate the tip along the stroke" },
                "smudge": { "type": "number", "minimum": 0, "maximum": 1,
                            "description": "Smudge instead of paint, with this strength (0.5 typical, 1 = drag colour all the way); colour not needed" }
            }), &["points", "size"], false),
        tool("fill", "Fill the selection (or the whole layer if nothing is selected) with a colour; respects soft selection edges.",
            json!({ "layer": layer_param(), "color": color_param("Fill colour"), "opacity": { "type": "number", "minimum": 0, "maximum": 1 }, "target": target_param() }),
            &["color"], false),
        tool("clear", "Make the selected pixels (or the whole layer if nothing is selected) transparent.",
            json!({ "layer": layer_param() }), &[], false),
        tool("gradient", "Draw a linear or radial gradient from (x0, y0) to (x1, y1) into the selection (or whole layer). color1 defaults to transparent.",
            json!({
                "layer": layer_param(),
                "kind": { "type": "string", "enum": ["linear", "radial"] },
                "x0": { "type": "number" }, "y0": { "type": "number" }, "x1": { "type": "number" }, "y1": { "type": "number" },
                "color0": color_param("Start colour"), "color1": color_param("End colour (default transparent)"),
                "opacity": { "type": "number", "minimum": 0, "maximum": 1 },
                "target": target_param()
            }), &["x0", "y0", "x1", "y1", "color0"], false),
        tool("select_rect", "Select a rectangle (anti-aliased). Painting, fill, clear and gradient only affect the selection.",
            sel_rect("rect"), &["x", "y", "width", "height"], false),
        tool("select_ellipse", "Select the ellipse inscribed in the given rectangle.", sel_rect("ellipse"), &["x", "y", "width", "height"], false),
        tool("select_polygon", "Select a closed polygon (lasso), even-odd fill.",
            json!({ "points": { "type": "array", "minItems": 3, "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 } }, "mode": mode_param() }),
            &["points"], false),
        tool("select_all", "Select the whole document.", json!({}), &[], false),
        tool("select_none", "Remove the selection (everything becomes editable).", json!({}), &[], false),
        tool("select_invert", "Invert the selection.", json!({}), &[], false),
        tool("select_feather", "Soften the selection edge by a radius in px.",
            json!({ "radius": { "type": "number", "minimum": 0, "maximum": 1000 } }), &["radius"], false),
        tool("select_magic_wand",
            "Magic wand: select pixels similar in colour to the one at (x, y). tolerance 0–255 (default 32) is the largest \
             allowed channel difference in 8-bit sRGB; contiguous (default true) keeps only the connected region; \
             sample \"layer\" (default, active layer) or \"image\" (all layers).",
            json!({ "x": { "type": "integer" }, "y": { "type": "integer" },
                    "tolerance": { "type": "number", "minimum": 0, "maximum": 255 },
                    "contiguous": { "type": "boolean" }, "antialias": { "type": "boolean" },
                    "sample": { "type": "string", "enum": ["layer", "image"] }, "mode": mode_param() }),
            &["x", "y"], false),
        tool("select_color_range",
            "Select all pixels of a colour anywhere in the image (soft edges), given as color or picked at (x, y). \
             fuzziness 0–200 (default 40) widens the range.",
            json!({ "color": color_param("colour to select (alternative to x/y)"), "x": { "type": "integer" }, "y": { "type": "integer" },
                    "fuzziness": { "type": "number", "minimum": 0, "maximum": 200 },
                    "sample": { "type": "string", "enum": ["layer", "image"] }, "mode": mode_param() }),
            &[], false),
        tool("layer_mask_add", "Add a mask to a layer or group: from the selection (default when one exists) or revealing everything.",
            json!({ "layer": layer_param(), "from_selection": { "type": "boolean" } }), &[], false),
        tool("layer_mask_delete", "Remove a layer's mask.", json!({ "layer": layer_param() }), &[], false),
        tool("layer_mask_set", "Enable or disable a layer's mask.",
            json!({ "layer": layer_param(), "enabled": { "type": "boolean" } }), &["enabled"], false),
        tool("layer_mask_apply", "Bake a layer's mask into its pixels and remove the mask.", json!({ "layer": layer_param() }), &[], false),
        tool("layer_translate", "Move a whole layer (pixels and mask) by whole pixels, ignoring the selection. Content outside the canvas is kept.",
            json!({ "layer": layer_param(), "dx": { "type": "integer" }, "dy": { "type": "integer" } }), &["dx", "dy"], false),
        tool("layer_transform",
            "Scale, rotate and/or move a layer (with its mask) around a pivot (default: centre of its content). With a selection, \
             only the selected pixels move and the selection follows (like Photoshop's free transform); set use_selection=false \
             to transform the whole layer anyway. Order: scale, rotate, then move by dx/dy. Returns the new content bounds.",
            json!({
                "layer": layer_param(),
                "scale": { "type": "number", "description": "uniform factor, e.g. 0.5" },
                "scale_x": { "type": "number" }, "scale_y": { "type": "number", "description": "negative values mirror" },
                "rotate": { "type": "number", "description": "degrees, clockwise" },
                "dx": { "type": "number" }, "dy": { "type": "number" },
                "pivot_x": { "type": "number" }, "pivot_y": { "type": "number" },
                "interp": { "type": "string", "enum": ["bicubic", "bilinear", "nearest"], "description": "default bicubic; nearest for pixel art" },
                "use_selection": { "type": "boolean" }
            }), &[], false),
        tool("layer_bounds", "Pixel-exact bounding box of a layer's visible content (within the selection, if any); null if empty.",
            json!({ "layer": layer_param() }), &[], true),
        tool("crop", "Crop the canvas to a rectangle (may extend beyond the canvas to enlarge it). Pixels outside are kept, not deleted. Drops the selection.",
            json!({ "x": { "type": "integer" }, "y": { "type": "integer" }, "width": { "type": "integer" }, "height": { "type": "integer" } }),
            &["x", "y", "width", "height"], false),
        tool("canvas_size", "Change the canvas size without scaling; anchor decides where the existing image stays (default center).",
            json!({ "width": { "type": "integer" }, "height": { "type": "integer" },
                    "anchor": { "type": "string", "enum": ["center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"] } }),
            &["width", "height"], false),
        tool("image_resize", "Scale the whole image (all layers and masks). Give width and/or height; a missing one keeps the aspect ratio.",
            json!({ "width": { "type": "integer" }, "height": { "type": "integer" },
                    "interp": { "type": "string", "enum": ["bicubic", "bilinear", "nearest"] } }), &[], false),
        tool("fill_rect", "Fill a rectangle on a pixel layer with a colour (normal blending onto existing pixels).",
            Value::Object(fill), &["x", "y", "width", "height", "color"], false),
        tool("clear_rect", "Make a rectangle on a pixel layer fully transparent.",
            Value::Object(clear), &["x", "y", "width", "height"], false),
        tool("place_image", "Load an image file into a new layer, top-left corner at (x, y). Returns the layer id.",
            json!({ "path": { "type": "string" }, "x": { "type": "integer" }, "y": { "type": "integer" }, "name": { "type": "string" } }),
            &["path"], false),
        tool("sample_color", "Colour of the flattened image at a document pixel (sRGB hex + alpha).",
            json!({ "x": { "type": "number" }, "y": { "type": "number" } }), &["x", "y"], true),
        tool("render_preview",
            "Render the flattened image (or the region x/y/width/height) as a PNG you can look at. Use it to check your work.",
            Value::Object(prev), &[], true),
        tool("undo", "Undo the last edit (yours or the user's). Returns its label.", json!({}), &[], false),
        tool("redo", "Redo the last undone edit.", json!({}), &[], false),
        tool("save", "Save as .imagine (keeps layers; incremental when saving to the same file again).",
            json!({ "path": { "type": "string", "description": "Default: next to the source file" } }), &[], false),
        tool("export", "Export the flattened image; format by extension: png, jpg, webp, tif.",
            json!({ "path": { "type": "string" } }), &["path"], false),
    ]
}

/// Handles one JSON-RPC message. Returns the response, or `None` for notifications.
pub fn handle(backend: &mut dyn Backend, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned().filter(|v| !v.is_null());
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result: Result<Value, (i64, String)> = match method {
        "initialize" => {
            let requested = params.get("protocolVersion").and_then(Value::as_str).unwrap_or("");
            let version = PROTOCOL_VERSIONS.iter().find(|v| **v == requested).unwrap_or(&PROTOCOL_VERSIONS[0]);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "imagine", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            if !tools().iter().any(|t| t["name"] == name) {
                Err((-32602, format!("unknown tool {name:?}")))
            } else {
                let mut cmd = match params.get("arguments") {
                    Some(Value::Object(m)) => m.clone(),
                    _ => Map::new(),
                };
                cmd.insert("cmd".into(), name.into());
                Ok(match backend.call(Value::Object(cmd)) {
                    Ok(out) => {
                        let mut content =
                            vec![json!({ "type": "text", "text": serde_json::to_string_pretty(&out.result).unwrap() })];
                        if let Some(png) = out.image_png {
                            content.push(json!({
                                "type": "image",
                                "mimeType": "image/png",
                                "data": base64::engine::general_purpose::STANDARD.encode(png),
                            }));
                        }
                        json!({ "content": content })
                    }
                    Err(e) => json!({ "content": [{ "type": "text", "text": format!("{e:#}") }], "isError": true }),
                })
            }
        }
        _ if id.is_none() => return None, // notifications (e.g. notifications/initialized)
        other => Err((-32601, format!("method not found: {other}"))),
    };
    let id = id?;
    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } }),
    })
}

/// Serves MCP on stdin/stdout until stdin closes.
pub fn serve(backend: &mut dyn Backend) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle(backend, &msg),
            Err(e) => Some(json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } })),
        };
        if let Some(r) = response {
            serde_json::to_writer(&mut stdout, &r)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::CallOutput;

    /// Records calls instead of executing them.
    struct Echo(Vec<Value>);

    impl Backend for Echo {
        fn call(&mut self, cmd: Value) -> anyhow::Result<CallOutput> {
            self.0.push(cmd.clone());
            if cmd["cmd"] == "undo" {
                anyhow::bail!("nothing to undo");
            }
            Ok(CallOutput { result: json!({ "ok": true }), image_png: Some(vec![1, 2, 3]) })
        }
    }

    #[test]
    fn handshake_list_and_call() {
        let mut b = Echo(Vec::new());
        let init = handle(&mut b, &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}})).unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(handle(&mut b, &json!({"jsonrpc":"2.0","method":"notifications/initialized"})).is_none());

        let list = handle(&mut b, &json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).unwrap();
        let tools = list["result"]["tools"].as_array().unwrap();
        assert!(tools.len() >= 37);
        // Every tool schema parses as a command when given its required fields? At least: names are unique.
        let mut names: Vec<_> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), tools.len());

        let r = handle(&mut b, &json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"render_preview","arguments":{"max_size":64}}})).unwrap();
        assert_eq!(b.0[0], json!({"cmd": "render_preview", "max_size": 64}));
        let content = r["result"]["content"].as_array().unwrap();
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["data"], "AQID");

        let r = handle(&mut b, &json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"undo"}})).unwrap();
        assert_eq!(r["result"]["isError"], true);
        let r = handle(&mut b, &json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"rm_rf"}})).unwrap();
        assert_eq!(r["error"]["code"], -32602);
        let r = handle(&mut b, &json!({"jsonrpc":"2.0","id":6,"method":"bogus"})).unwrap();
        assert_eq!(r["error"]["code"], -32601);
    }

    #[test]
    fn tool_names_match_commands() {
        use crate::agent::command::Command;
        // Minimal valid arguments per tool must parse into the command of the same name.
        let samples = json!({
            "document_new": {"width": 10, "height": 10}, "document_open": {"path": "x.png"}, "document_info": {},
            "layer_add": {}, "layer_duplicate": {}, "layer_delete": {}, "layer_set": {}, "layer_select": {"layer": 1},
            "layer_move": {"steps": 1}, "layer_group": {}, "layer_ungroup": {}, "set_blend_space": {"space": "linear"},
            "paint_stroke": {"points": [[0, 0]], "size": 3, "color": "#000000"},
            "fill_rect": {"x": 0, "y": 0, "width": 1, "height": 1, "color": "#fff"},
            "clear_rect": {"x": 0, "y": 0, "width": 1, "height": 1}, "place_image": {"path": "x.png"},
            "sample_color": {"x": 0, "y": 0}, "render_preview": {}, "undo": {}, "redo": {}, "save": {}, "export": {"path": "x.png"},
            "fill": {"color": "#ff0000"}, "clear": {}, "gradient": {"x0": 0, "y0": 0, "x1": 1, "y1": 1, "color0": "#000000"},
            "select_rect": {"x": 0, "y": 0, "width": 1, "height": 1}, "select_ellipse": {"x": 0, "y": 0, "width": 1, "height": 1},
            "select_polygon": {"points": [[0, 0], [1, 0], [0, 1]]}, "select_all": {}, "select_none": {}, "select_invert": {},
            "select_feather": {"radius": 2}, "layer_mask_add": {}, "layer_mask_delete": {}, "layer_mask_set": {"enabled": true},
            "layer_mask_apply": {}, "layer_translate": {"dx": 1, "dy": 0}
        });
        let mut samples = samples;
        let more = json!({
            "layer_transform": {"scale": 2}, "layer_bounds": {}, "crop": {"x": 0, "y": 0, "width": 1, "height": 1},
            "canvas_size": {"width": 1, "height": 1}, "image_resize": {"width": 1},
            "select_magic_wand": {"x": 0, "y": 0}, "select_color_range": {"color": "#ff0000"}
        });
        samples.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
        for t in tools() {
            let name = t["name"].as_str().unwrap();
            let mut args = samples[name].as_object().unwrap_or_else(|| panic!("no sample for {name}")).clone();
            args.insert("cmd".into(), name.into());
            let cmd = Command::parse(Value::Object(args)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(cmd.name(), name);
            // Every schema property must be accepted by the command (deny_unknown_fields
            // would reject it otherwise): try each one with a plausible value.
            for (prop, schema) in t["inputSchema"]["properties"].as_object().unwrap() {
                let mut a = samples[name].as_object().unwrap().clone();
                a.insert("cmd".into(), name.into());
                if !a.contains_key(prop) {
                    let v = match (prop.as_str(), schema["type"].as_str()) {
                        (p, _) if p.starts_with("color") => json!("#123456"),
                        ("target", _) => json!("pixels"),
                        ("mode", _) => json!("add"),
                        ("kind", _) => json!("radial"),
                        ("blend", _) => json!("screen"),
                        ("space", _) => json!("linear"),
                        ("points", _) => json!([[0, 0], [1, 1], [2, 0]]),
                        ("interp", _) => json!("nearest"),
                        ("anchor", _) => json!("top-left"),
                        ("sample", _) => json!("image"),
                        ("tip", _) => json!("chalk"),
                        (_, Some("boolean")) => json!(true),
                        (_, Some("string")) => json!("x"),
                        _ => json!(1),
                    };
                    a.insert(prop.clone(), v);
                }
                Command::parse(Value::Object(a)).unwrap_or_else(|e| panic!("{name}.{prop}: {e}"));
            }
        }
    }
}
