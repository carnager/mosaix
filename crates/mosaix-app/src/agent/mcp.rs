//! Model Context Protocol server over stdio (newline-delimited JSON-RPC 2.0).
//!
//! `mosaix mcp` runs its own headless engine; `mosaix mcp --attach`
//! forwards every tool call to a running mosaix window, so the user watches
//! the agent work live and can undo its steps.

use std::io::{BufRead, Write};

use base64::Engine as _;
use serde_json::{Map, Value, json};

use super::Backend;

const PROTOCOL_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "\
mosaix is a GPU-accelerated layered image editor. Conventions: coordinates are document \
pixels with the origin at the top-left; colours are sRGB hex (\"#rrggbb\" or \"#rrggbbaa\") \
or [r, g, b(, a)] in 0–255; `layer` parameters default to the active layer and layer ids \
are stable. Typical workflow: document_open or document_new → document_info → edit \
(layer_add, paint_stroke, fill_rect, place_image, layer_set …) → render_preview to look at \
the result → save (.mosaix keeps layers) or export (flattened png/jpg/webp/tiff, or layered psd). Every \
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

fn tool(
    name: &str,
    description: &str,
    properties: Value,
    required: &[&str],
    read_only: bool,
) -> Value {
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
    let blend_keys: Vec<&str> = mosaix_core::BlendMode::ALL
        .iter()
        .map(|m| m.key())
        .collect();
    let mut fill = rect_props();
    fill.insert("layer".into(), layer_param());
    fill.insert(
        "color".into(),
        color_param("Fill colour (alpha component = coverage)"),
    );
    fill.insert(
        "opacity".into(),
        json!({ "type": "number", "minimum": 0, "maximum": 1 }),
    );
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
    prev.insert(
        "output".into(),
        json!({ "type": "string", "description": "Optional path to also write the PNG to" }),
    );
    vec![
        tool(
            "document_new",
            "Create a new empty document (replaces the open one).",
            json!({ "width": { "type": "integer", "minimum": 1 }, "height": { "type": "integer", "minimum": 1 } }),
            &["width", "height"],
            false,
        ),
        tool(
            "document_open",
            "Open an image (psd/psb with layers, png, jpg, webp, avif, jxl, tiff, exr, camera RAW such as cr2/cr3/nef/arw/raf/orf/rw2/dng) or .mosaix file (replaces the open document). RAW files are developed to scene-linear sRGB; `raw` sets white balance and exposure (see raw_info for the as-shot values).",
            json!({ "path": { "type": "string" },
                "raw": { "type": "object", "description": "RAW development: {white_balance: {mode: as_shot|auto|daylight} or {mode: custom, temperature: kelvin, tint: -150…150 (+ = magenta)}, exposure: stops, highlights: reconstruct|clip, noise: {luminance: 0–100, color: 0–100}, lens: true|false, tone: {contrast, highlights, shadows, vibrance, saturation: −100…100}}",
                    "properties": {
                        "white_balance": { "type": "object", "properties": {
                            "mode": { "type": "string", "enum": ["as_shot", "auto", "daylight", "custom"] },
                            "temperature": { "type": "number" }, "tint": { "type": "number" } }, "required": ["mode"] },
                        "exposure": { "type": "number" },
                        "highlights": { "type": "string", "enum": ["reconstruct", "clip"] },
                        "noise": { "type": "object", "properties": { "luminance": { "type": "number" }, "color": { "type": "number" } } },
                        "lens": { "type": "boolean" } } } }),
            &["path"],
            false,
        ),
        tool(
            "raw_set",
            "Re-develop a RAW layer (opened or placed camera RAW files stay RAW layers): white balance, exposure, highlights (reconstruct: blown channels estimated, values may exceed 1 so lower exposure recovers detail; clip: neutral white), wavelet noise reduction (luminance/color 0–100, default 0/25; slow: ~1 s per 24 MP), tone (contrast, highlights — negative rolls bright and reconstructed areas off softly, shadows, vibrance, saturation; each −100…100, given fields replace the current ones), lens correction (distortion, colour fringes, vignetting from the system's lensfun database; on by default when a profile matches — see lens_profile). Returns the settings and the resulting temperature/tint. layer_rasterize turns it into plain pixels.",
            json!({ "layer": layer_param(),
                "white_balance": { "type": "object", "description": "{mode: as_shot|auto|daylight} or {mode: custom, temperature: kelvin, tint: -150…150}",
                    "properties": { "mode": { "type": "string", "enum": ["as_shot", "auto", "daylight", "custom"] },
                        "temperature": { "type": "number" }, "tint": { "type": "number" } }, "required": ["mode"] },
                "exposure": { "type": "number", "description": "stops" },
                "highlights": { "type": "string", "enum": ["reconstruct", "clip"] },
                "noise": { "type": "object", "properties": { "luminance": { "type": "number" }, "color": { "type": "number" } } },
                "lens": { "type": "boolean", "description": "lens correction" },
                "tone": { "type": "object", "properties": {
                    "contrast": { "type": "number" }, "highlights": { "type": "number" }, "shadows": { "type": "number" },
                    "vibrance": { "type": "number" }, "saturation": { "type": "number" } } } }),
            &[],
            false,
        ),
        tool(
            "select_subject",
            "Select the main subject of the picture with a local AI model (BiRefNet, runs on this computer; ~5 s). mode: replace|add|subtract|intersect. Needs the model: ai_model_download {model: subject} once, after asking the user.",
            json!({ "mode": { "type": "string", "enum": ["replace", "add", "subtract", "intersect"] } }),
            &[],
            false,
        ),
        tool(
            "select_object",
            "Select the object at points with a local AI model (SAM 2.1; first call on an image ~1–2 s, further ones ~0.1 s): points [[x, y], …] on the object, optional negative [[x, y], …] on what does not belong to it. mode: replace|add|subtract|intersect. Check with render_preview show_selection. Needs ai_model_download {model: object} once.",
            json!({ "points": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } },
                "negative": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } },
                "mode": { "type": "string", "enum": ["replace", "add", "subtract", "intersect"] } }),
            &["points"],
            false,
        ),
        tool(
            "remove_background",
            "Hide the background of a layer: a layer mask from the main subject of the visible picture (local AI, non-destructive).",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "remove_area",
            "Remove what is selected: the area is replaced by plausible surroundings generated with a local AI model (LaMa; ~1–3 s). The selection is grown by expand px first (default 4) so the object's fringe goes too. all_layers: learn from the whole visible image (to work on an empty layer). Needs ai_model_download {model: remove} once.",
            json!({ "layer": layer_param(), "all_layers": { "type": "boolean" }, "expand": { "type": "integer" } }),
            &[],
            false,
        ),
        tool(
            "image_upscale",
            "Enlarge the image 2× or 4× with local AI (Real-ESRGAN: adds plausible detail instead of blur; ~10 s per megapixel of the source on the CPU). Every pixel layer is enlarged; text, shapes, masks scale as with image_resize. Needs ai_model_download {model: upscale} once.",
            json!({ "factor": { "type": "integer", "enum": [2, 4] } }),
            &[],
            false,
        ),
        tool(
            "ai_model_download",
            "Download a local AI model from huggingface.co (subject: BiRefNet lite, 224 MB, MIT; object: SAM 2.1 small, 184 MB, Apache-2.0; remove: LaMa, 208 MB, Apache-2.0; upscale: Real-ESRGAN x4v3, 5 MB, BSD-3-Clause). Ask the user before calling this.",
            json!({ "model": { "type": "string", "enum": ["subject", "object", "remove", "upscale"] } }),
            &["model"],
            false,
        ),
        tool(
            "action_list",
            "The user's actions (recorded macros: .jsonl files of these commands) and scripts, and the folder they live in. You can write a macro there yourself (one command per line, no layer ids).",
            json!({}),
            &[],
            true,
        ),
        tool(
            "action_run",
            "Play a macro (action) on the open document; each step is one undo step, stops at the first failing step.",
            json!({ "name": { "type": "string" } }),
            &["name"],
            false,
        ),
        tool(
            "filter_plugin_list",
            "The GPU filter plugins (WGSL files): names, descriptions, parameters, load errors, the plugin folder and the file format with all helper functions.",
            json!({}),
            &[],
            true,
        ),
        tool(
            "filter_plugin_install",
            "Write your own GPU filter: checks a WGSL filter plugin (header + fn process(p: vec2<i32>) -> vec4<f32>, format in filter_plugin_list) and saves it to the plugin folder; compile errors name the line. check_only=true only checks. Then use it with filter_apply {filter: {kind: plugin, name}}.",
            json!({ "source": { "type": "string" }, "check_only": { "type": "boolean" } }),
            &["source"],
            false,
        ),
        tool(
            "raw_info",
            "Camera, lens, shot data (ISO, shutter, aperture, focal length) and the as-shot white balance (temperature/tint) of a camera RAW file, without opening it.",
            json!({ "path": { "type": "string" } }),
            &["path"],
            true,
        ),
        tool(
            "document_info",
            "Size, blend space, layer tree (top to bottom, with ids, blend modes, opacity, visibility, coarse tile bounds) and undo history.",
            json!({}),
            &[],
            true,
        ),
        tool(
            "layer_add",
            "Add an empty pixel layer above the active layer and make it active. Returns its id.",
            json!({ "name": { "type": "string" } }),
            &[],
            false,
        ),
        tool(
            "layer_duplicate",
            "Duplicate a layer (instant, shares pixel data). Returns the new id.",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "layer_delete",
            "Delete a layer or group (the last layer cannot be deleted).",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "layer_set",
            "Change layer properties.",
            json!({
                "layer": layer_param(),
                "name": { "type": "string" },
                "opacity": { "type": "number", "minimum": 0, "maximum": 1 },
                "fill_opacity": { "type": "number", "minimum": 0, "maximum": 1, "description": "Photoshop's Fill: fades the layer's own pixels but not its layer style (0 = only the effects show)" },
                "blend": { "type": "string", "enum": blend_keys, "description": "Blend mode (Photoshop semantics)" },
                "visible": { "type": "boolean" },
                "clip": { "type": "boolean", "description": "Clipping mask: the layer shows only where the layers below it (down to the first unclipped one) have pixels; that base layer's opacity and blend mode apply to the group" },
                "pass_through": { "type": "boolean", "description": "Groups only: Pass Through (Photoshop's default for groups) – children blend directly with what is below the group; false = isolated group using its own blend mode" }
            }),
            &[],
            false,
        ),
        tool(
            "layer_select",
            "Make a layer active.",
            json!({ "layer": { "type": "integer" } }),
            &["layer"],
            false,
        ),
        tool(
            "layer_move",
            "Move a layer up (positive steps) or down (negative) within its group.",
            json!({ "layer": layer_param(), "steps": { "type": "integer" } }),
            &["steps"],
            false,
        ),
        tool(
            "layer_group",
            "Wrap a layer in a new group. Returns the group id.",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "layer_ungroup",
            "Dissolve a group into its parent.",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "set_blend_space",
            "Colour space for blend modes: \"srgb\" (Photoshop-like, default) or \"linear\".",
            json!({ "space": { "type": "string", "enum": ["srgb", "linear"] } }),
            &["space"],
            false,
        ),
        tool(
            "paint_stroke",
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
                            "description": "Smudge instead of paint, with this strength (0.5 typical, 1 = drag colour all the way); colour not needed" },
                "clone_offset": { "type": "array", "items": { "type": "integer" },
                                  "description": "Clone stamp: paint the pixels found at point + [dx, dy] (exact copy); colour not needed" },
                "clone_all_layers": { "type": "boolean", "description": "Clone from the flattened image instead of the layer (stamp onto an empty layer)" },
                "heal": { "type": "boolean", "description": "With clone_offset: healing brush – texture from the source, colour and brightness matched to the surroundings when the stroke ends" },
                "spot_heal": { "type": "boolean", "description": "Spot healing brush: paint over a blemish, a matching source is found automatically (no clone_offset, no colour)" },
                "tone": { "type": "string", "enum": ["dodge", "burn", "sponge_saturate", "sponge_desaturate", "blur", "sharpen"],
                          "description": "Retouching brush: changes the pixels under it (no colour); one stroke applies it once, strokes add up" },
                "range": { "type": "string", "enum": ["midtones", "shadows", "highlights"], "description": "dodge/burn: tonal range (default midtones)" },
                "amount": { "type": "number", "minimum": 0, "maximum": 1, "description": "retouching: exposure/strength (default 0.5)" }
            }),
            &["points", "size"],
            false,
        ),
        tool(
            "fill",
            "Fill the selection (or the whole layer if nothing is selected) with a colour; respects soft selection edges.",
            json!({ "layer": layer_param(), "color": color_param("Fill colour"), "opacity": { "type": "number", "minimum": 0, "maximum": 1 }, "target": target_param() }),
            &["color"],
            false,
        ),
        tool(
            "clear",
            "Make the selected pixels (or the whole layer if nothing is selected) transparent.",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "gradient",
            "Draw a linear or radial gradient from (x0, y0) to (x1, y1) into the selection (or whole layer). color1 defaults to transparent.",
            json!({
                "layer": layer_param(),
                "kind": { "type": "string", "enum": ["linear", "radial"] },
                "x0": { "type": "number" }, "y0": { "type": "number" }, "x1": { "type": "number" }, "y1": { "type": "number" },
                "color0": color_param("Start colour"), "color1": color_param("End colour (default transparent)"),
                "opacity": { "type": "number", "minimum": 0, "maximum": 1 },
                "target": target_param()
            }),
            &["x0", "y0", "x1", "y1", "color0"],
            false,
        ),
        tool(
            "select_rect",
            "Select a rectangle (anti-aliased). Painting, fill, clear and gradient only affect the selection.",
            sel_rect("rect"),
            &["x", "y", "width", "height"],
            false,
        ),
        tool(
            "select_ellipse",
            "Select the ellipse inscribed in the given rectangle.",
            sel_rect("ellipse"),
            &["x", "y", "width", "height"],
            false,
        ),
        tool(
            "select_polygon",
            "Select a closed polygon (lasso), even-odd fill.",
            json!({ "points": { "type": "array", "minItems": 3, "items": { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 } }, "mode": mode_param() }),
            &["points"],
            false,
        ),
        tool(
            "select_all",
            "Select the whole document.",
            json!({}),
            &[],
            false,
        ),
        tool(
            "select_none",
            "Remove the selection (everything becomes editable).",
            json!({}),
            &[],
            false,
        ),
        tool(
            "select_invert",
            "Invert the selection.",
            json!({}),
            &[],
            false,
        ),
        tool(
            "select_feather",
            "Soften the selection edge by a radius in px.",
            json!({ "radius": { "type": "number", "minimum": 0, "maximum": 1000 } }),
            &["radius"],
            false,
        ),
        tool(
            "select_modify",
            "Select › Modify: expand or contract the selection by amount px (round corners), border (a soft ring of that \
             width around the edge), smooth (round off corners, drop specks smaller than amount) or feather.",
            json!({ "op": { "type": "string", "enum": ["expand", "contract", "border", "smooth", "feather"] },
                    "amount": { "type": "number", "minimum": 0, "maximum": 500 } }),
            &["op", "amount"],
            false,
        ),
        tool(
            "select_save",
            "Save the selection as a channel in the document (Photoshop: alpha channel); the same name replaces it. \
             Channels are stored in .mosaix files.",
            json!({ "name": { "type": "string" } }),
            &["name"],
            false,
        ),
        tool(
            "select_load",
            "Load a saved selection (channel by index or name), combined with the current one by mode; invert loads its inverse.",
            json!({ "channel": { "type": "integer", "minimum": 0 }, "name": { "type": "string" }, "mode": mode_param(),
                    "invert": { "type": "boolean" } }),
            &[],
            false,
        ),
        tool(
            "guide_add",
            "Add a guide line (snapping aid, saved in .mosaix): vertical at x = position or horizontal at y = position (document px).",
            json!({ "orientation": { "type": "string", "enum": ["vertical", "horizontal"] }, "position": { "type": "number" } }),
            &["orientation", "position"],
            false,
        ),
        tool(
            "layer_place",
            "Place an image file (png, jpg, webp, tif, avif, exr) as a smart object above the active layer: centred, scaled to fit \
             the canvas if larger. Smart objects are drawn from the embedded original, so layer_transform never loses detail. \
             layer_rasterize turns one into pixels.",
            json!({ "path": { "type": "string" } }),
            &["path"],
            false,
        ),
        tool(
            "layer_convert_smart",
            "Turn a pixel layer into a smart object (its pixels embedded), so later transforms are lossless.",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "layer_merge",
            "Merge layers into one pixel layer, as they look (blend modes, masks, styles, text baked in; pixels outside the canvas kept). mode: down = the active layer onto the one below (default), visible = all visible top-level layers (hidden ones stay), flatten = everything into one layer (hidden layers are dropped).",
            json!({ "mode": { "type": "string", "enum": ["down", "visible", "flatten"] } }),
            &[],
            false,
        ),
        tool(
            "smart_replace",
            "Replace the image inside a smart object (its placement stays).",
            json!({ "layer": layer_param(), "path": { "type": "string" } }),
            &["path"],
            false,
        ),
        tool(
            "generative_fill",
            "Paint the selection from a text prompt with the provider set up under Edit › Generative Fill (OpenAI image \
             edits – only after the user agreed there – or the user's own program). Applied to the layer as one undo step.",
            json!({ "layer": layer_param(), "prompt": { "type": "string" } }),
            &["prompt"],
            false,
        ),
        tool("guide_list", "The guides: [{vertical, pos}] by index.", json!({}), &[], false),
        tool(
            "guide_delete",
            "Delete a guide by index.",
            json!({ "index": { "type": "integer", "minimum": 0 } }),
            &["index"],
            false,
        ),
        tool("guide_clear", "Delete all guides.", json!({}), &[], false),
        tool("channel_list", "Names of the saved selections (channels), by index.", json!({}), &[], false),
        tool(
            "channel_delete",
            "Delete a saved selection (channel) by index.",
            json!({ "channel": { "type": "integer", "minimum": 0 } }),
            &["channel"],
            false,
        ),
        tool(
            "channel_rename",
            "Rename a saved selection (channel).",
            json!({ "channel": { "type": "integer", "minimum": 0 }, "name": { "type": "string" } }),
            &["channel", "name"],
            false,
        ),
        tool(
            "select_magic_wand",
            "Magic wand: select pixels similar in colour to the one at (x, y). tolerance 0–255 (default 32) is the largest \
             allowed channel difference in 8-bit sRGB; contiguous (default true) keeps only the connected region; \
             sample \"layer\" (default, active layer) or \"image\" (all layers).",
            json!({ "x": { "type": "integer" }, "y": { "type": "integer" },
                    "tolerance": { "type": "number", "minimum": 0, "maximum": 255 },
                    "contiguous": { "type": "boolean" }, "antialias": { "type": "boolean" },
                    "sample": { "type": "string", "enum": ["layer", "image"] }, "mode": mode_param() }),
            &["x", "y"],
            false,
        ),
        tool(
            "select_color_range",
            "Select all pixels of a colour anywhere in the image (soft edges), given as color or picked at (x, y). \
             fuzziness 0–200 (default 40) widens the range.",
            json!({ "color": color_param("colour to select (alternative to x/y)"), "x": { "type": "integer" }, "y": { "type": "integer" },
                    "fuzziness": { "type": "number", "minimum": 0, "maximum": 200 },
                    "sample": { "type": "string", "enum": ["layer", "image"] }, "mode": mode_param() }),
            &[],
            false,
        ),
        tool(
            "layer_mask_add",
            "Add a mask to a layer or group: from the selection (default when one exists) or revealing everything.",
            json!({ "layer": layer_param(), "from_selection": { "type": "boolean" } }),
            &[],
            false,
        ),
        tool(
            "layer_mask_delete",
            "Remove a layer's mask.",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "layer_mask_set",
            "Enable or disable a layer's mask.",
            json!({ "layer": layer_param(), "enabled": { "type": "boolean" } }),
            &["enabled"],
            false,
        ),
        tool(
            "layer_mask_apply",
            "Bake a layer's mask into its pixels and remove the mask.",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "layer_translate",
            "Move a whole layer (pixels and mask) by whole pixels, ignoring the selection. Content outside the canvas is kept.",
            json!({ "layer": layer_param(), "dx": { "type": "integer" }, "dy": { "type": "integer" } }),
            &["dx", "dy"],
            false,
        ),
        tool(
            "layer_transform",
            "Scale, skew, rotate and/or move a layer (with its mask) around a pivot (default: centre of its content). With a selection, \
             only the selected pixels move and the selection follows (like Photoshop's free transform); set use_selection=false \
             to transform the whole layer anyway. Order: scale, skew, rotate, then move by dx/dy. Distort / perspective: give \
             corners instead, the new positions of the content bounds' corners [[x,y]×4] clockwise from the top left (convex; \
             pixel layers only). Returns the new content bounds.",
            json!({
                "layer": layer_param(),
                "scale": { "type": "number", "description": "uniform factor, e.g. 0.5" },
                "scale_x": { "type": "number" }, "scale_y": { "type": "number", "description": "negative values mirror" },
                "rotate": { "type": "number", "description": "degrees, clockwise" },
                "skew_x": { "type": "number", "description": "degrees: slants vertical lines (−89…89)" },
                "skew_y": { "type": "number", "description": "degrees: slants horizontal lines (−89…89)" },
                "dx": { "type": "number" }, "dy": { "type": "number" },
                "pivot_x": { "type": "number" }, "pivot_y": { "type": "number" },
                "corners": { "type": "array", "items": { "type": "array", "items": { "type": "number" } }, "description": "distort/perspective: 4 [x, y] corners, clockwise from the top left" },
                "interp": { "type": "string", "enum": ["bicubic", "bilinear", "nearest"], "description": "default bicubic; nearest for pixel art" },
                "use_selection": { "type": "boolean" }
            }),
            &[],
            false,
        ),
        tool(
            "adjustment_add",
            "Add a non-destructive adjustment layer above the active layer; it changes everything below it (within its group). \
             With a selection, the selection becomes its mask. \
             and can be edited later with adjustment_set, masked (layer_mask_add, paint_stroke target mask) and faded with layer_set opacity. \
             adjustment is an object with \"kind\" and parameters, e.g. {\"kind\": \"curves\", \"master\": [[0,0],[0.25,0.18],[0.75,0.85],[1,1]]}. \
             Kinds: brightness_contrast {brightness −150…150, contrast −50…100}; levels {master/red/green/blue: {in_black, in_white, gamma, out_black, out_white} 0…1}; \
             curves {master/red/green/blue: [[x,y],…] 0…1}; exposure {exposure stops, offset, gamma}; vibrance {vibrance, saturation −100…100}; \
             hue_saturation {hue −180…180, saturation, lightness −100…100, colorize}; black_white {weights: 6 percentages for red, yellow, green, cyan, blue, magenta}; \
             invert; posterize {levels}; threshold {level 1…255}; color_balance {shadows/midtones/highlights: [cyan–red, magenta–green, \
             yellow–blue] −100…100, preserve_luminosity}; channel_mixer {red/green/blue/gray: [r %, g %, b %, constant %], monochrome}; \
             gradient_map {gradient: {colors: [{at 0…1, color [r,g,b] linear}]}, reverse}. Omitted parameters are neutral.",
            json!({ "adjustment": { "type": "object" }, "name": { "type": "string" } }),
            &["adjustment"],
            false,
        ),
        tool(
            "filter_apply",
            "Apply a pixel filter to a layer (within the selection, if any). filter is an object with \"kind\": \
             gaussian_blur {radius px}; unsharp_mask {amount % 1…500, radius px, threshold 0…255}; \
             high_pass {radius px}; shadows_highlights {shadows %, highlights %, radius px}; motion_blur {angle °, distance px}; median {radius 1…10}; pixelate {size px}; add_noise {amount % 0…100, monochrome, gaussian, seed}; \
             plugin {name, values: {param: number | bool | \"#rrggbb\"}} for a filter plugin (see filter_plugin_list). \
             With smart=true it is added as a smart filter instead: non-destructive, on the whole layer, \
             listed under smart_filters in document_info and editable with smart_filter_set.",
            json!({ "layer": layer_param(), "filter": { "type": "object" }, "smart": { "type": "boolean" } }),
            &["filter"],
            false,
        ),
        tool(
            "text_add",
            "Add a text layer (stays editable; system fonts). (x, y) is the top-left of the text box; '\\n' starts a new line; \
             width wraps lines (paragraph text). Returns the layer id and its bounds.",
            json!({ "text": { "type": "string" }, "x": { "type": "number" }, "y": { "type": "number" },
                    "size": { "type": "number", "description": "px" }, "font": { "type": "string", "description": "family, or sans-serif / serif / monospace" },
                    "color": color_param("Text colour (default black)"), "weight": { "type": "number", "description": "100–900, 700 = bold" },
                    "italic": { "type": "boolean" }, "align": { "type": "string", "enum": ["left", "center", "right", "justify"] },
                    "line_height": { "type": "number", "description": "multiple of the size (1.2)" }, "letter_spacing": { "type": "number" },
                    "width": { "type": "number" }, "name": { "type": "string" } }),
            &["text", "x", "y", "size"],
            false,
        ),
        tool(
            "shape_add",
            "Add a shape layer (stays sharp and editable): rect / ellipse (x, y, width, height; rect radius), polygon (points), \
             path (SVG path data d). Fill defaults to black; no_fill for outlines; stroke_color / stroke_width for a contour; \
             gradient (colours, evenly spaced; gradient_style, gradient_angle) or pattern (a name from pattern_list; pattern_scale) \
             fill it like Photoshop's gradient / pattern fill layers.",
            json!({ "shape": { "type": "string", "enum": ["rect", "ellipse", "polygon", "path"] },
                    "x": { "type": "number" }, "y": { "type": "number" }, "width": { "type": "number" }, "height": { "type": "number" },
                    "radius": { "type": "number" }, "points": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } },
                    "d": { "type": "string" }, "fill": color_param("Fill colour"), "no_fill": { "type": "boolean" },
                    "stroke_color": color_param("Contour colour"), "stroke_width": { "type": "number" },
                    "stroke_align": { "type": "string", "enum": ["inside", "center", "outside"], "description": "Where the contour lies on closed shapes (default center)" },
                    "gradient": { "type": "array", "items": color_param("Gradient colour"), "description": "Gradient fill through these colours" },
                    "gradient_style": { "type": "string", "enum": ["linear", "radial", "angle", "reflected", "diamond"] },
                    "gradient_angle": { "type": "number", "description": "Degrees, 90 (default) = from bottom to top" },
                    "pattern": { "type": "string", "description": "Pattern fill: a name from pattern_list" },
                    "pattern_scale": { "type": "number" },
                    "name": { "type": "string" } }),
            &["shape"],
            false,
        ),
        tool(
            "path_smooth",
            "Make every node of a pen path or polygon shape a smooth curve point (a polygon becomes a pen path) – handles along the neighbours' direction, the curve runs through the nodes – or, with smooth: false, a corner (straight segments).",
            json!({ "layer": layer_param(), "smooth": { "type": "boolean" } }),
            &[],
            false,
        ),
        tool(
            "vector_set",
            "Change a text or shape layer: `changes` is merged into its description as shown by document_info \
             (e.g. {\"text\": \"New\", \"size\": 64, \"color\": \"#ff0000\"} or {\"fill\": \"#00ff00\", \"radius\": 12}; \
             \"transform\": [a, b, c, d, e, f] places it). Text on a path: {\"path\": {\"nodes\": [{\"p\": [x, y], \"h_in\": [x, y], \"h_out\": [x, y]}, …], \
             \"closed\": false, \"offset\": 0.0–1.0, \"flip\": false}} (null removes it). OpenType: {\"features\": {\"smcp\": 1, \"liga\": 0}}. \
             Shape paint: {\"paint\": {\"type\": \"gradient\", \"gradient\": {\"colors\": [{\"at\": 0, \"color\": [r, g, b]}, …], \"alphas\": [{\"at\": 0, \"opacity\": 1}, …], \
             \"smoothness\": 0–1}, \"style\": \"linear|radial|angle|reflected|diamond\", \"angle\": 90, \"scale\": 1, \"reverse\": false}} or \
             {\"paint\": {\"type\": \"pattern\", \"pattern\": \"Dots\", \"scale\": 1, \"offset\": [x, y]}}; null: the plain fill colour.",
            json!({ "layer": layer_param(), "changes": { "type": "object" } }),
            &["changes"],
            false,
        ),
        tool(
            "layer_effects_set",
            "Set a layer's styles (replaces the list; [] removes all). Each effect: {\"effect\": kind, \"enabled\": true, …}. \
             Kinds and fields (colours linear [r, g, b] 0…1, opacity 0…1, angle = light direction in degrees, 90 = from the top): \
             stroke {size, position: outside|inside|center, color, opacity}; drop_shadow {color, opacity, angle, distance, size, spread}; \
             inner_shadow {color, opacity, angle, distance, size}; outer_glow {color, opacity, size, spread, range}; inner_glow {color, opacity, size, range} (range 0…1, default 0.5); \
             satin {color, opacity, angle, distance, size, invert (default true)} (default blend multiply); \
             color_overlay {color, opacity}; gradient_overlay {start, end, opacity, angle (90 = bottom to top), style: linear|radial|angle|reflected|diamond, \
             scale, reverse, gradient (optional, more stops: {colors: [{at, color}], alphas: [{at, opacity}], smoothness})}; pattern_overlay {pattern: a name from pattern_list, opacity, scale (1 = pattern pixels), offset: [x, y]}; bevel {style: inner|emboss|pillow, size, depth, soften, angle, altitude, highlight, \
             highlight_opacity, shadow, shadow_opacity}. Optional on every effect: \"blend\" (a blend mode key; default: \
             multiply for shadows and satin, screen for glows, normal otherwise) and \"global_light\": true (shadows/bevel share one light angle).",
            json!({ "layer": layer_param(), "effects": { "type": "array", "items": { "type": "object" } } }),
            &["effects"],
            false,
        ),
        tool(
            "pattern_list",
            "The pattern library for pattern overlays: built-in patterns and the user's (folder included).",
            json!({}),
            &[],
            true,
        ),
        tool(
            "pattern_import",
            "Add an image or a GIMP pattern (.pat) to the pattern library.",
            json!({ "path": { "type": "string" } }),
            &["path"],
            false,
        ),
        tool(
            "pattern_define",
            "Make a library pattern from the visible pixels in the selection (or the whole image), like Photoshop's Edit › Define Pattern.",
            json!({ "name": { "type": "string" } }),
            &[],
            false,
        ),
        tool(
            "layer_rasterize",
            "Turn a text or shape layer into ordinary pixels (needed before painting or pixel filters).",
            json!({ "layer": layer_param() }),
            &[],
            false,
        ),
        tool(
            "content_aware_fill",
            "Fill the selection on a pixel layer with texture from around it so the hole continues the surroundings \
             (remove objects, extend backgrounds). Select a bit more than the object. all_layers: learn from the \
             flattened image (fill on an empty layer). One undo step.",
            json!({ "layer": layer_param(), "all_layers": { "type": "boolean" } }),
            &[],
            false,
        ),
        tool(
            "liquify",
            "Liquify a pixel layer (one undo step). Each stroke: tool push (moves pixels along the path of `points`), \
             reconstruct, twirl_cw, twirl_ccw, pucker (shrink towards the point), bloat (grow); radius px; strength 0…1; \
             for the non-push tools `steps` dabs per point (default 10, ≈ how long the brush is held).",
            json!({ "layer": layer_param(), "strokes": { "type": "array", "items": { "type": "object", "properties": {
                "tool": { "type": "string", "enum": ["push", "reconstruct", "twirl_cw", "twirl_ccw", "pucker", "bloat"] },
                "points": { "type": "array", "items": { "type": "array", "items": { "type": "number" } } },
                "radius": { "type": "number" }, "strength": { "type": "number" }, "steps": { "type": "integer" } },
                "required": ["points", "radius"] } } }),
            &["strokes"],
            false,
        ),
        tool(
            "smart_filter_set",
            "Change a layer's smart filter number `index` (0 = applied first): new parameters (`filter`, same kind), \
             switch it on/off (`enabled`) or `remove` it. `apply`=true instead writes all smart filters into the pixels and removes them.",
            json!({ "layer": layer_param(), "index": { "type": "integer" }, "filter": { "type": "object" },
                    "enabled": { "type": "boolean" }, "remove": { "type": "boolean" }, "apply": { "type": "boolean" } }),
            &[],
            false,
        ),
        tool(
            "adjustment_set",
            "Change the parameters of an adjustment layer (same kind; see adjustment_add).",
            json!({ "layer": layer_param(), "adjustment": { "type": "object" } }),
            &["adjustment"],
            false,
        ),
        tool(
            "layer_bounds",
            "Pixel-exact bounding box of a layer's visible content (within the selection, if any); null if empty.",
            json!({ "layer": layer_param() }),
            &[],
            true,
        ),
        tool(
            "histogram",
            "Tonal histogram per channel (rgb, red, green, blue) with mean, median, 1st/99th percentile and clipping. With `below`, of the image underneath that layer (the input of an adjustment layer); otherwise of the whole image.",
            json!({ "below": layer_param(), "bins": { "type": "integer", "description": "number of bins, a divisor of 256 (default 32)" } }),
            &[],
            true,
        ),
        tool(
            "crop",
            "Crop the canvas to a rectangle (may extend beyond the canvas to enlarge it). Pixels outside are kept, not deleted. Drops the selection.",
            json!({ "x": { "type": "integer" }, "y": { "type": "integer" }, "width": { "type": "integer" }, "height": { "type": "integer" } }),
            &["x", "y", "width", "height"],
            false,
        ),
        tool(
            "canvas_size",
            "Change the canvas size without scaling; anchor decides where the existing image stays (default center).",
            json!({ "width": { "type": "integer" }, "height": { "type": "integer" },
                    "anchor": { "type": "string", "enum": ["center", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"] } }),
            &["width", "height"],
            false,
        ),
        tool(
            "image_resize",
            "Scale the whole image (all layers and masks). Give width and/or height; a missing one keeps the aspect ratio.",
            json!({ "width": { "type": "integer" }, "height": { "type": "integer" },
                    "interp": { "type": "string", "enum": ["bicubic", "bilinear", "nearest"] } }),
            &[],
            false,
        ),
        tool(
            "fill_rect",
            "Fill a rectangle on a pixel layer with a colour (normal blending onto existing pixels).",
            Value::Object(fill),
            &["x", "y", "width", "height", "color"],
            false,
        ),
        tool(
            "clear_rect",
            "Make a rectangle on a pixel layer fully transparent.",
            Value::Object(clear),
            &["x", "y", "width", "height"],
            false,
        ),
        tool(
            "place_image",
            "Load an image file into a new layer, top-left corner at (x, y). Returns the layer id.",
            json!({ "path": { "type": "string" }, "x": { "type": "integer" }, "y": { "type": "integer" }, "name": { "type": "string" } }),
            &["path"],
            false,
        ),
        tool(
            "sample_color",
            "Colour of the flattened image at a document pixel (sRGB hex + alpha).",
            json!({ "x": { "type": "number" }, "y": { "type": "number" } }),
            &["x", "y"],
            true,
        ),
        tool(
            "render_preview",
            "Render the flattened image (or the region x/y/width/height) as a PNG you can look at. Use it to check your work.",
            Value::Object(prev),
            &[],
            true,
        ),
        tool(
            "undo",
            "Undo the last edit (yours or the user's). Returns its label.",
            json!({}),
            &[],
            false,
        ),
        tool("redo", "Redo the last undone edit.", json!({}), &[], false),
        tool(
            "save",
            "Save as .mosaix (keeps layers; incremental when saving to the same file again).",
            json!({ "path": { "type": "string", "description": "Default: next to the source file" } }),
            &[],
            false,
        ),
        tool(
            "export",
            "Export the image; format by extension: png, jpg, webp, avif, jxl, tif (flattened) or psd/psb (layers, groups, masks, adjustment layers, layer styles, fill opacity; text and shape layers stay editable; smart filters and text on a path rasterised). Wide-gamut colour spaces embed their ICC profile; 16 bit works for png, tif, psd and jxl; jxl takes every colour space, avif srgb, display_p3 and rec2020 (signalled by CICP code); quality 100 makes jxl lossless.",
            json!({ "path": { "type": "string" },
                    "color_space": { "type": "string", "enum": ["srgb", "display_p3", "adobe_rgb", "rec2020"], "description": "default srgb" },
                    "bit_depth": { "type": "integer", "enum": [8, 16] },
                    "quality": { "type": "integer", "description": "JPEG quality 1–100 (default 92)" } }),
            &["path"],
            false,
        ),
    ]
}

/// Handles one JSON-RPC message. Returns the response, or `None` for notifications.
pub fn handle(backend: &mut dyn Backend, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned().filter(|v| !v.is_null());
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result: Result<Value, (i64, String)> = match method {
        "initialize" => {
            let requested = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or("");
            let version = PROTOCOL_VERSIONS
                .iter()
                .find(|v| **v == requested)
                .unwrap_or(&PROTOCOL_VERSIONS[0]);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "mosaix", "version": env!("CARGO_PKG_VERSION") },
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
                        let mut content = vec![
                            json!({ "type": "text", "text": serde_json::to_string_pretty(&out.result).unwrap() }),
                        ];
                        if let Some(png) = out.image_png {
                            content.push(json!({
                                "type": "image",
                                "mimeType": "image/png",
                                "data": base64::engine::general_purpose::STANDARD.encode(png),
                            }));
                        }
                        json!({ "content": content })
                    }
                    Err(e) => {
                        json!({ "content": [{ "type": "text", "text": format!("{e:#}") }], "isError": true })
                    }
                })
            }
        }
        _ if id.is_none() => return None, // notifications (e.g. notifications/initialized)
        other => Err((-32601, format!("method not found: {other}"))),
    };
    let id = id?;
    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        }
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
            Err(e) => Some(
                json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } }),
            ),
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
            Ok(CallOutput {
                result: json!({ "ok": true }),
                image_png: Some(vec![1, 2, 3]),
            })
        }
    }

    #[test]
    fn handshake_list_and_call() {
        let mut b = Echo(Vec::new());
        let init = handle(&mut b, &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}})).unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(
            handle(
                &mut b,
                &json!({"jsonrpc":"2.0","method":"notifications/initialized"})
            )
            .is_none()
        );

        let list = handle(
            &mut b,
            &json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        )
        .unwrap();
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

        let r = handle(
            &mut b,
            &json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"undo"}}),
        )
        .unwrap();
        assert_eq!(r["result"]["isError"], true);
        let r = handle(
            &mut b,
            &json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"rm_rf"}}),
        )
        .unwrap();
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
            "layer_transform": {"scale": 2}, "layer_bounds": {}, "histogram": {"bins": 16}, "crop": {"x": 0, "y": 0, "width": 1, "height": 1},
            "canvas_size": {"width": 1, "height": 1}, "image_resize": {"width": 1},
            "select_magic_wand": {"x": 0, "y": 0}, "select_color_range": {"color": "#ff0000"},
            "adjustment_add": {"adjustment": {"kind": "invert"}}, "filter_apply": {"filter": {"kind": "gaussian_blur", "radius": 2}}, "smart_filter_set": {"index": 0, "enabled": false}, "content_aware_fill": {}, "text_add": {"text": "Hi", "x": 1, "y": 2, "size": 20}, "shape_add": {"shape": "rect", "x": 0, "y": 0, "width": 5, "height": 5}, "vector_set": {"changes": {"text": "x"}}, "layer_rasterize": {}, "pattern_list": {}, "raw_info": {"path": "x.nef"}, "filter_plugin_list": {}, "action_list": {}, "select_subject": {}, "select_object": {"points": [[1, 2]]}, "remove_background": {}, "ai_model_download": {"model": "subject"}, "action_run": {"name": "x"}, "filter_plugin_install": {"source": "x", "check_only": true}, "raw_set": {"exposure": 1}, "pattern_import": {"path": "p.png"}, "pattern_define": {}, "layer_effects_set": {"effects": [{"effect": "color_overlay", "color": [1, 0, 0], "opacity": 1}]}, "liquify": {"strokes": [{"points": [[0, 0], [5, 0]], "radius": 20}]}, "adjustment_set": {"adjustment": {"kind": "invert"}}
        });
        samples
            .as_object_mut()
            .unwrap()
            .extend(more.as_object().unwrap().clone());
        let selection = json!({
            "select_modify": {"op": "expand", "amount": 3}, "select_save": {"name": "a"},
            "channel_delete": {"channel": 0}, "channel_rename": {"channel": 0, "name": "b"},
            "guide_add": {"orientation": "vertical", "position": 10}, "guide_delete": {"index": 0},
            "layer_place": {"path": "a.png"}, "smart_replace": {"path": "a.png"},
            "generative_fill": {"prompt": "a bench"}
        });
        samples.as_object_mut().unwrap().extend(selection.as_object().unwrap().clone());
        for t in tools() {
            let name = t["name"].as_str().unwrap();
            // Tools without required parameters need no sample.
            let sample = samples[name].as_object().cloned().unwrap_or_default();
            let mut args = sample.clone();
            args.insert("cmd".into(), name.into());
            let cmd = Command::parse(Value::Object(args)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(cmd.name(), name);
            // Every schema property must be accepted by the command (deny_unknown_fields
            // would reject it otherwise): try each one with a plausible value.
            for (prop, schema) in t["inputSchema"]["properties"].as_object().unwrap() {
                let mut a = sample.clone();
                a.insert("cmd".into(), name.into());
                if !a.contains_key(prop) {
                    let v = match (prop.as_str(), schema["type"].as_str()) {
                        // Enumerations: their first value.
                        _ if schema["enum"].is_array() => schema["enum"][0].clone(),
                        ("color_space", _) => json!("display_p3"),
                        ("clone_offset", _) => json!([10, -5]),
                        ("align", _) => json!("center"),
                        ("fill" | "stroke_color", _) => json!("#123456"),
                        (p, _) if p.starts_with("color") => json!("#123456"),
                        ("target", _) => json!("pixels"),
                        ("mode", _) => json!("add"),
                        ("kind", _) => json!("radial"),
                        ("blend", _) => json!("screen"),
                        ("space", _) => json!("linear"),
                        ("points", _) => json!([[0, 0], [1, 1], [2, 0]]),
                        ("gradient", _) => json!(["#000000", "#ffffff"]),
                        ("interp", _) => json!("nearest"),
                        ("anchor", _) => json!("top-left"),
                        ("sample", _) => json!("image"),
                        ("tip", _) => json!("chalk"),
                        ("adjustment", _) => json!({"kind": "invert"}),
                        ("white_balance", _) => json!({"mode": "auto"}),
                        ("highlights", _) => json!("clip"),
                        ("tone", _) => json!({"highlights": -50}),
                        ("negative", _) => json!([[1, 2]]),
                        ("corners", _) => json!([[0, 0], [10, 0], [10, 10], [0, 10]]),
                        ("noise", _) => json!({"luminance": 10, "color": 30}),
                        ("raw", _) => json!({"white_balance": {"mode": "custom", "temperature": 5000, "tint": 0}, "exposure": 1}),
                        ("filter", _) => json!({"kind": "high_pass", "radius": 3}),
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
