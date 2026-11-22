# Mosaix for agents

AI agents can work with Mosaix just like people do. All ways in share the same command layer (`crates/mosaix-app/src/agent/command.rs`). Every change is **one undo step**, whether it comes from an agent or from you.

| Way in | What for | Command |
|---|---|---|
| **MCP server, own engine** | The agent works alone, without a window | `mosaix mcp [FILE]` |
| **MCP server, live** | The agent drives your open window; you watch and can step in | `mosaix mcp --attach` |
| **JSON Lines script** | Agents with shell access, batch jobs, tests | `mosaix run SCRIPT.jsonl` / `… --attach -` |

## Setup (Claude Code)

```sh
# live: the agent works in your running Mosaix window
claude mcp add mosaix -- /path/to/mosaix mcp --attach

# or headless with its own engine
claude mcp add mosaix-headless -- /path/to/mosaix mcp
```

Or as `.mcp.json` in the project:

```json
{ "mcpServers": { "mosaix": { "command": "/path/to/mosaix", "args": ["mcp", "--attach"] } } }
```

Every Mosaix window listens on `$XDG_RUNTIME_DIR/mosaix/<pid>.sock` (accessible only to your user). `--attach` connects to the most recently started window; `--socket PATH` or `MOSAIX_SOCKET` picks a specific one.

## Conventions

- Coordinates in **document pixels**, origin top left
- Colors in **sRGB**: `"#rrggbb"`, `"#rrggbbaa"` or `[r, g, b(, a)]` with 0–255
- `layer` is optional and otherwise means the active layer. Layer IDs stay stable, also across saving and loading.
- Errors come back as a readable message (MCP: `isError`, script: `"ok": false`). Typos in parameters are rejected, not ignored.
- While you are painting yourself, Mosaix briefly refuses changing agent commands ("the user is painting right now").

## Tools

| Tool | Purpose |
|---|---|
| `document_new`, `document_open`, `document_info` | Create/open a document; query structure, layer tree, history. Camera RAW (CR2/CR3, NEF, ARW, RAF, ORF, RW2, DNG …) is developed scene-linear; `raw: {"white_balance": {"mode": "as_shot"\|"auto"\|"daylight"} or {"mode": "custom", "temperature": 5200, "tint": 10}, "exposure": 0.5}` |
| `filter_plugin_list`, `filter_plugin_install` | List GPU filter plugins (WGSL), or check and install your own filter, then `filter_apply {"filter": {"kind": "plugin", "name": …, "values": {…}}}` – format in `docs/filter-plugins.md` |
| `select_subject`, `select_object`, `remove_background` | Local AI (runs on the computer, CPU): select the main subject (BiRefNet), select an object by points (SAM 2.1; `points`, optional `negative`, `mode`), hide the background with a layer mask. Download the models once with `ai_model_download {model: subject|object|remove|upscale}` – ask the user first |
| `remove_area` | Local AI (LaMa): remove the selection – it is replaced by generated surroundings (~1–3 s). The selection is first grown by `expand` px (default 4); `all_layers`: learn from the whole visible image. Typical: `select_object` → `remove_area` |
| `image_upscale` | Local AI (Real-ESRGAN): enlarge the image 2× or 4× (`factor`), with invented detail instead of blur; ~10 s per megapixel of the source image. Text, shapes and masks scale as with `image_resize` |
| `raw_info` | Camera, lens, exposure data and white balance (temperature/tint) of a RAW file, without opening it |
| `raw_set` | Re-develop a RAW layer (`white_balance`, `exposure`, `highlights`: `reconstruct`/`clip`, `noise`: `{luminance, color}` 0–100, `lens`: lens correction on/off, `tone`: `{contrast, highlights, shadows, vibrance, saturation}` −100…100). Opened RAW files and those placed with `place_image` stay RAW layers (the file is inside the document, transforms are lossless); `layer_rasterize` turns them into pixels |
| `layer_add`, `layer_duplicate`, `layer_delete`, `layer_select`, `layer_move`, `layer_group`, `layer_ungroup` | Layer management |
| `layer_select` with `also` | Select several layers; `layer_delete`, `layer_group`, `layer_duplicate`, `layer_merge` (mode `selected`, the default then) and `layer_translate`/`layer_transform` act on all of them |
| `layer_set` | Name, opacity, blend mode (`multiply`, `screen`, `overlay`, … 19 in all), visibility |
| `set_blend_space` | `srgb` (Photoshop-like) or `linear` |
| `paint_stroke` | Brush stroke along a list of points, optionally with pressure per point; size, hardness, opacity, flow, eraser; tips (`tip`: round/chalk/bristles/spray/rough/charcoal), angle, roundness, jitter (`size_jitter`, `angle_jitter`, `scatter`, `flow_jitter`), `follow_direction`; `smudge: 0…1` smudges instead of painting; `tone` (dodge/burn/sponge_saturate/sponge_desaturate/blur/sharpen) with `range` and `amount` retouches instead of painting; `target: "mask"` paints into the layer mask |
| `select_rect`, `select_ellipse`, `select_polygon` | Anti-aliased selection; `mode`: `replace`, `add`, `subtract`, `intersect` |
| `select_all`, `select_none`, `select_invert`, `select_feather` | Manage the selection, feather |
| `generative_fill` | Paint the selection from a prompt, with the provider set up in the dialog (OpenAI only after the user agreed there). Sees all visible layers; the result goes onto a new layer above the active one unless `new_layer: false` or a `layer` is given |
| `layer_via` | Layer via Copy (`cut: false`) or Cut (`cut: true`): the selected pixels of the active layer onto a new layer above, in place |
| `select_layer_pixels` | Select a layer's pixels by their opacity (`mode` replace/add/subtract/intersect) |
| `bucket_fill` | Paint Bucket at (x, y): `color`, `opacity`, `tolerance` 0–255, `contiguous`, `sample` layer/image; within the selection |
| `adjustment_apply` | Apply an adjustment (same objects as `adjustment_add`) to the pixel layer's own pixels, inside the selection |
| `layer_align`, `layer_distribute` | Align the selected layers (`edge` left/center_x/right/top/center_y/bottom; one layer: to the canvas) or space three or more evenly (`direction`) |
| `stroke_selection` | Edit › Stroke: `width`, `position` inside/center/outside, `color`, `opacity` |
| `layer_set` with `lock` | `{transparency, pixels, position}` |
| `select_refine` | Select and Mask: `radius` (edge follows the image: hair, fur), `contrast`, `shift`, `feather`, `output` selection/mask |
| `select_transform` | Transform the selection outline only: `scale`, `scale_x/y`, `rotate`, `dx`, `dy` |
| `paint_stroke` with `pencil` / `history` | Pencil (no anti-aliasing); History Brush, painting back the state set with `history_source {position}` |
| `layer_merge` | Merge layers as they look (blend modes, masks, styles, text baked in; pixels outside the canvas are kept): `mode` `down` (the active layer onto the one below, default), `visible` (all visible top-level layers; hidden ones stay), `flatten` (everything into one layer; hidden layers are dropped) |
| `layer_place`, `layer_convert_smart`, `smart_replace` | Smart objects: embed an image, convert a pixel layer, replace the contents; transforms stay lossless |
| `guide_add`, `guide_list`, `guide_delete`, `guide_clear` | Guides (`orientation` vertical/horizontal, `position` in px) |
| `select_modify` | `op` expand/contract/border/smooth/feather by `amount` px |
| `select_save`, `select_load`, `channel_list`, `channel_rename`, `channel_delete` | Saved selections (channels) in the document; load with `mode` and `invert` |
| `fill`, `clear`, `gradient` | Fill or clear the selection (or the whole layer), linear/radial gradient; layer or mask |
| `layer_mask_add`, `layer_mask_delete`, `layer_mask_set`, `layer_mask_apply` | Layer masks: create (from the selection), delete, on/off, apply to the pixels |
| `layer_translate` | Move a whole layer with its mask by whole pixels (ignores the selection) |
| `layer_transform` | Scale, skew (`skew_x`/`skew_y` in degrees), rotate (degrees, clockwise), move around a pivot (default: center of the content); distort/perspective with `corners` (new position of the four corners of the content, clockwise from top left); with a selection only the selected pixels; `interp` bicubic/bilinear/nearest |
| `layer_bounds` | Pixel-exact bounds of a layer's visible content |
| `filter_apply` with `smart: true`, `smart_filter_set` | Smart filters: non-destructive on the layer; change, on/off, remove, bake into the pixels (`apply`); `document_info` lists them under `smart_filters` |
| `liquify` | Liquify with strokes (push along a path, reconstruct, twirl_cw/ccw, pucker, bloat), one undo step |
| `paint_stroke` with `clone_offset` | Clone stamp: copies pixel-exact from point + offset; `clone_all_layers` uses the whole image as the source; with `heal: true` the Healing Brush (color/brightness matched to the surroundings); `spot_heal: true` Spot Healing without a source |
| `content_aware_fill` | Fills the selection content-aware (removing objects); `all_layers` learns from the whole image |
| `text_add`, `shape_add`, `vector_set`, `layer_rasterize` | Create text and shape layers, change them through their description (JSON merge, colors also as `#rrggbb`) and turn them into pixels |
| `path_smooth` | Smooth all nodes of a pen path or polygon (the curve runs through the nodes; a polygon becomes a path), or with `smooth: false` make them corners |
| `layer_effects_set` | Set layer styles: stroke, drop shadow, inner/outer shadow and glow, color overlay, bevel & emboss |
| `histogram` | Histogram per channel with mean, median, 1st/99th percentile and clipping, optionally of what lies below a layer |
| `crop`, `canvas_size`, `image_resize` | Crop (may go beyond the edge), canvas size with anchor, image size; pixels outside are kept |
| `select_magic_wand`, `select_color_range` | Magic wand (tolerance 0–255, contiguous or not, `sample` layer/image) and color range (color or point, fuzziness 0–200) |
| `fill_rect`, `clear_rect` | Fill a rectangle or make it transparent (regardless of the selection) |
| `place_image` | Image file as a new layer at position (x, y) |
| `sample_color` | Color of the whole image at a point |
| `render_preview` | **Look at the result:** the whole image or a region as PNG, optionally also as a file (`output`); `show_selection` tints what is not selected red |
| `adjustment_add`, `adjustment_set` | Add and change adjustment layers (curves, levels, hue/saturation, exposure, color balance, channel mixer, gradient map, …); with a selection it becomes the mask |
| `filter_apply` | Filters: Gaussian blur, unsharp mask, high pass, shadows/highlights, motion blur, median, mosaic, noise (through the selection) |
| `undo`, `redo` | History |
| `save`, `export` | `.mosaix` (with layers, incremental) or PNG/JPEG/WebP/TIFF; `export` optionally in Display P3, Adobe RGB or Rec. 2020 (profile embedded), with 16 bit (PNG/TIFF) and JPEG quality |

The server provides the full JSON schemas via `tools/list`.

## Example: script

```jsonl
# comments with # or //
{"cmd": "document_new", "width": 800, "height": 500}
{"cmd": "fill_rect", "x": 0, "y": 0, "width": 800, "height": 500, "color": "#1e2a44"}
{"cmd": "layer_add", "name": "Sun"}
{"cmd": "paint_stroke", "points": [[600, 120]], "size": 140, "hardness": 0.4, "color": "#ffcc33"}
{"cmd": "layer_set", "blend": "screen", "opacity": 0.8}
{"cmd": "render_preview", "max_size": 512, "output": "preview.png"}
{"cmd": "save", "path": "night.mosaix"}
```

```sh
mosaix run night.jsonl            # own engine
mosaix run --attach night.jsonl   # in the open window
```

Each line produces one JSON result line on stdout. At the first error the script stops with exit code 1. Logs go to stderr.

## Limits

- All selection, fill and paint commands respect the current selection, also a soft one.
- `render_preview` shows the image, not the window view (no zoom, no rotation).
- In live mode `document_open`/`document_new` replace the open document without asking. Unsaved changes are lost.
