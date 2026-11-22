# Usage

Detailed reference for the command line, shortcuts and tools. Short overview: [README](../README.md).

## Build & run

```sh
cargo run --release -- image.jpg             # open an image (PSD/PSB with layers, PNG, JPEG, WebP, TIFF, EXR, .mosaix)
cargo run --release -- --new 30000x30000     # empty document
cargo run --release -- --demo 10000x10000    # 100 MP test image
cargo run --release -- in.jpg --save out.mosaix --export out.png   # convert without a window
cargo run --release -- --demo 10000x10000 --bench-offscreen         # benchmark without a window
cargo run --release --features profile -- …  # with Tracy profiling
cargo test --workspace                       # includes headless GPU tests (Vulkan)
```

Requirements: a Wayland compositor and a Vulkan driver with descriptor indexing (RADV, ANV, NVIDIA).

## Language

The interface is English; other languages come from gettext catalogs (`crates/mosaix-core/i18n/<lang>.po`, currently German). The language follows `LANGUAGE`, `LC_ALL`, `LC_MESSAGES`, `LANG` (C/POSIX → English); `"language": "de"` in the settings file overrides it. `scripts/i18n-check.py check` lists untranslated or stale texts.

## For agents

Mosaix can be driven over **MCP** and JSON scripts, also live in the open window:

```sh
claude mcp add mosaix -- /path/to/mosaix mcp --attach
```

Details: [docs/agents.md](agents.md)

## Working with Mosaix

Without a file Mosaix opens a start screen: New Image, Open, quick-start sizes and the recently opened files with thumbnails (cached in `~/.cache/mosaix/thumbs`, written when saving, when opening another file and when quitting). Esc or "Empty Canvas" closes it.

Open, Save, Export and Import use the system file dialog (XDG desktop portal, e.g. GTK/GNOME); color space, 16 bit and JPEG quality are right in the dialog when exporting. Without a portal, or with `MOSAIX_BUILTIN_DIALOG=1`, the built-in dialog is used.

Related tools share one slot in the tool bar (small triangle in the corner): right-click or press and hold to see them all; the slot remembers the last one used. Menu entries that cannot run right now are greyed out; the tooltip says why.

The window has a menu bar, the tool bar on the left and panels on the right for color, tool settings, layers (drag & drop, double-click to rename, click the mask icon to paint the mask) and history (click to jump back). At the bottom is a status bar with messages; clicking it opens all messages. Ctrl+`K` or Ctrl+Shift+`P` opens the command palette, which finds every menu entry, including dialogs, filters and adjustment layers. `Tab` hides the interface. Ctrl+`N` new image, Ctrl+`O` open (with "Open Recent"), Ctrl+Shift+`S` save as. Alt+right-drag changes brush size (horizontally) and hardness (vertically). Mosaix remembers brush, color, tool and recent files in `~/.config/mosaix/settings.json`. Every three minutes, if there are unsaved changes, a recovery copy is saved to `~/.local/state/mosaix/autosave/`; your file is left untouched.

Shortcuts follow Photoshop.

| Action | Input |
|---|---|
| Tools | `B` Brush, `U` Smudge, `E` Eraser, `M` Rectangular/Elliptical Marquee, `L` Lasso, `W` Magic Wand, `G` Gradient linear/radial, `V` Move, `C` Crop |
| Painting | left mouse button / pen; eraser end of the pen; `[` `]` size, digits opacity (Smudge: strength), `H` hardness, Shift+`S` smoothing, Shift+`B` tip, Shift+`J` dynamics, Shift+`P` palette color, `X` swap foreground/background color, `D` black/white, `I` eyedropper (also Alt+click while painting; 1 px up to a 31×31 average, layer or all layers) |
| Clone Stamp / Healing | `S` Clone Stamp; `J` Spot Healing (just paint over the spot, the source is found automatically), `J` again: Healing Brush. Clone Stamp and Healing Brush: Alt+click sets the source, then paint. On release the Healing Brush matches color and brightness to the surroundings. Options: sample all layers (stamp onto an empty layer), aligned. A cross shows where pixels are copied from |
| Dodge, Burn, Sponge | `O` (again: next): Dodge lightens, Burn darkens (range shadows/midtones/highlights, exposure), Sponge raises or lowers saturation. One stroke acts at most once; more strokes add up |
| Blur & Sharpen | In the Smudge tool's flyout: blurs or sharpens under the brush (strength) |
| Content-Aware Fill | Select the object generously, Shift+Backspace (Ctrl+Shift+Backspace: learn from all layers, e.g. on an empty layer). Runs in the background |
| Text & shapes | `T` Text: click into the image to create a text and type right away; click into a text to place the cursor (double-click word, triple-click line, Shift extends, drag selects). Arrows/Home/End (with Ctrl by word, with Shift selecting), Ctrl+A all, Enter new line, Esc or Ctrl+Enter finishes. Font, size, color under Properties. Shift+`U` Shape: drag a rectangle/ellipse. Both stay sharp and editable, also after Ctrl+`T`; Ctrl+Alt+`R` rasterizes |
| Pen | `P`: click = corner, drag = curve with handles, click the start point to close (filled shape), Enter/Esc finishes open (line), Backspace removes the last point. On a path: drag points and handles, Alt+click corner ↔ curve, Alt+drag a single handle, `Del` deletes the point. Color = brush color, later under Properties (fill/stroke). Shift+`P` cycles the palette color |
| Layer styles | Panel "Styles" on the right: stroke, drop shadow, inner/outer shadow and glow, color overlay, bevel & emboss – on/off, adjust, remove; they stay editable |
| View | Mouse wheel pans (Shift: sideways), Alt+wheel zooms (as in Photoshop), Alt+Shift+wheel rotates, Space+drag pans; zooming out stops at ~128 px image size, `R`+drag rotates, Ctrl+`0` fit, Ctrl+`1` 100 %, `Q` Quick Mask view |
| Selection | Drag with Shift/Alt/both = add/subtract/intersect; Ctrl+`A` / Ctrl+`D` / Ctrl+Shift+`I`; Shift+`F6` feather (dialog) |
| Rulers & guides | Ctrl+`R` rulers. Drag a guide out of a ruler (Shift: onto the ticks), move it with the Move tool or Ctrl, drag it back into the ruler to delete it. View › New Guide … for exact positions. Ctrl+`;` show/hide, Ctrl+Alt+`;` clear all. Guides belong to the document (undo, `.mosaix`) |
| Grid & snapping | Ctrl+`'` grid (View › Grid Settings: spacing, subdivisions). Ctrl+Shift+`;` snap: moving, marquee, shapes, crop and transform snap to guides, image edges, image center and the (visible) grid |
| Modify selection | Select › Modify: expand, contract (round corners), border (soft ring around the edge), smooth (round corners, crumbs gone). Select › Color Range …: select a color everywhere, soft by fuzziness, with a live preview; click into the image to pick the color |
| Save selection | Select › Save Selection … stores it as a channel in the document (saved in `.mosaix`, follows canvas size, rotation and image size). Load Selection: click replaces, Shift adds, Alt subtracts, Shift+Alt intersects. PSD export does not include channels yet |
| Transform | Ctrl+`T`: corners scale (Shift free), edges stretch, outside rotates (Shift 15°), Alt around the center; Enter applies, Esc cancels. With a selection only the selected pixels (also `V`) |
| Skew, Distort, Perspective | Edit › Transform or switch in the options bar. Ctrl + drag a corner distorts freely, Ctrl + edge moves the edge. Skew: edges slide along themselves. Perspective: the neighboring corner moves the opposite way. Distort and Perspective work on pixel layers only (rasterize text, shapes and RAW first) |
| Rotate & flip | While transforming: type an angle, 90° left/right, 180°, flip (panel on the right). Layer: Ctrl+Alt+arrow left/right/down, Ctrl+Alt+`H`/`V`; whole image: add Shift (Image and Layer menus) |
| Liquify | Ctrl+Shift+`X`: `W` push, `R` reconstruct, `C` twirl (Alt: counter-clockwise), `S` pucker, `B` bloat (Alt swaps), `[` `]` size; Enter applies, Esc cancels, Ctrl+`Z` takes back the last stroke |
| Magic Wand | Click selects similar colors, Shift/Alt combine, `[` `]` tolerance, `W` again: contiguous ↔ anywhere |
| Crop | `C`, drag/adjust the frame, beyond the edge enlarges the canvas, Enter / Esc. Nothing is deleted |
| Fill & delete | Alt+Backspace fills the selection (or the layer), `Del` or Backspace deletes the selected pixels |
| Layers | Ctrl+Shift+`N` new, Ctrl+`J` duplicate, `Del` (without selection) delete, Alt+`[` `]` select, Ctrl+`[` `]` move, Ctrl+Alt+`G` clipping mask (clips to the layer below; arrow in the layer list, the base layer's opacity/blend mode apply to the group) |
| Merge layers | Layer › Merge Down (Ctrl+`E`): the active layer onto the one below. Merge Visible (Ctrl+Alt+`E`): all visible layers; hidden ones stay. Flatten Image (Ctrl+Alt+Shift+`F`): everything into one layer; hidden layers are dropped, transparency is kept. The result looks exactly as before (blend mode, opacity, masks, styles and text are baked in), and content outside the canvas is kept |
| Layer properties | Shift+digit opacity, Shift+`+`/`-` blend mode, Ctrl+`,` show/hide, Ctrl+`G` group, Ctrl+Shift+`G` ungroup, `F7` blend space |
| Generative Fill | Make a selection, Edit › Generative Fill …, describe what should appear. Providers: OpenAI (enter and save the key in the dialog – it goes to `~/.config/mosaix/openai-key`, readable only by you – or set `OPENAI_API_KEY`; only after you explicitly agree in the dialog, because the image area is sent out) or your own program (`program image.png mask.png prompt out.png`, e.g. a local Stable Diffusion script). The model sees all visible layers; by default the result goes onto a new layer above the active one (only the filled area, soft at the edge, in place – switch off "Result on a new layer" to paint into the active layer instead). One undo step. The last OpenAI exchange is kept in `~/.cache/mosaix/generative-last/` for checking |
| Smart objects | File › Place Embedded … brings in an image as a smart object (centered, fitted if too large); Layer › Convert to Smart Object turns a pixel layer into one. Scaling, rotating and transforming stay lossless because they always draw from the embedded original. Properties: Replace Contents …, Rasterize. Distort/Perspective only after rasterizing; PSD gets them as pixels |
| Adjustment layers | Layer › New Adjustment Layer: Brightness/Contrast, Levels, Curves, Exposure, Vibrance, Hue/Saturation, Black & White, Color Balance (shadows/midtones/highlights, preserve luminosity), Channel Mixer (also monochrome), Gradient Map, Invert, Posterize, Threshold. All stay editable and go to PSD as Photoshop adjustment layers |
| Filters | Image › Filter: Gaussian Blur, Unsharp Mask, High Pass, Shadows/Highlights (lightens dark and tames bright areas based on their surroundings), Motion Blur, Median (removes dust and crumbs, keeps edges), Mosaic, Noise; also as smart filters |
| Layer masks | Ctrl+Shift+`M` new (from the selection), `\` paint mask ↔ pixels, Ctrl+Alt+`M` on/off, Ctrl+Alt+Shift+`M` apply |
| History & file | Ctrl+`Z` / Ctrl+Shift+`Z` / Ctrl+`Y`, Ctrl+`S` save (in the background), Ctrl+Shift+`E` export (in the background), Ctrl+Alt+Shift+`E` export as (color space, 16 bit, JPEG quality) |
| Clipboard | Ctrl+`C` copies (text while editing, otherwise the layer's pixels in the selection), Ctrl+Shift+`C` from all layers, Ctrl+`X` cut, Ctrl+`V` paste (image → new layer, at the old position if it came from Mosaix; text → into the edited text or as a new text layer). Also in the interface's text fields |
| Keyboard shortcuts | Edit › Keyboard Shortcuts … (Ctrl+Alt+Shift+`K`): rebind any menu command (click the shortcut, press the keys; Backspace removes, Esc cancels); a shortcut another command had is removed there; "Default" or "Reset All" |
| Brush tips | Edit › Import Brushes … (Ctrl+Alt+Shift+`B`): Photoshop brushes (.abr), GIMP brushes (.gbr/.gih) or an image become your own tips (folder `~/.local/share/mosaix/brushes`); pick them in the tip menu of the options bar, Shift+`B` cycles through all |
| Navigator & Info | Panels on the right (or the Window menu): the Navigator shows the whole image with the visible area (click or drag to look elsewhere, zoom slider); Info shows the histogram, position and color under the pointer |
| Other | `F12` benchmark, `F1` help, Ctrl+`Q` quit |

## Layout

- `crates/mosaix-core` – document model, blend modes (CPU reference), history, tile math, view, brush stroke → dabs (no GPU)
- `crates/mosaix-gpu` – wgpu engine: tile pools, stroke buffer, composite cache, WGSL shaders
- `crates/mosaix-io` – native `.mosaix` format (append-only), export
- `crates/mosaix-app` – Wayland (smithay-client-toolkit), tablet protocol, input, loader, headless CLI, agent interface (`src/agent/`: commands, MCP, socket)
