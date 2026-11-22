# mosaix – Plan for a GPU-first image editor on Linux

> Goal: a Photoshop replacement that feels **instant**, is fun to use
> and uses the GPU everywhere it makes sense.
> Three pillars, equally weighted: **Performance · Usability · Features.**

Target hardware for development: AMD RX 6700 XT (Navi 22, 12 GB VRAM, RADV/Vulkan), Wayland, Arch.

---

## 1. Guiding principles

1. **The UI thread never blocks.** Anything that can take > 2 ms runs asynchronously, is cancellable and shows a preview immediately.
2. **Only compute what you see.** Dirty tiles + level of detail: at 25 % zoom we composite at mip level 2 (16× less work), full resolution only when zooming in/exporting.
3. **Everything is non-destructive where possible.** Filters and adjustments are layers/nodes with live parameters, not modal dialogs with an "OK" button.
4. **Everything is a command.** Every action is a serialisable command → undo, macros, scripting, headless batch and the command palette come "for free".
5. **Linux-only is a feature.** We use Wayland protocols directly (tablet, color management/HDR, fractional scaling, presentation time) instead of aiming for the lowest common denominator.
6. **Agents are first-class users.** Every function is also reachable as a command (MCP tool / JSON); agents can *see* the result (rendered preview), and on request they work live in the same document as the human – with shared undo.
7. **Measure, don't guess.** Performance budgets (section 7) are part of CI, profiling (Tracy) is built in from day 1.

---

## 2. Technology stack

| Area | Choice | Rationale |
|---|---|---|
| Language | **Rust** | Memory safety without GC pauses, good threading (rayon), strong GPU ecosystem |
| GPU API | **wgpu** (Vulkan backend) with native features (`shader-f16`, `subgroup`, `push-constants`, `timestamp-query`, `texture-binding-array`) | Modern API, WGSL compute, well maintained. Escape hatch: raw Vulkan via `wgpu-hal` or `ash` if needed |
| Window/input | **smithay-client-toolkit** directly on Wayland (no winit): `zwp_tablet_v2`, `wp_fractional_scale_v1` + `wp_viewporter`, `wp_cursor_shape`, frame callbacks; later `wp_color_management_v1`, `wp_presentation` | winit doesn't deliver full tablet data on Wayland; direct protocol access gives full control (decided in Phase 0) |
| UI | **egui** + `egui_dock`, own theme, own widgets; in the same wgpu device as the canvas | One GPU context, no texture copies between toolkit and canvas; very fast; Rerun/Graphite prove that it scales |
| Vector & text | **vello** (GPU compute 2D renderer), **parley/cosmic-text** + **swash** for layout/shaping | Vector layers, shapes, text and paths entirely on the GPU |
| Color management | **lcms2** (ICC), optionally **OpenColorIO** | Display transform is baked into the final shader as a 3D LUT |
| Parallelism | **rayon** (CPU jobs), own render thread, `crossbeam` channels | |
| Compression | **zstd** / **lz4** | Tile swap, native file format |
| Profiling | **tracy** (`tracing-tracy`) + GPU timestamp queries | |
| ML (optional) | **ort** (ONNX Runtime, ROCm/MIGraphX EP) or **burn** with wgpu backend | Local models, no cloud requirement |

**References for learning (not a base):** *Krita* (brush engine, tablet handling, tile storage) and *Graphite* – the latter is a node-based **vector** editor, so not a raster competitor, but useful as illustrative material for Rust + wgpu + vello in a desktop app.

---

## 3. Architecture

```
┌──────────────────────────── mosaix-app ────────────────────────────┐
│  Wayland/winit · Input (Tablet, Mouse, Keyboard) · Keymap · Palette  │
│                                                                      │
│   ┌────────── mosaix-ui (egui) ──────────┐   ┌──── Canvas-View ───┐ │
│   │ Panels · Docking · Inspector · HUDs   │   │ Pan/Zoom/Rotate    │ │
│   └───────────────┬───────────────────────┘   │ Overlays/Gizmos    │ │
│                   │ Commands                   └─────────┬──────────┘ │
└───────────────────┼──────────────────────────────────────┼────────────┘
                    ▼                                      │ View-Request (Rect, LOD)
          ┌──── mosaix-core ────┐                         ▼
          │ Document model       │──Snapshot──▶ ┌──── mosaix-render ────┐
          │ (persistent, CoW)    │              │ Render-Graph (DAG)     │
          │ Commands · Undo      │              │ Dirty-Tracking · Cache │
          └──────────┬───────────┘              │ Scheduler · LOD        │
                     │                          └──────────┬─────────────┘
          ┌──────────▼───────────┐              ┌──────────▼─────────────┐
          │ mosaix-tiles        │◀────────────▶│ mosaix-gpu            │
          │ CPU-Tiles, Swap,     │   Upload/    │ Device, Tile-Pool      │
          │ Compression          │   Readback   │ (Texture-Arrays), LRU  │
          └──────────────────────┘              └──────────┬─────────────┘
                                                ┌──────────▼─────────────┐
                                                │ mosaix-ops (WGSL)     │
                                                │ Blend · Filter · Brush │
                                                └────────────────────────┘
   mosaix-io (formats)   mosaix-color (ICC/LUT)   mosaix-cli (headless)
```

### 3.1 Document model
- Tree of nodes: group, pixel layer, adjustment layer, filter layer, vector layer, text layer, smart object, mask (pixel + vector), clipping.
- **Persistent data structure** (`Arc` + copy-on-write at tile level). An undo step costs only the tiles actually changed. Snapshots for the renderer are O(1).
- Precision selectable per document: **8 bit**, **16 bit float (default)**, **32 bit float**.

### 3.2 Tiles & memory
- Tile size 256×256 (tunable), **sparse**: empty/single-color tiles are not allocated.
- GPU side: tile pool as large `Texture2DArray`s per format; page table in a storage buffer. Shaders address tiles indirectly ("virtual texturing light", without sparse binding).
- Residency manager with LRU and VRAM budget (e.g. 70 % of 12 GB). Non-visible tiles move to RAM (lz4), under pressure to disk (zstd, mmap).
- **Mip pyramid per layer**, updated incrementally so zoomed-out views are cheap.

### 3.3 Render graph
- The layer stack is compiled into a DAG; each node = compute/render pass.
- **Dirty regions** propagate through the graph (blur grows the region by its radius, etc.).
- Cache for intermediate results (e.g. group composite below the active layer) → while painting, only "cache below + active layer + cache above" is composited.
- **Progressive refinement:** show low LOD immediately first, then recompute at full resolution. Jobs carry a generation ID and are discarded when parameters change (dragging a slider ⇒ no backlog).
- Filters declare whether they are **scale-invariant** (blur with scaled radius) or need full resolution.

### 3.4 Threads
- **UI thread:** input, egui, command dispatch. Never GPU waits.
- **Render thread:** (deferred, see status Phase 2b) owns the queue, builds frames, presents (mailbox/FIFO-relaxed, frame pacing via `wp_presentation`).
- **Worker pool (rayon):** decoding, encoding, compression, CPU fallbacks, histograms.
- **IO thread(s):** loading/saving, autosave.

### 3.5 Color
- Working color space: **linear** (default: linear Rec.2020 or ACEScg selectable), optionally **"Photoshop-compatible gamma blending"** – users expect the PS look with blend modes, linear blending is physically more correct. Offer both, per document.
- Display transform (monitor ICC via lcms2) → 3D LUT → last shader pass.
- HDR/wide-gamut output via `wp_color_management_v1` (KWin/Mutter support it).
- Soft proofing (CMYK/print) as an additional LUT pass.

---

## 4. Feature catalogue (by priority)

### P0 – Foundation (without this it's no replacement)
- Open/save: PNG, JPEG, WebP, TIFF, AVIF, **JPEG XL**, EXR, native format
- Layers, groups, opacity, all common **blend modes** (PS parity), layer masks, clipping
- Pan / zoom / **canvas rotation**, smooth at any document size
- Brush, eraser, color picker, fill, gradient
- Selection: rectangle, ellipse, lasso, polygon, **magic wand (GPU flood fill)**, color range, add/sub/intersect, feather, quick mask
- Free transform (scale, rotate, distort, perspective), crop with straighten
- Unlimited undo/redo with history panel
- Copy/paste with the Wayland clipboard (incl. image data)

### P1 – What makes it a working tool
- **Adjustment layers (live):** Levels, Curves, Hue/Saturation, Color Balance, Vibrance, Exposure, Black & White, Channel Mixer, Selective Color, Color Lookup (.cube), Gradient Map, Threshold, Posterize
- **Filter layers (live):** Gaussian/box/motion/lens blur (bokeh), Unsharp Mask, High Pass, add/reduce noise, Clarity/local contrast, Liquify (real-time displacement field)
- Retouching: **clone stamp, healing brush (Poisson/multigrid on GPU), spot healing, content-aware fill (PatchMatch on GPU)**
- Brush engine v1: pressure/tilt/rotation curves, stabiliser, texture/image tips, smudge/mixer, ABR import
- Text layers (shaping, OpenType features, paragraph text)
- Vector shapes and paths (pen tool) via vello
- **PSD import/export** (layers, masks, blend modes, adjustment layers as far as they can be mapped) – crucial for adoption
- Complete color management, soft proofing
- Warp, puppet warp
- Layer styles: drop shadow, stroke, glow, color overlay (as GPU effects)

### P2 – Extras
- **RAW development** (`rawler`, demosaicing on GPU) as a Camera-Raw-like smart filter
- AI (local, optional): select subject / segmentation (SAM-like), generative fill (inpainting), upscaling, denoising
- Macros/actions (record & play back the command log), **batch via CLI** (`mosaix-cli run makro.toml *.jpg`)
- Plugins: **WASM** (wasmtime, sandboxed) + **custom WGSL filters** as a plugin type
- Scripting (Rhai or Lua)
- Artboards, slices/Export As with multiple sizes
- Frequency separation as a one-click workflow, panorama/focus stacking

### Deliberately NOT (for now)
3D, video timeline, animation, prepress specialities. Focus: **photo editing, compositing, retouching, digital painting.**

---

## 5. Usability – "fun to use"

- **Command palette** (`Ctrl+K` / `Space` tap): every action, every tool, every layer, every filter searchable – with fuzzy search and recently used.
- **Keymap presets:** Photoshop, GIMP, Krita – plus freely configurable. Switchers should be productive within 5 minutes.
- **On-canvas instead of dialogs:** brush size/hardness via `Alt+right-click drag` (HUD), transform gizmos, gradient directly on the canvas, curves adjustment by dragging in the image (targeted adjustment).
- **Scrubby sliders:** dragging on any number field label changes the value; Shift = fine, Ctrl = coarse. Values can also be entered as an expression (`120*2`).
- **Live preview on hover:** blend modes, filter presets, LUTs – preview without clicking.
- **No spinner in normal operation.** If something takes longer: progressive preview + subtle progress indicator, cancellable at any time.
- Smooth animations (zoom with inertia, panel transitions) – but can be switched off at any time.
- Tabs + fast document switching, **crash recovery and autosave** (incremental, costs nothing).
- Consistent, honest error messages; never data loss.
- Tooltips with shortcut, "What's this?" hover explanations.
- Accessibility: scalable UI (fractional scaling), contrast theme, everything reachable by keyboard.

---

## 6. Native file format (`.mosaix`)

- Container with manifest (document tree, JSON/CBOR) + tile blobs (zstd).
- **Append-only/incremental:** saving writes only changed tiles → saving large files in milliseconds. Occasional compaction in the background.
- Embedded preview (for file manager thumbnailers, ship a `.thumbnailer` file).
- Versioned, with forward-compatibility rules.
- Export to OpenRaster (`.ora`) for interop with Krita/GIMP.

---

## 7. Performance budgets (reference: RX 6700 XT, Ryzen desktop)

| Scenario | Budget |
|---|---|
| Cold start until usable | < 400 ms |
| Open 50 MP JPEG → first pixels visible | < 300 ms (full resolution < 800 ms) |
| Pan/zoom/rotation, 100 layers, 8K document | Screen refresh rate (up to 165 Hz), no frame drops |
| Brush: input → pixels in the swapchain image | < 1 frame (+ compositor) |
| Slider on an adjustment layer (24 MP) | Preview in the same frame |
| Gaussian 100 px on 24 MP, full resolution | < 30 ms |
| Content-aware fill, 1 MP hole | < 1 s (progressive preview immediately) |
| Save (incremental) after a brush stroke | < 50 ms |
| RAM at 8K × 8K × 50 layers (16F) | Stays within budget thanks to sparse tiles + swap |

Measured in a **benchmark suite** (criterion + GPU timestamps); regressions > 5 % break the build.

---

## 8. Crate structure

```
mosaix/
├── crates/
│   ├── mosaix-core     # document model, commands, undo (no GPU code)
│   ├── mosaix-tiles    # tile storage, CoW, swap, compression
│   ├── mosaix-gpu      # device, tile pool, residency, uploads
│   ├── mosaix-render   # render graph, scheduler, LOD, caches
│   ├── mosaix-ops      # WGSL shaders: blend, filters, adjustments
│   ├── mosaix-brush    # brush engine, stroke processing, stabiliser
│   ├── mosaix-select   # selection algorithms
│   ├── mosaix-color    # ICC, LUTs, color spaces
│   ├── mosaix-io       # formats (PSD, JXL, EXR, …), native format
│   ├── mosaix-ui       # egui panels, widgets, theme, palette
│   ├── mosaix-app      # binary: window, Wayland, input, glue
│   └── mosaix-cli      # headless batch processing
├── shaders/             # WGSL, hot reload in debug build
├── benches/
└── tests/golden/        # reference images for pixel tests
```

---

## 9. Quality assurance

- **Golden image tests:** every operation against reference images (with tolerance, ΔE metric). Blend modes additionally against Photoshop reference exports.
- **Fuzzing** of all file parsers (`cargo-fuzz`) – PSD and RAW are minefields.
- Property tests for undo/redo (random command sequences → undo everything = original state).
- **Shader hot reload** and a debug overlay (tile boundaries, dirty regions, residency, frame timing).
- CI: build + tests + benchmarks; GPU tests on lavapipe (CPU Vulkan) in CI, real GPU locally.

---

## 10. Roadmap

| Phase | Duration (rough, 1 person) | Result |
|---|---|---|
| **0 – Spike** | 2–4 weeks | Window on Wayland, wgpu device, **tablet input with pressure**, sparse tile canvas, open a huge image, pan/zoom/rotate, Tracy. *Go/no-go for the stack.* |
| **1 – Core** | 6–8 weeks | Document model, layers/groups, blend modes, undo, render graph with dirty tracking & LOD, native format, PNG/JPEG/WebP |
| **2 – Painting & selecting** | 6–8 weeks | Brush engine v1, eraser, fill, gradient, selection tools, masks, transform, crop |
| **3 – UI polish (MVP)** | 4 weeks | Docking layout, command palette, keymaps, HUDs, scrubby sliders, autosave → **first usable release** |
| **4 – Non-destructive** | 6–8 weeks | Adjustment & filter layers, Liquify, complete color management |
| **5 – Retouching** | 6 weeks | Clone stamp, healing brush, content-aware fill |
| **6 – Text & vector** | 6 weeks | vello integration, text layers, shapes, paths, layer styles |
| **7 – Interop** | 6 weeks | PSD import/export, complete TIFF/EXR/JXL/AVIF, ABR brushes |
| **8 – Extensibility & extras** | ongoing | Macros, CLI batch, WASM plugins, RAW, local AI |

Realistically: **MVP after ~5–6 months**, "serious Photoshop replacement for photo/retouching" after ~12–18 months full-time.

---

## 11. Risks & countermeasures

| Risk | Countermeasure |
|---|---|
| Tablet support on Wayland (winit) incomplete | Phase 0 settles this first; direct `tablet-v2` binding via a second event queue on the same Wayland connection |
| wgpu limits (no sparse binding, f16 storage) | Indirect tile addressing instead of sparse; `wgpu-hal`/`ash` as a way out for individual passes |
| PSD compatibility is endless | Clear compatibility scope, fallback: rasterised layer + warning, test corpus of real files |
| egui feels "like a dev tool" | Invest early in own theme/widgets; encapsulate the UI behind its own layer so a switch would remain possible |
| Scope explosion (Photoshop has a 35-year head start) | Strict prioritisation (P0/P1/P2), focus on photo/retouching/painting; every phase ends with something usable |
| Driver bugs (RADV is good, but …) | Golden tests on several drivers (RADV, NVIDIA proprietary, Intel ANV, lavapipe) |

---

## 12. Open decisions

1. **UI toolkit:** egui (recommended, fastest route, one GPU context) vs. own retained UI vs. Qt/QML shell with Rust core (best tablet/IME handling, but two languages).
2. **Default blending:** linear (correct) or gamma (PS-compatible) as the default?
3. **X11 support:** drop entirely or later via winit as a fallback?
4. **License:** GPL-3 (like GIMP/Krita) or MPL-2/Apache-2?

---

## Status Phase 0 (20.11.2026)

Implemented and measured on RX 6700 XT / RADV / niri, 1872×2046 px window:

- Sparse tile canvas (256², RGBA16F linear, premultiplied) in a growing GPU texture array with a complete mip chain per tile; mips via compute only for changed tiles
- Brush/eraser via hardware blending directly into the tiles, pressure → size/opacity, smoothing, even dab spacing
- Tablet via `tablet-v2` (pressure, tilt, stylus buttons, eraser end), fractional scaling, frame-callback-driven rendering (mailbox)
- Streaming load: document size immediately, tiles converted in parallel, uploaded from the centre outwards
- Benchmark built in (`--bench`, F12)

| Measurement | Result | Budget |
|---|---|---|
| 100 MP (10000², 1600 tiles), zoom/rotation/pan | 144 fps (= monitor), CPU avg 0.07–0.2 ms/frame | Refresh rate ✅ |
| 48 MP JPEG (38 MB, worst-case noise) | decoded 271 ms, complete 354 ms | < 300 / < 800 ms ✅ |

**Known limits (deliberate, for Phase 1):** a single texture array → max. 2048 tiles (~130 MP fully occupied; empty documents any size); no layers, no undo; stroke opacity accumulates (no stroke buffer); rendering still on the UI thread; tablet pressure still to be verified with a real stylus.

## Status Phase 1 (20.11.2026) – Engine

Implemented (operation still via keyboard, layer list in the terminal – the UI comes in Phase 3):

- **Document model:** layers, groups (isolated, nestable up to 7 deep), opacity, visibility, 19 blend modes (W3C/Photoshop formulas) with a CPU reference; GPU deviates by at most 7.6·10⁻⁴
- **Blend space per document:** sRGB (Photoshop-compatible, default) or linear – F7. *Open decision 2 is thus provisionally decided.*
- **Undo/redo on the GPU:** document snapshots with shared, ref-counted tiles; a step costs only the changed tiles; duplicating a layer costs nothing
- **Stroke buffer:** flow builds up within a stroke, opacity caps it (like Photoshop); eraser via the same mechanism
- **Tile pools in blocks** (256 tiles/block, binding arrays): no copying when growing, up to 16 384 layer + 16 384 image tiles – the 130 MP limit from Phase 0 is gone
- **Composite cache with signatures:** the layer program is hashed per tile; only tiles with a changed signature are recomposited – visible ones first, within a per-frame budget
- **Native format `.mosaix`:** zstd tiles + JSON manifest, append-only (saving writes only changed tiles), recovery after a crash while appending, compaction when there is a lot of dead space
- **Export** PNG/JPEG/WebP/TIFF; **headless CLI** (`--save`, `--export`, `--bench-offscreen`)

| Measurement (RX 6700 XT) | Result |
|---|---|
| Display 100 MP / 192 MP, 2560×1440, incl. GPU wait time | avg 0.14 / 0.17 ms per frame |
| Change blend mode → recomposite 100 MP, 2 layers completely (incl. mips) | 10–11 ms |
| Save (complete) / load 100 MP `.mosaix` | 457 ms / 96 ms |
| Incremental save after a brush stroke | only touched tiles (test: 1 of 16) |
| PNG → .mosaix → PNG | bit-exact identical |

**Known limits:** no UI (Phase 3), no file dialog (saves next to the source file); saving still runs on the UI thread (first save of 100 MP ≈ 0.5 s pause, incremental afterwards); brush edges are blended linearly; pass-through groups are missing; tiles live entirely in VRAM (paging out = residency manager, later); the window benchmark needs a visible window (niri throttles hidden windows) – that's what `--bench-offscreen` is for.

## Agent interface (since 20.11.2026)

Cross-cutting topic alongside the phases; details in [docs/agents.md](docs/agents.md).

- Shared command layer (22 commands: document, layers, painting, filling, placing an image, querying colors, preview, history, save/export) with strict parameter validation
- **MCP server** over stdio (`mosaix mcp`), preview as image content
- **Live mode** `--attach`: Unix socket per window, agent and human share the document and undo history
- **JSON Lines scripts** (`mosaix run`) for shell agents and batch
- Tested: protocol handshake, tool↔command consistency, complete agent flow against the real GPU engine, live control of a window, cleanup of orphaned sockets

**Rule for all following phases:** every new tool immediately gets its command and its MCP tool. In the medium term the app's keyboard shortcuts will also go through the command layer; then macros and recording come for free.

## Status Phase 2a (20.11.2026) – Selection & masks

- **Mask infrastructure:** own R8 tile pool; uniform tiles stay `Solid` (no memory) – a rectangular selection over 100 MP occupies only the edge tiles
- **Selection:** rectangle, ellipse, lasso/polygon with anti-aliasing (4× vertical, analytic horizontal); add/subtract/intersect (Photoshop modifiers), all/none/invert, feather (3× box ≈ Gaussian); marching ants via a screen mask (always 1 px sharp, animated), quick mask view
- **Selection constrains everything:** brush (already in the live preview), fill, delete, gradient – soft selection edges too
- **Layer masks** for layers and groups: create (from selection), paint on (live preview), on/off, apply; in the compositing program, a mask tile with value 0 skips the layer entirely
- **Apply kernel:** one GPU path for committing a stroke, fill, gradient (sRGB interpolation like Photoshop), delete, move (pixels + mask), apply mask – always as new tiles, hence one undo step
- **Tools in the window:** M/L/G/V/B/E, overlays for the marquee, lasso, gradient line and move outline
- **Background saving:** snapshot + asynchronous GPU readback, compression/writing in a worker thread; keep working in the meantime
- **Format v2:** layer masks and selection are saved (v1 files remain readable)
- **Agents:** 15 new commands (selection, fill, gradient, masks, move), preview with selection tint

| Measurement (100 MP, RX 6700 XT, headless) | Result |
|---|---|
| Fill / gradient (1600 tiles) | 27 ms / 19 ms |
| Select ellipse / subtract rectangle | 49 ms / 15 ms |
| Feather 40 px | 428 ms (CPU) |
| Move layer including mask | 27 ms |
| Display (regression) | avg 0.14 ms/frame, unchanged |

**Known limits:** ~~moving only shows the outline while dragging~~ (now a live preview of the content); feather and selection combination run on the CPU; content moved beyond the edge is clipped; marching ants and tool overlays are only covered by an offscreen test (niri doesn't show the test window) – please try them by hand.

## Status Phase 2b (20.11.2026) – Transform, magic wand, brushes

- **Warp kernel:** affine resampling of whole layers on the GPU (nearest neighbour / bilinear / bicubic Catmull-Rom; supersampling when downscaling). Source tiles are found via a lookup table, target tiles are determined via an exact parallelogram-tile test. Replaces the old move path.
- **Floating selection:** with a selection, V and Ctrl+T move/transform only the selected pixels. The selection moves along, the hole stays behind (like Photoshop).
- **Live preview:** when moving and transforming, only the visible area (plus margin) is recomputed, without an undo step. On a view change it is recomputed.
- **Content outside the canvas is preserved** (just not displayed). Cropping is therefore lossless; enlarging the canvas brings everything back. Cropping on tile boundaries only renumbers tiles.
- **Free transform (Ctrl+T):** frame with 8 handles. Corners scale proportionally (Shift = free), edges stretch, Alt = around the centre, outside rotates (Shift = 15°), inside moves, Enter/Esc. Content bounds pixel-exact via GPU reduction.
- **Crop (C)**, canvas size and image size (agents), all layers and masks.
- **Magic wand (W) and color range:** GPU similarity map (active layer or whole image), scanline fill on the CPU with a soft 1 px edge, tolerance like Photoshop in 8-bit steps.
- **Brush engine v1:** five procedural texture tips (chalk, bristles, spray, rough, charcoal) with mipmaps; angle, roundness, follow direction; jitter for size, angle, flow and scatter (deterministic per stroke).
- **Smudge (U):** stroke tiles start as a copy of the layer. Per dab, a 128² buffer picks up color and is stamped through the tip (dabs sequential, dependent on each other).
- **UI stays responsive:** feather, magic wand fill and export encoding run in worker threads, export reads back asynchronously. Results on an outdated selection are discarded.
- **Agents:** `layer_transform`, `layer_bounds`, `crop`, `canvas_size`, `image_resize`, `select_magic_wand`, `select_color_range`; `paint_stroke` with tips, dynamics and `smudge`.

| Measurement (100 MP, RX 6700 XT, headless) | Result |
|---|---|
| Rotate 17° bicubic | ≈ 40 ms |
| Scale ×1.3 bicubic (→ 169 MP) | ≈ 73 ms |
| Whole-pixel move (exact) | ≈ 20 ms |
| Crop outside / on tile boundaries | ≈ 20 ms / < 1 ms |
| Content bounds (GPU reduction) | ≈ 19 ms |
| Magic wand contiguous / everywhere | 120–190 ms / 70 ms |
| Color range | 70 ms |

**Render thread decision:** a dedicated render thread would not have solved the actual problem. The GPU work is already asynchronous, and input goes through the UI thread anyway. The stutter came from heavy CPU work on the UI thread, which now runs in worker threads. A real render thread becomes interesting again when egui (Phase 3) costs layout time.

**Known limits:**
- At 100 MP in fit-to-screen view, the transform preview computes all visible tiles at full resolution, that's about 40 ms per change. A reduced-resolution preview is still missing.
- ~~Groups can't be transformed yet.~~ Done 2026-11-20: Move and Free Transform act on whole groups (all layers, masks, text/shapes stay editable).
- Smudge only picks up from the active layer.
- Brush tips are built in (custom tips and ABR come with Phase 7).
- New window tools (Ctrl+T, C, W, U, Shift+B, J) are only covered by engine, agent and geometry tests and need hand testing.

## Status Phase 3 (20.11.2026) – User interface (MVP)

- **egui** (0.37, egui-wgpu on the same wgpu 30 instance), without winit: Wayland input (mouse, stylus, keyboard, modifiers, key repeat) is translated by us. The UI is drawn after the canvas in the same frame.
- **Input routing:** clicks over panels or windows go to the UI, strokes started on the canvas stay with the canvas. The stylus also operates the UI. Text fields and the palette get the keyboard, otherwise the keyboard shortcuts apply.
- **Layout:** menu bar (all commands with shortcuts), toolbar with self-drawn vector icons, right sidebar with color picker (own HSV field, swatches, hex), context-dependent tool options (brush incl. tip and dynamics, smudge, magic wand, gradient), layers (visibility, blend mode, opacity with one undo step per drag, drag & drop incl. into groups, rename, mask icon) and history list with jumps. Status bar with zoom, size, cursor position, background work and the last message (all `tracing` messages land in the message window).
- **Command palette** (Ctrl+K / Ctrl+Shift+P) and menus use a command list that replays the keyboard shortcuts. Menu, palette and keyboard thus can't drift apart.
- **Dialogs:** canvas size (with anchor) and image size (aspect ratio).
- **Autosave:** every 3 min when there are changes, a full copy in the background to `~/.local/state/mosaix/autosave/`. The document's own file and incremental saving stay untouched. After saving or quitting the copy is deleted; orphaned copies are reported at startup.
- **Fit to screen** takes the panels into account. `Tab` hides the UI.
- **Cost:** about 0.4 ms per UI pass incl. GPU. Without input and without a model change, the last UI image is reused, so no UI pass runs per frame while painting.
- **Checking without a window:** `mosaix --ui-snapshot OUT.png [FILE]` renders canvas and UI offscreen (layout check via image).

- **Added later:**
  - File dialogs (Open with "Open Recent", Save As, Export As) and New dialog (presets, white/transparent), switching documents at runtime.
  - Prompt on unsaved changes (Save / Discard / Cancel), also when quitting. Saving without an own file asks for the location.
  - Settings in `~/.config/mosaix/settings.json` (brush, color, tool, magic wand, smudge, recently opened, last folder).
  - Brush HUD via Alt+right-drag.

- **Layer thumbnails:** GPU-downscaled (warp kernel, up to 8×8 supersampling) into a 64² texture array, only regenerated for changed tiles, at most 4 per frame and never during a stroke.

**Phase 3 completed** (2026-11-20): finally thumbnails for groups (stacked layer thumbnails) and masks (downscaled on the GPU, grey, red cross when off). (Customisable keyboard shortcuts: done 2026-11-20 – Edit › Keyboard Shortcuts, Ctrl+Alt+Shift+K; stored in the settings; conflicts like Photoshop.) (Docking/detachable panels: done 2026-11-20 with egui_dock, layout is saved.)

## Status Phase 4 (20.11.2026) – Non-destructive, filters, color (first part)

- **Adjustment layers:** Brightness/Contrast, Levels, Curves (monotone cubic), Exposure (linear), Vibrance, Hue/Saturation (incl. Colorize), Black & White, Invert, Posterize, Threshold.
  - Per-channel kinds are baked into a 256-entry table, the flatten shader computes the rest directly. A CPU reference checks every kind on the GPU.
  - Every adjustment layer has a mask (from the selection) that can be painted live. Opacity mixes.
  - Blend modes act as for a layer that contains the adjusted image and is clipped to what lies below (e.g. Curves in Luminosity mode).
  - Properties panel with curves editor, file format v3.
- **Filters (GPU):** Gaussian Blur (separable; from 10 px radius on a downscaled copy, matching the analytic step response), Unsharp Mask, High Pass, Noise. They act through the selection, as one undo step, with a live preview of the visible area.
- **Color management on import:** embedded ICC profiles (e.g. Display P3, Adobe RGB) are converted with moxcms to linear sRGB in the extended range. Colors outside sRGB are preserved in f16 instead of being clipped.
- **Histogram** in Levels (with draggable sliders for shadows, gamma, highlights) and Curves. It shows the image below the adjustment layer: that part is flattened in batches into temporary tiles, one mip level is read and counted; cached until something below changes.
- **Smart filters:** filters can hang non-destructively on a pixel layer (option in the filter dialog). They appear under the layer in the layer list, can be switched on and off, edited by clicking (with live preview), removed or applied to the pixels. The engine computes the filter chain once per change over the whole layer and caches the result. Composite, histogram, eyedropper and export see the filtered image; file format v4.
- **Liquify** (Ctrl+Shift+X): push, reconstruct, twirl, pucker, bloat. A displacement field (one control point every 8 px, coarser for huge images) is painted on the CPU; the GPU always samples the unchanged layer bicubically at the displaced position, so no loss of sharpness from repeated resampling. The preview computes only touched, visible tiles. Ctrl+Z undoes strokes within the session; applying is one undo step.
- **Export with color profile:** sRGB, Display P3, Adobe RGB or Rec. 2020, 8 or 16 bit (PNG/TIFF), JPEG quality. Wide color spaces embed their ICC profile (PNG, JPEG, WebP, TIFF); the conversion starts from the linear working color space, so P3 colors from the import are preserved. The program remembers the settings.
- **Agents:** `adjustment_add`, `adjustment_set`, `filter_apply`, `histogram`; `export` with `color_space`, `bit_depth`, `quality`; `filter_apply` with `smart`, `smart_filter_set`, `liquify`.

| Measurement (100 MP, RX 6700 XT) | Result |
|---|---|
| Gaussian r = 2 / 10 / 40 / 200 | 52 / 114 / 64 / 29 ms |
| Unsharp Mask r = 2 | 51 ms |
| Noise | 17 ms |
| Curves adjustment layer (complete rebuild) | 42 ms |
| Load 24 MP PNG with Display P3 profile | 133 ms (without profile 67 ms) |
| Liquify 100 MP: dab r = 300 / apply | 0.09 ms / 21 ms |
| Histogram under a layer | 36 ms, afterwards from cache 0.08 ms |
| Export 24 MP TIFF: sRGB / Display P3 / P3 16 bit | 65 / 112 / 118 ms |

**Still open in Phase 4:**
- Monitor profile: deferred (niri can't do Wayland color management, the user has no profile)

## Status Phase 5 (20.11.2026) – Retouching (started)

- **Clone Stamp** (`S`): Alt+click sets the source, a cross shows while painting where it copies from. Aligned (offset is kept) or not; the source is the layer or the whole image (then also onto an empty layer). Runs via smudge's working copy; every dab copies pixel-exactly with an integer offset (no blurring from resampling). Agent: `paint_stroke` with `clone_offset`, `clone_all_layers`.
- **Healing Brush** (`J`): paints like the stamp; on release the area (union of the brush dabs) is fitted into its surroundings via Poisson matching – structure from the source, color/brightness from the edge. Solver on the CPU: cascadic multigrid with red-black SOR, parallel over rows (4 MP area: 97 ms, 0.25 MP: 10 ms). Agent: `paint_stroke` with `heal: true`.
- **Spot Healing** (`J`, pressing `J` again switches to the Healing Brush): paint over the blemish; on release the engine finds a source itself – 64 candidate offsets (16 directions × 4 distances) are compared via the gradients in the ring around the spot – and heals from there. Agent: `paint_stroke` with `spot_heal: true`.
- **Content-Aware Fill** (Shift+Backspace; Ctrl+Shift+Backspace learns from all layers): fills the selection with structure from the surroundings (Wexler/PatchMatch, multi-scale). Coarsest level: "onion peel" filling from outside in; per level PatchMatch (propagation + random search, parallel) and weighted voting (similarity × edge proximity 1.3^−d); in the first pass of each level, edge patches compare only known pixels. Computes in the background, result as one undo step (discarded if the layer has been changed in the meantime). Agent: `content_aware_fill`.
  - Large levels (from 4000 target pixels) run on the GPU (compute shader `cafill.wgsl`: scoring, PatchMatch with jump distances 8/4/2/1 and local random search, histogram for σ², weights, voting as gather), the small ones and the onion-peel start on the CPU. Everything on the worker thread.
  - Timings: 0.07 MP hole 0.1 s (CPU 0.25 s), 1 MP hole 0.77 s incl. readback (CPU 3.6 s) – within budget (< 1 s).
  - Weakness: strictly periodic patterns (bricks) in image corners can continue offset; good in the open image and with photo textures.
- Brush smoothing now toggles with Shift+S (previously S), brush dynamics with Shift+J (previously J).

Phase 5 is finished except for a progressive preview for Content-Aware Fill.

## Status Phase 6 (20.11.2026) – Text & vector (first part)

- **Text and shape layers** (`Content::Vector`): the description is preserved (text with font, size, style, color, alignment, line spacing, tracking, optional paragraph width; shapes: rectangle with corner radius, ellipse, polygon, SVG path, with fill and stroke), plus a transformation.
  - New crate `mosaix-vector`: layout with parley (system fonts via fontique), rasterising with vello_cpu in parallel 4×4 tile blocks directly into linear f16 tiles – only where there is content.
  - The engine rasterises changed vector layers and inserts the tiles into the render document (like smart filters, which also work on text layers). Ctrl+T and Move change only the transformation: text stays sharp. Layer › Rasterize (Ctrl+Alt+R) turns it into pixels; painting on a text layer is refused with a hint. File format v5.
- **Tools:** Text (`T`: click creates text or selects existing text for editing), Shape (Shift+`U`: drag out a rectangle/ellipse, Shift square, Alt from the centre; fill in the brush color). Properties panel with text field, font list with search, size, bold/italic, alignment, color, line spacing, tracking, paragraph; for shapes corners, fill, stroke. Typing and sliders each produce one undo step.
- **Agents:** `text_add`, `shape_add`, `vector_set` (JSON merge into the description, also `transform`), `layer_rasterize`; `document_info` shows vector layers with description and bounds.

- **Rotation made discoverable:** Transform panel with angle field, 90°/180° buttons and flipping; menus Image › rotate/flip (exact, the canvas swaps width/height) and Layer › rotate/flip (around the content centre, 90° steps pixel-exact). Fixed: canvas size/crop by whole tiles didn't move vector layers along.

- **Layer styles** (on pixel, text and shape layers; "Layer Styles" panel, "fx" in the layer list): Stroke (outside/centre/inside), Drop Shadow, Inner Shadow, Outer/Inner Glow, Color Overlay, Bevel & Emboss (inner bevel, emboss, pillow emboss; light angle and altitude, depth, softness).
  - Computed on the CPU from the layer's alpha (after smart filters or rasterising): exact Euclidean distance fields (Felzenszwalb–Huttenlocher), triple box blur ≈ Gaussian, lit height field. Cached until pixels or style change; skipped above 24 MP.
  - Effects outside the shape (shadow: Multiply, glow: Screen, outer stroke) are placed in the render document as separate layers directly below the layer – with its opacity, mask and visibility – and thus really act on what lies below; inner effects are baked into the layer. File format v6. Agent: `layer_effects_set`.

- **Font picker:** window with search, filter by kind (Sans, Serif, Mono, Script, Display, Symbols – from the OS/2 family class, PANOSE and fixed pitch, name as a fallback), favourites (star) and recently used (both stored in the settings); fonts without Latin letters hidden (can be switched on). Each family with your own text as a preview; arrow keys/click show the font live in the image (engine preview, document unchanged), Enter applies, Esc reverts. Families with styles as separate system families (Plex: "… Medm", "… SmBld") are merged into one family with a style selection (937 → 294).
- Layer style stroke edges subpixel-accurate (distance to edge points from coverage and gradient), no more pixel staircases.

- **Edit text directly in the image:** cursor, selecting with the mouse (click, double/triple click, drag, Shift+click) and keyboard (arrows, word-wise with Ctrl, Home/End, Ctrl+A), typing, Backspace/Del (word-wise with Ctrl), Enter = new line, Esc/Ctrl+Enter finishes; text layers left empty disappear. Works with parley's Selection on the same layout as the rendering, also on rotated text; the UI draws the selection and cursor over the canvas. One session = one undo step.

- **Clipboard** (Wayland data device via SCTK, works in every compositor): text while editing and in text fields; pixels of the layer or of all layers in the selection as PNG (soft edges → alpha, cropped to the content); paste as a new layer (from mosaix at the old position, from outside centred) or as a text layer. Reading and writing on separate threads. An open Free Transform frame is committed when other things are changed (like Photoshop).

- **Pen** (`P`): draw Bézier paths (corner/curve, closing at the start point → filled shape, open → line) and edit points/handles (Alt: corner↔curve, handles individually), also on transformed paths. New geometry `Bezier` with nodes; overlay above the canvas. Palette color now toggles with Shift+P.

- **Layer styles extended** (2026-11-20): blend mode per effect (defaults like Photoshop), Gradient Overlay (two colors, linear/radial, angle, scale, reverse), "Use Global Light" for shadows and Bevel & Emboss (one angle for all layers, one undo step). Overlays keep soft edges.

- **OpenType features** (ligatures, small caps, oldstyle/tabular figures, fractions, ordinals, swashes, stylistic alternates, superscript/subscript, kerning); whatever the font lacks is greyed out; PSD import takes over Photoshop's switches.
- **Text on path:** Properties › Path selects a path layer (Pen), path start and flip; every glyph is placed and rotated on the curve, open paths clip overflowing text, closed ones wrap around. Editing happens in the Properties text field (no cursor on the curve).

- **Layer styles in PSD** (2026-11-20): import and export of Photoshop's `lfx2`/`lmfx` – Drop Shadow, Inner Shadow, Outer/Inner Glow (with "Range"), Bevel & Emboss, Color and Gradient Overlay, Stroke (color); blend modes (also the spelled-out names of newer versions), Global Light (resources 1037/1049), several effects of the same kind (CC lists), disabled effects. PSD export writes the layers unrendered plus style (like Photoshop), no more "– Effect" layers; round trip pixel-identical. Not carried over (warning): Satin, Pattern Overlay, stroke with gradient/pattern, outer bevel, gradients with more than two colors. Bevel & Emboss edge profile now round (steep at the edge) – thin type shows the emboss like in Photoshop.

- **Shape and fill layers from PSD** (2026-11-20): solid color fill layers (`SoCo`/`vscg`) with vector mask (`vmsk`/`vsms`) and stroke (`vstk`) become editable shape layers; a single path stays a Bézier path (Pen), several are combined into one outline with Photoshop's path operations (combine, subtract, intersect, exclude, also shape groups) via i_overlay. Fill layers without a path: rectangle over the area. Strokes now inside/centre/outside with miter/round/bevel corners and line caps (also in the Shapes panel and for agents: `stroke_align`). Gradient and pattern fills see below.
- **Fill** (2026-11-20): opacity of the layer pixels only, the layer style stays fully visible (Layers panel next to Opacity, agent `layer_set fill_opacity`, PSD `iOpa` in both directions, file format v7). Overlays are computed "inside the shape" and work even at 0 % Fill.

- **Layer mask before layer style** (2026-11-20): as in Photoshop, the effects are computed from the masked content – a stroke runs along the mask edge instead of being cut off by the mask.

- **Shape layers in PSD export** (2026-11-20): shape layers are written as Photoshop shapes (solid fill `SoCo`, vector mask `vsms`, stroke `vstk` with alignment/corners/caps, stroke width scaled with the transformation) – still editable in Photoshop; round trip pixel-identical. Ellipse paths are now closed (inside/outside strokes on ellipses previously acted as centred).

- **Editable text in PSD export** (2026-11-20): text layers are written as Photoshop text (`TySh`) – point and paragraph text, font (PostScript name of the font actually used), size, color, alignment per paragraph, line spacing, tracking, OpenType switches, transformation. The EngineData is based on Photoshop's own (the writer regenerates it byte-identically). Round trip pixel-identical (also rotated); text on path stays pixels. On import, text now sits on Photoshop's baseline (deviation from Photoshop's rendering 0.036 → 0.0095 RMSE).

- **Wide-gamut AVIF/JPEG XL** (2026-11-21): JPEG XL in all export color spaces (Display P3, Adobe RGB, Rec. 2020; signalled as color encoding, 8/16 bit, lossless bit-exact), AVIF in Display P3 and Rec. 2020 (10 bit, CICP codes in the `nclx` box; Adobe RGB has no code). On opening, AVIF `nclx` data is evaluated; profiles like "…_P3_Srgb" are no longer wrongly treated as sRGB.

- **Pattern Overlay** (2026-11-21): layer style with pattern, opacity, scale, offset (aligned to the document, blend mode like all effects). Pattern library: six built-in patterns plus PNGs in `$XDG_DATA_HOME/mosaix/patterns`; Edit › Define Pattern (Ctrl+Alt+Shift+D, visible pixels of the selection) and Import Patterns (Ctrl+Alt+Shift+Y: images, GIMP .pat). The pattern lives in the effect, documents stay self-contained. PSD: read and write the pattern block (`Patt`/`Pat2`/`Pat3`), `patternFill` in the layer style – Photoshop's "Water" pattern arrives, round trip pixel-identical. Agent: `pattern_list`, `pattern_import`, `pattern_define`, patterns by name in `layer_effects_set`. Translations with context (`msgctxt`, `tr_ctx`) for ambiguous words ("Fill" = Füllen / Fläche).

- **Gradient and pattern fills** (2026-11-21): shape layers fill with color, gradient or pattern as desired (Shapes panel: "Fill"). Gradients with any number of color and opacity stops, midpoints and smoothness; styles Linear, Radial, Angle, Reflected, Diamond; angle, scale, reverse. New gradient editor (bar with stops: click selects, drag moves, clicking next to the bar adds, dragging away or the trash can deletes), also in Gradient Overlay (which now also supports multiple colors and all styles). Geometry and smoothness measured against Photoshop's stored pixels (length = centre to edge along the angle, angle gradient clockwise, 100 % smoothness = half smoothstep): Linear/Radial 0.003–0.011 RMSE. PSD: gradient (`GdFl`) and pattern fill layers (`PtFl`, also in `vscg`) become editable shape layers, also fill layers with an empty vector mask; export writes them back (patterns into the pattern block), round trip pixel-identical. Only noise gradients stay pixels. Agent: `shape_add gradient/gradient_style/gradient_angle/pattern/pattern_scale`, `vector_set {"paint": …}`, `gradient_overlay` with `gradient`. File format v8.

- **Satin** (2026-11-21): the last missing Photoshop layer style. Shape blurred, two copies shifted by ±distance along the angle, absolute difference, invert, limited to the shape (default Multiply). Measured against Photoshop's own rendering (`fxcmp`: effects on Photoshop's stored layer pixels against the composite): blurring before the comparison, strength 0.8 × size – RMSE 0.0049 (previously missing). Inner effects now in Photoshop's fixed stacking order (pattern, gradient, color, satin, inner glow, inner shadow). PSD read/write (`ChFX`/`satinMulti`, linear contour), round trip pixel-identical; agent `satin` in `layer_effects_set`.

- **Edit text on path directly** (2026-11-21): clicks are projected onto the curve (arc length + distance → coordinates of the flat line), cursor and selection run curved along the curve; typing, selecting, double/triple click as with normal text.

### Phase 7 – Interop (started 2026-11-20)

- **PSD/PSB import** (own reader, `mosaix-io/psd.rs`): layers with names (Unicode), opacity (incl. fill opacity), blend modes, visibility, groups (section dividers), layer masks; 8/16/32 bit; RGB, grayscale, CMYK (naive); raw/RLE/ZIP/ZIP with prediction; embedded ICC profile; merged image as a fallback. Layers are streamed tile by tile in parallel (619 MB PSD with 4 layers: 1.4 s until export). Notices for clipping masks, text/smart/adjustment/fill layers (pixels only) and unknown blend modes. Tested with files from GIMP 3 and ImageMagick.

- **PSD/PSB export** (Export As … `.psd`/`.psb`, also for agents): layers, groups (isolated = "Normal"), masks, opacity, blend modes, visibility, Unicode names; text, shapes and smart filters are rasterised as displayed, layer styles stay editable as Photoshop layer styles; 8/16 bit, RLE; embedded ICC profile of the chosen color space; merged image; PSB automatically above 30 000 px. Adjustment layers are still missing as layers (only in the merged image). Checked: round trip pixel-identical, GIMP 3 and ImageMagick read the files.

- **Clipping masks** (Ctrl+Alt+G, Layer menu, agents `layer_set clip`): a layer only shows where the base layer below has pixels; base plus clipped layers are computed as their own group (clipped layers "atop"), opacity and blend mode of the base apply to the group, a hidden base hides all of them. Groups, text, shapes and adjustment layers can also be the base or clipped; effect layers inherit this. PSD import and export carry clipping masks over, `.mosaix` saves them.

- **"Pass Through" groups** (Photoshop's default, now also for new groups): the group's layers act directly on everything below (blend modes, adjustment layers), the group's opacity and mask fade the result. At the very top of a group's blend mode menu; a different blend mode makes the group isolated. Older `.mosaix` files stay isolated (unchanged appearance). PSD import/export carry it over; agents: `layer_set pass_through`.

- **PSD: adjustment layers and text.** Import and export of adjustment layers as real Photoshop adjustment layers (Levels, Curves, Brightness/Contrast modern and "Use Legacy", Hue/Saturation incl. Colorize, Exposure, Vibrance, Black & White, Invert, Posterize, Threshold); reading/writing descriptors (`psd_data.rs`). Text layers are imported editable (text, font from PostScript name, size, color, alignment, line spacing, tracking, paragraph width, transformation; position aligned to Photoshop's pixels); missing Arial/Helvetica/Times/Courier/Calibri/Cambria/Georgia are replaced by metrically identical free fonts, otherwise the pixels stay. Text export stays pixels (Photoshop's EngineData can't be verified here without Photoshop).
- **Adjustments compute like Photoshop** (measured against Photoshop's own renderings, `scripts/psd-reference.sh`): curves as a natural cubic spline, channel before composite for Levels/Curves, Brightness/Contrast after psd-tools (modern) or a straight line (legacy), Hue/Saturation in Photoshop's order, Threshold/Posterize in 8-bit steps. Deviation (RMSE) now 0.001–0.004, modern Brightness/Contrast 0.013. Hue/Saturation for individual color ranges is still missing (notice on import).
- PSD masks with a vector mask (channel −3) are read.

- **Hue/Saturation with color ranges** (Reds … Magentas, like Photoshop: trapezoid weighting on the hue, lightness in HSV, saturations via Photoshop's measured combination table from psd-tools), CPU and GPU identical; PSD reads and writes the ranges. Deviation from Photoshop now 0.002–0.007 (previously up to 0.085).

- **AVIF and JPEG XL:** opening (AVIF via dav1d, JPEG XL via jxl-oxide, also HDR/ICC) and exporting (AVIF via ravif; JPEG XL via libjxl, 8/16 bit, quality 100 = lossless – verified bit-identical with the PNG export). Export only sRGB for now.

- **Custom brush tips and ABR import:** Edit › Import Brushes … (Ctrl+Alt+Shift+B, also in the tip menu): Photoshop ABR (v1/v2 and v6–10, names from the `desc` section; verified with a real 6.2 file from Krita's test data) or any image (transparency or dark = color). Tips are stored as grey PNGs in `~/.local/share/mosaix/brushes`, up to 59 in the brush; Shift+B cycles through all of them.

**Still open in Phase 7:** editable text in PSD export, AVIF/JPEG XL in wide color spaces, brush dynamics from ABR.

### Phase 8 – Extensibility & extras (started 2026-11-21)

- **Camera RAW** (2026-11-21): CR2/CR3, NEF, ARW, RAF (also X-Trans), ORF, RW2, PEF, DNG … via `rawler` (decoding, demosaicing PPG/X-Trans bilinear, crop); own color step in `mosaix-io/raw.rs`: white balance (As Shot, Auto = gray world, Daylight, temperature/tint via the Planckian locus), camera matrix to linear sRGB without gamut clipping (scene-linear, colors outside sRGB are kept), blown highlights neutral, exposure, orientation. On opening, "Develop RAW" appears with a preview (1400 px, sliders live); command line/`--export` develop with "As Shot". 24 MP in 0.3–0.4 s. Checked against LibRaw (`dcraw_emu -w -o 1 -4`) with Canon 6D, Nikon D750, Sony A7 III, Fuji X-T2, iPhone DNG: matching except for the Canon, where rawler reads the camera's auto setting instead of "As Shot" (1,1,1). Agent: `document_open raw: {white_balance, exposure}`, `raw_info`. Tool: `cargo run --example rawdev`.
- **RAW layers** (2026-11-21): an opened or placed RAW file stays a RAW layer – the original file lives in the document (`.mosaix` stores it once, incremental saves keep reusing it; file format v9), white balance/exposure stay adjustable in the Properties panel (redeveloping 24 MP ≈ 80 ms, the demosaiced image of the most recently developed file stays in memory). Move, Ctrl+T, crop, canvas size changes act losslessly via an own transformation (bilinear with smoothed edges, strong downscaling via a pre-averaged copy). Rasterize turns it into pixels, PSD export writes pixels. Layer thumbnails now also for text, shape and RAW layers. Crate `mosaix-raw` (rawler + color step). Agent: `raw_set`, `place_image` with RAW.
- **Highlight reconstruction and noise reduction** (2026-11-21): clipped channels are estimated from the remaining ones plus the color at the edge of the area ("inpaint opposed"-like), fully clipped stays neutral; values above 1 are preserved, lowering exposure brings back skies with color and gradient (default; "Clip" as before). Wavelet noise reduction (à trous, B3 spline, soft threshold) on square-root values (shot noise uniform), noise level measured from the finest level; luminance and color separate (color from the second level at half size), default color 25. 24 MP: 0.2 s (color only) to 0.5 s. Sliders take effect on release; white balance/exposure stay fast (denoised image in the cache). Agent: `raw_set highlights/noise`.
- **Lens correction** (2026-11-21): profiles from the system's lensfun database (`/usr/share/lensfun`, updates from `lensfun-update-data` take precedence; nothing is bundled). Camera and lens are found via normalised names (all test files except the iPhone DNG: correct profile), distortion (ptlens/poly3/poly5) and color fringes (TCA linear/poly3) interpolated by focal length, vignetting (pa) by focal length and aperture; crop factor and aspect ratio differences to the calibration sensor are converted. Automatic zoom without empty borders. In the develop step (sampling per color channel), practically without extra cost. On by default; switch in dialog/Properties, agent `raw_set lens`, `raw_info lens_profile`.
- **Tone and color** (2026-11-21): contrast, highlights, shadows, vibrance, saturation (−100…100, live). Contrast/shadows in stops around middle grey (monotone), negative highlights = soft shoulder on the largest channel (reconstructed highlights above 1 become visible, colors stay in range), vibrance mainly affects pale colors. The reconstruction option is now called "Clipped Highlights". Agent: `raw_set tone`.
- **X-Trans and DNG corrections** (2026-11-21): own pipeline after rawler's scaling – Bayer still PPG (bit-identical to before), X-Trans now Markesteijn (1 pass; against LibRaw: RMSE 0.0131 → 0.0123, sharpness +25 %, 3 passes brought nothing at double the time; 0.65 s for 24 MP). DNG opcodes from phones: GainMap (lens shading, up to 3.4× in the corners on the Pixel 4a) on the mosaic before demosaicing, WarpRectilinear (distortion/color fringes per plane, tangential) and FixVignetteRadial in the develop step – precedence over lensfun, profile name "embedded in the DNG".

- **Filter plugins in WGSL** (2026-11-21, instead of WASM: pixel plugins belong on the GPU): `.wgsl` file with header lines (`@name`, `@param` number/bool/color, `@reach`, `@canvas`) and `fn process(p) -> vec4<f32>`; mosaix prepends a prelude (source pixels across tile boundaries, `sample`, sRGB, random, one function per parameter) and compiles a separate pipeline (cached by source text). Checked beforehand with naga, errors with the line of the plugin file. Runs like built-in filters: preview, smart filter (the shader lives in the document), undo, selection; 24 MP < 20 ms. Folder `~/.local/share/mosaix/filters`, menu Image › Filter › Plugins (+ reload), four examples (Vignette, Pixelate, Chromatic Aberration, Clouds). Agents: `filter_plugin_list`, `filter_plugin_install` (agents write their own filters), `filter_apply {kind: plugin}`. Docs: `docs/filter-plugins.md`.
- **Actions, scripts, batch processing** (2026-11-21): File › Actions – recording translates UI actions into agent commands (filters, adjustments, styles, RAW, layer and selection commands, marquee/lasso, move, image size …; without layer IDs, acts on the active layer), saved as `NAME.jsonl` in `~/.local/share/mosaix/actions`; playback step by step, aborting at the first error. Executable files in the folder are scripts (any language, started with `MOSAIX_SOCKET`). `mosaix batch ACTION FILES… [--out] [--format]`. Agents: `action_list`, `action_run`. Docs: `docs/actions.md`.

- **Local AI: selection** (2026-11-21): crate `mosaix-ai` with ONNX Runtime from the system (loaded at runtime, Arch: `onnxruntime-cpu`; if it's missing, the functions say so). After confirmation, models are downloaded once from huggingface.co to `~/.local/share/mosaix/models`: SAM 2.1 small (Apache-2.0, 184 MB) and BiRefNet lite (MIT, 224 MB). Object Selection tool (Shift+W; click selects, Shift/Alt combine): analysing the image once 0.8–1.7 s, every further click ~0.1 s (embedding in the cache). Select › Subject (Ctrl+Alt+S, ~5 s), Layer › Remove Background (Ctrl+Alt+B, layer mask). Computes on the CPU (Ryzen 9700X); runs in the background. Agents: `select_subject`, `select_object`, `remove_background`, `ai_model_download`.
- **Local AI: remove** (2026-11-21): LaMa (Apache-2.0, 208 MB, ONNX export by Carve, fixed 512²). Remove tool (J cycles Spot Healing → Remove → Healing Brush): paint over the distraction (pink preview), on release LaMa fills it in. With a selection: Edit › Remove (AI), Shift+Alt+Backspace (selection +4 px). Runs via the Content-Aware Fill machinery (`FillJob::run_with`, one undo step, layer-changed guard); context at least 512 px per side or ¾ of the hole size, larger areas are downscaled for the model, only the hole takes the result. ~1–1.5 s on the CPU. "Sample All Layers" as with Spot Healing. Along the way: SAM with several points now takes the best-scored proposal (previously index 0 – unusable with this export). Agent: `remove_area {all_layers, expand}`.
- **Local AI: upscaling** (2026-11-21): Image › Upscale (AI) 2×/4× with Real-ESRGAN General x4v3 (BSD-3, 5 MB; Qualcomm AI Hub export, weights byte-identical with their release, HF mirror). Fixed 128 tiles with 12 px overlap, empty tiles skipped; 2× = 4× + box downscale. All pixel layers (also outside the canvas, transparent borders via color "bleed"), the rest like Image Size – one undo step (`UpscaleJob`/`apply_upscale`). ~10 s per MP of source image on the CPU (x4plus would be ~7× slower with barely visible gain). Agent: `image_upscale {factor}`.
  Next: Remove tool (LaMa), upscaling (Real-ESRGAN), AI denoising; generative fill via the OpenAI API (provider interchangeable).

## Next concrete step

Continue Phase 6: Pen, edit text directly in the image.

## Next concrete step (Phase 0, done)

**Start Phase 0:** create the Cargo workspace, wgpu + winit window, read tablet events via `tablet-v2` and paint with pressure onto a sparse tile canvas, open a 100 MP image and pan/zoom/rotate at 165 Hz. Once that works, the stack is settled.
