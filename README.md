# Mosaix

A layered image editor for Linux that does its work on the GPU. I wanted
something that feels like Photoshop, opens a 30 000 × 30 000 px document
without blinking, and doesn't make me fight the interface the way GIMP does.
This is that attempt.

It's young and moves fast, and most of it has been tested by one person. Expect
rough edges, and please report them.

## What it does

- **Layers:** groups, masks, clipping masks, blend modes, adjustment layers,
  smart filters and the full set of Photoshop layer styles. Everything stays
  editable.
- **Painting and retouching:** brushes (including imported `.abr` and GIMP
  brushes), smudge, clone stamp, healing brush, spot healing, content-aware
  fill and liquify.
- **Text and vectors:** text with OpenType features and text on a path, shapes,
  and a pen tool for Bézier paths.
- **Camera RAW:** CR2/CR3, NEF, ARW, RAF, DNG and more, kept as a RAW layer you
  can re-develop at any time. Lens correction comes from lensfun.
- **Files:** PSD/PSB in and out, with layers, masks, styles and editable text.
  Also PNG, JPEG, WebP, AVIF, JPEG XL, TIFF and EXR, with colour profiles. The
  native `.mosaix` format saves incrementally in the background.
- **Local AI, optional:** select subject, remove background, remove objects
  (LaMa) and upscale (Real-ESRGAN). It runs on your machine through ONNX
  Runtime. Models are only downloaded when you ask for them.
- **Automation:** recordable actions, scripts, batch processing and filter
  plugins written in WGSL.
- **Agents:** every tool is also an MCP tool, so an AI agent can edit images,
  headless or live in the open window.

Keyboard shortcuts follow Photoshop. The interface is in English and German.

## Requirements

- A Wayland compositor. There is no X11 support.
- A Vulkan driver with descriptor indexing, for example RADV, ANV or NVIDIA.
- Optional: `onnxruntime` for the AI tools, `lensfun` for lens profiles.

## Install

On Arch, build the package from `packaging/arch`:

```sh
cd packaging/arch && makepkg -si
```

Or build it yourself with a recent stable Rust:

```sh
cargo build --release
./target/release/mosaix photo.jpg
```

`mosaix --help` lists the command-line options, among them headless
conversion (`mosaix in.psd --export out.png`) and batch processing.

## Use it from an agent

```sh
claude mcp add mosaix -- mosaix mcp --attach
```

With `--attach`, the agent works in the window you have open and you watch it
happen. Without it, the agent gets its own headless editor. See
[docs/agenten.md](docs/agenten.md).

## More

- [docs/bedienung.md](docs/bedienung.md): every tool and shortcut (German)
- [docs/aktionen.md](docs/aktionen.md): actions and scripts (German)
- [docs/filter-plugins.md](docs/filter-plugins.md): writing filter plugins
  (German)
- [PLAN.md](PLAN.md): design notes and roadmap (German)

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
