//! mosaix – GPU-first image editor for Linux/Wayland.

mod agent;
mod ai;
mod app;
mod boxedit;
mod loader;
mod portal;
mod session;
mod settings;
mod tablet;
mod ui;

use std::path::PathBuf;

use anyhow::{Context, bail};
use mosaix_core::tr;
use loader::Source;

const HELP: &str = "\
mosaix – GPU image editor (Phase 2: selections, masks, transform)

Usage:
  mosaix [FILE]                open an image (PSD/PSB, PNG, JPEG, WebP, AVIF, JPEG XL, TIFF, EXR, camera RAW, .mosaix)
  mosaix --new WxH             empty document, e.g. --new 30000x30000
  mosaix --demo WxH            procedural test image, e.g. --demo 10000x10000
  mosaix ... --bench           run a pan/zoom/rotate benchmark after loading, then quit
  mosaix ... --bench-offscreen the same benchmark without a window, measures real GPU time per frame
  mosaix FILE --save A.mosaix --export B.png
                                convert without a window (both optional, can be combined)

For agents (see docs/agents.md):
  mosaix mcp [FILE]            MCP server over stdio with its own engine
  mosaix mcp --attach          MCP server that controls the running window live
  mosaix run SCRIPT.jsonl      run JSON commands line by line (`-` = stdin)
  mosaix run --attach SCRIPT   … in the running window
  mosaix batch ACTION FILE… [--out DIR] [--format png|jpg|webp|avif|jxl|tiff|psd]
                                play an action (File › Actions) on each file and export it
                                (default: next to the input in mosaix-batch/, same format)

View:
  Pan             mouse wheel (Shift: sideways), Space + drag, middle mouse button, pen button 1
  Zoom            Alt + mouse wheel (towards the cursor), Ctrl + Space + drag
  Rotate          hold R + drag, Alt + Shift + mouse wheel, pen button 2; Shift+R resets
  Fit/100 %       Ctrl+0 / Ctrl+1

Tools:
  B Brush (Shift+B: tip round/chalk/bristle/spray/rough/charcoal; Shift+J: dynamics)   U Smudge
  P Pen   I Eyedropper   S Clone Stamp (Alt+click: source)   J Spot Healing (again: Healing Brush, Alt+click: source)
  E Eraser   M Rectangular Marquee (again: ellipse)   L Lasso
  G Gradient (again: radial)   V Move (arrow keys: 1 px, with Shift 10 px)
  W Magic Wand (click; Shift/Alt combine; [ / ] tolerance; W again: contiguous ↔ anywhere)
  C Crop (drag a frame, beyond the edge = enlarge the canvas; Enter / Esc)
  With a selection, V and Ctrl+T move/transform only the selected pixels.

Free Transform (Ctrl+T):
  corners scale (proportional, Shift = free), edges stretch, Alt = around the center,
  drag outside = rotate (Shift = 15° steps), inside = move.
  Enter applies, Esc or Ctrl+Z cancels. Content outside the image is kept.

Painting:
  Paint           left mouse button / pen (pressure = size & flow), eraser end of the pen
  Size            [ / ], Ctrl + mouse wheel
  Opacity         digits 1…9, 0 (10 … 100 %; with Smudge: strength)
  Hardness/Smoothing  H / Shift+S   Color  Shift+P (palette), X swap, D black/white, I / Alt+click eyedropper
  Fill            Alt+Backspace (selection or whole layer)

Selection (while dragging: Shift = add, Alt = subtract, both = intersect):
  All / none      Ctrl+A / Ctrl+D        Inverse    Ctrl+Shift+I
  Feather         Shift+F6 (10 px)       Delete     Del or Backspace (pixels in the selection)
  Quick Mask view  Q (red = not selected)

Layer masks:
  New mask        Ctrl+Shift+M (from the selection, otherwise reveal all)
  Paint mask      \\ (toggle pixels ↔ mask; white reveals, black hides)
  On/Off          Ctrl+Alt+M             Apply      Ctrl+Alt+Shift+M

Layers (Photoshop shortcuts):
  New layer       Ctrl+Shift+N         Duplicate     Ctrl+J      Delete  Del (without selection)
  Select          Alt+[ / Alt+]        Move          Ctrl+[ / Ctrl+]
  Group           Ctrl+G               Ungroup       Ctrl+Shift+G
  Clipping mask   Ctrl+Alt+G (clip to the layer below)
  Opacity         Shift + digit        Blend mode    Shift + Plus/Minus
  On/Off          Ctrl+,               Blend space   F7 (sRGB ↔ linear)

File & History:
  Undo            Ctrl+Z               Redo          Ctrl+Shift+Z / Ctrl+Y
  Save            Ctrl+S (.mosaix, incremental, in the background)
  Export          Ctrl+Shift+E (PNG next to the file)
  Other           Ctrl+Shift+P command palette, F12 benchmark, F1 help, Ctrl+Q quit
";

fn parse_size(s: &str) -> anyhow::Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X', '×'])
        .context(tr("give the size as WIDTHxHEIGHT"))?;
    let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
    if w == 0 || h == 0 {
        bail!("{}", tr("size must be > 0"));
    }
    Ok((w, h))
}

fn main() -> anyhow::Result<()> {
    // Language first: everything after this may show text.
    mosaix_core::i18n::init(&settings::Settings::load_from(&settings::config_path()).language);
    init_tracing();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("mcp") => return agent_cmd(true, &argv[1..]),
        Some("run") => return agent_cmd(false, &argv[1..]),
        Some("batch") => return batch(&argv[1..]),
        _ => {}
    }

    let mut source = Source::Blank(4000, 3000);
    let mut explicit = false;
    let mut bench = false;
    let mut offscreen = false;
    let mut save: Option<PathBuf> = None;
    let mut export: Option<PathBuf> = None;
    let mut snapshot: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{}", tr(HELP));
                return Ok(());
            }
            "--new" => {
                let (w, h) = parse_size(&args.next().context(tr("--new needs a size"))?)?;
                source = Source::Blank(w, h);
                explicit = true;
            }
            "--demo" => {
                let (w, h) = parse_size(&args.next().context(tr("--demo needs a size"))?)?;
                source = Source::Demo(w, h);
                explicit = true;
            }
            "--bench" => bench = true,
            "--bench-offscreen" => offscreen = true,
            "--save" => save = Some(args.next().context(tr("--save needs a path"))?.into()),
            "--export" => export = Some(args.next().context(tr("--export needs a path"))?.into()),
            "--ui-snapshot" => {
                snapshot = Some(PathBuf::from(
                    args.next().context(tr("--ui-snapshot needs a path"))?,
                ))
            }
            s if s.starts_with('-') => bail!(
                "{}\n\n{}",
                tr!("unknown option {option}", option = s),
                tr(HELP)
            ),
            path => {
                source = Source::File(PathBuf::from(path));
                explicit = true;
            }
        }
    }
    if let Some(out) = snapshot {
        return ui::snapshot::run(source, &out, (1600, 1000));
    }
    if save.is_some() || export.is_some() || offscreen {
        return headless(source, save, export, offscreen);
    }
    // Without a file (or --new/--demo) the start screen shows.
    let home = !explicit && !bench;
    app::run(source, bench, home)
}

/// `mosaix batch ACTION FILE… [--out DIR] [--format EXT]`: plays an action
/// (name in the actions folder or a .jsonl path) on each file and exports it.
fn batch(args: &[String]) -> anyhow::Result<()> {
    use agent::Backend;
    use serde_json::json;
    let mut out_dir: Option<PathBuf> = None;
    let mut format: Option<String> = None;
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out_dir = Some(it.next().context(tr("--out needs a folder"))?.into()),
            "--format" => format = Some(it.next().context(tr("--format needs an extension"))?.trim_start_matches('.').to_lowercase()),
            "-h" | "--help" => {
                print!("{}", tr(HELP));
                return Ok(());
            }
            s if s.starts_with("--") => bail!("{}", tr!("unknown option {option}", option = s)),
            p => positional.push(p.to_string()),
        }
    }
    let (action, files) = positional.split_first().context(tr("mosaix batch needs an action and files"))?;
    anyhow::ensure!(!files.is_empty(), "{}", tr("mosaix batch needs an action and files"));
    let path = if action.ends_with(".jsonl") && std::path::Path::new(action).exists() {
        PathBuf::from(action)
    } else {
        app::actions::find(&app::actions::actions_dir(), action)
            .filter(|e| e.kind == app::actions::Kind::Macro)
            .with_context(|| tr!("no action named {name}", name = action))?
            .path
    };
    let steps = app::actions::read_macro(&path)?;
    let mut backend = agent::Local::new(None)?;
    let exportable = ["png", "jpg", "jpeg", "webp", "avif", "jxl", "tif", "tiff", "psd", "psb"];
    let (mut done, mut failed) = (0, 0);
    for f in files {
        let input = PathBuf::from(f);
        let ext = format.clone().unwrap_or_else(|| {
            input
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .filter(|e| exportable.contains(&e.as_str()))
                .unwrap_or_else(|| "png".into())
        });
        let dir = out_dir.clone().unwrap_or_else(|| input.parent().unwrap_or(std::path::Path::new(".")).join("mosaix-batch"));
        let target = dir.join(format!("{}.{ext}", input.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default()));
        let t0 = std::time::Instant::now();
        let r = (|| -> anyhow::Result<()> {
            std::fs::create_dir_all(&dir)?;
            backend.call(json!({ "cmd": "document_open", "path": input }))?;
            for (i, s) in steps.iter().enumerate() {
                backend.call(s.clone()).with_context(|| format!("step {} ({})", i + 1, s["cmd"].as_str().unwrap_or("?")))?;
            }
            backend.call(json!({ "cmd": "export", "path": target }))?;
            Ok(())
        })();
        match r {
            Ok(()) => {
                done += 1;
                println!("{} → {} ({:.0} ms)", input.display(), target.display(), t0.elapsed().as_secs_f64() * 1e3);
            }
            Err(e) => {
                failed += 1;
                eprintln!("{}: {e:#}", input.display());
            }
        }
    }
    println!("{}", tr!("{done} done, {failed} failed", done = done, failed = failed));
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// `mosaix mcp …` / `mosaix run …`
fn agent_cmd(mcp: bool, args: &[String]) -> anyhow::Result<()> {
    let mut attach = false;
    let mut socket: Option<PathBuf> = None;
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--attach" => attach = true,
            "--socket" => {
                attach = true;
                socket = Some(it.next().context(tr("--socket needs a path"))?.into());
            }
            "-h" | "--help" => {
                print!("{}", tr(HELP));
                return Ok(());
            }
            s if s.starts_with("--") => bail!("{}", tr!("unknown option {option}", option = s)),
            p => positional.push(PathBuf::from(p)),
        }
    }
    let mut backend: Box<dyn agent::Backend> = if attach {
        Box::new(agent::ipc::Client::connect(socket.as_deref())?)
    } else {
        let open = if mcp {
            positional.first().map(PathBuf::as_path)
        } else {
            None
        };
        Box::new(agent::Local::new(open)?)
    };
    if mcp {
        agent::mcp::serve(backend.as_mut())
    } else {
        let script = positional
            .last()
            .context(tr("mosaix run needs a script (or - for stdin)"))?;
        if !agent::run_script(backend.as_mut(), script)? {
            std::process::exit(1);
        }
        Ok(())
    }
}

/// Loads, optionally saves and exports, without opening a window.
fn headless(
    source: Source,
    save: Option<PathBuf>,
    export: Option<PathBuf>,
    bench: bool,
) -> anyhow::Result<()> {
    let gpu = pollster::block_on(mosaix_gpu::GpuContext::new(
        mosaix_gpu::GpuContext::create_instance(),
        None,
    ))?;
    let mut s = session::load(&gpu, wgpu::TextureFormat::Rgba8UnormSrgb, source)?;
    if let Some(p) = save {
        let r = s.save(Some(&p))?;
        tracing::info!(
            "{}",
            tr!(
                "saved: {path} ({tiles} tiles, {mb} MB, {ms} ms)",
                path = r.path.display(),
                tiles = r.tiles_written,
                mb = format!("{:.1}", r.bytes as f64 / 1e6),
                ms = format!("{:.0}", r.ms)
            )
        );
    }
    if let Some(p) = export {
        let ms = s.export(&p, &Default::default())?;
        tracing::info!(
            "{}",
            tr!(
                "exported: {path} ({ms} ms)",
                path = p.display(),
                ms = format!("{ms:.0}")
            )
        );
    }
    if bench {
        offscreen_bench(&gpu, &mut s.engine);
    }
    Ok(())
}

/// Renders the benchmark camera path into an offscreen target, waiting for the
/// GPU after every frame, so the numbers are real per-frame GPU+CPU cost.
fn offscreen_bench(gpu: &mosaix_gpu::GpuContext, engine: &mut mosaix_gpu::Engine) {
    use std::time::Instant;
    const FRAMES: u32 = 900;
    let (w, h) = (2560u32, 1440u32);
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let tv = target.create_view(&Default::default());
    engine.flush();
    let wait = || {
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll")
    };
    wait();
    let mut base = mosaix_core::View {
        viewport: glam::DVec2::new(w as f64, h as f64),
        ..Default::default()
    };
    let (dw, dh) = engine.size();
    base.fit(dw, dh);
    let overlay = mosaix_gpu::Overlay {
        checker_px: 8.0,
        ..Default::default()
    };
    let mut times = Vec::with_capacity(FRAMES as usize);
    for f in 0..FRAMES {
        let t0 = Instant::now();
        engine.render(&tv, &app::bench_view(&base, f, FRAMES), &overlay);
        wait();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let pct = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
    println!(
        "{}",
        tr!(
            "\n=== Offscreen benchmark ({frames} frames, {w}×{h}, {dw}×{dh} px document, {tiles} tiles) ===\n\
             Frame incl. GPU:  Ø {avg} ms   p50 {p50}   p99 {p99}   max {max}   (≈ {fps} fps possible)\n",
            frames = FRAMES,
            w = w,
            h = h,
            dw = dw,
            dh = dh,
            tiles = engine.composite_tiles_live(),
            avg = format!("{avg:.2}"),
            p50 = format!("{:.2}", pct(0.5)),
            p99 = format!("{:.2}", pct(0.99)),
            max = format!("{:.2}", times[times.len() - 1]),
            fps = format!("{:.0}", 1000.0 / avg)
        )
    );

    // Worst case for the compositor: a layer property change re-flattens every tile.
    engine.edit(tr("Duplicate"), |d| d.duplicate_active().is_some());
    engine.flush();
    wait();
    for (label, mode) in [
        (tr("Multiply"), mosaix_core::BlendMode::Multiply),
        (tr("Soft Light"), mosaix_core::BlendMode::SoftLight),
    ] {
        let t0 = Instant::now();
        engine.edit(tr("Blend Mode"), |d| {
            let id = d.active;
            d.find_mut(id).unwrap().blend = mode;
            true
        });
        let stats = engine.flush();
        wait();
        println!(
            "{}",
            tr!(
                "Blend mode → {label}: {tiles} tiles (2 layers) fully recomposited in {ms} ms",
                label = label,
                tiles = stats.flattened,
                ms = format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3)
            )
        );
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"));
    // Logs go to stderr: stdout belongs to MCP / script results.
    let registry = tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(filter),
        )
        .with(ui::log::UiLog);
    #[cfg(feature = "profile")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());
    registry.init();
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_size() {
        assert_eq!(super::parse_size("100x20").unwrap(), (100, 20));
        assert_eq!(super::parse_size("3×4").unwrap(), (3, 4));
        assert!(super::parse_size("0x5").is_err());
        assert!(super::parse_size("abc").is_err());
    }
}
