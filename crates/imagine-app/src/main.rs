//! imagine – GPU-first image editor for Linux/Wayland.

mod agent;
mod app;
mod boxedit;
mod loader;
mod session;
mod tablet;

use std::path::PathBuf;

use anyhow::{Context, bail};
use loader::Source;

const HELP: &str = "\
imagine – GPU-Bildeditor (Phase 2: Auswahl, Masken, Transformieren)

Aufruf:
  imagine [DATEI]               Bild öffnen (PNG, JPEG, WebP, TIFF, EXR, .imagine)
  imagine --new BxH             leeres Dokument, z. B. --new 30000x30000
  imagine --demo BxH            prozedurales Testbild, z. B. --demo 10000x10000
  imagine ... --bench           nach dem Laden Pan/Zoom/Rotate-Benchmark fahren und beenden
  imagine ... --bench-offscreen derselbe Benchmark ohne Fenster, misst echte GPU-Zeit pro Frame
  imagine DATEI --save A.imagine --export B.png
                                ohne Fenster konvertieren (beides optional kombinierbar)

Für Agenten (siehe docs/agenten.md):
  imagine mcp [DATEI]           MCP-Server über stdio mit eigener Engine
  imagine mcp --attach          MCP-Server, der das laufende Fenster live steuert
  imagine run SKRIPT.jsonl      JSON-Befehle zeilenweise ausführen (`-` = stdin)
  imagine run --attach SKRIPT   … im laufenden Fenster

Ansicht:
  Schwenken       Leertaste + Ziehen, mittlere Maustaste, Stifttaste 1
  Zoomen          Mausrad (zum Cursor), Strg + Leertaste + Ziehen
  Drehen          R halten + Ziehen, Umschalt + Mausrad, Stifttaste 2; Umschalt+R zurücksetzen
  Einpassen/100 % Strg+0 / Strg+1

Werkzeuge:
  B Pinsel (Umschalt+B: Spitze rund/Kreide/Borsten/Spray/rau/Kohle; J: Dynamik)   U Wischfinger
  E Radierer   M Auswahlrechteck (nochmal: Ellipse)   L Lasso
  G Verlauf (nochmal: radial)   V Verschieben (Pfeiltasten: 1 px, mit Umschalt 10 px)
  W Zauberstab (Klick; Umschalt/Alt kombinieren; [ / ] Toleranz; W nochmal: zusammenhängend ↔ überall)
  C Zuschneiden (Rahmen ziehen, über den Rand hinaus = Arbeitsfläche vergrößern; Enter / Esc)
  Mit Auswahl verschieben/transformieren V und Strg+T nur die ausgewählten Pixel.

Frei transformieren (Strg+T):
  Ecken skalieren (proportional, Umschalt = frei), Kanten strecken, Alt = um die Mitte,
  außerhalb ziehen = drehen (Umschalt = 15°-Schritte), innen = verschieben.
  Enter übernimmt, Esc oder Strg+Z bricht ab. Inhalt außerhalb des Bildes bleibt erhalten.

Malen:
  Malen           linke Maustaste / Stift (Druck = Größe & Fluss), Radierer-Ende des Stifts
  Größe           [ / ], Strg + Mausrad
  Deckkraft       Ziffern 1…9, 0 (10 … 100 %; beim Wischfinger: Stärke)
  Härte/Glättung  H / S       Farbe  P (Palette), X (Schwarz ↔ Weiß)
  Füllen          Alt+Rücktaste (Auswahl bzw. ganze Ebene)

Auswahl (beim Aufziehen: Umschalt = hinzufügen, Alt = abziehen, beides = Schnittmenge):
  Alles / nichts  Strg+A / Strg+D        Umkehren   Strg+Umschalt+I
  Weiche Kante    Umschalt+F6 (10 px)    Löschen    Entf oder Rücktaste (Pixel in der Auswahl)
  Maskierungsmodus-Ansicht  Q (rot = nicht ausgewählt)

Ebenenmasken:
  Neue Maske      Strg+Umschalt+M (aus der Auswahl, sonst alles sichtbar)
  Maske bemalen   \\ (umschalten Pixel ↔ Maske; Weiß zeigt, Schwarz verbirgt)
  Ein/Aus         Strg+Alt+M             Anwenden   Strg+Alt+Umschalt+M

Ebenen (Photoshop-Tastenkürzel):
  Neue Ebene      Strg+Umschalt+N      Duplizieren   Strg+J      Löschen  Entf (ohne Auswahl)
  Auswählen       Alt+[ / Alt+]        Verschieben   Strg+[ / Strg+]
  Gruppieren      Strg+G               Aufheben      Strg+Umschalt+G
  Deckkraft       Umschalt + Ziffer    Füllmethode   Umschalt + Plus/Minus
  Ein/Aus         Strg+,               Blend-Raum    F7 (sRGB ↔ linear)

Datei & Verlauf:
  Rückgängig      Strg+Z               Wiederholen   Strg+Umschalt+Z / Strg+Y
  Speichern       Strg+S (.imagine, inkrementell, im Hintergrund)
  Exportieren     Strg+Umschalt+E (PNG neben der Datei)
  Sonstiges       F12 Benchmark, F1 Hilfe, Strg+Q beenden
";

fn parse_size(s: &str) -> anyhow::Result<(u32, u32)> {
    let (w, h) = s.split_once(['x', 'X', '×']).context("Größe als BREITExHÖHE angeben")?;
    let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
    if w == 0 || h == 0 {
        bail!("Größe muss > 0 sein");
    }
    Ok((w, h))
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("mcp") => return agent_cmd(true, &argv[1..]),
        Some("run") => return agent_cmd(false, &argv[1..]),
        _ => {}
    }

    let mut source = Source::Blank(4000, 3000);
    let mut bench = false;
    let mut offscreen = false;
    let mut save: Option<PathBuf> = None;
    let mut export: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            "--new" => {
                let (w, h) = parse_size(&args.next().context("--new braucht eine Größe")?)?;
                source = Source::Blank(w, h);
            }
            "--demo" => {
                let (w, h) = parse_size(&args.next().context("--demo braucht eine Größe")?)?;
                source = Source::Demo(w, h);
            }
            "--bench" => bench = true,
            "--bench-offscreen" => offscreen = true,
            "--save" => save = Some(args.next().context("--save braucht einen Pfad")?.into()),
            "--export" => export = Some(args.next().context("--export braucht einen Pfad")?.into()),
            s if s.starts_with('-') => bail!("unbekannte Option {s}\n\n{HELP}"),
            path => source = Source::File(PathBuf::from(path)),
        }
    }
    if save.is_some() || export.is_some() || offscreen {
        return headless(source, save, export, offscreen);
    }
    app::run(source, bench)
}

/// `imagine mcp …` / `imagine run …`
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
                socket = Some(it.next().context("--socket braucht einen Pfad")?.into());
            }
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            s if s.starts_with("--") => bail!("unbekannte Option {s}"),
            p => positional.push(PathBuf::from(p)),
        }
    }
    let mut backend: Box<dyn agent::Backend> = if attach {
        Box::new(agent::ipc::Client::connect(socket.as_deref())?)
    } else {
        let open = if mcp { positional.first().map(PathBuf::as_path) } else { None };
        Box::new(agent::Local::new(open)?)
    };
    if mcp {
        agent::mcp::serve(backend.as_mut())
    } else {
        let script = positional.last().context("imagine run braucht ein Skript (oder - für stdin)")?;
        if !agent::run_script(backend.as_mut(), script)? {
            std::process::exit(1);
        }
        Ok(())
    }
}

/// Loads, optionally saves and exports, without opening a window.
fn headless(source: Source, save: Option<PathBuf>, export: Option<PathBuf>, bench: bool) -> anyhow::Result<()> {
    let gpu = pollster::block_on(imagine_gpu::GpuContext::new(imagine_gpu::GpuContext::create_instance(), None))?;
    let mut s = session::load(&gpu, wgpu::TextureFormat::Rgba8UnormSrgb, source)?;
    if let Some(p) = save {
        let r = s.save(Some(&p))?;
        tracing::info!("gespeichert: {} ({} Kacheln, {:.1} MB, {:.0} ms)", r.path.display(), r.tiles_written, r.bytes as f64 / 1e6, r.ms);
    }
    if let Some(p) = export {
        let ms = s.export(&p)?;
        tracing::info!("exportiert: {} ({ms:.0} ms)", p.display());
    }
    if bench {
        offscreen_bench(&gpu, &mut s.engine);
    }
    Ok(())
}

/// Renders the benchmark camera path into an offscreen target, waiting for the
/// GPU after every frame, so the numbers are real per-frame GPU+CPU cost.
fn offscreen_bench(gpu: &imagine_gpu::GpuContext, engine: &mut imagine_gpu::Engine) {
    use std::time::Instant;
    const FRAMES: u32 = 900;
    let (w, h) = (2560u32, 1440u32);
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let tv = target.create_view(&Default::default());
    engine.flush();
    let wait = || gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    wait();
    let mut base = imagine_core::View { viewport: glam::DVec2::new(w as f64, h as f64), ..Default::default() };
    let (dw, dh) = engine.size();
    base.fit(dw, dh);
    let overlay = imagine_gpu::Overlay { checker_px: 8.0, ..Default::default() };
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
        "\n=== Offscreen-Benchmark ({FRAMES} Frames, {w}×{h}, {dw}×{dh} px Dokument, {} Kacheln) ===\n\
         Frame inkl. GPU:  Ø {avg:.2} ms   p50 {:.2}   p99 {:.2}   max {:.2}   (≈ {:.0} fps möglich)\n",
        engine.composite_tiles_live(),
        pct(0.5),
        pct(0.99),
        times[times.len() - 1],
        1000.0 / avg
    );

    // Worst case for the compositor: a layer property change re-flattens every tile.
    engine.edit("Duplizieren", |d| d.duplicate_active().is_some());
    engine.flush();
    wait();
    for (label, mode) in [("Multiplizieren", imagine_core::BlendMode::Multiply), ("Weiches Licht", imagine_core::BlendMode::SoftLight)] {
        let t0 = Instant::now();
        engine.edit("Füllmethode", |d| {
            let id = d.active;
            d.find_mut(id).unwrap().blend = mode;
            true
        });
        let stats = engine.flush();
        wait();
        println!(
            "Füllmethode → {label}: {} Kacheln (2 Ebenen) komplett neu zusammengesetzt in {:.1} ms",
            stats.flattened,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"));
    // Logs go to stderr: stdout belongs to MCP / script results.
    let registry =
        tracing_subscriber::registry().with(fmt::layer().with_writer(std::io::stderr).with_filter(filter));
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
