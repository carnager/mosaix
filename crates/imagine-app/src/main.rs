//! imagine – Phase 0 spike: Wayland window, tablet input, sparse GPU tile
//! canvas, smooth pan/zoom/rotate, streaming image loading.

mod app;
mod loader;
mod tablet;

use std::path::PathBuf;

use anyhow::{Context, bail};
use loader::Source;

const HELP: &str = "\
imagine – Phase-0-Spike

Aufruf:
  imagine [BILD]            Bild öffnen (PNG, JPEG, WebP, TIFF, EXR)
  imagine --new BxH         leeres Dokument, z. B. --new 30000x30000
  imagine --demo BxH        prozedurales Testbild, z. B. --demo 10000x10000
  imagine ... --bench       nach dem Laden Pan/Zoom/Rotate-Benchmark fahren und beenden

Bedienung:
  Malen           linke Maustaste / Stift (Druck steuert Größe & Deckkraft)
  Radieren        E umschalten, oder Radierer-Ende des Stifts
  Schwenken       Leertaste + Ziehen, mittlere Maustaste, Stifttaste 1
  Zoomen          Mausrad (zum Cursor), Strg + Leertaste + Ziehen
  Drehen          R halten + Ziehen, Umschalt + Mausrad, Stifttaste 2; Umschalt+R = zurücksetzen
  Ansicht         0 = einpassen, 1 = 100 %
  Pinsel          [ / ] Größe, Strg + Mausrad Größe, H Härte, S Glättung, C Farbe, B Pinsel
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

    let mut source = Source::Blank(4000, 3000);
    let mut bench = false;
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
            s if s.starts_with('-') => bail!("unbekannte Option {s}\n\n{HELP}"),
            path => source = Source::File(PathBuf::from(path)),
        }
    }
    app::run(source, bench)
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,wgpu_core=warn,wgpu_hal=warn,naga=warn"));
    let registry = tracing_subscriber::registry().with(fmt::layer().with_filter(filter));
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
