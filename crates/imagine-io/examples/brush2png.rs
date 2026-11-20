//! Writes the first tip of brush files (.abr, .gbr, .gih) as 256² grey PNGs
//! into a folder (imagine's brush library format):
//! `cargo run -p imagine-io --example brush2png OUTDIR FILE…`.
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().expect("output dir"));
    std::fs::create_dir_all(&out)?;
    for f in args {
        let p = std::path::Path::new(&f);
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(p)?;
        let tips = if f.ends_with(".abr") { imagine_io::abr::read(&bytes, &stem)? } else { imagine_io::gbr::read(&bytes, &stem)? };
        let t = &tips[0];
        let name = stem.replace('-', " ");
        let dst = out.join(format!("{name}.png"));
        image::GrayImage::from_raw(256, 256, imagine_io::abr::fit(t, 256)).unwrap().save(&dst)?;
        println!("{} ({}×{}) -> {}", f, t.width, t.height, dst.display());
    }
    Ok(())
}
