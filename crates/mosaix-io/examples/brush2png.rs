//! Writes the first tip of brush files (.abr, .gbr, .gih) as 256² grey PNGs
//! into a folder (mosaix's brush library format):
//! `cargo run -p mosaix-io --example brush2png OUTDIR FILE…`.
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().expect("output dir"));
    std::fs::create_dir_all(&out)?;
    for f in args {
        let p = std::path::Path::new(&f);
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(p)?;
        let tips = if f.ends_with(".abr") { mosaix_io::abr::read(&bytes, &stem)? } else { mosaix_io::gbr::read(&bytes, &stem)? };
        let t = &tips[0];
        let name = stem.replace('-', " ");
        let dst = out.join(format!("{name}.png"));
        image::GrayImage::from_raw(256, 256, mosaix_io::abr::fit(t, 256)).unwrap().save(&dst)?;
        println!("{} ({}×{}) -> {}", f, t.width, t.height, dst.display());
    }
    Ok(())
}
