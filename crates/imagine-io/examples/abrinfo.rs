//! Lists the tips of an ABR file and writes them as PNGs:
//! `cargo run -p imagine-io --example abrinfo FILE.abr [OUTDIR]`.
fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).expect("file");
    let out = std::env::args().nth(2);
    let tips = imagine_io::abr::read(&std::fs::read(&p)?, "tip")?;
    for (i, t) in tips.iter().enumerate() {
        let mean = t.data.iter().map(|&v| v as u64).sum::<u64>() / t.data.len().max(1) as u64;
        println!("{i}: {:?} {}×{} mean {mean}", t.name, t.width, t.height);
        if let Some(dir) = &out {
            let px = imagine_io::abr::fit(t, 256);
            image::GrayImage::from_raw(256, 256, px).unwrap().save(format!("{dir}/{i}.png"))?;
        }
    }
    Ok(())
}
