//! inpaint IMAGE MASK OUT: fills the white area of MASK.
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let img = image::open(&a[0])?.to_rgb8();
    let mask = image::open(&a[1])?.resize_exact(img.width(), img.height(), image::imageops::FilterType::Nearest).to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut px: Vec<[f32; 3]> = img.pixels().map(|p| p.0.map(|v| v as f32 / 255.0)).collect();
    let hole: Vec<bool> = mask.pixels().map(|p| p[0] > 127).collect();
    let t = std::time::Instant::now();
    let mut m = imagine_ai::Inpaint::load()?;
    let t1 = std::time::Instant::now();
    m.fill(w, h, &mut px, &hole)?;
    println!("load {:.0} ms, fill {:.0} ms", (t1 - t).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
    let out = image::RgbImage::from_raw(w as u32, h as u32, px.iter().flat_map(|p| p.map(|v| (v * 255.0 + 0.5) as u8)).collect()).unwrap();
    out.save(&a[2])?;
    Ok(())
}
