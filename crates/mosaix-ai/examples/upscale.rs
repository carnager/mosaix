//! upscale MODEL.onnx IMAGE OUT
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let img = image::open(&a[1])?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let px: Vec<[f32; 3]> = img.pixels().map(|p| p.0.map(|v| v as f32 / 255.0)).collect();
    let mut m = mosaix_ai::Upscale::load_file(std::path::Path::new(&a[0]))?;
    let t = std::time::Instant::now();
    let out = m.run(w, h, &px, None, &mut |_, _| {})?;
    let k = m.scale();
    println!("{w}×{h} ×{k}: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    image::RgbImage::from_raw((w * k) as u32, (h * k) as u32, out.iter().flat_map(|p| p.map(|v| (v * 255.0 + 0.5) as u8)).collect())
        .unwrap()
        .save(&a[2])?;
    Ok(())
}
