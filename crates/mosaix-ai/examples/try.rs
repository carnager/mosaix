//! try IMAGE OUT_PREFIX [x,y …]: subject mask and SAM masks for clicks (x,y; -x,y = background).
use mosaix_ai::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let img = image::open(&a[0])?.to_rgb8();
    let im = Image { width: img.width() as usize, height: img.height() as usize, rgb: img.as_raw() };
    let save = |m: &Mask, name: &str| image::GrayImage::from_raw(m.width as u32, m.height as u32, m.data.clone()).unwrap().save(format!("{}{name}.png", a[1])).unwrap();
    let t = Instant::now();
    let mut s = Subject::load()?;
    let t1 = Instant::now();
    let m = s.mask(&im)?;
    println!("subject: load {:.0} ms, run {:.0} ms", (t1 - t).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
    save(&m, "subject");
    if a.len() > 2 {
        let t = Instant::now();
        let mut sam = Sam::load()?;
        let t1 = Instant::now();
        let e = sam.encode(&im)?;
        let t2 = Instant::now();
        let pts: Vec<Point> = a[2..].iter().map(|p| {
            let neg = p.starts_with('-');
            let (x, y) = p.trim_start_matches('-').split_once(',').unwrap();
            Point { x: x.parse().unwrap(), y: y.parse().unwrap(), positive: !neg }
        }).collect();
        let m = sam.segment(&e, &pts)?;
        println!("sam: load {:.0} ms, encode {:.0} ms, decode {:.0} ms", (t1 - t).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, t2.elapsed().as_secs_f64() * 1e3);
        save(&m, "sam");
    }
    Ok(())
}
