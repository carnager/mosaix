//! Develops RAW files to 16-bit sRGB PNGs (for comparing with other converters).
//! Usage: rawdev OUT_DIR FILE… (env WB=as_shot|auto|daylight|<kelvin>, EV=stops, MAX=px,
//! HL=clip|reconstruct, NR=luminance,color, LENS=0, CROP=x,y,w,h of the output)
use mosaix_raw::{Develop, Highlights, NoiseReduction, RawFile, WhiteBalance};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().expect("OUT_DIR"));
    let wb = match std::env::var("WB").as_deref() {
        Ok("auto") => WhiteBalance::Auto,
        Ok("daylight") => WhiteBalance::Daylight,
        Ok(k) if k.parse::<f32>().is_ok() => WhiteBalance::Custom { temperature: k.parse()?, tint: 0.0 },
        _ => WhiteBalance::AsShot,
    };
    let exposure = std::env::var("EV").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let highlights = if std::env::var("HL").as_deref() == Ok("clip") { Highlights::Clip } else { Highlights::Reconstruct };
    let noise = std::env::var("NR")
        .ok()
        .and_then(|v| {
            let (l, c) = v.split_once(',')?;
            Some(NoiseReduction { luminance: l.parse().ok()?, color: c.parse().ok()? })
        })
        .unwrap_or_default();
    let d = Develop { white_balance: wb, exposure, highlights, noise, lens: std::env::var("LENS").as_deref() != Ok("0"), tone: Default::default() };
    let mut d = d;
    // TONE=contrast,highlights,shadows,vibrance,saturation
    if let Ok(t) = std::env::var("TONE") {
        let v: Vec<f32> = t.split(',').filter_map(|x| x.parse().ok()).collect();
        if v.len() == 5 {
            d.tone = mosaix_raw::Tone { contrast: v[0], highlights: v[1], shadows: v[2], vibrance: v[3], saturation: v[4] };
        }
    }
    let crop: Option<Vec<u32>> = std::env::var("CROP").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
    let max: usize = std::env::var("MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for f in args {
        let p = std::path::Path::new(&f);
        let t0 = Instant::now();
        let raw = RawFile::open(p)?;
        let t1 = Instant::now();
        let mut cam = raw.camera_image()?;
        let t2 = Instant::now();
        raw.denoise(&mut cam, &noise);
        let t2b = Instant::now();
        if max > 0 {
            cam = cam.downscale(max);
        }
        let (w, h, px) = raw.develop(&cam, &d);
        let t3 = Instant::now();
        let mean: Vec<f64> = (0..3).map(|c| px.chunks(4).map(|p| p[c] as f64).sum::<f64>() / (w * h) as f64).collect();
        println!(
            "{f}: {} | {} | lens {:?} | as shot {:?} | {w}×{h} | decode {:.0} ms, demosaic {:.0} ms, denoise {:.0} ms, develop {:.0} ms | mean {:.4} {:.4} {:.4}",
            raw.info.camera,
            raw.info.shot(),
            raw.info.lens_profile,
            raw.info.as_shot.map(|(t, ti)| (t.round(), ti.round())),
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            (t2b - t2).as_secs_f64() * 1e3,
            (t3 - t2b).as_secs_f64() * 1e3,
            mean[0], mean[1], mean[2]
        );
        let enc = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            let s = if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
            (s * 65535.0 + 0.5) as u16
        };
        let rgb: Vec<u16> = px.chunks(4).flat_map(|p| [enc(p[0]), enc(p[1]), enc(p[2])]).collect();
        let mut img = image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(w, h, rgb).unwrap();
        if let Some(c) = &crop
            && c.len() == 4
        {
            img = image::imageops::crop_imm(&img, c[0], c[1], c[2], c[3]).to_image();
        }
        img.save(out.join(format!("{}.png", p.file_stem().unwrap().to_string_lossy())))?;
    }
    Ok(())
}
