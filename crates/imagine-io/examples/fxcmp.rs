//! Compares a layer's effects, computed by imagine from the pixels Photoshop
//! stored for the layer, with Photoshop's composite:
//! `fxcmp FILE.psd COMPOSITE.png LAYER_NAME [OUT.png]` (env SCALE_SIZE, SCALE_DIST, ANGLE_ADD).
use imagine_core::color::{linear_to_srgb, srgb_to_linear};
use imagine_core::style::{EffectKind, Region, apply_at};
use imagine_io::psd::{self, Node, Raster};

fn find<'a>(nodes: &'a [Node], name: &str) -> Option<&'a Node> {
    for n in nodes {
        match n {
            Node::Group { children, .. } => {
                if let Some(f) = find(children, name) {
                    return Some(f);
                }
            }
            Node::Layer { info, .. } if info.name == name => return Some(n),
            _ => {}
        }
    }
    None
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let d = psd::open(a[0].as_ref()).unwrap();
    let comp = image::open(&a[1]).unwrap().to_rgba8();
    let Some(Node::Layer { info, rect, pixels: Some(Raster::U8(px)) }) = find(&d.layers, &a[2]) else { panic!("no such 8-bit layer") };
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let mut effects = info.effects.clone();
    for e in &mut effects {
        if let EffectKind::Satin { size, distance, angle, .. } = &mut e.kind {
            *size *= env("SCALE_SIZE", 1.0);
            *distance *= env("SCALE_DIST", 1.0);
            *angle += env("ANGLE_ADD", 0.0);
        }
        if std::env::var("NO_FX").is_ok() {
            e.enabled = false;
        }
        println!("{} {:?}", e.enabled, e.kind);
    }
    let pad = 100usize;
    let (w, h) = (rect.w as usize + 2 * pad, rect.h as usize + 2 * pad);
    let mut reg = Region::new(w, h);
    for y in 0..rect.h as usize {
        for x in 0..rect.w as usize {
            let o = (y * rect.w as usize + x) * 4;
            let al = px[o + 3] as f32 / 255.0;
            let c = |k: usize| srgb_to_linear(px[o + k] as f32 / 255.0) * al;
            reg.px[(y + pad) * w + x + pad] = [c(0), c(1), c(2), al];
        }
    }
    let st = apply_at(&effects, &reg, info.fill_opacity, (rect.x - pad as i64, rect.y - pad as i64));
    let op = info.opacity;
    let (mut se, mut n) = (0.0f64, 0usize);
    let mut out = image::RgbaImage::new(rect.w, rect.h * 2);
    for y in 0..rect.h as usize {
        for x in 0..rect.w as usize {
            let (gx, gy) = (rect.x + x as i64, rect.y + y as i64);
            if gx < 0 || gy < 0 || gx >= comp.width() as i64 || gy >= comp.height() as i64 {
                continue;
            }
            let ps = comp.get_pixel(gx as u32, gy as u32).0;
            // Background: Photoshop's composite where the layer is empty (grey).
            let bg = [128.0 / 255.0; 3];
            let p = st.content.px[(y + pad) * w + x + pad];
            let al = p[3] * op;
            let im: [f32; 3] = std::array::from_fn(|k| {
                let c = if p[3] > 0.0 { linear_to_srgb(p[k] / p[3]) } else { 0.0 };
                c * al + bg[k] * (1.0 - al)
            });
            let v = |f: f32| (f.clamp(0.0, 1.0) * 255.0) as u8;
            out.put_pixel(x as u32, y as u32, image::Rgba([ps[0], ps[1], ps[2], 255]));
            out.put_pixel(x as u32, (y + rect.h as usize) as u32, image::Rgba([v(im[0]), v(im[1]), v(im[2]), 255]));
            if px[(y * rect.w as usize + x) * 4 + 3] == 255 {
                se += (0..3).map(|k| (ps[k] as f64 / 255.0 - im[k] as f64).powi(2)).sum::<f64>();
                n += 1;
            }
        }
    }
    let mid = ((rect.h / 2) * rect.w + 20) as usize * 4;
    println!("layer px at (20, mid): {:?}", &px[mid..mid + 4]);
    println!("{}: inside rmse {:.4} ({n} px), opacity {op} fill {}", info.name, (se / (3 * n.max(1)) as f64).sqrt(), info.fill_opacity);
    if let Some(o) = a.get(3) {
        out.save(o).unwrap();
    }
}
