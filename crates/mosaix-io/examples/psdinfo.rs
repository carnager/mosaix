//! Prints the structure of a PSD/PSB file: `cargo run -p mosaix-io --example psdinfo FILE`.
use mosaix_io::psd::{self, Node, Raster};

fn show(nodes: &[Node], depth: usize) {
    for n in nodes.iter().rev() {
        let pad = "  ".repeat(depth);
        match n {
            Node::Layer { info, rect, pixels } => {
                let px = match pixels {
                    Some(Raster::U8(v)) => format!("u8 first {:?}", &v[..4.min(v.len())]),
                    Some(Raster::U16(v)) => format!("u16 first {:?}", &v[..4.min(v.len())]),
                    Some(Raster::F32(v)) => format!("f32 first {:?}", &v[..4.min(v.len())]),
                    None => "no pixels".into(),
                };
                println!(
                    "{pad}{:?} {:?} {:.0}% vis={} clip={} rect={:?} mask={:?} {px}",
                    info.name,
                    info.blend,
                    info.opacity * 100.0,
                    info.visible,
                    info.clipped,
                    rect,
                    info.mask
                        .as_ref()
                        .map(|m| (m.rect, m.default, m.enabled, m.data.len()))
                );
                if let Some(a) = &info.adjustment {
                    println!("{pad}    adjustment: {a:?}");
                }
                if let Some(sh) = &info.shape {
                    println!("{pad}    shape: {sh:?}");
                }
                for e in &info.effects {
                    println!("{pad}    fx: {e:?}");
                }
                if let Some(t) = &info.text {
                    println!("{pad}    text: {t:?}");
                }
            }
            Node::Group {
                info,
                children,
                open,
            } => {
                println!(
                    "{pad}[{}] {:?} {:.0}% vis={} open={open}",
                    info.name,
                    info.blend,
                    info.opacity * 100.0,
                    info.visible
                );
                show(children, depth + 1);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).expect("file");
    let d = psd::open(p.as_ref())?;
    println!(
        "{}×{} depth {} icc {:?} merged {}",
        d.width,
        d.height,
        d.depth,
        d.icc.as_ref().map(Vec::len),
        d.merged.is_some()
    );
    show(&d.layers, 0);
    // Optional second argument: write the merged image (Photoshop's composite) as PNG.
    if let (Some(out), Some(Raster::U8(px))) = (std::env::args().nth(2), &d.merged) {
        image::RgbaImage::from_raw(d.width, d.height, px.clone())
            .expect("size")
            .save(out)?;
    }
    for w in &d.warnings {
        println!("warning: {w}");
    }
    Ok(())
}
