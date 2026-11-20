//! Prints the structure of a PSD/PSB file: `cargo run -p imagine-io --example psdinfo FILE`.
use imagine_io::psd::{self, Node, Raster};

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
                    "{pad}{:?} {:?} {:.0}% vis={} rect={:?} mask={:?} {px}",
                    info.name,
                    info.blend,
                    info.opacity * 100.0,
                    info.visible,
                    rect,
                    info.mask
                        .as_ref()
                        .map(|m| (m.rect, m.default, m.enabled, m.data.len()))
                );
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
    for w in &d.warnings {
        println!("warning: {w}");
    }
    Ok(())
}
