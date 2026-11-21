//! Compares imagine's rendering of a PSD's shape / fill layers with the
//! pixels Photoshop stored: `cargo run -p imagine-vector --example fillcmp FILE.psd`.
use imagine_core::color::linear_to_srgb;
use imagine_core::vector::VectorLayer;
use imagine_core::{TILE_SIZE, TileCoord};
use imagine_io::psd::{self, Node, Raster};

fn walk(nodes: &[Node], path: &str) {
    for n in nodes.iter().rev() {
        match n {
            Node::Group { info, children, .. } => walk(children, &format!("{path}{}/", info.name)),
            Node::Layer { info, rect, pixels: Some(Raster::U8(px)) } => {
                let Some(shape) = &info.shape else { continue };
                let tiles = imagine_vector::render(&VectorLayer::shape(shape.clone()));
                let ts = TILE_SIZE as i64;
                let get = |x: i64, y: i64| -> [f32; 4] {
                    let c = TileCoord::new(x.div_euclid(ts) as i32, y.div_euclid(ts) as i32);
                    tiles.iter().find(|(t, _)| *t == c).map_or([0.0; 4], |(_, p)| {
                        let o = ((y.rem_euclid(ts) * ts + x.rem_euclid(ts)) * 4) as usize;
                        let a = p[o + 3].to_f32();
                        let un = |v: f32| if a > 0.0 { linear_to_srgb(v / a) } else { 0.0 };
                        [un(p[o].to_f32()), un(p[o + 1].to_f32()), un(p[o + 2].to_f32()), a]
                    })
                };
                let (mut se, mut n) = (0.0f64, 0usize);
                let mut worst = (0.0f32, 0, 0, [0.0; 4], [0.0; 4]);
                for y in 0..rect.h as i64 {
                    for x in 0..rect.w as i64 {
                        let o = ((y * rect.w as i64 + x) * 4) as usize;
                        let ps: [f32; 4] = std::array::from_fn(|k| px[o + k] as f32 / 255.0);
                        if ps[3] < 1.0 {
                            continue; // compare inside only
                        }
                        let im = get(rect.x + x, rect.y + y);
                        let d = (0..3).map(|k| (ps[k] - im[k]).powi(2)).sum::<f32>();
                        se += d as f64;
                        n += 1;
                        if d > worst.0 {
                            worst = (d, x, y, ps, im);
                        }
                    }
                }
                if std::env::var("GRID").is_ok_and(|g| info.name.contains(&g) && path.contains(&std::env::var("GROUP").unwrap_or_default())) {
                    println!("{:?}", shape.geometry);
                    if let Ok(f) = std::env::var("CSV") {
                        let mut out = String::new();
                        for y in 0..rect.h as i64 {
                            let row: Vec<String> = (0..rect.w as i64).map(|x| px[((y * rect.w as i64 + x) * 4 + 1) as usize].to_string()).collect();
                            out += &(row.join(",") + "\n");
                        }
                        std::fs::write(f, out).unwrap();
                    }
                    for y in (0..rect.h as i64).step_by(2) {
                        let row = |f: &dyn Fn(i64) -> f32| (0..rect.w as i64).step_by(2).map(|x| format!("{:>2}", (f(x) * 99.0).round() as i32)).collect::<Vec<_>>().join("");
                        println!("{}   {}", row(&|x| px[((y * rect.w as i64 + x) * 4 + 1) as usize] as f32 / 255.0), row(&|x| get(rect.x + x, rect.y + y)[1]));
                    }
                }
                let rmse = (se / (n.max(1) * 3) as f64).sqrt();
                println!("{path}{:<32} rmse {rmse:.4}  worst at ({},{}) ps {:.2?} im {:.2?}", info.name, worst.1, worst.2, &worst.3[..3], &worst.4[..3]);
            }
            _ => {}
        }
    }
}

fn main() {
    let p = std::env::args().nth(1).expect("file");
    let d = psd::open(p.as_ref()).unwrap();
    walk(&d.layers, "");
}
