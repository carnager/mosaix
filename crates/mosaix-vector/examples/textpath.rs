//! Renders a text on an arc into a PNG: `cargo run -p mosaix-vector --example textpath OUT.png`.
use mosaix_core::TILE_SIZE;
use mosaix_core::vector::{PathNode, TextPath, TextSpec, VectorLayer};

fn main() {
    let out = std::env::args().nth(1).expect("output png");
    let mut a = PathNode::corner([60.0, 330.0]);
    a.h_out = [60.0, 80.0];
    let mut b = PathNode::corner([540.0, 330.0]);
    b.h_in = [540.0, 80.0];
    let mut t = TextSpec::new("Text along a curved path ✓", 0.0, 0.0, 44.0, [0.05, 0.1, 0.4, 1.0]);
    t.font = "Liberation Serif".into();
    t.path = Some(TextPath { nodes: vec![a, b], closed: false, offset: 0.02, flip: false });
    let tiles = mosaix_vector::render(&VectorLayer::text(t));
    let (w, h) = (600u32, 400u32);
    let mut img = vec![255u8; (w * h * 4) as usize];
    let ts = TILE_SIZE as i64;
    for (c, px) in tiles {
        let (ox, oy) = c.origin();
        for i in 0..(ts * ts) as usize {
            let (x, y) = (ox + i as i64 % ts, oy + i as i64 / ts);
            if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                continue;
            }
            let a = px[i * 4 + 3].to_f32();
            let d = ((y as u32 * w + x as u32) * 4) as usize;
            for k in 0..3 {
                let src = mosaix_core::color::linear_to_srgb(px[i * 4 + k].to_f32() / a.max(1e-6));
                img[d + k] = ((src * a + (1.0 - a)) * 255.0).round() as u8;
            }
        }
    }
    image::RgbaImage::from_raw(w, h, img).unwrap().save(out).unwrap();
}
