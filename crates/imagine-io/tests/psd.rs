//! PSD reading against files written by GIMP 3 (8 and 16 bit, RLE).
use imagine_core::BlendMode;
use imagine_io::psd::{self, Node, Raster};

fn data(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn check(name: &str) {
    let d = psd::open(&data(name)).unwrap();
    assert_eq!((d.width, d.height), (400, 300));
    assert!(d.icc.is_some());
    assert_eq!(d.layers.len(), 3, "background, red, group");
    let Node::Layer { info, rect, pixels } = &d.layers[1] else {
        panic!()
    };
    assert_eq!(info.name, "Red Ümlaut");
    assert_eq!(info.blend, BlendMode::Multiply);
    assert!((info.opacity - 0.5).abs() < 0.01);
    assert_eq!((rect.x, rect.y, rect.w, rect.h), (30, 40, 100, 80));
    match pixels.as_ref().unwrap() {
        Raster::U8(v) => assert_eq!(&v[..4], &[255, 0, 0, 255]),
        Raster::U16(v) => assert_eq!(&v[..4], &[65535, 0, 0, 65535]),
        Raster::F32(_) => panic!(),
    }
    let Node::Group { info, children, .. } = &d.layers[2] else {
        panic!("group")
    };
    assert_eq!(info.name, "My Group");
    assert_eq!(children.len(), 2);
    let Node::Layer { info: green, .. } = &children[0] else {
        panic!()
    };
    let mask = green.mask.as_ref().unwrap();
    // Left 60 px of the mask are black, the rest white.
    assert_eq!(mask.data[0], 0);
    assert_eq!(mask.data[59], 0);
    assert_eq!(mask.data[60], 255);
    let Node::Layer { info: hidden, .. } = &children[1] else {
        panic!()
    };
    assert!(!hidden.visible);
    assert_eq!(hidden.blend, BlendMode::Screen);
}

#[test]
fn reads_gimp_8_bit() {
    check("gimp8.psd");
}

#[test]
fn reads_gimp_16_bit() {
    check("gimp16.psd");
}

#[test]
fn rejects_other_files() {
    assert!(psd::read(b"\x89PNG....").is_err());
    assert!(psd::read(b"8BPS\x00\x01").is_err(), "truncated");
}

#[test]
fn reads_imagemagick_zip() {
    let d = psd::open(&data("magick-zip.psd")).unwrap();
    assert!(d.warnings.is_empty(), "{:?}", d.warnings);
    let Node::Layer { info, rect, pixels } = &d.layers[1] else {
        panic!()
    };
    assert_eq!(info.name, "Rot");
    assert_eq!((rect.x, rect.y, rect.w, rect.h), (20, 30, 60, 40));
    assert_eq!(
        pixels.as_ref().unwrap(),
        &Raster::U16([65535, 0, 0, 65535].repeat(60 * 40))
    );
}

mod roundtrip {
    use super::*;
    use half::f16;
    use imagine_core::{TILE_SIZE, TileCoord};
    use imagine_io::export::ExportOptions;
    use imagine_io::psd::{WLayer, WMask, WNode};

    /// One tile filled with an opaque colour in a rectangle (premultiplied linear).
    fn tile(rgb: [f32; 3], x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<f16> {
        let ts = TILE_SIZE as usize;
        let mut t = vec![f16::ZERO; ts * ts * 4];
        for y in y0..y1 {
            for x in x0..x1 {
                let i = (y * ts + x) * 4;
                t[i..i + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0].map(f16::from_f32));
            }
        }
        t
    }

    fn layer(name: &str, tiles: Vec<(TileCoord, Vec<f16>)>) -> WLayer {
        WLayer {
            name: name.into(),
            opacity: 1.0,
            blend: BlendMode::Normal,
            visible: true,
            clip: false,
            pass_through: false,
            tiles,
            mask: None,
        }
    }

    pub fn write(path: &std::path::Path, sixteen_bit: bool) {
        let bg = layer(
            "Background",
            vec![(
                TileCoord::new(0, 0),
                tile([0.0, 0.0, 1.0], 0, 0, 300.min(TILE_SIZE as usize), 200),
            )],
        );
        let mut red = layer(
            "Rot Ä",
            vec![(TileCoord::new(0, 0), tile([1.0, 0.0, 0.0], 10, 20, 60, 50))],
        );
        red.blend = BlendMode::Multiply;
        red.opacity = 0.5;
        red.clip = true;
        let mut green = layer(
            "Green",
            vec![(
                TileCoord::new(0, 0),
                tile([0.0, 1.0, 0.0], 100, 100, 150, 150),
            )],
        );
        let ts = TILE_SIZE as usize;
        let mut m = vec![255u8; ts * ts];
        for y in 0..ts {
            for x in 0..120 {
                m[y * ts + x] = 0;
            }
        }
        green.mask = Some(WMask {
            default: 255,
            enabled: true,
            tiles: vec![(TileCoord::new(0, 0), m)],
        });
        let mut group = layer("Gruppe", Vec::new());
        group.visible = false;
        group.pass_through = true;
        let nodes = vec![
            WNode::Layer(bg),
            WNode::Layer(red),
            WNode::Group {
                layer: group,
                children: vec![WNode::Layer(green)],
            },
        ];
        let composite = vec![(TileCoord::new(0, 0), tile([0.0, 0.0, 1.0], 0, 0, 200, 200))];
        let opts = ExportOptions {
            sixteen_bit,
            ..Default::default()
        };
        imagine_io::psd::write(path, (200, 200), &nodes, &composite, &opts).unwrap();
    }

    fn check(sixteen: bool) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("out.psd");
        write(&p, sixteen);
        let d = psd::open(&p).unwrap();
        assert_eq!(
            (d.width, d.height, d.depth),
            (200, 200, if sixteen { 16 } else { 8 })
        );
        assert_eq!(d.layers.len(), 3);
        let Node::Layer { info, rect, pixels } = &d.layers[1] else {
            panic!()
        };
        assert_eq!(info.name, "Rot Ä");
        assert!(info.clipped, "clipping mask survives");
        assert_eq!(info.blend, BlendMode::Multiply);
        assert!((info.opacity - 0.5).abs() < 0.01);
        assert_eq!((rect.x, rect.y, rect.w, rect.h), (10, 20, 50, 30));
        match pixels.as_ref().unwrap() {
            Raster::U8(v) => assert_eq!(&v[..4], &[255, 0, 0, 255]),
            Raster::U16(v) => assert_eq!(&v[..4], &[65535, 0, 0, 65535]),
            _ => panic!(),
        }
        let Node::Group { info, children, .. } = &d.layers[2] else {
            panic!()
        };
        assert_eq!(info.name, "Gruppe");
        assert_eq!(&info.blend_key, b"pass");
        assert!(!info.visible);
        let Node::Layer {
            info: g, rect: gr, ..
        } = &children[0]
        else {
            panic!()
        };
        assert_eq!((gr.x, gr.y, gr.w, gr.h), (100, 100, 50, 50));
        let m = g.mask.as_ref().unwrap();
        assert_eq!(m.default, 255);
        assert_eq!((m.data[0], m.data[119], m.data[120]), (0, 0, 255));
        let merged = d.merged.unwrap();
        match merged {
            Raster::U8(v) => assert_eq!(&v[..4], &[0, 0, 255, 255]),
            Raster::U16(v) => assert_eq!(&v[..4], &[0, 0, 65535, 65535]),
            _ => panic!(),
        }
    }

    #[test]
    fn write_and_read_8_bit() {
        check(false);
    }

    #[test]
    fn write_and_read_16_bit() {
        check(true);
    }

    /// Writes sample files for checking with other programs:
    /// `IMAGINE_PSD_OUT=dir cargo test -p imagine-io --test psd sample -- --ignored`.
    #[test]
    #[ignore]
    fn sample() {
        let dir = std::path::PathBuf::from(std::env::var("IMAGINE_PSD_OUT").unwrap());
        write(&dir.join("imagine8.psd"), false);
        write(&dir.join("imagine16.psd"), true);
    }
}
