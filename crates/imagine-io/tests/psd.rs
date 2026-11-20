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
