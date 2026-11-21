//! Dumps descriptors found after a 4-byte key (+4 version) anywhere in a file.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let b = std::fs::read(&a[1]).unwrap();
    let key = a[2].as_bytes();
    let mut i = 0;
    while let Some(p) = b[i..].windows(4).position(|w| w == key) {
        let s = i + p + 4;
        // Additional layer info: key, u32 length, then version u32 + descriptor.
        for skip in [8usize, 4, 12] {
            if let Some(d) = b.get(s + skip..).and_then(|x| mosaix_io::psd_data::read_desc(x).ok()) {
                println!("@{s} +{skip}: {d:#?}");
                break;
            }
        }
        i = s;
    }
}
