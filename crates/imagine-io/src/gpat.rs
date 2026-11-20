//! GIMP patterns (.pat): header (size, version 1, width, height, bytes per
//! pixel, "GPAT", UTF-8 name), then the pixels (grey, grey+alpha, RGB or
//! RGBA, 8 bit).

use anyhow::{Result, bail, ensure};

/// Name and straight RGBA pixels of a .pat file.
pub fn read(b: &[u8]) -> Result<(String, u32, u32, Vec<u8>)> {
    let u = |o: usize| -> Result<u32> {
        Ok(u32::from_be_bytes(b.get(o..o + 4).ok_or_else(|| anyhow::anyhow!("pattern truncated"))?.try_into()?))
    };
    let (hs, version, w, h, bpp) = (u(0)? as usize, u(4)?, u(8)?, u(12)?, u(16)? as usize);
    ensure!(version == 1 && b.get(20..24) == Some(b"GPAT"), "not a GIMP pattern");
    ensure!(w > 0 && h > 0 && w <= 8192 && h <= 8192 && (1..=4).contains(&bpp), "pattern size");
    let name = String::from_utf8_lossy(b.get(24..hs).unwrap_or(&[])).trim_end_matches('\0').to_string();
    let n = w as usize * h as usize;
    let Some(src) = b.get(hs..hs + n * bpp) else { bail!("pattern truncated") };
    let px = src
        .chunks_exact(bpp)
        .flat_map(|p| match bpp {
            1 => [p[0], p[0], p[0], 255],
            2 => [p[0], p[0], p[0], p[1]],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        })
        .collect();
    Ok((name, w, h, px))
}

#[cfg(test)]
mod tests {
    #[test]
    fn reads_rgb_pattern() {
        let name = b"Test\0";
        let hs = 24 + name.len();
        let mut b = Vec::new();
        for v in [hs as u32, 1, 2, 1, 3] {
            b.extend(v.to_be_bytes());
        }
        b.extend(b"GPAT");
        b.extend(name);
        b.extend([255, 0, 0, 0, 0, 255]);
        let (n, w, h, px) = super::read(&b).unwrap();
        assert_eq!((n.as_str(), w, h), ("Test", 2, 1));
        assert_eq!(px, vec![255, 0, 0, 255, 0, 0, 255, 255]);
    }
}
