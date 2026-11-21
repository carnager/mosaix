//! GIMP brushes: `.gbr` (one tip, grey or RGBA) and `.gih` (image hose:
//! several tips; each cell becomes a tip). 255 = full paint; colour
//! brushes use their alpha.

use anyhow::{Result, ensure};
use mosaix_core::tr;

use crate::abr::AbrTip;

/// One GBR record at the start of `b`; returns it and its length.
fn record(b: &[u8]) -> Result<(AbrTip, usize)> {
    ensure!(b.len() >= 28, "{}", tr("the file is truncated"));
    let u = |i: usize| u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
    let (hs, version, w, h, bytes) = (u(0), u(4), u(8), u(12), u(16));
    ensure!(version == 2 || version == 1, "{}", tr("not a GIMP brush"));
    ensure!(matches!(bytes, 1 | 4) && w > 0 && h > 0 && w <= 10000 && h <= 10000, "{}", tr("not a GIMP brush"));
    ensure!(hs <= b.len() && hs >= 20 + if version == 2 { 8 } else { 0 }, "{}", tr("the file is truncated"));
    let name_start = if version == 2 { 28 } else { 20 };
    let name = String::from_utf8_lossy(&b[name_start..hs]).trim_end_matches('\0').to_string();
    let n = w * h * bytes;
    ensure!(hs + n <= b.len(), "{}", tr("the file is truncated"));
    let px = &b[hs..hs + n];
    let data = if bytes == 1 { px.to_vec() } else { px.chunks(4).map(|p| p[3]).collect() };
    Ok((AbrTip { name, width: w as u32, height: h as u32, data }, hs + n))
}

/// Tips of a .gbr or .gih file; unnamed ones get "<stem> N".
pub fn read(bytes: &[u8], stem: &str) -> Result<Vec<AbrTip>> {
    let mut tips = Vec::new();
    let mut rest = bytes;
    let mut hose_name = None;
    // A .gih starts with a text line (the name) and a parameter line.
    if !rest.get(20..24).is_some_and(|m| m == b"GIMP") {
        let line = |b: &[u8]| b.iter().position(|&c| c == b'\n');
        let n1 = line(rest).ok_or_else(|| anyhow::anyhow!(tr("not a GIMP brush")))?;
        hose_name = Some(String::from_utf8_lossy(&rest[..n1]).trim().to_string());
        let n2 = n1 + 1 + line(&rest[n1 + 1..]).ok_or_else(|| anyhow::anyhow!(tr("not a GIMP brush")))?;
        rest = &rest[n2 + 1..];
    }
    while rest.len() >= 28 {
        let (t, len) = record(rest)?;
        tips.push(t);
        rest = &rest[len..];
    }
    ensure!(!tips.is_empty(), "{}", tr("no brush tips in the file"));
    let base = hose_name.filter(|n| !n.is_empty());
    let many = tips.len() > 1;
    Ok(tips
        .into_iter()
        .enumerate()
        .map(|(i, mut t)| {
            // Hose cells: "<hose name> 1", "… 2"; single brushes keep their name.
            t.name = match (&base, many) {
                (Some(b), true) => format!("{b} {}", i + 1),
                (Some(b), false) => b.clone(),
                (None, _) if t.name.trim().is_empty() => format!("{stem} {}", i + 1),
                (None, _) => t.name,
            };
            t
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gbr(name: &str, w: u32, h: u32, bytes: u32, px: &[u8]) -> Vec<u8> {
        let mut n = name.as_bytes().to_vec();
        n.push(0);
        let hs = 28 + n.len() as u32;
        let mut b = Vec::new();
        for v in [hs, 2, w, h, bytes] {
            b.extend(v.to_be_bytes());
        }
        b.extend(b"GIMP");
        b.extend(25u32.to_be_bytes());
        b.extend(n);
        b.extend(px);
        b
    }

    #[test]
    fn reads_grey_and_rgba_brushes_and_hoses() {
        let t = read(&gbr("Dot", 2, 1, 1, &[0, 255]), "x").unwrap();
        assert_eq!((t[0].name.as_str(), t[0].data.as_slice()), ("Dot", &[0u8, 255][..]));
        let t = read(&gbr("Col", 1, 1, 4, &[10, 20, 30, 128]), "x").unwrap();
        assert_eq!(t[0].data, [128], "colour brush: alpha");
        let mut hose = b"Chalk\n2 ncells:2 cellwidth:1 cellheight:1\n".to_vec();
        hose.extend(gbr("a", 1, 1, 1, &[255]));
        hose.extend(gbr("b", 1, 1, 1, &[100]));
        let t = read(&hose, "x").unwrap();
        assert_eq!(t.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["Chalk 1", "Chalk 2"]);
        assert!(read(b"nonsense", "x").is_err());
    }
}
