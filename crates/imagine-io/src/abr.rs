//! Photoshop brush files (.abr): the sampled brush tips as grey images.
//!
//! Versions 6–10 (Photoshop 7 and later: `samp` section with the tips,
//! `desc` section with names) and the old versions 1/2. Values are coverage
//! (255 = full paint). Computed (round) brushes and brush dynamics are not
//! read – imagine has its own round tip and settings.

use anyhow::{Result, bail, ensure};
use imagine_core::tr;

/// One brush tip.
#[derive(Debug, Clone, PartialEq)]
pub struct AbrTip {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// `width × height` coverage values.
    pub data: Vec<u8>,
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> R<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "{}", tr("the file is truncated"));
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.bytes(2)?.try_into()?))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.bytes(4)?.try_into()?))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn left(&self) -> usize {
        self.b.len() - self.p
    }
}

/// Raw or RLE (PackBits, row byte counts first) image data.
fn image_data(r: &mut R, w: usize, h: usize, depth: u16, compressed: bool) -> Result<Vec<u8>> {
    let bpp = (depth as usize / 8).max(1);
    let row = w * bpp;
    let raw = if compressed {
        let mut counts = Vec::with_capacity(h);
        for _ in 0..h {
            counts.push(r.u16()? as usize);
        }
        let mut out = Vec::with_capacity(row * h);
        for n in counts {
            let src = r.bytes(n)?;
            let start = out.len();
            let mut i = 0;
            while out.len() - start < row && i < src.len() {
                let c = src[i] as i8;
                i += 1;
                if c >= 0 {
                    let k = (c as usize + 1).min(src.len() - i);
                    out.extend_from_slice(&src[i..i + k]);
                    i += k;
                } else if c != -128 && i < src.len() {
                    out.extend(std::iter::repeat_n(src[i], (1 - c as isize) as usize));
                    i += 1;
                }
            }
            out.resize(start + row, 0);
        }
        out
    } else {
        r.bytes(row * h)?.to_vec()
    };
    // 16-bit: keep the high byte.
    Ok(if bpp == 2 { raw.chunks(2).map(|c| c[0]).collect() } else { raw })
}

/// Names of the sampled brushes by their id (from the `desc` section).
fn names(desc: &[u8]) -> Vec<(String, String)> {
    let Some(d) = desc.get(4..).and_then(|b| crate::psd_data::read_desc(b).ok()) else { return Vec::new() };
    let mut out = Vec::new();
    if let Some(crate::psd_data::Value::List(list)) = d.get("Brsh") {
        for v in list {
            if let crate::psd_data::Value::Desc(b) = v {
                let name = b.text("Nm  ").unwrap_or("").to_string();
                if let Some(id) = b.desc("Brsh").and_then(|i| i.text("sampledData")) {
                    out.push((id.to_string(), name));
                }
            }
        }
    }
    out
}

fn read_v6(r: &mut R, subversion: u16) -> Result<Vec<(String, AbrTip)>> {
    let mut samp = None;
    let mut desc = None;
    while r.left() >= 12 {
        if r.bytes(4)? != b"8BIM" {
            break;
        }
        let key: [u8; 4] = r.bytes(4)?.try_into()?;
        let len = r.u32()? as usize;
        let data = r.bytes(len.min(r.left()))?;
        match &key {
            b"samp" => samp = Some(data),
            b"desc" => desc = Some(data),
            _ => {}
        }
    }
    let Some(samp) = samp else { bail!("{}", tr("no brush tips in the file")) };
    let mut s = R { b: samp, p: 0 };
    let mut tips = Vec::new();
    while s.left() >= 4 {
        let len = s.u32()? as usize;
        let end = (s.p + len + 3) & !3;
        let mut b = R { b: s.bytes(len.min(s.left()))?, p: 0 };
        s.p = end.min(samp.len());
        // Id: a Pascal string (usually a 36-character UUID).
        let n = b.u8()? as usize;
        let id = String::from_utf8_lossy(b.bytes(n)?).into_owned();
        // Header before the bounds: key and coordinates (47 bytes in
        // sub-version 1), key and unknown data (301 bytes in sub-version 2),
        // as GIMP and Krita read it.
        b.p = if subversion == 1 { 47 } else { 301 };
        let (top, left, bottom, right) = (b.i32()?, b.i32()?, b.i32()?, b.i32()?);
        let depth = b.u16()?;
        let compressed = b.u8()? != 0;
        let (w, h) = ((right - left).max(0) as usize, (bottom - top).max(0) as usize);
        if w == 0 || h == 0 || w > 10000 || h > 10000 {
            continue;
        }
        let data = image_data(&mut b, w, h, depth, compressed)?;
        tips.push((id, AbrTip { name: String::new(), width: w as u32, height: h as u32, data }));
    }
    let names = desc.map(names).unwrap_or_default();
    Ok(tips
        .into_iter()
        .map(|(id, mut t)| {
            t.name = names.iter().find(|(i, _)| *i == id).map(|(_, n)| n.clone()).unwrap_or_default();
            (id, t)
        })
        .collect())
}

fn read_v12(r: &mut R, version: u16) -> Result<Vec<(String, AbrTip)>> {
    let count = r.u16()?;
    let mut tips = Vec::new();
    for _ in 0..count {
        let kind = r.u16()?;
        let len = r.u32()? as usize;
        let mut b = R { b: r.bytes(len.min(r.left()))?, p: 0 };
        if kind != 2 {
            continue; // computed brush
        }
        b.u32()?; // misc
        b.u16()?; // spacing
        let mut name = String::new();
        if version == 2 {
            let n = b.u32()? as usize;
            let units: Vec<u16> = (0..n).map(|_| b.u16()).collect::<Result<_>>()?;
            name = String::from_utf16_lossy(&units).trim_end_matches('\0').to_string();
        }
        b.u8()?; // antialiasing
        for _ in 0..4 {
            b.i16()?; // short bounds
        }
        let (top, left, bottom, right) = (b.i32()?, b.i32()?, b.i32()?, b.i32()?);
        let depth = b.u16()?;
        let compressed = b.u8()? != 0;
        let (w, h) = ((right - left).max(0) as usize, (bottom - top).max(0) as usize);
        if w == 0 || h == 0 || w > 10000 || h > 10000 {
            continue;
        }
        let data = image_data(&mut b, w, h, depth, compressed)?;
        tips.push((String::new(), AbrTip { name, width: w as u32, height: h as u32, data }));
    }
    Ok(tips)
}

/// Reads the sampled tips of an ABR file. Unnamed tips get "<stem> N".
pub fn read(bytes: &[u8], stem: &str) -> Result<Vec<AbrTip>> {
    let mut r = R { b: bytes, p: 0 };
    let version = r.u16()?;
    let tips = match version {
        1 | 2 => read_v12(&mut r, version)?,
        6..=10 => {
            let sub = r.u16()?;
            read_v6(&mut r, sub)?
        }
        v => bail!("{}", imagine_core::tr!("unknown ABR version {v}", v = v)),
    };
    ensure!(!tips.is_empty(), "{}", tr("no brush tips in the file"));
    Ok(tips
        .into_iter()
        .enumerate()
        .map(|(i, (_, mut t))| {
            if t.name.trim().is_empty() {
                t.name = format!("{stem} {}", i + 1);
            }
            t
        })
        .collect())
}

/// A tip scaled into a `size × size` square (aspect kept, centred).
pub fn fit(t: &AbrTip, size: u32) -> Vec<u8> {
    let s = size as f32 / t.width.max(t.height) as f32;
    let (w, h) = ((t.width as f32 * s).max(1.0), (t.height as f32 * s).max(1.0));
    let (ox, oy) = ((size as f32 - w) * 0.5, (size as f32 - h) * 0.5);
    let mut out = vec![0u8; (size * size) as usize];
    // Box filter over the source pixels each target pixel covers.
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = ((x as f32 - ox) / s, (y as f32 - oy) / s);
            let (fx1, fy1) = (fx + 1.0 / s, fy + 1.0 / s);
            if fx1 <= 0.0 || fy1 <= 0.0 || fx >= t.width as f32 || fy >= t.height as f32 {
                continue;
            }
            let (x0, y0) = (fx.max(0.0) as u32, fy.max(0.0) as u32);
            let (x1, y1) = ((fx1.ceil() as u32).min(t.width).max(x0 + 1), (fy1.ceil() as u32).min(t.height).max(y0 + 1));
            let mut sum = 0u32;
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    sum += t.data[(sy * t.width + sx) as usize] as u32;
                    n += 1;
                }
            }
            out[(y * size + x) as usize] = (sum / n.max(1)) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A v6.2 file with one raw 4×2 tip.
    fn v6() -> Vec<u8> {
        let mut sample = vec![36u8];
        sample.extend(b"12345678-1234-1234-1234-123456789012");
        sample.resize(301, 0);
        for v in [0i32, 0, 2, 4] {
            sample.extend(v.to_be_bytes());
        }
        sample.extend(8u16.to_be_bytes());
        sample.push(0);
        sample.extend([0, 64, 128, 255, 255, 128, 64, 0]);
        let mut samp = (sample.len() as u32).to_be_bytes().to_vec();
        samp.extend(&sample);
        while !samp.len().is_multiple_of(4) {
            samp.push(0);
        }
        let mut f = vec![0, 6, 0, 2];
        f.extend(b"8BIMsamp");
        f.extend((samp.len() as u32).to_be_bytes());
        f.extend(samp);
        f
    }

    #[test]
    fn reads_v6_sampled_tips() {
        let tips = read(&v6(), "Set").unwrap();
        assert_eq!(tips.len(), 1);
        assert_eq!((tips[0].width, tips[0].height), (4, 2));
        assert_eq!(tips[0].data, [0, 64, 128, 255, 255, 128, 64, 0]);
        assert_eq!(tips[0].name, "Set 1", "unnamed tips are numbered");
    }

    #[test]
    fn reads_v2_with_rle_and_names() {
        let mut b = Vec::new();
        b.extend(0u32.to_be_bytes()); // misc
        b.extend(25u16.to_be_bytes()); // spacing
        let name: Vec<u16> = "Dot\0".encode_utf16().collect();
        b.extend((name.len() as u32).to_be_bytes());
        for u in name {
            b.extend(u.to_be_bytes());
        }
        b.push(1); // antialias
        b.extend([0u8; 8]);
        for v in [0i32, 0, 1, 3] {
            b.extend(v.to_be_bytes());
        }
        b.extend(8u16.to_be_bytes());
        b.push(1); // RLE
        let row = [(-2i8) as u8, 200]; // 3 × 200
        b.extend((row.len() as u16).to_be_bytes());
        b.extend(row);
        let mut f = vec![0, 2, 0, 1];
        f.extend(2u16.to_be_bytes());
        f.extend((b.len() as u32).to_be_bytes());
        f.extend(b);
        let tips = read(&f, "x").unwrap();
        assert_eq!(tips[0].name, "Dot");
        assert_eq!(tips[0].data, [200, 200, 200]);
    }

    #[test]
    fn fit_keeps_aspect_and_centres() {
        let t = AbrTip { name: String::new(), width: 4, height: 2, data: vec![255; 8] };
        let px = fit(&t, 8);
        // 8 × 4 band in the middle rows 2…5.
        assert_eq!(px[0], 0);
        assert_eq!(px[3 * 8 + 4], 255);
        assert_eq!(px[7 * 8 + 4], 0);
    }

    #[test]
    fn rejects_garbage() {
        assert!(read(&[0, 9, 0, 0], "x").is_err());
        assert!(read(&[0, 6, 0, 2], "x").is_err(), "no samp section");
    }
}
