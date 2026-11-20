//! Photoshop patterns ("Patt" / "Pat2" / "Pat3" blocks): the pixels behind
//! pattern overlays, referenced by ID from the layer style.
//!
//! Each pattern: version 1, colour mode, size, Unicode name, Pascal ID,
//! then a "virtual memory array list": a rectangle and one entry per
//! channel (written flag, length, depth, rectangle, pixel depth,
//! compression, data), followed by the user mask (used for transparency)
//! and the sheet mask.

use anyhow::{Result, bail, ensure};
use imagine_core::pattern::Pattern;

struct C<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> C<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p + n <= self.b.len(), "pattern truncated");
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
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.bytes(4)?.try_into()?))
    }
    fn left(&self) -> usize {
        self.b.len() - self.p
    }
}

/// One channel's pixels (8 bit), `None` if not written.
fn channel(c: &mut C) -> Result<Option<(u32, u32, Vec<u8>)>> {
    if c.u32()? == 0 {
        return Ok(None);
    }
    let len = c.u32()? as usize;
    if len == 0 {
        return Ok(None);
    }
    ensure!(len >= 23, "pattern channel length");
    let body = c.bytes(len)?;
    let mut b = C { b: body, p: 0 };
    let depth = b.u32()?;
    let (top, left, bottom, right) = (b.u32()?, b.u32()?, b.u32()?, b.u32()?);
    let _pixel_depth = b.u16()?;
    let comp = b.u8()?;
    ensure!(depth == 8, "only 8-bit patterns are read");
    let (w, h) = (right.saturating_sub(left), bottom.saturating_sub(top));
    let (wu, hu) = (w as usize, h as usize);
    let data = &body[23..];
    let px = match comp {
        0 => {
            ensure!(data.len() >= wu * hu, "pattern data truncated");
            data[..wu * hu].to_vec()
        }
        1 => {
            // Row byte counts, then PackBits rows.
            let mut r = C { b: data, p: 0 };
            let counts: Vec<usize> = (0..hu).map(|_| r.u16().map(|v| v as usize)).collect::<Result<_>>()?;
            let mut out = Vec::with_capacity(wu * hu);
            for n in counts {
                let row = r.bytes(n.min(r.left()))?;
                crate::psd::packbits(row, &mut out, wu)?;
            }
            out
        }
        c => bail!("pattern compression {c}"),
    };
    Ok(Some((w, h, px)))
}

fn one(b: &[u8]) -> Result<(String, Pattern)> {
    let mut c = C { b, p: 0 };
    ensure!(c.u32()? == 1, "pattern version");
    let mode = c.u32()?;
    let _size = (c.u16()?, c.u16()?);
    let n = c.u32()? as usize;
    let units: Vec<u16> = (0..n).map(|_| c.u16()).collect::<Result<_>>()?;
    let name = String::from_utf16_lossy(&units).trim_end_matches('\0').to_string();
    // Photoshop's own patterns: "$$$/Patterns/Defaults/Water=Water".
    let name = match name.strip_prefix("$$$/").and_then(|n| n.split_once('=')) {
        Some((_, shown)) => shown.to_string(),
        None => name,
    };
    let idl = c.u8()? as usize;
    let id = String::from_utf8_lossy(c.bytes(idl)?).into_owned();
    if mode == 2 {
        // Indexed: colour table and 4 bytes.
        c.bytes(256 * 3 + 4)?;
    }
    ensure!(c.u32()? == 3, "pattern array version");
    let len = c.u32()? as usize;
    let mut v = C { b: c.bytes(len.min(c.left()))?, p: 0 };
    let _rect = (v.u32()?, v.u32()?, v.u32()?, v.u32()?);
    let count = v.u32()? as usize;
    let mut chans = Vec::new();
    for _ in 0..count + 2 {
        if v.left() < 4 {
            break;
        }
        chans.push(channel(&mut v)?);
    }
    let colour = match mode {
        1 | 2 => 1,
        3 => 3,
        m => bail!("pattern colour mode {m}"),
    };
    let (w, h, _) = chans.first().cloned().flatten().ok_or_else(|| anyhow::anyhow!("empty pattern"))?;
    let n = w as usize * h as usize;
    let get = |i: usize| chans.get(i).cloned().flatten().filter(|c| c.0 == w && c.1 == h).map(|c| c.2);
    let planes: Vec<Vec<u8>> = (0..colour).map(|i| get(i).unwrap_or_else(|| vec![0; n])).collect();
    // Transparency: the user mask (first of the two extra entries), else a
    // fourth written channel.
    let alpha = get(count).or_else(|| (colour == 3).then(|| get(3)).flatten());
    let mut px = Vec::with_capacity(n * 4);
    for i in 0..n {
        let (r, g, b) = if colour == 3 { (planes[0][i], planes[1][i], planes[2][i]) } else { (planes[0][i], planes[0][i], planes[0][i]) };
        px.extend([r, g, b, alpha.as_ref().map_or(255, |a| a[i])]);
    }
    let p = Pattern::new(&name, w, h, px).ok_or_else(|| anyhow::anyhow!("pattern size"))?;
    Ok((id, p))
}

/// Reads a pattern block: (ID, pattern) for each one that imagine can use.
pub fn read(b: &[u8]) -> Vec<(String, Pattern)> {
    let mut out = Vec::new();
    let mut c = C { b, p: 0 };
    while c.left() >= 4 {
        let Ok(len) = c.u32() else { break };
        let len = len as usize;
        let Ok(body) = c.bytes(len.min(c.left())) else { break };
        if let Ok(p) = one(body) {
            out.push(p);
        }
        let pad = (4 - len % 4) % 4;
        if c.bytes(pad.min(c.left())).is_err() {
            break;
        }
    }
    out
}

/// A stable ID for a pattern (Photoshop uses UUID-like strings).
pub fn pattern_id(p: &Pattern) -> String {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut h = DefaultHasher::new();
    (&p.name, p.width, p.height, &p.pixels).hash(&mut h);
    let a = h.finish();
    a.hash(&mut h);
    let b = h.finish();
    let x = format!("{a:016x}{b:016x}");
    format!("{}-{}-{}-{}-{}", &x[0..8], &x[8..12], &x[12..16], &x[16..20], &x[20..32])
}

/// Writes a pattern block for the patterns (RGB, raw, alpha in the user mask).
pub fn write(patterns: &[(String, &Pattern)]) -> Vec<u8> {
    let mut o = Vec::new();
    for (id, p) in patterns {
        let mut b = Vec::new();
        b.extend(1u32.to_be_bytes());
        b.extend(3u32.to_be_bytes()); // RGB
        b.extend((p.height as u16).to_be_bytes());
        b.extend((p.width as u16).to_be_bytes());
        let name: Vec<u16> = p.name.encode_utf16().chain(std::iter::once(0)).collect();
        b.extend((name.len() as u32).to_be_bytes());
        for u in name {
            b.extend(u.to_be_bytes());
        }
        b.push(id.len() as u8);
        b.extend(id.as_bytes());
        // Virtual memory array list.
        let (w, h) = (p.width, p.height);
        let mut v = Vec::new();
        for x in [0, 0, h, w] {
            v.extend(x.to_be_bytes());
        }
        const SLOTS: u32 = 24;
        v.extend(SLOTS.to_be_bytes());
        let plane = |k: usize| -> Vec<u8> { p.pixels.iter().skip(k).step_by(4).copied().collect() };
        let has_alpha = p.pixels.iter().skip(3).step_by(4).any(|&a| a < 255);
        let chan = |v: &mut Vec<u8>, data: Option<Vec<u8>>| match data {
            None => v.extend(0u32.to_be_bytes()),
            Some(d) => {
                v.extend(1u32.to_be_bytes());
                v.extend((23 + d.len() as u32).to_be_bytes());
                v.extend(8u32.to_be_bytes());
                for x in [0, 0, h, w] {
                    v.extend(x.to_be_bytes());
                }
                v.extend(8u16.to_be_bytes());
                v.push(0); // raw
                v.extend(d);
            }
        };
        for i in 0..SLOTS as usize {
            chan(&mut v, (i < 3).then(|| plane(i)));
        }
        chan(&mut v, has_alpha.then(|| plane(3))); // user mask
        chan(&mut v, None); // sheet mask
        b.extend(3u32.to_be_bytes());
        b.extend((v.len() as u32).to_be_bytes());
        b.extend(v);
        o.extend((b.len() as u32).to_be_bytes());
        let pad = (4 - b.len() % 4) % 4;
        o.extend(b);
        o.extend(std::iter::repeat_n(0, pad));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_roundtrip() {
        let mut lib = Pattern::builtin();
        // One with transparency.
        lib.push(Pattern::new("Halo", 2, 2, vec![255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 9, 9, 9, 255]).unwrap());
        let ids: Vec<(String, &Pattern)> = lib.iter().map(|p| (pattern_id(p), p)).collect();
        let back = read(&write(&ids));
        assert_eq!(back.len(), lib.len());
        for ((id, p), (bid, bp)) in ids.iter().zip(&back) {
            assert_eq!((id, *p), (bid, bp));
        }
    }
}
