//! Photoshop documents (PSD and the large-document variant PSB).
//!
//! Reading: layers with names, opacity, blend mode, visibility, groups
//! (section dividers), layer masks, 8/16/32 bits per channel, RGB and
//! greyscale (CMYK is converted naively), raw / RLE / ZIP / ZIP-with-
//! prediction channel data, the embedded ICC profile, and the merged image
//! as a fallback for files without layers.
//!
//! Not (yet) carried over: layer styles, adjustment and fill layers,
//! clipping masks, smart objects and editable text (their pixels are kept).

use anyhow::{Context, Result, bail, ensure};
use imagine_core::{BlendMode, tr};

/// Pixel data of a layer or image: interleaved RGBA, straight alpha,
/// colour channels encoded as in the file (sRGB-like, or linear for 32 bit).
#[derive(Debug, Clone, PartialEq)]
pub enum Raster {
    U8(Vec<u8>),
    U16(Vec<u16>),
    /// 32-bit float, linear light.
    F32(Vec<f32>),
}

/// A rectangle in document pixels: left, top, width, height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerMask {
    pub rect: Rect,
    /// Value outside `rect`.
    pub default: u8,
    pub enabled: bool,
    /// `rect.w × rect.h` values.
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Layer {
        info: Info,
        rect: Rect,
        /// RGBA, `rect.w × rect.h`; `None` if the layer has no pixels.
        pixels: Option<Raster>,
    },
    Group {
        info: Info,
        /// Bottom → top.
        children: Vec<Node>,
        open: bool,
    },
}

/// What layers and groups share.
#[derive(Debug, Clone, PartialEq)]
pub struct Info {
    pub name: String,
    pub opacity: f32,
    pub blend: BlendMode,
    /// The blend key from the file (for modes imagine maps to Normal).
    pub blend_key: [u8; 4],
    pub visible: bool,
    pub clipped: bool,
    pub mask: Option<LayerMask>,
    /// Special content we don't carry over (adjustment, text, smart object …).
    pub special: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Psd {
    pub width: u32,
    pub height: u32,
    pub depth: u16,
    pub icc: Option<Vec<u8>>,
    /// Bottom → top. Empty if the file only has a merged image.
    pub layers: Vec<Node>,
    /// The flattened image (RGBA, full size), if stored.
    pub merged: Option<Raster>,
    /// Things that could not be carried over (shown to the user).
    pub warnings: Vec<String>,
}

// --- Reading primitives -------------------------------------------------

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
    psb: bool,
}

impl<'a> Cur<'a> {
    fn need(&self, n: usize) -> Result<()> {
        ensure!(
            self.p.checked_add(n).is_some_and(|e| e <= self.b.len()),
            "{}",
            tr("the file is truncated")
        );
        Ok(())
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
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
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.bytes(8)?.try_into()?))
    }
    /// A section length: 4 bytes, 8 in PSB where the format says so.
    fn len(&mut self, wide: bool) -> Result<usize> {
        Ok(if wide && self.psb {
            self.u64()? as usize
        } else {
            self.u32()? as usize
        })
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }
    fn sub(&mut self, n: usize) -> Result<Cur<'a>> {
        let b = self.bytes(n)?;
        Ok(Cur {
            b,
            p: 0,
            psb: self.psb,
        })
    }
    fn left(&self) -> usize {
        self.b.len() - self.p
    }
}

// --- Decompression -------------------------------------------------------

fn packbits(src: &[u8], out: &mut Vec<u8>, want: usize) -> Result<()> {
    let mut i = 0;
    let start = out.len();
    while out.len() - start < want && i < src.len() {
        let n = src[i] as i8;
        i += 1;
        if n >= 0 {
            let k = n as usize + 1;
            ensure!(i + k <= src.len(), "{}", tr("damaged RLE data"));
            out.extend_from_slice(&src[i..i + k]);
            i += k;
        } else if n != -128 {
            let k = (1 - n as isize) as usize;
            ensure!(i < src.len(), "{}", tr("damaged RLE data"));
            out.extend(std::iter::repeat_n(src[i], k));
            i += 1;
        }
    }
    out.resize(start + want, 0);
    Ok(())
}

fn inflate(data: &[u8], want: usize) -> Result<Vec<u8>> {
    let mut v = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(data, want.max(1) * 2)
        .map_err(|e| anyhow::anyhow!("{} ({e:?})", tr("damaged ZIP data")))?;
    v.resize(want, 0);
    Ok(v)
}

/// Undoes the row prediction of "ZIP with prediction".
fn unpredict(data: &mut [u8], w: usize, h: usize, depth: u16) {
    match depth {
        8 => {
            for row in data.chunks_mut(w) {
                for i in 1..row.len() {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
            }
        }
        16 => {
            for row in data.chunks_mut(w * 2) {
                let mut prev = 0u16;
                for px in row.chunks_mut(2) {
                    let v = u16::from_be_bytes([px[0], px[1]]).wrapping_add(prev);
                    px.copy_from_slice(&v.to_be_bytes());
                    prev = v;
                }
            }
        }
        32 => {
            // Bytes delta-coded across the row, stored as four byte planes.
            for row in data.chunks_mut(w * 4) {
                for i in 1..row.len() {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
                let planes = row.to_vec();
                for x in 0..w {
                    for b in 0..4 {
                        row[x * 4 + b] = planes[b * w + x];
                    }
                }
            }
        }
        _ => {}
    }
    let _ = h;
}

/// Reads one channel of `w × h` samples (compression already read).
fn channel(
    c: &mut Cur,
    compression: u16,
    w: usize,
    h: usize,
    depth: u16,
    data_len: usize,
) -> Result<Vec<u8>> {
    let bpp = (depth as usize / 8).max(1);
    let row = if depth == 1 { w.div_ceil(8) } else { w * bpp };
    let want = row * h;
    Ok(match compression {
        0 => {
            let d = c.bytes(want.min(data_len))?;
            let mut v = d.to_vec();
            v.resize(want, 0);
            v
        }
        1 => {
            let mut counts = Vec::with_capacity(h);
            for _ in 0..h {
                counts.push(if c.psb {
                    c.u32()? as usize
                } else {
                    c.u16()? as usize
                });
            }
            let mut out = Vec::with_capacity(want);
            for n in counts {
                let src = c.bytes(n)?;
                packbits(src, &mut out, row)?;
            }
            out
        }
        2 | 3 => {
            let src = c.bytes(data_len)?;
            let mut v = inflate(src, want)?;
            if compression == 3 {
                unpredict(&mut v, w, h, depth);
            }
            v
        }
        x => bail!("{}", imagine_core::tr!("unknown compression {x}", x = x)),
    })
}

// --- Layer records ------------------------------------------------------

struct Record {
    rect: Rect,
    channels: Vec<(i16, usize)>,
    blend_key: [u8; 4],
    opacity: u8,
    clipping: u8,
    flags: u8,
    mask: Option<(Rect, u8, bool)>,
    name: String,
    divider: Option<u32>,
    fill_opacity: Option<u8>,
    special: Option<&'static str>,
}

fn rect(c: &mut Cur) -> Result<Rect> {
    let (t, l, b, r) = (c.i32()?, c.i32()?, c.i32()?, c.i32()?);
    Ok(Rect {
        x: l as i64,
        y: t as i64,
        w: (r - l).max(0) as u32,
        h: (b - t).max(0) as u32,
    })
}

/// Keys whose length is 8 bytes in PSB files.
fn wide_key(k: &[u8]) -> bool {
    matches!(
        k,
        b"LMsk"
            | b"Lr16"
            | b"Lr32"
            | b"Layr"
            | b"Mt16"
            | b"Mt32"
            | b"Mtrn"
            | b"Alph"
            | b"FMsk"
            | b"lnk2"
            | b"FEid"
            | b"FXid"
            | b"PxSD"
    )
}

fn special_name(k: &[u8]) -> Option<&'static str> {
    Some(match k {
        b"TySh" | b"tySh" => "text",
        b"SoLd" | b"PlLd" | b"SoLE" => "smart object",
        b"SoCo" | b"GdFl" | b"PtFl" => "fill layer",
        b"levl" | b"curv" | b"brit" | b"hue2" | b"hue " | b"blnc" | b"selc" | b"mixr" | b"grdm"
        | b"phfl" | b"expA" | b"vibA" | b"nvrt" | b"post" | b"thrs" | b"clrL" | b"blwh" => {
            "adjustment layer"
        }
        _ => return None,
    })
}

fn record(c: &mut Cur) -> Result<Record> {
    let r = rect(c)?;
    let n = c.u16()? as usize;
    let mut channels = Vec::with_capacity(n);
    for _ in 0..n {
        let id = c.i16()?;
        let len = c.len(true)?;
        channels.push((id, len));
    }
    ensure!(c.bytes(4)? == b"8BIM", "{}", tr("damaged layer record"));
    let blend_key: [u8; 4] = c.bytes(4)?.try_into()?;
    let opacity = c.u8()?;
    let clipping = c.u8()?;
    let flags = c.u8()?;
    c.skip(1)?;
    let extra = c.u32()? as usize;
    let mut e = c.sub(extra)?;
    // Layer mask.
    let mlen = e.u32()? as usize;
    let mut mask = None;
    if mlen > 0 {
        let mut m = e.sub(mlen)?;
        let mr = rect(&mut m)?;
        let default = m.u8()?;
        let mflags = m.u8()?;
        mask = Some((mr, default, mflags & 2 == 0));
    }
    // Blending ranges.
    let blen = e.u32()? as usize;
    e.skip(blen)?;
    // Pascal name, padded to 4.
    let nl = e.u8()? as usize;
    let raw = e.bytes(nl)?;
    let mut name: String = raw.iter().map(|&b| b as char).collect();
    let pad = (4 - (nl + 1) % 4) % 4;
    e.skip(pad.min(e.left()))?;
    let mut divider = None;
    let mut fill_opacity = None;
    let mut special = None;
    while e.left() >= 12 {
        let sig = e.bytes(4)?;
        if sig != b"8BIM" && sig != b"8B64" {
            break;
        }
        let key: [u8; 4] = e.bytes(4)?.try_into()?;
        let len = e.len(wide_key(&key))?;
        let mut d = e.sub(len.min(e.left()))?;
        match &key {
            b"luni" => {
                let count = d.u32()? as usize;
                let units: Vec<u16> = (0..count).map(|_| d.u16()).collect::<Result<_>>()?;
                name = String::from_utf16_lossy(&units)
                    .trim_end_matches('\0')
                    .to_string();
            }
            b"lsct" | b"lsdk" => divider = Some(d.u32()?),
            b"iOpa" => fill_opacity = Some(d.u8()?),
            k => special = special.or(special_name(k)),
        }
    }
    Ok(Record {
        rect: r,
        channels,
        blend_key,
        opacity,
        clipping,
        flags,
        mask,
        name,
        divider,
        fill_opacity,
        special,
    })
}

pub fn blend_from_key(k: &[u8; 4]) -> Option<BlendMode> {
    // Some writers (ImageMagick) store the key byte-reversed.
    let rev = [k[3], k[2], k[1], k[0]];
    let k = if blend_from_key_exact(k).is_none() && blend_from_key_exact(&rev).is_some() {
        &rev
    } else {
        k
    };
    blend_from_key_exact(k)
}

fn blend_from_key_exact(k: &[u8; 4]) -> Option<BlendMode> {
    Some(match k {
        b"norm" | b"pass" => BlendMode::Normal,
        b"mul " => BlendMode::Multiply,
        b"scrn" => BlendMode::Screen,
        b"over" => BlendMode::Overlay,
        b"dark" => BlendMode::Darken,
        b"lite" => BlendMode::Lighten,
        b"div " => BlendMode::ColorDodge,
        b"idiv" => BlendMode::ColorBurn,
        b"hLit" => BlendMode::HardLight,
        b"sLit" => BlendMode::SoftLight,
        b"diff" => BlendMode::Difference,
        b"smud" => BlendMode::Exclusion,
        b"lddg" => BlendMode::Add,
        b"fsub" => BlendMode::Subtract,
        b"fdiv" => BlendMode::Divide,
        b"hue " => BlendMode::Hue,
        b"sat " => BlendMode::Saturation,
        b"colr" => BlendMode::Color,
        b"lum " => BlendMode::Luminosity,
        _ => return None,
    })
}

pub fn blend_key(b: BlendMode) -> &'static [u8; 4] {
    match b {
        BlendMode::Normal => b"norm",
        BlendMode::Multiply => b"mul ",
        BlendMode::Screen => b"scrn",
        BlendMode::Overlay => b"over",
        BlendMode::Darken => b"dark",
        BlendMode::Lighten => b"lite",
        BlendMode::ColorDodge => b"div ",
        BlendMode::ColorBurn => b"idiv",
        BlendMode::HardLight => b"hLit",
        BlendMode::SoftLight => b"sLit",
        BlendMode::Difference => b"diff",
        BlendMode::Exclusion => b"smud",
        BlendMode::Add => b"lddg",
        BlendMode::Subtract => b"fsub",
        BlendMode::Divide => b"fdiv",
        BlendMode::Hue => b"hue ",
        BlendMode::Saturation => b"sat ",
        BlendMode::Color => b"colr",
        BlendMode::Luminosity => b"lum ",
    }
}

// --- Channels → RGBA ----------------------------------------------------

/// Colour modes we read.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Gray,
    Rgb,
    Cmyk,
}

/// Builds interleaved RGBA from planar channels (`get(id)` → samples).
fn to_rgba(n: usize, depth: u16, mode: Mode, get: &dyn Fn(i16) -> Option<Vec<u8>>) -> Raster {
    let ch = |id: i16| get(id);
    let (r, g, b) = match mode {
        Mode::Gray => {
            let k = ch(0);
            (k.clone(), k.clone(), k)
        }
        Mode::Rgb | Mode::Cmyk => (ch(0), ch(1), ch(2)),
    };
    let k = if mode == Mode::Cmyk { ch(3) } else { None };
    let a = ch(-1);
    match depth {
        16 => {
            let s = |v: &Option<Vec<u8>>, i: usize, d: u16| {
                v.as_ref()
                    .map_or(d, |v| u16::from_be_bytes([v[i * 2], v[i * 2 + 1]]))
            };
            let mut out = Vec::with_capacity(n * 4);
            for i in 0..n {
                let (mut rr, mut gg, mut bb) = (s(&r, i, 0), s(&g, i, 0), s(&b, i, 0));
                if mode == Mode::Cmyk {
                    // Stored inverted (0 = full ink); naive conversion.
                    let kk = s(&k, i, 65535) as u32;
                    let f = |c: u16| ((c as u32 * kk) / 65535) as u16;
                    (rr, gg, bb) = (f(rr), f(gg), f(bb));
                }
                out.extend([rr, gg, bb, s(&a, i, 65535)]);
            }
            Raster::U16(out)
        }
        32 => {
            let s = |v: &Option<Vec<u8>>, i: usize, d: f32| {
                v.as_ref().map_or(d, |v| {
                    f32::from_be_bytes([v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]])
                })
            };
            let mut out = Vec::with_capacity(n * 4);
            for i in 0..n {
                out.extend([s(&r, i, 0.0), s(&g, i, 0.0), s(&b, i, 0.0), s(&a, i, 1.0)]);
            }
            Raster::F32(out)
        }
        _ => {
            let s = |v: &Option<Vec<u8>>, i: usize, d: u8| v.as_ref().map_or(d, |v| v[i]);
            let mut out = Vec::with_capacity(n * 4);
            for i in 0..n {
                let (mut rr, mut gg, mut bb) = (s(&r, i, 0), s(&g, i, 0), s(&b, i, 0));
                if mode == Mode::Cmyk {
                    let kk = s(&k, i, 255) as u32;
                    let f = |c: u8| ((c as u32 * kk) / 255) as u8;
                    (rr, gg, bb) = (f(rr), f(gg), f(bb));
                }
                out.extend([rr, gg, bb, s(&a, i, 255)]);
            }
            Raster::U8(out)
        }
    }
}

/// Mask channel samples → 8 bit.
fn to_u8(v: &[u8], depth: u16) -> Vec<u8> {
    match depth {
        16 => v.chunks(2).map(|p| p[0]).collect(),
        32 => v
            .chunks(4)
            .map(|p| {
                (f32::from_be_bytes([p[0], p[1], p[2], p[3]]).clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect(),
        _ => v.to_vec(),
    }
}

// --- The file -----------------------------------------------------------

pub fn read(bytes: &[u8]) -> Result<Psd> {
    let mut c = Cur {
        b: bytes,
        p: 0,
        psb: false,
    };
    ensure!(c.bytes(4)? == b"8BPS", "{}", tr("not a Photoshop file"));
    let version = c.u16()?;
    ensure!(
        version == 1 || version == 2,
        "{}",
        imagine_core::tr!("unknown PSD version {v}", v = version)
    );
    c.psb = version == 2;
    c.skip(6)?;
    let nch = c.u16()? as usize;
    let height = c.u32()?;
    let width = c.u32()?;
    let depth = c.u16()?;
    let mode = match c.u16()? {
        1 => Mode::Gray,
        3 => Mode::Rgb,
        4 => Mode::Cmyk,
        8 if depth == 8 || depth == 16 => Mode::Gray, // duotone: the grey data
        m => bail!(
            "{}",
            imagine_core::tr!(
                "unsupported PSD color mode {m} (only RGB, grayscale and CMYK)",
                m = m
            )
        ),
    };
    ensure!(
        matches!(depth, 8 | 16 | 32),
        "{}",
        imagine_core::tr!("unsupported bit depth {d}", d = depth)
    );
    let mut warnings = Vec::new();
    if mode == Mode::Cmyk {
        warnings.push(
            tr("CMYK converted to RGB without a color profile (colors are approximate)")
                .to_string(),
        );
    }
    // Colour mode data.
    let n = c.u32()? as usize;
    c.skip(n)?;
    // Image resources: the ICC profile (1039).
    let n = c.u32()? as usize;
    let mut res = c.sub(n)?;
    let mut icc = None;
    while res.left() >= 12 {
        if res.bytes(4)? != b"8BIM" {
            break;
        }
        let id = res.u16()?;
        let nl = res.u8()? as usize;
        // Pascal name, padded to an even size (length byte included).
        res.skip(nl + (nl + 1) % 2)?;
        let len = res.u32()? as usize;
        let data = res.bytes(len.min(res.left()))?;
        if len % 2 == 1 && res.left() > 0 {
            res.skip(1)?;
        }
        if id == 1039 {
            icc = Some(data.to_vec());
        }
    }
    // Layer and mask information.
    let lm_len = c.len(true)?;
    let mut lm = c.sub(lm_len)?;
    let mut records = Vec::new();
    let mut layer_data: Vec<Vec<(i16, Vec<u8>)>> = Vec::new();
    let parse_layers = |info: &mut Cur,
                        records: &mut Vec<Record>,
                        layer_data: &mut Vec<Vec<(i16, Vec<u8>)>>|
     -> Result<()> {
        let count = info.i16()?.unsigned_abs() as usize;
        for _ in 0..count {
            records.push(record(info)?);
        }
        for r in records.iter() {
            let mut chans = Vec::new();
            for &(id, len) in &r.channels {
                let mut d = info.sub(len.min(info.left()))?;
                if len < 2 {
                    continue;
                }
                let comp = d.u16()?;
                let rr = if id == -2 || id == -3 {
                    r.mask.map(|m| m.0).unwrap_or_default()
                } else {
                    r.rect
                };
                if rr.w == 0 || rr.h == 0 {
                    continue;
                }
                let samples = channel(&mut d, comp, rr.w as usize, rr.h as usize, depth, len - 2)?;
                chans.push((id, samples));
            }
            layer_data.push(chans);
        }
        Ok(())
    };
    if lm_len > 0 {
        let li_len = lm.len(true)?;
        if li_len > 0 {
            let mut info = lm.sub(li_len)?;
            parse_layers(&mut info, &mut records, &mut layer_data)?;
        }
        // Global layer mask info.
        if lm.left() >= 4 {
            let g = lm.u32()? as usize;
            lm.skip(g.min(lm.left()))?;
        }
        // 16/32-bit documents keep their layers in a tagged block.
        while records.is_empty() && lm.left() >= 12 {
            let sig = lm.bytes(4)?;
            if sig != b"8BIM" && sig != b"8B64" {
                break;
            }
            let key: [u8; 4] = lm.bytes(4)?.try_into()?;
            let len = lm.len(wide_key(&key))?;
            let mut d = lm.sub(len.min(lm.left()))?;
            if matches!(&key, b"Lr16" | b"Lr32" | b"Layr") {
                parse_layers(&mut d, &mut records, &mut layer_data)?;
            }
            // Blocks are padded to 4 bytes.
            let pad = (4 - len % 4) % 4;
            lm.skip(pad.min(lm.left()))?;
        }
    }
    // Merged image.
    let mut merged = None;
    if c.left() >= 2 {
        let comp = c.u16()?;
        let (w, h) = (width as usize, height as usize);
        let bpp = depth as usize / 8;
        let plane = w * h * bpp;
        let planes: Result<Vec<Vec<u8>>> = (|| {
            Ok(match comp {
                0 => (0..nch)
                    .map(|_| c.bytes(plane).map(<[u8]>::to_vec))
                    .collect::<Result<_>>()?,
                1 => {
                    let total = h * nch;
                    let mut counts = Vec::with_capacity(total);
                    for _ in 0..total {
                        counts.push(if c.psb {
                            c.u32()? as usize
                        } else {
                            c.u16()? as usize
                        });
                    }
                    let mut out = vec![Vec::with_capacity(plane); nch];
                    for (i, n) in counts.into_iter().enumerate() {
                        let src = c.bytes(n)?;
                        packbits(src, &mut out[i / h], w * bpp)?;
                    }
                    out
                }
                _ => {
                    let all = inflate(c.bytes(c.left())?, plane * nch)?;
                    let mut out: Vec<Vec<u8>> = all.chunks(plane).map(<[u8]>::to_vec).collect();
                    if comp == 3 {
                        for p in &mut out {
                            unpredict(p, w, h, depth);
                        }
                    }
                    out
                }
            })
        })();
        if let Ok(planes) = planes {
            let colours = match mode {
                Mode::Gray => 1,
                Mode::Rgb => 3,
                Mode::Cmyk => 4,
            };
            let get = |id: i16| -> Option<Vec<u8>> {
                let i = if id == -1 { colours } else { id as usize };
                planes.get(i).cloned()
            };
            merged = Some(to_rgba(w * h, depth, mode, &get));
        }
    }
    // Records (bottom → top) → tree.
    let mut stack: Vec<Vec<Node>> = vec![Vec::new()];
    let mut clipped = 0;
    let mut specials = std::collections::BTreeMap::<&str, usize>::new();
    let mut unknown_blend = 0;
    for (r, data) in records.into_iter().zip(layer_data) {
        let get_mask = |id: i16| data.iter().find(|(i, _)| *i == id).map(|(_, v)| v.clone());
        let mask = r.mask.and_then(|(mr, default, enabled)| {
            let d = get_mask(-2)?;
            Some(LayerMask {
                rect: mr,
                default,
                enabled,
                data: to_u8(&d, depth),
            })
        });
        let blend = blend_from_key(&r.blend_key);
        if blend.is_none() {
            unknown_blend += 1;
        }
        let mut opacity = r.opacity as f32 / 255.0;
        if let Some(f) = r.fill_opacity {
            opacity *= f as f32 / 255.0;
        }
        let info = Info {
            name: r.name.clone(),
            opacity,
            blend: blend.unwrap_or_default(),
            blend_key: r.blend_key,
            visible: r.flags & 2 == 0,
            clipped: r.clipping != 0,
            mask,
            special: r.special,
        };
        if info.clipped && r.divider.is_none() {
            clipped += 1;
        }
        if let Some(s) = r.special {
            *specials.entry(s).or_default() += 1;
        }
        match r.divider {
            Some(3) => stack.push(Vec::new()),
            Some(t @ (1 | 2)) => {
                let children = if stack.len() > 1 {
                    stack.pop().unwrap()
                } else {
                    Vec::new()
                };
                stack.last_mut().unwrap().push(Node::Group {
                    info,
                    children,
                    open: t == 1,
                });
            }
            _ => {
                let n = r.rect.w as usize * r.rect.h as usize;
                let has = data.iter().any(|(id, _)| *id >= 0);
                let pixels = (n > 0 && has).then(|| {
                    let get = |id: i16| data.iter().find(|(i, _)| *i == id).map(|(_, v)| v.clone());
                    to_rgba(n, depth, mode, &get)
                });
                stack.last_mut().unwrap().push(Node::Layer {
                    info,
                    rect: r.rect,
                    pixels,
                });
            }
        }
    }
    // Unbalanced groups: flatten what's left.
    while stack.len() > 1 {
        let inner = stack.pop().unwrap();
        stack.last_mut().unwrap().extend(inner);
    }
    let layers = stack.pop().unwrap();
    if clipped > 0 {
        warnings.push(imagine_core::tr!(
            "{n} clipping masks are not supported yet: those layers are shown unclipped",
            n = clipped
        ));
    }
    for (what, n) in specials {
        warnings.push(imagine_core::tr!(
            "{n} × {what}: only the pixels were taken over",
            n = n,
            what = tr(what)
        ));
    }
    if unknown_blend > 0 {
        warnings.push(imagine_core::tr!(
            "{n} layers use a blend mode imagine does not have yet (shown as Normal)",
            n = unknown_blend
        ));
    }
    Ok(Psd {
        width,
        height,
        depth,
        icc,
        layers,
        merged,
        warnings,
    })
}

/// Reads a PSD/PSB file.
pub fn open(path: &std::path::Path) -> Result<Psd> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    read(&bytes)
}

pub fn is_psd(path: &std::path::Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("psd") || e.eq_ignore_ascii_case("psb"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_decodes_runs_and_literals() {
        let mut out = Vec::new();
        // 3 literals, then 4× 0xAA.
        packbits(&[2, 1, 2, 3, (-3i8) as u8, 0xAA], &mut out, 7).unwrap();
        assert_eq!(out, [1, 2, 3, 0xAA, 0xAA, 0xAA, 0xAA]);
    }

    #[test]
    fn prediction_16_bit() {
        // Row 10, +5, +5 → 10, 15, 20.
        let mut d = vec![0, 10, 0, 5, 0, 5];
        unpredict(&mut d, 3, 1, 16);
        assert_eq!(d, [0, 10, 0, 15, 0, 20]);
    }
}
