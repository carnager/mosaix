//! Photoshop documents (PSD and the large-document variant PSB).
//!
//! Reading: layers with names, opacity, blend mode, visibility, groups
//! (section dividers), layer masks, 8/16/32 bits per channel, RGB and
//! greyscale (CMYK is converted naively), raw / RLE / ZIP / ZIP-with-
//! prediction channel data, the embedded ICC profile, and the merged image
//! as a fallback for files without layers.
//!
//! Also: layer styles, most adjustment layers, shape and fill layers
//! (solid, gradient, pattern; as mosaix shape layers) and text.
//!
//! Not (yet) carried over: smart objects, noise gradients and some
//! adjustments (their pixels are kept).

use anyhow::{Context, Result, bail, ensure};
use mosaix_core::{BlendMode, tr};

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
    /// The blend key from the file (for modes mosaix maps to Normal).
    pub blend_key: [u8; 4],
    pub visible: bool,
    pub clipped: bool,
    pub mask: Option<LayerMask>,
    /// Special content we don't carry over (adjustment, text, smart object …).
    pub special: Option<&'static str>,
    /// An adjustment layer mosaix can represent.
    pub adjustment: Option<mosaix_core::Adjustment>,
    /// A text layer (editable if the font is available).
    pub text: Option<Text>,
    /// Layer style.
    pub effects: Vec<mosaix_core::style::Effect>,
    /// A shape or solid-colour fill layer mosaix can represent.
    pub shape: Option<mosaix_core::vector::ShapeSpec>,
    /// Fill opacity ("iOpa"): the layer's pixels only, not its style.
    pub fill_opacity: f32,
}

/// A Photoshop text layer, as far as mosaix can use it.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub text: String,
    /// PostScript name of the first run's font ("ArialMT", "Helvetica-Bold" …).
    pub font: String,
    /// Size in px before `transform`.
    pub size: f32,
    /// sRGB-encoded RGBA.
    pub color: [f32; 4],
    pub align: mosaix_core::vector::TextAlign,
    /// Line height as a multiple of the size (auto leading: 1.2).
    pub line_height: f32,
    /// Letter spacing in px.
    pub tracking: f32,
    pub faux_bold: bool,
    pub faux_italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
    /// Paragraph text: box width (px).
    pub width: Option<f32>,
    /// [xx, xy, yx, yy, tx, ty]: baseline origin of the first line.
    pub transform: [f64; 6],
    /// OpenType features Photoshop switched (tag → value).
    pub features: std::collections::BTreeMap<String, u16>,
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
    pub guides: Vec<mosaix_core::Guide>,
    /// Alpha channels (saved selections): name and one byte per pixel.
    pub channels: Vec<(String, Vec<u8>)>,
}

/// What a PSD export carries besides layers: guides and saved selections
/// (as named alpha channels, one byte per pixel of the canvas).
#[derive(Default)]
pub struct Extras {
    pub guides: Vec<mosaix_core::Guide>,
    pub channels: Vec<(String, Vec<u8>)>,
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

pub(crate) fn packbits(src: &[u8], out: &mut Vec<u8>, want: usize) -> Result<()> {
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
        x => bail!("{}", mosaix_core::tr!("unknown compression {x}", x = x)),
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
    /// The pixel mask when the layer also has a vector mask (channel −3).
    real_mask: Option<(Rect, u8, bool)>,
    name: String,
    divider: Option<u32>,
    fill_opacity: Option<u8>,
    special: Option<&'static str>,
    adjustment: Option<mosaix_core::Adjustment>,
    text: Option<Text>,
    /// Layer style ("lfx2" / "lmfx" descriptor).
    fx: Option<crate::psd_data::Desc>,
    /// The mask channel −2 is a rendering of the vector mask.
    mask_from_vector: bool,
    /// Vector mask ("vmsk" / "vsms").
    vmask: Option<Vec<u8>>,
    /// Solid fill content ("SoCo" / "vscg" with SoCo).
    solid: Option<crate::psd_data::Desc>,
    /// Gradient or pattern fill content ("GdFl" / "PtFl", or in "vscg").
    paint: Option<([u8; 4], crate::psd_data::Desc)>,
    /// Shape stroke ("vstk").
    vstroke: Option<crate::psd_data::Desc>,
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
    let mut real_mask = None;
    let mut mask_from_vector = false;
    if mlen > 0 {
        let mut m = e.sub(mlen)?;
        let mr = rect(&mut m)?;
        let default = m.u8()?;
        let mflags = m.u8()?;
        mask = Some((mr, default, mflags & 2 == 0));
        mask_from_vector = mflags & 8 != 0;
        if mlen >= 36 {
            // Optional mask parameters, then the real (pixel) mask.
            if mflags & 0x10 != 0 {
                let params = m.u8()?;
                for (bit, n) in [(1u8, 1usize), (2, 8), (4, 1), (8, 8)] {
                    if params & bit != 0 {
                        m.skip(n)?;
                    }
                }
            }
            if m.left() >= 18 {
                let rflags = m.u8()?;
                let rdefault = m.u8()?;
                let rr = rect(&mut m)?;
                real_mask = Some((rr, rdefault, rflags & 2 == 0));
            }
        }
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
    let mut adjustment = None;
    let mut text = None;
    let mut cged: Option<crate::psd_data::Desc> = None;
    let mut fx: Option<crate::psd_data::Desc> = None;
    let mut vmask = None;
    let mut solid = None;
    let mut paint = None;
    let mut vstroke = None;
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
            b"CgEd" => {
                // New-style brightness/contrast (overrides "brit").
                // Descriptor version (16), then the descriptor.
                if d.left() > 4 {
                    d.skip(4)?;
                    cged = crate::psd_data::read_desc(d.bytes(d.left())?).ok();
                }
            }
            b"lfx2" | b"lmfx" => {
                // Effects version, descriptor version, descriptor. "lmfx"
                // (several strokes/shadows) supersedes "lfx2".
                if d.left() > 8 && (fx.is_none() || &key == b"lmfx") {
                    d.skip(8)?;
                    if let Ok(desc) = crate::psd_data::read_desc(d.bytes(d.left())?) {
                        fx = Some(desc);
                    }
                }
            }
            b"vmsk" | b"vsms" => vmask = Some(d.bytes(d.left())?.to_vec()),
            b"vstk" if d.left() > 4 => {
                d.skip(4)?;
                vstroke = crate::psd_data::read_desc(d.bytes(d.left())?).ok();
            }
            b"vscg" if d.left() > 8 => {
                // Content key, then version and descriptor.
                let content: [u8; 4] = d.bytes(4)?.try_into()?;
                d.skip(4)?;
                let desc = crate::psd_data::read_desc(d.bytes(d.left())?).ok();
                match &content {
                    b"SoCo" => solid = desc,
                    b"GdFl" | b"PtFl" => {
                        special = special.or(Some("fill layer"));
                        paint = desc.map(|d| (content, d));
                    }
                    _ => special = special.or(Some("fill layer")),
                }
            }
            b"SoCo" if d.left() > 4 => {
                special = special.or(Some("fill layer"));
                d.skip(4)?;
                solid = solid.or(crate::psd_data::read_desc(d.bytes(d.left())?).ok());
            }
            b"GdFl" | b"PtFl" if d.left() > 4 => {
                special = special.or(Some("fill layer"));
                d.skip(4)?;
                if paint.is_none() {
                    paint = crate::psd_data::read_desc(d.bytes(d.left())?).ok().map(|x| (key, x));
                }
            }
            b"TySh" => {
                special = special.or(Some("text"));
                text = crate::psd_layers::read_text(d.bytes(d.left())?).ok();
            }
            k => {
                if let Some(s) = special_name(k) {
                    special = special.or(Some(s));
                    if s == "adjustment layer" {
                        adjustment = crate::psd_layers::read_adjustment(k, d.bytes(d.left())?);
                    }
                }
            }
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
        real_mask,
        name,
        divider,
        fill_opacity,
        special,
        adjustment: match (adjustment, cged) {
            (Some(mosaix_core::Adjustment::BrightnessContrast { .. }), Some(c))
                if c.bool("useLegacy") != Some(true) =>
            {
                Some(mosaix_core::Adjustment::BrightnessContrast {
                    brightness: c.num("Brgh").unwrap_or(0.0) as f32,
                    contrast: c.num("Cntr").unwrap_or(0.0) as f32,
                    legacy: false,
                })
            }
            (a, _) => a,
        },
        text,
        fx,
        mask_from_vector,
        vmask,
        solid,
        paint,
        vstroke,
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
        mosaix_core::tr!("unknown PSD version {v}", v = version)
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
            mosaix_core::tr!(
                "unsupported PSD color mode {m} (only RGB, grayscale and CMYK)",
                m = m
            )
        ),
    };
    ensure!(
        matches!(depth, 8 | 16 | 32),
        "{}",
        mosaix_core::tr!("unsupported bit depth {d}", d = depth)
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
    let mut guides = Vec::new();
    let mut names_pascal: Vec<String> = Vec::new();
    let mut names_unicode: Vec<String> = Vec::new();
    let mut light = crate::psd_fx::GlobalLight::default();
    let mut dpi = 72.0f64;
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
        // Grid and guides: version, grid (2 × 4), count, then (position in
        // 1/32 px, direction: 0 vertical, 1 horizontal).
        if id == 1032 && data.len() >= 16 {
            let n = u32::from_be_bytes(data[12..16].try_into()?) as usize;
            for k in 0..n {
                let at = 16 + k * 5;
                if data.len() < at + 5 {
                    break;
                }
                let pos = i32::from_be_bytes(data[at..at + 4].try_into()?) as f64 / 32.0;
                guides.push(mosaix_core::Guide { vertical: data[at + 4] == 0, pos });
            }
        }
        // Alpha channel names: Pascal strings (1006) or Unicode (1045).
        if id == 1006 {
            let mut i = 0;
            while i < data.len() {
                let n = data[i] as usize;
                names_pascal.push(String::from_utf8_lossy(&data[i + 1..(i + 1 + n).min(data.len())]).into_owned());
                i += 1 + n;
            }
        }
        if id == 1045 {
            let mut i = 0;
            while i + 4 <= data.len() {
                let n = u32::from_be_bytes(data[i..i + 4].try_into()?) as usize;
                i += 4;
                let units: Vec<u16> = (0..n)
                    .filter_map(|k| data.get(i + 2 * k..i + 2 * k + 2).map(|b| u16::from_be_bytes([b[0], b[1]])))
                    .collect();
                i += 2 * n;
                names_unicode.push(String::from_utf16_lossy(&units).trim_end_matches('\0').to_string());
            }
        }
        if id == 1005 && data.len() >= 4 {
            // Resolution: horizontal, 16.16 fixed point.
            dpi = u32::from_be_bytes(data[..4].try_into()?) as f64 / 65536.0;
        }
        if data.len() >= 4 && (id == 1037 || id == 1049) {
            let v = i32::from_be_bytes(data[..4].try_into()?) as f32;
            if id == 1037 {
                light.angle = v;
            } else {
                light.altitude = v;
            }
        }
    }
    // Layer and mask information.
    let lm_len = c.len(true)?;
    let mut lm = c.sub(lm_len)?;
    let mut records = Vec::new();
    let mut layer_data: Vec<Vec<(i16, Vec<u8>)>> = Vec::new();
    let mut patterns: Vec<(String, mosaix_core::pattern::Pattern)> = Vec::new();
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
                let rr = if id == -3 {
                    r.real_mask.or(r.mask).map(|m| m.0).unwrap_or_default()
                } else if id == -2 {
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
        // Tagged blocks: the layers of 16/32-bit documents, the patterns.
        while lm.left() >= 12 {
            let sig = lm.bytes(4)?;
            if sig != b"8BIM" && sig != b"8B64" {
                break;
            }
            let key: [u8; 4] = lm.bytes(4)?.try_into()?;
            let len = lm.len(wide_key(&key))?;
            let mut d = lm.sub(len.min(lm.left()))?;
            if matches!(&key, b"Lr16" | b"Lr32" | b"Layr") && records.is_empty() {
                parse_layers(&mut d, &mut records, &mut layer_data)?;
            }
            if matches!(&key, b"Patt" | b"Pat2" | b"Pat3") {
                patterns.extend(crate::psd_patterns::read(d.bytes(d.left())?));
            }
            // Blocks are padded to 4 bytes.
            let pad = (4 - len % 4) % 4;
            lm.skip(pad.min(lm.left()))?;
        }
    }
    // Merged image.
    let mut merged = None;
    let mut channels = Vec::new();
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
            // The named alpha channels are the last planes.
            let names = if names_unicode.is_empty() { &names_pascal } else { &names_unicode };
            let extra = planes.len().saturating_sub(colours);
            let k = names.len().min(extra);
            for (name, plane) in names.iter().take(k).zip(&planes[planes.len() - k..]) {
                let bytes: Vec<u8> = match depth {
                    16 => plane.chunks(2).map(|b| b[0]).collect(),
                    32 => plane.chunks(4).map(|b| (f32::from_be_bytes([b[0], b[1], b[2], b[3]]).clamp(0.0, 1.0) * 255.0).round() as u8).collect(),
                    _ => plane.clone(),
                };
                channels.push((name.clone(), bytes));
            }
        }
    }
    // Records (bottom → top) → tree.
    let mut stack: Vec<Vec<Node>> = vec![Vec::new()];
    let mut specials = std::collections::BTreeMap::<&str, usize>::new();
    let mut lost_fx = std::collections::BTreeMap::<&str, usize>::new();
    let mut unknown_blend = 0;
    for (r, data) in records.into_iter().zip(layer_data) {
        let get_mask = |id: i16| data.iter().find(|(i, _)| *i == id).map(|(_, v)| v.clone());
        // With a vector mask too, −2 is the rasterised vector mask and −3
        // the user's pixel mask; prefer the pixel mask.
        // Shape and solid fill layers become mosaix shape layers.
        // Gradient and pattern fill layers too (shapes painted with them).
        let paint = r.paint.as_ref().and_then(|(k, d)| crate::psd_fx::read_paint(k, d, &patterns, (width, height)));
        let fill = match &paint {
            Some(mosaix_core::vector::Paint::Gradient { gradient, .. }) => {
                let c = gradient.ends().0;
                Some([c[0], c[1], c[2], 1.0])
            }
            Some(mosaix_core::vector::Paint::Pattern { .. }) => Some([0.5, 0.5, 0.5, 1.0]),
            None => r.solid.as_ref().and_then(crate::psd_shape::solid_fill),
        };
        let shape = fill.and_then(|fill| {
            let vm = match &r.vmask {
                Some(b) => Some(crate::psd_shape::read_vector_mask(b, (width, height))?),
                None => None,
            };
            let st = r.vstroke.as_ref().map(|d| crate::psd_shape::stroke(d, dpi.max(1.0)));
            crate::psd_shape::shape(vm.as_ref(), fill, st, (width, height)).map(|mut s| {
                s.paint = paint;
                s
            })
        });
        let (which, m) = match (get_mask(-3).is_some(), r.real_mask) {
            (true, Some(rm)) => (-3, Some(rm)),
            // A rendering of the vector mask is not needed for a shape.
            _ if shape.is_some() && (r.mask_from_vector || r.vmask.is_some() && r.real_mask.is_some()) => (-2, None),
            _ => (-2, r.mask),
        };
        let mask = m.and_then(|(mr, default, enabled)| {
            let d = get_mask(which)?;
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
        let opacity = r.opacity as f32 / 255.0;
        let fill_opacity = r.fill_opacity.map_or(1.0, |f| f as f32 / 255.0);
        let effects = match &r.fx {
            Some(fx) => {
                let (fx, lost) = crate::psd_fx::read(fx, light, &patterns);
                for l in lost {
                    *lost_fx.entry(l).or_default() += 1;
                }
                fx
            }
            None => Vec::new(),
        };
        let info = Info {
            fill_opacity,
            shape,
            effects,
            name: r.name.clone(),
            opacity,
            blend: blend.unwrap_or_default(),
            blend_key: r.blend_key,
            visible: r.flags & 2 == 0,
            clipped: r.clipping != 0,
            mask,
            special: r.special,
            adjustment: r.adjustment.clone(),
            text: r.text.clone(),
        };
        // Report only what is not carried over.
        let handled = match r.special {
            Some("adjustment layer") => r.adjustment.is_some(),
            Some("text") => r.text.is_some(),
            Some("fill layer") => info.shape.is_some(),
            _ => false,
        };
        if let Some(s) = r.special.filter(|_| !handled) {
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
    for (what, n) in specials {
        warnings.push(mosaix_core::tr!(
            "{n} × {what}: only the pixels were taken over",
            n = n,
            what = tr(what)
        ));
    }
    for (what, n) in lost_fx {
        warnings.push(mosaix_core::tr!(
            "Layer style: {n} × {what} not carried over",
            n = n,
            what = tr(what)
        ));
    }
    if unknown_blend > 0 {
        warnings.push(mosaix_core::tr!(
            "{n} layers use a blend mode mosaix does not have yet (shown as Normal)",
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
        guides,
        channels,
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

// --- Writing ------------------------------------------------------------

/// A layer to write: premultiplied linear tiles (as the engine stores them)
/// and an optional mask (default value, enabled, 8-bit tiles).
pub struct WLayer {
    pub name: String,
    pub opacity: f32,
    pub blend: BlendMode,
    pub visible: bool,
    pub clip: bool,
    /// Groups: Pass Through.
    pub pass_through: bool,
    /// An adjustment layer (written as Photoshop's adjustment layer).
    pub adjustment: Option<mosaix_core::Adjustment>,
    /// Layer style (written as Photoshop's "lfx2").
    pub effects: Vec<mosaix_core::style::Effect>,
    /// Fill opacity (written as "iOpa").
    pub fill_opacity: f32,
    /// A shape layer (written as a fill layer with vector mask and stroke;
    /// the pixels stay as Photoshop's rendering of it).
    pub shape: Option<crate::psd_shape::PsdShape>,
    /// A text layer (written as Photoshop's editable text, "TySh") and its
    /// box height for paragraph text.
    pub text: Option<(Text, f32)>,
    pub tiles: Vec<(mosaix_core::TileCoord, Vec<half::f16>)>,
    pub mask: Option<WMask>,
}

#[derive(Clone)]
pub struct WMask {
    pub default: u8,
    pub enabled: bool,
    pub tiles: Vec<(mosaix_core::TileCoord, Vec<u8>)>,
}

pub enum WNode {
    Layer(WLayer),
    Group { layer: WLayer, children: Vec<WNode> },
}

struct Out {
    b: Vec<u8>,
    psb: bool,
}

impl Out {
    fn u8(&mut self, v: u8) {
        self.b.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.b.extend(v.to_be_bytes());
    }
    fn i16(&mut self, v: i16) {
        self.b.extend(v.to_be_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.b.extend(v.to_be_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.b.extend(v.to_be_bytes());
    }
    fn len(&mut self, v: usize, wide: bool) {
        if wide && self.psb {
            self.b.extend((v as u64).to_be_bytes());
        } else {
            self.b.extend((v as u32).to_be_bytes());
        }
    }
    /// Reserves a length field; `close` fills it with the bytes written since,
    /// padded to `align`.
    fn open(&mut self, wide: bool) -> (usize, bool) {
        let at = self.b.len();
        self.len(0, wide);
        (at, wide && self.psb)
    }
    fn close(&mut self, (at, wide): (usize, bool), align: usize) {
        let start = at + if wide { 8 } else { 4 };
        while !(self.b.len() - start).is_multiple_of(align) {
            self.b.push(0);
        }
        let n = self.b.len() - start;
        if wide {
            self.b[at..at + 8].copy_from_slice(&(n as u64).to_be_bytes());
        } else {
            self.b[at..at + 4].copy_from_slice(&(n as u32).to_be_bytes());
        }
    }
}

/// PackBits encoding of one row.
fn packbits_encode(src: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    while i < src.len() {
        // A run of at least 3 equal bytes?
        let mut run = 1;
        while i + run < src.len() && run < 128 && src[i + run] == src[i] {
            run += 1;
        }
        if run >= 3 {
            out.push((1i16 - run as i16) as i8 as u8);
            out.push(src[i]);
            i += run;
            continue;
        }
        // Literals until the next run of 3.
        let start = i;
        while i < src.len() && i - start < 128 {
            if i + 2 < src.len() && src[i] == src[i + 1] && src[i] == src[i + 2] {
                break;
            }
            i += 1;
        }
        out.push((i - start - 1) as u8);
        out.extend_from_slice(&src[start..i]);
    }
}

/// A channel plane (`w × h` samples, big-endian) → RLE data incl. compression
/// field and row counts.
fn rle_channel(plane: &[u8], row_bytes: usize, h: usize, psb: bool) -> Vec<u8> {
    let mut rows = Vec::with_capacity(h);
    for r in 0..h {
        let mut v = Vec::new();
        packbits_encode(&plane[r * row_bytes..(r + 1) * row_bytes], &mut v);
        rows.push(v);
    }
    let mut out = Vec::new();
    out.extend(1u16.to_be_bytes());
    for r in &rows {
        if psb {
            out.extend((r.len() as u32).to_be_bytes());
        } else {
            out.extend((r.len() as u16).to_be_bytes());
        }
    }
    for r in rows {
        out.extend(r);
    }
    out
}

/// Layer pixels ready to write: rect and RLE-coded channels (id, data).
struct Encoded {
    rect: Rect,
    channels: Vec<(i16, Vec<u8>)>,
    mask: Option<(Rect, u8, bool)>,
}

/// Premultiplied linear tiles → trimmed rect and planar, encoded channels.
fn encode_layer(l: &WLayer, depth: u16, convert: &crate::export::Converter, psb: bool) -> Encoded {
    use mosaix_core::TILE_SIZE;
    let ts = TILE_SIZE as i64;
    // Straight, encoded colour + alpha per tile (0…1).
    let tiles: Vec<(mosaix_core::TileCoord, Vec<f32>)> = l
        .tiles
        .iter()
        .map(|(c, px)| {
            let n = (ts * ts) as usize;
            let mut rgb = vec![0.0f32; n * 3];
            let mut a = vec![0.0f32; n];
            for i in 0..n {
                let al = px[i * 4 + 3].to_f32().clamp(0.0, 1.0);
                a[i] = al;
                if al > 0.0 {
                    for k in 0..3 {
                        rgb[i * 3 + k] = px[i * 4 + k].to_f32() / al;
                    }
                }
            }
            for row in rgb.chunks_mut(ts as usize * 3) {
                convert(row);
            }
            let mut out = vec![0.0f32; n * 4];
            for i in 0..n {
                out[i * 4..i * 4 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
                out[i * 4 + 3] = a[i];
            }
            (*c, out)
        })
        .collect();
    // Bounds of the visible pixels.
    let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for (c, px) in &tiles {
        let (ox, oy) = c.origin();
        for y in 0..ts {
            for x in 0..ts {
                if px[((y * ts + x) * 4 + 3) as usize] > 0.0 {
                    x0 = x0.min(ox + x);
                    y0 = y0.min(oy + y);
                    x1 = x1.max(ox + x + 1);
                    y1 = y1.max(oy + y + 1);
                }
            }
        }
    }
    let rect = if x1 > x0 {
        Rect {
            x: x0,
            y: y0,
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        }
    } else {
        Rect::default()
    };
    let bpp = depth as usize / 8;
    let (w, h) = (rect.w as usize, rect.h as usize);
    let mut planes = vec![vec![0u8; w * h * bpp]; 4];
    for (c, px) in &tiles {
        let (ox, oy) = c.origin();
        for y in 0..ts {
            let dy = oy + y - rect.y;
            if dy < 0 || dy >= h as i64 {
                continue;
            }
            for x in 0..ts {
                let dx = ox + x - rect.x;
                if dx < 0 || dx >= w as i64 {
                    continue;
                }
                let s = ((y * ts + x) * 4) as usize;
                let d = (dy as usize * w + dx as usize) * bpp;
                for (k, plane) in planes.iter_mut().enumerate() {
                    let v = px[s + k].clamp(0.0, 1.0);
                    if bpp == 1 {
                        plane[d] = (v * 255.0).round() as u8;
                    } else {
                        plane[d..d + 2]
                            .copy_from_slice(&((v * 65535.0).round() as u16).to_be_bytes());
                    }
                }
            }
        }
    }
    let ids = [0i16, 1, 2, -1];
    let mut channels: Vec<(i16, Vec<u8>)> = ids
        .iter()
        .zip(&planes)
        .map(|(id, p)| {
            (
                *id,
                if w * h == 0 {
                    0u16.to_be_bytes().to_vec()
                } else {
                    rle_channel(p, w * bpp, h, psb)
                },
            )
        })
        .collect();
    // Mask: bounds of its tiles, default outside.
    let mut mask = None;
    if let Some(m) = &l.mask {
        let (mut mx0, mut my0, mut mx1, mut my1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for (c, _) in &m.tiles {
            let (ox, oy) = c.origin();
            mx0 = mx0.min(ox);
            my0 = my0.min(oy);
            mx1 = mx1.max(ox + ts);
            my1 = my1.max(oy + ts);
        }
        let mr = if mx1 > mx0 {
            Rect {
                x: mx0,
                y: my0,
                w: (mx1 - mx0) as u32,
                h: (my1 - my0) as u32,
            }
        } else {
            Rect::default()
        };
        let (mw, mh) = (mr.w as usize, mr.h as usize);
        let mut plane = vec![m.default; mw * mh];
        for (c, px) in &m.tiles {
            let (ox, oy) = c.origin();
            for y in 0..ts {
                let row = ((oy + y - mr.y) as usize) * mw + (ox - mr.x) as usize;
                plane[row..row + ts as usize]
                    .copy_from_slice(&px[(y * ts) as usize..((y + 1) * ts) as usize]);
            }
        }
        let data = if mw * mh == 0 {
            0u16.to_be_bytes().to_vec()
        } else if bpp == 1 {
            rle_channel(&plane, mw, mh, psb)
        } else {
            let wide: Vec<u8> = plane
                .iter()
                .flat_map(|&v| (v as u16 * 257).to_be_bytes())
                .collect();
            rle_channel(&wide, mw * 2, mh, psb)
        };
        channels.push((-2, data));
        mask = Some((mr, m.default, m.enabled));
    }
    Encoded {
        rect,
        channels,
        mask,
    }
}

fn write_rect(o: &mut Out, r: Rect) {
    o.i32(r.y as i32);
    o.i32(r.x as i32);
    o.i32((r.y + r.h as i64) as i32);
    o.i32((r.x + r.w as i64) as i32);
}

/// One layer record. `divider`: section divider type (groups).
fn write_record(
    o: &mut Out,
    l: &WLayer,
    e: &Encoded,
    divider: Option<u32>,
    light: crate::psd_fx::GlobalLight,
    size: (u32, u32),
) {
    write_rect(o, e.rect);
    o.u16(e.channels.len() as u16);
    for (id, data) in &e.channels {
        o.i16(*id);
        o.len(data.len(), true);
    }
    o.b.extend(b"8BIM");
    o.b.extend(if l.pass_through {
        b"pass"
    } else {
        blend_key(l.blend)
    });
    o.u8((l.opacity.clamp(0.0, 1.0) * 255.0).round() as u8);
    o.u8(l.clip as u8); // clipping: 0 = base, 1 = clipped
    o.u8(if l.visible { 0 } else { 2 } | 8); // bit 3: bit 4 is meaningful
    o.u8(0);
    let extra = o.open(false);
    // Mask.
    match e.mask {
        Some((r, default, enabled)) => {
            o.u32(20);
            write_rect(o, r);
            o.u8(default);
            o.u8(if enabled { 0 } else { 2 });
            o.u16(0);
        }
        None => o.u32(0),
    }
    o.u32(0); // blending ranges
    // Pascal name (MacRoman-ish: ASCII, '?' otherwise), padded to 4.
    let ascii: Vec<u8> = l
        .name
        .chars()
        .map(|c| if c.is_ascii() { c as u8 } else { b'?' })
        .take(255)
        .collect();
    let start = o.b.len();
    o.u8(ascii.len() as u8);
    o.b.extend(&ascii);
    while !(o.b.len() - start).is_multiple_of(4) {
        o.u8(0);
    }
    // Unicode name.
    o.b.extend(b"8BIMluni");
    let units: Vec<u16> = l.name.encode_utf16().collect();
    let blk = o.open(false);
    o.u32(units.len() as u32);
    for u in units {
        o.u16(u);
    }
    o.close(blk, 4);
    if let Some(a) = &l.adjustment {
        for (key, data) in crate::psd_layers::adjustment_blocks(a) {
            o.b.extend(b"8BIM");
            o.b.extend(key);
            let blk = o.open(false);
            o.b.extend(&data);
            o.close(blk, 2);
        }
    }
    if let Some(sh) = &l.shape {
        let block = |o: &mut Out, key: &[u8; 4], data: &[u8]| {
            o.b.extend(b"8BIM");
            o.b.extend(key);
            let blk = o.open(false);
            o.b.extend(data);
            o.close(blk, 4);
        };
        let desc = |d: &crate::psd_data::Desc| {
            let mut v = 16u32.to_be_bytes().to_vec();
            crate::psd_data::w_desc(&mut v, d);
            v
        };
        let fill = sh.fill.unwrap_or([0.0, 0.0, 0.0, 1.0]);
        match &sh.paint {
            Some(p) => {
                let (key, d) = crate::psd_fx::write_paint(p, &|p| crate::psd_patterns::pattern_id(p));
                block(o, &key, &desc(&d));
            }
            None => block(o, b"SoCo", &desc(&crate::psd_shape::write_solid(fill))),
        }
        block(o, b"vsms", &crate::psd_shape::write_vector_mask(&sh.subpaths, size));
        block(o, b"vstk", &desc(&crate::psd_shape::write_stroke(sh.stroke.as_ref(), sh.fill.is_some())));
    }
    if let Some((t, height)) = &l.text {
        o.b.extend(b"8BIMTySh");
        let blk = o.open(false);
        o.b.extend(crate::psd_layers::write_text(t, *height));
        o.close(blk, 4);
    }
    if l.fill_opacity < 1.0 {
        o.b.extend(b"8BIMiOpa");
        let blk = o.open(false);
        o.u8((l.fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8);
        o.close(blk, 4);
    }
    if let Some(fx) = crate::psd_fx::write(&l.effects, light) {
        o.b.extend(b"8BIMlfx2");
        let blk = o.open(false);
        o.u32(0); // effects version
        o.u32(16); // descriptor version
        crate::psd_data::w_desc(&mut o.b, &fx);
        o.close(blk, 4);
    }
    if let Some(t) = divider {
        o.b.extend(b"8BIMlsct");
        let blk = o.open(false);
        o.u32(t);
        if t != 3 {
            o.b.extend(b"8BIM");
            o.b.extend(if l.pass_through {
                b"pass"
            } else {
                blend_key(l.blend)
            });
        }
        o.close(blk, 4);
    }
    o.close(extra, 1);
}

/// Writes a layered Photoshop file (PSB if the canvas is larger than PSD
/// allows). `composite`: the flattened image (premultiplied linear tiles).
pub fn write(
    path: &std::path::Path,
    (width, height): (u32, u32),
    nodes: &[WNode],
    composite: &[(mosaix_core::TileCoord, Vec<half::f16>)],
    opts: &crate::export::ExportOptions,
    extras: &Extras,
) -> Result<()> {
    use rayon::prelude::*;
    let plane_len = width as usize * height as usize;
    let channels: Vec<&(String, Vec<u8>)> = extras.channels.iter().filter(|(_, p)| p.len() == plane_len).collect();
    let psb = width > 30000
        || height > 30000
        || path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("psb"));
    let depth: u16 = if opts.sixteen_bit { 16 } else { 8 };
    let convert = opts.space.converter()?;
    // Flatten the tree into records, bottom → top (groups: divider, children, folder).
    enum Rec<'a> {
        Layer(&'a WLayer),
        End,
        Folder(&'a WLayer),
    }
    fn walk<'a>(nodes: &'a [WNode], out: &mut Vec<Rec<'a>>) {
        for n in nodes {
            match n {
                WNode::Layer(l) => out.push(Rec::Layer(l)),
                WNode::Group { layer, children } => {
                    out.push(Rec::End);
                    walk(children, out);
                    out.push(Rec::Folder(layer));
                }
            }
        }
    }
    let mut recs = Vec::new();
    walk(nodes, &mut recs);
    // The global light: from the first effect that uses it.
    let light = {
        let mut l = crate::psd_fx::GlobalLight::default();
        let found = recs.iter().find_map(|r| match r {
            Rec::Layer(w) | Rec::Folder(w) => w.effects.iter().find(|e| e.global_light).cloned(),
            Rec::End => None,
        });
        if let Some(e) = found {
            if let Some(a) = e.kind.angle() {
                l.angle = a;
            }
            if let mosaix_core::style::EffectKind::Bevel { altitude, .. } = e.kind {
                l.altitude = altitude;
            }
        }
        l
    };
    let end_marker = WLayer {
        name: "</Layer group>".into(),
        opacity: 1.0,
        blend: BlendMode::Normal,
        visible: true,
        clip: false,
        pass_through: false,
        adjustment: None,
        effects: Vec::new(),
        fill_opacity: 1.0,
        shape: None,
        text: None,
        tiles: Vec::new(),
        mask: None,
    };
    let encoded: Vec<Encoded> = recs
        .par_iter()
        .map(|r| match r {
            Rec::Layer(l) => encode_layer(l, depth, &convert, psb),
            Rec::End => encode_layer(&end_marker, depth, &convert, psb),
            // Groups have no pixels of their own, only a mask.
            Rec::Folder(l) => encode_layer(
                &WLayer {
                    name: String::new(),
                    opacity: l.opacity,
                    blend: l.blend,
                    visible: l.visible,
                    clip: false,
                    pass_through: l.pass_through,
                    adjustment: None,
                    effects: Vec::new(),
                    fill_opacity: 1.0,
                    shape: None,
                    text: None,
                    tiles: Vec::new(),
                    mask: l.mask.clone(),
                },
                depth,
                &convert,
                psb,
            ),
        })
        .collect();
    let mut o = Out {
        b: Vec::with_capacity(1 << 20),
        psb,
    };
    o.b.extend(b"8BPS");
    o.u16(if psb { 2 } else { 1 });
    o.b.extend([0u8; 6]);
    o.u16(4 + channels.len() as u16);
    o.u32(height);
    o.u32(width);
    o.u16(depth);
    o.u16(3); // RGB
    o.u32(0); // colour mode data
    // Image resources: the ICC profile.
    let res = o.open(false);
    let mut profile = opts.space.profile();
    profile.description = Some(moxcms::ProfileText::PlainString(opts.space.name().into()));
    if let Ok(icc) = profile.encode() {
        o.b.extend(b"8BIM");
        o.u16(1039);
        o.u16(0); // empty name, padded
        o.u32(icc.len() as u32);
        o.b.extend(&icc);
        if icc.len() % 2 == 1 {
            o.u8(0);
        }
    }
    // Guides (1032): version 1, grid spacing (unused), count, then each
    // position in 1/32 px and its direction (0 vertical, 1 horizontal).
    if !extras.guides.is_empty() {
        o.b.extend(b"8BIM");
        o.u16(1032);
        o.u16(0);
        o.u32(16 + 5 * extras.guides.len() as u32);
        o.u32(1);
        o.u32(576);
        o.u32(576);
        o.u32(extras.guides.len() as u32);
        for g in &extras.guides {
            o.i32((g.pos * 32.0).round() as i32);
            o.u8(if g.vertical { 0 } else { 1 });
        }
        if extras.guides.len() % 2 == 1 {
            o.u8(0);
        }
    }
    // Names of the alpha channels: Pascal (1006) and Unicode (1045).
    if !channels.is_empty() {
        let mut pascal = Vec::new();
        let mut unicode = Vec::new();
        for (name, _) in &channels {
            let ascii: Vec<u8> = name.chars().map(|c| if c.is_ascii() { c as u8 } else { b'?' }).take(255).collect();
            pascal.push(ascii.len() as u8);
            pascal.extend(&ascii);
            let units: Vec<u16> = name.encode_utf16().collect();
            unicode.extend((units.len() as u32).to_be_bytes());
            for u in units {
                unicode.extend(u.to_be_bytes());
            }
        }
        for (id, data) in [(1006u16, pascal), (1045, unicode)] {
            o.b.extend(b"8BIM");
            o.u16(id);
            o.u16(0);
            o.u32(data.len() as u32);
            o.b.extend(&data);
            if data.len() % 2 == 1 {
                o.u8(0);
            }
        }
    }
    // The global light (angle 1037, altitude 1049).
    for (id, v) in [(1037u16, light.angle), (1049, light.altitude)] {
        o.b.extend(b"8BIM");
        o.u16(id);
        o.u16(0);
        o.u32(4);
        o.i32(v.round() as i32);
    }
    o.close(res, 1);
    // Layer and mask information.
    let lm = o.open(true);
    let li = o.open(true);
    o.i16(-(recs.len() as i16)); // negative: the merged alpha is transparency
    for (r, e) in recs.iter().zip(&encoded) {
        match r {
            Rec::Layer(l) => write_record(&mut o, l, e, None, light, (width, height)),
            Rec::End => write_record(&mut o, &end_marker, e, Some(3), light, (width, height)),
            Rec::Folder(l) => write_record(&mut o, l, e, Some(1), light, (width, height)),
        }
    }
    for e in &encoded {
        for (_, data) in &e.channels {
            o.b.extend(data);
        }
    }
    o.close(li, 2);
    o.u32(0); // global layer mask info
    // The patterns of pattern overlays.
    let mut pats: Vec<(String, &mosaix_core::pattern::Pattern)> = Vec::new();
    for r in &recs {
        if let Rec::Layer(w) | Rec::Folder(w) = r {
            let fill = match w.shape.as_ref().and_then(|s| s.paint.as_ref()) {
                Some(mosaix_core::vector::Paint::Pattern { pattern, .. }) => Some(pattern),
                _ => None,
            };
            let overlays = w.effects.iter().filter_map(|e| match &e.kind {
                mosaix_core::style::EffectKind::PatternOverlay { pattern, .. } => Some(pattern),
                _ => None,
            });
            for pattern in fill.into_iter().chain(overlays) {
                let id = crate::psd_patterns::pattern_id(pattern);
                if !pats.iter().any(|(i, _)| *i == id) {
                    pats.push((id, pattern));
                }
            }
        }
    }
    if !pats.is_empty() {
        o.b.extend(b"8BIMPatt");
        let blk = o.open(false);
        o.b.extend(crate::psd_patterns::write(&pats));
        o.close(blk, 4);
    }
    o.close(lm, 2);
    // Merged image, RLE: all row counts first, then the rows.
    let merged = WLayer {
        name: String::new(),
        opacity: 1.0,
        blend: BlendMode::Normal,
        visible: true,
        clip: false,
        pass_through: false,
        adjustment: None,
        effects: Vec::new(),
        fill_opacity: 1.0,
        shape: None,
        text: None,
        tiles: composite.to_vec(),
        mask: None,
    };
    let mut full = encode_full(&merged, (width, height), depth, &convert);
    // Alpha channels follow the colour and transparency planes.
    for (_, plane) in &channels {
        full.push(if depth == 16 { plane.iter().flat_map(|&v| (v as u16 * 257).to_be_bytes()).collect() } else { plane.clone() });
    }
    let bpp = depth as usize / 8;
    let row = width as usize * bpp;
    let rows: Vec<Vec<u8>> = full
        .par_iter()
        .flat_map_iter(|plane| {
            (0..height as usize).map(move |r| {
                let mut v = Vec::new();
                packbits_encode(&plane[r * row..(r + 1) * row], &mut v);
                v
            })
        })
        .collect();
    o.u16(1);
    for r in &rows {
        if psb {
            o.u32(r.len() as u32);
        } else {
            o.u16(r.len() as u16);
        }
    }
    for r in rows {
        o.b.extend(r);
    }
    std::fs::write(path, &o.b).with_context(|| path.display().to_string())?;
    Ok(())
}

/// The whole canvas as four planes (R, G, B, A), big-endian samples.
fn encode_full(
    l: &WLayer,
    (w, h): (u32, u32),
    depth: u16,
    convert: &crate::export::Converter,
) -> Vec<Vec<u8>> {
    use mosaix_core::TILE_SIZE;
    let ts = TILE_SIZE as i64;
    let bpp = depth as usize / 8;
    let (w, h) = (w as usize, h as usize);
    let mut planes = vec![vec![0u8; w * h * bpp]; 4];
    for (c, px) in &l.tiles {
        let (ox, oy) = c.origin();
        let n = (ts * ts) as usize;
        let mut rgb = vec![0.0f32; n * 3];
        let mut a = vec![0.0f32; n];
        for i in 0..n {
            a[i] = px[i * 4 + 3].to_f32().clamp(0.0, 1.0);
            if a[i] > 0.0 {
                for k in 0..3 {
                    rgb[i * 3 + k] = px[i * 4 + k].to_f32() / a[i];
                }
            }
        }
        for row in rgb.chunks_mut(ts as usize * 3) {
            convert(row);
        }
        for y in 0..ts {
            let dy = oy + y;
            if dy < 0 || dy >= h as i64 {
                continue;
            }
            for x in 0..ts {
                let dx = ox + x;
                if dx < 0 || dx >= w as i64 {
                    continue;
                }
                let i = (y * ts + x) as usize;
                let d = (dy as usize * w + dx as usize) * bpp;
                let vals = [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], a[i]];
                for (k, plane) in planes.iter_mut().enumerate() {
                    let v = vals[k].clamp(0.0, 1.0);
                    if bpp == 1 {
                        plane[d] = (v * 255.0).round() as u8;
                    } else {
                        plane[d..d + 2]
                            .copy_from_slice(&((v * 65535.0).round() as u16).to_be_bytes());
                    }
                }
            }
        }
    }
    planes
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
