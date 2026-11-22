//! Photoshop adjustment layers and text layers ↔ mosaix.
//!
//! Adjustments: Levels (`levl`), Curves (`curv`), Brightness/Contrast
//! (`brit` + `CgEd`), Hue/Saturation (`hue2`, master and colorize),
//! Exposure (`expA`), Vibrance (`vibA`), Black & White (`blwh`), Invert
//! (`nvrt`), Posterize (`post`), Threshold (`thrs`), Color Balance
//! (`blnc`), Channel Mixer (`mixr`), Gradient Map (`grdm`). Text: the `TySh`
//! block (text, first run's font, size, colour, alignment, leading,
//! tracking, transform, paragraph width).

use anyhow::{Context, Result, ensure};
use mosaix_core::Adjustment;
use mosaix_core::adjust::Levels;
use mosaix_core::vector::TextAlign;

use crate::psd_data::{Desc, Ps, Value, engine_data, read_desc, w_desc};

struct B<'a> {
    b: &'a [u8],
    p: usize,
}

impl B<'_> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        let v = u16::from_be_bytes(self.b.get(self.p..self.p + 2)?.try_into().ok()?);
        self.p += 2;
        Some(v)
    }
    fn i16(&mut self) -> Option<i16> {
        self.u16().map(|v| v as i16)
    }
    fn u32(&mut self) -> Option<u32> {
        let v = u32::from_be_bytes(self.b.get(self.p..self.p + 4)?.try_into().ok()?);
        self.p += 4;
        Some(v)
    }
    fn i32(&mut self) -> Option<i32> {
        self.u32().map(|v| v as i32)
    }
    fn f32(&mut self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }
    fn f64(&mut self) -> Option<f64> {
        let v = f64::from_be_bytes(self.b.get(self.p..self.p + 8)?.try_into().ok()?);
        self.p += 8;
        Some(v)
    }
    fn rest(&self) -> &[u8] {
        &self.b[self.p.min(self.b.len())..]
    }
}

/// A descriptor block: 4-byte version, then the descriptor.
fn versioned_desc(b: &[u8]) -> Option<Desc> {
    read_desc(b.get(4..)?).ok()
}

pub(crate) fn read_adjustment(key: &[u8], data: &[u8]) -> Option<Adjustment> {
    let mut b = B { b: data, p: 0 };
    Some(match key {
        b"nvrt" => Adjustment::Invert,
        b"blnc" => {
            let mut t = || -> Option<[f32; 3]> { Some([b.i16()? as f32, b.i16()? as f32, b.i16()? as f32]) };
            let (shadows, midtones, highlights) = (t()?, t()?, t()?);
            Adjustment::ColorBalance { shadows, midtones, highlights, preserve_luminosity: b.u8().unwrap_or(1) != 0 }
        }
        b"mixr" => {
            b.u16()?;
            let monochrome = b.u16()? != 0;
            let mut row = || -> Option<[f32; 4]> {
                let v: Vec<f32> = (0..5).map(|_| b.i16().map(|x| x as f32)).collect::<Option<_>>()?;
                Some([v[0], v[1], v[2], v[4]])
            };
            let (red, green, blue) = (row()?, row()?, row()?);
            let gray = row().unwrap_or([40.0, 40.0, 20.0, 0.0]);
            Adjustment::ChannelMixer { red, green, blue, gray, monochrome }
        }
        b"grdm" => read_gradient_map(&mut b)?,
        // Photo filter: version (2: colour space + 4 × 16 bit; 3: XYZ),
        // density %, preserve luminosity.
        b"phfl" => {
            let version = b.u16()?;
            let color = if version == 2 {
                let space = b.u16()?;
                let c: Vec<f32> = (0..4).map(|_| b.u16().map(|v| v as f32 / 65535.0)).collect::<Option<_>>()?;
                if space == 0 { [c[0], c[1], c[2]].map(mosaix_core::color::srgb_to_linear) } else { [0.7, 0.25, 0.0] }
            } else {
                for _ in 0..3 {
                    b.i32()?;
                }
                [0.7, 0.25, 0.0]
            };
            let density = b.i32()? as f32;
            let preserve_luminosity = b.u8().unwrap_or(1) != 0;
            Adjustment::PhotoFilter { color, density, preserve_luminosity }
        }
        // Selective color: version, method (0 relative, 1 absolute), ten
        // records of c, m, y, k (the first unused), percent.
        b"selc" => {
            b.u16()?;
            let relative = b.u16()? == 0;
            for _ in 0..4 {
                b.i16()?;
            }
            let mut ranges = [[0.0f32; 4]; 9];
            for r in ranges.iter_mut() {
                for v in r.iter_mut() {
                    *v = b.i16()? as f32;
                }
            }
            Adjustment::SelectiveColor { ranges, relative }
        }
        b"post" => Adjustment::Posterize {
            levels: b.u16()?.clamp(2, 255) as f32,
        },
        b"thrs" => Adjustment::Threshold {
            level: b.u16()?.clamp(1, 255) as f32,
        },
        b"brit" => {
            let brightness = b.i16()? as f32;
            let contrast = b.i16()? as f32;
            // Without a modern ("CgEd") block, the values are legacy ones.
            Adjustment::BrightnessContrast {
                brightness,
                contrast,
                legacy: true,
            }
        }
        b"expA" => {
            b.u16()?;
            let exposure = b.f32()?;
            let offset = b.f32()?;
            let gamma = b.f32()?;
            Adjustment::Exposure {
                exposure,
                offset,
                gamma,
            }
        }
        b"levl" => {
            b.u16()?;
            let mut rec = || -> Option<Levels> {
                let (inf, inc, outf, outc, g) = (b.u16()?, b.u16()?, b.u16()?, b.u16()?, b.u16()?);
                Some(Levels {
                    in_black: inf as f32 / 255.0,
                    in_white: inc as f32 / 255.0,
                    out_black: outf as f32 / 255.0,
                    out_white: outc as f32 / 255.0,
                    gamma: (g as f32 / 100.0).clamp(0.1, 9.99),
                })
            };
            let master = rec()?;
            let red = rec()?;
            let green = rec()?;
            let blue = rec()?;
            Adjustment::Levels {
                master,
                red,
                green,
                blue,
            }
        }
        b"curv" => {
            b.u8()?;
            let _version = b.u16()?;
            let mask = b.u32()?;
            let ident = || vec![[0.0, 0.0], [1.0, 1.0]];
            let mut curves = [ident(), ident(), ident(), ident()];
            for (i, c) in curves.iter_mut().enumerate() {
                if mask & (1 << i) == 0 {
                    continue;
                }
                let n = b.u16()? as usize;
                let mut pts = Vec::with_capacity(n);
                for _ in 0..n {
                    let out = b.u16()? as f32 / 255.0;
                    let inp = b.u16()? as f32 / 255.0;
                    pts.push([inp, out]);
                }
                if pts.len() >= 2 {
                    *c = pts;
                }
            }
            let [master, red, green, blue] = curves;
            Adjustment::Curves {
                master,
                red,
                green,
                blue,
            }
        }
        b"hue2" | b"hue " => {
            b.u16()?;
            let colorize = b.u8()? != 0;
            b.u8()?;
            let (ch, cs, cl) = (b.i16()?, b.i16()?, b.i16()?);
            let (mh, ms, ml) = (b.i16()?, b.i16()?, b.i16()?);
            // Six colour ranges (Reds, Yellows …): four hue points and the edits.
            let mut ranges = Vec::with_capacity(6);
            for _ in 0..6 {
                let mut r = [0.0f32; 4];
                for v in &mut r {
                    *v = b.i16()? as f32;
                }
                let (h, s, l) = (b.i16()?, b.i16()?, b.i16()?);
                ranges.push(mosaix_core::adjust::HueRange { range: r, hue: h as f32, saturation: s as f32, lightness: l as f32 });
            }
            if ranges.iter().all(|r| r.is_identity()) {
                ranges.clear();
            }
            if colorize {
                // mosaix's colorize saturation: 0 → 25 %, 100 → 100 %.
                let sat = ((cs as f32 / 100.0 - 0.25) / 0.75 * 100.0).clamp(-100.0, 100.0);
                Adjustment::HueSaturation {
                    hue: ch as f32,
                    saturation: sat,
                    lightness: cl as f32,
                    colorize: true,
                    ranges,
                }
            } else {
                Adjustment::HueSaturation {
                    hue: mh as f32,
                    saturation: ms as f32,
                    lightness: ml as f32,
                    colorize: false,
                    ranges,
                }
            }
        }
        b"vibA" => {
            let d = versioned_desc(data)?;
            Adjustment::Vibrance {
                vibrance: d.num("vibrance").unwrap_or(0.0) as f32,
                saturation: d.num("Strt").unwrap_or(0.0) as f32,
            }
        }
        b"blwh" => {
            let d = versioned_desc(data)?;
            let w = |k: &str, def: f32| d.num(k).map_or(def, |v| v as f32);
            Adjustment::BlackWhite {
                weights: [
                    w("Rd  ", 40.0),
                    w("Yllw", 60.0),
                    w("Grn ", 40.0),
                    w("Cyn ", 60.0),
                    w("Bl  ", 20.0),
                    w("Mgnt", 80.0),
                ],
            }
        }
        _ => return None,
    })
}

fn read_gradient_map(b: &mut B) -> Option<Adjustment> {
    use mosaix_core::gradient::{AlphaStop, ColorStop, Gradient};
    b.u16()?;
    let reverse = b.u8()? != 0;
    let _dither = b.u8()?;
    let name_len = b.u32()? as usize;
    b.p += name_len * 2;
    let n = b.u16()? as usize;
    let mut colors = Vec::with_capacity(n);
    for _ in 0..n {
        let at = b.u32()? as f32 / 4096.0;
        let mid = b.u32()? as f32 / 100.0;
        let _mode = b.u16()?;
        let c = [b.u16()?, b.u16()?, b.u16()?, b.u16()?];
        b.u16()?;
        let color = [c[0], c[1], c[2]].map(|v| mosaix_core::color::srgb_to_linear(v as f32 / 65535.0));
        colors.push(ColorStop { at: at.clamp(0.0, 1.0), color, mid: mid.clamp(0.0, 1.0) });
    }
    let n = b.u16()? as usize;
    let mut alphas = Vec::with_capacity(n);
    for _ in 0..n {
        let at = b.u32()? as f32 / 4096.0;
        let mid = b.u32()? as f32 / 100.0;
        let opacity = b.u16()? as f32 / 255.0;
        alphas.push(AlphaStop { at: at.clamp(0.0, 1.0), opacity: opacity.clamp(0.0, 1.0), mid: mid.clamp(0.0, 1.0) });
    }
    // Interpolation (smoothness, 0…4096) follows the expansion count.
    let smoothness = b.u16().and_then(|_| b.u16()).map_or(0.0, |v| v as f32 / 4096.0);
    if colors.is_empty() {
        return None;
    }
    if alphas.iter().all(|a| a.opacity >= 1.0) {
        alphas.clear();
    }
    Some(Adjustment::GradientMap { gradient: Gradient { colors, alphas, smoothness }, reverse })
}

fn write_gradient_map(o: &mut Vec<u8>, g: &mosaix_core::gradient::Gradient, reverse: bool) {
    let be16 = |o: &mut Vec<u8>, v: u16| o.extend(v.to_be_bytes());
    let be32 = |o: &mut Vec<u8>, v: u32| o.extend(v.to_be_bytes());
    let g = g.clone().sorted();
    be16(o, 1);
    o.push(reverse as u8);
    o.push(0);
    // Name (Unicode).
    let name: Vec<u16> = "Custom".encode_utf16().collect();
    be32(o, name.len() as u32);
    for c in &name {
        be16(o, *c);
    }
    be16(o, g.colors.len() as u16);
    for s in &g.colors {
        be32(o, (s.at.clamp(0.0, 1.0) * 4096.0).round() as u32);
        be32(o, (s.mid.clamp(0.0, 1.0) * 100.0).round() as u32);
        be16(o, 0); // RGB
        for c in s.color {
            be16(o, (mosaix_core::color::linear_to_srgb(c).clamp(0.0, 1.0) * 65535.0).round() as u16);
        }
        be16(o, 0);
        be16(o, 0);
    }
    let alphas = if g.alphas.is_empty() {
        vec![(0.0, 1.0, 0.5), (1.0, 1.0, 0.5)]
    } else {
        g.alphas.iter().map(|a| (a.at, a.opacity, a.mid)).collect()
    };
    be16(o, alphas.len() as u16);
    for (at, op, mid) in alphas {
        be32(o, (at.clamp(0.0, 1.0) * 4096.0).round() as u32);
        be32(o, (mid.clamp(0.0, 1.0) * 100.0).round() as u32);
        be16(o, (op.clamp(0.0, 1.0) * 255.0).round() as u16);
    }
    be16(o, 2); // expansion count
    be16(o, (g.smoothness.clamp(0.0, 1.0) * 4096.0).round() as u16);
    be16(o, 32); // length
    be16(o, 0); // mode
    be32(o, 0); // random seed
    be16(o, 0); // showing transparency
    be16(o, 0); // using vector colour
    be32(o, 2048); // roughness
    be16(o, 3); // colour model (RGB)
    for _ in 0..4 {
        be16(o, 0);
    }
    for _ in 0..4 {
        be16(o, 0x8000);
    }
    be16(o, 0);
}

/// Tagged blocks for an adjustment layer: (key, data), in writing order.
pub(crate) fn adjustment_blocks(a: &Adjustment) -> Vec<([u8; 4], Vec<u8>)> {
    let mut o = Vec::new();
    let be16 = |o: &mut Vec<u8>, v: u16| o.extend(v.to_be_bytes());
    let mut out = Vec::new();
    match a {
        Adjustment::SelectiveColor { ranges, relative } => {
            be16(&mut o, 1);
            be16(&mut o, if *relative { 0 } else { 1 });
            for _ in 0..4 {
                be16(&mut o, 0);
            }
            for r in ranges {
                for v in r {
                    o.extend((v.round().clamp(-100.0, 100.0) as i16).to_be_bytes());
                }
            }
            out.push((*b"selc", o));
        }
        Adjustment::PhotoFilter { color, density, preserve_luminosity } => {
            be16(&mut o, 2);
            be16(&mut o, 0); // RGB
            for v in color.map(|c| mosaix_core::color::linear_to_srgb(c.clamp(0.0, 1.0))) {
                be16(&mut o, (v * 65535.0).round() as u16);
            }
            be16(&mut o, 0);
            o.extend((density.round().clamp(0.0, 100.0) as i32).to_be_bytes());
            o.push(*preserve_luminosity as u8);
            out.push((*b"phfl", o));
        }
        // No Photoshop block written (the layer has no effect there).
        Adjustment::ColorLookup { .. } => {}
        Adjustment::Invert => out.push((*b"nvrt", o)),
        Adjustment::Posterize { levels } => {
            be16(&mut o, levels.round().clamp(2.0, 255.0) as u16);
            be16(&mut o, 0);
            out.push((*b"post", o));
        }
        Adjustment::ColorBalance {
            shadows,
            midtones,
            highlights,
            preserve_luminosity,
        } => {
            for v in shadows.iter().chain(midtones).chain(highlights) {
                o.extend((v.round().clamp(-100.0, 100.0) as i16).to_be_bytes());
            }
            o.push(*preserve_luminosity as u8);
            o.push(0);
            out.push((*b"blnc", o));
        }
        Adjustment::ChannelMixer {
            red,
            green,
            blue,
            gray,
            monochrome,
        } => {
            be16(&mut o, 1);
            be16(&mut o, *monochrome as u16);
            for r in [red, green, blue, gray] {
                for v in [r[0], r[1], r[2], 0.0, r[3]] {
                    o.extend((v.round().clamp(-200.0, 200.0) as i16).to_be_bytes());
                }
            }
            out.push((*b"mixr", o));
        }
        Adjustment::GradientMap { gradient, reverse } => {
            write_gradient_map(&mut o, gradient, *reverse);
            out.push((*b"grdm", o));
        }
        Adjustment::Threshold { level } => {
            be16(&mut o, level.round().clamp(1.0, 255.0) as u16);
            be16(&mut o, 0);
            out.push((*b"thrs", o));
        }
        Adjustment::BrightnessContrast {
            brightness,
            contrast,
            legacy,
        } => {
            o.extend((brightness.round() as i16).to_be_bytes());
            o.extend((contrast.round() as i16).to_be_bytes());
            be16(&mut o, 127);
            o.extend([0, 0]);
            out.push((*b"brit", o));
            // The modern (non-legacy) algorithm, like mosaix's.
            // Descriptor version, then the descriptor (as Photoshop writes it).
            let mut c = Vec::new();
            c.extend(16u32.to_be_bytes());
            w_desc(
                &mut c,
                &Desc {
                    class: "null".into(),
                    items: vec![
                        ("Vrsn".into(), Value::Long(1)),
                        ("Brgh".into(), Value::Long(brightness.round() as i32)),
                        ("Cntr".into(), Value::Long(contrast.round() as i32)),
                        ("means".into(), Value::Long(127)),
                        ("Lab ".into(), Value::Bool(false)),
                        ("useLegacy".into(), Value::Bool(*legacy)),
                        ("Auto".into(), Value::Bool(false)),
                    ],
                },
            );
            out.push((*b"CgEd", c));
        }
        Adjustment::Exposure {
            exposure,
            offset,
            gamma,
        } => {
            be16(&mut o, 1);
            for v in [exposure, offset, gamma] {
                o.extend(v.to_bits().to_be_bytes());
            }
            out.push((*b"expA", o));
        }
        Adjustment::Levels {
            master,
            red,
            green,
            blue,
        } => {
            be16(&mut o, 2);
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u16;
            let neutral = Levels::default();
            for l in [master, red, green, blue]
                .into_iter()
                .chain(std::iter::repeat_n(&neutral, 25))
            {
                for v in [
                    q(l.in_black),
                    q(l.in_white),
                    q(l.out_black),
                    q(l.out_white),
                    (l.gamma * 100.0).round().clamp(10.0, 999.0) as u16,
                ] {
                    be16(&mut o, v);
                }
            }
            out.push((*b"levl", o));
        }
        Adjustment::Curves {
            master,
            red,
            green,
            blue,
        } => {
            o.push(0);
            be16(&mut o, 1);
            o.extend(0b1111u32.to_be_bytes());
            for c in [master, red, green, blue] {
                be16(&mut o, c.len() as u16);
                for p in c {
                    be16(&mut o, (p[1].clamp(0.0, 1.0) * 255.0).round() as u16);
                    be16(&mut o, (p[0].clamp(0.0, 1.0) * 255.0).round() as u16);
                }
            }
            out.push((*b"curv", o));
        }
        Adjustment::HueSaturation {
            hue,
            saturation,
            lightness,
            colorize,
            ranges,
        } => {
            be16(&mut o, 2);
            o.push(*colorize as u8);
            o.push(0);
            let i = |v: f32| (v.round() as i16).to_be_bytes();
            if *colorize {
                let sat = ((saturation / 100.0) * 0.75 + 0.25) * 100.0;
                o.extend(i(hue.rem_euclid(360.0)));
                o.extend(i(sat));
                o.extend(i(*lightness));
                o.extend([0u8; 6]);
            } else {
                o.extend(i(0.0));
                o.extend(i(25.0));
                o.extend(i(0.0));
                o.extend(i(*hue));
                o.extend(i(*saturation));
                o.extend(i(*lightness));
            }
            // Six colour ranges (Photoshop's defaults where none are set).
            let all = if ranges.is_empty() { mosaix_core::adjust::HueRange::defaults() } else { ranges.clone() };
            for r in all.iter().chain(mosaix_core::adjust::HueRange::defaults().iter().skip(all.len())).take(6) {
                for v in r.range {
                    o.extend((v.round() as i16).to_be_bytes());
                }
                o.extend(i(r.hue));
                o.extend(i(r.saturation));
                o.extend(i(r.lightness));
            }
            out.push((*b"hue2", o));
        }
        Adjustment::Vibrance {
            vibrance,
            saturation,
        } => {
            o.extend(16u32.to_be_bytes());
            w_desc(
                &mut o,
                &Desc {
                    class: "null".into(),
                    items: vec![
                        ("vibrance".into(), Value::Long(vibrance.round() as i32)),
                        ("Strt".into(), Value::Long(saturation.round() as i32)),
                    ],
                },
            );
            out.push((*b"vibA", o));
        }
        Adjustment::BlackWhite { weights } => {
            o.extend(16u32.to_be_bytes());
            let keys = ["Rd  ", "Yllw", "Grn ", "Cyn ", "Bl  ", "Mgnt"];
            let mut items: Vec<(String, Value)> = keys
                .iter()
                .zip(weights)
                .map(|(k, w)| (k.to_string(), Value::Long(w.round() as i32)))
                .collect();
            items.push(("useTint".into(), Value::Bool(false)));
            w_desc(
                &mut o,
                &Desc {
                    class: "BanW".into(),
                    items,
                },
            );
            out.push((*b"blwh", o));
        }
    }
    out
}

// --- Text -----------------------------------------------------------------

pub(crate) fn read_text(data: &[u8]) -> Result<crate::psd::Text> {
    let mut b = B { b: data, p: 0 };
    let bad = || anyhow::anyhow!("text block truncated");
    ensure!(b.u16().ok_or_else(bad)? == 1, "unknown text version");
    let mut t = [0.0f64; 6];
    for v in &mut t {
        *v = b.f64().ok_or_else(bad)?;
    }
    b.u16().ok_or_else(bad)?; // text version (50)
    b.u32().ok_or_else(bad)?; // descriptor version (16)
    let d = read_desc(b.rest())?;
    let mut text = d.text("Txt ").unwrap_or("").to_string();
    let engine = d
        .raw("EngineData")
        .map(engine_data)
        .transpose()
        .context("EngineData")?;
    let get = |path: &str| engine.as_ref().and_then(|e| e.at(path));
    if text.is_empty()
        && let Some(s) = get("EngineDict/Editor/Text").and_then(Ps::str)
    {
        text = s.to_string();
    }
    // Photoshop ends lines with \r and the text with one more.
    let text = text.trim_end_matches(['\r', '\n']).replace('\r', "\n");
    let style = get("EngineDict/StyleRun/RunArray/0/StyleSheet/StyleSheetData");
    let sget = |k: &str| style.and_then(|s| s.get(k));
    let fonts = get("ResourceDict/FontSet")
        .and_then(Ps::array)
        .unwrap_or(&[]);
    let font_idx = sget("Font").and_then(Ps::num).unwrap_or(0.0) as usize;
    let font = fonts
        .get(font_idx)
        .and_then(|f| f.get("Name"))
        .and_then(Ps::str)
        .unwrap_or("ArialMT")
        .to_string();
    let size = sget("FontSize").and_then(Ps::num).unwrap_or(12.0) as f32;
    let color = sget("FillColor")
        .and_then(|c| c.get("Values"))
        .and_then(Ps::array)
        .filter(|v| v.len() == 4)
        .map(|v| {
            let n = |i: usize| v[i].num().unwrap_or(0.0) as f32;
            [n(1), n(2), n(3), n(0)]
        })
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let auto_leading = sget("AutoLeading").is_none_or(|v| *v != Ps::Bool(false));
    let leading = sget("Leading").and_then(Ps::num).unwrap_or(0.0) as f32;
    let line_height = if auto_leading || leading <= 0.0 {
        1.2
    } else {
        leading / size.max(1.0)
    };
    // Tracking is in 1/1000 em.
    let tracking = sget("Tracking").and_then(Ps::num).unwrap_or(0.0) as f32 * size / 1000.0;
    let faux_bold = sget("FauxBold") == Some(&Ps::Bool(true));
    let faux_italic = sget("FauxItalic") == Some(&Ps::Bool(true));
    let align =
        match get("EngineDict/ParagraphRun/RunArray/0/ParagraphSheet/Properties/Justification")
            .and_then(Ps::num)
            .unwrap_or(0.0) as i32
        {
            1 => TextAlign::Right,
            2 => TextAlign::Center,
            3..=6 => TextAlign::Justify,
            _ => TextAlign::Left,
        };
    // Paragraph (box) text: width from the box bounds.
    let shape = get("EngineDict/Rendered/Shapes/Children/0/ShapeType")
        .and_then(Ps::num)
        .unwrap_or(0.0);
    let width = if shape == 1.0 {
        get("EngineDict/Rendered/Shapes/Children/0/Cookie/Photoshop/BoxBounds")
            .and_then(Ps::array)
            .and_then(|b| {
                let n = |i: usize| b.get(i).and_then(Ps::num);
                Some((n(2)? - n(0)?) as f32)
            })
    } else {
        None
    };
    // Photoshop's OpenType switches.
    let flag = |k: &str| sget(k).map(|v| *v == Ps::Bool(true));
    let mut features = std::collections::BTreeMap::new();
    let mut set = |tag: &str, on: Option<bool>, default: bool| {
        if let Some(on) = on.filter(|on| *on != default) {
            features.insert(tag.to_string(), on as u16);
        }
    };
    set("liga", flag("Ligatures"), true);
    set("dlig", flag("DLigatures"), false);
    set("onum", flag("OldStyle"), false);
    set("frac", flag("Fractions"), false);
    set("ordn", flag("Ordinals"), false);
    set("swsh", flag("Swash"), false);
    set("salt", flag("StylisticAlternates"), false);
    set("smcp", sget("FontCaps").and_then(Ps::num).map(|v| v == 1.0), false);
    match sget("FontBaseline").and_then(Ps::num).unwrap_or(0.0) as i32 {
        1 => set("sups", Some(true), false),
        2 => set("subs", Some(true), false),
        _ => {}
    }
    Ok(crate::psd::Text {
        text,
        font,
        size,
        color,
        align,
        line_height,
        tracking,
        faux_bold,
        faux_italic,
        width,
        transform: t,
        features,
    })
}

/// Photoshop's own EngineData of a one-run text layer (from a file it wrote),
/// used as the template for writing text layers.
const ENGINE_TEMPLATE: &[u8] = include_bytes!("psd_engine_template.dat");

/// The "TySh" block for a text layer. `height`: box height of paragraph text.
pub(crate) fn write_text(t: &crate::psd::Text, height: f32) -> Vec<u8> {
    let mut e = engine_data(ENGINE_TEMPLATE).expect("template");
    // Photoshop: lines end with \r, the text with one more.
    let body = t.text.replace("\r\n", "\r").replace('\n', "\r");
    let full = format!("{body}\r");
    let len = |s: &str| s.encode_utf16().count() as i64;
    if let Some(v) = e.at_mut("EngineDict/Editor/Text") {
        *v = Ps::Str(full.clone());
    }
    // One paragraph run per line, with the alignment.
    let just = match t.align {
        TextAlign::Left => 0,
        TextAlign::Right => 1,
        TextAlign::Center => 2,
        TextAlign::Justify => 3,
    };
    let lines: Vec<&str> = body.split('\r').collect();
    if let Some(pr) = e.at_mut("EngineDict/ParagraphRun") {
        let first = pr.at("RunArray/0").cloned().expect("paragraph run");
        let mut runs = Vec::new();
        for _ in &lines {
            let mut r = first.clone();
            if let Some(p) = r.at_mut("ParagraphSheet/Properties") {
                p.set("Justification", Ps::Int(just));
            }
            runs.push(r);
        }
        let mut lens: Vec<Ps> = lines.iter().map(|l| Ps::Int(len(l) + 1)).collect();
        if lens.is_empty() {
            lens.push(Ps::Int(1));
        }
        pr.set("RunArray", Ps::Array(runs));
        pr.set("RunLengthArray", Ps::Array(lens));
    }
    if let Some(sr) = e.at_mut("EngineDict/StyleRun") {
        sr.set("RunLengthArray", Ps::Array(vec![Ps::Int(len(&full))]));
    }
    if let Some(st) = e.at_mut("EngineDict/StyleRun/RunArray/0/StyleSheet/StyleSheetData") {
        st.set("Font", Ps::Int(0));
        st.set("FontSize", Ps::Num(t.size as f64));
        st.set("FauxBold", Ps::Bool(t.faux_bold));
        st.set("FauxItalic", Ps::Bool(t.faux_italic));
        let auto = (t.line_height - 1.2).abs() < 1e-3;
        st.set("AutoLeading", Ps::Bool(auto));
        st.set("Leading", Ps::Num(if auto { 0.01 } else { (t.size * t.line_height) as f64 }));
        st.set("Tracking", Ps::Int((t.tracking / t.size.max(1.0) * 1000.0).round() as i64));
        let c = t.color;
        st.set(
            "FillColor",
            Ps::Dict(vec![
                ("Type".into(), Ps::Int(1)),
                ("Values".into(), Ps::Array([c[3], c[0], c[1], c[2]].iter().map(|v| Ps::Num(v.clamp(0.0, 1.0) as f64)).collect())),
            ]),
        );
        let f = |k: &str| t.features.get(k).map(|v| *v != 0);
        for (key, tag, default) in [
            ("Ligatures", "liga", true),
            ("DLigatures", "dlig", false),
            ("OldStyle", "onum", false),
            ("Fractions", "frac", false),
            ("Ordinals", "ordn", false),
            ("Swash", "swsh", false),
            ("StylisticAlternates", "salt", false),
        ] {
            let on = f(tag).unwrap_or(default);
            if on != default || st.get(key).is_some() {
                st.set(key, Ps::Bool(on));
            }
        }
        st.set("FontCaps", Ps::Int(f("smcp").unwrap_or(false) as i64));
        let base = if f("sups") == Some(true) { 1 } else if f("subs") == Some(true) { 2 } else { 0 };
        st.set("FontBaseline", Ps::Int(base));
    }
    // The font: first entry of both font sets.
    for set in ["ResourceDict/FontSet/0", "DocumentResources/FontSet/0"] {
        if let Some(fe) = e.at_mut(set) {
            fe.set("Name", Ps::Str(t.font.clone()));
            fe.set("FontType", Ps::Int(1));
        }
    }
    // Point or paragraph (box) text.
    if let (Some(w), Some(ch)) = (t.width, e.at_mut("EngineDict/Rendered/Shapes/Children/0")) {
        ch.set("ShapeType", Ps::Int(1));
        if let Some(ps) = ch.at_mut("Cookie/Photoshop") {
            ps.set("ShapeType", Ps::Int(1));
            if let Ps::Dict(items) = ps {
                items.retain(|(k, _)| k != "PointBase");
            }
            ps.set("BoxBounds", Ps::Array([0.0, 0.0, w as f64, height as f64].map(Ps::Num).to_vec()));
            if let Some(b) = ps.get_mut("Base") {
                b.set("ShapeType", Ps::Int(1));
            }
        }
    }
    let engine = crate::psd_data::write_engine_data(&e);
    // TySh: version, transform, text version, descriptor, warp, bounds.
    let mut o = Vec::new();
    o.extend(1u16.to_be_bytes());
    for v in t.transform {
        o.extend(v.to_be_bytes());
    }
    o.extend(50u16.to_be_bytes());
    o.extend(16u32.to_be_bytes());
    let en = |a: &str, b: &str| Value::Enum(a.into(), b.into());
    let txt = Desc {
        class: "TxLr".into(),
        items: vec![
            ("Txt ".into(), Value::Text(body)),
            ("textGridding".into(), en("textGridding", "None")),
            ("Ornt".into(), en("Ornt", "Hrzn")),
            ("AntA".into(), en("Annt", "AnSm")),
            ("TextIndex".into(), Value::Long(0)),
            ("EngineData".into(), Value::Raw(engine)),
        ],
    };
    w_desc(&mut o, &txt);
    o.extend(1u16.to_be_bytes());
    o.extend(16u32.to_be_bytes());
    let warp = Desc {
        class: "warp".into(),
        items: vec![
            ("warpStyle".into(), en("warpStyle", "warpNone")),
            ("warpValue".into(), Value::Double(0.0)),
            ("warpPerspective".into(), Value::Double(0.0)),
            ("warpPerspectiveOther".into(), Value::Double(0.0)),
            ("warpRotate".into(), en("Ornt", "Hrzn")),
        ],
    };
    w_desc(&mut o, &warp);
    o.extend([0u8; 16]);
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn photo_filter_round_trips() {
        let a = Adjustment::PhotoFilter { color: [0.0, 0.2, 1.0], density: 40.0, preserve_luminosity: false };
        let (k, d) = &adjustment_blocks(&a)[0];
        let Some(Adjustment::PhotoFilter { color, density, preserve_luminosity }) = read_adjustment(k, d) else { panic!() };
        assert!((density - 40.0).abs() < 1e-3 && !preserve_luminosity);
        assert!(color.iter().zip([0.0, 0.2, 1.0]).all(|(a, b)| (a - b).abs() < 1e-3), "{color:?}");
    }

    #[test]
    fn gradient_map_roundtrips() {
        use mosaix_core::gradient::{ColorStop, Gradient};
        let g = Gradient {
            colors: vec![
                ColorStop { at: 0.0, color: [0.0, 0.0, 0.2], mid: 0.5 },
                ColorStop { at: 0.25, color: [1.0, 0.2, 0.0], mid: 0.3 },
                ColorStop { at: 1.0, color: [1.0, 1.0, 1.0], mid: 0.5 },
            ],
            alphas: Vec::new(),
            smoothness: 1.0,
        };
        let a = Adjustment::GradientMap { gradient: g.clone(), reverse: true };
        let blocks = adjustment_blocks(&a);
        let back = read_adjustment(&blocks[0].0, &blocks[0].1).unwrap();
        let Adjustment::GradientMap { gradient, reverse } = back else { panic!() };
        assert!(reverse);
        assert_eq!(gradient.colors.len(), 3);
        for (x, y) in gradient.colors.iter().zip(&g.colors) {
            assert!((x.at - y.at).abs() < 1e-3 && (x.mid - y.mid).abs() < 0.01);
            assert!(x.color.iter().zip(y.color).all(|(a, b)| (a - b).abs() < 1e-3), "{:?} {:?}", x.color, y.color);
        }
        assert!((gradient.smoothness - 1.0).abs() < 1e-3);
    }

    #[test]
    fn adjustments_roundtrip_through_blocks() {
        let cases = [
            Adjustment::Invert,
            Adjustment::Posterize { levels: 5.0 },
            Adjustment::Threshold { level: 100.0 },
            Adjustment::Exposure {
                exposure: 1.5,
                offset: -0.1,
                gamma: 1.2,
            },
            Adjustment::Vibrance {
                vibrance: 40.0,
                saturation: -10.0,
            },
            Adjustment::HueSaturation {
                hue: 30.0,
                saturation: -20.0,
                lightness: 10.0,
                colorize: false,
                ranges: {
                    let mut r = mosaix_core::adjust::HueRange::defaults();
                    r[2].saturation = 45.0;
                    r[5].hue = -20.0;
                    r
                },
            },
            Adjustment::BlackWhite {
                weights: [40.0, 60.0, 40.0, 60.0, 20.0, 80.0],
            },
            Adjustment::ColorBalance {
                shadows: [-20.0, 10.0, 30.0],
                midtones: [40.0, -10.0, 0.0],
                highlights: [0.0, 20.0, -50.0],
                preserve_luminosity: false,
            },
            Adjustment::ChannelMixer {
                red: [80.0, 30.0, -10.0, 5.0],
                green: [0.0, 100.0, 0.0, 0.0],
                blue: [20.0, 0.0, 70.0, -10.0],
                gray: [40.0, 40.0, 20.0, 0.0],
                monochrome: true,
            },
            Adjustment::SelectiveColor { ranges: [[10.0, -20.0, 30.0, -5.0]; 9], relative: false },
        ];
        for a in cases {
            let blocks = adjustment_blocks(&a);
            let (k, d) = &blocks[0];
            let back = read_adjustment(k, d).unwrap();
            assert_eq!(back, a, "{}", String::from_utf8_lossy(k));
        }
        // Modern brightness/contrast: the CgEd descriptor carries the values.
        let bc = Adjustment::BrightnessContrast {
            brightness: 42.0,
            contrast: -17.0,
            legacy: false,
        };
        let blocks = adjustment_blocks(&bc);
        let (k, d) = blocks.iter().find(|(k, _)| k == b"CgEd").unwrap();
        assert_eq!(k, b"CgEd");
        let desc = crate::psd_data::read_desc(&d[4..]).unwrap();
        assert_eq!(
            (desc.num("Brgh"), desc.num("Cntr"), desc.bool("useLegacy")),
            (Some(42.0), Some(-17.0), Some(false))
        );
        let lv = Adjustment::Levels {
            master: Levels {
                in_black: 10.0 / 255.0,
                in_white: 240.0 / 255.0,
                gamma: 1.5,
                out_black: 0.0,
                out_white: 1.0,
            },
            red: Levels::default(),
            green: Levels::default(),
            blue: Levels::default(),
        };
        let (k, d) = &adjustment_blocks(&lv)[0];
        assert_eq!(read_adjustment(k, d).unwrap(), lv);
        let cv = Adjustment::Curves {
            master: vec![[0.0, 0.0], [128.0 / 255.0, 160.0 / 255.0], [1.0, 1.0]],
            red: vec![[0.0, 0.0], [1.0, 1.0]],
            green: vec![[0.0, 0.0], [1.0, 1.0]],
            blue: vec![[0.0, 0.0], [1.0, 1.0]],
        };
        let (k, d) = &adjustment_blocks(&cv)[0];
        assert_eq!(read_adjustment(k, d).unwrap(), cv);
    }

    #[test]
    fn text_layer_roundtrips_through_tysh() {
        let t = crate::psd::Text {
            text: "Hello (world)\nSecond line ü".into(),
            font: "LiberationSans-Bold".into(),
            size: 36.0,
            color: [1.0, 0.5, 0.0, 1.0],
            align: TextAlign::Center,
            line_height: 1.5,
            tracking: 3.6,
            faux_bold: false,
            faux_italic: true,
            width: Some(400.0),
            transform: [1.0, 0.0, 0.0, 1.0, 20.0, 60.0],
            features: [("dlig".to_string(), 1u16), ("liga".to_string(), 0)].into_iter().collect(),
        };
        let b = write_text(&t, 200.0);
        let back = read_text(&b).unwrap();
        assert_eq!(back.text, t.text);
        assert_eq!(back.font, t.font);
        assert_eq!((back.size, back.align, back.width, back.transform), (t.size, t.align, t.width, t.transform));
        assert!((back.line_height - 1.5).abs() < 1e-4 && (back.tracking - 3.6).abs() < 1e-3);
        assert!(back.faux_italic && !back.faux_bold);
        assert!((back.color[1] - 0.5).abs() < 1e-6);
        assert_eq!(back.features, t.features);
        // Point text with default leading.
        let p = crate::psd::Text { width: None, line_height: 1.2, align: TextAlign::Left, features: Default::default(), ..t };
        let back = read_text(&write_text(&p, 0.0)).unwrap();
        assert_eq!((back.width, back.line_height, back.align), (None, 1.2, TextAlign::Left));
    }
}
