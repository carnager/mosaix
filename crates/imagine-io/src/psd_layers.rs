//! Photoshop adjustment layers and text layers ↔ imagine.
//!
//! Adjustments: Levels (`levl`), Curves (`curv`), Brightness/Contrast
//! (`brit` + `CgEd`), Hue/Saturation (`hue2`, master and colorize),
//! Exposure (`expA`), Vibrance (`vibA`), Black & White (`blwh`), Invert
//! (`nvrt`), Posterize (`post`), Threshold (`thrs`). Text: the `TySh`
//! block (text, first run's font, size, colour, alignment, leading,
//! tracking, transform, paragraph width).

use anyhow::{Context, Result, ensure};
use imagine_core::Adjustment;
use imagine_core::adjust::Levels;
use imagine_core::vector::TextAlign;

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

thread_local! {
    /// Hue/saturation layers with colour-range edits seen while reading (for a warning).
    pub(crate) static RANGED_HUE_SAT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn read_adjustment(key: &[u8], data: &[u8]) -> Option<Adjustment> {
    let mut b = B { b: data, p: 0 };
    Some(match key {
        b"nvrt" => Adjustment::Invert,
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
            // Per-colour-range edits (Reds, Yellows …) are not supported yet.
            let mut ranged = false;
            for _ in 0..6 {
                for _ in 0..4 {
                    b.i16()?;
                }
                let (h, s, l) = (b.i16()?, b.i16()?, b.i16()?);
                ranged |= h != 0 || s != 0 || l != 0;
            }
            if ranged && !colorize {
                RANGED_HUE_SAT.with(|c| c.set(c.get() + 1));
            }
            if colorize {
                // imagine's colorize saturation: 0 → 25 %, 100 → 100 %.
                let sat = ((cs as f32 / 100.0 - 0.25) / 0.75 * 100.0).clamp(-100.0, 100.0);
                Adjustment::HueSaturation {
                    hue: ch as f32,
                    saturation: sat,
                    lightness: cl as f32,
                    colorize: true,
                }
            } else {
                // Per-colour ranges are not supported: master only.
                Adjustment::HueSaturation {
                    hue: mh as f32,
                    saturation: ms as f32,
                    lightness: ml as f32,
                    colorize: false,
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

/// Tagged blocks for an adjustment layer: (key, data), in writing order.
pub(crate) fn adjustment_blocks(a: &Adjustment) -> Vec<([u8; 4], Vec<u8>)> {
    let mut o = Vec::new();
    let be16 = |o: &mut Vec<u8>, v: u16| o.extend(v.to_be_bytes());
    let mut out = Vec::new();
    match a {
        Adjustment::Invert => out.push((*b"nvrt", o)),
        Adjustment::Posterize { levels } => {
            be16(&mut o, levels.round().clamp(2.0, 255.0) as u16);
            be16(&mut o, 0);
            out.push((*b"post", o));
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
            // The modern (non-legacy) algorithm, like imagine's.
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
            // Six colour ranges (Photoshop's defaults), no changes.
            let ranges: [[i16; 4]; 6] = [
                [315, 345, 15, 45],
                [15, 45, 75, 105],
                [75, 105, 135, 165],
                [135, 165, 195, 225],
                [195, 225, 255, 285],
                [255, 285, 315, 345],
            ];
            for r in ranges {
                for v in r {
                    o.extend(v.to_be_bytes());
                }
                o.extend([0u8; 6]);
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
            },
            Adjustment::BlackWhite {
                weights: [40.0, 60.0, 40.0, 60.0, 20.0, 80.0],
            },
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
}
