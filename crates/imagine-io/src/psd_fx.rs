//! Photoshop layer styles (`lfx2` / `lmfx` descriptors) ↔ imagine effects.
//!
//! Carried over: drop and inner shadow, outer and inner glow, bevel and
//! emboss, colour and gradient overlay, stroke (solid colour). Satin and
//! pattern overlay are reported as not carried over.

use imagine_core::BlendMode;
use imagine_core::color::{linear_to_srgb, srgb_to_linear};
use imagine_core::style::{BevelStyle, Effect, EffectKind, GradientStyle, StrokePosition};

use crate::psd_data::{Desc, Value};

/// The document's light ("Use Global Light"): angle and altitude in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalLight {
    pub angle: f32,
    pub altitude: f32,
}

impl Default for GlobalLight {
    fn default() -> Self {
        Self { angle: 120.0, altitude: 30.0 }
    }
}

/// Blend mode from a descriptor enum ("BlnM"): four-character ids, or the
/// spelled-out names newer Photoshop versions write.
pub fn blend_from_enum(v: &str) -> Option<BlendMode> {
    Some(match v {
        "Nrml" | "normal" => BlendMode::Normal,
        "Mltp" | "multiply" => BlendMode::Multiply,
        "Scrn" | "screen" => BlendMode::Screen,
        "Ovrl" | "overlay" => BlendMode::Overlay,
        "Drkn" | "darken" => BlendMode::Darken,
        "Lghn" | "lighten" => BlendMode::Lighten,
        "CDdg" | "colorDodge" => BlendMode::ColorDodge,
        "CBrn" | "colorBurn" => BlendMode::ColorBurn,
        "HrdL" | "hardLight" => BlendMode::HardLight,
        "SftL" | "softLight" => BlendMode::SoftLight,
        "Dfrn" | "difference" => BlendMode::Difference,
        "Xclu" | "exclusion" => BlendMode::Exclusion,
        "linearDodge" => BlendMode::Add,
        "blendSubtraction" | "subtract" => BlendMode::Subtract,
        "blendDivide" | "divide" => BlendMode::Divide,
        "H   " | "hue" => BlendMode::Hue,
        "Strt" | "saturation" => BlendMode::Saturation,
        "Clr " | "color" => BlendMode::Color,
        "Lmns" | "luminosity" => BlendMode::Luminosity,
        _ => return None,
    })
}

pub fn blend_enum(b: BlendMode) -> &'static str {
    match b {
        BlendMode::Normal => "Nrml",
        BlendMode::Multiply => "Mltp",
        BlendMode::Screen => "Scrn",
        BlendMode::Overlay => "Ovrl",
        BlendMode::Darken => "Drkn",
        BlendMode::Lighten => "Lghn",
        BlendMode::ColorDodge => "CDdg",
        BlendMode::ColorBurn => "CBrn",
        BlendMode::HardLight => "HrdL",
        BlendMode::SoftLight => "SftL",
        BlendMode::Difference => "Dfrn",
        BlendMode::Exclusion => "Xclu",
        BlendMode::Add => "linearDodge",
        BlendMode::Subtract => "blendSubtraction",
        BlendMode::Divide => "blendDivide",
        BlendMode::Hue => "H   ",
        BlendMode::Saturation => "Strt",
        BlendMode::Color => "Clr ",
        BlendMode::Luminosity => "Lmns",
    }
}

fn enum_of<'a>(d: &'a Desc, key: &str) -> Option<&'a str> {
    match d.get(key)? {
        Value::Enum(_, v) => Some(v),
        _ => None,
    }
}

/// A colour descriptor (RGB, grey, HSB or Lab) as linear RGB.
fn color(d: Option<&Desc>) -> [f32; 3] {
    let Some(d) = d else { return [0.0; 3] };
    let srgb = match d.class.as_str() {
        "Grsc" => {
            // Grey: percentage of ink.
            let g = 1.0 - d.num("Gry ").unwrap_or(0.0) as f32 / 100.0;
            [g; 3]
        }
        "HSBC" => {
            let h = d.num("H   ").unwrap_or(0.0) as f32 / 60.0;
            let s = d.num("Strt").unwrap_or(0.0) as f32 / 100.0;
            let v = d.num("Brgh").unwrap_or(0.0) as f32 / 100.0;
            let f = |n: f32| {
                let k = (n + h).rem_euclid(6.0);
                v - v * s * k.min(4.0 - k).clamp(0.0, 1.0)
            };
            [f(5.0), f(3.0), f(1.0)]
        }
        "LbCl" => lab_to_srgb(
            d.num("Lmnc").unwrap_or(0.0) as f32,
            d.num("A   ").unwrap_or(0.0) as f32,
            d.num("B   ").unwrap_or(0.0) as f32,
        ),
        _ => {
            if let Some(r) = d.num("redFloat") {
                // Newer files: 0…1 floats.
                [r as f32, d.num("greenFloat").unwrap_or(0.0) as f32, d.num("blueFloat").unwrap_or(0.0) as f32]
            } else {
                [
                    d.num("Rd  ").unwrap_or(0.0) as f32 / 255.0,
                    d.num("Grn ").unwrap_or(0.0) as f32 / 255.0,
                    d.num("Bl  ").unwrap_or(0.0) as f32 / 255.0,
                ]
            }
        }
    };
    srgb.map(|c| srgb_to_linear(c.clamp(0.0, 1.0)))
}

/// CIE Lab (D50, as Photoshop) → sRGB 0…1.
fn lab_to_srgb(l: f32, a: f32, b: f32) -> [f32; 3] {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let finv = |t: f32| if t > 6.0 / 29.0 { t * t * t } else { 3.0 * (6.0f32 / 29.0).powi(2) * (t - 4.0 / 29.0) };
    let (x, y, z) = (0.9642 * finv(fx), finv(fy), 0.8249 * finv(fz));
    // D50 XYZ → linear sRGB (Bradford-adapted).
    let r = 3.1339 * x - 1.6169 * y - 0.4906 * z;
    let g = -0.9788 * x + 1.9161 * y + 0.0335 * z;
    let bl = 0.0719 * x - 0.2290 * y + 1.4052 * z;
    [r, g, bl].map(|c| linear_to_srgb(c.clamp(0.0, 1.0)))
}

/// The effect descriptors of one kind: the single one and/or the "…Multi" list.
fn all<'a>(fx: &'a Desc, single: &str, multi: &str) -> Vec<&'a Desc> {
    let mut v: Vec<&Desc> = fx.desc(single).into_iter().collect();
    if let Some(Value::List(l)) = fx.get(multi) {
        // The list repeats the single one in files that have both.
        v = l.iter().filter_map(|x| if let Value::Desc(d) = x { Some(d) } else { None }).collect();
    }
    // "present": part of the style (switched off ones are still listed).
    v.into_iter().filter(|d| d.bool("present").unwrap_or(true)).collect()
}

fn pct_or(d: &Desc, key: &str, default: f64) -> f32 {
    d.num(key).unwrap_or(default) as f32 / 100.0
}

/// Photoshop's spread is a share of the size: that part grows the shape,
/// the rest is blur. Returns (blur size, spread) in px.
fn spread_size(d: &Desc, scale: f32) -> (f32, f32) {
    let size = d.num("blur").unwrap_or(0.0) as f32 * scale;
    let k = (d.num("Ckmt").unwrap_or(0.0) as f32 / 100.0).clamp(0.0, 1.0);
    (size * (1.0 - k), size * k)
}

/// Reads a layer style. Returns the effects and the names of effects
/// that are switched on but not carried over.
pub fn read(fx: &Desc, light: GlobalLight) -> (Vec<Effect>, Vec<&'static str>) {
    let mut out = Vec::new();
    let mut lost = Vec::new();
    // "Scl " is the style's scale at the time it was saved; Photoshop
    // stores the sizes already scaled, so it is not applied again.
    let scale = 1.0;
    let master = fx.bool("masterFXSwitch").unwrap_or(true);
    let px = |d: &Desc, k: &str| d.num(k).unwrap_or(0.0) as f32 * scale;
    let pct = |d: &Desc, k: &str| d.num(k).unwrap_or(100.0) as f32 / 100.0;
    let angle = |d: &Desc| -> (f32, bool) {
        let global = d.bool("uglg").unwrap_or(false);
        (if global { light.angle } else { d.num("lagl").unwrap_or(light.angle as f64) as f32 }, global)
    };
    let mut push = |d: &Desc, kind: EffectKind, global: bool| {
        let blend = enum_of(d, "Md  ").and_then(blend_from_enum);
        let mut e = Effect::new(kind);
        e.enabled = master && d.bool("enab").unwrap_or(true);
        e.global_light = global;
        e.blend = blend.filter(|b| *b != e.kind.default_blend());
        out.push(e);
    };

    // Photoshop's stacking order, bottom to top.
    for d in all(fx, "DrSh", "dropShadowMulti") {
        let (a, g) = angle(d);
        push(
            d,
            EffectKind::DropShadow {
                color: color(d.desc("Clr ")),
                opacity: pct(d, "Opct"),
                angle: a,
                distance: px(d, "Dstn"),
                size: spread_size(d, scale).0,
                spread: spread_size(d, scale).1,
            },
            g,
        );
    }
    for d in all(fx, "OrGl", "outerGlowMulti") {
        push(
            d,
            EffectKind::OuterGlow {
                color: color(d.desc("Clr ")),
                opacity: pct(d, "Opct"),
                size: spread_size(d, scale).0,
                spread: spread_size(d, scale).1,
                range: pct_or(d, "Inpr", 50.0),
            },
            false,
        );
    }
    for d in all(fx, "ebbl", "bevelEmbossMulti") {
        let (a, g) = angle(d);
        let style = match enum_of(d, "bvlS") {
            Some("Embs") => BevelStyle::Emboss,
            Some("PlEb") => BevelStyle::Pillow,
            Some("OtrB") | Some("strokeEmboss") => {
                if d.bool("enab").unwrap_or(true) {
                    lost.push("bevel style");
                }
                BevelStyle::Emboss
            }
            _ => BevelStyle::Inner,
        };
        // Direction "Down": the light comes from the other side.
        let down = enum_of(d, "bvlD") == Some("Out ");
        let e = EffectKind::Bevel {
            style,
            size: px(d, "blur"),
            depth: pct(d, "srgR"),
            soften: px(d, "Sftn"),
            angle: if down { a + 180.0 } else { a },
            altitude: if g { light.altitude } else { d.num("Lald").unwrap_or(30.0) as f32 },
            highlight: d.desc("hglC").map_or([1.0; 3], |c| color(Some(c))),
            highlight_opacity: pct(d, "hglO"),
            shadow: color(d.desc("sdwC")),
            shadow_opacity: pct(d, "sdwO"),
        };
        push(d, e, g && !down);
        // Highlight/shadow modes: imagine has one mode per effect.
    }
    for d in all(fx, "IrGl", "innerGlowMulti") {
        push(
            d,
            EffectKind::InnerGlow {
                color: color(d.desc("Clr ")),
                opacity: pct(d, "Opct"),
                size: px(d, "blur"),
                range: pct_or(d, "Inpr", 50.0),
            },
            false,
        );
    }
    for d in all(fx, "IrSh", "innerShadowMulti") {
        let (a, g) = angle(d);
        push(
            d,
            EffectKind::InnerShadow {
                color: color(d.desc("Clr ")),
                opacity: pct(d, "Opct"),
                angle: a,
                distance: px(d, "Dstn"),
                size: px(d, "blur"),
            },
            g,
        );
    }
    for d in all(fx, "GrFl", "gradientFillMulti") {
        let grad = d.desc("Grad");
        let stops: Vec<&Desc> = match grad.and_then(|g| g.get("Clrs")) {
            Some(Value::List(l)) => l.iter().filter_map(|x| if let Value::Desc(d) = x { Some(d) } else { None }).collect(),
            _ => Vec::new(),
        };
        let by_loc = |first: bool| {
            stops
                .iter()
                .copied()
                .reduce(|a, b| {
                    let (la, lb) = (a.num("Lctn").unwrap_or(0.0), b.num("Lctn").unwrap_or(0.0));
                    if (lb < la) == first { b } else { a }
                })
                .map(|s| color(s.desc("Clr ")))
        };
        if stops.len() > 2 && d.bool("enab").unwrap_or(true) {
            lost.push("gradient stops");
        }
        push(
            d,
            EffectKind::GradientOverlay {
                start: by_loc(true).unwrap_or([0.0; 3]),
                end: by_loc(false).unwrap_or([1.0; 3]),
                opacity: pct(d, "Opct"),
                angle: d.num("Angl").unwrap_or(90.0) as f32,
                style: if enum_of(d, "Type") == Some("Rdl ") { GradientStyle::Radial } else { GradientStyle::Linear },
                scale: pct(d, "Scl "),
                reverse: d.bool("Rvrs").unwrap_or(false),
            },
            false,
        );
    }
    for d in all(fx, "SoFi", "solidFillMulti") {
        push(d, EffectKind::ColorOverlay { color: color(d.desc("Clr ")), opacity: pct(d, "Opct") }, false);
    }
    for d in all(fx, "FrFX", "frameFXMulti") {
        if enum_of(d, "PntT").is_some_and(|t| t != "SClr") && d.bool("enab").unwrap_or(true) {
            lost.push("stroke fill");
        }
        push(
            d,
            EffectKind::Stroke {
                size: px(d, "Sz  "),
                position: match enum_of(d, "Styl") {
                    Some("InsF") => StrokePosition::Inside,
                    Some("CtrF") => StrokePosition::Center,
                    _ => StrokePosition::Outside,
                },
                color: color(d.desc("Clr ")),
                opacity: pct(d, "Opct"),
            },
            false,
        );
    }
    for (single, multi, name) in [("ChFX", "satinMulti", "satin"), ("patternFill", "patternFillMulti", "pattern overlay")] {
        if master && all(fx, single, multi).iter().any(|d| d.bool("enab").unwrap_or(true)) {
            lost.push(name);
        }
    }
    (out, lost)
}

// --- Writing ----------------------------------------------------------------

fn unit(u: &str, v: f32) -> Value {
    Value::Unit(u.into(), v as f64)
}

fn rgb(c: [f32; 3]) -> Value {
    let s = c.map(|v| (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0) as f64);
    Value::Desc(Desc {
        class: "RGBC".into(),
        items: vec![("Rd  ".into(), Value::Double(s[0])), ("Grn ".into(), Value::Double(s[1])), ("Bl  ".into(), Value::Double(s[2]))],
    })
}

fn gradient(start: [f32; 3], end: [f32; 3]) -> Value {
    let stop = |c: [f32; 3], at: i32| {
        Value::Desc(Desc {
            class: "Clrt".into(),
            items: vec![
                ("Clr ".into(), rgb(c)),
                ("Type".into(), Value::Enum("Clry".into(), "UsrS".into())),
                ("Lctn".into(), Value::Long(at)),
                ("Mdpn".into(), Value::Long(50)),
            ],
        })
    };
    let trans = |at: i32| {
        Value::Desc(Desc {
            class: "TrnS".into(),
            items: vec![("Opct".into(), unit("#Prc", 100.0)), ("Lctn".into(), Value::Long(at)), ("Mdpn".into(), Value::Long(50))],
        })
    };
    Value::Desc(Desc {
        class: "Grdn".into(),
        items: vec![
            ("Nm  ".into(), Value::Text("Custom".into())),
            ("GrdF".into(), Value::Enum("GrdF".into(), "CstS".into())),
            ("Intr".into(), Value::Double(4096.0)),
            ("Clrs".into(), Value::List(vec![stop(start, 0), stop(end, 4096)])),
            ("Trns".into(), Value::List(vec![trans(0), trans(4096)])),
        ],
    })
}

/// The layer style descriptor for `effects` ("lfx2"); `None` if there are none.
/// `light`: the document's global light (for effects that use it).
pub fn write(effects: &[Effect], light: GlobalLight) -> Option<Desc> {
    if effects.is_empty() {
        return None;
    }
    // Per Photoshop key: the effect descriptors in order.
    let mut groups: Vec<(&'static str, &'static str, Vec<Desc>)> = Vec::new();
    for e in effects {
        let mut items: Vec<(String, Value)> = vec![
            ("enab".into(), Value::Bool(e.enabled)),
            ("present".into(), Value::Bool(true)),
            ("showInDialog".into(), Value::Bool(true)),
            ("Md  ".into(), Value::Enum("BlnM".into(), blend_enum(e.mode()).into())),
        ];
        let mut put = |k: &str, v: Value| items.push((k.into(), v));
        let lit = |a: f32| if e.global_light { light.angle } else { a };
        let (single, multi) = match &e.kind {
            EffectKind::DropShadow { color, opacity, angle, distance, size, spread } => {
                let total = size + spread;
                put("Clr ", rgb(*color));
                put("Opct", unit("#Prc", opacity * 100.0));
                put("uglg", Value::Bool(e.global_light));
                put("lagl", unit("#Ang", lit(*angle)));
                put("Dstn", unit("#Pxl", *distance));
                put("Ckmt", unit("#Pxl", if total > 0.0 { spread / total * 100.0 } else { 0.0 }));
                put("blur", unit("#Pxl", total));
                put("layerConceals", Value::Bool(true));
                ("DrSh", "dropShadowMulti")
            }
            EffectKind::InnerShadow { color, opacity, angle, distance, size } => {
                put("Clr ", rgb(*color));
                put("Opct", unit("#Prc", opacity * 100.0));
                put("uglg", Value::Bool(e.global_light));
                put("lagl", unit("#Ang", lit(*angle)));
                put("Dstn", unit("#Pxl", *distance));
                put("Ckmt", unit("#Pxl", 0.0));
                put("blur", unit("#Pxl", *size));
                ("IrSh", "innerShadowMulti")
            }
            EffectKind::OuterGlow { color, opacity, size, spread, range } => {
                let total = size + spread;
                put("Clr ", rgb(*color));
                put("Opct", unit("#Prc", opacity * 100.0));
                put("GlwT", Value::Enum("BETE".into(), "SfBL".into()));
                put("Ckmt", unit("#Pxl", if total > 0.0 { spread / total * 100.0 } else { 0.0 }));
                put("blur", unit("#Pxl", total));
                put("Inpr", unit("#Prc", range * 100.0));
                ("OrGl", "")
            }
            EffectKind::InnerGlow { color, opacity, size, range } => {
                put("Clr ", rgb(*color));
                put("Opct", unit("#Prc", opacity * 100.0));
                put("GlwT", Value::Enum("BETE".into(), "SfBL".into()));
                put("Ckmt", unit("#Pxl", 0.0));
                put("blur", unit("#Pxl", *size));
                put("Inpr", unit("#Prc", range * 100.0));
                put("glwS", Value::Enum("IGSr".into(), "SrcE".into()));
                ("IrGl", "")
            }
            EffectKind::ColorOverlay { color, opacity } => {
                put("Clr ", rgb(*color));
                put("Opct", unit("#Prc", opacity * 100.0));
                ("SoFi", "solidFillMulti")
            }
            EffectKind::GradientOverlay { start, end, opacity, angle, style, scale, reverse } => {
                put("Opct", unit("#Prc", opacity * 100.0));
                put("Grad", gradient(*start, *end));
                put("Angl", unit("#Ang", *angle));
                put(
                    "Type",
                    Value::Enum("GrdT".into(), if *style == GradientStyle::Radial { "Rdl " } else { "Lnr " }.into()),
                );
                put("Rvrs", Value::Bool(*reverse));
                put("Algn", Value::Bool(true));
                put("Scl ", unit("#Prc", scale * 100.0));
                ("GrFl", "gradientFillMulti")
            }
            EffectKind::Stroke { size, position, color, opacity } => {
                let p = match position {
                    StrokePosition::Outside => "OutF",
                    StrokePosition::Inside => "InsF",
                    StrokePosition::Center => "CtrF",
                };
                put("Styl", Value::Enum("FStl".into(), p.into()));
                put("PntT", Value::Enum("FrFl".into(), "SClr".into()));
                put("Opct", unit("#Prc", opacity * 100.0));
                put("Sz  ", unit("#Pxl", *size));
                put("Clr ", rgb(*color));
                ("FrFX", "frameFXMulti")
            }
            EffectKind::Bevel {
                style,
                size,
                depth,
                soften,
                angle,
                altitude,
                highlight,
                highlight_opacity,
                shadow,
                shadow_opacity,
            } => {
                // Separate modes for highlight and shadow; "Md  " is not used.
                items.pop();
                items.push(("hglM".into(), Value::Enum("BlnM".into(), "Scrn".into())));
                items.push(("hglC".into(), rgb(*highlight)));
                items.push(("hglO".into(), unit("#Prc", highlight_opacity * 100.0)));
                items.push(("sdwM".into(), Value::Enum("BlnM".into(), "Mltp".into())));
                items.push(("sdwC".into(), rgb(*shadow)));
                items.push(("sdwO".into(), unit("#Prc", shadow_opacity * 100.0)));
                items.push(("bvlT".into(), Value::Enum("bvlT".into(), "SfBL".into())));
                let s = match style {
                    BevelStyle::Inner => "InrB",
                    BevelStyle::Emboss => "Embs",
                    BevelStyle::Pillow => "PlEb",
                };
                items.push(("bvlS".into(), Value::Enum("BESl".into(), s.into())));
                items.push(("uglg".into(), Value::Bool(e.global_light)));
                items.push(("lagl".into(), unit("#Ang", lit(*angle))));
                items.push(("Lald".into(), unit("#Ang", if e.global_light { light.altitude } else { *altitude })));
                items.push(("srgR".into(), unit("#Prc", depth * 100.0)));
                items.push(("blur".into(), unit("#Pxl", *size)));
                items.push(("bvlD".into(), Value::Enum("BESs".into(), "In  ".into())));
                items.push(("Sftn".into(), unit("#Pxl", *soften)));
                ("ebbl", "")
            }
        };
        let d = Desc { class: single.into(), items };
        match groups.iter_mut().find(|g| g.0 == single) {
            Some(g) => g.2.push(d),
            None => groups.push((single, multi, vec![d])),
        }
    }
    let mut items: Vec<(String, Value)> =
        vec![("Scl ".into(), unit("#Prc", 100.0)), ("masterFXSwitch".into(), Value::Bool(true))];
    for (single, multi, list) in groups {
        items.push((single.into(), Value::Desc(list[0].clone())));
        // Several of one kind: Photoshop CC's lists (older versions use the first).
        if list.len() > 1 && !multi.is_empty() {
            items.push((multi.into(), Value::List(list.into_iter().map(Value::Desc).collect())));
        }
    }
    Some(Desc { class: "null".into(), items })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effects_roundtrip_through_the_descriptor() {
        let light = GlobalLight { angle: 120.0, altitude: 30.0 };
        let mut fx: Vec<Effect> = EffectKind::defaults().into_iter().map(Effect::new).collect();
        // Two strokes (Photoshop CC's multi list), one switched off, one non-default mode.
        fx.push(Effect::new(EffectKind::Stroke {
            size: 7.0,
            position: StrokePosition::Center,
            color: [0.0, 1.0, 0.0],
            opacity: 0.5,
        }));
        fx[0].enabled = false;
        fx[1].blend = Some(BlendMode::Overlay);
        for e in &mut fx {
            if let (true, Some((a, _))) = (e.global_light, e.kind.light_mut()) {
                *a = light.angle;
            }
        }
        let d = write(&fx, light).unwrap();
        // Through the binary form, as in a file.
        let mut bytes = Vec::new();
        crate::psd_data::w_desc(&mut bytes, &d);
        let d = crate::psd_data::read_desc(&bytes).unwrap();
        let (back, lost) = read(&d, light);
        assert!(lost.is_empty(), "{lost:?}");
        assert_eq!(back.len(), fx.len());
        for e in &fx {
            let b = back
                .iter()
                .find(|b| std::mem::discriminant(&b.kind) == std::mem::discriminant(&e.kind) && b.enabled == e.enabled && close(&b.kind, &e.kind))
                .unwrap_or_else(|| panic!("{e:?} not in {back:#?}"));
            assert_eq!(b.mode(), e.mode(), "{:?}", e.kind);
            assert_eq!(b.global_light, e.global_light, "{:?}", e.kind);
        }
    }

    /// Same effect up to rounding of colours to 8 bits.
    fn close(a: &EffectKind, b: &EffectKind) -> bool {
        let nums = |k: &EffectKind| -> Vec<f32> {
            format!("{k:?}")
                .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
                .filter_map(|s| s.parse().ok())
                .collect()
        };
        let (x, y) = (nums(a), nums(b));
        x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 0.01)
    }

    #[test]
    fn long_blend_names() {
        assert_eq!(blend_from_enum("multiply"), Some(BlendMode::Multiply));
        assert_eq!(blend_from_enum("Mltp"), Some(BlendMode::Multiply));
        assert_eq!(blend_from_enum("linearDodge"), Some(BlendMode::Add));
    }
}
