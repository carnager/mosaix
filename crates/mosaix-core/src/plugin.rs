//! Filter plugins: a WGSL file whose comment header describes the filter.
//!
//! ```wgsl
//! // @name Vignette
//! // @description Darkens the edges of the image
//! // @param amount "Amount" 0 100 50 %
//! // @param radius "Radius" 1 3000 800 px log
//! // @param tint "Tint" color #000000
//! // @param inverted "Invert" bool false
//! // @reach 0
//! fn process(p: vec2<i32>) -> vec4<f32> { … }
//! ```
//!
//! `@canvas` makes it paint the whole canvas, also where the layer is empty.
//! `@reach` says how far (px) the filter reads around a pixel: a number or
//! a parameter name (its value). Parameters become functions of the same
//! name in the shader (`amount()` → f32, `inverted()` → bool, `tint()` →
//! vec3<f32> linear). See [`PRELUDE_DOC`] for the helper functions.

use serde::{Deserialize, Serialize};

/// Parameter types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParamKind {
    Number { min: f32, max: f32, unit: String, log: bool },
    Bool,
    /// Linear RGB.
    Color,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub id: String,
    pub label: String,
    pub kind: ParamKind,
    /// Number: the value; bool: 0/1; colour: first of three.
    pub default: Vec<f32>,
}

impl Param {
    /// How many floats the parameter takes.
    pub fn width(&self) -> usize {
        match self.kind {
            ParamKind::Color => 3,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Reach {
    Px(f32),
    Param(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PluginSpec {
    pub name: String,
    pub description: String,
    pub params: Vec<Param>,
    pub reach: Reach,
    /// `@canvas`: also paints where the layer is empty (generators).
    pub canvas: bool,
}

/// At most this many floats of parameters (four vec4 in the job).
pub const MAX_FLOATS: usize = 16;

/// What the shader can use besides its parameters (for documentation and
/// error messages).
pub const PRELUDE_DOC: &str = "\
src(p: vec2<i32>) -> vec4<f32>        source pixel at document position p (premultiplied linear RGBA; transparent outside)
sample(p: vec2<f32>) -> vec4<f32>     bilinear source at a document position (pixel centres at +0.5)
unpremultiply(c) / premultiply(c)     straight ↔ premultiplied alpha
to_srgb(v: vec3<f32>) / to_linear(v)  sRGB transfer curve
luminance(rgb: vec3<f32>) -> f32      Rec. 709 luminance of linear RGB
canvas() -> vec2<f32>                 document size in px
rand(p: vec2<i32>, seed: u32) -> f32  0…1, stable per pixel";

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && c.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Words, with "quoted strings" kept together.
fn words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for ch in s.chars() {
        match ch {
            '"' => {
                if quoted {
                    out.push(std::mem::take(&mut cur));
                }
                quoted = !quoted;
            }
            c if c.is_whitespace() && !quoted => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn hex_color(s: &str) -> Option<Vec<f32>> {
    let h = s.trim_start_matches('#');
    if h.len() != 6 {
        return None;
    }
    let v = u32::from_str_radix(h, 16).ok()?;
    Some([16, 8, 0].iter().map(|sh| crate::color::srgb_to_linear(((v >> sh) & 255) as f32 / 255.0)).collect())
}

/// Reads the header of a filter plugin.
pub fn parse(source: &str) -> Result<PluginSpec, String> {
    let mut spec = PluginSpec { name: String::new(), description: String::new(), params: Vec::new(), reach: Reach::Px(0.0), canvas: false };
    for (n, line) in source.lines().enumerate() {
        let Some(rest) = line.trim().strip_prefix("//") else { continue };
        let rest = rest.trim();
        let Some(rest) = rest.strip_prefix('@') else { continue };
        let (key, val) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
        let val = val.trim();
        let err = |m: &str| Err(format!("line {}: {m}", n + 1));
        match key {
            "name" => spec.name = val.to_string(),
            "description" => spec.description = val.to_string(),
            "canvas" => spec.canvas = true,
            "reach" => {
                spec.reach = match val.parse::<f32>() {
                    Ok(v) => Reach::Px(v.max(0.0)),
                    Err(_) if is_ident(val) => Reach::Param(val.to_string()),
                    Err(_) => return err("@reach needs a number or a parameter name"),
                }
            }
            "param" => {
                let w = words(val);
                if w.len() < 3 || !is_ident(&w[0]) {
                    return err("@param id \"Label\" …");
                }
                let (id, label) = (w[0].clone(), w[1].clone());
                let p = match w[2].as_str() {
                    "bool" => Param {
                        id,
                        label,
                        kind: ParamKind::Bool,
                        default: vec![(w.get(3).map(String::as_str) == Some("true")) as u8 as f32],
                    },
                    "color" => Param {
                        id,
                        label,
                        kind: ParamKind::Color,
                        default: w.get(3).and_then(|c| hex_color(c)).unwrap_or(vec![0.0; 3]),
                    },
                    _ => {
                        let num = |i: usize| w.get(i).and_then(|v| v.parse::<f32>().ok());
                        let (Some(min), Some(max), Some(def)) = (num(2), num(3), num(4)) else {
                            return err("@param id \"Label\" min max default [unit] [log]");
                        };
                        if min >= max {
                            return err("min must be below max");
                        }
                        let rest: Vec<&str> = w[5..].iter().map(String::as_str).collect();
                        Param {
                            id,
                            label,
                            kind: ParamKind::Number {
                                min,
                                max,
                                unit: rest.iter().find(|s| **s != "log").map(|s| s.to_string()).unwrap_or_default(),
                                log: rest.contains(&"log") && min > 0.0,
                            },
                            default: vec![def.clamp(min, max)],
                        }
                    }
                };
                if spec.params.iter().any(|q| q.id == p.id) {
                    return err("parameter defined twice");
                }
                spec.params.push(p);
            }
            _ => {}
        }
    }
    if spec.name.is_empty() {
        return Err("missing // @name".into());
    }
    if spec.params.iter().map(Param::width).sum::<usize>() > MAX_FLOATS {
        return Err(format!("too many parameters (at most {MAX_FLOATS} numbers; a colour counts 3)"));
    }
    if let Reach::Param(r) = &spec.reach
        && !spec.params.iter().any(|p| p.id == *r && matches!(p.kind, ParamKind::Number { .. }))
    {
        return Err(format!("@reach {r}: no such number parameter"));
    }
    Ok(spec)
}

impl PluginSpec {
    /// The parameter floats for the GPU (in order; missing values = defaults).
    pub fn floats(&self, values: &std::collections::BTreeMap<String, Vec<f32>>) -> [f32; MAX_FLOATS] {
        let mut out = [0.0; MAX_FLOATS];
        let mut i = 0;
        for p in &self.params {
            let v = values.get(&p.id).filter(|v| v.len() == p.width()).unwrap_or(&p.default);
            for (k, x) in v.iter().enumerate() {
                out[i + k] = match &p.kind {
                    ParamKind::Number { min, max, .. } => x.clamp(*min, *max),
                    _ => *x,
                };
            }
            i += p.width();
        }
        out
    }

    /// How far the filter reads around a pixel, px.
    pub fn reach(&self, values: &std::collections::BTreeMap<String, Vec<f32>>) -> f32 {
        match &self.reach {
            Reach::Px(v) => *v,
            Reach::Param(id) => {
                let f = self.floats(values);
                let mut i = 0;
                for p in &self.params {
                    if p.id == *id {
                        return f[i].abs();
                    }
                    i += p.width();
                }
                0.0
            }
        }
    }

    /// WGSL functions returning each parameter (`amount()`, `tint()` …).
    pub fn accessors(&self) -> String {
        let mut s = String::new();
        let mut i = 0;
        let at = |i: usize| format!("JOB.p[{}][{}]", i / 4, i % 4);
        for p in &self.params {
            match p.kind {
                ParamKind::Number { .. } => s += &format!("fn {}() -> f32 {{ return {}; }}\n", p.id, at(i)),
                ParamKind::Bool => s += &format!("fn {}() -> bool {{ return {} > 0.5; }}\n", p.id, at(i)),
                ParamKind::Color => {
                    s += &format!("fn {}() -> vec3<f32> {{ return vec3<f32>({}, {}, {}); }}\n", p.id, at(i), at(i + 1), at(i + 2))
                }
            }
            i += p.width();
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header() {
        let src = "// @name Vignette\n// @param amount \"Amount\" 0 100 50 %\n// @param radius \"Radius\" 1 3000 800 px log\n// @param tint \"Tint\" color #ff0000\n// @param inv \"Invert\" bool true\n// @reach radius\nfn process(p: vec2<i32>) -> vec4<f32> { return src(p); }\n";
        let s = parse(src).unwrap();
        assert_eq!(s.name, "Vignette");
        assert_eq!(s.params.len(), 4);
        assert!(matches!(&s.params[1].kind, ParamKind::Number { log: true, unit, .. } if unit == "px"));
        let mut v = std::collections::BTreeMap::new();
        v.insert("radius".to_string(), vec![5000.0]);
        let f = s.floats(&v);
        assert_eq!(&f[..6], &[50.0, 3000.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.reach(&v), 3000.0);
        assert!(s.accessors().contains("fn tint() -> vec3<f32> { return vec3<f32>(JOB.p[0][2], JOB.p[0][3], JOB.p[1][0]); }"));
        assert!(parse("// @param x \"X\" 0 1 0").is_err());
        assert!(parse("// @name A\n// @reach nope").is_err());
    }
}
