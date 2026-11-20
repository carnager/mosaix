//! Vector layers: text and shapes kept as a description and rasterised
//! when composited (crate imagine-vector), so they stay sharp and editable.

use serde::{Deserialize, Serialize};

/// Affine transform [a, b, c, d, e, f]: x' = a·x + c·y + e, y' = b·x + d·y + f.
pub type Affine = [f64; 6];

pub const IDENTITY: Affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn identity() -> Affine {
    IDENTITY
}

/// Applies `t` after `m` (the result maps x to t(m(x))).
pub fn compose(t: Affine, m: Affine) -> Affine {
    [
        t[0] * m[0] + t[2] * m[1],
        t[1] * m[0] + t[3] * m[1],
        t[0] * m[2] + t[2] * m[3],
        t[1] * m[2] + t[3] * m[3],
        t[0] * m[4] + t[2] * m[5] + t[4],
        t[1] * m[4] + t[3] * m[5] + t[5],
    ]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorLayer {
    #[serde(flatten)]
    pub kind: VectorKind,
    /// Applied to the content (free transform keeps it editable).
    #[serde(default = "identity")]
    pub transform: Affine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VectorKind {
    Text(TextSpec),
    Shape(ShapeSpec),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
    Justify,
}

impl TextAlign {
    pub const ALL: [Self; 4] = [Self::Left, Self::Center, Self::Right, Self::Justify];

    pub fn name(self) -> &'static str {
        match self {
            Self::Left => crate::tr("Left"),
            Self::Center => crate::tr("Center"),
            Self::Right => crate::tr("Right"),
            Self::Justify => crate::tr("Justify"),
        }
    }
}

fn default_font() -> String {
    "sans-serif".into()
}
fn default_weight() -> f32 {
    400.0
}
fn default_line_height() -> f32 {
    1.2
}

/// A text block. Position (x, y) is the top-left of the text box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSpec {
    pub text: String,
    /// Font family (or CSS generic: sans-serif, serif, monospace).
    #[serde(default = "default_font")]
    pub font: String,
    /// Size in px.
    pub size: f32,
    /// 100 … 900 (400 regular, 700 bold).
    #[serde(default = "default_weight")]
    pub weight: f32,
    #[serde(default)]
    pub italic: bool,
    /// Linear RGB + alpha (straight).
    pub color: [f32; 4],
    #[serde(default)]
    pub align: TextAlign,
    /// Line height as a multiple of the size.
    #[serde(default = "default_line_height")]
    pub line_height: f32,
    /// Extra space between letters, px.
    #[serde(default)]
    pub letter_spacing: f32,
    /// Paragraph text: wrap at this width (px); point text if `None`.
    #[serde(default)]
    pub width: Option<f32>,
    pub x: f64,
    pub y: f64,
}

impl TextSpec {
    pub fn new(text: &str, x: f64, y: f64, size: f32, color: [f32; 4]) -> Self {
        Self {
            text: text.into(),
            font: default_font(),
            size,
            weight: default_weight(),
            italic: false,
            color,
            align: TextAlign::Left,
            line_height: default_line_height(),
            letter_spacing: 0.0,
            width: None,
            x,
            y,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum Geometry {
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        #[serde(default)]
        radius: f64,
    },
    Ellipse {
        cx: f64,
        cy: f64,
        rx: f64,
        ry: f64,
    },
    /// Straight segments through the points.
    Polygon {
        points: Vec<[f64; 2]>,
        #[serde(default = "yes")]
        closed: bool,
    },
    /// SVG path data ("M 0 0 L 10 0 C …").
    Path {
        d: String,
    },
    /// An editable Bézier path (pen tool).
    Bezier {
        nodes: Vec<PathNode>,
        #[serde(default)]
        closed: bool,
    },
}

/// A node of a Bézier path: position and the two handles (absolute; a
/// handle at the node itself means a corner on that side).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PathNode {
    pub p: [f64; 2],
    pub h_in: [f64; 2],
    pub h_out: [f64; 2],
}

impl PathNode {
    pub fn corner(p: [f64; 2]) -> Self {
        Self {
            p,
            h_in: p,
            h_out: p,
        }
    }

    /// Has handles (a smooth curve point).
    pub fn is_smooth(&self) -> bool {
        self.h_in != self.p || self.h_out != self.p
    }
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrokeStyle {
    /// Linear RGB + alpha.
    pub color: [f32; 4],
    pub width: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeSpec {
    #[serde(flatten)]
    pub geometry: Geometry,
    /// Linear RGB + alpha; no fill if `None`.
    #[serde(default)]
    pub fill: Option<[f32; 4]>,
    #[serde(default)]
    pub stroke: Option<StrokeStyle>,
}

impl VectorLayer {
    pub fn text(spec: TextSpec) -> Self {
        Self {
            kind: VectorKind::Text(spec),
            transform: IDENTITY,
        }
    }

    pub fn shape(spec: ShapeSpec) -> Self {
        Self {
            kind: VectorKind::Shape(spec),
            transform: IDENTITY,
        }
    }

    /// A short layer name from the content.
    pub fn default_name(&self) -> String {
        match &self.kind {
            VectorKind::Text(t) => {
                let first = t.text.lines().next().unwrap_or("").trim();
                let mut name: String = first.chars().take(24).collect();
                if first.chars().count() > 24 {
                    name.push('…');
                }
                if name.is_empty() { "Text".into() } else { name }
            }
            VectorKind::Shape(s) => match s.geometry {
                Geometry::Rect { .. } => crate::tr("Rectangle").into(),
                Geometry::Ellipse { .. } => "Ellipse".into(),
                Geometry::Polygon { .. } => "Polygon".into(),
                Geometry::Path { .. } | Geometry::Bezier { .. } => crate::tr("Path").into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip_and_defaults() {
        let t = VectorLayer::text(TextSpec::new(
            "Hallo\nWelt",
            10.0,
            20.0,
            48.0,
            [1.0, 0.0, 0.0, 1.0],
        ));
        let j = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<VectorLayer>(&j).unwrap(), t);
        let s: VectorLayer = serde_json::from_str(
            r#"{"type":"shape","shape":"rect","x":1,"y":2,"w":3,"h":4,"fill":[0,0,1,1]}"#,
        )
        .unwrap();
        assert_eq!(s.transform, IDENTITY);
        assert!(matches!(
            s.kind,
            VectorKind::Shape(ShapeSpec {
                geometry: Geometry::Rect { radius: 0.0, .. },
                stroke: None,
                ..
            })
        ));
        assert_eq!(t.default_name(), "Hallo");
    }

    #[test]
    fn compose_applies_right_to_left() {
        let scale = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let shift = [1.0, 0.0, 0.0, 1.0, 10.0, 0.0];
        // Scale, then shift: x = 1 → 2 → 12.
        let m = compose(shift, scale);
        assert_eq!(m[0] * 1.0 + m[4], 12.0);
    }
}
