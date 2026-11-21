//! Vector layers: text and shapes kept as a description and rasterised
//! when composited (crate mosaix-vector), so they stay sharp and editable.

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
    /// OpenType features switched away from the font's defaults
    /// ("liga" → 0, "smcp" → 1 …).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub features: std::collections::BTreeMap<String, u16>,
    /// Text on a path: one line set along the curve (x/y and width unused).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<TextPath>,
    pub x: f64,
    pub y: f64,
}

/// The curve a text runs along (in the text layer's own coordinates).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextPath {
    pub nodes: Vec<PathNode>,
    #[serde(default)]
    pub closed: bool,
    /// Where the text starts, as a fraction of the path length (0…1).
    #[serde(default)]
    pub offset: f32,
    /// Run along the other side (reversed direction).
    #[serde(default)]
    pub flip: bool,
}

/// The OpenType features offered in the UI: tag, label, on by default.
pub const FEATURES: &[(&str, &str, bool)] = &[
    ("liga", "Standard Ligatures", true),
    ("calt", "Contextual Alternates", true),
    ("kern", "Kerning", true),
    ("dlig", "Discretionary Ligatures", false),
    ("smcp", "Small Caps", false),
    ("c2sc", "Capitals to Small Caps", false),
    ("onum", "Old-Style Figures", false),
    ("tnum", "Tabular Figures", false),
    ("frac", "Fractions", false),
    ("ordn", "Ordinals", false),
    ("swsh", "Swash", false),
    ("salt", "Stylistic Alternates", false),
    ("sups", "Superscript", false),
    ("subs", "Subscript", false),
];

impl TextSpec {
    /// Whether a feature is on (switched, or by default).
    pub fn feature(&self, tag: &str) -> bool {
        match self.features.get(tag) {
            Some(v) => *v != 0,
            None => FEATURES.iter().any(|(t, _, on)| *t == tag && *on),
        }
    }

    /// Switches a feature; values equal to the default are not stored.
    pub fn set_feature(&mut self, tag: &str, on: bool) {
        let default = FEATURES.iter().any(|(t, _, d)| *t == tag && *d);
        if on == default {
            self.features.remove(tag);
        } else {
            self.features.insert(tag.to_string(), on as u16);
        }
    }

    /// The features as CSS `font-feature-settings` ("" if none).
    pub fn feature_css(&self) -> String {
        self.features.iter().map(|(t, v)| format!("\"{t}\" {v}")).collect::<Vec<_>>().join(", ")
    }

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
            features: Default::default(),
            path: None,
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

/// Makes node `i` smooth: handles along the direction from its previous to
/// its next node, a sixth of that distance long (the curve then runs through
/// the nodes like a Catmull-Rom spline). The ends of an open path get one
/// handle towards their neighbour.
pub fn smooth_node(nodes: &mut [PathNode], closed: bool, i: usize) {
    let n = nodes.len();
    if n < 2 || i >= n {
        return;
    }
    let v = |p: [f64; 2]| glam::DVec2::new(p[0], p[1]);
    let a = |p: glam::DVec2| [p.x, p.y];
    let p = v(nodes[i].p);
    let prev = if i > 0 { Some(v(nodes[i - 1].p)) } else if closed { Some(v(nodes[n - 1].p)) } else { None };
    let next = if i + 1 < n { Some(v(nodes[i + 1].p)) } else if closed { Some(v(nodes[0].p)) } else { None };
    let (h_in, h_out) = match (prev, next) {
        (Some(pp), Some(nn)) => {
            let d = (nn - pp) / 6.0;
            (p - d, p + d)
        }
        (None, Some(nn)) => (p, p + (nn - p) / 3.0),
        (Some(pp), None) => (p + (pp - p) / 3.0, p),
        (None, None) => (p, p),
    };
    nodes[i].h_in = a(h_in);
    nodes[i].h_out = a(h_out);
}

/// Makes a pen path or a polygon smooth (`true`) or all corners (`false`);
/// a polygon becomes a pen path. False if the geometry has no nodes.
pub fn smooth_geometry(g: &mut Geometry, smooth: bool) -> bool {
    if let Geometry::Polygon { points, closed } = g {
        let nodes = points.iter().map(|p| PathNode::corner(*p)).collect();
        *g = Geometry::Bezier { nodes, closed: *closed };
    }
    match g {
        Geometry::Bezier { nodes, closed } => {
            smooth_all(nodes, *closed, smooth);
            true
        }
        _ => false,
    }
}

/// Makes every node smooth (`true`) or a corner (`false`).
pub fn smooth_all(nodes: &mut [PathNode], closed: bool, smooth: bool) {
    for i in 0..nodes.len() {
        if smooth {
            smooth_node(nodes, closed, i);
        } else {
            nodes[i] = PathNode::corner(nodes[i].p);
        }
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
    /// Where the stroke lies relative to the path (closed paths).
    #[serde(default, skip_serializing_if = "is_default")]
    pub align: StrokeAlign,
    #[serde(default, skip_serializing_if = "is_default")]
    pub join: StrokeJoin,
    #[serde(default, skip_serializing_if = "is_default")]
    pub cap: StrokeCap,
}

impl StrokeStyle {
    /// A centred stroke with round joins and caps.
    pub fn new(color: [f32; 4], width: f32) -> Self {
        Self { color, width, align: StrokeAlign::default(), join: StrokeJoin::default(), cap: StrokeCap::default() }
    }
}

fn is_default<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrokeAlign {
    #[default]
    Center,
    Inside,
    Outside,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrokeJoin {
    #[default]
    Round,
    Miter,
    Bevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrokeCap {
    #[default]
    Round,
    Butt,
    Square,
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
    /// Fills with a gradient or pattern instead of `fill`'s colour
    /// (Photoshop's gradient / pattern fill layers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paint: Option<Paint>,
}

fn one() -> f32 {
    1.0
}

fn ninety() -> f32 {
    90.0
}

/// What a shape is filled with, other than a solid colour.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Paint {
    Gradient {
        #[serde(default)]
        gradient: crate::gradient::Gradient,
        #[serde(default)]
        style: crate::gradient::GradientStyle,
        /// Degrees, 90 = from bottom to top.
        #[serde(default = "ninety")]
        angle: f32,
        /// Length relative to the reference box (1 = edge to edge).
        #[serde(default = "one")]
        scale: f32,
        #[serde(default)]
        reverse: bool,
        /// Shift of the centre, as a share of the reference box's size.
        #[serde(default)]
        offset: [f32; 2],
        /// Reference box in layer coordinates; none = the shape's bounds.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bounds: Option<[f32; 4]>,
    },
    /// Tiled in document coordinates.
    Pattern {
        pattern: crate::pattern::Pattern,
        #[serde(default = "one")]
        scale: f32,
        #[serde(default)]
        offset: [f32; 2],
    },
}

impl Paint {
    pub fn gradient(gradient: crate::gradient::Gradient) -> Self {
        Self::Gradient {
            gradient,
            style: Default::default(),
            angle: 90.0,
            scale: 1.0,
            reverse: false,
            offset: [0.0; 2],
            bounds: None,
        }
    }
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
    fn smoothing_nodes() {
        let pts = [[0.0, 0.0], [60.0, 30.0], [120.0, 0.0]];
        let mut g = Geometry::Polygon { points: pts.to_vec(), closed: false };
        assert!(smooth_geometry(&mut g, true));
        let Geometry::Bezier { nodes, closed: false } = &g else { panic!("{g:?}") };
        // Middle: handles parallel to next − prev, a sixth of it.
        assert_eq!(nodes[1].h_in, [40.0, 30.0]);
        assert_eq!(nodes[1].h_out, [80.0, 30.0]);
        // Open ends: one handle towards the neighbour.
        assert_eq!(nodes[0].h_in, [0.0, 0.0]);
        assert_eq!(nodes[0].h_out, [20.0, 10.0]);
        assert_eq!(nodes[2].h_out, [120.0, 0.0]);
        let mut g2 = g.clone();
        assert!(smooth_geometry(&mut g2, false));
        let Geometry::Bezier { nodes, .. } = &g2 else { unreachable!() };
        assert!(nodes.iter().all(|n| !n.is_smooth()));
        assert!(!smooth_geometry(&mut Geometry::Ellipse { cx: 0.0, cy: 0.0, rx: 1.0, ry: 1.0 }, true));
    }

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
