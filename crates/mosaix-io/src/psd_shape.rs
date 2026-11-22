//! Photoshop shape and fill layers → mosaix shape layers.
//!
//! A shape layer is a solid-colour fill layer (`SoCo`, or `vscg` in newer
//! files) clipped by a vector mask (`vmsk` / `vsms`), optionally with a
//! stroke (`vstk`). Paths are Bézier knots in 8.24 fixed point, relative to
//! the canvas. Several subpaths are combined with their path operations
//! (union, subtract, intersect, exclude) into one outline (i_overlay).

use mosaix_core::color::srgb_to_linear;
use mosaix_core::vector::{Geometry, PathNode, ShapeSpec, StrokeAlign, StrokeCap, StrokeJoin, StrokeStyle};

use crate::psd_data::{Desc, Value};

/// A subpath: its operation (1 union, 2 subtract, 3 intersect, 0 exclude,
/// −1 unspecified = union), closed or open, its knots.
#[derive(Debug, Clone, PartialEq)]
pub struct Subpath {
    pub op: i16,
    pub closed: bool,
    pub nodes: Vec<PathNode>,
}

/// A parsed vector mask.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorMask {
    pub invert: bool,
    pub disabled: bool,
    /// The path starts with everything filled ("initial fill").
    pub initial_fill: bool,
    pub subpaths: Vec<Subpath>,
    /// Canvas size (for operations that start from the whole canvas).
    pub size: (u32, u32),
}

/// Reads a `vmsk` / `vsms` block. `size`: canvas size (the coordinates are
/// fractions of it).
pub fn read_vector_mask(b: &[u8], (w, h): (u32, u32)) -> Option<VectorMask> {
    let u32_at = |p: usize| Some(u32::from_be_bytes(b.get(p..p + 4)?.try_into().ok()?));
    let _version = u32_at(0)?;
    let flags = u32_at(4)?;
    let mut m = VectorMask { invert: flags & 1 != 0, disabled: flags & 4 != 0, initial_fill: false, subpaths: Vec::new(), size: (w, h) };
    let fixed = |p: usize| -> Option<f64> { Some(i32::from_be_bytes(b.get(p..p + 4)?.try_into().ok()?) as f64 / (1 << 24) as f64) };
    let point = |p: usize| -> Option<[f64; 2]> { Some([fixed(p + 4)? * w as f64, fixed(p)? * h as f64]) };
    let mut p = 8;
    let mut left = 0usize;
    while p + 26 <= b.len() {
        let sel = u16::from_be_bytes([b[p], b[p + 1]]);
        let r = p + 2;
        match sel {
            0 | 3 => {
                left = u16::from_be_bytes([b[r], b[r + 1]]) as usize;
                let op = i16::from_be_bytes([b[r + 2], b[r + 3]]);
                m.subpaths.push(Subpath { op, closed: sel == 0, nodes: Vec::with_capacity(left) });
            }
            1 | 2 | 4 | 5 if left > 0 => {
                let node = PathNode { h_in: point(r)?, p: point(r + 8)?, h_out: point(r + 16)? };
                m.subpaths.last_mut()?.nodes.push(node);
                left -= 1;
            }
            8 => m.initial_fill = u16::from_be_bytes([b[r], b[r + 1]]) == 1,
            _ => {}
        }
        p += 26;
    }
    m.subpaths.retain(|s| !s.nodes.is_empty());
    Some(m)
}

/// The subpath as a polygon (curves flattened to ≈ 0.1 px).
fn flatten(s: &Subpath) -> Vec<[f64; 2]> {
    let n = s.nodes.len();
    let mut out = Vec::new();
    let segs = if s.closed { n } else { n.saturating_sub(1) };
    for i in 0..segs {
        let (a, b) = (&s.nodes[i], &s.nodes[(i + 1) % n]);
        out.push(a.p);
        if a.h_out == a.p && b.h_in == b.p {
            continue;
        }
        // Cubic: steps from the control polygon's length.
        let d = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
        let len = d(a.p, a.h_out) + d(a.h_out, b.h_in) + d(b.h_in, b.p);
        let steps = (len.sqrt() * 2.0).clamp(4.0, 256.0) as usize;
        for k in 1..steps {
            let t = k as f64 / steps as f64;
            let u = 1.0 - t;
            let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            out.push([
                w0 * a.p[0] + w1 * a.h_out[0] + w2 * b.h_in[0] + w3 * b.p[0],
                w0 * a.p[1] + w1 * a.h_out[1] + w2 * b.h_in[1] + w3 * b.p[1],
            ]);
        }
    }
    if let (false, Some(l)) = (s.closed, s.nodes.last()) {
        out.push(l.p);
    }
    out
}

/// The geometry of a vector mask, if mosaix can represent it. One path
/// stays an editable Bézier path; several are combined with Photoshop's
/// path operations into their outline.
pub fn geometry(m: &VectorMask) -> Option<Geometry> {
    geometry_and_stroke(m).map(|g| g.0)
}

/// [`geometry`], and whether Photoshop strokes it (not when an
/// intersection is involved or the path starts from the whole canvas).
fn geometry_and_stroke(m: &VectorMask) -> Option<(Geometry, bool)> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::float::single::SingleFloatOverlay;
    if m.initial_fill && m.subpaths.is_empty() && !m.invert && !m.disabled {
        // An empty vector mask showing everything: a fill of the whole canvas.
        let (w, h) = (m.size.0 as f64, m.size.1 as f64);
        return Some((Geometry::Rect { x: 0.0, y: 0.0, w, h, radius: 0.0 }, false));
    }
    if m.invert || m.disabled || m.initial_fill || m.subpaths.is_empty() {
        return None;
    }
    if m.subpaths.len() == 1 && m.subpaths[0].op != 2 {
        let s = &m.subpaths[0];
        return Some((Geometry::Bezier { nodes: s.nodes.clone(), closed: s.closed }, s.op != 3));
    }
    // Components: a subpath with an operation starts one, the following
    // subpaths without (−1) belong to it (custom shapes with holes).
    let mut comps: Vec<(i16, Vec<Vec<[f64; 2]>>)> = Vec::new();
    for s in &m.subpaths {
        match comps.last_mut() {
            Some(c) if s.op == -1 => c.1.push(flatten(s)),
            _ => comps.push((s.op, vec![flatten(s)])),
        }
    }
    let shape = |c: &Vec<Vec<[f64; 2]>>| c.overlay(&Vec::<Vec<[f64; 2]>>::new(), OverlayRule::Subject, FillRule::EvenOdd);
    // The first component applies to an empty surface, except that
    // subtracting starts from the whole canvas.
    let (w, h) = (m.size.0 as f64, m.size.1 as f64);
    let canvas = vec![vec![vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]]];
    let mut acc: Vec<Vec<Vec<[f64; 2]>>> = match comps[0].0 {
        2 => canvas.overlay(&shape(&comps[0].1), OverlayRule::Difference, FillRule::NonZero),
        _ => shape(&comps[0].1),
    };
    for (op, c) in &comps[1..] {
        let rule = match op {
            2 => OverlayRule::Difference,
            3 => OverlayRule::Intersect,
            0 => OverlayRule::Xor,
            _ => OverlayRule::Union,
        };
        acc = acc.overlay(&shape(c), rule, FillRule::NonZero);
    }
    // Outer contours and holes come wound in opposite directions: nonzero fill.
    use std::fmt::Write;
    let mut d = String::new();
    for contour in acc.iter().flatten() {
        for (i, p) in contour.iter().enumerate() {
            let _ = write!(d, "{} {:.3} {:.3} ", if i == 0 { "M" } else { "L" }, p[0], p[1]);
        }
        d.push_str("Z ");
    }
    if d.is_empty() {
        return None;
    }
    let stroked = comps[0].0 != 2 && !comps.iter().any(|c| c.0 == 3);
    Some((Geometry::Path { d: d.trim_end().to_string() }, stroked))
}

fn color(d: Option<&Desc>) -> Option<[f32; 4]> {
    let d = d?;
    let c = |k: &str| (d.num(k).unwrap_or(0.0) as f32 / 255.0).clamp(0.0, 1.0);
    let rgb = match d.class.as_str() {
        "Grsc" => [1.0 - d.num("Gry ").unwrap_or(0.0) as f32 / 100.0; 3],
        _ => match d.num("redFloat") {
            Some(r) => [r as f32, d.num("greenFloat").unwrap_or(0.0) as f32, d.num("blueFloat").unwrap_or(0.0) as f32],
            None => [c("Rd  "), c("Grn "), c("Bl  ")],
        },
    };
    let l = rgb.map(|v| srgb_to_linear(v.clamp(0.0, 1.0)));
    Some([l[0], l[1], l[2], 1.0])
}

/// The colour of a solid fill descriptor (`SoCo` / `vscg` content).
pub fn solid_fill(d: &Desc) -> Option<[f32; 4]> {
    color(d.desc("Clr "))
}

/// The stroke of a `vstk` descriptor and whether the fill is on.
/// `dpi`: document resolution (stroke widths may be in points).
pub fn stroke(d: &Desc, dpi: f64) -> (Option<StrokeStyle>, bool) {
    let fill = d.bool("fillEnabled").unwrap_or(true);
    if !d.bool("strokeEnabled").unwrap_or(false) {
        return (None, fill);
    }
    let width = match d.get("strokeStyleLineWidth") {
        Some(Value::Unit(u, v)) if u == "#Pnt" => v * dpi / 72.0,
        Some(Value::Unit(_, v)) | Some(Value::Double(v)) => *v,
        _ => 1.0,
    };
    let opacity = d.num("strokeStyleOpacity").unwrap_or(100.0) as f32 / 100.0;
    let Some(mut c) = d.desc("strokeStyleContent").and_then(|c| color(c.desc("Clr "))) else {
        // Gradient or pattern strokes: not carried over.
        return (None, fill);
    };
    c[3] = opacity;
    let e = |k: &str| match d.get(k) {
        Some(Value::Enum(_, v)) => v.as_str(),
        _ => "",
    };
    let align = match e("strokeStyleLineAlignment") {
        "strokeStyleAlignInside" => StrokeAlign::Inside,
        "strokeStyleAlignOutside" => StrokeAlign::Outside,
        _ => StrokeAlign::Center,
    };
    // Photoshop's defaults: miter joins, butt caps.
    let join = match e("strokeStyleLineJoinType") {
        "strokeStyleRoundJoin" => StrokeJoin::Round,
        "strokeStyleBevelJoin" => StrokeJoin::Bevel,
        _ => StrokeJoin::Miter,
    };
    let cap = match e("strokeStyleLineCapType") {
        "strokeStyleRoundCap" => StrokeCap::Round,
        "strokeStyleSquareCap" => StrokeCap::Square,
        _ => StrokeCap::Butt,
    };
    (Some(StrokeStyle { align, join, cap, ..StrokeStyle::new(c, width as f32) }), fill)
}

/// A shape layer for mosaix: the path (or the whole canvas for a fill
/// layer without one), the fill colour and the stroke.
pub fn shape(
    mask: Option<&VectorMask>,
    fill: [f32; 4],
    stroke: Option<(Option<StrokeStyle>, bool)>,
    (w, h): (u32, u32),
) -> Option<ShapeSpec> {
    let (geometry, stroked) = match mask {
        Some(m) => geometry_and_stroke(m)?,
        None => (Geometry::Rect { x: 0.0, y: 0.0, w: w as f64, h: h as f64, radius: 0.0 }, true),
    };
    let (stroke, fill_on) = stroke.unwrap_or((None, true));
    let stroke = stroke.filter(|_| stroked);
    Some(ShapeSpec { geometry, fill: fill_on.then_some(fill), stroke, paint: None, parts: Vec::new() })
}

// --- Writing ----------------------------------------------------------------

/// A shape layer as Photoshop stores it: path (document px), fill, stroke.
#[derive(Debug, Clone, PartialEq)]
pub struct PsdShape {
    /// Bézier subpaths and whether each is closed.
    pub subpaths: Vec<(Vec<PathNode>, bool)>,
    /// Linear RGB(A); `None`: no fill.
    pub fill: Option<[f32; 4]>,
    pub stroke: Option<StrokeStyle>,
    /// A gradient or pattern instead of `fill`'s colour (in document px).
    pub paint: Option<mosaix_core::vector::Paint>,
}

/// The vector mask block ("vsms"). The subpaths form one component (holes
/// by even-odd), as written by Photoshop for custom shapes.
pub fn write_vector_mask(subpaths: &[(Vec<PathNode>, bool)], (w, h): (u32, u32)) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend(3u32.to_be_bytes()); // version
    o.extend(0u32.to_be_bytes()); // flags
    let record = |o: &mut Vec<u8>, sel: u16, body: &[u8]| {
        let start = o.len();
        o.extend(sel.to_be_bytes());
        o.extend(body);
        o.resize(start + 26, 0);
    };
    record(&mut o, 6, &[]); // path fill rule record
    record(&mut o, 8, &[0, 0]); // initial fill: empty
    let fixed = |v: f64| ((v * (1 << 24) as f64).round() as i32).to_be_bytes();
    for (i, (nodes, closed)) in subpaths.iter().enumerate() {
        let mut len = (nodes.len() as u16).to_be_bytes().to_vec();
        // First: "combine" (starts a component), the others belong to it.
        len.extend(if i == 0 { 1i16 } else { -1 }.to_be_bytes());
        len.extend(if i == 0 { 1u16 } else { 0 }.to_be_bytes());
        record(&mut o, if *closed { 0 } else { 3 }, &len);
        for n in nodes {
            let mut b = Vec::with_capacity(24);
            for p in [n.h_in, n.p, n.h_out] {
                b.extend(fixed(p[1] / h as f64));
                b.extend(fixed(p[0] / w as f64));
            }
            let smooth = n.is_smooth();
            let sel = match (*closed, smooth) {
                (true, true) => 1,
                (true, false) => 2,
                (false, true) => 4,
                (false, false) => 5,
            };
            record(&mut o, sel, &b);
        }
    }
    o
}

fn rgb_desc(c: [f32; 4]) -> Desc {
    let v = |x: f32| Value::Double((mosaix_core::color::linear_to_srgb(x.clamp(0.0, 1.0)) * 255.0) as f64);
    Desc {
        class: "RGBC".into(),
        items: vec![("Rd  ".into(), v(c[0])), ("Grn ".into(), v(c[1])), ("Bl  ".into(), v(c[2]))],
    }
}

/// The solid fill descriptor ("SoCo" block).
pub fn write_solid(c: [f32; 4]) -> Desc {
    Desc { class: "null".into(), items: vec![("Clr ".into(), Value::Desc(rgb_desc(c)))] }
}

/// The stroke descriptor ("vstk" block).
pub fn write_stroke(st: Option<&StrokeStyle>, fill_on: bool) -> Desc {
    let en = |t: &str, v: &str| Value::Enum(t.into(), v.into());
    let unit = |u: &str, v: f64| Value::Unit(u.into(), v);
    let none = StrokeStyle::new([0.0, 0.0, 0.0, 1.0], 1.0);
    let s = st.unwrap_or(&none);
    let align = match s.align {
        StrokeAlign::Inside => "strokeStyleAlignInside",
        StrokeAlign::Center => "strokeStyleAlignCenter",
        StrokeAlign::Outside => "strokeStyleAlignOutside",
    };
    let join = match s.join {
        StrokeJoin::Miter => "strokeStyleMiterJoin",
        StrokeJoin::Round => "strokeStyleRoundJoin",
        StrokeJoin::Bevel => "strokeStyleBevelJoin",
    };
    let cap = match s.cap {
        StrokeCap::Butt => "strokeStyleButtCap",
        StrokeCap::Round => "strokeStyleRoundCap",
        StrokeCap::Square => "strokeStyleSquareCap",
    };
    Desc {
        class: "strokeStyle".into(),
        items: vec![
            ("strokeStyleVersion".into(), Value::Long(2)),
            ("strokeEnabled".into(), Value::Bool(st.is_some())),
            ("fillEnabled".into(), Value::Bool(fill_on)),
            ("strokeStyleLineWidth".into(), unit("#Pxl", s.width as f64)),
            ("strokeStyleLineDashOffset".into(), unit("#Pnt", 0.0)),
            ("strokeStyleMiterLimit".into(), Value::Double(100.0)),
            ("strokeStyleLineCapType".into(), en("strokeStyleLineCapType", cap)),
            ("strokeStyleLineJoinType".into(), en("strokeStyleLineJoinType", join)),
            ("strokeStyleLineAlignment".into(), en("strokeStyleLineAlignment", align)),
            ("strokeStyleScaleLock".into(), Value::Bool(false)),
            ("strokeStyleStrokeAdjust".into(), Value::Bool(false)),
            ("strokeStyleLineDashSet".into(), Value::List(Vec::new())),
            ("strokeStyleBlendMode".into(), en("BlnM", "Nrml")),
            ("strokeStyleOpacity".into(), unit("#Prc", s.color[3].clamp(0.0, 1.0) as f64 * 100.0)),
            (
                "strokeStyleContent".into(),
                Value::Desc(Desc { class: "solidColorLayer".into(), items: vec![("Clr ".into(), Value::Desc(rgb_desc(s.color)))] }),
            ),
            ("strokeStyleResolution".into(), Value::Double(72.0)),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(sel: u16, body: &[u8]) -> Vec<u8> {
        let mut v = sel.to_be_bytes().to_vec();
        v.extend(body);
        v.resize(26, 0);
        v
    }

    fn knot(x: f64, y: f64, (w, h): (f64, f64)) -> Vec<u8> {
        let f = |v: f64| ((v * (1 << 24) as f64) as i32).to_be_bytes();
        let mut b = Vec::new();
        for _ in 0..3 {
            b.extend(f(y / h));
            b.extend(f(x / w));
        }
        b
    }

    fn square(x: f64, y: f64, s: f64, size: (f64, f64)) -> Vec<Vec<u8>> {
        [(x, y), (x + s, y), (x + s, y + s), (x, y + s)].iter().map(|&(a, b)| record(1, &knot(a, b, size))).collect()
    }

    fn mask(paths: &[(i16, Vec<Vec<u8>>)]) -> Vec<u8> {
        let mut b = vec![0, 0, 0, 3, 0, 0, 0, 0];
        for (op, knots) in paths {
            let mut len = (knots.len() as u16).to_be_bytes().to_vec();
            len.extend(op.to_be_bytes());
            b.extend(record(0, &len));
            for k in knots {
                b.extend(k);
            }
        }
        b
    }

    #[test]
    fn square_with_a_hole() {
        let size = (100.0, 50.0);
        let b = mask(&[(1, square(10.0, 10.0, 30.0, size)), (2, square(20.0, 20.0, 10.0, size))]);
        let m = read_vector_mask(&b, (100, 50)).unwrap();
        assert_eq!(m.subpaths.len(), 2);
        assert!((m.subpaths[0].nodes[1].p[0] - 40.0).abs() < 1e-4, "{:?}", m.subpaths[0].nodes);
        let Some(Geometry::Path { d }) = geometry(&m) else { panic!() };
        // Outer square and hole, wound in opposite directions.
        assert!((signed_area(&d).abs() - (900.0 - 100.0)).abs() < 1.0, "{d}");
    }

    /// Sum of the signed areas of the contours of a "M … L … Z" path.
    fn signed_area(d: &str) -> f64 {
        d.split('Z')
            .map(|c| {
                let n: Vec<f64> = c.split(' ').filter_map(|s| s.parse().ok()).collect();
                let p: Vec<[f64; 2]> = n.chunks(2).filter(|c| c.len() == 2).map(|c| [c[0], c[1]]).collect();
                (0..p.len()).map(|i| p[i][0] * p[(i + 1) % p.len()][1] - p[(i + 1) % p.len()][0] * p[i][1]).sum::<f64>() * 0.5
            })
            .sum()
    }

    #[test]
    fn intersect_and_exclude() {
        let size = (64.0, 64.0);
        // Two 8×8 squares overlapping in a 4×4 corner.
        let two = |op| read_vector_mask(&mask(&[(1, square(0.0, 0.0, 8.0, size)), (op, square(4.0, 4.0, 8.0, size))]), (64, 64)).unwrap();
        let area = |op| match geometry(&two(op)) {
            Some(Geometry::Path { d }) => signed_area(&d).abs(),
            g => panic!("{g:?}"),
        };
        assert!((area(3) - 16.0).abs() < 0.1, "intersect");
        assert!((area(0) - (64.0 + 64.0 - 32.0)).abs() < 0.1, "exclude");
        assert!((area(1) - (128.0 - 16.0)).abs() < 0.1, "union");
    }

    #[test]
    fn single_path_stays_editable() {
        let size = (64.0, 64.0);
        let m = read_vector_mask(&mask(&[(-1, square(1.0, 1.0, 8.0, size))]), (64, 64)).unwrap();
        assert!(matches!(geometry(&m), Some(Geometry::Bezier { closed: true, .. })));
    }

    #[test]
    fn written_shape_reads_back() {
        let size = (200u32, 100u32);
        let tri = vec![
            PathNode::corner([10.0, 10.0]),
            PathNode { p: [90.0, 20.0], h_in: [80.0, 5.0], h_out: [100.0, 35.0] },
            PathNode::corner([20.0, 80.0]),
        ];
        let b = write_vector_mask(&[(tri.clone(), true)], size);
        let m = read_vector_mask(&b, size).unwrap();
        let Some(Geometry::Bezier { nodes, closed: true }) = geometry(&m) else { panic!("{m:?}") };
        for (a, b) in nodes.iter().zip(&tri) {
            for (p, q) in [(a.p, b.p), (a.h_in, b.h_in), (a.h_out, b.h_out)] {
                assert!((p[0] - q[0]).abs() < 1e-3 && (p[1] - q[1]).abs() < 1e-3, "{a:?} vs {b:?}");
            }
        }
        let st = StrokeStyle { align: StrokeAlign::Outside, join: StrokeJoin::Bevel, ..StrokeStyle::new([1.0, 0.0, 0.0, 0.5], 3.0) };
        let (back, fill) = stroke(&write_stroke(Some(&st), false), 72.0);
        assert!(!fill);
        let back = back.unwrap();
        assert_eq!((back.align, back.join, back.cap, back.width), (st.align, st.join, st.cap, st.width));
        assert!((back.color[3] - 0.5).abs() < 1e-3 && back.color[0] > 0.99);
        assert_eq!(solid_fill(&write_solid([0.0, 1.0, 0.0, 1.0])).map(|c| c[1] > 0.99), Some(true));
    }
}
