//! Affine and projective transforms (move, scale, rotate, skew, distort,
//! perspective) of the active layer through the
//! warp kernel, with a floating selection, live preview and pixel-exact
//! content bounds.
//!
//! Without a selection, the layer's pixels and its mask are transformed.
//! With a selection, only the selected pixels move ("float"): they are cut
//! out, transformed together with the selection outline and composited over
//! what stays behind – like Photoshop's move / free transform.
//!
//! Content may end up outside the canvas; it is kept (not cut off) and
//! becomes visible again when moved back or the canvas grows.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{DAffine2, DMat3, DVec2, DVec3};
use mosaix_core::doc::{Content, MaskTile};
use mosaix_core::{Document, LayerId, Mask, MaskRef, TILE_SIZE, TileCoord};

use super::{Engine, NONE, StrokeTarget, mask_slot};
use crate::pool::Tile;

/// Resampling filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Interp {
    /// Hard pixels (pixel art, exact integer moves).
    Nearest,
    Bilinear,
    /// Catmull-Rom; downscaling switches to supersampled bilinear.
    #[default]
    Bicubic,
}

impl Interp {
    pub fn key(self) -> &'static str {
        match self {
            Interp::Nearest => "nearest",
            Interp::Bilinear => "bilinear",
            Interp::Bicubic => "bicubic",
        }
    }

    pub fn from_key(s: &str) -> Option<Self> {
        Some(match s {
            "nearest" => Interp::Nearest,
            "bilinear" => Interp::Bilinear,
            "bicubic" => Interp::Bicubic,
            _ => return None,
        })
    }
}

/// A map of document px to document px: affine, or projective for distort
/// and perspective. `p' = M · (x, y, 1)`, divided by the third component;
/// normalised so that the content's own side of the horizon has w > 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Warp(pub DMat3);

impl From<DAffine2> for Warp {
    fn from(a: DAffine2) -> Self {
        Warp(DMat3::from(a))
    }
}

impl Warp {
    pub const IDENTITY: Warp = Warp(DMat3::IDENTITY);

    /// The affine part, if the map has no perspective.
    pub fn affine(&self) -> Option<DAffine2> {
        let m = self.0;
        let r = m.row(2);
        let w = r.z;
        (r.x.abs() <= 1e-12 * w.abs() && r.y.abs() <= 1e-12 * w.abs() && w.abs() > 1e-300).then(|| {
            let m = m * (1.0 / w);
            DAffine2::from_cols(m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate())
        })
    }

    /// The image of `p`, or `None` behind the horizon.
    pub fn apply(&self, p: DVec2) -> Option<DVec2> {
        let v = self.0 * DVec3::new(p.x, p.y, 1.0);
        (v.z > 1e-12).then(|| DVec2::new(v.x / v.z, v.y / v.z))
    }

    pub fn inverse(&self) -> Warp {
        Warp(self.0.inverse())
    }

    /// The map taking the rectangle `(min, max)` to the quadrilateral
    /// `corners` (clockwise from the top left). `None` unless the
    /// quadrilateral is convex and has the rectangle's orientation.
    pub fn from_quad(min: DVec2, max: DVec2, corners: [DVec2; 4]) -> Option<Warp> {
        if !convex(&corners) || (max - min).abs().min_element() < 1e-9 {
            return None;
        }
        // Unit square → quad (Heckbert), then rectangle → unit square.
        let [p0, p1, p2, p3] = corners;
        let s = p0 - p1 + p2 - p3;
        let (d1, d2) = (p1 - p2, p3 - p2);
        let den = d1.x * d2.y - d2.x * d1.y;
        if den.abs() < 1e-12 {
            return None;
        }
        let g = (s.x * d2.y - d2.x * s.y) / den;
        let h = (d1.x * s.y - s.x * d1.y) / den;
        let sq = DMat3::from_cols(
            DVec3::new(p1.x - p0.x + g * p1.x, p1.y - p0.y + g * p1.y, g),
            DVec3::new(p3.x - p0.x + h * p3.x, p3.y - p0.y + h * p3.y, h),
            DVec3::new(p0.x, p0.y, 1.0),
        );
        let size = max - min;
        let unit = DMat3::from_cols(
            DVec3::new(1.0 / size.x, 0.0, 0.0),
            DVec3::new(0.0, 1.0 / size.y, 0.0),
            DVec3::new(-min.x / size.x, -min.y / size.y, 1.0),
        );
        let mut m = sq * unit;
        // w > 0 inside the rectangle (it is, for a convex quad: no sign change).
        let c = (min + max) * 0.5;
        if (m * DVec3::new(c.x, c.y, 1.0)).z < 0.0 {
            m = -m;
        }
        Some(Warp(m))
    }

    fn determinant(&self) -> f64 {
        self.0.determinant()
    }
}

/// Whether the quadrilateral is convex (either orientation) and not degenerate.
pub fn convex(q: &[DVec2; 4]) -> bool {
    let mut sign = 0.0;
    for i in 0..4 {
        let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
        let cross = (b - a).perp_dot(c - b);
        if cross.abs() < 1e-9 {
            return false;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    true
}

/// Per-tile sampling geometry for the warp kernel: the source position of
/// a reference point in the tile and the (projective) change around it.
struct TileGeom {
    s0_int: [i32; 2],
    s0_frac: [f32; 2],
    lin: [f32; 4],
    persp: [f32; 2],
    side: f32,
    /// The reference point, relative to the tile's corner.
    ref_pt: [f32; 2],
    /// Largest source step per destination pixel (for supersampling).
    stretch: f64,
}

fn tile_geom(inv: &Warp, origin: DVec2) -> TileGeom {
    let m = inv.0;
    let ts = TILE_SIZE as f64;
    // The reference: the point of the tile furthest from the horizon (for
    // affine maps all are equal and the corner wins).
    let r = [DVec2::ZERO, DVec2::new(ts, 0.0), DVec2::new(0.0, ts), DVec2::splat(ts), DVec2::splat(ts * 0.5)]
        .into_iter()
        .max_by(|a, b| {
            let w = |p: DVec2| (m * (origin + p).extend(1.0)).z.abs();
            w(*a).total_cmp(&w(*b)).then(std::cmp::Ordering::Greater)
        })
        .unwrap();
    let n0 = m * (origin + r).extend(1.0);
    let w0 = if n0.z.abs() < 1e-300 { 1e-300 } else { n0.z };
    let s0 = DVec2::new(n0.x / w0, n0.y / w0);
    // Columns of the linear part, divided by w0.
    let (ax, ay) = (m.x_axis / w0, m.y_axis / w0);
    let q = DVec2::new(ax.z, ay.z);
    // s(d) = s0 + (M d − s0 (q·d)) / (1 + q·d)
    let cx = DVec2::new(ax.x, ax.y) - s0 * q.x;
    let cy = DVec2::new(ay.x, ay.y) - s0 * q.y;
    let si = s0.floor();
    // The step at the tile's centre (exact for affine maps).
    let d = DVec2::splat(ts * 0.5) - r;
    let den = 1.0 + q.dot(d);
    let stretch = if den * w0.signum() > 1e-9 {
        let n = cx * d.x + cy * d.y;
        let jx = (cx * den - n * q.x) / (den * den);
        let jy = (cy * den - n * q.y) / (den * den);
        jx.length().max(jy.length())
    } else {
        1.0
    };
    TileGeom {
        s0_int: [si.x as i32, si.y as i32],
        s0_frac: [(s0.x - si.x) as f32, (s0.y - si.y) as f32],
        lin: [cx.x as f32, cx.y as f32, cy.x as f32, cy.y as f32],
        persp: [q.x as f32, q.y as f32],
        side: w0.signum() as f32,
        ref_pt: [r.x as f32, r.y as f32],
        stretch,
    }
}

const CONST_TAG: u32 = 0xff00_0000;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(super) struct GridRef {
    base: u32,
    w: u32,
    h: u32,
    _pad: u32,
    x0: i32,
    y0: i32,
    fallback: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct WarpJob {
    s0_int: [i32; 2],
    s0_frac: [f32; 2],
    lin: [f32; 4],
    src: GridRef,
    src_sel: GridRef,
    old_slot: u32,
    old_const: f32,
    dsel_slot: u32,
    dsel_const: f32,
    mode: u32,
    interp: u32,
    samples: u32,
    side: f32,
    persp: [f32; 2],
    ref_pt: [f32; 2],
}

const MODE_REPLACE: u32 = 0;
const MODE_FLOAT: u32 = 1;

/// Tile → slot lookup tables for the warp kernel, packed into one buffer.
#[derive(Default)]
pub(super) struct Grids {
    pub(super) data: Vec<u32>,
}

impl Grids {
    pub(super) fn add(&mut self, entries: &[(TileCoord, u32)], fallback: f32) -> GridRef {
        if entries.is_empty() {
            return GridRef {
                fallback,
                ..Default::default()
            };
        }
        let x0 = entries.iter().map(|e| e.0.x).min().unwrap();
        let y0 = entries.iter().map(|e| e.0.y).min().unwrap();
        let w = (entries.iter().map(|e| e.0.x).max().unwrap() - x0 + 1) as u32;
        let h = (entries.iter().map(|e| e.0.y).max().unwrap() - y0 + 1) as u32;
        let base = self.data.len() as u32;
        self.data.resize(self.data.len() + (w * h) as usize, NONE);
        for (c, v) in entries {
            self.data[(base + (c.y - y0) as u32 * w + (c.x - x0) as u32) as usize] = *v;
        }
        GridRef {
            base,
            w,
            h,
            x0,
            y0,
            fallback,
            ..Default::default()
        }
    }

    pub(super) fn layer(&mut self, tiles: &HashMap<TileCoord, Tile>) -> GridRef {
        let e: Vec<_> = tiles.iter().map(|(c, t)| (*c, t.slot().packed())).collect();
        self.add(&e, 0.0)
    }

    pub(super) fn mask(&mut self, m: &Mask<Tile>) -> GridRef {
        let e: Vec<_> = m
            .tiles
            .iter()
            .map(|(c, t)| {
                (
                    *c,
                    match t {
                        MaskTile::Pixels(t) => t.slot().packed(),
                        MaskTile::Solid(v) => CONST_TAG | *v as u32,
                    },
                )
            })
            .collect();
        self.add(&e, m.default as f32 / 255.0)
    }
}

/// Destination tiles covered by the transformed source tiles (exact
/// parallelogram/tile overlap test, `margin` = filter support in source px).
pub(super) fn footprint(
    src: impl IntoIterator<Item = TileCoord>,
    fwd: DAffine2,
    margin: f64,
) -> BTreeSet<TileCoord> {
    footprint_warp(src, &Warp::from(fwd), margin, None)
}

/// Like [`footprint`] for any [`Warp`]. Source tiles that reach behind the
/// horizon have no bounded image: they cover `clip` (tile rectangle
/// `(x0, y0, x1, y1)`, inclusive), or nothing without one.
pub(super) fn footprint_warp(
    src: impl IntoIterator<Item = TileCoord>,
    fwd: &Warp,
    margin: f64,
    clip: Option<(i64, i64, i64, i64)>,
) -> BTreeSet<TileCoord> {
    let ts = TILE_SIZE as f64;
    let mut out = BTreeSet::new();
    for c in src {
        let (ox, oy) = c.origin();
        let (x0, y0) = (ox as f64 - margin, oy as f64 - margin);
        let (x1, y1) = (ox as f64 + ts + margin, oy as f64 + ts + margin);
        let corners = [
            DVec2::new(x0, y0),
            DVec2::new(x1, y0),
            DVec2::new(x1, y1),
            DVec2::new(x0, y1),
        ];
        let mapped = corners.map(|p| fwd.apply(p));
        let Some(quad) = mapped.iter().copied().collect::<Option<Vec<_>>>() else {
            if let Some((cx0, cy0, cx1, cy1)) = clip {
                for ty in cy0..=cy1 {
                    for tx in cx0..=cx1 {
                        out.insert(TileCoord::new(tx as i32, ty as i32));
                    }
                }
            }
            continue;
        };
        let quad: [DVec2; 4] = [quad[0], quad[1], quad[2], quad[3]];
        let min = quad.iter().fold(DVec2::INFINITY, |a, p| a.min(*p));
        let max = quad.iter().fold(DVec2::NEG_INFINITY, |a, p| a.max(*p));
        let (mut tx0, mut ty0) = ((min.x / ts).floor() as i64, (min.y / ts).floor() as i64);
        let (mut tx1, mut ty1) = (
            ((max.x / ts).ceil() as i64) - 1,
            ((max.y / ts).ceil() as i64) - 1,
        );
        if let Some((cx0, cy0, cx1, cy1)) = clip {
            (tx0, ty0, tx1, ty1) = (tx0.max(cx0), ty0.max(cy0), tx1.min(cx1), ty1.min(cy1));
        }
        // Separating axes: the tile axes plus the normals of the quad's edges.
        let normals: [DVec2; 4] = std::array::from_fn(|i| (quad[(i + 1) % 4] - quad[i]).perp());
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let (bx0, by0) = (tx as f64 * ts, ty as f64 * ts);
                let corners = [
                    DVec2::new(bx0, by0),
                    DVec2::new(bx0 + ts, by0),
                    DVec2::new(bx0 + ts, by0 + ts),
                    DVec2::new(bx0, by0 + ts),
                ];
                // Overlap along the tile axes is given by the bounding box (with
                // a strict test so touching edges do not count).
                if max.x <= bx0 || min.x >= bx0 + ts || max.y <= by0 || min.y >= by0 + ts {
                    continue;
                }
                let separated = normals.iter().any(|n| {
                    if n.length_squared() < 1e-12 {
                        return false;
                    }
                    let (qa, qb) = proj(&quad, *n);
                    let (ta, tb) = proj(&corners, *n);
                    qb <= ta + 1e-9 || tb <= qa + 1e-9
                });
                if !separated {
                    out.insert(TileCoord::new(tx as i32, ty as i32));
                }
            }
        }
    }
    out
}

fn proj(pts: &[DVec2; 4], n: DVec2) -> (f64, f64) {
    pts.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
            let d = p.dot(n);
            (a.min(d), b.max(d))
        })
}

/// Whether `fwd` moves by whole pixels only.
pub fn is_integer_translation(fwd: impl Into<Warp>) -> bool {
    fwd.into()
        .affine()
        .is_some_and(|fwd| fwd.matrix2 == glam::DMat2::IDENTITY && fwd.translation.round() == fwd.translation)
}

/// What a live preview shows.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum PreviewKind {
    Transform(Warp, Interp),
    Filter(mosaix_core::Filter),
    /// The active layer with a smart filter replaced (`Some(index)`) or added.
    SmartFilter(Option<usize>, mosaix_core::Filter),
    /// Maintained by the liquify session itself.
    Liquify,
    /// A vector layer shown with a different description (font picker).
    Vector(mosaix_core::LayerId, Box<mosaix_core::VectorLayer>),
    /// Image › Adjustments: the active layer as it would look with the
    /// adjustment applied inside the selection.
    Adjust(mosaix_core::Adjustment),
}

pub(super) struct Preview {
    pub kind: PreviewKind,
    /// Area (document px) the preview was computed for.
    pub rect: (DVec2, DVec2),
    pub doc: Document<Tile>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoundsJob {
    slot: u32,
    sel_slot: u32,
    sel_const: f32,
    _pad: u32,
}

impl Engine {
    /// The document shown on screen: the transform preview, if any.
    pub(super) fn shown(&self) -> &Document<Tile> {
        match &self.render {
            Some(r) => r,
            None => self.preview.as_ref().map_or(&self.doc, |p| &p.doc),
        }
    }

    /// Computes the transformed document. `only` restricts the work to
    /// destination tiles intersecting a rectangle (preview).
    fn transformed_doc(
        &mut self,
        fwd: Warp,
        interp: Interp,
        use_selection: bool,
        only: Option<(DVec2, DVec2)>,
    ) -> Result<(LayerId, Document<Tile>), &'static str> {
        // A group, or several selected layers: every layer (and a group's
        // own mask) moves along; the selection does not apply.
        let several = self.doc.selected_layers();
        let several = several.len() > 1 && !(use_selection && self.doc.selection.is_some());
        if self.stroke.is_none() && (several || self.doc.active_layer().is_some_and(|l| l.is_group())) {
            let id = self.doc.active;
            let roots = if several { self.doc.selected_layers() } else { vec![id] };
            let mut ids = Vec::new();
            fn collect(ls: &[mosaix_core::Layer<Tile>], out: &mut Vec<LayerId>) {
                for l in ls {
                    out.push(l.id);
                    if let Content::Group(c) = &l.content {
                        collect(c, out);
                    }
                }
            }
            for r in roots {
                ids.push(r);
                if let Some(Content::Group(c)) = self.doc.find(r).map(|l| &l.content) {
                    collect(c, &mut ids);
                }
            }
            let mut doc = self.doc.clone();
            for l in ids {
                self.warp_layer(l, fwd, interp, false, only, &mut doc)?;
            }
            return Ok((id, doc));
        }
        // Text and shape layers keep their description: the transform is composed.
        let vector = self.stroke.is_none()
            && self
                .doc
                .active_layer()
                .is_some_and(|l| l.is_rendered());
        let layer = if vector {
            self.doc.active
        } else {
            self.check_target(StrokeTarget::Pixels)?
        };
        let mut doc = self.doc.clone();
        self.warp_layer(layer, fwd, interp, use_selection && !vector, only, &mut doc)?;
        Ok((layer, doc))
    }

    /// Warps one layer's pixels and mask (or, with `use_selection`, only its
    /// selected pixels plus the selection outline) from `self.doc` into `out`.
    pub(super) fn warp_layer(
        &mut self,
        layer: LayerId,
        fwd: Warp,
        interp: Interp,
        use_selection: bool,
        only: Option<(DVec2, DVec2)>,
        out: &mut Document<Tile>,
    ) -> Result<(), &'static str> {
        if fwd.determinant().abs() < 1e-9 {
            return Err(mosaix_core::tr("transformation is too flat (size 0)"));
        }
        let l = self
            .doc
            .find(layer)
            .ok_or(mosaix_core::tr("layer does not exist"))?;
        let affine = fwd.affine();
        if affine.is_none() && (l.vector().is_some() || l.raw().is_some()) {
            return Err(mosaix_core::tr(
                "distort and perspective work on pixels: rasterize text, shape and RAW layers first",
            ));
        }
        if let Some(v) = l.vector() {
            let fwd = affine.unwrap();
            let mut v = v.clone();
            let m = fwd.matrix2;
            let f = [
                m.x_axis.x,
                m.x_axis.y,
                m.y_axis.x,
                m.y_axis.y,
                fwd.translation.x,
                fwd.translation.y,
            ];
            v.transform = mosaix_core::vector::compose(f, v.transform);
            if let Some(o) = out.find_mut(layer) {
                o.content = mosaix_core::doc::Content::Vector(v);
            }
        }
        if let Some(r) = l.raw() {
            let fwd = affine.unwrap();
            let mut r = r.clone();
            let m = fwd.matrix2;
            let f = [m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y, fwd.translation.x, fwd.translation.y];
            r.transform = mosaix_core::vector::compose(f, r.transform);
            if let Some(o) = out.find_mut(layer) {
                o.content = mosaix_core::doc::Content::Raw(r);
            }
        }
        let has_pixels = l.tiles().is_some();
        let src: HashMap<TileCoord, Tile> = l.tiles().map(|t| (**t).clone()).unwrap_or_default();
        let mask = l.mask.clone();
        let sel = if use_selection {
            self.doc.selection.clone()
        } else {
            None
        };

        let inv = fwd.inverse();
        // Supersampling where the map shrinks (per tile for perspective).
        let samples_for = |stretch: f64| {
            if interp == Interp::Nearest {
                1
            } else {
                (stretch - 0.01).ceil().clamp(1.0, 4.0) as u32
            }
        };
        let max_samples = match affine {
            Some(_) => samples_for(tile_geom(&inv, DVec2::ZERO).stretch),
            None => 4,
        };
        let interp_for = |samples: u32| {
            if samples > 1 && interp == Interp::Bicubic {
                Interp::Bilinear
            } else {
                interp
            }
        };
        let margin = match interp_for(max_samples) {
            Interp::Nearest => 0.0,
            Interp::Bilinear => 1.0,
            Interp::Bicubic => 2.0,
        } + max_samples as f64 * 0.5;
        // Perspective can throw content towards infinity: what lands far
        // outside the canvas is dropped.
        let clip = affine.is_none().then(|| {
            let ts = TILE_SIZE as i64;
            let (w, h) = (out.width as i64, out.height as i64);
            (-w / ts - 4, -h / ts - 4, 2 * w / ts + 4, 2 * h / ts + 4)
        });
        let keep = |c: &TileCoord| {
            only.is_none_or(|(a, b)| {
                let (x, y) = c.origin();
                let ts = TILE_SIZE as f64;
                (x as f64) < b.x
                    && (x as f64 + ts) > a.x
                    && (y as f64) < b.y
                    && (y as f64 + ts) > a.y
            })
        };

        let mut grids = Grids::default();
        let src_grid = grids.layer(&src);
        let float = sel.is_some();
        let sel_grid = sel.as_ref().map(|s| grids.mask(s)).unwrap_or_default();

        let base_job = |c: TileCoord| {
            let (x, y) = c.origin();
            let g = tile_geom(&inv, DVec2::new(x as f64, y as f64));
            let samples = if affine.is_some() { max_samples } else { samples_for(g.stretch) };
            WarpJob {
                s0_int: g.s0_int,
                s0_frac: g.s0_frac,
                lin: g.lin,
                persp: g.persp,
                side: g.side,
                ref_pt: g.ref_pt,
                src: GridRef::default(),
                src_sel: GridRef::default(),
                old_slot: NONE,
                old_const: 0.0,
                dsel_slot: NONE,
                dsel_const: 0.0,
                mode: MODE_REPLACE,
                interp: interp_for(samples) as u32,
                samples,
            }
        };

        // ---- pixels ----
        let moving: Vec<TileCoord> = if float {
            let s = sel.as_ref().unwrap();
            src.keys()
                .copied()
                .filter(|c| !matches!(s.at(*c), MaskRef::Solid(0)))
                .collect()
        } else {
            src.keys().copied().collect()
        };
        let mut dest = footprint_warp(moving.iter().copied(), &fwd, margin, clip);
        if float {
            dest.extend(moving.iter().copied()); // the hole left behind
        }
        let dest: Vec<TileCoord> = dest.into_iter().filter(keep).collect();
        let jobs: Vec<WarpJob> = dest
            .iter()
            .map(|&c| {
                let mut j = base_job(c);
                j.src = src_grid;
                if float {
                    j.mode = MODE_FLOAT;
                    j.src_sel = sel_grid;
                    j.old_slot = src.get(&c).map_or(NONE, |t| t.slot().packed());
                    let (slot, val) = mask_slot(sel.as_ref().map(|s| s.at(c)));
                    j.dsel_slot = slot;
                    j.dsel_const = val;
                }
                j
            })
            .collect();

        // ---- layer mask (whole-layer transforms) / selection outline (float) ----
        let outline = if float { sel.clone() } else { mask.clone() };
        let mut mask_jobs = Vec::new();
        let mut mask_dest = Vec::new();
        if let Some(m) = &outline {
            let g = grids.mask(m);
            let coords: Vec<TileCoord> = m
                .tiles
                .iter()
                .filter(|(_, t)| !matches!(t, MaskTile::Solid(v) if *v == m.default))
                .map(|(c, _)| *c)
                .collect();
            mask_dest = footprint_warp(coords, &fwd, margin, clip)
                .into_iter()
                .filter(keep)
                .collect();
            mask_jobs = mask_dest
                .iter()
                .map(|&c| {
                    let mut j = base_job(c);
                    j.src = g;
                    j
                })
                .collect();
        }

        self.grid_buf
            .write(&self.device, &self.queue, bytemuck::cast_slice(&grids.data));
        let new_pixels = self
            .run_warp(StrokeTarget::Pixels, &jobs)
            .ok_or(mosaix_core::tr("GPU memory full"))?;
        let new_mask = self
            .run_warp(StrokeTarget::Mask, &mask_jobs)
            .ok_or(mosaix_core::tr("GPU memory full"))?;

        let l = out.find_mut(layer).unwrap();
        if has_pixels {
            let mut map: HashMap<TileCoord, Tile> = if float { src } else { HashMap::new() };
            map.extend(dest.into_iter().zip(new_pixels));
            l.content = Content::Pixels(Arc::new(map));
        }
        if let Some(m) = outline {
            let tiles = Arc::new(
                mask_dest
                    .into_iter()
                    .zip(new_mask.into_iter().map(MaskTile::Pixels))
                    .collect(),
            );
            let new = Mask {
                tiles,
                default: m.default,
            };
            if float {
                out.selection = Some(new);
            } else {
                out.find_mut(layer).unwrap().mask = Some(new);
            }
        }
        Ok(())
    }

    /// A mask (e.g. a saved selection) resampled by `fwd`.
    fn warp_mask(&mut self, m: &Mask<Tile>, fwd: Warp, interp: Interp) -> Result<Mask<Tile>, &'static str> {
        let inv = fwd.inverse();
        let mut grids = Grids::default();
        let g = grids.mask(m);
        let coords: Vec<TileCoord> = m
            .tiles
            .iter()
            .filter(|(_, t)| !matches!(t, MaskTile::Solid(v) if *v == m.default))
            .map(|(c, _)| *c)
            .collect();
        let dest: Vec<TileCoord> = footprint_warp(coords, &fwd, 4.0, None).into_iter().collect();
        let jobs: Vec<WarpJob> = dest
            .iter()
            .map(|c| {
                let (x, y) = c.origin();
                let tg = tile_geom(&inv, DVec2::new(x as f64, y as f64));
                let samples = if interp == Interp::Nearest { 1 } else { (tg.stretch - 0.01).ceil().clamp(1.0, 4.0) as u32 };
                WarpJob {
                    s0_int: tg.s0_int,
                    s0_frac: tg.s0_frac,
                    lin: tg.lin,
                    src: g,
                    src_sel: GridRef::default(),
                    old_slot: NONE,
                    old_const: 0.0,
                    dsel_slot: NONE,
                    dsel_const: 0.0,
                    mode: MODE_REPLACE,
                    interp: if samples > 1 && interp == Interp::Bicubic { Interp::Bilinear } else { interp } as u32,
                    samples,
                    side: tg.side,
                    persp: tg.persp,
                    ref_pt: tg.ref_pt,
                }
            })
            .collect();
        self.grid_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&grids.data));
        let tiles = self.run_warp(StrokeTarget::Mask, &jobs).ok_or(mosaix_core::tr("GPU memory full"))?;
        Ok(Mask {
            tiles: Arc::new(dest.into_iter().zip(tiles.into_iter().map(MaskTile::Pixels)).collect()),
            default: m.default,
        })
    }

    /// Crops (or extends) the canvas to `w × h` starting at `(x, y)`; content
    /// outside the new canvas is kept, not deleted. Drops the selection.
    pub fn crop(
        &mut self,
        x: i64,
        y: i64,
        w: u32,
        h: u32,
        label: &str,
    ) -> Result<(), &'static str> {
        self.canvas_op(
            w,
            h,
            DAffine2::from_translation(DVec2::new(-x as f64, -y as f64)),
            Interp::Nearest,
            label,
        )
    }

    /// Scales the whole image (all layers and masks) to `w × h`.
    /// Rotates the whole image by `quarter_turns` × 90° clockwise and/or
    /// mirrors it (`flip_x`: left ↔ right, `flip_y`: top ↔ bottom). Exact (no
    /// resampling); the canvas swaps width and height for odd turns.
    pub fn rotate_image(
        &mut self,
        quarter_turns: i32,
        flip_x: bool,
        flip_y: bool,
        label: &str,
    ) -> Result<(), &'static str> {
        let (w, h) = self.size();
        let (w, h) = (w as f64, h as f64);
        let turns = quarter_turns.rem_euclid(4);
        // Rotation about the origin, then moved back into the canvas.
        let (rot, nw, nh) = match turns {
            1 => (
                DAffine2::from_cols_array(&[0.0, 1.0, -1.0, 0.0, h, 0.0]),
                h,
                w,
            ),
            2 => (
                DAffine2::from_cols_array(&[-1.0, 0.0, 0.0, -1.0, w, h]),
                w,
                h,
            ),
            3 => (
                DAffine2::from_cols_array(&[0.0, -1.0, 1.0, 0.0, 0.0, w]),
                h,
                w,
            ),
            _ => (DAffine2::IDENTITY, w, h),
        };
        let mut flip = DAffine2::IDENTITY;
        if flip_x {
            flip = DAffine2::from_cols_array(&[-1.0, 0.0, 0.0, 1.0, nw, 0.0]) * flip;
        }
        if flip_y {
            flip = DAffine2::from_cols_array(&[1.0, 0.0, 0.0, -1.0, 0.0, nh]) * flip;
        }
        let fwd = flip * rot;
        if fwd == DAffine2::IDENTITY {
            return Ok(());
        }
        self.canvas_op(nw as u32, nh as u32, fwd, Interp::Nearest, label)
    }

    /// Rotates / mirrors the active layer about the centre of its content.
    pub fn rotate_layer(
        &mut self,
        degrees: f64,
        flip_x: bool,
        flip_y: bool,
        label: &str,
    ) -> Result<(), &'static str> {
        let id = self.doc.active;
        let (x0, y0, x1, y1) = self
            .content_bounds(id)
            .ok_or(mosaix_core::tr("the layer is empty"))?;
        let c = DVec2::new((x0 + x1) as f64 * 0.5, (y0 + y1) as f64 * 0.5);
        let mut m = DAffine2::from_angle(degrees.to_radians());
        // Exact quarter turns (no rounding noise in the matrix).
        if (degrees / 90.0).fract() == 0.0 {
            m.matrix2 = glam::DMat2::from_cols_array(&m.matrix2.to_cols_array().map(f64::round));
        }
        let f = DAffine2::from_scale(DVec2::new(
            if flip_x { -1.0 } else { 1.0 },
            if flip_y { -1.0 } else { 1.0 },
        ));
        let lin = m * f;
        // About the centre; for quarter turns keep pixel centres on pixel centres.
        let mut fwd = DAffine2::from_translation(c) * lin * DAffine2::from_translation(-c);
        fwd.translation = fwd.translation.round();
        let interp = if (degrees / 90.0).fract() == 0.0 {
            Interp::Nearest
        } else {
            Interp::Bicubic
        };
        self.transform_layer(fwd, interp, false, label)
    }

    pub fn resize_image(
        &mut self,
        w: u32,
        h: u32,
        interp: Interp,
        label: &str,
    ) -> Result<(), &'static str> {
        let (ow, oh) = self.size();
        let k = DVec2::new(w as f64 / ow as f64, h as f64 / oh as f64);
        self.canvas_op(w, h, DAffine2::from_scale(k), interp, label)
    }

    fn canvas_op(
        &mut self,
        w: u32,
        h: u32,
        fwd: DAffine2,
        interp: Interp,
        label: &str,
    ) -> Result<(), &'static str> {
        let out = self.canvas_doc(w, h, fwd, interp)?;
        self.edit(label, |d| {
            *d = out;
            true
        });
        Ok(())
    }

    /// The document after a canvas operation (every layer moved by `fwd`,
    /// new size), without changing anything yet.
    pub(super) fn canvas_doc(
        &mut self,
        w: u32,
        h: u32,
        fwd: DAffine2,
        interp: Interp,
    ) -> Result<Document<Tile>, &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        if w == 0 || h == 0 || w > 300_000 || h > 300_000 {
            return Err(mosaix_core::tr("invalid image size"));
        }
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        let mut out = self.doc.clone();
        let ts = TILE_SIZE as f64;
        let t = fwd.translation;
        let tile_shift = fwd.matrix2 == glam::DMat2::IDENTITY && (t / ts).round() * ts == t;
        let mut ids = Vec::new();
        self.doc.for_each(&mut |l, _| ids.push(l.id));
        for id in ids {
            if tile_shift {
                // Whole tiles: just renumber them.
                let (dx, dy) = ((t.x / ts) as i32, (t.y / ts) as i32);
                let shift = |c: &TileCoord| TileCoord::new(c.x + dx, c.y + dy);
                let l = out.find_mut(id).unwrap();
                match &mut l.content {
                    Content::Pixels(map) => {
                        *map = Arc::new(map.iter().map(|(c, t)| (shift(c), t.clone())).collect())
                    }
                    Content::Vector(v) => {
                        v.transform = mosaix_core::vector::compose(
                            [1.0, 0.0, 0.0, 1.0, t.x, t.y],
                            v.transform,
                        )
                    }
                    Content::Raw(r) => {
                        r.transform = mosaix_core::vector::compose([1.0, 0.0, 0.0, 1.0, t.x, t.y], r.transform)
                    }
                    _ => {}
                }
                if let Some(m) = &mut l.mask {
                    m.tiles =
                        Arc::new(m.tiles.iter().map(|(c, t)| (shift(c), t.clone())).collect());
                }
            } else {
                self.warp_layer(id, fwd.into(), interp, false, None, &mut out)?;
            }
        }
        // Guides too (as long as they stay horizontal or vertical).
        out.guides = out
            .guides
            .iter()
            .filter_map(|g| {
                let (p, d) = if g.vertical {
                    (DVec2::new(g.pos, 0.0), DVec2::Y)
                } else {
                    (DVec2::new(0.0, g.pos), DVec2::X)
                };
                let (p, d) = (fwd.transform_point2(p), fwd.transform_vector2(d));
                if d.x.abs() < 1e-9 {
                    Some(mosaix_core::Guide { vertical: true, pos: p.x })
                } else if d.y.abs() < 1e-9 {
                    Some(mosaix_core::Guide { vertical: false, pos: p.y })
                } else {
                    None
                }
            })
            .collect();
        // Saved selections move with the image.
        for i in 0..out.channels.len() {
            let m = out.channels[i].mask.clone();
            out.channels[i].mask = if tile_shift {
                let (dx, dy) = ((t.x / ts) as i32, (t.y / ts) as i32);
                Mask {
                    tiles: Arc::new(m.tiles.iter().map(|(c, t)| (TileCoord::new(c.x + dx, c.y + dy), t.clone())).collect()),
                    default: m.default,
                }
            } else {
                self.warp_mask(&m, fwd.into(), interp)?
            };
        }
        out.width = w;
        out.height = h;
        out.selection = None;
        Ok(out)
    }

    /// Transforms the active layer (or the selected part of it) by `fwd`
    /// (document px → document px) as one undo step.
    pub fn transform_active(
        &mut self,
        fwd: impl Into<Warp>,
        interp: Interp,
        label: &str,
    ) -> Result<(), &'static str> {
        self.transform_layer(fwd, interp, true, label)
    }

    /// Like [`Self::transform_active`]; `use_selection = false` transforms the
    /// whole layer even if there is a selection.
    pub fn transform_layer(
        &mut self,
        fwd: impl Into<Warp>,
        interp: Interp,
        use_selection: bool,
        label: &str,
    ) -> Result<(), &'static str> {
        let fwd: Warp = fwd.into();
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        if self.doc.selected_layers().len() <= 1
            && !self
                .doc
                .active_layer()
                .is_some_and(|l| l.is_rendered() || l.is_group())
        {
            self.check_target(StrokeTarget::Pixels)?;
        }
        if fwd.affine() == Some(DAffine2::IDENTITY) {
            return Ok(());
        }
        // Whole-pixel moves are exact with nearest sampling.
        let interp = if is_integer_translation(fwd) {
            Interp::Nearest
        } else {
            interp
        };
        let (_, doc) = self.transformed_doc(fwd, interp, use_selection, None)?;
        self.edit(label, |d| {
            *d = doc;
            true
        });
        Ok(())
    }

    /// Shows the transform live without changing the document (only the
    /// visible area is computed). `None` ends the preview.
    pub fn set_transform_preview(
        &mut self,
        t: Option<(Warp, Interp)>,
    ) -> Result<(), &'static str> {
        match t {
            None => {
                if self.preview.take().is_some() {
                    self.recheck_all = true;
                }
                Ok(())
            }
            Some((fwd, interp)) => {
                let interp = if is_integer_translation(fwd) {
                    Interp::Nearest
                } else {
                    interp
                };
                let kind = PreviewKind::Transform(fwd, interp);
                if self.preview.as_ref().is_some_and(|p| p.kind == kind) {
                    return Ok(());
                }
                self.compute_preview(kind)
            }
        }
    }

    pub fn preview_active(&self) -> bool {
        self.preview.is_some()
    }

    pub(super) fn compute_preview(&mut self, kind: PreviewKind) -> Result<(), &'static str> {
        let (min, max) = self.visible;
        let pad = (max - min) * 0.25;
        let rect = (min - pad, max + pad);
        self.preview = None;
        let doc = match &kind {
            PreviewKind::Transform(fwd, interp) => {
                self.transformed_doc(*fwd, *interp, true, Some(rect))?.1
            }
            PreviewKind::Filter(f) => self.filtered_doc(f, Some(rect))?,
            PreviewKind::Liquify => {
                return Err(mosaix_core::tr("the Liquify preview is not computed here"));
            }
            PreviewKind::Vector(id, v) => {
                let mut d = self.doc.clone();
                let l = d
                    .find_mut(*id)
                    .ok_or(mosaix_core::tr("layer does not exist"))?;
                // Only ever replaces a text or shape layer (never pixels or a group).
                if l.vector().is_none() {
                    return Err(mosaix_core::tr(
                        "the active layer is not a text or shape layer",
                    ));
                }
                l.content = mosaix_core::doc::Content::Vector((**v).clone());
                d
            }
            PreviewKind::Adjust(a) => {
                let layer = self.check_target(StrokeTarget::Pixels)?;
                let mut d = self.doc.clone();
                d.active = layer;
                let id = d.add_adjustment(a.clone());
                let l = d.find_mut(id).unwrap();
                // Clipped: it changes only the layer below, like the baked result.
                l.clip = true;
                l.mask = self.doc.selection.clone();
                l.mask_enabled = true;
                d.active = layer;
                d
            }
            PreviewKind::SmartFilter(index, f) => {
                let layer = self.check_target(StrokeTarget::Pixels)?;
                let mut d = self.doc.clone();
                let fs = &mut d.find_mut(layer).unwrap().filters;
                match index.and_then(|i| fs.get_mut(i)) {
                    Some(sf) => sf.filter = f.clone(),
                    None => fs.push(mosaix_core::SmartFilter::new(f.clone())),
                }
                d
            }
        };
        self.preview = Some(Preview { kind, rect, doc });
        self.recheck_all = true;
        Ok(())
    }

    /// Recomputes the preview when the view moved beyond its area.
    pub(super) fn refresh_preview(&mut self) {
        let Some(p) = &self.preview else { return };
        let (min, max) = self.visible;
        let (a, b) = p.rect;
        if min.x >= a.x && min.y >= a.y && max.x <= b.x && max.y <= b.y {
            return;
        }
        let kind = p.kind.clone();
        if let Err(e) = self.compute_preview(kind) {
            tracing::warn!("{}", mosaix_core::tr!("Preview: {err}", err = e));
            self.preview = None;
        }
    }

    /// Resamples a set of tiles by `fwd` into the destination tiles `dest`
    /// (no selection, no mask).
    pub(super) fn warp_map(
        &mut self,
        src: &HashMap<TileCoord, Tile>,
        fwd: DAffine2,
        interp: Interp,
        samples: u32,
        dest: &[TileCoord],
    ) -> Option<HashMap<TileCoord, Tile>> {
        let mut grids = Grids::default();
        let g = grids.layer(src);
        let inv = fwd.inverse();
        let lin = inv.matrix2;
        let jobs: Vec<WarpJob> = dest
            .iter()
            .map(|c| {
                let (x, y) = c.origin();
                let s0 = inv.transform_point2(DVec2::new(x as f64, y as f64));
                let si = s0.floor();
                WarpJob {
                    s0_int: [si.x as i32, si.y as i32],
                    s0_frac: [(s0.x - si.x) as f32, (s0.y - si.y) as f32],
                    lin: [
                        lin.x_axis.x as f32,
                        lin.x_axis.y as f32,
                        lin.y_axis.x as f32,
                        lin.y_axis.y as f32,
                    ],
                    src: g,
                    src_sel: GridRef::default(),
                    old_slot: NONE,
                    old_const: 0.0,
                    dsel_slot: NONE,
                    dsel_const: 0.0,
                    mode: MODE_REPLACE,
                    interp: interp as u32,
                    samples,
                    side: 1.0,
                    persp: [0.0; 2],
                    ref_pt: [0.0; 2],
                }
            })
            .collect();
        self.grid_buf
            .write(&self.device, &self.queue, bytemuck::cast_slice(&grids.data));
        let tiles = self.run_warp(StrokeTarget::Pixels, &jobs)?;
        Some(dest.iter().copied().zip(tiles).collect())
    }

    /// Renders a layer scaled by `scale` into tile (0, 0) of a new layer-pool
    /// tile (thumbnails). Supersampled, `samples` per axis.
    pub(super) fn downscaled_tile(
        &mut self,
        layer: LayerId,
        scale: f64,
        samples: u32,
    ) -> Option<Tile> {
        let l = self.doc.find(layer)?;
        let tiles: HashMap<TileCoord, Tile> = match l.tiles() {
            Some(t) => (**t).clone(),
            None if l.is_rendered() => (*self.vectors.get(&layer)?.1).clone(),
            None => return None,
        };
        let mut grids = Grids::default();
        let src = grids.layer(&tiles);
        let inv = DAffine2::from_scale(DVec2::splat(1.0 / scale));
        let lin = inv.matrix2;
        let job = WarpJob {
            s0_int: [0, 0],
            s0_frac: [0.0, 0.0],
            lin: [
                lin.x_axis.x as f32,
                lin.x_axis.y as f32,
                lin.y_axis.x as f32,
                lin.y_axis.y as f32,
            ],
            src,
            src_sel: GridRef::default(),
            old_slot: NONE,
            old_const: 0.0,
            dsel_slot: NONE,
            dsel_const: 0.0,
            mode: MODE_REPLACE,
            interp: Interp::Bilinear as u32,
            samples,
            side: 1.0,
            persp: [0.0; 2],
            ref_pt: [0.0; 2],
        };
        self.grid_buf
            .write(&self.device, &self.queue, bytemuck::cast_slice(&grids.data));
        self.run_warp(StrokeTarget::Pixels, &[job])?.pop()
    }

    /// A layer's mask scaled into one mask tile (thumbnail).
    pub(super) fn downscaled_mask(&mut self, layer: LayerId, scale: f64, samples: u32) -> Option<Tile> {
        let m = self.doc.find(layer)?.mask.clone()?;
        let mut grids = Grids::default();
        let src = grids.mask(&m);
        let lin = DAffine2::from_scale(DVec2::splat(1.0 / scale)).matrix2;
        let job = WarpJob {
            s0_int: [0, 0],
            s0_frac: [0.0, 0.0],
            lin: [lin.x_axis.x as f32, lin.x_axis.y as f32, lin.y_axis.x as f32, lin.y_axis.y as f32],
            src,
            src_sel: GridRef::default(),
            old_slot: NONE,
            old_const: 0.0,
            dsel_slot: NONE,
            dsel_const: 0.0,
            mode: MODE_REPLACE,
            interp: Interp::Bilinear as u32,
            samples,
            side: 1.0,
            persp: [0.0; 2],
            ref_pt: [0.0; 2],
        };
        self.grid_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&grids.data));
        self.run_warp(StrokeTarget::Mask, &[job])?.pop()
    }

    /// Pixel-exact bounds `(x0, y0, x1, y1)` (exclusive end) of a layer's
    /// visible pixels, restricted to the selection if there is one.
    pub fn content_bounds(&mut self, layer: LayerId) -> Option<(i64, i64, i64, i64)> {
        let l = self.doc.find(layer)?;
        // A group: the union of what its layers contain.
        if let Content::Group(c) = &l.content {
            let ids: Vec<LayerId> = c.iter().map(|l| l.id).collect();
            return ids.into_iter().filter_map(|id| self.content_bounds(id)).reduce(|a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)));
        }
        if let Some(v) = l.vector() {
            let b = mosaix_vector::bounds(v)?;
            return Some((
                b[0].floor() as i64,
                b[1].floor() as i64,
                b[2].ceil() as i64,
                b[3].ceil() as i64,
            ));
        }
        if let Some(r) = l.raw() {
            let (data, t) = (r.data.clone(), r.transform);
            let (w, h) = if data.is_image() {
                self.smart_size(&data)?
            } else {
                self.raw_summaries.get(&data.id)?.size
            };
            let pts = [(0.0, 0.0), (w as f64, 0.0), (0.0, h as f64), (w as f64, h as f64)]
                .map(|(x, y)| (t[0] * x + t[2] * y + t[4], t[1] * x + t[3] * y + t[5]));
            let f = |g: fn(f64, f64) -> f64, k: usize| pts.iter().map(|p| if k == 0 { p.0 } else { p.1 }).reduce(g).unwrap();
            return Some((
                f(f64::min, 0).floor() as i64,
                f(f64::min, 1).floor() as i64,
                f(f64::max, 0).ceil() as i64,
                f(f64::max, 1).ceil() as i64,
            ));
        }
        let tiles: Vec<(TileCoord, Tile)> =
            l.tiles()?.iter().map(|(c, t)| (*c, t.clone())).collect();
        let jobs: Vec<BoundsJob> = tiles
            .iter()
            .filter(|(c, _)| self.selected_any(*c))
            .map(|(c, t)| {
                let (sel_slot, sel_const) = self.sel_slot(*c);
                BoundsJob {
                    slot: t.slot().packed(),
                    sel_slot,
                    sel_const,
                    _pad: 0,
                }
            })
            .collect();
        let coords: Vec<TileCoord> = tiles
            .iter()
            .filter(|(c, _)| self.selected_any(*c))
            .map(|(c, _)| *c)
            .collect();
        if jobs.is_empty() {
            return None;
        }
        self.rebuild_arrays_bg();
        self.bounds_jobs
            .write(&self.device, &self.queue, bytemuck::cast_slice(&jobs));
        let size = (jobs.len() * 16) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bounds"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bounds readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let off = self.params.push([jobs.len() as u32, 0, 0, 0]);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bounds"),
            layout: &self.bounds_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.bounds_jobs.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params.binding(),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.bounds_pipeline);
            cp.set_bind_group(0, self.arrays_bg.as_ref().unwrap(), &[]);
            cp.set_bind_group(1, &bg, &[off]);
            let n = jobs.len() as u32;
            cp.dispatch_workgroups(n.min(65535), n.div_ceil(65535), 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, size);
        self.params.upload(&self.queue);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map bounds"));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("mapped range");
        let b: &[[i32; 4]] = bytemuck::cast_slice(&data);
        let mut acc: Option<(i64, i64, i64, i64)> = None;
        for (c, r) in coords.iter().zip(b) {
            if r[2] < 0 {
                continue;
            }
            let (ox, oy) = c.origin();
            let bb = (
                ox + r[0] as i64,
                oy + r[1] as i64,
                ox + r[2] as i64 + 1,
                oy + r[3] as i64 + 1,
            );
            acc = Some(match acc {
                None => bb,
                Some(a) => (a.0.min(bb.0), a.1.min(bb.1), a.2.max(bb.2), a.3.max(bb.3)),
            });
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footprint_of_translation_is_shifted_tiles() {
        let fwd = DAffine2::from_translation(DVec2::new(256.0, 0.0));
        let f = footprint([TileCoord::new(0, 0)], fwd, 0.0);
        assert_eq!(
            f.into_iter().collect::<Vec<_>>(),
            vec![TileCoord::new(1, 0)]
        );
        let f = footprint(
            [TileCoord::new(0, 0)],
            DAffine2::from_translation(DVec2::new(10.0, 0.0)),
            0.0,
        );
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn footprint_of_rotation_skips_corner_tiles() {
        // Tile (0, 0) rotated 45° and placed with its centre at the middle of
        // tile (1, 1): its bounding box covers 3×3 tiles, the diamond itself
        // only the middle cross.
        let fwd = DAffine2::from_translation(DVec2::splat(384.0))
            * DAffine2::from_angle(std::f64::consts::FRAC_PI_4)
            * DAffine2::from_translation(-DVec2::splat(128.0));
        let f = footprint([TileCoord::new(0, 0)], fwd, 0.0);
        for c in [(1, 1), (1, 0), (0, 1), (2, 1), (1, 2)] {
            assert!(f.contains(&TileCoord::new(c.0, c.1)), "{c:?} missing");
        }
        assert!(
            !f.contains(&TileCoord::new(0, 0)) && !f.contains(&TileCoord::new(2, 2)),
            "{f:?}"
        );
    }

    #[test]
    fn quad_warp_maps_the_corners() {
        let (min, max) = (DVec2::new(10.0, 20.0), DVec2::new(110.0, 70.0));
        let q = [DVec2::new(0.0, 0.0), DVec2::new(200.0, 30.0), DVec2::new(180.0, 150.0), DVec2::new(20.0, 100.0)];
        let w = Warp::from_quad(min, max, q).unwrap();
        let rect = [min, DVec2::new(max.x, min.y), max, DVec2::new(min.x, max.y)];
        for (r, q) in rect.iter().zip(q) {
            assert!(w.apply(*r).unwrap().distance(q) < 1e-9);
        }
        assert!(w.affine().is_none());
        // A parallelogram is affine.
        let p = [DVec2::new(0.0, 0.0), DVec2::new(100.0, 0.0), DVec2::new(130.0, 50.0), DVec2::new(30.0, 50.0)];
        assert!(Warp::from_quad(min, max, p).unwrap().affine().is_some());
        // Not convex: refused.
        let bad = [DVec2::new(0.0, 0.0), DVec2::new(100.0, 0.0), DVec2::new(20.0, 20.0), DVec2::new(0.0, 100.0)];
        assert!(Warp::from_quad(min, max, bad).is_none());
        // Per-tile geometry reproduces the map.
        let inv = w.inverse();
        let g = tile_geom(&inv, DVec2::new(256.0, 0.0));
        let d = DVec2::new(40.0, 90.0) - DVec2::new(g.ref_pt[0] as f64, g.ref_pt[1] as f64);
        let lin = glam::DMat2::from_cols_array(&g.lin.map(|v| v as f64));
        let den = 1.0 + DVec2::new(g.persp[0] as f64, g.persp[1] as f64).dot(d);
        let s = DVec2::new(g.s0_int[0] as f64 + g.s0_frac[0] as f64, g.s0_int[1] as f64 + g.s0_frac[1] as f64) + lin * d / den;
        let want = inv.apply(DVec2::new(296.0, 90.0)).unwrap();
        assert!(s.distance(want) < 1e-3, "{s} vs {want}");
    }

    #[test]
    fn integer_translation_detection() {
        assert!(is_integer_translation(DAffine2::from_translation(
            DVec2::new(3.0, -7.0)
        )));
        assert!(!is_integer_translation(DAffine2::from_translation(
            DVec2::new(3.5, 0.0)
        )));
        assert!(!is_integer_translation(DAffine2::from_scale(DVec2::splat(
            2.0
        ))));
    }
}
