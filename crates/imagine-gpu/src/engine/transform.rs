//! Affine transforms (move, scale, rotate) of the active layer through the
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
use glam::{DAffine2, DVec2};
use imagine_core::doc::{Content, MaskTile};
use imagine_core::{Document, LayerId, Mask, MaskRef, TILE_SIZE, TileCoord};

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
    _pad: u32,
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
    let ts = TILE_SIZE as f64;
    let mut out = BTreeSet::new();
    let ax = fwd.matrix2.x_axis;
    let ay = fwd.matrix2.y_axis;
    // Separating axes: the tile axes plus the normals of the parallelogram edges.
    let normals = [DVec2::new(-ax.y, ax.x), DVec2::new(-ay.y, ay.x)];
    for c in src {
        let (ox, oy) = c.origin();
        let (x0, y0) = (ox as f64 - margin, oy as f64 - margin);
        let (x1, y1) = (ox as f64 + ts + margin, oy as f64 + ts + margin);
        let quad = [
            DVec2::new(x0, y0),
            DVec2::new(x1, y0),
            DVec2::new(x1, y1),
            DVec2::new(x0, y1),
        ]
        .map(|p| fwd.transform_point2(p));
        let min = quad.iter().fold(DVec2::INFINITY, |a, p| a.min(*p));
        let max = quad.iter().fold(DVec2::NEG_INFINITY, |a, p| a.max(*p));
        let (tx0, ty0) = ((min.x / ts).floor() as i64, (min.y / ts).floor() as i64);
        let (tx1, ty1) = (
            ((max.x / ts).ceil() as i64) - 1,
            ((max.y / ts).ceil() as i64) - 1,
        );
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
pub fn is_integer_translation(fwd: DAffine2) -> bool {
    fwd.matrix2 == glam::DMat2::IDENTITY && fwd.translation.round() == fwd.translation
}

/// What a live preview shows.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum PreviewKind {
    Transform(DAffine2, Interp),
    Filter(imagine_core::Filter),
    /// The active layer with a smart filter replaced (`Some(index)`) or added.
    SmartFilter(Option<usize>, imagine_core::Filter),
    /// Maintained by the liquify session itself.
    Liquify,
    /// A vector layer shown with a different description (font picker).
    Vector(imagine_core::LayerId, Box<imagine_core::VectorLayer>),
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
        fwd: DAffine2,
        interp: Interp,
        use_selection: bool,
        only: Option<(DVec2, DVec2)>,
    ) -> Result<(LayerId, Document<Tile>), &'static str> {
        // A group: every layer in it (and the group's own mask) moves along;
        // the selection does not apply.
        if self.stroke.is_none()
            && let Some(g) = self.doc.active_layer().filter(|l| l.is_group())
        {
            let id = g.id;
            let mut ids = vec![id];
            fn collect(ls: &[imagine_core::Layer<Tile>], out: &mut Vec<LayerId>) {
                for l in ls {
                    out.push(l.id);
                    if let Content::Group(c) = &l.content {
                        collect(c, out);
                    }
                }
            }
            if let Content::Group(c) = &g.content {
                collect(c, &mut ids);
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
        fwd: DAffine2,
        interp: Interp,
        use_selection: bool,
        only: Option<(DVec2, DVec2)>,
        out: &mut Document<Tile>,
    ) -> Result<(), &'static str> {
        if fwd.matrix2.determinant().abs() < 1e-9 {
            return Err(imagine_core::tr("transformation is too flat (size 0)"));
        }
        let l = self
            .doc
            .find(layer)
            .ok_or(imagine_core::tr("layer does not exist"))?;
        if let Some(v) = l.vector() {
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
            v.transform = imagine_core::vector::compose(f, v.transform);
            if let Some(o) = out.find_mut(layer) {
                o.content = imagine_core::doc::Content::Vector(v);
            }
        }
        if let Some(r) = l.raw() {
            let mut r = r.clone();
            let m = fwd.matrix2;
            let f = [m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y, fwd.translation.x, fwd.translation.y];
            r.transform = imagine_core::vector::compose(f, r.transform);
            if let Some(o) = out.find_mut(layer) {
                o.content = imagine_core::doc::Content::Raw(r);
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
        let lin = inv.matrix2;
        let stretch = lin.x_axis.length().max(lin.y_axis.length());
        let mut interp = interp;
        let samples = if interp == Interp::Nearest {
            1
        } else {
            (stretch - 0.01).ceil().clamp(1.0, 4.0) as u32
        };
        if samples > 1 && interp == Interp::Bicubic {
            interp = Interp::Bilinear;
        }
        let margin = match interp {
            Interp::Nearest => 0.0,
            Interp::Bilinear => 1.0,
            Interp::Bicubic => 2.0,
        } + samples as f64 * 0.5;
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
                src: GridRef::default(),
                src_sel: GridRef::default(),
                old_slot: NONE,
                old_const: 0.0,
                dsel_slot: NONE,
                dsel_const: 0.0,
                mode: MODE_REPLACE,
                interp: interp as u32,
                samples,
                _pad: 0,
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
        let mut dest = footprint(moving.iter().copied(), fwd, margin);
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
            mask_dest = footprint(coords, fwd, margin)
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
            .ok_or(imagine_core::tr("GPU memory full"))?;
        let new_mask = self
            .run_warp(StrokeTarget::Mask, &mask_jobs)
            .ok_or(imagine_core::tr("GPU memory full"))?;

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
            .ok_or(imagine_core::tr("the layer is empty"))?;
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
        if self.stroke.is_some() {
            return Err(imagine_core::tr("not possible during a brush stroke"));
        }
        if w == 0 || h == 0 || w > 300_000 || h > 300_000 {
            return Err(imagine_core::tr("invalid image size"));
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
                        v.transform = imagine_core::vector::compose(
                            [1.0, 0.0, 0.0, 1.0, t.x, t.y],
                            v.transform,
                        )
                    }
                    Content::Raw(r) => {
                        r.transform = imagine_core::vector::compose([1.0, 0.0, 0.0, 1.0, t.x, t.y], r.transform)
                    }
                    _ => {}
                }
                if let Some(m) = &mut l.mask {
                    m.tiles =
                        Arc::new(m.tiles.iter().map(|(c, t)| (shift(c), t.clone())).collect());
                }
            } else {
                self.warp_layer(id, fwd, interp, false, None, &mut out)?;
            }
        }
        out.width = w;
        out.height = h;
        out.selection = None;
        self.edit(label, |d| {
            *d = out;
            true
        });
        Ok(())
    }

    /// Transforms the active layer (or the selected part of it) by `fwd`
    /// (document px → document px) as one undo step.
    pub fn transform_active(
        &mut self,
        fwd: DAffine2,
        interp: Interp,
        label: &str,
    ) -> Result<(), &'static str> {
        self.transform_layer(fwd, interp, true, label)
    }

    /// Like [`Self::transform_active`]; `use_selection = false` transforms the
    /// whole layer even if there is a selection.
    pub fn transform_layer(
        &mut self,
        fwd: DAffine2,
        interp: Interp,
        use_selection: bool,
        label: &str,
    ) -> Result<(), &'static str> {
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        if !self
            .doc
            .active_layer()
            .is_some_and(|l| l.is_rendered() || l.is_group())
        {
            self.check_target(StrokeTarget::Pixels)?;
        }
        if fwd == DAffine2::IDENTITY {
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
        t: Option<(DAffine2, Interp)>,
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
                return Err(imagine_core::tr("the Liquify preview is not computed here"));
            }
            PreviewKind::Vector(id, v) => {
                let mut d = self.doc.clone();
                let l = d
                    .find_mut(*id)
                    .ok_or(imagine_core::tr("layer does not exist"))?;
                // Only ever replaces a text or shape layer (never pixels or a group).
                if l.vector().is_none() {
                    return Err(imagine_core::tr(
                        "the active layer is not a text or shape layer",
                    ));
                }
                l.content = imagine_core::doc::Content::Vector((**v).clone());
                d
            }
            PreviewKind::SmartFilter(index, f) => {
                let layer = self.check_target(StrokeTarget::Pixels)?;
                let mut d = self.doc.clone();
                let fs = &mut d.find_mut(layer).unwrap().filters;
                match index.and_then(|i| fs.get_mut(i)) {
                    Some(sf) => sf.filter = f.clone(),
                    None => fs.push(imagine_core::SmartFilter::new(f.clone())),
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
            tracing::warn!("{}", imagine_core::tr!("Preview: {err}", err = e));
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
                    _pad: 0,
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
            _pad: 0,
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
            _pad: 0,
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
            let b = imagine_vector::bounds(v)?;
            return Some((
                b[0].floor() as i64,
                b[1].floor() as i64,
                b[2].ceil() as i64,
                b[3].ceil() as i64,
            ));
        }
        if let Some(r) = l.raw() {
            let (w, h) = self.raw_summaries.get(&r.data.id)?.size;
            let t = r.transform;
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
