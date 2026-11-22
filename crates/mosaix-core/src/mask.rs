//! Masks (selections and layer masks): sparse 8-bit tiles where a tile that
//! has one value everywhere is stored as `Solid` (no memory), plus CPU
//! algorithms to build and modify them: anti-aliased shape rasterisation,
//! combining (add / subtract / intersect), inversion and feathering.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::tile::{TILE_SIZE, TileCoord, grid_size};

const TS: usize = TILE_SIZE as usize;
const TILE_PX: usize = TS * TS;
/// Vertical sub-samples per pixel row for anti-aliasing.
const SUB: usize = 4;

/// CPU-side content of one mask tile.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskData {
    Solid(u8),
    /// TILE_SIZE² values, row-major.
    Pixels(Vec<u8>),
}

impl MaskData {
    pub fn get(&self, x: usize, y: usize) -> u8 {
        match self {
            MaskData::Solid(v) => *v,
            MaskData::Pixels(p) => p[y * TS + x],
        }
    }

    /// Collapses uniform pixel tiles into `Solid`.
    pub fn normalized(self) -> Self {
        match self {
            MaskData::Pixels(p) if p.iter().all(|&v| v == p[0]) => MaskData::Solid(p[0]),
            other => other,
        }
    }

    fn pixels(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            MaskData::Solid(v) => std::borrow::Cow::Owned(vec![*v; TILE_PX]),
            MaskData::Pixels(p) => std::borrow::Cow::Borrowed(p),
        }
    }
}

/// A complete mask on the CPU: listed tiles plus a default for all others.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuMask {
    pub tiles: HashMap<TileCoord, MaskData>,
    pub default: u8,
}

impl CpuMask {
    pub fn filled(v: u8) -> Self {
        Self {
            tiles: HashMap::new(),
            default: v,
        }
    }

    pub fn tile(&self, c: TileCoord) -> MaskData {
        self.tiles
            .get(&c)
            .cloned()
            .unwrap_or(MaskData::Solid(self.default))
    }

    pub fn value_at(&self, x: i64, y: i64) -> u8 {
        let c = TileCoord::containing(x as f64, y as f64);
        let (ox, oy) = c.origin();
        self.tile(c).get((x - ox) as usize, (y - oy) as usize)
    }

    /// Drops tiles equal to the default.
    pub fn compact(&mut self) {
        let d = self.default;
        self.tiles = std::mem::take(&mut self.tiles)
            .into_iter()
            .map(|(c, t)| (c, t.normalized()))
            .filter(|(_, t)| *t != MaskData::Solid(d))
            .collect();
    }

    /// True if no pixel is selected (> 0).
    pub fn is_empty(&self) -> bool {
        self.default == 0 && self.tiles.values().all(|t| *t == MaskData::Solid(0))
    }

    pub fn invert(&self) -> Self {
        let tiles = self
            .tiles
            .iter()
            .map(|(c, t)| {
                let inv = match t {
                    MaskData::Solid(v) => MaskData::Solid(255 - v),
                    MaskData::Pixels(p) => MaskData::Pixels(p.iter().map(|v| 255 - v).collect()),
                };
                (*c, inv)
            })
            .collect();
        Self {
            tiles,
            default: 255 - self.default,
        }
    }
}

/// How a new shape combines with the existing selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CombineOp {
    #[default]
    Replace,
    Add,
    Subtract,
    Intersect,
}

impl CombineOp {
    pub fn key(self) -> &'static str {
        match self {
            Self::Replace => "replace",
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Intersect => "intersect",
        }
    }

    pub fn from_key(k: &str) -> Option<Self> {
        [Self::Replace, Self::Add, Self::Subtract, Self::Intersect]
            .into_iter()
            .find(|o| o.key() == k)
    }

    fn apply(self, old: u8, new: u8) -> u8 {
        match self {
            Self::Replace => new,
            Self::Add => old.max(new),
            Self::Subtract => ((old as u32 * (255 - new as u32) + 127) / 255) as u8,
            Self::Intersect => old.min(new),
        }
    }
}

/// Combines two tiles pixel-wise.
pub fn combine_tile(old: &MaskData, new: &MaskData, op: CombineOp) -> MaskData {
    if let (MaskData::Solid(a), MaskData::Solid(b)) = (old, new) {
        return MaskData::Solid(op.apply(*a, *b));
    }
    let (a, b) = (old.pixels(), new.pixels());
    MaskData::Pixels(
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| op.apply(x, y))
            .collect(),
    )
    .normalized()
}

/// Combines whole masks. Only tiles listed in either input are computed.
pub fn combine(old: &CpuMask, new: &CpuMask, op: CombineOp) -> CpuMask {
    let mut coords: Vec<TileCoord> = old.tiles.keys().chain(new.tiles.keys()).copied().collect();
    coords.sort();
    coords.dedup();
    let tiles = coords
        .into_par_iter()
        .map(|c| (c, combine_tile(&old.tile(c), &new.tile(c), op)))
        .collect();
    let mut m = CpuMask {
        tiles,
        default: op.apply(old.default, new.default),
    };
    m.compact();
    m
}

/// Shapes in document pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Rect {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    },
    Ellipse {
        cx: f64,
        cy: f64,
        rx: f64,
        ry: f64,
    },
    /// Closed polygon, even-odd fill.
    Polygon(Vec<(f64, f64)>),
}

impl Shape {
    pub fn to_polygon(&self) -> Vec<(f64, f64)> {
        match *self {
            Shape::Rect { x, y, w, h } => vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            Shape::Ellipse { cx, cy, rx, ry } => {
                let n = ((rx.abs() + ry.abs()) * 1.5).clamp(32.0, 4096.0) as usize;
                (0..n)
                    .map(|i| {
                        let t = i as f64 / n as f64 * std::f64::consts::TAU;
                        (cx + rx * t.cos(), cy + ry * t.sin())
                    })
                    .collect()
            }
            Shape::Polygon(ref p) => p.clone(),
        }
    }
}

/// Rasterises a shape into an anti-aliased mask (outside = 0, inside = 255),
/// clipped to a `doc_w × doc_h` document. Fully covered tiles become `Solid(255)`.
pub fn rasterize(shape: &Shape, doc_w: u32, doc_h: u32) -> CpuMask {
    let poly = shape.to_polygon();
    let mut mask = CpuMask::filled(0);
    if poly.len() < 3 {
        return mask;
    }
    let (gw, gh) = grid_size(doc_w, doc_h);
    let min_y = poly
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let max_y = poly
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(doc_h as f64);
    let min_x = poly
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let max_x = poly
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(doc_w as f64);
    if min_y >= max_y || min_x >= max_x {
        return mask;
    }
    let ty0 = (min_y / TS as f64).floor() as u32;
    let ty1 = ((max_y / TS as f64).ceil() as u32).min(gh);
    let tx0 = (min_x / TS as f64).floor() as u32;
    let tx1 = ((max_x / TS as f64).ceil() as u32).min(gw);
    let edges: Vec<((f64, f64), (f64, f64))> = (0..poly.len())
        .map(|i| (poly[i], poly[(i + 1) % poly.len()]))
        .filter(|(a, b)| a.1 != b.1)
        .collect();
    let width = ((tx1 - tx0) as usize) * TS;
    let x_base = (tx0 as usize * TS) as f64;

    let rows: Vec<Vec<(TileCoord, MaskData)>> = (ty0..ty1)
        .into_par_iter()
        .map(|ty| {
            // Coverage accumulator for one tile row: width × TS.
            let mut acc = vec![0f32; width * TS];
            let mut xs: Vec<f64> = Vec::new();
            for row in 0..TS {
                let py = (ty as usize * TS + row) as f64;
                for s in 0..SUB {
                    let y = py + (s as f64 + 0.5) / SUB as f64;
                    xs.clear();
                    for &((x0, y0), (x1, y1)) in &edges {
                        if (y >= y0.min(y1)) && (y < y0.max(y1)) {
                            xs.push(x0 + (y - y0) / (y1 - y0) * (x1 - x0));
                        }
                    }
                    xs.sort_by(f64::total_cmp);
                    let line = &mut acc[row * width..(row + 1) * width];
                    for span in xs.as_chunks::<2>().0 {
                        add_span(line, span[0] - x_base, span[1] - x_base, 1.0 / SUB as f32);
                    }
                }
            }
            (tx0..tx1)
                .filter_map(|tx| {
                    let off = (tx - tx0) as usize * TS;
                    let mut px = vec![0u8; TILE_PX];
                    for row in 0..TS {
                        for col in 0..TS {
                            px[row * TS + col] = (acc[row * width + off + col].clamp(0.0, 1.0)
                                * 255.0)
                                .round() as u8;
                        }
                    }
                    let t = MaskData::Pixels(px).normalized();
                    (t != MaskData::Solid(0)).then_some((TileCoord::new(tx as i32, ty as i32), t))
                })
                .collect()
        })
        .collect();
    mask.tiles = rows.into_iter().flatten().collect();
    mask
}

/// Adds horizontal coverage `weight` over [x0, x1) with fractional end pixels.
fn add_span(line: &mut [f32], x0: f64, x1: f64, weight: f32) {
    let w = line.len() as f64;
    let (x0, x1) = (x0.clamp(0.0, w), x1.clamp(0.0, w));
    if x1 <= x0 {
        return;
    }
    let (i0, i1) = (x0.floor() as usize, x1.floor() as usize);
    if i0 == i1 {
        line[i0] += ((x1 - x0) as f32) * weight;
        return;
    }
    line[i0] += ((i0 as f64 + 1.0 - x0) as f32) * weight;
    let end = i1.min(line.len());
    for v in &mut line[i0 + 1..end] {
        *v += weight;
    }
    if i1 < line.len() {
        line[i1] += ((x1 - i1 as f64) as f32) * weight;
    }
}

/// A part of the document gathered into one image (f32, 0…255), for the
/// CPU mask operations: the bounding box of all non-default tiles, grown by
/// `reach` px, clipped to the document.
struct Region {
    img: Vec<f32>,
    w: usize,
    h: usize,
    tx0: i32,
    ty0: i32,
    tx1: i32,
    ty1: i32,
}

fn gather_region(mask: &CpuMask, reach: usize, doc_w: u32, doc_h: u32) -> Option<Region> {
    if mask.tiles.is_empty() {
        return None;
    }
    let (gw, gh) = grid_size(doc_w, doc_h);
    let reach_tiles = reach.div_ceil(TS) as i32;
    let tx0 = (mask.tiles.keys().map(|c| c.x).min().unwrap() - reach_tiles).max(0);
    let ty0 = (mask.tiles.keys().map(|c| c.y).min().unwrap() - reach_tiles).max(0);
    let tx1 = (mask.tiles.keys().map(|c| c.x).max().unwrap() + reach_tiles + 1).min(gw as i32);
    let ty1 = (mask.tiles.keys().map(|c| c.y).max().unwrap() + reach_tiles + 1).min(gh as i32);
    if tx1 <= tx0 || ty1 <= ty0 {
        return None;
    }
    let (w, h) = (((tx1 - tx0) as usize) * TS, ((ty1 - ty0) as usize) * TS);
    let mut img = vec![mask.default as f32; w * h];
    img.par_chunks_mut(w * TS)
        .enumerate()
        .for_each(|(trow, band)| {
            for tcol in 0..(tx1 - tx0) as usize {
                let t = mask.tile(TileCoord::new(tx0 + tcol as i32, ty0 + trow as i32));
                if let MaskData::Solid(v) = t
                    && v == mask.default
                {
                    continue;
                }
                for y in 0..TS {
                    for x in 0..TS {
                        band[y * w + tcol * TS + x] = t.get(x, y) as f32;
                    }
                }
            }
        });
    Some(Region { img, w, h, tx0, ty0, tx1, ty1 })
}

fn scatter_region(r: &Region, default: u8) -> CpuMask {
    let tiles: HashMap<TileCoord, MaskData> = (r.ty0..r.ty1)
        .flat_map(|ty| (r.tx0..r.tx1).map(move |tx| (tx, ty)))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(tx, ty)| {
            let mut px = vec![0u8; TILE_PX];
            let (bx, by) = (((tx - r.tx0) as usize) * TS, ((ty - r.ty0) as usize) * TS);
            for y in 0..TS {
                for x in 0..TS {
                    px[y * TS + x] = r.img[(by + y) * r.w + bx + x].round().clamp(0.0, 255.0) as u8;
                }
            }
            (TileCoord::new(tx, ty), MaskData::Pixels(px).normalized())
        })
        .collect();
    let mut out = CpuMask { tiles, default };
    out.compact();
    out
}

/// The outlines of a mask (at 50 %) as closed polygons in document px,
/// simplified to within `tolerance` px. Outer outlines run clockwise,
/// holes counter-clockwise (so a non-zero fill leaves the holes open).
pub fn trace(mask: &CpuMask, tolerance: f64, doc_w: u32, doc_h: u32) -> Vec<Vec<[f64; 2]>> {
    use std::collections::HashMap as Map;
    let Some(reg) = gather_region(mask, 1, doc_w, doc_h) else {
        return Vec::new();
    };
    let (w, h) = (reg.w as i64, reg.h as i64);
    let (ox, oy) = (reg.tx0 as i64 * TS as i64, reg.ty0 as i64 * TS as i64);
    let inside = |x: i64, y: i64| x >= 0 && y >= 0 && x < w && y < h && reg.img[(y * w + x) as usize] >= 127.5;
    // Directed edges between pixel corners, the inside on the right (screen
    // y down: clockwise around filled areas).
    let mut next: Map<(i64, i64), Vec<(i64, i64)>> = Map::new();
    for y in -1..=h {
        for x in -1..=w {
            if !inside(x, y) {
                continue;
            }
            if !inside(x, y - 1) {
                next.entry((x, y)).or_default().push((x + 1, y));
            }
            if !inside(x + 1, y) {
                next.entry((x + 1, y)).or_default().push((x + 1, y + 1));
            }
            if !inside(x, y + 1) {
                next.entry((x + 1, y + 1)).or_default().push((x, y + 1));
            }
            if !inside(x - 1, y) {
                next.entry((x, y + 1)).or_default().push((x, y));
            }
        }
    }
    let mut loops = Vec::new();
    while let Some(&start) = next.keys().next() {
        let mut pts = vec![start];
        let mut at = start;
        let mut came = (0i64, 0i64);
        while let Some(outs) = next.get_mut(&at) {
            // At a corner touched twice, turn right first (keeps loops apart).
            let pick = if outs.len() > 1 {
                let right = (-came.1, came.0);
                outs.iter().position(|o| (o.0 - at.0, o.1 - at.1) == right).unwrap_or(0)
            } else {
                0
            };
            let to = outs.swap_remove(pick);
            if outs.is_empty() {
                next.remove(&at);
            }
            came = (to.0 - at.0, to.1 - at.1);
            at = to;
            if at == start {
                break;
            }
            pts.push(at);
        }
        if pts.len() >= 3 {
            let poly: Vec<[f64; 2]> = pts.iter().map(|&(x, y)| [(x + ox) as f64, (y + oy) as f64]).collect();
            loops.push(simplify_closed(&poly, tolerance));
        }
    }
    loops.retain(|l| l.len() >= 3);
    loops
}

/// Ramer–Douglas–Peucker on a closed polygon.
fn simplify_closed(p: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    fn rdp(p: &[[f64; 2]], tol: f64, out: &mut Vec<[f64; 2]>) {
        let (a, b) = (p[0], p[p.len() - 1]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let (mut far, mut at) = (0.0, 0);
        for (i, q) in p.iter().enumerate().take(p.len() - 1).skip(1) {
            let d = ((q[0] - a[0]) * dy - (q[1] - a[1]) * dx).abs() / len;
            if d > far {
                (far, at) = (d, i);
            }
        }
        if far > tol && at > 0 {
            rdp(&p[..=at], tol, out);
            rdp(&p[at..], tol, out);
        } else {
            out.push(a);
        }
    }
    if p.len() < 4 {
        return p.to_vec();
    }
    // Split at the point farthest from the first, so both halves are open.
    let far = (1..p.len()).max_by(|&i, &j| {
        let d = |k: usize| (p[k][0] - p[0][0]).powi(2) + (p[k][1] - p[0][1]).powi(2);
        d(i).total_cmp(&d(j))
    }).unwrap_or(p.len() / 2);
    let mut out = Vec::new();
    rdp(&p[..=far], tol, &mut out);
    let mut second: Vec<[f64; 2]> = p[far..].to_vec();
    second.push(p[0]);
    rdp(&second, tol, &mut out);
    // The start point stays even on a straight edge: drop such points.
    let mut i = 0;
    while out.len() > 3 && i < out.len() {
        let n = out.len();
        let (a, q, b) = (out[(i + n - 1) % n], out[i], out[(i + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        if ((q[0] - a[0]) * dy - (q[1] - a[1]) * dx).abs() / len <= tol {
            out.remove(i);
        } else {
            i += 1;
        }
    }
    out
}

/// Mean over a (2r+1)² window (clipped at the borders), via an integral image.
fn box_mean(img: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut sum = vec![0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0f64;
        for x in 0..w {
            row += img[y * w + x] as f64;
            sum[(y + 1) * (w + 1) + x + 1] = sum[y * (w + 1) + x + 1] + row;
        }
    }
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for (x, o) in line.iter_mut().enumerate() {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let s = sum[y1 * (w + 1) + x1] - sum[y0 * (w + 1) + x1] - sum[y1 * (w + 1) + x0] + sum[y0 * (w + 1) + x0];
            *o = (s / ((x1 - x0) * (y1 - y0)) as f64) as f32;
        }
    });
    out
}

/// Select and Mask: within `radius` px of the selection's edge, the mask
/// follows the edges of the image (`guide(x0, y0, w, h)` gives its
/// luminance 0…1 there) – a guided filter, which pulls hair and fur out of
/// the background. `contrast` 0…1 hardens the result.
pub fn refine_edge(
    mask: &CpuMask,
    radius: f64,
    contrast: f32,
    guide: impl FnOnce(i64, i64, usize, usize) -> Vec<f32>,
    doc_w: u32,
    doc_h: u32,
) -> CpuMask {
    let r = radius.round().max(1.0) as usize;
    let Some(mut reg) = gather_region(mask, 2 * r, doc_w, doc_h) else {
        return mask.clone();
    };
    let (w, h) = (reg.w, reg.h);
    let lum = guide(reg.tx0 as i64 * TS as i64, reg.ty0 as i64 * TS as i64, w, h);
    if lum.len() != w * h {
        return mask.clone();
    }
    let p: Vec<f32> = reg.img.iter().map(|v| v / 255.0).collect();
    // The band around the edge (where a window holds both inside and
    // outside) is unknown; elsewhere the selection is trusted.
    let hard: Vec<f32> = p.iter().map(|&v| (v > 0.5) as u8 as f32).collect();
    let mixed = box_mean(&hard, w, h, r);
    let band: Vec<bool> = mixed.iter().map(|&m| m > 0.0 && m < 1.0).collect();
    let fg: Vec<f32> = (0..w * h).map(|k| (hard[k] > 0.5 && !band[k]) as u8 as f32).collect();
    let bg: Vec<f32> = (0..w * h).map(|k| (hard[k] < 0.5 && !band[k]) as u8 as f32).collect();
    // Local brightness of the known inside and outside (a window twice the band).
    let weighted = |known: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let li: Vec<f32> = lum.iter().zip(known).map(|(l, k)| l * k).collect();
        (box_mean(&li, w, h, 2 * r), box_mean(known, w, h, 2 * r))
    };
    let (fi, fn_) = weighted(&fg);
    let (bi, bn) = weighted(&bg);
    let mut alpha = p.clone();
    for k in 0..w * h {
        if !band[k] || fn_[k] <= 0.0 || bn[k] <= 0.0 {
            continue;
        }
        let (f, b) = (fi[k] / fn_[k], bi[k] / bn[k]);
        if (f - b).abs() > 0.04 {
            // How far this pixel is from the background towards the subject.
            alpha[k] = ((lum[k] - b) / (f - b)).clamp(0.0, 1.0);
        }
    }
    // A light guided filter evens out noise along the edge (He et al.).
    let rg = 2;
    let eps = 1e-3f32;
    let mean_i = box_mean(&lum, w, h, rg);
    let mean_p = box_mean(&alpha, w, h, rg);
    let ii: Vec<f32> = lum.iter().map(|v| v * v).collect();
    let ip: Vec<f32> = lum.iter().zip(&alpha).map(|(a, b)| a * b).collect();
    let corr_i = box_mean(&ii, w, h, rg);
    let corr_ip = box_mean(&ip, w, h, rg);
    let mut a = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    for k in 0..w * h {
        let var = corr_i[k] - mean_i[k] * mean_i[k];
        let cov = corr_ip[k] - mean_i[k] * mean_p[k];
        a[k] = cov / (var + eps);
        b[k] = mean_p[k] - a[k] * mean_i[k];
    }
    let (ma, mb) = (box_mean(&a, w, h, rg), box_mean(&b, w, h, rg));
    let gain = 1.0 + 6.0 * contrast.clamp(0.0, 1.0);
    for k in 0..w * h {
        let mut v = if band[k] { (ma[k] * lum[k] + mb[k]).clamp(0.0, 1.0) } else { p[k] };
        if band[k] && contrast > 0.0 {
            v = ((v - 0.5) * gain + 0.5).clamp(0.0, 1.0);
        }
        reg.img[k] = v * 255.0;
    }
    scatter_region(&reg, mask.default)
}

/// Softens a mask with an approximately Gaussian blur (three box passes) of
/// the given radius. Works on the bounding box of all non-default tiles,
/// expanded by the blur reach, clipped to the document.
pub fn feather(mask: &CpuMask, radius: f64, doc_w: u32, doc_h: u32) -> CpuMask {
    if radius < 0.5 || mask.tiles.is_empty() {
        return mask.clone();
    }
    // Box radius for 3 passes ≈ Gaussian with sigma = radius / 2.
    let sigma = radius / 2.0;
    let r = (((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) / 2.0)
        .round()
        .max(1.0) as usize;
    let Some(mut reg) = gather_region(mask, 3 * r, doc_w, doc_h) else {
        return mask.clone();
    };
    let (w, h) = (reg.w, reg.h);
    for _ in 0..3 {
        box_rows(&mut reg.img, w, r, mask.default as f32);
        let mut t = transpose(&reg.img, w, h);
        box_rows(&mut t, h, r, mask.default as f32);
        reg.img = transpose(&t, h, w);
    }
    scatter_region(&reg, mask.default)
}

/// Select › Modify.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Modify {
    /// Grow the selection by this many px (round corners).
    Expand(f64),
    /// Shrink it by this many px.
    Contract(f64),
    /// A soft ring of this width around the selection's edge.
    Border(f64),
    /// Round off corners and remove specks smaller than the radius.
    Smooth(f64),
}

/// Expands, contracts, borders or smooths a mask (edges at 50 %). Soft
/// selections keep their softness where the operation does not reach.
pub fn modify(mask: &CpuMask, op: Modify, doc_w: u32, doc_h: u32) -> CpuMask {
    let amount = match op {
        Modify::Expand(r) | Modify::Contract(r) | Modify::Border(r) | Modify::Smooth(r) => r,
    };
    if amount < 0.5 || mask.tiles.is_empty() {
        return mask.clone();
    }
    if let Modify::Smooth(r) = op {
        // Blur, then back to a crisp (anti-aliased) edge at the same place.
        let mut m = feather(mask, r * 2.0, doc_w, doc_h);
        for t in m.tiles.values_mut() {
            if let MaskData::Pixels(p) = t {
                for v in p.iter_mut() {
                    *v = ((*v as f32 - 127.5) * 4.0 + 127.5).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        m.compact();
        return m;
    }
    let reach = amount.ceil() as usize + 2;
    let Some(mut reg) = gather_region(mask, reach, doc_w, doc_h) else {
        return mask.clone();
    };
    let (w, h) = (reg.w, reg.h);
    let inside: Vec<bool> = reg.img.par_iter().map(|v| *v >= 127.5).collect();
    // Distance from every pixel to the nearest pixel on the other side.
    let d_out = distance(&inside, w, h, true); // outside pixels → nearest inside
    let d_in = distance(&inside, w, h, false); // inside pixels → nearest outside
    let cover = |t: f64| t.clamp(0.0, 1.0) as f32 * 255.0;
    reg.img.par_iter_mut().enumerate().for_each(|(i, v)| {
        *v = match op {
            Modify::Expand(r) => v.max(if inside[i] { 255.0 } else { cover(r + 1.0 - d_out[i]) }),
            Modify::Contract(r) => v.min(if inside[i] { cover(d_in[i] - r) } else { 0.0 }),
            Modify::Border(wd) => {
                // Signed distance to the edge, which lies between pixels.
                let d = if inside[i] { d_in[i] - 0.5 } else { d_out[i] - 0.5 };
                cover(wd * 0.5 + 0.5 - d)
            }
            Modify::Smooth(_) => unreachable!(),
        };
    });
    let default = match op {
        // Far from every edge a ring is empty.
        Modify::Border(_) => 0,
        _ => mask.default,
    };
    let out = scatter_region(&reg, default);
    match op {
        Modify::Border(wd) => feather(&out, (wd * 0.25).max(1.0), doc_w, doc_h),
        _ => out,
    }
}

/// Euclidean distance of every pixel to the nearest pixel whose `inside` is
/// `target` (0 for those pixels themselves). Felzenszwalb & Huttenlocher.
fn distance(inside: &[bool], w: usize, h: usize, target: bool) -> Vec<f64> {
    const INF: f64 = 1e20;
    let mut f: Vec<f64> = inside.par_iter().map(|&v| if v == target { 0.0 } else { INF }).collect();
    // Columns (via transpose), then rows.
    let mut t = transpose_f64(&f, w, h);
    t.par_chunks_mut(h).for_each(dt1d);
    f = transpose_f64(&t, h, w);
    f.par_chunks_mut(w).for_each(dt1d);
    f.par_iter_mut().for_each(|v| *v = v.sqrt());
    f
}

/// Squared distance transform of one line, in place.
fn dt1d(f: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let src = f.to_vec();
    let mut v = vec![0usize; n];
    let mut z = vec![0f64; n + 1];
    let meet = |q: usize, p: usize| ((src[q] + (q * q) as f64) - (src[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64));
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = meet(q, v[k]);
        while s <= z[k] {
            k -= 1; // z[0] = −∞ stops this at 0
            s = meet(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, out) in f.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let d = q as f64 - v[k] as f64;
        *out = d * d + src[v[k]];
    }
}

fn transpose_f64(img: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0f64; w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for y in 0..h {
            col[y] = img[y * w + x];
        }
    });
    out
}

/// In-place running-sum box blur of every row; outside samples equal `edge`.
fn box_rows(img: &mut [f32], w: usize, r: usize, edge: f32) {
    let n = (2 * r + 1) as f32;
    img.par_chunks_mut(w).for_each(|row| {
        let src = row.to_vec();
        let at = |i: isize| {
            if i < 0 || i >= w as isize {
                edge
            } else {
                src[i as usize]
            }
        };
        let mut sum: f32 = (-(r as isize)..=r as isize).map(at).sum();
        for (x, v) in row.iter_mut().enumerate() {
            *v = sum / n;
            sum += at(x as isize + r as isize + 1) - at(x as isize - r as isize);
        }
    });
}

fn transpose(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for y in 0..h {
            col[y] = img[y * w + x];
        }
    });
    out
}

#[cfg(test)]
mod tests {

    #[test]
    fn trace_finds_outline_and_hole() {
        // A 100 × 60 rectangle with a 20 × 20 hole.
        let mut m = CpuMask::filled(0);
        let mut px = vec![0u8; TILE_PX];
        for y in 50..110 {
            for x in 30..130 {
                if !((70..90).contains(&x) && (70..90).contains(&y)) {
                    px[y * TS + x] = 255;
                }
            }
        }
        m.tiles.insert(TileCoord::new(0, 0), MaskData::Pixels(px));
        let mut loops = trace(&m, 0.5, 256, 256);
        assert_eq!(loops.len(), 2, "{loops:?}");
        let area = |l: &Vec<[f64; 2]>| {
            let mut a = 0.0;
            for i in 0..l.len() {
                let (p, q) = (l[i], l[(i + 1) % l.len()]);
                a += p[0] * q[1] - q[0] * p[1];
            }
            a / 2.0
        };
        loops.sort_by(|a, b| area(a).abs().total_cmp(&area(b).abs()));
        let (hole, outer) = (&loops[0], &loops[1]);
        assert_eq!(hole.len(), 4);
        assert_eq!(outer.len(), 4, "a rectangle has four corners: {outer:?}");
        assert!((area(outer).abs() - 6000.0).abs() < 1.0 && (area(hole).abs() - 400.0).abs() < 1.0);
        assert!(area(outer).signum() != area(hole).signum(), "holes run the other way");
    }


    #[test]
    fn refine_edge_follows_the_image() {
        // The image: bright left of x = 150 (with a thin bright "hair" at
        // y = 40 reaching to x = 180), dark right. The selection: a rough box
        // ending at x = 140.
        let (w, h) = (256u32, 256u32);
        let mut m = CpuMask::filled(0);
        let mut px = vec![0u8; TILE_PX];
        for y in 0..256 {
            for x in 0..140 {
                px[y * TS + x] = 255;
            }
        }
        m.tiles.insert(TileCoord::new(0, 0), MaskData::Pixels(px));
        let guide = |x0: i64, y0: i64, gw: usize, gh: usize| {
            let mut v = vec![0.0f32; gw * gh];
            for y in 0..gh {
                for x in 0..gw {
                    let (dx, dy) = (x0 + x as i64, y0 + y as i64);
                    if dx < 150 || ((38..=42).contains(&dy) && dx < 180) {
                        v[y * gw + x] = 1.0;
                    }
                }
            }
            v
        };
        let r = refine_edge(&m, 16.0, 0.5, guide, w, h);
        let at = |x, y| r.tile(TileCoord::new(0, 0)).get(x, y);
        assert!(at(146, 100) > 200, "selection grew to the image edge: {}", at(146, 100));
        assert!(at(154, 100) < 60, "and stops there: {}", at(154, 100));
        assert!(at(50, 100) == 255 && at(220, 100) == 0, "far from the edge nothing changes");
        assert!(at(153, 40) > 180, "the hair is kept: {}", at(153, 40));
        assert!(at(153, 60) < 60, "beside it nothing: {}", at(153, 60));
    }

    use super::*;

    #[test]
    fn rect_is_exact_and_sparse() {
        // A big axis-aligned rect: interior tiles solid, edges anti-aliased.
        let m = rasterize(
            &Shape::Rect {
                x: 10.0,
                y: 20.5,
                w: 1000.0,
                h: 600.0,
            },
            2000,
            2000,
        );
        assert_eq!(m.value_at(500, 300), 255);
        assert_eq!(m.value_at(9, 300), 0);
        assert_eq!(m.value_at(10, 300), 255);
        assert_eq!(m.value_at(1009, 300), 255);
        assert_eq!(m.value_at(1010, 300), 0);
        assert_eq!(m.value_at(500, 20), 128, "half-covered row");
        let solid = m
            .tiles
            .values()
            .filter(|t| matches!(t, MaskData::Solid(255)))
            .count();
        assert!(solid >= 2, "interior tiles are solid");
        assert!(
            !m.tiles.contains_key(&TileCoord::new(5, 5)),
            "outside tiles absent"
        );
    }

    #[test]
    fn ellipse_and_polygon() {
        let m = rasterize(
            &Shape::Ellipse {
                cx: 300.0,
                cy: 300.0,
                rx: 200.0,
                ry: 100.0,
            },
            1000,
            1000,
        );
        assert_eq!(m.value_at(300, 300), 255);
        assert_eq!(m.value_at(300, 410), 0);
        assert_eq!(m.value_at(495, 300), 255);
        assert_eq!(m.value_at(505, 300), 0);
        // Triangle: even-odd fill.
        let t = rasterize(
            &Shape::Polygon(vec![(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]),
            300,
            300,
        );
        assert_eq!(t.value_at(10, 10), 255);
        assert_eq!(t.value_at(90, 90), 0);
        // Degenerate input.
        assert!(rasterize(&Shape::Polygon(vec![(0.0, 0.0), (1.0, 1.0)]), 100, 100).is_empty());
        // Clipped to the document.
        let c = rasterize(
            &Shape::Rect {
                x: -50.0,
                y: -50.0,
                w: 100.0,
                h: 100.0,
            },
            300,
            300,
        );
        assert_eq!(c.value_at(0, 0), 255);
        assert!(c.tiles.keys().all(|k| k.x >= 0 && k.y >= 0));
    }

    #[test]
    fn combine_ops() {
        let a = rasterize(
            &Shape::Rect {
                x: 0.0,
                y: 0.0,
                w: 300.0,
                h: 300.0,
            },
            1000,
            1000,
        );
        let b = rasterize(
            &Shape::Rect {
                x: 200.0,
                y: 0.0,
                w: 300.0,
                h: 300.0,
            },
            1000,
            1000,
        );
        let add = combine(&a, &b, CombineOp::Add);
        assert_eq!((add.value_at(100, 10), add.value_at(450, 10)), (255, 255));
        let sub = combine(&a, &b, CombineOp::Subtract);
        assert_eq!((sub.value_at(100, 10), sub.value_at(250, 10)), (255, 0));
        let int = combine(&a, &b, CombineOp::Intersect);
        assert_eq!(
            (
                int.value_at(100, 10),
                int.value_at(250, 10),
                int.value_at(450, 10)
            ),
            (0, 255, 0)
        );
        let rep = combine(&a, &b, CombineOp::Replace);
        assert_eq!(rep, b);
        // Subtracting everything leaves an empty selection.
        assert!(combine(&a, &CpuMask::filled(255), CombineOp::Subtract).is_empty());
    }

    #[test]
    fn invert_and_compact() {
        let a = rasterize(
            &Shape::Rect {
                x: 0.0,
                y: 0.0,
                w: 300.0,
                h: 300.0,
            },
            1000,
            1000,
        );
        let inv = a.invert();
        assert_eq!(inv.default, 255);
        assert_eq!((inv.value_at(100, 100), inv.value_at(800, 800)), (0, 255));
        assert_eq!(inv.invert(), a);
    }

    #[test]
    fn feather_softens_edges_symmetrically() {
        let a = rasterize(
            &Shape::Rect {
                x: 256.0,
                y: 256.0,
                w: 512.0,
                h: 512.0,
            },
            1024,
            1024,
        );
        let f = feather(&a, 20.0, 1024, 1024);
        let edge = f.value_at(256, 500);
        assert!((100..=155).contains(&edge), "edge ≈ 50 %: {edge}");
        assert!(f.value_at(236, 500) < edge && f.value_at(276, 500) > edge);
        assert_eq!(f.value_at(512, 512), 255, "centre stays selected");
        assert_eq!(f.value_at(10, 10), 0, "far outside stays empty");
    }
}

/// Gathers the document area of a mask into one row-major buffer.
fn gather(mask: &CpuMask, w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![mask.default; w * h];
    let (gw, _) = grid_size(w as u32, h as u32);
    img.par_chunks_mut(w * TS)
        .enumerate()
        .for_each(|(trow, band)| {
            let rows = band.len() / w;
            for tcol in 0..gw as usize {
                let t = mask.tile(TileCoord::new(tcol as i32, trow as i32));
                let x0 = tcol * TS;
                let cols = TS.min(w - x0);
                match &t {
                    MaskData::Solid(v) if *v == mask.default => {}
                    MaskData::Solid(v) => {
                        (0..rows).for_each(|y| band[y * w + x0..][..cols].fill(*v))
                    }
                    MaskData::Pixels(p) => (0..rows).for_each(|y| {
                        band[y * w + x0..][..cols].copy_from_slice(&p[y * TS..][..cols])
                    }),
                }
            }
        });
    img
}

/// Splits a row-major document buffer back into (compacted) tiles.
fn scatter(img: &[u8], w: usize, h: usize) -> CpuMask {
    let (gw, gh) = grid_size(w as u32, h as u32);
    let coords: Vec<(u32, u32)> = (0..gh).flat_map(|y| (0..gw).map(move |x| (x, y))).collect();
    let tiles: HashMap<TileCoord, MaskData> = coords
        .par_iter()
        .filter_map(|&(tx, ty)| {
            let (x0, y0) = (tx as usize * TS, ty as usize * TS);
            let (cols, rows) = (TS.min(w - x0), TS.min(h - y0));
            let mut p = vec![0u8; TILE_PX];
            for y in 0..rows {
                p[y * TS..][..cols].copy_from_slice(&img[(y0 + y) * w + x0..][..cols]);
            }
            let t = MaskData::Pixels(p).normalized();
            (t != MaskData::Solid(0)).then_some((TileCoord::new(tx as i32, ty as i32), t))
        })
        .collect();
    CpuMask { tiles, default: 0 }
}

/// Magic wand: the 4-connected region of fully matching pixels (value 255)
/// around `(x, y)`, plus a one-pixel ring of partially matching neighbours
/// (soft, anti-aliased edge). Restricted to the `w × h` document.
pub fn flood(similar: &CpuMask, x: u32, y: u32, w: u32, h: u32) -> CpuMask {
    let (w, h) = (w as usize, h as usize);
    if x as usize >= w || y as usize >= h {
        return CpuMask::filled(0);
    }
    let src = gather(similar, w, h);
    let mut out = vec![0u8; w * h];
    let core = |i: usize| src[i] == 255;
    if !core(y as usize * w + x as usize) {
        return CpuMask::filled(0);
    }
    // Scanline fill: each stack entry is a seed pixel; fill its whole run,
    // then queue the runs above and below.
    let mut stack = vec![(x as usize, y as usize)];
    while let Some((sx, sy)) = stack.pop() {
        let row = sy * w;
        if out[row + sx] != 0 {
            continue;
        }
        let mut l = sx;
        while l > 0 && core(row + l - 1) && out[row + l - 1] == 0 {
            l -= 1;
        }
        let mut r = sx;
        while r + 1 < w && core(row + r + 1) && out[row + r + 1] == 0 {
            r += 1;
        }
        out[row + l..=row + r].fill(255);
        for ny in [sy.wrapping_sub(1), sy + 1] {
            if ny >= h {
                continue;
            }
            let nrow = ny * w;
            let mut i = l;
            while i <= r {
                if core(nrow + i) && out[nrow + i] == 0 {
                    stack.push((i, ny));
                    // Skip the rest of this run; it is filled from the seed.
                    while i <= r && core(nrow + i) {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
        }
    }
    // Soft ring: partially similar pixels touching the region.
    let ring: Vec<(usize, u8)> = (0..h)
        .into_par_iter()
        .flat_map_iter(|yy| {
            let (src, out) = (&src, &out);
            (0..w).filter_map(move |xx| {
                let i = yy * w + xx;
                let v = src[i];
                if v == 0 || v == 255 || out[i] != 0 {
                    return None;
                }
                let touches = (xx > 0 && out[i - 1] == 255)
                    || (xx + 1 < w && out[i + 1] == 255)
                    || (yy > 0 && out[i - w] == 255)
                    || (yy + 1 < h && out[i + w] == 255);
                touches.then_some((i, v))
            })
        })
        .collect();
    for (i, v) in ring {
        out[i] = v;
    }
    scatter(&out, w, h)
}

#[cfg(test)]
mod flood_tests {
    use super::*;

    fn from_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> CpuMask {
        let img: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        let mut m = scatter(&img, w, h);
        m.compact();
        m
    }

    #[test]
    fn flood_stays_inside_the_region_and_crosses_tiles() {
        // Two separate bands (x < 300 and x > 400) of 255; a wall between.
        let (w, h) = (700, 600);
        let m = from_fn(w, h, |x, _| if !(300..=400).contains(&x) { 255 } else { 0 });
        let f = flood(&m, 10, 10, w as u32, h as u32);
        assert_eq!(f.value_at(299, 599), 255, "crosses tile borders");
        assert_eq!(f.value_at(500, 10), 0, "other region untouched");
        assert_eq!(f.value_at(350, 10), 0);
    }

    #[test]
    fn flood_follows_a_spiral_and_adds_a_soft_ring() {
        // A U-shaped corridor: down along x = 0..10, across y = 590..600, up along x = 500..510.
        let (w, h) = (600, 600);
        let m = from_fn(w, h, |x, y| {
            let in_u = x < 10 || (y >= 590) || (500..510).contains(&x);
            if in_u {
                255
            } else if x == 10 && y < 590 {
                100 // soft edge right of the left arm
            } else {
                0
            }
        });
        let f = flood(&m, 505, 0, w as u32, h as u32);
        assert_eq!(f.value_at(0, 0), 255, "reached through the U");
        assert_eq!(f.value_at(10, 300), 100, "ring keeps its partial value");
        assert_eq!(f.value_at(11, 300), 0);
        assert_eq!(f.value_at(300, 300), 0);
    }

    #[test]
    fn flood_from_a_non_matching_pixel_is_empty() {
        let m = from_fn(300, 300, |x, _| if x < 100 { 255 } else { 0 });
        assert!(flood(&m, 200, 10, 300, 300).is_empty());
    }

    #[test]
    fn modify_expands_contracts_and_borders() {
        let (w, h) = (600, 400);
        let sel = rasterize(&Shape::Rect { x: 100.0, y: 100.0, w: 200.0, h: 100.0 }, w, h);
        let grown = modify(&sel, Modify::Expand(10.0), w, h);
        assert_eq!(grown.value_at(92, 150), 255);
        assert_eq!(grown.value_at(85, 150), 0);
        // Corners are rounded: (92, 92) is 11.3 px from the corner, outside.
        assert_eq!(grown.value_at(92, 92), 0);
        assert_eq!(grown.value_at(97, 97), 255);
        let shrunk = modify(&sel, Modify::Contract(10.0), w, h);
        assert_eq!(shrunk.value_at(105, 150), 0);
        assert_eq!(shrunk.value_at(112, 150), 255);
        assert_eq!(shrunk.value_at(200, 150), 255);
        let ring = modify(&sel, Modify::Border(10.0), w, h);
        assert!(ring.value_at(100, 150) > 200, "on the edge");
        assert_eq!(ring.value_at(200, 150), 0, "middle is empty");
        assert_eq!(ring.value_at(50, 150), 0, "far outside is empty");
        // Smoothing removes a speck smaller than the radius.
        let speck = rasterize(&Shape::Rect { x: 400.0, y: 300.0, w: 3.0, h: 3.0 }, w, h);
        let both = combine(&sel, &speck, CombineOp::Add);
        let smooth = modify(&both, Modify::Smooth(5.0), w, h);
        assert_eq!(smooth.value_at(401, 301), 0);
        assert_eq!(smooth.value_at(200, 150), 255);
    }

    #[test]
    fn distance_transform_is_euclidean() {
        let (w, h) = (7, 5);
        let mut inside = vec![false; w * h];
        inside[2 * w + 3] = true;
        let d = distance(&inside, w, h, true);
        assert_eq!(d[2 * w + 3], 0.0);
        assert!((d[0] - (9.0f64 + 4.0).sqrt()).abs() < 1e-9);
        assert!((d[2 * w + 6] - 3.0).abs() < 1e-9);
    }
}
