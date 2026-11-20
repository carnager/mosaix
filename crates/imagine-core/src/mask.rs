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
        Self { tiles: HashMap::new(), default: v }
    }

    pub fn tile(&self, c: TileCoord) -> MaskData {
        self.tiles.get(&c).cloned().unwrap_or(MaskData::Solid(self.default))
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
        Self { tiles, default: 255 - self.default }
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
        [Self::Replace, Self::Add, Self::Subtract, Self::Intersect].into_iter().find(|o| o.key() == k)
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
    MaskData::Pixels(a.iter().zip(b.iter()).map(|(&x, &y)| op.apply(x, y)).collect()).normalized()
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
    let mut m = CpuMask { tiles, default: op.apply(old.default, new.default) };
    m.compact();
    m
}

/// Shapes in document pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Rect { x: f64, y: f64, w: f64, h: f64 },
    Ellipse { cx: f64, cy: f64, rx: f64, ry: f64 },
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
    let min_y = poly.iter().map(|p| p.1).fold(f64::INFINITY, f64::min).max(0.0);
    let max_y = poly.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max).min(doc_h as f64);
    let min_x = poly.iter().map(|p| p.0).fold(f64::INFINITY, f64::min).max(0.0);
    let max_x = poly.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max).min(doc_w as f64);
    if min_y >= max_y || min_x >= max_x {
        return mask;
    }
    let ty0 = (min_y / TS as f64).floor() as u32;
    let ty1 = ((max_y / TS as f64).ceil() as u32).min(gh);
    let tx0 = (min_x / TS as f64).floor() as u32;
    let tx1 = ((max_x / TS as f64).ceil() as u32).min(gw);
    let edges: Vec<((f64, f64), (f64, f64))> =
        (0..poly.len()).map(|i| (poly[i], poly[(i + 1) % poly.len()])).filter(|(a, b)| a.1 != b.1).collect();
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
                            px[row * TS + col] = (acc[row * width + off + col].clamp(0.0, 1.0) * 255.0).round() as u8;
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

/// Softens a mask with an approximately Gaussian blur (three box passes) of
/// the given radius. Works on the bounding box of all non-default tiles,
/// expanded by the blur reach, clipped to the document.
pub fn feather(mask: &CpuMask, radius: f64, doc_w: u32, doc_h: u32) -> CpuMask {
    if radius < 0.5 || mask.tiles.is_empty() {
        return mask.clone();
    }
    let (gw, gh) = grid_size(doc_w, doc_h);
    // Box radius for 3 passes ≈ Gaussian with sigma = radius / 2.
    let sigma = radius / 2.0;
    let r = (((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
    let reach_tiles = (3 * r).div_ceil(TS) as i32;
    let tx0 = (mask.tiles.keys().map(|c| c.x).min().unwrap() - reach_tiles).max(0);
    let ty0 = (mask.tiles.keys().map(|c| c.y).min().unwrap() - reach_tiles).max(0);
    let tx1 = (mask.tiles.keys().map(|c| c.x).max().unwrap() + reach_tiles + 1).min(gw as i32);
    let ty1 = (mask.tiles.keys().map(|c| c.y).max().unwrap() + reach_tiles + 1).min(gh as i32);
    let (w, h) = (((tx1 - tx0) as usize) * TS, ((ty1 - ty0) as usize) * TS);

    // Gather into one f32 image.
    let mut img = vec![mask.default as f32; w * h];
    img.par_chunks_mut(w * TS).enumerate().for_each(|(trow, band)| {
        for tcol in 0..(tx1 - tx0) as usize {
            let t = mask.tile(TileCoord::new(tx0 + tcol as i32, ty0 + trow as i32));
            if let MaskData::Solid(v) = t
                && v == mask.default {
                    continue;
                }
            for y in 0..TS {
                for x in 0..TS {
                    band[y * w + tcol * TS + x] = t.get(x, y) as f32;
                }
            }
        }
    });
    for _ in 0..3 {
        box_rows(&mut img, w, r, mask.default as f32);
        let mut t = transpose(&img, w, h);
        box_rows(&mut t, h, r, mask.default as f32);
        img = transpose(&t, h, w);
    }
    // Scatter back into tiles.
    let mut tiles = HashMap::new();
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            let mut px = vec![0u8; TILE_PX];
            let (bx, by) = (((tx - tx0) as usize) * TS, ((ty - ty0) as usize) * TS);
            for y in 0..TS {
                for x in 0..TS {
                    px[y * TS + x] = img[(by + y) * w + bx + x].round().clamp(0.0, 255.0) as u8;
                }
            }
            tiles.insert(TileCoord::new(tx, ty), MaskData::Pixels(px));
        }
    }
    let mut out = CpuMask { tiles, default: mask.default };
    out.compact();
    out
}

/// In-place running-sum box blur of every row; outside samples equal `edge`.
fn box_rows(img: &mut [f32], w: usize, r: usize, edge: f32) {
    let n = (2 * r + 1) as f32;
    img.par_chunks_mut(w).for_each(|row| {
        let src = row.to_vec();
        let at = |i: isize| if i < 0 || i >= w as isize { edge } else { src[i as usize] };
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
    use super::*;

    #[test]
    fn rect_is_exact_and_sparse() {
        // A big axis-aligned rect: interior tiles solid, edges anti-aliased.
        let m = rasterize(&Shape::Rect { x: 10.0, y: 20.5, w: 1000.0, h: 600.0 }, 2000, 2000);
        assert_eq!(m.value_at(500, 300), 255);
        assert_eq!(m.value_at(9, 300), 0);
        assert_eq!(m.value_at(10, 300), 255);
        assert_eq!(m.value_at(1009, 300), 255);
        assert_eq!(m.value_at(1010, 300), 0);
        assert_eq!(m.value_at(500, 20), 128, "half-covered row");
        let solid = m.tiles.values().filter(|t| matches!(t, MaskData::Solid(255))).count();
        assert!(solid >= 2, "interior tiles are solid");
        assert!(!m.tiles.contains_key(&TileCoord::new(5, 5)), "outside tiles absent");
    }

    #[test]
    fn ellipse_and_polygon() {
        let m = rasterize(&Shape::Ellipse { cx: 300.0, cy: 300.0, rx: 200.0, ry: 100.0 }, 1000, 1000);
        assert_eq!(m.value_at(300, 300), 255);
        assert_eq!(m.value_at(300, 410), 0);
        assert_eq!(m.value_at(495, 300), 255);
        assert_eq!(m.value_at(505, 300), 0);
        // Triangle: even-odd fill.
        let t = rasterize(&Shape::Polygon(vec![(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]), 300, 300);
        assert_eq!(t.value_at(10, 10), 255);
        assert_eq!(t.value_at(90, 90), 0);
        // Degenerate input.
        assert!(rasterize(&Shape::Polygon(vec![(0.0, 0.0), (1.0, 1.0)]), 100, 100).is_empty());
        // Clipped to the document.
        let c = rasterize(&Shape::Rect { x: -50.0, y: -50.0, w: 100.0, h: 100.0 }, 300, 300);
        assert_eq!(c.value_at(0, 0), 255);
        assert!(c.tiles.keys().all(|k| k.x >= 0 && k.y >= 0));
    }

    #[test]
    fn combine_ops() {
        let a = rasterize(&Shape::Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 }, 1000, 1000);
        let b = rasterize(&Shape::Rect { x: 200.0, y: 0.0, w: 300.0, h: 300.0 }, 1000, 1000);
        let add = combine(&a, &b, CombineOp::Add);
        assert_eq!((add.value_at(100, 10), add.value_at(450, 10)), (255, 255));
        let sub = combine(&a, &b, CombineOp::Subtract);
        assert_eq!((sub.value_at(100, 10), sub.value_at(250, 10)), (255, 0));
        let int = combine(&a, &b, CombineOp::Intersect);
        assert_eq!((int.value_at(100, 10), int.value_at(250, 10), int.value_at(450, 10)), (0, 255, 0));
        let rep = combine(&a, &b, CombineOp::Replace);
        assert_eq!(rep, b);
        // Subtracting everything leaves an empty selection.
        assert!(combine(&a, &CpuMask::filled(255), CombineOp::Subtract).is_empty());
    }

    #[test]
    fn invert_and_compact() {
        let a = rasterize(&Shape::Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 }, 1000, 1000);
        let inv = a.invert();
        assert_eq!(inv.default, 255);
        assert_eq!((inv.value_at(100, 100), inv.value_at(800, 800)), (0, 255));
        assert_eq!(inv.invert(), a);
    }

    #[test]
    fn feather_softens_edges_symmetrically() {
        let a = rasterize(&Shape::Rect { x: 256.0, y: 256.0, w: 512.0, h: 512.0 }, 1024, 1024);
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
    img.par_chunks_mut(w * TS).enumerate().for_each(|(trow, band)| {
        let rows = band.len() / w;
        for tcol in 0..gw as usize {
            let t = mask.tile(TileCoord::new(tcol as i32, trow as i32));
            let x0 = tcol * TS;
            let cols = TS.min(w - x0);
            match &t {
                MaskData::Solid(v) if *v == mask.default => {}
                MaskData::Solid(v) => (0..rows).for_each(|y| band[y * w + x0..][..cols].fill(*v)),
                MaskData::Pixels(p) => (0..rows).for_each(|y| band[y * w + x0..][..cols].copy_from_slice(&p[y * TS..][..cols])),
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
        let img: Vec<u8> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect();
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
}
