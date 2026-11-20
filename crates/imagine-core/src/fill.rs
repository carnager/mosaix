//! Content-aware fill: fills a hole with texture from the rest of the image
//! so that it continues the surroundings (Wexler, Shechtman & Irani,
//! "Space-Time Completion of Video", with PatchMatch by Barnes et al.).
//!
//! Multi-scale: the hole is first filled smoothly from its border on a
//! coarse level (membrane), then on every level a nearest-neighbour field
//! maps each patch touching the hole to its most similar patch entirely
//! outside the hole, and every hole pixel becomes the average of what the
//! overlapping matched patches say it should be. The result is upsampled as
//! the start of the next finer level.

use rayon::prelude::*;

use crate::heal::membrane;

type Px = [f32; 4];
/// Nearest-neighbour field: source patch centre per pixel ((-1, -1) = none).
type Field = Vec<(i32, i32)>;

/// Patch radius (7 × 7 patches).
const R: i32 = 3;

struct Level {
    w: usize,
    h: usize,
    img: Vec<Px>,
    hole: Vec<bool>,
    /// Pixels usable as source (known, inside the allowed area).
    src: Vec<bool>,
}

impl Level {
    fn idx(&self, x: i32, y: i32) -> usize {
        y as usize * self.w + x as usize
    }

    /// Whether a whole patch around (x, y) lies in source pixels.
    fn valid_source(&self, x: i32, y: i32) -> bool {
        if x < R || y < R || x >= self.w as i32 - R || y >= self.h as i32 - R {
            return false;
        }
        (-R..=R).all(|dy| (-R..=R).all(|dx| self.src[self.idx(x + dx, y + dy)]))
    }

    fn downsample(&self) -> Level {
        let (w, h) = (self.w.div_ceil(2), self.h.div_ceil(2));
        let mut img = vec![[0.0; 4]; w * h];
        let mut hole = vec![false; w * h];
        let mut src = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mut sum, mut n, mut any_hole, mut all_src) = ([0.0f32; 4], 0, false, true);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (fx, fy) = (x * 2 + dx, y * 2 + dy);
                    if fx >= self.w || fy >= self.h {
                        continue;
                    }
                    let i = fy * self.w + fx;
                    any_hole |= self.hole[i];
                    all_src &= self.src[i];
                    if !self.hole[i] {
                        for c in 0..4 {
                            sum[c] += self.img[i][c];
                        }
                        n += 1;
                    }
                }
                let i = y * w + x;
                hole[i] = any_hole;
                src[i] = all_src && !any_hole;
                if n > 0 {
                    img[i] = sum.map(|s| s / n as f32);
                }
            }
        }
        Level { w, h, img, hole, src }
    }
}

/// Small deterministic generator (per pixel and iteration).
fn rand(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Patch distance between the patch at target `t` (current estimate) and
/// source `s`; stops early once above `limit`.
/// With `known_only`, hole pixels of the target are left out (the patch
/// locks onto the known surroundings).
fn dist(l: &Level, t: (i32, i32), s: (i32, i32), limit: f32, known_only: bool) -> f32 {
    let mut d = 0.0;
    for dy in -R..=R {
        let ty = t.1 + dy;
        if ty < 0 || ty >= l.h as i32 {
            continue;
        }
        for dx in -R..=R {
            let tx = t.0 + dx;
            if tx < 0 || tx >= l.w as i32 || (known_only && l.hole[l.idx(tx, ty)]) {
                continue;
            }
            let a = l.img[l.idx(tx, ty)];
            let b = l.img[l.idx(s.0 + dx, s.1 + dy)];
            for c in 0..4 {
                let e = a[c] - b[c];
                d += e * e;
            }
        }
        if d > limit {
            return d;
        }
    }
    d
}

/// One level handed to a [`LevelSolver`] (all per-pixel, row-major).
pub struct LevelView<'a> {
    pub w: usize,
    pub h: usize,
    /// Current estimate; the solver writes the hole pixels.
    pub img: &'a mut [Px],
    pub hole: &'a [bool],
    /// A patch centred here lies entirely in source pixels.
    pub valid: &'a [bool],
    /// Patch centres whose patch touches the hole (the targets).
    pub target: &'a [bool],
    /// Targets that see enough of the known border to anchor on it.
    pub anchor: &'a [bool],
    /// 1 outside the hole, 1.3^−d inside.
    pub conf: &'a [f32],
    /// Match per target (source patch centre); updated by the solver.
    pub field: &'a mut [(i32, i32)],
    /// Patch radius.
    pub radius: i32,
}

/// Runs the EM iterations of a level elsewhere (the GPU).
pub trait LevelSolver {
    /// `em` iterations of PatchMatch + voting. False: not done (CPU fallback).
    fn solve(&mut self, l: LevelView<'_>, em: usize, seed: u64) -> bool;
    /// Levels with fewer targets stay on the CPU.
    fn min_targets(&self) -> usize;
}

/// Whether the patch around each pixel lies entirely in `src` (integral image).
fn valid_centres(w: usize, h: usize, src: &[bool]) -> Vec<bool> {
    let mut sum = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0;
        for x in 0..w {
            row += src[y * w + x] as u32;
            sum[(y + 1) * (w + 1) + x + 1] = sum[y * (w + 1) + x + 1] + row;
        }
    }
    let full = ((2 * R + 1) * (2 * R + 1)) as u32;
    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            if x < R || y < R || x >= w as i32 - R || y >= h as i32 - R {
                return false;
            }
            let (x0, y0, x1, y1) = ((x - R) as usize, (y - R) as usize, (x + R + 1) as usize, (y + R + 1) as usize);
            let s = sum[y1 * (w + 1) + x1] + sum[y0 * (w + 1) + x0] - sum[y0 * (w + 1) + x1] - sum[y1 * (w + 1) + x0];
            s == full
        })
        .collect()
}

/// Fills the `hole` pixels of `img` (row-major `w` × `h`) from patches made
/// only of `allowed` non-hole pixels. Values are compared directly, so pass
/// perceptual (e.g. sRGB-encoded) values. Returns false if there is no source.
pub fn content_aware_fill(w: usize, h: usize, img: &mut [Px], hole: &[bool], allowed: &[bool], seed: u64) -> bool {
    content_aware_fill_with(w, h, img, hole, allowed, seed, None)
}

/// Like [`content_aware_fill`], with large levels solved by `solver`.
pub fn content_aware_fill_with(
    w: usize,
    h: usize,
    img: &mut [Px],
    hole: &[bool],
    allowed: &[bool],
    seed: u64,
    mut solver: Option<&mut dyn LevelSolver>,
) -> bool {
    assert!(img.len() == w * h && hole.len() == w * h && allowed.len() == w * h);
    let src: Vec<bool> = hole.iter().zip(allowed).map(|(&h, &a)| a && !h).collect();
    let mut levels = vec![Level { w, h, img: img.to_vec(), hole: hole.to_vec(), src }];
    // Coarsen until the hole is small (but keep enough source patches).
    loop {
        let l = levels.last().unwrap();
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for (i, &hh) in l.hole.iter().enumerate() {
            if hh {
                let (x, y) = (i % l.w, i / l.w);
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
        if x0 == usize::MAX {
            return true; // nothing to fill
        }
        let extent = (x1 - x0).max(y1 - y0) + 1;
        if extent <= 24 || l.w < 48 || l.h < 48 {
            break;
        }
        let next = l.downsample();
        if !valid_centres(next.w, next.h, &next.src).iter().any(|&v| v) {
            break;
        }
        levels.push(next);
    }
    let n = levels.len();
    // Coarsest: fill ring by ring from the border inwards, so the texture's
    // phase continues from the surroundings.
    if !onion_peel(levels.last_mut().unwrap(), seed) {
        let l = levels.last_mut().unwrap();
        let mut v = l.img.clone();
        membrane(l.w, l.h, &l.hole, &mut v);
        l.img = v;
    }
    // The previous (coarser) level's field and width.
    let mut prev: Option<(Field, usize, usize)> = None;
    for li in (0..n).rev() {
        let l = &mut levels[li];
        let valid = valid_centres(l.w, l.h, &l.src);
        let sources: Vec<(i32, i32)> = (0..l.w * l.h).filter(|&i| valid[i]).map(|i| ((i % l.w) as i32, (i / l.w) as i32)).collect();
        if sources.is_empty() {
            return false;
        }
        // Targets: patch centres whose patch touches the hole (the hole
        // dilated by R, separably: rows, then columns).
        let (lw, lh) = (l.w, l.h);
        let mut rows = vec![false; lw * lh];
        rows.par_chunks_mut(lw).enumerate().for_each(|(y, out)| {
            let row = &l.hole[y * lw..(y + 1) * lw];
            let mut last: i64 = i64::MIN / 2;
            for x in 0..lw {
                if row[x] {
                    last = x as i64;
                }
                // Nearest hole pixel to the left within R …
                out[x] = x as i64 - last <= R as i64;
            }
            let mut next: i64 = i64::MAX / 2;
            for x in (0..lw).rev() {
                if row[x] {
                    next = x as i64;
                }
                out[x] |= next - x as i64 <= R as i64;
            }
        });
        let mut near = vec![false; lw * lh];
        near.par_chunks_mut(lw).enumerate().for_each(|(y, out)| {
            let (y0, y1) = (y.saturating_sub(R as usize), (y + R as usize + 1).min(lh));
            for (x, o) in out.iter_mut().enumerate() {
                *o = (y0..y1).any(|yy| rows[yy * lw + x]);
            }
        });
        let targets: Vec<(i32, i32)> =
            (0..l.h as i32).flat_map(|y| (0..l.w as i32).map(move |x| (x, y))).filter(|&(x, y)| near[y as usize * l.w + x as usize]).collect();
        // Initial field: upsampled from the coarser level, or random.
        let mut field: Vec<(i32, i32)> = vec![(-1, -1); l.w * l.h];
        for (k, &(x, y)) in targets.iter().enumerate() {
            let up = prev.as_ref().and_then(|(f, pw, ph)| {
                let (cx, cy) = ((x / 2) as usize, (y / 2) as usize);
                if cx >= *pw || cy >= *ph {
                    return None;
                }
                let c = f[cy * pw + cx];
                let s = (c.0 * 2 + x % 2, c.1 * 2 + y % 2);
                (c.0 >= 0 && s.0 >= 0 && s.1 >= 0 && (s.0 as usize) < l.w && (s.1 as usize) < l.h && valid[l.idx(s.0, s.1)]).then_some(s)
            });
            let r = rand(seed ^ ((k as u64) << 1) ^ li as u64);
            field[l.idx(x, y)] = up.unwrap_or(sources[(r % sources.len() as u64) as usize]);
        }
        // Confidence: 1 outside the hole, 1.3^−distance inside (Wexler), so
        // patches near the known border dominate the vote.
        let conf = confidence(l);
        let em = if li == 0 { 4 } else { 5 };
        let level_seed = seed ^ ((li as u64) << 32);
        let on_solver = solver.as_ref().is_some_and(|s| targets.len() >= s.min_targets());
        let mut solved = false;
        if on_solver {
            let anchor: Vec<bool> = (0..l.w * l.h).into_par_iter().map(|i| near[i] && anchored(l, ((i % l.w) as i32, (i / l.w) as i32))).collect();
            let (lw, lh) = (l.w, l.h);
            let view = LevelView {
                w: lw,
                h: lh,
                img: &mut l.img,
                hole: &l.hole,
                valid: &valid,
                target: &near,
                anchor: &anchor,
                conf: &conf,
                field: &mut field,
                radius: R,
            };
            solved = solver.as_mut().unwrap().solve(view, em, level_seed);
        }
        if !solved {
            for it in 0..em {
                let d = patchmatch(l, &targets, &sources, &mut field, level_seed ^ ((it as u64) << 16), 4, it == 0);
                vote(l, &targets, &field, &d, &conf);
            }
        }
        prev = Some((field, l.w, l.h));
        // Upsample the estimate into the next finer level's hole.
        if li > 0 {
            let (cw, ch) = (levels[li].w, levels[li].h);
            let coarse = levels[li].img.clone();
            let f = &mut levels[li - 1];
            for y in 0..f.h {
                for x in 0..f.w {
                    let i = y * f.w + x;
                    if f.hole[i] {
                        let (cx, cy) = ((x / 2).min(cw - 1), (y / 2).min(ch - 1));
                        f.img[i] = coarse[cy * cw + cx];
                    }
                }
            }
        }
    }
    img.copy_from_slice(&levels[0].img);
    true
}

/// A target patch sees at least a quarter known pixels.
fn anchored(l: &Level, t: (i32, i32)) -> bool {
    let (mut k, mut n) = (0, 0);
    for dy in -R..=R {
        for dx in -R..=R {
            let (x, y) = (t.0 + dx, t.1 + dy);
            if x >= 0 && y >= 0 && x < l.w as i32 && y < l.h as i32 {
                n += 1;
                k += !l.hole[l.idx(x, y)] as i32;
            }
        }
    }
    k * 4 >= n
}

/// 1 outside the hole, 1.3^−d inside, d = 4-neighbour distance to the border.
fn confidence(l: &Level) -> Vec<f32> {
    let mut dist = vec![u32::MAX; l.w * l.h];
    let mut queue = std::collections::VecDeque::new();
    for (i, &h) in l.hole.iter().enumerate() {
        if !h {
            dist[i] = 0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % l.w, i / l.w);
        let d = dist[i] + 1;
        let mut visit = |j: usize| {
            if dist[j] > d {
                dist[j] = d;
                queue.push_back(j);
            }
        };
        if x > 0 {
            visit(i - 1);
        }
        if x + 1 < l.w {
            visit(i + 1);
        }
        if y > 0 {
            visit(i - l.w);
        }
        if y + 1 < l.h {
            visit(i + l.w);
        }
    }
    dist.iter().map(|&d| if d == u32::MAX { 1e-3 } else { 1.3f32.powi(-(d.min(60) as i32)) }).collect()
}

/// Fills the hole of the (small) coarsest level from the outside in: each
/// pixel on the current front takes the centre of the source patch that best
/// matches its already known neighbourhood. False if there is no source.
fn onion_peel(l: &mut Level, seed: u64) -> bool {
    let mut sources: Vec<(i32, i32)> = (0..l.h as i32)
        .flat_map(|y| (0..l.w as i32).map(move |x| (x, y)))
        .filter(|&(x, y)| l.valid_source(x, y))
        .collect();
    if sources.is_empty() {
        return false;
    }
    // Keep the exhaustive search bounded.
    const MAX_SOURCES: usize = 6000;
    if sources.len() > MAX_SOURCES {
        let n = sources.len();
        sources = (0..MAX_SOURCES).map(|k| sources[(rand(seed ^ k as u64 ^ 0x5eed) % n as u64) as usize]).collect();
    }
    let mut known: Vec<bool> = l.hole.iter().map(|&h| !h).collect();
    loop {
        let front: Vec<(i32, i32)> = (0..l.h as i32)
            .flat_map(|y| (0..l.w as i32).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let i = l.idx(x, y);
                !known[i]
                    && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dy)| {
                        let (nx, ny) = (x + dx, y + dy);
                        nx >= 0 && ny >= 0 && nx < l.w as i32 && ny < l.h as i32 && known[l.idx(nx, ny)]
                    })
            })
            .collect();
        if front.is_empty() {
            break;
        }
        let picks: Vec<Px> = front
            .par_iter()
            .map(|&t| {
                let mut best = (f32::MAX, sources[0]);
                for &s in &sources {
                    let (mut d, mut n) = (0.0f32, 0u32);
                    for dy in -R..=R {
                        let ty = t.1 + dy;
                        if ty < 0 || ty >= l.h as i32 {
                            continue;
                        }
                        for dx in -R..=R {
                            let tx = t.0 + dx;
                            if tx < 0 || tx >= l.w as i32 || !known[l.idx(tx, ty)] {
                                continue;
                            }
                            let a = l.img[l.idx(tx, ty)];
                            let b = l.img[l.idx(s.0 + dx, s.1 + dy)];
                            for c in 0..4 {
                                let e = a[c] - b[c];
                                d += e * e;
                            }
                            n += 1;
                        }
                    }
                    let d = d / n.max(1) as f32;
                    if d < best.0 {
                        best = (d, s);
                    }
                }
                l.img[l.idx(best.1 .0, best.1 .1)]
            })
            .collect();
        for (&t, v) in front.iter().zip(picks) {
            let i = l.idx(t.0, t.1);
            l.img[i] = v;
            known[i] = true;
        }
    }
    true
}

/// PatchMatch iterations: propagation from the neighbours' matches and a
/// random search around the current match (parallel, Jacobi style).
/// Returns the patch distance of every target's match.
fn patchmatch(l: &Level, targets: &[(i32, i32)], sources: &[(i32, i32)], field: &mut [(i32, i32)], seed: u64, iters: usize, anchor: bool) -> Vec<f32> {
    let w = l.w;
    // Anchoring: targets that see enough of the known border compare only that.
    let known_only: Vec<bool> = targets.iter().map(|&t| anchor && anchored(l, t)).collect();
    let mut best: Vec<f32> =
        targets.par_iter().enumerate().map(|(k, &t)| dist(l, t, field[t.1 as usize * w + t.0 as usize], f32::MAX, known_only[k])).collect();
    let max_r = l.w.max(l.h) as f32;
    for it in 0..iters {
        let step: i32 = if it % 2 == 0 { 1 } else { -1 };
        let cur: &[(i32, i32)] = field;
        let updated: Vec<((i32, i32), f32)> = targets
            .par_iter()
            .enumerate()
            .map(|(k, &t)| {
                let i = t.1 as usize * w + t.0 as usize;
                let (mut bs, mut bd) = (cur[i], best[k]);
                let try_s = |s: (i32, i32), bs: &mut (i32, i32), bd: &mut f32| {
                    if s != *bs && l.valid_source(s.0, s.1) {
                        let d = dist(l, t, s, *bd, known_only[k]);
                        if d < *bd {
                            (*bs, *bd) = (s, d);
                        }
                    }
                };
                // Propagation: a neighbour's match, shifted back.
                for (dx, dy) in [(step, 0), (0, step), (-step, 0), (0, -step)] {
                    let (nx, ny) = (t.0 - dx, t.1 - dy);
                    if nx < 0 || ny < 0 || nx >= l.w as i32 || ny >= l.h as i32 {
                        continue;
                    }
                    let ns = cur[ny as usize * w + nx as usize];
                    if ns.0 >= 0 {
                        try_s((ns.0 + dx, ns.1 + dy), &mut bs, &mut bd);
                    }
                }
                // Random search with shrinking radius, plus one global sample.
                let mut r = max_r;
                let mut n = 0u64;
                while r >= 1.0 {
                    let z = rand(seed ^ ((k as u64) << 20) ^ ((it as u64) << 8) ^ n);
                    let ox = ((z & 0xffff) as f32 / 65535.0 * 2.0 - 1.0) * r;
                    let oy = (((z >> 16) & 0xffff) as f32 / 65535.0 * 2.0 - 1.0) * r;
                    try_s((bs.0 + ox.round() as i32, bs.1 + oy.round() as i32), &mut bs, &mut bd);
                    r *= 0.5;
                    n += 1;
                }
                let z = rand(seed ^ ((k as u64) << 20) ^ ((it as u64) << 8) ^ 0xabcd);
                try_s(sources[(z % sources.len() as u64) as usize], &mut bs, &mut bd);
                (bs, bd)
            })
            .collect();
        for (k, (s, d)) in updated.into_iter().enumerate() {
            let t = targets[k];
            field[t.1 as usize * w + t.0 as usize] = s;
            best[k] = d;
        }
    }
    best
}

/// Every hole pixel becomes the average of the colours the overlapping
/// matched patches propose for it, weighted by how well each patch matched
/// (Wexler et al.: exp(−d / 2σ²), σ² from the 75th percentile).
fn vote(l: &mut Level, targets: &[(i32, i32)], field: &[(i32, i32)], dists: &[f32], conf: &[f32]) {
    let w = l.w;
    let mut sorted = dists.to_vec();
    sorted.sort_by(f32::total_cmp);
    let sigma2 = sorted.get(sorted.len() * 3 / 4).copied().unwrap_or(1.0).max(1e-6);
    let weights: Vec<f32> = dists
        .iter()
        .zip(targets)
        .map(|(&d, &t)| {
            // Mean confidence of the target patch.
            let (mut c, mut n) = (0.0f32, 0.0f32);
            for dy in -R..=R {
                for dx in -R..=R {
                    let (x, y) = (t.0 + dx, t.1 + dy);
                    if x >= 0 && y >= 0 && x < l.w as i32 && y < l.h as i32 {
                        c += conf[y as usize * l.w + x as usize];
                        n += 1.0;
                    }
                }
            }
            (-d / (2.0 * sigma2)).exp().max(1e-6) * (c / n.max(1.0))
        })
        .collect();
    let mut acc = vec![[0.0f32; 5]; l.w * l.h];
    for (k, &t) in targets.iter().enumerate() {
        let s = field[t.1 as usize * w + t.0 as usize];
        let wt = weights[k];
        for dy in -R..=R {
            for dx in -R..=R {
                let (x, y) = (t.0 + dx, t.1 + dy);
                if x < 0 || y < 0 || x >= l.w as i32 || y >= l.h as i32 {
                    continue;
                }
                let i = y as usize * w + x as usize;
                if !l.hole[i] {
                    continue;
                }
                let v = l.img[(s.1 + dy) as usize * w + (s.0 + dx) as usize];
                for c in 0..4 {
                    acc[i][c] += v[c] * wt;
                }
                acc[i][4] += wt;
            }
        }
    }
    for (i, a) in acc.iter().enumerate() {
        if l.hole[i] && a[4] > 0.0 {
            l.img[i] = [a[0] / a[4], a[1] / a[4], a[2] / a[4], a[3] / a[4]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stripes(w: usize, h: usize) -> Vec<Px> {
        (0..w * h).map(|i| if (i % w / 8).is_multiple_of(2) { [0.8, 0.3, 0.2, 1.0] } else { [0.2, 0.3, 0.8, 1.0] }).collect()
    }

    fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<bool> {
        (0..w * h).map(|i| ((i % w) as f32 + 0.5 - cx).hypot((i / w) as f32 + 0.5 - cy) < r).collect()
    }

    #[test]
    fn continues_stripes_through_the_hole() {
        let (w, h) = (240, 200);
        let truth = stripes(w, h);
        let hole = disc(w, h, 120.0, 100.0, 30.0);
        let mut img: Vec<Px> = truth.iter().zip(&hole).map(|(p, &hh)| if hh { [1.0, 1.0, 1.0, 1.0] } else { *p }).collect();
        assert!(content_aware_fill(w, h, &mut img, &hole, &vec![true; w * h], 7));
        let (mut err, mut n) = (0.0, 0);
        for i in 0..w * h {
            if hole[i] {
                err += (0..3).map(|c| (img[i][c] - truth[i][c]).abs()).sum::<f32>() / 3.0;
                n += 1;
            } else {
                assert_eq!(img[i], truth[i], "outside the hole nothing changes");
            }
        }
        let mean = err / n as f32;
        assert!(mean < 0.06, "mean error {mean}");
    }

    #[test]
    fn needs_a_source() {
        let (w, h) = (40, 40);
        let mut img = stripes(w, h);
        let hole = disc(w, h, 20.0, 20.0, 10.0);
        // Only a thin ring is allowed: no 7×7 patch fits.
        let allowed = disc(w, h, 20.0, 20.0, 12.0);
        assert!(!content_aware_fill(w, h, &mut img, &hole, &allowed, 1));
        let none = vec![false; w * h];
        assert!(content_aware_fill(w, h, &mut img, &none, &allowed, 1), "empty hole is trivially done");
    }

    #[test]
    #[ignore = "benchmark"]
    fn fill_speed() {
        for (n, r) in [(800usize, 150.0f32), (2000, 560.0)] {
            let truth = stripes(n, n);
            let hole = disc(n, n, n as f32 / 2.0, n as f32 / 2.0, r);
            let mut img = truth.clone();
            let t = std::time::Instant::now();
            content_aware_fill(n, n, &mut img, &hole, &vec![true; n * n], 3);
            let holes = hole.iter().filter(|&&b| b).count();
            eprintln!("{n}², hole {:.2} MP: {:?}", holes as f64 / 1e6, t.elapsed());
        }
    }
}
