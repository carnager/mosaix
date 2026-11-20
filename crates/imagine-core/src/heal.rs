//! Healing: the membrane (harmonic) interpolation used by the healing brush.
//! Inside a region the result is `source + h`, where `h` is smooth
//! (Laplace's equation) and equals `destination − source` on the region's
//! border, so the copied texture takes on the surrounding colour and
//! brightness without a seam (Pérez et al., "Poisson Image Editing").
//!
//! Solved with a cascadic multigrid: the problem is first solved on a
//! coarser grid, upsampled as the starting guess, and refined with
//! red-black SOR sweeps (parallel over rows).

use rayon::prelude::*;

type Px = [f32; 4];

/// Fills the `inside` pixels of `v` (row-major, `w` × `h`) with the
/// harmonic interpolation of the values of all other pixels. Pixels at the
/// array border are treated as reflecting if they are inside.
pub fn membrane(w: usize, h: usize, inside: &[bool], v: &mut [Px]) {
    assert_eq!(inside.len(), w * h);
    assert_eq!(v.len(), w * h);
    let n_inside = inside.iter().filter(|&&b| b).count();
    if n_inside == 0 {
        return;
    }
    let coarse = w >= 8 && h >= 8 && n_inside > 64;
    if coarse {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut cin = vec![false; cw * ch];
        let mut cv = vec![[0.0f32; 4]; cw * ch];
        for cy in 0..ch {
            for cx in 0..cw {
                let mut sum = [0.0f32; 4];
                let mut n = 0;
                let mut all_inside = true;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (x, y) = (cx * 2 + dx, cy * 2 + dy);
                    if x >= w || y >= h {
                        continue;
                    }
                    let i = y * w + x;
                    if inside[i] {
                        continue;
                    }
                    all_inside = false;
                    for c in 0..4 {
                        sum[c] += v[i][c];
                    }
                    n += 1;
                }
                let ci = cy * cw + cx;
                cin[ci] = all_inside;
                if n > 0 {
                    cv[ci] = sum.map(|s| s / n as f32);
                }
            }
        }
        membrane(cw, ch, &cin, &mut cv);
        // Bilinear upsampling of the coarse solution as the starting guess.
        let at = |x: isize, y: isize| {
            cv[(y.clamp(0, ch as isize - 1) as usize) * cw + x.clamp(0, cw as isize - 1) as usize]
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if !inside[i] {
                    continue;
                }
                let (gx, gy) = ((x as f32 + 0.5) / 2.0 - 0.5, (y as f32 + 0.5) / 2.0 - 0.5);
                let (fx, fy) = (gx.floor(), gy.floor());
                let (tx, ty) = (gx - fx, gy - fy);
                let (x0, y0) = (fx as isize, fy as isize);
                let (a, b, c, d) = (
                    at(x0, y0),
                    at(x0 + 1, y0),
                    at(x0, y0 + 1),
                    at(x0 + 1, y0 + 1),
                );
                for k in 0..4 {
                    let top = a[k] + (b[k] - a[k]) * tx;
                    let bot = c[k] + (d[k] - c[k]) * tx;
                    v[i][k] = top + (bot - top) * ty;
                }
            }
        }
    } else {
        // Coarsest level: start from the mean of the known values.
        let (mut sum, mut n) = ([0.0f32; 4], 0usize);
        for (p, &ins) in v.iter().zip(inside) {
            if !ins {
                for c in 0..4 {
                    sum[c] += p[c];
                }
                n += 1;
            }
        }
        let mean = sum.map(|s| s / n.max(1) as f32);
        for (p, &ins) in v.iter_mut().zip(inside) {
            if ins {
                *p = mean;
            }
        }
    }
    let sweeps = if coarse { 30 } else { 200 };
    relax(w, h, inside, v, sweeps);
}

/// Red-black successive over-relaxation: in each half sweep only pixels of
/// one colour change, and they only read the other colour, so rows can be
/// updated in parallel from a snapshot.
fn relax(w: usize, h: usize, inside: &[bool], v: &mut [Px], sweeps: usize) {
    const OMEGA: f32 = 1.8;
    /// Shared write access for the red-black update (see below).
    #[derive(Clone, Copy)]
    struct Grid(*mut Px);
    unsafe impl Send for Grid {}
    unsafe impl Sync for Grid {}
    let rows: Vec<usize> = (0..h)
        .filter(|&y| inside[y * w..(y + 1) * w].iter().any(|&b| b))
        .collect();
    let grid = Grid(v.as_mut_ptr());
    for _ in 0..sweeps {
        for colour in 0..2 {
            rows.par_iter().for_each(|&y| {
                let g = grid;
                for x in ((y + colour) % 2..w).step_by(2) {
                    let i = y * w + x;
                    if !inside[i] {
                        continue;
                    }
                    // Reflecting at the array border.
                    let l = if x > 0 { i - 1 } else { i + 1 };
                    let r = if x + 1 < w { i + 1 } else { i - 1 };
                    let u = if y > 0 { i - w } else { i + w };
                    let d = if y + 1 < h { i + w } else { i - w };
                    // SAFETY: in this half sweep only pixels of `colour` are
                    // written, each by exactly one task (its row); the four
                    // neighbours have the other colour and are only read.
                    unsafe {
                        let p = g.0;
                        let (nl, nr, nu, nd) = (*p.add(l), *p.add(r), *p.add(u), *p.add(d));
                        let me = &mut *p.add(i);
                        for c in 0..4 {
                            let gs = 0.25 * (nl[c] + nr[c] + nu[c] + nd[c]);
                            me[c] += OMEGA * (gs - me[c]);
                        }
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disc in the middle of a w × h grid.
    fn disc(w: usize, h: usize, r: f32) -> Vec<bool> {
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        (0..w * h)
            .map(|i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) < r)
            .collect()
    }

    #[test]
    fn linear_boundary_gives_the_plane() {
        // A linear function is harmonic: the membrane reproduces it exactly.
        let (w, h) = (300, 200);
        let inside = disc(w, h, 80.0);
        let plane = |x: usize, y: usize| [x as f32 * 0.01, y as f32 * 0.02, 0.5, 1.0];
        let mut v: Vec<Px> = (0..w * h)
            .map(|i| {
                if inside[i] {
                    [9.0; 4]
                } else {
                    plane(i % w, i / w)
                }
            })
            .collect();
        let t = std::time::Instant::now();
        membrane(w, h, &inside, &mut v);
        let worst = (0..w * h)
            .filter(|&i| inside[i])
            .map(|i| {
                (0..4)
                    .map(|c| (v[i][c] - plane(i % w, i / w)[c]).abs())
                    .fold(0.0, f32::max)
            })
            .fold(0.0, f32::max);
        assert!(worst < 0.01, "max error {worst} ({:?})", t.elapsed());
    }

    #[test]
    fn untouched_outside_and_empty_inside() {
        let (w, h) = (20, 10);
        let mut v: Vec<Px> = (0..w * h).map(|i| [i as f32; 4]).collect();
        let before = v.clone();
        membrane(w, h, &vec![false; w * h], &mut v);
        assert_eq!(v, before);
        let inside = disc(w, h, 3.0);
        membrane(w, h, &inside, &mut v);
        for i in 0..w * h {
            if !inside[i] {
                assert_eq!(v[i], before[i]);
            }
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn membrane_speed() {
        for n in [500usize, 2000] {
            let inside = disc(n, n, n as f32 * 0.45);
            let plane = |x: usize, y: usize| [x as f32 * 0.001, y as f32 * 0.002, 0.5, 1.0];
            let mut v: Vec<Px> = (0..n * n)
                .map(|i| {
                    if inside[i] {
                        [0.0; 4]
                    } else {
                        plane(i % n, i / n)
                    }
                })
                .collect();
            let t = std::time::Instant::now();
            membrane(n, n, &inside, &mut v);
            let worst = (0..n * n)
                .filter(|&i| inside[i])
                .map(|i| {
                    (0..4)
                        .map(|c| (v[i][c] - plane(i % n, i / n)[c]).abs())
                        .fold(0.0, f32::max)
                })
                .fold(0.0, f32::max);
            eprintln!("{n}²: {:?}, max error {worst}", t.elapsed());
        }
    }
}
