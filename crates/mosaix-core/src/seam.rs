//! Content-Aware Scale: changes an image's size by removing or doubling the
//! paths of pixels (seams) that show least, so subjects keep their shape.
//!
//! Seams are searched on a small proxy, which yields a map from output to
//! source positions; the full-resolution pixels are then sampled through it,
//! so large images stay fast and sharp.

use rayon::prelude::*;

/// Longest proxy side.
const PROXY: f32 = 800.0;
/// What crossing a fully protected pixel costs a seam.
const PROTECT_COST: f32 = 1.0e4;

#[derive(Clone, Copy)]
struct Cell {
    /// Perceptual premultiplied colour and alpha.
    c: [f32; 4],
    keep: f32,
    /// Where this cell comes from (proxy px).
    s: [f32; 2],
    /// Column at the start of an enlarging round.
    col: u32,
}

#[derive(Clone)]
struct Grid {
    w: usize,
    h: usize,
    cells: Vec<Cell>,
}

fn diff(a: &Cell, b: &Cell) -> f32 {
    (a.c[0] - b.c[0]).abs() + (a.c[1] - b.c[1]).abs() + (a.c[2] - b.c[2]).abs() + (a.c[3] - b.c[3]).abs()
}

impl Grid {
    fn transposed(&self) -> Self {
        let mut cells = Vec::with_capacity(self.cells.len());
        for x in 0..self.w {
            for y in 0..self.h {
                cells.push(self.cells[y * self.w + x]);
            }
        }
        Self { w: self.h, h: self.w, cells }
    }

    /// The cheapest top-to-bottom seam by forward energy (what removing it
    /// would newly put side by side): its column in each row.
    fn seam(&self, back: &mut Vec<i8>) -> Vec<u32> {
        let (w, h) = (self.w, self.h);
        back.clear();
        back.resize(w * h, 0);
        let mut prev: Vec<f32> = self.cells[..w].iter().map(|c| c.keep * PROTECT_COST).collect();
        let mut cur = vec![0.0f32; w];
        for y in 1..h {
            let row = &self.cells[y * w..(y + 1) * w];
            let up = &self.cells[(y - 1) * w..y * w];
            let dirs = &mut back[y * w..(y + 1) * w];
            for x in 0..w {
                let (l, r) = (x.saturating_sub(1), (x + 1).min(w - 1));
                let cu = diff(&row[r], &row[l]);
                let mut best = prev[x] + cu;
                let mut dir = 0i8;
                if x > 0 {
                    let v = prev[x - 1] + cu + diff(&up[x], &row[l]);
                    if v < best {
                        (best, dir) = (v, -1);
                    }
                }
                if x + 1 < w {
                    let v = prev[x + 1] + cu + diff(&up[x], &row[r]);
                    if v < best {
                        (best, dir) = (v, 1);
                    }
                }
                // Entering or leaving a shape costs too: otherwise a seam could
                // run straight through a flat-coloured subject for free.
                cur[x] = best + diff(&row[x], &up[x]) + row[x].keep * PROTECT_COST;
                dirs[x] = dir;
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        let mut x = prev.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        let mut path = vec![0u32; h];
        for y in (0..h).rev() {
            path[y] = x as u32;
            x = (x as i64 + back[y * w + x] as i64) as usize;
        }
        path
    }

    fn remove(&mut self, seam: &[u32]) {
        let w = self.w;
        let mut out = 0;
        for y in 0..self.h {
            for x in 0..w {
                if x as u32 != seam[y] {
                    self.cells[out] = self.cells[y * w + x];
                    out += 1;
                }
            }
        }
        self.cells.truncate(out);
        self.w = w - 1;
    }

    /// Doubles the `k` cheapest seams; each new cell lies between two old ones.
    fn grow(&mut self, k: usize, back: &mut Vec<i8>) {
        let (w, h) = (self.w, self.h);
        let mut marks = vec![false; w * h];
        if w == 1 {
            marks.fill(true);
        } else {
            let mut t = self.clone();
            for (i, c) in t.cells.iter_mut().enumerate() {
                c.col = (i % w) as u32;
            }
            for _ in 0..k {
                let s = t.seam(back);
                for y in 0..h {
                    marks[y * w + t.cells[y * t.w + s[y] as usize].col as usize] = true;
                }
                t.remove(&s);
            }
        }
        let mut cells = Vec::with_capacity((w + k) * h);
        for y in 0..h {
            for x in 0..w {
                let a = self.cells[y * w + x];
                cells.push(a);
                if marks[y * w + x] {
                    let b = self.cells[y * w + (x + 1).min(w - 1)];
                    cells.push(Cell {
                        c: [0, 1, 2, 3].map(|i| (a.c[i] + b.c[i]) * 0.5),
                        keep: a.keep.max(b.keep),
                        s: [(a.s[0] + b.s[0]) * 0.5, (a.s[1] + b.s[1]) * 0.5],
                        col: 0,
                    });
                }
            }
        }
        self.w = w + k;
        self.cells = cells;
    }

    fn fit_width(&mut self, target: usize) {
        let mut back = Vec::new();
        while self.w > target {
            let s = self.seam(&mut back);
            self.remove(&s);
        }
        while self.w < target {
            // At most half the width per round, so seams stay distinct.
            let k = (target - self.w).min((self.w / 2).max(1));
            self.grow(k, &mut back);
        }
    }
}

/// The proxy of the image: box-filtered colour, strongest protection.
fn proxy(w: usize, h: usize, src: &[[f32; 4]], protect: Option<&[u8]>, pw: usize, ph: usize) -> Grid {
    let mut cells = vec![Cell { c: [0.0; 4], keep: 0.0, s: [0.0; 2], col: 0 }; pw * ph];
    cells.par_chunks_mut(pw).enumerate().for_each(|(py, row)| {
        let (y0, y1) = (py * h / ph, ((py + 1) * h / ph).max(py * h / ph + 1));
        for (px, cell) in row.iter_mut().enumerate() {
            let (x0, x1) = (px * w / pw, ((px + 1) * w / pw).max(px * w / pw + 1));
            let (mut sum, mut keep) = ([0.0f32; 4], 0u8);
            for y in y0..y1.min(h) {
                for x in x0..x1.min(w) {
                    let p = src[y * w + x];
                    for k in 0..4 {
                        sum[k] += p[k];
                    }
                    if let Some(m) = protect {
                        keep = keep.max(m[y * w + x]);
                    }
                }
            }
            let n = ((y1.min(h) - y0) * (x1.min(w) - x0)).max(1) as f32;
            let a = sum.map(|v| v / n);
            cell.c = [a[0].max(0.0).sqrt(), a[1].max(0.0).sqrt(), a[2].max(0.0).sqrt(), a[3]];
            cell.keep = keep as f32 / 255.0;
            cell.s = [px as f32, py as f32];
        }
    });
    Grid { w: pw, h: ph, cells }
}

fn bilinear<const N: usize>(d: &[[f32; N]], w: usize, h: usize, x: f32, y: f32) -> [f32; N] {
    let (x, y) = (x.clamp(0.0, (w - 1) as f32), y.clamp(0.0, (h - 1) as f32));
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (ax, ay) = (x - x0 as f32, y - y0 as f32);
    let (a, b, c, e) = (d[y0 * w + x0], d[y0 * w + x1], d[y1 * w + x0], d[y1 * w + x1]);
    std::array::from_fn(|k| {
        let top = a[k] + (b[k] - a[k]) * ax;
        let bot = c[k] + (e[k] - c[k]) * ax;
        top + (bot - top) * ay
    })
}

/// `src` (`w` × `h`, premultiplied) at `nw` × `nh`. `protect` (0–255 per
/// pixel) marks what seams avoid; `amount` blends from a plain resize (0) to
/// fully content-aware (1).
pub fn scale(w: usize, h: usize, src: &[[f32; 4]], protect: Option<&[u8]>, nw: usize, nh: usize, amount: f32) -> Vec<[f32; 4]> {
    assert_eq!(src.len(), w * h);
    let amount = amount.clamp(0.0, 1.0);
    let s = (w.max(h).max(nw).max(nh) as f32 / PROXY).max(1.0);
    let side = |v: usize| ((v as f32 / s).round() as usize).max(1);
    let (pw, ph, tw, th) = (side(w), side(h), side(nw), side(nh));
    let (rx, ry) = (w as f32 / pw as f32, h as f32 / ph as f32);

    let map: Vec<[f32; 2]> = if amount > 0.0 {
        let mut g = proxy(w, h, src, protect, pw, ph);
        g.fit_width(tw);
        let mut g = g.transposed();
        g.fit_width(th);
        let g = g.transposed();
        g.cells.iter().map(|c| [(c.s[0] + 0.5) * rx - 0.5, (c.s[1] + 0.5) * ry - 0.5]).collect()
    } else {
        Vec::new()
    };

    let (fx, fy) = (w as f32 / nw as f32, h as f32 / nh as f32);
    let (mx, my) = (tw as f32 / nw as f32, th as f32 / nh as f32);
    let mut out = vec![[0.0f32; 4]; nw * nh];
    out.par_chunks_mut(nw).enumerate().for_each(|(y, row)| {
        for (x, p) in row.iter_mut().enumerate() {
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            let plain = [cx * fx - 0.5, cy * fy - 0.5];
            let at = if map.is_empty() {
                plain
            } else {
                let q = bilinear(&map, tw, th, cx * mx - 0.5, cy * my - 0.5);
                [plain[0] + (q[0] - plain[0]) * amount, plain[1] + (q[1] - plain[1]) * amount]
            };
            *p = bilinear(src, w, h, at[0], at[1]);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: [f32; 4] = [1.0; 4];
    const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    /// White with a black square `[x0, x0 + side)` × `[y0, y0 + side)`.
    fn square(w: usize, h: usize, x0: usize, y0: usize, side: usize) -> Vec<[f32; 4]> {
        let mut v = vec![WHITE; w * h];
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                v[y * w + x] = BLACK;
            }
        }
        v
    }

    /// Dark pixels in the row and column through the image's dark centre.
    fn extent(p: &[[f32; 4]], w: usize, h: usize) -> (usize, usize) {
        let dark: Vec<(usize, usize)> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).filter(|&(x, y)| p[y * w + x][0] < 0.5).collect();
        let cx = dark.iter().map(|d| d.0).sum::<usize>() / dark.len();
        let cy = dark.iter().map(|d| d.1).sum::<usize>() / dark.len();
        (dark.iter().filter(|d| d.1 == cy).count(), dark.iter().filter(|d| d.0 == cx).count())
    }

    #[test]
    fn same_size_changes_nothing() {
        let src = square(60, 40, 20, 10, 12);
        assert_eq!(scale(60, 40, &src, None, 60, 40, 1.0), src);
    }

    #[test]
    fn narrowing_keeps_the_subject() {
        let src = square(200, 100, 80, 30, 40);
        let out = scale(200, 100, &src, None, 120, 100, 1.0);
        let (w, h) = extent(&out, 120, 100);
        assert!(w.abs_diff(40) <= 2 && h.abs_diff(40) <= 2, "{w} × {h}");
    }

    #[test]
    fn both_axes_and_enlarging_keep_the_subject() {
        let src = square(120, 90, 45, 30, 30);
        let out = scale(120, 90, &src, None, 200, 60, 1.0);
        let (w, h) = extent(&out, 200, 60);
        assert!(w.abs_diff(30) <= 3 && h.abs_diff(30) <= 3, "{w} × {h}");
    }

    #[test]
    fn enlarges_past_double() {
        let src = square(40, 30, 15, 10, 10);
        let out = scale(40, 30, &src, None, 130, 30, 1.0);
        let (w, _) = extent(&out, 130, 30);
        assert!(w.abs_diff(10) <= 3, "{w}");
    }

    #[test]
    fn amount_zero_is_a_plain_resize() {
        let src = square(200, 100, 80, 30, 40);
        let out = scale(200, 100, &src, None, 100, 100, 0.0);
        let (w, h) = extent(&out, 100, 100);
        assert!(w.abs_diff(20) <= 1 && h.abs_diff(40) <= 1, "{w} × {h}");
    }

    #[test]
    fn protection_steers_seams_off_flat_areas() {
        // A faint square on white would be carved through; protected it stays.
        let (w, h) = (120, 60);
        let mut src = vec![WHITE; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = if (x * 7 + y * 13) % 5 == 0 { 0.8 } else { 1.0 };
                src[y * w + x] = [v, v, v, 1.0];
            }
        }
        let mut keep = vec![0u8; w * h];
        for y in 0..h {
            for x in 40..70 {
                src[y * w + x] = [0.0, 0.0, 0.0, 1.0];
                keep[y * w + x] = 255;
            }
        }
        let out = scale(w, h, &src, Some(&keep), 70, h, 1.0);
        let dark = (0..70).filter(|&x| out[30 * 70 + x][0] < 0.5).count();
        assert!(dark.abs_diff(30) <= 1, "{dark}");
    }

    #[test]
    fn large_images_use_the_proxy() {
        let src = square(1600, 400, 700, 100, 200);
        let out = scale(1600, 400, &src, None, 1000, 400, 1.0);
        let (w, h) = extent(&out, 1000, 400);
        assert!(w.abs_diff(200) <= 8 && h.abs_diff(200) <= 8, "{w} × {h}");
    }
}
