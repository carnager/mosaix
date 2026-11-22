//! Puppet Warp: a triangle mesh over a layer's visible pixels, deformed
//! "as rigidly as possible" (Sorkine & Alexa 2007, uniform weights) when
//! pins move. Vertices sit on a grid and are numbered row by row, so the
//! Laplacian is a band matrix: its Cholesky factor is small and cheap,
//! computed once per set of pins; every drag only solves with it.

use glam::DVec2;

/// The mesh in document px.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub rest: Vec<DVec2>,
    pub tris: Vec<[u32; 3]>,
    /// Neighbours of each vertex (along triangle edges).
    pub nbrs: Vec<Vec<u32>>,
    pub spacing: f64,
}

impl Mesh {
    /// A grid with `spacing` px over the cells (cols × rows from `origin`)
    /// for which `occupied(col, row)` holds, each split into two triangles.
    pub fn from_cells(origin: DVec2, spacing: f64, cols: usize, rows: usize, occupied: impl Fn(usize, usize) -> bool) -> Self {
        let (vw, vh) = (cols + 1, rows + 1);
        let mut index = vec![u32::MAX; vw * vh];
        let mut rest = Vec::new();
        let mut tris = Vec::new();
        let cells: Vec<(usize, usize)> = (0..rows).flat_map(|r| (0..cols).map(move |c| (c, r))).filter(|&(c, r)| occupied(c, r)).collect();
        // Vertices row by row (keeps the band of the Laplacian narrow).
        let mut used = vec![false; vw * vh];
        for &(c, r) in &cells {
            for (dc, dr) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                used[(r + dr) * vw + c + dc] = true;
            }
        }
        for (k, u) in used.iter().enumerate() {
            if *u {
                index[k] = rest.len() as u32;
                rest.push(origin + DVec2::new((k % vw) as f64, (k / vw) as f64) * spacing);
            }
        }
        for &(c, r) in &cells {
            let v = |dc: usize, dr: usize| index[(r + dr) * vw + c + dc];
            tris.push([v(0, 0), v(1, 0), v(1, 1)]);
            tris.push([v(0, 0), v(1, 1), v(0, 1)]);
        }
        let mut nbrs = vec![Vec::new(); rest.len()];
        for t in &tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if !nbrs[a as usize].contains(&b) {
                    nbrs[a as usize].push(b);
                    nbrs[b as usize].push(a);
                }
            }
        }
        Self { rest, tris, nbrs, spacing }
    }

    /// The vertex nearest to `p` within `radius`.
    pub fn nearest(&self, p: DVec2, radius: f64) -> Option<usize> {
        self.rest
            .iter()
            .enumerate()
            .map(|(i, q)| (i, q.distance_squared(p)))
            .filter(|(_, d)| *d <= radius * radius)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }
}

/// A symmetric positive definite band matrix, factored (L Lᵀ).
struct Banded {
    n: usize,
    bw: usize,
    /// Row i holds L[i][i − bw ..= i].
    l: Vec<f64>,
}

impl Banded {
    fn at(&self, i: usize, j: usize) -> f64 {
        // j ≤ i, i − j ≤ bw.
        self.l[i * (self.bw + 1) + (self.bw + j - i)]
    }

    /// Factors `a` (given as a function of i, j with j ≤ i).
    fn factor(n: usize, bw: usize, a: impl Fn(usize, usize) -> f64) -> Option<Self> {
        let mut m = Self { n, bw, l: vec![0.0; n * (bw + 1)] };
        for i in 0..n {
            let j0 = i.saturating_sub(bw);
            for j in j0..=i {
                let mut s = a(i, j);
                let k0 = j0.max(j.saturating_sub(bw));
                for k in k0..j {
                    s -= m.at(i, k) * m.at(j, k);
                }
                let v = if i == j {
                    if s <= 0.0 {
                        return None;
                    }
                    s.sqrt()
                } else {
                    s / m.at(j, j)
                };
                m.l[i * (bw + 1) + (bw + j - i)] = v;
            }
        }
        Some(m)
    }

    fn solve(&self, b: &mut [f64]) {
        let (n, bw) = (self.n, self.bw);
        for i in 0..n {
            let lo = i.saturating_sub(bw);
            let s = b[i] - b[lo..i].iter().enumerate().map(|(o, v)| self.at(i, lo + o) * v).sum::<f64>();
            b[i] = s / self.at(i, i);
        }
        for i in (0..n).rev() {
            let hi = (i + bw + 1).min(n);
            let s = b[i] - b[i + 1..hi].iter().enumerate().map(|(o, v)| self.at(i + 1 + o, i) * v).sum::<f64>();
            b[i] = s / self.at(i, i);
        }
    }
}

/// Keeps vertices without a pin near their place (and makes a mesh part
/// without any pin stay where it is).
const ANCHOR: f64 = 1e-4;

/// The deformation for one set of pinned vertices.
pub struct Solver {
    pinned: Vec<bool>,
    /// Free vertex → row in the system, or usize::MAX.
    row: Vec<usize>,
    free: Vec<usize>,
    chol: Banded,
}

impl Solver {
    pub fn new(mesh: &Mesh, pins: &[usize]) -> Option<Self> {
        let n = mesh.rest.len();
        let mut pinned = vec![false; n];
        for &p in pins {
            pinned[p] = true;
        }
        let free: Vec<usize> = (0..n).filter(|&i| !pinned[i]).collect();
        let mut row = vec![usize::MAX; n];
        for (r, &i) in free.iter().enumerate() {
            row[i] = r;
        }
        let (row_r, pinned_r) = (&row, &pinned);
        let bw = free
            .iter()
            .flat_map(|&i| mesh.nbrs[i].iter().filter(|&&j| !pinned_r[j as usize]).map(move |&j| row_r[i].abs_diff(row_r[j as usize])))
            .max()
            .unwrap_or(0);
        // L_FF: degree on the diagonal (plus the anchor), −1 per edge.
        let chol = Banded::factor(free.len(), bw, |a, b| {
            let (i, j) = (free[a], free[b]);
            if a == b {
                mesh.nbrs[i].len() as f64 + ANCHOR
            } else if mesh.nbrs[i].contains(&(j as u32)) {
                -1.0
            } else {
                0.0
            }
        })?;
        Some(Self { pinned, row, free, chol })
    }

    /// Positions for pins at `targets` (vertex → place), starting from `cur`
    /// (e.g. the last result; `mesh.rest` the first time).
    pub fn solve(&self, mesh: &Mesh, targets: &[(usize, DVec2)], cur: &mut Vec<DVec2>, iterations: usize) {
        let n = mesh.rest.len();
        if cur.len() != n {
            *cur = mesh.rest.clone();
        }
        for &(i, q) in targets {
            cur[i] = q;
        }
        let mut rot = vec![(1.0f64, 0.0f64); n];
        for _ in 0..iterations {
            // Local step: the best rotation per vertex.
            for i in 0..n {
                let (mut c, mut s) = (0.0, 0.0);
                for &j in &mesh.nbrs[i] {
                    let e = mesh.rest[i] - mesh.rest[j as usize];
                    let f = cur[i] - cur[j as usize];
                    c += e.x * f.x + e.y * f.y;
                    s += e.x * f.y - e.y * f.x;
                }
                let a = s.atan2(c);
                rot[i] = (a.cos(), a.sin());
            }
            // Global step: L x = b for the free vertices, x and y apart.
            let mut bx = vec![0.0; self.free.len()];
            let mut by = vec![0.0; self.free.len()];
            for (r, &i) in self.free.iter().enumerate() {
                let mut b = mesh.rest[i] * ANCHOR;
                for &j in &mesh.nbrs[i] {
                    let j = j as usize;
                    let e = mesh.rest[i] - mesh.rest[j];
                    let (c, s) = ((rot[i].0 + rot[j].0) * 0.5, (rot[i].1 + rot[j].1) * 0.5);
                    b += DVec2::new(c * e.x - s * e.y, s * e.x + c * e.y);
                    if self.pinned[j] {
                        b += cur[j];
                    }
                }
                bx[r] = b.x;
                by[r] = b.y;
            }
            self.chol.solve(&mut bx);
            self.chol.solve(&mut by);
            for (r, &i) in self.free.iter().enumerate() {
                cur[i] = DVec2::new(bx[r], by[r]);
            }
        }
        debug_assert!(self.row.len() == n);
    }
}

/// Fills a displacement field (row-major, `fw` × `fh` nodes from `origin`,
/// `spacing` apart): for each node inside a deformed triangle, the offset
/// to where it comes from in the rest mesh; `away` elsewhere.
pub fn rasterize(mesh: &Mesh, deformed: &[DVec2], origin: DVec2, spacing: f64, fw: usize, fh: usize, away: f32) -> Vec<[f32; 2]> {
    let mut field = vec![[away; 2]; fw * fh];
    for t in &mesh.tris {
        let [a, b, c] = t.map(|i| deformed[i as usize]);
        let [ra, rb, rc] = t.map(|i| mesh.rest[i as usize]);
        let det = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if det.abs() < 1e-9 {
            continue;
        }
        let lo = (a.min(b).min(c) - origin) / spacing;
        let hi = (a.max(b).max(c) - origin) / spacing;
        let (x0, y0) = (lo.x.ceil().max(0.0) as usize, lo.y.ceil().max(0.0) as usize);
        let (x1, y1) = (hi.x.floor().min(fw as f64 - 1.0), hi.y.floor().min(fh as f64 - 1.0));
        if x1 < 0.0 || y1 < 0.0 {
            continue;
        }
        for y in y0..=y1 as usize {
            for x in x0..=x1 as usize {
                let p = origin + DVec2::new(x as f64, y as f64) * spacing;
                let w1 = ((p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y)) / det;
                let w2 = ((b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y)) / det;
                let w0 = 1.0 - w1 - w2;
                let eps = -1e-9;
                if w0 < eps || w1 < eps || w2 < eps {
                    continue;
                }
                let src = ra * w0 + rb * w1 + rc * w2;
                let d = src - p;
                field[y * fw + x] = [d.x as f32, d.y as f32];
            }
        }
    }
    field
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar() -> Mesh {
        // A 10 × 2 cell bar, 10 px cells, from (0, 0).
        Mesh::from_cells(DVec2::ZERO, 10.0, 10, 2, |_, _| true)
    }

    #[test]
    fn band_solver_matches_a_known_system() {
        // [4 1 0; 1 3 1; 0 1 2] x = [1 2 3].
        let a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let m = Banded::factor(3, 1, |i, j| a[i][j]).unwrap();
        let mut b = [1.0, 2.0, 3.0];
        m.solve(&mut b);
        for (i, row) in a.iter().enumerate() {
            let s: f64 = row.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((s - [1.0, 2.0, 3.0][i]).abs() < 1e-9);
        }
    }

    #[test]
    fn pins_in_place_change_nothing_and_a_rotation_stays_rigid() {
        let m = bar();
        assert_eq!(m.rest.len(), 33);
        let (left, right) = (m.nearest(DVec2::new(0.0, 10.0), 1.0).unwrap(), m.nearest(DVec2::new(100.0, 10.0), 1.0).unwrap());
        let s = Solver::new(&m, &[left, right]).unwrap();
        let mut cur = Vec::new();
        s.solve(&m, &[(left, m.rest[left]), (right, m.rest[right])], &mut cur, 10);
        for (a, b) in cur.iter().zip(&m.rest) {
            assert!(a.distance(*b) < 1e-3, "{a} vs {b}");
        }
        // Turn the bar by 90° about its left pin: it stays straight and as long.
        let q = DVec2::new(0.0, 10.0) + DVec2::new(0.0, 100.0);
        let mut cur = Vec::new();
        s.solve(&m, &[(left, m.rest[left]), (right, q)], &mut cur, 40);
        let mid = m.nearest(DVec2::new(50.0, 10.0), 1.0).unwrap();
        assert!((cur[mid] - DVec2::new(0.0, 60.0)).length() < 8.0, "middle follows: {}", cur[mid]);
        // Edge lengths are kept.
        for (i, ns) in m.nbrs.iter().enumerate() {
            for &j in ns {
                let (l0, l1) = (m.rest[i].distance(m.rest[j as usize]), cur[i].distance(cur[j as usize]));
                assert!((l1 / l0 - 1.0).abs() < 0.25, "edge {i}-{j}: {l0} → {l1}");
            }
        }
    }

    #[test]
    fn rasterized_field_points_back_to_the_rest_mesh() {
        let m = bar();
        let mut moved = m.rest.clone();
        for p in &mut moved {
            *p += DVec2::new(5.0, 3.0);
        }
        let f = rasterize(&m, &moved, DVec2::ZERO, 4.0, 40, 10, 1e7);
        // Node (12, 4) = (48, 16) lies in the moved bar: comes from 5, 3 back.
        let d = f[4 * 40 + 12];
        assert!((d[0] + 5.0).abs() < 1e-4 && (d[1] + 3.0).abs() < 1e-4, "{d:?}");
        assert_eq!(f[0], [1e7; 2], "outside");
    }
}
