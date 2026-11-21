//! Lens corrections embedded in DNG files (phones put them there instead of
//! relying on lens profiles): GainMap (lens shading, OpcodeList2, applied
//! to the sensor data), WarpRectilinear (distortion and colour fringes) and
//! FixVignetteRadial (OpcodeList3, applied to the demosaiced image).

/// One gain map: a grid of factors over the image for the pixels of one
/// CFA position (row/column pitch).
#[derive(Debug, Clone)]
pub struct GainMap {
    top: u32,
    left: u32,
    bottom: u32,
    right: u32,
    row_pitch: u32,
    col_pitch: u32,
    rows: usize,
    cols: usize,
    spacing: (f64, f64),
    origin: (f64, f64),
    gains: Vec<f32>,
}

impl GainMap {
    /// Gain at relative image position (v, h), bilinear between map points.
    fn at(&self, v: f64, h: f64) -> f32 {
        let fy = ((v - self.origin.0) / self.spacing.0.max(1e-9)).clamp(0.0, (self.rows - 1) as f64);
        let fx = ((h - self.origin.1) / self.spacing.1.max(1e-9)).clamp(0.0, (self.cols - 1) as f64);
        let (y0, x0) = (fy as usize, fx as usize);
        let (y1, x1) = ((y0 + 1).min(self.rows - 1), (x0 + 1).min(self.cols - 1));
        let (ay, ax) = ((fy - y0 as f64) as f32, (fx - x0 as f64) as f32);
        let g = |y: usize, x: usize| self.gains[y * self.cols + x];
        let top = g(y0, x0) + (g(y0, x1) - g(y0, x0)) * ax;
        let bot = g(y1, x0) + (g(y1, x1) - g(y1, x0)) * ax;
        top + (bot - top) * ay
    }

    /// Applies the map to a mosaic of `w × h` (stage-2 coordinates).
    /// Saturated values (≥ `clip`) stay as they are, so clipping is still seen.
    pub fn apply(&self, data: &mut [f32], w: usize, h: usize, clip: f32) {
        use rayon::prelude::*;
        let (b, r) = ((self.bottom as usize).min(h), (self.right as usize).min(w));
        data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y < self.top as usize || y >= b || !(y - self.top as usize).is_multiple_of(self.row_pitch.max(1) as usize) {
                return;
            }
            let v = (y as f64 + 0.5) / h as f64;
            let mut x = self.left as usize;
            while x < r {
                let p = &mut row[x];
                if *p < clip {
                    *p *= self.at(v, (x as f64 + 0.5) / w as f64);
                }
                x += self.col_pitch.max(1) as usize;
            }
        });
    }
}

/// Geometry and vignetting from OpcodeList3, in the coordinates of the
/// cropped image: `center` in pixels, `norm` = distance from the centre to
/// the farthest corner of the full (uncropped) image.
#[derive(Debug, Clone, PartialEq)]
pub struct Warp {
    /// Per colour plane (1 = all, 3 = R, G, B): kr0…kr3, kt0, kt1.
    pub planes: Vec<[f64; 6]>,
    pub vignette: Option<[f64; 5]>,
    pub center: (f64, f64),
    pub norm: f64,
}

impl Warp {
    /// Source position of the corrected position (x, y) for channel `ch`.
    pub fn source(&self, x: f32, y: f32, ch: usize) -> (f32, f32) {
        let Some(k) = self.planes.get(ch.min(self.planes.len().saturating_sub(1))) else {
            return (x, y);
        };
        let (dx, dy) = ((x as f64 - self.center.0) / self.norm, (y as f64 - self.center.1) / self.norm);
        let r2 = dx * dx + dy * dy;
        let f = k[0] + k[1] * r2 + k[2] * r2 * r2 + k[3] * r2 * r2 * r2;
        let tx = k[4] * 2.0 * dx * dy + k[5] * (r2 + 2.0 * dx * dx);
        let ty = k[5] * 2.0 * dx * dy + k[4] * (r2 + 2.0 * dy * dy);
        (
            (self.center.0 + self.norm * (f * dx + tx)) as f32,
            (self.center.1 + self.norm * (f * dy + ty)) as f32,
        )
    }

    /// Brightness factor at image position (x, y) (FixVignetteRadial).
    pub fn vignetting_gain(&self, x: f32, y: f32) -> f32 {
        let Some(k) = self.vignette else { return 1.0 };
        let (dx, dy) = ((x as f64 - self.center.0) / self.norm, (y as f64 - self.center.1) / self.norm);
        let r2 = dx * dx + dy * dy;
        let mut g = 1.0;
        let mut p = r2;
        for c in k {
            g += c * p;
            p *= r2;
        }
        g as f32
    }
}

/// kr0…kr3, kt0, kt1 per plane and the relative centre.
pub type WarpParams = (Vec<[f64; 6]>, (f64, f64));

/// What an opcode list holds (unknown opcodes are skipped).
#[derive(Debug, Default)]
pub struct Opcodes {
    pub gain_maps: Vec<GainMap>,
    /// Raw parameters (kr/kt per plane, centre) of WarpRectilinear.
    pub warp: Option<WarpParams>,
    pub vignette: Option<([f64; 5], (f64, f64))>,
}

/// Parses an opcode list (big-endian, DNG 1.3).
pub fn parse(b: &[u8], out: &mut Opcodes) {
    let u = |o: usize| b.get(o..o + 4).map(|s| u32::from_be_bytes(s.try_into().unwrap()));
    let Some(n) = u(0) else { return };
    let mut o = 4;
    for _ in 0..n.min(1000) {
        let (Some(id), Some(len)) = (u(o), u(o + 12)) else { return };
        let len = len as usize;
        let Some(p) = b.get(o + 16..o + 16 + len) else { return };
        let pu = |i: usize| p.get(i..i + 4).map(|s| u32::from_be_bytes(s.try_into().unwrap())).unwrap_or(0);
        let pf = |i: usize| p.get(i..i + 8).map(|s| f64::from_be_bytes(s.try_into().unwrap())).unwrap_or(0.0);
        match id {
            1 if len >= 4 => {
                let planes = pu(0).clamp(1, 4) as usize;
                let ks: Vec<[f64; 6]> = (0..planes).map(|pl| std::array::from_fn(|i| pf(4 + (pl * 6 + i) * 8))).collect();
                let c = 4 + planes * 48;
                out.warp = Some((ks, (pf(c), pf(c + 8))));
            }
            3 if len >= 56 => {
                out.vignette = Some((std::array::from_fn(|i| pf(i * 8)), (pf(40), pf(48))));
            }
            9 if len >= 76 => {
                let (rows, cols, planes) = (pu(32) as usize, pu(36) as usize, pu(72).max(1) as usize);
                if rows == 0 || cols == 0 || p.len() < 76 + rows * cols * planes * 4 {
                    o += 16 + len;
                    continue;
                }
                // Only the first plane (mosaic data has one).
                let gains = (0..rows * cols)
                    .map(|i| f32::from_be_bytes(p[76 + i * planes * 4..80 + i * planes * 4].try_into().unwrap()))
                    .collect();
                out.gain_maps.push(GainMap {
                    top: pu(0),
                    left: pu(4),
                    bottom: pu(8),
                    right: pu(12),
                    row_pitch: pu(24),
                    col_pitch: pu(28),
                    rows,
                    cols,
                    spacing: (pf(40), pf(48)),
                    origin: (pf(56), pf(64)),
                    gains,
                });
            }
            _ => {}
        }
        o += 16 + len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_identity_and_gain_map() {
        let w = Warp { planes: vec![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], vignette: Some([0.5, 0.0, 0.0, 0.0, 0.0]), center: (50.0, 40.0), norm: 64.0 };
        assert_eq!(w.source(10.0, 20.0, 2), (10.0, 20.0));
        assert!((w.vignetting_gain(50.0, 40.0) - 1.0).abs() < 1e-6);
        assert!(w.vignetting_gain(0.0, 0.0) > 1.2);
        // A 2×2 map: 1 at the top, 3 at the bottom, every pixel.
        let m = GainMap {
            top: 0, left: 0, bottom: 10, right: 10, row_pitch: 1, col_pitch: 1, rows: 2, cols: 2,
            spacing: (1.0, 1.0), origin: (0.0, 0.0), gains: vec![1.0, 1.0, 3.0, 3.0],
        };
        let mut d = vec![0.1f32; 100];
        m.apply(&mut d, 10, 10, 0.99);
        assert!((d[0] - 0.11).abs() < 1e-3 && (d[95] - 0.29).abs() < 1e-3, "{} {}", d[0], d[95]);
    }
}
