//! Edit › Puppet Warp: pins on a mesh over the active layer's pixels; moving
//! them bends the layer as rigidly as possible. Runs as a liquify session
//! (the deformed mesh becomes its displacement field).

use glam::DVec2;
use mosaix_core::puppet::{Mesh, Solver, rasterize};
use mosaix_gpu::Engine;

/// Field value for places the mesh does not cover: reads from far away (transparent).
const AWAY: f32 = 1.0e7;
/// Mesh vertices aimed for (more: finer, slower to set up).
const TARGET_VERTICES: f64 = 2500.0;

pub struct Puppet {
    pub mesh: Mesh,
    /// Pinned vertex and where it is now.
    pub pins: Vec<(usize, DVec2)>,
    solver: Option<Solver>,
    cur: Vec<DVec2>,
    pub show_mesh: bool,
}

impl Puppet {
    /// Starts on the active pixel layer.
    pub fn begin(e: &mut Engine) -> Result<Self, String> {
        let id = e.doc().active;
        let (x0, y0, x1, y1) = e.content_bounds(id).ok_or_else(|| mosaix_core::tr("Nothing to warp: the layer is empty.").to_string())?;
        e.liquify_begin().map_err(|err| err.to_string())?;
        let (_, fs, _, _) = e.liquify_field_grid().unwrap();
        let area = ((x1 - x0) as f64 * (y1 - y0) as f64).max(1.0);
        let cell = (area / TARGET_VERTICES).sqrt().max(fs * 2.0).max(6.0);
        let Some((origin, cols, rows, occ)) = e.layer_cells(id, cell) else {
            e.liquify_cancel();
            return Err(mosaix_core::tr("Nothing to warp: the layer is empty.").to_string());
        };
        let mesh = Mesh::from_cells(origin, cell, cols, rows, |c, r| occ[r * cols + c]);
        let cur = mesh.rest.clone();
        Ok(Self { mesh, pins: Vec::new(), solver: None, cur, show_mesh: true })
    }

    /// The pin within `radius` of `p`.
    pub fn pin_at(&self, p: DVec2, radius: f64) -> Option<usize> {
        self.pins
            .iter()
            .enumerate()
            .map(|(k, (_, q))| (k, q.distance(p)))
            .filter(|(_, d)| *d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(k, _)| k)
    }

    /// Adds a pin on the mesh near `p` (where it currently is); its index.
    pub fn add_pin(&mut self, p: DVec2) -> Option<usize> {
        // Nearest vertex in the deformed mesh.
        let r = self.mesh.spacing * 1.5;
        let v = self
            .cur
            .iter()
            .enumerate()
            .filter(|(i, q)| q.distance(p) <= r && !self.pins.iter().any(|(j, _)| j == i))
            .min_by(|a, b| a.1.distance(p).total_cmp(&b.1.distance(p)))
            .map(|(i, _)| i)?;
        self.pins.push((v, self.cur[v]));
        self.solver = None;
        Some(self.pins.len() - 1)
    }

    pub fn remove_pin(&mut self, k: usize) {
        if k < self.pins.len() {
            self.pins.remove(k);
            self.solver = None;
        }
    }

    pub fn move_pin(&mut self, k: usize, to: DVec2) {
        if let Some(p) = self.pins.get_mut(k) {
            p.1 = to;
        }
    }

    /// The mesh as it is now.
    pub fn deformed(&self) -> &[DVec2] {
        &self.cur
    }

    /// Solves for the pins and hands the result to the engine's preview.
    pub fn update(&mut self, e: &mut Engine) {
        if self.pins.is_empty() {
            self.cur = self.mesh.rest.clone();
        } else {
            if self.solver.is_none() {
                let ids: Vec<usize> = self.pins.iter().map(|p| p.0).collect();
                self.solver = Solver::new(&self.mesh, &ids);
            }
            if let Some(s) = &self.solver {
                s.solve(&self.mesh, &self.pins, &mut self.cur, 12);
            }
        }
        if let Some((origin, spacing, fw, fh)) = e.liquify_field_grid() {
            e.liquify_set_field(rasterize(&self.mesh, &self.cur, origin, spacing, fw, fh, AWAY));
        }
    }
}
