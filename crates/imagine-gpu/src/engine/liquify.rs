//! Liquify: a displacement field on a coarse grid over the active pixel
//! layer, edited by brush dabs on the CPU (cheap: only the cells under the
//! brush). The GPU renders `out(p) = src(p + D(p))` from the untouched source
//! tiles, only for tiles the field changed in; the preview recomputes the
//! visible stale tiles each frame. Committing renders the rest and records
//! one undo step.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::DVec2;
use imagine_core::doc::Content;
use imagine_core::{LayerId, TILE_SIZE, TileCoord};
use serde::{Deserialize, Serialize};

use super::transform::{GridRef, Grids, Preview, PreviewKind};
use super::{Engine, StrokeTarget};
use crate::pool::Tile;

type TileSet = HashMap<TileCoord, Tile>;

/// Undo steps kept inside a liquify session.
const UNDO_STEPS: usize = 16;
/// Target number of field cells (sets the grid spacing).
const MAX_CELLS: f64 = 4.0e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiquifyTool {
    /// Pushes pixels along the stroke.
    #[default]
    Push,
    /// Gradually restores the original.
    Reconstruct,
    /// Rotates clockwise around the brush centre (while held).
    TwirlCw,
    TwirlCcw,
    /// Pulls pixels towards the centre (while held).
    Pucker,
    /// Pushes pixels away from the centre (while held).
    Bloat,
}

impl LiquifyTool {
    pub const ALL: [Self; 6] = [Self::Push, Self::Reconstruct, Self::TwirlCw, Self::TwirlCcw, Self::Pucker, Self::Bloat];

    pub fn name(self) -> &'static str {
        match self {
            Self::Push => "Schieben",
            Self::Reconstruct => "Rekonstruieren",
            Self::TwirlCw => "Strudel rechts",
            Self::TwirlCcw => "Strudel links",
            Self::Pucker => "Zusammenziehen",
            Self::Bloat => "Aufblähen",
        }
    }

    /// Whether the tool keeps working while the pointer is held still.
    pub fn continuous(self) -> bool {
        !matches!(self, Self::Push)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct LiquifyJob {
    origin: [i32; 2],
    field_origin: [f32; 2],
    src: GridRef,
    spacing: f32,
    fw: u32,
    fh: u32,
    _pad: u32,
}

pub(super) struct Liquify {
    layer: LayerId,
    src: Arc<TileSet>,
    /// Displacement per cell (document px), row-major.
    field: Vec<[f32; 2]>,
    fw: usize,
    fh: usize,
    origin: DVec2,
    spacing: f64,
    pub(super) buf: wgpu::Buffer,
    /// Cell rectangle (x0, y0, x1, y1 exclusive) not yet uploaded.
    upload: Option<(usize, usize, usize, usize)>,
    /// Tiles the field ever changed in, and the rendered ones among them.
    touched: HashSet<TileCoord>,
    out: TileSet,
    stale: HashSet<TileCoord>,
    undo: Vec<Vec<[f32; 2]>>,
    /// The preview document needs rebuilding.
    changed: bool,
}

impl Liquify {
    fn at(&self, x: i64, y: i64) -> [f32; 2] {
        if x < 0 || y < 0 || x >= self.fw as i64 || y >= self.fh as i64 {
            return [0.0; 2];
        }
        self.field[y as usize * self.fw + x as usize]
    }

    /// Bilinear field value at a document position.
    fn sample(&self, p: DVec2) -> DVec2 {
        let g = (p - self.origin) / self.spacing;
        let (fx, fy) = (g.x.floor(), g.y.floor());
        let (tx, ty) = ((g.x - fx) as f32, (g.y - fy) as f32);
        let (x, y) = (fx as i64, fy as i64);
        let lerp = |a: [f32; 2], b: [f32; 2], t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let a = lerp(self.at(x, y), self.at(x + 1, y), tx);
        let b = lerp(self.at(x, y + 1), self.at(x + 1, y + 1), tx);
        let v = lerp(a, b, ty);
        DVec2::new(v[0] as f64, v[1] as f64)
    }

    /// Marks the cells (x0..x1, y0..y1) changed: upload and affected tiles.
    fn mark(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        self.upload = Some(match self.upload {
            None => (x0, y0, x1, y1),
            Some((a, b, c, d)) => (a.min(x0), b.min(y0), c.max(x1), d.max(y1)),
        });
        // Bilinear interpolation spreads a cell's influence one spacing.
        let ts = TILE_SIZE as f64;
        let p0 = self.origin + DVec2::new(x0 as f64 - 1.0, y0 as f64 - 1.0) * self.spacing;
        let p1 = self.origin + DVec2::new(x1 as f64, y1 as f64) * self.spacing;
        for ty in (p0.y / ts).floor() as i32..=(p1.y / ts).floor() as i32 {
            for tx in (p0.x / ts).floor() as i32..=(p1.x / ts).floor() as i32 {
                let c = TileCoord::new(tx, ty);
                self.touched.insert(c);
                self.stale.insert(c);
            }
        }
        self.changed = true;
    }
}

impl Engine {
    /// Starts liquifying the active pixel layer.
    pub fn liquify_begin(&mut self) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err("gerade wird gemalt");
        }
        let layer = self.check_target(StrokeTarget::Pixels)?;
        let src: Arc<TileSet> = self.doc.find(layer).and_then(|l| l.tiles()).cloned().unwrap_or_default();
        // Field domain: the canvas and the layer's content, plus a margin.
        let ts = TILE_SIZE as f64;
        let (mut a, mut b) = (DVec2::ZERO, DVec2::new(self.doc.width as f64, self.doc.height as f64));
        for c in src.keys() {
            let o = DVec2::new(c.x as f64 * ts, c.y as f64 * ts);
            a = a.min(o);
            b = b.max(o + ts);
        }
        a -= ts;
        b += ts;
        let area = (b.x - a.x) * (b.y - a.y);
        let spacing = (area / MAX_CELLS).sqrt().max(8.0).log2().ceil().exp2();
        let origin = (a / spacing).floor() * spacing;
        let fw = ((b.x - origin.x) / spacing).ceil() as usize + 2;
        let fh = ((b.y - origin.y) / spacing).ceil() as usize + 2;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquify field"),
            size: (fw * fh * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        self.liquify = Some(Liquify {
            layer,
            src,
            field: vec![[0.0; 2]; fw * fh],
            fw,
            fh,
            origin,
            spacing,
            buf,
            upload: None,
            touched: HashSet::new(),
            out: TileSet::new(),
            stale: HashSet::new(),
            undo: Vec::new(),
            changed: true,
        });
        Ok(())
    }

    pub fn liquify_active(&self) -> bool {
        self.liquify.is_some()
    }

    /// Remembers the field before a stroke (for [`Engine::liquify_undo`]).
    pub fn liquify_stroke_start(&mut self) {
        if let Some(l) = &mut self.liquify {
            if l.undo.len() >= UNDO_STEPS {
                l.undo.remove(0);
            }
            l.undo.push(l.field.clone());
        }
    }

    /// Undoes the last stroke inside the session. False if there is none.
    pub fn liquify_undo(&mut self) -> bool {
        let Some(l) = &mut self.liquify else { return false };
        let Some(f) = l.undo.pop() else { return false };
        l.field = f;
        let (fw, fh) = (l.fw, l.fh);
        let touched: Vec<TileCoord> = l.touched.iter().copied().collect();
        l.mark(0, 0, fw, fh);
        // Only the tiles that were ever touched can differ.
        l.stale = touched.iter().copied().collect();
        l.touched = touched.into_iter().collect();
        true
    }

    /// One brush dab at `center` (document px). `delta`: movement since the
    /// last dab (push tool); `strength` 0…1 (pressure × setting); `rate`
    /// scales continuous tools by elapsed time (1 = one frame at 60 Hz).
    pub fn liquify_dab(&mut self, tool: LiquifyTool, center: DVec2, delta: DVec2, radius: f64, strength: f64, rate: f64) {
        let Some(l) = &mut self.liquify else { return };
        let r = radius.max(1.0);
        let s = l.spacing;
        let lo = ((center - r - l.origin) / s).floor();
        let hi = ((center + r - l.origin) / s).ceil();
        let x0 = (lo.x.max(0.0) as usize).min(l.fw);
        let y0 = (lo.y.max(0.0) as usize).min(l.fh);
        let x1 = ((hi.x + 1.0).max(0.0) as usize).min(l.fw);
        let y1 = ((hi.y + 1.0).max(0.0) as usize).min(l.fh);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        // New values are computed from the old field (copy of the region).
        let mut new = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            for x in x0..x1 {
                let p = l.origin + DVec2::new(x as f64, y as f64) * s;
                let d = (p - center).length() / r;
                let old = l.field[y * l.fw + x];
                if d >= 1.0 {
                    new.push(old);
                    continue;
                }
                let falloff = (1.0 - d * d).powi(2);
                let w = (strength * falloff).clamp(0.0, 1.0);
                let v = match tool {
                    LiquifyTool::Reconstruct => {
                        let k = (1.0 - w * 0.12 * rate).max(0.0) as f32;
                        [old[0] * k, old[1] * k]
                    }
                    _ => {
                        // Sample what is shown at q: D'(p) = (q − p) + D(q).
                        let q = match tool {
                            LiquifyTool::Push => p - delta * w,
                            LiquifyTool::TwirlCw | LiquifyTool::TwirlCcw => {
                                let sign = if tool == LiquifyTool::TwirlCw { -1.0 } else { 1.0 };
                                let a = sign * w * 0.06 * rate;
                                center + DVec2::from_angle(a).rotate(p - center)
                            }
                            LiquifyTool::Pucker => center + (p - center) * (1.0 + w * 0.03 * rate),
                            LiquifyTool::Bloat => center + (p - center) * (1.0 - w * 0.03 * rate).max(0.05),
                            LiquifyTool::Reconstruct => unreachable!(),
                        };
                        let v = (q - p) + l.sample(q);
                        [v.x as f32, v.y as f32]
                    }
                };
                new.push(v);
            }
        }
        let w = x1 - x0;
        for (j, y) in (y0..y1).enumerate() {
            l.field[y * l.fw + x0..y * l.fw + x1].copy_from_slice(&new[j * w..(j + 1) * w]);
        }
        l.mark(x0, y0, x1, y1);
    }

    /// Renders the given tiles with the current field.
    fn liquify_render(&mut self, coords: &[TileCoord]) -> Result<Vec<Tile>, &'static str> {
        let l = self.liquify.as_mut().ok_or("Verflüssigen ist nicht aktiv")?;
        if let Some((x0, y0, x1, y1)) = l.upload.take() {
            for y in y0..y1 {
                let row = &l.field[y * l.fw + x0..y * l.fw + x1];
                self.queue.write_buffer(&l.buf, ((y * l.fw + x0) * 8) as u64, bytemuck::cast_slice(row));
            }
        }
        let mut grids = Grids::default();
        let g = grids.layer(&l.src);
        let jobs: Vec<LiquifyJob> = coords
            .iter()
            .map(|c| {
                let (x, y) = c.origin();
                LiquifyJob {
                    origin: [x as i32, y as i32],
                    field_origin: [l.origin.x as f32, l.origin.y as f32],
                    src: g,
                    spacing: l.spacing as f32,
                    fw: l.fw as u32,
                    fh: l.fh as u32,
                    _pad: 0,
                }
            })
            .collect();
        self.grid_buf.write(&self.device, &self.queue, bytemuck::cast_slice(&grids.data));
        self.run_liquify(&jobs).ok_or("GPU-Speicher voll")
    }

    /// Per frame: renders the stale tiles in the visible area and updates the preview.
    pub fn liquify_tick(&mut self) -> Result<(), &'static str> {
        let Some(l) = &self.liquify else { return Ok(()) };
        let (min, max) = self.visible;
        let pad = (max - min) * 0.1;
        let (a, b) = (min - pad, max + pad);
        let ts = TILE_SIZE as f64;
        let mut todo: Vec<TileCoord> = l
            .stale
            .iter()
            .copied()
            .filter(|c| {
                let (x, y) = (c.x as f64 * ts, c.y as f64 * ts);
                x < b.x && x + ts > a.x && y < b.y && y + ts > a.y
            })
            .collect();
        let rebuild = l.changed || self.preview.is_none();
        if todo.is_empty() && !rebuild {
            return Ok(());
        }
        todo.sort();
        let tiles = self.liquify_render(&todo)?;
        let l = self.liquify.as_mut().unwrap();
        for (c, t) in todo.iter().zip(tiles) {
            l.out.insert(*c, t);
            l.stale.remove(c);
        }
        l.changed = false;
        let mut map: TileSet = (*l.src).clone();
        map.extend(l.out.iter().map(|(c, t)| (*c, t.clone())));
        let mut doc = self.doc.clone();
        if let Some(layer) = doc.find_mut(l.layer) {
            layer.content = Content::Pixels(Arc::new(map));
        }
        let first = self.preview.is_none();
        let inf = DVec2::splat(f64::INFINITY);
        self.preview = Some(Preview { kind: PreviewKind::Liquify, rect: (-inf, inf), doc });
        if first {
            self.recheck_all = true;
        } else {
            self.recheck.extend(todo);
        }
        Ok(())
    }

    /// Applies the liquify result to the layer as one undo step and ends the session.
    pub fn liquify_commit(&mut self, label: &str) -> Result<(), &'static str> {
        let Some(l) = &self.liquify else { return Err("Verflüssigen ist nicht aktiv") };
        let todo: Vec<TileCoord> = l.stale.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
        let tiles = self.liquify_render(&todo)?;
        let l = self.liquify.take().unwrap();
        let mut out = l.out;
        out.extend(todo.into_iter().zip(tiles));
        let mut map: TileSet = (*l.src).clone();
        map.extend(out);
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        if l.touched.is_empty() {
            return Ok(());
        }
        let layer = l.layer;
        self.edit(label, |d| match d.find_mut(layer) {
            Some(x) => {
                x.content = Content::Pixels(Arc::new(map));
                true
            }
            None => false,
        });
        Ok(())
    }

    /// Ends the session without changes.
    pub fn liquify_cancel(&mut self) {
        if self.liquify.take().is_some() && self.preview.take().is_some() {
            self.recheck_all = true;
        }
    }
}
