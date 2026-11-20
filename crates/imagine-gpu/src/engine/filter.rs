//! Pixel filters on the active layer: Gaussian blur (separable; large radii
//! run on a downscaled copy), unsharp mask, high pass and noise. The result
//! is mixed in through the selection and recorded as one undo step; a live
//! preview computes only the visible area.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{DAffine2, DVec2};
use imagine_core::doc::Content;
use imagine_core::{Document, Filter, MaskRef, TILE_SIZE, TileCoord};

use super::transform::{GridRef, Grids, PreviewKind, footprint};
use super::{Engine, Interp, NONE, StrokeTarget};
use crate::pool::Tile;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct FilterJob {
    origin: [i32; 2],
    dir: [i32; 2],
    src: GridRef,
    blur_slot: u32,
    kind: u32,
    radius: u32,
    _pad: u32,
    p: [f32; 4],
}

const BLUR: u32 = 0;
const UNSHARP: u32 = 1;
const HIGH_PASS: u32 = 2;
const NOISE: u32 = 3;
/// Above this sigma the blur runs on a downscaled copy.
const DIRECT_SIGMA: f32 = 10.0;

type TileSet = HashMap<TileCoord, Tile>;

/// Tiles within `r` tiles of any tile in `src`.
fn dilate(src: impl IntoIterator<Item = TileCoord>, r: i32) -> BTreeSet<TileCoord> {
    let mut out = BTreeSet::new();
    for c in src {
        for dy in -r..=r {
            for dx in -r..=r {
                out.insert(TileCoord::new(c.x + dx, c.y + dy));
            }
        }
    }
    out
}

fn in_rect(c: &TileCoord, rect: Option<(DVec2, DVec2)>) -> bool {
    rect.is_none_or(|(a, b)| {
        let (x, y) = c.origin();
        let ts = TILE_SIZE as f64;
        (x as f64) < b.x && (x as f64 + ts) > a.x && (y as f64) < b.y && (y as f64 + ts) > a.y
    })
}

impl Engine {
    fn filter_jobs(&mut self, jobs: &[FilterJob], grid: &[u32]) -> Option<Vec<Tile>> {
        self.grid_buf.write(&self.device, &self.queue, bytemuck::cast_slice(grid));
        self.run_filter(jobs)
    }

    fn fjob(c: TileCoord, src: GridRef, kind: u32) -> FilterJob {
        let (x, y) = c.origin();
        FilterJob {
            origin: [x as i32, y as i32],
            dir: [0, 0],
            src,
            blur_slot: NONE,
            kind,
            radius: 0,
            _pad: 0,
            p: [0.0; 4],
        }
    }

    /// One separable Gaussian at full resolution into the tiles `dest`.
    fn blur_direct(&mut self, src: &TileSet, sigma: f32, dest: &[TileCoord]) -> Option<TileSet> {
        let r = (3.0 * sigma).ceil().max(1.0) as u32;
        let reach = (r as i32 + TILE_SIZE as i32 - 1) / TILE_SIZE as i32;
        // Horizontal pass also for the rows the vertical pass reads.
        let mid: Vec<TileCoord> = dest
            .iter()
            .flat_map(|c| (-reach..=reach).map(move |dy| TileCoord::new(c.x, c.y + dy)))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let pass = |eng: &mut Engine, src: &TileSet, coords: &[TileCoord], dir: [i32; 2]| -> Option<TileSet> {
            let mut grids = Grids::default();
            let g = grids.layer(src);
            let jobs: Vec<FilterJob> = coords
                .iter()
                .map(|&c| FilterJob { dir, radius: r, p: [sigma, 0.0, 0.0, 0.0], ..Self::fjob(c, g, BLUR) })
                .collect();
            let tiles = eng.filter_jobs(&jobs, &grids.data)?;
            Some(coords.iter().copied().zip(tiles).collect())
        };
        let h = pass(self, src, &mid, [1, 0])?;
        pass(self, &h, dest, [0, 1])
    }

    /// Gaussian blur of `src` into `dest`; large radii are computed at a
    /// lower resolution and scaled back up.
    fn blur(&mut self, src: &TileSet, sigma: f32, dest: &[TileCoord]) -> Option<TileSet> {
        if sigma <= DIRECT_SIGMA {
            return self.blur_direct(src, sigma, dest);
        }
        let k = 2f64.powi((sigma / (DIRECT_SIGMA / 2.0)).log2().floor() as i32).max(2.0);
        let down = DAffine2::from_scale(DVec2::splat(1.0 / k));
        let small_dest: Vec<TileCoord> = dilate(footprint(src.keys().copied(), down, 1.0), 1).into_iter().collect();
        let small = self.warp_map(src, down, Interp::Bilinear, (k as u32).min(8), &small_dest)?;
        let small_out: Vec<TileCoord> = dilate(footprint(dest.iter().copied(), down, 2.0), 1).into_iter().collect();
        let blurred = self.blur_direct(&small, sigma / k as f32, &small_out)?;
        self.warp_map(&blurred, DAffine2::from_scale(DVec2::splat(k)), Interp::Bilinear, 1, dest)
    }

    /// Filters the active layer's pixels for the tiles in `rect` (all if
    /// `None`) and returns the resulting document (selection respected).
    pub(super) fn filtered_doc(&mut self, f: &Filter, rect: Option<(DVec2, DVec2)>) -> Result<Document<Tile>, &'static str> {
        let layer = self.check_target(StrokeTarget::Pixels)?;
        let src: TileSet = self.doc.find(layer).and_then(|l| l.tiles()).map(|t| (**t).clone()).unwrap_or_default();
        let sel = self.doc.selection.clone();
        let spread = (3.0 * f.radius()).ceil() as i32;
        let reach = if matches!(f, Filter::GaussianBlur { .. }) { (spread + TILE_SIZE as i32 - 1) / TILE_SIZE as i32 } else { 0 };
        let dest: Vec<TileCoord> = dilate(src.keys().copied(), reach)
            .into_iter()
            .filter(|c| in_rect(c, rect))
            .filter(|c| sel.as_ref().is_none_or(|s| !matches!(s.at(*c), MaskRef::Solid(0))))
            .collect();
        let out: TileSet = match f {
            Filter::GaussianBlur { .. } => self.blur(&src, f.radius(), &dest).ok_or("GPU-Speicher voll")?,
            Filter::UnsharpMask { .. } | Filter::HighPass { .. } if !src.is_empty() => {
                let (kind, p) = match f {
                    Filter::UnsharpMask { amount, threshold, .. } => (UNSHARP, [amount / 100.0, *threshold, 0.0, 0.0]),
                    _ => (HIGH_PASS, [0.0; 4]),
                };
                let dest: Vec<TileCoord> = dest.into_iter().filter(|c| src.contains_key(c)).collect();
                let blurred = self.blur(&src, f.radius(), &dest).ok_or("GPU-Speicher voll")?;
                let mut grids = Grids::default();
                let g = grids.layer(&src);
                let jobs: Vec<FilterJob> = dest
                    .iter()
                    .map(|&c| FilterJob { blur_slot: blurred[&c].slot().packed(), p, ..Self::fjob(c, g, kind) })
                    .collect();
                let tiles = self.filter_jobs(&jobs, &grids.data).ok_or("GPU-Speicher voll")?;
                dest.into_iter().zip(tiles).collect()
            }
            Filter::AddNoise { amount, monochrome, gaussian, seed } => {
                let dest: Vec<TileCoord> = dest.into_iter().filter(|c| src.contains_key(c)).collect();
                let mut grids = Grids::default();
                let g = grids.layer(&src);
                let p = [amount / 100.0, *monochrome as u8 as f32, *gaussian as u8 as f32, *seed as f32];
                let jobs: Vec<FilterJob> = dest.iter().map(|&c| FilterJob { p, ..Self::fjob(c, g, NOISE) }).collect();
                let tiles = self.filter_jobs(&jobs, &grids.data).ok_or("GPU-Speicher voll")?;
                dest.into_iter().zip(tiles).collect()
            }
            _ => TileSet::new(),
        };
        // Through the selection: mix the old and the filtered tile.
        let out = match &sel {
            None => out,
            Some(_) => {
                let coords: Vec<TileCoord> = out.keys().copied().collect();
                let jobs = coords.iter().map(|&c| self.replace_job(layer, c, out[&c].slot().packed())).collect::<Vec<_>>();
                let tiles = self.run_mix(&jobs).ok_or("GPU-Speicher voll")?;
                coords.into_iter().zip(tiles).collect()
            }
        };
        let mut doc = self.doc.clone();
        let l = doc.find_mut(layer).unwrap();
        let mut map: TileSet = src;
        map.extend(out);
        l.content = Content::Pixels(Arc::new(map));
        Ok(doc)
    }

    /// Applies a filter to the active layer (within the selection) as one undo step.
    pub fn apply_filter(&mut self, f: &Filter, label: &str) -> Result<(), &'static str> {
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        let doc = self.filtered_doc(f, None)?;
        self.edit(label, |d| {
            *d = doc;
            true
        });
        Ok(())
    }

    /// Live preview of a filter on the visible area; `None` ends it.
    pub fn set_filter_preview(&mut self, f: Option<&Filter>) -> Result<(), &'static str> {
        match f {
            None => {
                if self.preview.take().is_some() {
                    self.recheck_all = true;
                }
                Ok(())
            }
            Some(f) => {
                let kind = PreviewKind::Filter(f.clone());
                if self.preview.as_ref().is_some_and(|p| p.kind == kind) {
                    return Ok(());
                }
                self.compute_preview(kind)
            }
        }
    }
}
