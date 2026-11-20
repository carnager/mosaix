//! Pixel filters on the active layer: Gaussian blur (separable; large radii
//! run on a downscaled copy), unsharp mask, high pass and noise. The result
//! is mixed in through the selection and recorded as one undo step; a live
//! preview computes only the visible area.

use std::collections::{BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{DAffine2, DVec2};
use imagine_core::doc::Content;
use imagine_core::{Document, Filter, LayerId, MaskRef, TILE_SIZE, TileCoord};

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

    /// Destination tiles of a filter on `src` (blur spreads beyond the source).
    fn filter_dest(src: &TileSet, f: &Filter) -> BTreeSet<TileCoord> {
        let spread = (3.0 * f.radius()).ceil() as i32;
        let reach = if matches!(f, Filter::GaussianBlur { .. }) { (spread + TILE_SIZE as i32 - 1) / TILE_SIZE as i32 } else { 0 };
        dilate(src.keys().copied(), reach)
    }

    /// Runs a filter on `src` for the tiles `dest` (no selection involved).
    fn run_filter_tiles(&mut self, src: &TileSet, f: &Filter, dest: Vec<TileCoord>) -> Result<TileSet, &'static str> {
        const FULL: &str = "GPU-Speicher voll";
        Ok(match f {
            Filter::GaussianBlur { .. } => self.blur(src, f.radius(), &dest).ok_or(FULL)?,
            Filter::UnsharpMask { .. } | Filter::HighPass { .. } if !src.is_empty() => {
                let (kind, p) = match f {
                    Filter::UnsharpMask { amount, threshold, .. } => (UNSHARP, [amount / 100.0, *threshold, 0.0, 0.0]),
                    _ => (HIGH_PASS, [0.0; 4]),
                };
                let dest: Vec<TileCoord> = dest.into_iter().filter(|c| src.contains_key(c)).collect();
                let blurred = self.blur(src, f.radius(), &dest).ok_or(FULL)?;
                let mut grids = Grids::default();
                let g = grids.layer(src);
                let jobs: Vec<FilterJob> = dest
                    .iter()
                    .map(|&c| FilterJob { blur_slot: blurred[&c].slot().packed(), p, ..Self::fjob(c, g, kind) })
                    .collect();
                let tiles = self.filter_jobs(&jobs, &grids.data).ok_or(FULL)?;
                dest.into_iter().zip(tiles).collect()
            }
            Filter::AddNoise { amount, monochrome, gaussian, seed } => {
                let dest: Vec<TileCoord> = dest.into_iter().filter(|c| src.contains_key(c)).collect();
                let mut grids = Grids::default();
                let g = grids.layer(src);
                let p = [amount / 100.0, *monochrome as u8 as f32, *gaussian as u8 as f32, *seed as f32];
                let jobs: Vec<FilterJob> = dest.iter().map(|&c| FilterJob { p, ..Self::fjob(c, g, NOISE) }).collect();
                let tiles = self.filter_jobs(&jobs, &grids.data).ok_or(FULL)?;
                dest.into_iter().zip(tiles).collect()
            }
            _ => TileSet::new(),
        })
    }

    /// The whole tile map with a filter applied (smart filters).
    pub(super) fn filter_tiles(&mut self, src: &TileSet, f: &Filter) -> Result<TileSet, &'static str> {
        let dest = Self::filter_dest(src, f).into_iter().collect();
        let out = self.run_filter_tiles(src, f, dest)?;
        let mut map = src.clone();
        map.extend(out);
        Ok(map)
    }

    /// Filters the active layer's pixels for the tiles in `rect` (all if
    /// `None`) and returns the resulting document (selection respected).
    pub(super) fn filtered_doc(&mut self, f: &Filter, rect: Option<(DVec2, DVec2)>) -> Result<Document<Tile>, &'static str> {
        let layer = self.check_target(StrokeTarget::Pixels)?;
        let src: TileSet = self.doc.find(layer).and_then(|l| l.tiles()).map(|t| (**t).clone()).unwrap_or_default();
        let sel = self.doc.selection.clone();
        let dest: Vec<TileCoord> = Self::filter_dest(&src, f)
            .into_iter()
            .filter(|c| in_rect(c, rect))
            .filter(|c| sel.as_ref().is_none_or(|s| !matches!(s.at(*c), MaskRef::Solid(0))))
            .collect();
        let out = self.run_filter_tiles(&src, f, dest)?;
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

    /// Adds a smart filter to the active pixel layer (one undo step).
    pub fn add_smart_filter(&mut self, f: &Filter, label: &str) -> Result<(), &'static str> {
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        let layer = self.check_target(StrokeTarget::Pixels)?;
        self.edit(label, |d| {
            d.find_mut(layer).unwrap().filters.push(imagine_core::SmartFilter::new(f.clone()));
            true
        });
        Ok(())
    }

    /// Writes a layer's smart filters into its pixels and removes them.
    pub fn bake_smart_filters(&mut self, layer: LayerId, label: &str) -> Result<(), &'static str> {
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        let l = self.doc.find(layer).ok_or("Ebene nicht gefunden")?;
        if l.filters.is_empty() {
            return Err("die Ebene hat keine Smart-Filter");
        }
        self.refresh_smart();
        let tiles = self.smart.get(&layer).map(|e| e.tiles.clone());
        self.edit(label, |d| {
            let l = d.find_mut(layer).unwrap();
            if let Some(t) = tiles {
                l.content = Content::Pixels(t);
            }
            l.filters.clear();
            true
        });
        Ok(())
    }

    /// Changes the parameters of a layer's smart filter (one undo step).
    pub fn set_smart_filter(&mut self, layer: LayerId, index: usize, f: &Filter, label: &str) -> Result<(), &'static str> {
        if self.preview.take().is_some() {
            self.recheck_all = true;
        }
        if self.doc.find(layer).is_none_or(|l| index >= l.filters.len()) {
            return Err("Smart-Filter nicht gefunden");
        }
        self.edit(label, |d| {
            let sf = &mut d.find_mut(layer).unwrap().filters[index];
            let changed = sf.filter != *f;
            sf.filter = f.clone();
            changed
        });
        Ok(())
    }

    /// Live preview of the active layer with smart filter `index` replaced
    /// (or, with `None`, one added); `None` ends it.
    pub fn set_smart_filter_preview(&mut self, f: Option<(Option<usize>, &Filter)>) -> Result<(), &'static str> {
        match f {
            None => self.set_filter_preview(None),
            Some((index, f)) => {
                let kind = PreviewKind::SmartFilter(index, f.clone());
                if self.preview.as_ref().is_some_and(|p| p.kind == kind) {
                    return Ok(());
                }
                self.compute_preview(kind)
            }
        }
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

/// Cached result of a layer's smart filters.
pub(super) struct SmartEntry {
    key: u64,
    /// The source the result was computed from (kept alive so the key stays unique).
    _src: Arc<TileSet>,
    tiles: Arc<TileSet>,
}

impl Engine {
    /// Brings the smart-filter results up to date and rebuilds the document
    /// that is composited (the shown document with filtered layer pixels).
    pub(super) fn refresh_smart(&mut self) {
        let base = self.preview.as_ref().map_or(&self.doc, |p| &p.doc);
        let mut want: Vec<(LayerId, u64, Arc<TileSet>, Vec<Filter>)> = Vec::new();
        base.for_each(&mut |l, _| {
            if let Some(t) = l.tiles() {
                let fs: Vec<Filter> = l.active_filters().cloned().collect();
                if !fs.is_empty() {
                    let mut h = DefaultHasher::new();
                    (Arc::as_ptr(t) as usize).hash(&mut h);
                    format!("{fs:?}").hash(&mut h);
                    want.push((l.id, h.finish(), t.clone(), fs));
                }
            }
        });
        if want.is_empty() {
            self.smart.clear();
            if self.render.take().is_some() {
                self.recheck_all = true;
            }
            return;
        }
        self.smart.retain(|id, _| want.iter().any(|w| w.0 == *id));
        let mut changed = false;
        for (id, key, src, fs) in want {
            if self.smart.get(&id).is_some_and(|e| e.key == key) {
                continue;
            }
            let t0 = std::time::Instant::now();
            let mut cur: TileSet = (*src).clone();
            for f in &fs {
                match self.filter_tiles(&cur, f) {
                    Ok(t) => cur = t,
                    Err(e) => {
                        tracing::warn!("Smart-Filter {}: {e}", f.name());
                        break;
                    }
                }
            }
            tracing::debug!("Smart-Filter für Ebene {id}: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
            self.smart.insert(id, SmartEntry { key, _src: src, tiles: Arc::new(cur) });
            changed = true;
        }
        if changed || self.render.is_none() || self.recheck_all || !self.recheck.is_empty() {
            fn substitute(ls: &mut [imagine_core::Layer<Tile>], smart: &HashMap<LayerId, SmartEntry>) {
                for l in ls {
                    match &mut l.content {
                        Content::Pixels(t) => {
                            if let Some(e) = smart.get(&l.id) {
                                *t = e.tiles.clone();
                            }
                        }
                        Content::Group(c) => substitute(c, smart),
                        Content::Adjustment(_) => {}
                    }
                }
            }
            let base = self.preview.as_ref().map_or(&self.doc, |p| &p.doc);
            let mut d = base.clone();
            substitute(&mut d.layers, &self.smart);
            self.render = Some(d);
            if changed {
                self.recheck_all = true;
            }
        }
    }
}
