//! Magic wand and "select colour range": the GPU computes a colour
//! similarity map (one mask tile per document tile), the CPU flood-fills the
//! contiguous region and combines the result with the selection.

use bytemuck::{Pod, Zeroable};
use mosaix_core::color::linear_to_srgb;
use mosaix_core::mask::{CombineOp, CpuMask, MaskData, combine, flood};
use mosaix_core::{TILE_SIZE, TileCoord};

use super::Engine;
use super::readback::PoolKind;
use crate::pool::Tile;

/// Which pixels the wand looks at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleFrom {
    /// The active layer only (Photoshop's default).
    #[default]
    Layer,
    /// The flattened image ("sample all layers").
    Image,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SimilarJob {
    reference: [f32; 4],
    slot: u32,
    from_composite: u32,
    tolerance: f32,
    band: f32,
}

/// Straight sRGB (0..1) + alpha of a premultiplied linear pixel.
fn straight(px: [f32; 4]) -> [f32; 4] {
    if px[3] <= 1e-6 {
        return [0.0; 4];
    }
    let u = |v: f32| linear_to_srgb((v / px[3]).clamp(0.0, 1.0));
    [u(px[0]), u(px[1]), u(px[2]), px[3].min(1.0)]
}

/// CPU twin of the shader's distance (for missing, i.e. transparent, tiles).
fn similarity(a: [f32; 4], reference: [f32; 4], tolerance: f32, band: f32) -> u8 {
    let dc = (0..3)
        .map(|i| (a[i] - reference[i]).abs())
        .fold(0.0f32, f32::max)
        * a[3].min(reference[3]);
    let d = dc.max((a[3] - reference[3]).abs()) * 255.0;
    if d <= tolerance {
        255
    } else {
        (((tolerance + band - d) / band.max(1e-3)).clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

impl Engine {
    /// Source tiles for sampling (premultiplied linear), keyed by document tile.
    fn sample_tiles(&mut self, from: SampleFrom) -> Result<Vec<(TileCoord, Tile)>, &'static str> {
        Ok(match from {
            SampleFrom::Layer => {
                let l = self
                    .doc
                    .active_layer()
                    .ok_or(mosaix_core::tr("no active layer"))?;
                let t = l.tiles().ok_or(mosaix_core::tr(
                    "groups have no pixels – choose a layer or sample all layers",
                ))?;
                t.iter()
                    .filter(|(c, _)| self.in_document(**c))
                    .map(|(c, t)| (*c, t.clone()))
                    .collect()
            }
            SampleFrom::Image => {
                self.flush();
                self.comp
                    .iter()
                    .filter_map(|(c, e)| e.tile.clone().map(|t| (*c, t)))
                    .collect()
            }
        })
    }

    /// Straight sRGB colour + alpha at a document pixel.
    pub fn sample_straight(
        &mut self,
        x: u32,
        y: u32,
        from: SampleFrom,
    ) -> Result<[f32; 4], &'static str> {
        let (w, h) = self.size();
        if x >= w || y >= h {
            return Err(mosaix_core::tr("point lies outside the image"));
        }
        let c = TileCoord::containing(x as f64, y as f64);
        let tiles = self.sample_tiles(from)?;
        let Some((_, t)) = tiles.into_iter().find(|(tc, _)| *tc == c) else {
            return Ok([0.0; 4]);
        };
        let kind = if from == SampleFrom::Image {
            PoolKind::Composite
        } else {
            PoolKind::Layer
        };
        let raw = self.read_tiles_raw(kind, &[t]).remove(0);
        let px: &[half::f16] = bytemuck::cast_slice(&raw);
        let (ox, oy) = c.origin();
        let i = ((y as i64 - oy) as usize * TILE_SIZE as usize + (x as i64 - ox) as usize) * 4;
        Ok(straight(
            [px[i], px[i + 1], px[i + 2], px[i + 3]].map(half::f16::to_f32),
        ))
    }

    /// Average straight sRGB colour + alpha of the square of `radius` px
    /// around a document pixel (0 = that pixel; eyedropper "3×3", "5×5" …).
    pub fn sample_average(
        &mut self,
        x: i64,
        y: i64,
        radius: i64,
        from: SampleFrom,
    ) -> Result<[f32; 4], &'static str> {
        let (w, h) = self.size();
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            return Err(mosaix_core::tr("point lies outside the image"));
        }
        let (x0, y0) = ((x - radius).max(0), (y - radius).max(0));
        let (x1, y1) = (
            (x + radius).min(w as i64 - 1),
            (y + radius).min(h as i64 - 1),
        );
        let wanted: Vec<TileCoord> = {
            let (a, b) = (
                TileCoord::containing(x0 as f64, y0 as f64),
                TileCoord::containing(x1 as f64, y1 as f64),
            );
            (a.y..=b.y)
                .flat_map(|ty| (a.x..=b.x).map(move |tx| TileCoord::new(tx, ty)))
                .collect()
        };
        let tiles: Vec<(TileCoord, Tile)> = self
            .sample_tiles(from)?
            .into_iter()
            .filter(|(c, _)| wanted.contains(c))
            .collect();
        let kind = if from == SampleFrom::Image {
            PoolKind::Composite
        } else {
            PoolKind::Layer
        };
        let raws = self.read_tiles_raw(
            kind,
            &tiles.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>(),
        );
        // Sum premultiplied linear values; missing tiles count as transparent.
        let mut sum = [0.0f64; 4];
        let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
        for ((c, _), raw) in tiles.iter().zip(&raws) {
            let px: &[half::f16] = bytemuck::cast_slice(raw);
            let (ox, oy) = c.origin();
            let ts = TILE_SIZE as i64;
            for py in y0.max(oy)..=y1.min(oy + ts - 1) {
                for qx in x0.max(ox)..=x1.min(ox + ts - 1) {
                    let i = ((py - oy) as usize * TILE_SIZE as usize + (qx - ox) as usize) * 4;
                    for k in 0..4 {
                        sum[k] += px[i + k].to_f32() as f64;
                    }
                }
            }
        }
        Ok(straight(sum.map(|v| (v / n) as f32)))
    }

    /// Similarity of every document pixel to `reference` (straight sRGB +
    /// alpha), 255 within `tolerance` (8-bit steps), fading over `band`.
    pub fn similarity_map(
        &mut self,
        reference: [f32; 4],
        tolerance: f32,
        band: f32,
        from: SampleFrom,
    ) -> Result<CpuMask, &'static str> {
        let tiles = self.sample_tiles(from)?;
        let composite = (from == SampleFrom::Image) as u32;
        let jobs: Vec<SimilarJob> = tiles
            .iter()
            .map(|(_, t)| SimilarJob {
                reference,
                slot: t.slot().packed(),
                from_composite: composite,
                tolerance,
                band,
            })
            .collect();
        let out = self
            .run_similar(&jobs)
            .ok_or(mosaix_core::tr("GPU memory full"))?;
        let data = self.read_tiles_raw(PoolKind::Mask, &out);
        drop(out);
        // Tiles without content are transparent.
        let transparent = similarity([0.0; 4], reference, tolerance, band);
        let mut m = CpuMask::filled(0);
        for c in self.all_doc_coords() {
            m.tiles.insert(c, MaskData::Solid(transparent));
        }
        for ((c, _), px) in tiles.iter().zip(data) {
            m.tiles.insert(*c, MaskData::Pixels(px));
        }
        m.compact();
        Ok(m)
    }

    /// Magic wand at `(x, y)`: selects similar colours (`contiguous`: only the
    /// connected region) and combines them with the selection.
    #[allow(clippy::too_many_arguments)]
    pub fn magic_wand(
        &mut self,
        x: u32,
        y: u32,
        tolerance: f32,
        contiguous: bool,
        antialias: bool,
        from: SampleFrom,
        op: CombineOp,
        label: &str,
    ) -> Result<(), &'static str> {
        let reference = self.sample_straight(x, y, from)?;
        let band = if antialias {
            (tolerance * 0.5).max(12.0)
        } else {
            0.0
        };
        let map = self.similarity_map(reference, tolerance, band, from)?;
        let (w, h) = self.size();
        let region = if contiguous {
            flood(&map, x, y, w, h)
        } else {
            map
        };
        self.combine_selection(region, op, label)
    }

    /// Selects all pixels similar to `reference` (straight sRGB + alpha);
    /// `fuzziness` (0…200, like Photoshop) widens the soft range.
    pub fn select_color_range(
        &mut self,
        reference: [f32; 4],
        fuzziness: f32,
        from: SampleFrom,
        op: CombineOp,
        label: &str,
    ) -> Result<(), &'static str> {
        let map = self.similarity_map(reference, fuzziness * 0.25, fuzziness * 0.75 + 1.0, from)?;
        self.combine_selection(map, op, label)
    }

    fn combine_selection(
        &mut self,
        new: CpuMask,
        op: CombineOp,
        label: &str,
    ) -> Result<(), &'static str> {
        let result = match self.selection_cpu() {
            Some(old) => combine(&old, &new, op),
            None if op == CombineOp::Subtract => CpuMask::filled(0),
            None => new,
        };
        self.set_selection(Some(result), label)
    }

    /// Paint Bucket at `(x, y)`: fills the similar (`contiguous`: connected)
    /// region with `color`, within the selection if there is one. The
    /// selection itself stays as it was. One undo step.
    #[allow(clippy::too_many_arguments)]
    pub fn bucket_fill(
        &mut self,
        x: u32,
        y: u32,
        tolerance: f32,
        contiguous: bool,
        antialias: bool,
        from: SampleFrom,
        color: [f32; 4],
        opacity: f32,
        label: &str,
    ) -> Result<usize, &'static str> {
        self.check_target(super::StrokeTarget::Pixels)?;
        let reference = self.sample_straight(x, y, from)?;
        let band = if antialias { (tolerance * 0.5).max(12.0) } else { 0.0 };
        let map = self.similarity_map(reference, tolerance, band, from)?;
        let (w, h) = self.size();
        let region = if contiguous { flood(&map, x, y, w, h) } else { map };
        let region = match self.selection_cpu() {
            Some(sel) => combine(&sel, &region, CombineOp::Intersect),
            None => region,
        };
        if region.is_empty() {
            return Err(mosaix_core::tr("nothing to fill there"));
        }
        self.fill_through(region, color, opacity, label)
    }

    /// Fills the active pixel layer through `region` (instead of the
    /// selection, which stays as it is). One undo step.
    fn fill_through(&mut self, region: CpuMask, color: [f32; 4], opacity: f32, label: &str) -> Result<usize, &'static str> {
        let gpu = self.mask_to_gpu(region).ok_or(mosaix_core::tr("GPU memory full"))?;
        let old = self.doc.selection.replace(gpu);
        let r = self.fill(super::StrokeTarget::Pixels, color, opacity, label);
        self.doc.selection = old.clone();
        if r.is_ok()
            && let Some(before) = self.history.last_before_mut()
        {
            before.selection = old;
        }
        self.recheck_all = true;
        r
    }

    /// Edit › Stroke: a line of `width` px along the selection's edge,
    /// inside it, centred on it or outside it. One undo step.
    pub fn stroke_selection(
        &mut self,
        width: f64,
        position: mosaix_core::style::StrokePosition,
        color: [f32; 4],
        opacity: f32,
        label: &str,
    ) -> Result<usize, &'static str> {
        use mosaix_core::mask::{Modify, modify};
        use mosaix_core::style::StrokePosition as P;
        self.check_target(super::StrokeTarget::Pixels)?;
        let sel = self.selection_cpu().ok_or(mosaix_core::tr("nothing is selected"))?;
        if width < 1.0 {
            return Err(mosaix_core::tr("the width must be at least 1 px"));
        }
        let (w, h) = self.size();
        let (outer, inner) = match position {
            P::Inside => (sel.clone(), modify(&sel, Modify::Contract(width), w, h)),
            P::Outside => (modify(&sel, Modify::Expand(width), w, h), sel.clone()),
            P::Center => (modify(&sel, Modify::Expand(width / 2.0), w, h), modify(&sel, Modify::Contract(width / 2.0), w, h)),
        };
        let band = combine(&outer, &inner, CombineOp::Subtract);
        if band.is_empty() {
            return Err(mosaix_core::tr("nothing to stroke"));
        }
        self.fill_through(band, color, opacity, label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_matches_transparent_reference() {
        assert_eq!(similarity([0.0; 4], [0.0; 4], 0.0, 0.0), 255);
        assert_eq!(similarity([0.0; 4], [1.0, 0.0, 0.0, 1.0], 32.0, 0.0), 0);
        // Colour differences only count where both are visible.
        assert_eq!(similarity([0.0; 4], [1.0, 1.0, 1.0, 0.0], 0.0, 0.0), 255);
    }

    #[test]
    fn band_fades_linearly() {
        let r = [0.5, 0.5, 0.5, 1.0];
        let a = [0.5 + 40.0 / 255.0, 0.5, 0.5, 1.0];
        assert_eq!(similarity(a, r, 30.0, 20.0), 128);
    }
}
