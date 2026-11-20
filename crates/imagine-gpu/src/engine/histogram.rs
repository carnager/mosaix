//! Histograms of the image underneath a layer (the input of an adjustment
//! layer). The document below is flattened in batches into temporary
//! composite tiles, a mip level of them is read back and binned on the CPU
//! in straight-alpha sRGB-encoded values (what levels and curves work on).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, OnceLock};

use half::f16;
use imagine_core::color::linear_to_srgb;
use imagine_core::doc::hash_program;
use imagine_core::{LayerId, TILE_MIP_LEVELS, TILE_SIZE, TileCoord};

use super::frame::FlattenBatch;
use super::readback::{Readback, to_f16};
use super::{Engine, mask_slot};
use crate::pool::Tile;

/// Tiles flattened per batch (bounds the temporary composite memory).
const BATCH: usize = 64;
/// Aim for at most about this many samples.
const MAX_SAMPLES: u64 = 4 << 20;

/// 256 bins per channel: 0 = RGB (all three channels), 1–3 = red, green,
/// blue. Each pixel counts with its alpha.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    pub bins: [[f32; 256]; 4],
}

impl Histogram {
    /// The largest bin of a channel, ignoring the two extremes so clipped
    /// pixels don't flatten the rest of the graph.
    pub fn peak(&self, ch: usize) -> f32 {
        self.bins[ch][1..255].iter().copied().fold(0.0, f32::max).max(1e-6)
    }
}

/// f16 bits of a straight linear value → its sRGB-encoded bin.
fn bin_table() -> &'static [u8; 65536] {
    static T: OnceLock<Box<[u8; 65536]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Box::new([0u8; 65536]);
        for (bits, b) in t.iter_mut().enumerate() {
            let v = f16::from_bits(bits as u16).to_f32();
            let v = if v.is_nan() { 0.0 } else { v };
            *b = (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0).round() as u8;
        }
        t
    })
}

fn add_tile(h: &mut Histogram, px: &[f16], (x0, y0): (i64, i64), size: usize, step: i64, (w, h_doc): (i64, i64)) {
    let table = bin_table();
    for y in 0..size {
        let dy = y0 + y as i64 * step;
        if dy >= h_doc {
            break;
        }
        for x in 0..size {
            let dx = x0 + x as i64 * step;
            if dx >= w {
                break;
            }
            let p = &px[(y * size + x) * 4..][..4];
            let a = p[3].to_f32();
            if a <= 1.0 / 512.0 {
                continue;
            }
            let ch = |c: f16| {
                if a >= 0.999 { table[c.to_bits() as usize] } else { table[f16::from_f32(c.to_f32() / a).to_bits() as usize] }
            };
            let (r, g, b) = (ch(p[0]), ch(p[1]), ch(p[2]));
            h.bins[1][r as usize] += a;
            h.bins[2][g as usize] += a;
            h.bins[3][b as usize] += a;
            h.bins[0][r as usize] += a;
            h.bins[0][g as usize] += a;
            h.bins[0][b as usize] += a;
        }
    }
}

impl Engine {
    /// Histogram of the image underneath layer `below` (the whole image if
    /// `None`). Cached until what is underneath changes.
    pub fn histogram(&mut self, below: Option<LayerId>) -> Arc<Histogram> {
        let doc = match below {
            Some(id) => self.doc.below(id),
            None => self.doc.clone(),
        };
        let coords: Vec<TileCoord> = self.all_doc_coords();
        // Key: the compositing program of every tile.
        let mut hasher = DefaultHasher::new();
        (doc.width, doc.height, doc.blend_space as u32).hash(&mut hasher);
        for &c in &coords {
            let ops = doc.program(c, None);
            c.hash(&mut hasher);
            hash_program(&ops, |t: &Tile| t.serial(), &mut hasher);
        }
        let key = hasher.finish();
        if let Some((k, h)) = &self.histogram
            && *k == key
        {
            return h.clone();
        }

        let pixels = doc.width as u64 * doc.height as u64;
        let mut mip = 0u32;
        while mip + 1 < TILE_MIP_LEVELS && pixels >> (2 * mip) > MAX_SAMPLES {
            mip += 1;
        }
        let size = (TILE_SIZE >> mip) as usize;
        let step = 1i64 << mip;
        let dims = (doc.width as i64, doc.height as i64);
        let mut hist = Histogram { bins: [[0.0; 256]; 4] };
        let saved_mips = std::mem::take(&mut self.pending_mips);
        let work: Vec<TileCoord> = coords.into_iter().filter(|c| !doc.program(*c, None).is_empty()).collect();
        for chunk in work.chunks(BATCH) {
            let mut b = FlattenBatch::default();
            let mut tiles: Vec<(TileCoord, Tile)> = Vec::new();
            for &c in chunk {
                let Some(t) = self.comp_pool.alloc() else {
                    tracing::warn!("Composite-Speicher voll – Histogramm unvollständig");
                    break;
                };
                b.push(&doc.program(c, None), None, mask_slot(None), t.slot());
                self.pending_mips.push(t.slot());
                tiles.push((c, t));
            }
            let mut enc = self.device.create_command_encoder(&Default::default());
            self.dispatch_flatten(&mut enc, b);
            self.encode_mips(&mut enc);
            self.params.upload(&self.queue);
            self.queue.submit([enc.finish()]);
            let slots: Vec<_> = tiles.iter().map(|(_, t)| t.slot()).collect();
            let keep: Vec<Tile> = tiles.iter().map(|(_, t)| t.clone()).collect();
            let data = Readback::start(&self.device, &self.queue, &self.comp_pool, &slots, mip, keep).wait(&self.device);
            for ((c, _), raw) in tiles.iter().zip(data) {
                let (x, y) = c.origin();
                add_tile(&mut hist, &to_f16(raw), (x, y), size, step, dims);
            }
        }
        self.pending_mips = saved_mips;
        let hist = Arc::new(hist);
        self.histogram = Some((key, hist.clone()));
        hist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_maps_linear_to_srgb_bins() {
        let t = bin_table();
        assert_eq!(t[f16::from_f32(0.0).to_bits() as usize], 0);
        assert_eq!(t[f16::from_f32(1.0).to_bits() as usize], 255);
        assert_eq!(t[f16::from_f32(0.2159).to_bits() as usize], 128);
        assert_eq!(t[f16::from_f32(-0.5).to_bits() as usize], 0);
        assert_eq!(t[f16::from_f32(3.0).to_bits() as usize], 255);
    }
}
