//! Layer thumbnails for the layers panel: each pixel layer is downscaled on
//! the GPU (warp kernel, supersampled) into one slot of a small texture
//! array. Thumbnails are recomputed only when a layer's tiles change.

use std::collections::HashMap;
use std::sync::Arc;

use imagine_core::{LayerId, TILE_SIZE};

use super::Engine;
use crate::pool::{TILE_FORMAT, texel_copy};

pub const THUMB: u32 = 64;
const SLOTS: u32 = 256;

pub(super) struct Thumbs {
    texture: wgpu::Texture,
    /// Layer → (slot, content key it was rendered from).
    entries: HashMap<LayerId, (u32, usize)>,
    free: Vec<u32>,
    /// Layer masks: content key and THUMB × THUMB grey values.
    masks: HashMap<LayerId, (usize, Arc<Vec<u8>>)>,
}

impl Thumbs {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer thumbnails"),
            size: wgpu::Extent3d {
                width: THUMB,
                height: THUMB,
                depth_or_array_layers: SLOTS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TILE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Self {
            texture,
            entries: HashMap::new(),
            free: (0..SLOTS).rev().collect(),
            masks: HashMap::new(),
        }
    }
}

impl Engine {
    /// Brings up to `max` stale thumbnails up to date. Returns true if any changed.
    pub fn update_thumbnails(&mut self, max: usize) -> bool {
        if self.stroke.is_some() {
            return false;
        }
        let doc = &self.doc;
        let (w, h) = (doc.width.max(1), doc.height.max(1));
        // Content key: the tile map's identity plus the canvas size.
        let mut want: Vec<(LayerId, usize)> = Vec::new();
        let vectors = &self.vectors;
        doc.for_each(&mut |l, _| {
            // Text, shape and RAW layers: their rendered pixels.
            let rendered = l.is_rendered().then(|| vectors.get(&l.id).map(|(_, t)| t)).flatten();
            if let Some(t) = l.tiles().or(rendered) {
                want.push((
                    l.id,
                    Arc::as_ptr(t) as usize ^ ((w as usize) << 20) ^ h as usize,
                ));
            }
        });
        let thumbs = self.thumbs.get_or_insert_with(|| Thumbs::new(&self.device));
        // Forget removed layers.
        let gone: Vec<LayerId> = thumbs
            .entries
            .keys()
            .copied()
            .filter(|id| !want.iter().any(|(w, _)| w == id))
            .collect();
        for id in gone {
            let (slot, _) = thumbs.entries.remove(&id).unwrap();
            thumbs.free.push(slot);
        }
        let stale: Vec<(LayerId, usize)> = want
            .into_iter()
            .filter(|(id, key)| thumbs.entries.get(id).is_none_or(|e| e.1 != *key))
            .take(max)
            .collect();
        if stale.is_empty() {
            return false;
        }
        let scale = THUMB as f64 / w.max(h) as f64;
        let samples = ((1.0 / scale).ceil() as u32).clamp(1, 8);
        for (id, key) in stale {
            let Some(tile) = self.downscaled_tile(id, scale, samples) else {
                continue;
            };
            let thumbs = self.thumbs.as_mut().unwrap();
            let slot = match thumbs.entries.get(&id) {
                Some((slot, _)) => *slot,
                None => match thumbs.free.pop() {
                    Some(s) => s,
                    None => continue,
                },
            };
            thumbs.entries.insert(id, (slot, key));
            let s = tile.slot();
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_texture_to_texture(
                texel_copy(&self.layer_pool.chunk(s).texture, 0, s.layer as u32),
                texel_copy(&thumbs.texture, 0, slot),
                wgpu::Extent3d {
                    width: THUMB.min(TILE_SIZE),
                    height: THUMB.min(TILE_SIZE),
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit([enc.finish()]);
        }
        true
    }

    /// Brings up to `max` stale mask thumbnails up to date (GPU downscale,
    /// small readback). Returns true if any changed.
    pub fn update_mask_thumbnails(&mut self, max: usize) -> bool {
        if self.stroke.is_some() {
            return false;
        }
        let (w, h) = (self.doc.width.max(1), self.doc.height.max(1));
        let mut want: Vec<(LayerId, usize)> = Vec::new();
        self.doc.for_each(&mut |l, _| {
            if let Some(m) = &l.mask {
                want.push((l.id, Arc::as_ptr(&m.tiles) as usize ^ ((m.default as usize) << 8) ^ ((w as usize) << 20) ^ h as usize));
            }
        });
        let thumbs = self.thumbs.get_or_insert_with(|| Thumbs::new(&self.device));
        thumbs.masks.retain(|id, _| want.iter().any(|(w, _)| w == id));
        let stale: Vec<(LayerId, usize)> = want.into_iter().filter(|(id, k)| thumbs.masks.get(id).is_none_or(|e| e.0 != *k)).take(max).collect();
        if stale.is_empty() {
            return false;
        }
        let scale = THUMB as f64 / w.max(h) as f64;
        let samples = ((1.0 / scale).ceil() as u32).clamp(1, 8);
        for (id, key) in stale {
            let Some(tile) = self.downscaled_mask(id, scale, samples) else { continue };
            let raw = self.read_mask_tiles(&[tile]).pop().unwrap_or_default();
            let n = THUMB.min(TILE_SIZE) as usize;
            let ts = TILE_SIZE as usize;
            let mut px = vec![0u8; n * n];
            for y in 0..n {
                if raw.len() >= (y + 1) * ts {
                    px[y * n..(y + 1) * n].copy_from_slice(&raw[y * ts..y * ts + n]);
                }
            }
            self.thumbs.as_mut().unwrap().masks.insert(id, (key, Arc::new(px)));
        }
        true
    }

    /// The used part of a thumbnail (the document's aspect ratio in THUMB × THUMB).
    pub fn thumbnail_uv(&self) -> (f32, f32) {
        let (w, h) = (self.doc.width.max(1) as f32, self.doc.height.max(1) as f32);
        let k = THUMB as f32 / w.max(h);
        (w * k / THUMB as f32, h * k / THUMB as f32)
    }

    /// A layer's mask thumbnail: content key and THUMB × THUMB grey values
    /// (the used part has the size `thumbnail` reports).
    pub fn mask_thumbnail(&self, layer: LayerId) -> Option<(usize, Arc<Vec<u8>>)> {
        self.thumbs.as_ref()?.masks.get(&layer).cloned()
    }

    /// Thumbnail slot of a layer (if rendered) and the size of the image in
    /// it (the document's aspect ratio, fitted into THUMB × THUMB).
    pub fn thumbnail(&self, layer: LayerId) -> Option<(u32, (f32, f32))> {
        let (slot, _) = self.thumbs.as_ref()?.entries.get(&layer)?;
        let (w, h) = (self.doc.width.max(1) as f32, self.doc.height.max(1) as f32);
        let k = THUMB as f32 / w.max(h);
        Some((*slot, (w * k / THUMB as f32, h * k / THUMB as f32)))
    }

    /// A 2D view of one thumbnail slot (register it once with the UI).
    pub fn thumbnail_view(&mut self, slot: u32) -> wgpu::TextureView {
        let thumbs = self.thumbs.get_or_insert_with(|| Thumbs::new(&self.device));
        thumbs.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("thumbnail"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: slot,
            array_layer_count: Some(1),
            ..Default::default()
        })
    }
}
