//! Chunked pools of 256² tiles in GPU texture arrays.
//!
//! A pool grows by adding fixed-size chunks (one texture array each), so it
//! never copies existing data. Tiles are handed out as ref-counted [`Tile`]
//! handles; when the last handle (in the document *or* in an undo snapshot)
//! goes away, the slot returns to the free list.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use imagine_core::TILE_SIZE;

pub const TILE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const BYTES_PER_PIXEL: u32 = 8;

/// Position of a tile: chunk index and array layer within the chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId {
    pub chunk: u16,
    pub layer: u16,
}

impl SlotId {
    /// Packed form used by shaders: `chunk << 16 | layer`.
    pub fn packed(self) -> u32 {
        (self.chunk as u32) << 16 | self.layer as u32
    }
}

struct TileInner {
    slot: SlotId,
    serial: u64,
    free: Arc<Mutex<Vec<SlotId>>>,
}

impl Drop for TileInner {
    fn drop(&mut self) {
        if let Ok(mut f) = self.free.lock() {
            f.push(self.slot);
        }
    }
}

/// Ref-counted handle to a tile slot. Its content is immutable once the
/// operation that produced it has finished, so `serial` identifies content.
#[derive(Clone)]
pub struct Tile(Arc<TileInner>);

impl Tile {
    pub fn slot(&self) -> SlotId {
        self.0.slot
    }

    /// Unique per allocation (never reused, unlike slots).
    pub fn serial(&self) -> u64 {
        self.0.serial
    }
}

impl std::fmt::Debug for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tile#{}@{}:{}", self.0.serial, self.0.slot.chunk, self.0.slot.layer)
    }
}

pub struct Chunk {
    pub texture: wgpu::Texture,
    /// All mips, all layers.
    pub view: wgpu::TextureView,
    /// One D2Array view per mip level (for compute read/write).
    pub mip_views: Vec<wgpu::TextureView>,
    layer_targets: Vec<Option<wgpu::TextureView>>,
}

pub struct PoolConfig {
    pub label: &'static str,
    pub mip_levels: u32,
    pub chunk_layers: u32,
    pub max_chunks: u32,
    pub usage: wgpu::TextureUsages,
}

pub struct Pool {
    device: wgpu::Device,
    cfg: PoolConfig,
    pub chunks: Vec<Chunk>,
    free: Arc<Mutex<Vec<SlotId>>>,
    /// Next never-used layer in the last chunk.
    fresh: u32,
    serial: Arc<AtomicU64>,
    /// Bumped whenever a chunk is added (bind groups referencing chunks must be rebuilt).
    pub generation: u64,
}

impl Pool {
    pub fn new(device: &wgpu::Device, cfg: PoolConfig, serial: Arc<AtomicU64>) -> Self {
        let mut p = Self {
            device: device.clone(),
            cfg,
            chunks: Vec::new(),
            free: Arc::new(Mutex::new(Vec::new())),
            fresh: 0,
            serial,
            generation: 0,
        };
        p.add_chunk();
        p
    }

    fn add_chunk(&mut self) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(self.cfg.label),
            size: wgpu::Extent3d {
                width: TILE_SIZE,
                height: TILE_SIZE,
                depth_or_array_layers: self.cfg.chunk_layers,
            },
            mip_level_count: self.cfg.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TILE_FORMAT,
            usage: self.cfg.usage,
            view_formats: &[],
        });
        let array_view = |base: u32, count: Option<u32>| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: base,
                mip_level_count: count,
                ..Default::default()
            })
        };
        let view = array_view(0, None);
        let mip_views = (0..self.cfg.mip_levels).map(|m| array_view(m, Some(1))).collect();
        self.chunks.push(Chunk {
            texture,
            view,
            mip_views,
            layer_targets: vec![None; self.cfg.chunk_layers as usize],
        });
        self.fresh = 0;
        self.generation += 1;
        tracing::debug!(pool = self.cfg.label, chunks = self.chunks.len(), "pool chunk added");
    }

    /// Allocates a slot. Contents are undefined; callers overwrite them fully.
    pub fn alloc(&mut self) -> Option<Tile> {
        let reused = self.free.lock().ok().and_then(|mut f| f.pop());
        let slot = match reused {
            Some(s) => s,
            None => {
                if self.fresh >= self.cfg.chunk_layers {
                    if self.chunks.len() as u32 >= self.cfg.max_chunks {
                        return None;
                    }
                    self.add_chunk();
                }
                let s = SlotId { chunk: (self.chunks.len() - 1) as u16, layer: self.fresh as u16 };
                self.fresh += 1;
                s
            }
        };
        Some(Tile(Arc::new(TileInner {
            slot,
            serial: self.serial.fetch_add(1, Ordering::Relaxed),
            free: self.free.clone(),
        })))
    }

    pub fn chunk(&self, slot: SlotId) -> &Chunk {
        &self.chunks[slot.chunk as usize]
    }

    /// Render-target view of mip 0 of one slot (cached).
    pub fn target(&mut self, slot: SlotId) -> &wgpu::TextureView {
        let chunk = &mut self.chunks[slot.chunk as usize];
        let texture = &chunk.texture;
        chunk.layer_targets[slot.layer as usize].get_or_insert_with(|| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("tile target"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: slot.layer as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
    }

    /// Slots currently referenced by at least one handle.
    pub fn live(&self) -> usize {
        let total = self.chunks.len().saturating_sub(1) * self.cfg.chunk_layers as usize + self.fresh as usize;
        total - self.free.lock().map(|f| f.len()).unwrap_or(0)
    }

    pub fn capacity_tiles(&self) -> u64 {
        self.cfg.max_chunks as u64 * self.cfg.chunk_layers as u64
    }

    /// Bytes of GPU memory allocated by this pool.
    pub fn allocated_bytes(&self) -> u64 {
        let per_layer: u64 = (0..self.cfg.mip_levels)
            .map(|m| ((TILE_SIZE >> m) as u64).pow(2) * BYTES_PER_PIXEL as u64)
            .sum();
        per_layer * self.cfg.chunk_layers as u64 * self.chunks.len() as u64
    }
}

pub fn texel_copy(texture: &wgpu::Texture, mip: u32, layer: u32) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: mip,
        origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
        aspect: wgpu::TextureAspect::All,
    }
}
