//! GPU → CPU transfers. [`Readback`] is asynchronous: copies are submitted
//! immediately, the data is collected later without stalling the UI (used
//! for background saving). The blocking helpers build on it.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use half::f16;
use imagine_core::{LayerId, TILE_MIP_LEVELS, TILE_SIZE, TileCoord};

use super::Engine;
use crate::pool::{Pool, SlotId, Tile, texel_copy};

const BATCH: usize = 256;

/// Which pool a tile handle belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Layer,
    Mask,
    Composite,
}

/// An in-flight copy of tiles to the CPU.
pub struct Readback {
    buffers: Vec<(wgpu::Buffer, usize)>,
    mapped: Arc<AtomicUsize>,
    row: usize,
    padded_row: usize,
    rows: usize,
    /// Keeps the source tiles alive (their slots must not be reused meanwhile).
    _tiles: Vec<Tile>,
}

impl Readback {
    pub(super) fn start(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pool: &Pool,
        slots: &[SlotId],
        mip: u32,
        keep: Vec<Tile>,
    ) -> Self {
        let size = TILE_SIZE >> mip;
        let row = (size * pool.bytes_per_pixel()) as usize;
        let padded_row = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let tile_bytes = (padded_row * size as usize) as u64;
        let mapped = Arc::new(AtomicUsize::new(0));
        let mut buffers = Vec::new();
        for batch in slots.chunks(BATCH) {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: tile_bytes * batch.len() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            for (i, s) in batch.iter().enumerate() {
                enc.copy_texture_to_buffer(
                    texel_copy(&pool.chunk(*s).texture, mip, s.layer as u32),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buf,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: i as u64 * tile_bytes,
                            bytes_per_row: Some(padded_row as u32),
                            rows_per_image: Some(size),
                        },
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
            }
            queue.submit([enc.finish()]);
            let m = mapped.clone();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                r.expect("map readback");
                m.fetch_add(1, Ordering::Release);
            });
            buffers.push((buf, batch.len()));
        }
        Self {
            buffers,
            mapped,
            row,
            padded_row,
            rows: size as usize,
            _tiles: keep,
        }
    }

    /// Advances the GPU without blocking; true once all data is available.
    pub fn poll(&self, device: &wgpu::Device) -> bool {
        let _ = device.poll(wgpu::PollType::Poll);
        self.mapped.load(Ordering::Acquire) == self.buffers.len()
    }

    /// Blocks until the data is available and returns it (one `Vec<u8>` per tile).
    pub fn wait(self, device: &wgpu::Device) -> Vec<Vec<u8>> {
        while !self.poll(device) {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.take()
    }

    /// The data; call only after [`Readback::poll`] returned true.
    pub fn take(self) -> Vec<Vec<u8>> {
        assert_eq!(
            self.mapped.load(Ordering::Acquire),
            self.buffers.len(),
            "readback not finished"
        );
        let mut out = Vec::new();
        for (buf, n) in &self.buffers {
            {
                let data = buf.slice(..).get_mapped_range().expect("mapped range");
                for t in 0..*n {
                    let tile =
                        &data[t * self.padded_row * self.rows..][..self.padded_row * self.rows];
                    let mut px = Vec::with_capacity(self.row * self.rows);
                    for r in tile.chunks_exact(self.padded_row) {
                        px.extend_from_slice(&r[..self.row]);
                    }
                    out.push(px);
                }
            }
            buf.unmap();
        }
        out
    }
}

pub fn to_f16(raw: Vec<u8>) -> Vec<f16> {
    bytemuck::pod_collect_to_vec(&raw)
}

impl Engine {
    fn pool(&self, kind: PoolKind) -> &Pool {
        match kind {
            PoolKind::Layer => &self.layer_pool,
            PoolKind::Mask => &self.mask_pool,
            PoolKind::Composite => &self.comp_pool,
        }
    }

    /// Starts an asynchronous readback of mip 0 of the given tiles.
    pub fn begin_readback(&self, kind: PoolKind, tiles: &[Tile]) -> Readback {
        let slots: Vec<SlotId> = tiles.iter().map(Tile::slot).collect();
        Readback::start(
            &self.device,
            &self.queue,
            self.pool(kind),
            &slots,
            0,
            tiles.to_vec(),
        )
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Raw bytes of mip 0 of the given tiles (blocking).
    pub fn read_tiles_raw(&self, kind: PoolKind, tiles: &[Tile]) -> Vec<Vec<u8>> {
        if tiles.is_empty() {
            return Vec::new();
        }
        self.begin_readback(kind, tiles).wait(&self.device)
    }

    /// Reads mip 0 of layer tiles (blocking).
    pub fn read_layer_tiles(&self, tiles: &[Tile]) -> Vec<Vec<f16>> {
        self.read_tiles_raw(PoolKind::Layer, tiles)
            .into_iter()
            .map(to_f16)
            .collect()
    }

    /// Reads mask tiles (blocking), 8-bit values.
    pub fn read_mask_tiles(&self, tiles: &[Tile]) -> Vec<Vec<u8>> {
        self.read_tiles_raw(PoolKind::Mask, tiles)
    }

    /// Reads the flattened image of the given tiles (call [`Engine::flush`] first).
    pub fn read_composite(&self, coords: &[TileCoord]) -> Vec<Vec<f16>> {
        self.read_composite_mip(coords, 0)
    }

    /// Like [`Engine::read_composite`] at a mip level: tiles are `TILE_SIZE >> mip` px wide.
    pub fn read_composite_mip(&self, coords: &[TileCoord], mip: u32) -> Vec<Vec<f16>> {
        let mip = mip.min(TILE_MIP_LEVELS - 1);
        let n = ((TILE_SIZE >> mip).pow(2) * 4) as usize;
        let tiles: Vec<Option<Tile>> = coords
            .iter()
            .map(|c| self.comp.get(c).and_then(|e| e.tile.clone()))
            .collect();
        let present: Vec<Tile> = tiles.iter().flatten().cloned().collect();
        let slots: Vec<SlotId> = present.iter().map(Tile::slot).collect();
        let mut data = if slots.is_empty() {
            Vec::new().into_iter()
        } else {
            Readback::start(
                &self.device,
                &self.queue,
                &self.comp_pool,
                &slots,
                mip,
                present,
            )
            .wait(&self.device)
            .into_iter()
        };
        tiles
            .iter()
            .map(|t| {
                if t.is_some() {
                    to_f16(data.next().unwrap())
                } else {
                    vec![f16::ZERO; n]
                }
            })
            .collect()
    }

    /// All composited tiles (call [`Engine::flush`] first), for asynchronous export.
    pub fn composite_tiles(&self) -> Vec<(TileCoord, Tile)> {
        let mut v: Vec<_> = self
            .comp
            .iter()
            .filter_map(|(c, e)| e.tile.clone().map(|t| (*c, t)))
            .collect();
        v.sort_by_key(|(c, _)| *c);
        v
    }

    /// The document as it is composited: text and shapes rasterised, smart
    /// filters and layer styles applied (effect layers inserted under their
    /// layer). Call [`Engine::flush`] first.
    pub fn layers_as_shown(&self) -> imagine_core::Document<Tile> {
        self.render.clone().unwrap_or_else(|| self.doc.clone())
    }

    /// Layer ids → their pixel tiles; helper for callers that save documents.
    pub fn layer_tile(&self, layer: LayerId, c: TileCoord) -> Option<Tile> {
        self.doc.find(layer)?.tiles()?.get(&c).cloned()
    }
}
