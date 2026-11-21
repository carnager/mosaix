//! Selection and layer masks. Shape rasterisation, combining and feathering
//! run on the CPU (`mosaix_core::mask`); results are uploaded as sparse R8
//! tiles. Uniform tiles stay `Solid` and cost no GPU memory.

use std::collections::HashMap;
use std::sync::Arc;

use mosaix_core::mask::{CombineOp, CpuMask, MaskData, Shape, combine, feather, rasterize};
use mosaix_core::{LayerId, Mask, MaskTile, TileCoord};

use super::Engine;
use super::readback::PoolKind;
use crate::pool::Tile;

impl Engine {
    /// Downloads a GPU mask to the CPU.
    fn mask_to_cpu(&self, m: &Mask<Tile>) -> CpuMask {
        let pixel_tiles: Vec<(TileCoord, Tile)> = m
            .tiles
            .iter()
            .filter_map(|(c, t)| match t {
                MaskTile::Pixels(p) => Some((*c, p.clone())),
                MaskTile::Solid(_) => None,
            })
            .collect();
        let handles: Vec<Tile> = pixel_tiles.iter().map(|(_, t)| t.clone()).collect();
        let data = self.read_tiles_raw(PoolKind::Mask, &handles);
        let mut tiles: HashMap<TileCoord, MaskData> = m
            .tiles
            .iter()
            .filter_map(|(c, t)| match t {
                MaskTile::Solid(v) => Some((*c, MaskData::Solid(*v))),
                MaskTile::Pixels(_) => None,
            })
            .collect();
        tiles.extend(
            pixel_tiles
                .into_iter()
                .map(|(c, _)| c)
                .zip(data.into_iter().map(MaskData::Pixels)),
        );
        CpuMask {
            tiles,
            default: m.default,
        }
    }

    /// Uploads a CPU mask (compacted first). `None` if GPU memory ran out.
    fn mask_to_gpu(&mut self, mut m: CpuMask) -> Option<Mask<Tile>> {
        m.compact();
        let mut tiles = HashMap::with_capacity(m.tiles.len());
        for (c, t) in m.tiles {
            let t = match t {
                MaskData::Solid(v) => MaskTile::Solid(v),
                MaskData::Pixels(p) => MaskTile::Pixels(self.write_new_mask_tile(&p)?),
            };
            tiles.insert(c, t);
        }
        Some(Mask {
            tiles: Arc::new(tiles),
            default: m.default,
        })
    }

    /// Identity of the current selection (changes whenever it is replaced);
    /// used to discard background results computed from an older one.
    pub fn selection_id(&self) -> usize {
        self.doc
            .selection
            .as_ref()
            .map_or(0, |s| Arc::as_ptr(&s.tiles) as usize)
    }

    pub fn has_selection(&self) -> bool {
        self.doc.selection.is_some()
    }

    /// The selection on the CPU (`None` = nothing selected).
    pub fn selection_cpu(&self) -> Option<CpuMask> {
        self.doc.selection.as_ref().map(|m| self.mask_to_cpu(m))
    }

    /// Replaces the selection as one undo step. An empty mask deselects.
    pub fn set_selection(
        &mut self,
        mask: Option<CpuMask>,
        label: &str,
    ) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        let gpu = match mask.filter(|m| !m.is_empty()) {
            Some(m) => Some(
                self.mask_to_gpu(m)
                    .ok_or(mosaix_core::tr("GPU memory full"))?,
            ),
            None => None,
        };
        let had = self.doc.selection.is_some();
        if !had && gpu.is_none() {
            return Ok(());
        }
        self.edit(label, |d| {
            d.selection = gpu;
            true
        });
        Ok(())
    }

    /// Adds/subtracts/intersects a shape with the selection.
    pub fn select_shape(
        &mut self,
        shape: &Shape,
        op: CombineOp,
        label: &str,
    ) -> Result<(), &'static str> {
        let (w, h) = self.size();
        let new = rasterize(shape, w, h);
        let result = match self.selection_cpu() {
            Some(old) => combine(&old, &new, op),
            // Without a selection, subtracting leaves nothing; everything else is the shape.
            None if op == CombineOp::Subtract => CpuMask::filled(0),
            None => new,
        };
        self.set_selection(Some(result), label)
    }

    pub fn select_all(&mut self, label: &str) -> Result<(), &'static str> {
        self.set_selection(Some(CpuMask::filled(255)), label)
    }

    pub fn deselect(&mut self, label: &str) -> Result<(), &'static str> {
        self.set_selection(None, label)
    }

    pub fn invert_selection(&mut self, label: &str) -> Result<(), &'static str> {
        let sel = self.selection_cpu().unwrap_or(CpuMask::filled(0));
        self.set_selection(Some(sel.invert()), label)
    }

    pub fn feather_selection(&mut self, radius: f64, label: &str) -> Result<(), &'static str> {
        let Some(sel) = self.selection_cpu() else {
            return Err(mosaix_core::tr("no selection"));
        };
        let (w, h) = self.size();
        self.set_selection(Some(feather(&sel, radius, w, h)), label)
    }

    // ---- layer masks --------------------------------------------------------------

    pub fn layer_mask_cpu(&self, layer: LayerId) -> Option<CpuMask> {
        let m = self.doc.find(layer)?.mask.as_ref()?;
        Some(self.mask_to_cpu(m))
    }

    /// Sets (or removes) a layer's mask as one undo step.
    pub fn set_layer_mask(
        &mut self,
        layer: LayerId,
        mask: Option<CpuMask>,
        label: &str,
    ) -> Result<(), &'static str> {
        if self.stroke.is_some() {
            return Err(mosaix_core::tr("not possible during a brush stroke"));
        }
        if self.doc.find(layer).is_none() {
            return Err(mosaix_core::tr("layer does not exist"));
        }
        let gpu = match mask {
            Some(m) => Some(
                self.mask_to_gpu(m)
                    .ok_or(mosaix_core::tr("GPU memory full"))?,
            ),
            None => None,
        };
        self.edit(label, |d| {
            let l = d.find_mut(layer).unwrap();
            l.mask = gpu;
            l.mask_enabled = true;
            true
        });
        Ok(())
    }

    /// Adds an adjustment layer above the active one; with a selection, the
    /// selection becomes its mask (shared tiles, no copy).
    pub fn add_adjustment(
        &mut self,
        adj: mosaix_core::Adjustment,
        label: &str,
    ) -> Option<LayerId> {
        let mut id = None;
        self.edit(label, |d| {
            let new = d.add_adjustment(adj);
            // Like Photoshop: always a mask to paint into (free while uniform).
            let mask = d.selection.clone().unwrap_or_else(|| Mask {
                tiles: Arc::new(HashMap::new()),
                default: 255,
            });
            let l = d.find_mut(new).unwrap();
            l.mask = Some(mask);
            l.mask_enabled = true;
            id = Some(new);
            true
        });
        id
    }

    /// Adds a mask to the active layer: from the selection, or revealing everything.
    pub fn add_layer_mask(&mut self, label: &str) -> Result<(), &'static str> {
        let layer = self.doc.active;
        if self.doc.find(layer).is_some_and(|l| l.mask.is_some()) {
            return Err(mosaix_core::tr("the layer already has a mask"));
        }
        let mask = self.selection_cpu().unwrap_or(CpuMask::filled(255));
        self.set_layer_mask(layer, Some(mask), label)
    }

    /// Loads a mask from a file without creating an undo step.
    pub fn load_mask(
        &mut self,
        layer: Option<LayerId>,
        mask: CpuMask,
        enabled: bool,
    ) -> Option<Vec<(TileCoord, Tile)>> {
        let gpu = self.mask_to_gpu(mask)?;
        let uploaded: Vec<(TileCoord, Tile)> = gpu
            .tiles
            .iter()
            .filter_map(|(c, t)| match t {
                MaskTile::Pixels(p) => Some((*c, p.clone())),
                MaskTile::Solid(_) => None,
            })
            .collect();
        match layer {
            Some(id) => {
                let l = self.doc.find_mut(id)?;
                l.mask = Some(gpu);
                l.mask_enabled = enabled;
                self.recheck_all = true;
            }
            None => self.doc.selection = Some(gpu),
        }
        Some(uploaded)
    }
}
