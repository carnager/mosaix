//! An open document: the engine plus its file state (incremental saving,
//! export). Used by the window and by the headless CLI.
//!
//! Saving runs in the background: [`Session::start_save`] snapshots the
//! document (cheap – tiles are shared and immutable) and starts an
//! asynchronous GPU readback of the tiles not yet in the file; once the data
//! arrived, compression and writing happen on a worker thread. Editing can
//! continue meanwhile.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use mosaix_core::mask::CpuMask;
use mosaix_core::{Document, LayerId, MaskTile, TileCoord, tr};
use mosaix_gpu::{Engine, PoolKind, Readback, Tile};
use mosaix_io::export::ExportOptions;
use mosaix_io::native::{Writer, compress_mask, compress_tile, live_bytes};
use mosaix_io::{BlobRef, export};
use rayon::prelude::*;

use crate::loader::{self, LoaderMsg, MaskLoad, Source, TileData};

pub struct SaveReport {
    pub path: PathBuf,
    pub tiles_written: usize,
    pub bytes: u64,
    pub incremental: bool,
    pub ms: f64,
}

/// Result of the worker thread, applied with [`Session::complete_save`].
pub struct SaveDone {
    report: SaveReport,
    refs: HashMap<u64, BlobRef>,
}

/// A save whose GPU readback is still in flight.
pub struct SaveJob {
    path: PathBuf,
    t0: Instant,
    incremental: bool,
    file_len: Option<u64>,
    /// Document with tiles replaced by their serials.
    doc: Document<u64>,
    refs: HashMap<u64, BlobRef>,
    layer_tiles: Vec<Tile>,
    mask_tiles: Vec<Tile>,
    layer_rb: Option<Readback>,
    mask_rb: Option<Readback>,
}

impl SaveJob {
    /// True once all tile data has arrived (non-blocking).
    pub fn ready(&self, device: &wgpu::Device) -> bool {
        self.layer_rb.as_ref().is_none_or(|r| r.poll(device))
            && self.mask_rb.as_ref().is_none_or(|r| r.poll(device))
    }

    /// Takes the data and returns the CPU part (compress + write) as a closure
    /// that can run on any thread.
    pub fn into_writer(self) -> impl FnOnce() -> anyhow::Result<SaveDone> + Send + 'static {
        let layer_data = self.layer_rb.map(Readback::take).unwrap_or_default();
        let mask_data = self.mask_rb.map(Readback::take).unwrap_or_default();
        let layer_serials: Vec<u64> = self.layer_tiles.iter().map(Tile::serial).collect();
        let mask_serials: Vec<u64> = self.mask_tiles.iter().map(Tile::serial).collect();
        let (path, t0, incremental, file_len, doc, mut refs) = (
            self.path,
            self.t0,
            self.incremental,
            self.file_len,
            self.doc,
            self.refs,
        );
        move || {
            let mut writer = if incremental {
                Writer::append(&path, file_len.expect("incremental save has a length"))?
            } else {
                Writer::create(&path)?
            };
            let layer_blobs: Vec<Vec<u8>> = layer_data
                .par_iter()
                .map(|raw| compress_tile(&bytemuck::pod_collect_to_vec(raw)))
                .collect();
            let mask_blobs: Vec<Vec<u8>> =
                mask_data.par_iter().map(|raw| compress_mask(raw)).collect();
            let tiles_written = layer_blobs.len() + mask_blobs.len();
            for (serial, blob) in layer_serials
                .iter()
                .zip(layer_blobs)
                .chain(mask_serials.iter().zip(mask_blobs))
            {
                refs.insert(*serial, writer.write_blob(&blob)?);
            }
            let doc_blobs = doc.map_tiles(&mut |s| refs[s]);
            writer.reuse(&refs);
            let (len, raw_refs) = writer
                .finish_refs(&doc_blobs)
                .with_context(|| tr!("saving {path}", path = path.display()))?;
            refs.extend(raw_refs);
            debug_assert!(live_bytes(&doc_blobs) <= len);
            let report = SaveReport {
                path,
                tiles_written,
                bytes: len,
                incremental,
                ms: t0.elapsed().as_secs_f64() * 1e3,
            };
            Ok(SaveDone { report, refs })
        }
    }
}

impl Session {
    /// The files this document stands for: where it came from and where it
    /// was saved (for thumbnails of recent files).
    pub fn files(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = self.source.iter().chain(self.save_path.iter()).cloned().collect();
        v.dedup();
        v
    }
}

pub struct Session {
    pub engine: Engine,
    pub name: String,
    pub source: Option<PathBuf>,
    /// The `.mosaix` file this document was last saved to / loaded from.
    save_path: Option<PathBuf>,
    /// Length of that file after our last write.
    file_len: Option<u64>,
    /// Tile content (by serial) already stored in that file.
    saved: HashMap<u64, BlobRef>,
}

/// Loads a document synchronously (headless use and agent commands).
pub fn load(
    gpu: &mosaix_gpu::GpuContext,
    format: wgpu::TextureFormat,
    source: Source,
) -> anyhow::Result<Session> {
    let (tx, rx) = std::sync::mpsc::channel();
    loader::spawn(source, tx);
    let mut session: Option<Session> = None;
    for msg in rx {
        match msg {
            LoaderMsg::Structure {
                doc,
                name,
                source,
                native_len,
                ..
            } => {
                let engine = mosaix_gpu::Engine::new(&gpu.device, &gpu.queue, format, &doc);
                session = Some(Session::new(engine, name, source, native_len));
            }
            LoaderMsg::Tiles { layer, tiles } => {
                let s = session
                    .as_mut()
                    .context(tr("tiles before document structure"))?;
                let failed = s.add_tiles(layer, &tiles);
                if failed > 0 {
                    bail!(
                        "{}",
                        tr!("{n} tiles do not fit into GPU memory", n = failed)
                    );
                }
            }
            LoaderMsg::Mask(m) => {
                let s = session
                    .as_mut()
                    .context(tr("mask before document structure"))?;
                s.add_mask(m)?;
            }
            m @ (LoaderMsg::RawDecoded { .. } | LoaderMsg::RawBlobs(_)) => {
                session.as_mut().context(tr("tiles before document structure"))?.add_raw(m);
            }
            LoaderMsg::Done { .. } => break,
            LoaderMsg::Error(e) => bail!("{}", tr!("Loading failed: {error}", error = e)),
        }
    }
    let mut s = session.context(tr("no document loaded"))?;
    s.engine.clear_history();
    Ok(s)
}

impl Session {
    pub fn new(
        engine: Engine,
        name: String,
        source: Option<PathBuf>,
        native_len: Option<u64>,
    ) -> Self {
        let save_path = native_len.and(source.clone());
        Self {
            engine,
            name,
            source,
            save_path,
            file_len: native_len,
            saved: HashMap::new(),
        }
    }

    /// Uploads loaded tiles; tiles from a native file remember their blob.
    pub fn add_tiles(&mut self, layer: LayerId, tiles: &[TileData]) -> usize {
        let mut failed = 0;
        for (c, px, blob) in tiles {
            match self.engine.upload_tile(layer, *c, px) {
                Some(t) => {
                    if let Some(b) = blob {
                        self.saved.insert(t.serial(), *b);
                    }
                }
                None => failed += 1,
            }
        }
        failed
    }

    /// RAW data from the loader: a decoded file or where a native file stores RAW files.
    pub fn add_raw(&mut self, m: LoaderMsg) {
        match m {
            LoaderMsg::RawDecoded { id, noise, file, cam } => self.engine.prime_raw(id, noise, file, cam),
            LoaderMsg::RawBlobs(b) => self.saved.extend(b),
            _ => {}
        }
    }

    /// Installs a loaded layer mask or selection.
    pub fn add_mask(&mut self, m: MaskLoad) -> anyhow::Result<()> {
        let cpu = CpuMask {
            tiles: m.tiles,
            default: m.default,
        };
        let uploaded = self
            .engine
            .load_mask(m.slot, cpu, m.enabled)
            .context(tr("GPU memory full"))?;
        for (c, t) in uploaded {
            if let Some(b) = m.blobs.get(&c) {
                self.saved.insert(t.serial(), *b);
            }
        }
        Ok(())
    }

    /// Whether the document has its own `.mosaix` file yet.
    /// The file File › Revert reloads: the last saved one, else the source.
    pub fn revert_path(&self) -> Option<PathBuf> {
        self.save_path.clone().or_else(|| self.source.clone()).filter(|p| p.exists())
    }

    pub fn has_save_path(&self) -> bool {
        self.save_path.is_some()
    }

    pub fn default_save_path(&self) -> PathBuf {
        if let Some(p) = &self.save_path {
            return p.clone();
        }
        match &self.source {
            Some(src) => src.with_extension("mosaix"),
            None => PathBuf::from(format!("{}.mosaix", tr("Untitled"))),
        }
    }

    pub fn default_export_path(&self) -> PathBuf {
        let base = self
            .save_path
            .clone()
            .or_else(|| self.source.clone())
            .unwrap_or_else(|| tr("Untitled").into());
        let stem = base
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        base.with_file_name(format!("{stem}-export.png"))
    }

    /// Starts saving to `.mosaix`. Appends only new tiles when the file is
    /// unchanged since our last write; rewrites (compacts) when it carries too
    /// much dead data.
    pub fn start_save(&mut self, path: Option<&Path>) -> anyhow::Result<SaveJob> {
        if self.engine.stroke_active() {
            bail!("{}", tr("cannot save during a brush stroke"));
        }
        let t0 = Instant::now();
        let path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.default_save_path());
        let doc = self.engine.doc().clone();

        // All referenced tiles, by pool.
        let mut layer_tiles: HashMap<u64, Tile> = HashMap::new();
        let mut mask_tiles: HashMap<u64, Tile> = HashMap::new();
        let mut raws: Vec<u64> = Vec::new();
        doc.for_each(&mut |l, _| {
            if let Some(r) = l.raw() {
                raws.push(r.data.id);
            }
            if let Some(t) = l.tiles() {
                for tile in t.values() {
                    layer_tiles
                        .entry(tile.serial())
                        .or_insert_with(|| tile.clone());
                }
            }
            if let Some(m) = &l.mask {
                for t in m.tiles.values() {
                    if let MaskTile::Pixels(tile) = t {
                        mask_tiles
                            .entry(tile.serial())
                            .or_insert_with(|| tile.clone());
                    }
                }
            }
        });
        for sel in doc.selection.iter().chain(doc.channels.iter().map(|c| &c.mask)) {
            for t in sel.tiles.values() {
                if let MaskTile::Pixels(tile) = t {
                    mask_tiles
                        .entry(tile.serial())
                        .or_insert_with(|| tile.clone());
                }
            }
        }

        let same_file = self.save_path.as_deref() == Some(path.as_path())
            && self.file_len.is_some()
            && std::fs::metadata(&path).ok().map(|m| m.len()) == self.file_len;
        let reusable: u64 = layer_tiles
            .keys()
            .chain(mask_tiles.keys())
            .chain(raws.iter())
            .filter_map(|s| self.saved.get(s))
            .map(|b| b.len)
            .sum();
        let compact = same_file && self.file_len.unwrap_or(0) > 2 * reusable + (64 << 20);
        let incremental = same_file && !compact;
        let refs: HashMap<u64, BlobRef> = if incremental {
            layer_tiles
                .keys()
                .chain(mask_tiles.keys())
                .chain(raws.iter())
                .filter_map(|s| self.saved.get(s).map(|b| (*s, *b)))
                .collect()
        } else {
            HashMap::new()
        };
        let missing = |m: HashMap<u64, Tile>| {
            let mut v: Vec<Tile> = m
                .into_values()
                .filter(|t| !refs.contains_key(&t.serial()))
                .collect();
            v.sort_by_key(Tile::serial);
            v
        };
        let layer_missing = missing(layer_tiles);
        let mask_missing = missing(mask_tiles);
        let start = |kind, tiles: &[Tile]| {
            (!tiles.is_empty()).then(|| self.engine.begin_readback(kind, tiles))
        };
        let layer_rb = start(PoolKind::Layer, &layer_missing);
        let mask_rb = start(PoolKind::Mask, &mask_missing);
        Ok(SaveJob {
            path,
            t0,
            incremental,
            file_len: self.file_len,
            doc: doc.map_tiles(&mut |t| t.serial()),
            refs,
            layer_tiles: layer_missing,
            mask_tiles: mask_missing,
            layer_rb,
            mask_rb,
        })
    }

    /// Starts writing a complete copy to `path` (recovery / autosave). Unlike
    /// [`Self::start_save`], the session keeps its own file: do not pass the
    /// result to [`Self::complete_save`].
    pub fn start_save_copy(&mut self, path: &Path) -> anyhow::Result<SaveJob> {
        let (save_path, file_len) = (self.save_path.take(), self.file_len.take());
        let job = self.start_save(Some(path));
        self.save_path = save_path;
        self.file_len = file_len;
        job
    }

    /// Records a finished save (call on the thread that owns the session).
    pub fn complete_save(&mut self, done: SaveDone) -> SaveReport {
        self.saved = done.refs;
        self.file_len = Some(done.report.bytes);
        self.save_path = Some(done.report.path.clone());
        done.report
    }

    /// Saves synchronously (headless, agents, tests).
    pub fn save(&mut self, path: Option<&Path>) -> anyhow::Result<SaveReport> {
        let job = self.start_save(path)?;
        while !job.ready(self.engine.device()) {
            let _ = self
                .engine
                .device()
                .poll(wgpu::PollType::wait_indefinitely());
        }
        let done = job.into_writer()()?;
        Ok(self.complete_save(done))
    }

    /// Flattens and exports to png/jpg/webp/tiff. Returns milliseconds taken.
    pub fn export(&mut self, path: &Path, opts: &ExportOptions) -> anyhow::Result<f64> {
        let job = self.start_export(path, opts);
        while !job.ready(self.engine.device()) {
            let _ = self
                .engine
                .device()
                .poll(wgpu::PollType::wait_indefinitely());
        }
        job.into_writer()()
    }

    /// Starts exporting the flattened image: an asynchronous GPU readback;
    /// conversion and encoding then run via [`ExportJob::into_writer`].
    pub fn start_export(&mut self, path: &Path, opts: &ExportOptions) -> ExportJob {
        let t0 = Instant::now();
        self.engine.flush();
        let size = self.engine.size();
        let tiles = self.engine.composite_tiles();
        let handles: Vec<Tile> = tiles.iter().map(|(_, t)| t.clone()).collect();
        let readback = (!handles.is_empty())
            .then(|| self.engine.begin_readback(PoolKind::Composite, &handles));
        let psd = mosaix_io::psd::is_psd(path).then(|| self.start_psd_layers());
        ExportJob {
            psd,
            path: path.to_path_buf(),
            opts: *opts,
            t0,
            size,
            coords: tiles.into_iter().map(|(c, _)| c).collect(),
            readback,
        }
    }
}

impl Session {
    /// Starts reading back every layer and mask as shown, layer styles
    /// kept live (PSD export).
    fn start_psd_layers(&mut self) -> PsdLayers {
        let doc = self.engine.layers_unstyled();
        let (mut layer_keys, mut layer_tiles, mut mask_keys, mut mask_tiles) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        doc.for_each(&mut |l, _| {
            if let Some(t) = l.tiles() {
                for (c, tile) in t.iter() {
                    layer_keys.push((l.id, *c));
                    layer_tiles.push(tile.clone());
                }
            }
            if let Some(m) = &l.mask {
                for (c, t) in m.tiles.iter() {
                    if let MaskTile::Pixels(tile) = t {
                        mask_keys.push((l.id, *c));
                        mask_tiles.push(tile.clone());
                    }
                }
            }
        });
        // Saved selections: their pixel tiles (solid ones need no readback).
        let (mut channel_keys, mut channel_tiles) = (Vec::new(), Vec::new());
        let channels: Vec<(String, mosaix_core::Mask<()>)> = self
            .engine
            .doc()
            .channels
            .iter()
            .enumerate()
            .map(|(i, ch)| {
                for (c, t) in ch.mask.tiles.iter() {
                    if let MaskTile::Pixels(tile) = t {
                        channel_keys.push((i, *c));
                        channel_tiles.push(tile.clone());
                    }
                }
                (ch.name.clone(), ch.mask.map(&mut |_| ()))
            })
            .collect();
        let guides = self.engine.doc().guides.clone();
        let start = |kind, tiles: &[Tile]| {
            (!tiles.is_empty()).then(|| self.engine.begin_readback(kind, tiles))
        };
        PsdLayers {
            channel_rb: start(PoolKind::Mask, &channel_tiles),
            channel_keys,
            channels,
            channel_data: Vec::new(),
            guides,
            layer_rb: start(PoolKind::Layer, &layer_tiles),
            mask_rb: start(PoolKind::Mask, &mask_tiles),
            doc: doc.map_tiles(&mut |_| ()),
            shapes: psd_shapes(self.engine.doc()),
            texts: psd_texts(self.engine.doc()),
            layer_keys,
            mask_keys,
            layer_data: Vec::new(),
            mask_data: Vec::new(),
        }
    }
}

/// The shape layers of a document as Photoshop shapes (path in document
/// px, fill, stroke scaled with the layer's transform).
fn psd_shapes<T: Clone>(doc: &Document<T>) -> HashMap<LayerId, mosaix_io::psd_shape::PsdShape> {
    use mosaix_core::vector::VectorKind;
    let mut out = HashMap::new();
    doc.for_each(&mut |l, _| {
        let Some(v) = l.vector() else { return };
        let VectorKind::Shape(s) = &v.kind else { return };
        let Some(subpaths) = mosaix_vector::shape_outline(v) else { return };
        let t = v.transform;
        let scale = (t[0] * t[3] - t[1] * t[2]).abs().sqrt() as f32;
        let stroke = s.stroke.clone().map(|mut st| {
            st.width *= scale;
            st
        });
        // Photoshop has no layer transform: turn the gradient with the layer.
        let paint = s.paint.clone().map(|mut p| {
            if let mosaix_core::vector::Paint::Gradient { angle, .. } = &mut p {
                *angle -= (t[1].atan2(t[0]) as f32).to_degrees();
            }
            p
        });
        out.insert(l.id, mosaix_io::psd_shape::PsdShape { subpaths, fill: s.fill, stroke, paint });
    });
    out
}

/// The text layers of a document as Photoshop text (not text on a path).
fn psd_texts<T: Clone>(doc: &Document<T>) -> HashMap<LayerId, (mosaix_io::psd::Text, f32)> {
    use mosaix_core::color::linear_to_srgb;
    use mosaix_core::vector::VectorKind;
    let mut out = HashMap::new();
    doc.for_each(&mut |l, _| {
        let Some(v) = l.vector() else { return };
        let VectorKind::Text(t) = &v.kind else { return };
        if t.path.is_some() {
            return;
        }
        let Some(font) = mosaix_vector::font_postscript_name(t) else { return };
        let [ax, ay] = mosaix_vector::text_anchor(t);
        let m = v.transform;
        let c = t.color;
        let text = mosaix_io::psd::Text {
            text: t.text.clone(),
            font,
            size: t.size,
            color: [linear_to_srgb(c[0]), linear_to_srgb(c[1]), linear_to_srgb(c[2]), c[3]],
            align: t.align,
            line_height: t.line_height,
            tracking: t.letter_spacing,
            faux_bold: false,
            faux_italic: false,
            underline: t.underline,
            strikethrough: t.strikethrough,
            width: t.width,
            transform: [m[0], m[1], m[2], m[3], m[0] * ax + m[2] * ay + m[4], m[1] * ax + m[3] * ay + m[5]],
            features: t.features.clone(),
        };
        out.insert(l.id, (text, mosaix_vector::text_box_size(t).1));
    });
    out
}

/// Layer and mask pixels of a PSD export (readbacks, then their data).
struct PsdLayers {
    doc: Document<()>,
    /// Shape layers' vector data (written as Photoshop shapes).
    shapes: HashMap<LayerId, mosaix_io::psd_shape::PsdShape>,
    /// Text layers as Photoshop text (and box height).
    texts: HashMap<LayerId, (mosaix_io::psd::Text, f32)>,
    layer_keys: Vec<(LayerId, TileCoord)>,
    mask_keys: Vec<(LayerId, TileCoord)>,
    layer_rb: Option<Readback>,
    mask_rb: Option<Readback>,
    layer_data: Vec<Vec<u8>>,
    mask_data: Vec<Vec<u8>>,
    /// Saved selections: names and structure, their pixel tiles' readback.
    channels: Vec<(String, mosaix_core::Mask<()>)>,
    channel_keys: Vec<(usize, TileCoord)>,
    channel_rb: Option<Readback>,
    channel_data: Vec<Vec<u8>>,
    guides: Vec<mosaix_core::Guide>,
}

impl PsdLayers {
    fn take(mut self) -> Self {
        self.layer_data = self.layer_rb.take().map(Readback::take).unwrap_or_default();
        self.mask_data = self.mask_rb.take().map(Readback::take).unwrap_or_default();
        self.channel_data = self.channel_rb.take().map(Readback::take).unwrap_or_default();
        self
    }

    /// Guides and saved selections as whole-canvas planes.
    fn extras(&mut self, (w, h): (u32, u32)) -> mosaix_io::psd::Extras {
        let ts = mosaix_core::TILE_SIZE as usize;
        let (w, h) = (w as usize, h as usize);
        let mut px: HashMap<(usize, TileCoord), Vec<u8>> =
            std::mem::take(&mut self.channel_keys).into_iter().zip(std::mem::take(&mut self.channel_data)).collect();
        let channels = std::mem::take(&mut self.channels)
            .into_iter()
            .enumerate()
            .map(|(i, (name, m))| {
                let mut plane = vec![m.default; w * h];
                for (c, t) in m.tiles.iter() {
                    let (ox, oy) = (c.x as i64 * ts as i64, c.y as i64 * ts as i64);
                    let data = match t {
                        MaskTile::Solid(v) => Some(vec![*v; ts * ts]),
                        MaskTile::Pixels(_) => px.remove(&(i, *c)),
                    };
                    let Some(data) = data else { continue };
                    for y in 0..ts {
                        let dy = oy + y as i64;
                        if dy < 0 || dy >= h as i64 {
                            continue;
                        }
                        for x in 0..ts {
                            let dx = ox + x as i64;
                            if dx >= 0 && dx < w as i64 {
                                plane[dy as usize * w + dx as usize] = data[y * ts + x];
                            }
                        }
                    }
                }
                (name, plane)
            })
            .collect();
        mosaix_io::psd::Extras { guides: std::mem::take(&mut self.guides), channels }
    }

    /// The layer tree to write (adjustment layers are left out).
    fn nodes(self) -> Vec<mosaix_io::psd::WNode> {
        use mosaix_core::doc::Content;
        use mosaix_io::psd::{WLayer, WMask, WNode};
        let mut px: HashMap<(LayerId, TileCoord), Vec<half::f16>> = HashMap::new();
        for (k, raw) in self.layer_keys.into_iter().zip(self.layer_data) {
            px.insert(k, bytemuck::pod_collect_to_vec(&raw));
        }
        let mut mk: HashMap<(LayerId, TileCoord), Vec<u8>> =
            self.mask_keys.into_iter().zip(self.mask_data).collect();
        let ts = mosaix_core::TILE_SIZE as usize;
        let mut skipped = 0;
        let mut shapes = self.shapes;
        let mut texts = self.texts;
        fn build(
            ls: &[mosaix_core::Layer<()>],
            shapes: &mut HashMap<LayerId, mosaix_io::psd_shape::PsdShape>,
            texts: &mut HashMap<LayerId, (mosaix_io::psd::Text, f32)>,
            px: &mut HashMap<(LayerId, TileCoord), Vec<half::f16>>,
            mk: &mut HashMap<(LayerId, TileCoord), Vec<u8>>,
            ts: usize,
            skipped: &mut usize,
        ) -> Vec<WNode> {
            let mut out = Vec::new();
            for l in ls {
                let mask = l.mask.as_ref().map(|m| WMask {
                    default: m.default,
                    enabled: l.mask_enabled,
                    tiles: m
                        .tiles
                        .iter()
                        .map(|(c, t)| match t {
                            MaskTile::Solid(v) => (*c, vec![*v; ts * ts]),
                            MaskTile::Pixels(()) => (
                                *c,
                                mk.remove(&(l.id, *c))
                                    .unwrap_or_else(|| vec![m.default; ts * ts]),
                            ),
                        })
                        .collect(),
                });
                let mut w = WLayer {
                    name: l.name.clone(),
                    opacity: l.opacity,
                    blend: l.blend,
                    visible: l.visible,
                    clip: l.clip,
                    pass_through: l.pass_through,
                    adjustment: None,
                    effects: l.effects.clone(),
                    fill_opacity: l.fill_opacity,
                    label: l.label,
                    shape: shapes.remove(&l.id),
                    text: texts.remove(&l.id),
                    tiles: Vec::new(),
                    mask,
                };
                match &l.content {
                    Content::Pixels(t) => {
                        w.tiles = t
                            .keys()
                            .filter_map(|c| px.remove(&(l.id, *c)).map(|d| (*c, d)))
                            .collect();
                        out.push(WNode::Layer(w));
                    }
                    Content::Group(children) => {
                        let children = build(children, shapes, texts, px, mk, ts, skipped);
                        out.push(WNode::Group { layer: w, children });
                    }
                    Content::Adjustment(a) => {
                        w.adjustment = Some(a.clone());
                        out.push(WNode::Layer(w));
                    }
                    // The render document has no vector layers left (rasterised).
                    Content::Vector(_) | Content::Raw(_) => *skipped += 1,
                }
            }
            out
        }
        let nodes = build(&self.doc.layers, &mut shapes, &mut texts, &mut px, &mut mk, ts, &mut skipped);
        if skipped > 0 {
            tracing::warn!(
                "{}",
                tr!(
                    "PSD: {n} layers were left out (the merged image includes them)",
                    n = skipped
                )
            );
        }
        nodes
    }
}

/// An export whose GPU readback is in flight.
pub struct ExportJob {
    pub path: PathBuf,
    /// Layered export (PSD): every layer's and mask's tiles in flight too.
    psd: Option<PsdLayers>,
    opts: ExportOptions,
    t0: Instant,
    size: (u32, u32),
    coords: Vec<TileCoord>,
    readback: Option<Readback>,
}

impl ExportJob {
    pub fn ready(&self, device: &wgpu::Device) -> bool {
        self.readback.as_ref().is_none_or(|r| r.poll(device))
            && self.psd.as_ref().is_none_or(|p| {
                p.layer_rb.as_ref().is_none_or(|r| r.poll(device))
                    && p.mask_rb.as_ref().is_none_or(|r| r.poll(device))
                    && p.channel_rb.as_ref().is_none_or(|r| r.poll(device))
            })
    }

    /// Conversion, encoding and writing (any thread). Returns the total time in ms.
    pub fn into_writer(self) -> impl FnOnce() -> anyhow::Result<f64> + Send + 'static {
        let data = self.readback.map(Readback::take).unwrap_or_default();
        let psd = self.psd.map(PsdLayers::take);
        let (path, opts, t0, size, coords) =
            (self.path, self.opts, self.t0, self.size, self.coords);
        move || {
            let tiles: Vec<(TileCoord, Vec<half::f16>)> = coords
                .into_iter()
                .zip(
                    data.into_iter()
                        .map(|raw| bytemuck::pod_collect_to_vec(&raw)),
                )
                .collect();
            if let Some(mut p) = psd {
                let extras = p.extras(size);
                let nodes = p.nodes();
                mosaix_io::psd::write(&path, size, &nodes, &tiles, &opts, &extras)
                    .with_context(|| tr!("exporting {path}", path = path.display()))?;
                return Ok(t0.elapsed().as_secs_f64() * 1e3);
            }
            export::write(&path, size, &tiles, &opts)
                .with_context(|| tr!("exporting {path}", path = path.display()))?;
            Ok(t0.elapsed().as_secs_f64() * 1e3)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;
    use half::f16;
    use mosaix_core::{BrushSettings, Document, InputSample, Stroke, TILE_SIZE};
    use mosaix_gpu::{GpuContext, StrokeParams};
    use mosaix_io::native::Reader;

    #[test]
    fn recovery_copy_keeps_the_documents_own_file() {
        let Ok(g) = pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)) else {
            eprintln!("skipping: no GPU");
            return;
        };
        let doc = Document::<()>::new(TILE_SIZE, TILE_SIZE);
        let layer = doc.active;
        let engine = Engine::new(
            &g.device,
            &g.queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &doc,
        );
        let mut s = Session::new(engine, "t".into(), None, None);
        let px = vec![f16::from_f32(0.5); (TILE_SIZE * TILE_SIZE * 4) as usize];
        s.add_tiles(layer, &[(TileCoord::new(0, 0), px, None)]);
        let dir = tempfile::tempdir().unwrap();
        let own = dir.path().join("own.mosaix");
        s.save(Some(&own)).unwrap();
        let copy = dir.path().join("copy.mosaix");
        let job = s.start_save_copy(&copy).unwrap();
        while !job.ready(s.engine.device()) {
            let _ = s.engine.device().poll(wgpu::PollType::wait_indefinitely());
        }
        job.into_writer()().unwrap();
        assert_eq!(s.default_save_path(), own);
        let (_, doc) = Reader::open(&copy).unwrap();
        assert_eq!(doc.find(layer).unwrap().tiles().unwrap().len(), 1);
        // The next real save is still incremental into the own file.
        assert!(s.save(None).unwrap().incremental);
    }

    #[test]
    fn incremental_save_writes_only_changed_tiles() {
        let Ok(g) = pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)) else {
            eprintln!("skipping: no GPU");
            return;
        };
        let doc = Document::<()>::new(TILE_SIZE * 4, TILE_SIZE * 4);
        let layer = doc.active;
        let engine = Engine::new(
            &g.device,
            &g.queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &doc,
        );
        let mut s = Session::new(engine, "t".into(), None, None);
        let px = vec![f16::from_f32(0.5); (TILE_SIZE * TILE_SIZE * 4) as usize];
        let tiles: Vec<_> = (0..16)
            .map(|i| (TileCoord::new(i % 4, i / 4), px.clone(), None))
            .collect();
        assert_eq!(s.add_tiles(layer, &tiles), 0);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.mosaix");
        let r1 = s.save(Some(&path)).unwrap();
        assert!(!r1.incremental);
        assert_eq!(r1.tiles_written, 16);

        // One small stroke inside tile (1,1).
        s.engine.begin_stroke(StrokeParams::default()).unwrap();
        let mut st = Stroke::new(BrushSettings {
            size: 10.0,
            smoothing: 0.0,
            ..Default::default()
        });
        s.engine.stroke_dabs(&st.add(InputSample {
            pos: DVec2::new(380.0, 380.0),
            pressure: 1.0,
        }));
        s.engine.end_stroke("Brush Stroke");

        let r2 = s.save(None).unwrap();
        assert!(r2.incremental);
        assert_eq!(r2.tiles_written, 1, "only the painted tile is appended");

        // Nothing changed → no tiles written (only a fresh manifest).
        let r3 = s.save(None).unwrap();
        assert_eq!(r3.tiles_written, 0);
        assert!(r3.bytes - r2.bytes < 4096);

        // The file reloads with the new tile content.
        let (reader, loaded) = Reader::open(&path).unwrap();
        let blob = loaded.find(layer).unwrap().tiles().unwrap()[&TileCoord::new(1, 1)];
        let t = mosaix_io::native::decompress_tile(&reader.read_blob(blob).unwrap()).unwrap();
        let i = ((380 - 256) * TILE_SIZE as usize + (380 - 256)) * 4;
        assert!(t[i].to_f32() < 0.4, "painted dark: {}", t[i].to_f32());
        assert_eq!(reader.len, r3.bytes);
    }

    #[test]
    fn background_save_snapshot_while_editing() {
        let Ok(g) = pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)) else {
            eprintln!("skipping: no GPU");
            return;
        };
        let doc = Document::<()>::new(TILE_SIZE * 2, TILE_SIZE * 2);
        let engine = Engine::new(
            &g.device,
            &g.queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &doc,
        );
        let mut s = Session::new(engine, "t".into(), None, None);
        s.engine
            .fill(
                mosaix_gpu::StrokeTarget::Pixels,
                [1.0, 0.0, 0.0, 1.0],
                1.0,
                "Fill",
            )
            .unwrap();
        s.engine
            .select_shape(
                &mosaix_core::mask::Shape::Rect {
                    x: 10.0,
                    y: 10.0,
                    w: 100.0,
                    h: 50.0,
                },
                mosaix_core::mask::CombineOp::Replace,
                "Selection",
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bg.mosaix");

        let job = s.start_save(Some(&path)).unwrap();
        // Keep editing while the save is in flight: the snapshot must not change.
        s.engine.clear_selected("Delete").unwrap();
        s.engine.deselect("Deselect").unwrap();
        while !job.ready(s.engine.device()) {
            std::thread::yield_now();
        }
        let done = std::thread::spawn(job.into_writer())
            .join()
            .unwrap()
            .unwrap();
        let r = s.complete_save(done);
        assert_eq!(
            r.tiles_written,
            4 + 1,
            "4 layer tiles + 1 selection edge tile"
        );

        let (reader, loaded) = Reader::open(&path).unwrap();
        assert!(loaded.selection.is_some(), "snapshot had a selection");
        let layer = loaded.layers[0].tiles().unwrap().clone();
        let t = mosaix_io::native::decompress_tile(
            &reader.read_blob(layer[&TileCoord::new(0, 0)]).unwrap(),
        )
        .unwrap();
        let i = (20 * TILE_SIZE as usize + 20) * 4 + 3;
        assert_eq!(
            t[i].to_f32(),
            1.0,
            "pixel cleared after the snapshot is still there in the file"
        );
    }
}
