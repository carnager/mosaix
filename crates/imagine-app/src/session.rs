//! An open document: the engine plus its file state (incremental saving,
//! export). Used by the window and by the headless CLI.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, bail};
use imagine_core::{LayerId, TileCoord};
use imagine_gpu::{Engine, Tile};
use imagine_io::native::{Writer, compress_tile, live_bytes};
use imagine_io::{BlobRef, export};
use rayon::prelude::*;

use crate::loader::TileData;

const READBACK_BATCH: usize = 256;

pub struct SaveReport {
    pub path: PathBuf,
    pub tiles_written: usize,
    pub bytes: u64,
    pub incremental: bool,
    pub ms: f64,
}

pub struct Session {
    pub engine: Engine,
    pub name: String,
    pub source: Option<PathBuf>,
    /// The `.imagine` file this document was last saved to / loaded from.
    save_path: Option<PathBuf>,
    /// Length of that file after our last write.
    file_len: Option<u64>,
    /// Tile content (by serial) already stored in that file.
    saved: HashMap<u64, BlobRef>,
}

impl Session {
    pub fn new(engine: Engine, name: String, source: Option<PathBuf>, native_len: Option<u64>) -> Self {
        let save_path = native_len.and(source.clone());
        Self { engine, name, source, save_path, file_len: native_len, saved: HashMap::new() }
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

    pub fn default_save_path(&self) -> PathBuf {
        if let Some(p) = &self.save_path {
            return p.clone();
        }
        match &self.source {
            Some(src) => src.with_extension("imagine"),
            None => PathBuf::from("Unbenannt.imagine"),
        }
    }

    pub fn default_export_path(&self) -> PathBuf {
        let base = self.save_path.clone().or_else(|| self.source.clone()).unwrap_or_else(|| "Unbenannt".into());
        let stem = base.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        base.with_file_name(format!("{stem}-export.png"))
    }

    /// Saves to `.imagine`. Appends only new tiles when the file is unchanged
    /// since our last write; rewrites (compacts) when it carries too much dead data.
    pub fn save(&mut self, path: Option<&Path>) -> anyhow::Result<SaveReport> {
        if self.engine.stroke_active() {
            bail!("während eines Pinselstrichs kann nicht gespeichert werden");
        }
        let t0 = Instant::now();
        let path = path.map(Path::to_path_buf).unwrap_or_else(|| self.default_save_path());
        let doc = self.engine.doc().clone();

        let mut tiles: HashMap<u64, Tile> = HashMap::new();
        doc.for_each(&mut |l, _| {
            if let Some(t) = l.tiles() {
                for tile in t.values() {
                    tiles.entry(tile.serial()).or_insert_with(|| tile.clone());
                }
            }
        });

        let same_file = self.save_path.as_deref() == Some(path.as_path())
            && self.file_len.is_some()
            && std::fs::metadata(&path).ok().map(|m| m.len()) == self.file_len;
        let reusable: u64 = tiles.keys().filter_map(|s| self.saved.get(s)).map(|b| b.len).sum();
        let compact = same_file && self.file_len.unwrap_or(0) > 2 * reusable + (64 << 20);
        let incremental = same_file && !compact;

        let (mut writer, mut refs) = if incremental {
            let refs: HashMap<u64, BlobRef> =
                tiles.keys().filter_map(|s| self.saved.get(s).map(|b| (*s, *b))).collect();
            (Writer::append(&path, self.file_len.unwrap())?, refs)
        } else {
            (Writer::create(&path)?, HashMap::new())
        };

        let mut missing: Vec<Tile> = tiles.values().filter(|t| !refs.contains_key(&t.serial())).cloned().collect();
        missing.sort_by_key(Tile::serial);
        let tiles_written = missing.len();
        for batch in missing.chunks(READBACK_BATCH) {
            let pixels = self.engine.read_layer_tiles(batch);
            let blobs: Vec<Vec<u8>> = pixels.par_iter().map(|p| compress_tile(p)).collect();
            for (t, blob) in batch.iter().zip(blobs) {
                refs.insert(t.serial(), writer.write_blob(&blob)?);
            }
        }
        let doc_blobs = doc.map_tiles(&mut |t| refs[&t.serial()]);
        let len = writer.finish(&doc_blobs).with_context(|| format!("{} speichern", path.display()))?;
        debug_assert!(live_bytes(&doc_blobs) <= len);

        self.saved = refs;
        self.file_len = Some(len);
        self.save_path = Some(path.clone());
        Ok(SaveReport { path, tiles_written, bytes: len, incremental, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }

    /// Flattens and exports to png/jpg/webp/tiff. Returns milliseconds taken.
    pub fn export(&mut self, path: &Path) -> anyhow::Result<f64> {
        let t0 = Instant::now();
        self.engine.flush();
        let (w, h) = self.engine.size();
        let mut img = image::RgbaImage::new(w, h);
        let coords = self.engine.composite_coords();
        for batch in coords.chunks(READBACK_BATCH) {
            let px = self.engine.read_composite(batch);
            let tiles: Vec<(TileCoord, _)> = batch.iter().copied().zip(px).collect();
            export::blit(&mut img, &tiles);
        }
        export::save(&img, path).with_context(|| format!("{} exportieren", path.display()))?;
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;
    use half::f16;
    use imagine_core::{BrushSettings, Document, InputSample, Stroke, TILE_SIZE};
    use imagine_gpu::{GpuContext, StrokeParams};
    use imagine_io::native::Reader;

    #[test]
    fn incremental_save_writes_only_changed_tiles() {
        let Ok(g) = pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)) else {
            eprintln!("skipping: no GPU");
            return;
        };
        let doc = Document::<()>::new(TILE_SIZE * 4, TILE_SIZE * 4);
        let layer = doc.active;
        let engine = Engine::new(&g.device, &g.queue, wgpu::TextureFormat::Rgba8UnormSrgb, &doc);
        let mut s = Session::new(engine, "t".into(), None, None);
        let px = vec![f16::from_f32(0.5); (TILE_SIZE * TILE_SIZE * 4) as usize];
        let tiles: Vec<_> =
            (0..16).map(|i| (TileCoord::new(i % 4, i / 4), px.clone(), None)).collect();
        assert_eq!(s.add_tiles(layer, &tiles), 0);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.imagine");
        let r1 = s.save(Some(&path)).unwrap();
        assert!(!r1.incremental);
        assert_eq!(r1.tiles_written, 16);

        // One small stroke inside tile (1,1).
        s.engine.begin_stroke(StrokeParams { opacity: 1.0, erase: false }).unwrap();
        let mut st = Stroke::new(BrushSettings { size: 10.0, smoothing: 0.0, ..Default::default() });
        s.engine.stroke_dabs(&st.add(InputSample { pos: DVec2::new(380.0, 380.0), pressure: 1.0 }));
        s.engine.end_stroke("Pinselstrich");

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
        let t = imagine_io::native::decompress_tile(&reader.read_blob(blob).unwrap()).unwrap();
        let i = ((380 - 256) * TILE_SIZE as usize + (380 - 256)) * 4;
        assert!(t[i].to_f32() < 0.4, "painted dark: {}", t[i].to_f32());
        assert_eq!(reader.len, r3.bytes);
    }
}
