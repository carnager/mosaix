//! The native `.imagine` format.
//!
//! ```text
//! "IMAGINE\x01"                       magic
//! blob*                               zstd-compressed tiles (RGBA f16 LE, premultiplied linear)
//! manifest                            zstd-compressed JSON (layer tree, tile → blob refs)
//! footer: manifest_offset u64 LE, manifest_len u64 LE, "IMGNFOOT"
//! ```
//!
//! Saving is append-only: an incremental save writes only tiles that are not
//! in the file yet, then a new manifest and footer. Readers use the last
//! footer; if the tail is damaged (crash during an append) they fall back to
//! the previous valid footer. Dead space is reclaimed by a full rewrite.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, bail, ensure};
use half::f16;
use imagine_core::doc::{Content, Layer, TileMap};
use imagine_core::{BlendMode, BlendSpace, Document, Mask, MaskTile, TILE_SIZE, TileCoord};
use serde::{Deserialize, Serialize};

#[cfg(target_endian = "big")]
compile_error!("the .imagine tile encoding assumes a little-endian host");

pub const MAGIC: &[u8; 8] = b"IMAGINE\x01";
const FOOTER_MAGIC: &[u8; 8] = b"IMGNFOOT";
const FOOTER_LEN: u64 = 24;
/// 1: layers. 2: + layer masks and selection.
const FORMAT_VERSION: u32 = 7;
const TILE_VALUES: usize = (TILE_SIZE * TILE_SIZE * 4) as usize;
const MASK_VALUES: usize = (TILE_SIZE * TILE_SIZE) as usize;

/// Location of a blob inside the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlobRef {
    pub offset: u64,
    pub len: u64,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: u32,
    width: u32,
    height: u32,
    blend_space: String,
    active: u32,
    layers: Vec<LayerEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    selection: Option<MaskEntry>,
}

fn yes() -> bool {
    true
}

#[derive(Serialize, Deserialize)]
struct MaskEntry {
    default: u8,
    #[serde(default = "yes")]
    enabled: bool,
    tiles: Vec<MaskTileEntry>,
}

#[derive(Serialize, Deserialize)]
struct MaskTileEntry {
    x: i32,
    y: i32,
    /// Uniform tile (no blob).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    solid: Option<u8>,
    #[serde(default)]
    offset: u64,
    #[serde(default)]
    len: u64,
}

#[derive(Serialize, Deserialize)]
struct LayerEntry {
    id: u32,
    name: String,
    opacity: f32,
    blend: String,
    visible: bool,
    #[serde(flatten)]
    content: ContentEntry,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mask: Option<MaskEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    filters: Vec<imagine_core::SmartFilter>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    effects: Vec<imagine_core::style::Effect>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    clip: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pass_through: bool,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    fill_opacity: f32,
}

fn one() -> f32 {
    1.0
}

fn is_one(v: &f32) -> bool {
    *v == 1.0
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ContentEntry {
    Pixels {
        tiles: Vec<TileEntry>,
    },
    Group {
        children: Vec<LayerEntry>,
    },
    /// Since format 3.
    Adjustment {
        adjustment: imagine_core::Adjustment,
    },
    Vector {
        vector: imagine_core::VectorLayer,
    },
}

#[derive(Serialize, Deserialize)]
struct TileEntry {
    x: i32,
    y: i32,
    offset: u64,
    len: u64,
}

fn mask_to_entry(m: &Mask<BlobRef>, enabled: bool) -> MaskEntry {
    let mut tiles: Vec<MaskTileEntry> = m
        .tiles
        .iter()
        .map(|(c, t)| match t {
            MaskTile::Solid(v) => MaskTileEntry {
                x: c.x,
                y: c.y,
                solid: Some(*v),
                offset: 0,
                len: 0,
            },
            MaskTile::Pixels(b) => MaskTileEntry {
                x: c.x,
                y: c.y,
                solid: None,
                offset: b.offset,
                len: b.len,
            },
        })
        .collect();
    tiles.sort_by_key(|t| (t.y, t.x));
    MaskEntry {
        default: m.default,
        enabled,
        tiles,
    }
}

fn entry_to_mask(e: MaskEntry, file_len: u64) -> anyhow::Result<(Mask<BlobRef>, bool)> {
    let mut tiles = TileMap::new();
    for t in e.tiles {
        let v = match t.solid {
            Some(v) => MaskTile::Solid(v),
            None => {
                ensure!(
                    t.offset + t.len <= file_len,
                    "{}",
                    imagine_core::tr("mask tile outside the file")
                );
                MaskTile::Pixels(BlobRef {
                    offset: t.offset,
                    len: t.len,
                })
            }
        };
        tiles.insert(TileCoord::new(t.x, t.y), v);
    }
    Ok((
        Mask {
            tiles: Arc::new(tiles),
            default: e.default,
        },
        e.enabled,
    ))
}

fn to_manifest(doc: &Document<BlobRef>) -> Manifest {
    fn layers(ls: &[Layer<BlobRef>]) -> Vec<LayerEntry> {
        ls.iter()
            .map(|l| LayerEntry {
                id: l.id,
                name: l.name.clone(),
                opacity: l.opacity,
                blend: l.blend.key().into(),
                visible: l.visible,
                content: match &l.content {
                    Content::Pixels(t) => {
                        let mut tiles: Vec<TileEntry> = t
                            .iter()
                            .map(|(c, b)| TileEntry {
                                x: c.x,
                                y: c.y,
                                offset: b.offset,
                                len: b.len,
                            })
                            .collect();
                        tiles.sort_by_key(|t| (t.y, t.x));
                        ContentEntry::Pixels { tiles }
                    }
                    Content::Group(c) => ContentEntry::Group {
                        children: layers(c),
                    },
                    Content::Adjustment(a) => ContentEntry::Adjustment {
                        adjustment: a.clone(),
                    },
                    Content::Vector(v) => ContentEntry::Vector { vector: v.clone() },
                },
                mask: l.mask.as_ref().map(|m| mask_to_entry(m, l.mask_enabled)),
                filters: l.filters.clone(),
                effects: l.effects.clone(),
                clip: l.clip,
                pass_through: l.pass_through,
                fill_opacity: l.fill_opacity,
            })
            .collect()
    }
    Manifest {
        format: FORMAT_VERSION,
        width: doc.width,
        height: doc.height,
        blend_space: doc.blend_space.key().into(),
        active: doc.active,
        layers: layers(&doc.layers),
        selection: doc.selection.as_ref().map(|m| mask_to_entry(m, true)),
    }
}

fn from_manifest(m: Manifest, file_len: u64) -> anyhow::Result<Document<BlobRef>> {
    ensure!(
        m.format <= FORMAT_VERSION,
        "{}",
        imagine_core::tr!(
            "file version {version} is newer than this program",
            version = m.format
        )
    );
    fn layers(es: Vec<LayerEntry>, file_len: u64) -> anyhow::Result<Vec<Layer<BlobRef>>> {
        es.into_iter()
            .map(|e| {
                let content = match e.content {
                    ContentEntry::Pixels { tiles } => {
                        let mut map = TileMap::new();
                        for t in tiles {
                            ensure!(
                                t.offset + t.len <= file_len,
                                "{}",
                                imagine_core::tr("tile outside the file")
                            );
                            map.insert(
                                TileCoord::new(t.x, t.y),
                                BlobRef {
                                    offset: t.offset,
                                    len: t.len,
                                },
                            );
                        }
                        Content::Pixels(Arc::new(map))
                    }
                    ContentEntry::Group { children } => Content::Group(layers(children, file_len)?),
                    ContentEntry::Adjustment { adjustment } => Content::Adjustment(adjustment),
                    ContentEntry::Vector { vector } => Content::Vector(vector),
                };
                let (mask, mask_enabled) = match e.mask {
                    Some(m) => {
                        let (m, en) = entry_to_mask(m, file_len)?;
                        (Some(m), en)
                    }
                    None => (None, true),
                };
                Ok(Layer {
                    id: e.id,
                    name: e.name,
                    opacity: e.opacity.clamp(0.0, 1.0),
                    blend: BlendMode::from_key(&e.blend).unwrap_or_default(),
                    visible: e.visible,
                    content,
                    mask,
                    mask_enabled,
                    filters: e.filters,
                    effects: e.effects,
                    clip: e.clip,
                    pass_through: e.pass_through,
                    fill_opacity: e.fill_opacity.clamp(0.0, 1.0),
                })
            })
            .collect()
    }
    let mut doc = Document::empty(m.width, m.height);
    doc.blend_space = BlendSpace::from_key(&m.blend_space).unwrap_or_default();
    doc.layers = layers(m.layers, file_len)?;
    doc.selection = m
        .selection
        .map(|e| entry_to_mask(e, file_len).map(|m| m.0))
        .transpose()?;
    ensure!(
        !doc.layers.is_empty(),
        "{}",
        imagine_core::tr("document without layers")
    );
    doc.active = if doc.find(m.active).is_some() {
        m.active
    } else {
        doc.display_order()[0]
    };
    doc.fix_next_id();
    Ok(doc)
}

pub fn compress_tile(px: &[f16]) -> Vec<u8> {
    zstd::bulk::compress(bytemuck::cast_slice(px), 1).expect("zstd compress")
}

pub fn decompress_tile(data: &[u8]) -> anyhow::Result<Vec<f16>> {
    let raw = zstd::bulk::decompress(data, TILE_VALUES * 2)?;
    ensure!(
        raw.len() == TILE_VALUES * 2,
        "{}",
        imagine_core::tr("tile has the wrong size")
    );
    let mut out = vec![f16::ZERO; TILE_VALUES];
    bytemuck::cast_slice_mut::<f16, u8>(&mut out).copy_from_slice(&raw);
    Ok(out)
}

pub fn compress_mask(values: &[u8]) -> Vec<u8> {
    zstd::bulk::compress(values, 1).expect("zstd compress")
}

pub fn decompress_mask(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let raw = zstd::bulk::decompress(data, MASK_VALUES)?;
    ensure!(
        raw.len() == MASK_VALUES,
        "{}",
        imagine_core::tr("mask tile has the wrong size")
    );
    Ok(raw)
}

/// Bytes referenced by the document (tile and mask blobs).
pub fn live_bytes(doc: &Document<BlobRef>) -> u64 {
    let mut seen = std::collections::HashSet::new();
    let mut n = 0;
    doc.for_each_tile(&mut |b| {
        if seen.insert(b.offset) {
            n += b.len;
        }
    });
    n
}

pub struct Writer {
    file: File,
    pos: u64,
    /// For full writes: temp file to rename over `path` on finish.
    rename: Option<(PathBuf, PathBuf)>,
}

impl Writer {
    /// Starts a full write (into a temp file, renamed atomically on finish).
    pub fn create(path: &Path) -> anyhow::Result<Self> {
        let tmp = path.with_extension("imagine.tmp");
        let mut file = File::create(&tmp)
            .with_context(|| imagine_core::tr!("creating {path}", path = tmp.display()))?;
        file.write_all(MAGIC)?;
        Ok(Self {
            file,
            pos: MAGIC.len() as u64,
            rename: Some((tmp, path.to_path_buf())),
        })
    }

    /// Continues an existing file, which must still be exactly as we left it.
    pub fn append(path: &Path, expected_len: u64) -> anyhow::Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        ensure!(
            len == expected_len,
            "{}",
            imagine_core::tr("file was changed in the meantime")
        );
        let mut magic = [0u8; 8];
        file.read_exact_at(&mut magic, 0)?;
        ensure!(
            &magic == MAGIC,
            "{}",
            imagine_core::tr("not an imagine file")
        );
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            file,
            pos: len,
            rename: None,
        })
    }

    pub fn write_blob(&mut self, data: &[u8]) -> anyhow::Result<BlobRef> {
        self.file.write_all(data)?;
        let r = BlobRef {
            offset: self.pos,
            len: data.len() as u64,
        };
        self.pos += data.len() as u64;
        Ok(r)
    }

    /// Writes manifest and footer, syncs, and (for full writes) renames into place.
    /// Returns the final file length.
    pub fn finish(mut self, doc: &Document<BlobRef>) -> anyhow::Result<u64> {
        let json = serde_json::to_vec(&to_manifest(doc))?;
        let manifest = zstd::bulk::compress(&json, 3)?;
        let m = self.write_blob(&manifest)?;
        self.file.write_all(&m.offset.to_le_bytes())?;
        self.file.write_all(&m.len.to_le_bytes())?;
        self.file.write_all(FOOTER_MAGIC)?;
        self.pos += FOOTER_LEN;
        self.file.sync_all()?;
        if let Some((tmp, path)) = self.rename.take() {
            std::fs::rename(&tmp, &path)
                .with_context(|| imagine_core::tr!("replacing {path}", path = path.display()))?;
        }
        Ok(self.pos)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // An unfinished full write leaves no temp file behind.
        if let Some((tmp, _)) = &self.rename {
            let _ = std::fs::remove_file(tmp);
        }
    }
}

pub struct Reader {
    file: File,
    pub len: u64,
}

impl Reader {
    pub fn open(path: &Path) -> anyhow::Result<(Self, Document<BlobRef>)> {
        let file = File::open(path)
            .with_context(|| imagine_core::tr!("opening {path}", path = path.display()))?;
        let len = file.metadata()?.len();
        let mut magic = [0u8; 8];
        file.read_exact_at(&mut magic, 0)
            .context(imagine_core::tr("file too short"))?;
        ensure!(
            &magic == MAGIC,
            "{}",
            imagine_core::tr("not an imagine file")
        );
        let r = Self { file, len };

        // Newest footer first; fall back to older ones if the tail is damaged.
        let mut end = len;
        let mut last_err = None;
        while let Some(footer_at) = r.find_footer_before(end)? {
            match r.read_manifest(footer_at) {
                Ok(doc) => {
                    if end != len {
                        eprintln!(
                            "{}",
                            imagine_core::tr(
                                "Warning: end of file damaged, older version restored"
                            )
                        );
                    }
                    return Ok((
                        Self {
                            len: footer_at + FOOTER_LEN,
                            ..r
                        },
                        doc,
                    ));
                }
                Err(e) => last_err = Some(e),
            }
            end = footer_at + FOOTER_LEN - 1;
        }
        Err(last_err.unwrap_or_else(|| {
            anyhow::anyhow!("{}", imagine_core::tr("no valid table of contents found"))
        }))
    }

    /// Offset of the last footer that ends at or before `end`.
    fn find_footer_before(&self, end: u64) -> anyhow::Result<Option<u64>> {
        if end >= FOOTER_LEN + MAGIC.len() as u64 {
            let at = end - FOOTER_LEN;
            let mut m = [0u8; 8];
            self.file.read_exact_at(&mut m, at + 16)?;
            if &m == FOOTER_MAGIC {
                return Ok(Some(at));
            }
        }
        // Slow path: scan backwards in 1 MiB windows.
        let mut hi = end;
        let window = 1 << 20;
        while hi > MAGIC.len() as u64 + FOOTER_LEN {
            let lo = hi.saturating_sub(window).max(MAGIC.len() as u64);
            let mut buf = vec![0u8; (hi - lo) as usize];
            self.file.read_exact_at(&mut buf, lo)?;
            if let Some(i) = buf.windows(8).rposition(|w| w == FOOTER_MAGIC) {
                let magic_at = lo + i as u64;
                if magic_at >= 16 + MAGIC.len() as u64 && magic_at + 8 <= end {
                    return Ok(Some(magic_at - 16));
                }
            }
            if lo == MAGIC.len() as u64 {
                break;
            }
            hi = lo + 7; // overlap so a magic across the boundary is found
        }
        Ok(None)
    }

    fn read_manifest(&self, footer_at: u64) -> anyhow::Result<Document<BlobRef>> {
        let mut f = [0u8; 16];
        self.file.read_exact_at(&mut f, footer_at)?;
        let offset = u64::from_le_bytes(f[..8].try_into().unwrap());
        let len = u64::from_le_bytes(f[8..].try_into().unwrap());
        if offset + len > footer_at || len > 1 << 30 {
            bail!("{}", imagine_core::tr("table of contents invalid"));
        }
        let data = self.read_blob(BlobRef { offset, len })?;
        let json = zstd::stream::decode_all(&data[..])?;
        let manifest: Manifest = serde_json::from_slice(&json)
            .context(imagine_core::tr("table of contents unreadable"))?;
        from_manifest(manifest, footer_at)
    }

    /// Reads one blob (thread-safe, positional read).
    pub fn read_blob(&self, b: BlobRef) -> anyhow::Result<Vec<u8>> {
        let mut buf = vec![0u8; b.len as usize];
        self.file.read_exact_at(&mut buf, b.offset)?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(v: f32) -> Vec<f16> {
        vec![f16::from_f32(v); TILE_VALUES]
    }

    fn sample_doc(w: &mut Writer) -> Document<BlobRef> {
        let mut d = Document::<BlobRef>::new(600, 400);
        let bg = d.active;
        let b = w.write_blob(&compress_tile(&tile(0.25))).unwrap();
        d.tiles_mut(bg).unwrap().insert(TileCoord::new(1, 0), b);
        let top = d.add_pixel_layer();
        d.find_mut(top).unwrap().blend = BlendMode::Multiply;
        d.find_mut(top).unwrap().opacity = 0.5;
        d.group_active();
        d.blend_space = BlendSpace::Linear;
        let mb = w
            .write_blob(&compress_mask(&vec![77u8; MASK_VALUES]))
            .unwrap();
        let mut m = Mask::filled(255);
        Arc::make_mut(&mut m.tiles).insert(TileCoord::new(0, 0), MaskTile::Pixels(mb));
        Arc::make_mut(&mut m.tiles).insert(TileCoord::new(1, 0), MaskTile::Solid(0));
        d.find_mut(bg).unwrap().mask = Some(m);
        d.find_mut(bg).unwrap().mask_enabled = false;
        d.find_mut(bg).unwrap().filters = vec![
            imagine_core::SmartFilter::new(imagine_core::Filter::GaussianBlur { radius: 3.5 }),
            imagine_core::SmartFilter {
                filter: imagine_core::Filter::HighPass { radius: 9.0 },
                enabled: false,
            },
        ];
        d.selection = Some(Mask::filled(255));
        d
    }

    #[test]
    fn roundtrip_and_incremental_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.imagine");
        let mut w = Writer::create(&path).unwrap();
        let mut doc = sample_doc(&mut w);
        let len = w.finish(&doc).unwrap();

        let (r, loaded) = Reader::open(&path).unwrap();
        assert_eq!(r.len, len);
        assert_eq!(loaded.describe(), doc.describe());
        assert_eq!(loaded.blend_space, BlendSpace::Linear);
        let bg = loaded.layers[0].id;
        let blob = loaded.find(bg).unwrap().tiles().unwrap()[&TileCoord::new(1, 0)];
        assert_eq!(
            decompress_tile(&r.read_blob(blob).unwrap()).unwrap()[5].to_f32(),
            0.25
        );
        let l = loaded.find(bg).unwrap();
        assert!(!l.mask_enabled);
        assert_eq!(l.filters, doc.find(bg).unwrap().filters);
        let m = l.mask.as_ref().unwrap();
        assert_eq!(m.default, 255);
        assert!(matches!(m.tiles[&TileCoord::new(1, 0)], MaskTile::Solid(0)));
        let MaskTile::Pixels(mb) = m.tiles[&TileCoord::new(0, 0)] else {
            panic!("pixel mask tile")
        };
        assert_eq!(decompress_mask(&r.read_blob(mb).unwrap()).unwrap()[100], 77);
        assert_eq!(loaded.selection.as_ref().unwrap().default, 255);

        // Append a tile without rewriting the existing one.
        let mut w = Writer::append(&path, len).unwrap();
        let b2 = w.write_blob(&compress_tile(&tile(0.75))).unwrap();
        doc.tiles_mut(bg).unwrap().insert(TileCoord::new(0, 0), b2);
        let len2 = w.finish(&doc).unwrap();
        assert!(len2 > len);
        let (r, loaded) = Reader::open(&path).unwrap();
        let tiles = loaded.find(bg).unwrap().tiles().unwrap().clone();
        assert_eq!(
            tiles[&TileCoord::new(1, 0)],
            blob,
            "old blob reused in place"
        );
        assert_eq!(
            decompress_tile(&r.read_blob(tiles[&TileCoord::new(0, 0)]).unwrap()).unwrap()[0]
                .to_f32(),
            0.75
        );

        // Stale length → append refused.
        assert!(Writer::append(&path, len).is_err());
    }

    #[test]
    fn damaged_tail_falls_back_to_previous_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.imagine");
        let mut w = Writer::create(&path).unwrap();
        let doc = sample_doc(&mut w);
        let len = w.finish(&doc).unwrap();
        // Simulate a crash in the middle of an append.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&compress_tile(&tile(1.0))).unwrap();
        f.write_all(b"half a footer").unwrap();
        drop(f);
        let (r, loaded) = Reader::open(&path).unwrap();
        assert_eq!(r.len, len);
        assert_eq!(loaded.describe(), doc.describe());
    }

    #[test]
    fn rejects_garbage_and_cleans_up_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.imagine");
        std::fs::write(&path, b"not an image at all").unwrap();
        assert!(Reader::open(&path).is_err());
        {
            let _w = Writer::create(&dir.path().join("d.imagine")).unwrap();
        }
        assert!(!dir.path().join("d.imagine.tmp").exists());
    }

    #[test]
    fn live_bytes_counts_shared_blobs_once() {
        let mut d = Document::<BlobRef>::new(300, 300);
        let b = BlobRef {
            offset: 8,
            len: 100,
        };
        d.tiles_mut(d.active)
            .unwrap()
            .insert(TileCoord::new(0, 0), b);
        d.duplicate_active();
        assert_eq!(live_bytes(&d), 100);
    }
}
