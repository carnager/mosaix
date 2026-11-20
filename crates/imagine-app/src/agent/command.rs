//! The agent command set: one JSON-serialisable command per operation,
//! executed against a [`Session`]. Shared by the MCP server, `imagine run`
//! scripts and the live socket of a running window.
//!
//! Conventions (also stated in the tool descriptions):
//! - coordinates are document pixels, origin top-left;
//! - colours are sRGB: `"#rrggbb"`, `"#rrggbbaa"` or `[r, g, b(, a)]` in 0–255;
//! - `layer` defaults to the active layer; layer ids are stable.

use std::path::PathBuf;

use anyhow::{Context, bail, ensure};
use glam::DVec2;
use half::f16;
use image::RgbaImage;
use imagine_core::color::{Pixels, extract_region, srgb_to_linear, linear_to_srgb};
use imagine_core::doc::Content;
use imagine_core::tile::tiles_in_rect;
use imagine_core::{BlendMode, BlendSpace, BrushSettings, Document, InputSample, LayerId, Stroke, TILE_SIZE, TileCoord};
use imagine_gpu::{GpuContext, StrokeParams, Tile};
use imagine_io::export::blit_region;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::loader::Source;
use crate::session::{self, Session};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ColorSpec {
    Hex(String),
    Rgb(Vec<f32>),
}

impl ColorSpec {
    /// Linear RGB + straight alpha.
    fn linear(&self) -> anyhow::Result<([f32; 3], f32)> {
        let v: Vec<f32> = match self {
            ColorSpec::Hex(h) => {
                let h = h.trim_start_matches('#');
                ensure!(h.len() == 6 || h.len() == 8, "colour must be #rrggbb or #rrggbbaa, got {h:?}");
                (0..h.len() / 2)
                    .map(|i| u8::from_str_radix(&h[i * 2..i * 2 + 2], 16).map(f32::from))
                    .collect::<Result<_, _>>()
                    .context("invalid hex colour")?
            }
            ColorSpec::Rgb(v) => v.clone(),
        };
        ensure!(v.len() == 3 || v.len() == 4, "colour needs 3 or 4 components");
        ensure!(v.iter().all(|c| (0.0..=255.0).contains(c)), "colour components must be 0–255");
        let rgb = [0, 1, 2].map(|i| srgb_to_linear(v[i] / 255.0));
        Ok((rgb, v.get(3).map_or(1.0, |a| a / 255.0)))
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    DocumentNew { width: u32, height: u32 },
    DocumentOpen { path: PathBuf },
    DocumentInfo,
    LayerAdd { name: Option<String> },
    LayerDuplicate { layer: Option<LayerId> },
    LayerDelete { layer: Option<LayerId> },
    LayerSet {
        layer: Option<LayerId>,
        name: Option<String>,
        opacity: Option<f32>,
        blend: Option<String>,
        visible: Option<bool>,
    },
    LayerSelect { layer: LayerId },
    LayerMove { layer: Option<LayerId>, steps: i32 },
    LayerGroup { layer: Option<LayerId> },
    LayerUngroup { layer: Option<LayerId> },
    SetBlendSpace { space: String },
    PaintStroke {
        layer: Option<LayerId>,
        points: Vec<Vec<f64>>,
        color: Option<ColorSpec>,
        size: f32,
        hardness: Option<f32>,
        opacity: Option<f32>,
        flow: Option<f32>,
        spacing: Option<f32>,
        #[serde(default)]
        eraser: bool,
    },
    FillRect {
        layer: Option<LayerId>,
        x: i64,
        y: i64,
        width: i64,
        height: i64,
        color: ColorSpec,
        opacity: Option<f32>,
    },
    ClearRect { layer: Option<LayerId>, x: i64, y: i64, width: i64, height: i64 },
    PlaceImage { path: PathBuf, x: Option<i64>, y: Option<i64>, name: Option<String> },
    SampleColor { x: f64, y: f64 },
    RenderPreview {
        max_size: Option<u32>,
        x: Option<i64>,
        y: Option<i64>,
        width: Option<i64>,
        height: Option<i64>,
        #[serde(default = "default_true")]
        checkerboard: bool,
        /// Also write the PNG to this path.
        output: Option<PathBuf>,
    },
    Undo,
    Redo,
    Save { path: Option<PathBuf> },
    Export { path: PathBuf },
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::DocumentNew { .. } => "document_new",
            Self::DocumentOpen { .. } => "document_open",
            Self::DocumentInfo => "document_info",
            Self::LayerAdd { .. } => "layer_add",
            Self::LayerDuplicate { .. } => "layer_duplicate",
            Self::LayerDelete { .. } => "layer_delete",
            Self::LayerSet { .. } => "layer_set",
            Self::LayerSelect { .. } => "layer_select",
            Self::LayerMove { .. } => "layer_move",
            Self::LayerGroup { .. } => "layer_group",
            Self::LayerUngroup { .. } => "layer_ungroup",
            Self::SetBlendSpace { .. } => "set_blend_space",
            Self::PaintStroke { .. } => "paint_stroke",
            Self::FillRect { .. } => "fill_rect",
            Self::ClearRect { .. } => "clear_rect",
            Self::PlaceImage { .. } => "place_image",
            Self::SampleColor { .. } => "sample_color",
            Self::RenderPreview { .. } => "render_preview",
            Self::Undo => "undo",
            Self::Redo => "redo",
            Self::Save { .. } => "save",
            Self::Export { .. } => "export",
        }
    }

    /// Whether the command changes the document (for "modified" tracking).
    pub fn mutates(&self) -> bool {
        !matches!(
            self,
            Self::DocumentInfo | Self::SampleColor { .. } | Self::RenderPreview { .. } | Self::Save { .. } | Self::Export { .. }
        )
    }

    pub fn parse(v: Value) -> anyhow::Result<Self> {
        serde_json::from_value(v).map_err(|e| anyhow::anyhow!("invalid command: {e}"))
    }
}

#[derive(Debug, Default)]
pub struct Output {
    pub result: Value,
    /// PNG bytes (render_preview).
    pub image_png: Option<Vec<u8>>,
    /// The session was replaced (document_new / document_open).
    pub replaced_document: bool,
}

impl Output {
    fn json(result: Value) -> Self {
        Self { result, ..Default::default() }
    }
}

pub struct Ctx<'a> {
    pub gpu: &'a GpuContext,
    pub format: wgpu::TextureFormat,
    pub session: &'a mut Option<Session>,
}

fn session(s: &mut Option<Session>) -> anyhow::Result<&mut Session> {
    s.as_mut().context("no document open – use document_new or document_open first")
}

/// Resolves `layer` (default: active), checks it exists and makes it active.
fn target_layer(s: &mut Session, layer: Option<LayerId>) -> anyhow::Result<LayerId> {
    let id = layer.unwrap_or(s.engine.doc().active);
    ensure!(s.engine.doc().find(id).is_some(), "layer {id} does not exist");
    s.engine.select(|d| d.active = id);
    Ok(id)
}

fn pixel_layer(s: &mut Session, layer: Option<LayerId>) -> anyhow::Result<LayerId> {
    let id = target_layer(s, layer)?;
    ensure!(!s.engine.doc().find(id).unwrap().is_group(), "layer {id} is a group – pick a pixel layer");
    Ok(id)
}

fn not_painting(s: &Session) -> anyhow::Result<()> {
    ensure!(!s.engine.stroke_active(), "the user is painting right now – retry in a moment");
    Ok(())
}

pub fn execute(ctx: &mut Ctx, cmd: Command) -> anyhow::Result<Output> {
    match cmd {
        Command::DocumentNew { width, height } => {
            ensure!((1..=200_000).contains(&width) && (1..=200_000).contains(&height), "size out of range");
            *ctx.session = Some(session::load(ctx.gpu, ctx.format, Source::Blank(width, height))?);
            let mut out = Output::json(info(session(ctx.session)?));
            out.replaced_document = true;
            Ok(out)
        }
        Command::DocumentOpen { path } => {
            ensure!(path.exists(), "{} does not exist", path.display());
            *ctx.session = Some(session::load(ctx.gpu, ctx.format, Source::File(path))?);
            let mut out = Output::json(info(session(ctx.session)?));
            out.replaced_document = true;
            Ok(out)
        }
        cmd => {
            let s = session(ctx.session)?;
            execute_on(s, cmd)
        }
    }
}

fn edit(s: &mut Session, label: &str, f: impl FnOnce(&mut Document<Tile>) -> bool) -> anyhow::Result<()> {
    not_painting(s)?;
    ensure!(s.engine.edit(label, f), "{label}: nothing changed");
    Ok(())
}

fn execute_on(s: &mut Session, cmd: Command) -> anyhow::Result<Output> {
    let ok = |s: &Session| Ok(Output::json(json!({ "ok": true, "active_layer": s.engine.doc().active })));
    match cmd {
        Command::DocumentNew { .. } | Command::DocumentOpen { .. } => unreachable!(),
        Command::DocumentInfo => Ok(Output::json(info(s))),
        Command::LayerAdd { name } => {
            not_painting(s)?;
            let mut id = 0;
            s.engine.edit("Neue Ebene", |d| {
                id = d.add_pixel_layer();
                if let Some(n) = name {
                    d.find_mut(id).unwrap().name = n;
                }
                true
            });
            Ok(Output::json(json!({ "layer": id })))
        }
        Command::LayerDuplicate { layer } => {
            target_layer(s, layer)?;
            let mut id = None;
            edit(s, "Ebene duplizieren", |d| {
                id = d.duplicate_active();
                id.is_some()
            })?;
            Ok(Output::json(json!({ "layer": id })))
        }
        Command::LayerDelete { layer } => {
            target_layer(s, layer)?;
            edit(s, "Ebene löschen", |d| d.remove_active()).context("cannot delete the last layer")?;
            ok(s)
        }
        Command::LayerSet { layer, name, opacity, blend, visible } => {
            let id = target_layer(s, layer)?;
            let blend = blend
                .map(|b| {
                    BlendMode::from_key(&b).with_context(|| {
                        let keys: Vec<_> = BlendMode::ALL.iter().map(|m| m.key()).collect();
                        format!("unknown blend mode {b:?}; valid: {}", keys.join(", "))
                    })
                })
                .transpose()?;
            if let Some(o) = opacity {
                ensure!((0.0..=1.0).contains(&o), "opacity must be 0.0–1.0");
            }
            edit(s, "Ebeneneigenschaften", |d| {
                let l = d.find_mut(id).unwrap();
                if let Some(n) = name {
                    l.name = n;
                }
                if let Some(o) = opacity {
                    l.opacity = o;
                }
                if let Some(b) = blend {
                    l.blend = b;
                }
                if let Some(v) = visible {
                    l.visible = v;
                }
                true
            })?;
            ok(s)
        }
        Command::LayerSelect { layer } => {
            target_layer(s, Some(layer))?;
            ok(s)
        }
        Command::LayerMove { layer, steps } => {
            target_layer(s, layer)?;
            not_painting(s)?;
            let mut moved = 0;
            s.engine.edit("Ebene verschieben", |d| {
                for _ in 0..steps.unsigned_abs() {
                    if !d.move_active(steps.signum()) {
                        break;
                    }
                    moved += steps.signum();
                }
                moved != 0
            });
            Ok(Output::json(json!({ "moved": moved })))
        }
        Command::LayerGroup { layer } => {
            target_layer(s, layer)?;
            let mut id = None;
            edit(s, "Gruppieren", |d| {
                id = d.group_active();
                id.is_some()
            })?;
            Ok(Output::json(json!({ "group": id })))
        }
        Command::LayerUngroup { layer } => {
            target_layer(s, layer)?;
            edit(s, "Gruppierung aufheben", |d| d.ungroup_active()).context("layer is not a group")?;
            ok(s)
        }
        Command::SetBlendSpace { space } => {
            let space = BlendSpace::from_key(&space).context("space must be \"srgb\" or \"linear\"")?;
            edit(s, "Blend-Farbraum", |d| {
                let changed = d.blend_space != space;
                d.blend_space = space;
                changed
            })?;
            ok(s)
        }
        Command::PaintStroke { layer, points, color, size, hardness, opacity, flow, spacing, eraser } => {
            pixel_layer(s, layer)?;
            not_painting(s)?;
            ensure!(!points.is_empty(), "points must not be empty");
            ensure!(size > 0.0 && size <= 5000.0, "size must be 0–5000 px");
            let (color, alpha) = match &color {
                Some(c) => c.linear()?,
                None if eraser => ([0.0; 3], 1.0),
                None => bail!("color is required unless eraser is true"),
            };
            let settings = BrushSettings {
                size,
                hardness: hardness.unwrap_or(0.8).clamp(0.0, 1.0),
                flow: flow.unwrap_or(1.0).clamp(0.0, 1.0),
                opacity: (opacity.unwrap_or(1.0) * alpha).clamp(0.0, 1.0),
                spacing: spacing.unwrap_or(0.1).clamp(0.01, 10.0),
                min_size: 0.15,
                smoothing: 0.0,
                color,
                eraser,
            };
            let samples = points
                .iter()
                .map(|p| {
                    ensure!(p.len() == 2 || p.len() == 3, "each point is [x, y] or [x, y, pressure]");
                    Ok(InputSample { pos: DVec2::new(p[0], p[1]), pressure: p.get(2).map_or(1.0, |&v| v as f32) })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            s.engine
                .begin_stroke(StrokeParams { opacity: settings.opacity, erase: eraser })
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut stroke = Stroke::new(settings);
            let mut dabs = 0;
            for p in &samples {
                let d = stroke.add(*p);
                dabs += d.len();
                s.engine.stroke_dabs(&d);
            }
            let d = stroke.finish(*samples.last().unwrap());
            dabs += d.len();
            s.engine.stroke_dabs(&d);
            s.engine.end_stroke(if eraser { "Radieren" } else { "Pinselstrich" });
            Ok(Output::json(json!({ "ok": true, "dabs": dabs })))
        }
        Command::FillRect { layer, x, y, width, height, color, opacity } => {
            let id = pixel_layer(s, layer)?;
            let (rgb, a) = color.linear()?;
            let a = a * opacity.unwrap_or(1.0).clamp(0.0, 1.0);
            modify_rect(s, id, (x, y, width, height), "Fläche füllen", |px| {
                for k in 0..3 {
                    px[k] = rgb[k] * a + px[k] * (1.0 - a);
                }
                px[3] = a + px[3] * (1.0 - a);
            })
        }
        Command::ClearRect { layer, x, y, width, height } => {
            let id = pixel_layer(s, layer)?;
            modify_rect(s, id, (x, y, width, height), "Fläche löschen", |px| *px = [0.0; 4])
        }
        Command::PlaceImage { path, x, y, name } => place_image(s, &path, x.unwrap_or(0), y.unwrap_or(0), name),
        Command::SampleColor { x, y } => sample(s, x, y),
        Command::RenderPreview { max_size, x, y, width, height, checkerboard, output } => {
            let mut out = preview(s, max_size.unwrap_or(1024), (x, y, width, height), checkerboard)?;
            if let (Some(path), Some(png)) = (&output, &out.image_png) {
                std::fs::write(path, png).with_context(|| format!("cannot write {}", path.display()))?;
                out.result["output"] = json!(path);
            }
            Ok(out)
        }
        Command::Undo => {
            not_painting(s)?;
            Ok(Output::json(json!({ "undone": s.engine.undo() })))
        }
        Command::Redo => {
            not_painting(s)?;
            Ok(Output::json(json!({ "redone": s.engine.redo() })))
        }
        Command::Save { path } => {
            let r = s.save(path.as_deref())?;
            Ok(Output::json(json!({
                "path": r.path, "tiles_written": r.tiles_written, "bytes": r.bytes, "incremental": r.incremental
            })))
        }
        Command::Export { path } => {
            let ms = s.export(&path)?;
            Ok(Output::json(json!({ "path": path, "ms": ms.round() })))
        }
    }
}

/// Document summary for agents.
pub fn info(s: &Session) -> Value {
    fn layers(ls: &[imagine_core::Layer<Tile>]) -> Vec<Value> {
        ls.iter()
            .rev() // top → bottom, like a layers panel
            .map(|l| {
                let mut v = json!({
                    "id": l.id,
                    "name": l.name,
                    "blend": l.blend.key(),
                    "opacity": l.opacity,
                    "visible": l.visible,
                });
                match &l.content {
                    Content::Pixels(t) => {
                        v["type"] = "pixels".into();
                        v["tiles"] = t.len().into();
                        if let (Some(x0), Some(y0), Some(x1), Some(y1)) = (
                            t.keys().map(|c| c.x).min(),
                            t.keys().map(|c| c.y).min(),
                            t.keys().map(|c| c.x).max(),
                            t.keys().map(|c| c.y).max(),
                        ) {
                            let ts = TILE_SIZE as i64;
                            v["tile_bounds"] = json!([
                                x0 as i64 * ts,
                                y0 as i64 * ts,
                                (x1 - x0 + 1) as i64 * ts,
                                (y1 - y0 + 1) as i64 * ts
                            ]);
                        }
                    }
                    Content::Group(c) => {
                        v["type"] = "group".into();
                        v["children"] = layers(c).into();
                    }
                }
                v
            })
            .collect()
    }
    let d = s.engine.doc();
    let (undo, redo) = s.engine.history_len();
    json!({
        "name": s.name,
        "width": d.width,
        "height": d.height,
        "blend_space": d.blend_space.key(),
        "active_layer": d.active,
        "layers_top_to_bottom": layers(&d.layers),
        "history": { "undo_steps": undo, "redo_steps": redo },
        "source": s.source,
        "save_path": s.default_save_path(),
    })
}

/// Applies `f` to every pixel (premultiplied linear RGBA) inside the rectangle, as one undo step.
fn modify_rect(
    s: &mut Session,
    layer: LayerId,
    (x, y, w, h): (i64, i64, i64, i64),
    label: &str,
    f: impl Fn(&mut [f32; 4]),
) -> anyhow::Result<Output> {
    not_painting(s)?;
    ensure!(w > 0 && h > 0, "width and height must be > 0");
    let (dw, dh) = s.engine.size();
    let (x0, y0) = (x.max(0), y.max(0));
    let (x1, y1) = ((x + w).min(dw as i64), (y + h).min(dh as i64));
    ensure!(x0 < x1 && y0 < y1, "rectangle lies outside the document");
    let ts = TILE_SIZE as i64;
    let coords: Vec<TileCoord> = tiles_in_rect((x0 as f64, y0 as f64), (x1 as f64, y1 as f64), dw, dh).collect();
    let mut tiles = Vec::with_capacity(coords.len());
    for c in coords {
        let mut px = s.engine.read_layer_tile(layer, c).unwrap_or_else(|| vec![f16::ZERO; (ts * ts * 4) as usize]);
        let (ox, oy) = c.origin();
        for py in (y0 - oy).max(0)..(y1 - oy).min(ts) {
            for pxx in (x0 - ox).max(0)..(x1 - ox).min(ts) {
                let i = ((py * ts + pxx) * 4) as usize;
                let mut v = [px[i], px[i + 1], px[i + 2], px[i + 3]].map(f16::to_f32);
                f(&mut v);
                for k in 0..4 {
                    px[i + k] = f16::from_f32(v[k]);
                }
            }
        }
        let empty = px.iter().skip(3).step_by(4).all(|a| a.to_f32() == 0.0);
        tiles.push((c, (!empty).then_some(px)));
    }
    let n = tiles.len();
    ensure!(s.engine.replace_tiles(layer, tiles, label), "{label} failed (GPU memory full?)");
    Ok(Output::json(json!({ "ok": true, "tiles": n })))
}

fn place_image(s: &mut Session, path: &std::path::Path, x: i64, y: i64, name: Option<String>) -> anyhow::Result<Output> {
    not_painting(s)?;
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    reader.no_limits();
    let img = reader.decode().with_context(|| format!("cannot decode {}", path.display()))?.into_rgba8();
    let (iw, ih) = img.dimensions();
    let (dw, dh) = s.engine.size();
    let px = Pixels::Srgb8(&img);
    let coords: Vec<TileCoord> =
        tiles_in_rect((x as f64, y as f64), ((x + iw as i64) as f64, (y + ih as i64) as f64), dw, dh).collect();
    let tiles: Vec<(TileCoord, Option<Vec<f16>>)> = coords
        .into_iter()
        .filter_map(|c| {
            let (ox, oy) = c.origin();
            extract_region(&px, iw, ih, ox - x, oy - y).map(|t| (c, Some(t)))
        })
        .collect();
    ensure!(!tiles.is_empty(), "image does not overlap the document");
    let name = name.unwrap_or_else(|| path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    let mut id = 0;
    s.engine.edit("Neue Ebene", |d| {
        id = d.add_pixel_layer();
        d.find_mut(id).unwrap().name = name;
        true
    });
    let n = tiles.len();
    ensure!(s.engine.replace_tiles(id, tiles, "Bild platzieren"), "placing failed (GPU memory full?)");
    Ok(Output::json(json!({ "layer": id, "width": iw, "height": ih, "tiles": n })))
}

fn to_srgb8(px: [f32; 4]) -> ([u8; 3], f32) {
    let a = px[3].clamp(0.0, 1.0);
    let c = if a > 0.0 { [0, 1, 2].map(|i| (linear_to_srgb((px[i] / a).clamp(0.0, 1.0)) * 255.0).round() as u8) } else { [0; 3] };
    (c, a)
}

fn sample(s: &mut Session, x: f64, y: f64) -> anyhow::Result<Output> {
    let (dw, dh) = s.engine.size();
    ensure!(x >= 0.0 && y >= 0.0 && x < dw as f64 && y < dh as f64, "point outside the document");
    s.engine.flush();
    let c = TileCoord::containing(x, y);
    let tile = s.engine.read_composite(&[c]).remove(0);
    let (ox, oy) = c.origin();
    let i = (((y as i64 - oy) * TILE_SIZE as i64 + (x as i64 - ox)) * 4) as usize;
    let (rgb, a) = to_srgb8([tile[i], tile[i + 1], tile[i + 2], tile[i + 3]].map(f16::to_f32));
    Ok(Output::json(json!({
        "hex": format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2]),
        "rgb": rgb,
        "alpha": (a * 1000.0).round() / 1000.0,
    })))
}

/// Flattened image (or a region of it) as PNG, at most `max_size` px on its longer side.
fn preview(
    s: &mut Session,
    max_size: u32,
    region: (Option<i64>, Option<i64>, Option<i64>, Option<i64>),
    checkerboard: bool,
) -> anyhow::Result<Output> {
    let max_size = max_size.clamp(16, 4096);
    let (dw, dh) = s.engine.size();
    let rx = region.0.unwrap_or(0).clamp(0, dw as i64 - 1);
    let ry = region.1.unwrap_or(0).clamp(0, dh as i64 - 1);
    let rw = region.2.unwrap_or(dw as i64 - rx).clamp(1, dw as i64 - rx);
    let rh = region.3.unwrap_or(dh as i64 - ry).clamp(1, dh as i64 - ry);

    // Read the composite at the mip level closest to the requested size.
    let longest = rw.max(rh) as f64;
    let mip = (longest / max_size as f64).log2().floor().clamp(0.0, 8.0) as u32;
    let scale = 1i64 << mip;
    s.engine.flush();
    let coords: Vec<TileCoord> =
        tiles_in_rect((rx as f64, ry as f64), ((rx + rw) as f64, (ry + rh) as f64), dw, dh).collect();
    let present: Vec<TileCoord> = {
        let have: std::collections::HashSet<_> = s.engine.composite_coords().into_iter().collect();
        coords.into_iter().filter(|c| have.contains(c)).collect()
    };
    let (iw, ih) = (((rw + scale - 1) / scale) as u32, ((rh + scale - 1) / scale) as u32);
    let mut img = RgbaImage::new(iw.max(1), ih.max(1));
    for batch in present.chunks(256) {
        let px = s.engine.read_composite_mip(batch, mip);
        let tiles: Vec<_> = batch.iter().copied().zip(px).collect();
        blit_region(&mut img, &tiles, TILE_SIZE >> mip, (rx / scale, ry / scale));
    }
    if img.width().max(img.height()) > max_size {
        let f = max_size as f64 / img.width().max(img.height()) as f64;
        let (w, h) = (((img.width() as f64 * f).round() as u32).max(1), ((img.height() as f64 * f).round() as u32).max(1));
        img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
    }
    if checkerboard {
        for (x, y, p) in img.enumerate_pixels_mut() {
            let bg = if ((x / 8) + (y / 8)) % 2 == 0 { 204u32 } else { 153 };
            let a = p[3] as u32;
            let mix = |c: u8| ((c as u32 * a + bg * (255 - a)) / 255) as u8;
            *p = image::Rgba([mix(p[0]), mix(p[1]), mix(p[2]), 255]);
        }
    }
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    Ok(Output {
        result: json!({
            "width": img.width(),
            "height": img.height(),
            "region": [rx, ry, rw, rh],
            "scale": rw as f64 / img.width() as f64,
        }),
        image_png: Some(png),
        replaced_document: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_commands_and_rejects_typos() {
        let c = Command::parse(json!({"cmd": "fill_rect", "x": 1, "y": 2, "width": 3, "height": 4, "color": "#ff0000"})).unwrap();
        assert_eq!(c.name(), "fill_rect");
        assert!(Command::parse(json!({"cmd": "fill_rect", "x": 1, "y": 2, "widht": 3, "height": 4, "color": "#f00"})).is_err());
        assert!(Command::parse(json!({"cmd": "explode"})).is_err());
        let c = Command::parse(json!({"cmd": "paint_stroke", "points": [[0, 0], [10, 5, 0.5]], "size": 4, "color": [255, 0, 0]})).unwrap();
        assert!(matches!(c, Command::PaintStroke { eraser: false, .. }));
        assert!(!Command::parse(json!({"cmd": "document_info"})).unwrap().mutates());
    }

    #[test]
    fn colours() {
        let (rgb, a) = ColorSpec::Hex("#ffffff80".into()).linear().unwrap();
        assert_eq!(rgb, [1.0; 3]);
        assert!((a - 128.0 / 255.0).abs() < 1e-6);
        assert!(ColorSpec::Hex("#12".into()).linear().is_err());
        assert!(ColorSpec::Rgb(vec![300.0, 0.0, 0.0]).linear().is_err());
        assert_eq!(ColorSpec::Rgb(vec![0.0, 0.0, 0.0]).linear().unwrap().1, 1.0);
    }
}
