//! `mosaix --ui-snapshot OUT.png [DATEI]`: renders canvas and interface
//! offscreen into a PNG (layout checks without opening a window).

use std::path::Path;

use anyhow::Context;
use glam::DVec2;
use mosaix_core::{BlendMode, View, tr};
use mosaix_gpu::Overlay;

use super::Gui;
use super::panels::{self, BrushModel, Model, State, ToolId};
use crate::loader::Source;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn run(source: Source, out: &Path, size: (u32, u32)) -> anyhow::Result<()> {
    let gpu = pollster::block_on(mosaix_gpu::GpuContext::new(
        mosaix_gpu::GpuContext::create_instance(),
        None,
    ))?;
    let mut s = crate::session::load(&gpu, FORMAT, source)?;
    let (w, h) = size;
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snapshot"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target = tex.create_view(&Default::default());
    let mut gui = Gui::new(&gpu.device, FORMAT);
    let mut state = State::default();
    let ppp = 1.0;

    // Some structure for the layer panel.
    s.engine.edit(tr("Layer"), |d| {
        let id = d.add_pixel_layer();
        d.find_mut(id).unwrap().name = tr("Shadow").into();
        d.group_active();
        let id = d.add_pixel_layer();
        d.find_mut(id).unwrap().name = tr("Light").into();
        d.find_mut(id).unwrap().blend = BlendMode::Screen;
        true
    });
    let _ = s.engine.add_layer_mask(tr("Mask"));
    s.engine.edit(tr("Hide"), |d| {
        let id = d.active;
        d.find_mut(id).unwrap().visible = false;
        true
    });
    let mut view = View {
        viewport: DVec2::new(w as f64, h as f64),
        ..Default::default()
    };
    let (dw, dh) = s.engine.size();
    view.fit(dw, dh);
    let doc = s.engine.doc();
    let active = doc.active_layer().context(tr("no layer"))?;
    let (labels, done) = s.engine.history_labels();
    let model = Model {
        doc_size: (dw, dh),
        name: s.name.clone(),
        modified: true,
        tabs: std::env::var("MOSAIX_SNAPSHOT_TABS").map(|v| v.split(',').map(|n| (n.to_string(), n.starts_with('*'))).collect()).unwrap_or_default(),
        tab: 0,
        history_source: 0,
        prefs: Default::default(),
        language: "auto".into(),
        zoom: view.zoom,
        rotation_deg: 0.0,
        cursor_doc: Some((dw as f64 * 0.4, dh as f64 * 0.3)),
        tool: ToolId::Brush,
        color: [0.8, 0.1, 0.05],
        brush: BrushModel {
            size: 40.0,
            hardness: 0.8,
            flow: 0.25,
            opacity: 1.0,
            spacing: 0.1,
            smoothing: 0.4,
            tip: 0,
            angle: 0.0,
            roundness: 1.0,
            size_jitter: 0.0,
            angle_jitter: 0.0,
            scatter: 0.0,
            flow_jitter: 0.0,
            follow_direction: false,
        },
        smudge_strength: 0.7,
        tone: Default::default(),
        clone_all: false,
        clone_aligned: true,
        clone_source: false,
        wand_tolerance: 32.0,
        wand_contiguous: true,
        bucket_all: false,
        bg_color: [1.0; 3],
        custom_tips: Vec::new(),
        patterns: std::sync::Arc::new(mosaix_core::pattern::Pattern::builtin()),
        filter_plugins: std::sync::Arc::new(crate::app::filters::load_library(&crate::app::filters::filter_dir())),
        actions: std::sync::Arc::new(crate::app::actions::list(&crate::app::actions::actions_dir())),
        recording: std::env::var("MOSAIX_SNAPSHOT_RECORDING").ok().and_then(|v| v.parse().ok()),
        eyedrop_radius: 0,
        eyedrop_all: true,
        boxed: std::env::var_os("MOSAIX_SNAPSHOT_BOX").map(|_| "Free Transform"),
        box_angle: std::env::var_os("MOSAIX_SNAPSHOT_BOX").map(|_| 30.0),
        box_geom: std::env::var_os("MOSAIX_SNAPSHOT_BOX").map(|_| [1200.0, 800.0, 640.0, 480.0]),
        box_mode: std::env::var_os("MOSAIX_SNAPSHOT_BOX").map(|_| crate::boxedit::Mode::Scale),
        mask_edit: false,
        text_overlay: None,
        path_overlay: None,
        liquify: std::env::var_os("MOSAIX_SNAPSHOT_LIQUIFY")
            .map(|_| (mosaix_gpu::LiquifyTool::TwirlCw, 300.0, 0.6)),
        layers: panels::layer_rows(doc),
        active: active.id,
        selected: std::env::var("MOSAIX_SNAPSHOT_PICKED").map(|v| v.split(',').filter_map(|x| x.parse().ok()).chain([active.id]).collect()).unwrap_or_default(),
        active_opacity: active.opacity,
        active_fill: active.fill_opacity,
        active_blend: active.blend,
        keymap: Default::default(),
        active_pass_through: active.is_group().then_some(active.pass_through),
        history: labels,
        history_done: done,
        has_selection: doc.selection.is_some(),
        can_revert: false,
        channels: doc.channels.iter().map(|c| c.name.clone()).collect(),
        active_bounds: None,
        view,
        guides: if std::env::var_os("MOSAIX_SNAPSHOT_RULERS").is_some() {
            vec![mosaix_core::Guide { vertical: true, pos: dw as f64 / 3.0 }, mosaix_core::Guide { vertical: false, pos: dh as f64 / 2.0 }]
        } else {
            doc.guides.clone()
        },
        panels: Default::default(),
        generative: Default::default(),
        openai_key: Default::default(),
        view_opts: crate::settings::ViewOpts {
            rulers: std::env::var_os("MOSAIX_SNAPSHOT_RULERS").is_some(),
            grid: std::env::var_os("MOSAIX_SNAPSHOT_GRID").is_some(),
            ..Default::default()
        },
        status: "0.14 ms".into(),
        busy: false,
        recent: std::env::var("MOSAIX_SNAPSHOT_HOME")
            .map(|v| v.split(':').filter(|s| !s.is_empty()).map(Into::into).collect())
            .unwrap_or_default(),
        active_adjustment: None,
        active_raw: None,
        active_vector: None,
        active_effects: Some(Vec::new()),
        fonts: Default::default(),
        paths: Vec::new(),
        favourite_fonts: vec!["IBM Plex Sans".into()],
        recent_fonts: vec!["IBM Plex Mono".into()],
        shape_ellipse: false,
        histogram: None,
    };
    tracing::info!("{}", tr!("UI snapshot {w}×{h}", w = w, h = h));
    // The start screen, with these files (":"-separated) as recently opened.
    let home_files: Option<Vec<std::path::PathBuf>> =
        std::env::var("MOSAIX_SNAPSHOT_HOME").ok().map(|v| v.split(':').filter(|s| !s.is_empty()).map(Into::into).collect());
    if let Some(files) = &home_files {
        let mut h = super::home::Home::default();
        h.load_now(files);
        state.home = Some(h);
    }
    state.focus_pane = match std::env::var("MOSAIX_SNAPSHOT_PANE").as_deref() {
        Ok("navigator") => Some(super::dock::Pane::Navigator),
        Ok("info") => Some(super::dock::Pane::Info),
        _ => None,
    };
    state.snapshot_flyout = std::env::var("MOSAIX_SNAPSHOT_FLYOUT").ok().and_then(|v| v.parse().ok());
    if let Ok(menu) = std::env::var("MOSAIX_SNAPSHOT_MENU")
        && let Some(i) = super::palette::MENUS.iter().position(|m| m.eq_ignore_ascii_case(&menu))
    {
        // Roughly where the menu bar puts it.
        const X: [f32; 7] = [4.0, 34.0, 68.0, 114.0, 154.0, 200.0, 236.0];
        state.snapshot_menu = Some((super::palette::MENUS[i], X[i]));
    }
    let palette = std::env::var("MOSAIX_SNAPSHOT_PALETTE").ok();
    if let Some(q) = &palette {
        state.palette = Some((q.clone(), 0));
    }
    match std::env::var("MOSAIX_SNAPSHOT_DIALOG").as_deref() {
        Ok("open") => {
            let home = std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_default();
            state.file = Some(super::files::FileDialog::new(
                super::files::FileMode::Open,
                home.join("Pictures"),
                String::new(),
            ));
        }
        Ok("new") => state.new_doc = Some(Default::default()),
        Ok("record") => state.record_name = Some("Web Look".into()),
        Ok("ai") => state.ai_download = Some(mosaix_ai::SUBJECT.id),
        Ok(d) if d.starts_with("raw:") => {
            let mut r = super::raw_ui::RawDialog::new(d[4..].into());
            r.wait();
            state.raw = Some(r);
        }
        Ok("export") => {
            let mut d = super::files::FileDialog::new(
                super::files::FileMode::ExportAs,
                std::env::temp_dir(),
                tr("image.jpg").into(),
            );
            d.export.space = mosaix_io::export::ColorSpace::DisplayP3;
            state.file = Some(d);
        }
        Ok("confirm") => state.confirm = Some(tr("They will be lost when quitting.").into()),
        Ok("generative") => state.open_generative(),
        Ok("stroke") => state.open_stroke(),
        Ok("prefs") => state.open_prefs(),
        Ok("adjust") => state.open_adjust(mosaix_core::Adjustment::defaults().into_iter().find(|a| a.key() == "hue_saturation").unwrap()),
        Ok("filter") => state.open_filter(mosaix_core::Filter::GaussianBlur { radius: 25.0 }),
        Ok(d) if d.starts_with("filter:") => {
            let lib = crate::app::filters::load_library(&crate::app::filters::filter_dir());
            if let Some(f) = crate::app::filters::find(&lib, &d[7..]) {
                state.open_filter(f);
            }
        }
        _ => {}
    }
    let mut model = model;
    // Paint something into "Schatten" so its thumbnail differs.
    if let Some(id) = model
        .layers
        .iter()
        .find(|r| r.name == tr("Shadow"))
        .map(|r| r.id)
    {
        s.engine.select(|d| d.active = id);
        let _ = s.engine.select_shape(
            &mosaix_core::mask::Shape::Ellipse {
                cx: dw as f64 * 0.3,
                cy: dh as f64 * 0.5,
                rx: dw as f64 * 0.2,
                ry: dh as f64 * 0.3,
            },
            mosaix_core::mask::CombineOp::Replace,
            tr("Selection"),
        );
        let _ = s.engine.fill(
            mosaix_gpu::StrokeTarget::Pixels,
            [0.0, 0.0, 0.0, 0.8],
            1.0,
            tr("Fill"),
        );
        let _ = s.engine.deselect("x");
        if std::env::var_os("MOSAIX_SNAPSHOT_SMART").is_some() {
            let _ = s.engine.add_smart_filter(
                &mosaix_core::Filter::GaussianBlur { radius: 40.0 },
                "Smart",
            );
            let _ = s.engine.add_smart_filter(
                &mosaix_core::Filter::AddNoise {
                    amount: 10.0,
                    monochrome: true,
                    gaussian: true,
                    seed: 0,
                },
                "Smart",
            );
            s.engine.edit(tr("Off"), |d| {
                let id = d.active;
                d.find_mut(id).unwrap().filters[1].enabled = false;
                true
            });
            model.layers = panels::layer_rows(s.engine.doc());
            model.active = id;
        }
    }
    if let Ok(j) = std::env::var("MOSAIX_SNAPSHOT_FX") {
        model.active_effects = Some(serde_json::from_str(&j)?);
        state.open_effect(0);
    }
    if let Ok(j) = std::env::var("MOSAIX_SNAPSHOT_PATH") {
        // A path being drawn (JSON list of nodes), the cursor near its start.
        let nodes: Vec<mosaix_core::vector::PathNode> = serde_json::from_str(&j)?;
        let cursor = nodes
            .first()
            .map(|n| view.doc_to_screen(DVec2::new(n.p[0] + 3.0, n.p[1])));
        model.path_overlay = Some(crate::app::pen::overlay(
            &nodes,
            false,
            glam::DAffine2::IDENTITY,
            true,
            None,
            cursor,
            &view,
            ppp,
        ));
        model.tool = ToolId::Pen;
    }
    if let Ok(j) = std::env::var("MOSAIX_SNAPSHOT_VECTOR") {
        let v: mosaix_core::VectorLayer = serde_json::from_str(&j)?;
        s.engine.add_vector(v.clone(), tr("Vector"));
        model.active_vector = Some(v.clone());
        if let Ok(r) = std::env::var("MOSAIX_SNAPSHOT_TEXTEDIT")
            && let mosaix_core::vector::VectorKind::Text(t) = &v.kind
        {
            // "a,b": selection by byte offsets.
            let mut it = r.split(',').filter_map(|x| x.trim().parse::<usize>().ok());
            let (a, b) = (it.next().unwrap_or(0), it.next().unwrap_or(0));
            let sel = mosaix_vector::text_range(t, a, b);
            model.text_overlay = crate::app::textedit::overlay(&v, &sel, &view, ppp);
        }
        if let Ok(q) = std::env::var("MOSAIX_SNAPSHOT_FONTS") {
            let fonts = model.fonts.clone();
            state.vec_editor.fonts.open_for(&v, &fonts, &q);
            if let Ok(f) = std::env::var("MOSAIX_SNAPSHOT_TRY_FONT") {
                state.vec_editor.fonts.try_family(&f);
            }
        }
        model.fonts = std::sync::Arc::new(mosaix_vector::font_groups());
        model.layers = panels::layer_rows(s.engine.doc());
        model.active = s.engine.doc().active;
    }
    if let Ok(j) = std::env::var("MOSAIX_SNAPSHOT_ADJ") {
        let adj: mosaix_core::Adjustment = serde_json::from_str(&j)?;
        s.engine.add_adjustment(adj.clone(), tr("Adjustment"));
        model.active_adjustment = Some(adj);
        model.layers = panels::layer_rows(s.engine.doc());
        model.active = s.engine.doc().active;
        if super::adjust_ui::wants_histogram(model.active_adjustment.as_ref()) {
            model.histogram = Some(s.engine.histogram(Some(model.active)));
        }
    }
    if let Ok(path) = std::env::var("MOSAIX_SNAPSHOT_RAW") {
        // A RAW layer (its Properties panel).
        let p = std::path::Path::new(&path);
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let data = mosaix_core::raw::RawData::new(std::fs::read(p)?, name.clone());
        let mut r = mosaix_core::raw::RawLayer::new(data, Default::default());
        r.develop.exposure = 0.5;
        r.transform = [0.4, 0.0, 0.0, 0.4, 0.0, 0.0];
        let id = s.engine.add_raw(r.clone(), "RAW", tr("Place Image")).unwrap_or_default();
        s.engine.flush();
        model.active_raw = Some(super::raw_ui::ActiveRaw { file: name, develop: r.develop, summary: s.engine.raw_summary(id), smart: r.data.is_image() });
        model.layers = panels::layer_rows(s.engine.doc());
        model.active = s.engine.doc().active;
    }
    if std::env::var_os("MOSAIX_SNAPSHOT_CLIP").is_some() {
        // Clip the second-lowest layer to the one below (row display only).
        let n = model.layers.len();
        if n >= 2 {
            model.layers[n - 2].clip = true;
        }
    }
    while s.engine.update_thumbnails(8) | s.engine.update_mask_thumbnails(8) {}
    for row in &mut model.layers {
        if let Some((slot, uv)) = s.engine.thumbnail(row.id) {
            let view = s.engine.thumbnail_view(slot);
            row.thumb = Some((gui.register_texture(&gpu.device, &view), uv));
        }
        if let Some((_, px)) = s.engine.mask_thumbnail(row.id) {
            let n = panels::THUMB_PX;
            let tex = gui.ctx.load_texture(format!("mask {}", row.id), egui::ColorImage::from_gray([n, n], &px), egui::TextureOptions::LINEAR);
            row.mask_thumb = Some((tex.id(), s.engine.thumbnail_uv()));
            std::mem::forget(tex); // keep the texture alive for the snapshot
        }
    }
    if let Ok(q) = std::env::var("MOSAIX_SNAPSHOT_SHORTCUTS") {
        state.shortcuts = Some(panels::ShortcutsDialog { query: q, capturing: Some("Edit/Copy".into()) });
    }
    if let Ok(t) = std::env::var("MOSAIX_SNAPSHOT_TAB") {
        state.focus_pane = Some(match t.as_str() {
            "styles" => crate::ui::dock::Pane::Styles,
            "color" => crate::ui::dock::Pane::Color,
            "history" => crate::ui::dock::Pane::History,
            "layers" => crate::ui::dock::Pane::Layers,
            _ => crate::ui::dock::Pane::Properties,
        });
    }
    if let Ok(t) = std::env::var("MOSAIX_SNAPSHOT_TOOL") {
        model.tool = match t.as_str() {
            "wand" => ToolId::Wand,
            "object" => ToolId::ObjectSelect,
            "remove" => ToolId::Remove,
            "smudge" => ToolId::Smudge,
            "dodge" => ToolId::Dodge,
            "sponge" => ToolId::Sponge,
            "blur" => ToolId::Blur,
            "clone" => ToolId::Clone,
            "heal" => ToolId::Heal,
            "spot_heal" => ToolId::SpotHeal,
            "text" => ToolId::Text,
            "shape" => ToolId::Shape,
            "move" => ToolId::Move,
            "eyedropper" => ToolId::Eyedropper,
            "bucket" => ToolId::Bucket,
            "poly_lasso" => ToolId::PolyLasso,
            _ => ToolId::Brush,
        };
        // "transform": the Move tool with a free transform open.
        if t == "transform" {
            model.tool = ToolId::Move;
            model.boxed = Some("Free Transform");
        }
    }
    // A few passes so that egui's layout settles.
    // Enough passes (and time) for layout and fade-in animations to settle.
    let mut ui_ms = Vec::new();
    let frames = std::env::var("MOSAIX_SNAPSHOT_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(30);
    // The navigator's picture (like the app does after rendering).
    let nav = {
        let n = super::navigator::NAV;
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("navigator"),
            size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let v = tex.create_view(&Default::default());
        let id = gui.register_texture(&gpu.device, &v);
        (tex, v, id)
    };
    model.panels.navigator = Some(nav.2);
    model.panels.histogram = Some(s.engine.histogram(None));
    for _ in 0..frames {
        std::thread::sleep(std::time::Duration::from_millis(15));
        let _ = match &state.filter_preview {
            Some(panels::FilterPreview::Pixels(f)) => s.engine.set_filter_preview(Some(f)),
            Some(panels::FilterPreview::Smart(i, f)) => {
                s.engine.set_smart_filter_preview(Some((*i, f)))
            }
            None if state.vec_editor.fonts.preview.is_some() => {
                let id = s.engine.doc().active;
                s.engine
                    .set_vector_preview(state.vec_editor.fonts.preview.as_ref().map(|v| (id, v)))
            }
            None => Ok(()),
        };
        s.engine.render(
            &target,
            &view,
            &Overlay {
                checker_px: 8.0,
                ..Default::default()
            },
        );
        s.engine.render_overview(&nav.1, &super::navigator::nav_view(s.engine.size()));
        let t0 = std::time::Instant::now();
        gui.run((w, h), ppp, |ctx| {
            panels::build(ctx, &model, &mut state);
        });
        gui.paint(&gpu.device, &gpu.queue, &target, (w, h));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        ui_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let tail = &ui_ms[ui_ms.len().min(10)..];
    println!(
        "{}",
        tr!(
            "UI per frame (incl. GPU): avg {avg} ms, max {max} ms",
            avg = format!("{:.2}", tail.iter().sum::<f64>() / tail.len() as f64),
            max = format!("{:.2}", tail.iter().cloned().fold(0.0, f64::max))
        )
    );

    // Read back.
    let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snapshot readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([enc.finish()]);
    buf.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let data = buf.slice(..).get_mapped_range().expect("mapped");
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        let src = &data[(y * row) as usize..][..(w * 4) as usize];
        img.as_mut()[(y * w * 4) as usize..][..(w * 4) as usize].copy_from_slice(src);
    }
    img.save(out)
        .with_context(|| tr!("writing {path}", path = out.display()))?;
    println!("{}", out.display());
    Ok(())
}
