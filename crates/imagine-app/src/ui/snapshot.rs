//! `imagine --ui-snapshot OUT.png [DATEI]`: renders canvas and interface
//! offscreen into a PNG (layout checks without opening a window).

use std::path::Path;

use anyhow::Context;
use glam::DVec2;
use imagine_core::{BlendMode, View};
use imagine_gpu::Overlay;

use super::Gui;
use super::panels::{self, BrushModel, Model, State, ToolId};
use crate::loader::Source;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn run(source: Source, out: &Path, size: (u32, u32)) -> anyhow::Result<()> {
    let gpu = pollster::block_on(imagine_gpu::GpuContext::new(imagine_gpu::GpuContext::create_instance(), None))?;
    let mut s = crate::session::load(&gpu, FORMAT, source)?;
    let (w, h) = size;
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snapshot"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
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
    s.engine.edit("Ebene", |d| {
        let id = d.add_pixel_layer();
        d.find_mut(id).unwrap().name = "Schatten".into();
        d.group_active();
        let id = d.add_pixel_layer();
        d.find_mut(id).unwrap().name = "Licht".into();
        d.find_mut(id).unwrap().blend = BlendMode::Screen;
        true
    });
    let _ = s.engine.add_layer_mask("Maske");
    s.engine.edit("Ausblenden", |d| {
        let id = d.active;
        d.find_mut(id).unwrap().visible = false;
        true
    });
    let mut view = View { viewport: DVec2::new(w as f64, h as f64), ..Default::default() };
    let (dw, dh) = s.engine.size();
    view.fit(dw, dh);
    let doc = s.engine.doc();
    let active = doc.active_layer().context("keine Ebene")?;
    let (labels, done) = s.engine.history_labels();
    let model = Model {
        doc_size: (dw, dh),
        name: s.name.clone(),
        modified: true,
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
        wand_tolerance: 32.0,
        wand_contiguous: true,
        boxed: None,
        mask_edit: false,
        layers: panels::layer_rows(doc),
        active: active.id,
        active_opacity: active.opacity,
        active_blend: active.blend,
        history: labels,
        history_done: done,
        has_selection: doc.selection.is_some(),
        status: "0.14 ms".into(),
        busy: false,
    };
    tracing::info!("UI-Schnappschuss {w}×{h}");
    let palette = std::env::var("IMAGINE_SNAPSHOT_PALETTE").ok();
    if let Some(q) = &palette {
        state.palette = Some((q.clone(), 0));
    }
    let mut model = model;
    if let Ok(t) = std::env::var("IMAGINE_SNAPSHOT_TOOL") {
        model.tool = match t.as_str() {
            "wand" => ToolId::Wand,
            "smudge" => ToolId::Smudge,
            "move" => ToolId::Move,
            _ => ToolId::Brush,
        };
        model.boxed = (t == "move").then_some("Frei transformieren");
    }
    // A few passes so that egui's layout settles.
    // Enough passes (and time) for layout and fade-in animations to settle.
    let mut ui_ms = Vec::new();
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_millis(15));
        s.engine.render(&target, &view, &Overlay { checker_px: 8.0, ..Default::default() });
        let t0 = std::time::Instant::now();
        gui.run((w, h), ppp, |ctx| {
            panels::build(ctx, &model, &mut state);
        });
        gui.paint(&gpu.device, &gpu.queue, &target, (w, h));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        ui_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let tail = &ui_ms[10..];
    println!("Oberfläche pro Frame (inkl. GPU): Ø {:.2} ms, max {:.2} ms", tail.iter().sum::<f64>() / tail.len() as f64, tail.iter().cloned().fold(0.0, f64::max));

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
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let data = buf.slice(..).get_mapped_range().expect("mapped");
    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        let src = &data[(y * row) as usize..][..(w * 4) as usize];
        img.as_mut()[(y * w * 4) as usize..][..(w * 4) as usize].copy_from_slice(src);
    }
    img.save(out).with_context(|| format!("{} schreiben", out.display()))?;
    println!("{}", out.display());
    Ok(())
}
