//! Headless GPU tests. Skipped (with a message) when no Vulkan adapter exists.

use glam::DVec2;
use half::f16;
use imagine_core::{BrushSettings, InputSample, Stroke, TILE_SIZE, TileCoord, View};
use imagine_gpu::{Canvas, GpuContext, Overlay};

fn gpu() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn px(tile: &[f16], x: u32, y: u32) -> [f32; 4] {
    let i = ((y * TILE_SIZE + x) * 4) as usize;
    [tile[i], tile[i + 1], tile[i + 2], tile[i + 3]].map(f16::to_f32)
}

#[test]
fn paint_erase_and_sparsity() {
    let Some(g) = gpu() else { return };
    let mut c = Canvas::new(&g.device, &g.queue, wgpu::TextureFormat::Rgba8UnormSrgb, 1000, 1000);
    let settings = BrushSettings {
        size: 20.0,
        hardness: 1.0,
        flow: 1.0,
        smoothing: 0.0,
        color: [1.0, 0.0, 0.0],
        ..Default::default()
    };
    // A dab right on the corner of four tiles.
    let mut stroke = Stroke::new(settings);
    let dabs = stroke.add(InputSample { pos: DVec2::new(256.0, 256.0), pressure: 1.0 });
    c.paint(&dabs, false);
    let stats = c.flush();
    assert_eq!(stats.tile_paint_passes, 4);
    assert_eq!(c.tile_count(), 4, "only touched tiles are allocated");

    let t = c.read_tile(TileCoord::new(1, 1));
    let p = px(&t, 2, 2);
    assert!(p[3] > 0.99 && p[0] > 0.99 && p[1] < 0.01, "painted red: {p:?}");
    assert_eq!(px(&t, 100, 100)[3], 0.0, "far pixel untouched");

    // Erase it again.
    let mut eraser = Stroke::new(BrushSettings { eraser: true, ..settings });
    c.paint(&eraser.add(InputSample { pos: DVec2::new(256.0, 256.0), pressure: 1.0 }), true);
    c.flush();
    let t = c.read_tile(TileCoord::new(1, 1));
    assert!(px(&t, 2, 2)[3] < 0.01);

    // Erasing where nothing exists allocates nothing.
    c.paint(&eraser.add(InputSample { pos: DVec2::new(900.0, 900.0), pressure: 1.0 }), true);
    c.flush();
    assert_eq!(c.tile_count(), 4);
}

#[test]
fn upload_survives_pool_growth_and_renders() {
    let Some(g) = gpu() else { return };
    let (w, h) = (TILE_SIZE * 12, TILE_SIZE * 12); // 144 tiles > initial capacity (64)
    let mut c = Canvas::new(&g.device, &g.queue, wgpu::TextureFormat::Rgba8UnormSrgb, w, h);
    let n = (TILE_SIZE * TILE_SIZE * 4) as usize;
    for y in 0..12 {
        for x in 0..12 {
            let v = f16::from_f32((y * 12 + x) as f32 / 144.0);
            let mut data = vec![v; n];
            data.iter_mut().skip(3).step_by(4).for_each(|a| *a = f16::ONE);
            assert!(c.upload_tile(TileCoord::new(x, y), &data));
        }
    }
    assert_eq!(c.tile_count(), 144);
    for (x, y) in [(0, 0), (5, 3), (11, 11)] {
        let t = c.read_tile(TileCoord::new(x, y));
        let expected = (y * 12 + x) as f32 / 144.0;
        assert!((px(&t, 7, 9)[0] - expected).abs() < 1e-3, "tile ({x},{y}) after growth");
    }

    // Draw into an offscreen target to validate the display pipelines.
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 640, height: 480, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view_tex = target.create_view(&Default::default());
    let mut view = View { viewport: DVec2::new(640.0, 480.0), ..Default::default() };
    view.fit(w, h);
    let stats = c.render(
        &view_tex,
        &view,
        &Overlay { cursor: Some((DVec2::new(10.0, 10.0), 5.0)), checker_px: 8.0 },
    );
    assert_eq!(stats.visible_tiles, 144);
    view.set_zoom_at(view.viewport * 0.5, 8.0);
    let stats = c.render(&view_tex, &view, &Overlay::default());
    assert!(stats.visible_tiles <= 4, "culling at high zoom: {}", stats.visible_tiles);
    g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
