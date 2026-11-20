//! Headless GPU tests for the engine. Skipped (with a message) when no
//! suitable Vulkan adapter exists.

use glam::DVec2;
use half::f16;
use imagine_core::blend::composite;
use imagine_core::{BlendMode, BlendSpace, BrushSettings, Document, InputSample, Stroke, TILE_SIZE, TileCoord, View};
use imagine_gpu::{Engine, GpuContext, Overlay, StrokeParams};

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const TS: usize = TILE_SIZE as usize;

fn gpu() -> Option<GpuContext> {
    match pollster::block_on(GpuContext::new(GpuContext::create_instance(), None)) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn px(tile: &[f16], x: usize, y: usize) -> [f32; 4] {
    let i = (y * TS + x) * 4;
    [tile[i], tile[i + 1], tile[i + 2], tile[i + 3]].map(f16::to_f32)
}

fn pattern(seed: usize, opaque: bool) -> Vec<f16> {
    let mut v = vec![f16::ZERO; TS * TS * 4];
    for y in 0..TS {
        for x in 0..TS {
            let a = if opaque { 1.0 } else { 0.2 + 0.8 * (((x * 7 + y * 3 + seed) % 5) as f32 / 4.0) };
            let c = [
                ((x + seed) % 17) as f32 / 16.0,
                ((y + 2 * seed) % 13) as f32 / 12.0,
                ((x + y + seed) % 7) as f32 / 6.0,
            ];
            let i = (y * TS + x) * 4;
            for k in 0..3 {
                v[i + k] = f16::from_f32(c[k] * a);
            }
            v[i + 3] = f16::from_f32(a);
        }
    }
    v
}

fn red_brush() -> BrushSettings {
    BrushSettings { size: 20.0, hardness: 1.0, flow: 1.0, smoothing: 0.0, color: [1.0, 0.0, 0.0], ..Default::default() }
}

fn stroke(e: &mut Engine, at: DVec2, params: StrokeParams) {
    e.begin_stroke(params).unwrap();
    let mut s = Stroke::new(red_brush());
    let mut dabs = s.add(InputSample { pos: at, pressure: 1.0 });
    for i in 1..20 {
        dabs.extend(s.add(InputSample { pos: at + DVec2::new(i as f64 * 0.1, 0.0), pressure: 1.0 }));
    }
    e.stroke_dabs(&dabs);
    e.end_stroke("Pinselstrich");
    e.flush();
}

fn composite_px(e: &Engine, c: TileCoord, x: usize, y: usize) -> [f32; 4] {
    px(&e.read_composite(&[c])[0], x, y)
}

#[test]
fn paint_undo_redo_and_sparsity() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1000, 1000));
    let paint = StrokeParams { opacity: 1.0, erase: false };
    stroke(&mut e, DVec2::new(256.0, 256.0), paint);
    assert_eq!(e.layer_tiles_live(), 4, "corner stroke touches exactly 4 tiles");
    assert_eq!(e.composite_coords().len(), 4);
    let c = TileCoord::new(1, 1);
    let p = composite_px(&e, c, 2, 2);
    assert!(p[0] > 0.99 && p[3] > 0.99 && p[1] < 0.01, "red: {p:?}");

    stroke(&mut e, DVec2::new(256.0, 256.0), StrokeParams { opacity: 1.0, erase: true });
    assert!(composite_px(&e, c, 2, 2)[3] < 0.01, "erased");

    assert_eq!(e.undo().as_deref(), Some("Pinselstrich"));
    e.flush();
    assert!(composite_px(&e, c, 2, 2)[3] > 0.99, "undo restores");
    e.redo();
    e.flush();
    assert!(composite_px(&e, c, 2, 2)[3] < 0.01, "redo erases again");

    // Erasing where nothing exists allocates nothing.
    let before = e.layer_tiles_live();
    stroke(&mut e, DVec2::new(900.0, 900.0), StrokeParams { opacity: 1.0, erase: true });
    assert_eq!(e.layer_tiles_live(), before);

    // Dropping the history frees the tiles only it referenced.
    e.clear_history();
    assert_eq!(e.layer_tiles_live(), 4);
}

#[test]
fn stroke_opacity_caps_buildup() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    // 20 overlapping dabs at full flow would be opaque; stroke opacity caps at 50 %.
    stroke(&mut e, DVec2::new(100.0, 100.0), StrokeParams { opacity: 0.5, erase: false });
    let p = composite_px(&e, TileCoord::new(0, 0), 100, 100);
    assert!((p[3] - 0.5).abs() < 0.01, "alpha capped at stroke opacity: {p:?}");
}

#[test]
fn blend_modes_match_cpu_reference() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let bottom_px = pattern(1, false);
    let top_px = pattern(4, false);
    let bottom = e.doc().active;
    e.upload_tile(bottom, TileCoord::new(0, 0), &bottom_px);
    let mut top = 0;
    e.edit("Neue Ebene", |d| {
        top = d.add_pixel_layer();
        true
    });
    e.upload_tile(top, TileCoord::new(0, 0), &top_px);

    let samples = [(0, 0), (3, 7), (31, 200), (128, 128), (255, 255), (77, 13)];
    let mut worst = 0.0f32;
    for space in [BlendSpace::Srgb, BlendSpace::Linear] {
        for mode in BlendMode::ALL {
            e.edit("Modus", |d| {
                d.blend_space = space;
                let l = d.find_mut(top).unwrap();
                l.blend = mode;
                l.opacity = 0.8;
                true
            });
            e.flush();
            let out = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
            for &(x, y) in &samples {
                let b = px(&bottom_px, x, y);
                let t = px(&top_px, x, y).map(|v| v * 0.8);
                let expected = composite(composite([0.0; 4], b, BlendMode::Normal, space), t, mode, space);
                let got = px(&out, x, y);
                for k in 0..4 {
                    let err = (got[k] - expected[k]).abs();
                    worst = worst.max(err);
                    assert!(err < 4e-3, "{mode:?}/{space:?} at ({x},{y}): gpu {got:?} cpu {expected:?}");
                }
            }
        }
    }
    eprintln!("worst GPU/CPU deviation: {worst:.2e}");

    // Groups: isolated composite, then group opacity/blend.
    e.edit("Gruppe", |d| {
        d.blend_space = BlendSpace::Srgb;
        d.find_mut(top).unwrap().blend = BlendMode::Screen;
        d.find_mut(top).unwrap().opacity = 1.0;
        d.active = top;
        let gid = d.group_active().unwrap();
        let grp = d.find_mut(gid).unwrap();
        grp.blend = BlendMode::Multiply;
        grp.opacity = 0.6;
        true
    });
    e.flush();
    let out = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let s = BlendSpace::Srgb;
    for &(x, y) in &samples {
        let b = composite([0.0; 4], px(&bottom_px, x, y), BlendMode::Normal, s);
        let inner = composite([0.0; 4], px(&top_px, x, y), BlendMode::Screen, s);
        let expected = composite(b, inner.map(|v| v * 0.6), BlendMode::Multiply, s);
        let got = px(&out, x, y);
        assert!(got.iter().zip(expected).all(|(a, b)| (a - b).abs() < 4e-3), "group: {got:?} vs {expected:?}");
    }
}

#[test]
fn many_tiles_span_chunks_and_duplicates_share() {
    let Some(g) = gpu() else { return };
    let (gw, gh) = (30, 20); // 600 tiles > one 256-layer chunk
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(gw * TILE_SIZE, gh * TILE_SIZE));
    let layer = e.doc().active;
    for y in 0..gh as i32 {
        for x in 0..gw as i32 {
            let v = f16::from_f32((y * gw as i32 + x) as f32 / 600.0);
            let mut d = vec![v; TS * TS * 4];
            d.iter_mut().skip(3).step_by(4).for_each(|a| *a = f16::ONE);
            e.upload_tile(layer, TileCoord::new(x, y), &d);
        }
    }
    e.flush();
    assert_eq!(e.composite_coords().len(), 600);
    for (x, y) in [(0, 0), (29, 19), (15, 10)] {
        let expected = (y * gw as i32 + x) as f32 / 600.0;
        let got = composite_px(&e, TileCoord::new(x, y), 5, 5)[0];
        assert!((got - expected).abs() < 1e-3, "tile ({x},{y}): {got} vs {expected}");
    }

    let live = e.layer_tiles_live();
    e.edit("Duplizieren", |d| d.duplicate_active().is_some());
    assert_eq!(e.layer_tiles_live(), live, "duplicate shares tiles");

    // Hiding the copy and the original leaves nothing to composite.
    e.edit("Ausblenden", |d| {
        d.for_each(&mut |_, _| {});
        let ids: Vec<_> = d.display_order();
        for id in ids {
            d.find_mut(id).unwrap().visible = false;
        }
        true
    });
    e.flush();
    assert!(e.composite_coords().is_empty());
    e.undo();
    e.flush();
    assert_eq!(e.composite_coords().len(), 600);

    // Render offscreen: culling + per-chunk draws.
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 640, height: 480, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let tv = target.create_view(&Default::default());
    let mut view = View { viewport: DVec2::new(640.0, 480.0), ..Default::default() };
    let (w, h) = e.size();
    view.fit(w, h);
    let stats = e.render(&tv, &view, &Overlay { cursor: Some((DVec2::splat(10.0), 5.0)), checker_px: 8.0 });
    assert_eq!(stats.visible_tiles, 600);
    view.set_zoom_at(view.viewport * 0.5, 8.0);
    assert!(e.render(&tv, &view, &Overlay::default()).visible_tiles <= 4);
    g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
