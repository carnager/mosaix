//! Headless GPU tests for the engine. Skipped (with a message) when no
//! suitable Vulkan adapter exists.

use glam::DVec2;
use half::f16;
use imagine_core::blend::composite;
use imagine_core::{BlendMode, BlendSpace, BrushSettings, Document, InputSample, Stroke, TILE_SIZE, TileCoord, View};
use imagine_core::mask::{CombineOp, Shape};
use imagine_gpu::{Engine, GpuContext, GradientKind, Overlay, OverlayLine, StrokeParams, StrokeTarget};

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
    let paint = StrokeParams { opacity: 1.0, erase: false, ..Default::default() };
    stroke(&mut e, DVec2::new(256.0, 256.0), paint);
    assert_eq!(e.layer_tiles_live(), 4, "corner stroke touches exactly 4 tiles");
    assert_eq!(e.composite_coords().len(), 4);
    let c = TileCoord::new(1, 1);
    let p = composite_px(&e, c, 2, 2);
    assert!(p[0] > 0.99 && p[3] > 0.99 && p[1] < 0.01, "red: {p:?}");

    stroke(&mut e, DVec2::new(256.0, 256.0), StrokeParams { opacity: 1.0, erase: true, ..Default::default() });
    assert!(composite_px(&e, c, 2, 2)[3] < 0.01, "erased");

    assert_eq!(e.undo().as_deref(), Some("Pinselstrich"));
    e.flush();
    assert!(composite_px(&e, c, 2, 2)[3] > 0.99, "undo restores");
    e.redo();
    e.flush();
    assert!(composite_px(&e, c, 2, 2)[3] < 0.01, "redo erases again");

    // Erasing where nothing exists allocates nothing.
    let before = e.layer_tiles_live();
    stroke(&mut e, DVec2::new(900.0, 900.0), StrokeParams { opacity: 1.0, erase: true, ..Default::default() });
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
    stroke(&mut e, DVec2::new(100.0, 100.0), StrokeParams { opacity: 0.5, erase: false, ..Default::default() });
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
    let stats = e.render(&tv, &view, &Overlay { cursor: Some((DVec2::splat(10.0), 5.0)), checker_px: 8.0, ..Default::default() });
    assert_eq!(stats.visible_tiles, 600);
    view.set_zoom_at(view.viewport * 0.5, 8.0);
    assert!(e.render(&tv, &view, &Overlay::default()).visible_tiles <= 4);
    g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

fn alpha_at(e: &mut Engine, x: i64, y: i64) -> f32 {
    e.flush();
    let c = TileCoord::containing(x as f64, y as f64);
    let (ox, oy) = c.origin();
    composite_px(e, c, (x - ox) as usize, (y - oy) as usize)[3]
}

#[test]
fn selection_clips_strokes_fills_and_clears() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1000, 600));
    // Select the left half, then paint a stroke across the whole width.
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 500.0, h: 600.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.begin_stroke(StrokeParams::default()).unwrap();
    let mut s = Stroke::new(red_brush());
    let mut dabs = s.add(InputSample { pos: DVec2::new(100.0, 300.0), pressure: 1.0 });
    dabs.extend(s.add(InputSample { pos: DVec2::new(900.0, 300.0), pressure: 1.0 }));
    dabs.extend(s.finish(InputSample { pos: DVec2::new(900.0, 300.0), pressure: 1.0 }));
    e.stroke_dabs(&dabs);
    // Live preview already respects the selection …
    assert!(alpha_at(&mut e, 300, 300) > 0.99);
    assert!(alpha_at(&mut e, 700, 300) < 0.01);
    e.end_stroke("Pinselstrich");
    // … and so does the committed result.
    assert!(alpha_at(&mut e, 300, 300) > 0.99);
    assert!(alpha_at(&mut e, 700, 300) < 0.01);

    // Fill an ellipse added to the selection, with 50 % opacity.
    e.select_shape(&Shape::Ellipse { cx: 800.0, cy: 300.0, rx: 100.0, ry: 100.0 }, CombineOp::Add, "Auswahl").unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 0.5, "Füllen").unwrap();
    assert!((alpha_at(&mut e, 800, 300) - 0.5).abs() < 0.01);
    assert!(alpha_at(&mut e, 950, 50) < 0.01, "outside selection untouched");

    // Invert and clear: only the unselected part remains … i.e. clearing the inverse removes nothing selected.
    e.invert_selection("Umkehren").unwrap();
    e.clear_selected("Löschen").unwrap();
    assert!(alpha_at(&mut e, 300, 300) > 0.99, "inside original selection kept");
    e.deselect("Auswahl aufheben").unwrap();
    assert!(!e.has_selection());

    // Undo walks back through all of it.
    for _ in 0..3 {
        e.undo();
    }
    assert!(e.has_selection());
}

#[test]
fn gradient_and_feather() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 64));
    e.gradient(
        StrokeTarget::Pixels,
        GradientKind::Linear,
        (0.0, 0.0),
        (512.0, 0.0),
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        1.0,
        "Verlauf",
    )
    .unwrap();
    let a0 = alpha_at(&mut e, 2, 10);
    let a1 = alpha_at(&mut e, 256, 10);
    let a2 = alpha_at(&mut e, 509, 10);
    assert!(a0 > 0.98 && (a1 - 0.5).abs() < 0.02 && a2 < 0.02, "{a0} {a1} {a2}");

    // Feathered selection gives a soft fill edge.
    e.select_shape(&Shape::Rect { x: 100.0, y: 0.0, w: 300.0, h: 64.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.feather_selection(20.0, "Weiche Kante").unwrap();
    let m = e.selection_cpu().unwrap();
    let edge = m.value_at(100, 32);
    assert!((90..=165).contains(&edge), "feathered edge ≈ 50 %: {edge}");
}

#[test]
fn layer_mask_paint_toggle_and_apply() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Füllen").unwrap();
    // Mask from a selection: only the left 200 px stay visible.
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 200.0, h: 300.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.add_layer_mask("Maske").unwrap();
    e.deselect("Auswahl aufheben").unwrap();
    assert!(alpha_at(&mut e, 100, 100) > 0.99);
    assert!(alpha_at(&mut e, 400, 100) < 0.01);

    // Paint white into the mask at x = 400 → revealed there.
    e.begin_stroke(StrokeParams { target: StrokeTarget::Mask, ..Default::default() }).unwrap();
    let mut s = Stroke::new(BrushSettings { color: [1.0; 3], ..red_brush() });
    e.stroke_dabs(&s.add(InputSample { pos: DVec2::new(400.0, 100.0), pressure: 1.0 }));
    assert!(alpha_at(&mut e, 400, 100) > 0.99, "live mask preview");
    e.end_stroke("Maske malen");
    assert!(alpha_at(&mut e, 400, 100) > 0.99);
    assert!(alpha_at(&mut e, 500, 100) < 0.01);

    // Disable the mask → everything visible; enable and bake it.
    let id = e.doc().active;
    e.edit("Maske aus", |d| {
        d.find_mut(id).unwrap().mask_enabled = false;
        true
    });
    assert!(alpha_at(&mut e, 500, 100) > 0.99);
    e.undo();
    e.apply_layer_mask("Maske anwenden").unwrap();
    assert!(e.doc().active_layer().unwrap().mask.is_none());
    assert!(alpha_at(&mut e, 100, 100) > 0.99);
    assert!(alpha_at(&mut e, 400, 100) > 0.99);
    assert!(alpha_at(&mut e, 500, 100) < 0.01);
}

#[test]
fn translate_moves_pixels_and_mask() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1000, 1000));
    e.select_shape(&Shape::Rect { x: 10.0, y: 10.0, w: 100.0, h: 100.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 1.0, 0.0, 1.0], 1.0, "Füllen").unwrap();
    e.deselect("Auswahl aufheben").unwrap();
    e.add_layer_mask("Maske").unwrap(); // reveal all
    e.translate_active(300, 250, "Verschieben").unwrap();
    assert!(alpha_at(&mut e, 50, 50) < 0.01, "old place empty");
    assert!(alpha_at(&mut e, 350, 300) > 0.99, "moved by (300, 250)");
    assert!(alpha_at(&mut e, 309, 259) < 0.01 && alpha_at(&mut e, 310, 260) > 0.99, "exact edge");
    // Content moved off the canvas is kept and comes back.
    e.translate_active(-2000, 0, "Verschieben").unwrap();
    e.flush();
    assert!(e.composite_coords().is_empty());
    e.translate_active(2000, 0, "Zurück").unwrap();
    assert!(alpha_at(&mut e, 350, 300) > 0.99, "nothing was cut off");
    e.undo();
    e.undo();
    assert!(alpha_at(&mut e, 350, 300) > 0.99);
}

#[test]
fn transform_scale_rotate_float_and_bounds() {
    use glam::DAffine2;
    use imagine_gpu::Interp;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1200, 1000));
    e.select_shape(&Shape::Rect { x: 100.0, y: 100.0, w: 200.0, h: 100.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Füllen").unwrap();
    e.deselect("Auswahl aufheben").unwrap();
    let id = e.doc().active;
    assert_eq!(e.content_bounds(id), Some((100, 100, 300, 200)));

    // Scale ×2 around (100, 100): covers 100..500 × 100..300.
    let fwd = DAffine2::from_translation(DVec2::splat(100.0))
        * DAffine2::from_scale(DVec2::splat(2.0))
        * DAffine2::from_translation(DVec2::splat(-100.0));
    e.transform_active(fwd, Interp::Bicubic, "Skalieren").unwrap();
    assert!(alpha_at(&mut e, 450, 250) > 0.99);
    assert!(alpha_at(&mut e, 510, 250) < 0.01);
    // Interpolation softens the edge by about a pixel (bicubic corners may
    // leave faint ringing one pixel further out).
    assert!(alpha_at(&mut e, 98, 150) < 0.01 && (alpha_at(&mut e, 99, 150) - 0.2).abs() < 0.01);
    let b = e.content_bounds(id).unwrap();
    assert!((b.0 - 99).abs() <= 1 && (b.1 - 99).abs() <= 1 && (b.2 - 501).abs() <= 1 && (b.3 - 301).abs() <= 1, "{b:?}");
    e.undo();

    // Rotate 90° around the rectangle centre (200, 150): becomes 150..250 × 50..250.
    let c = DVec2::new(200.0, 150.0);
    let fwd = DAffine2::from_translation(c)
        * DAffine2::from_angle(std::f64::consts::FRAC_PI_2)
        * DAffine2::from_translation(-c);
    e.transform_active(fwd, Interp::Bilinear, "Drehen").unwrap();
    assert!(alpha_at(&mut e, 200, 60) > 0.99);
    assert!(alpha_at(&mut e, 120, 150) < 0.01);
    let b = e.content_bounds(id).unwrap();
    assert!((b.0 - 150).abs() <= 1 && (b.1 - 50).abs() <= 1 && (b.2 - 250).abs() <= 1 && (b.3 - 250).abs() <= 1, "{b:?}");
    e.undo();

    // Downscale ×0.25 stays smooth (supersampled): a half-covered pixel is ≈ 50 %.
    let fwd = DAffine2::from_scale(DVec2::splat(0.25));
    e.transform_active(fwd, Interp::Bicubic, "Verkleinern").unwrap();
    assert!(alpha_at(&mut e, 40, 40) > 0.99);
    assert!(alpha_at(&mut e, 80, 60) < 0.01);
    e.undo();

    // Floating selection: only the selected left half moves 300 px right,
    // the selection outline moves along.
    e.select_shape(&Shape::Rect { x: 100.0, y: 100.0, w: 100.0, h: 100.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.transform_active(DAffine2::from_translation(DVec2::new(300.0, 0.0)), Interp::Bicubic, "Verschieben").unwrap();
    assert!(alpha_at(&mut e, 150, 150) < 0.01, "hole left behind");
    assert!(alpha_at(&mut e, 250, 150) > 0.99, "unselected half stays");
    assert!(alpha_at(&mut e, 450, 150) > 0.99, "selected half moved");
    let sel = e.selection_cpu().unwrap();
    assert_eq!(sel.value_at(450, 150), 255);
    assert_eq!(sel.value_at(150, 150), 0);

    // Preview shows the result without touching the document or history.
    let before = e.history_len();
    e.deselect("Auswahl aufheben").unwrap();
    let before_px = alpha_at(&mut e, 700, 150);
    e.set_transform_preview(Some((DAffine2::from_translation(DVec2::new(300.0, 0.0)), Interp::Bicubic))).unwrap();
    let mut view = View { viewport: DVec2::new(1200.0, 1000.0), ..Default::default() };
    view.fit(1200, 1000);
    let tex = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 1200, height: 1000, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    e.render(&tex.create_view(&Default::default()), &view, &Overlay::default());
    e.set_transform_preview(Some((DAffine2::from_translation(DVec2::new(300.0, 0.0)), Interp::Bicubic))).unwrap();
    assert!(alpha_at(&mut e, 750, 150) > 0.99, "preview visible");
    assert_eq!(e.history_len().0, before.0 + 1);
    e.set_transform_preview(None).unwrap();
    assert!((alpha_at(&mut e, 700, 150) - before_px).abs() < 1e-3, "preview gone");
}

#[test]
fn renders_selection_overlay_and_lines() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(800, 600));
    e.select_shape(&Shape::Ellipse { cx: 400.0, cy: 300.0, rx: 200.0, ry: 150.0 }, CombineOp::Replace, "Auswahl").unwrap();
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 320, height: 240, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let tv = target.create_view(&Default::default());
    let mut view = View { viewport: DVec2::new(320.0, 240.0), ..Default::default() };
    view.fit(800, 600);
    let overlay = Overlay {
        show_selection: true,
        quick_mask: true,
        time: 1.5,
        lines: vec![OverlayLine { points: vec![DVec2::ZERO, DVec2::new(100.0, 50.0), DVec2::new(30.0, 200.0)], closed: true }],
        ..Default::default()
    };
    e.render(&tv, &view, &overlay);
    g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

#[test]
fn crop_extend_and_resize_keep_content() {
    use imagine_gpu::Interp;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1000, 800));
    e.select_shape(&Shape::Rect { x: 600.0, y: 500.0, w: 100.0, h: 100.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 1.0, "Füllen").unwrap();
    e.add_layer_mask("Maske").unwrap();
    e.deselect("Auswahl aufheben").unwrap();

    // Off-grid crop: the square moves to (100, 150).
    e.crop(500, 350, 300, 300, "Zuschneiden").unwrap();
    assert_eq!(e.size(), (300, 300));
    assert!(alpha_at(&mut e, 150, 200) > 0.99);
    assert!(alpha_at(&mut e, 90, 200) < 0.01);
    // Extending the canvas back brings the cropped-away area back (nothing deleted).
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 10.0, h: 10.0 }, CombineOp::Replace, "x").unwrap();
    e.crop(-500, -350, 1000, 800, "Arbeitsfläche").unwrap();
    assert!(!e.has_selection(), "crop drops the selection");
    assert!(alpha_at(&mut e, 650, 550) > 0.99);
    assert!(alpha_at(&mut e, 590, 550) < 0.01);

    // Tile-aligned crop just renumbers tiles.
    let live = e.layer_tiles_live();
    e.crop(512, 256, 488, 544, "Zuschneiden").unwrap();
    assert_eq!(e.layer_tiles_live(), live, "no new tiles");
    assert!(alpha_at(&mut e, 650 - 512, 550 - 256) > 0.99);
    e.undo();

    // Image size ×0.5: square at 300..350 × 250..300, mask scaled along.
    e.resize_image(500, 400, Interp::Bicubic, "Bildgröße").unwrap();
    assert_eq!(e.size(), (500, 400));
    assert!(alpha_at(&mut e, 325, 275) > 0.99);
    assert!(alpha_at(&mut e, 290, 275) < 0.01);
    e.undo();
    assert_eq!(e.size(), (1000, 800));
    assert!(alpha_at(&mut e, 650, 550) > 0.99);
}

#[test]
fn magic_wand_and_color_range() {
    use imagine_gpu::SampleFrom;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(700, 400));
    // White background, two red squares (one far right), a blue bar between.
    e.fill(StrokeTarget::Pixels, [1.0, 1.0, 1.0, 1.0], 1.0, "Weiß").unwrap();
    for x in [50.0, 500.0] {
        e.select_shape(&Shape::Rect { x, y: 50.0, w: 100.0, h: 100.0 }, CombineOp::Replace, "Auswahl").unwrap();
        e.fill(StrokeTarget::Pixels, [0.8, 0.0, 0.0, 1.0], 1.0, "Rot").unwrap();
    }
    e.deselect("Auswahl aufheben").unwrap();

    // Contiguous: only the clicked square (exact edges with antialiasing off).
    e.magic_wand(60, 60, 32.0, true, false, SampleFrom::Layer, CombineOp::Replace, "Zauberstab").unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(100, 100), 255);
    assert_eq!(s.value_at(149, 149), 255);
    assert_eq!(s.value_at(150, 100), 0);
    assert_eq!(s.value_at(550, 100), 0, "other square not connected");

    // Shift-click (add) the second one, non-contiguous selects both anyway.
    e.magic_wand(510, 60, 32.0, true, true, SampleFrom::Layer, CombineOp::Add, "Zauberstab").unwrap();
    assert_eq!(e.selection_cpu().unwrap().value_at(550, 100), 255);
    e.magic_wand(10, 10, 10.0, false, false, SampleFrom::Image, CombineOp::Replace, "Zauberstab").unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(300, 300), 255, "white everywhere");
    assert_eq!(s.value_at(100, 100), 0, "red excluded");

    // Colour range by colour: the red squares, soft edge by fuzziness.
    let red = e.sample_straight(100, 100, SampleFrom::Layer).unwrap();
    e.select_color_range(red, 40.0, SampleFrom::Layer, CombineOp::Replace, "Farbbereich").unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(100, 100), 255);
    assert_eq!(s.value_at(550, 100), 255);
    assert_eq!(s.value_at(300, 300), 0);

    // Clicking a transparent area of an empty layer selects all of it.
    e.edit("Neue Ebene", |d| {
        d.add_pixel_layer();
        true
    });
    e.magic_wand(10, 10, 0.0, true, false, SampleFrom::Layer, CombineOp::Replace, "Zauberstab").unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(699, 399), 255);
}

#[test]
fn textured_tips_and_smudge() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    // Chalk tip: grainy, so coverage inside the dab varies.
    e.begin_stroke(StrokeParams::default()).unwrap();
    let mut s = Stroke::new(BrushSettings { size: 120.0, tip: 1, spacing: 5.0, ..red_brush() });
    e.stroke_dabs(&s.add(InputSample { pos: DVec2::new(100.0, 100.0), pressure: 1.0 }));
    e.end_stroke("Kreide");
    e.flush();
    let tile = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let alphas: Vec<f32> = (60..140).flat_map(|y| (60..140).map(move |x| (x, y))).map(|(x, y)| px(&tile, x, y)[3]).collect();
    let full = alphas.iter().filter(|a| **a > 0.95).count();
    let empty = alphas.iter().filter(|a| **a < 0.05).count();
    assert!(full > 300 && empty > 300, "grain: {full} full, {empty} empty of {}", alphas.len());
    e.undo();

    // Smudge: red left half, blue right half; drag from red into blue.
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 300.0, h: 300.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Rot").unwrap();
    e.invert_selection("Umkehren").unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 1.0, "Blau").unwrap();
    e.deselect("Auswahl aufheben").unwrap();
    e.flush();
    let before = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 150);
    assert!(before[0] < 0.01 && before[2] > 0.99);

    e.begin_stroke(StrokeParams { smudge: true, ..Default::default() }).unwrap();
    let mut s = Stroke::new(BrushSettings { size: 40.0, hardness: 0.9, smudge: Some(0.8), spacing: 0.1, ..red_brush() });
    let mut dabs = s.add(InputSample { pos: DVec2::new(250.0, 150.0), pressure: 1.0 });
    for i in 1..=30 {
        dabs.extend(s.add(InputSample { pos: DVec2::new(250.0 + i as f64 * 5.0, 150.0), pressure: 1.0 }));
    }
    e.stroke_dabs(&dabs);
    e.flush();
    let live = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 150);
    assert!(live[0] > 0.2, "red dragged into blue (live): {live:?}");
    e.end_stroke("Verwischen");
    e.flush();
    let after = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 150);
    assert!((after[0] - live[0]).abs() < 0.02 && (after[3] - 1.0).abs() < 0.01, "{after:?} vs {live:?}");
    let far = composite_px(&e, TileCoord::new(1, 0), 500 - 256, 150);
    assert!(far[2] > 0.99 && far[0] < 0.01, "untouched beyond the stroke");
    let outside = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 250);
    assert!(outside[2] > 0.99, "untouched outside the brush");
    e.undo();
    e.flush();
    let undone = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 150);
    assert!(undone[0] < 0.01);
}

#[test]
fn adjustment_layers_match_cpu_reference() {
    use imagine_core::Adjustment;
    use imagine_core::adjust::Levels;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let base = [0.6f32, 0.25, 0.05];
    e.fill(StrokeTarget::Pixels, [base[0], base[1], base[2], 1.0], 1.0, "Füllen").unwrap();
    let mut cases = Adjustment::defaults();
    cases.extend([
        Adjustment::BrightnessContrast { brightness: 60.0, contrast: 40.0 },
        Adjustment::Levels {
            master: Levels { in_black: 0.1, in_white: 0.9, gamma: 1.4, ..Default::default() },
            red: Levels { out_white: 0.8, ..Default::default() },
            green: Levels::default(),
            blue: Levels { out_black: 0.2, ..Default::default() },
        },
        Adjustment::Curves {
            master: vec![[0.0, 0.0], [0.3, 0.2], [0.7, 0.85], [1.0, 1.0]],
            red: vec![[0.0, 0.1], [1.0, 0.9]],
            green: vec![[0.0, 0.0], [1.0, 1.0]],
            blue: vec![[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]],
        },
        Adjustment::HueSaturation { hue: 70.0, saturation: 40.0, lightness: -20.0, colorize: false },
        Adjustment::HueSaturation { hue: 200.0, saturation: -30.0, lightness: 10.0, colorize: true },
        Adjustment::Vibrance { vibrance: 60.0, saturation: -10.0 },
        Adjustment::Exposure { exposure: 0.7, offset: 0.02, gamma: 1.2 },
        Adjustment::Threshold { level: 90.0 },
    ]);
    for adj in cases {
        let want = adj.apply(base);
        let name = adj.name();
        e.edit("Einstellung", |d| {
            d.add_adjustment(adj.clone());
            true
        });
        e.flush();
        let got = composite_px(&e, TileCoord::new(0, 0), 10, 10);
        let ok = (0..3).all(|i| (got[i] - want[i]).abs() < 0.004 + want[i] * 0.01);
        assert!(ok, "{name}: GPU {got:?} vs CPU {want:?}");
        // Half opacity mixes with what is below.
        e.edit("Deckkraft", |d| {
            let id = d.active;
            d.find_mut(id).unwrap().opacity = 0.5;
            true
        });
        e.flush();
        let half = composite_px(&e, TileCoord::new(0, 0), 10, 10);
        assert!((half[0] - (want[0] + base[0]) / 2.0).abs() < 0.01, "{name} at 50 %");
        e.undo();
        e.undo();
    }
}

#[test]
fn filters_blur_sharpen_noise_and_preview() {
    use imagine_core::Filter;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1024, 512));
    // White right half (x ≥ 512) on transparent.
    e.select_shape(&Shape::Rect { x: 512.0, y: 0.0, w: 512.0, h: 512.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.fill(StrokeTarget::Pixels, [1.0; 4], 1.0, "Weiß").unwrap();
    e.deselect("x").unwrap();
    // Gaussian step response: 0.5 at the edge, ≈ 0.16 one sigma outside.
    for sigma in [4.0f32, 40.0] {
        e.apply_filter(&Filter::GaussianBlur { radius: sigma }, "Weichzeichnen").unwrap();
        // Φ(z) via the Abramowitz–Stegun erf approximation.
        #[allow(clippy::excessive_precision)]
        let phi = |z: f32| {
            let x = z.abs() / std::f32::consts::SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * x);
            let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
            0.5 * (1.0 + y.copysign(z))
        };
        // Pixel x covers [x, x+1); the edge is at 512.
        let expect = |x: i64| phi(((x as f32 + 0.5) - 512.0) / sigma);
        for x in [511, 512, 511 - sigma as i64, 512 + sigma as i64] {
            let got = alpha_at(&mut e, x, 256);
            assert!((got - expect(x)).abs() < 0.03, "sigma {sigma} x {x}: {got} vs {}", expect(x));
        }
        assert!(alpha_at(&mut e, 512 - 4 * sigma as i64, 256) < 0.01);
        e.undo();
    }
    // Unsharp mask overshoots next to the edge (inside brighter … capped by alpha).
    e.edit("grau", |_| false);
    let grey = imagine_core::color::srgb_to_linear(0.5);
    e.fill(StrokeTarget::Pixels, [grey, grey, grey, 1.0], 1.0, "Grau").unwrap();
    e.select_shape(&Shape::Rect { x: 512.0, y: 0.0, w: 512.0, h: 512.0 }, CombineOp::Replace, "Auswahl").unwrap();
    e.fill(StrokeTarget::Pixels, [0.8, 0.8, 0.8, 1.0], 1.0, "Hell").unwrap();
    e.deselect("x").unwrap();
    e.apply_filter(&Filter::UnsharpMask { amount: 150.0, radius: 3.0, threshold: 0.0 }, "Scharf").unwrap();
    e.flush();
    let inside = composite_px(&e, TileCoord::new(2, 0), 2, 100)[0];
    let outside = composite_px(&e, TileCoord::new(1, 0), 253, 100)[0];
    assert!(inside > 0.85 && outside < grey - 0.05, "{inside} {outside}");
    e.undo();
    // Noise changes pixels but keeps the mean.
    e.apply_filter(&Filter::AddNoise { amount: 20.0, monochrome: true, gaussian: false, seed: 1 }, "Rauschen").unwrap();
    e.flush();
    let tile = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let vals: Vec<f32> = (0..256).map(|x| px(&tile, x, 7)[0]).collect();
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(vals.iter().any(|v| (v - grey).abs() > 0.02));
    assert!((mean - grey).abs() < 0.04, "mean {mean}");
    e.undo();
    // Preview leaves the history alone.
    let before = e.history_len();
    let mut view = View { viewport: DVec2::new(1024.0, 512.0), ..Default::default() };
    view.fit(1024, 512);
    let tex = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 1024, height: 512, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    e.render(&tex.create_view(&Default::default()), &view, &Overlay::default());
    e.set_filter_preview(Some(&Filter::GaussianBlur { radius: 20.0 })).unwrap();
    e.flush();
    let blurred = composite_px(&e, TileCoord::new(1, 0), 255, 100)[0];
    assert!(blurred > grey + 0.05, "preview visible: {blurred}");
    assert_eq!(e.history_len(), before);
    e.set_filter_preview(None).unwrap();
}
