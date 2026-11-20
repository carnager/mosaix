//! Headless GPU tests for the engine. Skipped (with a message) when no
//! suitable Vulkan adapter exists.

use glam::DVec2;
use half::f16;
use imagine_core::blend::composite;
use imagine_core::mask::{CombineOp, Shape};
use imagine_core::{
    BlendMode, BlendSpace, BrushSettings, Document, InputSample, Stroke, TILE_SIZE, TileCoord, View,
};
use imagine_gpu::{
    Engine, GpuContext, GradientKind, Overlay, OverlayLine, StrokeParams, StrokeTarget,
};

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
            let a = if opaque {
                1.0
            } else {
                0.2 + 0.8 * (((x * 7 + y * 3 + seed) % 5) as f32 / 4.0)
            };
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
    BrushSettings {
        size: 20.0,
        hardness: 1.0,
        flow: 1.0,
        smoothing: 0.0,
        color: [1.0, 0.0, 0.0],
        ..Default::default()
    }
}

fn stroke(e: &mut Engine, at: DVec2, params: StrokeParams) {
    e.begin_stroke(params).unwrap();
    let mut s = Stroke::new(red_brush());
    let mut dabs = s.add(InputSample {
        pos: at,
        pressure: 1.0,
    });
    for i in 1..20 {
        dabs.extend(s.add(InputSample {
            pos: at + DVec2::new(i as f64 * 0.1, 0.0),
            pressure: 1.0,
        }));
    }
    e.stroke_dabs(&dabs);
    e.end_stroke("Brush stroke");
    e.flush();
}

fn composite_px(e: &Engine, c: TileCoord, x: usize, y: usize) -> [f32; 4] {
    px(&e.read_composite(&[c])[0], x, y)
}

#[test]
fn paint_undo_redo_and_sparsity() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1000, 1000));
    let paint = StrokeParams {
        opacity: 1.0,
        erase: false,
        ..Default::default()
    };
    stroke(&mut e, DVec2::new(256.0, 256.0), paint);
    assert_eq!(
        e.layer_tiles_live(),
        4,
        "corner stroke touches exactly 4 tiles"
    );
    assert_eq!(e.composite_coords().len(), 4);
    let c = TileCoord::new(1, 1);
    let p = composite_px(&e, c, 2, 2);
    assert!(p[0] > 0.99 && p[3] > 0.99 && p[1] < 0.01, "red: {p:?}");

    stroke(
        &mut e,
        DVec2::new(256.0, 256.0),
        StrokeParams {
            opacity: 1.0,
            erase: true,
            ..Default::default()
        },
    );
    assert!(composite_px(&e, c, 2, 2)[3] < 0.01, "erased");

    assert_eq!(e.undo().as_deref(), Some("Brush stroke"));
    e.flush();
    assert!(composite_px(&e, c, 2, 2)[3] > 0.99, "undo restores");
    e.redo();
    e.flush();
    assert!(composite_px(&e, c, 2, 2)[3] < 0.01, "redo erases again");

    // Erasing where nothing exists allocates nothing.
    let before = e.layer_tiles_live();
    stroke(
        &mut e,
        DVec2::new(900.0, 900.0),
        StrokeParams {
            opacity: 1.0,
            erase: true,
            ..Default::default()
        },
    );
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
    stroke(
        &mut e,
        DVec2::new(100.0, 100.0),
        StrokeParams {
            opacity: 0.5,
            erase: false,
            ..Default::default()
        },
    );
    let p = composite_px(&e, TileCoord::new(0, 0), 100, 100);
    assert!(
        (p[3] - 0.5).abs() < 0.01,
        "alpha capped at stroke opacity: {p:?}"
    );
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
    e.edit("New layer", |d| {
        top = d.add_pixel_layer();
        true
    });
    e.upload_tile(top, TileCoord::new(0, 0), &top_px);

    let samples = [(0, 0), (3, 7), (31, 200), (128, 128), (255, 255), (77, 13)];
    let mut worst = 0.0f32;
    for space in [BlendSpace::Srgb, BlendSpace::Linear] {
        for mode in BlendMode::ALL {
            e.edit("Mode", |d| {
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
                let expected = composite(
                    composite([0.0; 4], b, BlendMode::Normal, space),
                    t,
                    mode,
                    space,
                );
                let got = px(&out, x, y);
                for k in 0..4 {
                    let err = (got[k] - expected[k]).abs();
                    worst = worst.max(err);
                    assert!(
                        err < 4e-3,
                        "{mode:?}/{space:?} at ({x},{y}): gpu {got:?} cpu {expected:?}"
                    );
                }
            }
        }
    }
    eprintln!("worst GPU/CPU deviation: {worst:.2e}");

    // Groups: isolated composite, then group opacity/blend.
    e.edit("Group", |d| {
        d.blend_space = BlendSpace::Srgb;
        d.find_mut(top).unwrap().blend = BlendMode::Screen;
        d.find_mut(top).unwrap().opacity = 1.0;
        d.active = top;
        let gid = d.group_active().unwrap();
        let grp = d.find_mut(gid).unwrap();
        grp.blend = BlendMode::Multiply;
        grp.pass_through = false; // a blend mode makes the group isolated
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
        assert!(
            got.iter().zip(expected).all(|(a, b)| (a - b).abs() < 4e-3),
            "group: {got:?} vs {expected:?}"
        );
    }
}

#[test]
fn many_tiles_span_chunks_and_duplicates_share() {
    let Some(g) = gpu() else { return };
    let (gw, gh) = (30, 20); // 600 tiles > one 256-layer chunk
    let mut e = Engine::new(
        &g.device,
        &g.queue,
        FMT,
        &Document::<()>::new(gw * TILE_SIZE, gh * TILE_SIZE),
    );
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
        assert!(
            (got - expected).abs() < 1e-3,
            "tile ({x},{y}): {got} vs {expected}"
        );
    }

    let live = e.layer_tiles_live();
    e.edit("Duplicate", |d| d.duplicate_active().is_some());
    assert_eq!(e.layer_tiles_live(), live, "duplicate shares tiles");

    // Hiding the copy and the original leaves nothing to composite.
    e.edit("Hide", |d| {
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
        size: wgpu::Extent3d {
            width: 640,
            height: 480,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let tv = target.create_view(&Default::default());
    let mut view = View {
        viewport: DVec2::new(640.0, 480.0),
        ..Default::default()
    };
    let (w, h) = e.size();
    view.fit(w, h);
    let stats = e.render(
        &tv,
        &view,
        &Overlay {
            cursor: Some((DVec2::splat(10.0), 5.0)),
            checker_px: 8.0,
            ..Default::default()
        },
    );
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
    e.select_shape(
        &Shape::Rect {
            x: 0.0,
            y: 0.0,
            w: 500.0,
            h: 600.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.begin_stroke(StrokeParams::default()).unwrap();
    let mut s = Stroke::new(red_brush());
    let mut dabs = s.add(InputSample {
        pos: DVec2::new(100.0, 300.0),
        pressure: 1.0,
    });
    dabs.extend(s.add(InputSample {
        pos: DVec2::new(900.0, 300.0),
        pressure: 1.0,
    }));
    dabs.extend(s.finish(InputSample {
        pos: DVec2::new(900.0, 300.0),
        pressure: 1.0,
    }));
    e.stroke_dabs(&dabs);
    // Live preview already respects the selection …
    assert!(alpha_at(&mut e, 300, 300) > 0.99);
    assert!(alpha_at(&mut e, 700, 300) < 0.01);
    e.end_stroke("Brush stroke");
    // … and so does the committed result.
    assert!(alpha_at(&mut e, 300, 300) > 0.99);
    assert!(alpha_at(&mut e, 700, 300) < 0.01);

    // Fill an ellipse added to the selection, with 50 % opacity.
    e.select_shape(
        &Shape::Ellipse {
            cx: 800.0,
            cy: 300.0,
            rx: 100.0,
            ry: 100.0,
        },
        CombineOp::Add,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 0.5, "Fill")
        .unwrap();
    assert!((alpha_at(&mut e, 800, 300) - 0.5).abs() < 0.01);
    assert!(
        alpha_at(&mut e, 950, 50) < 0.01,
        "outside selection untouched"
    );

    // Invert and clear: only the unselected part remains … i.e. clearing the inverse removes nothing selected.
    e.invert_selection("Invert").unwrap();
    e.clear_selected("Delete").unwrap();
    assert!(
        alpha_at(&mut e, 300, 300) > 0.99,
        "inside original selection kept"
    );
    e.deselect("Deselect").unwrap();
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
        "Gradient",
    )
    .unwrap();
    let a0 = alpha_at(&mut e, 2, 10);
    let a1 = alpha_at(&mut e, 256, 10);
    let a2 = alpha_at(&mut e, 509, 10);
    assert!(
        a0 > 0.98 && (a1 - 0.5).abs() < 0.02 && a2 < 0.02,
        "{a0} {a1} {a2}"
    );

    // Feathered selection gives a soft fill edge.
    e.select_shape(
        &Shape::Rect {
            x: 100.0,
            y: 0.0,
            w: 300.0,
            h: 64.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.feather_selection(20.0, "Feather").unwrap();
    let m = e.selection_cpu().unwrap();
    let edge = m.value_at(100, 32);
    assert!((90..=165).contains(&edge), "feathered edge ≈ 50 %: {edge}");
}

#[test]
fn layer_mask_paint_toggle_and_apply() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Fill")
        .unwrap();
    // Mask from a selection: only the left 200 px stay visible.
    e.select_shape(
        &Shape::Rect {
            x: 0.0,
            y: 0.0,
            w: 200.0,
            h: 300.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.add_layer_mask("Mask").unwrap();
    e.deselect("Deselect").unwrap();
    assert!(alpha_at(&mut e, 100, 100) > 0.99);
    assert!(alpha_at(&mut e, 400, 100) < 0.01);

    // Paint white into the mask at x = 400 → revealed there.
    e.begin_stroke(StrokeParams {
        target: StrokeTarget::Mask,
        ..Default::default()
    })
    .unwrap();
    let mut s = Stroke::new(BrushSettings {
        color: [1.0; 3],
        ..red_brush()
    });
    e.stroke_dabs(&s.add(InputSample {
        pos: DVec2::new(400.0, 100.0),
        pressure: 1.0,
    }));
    assert!(alpha_at(&mut e, 400, 100) > 0.99, "live mask preview");
    e.end_stroke("Paint mask");
    assert!(alpha_at(&mut e, 400, 100) > 0.99);
    assert!(alpha_at(&mut e, 500, 100) < 0.01);

    // Disable the mask → everything visible; enable and bake it.
    let id = e.doc().active;
    e.edit("Mask off", |d| {
        d.find_mut(id).unwrap().mask_enabled = false;
        true
    });
    assert!(alpha_at(&mut e, 500, 100) > 0.99);
    e.undo();
    e.apply_layer_mask("Apply mask").unwrap();
    assert!(e.doc().active_layer().unwrap().mask.is_none());
    assert!(alpha_at(&mut e, 100, 100) > 0.99);
    assert!(alpha_at(&mut e, 400, 100) > 0.99);
    assert!(alpha_at(&mut e, 500, 100) < 0.01);
}

#[test]
fn translate_moves_pixels_and_mask() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1000, 1000));
    e.select_shape(
        &Shape::Rect {
            x: 10.0,
            y: 10.0,
            w: 100.0,
            h: 100.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 1.0, 0.0, 1.0], 1.0, "Fill")
        .unwrap();
    e.deselect("Deselect").unwrap();
    e.add_layer_mask("Mask").unwrap(); // reveal all
    e.translate_active(300, 250, "Move").unwrap();
    assert!(alpha_at(&mut e, 50, 50) < 0.01, "old place empty");
    assert!(alpha_at(&mut e, 350, 300) > 0.99, "moved by (300, 250)");
    assert!(
        alpha_at(&mut e, 309, 259) < 0.01 && alpha_at(&mut e, 310, 260) > 0.99,
        "exact edge"
    );
    // Content moved off the canvas is kept and comes back.
    e.translate_active(-2000, 0, "Move").unwrap();
    e.flush();
    assert!(e.composite_coords().is_empty());
    e.translate_active(2000, 0, "Back").unwrap();
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
    e.select_shape(
        &Shape::Rect {
            x: 100.0,
            y: 100.0,
            w: 200.0,
            h: 100.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Fill")
        .unwrap();
    e.deselect("Deselect").unwrap();
    let id = e.doc().active;
    assert_eq!(e.content_bounds(id), Some((100, 100, 300, 200)));

    // Scale ×2 around (100, 100): covers 100..500 × 100..300.
    let fwd = DAffine2::from_translation(DVec2::splat(100.0))
        * DAffine2::from_scale(DVec2::splat(2.0))
        * DAffine2::from_translation(DVec2::splat(-100.0));
    e.transform_active(fwd, Interp::Bicubic, "Scale").unwrap();
    assert!(alpha_at(&mut e, 450, 250) > 0.99);
    assert!(alpha_at(&mut e, 510, 250) < 0.01);
    // Interpolation softens the edge by about a pixel (bicubic corners may
    // leave faint ringing one pixel further out).
    assert!(alpha_at(&mut e, 98, 150) < 0.01 && (alpha_at(&mut e, 99, 150) - 0.2).abs() < 0.01);
    let b = e.content_bounds(id).unwrap();
    assert!(
        (b.0 - 99).abs() <= 1
            && (b.1 - 99).abs() <= 1
            && (b.2 - 501).abs() <= 1
            && (b.3 - 301).abs() <= 1,
        "{b:?}"
    );
    e.undo();

    // Rotate 90° around the rectangle centre (200, 150): becomes 150..250 × 50..250.
    let c = DVec2::new(200.0, 150.0);
    let fwd = DAffine2::from_translation(c)
        * DAffine2::from_angle(std::f64::consts::FRAC_PI_2)
        * DAffine2::from_translation(-c);
    e.transform_active(fwd, Interp::Bilinear, "Rotate").unwrap();
    assert!(alpha_at(&mut e, 200, 60) > 0.99);
    assert!(alpha_at(&mut e, 120, 150) < 0.01);
    let b = e.content_bounds(id).unwrap();
    assert!(
        (b.0 - 150).abs() <= 1
            && (b.1 - 50).abs() <= 1
            && (b.2 - 250).abs() <= 1
            && (b.3 - 250).abs() <= 1,
        "{b:?}"
    );
    e.undo();

    // Downscale ×0.25 stays smooth (supersampled): a half-covered pixel is ≈ 50 %.
    let fwd = DAffine2::from_scale(DVec2::splat(0.25));
    e.transform_active(fwd, Interp::Bicubic, "Shrink").unwrap();
    assert!(alpha_at(&mut e, 40, 40) > 0.99);
    assert!(alpha_at(&mut e, 80, 60) < 0.01);
    e.undo();

    // Floating selection: only the selected left half moves 300 px right,
    // the selection outline moves along.
    e.select_shape(
        &Shape::Rect {
            x: 100.0,
            y: 100.0,
            w: 100.0,
            h: 100.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.transform_active(
        DAffine2::from_translation(DVec2::new(300.0, 0.0)),
        Interp::Bicubic,
        "Move",
    )
    .unwrap();
    assert!(alpha_at(&mut e, 150, 150) < 0.01, "hole left behind");
    assert!(alpha_at(&mut e, 250, 150) > 0.99, "unselected half stays");
    assert!(alpha_at(&mut e, 450, 150) > 0.99, "selected half moved");
    let sel = e.selection_cpu().unwrap();
    assert_eq!(sel.value_at(450, 150), 255);
    assert_eq!(sel.value_at(150, 150), 0);

    // Preview shows the result without touching the document or history.
    let before = e.history_len();
    e.deselect("Deselect").unwrap();
    let before_px = alpha_at(&mut e, 700, 150);
    e.set_transform_preview(Some((
        DAffine2::from_translation(DVec2::new(300.0, 0.0)),
        Interp::Bicubic,
    )))
    .unwrap();
    let mut view = View {
        viewport: DVec2::new(1200.0, 1000.0),
        ..Default::default()
    };
    view.fit(1200, 1000);
    let tex = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1200,
            height: 1000,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    e.render(
        &tex.create_view(&Default::default()),
        &view,
        &Overlay::default(),
    );
    e.set_transform_preview(Some((
        DAffine2::from_translation(DVec2::new(300.0, 0.0)),
        Interp::Bicubic,
    )))
    .unwrap();
    assert!(alpha_at(&mut e, 750, 150) > 0.99, "preview visible");
    assert_eq!(e.history_len().0, before.0 + 1);
    e.set_transform_preview(None).unwrap();
    assert!(
        (alpha_at(&mut e, 700, 150) - before_px).abs() < 1e-3,
        "preview gone"
    );
}

#[test]
fn renders_selection_overlay_and_lines() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(800, 600));
    e.select_shape(
        &Shape::Ellipse {
            cx: 400.0,
            cy: 300.0,
            rx: 200.0,
            ry: 150.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    let target = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 320,
            height: 240,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let tv = target.create_view(&Default::default());
    let mut view = View {
        viewport: DVec2::new(320.0, 240.0),
        ..Default::default()
    };
    view.fit(800, 600);
    let overlay = Overlay {
        show_selection: true,
        quick_mask: true,
        time: 1.5,
        lines: vec![OverlayLine {
            points: vec![
                DVec2::ZERO,
                DVec2::new(100.0, 50.0),
                DVec2::new(30.0, 200.0),
            ],
            closed: true,
        }],
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
    e.select_shape(
        &Shape::Rect {
            x: 600.0,
            y: 500.0,
            w: 100.0,
            h: 100.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 1.0, "Fill")
        .unwrap();
    e.add_layer_mask("Mask").unwrap();
    e.deselect("Deselect").unwrap();

    // Off-grid crop: the square moves to (100, 150).
    e.crop(500, 350, 300, 300, "Crop").unwrap();
    assert_eq!(e.size(), (300, 300));
    assert!(alpha_at(&mut e, 150, 200) > 0.99);
    assert!(alpha_at(&mut e, 90, 200) < 0.01);
    // Extending the canvas back brings the cropped-away area back (nothing deleted).
    e.select_shape(
        &Shape::Rect {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
        },
        CombineOp::Replace,
        "x",
    )
    .unwrap();
    e.crop(-500, -350, 1000, 800, "Canvas Size").unwrap();
    assert!(!e.has_selection(), "crop drops the selection");
    assert!(alpha_at(&mut e, 650, 550) > 0.99);
    assert!(alpha_at(&mut e, 590, 550) < 0.01);

    // Tile-aligned crop just renumbers tiles.
    let live = e.layer_tiles_live();
    e.crop(512, 256, 488, 544, "Crop").unwrap();
    assert_eq!(e.layer_tiles_live(), live, "no new tiles");
    assert!(alpha_at(&mut e, 650 - 512, 550 - 256) > 0.99);
    e.undo();

    // Image size ×0.5: square at 300..350 × 250..300, mask scaled along.
    e.resize_image(500, 400, Interp::Bicubic, "Image Size")
        .unwrap();
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
    e.fill(StrokeTarget::Pixels, [1.0, 1.0, 1.0, 1.0], 1.0, "White")
        .unwrap();
    for x in [50.0, 500.0] {
        e.select_shape(
            &Shape::Rect {
                x,
                y: 50.0,
                w: 100.0,
                h: 100.0,
            },
            CombineOp::Replace,
            "Selection",
        )
        .unwrap();
        e.fill(StrokeTarget::Pixels, [0.8, 0.0, 0.0, 1.0], 1.0, "Red")
            .unwrap();
    }
    e.deselect("Deselect").unwrap();

    // Contiguous: only the clicked square (exact edges with antialiasing off).
    e.magic_wand(
        60,
        60,
        32.0,
        true,
        false,
        SampleFrom::Layer,
        CombineOp::Replace,
        "Magic Wand",
    )
    .unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(100, 100), 255);
    assert_eq!(s.value_at(149, 149), 255);
    assert_eq!(s.value_at(150, 100), 0);
    assert_eq!(s.value_at(550, 100), 0, "other square not connected");

    // Shift-click (add) the second one, non-contiguous selects both anyway.
    e.magic_wand(
        510,
        60,
        32.0,
        true,
        true,
        SampleFrom::Layer,
        CombineOp::Add,
        "Magic Wand",
    )
    .unwrap();
    assert_eq!(e.selection_cpu().unwrap().value_at(550, 100), 255);
    e.magic_wand(
        10,
        10,
        10.0,
        false,
        false,
        SampleFrom::Image,
        CombineOp::Replace,
        "Magic Wand",
    )
    .unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(300, 300), 255, "white everywhere");
    assert_eq!(s.value_at(100, 100), 0, "red excluded");

    // Colour range by colour: the red squares, soft edge by fuzziness.
    let red = e.sample_straight(100, 100, SampleFrom::Layer).unwrap();
    e.select_color_range(
        red,
        40.0,
        SampleFrom::Layer,
        CombineOp::Replace,
        "Color Range",
    )
    .unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(100, 100), 255);
    assert_eq!(s.value_at(550, 100), 255);
    assert_eq!(s.value_at(300, 300), 0);

    // Clicking a transparent area of an empty layer selects all of it.
    e.edit("New layer", |d| {
        d.add_pixel_layer();
        true
    });
    e.magic_wand(
        10,
        10,
        0.0,
        true,
        false,
        SampleFrom::Layer,
        CombineOp::Replace,
        "Magic Wand",
    )
    .unwrap();
    let s = e.selection_cpu().unwrap();
    assert_eq!(s.value_at(699, 399), 255);
}

#[test]
fn textured_tips_and_smudge() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    // Chalk tip: grainy, so coverage inside the dab varies.
    e.begin_stroke(StrokeParams::default()).unwrap();
    let mut s = Stroke::new(BrushSettings {
        size: 120.0,
        tip: 1,
        spacing: 5.0,
        ..red_brush()
    });
    e.stroke_dabs(&s.add(InputSample {
        pos: DVec2::new(100.0, 100.0),
        pressure: 1.0,
    }));
    e.end_stroke("Chalk");
    e.flush();
    let tile = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let alphas: Vec<f32> = (60..140)
        .flat_map(|y| (60..140).map(move |x| (x, y)))
        .map(|(x, y)| px(&tile, x, y)[3])
        .collect();
    let full = alphas.iter().filter(|a| **a > 0.95).count();
    let empty = alphas.iter().filter(|a| **a < 0.05).count();
    assert!(
        full > 300 && empty > 300,
        "grain: {full} full, {empty} empty of {}",
        alphas.len()
    );
    e.undo();

    // Smudge: red left half, blue right half; drag from red into blue.
    e.select_shape(
        &Shape::Rect {
            x: 0.0,
            y: 0.0,
            w: 300.0,
            h: 300.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Red")
        .unwrap();
    e.invert_selection("Invert").unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 1.0, "Blue")
        .unwrap();
    e.deselect("Deselect").unwrap();
    e.flush();
    let before = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 150);
    assert!(before[0] < 0.01 && before[2] > 0.99);

    e.begin_stroke(StrokeParams {
        smudge: true,
        ..Default::default()
    })
    .unwrap();
    let mut s = Stroke::new(BrushSettings {
        size: 40.0,
        hardness: 0.9,
        smudge: Some(0.8),
        spacing: 0.1,
        ..red_brush()
    });
    let mut dabs = s.add(InputSample {
        pos: DVec2::new(250.0, 150.0),
        pressure: 1.0,
    });
    for i in 1..=30 {
        dabs.extend(s.add(InputSample {
            pos: DVec2::new(250.0 + i as f64 * 5.0, 150.0),
            pressure: 1.0,
        }));
    }
    e.stroke_dabs(&dabs);
    e.flush();
    let live = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 150);
    assert!(live[0] > 0.2, "red dragged into blue (live): {live:?}");
    e.end_stroke("Smudge");
    e.flush();
    let after = composite_px(&e, TileCoord::new(1, 0), 340 - 256, 150);
    assert!(
        (after[0] - live[0]).abs() < 0.02 && (after[3] - 1.0).abs() < 0.01,
        "{after:?} vs {live:?}"
    );
    let far = composite_px(&e, TileCoord::new(1, 0), 500 - 256, 150);
    assert!(
        far[2] > 0.99 && far[0] < 0.01,
        "untouched beyond the stroke"
    );
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
    e.fill(
        StrokeTarget::Pixels,
        [base[0], base[1], base[2], 1.0],
        1.0,
        "Fill",
    )
    .unwrap();
    let mut cases = Adjustment::defaults();
    cases.extend([
        Adjustment::BrightnessContrast {
            brightness: 60.0,
            contrast: 40.0,
            legacy: false,
        },
        Adjustment::BrightnessContrast {
            brightness: -30.0,
            contrast: 60.0,
            legacy: true,
        },
        Adjustment::Levels {
            master: Levels {
                in_black: 0.1,
                in_white: 0.9,
                gamma: 1.4,
                ..Default::default()
            },
            red: Levels {
                out_white: 0.8,
                ..Default::default()
            },
            green: Levels::default(),
            blue: Levels {
                out_black: 0.2,
                ..Default::default()
            },
        },
        Adjustment::Curves {
            master: vec![[0.0, 0.0], [0.3, 0.2], [0.7, 0.85], [1.0, 1.0]],
            red: vec![[0.0, 0.1], [1.0, 0.9]],
            green: vec![[0.0, 0.0], [1.0, 1.0]],
            blue: vec![[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]],
        },
        Adjustment::HueSaturation {
            hue: 70.0,
            saturation: 40.0,
            lightness: -20.0,
            colorize: false,
        },
        Adjustment::HueSaturation {
            hue: 200.0,
            saturation: -30.0,
            lightness: 10.0,
            colorize: true,
        },
        Adjustment::Vibrance {
            vibrance: 60.0,
            saturation: -10.0,
        },
        Adjustment::Exposure {
            exposure: 0.7,
            offset: 0.02,
            gamma: 1.2,
        },
        Adjustment::Threshold { level: 90.0 },
    ]);
    for adj in cases {
        let want = adj.apply(base);
        let name = adj.name();
        e.edit("Adjustment", |d| {
            d.add_adjustment(adj.clone());
            true
        });
        e.flush();
        let got = composite_px(&e, TileCoord::new(0, 0), 10, 10);
        let ok = (0..3).all(|i| (got[i] - want[i]).abs() < 0.004 + want[i] * 0.01);
        assert!(ok, "{name}: GPU {got:?} vs CPU {want:?}");
        // Half opacity mixes with what is below.
        e.edit("Opacity", |d| {
            let id = d.active;
            d.find_mut(id).unwrap().opacity = 0.5;
            true
        });
        e.flush();
        let half = composite_px(&e, TileCoord::new(0, 0), 10, 10);
        assert!(
            (half[0] - (want[0] + base[0]) / 2.0).abs() < 0.01,
            "{name} at 50 %"
        );
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
    e.select_shape(
        &Shape::Rect {
            x: 512.0,
            y: 0.0,
            w: 512.0,
            h: 512.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0; 4], 1.0, "White")
        .unwrap();
    e.deselect("x").unwrap();
    // Gaussian step response: 0.5 at the edge, ≈ 0.16 one sigma outside.
    for sigma in [4.0f32, 40.0] {
        e.apply_filter(&Filter::GaussianBlur { radius: sigma }, "Blur")
            .unwrap();
        // Φ(z) via the Abramowitz–Stegun erf approximation.
        #[allow(clippy::excessive_precision)]
        let phi = |z: f32| {
            let x = z.abs() / std::f32::consts::SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * x);
            let y = 1.0
                - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                    + 0.254829592)
                    * t
                    * (-x * x).exp();
            0.5 * (1.0 + y.copysign(z))
        };
        // Pixel x covers [x, x+1); the edge is at 512.
        let expect = |x: i64| phi(((x as f32 + 0.5) - 512.0) / sigma);
        for x in [511, 512, 511 - sigma as i64, 512 + sigma as i64] {
            let got = alpha_at(&mut e, x, 256);
            assert!(
                (got - expect(x)).abs() < 0.03,
                "sigma {sigma} x {x}: {got} vs {}",
                expect(x)
            );
        }
        assert!(alpha_at(&mut e, 512 - 4 * sigma as i64, 256) < 0.01);
        e.undo();
    }
    // Unsharp mask overshoots next to the edge (inside brighter … capped by alpha).
    e.edit("gray", |_| false);
    let grey = imagine_core::color::srgb_to_linear(0.5);
    e.fill(StrokeTarget::Pixels, [grey, grey, grey, 1.0], 1.0, "Gray")
        .unwrap();
    e.select_shape(
        &Shape::Rect {
            x: 512.0,
            y: 0.0,
            w: 512.0,
            h: 512.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [0.8, 0.8, 0.8, 1.0], 1.0, "Light")
        .unwrap();
    e.deselect("x").unwrap();
    e.apply_filter(
        &Filter::UnsharpMask {
            amount: 150.0,
            radius: 3.0,
            threshold: 0.0,
        },
        "Sharpen",
    )
    .unwrap();
    e.flush();
    let inside = composite_px(&e, TileCoord::new(2, 0), 2, 100)[0];
    let outside = composite_px(&e, TileCoord::new(1, 0), 253, 100)[0];
    assert!(inside > 0.85 && outside < grey - 0.05, "{inside} {outside}");
    e.undo();
    // Noise changes pixels but keeps the mean.
    e.apply_filter(
        &Filter::AddNoise {
            amount: 20.0,
            monochrome: true,
            gaussian: false,
            seed: 1,
        },
        "Noise",
    )
    .unwrap();
    e.flush();
    let tile = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let vals: Vec<f32> = (0..256).map(|x| px(&tile, x, 7)[0]).collect();
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(vals.iter().any(|v| (v - grey).abs() > 0.02));
    assert!((mean - grey).abs() < 0.04, "mean {mean}");
    e.undo();
    // Preview leaves the history alone.
    let before = e.history_len();
    let mut view = View {
        viewport: DVec2::new(1024.0, 512.0),
        ..Default::default()
    };
    view.fit(1024, 512);
    let tex = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1024,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    e.render(
        &tex.create_view(&Default::default()),
        &view,
        &Overlay::default(),
    );
    e.set_filter_preview(Some(&Filter::GaussianBlur { radius: 20.0 }))
        .unwrap();
    e.flush();
    let blurred = composite_px(&e, TileCoord::new(1, 0), 255, 100)[0];
    assert!(blurred > grey + 0.05, "preview visible: {blurred}");
    assert_eq!(e.history_len(), before);
    e.set_filter_preview(None).unwrap();
}

#[test]
fn histogram_of_what_is_below() {
    use imagine_core::Adjustment;
    use imagine_core::color::linear_to_srgb;
    let Some(g) = gpu() else { return };
    // Not a multiple of the tile size: pixels past the edge must not count.
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    let base = [0.6f32, 0.25, 0.05];
    e.fill(
        StrokeTarget::Pixels,
        [base[0], base[1], base[2], 1.0],
        1.0,
        "Fill",
    )
    .unwrap();
    let bin = |v: f32| (linear_to_srgb(v) * 255.0).round() as usize;
    let h = e.histogram(None);
    let total = 600.0 * 300.0;
    for c in 0..3 {
        assert!(
            (h.bins[c + 1][bin(base[c])] - total).abs() < 1.0,
            "channel {c}: {:?}",
            h.bins[c + 1].iter().sum::<f32>()
        );
    }
    assert!((h.bins[0].iter().sum::<f32>() - 3.0 * total).abs() < 3.0);
    e.edit("Invert", |d| {
        d.add_adjustment(Adjustment::Invert);
        true
    });
    let adj = e.doc().active;
    let below = e.histogram(Some(adj));
    assert_eq!(*below, *h, "the adjustment itself is not part of its input");
    let all = e.histogram(None);
    assert!(
        (all.bins[1][255 - bin(base[0])] - total).abs() < 1.0,
        "inverted red"
    );
}

#[test]
#[ignore = "benchmark"]
fn histogram_speed_100mp() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(
        &g.device,
        &g.queue,
        FMT,
        &Document::<()>::new(10_000, 10_000),
    );
    e.fill(StrokeTarget::Pixels, [0.3, 0.5, 0.1, 1.0], 1.0, "Fill")
        .unwrap();
    e.flush();
    let t = std::time::Instant::now();
    let h = e.histogram(None);
    eprintln!("histogram 100 MP: {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let _ = e.histogram(None);
    eprintln!(
        "cached: {:?} ({} px)",
        t.elapsed(),
        h.bins[1].iter().sum::<f32>()
    );
}

#[test]
fn adjustment_blend_modes_match_layer_semantics() {
    use imagine_core::Adjustment;
    let Some(g) = gpu() else { return };
    let base = [0.6f32, 0.25, 0.05];
    for space in [BlendSpace::Srgb, BlendSpace::Linear] {
        let mut doc = Document::<()>::new(256, 256);
        doc.blend_space = space;
        let mut e = Engine::new(&g.device, &g.queue, FMT, &doc);
        e.fill(
            StrokeTarget::Pixels,
            [base[0], base[1], base[2], 1.0],
            1.0,
            "Fill",
        )
        .unwrap();
        let adj = Adjustment::HueSaturation {
            hue: 120.0,
            saturation: 30.0,
            lightness: 0.0,
            colorize: false,
        };
        let a = adj.apply(base);
        e.edit("Adjustment", |d| {
            let id = d.add_adjustment(adj.clone());
            d.find_mut(id).unwrap().opacity = 0.7;
            true
        });
        for mode in [
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Luminosity,
            BlendMode::Color,
            BlendMode::Overlay,
        ] {
            e.edit("Mode", |d| {
                let id = d.active;
                d.find_mut(id).unwrap().blend = mode;
                true
            });
            e.flush();
            let got = composite_px(&e, TileCoord::new(0, 0), 5, 5);
            // A layer holding the adjusted image at 70 %.
            let want = composite(
                [base[0], base[1], base[2], 1.0],
                [a[0] * 0.7, a[1] * 0.7, a[2] * 0.7, 0.7],
                mode,
                space,
            );
            let ok = (0..4).all(|i| (got[i] - want[i]).abs() < 0.004 + want[i] * 0.01);
            assert!(ok, "{mode:?} in {space:?}: GPU {got:?} vs CPU {want:?}");
        }
    }
}

#[test]
fn smart_filters_are_live_toggle_and_bake() {
    use imagine_core::Filter;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1024, 512));
    e.select_shape(
        &Shape::Rect {
            x: 512.0,
            y: 0.0,
            w: 512.0,
            h: 512.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0; 4], 1.0, "White")
        .unwrap();
    e.deselect("x").unwrap();
    let layer = e.doc().active;
    let edge = |e: &mut Engine| {
        e.flush();
        [508, 511, 512, 516]
            .map(|x| composite_px(e, TileCoord::new(x / 256, 0), x as usize % 256, 100)[3])
    };
    let raw = edge(&mut e);
    assert_eq!(raw, [0.0, 0.0, 1.0, 1.0]);
    let blur = Filter::GaussianBlur { radius: 4.0 };
    // Reference: the destructive filter.
    e.apply_filter(&blur, "Blur").unwrap();
    let want = edge(&mut e);
    e.undo();
    assert_eq!(edge(&mut e), raw);

    e.add_smart_filter(&blur, "Smart-Filter").unwrap();
    let got = edge(&mut e);
    assert!(
        got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-3),
        "{got:?} vs {want:?}"
    );
    // The layer's own pixels are untouched.
    assert_eq!(e.doc().find(layer).unwrap().tiles().unwrap().len(), 4);
    // Switched off → raw; on again → filtered (from the cache).
    e.edit("Off", |d| {
        d.find_mut(layer).unwrap().filters[0].enabled = false;
        true
    });
    assert_eq!(edge(&mut e), raw);
    e.undo();
    assert_eq!(edge(&mut e), got);
    // Undo removes it, redo brings it back.
    e.undo();
    assert_eq!(edge(&mut e), raw);
    e.redo();
    assert_eq!(edge(&mut e), got);
    // Baking writes the result into the pixels.
    e.bake_smart_filters(layer, "Anwenden").unwrap();
    let l = e.doc().find(layer).unwrap();
    assert!(
        l.filters.is_empty() && l.tiles().unwrap().len() > 4,
        "blur spreads into new tiles"
    );
    assert_eq!(edge(&mut e), got);
    // Preview of a smart filter shows it without changing the document.
    e.undo();
    e.set_smart_filter_preview(Some((Some(0), &Filter::GaussianBlur { radius: 8.0 })))
        .unwrap();
    let wider = edge(&mut e);
    assert!(wider[1] > got[1] + 0.02, "{wider:?} vs {got:?}");
    assert_eq!(e.doc().find(layer).unwrap().filters.len(), 1);
    e.set_smart_filter_preview(None).unwrap();
    assert_eq!(edge(&mut e), got);
    e.set_smart_filter(layer, 0, &Filter::GaussianBlur { radius: 8.0 }, "Change")
        .unwrap();
    assert_eq!(edge(&mut e), wider);
    assert!(e.set_smart_filter(layer, 3, &blur, "x").is_err());
}

#[test]
fn liquify_push_reconstruct_undo_and_commit() {
    use imagine_gpu::LiquifyTool;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    // A white vertical stripe x ∈ [200, 260).
    e.select_shape(
        &Shape::Rect {
            x: 200.0,
            y: 0.0,
            w: 60.0,
            h: 512.0,
        },
        CombineOp::Replace,
        "Selection",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0; 4], 1.0, "White")
        .unwrap();
    e.deselect("x").unwrap();
    let a = |e: &mut Engine, x: i64, y: i64| {
        e.flush();
        composite_px(
            e,
            TileCoord::new((x / 256) as i32, (y / 256) as i32),
            (x % 256) as usize,
            (y % 256) as usize,
        )[3]
    };
    assert_eq!((a(&mut e, 230, 300), a(&mut e, 280, 300)), (1.0, 0.0));
    let steps = e.history_len().0;

    // Push right by 40 px at (230, 300).
    e.liquify_begin().unwrap();
    e.liquify_stroke_start();
    for i in 0..8 {
        e.liquify_dab(
            LiquifyTool::Push,
            DVec2::new(230.0 + i as f64 * 5.0, 300.0),
            DVec2::new(5.0, 0.0),
            100.0,
            1.0,
            1.0,
        );
    }
    e.liquify_tick().unwrap();
    e.liquify_commit("Liquify").unwrap();
    assert!(!e.liquify_active());
    assert_eq!(e.history_len().0, steps + 1);
    assert!(
        a(&mut e, 280, 300) > 0.9,
        "stripe pushed right: {}",
        a(&mut e, 280, 300)
    );
    assert!(
        a(&mut e, 205, 300) < 0.1,
        "left edge pushed away: {}",
        a(&mut e, 205, 300)
    );
    // Far from the brush nothing changes (exact copy).
    assert_eq!((a(&mut e, 230, 20), a(&mut e, 280, 20)), (1.0, 0.0));
    e.undo();
    assert_eq!(a(&mut e, 280, 300), 0.0);

    // Undo inside the session and reconstruct restore the original.
    for tool in ["undo", "reconstruct"] {
        e.liquify_begin().unwrap();
        e.liquify_stroke_start();
        e.liquify_dab(
            LiquifyTool::Push,
            DVec2::new(230.0, 300.0),
            DVec2::new(40.0, 0.0),
            100.0,
            1.0,
            1.0,
        );
        if tool == "undo" {
            assert!(e.liquify_undo());
        } else {
            for _ in 0..200 {
                e.liquify_dab(
                    LiquifyTool::Reconstruct,
                    DVec2::new(230.0, 300.0),
                    DVec2::ZERO,
                    150.0,
                    1.0,
                    1.0,
                );
            }
        }
        e.liquify_commit("Liquify").unwrap();
        assert!(
            (a(&mut e, 230, 300) - 1.0).abs() < 1e-3 && a(&mut e, 280, 300) < 1e-3,
            "{tool}"
        );
        e.undo();
    }

    // Twirl and bloat move pixels; cancel leaves the layer as it was.
    e.liquify_begin().unwrap();
    for _ in 0..30 {
        e.liquify_dab(
            LiquifyTool::TwirlCw,
            DVec2::new(230.0, 300.0),
            DVec2::ZERO,
            120.0,
            1.0,
            1.0,
        );
        e.liquify_dab(
            LiquifyTool::Bloat,
            DVec2::new(230.0, 300.0),
            DVec2::ZERO,
            120.0,
            1.0,
            1.0,
        );
    }
    e.liquify_tick().unwrap();
    e.liquify_cancel();
    assert_eq!((a(&mut e, 230, 300), a(&mut e, 280, 300)), (1.0, 0.0));
    assert_eq!(e.history_len().0, steps);
}

#[test]
#[ignore = "benchmark"]
fn liquify_speed_100mp() {
    use imagine_gpu::LiquifyTool;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(
        &g.device,
        &g.queue,
        FMT,
        &Document::<()>::new(10_000, 10_000),
    );
    e.fill(StrokeTarget::Pixels, [0.3, 0.5, 0.1, 1.0], 1.0, "Fill")
        .unwrap();
    e.flush();
    let t = std::time::Instant::now();
    e.liquify_begin().unwrap();
    eprintln!("begin: {:?}", t.elapsed());
    let t = std::time::Instant::now();
    for i in 0..100 {
        e.liquify_dab(
            LiquifyTool::Push,
            DVec2::new(5000.0 + i as f64 * 4.0, 5000.0),
            DVec2::new(4.0, 0.0),
            300.0,
            0.8,
            1.0,
        );
    }
    eprintln!("100 dabs r=300: {:?}", t.elapsed());
    let t = std::time::Instant::now();
    e.liquify_commit("x").unwrap();
    e.flush();
    let _ = e.read_composite(&[TileCoord::new(20, 20)]);
    eprintln!("commit + flush: {:?}", t.elapsed());
}

#[test]
fn clone_stamp_copies_exactly_from_layer_and_image() {
    use imagine_gpu::CloneSource;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    // Source: a horizontal ramp in x ∈ [0, 120) on the background.
    for x in 0..120 {
        let v = x as f32 / 120.0;
        e.select_shape(
            &Shape::Rect {
                x: x as f64,
                y: 0.0,
                w: 1.0,
                h: 300.0,
            },
            CombineOp::Replace,
            "A",
        )
        .unwrap();
        e.fill(StrokeTarget::Pixels, [v, 0.5, 1.0 - v, 1.0], 1.0, "F")
            .unwrap();
    }
    e.deselect("x").unwrap();
    e.flush();
    let before = |e: &Engine, x: usize, y: usize| composite_px(e, TileCoord::new(0, 0), x, y);
    let want = before(&e, 60, 150);
    let params = |all| StrokeParams {
        clone: Some(CloneSource {
            offset: [-300, 0],
            all_layers: all,
            heal: false,
            spot: false,
        }),
        ..Default::default()
    };
    // Stamp at x = 360 → copies x = 60 exactly (hard brush centre).
    stroke(&mut e, DVec2::new(360.0, 150.0), params(false));
    let got = composite_px(&e, TileCoord::new(1, 0), 360 - 256, 150);
    assert_eq!(got, want, "exact copy from the layer");
    // Outside the brush nothing changes.
    assert_eq!(
        composite_px(&e, TileCoord::new(1, 0), 400 - 256, 150)[3],
        0.0
    );
    e.undo();
    // From all layers onto a new, empty layer.
    e.edit("Layer", |d| {
        d.add_pixel_layer();
        true
    });
    assert!(e.begin_stroke(params(false)).is_ok());
    e.end_stroke("x");
    e.flush();
    stroke(&mut e, DVec2::new(360.0, 150.0), params(true));
    let got = composite_px(&e, TileCoord::new(1, 0), 360 - 256, 150);
    assert_eq!(got, want, "exact copy from the image");
    let layer = e.doc().active;
    assert!(
        e.doc()
            .find(layer)
            .unwrap()
            .tiles()
            .unwrap()
            .contains_key(&TileCoord::new(1, 0)),
        "painted into the new layer"
    );
}

#[test]
fn healing_matches_the_surroundings() {
    use imagine_gpu::CloneSource;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    // A horizontal brightness ramp, plus a dark blemish at (400, 150).
    for x in 0..600 {
        let v = 0.1 + 0.8 * x as f32 / 600.0;
        e.select_shape(
            &Shape::Rect {
                x: x as f64,
                y: 0.0,
                w: 1.0,
                h: 300.0,
            },
            CombineOp::Replace,
            "A",
        )
        .unwrap();
        e.fill(StrokeTarget::Pixels, [v, v, v, 1.0], 1.0, "F")
            .unwrap();
    }
    e.select_shape(
        &Shape::Ellipse {
            cx: 400.0,
            cy: 150.0,
            rx: 6.0,
            ry: 6.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 0.0, 1.0], 1.0, "Spot")
        .unwrap();
    e.deselect("x").unwrap();
    e.flush();
    let ramp = |x: usize| 0.1 + 0.8 * x as f32 / 600.0;
    let at = |e: &Engine, x: usize| {
        composite_px(e, TileCoord::new((x / 256) as i32, 0), x % 256, 150)[0]
    };
    assert!(at(&e, 400) < 0.01);
    // Source 250 px to the left: much darker, but clean.
    let heal = StrokeParams {
        clone: Some(CloneSource {
            offset: [-250, 0],
            all_layers: false,
            heal: true,
            spot: false,
        }),
        ..Default::default()
    };
    stroke(&mut e, DVec2::new(400.0, 150.0), heal);
    for x in [395, 400, 405] {
        let (got, want) = (at(&e, x), ramp(x));
        assert!(
            (got - want).abs() < 0.01,
            "x = {x}: healed {got}, surroundings {want}"
        );
    }
    // A plain clone would have brought the darker source along.
    e.undo();
    let clone = StrokeParams {
        clone: Some(CloneSource {
            offset: [-250, 0],
            all_layers: false,
            heal: false,
            spot: false,
        }),
        ..Default::default()
    };
    stroke(&mut e, DVec2::new(400.0, 150.0), clone);
    assert!((at(&e, 400) - ramp(150)).abs() < 0.01);
}

#[test]
fn spot_healing_finds_a_matching_source() {
    use imagine_gpu::CloneSource;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    // Vertical stripes (8 px light, 8 px dark) on a vertical brightness ramp.
    let want = |x: usize, y: usize| {
        0.2 + 0.5 * y as f32 / 512.0 + if (x / 8).is_multiple_of(2) { 0.15 } else { 0.0 }
    };
    let layer = e.doc().active;
    let ts = TILE_SIZE as usize;
    let tiles: Vec<(TileCoord, Option<Vec<f16>>)> = (0..2)
        .flat_map(|ty| (0..2).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| {
            let mut px = vec![f16::ONE; ts * ts * 4];
            for y in 0..ts {
                for x in 0..ts {
                    let v = f16::from_f32(want(tx * ts + x, ty * ts + y));
                    px[(y * ts + x) * 4..][..3].fill(v);
                }
            }
            (TileCoord::new(tx as i32, ty as i32), Some(px))
        })
        .collect();
    assert!(e.replace_tiles(layer, tiles, "Muster"));
    e.select_shape(
        &Shape::Ellipse {
            cx: 260.0,
            cy: 250.0,
            rx: 10.0,
            ry: 10.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Spot")
        .unwrap();
    e.deselect("x").unwrap();
    e.flush();
    let at = |e: &Engine, x: usize, y: usize| {
        composite_px(
            e,
            TileCoord::new((x / 256) as i32, (y / 256) as i32),
            x % 256,
            y % 256,
        )
    };
    assert_eq!(at(&e, 260, 250)[1], 0.0);
    let spot = StrokeParams {
        opacity: 0.5,
        clone: Some(CloneSource {
            offset: [0, 0],
            all_layers: false,
            heal: true,
            spot: true,
        }),
        ..Default::default()
    };
    e.begin_stroke(spot).unwrap();
    let mut s = Stroke::new(BrushSettings {
        size: 34.0,
        hardness: 1.0,
        flow: 1.0,
        smoothing: 0.0,
        color: [0.35; 3],
        ..Default::default()
    });
    let dabs = s.add(InputSample {
        pos: DVec2::new(260.0, 250.0),
        pressure: 1.0,
    });
    e.stroke_dabs(&dabs);
    e.end_stroke("Spot Healing");
    e.flush();
    // The stripes continue through the former blemish.
    let mut worst = 0.0f32;
    for y in [244, 250, 256] {
        for x in 252..268 {
            let p = at(&e, x, y);
            worst = worst
                .max((p[1] - want(x, y)).abs())
                .max((p[0] - want(x, y)).abs());
        }
    }
    assert!(worst < 0.03, "max deviation {worst}");
}

#[test]
fn content_aware_fill_continues_texture() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    let lin = imagine_core::color::srgb_to_linear;
    // Diagonal-free vertical stripes, 8 px wide, in two colours.
    let colour = |x: usize| {
        if (x / 8).is_multiple_of(2) {
            [lin(0.8), lin(0.3), lin(0.2)]
        } else {
            [lin(0.2), lin(0.3), lin(0.8)]
        }
    };
    let layer = e.doc().active;
    let ts = TILE_SIZE as usize;
    let tiles: Vec<(TileCoord, Option<Vec<f16>>)> = (0..2)
        .flat_map(|ty| (0..2).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| {
            let mut px = vec![f16::ONE; ts * ts * 4];
            for y in 0..ts {
                for x in 0..ts {
                    let c = colour(tx * ts + x);
                    for k in 0..3 {
                        px[(y * ts + x) * 4 + k] = f16::from_f32(c[k]);
                    }
                }
            }
            (TileCoord::new(tx as i32, ty), Some(px))
        })
        .collect();
    assert!(e.replace_tiles(layer, tiles, "Muster"));
    // Something to remove: a white blob, selected a bit larger.
    e.select_shape(
        &Shape::Ellipse {
            cx: 250.0,
            cy: 260.0,
            rx: 30.0,
            ry: 30.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 1.0, 1.0, 1.0], 1.0, "Object")
        .unwrap();
    e.select_shape(
        &Shape::Ellipse {
            cx: 250.0,
            cy: 260.0,
            rx: 36.0,
            ry: 36.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    let before = e.history_len().0;
    let ms = e.content_aware_fill(false, "Content-Aware Fill").unwrap();
    assert_eq!(e.history_len().0, before + 1);
    e.deselect("x").unwrap();
    e.flush();
    let at = |e: &Engine, x: usize, y: usize| {
        composite_px(
            e,
            TileCoord::new((x / 256) as i32, (y / 256) as i32),
            x % 256,
            y % 256,
        )
    };
    let (mut err, mut n) = (0.0, 0);
    for y in (232..290).step_by(3) {
        for x in 222..278 {
            if ((x as f32 - 250.0).hypot(y as f32 - 260.0)) < 30.0 {
                let p = at(&e, x, y);
                let c = colour(x);
                err += (0..3).map(|k| (p[k] - c[k]).abs()).sum::<f32>() / 3.0;
                n += 1;
            }
        }
    }
    let mean = err / n as f32;
    assert!(
        mean < 0.08,
        "mean deviation from the stripes {mean} ({ms:.0} ms)"
    );
    // The white blob is gone completely.
    assert!(at(&e, 250, 260)[1] < 0.5);
    // Without a selection it explains what to do.
    assert!(e.content_aware_fill(false, "x").is_err());
}

/// A layer of vertical stripes (8 px, two colours) over the whole document.
fn stripes_doc(e: &mut Engine, size: usize) -> impl Fn(usize) -> [f32; 3] + use<> {
    let lin = imagine_core::color::srgb_to_linear;
    let colour = move |x: usize| {
        if (x / 8).is_multiple_of(2) {
            [lin(0.8), lin(0.3), lin(0.2)]
        } else {
            [lin(0.2), lin(0.3), lin(0.8)]
        }
    };
    let layer = e.doc().active;
    let ts = TILE_SIZE as usize;
    let n = size.div_ceil(ts);
    let tiles: Vec<(TileCoord, Option<Vec<f16>>)> = (0..n)
        .flat_map(|ty| (0..n).map(move |tx| (tx, ty)))
        .map(|(tx, ty)| {
            let mut px = vec![f16::ONE; ts * ts * 4];
            for y in 0..ts {
                for x in 0..ts {
                    let c = colour(tx * ts + x);
                    for k in 0..3 {
                        px[(y * ts + x) * 4 + k] = f16::from_f32(c[k]);
                    }
                }
            }
            (TileCoord::new(tx as i32, ty as i32), Some(px))
        })
        .collect();
    assert!(e.replace_tiles(layer, tiles, "Muster"));
    colour
}

#[test]
fn content_aware_fill_gpu_levels() {
    let Some(g) = gpu() else { return };
    let mut means = Vec::new();
    // The CPU comparison is slow unoptimised; IMAGINE_FILL_COMPARE=1 runs it.
    let compare = std::env::var_os("IMAGINE_FILL_COMPARE").is_some();
    for cpu in [false, true].into_iter().take(if compare { 2 } else { 1 }) {
        let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1024, 1024));
        let colour = stripes_doc(&mut e, 1024);
        e.select_shape(
            &Shape::Ellipse {
                cx: 512.0,
                cy: 512.0,
                rx: 130.0,
                ry: 130.0,
            },
            CombineOp::Replace,
            "A",
        )
        .unwrap();
        e.fill(StrokeTarget::Pixels, [1.0, 1.0, 1.0, 1.0], 1.0, "Object")
            .unwrap();
        let job = e.fill_job(false).unwrap();
        let job = if cpu { job.cpu_only() } else { job };
        let r = job.run().unwrap();
        assert!(e.apply_fill(r, "Content-Aware Fill"));
        e.deselect("x").unwrap();
        e.flush();
        let at = |e: &Engine, x: usize, y: usize| {
            composite_px(
                e,
                TileCoord::new((x / 256) as i32, (y / 256) as i32),
                x % 256,
                y % 256,
            )
        };
        let (mut err, mut n) = (0.0, 0);
        for y in (400..624).step_by(7) {
            for x in 400..624 {
                if ((x as f32 - 512.0).hypot(y as f32 - 512.0)) < 125.0 {
                    let p = at(&e, x, y);
                    let c = colour(x);
                    err += (0..3).map(|k| (p[k] - c[k]).abs()).sum::<f32>() / 3.0;
                    n += 1;
                }
            }
        }
        means.push(err / n as f32);
    }
    // Perfect 8 px stripes vanish on the coarse levels, so the phase is not
    // always found; the CPU reaches ≈ 0.124 here. The GPU must be comparable.
    let cpu = means.get(1).copied().unwrap_or(0.124);
    eprintln!("mean deviation GPU {} CPU {cpu}", means[0]);
    assert!(means[0] < cpu * 1.2 + 0.01, "GPU {} vs CPU {cpu}", means[0]);
}

#[test]
#[ignore = "benchmark"]
fn content_aware_fill_speed() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(2048, 2048));
    let _ = stripes_doc(&mut e, 2048);
    for r in [150.0, 560.0] {
        e.select_shape(
            &Shape::Ellipse {
                cx: 1024.0,
                cy: 1024.0,
                rx: r,
                ry: r,
            },
            CombineOp::Replace,
            "A",
        )
        .unwrap();
        let mp = std::f64::consts::PI * r * r / 1e6;
        let t = std::time::Instant::now();
        let job = e.fill_job(false).unwrap();
        let read = t.elapsed();
        let gpu_ms = job.run().unwrap().ms;
        let cpu_ms = e.fill_job(false).unwrap().cpu_only().run().unwrap().ms;
        eprintln!("hole {mp:.2} MP: read {read:?}, GPU {gpu_ms:.0} ms, CPU {cpu_ms:.0} ms");
    }
}

#[test]
fn vector_layers_render_transform_and_rasterize() {
    use imagine_core::VectorLayer;
    use imagine_core::vector::{Geometry, ShapeSpec, TextSpec};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1024, 512));
    let red = [1.0, 0.0, 0.0, 1.0];
    let rect = VectorLayer::shape(ShapeSpec {
        geometry: Geometry::Rect {
            x: 300.0,
            y: 100.0,
            w: 200.0,
            h: 100.0,
            radius: 0.0,
        },
        fill: Some(red),
        stroke: None,
    });
    let id = e.add_vector(rect.clone(), "Rectangle").unwrap();
    e.flush();
    let at = |e: &Engine, x: usize, y: usize| {
        composite_px(
            e,
            TileCoord::new((x / 256) as i32, (y / 256) as i32),
            x % 256,
            y % 256,
        )
    };
    assert_eq!(at(&e, 350, 150), red);
    assert_eq!(at(&e, 520, 150)[3], 0.0);
    // Painting on it is refused with an explanation.
    assert!(e.begin_stroke(StrokeParams::default()).is_err());
    // Free transform stays a description: moved by 200 px.
    e.transform_active(
        glam::DAffine2::from_translation(DVec2::new(200.0, 0.0)),
        imagine_gpu::Interp::Bicubic,
        "T",
    )
    .unwrap();
    assert!(e.doc().find(id).unwrap().vector().is_some());
    e.flush();
    assert_eq!(at(&e, 350, 150)[3], 0.0);
    assert_eq!(at(&e, 650, 150), red);
    assert_eq!(
        e.content_bounds(id).map(|b| b.0 <= 500 && b.2 >= 700),
        Some(true)
    );
    // Editing the description re-renders; undo goes back.
    let mut green = rect.clone();
    if let imagine_core::vector::VectorKind::Shape(s) = &mut green.kind {
        s.fill = Some([0.0, 1.0, 0.0, 1.0]);
    }
    e.set_vector(id, green, "Farbe", None).unwrap();
    e.flush();
    assert_eq!(
        at(&e, 350, 150),
        [0.0, 1.0, 0.0, 1.0],
        "the edit replaced the transformed description"
    );
    e.undo();
    e.flush();
    assert_eq!(at(&e, 650, 150), red);
    // Text renders; rasterising keeps the look and makes it paintable.
    let text = e
        .add_vector(
            VectorLayer::text(TextSpec::new(
                "Imagine",
                40.0,
                300.0,
                120.0,
                [1.0, 1.0, 1.0, 1.0],
            )),
            "Text",
        )
        .unwrap();
    e.flush();
    let ink = (40..500).filter(|&x| at(&e, x, 370)[3] > 0.5).count();
    assert!(ink > 40, "text ink along a line: {ink}");
    e.rasterize(text, "Rastern").unwrap();
    e.flush();
    assert_eq!((40..500).filter(|&x| at(&e, x, 370)[3] > 0.5).count(), ink);
    assert!(
        e.doc()
            .find(text)
            .unwrap()
            .tiles()
            .is_some_and(|t| !t.is_empty())
    );
}

#[test]
fn rotate_and_flip_image_and_layer() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 300));
    // A single red pixel at (10, 20) and a text layer that must follow.
    e.select_shape(
        &Shape::Rect {
            x: 10.0,
            y: 20.0,
            w: 1.0,
            h: 1.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "F")
        .unwrap();
    e.deselect("x").unwrap();
    let a = |e: &mut Engine, x: usize, y: usize| {
        e.flush();
        composite_px(
            e,
            TileCoord::new((x / 256) as i32, (y / 256) as i32),
            x % 256,
            y % 256,
        )[0]
    };
    assert_eq!(a(&mut e, 10, 20), 1.0);
    // 90° clockwise: (x, y) → (H − 1 − y, x), canvas 300 × 600.
    e.rotate_image(1, false, false, "Rotate").unwrap();
    assert_eq!(e.size(), (300, 600));
    assert_eq!(a(&mut e, 300 - 1 - 20, 10), 1.0);
    // Back, then mirror left ↔ right: (x, y) → (W − 1 − x, y).
    e.rotate_image(3, false, false, "Rotate").unwrap();
    assert_eq!(e.size(), (600, 300));
    assert_eq!(a(&mut e, 10, 20), 1.0);
    e.rotate_image(0, true, false, "Spiegeln").unwrap();
    assert_eq!(a(&mut e, 600 - 1 - 10, 20), 1.0);
    assert_eq!(a(&mut e, 10, 20), 0.0);
    e.undo();
    // 180°: (x, y) → (W − 1 − x, H − 1 − y).
    e.rotate_image(2, false, false, "Rotate").unwrap();
    assert_eq!(a(&mut e, 600 - 1 - 10, 300 - 1 - 20), 1.0);
    e.undo();
    // Layer about its content centre (a 1 px dot stays put when turned).
    e.rotate_layer(90.0, false, false, "Rotate layer").unwrap();
    assert_eq!(a(&mut e, 10, 20), 1.0);
}

#[test]
fn layer_styles_stroke_and_shadow() {
    use imagine_core::VectorLayer;
    use imagine_core::style::{Effect, EffectKind, StrokePosition};
    use imagine_core::vector::{Geometry, ShapeSpec};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    e.fill(StrokeTarget::Pixels, [1.0, 1.0, 1.0, 1.0], 1.0, "White")
        .unwrap();
    let red = [1.0, 0.0, 0.0, 1.0];
    let id = e
        .add_vector(
            VectorLayer::shape(ShapeSpec {
                geometry: Geometry::Rect {
                    x: 100.0,
                    y: 100.0,
                    w: 200.0,
                    h: 200.0,
                    radius: 0.0,
                },
                fill: Some(red),
                stroke: None,
            }),
            "R",
        )
        .unwrap();
    let at = |e: &mut Engine, x: usize, y: usize| {
        e.flush();
        composite_px(
            e,
            TileCoord::new((x / 256) as i32, (y / 256) as i32),
            x % 256,
            y % 256,
        )
    };
    assert_eq!(at(&mut e, 95, 200), [1.0, 1.0, 1.0, 1.0]);
    e.edit("Style", |d| {
        d.find_mut(id).unwrap().effects = vec![
            Effect {
                enabled: true,
                kind: EffectKind::Stroke {
                    size: 8.0,
                    position: StrokePosition::Outside,
                    color: [0.0, 0.0, 1.0],
                    opacity: 1.0,
                },
            },
            Effect {
                enabled: true,
                kind: EffectKind::DropShadow {
                    color: [0.0; 3],
                    opacity: 0.8,
                    angle: 90.0,
                    distance: 30.0,
                    size: 4.0,
                    spread: 0.0,
                },
            },
        ];
        true
    });
    // Outside stroke: blue just left of the shape, the shape itself stays red.
    let p = at(&mut e, 96, 200);
    assert!(p[2] > 0.95 && p[0] < 0.05, "stroke {p:?}");
    assert_eq!(at(&mut e, 200, 200), red);
    // Shadow (light from the top → below) darkens the white background.
    let below = at(&mut e, 200, 318);
    assert!(below[0] < 0.4, "shadow {below:?}");
    assert_eq!(at(&mut e, 200, 85), [1.0, 1.0, 1.0, 1.0], "no shadow above");
    // Switched off → as before.
    e.edit("Off", |d| {
        for fx in &mut d.find_mut(id).unwrap().effects {
            fx.enabled = false;
        }
        true
    });
    assert_eq!(at(&mut e, 96, 200), [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(at(&mut e, 200, 318), [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn vector_preview_does_not_touch_the_document() {
    use imagine_core::VectorLayer;
    use imagine_core::vector::{Geometry, ShapeSpec, VectorKind};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let red = [1.0, 0.0, 0.0, 1.0];
    let v = VectorLayer::shape(ShapeSpec {
        geometry: Geometry::Rect {
            x: 10.0,
            y: 10.0,
            w: 100.0,
            h: 100.0,
            radius: 0.0,
        },
        fill: Some(red),
        stroke: None,
    });
    let id = e.add_vector(v.clone(), "R").unwrap();
    let mut blue = v.clone();
    if let VectorKind::Shape(s) = &mut blue.kind {
        s.fill = Some([0.0, 0.0, 1.0, 1.0]);
    }
    let steps = e.history_len().0;
    e.set_vector_preview(Some((id, &blue))).unwrap();
    e.flush();
    assert_eq!(
        composite_px(&e, TileCoord::new(0, 0), 50, 50),
        [0.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(e.history_len().0, steps);
    assert_eq!(e.doc().find(id).unwrap().vector(), Some(&v));
    e.set_vector_preview(None).unwrap();
    e.flush();
    assert_eq!(composite_px(&e, TileCoord::new(0, 0), 50, 50), red);
}

#[test]
#[ignore = "benchmark"]
fn text_resize_speed() {
    use imagine_core::VectorLayer;
    use imagine_core::style::{Effect, EffectKind, StrokePosition};
    use imagine_core::vector::{TextSpec, VectorKind};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(4000, 3000));
    e.fill(StrokeTarget::Pixels, [0.5, 0.5, 0.5, 1.0], 1.0, "F")
        .unwrap();
    let v = VectorLayer::text(TextSpec::new(
        "Ich bin eine Textebene",
        200.0,
        800.0,
        150.0,
        [1.0, 1.0, 0.0, 1.0],
    ));
    let id = e.add_vector(v.clone(), "T").unwrap();
    e.flush();
    for styled in [false, true] {
        if styled {
            e.edit("fx", |d| {
                d.find_mut(id).unwrap().effects = vec![Effect {
                    enabled: true,
                    kind: EffectKind::Stroke {
                        size: 6.0,
                        position: StrokePosition::Outside,
                        color: [1.0; 3],
                        opacity: 1.0,
                    },
                }];
                true
            });
            e.flush();
        }
        let t = std::time::Instant::now();
        for i in 0..10 {
            let mut w = v.clone();
            if let VectorKind::Text(s) = &mut w.kind {
                s.size = 150.0 + i as f32 * 3.0;
            }
            e.set_vector(id, w, "Size", Some("size")).unwrap();
            let t1 = std::time::Instant::now();
            e.flush();
            if i == 0 {
                eprintln!("  first flush {:?}", t1.elapsed());
            }
        }
        eprintln!("styled {styled}: {:?} per change", t.elapsed() / 10);
    }
}

#[test]
fn styled_rotated_text_updates_when_resized() {
    use imagine_core::VectorLayer;
    use imagine_core::style::{Effect, EffectKind, StrokePosition};
    use imagine_core::vector::{TextSpec, VectorKind};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(1024, 512));
    let mut v = VectorLayer::text(TextSpec::new(
        "Hallo",
        100.0,
        100.0,
        60.0,
        [1.0, 1.0, 0.0, 1.0],
    ));
    v.transform = [0.966, -0.259, 0.259, 0.966, -20.0, 60.0];
    let id = e.add_vector(v.clone(), "T").unwrap();
    e.edit("fx", |d| {
        d.find_mut(id).unwrap().effects = vec![Effect {
            enabled: true,
            kind: EffectKind::Stroke {
                size: 4.0,
                position: StrokePosition::Outside,
                color: [1.0; 3],
                opacity: 1.0,
            },
        }];
        true
    });
    e.flush();
    let ink = |e: &Engine| -> usize {
        (0..4)
            .flat_map(|tx| (0..2).map(move |ty| TileCoord::new(tx, ty)))
            .map(|c| {
                e.read_composite(&[c])[0]
                    .chunks(4)
                    .filter(|p| p[3].to_f32() > 0.5)
                    .count()
            })
            .sum()
    };
    let before = ink(&e);
    assert!(before > 1000);
    let mut big = v.clone();
    if let VectorKind::Text(t) = &mut big.kind {
        t.size = 120.0;
    }
    e.set_vector(id, big, "Size", Some("size")).unwrap();
    e.flush();
    let after = ink(&e);
    assert!(
        after > before * 2,
        "bigger text must show: {before} → {after}"
    );
}

#[test]
fn merged_edits_refresh_an_open_transform_preview() {
    use imagine_core::VectorLayer;
    use imagine_core::vector::{Geometry, ShapeSpec, VectorKind};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    let v = VectorLayer::shape(ShapeSpec {
        geometry: Geometry::Rect {
            x: 100.0,
            y: 100.0,
            w: 100.0,
            h: 100.0,
            radius: 0.0,
        },
        fill: Some([1.0, 0.0, 0.0, 1.0]),
        stroke: None,
    });
    let id = e.add_vector(v.clone(), "R").unwrap();
    let rot = glam::DAffine2::from_angle(0.1);
    let colour = |e: &mut Engine, c: [f32; 4]| {
        let mut w = v.clone();
        if let VectorKind::Shape(s) = &mut w.kind {
            s.fill = Some(c);
        }
        e.set_vector(id, w, "Farbe", Some("fill")).unwrap();
        // The app re-sets the transform preview every frame.
        e.set_transform_preview(Some((rot, imagine_gpu::Interp::Bicubic)))
            .unwrap();
        e.flush();
        composite_px(e, TileCoord::new(0, 0), 150, 160)
    };
    e.set_transform_preview(Some((rot, imagine_gpu::Interp::Bicubic)))
        .unwrap();
    assert_eq!(
        colour(&mut e, [0.0, 1.0, 0.0, 1.0])[1],
        1.0,
        "first change of a drag"
    );
    assert_eq!(
        colour(&mut e, [0.0, 0.0, 1.0, 1.0])[2],
        1.0,
        "later changes of the same drag"
    );
}

#[test]
fn font_preview_keeps_the_rest_of_the_image() {
    use imagine_core::VectorLayer;
    use imagine_core::vector::{TextSpec, VectorKind};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    e.fill(StrokeTarget::Pixels, [0.2, 0.4, 0.6, 1.0], 1.0, "BG")
        .unwrap();
    let v = VectorLayer::text(TextSpec::new(
        "Text",
        50.0,
        50.0,
        60.0,
        [1.0, 1.0, 0.0, 1.0],
    ));
    let id = e.add_vector(v.clone(), "T").unwrap();
    e.flush();
    let bg = composite_px(&e, TileCoord::new(1, 1), 100, 100);
    assert!(bg[3] > 0.99);
    let mut w = v.clone();
    if let VectorKind::Text(t) = &mut w.kind {
        t.font = "serif".into();
    }
    for _ in 0..3 {
        // Like the app: set every frame, then render.
        e.set_vector_preview(Some((id, &w))).unwrap();
        e.flush();
        assert_eq!(
            composite_px(&e, TileCoord::new(1, 1), 100, 100),
            bg,
            "background stays during the preview"
        );
    }
    let ink = |e: &Engine| {
        e.read_composite(&[TileCoord::new(0, 0)])[0]
            .chunks(4)
            .filter(|p| p[1].to_f32() > 0.9 && p[2].to_f32() < 0.1)
            .count()
    };
    let previewed = ink(&e);
    assert!(previewed > 100);
    // Apply: the same description becomes the document; the window closes.
    e.set_vector(id, w.clone(), "Edit text", Some(&format!("vector-{id}")))
        .unwrap();
    e.end_merge();
    e.set_transform_preview(None).unwrap();
    e.flush();
    assert_eq!(
        composite_px(&e, TileCoord::new(1, 1), 100, 100),
        bg,
        "background after applying"
    );
    assert_eq!(ink(&e), previewed, "the text after applying");
}

#[test]
fn vector_preview_never_replaces_pixels() {
    use imagine_core::VectorLayer;
    use imagine_core::vector::TextSpec;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    e.fill(StrokeTarget::Pixels, [0.2, 0.4, 0.6, 1.0], 1.0, "BG")
        .unwrap();
    e.flush();
    let bg = composite_px(&e, TileCoord::new(0, 0), 100, 100);
    let pixels = e.doc().active;
    let v = VectorLayer::text(TextSpec::new("", 0.0, 0.0, 40.0, [1.0; 4]));
    assert!(e.set_vector_preview(Some((pixels, &v))).is_err());
    e.flush();
    assert_eq!(composite_px(&e, TileCoord::new(0, 0), 100, 100), bg);
}

#[test]
fn copy_pixels_layer_selection_and_merged() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 400));
    // A red square 100..200 on an empty layer.
    e.select_shape(
        &Shape::Rect {
            x: 100.0,
            y: 100.0,
            w: 100.0,
            h: 100.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "R")
        .unwrap();
    e.deselect("x").unwrap();
    // Without a selection: cropped to the pixels.
    let c = e.copy_pixels(false).unwrap();
    assert_eq!((c.x, c.y, c.w, c.h), (100, 100, 100, 100));
    assert_eq!(c.px[0], [1.0, 0.0, 0.0, 1.0]);
    // Selection: only its part.
    e.select_shape(
        &Shape::Rect {
            x: 150.0,
            y: 150.0,
            w: 200.0,
            h: 200.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    let c = e.copy_pixels(false).unwrap();
    assert_eq!((c.x, c.y, c.w, c.h), (150, 150, 50, 50));
    // Merged sees the same.
    let m = e.copy_pixels(true).unwrap();
    assert_eq!((m.x, m.y, m.w, m.h), (150, 150, 50, 50));
    // Nothing there.
    e.select_shape(
        &Shape::Rect {
            x: 400.0,
            y: 300.0,
            w: 50.0,
            h: 50.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    assert!(e.copy_pixels(false).is_err());
}

#[test]
fn sample_average_point_and_area() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 400));
    // Red square 100..200 on an empty layer.
    e.select_shape(
        &Shape::Rect {
            x: 100.0,
            y: 100.0,
            w: 100.0,
            h: 100.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "R")
        .unwrap();
    e.deselect("x").unwrap();
    use imagine_gpu::SampleFrom;
    let c = e.sample_average(150, 150, 0, SampleFrom::Layer).unwrap();
    assert!(
        (c[0] - 1.0).abs() < 1e-3 && c[1] < 1e-3 && (c[3] - 1.0).abs() < 1e-3,
        "{c:?}"
    );
    // 3×3 across the left edge: x 99..101 → 2 of 3 columns red.
    let c = e.sample_average(100, 150, 1, SampleFrom::Layer).unwrap();
    assert!((c[3] - 2.0 / 3.0).abs() < 1e-2, "{c:?}");
    assert!(
        (c[0] - 1.0).abs() < 1e-2,
        "straight colour stays red: {c:?}"
    );
    assert!(e.sample_average(-1, 0, 0, SampleFrom::Layer).is_err());
}

#[test]
fn clipping_mask_limits_to_base_and_uses_base_opacity() {
    let Some(g) = gpu() else { return };
    let mut doc = Document::<()>::new(512, 256);
    doc.layers[0].name = "base".into();
    let mut e = Engine::new(&g.device, &g.queue, FMT, &doc);
    // Base: red square 50..100.
    e.select_shape(
        &Shape::Rect {
            x: 50.0,
            y: 50.0,
            w: 50.0,
            h: 50.0,
        },
        CombineOp::Replace,
        "A",
    )
    .unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "R")
        .unwrap();
    e.deselect("x").unwrap();
    let base = e.doc().active;
    // Clipped: green everywhere.
    e.edit("new", |d| {
        d.add_pixel_layer();
        true
    });
    e.fill(StrokeTarget::Pixels, [0.0, 1.0, 0.0, 1.0], 1.0, "G")
        .unwrap();
    let top = e.doc().active;
    e.edit("clip", |d| {
        d.find_mut(top).unwrap().clip = true;
        d.find_mut(base).unwrap().opacity = 0.5;
        true
    });
    e.flush();
    let t = TileCoord::new(0, 0);
    let inside = composite_px(&e, t, 70, 70);
    let outside = composite_px(&e, t, 20, 20);
    assert!(
        (inside[1] - 0.5).abs() < 0.02 && inside[0] < 0.02,
        "green at half opacity: {inside:?}"
    );
    assert!((inside[3] - 0.5).abs() < 0.02, "{inside:?}");
    assert!(outside[3] < 0.01, "nothing outside the base: {outside:?}");
    // Unclipping shows the green everywhere again.
    e.edit("unclip", |d| {
        d.find_mut(top).unwrap().clip = false;
        true
    });
    e.flush();
    assert!(composite_px(&e, t, 20, 20)[1] > 0.99);
}

#[test]
fn pass_through_group_blends_with_what_is_below() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 1.0, "blue")
        .unwrap();
    e.edit("new", |d| {
        d.add_pixel_layer();
        true
    });
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "red")
        .unwrap();
    let red = e.doc().active;
    e.edit("multiply + group", |d| {
        d.find_mut(red).unwrap().blend = BlendMode::Multiply;
        d.group_active().is_some()
    });
    let group = e.doc().active;
    assert!(
        e.doc().find(group).unwrap().pass_through,
        "new groups are Pass Through"
    );
    e.flush();
    let t = TileCoord::new(0, 0);
    let c = composite_px(&e, t, 10, 10);
    assert!(c[0] < 0.01 && c[2] < 0.01, "red × blue = black: {c:?}");
    // Isolated: multiply acts inside the group only → red over blue.
    e.edit("isolate", |d| {
        d.find_mut(group).unwrap().pass_through = false;
        true
    });
    e.flush();
    let c = composite_px(&e, t, 10, 10);
    assert!(c[0] > 0.99 && c[2] < 0.01, "isolated: red: {c:?}");
    // Pass Through with group opacity 50 %: halfway between blue and black.
    e.edit("half", |d| {
        let l = d.find_mut(group).unwrap();
        l.pass_through = true;
        l.opacity = 0.5;
        true
    });
    e.flush();
    let c = composite_px(&e, t, 10, 10);
    assert!((c[2] - 0.5).abs() < 0.02 && c[0] < 0.01, "{c:?}");
}
