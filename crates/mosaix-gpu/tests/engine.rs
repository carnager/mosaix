//! Headless GPU tests for the engine. Skipped (with a message) when no
//! suitable Vulkan adapter exists.

use glam::DVec2;
use half::f16;
use mosaix_core::blend::composite;
use mosaix_core::mask::{CombineOp, Shape};
use mosaix_core::{
    BlendMode, BlendSpace, BrushSettings, Document, InputSample, Stroke, TILE_SIZE, TileCoord, View,
};
use mosaix_gpu::{
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
    use mosaix_gpu::Interp;
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

    // Perspective: the rectangle 100..300 × 100..200 becomes a trapezoid
    // whose top edge shrinks to 150..250 and whose bottom stays.
    let quad = [
        DVec2::new(150.0, 100.0),
        DVec2::new(250.0, 100.0),
        DVec2::new(300.0, 200.0),
        DVec2::new(100.0, 200.0),
    ];
    let w = mosaix_gpu::Warp::from_quad(DVec2::new(100.0, 100.0), DVec2::new(300.0, 200.0), quad).unwrap();
    e.transform_active(w, Interp::Bilinear, "Perspective").unwrap();
    assert!(alpha_at(&mut e, 200, 105) > 0.99, "top middle");
    assert!(alpha_at(&mut e, 130, 105) < 0.01, "top corner cut off");
    assert!(alpha_at(&mut e, 105, 195) > 0.9, "bottom corner stays");
    let b = e.content_bounds(id).unwrap();
    assert!((b.0 - 100).abs() <= 1 && (b.2 - 300).abs() <= 1 && (b.1 - 100).abs() <= 1 && (b.3 - 200).abs() <= 1, "{b:?}");
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
        DAffine2::from_translation(DVec2::new(300.0, 0.0)).into(),
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
        DAffine2::from_translation(DVec2::new(300.0, 0.0)).into(),
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
    use mosaix_gpu::Interp;
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
    use mosaix_gpu::SampleFrom;
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
    use mosaix_core::Adjustment;
    use mosaix_core::adjust::Levels;
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
            ranges: Vec::new(),
        },
        // Colour ranges: reds desaturated and darker, blues shifted and brighter.
        Adjustment::HueSaturation {
            hue: 0.0,
            saturation: 10.0,
            lightness: 0.0,
            colorize: false,
            ranges: {
                let mut r = mosaix_core::adjust::HueRange::defaults();
                r[0].saturation = -70.0;
                r[0].lightness = -30.0;
                r[4].hue = 40.0;
                r[4].saturation = 60.0;
                r[4].lightness = 20.0;
                r
            },
        },
        Adjustment::HueSaturation {
            hue: 200.0,
            saturation: -30.0,
            lightness: 10.0,
            colorize: true,
            ranges: Vec::new(),
        },
        Adjustment::Vibrance {
            vibrance: 60.0,
            saturation: -10.0,
        },
        Adjustment::ColorBalance {
            shadows: [-20.0, 10.0, 30.0],
            midtones: [40.0, -10.0, 0.0],
            highlights: [0.0, 20.0, -50.0],
            preserve_luminosity: true,
        },
        Adjustment::ColorBalance {
            shadows: [0.0; 3],
            midtones: [-40.0, 30.0, 10.0],
            highlights: [0.0; 3],
            preserve_luminosity: false,
        },
        Adjustment::ChannelMixer {
            red: [80.0, 30.0, -10.0, 5.0],
            green: [0.0, 100.0, 0.0, 0.0],
            blue: [20.0, 0.0, 70.0, -10.0],
            gray: [40.0, 40.0, 20.0, 0.0],
            monochrome: false,
        },
        Adjustment::GradientMap {
            gradient: mosaix_core::gradient::Gradient::two([0.05, 0.0, 0.3], [1.0, 0.8, 0.2]),
            reverse: true,
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
    use mosaix_core::Filter;
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
    let grey = mosaix_core::color::srgb_to_linear(0.5);
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
    use mosaix_core::Adjustment;
    use mosaix_core::color::linear_to_srgb;
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
    use mosaix_core::Adjustment;
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
            ranges: Vec::new(),
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
    use mosaix_core::Filter;
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
    use mosaix_gpu::LiquifyTool;
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
    use mosaix_gpu::LiquifyTool;
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
    use mosaix_gpu::CloneSource;
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
    use mosaix_gpu::CloneSource;
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
    use mosaix_gpu::CloneSource;
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
    let lin = mosaix_core::color::srgb_to_linear;
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
    let lin = mosaix_core::color::srgb_to_linear;
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
    // The CPU comparison is slow unoptimised; MOSAIX_FILL_COMPARE=1 runs it.
    let compare = std::env::var_os("MOSAIX_FILL_COMPARE").is_some();
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
    use mosaix_core::VectorLayer;
    use mosaix_core::vector::{Geometry, ShapeSpec, TextSpec};
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
        paint: None,
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
        mosaix_gpu::Interp::Bicubic,
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
    if let mosaix_core::vector::VectorKind::Shape(s) = &mut green.kind {
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
                "Mosaix",
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
    use mosaix_core::VectorLayer;
    use mosaix_core::style::{Effect, EffectKind, StrokePosition};
    use mosaix_core::vector::{Geometry, ShapeSpec};
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
                paint: None,
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
                blend: None,
                global_light: false,
                kind: EffectKind::Stroke {
                    size: 8.0,
                    position: StrokePosition::Outside,
                    color: [0.0, 0.0, 1.0],
                    opacity: 1.0,
                },
            },
            Effect {
                enabled: true,
                blend: None,
                global_light: false,
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
    use mosaix_core::VectorLayer;
    use mosaix_core::vector::{Geometry, ShapeSpec, VectorKind};
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
        paint: None,
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
    use mosaix_core::VectorLayer;
    use mosaix_core::style::{Effect, EffectKind, StrokePosition};
    use mosaix_core::vector::{TextSpec, VectorKind};
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
                    blend: None,
                    global_light: false,
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
    use mosaix_core::VectorLayer;
    use mosaix_core::style::{Effect, EffectKind, StrokePosition};
    use mosaix_core::vector::{TextSpec, VectorKind};
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
            blend: None,
            global_light: false,
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
    use mosaix_core::VectorLayer;
    use mosaix_core::vector::{Geometry, ShapeSpec, VectorKind};
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
        paint: None,
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
        e.set_transform_preview(Some((rot.into(), mosaix_gpu::Interp::Bicubic)))
            .unwrap();
        e.flush();
        composite_px(e, TileCoord::new(0, 0), 150, 160)
    };
    e.set_transform_preview(Some((rot.into(), mosaix_gpu::Interp::Bicubic)))
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
    use mosaix_core::VectorLayer;
    use mosaix_core::vector::{TextSpec, VectorKind};
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
    use mosaix_core::VectorLayer;
    use mosaix_core::vector::TextSpec;
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
    use mosaix_gpu::SampleFrom;
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

#[test]
fn transform_moves_a_whole_group() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 256));
    // Red square 20..60 on the background, a text layer, both grouped.
    e.select_shape(&Shape::Rect { x: 20.0, y: 20.0, w: 40.0, h: 40.0 }, CombineOp::Replace, "A").unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "R").unwrap();
    e.deselect("x").unwrap();
    let px = e.doc().active;
    let text = mosaix_core::VectorLayer::text(mosaix_core::vector::TextSpec::new("Hi", 10.0, 100.0, 30.0, [0.0, 0.0, 0.0, 1.0]));
    let t = e.add_vector(text, "T").unwrap();
    e.edit("group", |d| {
        d.active = px;
        let g = d.group_active().unwrap();
        let tl = d.layers.iter().position(|l| l.id == t).unwrap();
        let tl = d.layers.remove(tl);
        if let mosaix_core::doc::Content::Group(c) = &mut d.find_mut(g).unwrap().content {
            c.push(tl);
        }
        d.active = g;
        true
    });
    let b = e.content_bounds(e.doc().active).unwrap();
    assert!(b.0 <= 20 && b.1 <= 20 && b.3 >= 100, "group bounds cover both: {b:?}");
    e.transform_active(glam::DAffine2::from_translation(glam::DVec2::new(100.0, 20.0)), mosaix_gpu::Interp::Bilinear, "move").unwrap();
    e.flush();
    let tc = TileCoord::new(0, 0);
    assert!(composite_px(&e, tc, 140, 60)[0] > 0.99, "square moved to 120..160 / 40..80");
    assert!(composite_px(&e, tc, 30, 30)[3] < 0.01, "old place empty");
    let v = e.doc().find(t).unwrap().vector().unwrap().transform;
    assert_eq!((v[4], v[5]), (100.0, 20.0), "text stays editable, transform composed");
}

#[test]
fn custom_tip_paints_its_shape() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(300, 300));
    // A tip covering only its left half.
    let n = mosaix_gpu::TIP_SIZE as usize;
    let tip: Vec<u8> = (0..n * n).map(|i| if i % n < n / 2 { 255 } else { 0 }).collect();
    e.set_custom_tips(&[tip]);
    e.begin_stroke(StrokeParams::default()).unwrap();
    let mut s = Stroke::new(BrushSettings {
        size: 120.0,
        tip: mosaix_gpu::FIRST_CUSTOM_TIP,
        spacing: 5.0,
        ..red_brush()
    });
    e.stroke_dabs(&s.add(InputSample { pos: DVec2::new(150.0, 150.0), pressure: 1.0 }));
    e.end_stroke("Custom tip");
    e.flush();
    let tile = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    assert!(px(&tile, 120, 150)[3] > 0.9, "left half painted: {:?}", px(&tile, 120, 150));
    assert!(px(&tile, 180, 150)[3] < 0.05, "right half empty: {:?}", px(&tile, 180, 150));
}

#[test]
fn fill_opacity_fades_the_pixels_but_not_the_style() {
    use mosaix_core::VectorLayer;
    use mosaix_core::style::{Effect, EffectKind, StrokePosition};
    use mosaix_core::vector::{Geometry, ShapeSpec};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    e.fill(StrokeTarget::Pixels, [1.0, 1.0, 1.0, 1.0], 1.0, "White").unwrap();
    let id = e
        .add_vector(
            VectorLayer::shape(ShapeSpec {
                geometry: Geometry::Rect { x: 50.0, y: 50.0, w: 100.0, h: 100.0, radius: 0.0 },
                fill: Some([0.0, 0.0, 0.0, 1.0]),
                stroke: None,
                paint: None,
            }),
            "R",
        )
        .unwrap();
    let at = |e: &mut Engine, x: usize, y: usize| {
        e.flush();
        composite_px(e, TileCoord::new(0, 0), x, y)
    };
    // No style: fill acts like opacity.
    e.edit("Fill", |d| {
        d.find_mut(id).unwrap().fill_opacity = 0.5;
        true
    });
    let p = at(&mut e, 100, 100);
    // 50 % black over white, mixed in sRGB: 0.5 encoded = 0.214 linear.
    assert!((p[0] - 0.214).abs() < 0.02, "half fill {p:?}");
    // With a stroke and no fill: only the stroke shows, at full strength.
    e.edit("Style", |d| {
        let l = d.find_mut(id).unwrap();
        l.fill_opacity = 0.0;
        l.effects = vec![Effect::new(EffectKind::Stroke {
            size: 6.0,
            position: StrokePosition::Inside,
            color: [1.0, 0.0, 0.0],
            opacity: 1.0,
        })];
        true
    });
    assert_eq!(at(&mut e, 100, 100), [1.0, 1.0, 1.0, 1.0], "inside empty");
    let edge = at(&mut e, 52, 100);
    assert!(edge[0] > 0.95 && edge[1] < 0.05, "stroke {edge:?}");
}

#[test]
fn layer_style_follows_the_layer_mask() {
    use mosaix_core::style::{Effect, EffectKind, StrokePosition};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(300, 200));
    e.fill(StrokeTarget::Pixels, [1.0, 1.0, 1.0, 1.0], 1.0, "White").unwrap();
    let mut id = 0;
    e.edit("New layer", |d| {
        id = d.add_pixel_layer();
        true
    });
    e.select_shape(&Shape::Rect { x: 50.0, y: 50.0, w: 200.0, h: 100.0 }, CombineOp::Replace, "Sel").unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Fill").unwrap();
    // Mask: only the left half of the rectangle (x < 150) shows.
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 150.0, h: 200.0 }, CombineOp::Replace, "Sel").unwrap();
    e.add_layer_mask("Mask").unwrap();
    e.deselect("Deselect").unwrap();
    e.edit("Style", |d| {
        d.find_mut(id).unwrap().effects = vec![Effect::new(EffectKind::Stroke {
            size: 6.0,
            position: StrokePosition::Outside,
            color: [0.0, 0.0, 1.0],
            opacity: 1.0,
        })];
        true
    });
    let at = |e: &mut Engine, x: usize, y: usize| {
        e.flush();
        composite_px(e, TileCoord::new((x / 256) as i32, (y / 256) as i32), x % 256, y % 256)
    };
    // The stroke runs along the mask edge (x = 150), like in Photoshop …
    let p = at(&mut e, 152, 100);
    assert!(p[2] > 0.95 && p[0] < 0.05, "stroke at the mask edge {p:?}");
    // … and still along the visible part of the shape.
    let p = at(&mut e, 47, 100);
    assert!(p[2] > 0.95 && p[0] < 0.05, "stroke at the left edge {p:?}");
    assert_eq!(at(&mut e, 100, 100), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(at(&mut e, 200, 100), [1.0, 1.0, 1.0, 1.0], "masked part hidden");
}

/// Remove tool path: a painted stroke is not kept; its footprint becomes the
/// hole of a fill job, and only the hole takes the computed pixels.
#[test]
fn painted_area_fill_job() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(600, 400));
    e.select_all("all").unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 1.0, "blue").unwrap();
    e.deselect("none").unwrap();
    let before = e.history_len().0;
    e.begin_stroke(StrokeParams { opacity: 1.0, ..Default::default() }).unwrap();
    let mut s = Stroke::new(red_brush());
    let mut dabs = Vec::new();
    for i in 0..=40 {
        dabs.extend(s.add(InputSample { pos: DVec2::new(200.0 + i as f64 * 5.0, 200.0), pressure: 1.0 }));
    }
    e.stroke_dabs(&dabs);
    let (layer, footprint) = e.take_stroke_footprint().unwrap();
    assert!(!footprint.is_empty());
    assert_eq!(e.history_len().0, before, "the preview stroke is not kept");
    // The preview is gone: still blue under the stroke.
    e.flush();
    let at = |e: &Engine, x: usize, y: usize| composite_px(e, TileCoord::new((x / TS) as i32, (y / TS) as i32), x % TS, y % TS);
    assert!(at(&e, 300, 200)[0] < 0.01);
    let job = e.fill_job_dabs(layer, &footprint, false, 512).unwrap();
    let r = job
        .run_with(|w, h, img, hole| {
            // The stroke (x 190…410) plus ¾ of its size on each side; full height.
            assert_eq!((w, h), (550, 400));
            assert!(hole.iter().any(|&b| b) && !hole.iter().all(|&b| b));
            for (p, &hl) in img.iter_mut().zip(hole) {
                if hl {
                    *p = [0.0, 1.0, 0.0];
                }
            }
            Ok::<(), ()>(())
        })
        .unwrap();
    assert!(e.apply_fill(r, "Remove"));
    assert_eq!(e.history_len().0, before + 1);
    e.flush();
    let mid = at(&e, 300, 200);
    assert!(mid[1] > 0.99 && mid[2] < 0.01, "{mid:?}");
    let away = at(&e, 300, 260);
    assert!(away[2] > 0.99 && away[1] < 0.01, "{away:?}");
}

/// Upscale: pixel layers are replaced by the computed pixels (here a 4×
/// nearest "model", reduced to 2×), alpha stays, off-canvas content too,
/// one undo step.
#[test]
fn upscale_job_scales_layers() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(300, 200));
    e.select_shape(&Shape::Rect { x: 10.0, y: 10.0, w: 50.0, h: 40.0 }, CombineOp::Replace, "r").unwrap();
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "red").unwrap();
    e.deselect("d").unwrap();
    // Something left of the canvas (kept, like everything off-canvas).
    let layer = e.doc().active;
    let mut px = vec![f16::ZERO; TS * TS * 4];
    for i in 0..TS * TS {
        px[i * 4 + 2] = f16::ONE;
        px[i * 4 + 3] = f16::ONE;
    }
    assert!(e.replace_tiles(layer, vec![(TileCoord::new(-1, 0), Some(px))], "off"));
    let before = e.history_len().0;
    let job = e.upscale_job(2).unwrap();
    let r = job
        .run_with(
            |w, h, rgb, _alpha| {
                let mut out = vec![[0.0f32; 3]; w * h * 16];
                for y in 0..h * 4 {
                    for x in 0..w * 4 {
                        out[y * w * 4 + x] = rgb[(y / 4) * w + x / 4];
                    }
                }
                Ok::<_, ()>((4, out))
            },
            || (),
        )
        .unwrap();
    e.apply_upscale(r, "Upscale").unwrap();
    assert_eq!(e.size(), (600, 400));
    assert_eq!(e.history_len().0, before + 1);
    e.flush();
    let at = |e: &Engine, x: usize, y: usize| composite_px(e, TileCoord::new((x / TS) as i32, (y / TS) as i32), x % TS, y % TS);
    let inside = at(&e, 70, 50);
    assert!(inside[0] > 0.99 && inside[3] > 0.99, "{inside:?}");
    let outside = at(&e, 200, 150);
    assert!(outside[3] < 0.01, "{outside:?}");
    // The off-canvas tile became two tiles further left.
    let tiles = e.doc().find(layer).unwrap().tiles().unwrap().clone();
    assert!(tiles.contains_key(&TileCoord::new(-2, 0)) && tiles.contains_key(&TileCoord::new(-1, 1)));
    e.undo();
    assert_eq!(e.size(), (300, 200));
}

#[test]
fn shadows_highlights_lifts_dark_areas() {
    use mosaix_core::Filter;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 256));
    // Dark grey left half, light grey right half.
    let dark = mosaix_core::color::srgb_to_linear(0.15);
    let light = mosaix_core::color::srgb_to_linear(0.85);
    e.fill(StrokeTarget::Pixels, [light, light, light, 1.0], 1.0, "Light").unwrap();
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 256.0, h: 256.0 }, CombineOp::Replace, "Selection").unwrap();
    e.fill(StrokeTarget::Pixels, [dark, dark, dark, 1.0], 1.0, "Dark").unwrap();
    e.deselect("x").unwrap();
    let px = |e: &mut Engine, x: usize| {
        e.flush();
        let c = TileCoord::containing(x as f64, 128.0);
        let v = composite_px(e, c, x % 256, 128);
        mosaix_core::color::linear_to_srgb(v[0])
    };
    e.apply_filter(&Filter::ShadowsHighlights { shadows: 50.0, highlights: 50.0, radius: 10.0 }, "S/H").unwrap();
    let (d, l) = (px(&mut e, 64), px(&mut e, 448));
    assert!(d > 0.25, "shadows lifted: {d}");
    assert!(l < 0.8, "highlights lowered: {l}");
}

#[test]
fn tone_brushes_dodge_burn_sponge_blur_sharpen() {
    use mosaix_gpu::{Tone, ToneKind};
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let srgb = mosaix_core::color::srgb_to_linear;
    // Left half muted red (sRGB 0.6, 0.3, 0.3), right half black: an edge at x = 128.
    let (r, gg) = (srgb(0.6), srgb(0.3));
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 0.0, 1.0], 1.0, "Black").unwrap();
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 128.0, h: 256.0 }, CombineOp::Replace, "Selection").unwrap();
    e.fill(StrokeTarget::Pixels, [r, gg, gg, 1.0], 1.0, "Red").unwrap();
    e.deselect("x").unwrap();
    let px = |e: &mut Engine, x: usize, y: usize| {
        e.flush();
        let c = TileCoord::containing(x as f64, y as f64);
        composite_px(e, c, x % 256, y % 256).map(mosaix_core::color::linear_to_srgb)
    };
    let tone = |kind| StrokeParams { tone: Some(Tone { kind, range: 1, amount: 0.5 }), ..Default::default() };
    let before = px(&mut e, 64, 64);
    stroke(&mut e, DVec2::new(64.0, 64.0), tone(ToneKind::Dodge));
    let d = px(&mut e, 64, 64);
    assert!(d[0] > before[0] + 0.05 && d[1] > before[1] + 0.05, "dodge brightens: {before:?} → {d:?}");
    e.undo();
    stroke(&mut e, DVec2::new(64.0, 64.0), tone(ToneKind::Burn));
    let b = px(&mut e, 64, 64);
    assert!(b[0] < before[0] - 0.05, "burn darkens: {b:?}");
    e.undo();
    stroke(&mut e, DVec2::new(64.0, 64.0), tone(ToneKind::Desaturate));
    let s = px(&mut e, 64, 64);
    assert!((s[0] - s[1]) < (before[0] - before[1]) - 0.05, "sponge desaturates: {s:?}");
    e.undo();
    // One stroke applies the effect once, even where its dabs overlap.
    e.begin_stroke(tone(ToneKind::Dodge)).unwrap();
    let mut st = Stroke::new(red_brush());
    for i in 0..20 {
        e.stroke_dabs(&st.add(InputSample { pos: DVec2::new(64.0 + (i % 2) as f64, 64.0), pressure: 1.0 }));
    }
    e.end_stroke("Dodge");
    let again = px(&mut e, 64, 64);
    assert!((again[0] - d[0]).abs() < 0.02, "no build-up within a stroke: {d:?} vs {again:?}");
    e.undo();
    // Blur softens the edge, sharpen makes it harsher.
    let edge = |e: &mut Engine| px(e, 127, 64)[0];
    let e0 = edge(&mut e);
    stroke(&mut e, DVec2::new(128.0, 64.0), StrokeParams { tone: Some(Tone { kind: ToneKind::Blur, range: 1, amount: 1.0 }), ..Default::default() });
    assert!(edge(&mut e) < e0 - 0.05, "blur: {e0} → {}", edge(&mut e));
    e.undo();
    stroke(&mut e, DVec2::new(128.0, 64.0), StrokeParams { tone: Some(Tone { kind: ToneKind::Sharpen, range: 1, amount: 1.0 }), ..Default::default() });
    assert!(edge(&mut e) > e0 + 0.03, "sharpen: {e0} → {}", edge(&mut e));
}

#[test]
fn motion_blur_median_and_mosaic() {
    use mosaix_core::Filter;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 256));
    // White right half (x ≥ 256) on transparent, plus a 2 px white speck at (100, 100).
    let white = |e: &mut Engine, x: f64, y: f64, w: f64, h: f64| {
        e.select_shape(&Shape::Rect { x, y, w, h }, CombineOp::Replace, "Selection").unwrap();
        e.fill(StrokeTarget::Pixels, [1.0; 4], 1.0, "White").unwrap();
        e.deselect("x").unwrap();
    };
    white(&mut e, 256.0, 0.0, 256.0, 256.0);
    white(&mut e, 100.0, 100.0, 2.0, 2.0);
    // Horizontal motion blur smears the vertical edge sideways; vertical does not.
    e.apply_filter(&Filter::MotionBlur { angle: 0.0, distance: 40.0 }, "Motion").unwrap();
    let a = alpha_at(&mut e, 246, 50);
    assert!(a > 0.15 && a < 0.5, "smeared edge: {a}");
    assert!(alpha_at(&mut e, 200, 50) < 0.01);
    e.undo();
    e.apply_filter(&Filter::MotionBlur { angle: 90.0, distance: 40.0 }, "Motion").unwrap();
    assert!(alpha_at(&mut e, 250, 50) < 0.01 && alpha_at(&mut e, 260, 50) > 0.99);
    e.undo();
    // Median: the speck goes, the edge stays.
    e.apply_filter(&Filter::Median { radius: 2.0 }, "Median").unwrap();
    assert!(alpha_at(&mut e, 100, 100) < 0.01, "speck removed");
    assert!(alpha_at(&mut e, 255, 50) < 0.01 && alpha_at(&mut e, 256, 50) > 0.99, "edge kept");
    e.undo();
    // Mosaic 100 px: the cell 200…300 straddles the edge → 44 % covered.
    e.apply_filter(&Filter::Pixelate { size: 100.0 }, "Mosaic").unwrap();
    let (l, r) = (alpha_at(&mut e, 210, 10), alpha_at(&mut e, 290, 10));
    assert!((l - 0.44).abs() < 0.05 && (l - r).abs() < 1e-3, "{l} {r}");
}

#[test]
fn smart_objects_scale_without_loss() {
    use glam::DAffine2;
    use mosaix_gpu::Interp;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    // A 1 px checker in the square 100…164: fine detail that a pixel
    // layer would lose when shrunk and enlarged again.
    let layer = e.doc().active;
    let ts = TILE_SIZE as usize;
    let mut px = vec![f16::ZERO; ts * ts * 4];
    for y in 100..164 {
        for x in 100..164 {
            let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            let i = (y * ts + x) * 4;
            px[i..i + 4].copy_from_slice(&[v, v, v, 1.0].map(f16::from_f32));
        }
    }
    assert!(e.replace_tiles(layer, vec![(TileCoord::new(0, 0), Some(px))], "Checker"));
    let at = |e: &mut Engine, x: usize, y: usize| {
        e.flush();
        composite_px(e, TileCoord::new(0, 0), x, y)
    };
    let before = [at(&mut e, 120, 130), at(&mut e, 121, 130)];
    e.convert_to_smart(layer, "Smart").unwrap();
    assert!(e.doc().find(layer).unwrap().raw().is_some_and(|r| r.data.is_image()));
    assert_eq!(e.content_bounds(layer), Some((100, 100, 164, 164)));
    let c = DVec2::splat(132.0);
    let scale = |k: f64| DAffine2::from_translation(c) * DAffine2::from_scale(DVec2::splat(k)) * DAffine2::from_translation(-c);
    e.transform_layer(scale(0.25), Interp::Bicubic, false, "Shrink").unwrap();
    e.transform_layer(scale(4.0), Interp::Bicubic, false, "Grow").unwrap();
    let after = [at(&mut e, 120, 130), at(&mut e, 121, 130)];
    for (b, a) in before.iter().zip(after) {
        assert!((b[0] - a[0]).abs() < 0.02 && (a[3] - 1.0).abs() < 0.02, "{b:?} vs {a:?}");
    }
    // Rasterized it is pixels again.
    e.rasterize(layer, "Rasterize").unwrap();
    assert!(e.doc().find(layer).unwrap().tiles().is_some());
}

#[test]
fn merge_keeps_the_look_and_off_canvas_pixels() {
    use mosaix_core::MergeKind;
    let Some(g) = gpu() else { return };
    for space in [BlendSpace::Srgb, BlendSpace::Linear] {
        let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
        let bottom = e.doc().active;
        e.upload_tile(bottom, TileCoord::new(0, 0), &pattern(1, true));
        let (mut top, mut hidden) = (0, 0);
        e.edit("Layers", |d| {
            d.blend_space = space;
            top = d.add_pixel_layer();
            let l = d.find_mut(top).unwrap();
            l.blend = BlendMode::Multiply;
            l.opacity = 0.7;
            hidden = d.add_pixel_layer();
            d.find_mut(hidden).unwrap().visible = false;
            d.active = top;
            true
        });
        e.upload_tile(top, TileCoord::new(0, 0), &pattern(4, false));
        // Outside the canvas: must survive the merge.
        e.upload_tile(top, TileCoord::new(-1, 0), &pattern(2, true));
        e.upload_tile(hidden, TileCoord::new(0, 0), &pattern(3, true));
        e.flush();
        let before = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);

        let merged = e.merge(MergeKind::Down, "Merge Down").unwrap();
        e.flush();
        let after = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
        let worst = before.iter().zip(&after).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0.0, f32::max);
        assert!(worst < 2e-3, "{space:?}: looks different after merging ({worst})");
        let d = e.doc();
        assert_eq!(d.layers.len(), 2, "bottom and top became one, the hidden layer stays");
        assert_eq!(d.active, merged);
        let l = d.find(merged).unwrap();
        assert_eq!((l.blend, l.opacity), (BlendMode::Normal, 1.0));
        // Off canvas: the top layer's pixels, Multiply onto nothing below.
        let off = e.read_layer_tile(merged, TileCoord::new(-1, 0)).expect("off-canvas tile kept");
        assert!(px(&off, 10, 10)[3] > 0.6, "{:?}", px(&off, 10, 10));

        // Undo brings both back.
        e.undo();
        assert!(e.doc().find(top).is_some() && e.doc().find(bottom).is_some());
        e.redo();

        // Flatten: one layer, the hidden one is gone, the look the same.
        e.merge(MergeKind::Flatten, "Flatten").unwrap();
        e.flush();
        let flat = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
        let worst = before.iter().zip(&flat).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0.0, f32::max);
        assert!(worst < 2e-3, "{space:?}: flatten changed the look ({worst})");
        assert_eq!(e.doc().layers.len(), 1);
        assert!(e.merge(MergeKind::Flatten, "Flatten").is_err(), "nothing left to flatten");
        assert!(e.merge(MergeKind::Down, "Merge Down").is_err(), "nothing below");
    }
}

#[test]
fn fill_onto_a_new_layer_sees_all_layers_and_leaves_them_alone() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let bottom = e.doc().active;
    e.upload_tile(bottom, TileCoord::new(0, 0), &pattern(1, true));
    let mut top = 0;
    e.edit("Layer", |d| {
        top = d.add_pixel_layer();
        true
    });
    // The top layer covers only the right half: the fill must see both.
    let mut half = pattern(5, true);
    for y in 0..TS {
        for x in 0..TS / 2 {
            half[(y * TS + x) * 4..][..4].fill(f16::ZERO);
        }
    }
    e.upload_tile(top, TileCoord::new(0, 0), &half);
    e.flush();
    let shown = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let (b0, t0) = (e.read_layer_tile(bottom, TileCoord::new(0, 0)), e.read_layer_tile(top, TileCoord::new(0, 0)));
    e.select_shape(&Shape::Rect { x: 64.0, y: 64.0, w: 128.0, h: 64.0 }, CombineOp::Replace, "Selection").unwrap();

    let job = e.fill_job_new_layer("Generated", 0, 0).unwrap();
    let r = job
        .run_with(|w, h, rgb, hole| {
            // Outside the hole the provider sees the image as shown.
            assert!(w > 0 && h > 0 && hole.iter().any(|h| !h));
            for (c, &hl) in rgb.iter_mut().zip(hole) {
                if hl {
                    *c = [1.0, 0.0, 0.0];
                }
            }
            Ok::<(), ()>(())
        })
        .unwrap();
    assert!(e.apply_fill(r, "Generative Fill"));
    e.flush();
    let d = e.doc();
    assert_eq!(d.layers.len(), 3);
    let new = d.active;
    assert!(new != top && new != bottom);
    assert_eq!(d.layers[2].id, new, "above the active layer");
    assert_eq!(e.read_layer_tile(bottom, TileCoord::new(0, 0)), b0, "bottom layer untouched");
    assert_eq!(e.read_layer_tile(top, TileCoord::new(0, 0)), t0, "top layer untouched");
    let t = e.read_layer_tile(new, TileCoord::new(0, 0)).unwrap();
    let inside = px(&t, 100, 90);
    assert!((inside[0] - 1.0).abs() < 1e-3 && inside[1] < 1e-3 && (inside[3] - 1.0).abs() < 1e-3, "{inside:?}");
    assert_eq!(px(&t, 10, 10)[3], 0.0, "nothing outside the selection");
    // The image outside the selection looks as before.
    let after = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    assert_eq!(px(&after, 10, 10), px(&shown, 10, 10));
    assert_eq!(px(&after, 250, 250), px(&shown, 250, 250));
    // One undo step takes it away again.
    e.undo();
    assert_eq!(e.doc().layers.len(), 2);
}

#[test]
fn layer_via_copy_cut_and_select_layer_pixels() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let layer = e.doc().active;
    let src = pattern(2, false);
    e.upload_tile(layer, TileCoord::new(0, 0), &src);
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 100.0, h: 256.0 }, CombineOp::Replace, "Sel").unwrap();

    let copy = e.layer_via(false, "Layer via Copy").unwrap();
    assert_eq!(e.doc().layers.len(), 2);
    assert_eq!(e.doc().active, copy);
    let t = e.read_layer_tile(copy, TileCoord::new(0, 0)).unwrap();
    assert_eq!(px(&t, 50, 50), px(&src, 50, 50), "copied exactly");
    assert_eq!(px(&t, 150, 50)[3], 0.0, "only the selection");
    assert_eq!(e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap(), src, "copy leaves the layer alone");
    e.undo();

    // Cut: same new layer, the pixels leave the source (one undo step).
    e.edit("Active", |d| {
        d.active = layer;
        true
    });
    let cut = e.layer_via(true, "Layer via Cut").unwrap();
    let rest = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    assert_eq!(px(&rest, 50, 50)[3], 0.0, "cut out");
    assert_eq!(px(&rest, 150, 50), px(&src, 150, 50), "outside stays");
    assert_eq!(px(&e.read_layer_tile(cut, TileCoord::new(0, 0)).unwrap(), 50, 50), px(&src, 50, 50));
    e.undo();
    assert_eq!(e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap(), src);
    assert_eq!(e.doc().layers.len(), 1);

    // Ctrl+click: the selection follows the layer's opacity.
    e.deselect("None").unwrap();
    e.select_layer_pixels(layer, CombineOp::Replace, "Select Pixels").unwrap();
    let sel = e.selection_cpu().unwrap();
    let a = |x: usize, y: usize| sel.tiles[&TileCoord::new(0, 0)].get(x, y);
    for (x, y) in [(3, 7), (100, 200), (31, 31)] {
        let want = (px(&src, x, y)[3] * 255.0).round() as i32;
        assert!((a(x, y) as i32 - want).abs() <= 1, "({x},{y}): {} vs {want}", a(x, y));
    }
}

#[test]
fn adjustments_bake_into_pixels_as_previewed() {
    use mosaix_core::Adjustment;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let layer = e.doc().active;
    e.upload_tile(layer, TileCoord::new(0, 0), &pattern(3, false));
    e.edit("Look", |d| {
        let l = d.find_mut(layer).unwrap();
        l.opacity = 0.6;
        l.blend = BlendMode::Screen;
        true
    });
    e.select_shape(&Shape::Ellipse { cx: 115.0, cy: 100.0, rx: 75.0, ry: 60.0 }, CombineOp::Replace, "Sel").unwrap();
    e.feather_selection(6.0, "Feather").unwrap();
    let mut hue = Adjustment::defaults().into_iter().find(|a| matches!(a, Adjustment::HueSaturation { .. })).unwrap();
    if let Adjustment::HueSaturation { hue: h, saturation, lightness, .. } = &mut hue {
        (*h, *saturation, *lightness) = (40.0, -30.0, 10.0);
    }
    for a in [Adjustment::Invert, hue] {
        let before = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
        e.set_adjust_preview(Some(&a)).unwrap();
        e.flush();
        let preview = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
        e.apply_adjustment(&a, "Adjust").unwrap();
        e.flush();
        let baked = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
        let worst = preview.iter().zip(&baked).map(|(a, b)| (a.to_f32() - b.to_f32()).abs()).fold(0.0, f32::max);
        assert!(worst < 3e-3, "{a:?}: baked differs from the preview ({worst})");
        let after = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
        // Outside the selection nothing changed; inside it did; alpha kept.
        let (o, n) = (px(&before, 2, 2), px(&after, 2, 2));
        assert!(o.iter().zip(n).all(|(a, b)| (a - b).abs() < 1e-3), "{o:?} vs {n:?}");
        assert!((px(&after, 115, 100)[3] - px(&before, 115, 100)[3]).abs() < 1e-3);
        assert_ne!(px(&after, 115, 100), px(&before, 115, 100));
        let l = e.doc().find(layer).unwrap();
        assert_eq!((l.opacity, l.blend), (0.6, BlendMode::Screen), "the layer keeps its look");
    }
}

#[test]
fn paint_bucket_fills_the_region_and_keeps_the_selection() {
    use mosaix_gpu::SampleFrom;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let layer = e.doc().active;
    let mut t = vec![f16::ZERO; TS * TS * 4];
    for y in 0..TS {
        for x in 0..TS {
            let c = if x < 128 { [1.0, 0.0, 0.0, 1.0] } else { [0.0, 0.0, 1.0, 1.0] };
            t[(y * TS + x) * 4..][..4].copy_from_slice(&c.map(f16::from_f32));
        }
    }
    e.upload_tile(layer, TileCoord::new(0, 0), &t);
    e.bucket_fill(20, 20, 32.0, true, false, SampleFrom::Layer, [0.0, 1.0, 0.0, 1.0], 1.0, "Paint Bucket").unwrap();
    let after = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    assert_eq!(px(&after, 20, 200), [0.0, 1.0, 0.0, 1.0], "the red half is green");
    assert_eq!(px(&after, 200, 20), [0.0, 0.0, 1.0, 1.0], "the blue half stays");
    assert!(!e.has_selection(), "no selection left behind");
    e.undo();
    assert_eq!(e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap(), t);
    assert!(!e.has_selection(), "undo does not bring a temporary selection back");
    // Within a selection only the selected part of the region.
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 256.0, h: 100.0 }, CombineOp::Replace, "Sel").unwrap();
    e.bucket_fill(20, 20, 32.0, true, false, SampleFrom::Layer, [0.0, 1.0, 0.0, 1.0], 1.0, "Paint Bucket").unwrap();
    let after = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    assert_eq!(px(&after, 20, 50)[1], 1.0);
    assert_eq!(px(&after, 20, 200), [1.0, 0.0, 0.0, 1.0]);
    assert!(e.has_selection());
}

#[test]
fn several_selected_layers_move_and_merge_together() {
    use mosaix_core::MergeKind;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let a = e.doc().active;
    let (mut b, mut c) = (0, 0);
    e.edit("Layers", |d| {
        b = d.add_pixel_layer();
        c = d.add_pixel_layer();
        true
    });
    for id in [a, b, c] {
        e.upload_tile(id, TileCoord::new(0, 0), &pattern(id as usize, false));
    }
    let before = |e: &Engine, id| e.read_layer_tile(id, TileCoord::new(0, 0)).unwrap();
    let (pa, pb, pc) = (before(&e, a), before(&e, b), before(&e, c));
    e.select(|d| {
        d.active = c;
        d.picked = vec![a];
    });
    e.transform_active(glam::DAffine2::from_translation(DVec2::new(10.0, 0.0)), mosaix_gpu::Interp::Nearest, "Move").unwrap();
    let moved = |e: &Engine, id| px(&e.read_layer_tile(id, TileCoord::new(0, 0)).unwrap(), 30, 40);
    assert_eq!(moved(&e, a), px(&pa, 20, 40), "a moved");
    assert_eq!(moved(&e, c), px(&pc, 20, 40), "c moved");
    assert_eq!(moved(&e, b), px(&pb, 30, 40), "b, not selected, stays");
    e.undo();
    assert_eq!(moved(&e, a), px(&pa, 30, 40), "one undo step for both");
    assert_eq!(moved(&e, c), px(&pc, 30, 40));

    // Merging the selected (non-adjacent) layers: b stays, the look too.
    e.select(|d| {
        d.active = c;
        d.picked = vec![b];
    });
    e.flush();
    let look = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let m = e.merge(MergeKind::Selected, "Merge Layers").unwrap();
    e.flush();
    let after = e.read_composite(&[TileCoord::new(0, 0)]).remove(0);
    let worst = look.iter().zip(&after).map(|(x, y)| (x.to_f32() - y.to_f32()).abs()).fold(0.0, f32::max);
    assert!(worst < 2e-3, "{worst}");
    let ids: Vec<_> = e.doc().layers.iter().map(|l| l.id).collect();
    assert_eq!(ids, vec![a, m]);
}

#[test]
fn stroke_selection_inside_centre_outside() {
    use mosaix_core::style::StrokePosition as P;
    let Some(g) = gpu() else { return };
    for (pos, red, clear) in [(P::Inside, (55, 100), (45, 100)), (P::Outside, (45, 100), (55, 100)), (P::Center, (47, 100), (60, 100))] {
        let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
        let layer = e.doc().active;
        e.select_shape(&Shape::Rect { x: 50.0, y: 50.0, w: 100.0, h: 100.0 }, CombineOp::Replace, "Sel").unwrap();
        e.stroke_selection(10.0, pos, [1.0, 0.0, 0.0, 1.0], 1.0, "Stroke").unwrap();
        let t = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
        assert!(px(&t, red.0, red.1)[0] > 0.99, "{pos:?}: {red:?} red");
        assert_eq!(px(&t, clear.0, clear.1)[3], 0.0, "{pos:?}: {clear:?} empty");
        assert_eq!(px(&t, 100, 100)[3], 0.0, "{pos:?}: middle empty");
        assert!(e.has_selection(), "the selection stays");
    }
}

#[test]
fn locked_layers() {
    use mosaix_core::doc::Lock;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let layer = e.doc().active;
    // Left half half-transparent blue, right half empty.
    let mut t = vec![f16::ZERO; TS * TS * 4];
    for y in 0..TS {
        for x in 0..TS / 2 {
            t[(y * TS + x) * 4..][..4].copy_from_slice(&[0.0, 0.0, 0.5, 0.5].map(f16::from_f32));
        }
    }
    e.upload_tile(layer, TileCoord::new(0, 0), &t);
    e.edit("Lock", |d| {
        d.find_mut(layer).unwrap().lock = Lock { transparency: true, ..Default::default() };
        true
    });
    // Fill and brush keep each pixel's opacity.
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Fill").unwrap();
    let after = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    let l = px(&after, 10, 10);
    assert!((l[0] - 0.5).abs() < 1e-3 && l[2] < 1e-3 && (l[3] - 0.5).abs() < 1e-3, "red at the old opacity: {l:?}");
    assert_eq!(px(&after, 200, 10)[3], 0.0, "transparent stays transparent");
    stroke(&mut e, DVec2::new(200.0, 100.0), StrokeParams::default());
    let after = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    assert_eq!(px(&after, 200, 100)[3], 0.0, "no paint where it was empty");
    // Pixels locked: nothing paints; position locked: nothing moves.
    e.edit("Lock", |d| {
        d.find_mut(layer).unwrap().lock = Lock { pixels: true, position: true, ..Default::default() };
        true
    });
    assert!(e.fill(StrokeTarget::Pixels, [0.0, 1.0, 0.0, 1.0], 1.0, "Fill").is_err());
    assert!(e.begin_stroke(StrokeParams::default()).is_err());
    assert!(e.translate_active(5, 0, "Move").is_err());
}

#[test]
fn align_and_distribute_layers() {
    use mosaix_gpu::Align;
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let a = e.doc().active;
    let (mut b, mut c) = (0, 0);
    e.edit("Layers", |d| {
        b = d.add_pixel_layer();
        c = d.add_pixel_layer();
        true
    });
    let square = |x0: usize, y0: usize| {
        let mut t = vec![f16::ZERO; TS * TS * 4];
        for y in y0..y0 + 10 {
            for x in x0..x0 + 10 {
                t[(y * TS + x) * 4..][..4].fill(f16::ONE);
            }
        }
        t
    };
    e.upload_tile(a, TileCoord::new(0, 0), &square(0, 0));
    e.upload_tile(b, TileCoord::new(0, 0), &square(30, 40));
    e.upload_tile(c, TileCoord::new(0, 0), &square(200, 100));
    e.select(|d| {
        d.active = c;
        d.picked = vec![a, b];
    });
    e.align_layers(Align::Left, "Align").unwrap();
    for id in [a, b, c] {
        assert_eq!(e.content_bounds(id).unwrap().0, 0, "layer {id} at the left edge");
    }
    assert_eq!(e.history_len().0, 2, "one undo step (after the layers step)");
    e.undo();
    assert_eq!(e.content_bounds(c).unwrap().0, 200);
    e.distribute_layers(true, "Distribute").unwrap();
    let cx = |e: &mut Engine, id| {
        let (x0, _, x1, _) = e.content_bounds(id).unwrap();
        (x0 + x1) / 2
    };
    let (ca, cb, cc) = (cx(&mut e, a), cx(&mut e, b), cx(&mut e, c));
    assert_eq!((ca, cc), (5, 205), "the outer ones stay");
    assert!((cb - 105).abs() <= 1, "the middle one is centred: {cb}");
}

#[test]
fn pencil_has_no_soft_pixels() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let layer = e.doc().active;
    e.begin_stroke(StrokeParams::default()).unwrap();
    let mut b = red_brush();
    (b.size, b.hardness, b.spacing) = (7.0, 2.0, 0.1);
    let mut s = Stroke::new(b);
    let mut dabs = Vec::new();
    for i in 0..60 {
        let t = i as f64;
        dabs.extend(s.add(InputSample { pos: DVec2::new(40.0 + t * 2.0, 50.0 + t * 1.3), pressure: 1.0 }));
    }
    e.stroke_dabs(&dabs);
    e.end_stroke("Pencil");
    e.flush();
    let t = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    let mut painted = 0;
    for y in 0..TS {
        for x in 0..TS {
            let a = px(&t, x, y)[3];
            assert!(a == 0.0 || (a - 1.0).abs() < 1e-3, "soft pixel {a} at ({x},{y})");
            painted += (a > 0.5) as usize;
        }
    }
    assert!(painted > 300, "{painted}");
}

#[test]
fn quick_mask_round_trip_and_painting() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let back = e.doc().active;
    e.select_shape(&Shape::Rect { x: 0.0, y: 0.0, w: 128.0, h: 256.0 }, CombineOp::Replace, "Sel").unwrap();
    let qm = e.enter_quick_mask("Quick Mask").unwrap();
    assert!(!e.has_selection(), "painting is not clipped");
    assert_eq!(e.doc().active, qm);
    let t = e.read_layer_tile(qm, TileCoord::new(0, 0)).unwrap();
    assert_eq!(px(&t, 10, 10)[3], 0.0, "selected: no red");
    assert_eq!(px(&t, 200, 10)[3], 1.0, "not selected: red");
    // Erase the red in the top right corner: that becomes selected too.
    e.select_shape(&Shape::Rect { x: 128.0, y: 0.0, w: 128.0, h: 64.0 }, CombineOp::Replace, "Sel").unwrap();
    e.clear_selected("Clear").unwrap();
    e.deselect("None").unwrap();
    e.leave_quick_mask(qm, back, "Quick Mask").unwrap();
    assert_eq!(e.doc().active, back);
    assert!(e.doc().find(qm).is_none(), "the helper layer is gone");
    let sel = e.selection_cpu().unwrap();
    let at = |x: usize, y: usize| sel.tiles.get(&TileCoord::new(0, 0)).map_or(sel.default, |m| m.get(x, y));
    assert_eq!(at(10, 200), 255);
    assert_eq!(at(200, 20), 255, "erased mask = selected");
    assert_eq!(at(200, 200), 0);
}

#[test]
fn transform_selection_moves_only_the_outline() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(512, 512));
    let layer = e.doc().active;
    e.upload_tile(layer, TileCoord::new(0, 0), &pattern(1, true));
    let before = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    e.select_shape(&Shape::Rect { x: 100.0, y: 100.0, w: 50.0, h: 50.0 }, CombineOp::Replace, "Sel").unwrap();
    assert_eq!(e.selection_bounds(), Some((100, 100, 150, 150)));
    let c = DVec2::splat(125.0);
    let fwd = glam::DAffine2::from_translation(c + DVec2::new(100.0, 0.0)) * glam::DAffine2::from_scale(DVec2::splat(2.0)) * glam::DAffine2::from_translation(-c);
    e.transform_selection(fwd, "Transform Selection").unwrap();
    let b = e.selection_bounds().unwrap();
    assert!((b.0 - 175).abs() <= 1 && (b.2 - 275).abs() <= 1 && (b.1 - 75).abs() <= 1 && (b.3 - 175).abs() <= 1, "{b:?}");
    assert_eq!(e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap(), before, "pixels untouched");
    e.undo();
    assert_eq!(e.selection_bounds(), Some((100, 100, 150, 150)));
}

#[test]
fn history_brush_paints_back_an_earlier_state() {
    let Some(g) = gpu() else { return };
    let mut e = Engine::new(&g.device, &g.queue, FMT, &Document::<()>::new(256, 256));
    let layer = e.doc().active;
    e.fill(StrokeTarget::Pixels, [1.0, 0.0, 0.0, 1.0], 1.0, "Red").unwrap();
    e.fill(StrokeTarget::Pixels, [0.0, 0.0, 1.0, 1.0], 1.0, "Blue").unwrap();
    // Source: after "Red" (position 1).
    e.set_history_source(1);
    let params = StrokeParams {
        clone: Some(mosaix_gpu::CloneSource { offset: [0, 0], all_layers: false, heal: false, spot: false }),
        history: true,
        ..Default::default()
    };
    stroke(&mut e, DVec2::new(100.0, 100.0), params);
    let t = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    let p = px(&t, 100, 100);
    assert!(p[0] > 0.99 && p[2] < 0.01, "red again under the brush: {p:?}");
    assert!(px(&t, 200, 200)[2] > 0.99, "blue elsewhere");
    // A state where the layer had no pixels at all is refused when the
    // layer did not exist; position 0 (empty layer) paints transparency.
    e.set_history_source(0);
    let params = StrokeParams {
        clone: Some(mosaix_gpu::CloneSource { offset: [0, 0], all_layers: false, heal: false, spot: false }),
        history: true,
        ..Default::default()
    };
    stroke(&mut e, DVec2::new(200.0, 200.0), params);
    let t = e.read_layer_tile(layer, TileCoord::new(0, 0)).unwrap();
    assert!(px(&t, 200, 200)[3] < 0.01, "back to the empty layer");
}
