//! Window, event loop, input handling and frame scheduling.
//!
//! Wayland is used directly through smithay-client-toolkit. Frames are driven
//! by `wl_surface.frame` callbacks: input is batched per event-loop dispatch
//! and at most one frame is rendered per compositor refresh.

use std::collections::{HashSet, VecDeque};
use std::ptr::NonNull;
use std::time::Instant;

use anyhow::Context;
use glam::DVec2;
use half::f16;
use imagine_core::{BrushSettings, InputSample, Stroke, TileCoord, View};
use imagine_gpu::{Canvas, GpuContext, Overlay};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, FrameCallbackData},
    delegate_registry,
    dispatch2::Dispatch2,
    output::{OutputHandler, OutputState},
    reexports::{
        calloop::{EventLoop, channel},
        calloop_wayland_source::WaylandSource,
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        Capability, SeatHandler, SeatState,
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers, RawModifiers},
        pointer::{CursorIcon, PointerEvent, PointerEventKind, PointerHandler, ThemeSpec, ThemedPointer},
    },
    shell::{
        WaylandSurface,
        xdg::{
            XdgShell,
            window::{Window, WindowConfigure, WindowDecorations, WindowHandler},
        },
    },
    shm::{Shm, ShmHandler},
};
use wayland_client::{
    Connection, Proxy, QueueHandle,
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_surface},
};
use wayland_protocols::wp::{
    fractional_scale::v1::client::{
        wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
        wp_fractional_scale_v1::{self, WpFractionalScaleV1},
    },
    tablet::zv2::client::{zwp_tablet_manager_v2::ZwpTabletManagerV2, zwp_tablet_seat_v2::ZwpTabletSeatV2},
    viewporter::client::{wp_viewport::WpViewport, wp_viewporter::WpViewporter},
};

use crate::loader::{self, LoaderMsg, Source};
use crate::tablet::{self, TabletFrame};

const BTN_LEFT: u32 = 0x110;
const BTN_MIDDLE: u32 = 0x112;
/// Tiles uploaded per frame while loading (≈ 64 MB at 512 KB/tile).
const UPLOADS_PER_FRAME: usize = 128;
const BENCH_FRAMES: u32 = 900;
const PALETTE: [[f32; 3]; 6] = [
    [0.02, 0.02, 0.025],
    [0.80, 0.05, 0.03],
    [0.03, 0.20, 0.75],
    [0.05, 0.45, 0.10],
    [0.95, 0.60, 0.05],
    [1.0, 1.0, 1.0],
];

#[derive(Debug, Clone, Copy)]
enum Drag {
    None,
    Paint,
    Pan { last: DVec2 },
    Rotate { last: DVec2 },
    Zoom { last: DVec2, anchor: DVec2 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Button {
    Primary,
    Middle,
}

struct Bench {
    frame: u32,
    base: View,
    last: Option<Instant>,
    cpu_ms: Vec<f64>,
    interval_ms: Vec<f64>,
    exit_after: bool,
}

pub struct App {
    // Field order matters for drop order: the wgpu surface must go before the window.
    surface: wgpu::Surface<'static>,
    canvas: Option<Canvas>,
    gpu: GpuContext,
    surface_config: wgpu::SurfaceConfiguration,

    qh: QueueHandle<App>,
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    compositor: CompositorState,
    shm: Shm,
    window: Window,
    pointer: Option<ThemedPointer>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    tablet_manager: Option<ZwpTabletManagerV2>,
    tablet_seats: Vec<ZwpTabletSeatV2>,
    tablet_seat_ids: HashSet<wayland_client::backend::ObjectId>,
    viewport: Option<WpViewport>,
    _fractional: Option<WpFractionalScaleV1>,

    // Window state
    logical: (u32, u32),
    scale: f64,
    configured: bool,
    needs_redraw: bool,
    frame_pending: bool,
    exit: bool,

    // Document & view
    doc_name: String,
    view: View,
    view_fitted: bool,
    pending_uploads: VecDeque<(TileCoord, Vec<f16>)>,
    loading: bool,
    capacity_warned: bool,

    // Input
    mods: Modifiers,
    space: bool,
    r_key: bool,
    cursor: Option<DVec2>,
    drag: Drag,
    stroke: Option<Stroke>,
    brush: BrushSettings,
    palette_idx: usize,
    tablet_eraser: bool,
    tablet_contact: bool,

    // Stats
    bench: Option<Bench>,
    bench_on_load: bool,
    title_at: Instant,
    frame_cpu_ms: f64,
}

pub fn run(source: Source, bench: bool) -> anyhow::Result<()> {
    let conn = Connection::connect_to_env().context("keine Wayland-Verbindung (WAYLAND_DISPLAY?)")?;
    let (globals, event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();
    let mut event_loop: EventLoop<App> = EventLoop::try_new()?;
    WaylandSource::new(conn.clone(), event_queue)
        .insert(event_loop.handle())
        .map_err(|e| anyhow::anyhow!("wayland source: {e}"))?;

    let compositor = CompositorState::bind(&globals, &qh).context("wl_compositor fehlt")?;
    let xdg = XdgShell::bind(&globals, &qh).context("xdg_shell fehlt")?;
    let shm = Shm::bind(&globals, &qh).context("wl_shm fehlt")?;

    let wl_surface = compositor.create_surface(&qh);
    let window = xdg.create_window(wl_surface, WindowDecorations::RequestServer, &qh);
    window.set_title("imagine");
    window.set_app_id("io.github.imagine");
    window.set_min_size(Some((320, 240)));

    // Fractional scaling (wp_fractional_scale + wp_viewporter) where available.
    let viewporter: Option<WpViewporter> = globals.bind(&qh, 1..=1, tablet::Ignore).ok();
    let frac_mgr: Option<WpFractionalScaleManagerV1> = globals.bind(&qh, 1..=1, tablet::Ignore).ok();
    let (viewport, fractional) = match (viewporter, frac_mgr) {
        (Some(vp), Some(fm)) => (
            Some(vp.get_viewport(window.wl_surface(), &qh, tablet::Ignore)),
            Some(fm.get_fractional_scale(window.wl_surface(), &qh, FractionalData)),
        ),
        _ => (None, None),
    };
    let tablet_manager: Option<ZwpTabletManagerV2> = globals.bind(&qh, 1..=1, tablet::Ignore).ok();
    if tablet_manager.is_none() {
        tracing::warn!("Compositor unterstützt kein tablet-v2 – Stift funktioniert nur als Maus");
    }
    window.commit();

    // wgpu on the raw Wayland surface.
    let instance = GpuContext::create_instance();
    let raw_display = RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
        NonNull::new(conn.backend().display_ptr() as *mut _).context("display ptr")?,
    ));
    let raw_window = RawWindowHandle::Wayland(WaylandWindowHandle::new(
        NonNull::new(window.wl_surface().id().as_ptr() as *mut _).context("surface ptr")?,
    ));
    // SAFETY: display and surface outlive the wgpu surface (dropped first, see `App` field order).
    let surface = unsafe {
        instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(raw_display),
            raw_window_handle: raw_window,
        })?
    };
    let gpu = pollster::block_on(GpuContext::new(instance, Some(&surface)))?;
    let caps = surface.get_capabilities(&gpu.adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else {
        wgpu::PresentMode::Fifo
    };
    tracing::info!("Swapchain: {format:?}, {present_mode:?}");
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        color_space: Default::default(),
        width: 1,
        height: 1,
        present_mode,
        desired_maximum_frame_latency: 1,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
    };

    let (tx, rx) = channel::channel::<LoaderMsg>();
    event_loop
        .handle()
        .insert_source(rx, |ev, _, app: &mut App| {
            if let channel::Event::Msg(m) = ev {
                app.on_loader(m);
            }
        })
        .map_err(|e| anyhow::anyhow!("loader channel: {e}"))?;
    loader::spawn(source, tx);

    let mut app = App {
        surface,
        canvas: None,
        gpu,
        surface_config,
        qh: qh.clone(),
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        compositor,
        shm,
        window,
        pointer: None,
        keyboard: None,
        tablet_manager,
        tablet_seats: Vec::new(),
        tablet_seat_ids: HashSet::new(),
        viewport,
        _fractional: fractional,
        logical: (1600, 1000),
        scale: 1.0,
        configured: false,
        needs_redraw: true,
        frame_pending: false,
        exit: false,
        doc_name: String::new(),
        view: View::default(),
        view_fitted: false,
        pending_uploads: VecDeque::new(),
        loading: true,
        capacity_warned: false,
        mods: Modifiers::default(),
        space: false,
        r_key: false,
        cursor: None,
        drag: Drag::None,
        stroke: None,
        brush: BrushSettings::default(),
        palette_idx: 0,
        tablet_eraser: false,
        tablet_contact: false,
        bench: None,
        bench_on_load: bench,
        title_at: Instant::now(),
        frame_cpu_ms: 0.0,
    };
    let seats: Vec<_> = app.seat_state.seats().collect();
    for seat in seats {
        app.bind_tablet_seat(&seat);
    }

    while !app.exit {
        event_loop.dispatch(None, &mut app)?;
        if app.configured && app.needs_redraw && !app.frame_pending {
            app.draw();
        }
    }
    Ok(())
}

impl App {
    fn bind_tablet_seat(&mut self, seat: &wl_seat::WlSeat) {
        let Some(mgr) = &self.tablet_manager else { return };
        if self.tablet_seat_ids.insert(seat.id()) {
            self.tablet_seats.push(mgr.get_tablet_seat(seat, &self.qh, tablet::SeatData));
        }
    }

    fn physical_size(&self) -> (u32, u32) {
        let (w, h) = self.logical;
        (
            ((w as f64 * self.scale).round() as u32).max(1),
            ((h as f64 * self.scale).round() as u32).max(1),
        )
    }

    fn reconfigure_surface(&mut self) {
        let (pw, ph) = self.physical_size();
        self.surface_config.width = pw;
        self.surface_config.height = ph;
        self.surface.configure(&self.gpu.device, &self.surface_config);
        match &self.viewport {
            Some(vp) => vp.set_destination(self.logical.0 as i32, self.logical.1 as i32),
            None => self.window.wl_surface().set_buffer_scale(self.scale.round() as i32),
        }
        self.view.viewport = DVec2::new(pw as f64, ph as f64);
        self.fit_if_needed();
        self.needs_redraw = true;
    }

    fn set_scale(&mut self, scale: f64) {
        if (scale - self.scale).abs() > 1e-6 {
            // Keep the view centred on the same document point at the same physical zoom ratio.
            self.view.zoom *= scale / self.scale;
            self.scale = scale;
            if self.configured {
                self.reconfigure_surface();
            }
        }
    }

    fn fit_if_needed(&mut self) {
        if !self.view_fitted && self.configured
            && let Some(c) = &self.canvas {
                let (w, h) = c.size();
                self.view.fit(w, h);
                self.view_fitted = true;
            }
    }

    fn to_physical(&self, (x, y): (f64, f64)) -> DVec2 {
        DVec2::new(x, y) * self.scale
    }

    // ---- loader -----------------------------------------------------------

    fn on_loader(&mut self, msg: LoaderMsg) {
        match msg {
            LoaderMsg::Size { width, height, name, expected_tiles } => {
                tracing::info!("Dokument {name}: {width}×{height}");
                let mut canvas = Canvas::new(
                    &self.gpu.device,
                    &self.gpu.queue,
                    self.surface_config.format,
                    width,
                    height,
                );
                if expected_tiles > canvas.max_tiles() {
                    tracing::warn!(
                        "Bild braucht {expected_tiles} Kacheln, GPU-Array fasst {} – Rest wird nicht geladen (Phase-0-Limit)",
                        canvas.max_tiles()
                    );
                }
                canvas.reserve(expected_tiles);
                self.canvas = Some(canvas);
                self.doc_name = name;
                self.fit_if_needed();
                self.needs_redraw = true;
            }
            LoaderMsg::Tiles(tiles) => {
                self.pending_uploads.extend(tiles);
                self.needs_redraw = true;
            }
            LoaderMsg::Done { decode_ms, total_ms } => {
                tracing::info!("Laden fertig: Dekodieren {decode_ms:.0} ms, gesamt {total_ms:.0} ms");
                self.loading = false;
                self.needs_redraw = true;
            }
            LoaderMsg::Error(e) => {
                tracing::error!("Laden fehlgeschlagen: {e}");
                self.loading = false;
                if self.canvas.is_none() {
                    self.exit = true;
                }
            }
        }
    }

    // ---- input ------------------------------------------------------------

    fn press(&mut self, pos: DVec2, pressure: f32, button: Button, eraser_tool: bool) {
        self.cursor = Some(pos);
        self.drag = match button {
            Button::Middle => Drag::Pan { last: pos },
            Button::Primary if self.space && self.mods.ctrl => Drag::Zoom { last: pos, anchor: pos },
            Button::Primary if self.space => Drag::Pan { last: pos },
            Button::Primary if self.r_key => Drag::Rotate { last: pos },
            Button::Primary => {
                let mut settings = self.brush;
                settings.eraser |= eraser_tool;
                let mut stroke = Stroke::new(settings);
                let dabs = stroke.add(InputSample { pos: self.view.screen_to_doc(pos), pressure });
                if let Some(c) = &mut self.canvas {
                    c.paint(&dabs, settings.eraser);
                }
                self.stroke = Some(stroke);
                Drag::Paint
            }
        };
        self.needs_redraw = true;
    }

    fn motion(&mut self, pos: DVec2, pressure: f32) {
        self.cursor = Some(pos);
        match &mut self.drag {
            Drag::None => {}
            Drag::Paint => {
                if let (Some(stroke), Some(canvas)) = (&mut self.stroke, &mut self.canvas) {
                    let dabs = stroke.add(InputSample { pos: self.view.screen_to_doc(pos), pressure });
                    canvas.paint(&dabs, stroke.settings.eraser);
                }
            }
            Drag::Pan { last } => {
                let d = pos - *last;
                *last = pos;
                self.view.pan(d);
            }
            Drag::Rotate { last } => {
                let c = self.view.viewport * 0.5;
                let (a, b) = (*last - c, pos - c);
                *last = pos;
                if a.length() > 4.0 && b.length() > 4.0 {
                    self.view.rotate_at(c, b.y.atan2(b.x) - a.y.atan2(a.x));
                }
            }
            Drag::Zoom { last, anchor } => {
                let f = ((pos.x - last.x) * 0.01).exp();
                *last = pos;
                let anchor = *anchor;
                self.view.zoom_at(anchor, f);
            }
        }
        self.needs_redraw = true;
    }

    fn release(&mut self, pos: DVec2, pressure: f32) {
        if let Drag::Paint = self.drag
            && let (Some(mut stroke), Some(canvas)) = (self.stroke.take(), &mut self.canvas) {
                let dabs = stroke.finish(InputSample { pos: self.view.screen_to_doc(pos), pressure });
                canvas.paint(&dabs, stroke.settings.eraser);
            }
        self.drag = Drag::None;
        self.needs_redraw = true;
    }

    pub fn on_tablet_frame(&mut self, f: TabletFrame) {
        let pos = self.to_physical(f.pos);
        if f.proximity == Some(false) {
            if self.tablet_contact {
                self.release(pos, 0.0);
                self.tablet_contact = false;
            }
            self.cursor = None;
            self.needs_redraw = true;
            return;
        }
        self.tablet_eraser = f.is_eraser;
        for (button, pressed) in &f.buttons {
            match (*button, *pressed) {
                (tablet::BTN_STYLUS, true) => self.drag = Drag::Pan { last: pos },
                (tablet::BTN_STYLUS2, true) => self.drag = Drag::Rotate { last: pos },
                (tablet::BTN_STYLUS | tablet::BTN_STYLUS2, false) => self.drag = Drag::None,
                _ => {}
            }
        }
        match f.contact {
            Some(true) if matches!(self.drag, Drag::None) => {
                self.tablet_contact = true;
                self.press(pos, f.pressure, Button::Primary, f.is_eraser);
            }
            Some(false) => {
                self.tablet_contact = false;
                self.release(pos, f.pressure);
            }
            _ => self.motion(pos, f.pressure),
        }
    }

    fn key(&mut self, sym: Keysym, pressed: bool) {
        match sym {
            Keysym::space => self.space = pressed,
            Keysym::r => self.r_key = pressed,
            _ => {}
        }
        if !pressed {
            return;
        }
        let center = self.view.viewport * 0.5;
        let anchor = self.cursor.unwrap_or(center);
        match sym {
            Keysym::q | Keysym::Q if self.mods.ctrl => self.exit = true,
            Keysym::_0 => {
                self.view_fitted = false;
                self.fit_if_needed();
            }
            Keysym::_1 => self.view.set_zoom_at(anchor, 1.0),
            Keysym::R => {
                let r = self.view.rotation;
                self.view.rotate_at(center, -r);
            }
            Keysym::bracketleft => self.brush.size = (self.brush.size / 1.2).max(1.0),
            Keysym::bracketright => self.brush.size = (self.brush.size * 1.2).min(5000.0),
            Keysym::e => self.brush.eraser = !self.brush.eraser,
            Keysym::b => self.brush.eraser = false,
            Keysym::c => {
                self.palette_idx = (self.palette_idx + 1) % PALETTE.len();
                self.brush.color = PALETTE[self.palette_idx];
            }
            Keysym::h => {
                let steps = [0.0, 0.5, 0.8, 1.0];
                let i = steps.iter().position(|&h| h >= self.brush.hardness - 1e-3).unwrap_or(0);
                self.brush.hardness = steps[(i + 1) % steps.len()];
                tracing::info!("Härte {:.0} %", self.brush.hardness * 100.0);
            }
            Keysym::s => {
                let steps = [0.0, 0.4, 0.7, 0.85];
                let i = steps.iter().position(|&s| s >= self.brush.smoothing - 1e-3).unwrap_or(0);
                self.brush.smoothing = steps[(i + 1) % steps.len()];
                tracing::info!("Glättung {:.0} %", self.brush.smoothing * 100.0);
            }
            Keysym::F1 => print!("{}", crate::HELP),
            Keysym::F12 => self.start_bench(false),
            _ => return,
        }
        self.needs_redraw = true;
    }

    // ---- rendering --------------------------------------------------------

    fn start_bench(&mut self, exit_after: bool) {
        if self.canvas.is_none() {
            return;
        }
        tracing::info!("Benchmark: {BENCH_FRAMES} Frames Zoom/Rotation/Pan …");
        self.view_fitted = false;
        self.fit_if_needed();
        self.bench = Some(Bench {
            frame: 0,
            base: self.view,
            last: None,
            cpu_ms: Vec::with_capacity(BENCH_FRAMES as usize),
            interval_ms: Vec::with_capacity(BENCH_FRAMES as usize),
            exit_after,
        });
        self.needs_redraw = true;
    }

    fn step_bench(&mut self) {
        let Some(b) = &mut self.bench else { return };
        let t = b.frame as f64 / BENCH_FRAMES as f64 * std::f64::consts::TAU;
        let mut v = b.base;
        v.zoom = b.base.zoom * 2f64.powf((1.0 - t.cos()) * 3.5); // fit → ~128× fit
        v.rotation = t * 0.5;
        v.center = b.base.center + DVec2::new(t.sin(), (2.0 * t).sin() * 0.5) * b.base.viewport / b.base.zoom * 0.25;
        self.view = v;
    }

    fn finish_bench(&mut self) {
        let Some(b) = self.bench.take() else { return };
        let stat = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            let avg = v.iter().sum::<f64>() / v.len().max(1) as f64;
            let pct = |p: f64| v.get(((v.len() as f64 - 1.0) * p) as usize).copied().unwrap_or(0.0);
            (avg, pct(0.5), pct(0.99), v.last().copied().unwrap_or(0.0))
        };
        let (mut cpu, mut iv) = (b.cpu_ms, b.interval_ms);
        let c = stat(&mut cpu);
        let i = stat(&mut iv);
        let tiles = self.canvas.as_ref().map_or(0, |c| c.tile_count());
        println!(
            "\n=== Benchmark ({} Frames, {}×{} px, {tiles} Kacheln) ===\n\
             CPU pro Frame:    Ø {:.2} ms   p50 {:.2}   p99 {:.2}   max {:.2}\n\
             Frame-Abstand:    Ø {:.2} ms   p50 {:.2}   p99 {:.2}   max {:.2}   (≈ {:.0} fps)\n",
            cpu.len(),
            self.surface_config.width,
            self.surface_config.height,
            c.0, c.1, c.2, c.3,
            i.0, i.1, i.2, i.3,
            1000.0 / i.0.max(1e-6),
        );
        self.view = b.base;
        if b.exit_after {
            self.exit = true;
        }
    }

    fn draw(&mut self) {
        let t0 = Instant::now();
        if self.bench_on_load && !self.loading && self.pending_uploads.is_empty() {
            self.bench_on_load = false;
            self.start_bench(true);
        }
        self.step_bench();

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure_surface();
                return;
            }
            other => {
                tracing::debug!("kein Swapchain-Bild: {:?}", std::mem::discriminant(&other));
                return;
            }
        };
        let target = frame.texture.create_view(&Default::default());

        if let Some(canvas) = &mut self.canvas {
            for _ in 0..UPLOADS_PER_FRAME {
                let Some((c, px)) = self.pending_uploads.pop_front() else { break };
                if !canvas.upload_tile(c, &px) && !self.capacity_warned {
                    tracing::warn!("Kachel {c:?} passt nicht mehr ins GPU-Array");
                    self.capacity_warned = true;
                }
            }
            let show_ring = !matches!(self.drag, Drag::Pan { .. } | Drag::Rotate { .. } | Drag::Zoom { .. })
                && !self.space
                && self.bench.is_none();
            let overlay = Overlay {
                cursor: self
                    .cursor
                    .filter(|_| show_ring)
                    .map(|p| (p, (self.brush.size * 0.5) * self.view.zoom as f32)),
                checker_px: (8.0 * self.scale) as f32,
            };
            canvas.render(&target, &self.view, &overlay);
        } else {
            clear(&self.gpu, &target);
        }

        let surface = self.window.wl_surface();
        surface.frame(&self.qh, FrameCallbackData(surface.clone()));
        self.frame_pending = true;
        self.gpu.queue.present(frame);

        self.frame_cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
        let now = Instant::now();
        if let Some(b) = &mut self.bench {
            b.cpu_ms.push(self.frame_cpu_ms);
            if let Some(l) = b.last {
                b.interval_ms.push((now - l).as_secs_f64() * 1e3);
            }
            b.last = Some(now);
            b.frame += 1;
            if b.frame >= BENCH_FRAMES {
                self.finish_bench();
            }
        }

        self.needs_redraw = !self.pending_uploads.is_empty()
            || self.bench.is_some()
            || (self.bench_on_load && !self.loading)
            || self.canvas.as_ref().is_some_and(|c| c.has_pending_work());
        if now.duration_since(self.title_at).as_millis() > 250 {
            self.title_at = now;
            self.update_title();
        }
    }

    fn update_title(&self) {
        let Some(c) = &self.canvas else { return };
        let (w, h) = c.size();
        let tool = if self.brush.eraser { "Radierer" } else { "Pinsel" };
        let loading = if self.loading || !self.pending_uploads.is_empty() { " – lädt …" } else { "" };
        self.window.set_title(format!(
            "{} – {w}×{h} – {:.1} % – {:.0}° – {tool} {:.0} px – {} Kacheln ({:.0} MB VRAM) – {:.2} ms{loading}",
            self.doc_name,
            self.view.zoom * 100.0 / self.scale,
            self.view.rotation.to_degrees(),
            self.brush.size,
            c.tile_count(),
            c.gpu_bytes() as f64 / 1e6,
            self.frame_cpu_ms,
        ));
    }
}

fn clear(gpu: &GpuContext, target: &wgpu::TextureView) {
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.035, g: 0.037, b: 0.042, a: 1.0 }),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    gpu.queue.submit([enc.finish()]);
}

// ---- fractional scale -------------------------------------------------------

pub struct FractionalData;

impl Dispatch2<WpFractionalScaleV1, App> for FractionalData {
    fn event(
        &self,
        app: &mut App,
        _: &WpFractionalScaleV1,
        event: wp_fractional_scale_v1::Event,
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            app.set_scale(scale as f64 / 120.0);
        }
    }
}

impl Dispatch2<WpViewporter, App> for tablet::Ignore {
    fn event(&self, _: &mut App, _: &WpViewporter, _: <WpViewporter as Proxy>::Event, _: &Connection, _: &QueueHandle<App>) {}
}
impl Dispatch2<WpViewport, App> for tablet::Ignore {
    fn event(&self, _: &mut App, _: &WpViewport, _: <WpViewport as Proxy>::Event, _: &Connection, _: &QueueHandle<App>) {}
}
impl Dispatch2<WpFractionalScaleManagerV1, App> for tablet::Ignore {
    fn event(
        &self,
        _: &mut App,
        _: &WpFractionalScaleManagerV1,
        _: <WpFractionalScaleManagerV1 as Proxy>::Event,
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
    }
}

// ---- SCTK handlers ----------------------------------------------------------

impl CompositorHandler for App {
    fn scale_factor_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, factor: i32) {
        if self.viewport.is_none() {
            self.set_scale(factor as f64);
        }
    }

    fn transform_changed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: wl_output::Transform) {}

    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {
        self.frame_pending = false;
    }

    fn surface_enter(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}

    fn surface_leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: &wl_output::WlOutput) {}
}

impl OutputHandler for App {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl WindowHandler for App {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {
        self.exit = true;
    }

    fn configure(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window, configure: WindowConfigure, _: u32) {
        let (w, h) = configure.new_size;
        let w = w.map_or(self.logical.0, |v| v.get());
        let h = h.map_or(self.logical.1, |v| v.get());
        let first = !self.configured;
        self.configured = true;
        if first || (w, h) != self.logical {
            self.logical = (w, h);
            self.reconfigure_surface();
        }
        // The first configure must be answered with a buffer.
        if first {
            self.frame_pending = false;
        }
        self.needs_redraw = true;
    }
}

impl SeatHandler for App {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, seat: wl_seat::WlSeat) {
        self.bind_tablet_seat(&seat);
    }

    fn new_capability(&mut self, _: &Connection, qh: &QueueHandle<Self>, seat: wl_seat::WlSeat, cap: Capability) {
        if cap == Capability::Keyboard && self.keyboard.is_none() {
            self.keyboard = self.seat_state.get_keyboard(qh, &seat, None).ok();
        }
        if cap == Capability::Pointer && self.pointer.is_none() {
            let cursor_surface = self.compositor.create_surface(qh);
            self.pointer = self
                .seat_state
                .get_pointer_with_theme::<_, ()>(qh, &seat, self.shm.wl_shm(), cursor_surface, ThemeSpec::default())
                .ok();
        }
    }

    fn remove_capability(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat, cap: Capability) {
        if cap == Capability::Keyboard
            && let Some(k) = self.keyboard.take() {
                k.release();
            }
        if cap == Capability::Pointer
            && let Some(p) = self.pointer.take() {
                p.pointer().release();
            }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for App {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
    }

    fn leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_keyboard::WlKeyboard, _: &wl_surface::WlSurface, _: u32) {
        // Keys released while unfocused never reach us.
        self.space = false;
        self.r_key = false;
    }

    fn press_key(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_keyboard::WlKeyboard, _: u32, ev: KeyEvent) {
        self.key(ev.keysym, true);
    }

    fn repeat_key(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_keyboard::WlKeyboard, _: u32, _: KeyEvent) {}

    fn release_key(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_keyboard::WlKeyboard, _: u32, ev: KeyEvent) {
        self.key(ev.keysym, false);
    }

    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        mods: Modifiers,
        _: RawModifiers,
        _: u32,
    ) {
        self.mods = mods;
    }
}

impl PointerHandler for App {
    fn pointer_frame(&mut self, conn: &Connection, _: &QueueHandle<Self>, _: &wl_pointer::WlPointer, events: &[PointerEvent]) {
        for ev in events {
            if &ev.surface != self.window.wl_surface() {
                continue;
            }
            let pos = self.to_physical(ev.position);
            match &ev.kind {
                PointerEventKind::Enter { .. } => {
                    if let Some(p) = &self.pointer {
                        let _ = p.set_cursor(conn, CursorIcon::Crosshair);
                    }
                    self.cursor = Some(pos);
                    self.needs_redraw = true;
                }
                PointerEventKind::Leave { .. } => {
                    if !matches!(self.drag, Drag::None) && !self.tablet_contact {
                        self.release(pos, 1.0);
                    }
                    self.cursor = None;
                    self.needs_redraw = true;
                }
                PointerEventKind::Motion { .. } => {
                    // Tablets also emulate pointer events; ignore those while a stylus is active.
                    if !self.tablet_contact {
                        self.motion(pos, 1.0);
                    }
                }
                PointerEventKind::Press { button, .. } if !self.tablet_contact => match *button {
                    BTN_LEFT => self.press(pos, 1.0, Button::Primary, false),
                    BTN_MIDDLE => self.press(pos, 1.0, Button::Middle, false),
                    _ => {}
                },
                PointerEventKind::Release { button, .. } if !self.tablet_contact => {
                    if matches!(*button, BTN_LEFT | BTN_MIDDLE) {
                        self.release(pos, 1.0);
                    }
                }
                PointerEventKind::Axis { vertical, .. } => {
                    let steps = if vertical.value120 != 0 {
                        vertical.value120 as f64 / 120.0
                    } else if vertical.discrete != 0 {
                        vertical.discrete as f64
                    } else {
                        vertical.absolute / 15.0
                    };
                    if steps == 0.0 {
                        continue;
                    }
                    if self.mods.ctrl {
                        self.brush.size = (self.brush.size * 1.1f32.powf(-steps as f32)).clamp(1.0, 5000.0);
                    } else if self.mods.shift {
                        self.view.rotate_at(pos, (steps * 7.5).to_radians());
                    } else {
                        self.view.zoom_at(pos, 1.2f64.powf(-steps));
                    }
                    self.cursor = Some(pos);
                    self.needs_redraw = true;
                }
                _ => {}
            }
        }
    }
}

impl ShmHandler for App {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_registry!(App);

impl ProvidesRegistryState for App {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

smithay_client_toolkit::delegate_dispatch2!(App);
