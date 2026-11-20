//! Window, event loop, input handling and frame scheduling.
//!
//! Wayland is used directly through smithay-client-toolkit. Frames are driven
//! by `wl_surface.frame` callbacks: input is batched per event-loop dispatch
//! and at most one frame is rendered per compositor refresh.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use anyhow::Context;
use glam::{DAffine2, DVec2};
use imagine_core::color::linear_to_srgb;
use imagine_core::mask::{CombineOp, CpuMask, Shape};
use imagine_core::{BlendSpace, BrushSettings, InputSample, LayerId, Stroke, View};
use imagine_gpu::{Engine, GpuContext, GradientKind, Interp, Overlay, SampleFrom, OverlayLine, StrokeParams, StrokeTarget};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, FrameCallbackData},
    delegate_registry,
    dispatch2::Dispatch2,
    output::{OutputHandler, OutputState},
    reexports::{
        calloop::{EventLoop, Interest, Mode, PostAction, channel, generic::Generic},
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

use crate::agent::{command, ipc};
use crate::boxedit::{self, BoxEdit};
use crate::settings::{self, Settings};
use crate::ui::{self, panels::{Action, ToolId}};
use crate::loader::{self, LoaderMsg, Source, TileData};
use crate::session::{ExportJob, SaveDone, SaveJob, Session};
use crate::tablet::{self, TabletFrame};

const BTN_LEFT: u32 = 0x110;
const BTN_MIDDLE: u32 = 0x112;
const BTN_RIGHT: u32 = 0x111;
/// evdev codes of the digit row: KEY_1 = 2 … KEY_9 = 10, KEY_0 = 11.
const KEY_1: u32 = 2;
const KEY_0: u32 = 11;
/// Tiles uploaded per frame while loading (≈ 64 MB at 512 KB/tile).
const UPLOADS_PER_FRAME: usize = 128;
const BENCH_FRAMES: u32 = 900;
const QUIT_CONFIRM: Duration = Duration::from_secs(3);
const PALETTE: [[f32; 3]; 6] = [
    [0.02, 0.02, 0.025],
    [0.80, 0.05, 0.03],
    [0.03, 0.20, 0.75],
    [0.05, 0.45, 0.10],
    [0.95, 0.60, 0.05],
    [1.0, 1.0, 1.0],
];

#[derive(Debug, Clone)]
enum Drag {
    None,
    Paint,
    Pan { last: DVec2 },
    Rotate { last: DVec2 },
    Zoom { last: DVec2, anchor: DVec2 },
    /// Document coordinates.
    Marquee { start: DVec2, end: DVec2, op: CombineOp, ellipse: bool },
    Lasso { points: Vec<DVec2>, op: CombineOp },
    Gradient { start: DVec2, end: DVec2 },
    Move { start: DVec2, end: DVec2 },
    /// Dragging a free-transform or crop handle.
    Box,
    /// Alt + right drag: brush size (horizontal) and hardness (vertical), shown at `origin`.
    BrushHud { origin: DVec2, start: DVec2, size: f32, hardness: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tool {
    Brush,
    Eraser,
    Marquee { ellipse: bool },
    Lasso,
    Gradient { radial: bool },
    Move,
    Crop,
    Wand,
    Smudge,
}

impl Tool {
    fn name(self) -> &'static str {
        match self {
            Tool::Brush => "Pinsel",
            Tool::Eraser => "Radierer",
            Tool::Marquee { ellipse: false } => "Auswahlrechteck",
            Tool::Marquee { ellipse: true } => "Auswahlellipse",
            Tool::Lasso => "Lasso",
            Tool::Gradient { radial: false } => "Verlauf linear",
            Tool::Gradient { radial: true } => "Verlauf radial",
            Tool::Move => "Verschieben",
            Tool::Crop => "Zuschneiden",
            Tool::Wand => "Zauberstab",
            Tool::Smudge => "Wischfinger",
        }
    }

    fn paints(self) -> bool {
        matches!(self, Tool::Brush | Tool::Eraser | Tool::Smudge)
    }
}

/// Result of a background job.
enum BgDone {
    /// A new selection computed from the one identified by `base`.
    Selection { label: &'static str, base: usize, mask: anyhow::Result<CpuMask>, started: Instant },
    Exported { path: std::path::PathBuf, result: anyhow::Result<f64> },
    Autosaved { path: std::path::PathBuf, result: anyhow::Result<()> },
}

/// An action that replaces or closes the document, held back while the
/// user decides about unsaved changes.
#[derive(Debug, Clone)]
enum Pending {
    Quit,
    Open(std::path::PathBuf),
    New { w: u32, h: u32, white: bool },
}

/// Recovery copies: `~/.local/state/imagine/autosave/` (or `$XDG_STATE_HOME`).
fn autosave_dir() -> std::path::PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/state")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("imagine/autosave")
}

/// Mentions recovery copies left behind by windows that no longer run.
fn report_orphaned_autosaves() {
    let Ok(dir) = std::fs::read_dir(autosave_dir()) else { return };
    for e in dir.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let pid = name.rsplit_once('-').and_then(|(_, p)| p.strip_suffix(".imagine")).and_then(|p| p.parse::<u32>().ok());
        if pid.is_some_and(|p| !std::path::Path::new(&format!("/proc/{p}")).exists()) {
            tracing::warn!("Wiederherstellungskopie gefunden: imagine {}", e.path().display());
        }
    }
}

const AUTOSAVE_EVERY: Duration = Duration::from_secs(180);

#[derive(Debug, Clone, Copy, PartialEq)]
enum BoxKind {
    Transform,
    Crop,
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
    session: Option<Session>,
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
    quit_armed: Option<Instant>,

    // Document & view
    view: View,
    view_fitted: bool,
    pending_uploads: VecDeque<(LayerId, TileData)>,
    loading: bool,
    upload_failures: usize,
    modified: bool,

    // Input
    mods: Modifiers,
    space: bool,
    r_key: bool,
    cursor: Option<DVec2>,
    drag: Drag,
    stroke: Option<Stroke>,
    brush: BrushSettings,
    palette_idx: usize,
    tablet_contact: bool,
    tool: Tool,
    /// Paint into the active layer's mask instead of its pixels.
    mask_edit: bool,
    quick_mask: bool,
    /// Free transform (Strg+T) or crop box being edited.
    boxed: Option<(BoxKind, BoxEdit)>,
    wand_tolerance: f32,
    smudge_strength: f32,
    wand_contiguous: bool,
    started: Instant,

    // Saving
    save_job: Option<SaveJob>,
    saving: bool,
    save_tx: channel::Sender<anyhow::Result<SaveDone>>,
    /// CPU-heavy work (feathering, wand fill, export encoding) off the UI thread.
    bg_tx: channel::Sender<BgDone>,
    bg_running: usize,
    export_job: Option<ExportJob>,
    autosave_job: Option<(std::path::PathBuf, SaveJob)>,
    autosave_path: Option<std::path::PathBuf>,
    autosave_mark: u64,
    autosave_at: Instant,
    edit_count: u64,
    save_mark: u64,

    // Interface
    gui: ui::Gui,
    loader_tx: channel::Sender<LoaderMsg>,
    settings: Settings,
    /// Waiting for the unsaved-changes answer (or for the save it triggered).
    pending: Option<Pending>,
    after_save: Option<Pending>,
    fill_white_on_load: bool,
    ever_loaded: bool,
    ui_state: ui::panels::State,
    /// Canvas area not covered by panels (physical px).
    canvas_rect: (DVec2, DVec2),
    /// The current mouse press belongs to the interface.
    ui_pointer: bool,
    tablet_ui: bool,
    ui_log_gen: usize,
    last_model: Option<ui::panels::Model>,
    /// egui textures of the thumbnail slots.
    thumb_ids: HashMap<u32, egui::TextureId>,
    /// Status text, refreshed a few times per second (not every frame).
    ui_status: String,

    // Agents
    socket: Option<ipc::SocketGuard>,

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
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
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
    let loader_tx = tx.clone();
    loader::spawn(source, tx);
    let (save_tx, save_rx) = channel::channel::<anyhow::Result<SaveDone>>();
    event_loop
        .handle()
        .insert_source(save_rx, |ev, _, app: &mut App| {
            if let channel::Event::Msg(r) = ev {
                app.on_save_done(r);
            }
        })
        .map_err(|e| anyhow::anyhow!("save channel: {e}"))?;

    let (bg_tx, bg_rx) = channel::channel::<BgDone>();
    event_loop
        .handle()
        .insert_source(bg_rx, |ev, _, app: &mut App| {
            if let channel::Event::Msg(r) = ev {
                app.on_background_done(r);
            }
        })
        .map_err(|e| anyhow::anyhow!("background channel: {e}"))?;

    let gui = ui::Gui::new(&gpu.device, surface_config.format);
    let mut app = App {
        surface,
        session: None,
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
        quit_armed: None,
        view: View::default(),
        view_fitted: false,
        pending_uploads: VecDeque::new(),
        loading: true,
        upload_failures: 0,
        modified: false,
        mods: Modifiers::default(),
        space: false,
        r_key: false,
        cursor: None,
        drag: Drag::None,
        stroke: None,
        brush: BrushSettings::default(),
        palette_idx: 0,
        tablet_contact: false,
        tool: Tool::Brush,
        mask_edit: false,
        quick_mask: false,
        boxed: None,
        wand_tolerance: 32.0,
        smudge_strength: 0.7,
        wand_contiguous: true,
        started: Instant::now(),
        save_job: None,
        saving: false,
        save_tx,
        bg_tx,
        bg_running: 0,
        export_job: None,
        autosave_job: None,
        autosave_path: None,
        autosave_mark: 0,
        autosave_at: Instant::now(),
        edit_count: 0,
        save_mark: 0,
        gui,
        loader_tx,
        settings: Settings::load_from(&settings::config_path()),
        pending: None,
        after_save: None,
        fill_white_on_load: false,
        ever_loaded: false,
        ui_state: ui::panels::State::default(),
        canvas_rect: (DVec2::ZERO, DVec2::ZERO),
        ui_pointer: false,
        tablet_ui: false,
        ui_log_gen: 0,
        last_model: None,
        thumb_ids: HashMap::new(),
        ui_status: String::new(),
        socket: None,
        bench: None,
        bench_on_load: bench,
        title_at: Instant::now(),
        frame_cpu_ms: 0.0,
    };
    match ipc::bind() {
        Ok((listener, guard)) => {
            tracing::info!("Agenten-Socket: {} (imagine mcp --attach)", guard.path().display());
            app.socket = Some(guard);
            let handle = event_loop.handle();
            event_loop
                .handle()
                .insert_source(Generic::new(listener, Interest::READ, Mode::Level), move |_, listener, _app: &mut App| {
                    while let Ok((stream, _)) = listener.accept() {
                        if stream.set_nonblocking(true).is_err() {
                            continue;
                        }
                        let mut buf = Vec::new();
                        let _ = handle.insert_source(
                            Generic::new(stream, Interest::READ, Mode::Level),
                            move |_, stream, app: &mut App| Ok(app.agent_io(stream, &mut buf)),
                        );
                    }
                    Ok(PostAction::Continue)
                })
                .map_err(|e| anyhow::anyhow!("agent socket: {e}"))?;
        }
        Err(e) => tracing::warn!("Agenten-Socket nicht verfügbar: {e:#}"),
    }
    let seats: Vec<_> = app.seat_state.seats().collect();
    for seat in seats {
        app.bind_tablet_seat(&seat);
    }

    app.apply_settings();
    report_orphaned_autosaves();
    while !app.exit {
        // Wake up periodically while a save is in flight or ants are marching.
        let ants = app.session.as_ref().is_some_and(|s| s.engine.has_selection());
        let timeout = if app.save_job.is_some() || app.export_job.is_some() {
            Some(Duration::from_millis(10))
        } else if ants {
            Some(Duration::from_millis(100))
        } else if app.modified || app.autosave_job.is_some() {
            Some(Duration::from_secs(5))
        } else {
            None
        };
        // The interface may want a frame later (tooltips, fading messages).
        let ui_wait = app.gui.repaint_at.map(|t| t.saturating_duration_since(Instant::now()));
        let timeout = match (timeout, ui_wait) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        event_loop.dispatch(timeout, &mut app)?;
        app.poll_save();
        app.poll_export();
        app.maybe_autosave();
        if ants || app.gui.repaint_at.is_some_and(|t| t <= Instant::now()) || ui::log::generation() != app.ui_log_gen {
            app.needs_redraw = true;
        }
        if app.configured && app.needs_redraw && !app.frame_pending {
            app.draw();
        }
    }
    // A deliberate quit: the recovery copy is no longer needed.
    app.drop_autosave();
    app.store_settings();
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
        (((w as f64 * self.scale).round() as u32).max(1), ((h as f64 * self.scale).round() as u32).max(1))
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
            self.view.zoom *= scale / self.scale;
            self.scale = scale;
            if self.configured {
                self.reconfigure_surface();
            }
        }
    }

    fn fit_if_needed(&mut self) {
        if !self.view_fitted
            && self.configured
            && let Some(s) = &self.session
        {
            let (w, h) = s.engine.size();
            let (a, b) = self.canvas_rect;
            if b != DVec2::ZERO {
                self.view.fit_in(w, h, a, b);
            } else {
                self.view.fit(w, h);
            }
            self.view_fitted = true;
        }
    }

    fn to_physical(&self, (x, y): (f64, f64)) -> DVec2 {
        DVec2::new(x, y) * self.scale
    }

    fn engine(&mut self) -> Option<&mut Engine> {
        self.session.as_mut().map(|s| &mut s.engine)
    }

    fn print_layers(&self) {
        if let Some(s) = &self.session {
            let d = s.engine.doc();
            tracing::debug!("Ebenen (oben → unten), Blend-Raum: {}\n{}", d.blend_space.name(), d.describe());
        }
    }

    // ---- loader -----------------------------------------------------------

    fn on_loader(&mut self, msg: LoaderMsg) {
        match msg {
            LoaderMsg::Structure { doc, name, source, native_len, expected_tiles } => {
                tracing::info!("Dokument {name}: {}×{}", doc.width, doc.height);
                let engine = Engine::new(&self.gpu.device, &self.gpu.queue, self.surface_config.format, &doc);
                if expected_tiles > engine.max_layer_tiles() {
                    tracing::warn!(
                        "Dokument braucht {expected_tiles} Kacheln, der Ebenen-Speicher fasst {} – Rest fehlt",
                        engine.max_layer_tiles()
                    );
                }
                if let Some(p) = &source {
                    self.settings.add_recent(p);
                    self.store_settings();
                }
                self.session = Some(Session::new(engine, name, source, native_len));
                self.ever_loaded = true;
                self.view_fitted = false;
                self.fit_if_needed();
                self.print_layers();
                self.needs_redraw = true;
            }
            LoaderMsg::Tiles { layer, tiles } => {
                self.pending_uploads.extend(tiles.into_iter().map(|t| (layer, t)));
                self.needs_redraw = true;
            }
            LoaderMsg::Mask(m) => {
                if let Some(s) = &mut self.session
                    && let Err(e) = s.add_mask(m)
                {
                    tracing::warn!("Maske nicht geladen: {e:#}");
                }
                self.needs_redraw = true;
            }
            LoaderMsg::Done { decode_ms, total_ms } => {
                tracing::info!("Laden fertig: Dekodieren {decode_ms:.0} ms, gesamt {total_ms:.0} ms");
                self.loading = false;
                if std::mem::take(&mut self.fill_white_on_load)
                    && let Some(s) = &mut self.session
                {
                    let _ = s.engine.fill(StrokeTarget::Pixels, [1.0; 4], 1.0, "Weiß");
                    s.engine.clear_history();
                }
                self.needs_redraw = true;
            }
            LoaderMsg::Error(e) => {
                tracing::error!("Laden fehlgeschlagen: {e}");
                self.loading = false;
                self.fill_white_on_load = false;
                if !self.ever_loaded {
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
            Button::Primary if self.boxed.is_some() || self.tool == Tool::Crop => self.begin_box(pos),
            Button::Primary => self.begin_tool(pos, pressure, eraser_tool),
        };
        self.needs_redraw = true;
    }

    /// Selection combine mode from the modifiers held when a selection starts (Photoshop).
    fn combine_op(&self) -> CombineOp {
        match (self.mods.shift, self.mods.alt) {
            (true, true) => CombineOp::Intersect,
            (true, false) => CombineOp::Add,
            (false, true) => CombineOp::Subtract,
            (false, false) => CombineOp::Replace,
        }
    }

    fn begin_tool(&mut self, pos: DVec2, pressure: f32, eraser_tool: bool) -> Drag {
        let doc = self.view.screen_to_doc(pos);
        let op = self.combine_op();
        match self.tool {
            Tool::Brush | Tool::Eraser | Tool::Smudge => {
                self.begin_stroke(pos, pressure, eraser_tool || self.tool == Tool::Eraser)
            }
            Tool::Marquee { ellipse } => Drag::Marquee { start: doc, end: doc, op, ellipse },
            Tool::Lasso => Drag::Lasso { points: vec![doc], op },
            Tool::Gradient { .. } => Drag::Gradient { start: doc, end: doc },
            Tool::Move => Drag::Move { start: doc, end: doc },
            Tool::Crop => Drag::None,
            Tool::Wand => {
                let (x, y) = (doc.x.floor(), doc.y.floor());
                if x >= 0.0 && y >= 0.0 {
                    self.magic_wand(x as u32, y as u32, op);
                }
                Drag::None
            }
        }
    }

    fn box_mods(&self) -> boxedit::Modifiers {
        boxedit::Modifiers { shift: self.mods.shift, alt: self.mods.alt }
    }

    /// Grabs a handle of the transform / crop box (creating the crop box on first use).
    fn begin_box(&mut self, pos: DVec2) -> Drag {
        let doc = self.view.screen_to_doc(pos);
        if self.boxed.is_none() {
            let Some(s) = &self.session else { return Drag::None };
            let (w, h) = s.engine.size();
            self.boxed = Some((BoxKind::Crop, BoxEdit::new(DVec2::ZERO, DVec2::new(w as f64, h as f64), false)));
        }
        let (view, scale) = (self.view, self.scale);
        let (_, b) = self.boxed.as_mut().unwrap();
        let handle = b.hit(&view, pos, scale);
        b.begin(handle, doc);
        Drag::Box
    }

    /// Ctrl+T: free transform of the active layer (or the selected pixels).
    fn begin_transform(&mut self) {
        if self.boxed.is_some() {
            return;
        }
        let Some(s) = &mut self.session else { return };
        let e = &mut s.engine;
        let Some(l) = e.doc().active_layer() else { return };
        if l.is_group() {
            tracing::warn!("Gruppen lassen sich noch nicht transformieren.");
            return;
        }
        let id = l.id;
        match e.content_bounds(id) {
            Some((x0, y0, x1, y1)) => {
                let b = BoxEdit::new(DVec2::new(x0 as f64, y0 as f64), DVec2::new(x1 as f64, y1 as f64), true);
                self.boxed = Some((BoxKind::Transform, b));
                tracing::info!(
                    "Frei transformieren: Ecken skalieren (Umschalt = frei), Kanten strecken, außerhalb drehen, Alt = um die Mitte. Enter übernimmt, Esc bricht ab."
                );
            }
            None => tracing::warn!("Nichts zu transformieren: die Ebene (bzw. Auswahl) ist leer."),
        }
        self.needs_redraw = true;
    }

    /// Enter: applies the transform or crop.
    fn commit_box(&mut self) {
        let Some((kind, b)) = self.boxed.take() else { return };
        match kind {
            BoxKind::Transform => {
                let fwd = b.fwd;
                self.engine_op(|e| e.transform_active(fwd, Interp::Bicubic, "Frei transformieren"));
            }
            BoxKind::Crop => {
                let (x, y, w, h) = b.rect();
                if w < 1 || h < 1 {
                    tracing::warn!("Zuschneiden: Rahmen ist leer");
                    return;
                }
                self.engine_op(|e| e.crop(x, y, w as u32, h as u32, "Zuschneiden"));
                self.view_fitted = false;
                self.fit_if_needed();
                tracing::info!("Neue Bildgröße: {w}×{h}");
            }
        }
        self.needs_redraw = true;
    }

    fn cancel_box(&mut self) {
        if self.boxed.take().is_some() {
            if let Some(e) = self.engine() {
                let _ = e.set_transform_preview(None);
            }
            tracing::info!("Abgebrochen");
            self.needs_redraw = true;
        }
    }

    /// The transform to show live (free transform, or a move in progress).
    fn preview_transform(&self) -> Option<DAffine2> {
        match (&self.boxed, &self.drag) {
            (Some((BoxKind::Transform, b)), _) => Some(b.fwd),
            (_, Drag::Move { start, end }) => Some(DAffine2::from_translation(self.move_delta(*start, *end))),
            _ => None,
        }
    }

    fn move_delta(&self, start: DVec2, end: DVec2) -> DVec2 {
        let mut d = (end - start).round();
        if self.mods.shift {
            if d.x.abs() > d.y.abs() {
                d.y = 0.0;
            } else {
                d.x = 0.0;
            }
        }
        d
    }

    fn stroke_target(&self) -> StrokeTarget {
        let adjustment = self.session.as_ref().and_then(|s| s.engine.doc().active_layer()).is_some_and(|l| l.adjustment().is_some());
        if self.mask_edit || adjustment { StrokeTarget::Mask } else { StrokeTarget::Pixels }
    }

    fn begin_stroke(&mut self, pos: DVec2, pressure: f32, eraser: bool) -> Drag {
        let mut settings = self.brush;
        settings.eraser = eraser;
        if self.tool == Tool::Smudge && !eraser {
            settings.smudge = Some(self.smudge_strength);
        }
        let target = self.stroke_target();
        if target == StrokeTarget::Mask {
            // Masks are grey: paint the (perceptual) luminance of the brush colour.
            let [r, g, b] = settings.color;
            let lum = linear_to_srgb(0.2126 * r + 0.7152 * g + 0.0722 * b);
            settings.color = [lum; 3];
        }
        let doc_pos = self.view.screen_to_doc(pos);
        let Some(engine) = self.engine() else { return Drag::None };
        let params = StrokeParams { opacity: settings.opacity, erase: settings.eraser, target, smudge: settings.smudge.is_some() };
        if let Err(e) = engine.begin_stroke(params) {
            tracing::warn!("Malen nicht möglich: {e}");
            return Drag::None;
        }
        let mut stroke = Stroke::new(settings);
        let dabs = stroke.add(InputSample { pos: doc_pos, pressure });
        engine.stroke_dabs(&dabs);
        self.stroke = Some(stroke);
        Drag::Paint
    }

    fn motion(&mut self, pos: DVec2, pressure: f32) {
        self.cursor = Some(pos);
        match &mut self.drag {
            Drag::None => {}
            Drag::Paint => {
                let doc_pos = self.view.screen_to_doc(pos);
                if let (Some(stroke), Some(s)) = (&mut self.stroke, &mut self.session) {
                    s.engine.stroke_dabs(&stroke.add(InputSample { pos: doc_pos, pressure }));
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
            Drag::Marquee { end, .. } | Drag::Gradient { end, .. } | Drag::Move { end, .. } => {
                *end = self.view.screen_to_doc(pos);
            }
            Drag::BrushHud { start, size, hardness, .. } => {
                let d = pos - *start;
                // 1 screen px ≈ 1 px of diameter at the current zoom; up = harder.
                self.brush.size = (*size + (d.x / self.view.zoom) as f32 * 2.0).clamp(1.0, 5000.0);
                if self.brush.tip == 0 {
                    self.brush.hardness = (*hardness - d.y as f32 * 0.004).clamp(0.0, 1.0);
                }
            }
            Drag::Box => {
                let doc = self.view.screen_to_doc(pos);
                let m = self.box_mods();
                if let Some((_, b)) = &mut self.boxed {
                    b.update(doc, m);
                }
            }
            Drag::Lasso { points, .. } => {
                let doc = self.view.screen_to_doc(pos);
                if points.last().is_none_or(|p| p.distance(doc) * self.view.zoom >= 2.0) {
                    points.push(doc);
                }
            }
        }
        self.needs_redraw = true;
    }

    fn release(&mut self, pos: DVec2, pressure: f32) {
        let drag = std::mem::replace(&mut self.drag, Drag::None);
        let target = self.stroke_target();
        let tool = self.tool;
        let color = self.brush.color;
        let opacity = self.brush.opacity;
        let zoom = self.view.zoom;
        let doc_pos = self.view.screen_to_doc(pos);
        let move_delta = match &drag {
            Drag::Move { start, end } => self.move_delta(*start, *end),
            _ => DVec2::ZERO,
        };
        let Some(s) = &mut self.session else { return };
        let e = &mut s.engine;
        let result: Result<bool, &str> = match drag {
            Drag::Paint => match self.stroke.take() {
                Some(mut stroke) => {
                    e.stroke_dabs(&stroke.finish(InputSample { pos: doc_pos, pressure }));
                    e.end_stroke(if stroke.settings.eraser { "Radieren" } else { "Pinselstrich" });
                    Ok(true)
                }
                None => Ok(false),
            },
            Drag::Marquee { start, end, op, ellipse } => {
                let (min, max) = (start.min(end), start.max(end));
                let size = max - min;
                if size.min_element() * zoom < 2.0 {
                    // A click without dragging deselects (Photoshop).
                    if op == CombineOp::Replace { e.deselect("Auswahl aufheben").map(|_| true) } else { Ok(false) }
                } else {
                    let shape = if ellipse {
                        let c = (min + max) * 0.5;
                        Shape::Ellipse { cx: c.x, cy: c.y, rx: size.x * 0.5, ry: size.y * 0.5 }
                    } else {
                        Shape::Rect { x: min.x, y: min.y, w: size.x, h: size.y }
                    };
                    e.select_shape(&shape, op, "Auswahl").map(|_| true)
                }
            }
            Drag::Lasso { points, op } => {
                if points.len() >= 3 {
                    let poly = points.iter().map(|p| (p.x, p.y)).collect();
                    e.select_shape(&Shape::Polygon(poly), op, "Lasso-Auswahl").map(|_| true)
                } else if op == CombineOp::Replace {
                    e.deselect("Auswahl aufheben").map(|_| true)
                } else {
                    Ok(false)
                }
            }
            Drag::Gradient { start, end } if start.distance(end) * zoom >= 2.0 => {
                let kind = if tool == (Tool::Gradient { radial: true }) { GradientKind::Radial } else { GradientKind::Linear };
                let (c0, c1) = match target {
                    StrokeTarget::Pixels => ([color[0], color[1], color[2], 1.0], [0.0; 4]),
                    StrokeTarget::Mask => ([1.0; 4], [0.0, 0.0, 0.0, 1.0]),
                };
                e.gradient(target, kind, (start.x, start.y), (end.x, end.y), c0, c1, opacity, "Verlauf").map(|_| true)
            }
            Drag::Move { .. } => {
                let d = move_delta;
                if d == DVec2::ZERO {
                    let _ = e.set_transform_preview(None);
                    Ok(false)
                } else {
                    e.transform_active(DAffine2::from_translation(d), Interp::Nearest, "Verschieben").map(|_| true)
                }
            }
            Drag::Box => {
                if let Some((_, b)) = &mut self.boxed {
                    b.end();
                }
                Ok(false)
            }
            _ => Ok(false),
        };
        match result {
            Ok(true) => self.touch(),
            Ok(false) => {}
            Err(err) => tracing::warn!("{err}"),
        }
        self.needs_redraw = true;
    }

    /// Marks the document as changed.
    fn touch(&mut self) {
        self.modified = true;
        self.edit_count += 1;
    }

    pub fn on_tablet_frame(&mut self, f: TabletFrame) {
        let pos = self.to_physical(f.pos);
        // The pen also operates the interface.
        self.gui.pointer_moved(pos, self.scale);
        self.needs_redraw = true;
        if self.tablet_ui || (self.over_ui(pos) && !self.tablet_contact && matches!(self.drag, Drag::None)) {
            self.cursor = None;
            match f.contact {
                Some(true) => {
                    self.tablet_ui = true;
                    self.gui.pointer_button(pos, self.scale, false, true);
                }
                Some(false) => {
                    self.tablet_ui = false;
                    self.gui.pointer_button(pos, self.scale, false, false);
                }
                _ => {}
            }
            if f.proximity == Some(false) {
                self.gui.pointer_gone();
            }
            return;
        }
        if f.proximity == Some(false) {
            if self.tablet_contact {
                self.release(pos, 0.0);
                self.tablet_contact = false;
            }
            self.cursor = None;
            self.needs_redraw = true;
            return;
        }
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

    /// Undoable document edit with terminal feedback.
    fn edit(&mut self, label: &str, f: impl FnOnce(&mut imagine_core::Document<imagine_gpu::Tile>) -> bool) {
        let Some(engine) = self.engine() else { return };
        if engine.edit(label, f) {
            self.touch();
            self.print_layers();
        }
    }

    fn request_quit(&mut self) {
        if self.save_job.is_some() || self.saving {
            tracing::warn!("Speichern läuft noch – bitte einen Moment warten.");
            return;
        }
        if self.bg_running > 0 && self.quit_armed.is_none_or(|t| t.elapsed() >= QUIT_CONFIRM) {
            tracing::warn!("Im Hintergrund läuft noch etwas (z. B. ein Export). Nochmal Strg+Q bricht es ab.");
            self.quit_armed = Some(Instant::now());
            self.update_title();
            return;
        }
        self.guard_unsaved(Pending::Quit);
    }

    /// Runs `action` now, or first asks what to do with unsaved changes.
    fn guard_unsaved(&mut self, action: Pending) {
        if !self.modified {
            return self.run_pending(action);
        }
        let what = match action {
            Pending::Quit => "Beim Beenden gehen sie verloren.",
            Pending::Open(_) | Pending::New { .. } => "Das neue Bild ersetzt es.",
        };
        self.ui_state.confirm = Some(what.into());
        self.ui_state.hidden = false;
        self.pending = Some(action);
        self.needs_redraw = true;
    }

    fn run_pending(&mut self, action: Pending) {
        match action {
            Pending::Quit => self.exit = true,
            Pending::Open(p) => self.open_source(Source::File(p), false),
            Pending::New { w, h, white } => self.open_source(Source::Blank(w, h), white),
        }
    }

    fn answer_unsaved(&mut self, a: ui::files::Answer) {
        let Some(action) = self.pending.take() else { return };
        match a {
            ui::files::Answer::Discard => self.run_pending(action),
            ui::files::Answer::Save => {
                self.after_save = Some(action);
                self.save();
            }
            ui::files::Answer::Cancel => {}
        }
    }

    /// Replaces the document (the old one stays until the new structure arrives).
    fn open_source(&mut self, source: Source, white: bool) {
        if self.loading {
            tracing::warn!("Es wird noch geladen.");
            return;
        }
        if self.saving || self.save_job.is_some() {
            tracing::warn!("Speichern läuft noch – bitte einen Moment warten.");
            return;
        }
        self.drop_autosave();
        self.pending_uploads.clear();
        self.loading = true;
        self.modified = false;
        self.boxed = None;
        self.mask_edit = false;
        self.drag = Drag::None;
        self.stroke = None;
        self.fill_white_on_load = white;
        self.edit_count = 0;
        self.save_mark = 0;
        self.autosave_mark = 0;
        loader::spawn(source, self.loader_tx.clone());
    }

    fn file_dialog(&mut self, mode: ui::files::FileMode) {
        let s = self.session.as_ref();
        let dir = match mode {
            ui::files::FileMode::Open => self.settings.last_dir.clone(),
            _ => s.and_then(|s| s.default_save_path().parent().map(|p| p.to_path_buf())).filter(|p| !p.as_os_str().is_empty()),
        }
        .or_else(|| self.settings.last_dir.clone())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let name = match mode {
            ui::files::FileMode::Open => None,
            ui::files::FileMode::SaveAs => s.map(|s| s.default_save_path()),
            ui::files::FileMode::ExportAs => s.map(|s| s.default_export_path()),
        }
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
        let mut d = ui::files::FileDialog::new(mode, dir, name);
        d.export = self.settings.export;
        self.ui_state.file = Some(d);
        self.ui_state.hidden = false;
    }

    fn apply_settings(&mut self) {
        let st = self.settings.clone();
        st.brush.apply(&mut self.brush);
        self.wand_tolerance = st.wand_tolerance.clamp(0.0, 255.0);
        self.wand_contiguous = st.wand_contiguous;
        self.smudge_strength = st.smudge_strength.clamp(0.0, 1.0);
        self.tool = match st.tool.as_str() {
            "eraser" => Tool::Eraser,
            "smudge" => Tool::Smudge,
            "marquee" => Tool::Marquee { ellipse: false },
            "ellipse" => Tool::Marquee { ellipse: true },
            "lasso" => Tool::Lasso,
            "wand" => Tool::Wand,
            "gradient" => Tool::Gradient { radial: false },
            "radial" => Tool::Gradient { radial: true },
            "move" => Tool::Move,
            _ => Tool::Brush,
        };
    }

    fn store_settings(&mut self) {
        let mut b = self.brush;
        b.eraser = false;
        self.settings.brush = (&b).into();
        self.settings.wand_tolerance = self.wand_tolerance;
        self.settings.wand_contiguous = self.wand_contiguous;
        self.settings.smudge_strength = self.smudge_strength;
        self.settings.tool = match self.tool {
            Tool::Brush => "brush",
            Tool::Eraser => "eraser",
            Tool::Smudge => "smudge",
            Tool::Marquee { ellipse: false } => "marquee",
            Tool::Marquee { ellipse: true } => "ellipse",
            Tool::Lasso => "lasso",
            Tool::Wand => "wand",
            Tool::Gradient { radial: false } => "gradient",
            Tool::Gradient { radial: true } => "radial",
            Tool::Move => "move",
            Tool::Crop => "brush",
        }
        .into();
        if let Err(e) = self.settings.save_to(&settings::config_path()) {
            tracing::warn!("Einstellungen nicht gespeichert: {e:#}");
        }
    }

    fn save(&mut self) {
        // Never saved as .imagine yet: ask where.
        if self.session.as_ref().is_some_and(|s| !s.has_save_path()) {
            self.file_dialog(ui::files::FileMode::SaveAs);
            return;
        }
        self.save_to(None);
    }

    fn save_to(&mut self, path: Option<std::path::PathBuf>) {
        if self.save_job.is_some() || self.saving {
            tracing::warn!("Speichern läuft noch.");
            return;
        }
        let Some(s) = &mut self.session else { return };
        match s.start_save(path.as_deref()) {
            Ok(job) => {
                self.save_job = Some(job);
                self.save_mark = self.edit_count;
                tracing::info!("Speichern …");
            }
            Err(e) => tracing::error!("Speichern fehlgeschlagen: {e:#}"),
        }
        self.update_title();
    }

    /// Hands a finished readback to a worker thread (compress + write).
    fn poll_save(&mut self) {
        let ready = self.save_job.as_ref().is_some_and(|j| j.ready(&self.gpu.device));
        if ready {
            let writer = self.save_job.take().unwrap().into_writer();
            let tx = self.save_tx.clone();
            self.saving = true;
            std::thread::Builder::new()
                .name("save".into())
                .spawn(move || {
                    let _ = tx.send(writer());
                })
                .expect("spawn save thread");
        }
    }

    fn on_save_done(&mut self, result: anyhow::Result<SaveDone>) {
        self.saving = false;
        let Some(s) = &mut self.session else { return };
        match result {
            Ok(done) => {
                let r = s.complete_save(done);
                if self.edit_count == self.save_mark {
                    self.modified = false;
                    self.drop_autosave();
                }
                self.settings.add_recent(&r.path);
                tracing::info!(
                    "Gespeichert: {} ({} Kacheln geschrieben, {}, {:.1} MB, {:.0} ms im Hintergrund)",
                    r.path.display(),
                    r.tiles_written,
                    if r.incremental { "inkrementell" } else { "komplett" },
                    r.bytes as f64 / 1e6,
                    r.ms
                );
            }
            Err(e) => {
                tracing::error!("Speichern fehlgeschlagen: {e:#}");
                self.after_save = None;
            }
        }
        self.update_title();
        // Continue what the unsaved-changes question was about.
        if !self.modified
            && let Some(a) = self.after_save.take()
        {
            self.run_pending(a);
        }
    }

    fn export_to(&mut self, path: std::path::PathBuf, opts: imagine_io::export::ExportOptions) {
        if self.settings.export != opts {
            self.settings.export = opts;
            self.store_settings();
        }
        let Some(s) = &mut self.session else { return };
        if self.export_job.is_some() {
            tracing::warn!("Export läuft noch.");
            return;
        }
        tracing::info!("Exportiere {} ({}) im Hintergrund …", path.display(), opts.space.name());
        self.export_job = Some(s.start_export(&path, &opts));
        self.bg_running += 1;
    }

    fn export(&mut self) {
        let Some(s) = &mut self.session else { return };
        if self.export_job.is_some() {
            tracing::warn!("Export läuft noch.");
            return;
        }
        let path = s.default_export_path();
        let opts = self.settings.export;
        tracing::info!("Exportiere {} ({}) im Hintergrund …", path.display(), opts.space.name());
        self.export_job = Some(s.start_export(&path, &opts));
        self.bg_running += 1;
    }

    /// Starts a background recovery copy if there are unsaved changes and
    /// the last one is old enough. The document's own file is not touched.
    fn maybe_autosave(&mut self) {
        if let Some((_, job)) = &self.autosave_job {
            if job.ready(&self.gpu.device) {
                let (path, job) = self.autosave_job.take().unwrap();
                let work = job.into_writer();
                self.spawn_bg("autosave", move || BgDone::Autosaved { path, result: work().map(|_| ()) });
            }
            return;
        }
        if !self.modified
            || self.edit_count == self.autosave_mark
            || self.autosave_at.elapsed() < AUTOSAVE_EVERY
            || self.stroke.is_some()
            || self.loading
        {
            return;
        }
        let Some(s) = &mut self.session else { return };
        let dir = autosave_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!("Automatisches Sichern: {}: {e}", dir.display());
            self.autosave_at = Instant::now();
            return;
        }
        let stem = std::path::Path::new(&s.name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let path = dir.join(format!("{}-{}.imagine", if stem.is_empty() { "Unbenannt" } else { &stem }, std::process::id()));
        match s.start_save_copy(&path) {
            Ok(job) => self.autosave_job = Some((path, job)),
            Err(e) => tracing::warn!("Automatisches Sichern: {e:#}"),
        }
        self.autosave_mark = self.edit_count;
        self.autosave_at = Instant::now();
    }

    /// Removes the recovery copy (after a real save or a clean exit).
    fn drop_autosave(&mut self) {
        let paths = [self.autosave_path.take(), self.autosave_job.take().map(|(p, _)| p)];
        for p in paths.into_iter().flatten() {
            let _ = std::fs::remove_file(p);
        }
    }

    /// Hands a finished export readback to a worker thread.
    fn poll_export(&mut self) {
        if self.export_job.as_ref().is_some_and(|j| j.ready(&self.gpu.device)) {
            let job = self.export_job.take().unwrap();
            let path = job.path.clone();
            let work = job.into_writer();
            self.bg_running -= 1; // spawn_bg counts it again
            self.spawn_bg("export", move || BgDone::Exported { path, result: work() });
        }
    }

    /// Runs `f` on a worker thread; its result arrives in [`Self::on_background_done`].
    fn spawn_bg(&mut self, name: &str, f: impl FnOnce() -> BgDone + Send + 'static) {
        let tx = self.bg_tx.clone();
        self.bg_running += 1;
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _ = tx.send(f());
            })
            .expect("spawn worker thread");
        self.update_title();
    }

    /// Computes a new selection from the current one on a worker thread.
    fn selection_job(
        &mut self,
        label: &'static str,
        f: impl FnOnce(Option<CpuMask>) -> anyhow::Result<CpuMask> + Send + 'static,
    ) {
        let Some(e) = self.engine() else { return };
        let base = e.selection_id();
        let current = e.selection_cpu();
        let started = Instant::now();
        self.spawn_bg("selection", move || BgDone::Selection { label, base, mask: f(current), started });
    }

    fn on_background_done(&mut self, done: BgDone) {
        self.bg_running = self.bg_running.saturating_sub(1);
        match done {
            BgDone::Autosaved { path, result } => match result {
                Ok(()) => {
                    tracing::info!("Wiederherstellungskopie gesichert: {}", path.display());
                    self.autosave_path = Some(path);
                }
                Err(e) => tracing::warn!("Automatisches Sichern fehlgeschlagen: {e:#}"),
            },
            BgDone::Exported { path, result } => match result {
                Ok(ms) => tracing::info!("Exportiert: {} ({ms:.0} ms)", path.display()),
                Err(e) => tracing::error!("Export fehlgeschlagen: {e:#}"),
            },
            BgDone::Selection { label, base, mask, started } => {
                let Some(e) = self.engine() else { return };
                if e.selection_id() != base {
                    tracing::warn!("{label}: verworfen, die Auswahl hat sich inzwischen geändert");
                } else {
                    match mask {
                        Ok(m) => {
                            self.engine_op(|e| e.set_selection(Some(m), label));
                            tracing::info!("{label}: {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
                        }
                        Err(err) => tracing::warn!("{label}: {err:#}"),
                    }
                }
            }
        }
        self.update_title();
        self.needs_redraw = true;
    }

    /// Magic wand: similarity map on the GPU here, flood fill and combining on a worker.
    fn magic_wand(&mut self, x: u32, y: u32, op: CombineOp) {
        let (tol, contiguous) = (self.wand_tolerance, self.wand_contiguous);
        let Some(e) = self.engine() else { return };
        let (w, h) = e.size();
        let map = e
            .sample_straight(x, y, SampleFrom::Image)
            .and_then(|reference| e.similarity_map(reference, tol, (tol * 0.5).max(12.0), SampleFrom::Image));
        let map = match map {
            Ok(m) => m,
            Err(err) => return tracing::warn!("Zauberstab: {err}"),
        };
        self.selection_job("Zauberstab", move |old| {
            let region = if contiguous { imagine_core::mask::flood(&map, x, y, w, h) } else { map };
            Ok(match old {
                Some(old) => imagine_core::mask::combine(&old, &region, op),
                None if op == CombineOp::Subtract => CpuMask::filled(0),
                None => region,
            })
        });
    }

    fn key(&mut self, ev: &KeyEvent, pressed: bool) {
        let sym = ev.keysym;
        tracing::debug!(?sym, raw = ev.raw_code, pressed, "Taste");
        // Text fields and the command palette take the keyboard.
        if self.ui_state.captures_keyboard() || self.gui.wants_keyboard() {
            if let Some(k) = ui::egui_key(sym) {
                self.gui.key(k, pressed);
            }
            if pressed && let Some(t) = &ev.utf8 {
                self.gui.text(t);
            }
            self.needs_redraw = true;
            return;
        }
        match sym {
            Keysym::space => self.space = pressed,
            Keysym::r => self.r_key = pressed,
            _ => {}
        }
        if !pressed || self.stroke.is_some() {
            return;
        }
        self.hotkey(ev);
        self.needs_redraw = true;
    }

    /// Runs a menu / palette command by replaying its shortcut.
    fn run_cmd(&mut self, c: ui::palette::Cmd) {
        let saved = self.mods;
        self.mods.ctrl = c.ctrl;
        self.mods.shift = c.shift;
        self.mods.alt = c.alt;
        self.hotkey(&KeyEvent { time: 0, raw_code: 0, keysym: c.key, utf8: None });
        self.mods = saved;
        self.needs_redraw = true;
    }

    fn hotkey(&mut self, ev: &KeyEvent) {
        let sym = ev.keysym;
        let (ctrl, shift, alt) = (self.mods.ctrl, self.mods.shift, self.mods.alt);
        let center = self.view.viewport * 0.5;
        let anchor = self.cursor.unwrap_or(center);

        // Digit row by physical key, so Shift+digit works on every layout.
        if (KEY_1..=KEY_0).contains(&ev.raw_code) && !ctrl && !alt {
            let pct = if ev.raw_code == KEY_0 { 1.0 } else { (ev.raw_code - KEY_1 + 1) as f32 / 10.0 };
            if shift {
                self.edit("Deckkraft", |d| {
                    let id = d.active;
                    let l = d.find_mut(id).unwrap();
                    let changed = l.opacity != pct;
                    l.opacity = pct;
                    changed
                });
            } else if self.tool == Tool::Smudge {
                self.smudge_strength = pct;
                tracing::info!("Wischfinger-Stärke {:.0} %", pct * 100.0);
            } else {
                self.brush.opacity = pct;
                tracing::info!("Pinsel-Deckkraft {:.0} %", pct * 100.0);
            }
            self.needs_redraw = true;
            return;
        }

        match sym {
            Keysym::q | Keysym::Q if ctrl => self.request_quit(),
            Keysym::p | Keysym::P if ctrl && shift => self.ui_state.palette = Some((String::new(), 0)),
            Keysym::k if ctrl => self.ui_state.palette = Some((String::new(), 0)),
            Keysym::Tab => self.ui_state.hidden = !self.ui_state.hidden,
            Keysym::Escape => self.cancel_box(),
            Keysym::Return | Keysym::KP_Enter => self.commit_box(),
            Keysym::t if ctrl => self.begin_transform(),
            // While a transform/crop box is open, undo just cancels it.
            Keysym::z | Keysym::Z if ctrl && self.boxed.is_some() => self.cancel_box(),
            Keysym::z | Keysym::Z if ctrl && shift => self.redo(),
            Keysym::z if ctrl => self.undo(),
            Keysym::y if ctrl => self.redo(),
            Keysym::s | Keysym::S if ctrl && shift => self.file_dialog(ui::files::FileMode::SaveAs),
            Keysym::s | Keysym::S if ctrl => self.save(),
            Keysym::e | Keysym::E if ctrl && shift && alt => self.file_dialog(ui::files::FileMode::ExportAs),
            Keysym::e | Keysym::E if ctrl && shift => self.export(),
            Keysym::o if ctrl => self.file_dialog(ui::files::FileMode::Open),
            Keysym::n if ctrl && !shift => {
                self.ui_state.new_doc = Some(Default::default());
                self.ui_state.hidden = false;
            }
            Keysym::n | Keysym::N if ctrl && shift => self.edit("Neue Ebene", |d| {
                d.add_pixel_layer();
                true
            }),
            Keysym::j if ctrl => self.edit("Ebene duplizieren", |d| d.duplicate_active().is_some()),
            Keysym::g | Keysym::G if ctrl && shift => self.edit("Gruppierung aufheben", |d| d.ungroup_active()),
            Keysym::g if ctrl => self.edit("Gruppieren", |d| d.group_active().is_some()),
            // Photoshop: Entf and Rücktaste both clear the selection.
            Keysym::Delete | Keysym::BackSpace
                if !alt && self.session.as_ref().is_some_and(|s| s.engine.has_selection()) =>
            {
                self.engine_op(|e| {
                    let n = e.clear_selected("Löschen")?;
                    let name = e.doc().active_layer().map(|l| l.name.clone()).unwrap_or_default();
                    if n == 0 {
                        tracing::warn!("Nichts gelöscht: die Auswahl enthält auf Ebene „{name}“ keine Pixel. Andere Ebene aktiv?");
                    } else {
                        tracing::info!("Auswahl auf Ebene „{name}“ gelöscht ({n} Kacheln)");
                    }
                    Ok(())
                })
            }
            Keysym::Delete => self.edit("Ebene löschen", |d| d.remove_active()),
            Keysym::BackSpace if alt => {
                let [r, g, b] = self.brush.color;
                let (target, opacity) = (self.stroke_target(), self.brush.opacity);
                let color = if target == StrokeTarget::Mask {
                    let l = linear_to_srgb(0.2126 * r + 0.7152 * g + 0.0722 * b);
                    [l, l, l, 1.0]
                } else {
                    [r, g, b, 1.0]
                };
                self.engine_op(|e| e.fill(target, color, opacity, "Füllen").map(|_| ()))
            }
            Keysym::a if ctrl => self.engine_op(|e| e.select_all("Alles auswählen")),
            Keysym::d if ctrl => self.engine_op(|e| e.deselect("Auswahl aufheben")),
            Keysym::i | Keysym::I if ctrl && shift => self.engine_op(|e| e.invert_selection("Auswahl umkehren")),
            Keysym::F6 if shift => {
                let Some((w, h)) = self.session.as_ref().map(|s| s.engine.size()) else { return };
                if self.session.as_ref().is_some_and(|s| !s.engine.has_selection()) {
                    tracing::warn!("keine Auswahl");
                } else {
                    self.selection_job("Weiche Auswahlkante (10 px)", move |sel| {
                        let sel = sel.context("keine Auswahl")?;
                        Ok(imagine_core::mask::feather(&sel, 10.0, w, h))
                    });
                }
            }
            Keysym::m | Keysym::M if ctrl && alt && shift => {
                self.mask_edit = false;
                self.engine_op(|e| e.apply_layer_mask("Ebenenmaske anwenden"))
            }
            Keysym::m | Keysym::M if ctrl && alt => self.edit("Ebenenmaske ein/aus", |d| {
                let id = d.active;
                let l = d.find_mut(id).unwrap();
                l.mask_enabled = !l.mask_enabled;
                l.mask.is_some()
            }),
            Keysym::m | Keysym::M if ctrl && shift => self.engine_op(|e| e.add_layer_mask("Ebenenmaske")),
            Keysym::backslash => {
                let has_mask = self.session.as_ref().and_then(|s| s.engine.doc().active_layer()).is_some_and(|l| l.mask.is_some());
                if self.mask_edit || has_mask {
                    self.mask_edit = !self.mask_edit;
                    tracing::info!("Ziel: {}", if self.mask_edit { "Ebenenmaske" } else { "Pixel" });
                } else {
                    tracing::warn!("Die Ebene hat keine Maske (Strg+Umschalt+M legt eine an).");
                }
            }
            Keysym::Left | Keysym::Right | Keysym::Up | Keysym::Down if self.tool == Tool::Move => {
                let step = if shift { 10 } else { 1 };
                let (dx, dy) = match sym {
                    Keysym::Left => (-step, 0),
                    Keysym::Right => (step, 0),
                    Keysym::Up => (0, -step),
                    _ => (0, step),
                };
                let fwd = DAffine2::from_translation(DVec2::new(dx as f64, dy as f64));
                self.engine_op(|e| e.transform_active(fwd, Interp::Nearest, "Verschieben"))
            }
            Keysym::comma if ctrl => self.edit("Sichtbarkeit", |d| {
                let id = d.active;
                let l = d.find_mut(id).unwrap();
                l.visible = !l.visible;
                true
            }),
            Keysym::bracketleft if alt => self.select_layer(1),
            Keysym::bracketright if alt => self.select_layer(-1),
            Keysym::bracketleft if ctrl => self.edit("Ebene nach unten", |d| d.move_active(-1)),
            Keysym::bracketright if ctrl => self.edit("Ebene nach oben", |d| d.move_active(1)),
            Keysym::plus | Keysym::asterisk | Keysym::KP_Add if shift || sym == Keysym::KP_Add => self.cycle_blend(1),
            Keysym::minus | Keysym::underscore | Keysym::KP_Subtract if shift || sym == Keysym::KP_Subtract => {
                self.cycle_blend(-1)
            }
            Keysym::F7 => self.edit("Blend-Farbraum", |d| {
                d.blend_space = match d.blend_space {
                    BlendSpace::Srgb => BlendSpace::Linear,
                    BlendSpace::Linear => BlendSpace::Srgb,
                };
                true
            }),
            Keysym::_0 if ctrl => {
                self.view_fitted = false;
                self.fit_if_needed();
            }
            Keysym::_1 if ctrl => self.view.set_zoom_at(anchor, 1.0),
            Keysym::R => {
                let r = self.view.rotation;
                self.view.rotate_at(center, -r);
            }
            Keysym::bracketleft | Keysym::bracketright if self.tool == Tool::Wand => {
                let step = if sym == Keysym::bracketleft { -8.0 } else { 8.0 };
                self.wand_tolerance = (self.wand_tolerance + step).clamp(0.0, 255.0);
                tracing::info!("Toleranz {:.0}", self.wand_tolerance);
            }
            Keysym::bracketleft => self.brush.size = (self.brush.size / 1.2).max(1.0),
            Keysym::bracketright => self.brush.size = (self.brush.size * 1.2).min(5000.0),
            Keysym::e => self.set_tool(Tool::Eraser),
            Keysym::b => self.set_tool(Tool::Brush),
            Keysym::B => {
                self.brush.tip = (self.brush.tip + 1) % imagine_core::TIP_NAMES.len() as u32;
                tracing::info!("Pinselspitze: {}", imagine_core::TIP_NAMES[self.brush.tip as usize]);
            }
            Keysym::u => self.set_tool(Tool::Smudge),
            Keysym::j => {
                let on = self.brush.size_jitter == 0.0;
                let b = &mut self.brush;
                (b.size_jitter, b.angle_jitter, b.scatter, b.flow_jitter) =
                    if on { (0.3, 30.0, 0.25, 0.3) } else { (0.0, 0.0, 0.0, 0.0) };
                tracing::info!("Pinsel-Dynamik (Größe, Winkel, Streuung, Fluss zufällig) {}", if on { "an" } else { "aus" });
            }
            Keysym::m => {
                let ellipse = matches!(self.tool, Tool::Marquee { ellipse: false });
                self.set_tool(Tool::Marquee { ellipse })
            }
            Keysym::l => self.set_tool(Tool::Lasso),
            Keysym::g => {
                let radial = matches!(self.tool, Tool::Gradient { radial: false });
                self.set_tool(Tool::Gradient { radial })
            }
            Keysym::v => self.set_tool(Tool::Move),
            Keysym::q => {
                self.quick_mask = !self.quick_mask;
                tracing::info!("Maskierungsmodus-Ansicht {}", if self.quick_mask { "an" } else { "aus" });
            }
            Keysym::x => {
                let [r, g, b] = self.brush.color;
                let dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.2;
                self.brush.color = if dark { [1.0; 3] } else { [0.0; 3] };
                tracing::info!("Farbe: {}", if dark { "Weiß" } else { "Schwarz" });
            }
            Keysym::c => self.set_tool(Tool::Crop),
            Keysym::w => {
                if self.tool == Tool::Wand {
                    self.wand_contiguous = !self.wand_contiguous;
                }
                self.set_tool(Tool::Wand);
                tracing::info!(
                    "Zauberstab: Toleranz {:.0} ([ / ]), {} (W wechselt), nimmt alle Ebenen auf",
                    self.wand_tolerance,
                    if self.wand_contiguous { "nur zusammenhängend" } else { "überall im Bild" }
                );
            }
            Keysym::p => {
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

    fn set_tool(&mut self, tool: Tool) {
        match self.boxed.as_ref().map(|b| b.0) {
            Some(BoxKind::Transform) => self.commit_box(),
            Some(BoxKind::Crop) if tool != Tool::Crop => self.boxed = None,
            _ => {}
        }
        if tool == Tool::Crop {
            tracing::info!("Zuschneiden: Rahmen ziehen oder Anfasser verschieben (über den Rand hinaus vergrößert die Arbeitsfläche), Enter übernimmt, Esc bricht ab.");
        }
        self.tool = tool;
        self.brush.eraser = tool == Tool::Eraser;
        tracing::info!("Werkzeug: {}", tool.name());
        self.update_title();
    }

    /// Runs an engine operation, reporting errors and marking the document changed.
    fn engine_op(&mut self, f: impl FnOnce(&mut Engine) -> Result<(), &'static str>) {
        let Some(e) = self.engine() else { return };
        match f(e) {
            Ok(()) => {
                self.touch();
                self.print_layers();
            }
            Err(err) => tracing::warn!("{err}"),
        }
    }

    fn undo(&mut self) {
        if let Some(label) = self.engine().and_then(|e| e.undo()) {
            tracing::info!("Rückgängig: {label}");
            self.touch();
            self.print_layers();
        }
    }

    fn redo(&mut self) {
        if let Some(label) = self.engine().and_then(|e| e.redo()) {
            tracing::info!("Wiederholen: {label}");
            self.touch();
            self.print_layers();
        }
    }

    fn select_layer(&mut self, step: i32) {
        if let Some(e) = self.engine() {
            e.select(|d| d.select_relative(step));
        }
        self.print_layers();
    }

    fn cycle_blend(&mut self, step: i32) {
        self.edit("Füllmethode", |d| {
            let id = d.active;
            let l = d.find_mut(id).unwrap();
            l.blend = l.blend.cycle(step);
            true
        });
    }

    // ---- agents -----------------------------------------------------------

    /// Reads from an agent connection and answers every complete line.
    fn agent_io(&mut self, stream: &std::os::unix::net::UnixStream, buf: &mut Vec<u8>) -> PostAction {
        use std::io::{Read, Write};
        let mut chunk = [0u8; 64 << 10];
        loop {
            match (&*stream).read(&mut chunk) {
                Ok(0) => return PostAction::Remove,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => return PostAction::Remove,
            }
        }
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let reply = self.agent_line(&line);
            // Large replies (previews) are written blocking.
            let _ = stream.set_nonblocking(false);
            let ok = (&*stream).write_all(reply.as_bytes()).is_ok();
            let _ = stream.set_nonblocking(true);
            if !ok {
                return PostAction::Remove;
            }
        }
        PostAction::Continue
    }

    fn agent_line(&mut self, line: &[u8]) -> String {
        let req: serde_json::Value = match serde_json::from_slice(line) {
            Ok(v) => v,
            Err(e) => return ipc::response(serde_json::Value::Null, Err(anyhow::anyhow!("invalid JSON: {e}"))),
        };
        let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
        let mut cmd_json = req;
        if let Some(o) = cmd_json.as_object_mut() {
            o.remove("id");
        }
        let cmd = match command::Command::parse(cmd_json) {
            Ok(c) => c,
            Err(e) => return ipc::response(id, Err(e)),
        };
        if self.stroke.is_some() && cmd.mutates() {
            return ipc::response(id, Err(anyhow::anyhow!("the user is painting right now – retry in a moment")));
        }
        let (name, mutates) = (cmd.name(), cmd.mutates());
        tracing::info!("Agent: {name}");
        let mut ctx = command::Ctx { gpu: &self.gpu, format: self.surface_config.format, session: &mut self.session };
        let result = command::execute(&mut ctx, cmd);
        if let Ok(out) = &result {
            if out.replaced_document {
                self.pending_uploads.clear();
                self.loading = false;
                self.view_fitted = false;
                self.fit_if_needed();
                self.modified = false;
            } else if mutates {
                self.touch();
            }
            if mutates {
                self.print_layers();
            }
        }
        self.needs_redraw = true;
        self.update_title();
        ipc::response(id, result)
    }

    // ---- rendering --------------------------------------------------------

    fn start_bench(&mut self, exit_after: bool) {
        if self.session.is_none() {
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
        self.view = bench_view(&b.base, b.frame, BENCH_FRAMES);
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
        let tiles = self.session.as_ref().map_or(0, |s| s.engine.composite_tiles_live());
        println!(
            "\n=== Benchmark ({} Frames, {}×{} px, {tiles} Kacheln) ===\n\
             CPU pro Frame:    Ø {:.2} ms   p50 {:.2}   p99 {:.2}   max {:.2}\n\
             Frame-Abstand:    Ø {:.2} ms   p50 {:.2}   p99 {:.2}   max {:.2}   (≈ {:.0} fps)\n",
            cpu.len(),
            self.surface_config.width,
            self.surface_config.height,
            c.0,
            c.1,
            c.2,
            c.3,
            i.0,
            i.1,
            i.2,
            i.3,
            1000.0 / i.0.max(1e-6),
        );
        self.view = b.base;
        if b.exit_after {
            self.exit = true;
        }
    }

    /// Whether a screen position (physical px) is covered by the interface.
    fn over_ui(&self, pos: DVec2) -> bool {
        let (a, b) = self.canvas_rect;
        let outside = b != DVec2::ZERO && (pos.x < a.x || pos.y < a.y || pos.x >= b.x || pos.y >= b.y);
        let p = egui::pos2((pos.x / self.scale) as f32, (pos.y / self.scale) as f32);
        let floating = self.gui.ctx.layer_id_at(p).is_some_and(|l| l.order != egui::Order::Background);
        outside || floating
    }

    fn tool_id(&self) -> ToolId {
        match self.tool {
            Tool::Brush => ToolId::Brush,
            Tool::Eraser => ToolId::Eraser,
            Tool::Smudge => ToolId::Smudge,
            Tool::Marquee { ellipse: false } => ToolId::Marquee,
            Tool::Marquee { ellipse: true } => ToolId::Ellipse,
            Tool::Lasso => ToolId::Lasso,
            Tool::Wand => ToolId::Wand,
            Tool::Gradient { radial: false } => ToolId::Gradient,
            Tool::Gradient { radial: true } => ToolId::Radial,
            Tool::Move => ToolId::Move,
            Tool::Crop => ToolId::Crop,
        }
    }

    /// Snapshot of the app state for the panels.
    fn ui_model(&self) -> Option<ui::panels::Model> {
        let s = self.session.as_ref()?;
        let e = &s.engine;
        let doc = e.doc();
        let active = doc.active_layer();
        let (history, history_done) = e.history_labels();
        let b = &self.brush;
        Some(ui::panels::Model {
            doc_size: e.size(),
            name: s.name.clone(),
            modified: self.modified,
            zoom: self.view.zoom / self.scale,
            rotation_deg: self.view.rotation.to_degrees(),
            // Not while dragging: the panels need not be rebuilt for every stroke frame.
            cursor_doc: self.cursor.filter(|_| matches!(self.drag, Drag::None)).map(|c| {
                let d = self.view.screen_to_doc(c);
                (d.x, d.y)
            }),
            tool: self.tool_id(),
            color: b.color,
            brush: ui::panels::BrushModel {
                size: b.size,
                hardness: b.hardness,
                flow: b.flow,
                opacity: b.opacity,
                spacing: b.spacing,
                smoothing: b.smoothing,
                tip: b.tip,
                angle: b.angle,
                roundness: b.roundness,
                size_jitter: b.size_jitter,
                angle_jitter: b.angle_jitter,
                scatter: b.scatter,
                flow_jitter: b.flow_jitter,
                follow_direction: b.follow_direction,
            },
            smudge_strength: self.smudge_strength,
            wand_tolerance: self.wand_tolerance,
            wand_contiguous: self.wand_contiguous,
            boxed: self.boxed.as_ref().map(|(k, _)| match k {
                BoxKind::Transform => "Frei transformieren",
                BoxKind::Crop => "Zuschneiden",
            }),
            mask_edit: self.mask_edit,
            layers: ui::panels::layer_rows(doc),
            active: doc.active,
            active_opacity: active.map_or(1.0, |l| l.opacity),
            active_blend: active.map_or(imagine_core::BlendMode::Normal, |l| l.blend),
            history,
            history_done,
            has_selection: doc.selection.is_some(),
            status: self.ui_status.clone(),
            busy: self.bg_running > 0 || self.saving || self.save_job.is_some() || self.loading,
            recent: self.settings.recent.clone(),
            active_adjustment: active.and_then(|l| l.adjustment().cloned()),
            histogram: None,
        })
    }

    fn apply_ui(&mut self, actions: Vec<Action>) {
        for a in actions {
            self.needs_redraw = true;
            match a {
                Action::Command(c) => self.run_cmd(c),
                Action::Tool(t) => {
                    let tool = match t {
                        ToolId::Brush => Tool::Brush,
                        ToolId::Eraser => Tool::Eraser,
                        ToolId::Smudge => Tool::Smudge,
                        ToolId::Marquee => Tool::Marquee { ellipse: false },
                        ToolId::Ellipse => Tool::Marquee { ellipse: true },
                        ToolId::Lasso => Tool::Lasso,
                        ToolId::Wand => Tool::Wand,
                        ToolId::Gradient => Tool::Gradient { radial: false },
                        ToolId::Radial => Tool::Gradient { radial: true },
                        ToolId::Move => Tool::Move,
                        ToolId::Crop => Tool::Crop,
                    };
                    self.set_tool(tool);
                }
                Action::Color(c) => self.brush.color = c,
                Action::Brush(m) => {
                    let b = &mut self.brush;
                    (b.size, b.hardness, b.flow, b.opacity, b.spacing, b.smoothing) =
                        (m.size, m.hardness, m.flow, m.opacity, m.spacing, m.smoothing);
                    (b.tip, b.angle, b.roundness, b.follow_direction) = (m.tip, m.angle, m.roundness, m.follow_direction);
                    (b.size_jitter, b.angle_jitter, b.scatter, b.flow_jitter) =
                        (m.size_jitter, m.angle_jitter, m.scatter, m.flow_jitter);
                }
                Action::SmudgeStrength(v) => self.smudge_strength = v,
                Action::WandTolerance(v) => self.wand_tolerance = v,
                Action::WandContiguous(v) => self.wand_contiguous = v,
                Action::SelectLayer(id) => {
                    if let Some(e) = self.engine() {
                        e.select(|d| d.active = id);
                    }
                    let has_mask = self.engine().and_then(|e| e.doc().find(id).map(|l| l.mask.is_some()));
                    if has_mask == Some(false) {
                        self.mask_edit = false;
                    }
                }
                Action::ToggleVisible(id) => self.edit("Sichtbarkeit", |d| {
                    let l = d.find_mut(id).unwrap();
                    l.visible = !l.visible;
                    true
                }),
                Action::ToggleMaskEdit(id) => {
                    let was_active = self.engine().is_some_and(|e| e.doc().active == id);
                    if let Some(e) = self.engine() {
                        e.select(|d| d.active = id);
                    }
                    self.mask_edit = !(was_active && self.mask_edit);
                    tracing::info!("Ziel: {}", if self.mask_edit { "Ebenenmaske" } else { "Pixel" });
                }
                Action::RenameLayer(id, name) => self.edit("Umbenennen", |d| {
                    let l = d.find_mut(id).unwrap();
                    let changed = l.name != name;
                    l.name = name;
                    changed
                }),
                Action::MoveLayer { id, target, above, into } => {
                    self.edit("Ebene verschieben", |d| d.move_layer(id, target, above, into))
                }
                Action::LayerOpacity { value, done } => {
                    if let Some(e) = self.engine() {
                        let changed = e.edit_merged("Deckkraft", "layer-opacity", |d| {
                            let id = d.active;
                            let l = d.find_mut(id).unwrap();
                            let c = l.opacity != value;
                            l.opacity = value;
                            c
                        });
                        if done {
                            e.end_merge();
                        }
                        if changed {
                            self.touch();
                        }
                    }
                }
                Action::LayerBlend(b) => self.edit("Füllmethode", |d| {
                    let id = d.active;
                    let l = d.find_mut(id).unwrap();
                    let c = l.blend != b;
                    l.blend = b;
                    c
                }),
                Action::DeleteLayer => self.edit("Ebene löschen", |d| d.remove_active()),
                Action::HistoryJump(n) => {
                    if self.engine().is_some_and(|e| e.jump_history(n)) {
                        self.touch();
                    }
                }
                Action::CommitBox => self.commit_box(),
                Action::CancelBox => self.cancel_box(),
                Action::CanvasSize { w, h, anchor } => {
                    let Some((ow, oh)) = self.session.as_ref().map(|s| s.engine.size()) else { continue };
                    let x = ((ow as f64 - w as f64) * anchor.0).round() as i64;
                    let y = ((oh as f64 - h as f64) * anchor.1).round() as i64;
                    self.engine_op(|e| e.crop(x, y, w, h, "Arbeitsfläche"));
                    self.view_fitted = false;
                }
                Action::AddAdjustment(adj) => {
                    let label = format!("Einstellungsebene {}", adj.name());
                    if self.engine().and_then(|e| e.add_adjustment(adj, &label)).is_some() {
                        self.touch();
                    }
                }
                Action::SetAdjustment { adj, done } => {
                    if let Some(e) = self.engine() {
                        let id = e.doc().active;
                        let changed = e.edit_merged("Einstellung ändern", &format!("adjust-{id}"), |d| {
                            let l = d.find_mut(id).unwrap();
                            let c = l.adjustment() != Some(&adj);
                            l.content = imagine_core::Content::Adjustment(adj);
                            c
                        });
                        if done {
                            e.end_merge();
                        }
                        if changed {
                            self.touch();
                        }
                    }
                }
                Action::ApplyFilter(f) => {
                    let t0 = Instant::now();
                    let label = f.name();
                    self.engine_op(|e| e.apply_filter(&f, label));
                    tracing::info!("{label}: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                }
                Action::AddSmartFilter(f) => {
                    let label = f.name();
                    self.engine_op(|e| e.add_smart_filter(&f, label));
                }
                Action::SetSmartFilter { layer, index, filter } => {
                    self.engine_op(|e| e.set_smart_filter(layer, index, &filter, "Smart-Filter ändern"));
                }
                Action::ToggleSmartFilter { layer, index } => self.edit("Smart-Filter ein/aus", |d| {
                    d.find_mut(layer).and_then(|l| l.filters.get_mut(index)).map(|f| f.enabled = !f.enabled).is_some()
                }),
                Action::RemoveSmartFilter { layer, index } => self.edit("Smart-Filter entfernen", |d| {
                    d.find_mut(layer).filter(|l| index < l.filters.len()).map(|l| l.filters.remove(index)).is_some()
                }),
                Action::BakeSmartFilters(layer) => self.engine_op(|e| e.bake_smart_filters(layer, "Smart-Filter anwenden")),
                Action::OpenFile(p) => self.guard_unsaved(Pending::Open(p)),
                Action::NewDocument { w, h, white } => self.guard_unsaved(Pending::New { w, h, white }),
                Action::SaveAs(p) => self.save_to(Some(p)),
                Action::ExportAs(p, opts) => self.export_to(p, opts),
                Action::Unsaved(a) => self.answer_unsaved(a),
                Action::ImageSize { w, h } => {
                    self.engine_op(|e| e.resize_image(w, h, Interp::Bicubic, "Bildgröße"));
                    self.view_fitted = false;
                }
            }
        }
    }

    /// Builds the interface for this frame and applies what the user did.
    fn run_ui(&mut self) {
        // Thumbnails: a few per frame, not while dragging.
        if matches!(self.drag, Drag::None)
            && let Some(s) = &mut self.session
            && s.engine.update_thumbnails(4)
        {
            self.needs_redraw = true;
        }
        let Some(mut model) = self.ui_model() else { return };
        if let Some(s) = &mut self.session {
            if ui::adjust_ui::wants_histogram(model.active_adjustment.as_ref()) {
                model.histogram = Some(s.engine.histogram(Some(model.active)));
            }
            for row in &mut model.layers {
                if let Some((slot, uv)) = s.engine.thumbnail(row.id) {
                    let id = *self.thumb_ids.entry(slot).or_insert_with(|| {
                        let view = s.engine.thumbnail_view(slot);
                        self.gui.register_texture(&self.gpu.device, &view)
                    });
                    row.thumb = Some((id, uv));
                }
            }
        }
        let size = (self.surface_config.width, self.surface_config.height);
        let changed = self.last_model.as_ref() != Some(&model) || ui::log::generation() != self.ui_log_gen;
        if !self.gui.needs_pass(size, self.scale, changed) {
            return;
        }
        let mut actions = Vec::new();
        let state = &mut self.ui_state;
        self.gui.run(size, self.scale, |ctx| actions = ui::panels::build(ctx, &model, state));
        let s = self.scale;
        let rect = match self.ui_state.canvas {
            Some(r) => (DVec2::new(r.min.x as f64 * s, r.min.y as f64 * s), DVec2::new(r.max.x as f64 * s, r.max.y as f64 * s)),
            None => (DVec2::ZERO, DVec2::new(size.0 as f64, size.1 as f64)),
        };
        if rect != self.canvas_rect {
            // First layout: fit the image into the free area.
            if self.canvas_rect.1 == DVec2::ZERO {
                self.view_fitted = false;
            }
            self.canvas_rect = rect;
        }
        self.ui_log_gen = ui::log::generation();
        self.last_model = Some(model);
        self.apply_ui(actions);
        self.fit_if_needed();
    }

    fn draw(&mut self) {
        let t0 = Instant::now();
        let busy = self.session.as_ref().is_some_and(|s| s.engine.busy());
        if self.bench_on_load && !self.loading && self.pending_uploads.is_empty() && !busy {
            self.bench_on_load = false;
            if let Some(s) = &mut self.session {
                s.engine.clear_history();
            }
            self.start_bench(true);
        }
        self.step_bench();

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure_surface();
                return;
            }
            _ => return,
        };
        let target = frame.texture.create_view(&Default::default());

        let show_ring = self.tool.paints()
            && !matches!(self.drag, Drag::Pan { .. } | Drag::Rotate { .. } | Drag::Zoom { .. })
            && !self.space
            && self.bench.is_none();
        let ring_at = match self.drag {
            Drag::BrushHud { origin, .. } => Some(origin),
            _ => self.cursor.filter(|_| show_ring),
        };
        let overlay = Overlay {
            cursor: ring_at.map(|p| (p, self.brush.size * 0.5 * self.view.zoom as f32)),
            checker_px: (8.0 * self.scale) as f32,
            lines: self.overlay_lines(),
            show_selection: self.bench.is_none(),
            quick_mask: self.quick_mask,
            time: self.started.elapsed().as_secs_f32(),
        };
        self.run_ui();
        let preview = self.preview_transform();
        if let Some(s) = &mut self.session {
            let mut batch: Vec<(LayerId, TileData)> = Vec::new();
            for _ in 0..UPLOADS_PER_FRAME {
                match self.pending_uploads.pop_front() {
                    Some(u) => batch.push(u),
                    None => break,
                }
            }
            for (layer, group) in batch.chunk_by(|a, b| a.0 == b.0).map(|g| (g[0].0, g)) {
                let tiles: Vec<TileData> = group.iter().map(|(_, t)| t.clone()).collect();
                self.upload_failures += s.add_tiles(layer, &tiles);
            }
            if self.pending_uploads.is_empty() && !self.loading && self.upload_failures > 0 {
                tracing::warn!("{} Kacheln passten nicht in den GPU-Speicher", self.upload_failures);
                self.upload_failures = 0;
            }
            let result = match &self.ui_state.filter_preview {
                Some(ui::panels::FilterPreview::Pixels(f)) => s.engine.set_filter_preview(Some(f)),
                Some(ui::panels::FilterPreview::Smart(i, f)) => s.engine.set_smart_filter_preview(Some((*i, f))),
                None => s.engine.set_transform_preview(preview.map(|f| (f, Interp::Bicubic))),
            };
            if let Err(e) = result {
                tracing::warn!("Vorschau: {e}");
            }
            s.engine.render(&target, &self.view, &overlay);
        } else {
            clear(&self.gpu, &target);
        }
        let size = (self.surface_config.width, self.surface_config.height);
        self.gui.paint(&self.gpu.device, &self.gpu.queue, &target, size);

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

        self.needs_redraw = self.gui.repaint_at.is_some_and(|t| t <= Instant::now() + Duration::from_millis(4))
            || self.gui.has_events()
            || ui::log::generation() != self.ui_log_gen
            || !self.pending_uploads.is_empty()
            || self.bench.is_some()
            || (self.bench_on_load && !self.loading)
            || self.session.as_ref().is_some_and(|s| s.engine.busy());
        if now.duration_since(self.title_at).as_millis() > 250 {
            self.title_at = now;
            self.update_title();
            if let Some(s) = &self.session {
                let e = &s.engine;
                self.ui_status = format!(
                    "{} Kacheln · {:.0} MB GPU · {:.2} ms",
                    e.layer_tiles_live(),
                    e.gpu_bytes() as f64 / 1e6,
                    self.frame_cpu_ms
                );
            }
        }
    }

    /// Rubber bands for the drag in progress, in screen pixels.
    fn overlay_lines(&self) -> Vec<OverlayLine> {
        let v = &self.view;
        let rect = |a: DVec2, b: DVec2| {
            let (min, max) = (a.min(b), a.max(b));
            [min, DVec2::new(max.x, min.y), max, DVec2::new(min.x, max.y)].map(|p| v.doc_to_screen(p)).to_vec()
        };
        match &self.drag {
            Drag::Marquee { start, end, ellipse: false, .. } => vec![OverlayLine { points: rect(*start, *end), closed: true }],
            Drag::Marquee { start, end, ellipse: true, .. } => {
                let (c, r) = ((*start + *end) * 0.5, (*end - *start).abs() * 0.5);
                let points = (0..96)
                    .map(|i| {
                        let t = i as f64 / 96.0 * std::f64::consts::TAU;
                        v.doc_to_screen(c + DVec2::new(r.x * t.cos(), r.y * t.sin()))
                    })
                    .collect();
                vec![OverlayLine { points, closed: true }]
            }
            Drag::Lasso { points, .. } => {
                vec![OverlayLine { points: points.iter().map(|p| v.doc_to_screen(*p)).collect(), closed: true }]
            }
            Drag::Gradient { start, end } => {
                vec![OverlayLine { points: vec![v.doc_to_screen(*start), v.doc_to_screen(*end)], closed: false }]
            }
            _ => match &self.boxed {
                Some((_, b)) => b.lines(v, self.scale),
                None => Vec::new(),
            },
        }
    }

    fn update_title(&self) {
        let Some(s) = &self.session else { return };
        let e = &s.engine;
        let (w, h) = e.size();
        let tool = format!("{}{}", self.tool.name(), if self.mask_edit { " → Maske" } else { "" });
        let layer = e
            .doc()
            .active_layer()
            .map(|l| {
                format!(
                    "„{}“ {} {:.0} %{}",
                    l.name,
                    l.blend.name(),
                    l.opacity * 100.0,
                    if l.visible { "" } else { " (aus)" }
                )
            })
            .unwrap_or_default();
        let status = if self.quit_armed.is_some_and(|t| t.elapsed() < QUIT_CONFIRM) {
            " – NOCHMAL Strg+Q ZUM VERWERFEN"
        } else if self.save_job.is_some() || self.saving {
            " – speichert …"
        } else if self.loading || !self.pending_uploads.is_empty() {
            " – lädt …"
        } else if self.bg_running > 0 {
            " – rechnet …"
        } else {
            ""
        };
        self.window.set_title(format!(
            "{}{} – {w}×{h} – {:.1} % – {:.0}° – {layer} – {tool} {:.0} px {:.0} % – {}+{} Kacheln ({:.0} MB) – {:.2} ms{status}",
            if self.modified { "• " } else { "" },
            s.name,
            self.view.zoom * 100.0 / self.scale,
            self.view.rotation.to_degrees(),
            self.brush.size,
            self.brush.opacity * 100.0,
            e.layer_tiles_live(),
            e.composite_tiles_live(),
            e.gpu_bytes() as f64 / 1e6,
            self.frame_cpu_ms,
        ));
    }
}

/// Benchmark camera path: fit → ~128× zoom and back, half a turn, some panning.
pub fn bench_view(base: &View, frame: u32, frames: u32) -> View {
    let t = frame as f64 / frames as f64 * std::f64::consts::TAU;
    let mut v = *base;
    v.zoom = base.zoom * 2f64.powf((1.0 - t.cos()) * 3.5);
    v.rotation = t * 0.5;
    v.center = base.center + DVec2::new(t.sin(), (2.0 * t).sin() * 0.5) * base.viewport / base.zoom * 0.25;
    v
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
        self.request_quit();
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
        self.key(&ev, true);
    }

    fn repeat_key(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_keyboard::WlKeyboard, _: u32, ev: KeyEvent) {
        if self.ui_state.captures_keyboard() || self.gui.wants_keyboard() {
            self.key(&ev, true);
        }
    }

    fn release_key(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_keyboard::WlKeyboard, _: u32, ev: KeyEvent) {
        self.key(&ev, false);
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
        self.gui.set_modifiers(mods.ctrl, mods.shift, mods.alt);
    }
}

impl PointerHandler for App {
    fn pointer_frame(&mut self, conn: &Connection, _: &QueueHandle<Self>, _: &wl_pointer::WlPointer, events: &[PointerEvent]) {
        for ev in events {
            if &ev.surface != self.window.wl_surface() {
                continue;
            }
            let pos = self.to_physical(ev.position);
            let scale = self.scale;
            self.needs_redraw = true;
            match &ev.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => self.gui.pointer_moved(pos, scale),
                PointerEventKind::Leave { .. } => self.gui.pointer_gone(),
                _ => {}
            }
            let canvas_busy = !matches!(self.drag, Drag::None);
            let over_ui = self.over_ui(pos);
            match &ev.kind {
                PointerEventKind::Press { button, .. } | PointerEventKind::Release { button, .. }
                    if !matches!(*button, BTN_LEFT | BTN_MIDDLE) && (over_ui || self.ui_pointer) =>
                {
                    continue; // other buttons do nothing on the interface
                }
                PointerEventKind::Press { button, .. } if !self.tablet_contact && (over_ui || self.ui_pointer) && !canvas_busy => {
                    self.ui_pointer = true;
                    self.gui.pointer_button(pos, scale, *button == BTN_MIDDLE, true);
                    continue;
                }
                PointerEventKind::Release { button, .. } if self.ui_pointer => {
                    self.ui_pointer = false;
                    self.gui.pointer_button(pos, scale, *button == BTN_MIDDLE, false);
                    continue;
                }
                PointerEventKind::Motion { .. } if (over_ui || self.ui_pointer) && !canvas_busy => {
                    self.cursor = None;
                    continue;
                }
                PointerEventKind::Axis { vertical, .. } if over_ui && !canvas_busy => {
                    let steps = if vertical.value120 != 0 {
                        vertical.value120 as f64 / 120.0
                    } else if vertical.discrete != 0 {
                        vertical.discrete as f64
                    } else {
                        vertical.absolute / 15.0
                    };
                    self.gui.scroll(steps);
                    continue;
                }
                // Canvas clicks also reach egui (closes open menus and popups).
                PointerEventKind::Press { button, .. } if !self.tablet_contact && matches!(*button, BTN_LEFT | BTN_MIDDLE) => {
                    self.gui.pointer_button(pos, scale, *button == BTN_MIDDLE, true);
                }
                PointerEventKind::Release { button, .. } if !self.tablet_contact && matches!(*button, BTN_LEFT | BTN_MIDDLE) => {
                    self.gui.pointer_button(pos, scale, *button == BTN_MIDDLE, false);
                }
                _ => {}
            }
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
                    BTN_RIGHT if self.mods.alt && self.tool.paints() => {
                        self.drag = Drag::BrushHud {
                            origin: pos,
                            start: pos,
                            size: self.brush.size,
                            hardness: self.brush.hardness,
                        };
                    }
                    BTN_LEFT => self.press(pos, 1.0, Button::Primary, false),
                    BTN_MIDDLE => self.press(pos, 1.0, Button::Middle, false),
                    _ => {}
                },
                PointerEventKind::Release { button, .. } if !self.tablet_contact => {
                    if matches!(*button, BTN_LEFT | BTN_MIDDLE | BTN_RIGHT) {
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
