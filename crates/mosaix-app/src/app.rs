//! Window, event loop, input handling and frame scheduling.
//!
//! Wayland is used directly through smithay-client-toolkit. Frames are driven
//! by `wl_surface.frame` callbacks: input is batched per event-loop dispatch
//! and at most one frame is rendered per compositor refresh.

pub(crate) mod clipboard;
mod brushes;
pub(crate) mod patterns;
pub(crate) mod filters;
pub(crate) mod actions;
pub(crate) mod pen;
pub(crate) mod textedit;

use std::collections::{HashMap, HashSet, VecDeque};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use anyhow::Context;
use glam::{DAffine2, DVec2};
use mosaix_core::color::linear_to_srgb;
use mosaix_core::mask::{CombineOp, CpuMask, Shape};
use mosaix_core::{BlendSpace, BrushSettings, InputSample, LayerId, Stroke, View, tr};
use mosaix_gpu::{
    Engine, GpuContext, GradientKind, Interp, LiquifyTool, Overlay, OverlayLine, SampleFrom,
    StrokeParams, StrokeTarget,
};
use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
};
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
        pointer::{
            CursorIcon, PointerEvent, PointerEventKind, PointerHandler, ThemeSpec, ThemedPointer,
        },
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
    tablet::zv2::client::{
        zwp_tablet_manager_v2::ZwpTabletManagerV2, zwp_tablet_seat_v2::ZwpTabletSeatV2,
    },
    viewporter::client::{wp_viewport::WpViewport, wp_viewporter::WpViewporter},
};

use crate::agent::{command, ipc};
use crate::boxedit::{self, BoxEdit};
use crate::loader::{self, LoaderMsg, Source, TileData};
use crate::session::{ExportJob, SaveDone, SaveJob, Session};
use crate::settings::{self, Settings};
use crate::tablet::{self, TabletFrame};
use crate::ui::{
    self,
    panels::{Action, ToolId},
};

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
    /// Liquify brush: last position (document px) and pressure.
    Liquify {
        last: DVec2,
        pressure: f32,
    },
    /// Shape tool: dragging out a rectangle / ellipse (document px).
    Shape {
        start: DVec2,
        end: DVec2,
    },
    /// Selecting text with the mouse while editing it on the canvas.
    TextSelect,
    /// Pen tool: placing a node's handles, moving nodes or handles.
    Pen(pen::PenDrag),
    /// Eyedropper: picks the colour under the pointer while dragging.
    Eyedrop,
    Pan {
        last: DVec2,
    },
    Rotate {
        last: DVec2,
    },
    Zoom {
        last: DVec2,
        anchor: DVec2,
    },
    /// Document coordinates.
    Marquee {
        start: DVec2,
        end: DVec2,
        op: CombineOp,
        ellipse: bool,
    },
    Lasso {
        points: Vec<DVec2>,
        op: CombineOp,
    },
    Gradient {
        start: DVec2,
        end: DVec2,
    },
    Move {
        start: DVec2,
        end: DVec2,
    },
    /// Dragging a free-transform or crop handle.
    Box,
    /// Alt + right drag: brush size (horizontal) and hardness (vertical), shown at `origin`.
    BrushHud {
        origin: DVec2,
        start: DVec2,
        size: f32,
        hardness: f32,
    },
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
    Clone,
    Heal,
    SpotHeal,
    Text,
    Shape { ellipse: bool },
    Pen,
    Eyedropper,
    /// Local AI: click an object to select it.
    ObjectSelect,
    /// Local AI: paint over something, it is removed (LaMa).
    Remove,
    Dodge,
    Burn,
    Sponge,
    Blur,
    Sharpen,
}

impl Tool {
    fn name(self) -> &'static str {
        match self {
            Tool::Brush => tr("Brush"),
            Tool::Eraser => tr("Eraser"),
            Tool::Marquee { ellipse: false } => tr("Rectangular Marquee"),
            Tool::Marquee { ellipse: true } => tr("Elliptical Marquee"),
            Tool::Lasso => tr("Lasso"),
            Tool::Gradient { radial: false } => tr("Linear Gradient"),
            Tool::Gradient { radial: true } => tr("Radial Gradient"),
            Tool::Move => tr("Move"),
            Tool::Crop => tr("Crop"),
            Tool::Wand => tr("Magic Wand"),
            Tool::Smudge => tr("Smudge"),
            Tool::Clone => tr("Clone Stamp"),
            Tool::Heal => tr("Healing Brush"),
            Tool::SpotHeal => tr("Spot Healing"),
            Tool::Text => tr("Text"),
            Tool::Shape { ellipse: false } => tr("Rectangle"),
            Tool::Shape { ellipse: true } => tr("Ellipse"),
            Tool::Pen => tr("Pen"),
            Tool::Eyedropper => tr("Eyedropper"),
            Tool::ObjectSelect => tr("Object Selection"),
            Tool::Remove => tr("Remove"),
            Tool::Dodge => tr("Dodge"),
            Tool::Burn => tr("Burn"),
            Tool::Sponge => tr("Sponge"),
            Tool::Blur => tr("Blur"),
            Tool::Sharpen => tr("Sharpen"),
        }
    }

    fn paints(self) -> bool {
        matches!(
            self,
            Tool::Brush
                | Tool::Eraser
                | Tool::Smudge
                | Tool::Clone
                | Tool::Heal
                | Tool::SpotHeal
                | Tool::Remove
                | Tool::Dodge
                | Tool::Burn
                | Tool::Sponge
                | Tool::Blur
                | Tool::Sharpen
        )
    }
}

/// Result of a background job.
enum BgDone {
    /// The system file chooser was answered.
    FilePicked {
        req: crate::portal::Request,
        picked: crate::portal::Picked,
    },
    /// A new selection computed from the one identified by `base`.
    Selection {
        label: &'static str,
        base: usize,
        mask: anyhow::Result<CpuMask>,
        started: Instant,
    },
    Exported {
        path: std::path::PathBuf,
        result: anyhow::Result<f64>,
    },
    Autosaved {
        path: std::path::PathBuf,
        result: anyhow::Result<()>,
    },
    Filled {
        result: Result<mosaix_gpu::FillResult, String>,
        started: Instant,
        /// Content-Aware Fill or Remove.
        label: &'static str,
    },
    /// Clipboard data read from another program: (mime type, bytes).
    Pasted(Result<(String, Vec<u8>), String>),
    /// A layer mask computed by a model (Remove Background).
    LayerMask {
        layer: mosaix_core::LayerId,
        mask: anyhow::Result<CpuMask>,
        started: Instant,
    },
    ModelDownloaded {
        name: &'static str,
        result: anyhow::Result<()>,
    },
    Upscaled {
        result: anyhow::Result<mosaix_gpu::UpscaleResult>,
        started: Instant,
    },
}

/// The rectangle of a shape drag: Shift = square, Alt = from the centre.
fn shape_rect(start: DVec2, end: DVec2, square: bool, centre: bool) -> (DVec2, DVec2) {
    let mut d = end - start;
    if square {
        let m = d.x.abs().max(d.y.abs());
        d = DVec2::new(m.copysign(d.x), m.copysign(d.y));
    }
    let (a, b) = if centre {
        (start - d, start + d)
    } else {
        (start, start + d)
    };
    (a.min(b), a.max(b))
}

/// An action that replaces or closes the document, held back while the
/// user decides about unsaved changes.
#[derive(Debug, Clone)]
enum Pending {
    Quit,
    Open(std::path::PathBuf),
    New { w: u32, h: u32, white: bool },
}

/// Recovery copies: `~/.local/state/mosaix/autosave/` (or `$XDG_STATE_HOME`).
fn autosave_dir() -> std::path::PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/state"))
        })
        .unwrap_or_else(std::env::temp_dir);
    mosaix_core::paths::app_dir(&base).join("autosave")
}

/// Mentions recovery copies left behind by windows that no longer run.
fn report_orphaned_autosaves() {
    let Ok(dir) = std::fs::read_dir(autosave_dir()) else {
        return;
    };
    for e in dir.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let pid = name
            .rsplit_once('-')
            .and_then(|(_, p)| p.strip_suffix(".mosaix").or_else(|| p.strip_suffix(".imagine")))
            .and_then(|p| p.parse::<u32>().ok());
        if pid.is_some_and(|p| !std::path::Path::new(&format!("/proc/{p}")).exists()) {
            tracing::warn!(
                "{}",
                tr!(
                    "Recovery copy found: mosaix {path}",
                    path = e.path().display()
                )
            );
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
    /// For key repeat (the keyboard is made once the seat shows up).
    loop_handle: smithay_client_toolkit::reexports::calloop::LoopHandle<'static, App>,
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
    /// Move tool: bounds of what moves, for snapping (document px).
    move_bounds: Option<(DVec2, DVec2)>,
    /// Navigator: its texture (view + UI id) and when it was last drawn.
    nav: Option<(wgpu::Texture, wgpu::TextureView, egui::TextureId)>,
    nav_drawn: Option<(Instant, u64)>,
    /// Info panel: histogram (for an edit count) and the colour under the pointer.
    info_hist: Option<(u64, std::sync::Arc<mosaix_gpu::Histogram>)>,
    info_color: Option<((i64, i64), [f32; 4])>,
    /// The last undo step is a Color Range preview (replaced by the next one).
    range_preview: bool,
    /// Whether an OpenAI key is saved or in the environment (for the dialog).
    openai_key: settings::KeyState,
    wand_tolerance: f32,
    smudge_strength: f32,
    /// Clone stamp: source point (document px), the aligned offset once a
    /// stroke used it, and the options.
    clone_source: Option<DVec2>,
    clone_offset: Option<[i32; 2]>,
    clone_all: bool,
    clone_aligned: bool,
    /// Font families for the text properties (loaded on first use).
    fonts: std::sync::Arc<Vec<mosaix_vector::FontGroup>>,
    /// Text being edited on the canvas.
    text_edit: Option<textedit::TextEdit>,
    clipboard: clipboard::Clipboard,
    pen: pen::Pen,
    liq_tool: LiquifyTool,
    /// Liquify brush diameter (document px) and strength 0…1.
    liq_size: f32,
    liq_strength: f32,
    liq_tick: Instant,
    wand_contiguous: bool,
    /// Background colour (linear RGB); X swaps it with the brush colour.
    bg_color: [f32; 3],
    /// Eyedropper: averaging radius (0 = point) and whether it samples all layers.
    eyedrop_radius: i64,
    eyedrop_all: bool,
    started: Instant,

    // Saving
    save_job: Option<SaveJob>,
    saving: bool,
    save_tx: channel::Sender<anyhow::Result<SaveDone>>,
    /// CPU-heavy work (feathering, wand fill, export encoding) off the UI thread.
    bg_tx: channel::Sender<BgDone>,
    bg_running: usize,
    /// The system file chooser is open.
    picking: bool,
    /// Show the start screen once the first document is there.
    home_pending: bool,
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
    /// Mask thumbnails as UI textures: content key and texture.
    mask_thumbs: HashMap<LayerId, (usize, egui::TextureHandle)>,
    /// Names of the custom brush tips (tips FIRST_CUSTOM_TIP, …).
    custom_tips: Vec<String>,
    /// Pattern library: built-ins and the user's patterns.
    patterns: std::sync::Arc<Vec<mosaix_core::pattern::Pattern>>,
    /// Steps of the action being recorded.
    recording: Option<Vec<serde_json::Value>>,
    /// Macros and scripts in the actions folder.
    actions: std::sync::Arc<Vec<actions::Entry>>,
    /// Filter plugins (built-in examples and the user's folder).
    filter_plugins: std::sync::Arc<Vec<filters::Plugin>>,
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

pub fn run(source: Source, bench: bool, home: bool) -> anyhow::Result<()> {
    let conn =
        Connection::connect_to_env().context(tr("no Wayland connection (WAYLAND_DISPLAY?)"))?;
    let (globals, event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();
    let mut event_loop: EventLoop<App> = EventLoop::try_new()?;
    WaylandSource::new(conn.clone(), event_queue)
        .insert(event_loop.handle())
        .map_err(|e| anyhow::anyhow!("wayland source: {e}"))?;

    let compositor = CompositorState::bind(&globals, &qh).context(tr("wl_compositor missing"))?;
    let xdg = XdgShell::bind(&globals, &qh).context(tr("xdg_shell missing"))?;
    let shm = Shm::bind(&globals, &qh).context(tr("wl_shm missing"))?;

    let wl_surface = compositor.create_surface(&qh);
    let window = xdg.create_window(wl_surface, WindowDecorations::RequestServer, &qh);
    window.set_title("Mosaix");
    window.set_app_id("io.github.mosaix");
    window.set_min_size(Some((320, 240)));

    // Fractional scaling (wp_fractional_scale + wp_viewporter) where available.
    let viewporter: Option<WpViewporter> = globals.bind(&qh, 1..=1, tablet::Ignore).ok();
    let frac_mgr: Option<WpFractionalScaleManagerV1> =
        globals.bind(&qh, 1..=1, tablet::Ignore).ok();
    let (viewport, fractional) = match (viewporter, frac_mgr) {
        (Some(vp), Some(fm)) => (
            Some(vp.get_viewport(window.wl_surface(), &qh, tablet::Ignore)),
            Some(fm.get_fractional_scale(window.wl_surface(), &qh, FractionalData)),
        ),
        _ => (None, None),
    };
    let tablet_manager: Option<ZwpTabletManagerV2> = globals.bind(&qh, 1..=1, tablet::Ignore).ok();
    if tablet_manager.is_none() {
        tracing::warn!(
            "{}",
            tr("Compositor does not support tablet-v2 – the pen only works as a mouse")
        );
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
        loop_handle: event_loop.handle(),
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
        move_bounds: None,
        nav: None,
        nav_drawn: None,
        info_hist: None,
        info_color: None,
        range_preview: false,
        openai_key: settings::key_state(),
        wand_tolerance: 32.0,
        smudge_strength: 0.7,
        clone_source: None,
        clone_offset: None,
        clone_all: false,
        clone_aligned: true,
        fonts: Default::default(),
        text_edit: None,
        pen: Default::default(),
        clipboard: clipboard::Clipboard::new(
            smithay_client_toolkit::data_device_manager::DataDeviceManagerState::bind(
                &globals, &qh,
            )
            .ok(),
        ),
        liq_tool: LiquifyTool::Push,
        liq_size: 200.0,
        liq_strength: 0.6,
        liq_tick: Instant::now(),
        wand_contiguous: true,
        bg_color: [1.0; 3],
        eyedrop_radius: 0,
        eyedrop_all: true,
        started: Instant::now(),
        save_job: None,
        saving: false,
        save_tx,
        bg_tx,
        bg_running: 0,
        picking: false,
        home_pending: home,
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
        mask_thumbs: HashMap::new(),
        custom_tips: Vec::new(),
        patterns: std::sync::Arc::new(patterns::load_library(&patterns::pattern_dir())),
        filter_plugins: std::sync::Arc::new(filters::load_library(&filters::filter_dir())),
        recording: None,
        actions: std::sync::Arc::new(actions::list(&actions::actions_dir())),
        ui_status: String::new(),
        socket: None,
        bench: None,
        bench_on_load: bench,
        title_at: Instant::now(),
        frame_cpu_ms: 0.0,
    };
    match ipc::bind() {
        Ok((listener, guard)) => {
            tracing::info!(
                "{}",
                tr!(
                    "Agent socket: {path} (mosaix mcp --attach)",
                    path = guard.path().display()
                )
            );
            app.socket = Some(guard);
            let handle = event_loop.handle();
            event_loop
                .handle()
                .insert_source(
                    Generic::new(listener, Interest::READ, Mode::Level),
                    move |_, listener, _app: &mut App| {
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
                    },
                )
                .map_err(|e| anyhow::anyhow!("agent socket: {e}"))?;
        }
        Err(e) => tracing::warn!(
            "{}",
            tr!(
                "Agent socket not available: {error}",
                error = format!("{e:#}")
            )
        ),
    }
    let seats: Vec<_> = app.seat_state.seats().collect();
    for seat in seats {
        app.bind_tablet_seat(&seat);
    }

    app.apply_settings();
    report_orphaned_autosaves();
    while !app.exit {
        // Wake up periodically while a save is in flight or ants are marching.
        let ants = app
            .session
            .as_ref()
            .is_some_and(|s| s.engine.has_selection());
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
        let ui_wait = app
            .gui
            .repaint_at
            .map(|t| t.saturating_duration_since(Instant::now()));
        let timeout = match (timeout, ui_wait) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        event_loop.dispatch(timeout, &mut app)?;
        app.poll_save();
        app.poll_export();
        app.maybe_autosave();
        if ants
            || app.gui.repaint_at.is_some_and(|t| t <= Instant::now())
            || ui::log::generation() != app.ui_log_gen
        {
            app.needs_redraw = true;
        }
        if app.configured && app.needs_redraw && !app.frame_pending {
            app.draw();
        }
    }
    app.remember_thumbnail();
    // A deliberate quit: the recovery copy is no longer needed.
    app.drop_autosave();
    app.store_settings();
    Ok(())
}

impl App {
    fn bind_tablet_seat(&mut self, seat: &wl_seat::WlSeat) {
        let Some(mgr) = &self.tablet_manager else {
            return;
        };
        if self.tablet_seat_ids.insert(seat.id()) {
            self.tablet_seats
                .push(mgr.get_tablet_seat(seat, &self.qh, tablet::SeatData));
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
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
        match &self.viewport {
            Some(vp) => vp.set_destination(self.logical.0 as i32, self.logical.1 as i32),
            None => self
                .window
                .wl_surface()
                .set_buffer_scale(self.scale.round() as i32),
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

    /// Zooming out stops when the image's long side is ~128 screen pixels
    /// (a free-spinning wheel would otherwise shrink it to an invisible dot).
    fn limit_zoom_out(&mut self, anchor: DVec2) {
        let Some(s) = &self.session else { return };
        let (w, h) = s.engine.size();
        let floor = (128.0 * self.scale / w.max(h).max(1) as f64).min(1.0);
        if self.view.zoom < floor {
            self.view.set_zoom_at(anchor, floor);
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
            tracing::debug!(
                "{}",
                tr!(
                    "Layers (top → bottom), blend space: {space}\n{layers}",
                    space = d.blend_space.name(),
                    layers = d.describe()
                )
            );
        }
    }

    // ---- loader -----------------------------------------------------------

    fn on_loader(&mut self, msg: LoaderMsg) {
        match msg {
            LoaderMsg::Structure {
                doc,
                name,
                source,
                native_len,
                expected_tiles,
            } => {
                tracing::info!(
                    "{}",
                    tr!(
                        "Document {name}: {w}×{h}",
                        name = name,
                        w = doc.width,
                        h = doc.height
                    )
                );
                let engine = Engine::new(
                    &self.gpu.device,
                    &self.gpu.queue,
                    self.surface_config.format,
                    &doc,
                );
                if expected_tiles > engine.max_layer_tiles() {
                    tracing::warn!(
                        "{}",
                        tr!(
                            "Document needs {expected} tiles, the layer store holds {max} – the rest is missing",
                            expected = expected_tiles,
                            max = engine.max_layer_tiles()
                        )
                    );
                }
                if let Some(p) = &source {
                    self.settings.add_recent(p);
                    self.store_settings();
                }
                self.session = Some(Session::new(engine, name, source, native_len));
                // The start screen comes with the first (blank) document and
                // goes when another one is opened.
                self.ui_state.home = std::mem::take(&mut self.home_pending).then(Default::default);
                self.load_custom_tips();
                self.ever_loaded = true;
                self.view_fitted = false;
                self.fit_if_needed();
                self.print_layers();
                self.needs_redraw = true;
            }
            LoaderMsg::Tiles { layer, tiles } => {
                self.pending_uploads
                    .extend(tiles.into_iter().map(|t| (layer, t)));
                self.needs_redraw = true;
            }
            m @ (LoaderMsg::RawDecoded { .. } | LoaderMsg::RawBlobs(_)) => {
                if let Some(s) = &mut self.session {
                    s.add_raw(m);
                }
                self.needs_redraw = true;
            }
            LoaderMsg::Mask(m) => {
                if let Some(s) = &mut self.session
                    && let Err(e) = s.add_mask(m)
                {
                    tracing::warn!(
                        "{}",
                        tr!("Mask not loaded: {error}", error = format!("{e:#}"))
                    );
                }
                self.needs_redraw = true;
            }
            LoaderMsg::Done {
                decode_ms,
                total_ms,
            } => {
                tracing::info!(
                    "{}",
                    tr!(
                        "Loading done: decode {decode} ms, total {total} ms",
                        decode = format!("{decode_ms:.0}"),
                        total = format!("{total_ms:.0}")
                    )
                );
                self.loading = false;
                if std::mem::take(&mut self.fill_white_on_load)
                    && let Some(s) = &mut self.session
                {
                    let _ = s
                        .engine
                        .fill(StrokeTarget::Pixels, [1.0; 4], 1.0, tr("White"));
                    s.engine.clear_history();
                }
                self.needs_redraw = true;
            }
            LoaderMsg::Error(e) => {
                tracing::error!("{}", tr!("Loading failed: {error}", error = e));
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
            Button::Primary if self.space && self.mods.ctrl => Drag::Zoom {
                last: pos,
                anchor: pos,
            },
            Button::Primary if self.space => Drag::Pan { last: pos },
            Button::Primary if self.r_key => Drag::Rotate { last: pos },
            Button::Primary if self.liquifying() => self.begin_liquify(pos, pressure),
            Button::Primary if self.ui_state.color_range_open() => {
                // Color Range: pick the colour to select from the picture.
                let d = self.view.screen_to_doc(pos);
                if d.x >= 0.0
                    && d.y >= 0.0
                    && let Some(c) = self.engine().and_then(|e| e.sample_straight(d.x as u32, d.y as u32, mosaix_gpu::SampleFrom::Image).ok())
                {
                    self.ui_state.color_range_pick = Some([c[0], c[1], c[2]].map(mosaix_core::color::srgb_to_linear));
                }
                Drag::None
            }
            Button::Primary if self.boxed.is_some() || self.tool == Tool::Crop => {
                self.begin_box(pos)
            }
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
        let alt_picks = matches!(
            self.tool,
            Tool::Brush | Tool::Eraser | Tool::Gradient { .. } | Tool::Shape { .. } | Tool::Text
        );
        if self.tool == Tool::Eyedropper || (self.mods.alt && alt_picks) {
            self.eyedrop(doc);
            return Drag::Eyedrop;
        }
        match self.tool {
            Tool::Clone | Tool::Heal if self.mods.alt => {
                self.clone_source = Some(doc);
                self.clone_offset = None;
                tracing::info!(
                    "{}",
                    tr!(
                        "Clone source: {x}, {y}",
                        x = format!("{:.0}", doc.x),
                        y = format!("{:.0}", doc.y)
                    )
                );
                Drag::None
            }
            Tool::Clone | Tool::Heal if self.clone_source.is_none() => {
                tracing::warn!(
                    "{}",
                    tr!(
                        "{tool}: Alt+click to set the source first",
                        tool = self.tool.name()
                    )
                );
                Drag::None
            }
            Tool::Brush
            | Tool::Eraser
            | Tool::Smudge
            | Tool::Clone
            | Tool::Heal
            | Tool::SpotHeal
            | Tool::Remove
            | Tool::Dodge
            | Tool::Burn
            | Tool::Sponge
            | Tool::Blur
            | Tool::Sharpen => {
                self.begin_stroke(pos, pressure, eraser_tool || self.tool == Tool::Eraser)
            }
            Tool::Marquee { ellipse } => Drag::Marquee {
                start: self.snap_point(doc),
                end: self.snap_point(doc),
                op,
                ellipse,
            },
            Tool::Lasso => Drag::Lasso {
                points: vec![doc],
                op,
            },
            Tool::Gradient { .. } => Drag::Gradient {
                start: doc,
                end: doc,
            },
            Tool::Move => {
                self.move_bounds = self.session.as_mut().and_then(|s| {
                    let id = s.engine.doc().active;
                    s.engine
                        .content_bounds(id)
                        .map(|(x0, y0, x1, y1)| (DVec2::new(x0 as f64, y0 as f64), DVec2::new(x1 as f64, y1 as f64)))
                });
                Drag::Move { start: doc, end: doc }
            }
            Tool::Shape { .. } => Drag::Shape {
                start: self.snap_point(doc),
                end: self.snap_point(doc),
            },
            Tool::Pen => self.pen_press(pos, doc),
            // Handled above.
            Tool::Eyedropper => Drag::None,
            Tool::Text => {
                if self.text_edit.is_some() && self.text_edit_press(pos, doc) {
                    return Drag::TextSelect;
                }
                self.end_text_edit();
                self.text_click(doc);
                Drag::None
            }
            Tool::Crop => Drag::None,
            Tool::Wand => {
                let (x, y) = (doc.x.floor(), doc.y.floor());
                if x >= 0.0 && y >= 0.0 {
                    self.magic_wand(x as u32, y as u32, op);
                }
                Drag::None
            }
            Tool::ObjectSelect => {
                self.ai_select(Some(doc), op);
                Drag::None
            }
        }
    }

    fn liquifying(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|s| s.engine.liquify_active())
    }

    /// Ctrl+Shift+X: starts liquifying the active layer.
    fn begin_liquify_mode(&mut self) {
        if self.liquifying() || self.boxed.is_some() {
            return;
        }
        let Some(e) = self.engine() else { return };
        match e.liquify_begin() {
            Ok(()) => tracing::info!(
                "{}",
                tr(
                    "Liquify: W forward warp, R reconstruct, C twirl (Alt: counterclockwise), S pucker, B bloat (Alt swaps), [ ] size. Enter applies, Esc cancels, Ctrl+Z undoes the last stroke."
                )
            ),
            Err(e) => tracing::warn!("{}", tr!("Cannot liquify: {error}", error = e)),
        }
        self.needs_redraw = true;
    }

    fn commit_liquify(&mut self) {
        self.engine_op(|e| e.liquify_commit(tr("Liquify")));
        self.needs_redraw = true;
    }

    fn cancel_liquify(&mut self) {
        if let Some(e) = self.engine() {
            e.liquify_cancel();
        }
        tracing::info!("{}", tr("Liquify cancelled"));
        self.needs_redraw = true;
    }

    /// The liquify tool in effect (Alt reverses twirl and pucker/bloat).
    fn liq_effective(&self) -> LiquifyTool {
        match (self.liq_tool, self.mods.alt) {
            (LiquifyTool::TwirlCw, true) => LiquifyTool::TwirlCcw,
            (LiquifyTool::TwirlCcw, true) => LiquifyTool::TwirlCw,
            (LiquifyTool::Pucker, true) => LiquifyTool::Bloat,
            (LiquifyTool::Bloat, true) => LiquifyTool::Pucker,
            (t, _) => t,
        }
    }

    fn begin_liquify(&mut self, pos: DVec2, pressure: f32) -> Drag {
        let doc = self.view.screen_to_doc(pos);
        if let Some(e) = self.engine() {
            e.liquify_stroke_start();
        }
        self.liq_tick = Instant::now();
        self.liquify_continuous(doc, pressure, 1.0);
        Drag::Liquify {
            last: doc,
            pressure,
        }
    }

    /// One time-based dab of a continuous tool (twirl, pucker, …).
    fn liquify_continuous(&mut self, at: DVec2, pressure: f32, rate: f64) {
        let tool = self.liq_effective();
        if !tool.continuous() {
            return;
        }
        let (r, s) = (
            self.liq_size as f64 * 0.5,
            (self.liq_strength * pressure) as f64,
        );
        if let Some(e) = self.engine() {
            e.liquify_dab(tool, at, DVec2::ZERO, r, s, rate);
        }
    }

    /// Push dabs along the movement from `from` to `to`.
    fn liquify_push(&mut self, from: DVec2, to: DVec2, pressure: f32) {
        let tool = self.liq_effective();
        if tool.continuous() {
            return;
        }
        let (r, s) = (
            self.liq_size as f64 * 0.5,
            (self.liq_strength * pressure) as f64,
        );
        let d = to - from;
        let n = (d.length() / (r * 0.1).max(1.0)).ceil().max(1.0) as usize;
        let step = d / n as f64;
        if let Some(e) = self.engine() {
            for i in 1..=n {
                e.liquify_dab(tool, from + step * i as f64, step, r, s, 1.0);
            }
        }
    }

    fn box_mods(&self) -> boxedit::Modifiers {
        boxedit::Modifiers {
            shift: self.mods.shift,
            alt: self.mods.alt,
            ctrl: self.mods.ctrl,
        }
    }

    /// Grabs a handle of the transform / crop box (creating the crop box on first use).
    fn begin_box(&mut self, pos: DVec2) -> Drag {
        let doc = self.view.screen_to_doc(pos);
        if self.boxed.is_none() {
            let Some(s) = &self.session else {
                return Drag::None;
            };
            let (w, h) = s.engine.size();
            self.boxed = Some((
                BoxKind::Crop,
                BoxEdit::new(DVec2::ZERO, DVec2::new(w as f64, h as f64), false),
            ));
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
        let Some(l) = e.doc().active_layer() else {
            return;
        };
        let id = l.id;
        // Text and shapes: the box follows their own (rotated, scaled) frame,
        // like in Photoshop, instead of the axis-aligned bounds.
        let own_frame = l.vector().and_then(|v| {
            let m = DAffine2::from_cols_array(&v.transform);
            let mut plain = v.clone();
            plain.transform = mosaix_core::vector::IDENTITY;
            let [x0, y0, x1, y1] = mosaix_vector::bounds(&plain)?;
            (m.matrix2.determinant().abs() > 1e-9 && x1 > x0 && y1 > y0)
                .then(|| BoxEdit::new(DVec2::new(x0, y0), DVec2::new(x1, y1), true).with_origin(m))
        });
        let bounds = e.content_bounds(id).map(|(x0, y0, x1, y1)| {
            BoxEdit::new(
                DVec2::new(x0 as f64, y0 as f64),
                DVec2::new(x1 as f64, y1 as f64),
                true,
            )
        });
        match own_frame.or(bounds) {
            Some(b) => {
                self.boxed = Some((BoxKind::Transform, b));
                tracing::info!(
                    "{}",
                    tr(
                        "Free Transform: corners scale (Shift = free), edges stretch, outside rotates, Alt = around the center. Enter applies, Esc cancels."
                    )
                );
            }
            None => tracing::warn!(
                "{}",
                tr("Nothing to transform: the layer (or selection) is empty.")
            ),
        }
        self.needs_redraw = true;
    }

    /// Enter: applies the transform or crop.
    fn commit_box(&mut self) {
        let Some((kind, b)) = self.boxed.take() else {
            return;
        };
        match kind {
            BoxKind::Transform => {
                let fwd = b.delta();
                self.engine_op(|e| e.transform_active(fwd, Interp::Bicubic, tr("Free Transform")));
            }
            BoxKind::Crop => {
                let (x, y, w, h) = b.rect();
                if w < 1 || h < 1 {
                    tracing::warn!("{}", tr("Crop: the frame is empty"));
                    return;
                }
                self.engine_op(|e| e.crop(x, y, w as u32, h as u32, tr("Crop")));
                self.view_fitted = false;
                self.fit_if_needed();
                tracing::info!("{}", tr!("New image size: {w}×{h}", w = w, h = h));
            }
        }
        self.needs_redraw = true;
    }

    fn cancel_box(&mut self) {
        if self.boxed.take().is_some() {
            if let Some(e) = self.engine() {
                let _ = e.set_transform_preview(None);
            }
            tracing::info!("{}", tr("Cancelled"));
            self.needs_redraw = true;
        }
    }

    /// The transform to show live (free transform, or a move in progress).
    fn preview_transform(&self) -> Option<mosaix_gpu::Warp> {
        match (&self.boxed, &self.drag) {
            (Some((BoxKind::Transform, b)), _) => Some(b.delta()),
            (_, Drag::Move { start, end }) => {
                Some(DAffine2::from_translation(self.move_delta(*start, *end)).into())
            }
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
        if let Some((min, max)) = self.move_bounds {
            let mut off = self.snap_offset(min + d, max + d).round();
            if self.mods.shift {
                // Stay on the constrained axis.
                if d.y == 0.0 {
                    off.y = 0.0;
                } else {
                    off.x = 0.0;
                }
            }
            d += off;
        }
        d
    }

    /// What snaps (document px): guides, the canvas edges and centre, the
    /// grid. `None` with snapping off.
    fn snap_lines(&self) -> Option<([Vec<f64>; 2], Option<f64>)> {
        let v = self.settings.view;
        if !v.snap {
            return None;
        }
        let s = self.session.as_ref()?;
        let (w, h) = s.engine.size();
        let mut xs = vec![0.0, w as f64 * 0.5, w as f64];
        let mut ys = vec![0.0, h as f64 * 0.5, h as f64];
        if v.guides {
            for g in &s.engine.doc().guides {
                if g.vertical {
                    xs.push(g.pos);
                } else {
                    ys.push(g.pos);
                }
            }
        }
        let grid = v.grid.then(|| v.grid_step.max(1.0) / v.grid_div.max(1) as f64);
        Some(([xs, ys], grid))
    }

    /// Distance within which things snap (document px).
    fn snap_reach(&self) -> f64 {
        8.0 * self.scale / self.view.zoom
    }

    /// The nearest line to `v` on one axis within reach.
    fn snap_axis(lines: &[f64], grid: Option<f64>, v: f64, reach: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        let mut consider = |t: f64| {
            if (t - v).abs() <= reach && best.is_none_or(|b| (t - v).abs() < (b - v).abs()) {
                best = Some(t);
            }
        };
        lines.iter().for_each(|t| consider(*t));
        if let Some(g) = grid {
            consider((v / g).round() * g);
        }
        best
    }

    /// Snaps a point (marquee, shape, crop and transform handles).
    fn snap_point(&self, p: DVec2) -> DVec2 {
        let Some(([xs, ys], grid)) = self.snap_lines() else { return p };
        let r = self.snap_reach();
        DVec2::new(
            Self::snap_axis(&xs, grid, p.x, r).unwrap_or(p.x),
            Self::snap_axis(&ys, grid, p.y, r).unwrap_or(p.y),
        )
    }

    /// How far to shift a box `(min, max)` so that an edge or its centre
    /// snaps (zero when nothing is in reach).
    fn snap_offset(&self, min: DVec2, max: DVec2) -> DVec2 {
        let Some(([xs, ys], grid)) = self.snap_lines() else { return DVec2::ZERO };
        let r = self.snap_reach();
        let best = |lines: &[f64], vals: [f64; 3]| {
            vals.iter()
                .filter_map(|v| Self::snap_axis(lines, grid, *v, r).map(|t| t - v))
                .min_by(|a, b| a.abs().total_cmp(&b.abs()))
                .unwrap_or(0.0)
        };
        DVec2::new(
            best(&xs, [min.x, (min.x + max.x) * 0.5, max.x]),
            best(&ys, [min.y, (min.y + max.y) * 0.5, max.y]),
        )
    }

    fn stroke_target(&self) -> StrokeTarget {
        let adjustment = self
            .session
            .as_ref()
            .and_then(|s| s.engine.doc().active_layer())
            .is_some_and(|l| l.adjustment().is_some());
        if self.mask_edit || adjustment {
            StrokeTarget::Mask
        } else {
            StrokeTarget::Pixels
        }
    }

    fn begin_stroke(&mut self, pos: DVec2, pressure: f32, eraser: bool) -> Drag {
        let mut settings = self.brush;
        settings.eraser = eraser;
        if self.tool == Tool::Smudge && !eraser {
            settings.smudge = Some(self.smudge_strength);
        }
        let start = self.view.screen_to_doc(pos);
        let spot = self.tool == Tool::SpotHeal && !eraser;
        if spot {
            // Preview: a translucent grey trail; healing happens on release.
            settings.color = [0.35; 3];
            settings.opacity = 0.5;
        }
        if self.tool == Tool::Remove {
            // Preview: a translucent pink trail; removed on release.
            (settings.color, settings.opacity, settings.flow, settings.eraser) = ([1.0, 0.1, 0.35], 0.45, 1.0, false);
        }
        let clone = (matches!(self.tool, Tool::Clone | Tool::Heal) && !eraser).then(|| {
            let offset = match self.clone_offset {
                Some(o) if self.clone_aligned => o,
                _ => {
                    let d = (self.clone_source.unwrap_or(start) - start).round();
                    [d.x as i32, d.y as i32]
                }
            };
            self.clone_offset = Some(offset);
            mosaix_gpu::CloneSource {
                offset,
                all_layers: self.clone_all,
                heal: self.tool == Tool::Heal,
                spot: false,
            }
        });
        let clone = clone.or_else(|| {
            spot.then_some(mosaix_gpu::CloneSource {
                offset: [0, 0],
                all_layers: self.clone_all,
                heal: true,
                spot: true,
            })
        });
        let target = self.stroke_target();
        if target == StrokeTarget::Mask {
            // Masks are grey: paint the (perceptual) luminance of the brush colour.
            let [r, g, b] = settings.color;
            let lum = linear_to_srgb(0.2126 * r + 0.7152 * g + 0.0722 * b);
            settings.color = [lum; 3];
        }
        let t = self.settings.tone;
        let tone = (!eraser).then(|| {
            use mosaix_gpu::ToneKind as K;
            let (kind, amount) = match self.tool {
                Tool::Dodge => (K::Dodge, t.exposure),
                Tool::Burn => (K::Burn, t.exposure),
                Tool::Sponge if t.sponge_saturate => (K::Saturate, t.sponge_flow),
                Tool::Sponge => (K::Desaturate, t.sponge_flow),
                Tool::Blur => (K::Blur, t.strength),
                Tool::Sharpen => (K::Sharpen, t.strength),
                _ => return None,
            };
            Some(mosaix_gpu::Tone { kind, range: t.range.min(2), amount })
        })
        .flatten();
        if tone.is_some() {
            settings.flow = 1.0;
        }
        if tone.is_some() && target == StrokeTarget::Mask {
            tracing::warn!("{}", tr("Retouching brushes work on pixels, not on masks"));
            return Drag::None;
        }
        let doc_pos = self.view.screen_to_doc(pos);
        let Some(engine) = self.engine() else {
            return Drag::None;
        };
        let params = StrokeParams {
            // The effect itself is the strength; the brush opacity does not apply.
            opacity: if tone.is_some() { 1.0 } else { settings.opacity },
            erase: settings.eraser,
            target,
            smudge: settings.smudge.is_some(),
            clone,
            tone,
        };
        if let Err(e) = engine.begin_stroke(params) {
            tracing::warn!("{}", tr!("Cannot paint: {error}", error = e));
            return Drag::None;
        }
        let mut stroke = Stroke::new(settings);
        let dabs = stroke.add(InputSample {
            pos: doc_pos,
            pressure,
        });
        engine.stroke_dabs(&dabs);
        self.stroke = Some(stroke);
        Drag::Paint
    }

    fn motion(&mut self, pos: DVec2, pressure: f32) {
        self.cursor = Some(pos);
        if matches!(self.drag, Drag::Eyedrop) {
            self.eyedrop(self.view.screen_to_doc(pos));
            return;
        }
        if let Drag::Pen(d) = self.drag {
            let doc = self.view.screen_to_doc(pos);
            self.drag = Drag::Pen(self.pen_drag(d, doc));
            self.needs_redraw = true;
            return;
        }
        let snapped = matches!(self.drag, Drag::Marquee { .. } | Drag::Shape { .. })
            .then(|| self.snap_point(self.view.screen_to_doc(pos)));
        match &mut self.drag {
            // Handled above.
            Drag::None | Drag::Pen(_) | Drag::Eyedrop => {}
            Drag::Paint => {
                let doc_pos = self.view.screen_to_doc(pos);
                if let (Some(stroke), Some(s)) = (&mut self.stroke, &mut self.session) {
                    s.engine.stroke_dabs(&stroke.add(InputSample {
                        pos: doc_pos,
                        pressure,
                    }));
                }
            }
            Drag::TextSelect => {
                let doc = self.view.screen_to_doc(pos);
                self.text_edit_drag(doc);
            }
            Drag::Liquify { last, pressure: p } => {
                let (from, to) = (*last, self.view.screen_to_doc(pos));
                *last = to;
                *p = pressure;
                self.liquify_push(from, to, pressure);
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
                self.limit_zoom_out(anchor);
            }
            Drag::Marquee { end, .. } | Drag::Shape { end, .. } => {
                *end = snapped.unwrap_or(*end);
            }
            Drag::Gradient { end, .. } | Drag::Move { end, .. } => {
                *end = self.view.screen_to_doc(pos);
            }
            Drag::BrushHud {
                start,
                size,
                hardness,
                ..
            } => {
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
                let handle = self.boxed.as_ref().and_then(|(_, b)| b.dragging());
                let doc = match handle {
                    Some(boxedit::Handle::Scale { .. } | boxedit::Handle::Create) => self.snap_point(doc),
                    _ => doc,
                };
                if let Some((_, b)) = &mut self.boxed {
                    b.update(doc, m);
                }
                // A moved box snaps with its edges and centre.
                if handle == Some(boxedit::Handle::Move)
                    && let Some((_, b)) = &self.boxed
                {
                    let c = b.corners();
                    let min = c.iter().fold(DVec2::INFINITY, |a, p| a.min(*p));
                    let max = c.iter().fold(DVec2::NEG_INFINITY, |a, p| a.max(*p));
                    let off = self.snap_offset(min, max);
                    if off != DVec2::ZERO
                        && let Some((_, b)) = &mut self.boxed
                    {
                        b.update(doc + off, m);
                    }
                }
            }
            Drag::Lasso { points, .. } => {
                let doc = self.view.screen_to_doc(pos);
                if points
                    .last()
                    .is_none_or(|p| p.distance(doc) * self.view.zoom >= 2.0)
                {
                    points.push(doc);
                }
            }
        }
        self.needs_redraw = true;
    }

    fn release(&mut self, pos: DVec2, pressure: f32) {
        if matches!(self.drag, Drag::Eyedrop) {
            self.drag = Drag::None;
            self.needs_redraw = true;
            return;
        }
        if matches!(self.drag, Drag::Pen(_)) {
            self.drag = Drag::None;
            self.pen_release();
            self.needs_redraw = true;
            return;
        }
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
        let step = if self.recording.is_some() { recorded_drag(&drag, zoom, move_delta) } else { None };
        let Some(s) = &mut self.session else { return };
        let e = &mut s.engine;
        let result: Result<bool, &str> = match drag {
            Drag::Paint => match self.stroke.take() {
                Some(mut stroke) if tool == Tool::Remove && !stroke.settings.eraser => {
                    e.stroke_dabs(&stroke.finish(InputSample { pos: doc_pos, pressure }));
                    let fp = e.take_stroke_footprint();
                    self.needs_redraw = true;
                    if let Some((layer, dabs)) = fp {
                        self.ai_remove(Some((layer, dabs)));
                    }
                    return;
                }
                Some(mut stroke) => {
                    e.stroke_dabs(&stroke.finish(InputSample {
                        pos: doc_pos,
                        pressure,
                    }));
                    e.end_stroke(if stroke.settings.eraser {
                        tr("Erase")
                    } else {
                        tr("Brush Stroke")
                    });
                    Ok(true)
                }
                None => Ok(false),
            },
            Drag::Shape { start, end } => {
                let (start, end) = shape_rect(start, end, self.mods.shift, self.mods.alt);
                let size = end - start;
                if size.min_element() * zoom < 3.0 {
                    Ok(false)
                } else {
                    let ellipse = matches!(tool, Tool::Shape { ellipse: true });
                    let geometry = if ellipse {
                        let c = (start + end) * 0.5;
                        mosaix_core::vector::Geometry::Ellipse {
                            cx: c.x,
                            cy: c.y,
                            rx: size.x * 0.5,
                            ry: size.y * 0.5,
                        }
                    } else {
                        mosaix_core::vector::Geometry::Rect {
                            x: start.x,
                            y: start.y,
                            w: size.x,
                            h: size.y,
                            radius: 0.0,
                        }
                    };
                    let v = mosaix_core::VectorLayer::shape(mosaix_core::vector::ShapeSpec {
                        geometry,
                        fill: Some([color[0], color[1], color[2], 1.0]),
                        stroke: None,
                        paint: None,
                    });
                    Ok(e.add_vector(
                        v,
                        if ellipse {
                            tr("Ellipse")
                        } else {
                            tr("Rectangle")
                        },
                    )
                    .is_some())
                }
            }
            Drag::Marquee {
                start,
                end,
                op,
                ellipse,
            } => {
                let (min, max) = (start.min(end), start.max(end));
                let size = max - min;
                if size.min_element() * zoom < 2.0 {
                    // A click without dragging deselects (Photoshop).
                    if op == CombineOp::Replace {
                        e.deselect(tr("Deselect")).map(|_| true)
                    } else {
                        Ok(false)
                    }
                } else {
                    let shape = if ellipse {
                        let c = (min + max) * 0.5;
                        Shape::Ellipse {
                            cx: c.x,
                            cy: c.y,
                            rx: size.x * 0.5,
                            ry: size.y * 0.5,
                        }
                    } else {
                        Shape::Rect {
                            x: min.x,
                            y: min.y,
                            w: size.x,
                            h: size.y,
                        }
                    };
                    e.select_shape(&shape, op, tr("Selection")).map(|_| true)
                }
            }
            Drag::Lasso { points, op } => {
                if points.len() >= 3 {
                    let poly = points.iter().map(|p| (p.x, p.y)).collect();
                    e.select_shape(&Shape::Polygon(poly), op, tr("Lasso Selection"))
                        .map(|_| true)
                } else if op == CombineOp::Replace {
                    e.deselect(tr("Deselect")).map(|_| true)
                } else {
                    Ok(false)
                }
            }
            Drag::Gradient { start, end } if start.distance(end) * zoom >= 2.0 => {
                let kind = if tool == (Tool::Gradient { radial: true }) {
                    GradientKind::Radial
                } else {
                    GradientKind::Linear
                };
                let (c0, c1) = match target {
                    StrokeTarget::Pixels => ([color[0], color[1], color[2], 1.0], [0.0; 4]),
                    StrokeTarget::Mask => ([1.0; 4], [0.0, 0.0, 0.0, 1.0]),
                };
                e.gradient(
                    target,
                    kind,
                    (start.x, start.y),
                    (end.x, end.y),
                    c0,
                    c1,
                    opacity,
                    tr("Gradient"),
                )
                .map(|_| true)
            }
            Drag::Move { .. } => {
                let d = move_delta;
                if d == DVec2::ZERO {
                    let _ = e.set_transform_preview(None);
                    Ok(false)
                } else {
                    e.transform_active(DAffine2::from_translation(d), Interp::Nearest, tr("Move"))
                        .map(|_| true)
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
            Ok(true) => {
                self.touch();
                if let Some(step) = step {
                    self.record(step);
                }
            }
            Ok(false) => {}
            Err(err) => tracing::warn!("{}", tr(err)),
        }
        self.needs_redraw = true;
    }

    /// The agent command a shortcut stands for (when recording an action).
    fn recorded_hotkey(&self, sym: Keysym, ctrl: bool, shift: bool, alt: bool) -> Option<serde_json::Value> {
        use serde_json::json;
        let has_sel = self.session.as_ref().is_some_and(|s| s.engine.has_selection());
        Some(match sym {
            Keysym::z | Keysym::Z if ctrl && shift && self.boxed.is_none() => json!({ "cmd": "redo" }),
            Keysym::z if ctrl && self.boxed.is_none() => json!({ "cmd": "undo" }),
            Keysym::y if ctrl => json!({ "cmd": "redo" }),
            Keysym::n | Keysym::N if ctrl && shift => json!({ "cmd": "layer_add" }),
            Keysym::j if ctrl => json!({ "cmd": "layer_duplicate" }),
            Keysym::e | Keysym::E if ctrl && !shift && alt => json!({ "cmd": "layer_merge", "mode": "visible" }),
            Keysym::e if ctrl && !shift && !alt => json!({ "cmd": "layer_merge", "mode": "down" }),
            Keysym::f | Keysym::F if ctrl && shift && alt => json!({ "cmd": "layer_merge", "mode": "flatten" }),
            Keysym::g | Keysym::G if ctrl && shift && !alt => json!({ "cmd": "layer_ungroup" }),
            Keysym::g if ctrl && !alt => json!({ "cmd": "layer_group" }),
            Keysym::Delete | Keysym::BackSpace if !alt && !shift && has_sel => json!({ "cmd": "clear" }),
            Keysym::Delete if !has_sel => json!({ "cmd": "layer_delete" }),
            Keysym::BackSpace if alt => json!({
                "cmd": "fill", "color": actions::hex(self.brush.color), "opacity": self.brush.opacity,
                "target": if self.stroke_target() == StrokeTarget::Mask { "mask" } else { "pixels" },
            }),
            Keysym::a if ctrl => json!({ "cmd": "select_all" }),
            Keysym::d if ctrl => json!({ "cmd": "select_none" }),
            Keysym::i | Keysym::I if ctrl && shift => json!({ "cmd": "select_invert" }),
            Keysym::m | Keysym::M if ctrl && shift && !alt => json!({ "cmd": "layer_mask_add" }),
            Keysym::m | Keysym::M if ctrl && alt && shift => json!({ "cmd": "layer_mask_apply" }),
            Keysym::r if ctrl && alt => json!({ "cmd": "layer_rasterize" }),
            _ => return None,
        })
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
        if self.tablet_ui
            || (self.over_ui(pos) && !self.tablet_contact && matches!(self.drag, Drag::None))
        {
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
    fn edit(
        &mut self,
        label: &str,
        f: impl FnOnce(&mut mosaix_core::Document<mosaix_gpu::Tile>) -> bool,
    ) {
        let Some(engine) = self.engine() else { return };
        if engine.edit(label, f) {
            self.touch();
            self.print_layers();
        }
    }

    fn request_quit(&mut self) {
        if self.save_job.is_some() || self.saving {
            tracing::warn!("{}", tr("Still saving – please wait a moment."));
            return;
        }
        if self.bg_running > 0 && self.quit_armed.is_none_or(|t| t.elapsed() >= QUIT_CONFIRM) {
            tracing::warn!(
                "{}",
                tr(
                    "Something is still running in the background (e.g. an export). Press Ctrl+Q again to abort it."
                )
            );
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
            Pending::Quit => tr("They will be lost when you quit."),
            Pending::Open(_) | Pending::New { .. } => tr("The new image replaces it."),
        };
        self.ui_state.confirm = Some(what.into());
        self.ui_state.hidden = false;
        self.pending = Some(action);
        self.needs_redraw = true;
    }

    fn run_pending(&mut self, action: Pending) {
        match action {
            Pending::Quit => self.exit = true,
            Pending::Open(p) if mosaix_io::raw::is_raw(&p) => {
                self.ui_state.raw = Some(ui::raw_ui::RawDialog::new(p));
                self.ui_state.hidden = false;
                self.needs_redraw = true;
            }
            Pending::Open(p) => self.open_source(Source::File(p), false),
            Pending::New { w, h, white } => self.open_source(Source::Blank(w, h), white),
        }
    }

    fn answer_unsaved(&mut self, a: ui::files::Answer) {
        let Some(action) = self.pending.take() else {
            return;
        };
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
            tracing::warn!("{}", tr("Still loading."));
            return;
        }
        if self.saving || self.save_job.is_some() {
            tracing::warn!("{}", tr("Still saving – please wait a moment."));
            return;
        }
        self.remember_thumbnail();
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

    /// Keeps a small picture of the document for the start screen.
    fn remember_thumbnail(&mut self) {
        if self.loading {
            return;
        }
        let Some(s) = &mut self.session else { return };
        let files = s.files();
        if files.is_empty() {
            return;
        }
        let img = crate::agent::command::read_composite(s, 320);
        for f in files {
            ui::home::store_thumbnail(&f, img.clone());
        }
    }

    fn file_dialog(&mut self, mode: ui::files::FileMode) {
        let s = self.session.as_ref();
        let dir = match mode {
            ui::files::FileMode::Open => self.settings.last_dir.clone(),
            _ => s
                .and_then(|s| s.default_save_path().parent().map(|p| p.to_path_buf()))
                .filter(|p| !p.as_os_str().is_empty()),
        }
        .or_else(|| self.settings.last_dir.clone())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let name = match mode {
            ui::files::FileMode::Open
            | ui::files::FileMode::ImportBrushes
            | ui::files::FileMode::ImportPatterns
            | ui::files::FileMode::PlaceEmbedded
            | ui::files::FileMode::ReplaceContents => None,
            ui::files::FileMode::SaveAs => s.map(|s| s.default_save_path()),
            ui::files::FileMode::ExportAs => s.map(|s| s.default_export_path()),
        }
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
        let req = crate::portal::Request { mode, folder: dir, name, export: self.settings.export };
        if self.picking || self.ui_state.file.is_some() {
            return;
        }
        if std::env::var_os("MOSAIX_BUILTIN_DIALOG").is_some() {
            self.builtin_file_dialog(req);
            return;
        }
        // The system dialog runs on its own thread; the window stays live meanwhile.
        self.picking = true;
        let tx = self.bg_tx.clone();
        let surface = self.window.wl_surface().clone();
        std::thread::Builder::new()
            .name("file chooser".into())
            .spawn(move || {
                let picked = crate::portal::pick(&req, Some(surface));
                let _ = tx.send(BgDone::FilePicked { req, picked });
            })
            .expect("spawn file chooser thread");
    }

    /// The dialog drawn in the window (when there is no desktop portal).
    fn builtin_file_dialog(&mut self, req: crate::portal::Request) {
        let mut d = ui::files::FileDialog::new(req.mode, req.folder, req.name);
        d.export = req.export;
        self.ui_state.file = Some(d);
        self.ui_state.hidden = false;
    }

    fn file_picked(&mut self, req: crate::portal::Request, picked: crate::portal::Picked) {
        self.picking = false;
        match picked {
            crate::portal::Picked::File(path, export) => {
                self.apply_ui(vec![ui::files::action_for(req.mode, path, export)]);
            }
            crate::portal::Picked::Cancelled => {}
            crate::portal::Picked::Unavailable(e) => {
                tracing::info!("{}", tr!("No system file dialog ({e}), using the built-in one", e = e));
                self.builtin_file_dialog(req);
            }
        }
        self.needs_redraw = true;
    }

    fn apply_settings(&mut self) {
        let st = self.settings.clone();
        st.brush.apply(&mut self.brush);
        self.wand_tolerance = st.wand_tolerance.clamp(0.0, 255.0);
        self.wand_contiguous = st.wand_contiguous;
        self.bg_color = st.bg_color;
        self.eyedrop_radius = st.eyedrop_radius;
        self.eyedrop_all = st.eyedrop_all;
        self.smudge_strength = st.smudge_strength.clamp(0.0, 1.0);
        self.tool = match st.tool.as_str() {
            "eraser" => Tool::Eraser,
            "smudge" => Tool::Smudge,
            "clone" => Tool::Clone,
            "heal" => Tool::Heal,
            "spot_heal" => Tool::SpotHeal,
            "text" => Tool::Text,
            "shape" => Tool::Shape { ellipse: false },
            "pen" => Tool::Pen,
            "eyedropper" => Tool::Eyedropper,
            "marquee" => Tool::Marquee { ellipse: false },
            "ellipse" => Tool::Marquee { ellipse: true },
            "lasso" => Tool::Lasso,
            "wand" => Tool::Wand,
            "object_select" => Tool::ObjectSelect,
            "remove" => Tool::Remove,
            "dodge" => Tool::Dodge,
            "burn" => Tool::Burn,
            "sponge" => Tool::Sponge,
            "blur" => Tool::Blur,
            "sharpen" => Tool::Sharpen,
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
        self.settings.bg_color = self.bg_color;
        self.settings.eyedrop_radius = self.eyedrop_radius;
        self.settings.eyedrop_all = self.eyedrop_all;
        self.settings.smudge_strength = self.smudge_strength;
        self.settings.tool = match self.tool {
            Tool::Brush => "brush",
            Tool::Eraser => "eraser",
            Tool::Smudge => "smudge",
            Tool::Clone => "clone",
            Tool::Heal => "heal",
            Tool::SpotHeal => "spot_heal",
            Tool::Text => "text",
            Tool::Shape { .. } => "shape",
            Tool::Pen => "pen",
            Tool::Eyedropper => "eyedropper",
            Tool::Marquee { ellipse: false } => "marquee",
            Tool::Marquee { ellipse: true } => "ellipse",
            Tool::Lasso => "lasso",
            Tool::Wand => "wand",
            Tool::ObjectSelect => "object_select",
            Tool::Remove => "remove",
            Tool::Dodge => "dodge",
            Tool::Burn => "burn",
            Tool::Sponge => "sponge",
            Tool::Blur => "blur",
            Tool::Sharpen => "sharpen",
            Tool::Gradient { radial: false } => "gradient",
            Tool::Gradient { radial: true } => "radial",
            Tool::Move => "move",
            Tool::Crop => "brush",
        }
        .into();
        if let Err(e) = self.settings.save_to(&settings::config_path()) {
            tracing::warn!(
                "{}",
                tr!("Settings not saved: {error}", error = format!("{e:#}"))
            );
        }
    }

    fn save(&mut self) {
        // Never saved as .mosaix yet: ask where.
        if self.session.as_ref().is_some_and(|s| !s.has_save_path()) {
            self.file_dialog(ui::files::FileMode::SaveAs);
            return;
        }
        self.save_to(None);
    }

    fn save_to(&mut self, path: Option<std::path::PathBuf>) {
        if self.save_job.is_some() || self.saving {
            tracing::warn!("{}", tr("Still saving."));
            return;
        }
        let Some(s) = &mut self.session else { return };
        match s.start_save(path.as_deref()) {
            Ok(job) => {
                self.save_job = Some(job);
                self.save_mark = self.edit_count;
                tracing::info!("{}", tr("Saving …"));
            }
            Err(e) => tracing::error!("{}", tr!("Save failed: {error}", error = format!("{e:#}"))),
        }
        self.update_title();
    }

    /// Hands a finished readback to a worker thread (compress + write).
    fn poll_save(&mut self) {
        let ready = self
            .save_job
            .as_ref()
            .is_some_and(|j| j.ready(&self.gpu.device));
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
                self.remember_thumbnail();
                tracing::info!(
                    "{}",
                    tr!(
                        "Saved: {path} ({tiles} tiles written, {kind}, {mb} MB, {ms} ms in the background)",
                        path = r.path.display(),
                        tiles = r.tiles_written,
                        kind = if r.incremental {
                            tr("incremental")
                        } else {
                            tr("full")
                        },
                        mb = format!("{:.1}", r.bytes as f64 / 1e6),
                        ms = format!("{:.0}", r.ms)
                    )
                );
            }
            Err(e) => {
                tracing::error!("{}", tr!("Save failed: {error}", error = format!("{e:#}")));
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

    fn export_to(&mut self, path: std::path::PathBuf, opts: mosaix_io::export::ExportOptions) {
        if self.settings.export != opts {
            self.settings.export = opts;
            self.store_settings();
        }
        let Some(s) = &mut self.session else { return };
        if self.export_job.is_some() {
            tracing::warn!("{}", tr("Export still running."));
            return;
        }
        tracing::info!(
            "{}",
            tr!(
                "Exporting {path} ({space}) in the background …",
                path = path.display(),
                space = opts.space.name()
            )
        );
        self.export_job = Some(s.start_export(&path, &opts));
        self.bg_running += 1;
    }

    fn export(&mut self) {
        let Some(s) = &mut self.session else { return };
        if self.export_job.is_some() {
            tracing::warn!("{}", tr("Export still running."));
            return;
        }
        let path = s.default_export_path();
        let opts = self.settings.export;
        tracing::info!(
            "{}",
            tr!(
                "Exporting {path} ({space}) in the background …",
                path = path.display(),
                space = opts.space.name()
            )
        );
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
                self.spawn_bg("autosave", move || BgDone::Autosaved {
                    path,
                    result: work().map(|_| ()),
                });
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
            tracing::warn!(
                "{}",
                tr!("Autosave: {path}: {error}", path = dir.display(), error = e)
            );
            self.autosave_at = Instant::now();
            return;
        }
        let stem = std::path::Path::new(&s.name)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let path = dir.join(format!(
            "{}-{}.mosaix",
            if stem.is_empty() {
                tr("Untitled")
            } else {
                &stem
            },
            std::process::id()
        ));
        match s.start_save_copy(&path) {
            Ok(job) => self.autosave_job = Some((path, job)),
            Err(e) => tracing::warn!("{}", tr!("Autosave: {error}", error = format!("{e:#}"))),
        }
        self.autosave_mark = self.edit_count;
        self.autosave_at = Instant::now();
    }

    /// Removes the recovery copy (after a real save or a clean exit).
    fn drop_autosave(&mut self) {
        let paths = [
            self.autosave_path.take(),
            self.autosave_job.take().map(|(p, _)| p),
        ];
        for p in paths.into_iter().flatten() {
            let _ = std::fs::remove_file(p);
        }
    }

    /// Hands a finished export readback to a worker thread.
    fn poll_export(&mut self) {
        if self
            .export_job
            .as_ref()
            .is_some_and(|j| j.ready(&self.gpu.device))
        {
            let job = self.export_job.take().unwrap();
            let path = job.path.clone();
            let work = job.into_writer();
            self.bg_running -= 1; // spawn_bg counts it again
            self.spawn_bg("export", move || BgDone::Exported {
                path,
                result: work(),
            });
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
    /// Select › Modify in the background (large selections take a moment).
    fn modify_selection(&mut self, op: &'static str, amount: f64) {
        use mosaix_core::mask::Modify;
        let Some((w, h)) = self.session.as_ref().map(|s| s.engine.size()) else { return };
        if self.session.as_ref().is_some_and(|s| !s.engine.has_selection()) {
            tracing::warn!("{}", tr("no selection"));
            return;
        }
        let (label, m) = match op {
            "expand" => (tr("Expand Selection"), Some(Modify::Expand(amount))),
            "contract" => (tr("Contract Selection"), Some(Modify::Contract(amount))),
            "border" => (tr("Border Selection"), Some(Modify::Border(amount))),
            "smooth" => (tr("Smooth Selection"), Some(Modify::Smooth(amount))),
            _ => (tr("Feather Selection"), None),
        };
        self.selection_job(label, move |sel| {
            let sel = sel.context(tr("no selection"))?;
            Ok(match m {
                Some(m) => mosaix_core::mask::modify(&sel, m, w, h),
                None => mosaix_core::mask::feather(&sel, amount, w, h),
            })
        });
    }

    fn selection_job(
        &mut self,
        label: &'static str,
        f: impl FnOnce(Option<CpuMask>) -> anyhow::Result<CpuMask> + Send + 'static,
    ) {
        let Some(e) = self.engine() else { return };
        let base = e.selection_id();
        let current = e.selection_cpu();
        let started = Instant::now();
        self.spawn_bg("selection", move || BgDone::Selection {
            label,
            base,
            mask: f(current),
            started,
        });
    }

    fn on_background_done(&mut self, done: BgDone) {
        // The file chooser is not counted as background work.
        if !matches!(done, BgDone::FilePicked { .. }) {
            self.bg_running = self.bg_running.saturating_sub(1);
        }
        match done {
            BgDone::FilePicked { req, picked } => self.file_picked(req, picked),
            BgDone::Autosaved { path, result } => match result {
                Ok(()) => {
                    tracing::info!(
                        "{}",
                        tr!("Recovery copy saved: {path}", path = path.display())
                    );
                    self.autosave_path = Some(path);
                }
                Err(e) => tracing::warn!(
                    "{}",
                    tr!("Autosave failed: {error}", error = format!("{e:#}"))
                ),
            },
            BgDone::Exported { path, result } => match result {
                Ok(ms) => tracing::info!(
                    "{}",
                    tr!(
                        "Exported: {path} ({ms} ms)",
                        path = path.display(),
                        ms = format!("{ms:.0}")
                    )
                ),
                Err(e) => tracing::error!(
                    "{}",
                    tr!("Export failed: {error}", error = format!("{e:#}"))
                ),
            },
            BgDone::Pasted(result) => self.on_pasted(result),
            BgDone::Filled { result, started, label } => match result {
                Ok(r) => {
                    let ms = r.ms;
                    if self.engine().is_some_and(|e| e.apply_fill(r, label)) {
                        self.touch();
                        tracing::info!(
                            "{}",
                            tr!(
                                "{label} done: {total} ms (computation {ms} ms)",
                                label = label,
                                total = format!("{:.0}", started.elapsed().as_secs_f64() * 1e3),
                                ms = format!("{ms:.0}")
                            )
                        );
                    }
                }
                Err(e) => tracing::warn!("{}", tr!("{label}: {error}", label = label, error = e)),
            },
            BgDone::LayerMask { layer, mask, started } => match mask {
                Ok(m) if !m.is_empty() => {
                    self.engine_op(|e| e.set_layer_mask(layer, Some(m), tr("Remove Background")));
                    self.touch();
                    tracing::info!("{}", tr!("Background removed ({ms} ms)", ms = started.elapsed().as_millis()));
                }
                Ok(_) => tracing::warn!("{}", tr("No subject found")),
                Err(e) => tracing::warn!("{}", tr!("Remove Background: {error}", error = format!("{e:#}"))),
            },
            BgDone::Upscaled { result, started } => match result {
                Ok(r) => match self.engine().map(|e| e.apply_upscale(r, tr("Upscale (AI)"))) {
                    Some(Ok(())) => {
                        self.touch();
                        self.view_fitted = false;
                        tracing::info!("{}", tr!("Upscaled in {s} s", s = format!("{:.1}", started.elapsed().as_secs_f64())));
                    }
                    Some(Err(e)) => tracing::warn!("{}", tr!("{label}: {error}", label = tr("Upscale (AI)"), error = tr(e))),
                    None => {}
                },
                Err(e) => tracing::warn!("{}", tr!("{label}: {error}", label = tr("Upscale (AI)"), error = format!("{e:#}"))),
            },
            BgDone::ModelDownloaded { name, result } => match result {
                Ok(()) => tracing::info!("{}", tr!("{name} is ready – try again", name = name)),
                Err(e) => tracing::error!("{}", tr!("Download of {name} failed: {error}", name = name, error = format!("{e:#}"))),
            },
            BgDone::Selection {
                label,
                base,
                mask,
                started,
            } => {
                let Some(e) = self.engine() else { return };
                if e.selection_id() != base {
                    tracing::warn!(
                        "{}",
                        tr!(
                            "{label}: discarded, the selection has changed in the meantime",
                            label = label
                        )
                    );
                } else {
                    match mask {
                        Ok(m) => {
                            self.engine_op(|e| e.set_selection(Some(m), label));
                            tracing::info!(
                                "{label}: {:.0} ms",
                                started.elapsed().as_secs_f64() * 1e3
                            );
                        }
                        Err(err) => tracing::warn!("{label}: {err:#}"),
                    }
                }
            }
        }
        self.update_title();
        self.needs_redraw = true;
    }

    /// Loads the installed font families once (for the text properties).
    fn ensure_fonts(&mut self) {
        if self.fonts.is_empty() {
            let t0 = Instant::now();
            self.fonts = std::sync::Arc::new(mosaix_vector::font_groups());
            tracing::debug!(
                "{}",
                tr!(
                    "{n} font families in {ms} ms",
                    n = self.fonts.len(),
                    ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3)
                )
            );
        }
    }

    /// Text tool: a click on a text layer selects it for editing, anywhere
    /// else creates a new text there.
    fn text_click(&mut self, at: DVec2) {
        self.ensure_fonts();
        let color = self.brush.color;
        let size = (24.0 / self.view.zoom as f32).clamp(8.0, 2000.0).round();
        let Some(e) = self.engine() else { return };
        let doc = e.doc();
        let hit = doc.panel_rows().into_iter().map(|(id, _)| id).find(|&id| {
            doc.find(id).is_some_and(|l| {
                l.visible
                    && matches!(
                        l.vector().map(|v| &v.kind),
                        Some(mosaix_core::vector::VectorKind::Text(_))
                    )
                    && mosaix_vector::bounds(l.vector().unwrap()).is_some_and(|b| {
                        at.x >= b[0] && at.x <= b[2] && at.y >= b[1] && at.y <= b[3]
                    })
            })
        });
        match hit {
            Some(id) => self.begin_text_edit(id, Some(at)),
            None => {
                // A new, empty text with the caret at the click (removed again if left empty).
                let spec = mosaix_core::vector::TextSpec::new(
                    "",
                    at.x,
                    at.y - size as f64 * 0.6,
                    size,
                    [color[0], color[1], color[2], 1.0],
                );
                if let Some(id) = e.add_vector(mosaix_core::VectorLayer::text(spec), tr("Text")) {
                    self.begin_text_edit(id, None);
                }
            }
        }
        self.ui_state.hidden = false;
        self.needs_redraw = true;
    }

    /// Layer › Rasterize: a text, shape or RAW layer becomes pixels.
    fn rasterize_active(&mut self) {
        self.engine_op(|e| {
            let id = e.doc().active;
            e.rasterize(id, tr("Rasterize"))
        });
    }

    /// Content-aware fill of the selection: area read here, computed on a worker.
    fn content_fill(&mut self, all_layers: bool) {
        let Some(e) = self.engine() else { return };
        let started = Instant::now();
        match e.fill_job(all_layers) {
            Ok(job) => {
                tracing::info!(
                    "{}",
                    if all_layers {
                        tr("Content-Aware Fill (all layers) …")
                    } else {
                        tr("Content-Aware Fill …")
                    }
                );
                self.spawn_bg("content fill", move || BgDone::Filled {
                    result: job.run().map_err(|e| tr(e).to_string()),
                    started,
                    label: tr("Content-Aware Fill"),
                });
            }
            Err(err) => tracing::warn!("{}", tr!("Content-Aware Fill: {error}", error = tr(err))),
        }
    }

    /// Magic wand: similarity map on the GPU here, flood fill and combining on a worker.
    /// Asks for the download if a model is missing; true if it can run.
    fn ai_ready(&mut self, m: &'static mosaix_ai::Model) -> bool {
        if let Err(e) = mosaix_ai::available() {
            tracing::warn!("{e:#}");
            return false;
        }
        if mosaix_ai::models::path(m).is_none() {
            self.ui_state.ai_download = Some(m.id);
            self.ui_state.hidden = false;
            self.needs_redraw = true;
            return false;
        }
        true
    }

    /// Object Selection (click at `at`) or Select › Subject (`None`).
    fn ai_select(&mut self, at: Option<DVec2>, op: CombineOp) {
        let model = if at.is_some() { &mosaix_ai::SAM } else { &mosaix_ai::SUBJECT };
        if !self.ai_ready(model) {
            return;
        }
        let Some(s) = self.session.as_mut() else { return };
        if let Some(p) = at {
            let (w, h) = s.engine.size();
            if p.x < 0.0 || p.y < 0.0 || p.x >= w as f64 || p.y >= h as f64 {
                return;
            }
        }
        let snap = crate::ai::snapshot(s);
        tracing::info!("{}", if at.is_some() { tr("Finding the object …") } else { tr("Finding the subject …") });
        let label = if at.is_some() { tr("Select Object") } else { tr("Select Subject") };
        self.selection_job(label, move |old| {
            let region = match at {
                Some(p) => crate::ai::object(&snap, &[(p.x, p.y, true)])?,
                None => crate::ai::subject(&snap)?,
            };
            Ok(match old {
                Some(old) if op != CombineOp::Replace => mosaix_core::mask::combine(&old, &region, op),
                None if matches!(op, CombineOp::Subtract | CombineOp::Intersect) => CpuMask::filled(0),
                _ => region,
            })
        });
    }

    /// Layer › Remove Background: a mask from the subject on the active layer.
    fn remove_background(&mut self) {
        if !self.ai_ready(&mosaix_ai::SUBJECT) {
            return;
        }
        let Some(s) = self.session.as_mut() else { return };
        let layer = s.engine.doc().active;
        let snap = crate::ai::snapshot(s);
        tracing::info!("{}", tr("Finding the subject …"));
        let started = Instant::now();
        self.spawn_bg("remove background", move || BgDone::LayerMask { layer, mask: crate::ai::subject(&snap), started });
    }

    /// Remove (local AI): fills a painted area (`Some((layer, dabs))`) or the
    /// selection (`None`) with generated surroundings, on a worker.
    fn ai_remove(&mut self, painted: Option<(mosaix_core::LayerId, Vec<[f32; 3]>)>) {
        if !self.ai_ready(&mosaix_ai::INPAINT) {
            return;
        }
        let all = self.clone_all;
        let Some(e) = self.engine() else { return };
        let job = match painted {
            Some((layer, dabs)) => e.fill_job_dabs(layer, &dabs, all, crate::ai::INPAINT_CONTEXT),
            None => e.fill_job_sized(all, crate::ai::INPAINT_CONTEXT, crate::ai::REMOVE_GROW),
        };
        match job {
            Ok(job) => {
                tracing::info!("{}", tr("Removing …"));
                let started = Instant::now();
                self.spawn_bg("remove", move || BgDone::Filled {
                    result: crate::ai::inpaint(job).map_err(|e| format!("{e:#}")),
                    started,
                    label: tr("Remove"),
                });
            }
            Err(err) => tracing::warn!("{}", tr!("{label}: {error}", label = tr("Remove"), error = tr(err))),
        }
    }

    /// Edit › Generative Fill: the selection painted from a prompt (background).
    fn generative_fill(&mut self, prompt: String) {
        let provider = match self.settings.generative.provider() {
            Ok(p) => p,
            Err(e) => return tracing::warn!("{e}"),
        };
        let Some(e) = self.engine() else { return };
        if !e.has_selection() {
            return tracing::warn!("{}", tr("Generative fill needs a selection"));
        }
        match e.fill_job_sized(true, crate::ai::GENERATIVE_CONTEXT, crate::ai::REMOVE_GROW) {
            Ok(job) => {
                tracing::info!("{}", tr!("Generative fill: “{prompt}” …", prompt = prompt.clone()));
                let started = Instant::now();
                self.spawn_bg("generative fill", move || BgDone::Filled {
                    result: crate::ai::generative_fill(job, &provider, &prompt).map_err(|e| format!("{e:#}")),
                    started,
                    label: tr("Generative Fill"),
                });
            }
            Err(err) => tracing::warn!("{}", tr!("{label}: {error}", label = tr("Generative Fill"), error = tr(err))),
        }
    }

    /// Image › Upscale (AI): every pixel layer enlarged on a worker, then the
    /// document scaled (one undo step).
    fn ai_upscale(&mut self, factor: u32) {
        if !self.ai_ready(&mosaix_ai::UPSCALE) {
            return;
        }
        let Some(e) = self.engine() else { return };
        match e.upscale_job(factor) {
            Ok(job) => {
                let secs = crate::ai::upscale_estimate(job.pixels());
                tracing::info!("{}", tr!("Upscaling {k}× – about {s} s …", k = factor, s = format!("{:.0}", secs.max(1.0))));
                let started = Instant::now();
                self.spawn_bg("upscale", move || BgDone::Upscaled { result: crate::ai::upscale(job), started });
            }
            Err(err) => tracing::warn!("{}", tr!("{label}: {error}", label = tr("Upscale (AI)"), error = tr(err))),
        }
    }

    /// Downloads a model in the background (the user agreed).
    fn download_model(&mut self, id: &str) {
        let Some(m) = mosaix_ai::models::by_id(id) else { return };
        tracing::info!("{}", tr!("Downloading {name} ({mb} MB) …", name = m.name, mb = m.size() / 1_000_000));
        self.spawn_bg("model download", move || {
            let mut last = 0;
            let result = mosaix_ai::models::download(m, &mut |done, total| {
                let pct = done * 10 / total.max(1);
                if pct != last {
                    last = pct;
                    tracing::info!("{}: {} %", m.name, pct * 10);
                }
            })
            .map(|_| ());
            BgDone::ModelDownloaded { name: m.name, result }
        });
    }

    fn magic_wand(&mut self, x: u32, y: u32, op: CombineOp) {
        let (tol, contiguous) = (self.wand_tolerance, self.wand_contiguous);
        let Some(e) = self.engine() else { return };
        let (w, h) = e.size();
        let map = e
            .sample_straight(x, y, SampleFrom::Image)
            .and_then(|reference| {
                e.similarity_map(reference, tol, (tol * 0.5).max(12.0), SampleFrom::Image)
            });
        let map = match map {
            Ok(m) => m,
            Err(err) => return tracing::warn!("{}", tr!("Magic Wand: {error}", error = err)),
        };
        self.selection_job(tr("Magic Wand"), move |old| {
            let region = if contiguous {
                mosaix_core::mask::flood(&map, x, y, w, h)
            } else {
                map
            };
            Ok(match old {
                Some(old) => mosaix_core::mask::combine(&old, &region, op),
                None if op == CombineOp::Subtract => CpuMask::filled(0),
                None => region,
            })
        });
    }

    /// A held key repeating: only typing repeats, never tools or commands.
    fn repeated_key(&mut self, ev: KeyEvent) {
        if self.ui_state.captures_keyboard() || self.gui.wants_keyboard() || self.text_edit.is_some() {
            self.key(&ev, true);
        }
    }

    fn key(&mut self, ev: &KeyEvent, pressed: bool) {
        let sym = ev.keysym;
        tracing::debug!(?sym, raw = ev.raw_code, pressed, "key");
        // Recording a new shortcut (keyboard-shortcuts dialog).
        if let Some(id) = self.ui_state.shortcuts.as_mut().and_then(|d| d.capturing.clone()) {
            let modifier = matches!(
                sym,
                Keysym::Shift_L | Keysym::Shift_R | Keysym::Control_L | Keysym::Control_R | Keysym::Alt_L | Keysym::Alt_R | Keysym::Super_L | Keysym::Super_R | Keysym::ISO_Level3_Shift
            );
            if pressed && !modifier {
                let (ctrl, shift, alt) = (self.mods.ctrl, self.mods.shift, self.mods.alt);
                let plain = !ctrl && !shift && !alt;
                match sym {
                    Keysym::Escape if plain => {}
                    Keysym::BackSpace | Keysym::Delete if plain => {
                        ui::palette::rebind(&mut self.settings.keymap, &id, None, false);
                        self.store_settings();
                    }
                    _ => {
                        let combo = ui::palette::Combo::new(sym, ctrl, shift, alt);
                        ui::palette::rebind(&mut self.settings.keymap, &id, Some(combo), false);
                        self.store_settings();
                    }
                }
                if let Some(d) = &mut self.ui_state.shortcuts {
                    d.capturing = None;
                }
                self.needs_redraw = true;
            }
            return;
        }
        // Text fields and the command palette take the keyboard.
        if self.ui_state.captures_keyboard() || self.gui.wants_keyboard() {
            // Clipboard in text fields.
            if pressed && self.mods.ctrl {
                match sym {
                    Keysym::c | Keysym::C => self.gui.copy_event(false),
                    Keysym::x | Keysym::X => self.gui.copy_event(true),
                    Keysym::v | Keysym::V => self.paste(clipboard::PasteTarget::Ui),
                    _ => {}
                }
            }
            if let Some(k) = ui::egui_key(sym) {
                self.gui.key(k, pressed);
            }
            if pressed && let Some(t) = &ev.utf8 {
                self.gui.text(t);
            }
            self.needs_redraw = true;
            return;
        }
        // Pen: finishing, removing nodes.
        if self.tool == Tool::Pen && pressed && self.pen_key(ev) {
            self.needs_redraw = true;
            return;
        }
        // Typing into a text on the canvas.
        if self.text_edit.is_some() {
            if pressed && self.text_edit_key(ev) {
                self.needs_redraw = true;
                return;
            }
            if matches!(sym, Keysym::space | Keysym::r) && !self.mods.ctrl {
                return;
            }
        }
        match sym {
            Keysym::space => self.space = pressed,
            Keysym::r => self.r_key = pressed,
            _ => {}
        }
        if !pressed || self.stroke.is_some() {
            return;
        }
        // The user's shortcuts come first; a default given away does nothing.
        let combo = ui::palette::Combo::new(sym, self.mods.ctrl, self.mods.shift, self.mods.alt);
        match ui::palette::lookup(combo, &self.settings.keymap) {
            ui::palette::Lookup::Command(c) => self.run_cmd(c),
            ui::palette::Lookup::Freed => {}
            ui::palette::Lookup::Default => self.hotkey(ev),
        }
        self.needs_redraw = true;
    }

    /// Runs a menu / palette command by replaying its shortcut.
    fn run_cmd(&mut self, c: ui::palette::Cmd) {
        let saved = self.mods;
        self.mods.ctrl = c.ctrl;
        self.mods.shift = c.shift;
        self.mods.alt = c.alt;
        self.hotkey(&KeyEvent {
            time: 0,
            raw_code: 0,
            keysym: c.key,
            utf8: None,
        });
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
            let pct = if ev.raw_code == KEY_0 {
                1.0
            } else {
                (ev.raw_code - KEY_1 + 1) as f32 / 10.0
            };
            if shift {
                self.record(serde_json::json!({ "cmd": "layer_set", "opacity": pct }));
                self.edit(tr("Opacity"), |d| {
                    let id = d.active;
                    let l = d.find_mut(id).unwrap();
                    let changed = l.opacity != pct;
                    l.opacity = pct;
                    changed
                });
            } else if self.tool == Tool::Smudge {
                self.smudge_strength = pct;
                tracing::info!(
                    "{}",
                    tr!(
                        "Smudge strength {pct} %",
                        pct = format!("{:.0}", pct * 100.0)
                    )
                );
            } else {
                self.brush.opacity = pct;
                tracing::info!(
                    "{}",
                    tr!("Brush opacity {pct} %", pct = format!("{:.0}", pct * 100.0))
                );
            }
            self.needs_redraw = true;
            return;
        }

        if self.recording.is_some()
            && let Some(step) = self.recorded_hotkey(sym, ctrl, shift, alt)
        {
            self.record(step);
        }
        match sym {
            Keysym::q | Keysym::Q if ctrl => self.request_quit(),
            Keysym::p | Keysym::P if ctrl && shift => self.ui_state.palette = Some((String::new(), 0)),
            Keysym::k if ctrl => self.ui_state.palette = Some((String::new(), 0)),
            Keysym::Tab => self.ui_state.hidden = !self.ui_state.hidden,
            Keysym::Escape if self.liquifying() => self.cancel_liquify(),
            Keysym::Return | Keysym::KP_Enter if self.liquifying() => self.commit_liquify(),
            Keysym::z | Keysym::Z if ctrl && !shift && self.liquifying() => {
                if !self.engine().is_some_and(|e| e.liquify_undo()) {
                    tracing::info!("{}", tr("Liquify: nothing left to undo"));
                }
            }
            Keysym::x | Keysym::X if ctrl && shift => self.begin_liquify_mode(),
            Keysym::c | Keysym::C if ctrl && shift && !self.liquifying() => self.copy(false, true),
            Keysym::c if ctrl && !self.liquifying() => self.copy(false, false),
            Keysym::x if ctrl && !self.liquifying() => self.copy(true, false),
            Keysym::v if ctrl && !alt && !self.liquifying() => self.paste(clipboard::PasteTarget::Document),
            _ if self.liquifying() && !ctrl => {
                let tool = match sym {
                    Keysym::w | Keysym::W => Some(LiquifyTool::Push),
                    Keysym::r | Keysym::R => Some(LiquifyTool::Reconstruct),
                    Keysym::c | Keysym::C => Some(LiquifyTool::TwirlCw),
                    Keysym::s | Keysym::S => Some(LiquifyTool::Pucker),
                    Keysym::b | Keysym::B => Some(LiquifyTool::Bloat),
                    Keysym::bracketleft => {
                        self.liq_size = (self.liq_size / 1.2).max(4.0);
                        None
                    }
                    Keysym::bracketright => {
                        self.liq_size = (self.liq_size * 1.2).min(5000.0);
                        None
                    }
                    _ => None,
                };
                if let Some(t) = tool {
                    self.liq_tool = t;
                    tracing::info!("{}", tr!("Liquify: {tool}", tool = t.name()));
                }
            }
            // Other edits wait until liquify is applied or cancelled.
            _ if self.liquifying() => tracing::info!("{}", tr("Apply (Enter) or cancel (Esc) Liquify first")),
            Keysym::Escape => self.cancel_box(),
            Keysym::Return | Keysym::KP_Enter => self.commit_box(),
            Keysym::t if ctrl => self.begin_transform(),
            // While a transform/crop box is open, undo just cancels it.
            Keysym::z | Keysym::Z if ctrl && self.boxed.is_some() => self.cancel_box(),
            Keysym::z | Keysym::Z if ctrl && shift => self.redo(),
            Keysym::z if ctrl => self.undo(),
            Keysym::y if ctrl => self.redo(),
            Keysym::s | Keysym::S if ctrl && alt && !shift => self.ai_select(None, CombineOp::Replace),
            Keysym::b if ctrl && alt && !shift => self.remove_background(),
            Keysym::W if shift && !ctrl && !alt => {
                self.set_tool(Tool::ObjectSelect);
                tracing::info!("{}", tr("Object Selection: click an object (Shift adds, Alt subtracts) – local AI"));
            }
            Keysym::s | Keysym::S if ctrl && shift => self.file_dialog(ui::files::FileMode::SaveAs),
            Keysym::s | Keysym::S if ctrl => self.save(),
            Keysym::e | Keysym::E if ctrl && shift && alt => self.file_dialog(ui::files::FileMode::ExportAs),
            Keysym::e | Keysym::E if ctrl && shift => self.export(),
            Keysym::o if ctrl => self.file_dialog(ui::files::FileMode::Open),
            Keysym::n if ctrl && !shift => {
                self.ui_state.new_doc = Some(Default::default());
                self.ui_state.hidden = false;
            }
            Keysym::n | Keysym::N if ctrl && shift => self.edit(tr("New Layer"), |d| {
                d.add_pixel_layer();
                true
            }),
            Keysym::j if ctrl => self.edit(tr("Duplicate Layer"), |d| d.duplicate_active().is_some()),
            Keysym::e | Keysym::E if ctrl && !shift && alt => self.run_agent_cmd(serde_json::json!({ "cmd": "layer_merge", "mode": "visible" })),
            Keysym::e if ctrl && !shift && !alt => self.run_agent_cmd(serde_json::json!({ "cmd": "layer_merge", "mode": "down" })),
            Keysym::f | Keysym::F if ctrl && shift && alt => self.run_agent_cmd(serde_json::json!({ "cmd": "layer_merge", "mode": "flatten" })),
            Keysym::g | Keysym::G if ctrl && shift => self.edit(tr("Ungroup Layers"), |d| d.ungroup_active()),
            // Ctrl+Alt+G: clipping mask on/off (clips to the layer below).
            Keysym::b | Keysym::B if ctrl && alt && shift => self.file_dialog(ui::files::FileMode::ImportBrushes),
            Keysym::y | Keysym::Y if ctrl && alt && shift => self.file_dialog(ui::files::FileMode::ImportPatterns),
            Keysym::d | Keysym::D if ctrl && alt && shift => self.define_pattern(),
            Keysym::k | Keysym::K if ctrl && alt && shift => {
                self.ui_state.shortcuts = Some(Default::default());
            }
            Keysym::g if ctrl && alt => self.edit(tr("Clipping Mask"), |d| d.toggle_clip_active()),
            Keysym::g if ctrl => self.edit(tr("Group Layers"), |d| d.group_active().is_some()),
            // Shift+Rücktaste: content-aware fill (Strg: learn from all layers).
            Keysym::BackSpace if shift && !alt => self.content_fill(ctrl),
            // Shift+Alt+Rücktaste: remove the selected (local AI).
            Keysym::BackSpace if shift && alt && !ctrl => self.ai_remove(None),
            // Photoshop: Entf and Rücktaste both clear the selection.
            Keysym::Delete | Keysym::BackSpace
                if !alt && self.session.as_ref().is_some_and(|s| s.engine.has_selection()) =>
            {
                self.engine_op(|e| {
                    let n = e.clear_selected(tr("Delete"))?;
                    let name = e.doc().active_layer().map(|l| l.name.clone()).unwrap_or_default();
                    if n == 0 {
                        tracing::warn!(
                            "{}",
                            tr!("Nothing deleted: the selection contains no pixels on layer “{name}”. Is another layer active?", name = name)
                        );
                    } else {
                        tracing::info!("{}", tr!("Selection deleted on layer “{name}” ({n} tiles)", name = name, n = n));
                    }
                    Ok(())
                })
            }
            Keysym::Delete => self.edit(tr("Delete Layer"), |d| d.remove_active()),
            Keysym::BackSpace if alt => {
                let [r, g, b] = self.brush.color;
                let (target, opacity) = (self.stroke_target(), self.brush.opacity);
                let color = if target == StrokeTarget::Mask {
                    let l = linear_to_srgb(0.2126 * r + 0.7152 * g + 0.0722 * b);
                    [l, l, l, 1.0]
                } else {
                    [r, g, b, 1.0]
                };
                self.engine_op(|e| e.fill(target, color, opacity, tr("Fill")).map(|_| ()))
            }
            Keysym::a if ctrl => self.engine_op(|e| e.select_all(tr("Select All"))),
            Keysym::d if ctrl => self.engine_op(|e| e.deselect(tr("Deselect"))),
            Keysym::i | Keysym::I if ctrl && shift => self.engine_op(|e| e.invert_selection(tr("Inverse"))),
            Keysym::F6 if shift => {
                if self.session.as_ref().is_some_and(|s| !s.engine.has_selection()) {
                    tracing::warn!("{}", tr("no selection"));
                } else {
                    self.ui_state.open_modify("feather");
                    self.ui_state.hidden = false;
                }
            }
            Keysym::m | Keysym::M if ctrl && alt && shift => {
                self.mask_edit = false;
                self.engine_op(|e| e.apply_layer_mask(tr("Apply Layer Mask")))
            }
            Keysym::m | Keysym::M if ctrl && alt => self.edit(tr("Toggle Layer Mask"), |d| {
                let id = d.active;
                let l = d.find_mut(id).unwrap();
                l.mask_enabled = !l.mask_enabled;
                l.mask.is_some()
            }),
            Keysym::m | Keysym::M if ctrl && shift => self.engine_op(|e| e.add_layer_mask(tr("Layer Mask"))),
            Keysym::backslash => {
                let has_mask = self.session.as_ref().and_then(|s| s.engine.doc().active_layer()).is_some_and(|l| l.mask.is_some());
                if self.mask_edit || has_mask {
                    self.mask_edit = !self.mask_edit;
                    tracing::info!("{}", tr!("Target: {target}", target = if self.mask_edit { tr("Layer Mask") } else { tr("Pixels") }));
                } else {
                    tracing::warn!("{}", tr("The layer has no mask (Ctrl+Shift+M adds one)."));
                }
            }
            // Rotate / mirror: Strg+Alt (layer) or Strg+Alt+Umschalt (image) with arrows, H, V.
            Keysym::Left | Keysym::Right | Keysym::Down | Keysym::h | Keysym::H | Keysym::v | Keysym::V if ctrl && alt => {
                let (turn, fx, fy) = match sym {
                    Keysym::Left => (-1, false, false),
                    Keysym::Right => (1, false, false),
                    Keysym::Down => (2, false, false),
                    Keysym::h | Keysym::H => (0, true, false),
                    _ => (0, false, true),
                };
                let label = match (shift, turn, fx) {
                    (true, 0, true) => tr("Flip Image Horizontal"),
                    (true, 0, false) => tr("Flip Image Vertical"),
                    (true, 2, _) => tr("Rotate Image 180°"),
                    (true, 1, _) => tr("Rotate Image 90° Clockwise"),
                    (true, _, _) => tr("Rotate Image 90° Counterclockwise"),
                    (false, 0, true) => tr("Flip Layer Horizontal"),
                    (false, 0, false) => tr("Flip Layer Vertical"),
                    (false, 2, _) => tr("Rotate Layer 180°"),
                    (false, 1, _) => tr("Rotate Layer 90° Clockwise"),
                    (false, _, _) => tr("Rotate Layer 90° Counterclockwise"),
                };
                if shift {
                    self.engine_op(|e| e.rotate_image(turn, fx, fy, label));
                    self.view_fitted = false;
                    self.fit_if_needed();
                } else {
                    self.engine_op(|e| e.rotate_layer(turn as f64 * 90.0, fx, fy, label));
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
                self.engine_op(|e| e.transform_active(fwd, Interp::Nearest, tr("Move")))
            }
            Keysym::comma if ctrl => self.edit(tr("Visibility"), |d| {
                let id = d.active;
                let l = d.find_mut(id).unwrap();
                l.visible = !l.visible;
                true
            }),
            Keysym::bracketleft if alt => self.select_layer(1),
            Keysym::bracketright if alt => self.select_layer(-1),
            Keysym::bracketleft if ctrl => self.edit(tr("Move Layer Down"), |d| d.move_active(-1)),
            Keysym::bracketright if ctrl => self.edit(tr("Move Layer Up"), |d| d.move_active(1)),
            Keysym::plus | Keysym::asterisk | Keysym::KP_Add if shift || sym == Keysym::KP_Add => self.cycle_blend(1),
            Keysym::minus | Keysym::underscore | Keysym::KP_Subtract if shift || sym == Keysym::KP_Subtract => {
                self.cycle_blend(-1)
            }
            Keysym::F7 => self.edit(tr("Blend Color Space"), |d| {
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
                tracing::info!("{}", tr!("Tolerance {tol}", tol = format!("{:.0}", self.wand_tolerance)));
            }
            Keysym::bracketleft => self.brush.size = (self.brush.size / 1.2).max(1.0),
            Keysym::bracketright => self.brush.size = (self.brush.size * 1.2).min(5000.0),
            Keysym::e => self.set_tool(Tool::Eraser),
            Keysym::b => self.set_tool(Tool::Brush),
            Keysym::B => {
                // Built-in tips, then the custom ones.
                let builtin = mosaix_core::TIP_NAMES.len() as u32;
                let custom = self.custom_tips.len() as u32;
                self.brush.tip = match self.brush.tip + 1 {
                    t if t == builtin && custom > 0 => mosaix_gpu::FIRST_CUSTOM_TIP,
                    t if t < builtin => t,
                    t if t >= mosaix_gpu::FIRST_CUSTOM_TIP && t < mosaix_gpu::FIRST_CUSTOM_TIP + custom => t,
                    _ => 0,
                };
                tracing::info!("{}", tr!("Brush tip: {tip}", tip = tr(mosaix_core::TIP_NAMES[self.brush.tip as usize])));
            }
            Keysym::u => self.set_tool(Tool::Smudge),
            Keysym::U if shift => {
                let ellipse = matches!(self.tool, Tool::Shape { ellipse: false });
                self.set_tool(Tool::Shape { ellipse });
            }
            Keysym::t if !ctrl => self.set_tool(Tool::Text),
            Keysym::r if ctrl && alt => self.rasterize_active(),
            Keysym::s if !ctrl => self.set_tool(Tool::Clone),
            // j: spot healing, again: remove (AI), again: healing brush (and around).
            Keysym::j if !ctrl && !shift => self.set_tool(match self.tool {
                Tool::SpotHeal => Tool::Remove,
                Tool::Remove => Tool::Heal,
                _ => Tool::SpotHeal,
            }),
            Keysym::J if shift => {
                let on = self.brush.size_jitter == 0.0;
                let b = &mut self.brush;
                (b.size_jitter, b.angle_jitter, b.scatter, b.flow_jitter) =
                    if on { (0.3, 30.0, 0.25, 0.3) } else { (0.0, 0.0, 0.0, 0.0) };
                tracing::info!(
                    "{}",
                    if on {
                        tr("Brush dynamics (random size, angle, scatter, flow) on")
                    } else {
                        tr("Brush dynamics (random size, angle, scatter, flow) off")
                    }
                );
            }
            Keysym::m => {
                let ellipse = matches!(self.tool, Tool::Marquee { ellipse: false });
                self.set_tool(Tool::Marquee { ellipse })
            }
            // Like Photoshop's O (Shift+O there): dodge → burn → sponge.
            Keysym::o if !ctrl => self.set_tool(match self.tool {
                Tool::Dodge => Tool::Burn,
                Tool::Burn => Tool::Sponge,
                _ => Tool::Dodge,
            }),
            Keysym::l => self.set_tool(Tool::Lasso),
            Keysym::g => {
                let radial = matches!(self.tool, Tool::Gradient { radial: false });
                self.set_tool(Tool::Gradient { radial })
            }
            Keysym::v => self.set_tool(Tool::Move),
            Keysym::q => {
                self.quick_mask = !self.quick_mask;
                tracing::info!("{}", if self.quick_mask { tr("Quick Mask view on") } else { tr("Quick Mask view off") });
            }
            Keysym::x => self.swap_colors(),
            Keysym::d if !ctrl => self.reset_colors(),
            Keysym::i if !ctrl => self.set_tool(Tool::Eyedropper),
            Keysym::c => self.set_tool(Tool::Crop),
            Keysym::w => {
                if self.tool == Tool::Wand {
                    self.wand_contiguous = !self.wand_contiguous;
                }
                self.set_tool(Tool::Wand);
                tracing::info!(
                    "{}",
                    tr!(
                        "Magic Wand: Tolerance {tol} ([ / ]), {mode} (W toggles), samples all layers",
                        tol = format!("{:.0}", self.wand_tolerance),
                        mode = if self.wand_contiguous { tr("contiguous only") } else { tr("anywhere in the image") }
                    )
                );
            }
            Keysym::p if !ctrl => self.set_tool(Tool::Pen),
            Keysym::P if shift && !ctrl => {
                self.palette_idx = (self.palette_idx + 1) % PALETTE.len();
                self.brush.color = PALETTE[self.palette_idx];
            }
            Keysym::h => {
                let steps = [0.0, 0.5, 0.8, 1.0];
                let i = steps.iter().position(|&h| h >= self.brush.hardness - 1e-3).unwrap_or(0);
                self.brush.hardness = steps[(i + 1) % steps.len()];
                tracing::info!("{}", tr!("Hardness {pct} %", pct = format!("{:.0}", self.brush.hardness * 100.0)));
            }
            Keysym::S if shift => {
                let steps = [0.0, 0.4, 0.7, 0.85];
                let i = steps.iter().position(|&s| s >= self.brush.smoothing - 1e-3).unwrap_or(0);
                self.brush.smoothing = steps[(i + 1) % steps.len()];
                tracing::info!("{}", tr!("Smoothing {pct} %", pct = format!("{:.0}", self.brush.smoothing * 100.0)));
            }
            Keysym::F1 => print!("{}", tr(crate::HELP)),
            Keysym::F12 => self.start_bench(false),
            Keysym::r if ctrl && !alt && !shift => {
                self.settings.view.rulers = !self.settings.view.rulers;
                self.r_key = false;
            }
            Keysym::semicolon if ctrl && alt => {
                if let Some(e) = self.engine() {
                    e.set_guides(Vec::new(), tr("Clear Guides"));
                    self.touch();
                }
            }
            Keysym::semicolon if ctrl => self.settings.view.guides = !self.settings.view.guides,
            Keysym::colon if ctrl => {
                self.settings.view.snap = !self.settings.view.snap;
                tracing::info!("{}", if self.settings.view.snap { tr("Snap on") } else { tr("Snap off") });
            }
            Keysym::apostrophe if ctrl => self.settings.view.grid = !self.settings.view.grid,
            _ => return,
        }
        self.needs_redraw = true;
    }

    /// X: swaps brush and background colour.
    fn swap_colors(&mut self) {
        std::mem::swap(&mut self.brush.color, &mut self.bg_color);
        self.needs_redraw = true;
    }

    /// D: black brush, white background.
    fn reset_colors(&mut self) {
        self.brush.color = [0.0; 3];
        self.bg_color = [1.0; 3];
        self.needs_redraw = true;
    }

    /// Eyedropper: the (averaged) colour at a document point becomes the brush colour.
    fn eyedrop(&mut self, doc: DVec2) {
        let (r, all) = (self.eyedrop_radius, self.eyedrop_all);
        let Some(e) = self.engine() else { return };
        let (x, y) = (doc.x.floor() as i64, doc.y.floor() as i64);
        let from = if all {
            SampleFrom::Image
        } else {
            SampleFrom::Layer
        };
        // A text/shape/group layer has no pixels of its own: use the image.
        let c = e
            .sample_average(x, y, r, from)
            .or_else(|_| e.sample_average(x, y, r, SampleFrom::Image));
        if let Ok(c) = c
            && c[3] > 0.0
        {
            let lin = |v: f32| mosaix_core::color::srgb_to_linear(v);
            self.brush.color = [lin(c[0]), lin(c[1]), lin(c[2])];
            self.needs_redraw = true;
        }
    }

    fn set_tool(&mut self, tool: Tool) {
        if tool != Tool::Text {
            self.end_text_edit();
        }
        if tool != Tool::Pen {
            self.pen_finish(false);
        }
        match self.boxed.as_ref().map(|b| b.0) {
            Some(BoxKind::Transform) => self.commit_box(),
            Some(BoxKind::Crop) if tool != Tool::Crop => self.boxed = None,
            _ => {}
        }
        if tool == Tool::Crop {
            tracing::info!(
                "{}",
                tr(
                    "Crop: drag a frame or move the handles (beyond the edge enlarges the canvas), Enter applies, Esc cancels."
                )
            );
        }
        self.tool = tool;
        self.brush.eraser = tool == Tool::Eraser;
        tracing::info!("{}", tr!("Tool: {tool}", tool = tool.name()));
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
            Err(err) => tracing::warn!("{}", tr(err)),
        }
    }

    fn undo(&mut self) {
        if let Some(label) = self.engine().and_then(|e| e.undo()) {
            tracing::info!("{}", tr!("Undo: {label}", label = label));
            self.touch();
            self.print_layers();
        }
    }

    fn redo(&mut self) {
        if let Some(label) = self.engine().and_then(|e| e.redo()) {
            tracing::info!("{}", tr!("Redo: {label}", label = label));
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
        self.edit(tr("Blend Mode"), |d| {
            let id = d.active;
            let l = d.find_mut(id).unwrap();
            l.blend = l.blend.cycle(step);
            true
        });
    }

    // ---- agents -----------------------------------------------------------

    /// Reads from an agent connection and answers every complete line.
    fn agent_io(
        &mut self,
        stream: &std::os::unix::net::UnixStream,
        buf: &mut Vec<u8>,
    ) -> PostAction {
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

    /// Runs an agent command from the interface (dialogs that have one);
    /// errors go to the message line.
    fn run_agent_cmd(&mut self, v: serde_json::Value) {
        let cmd = match command::Command::parse(v) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("{e:#}");
                return;
            }
        };
        let mutates = cmd.mutates();
        let mut ctx = command::Ctx {
            gpu: &self.gpu,
            format: self.surface_config.format,
            session: &mut self.session,
        };
        match command::execute(&mut ctx, cmd) {
            Ok(_) if mutates => self.touch(),
            Ok(_) => {}
            Err(e) => tracing::warn!("{e:#}"),
        }
        self.needs_redraw = true;
        self.update_title();
    }

    fn agent_line(&mut self, line: &[u8]) -> String {
        let req: serde_json::Value = match serde_json::from_slice(line) {
            Ok(v) => v,
            Err(e) => {
                return ipc::response(
                    serde_json::Value::Null,
                    Err(anyhow::anyhow!("invalid JSON: {e}")),
                );
            }
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
            return ipc::response(
                id,
                Err(anyhow::anyhow!(
                    "the user is painting right now – retry in a moment"
                )),
            );
        }
        let (name, mutates) = (cmd.name(), cmd.mutates());
        tracing::info!("Agent: {name}");
        let mut ctx = command::Ctx {
            gpu: &self.gpu,
            format: self.surface_config.format,
            session: &mut self.session,
        };
        let result = command::execute(&mut ctx, cmd);
        if let Ok(out) = &result {
            if out.replaced_document {
                self.load_custom_tips();
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
        tracing::info!(
            "{}",
            tr!(
                "Benchmark: {n} frames zoom/rotation/pan …",
                n = BENCH_FRAMES
            )
        );
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
            let pct = |p: f64| {
                v.get(((v.len() as f64 - 1.0) * p) as usize)
                    .copied()
                    .unwrap_or(0.0)
            };
            (avg, pct(0.5), pct(0.99), v.last().copied().unwrap_or(0.0))
        };
        let (mut cpu, mut iv) = (b.cpu_ms, b.interval_ms);
        let c = stat(&mut cpu);
        let i = stat(&mut iv);
        let tiles = self
            .session
            .as_ref()
            .map_or(0, |s| s.engine.composite_tiles_live());
        let f2 = |x: f64| format!("{x:.2}");
        println!(
            "{}",
            tr!(
                "\n=== Benchmark ({frames} frames, {w}×{h} px, {tiles} tiles) ===\n\
                 CPU per frame:    Ø {c0} ms   p50 {c1}   p99 {c2}   max {c3}\n\
                 Frame interval:   Ø {i0} ms   p50 {i1}   p99 {i2}   max {i3}   (≈ {fps} fps)\n",
                frames = cpu.len(),
                w = self.surface_config.width,
                h = self.surface_config.height,
                tiles = tiles,
                c0 = f2(c.0),
                c1 = f2(c.1),
                c2 = f2(c.2),
                c3 = f2(c.3),
                i0 = f2(i.0),
                i1 = f2(i.1),
                i2 = f2(i.2),
                i3 = f2(i.3),
                fps = format!("{:.0}", 1000.0 / i.0.max(1e-6)),
            )
        );
        self.view = b.base;
        if b.exit_after {
            self.exit = true;
        }
    }

    /// Whether a screen position (physical px) is covered by the interface.
    fn over_ui(&self, pos: DVec2) -> bool {
        let (a, b) = self.canvas_rect;
        let outside =
            b != DVec2::ZERO && (pos.x < a.x || pos.y < a.y || pos.x >= b.x || pos.y >= b.y);
        let p = egui::pos2((pos.x / self.scale) as f32, (pos.y / self.scale) as f32);
        let floating = self
            .gui
            .ctx
            .layer_id_at(p)
            .is_some_and(|l| l.order != egui::Order::Background);
        outside || floating
    }

    fn tool_id(&self) -> ToolId {
        match self.tool {
            Tool::Brush => ToolId::Brush,
            Tool::Eraser => ToolId::Eraser,
            Tool::Smudge => ToolId::Smudge,
            Tool::Clone => ToolId::Clone,
            Tool::Heal => ToolId::Heal,
            Tool::SpotHeal => ToolId::SpotHeal,
            Tool::Text => ToolId::Text,
            Tool::Shape { .. } => ToolId::Shape,
            Tool::Pen => ToolId::Pen,
            Tool::Eyedropper => ToolId::Eyedropper,
            Tool::Marquee { ellipse: false } => ToolId::Marquee,
            Tool::Marquee { ellipse: true } => ToolId::Ellipse,
            Tool::Lasso => ToolId::Lasso,
            Tool::Wand => ToolId::Wand,
            Tool::ObjectSelect => ToolId::ObjectSelect,
            Tool::Remove => ToolId::Remove,
            Tool::Dodge => ToolId::Dodge,
            Tool::Burn => ToolId::Burn,
            Tool::Sponge => ToolId::Sponge,
            Tool::Blur => ToolId::Blur,
            Tool::Sharpen => ToolId::Sharpen,
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
            cursor_doc: self
                .cursor
                .filter(|_| matches!(self.drag, Drag::None))
                .map(|c| {
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
            tone: self.settings.tone,
            clone_all: self.clone_all,
            clone_aligned: self.clone_aligned,
            clone_source: self.clone_source.is_some(),
            wand_tolerance: self.wand_tolerance,
            wand_contiguous: self.wand_contiguous,
            bg_color: self.bg_color,
            custom_tips: self.custom_tips.clone(),
            patterns: self.patterns.clone(),
            filter_plugins: self.filter_plugins.clone(),
            actions: self.actions.clone(),
            recording: self.recording.as_ref().map(Vec::len),
            eyedrop_radius: self.eyedrop_radius,
            eyedrop_all: self.eyedrop_all,
            boxed: self.boxed.as_ref().map(|(k, _)| match k {
                BoxKind::Transform => tr("Free Transform"),
                BoxKind::Crop => tr("Crop"),
            }),
            mask_edit: self.mask_edit,
            box_angle: match &self.boxed {
                Some((BoxKind::Transform, b)) if b.quad.is_none() => Some(b.angle()),
                _ => None,
            },
            box_mode: match &self.boxed {
                Some((BoxKind::Transform, b)) => Some(b.mode),
                _ => None,
            },
            liquify: e.liquify_active().then_some((
                self.liq_tool,
                self.liq_size,
                self.liq_strength,
            )),
            text_overlay: self.text_overlay(),
            path_overlay: (self.tool == Tool::Pen)
                .then(|| self.path_overlay())
                .flatten(),
            layers: ui::panels::layer_rows(doc),
            active: doc.active,
            active_opacity: active.map_or(1.0, |l| l.opacity),
            active_fill: active.map_or(1.0, |l| l.fill_opacity),
            active_blend: active.map_or(mosaix_core::BlendMode::Normal, |l| l.blend),
            keymap: std::sync::Arc::new(self.settings.keymap.clone()),
            active_pass_through: active.filter(|l| l.is_group()).map(|l| l.pass_through),
            history,
            history_done,
            has_selection: doc.selection.is_some(),
            channels: doc.channels.iter().map(|c| c.name.clone()).collect(),
            active_bounds: None,
            view: self.view,
            guides: doc.guides.clone(),
            view_opts: self.settings.view,
            generative: self.settings.generative.clone(),
            openai_key: self.openai_key.clone(),
            panels: Default::default(),
            status: self.ui_status.clone(),
            busy: self.bg_running > 0 || self.saving || self.save_job.is_some() || self.loading,
            recent: self.settings.recent.clone(),
            active_adjustment: active.and_then(|l| l.adjustment().cloned()),
            active_vector: active.and_then(|l| l.vector().cloned()),
            active_raw: active.and_then(|l| {
                let r = l.raw()?;
                Some(ui::raw_ui::ActiveRaw {
                    file: r.data.name.clone(),
                    develop: r.develop,
                    summary: self.session.as_ref().and_then(|s| s.engine.raw_summary(l.id)),
                    smart: r.data.is_image(),
                })
            }),
            active_effects: active
                .filter(|l| l.is_pixels() || l.is_rendered())
                .map(|l| l.effects.clone()),
            fonts: self.fonts.clone(),
            paths: {
                // Bézier path layers, in document coordinates.
                let mut v = Vec::new();
                doc.for_each(&mut |l, _| {
                    if let Some(vl) = l.vector()
                        && let mosaix_core::vector::VectorKind::Shape(mosaix_core::vector::ShapeSpec {
                            geometry: mosaix_core::vector::Geometry::Bezier { nodes, closed },
                            ..
                        }) = &vl.kind
                    {
                        let m = DAffine2::from_cols_array(&vl.transform);
                        let t = |p: [f64; 2]| {
                            let q = m.transform_point2(DVec2::new(p[0], p[1]));
                            [q.x, q.y]
                        };
                        let nodes = nodes
                            .iter()
                            .map(|n| mosaix_core::vector::PathNode { p: t(n.p), h_in: t(n.h_in), h_out: t(n.h_out) })
                            .collect();
                        v.push(ui::panels::PathChoice { name: l.name.clone(), nodes, closed: *closed });
                    }
                });
                v
            },
            favourite_fonts: self.settings.favourite_fonts.clone(),
            recent_fonts: self.settings.recent_fonts.clone(),
            shape_ellipse: matches!(self.tool, Tool::Shape { ellipse: true }),
            histogram: None,
        })
    }

    fn apply_ui(&mut self, actions: Vec<Action>) {
        for a in actions {
            self.needs_redraw = true;
            // Like Photoshop: changing anything else confirms an open free transform.
            if matches!(self.boxed, Some((BoxKind::Transform, _)))
                && !matches!(
                    a,
                    Action::CommitBox
                        | Action::CancelBox
                        | Action::BoxRotate(_)
                        | Action::BoxAngle(_)
                        | Action::BoxFlip { .. }
                        | Action::BoxMode(_)
                )
            {
                self.commit_box();
            }
            let step = if self.recording.is_some() { recorded_action(&a) } else { None };
            if let Some(step) = step {
                self.record(step);
            }
            match a {
                Action::Command(c) => self.run_cmd(c),
                Action::Agent(v) => self.run_agent_cmd(v),
                Action::ModifySelection { op, amount } => self.modify_selection(op, amount),
                Action::ColorRange { cmd, phase } => {
                    if std::mem::take(&mut self.range_preview)
                        && let Some(e) = self.engine()
                    {
                        e.undo();
                    }
                    match phase {
                        ui::panels::RangePhase::Cancel => {}
                        ui::panels::RangePhase::Preview => {
                            self.run_agent_cmd(cmd);
                            self.range_preview = true;
                        }
                        ui::panels::RangePhase::Commit => {
                            if self.recording.is_some() {
                                self.record(cmd.clone());
                            }
                            self.run_agent_cmd(cmd);
                        }
                    }
                }
                Action::SetGuides(g) => {
                    let label = match g.len().cmp(&self.session.as_ref().map_or(0, |s| s.engine.doc().guides.len())) {
                        std::cmp::Ordering::Greater => tr("New Guide"),
                        std::cmp::Ordering::Less => tr("Delete Guide"),
                        std::cmp::Ordering::Equal => tr("Move Guide"),
                    };
                    if let Some(e) = self.engine() {
                        e.set_guides(g, label);
                        self.touch();
                    }
                    // A new guide is visible.
                    self.settings.view.guides = true;
                }
                Action::ViewOpts(v) => self.settings.view = v,
                Action::ToneOpts(t) => self.settings.tone = t,
                Action::PickFile(mode) => self.file_dialog(mode),
                Action::GenSettings(g) => {
                    self.settings.generative = g;
                    if let Err(e) = self.settings.save_to(&settings::config_path()) {
                        tracing::warn!("{}", tr!("Settings not saved: {error}", error = format!("{e:#}")));
                    }
                }
                Action::GenerativeFill(prompt) => self.generative_fill(prompt),
                Action::SaveApiKey(key) => {
                    match settings::store_openai_key(key.as_deref()) {
                        Ok(()) => tracing::info!("{}", if key.is_some() { tr("API key saved") } else { tr("API key removed") }),
                        Err(e) => tracing::warn!("{}", tr!("API key not saved: {error}", error = format!("{e:#}"))),
                    }
                    self.openai_key = settings::key_state();
                }
                Action::PlaceEmbedded(p) => self.run_agent_cmd(serde_json::json!({ "cmd": "layer_place", "path": p })),
                Action::ReplaceContents(p) => self.run_agent_cmd(serde_json::json!({ "cmd": "smart_replace", "path": p })),
                Action::ConvertToSmart => self.run_agent_cmd(serde_json::json!({ "cmd": "layer_convert_smart" })),
                Action::ViewCenter(x, y) => {
                    let (a, b) = self.canvas_rect;
                    let mid = if b == DVec2::ZERO { self.view.viewport * 0.5 } else { (a + b) * 0.5 };
                    let now = self.view.doc_to_screen(DVec2::new(x, y));
                    self.view.pan(mid - now);
                }
                Action::ViewZoom(z) => {
                    let (a, b) = self.canvas_rect;
                    let mid = if b == DVec2::ZERO { self.view.viewport * 0.5 } else { (a + b) * 0.5 };
                    self.view.set_zoom_at(mid, z.clamp(mosaix_core::view::MIN_ZOOM, mosaix_core::view::MAX_ZOOM));
                }
                Action::Run(e) => self.run_action(&e),
                Action::StartRecording => self.start_recording(),
                Action::StopRecording(name) => {
                    self.stop_recording(name);
                    self.actions = std::sync::Arc::new(actions::list(&actions::actions_dir()));
                }
                Action::DownloadModel(id) => self.download_model(&id),
                Action::ReloadActions => {
                    self.actions = std::sync::Arc::new(actions::list(&actions::actions_dir()));
                    self.needs_redraw = true;
                }
                Action::Tool(t) => {
                    let tool = match t {
                        ToolId::Brush => Tool::Brush,
                        ToolId::Eraser => Tool::Eraser,
                        ToolId::Smudge => Tool::Smudge,
                        ToolId::Clone => Tool::Clone,
                        ToolId::Heal => Tool::Heal,
                        ToolId::SpotHeal => Tool::SpotHeal,
                        ToolId::Text => Tool::Text,
                        ToolId::Shape => Tool::Shape { ellipse: false },
                        ToolId::Pen => Tool::Pen,
                        ToolId::Eyedropper => Tool::Eyedropper,
                        ToolId::Marquee => Tool::Marquee { ellipse: false },
                        ToolId::Ellipse => Tool::Marquee { ellipse: true },
                        ToolId::Lasso => Tool::Lasso,
                        ToolId::Wand => Tool::Wand,
                        ToolId::ObjectSelect => Tool::ObjectSelect,
                        ToolId::Remove => Tool::Remove,
                        ToolId::Dodge => Tool::Dodge,
                        ToolId::Burn => Tool::Burn,
                        ToolId::Sponge => Tool::Sponge,
                        ToolId::Blur => Tool::Blur,
                        ToolId::Sharpen => Tool::Sharpen,
                        ToolId::Gradient => Tool::Gradient { radial: false },
                        ToolId::Radial => Tool::Gradient { radial: true },
                        ToolId::Move => Tool::Move,
                        ToolId::Crop => Tool::Crop,
                    };
                    self.set_tool(tool);
                }
                Action::Color(c) => self.brush.color = c,
                Action::BgColor(c) => self.bg_color = c,
                Action::ImportBrushes(p) => self.import_brushes(&p),
                Action::ImportPatterns(p) => self.import_patterns(&p),
                Action::ReloadFilterPlugins => {
                    let lib = filters::load_library(&filters::filter_dir());
                    let (ok, bad): (Vec<_>, Vec<_>) = lib.iter().partition(|p| p.result.is_ok());
                    for b in &bad {
                        if let Err(e) = &b.result {
                            tracing::warn!("{}: {e}", b.file);
                        }
                    }
                    tracing::info!("{}", tr!("{n} filter plugins loaded, {bad} with errors", n = ok.len(), bad = bad.len()));
                    self.filter_plugins = std::sync::Arc::new(lib);
                    self.needs_redraw = true;
                }
                Action::Shortcut { id, combo, reset } => {
                    ui::palette::rebind(&mut self.settings.keymap, &id, combo, reset);
                    self.store_settings();
                }
                Action::ResetShortcuts => {
                    self.settings.keymap.clear();
                    self.store_settings();
                }
                Action::SwapColors => self.swap_colors(),
                Action::ResetColors => self.reset_colors(),
                Action::Eyedrop { radius, all } => {
                    self.eyedrop_radius = radius;
                    self.eyedrop_all = all;
                }
                Action::Brush(m) => {
                    let b = &mut self.brush;
                    (
                        b.size,
                        b.hardness,
                        b.flow,
                        b.opacity,
                        b.spacing,
                        b.smoothing,
                    ) = (
                        m.size,
                        m.hardness,
                        m.flow,
                        m.opacity,
                        m.spacing,
                        m.smoothing,
                    );
                    (b.tip, b.angle, b.roundness, b.follow_direction) =
                        (m.tip, m.angle, m.roundness, m.follow_direction);
                    (b.size_jitter, b.angle_jitter, b.scatter, b.flow_jitter) =
                        (m.size_jitter, m.angle_jitter, m.scatter, m.flow_jitter);
                }
                Action::SmudgeStrength(v) => self.smudge_strength = v,
                Action::ShapeEllipse(ellipse) => self.tool = Tool::Shape { ellipse },
                Action::FontUsed(name) => {
                    let r = &mut self.settings.recent_fonts;
                    r.retain(|f| f != &name);
                    r.insert(0, name);
                    r.truncate(8);
                    self.store_settings();
                }
                Action::ToggleFavouriteFont(name) => {
                    let f = &mut self.settings.favourite_fonts;
                    if let Some(i) = f.iter().position(|x| x == &name) {
                        f.remove(i);
                    } else {
                        f.push(name);
                        f.sort_by_key(|n| n.to_lowercase());
                    }
                    self.store_settings();
                }
                Action::SetEffects { effects, done } => {
                    if let Some(e) = self.engine() {
                        let id = e.doc().active;
                        e.edit_merged(tr("Layer Style"), &format!("fx-{id}"), |d| {
                            let l = d.find_mut(id).unwrap();
                            let changed = l.effects != effects;
                            // Global light: a changed angle reaches every layer's effects that use it.
                            let global = |fx: &[mosaix_core::style::Effect]| {
                                fx.iter().find(|e| e.global_light).and_then(|e| e.kind.angle())
                            };
                            let moved = global(&effects).filter(|g| global(&l.effects) != Some(*g));
                            l.effects = effects;
                            if let Some(g) = moved {
                                fn walk(ls: &mut [mosaix_core::Layer<mosaix_gpu::Tile>], g: f32) {
                                    for l in ls {
                                        for e in l.effects.iter_mut().filter(|e| e.global_light) {
                                            if let Some((a, _)) = e.kind.light_mut() {
                                                *a = g;
                                            }
                                        }
                                        if let mosaix_core::doc::Content::Group(c) = &mut l.content {
                                            walk(c, g);
                                        }
                                    }
                                }
                                walk(&mut d.layers, g);
                            }
                            changed
                        });
                        if done {
                            e.end_merge();
                        }
                        self.touch();
                    }
                }
                Action::SetVector { v, done } => {
                    if let Some(e) = self.engine() {
                        let id = e.doc().active;
                        let label = if matches!(v.kind, mosaix_core::vector::VectorKind::Text(_)) {
                            tr("Edit Text")
                        } else {
                            tr("Edit Shape")
                        };
                        let _ = e.set_vector(id, v, label, Some(&format!("vector-{id}")));
                        if done {
                            e.end_merge();
                        }
                        self.touch();
                    }
                }
                Action::SetRaw { d, done } => {
                    if let Some(e) = self.engine() {
                        let id = e.doc().active;
                        let _ = e.set_raw(id, d, tr("Develop RAW"), Some(&format!("raw-{id}")));
                        if done {
                            e.end_merge();
                        }
                        self.touch();
                    }
                }
                Action::CloneOptions { all, aligned } => {
                    self.clone_all = all;
                    if aligned != self.clone_aligned {
                        self.clone_offset = None;
                    }
                    self.clone_aligned = aligned;
                }
                Action::WandTolerance(v) => self.wand_tolerance = v,
                Action::WandContiguous(v) => self.wand_contiguous = v,
                Action::SelectLayer(id) => {
                    if let Some(e) = self.engine() {
                        e.select(|d| d.active = id);
                    }
                    let has_mask = self
                        .engine()
                        .and_then(|e| e.doc().find(id).map(|l| l.mask.is_some()));
                    if has_mask == Some(false) {
                        self.mask_edit = false;
                    }
                }
                Action::ToggleVisible(id) => self.edit(tr("Visibility"), |d| {
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
                    tracing::info!(
                        "{}",
                        tr!(
                            "Target: {target}",
                            target = if self.mask_edit {
                                tr("Layer Mask")
                            } else {
                                tr("Pixels")
                            }
                        )
                    );
                }
                Action::RenameLayer(id, name) => self.edit(tr("Rename"), |d| {
                    let l = d.find_mut(id).unwrap();
                    let changed = l.name != name;
                    l.name = name;
                    changed
                }),
                Action::MoveLayer {
                    id,
                    target,
                    above,
                    into,
                } => self.edit(tr("Move Layer"), |d| d.move_layer(id, target, above, into)),
                Action::LayerOpacity { value, done, fill } => {
                    if let Some(e) = self.engine() {
                        let (name, key) = if fill { (mosaix_core::i18n::tr_ctx("layer", "Fill"), "layer-fill") } else { (tr("Opacity"), "layer-opacity") };
                        let changed = e.edit_merged(name, key, |d| {
                            let id = d.active;
                            let l = d.find_mut(id).unwrap();
                            let v = if fill { &mut l.fill_opacity } else { &mut l.opacity };
                            let c = *v != value;
                            *v = value;
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
                Action::LayerBlend(b) => self.edit(tr("Blend Mode"), |d| {
                    let id = d.active;
                    let l = d.find_mut(id).unwrap();
                    let c = l.blend != b || l.pass_through;
                    l.blend = b;
                    l.pass_through = false;
                    c
                }),
                Action::PassThrough(on) => self.edit(tr("Blend Mode"), |d| {
                    let id = d.active;
                    let l = d.find_mut(id).unwrap();
                    let c = l.is_group() && l.pass_through != on;
                    if c {
                        l.pass_through = on;
                    }
                    c
                }),
                Action::DeleteLayer => self.edit(tr("Delete Layer"), |d| d.remove_active()),
                Action::HistoryJump(n) => {
                    if self.engine().is_some_and(|e| e.jump_history(n)) {
                        self.touch();
                    }
                }
                Action::Liquify {
                    tool,
                    size,
                    strength,
                } => {
                    self.liq_tool = tool;
                    self.liq_size = size;
                    self.liq_strength = strength;
                }
                Action::LiquifyCommit => self.commit_liquify(),
                Action::LiquifyCancel => self.cancel_liquify(),
                Action::LiquifyUndo => {
                    if let Some(e) = self.engine() {
                        e.liquify_undo();
                    }
                }
                Action::BoxRotate(deg) => {
                    if let Some((BoxKind::Transform, b)) = &mut self.boxed {
                        b.rotate_by(deg);
                    }
                }
                Action::BoxAngle(deg) => {
                    if let Some((BoxKind::Transform, b)) = &mut self.boxed {
                        let now = b.angle();
                        b.rotate_by(if b.fwd.matrix2.determinant() < 0.0 {
                            now - deg
                        } else {
                            deg - now
                        });
                    }
                }
                Action::BoxFlip { x, y } => {
                    if let Some((BoxKind::Transform, b)) = &mut self.boxed {
                        b.flip(x, y);
                    }
                }
                Action::BoxMode(mode) => {
                    if self.boxed.is_none() {
                        self.begin_transform();
                    }
                    if let Some((BoxKind::Transform, b)) = &mut self.boxed {
                        b.mode = mode;
                    }
                }
                Action::CommitBox => self.commit_box(),
                Action::CancelBox => self.cancel_box(),
                Action::CanvasSize { w, h, anchor } => {
                    let Some((ow, oh)) = self.session.as_ref().map(|s| s.engine.size()) else {
                        continue;
                    };
                    let x = ((ow as f64 - w as f64) * anchor.0).round() as i64;
                    let y = ((oh as f64 - h as f64) * anchor.1).round() as i64;
                    self.engine_op(|e| e.crop(x, y, w, h, tr("Canvas Size")));
                    self.view_fitted = false;
                }
                Action::AddAdjustment(adj) => {
                    let label = tr!("Adjustment Layer {name}", name = adj.name());
                    if self
                        .engine()
                        .and_then(|e| e.add_adjustment(adj, &label))
                        .is_some()
                    {
                        self.touch();
                    }
                }
                Action::SetAdjustment { adj, done } => {
                    if let Some(e) = self.engine() {
                        let id = e.doc().active;
                        let changed =
                            e.edit_merged(tr("Edit Adjustment"), &format!("adjust-{id}"), |d| {
                                let l = d.find_mut(id).unwrap();
                                let c = l.adjustment() != Some(&adj);
                                l.content = mosaix_core::Content::Adjustment(adj);
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
                    self.engine_op(|e| e.apply_filter(&f, &label));
                    tracing::info!("{label}: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                }
                Action::AddSmartFilter(f) => {
                    let label = f.name();
                    self.engine_op(|e| e.add_smart_filter(&f, &label));
                }
                Action::SetSmartFilter {
                    layer,
                    index,
                    filter,
                } => {
                    self.engine_op(|e| {
                        e.set_smart_filter(layer, index, &filter, tr("Edit Smart Filter"))
                    });
                }
                Action::ToggleSmartFilter { layer, index } => {
                    self.edit(tr("Toggle Smart Filter"), |d| {
                        d.find_mut(layer)
                            .and_then(|l| l.filters.get_mut(index))
                            .map(|f| f.enabled = !f.enabled)
                            .is_some()
                    })
                }
                Action::RemoveSmartFilter { layer, index } => {
                    self.edit(tr("Remove Smart Filter"), |d| {
                        d.find_mut(layer)
                            .filter(|l| index < l.filters.len())
                            .map(|l| l.filters.remove(index))
                            .is_some()
                    })
                }
                Action::BakeSmartFilters(layer) => {
                    self.engine_op(|e| e.bake_smart_filters(layer, tr("Apply Smart Filters")))
                }
                Action::OpenFile(p) => self.guard_unsaved(Pending::Open(p)),
                Action::OpenRaw(p, d) => self.open_source(Source::Raw(p, d), false),
                Action::NewDocument { w, h, white } => {
                    self.guard_unsaved(Pending::New { w, h, white })
                }
                Action::SaveAs(p) => self.save_to(Some(p)),
                Action::ExportAs(p, opts) => self.export_to(p, opts),
                Action::Unsaved(a) => self.answer_unsaved(a),
                Action::ImageSize { w, h } => {
                    self.engine_op(|e| e.resize_image(w, h, Interp::Bicubic, tr("Image Size")));
                    self.view_fitted = false;
                }
                Action::Upscale(k) => self.ai_upscale(k),
            }
        }
    }

    /// Builds the interface for this frame and applies what the user did.
    fn run_ui(&mut self) {
        // The font window belongs to the active text layer.
        let text_active = self
            .session
            .as_ref()
            .and_then(|s| s.engine.doc().active_layer())
            .is_some_and(|l| {
                matches!(
                    l.vector().map(|v| &v.kind),
                    Some(mosaix_core::vector::VectorKind::Text(_))
                )
            });
        if !text_active {
            self.ui_state.vec_editor.fonts.open = false;
        }
        if !self.ui_state.vec_editor.fonts.open {
            self.ui_state.vec_editor.fonts.preview = None;
        }
        // Editing ends when its layer is no longer the active one.
        if let Some(te) = &self.text_edit
            && self.session.as_ref().is_none_or(|s| {
                s.engine.doc().active != te.layer || s.engine.doc().find(te.layer).is_none()
            })
        {
            self.text_edit = None;
        }
        // Thumbnails: a few per frame, not while dragging.
        if matches!(self.drag, Drag::None)
            && let Some(s) = &mut self.session
            && (s.engine.update_thumbnails(4) | s.engine.update_mask_thumbnails(2))
        {
            self.needs_redraw = true;
        }
        if self.fonts.is_empty()
            && self.session.as_ref().is_some_and(|s| {
                s.engine
                    .doc()
                    .active_layer()
                    .is_some_and(|l| l.vector().is_some())
            })
        {
            self.ensure_fonts();
        }
        let Some(mut model) = self.ui_model() else {
            return;
        };
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
                if row.has_mask
                    && let Some((key, px)) = s.engine.mask_thumbnail(row.id)
                {
                    let uv = s.engine.thumbnail_uv();
                    let entry = self.mask_thumbs.entry(row.id);
                    let tex = match entry {
                        std::collections::hash_map::Entry::Occupied(mut o) => {
                            if o.get().0 != key {
                                let n = ui::panels::THUMB_PX;
                                o.get_mut().1.set(egui::ColorImage::from_gray([n, n], &px), egui::TextureOptions::LINEAR);
                                o.get_mut().0 = key;
                            }
                            o.get().1.id()
                        }
                        std::collections::hash_map::Entry::Vacant(v) => {
                            let n = ui::panels::THUMB_PX;
                            let h = self.gui.ctx.load_texture(format!("mask {}", row.id), egui::ColorImage::from_gray([n, n], &px), egui::TextureOptions::LINEAR);
                            v.insert((key, h)).1.id()
                        }
                    };
                    row.mask_thumb = Some((tex, uv));
                }
            }
            let rows: std::collections::HashSet<LayerId> = model.layers.iter().map(|r| r.id).collect();
            self.mask_thumbs.retain(|id, _| rows.contains(id));
        }
        self.panel_data(&mut model);
        let size = (self.surface_config.width, self.surface_config.height);
        let changed =
            self.last_model.as_ref() != Some(&model) || ui::log::generation() != self.ui_log_gen;
        if !self.gui.needs_pass(size, self.scale, changed) {
            return;
        }
        // First pass: the saved panel layout.
        if self.ui_state.dock.is_none() {
            self.ui_state.saved_layout = self.settings.dock_layout.clone();
            self.ui_state.dock = self
                .settings
                .dock_layout
                .as_ref()
                .and_then(ui::dock::layout_from_json);
        }
        let mut actions = Vec::new();
        self.ui_state.panels_shown = Default::default();
        let state = &mut self.ui_state;
        self.gui.run(size, self.scale, |ctx| {
            actions = ui::panels::build(ctx, &model, state)
        });
        let s = self.scale;
        let rect = match self.ui_state.canvas {
            Some(r) => (
                DVec2::new(r.min.x as f64 * s, r.min.y as f64 * s),
                DVec2::new(r.max.x as f64 * s, r.max.y as f64 * s),
            ),
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
        if let Some(v) = self.ui_state.layout_changed.take() {
            self.settings.dock_layout = Some(v);
            if let Err(e) = self.settings.save_to(&settings::config_path()) {
                tracing::warn!(
                    "{}",
                    tr!("Settings not saved: {error}", error = format!("{e:#}"))
                );
            }
        }
        if let Some(t) = self.gui.copied.take() {
            self.ui_copied(t);
        }
        self.apply_ui(actions);
        self.fit_if_needed();
    }

    /// Navigator texture, histogram and pointer colour for the panels that show.
    fn panel_data(&mut self, model: &mut ui::panels::Model) {
        // A smart object's size (computed on the CPU from its placement).
        if model.active_raw.as_ref().is_some_and(|r| r.smart)
            && let Some(s) = &mut self.session
        {
            model.active_bounds = s.engine.content_bounds(model.active);
        }
        let shown = self.ui_state.panels_shown;
        if shown.navigator {
            if self.nav.is_none() {
                let n = ui::navigator::NAV;
                let tex = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("navigator"),
                    size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.surface_config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                let id = self.gui.register_texture(&self.gpu.device, &view);
                self.nav = Some((tex, view, id));
                self.nav_drawn = None;
                self.needs_redraw = true;
            }
            model.panels.navigator = self.nav.as_ref().map(|n| n.2);
        }
        if !shown.info || !matches!(self.drag, Drag::None) {
            model.panels.histogram = self.info_hist.as_ref().map(|h| h.1.clone());
            model.panels.color = self.info_color.map(|c| c.1);
            return;
        }
        let Some(s) = &mut self.session else { return };
        if self.info_hist.as_ref().is_none_or(|h| h.0 != self.edit_count) && !self.loading {
            self.info_hist = Some((self.edit_count, s.engine.histogram(None)));
        }
        model.panels.histogram = self.info_hist.as_ref().map(|h| h.1.clone());
        if let Some((x, y)) = model.cursor_doc {
            let at = (x.floor() as i64, y.floor() as i64);
            if self.info_color.is_none_or(|c| c.0 != at) {
                let c = (x >= 0.0 && y >= 0.0)
                    .then(|| s.engine.sample_straight(at.0 as u32, at.1 as u32, mosaix_gpu::SampleFrom::Image).ok())
                    .flatten();
                self.info_color = c.map(|c| (at, c));
            }
        }
        model.panels.color = self.info_color.map(|c| c.1);
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
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure_surface();
                return;
            }
            _ => return,
        };
        let target = frame.texture.create_view(&Default::default());

        let liquifying = self.liquifying();
        if liquifying && let Drag::Liquify { last, pressure } = self.drag {
            let dt = self.liq_tick.elapsed().as_secs_f64();
            self.liq_tick = Instant::now();
            if self.liq_effective().continuous() {
                self.liquify_continuous(last, pressure, (dt * 60.0).clamp(0.1, 4.0));
                self.needs_redraw = true;
            }
        }
        let show_ring = (self.tool.paints() || liquifying)
            && !matches!(
                self.drag,
                Drag::Pan { .. } | Drag::Rotate { .. } | Drag::Zoom { .. }
            )
            && !self.space
            && self.bench.is_none();
        let ring_at = match self.drag {
            Drag::BrushHud { origin, .. } => Some(origin),
            _ => self.cursor.filter(|_| show_ring),
        };
        let ring_size = if liquifying {
            self.liq_size
        } else {
            self.brush.size
        };
        let overlay = Overlay {
            cursor: ring_at.map(|p| (p, ring_size * 0.5 * self.view.zoom as f32)),
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
                tracing::warn!(
                    "{}",
                    tr!(
                        "{n} tiles did not fit into GPU memory",
                        n = self.upload_failures
                    )
                );
                self.upload_failures = 0;
            }
            let result = match &self.ui_state.filter_preview {
                _ if s.engine.liquify_active() => s.engine.liquify_tick(),
                None if self.ui_state.vec_editor.fonts.preview.is_some() => {
                    let id = s.engine.doc().active;
                    s.engine.set_vector_preview(
                        self.ui_state
                            .vec_editor
                            .fonts
                            .preview
                            .as_ref()
                            .map(|v| (id, v)),
                    )
                }
                Some(ui::panels::FilterPreview::Pixels(f)) => s.engine.set_filter_preview(Some(f)),
                Some(ui::panels::FilterPreview::Smart(i, f)) => {
                    s.engine.set_smart_filter_preview(Some((*i, f)))
                }
                None => s
                    .engine
                    .set_transform_preview(preview.map(|f| (f, Interp::Bicubic))),
            };
            if let Err(e) = result {
                tracing::warn!("{}", tr!("Preview: {error}", error = e));
            }
            s.engine.render(&target, &self.view, &overlay);
            // The navigator: when it shows, after edits and while tiles still come in.
            if self.ui_state.panels_shown.navigator
                && let Some((_, view, _)) = &self.nav
                && self.nav_drawn.is_none_or(|(t, n)| n != self.edit_count || t.elapsed().as_millis() > 400)
            {
                let (w, h) = s.engine.size();
                s.engine.render_overview(view, &ui::navigator::nav_view((w, h)));
                self.nav_drawn = Some((Instant::now(), self.edit_count));
            }
        } else {
            clear(&self.gpu, &target);
        }
        let size = (self.surface_config.width, self.surface_config.height);
        self.gui
            .paint(&self.gpu.device, &self.gpu.queue, &target, size);

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

        self.needs_redraw = self
            .gui
            .repaint_at
            .is_some_and(|t| t <= Instant::now() + Duration::from_millis(4))
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
                self.ui_status = tr!(
                    "{tiles} tiles · {mb} MB GPU · {ms} ms",
                    tiles = e.layer_tiles_live(),
                    mb = format!("{:.0}", e.gpu_bytes() as f64 / 1e6),
                    ms = format!("{:.2}", self.frame_cpu_ms)
                );
            }
        }
    }

    /// Rubber bands for the drag in progress, in screen pixels.
    fn overlay_lines(&self) -> Vec<OverlayLine> {
        let v = &self.view;
        let rect = |a: DVec2, b: DVec2| {
            let (min, max) = (a.min(b), a.max(b));
            [min, DVec2::new(max.x, min.y), max, DVec2::new(min.x, max.y)]
                .map(|p| v.doc_to_screen(p))
                .to_vec()
        };
        match &self.drag {
            Drag::Marquee {
                start,
                end,
                ellipse: false,
                ..
            } => vec![OverlayLine {
                points: rect(*start, *end),
                closed: true,
            }],
            Drag::Marquee {
                start,
                end,
                ellipse: true,
                ..
            } => {
                let (c, r) = ((*start + *end) * 0.5, (*end - *start).abs() * 0.5);
                let points = (0..96)
                    .map(|i| {
                        let t = i as f64 / 96.0 * std::f64::consts::TAU;
                        v.doc_to_screen(c + DVec2::new(r.x * t.cos(), r.y * t.sin()))
                    })
                    .collect();
                vec![OverlayLine {
                    points,
                    closed: true,
                }]
            }
            Drag::Lasso { points, .. } => {
                vec![OverlayLine {
                    points: points.iter().map(|p| v.doc_to_screen(*p)).collect(),
                    closed: true,
                }]
            }
            Drag::Shape { start, end } => {
                let (a, b) = shape_rect(*start, *end, self.mods.shift, self.mods.alt);
                if matches!(self.tool, Tool::Shape { ellipse: true }) {
                    let (c, r) = ((a + b) * 0.5, (b - a) * 0.5);
                    let points = (0..96)
                        .map(|i| {
                            let t = i as f64 / 96.0 * std::f64::consts::TAU;
                            v.doc_to_screen(c + DVec2::new(r.x * t.cos(), r.y * t.sin()))
                        })
                        .collect();
                    vec![OverlayLine {
                        points,
                        closed: true,
                    }]
                } else {
                    vec![OverlayLine {
                        points: rect(a, b),
                        closed: true,
                    }]
                }
            }
            Drag::Gradient { start, end } => {
                vec![OverlayLine {
                    points: vec![v.doc_to_screen(*start), v.doc_to_screen(*end)],
                    closed: false,
                }]
            }
            _ => match &self.boxed {
                Some((_, b)) => b.lines(v, self.scale),
                None => self.clone_marker(),
            },
        }
    }

    /// Clone stamp: a cross where the source is sampled for the cursor position.
    fn clone_marker(&self) -> Vec<OverlayLine> {
        if !matches!(self.tool, Tool::Clone | Tool::Heal) || self.space {
            return Vec::new();
        }
        let at = match (
            self.clone_offset.filter(|_| self.clone_aligned),
            self.cursor,
            self.clone_source,
        ) {
            (Some([ox, oy]), Some(c), _) => {
                self.view.screen_to_doc(c) + DVec2::new(ox as f64, oy as f64)
            }
            (_, _, Some(src)) => src,
            _ => return Vec::new(),
        };
        let s = self.view.doc_to_screen(at);
        let k = 7.0 * self.scale;
        vec![
            OverlayLine {
                points: vec![s - DVec2::new(k, 0.0), s + DVec2::new(k, 0.0)],
                closed: false,
            },
            OverlayLine {
                points: vec![s - DVec2::new(0.0, k), s + DVec2::new(0.0, k)],
                closed: false,
            },
        ]
    }

    fn update_title(&self) {
        let Some(s) = &self.session else { return };
        let e = &s.engine;
        let (w, h) = e.size();
        let tool = if self.mask_edit {
            tr!("{tool} → Mask", tool = self.tool.name())
        } else {
            self.tool.name().to_string()
        };
        let layer = e
            .doc()
            .active_layer()
            .map(|l| {
                let (name, blend, opacity) =
                    (&l.name, l.blend.name(), format!("{:.0}", l.opacity * 100.0));
                if l.visible {
                    tr!(
                        "“{name}” {blend} {opacity} %",
                        name = name,
                        blend = blend,
                        opacity = opacity
                    )
                } else {
                    tr!(
                        "“{name}” {blend} {opacity} % (hidden)",
                        name = name,
                        blend = blend,
                        opacity = opacity
                    )
                }
            })
            .unwrap_or_default();
        let status = if self.quit_armed.is_some_and(|t| t.elapsed() < QUIT_CONFIRM) {
            format!(" – {}", tr("PRESS Ctrl+Q AGAIN TO DISCARD"))
        } else if self.save_job.is_some() || self.saving {
            format!(" – {}", tr("saving …"))
        } else if self.loading || !self.pending_uploads.is_empty() {
            format!(" – {}", tr("loading …"))
        } else if self.bg_running > 0 {
            format!(" – {}", tr("computing …"))
        } else {
            String::new()
        };
        self.window.set_title(tr!(
            "{modified}{name} – {w}×{h} – {zoom} % – {angle}° – {layer} – {tool} {size} px {opacity} % – {layer_tiles}+{composite_tiles} tiles ({mb} MB) – {ms} ms{status}",
            modified = if self.modified { "• " } else { "" },
            name = s.name,
            w = w,
            h = h,
            zoom = format!("{:.1}", self.view.zoom * 100.0 / self.scale),
            angle = format!("{:.0}", self.view.rotation.to_degrees()),
            layer = layer,
            tool = tool,
            size = format!("{:.0}", self.brush.size),
            opacity = format!("{:.0}", self.brush.opacity * 100.0),
            layer_tiles = e.layer_tiles_live(),
            composite_tiles = e.composite_tiles_live(),
            mb = format!("{:.0}", e.gpu_bytes() as f64 / 1e6),
            ms = format!("{:.2}", self.frame_cpu_ms),
            status = status,
        ));
    }
}

/// Benchmark camera path: fit → ~128× zoom and back, half a turn, some panning.
pub fn bench_view(base: &View, frame: u32, frames: u32) -> View {
    let t = frame as f64 / frames as f64 * std::f64::consts::TAU;
    let mut v = *base;
    v.zoom = base.zoom * 2f64.powf((1.0 - t.cos()) * 3.5);
    v.rotation = t * 0.5;
    v.center =
        base.center + DVec2::new(t.sin(), (2.0 * t).sin() * 0.5) * base.viewport / base.zoom * 0.25;
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
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.035,
                    g: 0.037,
                    b: 0.042,
                    a: 1.0,
                }),
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
    fn event(
        &self,
        _: &mut App,
        _: &WpViewporter,
        _: <WpViewporter as Proxy>::Event,
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
    }
}
impl Dispatch2<WpViewport, App> for tablet::Ignore {
    fn event(
        &self,
        _: &mut App,
        _: &WpViewport,
        _: <WpViewport as Proxy>::Event,
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
    }
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
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        factor: i32,
    ) {
        if self.viewport.is_none() {
            self.set_scale(factor as f64);
        }
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }

    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {
        self.frame_pending = false;
    }

    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
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

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &Window,
        configure: WindowConfigure,
        _: u32,
    ) {
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

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if cap == Capability::Keyboard && self.keyboard.is_none() {
            // Held keys repeat at the compositor's rate (the client repeats them).
            self.keyboard = self
                .seat_state
                .get_keyboard_with_repeat(
                    qh,
                    &seat,
                    None,
                    self.loop_handle.clone(),
                    Box::new(|app: &mut App, _, ev| app.repeated_key(ev)),
                )
                .or_else(|_| self.seat_state.get_keyboard(qh, &seat, None))
                .ok();
            if self.clipboard.device.is_none()
                && let Some(m) = &self.clipboard.manager
            {
                self.clipboard.device = Some(m.get_data_device(qh, &seat));
            }
        }
        if cap == Capability::Pointer && self.pointer.is_none() {
            let cursor_surface = self.compositor.create_surface(qh);
            self.pointer = self
                .seat_state
                .get_pointer_with_theme::<_, ()>(
                    qh,
                    &seat,
                    self.shm.wl_shm(),
                    cursor_surface,
                    ThemeSpec::default(),
                )
                .ok();
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if cap == Capability::Keyboard
            && let Some(k) = self.keyboard.take()
        {
            k.release();
        }
        if cap == Capability::Pointer
            && let Some(p) = self.pointer.take()
        {
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

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
        // Keys released while unfocused never reach us – modifiers included
        // (an Alt held while switching windows would otherwise stay "down";
        // the compositor sends the real state again on focus).
        self.space = false;
        self.r_key = false;
        self.mods = Modifiers::default();
        self.gui.set_modifiers(false, false, false);
    }

    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        serial: u32,
        ev: KeyEvent,
    ) {
        self.clipboard.serial = serial;
        self.key(&ev, true);
    }

    fn repeat_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        ev: KeyEvent,
    ) {
        self.repeated_key(ev);
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        ev: KeyEvent,
    ) {
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
    fn pointer_frame(
        &mut self,
        conn: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for ev in events {
            if &ev.surface != self.window.wl_surface() {
                continue;
            }
            let pos = self.to_physical(ev.position);
            let scale = self.scale;
            self.needs_redraw = true;
            match &ev.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
                    self.gui.pointer_moved(pos, scale)
                }
                PointerEventKind::Leave { .. } => self.gui.pointer_gone(),
                _ => {}
            }
            let canvas_busy = !matches!(self.drag, Drag::None);
            let over_ui = self.over_ui(pos);
            match &ev.kind {
                PointerEventKind::Press { button, .. }
                | PointerEventKind::Release { button, .. }
                    if !matches!(*button, BTN_LEFT | BTN_MIDDLE)
                        && (over_ui || self.ui_pointer) =>
                {
                    continue; // other buttons do nothing on the interface
                }
                PointerEventKind::Press { button, .. }
                    if !self.tablet_contact && (over_ui || self.ui_pointer) && !canvas_busy =>
                {
                    self.ui_pointer = true;
                    self.gui
                        .pointer_button(pos, scale, *button == BTN_MIDDLE, true);
                    continue;
                }
                PointerEventKind::Release { button, .. } if self.ui_pointer => {
                    self.ui_pointer = false;
                    self.gui
                        .pointer_button(pos, scale, *button == BTN_MIDDLE, false);
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
                PointerEventKind::Press { button, .. }
                    if !self.tablet_contact && matches!(*button, BTN_LEFT | BTN_MIDDLE) =>
                {
                    self.gui
                        .pointer_button(pos, scale, *button == BTN_MIDDLE, true);
                }
                PointerEventKind::Release { button, .. }
                    if !self.tablet_contact && matches!(*button, BTN_LEFT | BTN_MIDDLE) =>
                {
                    self.gui
                        .pointer_button(pos, scale, *button == BTN_MIDDLE, false);
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
                PointerEventKind::Axis { vertical, horizontal, .. } => {
                    let step = |a: &smithay_client_toolkit::seat::pointer::AxisScroll| {
                        if a.value120 != 0 {
                            a.value120 as f64 / 120.0
                        } else if a.discrete != 0 {
                            a.discrete as f64
                        } else {
                            a.absolute / 15.0
                        }
                    };
                    let (steps, hsteps) = (step(vertical), step(horizontal));
                    if steps == 0.0 && hsteps == 0.0 {
                        continue;
                    }
                    // Like Photoshop: the wheel scrolls, Alt+wheel zooms.
                    let line = 60.0 * self.scale;
                    if self.mods.ctrl {
                        self.brush.size =
                            (self.brush.size * 1.1f32.powf(-steps as f32)).clamp(1.0, 5000.0);
                    } else if self.mods.alt && self.mods.shift {
                        self.view.rotate_at(pos, (steps * 7.5).to_radians());
                    } else if self.mods.alt {
                        self.view.zoom_at(pos, 1.2f64.powf(-steps));
                        self.limit_zoom_out(pos);
                    } else if self.mods.shift {
                        self.view.pan(DVec2::new(-(steps + hsteps) * line, 0.0));
                    } else {
                        self.view.pan(DVec2::new(-hsteps * line, -steps * line));
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

/// The agent command a UI action stands for (when recording an action).
fn recorded_action(a: &Action) -> Option<serde_json::Value> {
    use serde_json::json;
    Some(match a {
        Action::Agent(v) => v.clone(),
        Action::ModifySelection { op, amount } => json!({ "cmd": "select_modify", "op": op, "amount": amount }),
        Action::ApplyFilter(f) => json!({ "cmd": "filter_apply", "filter": f }),
        Action::AddSmartFilter(f) => json!({ "cmd": "filter_apply", "filter": f, "smart": true }),
        Action::AddAdjustment(adj) => json!({ "cmd": "adjustment_add", "adjustment": adj }),
        Action::SetAdjustment { adj, done: true } => json!({ "cmd": "adjustment_set", "adjustment": adj }),
        Action::SetEffects { effects, done: true } => json!({ "cmd": "layer_effects_set", "effects": effects }),
        Action::SetRaw { d, done: true } => json!({
            "cmd": "raw_set", "white_balance": d.white_balance, "exposure": d.exposure, "highlights": d.highlights,
            "noise": d.noise, "lens": d.lens, "tone": d.tone,
        }),
        Action::LayerOpacity { value, done: true, fill } => {
            json!({ "cmd": "layer_set", (if *fill { "fill_opacity" } else { "opacity" }): value })
        }
        Action::LayerBlend(b) => json!({ "cmd": "layer_set", "blend": b.key() }),
        Action::PassThrough(on) => json!({ "cmd": "layer_set", "pass_through": on }),
        Action::DeleteLayer => json!({ "cmd": "layer_delete" }),
        Action::CanvasSize { w, h, anchor } => {
            let x = if anchor.0 < 0.25 { "left" } else if anchor.0 > 0.75 { "right" } else { "" };
            let y = if anchor.1 < 0.25 { "top" } else if anchor.1 > 0.75 { "bottom" } else { "" };
            let name = match (y, x) {
                ("", "") => "center".to_string(),
                ("", x) => x.to_string(),
                (y, "") => y.to_string(),
                (y, x) => format!("{y}-{x}"),
            };
            json!({ "cmd": "canvas_size", "width": w, "height": h, "anchor": name })
        }
        Action::ImageSize { w, h } => json!({ "cmd": "image_resize", "width": w, "height": h }),
        Action::Upscale(k) => json!({ "cmd": "image_upscale", "factor": k }),
        Action::HistoryJump(_) => {
            tracing::warn!("{}", tr("Jumps in the history are not recorded"));
            return None;
        }
        _ => return None,
    })
}

/// The agent command a finished drag stands for (selections and moves).
fn recorded_drag(drag: &Drag, zoom: f64, move_delta: DVec2) -> Option<serde_json::Value> {
    use serde_json::json;
    let mode = |op: &CombineOp| match op {
        CombineOp::Replace => "replace",
        CombineOp::Add => "add",
        CombineOp::Subtract => "subtract",
        CombineOp::Intersect => "intersect",
    };
    Some(match drag {
        Drag::Marquee { start, end, op, ellipse } => {
            let (min, max) = (start.min(*end), start.max(*end));
            let size = max - min;
            if size.min_element() * zoom < 2.0 {
                return (*op == CombineOp::Replace).then(|| json!({ "cmd": "select_none" }));
            }
            json!({
                "cmd": if *ellipse { "select_ellipse" } else { "select_rect" },
                "x": min.x, "y": min.y, "width": size.x, "height": size.y, "mode": mode(op),
            })
        }
        Drag::Lasso { points, op } if points.len() >= 3 => json!({
            "cmd": "select_polygon",
            "points": points.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
            "mode": mode(op),
        }),
        Drag::Move { .. } if move_delta != DVec2::ZERO => {
            json!({ "cmd": "layer_translate", "dx": move_delta.x.round() as i64, "dy": move_delta.y.round() as i64 })
        }
        _ => return None,
    })
}

#[cfg(test)]
mod record_tests {
    #[test]
    fn snapping_picks_the_nearest_line_in_reach() {
        let lines = [0.0, 500.0, 1000.0, 333.0];
        assert_eq!(App::snap_axis(&lines, None, 330.0, 8.0), Some(333.0));
        assert_eq!(App::snap_axis(&lines, None, 320.0, 8.0), None);
        // The grid counts too, the closer one wins.
        assert_eq!(App::snap_axis(&lines, Some(25.0), 322.0, 8.0), Some(325.0));
        assert_eq!(App::snap_axis(&lines, Some(25.0), 332.0, 8.0), Some(333.0));
    }

    use super::*;

    /// Every recorded step must be a valid agent command.
    #[test]
    fn recorded_steps_parse() {
        let actions = vec![
            Action::ApplyFilter(mosaix_core::Filter::GaussianBlur { radius: 3.0 }),
            Action::AddSmartFilter(mosaix_core::Filter::HighPass { radius: 5.0 }),
            Action::AddAdjustment(mosaix_core::Adjustment::defaults().remove(0)),
            Action::SetAdjustment { adj: mosaix_core::Adjustment::defaults().remove(1), done: true },
            Action::SetEffects { effects: vec![mosaix_core::style::Effect::new(mosaix_core::style::EffectKind::defaults().remove(0))], done: true },
            Action::SetRaw { d: Default::default(), done: true },
            Action::LayerOpacity { value: 0.5, done: true, fill: true },
            Action::LayerBlend(mosaix_core::BlendMode::Multiply),
            Action::PassThrough(true),
            Action::DeleteLayer,
            Action::CanvasSize { w: 100, h: 80, anchor: (0.0, 1.0) },
            Action::ImageSize { w: 50, h: 40 },
        ];
        for a in &actions {
            let v = recorded_action(a).expect("recorded");
            crate::agent::command::Command::parse(v.clone()).unwrap_or_else(|e| panic!("{v}: {e}"));
        }
        assert_eq!(recorded_action(&Action::CanvasSize { w: 1, h: 1, anchor: (0.0, 1.0) }).unwrap()["anchor"], "bottom-left");
        let drags = [
            Drag::Marquee { start: DVec2::new(1.0, 2.0), end: DVec2::new(50.0, 40.0), op: CombineOp::Add, ellipse: true },
            Drag::Lasso { points: vec![DVec2::ZERO, DVec2::new(5.0, 0.0), DVec2::new(0.0, 5.0)], op: CombineOp::Subtract },
            Drag::Move { start: DVec2::ZERO, end: DVec2::new(3.0, 4.0) },
        ];
        for d in &drags {
            let v = recorded_drag(d, 1.0, DVec2::new(3.0, 4.0)).expect("recorded");
            crate::agent::command::Command::parse(v.clone()).unwrap_or_else(|e| panic!("{v}: {e}"));
        }
        for v in [
            serde_json::json!({ "cmd": "fill", "color": "#ff0080", "opacity": 1.0, "target": "pixels" }),
            serde_json::json!({ "cmd": "select_feather", "radius": 10.0 }),
            serde_json::json!({ "cmd": "layer_mask_apply" }),
            serde_json::json!({ "cmd": "layer_rasterize" }),
            serde_json::json!({ "cmd": "layer_ungroup" }),
        ] {
            crate::agent::command::Command::parse(v.clone()).unwrap_or_else(|e| panic!("{v}: {e}"));
        }
    }
}
