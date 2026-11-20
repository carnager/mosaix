//! The on-screen interface: egui panels drawn over the canvas.
//!
//! There is no winit, so Wayland input is translated into egui events here.
//! Each frame the panels are built from a [`panels::Model`] snapshot of the
//! app state and return [`panels::Action`]s, which the app applies after the
//! UI pass (no borrowing of the engine from inside the UI code).

pub mod files;
pub mod icons;
pub mod log;
pub mod palette;
pub mod panels;
pub mod snapshot;

use std::time::{Duration, Instant};

use egui::{Modifiers, PointerButton, Pos2, Vec2};
use glam::DVec2;

pub struct Gui {
    pub ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    events: Vec<egui::Event>,
    modifiers: Modifiers,
    start: Instant,
    /// When egui wants the next frame (animations, tooltips).
    pub repaint_at: Option<Instant>,
    prims: Vec<egui::ClippedPrimitive>,
    ppp: f32,
    textures: egui::TexturesDelta,
    /// Size and scale the last pass was laid out for.
    last_layout: Option<((u32, u32), u64)>,
}

fn pos(phys: DVec2, ppp: f64) -> Pos2 {
    Pos2::new((phys.x / ppp) as f32, (phys.y / ppp) as f32)
}

impl Gui {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();
        panels::style(&ctx);
        let renderer = egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default());
        Self {
            ctx,
            renderer,
            events: Vec::new(),
            modifiers: Modifiers::default(),
            start: Instant::now(),
            repaint_at: Some(Instant::now()),
            prims: Vec::new(),
            ppp: 1.0,
            textures: Default::default(),
            last_layout: None,
        }
    }

    // ---- input (physical px) --------------------------------------------------

    pub fn set_modifiers(&mut self, ctrl: bool, shift: bool, alt: bool) {
        self.modifiers = Modifiers { alt, ctrl, shift, mac_cmd: false, command: ctrl };
        self.events.push(egui::Event::ModifiersChanged(self.modifiers));
    }

    pub fn pointer_moved(&mut self, phys: DVec2, ppp: f64) {
        self.events.push(egui::Event::PointerMoved(pos(phys, ppp)));
    }

    pub fn pointer_button(&mut self, phys: DVec2, ppp: f64, middle: bool, pressed: bool) {
        let button = if middle { PointerButton::Middle } else { PointerButton::Primary };
        self.events.push(egui::Event::PointerButton { pos: pos(phys, ppp), button, pressed, modifiers: self.modifiers });
    }

    pub fn pointer_gone(&mut self) {
        self.events.push(egui::Event::PointerGone);
    }

    /// Mouse wheel in notches (positive = towards the user / down).
    pub fn scroll(&mut self, steps: f64) {
        self.events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: Vec2::new(0.0, -steps as f32),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Wheel,
            modifiers: self.modifiers,
        });
    }

    pub fn key(&mut self, key: egui::Key, pressed: bool) {
        self.events.push(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: self.modifiers });
    }

    pub fn text(&mut self, text: &str) {
        if !text.is_empty() && !self.modifiers.ctrl && text.chars().all(|c| !c.is_control()) {
            self.events.push(egui::Event::Text(text.to_string()));
        }
    }

    /// A text field has focus: hotkeys stay out.
    pub fn wants_keyboard(&self) -> bool {
        self.ctx.egui_wants_keyboard_input()
    }

    /// Makes a wgpu texture (premultiplied linear RGBA) usable in egui.
    pub fn register_texture(&mut self, device: &wgpu::Device, view: &wgpu::TextureView) -> egui::TextureId {
        self.renderer.register_native_texture(device, view, wgpu::FilterMode::Linear)
    }

    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    // ---- frame ------------------------------------------------------------------

    /// Whether a new UI pass is needed (input, animation, resize, or
    /// `model_changed`); otherwise the last frame's shapes are reused.
    pub fn needs_pass(&self, size: (u32, u32), ppp: f64, model_changed: bool) -> bool {
        model_changed
            || !self.events.is_empty()
            || self.repaint_at.is_some_and(|t| t <= Instant::now())
            || self.last_layout != Some((size, ppp.to_bits()))
    }

    /// Runs the UI pass; `build` adds the panels.
    pub fn run(&mut self, size: (u32, u32), ppp: f64, build: impl FnMut(&egui::Context)) {
        self.last_layout = Some((size, ppp.to_bits()));
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                Vec2::new((size.0 as f64 / ppp) as f32, (size.1 as f64 / ppp) as f32),
            )),
            time: Some(self.start.elapsed().as_secs_f64()),
            events: std::mem::take(&mut self.events),
            focused: true,
            max_texture_side: Some(8192),
            ..Default::default()
        };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp as f32);
        let mut build = build;
        let out = self.ctx.run_pass(raw, |ctx| build(ctx));
        let delay = out.viewport_output.get(&egui::ViewportId::ROOT).map_or(Duration::MAX, |v| v.repaint_delay);
        self.repaint_at = (delay < Duration::from_secs(3600)).then(|| Instant::now() + delay);
        self.prims = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        self.ppp = out.pixels_per_point;
        self.textures.append(out.textures_delta);
    }

    /// Draws the last UI pass on top of `target`.
    pub fn paint(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::TextureView, size: (u32, u32)) {
        let textures = std::mem::take(&mut self.textures);
        let (prims, ppp) = (&self.prims, self.ppp);
        if prims.is_empty() {
            return;
        }
        for (id, deltas) in &textures.set {
            for delta in deltas {
                self.renderer.update_texture(device, queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.0, size.1], pixels_per_point: ppp };
        let mut enc = device.create_command_encoder(&Default::default());
        let extra = self.renderer.update_buffers(device, queue, &mut enc, prims, &screen);
        {
            let rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            self.renderer.render(&mut rp.forget_lifetime(), prims, &screen);
        }
        queue.submit(extra.into_iter().chain([enc.finish()]));
        for id in &textures.free {
            self.renderer.free_texture(id);
        }
    }
}

/// egui key for a keysym, for text fields and widget navigation.
pub fn egui_key(sym: smithay_client_toolkit::seat::keyboard::Keysym) -> Option<egui::Key> {
    use smithay_client_toolkit::seat::keyboard::Keysym as K;
    use egui::Key;
    Some(match sym {
        K::BackSpace => Key::Backspace,
        K::Delete => Key::Delete,
        K::Return | K::KP_Enter => Key::Enter,
        K::Escape => Key::Escape,
        K::Tab => Key::Tab,
        K::Left => Key::ArrowLeft,
        K::Right => Key::ArrowRight,
        K::Up => Key::ArrowUp,
        K::Down => Key::ArrowDown,
        K::Home => Key::Home,
        K::End => Key::End,
        K::a | K::A => Key::A,
        K::c | K::C => Key::C,
        K::v | K::V => Key::V,
        K::x | K::X => Key::X,
        K::z | K::Z => Key::Z,
        _ => return None,
    })
}
