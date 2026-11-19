//! Graphics tablet input via the Wayland `tablet-v2` protocol (pressure, tilt,
//! stylus buttons, eraser end). SCTK has no support for it, so it is wired
//! up here directly. Pads (express keys, rings, strips) are bound but ignored.

use std::sync::{Arc, Mutex};

use smithay_client_toolkit::dispatch2::Dispatch2;
use wayland_client::{Connection, QueueHandle, WEnum, backend::ObjectData};
use wayland_protocols::wp::tablet::zv2::client::{
    zwp_tablet_manager_v2::ZwpTabletManagerV2,
    zwp_tablet_pad_dial_v2::ZwpTabletPadDialV2,
    zwp_tablet_pad_group_v2::{self, ZwpTabletPadGroupV2},
    zwp_tablet_pad_ring_v2::ZwpTabletPadRingV2,
    zwp_tablet_pad_strip_v2::ZwpTabletPadStripV2,
    zwp_tablet_pad_v2::{self, ZwpTabletPadV2},
    zwp_tablet_seat_v2::{self, ZwpTabletSeatV2},
    zwp_tablet_tool_v2::{self, ZwpTabletToolV2},
    zwp_tablet_v2::ZwpTabletV2,
};

use crate::app::App;

/// Linux input event codes for the stylus side buttons.
pub const BTN_STYLUS: u32 = 0x14b;
pub const BTN_STYLUS2: u32 = 0x14c;

/// User data for protocol objects whose events we do not care about.
pub struct Ignore;

macro_rules! ignore_events {
    ($($iface:ty),*) => {$(
        impl Dispatch2<$iface, App> for Ignore {
            fn event(&self, _: &mut App, _: &$iface, _: <$iface as wayland_client::Proxy>::Event,
                     _: &Connection, _: &QueueHandle<App>) {}
        }
    )*};
}
ignore_events!(ZwpTabletManagerV2, ZwpTabletV2, ZwpTabletPadRingV2, ZwpTabletPadStripV2, ZwpTabletPadDialV2);

impl Dispatch2<ZwpTabletPadV2, App> for Ignore {
    fn event(&self, _: &mut App, _: &ZwpTabletPadV2, _: zwp_tablet_pad_v2::Event, _: &Connection, _: &QueueHandle<App>) {}

    fn event_created_child(opcode: u16, qh: &QueueHandle<App>) -> Arc<dyn ObjectData> {
        match opcode {
            zwp_tablet_pad_v2::EVT_GROUP_OPCODE => qh.make_data::<ZwpTabletPadGroupV2, _>(Ignore),
            _ => panic!("unexpected child of zwp_tablet_pad_v2: opcode {opcode}"),
        }
    }
}

impl Dispatch2<ZwpTabletPadGroupV2, App> for Ignore {
    fn event(&self, _: &mut App, _: &ZwpTabletPadGroupV2, _: zwp_tablet_pad_group_v2::Event, _: &Connection, _: &QueueHandle<App>) {}

    fn event_created_child(opcode: u16, qh: &QueueHandle<App>) -> Arc<dyn ObjectData> {
        match opcode {
            zwp_tablet_pad_group_v2::EVT_RING_OPCODE => qh.make_data::<ZwpTabletPadRingV2, _>(Ignore),
            zwp_tablet_pad_group_v2::EVT_STRIP_OPCODE => qh.make_data::<ZwpTabletPadStripV2, _>(Ignore),
            zwp_tablet_pad_group_v2::EVT_DIAL_OPCODE => qh.make_data::<ZwpTabletPadDialV2, _>(Ignore),
            _ => panic!("unexpected child of zwp_tablet_pad_group_v2: opcode {opcode}"),
        }
    }
}

/// User data of a tablet seat: creates the child objects.
pub struct SeatData;

impl Dispatch2<ZwpTabletSeatV2, App> for SeatData {
    fn event(&self, _: &mut App, _: &ZwpTabletSeatV2, _: zwp_tablet_seat_v2::Event, _: &Connection, _: &QueueHandle<App>) {}

    fn event_created_child(opcode: u16, qh: &QueueHandle<App>) -> Arc<dyn ObjectData> {
        match opcode {
            zwp_tablet_seat_v2::EVT_TABLET_ADDED_OPCODE => qh.make_data::<ZwpTabletV2, _>(Ignore),
            zwp_tablet_seat_v2::EVT_TOOL_ADDED_OPCODE => {
                qh.make_data::<ZwpTabletToolV2, _>(ToolData::default())
            }
            zwp_tablet_seat_v2::EVT_PAD_ADDED_OPCODE => qh.make_data::<ZwpTabletPadV2, _>(Ignore),
            _ => panic!("unexpected child of zwp_tablet_seat_v2: opcode {opcode}"),
        }
    }
}

/// Everything that happened to one tool within one `frame` event.
#[derive(Debug, Clone, Default)]
pub struct TabletFrame {
    /// Surface-local logical coordinates.
    pub pos: (f64, f64),
    /// 0..=1
    pub pressure: f32,
    /// Tilt in degrees (x, y).
    pub tilt: (f64, f64),
    pub is_eraser: bool,
    /// `Some(true)` = entered proximity, `Some(false)` = left.
    pub proximity: Option<bool>,
    /// `Some(true)` = tip down, `Some(false)` = tip up.
    pub contact: Option<bool>,
    pub buttons: Vec<(u32, bool)>,
}

#[derive(Default)]
pub struct ToolData(Mutex<TabletFrame>);

impl Dispatch2<ZwpTabletToolV2, App> for ToolData {
    fn event(
        &self,
        app: &mut App,
        tool: &ZwpTabletToolV2,
        event: zwp_tablet_tool_v2::Event,
        _: &Connection,
        _: &QueueHandle<App>,
    ) {
        use zwp_tablet_tool_v2::Event;
        let mut s = self.0.lock().unwrap();
        match event {
            Event::Type { tool_type } => {
                s.is_eraser = matches!(tool_type, WEnum::Value(zwp_tablet_tool_v2::Type::Eraser));
            }
            Event::ProximityIn { serial, .. } => {
                s.proximity = Some(true);
                // Hide the system cursor; the canvas draws the brush outline.
                tool.set_cursor(serial, None, 0, 0);
            }
            Event::ProximityOut => s.proximity = Some(false),
            Event::Down { .. } => s.contact = Some(true),
            Event::Up => s.contact = Some(false),
            Event::Motion { x, y } => s.pos = (x, y),
            Event::Pressure { pressure } => s.pressure = pressure as f32 / 65535.0,
            Event::Tilt { tilt_x, tilt_y } => s.tilt = (tilt_x, tilt_y),
            Event::Button { button, state, .. } => {
                let pressed =
                    matches!(state, WEnum::Value(zwp_tablet_tool_v2::ButtonState::Pressed));
                s.buttons.push((button, pressed));
            }
            Event::Frame { .. } => {
                let frame = s.clone();
                s.proximity = None;
                s.contact = None;
                s.buttons.clear();
                drop(s);
                app.on_tablet_frame(frame);
            }
            Event::Removed => tool.destroy(),
            _ => {}
        }
    }
}
