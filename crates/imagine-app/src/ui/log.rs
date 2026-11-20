//! Collects the app's own log messages (info and above) for the status bar,
//! so feedback like "nothing deleted" is visible without a terminal.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

#[derive(Debug, Clone)]
pub struct Message {
    pub at: Instant,
    pub level: Level,
    pub text: String,
}

static MESSAGES: Mutex<VecDeque<Message>> = Mutex::new(VecDeque::new());
static COUNT: AtomicUsize = AtomicUsize::new(0);
const KEEP: usize = 200;

/// Tracing layer feeding [`recent`].
pub struct UiLog;

struct Text(String);

impl Visit for Text {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0 = value.to_string();
        }
    }
}

impl<S: Subscriber> Layer<S> for UiLog {
    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        let meta = event.metadata();
        if *meta.level() > Level::INFO || !meta.target().starts_with("imagine") {
            return;
        }
        let mut t = Text(String::new());
        event.record(&mut t);
        if t.0.is_empty() {
            return;
        }
        let mut m = MESSAGES.lock().unwrap_or_else(|e| e.into_inner());
        m.push_back(Message {
            at: Instant::now(),
            level: *meta.level(),
            text: t.0,
        });
        while m.len() > KEEP {
            m.pop_front();
        }
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// The newest messages, oldest first.
pub fn recent(n: usize) -> Vec<Message> {
    let m = MESSAGES.lock().unwrap_or_else(|e| e.into_inner());
    m.iter().skip(m.len().saturating_sub(n)).cloned().collect()
}

/// Number of messages logged so far (changes → the UI should redraw).
pub fn generation() -> usize {
    COUNT.load(Ordering::Relaxed)
}
