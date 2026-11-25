#![cfg(target_os = "macos")]
//! macOS platform implementation for GPUI.
//!
//! macOS screens have a y axis that goes up from the bottom of the screen and
//! an origin at the bottom left of the main display.

mod dispatcher;
mod display;
mod display_link;
mod events;
mod haptic_feedback;
mod keyboard;
mod overlay_spike;
mod pasteboard;
mod system_notifications;
mod text_system;

#[cfg(feature = "screen-capture")]
mod screen_capture;

#[cfg(not(feature = "wgpu"))]
use gpui_apple::metal_renderer as renderer;
#[cfg(feature = "wgpu")]
mod wgpu_renderer;
#[cfg(feature = "wgpu")]
use wgpu_renderer as renderer;

mod platform;
mod window;
mod window_appearance;

pub(crate) use dispatcher::*;
pub(crate) use display::*;
pub(crate) use display_link::*;
pub(crate) use keyboard::*;
pub(crate) use platform::*;
pub(crate) use text_system::*;
pub(crate) use window::*;

pub use platform::MacPlatform;

/// Returns the main-thread marker, checking the current thread once per thread.
fn main_thread_marker() -> Option<objc2::MainThreadMarker> {
    thread_local! {
        static MAIN_THREAD_MARKER: Option<objc2::MainThreadMarker> =
            objc2::MainThreadMarker::new();
    }
    MAIN_THREAD_MARKER.with(|marker| *marker)
}

#[cfg(test)]
mod tests {
    use super::main_thread_marker;

    #[test]
    fn main_thread_marker_rejects_background_threads() {
        std::thread::spawn(|| {
            assert_eq!(main_thread_marker(), None);
            assert_eq!(main_thread_marker(), None);
        })
        .join()
        .expect("background thread panicked");
    }
}
