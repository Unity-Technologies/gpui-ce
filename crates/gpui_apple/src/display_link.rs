//! `CAMetalDisplayLink` frame pacing, ported from Zed #65472.
//!
//! A `CAMetalDisplayLink` hands each frame its drawable early enough to reach
//! the display on time. `-[CAMetalLayer nextDrawable]` instead waits for the
//! compositor to release one, which on macOS 26 base M-series Macs blocks
//! about 33 ms and caps animation at 30 fps.

use foreign_types::ForeignTypeRef;
use metal::{CAMetalDrawable as MetalCAMetalDrawable, MetalDrawable, MetalLayerRef};
use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, Message, define_class, msg_send,
    rc::Retained,
    runtime::{AnyClass, AnyObject, ProtocolObject},
};
use objc2_foundation::{NSObject, NSObjectProtocol, NSRunLoop, NSRunLoopCommonModes};
use objc2_quartz_core::{
    CAMetalDisplayLink, CAMetalDisplayLinkDelegate, CAMetalDisplayLinkUpdate, CAMetalLayer,
};

struct MetalDisplayLinkDelegateIvars {
    callback: Box<dyn Fn(MetalDrawable)>,
}

define_class!(
    // SAFETY: `NSObject` has no subclassing requirements.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = MetalDisplayLinkDelegateIvars]
    struct MetalDisplayLinkDelegate;

    unsafe impl NSObjectProtocol for MetalDisplayLinkDelegate {}

    unsafe impl CAMetalDisplayLinkDelegate for MetalDisplayLinkDelegate {
        #[unsafe(method(metalDisplayLink:needsUpdate:))]
        fn needs_update(&self, _link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
            // SAFETY: `drawable` is a non-null `id<CAMetalDrawable>` property.
            let drawable: Option<Retained<AnyObject>> = unsafe { msg_send![update, drawable] };
            let Some(drawable) = drawable else {
                return;
            };
            // SAFETY: `into_raw` hands over the +1 retain that `MetalDrawable` releases on drop.
            let drawable = unsafe {
                <MetalDrawable as foreign_types::ForeignType>::from_ptr(
                    Retained::into_raw(drawable).cast::<MetalCAMetalDrawable>(),
                )
            };
            // The callback may destroy its frame source, releasing our owner's retain.
            let delegate = self.retain();
            (delegate.ivars().callback)(drawable);
        }
    }
);

impl MetalDisplayLinkDelegate {
    fn new(
        main_thread: MainThreadMarker,
        callback: impl Fn(MetalDrawable) + 'static,
    ) -> Retained<Self> {
        let this = Self::alloc(main_thread).set_ivars(MetalDisplayLinkDelegateIvars {
            callback: Box::new(callback),
        });
        // SAFETY: `NSObject`'s `init` is its designated initializer.
        unsafe { msg_send![super(this), init] }
    }
}

/// A main-run-loop source of Metal drawables on macOS 14+.
///
/// Callbacks run on the run loop the link is registered with, here the main
/// one, because GPUI renders on the foreground thread:
/// <https://developer.apple.com/documentation/quartzcore/cametaldisplaylink>.
pub struct MetalFrameSource {
    link: Retained<CAMetalDisplayLink>,
    // The link holds its delegate weakly.
    _delegate: Retained<MetalDisplayLinkDelegate>,
    _main_thread: MainThreadMarker,
}

impl MetalFrameSource {
    /// Whether this system has `CAMetalDisplayLink` (macOS 14+).
    pub fn is_available() -> bool {
        AnyClass::get(c"CAMetalDisplayLink").is_some()
    }

    /// Starts delivering one drawable per display update for `layer`, or
    /// returns `None` where `CAMetalDisplayLink` is unavailable.
    pub fn new(
        layer: &MetalLayerRef,
        main_thread: MainThreadMarker,
        callback: impl Fn(MetalDrawable) + 'static,
    ) -> Option<Self> {
        if !Self::is_available() {
            return None;
        }
        // SAFETY: metal-rs and objc2 both wrap the same `CAMetalLayer` object.
        let layer = unsafe { &*(layer.as_ptr() as *const CAMetalLayer) };
        let delegate = MetalDisplayLinkDelegate::new(main_thread, callback);
        let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), layer);
        link.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
        // The default frame rate range follows the display and system power policy:
        // https://developer.apple.com/documentation/quartzcore/cametaldisplaylink/preferredframeraterange
        // SAFETY: the retained marker confines the source to the main thread,
        // and the main run loop outlives the source.
        unsafe { link.addToRunLoop_forMode(&NSRunLoop::mainRunLoop(), NSRunLoopCommonModes) };
        Some(Self {
            link,
            _delegate: delegate,
            _main_thread: main_thread,
        })
    }

    /// A paused link delivers no updates and acquires no drawables.
    pub fn set_paused(&self, paused: bool) {
        self.link.setPaused(paused);
    }
}

impl Drop for MetalFrameSource {
    fn drop(&mut self) {
        self.link.invalidate();
    }
}
