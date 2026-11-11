//! HUB-8754 spike, never merge-ready: draw deferred scene content (menus, tooltips) into a second
//! transparent CAMetalLayer hosted by a sibling NSView stacked ABOVE the GPUIView, so it sits above
//! native child views (WKWebView) that the Hub parents to the GPUIView. Opt in with
//! `HUB_OVERLAY_SPIKE=1`.
#![cfg(not(feature = "wgpu"))]

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Once};

use foreign_types::{ForeignType, ForeignTypeRef};
use gpui::{Scene, ScaledPixels};
use objc2::{
    class, msg_send,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel},
    sel,
};
use objc2_foundation::{NSPoint, NSRect, NSSize};

use crate::renderer;

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("HUB_OVERLAY_SPIKE").is_some_and(|v| v != "0"))
}

type Rects = Arc<Mutex<Vec<[f64; 4]>>>;

pub(crate) struct Overlay {
    view: Retained<AnyObject>,
    layer: metal::MetalLayer,
    rects: Rects,
}

const TARGET: &CStr = c"target";
const RECTS: &CStr = c"rects";

unsafe fn ivar_ptr(this: &AnyObject, name: &CStr) -> *mut c_void {
    let ivar = this.class().instance_variable(name).unwrap();
    unsafe { *ivar.load::<*mut c_void>(this) }
}

extern "C" fn hit_test(this: &AnyObject, _: Sel, point: NSPoint) -> *mut AnyObject {
    unsafe {
        let rects = &*(ivar_ptr(this, RECTS) as *const Mutex<Vec<[f64; 4]>>);
        let frame: NSRect = msg_send![this, frame];
        let x = point.x - frame.origin.x;
        // GPUIView is not flipped: scene y runs down from the top.
        let y = frame.size.height - (point.y - frame.origin.y);
        let hit = (gpui::overlay_modal()
            && x >= 0.0
            && y >= 0.0
            && x < frame.size.width
            && y < frame.size.height)
            || rects
            .lock()
            .map(|r| {
                r.iter()
                    .any(|[rx, ry, rw, rh]| x >= *rx && x < rx + rw && y >= *ry && y < ry + rh)
            })
            .unwrap_or(false);
        if hit { this as *const AnyObject as *mut AnyObject } else { std::ptr::null_mut() }
    }
}

macro_rules! forward {
    ($($name:ident => $sel:ident),* $(,)?) => {$(
        extern "C" fn $name(this: &AnyObject, _: Sel, event: *mut AnyObject) {
            unsafe {
                let target = ivar_ptr(this, TARGET) as *mut AnyObject;
                if !target.is_null() {
                    let _: () = msg_send![&*target, $sel: event];
                }
            }
        }
    )*};
}
forward!(
    f_md => mouseDown, f_mu => mouseUp, f_rmd => rightMouseDown, f_rmu => rightMouseUp,
    f_omd => otherMouseDown, f_omu => otherMouseUp, f_mm => mouseMoved, f_md2 => mouseDragged,
    f_rmd2 => rightMouseDragged, f_omd2 => otherMouseDragged, f_sw => scrollWheel,
);

fn overlay_class() -> &'static AnyClass {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        let mut d = ClassBuilder::new(c"GPUIOverlaySpikeView", class!(NSView)).unwrap();
        d.add_ivar::<*mut c_void>(TARGET);
        d.add_ivar::<*mut c_void>(RECTS);
        d.add_method(sel!(hitTest:), hit_test as extern "C" fn(_, _, _) -> _);
        d.add_method(sel!(mouseDown:), f_md as extern "C" fn(_, _, _));
        d.add_method(sel!(mouseUp:), f_mu as extern "C" fn(_, _, _));
        d.add_method(sel!(rightMouseDown:), f_rmd as extern "C" fn(_, _, _));
        d.add_method(sel!(rightMouseUp:), f_rmu as extern "C" fn(_, _, _));
        d.add_method(sel!(otherMouseDown:), f_omd as extern "C" fn(_, _, _));
        d.add_method(sel!(otherMouseUp:), f_omu as extern "C" fn(_, _, _));
        d.add_method(sel!(mouseMoved:), f_mm as extern "C" fn(_, _, _));
        d.add_method(sel!(mouseDragged:), f_md2 as extern "C" fn(_, _, _));
        d.add_method(sel!(rightMouseDragged:), f_rmd2 as extern "C" fn(_, _, _));
        d.add_method(sel!(otherMouseDragged:), f_omd2 as extern "C" fn(_, _, _));
        d.add_method(sel!(scrollWheel:), f_sw as extern "C" fn(_, _, _));
        d.register();
    });
    AnyClass::get(c"GPUIOverlaySpikeView").unwrap()
}

unsafe fn create(native_view: NonNull<AnyObject>) -> Option<Overlay> {
    unsafe {
        let native = native_view.as_ref();
        let superview: *mut AnyObject = msg_send![native, superview];
        if superview.is_null() {
            return None;
        }
        let frame: NSRect = msg_send![native, frame];
        let obj: *mut AnyObject = msg_send![overlay_class(), alloc];
        let obj: *mut AnyObject = msg_send![obj, initWithFrame: frame];
        let view = Retained::from_raw(obj)?;
        let layer = renderer::new_window_layer(true);
        let _: () = msg_send![&*view, setWantsLayer: Bool::YES];
        let host: *mut AnyObject = msg_send![&*view, layer];
        let _: () = msg_send![host, addSublayer: layer.as_ptr() as *mut AnyObject];
        // width|height sizable
        let _: () = msg_send![&*view, setAutoresizingMask: 18usize];
        let rects: Rects = Arc::new(Mutex::new(Vec::new()));
        let leaked = Arc::into_raw(rects.clone()) as *mut c_void; // spike: leaked per window
        let set = |name: &CStr, v: *mut c_void| {
            let iv = (*view).class().instance_variable(name).unwrap();
            *iv.load_mut::<*mut c_void>(&mut *(obj)) = v;
        };
        set(TARGET, native_view.as_ptr() as *mut c_void);
        set(RECTS, Arc::as_ptr(&rects) as *mut c_void);
        let _ = leaked;
        // NSWindowAbove = 1, relativeTo: the GPUIView, so it stacks over its WKWebView children.
        let _: () = msg_send![superview, addSubview: &*view, positioned: 1isize, relativeTo: native as *const AnyObject];
        Some(Overlay { view, layer, rects })
    }
}

/// Returns true when it drew the scene as base + overlay.
pub(crate) fn draw(
    slot: &mut Option<Overlay>,
    native_view: NonNull<AnyObject>,
    renderer: &mut renderer::Renderer,
    scene: &Scene,
) -> bool {
    if !enabled() {
        return false;
    }
    let Some((base, overlay_scene, rects)) = scene.split_overlay() else {
        // No deferred draws this frame: base only, and clear any stale overlay.
        if let Some(o) = slot.as_mut() {
            o.rects.lock().unwrap().clear();
            let empty = Scene::default();
            let mut empty = empty;
            empty.finish();
            renderer.draw_overlay(&empty, &o.layer);
        }
        return false;
    };
    if slot.is_none() {
        *slot = unsafe { create(native_view) };
    }
    let Some(o) = slot.as_mut() else { return false };
    unsafe {
        // Native children (wry's WKWebView) are added to the same superview, possibly after us:
        // keep the overlay the topmost sibling.
        let sv: *mut AnyObject = msg_send![&*o.view, superview];
        if !sv.is_null() {
            let subs: *mut AnyObject = msg_send![sv, subviews];
            let last: *mut AnyObject = msg_send![subs, lastObject];
            if last != Retained::as_ptr(&o.view) as *mut AnyObject {
                let nil: *const AnyObject = std::ptr::null();
                let _: () = msg_send![sv, addSubview: &*o.view, positioned: 1isize, relativeTo: nil];
            }
        }
        static WAS_MODAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let modal = gpui::overlay_modal();
        if modal && !WAS_MODAL.swap(true, std::sync::atomic::Ordering::Relaxed) {
            // Take keyboard focus back from a native child (WKWebView) so Escape reaches GPUI.
            let win: *mut AnyObject = msg_send![native_view.as_ref(), window];
            if !win.is_null() {
                let ok: Bool = msg_send![win, makeFirstResponder: native_view.as_ptr()];
                let fr: *mut AnyObject = msg_send![win, firstResponder];
                eprintln!("modal: makeFirstResponder={ok:?} first={:?} gpuiview={:?}", fr, native_view.as_ptr());
            }
        } else if !modal {
            WAS_MODAL.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(bl) = renderer.layer() {
            let ds = bl.drawable_size();
            let scale: f64 = msg_send![bl.as_ptr() as *mut AnyObject, contentsScale];
            let _: () = msg_send![o.layer.as_ptr() as *mut AnyObject, setContentsScale: scale];
            o.layer.set_drawable_size(ds);
            // Layer-hosting views do not size their layer for us.
            let b: NSRect = msg_send![&*o.view, bounds];
            let _: () = msg_send![o.layer.as_ptr() as *mut AnyObject, setFrame: b];
            let to_pt = |v: ScaledPixels| f64::from(v.0) / scale;
            *o.rects.lock().unwrap() = rects
                .iter()
                .map(|b| [to_pt(b.origin.x), to_pt(b.origin.y), to_pt(b.size.width), to_pt(b.size.height)])
                .collect();
        }
    }
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 120 == 0 {
        unsafe {
            let f: NSRect = msg_send![&*o.view, frame];
            let sv: *mut AnyObject = msg_send![&*o.view, superview];
            let subs: *mut AnyObject = msg_send![sv, subviews];
            let n: usize = msg_send![subs, count];
            let hidden: Bool = msg_send![&*o.view, isHidden];
            eprintln!("overlay: frame={:?} subviews={n} hidden={hidden:?} rects={} ds={:?}", f, rects.len(), o.layer.drawable_size());
        }
    }
    renderer.draw(&base);
    renderer.draw_overlay(&overlay_scene, &o.layer);
    let _ = (&o.view, NSSize::ZERO);
    true
}
