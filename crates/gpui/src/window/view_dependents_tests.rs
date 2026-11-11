//! Tests for invalidating cached views through the entities they read.

use super::*;
use crate::{Context, Entity, ParentElement, Render, Styled, TestAppContext, div, px};

struct Model {
    value: usize,
}

/// Reads `Model` while rendering, and counts its renders.
struct Reader {
    model: Entity<Model>,
    renders: Rc<Cell<usize>>,
    seen: Rc<Cell<usize>>,
}

impl Render for Reader {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        self.seen.set(self.model.read(cx).value);
        div().size(px(10.))
    }
}

/// Holds a `Reader` behind `.cached()`. It reads `Model` too, before the
/// reader does, so the reader must still count the model as its own read.
struct Host {
    reader: Entity<Reader>,
    model: Entity<Model>,
}

impl Render for Host {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let _ = self.model.read(cx).value;
        div().child(
            self.reader
                .clone()
                .cached(crate::StyleRefinement::default().size(px(10.))),
        )
    }
}

#[gpui::test]
fn a_cached_view_renders_again_when_an_entity_it_read_notifies(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let seen = Rc::new(Cell::new(0));
    let model = cx.new(|_| Model { value: 1 });
    let window = cx.add_window({
        let (renders, seen, model) = (renders.clone(), seen.clone(), model.clone());
        move |_, cx| Host {
            reader: cx.new(|_| Reader {
                model: model.clone(),
                renders,
                seen,
            }),
            model,
        }
    });
    let draw = |cx: &mut TestAppContext| {
        cx.run_until_parked();
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    };
    draw(cx);
    assert_eq!(seen.get(), 1);
    let first = renders.get();
    assert!(first > 0, "the reader rendered");

    // The host renders again; the cached reader replays its last frame.
    window.update(cx, |_, _, cx| cx.notify()).unwrap();
    draw(cx);
    assert_eq!(renders.get(), first, "a host notify reuses the reader");

    model.update(cx, |model, cx| {
        model.value = 2;
        cx.notify();
    });
    draw(cx);
    assert_eq!(
        (renders.get(), seen.get()),
        (first + 1, 2),
        "the reader renders the model's new state, not its last frame"
    );
}

/// A cached view that holds a cached `Reader` and reads nothing itself.
struct Middle {
    reader: Entity<Reader>,
    renders: Rc<Cell<usize>>,
}

impl Render for Middle {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().child(
            self.reader
                .clone()
                .cached(crate::StyleRefinement::default().size(px(10.))),
        )
    }
}

/// The window root: holds a cached `Middle`.
struct Root {
    middle: Entity<Middle>,
}

impl Render for Root {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(
            self.middle
                .clone()
                .cached(crate::StyleRefinement::default().size(px(10.))),
        )
    }
}

#[gpui::test]
fn a_nested_cached_view_keeps_its_dependencies_when_its_parent_is_reused(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let seen = Rc::new(Cell::new(0));
    let middle_renders = Rc::new(Cell::new(0));
    let model = cx.new(|_| Model { value: 1 });
    let window = cx.add_window({
        let (renders, seen, model) = (renders.clone(), seen.clone(), model.clone());
        let middle_renders = middle_renders.clone();
        move |_, cx| {
            let reader = cx.new(|_| Reader {
                model,
                renders,
                seen,
            });
            Root {
                middle: cx.new(|_| Middle {
                    reader,
                    renders: middle_renders,
                }),
            }
        }
    });
    let draw = |cx: &mut TestAppContext| {
        cx.run_until_parked();
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    };
    draw(cx);
    let first = renders.get();
    let middle_first = middle_renders.get();
    assert!(first > 0, "the reader rendered");

    // The root renders again and reuses the cached middle whole, so the
    // reader's prepaint never runs this frame.
    window.update(cx, |_, _, cx| cx.notify()).unwrap();
    draw(cx);
    assert_eq!(
        (middle_renders.get(), renders.get()),
        (middle_first, first),
        "a root notify reuses the middle and the reader"
    );

    model.update(cx, |model, cx| {
        model.value = 2;
        cx.notify();
    });
    draw(cx);
    assert_eq!(
        (renders.get(), seen.get()),
        (first + 1, 2),
        "the reader renders the model's new state after its parent was reused"
    );
}

/// Keeps a keyed state entity, and reports which one it got.
struct Keeper {
    state: Rc<Cell<Option<EntityId>>>,
}

impl Render for Keeper {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = window.use_keyed_state("kept", cx, |_, _| 0usize);
        self.state.set(Some(state.entity_id()));
        div().size(px(10.))
    }
}

/// Holds a `Keeper` behind `.cached()`.
struct KeeperHost {
    keeper: Entity<Keeper>,
}

impl Render for KeeperHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(
            self.keeper
                .clone()
                .cached(crate::StyleRefinement::default().size(px(10.))),
        )
    }
}

/// A dirty cached view renders in `request_layout` (the direction probe),
/// before the range a later reuse replays. The element state it read there
/// must survive that reuse.
#[gpui::test]
fn a_reused_cached_view_keeps_the_state_it_read_while_probed(cx: &mut TestAppContext) {
    let state = Rc::new(Cell::new(None));
    let keeper = cx.new({
        let state = state.clone();
        |_| Keeper { state }
    });
    let window = cx.add_window({
        let keeper = keeper.clone();
        move |_, _| KeeperHost { keeper }
    });
    let draw = |cx: &mut TestAppContext| {
        cx.run_until_parked();
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    };
    draw(cx);
    let first = state.get().expect("the keeper rendered");

    // The host renders again and the keeper replays its last frame.
    window.update(cx, |_, _, cx| cx.notify()).unwrap();
    draw(cx);
    // The keeper renders again and finds the same state.
    keeper.update(cx, |_, cx| cx.notify());
    draw(cx);
    assert_eq!(
        state.get(),
        Some(first),
        "the keyed state survived the reuse"
    );
}
