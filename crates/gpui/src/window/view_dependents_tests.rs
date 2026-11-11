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
