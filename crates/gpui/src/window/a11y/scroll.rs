//! Scroll containers recorded while the accessibility tree is built, so the
//! AccessKit scroll actions (`ScrollIntoView`, `ScrollUp`/`Down`/`Left`/`Right`,
//! `SetScrollOffset`) can move them between frames.
//!
//! A container is any element with its own scroll offset: a `div` with
//! `overflow_*_scroll` (tracked by a `ScrollHandle` or not) or a `list`. It
//! need not have an accessibility node of its own; `ScrollIntoView` walks a
//! node's containers from the innermost out either way.

// The action handlers that read these containers are native-only.
#![cfg_attr(target_family = "wasm", allow(dead_code))]

use crate::{Bounds, ListState, Pixels, Point, point, px};
use accesskit::{NodeId, ScrollHint};
use std::{cell::RefCell, rc::Rc};

/// What a container scrolls.
pub(crate) enum ScrollTarget {
    /// A `div`'s offset: negative as content moves up or left, clamped to
    /// `-max..=0` on each axis.
    Offset {
        offset: Rc<RefCell<Point<Pixels>>>,
        max: Point<Pixels>,
    },
    /// A virtualized list, which owns its own clamping.
    List(ListState),
}

pub(crate) struct ScrollContainer {
    /// The container's viewport in window space.
    pub(crate) bounds: Bounds<Pixels>,
    pub(crate) target: ScrollTarget,
    /// The next container out, if any.
    pub(crate) parent: Option<usize>,
    /// Whether this container or one around it can scroll at all.
    pub(crate) chain_scrollable: bool,
}

impl ScrollContainer {
    /// The current offset, as content displacement (`<= 0`).
    pub(crate) fn offset(&self) -> Point<Pixels> {
        match &self.target {
            ScrollTarget::Offset { offset, .. } => *offset.borrow(),
            ScrollTarget::List(list) => list.scroll_px_offset_for_scrollbar(),
        }
    }

    /// The largest scroll on each axis (`>= 0`).
    pub(crate) fn max(&self) -> Point<Pixels> {
        match &self.target {
            ScrollTarget::Offset { max, .. } => *max,
            ScrollTarget::List(list) => list.max_offset_for_scrollbar(),
        }
    }

    /// Move the content by `delta` (positive moves it down or right, which
    /// scrolls up or left), clamped to the scroll range. Returns the move
    /// actually made.
    pub(crate) fn scroll_content_by(&self, delta: Point<Pixels>) -> Point<Pixels> {
        let before = self.offset();
        let max = self.max();
        let wanted = point(
            (before.x + delta.x).clamp(-max.x, px(0.)),
            (before.y + delta.y).clamp(-max.y, px(0.)),
        );
        self.set_offset(wanted);
        self.offset() - before
    }

    /// Set the content displacement (`<= 0` on each axis), clamped.
    pub(crate) fn set_offset(&self, wanted: Point<Pixels>) {
        let max = self.max();
        let wanted = point(
            wanted.x.clamp(-max.x, px(0.)),
            wanted.y.clamp(-max.y, px(0.)),
        );
        match &self.target {
            ScrollTarget::Offset { offset, .. } => *offset.borrow_mut() = wanted,
            ScrollTarget::List(list) => {
                let delta = wanted.y - list.scroll_px_offset_for_scrollbar().y;
                // `scroll_by` takes a distance down the list, the opposite
                // sign of the content's displacement.
                list.scroll_by(-delta);
            }
        }
    }
}

/// How far to move content on one axis so `[start, end]` shows inside
/// `[view_start, view_end]`. `align` picks an edge to align to; without one the
/// smallest move that reveals the item, preferring its start when it does not
/// fit.
fn reveal_axis(
    start: Pixels,
    end: Pixels,
    view_start: Pixels,
    view_end: Pixels,
    align: Option<Edge>,
) -> Pixels {
    match align {
        Some(Edge::Start) => view_start - start,
        Some(Edge::End) => view_end - end,
        None if start < view_start => view_start - start,
        None if end > view_end => (view_end - end).max(view_start - start),
        None => px(0.),
    }
}

#[derive(Clone, Copy)]
enum Edge {
    Start,
    End,
}

/// The content move that reveals `item` inside `viewport`, honouring `hint`.
pub(crate) fn reveal_delta(
    item: Bounds<Pixels>,
    viewport: Bounds<Pixels>,
    hint: Option<ScrollHint>,
) -> Point<Pixels> {
    let (x_edge, y_edge) = match hint {
        None => (None, None),
        Some(ScrollHint::TopLeft) => (Some(Edge::Start), Some(Edge::Start)),
        Some(ScrollHint::BottomRight) => (Some(Edge::End), Some(Edge::End)),
        Some(ScrollHint::TopEdge) => (None, Some(Edge::Start)),
        Some(ScrollHint::BottomEdge) => (None, Some(Edge::End)),
        Some(ScrollHint::LeftEdge) => (Some(Edge::Start), None),
        Some(ScrollHint::RightEdge) => (Some(Edge::End), None),
    };
    point(
        reveal_axis(
            item.left(),
            item.right(),
            viewport.left(),
            viewport.right(),
            x_edge,
        ),
        reveal_axis(
            item.top(),
            item.bottom(),
            viewport.top(),
            viewport.bottom(),
            y_edge,
        ),
    )
}

/// Containers recorded this frame and the stack of those enclosing whatever is
/// being prepainted now.
#[derive(Default)]
pub(crate) struct ScrollContainers {
    pub(crate) containers: Vec<ScrollContainer>,
    stack: Vec<usize>,
    /// The innermost container around each node, as recorded when the node was
    /// pushed.
    pub(crate) node_parent: collections::FxHashMap<NodeId, usize>,
    /// The container a node IS, for nodes on scrollable elements.
    pub(crate) by_node: collections::FxHashMap<NodeId, usize>,
}

impl ScrollContainers {
    pub(crate) fn clear(&mut self) {
        self.containers.clear();
        self.stack.clear();
        self.node_parent.clear();
        self.by_node.clear();
    }

    /// The innermost container being prepainted, if any.
    pub(crate) fn current(&self) -> Option<usize> {
        self.stack.last().copied()
    }

    /// Whether anything enclosing the element being prepainted can scroll.
    pub(crate) fn current_chain_scrollable(&self) -> bool {
        self.current()
            .is_some_and(|index| self.containers[index].chain_scrollable)
    }

    /// Enter a container. `can_scroll` is passed in rather than read from
    /// `target`, because a list's state is borrowed while it prepaints.
    pub(crate) fn push(
        &mut self,
        bounds: Bounds<Pixels>,
        target: ScrollTarget,
        can_scroll: bool,
    ) -> usize {
        let parent = self.current();
        let chain_scrollable =
            can_scroll || parent.is_some_and(|index| self.containers[index].chain_scrollable);
        let container = ScrollContainer {
            bounds,
            target,
            parent,
            chain_scrollable,
        };
        let index = self.containers.len();
        self.containers.push(container);
        self.stack.push(index);
        index
    }

    pub(crate) fn pop(&mut self) {
        self.stack.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, point, px, size};

    fn rect(y: f32, height: f32) -> Bounds<Pixels> {
        Bounds::new(point(px(0.), px(y)), size(px(100.), px(height)))
    }

    #[test]
    fn an_item_below_the_viewport_moves_up_just_enough() {
        let delta = reveal_delta(rect(250., 20.), rect(0., 200.), None);
        assert_eq!(delta.y, px(-70.));
        assert_eq!(delta.x, px(0.));
    }

    #[test]
    fn an_item_above_the_viewport_moves_down_to_the_top() {
        let delta = reveal_delta(rect(-40., 20.), rect(0., 200.), None);
        assert_eq!(delta.y, px(40.));
    }

    #[test]
    fn a_visible_item_does_not_move_without_a_hint() {
        assert_eq!(
            reveal_delta(rect(50., 20.), rect(0., 200.), None),
            point(px(0.), px(0.))
        );
    }

    #[test]
    fn an_item_taller_than_the_viewport_aligns_its_top() {
        let delta = reveal_delta(rect(100., 400.), rect(0., 200.), None);
        assert_eq!(delta.y, px(-100.));
    }

    #[test]
    fn hints_pick_the_edge() {
        let item = rect(50., 20.);
        let view = rect(0., 200.);
        assert_eq!(
            reveal_delta(item, view, Some(ScrollHint::TopEdge)).y,
            px(-50.)
        );
        assert_eq!(
            reveal_delta(item, view, Some(ScrollHint::BottomEdge)).y,
            px(130.)
        );
    }

    #[test]
    fn scrolling_clamps_to_the_range_and_reports_the_applied_move() {
        let offset = Rc::new(RefCell::new(point(px(0.), px(-10.))));
        let container = ScrollContainer {
            bounds: rect(0., 100.),
            target: ScrollTarget::Offset {
                offset: offset.clone(),
                max: point(px(0.), px(50.)),
            },
            parent: None,
            chain_scrollable: true,
        };
        assert_eq!(
            container.scroll_content_by(point(px(0.), px(-100.))),
            point(px(0.), px(-40.))
        );
        assert_eq!(*offset.borrow(), point(px(0.), px(-50.)));
        assert_eq!(
            container.scroll_content_by(point(px(5.), px(500.))),
            point(px(0.), px(50.))
        );
    }
}
