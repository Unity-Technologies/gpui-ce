use std::cell::OnceCell;
use std::fmt::Debug;

use ::sum_tree::SumTree;
use collections::FxHashMap;
use sum_tree::Bias;

use crate::{FocusHandle, FocusId};

/// Represents a collection of focus handles using the tab-index APIs.
///
/// Painting only records `insertion_history`; the sorted order and the id
/// lookup are built on the first query after a change (a Tab press, or the a11y
/// debug harvest), so a frame nobody tabs through never pays for them.
#[derive(Debug, Default)]
pub(crate) struct TabStopMap {
    pub(crate) insertion_history: Vec<TabStopOperation>,
    built: OnceCell<BuiltTabStops>,
}

#[derive(Debug)]
struct BuiltTabStops {
    by_id: FxHashMap<FocusId, TabStopNode>,
    order: SumTree<TabStopNode>,
    /// Number of `by_id` entries that are tab stops.
    tab_stop_count: usize,
}

#[derive(Debug, Clone)]
pub enum TabStopOperation {
    Insert(FocusHandle),
    Group(TabIndex),
    GroupEnd,
}

impl TabStopOperation {
    fn focus_handle(&self) -> Option<&FocusHandle> {
        match self {
            TabStopOperation::Insert(focus_handle) => Some(focus_handle),
            _ => None,
        }
    }
}

type TabIndex = isize;

#[derive(Debug, Default, PartialEq, Eq, Clone, Ord, PartialOrd)]
struct TabStopPath(smallvec::SmallVec<[TabIndex; 6]>);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct TabStopNode {
    /// Path to access the node in the tree
    /// The final node in the list is a leaf node corresponding to an actual focus handle,
    /// all other nodes are group nodes
    path: TabStopPath,
    /// index into the backing array of nodes. Corresponds to insertion order
    node_insertion_index: usize,

    /// Whether this node is a tab stop
    tab_stop: bool,
}

impl Ord for TabStopNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.path
            .cmp(&other.path)
            .then(self.node_insertion_index.cmp(&other.node_insertion_index))
    }
}

impl PartialOrd for TabStopNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

impl TabStopMap {
    pub fn insert(&mut self, focus_handle: &FocusHandle) {
        self.built.take();
        self.insertion_history
            .push(TabStopOperation::Insert(focus_handle.clone()));
    }

    pub fn begin_group(&mut self, tab_index: isize) {
        self.built.take();
        self.insertion_history
            .push(TabStopOperation::Group(tab_index));
    }

    pub fn end_group(&mut self) {
        self.built.take();
        self.insertion_history.push(TabStopOperation::GroupEnd);
    }

    pub fn clear(&mut self) {
        self.built.take();
        self.insertion_history.clear();
    }

    fn built(&self) -> &BuiltTabStops {
        self.built.get_or_init(|| {
            let mut path = TabStopPath::default();
            let mut nodes = Vec::new();
            let mut by_id = FxHashMap::default();
            let mut tab_stop_count = 0;
            for (index, operation) in self.insertion_history.iter().enumerate() {
                match operation {
                    TabStopOperation::Insert(focus_handle) => {
                        let mut node_path = path.clone();
                        node_path.0.push(focus_handle.tab_index);
                        let node = TabStopNode {
                            node_insertion_index: index,
                            tab_stop: focus_handle.tab_stop,
                            path: node_path,
                        };
                        let replaced = by_id.insert(focus_handle.id, node.clone());
                        if replaced.is_some_and(|node: TabStopNode| node.tab_stop) {
                            tab_stop_count -= 1;
                        }
                        if node.tab_stop {
                            tab_stop_count += 1;
                        }
                        nodes.push(node);
                    }
                    TabStopOperation::Group(tab_index) => path.0.push(*tab_index),
                    TabStopOperation::GroupEnd => {
                        path.0.pop();
                    }
                }
            }
            // `node_insertion_index` is unique, so no two nodes compare equal.
            nodes.sort();
            BuiltTabStops {
                by_id,
                order: SumTree::from_iter(nodes, ()),
                tab_stop_count,
            }
        })
    }

    pub fn next(&self, focused_id: Option<&FocusId>) -> Option<FocusHandle> {
        let Some(focused_id) = focused_id else {
            let first = self.built().order.first()?;
            if first.tab_stop {
                return self.focus_handle_for_order(first);
            } else {
                return self
                    .next_inner(first)
                    .and_then(|order| self.focus_handle_for_order(order));
            }
        };

        let Some(node) = self.tab_node_for_focus_id(focused_id) else {
            return self.next(None);
        };
        let item = self.next_inner(node);

        if let Some(item) = item {
            self.focus_handle_for_order(&item)
        } else {
            self.next(None)
        }
    }

    fn next_inner(&self, node: &TabStopNode) -> Option<&TabStopNode> {
        let mut cursor = self.built().order.cursor::<TabStopNode>(());
        cursor.seek(&node, Bias::Left);
        cursor.next();
        while let Some(item) = cursor.item()
            && !item.tab_stop
        {
            cursor.next();
        }

        cursor.item()
    }

    pub fn prev(&self, focused_id: Option<&FocusId>) -> Option<FocusHandle> {
        let Some(focused_id) = focused_id else {
            let last = self.built().order.last()?;
            if last.tab_stop {
                return self.focus_handle_for_order(last);
            } else {
                return self
                    .prev_inner(last)
                    .and_then(|order| self.focus_handle_for_order(order));
            }
        };

        let Some(node) = self.tab_node_for_focus_id(focused_id) else {
            return self.prev(None);
        };
        let item = self.prev_inner(node);

        if let Some(item) = item {
            self.focus_handle_for_order(&item)
        } else {
            self.prev(None)
        }
    }

    fn prev_inner(&self, node: &TabStopNode) -> Option<&TabStopNode> {
        let mut cursor = self.built().order.cursor::<TabStopNode>(());
        cursor.seek(&node, Bias::Left);
        cursor.prev();
        while let Some(item) = cursor.item()
            && !item.tab_stop
        {
            cursor.prev();
        }

        cursor.item()
    }

    pub fn replay(&mut self, nodes: &[TabStopOperation]) {
        for node in nodes {
            match node {
                TabStopOperation::Insert(focus_handle) => self.insert(focus_handle),
                TabStopOperation::Group(tab_index) => self.begin_group(*tab_index),
                TabStopOperation::GroupEnd => self.end_group(),
            }
        }
    }

    pub fn paint_index(&self) -> usize {
        self.insertion_history.len()
    }

    pub(crate) fn tab_stop_count(&self) -> usize {
        self.built().tab_stop_count
    }

    fn focus_handle_for_order(&self, order: &TabStopNode) -> Option<FocusHandle> {
        let handle = self.insertion_history[order.node_insertion_index].focus_handle();
        debug_assert!(
            handle.is_some(),
            "The order node did not correspond to an element, this is a GPUI bug"
        );
        handle.cloned()
    }

    fn tab_node_for_focus_id(&self, focused_id: &FocusId) -> Option<&TabStopNode> {
        let Some(order) = self.built().by_id.get(focused_id) else {
            return None;
        };
        Some(order)
    }
}

mod sum_tree_impl {
    use sum_tree::SeekTarget;

    use crate::tab_stop::{TabStopNode, TabStopPath};

    #[derive(Clone, Debug)]
    pub struct TabStopOrderNodeSummary {
        max_index: usize,
        max_path: TabStopPath,
        pub tab_stops: usize,
    }

    pub type TabStopCount = usize;

    impl sum_tree::ContextLessSummary for TabStopOrderNodeSummary {
        fn zero() -> Self {
            TabStopOrderNodeSummary {
                max_index: 0,
                max_path: TabStopPath::default(),
                tab_stops: 0,
            }
        }

        fn add_summary(&mut self, summary: &Self) {
            self.max_index = summary.max_index;
            self.max_path = summary.max_path.clone();
            self.tab_stops += summary.tab_stops;
        }
    }

    impl sum_tree::KeyedItem for TabStopNode {
        type Key = Self;

        fn key(&self) -> Self::Key {
            self.clone()
        }
    }

    impl sum_tree::Item for TabStopNode {
        type Summary = TabStopOrderNodeSummary;

        fn summary(&self, _cx: <Self::Summary as sum_tree::Summary>::Context<'_>) -> Self::Summary {
            TabStopOrderNodeSummary {
                max_index: self.node_insertion_index,
                max_path: self.path.clone(),
                tab_stops: if self.tab_stop { 1 } else { 0 },
            }
        }
    }

    impl<'a> sum_tree::Dimension<'a, TabStopOrderNodeSummary> for TabStopCount {
        fn zero(_: <TabStopOrderNodeSummary as sum_tree::Summary>::Context<'_>) -> Self {
            0
        }

        fn add_summary(
            &mut self,
            summary: &'a TabStopOrderNodeSummary,
            _: <TabStopOrderNodeSummary as sum_tree::Summary>::Context<'_>,
        ) {
            *self += summary.tab_stops;
        }
    }

    impl<'a> sum_tree::Dimension<'a, TabStopOrderNodeSummary> for TabStopNode {
        fn zero(_: <TabStopOrderNodeSummary as sum_tree::Summary>::Context<'_>) -> Self {
            TabStopNode::default()
        }

        fn add_summary(
            &mut self,
            summary: &'a TabStopOrderNodeSummary,
            _: <TabStopOrderNodeSummary as sum_tree::Summary>::Context<'_>,
        ) {
            self.node_insertion_index = summary.max_index;
            self.path = summary.max_path.clone();
        }
    }

    impl<'a, 'b> SeekTarget<'a, TabStopOrderNodeSummary, TabStopNode> for &'b TabStopNode {
        fn cmp(
            &self,
            cursor_location: &TabStopNode,
            _: <TabStopOrderNodeSummary as sum_tree::Summary>::Context<'_>,
        ) -> std::cmp::Ordering {
            Iterator::cmp(self.path.0.iter(), cursor_location.path.0.iter()).then(
                <usize as Ord>::cmp(
                    &self.node_insertion_index,
                    &cursor_location.node_insertion_index,
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools as _;
    use parking_lot::RwLock;

    use crate::{FocusHandle, FocusId, FocusMap, TabStopMap};
    use std::sync::Arc;

    #[test]
    fn test_tab_handles() {
        let focus_map = Arc::new(RwLock::new(FocusMap::default()));
        let mut tab_index_map = TabStopMap::default();

        let focus_handles = [
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(0),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(1),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(1),
            FocusHandle::new(&focus_map),
            FocusHandle::new(&focus_map).tab_index(2),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(0),
            FocusHandle::new(&focus_map).tab_stop(true).tab_index(2),
        ];

        for handle in focus_handles.iter() {
            tab_index_map.insert(handle);
        }
        let expected = [
            focus_handles[0].clone(),
            focus_handles[5].clone(),
            focus_handles[1].clone(),
            focus_handles[2].clone(),
            focus_handles[6].clone(),
        ];

        let mut prev = None;
        let mut found = vec![];
        for _ in 0..expected.len() {
            let handle = tab_index_map.next(prev.as_ref()).unwrap();
            prev = Some(handle.id);
            found.push(handle.id);
        }

        assert_eq!(
            found,
            expected.iter().map(|handle| handle.id).collect::<Vec<_>>()
        );

        // Select first tab index if no handle is currently focused.
        assert_eq!(tab_index_map.next(None), Some(expected[0].clone()));
        // Select last tab index if no handle is currently focused.
        assert_eq!(tab_index_map.prev(None), expected.last().cloned(),);

        assert_eq!(
            tab_index_map.next(Some(&expected[0].id)),
            Some(expected[1].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[1].id)),
            Some(expected[2].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[2].id)),
            Some(expected[3].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[3].id)),
            Some(expected[4].clone())
        );
        assert_eq!(
            tab_index_map.next(Some(&expected[4].id)),
            Some(expected[0].clone())
        );

        // prev
        assert_eq!(tab_index_map.prev(None), Some(expected[4].clone()));
        assert_eq!(
            tab_index_map.prev(Some(&expected[0].id)),
            Some(expected[4].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[1].id)),
            Some(expected[0].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[2].id)),
            Some(expected[1].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[3].id)),
            Some(expected[2].clone())
        );
        assert_eq!(
            tab_index_map.prev(Some(&expected[4].id)),
            Some(expected[3].clone())
        );
    }

    #[test]
    fn test_tab_stop_count_tracks_inserts_replacements_and_clear() {
        let focus_map = Arc::new(RwLock::new(FocusMap::default()));
        let mut map = TabStopMap::default();
        let stop = FocusHandle::new(&focus_map).tab_stop(true).tab_index(0);
        let other_stop = FocusHandle::new(&focus_map).tab_stop(true).tab_index(1);
        let non_stop = FocusHandle::new(&focus_map).tab_stop(false).tab_index(2);

        map.insert(&stop);
        map.insert(&non_stop);
        map.insert(&other_stop);
        assert_eq!(map.tab_stop_count(), 2);

        // The same handle inserted again replaces its `by_id` entry.
        map.insert(&stop);
        assert_eq!(map.tab_stop_count(), 2);

        let mut replayed = TabStopMap::default();
        replayed.replay(&map.insertion_history);
        assert_eq!(replayed.tab_stop_count(), 2);

        map.clear();
        assert_eq!(map.tab_stop_count(), 0);
    }

    /// Timing probe, not a pass/fail test: `cargo test -p gpui-ce --lib
    /// bench_tab_stop_map -- --ignored --nocapture`. A frame inserts every
    /// focusable (groups every 10th), and only a Tab press asks for the order.
    #[test]
    #[ignore]
    fn bench_tab_stop_map_frame_cost() {
        const HANDLES: usize = 5_000;
        const FRAMES: usize = 100;
        let focus_map = Arc::new(FocusMap::default());
        let handles: Vec<FocusHandle> = (0..HANDLES)
            .map(|i| {
                FocusHandle::new(&focus_map)
                    .tab_stop(true)
                    .tab_index((i % 7) as isize)
            })
            .collect();
        let frame = |map: &mut TabStopMap, query: bool| {
            map.clear();
            for (i, handle) in handles.iter().enumerate() {
                if i % 10 == 0 {
                    map.begin_group(i as isize % 3);
                }
                map.insert(handle);
                if i % 10 == 9 {
                    map.end_group();
                }
            }
            if query {
                std::hint::black_box(map.next(Some(&handles[HANDLES / 2].id)));
                std::hint::black_box(map.tab_stop_count());
            }
        };
        for query in [false, true] {
            let mut map = TabStopMap::default();
            frame(&mut map, query);
            let start = std::time::Instant::now();
            for _ in 0..FRAMES {
                frame(&mut map, query);
            }
            println!(
                "tab_stop_map query_every_frame={query}: {:?}/frame ({HANDLES} handles)",
                start.elapsed() / FRAMES as u32
            );
        }
    }

    #[test]
    fn test_tab_non_stop_filtering() {
        let focus_map = Arc::new(RwLock::new(FocusMap::default()));
        let mut tab_index_map = TabStopMap::default();

        // Check that we can query next from a non-stop tab
        let tab_non_stop_1 = FocusHandle::new(&focus_map).tab_stop(false).tab_index(1);
        let tab_stop_2 = FocusHandle::new(&focus_map).tab_stop(true).tab_index(2);
        tab_index_map.insert(&tab_non_stop_1);
        tab_index_map.insert(&tab_stop_2);
        let result = tab_index_map.next(Some(&tab_non_stop_1.id)).unwrap();
        assert_eq!(result.id, tab_stop_2.id);

        // Check that we skip over non-stop tabs
        let tab_stop_0 = FocusHandle::new(&focus_map).tab_stop(true).tab_index(0);
        let tab_non_stop_0 = FocusHandle::new(&focus_map).tab_stop(false).tab_index(0);
        tab_index_map.insert(&tab_stop_0);
        tab_index_map.insert(&tab_non_stop_0);
        let result = tab_index_map.next(Some(&tab_stop_0.id)).unwrap();
        assert_eq!(result.id, tab_stop_2.id);
    }

    #[must_use]
    struct TabStopMapTest {
        tab_map: TabStopMap,
        focus_map: Arc<RwLock<FocusMap>>,
        expected: Vec<(usize, FocusId)>,
    }

    impl TabStopMapTest {
        #[must_use]
        fn new() -> Self {
            Self {
                tab_map: TabStopMap::default(),
                focus_map: Arc::new(RwLock::new(FocusMap::default())),
                expected: Vec::default(),
            }
        }

        #[must_use]
        fn tab_non_stop(mut self, index: isize) -> Self {
            let handle = FocusHandle::new(&self.focus_map)
                .tab_stop(false)
                .tab_index(index);
            self.tab_map.insert(&handle);
            self
        }

        #[must_use]
        fn tab_stop(mut self, index: isize, expected: usize) -> Self {
            let handle = FocusHandle::new(&self.focus_map)
                .tab_stop(true)
                .tab_index(index);
            self.tab_map.insert(&handle);
            self.expected.push((expected, handle.id));
            self.expected.sort_by_key(|(expected, _)| *expected);
            self
        }

        #[must_use]
        fn tab_group(mut self, tab_index: isize, children: impl FnOnce(Self) -> Self) -> Self {
            self.tab_map.begin_group(tab_index);
            self = children(self);
            self.tab_map.end_group();
            self
        }

        fn traverse_tab_map(
            &self,
            traverse: impl Fn(&TabStopMap, Option<&FocusId>) -> Option<FocusHandle>,
        ) -> Vec<FocusId> {
            let mut last_focus_id = None;
            let mut found = vec![];
            for _ in 0..self.expected.len() {
                let handle = traverse(&self.tab_map, last_focus_id.as_ref()).unwrap();
                last_focus_id = Some(handle.id);
                found.push(handle.id);
            }
            found
        }

        fn assert(self) {
            let mut expected = self.expected.iter().map(|(_, id)| *id).collect_vec();

            // Check next order
            let forward_found = self.traverse_tab_map(|tab_map, prev| tab_map.next(prev));
            assert_eq!(forward_found, expected);

            // Test overflow. Last to first
            assert_eq!(
                self.tab_map
                    .next(forward_found.last())
                    .map(|handle| handle.id),
                expected.first().cloned()
            );

            // Check previous order
            let reversed_found = self.traverse_tab_map(|tab_map, prev| tab_map.prev(prev));
            expected.reverse();
            assert_eq!(reversed_found, expected);

            // Test overflow. First to last
            assert_eq!(
                self.tab_map
                    .prev(reversed_found.last())
                    .map(|handle| handle.id),
                expected.first().cloned(),
            );
        }
    }

    #[test]
    fn test_with_disabled_tab_stop() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_non_stop(1)
            .tab_stop(2, 1)
            .tab_stop(3, 2)
            .assert();
    }

    #[test]
    fn test_with_multiple_disabled_tab_stops() {
        TabStopMapTest::new()
            .tab_non_stop(0)
            .tab_stop(1, 0)
            .tab_non_stop(3)
            .tab_stop(3, 1)
            .tab_non_stop(4)
            .assert();
    }

    #[test]
    fn test_tab_group_functionality() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(0, 1)
            .tab_group(2, |t| t.tab_stop(0, 2).tab_stop(1, 3))
            .tab_stop(3, 4)
            .tab_stop(4, 5)
            .assert()
    }

    #[test]
    fn test_sibling_groups() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(1, 1)
            .tab_group(2, |test| test.tab_stop(0, 2).tab_stop(1, 3))
            .tab_stop(3, 4)
            .tab_stop(4, 5)
            .tab_group(6, |test| test.tab_stop(0, 6).tab_stop(1, 7))
            .tab_stop(7, 8)
            .tab_stop(8, 9)
            .assert();
    }

    #[test]
    fn test_nested_group() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(1, 1)
            .tab_group(2, |t| {
                t.tab_group(0, |t| t.tab_stop(0, 2).tab_stop(1, 3))
                    .tab_stop(1, 4)
            })
            .tab_stop(3, 5)
            .tab_stop(4, 6)
            .assert();
    }

    #[test]
    fn test_sibling_nested_groups() {
        TabStopMapTest::new()
            .tab_stop(0, 0)
            .tab_stop(1, 1)
            .tab_group(2, |builder| {
                builder
                    .tab_stop(0, 2)
                    .tab_stop(2, 5)
                    .tab_group(1, |builder| builder.tab_stop(0, 3).tab_stop(1, 4))
                    .tab_group(3, |builder| builder.tab_stop(0, 6).tab_stop(1, 7))
            })
            .tab_stop(3, 8)
            .tab_stop(4, 9)
            .assert();
    }

    #[test]
    fn test_sibling_nested_groups_out_of_order() {
        TabStopMapTest::new()
            .tab_stop(9, 9)
            .tab_stop(8, 8)
            .tab_group(7, |builder| {
                builder
                    .tab_stop(0, 2)
                    .tab_stop(2, 5)
                    .tab_group(3, |builder| builder.tab_stop(1, 7).tab_stop(0, 6))
                    .tab_group(1, |builder| builder.tab_stop(0, 3).tab_stop(1, 4))
            })
            .tab_stop(3, 0)
            .tab_stop(4, 1)
            .assert();
    }
}
