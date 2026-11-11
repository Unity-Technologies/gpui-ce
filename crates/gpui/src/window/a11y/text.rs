//! Editable text exposed to assistive technology as AccessKit `TextRun`
//! children, plus the `text_selection` that places the caret.
//!
//! AccessKit reads a text field's caret and selection only through
//! [`accesskit::TextPosition`]s, which name a `TextRun` node and a character
//! index inside it. A field that sets only `value` therefore has no caret a
//! screen reader can follow, and a `SetTextSelection` request it receives
//! cannot be mapped back to the field's own text. This module builds those runs
//! from a plain string and keeps, per frame, the map back from a position to a
//! byte offset in that string.

use accesskit::{NodeId, Role, TextPosition};
use collections::FxHashMap;
use unicode_segmentation::UnicodeSegmentation;

/// A selection in an editable text, as byte offsets into its whole value.
///
/// `anchor` is the end that stays put while the selection grows; `focus` is the
/// moving end, the caret. They are equal when nothing is selected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct A11yTextSelection {
    /// Byte offset of the fixed end.
    pub anchor: usize,
    /// Byte offset of the moving end (the caret).
    pub focus: usize,
}

impl A11yTextSelection {
    /// A caret at `offset` with nothing selected.
    pub fn caret(offset: usize) -> Self {
        Self {
            anchor: offset,
            focus: offset,
        }
    }
}

/// One `TextRun` built this frame, kept so an incoming position can be mapped
/// back to a byte offset in its owner's value.
#[derive(Debug)]
pub(crate) struct TextRunInfo {
    /// The editable node whose value this run is a line of.
    owner: NodeId,
    /// Byte offset of the run's first character in the owner's value.
    byte_start: usize,
    /// Byte offset, relative to `byte_start`, of each character boundary:
    /// one more entry than the run has characters.
    boundaries: Vec<usize>,
}

/// Per-frame map from `TextRun` node ids to what they cover.
#[derive(Debug, Default)]
pub(crate) struct TextRuns {
    runs: FxHashMap<NodeId, TextRunInfo>,
}

impl TextRuns {
    pub(crate) fn clear(&mut self) {
        self.runs.clear();
    }

    pub(crate) fn insert(&mut self, id: NodeId, info: TextRunInfo) {
        self.runs.insert(id, info);
    }

    /// The owner node and byte offset a position names, if its run was built
    /// in the last frame and the index is in range.
    pub(crate) fn offset(&self, position: &TextPosition) -> Option<(NodeId, usize)> {
        let run = self.runs.get(&position.node)?;
        let relative = run.boundaries.get(position.character_index)?;
        Some((run.owner, run.byte_start + relative))
    }

    /// Both ends of a selection, when they fall in the same owner.
    pub(crate) fn selection(
        &self,
        selection: &accesskit::TextSelection,
    ) -> Option<(NodeId, A11yTextSelection)> {
        let (anchor_owner, anchor) = self.offset(&selection.anchor)?;
        let (focus_owner, focus) = self.offset(&selection.focus)?;
        (anchor_owner == focus_owner).then_some((anchor_owner, A11yTextSelection { anchor, focus }))
    }
}

/// A run ready to push: its node and the info to record for it.
pub(crate) struct BuiltRun {
    pub(crate) node: accesskit::Node,
    pub(crate) info: TextRunInfo,
}

/// Split `text` into one run per line. A line keeps its trailing `\n` as its
/// last character, and a text that is empty or ends in `\n` gets an empty last
/// run, so every caret position has a run to name. Characters are grapheme
/// clusters, as AccessKit asks.
pub(crate) fn build_runs(owner: NodeId, text: &str) -> Vec<BuiltRun> {
    let mut runs = Vec::new();
    let mut line_start = 0;
    for (index, _) in text.match_indices('\n') {
        runs.push(build_run(owner, text, line_start, index + 1));
        line_start = index + 1;
    }
    runs.push(build_run(owner, text, line_start, text.len()));
    runs
}

fn build_run(owner: NodeId, text: &str, start: usize, end: usize) -> BuiltRun {
    let line = &text[start..end];
    let mut lengths: Vec<u8> = Vec::new();
    let mut boundaries = vec![0];
    let mut word_starts: Vec<u8> = Vec::new();
    let mut previous_was_word = false;
    for (offset, grapheme) in line.grapheme_indices(true) {
        // `character_lengths` is a `u8` per character; a cluster longer than
        // 255 bytes (stacked combining marks) is split at its chars.
        let pieces: Vec<&str> = if grapheme.len() <= u8::MAX as usize {
            vec![grapheme]
        } else {
            grapheme
                .char_indices()
                .map(|(i, c)| &grapheme[i..i + c.len_utf8()])
                .collect()
        };
        let mut piece_offset = offset;
        for piece in pieces {
            let is_word = piece.chars().next().is_some_and(char::is_alphanumeric);
            if is_word && !previous_was_word {
                // `word_starts` indexes characters as `u8`, so words past the
                // 255th character of a line go unmarked.
                if let Ok(index) = u8::try_from(lengths.len()) {
                    word_starts.push(index);
                }
            }
            previous_was_word = is_word;
            lengths.push(piece.len() as u8);
            piece_offset += piece.len();
            boundaries.push(piece_offset);
        }
    }
    let mut node = accesskit::Node::new(Role::TextRun);
    node.set_value(line.to_string());
    node.set_character_lengths(lengths);
    node.set_word_starts(word_starts);
    BuiltRun {
        node,
        info: TextRunInfo {
            owner,
            byte_start: start,
            boundaries,
        },
    }
}

/// The position of byte `offset` among `runs`. An offset at a line break's end
/// names the start of the next run, as a caret after `\n` sits on the next line.
/// An offset inside a character snaps back to that character's start.
pub(crate) fn position_of(runs: &[(NodeId, &TextRunInfo)], offset: usize) -> Option<TextPosition> {
    let (last_id, _) = runs.last()?;
    for (index, (id, info)) in runs.iter().enumerate() {
        let end = info.byte_start + info.boundaries.last().copied().unwrap_or(0);
        let is_last = index + 1 == runs.len();
        if offset < end || (is_last && offset <= end) {
            let relative = offset.saturating_sub(info.byte_start);
            let character_index = info
                .boundaries
                .partition_point(|&boundary| boundary <= relative)
                .saturating_sub(1);
            return Some(TextPosition {
                node: *id,
                character_index,
            });
        }
    }
    let (_, info) = runs.last()?;
    Some(TextPosition {
        node: *last_id,
        character_index: info.boundaries.len() - 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs_of(text: &str) -> Vec<BuiltRun> {
        build_runs(NodeId(1), text)
    }

    fn table(built: &[BuiltRun]) -> Vec<(NodeId, &TextRunInfo)> {
        built
            .iter()
            .enumerate()
            .map(|(i, run)| (NodeId(100 + i as u64), &run.info))
            .collect()
    }

    #[test]
    fn an_empty_text_still_has_one_run_for_the_caret() {
        let built = runs_of("");
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].node.value(), Some(""));
        assert!(built[0].node.character_lengths().is_empty());
        let position = position_of(&table(&built), 0).unwrap();
        assert_eq!(position.character_index, 0);
    }

    #[test]
    fn lines_keep_their_break_and_a_trailing_break_adds_an_empty_run() {
        let built = runs_of("ab\ncd\n");
        let values: Vec<_> = built.iter().map(|r| r.node.value().unwrap()).collect();
        assert_eq!(values, ["ab\n", "cd\n", ""]);
        assert_eq!(built[0].node.character_lengths(), &[1, 1, 1]);
    }

    #[test]
    fn characters_are_grapheme_clusters_in_utf8_bytes() {
        // "e" + combining acute is one cluster of 3 bytes; the flag is 8.
        let built = runs_of("e\u{301}🇫🇷x");
        assert_eq!(built[0].node.character_lengths(), &[3, 8, 1]);
    }

    #[test]
    fn word_starts_mark_the_first_character_of_each_word() {
        let built = runs_of("hi there, you");
        assert_eq!(built[0].node.word_starts(), &[0, 3, 10]);
    }

    #[test]
    fn a_caret_after_a_line_break_sits_at_the_start_of_the_next_run() {
        let built = runs_of("ab\ncd");
        let table = table(&built);
        assert_eq!(
            position_of(&table, 3),
            Some(TextPosition {
                node: NodeId(101),
                character_index: 0
            })
        );
        assert_eq!(
            position_of(&table, 2),
            Some(TextPosition {
                node: NodeId(100),
                character_index: 2
            })
        );
        assert_eq!(
            position_of(&table, 5),
            Some(TextPosition {
                node: NodeId(101),
                character_index: 2
            })
        );
    }

    #[test]
    fn positions_round_trip_to_byte_offsets() {
        let text = "héllo\nwörld";
        let built = runs_of(text);
        let mut map = TextRuns::default();
        let ids: Vec<_> = table(&built).iter().map(|(id, _)| *id).collect();
        let positions: Vec<_> = (0..=text.len())
            .filter(|i| text.is_char_boundary(*i))
            .map(|i| (i, position_of(&table(&built), i).unwrap()))
            .collect();
        for (id, run) in ids.into_iter().zip(built) {
            map.insert(id, run.info);
        }
        for (offset, position) in positions {
            assert_eq!(map.offset(&position), Some((NodeId(1), offset)));
        }
    }

    #[test]
    fn a_selection_across_two_owners_is_refused() {
        let mut map = TextRuns::default();
        for (owner, id) in [(NodeId(1), NodeId(10)), (NodeId(2), NodeId(20))] {
            let run = build_runs(owner, "abc").pop().unwrap();
            map.insert(id, run.info);
        }
        let selection = accesskit::TextSelection {
            anchor: TextPosition {
                node: NodeId(10),
                character_index: 0,
            },
            focus: TextPosition {
                node: NodeId(20),
                character_index: 1,
            },
        };
        assert_eq!(map.selection(&selection), None);
    }
}
