use crate::{Action, App, KeyBinding, KeyBindingContextPredicate, SharedString};
use std::{any::TypeId, collections::HashMap};

/// A generic collection of `Action` -> keystrokes which can be bound under a shared key-context.
/// Use `as_keybindings` to create the `KeyBinding`s to provide to `App::bind_keys`.
///
/// ```rust
/// # use gpui::{ActionBindingCollection, Action};
/// gpui::actions!([MyAction]);
/// # #[gpui::test]
/// # fn test(app: &mut gpui::TestAppContext) {
/// app.update(|cx| {
///     let bindings = ActionBindingCollection::default()
///         .with::<MyAction>("enter")
///         .as_keybindings(None, cx);
///     cx.bind_keys(bindings);
/// });
/// # }
/// ```
#[derive(Default)]
pub struct ActionBindingCollection {
    entries: HashMap<TypeId, ActionBindingEntry>,
}

#[derive(Debug)]
struct ActionBindingEntry {
    // The action used to build Keybinding
    action: Box<dyn Action>,
    // The keystrokes that trigger the action
    // NOTE: could limit allocations by utilizing smallvec
    keystrokes: Vec<SharedString>,
}

impl ActionBindingCollection {
    /// Adds a keystroke to an action in the collection.
    pub fn with<A: Action + Default>(mut self, keystrokes: impl Into<SharedString>) -> Self {
        let action_id = TypeId::of::<A>();
        let entry = self.entries.entry(action_id);
        let entry = entry.or_insert_with(|| ActionBindingEntry {
            action: Box::new(A::default()),
            keystrokes: Vec::with_capacity(5),
        });
        entry.keystrokes.push(keystrokes.into());
        self
    }

    /// Creates the `KeyBinding`s which represent all action+keystrokes in this collection bound to the same key-context.
    /// This sequence can be provided to `App::bind_keys`, and any elements which use the same `InteractiveElement::key_context` will receive the bound actions.
    ///
    /// Keystrokes are parsed for the host OS that `cx` reports, so `secondary-` resolves there.
    pub fn as_keybindings(&self, context: Option<&str>, cx: &App) -> Vec<KeyBinding> {
        let context_predicate =
            context.map(|context| KeyBindingContextPredicate::parse(context).unwrap().into());
        let iter = self.entries.iter();
        let iter = iter.map(move |(_, entry)| {
            let action = &entry.action;
            let context_predicate = context_predicate.clone();
            entry.keystrokes.iter().map(move |keystrokes| {
                let keybinding = KeyBinding::load(
                    keystrokes.as_str(),
                    action.boxed_clone(),
                    context_predicate.clone(),
                    false,
                    None,
                    cx,
                );
                keybinding.expect("failed to load keybinding")
            })
        });
        iter.flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAppContext;

    actions!([TestAction]);

    #[gpui::test]
    fn test_default_empty(_cx: &mut TestAppContext) {
        assert!(ActionBindingCollection::default().entries.is_empty());
    }

    #[gpui::test]
    fn test_single_binding(cx: &mut TestAppContext) {
        let collection = ActionBindingCollection::default();
        let collection = collection.with::<TestAction>("enter");

        let entry = collection.entries.get(&TypeId::of::<TestAction>());
        let entry = entry.map(|entry| &entry.keystrokes);
        assert_eq!(entry, Some(&vec![SharedString::new_static("enter")]));

        let out_bindings = cx.update(|cx| collection.as_keybindings(None, cx));
        assert_eq!(out_bindings.len(), 1);
    }

    #[gpui::test]
    fn test_single_action_many_keys(cx: &mut TestAppContext) {
        let collection = ActionBindingCollection::default();
        let collection = collection.with::<TestAction>("enter");
        let collection = collection.with::<TestAction>("escape");

        let entry = collection.entries.get(&TypeId::of::<TestAction>());
        let entry = entry.map(|entry| &entry.keystrokes);
        assert_eq!(
            entry,
            Some(&vec![
                SharedString::new_static("enter"),
                SharedString::new_static("escape")
            ])
        );

        let out_bindings = cx.update(|cx| collection.as_keybindings(None, cx));
        assert_eq!(out_bindings.len(), 2);
    }

    #[gpui::test]
    fn test_single_action_many_keys_context(cx: &mut TestAppContext) {
        let collection = ActionBindingCollection::default();
        let collection = collection.with::<TestAction>("enter");
        let collection = collection.with::<TestAction>("escape");
        let out_bindings = cx.update(|cx| collection.as_keybindings(Some("test_content"), cx));
        assert_eq!(out_bindings.len(), 2);
    }

    #[gpui::test]
    fn test_secondary_resolves_for_the_reported_host(cx: &mut TestAppContext) {
        let collection = ActionBindingCollection::default().with::<TestAction>("secondary-s");
        let modifiers_on = |cx: &mut TestAppContext, os| {
            cx.set_platform_info(crate::PlatformInfo {
                os,
                target: crate::PlatformTarget::Native,
            });
            let bindings = cx.update(|cx| collection.as_keybindings(None, cx));
            bindings[0].keystrokes()[0].modifiers().clone()
        };

        let mac = modifiers_on(cx, crate::OperatingSystem::Mac);
        assert!(mac.platform && !mac.control);
        let linux = modifiers_on(cx, crate::OperatingSystem::Linux);
        assert!(linux.control && !linux.platform);
    }
}
