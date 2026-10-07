# Why this copy exists

This is `accesskit_unix` 0.24.0 from crates.io, vendored so `gpui_linux` can
carry one fix that no released AccessKit has yet. The first commit that adds
this directory is the published crate unchanged (plus the licence files the
published crate leaves out); every later change is a separate commit.

**Accessibility inside a Flatpak sandbox.** AccessKit only connects to the
AT-SPI bus once `org.a11y.Status.IsEnabled`, read from `org.a11y.Bus` on the
session bus, turns true. Flatpak hides `org.a11y.Bus` from the sandbox and
hands the app the accessibility bus directly in `AT_SPI_BUS_ADDRESS`
(`unix:path=/run/flatpak/at-spi-bus`) instead, so the property is never read,
the adapter never activates, and screen readers never see the app. The patch
in `src/context.rs` treats accessibility as enabled when that bus is provided
but the toggle can't be read, as GTK and Qt do.

Drop this directory, and point the workspace's `accesskit_unix` back at
crates.io, once an AccessKit release handles that case itself.
