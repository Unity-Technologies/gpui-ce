# Zed sync ledger (since b1a7ef0)

| Zed SHA | Zed PR | Subject | Decision | Reason |
|---|---|---|---|---|
| f0d8b0b0cd | #63614 | gpui: Add an option to keep hover listeners active after keyboard input (#63614) | adapt | fork routes on_hover through hover regions; mode applies to region listeners |
| d85eade7ab | #63343 | which_key: Add a pending key binding indicator (#63343) | take | gpui pending-binding APIs; which_key UI is Zed app |
| 3ce72bab20 | #63696 | gpui: Add `Hitbox::is_hovered_at` (#63696) | take | clean |
| 206a863a95 | #63235 | gpui_web: Update example to use `run_embedded()` instead of `run()` (#63235) | skip | gpui_web example only |
| 1057c2cf3d | #63606 | Redact environment secrets from dev container `docker exec` logs (#63606) | skip | Zed dev-container helper, unused by fork |
| 9bda6b4e03 | #63775 | gpui_web: Disable native canvas touch callouts (#63775) | skip | gpui_web only |
| c7801b0c44 | #63759 | macos: Honor "Tiled windows have margins" for the Fill titlebar action (#63759) | adapt | objc2 Bool and NIL in fork's macOS window |
| 13e5c99ab7 | #63877 | gpui_web: Disable native canvas selection (#63877) | skip | gpui_web only |
| 4a217d5316 | #63776 | gpui_linux: Disable IME in project panel when no input handler on Wayland (#63776) | take | clean |
| c8c07aea5c | #63580 | Add `disable_ai` check for "Agent Panel" `MenuItem` (#63580) | take | TestPlatform stores menus; import list merged |
| 20fa2fa809 | #63882 | Reduce LLVM IR bloat one last time (#63882) | take | clean |
| 5896274168 | #63934 | gpui: Replace shared asset futures with owning cache entries (#63934) | skip | DEFERRED: breaks fetch_asset/ImageCacheItem API that fork's fetch_asset docs and Hub scoped_image_cache use; needs a paired Hub migration |
| 33c6212b05 | #58886 | Restore macOS windows to their original Space after restart (#58886) | adapt | macOS window state archive/restore rewritten for objc2 (fork has no cocoa) |
| 450775244f | #53130 | gpui: Prevent idle sleep during AI response streaming (#53130) | adapt | fork import lists; macOS MacActivity on objc2 imports; web/windows parts kept |
| 907b55f7ea | #63945 | gpui_macos: Migrate `register_url_scheme` to `objc2` (#63945) | skip | fork's gpui_macos already fully on objc2 |
| d3865b099e | #63994 | gpui_macos: Migrate `prompt_for_path` to `objc2` (#63994) | skip | fork's gpui_macos already fully on objc2 |
| 6ad3c7f278 | #63989 | gpui_web: Simplify keyboard focus model (#63989) | adapt | gpui visual-viewport API + web parts; fork test-window and test imports merged |
| 86b2cf9662 | #64031 | gpui_macos: Migrate `get_sources` to `objc2` (#64031) | skip | fork's gpui_macos already fully on objc2 |
| 72c53bf0aa | #64033 | gpui_macos: Migrate `start_external_drag` to `objc2` (#64033) | skip | fork's gpui_macos already fully on objc2 |
| 5a773a406e | #64041 | gpui_web: Disable autoscroll (#64041) | skip | gpui_web only |
| 1a84d5d92b | #64038 | project_panel: Fix dragged folders remaining highlighted even after the drag has ended (#64038) | adapt | fork import lists |
| d9e1c024f3 | #63948 | gpui: Long-press to activate tooltips (#63948) | adapt | fork's flattened test imports and hover-region docs |
| 3cef31688a | #64077 | Upgrade ACP to 2.1 and futures to 0.3.34 (#64077) | skip | executor wrapper spawn_dedicated already absent in fork; only gpui hunk |
| d12e456be7 | #64034 | util: Raise open file soft limit at startup on Unix (#64034) | skip | Zed app startup helper, unused by fork |
| 5b4a2153a0 | #64095 | gpui: List only available font families in suggestions (#64095) | skip | fork TextSystem already delegates; Parley owns family list (keeps .SystemUIFont alias); cosmic test n/a |
| fb38178d74 | #64099 | gpui: Fix native object accumulation in headless Metal rendering (#64099) | adapt | autoreleasepool wrapped around fork's headless render bodies; objc2 dep already present |
| 3db02c2c77 | #64107 | gpui: Add is_visible and on_visibility_changed API on Window (#64107) | adapt | macOS sleep observer on the objc2 delegate; App sleep handler registered lazily and borrow-safe like the fork's wake handler; Windows/Linux merged but not compiled locally |
| a7c7219d5b | #64113 | gpui_linux: Avoid local XKB files for server keymaps (#64113) | take |  |
| 87a1ea30e8 | #64122 | gpui_linux: Discard stale Wayland file drag reads (#64122) | take |  |
| 7960b2a7c9 | #64110 | gpui: Add Canvas font fallback for web text (#64110) | skip | gpui_web text system on cosmic-text; fork's web text goes through Parley and has no web text_system.rs |
| 25b5569dd2 | #46744 | project_panel: Add Menu key support to project panel context menu (#46744) | take |  |
| f792c2d72d | #64185 | gpui: Add `ShapedLineCursor` API (#64185) | skip | DEFERRED: cursor walks glyph runs and decoration runs; fork's Parley LineLayout has neither (and no ShapedLine::split_at). Needs a Parley design |
| d62802d456 | #64192 | Fix Trunk link in GPUI examples documentation (#64192) | skip | fork has no crates/gpui/examples/README.md |
| 9e6e14160c | #63498 | gpui: Support dynamic font installation (#63498) | skip | DEFERRED: missing-glyph reports and font generations live in upstream text_system.rs and cosmic_text_system; fork's Parley text system has another layout. Needs a Parley design |
| b68add5b37 | #64204 | gpui: Remove mutex from dynamic font loading (#64204) | skip | DEFERRED: follow-up to 9e6e14160c (dynamic font installation), not ported |
| 07df438626 | #64237 | Remove `ztracing` dependency from GPUI (#64237) | skip | fork's gpui already has no ztracing dependency or instrument attributes |
| db7f9ceed7 | #64234 | gpui_macos: Improve mouse tracking for normal windows (#64234) | adapt | objc2 Bool for setAcceptsMouseMovedEvents; added NSTrackingArea app-kit feature and AnyThread import |
| 5c9efb75cf | #64298 | gpui: Avoid main-thread waits when unlocking web task queues (#64298) | adapt | kept fork's sender-count recheck around the std Condvar wait; web stress-test example files left out |
| f6838a7c4a | #64301 | gpui_macos: Fix hover feedback when another application is focused (#64301) | adapt | objc2 Bool for setAcceptsMouseMovedEvents |
| f0fb48c71a | #64269 | gpui: Make ForegroundJournal aware of sleep and window visibility  (#64269) | take | Zed app and input_latency_ui parts not in fork |
| b0f53ad23b | #64309 | Optimize GPUI inspector's runtime cost by an order of magnitude (#64309) | adapt | gpui_platform features and dev-deps merged into fork's; bench aliases gpui_ce_platform as gpui_platform for the bench macro; inspector_ui/lsp parts are Zed app |
| 9930d2c2a3 | #64260 | Share generics better and strip rodio audio features (#64260) | skip | flate re-export serves Zed's rust-embed compression feature, which the fork does not enable |
| 0dea8c63d9 | #63933 | editor: Render point diagnostics without glyphs (#63933) | skip | editor point-diagnostic squiggles; gpui half is a ShapedLine underline handler over upstream glyph runs (not Parley) plus editor-only underline-exclusion helpers |
| 87f65de697 | #64319 | gpui: Fix web fallback for text-default emoji (#64319) | skip | gpui_web cosmic/canvas text fallback; fork's web text goes through Parley |
| 6c9d10cb83 | #64337 | gpui_web: Flatten webgl texel loading (#64337) | skip | FOLLOW-UP: fork's WebGL/GLES data-texture decoder is generated by gpui_render build.rs with a different quad layout; the unrolled-fetch perf idea needs a port into that generator |
| 4b47ceb9d3 | #64251 | gpui_macos: Migrate screen capture to `objc2` (#64251) | skip | fork's screen_capture already on objc2 SCK with its own queue-depth/delegate/YUV changes; upstream drops the media crate (fork keeps gpui_media) and changes the public ScreenCaptureFrame type |
| 0ef92145ff | #64331 | gpui: Improve `PlatformAtlas::get_or_insert_with` (#64331) | skip | DEFERRED: fork atlases diverged (glyph cache entries, upload validation, texture-size limits, texture identity) in all four backends; the shared AtlasState refactor needs a fork redesign, and its gain is one clone and one map lookup |
| aec7395e30 | #64340 | Fix cargo shear warnings (#64340) | skip | cargo-shear test/doctest toggles for Zed's CI; fork's published crates keep their own lib test settings |
| cf4deb270f | #62073 | gpui: Fix centered window bounds to display settings header (#62073) | take |  |
| c24e309d93 | #62525 | gpui: Fix manifest path for cross-compilation (#62525) | take |  |
| 94c997e06f | #64433 | gpui: Fix debug selectors with cached views (#64433) | skip | fork already replays debug selectors for cached views (debug_bounds_log) |
| 0d08af1e53 | #64261 | Share GPUI and scheduler implementations (#64261) | adapt | fork's view.rs kept as is (direction/bidi layout probing diverged; helper extraction not ported); inline attrs placed on fork's documented methods; also adds trait imports the #64309 inspector tests need in fork's window.rs |
| 01448f565e | #55727 | Fix Windows task path handling for worktree variables with spaces (#55727) | take | util fix kept in fork's copy |
| 887149bbdd | #64525 | Ignore invalid file patterns individually (#64525) | take | util fix kept in fork's copy |
| a51d23c93d | #56553 | Windows: Fix files opened by UNC path getting no language servers (#56553) | take | util fix kept in fork's copy; #[perf] as #[test] |
| bcf6582ce3 | #62473 | gpui_util: Respect custom Scoop installation paths (#62473) | take |  |
| a434bb7ee2 | #64434 | wgpu: Cache atlas bind groups (#64434) | skip | fork's wgpu renderer already caches atlas bind groups keyed by texture identity (buffers.rs) |
| f25434f3c5 | #64570 | gpui_linux: Force a full render on X11 Expose after GPU recovery (#64570) | take | applied clean; Linux not compiled on this host |
| ba42ab9d98 | #64537 | gpui: Fix keystroke interception for standalone modifiers (#64537) | adapt | fork's activate() for activate_window(); fork's observe_keystrokes docs kept plus the release note |
| 9183ba4c55 | #64365 | ui: Fix context menu selecting first entry on open (#64365) | adapt | is_enabled added beside fork's forced a11y flag; test uses fork's 4-arg A11y::new |
| b89471c52d | #64641 | gpui_apple: Remove `block` as a direct dependency (#64641) | adapt | RcBlock port applied to fork's release_instance_buffer_when_complete |
| e71963a599 | #64672 | gpui: Fix panic when hit testing past a wrapped line's width (#64672) | skip | fork has no WrappedLineLayout; Parley owns wrapped-line hit testing (cosmic-text bug) |
| 382ff7a64b | #64175 | gpui: Stop synthetic mouse drags after button release on macOS (#64175) | take |  |
| e9ddcdaf98 | #64702 | gpui: Make create ElementId more easier from u64 (#64702) | take |  |
| e91b82c106 | #64718 | gpui_wgpu: Add Linux headless rendering (#64718) | skip | fork already has WGPU headless rendering on Linux/FreeBSD/Windows (WgpuHeadlessRenderer, offscreen target reuse); upstream's Result-returning renderer factory not adopted |
| bda9c0bd43 | #64788 | markdown: Fix search highlights hiding text with opaque theme colors (#64788) | adapt | TextLayout paint_background/paint_foreground over fork's Parley document; line.rs aligned paint-bounds fix is upstream glyph-run code, not ported |
| e683fd7b46 | #64753 | gpui: Add bench_metrics for multi-metric Criterion benchmarks (#64753) | skip | DEFERRED: adds a new 1.8k-line bench_metrics workspace crate (perf counters) that the fork does not carry; GPUI bench harness keeps wall-time Criterion |
| 4f70d91bda | #63468 | editor: Add support for customizable soft wrap indentation (#63468) | skip | editor soft-wrap indent in LineWrapper; fork has no LineWrapper (Parley wraps) |
| 98f75a23bc | #64862 | Remove some unnecessary dependencies (#64862) | adapt | x11-clipboard and take-until were unused in the fork too; dropped from crates and workspace |
| 86c20cc911 | #64859 | gpui: Add character input preference to keystroke events (#64859) | adapt | fork's activate() for activate_window(); KeystrokeEvent docs merged |
| e773a56d6c | #64882 | gpui_web: Preserve iOS keyboard edits and browser window activity (#64882) | skip | gpui_web only |
| 3135cdf8dc | #64842 | gpui: Add randomized element tree benchmarks and fix Linux benchmark leaks (#64842) | skip | fork already carries the tree, queue wake fix, Linux dispatcher stop and bench_text_system (HUB-8799, gpui-ce #56); upstream bench file needs bench_metrics (deferred #64753) |
| 7232ba04ef | #64891 | gpui: Drop foreground measurements that span an unnotified system suspend (#64891) | adapt | windows feature list merged into fork's; gpui-ce checked for macOS and the Linux target |
| 0f9c923e67 | #64944 | Add a GPUI-owned hang monitor and make hang_telemetry a consumer (#64944) | adapt | fork's spawn docs and asset_registry kept around the new hang monitor entry points |
| 4c0ead4a3e | #64993 | gpui: Fix flaky hang monitor shutdown test (#64993) | take |  |
| d284cbbc87 | #64996 | gpui: Fix flaky foreground hang detection test (#64996) | take |  |
| 40180d9c40 | #64990 | gpui_apple: Share dispatcher and Metal renderer with iOS (#64990) | skip | iOS enablement: moves dispatcher and Metal renderer into gpui_apple for iOS; fork ships desktop only and its macOS layout differs |
| 71456c40f3 | #64935 | gpui: Hide disabled bindings from the which-key popup (#64935) | take |  |
| 95cd535a5f | #65043 | gpui: Make test-support safe to enable in every build (#65043) | adapt | Took opt-in leak detection (build.rs cfg), is_test() and util home_dir; kept fork Cargo features; Metal framebuffer_only stays on for test, bench-support and render-to-image, test-support only in debug |
