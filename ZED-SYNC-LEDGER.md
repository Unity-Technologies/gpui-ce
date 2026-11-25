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
