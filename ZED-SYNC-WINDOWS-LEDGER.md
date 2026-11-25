# Zed sync ledger: Windows-only gpui commits since b1a7ef0

Fork branch `unity/HUB-0000-zed-sync-20261124-windows`, cut from `unity/hub-consolidated`.

| Zed PR | Zed commit | Verdict | Reason |
|---|---|---|---|
| #64001 | d2074f4e15 | skip | Fork has no `PowerRequest` / `PowerCreateRequest` code |
| #64270 | de2c85f255 | skip | Fork uses `oneshot` elsewhere in platform.rs, so no unused import to fix |
| #57543 | b9419ae728 | skip | No HLSL in fork; WGSL `gpui_render` transfer functions are already correct |
| #62338 | ac6818fb2d | skip | DirectWrite shaping only; fork shapes with Parley |
| #64735 | 8720fd1d09 | take | Clean pick: null `UserName` / `CredentialBlob` from `CredReadW` |
| #64911 | decbf641b1 | adapt | Fork already scrubs NULs; took only the regression test |
| #65628 | e462f873c6 | adapt | Fork validated cursor and device-lost already; added token check to force-update, dropped `WM_GPUI_KEYDOWN` |
| #66159 | 979cd3fdf0 | skip | DirectWrite shaping only; fork shapes with Parley |
| #55739 | 9cb5f90e3c | skip | Fork already drops same-position `WM_MOUSEMOVE` (`last_mouse_position`) |
| #66368 | bc6fc11740 | skip | Fork already calls `CreateTargetForHwnd(hwnd, false)` |
| #66468 | 0294cfb58e | skip | DirectWrite shaping only; fork shapes with Parley |
| #66588 | 4a6121bb0f | skip | Fork already composites COLR layers on the CPU (Direct2D on WIC) with union bounds |
| #66855 | 78f5d7446c | skip | Fork has no synchronous `display_changed` callback |
| #66945 | 2eff23878e | adapt | CRLF on write, LF on read, kept fork's NUL helper and hashed normalized text |
| #66001 | 58fea3d335 | skip | Fork never calls `GetSystemFontCollection`; fonts come from Parley |
| #67404 | 990491f0ea | adapt | Removed NVIDIA / AMD DLL queries from fork's directx_renderer.rs; DXGI version for all vendors |

Totals: 1 take, 4 adapt, 11 skip.
