use std::sync::OnceLock;

use gpui_util::ResultExt;
use windows::{
    UI::{
        Color,
        ViewManagement::{UIColorType, UISettings},
    },
    Win32::{
        Foundation::*,
        Graphics::{
            Dwm::*,
            Gdi::{
                COLOR_BTNFACE, COLOR_BTNTEXT, COLOR_GRAYTEXT, COLOR_HIGHLIGHT, COLOR_HIGHLIGHTTEXT,
                COLOR_HOTLIGHT, COLOR_WINDOW, COLOR_WINDOWTEXT, GetSysColor, SYS_COLOR_INDEX,
            },
        },
        UI::{
            Accessibility::{HCF_HIGHCONTRASTON, HIGHCONTRASTW, HIGHCONTRASTW_FLAGS},
            WindowsAndMessaging::*,
        },
    },
    core::BOOL,
};

use crate::*;
use gpui::*;

pub(crate) trait HiLoWord {
    fn hiword(&self) -> u16;
    fn loword(&self) -> u16;
    fn signed_hiword(&self) -> i16;
    fn signed_loword(&self) -> i16;
}

impl HiLoWord for WPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    fn signed_hiword(&self) -> i16 {
        ((self.0 >> 16) & 0xFFFF) as i16
    }

    fn signed_loword(&self) -> i16 {
        (self.0 & 0xFFFF) as i16
    }
}

impl HiLoWord for LPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    fn signed_hiword(&self) -> i16 {
        ((self.0 >> 16) & 0xFFFF) as i16
    }

    fn signed_loword(&self) -> i16 {
        (self.0 & 0xFFFF) as i16
    }
}

pub(crate) unsafe fn get_window_long(hwnd: HWND, nindex: WINDOW_LONG_PTR_INDEX) -> isize {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        GetWindowLongPtrW(hwnd, nindex)
    }
    #[cfg(target_pointer_width = "32")]
    unsafe {
        GetWindowLongW(hwnd, nindex) as isize
    }
}

pub(crate) unsafe fn set_window_long(
    hwnd: HWND,
    nindex: WINDOW_LONG_PTR_INDEX,
    dwnewlong: isize,
) -> isize {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        SetWindowLongPtrW(hwnd, nindex, dwnewlong)
    }
    #[cfg(target_pointer_width = "32")]
    unsafe {
        SetWindowLongW(hwnd, nindex, dwnewlong as i32) as isize
    }
}

pub(crate) fn windows_credentials_target_name(url: &str) -> String {
    format!("zed:url={}", url)
}

pub(crate) fn load_cursor(style: CursorStyle) -> Option<HCURSOR> {
    static ARROW: OnceLock<SafeCursor> = OnceLock::new();
    static IBEAM: OnceLock<SafeCursor> = OnceLock::new();
    static CROSS: OnceLock<SafeCursor> = OnceLock::new();
    static HAND: OnceLock<SafeCursor> = OnceLock::new();
    static SIZEWE: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENS: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENWSE: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENESW: OnceLock<SafeCursor> = OnceLock::new();
    static NO: OnceLock<SafeCursor> = OnceLock::new();
    let (lock, name) = match style {
        CursorStyle::IBeam | CursorStyle::IBeamCursorForVerticalLayout => (&IBEAM, IDC_IBEAM),
        CursorStyle::Crosshair => (&CROSS, IDC_CROSS),
        CursorStyle::PointingHand | CursorStyle::DragLink => (&HAND, IDC_HAND),
        CursorStyle::ResizeLeft
        | CursorStyle::ResizeRight
        | CursorStyle::ResizeLeftRight
        | CursorStyle::ResizeColumn => (&SIZEWE, IDC_SIZEWE),
        CursorStyle::ResizeUp
        | CursorStyle::ResizeDown
        | CursorStyle::ResizeUpDown
        | CursorStyle::ResizeRow => (&SIZENS, IDC_SIZENS),
        CursorStyle::ResizeUpLeftDownRight => (&SIZENWSE, IDC_SIZENWSE),
        CursorStyle::ResizeUpRightDownLeft => (&SIZENESW, IDC_SIZENESW),
        CursorStyle::ResizeUpLeft | CursorStyle::ResizeDownRight => (&SIZENWSE, IDC_SIZENWSE),
        CursorStyle::ResizeUpRight | CursorStyle::ResizeDownLeft => (&SIZENESW, IDC_SIZENESW),
        CursorStyle::OperationNotAllowed => (&NO, IDC_NO),
        _ => (&ARROW, IDC_ARROW),
    };
    Some(
        *(*lock.get_or_init(|| {
            HCURSOR(
                unsafe { LoadImageW(None, name, IMAGE_CURSOR, 0, 0, LR_DEFAULTSIZE | LR_SHARED) }
                    .log_err()
                    .unwrap_or_default()
                    .0,
            )
            .into()
        })),
    )
}

/// This function is used to configure the dark mode for the window built-in title bar.
pub(crate) fn configure_dwm_dark_mode(hwnd: HWND, appearance: WindowAppearance) {
    let dark_mode_enabled: BOOL = match appearance {
        WindowAppearance::Dark | WindowAppearance::VibrantDark => true.into(),
        WindowAppearance::Light | WindowAppearance::VibrantLight => false.into(),
    };
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            &dark_mode_enabled as *const _ as _,
            std::mem::size_of::<BOOL>() as u32,
        )
        .log_err();
    }
}

#[inline]
pub(crate) fn logical_point(x: f32, y: f32, scale_factor: f32) -> Point<Pixels> {
    Point {
        x: px(x / scale_factor),
        y: px(y / scale_factor),
    }
}

// https://learn.microsoft.com/en-us/windows/apps/desktop/modernize/apply-windows-themes
#[inline]
pub(crate) fn system_appearance() -> Result<WindowAppearance> {
    let ui_settings = UISettings::new()?;
    let foreground_color = ui_settings.GetColorValue(UIColorType::Foreground)?;
    // If the foreground is light, then is_color_light will evaluate to true,
    // meaning Dark mode is enabled.
    if is_color_light(&foreground_color) {
        Ok(WindowAppearance::Dark)
    } else {
        Ok(WindowAppearance::Light)
    }
}

/// Reads Windows High Contrast (contrast themes on Windows 11) from
/// `SPI_GETHIGHCONTRAST`, the setting Chromium maps to `forced-colors: active`.
pub(crate) fn system_contrast_preference() -> Result<ContrastPreference> {
    let mut high_contrast = HIGHCONTRASTW {
        cbSize: std::mem::size_of::<HIGHCONTRASTW>() as u32,
        ..Default::default()
    };
    unsafe {
        SystemParametersInfoW(
            SPI_GETHIGHCONTRAST,
            high_contrast.cbSize,
            Some(&mut high_contrast as *mut HIGHCONTRASTW as *mut _),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )?;
    }
    Ok(contrast_preference_from_flags(high_contrast.dwFlags))
}

/// Reads the active contrast theme's colors, the `GetSysColor` entries
/// Chromium resolves CSS system colors from under `forced-colors: active`.
pub(crate) fn system_forced_colors() -> SystemColors {
    let color = |index: SYS_COLOR_INDEX| colorref_to_rgba(unsafe { GetSysColor(index) });
    SystemColors {
        canvas: color(COLOR_WINDOW),
        canvas_text: color(COLOR_WINDOWTEXT),
        link_text: color(COLOR_HOTLIGHT),
        highlight: color(COLOR_HIGHLIGHT),
        highlight_text: color(COLOR_HIGHLIGHTTEXT),
        button_face: color(COLOR_BTNFACE),
        button_text: color(COLOR_BTNTEXT),
        gray_text: color(COLOR_GRAYTEXT),
    }
}

/// A `COLORREF` is `0x00BBGGRR`.
fn colorref_to_rgba(colorref: u32) -> Rgba {
    let [r, g, b, _] = colorref.to_le_bytes();
    rgb(u32::from_be_bytes([0, r, g, b]))
}

fn contrast_preference_from_flags(flags: HIGHCONTRASTW_FLAGS) -> ContrastPreference {
    if flags.contains(HCF_HIGHCONTRASTON) {
        ContrastPreference::Forced
    } else {
        ContrastPreference::NoPreference
    }
}

#[inline(always)]
fn is_color_light(color: &Color) -> bool {
    ((5 * color.G as u32) + (2 * color.R as u32) + color.B as u32) > (8 * 128)
}

#[cfg(test)]
mod tests {
    // Not `super::*`: that glob pulls in `gpui::test`, which shadows `#[test]`.
    use super::{
        ContrastPreference, HCF_HIGHCONTRASTON, HIGHCONTRASTW_FLAGS, colorref_to_rgba,
        contrast_preference_from_flags, rgb,
    };

    #[test]
    fn colorref_is_blue_green_red() {
        assert_eq!(colorref_to_rgba(0x00_33_22_11), rgb(0x112233));
        assert_eq!(colorref_to_rgba(0x00_ff_ff_00), rgb(0x00ffff));
    }

    #[test]
    fn high_contrast_flag_maps_to_forced_colors() {
        assert_eq!(
            contrast_preference_from_flags(HIGHCONTRASTW_FLAGS(0)),
            ContrastPreference::NoPreference
        );
        assert_eq!(
            contrast_preference_from_flags(HCF_HIGHCONTRASTON),
            ContrastPreference::Forced
        );
        // HCF_AVAILABLE (2) and HCF_HOTKEYACTIVE (4) alone mean the feature is
        // reachable, not on.
        assert_eq!(
            contrast_preference_from_flags(HIGHCONTRASTW_FLAGS(0x2 | 0x4)),
            ContrastPreference::NoPreference
        );
        assert_eq!(
            contrast_preference_from_flags(HIGHCONTRASTW_FLAGS(0x1 | 0x2)),
            ContrastPreference::Forced
        );
    }
}
