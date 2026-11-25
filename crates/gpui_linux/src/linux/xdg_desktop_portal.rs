//! Provides a [calloop] event source from [XDG Desktop Portal] events
//!
//! This module uses the [ashpd] crate

use ashpd::desktop::settings::{
    APPEARANCE_NAMESPACE, CONTRAST_KEY, ColorScheme, Contrast, Settings,
};
use calloop::channel::Channel;
use calloop::{EventSource, Poll, PostAction, Readiness, Token, TokenFactory};
use smol::stream::StreamExt;

use gpui::{BackgroundExecutor, ContrastPreference, WindowAppearance};

pub enum Event {
    WindowAppearance(WindowAppearance),
    #[cfg_attr(feature = "x11", allow(dead_code))]
    CursorTheme(String),
    #[cfg_attr(feature = "x11", allow(dead_code))]
    CursorSize(u32),
    ButtonLayout(String),
    Contrast(ContrastPreference),
    #[cfg_attr(feature = "x11", allow(dead_code))]
    TextScaleFactor(f32),
}

pub struct XDPEventSource {
    channel: Channel<Event>,
}

impl XDPEventSource {
    pub fn new(executor: &BackgroundExecutor) -> Self {
        let (sender, channel) = calloop::channel::channel();

        let background = executor.clone();

        executor
            .spawn(async move {
                let settings = Settings::new().await?;

                if let Ok(initial_appearance) = settings.color_scheme().await {
                    sender.send(Event::WindowAppearance(
                        window_appearance_from_color_scheme(initial_appearance),
                    ))?;
                }
                if let Ok(initial_theme) = settings
                    .read::<String>("org.gnome.desktop.interface", "cursor-theme")
                    .await
                {
                    sender.send(Event::CursorTheme(initial_theme))?;
                }

                // If u32 is used here, it throws invalid type error
                if let Ok(initial_size) = settings
                    .read::<i32>("org.gnome.desktop.interface", "cursor-size")
                    .await
                {
                    sender.send(Event::CursorSize(initial_size as u32))?;
                }

                if let Ok(initial_layout) = settings
                    .read::<String>("org.gnome.desktop.wm.preferences", "button-layout")
                    .await
                {
                    sender.send(Event::ButtonLayout(initial_layout))?;
                }

                if let Ok(initial_scale) = settings
                    .read::<f64>("org.gnome.desktop.interface", "text-scaling-factor")
                    .await
                    && let Some(scale) = text_scale_factor(initial_scale)
                {
                    sender.send(Event::TextScaleFactor(scale))?;
                }

                if let Ok(initial_contrast) = settings.contrast().await {
                    sender.send(Event::Contrast(contrast_preference_from_portal(
                        initial_contrast,
                    )))?;
                }

                // `receive_contrast_changed` returns a stream that borrows `settings`, which
                // cannot move into a 'static task; this form does not borrow it.
                if let Ok(mut contrast_changed) = settings
                    .receive_setting_changed_with_args::<Contrast>(
                        APPEARANCE_NAMESPACE,
                        CONTRAST_KEY,
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(contrast) = contrast_changed.next().await {
                                // Skip an unparsable value, as ashpd's own helper does.
                                let Ok(contrast) = contrast else { continue };
                                sender.send(Event::Contrast(contrast_preference_from_portal(
                                    contrast,
                                )))?;
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                if let Ok(mut cursor_theme_changed) = settings
                    .receive_setting_changed_with_args(
                        "org.gnome.desktop.interface",
                        "cursor-theme",
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(theme) = cursor_theme_changed.next().await {
                                let theme = theme?;
                                sender.send(Event::CursorTheme(theme))?;
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                if let Ok(mut cursor_size_changed) = settings
                    .receive_setting_changed_with_args::<i32>(
                        "org.gnome.desktop.interface",
                        "cursor-size",
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(size) = cursor_size_changed.next().await {
                                let size = size?;
                                sender.send(Event::CursorSize(size as u32))?;
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                if let Ok(mut button_layout_changed) = settings
                    .receive_setting_changed_with_args(
                        "org.gnome.desktop.wm.preferences",
                        "button-layout",
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(layout) = button_layout_changed.next().await {
                                let layout = layout?;
                                sender.send(Event::ButtonLayout(layout))?;
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                if let Ok(mut text_scale_changed) = settings
                    .receive_setting_changed_with_args::<f64>(
                        "org.gnome.desktop.interface",
                        "text-scaling-factor",
                    )
                    .await
                {
                    let sender = sender.clone();
                    background
                        .spawn(async move {
                            while let Some(scale) = text_scale_changed.next().await {
                                if let Some(scale) = text_scale_factor(scale?) {
                                    sender.send(Event::TextScaleFactor(scale))?;
                                }
                            }
                            anyhow::Ok(())
                        })
                        .detach();
                }

                let mut appearance_changed = settings.receive_color_scheme_changed().await?;
                while let Some(scheme) = appearance_changed.next().await {
                    sender.send(Event::WindowAppearance(
                        window_appearance_from_color_scheme(scheme),
                    ))?;
                }

                anyhow::Ok(())
            })
            .detach();

        Self { channel }
    }
}

impl EventSource for XDPEventSource {
    type Event = Event;
    type Metadata = ();
    type Ret = ();
    type Error = anyhow::Error;

    fn process_events<F>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut callback: F,
    ) -> Result<PostAction, Self::Error>
    where
        F: FnMut(Self::Event, &mut Self::Metadata) -> Self::Ret,
    {
        self.channel.process_events(readiness, token, |evt, _| {
            if let calloop::channel::Event::Msg(msg) = evt {
                (callback)(msg, &mut ())
            }
        })?;

        Ok(PostAction::Continue)
    }

    fn register(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.register(poll, token_factory)?;

        Ok(())
    }

    fn reregister(
        &mut self,
        poll: &mut Poll,
        token_factory: &mut TokenFactory,
    ) -> calloop::Result<()> {
        self.channel.reregister(poll, token_factory)?;

        Ok(())
    }

    fn unregister(&mut self, poll: &mut Poll) -> calloop::Result<()> {
        self.channel.unregister(poll)?;

        Ok(())
    }
}

const MIN_TEXT_SCALE_FACTOR: f32 = 0.5;
const MAX_TEXT_SCALE_FACTOR: f32 = 3.;

fn text_scale_factor(value: f64) -> Option<f32> {
    if !value.is_finite() || value <= 0. {
        return None;
    }
    Some((value as f32).clamp(MIN_TEXT_SCALE_FACTOR, MAX_TEXT_SCALE_FACTOR))
}

/// `org.freedesktop.appearance` `contrast`: `0` no preference, `1` high.
/// Chromium maps the same key to `prefers-contrast: more`.
fn contrast_preference_from_portal(contrast: Contrast) -> ContrastPreference {
    match contrast {
        Contrast::High => ContrastPreference::More,
        Contrast::NoPreference => ContrastPreference::NoPreference,
    }
}

fn window_appearance_from_color_scheme(cs: ColorScheme) -> WindowAppearance {
    match cs {
        ColorScheme::PreferDark => WindowAppearance::Dark,
        ColorScheme::PreferLight => WindowAppearance::Light,
        ColorScheme::NoPreference => WindowAppearance::Light,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_scale_factor_is_clamped_to_a_usable_range() {
        assert_eq!(text_scale_factor(1.0), Some(1.0));
        assert_eq!(text_scale_factor(1.25), Some(1.25));
        assert_eq!(text_scale_factor(2.0), Some(2.0));
        assert_eq!(text_scale_factor(0.1), Some(0.5), "clamped up");
        assert_eq!(text_scale_factor(10.0), Some(3.0), "clamped down");
        assert_eq!(text_scale_factor(0.0), None, "not a factor");
        assert_eq!(text_scale_factor(-1.0), None);
        assert_eq!(text_scale_factor(f64::NAN), None);
        assert_eq!(text_scale_factor(f64::INFINITY), None);
    }

    #[test]
    fn portal_contrast_maps_to_contrast_preference() {
        assert_eq!(
            contrast_preference_from_portal(Contrast::High),
            ContrastPreference::More
        );
        assert_eq!(
            contrast_preference_from_portal(Contrast::NoPreference),
            ContrastPreference::NoPreference
        );
    }
}
