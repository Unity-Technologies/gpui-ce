//! Renders a `TestAppContext::with_platform` window with this platform's headless
//! renderer and a real text system. It needs a GPU adapter, so it runs only when
//! `GPUI_RUN_RENDERING_TESTS` is set.

use gpui::{
    Context, IntoElement, ParentElement as _, Render, Styled as _, TestAppContext, TestDispatcher,
    VisualTestContext, Window, div, px, rgb, size, white,
};
use gpui_fonts::IBM_PLEX;
use gpui_parley::{ParleyTextSystem, SystemFonts};
use std::{borrow::Cow, sync::Arc};

const BACKGROUND: [u8; 3] = [0x10, 0x14, 0x18];

struct Fixture;

impl Render for Fixture {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .bg(rgb(0x101418))
            .text_color(white())
            .font_family("IBM Plex Sans")
            .text_size(px(24.0))
            .p(px(16.0))
            .child("Headless rendering")
    }
}

#[test]
fn test_app_context_renders_real_text_at_the_configured_scale() {
    if std::env::var_os("GPUI_RUN_RENDERING_TESTS").is_none() {
        return;
    }

    for scale_factor in [1.0, 1.5] {
        let mut cx = TestAppContext::with_platform(
            TestDispatcher::new(0),
            None,
            Arc::new(ParleyTextSystem::new_with_system_font(
                SystemFonts::Skip,
                "IBM Plex Sans",
            )),
            Arc::new(()),
            gpui_ce_platform::current_headless_renderer,
            scale_factor,
        );
        cx.update(|cx| {
            cx.text_system()
                .add_fonts(vec![Cow::Borrowed(IBM_PLEX.data)])
        })
        .expect("failed to load the IBM Plex fixture");

        let window = cx.open_window(size(px(320.0), px(80.0)), |_, _| Fixture);
        let mut visual_cx = VisualTestContext::from_window(window.into(), &cx);
        visual_cx.run_until_parked();

        let (glyph_sprites, image) = visual_cx.update(|window, _| {
            let (_, monochrome, subpixel, _) = window.rendered_primitive_counts();
            (monochrome + subpixel, window.render_to_image())
        });
        let image = image.expect("failed to render the window to an image");

        assert!(glyph_sprites > 0, "the scene held no glyph sprites");
        assert_eq!(
            image.dimensions(),
            ((320.0 * scale_factor) as u32, (80.0 * scale_factor) as u32),
            "at scale {scale_factor}"
        );

        let corner = image.get_pixel(0, 0).0;
        assert!(
            corner
                .iter()
                .zip(BACKGROUND)
                .all(|(channel, expected)| channel.abs_diff(expected) <= 2),
            "the corner pixel was {corner:?}, expected the background {BACKGROUND:?}"
        );
        let ink = image.pixels().filter(|pixel| pixel.0 != corner).count();
        assert!(ink > 200, "only {ink} pixels differ from the background");

        drop(visual_cx);
        cx.quit();
    }
}
