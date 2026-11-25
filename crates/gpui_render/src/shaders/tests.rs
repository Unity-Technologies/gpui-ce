use super::*;

#[test]
fn validates_subpixel_shader() {
    let generated = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    let source = format!("enable dual_source_blending;\n{generated}");
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn build_generated_shaders_match_rust_sources() {
    assert_eq!(
        base::WGSL_SOURCE.wgsl_source().unwrap(),
        crate::artifacts::BASE_WGSL
    );
    let subpixel = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    assert_eq!(
        format!("enable dual_source_blending;\n{subpixel}"),
        crate::artifacts::SUBPIXEL_DUAL_SOURCE_WGSL
    );
}

#[test]
fn shader_interface_matches_generated_sources() {
    let standard_source = base::WGSL_SOURCE.wgsl_source().unwrap();
    let subpixel_source = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    for pipeline in interface::ALL {
        let source = if pipeline.label == interface::SUBPIXEL_SPRITES.label {
            &subpixel_source
        } else {
            &standard_source
        };
        assert!(
            source.contains(&format!("fn {}(", pipeline.vertex_entry)),
            "missing vertex entry point {}",
            pipeline.vertex_entry,
        );
        assert!(
            source.contains(&format!("fn {}(", pipeline.fragment_entry)),
            "missing fragment entry point {}",
            pipeline.fragment_entry,
        );
    }

    assert_eq!(common::GLOBALS.group, interface::GLOBAL_BIND_GROUP);
    assert_eq!(common::GLOBALS.binding, interface::GLOBAL_UNIFORMS_BINDING);
    assert_eq!(
        common::FONT_RASTERIZATION.group,
        interface::GLOBAL_BIND_GROUP
    );
    assert_eq!(
        common::FONT_RASTERIZATION.binding,
        interface::FONT_RASTERIZATION_BINDING
    );
    assert_eq!(quad::QUADS.group(), interface::DATA_BIND_GROUP);
    assert_eq!(quad::QUADS.binding(), interface::DATA_BUFFER_BINDING);
}

fn vertex_output_shape(module: &naga::Module, entry_name: &str) -> (usize, u32) {
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_name)
        .unwrap_or_else(|| panic!("missing shader entry point {entry_name}"));
    let result = entry
        .function
        .result
        .as_ref()
        .unwrap_or_else(|| panic!("shader entry point {entry_name} has no output"));
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        panic!("shader entry point {entry_name} must return a struct");
    };

    members
        .iter()
        .filter(|member| matches!(member.binding, Some(naga::Binding::Location { .. })))
        .fold((0, 0), |(locations, components), member| {
            let member_components = match module.types[member.ty].inner {
                naga::TypeInner::Scalar(_) => 1,
                naga::TypeInner::Vector { size, .. } => u32::from(size),
                ref ty => panic!("unsupported stage output type {ty:?} in {entry_name}"),
            };
            (locations + 1, components + member_components)
        })
}

#[test]
fn ordinary_vertex_interfaces_stay_within_compact_budgets() {
    let module = naga::front::wgsl::parse_str(&base::WGSL_SOURCE.wgsl_source().unwrap()).unwrap();
    for (entry, maximum_shape) in [
        ("vertex_quad", (5, 17)),
        ("vertex_shadow", (7, 25)),
        ("vertex_underline", (2, 5)),
        ("vertex_polychrome_sprite", (3, 4)),
        ("vertex_surface", (1, 2)),
        ("vertex_blur_composite", (1, 2)),
    ] {
        let actual_shape = vertex_output_shape(&module, entry);
        assert!(
            actual_shape.0 <= maximum_shape.0 && actual_shape.1 <= maximum_shape.1,
            "{entry} uses {actual_shape:?}, exceeding {maximum_shape:?} (locations, components)"
        );
    }
}

#[test]
fn scene_instance_storage_sizes_are_bounded() {
    for (name, actual, maximum) in [
        ("Quad", std::mem::size_of::<gpui::Quad>(), 248),
        ("Shadow", std::mem::size_of::<gpui::Shadow>(), 184),
        (
            "PolychromeSprite",
            std::mem::size_of::<gpui::PolychromeSprite>(),
            112,
        ),
    ] {
        assert!(
            actual <= maximum,
            "{name} is {actual} bytes, exceeding its {maximum}-byte storage budget"
        );
    }
}

#[test]
fn shader_buffer_layouts_match_host_layouts() {
    let modules = [
        crate::artifacts::BASE_WGSL,
        crate::artifacts::SUBPIXEL_DUAL_SOURCE_WGSL,
    ]
    .map(|source| naga::front::wgsl::parse_str(source).unwrap());

    for layouts in [gpui::SCENE_BUFFER_LAYOUTS, interface::RENDER_BUFFER_LAYOUTS] {
        for layout in layouts {
            let ty = modules
                .iter()
                .find_map(|module| {
                    module
                        .types
                        .iter()
                        .find_map(|(_, ty)| (ty.name.as_deref() == Some(layout.name)).then_some(ty))
                })
                .unwrap_or_else(|| panic!("missing shader ABI type {}", layout.name));
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("shader ABI type {} must be a struct", layout.name);
            };

            assert_eq!(
                *span as usize, layout.size,
                "shader ABI size: {}",
                layout.name
            );
            assert_eq!(
                members.len(),
                layout.fields.len(),
                "shader ABI fields: {}",
                layout.name
            );
            for (member, (name, offset)) in members.iter().zip(layout.fields) {
                assert_eq!(
                    member.name.as_deref(),
                    Some(*name),
                    "shader ABI field: {}",
                    layout.name
                );
                assert_eq!(
                    member.offset as usize, *offset,
                    "shader ABI offset: {}.{name}",
                    layout.name
                );
            }
        }
    }
}

#[test]
fn shader_discriminants_match_scene_types() {
    assert_eq!(common::ShaderBool::Disabled as u32, 0);
    assert_eq!(common::ShaderBool::Enabled as u32, 1);
    assert_eq!(
        common::BackgroundTag::Solid as u32,
        gpui::BackgroundTag::Solid as u32
    );
    assert_eq!(
        common::BackgroundTag::LinearGradient as u32,
        gpui::BackgroundTag::LinearGradient as u32
    );
    assert_eq!(
        common::BackgroundTag::PatternSlash as u32,
        gpui::BackgroundTag::PatternSlash as u32
    );
    assert_eq!(
        common::BackgroundTag::Checkerboard as u32,
        gpui::BackgroundTag::Checkerboard as u32
    );
    assert_eq!(
        common::ColorSpace::Srgb as u32,
        gpui::ColorSpace::Srgb as u32
    );
    assert_eq!(
        common::ColorSpace::Oklab as u32,
        gpui::ColorSpace::Oklab as u32
    );
    assert_eq!(
        common::BorderStyle::Solid as u32,
        gpui::BorderStyle::Solid as u32
    );
    assert_eq!(
        common::BorderStyle::Dashed as u32,
        gpui::BorderStyle::Dashed as u32
    );
}

#[test]
fn linear_gradients_preserve_native_dithering() {
    use common::*;
    use wgsl_rs::std::*;

    let color = Hsla {
        h: 0.0,
        s: 0.0,
        l: 0.5,
        a: 0.5,
    };
    let background = Background {
        tag: BackgroundTag::LinearGradient,
        color_space: ColorSpace::Srgb,
        solid: color,
        gradient_angle_or_pattern_height: 90.0,
        colors: [
            LinearColorStop {
                color,
                percentage: 0.0,
            },
            LinearColorStop {
                color,
                percentage: 1.0,
            },
        ],
        padding: 0,
    };
    let bounds = Bounds {
        origin: vec2f(0.0, 0.0),
        size: vec2f(100.0, 100.0),
    };
    let paint = Paint::new(background, bounds);
    let prepared = prepare_paint(paint);
    let first = paint_color(paint, vec2f(10.0, 10.0), prepared);
    let second = paint_color(paint, vec2f(11.0, 10.0), prepared);

    assert_ne!(first.w, second.w);
}

#[test]
fn clip_to_mask_keeps_corners_inside_the_mask() {
    use common::*;
    use wgsl_rs::std::*;

    let bounds = Bounds {
        origin: vec2f(10.0, 20.0),
        size: vec2f(40.0, 20.0),
    };
    let mask = Bounds {
        origin: vec2f(20.0, 0.0),
        size: vec2f(100.0, 30.0),
    };
    let top_left = clip_to_mask(vec2f(0.0, 0.0), bounds, mask);
    assert_eq!((top_left.position.x, top_left.position.y), (20.0, 20.0));
    assert_eq!(
        (top_left.unit_position.x, top_left.unit_position.y),
        (0.25, 0.0)
    );
    let bottom_right = clip_to_mask(vec2f(1.0, 1.0), bounds, mask);
    assert_eq!(
        (bottom_right.position.x, bottom_right.position.y),
        (50.0, 30.0)
    );
    assert_eq!(
        (bottom_right.unit_position.x, bottom_right.unit_position.y),
        (1.0, 0.5)
    );

    // A mask inside the bounds becomes the geometry.
    let inner = Bounds {
        origin: vec2f(15.0, 25.0),
        size: vec2f(5.0, 5.0),
    };
    let corner = clip_to_mask(vec2f(1.0, 0.0), bounds, inner);
    assert_eq!((corner.position.x, corner.position.y), (20.0, 25.0));

    // A disjoint mask collapses to zero area, and an empty primitive stays finite.
    let disjoint = Bounds {
        origin: vec2f(200.0, 200.0),
        size: vec2f(10.0, 10.0),
    };
    let first = clip_to_mask(vec2f(0.0, 0.0), bounds, disjoint);
    let last = clip_to_mask(vec2f(1.0, 1.0), bounds, disjoint);
    assert_eq!(
        (first.position.x, first.position.y),
        (last.position.x, last.position.y)
    );
    let empty = Bounds {
        origin: vec2f(5.0, 5.0),
        size: vec2f(0.0, 0.0),
    };
    let degenerate = clip_to_mask(vec2f(1.0, 1.0), empty, mask);
    assert!(degenerate.unit_position.x.is_finite() && degenerate.unit_position.y.is_finite());
}

#[test]
fn simple_primitive_flag_round_trips_the_instance_index() {
    use common::*;

    for index in [0u32, 1, 4095, SIMPLE_PRIMITIVE_FLAG - 1] {
        let simple = mark_simple_primitive(index, true);
        assert!(is_simple_primitive(simple));
        assert_eq!(primitive_index(simple), index);
        let shaped = mark_simple_primitive(index, false);
        assert!(!is_simple_primitive(shaped));
        assert_eq!(primitive_index(shaped), index);
    }
}

/// A dense reference for a Gaussian-blurred rounded rectangle: the exact blur along x of each
/// horizontal slice, integrated along y with `samples` midpoint samples.
fn integrated_shadow(
    center_to_point: wgsl_rs::std::Vec2f,
    half_size: wgsl_rs::std::Vec2f,
    corner_radius: f32,
    standard_deviation: f32,
    samples: u32,
) -> f32 {
    use common::*;
    use wgsl_rs::std::*;

    let along_x = |x: f32, y: f32| {
        let beyond = (half_size.y - corner_radius - y.abs()).min(0.0);
        let curved = half_size.x - corner_radius
            + (corner_radius * corner_radius - beyond * beyond)
                .max(0.0)
                .sqrt();
        let bounds = vec2f(x - curved, x + curved) * (0.5f32.sqrt() / standard_deviation);
        let integral = approximate_error_function(bounds);
        0.5 * (integral.y - integral.x)
    };
    let low = center_to_point.y - half_size.y;
    let high = center_to_point.y + half_size.y;
    let start = (-3.0 * standard_deviation).clamp(low, high);
    let end = (3.0 * standard_deviation).clamp(low, high);
    let step = (end - start) / samples as f32;
    let mut y = start + step * 0.5;
    let mut coverage = 0.0;
    for _ in 0..samples {
        coverage += along_x(center_to_point.x, center_to_point.y - y)
            * gaussian(y, standard_deviation)
            * step;
        y += step;
    }
    coverage
}

#[test]
fn levien_shadow_tracks_the_exact_blur() {
    use common::*;
    use wgsl_rs::std::*;

    // (width, height, radius, sigma, tolerance). Levien's fit is tight for ordinary shadows
    // and loosest where the blur is wide against an eccentric shape.
    for (width, height, radius, sigma, tolerance) in [
        (50.0, 36.0, 0.0, 2.0, 0.02),
        (50.0, 36.0, 6.0, 4.0, 0.015),
        (200.0, 40.0, 8.0, 4.0, 0.015),
        (300.0, 200.0, 12.0, 10.0, 0.015),
        (40.0, 40.0, 20.0, 12.0, 0.04),
        (50.0, 36.0, 18.0, 8.0, 0.08),
        (50.0, 36.0, 0.0, 20.0, 0.05),
        (120.0, 20.0, 10.0, 6.0, 0.09),
    ] {
        let size = vec2f(width, height);
        let corners = Corners {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        };
        let prepared = prepare_blurred_rectangle(size, corners, sigma);
        let reach = 3.0 * sigma;
        let mut worst = 0.0f32;
        let mut x = -width / 2.0 - reach;
        while x <= width / 2.0 + reach {
            let mut y = -height / 2.0 - reach;
            while y <= height / 2.0 + reach {
                let point = vec2f(x, y);
                let levien = blurred_rectangle_coverage(point, prepared);
                assert!(levien.is_finite() && (-0.001..=1.001).contains(&levien));
                let exact = integrated_shadow(point, size / 2.0, radius, sigma, 256);
                worst = worst.max((levien - exact).abs());
                y += 1.5;
            }
            x += 1.5;
        }
        assert!(
            worst < tolerance,
            "{width}x{height} r{radius} sigma {sigma}: Levien drifts {worst} from the exact blur"
        );
    }
}

#[test]
fn levien_error_function_matches_known_values() {
    for (input, expected) in [
        (0.0f32, 0.0f32),
        (0.5, 0.5205),
        (1.0, 0.8427),
        (2.0, 0.9953),
    ] {
        let actual = common::levien_error_function(input);
        assert!((actual - expected).abs() < 0.001, "erf({input}) = {actual}");
        assert!((common::levien_error_function(-input) + expected).abs() < 0.001);
    }
}

#[test]
fn border_compositing_keeps_the_border_color_over_a_transparent_fill() {
    use common::*;
    use wgsl_rs::std::*;

    // A transparent black fill under a white border: every partly covered inner-edge pixel
    // keeps the border's color and only loses alpha, so the edge never darkens.
    let fill = vec4f(0.0, 0.0, 0.0, 0.0);
    for step in 0..=8 {
        let distance = -1.0 + step as f32 * 0.25;
        let mut border = vec4f(1.0, 1.0, 1.0, 1.0);
        border.w *= antialiased_coverage(distance);
        let color = over(fill, border);
        assert!((color.w - antialiased_coverage(distance)).abs() < 1e-6);
        if color.w > 0.0 {
            assert!(
                (color.x - 1.0).abs() < 1e-5,
                "border darkened at {distance}: {color:?}"
            );
        }
    }

    // An opaque fill still blends linearly toward the border across the edge.
    let fill = vec4f(0.0, 0.0, 1.0, 1.0);
    let mut border = vec4f(1.0, 0.0, 0.0, 1.0);
    border.w *= 0.25;
    let color = over(fill, border);
    assert!((color.x - 0.25).abs() < 1e-6 && (color.z - 0.75).abs() < 1e-6 && color.w == 1.0);
}
