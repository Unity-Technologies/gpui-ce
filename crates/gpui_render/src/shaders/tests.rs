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
        ("vertex_shadow", (3, 9)),
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
