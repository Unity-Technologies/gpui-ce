#[wgsl_rs::wgsl]
pub mod shadow {
    use super::super::common::*;
    use super::super::corner_smoothing::*;
    use wgsl_rs::std::*;

    #[derive(Clone, Copy, Wgsl)]
    pub struct Shadow {
        pub order: u32,
        pub blur_radius: f32,
        pub bounds: Bounds,
        pub corner_radii: Corners,
        pub content_mask: ContentMask,
        pub color: Background,
        /// The shape that is blurred: a drop shadow's silhouette, or an inset shadow's hole.
        pub shape_bounds: Bounds,
        pub shape_corner_radii: Corners,
        pub inset: ShaderBool,
        pub corner_smoothing: f32,
    }
    storage!(group(1), binding(0), SHADOWS: RuntimeArray<Shadow>);

    /// The area the shadow paints: the blurred extent of a drop shadow, or the
    /// element an inset shadow is clipped to.
    pub fn shadow_geometry(shadow: Shadow) -> Bounds {
        shadow.bounds
    }

    pub fn shadow_paint(shadow: Shadow) -> Paint {
        let mut bounds = shadow.shape_bounds;
        if is_enabled(shadow.inset) {
            bounds = shadow.bounds;
        }

        Paint::new(shadow.color, bounds)
    }

    pub fn prepare_shadow_blur(shadow: Shadow) -> PreparedBlurredRectangle {
        prepare_blurred_rectangle(
            shadow.shape_bounds.size,
            shadow.shape_corner_radii,
            shadow.blur_radius,
        )
    }

    pub fn blurred_shadow_coverage(
        shadow: Shadow,
        position: Vec2f,
        prepared: PreparedBlurredRectangle,
    ) -> f32 {
        blurred_rectangle_coverage(position - Bounds::center(shadow.shape_bounds), prepared)
    }

    pub fn shadow_coverage(
        shadow: Shadow,
        position: Vec2f,
        prepared_blur: PreparedBlurredRectangle,
    ) -> f32 {
        let mut coverage = 0.0;
        if shadow.blur_radius == 0.0 {
            coverage = antialiased_coverage(rounded_rectangle_signed_distance(
                position,
                shadow.shape_bounds,
                shadow.shape_corner_radii,
            ));
        } else {
            coverage = blurred_shadow_coverage(shadow, position, prepared_blur);
        }

        if is_enabled(shadow.inset) {
            coverage = (1.0 - coverage)
                * antialiased_coverage(rounded_rectangle_signed_distance(
                    position,
                    shadow.bounds,
                    shadow.corner_radii,
                ));
        }
        coverage
    }

    pub fn smoothed_shadow_coverage(
        shadow: Shadow,
        position: Vec2f,
        prepared: PreparedCorners,
        prepared_element: PreparedCorners,
    ) -> f32 {
        let mut coverage = 0.0;

        if shadow.blur_radius == 0.0 {
            coverage = antialiased_coverage(prepared_corner_signed_distance(
                position,
                shadow.shape_bounds,
                shadow.shape_corner_radii,
                shadow.corner_smoothing,
                prepared,
            ));
        } else if !Corners::is_zero(shadow.shape_corner_radii) {
            let blur_limit = GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * shadow.blur_radius;
            let half_size = Bounds::half_size(shadow.shape_bounds);
            let center_to_point = position - Bounds::center(shadow.shape_bounds);
            let box_delta = abs(center_to_point) - half_size;
            let outside_delta = max(box_delta, vec2f(0.0, 0.0));

            if dot(outside_delta, outside_delta) <= blur_limit * blur_limit {
                let local_point = position - shadow.shape_bounds.origin;
                let edge_depth = min(
                    min(local_point.x, shadow.shape_bounds.size.x - local_point.x),
                    min(local_point.y, shadow.shape_bounds.size.y - local_point.y),
                );
                let corner_candidate = figma_has_corner_candidate(
                    local_point,
                    shadow.shape_bounds.size,
                    prepared.horizontal_reaches,
                    prepared.vertical_reaches,
                );

                if !corner_candidate && edge_depth >= blur_limit {
                    coverage = 1.0;
                } else {
                    let distance = prepared_corner_signed_distance(
                        position,
                        shadow.shape_bounds,
                        shadow.shape_corner_radii,
                        shadow.corner_smoothing,
                        prepared,
                    );
                    coverage = gaussian_signed_distance_coverage(distance, shadow.blur_radius);
                }
            }
        } else {
            coverage = blurred_shadow_coverage(shadow, position, prepare_shadow_blur(shadow));
        }

        if is_enabled(shadow.inset) {
            coverage = (1.0 - coverage)
                * antialiased_coverage(prepared_corner_signed_distance(
                    position,
                    shadow.bounds,
                    shadow.corner_radii,
                    shadow.corner_smoothing,
                    prepared_element,
                ));
        }

        coverage
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct ShadowVertexData {
        pub position: Vec4f,
        pub paint: PreparedPaint,
        pub shadow_id: u32,
    }

    pub fn prepare_shadow_vertex(
        vertex_id: u32,
        instance_id: u32,
        shadow: Shadow,
    ) -> ShadowVertexData {
        let vertex = clipped_rectangle_vertex(
            vertex_id,
            shadow_geometry(shadow),
            shadow.content_mask.bounds,
        );
        ShadowVertexData {
            position: vertex.clip_position,
            paint: prepare_paint(shadow_paint(shadow)),
            shadow_id: instance_id,
        }
    }

    #[derive(Wgsl)]
    pub struct ShadowVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        #[interpolate(flat)]
        pub paint_color0: Vec4f,
        #[location(1)]
        #[interpolate(flat)]
        pub paint_color1: Vec4f,
        #[location(2)]
        #[interpolate(flat)]
        pub shadow_id: u32,
        #[location(3)]
        #[interpolate(flat)]
        pub blur_shape: Vec4f,
        #[location(4)]
        #[interpolate(flat)]
        pub blur_corner_radii: Vec4f,
        #[location(5)]
        #[interpolate(flat)]
        pub blur_exponents: Vec4f,
        #[location(6)]
        #[interpolate(flat)]
        pub blur_scales: Vec4f,
    }

    #[vertex]
    pub fn vertex_shadow(
        #[builtin(vertex_index)] vertex_id: u32,
        #[builtin(instance_index)] instance_id: u32,
    ) -> ShadowVarying {
        let shadow = get!(SHADOWS)[instance_id as usize];
        let vertex = prepare_shadow_vertex(vertex_id, instance_id, shadow);
        let blur = prepare_shadow_blur(shadow);
        ShadowVarying {
            position: vertex.position,
            paint_color0: vertex.paint.color0,
            paint_color1: vertex.paint.color1,
            shadow_id: vertex.shadow_id,
            blur_shape: blur.shape,
            blur_corner_radii: blur.corner_radii,
            blur_exponents: blur.exponents,
            blur_scales: blur.scales,
        }
    }

    #[fragment]
    pub fn fragment_shadow(input: ShadowVarying) -> Vec4f {
        let shadow = get!(SHADOWS)[input.shadow_id as usize];
        let color = paint_color(
            shadow_paint(shadow),
            input.position.xy(),
            PreparedPaint::new(input.paint_color0, input.paint_color1),
        );
        blend_color(
            color,
            shadow_coverage(
                shadow,
                input.position.xy(),
                PreparedBlurredRectangle {
                    shape: input.blur_shape,
                    corner_radii: input.blur_corner_radii,
                    exponents: input.blur_exponents,
                    scales: input.blur_scales,
                },
            ) * ContentMask::alpha(shadow.content_mask, input.position.xy()),
        )
    }

    #[derive(Wgsl)]
    pub struct SmoothedShadowVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        #[interpolate(flat)]
        pub paint_color0: Vec4f,
        #[location(1)]
        #[interpolate(flat)]
        pub paint_color1: Vec4f,
        #[location(2)]
        #[interpolate(flat)]
        pub shadow_id: u32,
        #[location(3)]
        #[interpolate(flat)]
        pub horizontal_corner_reaches: Vec4f,
        #[location(4)]
        #[interpolate(flat)]
        pub vertical_corner_reaches: Vec4f,
        #[location(5)]
        #[interpolate(flat)]
        pub element_horizontal_corner_reaches: Vec4f,
        #[location(6)]
        #[interpolate(flat)]
        pub element_vertical_corner_reaches: Vec4f,
        #[location(7)]
        #[interpolate(flat)]
        pub smoothing_factors: Vec4f,
    }

    #[vertex]
    pub fn vertex_smoothed_shadow(
        #[builtin(vertex_index)] vertex_id: u32,
        #[builtin(instance_index)] instance_id: u32,
    ) -> SmoothedShadowVarying {
        let shadow = get!(SHADOWS)[instance_id as usize];
        let vertex = prepare_shadow_vertex(vertex_id, instance_id, shadow);
        let prepared = prepare_corners(
            shadow.shape_bounds.size,
            shadow.shape_corner_radii,
            shadow.corner_smoothing,
            false,
        );
        let prepared_element = prepare_corners(
            shadow.bounds.size,
            shadow.corner_radii,
            shadow.corner_smoothing,
            false,
        );

        SmoothedShadowVarying {
            position: vertex.position,
            paint_color0: vertex.paint.color0,
            paint_color1: vertex.paint.color1,
            shadow_id: vertex.shadow_id,
            horizontal_corner_reaches: prepared.horizontal_reaches,
            vertical_corner_reaches: prepared.vertical_reaches,
            element_horizontal_corner_reaches: prepared_element.horizontal_reaches,
            element_vertical_corner_reaches: prepared_element.vertical_reaches,
            smoothing_factors: prepared.smoothing_factors,
        }
    }

    #[fragment]
    pub fn fragment_smoothed_shadow(input: SmoothedShadowVarying) -> Vec4f {
        let shadow = get!(SHADOWS)[input.shadow_id as usize];
        let coverage = smoothed_shadow_coverage(
            shadow,
            input.position.xy(),
            PreparedCorners {
                horizontal_reaches: input.horizontal_corner_reaches,
                vertical_reaches: input.vertical_corner_reaches,
                smoothing_factors: input.smoothing_factors,
                superellipse_power: 0.0,
            },
            PreparedCorners {
                horizontal_reaches: input.element_horizontal_corner_reaches,
                vertical_reaches: input.element_vertical_corner_reaches,
                smoothing_factors: input.smoothing_factors,
                superellipse_power: 0.0,
            },
        );
        let color = paint_color(
            shadow_paint(shadow),
            input.position.xy(),
            PreparedPaint::new(input.paint_color0, input.paint_color1),
        );
        blend_color(
            color,
            coverage * ContentMask::alpha(shadow.content_mask, input.position.xy()),
        )
    }
}
