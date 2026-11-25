//! Analytic path coverage, re-implemented from Zed's path tile shader (zed #66809).
//!
//! The CPU bins each path's monotone quadratic curves into tiles (`gpui::path_winding`). A tile
//! lists only the curves near it plus a backdrop winding for everything else, so the fragment
//! stage loops over a handful of curves and computes coverage from the winding number.

#[wgsl_rs::wgsl]
pub mod path_tile {
    use super::super::common::*;
    use wgsl_rs::std::*;

    #[derive(Clone, Copy, Wgsl)]
    pub struct PathTile {
        pub paint: u32,
        pub curve_start: u32,
        pub curve_count: u32,
        pub backdrop: i32,
        pub top_left: Vec2f,
        pub bottom_right: Vec2f,
        pub reference: Vec2f,
        pub curves_top_left: Vec2f,
        pub curves_bottom_right: Vec2f,
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct PathCurve {
        pub p0: Vec2f,
        pub p1: Vec2f,
        pub a: Vec2f,
        pub b: Vec2f,
        pub sign: f32,
        pub pad: u32,
    }

    #[derive(Clone, Copy, Wgsl)]
    pub struct PathPaint {
        pub bounds: Bounds,
        pub content_mask: ContentMask,
        pub color: Background,
        pub even_odd: ShaderBool,
        pub pad: u32,
    }

    // Tiles are this draw's instances. Curves and paints are indexed by the tiles, so the host
    // binds or offsets them to the start of the frame's curve and paint arrays.
    storage!(group(1), binding(0), PATH_TILES: RuntimeArray<PathTile>);
    storage!(group(1), binding(1), PATH_CURVES: RuntimeArray<PathCurve>);
    storage!(group(1), binding(2), PATH_PAINTS: RuntimeArray<PathPaint>);

    /// Winding along a pixel's leftward (x) and upward (y) rays, with the weight of each ray's
    /// nearest fractional crossing.
    #[derive(Clone, Copy, Wgsl)]
    pub struct PathWinding {
        pub winding: Vec2f,
        pub weight: Vec2f,
    }

    /// Solves `a*t^2 + b*t + c = 0` for the root in [0, 1] on a monotone piece.
    pub fn monotone_quadratic_root(a: f32, b: f32, c: f32, direction: f32) -> f32 {
        let discriminant = max(b * b - 4.0 * a * c, 0.0);
        let denominator = b + direction * sqrt(discriminant);
        if denominator == 0.0 {
            return 0.0;
        }
        saturate(-2.0 * c / denominator)
    }

    // Coverage estimate adapted from Eric Lengyel's Slug pixel shader (MIT, see
    // THIRD_PARTY_NOTICES.md). A crossing within half a pixel of the center counts fractionally.
    pub fn add_curve_winding(curve: PathCurve, center: Vec2f, state: PathWinding) -> PathWinding {
        let mut result = state;
        let min_x = min(curve.p0.x, curve.p1.x);
        let max_x = max(curve.p0.x, curve.p1.x);

        if curve.p0.y <= center.y && center.y < curve.p1.y {
            if max_x <= center.x - 0.5 {
                result.winding.x += curve.sign;
            } else {
                let t = monotone_quadratic_root(curve.a.y, curve.b.y, curve.p0.y - center.y, 1.0);
                let x = (curve.a.x * t + curve.b.x) * t + curve.p0.x;
                let distance = center.x - x;
                result.winding.x += curve.sign * saturate(distance + 0.5);
                result.weight.x = max(result.weight.x, saturate(1.0 - 2.0 * abs(distance)));
            }
        }

        if min_x <= center.x && center.x < max_x {
            let direction = select(-1.0, 1.0, curve.p1.x > curve.p0.x);
            if curve.p1.y <= center.y - 0.5 {
                result.winding.y -= curve.sign * direction;
            } else if curve.p0.y < center.y + 0.5 {
                let t =
                    monotone_quadratic_root(curve.a.x, curve.b.x, curve.p0.x - center.x, direction);
                let y = (curve.a.y * t + curve.b.y) * t + curve.p0.y;
                let distance = center.y - y;
                result.winding.y -= curve.sign * direction * saturate(distance + 0.5);
                result.weight.y = max(result.weight.y, saturate(1.0 - 2.0 * abs(distance)));
            }
        }
        result
    }

    /// The backdrop is the unlisted curves' winding along a leftward ray from the tile's reference
    /// corner. Paths are closed, so chains of unlisted curves begin and end only where a listed
    /// curve does: correcting the backdrop for each listed endpoint between its ray and the pixel
    /// gives the unlisted winding at the pixel, along both rays.
    pub fn correct_for_endpoint(
        endpoint: Vec2f,
        step: f32,
        reference: Vec2f,
        center: Vec2f,
        winding: Vec2f,
    ) -> Vec2f {
        let mut result = winding;
        if endpoint.x <= reference.x {
            let below_center = select(0.0, 1.0, endpoint.y <= center.y);
            let below_reference = select(0.0, 1.0, endpoint.y <= reference.y);
            result.x -= step * (below_center - below_reference);
        }
        if endpoint.y <= reference.y && endpoint.x <= center.x {
            result.y += step;
        }
        result
    }

    /// Weights each ray's winding by its nearest crossing's distance from the pixel center, as
    /// Slug does, then applies the fill rule.
    pub fn path_coverage(state: PathWinding, even_odd: bool) -> f32 {
        let weighted = abs(dot(state.winding, state.weight))
            / max(state.weight.x + state.weight.y, 1.0 / 65536.0);
        let coverage = max(weighted, min(abs(state.winding.x), abs(state.winding.y)));
        let mut result = saturate(coverage);
        if even_odd {
            result = 1.0 - abs(1.0 - fract(coverage * 0.5) * 2.0);
        }
        result
    }

    /// Accumulates a tile's listed curves into its backdrop at `center`.
    pub fn tile_winding(tile: PathTile, center: Vec2f) -> PathWinding {
        let backdrop = tile.backdrop as f32;
        let mut state = PathWinding {
            winding: vec2f(backdrop, backdrop),
            weight: vec2f(0.0, 0.0),
        };
        // Curves are sorted by their left edge, so the scan stops at the first one right of
        // the pixel.
        let mut offset = 0u32;
        let mut scanning = true;
        while scanning && offset < tile.curve_count {
            // The downlevel transport rewrites only a plain index expression.
            let curve_index = tile.curve_start + offset;
            let curve = get!(PATH_CURVES)[curve_index as usize];
            if min(curve.p0.x, curve.p1.x) >= center.x + 0.5 {
                scanning = false;
            } else {
                if curve.p0.y < center.y + 0.5 {
                    state = add_curve_winding(curve, center, state);
                    state.winding = correct_for_endpoint(
                        curve.p0,
                        curve.sign,
                        tile.reference,
                        center,
                        state.winding,
                    );
                    state.winding = correct_for_endpoint(
                        curve.p1,
                        -curve.sign,
                        tile.reference,
                        center,
                        state.winding,
                    );
                }
                offset += 1u32;
            }
        }
        state
    }

    #[derive(Wgsl)]
    pub struct PathTileVarying {
        #[builtin(position)]
        pub position: Vec4f,
        #[location(0)]
        #[interpolate(flat)]
        pub tile_id: u32,
        #[location(1)]
        #[interpolate(flat)]
        pub color0: Vec4f,
        #[location(2)]
        #[interpolate(flat)]
        pub color1: Vec4f,
        #[location(3)]
        #[interpolate(flat)]
        pub curve_box: Vec4f,
    }

    #[vertex]
    pub fn vertex_path_tile(
        #[builtin(vertex_index)] vertex_id: u32,
        #[builtin(instance_index)] instance_id: u32,
    ) -> PathTileVarying {
        let tile = get!(PATH_TILES)[instance_id as usize];
        let paint_index = tile.paint;
        let paint = get!(PATH_PAINTS)[paint_index as usize];

        let mask = paint.content_mask.bounds;
        let mask_bottom_right = mask.origin + mask.size;
        let top_left = clamp(tile.top_left, mask.origin, mask_bottom_right);
        let bottom_right = clamp(tile.bottom_right, top_left, mask_bottom_right);
        let corner = rectangle_corner(vertex_id);
        let position = vec2f(
            select(top_left.x, bottom_right.x, corner.x > 0.5),
            select(top_left.y, bottom_right.y, corner.y > 0.5),
        );

        let prepared = prepare_paint(Paint::new(paint.color, paint.bounds));
        // A solid fill without a fade covers every pixel outside the tile's curve box fully.
        let simple =
            paint.color.tag == BackgroundTag::Solid && !ContentMask::has_fade(paint.content_mask);
        PathTileVarying {
            position: viewport_to_clip_position(position),
            tile_id: mark_simple_primitive(instance_id, simple),
            color0: prepared.color0,
            color1: prepared.color1,
            curve_box: vec4f(
                tile.curves_top_left.x,
                tile.curves_top_left.y,
                tile.curves_bottom_right.x,
                tile.curves_bottom_right.y,
            ),
        }
    }

    #[fragment]
    pub fn fragment_path_tile(input: PathTileVarying) -> Vec4f {
        let center = floor(input.position.xy()) + vec2f(0.5, 0.5);
        let outside_curves = center.x < input.curve_box.x
            || center.y < input.curve_box.y
            || center.x > input.curve_box.z
            || center.y > input.curve_box.w;
        if is_simple_primitive(input.tile_id) && outside_curves {
            return blend_color(input.color0, 1.0);
        }

        let index = primitive_index(input.tile_id);
        let tile = get!(PATH_TILES)[index as usize];
        let paint_index = tile.paint;
        let paint = get!(PATH_PAINTS)[paint_index as usize];
        let coverage = path_coverage(tile_winding(tile, center), is_enabled(paint.even_odd))
            * ContentMask::alpha(paint.content_mask, input.position.xy());
        if coverage == 0.0 {
            return transparent();
        }

        let color = paint_color(
            Paint::new(paint.color, paint.bounds),
            input.position.xy(),
            PreparedPaint::new(input.color0, input.color1),
        );
        blend_color(color, coverage)
    }
}
