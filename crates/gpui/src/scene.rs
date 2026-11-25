// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, HitboxId, Hsla,
    NativeViewId, Path, PathTiles, Pixels, Point, Radians, ScaledFilter, ScaledPixels, Size,
    TimedTransformation, bounds_tree::BoundsTree, point,
};
use smallvec::SmallVec;
use std::{
    collections::VecDeque, fmt::Debug, iter::Peekable, ops::Range, slice, sync::Arc, time::Duration,
};

mod plan;
pub use plan::*;

static OVERLAY_MODAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// HUB-8754 spike: while true, the macOS overlay view claims every click in the window (so an
/// outside click reaches GPUI instead of a native child) and keyboard focus returns to GPUI.
/// A real design derives this from block-mouse hitboxes instead of an app call.
pub fn set_overlay_modal(modal: bool) {
    OVERLAY_MODAL.store(modal, std::sync::atomic::Ordering::Relaxed);
}

/// See [`set_overlay_modal`].
pub fn overlay_modal() -> bool {
    OVERLAY_MODAL.load(std::sync::atomic::Ordering::Relaxed)
}
mod abi;
#[doc(hidden)]
pub use abi::{SCENE_BUFFER_LAYOUTS, SceneBufferLayout};

#[expect(missing_docs)]
pub type DrawOrder = u32;

pub(crate) const DEFAULT_BORDER_DASHED_LENGTH: f32 = 2.0;
pub(crate) const DEFAULT_BORDER_DASHED_GAP: f32 = 1.0;

/// A boolean with the same four-byte representation in Rust and WGSL.
/// Scene structs use it over one-byte [`bool`] to keep the storage-buffer ABI explicit.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ShaderBool {
    /// The flag is disabled.
    #[default]
    Disabled = 0,
    /// The flag is enabled.
    Enabled = 1,
}

impl ShaderBool {
    /// Returns this flag as a regular Rust boolean.
    pub fn is_enabled(self) -> bool {
        self == Self::Enabled
    }
}

impl From<bool> for ShaderBool {
    fn from(value: bool) -> Self {
        if value { Self::Enabled } else { Self::Disabled }
    }
}

/// A layer whose draw order is assigned once the outermost open layer closes,
/// from the union of what it painted.
struct Layer {
    primitives: SmallVec<[Range<usize>; 1]>,
    bounds: Bounds<ScaledPixels>,
    max_kind: PrimitiveKind,
    max_shared_kind: PrimitiveKind,
    /// Whether a native view was painted in the layer, which then can't be
    /// drawn as one: what was painted after the view has to cover it.
    has_native_view: bool,
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    /// Open layers, as indices into `layers`. `None` marks a layer that a filter
    /// boundary split: what it paints afterwards gets draw orders one by one.
    layer_stack: Vec<Option<usize>>,
    /// Layers waiting for their draw order, in the order they were opened.
    layers: VecDeque<Layer>,
    filter_depth: usize,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<PaintedPath>,
    /// Every path's tiles in draw order, built by [`Scene::finish`]. Tiles index
    /// [`Self::path_curves`] and [`Self::path_paints`] from the start of those arrays.
    pub path_tiles: Vec<PathTile>,
    pub path_curves: Vec<PathCurve>,
    pub path_paints: Vec<PathPaint>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    surface_opacities: Vec<f32>,
    pub native_views: Vec<PaintNativeView>,
    pub backdrop_filters: Vec<BackdropFilter>,
    pub filter_boundaries: Vec<FilterBoundary>,
    render_plan: ScenePlan,
    is_finished: bool,
    /// HUB-8754 spike: index into `paint_operations` where deferred draws begin.
    overlay_start: Option<usize>,
    animations: Vec<SpriteAnimation>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.overlay_start = None;
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.layers.clear();
        self.filter_depth = 0;
        self.paths.clear();
        self.path_tiles.clear();
        self.path_curves.clear();
        self.path_paints.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.surface_opacities.clear();
        self.native_views.clear();
        self.backdrop_filters.clear();
        self.filter_boundaries.clear();
        self.render_plan.clear();
        self.is_finished = false;
        self.animations.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self) {
        self.is_finished = false;
        self.layer_stack.push(Some(self.layers.len()));
        self.layers.push_back(Layer {
            primitives: SmallVec::new(),
            bounds: Bounds::default(),
            max_kind: PrimitiveKind::FilterBoundaryStart,
            max_shared_kind: PrimitiveKind::FilterBoundaryEnd,
            has_native_view: false,
        });
        self.paint_operations.push(PaintOperation::StartLayer);
    }

    pub fn pop_layer(&mut self) {
        self.is_finished = false;
        self.layer_stack.pop();
        if !matches!(self.layer_stack.last(), Some(Some(_))) {
            self.insert_pending_layers();
        }
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Assigns draw orders to every pending layer, parents before the layers
    /// nested in them, so a nested layer draws above its parent.
    fn insert_pending_layers(&mut self) {
        while let Some(layer) = self.layers.pop_front() {
            self.insert_layer(layer);
        }
    }

    fn insert_layer(&mut self, layer: Layer) {
        if layer.primitives.is_empty() {
            return;
        }
        if layer.has_native_view {
            self.insert_layer_around_native_views(layer);
            return;
        }
        let order =
            self.primitive_bounds
                .insert(layer.bounds, layer.max_kind, layer.max_shared_kind);
        for index in layer.primitives.into_iter().flatten() {
            if let PaintOperation::Primitive(primitive) = &self.paint_operations[index] {
                let primitive = primitive.clone();
                self.push_primitive(primitive, order);
            }
        }
    }

    /// Inserts a layer that a native view was painted in.
    ///
    /// A layer's primitives share one place in the drawing order, where they
    /// are drawn by kind and not in the order they were painted. A native
    /// view erases what is drawn before it, so the layer is inserted in parts:
    /// what was painted before each view, the view, and what was painted after.
    fn insert_layer_around_native_views(&mut self, layer: Layer) {
        let mut part = Vec::new();
        for index in layer.primitives.into_iter().flatten() {
            let PaintOperation::Primitive(primitive) = &self.paint_operations[index] else {
                continue;
            };
            let primitive = primitive.clone();
            if primitive.kind() == PrimitiveKind::NativeView {
                self.insert_layer_part(std::mem::take(&mut part));
                self.insert_layer_part(vec![primitive]);
            } else {
                part.push(primitive);
            }
        }
        self.insert_layer_part(part);
    }

    fn insert_layer_part(&mut self, primitives: Vec<Primitive>) {
        let Some(first) = primitives.first() else {
            return;
        };
        let mut bounds = first.clipped_bounds();
        let mut max_kind = PrimitiveKind::FilterBoundaryStart;
        let mut max_shared_kind = PrimitiveKind::FilterBoundaryEnd;
        for primitive in &primitives {
            let kind = primitive.kind();
            bounds = bounds.join(&primitive.clipped_bounds());
            max_kind = max_kind.max(kind);
            max_shared_kind = max_shared_kind.min(kind.max_shared_kind());
        }
        let order = self
            .primitive_bounds
            .insert(bounds, max_kind, max_shared_kind);
        for primitive in primitives {
            self.push_primitive(primitive, order);
        }
    }

    /// Raise the draw-order floor so every primitive inserted afterwards sorts above everything
    /// inserted before. Called before painting deferred draws so overlays (tooltips, popovers,
    /// drag images) sort above the main scene — and a deferred backdrop's order can't fall inside
    /// a content-filter (`filter`) order range left behind by the main scene.
    pub fn raise_order_floor(&mut self) {
        self.is_finished = false;
        self.overlay_start
            .get_or_insert(self.paint_operations.len());
        let floor = self.primitive_bounds.max_order() + 1;
        self.primitive_bounds.set_order_floor(floor);
    }

    pub(crate) fn is_in_filter_group(&self) -> bool {
        self.filter_depth > 0
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        self.try_insert_primitive(primitive.into());
    }

    /// Inserts a sprite that [`Self::advance_animations`] keeps moving. The
    /// sprite's color is taken as its color before the animation's alpha.
    pub(crate) fn insert_animated_sprite(
        &mut self,
        mut sprite: MonochromeSprite,
        mut animation: SpriteAnimation,
    ) {
        sprite.padding = SpriteAnimationId::for_index(self.animations.len()).0;
        animation.base_color = sprite.color;
        if self.try_insert_primitive(sprite.into()) {
            self.animations.push(animation);
        }
    }

    /// How long the window may wait between presents of this scene while only
    /// its animations need a frame: `None` when nothing in it animates, and
    /// zero when an animation wants every frame.
    pub(crate) fn animation_frame_interval(&self) -> Option<Duration> {
        self.animations
            .iter()
            .map(|animation| {
                animation
                    .transformation
                    .min_frame_interval()
                    .unwrap_or_default()
            })
            .min()
    }

    /// Moves every animated sprite to its pose at `elapsed` time on the app's
    /// synced animation clock, without the scene being rebuilt.
    ///
    /// Only each sprite's transformation and color change, never its order,
    /// texture or count, so the finished scene's render plan stays valid.
    pub(crate) fn advance_animations(&mut self, elapsed: Duration) {
        for animation in &self.animations {
            let Some(sprite) = self.monochrome_sprites.get_mut(animation.sprite_index) else {
                continue;
            };
            let pose = animation.transformation.at(elapsed);
            sprite.transformation = pose.into_matrix(animation.center, animation.scale_factor);
            sprite.color = SceneHsla {
                a: animation.base_color.a * pose.alpha_factor(),
                ..animation.base_color
            };
        }
    }

    /// Returns whether the primitive was kept, rather than clipped away.
    fn try_insert_primitive(&mut self, mut primitive: Primitive) -> bool {
        self.is_finished = false;

        // Cached glyph batches may have been recorded outside isolation. The monochrome
        // pipeline also accepts RGB atlas masks and provides a scalar-alpha fallback.
        if self.is_in_filter_group()
            && let Primitive::SubpixelSprite(sprite) = primitive
        {
            primitive = Primitive::MonochromeSprite(MonochromeSprite {
                order: sprite.order,
                // Converted glyphs never animate.
                padding: SpriteAnimationId::NONE.0,
                bounds: sprite.bounds,
                content_mask: sprite.content_mask,
                color: sprite.color,
                tile: sprite.tile,
                transformation: sprite.transformation,
            });
        }

        let clipped_bounds = primitive.clipped_bounds();

        // Isolation boundaries stay paired even when their clipped region is empty.
        // Each marker sorts above prior content, and its order floor keeps following
        // content inside the intended group or after its closing marker.
        if let Primitive::FilterBoundary(boundary) = &mut primitive {
            // A boundary splits every open layer: what they painted so far draws
            // below it, and what they paint afterwards draws above it.
            self.insert_pending_layers();
            self.layer_stack.fill(None);

            let order_bounds = if clipped_bounds.is_empty() {
                boundary.bounds
            } else {
                clipped_bounds
            };
            let order = self.primitive_bounds.insert_above_all(order_bounds);
            boundary.order = order;
            self.primitive_bounds.set_order_floor(order + 1);
            if boundary.is_start {
                self.filter_depth += 1;
            } else {
                self.filter_depth = self.filter_depth.saturating_sub(1);
            }
            self.filter_boundaries.push(boundary.clone());
        } else {
            if clipped_bounds.is_empty() {
                return false;
            }

            let kind = primitive.kind();
            if let Some(&Some(layer)) = self.layer_stack.last()
                && let Some(layer) = self.layers.get_mut(layer)
            {
                layer.bounds = layer.bounds.join(&clipped_bounds);
                layer.max_kind = layer.max_kind.max(kind);
                layer.max_shared_kind = layer.max_shared_kind.min(kind.max_shared_kind());
                layer.has_native_view |= kind == PrimitiveKind::NativeView;
                let index = self.paint_operations.len();
                match layer.primitives.last_mut() {
                    Some(range) if range.end == index => range.end += 1,
                    _ => layer.primitives.push(index..index + 1),
                }
            } else {
                let order =
                    self.primitive_bounds
                        .insert(clipped_bounds, kind, kind.max_shared_kind());
                self.push_primitive(primitive.clone(), order);
            }
        }

        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
        true
    }

    fn push_primitive(&mut self, primitive: Primitive, order: DrawOrder) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(Shadow { order, ..shadow }),
            Primitive::Quad(quad) => self.quads.push(Quad { order, ..quad }),
            Primitive::Path(mut path) => {
                path.order = order;
                self.paths.push(path);
            }
            Primitive::Underline(underline) => {
                self.underlines.push(Underline { order, ..underline })
            }
            Primitive::MonochromeSprite(sprite) => self
                .monochrome_sprites
                .push(MonochromeSprite { order, ..sprite }),
            Primitive::SubpixelSprite(sprite) => self
                .subpixel_sprites
                .push(SubpixelSprite { order, ..sprite }),
            Primitive::PolychromeSprite(sprite) => self
                .polychrome_sprites
                .push(PolychromeSprite { order, ..sprite }),
            Primitive::Surface(mut surface) => {
                surface.order = order;
                self.surfaces.push(surface);
                self.surface_opacities.push(1.0);
            }
            Primitive::NativeView(native_view) => self.native_views.push(PaintNativeView {
                order,
                ..native_view
            }),
            Primitive::BackdropFilter(mut filter) => {
                filter.order = order;
                self.backdrop_filters.push(filter);
            }
            Primitive::FilterBoundary(_) => {
                unreachable!("filter boundaries are ordered when they are inserted")
            }
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    // Animation ids index the scene they were inserted into.
                    if let Primitive::MonochromeSprite(sprite) = primitive
                        && let Some(animation) = sprite
                            .animation()
                            .index()
                            .and_then(|index| prev_scene.animations.get(index))
                    {
                        self.insert_animated_sprite(*sprite, *animation);
                    } else {
                        self.insert_primitive(primitive.clone());
                    }
                }
                PaintOperation::StartLayer => self.push_layer(),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    /// HUB-8754 spike: split a finished scene into (base, overlay, overlay_rects). The overlay
    /// holds everything painted by deferred draws (menus, tooltips, popovers).
    pub fn split_overlay(&self) -> Option<(Scene, Scene, Vec<Bounds<ScaledPixels>>)> {
        let start = self.overlay_start?;
        let mut base = Scene::default();
        base.replay(0..start, self);
        base.finish();
        let mut overlay = Scene::default();
        overlay.replay(start..self.paint_operations.len(), self);
        let rects = overlay
            .paint_operations
            .iter()
            .filter_map(|op| match op {
                PaintOperation::Primitive(Primitive::Quad(q)) => Some(q.bounds),
                PaintOperation::Primitive(Primitive::Shadow(s)) => Some(s.bounds),
                _ => None,
            })
            .collect();
        overlay.finish();
        Some((base, overlay, rects))
    }

    pub fn finish(&mut self) {
        debug_assert!(self.layer_stack.is_empty(), "unbalanced scene layers");
        let keys = &mut Vec::new();
        sort_in_place(&mut self.shadows, keys, |shadow| (shadow.order, 0));
        sort_in_place(&mut self.quads, keys, |quad| (quad.order, 0));
        self.paths.sort_by_key(|path| path.order);
        self.path_tiles.clear();
        self.path_curves.clear();
        self.path_paints.clear();
        for path in &mut self.paths {
            path.tile_start = self.path_tiles.len();
            path.append_to(
                &mut self.path_paints,
                &mut self.path_curves,
                &mut self.path_tiles,
            );
        }
        sort_in_place(&mut self.underlines, keys, |underline| (underline.order, 0));
        sort_in_place(&mut self.monochrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.subpixel_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.polychrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        self.surfaces.sort_by_key(|surface| surface.order);
        self.surface_opacities.resize(self.surfaces.len(), 1.0);
        self.native_views
            .sort_by_key(|native_view| native_view.order);
        // A platform view can only be in one place, which is where it was painted first.
        let mut shown = Vec::new();
        self.native_views.retain(|native_view| {
            let is_first = !shown.contains(&native_view.id);
            shown.push(native_view.id);
            is_first
        });
        self.backdrop_filters.sort_by_key(|filter| filter.order);
        // Markers normally get distinct, monotonically-increasing orders (children overlap
        // their group bounds and so sort strictly between the start and end). The `!is_start`
        // tiebreak only matters for a degenerate empty group whose start and end tie: it keeps
        // the start (false = 0) ahead of the end (true = 1) so the pair stays well-formed.
        self.filter_boundaries
            .sort_by_key(|boundary| (boundary.order, !boundary.is_start));
        // Sorting moved the sprites, so animations learn where theirs ended up.
        if !self.animations.is_empty() {
            for (index, sprite) in self.monochrome_sprites.iter().enumerate() {
                if let Some(animation) = sprite
                    .animation()
                    .index()
                    .and_then(|id| self.animations.get_mut(id))
                {
                    animation.sprite_index = index;
                }
            }
        }
        let storage = self.render_plan.take_storage();
        self.render_plan = ScenePlan::build(self, storage);
        self.is_finished = true;
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        self.render_commands().iter().map(|command| match command {
            RenderCommand::Batch(batch) => batch.clone(),
            RenderCommand::BeginFilter { boundary_index, .. } => {
                PrimitiveBatch::FilterBoundary(*boundary_index)
            }
            RenderCommand::EndFilter {
                closing_boundary_index,
                ..
            } => PrimitiveBatch::FilterBoundary(*closing_boundary_index),
        })
    }

    /// Returns the backend-neutral sequence of work needed to render this scene.
    ///
    /// In addition to preserving primitive batch order, this pairs content-filter boundaries and
    /// assigns the bounded offscreen target used by each isolated group. GPU backends only need to
    /// manage their target handles; nesting and overflow behavior stay consistent everywhere.
    pub fn render_commands(&self) -> &[RenderCommand] {
        self.render_plan().commands()
    }

    /// Renders the finished scene without the opaque pre-pass: every quad draws blended.
    #[cfg(any(test, feature = "test-support"))]
    pub fn disable_opaque_quads(&mut self) {
        self.render_plan.forget_opaque_quads();
    }

    /// Returns the compiled render plan for this finished scene.
    pub fn render_plan(&self) -> &ScenePlan {
        debug_assert!(
            self.is_finished,
            "Scene::finish must be called before rendering"
        );
        self.render_plan.assert_matches(self);
        &self.render_plan
    }

    /// Returns the intrinsic opacity of each surface in [`Self::surfaces`].
    /// Surfaces are fully opaque here; group opacity is applied when compositing.
    pub fn surface_opacities(&self) -> &[f32] {
        &self.surface_opacities
    }

    /// Whether rendering needs an offscreen scene target for filters or grouped opacity.
    pub fn requires_offscreen_rendering(&self) -> bool {
        self.render_plan().requirements().uses_offscreen_target
    }
}

/// Sorts small `(key, index)` pairs instead of the primitives themselves, then moves each
/// primitive into place once. Primitives are large, so a comparison sort that shuffles them
/// directly spends most of its time copying. Ties keep insertion order, as `sort_by_key`
/// does, because the index is part of what is sorted.
fn sort_in_place<T: Copy>(
    primitives: &mut [T],
    keys: &mut Vec<(u64, u32)>,
    key: impl Fn(&T) -> (u32, u32),
) {
    if primitives.is_sorted_by_key(&key) {
        return;
    }
    keys.clear();
    keys.extend(primitives.iter().enumerate().map(|(index, primitive)| {
        let (major, minor) = key(primitive);
        ((major as u64) << 32 | minor as u64, index as u32)
    }));
    keys.sort_unstable();

    // `keys[target].1` is the source index for `target`. Follow each cycle of that
    // permutation, marking visited slots by pointing them at themselves.
    for start in 0..primitives.len() {
        if keys[start].1 as usize == start {
            continue;
        }
        let displaced = primitives[start];
        let mut target = start;
        loop {
            let source = keys[target].1 as usize;
            keys[target].1 = target as u32;
            if source == start {
                primitives[target] = displaced;
                break;
            }
            primitives[target] = primitives[source];
            target = source;
        }
    }
}

/// Internal representation of [`palette::Hsla`] which is layout sensitive, as its provided to the renderer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub struct SceneHsla {
    /// Hue, in a range from 0 to 1
    pub(crate) h: f32,
    /// Saturation, in a range from 0 to 1
    pub(crate) s: f32,
    /// Lightness, in a range from 0 to 1
    pub(crate) l: f32,
    /// Alpha, in a range from 0 to 1
    pub(crate) a: f32,
}
impl Into<palette::Hsla> for SceneHsla {
    fn into(self) -> palette::Hsla {
        palette::Hsla::new(self.h * 360.0, self.s, self.l, self.a)
    }
}
impl From<palette::Hsla> for SceneHsla {
    fn from(hsla: palette::Hsla) -> Self {
        Self {
            h: hsla.hue.into_positive_degrees() / 360.0,
            s: hsla.saturation,
            l: hsla.lightness,
            a: hsla.alpha,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    // Lowest discriminant: at an equal order, a content-filter group-start is emitted before
    // the group's own content so the renderer redirects rendering before any child draws.
    FilterBoundaryStart,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    BackdropFilter,
    NativeView,
    // Highest discriminant: at an equal order, a group-end is emitted after the group's content
    // so the renderer composites the filtered group only once every child has been drawn.
    FilterBoundaryEnd,
}

impl PrimitiveKind {
    /// Overlapping primitives are drawn in paint order. But to batch more draws,
    /// the renderer sorts primitives that share a draw order by kind, then
    /// sprites by atlas texture, and only then by paint order. So a primitive
    /// can only share the draw order of the topmost primitive it overlaps if
    /// that primitive's kind is at most the one returned here. Otherwise it
    /// needs a new draw order.
    pub(crate) fn max_shared_kind(self) -> Self {
        match self {
            Self::MonochromeSprite => Self::Underline,
            Self::SubpixelSprite => Self::MonochromeSprite,
            Self::PolychromeSprite => Self::SubpixelSprite,
            // A backdrop filter samples what is drawn before it, so one backdrop
            // never shares an order with another it overlaps.
            Self::BackdropFilter => Self::Surface,
            // Boundaries take orders above everything, never a shared one.
            Self::FilterBoundaryStart | Self::FilterBoundaryEnd => Self::FilterBoundaryStart,
            kind => kind,
        }
    }
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer,
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(PaintedPath),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    BackdropFilter(BackdropFilter),
    FilterBoundary(FilterBoundary),
    NativeView(PaintNativeView),
}

#[expect(missing_docs)]
impl Primitive {
    pub(crate) fn kind(&self) -> PrimitiveKind {
        match self {
            Primitive::Shadow(_) => PrimitiveKind::Shadow,
            Primitive::Quad(_) => PrimitiveKind::Quad,
            Primitive::Path(_) => PrimitiveKind::Path,
            Primitive::Underline(_) => PrimitiveKind::Underline,
            Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
            Primitive::SubpixelSprite(_) => PrimitiveKind::SubpixelSprite,
            Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
            Primitive::Surface(_) => PrimitiveKind::Surface,
            Primitive::BackdropFilter(_) => PrimitiveKind::BackdropFilter,
            Primitive::NativeView(_) => PrimitiveKind::NativeView,
            Primitive::FilterBoundary(boundary) if boundary.is_start => {
                PrimitiveKind::FilterBoundaryStart
            }
            Primitive::FilterBoundary(_) => PrimitiveKind::FilterBoundaryEnd,
        }
    }

    fn transformation(&self) -> TransformationMatrix {
        match self {
            Primitive::MonochromeSprite(sprite) => sprite.transformation,
            Primitive::SubpixelSprite(sprite) => sprite.transformation,
            _ => TransformationMatrix::unit(),
        }
    }

    /// The area this primitive can paint: its transformed bounds within its
    /// content mask.
    pub(crate) fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
        self.transformation()
            .transform_bounds(*self.bounds())
            .intersect(&self.content_mask().bounds)
    }

    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::BackdropFilter(filter) => &filter.bounds,
            Primitive::FilterBoundary(boundary) => &boundary.bounds,
            Primitive::NativeView(native_view) => &native_view.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.paint.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::BackdropFilter(filter) => &filter.content_mask,
            Primitive::FilterBoundary(boundary) => &boundary.content_mask,
            Primitive::NativeView(native_view) => &native_view.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths: &'a [PaintedPath],
    paths_iter: Peekable<slice::Iter<'a, PaintedPath>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    backdrop_filters_start: usize,
    backdrop_filters_iter: Peekable<slice::Iter<'a, BackdropFilter>>,
    native_views_start: usize,
    native_views_iter: Peekable<slice::Iter<'a, PaintNativeView>>,
    filter_boundaries_start: usize,
    filter_boundaries_iter: Peekable<slice::Iter<'a, FilterBoundary>>,
}

impl<'a> BatchIterator<'a> {
    fn new(scene: &'a Scene) -> Self {
        Self {
            shadows_start: 0,
            shadows_iter: scene.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: scene.quads.iter().peekable(),
            paths_start: 0,
            paths: &scene.paths,
            paths_iter: scene.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: scene.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: scene.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: scene.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: scene.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: scene.surfaces.iter().peekable(),
            backdrop_filters_start: 0,
            backdrop_filters_iter: scene.backdrop_filters.iter().peekable(),
            native_views_start: 0,
            native_views_iter: scene.native_views.iter().peekable(),
            filter_boundaries_start: 0,
            filter_boundaries_iter: scene.filter_boundaries.iter().peekable(),
        }
    }
}

fn precedes_limit(
    order: DrawOrder,
    kind: PrimitiveKind,
    limit: Option<(DrawOrder, PrimitiveKind)>,
) -> bool {
    limit.is_none_or(|limit| (order, kind) < limit)
}

fn has_corner_smoothing(corner_smoothing: f32) -> bool {
    corner_smoothing > 0.0
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.backdrop_filters_iter.peek().map(|f| f.order),
                PrimitiveKind::BackdropFilter,
            ),
            (
                self.native_views_iter.peek().map(|v| v.order),
                PrimitiveKind::NativeView,
            ),
            (
                self.filter_boundaries_iter.peek().map(|b| b.order),
                // The same vec yields both start and end markers; the discriminant decides
                // where the next marker sorts relative to draw batches at an equal order
                // (start before content, end after).
                match self.filter_boundaries_iter.peek() {
                    Some(boundary) if boundary.is_start => PrimitiveKind::FilterBoundaryStart,
                    _ => PrimitiveKind::FilterBoundaryEnd,
                },
            ),
        ];
        // Find the two lowest live cursors in one fixed-size scan.
        let mut first = None;
        let mut second = None;
        for (order, kind) in orders_and_kinds {
            let Some(order) = order else {
                continue;
            };
            let candidate = (order, kind);
            if first.is_none_or(|current| candidate < current) {
                second = first;
                first = Some(candidate);
            } else if second.is_none_or(|current| candidate < current) {
                second = Some(candidate);
            }
        }
        let (_, batch_kind) = first?;
        let max_order_and_kind = second;

        match batch_kind {
            PrimitiveKind::Shadow => {
                let smoothed =
                    has_corner_smoothing(self.shadows_iter.peek().unwrap().corner_smoothing);
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| {
                        precedes_limit(shadow.order, batch_kind, max_order_and_kind)
                            && has_corner_smoothing(shadow.corner_smoothing) == smoothed
                    })
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows {
                    range: shadows_start..shadows_end,
                    smoothed,
                })
            }
            PrimitiveKind::Quad => {
                let smoothed =
                    has_corner_smoothing(self.quads_iter.peek().unwrap().corner_smoothing);
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| {
                        precedes_limit(quad.order, batch_kind, max_order_and_kind)
                            && has_corner_smoothing(quad.corner_smoothing) == smoothed
                    })
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads {
                    range: quads_start..quads_end,
                    smoothed,
                })
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| precedes_limit(path.order, batch_kind, max_order_and_kind))
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                let range = paths_start..paths_end;
                let paths = &self.paths[range.clone()];
                let tiles_start = paths[0].tile_start;
                let last = &paths[paths.len() - 1];
                let tiles = tiles_start..last.tile_start + last.tile_count();
                Some(PrimitiveBatch::Paths { range, tiles })
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| {
                        precedes_limit(underline.order, batch_kind, max_order_and_kind)
                    })
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        precedes_limit(sprite.order, batch_kind, max_order_and_kind)
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        precedes_limit(sprite.order, batch_kind, max_order_and_kind)
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let first_sprite = self.polychrome_sprites_iter.peek().unwrap();
                let texture_id = first_sprite.tile.texture_id;
                let smoothed = has_corner_smoothing(first_sprite.corner_smoothing);
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        precedes_limit(sprite.order, batch_kind, max_order_and_kind)
                            && sprite.tile.texture_id == texture_id
                            && has_corner_smoothing(sprite.corner_smoothing) == smoothed
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                    smoothed,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| {
                        precedes_limit(surface.order, batch_kind, max_order_and_kind)
                    })
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            PrimitiveKind::BackdropFilter => {
                let backdrop_filters_start = self.backdrop_filters_start;
                let mut backdrop_filters_end = backdrop_filters_start + 1;
                self.backdrop_filters_iter.next();
                while self
                    .backdrop_filters_iter
                    .next_if(|filter| precedes_limit(filter.order, batch_kind, max_order_and_kind))
                    .is_some()
                {
                    backdrop_filters_end += 1;
                }
                self.backdrop_filters_start = backdrop_filters_end;
                Some(PrimitiveBatch::BackdropFilters(
                    backdrop_filters_start..backdrop_filters_end,
                ))
            }
            PrimitiveKind::NativeView => {
                let native_views_start = self.native_views_start;
                let mut native_views_end = native_views_start + 1;
                self.native_views_iter.next();
                while self
                    .native_views_iter
                    .next_if(|view| precedes_limit(view.order, batch_kind, max_order_and_kind))
                    .is_some()
                {
                    native_views_end += 1;
                }
                self.native_views_start = native_views_end;
                Some(PrimitiveBatch::NativeViews(
                    native_views_start..native_views_end,
                ))
            }
            // Boundaries are emitted one at a time (never merged) so the renderer can switch
            // render targets at exactly the right point in the batch stream.
            PrimitiveKind::FilterBoundaryStart | PrimitiveKind::FilterBoundaryEnd => {
                let index = self.filter_boundaries_start;
                self.filter_boundaries_iter.next();
                self.filter_boundaries_start = index + 1;
                Some(PrimitiveBatch::FilterBoundary(index))
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub border_dashed_length: f32,
    pub border_dashed_gap: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Background,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    pub corner_smoothing: f32,
    /// Depth a depth-buffered renderer assigns while uploading the quad, greater values
    /// closer. Scene code leaves it at zero.
    pub depth: f32,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            order: Default::default(),
            border_style: Default::default(),
            border_dashed_length: DEFAULT_BORDER_DASHED_LENGTH,
            border_dashed_gap: DEFAULT_BORDER_DASHED_GAP,
            bounds: Default::default(),
            content_mask: Default::default(),
            background: Default::default(),
            border_color: Default::default(),
            corner_radii: Default::default(),
            border_widths: Default::default(),
            corner_smoothing: Default::default(),
            depth: Default::default(),
        }
    }
}

impl Quad {
    /// Whether the quad paints every pixel of its clipped bounds with one fully opaque color,
    /// exactly as an unblended rectangle of that color would. Only such quads may write depth
    /// and hide what is drawn behind them.
    ///
    /// The checks mirror the quad shader's simple path: a solid background, no border, no
    /// rounded or smoothed corners, and no content-mask fade. Alpha must be exactly one, so
    /// source-over blending and replacing the destination produce the same pixel.
    pub fn is_opaque_rectangle(&self) -> bool {
        self.background.tag == crate::BackgroundTag::Solid
            && self.background.solid.a == 1.0
            && self.border_widths == Edges::default()
            && self.corner_radii == Corners::default()
            && !has_corner_smoothing(self.corner_smoothing)
            && self.content_mask.fade_out == Edges::default()
    }

    /// The region where the quad paints exactly its fully opaque fill color, as a
    /// depth-buffered renderer may draw it unblended: the whole quad for an
    /// [opaque rectangle](Self::is_opaque_rectangle), the [border interior](Self::border_interior)
    /// of a quad with borders or rounded corners, `None` when no such region is certain.
    pub fn opaque_fill_bounds(&self) -> Option<Bounds<ScaledPixels>> {
        if self.background.tag != crate::BackgroundTag::Solid
            || self.background.solid.a != 1.0
            || has_corner_smoothing(self.corner_smoothing)
            || self.content_mask.fade_out != Edges::default()
        {
            return None;
        }
        if self.is_opaque_rectangle() {
            return Some(self.bounds);
        }
        let non_negative = |length: &ScaledPixels| length.0 >= 0.0;
        let widths = &self.border_widths;
        let radii = &self.corner_radii;
        if ![widths.top, widths.right, widths.bottom, widths.left]
            .iter()
            .chain(&[
                radii.top_left,
                radii.top_right,
                radii.bottom_right,
                radii.bottom_left,
            ])
            .all(non_negative)
        {
            return None;
        }
        let interior = self.border_interior();
        (!interior.is_empty()).then_some(interior)
    }

    /// The largest rectangle inside the quad that no border, corner or edge antialiasing
    /// reaches, so the quad paints only its fill there.
    pub(crate) fn border_interior(&self) -> Bounds<ScaledPixels> {
        let quad = self;
        let radii = &quad.corner_radii;
        let widths = &quad.border_widths;
        let reach_factor = 1.0 + quad.corner_smoothing;
        let edge_reaches = Edges {
            top: radii.top_left.max(radii.top_right) * reach_factor,
            right: radii.top_right.max(radii.bottom_right) * reach_factor,
            bottom: radii.bottom_left.max(radii.bottom_right) * reach_factor,
            left: radii.top_left.max(radii.bottom_left) * reach_factor,
        };

        let antialias_inset = point(ScaledPixels(1.0), ScaledPixels(1.0));
        let inset_bounds = |top_left_inset, bottom_right_inset| {
            Bounds::from_corners(
                quad.bounds.origin + top_left_inset + antialias_inset,
                quad.bounds.bottom_right() - bottom_right_inset - antialias_inset,
            )
        };

        // Rounded corners need only be excluded on one axis. Either candidate
        // is empty of border pixels, so use the larger interior.
        let horizontal_band = inset_bounds(
            point(widths.left, widths.top.max(edge_reaches.top)),
            point(widths.right, widths.bottom.max(edge_reaches.bottom)),
        );
        let vertical_band = inset_bounds(
            point(widths.left.max(edge_reaches.left), widths.top),
            point(widths.right.max(edge_reaches.right), widths.bottom),
        );

        let area = |bounds: &Bounds<ScaledPixels>| {
            bounds.size.width.0.max(0.) * bounds.size.height.0.max(0.)
        };
        if area(&horizontal_band) >= area(&vertical_band) {
            horizontal_band
        } else {
            vertical_band
        }
    }
}

/// Quad counts above this draw without depth-buffer occlusion: every depth step must stay
/// an exact multiple of a power of two below one.
pub const MAX_DEPTH_SORTED_QUADS: usize = (1 << 24) - 1;

/// The depth step between consecutive quads in a frame with `quad_count` quads.
pub fn quad_depth_scale(quad_count: usize) -> f32 {
    let denominator = (quad_count.min(MAX_DEPTH_SORTED_QUADS) + 1).next_power_of_two();
    1.0 / denominator as f32
}

/// Maps the index of a quad in [`Scene::quads`] to its depth, greater values closer.
/// A batch drawn after `quad_index` quads uses `quad_depth(quad_index, scale)` too: it
/// passes the depth test against every earlier quad and fails against the next one.
pub fn quad_depth(quad_index: usize, scale: f32) -> f32 {
    ((quad_index.min(MAX_DEPTH_SORTED_QUADS) + 1) as f32 * scale).min(1.0)
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub padding: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub thickness: ScaledPixels,
    pub wavy: ShaderBool,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Background,
    /// The shape that is blurred: a drop shadow's silhouette, or an inset shadow's hole.
    pub shape_bounds: Bounds<ScaledPixels>,
    pub shape_corner_radii: Corners<ScaledPixels>,
    /// Whether this shadow is rendered inside the element instead of outside it.
    pub inset: ShaderBool,
    pub corner_smoothing: f32,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// A backdrop filter blurs (and may otherwise filter) the content already rendered behind
/// `bounds`, compositing the result into a rounded rectangle — the frosted-glass effect.
/// Emitted by [`crate::Window::paint_backdrop_filter`]; produces the CSS `backdrop-filter` effect.
#[derive(Default, Debug, Clone)]
#[expect(missing_docs)]
pub struct BackdropFilter {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub corner_smoothing: f32,
    /// The filter chain applied to the backdrop, in scene (device-pixel) space. Identity filters
    /// are dropped at paint time, so a `BackdropFilter` is only emitted when this is non-empty.
    ///
    /// Inline capacity is 4: a `SmallVec<[ScaledFilter; 4]>` is the same size as capacity 1 here
    /// (the heap repr already occupies that space), so chains up to 4 filters avoid allocating
    /// at no extra struct size.
    pub filters: SmallVec<[ScaledFilter; 4]>,
    /// Element opacity captured at paint time, multiplied into the composited result.
    pub opacity: f32,
}

impl BackdropFilter {
    /// Largest gaussian blur radius in this filter chain, in device pixels.
    pub fn max_blur_radius(&self) -> f32 {
        max_blur_radius(&self.filters)
    }
}

impl From<BackdropFilter> for Primitive {
    fn from(filter: BackdropFilter) -> Self {
        Primitive::BackdropFilter(filter)
    }
}

/// The start or end marker of a content-filter or opacity isolation group. The element's
/// subtree is painted between a matched start/end pair; the renderer redirects that span into
/// an offscreen target, filters it, and composites it back at `bounds`. Produces the CSS
/// `filter` effect (e.g. blurring the element and its children as a single group).
#[derive(Debug, Clone)]
#[expect(missing_docs)]
pub struct FilterBoundary {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub corner_smoothing: f32,
    /// The filter chain applied to the isolated group, in scene (device-pixel) space. Identity
    /// filters are dropped at paint time. An empty chain represents an opacity-only group.
    /// Inline capacity 4 (same struct size as 1 here — see [`BackdropFilter::filters`]).
    pub filters: SmallVec<[ScaledFilter; 4]>,
    /// Opacity applied once when compositing the completed group into its parent.
    pub opacity: f32,
    /// `true` for the start marker (opens the group), `false` for the end marker (closes it).
    pub is_start: bool,
}

impl FilterBoundary {
    /// Largest gaussian blur radius in this filter chain, in device pixels.
    pub fn max_blur_radius(&self) -> f32 {
        max_blur_radius(&self.filters)
    }
}

/// Returns the largest blur radius in a scene-space filter chain.
///
/// This match is deliberately exhaustive so adding a filter requires one shared scheduling
/// decision instead of three backend-specific implementations that can drift.
fn max_blur_radius(filters: &[ScaledFilter]) -> f32 {
    filters.iter().fold(0.0, |radius, filter| match filter {
        ScaledFilter::Blur(filter_radius) => radius.max(filter_radius.0),
    })
}

impl From<FilterBoundary> for Primitive {
    fn from(boundary: FilterBoundary) -> Self {
        Primitive::FilterBoundary(boundary)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(u32)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// The smallest bounds that contain `bounds` once it is transformed.
    pub(crate) fn transform_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let [[xx, xy], [yx, yy]] = self.rotation_scale;
        let [translate_x, translate_y] = self.translation;
        let corners = [
            bounds.origin,
            bounds.top_right(),
            bounds.bottom_left(),
            bounds.bottom_right(),
        ]
        .map(|corner| {
            Point::new(
                ScaledPixels(xx * corner.x.0 + xy * corner.y.0 + translate_x),
                ScaledPixels(yx * corner.x.0 + yy * corner.y.0 + translate_y),
            )
        });
        let [first, rest @ ..] = corners;
        let top_left = rest.iter().fold(first, |min, corner| min.min(corner));
        let bottom_right = rest.iter().fold(first, |max, corner| max.max(corner));
        Bounds::from_corners(top_left, bottom_right)
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub padding: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl MonochromeSprite {
    /// The scene animation that moves this sprite, read from its padding slot.
    pub fn animation(&self) -> SpriteAnimationId {
        SpriteAnimationId(self.padding)
    }
}

/// Refers to an animation of the scene a sprite is inserted into.
///
/// Stored in [`MonochromeSprite::padding`] so the instance layout stays the
/// same on every backend; renderers ignore it. Zero means no animation, so
/// sprites built with `padding: 0` stay still.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpriteAnimationId(u32);

impl SpriteAnimationId {
    /// No animation.
    pub const NONE: Self = Self(0);

    fn for_index(index: usize) -> Self {
        Self(index as u32 + 1)
    }

    fn index(self) -> Option<usize> {
        (self.0 as usize).checked_sub(1)
    }
}

impl Default for SpriteAnimationId {
    fn default() -> Self {
        Self::NONE
    }
}

/// A [`TimedTransformation`] the window applies to a sprite right before
/// every present, so the scene can be presented again without being redrawn.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SpriteAnimation {
    transformation: TimedTransformation,
    center: Point<Pixels>,
    scale_factor: f32,
    base_color: SceneHsla,
    sprite_index: usize,
}

impl SpriteAnimation {
    pub(crate) fn new(
        transformation: TimedTransformation,
        center: Point<Pixels>,
        scale_factor: f32,
    ) -> Self {
        Self {
            transformation,
            center,
            scale_factor,
            base_color: SceneHsla::default(),
            sprite_index: usize::MAX,
        }
    }
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub padding: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub grayscale: ShaderBool,
    pub opacity: f32,
    pub corner_smoothing: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub source: crate::SurfaceSource,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

/// Where a platform view shows through the scene.
///
/// The view sits beneath the window's rendered content, so the renderer
/// erases everything painted earlier within these bounds, and everything
/// painted later covers the view. The view is therefore treated as opaque.
#[derive(Clone, Copy, Debug)]
pub struct PaintNativeView {
    /// The view's position among the scene's other primitives.
    pub order: DrawOrder,
    /// The view these bounds belong to.
    pub id: NativeViewId,
    /// The hitbox that decides whether a mouse position reaches the view.
    pub hitbox_id: HitboxId,
    /// Where the view is laid out, before clipping.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip applied to the view.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The color to show where the view has not caught up with these bounds,
    /// or a transparent color for none.
    pub background: Hsla,
}

impl From<PaintNativeView> for Primitive {
    fn from(native_view: PaintNativeView) -> Self {
        Primitive::NativeView(native_view)
    }
}

/// A monotone quadratic curve of a path, in device pixels: `p(t) = a*t^2 + b*t + p0`, from
/// `p0` to `p1`, with `sign` the winding it adds (zed #66809).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathCurve {
    pub p0: Point<f32>,
    pub p1: Point<f32>,
    pub a: Point<f32>,
    pub b: Point<f32>,
    pub sign: f32,
    /// Aligns the struct to 8 bytes.
    pub pad: u32,
}

/// How one painted path fills its tiles.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathPaint {
    /// The path's bounds, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Background,
    pub even_odd: ShaderBool,
    /// Aligns the struct to 8 bytes.
    pub pad: u32,
}

/// A rectangle of a path that shares one curve list and backdrop winding.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathTile {
    pub paint: u32,
    pub curve_start: u32,
    pub curve_count: u32,
    pub backdrop: i32,
    pub top_left: Point<f32>,
    pub bottom_right: Point<f32>,
    pub reference: Point<f32>,
    pub curves_top_left: Point<f32>,
    pub curves_bottom_right: Point<f32>,
}

/// A [`Path`] binned into tiles for one display scale, ready to draw.
#[derive(Clone, Debug)]
pub struct PaintedPath {
    pub(crate) order: DrawOrder,
    pub(crate) bounds: Bounds<ScaledPixels>,
    // Shared so replaying a cached frame does not copy the tiles.
    tiles: Arc<PathTiles>,
    pub(crate) paint: PathPaint,
    /// Index of this path's first tile in [`Scene::path_tiles`], set by [`Scene::finish`].
    pub(crate) tile_start: usize,
}

impl PaintedPath {
    /// Bins `path` into tiles, to be filled with `color` within `content_mask`.
    pub fn new(
        path: Path,
        scale_factor: f32,
        content_mask: ContentMask<Pixels>,
        color: Background,
    ) -> Self {
        let content_mask = content_mask.scale(scale_factor);
        let mut tiles = path.bin(scale_factor);
        tiles.retain_visible(content_mask.bounds);
        let paint = PathPaint {
            bounds: tiles.bounds(),
            content_mask,
            color,
            even_odd: path.even_odd().into(),
            pad: 0,
        };
        Self {
            order: 0,
            bounds: tiles.painted_bounds(),
            tiles: Arc::new(tiles),
            paint,
            tile_start: 0,
        }
    }

    /// The draw order the scene assigned this path.
    pub fn order(&self) -> DrawOrder {
        self.order
    }

    /// The device-pixel bounds this path may paint, including its antialiasing margin.
    pub fn bounds(&self) -> Bounds<ScaledPixels> {
        self.bounds
    }

    /// How this path fills its tiles.
    pub fn paint(&self) -> &PathPaint {
        &self.paint
    }

    /// The number of tiles this path draws.
    pub fn tile_count(&self) -> usize {
        self.tiles.tiles().len()
    }

    /// Appends this path's paint, curves and tiles, pointing the tiles at the appended paint
    /// and curves.
    pub fn append_to(
        &self,
        paints: &mut Vec<PathPaint>,
        curves: &mut Vec<PathCurve>,
        tiles: &mut Vec<PathTile>,
    ) {
        let paint_index = paints.len() as u32;
        let curve_start = curves.len() as u32;
        paints.push(self.paint);
        curves.extend_from_slice(self.tiles.curves());
        tiles.extend(self.tiles.tiles().iter().map(|tile| PathTile {
            paint: paint_index,
            curve_start: curve_start + tile.curve_start,
            ..*tile
        }));
    }
}

impl From<PaintedPath> for Primitive {
    fn from(path: PaintedPath) -> Self {
        Primitive::Path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AtlasTextureKind, DevicePixels, Point, ShaderBool, Size, SurfaceSource, TileId, point,
    };

    fn scaled(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    /// The permutation sort must match a stable `sort_by_key`: ties keep insertion order.
    #[test]
    fn sort_in_place_matches_stable_sort_by_key() {
        let mut keys = Vec::new();
        let mut seed = 0x9e37_79b9_u32;
        for len in [0usize, 1, 2, 7, 64, 513] {
            let mut items = (0..len as u32)
                .map(|index| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    (seed % 8, seed % 3, index)
                })
                .collect::<Vec<_>>();
            let mut expected = items.clone();
            expected.sort_by_key(|&(major, minor, _)| (major, minor));
            sort_in_place(&mut items, &mut keys, |&(major, minor, _)| (major, minor));
            assert_eq!(items, expected, "len {len}");
        }
    }

    /// Default primitives overlap so their draw orders follow insertion order.
    fn full_bounds() -> Bounds<ScaledPixels> {
        Bounds::new(Point::default(), Size::new(scaled(100.0), scaled(100.0)))
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: full_bounds(),
            ..Default::default()
        }
    }

    fn quad() -> Quad {
        Quad {
            bounds: full_bounds(),
            content_mask: mask(),
            ..Default::default()
        }
    }

    fn shadow() -> Shadow {
        Shadow {
            order: 0,
            blur_radius: scaled(0.0),
            bounds: full_bounds(),
            corner_radii: Corners::default(),
            content_mask: mask(),
            color: Default::default(),
            shape_bounds: full_bounds(),
            shape_corner_radii: Corners::default(),
            inset: ShaderBool::Disabled,
            corner_smoothing: 0.0,
        }
    }

    fn polychrome_sprite(texture_index: u32) -> PolychromeSprite {
        PolychromeSprite {
            order: 0,
            grayscale: ShaderBool::Disabled,
            opacity: 1.0,
            corner_smoothing: 0.0,
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: texture_index,
                    kind: AtlasTextureKind::Polychrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: Bounds::<DevicePixels>::default(),
            },
        }
    }

    fn batches(scene: &mut Scene) -> Vec<PrimitiveBatch> {
        scene.finish();
        scene.batches().collect()
    }

    /// A filled triangle whose top-left corner is at (`x`, `y`), 40 logical pixels wide.
    fn triangle(x: f32, y: f32) -> PaintedPath {
        let mut builder = crate::PathBuilder::fill();
        builder.move_to(point(crate::px(x), crate::px(y)));
        builder.line_to(point(crate::px(x + 40.0), crate::px(y)));
        builder.line_to(point(crate::px(x + 20.0), crate::px(y + 40.0)));
        builder.close();
        let content_mask = ContentMask {
            bounds: Bounds::new(
                Point::default(),
                Size::new(crate::px(400.0), crate::px(400.0)),
            ),
            ..Default::default()
        };
        PaintedPath::new(
            builder.build().unwrap(),
            1.0,
            content_mask,
            crate::black().into(),
        )
    }

    /// Each path batch's tile range is exactly its paths' tiles, and every tile points at its
    /// own path's paint and at curves inside the frame's curve array.
    #[test]
    fn path_batches_cover_their_paths_tiles() {
        let mut scene = Scene::default();
        scene.insert_primitive(triangle(10.0, 10.0));
        scene.insert_primitive(triangle(200.0, 10.0));
        let batches = batches(&mut scene);

        let [PrimitiveBatch::Paths { range, tiles }] = batches.as_slice() else {
            panic!("disjoint paths share one batch: {batches:?}");
        };
        assert_eq!(*range, 0..2);
        assert_eq!(*tiles, 0..scene.path_tiles.len());
        assert_eq!(scene.path_paints.len(), 2);

        let first_tiles = scene.paths[0].tile_count();
        assert!(first_tiles > 0 && first_tiles < scene.path_tiles.len());
        for (index, tile) in scene.path_tiles.iter().enumerate() {
            let expected_paint = u32::from(index >= first_tiles);
            assert_eq!(tile.paint, expected_paint, "tile {index}");
            let curves_end = tile.curve_start + tile.curve_count;
            assert!(
                curves_end as usize <= scene.path_curves.len(),
                "tile {index}"
            );
        }
    }

    /// A quad painted between two overlapping paths draws between them: the paths split into
    /// two batches around it, in paint order, with consecutive tile ranges.
    #[test]
    fn overlapping_paths_and_quads_draw_in_paint_order() {
        let mut scene = Scene::default();
        scene.insert_primitive(triangle(10.0, 10.0));
        scene.insert_primitive(Quad {
            bounds: full_bounds(),
            content_mask: mask(),
            ..Default::default()
        });
        scene.insert_primitive(triangle(20.0, 20.0));
        let batches = batches(&mut scene);

        let [
            PrimitiveBatch::Paths {
                range: first,
                tiles: first_tiles,
            },
            PrimitiveBatch::Quads { range: quads, .. },
            PrimitiveBatch::Paths {
                range: second,
                tiles: second_tiles,
            },
        ] = batches.as_slice()
        else {
            panic!("expected path, quad, path: {batches:?}");
        };
        assert_eq!(
            (first.clone(), quads.clone(), second.clone()),
            (0..1, 0..1, 1..2)
        );
        assert_eq!(first_tiles.start, 0);
        assert_eq!(first_tiles.end, second_tiles.start);
        assert_eq!(second_tiles.end, scene.path_tiles.len());
        // The quad covers the first path, so it needs a later order; the second path
        // draws after quads at a shared order, so it shares the quad's.
        assert!(scene.paths[0].order() < scene.quads[0].order);
        assert_eq!(scene.quads[0].order, scene.paths[1].order());
    }

    #[test]
    fn smoothing_and_texture_changes_define_batches() {
        let mut scene = Scene::default();
        for smoothing in [0.0, 0.0, 0.5, 1.0, 0.0] {
            let mut quad = quad();
            quad.corner_smoothing = smoothing;
            scene.insert_primitive(quad);
        }

        for smoothing in [0.0, 0.0, 0.5, 1.0, 0.0] {
            let mut shadow = shadow();
            shadow.corner_smoothing = smoothing;
            scene.insert_primitive(shadow);
        }

        for (texture, smoothing) in [
            (0, 0.0),
            (0, 0.0),
            (0, 0.5),
            (0, 1.0),
            (1, 1.0),
            (1, 0.5),
            (1, 0.0),
        ] {
            let mut sprite = polychrome_sprite(texture);
            sprite.corner_smoothing = smoothing;
            scene.insert_primitive(sprite);
        }

        let batch_signatures = batches(&mut scene)
            .into_iter()
            .map(|batch| match batch {
                PrimitiveBatch::Quads { range, smoothed } => ("quad", None, range, smoothed),
                PrimitiveBatch::Shadows { range, smoothed } => ("shadow", None, range, smoothed),
                PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range,
                    smoothed,
                } => ("polychrome", Some(texture_id.index), range, smoothed),
                other => panic!("unexpected batch: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            batch_signatures,
            vec![
                ("quad", None, 0..2, false),
                ("quad", None, 2..4, true),
                ("quad", None, 4..5, false),
                ("shadow", None, 0..2, false),
                ("shadow", None, 2..4, true),
                ("shadow", None, 4..5, false),
                ("polychrome", Some(0), 0..2, false),
                ("polychrome", Some(0), 2..4, true),
                ("polychrome", Some(1), 4..6, true),
                ("polychrome", Some(1), 6..7, false),
            ]
        );
    }

    /// A 100x100 quad whose bounds don't overlap `full_bounds()` (used to exercise the
    /// order-reuse path: non-overlapping content reuses low draw-orders).
    fn detached_quad() -> Quad {
        let bounds = Bounds::new(point(scaled(200.0), scaled(200.0)), full_bounds().size);

        Quad {
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn boundary(is_start: bool) -> FilterBoundary {
        FilterBoundary {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            corner_smoothing: 0.0,
            filters: smallvec::smallvec![ScaledFilter::Blur(scaled(8.0))],
            opacity: 1.0,
            is_start,
        }
    }

    fn backdrop() -> BackdropFilter {
        BackdropFilter {
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(scaled(20.0))],
            opacity: 1.0,
            ..Default::default()
        }
    }

    fn surface() -> PaintSurface {
        PaintSurface {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            source: SurfaceSource::Unsupported(Size::default()),
        }
    }

    #[test]
    fn backdrop_filter_sorts_before_a_later_overlapping_quad() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad());
        scene.insert_primitive(backdrop());
        scene.insert_primitive(quad());

        assert!(matches!(
            batches(&mut scene).as_slice(),
            [
                PrimitiveBatch::Quads { .. },
                PrimitiveBatch::BackdropFilters(_),
                PrimitiveBatch::Quads { .. }
            ]
        ));
    }

    #[test]
    fn isolation_commands_preserve_order_pairing_and_replay() {
        for active_layer in [false, true] {
            let mut scene = Scene::default();

            if active_layer {
                scene.push_layer();
            }

            scene.insert_primitive(quad());
            scene.insert_primitive(boundary(true));
            scene.insert_primitive(detached_quad());
            scene.insert_primitive(boundary(true));
            scene.insert_primitive(quad());
            scene.insert_primitive(boundary(true));
            scene.insert_primitive(surface());
            scene.insert_primitive(boundary(false));
            scene.insert_primitive(boundary(false));
            scene.insert_primitive(detached_quad());
            scene.insert_primitive(boundary(false));
            scene.insert_primitive(detached_quad());

            if active_layer {
                scene.pop_layer();
            }

            scene.finish();
            assert_eq!(scene.render_plan().requirements().isolated_target_count, 3);

            let commands: Vec<_> = scene
                .render_commands()
                .iter()
                .map(|command| match command {
                    RenderCommand::BeginFilter {
                        boundary_index,
                        target: FilterRenderTarget::Isolated(target),
                    } => ("start", Some((*boundary_index, target.as_usize()))),
                    RenderCommand::EndFilter {
                        boundary_index,
                        target: FilterRenderTarget::Isolated(target),
                        ..
                    } => ("end", Some((*boundary_index, target.as_usize()))),
                    RenderCommand::Batch(PrimitiveBatch::Quads { .. }) => ("quad", None),
                    RenderCommand::Batch(PrimitiveBatch::Surfaces(_)) => ("surface", None),
                    other => panic!("unexpected command: {other:?}"),
                })
                .collect();
            assert_eq!(
                commands,
                [
                    ("quad", None),
                    ("start", Some((0, 0))),
                    ("quad", None),
                    ("start", Some((1, 1))),
                    ("quad", None),
                    ("start", Some((2, 2))),
                    ("surface", None),
                    ("end", Some((2, 2))),
                    ("end", Some((1, 1))),
                    ("quad", None),
                    ("end", Some((0, 0))),
                    ("quad", None),
                ]
            );

            let mut replay = Scene::default();
            replay.replay(0..scene.len(), &scene);
            replay.finish();
            assert_eq!(replay.render_commands(), scene.render_commands());
            assert_eq!(replay.surface_opacities(), &[1.0]);
            assert!(replay.layer_stack.is_empty());
            assert!(!replay.is_in_filter_group());
        }
    }

    #[test]
    fn unmatched_filter_start_renders_inline() {
        let mut scene = Scene::default();
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.finish();

        let mut commands = scene.render_commands().iter();
        assert!(matches!(
            commands.next(),
            Some(RenderCommand::BeginFilter {
                target: FilterRenderTarget::Inline,
                ..
            })
        ));
        assert!(matches!(
            commands.next(),
            Some(RenderCommand::Batch(PrimitiveBatch::Quads { .. }))
        ));
        assert!(commands.next().is_none());
        assert!(!scene.requires_offscreen_rendering());
    }

    #[test]
    fn finish_rebuilds_the_plan_after_direct_primitive_mutation() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad());
        scene.finish();
        let commands = scene.render_commands().to_vec();

        // Primitive storage is public, so direct mutation must change the compiled plan.
        scene.quads.push(quad());
        scene.finish();
        assert_ne!(scene.render_commands(), commands);
        assert_eq!(scene.render_plan().requirements().instance_batch_count, 1);
    }

    fn monochrome_sprite(texture_index: u32) -> MonochromeSprite {
        MonochromeSprite {
            order: 0,
            padding: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            color: SceneHsla {
                h: 0.0,
                s: 1.0,
                l: 0.5,
                a: 0.8,
            },
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: texture_index,
                    kind: AtlasTextureKind::Monochrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: Bounds::<DevicePixels>::default(),
            },
            transformation: TransformationMatrix::unit(),
        }
    }

    /// Advancing animations moves sprites in place after `finish` sorted them, so the
    /// compiled render plan must still match and the static sprite must not move.
    #[test]
    fn advancing_animations_keeps_the_render_plan_valid_after_sorting() {
        let spin = SpriteAnimation::new(
            TimedTransformation::rotation(
                std::time::Duration::from_secs(1),
                crate::Transformation::default(),
            )
            .to(
                crate::Transformation::rotate(crate::radians(std::f32::consts::TAU))
                    .with_alpha(0.0),
            ),
            Point::default(),
            1.0,
        );
        let mut scene = Scene::default();
        // One layer shares a draw order, so sorting by texture swaps the two sprites.
        scene.push_layer();
        scene.insert_animated_sprite(monochrome_sprite(2), spin);
        scene.insert_primitive(monochrome_sprite(1));
        scene.pop_layer();
        scene.finish();
        let commands = scene.render_commands().to_vec();
        assert_eq!(
            scene.monochrome_sprites[0].animation(),
            SpriteAnimationId::NONE
        );
        assert_ne!(
            scene.monochrome_sprites[1].animation(),
            SpriteAnimationId::NONE
        );

        scene.advance_animations(std::time::Duration::from_millis(250));

        assert_eq!(scene.render_commands(), commands);
        let still = &scene.monochrome_sprites[0];
        assert_eq!(still.transformation, TransformationMatrix::unit());
        assert_eq!(still.color.a, 0.8);
        let moved = &scene.monochrome_sprites[1];
        assert_eq!(moved.tile.texture_id.index, 2);
        assert!((moved.transformation.rotation_scale[0][0]).abs() < 1e-5);
        assert!((moved.transformation.rotation_scale[1][0] - 1.0).abs() < 1e-5);
        assert!((moved.color.a - 0.6).abs() < 1e-5);

        // Replaying keeps the animation, re-indexed into the new scene.
        let mut replayed = Scene::default();
        replayed.replay(0..scene.len(), &scene);
        replayed.finish();
        assert_eq!(replayed.animations.len(), 1);
        replayed.advance_animations(std::time::Duration::from_millis(500));
        assert!(
            (replayed.monochrome_sprites[1].transformation.rotation_scale[0][0] + 1.0).abs() < 1e-5
        );
    }

    #[test]
    fn render_plan_reuses_its_command_allocation_across_frames() {
        let mut scene = Scene::default();
        for _ in 0..32 {
            scene.insert_primitive(quad());
        }
        scene.finish();
        let capacity = scene.render_plan.commands.capacity();
        assert!(capacity >= 32);

        scene.clear();
        assert_eq!(scene.render_plan.commands.capacity(), capacity);
        scene.insert_primitive(quad());
        scene.finish();
        assert_eq!(scene.render_plan.commands.capacity(), capacity);
    }

    #[test]
    fn maximum_draw_order_is_not_treated_as_an_empty_batch_cursor() {
        let mut scene = Scene::default();
        let mut quad = quad();
        quad.order = DrawOrder::MAX;
        scene.quads.push(quad);
        scene.finish();

        assert!(matches!(
            scene.render_commands(),
            [RenderCommand::Batch(PrimitiveBatch::Quads { range, smoothed: false })]
                if range == &(0..1)
        ));
    }

    #[test]
    fn filter_requirements_distinguish_opacity_from_blur() {
        for (name, group, content_blur, backdrop_blur) in [
            ("ordinary quad", false, false, false),
            ("opacity", true, false, false),
            ("content blur", true, true, false),
            ("backdrop blur", false, false, true),
            ("backdrop inside opacity", true, false, true),
        ] {
            let mut scene = Scene::default();
            let mut start = boundary(true);

            if !content_blur {
                start.filters.clear();
                start.opacity = 0.5;
            }

            if group {
                scene.insert_primitive(start.clone());
            }

            if backdrop_blur {
                scene.insert_primitive(backdrop());
            }

            scene.insert_primitive(quad());

            if group {
                scene.insert_primitive(FilterBoundary {
                    is_start: false,
                    ..start
                });
            }

            scene.finish();

            let requirements = scene.render_plan().requirements();
            assert_eq!(
                (
                    requirements.uses_offscreen_target,
                    requirements.uses_blur_target
                ),
                (group || backdrop_blur, content_blur || backdrop_blur),
                "{name}"
            );
        }
    }

    fn square(x: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(scaled(x), scaled(0.)),
            Size::new(scaled(10.), scaled(10.)),
        )
    }

    fn square_quad(x: f32) -> Quad {
        Quad {
            bounds: square(x),
            content_mask: ContentMask {
                bounds: square(x),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn square_underline(x: f32) -> Underline {
        Underline {
            order: 0,
            padding: 0,
            bounds: square(x),
            content_mask: ContentMask {
                bounds: square(x),
                ..Default::default()
            },
            color: Default::default(),
            thickness: scaled(1.),
            wavy: ShaderBool::Disabled,
        }
    }

    fn batch_labels(scene: &mut Scene) -> Vec<String> {
        scene.finish();
        scene.batches().map(|batch| batch.label()).collect()
    }

    #[test]
    fn a_native_view_in_a_layer_is_covered_by_what_the_layer_paints_after_it() {
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(square_quad(0.));
        scene.insert_primitive(PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id: HitboxId::placeholder(),
            bounds: square(0.),
            content_mask: ContentMask {
                bounds: square(0.),
                ..Default::default()
            },
            background: Hsla::default(),
        });
        scene.insert_primitive(square_quad(0.));
        // A quad beside the view shares nothing with it, so it isn't held back.
        scene.insert_primitive(square_quad(20.));
        scene.pop_layer();
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "native views (1)", "quads (2)"]
        );
    }

    #[test]
    fn overlapping_primitives_share_an_order_when_their_kind_draws_later() {
        // An underline over a quad draws after it at the same order, so both quads
        // and the underline fit in two batches instead of three.
        let mut scene = Scene::default();
        scene.insert_primitive(square_quad(0.));
        scene.insert_primitive(square_underline(0.));
        scene.insert_primitive(square_quad(20.));
        assert_eq!(batch_labels(&mut scene), ["quads (2)", "underlines (1)"]);
        assert_eq!(scene.quads[0].order, scene.underlines[0].order);

        // A quad over an underline would draw below it at a shared order, so it
        // takes the next one.
        let mut scene = Scene::default();
        scene.insert_primitive(square_underline(0.));
        scene.insert_primitive(square_quad(5.));
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
        assert!(scene.quads[0].order > scene.underlines[0].order);
    }

    #[test]
    fn translucent_primitives_of_one_kind_blend_in_paint_order() {
        // Three overlapping translucent quads share one order and one batch; the
        // batch keeps paint order, so each blends over the one painted before it.
        let mut scene = Scene::default();
        for (index, alpha) in [0.25, 0.5, 0.75].into_iter().enumerate() {
            scene.insert_primitive(Quad {
                background: Background::from(crate::hsla(0., 1., 0.5, alpha)),
                ..square_quad(index as f32 * 2.)
            });
        }
        assert_eq!(batch_labels(&mut scene), ["quads (3)"]);
        let alphas: Vec<f32> = scene
            .quads
            .iter()
            .map(|quad| quad.background.as_solid().unwrap().alpha)
            .collect();
        assert_eq!(alphas, [0.25, 0.5, 0.75]);
    }

    #[test]
    fn overlapping_sprites_never_share_an_order() {
        // Sprites sort by atlas texture within an order, which would lose paint order.
        let mut scene = Scene::default();
        scene.insert_primitive(monochrome_sprite(2));
        scene.insert_primitive(monochrome_sprite(1));
        scene.insert_primitive(polychrome_sprite(2));
        scene.insert_primitive(polychrome_sprite(1));
        scene.finish();
        assert!(scene.monochrome_sprites[0].order < scene.monochrome_sprites[1].order);
        assert_eq!(scene.monochrome_sprites[0].tile.texture_id.index, 2);
        // Polychrome sprites draw after monochrome ones, so they may share an order.
        assert_eq!(
            scene.monochrome_sprites[1].order,
            scene.polychrome_sprites[0].order
        );
        assert!(scene.polychrome_sprites[0].order < scene.polychrome_sprites[1].order);
        assert_eq!(scene.polychrome_sprites[0].tile.texture_id.index, 2);

        // A sprite still shares the order of an overlapped kind that draws before it.
        let mut scene = Scene::default();
        scene.insert_primitive(quad());
        scene.insert_primitive(monochrome_sprite(1));
        scene.finish();
        assert_eq!(scene.quads[0].order, scene.monochrome_sprites[0].order);
    }

    #[test]
    fn overlapping_backdrop_filters_never_share_an_order() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad());
        scene.insert_primitive(backdrop());
        scene.insert_primitive(backdrop());
        scene.finish();
        // The first backdrop blurs the quad at the quad's order; the second must see
        // the first one's result, so it draws at a later order.
        assert_eq!(scene.backdrop_filters[0].order, scene.quads[0].order);
        assert!(scene.backdrop_filters[1].order > scene.backdrop_filters[0].order);
    }

    #[test]
    fn transformed_sprites_order_against_their_transformed_bounds() {
        let rotate = TransformationMatrix::unit().rotate(Radians(std::f32::consts::FRAC_PI_4));
        let bounds = Bounds::new(
            point(scaled(0.), scaled(0.)),
            Size::new(scaled(10.), scaled(10.)),
        );
        let rotated = rotate.transform_bounds(bounds);
        let half_diagonal = 10. * std::f32::consts::FRAC_1_SQRT_2;
        assert!((rotated.size.width.0 - 2. * half_diagonal).abs() < 1e-4);
        assert!((rotated.size.height.0 - 2. * half_diagonal).abs() < 1e-4);
        assert_eq!(
            TransformationMatrix::unit().transform_bounds(bounds),
            bounds
        );

        let shifted = TransformationMatrix::unit().translate(point(scaled(50.), scaled(0.)));
        let mut sprite = monochrome_sprite(1);
        sprite.bounds = bounds;
        sprite.transformation = shifted;
        assert_eq!(
            Primitive::from(sprite).clipped_bounds(),
            Bounds::new(point(scaled(50.), scaled(0.)), bounds.size)
        );

        // The sprite only overlaps the second quad once it is translated, so a quad
        // painted over the second quad's area afterwards goes above the sprite.
        let mut scene = Scene::default();
        scene.insert_primitive(square_quad(50.));
        scene.insert_primitive(sprite);
        scene.insert_primitive(square_quad(52.));
        scene.finish();
        assert!(scene.quads[1].order > scene.monochrome_sprites[0].order);
    }

    #[test]
    fn layers_are_drawn_in_paint_order() {
        // A quad painted after a layer goes above it.
        let mut scene = Scene::default();
        scene.insert_primitive(square_quad(0.));
        scene.push_layer();
        scene.insert_primitive(square_underline(0.));
        scene.insert_primitive(square_underline(5.));
        scene.pop_layer();
        scene.insert_primitive(square_quad(5.));
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "underlines (2)", "quads (1)"]
        );

        // A layer goes above the layer painted before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(square_underline(0.));
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(square_quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }

    #[test]
    fn nested_layers_are_drawn_above_their_parents() {
        // A layer has one draw order, so what it paints after a nested layer
        // still goes below that layer.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(square_underline(0.));
        scene.push_layer();
        scene.insert_primitive(square_quad(5.));
        scene.pop_layer();
        scene.insert_primitive(square_underline(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (2)", "quads (1)"]);

        // The same holds when the nested layer shares its parent's order.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(square_quad(0.));
        scene.push_layer();
        scene.insert_primitive(square_quad(5.));
        scene.pop_layer();
        scene.insert_primitive(square_quad(8.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["quads (3)"]);
        assert_eq!(
            scene
                .quads
                .iter()
                .map(|quad| quad.bounds)
                .collect::<Vec<_>>(),
            [square(0.), square(8.), square(5.)]
        );

        // A layer goes above the layers nested in the layer before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.push_layer();
        scene.insert_primitive(square_underline(0.));
        scene.pop_layer();
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(square_quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }

    #[test]
    fn a_filter_boundary_splits_an_open_layer() {
        // The layer's quad painted before the group draws below it, and the one
        // painted after the group closes draws above the group.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));
        scene.insert_primitive(quad());
        scene.pop_layer();
        scene.finish();
        let [start, end] = [0, 1].map(|index| scene.filter_boundaries[index].order);
        let orders: Vec<_> = scene.quads.iter().map(|quad| quad.order).collect();
        assert!(orders[0] < start);
        assert!(start < orders[1] && orders[1] < end);
        assert!(end < orders[2]);

        // A layer opened inside the split layer still orders above what it covers.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(boundary(true));
        scene.push_layer();
        scene.insert_primitive(square_underline(0.));
        scene.pop_layer();
        scene.insert_primitive(square_quad(5.));
        scene.insert_primitive(boundary(false));
        scene.pop_layer();
        scene.finish();
        assert!(scene.underlines[0].order < scene.quads[0].order);
        assert!(scene.quads[0].order < scene.filter_boundaries[1].order);
    }

    #[test]
    fn empty_and_clipped_layers_take_no_order() {
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(Quad {
            content_mask: ContentMask {
                bounds: square(50.),
                ..Default::default()
            },
            ..square_quad(0.)
        });
        scene.pop_layer();
        scene.push_layer();
        scene.pop_layer();
        scene.insert_primitive(square_quad(0.));
        scene.finish();
        assert_eq!(scene.quads.len(), 1);
        assert_eq!(scene.quads[0].order, 1);
    }

    #[test]
    fn replaying_layers_reproduces_their_orders() {
        let mut scene = Scene::default();
        scene.insert_primitive(square_quad(0.));
        scene.push_layer();
        scene.insert_primitive(square_underline(0.));
        scene.push_layer();
        scene.insert_primitive(square_quad(2.));
        scene.pop_layer();
        scene.pop_layer();
        scene.finish();

        let mut replayed = Scene::default();
        replayed.replay(0..scene.len(), &scene);
        replayed.finish();
        assert_eq!(
            replayed
                .batches()
                .map(|batch| batch.label())
                .collect::<Vec<_>>(),
            scene
                .batches()
                .map(|batch| batch.label())
                .collect::<Vec<_>>()
        );
        let orders = |scene: &Scene| {
            (
                scene
                    .quads
                    .iter()
                    .map(|quad| quad.order)
                    .collect::<Vec<_>>(),
                scene.underlines[0].order,
            )
        };
        assert_eq!(orders(&replayed), orders(&scene));
    }

    fn opaque_quad() -> Quad {
        Quad {
            background: crate::solid_background(crate::red()),
            ..quad()
        }
    }

    #[test]
    fn only_provably_opaque_quads_are_opaque_rectangles() {
        let translucent = |alpha| {
            let mut color = crate::red();
            color.alpha = alpha;
            Quad {
                background: crate::solid_background(color),
                ..quad()
            }
        };
        let faded_mask = ContentMask {
            fade_out: Edges {
                left: scaled(4.0),
                ..Default::default()
            },
            ..mask()
        };
        assert!(opaque_quad().is_opaque_rectangle());
        // A default background is opaque black, as the shaders have always drawn it.
        assert!(quad().is_opaque_rectangle());
        let not_opaque = [
            ("transparent", translucent(0.0)),
            ("translucent", translucent(0.999)),
            ("NaN alpha", translucent(f32::NAN)),
            (
                "gradient",
                Quad {
                    background: crate::linear_gradient(
                        0.0,
                        crate::linear_color_stop(crate::red(), 0.0),
                        crate::linear_color_stop(crate::blue(), 1.0),
                    ),
                    ..quad()
                },
            ),
            (
                "pattern",
                Quad {
                    background: crate::pattern_slash(crate::red(), 2.0, 2.0),
                    ..quad()
                },
            ),
            (
                "bordered",
                Quad {
                    border_widths: Edges::all(scaled(1.0)),
                    ..opaque_quad()
                },
            ),
            (
                "rounded",
                Quad {
                    corner_radii: Corners::all(scaled(4.0)),
                    ..opaque_quad()
                },
            ),
            (
                "smoothed",
                Quad {
                    corner_smoothing: 0.5,
                    ..opaque_quad()
                },
            ),
            (
                "faded mask",
                Quad {
                    content_mask: faded_mask,
                    ..opaque_quad()
                },
            ),
        ];
        for (name, quad) in not_opaque {
            assert!(!quad.is_opaque_rectangle(), "{name}");
        }
    }

    #[test]
    fn opaque_fill_bounds_keep_clear_of_borders_corners_and_antialiasing() {
        assert_eq!(opaque_quad().opaque_fill_bounds(), Some(full_bounds()));

        let card = Quad {
            border_widths: Edges::all(scaled(2.0)),
            border_color: crate::blue().into(),
            corner_radii: Corners::all(scaled(10.0)),
            ..opaque_quad()
        };
        let interior = card
            .opaque_fill_bounds()
            .expect("a large card has an interior");
        assert!(!card.is_opaque_rectangle());
        // The interior clears the 10px corners on one axis and, on both, the 2px border
        // plus one antialiasing pixel.
        let (origin, corner) = (interior.origin, interior.bottom_right());
        assert!(
            origin.x >= scaled(3.0) && origin.y >= scaled(3.0),
            "{interior:?}"
        );
        assert!(
            corner.x <= scaled(97.0) && corner.y <= scaled(97.0),
            "{interior:?}"
        );
        assert!(
            origin.x >= scaled(11.0) || origin.y >= scaled(11.0),
            "{interior:?} reaches a corner"
        );
        assert_eq!(interior, card.border_interior());

        for (name, quad) in [
            (
                "translucent fill",
                Quad {
                    background: crate::solid_background(crate::transparent_black()),
                    ..card
                },
            ),
            (
                "smoothed corners",
                Quad {
                    corner_smoothing: 0.6,
                    ..card
                },
            ),
            (
                "negative border",
                Quad {
                    border_widths: Edges {
                        top: scaled(-1.0),
                        ..card.border_widths
                    },
                    ..card
                },
            ),
            (
                "negative radius",
                Quad {
                    corner_radii: Corners {
                        top_left: scaled(-3.0),
                        ..card.corner_radii
                    },
                    ..card
                },
            ),
            (
                "border wider than the quad",
                Quad {
                    border_widths: Edges::all(scaled(60.0)),
                    ..card
                },
            ),
        ] {
            assert_eq!(quad.opaque_fill_bounds(), None, "{name}");
        }
    }

    #[test]
    fn quad_depths_are_exact_increasing_and_below_one() {
        for count in [0usize, 1, 2, 3, 255, 256, 1000, MAX_DEPTH_SORTED_QUADS] {
            let scale = quad_depth_scale(count);
            assert!(scale.is_finite() && scale > 0.0, "{count}");
            assert!(
                quad_depth(0, scale) > 0.0,
                "the first quad passes a cleared buffer"
            );
            assert!(quad_depth(count, scale) <= 1.0, "{count}");
            let indices = if count == 0 {
                Vec::new()
            } else {
                vec![0, count / 2, count - 1]
            };
            for index in indices {
                let (this, next) = (quad_depth(index, scale), quad_depth(index + 1, scale));
                assert!(this < next, "{count}: {index}");
                // Each depth is an exact multiple of the scale, so no two round together.
                assert_eq!(this / scale, (index + 1) as f32, "{count}: {index}");
            }
        }
    }

    #[test]
    fn opaque_quads_are_listed_front_to_back_per_scene_pass() {
        let group = |is_start, filters: SmallVec<[ScaledFilter; 4]>| FilterBoundary {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            corner_smoothing: 0.0,
            filters,
            opacity: 0.5,
            is_start,
        };
        let mut scene = Scene::default();
        scene.insert_primitive(opaque_quad()); // 0
        scene.insert_primitive(Quad {
            background: crate::solid_background(crate::transparent_black()),
            ..quad()
        }); // 1: transparent, never listed
        scene.insert_primitive(opaque_quad()); // 2
        scene.insert_primitive(BackdropFilter {
            bounds: full_bounds(),
            content_mask: mask(),
            filters: smallvec::smallvec![ScaledFilter::Blur(scaled(4.0))],
            opacity: 1.0,
            ..Default::default()
        });
        scene.insert_primitive(opaque_quad()); // 3
        scene.insert_primitive(group(true, SmallVec::new()));
        scene.insert_primitive(opaque_quad()); // 4
        scene.insert_primitive(Quad {
            corner_radii: Corners::all(scaled(4.0)),
            ..opaque_quad()
        }); // 5: listed for its interior
        scene.insert_primitive(group(false, SmallVec::new()));
        scene.insert_primitive(opaque_quad()); // 6
        scene.finish();

        let plan = scene.render_plan();
        let passes = (0..plan.scene_pass_count())
            .map(|pass| plan.opaque_quads(pass).to_vec())
            .collect::<Vec<_>>();
        assert_eq!(passes, [vec![2, 0], vec![3], vec![5, 4], vec![6]]);
        assert_eq!(plan.requirements().opaque_quad_count, 6);
        assert_eq!(plan.requirements().opaque_quad_pass_count, 4);

        scene.disable_opaque_quads();
        let plan = scene.render_plan();
        assert_eq!(plan.scene_pass_count(), 4);
        assert!((0..4).all(|pass| plan.opaque_quads(pass).is_empty()));
        assert_eq!(plan.requirements().opaque_quad_count, 0);
    }
}
