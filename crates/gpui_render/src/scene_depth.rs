use gpui::{
    ContentMask, Corners, Edges, PrimitiveBatch, Quad, Scene, quad_depth, quad_depth_scale,
};

/// Backend-neutral depth bookkeeping for the opaque pre-pass, shared by every renderer
/// with a depth buffer so they agree on each primitive's depth.
///
/// Quad `i` of [`Scene::quads`] draws at `quad_depth(i)`. Every other batch draws at the depth
/// of the next quad, through the viewport's depth range, so it hides behind that quad and
/// every later one, and shows in front of earlier ones: the frame's paint order. Opaque fills
/// write depth first, front to back, so nothing behind them is shaded.
#[derive(Debug)]
pub struct SceneDepth {
    scale: f32,
    /// Index of the next quad in paint order.
    quad_cursor: usize,
    scene_pass: usize,
    hides_opaque_rectangles: bool,
}

impl SceneDepth {
    pub fn new(scene: &Scene) -> Self {
        Self {
            scale: quad_depth_scale(scene.quads.len()),
            quad_cursor: 0,
            scene_pass: 0,
            hides_opaque_rectangles: scene.render_plan().requirements().opaque_quad_count > 0,
        }
    }

    /// Index of the current scene pass, for [`gpui::ScenePlan::opaque_quads`].
    pub fn scene_pass(&self) -> usize {
        self.scene_pass
    }

    /// Call after every command for which [`gpui::RenderCommand::ends_scene_pass`] holds.
    pub fn begin_scene_pass(&mut self) {
        self.scene_pass += 1;
    }

    pub fn quad_depth(&self, quad_index: usize) -> f32 {
        quad_depth(quad_index, self.scale)
    }

    /// Whether the quad was drawn whole by its pass's opaque fills, so the blended pass
    /// leaves it out.
    pub fn hides(&self, quad: &Quad) -> bool {
        self.hides_opaque_rectangles && quad.is_opaque_rectangle()
    }

    /// Advances past `batch` and returns the depth its fragments are pinned to through the
    /// viewport: `None` for quads, which carry their own depth.
    pub fn enter_batch(&mut self, batch: &PrimitiveBatch) -> Option<f32> {
        if let PrimitiveBatch::Quads { range, .. } = batch {
            debug_assert_eq!(range.start, self.quad_cursor, "quads draw in index order");
            self.quad_cursor = range.end;
            None
        } else {
            Some(self.quad_depth(self.quad_cursor))
        }
    }

    /// The current scene pass's opaque fill instances, front to back, for the
    /// `opaque_quads` pipeline.
    pub fn opaque_fills<'a>(
        &'a self,
        scene: &'a Scene,
    ) -> impl ExactSizeIterator<Item = Quad> + 'a {
        let opaque_quads = scene.render_plan().opaque_quads(self.scene_pass);
        opaque_quads.iter().map(move |&index| {
            let index = index as usize;
            let quad = &scene.quads[index];
            // An opaque rectangle keeps its own mask, so it covers exactly the pixels it would
            // cover blended. Other quads fill only their interior, which the blended quad
            // then skips by failing the depth test.
            let content_mask = if quad.is_opaque_rectangle() {
                quad.content_mask
            } else {
                let fill = quad
                    .opaque_fill_bounds()
                    .expect("the scene plan lists only quads with an opaque fill");
                ContentMask {
                    bounds: quad.content_mask.bounds.intersect(&fill),
                    fade_out: Edges::default(),
                }
            };
            Quad {
                content_mask,
                border_widths: Edges::default(),
                corner_radii: Corners::default(),
                depth: self.quad_depth(index),
                ..*quad
            }
        })
    }

    /// Number of quads [`Self::blended_quads`] yields for the same slice.
    pub fn blended_quad_count(&self, quads: &[Quad]) -> usize {
        quads.iter().filter(|quad| !self.hides(quad)).count()
    }

    /// Quads `first_index..` of the scene at their depths, leaving out the opaque rectangles
    /// the pass's opaque fills already drew.
    pub fn blended_quads<'a>(
        &'a self,
        quads: &'a [Quad],
        first_index: usize,
    ) -> impl Iterator<Item = Quad> + 'a {
        quads
            .iter()
            .enumerate()
            .filter(|(_, quad)| !self.hides(quad))
            .map(move |(offset, quad)| Quad {
                depth: self.quad_depth(first_index + offset),
                ..*quad
            })
    }
}
