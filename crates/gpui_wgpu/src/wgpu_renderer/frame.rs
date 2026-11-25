use super::{
    WgpuRenderer,
    buffers::{InstanceTransport, InstanceUpload},
    filters::{FILTER_UNIFORMS_PER_COMPOSITE, FrameUniformRequirements},
};
use gpui::{
    ContentMask, Corners, Edges, FilterRenderTarget, MonochromeSprite, PathCurve, PathPaint,
    PathTile, PolychromeSprite, PrimitiveBatch, Quad, RenderCommand, Scene, ScenePlanRequirements,
    Shadow, SubpixelSprite, Underline, quad_depth, quad_depth_scale,
};
use gpui_render::TargetStack;
use gpui_render::blur::{FilterCompositeClip, FilterCompositeParameters};
use gpui_render::shaders::{
    common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool},
    interface as shader_interface,
};

pub(super) fn render_to_view(
    renderer: &mut WgpuRenderer,
    scene: &Scene,
    frame_view: &wgpu::TextureView,
    readback: Option<ReadbackCopy<'_>>,
) -> Option<wgpu::SubmissionIndex> {
    let Some(targets) = PreparedTargets::prepare(renderer, scene, frame_view) else {
        return None;
    };
    // Prune once against the complete scene. Doing this inside each draw batch would evict a
    // texture that reappears after an intervening primitive and recreate its platform view.
    renderer.retain_surface_cache(&scene.surfaces);

    match FrameEncoder::new(renderer, scene, targets).encode(readback) {
        Ok(command_buffer) => Some(renderer.resources().queue.submit([command_buffer])),
        Err(DrawError::ExternalSurface) => None,
        Err(DrawError::CapacityPlanningInvariant) => {
            log::error!("frame storage exceeded its precomputed capacity");
            None
        }
    }
}

pub(super) struct ReadbackCopy<'a> {
    pub(super) texture: &'a wgpu::Texture,
    pub(super) buffer: &'a wgpu::Buffer,
    pub(super) bytes_per_row: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

struct PreparedTargets {
    active: wgpu::TextureView,
    presentation: wgpu::TextureView,
    offscreen: Option<wgpu::TextureView>,
    depth: wgpu::TextureView,
    instances: InstanceUpload,
}

impl PreparedTargets {
    fn prepare(
        renderer: &mut WgpuRenderer,
        scene: &Scene,
        frame_view: &wgpu::TextureView,
    ) -> Option<Self> {
        if !begin_frame(renderer) {
            return None;
        }
        let requirements = {
            let transport = renderer.resources().instances.transport();
            FrameRequirements::for_scene(scene, transport)
        };
        let device = renderer.resources().device.clone();
        let resources = renderer.resources_mut();
        if !resources.instances.ensure_capacity(
            &device,
            &resources.bind_group_layouts,
            requirements.storage_bytes,
            requirements.instance_batches,
        ) {
            return None;
        }
        let mut instances = {
            let resources = renderer.resources();
            resources.instances.begin_upload(
                &resources.queue,
                requirements.storage_bytes,
                requirements.instance_batches,
            )?
        };
        // Reserved first by `FrameRequirements::for_scene`, so the planned padding holds.
        if requirements.planned.path_tile_count > 0 {
            instances.write_path_frame_data(&scene.path_curves, &scene.path_paints)?;
        }
        if !renderer.ensure_uniform_capacity(requirements.uniforms) {
            return None;
        }

        if requirements.planned.uses_offscreen_target {
            renderer.ensure_filter_textures(&requirements.planned);
        }
        write_shader_globals(renderer);
        let depth = renderer.ensure_depth_texture();

        if requirements.planned.uses_offscreen_target {
            let resources = renderer.resources();
            let offscreen = resources
                .scene_color_view
                .as_ref()
                .expect("blur texture preparation must create a scene target")
                .clone();
            Some(Self {
                active: offscreen.clone(),
                presentation: frame_view.clone(),
                offscreen: Some(offscreen),
                depth,
                instances,
            })
        } else {
            Some(Self {
                active: frame_view.clone(),
                presentation: frame_view.clone(),
                offscreen: None,
                depth,
                instances,
            })
        }
    }
}

fn begin_frame(renderer: &mut WgpuRenderer) -> bool {
    let Some(error) = renderer.faults.pending_error.lock().unwrap().take() else {
        renderer.faults.consecutive_failed_frames = 0;
        renderer.atlas.before_frame();
        return true;
    };

    renderer.faults.consecutive_failed_frames += 1;
    log::error!(
        "GPU error during frame (failure {} of 10): {error}",
        renderer.faults.consecutive_failed_frames
    );
    if renderer.faults.consecutive_failed_frames > 10 {
        panic!("too many consecutive GPU errors; last error: {error}");
    }
    if renderer.faults.consecutive_failed_frames > 5 {
        if let Some(resources) = renderer.resources.as_mut() {
            resources.invalidate_intermediate_textures();
        }
        renderer.atlas.clear();
        renderer.target.request_redraw();
        renderer.faults.consecutive_failed_frames = 0;
        return false;
    }

    renderer.atlas.before_frame();
    true
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct GlobalUniformState {
    globals: GlobalUniforms,
    font_rasterization: FontRasterizationUniforms,
}

fn write_shader_globals(renderer: &mut WgpuRenderer) {
    let font = renderer.rendering_params.font_rasterization;
    let font_rasterization = FontRasterizationUniforms {
        gamma_ratios: wgsl_rs::std::vec4f(
            font.gamma_ratios[0],
            font.gamma_ratios[1],
            font.gamma_ratios[2],
            font.gamma_ratios[3],
        ),
        grayscale_enhanced_contrast: font.grayscale_enhanced_contrast,
        subpixel_enhanced_contrast: font.subpixel_enhanced_contrast,
        uses_blue_green_red_subpixel_order: ShaderBool::from(
            renderer.subpixel_order == super::SubpixelOrder::BlueGreenRed,
        ),
        padding: 0,
    };
    let globals = GlobalUniforms {
        viewport_size: wgsl_rs::std::vec2f(
            renderer.target.width() as f32,
            renderer.target.height() as f32,
        ),
        premultiplied_alpha: ShaderBool::from(
            renderer.target.alpha_mode() == wgpu::CompositeAlphaMode::PreMultiplied,
        ),
        padding: 0,
    };
    let state = GlobalUniformState {
        globals,
        font_rasterization,
    };
    if renderer.uploaded_globals == Some(state) {
        return;
    }

    let resources = renderer.resources();
    let globals_size = std::mem::size_of::<GlobalUniforms>();
    let font_size = std::mem::size_of::<FontRasterizationUniforms>();
    let upload_size = renderer.globals.font_offset + font_size as u64;
    let mut upload = resources
        .queue
        .write_buffer_with(
            &resources.globals_buffer,
            0,
            std::num::NonZeroU64::new(upload_size).expect("global uniforms are non-empty"),
        )
        .expect("global uniform upload must fit its buffer");
    upload.slice(..).fill(0);
    upload
        .slice(..globals_size)
        .copy_from_slice(shader_interface::bytes_of(&globals));
    upload
        .slice(
            renderer.globals.font_offset as usize
                ..renderer.globals.font_offset as usize + font_size,
        )
        .copy_from_slice(shader_interface::bytes_of(&font_rasterization));
    drop(upload);
    renderer.uploaded_globals = Some(state);
}

#[derive(Clone, Copy, Default)]
pub(super) struct FrameRequirements {
    storage_bytes: u64,
    /// Instance batches this frame; one downlevel range-uniform slot per batch.
    instance_batches: u64,
    pub(super) uniforms: FrameUniformRequirements,
    planned: ScenePlanRequirements,
}

impl FrameRequirements {
    pub(super) fn for_scene(scene: &Scene, transport: InstanceTransport) -> Self {
        let planned = scene.render_plan().requirements();
        let mut storage_bytes = 0_u64;
        let mut instance_batches = 0_u64;
        // Path tiles index the frame's curves and paints by absolute element, so each array
        // starts on a multiple of its own stride and takes no batch slot.
        if planned.path_tile_count > 0 {
            for (stride, count) in [
                (std::mem::size_of::<PathCurve>(), scene.path_curves.len()),
                (std::mem::size_of::<PathPaint>(), scene.path_paints.len()),
            ] {
                let stride = stride as u64;
                storage_bytes = storage_bytes.next_multiple_of(stride);
                storage_bytes = storage_bytes.saturating_add(stride.saturating_mul(count as u64));
            }
        }
        let mut reserve = |element_size: usize, count: usize| {
            if count > 0 {
                let stride = element_size as u64;
                storage_bytes = storage_bytes.next_multiple_of(transport.batch_alignment(stride));
                storage_bytes = storage_bytes.saturating_add(stride.saturating_mul(count as u64));
                instance_batches += 1;
            }
        };

        for command in scene.render_commands() {
            let RenderCommand::Batch(batch) = command else {
                continue;
            };
            match batch {
                PrimitiveBatch::Shadows { range, .. } => {
                    reserve(std::mem::size_of::<Shadow>(), range.len())
                }
                PrimitiveBatch::Quads { range, .. } => {
                    reserve(std::mem::size_of::<Quad>(), range.len())
                }
                PrimitiveBatch::Paths { tiles, .. } => {
                    reserve(std::mem::size_of::<PathTile>(), tiles.len())
                }
                PrimitiveBatch::Underlines(range) => {
                    reserve(std::mem::size_of::<Underline>(), range.len())
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<MonochromeSprite>(), range.len())
                }
                PrimitiveBatch::SubpixelSprites { range, .. } => {
                    reserve(std::mem::size_of::<SubpixelSprite>(), range.len())
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    reserve(std::mem::size_of::<PolychromeSprite>(), range.len())
                }
                PrimitiveBatch::Surfaces(_)
                | PrimitiveBatch::BackdropFilters(_)
                | PrimitiveBatch::FilterBoundary(_) => {}
            }
        }
        let plan = scene.render_plan();
        for scene_pass in 0..plan.scene_pass_count() {
            reserve(
                std::mem::size_of::<Quad>(),
                plan.opaque_quads(scene_pass).len(),
            );
        }
        debug_assert_eq!(
            instance_batches as usize,
            planned.instance_batch_count + planned.opaque_quad_pass_count
        );

        Self {
            storage_bytes,
            instance_batches,
            uniforms: FrameUniformRequirements {
                filter_count: FILTER_UNIFORMS_PER_COMPOSITE
                    * (planned.backdrop_filter_count + planned.isolated_filter_count) as u64
                    + u64::from(planned.uses_offscreen_target)
                    + if planned.isolated_target_count > 0 {
                        (planned.backdrop_filter_count * (planned.isolated_target_count + 1)) as u64
                    } else {
                        0
                    },
                surface_count: planned.surface_count as u64,
            },
            planned: *planned,
        }
    }
}

struct FrameEncoder<'a> {
    renderer: &'a WgpuRenderer,
    scene: &'a Scene,
    encoder: wgpu::CommandEncoder,
    targets: TargetStack<wgpu::TextureView>,
    offscreen: Option<wgpu::TextureView>,
    presentation: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    depth: SceneDepth,
    instances: InstanceUpload,
}

/// Depth bookkeeping across a frame's scene passes.
///
/// Quad `i` of [`Scene::quads`] draws at `quad_depth(i)`. Every other batch draws at the depth
/// of the next quad, through the viewport's depth range, so it hides behind that quad and
/// every later one, and shows in front of earlier ones: the frame's paint order. Opaque fills
/// write depth first, front to back, so nothing behind them is shaded.
pub(super) struct SceneDepth {
    scale: f32,
    /// Index of the next quad in paint order.
    quad_cursor: usize,
    scene_pass: usize,
    viewport: Option<f32>,
    width: f32,
    height: f32,
    hides_opaque_rectangles: bool,
}

impl SceneDepth {
    fn new(renderer: &WgpuRenderer, scene: &Scene) -> Self {
        Self {
            scale: quad_depth_scale(scene.quads.len()),
            quad_cursor: 0,
            scene_pass: 0,
            viewport: None,
            width: renderer.target.width() as f32,
            height: renderer.target.height() as f32,
            hides_opaque_rectangles: scene.render_plan().requirements().opaque_quad_count > 0,
        }
    }

    /// Sets the viewport's depth range: `None` lets quads place themselves, `Some(depth)`
    /// pins every fragment of the next batches to `depth`.
    fn set_viewport(&mut self, pass: &mut wgpu::RenderPass<'_>, depth: Option<f32>) {
        if self.viewport != depth {
            let (minimum, maximum) = depth.map_or((0.0, 1.0), |depth| (depth, depth));
            pass.set_viewport(0.0, 0.0, self.width, self.height, minimum, maximum);
            self.viewport = depth;
        }
    }

    pub(super) fn quad_depth(&self, quad_index: usize) -> f32 {
        quad_depth(quad_index, self.scale)
    }

    /// Whether the quad was drawn whole by its pass's opaque fills.
    pub(super) fn hides(&self, quad: &Quad) -> bool {
        self.hides_opaque_rectangles && quad.is_opaque_rectangle()
    }
}

impl<'a> FrameEncoder<'a> {
    fn new(renderer: &'a WgpuRenderer, scene: &'a Scene, targets: PreparedTargets) -> Self {
        let encoder =
            renderer
                .resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpui_frame"),
                });
        Self {
            renderer,
            scene,
            encoder,
            targets: TargetStack::new(targets.active),
            offscreen: targets.offscreen,
            presentation: targets.presentation,
            depth_view: targets.depth,
            depth: SceneDepth::new(renderer, scene),
            instances: targets.instances,
        }
    }

    fn encode(
        mut self,
        readback: Option<ReadbackCopy<'_>>,
    ) -> Result<wgpu::CommandBuffer, DrawError> {
        let result = self.encode_commands();
        if result.is_ok() {
            if let Some(offscreen) = &self.offscreen {
                self.renderer
                    .blit_to_frame(&mut self.encoder, offscreen, &self.presentation);
            }
            if let Some(readback) = readback {
                self.encoder.copy_texture_to_buffer(
                    readback.texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: readback.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(readback.bytes_per_row),
                            rows_per_image: Some(readback.height),
                        },
                    },
                    wgpu::Extent3d {
                        width: readback.width,
                        height: readback.height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
        self.instances.finish(&mut self.encoder);
        self.renderer.resources().finish_frame_uploads();
        let command_buffer = self.encoder.finish();
        result.map(|()| command_buffer)
    }

    fn encode_commands(&mut self) -> DrawResult {
        let mut pass = begin_scene_render_pass(
            self.renderer,
            &mut self.encoder,
            "main_pass",
            self.targets.current(),
            wgpu::LoadOp::Clear(self.renderer.target.clear_color()),
            &self.depth_view,
        );
        draw_opaque_fills(
            self.renderer,
            self.scene,
            &mut self.depth,
            &mut self.instances,
            &mut pass,
        )?;

        for command in self.scene.render_commands() {
            match command {
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    drop(pass);
                    for filter in &self.scene.backdrop_filters[range.clone()] {
                        let snapshot = self.targets.inherits_backdrop().then(|| {
                            self.renderer.snapshot_backdrop(
                                &mut self.encoder,
                                self.targets.backdrop_layers(),
                            )
                        });
                        let source = snapshot.as_ref().unwrap_or_else(|| self.targets.current());
                        self.renderer.blur_and_composite(
                            &mut self.encoder,
                            source,
                            self.targets.current(),
                            FilterCompositeParameters {
                                bounds: filter.bounds,
                                content_mask: filter.content_mask.bounds,
                                corner_radii: filter.corner_radii,
                                corner_smoothing: filter.corner_smoothing,
                                blur_radius: filter.max_blur_radius(),
                                opacity: filter.opacity,
                                clip: FilterCompositeClip::RoundedBounds,
                            },
                        );
                    }
                    self.depth.begin_scene_pass();
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_backdrop_filter",
                        self.targets.current(),
                        wgpu::LoadOp::Load,
                        &self.depth_view,
                    );
                    draw_opaque_fills(
                        self.renderer,
                        self.scene,
                        &mut self.depth,
                        &mut self.instances,
                        &mut pass,
                    )?;
                }
                RenderCommand::Batch(PrimitiveBatch::FilterBoundary(_)) => {
                    unreachable!("filter boundaries must be compiled into render commands")
                }
                RenderCommand::Batch(batch) => encode_inline_batch(
                    self.renderer,
                    self.scene,
                    batch,
                    &mut self.depth,
                    &mut self.instances,
                    &mut pass,
                )?,
                RenderCommand::BeginFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(index),
                } => {
                    drop(pass);
                    let target =
                        self.renderer.resources().filter_group_views[index.as_usize()].clone();
                    self.targets.enter(
                        target,
                        self.scene.filter_boundaries[*boundary_index]
                            .filters
                            .is_empty(),
                    );
                    self.depth.begin_scene_pass();
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "filter_group",
                        self.targets.current(),
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        &self.depth_view,
                    );
                    draw_opaque_fills(
                        self.renderer,
                        self.scene,
                        &mut self.depth,
                        &mut self.instances,
                        &mut pass,
                    )?;
                }
                RenderCommand::EndFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(_),
                    ..
                } => {
                    drop(pass);
                    let (filtered, parent) = self.targets.exit();
                    let boundary = &self.scene.filter_boundaries[*boundary_index];
                    self.renderer.blur_and_composite(
                        &mut self.encoder,
                        &filtered,
                        parent,
                        FilterCompositeParameters {
                            bounds: boundary.bounds,
                            content_mask: boundary.content_mask.bounds,
                            corner_radii: boundary.corner_radii,
                            corner_smoothing: boundary.corner_smoothing,
                            blur_radius: boundary.max_blur_radius(),
                            opacity: boundary.opacity,
                            clip: FilterCompositeClip::ContentShape,
                        },
                    );
                    self.depth.begin_scene_pass();
                    pass = begin_scene_render_pass(
                        self.renderer,
                        &mut self.encoder,
                        "after_content_filter",
                        self.targets.current(),
                        wgpu::LoadOp::Load,
                        &self.depth_view,
                    );
                    draw_opaque_fills(
                        self.renderer,
                        self.scene,
                        &mut self.depth,
                        &mut self.instances,
                        &mut pass,
                    )?;
                }
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                }
                | RenderCommand::EndFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                } => {}
            }
        }
        drop(pass);
        self.targets.assert_balanced();
        debug_assert_eq!(
            self.depth.scene_pass + 1,
            self.scene.render_plan().scene_pass_count()
        );
        Ok(())
    }
}

impl SceneDepth {
    fn begin_scene_pass(&mut self) {
        self.scene_pass += 1;
        // A new pass starts with the default viewport.
        self.viewport = None;
    }
}

/// Draws the opaque fill regions of the current scene pass, front to back, writing depth.
fn draw_opaque_fills(
    renderer: &WgpuRenderer,
    scene: &Scene,
    depth: &mut SceneDepth,
    instances: &mut InstanceUpload,
    pass: &mut wgpu::RenderPass<'_>,
) -> DrawResult {
    let opaque_quads = scene.render_plan().opaque_quads(depth.scene_pass);
    if opaque_quads.is_empty() {
        return Ok(());
    }
    depth.set_viewport(pass, None);
    let fills = opaque_quads.iter().map(|&index| {
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
            depth: depth.quad_depth(index),
            ..*quad
        }
    });
    let Some(slice) = instances.write_iter(opaque_quads.len(), fills) else {
        return Err(DrawError::CapacityPlanningInvariant);
    };
    renderer.draw_opaque_fills(&slice, pass);
    Ok(())
}

fn begin_scene_render_pass<'a>(
    renderer: &'a WgpuRenderer,
    encoder: &'a mut wgpu::CommandEncoder,
    label: &'a str,
    target: &'a wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    depth: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    pass.set_bind_group(
        shader_interface::GLOBAL_BIND_GROUP,
        &renderer.resources().globals_bind_group,
        &[],
    );
    pass
}

#[derive(Debug)]
pub(super) enum DrawError {
    CapacityPlanningInvariant,
    ExternalSurface,
}

pub(super) type DrawResult = Result<(), DrawError>;

fn encode_inline_batch(
    renderer: &WgpuRenderer,
    scene: &Scene,
    batch: &PrimitiveBatch,
    depth: &mut SceneDepth,
    instances: &mut InstanceUpload,
    pass: &mut wgpu::RenderPass<'_>,
) -> DrawResult {
    if let PrimitiveBatch::Quads { range, .. } = batch {
        debug_assert_eq!(range.start, depth.quad_cursor, "quads draw in index order");
        depth.set_viewport(pass, None);
        depth.quad_cursor = range.end;
    } else {
        let batch_depth = depth.quad_depth(depth.quad_cursor);
        depth.set_viewport(pass, Some(batch_depth));
    }
    match batch {
        PrimitiveBatch::Quads { range, smoothed } => renderer.draw_quads(
            &scene.quads[range.clone()],
            range.start,
            *smoothed,
            depth,
            instances,
            pass,
        ),
        PrimitiveBatch::Shadows { range, smoothed } => {
            renderer.draw_shadows(&scene.shadows[range.clone()], *smoothed, instances, pass)
        }
        PrimitiveBatch::Underlines(range) => {
            renderer.draw_underlines(&scene.underlines[range.clone()], instances, pass)
        }
        PrimitiveBatch::MonochromeSprites { texture_id, range } => renderer
            .draw_monochrome_sprites(
                &scene.monochrome_sprites[range.clone()],
                *texture_id,
                instances,
                pass,
            ),
        PrimitiveBatch::SubpixelSprites { texture_id, range } => renderer.draw_subpixel_sprites(
            &scene.subpixel_sprites[range.clone()],
            *texture_id,
            instances,
            pass,
        ),
        PrimitiveBatch::PolychromeSprites {
            texture_id,
            range,
            smoothed,
        } => renderer.draw_polychrome_sprites(
            &scene.polychrome_sprites[range.clone()],
            *texture_id,
            *smoothed,
            instances,
            pass,
        ),
        PrimitiveBatch::Surfaces(range) => {
            renderer.draw_surfaces(&scene.surfaces[range.clone()], pass)
        }
        PrimitiveBatch::Paths { tiles, .. } => {
            renderer.draw_path_tiles(&scene.path_tiles[tiles.clone()], instances, pass)
        }
        PrimitiveBatch::BackdropFilters(_) | PrimitiveBatch::FilterBoundary(_) => {
            unreachable!("pass-interrupting batches are handled by FrameEncoder")
        }
    }
}
