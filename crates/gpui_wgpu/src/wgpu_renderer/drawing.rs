use gpui::{
    AtlasTextureId, MonochromeSprite, PathTile, PolychromeSprite, Quad, Shadow, SubpixelSprite,
    Underline,
};

use crate::WgpuTextureInfo;
use gpui_render::shaders::interface::BufferData;

use super::{
    WgpuRenderer,
    buffers::{InstanceSlice, InstanceUpload},
    frame, pipelines,
};

impl WgpuRenderer {
    /// Draws quads `first_index..` of the scene at their depths, leaving out the opaque
    /// rectangles the pass's opaque fills already drew.
    pub(super) fn draw_quads(
        &self,
        quads: &[Quad],
        first_index: usize,
        smoothed: bool,
        depth: &frame::SceneDepth,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let pipelines = &self.resources().pipelines;
        let pipeline = if smoothed {
            &pipelines.smoothed_quads
        } else {
            &pipelines.quads
        };
        let count = depth.shared.blended_quad_count(quads);
        if count == 0 {
            return Ok(());
        }
        let blended = depth.shared.blended_quads(quads, first_index);
        let Some(slice) = instances.write_iter(count, blended) else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        self.draw_bound_slice(
            pipeline,
            self.resources().instances.bind_group(),
            &slice,
            pass,
        );
        Ok(())
    }

    pub(super) fn draw_opaque_fills(
        &self,
        fills: &InstanceSlice<Quad>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let resources = self.resources();
        self.draw_bound_slice(
            &resources.pipelines.opaque_quads,
            resources.instances.bind_group(),
            fills,
            pass,
        );
    }

    pub(super) fn draw_shadows(
        &self,
        shadows: &[Shadow],
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let pipelines = &self.resources().pipelines;
        let pipeline = if smoothed {
            &pipelines.smoothed_shadows
        } else {
            &pipelines.shadows
        };
        self.draw_instances(shadows, pipeline, instances, pass)
    }

    pub(super) fn draw_underlines(
        &self,
        underlines: &[Underline],
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        self.draw_instances(
            underlines,
            &self.resources().pipelines.underlines,
            instances,
            pass,
        )
    }

    pub(super) fn draw_monochrome_sprites(
        &self,
        sprites: &[MonochromeSprite],
        texture_id: AtlasTextureId,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(texture) = self.atlas.get_texture_info(texture_id) else {
            log::warn!("skipping sprite batch: atlas texture {texture_id:?} was released");
            return Ok(());
        };
        self.draw_instances_with_texture(
            sprites,
            texture_id,
            &texture,
            &self.resources().pipelines.monochrome_sprites,
            instances,
            pass,
        )
    }

    pub(super) fn draw_subpixel_sprites(
        &self,
        sprites: &[SubpixelSprite],
        texture_id: AtlasTextureId,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(texture) = self.atlas.get_texture_info(texture_id) else {
            log::warn!("skipping sprite batch: atlas texture {texture_id:?} was released");
            return Ok(());
        };
        let resources = self.resources();
        let pipeline = resources
            .pipelines
            .subpixel_sprites
            .as_ref()
            .unwrap_or(&resources.pipelines.monochrome_sprites);
        self.draw_instances_with_texture(sprites, texture_id, &texture, pipeline, instances, pass)
    }

    pub(super) fn draw_polychrome_sprites(
        &self,
        sprites: &[PolychromeSprite],
        texture_id: AtlasTextureId,
        smoothed: bool,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(texture) = self.atlas.get_texture_info(texture_id) else {
            log::warn!("skipping sprite batch: atlas texture {texture_id:?} was released");
            return Ok(());
        };
        let pipelines = &self.resources().pipelines;
        let pipeline = if smoothed {
            &pipelines.smoothed_polychrome_sprites
        } else {
            &pipelines.polychrome_sprites
        };
        self.draw_instances_with_texture(sprites, texture_id, &texture, pipeline, instances, pass)
    }

    fn draw_instances<T: BufferData>(
        &self,
        values: &[T],
        pipeline: &pipelines::WgpuRenderPipeline,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        if values.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        self.draw_bound_instances(
            values,
            pipeline,
            resources.instances.bind_group(),
            instances,
            pass,
        )
    }

    fn draw_bound_instances<T: BufferData>(
        &self,
        values: &[T],
        pipeline: &pipelines::WgpuRenderPipeline,
        bind_group: &wgpu::BindGroup,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        let Some(slice) = instances.write(values) else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        self.draw_bound_slice(pipeline, bind_group, &slice, pass);
        Ok(())
    }

    fn draw_bound_slice<T>(
        &self,
        pipeline: &pipelines::WgpuRenderPipeline,
        bind_group: &wgpu::BindGroup,
        slice: &InstanceSlice<T>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        pass.set_pipeline(pipeline);
        slice.set_data_bind_group(pass, bind_group);
        pass.draw(0..pipeline.fixed_vertex_count(), slice.range());
    }

    fn draw_instances_with_texture<T: BufferData>(
        &self,
        values: &[T],
        texture_id: AtlasTextureId,
        texture: &WgpuTextureInfo,
        pipeline: &pipelines::WgpuRenderPipeline,
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        if values.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        let bind_group = resources.instances.textured_bind_group(
            &resources.device,
            &resources.bind_group_layouts,
            pipeline.data_layout(),
            texture_id,
            texture.identity,
            &texture.view,
            &resources.atlas_sampler,
        );
        self.draw_bound_instances(values, pipeline, &bind_group, instances, pass)
    }

    /// Draws path tiles, pointing them at the frame's curves and paints, which
    /// `PreparedTargets::prepare` wrote ahead of every batch.
    pub(super) fn draw_path_tiles(
        &self,
        tiles: &[PathTile],
        instances: &mut InstanceUpload,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> frame::DrawResult {
        if tiles.is_empty() {
            return Ok(());
        }
        let (curve_base, paint_base) = instances.path_bases();
        let Some(slice) = instances.write_iter(
            tiles.len(),
            tiles.iter().map(|tile| PathTile {
                paint: tile.paint + paint_base,
                curve_start: tile.curve_start + curve_base,
                ..*tile
            }),
        ) else {
            return Err(frame::DrawError::CapacityPlanningInvariant);
        };
        let resources = self.resources();
        self.draw_bound_slice(
            &resources.pipelines.path_tiles,
            resources.instances.path_tile_bind_group(),
            &slice,
            pass,
        );
        Ok(())
    }
}
