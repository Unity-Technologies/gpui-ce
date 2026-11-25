use crate::metal_atlas::MetalAtlas;
use anyhow::Result;
use block2::RcBlock;
use gpui::{
    AtlasTextureId, Bounds, Corners, DevicePixels, FilterRenderTarget, MAX_FILTER_GROUP_DEPTH,
    MonochromeSprite, PaintSurface, PathCurve, PathPaint, PathTile, PolychromeSprite,
    PrimitiveBatch, Quad, RenderCommand, ScaledPixels, Scene, Shadow, Size, SurfaceSource,
    Underline, size,
};
use gpui_render::{
    TargetStack,
    artifacts::{NATIVE_SHADERS, NativeShader},
    blur::{
        BlurAxis, BlurKernel, BlurUniforms, GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS, ScissorRectangle,
        downsampled_dimension,
    },
    shaders::{
        common::{FontRasterizationUniforms, GlobalUniforms, ShaderBool, SurfaceColorFormat},
        surface::SurfaceUniforms,
    },
};
#[cfg(any(
    test,
    feature = "bench-support",
    feature = "test-support",
    feature = "render-to-image"
))]
use image::RgbaImage;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::{CFRetained, CGSize};
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceDisplayP3};
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBuffer,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidthOfPlane,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLCopyAllDevices,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLDrawable, MTLFunction, MTLGPUFamily, MTLLibrary,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassColorAttachmentDescriptor, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLResourceOptions, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerState, MTLScissorRect, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage, MTLViewport,
};
#[cfg(any(
    test,
    feature = "bench-support",
    feature = "test-support",
    feature = "render-to-image"
))]
use objc2_metal::{MTLBlitCommandEncoder, MTLOrigin, MTLRegion, MTLSize};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;
use smallvec::SmallVec;
use wgsl_rs::std::{vec2f, vec4f};

use std::{cell::Cell, mem, ptr, ptr::NonNull, sync::Arc};

pub use crate::display_link::MetalDrawable;

type MtlDevice = ProtocolObject<dyn MTLDevice>;
type MtlBuffer = ProtocolObject<dyn MTLBuffer>;
type MtlTexture = ProtocolObject<dyn MTLTexture>;
type MtlLibrary = ProtocolObject<dyn MTLLibrary>;
type MtlFunction = ProtocolObject<dyn MTLFunction>;
type MtlPipeline = ProtocolObject<dyn MTLRenderPipelineState>;
type MtlCommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
type MtlEncoder = ProtocolObject<dyn MTLRenderCommandEncoder>;

// Buffer slots declared by the generated MSL. Group 0 globals land at 0/1, the group 1 data
// binding at 2 (path tiles add curves at 3 and paints at 4), Naga's runtime-array sizes buffer
// at 5.
const GLOBALS_SLOT: usize = 0;
const FONT_SLOT: usize = 1;
const DATA_SLOT: usize = 2;
const PATH_CURVES_SLOT: usize = 3;
const PATH_PAINTS_SLOT: usize = 4;
const SIZES_SLOT: usize = 5;
const PRIMARY_TEXTURE_SLOT: usize = 0;
const SECONDARY_TEXTURE_SLOT: usize = 1;
const SAMPLER_SLOT: usize = 0;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

/// Per-frame global uniforms bound by every pipeline.
struct SceneUniforms {
    globals: GlobalUniforms,
    font: FontRasterizationUniforms,
}

impl SceneUniforms {
    fn new(viewport_size: Size<DevicePixels>) -> Self {
        Self {
            globals: GlobalUniforms {
                viewport_size: vec2f(
                    i32::from(viewport_size.width) as f32,
                    i32::from(viewport_size.height) as f32,
                ),
                // Metal composites straight-alpha; paths premultiply in-shader, matching
                // the neutral (disabled) shader behavior.
                premultiplied_alpha: ShaderBool::Disabled,
                padding: 0,
            },
            // Metal text is gamma-corrected grayscale; font corrections stay neutral.
            font: FontRasterizationUniforms {
                gamma_ratios: vec4f(0.0, 0.0, 0.0, 0.0),
                grayscale_enhanced_contrast: 0.0,
                subpixel_enhanced_contrast: 0.0,
                uses_blue_green_red_subpixel_order: ShaderBool::Disabled,
                padding: 0,
            },
        }
    }
}

fn native_shader(label: &str) -> &'static NativeShader {
    NATIVE_SHADERS
        .iter()
        .find(|shader| shader.label == label)
        .unwrap_or_else(|| panic!("missing generated native shader {label}"))
}

fn bind_scene_uniforms(encoder: &MtlEncoder, uniforms: &SceneUniforms) {
    set_stage_bytes(encoder, GLOBALS_SLOT, &uniforms.globals);
    set_stage_bytes(encoder, FONT_SLOT, &uniforms.font);
}

/// Copies `value` into both stages' argument tables at `slot`.
fn set_stage_bytes<T>(encoder: &MtlEncoder, slot: usize, value: &T) {
    // SAFETY: Metal copies `size_of::<T>()` bytes from the live value before returning, and
    // every caller passes the struct the generated MSL declares at `slot`.
    unsafe {
        encoder.setVertexBytes_length_atIndex(
            NonNull::from(value).cast(),
            mem::size_of::<T>(),
            slot,
        );
        encoder.setFragmentBytes_length_atIndex(
            NonNull::from(value).cast(),
            mem::size_of::<T>(),
            slot,
        );
    }
}

fn set_vertex_texture(encoder: &MtlEncoder, slot: usize, texture: &MtlTexture) {
    // SAFETY: The generated MSL declares a 2D texture at `slot`, and Metal retains it.
    unsafe { encoder.setVertexTexture_atIndex(Some(texture), slot) };
}

fn set_fragment_texture(encoder: &MtlEncoder, slot: usize, texture: &MtlTexture) {
    // SAFETY: The generated MSL declares a 2D texture at `slot`, and Metal retains it.
    unsafe { encoder.setFragmentTexture_atIndex(Some(texture), slot) };
}

fn set_fragment_sampler(encoder: &MtlEncoder, sampler: &ProtocolObject<dyn MTLSamplerState>) {
    // SAFETY: The generated MSL declares its only sampler at `SAMPLER_SLOT`.
    unsafe { encoder.setFragmentSamplerState_atIndex(Some(sampler), SAMPLER_SLOT) };
}

fn draw(encoder: &MtlEncoder, primitive: MTLPrimitiveType, vertex_count: usize) {
    // SAFETY: Every pipeline's vertex shader derives positions from `vertex_index` and the
    // instance data bound before the draw; no vertex buffer is read past its bounds.
    unsafe { encoder.drawPrimitives_vertexStart_vertexCount(primitive, 0, vertex_count) };
}

fn draw_instanced(
    encoder: &MtlEncoder,
    primitive: MTLPrimitiveType,
    vertex_count: usize,
    instance_count: usize,
) {
    // SAFETY: The caller bound `instance_count` instances, with their byte length, at
    // `DATA_SLOT`; shaders index instance data only through that runtime-sized array.
    unsafe {
        encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
            primitive,
            0,
            vertex_count,
            instance_count,
        )
    };
}

fn clear_color(red: f64, green: f64, blue: f64, alpha: f64) -> MTLClearColor {
    MTLClearColor {
        red,
        green,
        blue,
        alpha,
    }
}

/// Binds an instance slice and the byte length used by Naga's runtime-array size ABI.
fn bind_instances<T>(encoder: &MtlEncoder, buffer: &MtlBuffer, offset: usize, instances: &[T]) {
    bind_instance_bytes(encoder, buffer, offset, mem::size_of_val(instances));
}

fn bind_instance_bytes(encoder: &MtlEncoder, buffer: &MtlBuffer, offset: usize, byte_len: usize) {
    let byte_len = u32::try_from(byte_len).expect("Metal instance binding exceeds 4 GiB");
    // SAFETY: `offset` is 256-byte aligned within the buffer, Metal retains the buffer, and
    // the shaders read at most `byte_len` bytes, the length bound at `SIZES_SLOT`.
    unsafe {
        encoder.setVertexBuffer_offset_atIndex(Some(buffer), offset, DATA_SLOT);
        encoder.setFragmentBuffer_offset_atIndex(Some(buffer), offset, DATA_SLOT);
    }
    set_stage_bytes(encoder, SIZES_SLOT, &byte_len);
}

/// Where a frame's path curves and paints live in the instance buffer.
#[derive(Clone, Copy)]
struct PathFrameData {
    curves_offset: usize,
    curves_len: usize,
    paints_offset: usize,
    paints_len: usize,
}

impl PathFrameData {
    fn write(
        curves: &[PathCurve],
        paints: &[PathPaint],
        instance_buffer: &InstanceBuffer,
        instance_offset: &mut usize,
    ) -> Option<Self> {
        let mut write = |items_len: usize, write_items: &dyn Fn(usize)| {
            align_offset(instance_offset);
            let offset = *instance_offset;
            let next_offset = offset + items_len;
            if next_offset > instance_buffer.size {
                return None;
            }
            write_items(offset);
            *instance_offset = next_offset;
            Some(offset)
        };
        let curves_len = mem::size_of_val(curves);
        let curves_offset = write(curves_len, &|offset| {
            write_instances(instance_buffer, offset, curves)
        })?;
        let paints_len = mem::size_of_val(paints);
        let paints_offset = write(paints_len, &|offset| {
            write_instances(instance_buffer, offset, paints)
        })?;
        Some(Self {
            curves_offset,
            curves_len,
            paints_offset,
            paints_len,
        })
    }

    /// Binds a batch's tiles at `DATA_SLOT` beside the frame's curves and paints, with the
    /// three runtime-array lengths the generated MSL reads at `SIZES_SLOT`.
    fn bind(
        &self,
        encoder: &MtlEncoder,
        buffer: &MtlBuffer,
        tiles_offset: usize,
        tiles_len: usize,
    ) {
        let length = |len: usize| u32::try_from(len).expect("Metal instance binding exceeds 4 GiB");
        let sizes = [
            length(tiles_len),
            length(self.curves_len),
            length(self.paints_len),
        ];
        // SAFETY: Each offset is 256-byte aligned within the buffer, Metal retains the buffer,
        // and the shader reads at most the lengths bound at `SIZES_SLOT`.
        unsafe {
            for (offset, slot) in [
                (tiles_offset, DATA_SLOT),
                (self.curves_offset, PATH_CURVES_SLOT),
                (self.paints_offset, PATH_PAINTS_SLOT),
            ] {
                encoder.setVertexBuffer_offset_atIndex(Some(buffer), offset, slot);
                encoder.setFragmentBuffer_offset_atIndex(Some(buffer), offset, slot);
            }
        }
        set_stage_bytes(encoder, SIZES_SLOT, &sizes);
    }
}

/// Writes `items` into the instance buffer at `offset`, which the caller bounds-checked.
fn write_instances<T>(instance_buffer: &InstanceBuffer, offset: usize, items: &[T]) {
    // SAFETY: `acquire` checked `contents` is non-null, and the caller checked
    // `offset + size_of_val(items) <= instance_buffer.size`.
    unsafe {
        let contents = instance_buffer
            .metal_buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(offset);
        ptr::copy_nonoverlapping(
            items.as_ptr().cast::<u8>(),
            contents,
            mem::size_of_val(items),
        );
    }
}

/// Creates the CAMetalLayer that backs a window's view. The renderer drawing
/// into it sets its device and pixel format.
pub fn new_window_layer(transparent: bool) -> Retained<CAMetalLayer> {
    let layer = CAMetalLayer::new();
    // Support direct-to-display rendering if the window is not transparent
    // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
    layer.setOpaque(!transparent);
    layer.setAllowsNextDrawableTimeout(false);
    layer.setNeedsDisplayOnBoundsChange(true);
    layer.setAutoresizingMask(
        CAAutoresizingMask::LayerWidthSizable | CAAutoresizingMask::LayerHeightSizable,
    );
    layer
}

/// Tags the layer's contents as Display P3, so colors are not clamped to sRGB.
fn set_display_p3_color_space(layer: &CAMetalLayer) {
    // SAFETY: The constant is extern, from Core Graphics.
    let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
    let Some(display_p3_color_space) = CGColorSpace::with_name(Some(display_p3_name)) else {
        log::error!("unable to create the Display P3 color space");
        return;
    };
    layer.setColorspace(Some(&display_p3_color_space));
}

pub fn new_renderer(
    context: self::Context,
    _bounds: gpui::Size<f32>,
    transparent: bool,
    use_display_p3: bool,
) -> Result<Renderer> {
    Ok(MetalRenderer::new(context, transparent, use_display_p3)?)
}

/// Why a [`MetalRenderer`] could not be created.
///
/// Building the renderer compiles the generated MSL with the device's runtime
/// compiler and creates a pipeline state per primitive. A driver can refuse
/// either step (for example "Compiler encountered an internal error" on some
/// older GPUs), and a window cannot draw without them. The error says what
/// failed, so an application can report it and exit instead of panicking.
/// `open_window` returns it inside an `anyhow::Error`; use `downcast_ref`.
#[derive(Clone, Debug)]
pub struct MetalRendererError {
    /// The `MTLDevice` name of the GPU, or `None` when no Metal device was found.
    pub device_name: Option<String>,
    /// What the renderer was doing, such as "compiling the quads shader library".
    pub stage: String,
    /// Metal's own error text.
    pub message: String,
}

impl MetalRendererError {
    fn new(
        device: Option<&MtlDevice>,
        stage: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            device_name: device.map(|device| device.name().to_string()),
            stage: stage.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for MetalRendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device = self.device_name.as_deref().unwrap_or("no Metal device");
        write!(
            f,
            "failed to create the Metal renderer on {device} while {}: {}",
            self.stage, self.message
        )
    }
}

impl std::error::Error for MetalRendererError {}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<Retained<MtlBuffer>>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: Retained<MtlBuffer>,
    size: usize,
}

// SAFETY: Metal buffers may be used from any thread (Apple's "Metal Thread Safety"); the
// pool hands each one to a single frame at a time, and a command buffer's completion
// handler returns it from Metal's own thread.
unsafe impl Send for InstanceBuffer {}
// SAFETY: As for `InstanceBuffer`: the pool only stores idle buffers behind its mutex.
unsafe impl Send for InstanceBufferPool {}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    /// Returns `None` when Metal cannot give a buffer of the needed size: it
    /// is over the device's limit, or the allocation failed. A nil buffer has
    /// a null `contents()`, and writing instances through it crashes.
    pub(crate) fn acquire(
        &mut self,
        device: &MtlDevice,
        unified_memory: bool,
        minimum_size: usize,
    ) -> Option<InstanceBuffer> {
        let max_buffer_length = device.maxBufferLength();
        if minimum_size > max_buffer_length {
            log::error!(
                "skipped a frame: it needs a {minimum_size} byte instance buffer, over the device limit of {max_buffer_length}"
            );
            return None;
        }
        if minimum_size > self.buffer_size {
            self.reset(
                minimum_size
                    .checked_next_power_of_two()
                    .unwrap_or(minimum_size)
                    .min(max_buffer_length),
            );
        }
        let buffer = self.buffers.pop().or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.newBufferWithLength_options(self.buffer_size, options)
        });
        let Some(buffer) = buffer else {
            log::error!(
                "skipped a frame: Metal returned no {} byte instance buffer",
                self.buffer_size
            );
            return None;
        };
        Some(InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        })
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

enum DisplayLinkMode {
    Detached,
    Attached { drawable: Option<MetalDrawable> },
}

pub struct MetalRenderer {
    device: Retained<MtlDevice>,
    layer: Option<Retained<CAMetalLayer>>,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    display_link_mode: DisplayLinkMode,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    path_tiles_pipeline_state: Retained<MtlPipeline>,
    shadows_pipeline_state: Retained<MtlPipeline>,
    smoothed_shadows_pipeline_state: Retained<MtlPipeline>,
    quads_pipeline_state: Retained<MtlPipeline>,
    smoothed_quads_pipeline_state: Retained<MtlPipeline>,
    underlines_pipeline_state: Retained<MtlPipeline>,
    monochrome_sprites_pipeline_state: Retained<MtlPipeline>,
    polychrome_sprites_pipeline_state: Retained<MtlPipeline>,
    smoothed_polychrome_sprites_pipeline_state: Retained<MtlPipeline>,
    surfaces_pipeline_state: Retained<MtlPipeline>,
    // Blur pipelines: downsample (no blend, also used for the final blit), separable gaussian
    // (no blend), and composite (alpha blend into a rounded rect), from shared shader sources.
    blur_downsample_pipeline_state: Retained<MtlPipeline>,
    blur_pipeline_state: Retained<MtlPipeline>,
    blur_composite_pipeline_state: Retained<MtlPipeline>,
    smoothed_blur_composite_pipeline_state: Retained<MtlPipeline>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    // Offscreen scene target (the scene is rendered here, then blitted to the drawable, so blur
    // passes can sample already-painted content), the half-res ping/pong blur targets, and a
    // full-res target for content-filter groups.
    scene_color_texture: Option<Retained<MtlTexture>>,
    backdrop_snapshot_texture: Option<Retained<MtlTexture>>,
    blur_ping_texture: Option<Retained<MtlTexture>>,
    blur_pong_texture: Option<Retained<MtlTexture>>,
    /// Full-resolution offscreen targets a content-filter (`filter`) group renders into before
    /// being filtered and composited back. Targets are pooled by active nesting depth.
    group_textures: Vec<Retained<MtlTexture>>,
    intermediate_texture_size: Option<Size<DevicePixels>>,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<Retained<MtlTexture>>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    ///
    /// Fails when no Metal device is available, or when the device cannot build
    /// the renderer's shader libraries and pipeline states.
    pub fn new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
        use_display_p3: bool,
    ) -> Result<Self, MetalRendererError> {
        let device = Self::create_device()?;

        let layer = new_window_layer(transparent);
        if use_display_p3 {
            set_display_p3_color_space(&layer);
        }
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        layer.setMaximumDrawableCount(3);
        // Allow texture reading for visual tests, benches and `render-to-image`
        // (captures without ScreenCaptureKit). `test-support` alone only pays the
        // presentation cost in debug builds, so it is safe to compile in.
        #[cfg(any(
            test,
            feature = "bench-support",
            feature = "render-to-image",
            all(feature = "test-support", debug_assertions)
        ))]
        layer.setFramebufferOnly(false);

        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Result<Self, MetalRendererError> {
        let device = Self::create_device()?;
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> Result<Retained<MtlDevice>, MetalRendererError> {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `MTLCreateSystemDefaultDevice()`.
        if let Some(d) = MTLCopyAllDevices()
            .into_iter()
            .min_by_key(|d| (d.isRemovable(), !d.isLowPower()))
        {
            Ok(d)
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            MTLCreateSystemDefaultDevice().ok_or_else(|| {
                MetalRendererError::new(
                    None,
                    "selecting a graphics device",
                    "unable to access a compatible graphics device",
                )
            })
        }
    }

    fn new_internal(
        device: Retained<MtlDevice>,
        layer: Option<Retained<CAMetalLayer>>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Result<Self, MetalRendererError> {
        Self::new_internal_with_sources(device, layer, opaque, instance_buffer_pool, |shader| {
            shader.msl
        })
    }

    /// `new_internal`, with the MSL for each shader chosen by `msl_for`, so a test can
    /// hand the device a source it refuses.
    fn new_internal_with_sources(
        device: Retained<MtlDevice>,
        layer: Option<Retained<CAMetalLayer>>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        msl_for: impl Fn(&'static NativeShader) -> &'static str,
    ) -> Result<Self, MetalRendererError> {
        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        let is_unified_memory = device.hasUnifiedMemory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supportsFamily(MTLGPUFamily::Apple1);

        // Compile the Naga-generated MSL with the device's runtime compiler, deduplicating
        // per source so each module compiles exactly once.
        let mut libraries: Vec<(&'static str, Retained<MtlLibrary>)> = Vec::new();
        let mut library_for =
            |shader: &'static NativeShader| -> Result<Retained<MtlLibrary>, MetalRendererError> {
                let source = msl_for(shader);
                if let Some((_, library)) = libraries
                    .iter()
                    .find(|(registered, _)| *registered == source)
                {
                    return Ok(library.clone());
                }
                let library = compile_library(&device, shader.label, source)?;
                libraries.push((source, library.clone()));
                Ok(library)
            };

        let mut pipeline = |label: &str| -> Result<
            (&'static NativeShader, Retained<MtlLibrary>),
            MetalRendererError,
        > {
            let shader = native_shader(label);
            Ok((shader, library_for(shader)?))
        };

        let (path_tiles_shader, path_tiles_library) = pipeline("path_tiles")?;
        let path_tiles_pipeline_state = build_pipeline_state(
            &device,
            &path_tiles_library,
            path_tiles_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (shadows_shader, shadows_library) = pipeline("shadows")?;
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &shadows_library,
            shadows_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (smoothed_shadows_shader, smoothed_shadows_library) = pipeline("smoothed_shadows")?;
        let smoothed_shadows_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_shadows_library,
            smoothed_shadows_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (quads_shader, quads_library) = pipeline("quads")?;
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &quads_library,
            quads_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (smoothed_quads_shader, smoothed_quads_library) = pipeline("smoothed_quads")?;
        let smoothed_quads_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_quads_library,
            smoothed_quads_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (underlines_shader, underlines_library) = pipeline("underlines")?;
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &underlines_library,
            underlines_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (monochrome_shader, monochrome_library) = pipeline("monochrome_sprites")?;
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &monochrome_library,
            monochrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (polychrome_shader, polychrome_library) = pipeline("polychrome_sprites")?;
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &polychrome_library,
            polychrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (smoothed_polychrome_shader, smoothed_polychrome_library) =
            pipeline("smoothed_polychrome_sprites")?;
        let smoothed_polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_polychrome_library,
            smoothed_polychrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (surfaces_shader, surfaces_library) = pipeline("surfaces")?;
        let surfaces_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &surfaces_library,
            surfaces_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (blur_downsample_shader, blur_downsample_library) = pipeline("blur_downsample")?;
        let blur_downsample_pipeline_state = build_blur_pipeline_state(
            &device,
            &blur_downsample_library,
            blur_downsample_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (blur_shader, blur_library) = pipeline("blur")?;
        let blur_pipeline_state = build_blur_pipeline_state(
            &device,
            &blur_library,
            blur_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        // Premultiplied blend (One / OneMinusSourceAlpha) — the composite outputs a premultiplied
        // blurred sample; straight-alpha blending would darken the faded edges.
        let (blur_composite_shader, blur_composite_library) = pipeline("blur_composite")?;
        let blur_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &blur_composite_library,
            blur_composite_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let (smoothed_blur_composite_shader, smoothed_blur_composite_library) =
            pipeline("smoothed_blur_composite")?;
        let smoothed_blur_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &smoothed_blur_composite_library,
            smoothed_blur_composite_shader,
            MTLPixelFormat::BGRA8Unorm,
        )?;

        let sampler_descriptor = MTLSamplerDescriptor::new();
        sampler_descriptor.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.setMagFilter(MTLSamplerMinMagFilter::Linear);
        let sampler = device
            .newSamplerStateWithDescriptor(&sampler_descriptor)
            .ok_or_else(|| {
                MetalRendererError::new(Some(&device), "creating the sampler", "Metal returned nil")
            })?;

        let command_queue = device.newCommandQueue().ok_or_else(|| {
            MetalRendererError::new(
                Some(&device),
                "creating the command queue",
                "Metal returned nil",
            )
        })?;
        // Metal textures are not `Send`; the atlas lives on the thread that owns the window.
        #[expect(clippy::arc_with_non_send_sync)]
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let core_video_texture_cache = create_core_video_texture_cache(&device)?;

        Ok(Self {
            device,
            layer,
            presents_with_transaction: false,
            display_link_mode: DisplayLinkMode::Detached,
            is_unified_memory,
            opaque,
            command_queue,
            path_tiles_pipeline_state,
            shadows_pipeline_state,
            smoothed_shadows_pipeline_state,
            quads_pipeline_state,
            smoothed_quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            smoothed_polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            blur_downsample_pipeline_state,
            blur_pipeline_state,
            blur_composite_pipeline_state,
            smoothed_blur_composite_pipeline_state,
            sampler,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            scene_color_texture: None,
            backdrop_snapshot_texture: None,
            blur_ping_texture: None,
            blur_pong_texture: None,
            group_textures: Vec::new(),
            intermediate_texture_size: None,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        })
    }

    pub fn layer(&self) -> Option<&CAMetalLayer> {
        self.layer.as_deref()
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| Retained::as_ptr(l).cast_mut())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    /// Whether a window can pace this renderer with a `CAMetalDisplayLink`.
    pub const SUPPORTS_DISPLAY_LINK: bool = true;

    /// Draw the next frame into `drawable` instead of asking the layer for one.
    /// Pass `None` once the frame is done, so a drawable the frame did not use
    /// goes back to the layer.
    ///
    /// Returns the drawable this replaces: after a frame, `Some` means the
    /// frame neither drew nor presented. A setter rather than a frame-scoped
    /// guard, because the window releases its lock on the renderer while
    /// GPUI draws.
    pub fn set_frame_drawable(&mut self, drawable: Option<MetalDrawable>) -> Option<MetalDrawable> {
        match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable: current } => mem::replace(current, drawable),
            DisplayLinkMode::Detached => {
                // Only a live link hands out drawables, and a link is detached
                // only after it has been invalidated.
                debug_assert!(
                    drawable.is_none(),
                    "received a display link drawable while detached"
                );
                None
            }
        }
    }

    /// Record whether a `CAMetalDisplayLink` drives the layer. While one
    /// does, frames are drawn only into the drawables it hands out.
    pub fn set_display_link_attached(&mut self, attached: bool) {
        self.display_link_mode = if attached {
            DisplayLinkMode::Attached { drawable: None }
        } else {
            DisplayLinkMode::Detached
        };
    }

    pub fn is_display_link_attached(&self) -> bool {
        matches!(self.display_link_mode, DisplayLinkMode::Attached { .. })
    }

    pub fn presents_with_transaction(&self) -> bool {
        self.presents_with_transaction
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.setPresentsWithTransaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.setDrawableSize(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        self.update_intermediate_texture_size(size);
    }

    fn update_intermediate_texture_size(&mut self, size: Size<DevicePixels>) {
        if self.intermediate_texture_size == Some(size) {
            return;
        }
        self.scene_color_texture = None;
        self.backdrop_snapshot_texture = None;
        self.blur_ping_texture = None;
        self.blur_pong_texture = None;
        self.group_textures.clear();
        self.intermediate_texture_size = (size.width.0 > 0 && size.height.0 > 0).then_some(size);
    }

    fn prepare_intermediate_textures(&mut self, scene: &Scene, size: Size<DevicePixels>) {
        self.update_intermediate_texture_size(size);
        let Some(size) = self.intermediate_texture_size else {
            return;
        };
        let requirements = scene.render_plan().requirements();
        let full_w = size.width.0 as usize;
        let full_h = size.height.0 as usize;

        let device = &self.device;
        let make_color_texture = |width: usize, height: usize| {
            let texture = device.newTextureWithDescriptor(&new_render_target_descriptor(
                width.max(1),
                height.max(1),
            ));
            if texture.is_none() {
                log::error!("Metal returned no {width}x{height} render target");
            }
            texture
        };

        if requirements.uses_offscreen_target && self.scene_color_texture.is_none() {
            self.scene_color_texture = make_color_texture(full_w, full_h);
        }

        if requirements.uses_blur_target {
            let blur_width = downsampled_dimension(full_w as u32) as usize;
            let blur_height = downsampled_dimension(full_h as u32) as usize;
            if self.blur_ping_texture.is_none() {
                self.blur_ping_texture = make_color_texture(blur_width, blur_height);
            }
            if self.blur_pong_texture.is_none() {
                self.blur_pong_texture = make_color_texture(blur_width, blur_height);
            }
        }

        if requirements.backdrop_filter_count > 0
            && requirements.isolated_target_count > 0
            && self.backdrop_snapshot_texture.is_none()
        {
            self.backdrop_snapshot_texture = make_color_texture(full_w, full_h);
        }

        while self.group_textures.len() < requirements.isolated_target_count {
            let Some(texture) = make_color_texture(full_w, full_h) else {
                break;
            };
            self.group_textures.push(texture);
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.setOpaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    /// HUB-8754 spike: draw `scene` into a second, transparent layer (the overlay surface)
    /// using this renderer's pipelines and atlas. Clears to alpha 0.
    pub fn draw_overlay(&mut self, scene: &Scene, overlay: &CAMetalLayer) {
        // `new_window_layer` leaves device and pixel format to the renderer: without a device
        // the layer never vends a drawable, so the overlay silently never draws.
        let has_our_device = overlay
            .device()
            .is_some_and(|current| ptr::eq(&*current, &*self.device));
        if !has_our_device {
            overlay.setDevice(Some(&self.device));
            overlay.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            overlay.setMaximumDrawableCount(3);
        }
        let saved = (self.opaque, self.layer.take());
        self.opaque = std::env::var_os("OVERLAY_SPIKE_DEBUG").is_some();
        self.layer = Some(overlay.retain());
        self.draw(scene);
        self.opaque = saved.0;
        self.layer = saved.1;
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let drawable = match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable } => {
                let Some(drawable) = drawable.take() else {
                    log::error!("skipped a frame drawn outside its display link's update");
                    return;
                };
                drawable
            }
            DisplayLinkMode::Detached => {
                let Some(drawable) = layer.nextDrawable() else {
                    log::error!(
                        "failed to retrieve next drawable, drawable size: {:?}",
                        layer.drawableSize()
                    );
                    return;
                };
                drawable
            }
        };
        // A display link's drawable was sized when the link handed it out,
        // which can be before a resize the layer has since taken.
        let texture = drawable.texture();
        let viewport_size: Size<DevicePixels> = size(
            (texture.width() as i32).into(),
            (texture.height() as i32).into(),
        );
        let Some(mut instance_buffer) = self.acquire_instance_buffer(scene) else {
            return;
        };
        let command_buffer =
            match self.draw_primitives(scene, &mut instance_buffer, &drawable, viewport_size) {
                Ok(command_buffer) => command_buffer,
                Err(error) => {
                    log::error!("failed to render pre-sized scene: {error}");
                    return;
                }
            };
        self.release_instance_buffer_when_complete(&command_buffer, instance_buffer);

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.waitUntilScheduled();
            drawable.present();
        } else {
            command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
            command_buffer.commit();
        }
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(
        test,
        feature = "bench-support",
        feature = "test-support",
        feature = "render-to-image"
    ))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawableSize();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        // Never the layer's own drawable: while a CAMetalDisplayLink drives
        // the layer, `nextDrawable` throws.
        self.render_scene_to_image(scene, viewport_size)
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(
        test,
        feature = "bench-support",
        feature = "test-support",
        feature = "render-to-image"
    ))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            // Create an offscreen texture as render target
            let texture_descriptor =
                new_render_target_descriptor(size.width.0 as usize, size.height.0 as usize);
            texture_descriptor.setStorageMode(MTLStorageMode::Managed);
            let target_texture = self
                .device
                .newTextureWithDescriptor(&texture_descriptor)
                .ok_or_else(|| anyhow::anyhow!("Metal could not allocate the target texture"))?;

            let mut instance_buffer = self
                .acquire_instance_buffer(scene)
                .ok_or_else(|| anyhow::anyhow!("Metal could not allocate the instance buffer"))?;
            let command_buffer = self.draw_primitives_to_texture(
                scene,
                &mut instance_buffer,
                &target_texture,
                size,
            )?;

            if !self.is_unified_memory {
                let blit = command_buffer
                    .blitCommandEncoder()
                    .ok_or_else(|| anyhow::anyhow!("Metal returned no blit encoder"))?;
                blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
                blit.endEncoding();
            }
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            self.instance_buffer_pool.lock().release(instance_buffer);

            let width = size.width.0 as u32;
            let height = size.height.0 as u32;
            let bytes_per_row = width as usize * 4;
            let buffer_size = height as usize * bytes_per_row;
            let mut pixels = vec![0u8; buffer_size];
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: width as usize,
                    height: height as usize,
                    depth: 1,
                },
            };
            // SAFETY: `pixels` holds `height * bytes_per_row` bytes, exactly the region.
            unsafe {
                target_texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    NonNull::new(pixels.as_mut_ptr().cast()).expect("Vec pointer is non-null"),
                    bytes_per_row,
                    region,
                    0,
                );
            }
            for chunk in pixels.chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }
            RgbaImage::from_raw(width, height, pixels)
                .ok_or_else(|| anyhow::anyhow!("Failed to create RgbaImage from pixel data"))
        })
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as usize
                    || texture.height() != size.height.0 as usize
            });
            if needs_new_target {
                let texture_descriptor =
                    new_render_target_descriptor(size.width.0 as usize, size.height.0 as usize);
                self.headless_render_target =
                    self.device.newTextureWithDescriptor(&texture_descriptor);
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .ok_or_else(|| anyhow::anyhow!("Metal could not allocate the render target"))?;

            let mut instance_buffer = self
                .acquire_instance_buffer(scene)
                .ok_or_else(|| anyhow::anyhow!("Metal could not allocate the instance buffer"))?;
            let command_buffer = self.draw_primitives_to_texture(
                scene,
                &mut instance_buffer,
                &target_texture,
                size,
            )?;
            self.release_instance_buffer_when_complete(&command_buffer, instance_buffer);
            command_buffer.commit();
            Ok(())
        })
    }

    fn acquire_instance_buffer(&self, scene: &Scene) -> Option<InstanceBuffer> {
        self.instance_buffer_pool.lock().acquire(
            &self.device,
            self.is_unified_memory,
            required_instance_buffer_size(scene),
        )
    }

    fn release_instance_buffer_when_complete(
        &self,
        command_buffer: &MtlCommandBuffer,
        instance_buffer: InstanceBuffer,
    ) {
        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(instance_buffer));
        let block = RcBlock::new(move |_: NonNull<MtlCommandBuffer>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
        });
        // SAFETY: Metal copies the block, and the handler touches only what it owns.
        unsafe { command_buffer.addCompletedHandler(RcBlock::as_ptr(&block)) };
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &ProtocolObject<dyn CAMetalDrawable>,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<MtlCommandBuffer>> {
        self.draw_primitives_to_texture(scene, instance_buffer, &drawable.texture(), viewport_size)
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        texture: &MtlTexture,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<MtlCommandBuffer>> {
        self.prepare_intermediate_textures(scene, viewport_size);
        let command_buffer_owned = self
            .command_queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("Metal returned no command buffer"))?;
        let command_buffer: &MtlCommandBuffer = &command_buffer_owned;
        let alpha = if self.opaque { 1. } else { 0. };
        let mut instance_offset = 0;
        let scene_uniforms = SceneUniforms::new(viewport_size);
        // Every path batch's tiles index the frame's curve and paint arrays, so they are
        // written once up front.
        let path_frame = if scene.path_tiles.is_empty() {
            None
        } else {
            PathFrameData::write(
                &scene.path_curves,
                &scene.path_paints,
                instance_buffer,
                &mut instance_offset,
            )
        };
        if !scene.path_tiles.is_empty() && path_frame.is_none() {
            anyhow::bail!(
                "scene too large: {} path curves, {} path paints",
                scene.path_curves.len(),
                scene.path_paints.len()
            );
        }

        // Filters and opacity groups sample offscreen textures. Owned clones keep those
        // targets independent of mutable renderer calls such as surface painting.
        let use_offscreen = scene.requires_offscreen_rendering();
        let scene_color_owned = self.scene_color_texture.clone();
        let blur_ping_owned = self.blur_ping_texture.clone();
        let blur_pong_owned = self.blur_pong_texture.clone();
        let backdrop_snapshot_owned = self.backdrop_snapshot_texture.clone();
        let group_owned = self
            .group_textures
            .iter()
            .cloned()
            .collect::<SmallVec<[Retained<MtlTexture>; MAX_FILTER_GROUP_DEPTH]>>();
        let scene_color: &MtlTexture = if use_offscreen {
            scene_color_owned.as_deref().unwrap_or(texture)
        } else {
            texture
        };

        let mut targets = TargetStack::new(scene_color);

        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            targets.current(),
            viewport_size,
            |color_attachment| {
                color_attachment.setLoadAction(MTLLoadAction::Clear);
                color_attachment.setClearColor(clear_color(0., 0., 0., alpha));
            },
        );

        for command in scene.render_commands() {
            let ok = match command {
                RenderCommand::Batch(PrimitiveBatch::Shadows { range, smoothed }) => self
                    .draw_shadows(
                        &scene.shadows[range.clone()],
                        *smoothed,
                        instance_buffer,
                        &mut instance_offset,
                        &scene_uniforms,
                        &command_encoder,
                    ),
                RenderCommand::Batch(PrimitiveBatch::Quads { range, smoothed }) => self.draw_quads(
                    &scene.quads[range.clone()],
                    *smoothed,
                    instance_buffer,
                    &mut instance_offset,
                    &scene_uniforms,
                    &command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::Paths { tiles, .. }) => match path_frame {
                    Some(frame) => self.draw_path_tiles(
                        &scene.path_tiles[tiles.clone()],
                        frame,
                        instance_buffer,
                        &mut instance_offset,
                        &scene_uniforms,
                        &command_encoder,
                    ),
                    None => false,
                },
                RenderCommand::Batch(PrimitiveBatch::Underlines(range)) => self.draw_underlines(
                    &scene.underlines[range.clone()],
                    instance_buffer,
                    &mut instance_offset,
                    &scene_uniforms,
                    &command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::MonochromeSprites { texture_id, range }) => {
                    self.draw_monochrome_sprites(
                        *texture_id,
                        &scene.monochrome_sprites[range.clone()],
                        instance_buffer,
                        &mut instance_offset,
                        &scene_uniforms,
                        &command_encoder,
                    )
                }
                RenderCommand::Batch(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range,
                    smoothed,
                }) => self.draw_polychrome_sprites(
                    *texture_id,
                    &scene.polychrome_sprites[range.clone()],
                    *smoothed,
                    instance_buffer,
                    &mut instance_offset,
                    &scene_uniforms,
                    &command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::Surfaces(range)) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    &scene_uniforms,
                    &command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    command_encoder.endEncoding();
                    if let (Some(ping), Some(pong)) =
                        (blur_ping_owned.as_deref(), blur_pong_owned.as_deref())
                    {
                        for filter in &scene.backdrop_filters[range.clone()] {
                            let source = if targets.inherits_backdrop() {
                                let snapshot = backdrop_snapshot_owned
                                    .as_deref()
                                    .expect("backdrop snapshot was prepared");
                                self.metal_snapshot_backdrop(
                                    command_buffer,
                                    &scene_uniforms,
                                    targets.backdrop_layers().copied(),
                                    snapshot,
                                    viewport_size,
                                );

                                snapshot
                            } else {
                                *targets.current()
                            };
                            self.metal_blur_and_composite(
                                command_buffer,
                                &scene_uniforms,
                                source,
                                targets.current(),
                                Some(ping),
                                Some(pong),
                                viewport_size,
                                filter.bounds,
                                filter.content_mask.bounds,
                                filter.corner_radii,
                                filter.corner_smoothing,
                                filter.max_blur_radius(),
                                filter.opacity,
                                true,
                            );
                        }
                    }
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        targets.current(),
                        viewport_size,
                        |color_attachment| {
                            color_attachment.setLoadAction(MTLLoadAction::Load);
                        },
                    );
                    true
                }
                RenderCommand::BeginFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(target_index),
                } => {
                    command_encoder.endEncoding();
                    targets.enter(
                        &group_owned[target_index.as_usize()],
                        scene.filter_boundaries[*boundary_index].filters.is_empty(),
                    );
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        targets.current(),
                        viewport_size,
                        |color_attachment| {
                            color_attachment.setLoadAction(MTLLoadAction::Clear);
                            color_attachment.setClearColor(clear_color(0., 0., 0., 0.));
                        },
                    );
                    true
                }
                RenderCommand::EndFilter {
                    boundary_index,
                    target: FilterRenderTarget::Isolated(_),
                    ..
                } => {
                    let boundary = &scene.filter_boundaries[*boundary_index];
                    let (filtered, parent) = targets.exit();
                    command_encoder.endEncoding();
                    self.metal_blur_and_composite(
                        command_buffer,
                        &scene_uniforms,
                        filtered,
                        parent,
                        blur_ping_owned.as_deref(),
                        blur_pong_owned.as_deref(),
                        viewport_size,
                        boundary.bounds,
                        boundary.content_mask.bounds,
                        boundary.corner_radii,
                        boundary.corner_smoothing,
                        boundary.max_blur_radius(),
                        boundary.opacity,
                        false,
                    );
                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        parent,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.setLoadAction(MTLLoadAction::Load);
                        },
                    );
                    true
                }
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                }
                | RenderCommand::EndFilter {
                    target: FilterRenderTarget::Inline,
                    ..
                } => true,
                RenderCommand::Batch(PrimitiveBatch::SubpixelSprites { .. }) => unreachable!(),
                RenderCommand::Batch(PrimitiveBatch::FilterBoundary(_)) => {
                    unreachable!("filter boundaries are resolved by the render plan")
                }
            };
            if !ok {
                command_encoder.endEncoding();
                anyhow::bail!(
                    "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                );
            }
        }

        command_encoder.endEncoding();
        targets.assert_balanced();

        // Present the offscreen scene by copying it into the drawable/target texture.
        if use_offscreen && scene_color_owned.is_some() {
            self.run_metal_blur_pass(
                command_buffer,
                &self.blur_downsample_pipeline_state,
                texture,
                scene_color,
                viewport_size,
                &scene_uniforms,
                BlurUniforms::copy([
                    i32::from(viewport_size.width) as f32,
                    i32::from(viewport_size.height) as f32,
                ]),
                ScissorRectangle {
                    x: 0,
                    y: 0,
                    width: i32::from(viewport_size.width).max(0) as u32,
                    height: i32::from(viewport_size.height).max(0) as u32,
                },
                MTLPrimitiveType::Triangle,
                3,
                false,
            );
        }

        if !self.is_unified_memory {
            // Sync the instance buffer to the GPU
            instance_buffer.metal_buffer.didModifyRange(NSRange {
                location: 0,
                length: instance_offset,
            });
        }

        Ok(command_buffer_owned)
    }

    /// Run a single blur pass: draw a full-screen (or composite) quad sampling `source` into
    /// `target`. `params` is supplied to both shader stages; `load` keeps existing target
    /// contents (used by the composite), otherwise the target is cleared.
    #[allow(clippy::too_many_arguments)]
    fn run_metal_blur_pass(
        &self,
        command_buffer: &MtlCommandBuffer,
        pipeline: &MtlPipeline,
        target: &MtlTexture,
        source: &MtlTexture,
        target_viewport: Size<DevicePixels>,
        scene_uniforms: &SceneUniforms,
        params: BlurUniforms,
        scissor: ScissorRectangle,
        primitive: MTLPrimitiveType,
        vertex_count: usize,
        load: bool,
    ) {
        let encoder = new_command_encoder_for_texture(
            command_buffer,
            target,
            target_viewport,
            |color_attachment| {
                if load {
                    color_attachment.setLoadAction(MTLLoadAction::Load);
                } else {
                    color_attachment.setLoadAction(MTLLoadAction::Clear);
                    color_attachment.setClearColor(clear_color(0., 0., 0., 0.));
                }
            },
        );
        encoder.setScissorRect(MTLScissorRect {
            x: scissor.x as usize,
            y: scissor.y as usize,
            width: scissor.width as usize,
            height: scissor.height as usize,
        });
        encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(&encoder, scene_uniforms);
        set_stage_bytes(&encoder, DATA_SLOT, &params);
        set_fragment_texture(&encoder, PRIMARY_TEXTURE_SLOT, source);
        set_fragment_sampler(&encoder, &self.sampler);
        draw(&encoder, primitive, vertex_count);
        encoder.endEncoding();
    }

    /// Blur `source` (full-resolution) using the half-res ping/pong textures and composite the
    /// result into `target`, clipped to `bounds`/`corner_radii`/`content_mask` and modulated by
    /// `opacity`. Shared by the backdrop and content-filter paths, mirroring the wgpu
    /// backend's pass structure via the shared `gpui_render::blur` contracts.
    #[allow(clippy::too_many_arguments)]
    fn metal_blur_and_composite(
        &self,
        command_buffer: &MtlCommandBuffer,
        scene_uniforms: &SceneUniforms,
        source: &MtlTexture,
        target: &MtlTexture,
        ping: Option<&MtlTexture>,
        pong: Option<&MtlTexture>,
        viewport_size: Size<DevicePixels>,
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: Corners<ScaledPixels>,
        corner_smoothing: f32,
        blur_radius: f32,
        opacity: f32,
        // Backdrop clips to the rounded rect; content (`filter`) bleeds past its bounds.
        clip_rounded: bool,
    ) {
        if clip_rounded && blur_radius <= 0.0 {
            return;
        }

        let full_width = i32::from(viewport_size.width).max(0) as u32;
        let full_height = i32::from(viewport_size.height).max(0) as u32;
        let blur_size = [
            downsampled_dimension(full_width) as f32,
            downsampled_dimension(full_height) as f32,
        ];
        let blur_viewport_size = Size {
            width: DevicePixels(blur_size[0] as i32),
            height: DevicePixels(blur_size[1] as i32),
        };
        let dilation = GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS * blur_radius;
        let scissor =
            ScissorRectangle::for_blurred_bounds(bounds, dilation, full_width, full_height);
        if scissor.is_empty() {
            return;
        }
        let clip = if clip_rounded {
            gpui_render::blur::FilterCompositeClip::RoundedBounds
        } else {
            gpui_render::blur::FilterCompositeClip::ContentShape
        };

        let full_size = [full_width as f32, full_height as f32];
        let (composite_source, source_size) =
            if let Some(kernel) = BlurKernel::for_radius(blur_radius) {
                let (Some(ping), Some(pong)) = (ping, pong) else {
                    return;
                };
                // Downsample source -> ping, then separable gaussian ping -> pong -> ping.
                self.run_metal_blur_pass(
                    command_buffer,
                    &self.blur_downsample_pipeline_state,
                    ping,
                    source,
                    blur_viewport_size,
                    scene_uniforms,
                    BlurUniforms::downsample([full_width as f32, full_height as f32], blur_size),
                    scissor,
                    MTLPrimitiveType::Triangle,
                    3,
                    false,
                );

                for (axis, source, target) in [
                    (BlurAxis::Horizontal, ping, pong),
                    (BlurAxis::Vertical, pong, ping),
                ] {
                    self.run_metal_blur_pass(
                        command_buffer,
                        &self.blur_pipeline_state,
                        target,
                        source,
                        blur_viewport_size,
                        scene_uniforms,
                        BlurUniforms::gaussian(axis, blur_size, kernel),
                        scissor,
                        MTLPrimitiveType::Triangle,
                        3,
                        false,
                    );
                }

                (ping, blur_size)
            } else {
                (source, full_size)
            };

        // Composite the blurred result into the target (preserving its contents).
        let composite_bounds = if clip_rounded {
            bounds
        } else {
            bounds.dilate(ScaledPixels(dilation))
        };
        let composite_uniforms = BlurUniforms::composite(
            composite_bounds,
            content_mask,
            corner_radii,
            corner_smoothing,
            opacity,
            clip,
            source_size,
            full_size,
        );
        let encoder = new_command_encoder_for_texture(
            command_buffer,
            target,
            viewport_size,
            |color_attachment| {
                color_attachment.setLoadAction(MTLLoadAction::Load);
            },
        );
        let pipeline = if composite_uniforms.corner_smoothing > 0.0 {
            &self.smoothed_blur_composite_pipeline_state
        } else {
            &self.blur_composite_pipeline_state
        };
        encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(&encoder, scene_uniforms);
        set_stage_bytes(&encoder, DATA_SLOT, &composite_uniforms);
        set_fragment_texture(&encoder, PRIMARY_TEXTURE_SLOT, composite_source);
        set_fragment_sampler(&encoder, &self.sampler);
        draw(&encoder, MTLPrimitiveType::TriangleStrip, 4);
        encoder.endEncoding();
    }

    fn metal_snapshot_backdrop<'a>(
        &self,
        command_buffer: &MtlCommandBuffer,
        scene_uniforms: &SceneUniforms,
        sources: impl Iterator<Item = &'a MtlTexture>,
        snapshot: &MtlTexture,
        viewport_size: Size<DevicePixels>,
    ) {
        let encoder = new_command_encoder_for_texture(
            command_buffer,
            snapshot,
            viewport_size,
            |attachment| {
                attachment.setLoadAction(MTLLoadAction::Clear);
                attachment.setClearColor(clear_color(0.0, 0.0, 0.0, 0.0));
            },
        );
        encoder.endEncoding();
        let bounds = Bounds::new(
            gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(
                ScaledPixels(viewport_size.width.0 as f32),
                ScaledPixels(viewport_size.height.0 as f32),
            ),
        );

        for source in sources {
            self.metal_blur_and_composite(
                command_buffer,
                scene_uniforms,
                source,
                snapshot,
                None,
                None,
                viewport_size,
                bounds,
                bounds,
                Corners::default(),
                0.0,
                0.0,
                1.0,
                false,
            );
        }
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        smoothed: bool,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &MtlEncoder,
    ) -> bool {
        if shadows.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let shadow_bytes_len = mem::size_of_val(shadows);
        let next_offset = *instance_offset + shadow_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let pipeline = if smoothed {
            &self.smoothed_shadows_pipeline_state
        } else {
            &self.shadows_pipeline_state
        };
        command_encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        write_instances(instance_buffer, *instance_offset, shadows);
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            shadows,
        );

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            4,
            shadows.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        smoothed: bool,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &MtlEncoder,
    ) -> bool {
        if quads.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let quad_bytes_len = mem::size_of_val(quads);
        let next_offset = *instance_offset + quad_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let pipeline = if smoothed {
            &self.smoothed_quads_pipeline_state
        } else {
            &self.quads_pipeline_state
        };
        command_encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        write_instances(instance_buffer, *instance_offset, quads);
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            quads,
        );

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            4,
            quads.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_path_tiles(
        &self,
        tiles: &[PathTile],
        frame: PathFrameData,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &MtlEncoder,
    ) -> bool {
        if tiles.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let tile_bytes_len = mem::size_of_val(tiles);
        let next_offset = *instance_offset + tile_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.setRenderPipelineState(&self.path_tiles_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);
        write_instances(instance_buffer, *instance_offset, tiles);
        frame.bind(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            tile_bytes_len,
        );
        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            4,
            tiles.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_underlines(
        &self,
        underlines: &[Underline],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &MtlEncoder,
    ) -> bool {
        if underlines.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let underline_bytes_len = mem::size_of_val(underlines);
        let next_offset = *instance_offset + underline_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.setRenderPipelineState(&self.underlines_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        write_instances(instance_buffer, *instance_offset, underlines);
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            underlines,
        );

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            4,
            underlines.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[MonochromeSprite],
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &MtlEncoder,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            log::warn!("skipping sprite batch: atlas texture {texture_id:?} was released");
            return true;
        };
        command_encoder.setRenderPipelineState(&self.monochrome_sprites_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        write_instances(instance_buffer, *instance_offset, sprites);
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            sprites,
        );
        // Generated native shaders derive tile coordinates from the atlas size in the
        // vertex stage; bind the same texture to both stages, as WGPU and DirectX do.
        set_vertex_texture(command_encoder, PRIMARY_TEXTURE_SLOT, &texture);
        set_fragment_texture(command_encoder, PRIMARY_TEXTURE_SLOT, &texture);
        set_fragment_sampler(command_encoder, &self.sampler);

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            4,
            sprites.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: &[PolychromeSprite],
        smoothed: bool,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &MtlEncoder,
    ) -> bool {
        if sprites.is_empty() {
            return true;
        }
        align_offset(instance_offset);

        let sprite_bytes_len = mem::size_of_val(sprites);
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            log::warn!("skipping sprite batch: atlas texture {texture_id:?} was released");
            return true;
        };
        let pipeline = if smoothed {
            &self.smoothed_polychrome_sprites_pipeline_state
        } else {
            &self.polychrome_sprites_pipeline_state
        };
        command_encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        write_instances(instance_buffer, *instance_offset, sprites);
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            sprites,
        );
        set_vertex_texture(command_encoder, PRIMARY_TEXTURE_SLOT, &texture);
        set_fragment_texture(command_encoder, PRIMARY_TEXTURE_SLOT, &texture);
        set_fragment_sampler(command_encoder, &self.sampler);

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            4,
            sprites.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        scene_uniforms: &SceneUniforms,
        command_encoder: &MtlEncoder,
    ) -> bool {
        command_encoder.setRenderPipelineState(&self.surfaces_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);
        set_fragment_sampler(command_encoder, &self.sampler);

        for surface in surfaces {
            let image_buffer = match &surface.source {
                SurfaceSource::Surface(image_buffer) => image_buffer,
                SurfaceSource::Unsupported(size) => {
                    log::error!("Metal cannot draw unsupported surface source with size {size:?}");
                    continue;
                }
                // WGPU textures exist when gpui's `custom-gpu` feature is on; only
                // the WGPU renderer (gpui_macos's `wgpu` feature) can draw them.
                #[allow(unreachable_patterns)]
                _ => {
                    log::error!("Metal cannot draw WGPU texture surfaces");
                    continue;
                }
            };

            let pixel_format = CVPixelBufferGetPixelFormatType(image_buffer);
            if pixel_format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                log::error!("skipped a surface with unsupported pixel format {pixel_format:#x}");
                continue;
            }
            let Some((_y_texture, y_metal_texture)) = core_video_plane_texture(
                &self.core_video_texture_cache,
                image_buffer,
                0,
                MTLPixelFormat::R8Unorm,
            ) else {
                continue;
            };
            let Some((_cb_cr_texture, cb_cr_metal_texture)) = core_video_plane_texture(
                &self.core_video_texture_cache,
                image_buffer,
                1,
                MTLPixelFormat::RG8Unorm,
            ) else {
                continue;
            };

            let surface_uniforms = SurfaceUniforms {
                bounds: surface.bounds.into(),
                content_mask: surface.content_mask.into(),
                color_format: SurfaceColorFormat::Yuv,
                opacity: 1.0,
                padding0: 0,
                padding1: 0,
                padding2: 0,
                padding3: 0,
                padding4: 0,
                padding5: 0,
            };
            set_stage_bytes(command_encoder, DATA_SLOT, &surface_uniforms);
            set_fragment_texture(command_encoder, PRIMARY_TEXTURE_SLOT, &y_metal_texture);
            set_fragment_texture(
                command_encoder,
                SECONDARY_TEXTURE_SLOT,
                &cb_cr_metal_texture,
            );

            draw(command_encoder, MTLPrimitiveType::TriangleStrip, 4);
        }
        true
    }
}

fn new_command_encoder_for_texture(
    command_buffer: &MtlCommandBuffer,
    texture: &MtlTexture,
    viewport_size: Size<DevicePixels>,
    configure_color_attachment: impl Fn(&MTLRenderPassColorAttachmentDescriptor),
) -> Retained<MtlEncoder> {
    let render_pass_descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index 0 is always a valid color attachment slot.
    let color_attachment = unsafe {
        render_pass_descriptor
            .colorAttachments()
            .objectAtIndexedSubscript(0)
    };
    color_attachment.setTexture(Some(texture));
    color_attachment.setStoreAction(MTLStoreAction::Store);
    configure_color_attachment(&color_attachment);

    let command_encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
        .expect("Metal returned no render command encoder for a valid descriptor");
    command_encoder.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

/// A private, render-target BGRA texture descriptor of the given size.
fn new_render_target_descriptor(width: usize, height: usize) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: Callers pass non-zero sizes taken from a drawable or viewport.
    unsafe {
        descriptor.setWidth(width);
        descriptor.setHeight(height);
    }
    descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    descriptor
}

fn create_core_video_texture_cache(
    device: &MtlDevice,
) -> Result<CFRetained<CVMetalTextureCache>, MetalRendererError> {
    let mut cache = ptr::null_mut();
    // SAFETY: `cache` is a valid out pointer, and the device outlives the call.
    let status =
        unsafe { CVMetalTextureCache::create(None, None, device, None, NonNull::from(&mut cache)) };
    match NonNull::new(cache) {
        // SAFETY: A successful create returns a +1 reference.
        Some(cache) if status == kCVReturnSuccess => Ok(unsafe { CFRetained::from_raw(cache) }),
        _ => Err(MetalRendererError::new(
            Some(device),
            "creating the CoreVideo texture cache",
            format!("CVReturn {status}"),
        )),
    }
}

/// Wraps one plane of a bi-planar pixel buffer as a Metal texture.
fn core_video_plane_texture(
    cache: &CVMetalTextureCache,
    image_buffer: &CVPixelBuffer,
    plane: usize,
    pixel_format: MTLPixelFormat,
) -> Option<(CFRetained<CVMetalTexture>, Retained<MtlTexture>)> {
    let mut texture = ptr::null_mut();
    // SAFETY: `texture` is a valid out pointer and `plane` is below the plane count the
    // caller checked through the pixel format.
    let status = unsafe {
        CVMetalTextureCache::create_texture_from_image(
            None,
            cache,
            image_buffer,
            None,
            pixel_format,
            CVPixelBufferGetWidthOfPlane(image_buffer, plane),
            CVPixelBufferGetHeightOfPlane(image_buffer, plane),
            plane,
            NonNull::from(&mut texture),
        )
    };
    if status != kCVReturnSuccess {
        log::error!("skipped a surface: CoreVideo returned {status} for plane {plane}");
        return None;
    }
    // SAFETY: A successful create returns a +1 reference.
    let texture = unsafe { CFRetained::from_raw(NonNull::new(texture)?) };
    let metal_texture = CVMetalTextureGetTexture(&texture)?;
    // The CoreVideo texture keeps the Metal texture's backing alive, so return both.
    Some((texture, metal_texture))
}

/// Compiles one generated MSL module with the device's runtime compiler.
fn compile_library(
    device: &MtlDevice,
    label: &str,
    source: &str,
) -> Result<Retained<MtlLibrary>, MetalRendererError> {
    device
        .newLibraryWithSource_options_error(
            &NSString::from_str(source),
            Some(&MTLCompileOptions::new()),
        )
        .map_err(|error| {
            MetalRendererError::new(
                Some(device),
                format!("compiling the {label} shader library"),
                error.localizedDescription().to_string(),
            )
        })
}

fn locate_function(
    device: &MtlDevice,
    library: &MtlLibrary,
    entry: &str,
) -> Result<Retained<MtlFunction>, MetalRendererError> {
    library
        .newFunctionWithName(&NSString::from_str(entry))
        .ok_or_else(|| {
            MetalRendererError::new(
                Some(device),
                format!("locating {entry}"),
                format!("the library has no function named {entry}"),
            )
        })
}

fn new_pipeline_state(
    device: &MtlDevice,
    shader: &NativeShader,
    descriptor: &MTLRenderPipelineDescriptor,
) -> Result<Retained<MtlPipeline>, MetalRendererError> {
    device
        .newRenderPipelineStateWithDescriptor_error(descriptor)
        .map_err(|error| {
            MetalRendererError::new(
                Some(device),
                format!("creating the {} pipeline state", shader.label),
                error.localizedDescription().to_string(),
            )
        })
}

/// How a pipeline's single color attachment blends.
#[derive(Clone, Copy)]
enum Blend {
    /// Straight alpha: `SourceAlpha / OneMinusSourceAlpha`.
    Straight,
    /// Premultiplied alpha: `One / OneMinusSourceAlpha`.
    Premultiplied,
    /// Overwrite the target.
    None,
}

fn new_pipeline_descriptor(
    device: &MtlDevice,
    library: &MtlLibrary,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
    blend: Blend,
) -> Result<Retained<MTLRenderPipelineDescriptor>, MetalRendererError> {
    let vertex_fn = locate_function(device, library, shader.vertex_entry)?;
    let fragment_fn = locate_function(device, library, shader.fragment_entry)?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(shader.label)));
    descriptor.setVertexFunction(Some(&vertex_fn));
    descriptor.setFragmentFunction(Some(&fragment_fn));
    // SAFETY: Index 0 is always a valid color attachment slot.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    let source_rgb = match blend {
        Blend::None => {
            color_attachment.setBlendingEnabled(false);
            return Ok(descriptor);
        }
        Blend::Straight => MTLBlendFactor::SourceAlpha,
        Blend::Premultiplied => MTLBlendFactor::One,
    };
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(source_rgb);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    Ok(descriptor)
}

fn build_pipeline_state(
    device: &MtlDevice,
    library: &MtlLibrary,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<MtlPipeline>, MetalRendererError> {
    let descriptor =
        new_pipeline_descriptor(device, library, shader, pixel_format, Blend::Straight)?;
    new_pipeline_state(device, shader, &descriptor)
}

fn build_path_sprite_pipeline_state(
    device: &MtlDevice,
    library: &MtlLibrary,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<MtlPipeline>, MetalRendererError> {
    let descriptor =
        new_pipeline_descriptor(device, library, shader, pixel_format, Blend::Premultiplied)?;
    new_pipeline_state(device, shader, &descriptor)
}

// Blur downsample/gaussian passes overwrite their target (no blending). The composite pass
// uses the normal alpha-blending pipeline (`build_path_sprite_pipeline_state`) instead.
fn build_blur_pipeline_state(
    device: &MtlDevice,
    library: &MtlLibrary,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<MtlPipeline>, MetalRendererError> {
    let descriptor = new_pipeline_descriptor(device, library, shader, pixel_format, Blend::None)?;
    new_pipeline_state(device, shader, &descriptor)
}

fn required_instance_buffer_size(scene: &Scene) -> usize {
    let mut required = 0;
    let mut reserve = |element_size: usize, count: usize| {
        if count > 0 {
            align_offset(&mut required);
            required = required.saturating_add(element_size.saturating_mul(count));
        }
    };
    // `PathFrameData::write` places the frame's curves and paints ahead of every batch.
    if !scene.path_tiles.is_empty() {
        reserve(mem::size_of::<PathCurve>(), scene.path_curves.len());
        reserve(mem::size_of::<PathPaint>(), scene.path_paints.len());
    }

    for command in scene.render_commands() {
        let RenderCommand::Batch(batch) = command else {
            continue;
        };
        match batch {
            PrimitiveBatch::Shadows { range, .. } => reserve(mem::size_of::<Shadow>(), range.len()),
            PrimitiveBatch::Quads { range, .. } => reserve(mem::size_of::<Quad>(), range.len()),
            PrimitiveBatch::Paths { tiles, .. } => reserve(mem::size_of::<PathTile>(), tiles.len()),
            PrimitiveBatch::Underlines(range) => reserve(mem::size_of::<Underline>(), range.len()),
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                reserve(mem::size_of::<MonochromeSprite>(), range.len())
            }
            PrimitiveBatch::PolychromeSprites { range, .. } => {
                reserve(mem::size_of::<PolychromeSprite>(), range.len())
            }
            PrimitiveBatch::SubpixelSprites { .. }
            | PrimitiveBatch::Surfaces(_)
            | PrimitiveBatch::BackdropFilters(_)
            | PrimitiveBatch::FilterBoundary(_) => {}
        }
    }
    required
}

// Align to multiples of 256 make Metal happy.
fn align_offset(offset: &mut usize) {
    *offset = (*offset).div_ceil(256) * 256;
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Result<Self, MetalRendererError> {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool)?;
        Ok(Self { renderer })
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        AtlasKey, BackdropFilter, BorderStyle, ContentMask, Edges, ImageId, PlatformAtlas,
        PlatformHeadlessRenderer, RenderImageParams, RenderSvgParams, ScaledFilter,
        TransformationMatrix, checkerboard, hsla, linear_color_stop, linear_gradient,
        pattern_slash, px, solid_background, white,
    };
    use std::borrow::Cow;

    #[test]
    fn display_link_mode_consumes_and_clears_drawables() {
        let mut scene = Scene::default();
        scene.finish();
        let mut renderer = MetalRenderer::new(Default::default(), false, false).unwrap();
        renderer.update_drawable_size(size(DevicePixels(64), DevicePixels(64)));
        let drawable = renderer
            .layer()
            .and_then(|layer| layer.nextDrawable())
            .expect("window renderer must supply a drawable");

        renderer.set_display_link_attached(true);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        renderer.draw(&scene);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        // A frame that drew consumed its drawable; one that didn't hands it back.
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        assert!(renderer.set_frame_drawable(None).is_some());

        // Drawing while attached without a drawable skips the frame.
        renderer.draw(&scene);

        renderer.set_frame_drawable(Some(drawable));
        renderer.set_display_link_attached(false);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Detached
        ));
        // Detached again, the renderer asks the layer for its own drawable.
        renderer.draw(&scene);
    }

    #[test]
    fn an_instance_buffer_over_the_device_limit_skips_the_frame() {
        let device = MetalRenderer::create_device().unwrap();
        let mut pool = InstanceBufferPool::default();
        let max_buffer_length = device.maxBufferLength();

        assert!(pool.acquire(&device, true, max_buffer_length + 1).is_none());

        // The refusal leaves the pool usable, and every buffer it hands out
        // has memory to write instances into.
        let buffer = pool
            .acquire(&device, true, 1024)
            .expect("a small instance buffer");
        assert_eq!(buffer.metal_buffer.length(), buffer.size);
        assert!(buffer.size >= 1024);
    }

    #[test]
    fn a_shader_library_the_device_refuses_is_an_error() {
        let device = MetalRenderer::create_device().unwrap();
        let device_name = device.name().to_string();
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));

        let result = MetalRenderer::new_internal_with_sources(device, None, true, pool, |shader| {
            if shader.label == "quads" {
                "this is not Metal Shading Language"
            } else {
                shader.msl
            }
        });

        let Err(error) = result else {
            panic!("a shader library the compiler rejects still built a renderer");
        };
        assert_eq!(error.device_name.as_deref(), Some(device_name.as_str()));
        assert_eq!(error.stage, "compiling the quads shader library");
        assert!(
            !error.message.is_empty(),
            "the compiler's error text is kept"
        );
        assert!(error.to_string().contains(&device_name));
    }

    #[test]
    fn a_missing_shader_entry_point_is_an_error() {
        let device = MetalRenderer::create_device().unwrap();
        let shader = native_shader("quads");
        let library = compile_library(&device, shader.label, shader.msl).unwrap();

        let Err(error) = locate_function(&device, &library, "no_such_entry_point") else {
            panic!("a missing entry point was found");
        };
        assert_eq!(error.stage, "locating no_such_entry_point");
        assert_eq!(
            error.device_name.as_deref(),
            Some(device.name().to_string().as_str())
        );
    }

    /// A 4x4 pixel buffer of `format`, IOSurface-backed when `io_surface` is set (what
    /// `CVMetalTextureCache` needs), with every plane filled with `fill`.
    fn pixel_buffer(format: u32, io_surface: bool, fill: &[u8]) -> CFRetained<CVPixelBuffer> {
        use objc2_core_foundation::{CFDictionary, CFString};
        use objc2_core_video::{
            CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
            CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeightOfPlane,
            CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
            kCVPixelBufferIOSurfacePropertiesKey,
        };

        let empty = CFDictionary::<CFString, CFString>::from_slices(&[], &[]);
        // SAFETY: The key is an extern constant from CoreVideo.
        let key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        let attributes =
            CFDictionary::<CFString, CFDictionary>::from_slices(&[key], &[empty.as_opaque()]);
        let mut buffer = ptr::null_mut();
        // SAFETY: `buffer` is a valid out pointer and the attributes outlive the call.
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                4,
                4,
                format,
                io_surface.then(|| attributes.as_opaque()),
                NonNull::from(&mut buffer),
            )
        };
        assert_eq!(status, kCVReturnSuccess);
        // SAFETY: A successful create returns a +1 reference.
        let buffer = unsafe { CFRetained::from_raw(NonNull::new(buffer).unwrap()) };
        // SAFETY: The buffer is locked while its planes are written within their extent.
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&buffer, CVPixelBufferLockFlags(0)),
                kCVReturnSuccess
            );
            for (plane, value) in fill.iter().enumerate() {
                let base = CVPixelBufferGetBaseAddressOfPlane(&buffer, plane).cast::<u8>();
                let len = CVPixelBufferGetBytesPerRowOfPlane(&buffer, plane)
                    * CVPixelBufferGetHeightOfPlane(&buffer, plane);
                ptr::write_bytes(base, *value, len);
            }
            CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags(0));
        }
        buffer
    }

    fn surface_scene(buffer: CFRetained<CVPixelBuffer>) -> Scene {
        let bounds = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: size(ScaledPixels(8.0), ScaledPixels(8.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            source: SurfaceSource::from(buffer),
        });
        scene.finish();
        scene
    }

    #[test]
    fn a_core_video_surface_draws_its_luma_and_chroma_planes() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        // Full-range Y = 255 with neutral chroma is white.
        let buffer = pixel_buffer(
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            true,
            &[255, 128],
        );
        let image = renderer
            .render_scene_to_image(
                &surface_scene(buffer),
                size(DevicePixels(8), DevicePixels(8)),
            )
            .unwrap();
        for channel in &pixel(&image, 4, 4)[..3] {
            assert!(*channel >= 250, "surface pixel {:?}", pixel(&image, 4, 4));
        }
    }

    #[test]
    fn a_surface_metal_cannot_wrap_is_skipped_not_a_panic() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        // A packed BGRA buffer is not the bi-planar YUV the surface shader samples, and a
        // buffer without an IOSurface cannot back a Metal texture.
        let bgra = pixel_buffer(objc2_core_video::kCVPixelFormatType_32BGRA, true, &[255]);
        let no_io_surface = pixel_buffer(
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            false,
            &[255, 128],
        );
        for buffer in [bgra, no_io_surface] {
            let image = renderer
                .render_scene_to_image(
                    &surface_scene(buffer),
                    size(DevicePixels(8), DevicePixels(8)),
                )
                .unwrap();
            assert_eq!(pixel(&image, 4, 4), [0, 0, 0, 255]);
        }
    }

    #[test]
    fn intermediate_textures_follow_scene_requirements() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool).unwrap();
        let target_size = size(DevicePixels(5), DevicePixels(5));
        let mut empty_scene = Scene::default();
        empty_scene.finish();

        renderer.prepare_intermediate_textures(&empty_scene, target_size);
        assert!(renderer.scene_color_texture.is_none());
        assert!(renderer.blur_ping_texture.is_none());
        assert!(renderer.group_textures.is_empty());

        let bounds = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: size(ScaledPixels(5.0), ScaledPixels(5.0)),
        };
        let mut filtered_scene = Scene::default();
        filtered_scene.insert_primitive(BackdropFilter {
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(1.0))],
            opacity: 1.0,
            ..Default::default()
        });
        filtered_scene.finish();

        renderer.prepare_intermediate_textures(&filtered_scene, target_size);
        assert!(renderer.scene_color_texture.is_some());
        assert!(renderer.blur_ping_texture.is_some());
        assert!(renderer.blur_pong_texture.is_some());
        assert!(renderer.group_textures.is_empty());
    }

    #[test]
    fn generated_sprite_shader_preserves_monochrome_coverage() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool).unwrap();
        let tile_size = Size {
            width: DevicePixels(8),
            height: DevicePixels(8),
        };
        let key = AtlasKey::Svg(RenderSvgParams {
            path: "generated-sprite-coverage-test".into(),
            size: tile_size,
        });
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(key, &mut || Ok(Some((tile_size, Cow::Owned(vec![64; 64])))))
            .unwrap()
            .unwrap();
        let bounds = Bounds {
            origin: gpui::point(ScaledPixels(8.0), ScaledPixels(8.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let mut scene = Scene::default();
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            padding: 0,
            bounds,
            content_mask: ContentMask {
                bounds,
                ..Default::default()
            },
            color: white().into(),
            tile,
            transformation: TransformationMatrix::unit(),
        });
        scene.finish();

        let image = renderer
            .render_scene_to_image(
                &scene,
                Size {
                    width: DevicePixels(32),
                    height: DevicePixels(32),
                },
            )
            .unwrap();
        let center = image.get_pixel(16, 16).0;
        assert!(
            center[0] >= 63 && center[0] <= 65 && center[1] == center[0] && center[2] == center[0],
            "monochrome coverage was altered before blending: {center:?}"
        );
    }

    /// Pixel spot-check helpers for the generated-shader contract tests below.
    ///
    /// Headless Metal runs on whichever Apple GPU the host provides, and the
    /// shared shaders exercise `sin`/`pow`/`exp`/MSAA paths whose last-ulp
    /// rounding varies across GPU generations and Metal compiler versions.
    /// These tests therefore pin exact bytes only where the math is bit-exact
    /// (solid colors, cleared background) and use small tolerances plus
    /// structural checks (coverage ramps, dash gaps, blur falloff) everywhere
    /// else. Deterministic byte-exact parity for quad backgrounds is covered
    /// by `quad_backgrounds_match_legacy_metal_pixels` in `gpui_wgpu`, which
    /// runs on a software renderer.
    fn pixel(image: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
        image.get_pixel(x, y).0
    }

    fn assert_pixel_close(
        image: &RgbaImage,
        x: u32,
        y: u32,
        expected: [u8; 4],
        tolerance: u8,
        what: &str,
    ) {
        let actual = pixel(image, x, y);
        for (channel, (observed, reference)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                observed.abs_diff(*reference) <= tolerance,
                "{what} pixel ({x}, {y}) channel {channel}: got {actual:?}, expected {expected:?} ± {tolerance}",
            );
        }
    }

    /// Renders `scene` headlessly and checks the output size. The pixel
    /// contracts below only hold where Metal actually executes, so non-macOS
    /// builds stop after the dimension check.
    fn render_for_contracts(
        renderer: &mut MetalHeadlessRenderer,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Option<RgbaImage> {
        let image = renderer.render_scene_to_image(scene, size).unwrap();
        assert_eq!(
            image.dimensions(),
            (size.width.0 as u32, size.height.0 as u32)
        );
        if cfg!(target_os = "macos") {
            Some(image)
        } else {
            None
        }
    }

    #[test]
    fn generated_quad_shaders_render_background_contracts() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        let bounds = |x, y| Bounds {
            origin: gpui::point(ScaledPixels(x), ScaledPixels(y)),
            size: Size {
                width: ScaledPixels(2.0),
                height: ScaledPixels(2.0),
            },
        };
        let mut quads = Scene::default();
        for (bounds, background) in [
            (bounds(0.0, 0.0), solid_background(hsla(0.0, 1.0, 0.5, 1.0))),
            (
                bounds(2.0, 0.0),
                checkerboard(hsla(0.6, 0.7, 0.5, 1.0), 1.0),
            ),
            (
                bounds(0.0, 2.0),
                pattern_slash(hsla(0.3, 0.8, 0.5, 1.0), 1.0, 1.0),
            ),
            (
                bounds(2.0, 2.0),
                linear_gradient(
                    45.0,
                    linear_color_stop(hsla(0.8, 0.9, 0.4, 1.0), 0.0),
                    linear_color_stop(hsla(0.1, 0.8, 0.6, 1.0), 1.0),
                ),
            ),
        ] {
            quads.insert_primitive(Quad {
                bounds,
                content_mask: ContentMask {
                    bounds,
                    ..Default::default()
                },
                background,
                ..Default::default()
            });
        }
        quads.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &quads,
            Size {
                width: DevicePixels(4),
                height: DevicePixels(4),
            },
        ) else {
            return;
        };

        // Solid red is exactly representable, so it must round-trip untouched.
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert_pixel_close(&image, x, y, [255, 0, 0, 255], 0, "solid background");
        }
        // The checkerboard alternates cleared black with the pattern color.
        // The pattern color is pure ALU math (no transcendentals), so it is tight.
        assert_pixel_close(&image, 2, 0, [0, 0, 0, 255], 1, "checkerboard off-square");
        assert_pixel_close(
            &image,
            3,
            0,
            [38, 110, 217, 255],
            2,
            "checkerboard on-square",
        );
        assert_pixel_close(
            &image,
            2,
            1,
            [38, 110, 217, 255],
            2,
            "checkerboard on-square",
        );
        assert_pixel_close(&image, 3, 1, [0, 0, 0, 255], 1, "checkerboard off-square");
        // Slash coverage runs through trig (`sin`) and derivatives, so assert
        // hue structure instead of exact bytes: green-dominant, lit, varying.
        let slash = [
            pixel(&image, 0, 2),
            pixel(&image, 1, 2),
            pixel(&image, 0, 3),
            pixel(&image, 1, 3),
        ];
        for (i, texel) in slash.iter().enumerate() {
            assert!(
                texel[1] > texel[0] && texel[0] > texel[2] && texel[1] > 20,
                "slash texel {i} keeps its green hue: {texel:?}",
            );
        }
        let mut slash_distinct = 0;
        for (i, texel) in slash.iter().enumerate() {
            if slash[..i].iter().all(|other| other != texel) {
                slash_distinct += 1;
            }
        }
        assert!(
            slash_distinct >= 2,
            "slash pattern varies across the quadrant: {slash:?}",
        );
        // The gradient interpolates in Oklab with per-pixel dither, so its
        // texels must be shaded (never flat red/black) and almost all distinct.
        let gradient = [
            pixel(&image, 2, 2),
            pixel(&image, 3, 2),
            pixel(&image, 2, 3),
            pixel(&image, 3, 3),
        ];
        for (i, texel) in gradient.iter().enumerate() {
            assert_ne!(*texel, [255, 0, 0, 255], "gradient texel {i} is shaded");
            assert_ne!(*texel, [0, 0, 0, 255], "gradient texel {i} is shaded");
        }
        let mut gradient_distinct = 0;
        for (i, texel) in gradient.iter().enumerate() {
            if gradient[..i].iter().all(|other| other != texel) {
                gradient_distinct += 1;
            }
        }
        assert!(
            gradient_distinct >= 3,
            "gradient interpolates across the quadrant: {gradient:?}",
        );
    }

    #[test]
    fn generated_underline_shaders_render_line_contracts() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        let underline_bounds = Bounds {
            origin: gpui::point(ScaledPixels(1.5), ScaledPixels(3.5)),
            size: Size {
                width: ScaledPixels(5.0),
                height: ScaledPixels(1.5),
            },
        };
        let underline_frame = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut underline = Scene::default();
        underline.insert_primitive(Quad {
            bounds: underline_frame,
            content_mask: ContentMask {
                bounds: underline_frame,
                ..Default::default()
            },
            background: solid_background(hsla(0.0, 0.0, 0.1, 1.0)),
            ..Default::default()
        });
        underline.insert_primitive(Underline {
            order: 0,
            padding: 0,
            bounds: underline_bounds,
            content_mask: ContentMask {
                bounds: underline_bounds,
                ..Default::default()
            },
            color: hsla(0.6, 0.8, 0.6, 0.65).into(),
            thickness: ScaledPixels(1.0),
            wavy: gpui::ShaderBool::Disabled,
        });
        underline.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &underline,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(&image, 0, 0, [26, 26, 26, 255], 1, "underline backdrop");
        assert_pixel_close(&image, 7, 7, [26, 26, 26, 255], 1, "underline backdrop");
        assert_pixel_close(&image, 3, 3, [56, 98, 162, 255], 3, "straight underline");
        assert_pixel_close(&image, 3, 4, [56, 98, 162, 255], 3, "straight underline");

        let mut wavy = Scene::default();
        wavy.insert_primitive(Underline {
            order: 0,
            padding: 0,
            bounds: underline_bounds,
            content_mask: ContentMask {
                bounds: underline_bounds,
                ..Default::default()
            },
            color: hsla(0.1, 0.9, 0.55, 1.0).into(),
            thickness: ScaledPixels(1.0),
            wavy: gpui::ShaderBool::Enabled,
        });
        wavy.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &wavy,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            0,
            0,
            [0, 0, 0, 255],
            1,
            "wavy underline clears outside the line",
        );
        // The wave must actually oscillate instead of drawing a straight band,
        // and its warm line color must show up somewhere.
        let raw = image.as_raw();
        assert_ne!(
            &raw[3 * 8 * 4..4 * 8 * 4],
            &raw[4 * 8 * 4..5 * 8 * 4],
            "wavy underline oscillates between rows",
        );
        assert!(
            image
                .pixels()
                .any(|texel| texel.0[0] > 100 && texel.0[0] > texel.0[2]),
            "wavy underline draws its warm line color",
        );
    }

    #[test]
    fn generated_border_shader_renders_dashed_rounded_rect() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        let full = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let box_bounds = Bounds {
            origin: gpui::point(ScaledPixels(4.0), ScaledPixels(4.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut border = Scene::default();
        border.insert_primitive(Quad {
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            background: solid_background(hsla(0.05, 0.8, 0.45, 1.0)),
            border_style: BorderStyle::Dashed,
            border_color: hsla(0.6, 0.9, 0.7, 1.0).into(),
            corner_radii: Corners::all(ScaledPixels(2.0)),
            border_widths: Edges::all(ScaledPixels(1.0)),
            ..Default::default()
        });
        border.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &border,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            0,
            0,
            [0, 0, 0, 255],
            1,
            "outside the rounded rect stays clear",
        );
        assert_pixel_close(&image, 8, 8, [207, 78, 23, 255], 3, "interior fill");
        // Dashes: along the top edge some texels carry the border hue while
        // others fall back to gaps. A dash period spans 3px over an 8px edge,
        // so both must appear regardless of dash phase.
        let edge: Vec<[u8; 4]> = (4..12).map(|x| pixel(&image, x, 4)).collect();
        let border_like = edge
            .iter()
            .filter(|texel| texel[2] > 150 && texel[1] > 100)
            .count();
        assert!(
            (1..8).contains(&border_like),
            "top edge mixes dash and gap texels: {edge:?}",
        );

        let mut custom_border = Scene::default();
        custom_border.insert_primitive(Quad {
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            background: solid_background(hsla(0.05, 0.8, 0.45, 1.0)),
            border_style: BorderStyle::Dashed,
            border_dashed_length: 4.0,
            border_dashed_gap: 0.5,
            border_color: hsla(0.6, 0.9, 0.7, 1.0).into(),
            corner_radii: Corners::all(ScaledPixels(2.0)),
            border_widths: Edges::all(ScaledPixels(1.0)),
            ..Default::default()
        });

        custom_border.finish();
        let Some(custom_image) = render_for_contracts(
            &mut renderer,
            &custom_border,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };

        assert_ne!(
            image.as_raw(),
            custom_image.as_raw(),
            "custom dash length and gap change the rendered border",
        );
    }

    #[test]
    fn generated_shadow_shader_renders_soft_falloff() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        let full = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let box_bounds = Bounds {
            origin: gpui::point(ScaledPixels(4.0), ScaledPixels(4.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut shadow = Scene::default();
        shadow.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(2.0),
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            corner_radii: Corners::all(ScaledPixels(2.0)),
            color: hsla(0.7, 0.8, 0.3, 0.7).into(),
            element_bounds: box_bounds,
            element_corner_radii: Corners::all(ScaledPixels(2.0)),
            inset: gpui::ShaderBool::Disabled,
            corner_smoothing: 0.0,
        });
        shadow.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &shadow,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };
        let center = pixel(&image, 8, 8);
        let corner = pixel(&image, 0, 0);
        assert!(
            center[0] > 10,
            "shadow covers the element center: {center:?}",
        );
        assert!(
            corner[0] < 5,
            "shadow falls off to clear at the corner: {corner:?}",
        );
        assert!(
            center[0] > corner[0],
            "shadow has a coverage ramp: center {center:?}, corner {corner:?}",
        );
        // The opaque headless target composites over black, so alpha stays 1.
        assert_eq!(center[3], 255, "shadow center alpha: {center:?}");
        assert_eq!(corner[3], 255, "shadow corner alpha: {corner:?}");
    }

    #[test]
    fn generated_blur_shader_softens_backdrop() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        let full = Bounds {
            origin: gpui::point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size: Size {
                width: ScaledPixels(16.0),
                height: ScaledPixels(16.0),
            },
        };
        let box_bounds = Bounds {
            origin: gpui::point(ScaledPixels(4.0), ScaledPixels(4.0)),
            size: Size {
                width: ScaledPixels(8.0),
                height: ScaledPixels(8.0),
            },
        };
        let mut filter = Scene::default();
        filter.insert_primitive(Quad {
            bounds: full,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            background: checkerboard(hsla(0.2, 0.7, 0.5, 1.0), 2.0),
            ..Default::default()
        });
        filter.insert_primitive(BackdropFilter {
            bounds: box_bounds,
            content_mask: ContentMask {
                bounds: full,
                ..Default::default()
            },
            corner_radii: Corners::all(ScaledPixels(2.0)),
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(1.0))],
            opacity: 0.8,
            ..Default::default()
        });
        filter.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &filter,
            Size {
                width: DevicePixels(16),
                height: DevicePixels(16),
            },
        ) else {
            return;
        };
        // Outside the filter region the checkerboard is untouched.
        assert_pixel_close(&image, 0, 0, [0, 0, 0, 255], 1, "unfiltered off-square");
        assert_pixel_close(&image, 2, 0, [181, 217, 38, 255], 2, "unfiltered on-square");
        // Inside, the blur must mix neighbors: neither sharp nor clear.
        let soft = pixel(&image, 8, 8);
        let sharp = [181, 217, 38, 255];
        assert_ne!(
            soft,
            [0, 0, 0, 255],
            "blurred texel mixes content: {soft:?}"
        );
        let drift = soft
            .iter()
            .zip(sharp.iter())
            .map(|(observed, reference)| observed.abs_diff(*reference))
            .max()
            .unwrap_or(0);
        assert!(
            drift >= 15,
            "blur mixes neighboring texels instead of passing through: {soft:?}",
        );
    }

    #[test]
    fn generated_sprite_shader_blends_opacity() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(0),
            frame_index: 0,
            downscale_level: 0,
        });
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(key, &mut || {
                Ok(Some((
                    Size {
                        width: DevicePixels(2),
                        height: DevicePixels(2),
                    },
                    Cow::Borrowed(&[
                        0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255,
                    ]),
                )))
            })
            .unwrap()
            .unwrap();
        let sprite_bounds = Bounds {
            origin: gpui::point(ScaledPixels(1.0), ScaledPixels(1.0)),
            size: Size {
                width: ScaledPixels(6.0),
                height: ScaledPixels(6.0),
            },
        };
        let mut sprite = Scene::default();
        sprite.insert_primitive(PolychromeSprite {
            order: 0,
            grayscale: gpui::ShaderBool::Disabled,
            opacity: 0.75,
            corner_smoothing: 0.0,
            bounds: sprite_bounds,
            content_mask: ContentMask {
                bounds: sprite_bounds,
                ..Default::default()
            },
            corner_radii: Corners::all(ScaledPixels(1.0)),
            tile,
        });
        sprite.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &sprite,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            0,
            0,
            [0, 0, 0, 255],
            1,
            "outside the sprite stays clear",
        );
        // Texel centers sample exactly, so the red and white texels must land
        // at 75% opacity (255 * 0.75 = 191.25 -> 191).
        assert_pixel_close(
            &image,
            2,
            2,
            [191, 0, 0, 255],
            2,
            "red texel at 75% opacity",
        );
        assert_pixel_close(
            &image,
            5,
            5,
            [191, 191, 191, 255],
            2,
            "white texel at 75% opacity",
        );
        // The 2x2 tile holds saturated texels, so the sprite interior must be
        // lit but capped by the 75% opacity (a full white texel lands at 191).
        let mut sum = [0u64; 3];
        let mut count = 0u64;
        let mut brightest = 0u8;
        for y in 2..6 {
            for x in 2..6 {
                let texel = pixel(&image, x, y);
                for channel in 0..3 {
                    sum[channel] += u64::from(texel[channel]);
                    brightest = brightest.max(texel[channel]);
                }
                count += 1;
            }
        }
        let mean = [sum[0] / count, sum[1] / count, sum[2] / count];
        for (channel, level) in mean.iter().enumerate() {
            assert!(
                (40..=200).contains(level),
                "sprite interior channel {channel} looks sampled and blended, not degenerate: {mean:?}",
            );
        }
        assert!(
            (150..=195).contains(&brightest),
            "75% opacity caps the brightest texel near 191, got {brightest}",
        );
    }

    #[test]
    fn generated_path_shader_fills_and_antialiases() {
        let mut renderer = MetalHeadlessRenderer::new().unwrap();
        let mut builder = gpui::PathBuilder::fill();
        builder.move_to(gpui::point(px(1.0), px(1.0)));
        builder.line_to(gpui::point(px(7.0), px(1.0)));
        builder.line_to(gpui::point(px(4.0), px(7.0)));
        builder.close();
        let content_mask = ContentMask {
            bounds: Bounds {
                origin: gpui::point(px(0.0), px(0.0)),
                size: Size {
                    width: px(8.0),
                    height: px(8.0),
                },
            },
            ..Default::default()
        };
        let path = gpui::PaintedPath::new(
            builder.build().unwrap(),
            1.0,
            content_mask,
            solid_background(hsla(0.45, 0.9, 0.45, 1.0)),
        );
        let mut path_scene = Scene::default();
        path_scene.insert_primitive(path);
        path_scene.finish();
        let Some(image) = render_for_contracts(
            &mut renderer,
            &path_scene,
            Size {
                width: DevicePixels(8),
                height: DevicePixels(8),
            },
        ) else {
            return;
        };
        assert_pixel_close(
            &image,
            4,
            2,
            [11, 218, 156, 255],
            2,
            "triangle interior fill",
        );
        assert_pixel_close(
            &image,
            0,
            7,
            [0, 0, 0, 255],
            1,
            "outside the triangle stays clear",
        );
        // Analytic coverage must produce partially covered texels: strictly between
        // clear and the fill color along the green channel.
        assert!(
            image.pixels().any(|texel| {
                let g = texel.0[1];
                g > 20 && g < 200 && texel.0[0] < 11
            }),
            "triangle edges are antialiased",
        );
    }
}
