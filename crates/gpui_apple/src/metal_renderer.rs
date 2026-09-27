use crate::metal_atlas::MetalAtlas;
use anyhow::Result;
use block2::RcBlock;
use gpui::{
    AtlasTextureId, Bounds, Corners, DevicePixels, FilterRenderTarget, MAX_FILTER_GROUP_DEPTH,
    MonochromeSprite, PaintSurface, Path, PolychromeSprite, PrimitiveBatch, Quad, RenderCommand,
    ScaledPixels, Scene, Shadow, Size, SurfaceSource, Underline, size,
};
use gpui_render::{
    artifacts::{NATIVE_SHADERS, NativeShader},
    blur::{
        BlurAxis, BlurKernel, BlurUniforms, GAUSSIAN_CUTOFF_STANDARD_DEVIATIONS, ScissorRectangle,
        downsampled_dimension,
    },
    path_types::{self, PathRasterizationVertex},
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

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFRetained, CGSize};
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBuffer,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidthOfPlane,
    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder as _, MTLBuffer as _, MTLClearColor,
    MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLCompileOptions,
    MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLDrawable as _,
    MTLGPUFamily, MTLLibrary as _, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType,
    MTLRegion, MTLRenderCommandEncoder as _, MTLRenderPassColorAttachmentDescriptor,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLResourceOptions, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLScissorRect, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTexture as _, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLViewport,
};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;
use smallvec::SmallVec;
use wgsl_rs::std::{vec2f, vec4f};

use std::{
    cell::Cell,
    ffi::c_void,
    mem,
    ptr::{self, NonNull},
    sync::Arc,
};

type Device = ProtocolObject<dyn objc2_metal::MTLDevice>;
type Buffer = ProtocolObject<dyn objc2_metal::MTLBuffer>;
type Texture = ProtocolObject<dyn objc2_metal::MTLTexture>;
type Library = ProtocolObject<dyn objc2_metal::MTLLibrary>;
type SamplerState = ProtocolObject<dyn objc2_metal::MTLSamplerState>;
type CommandQueue = ProtocolObject<dyn objc2_metal::MTLCommandQueue>;
type CommandBuffer = ProtocolObject<dyn objc2_metal::MTLCommandBuffer>;
type RenderCommandEncoder = ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>;
type RenderPipelineState = ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>;
type MetalDrawable = ProtocolObject<dyn CAMetalDrawable>;

// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;

// Buffer slots declared by the generated MSL. Group 0 globals land at 0/1, the group 1 data
// binding at 2, Naga's runtime-array sizes buffer at 3.
const GLOBALS_SLOT: usize = 0;
const FONT_SLOT: usize = 1;
const DATA_SLOT: usize = 2;
const SIZES_SLOT: usize = 3;
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

fn clear_color(red: f64, green: f64, blue: f64, alpha: f64) -> MTLClearColor {
    MTLClearColor {
        red,
        green,
        blue,
        alpha,
    }
}

/// Copies `value` into the vertex-stage argument table at `index`.
fn set_vertex_bytes<T>(encoder: &RenderCommandEncoder, index: usize, value: &T) {
    // SAFETY: `value` is a live reference, so the pointer is non-null and valid for
    // `size_of::<T>()` bytes; Metal copies the bytes before this call returns.
    unsafe {
        encoder.setVertexBytes_length_atIndex(
            NonNull::from(value).cast(),
            mem::size_of::<T>(),
            index,
        );
    }
}

/// Copies `value` into the fragment-stage argument table at `index`.
fn set_fragment_bytes<T>(encoder: &RenderCommandEncoder, index: usize, value: &T) {
    // SAFETY: as in `set_vertex_bytes`.
    unsafe {
        encoder.setFragmentBytes_length_atIndex(
            NonNull::from(value).cast(),
            mem::size_of::<T>(),
            index,
        );
    }
}

/// Binds `texture` to the fragment-stage texture slot `index`.
fn set_fragment_texture(encoder: &RenderCommandEncoder, index: usize, texture: &Texture) {
    // SAFETY: `texture` is a live texture and every slot used here is declared by the
    // generated MSL; the encoder retains the texture until the command buffer completes.
    unsafe { encoder.setFragmentTexture_atIndex(Some(texture), index) };
}

/// Binds `texture` to the vertex-stage texture slot `index`.
fn set_vertex_texture(encoder: &RenderCommandEncoder, index: usize, texture: &Texture) {
    // SAFETY: as in `set_fragment_texture`.
    unsafe { encoder.setVertexTexture_atIndex(Some(texture), index) };
}

/// Binds `sampler` to the fragment-stage sampler slot `index`.
fn set_fragment_sampler(encoder: &RenderCommandEncoder, index: usize, sampler: &SamplerState) {
    // SAFETY: `sampler` is a live sampler state and the slot is declared by the generated MSL.
    unsafe { encoder.setFragmentSamplerState_atIndex(Some(sampler), index) };
}

/// Encodes a non-instanced draw.
fn draw(
    encoder: &RenderCommandEncoder,
    primitive: MTLPrimitiveType,
    vertex_start: usize,
    vertex_count: usize,
) {
    // SAFETY: every draw is preceded by binding a pipeline state and the buffers, bytes and
    // textures its shaders read, and the vertex count matches the data bound for it.
    unsafe {
        encoder.drawPrimitives_vertexStart_vertexCount(primitive, vertex_start, vertex_count)
    };
}

/// Encodes an instanced draw.
fn draw_instanced(
    encoder: &RenderCommandEncoder,
    primitive: MTLPrimitiveType,
    vertex_start: usize,
    vertex_count: usize,
    instance_count: usize,
) {
    // SAFETY: as in `draw`; the instance count equals the number of instances just written
    // to, and bound from, the instance buffer.
    unsafe {
        encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
            primitive,
            vertex_start,
            vertex_count,
            instance_count,
        )
    };
}

fn bind_scene_uniforms(encoder: &RenderCommandEncoder, uniforms: &SceneUniforms) {
    set_vertex_bytes(encoder, GLOBALS_SLOT, &uniforms.globals);
    set_fragment_bytes(encoder, GLOBALS_SLOT, &uniforms.globals);
    set_vertex_bytes(encoder, FONT_SLOT, &uniforms.font);
    set_fragment_bytes(encoder, FONT_SLOT, &uniforms.font);
}

/// Binds an instance slice and the byte length used by Naga's runtime-array size ABI.
fn bind_instances<T>(
    encoder: &RenderCommandEncoder,
    buffer: &Buffer,
    offset: usize,
    instances: &[T],
) {
    bind_instance_bytes(encoder, buffer, offset, mem::size_of_val(instances));
}

fn bind_instance_bytes(
    encoder: &RenderCommandEncoder,
    buffer: &Buffer,
    offset: usize,
    byte_len: usize,
) {
    let byte_len = u32::try_from(byte_len).expect("Metal instance binding exceeds 4 GiB");
    // SAFETY: `buffer` is a live instance buffer and `offset` lies within it (callers check
    // `next_offset <= instance_buffer.size` before binding).
    unsafe {
        encoder.setVertexBuffer_offset_atIndex(Some(buffer), offset, DATA_SLOT);
        encoder.setFragmentBuffer_offset_atIndex(Some(buffer), offset, DATA_SLOT);
    }
    set_vertex_bytes(encoder, SIZES_SLOT, &byte_len);
    set_fragment_bytes(encoder, SIZES_SLOT, &byte_len);
}

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<PooledBuffer>,
}

/// An instance buffer whose ownership moves between the render thread and the Metal thread that
/// runs command-buffer completion handlers.
///
/// objc2-metal leaves `MTLBuffer` `!Send` because a buffer's contents are not synchronized. The
/// pool only ever moves ownership of a whole buffer: the render thread writes it before commit,
/// and the completion handler hands it back after the GPU has finished reading it, so no two
/// threads touch its contents at once. That is the same contract the `metal` crate's
/// `Send`/`Sync` impls on `Buffer` asserted before this crate moved to objc2-metal.
struct PooledBuffer(Retained<Buffer>);

// SAFETY: see the type's documentation; retain/release of Metal objects is thread-safe and the
// buffer's contents are only accessed by one thread at a time.
unsafe impl Send for PooledBuffer {}

impl std::ops::Deref for PooledBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Buffer {
        &self.0
    }
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
    metal_buffer: PooledBuffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &Device,
        unified_memory: bool,
        minimum_size: usize,
    ) -> InstanceBuffer {
        if minimum_size > self.buffer_size {
            self.reset(
                minimum_size
                    .checked_next_power_of_two()
                    .unwrap_or(minimum_size),
            );
        }
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            PooledBuffer(
                device
                    .newBufferWithLength_options(self.buffer_size, options)
                    .expect("failed to allocate a Metal instance buffer"),
            )
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

pub struct MetalRenderer {
    device: Retained<Device>,
    layer: Option<Retained<CAMetalLayer>>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: Retained<CommandQueue>,
    paths_rasterization_pipeline_state: Retained<RenderPipelineState>,
    path_sprites_pipeline_state: Retained<RenderPipelineState>,
    shadows_pipeline_state: Retained<RenderPipelineState>,
    smoothed_shadows_pipeline_state: Retained<RenderPipelineState>,
    quads_pipeline_state: Retained<RenderPipelineState>,
    smoothed_quads_pipeline_state: Retained<RenderPipelineState>,
    underlines_pipeline_state: Retained<RenderPipelineState>,
    monochrome_sprites_pipeline_state: Retained<RenderPipelineState>,
    polychrome_sprites_pipeline_state: Retained<RenderPipelineState>,
    smoothed_polychrome_sprites_pipeline_state: Retained<RenderPipelineState>,
    surfaces_pipeline_state: Retained<RenderPipelineState>,
    // Blur pipelines: downsample (no blend, also used for the final blit), separable gaussian
    // (no blend), and composite (alpha blend into a rounded rect), from shared shader sources.
    blur_downsample_pipeline_state: Retained<RenderPipelineState>,
    blur_pipeline_state: Retained<RenderPipelineState>,
    blur_composite_pipeline_state: Retained<RenderPipelineState>,
    smoothed_blur_composite_pipeline_state: Retained<RenderPipelineState>,
    sampler: Retained<SamplerState>,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    path_intermediate_texture: Option<Retained<Texture>>,
    path_intermediate_msaa_texture: Option<Retained<Texture>>,
    // Offscreen scene target (the scene is rendered here, then blitted to the drawable, so blur
    // passes can sample already-painted content), the half-res ping/pong blur targets, and a
    // full-res target for content-filter groups.
    scene_color_texture: Option<Retained<Texture>>,
    blur_ping_texture: Option<Retained<Texture>>,
    blur_pong_texture: Option<Retained<Texture>>,
    /// Full-resolution offscreen targets a content-filter (`filter`) group renders into before
    /// being blurred and composited back. One per nesting level (indexed by isolation depth) so
    /// nested content blurs isolate consistently with [`MAX_FILTER_GROUP_DEPTH`]; deeper nests render
    /// inline.
    group_textures: Vec<Retained<Texture>>,
    intermediate_texture_size: Option<Size<DevicePixels>>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<Retained<Texture>>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.setOpaque(!transparent);
        layer.setMaximumDrawableCount(3);
        // Allow texture reading for visual tests (captures screenshots without ScreenCaptureKit)
        #[cfg(any(
            test,
            feature = "bench-support",
            feature = "test-support",
            feature = "render-to-image"
        ))]
        layer.setFramebufferOnly(false);
        layer.setAllowsNextDrawableTimeout(false);
        layer.setNeedsDisplayOnBoundsChange(true);
        layer.setAutoresizingMask(
            CAAutoresizingMask::LayerWidthSizable | CAAutoresizingMask::LayerHeightSizable,
        );

        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> Retained<Device> {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `MTLCreateSystemDefaultDevice()`.
        if let Some(d) = MTLCopyAllDevices()
            .iter()
            .min_by_key(|d| (d.isRemovable(), !d.isLowPower()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            MTLCreateSystemDefaultDevice().unwrap_or_else(|| {
                log::error!("unable to access a compatible graphics device");
                std::process::exit(1);
            })
        }
    }

    fn new_internal(
        device: Retained<Device>,
        layer: Option<Retained<CAMetalLayer>>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
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
        let mut libraries: Vec<(&'static str, Retained<Library>)> = Vec::new();
        let mut library_for = |source: &'static str| -> Retained<Library> {
            if let Some((_, library)) = libraries
                .iter()
                .find(|(registered, _)| *registered == source)
            {
                return library.clone();
            }
            let library = device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(source),
                    Some(&MTLCompileOptions::new()),
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "error building metal library: {}",
                        error.localizedDescription()
                    )
                });
            libraries.push((source, library.clone()));
            library
        };

        let mut pipeline = |label: &str| -> (&'static NativeShader, Retained<Library>) {
            let shader = native_shader(label);
            (shader, library_for(shader.msl))
        };

        let (path_rasterization_shader, path_rasterization_library) =
            pipeline("path_rasterization");
        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &path_rasterization_library,
            path_rasterization_shader,
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let (paths_shader, paths_library) = pipeline("paths");
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &paths_library,
            paths_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (shadows_shader, shadows_library) = pipeline("shadows");
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &shadows_library,
            shadows_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_shadows_shader, smoothed_shadows_library) = pipeline("smoothed_shadows");
        let smoothed_shadows_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_shadows_library,
            smoothed_shadows_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (quads_shader, quads_library) = pipeline("quads");
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &quads_library,
            quads_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_quads_shader, smoothed_quads_library) = pipeline("smoothed_quads");
        let smoothed_quads_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_quads_library,
            smoothed_quads_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (underlines_shader, underlines_library) = pipeline("underlines");
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &underlines_library,
            underlines_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (monochrome_shader, monochrome_library) = pipeline("monochrome_sprites");
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &monochrome_library,
            monochrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (polychrome_shader, polychrome_library) = pipeline("polychrome_sprites");
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &polychrome_library,
            polychrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_polychrome_shader, smoothed_polychrome_library) =
            pipeline("smoothed_polychrome_sprites");
        let smoothed_polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &smoothed_polychrome_library,
            smoothed_polychrome_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (surfaces_shader, surfaces_library) = pipeline("surfaces");
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &surfaces_library,
            surfaces_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (blur_downsample_shader, blur_downsample_library) = pipeline("blur_downsample");
        let blur_downsample_pipeline_state = build_blur_pipeline_state(
            &device,
            &blur_downsample_library,
            blur_downsample_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (blur_shader, blur_library) = pipeline("blur");
        let blur_pipeline_state = build_blur_pipeline_state(
            &device,
            &blur_library,
            blur_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        // Premultiplied blend (One / OneMinusSourceAlpha) — the composite outputs a premultiplied
        // blurred sample; straight-alpha blending would darken the faded edges.
        let (blur_composite_shader, blur_composite_library) = pipeline("blur_composite");
        let blur_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &blur_composite_library,
            blur_composite_shader,
            MTLPixelFormat::BGRA8Unorm,
        );
        let (smoothed_blur_composite_shader, smoothed_blur_composite_library) =
            pipeline("smoothed_blur_composite");
        let smoothed_blur_composite_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &smoothed_blur_composite_library,
            smoothed_blur_composite_shader,
            MTLPixelFormat::BGRA8Unorm,
        );

        let sampler_descriptor = MTLSamplerDescriptor::new();
        sampler_descriptor.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.setMagFilter(MTLSamplerMinMagFilter::Linear);
        let sampler = device
            .newSamplerStateWithDescriptor(&sampler_descriptor)
            .expect("failed to create a Metal sampler state");

        let command_queue = device
            .newCommandQueue()
            .expect("failed to create a Metal command queue");
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));
        let core_video_texture_cache = new_texture_cache(&device).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
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
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            scene_color_texture: None,
            blur_ping_texture: None,
            blur_pong_texture: None,
            group_textures: Vec::new(),
            intermediate_texture_size: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        }
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
        self.path_intermediate_texture = None;
        self.path_intermediate_msaa_texture = None;
        self.scene_color_texture = None;
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

        let make_color_texture = |width: usize, height: usize| {
            let descriptor = texture_descriptor(
                width.max(1),
                height.max(1),
                MTLStorageMode::Private,
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            );
            new_texture(&self.device, &descriptor)
        };

        if requirements.uses_path_target && self.path_intermediate_texture.is_none() {
            let texture_descriptor = texture_descriptor(
                full_w,
                full_h,
                MTLStorageMode::Private,
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            );
            self.path_intermediate_texture = Some(new_texture(&self.device, &texture_descriptor));

            // Storage mode guidance:
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            if self.path_sample_count > 1 {
                let storage_mode = if self.is_apple_gpu {
                    MTLStorageMode::Memoryless
                } else {
                    MTLStorageMode::Private
                };
                let msaa_descriptor = texture_descriptor;
                msaa_descriptor.setTextureType(MTLTextureType::Type2DMultisample);
                msaa_descriptor.setStorageMode(storage_mode);
                // SAFETY: PATH_SAMPLE_COUNT (4) is supported by every Metal device.
                unsafe { msaa_descriptor.setSampleCount(self.path_sample_count as _) };
                self.path_intermediate_msaa_texture =
                    Some(new_texture(&self.device, &msaa_descriptor));
            }
        }

        if requirements.uses_offscreen_target {
            self.scene_color_texture
                .get_or_insert_with(|| make_color_texture(full_w, full_h));
            let blur_width = downsampled_dimension(full_w as u32) as usize;
            let blur_height = downsampled_dimension(full_h as u32) as usize;
            self.blur_ping_texture
                .get_or_insert_with(|| make_color_texture(blur_width, blur_height));
            self.blur_pong_texture
                .get_or_insert_with(|| make_color_texture(blur_width, blur_height));
            while self.group_textures.len() < requirements.isolated_target_count {
                self.group_textures.push(make_color_texture(full_w, full_h));
            }
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
        let viewport_size = layer.drawableSize();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.nextDrawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let mut instance_buffer = self.acquire_instance_buffer(scene);
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
        let drawable = layer
            .nextDrawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            self.draw_primitives(scene, &mut instance_buffer, &drawable, viewport_size)?;

        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        self.instance_buffer_pool.lock().release(instance_buffer);

        let texture = drawable.texture();
        let width = texture.width() as u32;
        let height = texture.height() as u32;
        let mut pixels = read_bgra_pixels(&texture, width, height);
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create RgbaImage from pixel data"))
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Create an offscreen texture as render target
        let texture_descriptor = texture_descriptor(
            size.width.0 as usize,
            size.height.0 as usize,
            MTLStorageMode::Managed,
            MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
        );
        let target_texture = new_texture(&self.device, &texture_descriptor);

        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            self.draw_primitives_to_texture(scene, &mut instance_buffer, &target_texture, size)?;

        if !self.is_unified_memory {
            let blit = command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| anyhow::anyhow!("failed to create a Metal blit encoder"))?;
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        self.instance_buffer_pool.lock().release(instance_buffer);

        let width = size.width.0 as u32;
        let height = size.height.0 as u32;
        let mut pixels = read_bgra_pixels(&target_texture, width, height);
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create RgbaImage from pixel data"))
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

        let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
            texture.width() != size.width.0 as usize || texture.height() != size.height.0 as usize
        });
        if needs_new_target {
            let texture_descriptor = texture_descriptor(
                size.width.0 as usize,
                size.height.0 as usize,
                MTLStorageMode::Private,
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            );
            self.headless_render_target = Some(new_texture(&self.device, &texture_descriptor));
        }
        let target_texture = self
            .headless_render_target
            .clone()
            .expect("just ensured the render target exists");

        let mut instance_buffer = self.acquire_instance_buffer(scene);
        let command_buffer =
            self.draw_primitives_to_texture(scene, &mut instance_buffer, &target_texture, size)?;
        self.release_instance_buffer_when_complete(&command_buffer, instance_buffer);
        command_buffer.commit();
        Ok(())
    }

    fn acquire_instance_buffer(&self, scene: &Scene) -> InstanceBuffer {
        self.instance_buffer_pool.lock().acquire(
            &self.device,
            self.is_unified_memory,
            required_instance_buffer_size(scene),
        )
    }

    fn release_instance_buffer_when_complete(
        &self,
        command_buffer: &CommandBuffer,
        instance_buffer: InstanceBuffer,
    ) {
        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(instance_buffer));
        let block = RcBlock::new(move |_: NonNull<CommandBuffer>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
        });
        // SAFETY: `addCompletedHandler:` copies the block, and the block only touches the
        // captured pool and buffer, which it owns; it never dereferences the command buffer.
        unsafe { command_buffer.addCompletedHandler(RcBlock::as_ptr(&block)) };
    }

    fn draw_primitives(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        drawable: &MetalDrawable,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<CommandBuffer>> {
        self.draw_primitives_to_texture(scene, instance_buffer, &drawable.texture(), viewport_size)
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_buffer: &mut InstanceBuffer,
        texture: &Texture,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<CommandBuffer>> {
        self.prepare_intermediate_textures(scene, viewport_size);
        let command_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or_else(|| anyhow::anyhow!("failed to create a Metal command buffer"))?;
        let alpha = if self.opaque { 1. } else { 0. };
        let mut instance_offset = 0;
        let scene_uniforms = SceneUniforms::new(viewport_size);

        // Render the scene into an offscreen color texture (so filters can sample it), then
        // blit it to `texture`. Owned clones keep the textures borrowable without borrowing
        // `self` across the batch loop (which calls `&mut self` methods like `draw_surfaces`).
        // Only route through the offscreen scene texture when the scene actually contains blur
        // filters; otherwise render straight to `texture` exactly as before (no regression, no
        // extra blit for the common case).
        let use_offscreen = scene.requires_offscreen_rendering();
        let scene_color_owned = self.scene_color_texture.clone();
        let blur_ping_owned = self.blur_ping_texture.clone();
        let blur_pong_owned = self.blur_pong_texture.clone();
        let group_owned = self
            .group_textures
            .iter()
            .cloned()
            .collect::<SmallVec<[Retained<Texture>; MAX_FILTER_GROUP_DEPTH]>>();
        let scene_color: &Texture = if use_offscreen {
            scene_color_owned.as_deref().unwrap_or(texture)
        } else {
            texture
        };
        // The active render target; switches to the group texture inside a content-filter group.
        let mut current_target: &Texture = scene_color;
        let mut filter_stack = SmallVec::<[&Texture; MAX_FILTER_GROUP_DEPTH]>::new();

        let mut command_encoder = new_command_encoder_for_texture(
            &command_buffer,
            current_target,
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
                RenderCommand::Batch(PrimitiveBatch::Paths {
                    range,
                    rasterization_vertex_count,
                    sprite_count,
                }) => {
                    if *rasterization_vertex_count == 0 {
                        continue;
                    }
                    let paths = &scene.paths[range.clone()];
                    command_encoder.endEncoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        *rasterization_vertex_count,
                        instance_buffer,
                        &mut instance_offset,
                        &scene_uniforms,
                        &command_buffer,
                    );

                    command_encoder = new_command_encoder_for_texture(
                        &command_buffer,
                        current_target,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.setLoadAction(MTLLoadAction::Load);
                        },
                    );

                    if did_draw {
                        self.draw_paths_from_intermediate(
                            paths,
                            *sprite_count,
                            instance_buffer,
                            &mut instance_offset,
                            &scene_uniforms,
                            &command_encoder,
                        )
                    } else {
                        false
                    }
                }
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
                    &scene.surface_opacities()[range.clone()],
                    &scene_uniforms,
                    &command_encoder,
                ),
                RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) => {
                    command_encoder.endEncoding();
                    if let (Some(ping), Some(pong)) =
                        (blur_ping_owned.as_deref(), blur_pong_owned.as_deref())
                    {
                        for filter in &scene.backdrop_filters[range.clone()] {
                            self.metal_blur_and_composite(
                                &command_buffer,
                                &scene_uniforms,
                                current_target,
                                current_target,
                                ping,
                                pong,
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
                        &command_buffer,
                        current_target,
                        viewport_size,
                        |color_attachment| {
                            color_attachment.setLoadAction(MTLLoadAction::Load);
                        },
                    );
                    true
                }
                RenderCommand::BeginFilter {
                    target: FilterRenderTarget::Isolated(target_index),
                    ..
                } => {
                    command_encoder.endEncoding();
                    filter_stack.push(current_target);
                    current_target = &group_owned[target_index.as_usize()];
                    command_encoder = new_command_encoder_for_texture(
                        &command_buffer,
                        current_target,
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
                    let parent = filter_stack
                        .pop()
                        .expect("render plan emitted an unmatched isolated filter end");
                    command_encoder.endEncoding();
                    if let (Some(ping), Some(pong)) =
                        (blur_ping_owned.as_deref(), blur_pong_owned.as_deref())
                    {
                        self.metal_blur_and_composite(
                            &command_buffer,
                            &scene_uniforms,
                            current_target,
                            parent,
                            ping,
                            pong,
                            viewport_size,
                            boundary.bounds,
                            boundary.content_mask.bounds,
                            boundary.corner_radii,
                            boundary.corner_smoothing,
                            boundary.max_blur_radius(),
                            boundary.opacity,
                            false,
                        );
                    }
                    current_target = parent;
                    command_encoder = new_command_encoder_for_texture(
                        &command_buffer,
                        current_target,
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

        // Present the offscreen scene by copying it into the drawable/target texture.
        if use_offscreen && scene_color_owned.is_some() {
            self.run_metal_blur_pass(
                &command_buffer,
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
            instance_buffer
                .metal_buffer
                .didModifyRange(NSRange::new(0, instance_offset));
        }

        Ok(command_buffer)
    }

    /// Run a single blur pass: draw a full-screen (or composite) quad sampling `source` into
    /// `target`. `params` is supplied to both shader stages; `load` keeps existing target
    /// contents (used by the composite), otherwise the target is cleared.
    #[allow(clippy::too_many_arguments)]
    fn run_metal_blur_pass(
        &self,
        command_buffer: &CommandBuffer,
        pipeline: &RenderPipelineState,
        target: &Texture,
        source: &Texture,
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
        let encoder = &*encoder;
        encoder.setScissorRect(MTLScissorRect {
            x: scissor.x as usize,
            y: scissor.y as usize,
            width: scissor.width as usize,
            height: scissor.height as usize,
        });
        encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(encoder, scene_uniforms);
        set_vertex_bytes(encoder, DATA_SLOT, &params);
        set_fragment_bytes(encoder, DATA_SLOT, &params);
        set_fragment_texture(encoder, PRIMARY_TEXTURE_SLOT, source);
        set_fragment_sampler(encoder, SAMPLER_SLOT, &self.sampler);
        draw(encoder, primitive, 0, vertex_count);
        encoder.endEncoding();
    }

    /// Blur `source` (full-resolution) using the half-res ping/pong textures and composite the
    /// result into `target`, clipped to `bounds`/`corner_radii`/`content_mask` and modulated by
    /// `opacity`. Shared by the backdrop and content-filter paths, mirroring the wgpu
    /// backend's pass structure via the shared `gpui_render::blur` contracts.
    #[allow(clippy::too_many_arguments)]
    fn metal_blur_and_composite(
        &self,
        command_buffer: &CommandBuffer,
        scene_uniforms: &SceneUniforms,
        source: &Texture,
        target: &Texture,
        ping: &Texture,
        pong: &Texture,
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
        let Some(kernel) = BlurKernel::for_radius(blur_radius) else {
            return;
        };
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
        self.run_metal_blur_pass(
            command_buffer,
            &self.blur_pipeline_state,
            pong,
            ping,
            blur_viewport_size,
            scene_uniforms,
            BlurUniforms::gaussian(BlurAxis::Horizontal, blur_size, kernel),
            scissor,
            MTLPrimitiveType::Triangle,
            3,
            false,
        );
        self.run_metal_blur_pass(
            command_buffer,
            &self.blur_pipeline_state,
            ping,
            pong,
            blur_viewport_size,
            scene_uniforms,
            BlurUniforms::gaussian(BlurAxis::Vertical, blur_size, kernel),
            scissor,
            MTLPrimitiveType::Triangle,
            3,
            false,
        );

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
            blur_size,
            [full_width as f32, full_height as f32],
        );
        let encoder = new_command_encoder_for_texture(
            command_buffer,
            target,
            viewport_size,
            |color_attachment| {
                color_attachment.setLoadAction(MTLLoadAction::Load);
            },
        );
        let encoder = &*encoder;
        let pipeline = if composite_uniforms.corner_smoothing > 0.0 {
            &self.smoothed_blur_composite_pipeline_state
        } else {
            &self.blur_composite_pipeline_state
        };
        encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(encoder, scene_uniforms);
        set_vertex_bytes(encoder, DATA_SLOT, &composite_uniforms);
        set_fragment_bytes(encoder, DATA_SLOT, &composite_uniforms);
        set_fragment_texture(encoder, PRIMARY_TEXTURE_SLOT, ping);
        set_fragment_sampler(encoder, SAMPLER_SLOT, &self.sampler);
        draw(encoder, MTLPrimitiveType::TriangleStrip, 0, 4);
        encoder.endEncoding();
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        rasterization_vertex_count: usize,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_buffer: &CommandBuffer,
    ) -> bool {
        if paths.is_empty() {
            return true;
        }
        let Some(intermediate_texture) = &self.path_intermediate_texture else {
            return false;
        };

        let render_pass_descriptor = MTLRenderPassDescriptor::new();
        let color_attachment = render_pass_color_attachment(&render_pass_descriptor);
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(clear_color(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.setTexture(Some(msaa_texture));
            color_attachment.setResolveTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.setTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::Store);
        }

        let command_encoder = new_render_command_encoder(command_buffer, &render_pass_descriptor);
        let command_encoder = &*command_encoder;
        command_encoder.setRenderPipelineState(&self.paths_rasterization_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        align_offset(instance_offset);
        let vertices_bytes_len =
            mem::size_of::<PathRasterizationVertex>() * rasterization_vertex_count;
        let next_offset = *instance_offset + vertices_bytes_len;
        if next_offset > instance_buffer.size {
            command_encoder.endEncoding();
            return false;
        }
        let buffer_contents = unsafe {
            (instance_buffer.metal_buffer.contents().as_ptr() as *mut u8).add(*instance_offset)
        } as *mut PathRasterizationVertex;
        let mut vertices = path_types::rasterization_vertices(paths);
        for index in 0..rasterization_vertex_count {
            let Some(vertex) = vertices.next() else {
                command_encoder.endEncoding();
                return false;
            };
            unsafe { buffer_contents.add(index).write(vertex) };
        }
        if vertices.next().is_some() {
            command_encoder.endEncoding();
            return false;
        }
        bind_instance_bytes(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            vertices_bytes_len,
        );
        draw(
            command_encoder,
            MTLPrimitiveType::Triangle,
            0,
            rasterization_vertex_count,
        );
        *instance_offset = next_offset;

        command_encoder.endEncoding();
        true
    }

    fn draw_shadows(
        &self,
        shadows: &[Shadow],
        smoothed: bool,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &RenderCommandEncoder,
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

        let buffer_contents = unsafe {
            (instance_buffer.metal_buffer.contents().as_ptr() as *mut u8).add(*instance_offset)
        };
        unsafe {
            ptr::copy_nonoverlapping(
                shadows.as_ptr() as *const u8,
                buffer_contents,
                shadow_bytes_len,
            );
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            shadows,
        );

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            0,
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
        command_encoder: &RenderCommandEncoder,
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

        let buffer_contents = unsafe {
            (instance_buffer.metal_buffer.contents().as_ptr() as *mut u8).add(*instance_offset)
        };
        unsafe {
            ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, buffer_contents, quad_bytes_len);
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            quads,
        );

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            0,
            4,
            quads.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        sprite_count: usize,
        instance_buffer: &mut InstanceBuffer,
        instance_offset: &mut usize,
        scene_uniforms: &SceneUniforms,
        command_encoder: &RenderCommandEncoder,
    ) -> bool {
        let Some(_) = paths.first() else {
            return true;
        };

        let Some(ref intermediate_texture) = self.path_intermediate_texture else {
            return false;
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        align_offset(instance_offset);
        let sprite_bytes_len = mem::size_of::<path_types::PathSprite>() * sprite_count;
        let next_offset = *instance_offset + sprite_bytes_len;
        if next_offset > instance_buffer.size {
            return false;
        }

        command_encoder.setRenderPipelineState(&self.path_sprites_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);
        set_fragment_texture(command_encoder, PRIMARY_TEXTURE_SLOT, intermediate_texture);
        set_fragment_sampler(command_encoder, SAMPLER_SLOT, &self.sampler);

        let buffer_contents = unsafe {
            (instance_buffer.metal_buffer.contents().as_ptr() as *mut u8).add(*instance_offset)
        } as *mut path_types::PathSprite;
        let mut sprites = path_types::sprites(paths);
        for index in 0..sprite_count {
            let Some(sprite) = sprites.next() else {
                return false;
            };
            unsafe { buffer_contents.add(index).write(sprite) };
        }
        if sprites.next().is_some() {
            return false;
        }

        bind_instance_bytes(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            sprite_bytes_len,
        );
        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            0,
            4,
            sprite_count,
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
        command_encoder: &RenderCommandEncoder,
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

        let buffer_contents = unsafe {
            (instance_buffer.metal_buffer.contents().as_ptr() as *mut u8).add(*instance_offset)
        };
        unsafe {
            ptr::copy_nonoverlapping(
                underlines.as_ptr() as *const u8,
                buffer_contents,
                underline_bytes_len,
            );
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            underlines,
        );

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            0,
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
        command_encoder: &RenderCommandEncoder,
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

        let texture = self.sprite_atlas.metal_texture(texture_id);
        command_encoder.setRenderPipelineState(&self.monochrome_sprites_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        let buffer_contents = unsafe {
            (instance_buffer.metal_buffer.contents().as_ptr() as *mut u8).add(*instance_offset)
        };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }
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
        set_fragment_sampler(command_encoder, SAMPLER_SLOT, &self.sampler);

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            0,
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
        command_encoder: &RenderCommandEncoder,
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

        let texture = self.sprite_atlas.metal_texture(texture_id);
        let pipeline = if smoothed {
            &self.smoothed_polychrome_sprites_pipeline_state
        } else {
            &self.polychrome_sprites_pipeline_state
        };
        command_encoder.setRenderPipelineState(pipeline);
        bind_scene_uniforms(command_encoder, scene_uniforms);

        let buffer_contents = unsafe {
            (instance_buffer.metal_buffer.contents().as_ptr() as *mut u8).add(*instance_offset)
        };
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr() as *const u8,
                buffer_contents,
                sprite_bytes_len,
            );
        }
        bind_instances(
            command_encoder,
            &instance_buffer.metal_buffer,
            *instance_offset,
            sprites,
        );
        set_vertex_texture(command_encoder, PRIMARY_TEXTURE_SLOT, &texture);
        set_fragment_texture(command_encoder, PRIMARY_TEXTURE_SLOT, &texture);
        set_fragment_sampler(command_encoder, SAMPLER_SLOT, &self.sampler);

        draw_instanced(
            command_encoder,
            MTLPrimitiveType::TriangleStrip,
            0,
            4,
            sprites.len(),
        );
        *instance_offset = next_offset;
        true
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        opacities: &[f32],
        scene_uniforms: &SceneUniforms,
        command_encoder: &RenderCommandEncoder,
    ) -> bool {
        command_encoder.setRenderPipelineState(&self.surfaces_pipeline_state);
        bind_scene_uniforms(command_encoder, scene_uniforms);
        set_fragment_sampler(command_encoder, SAMPLER_SLOT, &self.sampler);

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = match &surface.source {
                SurfaceSource::Surface(image_buffer) => image_buffer,
                SurfaceSource::Unsupported(size) => {
                    log::error!("Metal cannot draw unsupported surface source with size {size:?}");
                    continue;
                }
            };

            assert_eq!(
                CVPixelBufferGetPixelFormatType(image_buffer),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let y_texture = create_plane_texture(
                &self.core_video_texture_cache,
                image_buffer,
                MTLPixelFormat::R8Unorm,
                0,
            )
            .unwrap();
            let cb_cr_texture = create_plane_texture(
                &self.core_video_texture_cache,
                image_buffer,
                MTLPixelFormat::RG8Unorm,
                1,
            )
            .unwrap();

            let surface_uniforms = SurfaceUniforms {
                bounds: surface.bounds.into(),
                content_mask: surface.content_mask.bounds.into(),
                color_format: SurfaceColorFormat::Yuv,
                opacity: opacities.get(index).copied().unwrap_or(1.0),
                padding0: 0,
                padding1: 0,
                padding2: 0,
                padding3: 0,
                padding4: 0,
                padding5: 0,
            };
            set_vertex_bytes(command_encoder, DATA_SLOT, &surface_uniforms);
            set_fragment_bytes(command_encoder, DATA_SLOT, &surface_uniforms);
            let y_metal_texture = CVMetalTextureGetTexture(&y_texture);
            let cb_cr_metal_texture = CVMetalTextureGetTexture(&cb_cr_texture);
            // SAFETY: both textures come from live CVMetalTextures created above and the slots
            // are declared by the generated surface shader; the encoder retains them.
            unsafe {
                command_encoder
                    .setFragmentTexture_atIndex(y_metal_texture.as_deref(), PRIMARY_TEXTURE_SLOT);
                command_encoder.setFragmentTexture_atIndex(
                    cb_cr_metal_texture.as_deref(),
                    SECONDARY_TEXTURE_SLOT,
                );
            }

            draw(command_encoder, MTLPrimitiveType::TriangleStrip, 0, 4);
        }
        true
    }
}

fn new_command_encoder_for_texture(
    command_buffer: &CommandBuffer,
    texture: &Texture,
    viewport_size: Size<DevicePixels>,
    configure_color_attachment: impl Fn(&MTLRenderPassColorAttachmentDescriptor),
) -> Retained<RenderCommandEncoder> {
    let render_pass_descriptor = MTLRenderPassDescriptor::new();
    let color_attachment = render_pass_color_attachment(&render_pass_descriptor);
    color_attachment.setTexture(Some(texture));
    color_attachment.setStoreAction(MTLStoreAction::Store);
    configure_color_attachment(&color_attachment);

    let command_encoder = new_render_command_encoder(command_buffer, &render_pass_descriptor);
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

/// Returns color attachment 0 of a render pass descriptor.
fn render_pass_color_attachment(
    descriptor: &MTLRenderPassDescriptor,
) -> Retained<MTLRenderPassColorAttachmentDescriptor> {
    // SAFETY: Metal guarantees at least one color attachment slot (index 0) on every device.
    unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) }
}

fn new_render_command_encoder(
    command_buffer: &CommandBuffer,
    descriptor: &MTLRenderPassDescriptor,
) -> Retained<RenderCommandEncoder> {
    command_buffer
        .renderCommandEncoderWithDescriptor(descriptor)
        .expect("failed to create a Metal render command encoder")
}

/// A BGRA8 texture descriptor with the given size, storage mode, and usage.
fn texture_descriptor(
    width: usize,
    height: usize,
    storage_mode: MTLStorageMode,
    usage: MTLTextureUsage,
) -> Retained<MTLTextureDescriptor> {
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: the sizes are drawable or render-target sizes, within Metal's texture limits.
    unsafe {
        descriptor.setWidth(width);
        descriptor.setHeight(height);
    }
    descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    descriptor.setStorageMode(storage_mode);
    descriptor.setUsage(usage);
    descriptor
}

fn new_texture(device: &Device, descriptor: &MTLTextureDescriptor) -> Retained<Texture> {
    device
        .newTextureWithDescriptor(descriptor)
        .expect("failed to allocate a Metal texture")
}

/// Reads a BGRA8 texture back into a tightly packed byte vector.
#[cfg(any(
    test,
    feature = "bench-support",
    feature = "test-support",
    feature = "render-to-image"
))]
fn read_bgra_pixels(texture: &Texture, width: u32, height: u32) -> Vec<u8> {
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
    // SAFETY: `pixels` holds `height` rows of `bytes_per_row` bytes, exactly the region read,
    // and the GPU work writing the texture has completed before this is called.
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            NonNull::new(pixels.as_mut_ptr())
                .expect("Vec::as_mut_ptr is never null")
                .cast(),
            bytes_per_row,
            region,
            0,
        );
    }
    pixels
}

/// Creates a CoreVideo texture cache backed by `device`.
fn new_texture_cache(device: &Device) -> Result<CFRetained<CVMetalTextureCache>> {
    let mut cache: *mut CVMetalTextureCache = ptr::null_mut();
    // SAFETY: `device` is a live MTLDevice and `cache` is a valid out pointer.
    let result =
        unsafe { CVMetalTextureCache::create(None, None, device, None, NonNull::from(&mut cache)) };
    let cache = NonNull::new(cache)
        .filter(|_| result == kCVReturnSuccess)
        .ok_or_else(|| anyhow::anyhow!("could not create texture cache, code: {result}"))?;
    // SAFETY: CVMetalTextureCacheCreate returns a +1 reference (create rule).
    Ok(unsafe { CFRetained::from_raw(cache) })
}

/// Creates a CoreVideo-backed Metal texture for one plane of `image_buffer`.
fn create_plane_texture(
    texture_cache: &CVMetalTextureCache,
    image_buffer: &CVPixelBuffer,
    pixel_format: MTLPixelFormat,
    plane: usize,
) -> Result<CFRetained<CVMetalTexture>> {
    let mut texture: *mut CVMetalTexture = ptr::null_mut();
    // SAFETY: the cache and image buffer are live CoreVideo objects, the plane index and size
    // come from the image buffer itself, and `texture` is a valid out pointer.
    let result = unsafe {
        CVMetalTextureCache::create_texture_from_image(
            None,
            texture_cache,
            image_buffer,
            None,
            pixel_format,
            CVPixelBufferGetWidthOfPlane(image_buffer, plane),
            CVPixelBufferGetHeightOfPlane(image_buffer, plane),
            plane,
            NonNull::from(&mut texture),
        )
    };
    let texture = NonNull::new(texture)
        .filter(|_| result == kCVReturnSuccess)
        .ok_or_else(|| anyhow::anyhow!("could not create texture, code: {result}"))?;
    // SAFETY: CVMetalTextureCacheCreateTextureFromImage returns a +1 reference (create rule).
    Ok(unsafe { CFRetained::from_raw(texture) })
}

fn shader_function(
    library: &Library,
    name: &str,
) -> Retained<ProtocolObject<dyn objc2_metal::MTLFunction>> {
    library
        .newFunctionWithName(&NSString::from_str(name))
        .unwrap_or_else(|| panic!("error locating {name}: function not found"))
}

/// A pipeline descriptor wired to `shader`'s entry points, plus its color attachment 0 with
/// `pixel_format` set.
fn pipeline_descriptor(
    library: &Library,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
) -> (
    Retained<MTLRenderPipelineDescriptor>,
    Retained<objc2_metal::MTLRenderPipelineColorAttachmentDescriptor>,
) {
    let vertex_fn = shader_function(library, shader.vertex_entry);
    let fragment_fn = shader_function(library, shader.fragment_entry);

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(shader.label)));
    descriptor.setVertexFunction(Some(&vertex_fn));
    descriptor.setFragmentFunction(Some(&fragment_fn));
    // SAFETY: Metal guarantees at least one color attachment slot (index 0) on every device.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    (descriptor, color_attachment)
}

fn new_render_pipeline_state(
    device: &Device,
    descriptor: &MTLRenderPipelineDescriptor,
) -> Retained<RenderPipelineState> {
    device
        .newRenderPipelineStateWithDescriptor_error(descriptor)
        .unwrap_or_else(|error| {
            panic!(
                "could not create render pipeline state: {}",
                error.localizedDescription()
            )
        })
}

fn build_pipeline_state(
    device: &Device,
    library: &Library,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
) -> Retained<RenderPipelineState> {
    let (descriptor, color_attachment) = pipeline_descriptor(library, shader, pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::One);

    new_render_pipeline_state(device, &descriptor)
}

fn build_path_sprite_pipeline_state(
    device: &Device,
    library: &Library,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
) -> Retained<RenderPipelineState> {
    let (descriptor, color_attachment) = pipeline_descriptor(library, shader, pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::One);

    new_render_pipeline_state(device, &descriptor)
}

fn build_path_rasterization_pipeline_state(
    device: &Device,
    library: &Library,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
    path_sample_count: u32,
) -> Retained<RenderPipelineState> {
    let (descriptor, color_attachment) = pipeline_descriptor(library, shader, pixel_format);
    if path_sample_count > 1 {
        descriptor.setRasterSampleCount(path_sample_count as _);
        descriptor.setAlphaToCoverageEnabled(false);
    }
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    new_render_pipeline_state(device, &descriptor)
}

// Blur downsample/gaussian passes overwrite their target (no blending). The composite pass
// uses the normal alpha-blending pipeline (`build_path_sprite_pipeline_state`) instead.
fn build_blur_pipeline_state(
    device: &Device,
    library: &Library,
    shader: &NativeShader,
    pixel_format: MTLPixelFormat,
) -> Retained<RenderPipelineState> {
    let (descriptor, color_attachment) = pipeline_descriptor(library, shader, pixel_format);
    color_attachment.setBlendingEnabled(false);

    new_render_pipeline_state(device, &descriptor)
}

fn required_instance_buffer_size(scene: &Scene) -> usize {
    let mut required = 0;
    let mut reserve = |element_size: usize, count: usize| {
        if count > 0 {
            align_offset(&mut required);
            required = required.saturating_add(element_size.saturating_mul(count));
        }
    };

    for command in scene.render_commands() {
        let RenderCommand::Batch(batch) = command else {
            continue;
        };
        match batch {
            PrimitiveBatch::Shadows { range, .. } => reserve(mem::size_of::<Shadow>(), range.len()),
            PrimitiveBatch::Quads { range, .. } => reserve(mem::size_of::<Quad>(), range.len()),
            PrimitiveBatch::Paths {
                rasterization_vertex_count,
                sprite_count,
                ..
            } if *rasterization_vertex_count > 0 => {
                reserve(
                    mem::size_of::<path_types::PathRasterizationVertex>(),
                    *rasterization_vertex_count,
                );
                reserve(mem::size_of::<path_types::PathSprite>(), *sprite_count);
            }
            PrimitiveBatch::Underlines(range) => reserve(mem::size_of::<Underline>(), range.len()),
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                reserve(mem::size_of::<MonochromeSprite>(), range.len())
            }
            PrimitiveBatch::PolychromeSprites { range, .. } => {
                reserve(mem::size_of::<PolychromeSprite>(), range.len())
            }
            PrimitiveBatch::Paths { .. }
            | PrimitiveBatch::SubpixelSprites { .. }
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
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
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
        AtlasKey, BackdropFilter, BorderStyle, ContentMask, Edges, ImageId, Path, PlatformAtlas,
        PlatformHeadlessRenderer, RenderImageParams, RenderSvgParams, ScaledFilter,
        TransformationMatrix, checkerboard, hsla, linear_color_stop, linear_gradient,
        pattern_slash, px, solid_background, white,
    };
    use std::borrow::Cow;

    #[test]
    fn intermediate_textures_follow_scene_requirements() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);
        let target_size = size(DevicePixels(5), DevicePixels(5));
        let mut empty_scene = Scene::default();
        empty_scene.finish();

        renderer.prepare_intermediate_textures(&empty_scene, target_size);
        assert!(renderer.path_intermediate_texture.is_none());
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
            content_mask: ContentMask { bounds },
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(1.0))],
            opacity: 1.0,
            ..Default::default()
        });
        filtered_scene.finish();

        renderer.prepare_intermediate_textures(&filtered_scene, target_size);
        assert!(renderer.path_intermediate_texture.is_none());
        assert!(renderer.scene_color_texture.is_some());
        assert!(renderer.blur_ping_texture.is_some());
        assert!(renderer.blur_pong_texture.is_some());
        assert!(renderer.group_textures.is_empty());
    }

    #[test]
    fn generated_sprite_shader_preserves_monochrome_coverage() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new_headless(pool);
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
            .get_or_insert_with(&key, &mut || {
                Ok(Some((tile_size, Cow::Owned(vec![64; 64]))))
            })
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
            content_mask: ContentMask { bounds },
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
        let mut renderer = MetalHeadlessRenderer::new();
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
                content_mask: ContentMask { bounds },
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
        let mut renderer = MetalHeadlessRenderer::new();
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
        let mut renderer = MetalHeadlessRenderer::new();
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
            content_mask: ContentMask { bounds: full },
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
            content_mask: ContentMask { bounds: full },
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
        let mut renderer = MetalHeadlessRenderer::new();
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
            content_mask: ContentMask { bounds: full },
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
        let mut renderer = MetalHeadlessRenderer::new();
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
            content_mask: ContentMask { bounds: full },
            background: checkerboard(hsla(0.2, 0.7, 0.5, 1.0), 2.0),
            ..Default::default()
        });
        filter.insert_primitive(BackdropFilter {
            bounds: box_bounds,
            content_mask: ContentMask { bounds: full },
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
        let mut renderer = MetalHeadlessRenderer::new();
        let key = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(0),
            frame_index: 0,
        });
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(&key, &mut || {
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
        let mut renderer = MetalHeadlessRenderer::new();
        let mut path = Path::new(gpui::point(px(1.0), px(1.0)));
        path.line_to(gpui::point(px(7.0), px(1.0)));
        path.line_to(gpui::point(px(4.0), px(7.0)));
        path.line_to(gpui::point(px(1.0), px(1.0)));
        path.content_mask = ContentMask {
            bounds: Bounds {
                origin: gpui::point(px(0.0), px(0.0)),
                size: Size {
                    width: px(8.0),
                    height: px(8.0),
                },
            },
        };
        path.color = solid_background(hsla(0.45, 0.9, 0.45, 1.0));
        let mut path_scene = Scene::default();
        path_scene.insert_primitive(path.scale(1.0));
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
        // MSAA edges must produce partially covered texels: strictly between
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
