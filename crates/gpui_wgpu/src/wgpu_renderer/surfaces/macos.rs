use super::*;
use collections::FxHashMap;
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBuffer,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidthOfPlane,
    kCVReturnSuccess,
};
use std::{
    ffi::c_void,
    ptr::{self, NonNull},
};

// Declared here rather than through objc2-core-video's binding, which would need the
// IOSurface framework crate only to name a return value this renderer uses as a cache key.
#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    fn CVPixelBufferGetIOSurface(pixel_buffer: &CVPixelBuffer) -> *const c_void;
}

pub(in crate::wgpu_renderer) struct SurfaceCache {
    texture_cache: CFRetained<CVMetalTextureCache>,
    surfaces: FxHashMap<usize, CachedSurface>,
}

impl SurfaceCache {
    pub(in crate::wgpu_renderer) fn new(device: &wgpu::Device) -> anyhow::Result<Self> {
        let hal_device = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
            .ok_or_else(|| anyhow::anyhow!("macOS WGPU device did not expose the Metal HAL"))?;
        let mut texture_cache: *mut CVMetalTextureCache = ptr::null_mut();
        // SAFETY: the HAL device is a live MTLDevice, and `texture_cache` is a valid out
        // pointer. CVMetalTextureCacheCreate retains the device for the cache's lifetime.
        let result = unsafe {
            CVMetalTextureCache::create(
                None,
                None,
                hal_device.raw_device(),
                None,
                NonNull::from(&mut texture_cache),
            )
        };
        let texture_cache = NonNull::new(texture_cache)
            .filter(|_| result == kCVReturnSuccess)
            .ok_or_else(|| {
                anyhow::anyhow!("failed to create CoreVideo Metal texture cache: code {result}")
            })?;
        // SAFETY: CVMetalTextureCacheCreate returns a +1 reference (create rule).
        let texture_cache = unsafe { CFRetained::from_raw(texture_cache) };
        Ok(Self {
            texture_cache,
            surfaces: FxHashMap::default(),
        })
    }
}

struct CachedSurface {
    _luma_texture: wgpu::Texture,
    _chroma_texture: wgpu::Texture,
    binding: SurfaceBinding,
}

pub(super) fn retain_surface_cache(renderer: &WgpuRenderer, surfaces: &[PaintSurface]) {
    let active_keys = surfaces
        .iter()
        .filter_map(|surface| {
            let gpui::SurfaceSource::Surface(image_buffer) = &surface.source else {
                return None;
            };
            core_video_surface_key(image_buffer).ok()
        })
        .collect::<smallvec::SmallVec<[usize; 4]>>();
    renderer
        .resources()
        .surface_cache
        .borrow_mut()
        .surfaces
        .retain(|key, _| active_keys.contains(key));
}

pub(super) fn draw_surfaces(
    renderer: &WgpuRenderer,
    surfaces: &[PaintSurface],
    opacities: &[f32],
    pass: &mut wgpu::RenderPass<'_>,
) -> frame::DrawResult {
    use objc2_core_video::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;

    let mut keyed_surfaces = smallvec::SmallVec::<[(&PaintSurface, usize, f32); 4]>::new();
    for (index, surface) in surfaces.iter().enumerate() {
        let gpui::SurfaceSource::Surface(image_buffer) = &surface.source else {
            log::error!("surface source cannot be imported by the macOS renderer");
            return Err(frame::DrawError::ExternalSurface);
        };
        if CVPixelBufferGetPixelFormatType(image_buffer)
            != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        {
            log::error!("unsupported CoreVideo surface pixel format");
            return Err(frame::DrawError::ExternalSurface);
        }
        keyed_surfaces.push((
            surface,
            core_video_surface_key(image_buffer)?,
            opacities.get(index).copied().unwrap_or(1.0),
        ));
    }

    let resources = renderer.resources();
    let mut cache = resources.surface_cache.borrow_mut();

    for (surface, key, opacity) in keyed_surfaces {
        let gpui::SurfaceSource::Surface(image_buffer) = &surface.source else {
            return Err(frame::DrawError::ExternalSurface);
        };
        let mut imported = cache.surfaces.remove(&key).map(Ok).unwrap_or_else(|| {
            create_core_video_surface(renderer, &cache.texture_cache, image_buffer)
        })?;
        renderer.draw_surface_binding(
            surface,
            SurfaceColorFormat::Yuv,
            opacity,
            &mut imported.binding,
            pass,
        )?;
        cache.surfaces.insert(key, imported);
    }
    Ok(())
}

fn core_video_surface_key(image_buffer: &CVPixelBuffer) -> Result<usize, frame::DrawError> {
    // SAFETY: `image_buffer` is a live CVPixelBuffer. The returned IOSurface is borrowed (get
    // rule) and only its address is kept, as the surface's identity.
    let io_surface = unsafe { CVPixelBufferGetIOSurface(image_buffer) };
    if io_surface.is_null() {
        log::error!(
            "CoreVideo surface is not IOSurface-backed; allocate it with \
             kCVPixelBufferIOSurfacePropertiesKey and \
             kCVPixelBufferMetalCompatibilityKey"
        );
        return Err(frame::DrawError::ExternalSurface);
    }
    Ok(io_surface as usize)
}

/// Wraps `CVMetalTextureCacheCreateTextureFromImage` for one plane of `image_buffer`.
fn create_plane_texture(
    texture_cache: &CVMetalTextureCache,
    image_buffer: &CVPixelBuffer,
    pixel_format: objc2_metal::MTLPixelFormat,
    plane: usize,
) -> Result<CFRetained<CVMetalTexture>, i32> {
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
    match NonNull::new(texture) {
        // SAFETY: CVMetalTextureCacheCreateTextureFromImage returns a +1 reference.
        Some(texture) if result == kCVReturnSuccess => Ok(unsafe { CFRetained::from_raw(texture) }),
        _ => Err(result),
    }
}

fn create_core_video_surface(
    renderer: &WgpuRenderer,
    texture_cache: &CVMetalTextureCache,
    image_buffer: &CVPixelBuffer,
) -> Result<CachedSurface, frame::DrawError> {
    let resources = renderer.resources();
    let luma = create_plane_texture(
        texture_cache,
        image_buffer,
        objc2_metal::MTLPixelFormat::R8Unorm,
        0,
    )
    .map_err(|error| {
        log::error!("failed to create CoreVideo luma texture: {error}");
        frame::DrawError::ExternalSurface
    })?;
    let chroma = create_plane_texture(
        texture_cache,
        image_buffer,
        objc2_metal::MTLPixelFormat::RG8Unorm,
        1,
    )
    .map_err(|error| {
        log::error!("failed to create CoreVideo chroma texture: {error}");
        frame::DrawError::ExternalSurface
    })?;
    let luma_texture = unsafe {
        import_core_video_texture(
            &resources.device,
            CVMetalTextureGetTexture(&luma),
            wgpu::TextureFormat::R8Unorm,
            plane_size(image_buffer, 0),
        )
    }
    .ok_or(frame::DrawError::ExternalSurface)?;
    let chroma_texture = unsafe {
        import_core_video_texture(
            &resources.device,
            CVMetalTextureGetTexture(&chroma),
            wgpu::TextureFormat::Rg8Unorm,
            plane_size(image_buffer, 1),
        )
    }
    .ok_or(frame::DrawError::ExternalSurface)?;
    let luma_view = luma_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let chroma_view = chroma_texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(CachedSurface {
        _luma_texture: luma_texture,
        _chroma_texture: chroma_texture,
        binding: SurfaceBinding::new(renderer, luma_view, chroma_view),
    })
}

fn plane_size(image_buffer: &CVPixelBuffer, plane: usize) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: CVPixelBufferGetWidthOfPlane(image_buffer, plane) as u32,
        height: CVPixelBufferGetHeightOfPlane(image_buffer, plane) as u32,
        depth_or_array_layers: 1,
    }
}

unsafe fn import_core_video_texture(
    device: &wgpu::Device,
    raw: Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>>,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
) -> Option<wgpu::Texture> {
    // CoreVideo returned a live MTLTexture, already retained by the binding; the retain count
    // transfers into the HAL texture so it outlives the CVMetalTexture wrapper.
    let raw = raw?;
    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            raw,
            format,
            objc2_metal::MTLTextureType::Type2D,
            1,
            1,
            size.into(),
            None,
        )
    };
    Some(unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Metal>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("core_video_surface_plane"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            // What wgpu 29 always assumed for a wrapped texture.
            wgpu::TextureUses::UNINITIALIZED,
        )
    })
}
