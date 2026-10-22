use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use gpui_util::ResultExt;
use smallvec::SmallVec;

use image::{Delay, EncodableLayout, Frame};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt,
    hash::Hash,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
};

/// One way to store a set of assets for gpui to access.
/// Can be provided to [`AssetRegistry`] to load asset binaries on-demand.
/// Alternatively, assets can be provided pre-loaded to the registry while still using this api (though there are newer apis to do so).
///
/// Generally considered deprecated for new usages since [`AssetRegistry`] has more specific
/// support for bundled, pre-loaded, and on-demand asset loading.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;

    /// Iterates over all paths (via [`list`]) and loads each asset in turn (via [`load`]),
    /// yielding an iterator over all assets that were loaded successfully.
    /// Output can be provided directly to [`AssetRegistry::extend`].
    fn iter_preloaded(&self) -> Box<dyn Iterator<Item = (SharedString, AssetEntry)> + '_> {
        let paths = self.list("").unwrap_or_default();
        let iter = paths.into_iter().filter_map(move |path| {
            let entry = AssetEntry::from(self.load(&path).ok().flatten()?);
            Some((path, entry))
        });
        Box::new(iter)
    }

    /// Iterates over all paths (via [`list`]) and constructs an [`AssetEntry::OnDemand`],
    /// which will call [`load`] on this `Arc<AssetSource>` when the path is requested, yielding the resulting iterator.
    /// Output can be provided directly to [`AssetRegistry::extend`].
    fn iter_ondemand(self: Arc<Self>) -> Box<dyn Iterator<Item = (SharedString, AssetEntry)>> {
        use gpui_util::ResultExt;
        let paths = self.list("").unwrap_or_default();
        let iter = paths.into_iter().map(move |path| {
            // constructing OnDemand for loading assets from shared:Arc only when they are requested
            let entry = AssetEntry::from({
                let source = self.clone();
                let path = path.clone();
                move || source.load(path.as_str()).log_err().flatten()
            });
            (path, entry)
        });
        Box::new(iter)
    }
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// Alias for a Cow byte vec, representing an asset that has been loaded.
pub type AssetData = Cow<'static, [u8]>;
/// A thread-safe callback to load an asset, possibly from disk.
pub type FnAssetLoader = Arc<dyn Fn() -> Option<AssetData> + 'static + Send + Sync>;
/// The entry stored in the [`AssetRegistry`].
pub enum AssetEntry {
    /// The asset is pre-loaded. It may be stored staticlly/embedded, or has already been loaded from disk.
    PreLoaded(AssetData),
    /// The entry should be loaded on demand.
    OnDemand(FnAssetLoader),
}
impl From<AssetData> for AssetEntry {
    fn from(value: AssetData) -> Self {
        Self::PreLoaded(value)
    }
}
impl<F> From<F> for AssetEntry
where
    F: Fn() -> Option<AssetData> + 'static + Send + Sync,
{
    fn from(value: F) -> Self {
        Self::OnDemand(Arc::new(value))
    }
}

/// Caused when multiple assets are registered to the [`AssetRegistry`] with the same path.
#[derive(thiserror::Error, Debug)]
#[error("An asset with the pathkey {0:?} already exists")]
pub struct DuplicateAssetPath(SharedString);

/// Collection of assets known to the [`App`](crate::App).
///
/// Users can provide assets via [`AssetSource`] or directly via `from_iter`, `insert`, `extend_from_source`, or `extend_from_embed`.
#[derive(Default)]
pub struct AssetRegistry {
    entries: BTreeMap<SharedString, AssetEntry>,
}

impl<T: AssetSource + 'static> From<T> for AssetRegistry {
    fn from(value: T) -> Self {
        let shared: Arc<dyn AssetSource> = Arc::new(value);
        Self::from(shared)
    }
}

/// Converts the provided [`AssetSource`] into a fresh [`AssetRegistry`] where all entries in
/// the asset source are loaded [OnDemand](AssetEntry::OnDemand) via [`AssetSource::iter_ondemand`].
impl From<Arc<dyn AssetSource>> for AssetRegistry {
    fn from(value: Arc<dyn AssetSource>) -> Self {
        let mut assets = Self::default();
        let _ = assets.extend(value.iter_ondemand());
        assets
    }
}

impl<KeyType, ValueType> FromIterator<(KeyType, ValueType)> for AssetRegistry
where
    KeyType: AsRef<str>,
    ValueType: Into<AssetEntry>,
{
    fn from_iter<T: IntoIterator<Item = (KeyType, ValueType)>>(iter: T) -> Self {
        iter.into_iter()
            .fold(Self::default(), |mut assets, (key, value)| {
                assets.insert(key.as_ref(), value).log_err();
                assets
            })
    }
}

impl AssetRegistry {
    /// Inserts a single asset into the registry. The asset can be static (embedded/bundled, `&'static [u8]`), pre-loaded ([`Vec<u8>`]), or loaded-on-demand ([`FnAssetLoader`]).
    /// If there is already an entry at the provided key/path, an error is returned.
    pub fn insert(
        &mut self,
        path: impl Into<SharedString>,
        asset: impl Into<AssetEntry>,
    ) -> Result<(), DuplicateAssetPath> {
        let path = path.into();
        if self.entries.contains_key(&path) {
            return Err(DuplicateAssetPath(path));
        }
        self.entries.insert(path, asset.into());
        Ok(())
    }

    /// Returns the asset data for a given path/key. If the asset is static/embedded or pre-loaded,
    /// this will return [`Cow::Borrowed(&'this [u8])`] (the data may be 'static lifetime aka embedded/bundled or simply owned by the registry).
    /// If the asset mapped to the path is load-on-demand, the loader is queried and its result returned as [`Cow::Owned(Vec<u8>)`].
    pub fn load<'this>(&'this self, path: impl AsRef<str>) -> Option<Cow<'this, [u8]>> {
        match self.entries.get(path.as_ref())? {
            AssetEntry::PreLoaded(Cow::Borrowed(data)) => Some(Cow::Borrowed(data)),
            AssetEntry::PreLoaded(Cow::Owned(data)) => Some(Cow::Borrowed(data.as_bytes())),
            AssetEntry::OnDemand(loader) => loader(),
        }
    }

    /// Iterates over the provided and inserts each entry using the provided path in the item tuple.
    /// Errors from insert are logged, and all errors are returned as a vec.
    /// Returns OK if no duplicate paths are found (can occur if the iterator provided a path
    /// that was already in the collection by the time that entry is inserted).
    pub fn extend<IterType, KeyType, EntryType>(
        &mut self,
        into_iter: IterType,
    ) -> Result<(), Vec<DuplicateAssetPath>>
    where
        IterType: IntoIterator<Item = (KeyType, EntryType)>,
        KeyType: AsRef<str>,
        EntryType: Into<AssetEntry>,
    {
        let mut result = Ok(());
        for (path_key, into_entry) in into_iter.into_iter() {
            if let Err(err) = self.insert(path_key.as_ref(), into_entry.into()) {
                gpui_util::log_err(&err);
                result = Err({
                    let mut errs: Vec<DuplicateAssetPath> = result.err().unwrap_or_default();
                    errs.push(err);
                    errs
                });
            }
        }
        result
    }

    /// Convenience method to create an asset iterator over paths and asset-data for
    /// [`RustEmbed`](rust_embed::RustEmbed), since the crate only provides an iterator over file names.
    ///
    /// Can be provided as the input to [`extend`].
    #[cfg(feature = "embedded-assets")]
    pub fn iter_embed<T: rust_embed::RustEmbed>()
    -> impl Iterator<Item = (Cow<'static, str>, AssetEntry)> + 'static {
        T::iter().filter_map(|path| {
            let file = T::get(path.as_ref())?;
            Some((path, AssetEntry::PreLoaded(file.data)))
        })
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
    /// How many times the frame was halved before upload; `0` is the frame as decoded.
    ///
    /// The atlas has no mipmaps and samples bilinearly, so drawing an image far below its
    /// pixel size skips most of its pixels and aliases. `Window::paint_image` uploads a
    /// box-filtered copy instead (see [`image_downscale_level`]).
    pub downscale_level: u8,
}

/// The number of halvings that bring an image of `source` pixels to within 2x of the
/// `target_width` x `target_height` device pixels it is drawn at, without dropping below them.
///
/// Bilinear sampling reads 2x2 texels per pixel, so it only averages every source pixel
/// while the source is at most twice the drawn size. A whole number of halvings (like a mip
/// chain) keeps the variants of one image down to a handful, however the draw size changes
/// during a resize or animation. The smaller of the two axis ratios decides, so neither
/// axis is ever magnified.
pub(crate) fn image_downscale_level(
    source: Size<DevicePixels>,
    target_width: f32,
    target_height: f32,
) -> u8 {
    if !(target_width > 0.0 && target_height > 0.0) {
        return 0;
    }
    let ratio = (source.width.0 as f32 / target_width).min(source.height.0 as f32 / target_height);
    if ratio < 2.0 {
        return 0;
    }
    (ratio.log2().floor() as u8).min(max_image_downscale_level(source))
}

/// The most halvings `source` allows before one side reaches a single pixel.
pub(crate) fn max_image_downscale_level(source: Size<DevicePixels>) -> u8 {
    let longest = source.width.0.max(source.height.0).max(1) as u32;
    longest.ilog2() as u8
}

/// The size of `source` after `level` halvings, rounding up so no side reaches zero.
pub(crate) fn downscaled_image_size(source: Size<DevicePixels>, level: u8) -> Size<DevicePixels> {
    let halve = |side: i32| (0..level).fold(side.max(1) as u32, |side, _| side.div_ceil(2)) as i32;
    size(
        DevicePixels(halve(source.width.0)),
        DevicePixels(halve(source.height.0)),
    )
}

/// Halves straight-alpha BGRA pixels `level` times with a 2x2 box filter.
///
/// Colors are weighted by alpha, so the color of a transparent pixel (often black) does not
/// darken the edge of an opaque one. An odd last row or column averages only the pixels it has.
pub(crate) fn downscale_bgra(bytes: &[u8], source: Size<DevicePixels>, level: u8) -> Vec<u8> {
    let mut width = source.width.0.max(0) as usize;
    let mut height = source.height.0.max(0) as usize;
    let mut pixels = Cow::Borrowed(bytes);
    for _ in 0..level {
        if width == 0 || height == 0 || (width <= 1 && height <= 1) {
            break;
        }
        let half_width = width.div_ceil(2);
        let half_height = height.div_ceil(2);
        let mut half = vec![0u8; half_width * half_height * 4];
        for (y, out_row) in half.chunks_exact_mut(half_width * 4).enumerate() {
            let out_row = out_row.as_chunks_mut::<4>().0;
            let row = |row: usize| pixels[row * width * 4..][..width * 4].as_chunks::<4>().0;
            let (top_pairs, top_rest) = row(2 * y).as_chunks::<2>();
            if 2 * y + 1 < height {
                let (bottom_pairs, bottom_rest) = row(2 * y + 1).as_chunks::<2>();
                for ((out, [a, b]), [c, d]) in out_row.iter_mut().zip(top_pairs).zip(bottom_pairs) {
                    *out = average_bgra([a, b, c, d]);
                }
                if let ([a], [c]) = (top_rest, bottom_rest) {
                    out_row[half_width - 1] = average_bgra([a, c]);
                }
            } else {
                for (out, [a, b]) in out_row.iter_mut().zip(top_pairs) {
                    *out = average_bgra([a, b]);
                }
                if let [a] = top_rest {
                    out_row[half_width - 1] = average_bgra([a]);
                }
            }
        }
        pixels = Cow::Owned(half);
        width = half_width;
        height = half_height;
    }
    pixels.into_owned()
}

/// The alpha-weighted average of `N` straight-alpha BGRA pixels, rounded to nearest.
///
/// A fixed `N` lets the compiler unroll the sums. Fully opaque blocks, the common case for
/// photos and avatars, skip the per-channel division by total alpha.
#[inline(always)]
fn average_bgra<const N: usize>(pixels: [&[u8; 4]; N]) -> [u8; 4] {
    let count = N as u32;
    let alpha: u32 = pixels.iter().map(|pixel| u32::from(pixel[3])).sum();
    let mut out = [0u8; 4];
    if alpha == 255 * count {
        for (channel, out) in out[..3].iter_mut().enumerate() {
            let sum: u32 = pixels.iter().map(|pixel| u32::from(pixel[channel])).sum();
            *out = ((sum + count / 2) / count) as u8;
        }
    } else {
        for (channel, out) in out[..3].iter_mut().enumerate() {
            let sum: u32 = pixels
                .iter()
                .map(|pixel| u32::from(pixel[channel]) * u32::from(pixel[3]))
                .sum();
            *out = (sum + alpha / 2).checked_div(alpha).unwrap_or(0) as u8;
        }
    }
    out[3] = ((alpha + count / 2) / count) as u8;
    out
}

/// A cached and processed image, in BGRA format
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: data.into(),
        }
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        self.data
            .get(frame_index)
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        self.data
            .get(frame_index)
            .map(|frame| {
                let (width, height) = frame.buffer().dimensions();
                size(width.into(), height.into())
            })
            .unwrap_or_default()
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.data
            .get(frame_index)
            .map(|frame| frame.delay())
            .unwrap_or(Delay::from_numer_denom_ms(100, 1))
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &self.data.first().map(|f| f.buffer().dimensions()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn empty_render_image_does_not_panic() {
        let image = RenderImage::new(SmallVec::new());
        assert_eq!(image.frame_count(), 0);
        assert_eq!(image.size(0), Size::default());
        assert_eq!(image.as_bytes(0), None);
        assert_eq!(image.render_size(0), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(100, 1));
        let _ = format!("{image:?}");
    }

    fn example_assets() -> Vec<(&'static str, AssetData)> {
        vec![
            ("asset0", Cow::Owned(vec![0u8])),
            ("asset1", Cow::Owned(vec![1u8])),
            ("asset2", Cow::Owned(vec![2u8])),
            ("asset3", Cow::Owned(vec![3u8])),
        ]
    }

    fn test_load_from(registry: &AssetRegistry) {
        for (key, value) in example_assets() {
            let loaded = registry.load(key).map(|cow| cow.to_vec());
            assert_eq!(loaded, Some(value.to_vec()));
        }
    }

    struct StubAssetSource;
    impl AssetSource for StubAssetSource {
        fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
            for (key, data) in example_assets() {
                if key == path {
                    return Ok(Some(data));
                }
            }
            unimplemented!()
        }

        fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
            Ok(example_assets()
                .into_iter()
                .map(|(key, _)| key.into())
                .collect())
        }
    }

    #[test]
    fn test_from_assetsource() {
        let assets = AssetRegistry::from(StubAssetSource);
        test_load_from(&assets);
    }

    #[test]
    fn test_from_arc_assetsource() {
        let arc: Arc<dyn AssetSource> = Arc::new(StubAssetSource);
        let assets = AssetRegistry::from(arc);
        test_load_from(&assets);
    }

    #[test]
    fn test_fromiter_assetdata() {
        let assets = AssetRegistry::from_iter(example_assets());
        test_load_from(&assets);
    }

    #[test]
    fn test_fromiter_assetentry() {
        let assets = AssetRegistry::from_iter(
            example_assets()
                .into_iter()
                .map(|(key, value)| (key, AssetEntry::from(value)))
                .chain(std::iter::once((
                    "asset6",
                    AssetEntry::OnDemand(Arc::new(|| Some(AssetData::Owned(vec![6u8])))),
                ))),
        );
        test_load_from(&assets);
        assert_eq!(
            assets.load("asset6").map(|cow| cow.to_vec()),
            Some(vec![6u8])
        );
    }

    #[test]
    fn test_insert() {
        let mut assets = AssetRegistry::default();
        for (key, value) in example_assets() {
            let result = assets.insert(key, value);
            assert!(result.is_ok());
        }
        test_load_from(&assets);
    }

    #[test]
    fn test_extend() {
        let mut assets = AssetRegistry::default();
        let result = assets.extend(
            example_assets()
                .into_iter()
                .map(|(key, value)| (key, AssetEntry::PreLoaded(value))),
        );
        assert!(result.is_ok());
        test_load_from(&assets);
    }

    #[test]
    fn test_extend_assetsource_preloaded() {
        let mut assets = AssetRegistry::default();
        let result = assets.extend(StubAssetSource.iter_preloaded());
        assert!(result.is_ok());
        test_load_from(&assets);
    }

    #[test]
    fn test_extend_assetsource_ondemand() {
        let mut assets = AssetRegistry::default();
        let result = assets.extend(Arc::new(StubAssetSource).iter_ondemand());
        assert!(result.is_ok());
        test_load_from(&assets);
    }

    fn device_size(width: i32, height: i32) -> Size<DevicePixels> {
        size(DevicePixels(width), DevicePixels(height))
    }

    #[test]
    fn downscale_level_brings_the_source_within_twice_the_draw_size() {
        let badge = device_size(512, 512);
        // 32px at 100%, 125%, 150% and 200% scale.
        assert_eq!(image_downscale_level(badge, 32.0, 32.0), 4);
        assert_eq!(image_downscale_level(badge, 40.0, 40.0), 3);
        assert_eq!(image_downscale_level(badge, 48.0, 48.0), 3);
        assert_eq!(image_downscale_level(badge, 64.0, 64.0), 3);
        for target in [24.0_f32, 32.0, 40.0, 48.0, 64.0, 100.0, 255.0] {
            let level = image_downscale_level(badge, target, target);
            let side = downscaled_image_size(badge, level).width.0 as f32;
            assert!(
                side >= target && side < 2.0 * target,
                "{target}px: level {level} gives {side}px"
            );
        }
    }

    #[test]
    fn downscale_level_leaves_images_at_or_near_their_size_alone() {
        let image = device_size(200, 100);
        assert_eq!(image_downscale_level(image, 200.0, 100.0), 0);
        assert_eq!(image_downscale_level(image, 101.0, 51.0), 0);
        assert_eq!(image_downscale_level(image, 400.0, 200.0), 0);
        assert_eq!(image_downscale_level(image, 0.0, 0.0), 0);
        assert_eq!(image_downscale_level(image, f32::NAN, 10.0), 0);
        assert_eq!(image_downscale_level(device_size(0, 0), 10.0, 10.0), 0);
    }

    #[test]
    fn downscale_level_follows_the_less_shrunk_axis() {
        // Squashed to a quarter of the width but only half the height: one halving, so the
        // height is never magnified.
        assert_eq!(
            image_downscale_level(device_size(400, 400), 100.0, 200.0),
            1
        );
    }

    #[test]
    fn downscale_level_stops_at_one_pixel() {
        let image = device_size(8, 2);
        assert_eq!(max_image_downscale_level(image), 3);
        assert_eq!(image_downscale_level(image, 0.01, 0.01), 3);
        assert_eq!(downscaled_image_size(image, 3), device_size(1, 1));
    }

    #[test]
    fn downscaled_size_rounds_odd_sides_up() {
        assert_eq!(
            downscaled_image_size(device_size(5, 3), 1),
            device_size(3, 2)
        );
        assert_eq!(
            downscaled_image_size(device_size(5, 3), 2),
            device_size(2, 1)
        );
    }

    #[test]
    fn downscale_averages_every_source_pixel() {
        // A 1px checkerboard aliases to a solid color under point or bilinear sampling at
        // 4:1; a box filter gives the true average grey.
        let mut bytes = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let value = if (x + y) % 2 == 0 { 255 } else { 0 };
                bytes.extend_from_slice(&[value, value, value, 255]);
            }
        }
        let out = downscale_bgra(&bytes, device_size(4, 4), 2);
        assert_eq!(out, vec![128, 128, 128, 255]);
    }

    #[test]
    fn downscale_weights_color_by_alpha() {
        // An opaque red pixel next to transparent black ones stays red, at a quarter of the
        // alpha, instead of darkening toward black. Pixels are BGRA.
        let bytes = [
            0, 0, 255, 255, //
            0, 0, 0, 0, //
            0, 0, 0, 0, //
            0, 0, 0, 0,
        ];
        assert_eq!(
            downscale_bgra(&bytes, device_size(2, 2), 1),
            vec![0, 0, 255, 64]
        );
        assert_eq!(
            downscale_bgra(&[0; 16], device_size(2, 2), 1),
            vec![0, 0, 0, 0]
        );
    }

    #[test]
    fn downscale_averages_only_the_pixels_an_odd_edge_has() {
        // 3x1: the second output pixel covers one source pixel, not a phantom black one.
        let bytes = [
            10, 20, 30, 255, //
            30, 40, 50, 255, //
            200, 100, 50, 255,
        ];
        assert_eq!(
            downscale_bgra(&bytes, device_size(3, 1), 1),
            vec![20, 30, 40, 255, 200, 100, 50, 255]
        );
    }
}
