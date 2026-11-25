//! Shared typed rendering and shader contracts.

pub mod artifacts;
pub mod blur;
mod instances;
mod scene_depth;
pub mod shaders;
mod target_stack;

pub use instances::InstanceRange;
pub use scene_depth::SceneDepth;
pub use target_stack::TargetStack;

#[cfg(feature = "swash")]
mod swash_glyph_rasterizer;

#[cfg(feature = "swash")]
pub use swash_glyph_rasterizer::{BitmapFallbackGlyphRasterizer, SwashGlyphRasterizer};
