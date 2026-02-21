//! Geometric primitives package.
//!
//! Port of OCCT's gp package.
//! Source: src/FoundationClasses/TKMath/gp/
//!
//! This is THE foundation - everything in OCCT uses these types.

mod xyz;
mod pnt;
mod vec;
mod dir;
mod mat;
mod trsf;
mod ax1;
mod ax2;
mod ax3;

// Re-export all types at module level (matching OCCT's flat namespace)
pub use xyz::XYZ;
pub use pnt::Pnt;
pub use vec::Vec3;
pub use dir::Dir;
pub use mat::Mat;
pub use trsf::{Trsf, TrsfForm};
pub use ax1::Ax1;
pub use ax2::Ax2;
pub use ax3::Ax3;

// 2D types - to be added
// pub use pnt2d::Pnt2d;
// pub use vec2d::Vec2d;
// pub use dir2d::Dir2d;
// etc.
