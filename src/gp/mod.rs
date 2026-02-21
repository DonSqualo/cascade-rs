//! Geometric primitives package.
//!
//! Port of OCCT's gp package.
//! Source: src/FoundationClasses/TKMath/gp/
//!
//! This is THE foundation - everything in OCCT uses these types.

// 3D Core types (complete)
mod xyz;
mod pnt;
mod vec;
mod dir;
mod mat;
mod trsf;
mod gtrsf;
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
pub use gtrsf::GTrsf;
pub use ax1::Ax1;
pub use ax2::Ax2;
pub use ax3::Ax3;

// Sub-agents are porting these - not yet integrated:
// - pln, lin, circ, elips, hypr, parab, cylinder, cone, sphere, torus
// - xy, pnt2d, vec2d, dir2d, mat2d, trsf2d, ax2d, ax22d
// - lin2d, circ2d, elips2d, hypr2d, parab2d
