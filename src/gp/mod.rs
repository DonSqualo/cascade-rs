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

// 3D Geometric types (in progress)
mod pln;
mod lin;
mod circ;
mod elips;
mod hypr;
mod parab;
mod cylinder;
mod cone;
mod sphere;
mod torus;

// 2D types
mod xy;
mod pnt2d;
mod vec2d;
mod dir2d;
mod mat2d;
mod trsf2d;
mod ax2d;
mod ax22d;
mod lin2d;
mod circ2d;
mod elips2d;
mod hypr2d;
mod parab2d;

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

// 3D Geometric types
pub use pln::Pln;
pub use lin::Lin;
pub use circ::Circ;
pub use elips::Elips;
pub use hypr::Hypr;
pub use parab::Parab;
pub use cylinder::Cylinder;
pub use cone::Cone;
pub use sphere::Sphere;
pub use torus::Torus;

// 2D types
pub use xy::XY;
pub use pnt2d::Pnt2d;
pub use vec2d::Vec2d;
pub use dir2d::Dir2d;
pub use mat2d::Mat2d;
pub use trsf2d::Trsf2d;
pub use ax2d::Ax2d;
pub use ax22d::Ax22d;
pub use lin2d::Lin2d;
pub use circ2d::Circ2d;
pub use elips2d::Elips2d;
pub use hypr2d::Hypr2d;
pub use parab2d::Parab2d;
