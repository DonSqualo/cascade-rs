//! Bounding box classes from OpenCASCADE
//! Provides various bounding box implementations for 2D and 3D geometry

pub mod box_3d;
pub mod box_2d;
pub mod sphere;
pub mod range;
pub mod obb;
pub mod b_generic;

pub use self::box_3d::BndBox;
pub use self::box_2d::BndBox2d;
pub use self::sphere::BndSphere;
pub use self::range::BndRange;
pub use self::obb::BndOBB;
pub use self::b_generic::{BndB2d, BndB2f, BndB3d, BndB3f};

/// Precision value for comparisons (matching OCCT Precision::Confusion)
pub const PRECISION_CONFUSION: f64 = 1e-7;

/// Maximum floating point value
pub const REAL_MAX: f64 = 1.7976931348623157e+308;

/// Minimum floating point value
pub const REAL_MIN: f64 = -1.7976931348623157e+308;
