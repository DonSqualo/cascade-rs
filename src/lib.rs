//! cascade-rs: Pure Rust CAD kernel
//!
//! Targeting 80/20 feature parity with OpenCASCADE.

pub mod geom;
pub mod primitive;
pub mod boolean;
pub mod brep;
pub mod curve;
pub mod surface;
pub mod sweep;
pub mod loft;
pub mod modify;
pub mod fillet;
pub mod chamfer;
pub mod draft;
pub mod feature;
pub mod heal;
pub mod mesh;
pub mod io;
pub mod query;
pub mod check;
pub mod intersect;
pub mod offset;
pub mod local;

// Re-exports for convenience
pub use geom::{Pnt, Vec3, Dir};
pub use brep::{Vertex, Edge, Wire, Face, Shell, Solid, Compound, CompSolid, Shape};
pub use brep::topology;
pub use curve::{Parabola, TrimmedCurve};
pub use primitive::{make_box, make_sphere, make_cylinder, make_cone, make_torus, make_half_space, HalfSpace};
pub use boolean::{fuse, cut, common, splitter, fuse_many, cut_many, common_many};
pub use sweep::{make_revol, make_draft_prism};
pub use loft::make_loft;
pub use fillet::{make_fillet, make_fillet_variable, RadiusLaw, InterpolationMethod};
pub use chamfer::make_chamfer;
pub use draft::add_draft;
pub use feature::{make_hole, make_slot, make_rib, make_groove, circular_pattern, circular_pattern_fused};
pub use heal::{sew_faces, fix_shape};
pub use check::{check_valid, check_watertight, check_self_intersection, ShapeError};
pub use intersect::intersect_surfaces;
pub use offset::{thick_solid, make_shell};
pub use local::{split_face, split_edge, split_edge_at_point, remove_face};

/// Tolerance for geometric comparisons
pub const TOLERANCE: f64 = 1e-6;

/// Result type for cascade operations
pub type Result<T> = std::result::Result<T, CascadeError>;

#[derive(Debug, thiserror::Error)]
pub enum CascadeError {
    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),
    
    #[error("Boolean operation failed: {0}")]
    BooleanFailed(String),
    
    #[error("Topology error: {0}")]
    TopologyError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}
