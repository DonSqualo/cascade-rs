//! cascade-rs: Pure Rust CAD kernel
//!
//! Targeting 80/20 feature parity with OpenCASCADE.

pub mod primitive;
pub mod boolean;
pub mod brep;
pub mod curve;
pub mod surface;
pub mod modify;
pub mod fillet;
pub mod mesh;
pub mod io;
pub mod query;

// Re-exports for convenience
pub use brep::{Vertex, Edge, Wire, Face, Shell, Solid, Compound, Shape};
pub use brep::topology;
pub use primitive::{make_box, make_sphere, make_cylinder, make_cone, make_torus};
pub use boolean::{fuse, cut, common};
pub use fillet::make_fillet;

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
