//! Tessellation and meshing

use crate::brep::Solid;
use crate::{Result, CascadeError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleMesh {
    pub vertices: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

pub fn triangulate(solid: &Solid, tolerance: f64) -> Result<TriangleMesh> {
    Err(CascadeError::NotImplemented("mesh::triangulate".into()))
}

pub fn export_stl(mesh: &TriangleMesh, path: &str) -> Result<()> {
    Err(CascadeError::NotImplemented("mesh::export_stl".into()))
}
