//! BREP (Boundary Representation) data structures

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

/// A vertex in 3D space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub point: [f64; 3],
}

impl Vertex {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { point: [x, y, z] }
    }
    
    pub fn as_point(&self) -> Point3<f64> {
        Point3::new(self.point[0], self.point[1], self.point[2])
    }
}

/// An edge connecting vertices, with curve geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub start: Vertex,
    pub end: Vertex,
    pub curve_type: CurveType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurveType {
    Line,
    Arc { center: [f64; 3], radius: f64 },
    BSpline { control_points: Vec<[f64; 3]>, knots: Vec<f64>, degree: usize },
}

/// A wire is a connected sequence of edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wire {
    pub edges: Vec<Edge>,
    pub closed: bool,
}

/// A face bounded by wires
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    pub outer_wire: Wire,
    pub inner_wires: Vec<Wire>,  // holes
    pub surface_type: SurfaceType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurfaceType {
    Plane { origin: [f64; 3], normal: [f64; 3] },
    Cylinder { origin: [f64; 3], axis: [f64; 3], radius: f64 },
    Sphere { center: [f64; 3], radius: f64 },
    Cone { origin: [f64; 3], axis: [f64; 3], half_angle_rad: f64 },
    Torus { center: [f64; 3], major_radius: f64, minor_radius: f64 },
    BSpline { /* TODO */ },
}

/// A shell is a connected set of faces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shell {
    pub faces: Vec<Face>,
    pub closed: bool,
}

/// A solid bounded by shells
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solid {
    pub outer_shell: Shell,
    pub inner_shells: Vec<Shell>,  // voids
}

/// A compound of multiple shapes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compound {
    pub solids: Vec<Solid>,
}

/// Generic shape enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Shape {
    Vertex(Vertex),
    Edge(Edge),
    Wire(Wire),
    Face(Face),
    Shell(Shell),
    Solid(Solid),
    Compound(Compound),
}

impl Shape {
    /// Get bounding box
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        // TODO: Implement properly
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }
    
    /// Check if shape is valid
    pub fn is_valid(&self) -> bool {
        // TODO: Implement topology checks
        true
    }
}
