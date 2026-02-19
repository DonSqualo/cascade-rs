//! BREP (Boundary Representation) data structures

pub mod topology;

use nalgebra::Point3;
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
    Ellipse { center: [f64; 3], major_axis: [f64; 3], minor_axis: [f64; 3] },
    Bezier { control_points: Vec<[f64; 3]> },
    BSpline { control_points: Vec<[f64; 3]>, knots: Vec<f64>, degree: usize, weights: Option<Vec<f64>> },
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
    BSpline {
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        control_points: Vec<Vec<[f64; 3]>>,  // 2D grid
        weights: Option<Vec<Vec<f64>>>,      // Optional NURBS weights
    },
}

impl SurfaceType {
    /// Evaluate point on surface at parameter (u, v)
    /// Returns 3D point on the surface
    pub fn point_at(&self, u: f64, v: f64) -> [f64; 3] {
        match self {
            SurfaceType::Plane { origin, normal: _ } => {
                // For a plane, we can't really evaluate using u,v without defining a coordinate system
                // This is a simplified fallback
                *origin
            }
            SurfaceType::Cylinder { origin, axis, radius } => {
                let axis_norm = normalize(axis);
                let perp1 = perpendicular_to(&axis_norm);
                let perp2 = cross(&axis_norm, &perp1);
                
                let z = v; // v parameter along the axis
                let angle = u * 2.0 * std::f64::consts::PI;
                
                let x = origin[0] + angle.cos() * radius * perp1[0] + angle.sin() * radius * perp2[0] + z * axis_norm[0];
                let y = origin[1] + angle.cos() * radius * perp1[1] + angle.sin() * radius * perp2[1] + z * axis_norm[1];
                let z_coord = origin[2] + angle.cos() * radius * perp1[2] + angle.sin() * radius * perp2[2] + z * axis_norm[2];
                
                [x, y, z_coord]
            }
            SurfaceType::Sphere { center, radius } => {
                let lat = -std::f64::consts::PI / 2.0 + v * std::f64::consts::PI;
                let lon = u * 2.0 * std::f64::consts::PI;
                
                let x = center[0] + radius * lat.cos() * lon.cos();
                let y = center[1] + radius * lat.cos() * lon.sin();
                let z = center[2] + radius * lat.sin();
                
                [x, y, z]
            }
            SurfaceType::Cone { origin, axis, half_angle_rad } => {
                let axis_norm = normalize(axis);
                let perp1 = perpendicular_to(&axis_norm);
                let perp2 = cross(&axis_norm, &perp1);
                
                let h = v;
                let radius = h * half_angle_rad.tan();
                let angle = u * 2.0 * std::f64::consts::PI;
                
                let x = origin[0] + angle.cos() * radius * perp1[0] + angle.sin() * radius * perp2[0] + h * axis_norm[0];
                let y = origin[1] + angle.cos() * radius * perp1[1] + angle.sin() * radius * perp2[1] + h * axis_norm[1];
                let z = origin[2] + angle.cos() * radius * perp1[2] + angle.sin() * radius * perp2[2] + h * axis_norm[2];
                
                [x, y, z]
            }
            SurfaceType::Torus { center, major_radius: _, minor_radius: _ } => {
                let _u_angle = u * 2.0 * std::f64::consts::PI;
                let _v_angle = v * 2.0 * std::f64::consts::PI;
                
                // Return center as placeholder - full implementation TODO
                *center
            }
            SurfaceType::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                control_points,
                weights,
            } => {
                evaluate_bspline_surface(u, v, *u_degree, *v_degree, u_knots, v_knots, control_points, weights.as_ref())
            }
        }
    }
    
    /// Evaluate normal on surface at parameter (u, v)
    pub fn normal_at(&self, u: f64, v: f64) -> [f64; 3] {
        match self {
            SurfaceType::Plane { normal, .. } => {
                normalize(normal)
            }
            SurfaceType::Cylinder { origin: _, axis, radius: _ } => {
                let axis_norm = normalize(axis);
                let perp1 = perpendicular_to(&axis_norm);
                let perp2 = cross(&axis_norm, &perp1);
                
                let angle = u * 2.0 * std::f64::consts::PI;
                let normal = [
                    angle.cos() * perp1[0] + angle.sin() * perp2[0],
                    angle.cos() * perp1[1] + angle.sin() * perp2[1],
                    angle.cos() * perp1[2] + angle.sin() * perp2[2],
                ];
                normalize(&normal)
            }
            SurfaceType::Sphere { center, .. } => {
                let lat = -std::f64::consts::PI / 2.0 + v * std::f64::consts::PI;
                let lon = u * 2.0 * std::f64::consts::PI;
                
                let pt = self.point_at(u, v);
                let normal = [
                    pt[0] - center[0],
                    pt[1] - center[1],
                    pt[2] - center[2],
                ];
                normalize(&normal)
            }
            SurfaceType::Cone { origin, axis, half_angle_rad } => {
                let axis_norm = normalize(axis);
                let perp1 = perpendicular_to(&axis_norm);
                let perp2 = cross(&axis_norm, &perp1);
                
                let angle = u * 2.0 * std::f64::consts::PI;
                let radial = [
                    angle.cos() * perp1[0] + angle.sin() * perp2[0],
                    angle.cos() * perp1[1] + angle.sin() * perp2[1],
                    angle.cos() * perp1[2] + angle.sin() * perp2[2],
                ];
                
                let cos_angle = half_angle_rad.cos();
                let sin_angle = half_angle_rad.sin();
                let normal = [
                    radial[0] * cos_angle - axis_norm[0] * sin_angle,
                    radial[1] * cos_angle - axis_norm[1] * sin_angle,
                    radial[2] * cos_angle - axis_norm[2] * sin_angle,
                ];
                normalize(&normal)
            }
            SurfaceType::Torus { center, major_radius, minor_radius } => {
                let u_angle = u * 2.0 * std::f64::consts::PI;
                let v_angle = v * 2.0 * std::f64::consts::PI;
                
                // Normal points outward from torus surface
                let normal = [
                    (major_radius + minor_radius * v_angle.cos()) * u_angle.cos(),
                    (major_radius + minor_radius * v_angle.cos()) * u_angle.sin(),
                    minor_radius * v_angle.sin(),
                ];
                normalize(&normal)
            }
            SurfaceType::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                control_points,
                weights,
            } => {
                bspline_surface_normal(u, v, *u_degree, *v_degree, u_knots, v_knots, control_points, weights.as_ref())
            }
        }
    }
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

// ===== BSpline Surface Functions =====

/// Normalize a 3D vector
fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Find a vector perpendicular to the given vector
fn perpendicular_to(v: &[f64; 3]) -> [f64; 3] {
    let abs_x = v[0].abs();
    let abs_y = v[1].abs();
    let abs_z = v[2].abs();
    
    let perp = if abs_x <= abs_y && abs_x <= abs_z {
        [1.0, 0.0, 0.0]
    } else if abs_y <= abs_x && abs_y <= abs_z {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    
    normalize(&cross(v, &perp))
}

/// Cross product
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Evaluate a univariate B-spline basis function
/// Returns N_{i,p}(t) where i is the knot span index, p is the degree
fn bspline_basis(t: f64, i: usize, p: usize, knots: &[f64]) -> f64 {
    if p == 0 {
        // Base case: 0-degree basis function is 1 if t is in [knots[i], knots[i+1]), 0 otherwise
        if i < knots.len() - 1 {
            if t >= knots[i] && t < knots[i + 1] {
                return 1.0;
            }
            // Handle the right endpoint
            if (t - knots[i + 1]).abs() < 1e-10 && i == knots.len() - 2 {
                return 1.0;
            }
        }
        0.0
    } else {
        // Recursive case
        let left = if knots[i + p] - knots[i] > 1e-10 {
            ((t - knots[i]) / (knots[i + p] - knots[i])) * bspline_basis(t, i, p - 1, knots)
        } else {
            0.0
        };
        
        let right = if knots[i + p + 1] - knots[i + 1] > 1e-10 {
            ((knots[i + p + 1] - t) / (knots[i + p + 1] - knots[i + 1])) * bspline_basis(t, i + 1, p - 1, knots)
        } else {
            0.0
        };
        
        left + right
    }
}

/// Find the knot span index for parameter t
fn find_knot_span(t: f64, knots: &[f64]) -> usize {
    let n = knots.len() - 1; // Number of basis functions - 1
    if t >= knots[n] {
        return n - 1;
    }
    
    let mut low = 0;
    let mut high = n;
    let mut mid = (low + high) / 2;
    
    while t < knots[mid] || t >= knots[mid + 1] {
        if t < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    
    mid
}

/// Evaluate a BSpline surface at parameters (u, v)
fn evaluate_bspline_surface(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    control_points: &[Vec<[f64; 3]>],
    weights: Option<&Vec<Vec<f64>>>,
) -> [f64; 3] {
    // Clamp parameters to knot range
    let u_clamped = u.max(*u_knots.first().unwrap_or(&0.0)).min(*u_knots.last().unwrap_or(&1.0));
    let v_clamped = v.max(*v_knots.first().unwrap_or(&0.0)).min(*v_knots.last().unwrap_or(&1.0));
    
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;
    let mut weight_sum = 0.0;
    
    // Sum over all control points
    for i in 0..control_points.len() {
        let n_u = bspline_basis(u_clamped, i, u_degree, u_knots);
        
        if n_u.abs() < 1e-10 {
            continue;
        }
        
        for j in 0..control_points[i].len() {
            let n_v = bspline_basis(v_clamped, j, v_degree, v_knots);
            
            if n_v.abs() < 1e-10 {
                continue;
            }
            
            let basis = n_u * n_v;
            let pt = control_points[i][j];
            
            // Apply weight if NURBS
            let w = if let Some(w_grid) = weights {
                if i < w_grid.len() && j < w_grid[i].len() {
                    w_grid[i][j]
                } else {
                    1.0
                }
            } else {
                1.0
            };
            
            let weighted_basis = basis * w;
            x += pt[0] * weighted_basis;
            y += pt[1] * weighted_basis;
            z += pt[2] * weighted_basis;
            weight_sum += weighted_basis;
        }
    }
    
    // Normalize by weights (for NURBS)
    if weight_sum.abs() > 1e-10 {
        [x / weight_sum, y / weight_sum, z / weight_sum]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Evaluate surface normal using finite differences
fn bspline_surface_normal(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    control_points: &[Vec<[f64; 3]>],
    weights: Option<&Vec<Vec<f64>>>,
) -> [f64; 3] {
    let delta = 1e-5;
    
    // Compute tangent vectors using finite differences
    let pt_u_plus = evaluate_bspline_surface(u + delta, v, u_degree, v_degree, u_knots, v_knots, control_points, weights);
    let pt_u_minus = evaluate_bspline_surface(u - delta, v, u_degree, v_degree, u_knots, v_knots, control_points, weights);
    let tangent_u = [
        pt_u_plus[0] - pt_u_minus[0],
        pt_u_plus[1] - pt_u_minus[1],
        pt_u_plus[2] - pt_u_minus[2],
    ];
    
    let pt_v_plus = evaluate_bspline_surface(u, v + delta, u_degree, v_degree, u_knots, v_knots, control_points, weights);
    let pt_v_minus = evaluate_bspline_surface(u, v - delta, u_degree, v_degree, u_knots, v_knots, control_points, weights);
    let tangent_v = [
        pt_v_plus[0] - pt_v_minus[0],
        pt_v_plus[1] - pt_v_minus[1],
        pt_v_plus[2] - pt_v_minus[2],
    ];
    
    // Normal is cross product of tangent vectors
    let normal = cross(&tangent_u, &tangent_v);
    normalize(&normal)
}
