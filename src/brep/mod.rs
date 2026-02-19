//! BREP (Boundary Representation) data structures

pub mod topology;

use nalgebra::Point3;
use serde::{Deserialize, Serialize};
use crate::{Result, CascadeError};

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
    /// Parabola defined by position (axis system) and focal parameter
    /// 
    /// The parabola is defined parametrically as:
    /// P(u) = origin + u * x_dir + (u² / (4 * focal)) * y_dir
    /// 
    /// - origin: vertex of the parabola (also the position origin)
    /// - x_dir: direction along the parabola axis (from vertex toward infinity)
    /// - y_dir: direction perpendicular to axis (in the plane of the parabola)
    /// - focal: focal parameter (distance from vertex to focus = focal/2)
    Parabola { 
        origin: [f64; 3], 
        x_dir: [f64; 3], 
        y_dir: [f64; 3], 
        focal: f64 
    },
    /// Hyperbola defined by center, axes, and semi-axis lengths
    /// 
    /// The hyperbola is defined parametrically as:
    /// P(u) = center + a*cosh(u)*x_dir + b*sinh(u)*y_dir
    /// 
    /// - center: center of the hyperbola
    /// - x_dir: direction along the transverse axis (toward vertices)
    /// - y_dir: direction along the conjugate axis
    /// - major_radius: semi-major axis length (a)
    /// - minor_radius: semi-minor axis length (b)
    /// 
    /// Properties:
    /// - Foci: center ± c*x_dir where c = sqrt(a² + b²)
    /// - Eccentricity: e = c / a
    /// - Asymptotes: direction vectors at ±(b/a) slope relative to x_dir
    Hyperbola {
        center: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        major_radius: f64,
        minor_radius: f64,
    },
    Bezier { control_points: Vec<[f64; 3]> },
    BSpline { control_points: Vec<[f64; 3]>, knots: Vec<f64>, degree: usize, weights: Option<Vec<f64>> },
    /// A bounded portion of an underlying curve defined by parameter limits
    /// 
    /// The trimmed curve restricts evaluation to the parameter range [u1, u2]
    /// of the basis curve. Useful for representing partial curves.
    Trimmed { 
        /// The underlying curve being trimmed
        basis_curve: Box<CurveType>, 
        /// First parameter (start of trimmed portion)
        u1: f64, 
        /// Second parameter (end of trimmed portion)
        u2: f64 
    },
    /// An offset curve is a curve displaced by a constant distance in a given direction
    /// 
    /// For 2D curves: the offset is perpendicular to the curve
    /// For 3D curves: the offset lies in the plane defined by the curve tangent and a reference direction
    /// 
    /// The basis curve is offset by moving each point along the normal direction.
    Offset {
        /// The underlying basis curve
        basis_curve: Box<CurveType>,
        /// The distance to offset (positive or negative)
        offset_distance: f64,
        /// Reference direction for 3D offset (defines the plane with the tangent)
        /// For 2D curves, this can be the Z-axis or any direction; the perpendicular direction is computed automatically
        offset_direction: [f64; 3],
    },
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
    /// Bezier surface defined by a 2D grid of control points
    /// 
    /// The surface is evaluated using de Casteljau's algorithm:
    /// S(u, v) = Σ Σ B_{i,u_degree}(u) * B_{j,v_degree}(v) * P_{i,j}
    /// 
    /// where B are Bernstein polynomials and P_{i,j} are control points
    BezierSurface {
        /// 2D grid of control points [i][j]
        control_points: Vec<Vec<[f64; 3]>>,
        /// Degree in U direction (= number of control points in U - 1)
        u_degree: usize,
        /// Degree in V direction (= number of control points in V - 1)
        v_degree: usize,
    },
    BSpline {
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        control_points: Vec<Vec<[f64; 3]>>,  // 2D grid
        weights: Option<Vec<Vec<f64>>>,      // Optional NURBS weights
    },
    /// Surface of revolution - created by rotating a curve around an axis
    /// u parameter: curve parameter (0 to 1)
    /// v parameter: rotation angle (0 to 2π for full revolution)
    SurfaceOfRevolution {
        /// The generatrix curve being revolved
        basis_curve: CurveType,
        /// Start point of the basis curve (needed for Line evaluation)
        curve_start: [f64; 3],
        /// End point of the basis curve (needed for Line evaluation)
        curve_end: [f64; 3],
        /// A point on the axis of revolution
        axis_location: [f64; 3],
        /// Direction of the axis of revolution (will be normalized)
        axis_direction: [f64; 3],
    },
    /// Surface of linear extrusion - created by sweeping a curve along a direction vector
    /// u parameter: curve parameter (0 to 1)
    /// v parameter: extrusion distance (scalar multiple along direction)
    /// Parametric equation: P(u,v) = C(u) + v*direction
    SurfaceOfLinearExtrusion {
        /// The generatrix curve being extruded
        basis_curve: CurveType,
        /// Start point of the basis curve (needed for Line evaluation)
        curve_start: [f64; 3],
        /// End point of the basis curve (needed for Line evaluation)
        curve_end: [f64; 3],
        /// Direction vector of extrusion (will be normalized for distances)
        direction: [f64; 3],
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
            SurfaceType::BezierSurface {
                control_points,
                u_degree,
                v_degree,
            } => {
                evaluate_bezier_surface(u, v, *u_degree, *v_degree, control_points)
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
            SurfaceType::SurfaceOfRevolution {
                basis_curve,
                curve_start,
                curve_end,
                axis_location,
                axis_direction,
            } => {
                evaluate_surface_of_revolution(
                    u, v,
                    basis_curve,
                    curve_start,
                    curve_end,
                    axis_location,
                    axis_direction,
                )
            }
            SurfaceType::SurfaceOfLinearExtrusion {
                basis_curve,
                curve_start,
                curve_end,
                direction,
            } => {
                evaluate_surface_of_linear_extrusion(
                    u, v,
                    basis_curve,
                    curve_start,
                    curve_end,
                    direction,
                )
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
            SurfaceType::BezierSurface {
                control_points,
                u_degree,
                v_degree,
            } => {
                bezier_surface_normal(u, v, *u_degree, *v_degree, control_points)
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
            SurfaceType::SurfaceOfRevolution {
                basis_curve,
                curve_start,
                curve_end,
                axis_location,
                axis_direction,
            } => {
                surface_of_revolution_normal(
                    u, v,
                    basis_curve,
                    curve_start,
                    curve_end,
                    axis_location,
                    axis_direction,
                )
            }
            SurfaceType::SurfaceOfLinearExtrusion {
                basis_curve,
                curve_start,
                curve_end,
                direction,
            } => {
                surface_of_linear_extrusion_normal(
                    u, v,
                    basis_curve,
                    curve_start,
                    curve_end,
                    direction,
                )
            }
        }
    }
}

impl SurfaceType {
    /// Create a SurfaceOfRevolution from a basis curve and axis
    pub fn revolution(
        basis_curve: CurveType,
        curve_start: [f64; 3],
        curve_end: [f64; 3],
        axis_location: [f64; 3],
        axis_direction: [f64; 3],
    ) -> Self {
        SurfaceType::SurfaceOfRevolution {
            basis_curve,
            curve_start,
            curve_end,
            axis_location,
            axis_direction,
        }
    }
    
    /// Create a SurfaceOfLinearExtrusion from a basis curve and extrusion direction
    pub fn linear_extrusion(
        basis_curve: CurveType,
        curve_start: [f64; 3],
        curve_end: [f64; 3],
        direction: [f64; 3],
    ) -> Self {
        SurfaceType::SurfaceOfLinearExtrusion {
            basis_curve,
            curve_start,
            curve_end,
            direction,
        }
    }
    
    /// Create a Bezier surface from a 2D grid of control points
    /// 
    /// # Arguments
    /// * `control_points` - 2D grid of control points [i][j]
    ///   The dimensions determine the degrees: u_degree = num_rows - 1, v_degree = num_cols - 1
    pub fn bezier(control_points: Vec<Vec<[f64; 3]>>) -> Result<Self> {
        if control_points.is_empty() || control_points[0].is_empty() {
            return Err(CascadeError::InvalidGeometry(
                "Bezier surface requires at least a 1x1 grid of control points".to_string()
            ));
        }
        
        let u_degree = control_points.len() - 1;
        let v_degree = control_points[0].len() - 1;
        
        Ok(SurfaceType::BezierSurface {
            control_points,
            u_degree,
            v_degree,
        })
    }
    
    /// Get the basis curve of a surface of linear extrusion
    /// Returns None if this is not a SurfaceOfLinearExtrusion
    pub fn basis_curve(&self) -> Option<&CurveType> {
        match self {
            SurfaceType::SurfaceOfLinearExtrusion { basis_curve, .. } => Some(basis_curve),
            SurfaceType::SurfaceOfRevolution { basis_curve, .. } => Some(basis_curve),
            _ => None,
        }
    }
    
    /// Get the extrusion direction of a surface of linear extrusion
    /// Returns None if this is not a SurfaceOfLinearExtrusion
    pub fn extrusion_direction(&self) -> Option<[f64; 3]> {
        match self {
            SurfaceType::SurfaceOfLinearExtrusion { direction, .. } => Some(*direction),
            _ => None,
        }
    }
    
    /// Get a control point from a Bezier surface
    /// Returns None if this is not a Bezier surface or indices are out of bounds
    pub fn control_point(&self, i: usize, j: usize) -> Option<[f64; 3]> {
        match self {
            SurfaceType::BezierSurface { control_points, .. } => {
                if i < control_points.len() && j < control_points[i].len() {
                    Some(control_points[i][j])
                } else {
                    None
                }
            }
            _ => None,
        }
    }
    
    /// Get the degree in U direction of a Bezier surface
    /// Returns None if this is not a Bezier surface
    pub fn degree_u(&self) -> Option<usize> {
        match self {
            SurfaceType::BezierSurface { u_degree, .. } => Some(*u_degree),
            _ => None,
        }
    }
    
    /// Get the degree in V direction of a Bezier surface
    /// Returns None if this is not a Bezier surface
    pub fn degree_v(&self) -> Option<usize> {
        match self {
            SurfaceType::BezierSurface { v_degree, .. } => Some(*v_degree),
            _ => None,
        }
    }
    
    /// Get all control points of a Bezier surface
    /// Returns None if this is not a Bezier surface
    pub fn control_points(&self) -> Option<&Vec<Vec<[f64; 3]>>> {
        match self {
            SurfaceType::BezierSurface { control_points, .. } => Some(control_points),
            _ => None,
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

// ===== Surface of Revolution Functions =====

/// Evaluate a point on the basis curve
/// For Line curves, we need the start/end points; for others we use curve evaluation
fn evaluate_basis_curve(
    curve: &CurveType,
    start: &[f64; 3],
    end: &[f64; 3],
    t: f64,
) -> [f64; 3] {
    match curve {
        CurveType::Line => {
            // Linear interpolation between start and end
            [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2]),
            ]
        }
        CurveType::Arc { center, radius } => {
            let angle = t * 2.0 * std::f64::consts::PI;
            [
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
                center[2],
            ]
        }
        CurveType::Ellipse { center, major_axis, minor_axis } => {
            let angle = t * 2.0 * std::f64::consts::PI;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            [
                center[0] + cos_a * major_axis[0] + sin_a * minor_axis[0],
                center[1] + cos_a * major_axis[1] + sin_a * minor_axis[1],
                center[2] + cos_a * major_axis[2] + sin_a * minor_axis[2],
            ]
        }
        CurveType::Bezier { control_points } => {
            evaluate_bezier_curve(control_points, t)
        }
        CurveType::BSpline { control_points, knots, degree, weights } => {
            evaluate_bspline_curve(control_points, knots, *degree, weights.as_deref(), t)
        }
        CurveType::Parabola { origin, x_dir, y_dir, focal } => {
            // P(u) = origin + u * x_dir + (u² / (4 * focal)) * y_dir
            // Remap t from [0,1] to a useful parameter range (e.g., [-10, 10])
            let u = (t - 0.5) * 20.0;
            let u_squared_term = u * u / (4.0 * focal);
            [
                origin[0] + u * x_dir[0] + u_squared_term * y_dir[0],
                origin[1] + u * x_dir[1] + u_squared_term * y_dir[1],
                origin[2] + u * x_dir[2] + u_squared_term * y_dir[2],
            ]
        }
        CurveType::Trimmed { basis_curve, u1, u2 } => {
            // Remap t ∈ [0,1] to [u1, u2]
            let u = u1 + t * (u2 - u1);
            evaluate_basis_curve(basis_curve, start, end, u)
        }
        CurveType::Offset { .. } => {
            // Offset curve: for now, fallback to linear interpolation
            // TODO: Implement proper offset curve evaluation
            [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2]),
            ]
        }
        CurveType::Hyperbola { .. } => {
            // Hyperbola curve: for now, fallback to linear interpolation
            // TODO: Implement proper hyperbola curve evaluation
            [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2]),
            ]
        }
    }
}

/// De Casteljau's algorithm for Bezier curve evaluation
fn evaluate_bezier_curve(control_points: &[[f64; 3]], t: f64) -> [f64; 3] {
    if control_points.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    
    let mut points = control_points.to_vec();
    let n = points.len();
    
    for _ in 0..n - 1 {
        for i in 0..points.len() - 1 {
            points[i] = [
                (1.0 - t) * points[i][0] + t * points[i + 1][0],
                (1.0 - t) * points[i][1] + t * points[i + 1][1],
                (1.0 - t) * points[i][2] + t * points[i + 1][2],
            ];
        }
        points.pop();
    }
    
    points[0]
}

/// Evaluate a B-spline curve at parameter t
fn evaluate_bspline_curve(
    control_points: &[[f64; 3]],
    knots: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
    t: f64,
) -> [f64; 3] {
    if control_points.is_empty() || knots.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    
    // Clamp t to valid range
    let t_min = knots.get(degree).copied().unwrap_or(0.0);
    let t_max = knots.get(knots.len() - degree - 1).copied().unwrap_or(1.0);
    let t_clamped = t.max(t_min).min(t_max);
    
    // Compute basis functions for all control points
    let mut point = [0.0; 3];
    let mut weight_sum = 0.0;
    
    for (i, cp) in control_points.iter().enumerate() {
        let basis = bspline_basis(t_clamped, i, degree, knots);
        let w = weights.map(|ws| ws.get(i).copied().unwrap_or(1.0)).unwrap_or(1.0);
        let weighted_basis = basis * w;
        
        point[0] += cp[0] * weighted_basis;
        point[1] += cp[1] * weighted_basis;
        point[2] += cp[2] * weighted_basis;
        weight_sum += weighted_basis;
    }
    
    if weight_sum.abs() > 1e-10 {
        [point[0] / weight_sum, point[1] / weight_sum, point[2] / weight_sum]
    } else {
        point
    }
}

/// Rotate a point around an axis using Rodrigues' rotation formula
fn rotate_point_around_axis(
    point: &[f64; 3],
    axis_location: &[f64; 3],
    axis_direction: &[f64; 3],
    angle: f64,
) -> [f64; 3] {
    // Normalize axis direction
    let axis = normalize(axis_direction);
    
    // Translate point relative to axis location
    let p = [
        point[0] - axis_location[0],
        point[1] - axis_location[1],
        point[2] - axis_location[2],
    ];
    
    // Rodrigues' rotation formula:
    // p_rot = p*cos(θ) + (axis × p)*sin(θ) + axis*(axis·p)*(1-cos(θ))
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    
    // axis × p
    let axis_cross_p = cross(&axis, &p);
    
    // axis · p
    let axis_dot_p = axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2];
    
    let rotated = [
        p[0] * cos_a + axis_cross_p[0] * sin_a + axis[0] * axis_dot_p * (1.0 - cos_a),
        p[1] * cos_a + axis_cross_p[1] * sin_a + axis[1] * axis_dot_p * (1.0 - cos_a),
        p[2] * cos_a + axis_cross_p[2] * sin_a + axis[2] * axis_dot_p * (1.0 - cos_a),
    ];
    
    // Translate back
    [
        rotated[0] + axis_location[0],
        rotated[1] + axis_location[1],
        rotated[2] + axis_location[2],
    ]
}

/// Evaluate a point on a surface of revolution
/// u: curve parameter [0, 1]
/// v: rotation angle [0, 2π]
fn evaluate_surface_of_revolution(
    u: f64,
    v: f64,
    basis_curve: &CurveType,
    curve_start: &[f64; 3],
    curve_end: &[f64; 3],
    axis_location: &[f64; 3],
    axis_direction: &[f64; 3],
) -> [f64; 3] {
    // Get point on basis curve at parameter u
    let curve_point = evaluate_basis_curve(basis_curve, curve_start, curve_end, u);
    
    // Rotate around axis by angle v (v in [0, 2π] for full revolution)
    let angle = v;
    rotate_point_around_axis(&curve_point, axis_location, axis_direction, angle)
}

/// Compute normal for a surface of revolution using finite differences
fn surface_of_revolution_normal(
    u: f64,
    v: f64,
    basis_curve: &CurveType,
    curve_start: &[f64; 3],
    curve_end: &[f64; 3],
    axis_location: &[f64; 3],
    axis_direction: &[f64; 3],
) -> [f64; 3] {
    let delta = 1e-5;
    
    // Compute tangent vectors using finite differences
    let pt_u_plus = evaluate_surface_of_revolution(
        (u + delta).min(1.0), v,
        basis_curve, curve_start, curve_end, axis_location, axis_direction
    );
    let pt_u_minus = evaluate_surface_of_revolution(
        (u - delta).max(0.0), v,
        basis_curve, curve_start, curve_end, axis_location, axis_direction
    );
    let tangent_u = [
        pt_u_plus[0] - pt_u_minus[0],
        pt_u_plus[1] - pt_u_minus[1],
        pt_u_plus[2] - pt_u_minus[2],
    ];
    
    let pt_v_plus = evaluate_surface_of_revolution(
        u, v + delta,
        basis_curve, curve_start, curve_end, axis_location, axis_direction
    );
    let pt_v_minus = evaluate_surface_of_revolution(
        u, v - delta,
        basis_curve, curve_start, curve_end, axis_location, axis_direction
    );
    let tangent_v = [
        pt_v_plus[0] - pt_v_minus[0],
        pt_v_plus[1] - pt_v_minus[1],
        pt_v_plus[2] - pt_v_minus[2],
    ];
    
    // Normal is cross product of tangent vectors
    let normal = cross(&tangent_u, &tangent_v);
    normalize(&normal)
}

// ===== Surface of Linear Extrusion Functions =====

/// Evaluate a point on a surface of linear extrusion
/// u: curve parameter [0, 1]
/// v: extrusion distance (scalar multiple along direction)
/// 
/// Parametric equation: P(u,v) = C(u) + v*direction
fn evaluate_surface_of_linear_extrusion(
    u: f64,
    v: f64,
    basis_curve: &CurveType,
    curve_start: &[f64; 3],
    curve_end: &[f64; 3],
    direction: &[f64; 3],
) -> [f64; 3] {
    // Get point on basis curve at parameter u
    let curve_point = evaluate_basis_curve(basis_curve, curve_start, curve_end, u);
    
    // Add the extrusion: P = C(u) + v*direction
    [
        curve_point[0] + v * direction[0],
        curve_point[1] + v * direction[1],
        curve_point[2] + v * direction[2],
    ]
}

/// Compute normal for a surface of linear extrusion using finite differences
fn surface_of_linear_extrusion_normal(
    u: f64,
    v: f64,
    basis_curve: &CurveType,
    curve_start: &[f64; 3],
    curve_end: &[f64; 3],
    direction: &[f64; 3],
) -> [f64; 3] {
    let delta = 1e-5;
    
    // Compute tangent vectors using finite differences
    let pt_u_plus = evaluate_surface_of_linear_extrusion(
        (u + delta).min(1.0), v,
        basis_curve, curve_start, curve_end, direction
    );
    let pt_u_minus = evaluate_surface_of_linear_extrusion(
        (u - delta).max(0.0), v,
        basis_curve, curve_start, curve_end, direction
    );
    let tangent_u = [
        pt_u_plus[0] - pt_u_minus[0],
        pt_u_plus[1] - pt_u_minus[1],
        pt_u_plus[2] - pt_u_minus[2],
    ];
    
    // For linear extrusion, tangent in v direction is just the direction vector
    let tangent_v = *direction;
    
    // Normal is cross product of tangent vectors
    let normal = cross(&tangent_u, &tangent_v);
    normalize(&normal)
}

// ===== Bezier Surface Functions =====

/// Compute Bernstein polynomial basis B_{n,i}(t) = C(n,i) * (1-t)^(n-i) * t^i
/// where C(n,i) is the binomial coefficient
fn bernstein_basis(n: usize, i: usize, t: f64) -> f64 {
    if i > n {
        return 0.0;
    }
    
    // Compute binomial coefficient C(n, i)
    let binom = if i == 0 || i == n {
        1.0
    } else {
        let mut coeff = 1.0;
        for k in 0..i {
            coeff *= (n - k) as f64 / (k + 1) as f64;
        }
        coeff
    };
    
    let one_minus_t = 1.0 - t;
    binom * one_minus_t.powi((n - i) as i32) * t.powi(i as i32)
}

/// Evaluate a point on a Bezier surface using de Casteljau's algorithm
/// 
/// Surface is defined by:
/// S(u, v) = Σ Σ B_{m,i}(u) * B_{n,j}(v) * P_{i,j}
/// 
/// where:
/// - P_{i,j} are the 2D grid of control points
/// - B are Bernstein basis functions
/// - m = u_degree, n = v_degree
fn evaluate_bezier_surface(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    control_points: &[Vec<[f64; 3]>],
) -> [f64; 3] {
    if control_points.is_empty() || control_points[0].is_empty() {
        return [0.0, 0.0, 0.0];
    }
    
    // Evaluate using tensor product of Bernstein polynomials
    let mut point = [0.0; 3];
    
    for i in 0..=u_degree {
        for j in 0..=v_degree {
            if i < control_points.len() && j < control_points[i].len() {
                let b_u = bernstein_basis(u_degree, i, u);
                let b_v = bernstein_basis(v_degree, j, v);
                let basis = b_u * b_v;
                
                let cp = control_points[i][j];
                point[0] += basis * cp[0];
                point[1] += basis * cp[1];
                point[2] += basis * cp[2];
            }
        }
    }
    
    point
}

/// Compute normal to a Bezier surface using finite differences
fn bezier_surface_normal(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    control_points: &[Vec<[f64; 3]>],
) -> [f64; 3] {
    let delta = 1e-5;
    
    // Compute tangent vectors using finite differences
    let pt_u_plus = evaluate_bezier_surface(
        (u + delta).min(1.0), v,
        u_degree, v_degree, control_points
    );
    let pt_u_minus = evaluate_bezier_surface(
        (u - delta).max(0.0), v,
        u_degree, v_degree, control_points
    );
    let tangent_u = [
        pt_u_plus[0] - pt_u_minus[0],
        pt_u_plus[1] - pt_u_minus[1],
        pt_u_plus[2] - pt_u_minus[2],
    ];
    
    let pt_v_plus = evaluate_bezier_surface(
        u, (v + delta).min(1.0),
        u_degree, v_degree, control_points
    );
    let pt_v_minus = evaluate_bezier_surface(
        u, (v - delta).max(0.0),
        u_degree, v_degree, control_points
    );
    let tangent_v = [
        pt_v_plus[0] - pt_v_minus[0],
        pt_v_plus[1] - pt_v_minus[1],
        pt_v_plus[2] - pt_v_minus[2],
    ];
    
    // Normal is cross product of tangent vectors
    let normal = cross(&tangent_u, &tangent_v);
    normalize(&normal)
}
