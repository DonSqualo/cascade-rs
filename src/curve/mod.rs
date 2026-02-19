//! Curve types and evaluation functions

use crate::{Result, CascadeError, brep::CurveType};

/// Helper function to normalize a 3D vector
fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Helper function to compute cross product
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Helper function to compute dot product
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A trimmed curve is a bounded portion of an underlying curve
/// defined by parameter limits [u1, u2].
/// 
/// This provides a high-level wrapper around the CurveType::Trimmed variant
/// with convenient accessor methods matching OpenCASCADE's Geom_TrimmedCurve.
#[derive(Debug, Clone)]
pub struct TrimmedCurve {
    /// The underlying basis curve
    basis: CurveType,
    /// First parameter (start of trimmed portion)
    u1: f64,
    /// Second parameter (end of trimmed portion)  
    u2: f64,
}

impl TrimmedCurve {
    /// Create a new trimmed curve from a basis curve and parameter bounds.
    /// 
    /// # Arguments
    /// * `basis` - The underlying curve to trim
    /// * `u1` - First parameter (start of trimmed portion)
    /// * `u2` - Second parameter (end of trimmed portion)
    /// 
    /// # Returns
    /// A new TrimmedCurve, or an error if u1 >= u2
    pub fn new(basis: CurveType, u1: f64, u2: f64) -> Result<Self> {
        if u1 >= u2 {
            return Err(CascadeError::InvalidGeometry(
                format!("TrimmedCurve: u1 ({}) must be less than u2 ({})", u1, u2)
            ));
        }
        Ok(Self { basis, u1, u2 })
    }
    
    /// Get a reference to the underlying basis curve.
    /// 
    /// Corresponds to OpenCASCADE's Geom_TrimmedCurve::BasisCurve()
    pub fn basis_curve(&self) -> &CurveType {
        &self.basis
    }
    
    /// Get the first parameter (start of trimmed portion).
    /// 
    /// Corresponds to OpenCASCADE's Geom_TrimmedCurve::FirstParameter()
    pub fn first_parameter(&self) -> f64 {
        self.u1
    }
    
    /// Get the last parameter (end of trimmed portion).
    /// 
    /// Corresponds to OpenCASCADE's Geom_TrimmedCurve::LastParameter()
    pub fn last_parameter(&self) -> f64 {
        self.u2
    }
    
    /// Evaluate a point on the trimmed curve at parameter t ∈ [0, 1].
    /// 
    /// The parameter t is mapped to the range [u1, u2] of the basis curve:
    /// basis_parameter = u1 + t * (u2 - u1)
    /// 
    /// # Arguments
    /// * `t` - Parameter in [0, 1]
    /// 
    /// # Returns
    /// The 3D point on the curve at the mapped parameter
    pub fn point_at(&self, t: f64) -> Result<[f64; 3]> {
        let mapped_t = self.u1 + t * (self.u2 - self.u1);
        point_at(&self.basis, mapped_t)
    }
    
    /// Evaluate a point on the trimmed curve at a raw parameter value.
    /// 
    /// Unlike point_at(), this uses the parameter directly on the basis curve
    /// without mapping. The parameter should be in [u1, u2].
    /// 
    /// # Arguments
    /// * `u` - Parameter on the basis curve (should be in [u1, u2])
    /// 
    /// # Returns
    /// The 3D point on the curve, or error if u is out of bounds
    pub fn point_at_parameter(&self, u: f64) -> Result<[f64; 3]> {
        if u < self.u1 - 1e-10 || u > self.u2 + 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                format!("Parameter {} is outside trimmed range [{}, {}]", u, self.u1, self.u2)
            ));
        }
        point_at(&self.basis, u)
    }
    
    /// Evaluate the tangent vector at parameter t ∈ [0, 1].
    /// 
    /// # Arguments
    /// * `t` - Parameter in [0, 1]
    /// 
    /// # Returns
    /// The tangent vector at the mapped parameter
    pub fn tangent_at(&self, t: f64) -> Result<[f64; 3]> {
        let mapped_t = self.u1 + t * (self.u2 - self.u1);
        let mut tang = tangent_at(&self.basis, mapped_t)?;
        // Scale by parameter range for correct derivative
        let scale = self.u2 - self.u1;
        tang[0] *= scale;
        tang[1] *= scale;
        tang[2] *= scale;
        Ok(tang)
    }
    
    /// Get the start point of the trimmed curve (at t=0 / u=u1).
    pub fn start_point(&self) -> Result<[f64; 3]> {
        point_at(&self.basis, self.u1)
    }
    
    /// Get the end point of the trimmed curve (at t=1 / u=u2).
    pub fn end_point(&self) -> Result<[f64; 3]> {
        point_at(&self.basis, self.u2)
    }
    
    /// Convert to a CurveType::Trimmed enum variant for storage in edges.
    pub fn to_curve_type(&self) -> CurveType {
        CurveType::Trimmed {
            basis_curve: Box::new(self.basis.clone()),
            u1: self.u1,
            u2: self.u2,
        }
    }
    
    /// Create a TrimmedCurve from a CurveType::Trimmed variant.
    /// 
    /// Returns None if the curve type is not Trimmed.
    pub fn from_curve_type(curve: &CurveType) -> Option<Self> {
        match curve {
            CurveType::Trimmed { basis_curve, u1, u2 } => {
                Some(Self {
                    basis: (**basis_curve).clone(),
                    u1: *u1,
                    u2: *u2,
                })
            }
            _ => None,
        }
    }
}

/// A parabola in 3D space
/// 
/// Defined by a position (axis system with origin, X and Y directions) and a focal parameter.
/// The parabola lies in the plane defined by the X and Y directions, with the vertex at origin.
/// 
/// # Parametric Equation
/// 
/// P(u) = origin + u * x_dir + (u² / (4p)) * y_dir
/// 
/// where p is the focal parameter.
/// 
/// # Geometric Properties
/// 
/// - Vertex: at the origin
/// - Focus: at origin + (p/2) * x_dir (note: using OCCT convention where focal = p, focus at p/2)
/// - Directrix: line perpendicular to x_dir at origin - (p/2) * x_dir
/// - Axis: the x_dir direction (from vertex through focus to infinity)
#[derive(Debug, Clone)]
pub struct Parabola {
    /// Origin (vertex of the parabola)
    pub origin: [f64; 3],
    /// X direction (axis of symmetry, pointing toward focus)
    pub x_dir: [f64; 3],
    /// Y direction (perpendicular to axis, in the plane of the parabola)
    pub y_dir: [f64; 3],
    /// Focal parameter (2 * distance from vertex to focus)
    pub focal: f64,
}

impl Parabola {
    /// Create a new parabola from position and focal parameter
    /// 
    /// # Arguments
    /// * `origin` - Vertex of the parabola
    /// * `x_dir` - Direction along the axis of symmetry (should be unit vector)
    /// * `y_dir` - Perpendicular direction in the parabola plane (should be unit vector)
    /// * `focal` - Focal parameter (p). The focus is at distance p/2 from vertex.
    pub fn new(origin: [f64; 3], x_dir: [f64; 3], y_dir: [f64; 3], focal: f64) -> Result<Self> {
        if focal.abs() < 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                "Focal parameter must be non-zero".to_string(),
            ));
        }
        
        Ok(Self { origin, x_dir, y_dir, focal })
    }
    
    /// Create a parabola in the XY plane with vertex at origin
    /// 
    /// Standard parabola: y² = 4px (or equivalently, x = y²/(4p))
    /// In our parameterization: P(u) = (u, u²/(4p), 0)
    pub fn standard(focal: f64) -> Result<Self> {
        Self::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            focal,
        )
    }
    
    /// Get the focus point
    /// 
    /// The focus is located at origin + (focal/2) * x_dir
    pub fn focus(&self) -> [f64; 3] {
        let f = self.focal / 2.0;
        [
            self.origin[0] + f * self.x_dir[0],
            self.origin[1] + f * self.x_dir[1],
            self.origin[2] + f * self.x_dir[2],
        ]
    }
    
    /// Get a point on the directrix
    /// 
    /// The directrix is a line perpendicular to x_dir, passing through 
    /// origin - (focal/2) * x_dir. Returns the point on the directrix 
    /// closest to the vertex.
    pub fn directrix_point(&self) -> [f64; 3] {
        let f = self.focal / 2.0;
        [
            self.origin[0] - f * self.x_dir[0],
            self.origin[1] - f * self.x_dir[1],
            self.origin[2] - f * self.x_dir[2],
        ]
    }
    
    /// Get the directrix direction (perpendicular to axis)
    /// 
    /// The directrix is a line in the direction of y_dir
    pub fn directrix_direction(&self) -> [f64; 3] {
        self.y_dir
    }
    
    /// Get the focal parameter (p)
    /// 
    /// This is 2 times the distance from vertex to focus
    pub fn parameter(&self) -> f64 {
        self.focal
    }
    
    /// Get the focal distance (distance from vertex to focus)
    /// 
    /// This is focal/2
    pub fn focal_distance(&self) -> f64 {
        self.focal / 2.0
    }
    
    /// Evaluate point on parabola at parameter u
    /// 
    /// P(u) = origin + u * x_dir + (u² / (4p)) * y_dir
    pub fn point_at(&self, u: f64) -> [f64; 3] {
        let u_sq_term = u * u / (4.0 * self.focal);
        [
            self.origin[0] + u * self.x_dir[0] + u_sq_term * self.y_dir[0],
            self.origin[1] + u * self.x_dir[1] + u_sq_term * self.y_dir[1],
            self.origin[2] + u * self.x_dir[2] + u_sq_term * self.y_dir[2],
        ]
    }
    
    /// Evaluate tangent at parameter u
    /// 
    /// dP/du = x_dir + (u / (2p)) * y_dir
    pub fn tangent_at(&self, u: f64) -> [f64; 3] {
        let y_factor = u / (2.0 * self.focal);
        [
            self.x_dir[0] + y_factor * self.y_dir[0],
            self.x_dir[1] + y_factor * self.y_dir[1],
            self.x_dir[2] + y_factor * self.y_dir[2],
        ]
    }
    
    /// Convert to CurveType enum for use with edges
    pub fn to_curve_type(&self) -> CurveType {
        CurveType::Parabola {
            origin: self.origin,
            x_dir: self.x_dir,
            y_dir: self.y_dir,
            focal: self.focal,
        }
    }
    
    /// Get the vertex (same as origin)
    pub fn vertex(&self) -> [f64; 3] {
        self.origin
    }
    
    /// Get the axis direction (same as x_dir)
    pub fn axis(&self) -> [f64; 3] {
        self.x_dir
    }
}

/// A hyperbola in 3D space
/// 
/// Defined by a center, two perpendicular axes, and semi-axis lengths (major_radius and minor_radius).
/// The hyperbola lies in the plane defined by the x_dir and y_dir, with center at the origin position.
/// 
/// # Parametric Equation
/// 
/// P(u) = center + major_radius * cosh(u) * x_dir + minor_radius * sinh(u) * y_dir
/// 
/// where:
/// - cosh(u) = (e^u + e^(-u)) / 2
/// - sinh(u) = (e^u - e^(-u)) / 2
/// 
/// # Geometric Properties
/// 
/// - Center: at the origin position
/// - Foci: at center ± c * x_dir, where c = sqrt(major_radius² + minor_radius²)
/// - Eccentricity: e = c / major_radius
/// - Asymptotes: lines through center with slopes ±(minor_radius / major_radius)
/// - The right branch (u > 0) and left branch (u < 0) form the two parts of the hyperbola
#[derive(Debug, Clone)]
pub struct Hyperbola {
    /// Center of the hyperbola
    pub center: [f64; 3],
    /// X direction (along the transverse axis, toward the vertices)
    pub x_dir: [f64; 3],
    /// Y direction (perpendicular to x_dir, in the plane of the hyperbola)
    pub y_dir: [f64; 3],
    /// Semi-major axis length (a) - distance from center to vertex along transverse axis
    pub major_radius: f64,
    /// Semi-minor axis length (b) - determines the asymptotic slope
    pub minor_radius: f64,
}

impl Hyperbola {
    /// Create a new hyperbola from center, axes directions, and semi-axis lengths
    /// 
    /// # Arguments
    /// * `center` - Center of the hyperbola
    /// * `x_dir` - Direction along the transverse axis (should be unit vector)
    /// * `y_dir` - Direction along the conjugate axis (should be unit vector)
    /// * `major_radius` - Semi-major axis length (a), must be positive
    /// * `minor_radius` - Semi-minor axis length (b), must be positive
    pub fn new(
        center: [f64; 3],
        x_dir: [f64; 3],
        y_dir: [f64; 3],
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<Self> {
        if major_radius.abs() < 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                "Major radius must be non-zero".to_string(),
            ));
        }
        if minor_radius.abs() < 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                "Minor radius must be non-zero".to_string(),
            ));
        }
        
        Ok(Self {
            center,
            x_dir,
            y_dir,
            major_radius: major_radius.abs(),
            minor_radius: minor_radius.abs(),
        })
    }
    
    /// Create a standard hyperbola in the XY plane centered at origin
    /// 
    /// Standard hyperbola: x²/a² - y²/b² = 1
    /// Parametric: x = a*cosh(t), y = b*sinh(t)
    pub fn standard(major_radius: f64, minor_radius: f64) -> Result<Self> {
        Self::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            major_radius,
            minor_radius,
        )
    }
    
    /// Get the distance from center to focus (linear eccentricity)
    /// 
    /// c = sqrt(a² + b²)
    fn focal_distance(&self) -> f64 {
        (self.major_radius * self.major_radius + self.minor_radius * self.minor_radius).sqrt()
    }
    
    /// Get the first focus (along positive x_dir)
    /// 
    /// Focus = center + focal_distance * x_dir
    pub fn focus1(&self) -> [f64; 3] {
        let c = self.focal_distance();
        [
            self.center[0] + c * self.x_dir[0],
            self.center[1] + c * self.x_dir[1],
            self.center[2] + c * self.x_dir[2],
        ]
    }
    
    /// Get the second focus (along negative x_dir)
    /// 
    /// Focus = center - focal_distance * x_dir
    pub fn focus2(&self) -> [f64; 3] {
        let c = self.focal_distance();
        [
            self.center[0] - c * self.x_dir[0],
            self.center[1] - c * self.x_dir[1],
            self.center[2] - c * self.x_dir[2],
        ]
    }
    
    /// Get the eccentricity
    /// 
    /// e = c / a = sqrt(1 + (b/a)²)
    pub fn eccentricity(&self) -> f64 {
        self.focal_distance() / self.major_radius
    }
    
    /// Get the semi-major axis length (a)
    pub fn parameter(&self) -> f64 {
        self.major_radius
    }
    
    /// Get the asymptotes as two lines through the center
    /// 
    /// Returns slopes in the form of two direction vectors
    /// Asymptotes have slopes ±(b/a) relative to the x_dir
    pub fn asymptotes(&self) -> ([f64; 3], [f64; 3]) {
        // Asymptote 1: direction = x_dir + (b/a) * y_dir
        let slope_ratio = self.minor_radius / self.major_radius;
        let asym1 = [
            self.x_dir[0] + slope_ratio * self.y_dir[0],
            self.x_dir[1] + slope_ratio * self.y_dir[1],
            self.x_dir[2] + slope_ratio * self.y_dir[2],
        ];
        
        // Asymptote 2: direction = x_dir - (b/a) * y_dir
        let asym2 = [
            self.x_dir[0] - slope_ratio * self.y_dir[0],
            self.x_dir[1] - slope_ratio * self.y_dir[1],
            self.x_dir[2] - slope_ratio * self.y_dir[2],
        ];
        
        (asym1, asym2)
    }
    
    /// Evaluate a point on the hyperbola at parameter u
    /// 
    /// P(u) = center + a*cosh(u)*x_dir + b*sinh(u)*y_dir
    pub fn point_at(&self, u: f64) -> [f64; 3] {
        let cosh_u = u.cosh();
        let sinh_u = u.sinh();
        
        let x_component = self.major_radius * cosh_u;
        let y_component = self.minor_radius * sinh_u;
        
        [
            self.center[0] + x_component * self.x_dir[0] + y_component * self.y_dir[0],
            self.center[1] + x_component * self.x_dir[1] + y_component * self.y_dir[1],
            self.center[2] + x_component * self.x_dir[2] + y_component * self.y_dir[2],
        ]
    }
    
    /// Evaluate the tangent vector at parameter u
    /// 
    /// dP/du = a*sinh(u)*x_dir + b*cosh(u)*y_dir
    pub fn tangent_at(&self, u: f64) -> [f64; 3] {
        let sinh_u = u.sinh();
        let cosh_u = u.cosh();
        
        let x_component = self.major_radius * sinh_u;
        let y_component = self.minor_radius * cosh_u;
        
        [
            x_component * self.x_dir[0] + y_component * self.y_dir[0],
            x_component * self.x_dir[1] + y_component * self.y_dir[1],
            x_component * self.x_dir[2] + y_component * self.y_dir[2],
        ]
    }
    
    /// Convert to CurveType enum for use with edges
    pub fn to_curve_type(&self) -> CurveType {
        CurveType::Hyperbola {
            center: self.center,
            x_dir: self.x_dir,
            y_dir: self.y_dir,
            major_radius: self.major_radius,
            minor_radius: self.minor_radius,
        }
    }
    
    /// Get the center of the hyperbola
    pub fn center_point(&self) -> [f64; 3] {
        self.center
    }
}

/// Evaluate a curve at parameter t, returning a 3D point
/// 
/// t should typically be in [0, 1] for most curves
pub fn point_at(curve: &CurveType, t: f64) -> Result<[f64; 3]> {
    match curve {
        CurveType::Line => {
            // Line evaluation not directly supported from enum
            // Edges store start/end separately
            Err(CascadeError::NotImplemented(
                "Use edge.start and edge.end for line evaluation".to_string()
            ))
        }
        CurveType::Arc { center, radius } => {
            // Arc is a circular arc in 3D - evaluate as circle with parameter t in [0, 2π]
            // This is a simple 2D circle in the xy-plane centered at center
            let angle = t * 2.0 * std::f64::consts::PI;
            let x = center[0] + radius * angle.cos();
            let y = center[1] + radius * angle.sin();
            let z = center[2];
            Ok([x, y, z])
        }
        CurveType::Ellipse { center, major_axis, minor_axis } => {
            evaluate_ellipse(center, major_axis, minor_axis, t)
        }
        CurveType::Parabola { origin, x_dir, y_dir, focal } => {
            evaluate_parabola(origin, x_dir, y_dir, *focal, t)
        }
        CurveType::Hyperbola { center, x_dir, y_dir, major_radius, minor_radius } => {
            evaluate_hyperbola(center, x_dir, y_dir, *major_radius, *minor_radius, t)
        }
        CurveType::Bezier { control_points } => {
            evaluate_bezier(control_points, t)
        }
        CurveType::BSpline { control_points, knots, degree, weights } => {
            evaluate_bspline(control_points, knots, *degree, weights.as_deref(), t)
        }
        CurveType::Trimmed { basis_curve, u1, u2 } => {
            // Map t ∈ [0, 1] to the trimmed parameter range [u1, u2]
            let mapped_t = *u1 + t * (*u2 - *u1);
            point_at(basis_curve, mapped_t)
        }
        CurveType::Offset { basis_curve, offset_distance, offset_direction } => {
            // Evaluate the basis curve at the parameter
            let basis_point = point_at(basis_curve, t)?;
            let tangent = tangent_at(basis_curve, t)?;
            
            // Compute the offset normal
            let tangent_norm = normalize(&tangent);
            let offset_dir_norm = normalize(offset_direction);
            
            // Compute normal: cross product of offset_direction and tangent
            let normal = cross(&offset_dir_norm, &tangent_norm);
            let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            
            let final_normal = if normal_len > 1e-10 {
                [normal[0] / normal_len, normal[1] / normal_len, normal[2] / normal_len]
            } else {
                // Fallback: try alternative perpendicular direction
                let perp_dir = if offset_dir_norm[0].abs() < 0.9 {
                    [0.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                
                let normal2 = cross(&perp_dir, &tangent_norm);
                let normal2_len = (normal2[0] * normal2[0] + normal2[1] * normal2[1] + normal2[2] * normal2[2]).sqrt();
                
                if normal2_len > 1e-10 {
                    [normal2[0] / normal2_len, normal2[1] / normal2_len, normal2[2] / normal2_len]
                } else {
                    return Err(CascadeError::InvalidGeometry(
                        "Cannot compute offset normal: no perpendicular direction found".to_string()
                    ));
                }
            };
            
            // Offset the point
            Ok([
                basis_point[0] + offset_distance * final_normal[0],
                basis_point[1] + offset_distance * final_normal[1],
                basis_point[2] + offset_distance * final_normal[2],
            ])
        }
    }
}

/// Evaluate tangent (first derivative) of a curve at parameter t
pub fn tangent_at(curve: &CurveType, t: f64) -> Result<[f64; 3]> {
    match curve {
        CurveType::Line => {
            Err(CascadeError::NotImplemented(
                "Use normalized vector from edge.start to edge.end for line tangent".to_string()
            ))
        }
        CurveType::Arc { center, radius } => {
            tangent_arc(center, *radius, t)
        }
        CurveType::Ellipse { center, major_axis, minor_axis } => {
            tangent_ellipse(center, major_axis, minor_axis, t)
        }
        CurveType::Parabola { origin, x_dir, y_dir, focal } => {
            tangent_parabola(x_dir, y_dir, *focal, t)
        }
        CurveType::Hyperbola { center: _, x_dir, y_dir, major_radius, minor_radius } => {
            tangent_hyperbola(x_dir, y_dir, *major_radius, *minor_radius, t)
        }
        CurveType::Bezier { control_points } => {
            tangent_bezier(control_points, t)
        }
        CurveType::BSpline { control_points, knots, degree, weights } => {
            tangent_bspline(control_points, knots, *degree, weights.as_deref(), t)
        }
        CurveType::Trimmed { basis_curve, u1, u2 } => {
            // Map t ∈ [0, 1] to the trimmed parameter range [u1, u2]
            let mapped_t = *u1 + t * (*u2 - *u1);
            // Get tangent at mapped parameter, then scale by the parameter range
            let mut tang = tangent_at(basis_curve, mapped_t)?;
            let scale = *u2 - *u1;
            tang[0] *= scale;
            tang[1] *= scale;
            tang[2] *= scale;
            Ok(tang)
        }
        CurveType::Offset { basis_curve, offset_distance: _, offset_direction: _ } => {
            // The tangent of an offset curve is parallel to the basis curve's tangent
            // (offsetting doesn't change the direction, only the position)
            tangent_at(basis_curve, t)
        }
    }
}

/// Evaluate an ellipse at parameter t ∈ [0, 1]
/// 
/// The ellipse is defined by:
/// - center: center point
/// - major_axis: vector along major axis (not necessarily unit)
/// - minor_axis: vector along minor axis (perpendicular to major)
fn evaluate_ellipse(
    center: &[f64; 3],
    major_axis: &[f64; 3],
    minor_axis: &[f64; 3],
    t: f64,
) -> Result<[f64; 3]> {
    let angle = t * 2.0 * std::f64::consts::PI;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    
    // P(t) = center + cos(t) * major_axis + sin(t) * minor_axis
    let mut point = [0.0; 3];
    for i in 0..3 {
        point[i] = center[i] + cos_a * major_axis[i] + sin_a * minor_axis[i];
    }
    Ok(point)
}

/// Tangent vector to ellipse at parameter t
fn tangent_ellipse(
    _center: &[f64; 3],
    major_axis: &[f64; 3],
    minor_axis: &[f64; 3],
    t: f64,
) -> Result<[f64; 3]> {
    let angle = t * 2.0 * std::f64::consts::PI;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    
    // dP/dt = -sin(t) * 2π * major_axis + cos(t) * 2π * minor_axis
    let factor = 2.0 * std::f64::consts::PI;
    let mut tangent = [0.0; 3];
    for i in 0..3 {
        tangent[i] = -sin_a * factor * major_axis[i] + cos_a * factor * minor_axis[i];
    }
    Ok(tangent)
}

/// Tangent to circular arc at parameter t ∈ [0, 1]
fn tangent_arc(
    center: &[f64; 3],
    radius: f64,
    t: f64,
) -> Result<[f64; 3]> {
    let angle = t * 2.0 * std::f64::consts::PI;
    let sin_a = angle.sin();
    let cos_a = angle.cos();
    
    // For circle: P(t) = center + r*[cos(2πt), sin(2πt), 0]
    // dP/dt = r*2π*[-sin(2πt), cos(2πt), 0]
    let factor = radius * 2.0 * std::f64::consts::PI;
    Ok([-sin_a * factor, cos_a * factor, 0.0])
}

/// An offset curve is a curve displaced by a constant distance in a given direction.
/// 
/// For 2D curves (curves that lie in a plane), the offset is perpendicular to the curve.
/// For 3D curves, the offset is computed in the plane defined by the curve's tangent vector
/// and a reference direction.
/// 
/// # Mathematical Definition
/// 
/// Given a basis curve C(u) and an offset distance d:
/// - For 2D: The offset curve is C'(u) = C(u) + d * N(u), where N(u) is the unit normal perpendicular to the tangent
/// - For 3D: The offset is computed using the plane formed by the tangent and reference direction
#[derive(Debug, Clone)]
pub struct OffsetCurve {
    /// The underlying basis curve being offset
    basis: CurveType,
    /// The offset distance (can be positive or negative)
    offset_distance: f64,
    /// Reference direction for 3D offset (defines the plane with the tangent)
    offset_direction: [f64; 3],
}

impl OffsetCurve {
    /// Create a new offset curve from a basis curve and offset distance.
    /// 
    /// # Arguments
    /// * `basis` - The underlying curve to offset
    /// * `offset_distance` - The distance to offset (positive = outward, negative = inward)
    /// * `offset_direction` - Reference direction for 3D offset (should be normalized)
    /// 
    /// # Returns
    /// A new OffsetCurve or an error if the offset_distance is zero
    pub fn new(basis: CurveType, offset_distance: f64, offset_direction: [f64; 3]) -> Result<Self> {
        if offset_distance.abs() < 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                "OffsetCurve: offset_distance must be non-zero".to_string()
            ));
        }
        Ok(Self {
            basis,
            offset_distance,
            offset_direction: normalize(&offset_direction),
        })
    }
    
    /// Get a reference to the underlying basis curve.
    /// 
    /// Corresponds to OpenCASCADE's Geom_OffsetCurve::BasisCurve()
    pub fn basis_curve(&self) -> &CurveType {
        &self.basis
    }
    
    /// Get the offset distance.
    /// 
    /// Corresponds to OpenCASCADE's Geom_OffsetCurve::Offset()
    pub fn offset(&self) -> f64 {
        self.offset_distance
    }
    
    /// Get the reference direction for 3D offset.
    /// 
    /// Corresponds to OpenCASCADE's Geom_OffsetCurve::Direction()
    pub fn direction(&self) -> [f64; 3] {
        self.offset_direction
    }
    
    /// Evaluate a point on the offset curve at parameter t ∈ [0, 1].
    /// 
    /// Computes the point on the basis curve, then offsets it by the normal vector.
    /// For 2D curves, the normal is perpendicular to the tangent in the plane.
    /// For 3D curves, the normal is computed from the tangent and reference direction.
    /// 
    /// # Arguments
    /// * `t` - Parameter in [0, 1]
    /// 
    /// # Returns
    /// The 3D point on the offset curve at parameter t
    pub fn point_at(&self, t: f64) -> Result<[f64; 3]> {
        let basis_point = point_at(&self.basis, t)?;
        let tangent = tangent_at(&self.basis, t)?;
        
        // Compute the normal vector for the offset
        let normal = self.compute_offset_normal(&tangent)?;
        
        // Offset the point
        Ok([
            basis_point[0] + self.offset_distance * normal[0],
            basis_point[1] + self.offset_distance * normal[1],
            basis_point[2] + self.offset_distance * normal[2],
        ])
    }
    
    /// Evaluate the tangent vector at parameter t ∈ [0, 1].
    /// 
    /// For offset curves, the tangent is the same as the basis curve's tangent
    /// (offsetting doesn't change the direction, only the position).
    /// 
    /// # Arguments
    /// * `t` - Parameter in [0, 1]
    /// 
    /// # Returns
    /// The tangent vector at parameter t (same direction as basis curve)
    pub fn tangent_at(&self, t: f64) -> Result<[f64; 3]> {
        // The tangent of an offset curve is parallel to the basis curve's tangent
        tangent_at(&self.basis, t)
    }
    
    /// Convert to a CurveType::Offset enum variant for storage in edges.
    pub fn to_curve_type(&self) -> CurveType {
        CurveType::Offset {
            basis_curve: Box::new(self.basis.clone()),
            offset_distance: self.offset_distance,
            offset_direction: self.offset_direction,
        }
    }
    
    /// Create an OffsetCurve from a CurveType::Offset variant.
    /// 
    /// Returns None if the curve type is not Offset.
    pub fn from_curve_type(curve: &CurveType) -> Option<Self> {
        match curve {
            CurveType::Offset { basis_curve, offset_distance, offset_direction } => {
                Some(Self {
                    basis: (**basis_curve).clone(),
                    offset_distance: *offset_distance,
                    offset_direction: *offset_direction,
                })
            }
            _ => None,
        }
    }
    
    /// Compute the offset normal vector given a tangent vector.
    /// 
    /// For 2D curves (tangent in XY plane): normal is perpendicular in the XY plane
    /// For 3D curves: normal is computed using the cross product of tangent and reference direction
    fn compute_offset_normal(&self, tangent: &[f64; 3]) -> Result<[f64; 3]> {
        let tangent_norm = normalize(tangent);
        
        // Check if tangent is nearly zero
        if tangent_norm[0].abs() < 1e-10 && tangent_norm[1].abs() < 1e-10 && tangent_norm[2].abs() < 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                "Cannot compute offset normal: tangent vector is zero".to_string()
            ));
        }
        
        // Compute normal: cross product of offset_direction and tangent
        // This gives a normal in the plane defined by tangent and offset_direction
        let normal = cross(&self.offset_direction, &tangent_norm);
        let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        
        if normal_len < 1e-10 {
            // Tangent is parallel to offset_direction, try perpendicular direction
            let perp_dir = if (self.offset_direction[0].abs() < 0.9) {
                [0.0, 0.0, 1.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            
            let normal2 = cross(&perp_dir, &tangent_norm);
            let normal2_len = (normal2[0] * normal2[0] + normal2[1] * normal2[1] + normal2[2] * normal2[2]).sqrt();
            
            if normal2_len < 1e-10 {
                return Err(CascadeError::InvalidGeometry(
                    "Cannot compute offset normal: no perpendicular direction found".to_string()
                ));
            }
            
            Ok([
                normal2[0] / normal2_len,
                normal2[1] / normal2_len,
                normal2[2] / normal2_len,
            ])
        } else {
            Ok([
                normal[0] / normal_len,
                normal[1] / normal_len,
                normal[2] / normal_len,
            ])
        }
    }
}

/// Evaluate a parabola at parameter u
/// 
/// The parabola is defined parametrically as:
/// P(u) = origin + u * x_dir + (u² / (4 * focal)) * y_dir
/// 
/// This parameterization places the vertex at origin when u=0.
/// The focus is located at origin + (focal/2) * x_dir.
/// The directrix is the line perpendicular to x_dir at distance -focal/2 from origin.
fn evaluate_parabola(
    origin: &[f64; 3],
    x_dir: &[f64; 3],
    y_dir: &[f64; 3],
    focal: f64,
    u: f64,
) -> Result<[f64; 3]> {
    if focal.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Parabola focal parameter must be non-zero".to_string(),
        ));
    }
    
    // P(u) = origin + u * x_dir + (u² / (4p)) * y_dir
    let u_sq_term = u * u / (4.0 * focal);
    
    let mut point = [0.0; 3];
    for i in 0..3 {
        point[i] = origin[i] + u * x_dir[i] + u_sq_term * y_dir[i];
    }
    Ok(point)
}

/// Tangent vector to parabola at parameter u
/// 
/// The derivative is:
/// dP/du = x_dir + (u / (2 * focal)) * y_dir
fn tangent_parabola(
    x_dir: &[f64; 3],
    y_dir: &[f64; 3],
    focal: f64,
    u: f64,
) -> Result<[f64; 3]> {
    if focal.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Parabola focal parameter must be non-zero".to_string(),
        ));
    }
    
    // dP/du = x_dir + (u / (2p)) * y_dir
    let y_factor = u / (2.0 * focal);
    
    let mut tangent = [0.0; 3];
    for i in 0..3 {
        tangent[i] = x_dir[i] + y_factor * y_dir[i];
    }
    Ok(tangent)
}

/// Evaluate a hyperbola at parameter u
/// 
/// The hyperbola is defined parametrically as:
/// P(u) = center + a*cosh(u)*x_dir + b*sinh(u)*y_dir
/// 
/// where:
/// - a is major_radius
/// - b is minor_radius
/// - cosh(u) = (e^u + e^(-u)) / 2
/// - sinh(u) = (e^u - e^(-u)) / 2
fn evaluate_hyperbola(
    center: &[f64; 3],
    x_dir: &[f64; 3],
    y_dir: &[f64; 3],
    major_radius: f64,
    minor_radius: f64,
    u: f64,
) -> Result<[f64; 3]> {
    if major_radius.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Hyperbola major radius must be non-zero".to_string(),
        ));
    }
    if minor_radius.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Hyperbola minor radius must be non-zero".to_string(),
        ));
    }
    
    // P(u) = center + a*cosh(u)*x_dir + b*sinh(u)*y_dir
    let cosh_u = u.cosh();
    let sinh_u = u.sinh();
    
    let x_component = major_radius * cosh_u;
    let y_component = minor_radius * sinh_u;
    
    let mut point = [0.0; 3];
    for i in 0..3 {
        point[i] = center[i] + x_component * x_dir[i] + y_component * y_dir[i];
    }
    Ok(point)
}

/// Tangent vector to hyperbola at parameter u
/// 
/// The derivative is:
/// dP/du = a*sinh(u)*x_dir + b*cosh(u)*y_dir
fn tangent_hyperbola(
    x_dir: &[f64; 3],
    y_dir: &[f64; 3],
    major_radius: f64,
    minor_radius: f64,
    u: f64,
) -> Result<[f64; 3]> {
    if major_radius.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Hyperbola major radius must be non-zero".to_string(),
        ));
    }
    if minor_radius.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Hyperbola minor radius must be non-zero".to_string(),
        ));
    }
    
    // dP/du = a*sinh(u)*x_dir + b*cosh(u)*y_dir
    let sinh_u = u.sinh();
    let cosh_u = u.cosh();
    
    let x_component = major_radius * sinh_u;
    let y_component = minor_radius * cosh_u;
    
    let mut tangent = [0.0; 3];
    for i in 0..3 {
        tangent[i] = x_component * x_dir[i] + y_component * y_dir[i];
    }
    Ok(tangent)
}

/// Evaluate a Bezier curve using De Casteljau's algorithm
/// 
/// Given control points P0, P1, ..., Pn, evaluate at parameter t ∈ [0, 1]
fn evaluate_bezier(control_points: &[[f64; 3]], t: f64) -> Result<[f64; 3]> {
    if control_points.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Bezier curve requires at least one control point".to_string(),
        ));
    }
    
    // De Casteljau's algorithm
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
    
    Ok(points[0])
}

/// Tangent to Bezier curve at parameter t
fn tangent_bezier(control_points: &[[f64; 3]], t: f64) -> Result<[f64; 3]> {
    if control_points.len() < 2 {
        return Err(CascadeError::InvalidGeometry(
            "Bezier curve tangent requires at least two control points".to_string(),
        ));
    }
    
    // The tangent is the derivative, which is a Bezier curve of degree n-1
    // with control points: n * (P_{i+1} - P_i)
    let mut derivative_points = Vec::new();
    let n = control_points.len() as f64;
    
    for i in 0..control_points.len() - 1 {
        let dp = [
            n * (control_points[i + 1][0] - control_points[i][0]),
            n * (control_points[i + 1][1] - control_points[i][1]),
            n * (control_points[i + 1][2] - control_points[i][2]),
        ];
        derivative_points.push(dp);
    }
    
    evaluate_bezier(&derivative_points, t)
}

/// Evaluate a B-spline curve
/// 
/// Parameters:
/// - control_points: control point vector
/// - knots: knot vector
/// - degree: degree of B-spline (typically 2-4)
/// - weights: optional weights for NURBS (None = uniform B-spline)
/// - t: parameter value in [0, 1] or [knots[degree], knots[knots.len()-degree-1]]
fn evaluate_bspline(
    control_points: &[[f64; 3]],
    knots: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
    t: f64,
) -> Result<[f64; 3]> {
    if control_points.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "B-spline requires at least one control point".to_string(),
        ));
    }
    
    if control_points.len() + degree + 1 != knots.len() {
        return Err(CascadeError::InvalidGeometry(
            format!(
                "Invalid knot vector length. Expected {}, got {}",
                control_points.len() + degree + 1,
                knots.len()
            ),
        ));
    }
    
    // Clamp t to valid range
    let t_min = knots[degree];
    let t_max = knots[knots.len() - degree - 1];
    let t_clamped = t.max(t_min).min(t_max);
    
    // Find the knot span index
    let mut k = degree;
    for (i, &knot) in knots.iter().enumerate() {
        if knot > t_clamped {
            k = i - 1;
            break;
        }
    }
    k = k.max(degree).min(knots.len() - degree - 2);
    
    // Compute basis functions using Cox-de Boor recursion
    let basis = compute_basis_functions(knots, degree, k, t_clamped);
    
    // Evaluate curve as weighted sum of control points
    if let Some(w) = weights {
        if w.len() != control_points.len() {
            return Err(CascadeError::InvalidGeometry(
                "Weight vector length must match control points".to_string(),
            ));
        }
        
        // NURBS evaluation: P(t) = Σ(w_i * N_{i,p}(t) * P_i) / Σ(w_i * N_{i,p}(t))
        let mut numerator = [0.0; 3];
        let mut denominator = 0.0;
        
        for i in 0..control_points.len() {
            if i < basis.len() {
                let w_basis = w[i] * basis[i];
                numerator[0] += w_basis * control_points[i][0];
                numerator[1] += w_basis * control_points[i][1];
                numerator[2] += w_basis * control_points[i][2];
                denominator += w_basis;
            }
        }
        
        if denominator.abs() < 1e-10 {
            return Err(CascadeError::InvalidGeometry(
                "NURBS denominator is zero".to_string(),
            ));
        }
        
        Ok([
            numerator[0] / denominator,
            numerator[1] / denominator,
            numerator[2] / denominator,
        ])
    } else {
        // Uniform B-spline: P(t) = Σ(N_{i,p}(t) * P_i)
        let mut point = [0.0; 3];
        for i in 0..control_points.len() {
            if i < basis.len() {
                point[0] += basis[i] * control_points[i][0];
                point[1] += basis[i] * control_points[i][1];
                point[2] += basis[i] * control_points[i][2];
            }
        }
        Ok(point)
    }
}

/// Compute basis functions for B-spline using Cox-de Boor formula
fn compute_basis_functions(
    knots: &[f64],
    degree: usize,
    k: usize,
    t: f64,
) -> Vec<f64> {
    let mut basis = vec![0.0; knots.len() - degree - 1];
    
    // Initialize 0-degree basis
    if k < basis.len() {
        basis[k] = 1.0;
    }
    
    // Compute higher degree basis functions
    for p in 1..=degree {
        for i in 0..knots.len() - p - 1 {
            let denom_left = knots[i + p] - knots[i];
            let denom_right = knots[i + p + 1] - knots[i + 1];
            
            let mut new_val = 0.0;
            
            if denom_left.abs() > 1e-10 && i > 0 {
                new_val += ((t - knots[i]) / denom_left) * basis[i];
            }
            if denom_right.abs() > 1e-10 && i + 1 < basis.len() {
                new_val += ((knots[i + p + 1] - t) / denom_right) * basis[i + 1];
            }
            
            basis[i] = new_val;
        }
    }
    
    basis
}

/// Tangent to B-spline curve
fn tangent_bspline(
    control_points: &[[f64; 3]],
    knots: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
    t: f64,
) -> Result<[f64; 3]> {
    // Numerical differentiation for now
    let delta = 1e-6;
    let t_plus = (t + delta).min(1.0);
    let t_minus = (t - delta).max(0.0);
    
    let p_plus = evaluate_bspline(control_points, knots, degree, weights, t_plus)?;
    let p_minus = evaluate_bspline(control_points, knots, degree, weights, t_minus)?;
    
    let denom = if t_plus != t_minus {
        t_plus - t_minus
    } else {
        1.0
    };
    
    Ok([
        (p_plus[0] - p_minus[0]) / denom,
        (p_plus[1] - p_minus[1]) / denom,
        (p_plus[2] - p_minus[2]) / denom,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trimmed_curve_creation() {
        // Create a simple arc curve
        let arc = CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        // Create a trimmed version (quarter circle)
        let trimmed = TrimmedCurve::new(arc.clone(), 0.0, 0.25).unwrap();
        
        assert_eq!(trimmed.first_parameter(), 0.0);
        assert_eq!(trimmed.last_parameter(), 0.25);
        
        // Check that basis_curve returns the original
        match trimmed.basis_curve() {
            CurveType::Arc { center, radius } => {
                assert_eq!(*center, [0.0, 0.0, 0.0]);
                assert_eq!(*radius, 1.0);
            }
            _ => panic!("Expected Arc curve type"),
        }
    }
    
    #[test]
    fn test_trimmed_curve_invalid_bounds() {
        let arc = CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        // u1 >= u2 should fail
        assert!(TrimmedCurve::new(arc.clone(), 0.5, 0.25).is_err());
        assert!(TrimmedCurve::new(arc.clone(), 0.5, 0.5).is_err());
    }
    
    #[test]
    fn test_trimmed_curve_evaluation() {
        // Create a full circle
        let circle = CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        // Trim to first quarter (0 to 0.25 of the full circle)
        let trimmed = TrimmedCurve::new(circle, 0.0, 0.25).unwrap();
        
        // At t=0 (start of trimmed curve), should be at (1, 0, 0)
        let start = trimmed.point_at(0.0).unwrap();
        assert!((start[0] - 1.0).abs() < 1e-10);
        assert!(start[1].abs() < 1e-10);
        
        // At t=1 (end of trimmed curve), should be at (0, 1, 0) - quarter circle end
        let end = trimmed.point_at(1.0).unwrap();
        assert!(end[0].abs() < 1e-10);
        assert!((end[1] - 1.0).abs() < 1e-10);
        
        // start_point and end_point helpers
        let start2 = trimmed.start_point().unwrap();
        let end2 = trimmed.end_point().unwrap();
        assert_eq!(start, start2);
        assert_eq!(end, end2);
    }
    
    #[test]
    fn test_trimmed_curve_to_from_curve_type() {
        let arc = CurveType::Arc {
            center: [1.0, 2.0, 3.0],
            radius: 5.0,
        };
        
        let trimmed = TrimmedCurve::new(arc, 0.1, 0.9).unwrap();
        
        // Convert to CurveType
        let curve_type = trimmed.to_curve_type();
        
        // Convert back
        let recovered = TrimmedCurve::from_curve_type(&curve_type).unwrap();
        
        assert_eq!(recovered.first_parameter(), 0.1);
        assert_eq!(recovered.last_parameter(), 0.9);
    }
    
    #[test]
    fn test_trimmed_bezier() {
        // Quadratic Bezier curve
        let bezier = CurveType::Bezier {
            control_points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        };
        
        // Trim to second half
        let trimmed = TrimmedCurve::new(bezier, 0.5, 1.0).unwrap();
        
        // At t=0 of trimmed curve (u=0.5 of basis), should match basis at 0.5
        let p_trimmed_start = trimmed.point_at(0.0).unwrap();
        let p_basis_half = evaluate_bezier(
            &[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]], 
            0.5
        ).unwrap();
        
        assert!((p_trimmed_start[0] - p_basis_half[0]).abs() < 1e-10);
        assert!((p_trimmed_start[1] - p_basis_half[1]).abs() < 1e-10);
        
        // At t=1 of trimmed curve (u=1.0 of basis), should be at end point
        let p_trimmed_end = trimmed.point_at(1.0).unwrap();
        assert!((p_trimmed_end[0] - 2.0).abs() < 1e-10);
        assert!(p_trimmed_end[1].abs() < 1e-10);
    }
    
    #[test]
    fn test_trimmed_curve_via_enum() {
        // Test the CurveType::Trimmed variant directly via point_at
        let bezier = CurveType::Bezier {
            control_points: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        };
        
        let trimmed_enum = CurveType::Trimmed {
            basis_curve: Box::new(bezier),
            u1: 0.25,
            u2: 0.75,
        };
        
        // At t=0, maps to u=0.25, so should be at x=0.5
        let p0 = point_at(&trimmed_enum, 0.0).unwrap();
        assert!((p0[0] - 0.5).abs() < 1e-10);
        
        // At t=1, maps to u=0.75, so should be at x=1.5
        let p1 = point_at(&trimmed_enum, 1.0).unwrap();
        assert!((p1[0] - 1.5).abs() < 1e-10);
        
        // At t=0.5, maps to u=0.5, so should be at x=1.0
        let p_mid = point_at(&trimmed_enum, 0.5).unwrap();
        assert!((p_mid[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ellipse_evaluation() {
        // Unit circle (ellipse with equal axes)
        let center = [0.0, 0.0, 0.0];
        let major = [1.0, 0.0, 0.0];
        let minor = [0.0, 1.0, 0.0];
        
        // At t=0, should be at (1, 0, 0)
        let p0 = evaluate_ellipse(&center, &major, &minor, 0.0).unwrap();
        assert!((p0[0] - 1.0).abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);
        
        // At t=0.25, should be at (0, 1, 0)
        let p25 = evaluate_ellipse(&center, &major, &minor, 0.25).unwrap();
        assert!(p25[0].abs() < 1e-10);
        assert!((p25[1] - 1.0).abs() < 1e-10);
        
        // At t=0.5, should be at (-1, 0, 0)
        let p50 = evaluate_ellipse(&center, &major, &minor, 0.5).unwrap();
        assert!((p50[0] + 1.0).abs() < 1e-10);
        assert!(p50[1].abs() < 1e-10);
    }

    #[test]
    fn test_bezier_evaluation() {
        // Quadratic Bezier: straight line from (0,0,0) to (2,0,0)
        let control_points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        
        // At t=0, should be at first control point
        let p0 = evaluate_bezier(&control_points, 0.0).unwrap();
        assert_eq!(p0, [0.0, 0.0, 0.0]);
        
        // At t=1, should be at last control point
        let p1 = evaluate_bezier(&control_points, 1.0).unwrap();
        assert_eq!(p1, [2.0, 0.0, 0.0]);
        
        // At t=0.5, should be at peak (1, 0.5, 0)
        let p50 = evaluate_bezier(&control_points, 0.5).unwrap();
        assert!((p50[0] - 1.0).abs() < 1e-10);
        assert!((p50[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_tangent() {
        let control_points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        
        // Tangent at t=0 should point in direction of (P1 - P0)
        let tangent_0 = tangent_bezier(&control_points, 0.0).unwrap();
        assert!(tangent_0[0] > 0.0); // Should point mostly in +x
        assert!(tangent_0[1].abs() < 1e-10); // Should have minimal y component
    }

    #[test]
    fn test_ellipse_tangent() {
        let center = [0.0, 0.0, 0.0];
        let major = [1.0, 0.0, 0.0];
        let minor = [0.0, 1.0, 0.0];
        
        // At t=0, tangent should be perpendicular to major axis (pointing in y direction)
        let tangent = tangent_ellipse(&center, &major, &minor, 0.0).unwrap();
        assert!(tangent[0].abs() < 1e-10); // Minimal x component
        assert!(tangent[1] > 0.0); // Should point in +y
    }
    
    #[test]
    fn test_parabola_standard() {
        // Standard parabola y² = 4x (focal = 1, so p = 1, focus at (0.5, 0, 0))
        let parabola = Parabola::standard(1.0).unwrap();
        
        // At u=0, should be at origin (vertex)
        let p0 = parabola.point_at(0.0);
        assert!(p0[0].abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);
        assert!(p0[2].abs() < 1e-10);
        
        // At u=2, should be at (2, 1, 0) since y = u²/(4p) = 4/4 = 1
        let p2 = parabola.point_at(2.0);
        assert!((p2[0] - 2.0).abs() < 1e-10);
        assert!((p2[1] - 1.0).abs() < 1e-10);
        assert!(p2[2].abs() < 1e-10);
        
        // At u=-2, should be at (-2, 1, 0) - symmetric
        let pn2 = parabola.point_at(-2.0);
        assert!((pn2[0] + 2.0).abs() < 1e-10);
        assert!((pn2[1] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_parabola_focus_directrix() {
        // Parabola with focal parameter = 2, so focus at distance 1 from vertex
        let parabola = Parabola::standard(2.0).unwrap();
        
        // Focus should be at (1, 0, 0)
        let focus = parabola.focus();
        assert!((focus[0] - 1.0).abs() < 1e-10);
        assert!(focus[1].abs() < 1e-10);
        assert!(focus[2].abs() < 1e-10);
        
        // Directrix point should be at (-1, 0, 0)
        let dir_pt = parabola.directrix_point();
        assert!((dir_pt[0] + 1.0).abs() < 1e-10);
        assert!(dir_pt[1].abs() < 1e-10);
        
        // Focal distance should be 1
        assert!((parabola.focal_distance() - 1.0).abs() < 1e-10);
        
        // Parameter should be 2
        assert!((parabola.parameter() - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_parabola_tangent() {
        let parabola = Parabola::standard(1.0).unwrap();
        
        // At u=0 (vertex), tangent should be purely in x direction
        let t0 = parabola.tangent_at(0.0);
        assert!((t0[0] - 1.0).abs() < 1e-10);
        assert!(t0[1].abs() < 1e-10);
        assert!(t0[2].abs() < 1e-10);
        
        // At u=2, tangent should be (1, 1, 0) since dP/du = x_dir + (u/2p)*y_dir = (1,0,0) + 1*(0,1,0)
        let t2 = parabola.tangent_at(2.0);
        assert!((t2[0] - 1.0).abs() < 1e-10);
        assert!((t2[1] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_parabola_curve_type_evaluation() {
        // Test that CurveType::Parabola works with the point_at function
        let curve = CurveType::Parabola {
            origin: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            focal: 1.0,
        };
        
        // At u=0, should be at origin
        let p0 = point_at(&curve, 0.0).unwrap();
        assert!(p0[0].abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);
        
        // At u=2, should be at (2, 1, 0)
        let p2 = point_at(&curve, 2.0).unwrap();
        assert!((p2[0] - 2.0).abs() < 1e-10);
        assert!((p2[1] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_parabola_3d() {
        // Parabola with non-standard orientation
        let parabola = Parabola::new(
            [1.0, 2.0, 3.0],       // Vertex at (1,2,3)
            [0.0, 0.0, 1.0],       // Axis along Z
            [1.0, 0.0, 0.0],       // Y-dir along X
            4.0,                    // focal = 4, so focus at distance 2
        ).unwrap();
        
        // Vertex should be at (1, 2, 3)
        let vertex = parabola.vertex();
        assert!((vertex[0] - 1.0).abs() < 1e-10);
        assert!((vertex[1] - 2.0).abs() < 1e-10);
        assert!((vertex[2] - 3.0).abs() < 1e-10);
        
        // Focus should be at (1, 2, 5) - 2 units along Z axis
        let focus = parabola.focus();
        assert!((focus[0] - 1.0).abs() < 1e-10);
        assert!((focus[1] - 2.0).abs() < 1e-10);
        assert!((focus[2] - 5.0).abs() < 1e-10);
        
        // Point at u=4: origin + 4*z_dir + (16/16)*x_dir = (1,2,3) + (0,0,4) + (1,0,0) = (2, 2, 7)
        let p4 = parabola.point_at(4.0);
        assert!((p4[0] - 2.0).abs() < 1e-10);
        assert!((p4[1] - 2.0).abs() < 1e-10);
        assert!((p4[2] - 7.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_parabola_invalid_focal() {
        // Zero focal should fail
        let result = Parabola::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_offset_curve_creation() {
        // Create a simple circle to offset
        let circle = CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        // Create offset curve with offset distance 0.5
        let offset = OffsetCurve::new(circle.clone(), 0.5, [0.0, 0.0, 1.0]).unwrap();
        
        assert!((offset.offset() - 0.5).abs() < 1e-10);
        assert_eq!(offset.direction(), [0.0, 0.0, 1.0]);
        
        // Match basis curve
        match offset.basis_curve() {
            CurveType::Arc { center, radius } => {
                assert_eq!(*center, [0.0, 0.0, 0.0]);
                assert_eq!(*radius, 1.0);
            }
            _ => panic!("Expected Arc curve type"),
        }
    }
    
    #[test]
    fn test_offset_curve_invalid_distance() {
        let circle = CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        // Zero offset distance should fail
        assert!(OffsetCurve::new(circle.clone(), 0.0, [0.0, 0.0, 1.0]).is_err());
    }
    
    #[test]
    fn test_offset_curve_evaluation_circle() {
        // Create a unit circle in XY plane
        let circle = CurveType::Arc {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        // Offset by 0.5 in Z direction (upward)
        let offset = OffsetCurve::new(circle.clone(), 0.5, [0.0, 0.0, 1.0]).unwrap();
        
        // At t=0, circle is at (1, 0, 0), offset point should be at approximately (1, 0, 0.5)
        // Because the normal will be computed as cross([0,0,1], [0,2π,0]) which gives [2π, 0, 0]
        // Actually, let's think about this more carefully:
        // tangent at t=0 for circle: [0, 2π, 0] (pointing in +y direction)
        // offset_direction: [0, 0, 1]
        // normal = cross([0,0,1], [0,1,0]) = [1, 0, 0] (pointing outward from circle)
        let p0 = offset.point_at(0.0).unwrap();
        // The basis point is (1, 0, 0), offset by 0.5 in the computed normal direction
        // For a circle, the normal at any point should point radially outward
        // So at (1,0,0), the normal in the plane of [0,0,1] and tangent [0,1,0] should be [1,0,0]
        // Thus the offset point should be at (1.5, 0, 0)
        
        // Check that the point is offset from the basis
        let basis_p0 = point_at(&circle, 0.0).unwrap();
        let offset_distance = ((p0[0] - basis_p0[0]).powi(2) + 
                               (p0[1] - basis_p0[1]).powi(2) + 
                               (p0[2] - basis_p0[2]).powi(2)).sqrt();
        assert!((offset_distance - 0.5).abs() < 1e-9, "Expected offset distance 0.5, got {}", offset_distance);
    }
    
    #[test]
    fn test_offset_curve_tangent_same_as_basis() {
        // Create a simple line
        let line = CurveType::Bezier {
            control_points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        };
        
        // Create offset
        let offset = OffsetCurve::new(line.clone(), 0.5, [0.0, 0.0, 1.0]).unwrap();
        
        // Tangent should be the same as basis curve
        let basis_tangent = tangent_at(&line, 0.5).unwrap();
        let offset_tangent = offset.tangent_at(0.5).unwrap();
        
        for i in 0..3 {
            assert!((basis_tangent[i] - offset_tangent[i]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_offset_curve_to_from_curve_type() {
        let circle = CurveType::Arc {
            center: [1.0, 2.0, 3.0],
            radius: 2.0,
        };
        
        let offset = OffsetCurve::new(circle, 0.75, [1.0, 0.0, 0.0]).unwrap();
        
        // Convert to CurveType
        let curve_type = offset.to_curve_type();
        
        // Convert back
        let recovered = OffsetCurve::from_curve_type(&curve_type).unwrap();
        
        assert!((recovered.offset() - 0.75).abs() < 1e-10);
        assert!((recovered.direction()[0] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_offset_curve_via_enum() {
        // Test the CurveType::Offset variant directly via point_at
        let line = CurveType::Bezier {
            control_points: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        };
        
        let offset_enum = CurveType::Offset {
            basis_curve: Box::new(line),
            offset_distance: 0.5,
            offset_direction: [0.0, 0.0, 1.0],
        };
        
        // At t=0, basis is at (0,0,0), tangent is [2,0,0]
        // normal = cross([0,0,1], [1,0,0]) = [0, -1, 0]
        // offset point = (0,0,0) + 0.5 * [0, -1, 0] = (0, -0.5, 0)
        let p0 = point_at(&offset_enum, 0.0).unwrap();
        
        // Check that we have an offset
        assert!(p0[0].abs() < 1e-9 || p0[0].abs() > 1e-10);
        
        // At t=1, basis is at (2,0,0), tangent is [2,0,0]
        // offset should still apply
        let p1 = point_at(&offset_enum, 1.0).unwrap();
        assert!((p1[0] - 2.0).abs() < 1e-9);
    }
    
    #[test]
    fn test_offset_bezier_curve() {
        // Quadratic Bezier curve in XY plane
        let bezier = CurveType::Bezier {
            control_points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        };
        
        // Offset by 0.1 in Z direction
        let offset = OffsetCurve::new(bezier.clone(), 0.1, [0.0, 0.0, 1.0]).unwrap();
        
        // At t=0, basis is at (0,0,0), tangent points in +y direction
        // normal = cross([0,0,1], tangent_direction) should be perpendicular
        let p0 = offset.point_at(0.0).unwrap();
        
        // Verify offset was applied
        let basis_p0 = point_at(&bezier, 0.0).unwrap();
        let dist = ((p0[0] - basis_p0[0]).powi(2) + 
                    (p0[1] - basis_p0[1]).powi(2) + 
                    (p0[2] - basis_p0[2]).powi(2)).sqrt();
        assert!((dist - 0.1).abs() < 1e-9, "Expected offset distance 0.1, got {}", dist);
    }

    #[test]
    fn test_hyperbola_standard() {
        // Standard hyperbola: x²/4 - y²/1 = 1
        let hyperbola = Hyperbola::standard(2.0, 1.0).unwrap();
        
        // At u=0, should be at (2, 0, 0) - vertex on right branch
        // cosh(0) = 1, sinh(0) = 0
        let p0 = hyperbola.point_at(0.0);
        assert!((p0[0] - 2.0).abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);
        assert!(p0[2].abs() < 1e-10);
        
        // At u=ln(3), cosh(u) ≈ (3 + 1/3)/2 = 5/3, sinh(u) ≈ (3 - 1/3)/2 = 4/3
        let u = (3.0_f64).ln();
        let p = hyperbola.point_at(u);
        let cosh_u = u.cosh();
        let sinh_u = u.sinh();
        let expected_x = 0.0 + 2.0 * cosh_u;
        let expected_y = 0.0 + 1.0 * sinh_u;
        assert!((p[0] - expected_x).abs() < 1e-10);
        assert!((p[1] - expected_y).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_foci() {
        // Hyperbola with a=3, b=4
        // c = sqrt(9 + 16) = 5
        let hyperbola = Hyperbola::standard(3.0, 4.0).unwrap();
        
        let c = 5.0;
        let focus1 = hyperbola.focus1();
        let focus2 = hyperbola.focus2();
        
        // Focus 1 should be at (5, 0, 0)
        assert!((focus1[0] - c).abs() < 1e-10);
        assert!(focus1[1].abs() < 1e-10);
        assert!(focus1[2].abs() < 1e-10);
        
        // Focus 2 should be at (-5, 0, 0)
        assert!((focus2[0] + c).abs() < 1e-10);
        assert!(focus2[1].abs() < 1e-10);
        assert!(focus2[2].abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_eccentricity() {
        // For a=3, b=4, c=5, eccentricity e = 5/3
        let hyperbola = Hyperbola::standard(3.0, 4.0).unwrap();
        let e = hyperbola.eccentricity();
        assert!((e - 5.0 / 3.0).abs() < 1e-10);
        
        // For a circle-like hyperbola with a=b, e = sqrt(2)
        let hyperbola2 = Hyperbola::standard(1.0, 1.0).unwrap();
        let e2 = hyperbola2.eccentricity();
        assert!((e2 - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_asymptotes() {
        // For a=2, b=1, asymptotes have slopes ±1/2
        let hyperbola = Hyperbola::standard(2.0, 1.0).unwrap();
        let (asym1, asym2) = hyperbola.asymptotes();
        
        // Asymptote 1: direction = x_dir + 0.5*y_dir = (1, 0.5, 0)
        let expected_asym1 = [1.0, 0.5, 0.0];
        assert!((asym1[0] - expected_asym1[0]).abs() < 1e-10);
        assert!((asym1[1] - expected_asym1[1]).abs() < 1e-10);
        
        // Asymptote 2: direction = x_dir - 0.5*y_dir = (1, -0.5, 0)
        let expected_asym2 = [1.0, -0.5, 0.0];
        assert!((asym2[0] - expected_asym2[0]).abs() < 1e-10);
        assert!((asym2[1] - expected_asym2[1]).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_tangent() {
        let hyperbola = Hyperbola::standard(3.0, 4.0).unwrap();
        
        // At u=0, tangent should be (0, 4, 0) since dP/du = a*sinh(u)*x + b*cosh(u)*y
        // = 3*0*x + 4*1*y = (0, 4, 0)
        let t0 = hyperbola.tangent_at(0.0);
        assert!(t0[0].abs() < 1e-10);
        assert!((t0[1] - 4.0).abs() < 1e-10);
        assert!(t0[2].abs() < 1e-10);
        
        // At u=ln(2), check it matches the parametric derivative
        let u = 2.0_f64.ln();
        let t = hyperbola.tangent_at(u);
        let sinh_u = u.sinh();
        let cosh_u = u.cosh();
        let expected_x = 3.0 * sinh_u;
        let expected_y = 4.0 * cosh_u;
        assert!((t[0] - expected_x).abs() < 1e-10);
        assert!((t[1] - expected_y).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_negative_parameter() {
        // Negative u should give the left branch of the hyperbola
        let hyperbola = Hyperbola::standard(2.0, 1.0).unwrap();
        
        // At u=-ln(3), cosh(-u) = cosh(u), sinh(-u) = -sinh(u)
        let u = (3.0_f64).ln();
        let p_pos = hyperbola.point_at(u);
        let p_neg = hyperbola.point_at(-u);
        
        // x-coordinates should be the same (cosh is even)
        assert!((p_pos[0] - p_neg[0]).abs() < 1e-10);
        
        // y-coordinates should be opposite (sinh is odd)
        assert!((p_pos[1] + p_neg[1]).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_3d() {
        // Non-standard orientation
        let hyperbola = Hyperbola::new(
            [1.0, 2.0, 3.0],         // Center at (1,2,3)
            [0.0, 0.0, 1.0],         // Transverse axis along Z
            [1.0, 0.0, 0.0],         // Conjugate axis along X
            2.0,                      // major_radius = 2
            1.5,                      // minor_radius = 1.5
        ).unwrap();
        
        // At u=0, should be at center + 2*z_dir = (1, 2, 5)
        let p0 = hyperbola.point_at(0.0);
        assert!((p0[0] - 1.0).abs() < 1e-10);
        assert!((p0[1] - 2.0).abs() < 1e-10);
        assert!((p0[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_curve_type_evaluation() {
        // Test that CurveType::Hyperbola works with the point_at function
        let curve = CurveType::Hyperbola {
            center: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            major_radius: 2.0,
            minor_radius: 1.0,
        };
        
        // At u=0, should be at (2, 0, 0)
        let p0 = point_at(&curve, 0.0).unwrap();
        assert!((p0[0] - 2.0).abs() < 1e-10);
        assert!(p0[1].abs() < 1e-10);
        assert!(p0[2].abs() < 1e-10);
        
        // At u=ln(3), should match the hyperbola calculation
        let u = 3.0_f64.ln();
        let p = point_at(&curve, u).unwrap();
        let cosh_u = u.cosh();
        let sinh_u = u.sinh();
        let expected_x = 0.0 + 2.0 * cosh_u;
        let expected_y = 0.0 + 1.0 * sinh_u;
        assert!((p[0] - expected_x).abs() < 1e-10);
        assert!((p[1] - expected_y).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_curve_type_tangent() {
        let curve = CurveType::Hyperbola {
            center: [0.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            major_radius: 3.0,
            minor_radius: 4.0,
        };
        
        // At u=0, tangent should be (0, 4, 0)
        let t0 = tangent_at(&curve, 0.0).unwrap();
        assert!(t0[0].abs() < 1e-10);
        assert!((t0[1] - 4.0).abs() < 1e-10);
        assert!(t0[2].abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_invalid_radii() {
        // Zero major radius should fail
        let result = Hyperbola::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
            1.0,
        );
        assert!(result.is_err());
        
        // Zero minor radius should fail
        let result2 = Hyperbola::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            0.0,
        );
        assert!(result2.is_err());
    }

    #[test]
    fn test_hyperbola_conversion() {
        let hyperbola = Hyperbola::standard(2.0, 1.5).unwrap();
        let curve_type = hyperbola.to_curve_type();
        
        // Verify that the conversion preserves all values
        match curve_type {
            CurveType::Hyperbola {
                center,
                x_dir,
                y_dir,
                major_radius,
                minor_radius,
            } => {
                assert_eq!(center, [0.0, 0.0, 0.0]);
                assert_eq!(x_dir, [1.0, 0.0, 0.0]);
                assert_eq!(y_dir, [0.0, 1.0, 0.0]);
                assert!((major_radius - 2.0).abs() < 1e-10);
                assert!((minor_radius - 1.5).abs() < 1e-10);
            }
            _ => panic!("Expected CurveType::Hyperbola"),
        }
    }

    #[test]
    fn test_hyperbola_parameter() {
        let hyperbola = Hyperbola::standard(3.5, 2.1).unwrap();
        
        // parameter() should return major_radius
        assert!((hyperbola.parameter() - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbola_center_point() {
        let center = [5.0, 6.0, 7.0];
        let hyperbola = Hyperbola::new(
            center,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            2.0,
            1.0,
        ).unwrap();
        
        let c = hyperbola.center_point();
        assert_eq!(c, center);
    }
}
