//! Curve types and evaluation functions

use crate::{Result, CascadeError, brep::CurveType};

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
}
