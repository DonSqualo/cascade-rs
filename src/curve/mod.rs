//! Curve types and evaluation functions

use crate::{Result, CascadeError, brep::CurveType};

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
        CurveType::Bezier { control_points } => {
            evaluate_bezier(control_points, t)
        }
        CurveType::BSpline { control_points, knots, degree, weights } => {
            evaluate_bspline(control_points, knots, *degree, weights.as_deref(), t)
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
        CurveType::Bezier { control_points } => {
            tangent_bezier(control_points, t)
        }
        CurveType::BSpline { control_points, knots, degree, weights } => {
            tangent_bspline(control_points, knots, *degree, weights.as_deref(), t)
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
}
