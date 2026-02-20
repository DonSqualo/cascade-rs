//! Curve and surface approximation algorithms
//!
//! This module provides least-squares fitting functions to approximate
//! geometric curves and surfaces with BSplines.

use crate::{Result, CascadeError, brep::CurveType};

/// Approximate a curve with a BSpline using least squares fitting
///
/// This function fits a BSpline curve to a set of 3D points, controlling the
/// number of control points based on the specified tolerance.
///
/// # Algorithm
/// - Uses least squares fitting with uniform knot vector
/// - Control points are determined by solving the normal equations
/// - The number of control points is adjusted based on tolerance
///
/// # Arguments
/// * `points` - Array of 3D points to fit (at least 2 points required)
/// * `degree` - Degree of the BSpline (typically 2-5, must be < number of control points)
/// * `tolerance` - Maximum deviation allowed (controls number of control points)
///
/// # Returns
/// A BSpline curve approximating the points, or an error if the input is invalid
///
/// # Example
/// ```no_run
/// use cascade_rs::approx::approximate_curve;
///
/// let points = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 1.0, 0.0],
///     [2.0, 0.0, 0.0],
///     [3.0, 1.0, 0.0],
/// ];
///
/// let curve = approximate_curve(&points, 3, 0.01)?;
/// # Ok::<_, cascade_rs::CascadeError>(())
/// ```
pub fn approximate_curve(
    points: &[[f64; 3]],
    degree: usize,
    tolerance: f64,
) -> Result<CurveType> {
    // Validate input
    if points.len() < 2 {
        return Err(CascadeError::InvalidGeometry(
            "At least 2 points required for curve approximation".to_string(),
        ));
    }

    if degree < 1 {
        return Err(CascadeError::InvalidGeometry(
            "Degree must be at least 1".to_string(),
        ));
    }

    if tolerance <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Tolerance must be positive".to_string(),
        ));
    }

    let n_points = points.len();
    
    // Determine optimal number of control points based on tolerance
    // Start with degree + 1 minimum, adjust based on tolerance
    let mut n_control = (degree + 1).max(4);
    
    // For small tolerances or many input points, use more control points
    if tolerance < 0.001 || n_points > 50 {
        n_control = ((n_points as f64) * 0.5) as usize;
        n_control = n_control.max(degree + 1).min(n_points - 1);
    } else if tolerance < 0.01 {
        n_control = ((n_points as f64) * 0.3) as usize;
        n_control = n_control.max(degree + 1).min(n_points - 1);
    }

    // Ensure control points don't exceed input points
    n_control = n_control.min(n_points);

    // Validate degree vs number of control points
    if degree >= n_control {
        return Err(CascadeError::InvalidGeometry(format!(
            "Degree ({}) must be less than number of control points ({})",
            degree, n_control
        )));
    }

    // Create uniform knot vector
    let knots = create_uniform_knot_vector(n_control, degree);

    // Perform least squares fitting
    let control_points = fit_control_points(points, n_control, degree, &knots)?;

    Ok(CurveType::BSpline {
        control_points,
        knots,
        degree,
        weights: None,
    })
}

/// Create a uniform (clamped) knot vector
fn create_uniform_knot_vector(n_control: usize, degree: usize) -> Vec<f64> {
    let mut knots = Vec::new();
    
    // Clamp at start (degree+1 repeated knots)
    for _ in 0..=degree {
        knots.push(0.0);
    }
    
    // Interior knots, uniformly spaced
    let n_interior = n_control - degree - 1;
    if n_interior > 0 {
        for i in 1..=n_interior {
            knots.push(i as f64 / (n_interior + 1) as f64);
        }
    }
    
    // Clamp at end (degree+1 repeated knots)
    for _ in 0..=degree {
        knots.push(1.0);
    }
    
    knots
}

/// Fit control points using least squares
fn fit_control_points(
    points: &[[f64; 3]],
    n_control: usize,
    degree: usize,
    knots: &[f64],
) -> Result<Vec<[f64; 3]>> {
    let n_points = points.len();
    
    // Map input points to parameters uniformly
    let params: Vec<f64> = (0..n_points)
        .map(|i| i as f64 / (n_points - 1) as f64)
        .collect();
    
    // Build basis function matrix N (n_points x n_control)
    // N[i][j] = N_{j,degree}(params[i])
    let mut matrix = vec![vec![0.0; n_control]; n_points];
    
    for (i, &param) in params.iter().enumerate() {
        for j in 0..n_control {
            matrix[i][j] = bspline_basis(j, degree, param, knots);
        }
    }
    
    // Solve normal equations: N^T * N * P = N^T * points
    // For each coordinate (x, y, z)
    let mut control_points = vec![[0.0; 3]; n_control];
    
    for coord in 0..3 {
        // Extract coordinate values
        let point_coords: Vec<f64> = points.iter().map(|p| p[coord]).collect();
        
        // Build normal matrix: A = N^T * N
        let mut normal_matrix = vec![vec![0.0; n_control]; n_control];
        for i in 0..n_control {
            for j in 0..n_control {
                let mut sum = 0.0;
                for k in 0..n_points {
                    sum += matrix[k][i] * matrix[k][j];
                }
                normal_matrix[i][j] = sum;
            }
        }
        
        // Build right-hand side: b = N^T * points[coord]
        let mut rhs = vec![0.0; n_control];
        for i in 0..n_control {
            let mut sum = 0.0;
            for k in 0..n_points {
                sum += matrix[k][i] * point_coords[k];
            }
            rhs[i] = sum;
        }
        
        // Solve using Gauss elimination with partial pivoting
        let solution = solve_linear_system(&normal_matrix, &rhs)?;
        
        for i in 0..n_control {
            control_points[i][coord] = solution[i];
        }
    }
    
    Ok(control_points)
}

/// Evaluate B-spline basis function N_{i,p}(u)
fn bspline_basis(i: usize, p: usize, u: f64, knots: &[f64]) -> f64 {
    if p == 0 {
        // Base case: N_{i,0}(u) = 1 if knots[i] <= u < knots[i+1], else 0
        if i < knots.len() - 1 {
            let at_last = u == knots[knots.len() - 1] && knots[i + 1] == u;
            if (knots[i] <= u && u < knots[i + 1]) || at_last {
                return 1.0;
            }
        }
        return 0.0;
    }
    
    // Recursive case: use Cox-de Boor recurrence
    let mut result = 0.0;
    
    // First term
    if knots[i + p] > knots[i] {
        let left = bspline_basis(i, p - 1, u, knots);
        result += ((u - knots[i]) / (knots[i + p] - knots[i])) * left;
    }
    
    // Second term
    if knots[i + p + 1] > knots[i + 1] {
        let right = bspline_basis(i + 1, p - 1, u, knots);
        result += ((knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1])) * right;
    }
    
    result
}

/// Solve Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    if n == 0 || a[0].len() != n || b.len() != n {
        return Err(CascadeError::InvalidGeometry(
            "Invalid matrix dimensions for linear system solver".to_string(),
        ));
    }
    
    // Copy matrix for modification
    let mut matrix = a.to_vec();
    let mut rhs = b.to_vec();
    
    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = matrix[col][col].abs();
        
        for row in (col + 1)..n {
            if matrix[row][col].abs() > max_val {
                max_val = matrix[row][col].abs();
                max_row = row;
            }
        }
        
        // Check for singular matrix
        if max_val < 1e-14 {
            return Err(CascadeError::InvalidGeometry(
                "Singular matrix encountered in linear system solver".to_string(),
            ));
        }
        
        // Swap rows
        if max_row != col {
            matrix.swap(col, max_row);
            rhs.swap(col, max_row);
        }
        
        // Eliminate column
        for row in (col + 1)..n {
            let factor = matrix[row][col] / matrix[col][col];
            for j in col..n {
                matrix[row][j] -= factor * matrix[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    
    // Back substitution
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= matrix[i][j] * solution[j];
        }
        solution[i] = sum / matrix[i][i];
    }
    
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximate_line() -> Result<()> {
        // Approximate a line with BSpline
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        
        let curve = approximate_curve(&points, 1, 0.01)?;
        
        // Verify we get a BSpline
        match curve {
            CurveType::BSpline { ref control_points, ref knots, degree, .. } => {
                assert_eq!(degree, 1);
                assert!(control_points.len() >= 2);
                assert!(knots.len() > 0);
                // Line approximation should have relatively small control point count
                assert!(control_points.len() <= points.len());
            }
            _ => panic!("Expected BSpline"),
        }
        
        Ok(())
    }

    #[test]
    fn test_approximate_cubic_with_tight_tolerance() -> Result<()> {
        // Approximate with tight tolerance should use more control points
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        
        let tight = approximate_curve(&points, 3, 0.0001)?;
        let loose = approximate_curve(&points, 3, 0.1)?;
        
        let tight_count = match &tight {
            CurveType::BSpline { control_points, .. } => control_points.len(),
            _ => panic!("Expected BSpline"),
        };
        
        let loose_count = match &loose {
            CurveType::BSpline { control_points, .. } => control_points.len(),
            _ => panic!("Expected BSpline"),
        };
        
        // Tighter tolerance should use more or equal control points
        assert!(tight_count >= loose_count);
        
        Ok(())
    }

    #[test]
    fn test_approximate_invalid_input() {
        // Too few points
        let result = approximate_curve(&[[0.0; 3]], 1, 0.01);
        assert!(result.is_err());
        
        // Invalid degree
        let points = vec![[0.0; 3], [1.0; 3]];
        let result = approximate_curve(&points, 0, 0.01);
        assert!(result.is_err());
        
        // Invalid tolerance
        let result = approximate_curve(&points, 1, -0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_degree_vs_control_points() -> Result<()> {
        let points: Vec<_> = (0..10)
            .map(|i| [(i as f64) * 0.1, ((i as f64) * 0.1).sin(), 0.0])
            .collect();
        
        // Degree too high relative to points should fail
        let result = approximate_curve(&points, 15, 0.01);
        assert!(result.is_err());
        
        Ok(())
    }

    #[test]
    fn test_sinusoidal_curve() -> Result<()> {
        // Generate points on a sinusoidal curve
        let points: Vec<_> = (0..20)
            .map(|i| {
                let t = i as f64 / 19.0;
                let x = t * 4.0 * std::f64::consts::PI;
                [t, x.sin(), 0.0]
            })
            .collect();
        
        let curve = approximate_curve(&points, 3, 0.01)?;
        
        // Verify curve was created
        match curve {
            CurveType::BSpline { control_points, degree, .. } => {
                assert_eq!(degree, 3);
                assert!(control_points.len() >= 4);
                assert!(control_points.len() <= points.len());
            }
            _ => panic!("Expected BSpline"),
        }
        
        Ok(())
    }

    #[test]
    fn test_knot_vector_properties() -> Result<()> {
        let n_control = 5;
        let degree = 3;
        let knots = create_uniform_knot_vector(n_control, degree);
        
        // Check knot vector length: n_control + degree + 1
        assert_eq!(knots.len(), n_control + degree + 1);
        
        // Check clamping at start
        for i in 0..=degree {
            assert_eq!(knots[i], 0.0, "Start knots should be clamped to 0");
        }
        
        // Check clamping at end
        for i in (knots.len() - degree - 1)..knots.len() {
            assert_eq!(knots[i], 1.0, "End knots should be clamped to 1");
        }
        
        // Check monotonicity
        for i in 0..knots.len() - 1 {
            assert!(knots[i] <= knots[i + 1], "Knots should be non-decreasing");
        }
        
        Ok(())
    }
}
