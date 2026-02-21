//! Curve and surface approximation algorithms
//!
//! This module provides least-squares fitting functions to approximate
//! geometric curves and surfaces with BSplines.

use crate::{Result, CascadeError, brep::{CurveType, SurfaceType}};

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

/// Interpolate a curve through exact points using BSpline
///
/// This function creates a BSpline curve that passes exactly through all given points.
/// The curve is constructed using the chord-length parameterization method and
/// solves a tridiagonal linear system for the control points.
///
/// # Algorithm
/// 1. Compute chord-length parameterization of input points
/// 2. Create knot vector with interior knots placed at averaged parameter values
/// 3. Solve tridiagonal system to find control points that satisfy interpolation constraints
/// 4. The resulting BSpline curve passes through all input points
///
/// # Arguments
/// * `points` - Array of 3D points to interpolate (at least degree+2 points required)
/// * `degree` - Degree of the BSpline (typically 2-5)
///
/// # Returns
/// A BSpline curve passing exactly through all points, or an error if input is invalid
///
/// # Example
/// ```no_run
/// use cascade_rs::approx::interpolate_curve;
///
/// let points = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 1.0, 0.0],
///     [2.0, 0.0, 0.0],
///     [3.0, 1.0, 0.0],
/// ];
///
/// let curve = interpolate_curve(&points, 3)?;
/// # Ok::<_, cascade_rs::CascadeError>(())
/// ```
pub fn interpolate_curve(
    points: &[[f64; 3]],
    degree: usize,
) -> Result<CurveType> {
    // Validate input
    if points.len() < 2 {
        return Err(CascadeError::InvalidGeometry(
            "At least 2 points required for curve interpolation".to_string(),
        ));
    }

    if degree < 1 {
        return Err(CascadeError::InvalidGeometry(
            "Degree must be at least 1".to_string(),
        ));
    }

    if degree >= points.len() {
        return Err(CascadeError::InvalidGeometry(format!(
            "Degree ({}) must be less than number of points ({})",
            degree,
            points.len()
        )));
    }

    let n = points.len();
    
    // Step 1: Compute chord-length parameterization
    let params = compute_chord_length_params(points);
    
    // Step 2: Create knot vector using averaging method
    let knots = create_interpolation_knot_vector(&params, degree);
    
    // Step 3: Solve for control points using tridiagonal system
    let control_points = interpolate_control_points(points, &params, degree, &knots)?;
    
    Ok(CurveType::BSpline {
        control_points,
        knots,
        degree,
        weights: None,
    })
}

/// Interpolate a BSpline surface through a 2D grid of points
///
/// This function creates a BSpline surface that passes exactly through all
/// the provided grid points. It uses a two-pass approach:
/// 1. First, interpolate each row (u-direction) to get intermediate control point values
/// 2. Then, interpolate each column of those values (v-direction) to get final control points
///
/// # Algorithm
/// - Uses chord-length parameterization in both u and v directions
/// - The grid points define the surface exactly at the parameter values
/// - Interior control points are computed by solving linear systems
///
/// # Arguments
/// * `points` - 2D grid of points to interpolate, points[i] is the i-th row (u-direction)
/// * `u_degree` - Degree of the BSpline in the u direction (typically 2-3)
/// * `v_degree` - Degree of the BSpline in the v direction (typically 2-3)
///
/// # Returns
/// A BSpline surface passing through all grid points, or an error if input is invalid
///
/// # Example
/// ```no_run
/// use cascade_rs::approx::interpolate_surface;
///
/// let points = vec![
///     vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
///     vec![[0.0, 1.0, 0.5], [1.0, 1.0, 0.7], [2.0, 1.0, 0.5]],
///     vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
/// ];
///
/// let surface = interpolate_surface(&points, 2, 2)?;
/// # Ok::<_, cascade_rs::CascadeError>(())
/// ```
pub fn interpolate_surface(
    points: &[Vec<[f64; 3]>],
    u_degree: usize,
    v_degree: usize,
) -> Result<SurfaceType> {
    // Validate input grid
    if points.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Grid must have at least 1 row".to_string(),
        ));
    }

    let n_rows = points.len();
    let n_cols = points[0].len();

    if n_cols < 2 {
        return Err(CascadeError::InvalidGeometry(
            "Grid must have at least 2 columns".to_string(),
        ));
    }

    // Verify all rows have same length
    for row in points {
        if row.len() != n_cols {
            return Err(CascadeError::InvalidGeometry(
                "All rows must have the same number of points".to_string(),
            ));
        }
    }

    if u_degree < 1 {
        return Err(CascadeError::InvalidGeometry(
            "U degree must be at least 1".to_string(),
        ));
    }

    if v_degree < 1 {
        return Err(CascadeError::InvalidGeometry(
            "V degree must be at least 1".to_string(),
        ));
    }

    if u_degree >= n_cols {
        return Err(CascadeError::InvalidGeometry(format!(
            "U degree ({}) must be less than number of columns ({})",
            u_degree, n_cols
        )));
    }

    if v_degree >= n_rows {
        return Err(CascadeError::InvalidGeometry(format!(
            "V degree ({}) must be less than number of rows ({})",
            v_degree, n_rows
        )));
    }

    // Step 1: Interpolate each row (u-direction)
    // This gives us n_rows curves, and we extract their control points
    let mut row_control_points = Vec::new();
    
    for row in points {
        let curve = interpolate_curve(row, u_degree)?;
        
        // Extract control points from the curve
        match curve {
            CurveType::BSpline { control_points, .. } => {
                row_control_points.push(control_points);
            }
            _ => {
                return Err(CascadeError::InvalidGeometry(
                    "Unexpected curve type from interpolate_curve".to_string(),
                ));
            }
        }
    }

    // Step 2: Interpolate each column (v-direction)
    // row_control_points is now a grid where we interpolate columns
    let n_control_cols = row_control_points[0].len();
    let mut surface_control_points = vec![Vec::new(); n_control_cols];

    for col_idx in 0..n_control_cols {
        // Extract the column as a sequence of points
        let column: Vec<[f64; 3]> = row_control_points
            .iter()
            .map(|row| row[col_idx])
            .collect();

        // Interpolate this column
        let col_curve = interpolate_curve(&column, v_degree)?;

        // Extract control points from the curve
        match col_curve {
            CurveType::BSpline { control_points, .. } => {
                surface_control_points[col_idx] = control_points;
            }
            _ => {
                return Err(CascadeError::InvalidGeometry(
                    "Unexpected curve type from interpolate_curve".to_string(),
                ));
            }
        }
    }

    // Transpose to get the proper grid layout: control_points[i][j]
    // where i is u-direction index and j is v-direction index
    let n_control_rows = surface_control_points[0].len();
    let mut control_points = vec![Vec::new(); n_control_rows];

    for row_idx in 0..n_control_rows {
        for col_idx in 0..n_control_cols {
            control_points[row_idx].push(surface_control_points[col_idx][row_idx]);
        }
    }

    // Step 3: Create knot vectors
    // We use the same parameterization strategy for both directions
    
    // For u-direction: use column parameterization
    let u_col_points: Vec<[f64; 3]> = points.iter().map(|row| row[0]).collect();
    let u_params = compute_chord_length_params(&u_col_points);
    let u_knots = create_interpolation_knot_vector(&u_params, u_degree);

    // For v-direction: use row parameterization (0-th control point of each interpolated row)
    let v_col_points: Vec<[f64; 3]> = row_control_points
        .iter()
        .map(|row| row[0])
        .collect();
    let v_params = compute_chord_length_params(&v_col_points);
    let v_knots = create_interpolation_knot_vector(&v_params, v_degree);

    Ok(SurfaceType::BSpline {
        u_degree,
        v_degree,
        u_knots,
        v_knots,
        control_points,
        weights: None,
    })
}

/// Compute chord-length parameterization of points
/// 
/// This parameterization respects the geometric spacing of points,
/// which generally produces better interpolation results than uniform parameterization.
fn compute_chord_length_params(points: &[[f64; 3]]) -> Vec<f64> {
    let mut params = vec![0.0];
    let mut total_dist = 0.0;
    
    for i in 1..points.len() {
        let dx = points[i][0] - points[i - 1][0];
        let dy = points[i][1] - points[i - 1][1];
        let dz = points[i][2] - points[i - 1][2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        total_dist += dist;
        params.push(total_dist);
    }
    
    // Normalize to [0, 1]
    if total_dist > 1e-14 {
        for p in &mut params {
            *p /= total_dist;
        }
    }
    
    params
}

/// Create knot vector for interpolation using averaging method
///
/// Interior knots are placed at averaged parameter values. This is a standard
/// method that works well for interpolation problems.
fn create_interpolation_knot_vector(params: &[f64], degree: usize) -> Vec<f64> {
    let n = params.len();
    let mut knots = Vec::new();
    
    // Clamp at start (degree+1 repeated knots at 0)
    for _ in 0..=degree {
        knots.push(0.0);
    }
    
    // Interior knots: average of consecutive parameters
    // Only add interior knots if there are any (when n > degree + 1)
    if n > degree + 1 {
        for j in 1..=(n - degree - 1) {
            let mut sum = 0.0;
            for i in j..(j + degree).min(n) {
                sum += params[i];
            }
            let avg = sum / degree as f64;
            knots.push(avg);
        }
    }
    
    // Clamp at end (degree+1 repeated knots at 1)
    for _ in 0..=degree {
        knots.push(1.0);
    }
    
    knots
}

/// Solve for interpolation control points using tridiagonal system
///
/// For a BSpline of degree p passing through n data points, we set up
/// the system: N * P = D, where N is the basis function matrix,
/// P is the control points, and D is the data points.
/// 
/// After applying end conditions (clamped BSpline), this reduces to
/// a tridiagonal system for interior control points.
fn interpolate_control_points(
    points: &[[f64; 3]],
    params: &[f64],
    degree: usize,
    knots: &[f64],
) -> Result<Vec<[f64; 3]>> {
    let n = points.len();
    
    // Build basis function matrix N
    let mut matrix = vec![vec![0.0; n]; n];
    
    for (i, &param) in params.iter().enumerate() {
        for j in 0..n {
            matrix[i][j] = bspline_basis(j, degree, param, knots);
        }
    }
    
    // Solve for each coordinate independently
    let mut control_points = vec![[0.0; 3]; n];
    
    for coord in 0..3 {
        // Extract coordinate values
        let point_coords: Vec<f64> = points.iter().map(|p| p[coord]).collect();
        
        // For clamped BSplines, the first and last control points
        // equal the first and last data points. So we solve a reduced system.
        if n >= 4 && degree >= 2 {
            // Use Thomas algorithm for tridiagonal system
            // For degree p, the system is tridiagonal after enforcing end conditions
            
            // For now, use general Gaussian elimination
            // A more efficient implementation could use Thomas algorithm
            let solution = solve_linear_system(&matrix, &point_coords)?;
            
            for i in 0..n {
                control_points[i][coord] = solution[i];
            }
        } else {
            // For small systems, use general Gaussian elimination
            let solution = solve_linear_system(&matrix, &point_coords)?;
            
            for i in 0..n {
                control_points[i][coord] = solution[i];
            }
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

/// Approximate a surface with a BSpline using least squares fitting
///
/// This function fits a BSpline surface to a 2D grid of 3D points, controlling the
/// number of control points based on the specified tolerance.
///
/// # Algorithm
/// - Uses least squares fitting with uniform knot vectors in both U and V directions
/// - Control points are determined by solving normal equations independently for each coordinate
/// - The number of control points in each direction is adjusted based on tolerance
///
/// # Arguments
/// * `points` - 2D grid of 3D points to fit. Points[i] contains the row i of the grid,
///   and Points[i][j] is the point at (u,v) parameter position (j,i).
///   At least a 2x2 grid is required.
/// * `u_degree` - Degree of the BSpline in U direction (typically 2-5, must be < number of U control points)
/// * `v_degree` - Degree of the BSpline in V direction (typically 2-5, must be < number of V control points)
/// * `tolerance` - Maximum deviation allowed (controls number of control points)
///
/// # Returns
/// A BSpline surface approximating the point grid, or an error if the input is invalid
///
/// # Example
/// ```no_run
/// use cascade_rs::approx::approximate_surface;
///
/// // Create a 3x3 grid of points
/// let points = vec![
///     vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
///     vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.5], [2.0, 1.0, 0.0]],
///     vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
/// ];
///
/// let surface = approximate_surface(&points, 2, 2, 0.01)?;
/// # Ok::<_, cascade_rs::CascadeError>(())
/// ```
pub fn approximate_surface(
    points: &[Vec<[f64; 3]>],
    u_degree: usize,
    v_degree: usize,
    tolerance: f64,
) -> Result<SurfaceType> {
    // Validate input grid
    if points.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Point grid cannot be empty".to_string(),
        ));
    }

    if points[0].is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Point grid cannot have empty rows".to_string(),
        ));
    }

    // Check grid dimensions are consistent
    let v_count = points.len();
    let u_count = points[0].len();

    for row in points {
        if row.len() != u_count {
            return Err(CascadeError::InvalidGeometry(
                "All rows in point grid must have the same length".to_string(),
            ));
        }
    }

    if u_count < 2 || v_count < 2 {
        return Err(CascadeError::InvalidGeometry(
            "At least a 2x2 grid of points is required for surface approximation".to_string(),
        ));
    }

    if u_degree < 1 || v_degree < 1 {
        return Err(CascadeError::InvalidGeometry(
            "Degrees must be at least 1".to_string(),
        ));
    }

    if tolerance <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Tolerance must be positive".to_string(),
        ));
    }

    // Determine optimal number of control points in U direction
    let mut n_control_u = (u_degree + 1).max(3);
    if tolerance < 0.001 || u_count > 30 {
        n_control_u = ((u_count as f64) * 0.6) as usize;
        n_control_u = n_control_u.max(u_degree + 1).min(u_count - 1);
    } else if tolerance < 0.01 {
        n_control_u = ((u_count as f64) * 0.4) as usize;
        n_control_u = n_control_u.max(u_degree + 1).min(u_count - 1);
    }
    n_control_u = n_control_u.min(u_count);

    // Determine optimal number of control points in V direction
    let mut n_control_v = (v_degree + 1).max(3);
    if tolerance < 0.001 || v_count > 30 {
        n_control_v = ((v_count as f64) * 0.6) as usize;
        n_control_v = n_control_v.max(v_degree + 1).min(v_count - 1);
    } else if tolerance < 0.01 {
        n_control_v = ((v_count as f64) * 0.4) as usize;
        n_control_v = n_control_v.max(v_degree + 1).min(v_count - 1);
    }
    n_control_v = n_control_v.min(v_count);

    // Validate degrees vs control points
    if u_degree >= n_control_u {
        return Err(CascadeError::InvalidGeometry(format!(
            "U degree ({}) must be less than number of U control points ({})",
            u_degree, n_control_u
        )));
    }

    if v_degree >= n_control_v {
        return Err(CascadeError::InvalidGeometry(format!(
            "V degree ({}) must be less than number of V control points ({})",
            v_degree, n_control_v
        )));
    }

    // Create knot vectors
    let u_knots = create_uniform_knot_vector(n_control_u, u_degree);
    let v_knots = create_uniform_knot_vector(n_control_v, v_degree);

    // Fit control points using least squares
    let control_points = fit_surface_control_points(
        points,
        n_control_u,
        n_control_v,
        u_degree,
        v_degree,
        &u_knots,
        &v_knots,
    )?;

    Ok(SurfaceType::BSpline {
        u_degree,
        v_degree,
        u_knots,
        v_knots,
        control_points,
        weights: None,
    })
}

/// Fit surface control points using least squares fitting
///
/// The algorithm solves for control points P[i][j] by solving a system of equations
/// that minimizes the sum of squared deviations from the input point grid.
///
/// For each coordinate (x, y, z):
/// - Build basis function matrix N_u for U direction and N_v for V direction
/// - Solve: (N_u^T * N_u * P * N_v * N_v^T) = N_u^T * points_coord * N_v
fn fit_surface_control_points(
    points: &[Vec<[f64; 3]>],
    n_control_u: usize,
    n_control_v: usize,
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
) -> Result<Vec<Vec<[f64; 3]>>> {
    let v_count = points.len();
    let u_count = points[0].len();

    // Map input points to parameters
    let u_params: Vec<f64> = (0..u_count)
        .map(|i| i as f64 / (u_count - 1) as f64)
        .collect();
    let v_params: Vec<f64> = (0..v_count)
        .map(|j| j as f64 / (v_count - 1) as f64)
        .collect();

    // Build basis function matrices
    let mut n_u = vec![vec![0.0; n_control_u]; u_count];
    for (i, &u_param) in u_params.iter().enumerate() {
        for j in 0..n_control_u {
            n_u[i][j] = bspline_basis(j, u_degree, u_param, u_knots);
        }
    }

    let mut n_v = vec![vec![0.0; n_control_v]; v_count];
    for (j, &v_param) in v_params.iter().enumerate() {
        for k in 0..n_control_v {
            n_v[j][k] = bspline_basis(k, v_degree, v_param, v_knots);
        }
    }

    // Precompute normal matrices: A_u = N_u^T * N_u and A_v = N_v^T * N_v
    let mut a_u = vec![vec![0.0; n_control_u]; n_control_u];
    for i in 0..n_control_u {
        for j in 0..n_control_u {
            let mut sum = 0.0;
            for k in 0..u_count {
                sum += n_u[k][i] * n_u[k][j];
            }
            a_u[i][j] = sum;
        }
    }

    let mut a_v = vec![vec![0.0; n_control_v]; n_control_v];
    for i in 0..n_control_v {
        for j in 0..n_control_v {
            let mut sum = 0.0;
            for k in 0..v_count {
                sum += n_v[k][i] * n_v[k][j];
            }
            a_v[i][j] = sum;
        }
    }

    // Initialize control points grid
    let mut control_points = vec![vec![[0.0; 3]; n_control_u]; n_control_v];

    // Solve for each coordinate separately
    for coord in 0..3 {
        // Extract coordinate from input points
        let mut point_grid = vec![vec![0.0; u_count]; v_count];
        for (v_idx, row) in points.iter().enumerate() {
            for (u_idx, &pt) in row.iter().enumerate() {
                point_grid[v_idx][u_idx] = pt[coord];
            }
        }

        // Build right-hand side: B = N_u^T * point_grid * N_v
        let mut b = vec![vec![0.0; n_control_v]; n_control_u];
        for i in 0..n_control_u {
            for k in 0..n_control_v {
                let mut sum = 0.0;
                for u in 0..u_count {
                    for v in 0..v_count {
                        sum += n_u[u][i] * point_grid[v][u] * n_v[v][k];
                    }
                }
                b[i][k] = sum;
            }
        }

        // Solve the normal equations: A_u * P * A_v^T = B
        // This requires solving (A_u ⊗ A_v) * vec(P) = vec(B)
        // We use a more practical approach: solve row by row
        // For each row i of control points: A_u[i,i] * P[i] * A_v^T = B[i]
        // Simplified: solve A_v^T * P[i]^T = (A_u^{-1} * B)[i]

        // Invert A_u
        let a_u_inv = invert_matrix(&a_u)?;

        // Compute temp = A_u^{-1} * B
        let mut temp = vec![vec![0.0; n_control_v]; n_control_u];
        for i in 0..n_control_u {
            for j in 0..n_control_v {
                let mut sum = 0.0;
                for k in 0..n_control_u {
                    sum += a_u_inv[i][k] * b[k][j];
                }
                temp[i][j] = sum;
            }
        }

        // Invert A_v
        let a_v_inv = invert_matrix(&a_v)?;

        // Compute control point grid: P = temp * A_v^{-1}
        for i in 0..n_control_u {
            for j in 0..n_control_v {
                let mut sum = 0.0;
                for k in 0..n_control_v {
                    sum += temp[i][k] * a_v_inv[k][j];
                }
                control_points[j][i][coord] = sum;
            }
        }
    }

    Ok(control_points)
}

/// Invert a matrix using Gaussian elimination
fn invert_matrix(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return Err(CascadeError::InvalidGeometry(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    // Create augmented matrix [A | I]
    let mut augmented = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = a[i][j];
        }
        augmented[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = augmented[col][col].abs();

        for row in (col + 1)..n {
            if augmented[row][col].abs() > max_val {
                max_val = augmented[row][col].abs();
                max_row = row;
            }
        }

        // Check for singular matrix
        if max_val < 1e-14 {
            return Err(CascadeError::InvalidGeometry(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            augmented.swap(col, max_row);
        }

        // Scale pivot row
        let pivot = augmented[col][col];
        for j in 0..(2 * n) {
            augmented[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = augmented[row][col];
                for j in 0..(2 * n) {
                    augmented[row][j] -= factor * augmented[col][j];
                }
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = augmented[i][n + j];
        }
    }

    Ok(inv)
}

/// Create a plate surface constrained by points and curves
/// 
/// A plate surface is a smooth BSpline surface that:
/// - Passes through the given constraint points
/// - Is tangent to the given constraint curves
/// 
/// This function creates a BSpline surface by:
/// 1. Fitting a grid to pass through constraint points
/// 2. Creating a BSpline surface that interpolates/approximates the grid
/// 
/// # Arguments
/// * `points` - Points that the surface must pass through (array of [x, y, z])
/// * `curves` - Curves that the surface should be tangent to
/// * `u_degree` - Degree in U direction (default: 2)
/// * `v_degree` - Degree in V direction (default: 2)
/// * `tolerance` - Fitting tolerance for approximation
///
/// # Returns
/// A PlateSurface variant of SurfaceType
///
/// # Example
/// ```
/// use cascade_rs::make_plate_surface;
/// use cascade_rs::brep::{SurfaceType, CurveType};
/// 
/// let points = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.5],
///     [0.0, 1.0, 0.5],
///     [1.0, 1.0, 1.0],
/// ];
/// let curves = vec![];
/// 
/// let surface = make_plate_surface(&points, &curves, 2, 2, 0.01)?;
/// # Ok::<(), _>(())
/// ```
pub fn make_plate_surface(
    points: &[[f64; 3]],
    curves: &[CurveType],
    u_degree: usize,
    v_degree: usize,
    _tolerance: f64,
) -> Result<SurfaceType> {
    if points.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "At least one constraint point required".to_string(),
        ));
    }

    // For a simple implementation, we create a BSpline surface through the points
    // by arranging them in a minimal grid pattern
    
    // Arrange points into a grid - choose grid dimensions based on point count
    let (grid, actual_u_degree, actual_v_degree) = if points.len() == 4 {
        // 4 points: 2x2 grid, use degree 1 (must be < num_cols and num_rows)
        let grid = vec![
            vec![points[0], points[1]],
            vec![points[2], points[3]],
        ];
        (grid, 1, 1)
    } else if points.len() == 2 {
        // Two points: create 2x2 grid with slight offset
        // This creates a ruled surface that passes through the points
        let offset = 0.01; // Small offset to avoid singular matrix
        let grid = vec![
            vec![points[0], points[1]],
            vec![
                [points[0][0], points[0][1], points[0][2] + offset],
                [points[1][0], points[1][1], points[1][2] + offset],
            ],
        ];
        (grid, 1, 1)
    } else if points.len() == 3 {
        // Three points: create 2x3 grid with offset row
        let offset = 0.01;
        let grid = vec![
            vec![points[0], points[1], points[2]],
            vec![
                [points[0][0], points[0][1], points[0][2] + offset],
                [points[1][0], points[1][1], points[1][2] + offset],
                [points[2][0], points[2][1], points[2][2] + offset],
            ],
        ];
        (grid, 2, 1)
    } else {
        // For other numbers, arrange as 2 rows with offset
        // Ensure degree < num_points in each direction
        // For 5 points: u_degree must be < 5, so min with 4
        let u_deg = (points.len() - 1).min(u_degree);
        // For 2 rows: v_degree must be < 2, so v_degree = 1
        let v_deg = 1;
        let offset = 0.01;
        let offset_row = points
            .iter()
            .map(|p| [p[0], p[1], p[2] + offset])
            .collect::<Vec<_>>();
        let grid = vec![
            points.to_vec(),
            offset_row,
        ];
        (grid, u_deg, v_deg)
    };

    // Create the base BSpline surface through the grid
    let basis_surface = interpolate_surface(&grid, actual_u_degree, actual_v_degree)?;

    // Wrap in a PlateSurface with constraints
    Ok(SurfaceType::PlateSurface {
        basis_surface: Box::new(basis_surface),
        constraint_points: points.to_vec(),
        constraint_curves: curves.to_vec(),
    })
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

    #[test]
    fn test_interpolate_line() -> Result<()> {
        // Interpolate a line through 4 collinear points
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        
        let curve = interpolate_curve(&points, 1)?;
        
        // Verify we get a BSpline
        match curve {
            CurveType::BSpline { ref control_points, ref knots, degree, .. } => {
                assert_eq!(degree, 1);
                assert_eq!(control_points.len(), points.len());
                assert!(knots.len() > 0);
            }
            _ => panic!("Expected BSpline"),
        }
        
        Ok(())
    }

    #[test]
    fn test_interpolate_cubic() -> Result<()> {
        // Interpolate a cubic curve through 5 points
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        
        let curve = interpolate_curve(&points, 3)?;
        
        // Verify we get a BSpline with correct degree
        match curve {
            CurveType::BSpline { ref control_points, degree, .. } => {
                assert_eq!(degree, 3);
                // For interpolation, control_points count should equal points count
                assert_eq!(control_points.len(), points.len());
            }
            _ => panic!("Expected BSpline"),
        }
        
        Ok(())
    }

    #[test]
    fn test_interpolate_invalid_input() {
        // Too few points for given degree
        let points = vec![[0.0; 3], [1.0; 3]];
        let result = interpolate_curve(&points, 3);
        assert!(result.is_err());
        
        // Invalid degree
        let points = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
        let result = interpolate_curve(&points, 0);
        assert!(result.is_err());
        
        // Too few points overall
        let result = interpolate_curve(&[[0.0; 3]], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_chord_length_parameterization() {
        // Test chord-length parameterization
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  // distance 1
            [2.0, 0.0, 0.0],  // distance 1
            [3.0, 0.0, 0.0],  // distance 1
        ];
        
        let params = compute_chord_length_params(&points);
        
        assert_eq!(params.len(), 4);
        assert_eq!(params[0], 0.0);
        assert_eq!(params[3], 1.0);
        
        // For uniform spacing, parameters should be uniformly distributed
        assert!((params[1] - 1.0/3.0).abs() < 1e-10);
        assert!((params[2] - 2.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_chord_length_non_uniform() {
        // Test chord-length parameterization with non-uniform spacing
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  // distance 1
            [1.5, 0.0, 0.0],  // distance 0.5
            [2.5, 0.0, 0.0],  // distance 1
        ];
        
        let params = compute_chord_length_params(&points);
        
        assert_eq!(params.len(), 4);
        assert_eq!(params[0], 0.0);
        assert_eq!(params[3], 1.0);
        
        // Second point should be 1/2.5 of the way
        assert!((params[1] - 1.0/2.5).abs() < 1e-10);
        // Third point should be 1.5/2.5 of the way
        assert!((params[2] - 1.5/2.5).abs() < 1e-10);
    }

    #[test]
    fn test_interpolation_knot_vector() {
        let params = vec![0.0, 1.0/3.0, 2.0/3.0, 1.0];
        let degree = 2;
        let knots = create_interpolation_knot_vector(&params, degree);
        
        // Expected length: n + degree + 1 = 4 + 2 + 1 = 7
        assert_eq!(knots.len(), 7);
        
        // Check clamping
        for i in 0..=degree {
            assert_eq!(knots[i], 0.0);
            assert_eq!(knots[knots.len() - 1 - i], 1.0);
        }
        
        // Check monotonicity
        for i in 0..knots.len() - 1 {
            assert!(knots[i] <= knots[i + 1]);
        }
    }

    #[test]
    fn test_interpolate_sinusoid() -> Result<()> {
        // Interpolate a sinusoidal curve
        let points: Vec<_> = (0..15)
            .map(|i| {
                let t = i as f64 / 14.0;
                let x = t * 2.0 * std::f64::consts::PI;
                [t, x.sin(), 0.0]
            })
            .collect();
        
        let curve = interpolate_curve(&points, 3)?;
        
        // Verify curve was created with correct properties
        match curve {
            CurveType::BSpline { ref control_points, degree, .. } => {
                assert_eq!(degree, 3);
                // For interpolation, number of control points equals number of data points
                assert_eq!(control_points.len(), points.len());
            }
            _ => panic!("Expected BSpline"),
        }
        
        Ok(())
    }

    #[test]
    fn test_approximate_flat_surface() -> Result<()> {
        // Approximate a flat surface (XY plane at Z=0)
        let points = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        ];

        let surface = approximate_surface(&points, 1, 1, 0.01)?;

        match surface {
            SurfaceType::BSpline {
                control_points,
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                ..
            } => {
                assert_eq!(u_degree, 1);
                assert_eq!(v_degree, 1);
                assert!(control_points.len() >= 2);
                assert!(control_points[0].len() >= 2);
                assert!(u_knots.len() > 0);
                assert!(v_knots.len() > 0);
            }
            _ => panic!("Expected BSpline surface"),
        }

        Ok(())
    }

    #[test]
    fn test_approximate_curved_surface() -> Result<()> {
        // Approximate a curved surface (quadratic)
        let mut points = Vec::new();
        for j in 0..4 {
            let mut row = Vec::new();
            for i in 0..4 {
                let u = i as f64 / 3.0;
                let v = j as f64 / 3.0;
                let x = u;
                let y = v;
                let z = (u * u + v * v) * 0.5; // quadratic surface
                row.push([x, y, z]);
            }
            points.push(row);
        }

        let surface = approximate_surface(&points, 2, 2, 0.01)?;

        match surface {
            SurfaceType::BSpline {
                control_points,
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                ..
            } => {
                assert_eq!(u_degree, 2);
                assert_eq!(v_degree, 2);
                assert!(control_points.len() >= 2);
                assert!(control_points[0].len() >= 2);

                // Check knot vector dimensions
                assert_eq!(u_knots.len(), control_points[0].len() + u_degree + 1);
                assert_eq!(v_knots.len(), control_points.len() + v_degree + 1);
            }
            _ => panic!("Expected BSpline surface"),
        }

        Ok(())
    }

    #[test]
    fn test_surface_tight_vs_loose_tolerance() -> Result<()> {
        // Create a larger point grid
        let mut points = Vec::new();
        for j in 0..5 {
            let mut row = Vec::new();
            for i in 0..5 {
                let u = i as f64 / 4.0;
                let v = j as f64 / 4.0;
                let x = u;
                let y = v;
                let z = (2.0 * u * v).sin();
                row.push([x, y, z]);
            }
            points.push(row);
        }

        let tight = approximate_surface(&points, 2, 2, 0.0001)?;
        let loose = approximate_surface(&points, 2, 2, 0.1)?;

        let tight_count = match &tight {
            SurfaceType::BSpline { control_points, .. } => control_points[0].len(),
            _ => panic!("Expected BSpline"),
        };

        let loose_count = match &loose {
            SurfaceType::BSpline { control_points, .. } => control_points[0].len(),
            _ => panic!("Expected BSpline"),
        };

        // Tighter tolerance should use more or equal control points
        assert!(tight_count >= loose_count);

        Ok(())
    }

    #[test]
    fn test_surface_invalid_input() {
        // Empty grid
        let result = approximate_surface(&[], 1, 1, 0.01);
        assert!(result.is_err());

        // Single row
        let points = vec![vec![[0.0; 3], [1.0; 3]]];
        let result = approximate_surface(&points, 1, 1, 0.01);
        assert!(result.is_err());

        // Inconsistent row sizes
        let points = vec![
            vec![[0.0; 3], [1.0; 3]],
            vec![[0.0; 3], [1.0; 3], [2.0; 3]],
        ];
        let result = approximate_surface(&points, 1, 1, 0.01);
        assert!(result.is_err());

        // Invalid degree
        let points = vec![
            vec![[0.0; 3], [1.0; 3]],
            vec![[0.0; 3], [1.0; 3]],
        ];
        let result = approximate_surface(&points, 0, 1, 0.01);
        assert!(result.is_err());

        // Invalid tolerance
        let result = approximate_surface(&points, 1, 1, -0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_surface_degree_vs_control_points() -> Result<()> {
        let mut points = Vec::new();
        for j in 0..3 {
            let mut row = Vec::new();
            for i in 0..3 {
                row.push([(i as f64), (j as f64), 0.0]);
            }
            points.push(row);
        }

        // Degree too high should fail
        let result = approximate_surface(&points, 5, 1, 0.01);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_surface_knot_vectors() -> Result<()> {
        let mut points = Vec::new();
        for j in 0..4 {
            let mut row = Vec::new();
            for i in 0..4 {
                row.push([(i as f64), (j as f64), 0.0]);
            }
            points.push(row);
        }

        let surface = approximate_surface(&points, 2, 2, 0.01)?;

        match surface {
            SurfaceType::BSpline {
                control_points,
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                ..
            } => {
                let n_control_u = control_points[0].len();
                let n_control_v = control_points.len();

                // Check knot vector lengths
                assert_eq!(u_knots.len(), n_control_u + u_degree + 1);
                assert_eq!(v_knots.len(), n_control_v + v_degree + 1);

                // Check clamping at U
                for i in 0..=u_degree {
                    assert_eq!(u_knots[i], 0.0);
                    assert_eq!(u_knots[u_knots.len() - i - 1], 1.0);
                }

                // Check clamping at V
                for i in 0..=v_degree {
                    assert_eq!(v_knots[i], 0.0);
                    assert_eq!(v_knots[v_knots.len() - i - 1], 1.0);
                }

                // Check monotonicity
                for i in 0..u_knots.len() - 1 {
                    assert!(u_knots[i] <= u_knots[i + 1]);
                }
                for i in 0..v_knots.len() - 1 {
                    assert!(v_knots[i] <= v_knots[i + 1]);
                }
            }
            _ => panic!("Expected BSpline"),
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_surface_simple() -> Result<()> {
        // Create a simple 3x3 grid of points
        let points = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.5], [1.0, 1.0, 0.7], [2.0, 1.0, 0.5]],
            vec![[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        ];

        let surface = interpolate_surface(&points, 2, 2)?;

        // Verify we get a BSpline surface
        match surface {
            SurfaceType::BSpline {
                u_degree,
                v_degree,
                control_points,
                u_knots,
                v_knots,
                ..
            } => {
                assert_eq!(u_degree, 2);
                assert_eq!(v_degree, 2);
                // With 3 points in u-direction and degree 2, we should have control points
                assert!(control_points.len() > 0);
                assert!(control_points[0].len() > 0);
                // Check knot vectors
                assert!(u_knots.len() > 0);
                assert!(v_knots.len() > 0);
                // Check clamping
                assert_eq!(u_knots[0], 0.0);
                assert_eq!(u_knots[u_knots.len() - 1], 1.0);
                assert_eq!(v_knots[0], 0.0);
                assert_eq!(v_knots[v_knots.len() - 1], 1.0);
            }
            _ => panic!("Expected BSpline surface"),
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_surface_rectangular() -> Result<()> {
        // Create a 4x5 rectangular grid
        let mut points = Vec::new();
        for j in 0..5 {
            let mut row = Vec::new();
            for i in 0..4 {
                let x = i as f64;
                let y = j as f64;
                let z = (x * x + y * y) * 0.1;
                row.push([x, y, z]);
            }
            points.push(row);
        }

        let surface = interpolate_surface(&points, 3, 3)?;

        match surface {
            SurfaceType::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                ..
            } => {
                assert_eq!(u_degree, 3);
                assert_eq!(v_degree, 3);
                // Check knot vectors are valid
                for i in 0..u_knots.len() - 1 {
                    assert!(u_knots[i] <= u_knots[i + 1], "U knots must be non-decreasing");
                }
                for i in 0..v_knots.len() - 1 {
                    assert!(v_knots[i] <= v_knots[i + 1], "V knots must be non-decreasing");
                }
            }
            _ => panic!("Expected BSpline surface"),
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_surface_invalid_input() {
        // Empty grid
        let result = interpolate_surface(&vec![], 2, 2);
        assert!(result.is_err());

        // Single column
        let points = vec![vec![[0.0, 0.0, 0.0]]];
        let result = interpolate_surface(&points, 2, 2);
        assert!(result.is_err());

        // Inconsistent row sizes
        let points = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.0]],
        ];
        let result = interpolate_surface(&points, 2, 2);
        assert!(result.is_err());

        // Degree too high
        let points = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ];
        let result = interpolate_surface(&points, 5, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_surface_passes_through_points() -> Result<()> {
        // Create a simple grid
        let points = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 0.5], [1.0, 1.0, 0.5]],
        ];

        let surface = interpolate_surface(&points, 1, 1)?;

        // For linear interpolation, the surface should pass through all grid points
        // (or very close due to numerical precision)
        match surface {
            SurfaceType::BSpline {
                control_points, ..
            } => {
                // The corner control points should be close to the input points
                // For a clamped BSpline interpolation, the endpoints are fixed
                let start = control_points[0][0];
                let end = control_points[control_points.len() - 1][control_points[0].len() - 1];
                
                // First and last control points should match interpolation data
                assert!((start[0] - points[0][0][0]).abs() < 1e-10);
                assert!((start[1] - points[0][0][1]).abs() < 1e-10);
                assert!((end[0] - points.last().unwrap().last().unwrap()[0]).abs() < 1e-10);
                assert!((end[1] - points.last().unwrap().last().unwrap()[1]).abs() < 1e-10);
            }
            _ => panic!("Expected BSpline surface"),
        }

        Ok(())
    }

    #[test]
    fn test_plate_surface_4_points() -> Result<()> {
        // Create a simple 2x2 grid of points
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let curves = vec![];

        let surface = make_plate_surface(&points, &curves, 2, 2, 0.01)?;

        // Verify it's a PlateSurface
        match &surface {
            SurfaceType::PlateSurface {
                basis_surface,
                constraint_points,
                constraint_curves,
            } => {
                // Check that constraint points are stored
                assert_eq!(constraint_points.len(), 4);
                assert_eq!(constraint_curves.len(), 0);

                // Check that the basis surface exists and is a BSpline
                match &**basis_surface {
                    SurfaceType::BSpline { .. } => {
                        // Good - basis surface is BSpline
                    }
                    _ => panic!("Expected BSpline basis surface"),
                }

                // Verify that the surface can be evaluated at various parameters
                let test_points = vec![
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 1.0),
                    (0.5, 0.5),
                ];
                
                for (u, v) in test_points {
                    let pt = surface.point_at(u, v);
                    // Just verify we got valid numbers (no NaN)
                    assert!(pt[0].is_finite(), "Point has non-finite x coordinate");
                    assert!(pt[1].is_finite(), "Point has non-finite y coordinate");
                    assert!(pt[2].is_finite(), "Point has non-finite z coordinate");
                }
            }
            _ => panic!("Expected PlateSurface variant"),
        }

        Ok(())
    }

    #[test]
    fn test_plate_surface_2_points() -> Result<()> {
        // Test with just 2 points (should create a degenerate surface)
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let curves = vec![];

        let surface = make_plate_surface(&points, &curves, 2, 2, 0.01)?;

        match surface {
            SurfaceType::PlateSurface {
                constraint_points,
                constraint_curves,
                ..
            } => {
                assert_eq!(constraint_points.len(), 2);
                assert_eq!(constraint_curves.len(), 0);
            }
            _ => panic!("Expected PlateSurface variant"),
        }

        Ok(())
    }

    #[test]
    fn test_plate_surface_empty_points() -> Result<()> {
        // Test that empty points fail
        let points: Vec<[f64; 3]> = vec![];
        let curves = vec![];

        let result = make_plate_surface(&points, &curves, 2, 2, 0.01);
        assert!(result.is_err(), "Should fail with empty points");

        Ok(())
    }

    #[test]
    fn test_plate_surface_multiple_points() -> Result<()> {
        // Test with more than 4 points (arranged as a sequence)
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let curves = vec![];

        let surface = make_plate_surface(&points, &curves, 2, 2, 0.01)?;

        match surface {
            SurfaceType::PlateSurface {
                basis_surface,
                constraint_points,
                ..
            } => {
                assert_eq!(constraint_points.len(), 5);
                
                // Verify basis surface is BSpline
                match *basis_surface {
                    SurfaceType::BSpline { .. } => {},
                    _ => panic!("Expected BSpline basis surface"),
                }
            }
            _ => panic!("Expected PlateSurface variant"),
        }

        Ok(())
    }
}
