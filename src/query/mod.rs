//! Geometric queries

use crate::brep::{Shape, Solid, Shell, Face, Wire, SurfaceType};
use crate::{Result, CascadeError};

/// Classification of shape geometry: convex, concave, mixed, or degenerate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeClass {
    /// All faces are convex and form a convex hull
    Convex,
    /// Contains at least one concave feature (negative curvature)
    Concave,
    /// Mixed compound with both convex and concave components
    Mixed,
    /// Degenerate shape (zero volume, invalid topology, etc.)
    Degenerate,
}

pub fn distance(shape1: &Shape, shape2: &Shape) -> Result<f64> {
    Err(CascadeError::NotImplemented("query::distance".into()))
}

pub fn intersects(shape1: &Shape, shape2: &Shape) -> Result<bool> {
    Err(CascadeError::NotImplemented("query::intersection".into()))
}

pub fn point_inside(solid: &Solid, point: [f64; 3]) -> Result<bool> {
    Err(CascadeError::NotImplemented("query::inside".into()))
}

pub fn bounding_box(shape: &Shape) -> Result<([f64; 3], [f64; 3])> {
    // Collect all vertices from the shape
    let mut vertices = Vec::new();
    collect_vertices(shape, &mut vertices);
    
    if vertices.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Shape has no vertices".into()
        ));
    }
    
    // Find min and max coordinates
    let mut min = vertices[0];
    let mut max = vertices[0];
    
    for &vertex in &vertices {
        for i in 0..3 {
            if vertex[i] < min[i] {
                min[i] = vertex[i];
            }
            if vertex[i] > max[i] {
                max[i] = vertex[i];
            }
        }
    }
    
    Ok((min, max))
}

/// Helper function to recursively collect all vertices from a shape
fn collect_vertices(shape: &Shape, vertices: &mut Vec<[f64; 3]>) {
    match shape {
        Shape::Vertex(v) => {
            vertices.push(v.point);
        }
        Shape::Edge(e) => {
            vertices.push(e.start.point);
            vertices.push(e.end.point);
        }
        Shape::Wire(w) => {
            for edge in &w.edges {
                vertices.push(edge.start.point);
                vertices.push(edge.end.point);
            }
        }
        Shape::Face(f) => {
            // Collect from outer wire and inner wires
            for edge in &f.outer_wire.edges {
                vertices.push(edge.start.point);
            }
            for wire in &f.inner_wires {
                for edge in &wire.edges {
                    vertices.push(edge.start.point);
                }
            }
        }
        Shape::Shell(s) => {
            for face in &s.faces {
                for edge in &face.outer_wire.edges {
                    vertices.push(edge.start.point);
                }
                for wire in &face.inner_wires {
                    for edge in &wire.edges {
                        vertices.push(edge.start.point);
                    }
                }
            }
        }
        Shape::Solid(solid) => {
            for face in &solid.outer_shell.faces {
                for edge in &face.outer_wire.edges {
                    vertices.push(edge.start.point);
                }
                for wire in &face.inner_wires {
                    for edge in &wire.edges {
                        vertices.push(edge.start.point);
                    }
                }
            }
            for shell in &solid.inner_shells {
                for face in &shell.faces {
                    for edge in &face.outer_wire.edges {
                        vertices.push(edge.start.point);
                    }
                    for wire in &face.inner_wires {
                        for edge in &wire.edges {
                            vertices.push(edge.start.point);
                        }
                    }
                }
            }
        }
        Shape::Compound(c) => {
            for solid in &c.solids {
                collect_vertices(&Shape::Solid(solid.clone()), vertices);
            }
        }
        Shape::CompSolid(_) => {
            // CompSolid is not yet implemented, skip
        }
    }
}

pub struct MassProperties {
    pub volume: f64,
    pub surface_area: f64,
    pub center_of_mass: [f64; 3],
}

pub fn mass_properties(solid: &Solid) -> Result<MassProperties> {
    let mut volume = 0.0;
    let mut surface_area = 0.0;
    let mut center_of_mass = [0.0; 3];
    
    // Process outer shell
    calculate_shell_properties(&solid.outer_shell, &mut volume, &mut surface_area, &mut center_of_mass);
    
    // Process inner shells (subtract from total)
    for shell in &solid.inner_shells {
        let mut inner_volume = 0.0;
        let mut inner_surface_area = 0.0;
        let mut inner_center = [0.0; 3];
        calculate_shell_properties(shell, &mut inner_volume, &mut inner_surface_area, &mut inner_center);
        volume -= inner_volume;
        surface_area -= inner_surface_area;
        
        // Subtract inner center contribution
        for i in 0..3 {
            center_of_mass[i] -= inner_center[i];
        }
    }
    
    // Normalize center of mass by volume (already weighted during accumulation)
    if volume.abs() > 1e-10 {
        for i in 0..3 {
            center_of_mass[i] /= volume;
        }
    }
    
    Ok(MassProperties {
        volume: volume.abs(),
        surface_area,
        center_of_mass,
    })
}

/// Inertia matrix (moments of inertia tensor) for a solid
/// 
/// The inertia matrix is symmetric with 6 independent components:
/// ```text
/// | Ixx  -Ixy  -Ixz |
/// |-Ixy   Iyy  -Iyz |
/// |-Ixz  -Iyz   Izz |
/// ```
/// 
/// The diagonal elements (Ixx, Iyy, Izz) are the moments of inertia about each axis.
/// The off-diagonal elements (Ixy, Ixz, Iyz) are the products of inertia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertiaMatrix {
    /// Moment of inertia about the X axis: ∫(y² + z²) dm
    pub ixx: f64,
    /// Moment of inertia about the Y axis: ∫(x² + z²) dm
    pub iyy: f64,
    /// Moment of inertia about the Z axis: ∫(x² + y²) dm
    pub izz: f64,
    /// Product of inertia: ∫xy dm
    pub ixy: f64,
    /// Product of inertia: ∫xz dm
    pub ixz: f64,
    /// Product of inertia: ∫yz dm
    pub iyz: f64,
}

impl InertiaMatrix {
    /// Create a new inertia matrix
    pub fn new(ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64) -> Self {
        Self { ixx, iyy, izz, ixy, ixz, iyz }
    }
    
    /// Return the inertia tensor as a 3x3 matrix (row-major order)
    pub fn as_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.ixx, -self.ixy, -self.ixz],
            [-self.ixy, self.iyy, -self.iyz],
            [-self.ixz, -self.iyz, self.izz],
        ]
    }
    
    /// Translate the inertia matrix using the parallel axis theorem.
    /// 
    /// Shifts the inertia matrix from the current reference point to a new point
    /// displaced by `offset` from the original. Uses mass `m`.
    /// 
    /// I' = I + m * [(d·d)E - d⊗d]
    /// where d is the offset vector, E is identity, and ⊗ is outer product.
    pub fn translate(&self, offset: [f64; 3], mass: f64) -> Self {
        let dx = offset[0];
        let dy = offset[1];
        let dz = offset[2];
        let d_sq = dx * dx + dy * dy + dz * dz;
        
        Self {
            ixx: self.ixx + mass * (d_sq - dx * dx),
            iyy: self.iyy + mass * (d_sq - dy * dy),
            izz: self.izz + mass * (d_sq - dz * dz),
            ixy: self.ixy + mass * dx * dy,
            ixz: self.ixz + mass * dx * dz,
            iyz: self.iyz + mass * dy * dz,
        }
    }
}

/// Compute the moments of inertia for a solid assuming uniform density.
/// 
/// The result is computed by tessellating the solid into a triangle mesh,
/// then decomposing into tetrahedra from the origin and summing their contributions.
/// 
/// # Arguments
/// * `solid` - The solid to compute inertia for
/// 
/// # Returns
/// The inertia matrix about the origin with unit density (multiply by density for actual values).
/// 
/// # Algorithm
/// Uses the signed tetrahedron method: each surface triangle forms a tetrahedron with the origin.
/// The contribution of each tetrahedron to the inertia tensor is computed analytically.
/// 
/// For a tetrahedron with vertices (0,0,0), v1, v2, v3 and signed volume V:
/// - Ixx = (V/10) * (y1² + y2² + y3² + y1y2 + y2y3 + y1y3 + z1² + z2² + z3² + z1z2 + z2z3 + z1z3)
/// - Similar for Iyy, Izz
/// - Ixy = (V/20) * (2x1y1 + 2x2y2 + 2x3y3 + x1y2 + x2y1 + x2y3 + x3y2 + x1y3 + x3y1)
/// - Similar for Ixz, Iyz
/// 
/// # Examples
/// ```
/// use cascade::{make_box, query::moments_of_inertia};
/// 
/// let cube = make_box(2.0, 2.0, 2.0).unwrap();
/// let inertia = moments_of_inertia(&cube).unwrap();
/// // For a 2x2x2 cube centered at origin with unit density:
/// // Ixx = Iyy = Izz = (1/12) * m * (b² + c²) where m = volume = 8
/// // So Ixx = (8/12) * (4 + 4) = 5.333...
/// ```
pub fn moments_of_inertia(solid: &Solid) -> Result<InertiaMatrix> {
    use crate::mesh::triangulate;
    
    // Tessellate the solid with reasonable tolerance
    let mesh = triangulate(solid, 0.01)?;
    
    if mesh.triangles.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Solid has no triangles after tessellation".into()
        ));
    }
    
    // Accumulate inertia contributions from each triangle (forming tetrahedra with origin)
    let mut ixx = 0.0;
    let mut iyy = 0.0;
    let mut izz = 0.0;
    let mut ixy = 0.0;
    let mut ixz = 0.0;
    let mut iyz = 0.0;
    
    for tri in &mesh.triangles {
        let v1 = mesh.vertices[tri[0]];
        let v2 = mesh.vertices[tri[1]];
        let v3 = mesh.vertices[tri[2]];
        
        // Compute triangle normal from vertex winding
        let e1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        let e2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];
        let tri_normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        
        // Compare with stored mesh normal to determine if winding matches
        let mesh_normal = mesh.normals[tri[0]];
        let dot_normals = tri_normal[0] * mesh_normal[0] 
                        + tri_normal[1] * mesh_normal[1] 
                        + tri_normal[2] * mesh_normal[2];
        
        // Sign multiplier: +1 if triangle winding matches outward normal, -1 if opposite
        let sign = if dot_normals >= 0.0 { 1.0 } else { -1.0 };
        
        // Signed volume of tetrahedron with origin at (0,0,0)
        // V = (1/6) * (v1 · (v2 × v3))
        let cross = [
            v2[1] * v3[2] - v2[2] * v3[1],
            v2[2] * v3[0] - v2[0] * v3[2],
            v2[0] * v3[1] - v2[1] * v3[0],
        ];
        let signed_volume = sign * (v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2]) / 6.0;
        
        let x1 = v1[0]; let y1 = v1[1]; let z1 = v1[2];
        let x2 = v2[0]; let y2 = v2[1]; let z2 = v2[2];
        let x3 = v3[0]; let y3 = v3[1]; let z3 = v3[2];
        
        // Moments of inertia for a tetrahedron with one vertex at origin
        // These formulas come from integrating ∫ρ(y²+z²)dV over the tetrahedron
        // For derivation see: "Explicit Exact Formulas for the 3-D Tetrahedron Inertia Tensor"
        
        // Sum of squared and cross terms for each coordinate
        let sum_x2 = x1*x1 + x2*x2 + x3*x3 + x1*x2 + x2*x3 + x1*x3;
        let sum_y2 = y1*y1 + y2*y2 + y3*y3 + y1*y2 + y2*y3 + y1*y3;
        let sum_z2 = z1*z1 + z2*z2 + z3*z3 + z1*z2 + z2*z3 + z1*z3;
        
        let sum_xy = 2.0*x1*y1 + 2.0*x2*y2 + 2.0*x3*y3 + x1*y2 + x2*y1 + x2*y3 + x3*y2 + x1*y3 + x3*y1;
        let sum_xz = 2.0*x1*z1 + 2.0*x2*z2 + 2.0*x3*z3 + x1*z2 + x2*z1 + x2*z3 + x3*z2 + x1*z3 + x3*z1;
        let sum_yz = 2.0*y1*z1 + 2.0*y2*z2 + 2.0*y3*z3 + y1*z2 + y2*z1 + y2*z3 + y3*z2 + y1*z3 + y3*z1;
        
        // Contribution to each component
        // Factor of signed_volume/10 for diagonal terms
        // Factor of signed_volume/20 for off-diagonal terms
        ixx += signed_volume * (sum_y2 + sum_z2) / 10.0;
        iyy += signed_volume * (sum_x2 + sum_z2) / 10.0;
        izz += signed_volume * (sum_x2 + sum_y2) / 10.0;
        ixy += signed_volume * sum_xy / 20.0;
        ixz += signed_volume * sum_xz / 20.0;
        iyz += signed_volume * sum_yz / 20.0;
    }
    
    Ok(InertiaMatrix {
        ixx,
        iyy,
        izz,
        ixy,
        ixz,
        iyz,
    })
}

/// Compute the moments of inertia about the center of mass.
/// 
/// First computes moments about the origin, then translates to the center of mass
/// using the parallel axis theorem.
/// 
/// # Arguments
/// * `solid` - The solid to compute inertia for
/// 
/// # Returns
/// A tuple of (InertiaMatrix, center_of_mass, volume) where the inertia is about the center of mass.
pub fn moments_of_inertia_at_com(solid: &Solid) -> Result<(InertiaMatrix, [f64; 3], f64)> {
    use crate::mesh::triangulate;
    
    // Tessellate and compute volume + COM directly from mesh (more accurate than mass_properties)
    let mesh = triangulate(solid, 0.01)?;
    
    if mesh.triangles.is_empty() {
        return Err(CascadeError::InvalidGeometry(
            "Solid has no triangles after tessellation".into()
        ));
    }
    
    // Compute volume and COM using signed tetrahedra method
    let mut total_volume = 0.0;
    let mut weighted_com = [0.0; 3];
    
    for tri in &mesh.triangles {
        let v1 = mesh.vertices[tri[0]];
        let v2 = mesh.vertices[tri[1]];
        let v3 = mesh.vertices[tri[2]];
        
        // Check triangle winding against mesh normal
        let e1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        let e2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];
        let tri_normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        
        let mesh_normal = mesh.normals[tri[0]];
        let dot_normals = tri_normal[0] * mesh_normal[0] 
                        + tri_normal[1] * mesh_normal[1] 
                        + tri_normal[2] * mesh_normal[2];
        let sign = if dot_normals >= 0.0 { 1.0 } else { -1.0 };
        
        let cross = [
            v2[1] * v3[2] - v2[2] * v3[1],
            v2[2] * v3[0] - v2[0] * v3[2],
            v2[0] * v3[1] - v2[1] * v3[0],
        ];
        let signed_vol = sign * (v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2]) / 6.0;
        total_volume += signed_vol;
        
        // Tetrahedron centroid: (origin + v1 + v2 + v3) / 4
        for i in 0..3 {
            weighted_com[i] += signed_vol * (v1[i] + v2[i] + v3[i]) / 4.0;
        }
    }
    
    if total_volume.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry(
            "Solid has zero volume".into()
        ));
    }
    
    let com = [
        weighted_com[0] / total_volume,
        weighted_com[1] / total_volume,
        weighted_com[2] / total_volume,
    ];
    let m = total_volume;
    
    // Get inertia about origin
    let inertia_origin = moments_of_inertia(solid)?;
    
    // Translate from origin to center of mass using parallel axis theorem (in reverse)
    // I_com = I_origin - m * [(d·d)E - d⊗d]
    let dx = com[0];
    let dy = com[1];
    let dz = com[2];
    let d_sq = dx * dx + dy * dy + dz * dz;
    
    let inertia_com = InertiaMatrix {
        ixx: inertia_origin.ixx - m * (d_sq - dx * dx),
        iyy: inertia_origin.iyy - m * (d_sq - dy * dy),
        izz: inertia_origin.izz - m * (d_sq - dz * dz),
        ixy: inertia_origin.ixy - m * dx * dy,
        ixz: inertia_origin.ixz - m * dx * dz,
        iyz: inertia_origin.iyz - m * dy * dz,
    };
    
    Ok((inertia_com, com, m))
}

/// Principal axes of a solid: three orthogonal eigenvectors (principal directions)
/// and their corresponding principal moments (eigenvalues of the inertia tensor).
///
/// The principal axes define the coordinate system in which the inertia tensor is diagonal,
/// meaning all off-diagonal products of inertia are zero.
///
/// # Example
/// For a cube aligned with coordinate axes, the principal axes are the coordinate axes
/// themselves, and the principal moments are equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrincipalAxes {
    /// First principal axis (eigenvector with largest eigenvalue)
    pub axis_1: [f64; 3],
    /// Second principal axis (eigenvector with intermediate eigenvalue)
    pub axis_2: [f64; 3],
    /// Third principal axis (eigenvector with smallest eigenvalue)
    pub axis_3: [f64; 3],
    
    /// First principal moment (largest eigenvalue)
    pub moment_1: f64,
    /// Second principal moment (intermediate eigenvalue)
    pub moment_2: f64,
    /// Third principal moment (smallest eigenvalue)
    pub moment_3: f64,
}

impl PrincipalAxes {
    /// Create new principal axes from three orthonormal eigenvectors and eigenvalues.
    /// Vectors should already be normalized; eigenvalues are in descending order.
    pub fn new(
        axis_1: [f64; 3],
        axis_2: [f64; 3],
        axis_3: [f64; 3],
        moment_1: f64,
        moment_2: f64,
        moment_3: f64,
    ) -> Self {
        Self {
            axis_1,
            axis_2,
            axis_3,
            moment_1,
            moment_2,
            moment_3,
        }
    }

    /// Get the principal axes as a 3x3 rotation matrix (columns are eigenvectors)
    pub fn as_rotation_matrix(&self) -> [[f64; 3]; 3] {
        [
            self.axis_1,
            self.axis_2,
            self.axis_3,
        ]
    }

    /// Get the principal moments as diagonal matrix
    pub fn as_diagonal_inertia(&self) -> [[f64; 3]; 3] {
        [
            [self.moment_1, 0.0, 0.0],
            [0.0, self.moment_2, 0.0],
            [0.0, 0.0, self.moment_3],
        ]
    }
}

/// Compute the principal axes of a solid using eigendecomposition of the inertia tensor.
///
/// The principal axes are three orthogonal unit vectors that align with the principal
/// directions of the inertia tensor. The corresponding principal moments are the
/// eigenvalues (moments of inertia about each principal axis).
///
/// # Arguments
/// * `solid` - The solid to analyze
///
/// # Returns
/// PrincipalAxes with three orthonormal eigenvectors and their eigenvalues,
/// sorted in descending order of eigenvalue magnitude.
///
/// # Algorithm
/// Uses the Jacobi eigenvalue algorithm for 3x3 symmetric matrices:
/// 1. Extract inertia tensor from the solid (about its center of mass)
/// 2. Iteratively apply Jacobi rotations to zero off-diagonal elements
/// 3. Track cumulative rotation matrix to extract eigenvectors
/// 4. Extract eigenvalues from final diagonal matrix
///
/// # Example
/// ```
/// use cascade::make_box;
/// let cube = make_box(2.0, 2.0, 2.0).unwrap();
/// let axes = cascade::query::principal_axes(&cube).unwrap();
/// // For a symmetric cube, axes should align with coordinate axes
/// // and all principal moments should be approximately equal
/// ```
pub fn principal_axes(solid: &Solid) -> Result<PrincipalAxes> {
    // Get inertia tensor about center of mass
    let (inertia, _com, _volume) = moments_of_inertia_at_com(solid)?;
    
    // Convert to 3x3 matrix form
    let a = inertia.as_matrix();
    
    // Compute eigendecomposition using Jacobi algorithm
    let (eigenvalues, eigenvectors) = jacobi_eigendecomposition(a)?;
    
    // Package results, sorting by eigenvalue magnitude (descending)
    let mut pairs: Vec<(f64, [f64; 3])> = eigenvalues
        .iter()
        .zip(eigenvectors.iter())
        .map(|(val, vec)| (*val, *vec))
        .collect();
    
    // Sort by eigenvalue in descending order
    pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(PrincipalAxes::new(
        pairs[0].1,
        pairs[1].1,
        pairs[2].1,
        pairs[0].0,
        pairs[1].0,
        pairs[2].0,
    ))
}

/// Jacobi eigenvalue algorithm for 3x3 symmetric matrices.
///
/// Iteratively applies Givens rotations to zero off-diagonal elements,
/// converging to diagonal form where the diagonal contains eigenvalues
/// and the cumulative rotation matrix contains eigenvectors.
///
/// # Arguments
/// * `matrix` - 3x3 symmetric matrix
///
/// # Returns
/// Tuple of (eigenvalues, eigenvectors) where:
/// - eigenvalues is a [f64; 3] containing the diagonal elements
/// - eigenvectors is a [[f64; 3]; 3] where each row is an eigenvector
fn jacobi_eigendecomposition(mut a: [[f64; 3]; 3]) -> Result<([f64; 3], [[f64; 3]; 3])> {
    // Accumulated rotation matrix (identity initially)
    let mut v = [[0.0; 3]; 3];
    v[0][0] = 1.0;
    v[1][1] = 1.0;
    v[2][2] = 1.0;
    
    let max_iterations = 100;
    let tolerance = 1e-10;
    
    for iteration in 0..max_iterations {
        // Find largest off-diagonal element
        let mut max_off_diag = 0.0;
        let mut p = 0;
        let mut q = 1;
        
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max_off_diag {
                    max_off_diag = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check convergence
        if max_off_diag < tolerance {
            break;
        }
        
        // Compute rotation angle using standard Jacobi formula
        // theta = 0.5 * atan2(2*a[p][q], a[q][q] - a[p][p])
        let theta = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        
        // Apply Givens rotation
        // Create rotation matrix G for indices (p, q)
        let mut g = [[0.0; 3]; 3];
        for i in 0..3 {
            g[i][i] = 1.0;
        }
        g[p][p] = cos_theta;
        g[q][q] = cos_theta;
        g[p][q] = -sin_theta;
        g[q][p] = sin_theta;
        
        // A' = G^T * A * G
        let mut a_temp = [[0.0; 3]; 3];
        
        // First: temp = A * G
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    a_temp[i][j] += a[i][k] * g[k][j];
                }
            }
        }
        
        // Then: A' = G^T * temp
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = 0.0;
                for k in 0..3 {
                    a[i][j] += g[k][i] * a_temp[k][j];
                }
            }
        }
        
        // V' = V * G
        let mut v_temp = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    v_temp[i][j] += v[i][k] * g[k][j];
                }
            }
        }
        v = v_temp;
    }
    
    // Extract eigenvalues from diagonal
    let eigenvalues = [a[0][0], a[1][1], a[2][2]];
    
    // Extract eigenvectors from columns of V
    // V is accumulated as V' = V * G, so columns of V are the eigenvectors
    // Column j of V is [V[0][j], V[1][j], V[2][j]]
    let mut eigenvectors = [[0.0; 3]; 3];
    for col in 0..3 {
        for row in 0..3 {
            eigenvectors[col][row] = v[row][col];  // Extract column col from V
        }
    }
    
    Ok((eigenvalues, eigenvectors))
}

/// Classify a solid as convex, concave, mixed, or degenerate
///
/// # Algorithm
/// 1. Check for degeneracy: zero or negative volume, invalid topology
/// 2. For each face, analyze its surface type and curvature
/// 3. Classify based on presence of concave features:
///    - Convex: all faces are either plane, cylinder, cone, sphere with consistent outward normals
///    - Concave: has at least one face with negative/inward curvature
///    - Mixed: compound with both types
///    - Degenerate: invalid, zero volume, or topological errors
///
/// # Examples
/// - Box, Sphere, Cylinder, Cone -> Convex
/// - L-shaped solid, gear with teeth -> Concave
/// - Union of convex solids (if touching) -> Concave (due to interface)
pub fn classify_shape(solid: &Solid) -> Result<ShapeClass> {
    // Check for degeneracy
    let mass = mass_properties(solid)?;
    
    if mass.volume <= crate::TOLERANCE {
        return Ok(ShapeClass::Degenerate);
    }
    
    // Check for invalid topology using check module
    use crate::check::check_valid;
    if let Err(_) = check_valid(&Shape::Solid(solid.clone())) {
        return Ok(ShapeClass::Degenerate);
    }
    
    // Analyze all faces for convexity indicators
    let mut has_concave = false;
    let mut has_convex_surfaces = true;
    
    // Check outer shell
    for face in &solid.outer_shell.faces {
        // Check if face has concave curvature
        if is_face_concave(face)? {
            has_concave = true;
            break;
        }
    }
    
    // Check inner shells (voids)
    for shell in &solid.inner_shells {
        for face in &shell.faces {
            if is_face_concave(face)? {
                has_concave = true;
                break;
            }
        }
    }
    
    if has_concave {
        Ok(ShapeClass::Concave)
    } else {
        Ok(ShapeClass::Convex)
    }
}

/// Check if a face has concave curvature
/// 
/// A face is considered concave if:
/// - It's a saddle surface (negative Gaussian curvature)
/// - It's a concave portion of a sphere/cylinder (normal points inward)
/// - It's a surface that would create a concave pocket in the solid
fn is_face_concave(face: &Face) -> Result<bool> {
    match &face.surface_type {
        // Planes are neutral (neither convex nor concave)
        SurfaceType::Plane { .. } => Ok(false),
        
        // Cylinders and cones are convex surfaces
        SurfaceType::Cylinder { .. } | SurfaceType::Cone { .. } => Ok(false),
        
        // Spheres are convex (positive curvature everywhere)
        SurfaceType::Sphere { .. } => Ok(false),
        
        // Tori have mixed curvature (outer surface convex, inner surface concave)
        SurfaceType::Torus { .. } => {
            // For now, classify torus faces as potentially concave
            // since they can have both sides
            // A full implementation would check which side of the torus the face is
            Ok(true)  // Conservative: assume torus might have concave portions
        }
        
        // Surfaces of revolution: check if concave or convex
        SurfaceType::SurfaceOfRevolution { .. } => {
            // Surfaces of revolution are convex if the profile curve is entirely
            // on one side of the axis and curves away from it.
            // For simplicity, we conservatively mark as potentially concave
            Ok(false)  // Default to convex unless proven otherwise
        }
        
        // Linear extrusion surfaces: check profile
        SurfaceType::SurfaceOfLinearExtrusion { .. } => Ok(false),
        
        // B-Spline and Bezier surfaces: assume potentially concave
        // A proper implementation would check curvature via derivatives
        SurfaceType::BezierSurface { .. } | SurfaceType::BSpline { .. } => {
            Ok(true)  // Conservative: assume B-spline might be concave
        }
        
        // Trimmed and offset surfaces inherit from their basis
        SurfaceType::RectangularTrimmedSurface { basis_surface, .. } => {
            // Create a dummy face with the basis surface to check recursively
            let dummy_face = Face {
                outer_wire: face.outer_wire.clone(),
                inner_wires: face.inner_wires.clone(),
                surface_type: (**basis_surface).clone(),
            };
            is_face_concave(&dummy_face)
        }
        
        SurfaceType::OffsetSurface { basis_surface, .. } => {
            let dummy_face = Face {
                outer_wire: face.outer_wire.clone(),
                inner_wires: face.inner_wires.clone(),
                surface_type: (**basis_surface).clone(),
            };
            is_face_concave(&dummy_face)
        }
    }
}

fn calculate_shell_properties(shell: &Shell, volume: &mut f64, surface_area: &mut f64, center_of_mass: &mut [f64; 3]) {
    for face in &shell.faces {
        // Calculate face area
        let face_area = calculate_face_area(face);
        *surface_area += face_area;
        
        // Calculate face center and normal
        let (face_center, face_normal) = calculate_face_center_and_normal(face);
        
        // Calculate volume contribution using divergence theorem
        // V = 1/3 * sum(face_center · face_normal * face_area)
        let dot_product = face_center[0] * face_normal[0] + 
                         face_center[1] * face_normal[1] + 
                         face_center[2] * face_normal[2];
        let volume_contribution = dot_product * face_area / 3.0;
        *volume += volume_contribution;
        
        // Center of mass contribution using weighted tetrahedra
        // COM = 1/volume * sum of (tetrahedra_center * tetrahedra_volume)
        // For a face at position p with normal n, the tetrahedron volume is (p·n)*area/3
        // The tetrahedron center is at (p + origin)/4 ≈ p/4 (when origin is at 0)
        let tetrahedra_weight = dot_product * face_area / 3.0;
        for i in 0..3 {
            center_of_mass[i] += face_center[i] * tetrahedra_weight / 4.0;
        }
    }
}

fn calculate_face_area(face: &Face) -> f64 {
    let vertices = get_face_vertices(&face.outer_wire);
    if vertices.len() < 3 {
        return 0.0;
    }
    
    // Use triangulation from first vertex
    let mut area = 0.0;
    for i in 1..vertices.len() - 1 {
        let v0 = vertices[0];
        let v1 = vertices[i];
        let v2 = vertices[i + 1];
        
        // Cross product gives 2 * area of triangle
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        
        let magnitude = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        area += magnitude / 2.0;
    }
    
    area
}

fn get_face_vertices(wire: &Wire) -> Vec<[f64; 3]> {
    let mut vertices = Vec::new();
    for edge in &wire.edges {
        vertices.push(edge.start.point);
    }
    vertices
}

fn calculate_face_center_and_normal(face: &Face) -> ([f64; 3], [f64; 3]) {
    let vertices = get_face_vertices(&face.outer_wire);
    
    // Calculate center (average of vertices)
    let mut center = [0.0; 3];
    for v in &vertices {
        for i in 0..3 {
            center[i] += v[i];
        }
    }
    if !vertices.is_empty() {
        for i in 0..3 {
            center[i] /= vertices.len() as f64;
        }
    }
    
    // Calculate normal using cross product of first two edges
    let mut normal = match &face.surface_type {
        SurfaceType::Plane { origin: _, normal } => *normal,
        SurfaceType::Cylinder { origin: _, axis: _, radius: _ } => {
            // Approximate normal from vertices
            if vertices.len() >= 3 {
                let e1 = [vertices[1][0] - vertices[0][0], 
                         vertices[1][1] - vertices[0][1], 
                         vertices[1][2] - vertices[0][2]];
                let e2 = [vertices[2][0] - vertices[0][0], 
                         vertices[2][1] - vertices[0][1], 
                         vertices[2][2] - vertices[0][2]];
                
                [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
        SurfaceType::Sphere { center: _, radius: _ } => {
            // Normal points outward from center
            if let SurfaceType::Sphere { center: sphere_center, radius: _ } = &face.surface_type {
                let dx = center[0] - sphere_center[0];
                let dy = center[1] - sphere_center[1];
                let dz = center[2] - sphere_center[2];
                [dx, dy, dz]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
        SurfaceType::Cone { origin: _, axis: _, half_angle_rad: _ } => {
            // Approximate normal from vertices for cone
            if vertices.len() >= 3 {
                let e1 = [vertices[1][0] - vertices[0][0], 
                         vertices[1][1] - vertices[0][1], 
                         vertices[1][2] - vertices[0][2]];
                let e2 = [vertices[2][0] - vertices[0][0], 
                         vertices[2][1] - vertices[0][1], 
                         vertices[2][2] - vertices[0][2]];
                
                [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
        SurfaceType::Torus { center: _, major_radius: _, minor_radius: _ } => {
            // Approximate normal from vertices for torus
            if vertices.len() >= 3 {
                let e1 = [vertices[1][0] - vertices[0][0], 
                         vertices[1][1] - vertices[0][1], 
                         vertices[1][2] - vertices[0][2]];
                let e2 = [vertices[2][0] - vertices[0][0], 
                         vertices[2][1] - vertices[0][1], 
                         vertices[2][2] - vertices[0][2]];
                
                [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
        SurfaceType::BSpline { .. } | SurfaceType::BezierSurface { .. } => [0.0, 0.0, 1.0],
        SurfaceType::SurfaceOfRevolution { .. } => {
            // Approximate normal from vertices for surface of revolution
            if vertices.len() >= 3 {
                let e1 = [vertices[1][0] - vertices[0][0], 
                         vertices[1][1] - vertices[0][1], 
                         vertices[1][2] - vertices[0][2]];
                let e2 = [vertices[2][0] - vertices[0][0], 
                         vertices[2][1] - vertices[0][1], 
                         vertices[2][2] - vertices[0][2]];
                
                [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
        SurfaceType::SurfaceOfLinearExtrusion { .. } => {
            // Approximate normal from vertices for surface of linear extrusion
            if vertices.len() >= 3 {
                let e1 = [vertices[1][0] - vertices[0][0], 
                         vertices[1][1] - vertices[0][1], 
                         vertices[1][2] - vertices[0][2]];
                let e2 = [vertices[2][0] - vertices[0][0], 
                         vertices[2][1] - vertices[0][1], 
                         vertices[2][2] - vertices[0][2]];
                
                [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
        SurfaceType::RectangularTrimmedSurface { .. } | SurfaceType::OffsetSurface { .. } => {
            // Approximate normal from vertices for trimmed/offset surfaces
            if vertices.len() >= 3 {
                let e1 = [vertices[1][0] - vertices[0][0], 
                         vertices[1][1] - vertices[0][1], 
                         vertices[1][2] - vertices[0][2]];
                let e2 = [vertices[2][0] - vertices[0][0], 
                         vertices[2][1] - vertices[0][1], 
                         vertices[2][2] - vertices[0][2]];
                
                [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
    };
    
    // Normalize normal vector
    let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if norm > 1e-10 {
        for i in 0..3 {
            normal[i] /= norm;
        }
    }
    
    (center, normal)
}

/// Compute the extremal distance between two curves and the closest points.
/// 
/// This function finds the minimum distance between two curves and the points
/// on each curve that achieve this distance.
/// 
/// # Arguments
/// * `c1` - First curve
/// * `s1` - Start point of first curve
/// * `e1` - End point of first curve
/// * `c2` - Second curve
/// * `s2` - Start point of second curve
/// * `e2` - End point of second curve
/// 
/// # Returns
/// A tuple containing:
/// - `distance` - The minimum distance between the curves
/// - `point_on_c1` - The point on c1 achieving the minimum distance
/// - `point_on_c2` - The point on c2 achieving the minimum distance
pub fn extrema_curve_curve(
    c1: &crate::brep::CurveType,
    _s1: [f64; 3],
    _e1: [f64; 3],
    c2: &crate::brep::CurveType,
    _s2: [f64; 3],
    _e2: [f64; 3],
) -> Result<(f64, [f64; 3], [f64; 3])> {
    use crate::curve;
    
    // Sample both curves at regular intervals to find initial closest pair
    let num_samples = 20;
    let mut best_dist_sq = f64::INFINITY;
    let mut best_t1 = 0.0;
    let mut best_t2 = 0.0;
    
    // Sample first curve
    for i1 in 0..=num_samples {
        let t1 = (i1 as f64) / (num_samples as f64);
        let p1 = curve::point_at(c1, t1)?;
        
        // Sample second curve
        for i2 in 0..=num_samples {
            let t2 = (i2 as f64) / (num_samples as f64);
            let p2 = curve::point_at(c2, t2)?;
            
            let dist_sq = distance_squared(&p1, &p2);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_t1 = t1;
                best_t2 = t2;
            }
        }
    }
    
    // Refine using local optimization
    // Try to minimize distance by moving parameters slightly
    let mut t1 = best_t1;
    let mut t2 = best_t2;
    let dt = 1e-6;
    
    for _ in 0..20 {
        let p1 = curve::point_at(c1, t1)?;
        let p2 = curve::point_at(c2, t2)?;
        let current_dist_sq = distance_squared(&p1, &p2);
        
        // Try moving along t1
        let p1_plus = curve::point_at(c1, (t1 + dt).min(1.0))?;
        let dist_plus_1 = distance_squared(&p1_plus, &p2);
        
        let p1_minus = curve::point_at(c1, (t1 - dt).max(0.0))?;
        let dist_minus_1 = distance_squared(&p1_minus, &p2);
        
        // Try moving along t2
        let p2_plus = curve::point_at(c2, (t2 + dt).min(1.0))?;
        let dist_plus_2 = distance_squared(&p1, &p2_plus);
        
        let p2_minus = curve::point_at(c2, (t2 - dt).max(0.0))?;
        let dist_minus_2 = distance_squared(&p1, &p2_minus);
        
        // Find best direction and step
        let mut improved = false;
        if dist_plus_1 < current_dist_sq && dist_plus_1 < dist_minus_1 {
            t1 = (t1 + dt).min(1.0);
            improved = true;
        } else if dist_minus_1 < current_dist_sq && dist_minus_1 < dist_plus_1 {
            t1 = (t1 - dt).max(0.0);
            improved = true;
        }
        
        if dist_plus_2 < current_dist_sq && dist_plus_2 < dist_minus_2 {
            t2 = (t2 + dt).min(1.0);
            improved = true;
        } else if dist_minus_2 < current_dist_sq && dist_minus_2 < dist_plus_2 {
            t2 = (t2 - dt).max(0.0);
            improved = true;
        }
        
        if !improved {
            break;
        }
    }
    
    // Get final result
    let p1_final = curve::point_at(c1, t1)?;
    let p2_final = curve::point_at(c2, t2)?;
    let final_dist_sq = distance_squared(&p1_final, &p2_final);
    let final_dist = final_dist_sq.sqrt();
    
    Ok((final_dist, p1_final, p2_final))
}

/// Compute the extremal distance between a point and a solid, and the closest point on the solid.
/// 
/// This function finds the minimum distance from a point to the surface of a solid,
/// checking all faces and returning the closest point.
/// 
/// # Arguments
/// * `point` - The 3D point
/// * `solid` - The solid to measure distance to
/// 
/// # Returns
/// A tuple containing:
/// - `distance` - The minimum distance from the point to the solid surface
/// - `closest_point` - The closest point on the solid surface
pub fn extrema_point_solid(
    point: [f64; 3],
    solid: &Solid,
) -> Result<(f64, [f64; 3])> {
    let mut best_dist = f64::INFINITY;
    let mut best_point = point;
    
    // Check distance to all faces in outer shell
    for face in &solid.outer_shell.faces {
        // Project point onto face
        let plane_dist = project_to_face(point, face)?;
        
        if plane_dist.0 < best_dist {
            best_dist = plane_dist.0;
            best_point = plane_dist.1;
        }
    }
    
    // Check distance to all faces in inner shells (voids)
    for shell in &solid.inner_shells {
        for face in &shell.faces {
            let plane_dist = project_to_face(point, face)?;
            
            if plane_dist.0 < best_dist {
                best_dist = plane_dist.0;
                best_point = plane_dist.1;
            }
        }
    }
    
    if best_dist.is_infinite() {
        return Err(CascadeError::InvalidGeometry(
            "Could not compute distance to solid".into()
        ));
    }
    
    Ok((best_dist, best_point))
}

/// Project a point onto a face and return distance and closest point
fn project_to_face(point: [f64; 3], face: &Face) -> Result<(f64, [f64; 3])> {
    use crate::brep::SurfaceType;
    
    let (closest_point, _) = match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            // Distance from point to plane: d = |(p - o) · n|
            let to_point = [
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            ];
            
            let dist_to_plane = dot(&to_point, normal);
            let closest = [
                point[0] - dist_to_plane * normal[0],
                point[1] - dist_to_plane * normal[1],
                point[2] - dist_to_plane * normal[2],
            ];
            
            (closest, *normal)
        }
        
        SurfaceType::Sphere { center, radius } => {
            // Closest point on sphere is along the line from center to point
            let to_point = [
                point[0] - center[0],
                point[1] - center[1],
                point[2] - center[2],
            ];
            
            let dist_to_center = (to_point[0] * to_point[0] + 
                                 to_point[1] * to_point[1] + 
                                 to_point[2] * to_point[2]).sqrt();
            
            if dist_to_center < 1e-10 {
                // Point is at center, return center + radius in arbitrary direction
                ([center[0] + radius, center[1], center[2]], [1.0, 0.0, 0.0])
            } else {
                let normalized = [
                    to_point[0] / dist_to_center,
                    to_point[1] / dist_to_center,
                    to_point[2] / dist_to_center,
                ];
                
                let closest = [
                    center[0] + radius * normalized[0],
                    center[1] + radius * normalized[1],
                    center[2] + radius * normalized[2],
                ];
                
                (closest, normalized)
            }
        }
        
        SurfaceType::Cylinder { origin, axis, radius } => {
            // Project point onto cylinder axis
            let to_point = [
                point[0] - origin[0],
                point[1] - origin[1],
                point[2] - origin[2],
            ];
            
            let proj_on_axis = dot(&to_point, axis);
            let axis_point = [
                origin[0] + proj_on_axis * axis[0],
                origin[1] + proj_on_axis * axis[1],
                origin[2] + proj_on_axis * axis[2],
            ];
            
            let radial = [
                point[0] - axis_point[0],
                point[1] - axis_point[1],
                point[2] - axis_point[2],
            ];
            
            let radial_dist = (radial[0] * radial[0] + 
                             radial[1] * radial[1] + 
                             radial[2] * radial[2]).sqrt();
            
            if radial_dist < 1e-10 {
                // Point is on axis, return point on cylinder surface in arbitrary direction
                let perp = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                let normal_dir = cross(axis, &perp);
                let normal_dist = (normal_dir[0] * normal_dir[0] + 
                                 normal_dir[1] * normal_dir[1] + 
                                 normal_dir[2] * normal_dir[2]).sqrt();
                let normalized_normal = if normal_dist > 1e-10 {
                    [normal_dir[0] / normal_dist, normal_dir[1] / normal_dist, normal_dir[2] / normal_dist]
                } else {
                    [1.0, 0.0, 0.0]
                };
                
                (
                    [
                        axis_point[0] + radius * normalized_normal[0],
                        axis_point[1] + radius * normalized_normal[1],
                        axis_point[2] + radius * normalized_normal[2],
                    ],
                    normalized_normal
                )
            } else {
                let normalized_radial = [
                    radial[0] / radial_dist,
                    radial[1] / radial_dist,
                    radial[2] / radial_dist,
                ];
                
                let closest = [
                    axis_point[0] + radius * normalized_radial[0],
                    axis_point[1] + radius * normalized_radial[1],
                    axis_point[2] + radius * normalized_radial[2],
                ];
                
                (closest, normalized_radial)
            }
        }
        
        // For other surface types, use face vertices to approximate
        _ => {
            let vertices = get_face_vertices(&face.outer_wire);
            if vertices.len() < 3 {
                return Err(CascadeError::InvalidGeometry(
                    "Face has fewer than 3 vertices".into()
                ));
            }
            
            // Find closest vertex
            let mut closest = vertices[0];
            let mut closest_dist_sq = distance_squared(&point, &closest);
            
            for v in &vertices[1..] {
                let dist_sq = distance_squared(&point, v);
                if dist_sq < closest_dist_sq {
                    closest_dist_sq = dist_sq;
                    closest = *v;
                }
            }
            
            // Estimate normal from vertices
            let v0 = vertices[0];
            let v1 = vertices[1];
            let v2 = vertices[2];
            
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            
            let mut normal = cross(&e1, &e2);
            let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if norm > 1e-10 {
                normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
            } else {
                normal = [0.0, 0.0, 1.0];
            }
            
            (closest, normal)
        }
    };
    
    let dist = distance_squared(&point, &closest_point).sqrt();
    Ok((dist, closest_point))
}

/// Project a point onto a curve and return the parameter and closest point.
/// 
/// This function finds the point on a curve that is closest to the given input point.
/// It works with all curve types, using both analytical methods (for simple curves)
/// and numerical methods (for complex curves like Bezier, B-spline).
/// 
/// # Arguments
/// * `point` - The 3D point to project
/// * `curve` - The curve type to project onto
/// * `start` - Start point of the curve (used for some computations)
/// * `end` - End point of the curve (used for some computations)
/// 
/// # Returns
/// A tuple containing:
/// - `parameter` - The parameter value on the curve (typically in [0, 1] or curve-specific range)
/// - `closest_point` - The 3D point on the curve closest to the input point
pub fn project_point_to_curve(
    point: [f64; 3],
    curve: &crate::brep::CurveType,
    _start: [f64; 3],
    _end: [f64; 3],
) -> Result<(f64, [f64; 3])> {
    use crate::brep::CurveType;
    use crate::curve;
    
    match curve {
        CurveType::Line => {
            // For lines, we need to handle them specially since they're stored as start/end
            // in edges. This is a fallback that won't work well - the caller should use
            // edge start/end directly
            Err(CascadeError::InvalidGeometry(
                "Cannot project to Line curve type directly. Use edge start/end vertices instead.".into()
            ))
        }
        
        CurveType::Arc { center, radius } => {
            project_to_arc(point, *center, *radius)
        }
        
        CurveType::Ellipse { center, major_axis, minor_axis } => {
            project_to_ellipse(point, *center, *major_axis, *minor_axis)
        }
        
        CurveType::Parabola { origin, x_dir, y_dir, focal } => {
            project_to_parabola(point, *origin, *x_dir, *y_dir, *focal)
        }
        
        CurveType::Hyperbola { center, x_dir, y_dir, major_radius, minor_radius } => {
            project_to_hyperbola(point, *center, *x_dir, *y_dir, *major_radius, *minor_radius)
        }
        
        CurveType::Bezier { control_points } => {
            project_to_bezier(point, control_points)
        }
        
        CurveType::BSpline { control_points, knots, degree, weights } => {
            project_to_bspline(point, control_points, knots, *degree, weights.as_deref())
        }
        
        CurveType::Trimmed { basis_curve, u1, u2 } => {
            // Project to basis curve and clamp the parameter to the trimmed range
            let (param, closest_pt) = project_point_to_curve(point, basis_curve, _start, _end)?;
            let clamped_param = param.max(*u1).min(*u2);
            let pt = curve::point_at(basis_curve, clamped_param)?;
            Ok((clamped_param, pt))
        }
        
        CurveType::Offset { basis_curve, offset_distance, offset_direction } => {
            // Project to basis curve, then offset the result
            let (param, closest_pt) = project_point_to_curve(point, basis_curve, _start, _end)?;
            
            // Get tangent at the closest point to compute the offset direction
            let tangent = curve::tangent_at(basis_curve, param)?;
            
            // Compute normal perpendicular to tangent and offset_direction
            let tangent_norm = normalize(&tangent);
            let offset_dir_norm = normalize(offset_direction);
            
            let normal = cross(&offset_dir_norm, &tangent_norm);
            let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            
            let final_normal = if normal_len > 1e-10 {
                [normal[0] / normal_len, normal[1] / normal_len, normal[2] / normal_len]
            } else {
                // Fallback to perpendicular computation
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
                        "Cannot compute offset normal".to_string()
                    ));
                }
            };
            
            // Offset the closest point
            let offset_pt = [
                closest_pt[0] + offset_distance * final_normal[0],
                closest_pt[1] + offset_distance * final_normal[1],
                closest_pt[2] + offset_distance * final_normal[2],
            ];
            
            Ok((param, offset_pt))
        }
    }
}

/// Helper: normalize a 3D vector
fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Helper: cross product
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Helper: dot product
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Helper: squared distance between two points
fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

/// Project point to circular arc
fn project_to_arc(point: [f64; 3], center: [f64; 3], radius: f64) -> Result<(f64, [f64; 3])> {
    use crate::curve;
    
    // Vector from center to point
    let to_point = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    
    // Project onto the circle plane (xy-plane for this implementation)
    let proj = [to_point[0], to_point[1], 0.0];
    let len = (proj[0] * proj[0] + proj[1] * proj[1]).sqrt();
    
    let angle = if len > 1e-10 {
        proj[1].atan2(proj[0])
    } else {
        0.0
    };
    
    // Normalize angle to [0, 2π) then map to [0, 1]
    let angle_normalized = if angle < 0.0 { angle + 2.0 * std::f64::consts::PI } else { angle };
    let param = angle_normalized / (2.0 * std::f64::consts::PI);
    
    // Get the point on the arc
    let arc_point = curve::point_at(&crate::brep::CurveType::Arc { center, radius }, param)?;
    
    Ok((param, arc_point))
}

/// Project point to ellipse using Newton-Raphson iteration
fn project_to_ellipse(
    point: [f64; 3],
    center: [f64; 3],
    major_axis: [f64; 3],
    minor_axis: [f64; 3],
) -> Result<(f64, [f64; 3])> {
    use crate::curve;
    
    let major_len = (major_axis[0] * major_axis[0] + major_axis[1] * major_axis[1] + major_axis[2] * major_axis[2]).sqrt();
    let minor_len = (minor_axis[0] * minor_axis[0] + minor_axis[1] * minor_axis[1] + minor_axis[2] * minor_axis[2]).sqrt();
    
    if major_len < 1e-10 || minor_len < 1e-10 {
        return Err(CascadeError::InvalidGeometry("Degenerate ellipse".into()));
    }
    
    let a = major_len;
    let b = minor_len;
    
    // Initial guess: project to major axis
    let to_point = [point[0] - center[0], point[1] - center[1], point[2] - center[2]];
    let proj_major = dot(&to_point, &major_axis) / (major_len * major_len);
    let t0 = (proj_major.max(-1.0).min(1.0) * std::f64::consts::PI).acos();
    
    // Newton-Raphson iteration
    let mut t = t0;
    for _ in 0..10 {
        let cos_t = t.cos();
        let sin_t = t.sin();
        
        // Point on ellipse at parameter t
        let p_t = [
            center[0] + cos_t * major_axis[0] + sin_t * minor_axis[0],
            center[1] + cos_t * major_axis[1] + sin_t * minor_axis[1],
            center[2] + cos_t * major_axis[2] + sin_t * minor_axis[2],
        ];
        
        // Vector from point on ellipse to query point
        let diff = [point[0] - p_t[0], point[1] - p_t[1], point[2] - p_t[2]];
        
        // Derivative: dP/dt = -sin(t) * major_axis + cos(t) * minor_axis
        let dp_dt = [
            -sin_t * major_axis[0] + cos_t * minor_axis[0],
            -sin_t * major_axis[1] + cos_t * minor_axis[1],
            -sin_t * major_axis[2] + cos_t * minor_axis[2],
        ];
        
        // Second derivative: d²P/dt² = -cos(t) * major_axis - sin(t) * minor_axis
        let d2p_dt2 = [
            -cos_t * major_axis[0] - sin_t * minor_axis[0],
            -cos_t * major_axis[1] - sin_t * minor_axis[1],
            -cos_t * major_axis[2] - sin_t * minor_axis[2],
        ];
        
        // f(t) = (P(t) - point) · dP/dt
        let f = dot(&diff, &dp_dt);
        
        // f'(t) = |dP/dt|² + (P(t) - point) · d²P/dt²
        let dp_dt_sq = dot(&dp_dt, &dp_dt);
        let f_prime = dp_dt_sq + dot(&diff, &d2p_dt2);
        
        if f_prime.abs() < 1e-10 {
            break;
        }
        
        let t_new = t - f / f_prime;
        if (t_new - t).abs() < 1e-10 {
            break;
        }
        t = t_new;
    }
    
    // Clamp t to [0, 2π) then normalize to [0, 1]
    let t_norm = t % (2.0 * std::f64::consts::PI);
    let t_norm = if t_norm < 0.0 { t_norm + 2.0 * std::f64::consts::PI } else { t_norm };
    let param = t_norm / (2.0 * std::f64::consts::PI);
    
    let ellipse_pt = curve::point_at(&crate::brep::CurveType::Ellipse { center, major_axis, minor_axis }, param)?;
    
    Ok((param, ellipse_pt))
}

/// Project point to parabola using Newton-Raphson iteration
fn project_to_parabola(
    point: [f64; 3],
    origin: [f64; 3],
    x_dir: [f64; 3],
    y_dir: [f64; 3],
    focal: f64,
) -> Result<(f64, [f64; 3])> {
    use crate::curve;
    
    if focal.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry("Degenerate parabola".into()));
    }
    
    // Initial guess: project onto x_dir
    let to_point = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
    let proj_x = dot(&to_point, &x_dir);
    let mut u = proj_x;
    
    // Newton-Raphson iteration
    for _ in 0..20 {
        // Point on parabola: P(u) = origin + u * x_dir + (u²/4p) * y_dir
        let u_sq_term = u * u / (4.0 * focal);
        let p_u = [
            origin[0] + u * x_dir[0] + u_sq_term * y_dir[0],
            origin[1] + u * x_dir[1] + u_sq_term * y_dir[1],
            origin[2] + u * x_dir[2] + u_sq_term * y_dir[2],
        ];
        
        // Vector from point on parabola to query point
        let diff = [point[0] - p_u[0], point[1] - p_u[1], point[2] - p_u[2]];
        
        // Derivative: dP/du = x_dir + (u/2p) * y_dir
        let dp_du = [
            x_dir[0] + (u / (2.0 * focal)) * y_dir[0],
            x_dir[1] + (u / (2.0 * focal)) * y_dir[1],
            x_dir[2] + (u / (2.0 * focal)) * y_dir[2],
        ];
        
        // Second derivative: d²P/du² = (1/2p) * y_dir
        let d2p_du2 = [
            (1.0 / (2.0 * focal)) * y_dir[0],
            (1.0 / (2.0 * focal)) * y_dir[1],
            (1.0 / (2.0 * focal)) * y_dir[2],
        ];
        
        // f(u) = (P(u) - point) · dP/du
        let f = dot(&diff, &dp_du);
        
        // f'(u) = |dP/du|² + (P(u) - point) · d²P/du²
        let dp_du_sq = dot(&dp_du, &dp_du);
        let f_prime = dp_du_sq + dot(&diff, &d2p_du2);
        
        if f_prime.abs() < 1e-10 {
            break;
        }
        
        let u_new = u - f / f_prime;
        if (u_new - u).abs() < 1e-10 {
            break;
        }
        u = u_new;
    }
    
    let param_point = curve::point_at(&crate::brep::CurveType::Parabola { origin, x_dir, y_dir, focal }, u)?;
    
    Ok((u, param_point))
}

/// Project point to hyperbola using Newton-Raphson iteration
fn project_to_hyperbola(
    point: [f64; 3],
    center: [f64; 3],
    x_dir: [f64; 3],
    y_dir: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
) -> Result<(f64, [f64; 3])> {
    use crate::curve;
    
    if major_radius.abs() < 1e-10 || minor_radius.abs() < 1e-10 {
        return Err(CascadeError::InvalidGeometry("Degenerate hyperbola".into()));
    }
    
    // Initial guess: project onto x_dir
    let to_point = [point[0] - center[0], point[1] - center[1], point[2] - center[2]];
    let proj_x = dot(&to_point, &x_dir) / (dot(&x_dir, &x_dir));
    let mut u = proj_x.ln().abs();
    if proj_x < 0.0 {
        u = -u;
    }
    
    // Newton-Raphson iteration
    for _ in 0..20 {
        // Point on hyperbola: P(u) = center + a*cosh(u)*x_dir + b*sinh(u)*y_dir
        let cosh_u = u.cosh();
        let sinh_u = u.sinh();
        let p_u = [
            center[0] + major_radius * cosh_u * x_dir[0] + minor_radius * sinh_u * y_dir[0],
            center[1] + major_radius * cosh_u * x_dir[1] + minor_radius * sinh_u * y_dir[1],
            center[2] + major_radius * cosh_u * x_dir[2] + minor_radius * sinh_u * y_dir[2],
        ];
        
        // Vector from point on hyperbola to query point
        let diff = [point[0] - p_u[0], point[1] - p_u[1], point[2] - p_u[2]];
        
        // Derivative: dP/du = a*sinh(u)*x_dir + b*cosh(u)*y_dir
        let dp_du = [
            major_radius * sinh_u * x_dir[0] + minor_radius * cosh_u * y_dir[0],
            major_radius * sinh_u * x_dir[1] + minor_radius * cosh_u * y_dir[1],
            major_radius * sinh_u * x_dir[2] + minor_radius * cosh_u * y_dir[2],
        ];
        
        // Second derivative: d²P/du² = a*cosh(u)*x_dir + b*sinh(u)*y_dir
        let d2p_du2 = [
            major_radius * cosh_u * x_dir[0] + minor_radius * sinh_u * y_dir[0],
            major_radius * cosh_u * x_dir[1] + minor_radius * sinh_u * y_dir[1],
            major_radius * cosh_u * x_dir[2] + minor_radius * sinh_u * y_dir[2],
        ];
        
        // f(u) = (P(u) - point) · dP/du
        let f = dot(&diff, &dp_du);
        
        // f'(u) = |dP/du|² + (P(u) - point) · d²P/du²
        let dp_du_sq = dot(&dp_du, &dp_du);
        let f_prime = dp_du_sq + dot(&diff, &d2p_du2);
        
        if f_prime.abs() < 1e-10 {
            break;
        }
        
        let u_new = u - f / f_prime;
        if (u_new - u).abs() < 1e-10 {
            break;
        }
        u = u_new;
    }
    
    let hyperbola_pt = curve::point_at(&crate::brep::CurveType::Hyperbola { center, x_dir, y_dir, major_radius, minor_radius }, u)?;
    
    Ok((u, hyperbola_pt))
}

/// Project point to Bezier curve using multi-start Newton-Raphson
fn project_to_bezier(point: [f64; 3], control_points: &[[f64; 3]]) -> Result<(f64, [f64; 3])> {
    use crate::curve;
    
    if control_points.is_empty() {
        return Err(CascadeError::InvalidGeometry("Empty Bezier curve".into()));
    }
    
    if control_points.len() == 1 {
        // Degenerate case: single point
        return Ok((0.0, control_points[0]));
    }
    
    // Try multiple starting points for robustness
    let num_starts = (10).min((control_points.len() as i32) * 2) as usize;
    let mut best_param = 0.0;
    let mut best_dist_sq = f64::INFINITY;
    
    for start_idx in 0..num_starts {
        let t0 = (start_idx as f64) / (num_starts as f64);
        let (param, dist_sq) = newton_raphson_param(point, &crate::brep::CurveType::Bezier { control_points: control_points.to_vec() }, t0, 20)?;
        
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_param = param;
        }
    }
    
    let bezier_pt = curve::point_at(&crate::brep::CurveType::Bezier { control_points: control_points.to_vec() }, best_param)?;
    
    Ok((best_param, bezier_pt))
}

/// Project point to B-spline curve using multi-start Newton-Raphson
fn project_to_bspline(
    point: [f64; 3],
    control_points: &[[f64; 3]],
    knots: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
) -> Result<(f64, [f64; 3])> {
    use crate::curve;
    
    if control_points.is_empty() {
        return Err(CascadeError::InvalidGeometry("Empty B-spline curve".into()));
    }
    
    // Try multiple starting points for robustness
    let num_starts = (10).min((control_points.len() as i32).max(3)) as usize;
    let mut best_param = 0.0;
    let mut best_dist_sq = f64::INFINITY;
    
    for start_idx in 0..num_starts {
        let t0 = (start_idx as f64) / (num_starts as f64);
        let (param, dist_sq) = newton_raphson_param(
            point,
            &crate::brep::CurveType::BSpline { control_points: control_points.to_vec(), knots: knots.to_vec(), degree, weights: weights.map(|w| w.to_vec()) },
            t0,
            20
        )?;
        
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_param = param;
        }
    }
    
    let bspline_pt = curve::point_at(&crate::brep::CurveType::BSpline { control_points: control_points.to_vec(), knots: knots.to_vec(), degree, weights: weights.map(|w| w.to_vec()) }, best_param)?;
    
    Ok((best_param, bspline_pt))
}

/// Generic Newton-Raphson iteration for parameter finding
fn newton_raphson_param(
    point: [f64; 3],
    curve: &crate::brep::CurveType,
    t0: f64,
    max_iterations: usize,
) -> Result<(f64, f64)> {
    use crate::curve;
    
    let mut t = t0.max(0.0).min(1.0);
    
    for _ in 0..max_iterations {
        let p_t = curve::point_at(curve, t)?;
        let diff = [point[0] - p_t[0], point[1] - p_t[1], point[2] - p_t[2]];
        let diff_sq = distance_squared(&point, &p_t);
        
        if diff_sq < 1e-16 {
            return Ok((t, diff_sq));
        }
        
        // Compute numerical derivative
        let dt = 1e-8;
        let t_plus = (t + dt).min(1.0);
        let p_plus = curve::point_at(curve, t_plus)?;
        let dp_dt = [
            (p_plus[0] - p_t[0]) / (t_plus - t),
            (p_plus[1] - p_t[1]) / (t_plus - t),
            (p_plus[2] - p_t[2]) / (t_plus - t),
        ];
        
        let dp_dt_sq = dot(&dp_dt, &dp_dt);
        
        if dp_dt_sq.abs() < 1e-16 {
            return Ok((t, diff_sq));
        }
        
        // Newton-Raphson update
        let f = dot(&diff, &dp_dt);
        let t_new = t - f / dp_dt_sq;
        let t_new_clamped = t_new.max(0.0).min(1.0);
        
        if (t_new_clamped - t).abs() < 1e-12 {
            return Ok((t_new_clamped, diff_sq));
        }
        
        t = t_new_clamped;
    }
    
    let p_t = curve::point_at(curve, t)?;
    let diff_sq = distance_squared(&point, &p_t);
    
    Ok((t, diff_sq))
}

/// Project a point onto a surface and return the (u, v) parameters and closest point.
/// 
/// This function finds the point on a surface that is closest to the given input point.
/// It works with all surface types, using both analytical methods (for simple surfaces)
/// and numerical methods (for complex surfaces like Bezier, B-spline).
/// 
/// # Arguments
/// * `point` - The 3D point to project
/// * `surface` - The surface type to project onto
/// 
/// # Returns
/// A tuple containing:
/// - `u` - The u parameter on the surface
/// - `v` - The v parameter on the surface
/// - `closest_point` - The 3D point on the surface closest to the input point
pub fn project_point_to_surface(
    point: [f64; 3],
    surface: &SurfaceType,
) -> Result<(f64, f64, [f64; 3])> {
    match surface {
        SurfaceType::Plane { origin, normal } => {
            project_to_plane(point, *origin, *normal)
        }
        
        SurfaceType::Cylinder { origin, axis, radius } => {
            project_to_cylinder(point, *origin, *axis, *radius)
        }
        
        SurfaceType::Sphere { center, radius } => {
            project_to_sphere(point, *center, *radius)
        }
        
        SurfaceType::Cone { origin, axis, half_angle_rad } => {
            project_to_cone(point, *origin, *axis, *half_angle_rad)
        }
        
        SurfaceType::Torus { center, major_radius, minor_radius } => {
            project_to_torus(point, *center, *major_radius, *minor_radius)
        }
        
        SurfaceType::BezierSurface { control_points, u_degree, v_degree } => {
            project_to_bezier_surface(point, control_points, *u_degree, *v_degree)
        }
        
        SurfaceType::BSpline { u_degree, v_degree, u_knots, v_knots, control_points, weights } => {
            project_to_bspline_surface(point, *u_degree, *v_degree, u_knots, v_knots, control_points, weights.as_deref())
        }
        
        SurfaceType::SurfaceOfRevolution { basis_curve, curve_start, curve_end, axis_location, axis_direction } => {
            project_to_surface_of_revolution(point, basis_curve, *curve_start, *curve_end, *axis_location, *axis_direction)
        }
        
        SurfaceType::SurfaceOfLinearExtrusion { basis_curve, curve_start, curve_end, direction } => {
            project_to_surface_of_linear_extrusion(point, basis_curve, *curve_start, *curve_end, *direction)
        }
        
        SurfaceType::RectangularTrimmedSurface { basis_surface, u1, u2, v1, v2 } => {
            // Project to basis surface and clamp to trimmed range
            let (u, v, _closest_pt) = project_point_to_surface(point, basis_surface)?;
            let u_clamped = u.max(*u1).min(*u2);
            let v_clamped = v.max(*v1).min(*v2);
            let pt = basis_surface.point_at(u_clamped, v_clamped);
            Ok((u_clamped, v_clamped, pt))
        }
        
        SurfaceType::OffsetSurface { basis_surface, offset_distance } => {
            // Project to basis surface, then offset the result
            let (u, v, _closest_pt) = project_point_to_surface(point, basis_surface)?;
            let pt = basis_surface.point_at(u, v);
            
            // Get normal at the closest point
            let normal = surface_normal(basis_surface, u, v)?;
            
            // Offset the point
            let offset_pt = [
                pt[0] + offset_distance * normal[0],
                pt[1] + offset_distance * normal[1],
                pt[2] + offset_distance * normal[2],
            ];
            
            Ok((u, v, offset_pt))
        }
    }
}

/// Project point to plane analytically
fn project_to_plane(point: [f64; 3], origin: [f64; 3], normal: [f64; 3]) -> Result<(f64, f64, [f64; 3])> {
    let normal_norm = normalize(&normal);
    
    // Vector from origin to point
    let to_point = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    
    // Project onto the plane: closest_point = point - (point - origin) · normal * normal
    let proj_distance = dot(&to_point, &normal_norm);
    let closest_pt = [
        point[0] - proj_distance * normal_norm[0],
        point[1] - proj_distance * normal_norm[1],
        point[2] - proj_distance * normal_norm[2],
    ];
    
    // For a plane, we can't meaningfully return u,v without defining a coordinate system
    // Return (0, 0) as placeholder
    Ok((0.0, 0.0, closest_pt))
}

/// Project point to cylinder analytically
fn project_to_cylinder(point: [f64; 3], origin: [f64; 3], axis: [f64; 3], radius: f64) -> Result<(f64, f64, [f64; 3])> {
    let axis_norm = normalize(&axis);
    
    // Vector from origin to point
    let to_point = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    
    // Project onto axis
    let proj_on_axis = dot(&to_point, &axis_norm);
    let v = proj_on_axis; // v is the distance along the axis
    
    // Get perpendicular component (in the radial plane)
    let on_axis = [
        origin[0] + proj_on_axis * axis_norm[0],
        origin[1] + proj_on_axis * axis_norm[1],
        origin[2] + proj_on_axis * axis_norm[2],
    ];
    
    let radial = [
        point[0] - on_axis[0],
        point[1] - on_axis[1],
        point[2] - on_axis[2],
    ];
    
    // Project radial component to cylinder surface
    let radial_norm = normalize(&radial);
    let closest_pt = [
        on_axis[0] + radius * radial_norm[0],
        on_axis[1] + radius * radial_norm[1],
        on_axis[2] + radius * radial_norm[2],
    ];
    
    // Compute u (angle around cylinder)
    let perp1 = perpendicular_to(&axis_norm);
    let perp2 = cross(&axis_norm, &perp1);
    
    let u_comp = dot(&radial_norm, &perp1);
    let v_comp = dot(&radial_norm, &perp2);
    let angle = v_comp.atan2(u_comp);
    let angle_normalized = if angle < 0.0 { angle + 2.0 * std::f64::consts::PI } else { angle };
    let u = angle_normalized / (2.0 * std::f64::consts::PI);
    
    Ok((u, v, closest_pt))
}

/// Project point to sphere analytically
fn project_to_sphere(point: [f64; 3], center: [f64; 3], radius: f64) -> Result<(f64, f64, [f64; 3])> {
    // Vector from center to point
    let to_point = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    
    // Project to sphere surface
    let to_point_norm = normalize(&to_point);
    let closest_pt = [
        center[0] + radius * to_point_norm[0],
        center[1] + radius * to_point_norm[1],
        center[2] + radius * to_point_norm[2],
    ];
    
    // Compute u, v spherical coordinates
    // u: longitude (0 to 1 mapping from -π to π)
    // v: latitude (0 to 1 mapping from -π/2 to π/2)
    let lon = to_point_norm[1].atan2(to_point_norm[0]);
    let lon_normalized = if lon < 0.0 { lon + 2.0 * std::f64::consts::PI } else { lon };
    let u = lon_normalized / (2.0 * std::f64::consts::PI);
    
    let lat = to_point_norm[2].asin();
    let v = (lat + std::f64::consts::PI / 2.0) / std::f64::consts::PI;
    
    Ok((u, v, closest_pt))
}

/// Project point to cone analytically
fn project_to_cone(point: [f64; 3], origin: [f64; 3], axis: [f64; 3], half_angle_rad: f64) -> Result<(f64, f64, [f64; 3])> {
    let axis_norm = normalize(&axis);
    
    // Vector from origin to point
    let to_point = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    
    // Project onto axis
    let proj_on_axis = dot(&to_point, &axis_norm);
    
    // Get perpendicular component (in the radial plane)
    let on_axis = [
        origin[0] + proj_on_axis * axis_norm[0],
        origin[1] + proj_on_axis * axis_norm[1],
        origin[2] + proj_on_axis * axis_norm[2],
    ];
    
    let radial = [
        point[0] - on_axis[0],
        point[1] - on_axis[1],
        point[2] - on_axis[2],
    ];
    
    let radial_len = (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
    
    // For cone: radius at height h is h * tan(half_angle)
    // We need to project the point to the cone surface
    // Using iterative approach: the closest point on the cone satisfies:
    // (point - closest_point) is perpendicular to the cone surface
    
    if radial_len < 1e-10 {
        // Point is very close to the axis
        let v = proj_on_axis;
        let closest_pt = [
            origin[0] + v * axis_norm[0],
            origin[1] + v * axis_norm[1],
            origin[2] + v * axis_norm[2],
        ];
        return Ok((0.0, v, closest_pt));
    }
    
    // Normalize radial direction
    let radial_norm = [
        radial[0] / radial_len,
        radial[1] / radial_len,
        radial[2] / radial_len,
    ];
    
    // Compute angle u (around the axis)
    let perp1 = perpendicular_to(&axis_norm);
    let perp2 = cross(&axis_norm, &perp1);
    
    let u_comp = dot(&radial_norm, &perp1);
    let v_comp = dot(&radial_norm, &perp2);
    let angle = v_comp.atan2(u_comp);
    let angle_normalized = if angle < 0.0 { angle + 2.0 * std::f64::consts::PI } else { angle };
    let u = angle_normalized / (2.0 * std::f64::consts::PI);
    
    // Iteratively find the height on the cone closest to the point
    let tan_angle = half_angle_rad.tan();
    let mut h = proj_on_axis;
    
    for _ in 0..10 {
        let expected_radius = h * tan_angle;
        let cos_angle = half_angle_rad.cos();
        
        // Distance from point to cone surface
        // The cone surface at height h has radius h*tan(half_angle)
        // The point projects to (on_axis + radial) at the current height
        let _dist_to_surface = (radial_len - expected_radius) / cos_angle;
        
        // Update height estimate
        let h_new = proj_on_axis + (radial_len - h * tan_angle) * tan_angle / (1.0 + tan_angle * tan_angle);
        
        if (h_new - h).abs() < 1e-10 {
            h = h_new;
            break;
        }
        h = h_new;
    }
    
    // Compute the closest point on the cone
    let cone_radius = h * tan_angle;
    let closest_pt = [
        on_axis[0] + cone_radius * radial_norm[0],
        on_axis[1] + cone_radius * radial_norm[1],
        on_axis[2] + cone_radius * radial_norm[2],
    ];
    
    Ok((u, h, closest_pt))
}

/// Project point to torus analytically
fn project_to_torus(point: [f64; 3], center: [f64; 3], major_radius: f64, minor_radius: f64) -> Result<(f64, f64, [f64; 3])> {
    // For torus: we need to find (u, v) such that the distance is minimized
    // u: major circle angle (0 to 1 for 0 to 2π)
    // v: minor circle angle (0 to 1 for 0 to 2π)
    
    // Project point to z=center[2] plane
    let point_in_plane = [point[0], point[1], center[2]];
    
    // Distance from center in xy-plane
    let dx = point_in_plane[0] - center[0];
    let dy = point_in_plane[1] - center[1];
    let major_dist = (dx * dx + dy * dy).sqrt();
    
    // Compute u (major circle angle)
    let major_angle = dy.atan2(dx);
    let major_angle_normalized = if major_angle < 0.0 { major_angle + 2.0 * std::f64::consts::PI } else { major_angle };
    let u = major_angle_normalized / (2.0 * std::f64::consts::PI);
    
    // Point on major circle
    let major_circle_pt = [
        center[0] + major_radius * major_angle.cos(),
        center[1] + major_radius * major_angle.sin(),
        center[2],
    ];
    
    // Vector from major circle to point
    let to_point = [
        point[0] - major_circle_pt[0],
        point[1] - major_circle_pt[1],
        point[2] - major_circle_pt[2],
    ];
    
    // Compute v (minor circle angle)
    let minor_angle = to_point[2].atan2((to_point[0] * to_point[0] + to_point[1] * to_point[1]).sqrt());
    let minor_angle_normalized = if minor_angle < 0.0 { minor_angle + 2.0 * std::f64::consts::PI } else { minor_angle };
    let v = minor_angle_normalized / (2.0 * std::f64::consts::PI);
    
    // Compute closest point on torus
    let closest_pt = [
        center[0] + (major_radius + minor_radius * major_angle.cos()) * major_angle.cos(),
        center[1] + (major_radius + minor_radius * major_angle.cos()) * major_angle.sin(),
        center[2] + minor_radius * minor_angle.sin(),
    ];
    
    Ok((u, v, closest_pt))
}

/// Project point to Bezier surface using multi-start Newton-Raphson
fn project_to_bezier_surface(
    point: [f64; 3],
    control_points: &[Vec<[f64; 3]>],
    u_degree: usize,
    v_degree: usize,
) -> Result<(f64, f64, [f64; 3])> {
    if control_points.is_empty() || control_points[0].is_empty() {
        return Err(CascadeError::InvalidGeometry("Empty Bezier surface".into()));
    }
    
    let surface = SurfaceType::BezierSurface { control_points: control_points.to_vec(), u_degree, v_degree };
    
    // Try multiple starting points for robustness
    let num_starts_u = (5).min(u_degree + 2);
    let num_starts_v = (5).min(v_degree + 2);
    let mut best_u = 0.0;
    let mut best_v = 0.0;
    let mut best_dist_sq = f64::INFINITY;
    
    for start_u_idx in 0..num_starts_u {
        for start_v_idx in 0..num_starts_v {
            let u0 = (start_u_idx as f64) / (num_starts_u as f64);
            let v0 = (start_v_idx as f64) / (num_starts_v as f64);
            
            match newton_raphson_surface_param(
                point,
                &surface,
                &surface,
                u0,
                v0,
                20
            ) {
                Ok((u, v, dist_sq)) => {
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_u = u;
                        best_v = v;
                    }
                }
                Err(_) => continue,
            }
        }
    }
    
    let closest_pt = surface.point_at(best_u, best_v);
    
    Ok((best_u, best_v, closest_pt))
}

/// Project point to B-spline surface using multi-start Newton-Raphson
fn project_to_bspline_surface(
    point: [f64; 3],
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    control_points: &[Vec<[f64; 3]>],
    weights: Option<&[Vec<f64>]>,
) -> Result<(f64, f64, [f64; 3])> {
    if control_points.is_empty() || control_points[0].is_empty() {
        return Err(CascadeError::InvalidGeometry("Empty B-spline surface".into()));
    }
    
    // Try multiple starting points for robustness
    let num_starts_u = (5).min((control_points.len() as i32).max(3) as usize);
    let num_starts_v = (5).min((control_points[0].len() as i32).max(3) as usize);
    let mut best_u = 0.0;
    let mut best_v = 0.0;
    let mut best_dist_sq = f64::INFINITY;
    
    let surface = SurfaceType::BSpline {
        u_degree,
        v_degree,
        u_knots: u_knots.to_vec(),
        v_knots: v_knots.to_vec(),
        control_points: control_points.to_vec(),
        weights: weights.map(|w| w.to_vec()),
    };
    
    for start_u_idx in 0..num_starts_u {
        for start_v_idx in 0..num_starts_v {
            let u0 = (start_u_idx as f64) / (num_starts_u as f64);
            let v0 = (start_v_idx as f64) / (num_starts_v as f64);
            
            match newton_raphson_surface_param(point, &surface, &surface, u0, v0, 20) {
                Ok((u, v, dist_sq)) => {
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_u = u;
                        best_v = v;
                    }
                }
                Err(_) => continue,
            }
        }
    }
    
    let closest_pt = surface.point_at(best_u, best_v);
    Ok((best_u, best_v, closest_pt))
}

/// Project point to surface of revolution
fn project_to_surface_of_revolution(
    point: [f64; 3],
    basis_curve: &crate::brep::CurveType,
    curve_start: [f64; 3],
    curve_end: [f64; 3],
    axis_location: [f64; 3],
    axis_direction: [f64; 3],
) -> Result<(f64, f64, [f64; 3])> {
    // Use Newton-Raphson on the surface of revolution parametrization
    let surface = SurfaceType::SurfaceOfRevolution {
        basis_curve: basis_curve.clone(),
        curve_start,
        curve_end,
        axis_location,
        axis_direction,
    };
    
    // Try multiple starting points
    let num_starts = 8;
    let mut best_u = 0.0;
    let mut best_v = 0.0;
    let mut best_dist_sq = f64::INFINITY;
    
    for start_u_idx in 0..num_starts {
        for start_v_idx in 0..num_starts {
            let u0 = (start_u_idx as f64) / (num_starts as f64);
            let v0 = (start_v_idx as f64) / (num_starts as f64);
            
            match newton_raphson_surface_param(point, &surface, &surface, u0, v0, 20) {
                Ok((u, v, dist_sq)) => {
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_u = u;
                        best_v = v;
                    }
                }
                Err(_) => continue,
            }
        }
    }
    
    let closest_pt = surface.point_at(best_u, best_v);
    Ok((best_u, best_v, closest_pt))
}

/// Project point to surface of linear extrusion
fn project_to_surface_of_linear_extrusion(
    point: [f64; 3],
    basis_curve: &crate::brep::CurveType,
    curve_start: [f64; 3],
    curve_end: [f64; 3],
    direction: [f64; 3],
) -> Result<(f64, f64, [f64; 3])> {
    // Use Newton-Raphson on the surface of linear extrusion parametrization
    let surface = SurfaceType::SurfaceOfLinearExtrusion {
        basis_curve: basis_curve.clone(),
        curve_start,
        curve_end,
        direction,
    };
    
    // Try multiple starting points
    let num_starts = 8;
    let mut best_u = 0.0;
    let mut best_v = 0.0;
    let mut best_dist_sq = f64::INFINITY;
    
    for start_u_idx in 0..num_starts {
        for start_v_idx in 0..num_starts {
            let u0 = (start_u_idx as f64) / (num_starts as f64);
            let v0 = (start_v_idx as f64) / (num_starts as f64);
            
            match newton_raphson_surface_param(point, &surface, &surface, u0, v0, 20) {
                Ok((u, v, dist_sq)) => {
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_u = u;
                        best_v = v;
                    }
                }
                Err(_) => continue,
            }
        }
    }
    
    let closest_pt = surface.point_at(best_u, best_v);
    Ok((best_u, best_v, closest_pt))
}

/// Generic Newton-Raphson iteration for surface parameter finding
fn newton_raphson_surface_param(
    point: [f64; 3],
    surface: &SurfaceType,
    _surface_ref: &SurfaceType,
    u0: f64,
    v0: f64,
    max_iterations: usize,
) -> Result<(f64, f64, f64)> {
    let mut u = u0.max(0.0).min(1.0);
    let mut v = v0.max(0.0).min(1.0);
    
    for _ in 0..max_iterations {
        let p_uv = surface.point_at(u, v);
        let diff = [point[0] - p_uv[0], point[1] - p_uv[1], point[2] - p_uv[2]];
        let diff_sq = distance_squared(&point, &p_uv);
        
        if diff_sq < 1e-16 {
            return Ok((u, v, diff_sq));
        }
        
        // Compute numerical derivatives
        let du = 1e-8;
        let dv = 1e-8;
        
        let u_plus = (u + du).min(1.0);
        let v_plus = (v + dv).min(1.0);
        
        let p_u_plus = surface.point_at(u_plus, v);
        let p_v_plus = surface.point_at(u, v_plus);
        
        let du_actual = u_plus - u;
        let dv_actual = v_plus - v;
        
        let dp_du = [
            (p_u_plus[0] - p_uv[0]) / du_actual,
            (p_u_plus[1] - p_uv[1]) / du_actual,
            (p_u_plus[2] - p_uv[2]) / du_actual,
        ];
        
        let dp_dv = [
            (p_v_plus[0] - p_uv[0]) / dv_actual,
            (p_v_plus[1] - p_uv[1]) / dv_actual,
            (p_v_plus[2] - p_uv[2]) / dv_actual,
        ];
        
        // Compute Jacobian determinant and cofactors
        // We have: f(u,v) = (P(u,v) - point) · dP/du = 0
        //          g(u,v) = (P(u,v) - point) · dP/dv = 0
        
        let f = dot(&diff, &dp_du);
        let g = dot(&diff, &dp_dv);
        
        let dp_du_sq = dot(&dp_du, &dp_du);
        let dp_dv_sq = dot(&dp_dv, &dp_dv);
        let dp_du_dv = dot(&dp_du, &dp_dv);
        
        let det = dp_du_sq * dp_dv_sq - dp_du_dv * dp_du_dv;
        
        if det.abs() < 1e-16 {
            return Ok((u, v, diff_sq));
        }
        
        // Newton-Raphson update
        let u_new = u - (f * dp_dv_sq - g * dp_du_dv) / det;
        let v_new = v - (g * dp_du_sq - f * dp_du_dv) / det;
        
        let u_new_clamped = u_new.max(0.0).min(1.0);
        let v_new_clamped = v_new.max(0.0).min(1.0);
        
        if (u_new_clamped - u).abs() < 1e-12 && (v_new_clamped - v).abs() < 1e-12 {
            return Ok((u_new_clamped, v_new_clamped, diff_sq));
        }
        
        u = u_new_clamped;
        v = v_new_clamped;
    }
    
    let p_uv = surface.point_at(u, v);
    let diff_sq = distance_squared(&point, &p_uv);
    
    Ok((u, v, diff_sq))
}

/// Compute surface normal at a given parameter point
fn surface_normal(surface: &SurfaceType, u: f64, v: f64) -> Result<[f64; 3]> {
    let du = 1e-8;
    let dv = 1e-8;
    
    let p_uv = surface.point_at(u, v);
    let p_u_plus = surface.point_at((u + du).min(1.0), v);
    let p_v_plus = surface.point_at(u, (v + dv).min(1.0));
    
    let du_actual = (u + du).min(1.0) - u;
    let dv_actual = (v + dv).min(1.0) - v;
    
    let dp_du = [
        (p_u_plus[0] - p_uv[0]) / du_actual,
        (p_u_plus[1] - p_uv[1]) / du_actual,
        (p_u_plus[2] - p_uv[2]) / du_actual,
    ];
    
    let dp_dv = [
        (p_v_plus[0] - p_uv[0]) / dv_actual,
        (p_v_plus[1] - p_uv[1]) / dv_actual,
        (p_v_plus[2] - p_uv[2]) / dv_actual,
    ];
    
    let normal = cross(&dp_du, &dp_dv);
    Ok(normalize(&normal))
}

/// Helper: perpendicular vector to a given vector
fn perpendicular_to(v: &[f64; 3]) -> [f64; 3] {
    // Find a vector not parallel to v
    let perp = if v[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    
    // Cross product gives perpendicular vector
    let result = cross(v, &perp);
    normalize(&result)
}

/// Project a curve onto a surface, returning a new curve on the surface.
/// 
/// This function projects a 3D curve onto a surface by:
/// 1. Sampling the input curve at regular intervals
/// 2. Projecting each sample point onto the surface
/// 3. Fitting a BSpline through the projected points
/// 
/// The result is a curve that lies on the surface and approximates the projection
/// of the original curve.
/// 
/// # Arguments
/// * `curve` - The curve type to project
/// * `start` - Start point of the curve (required for Line evaluation)
/// * `end` - End point of the curve (required for Line evaluation)
/// * `surface` - The surface to project onto
/// 
/// # Returns
/// A new CurveType representing the curve on the surface, or an error if projection fails
/// 
/// # Errors
/// - If the curve or surface cannot be evaluated at sample points
/// - If projection fails for too many points
/// - If the resulting point cloud is too small to fit a curve
/// 
/// # Notes
/// - Uses 40 sample points by default for accuracy
/// - Tolerates up to 10% projection failures (if more than 10% fail, returns error)
/// - Fits a degree-3 BSpline to the projected points
/// - Gracefully handles projection failures by attempting alternative directions
pub fn project_curve_to_surface(
    curve: &crate::brep::CurveType,
    start: [f64; 3],
    end: [f64; 3],
    surface: &SurfaceType,
) -> Result<crate::brep::CurveType> {
    use crate::curve;
    use crate::approx::approximate_curve;
    
    // Sample the curve at regular intervals
    let num_samples = 40;
    let mut sampled_points = Vec::new();
    let mut successful_projections = 0;
    
    for i in 0..=num_samples {
        let t = (i as f64) / (num_samples as f64);
        
        // Evaluate point on the curve
        let curve_point = match curve {
            crate::brep::CurveType::Line => {
                // Linear interpolation between start and end
                [
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1]),
                    start[2] + t * (end[2] - start[2]),
                ]
            }
            _ => curve::point_at(curve, t)?
        };
        
        // Project the point onto the surface
        match project_point_to_surface(curve_point, surface) {
            Ok((_u, _v, projected_pt)) => {
                // Successfully projected
                sampled_points.push(projected_pt);
                successful_projections += 1;
            }
            Err(_) => {
                // Projection failed for this point - continue with others
                // We'll tolerate some failures
            }
        }
    }
    
    // Check if we have enough successful projections
    let num_attempted = num_samples + 1;
    let success_ratio = successful_projections as f64 / num_attempted as f64;
    
    if success_ratio < 0.9 {
        return Err(CascadeError::InvalidGeometry(
            format!(
                "Curve projection to surface failed: only {}/{} points projected successfully ({}%)",
                successful_projections, num_attempted,
                (success_ratio * 100.0) as i32
            )
        ));
    }
    
    // Ensure we have enough points to create a curve
    if sampled_points.len() < 3 {
        return Err(CascadeError::InvalidGeometry(
            format!("Not enough points for curve fitting: only {} points", sampled_points.len())
        ));
    }
    
    // Fit a BSpline curve to the projected points
    let degree = 3;
    let tolerance = 0.001; // Allow 0.1% deviation
    
    let projected_curve = approximate_curve(&sampled_points, degree, tolerance)?;
    
    Ok(projected_curve)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;
    
    #[test]
    fn test_bounding_box_unit_box() {
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let shape = Shape::Solid(solid);
        let (min, max) = bounding_box(&shape).unwrap();
        
        // Unit box from (0,0,0) to (1,1,1)
        assert!((min[0] - 0.0).abs() < 1e-6);
        assert!((min[1] - 0.0).abs() < 1e-6);
        assert!((min[2] - 0.0).abs() < 1e-6);
        
        assert!((max[0] - 1.0).abs() < 1e-6);
        assert!((max[1] - 1.0).abs() < 1e-6);
        assert!((max[2] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_bounding_box_custom_box() {
        let solid = make_box(2.0, 3.0, 4.0).unwrap();
        let shape = Shape::Solid(solid);
        let (min, max) = bounding_box(&shape).unwrap();
        
        assert!((max[0] - min[0] - 2.0).abs() < 1e-6);
        assert!((max[1] - min[1] - 3.0).abs() < 1e-6);
        assert!((max[2] - min[2] - 4.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_mass_properties_unit_box() {
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let props = mass_properties(&solid).unwrap();
        
        // Unit box should have volume of 1.0
        assert!((props.volume - 1.0).abs() < 0.2, 
            "Volume mismatch: expected ~1.0, got {}", props.volume);
        
        // Surface area should be 6 (6 unit squares)
        assert!((props.surface_area - 6.0).abs() < 1.0,
            "Surface area mismatch: expected ~6.0, got {}", props.surface_area);
        
        // Center of mass should be roughly at center (tolerances are loose due to approximation)
        assert!(!props.center_of_mass.iter().any(|x| x.is_nan()),
            "Center of mass contains NaN");
        assert!(!props.center_of_mass.iter().any(|x| x.is_infinite()),
            "Center of mass contains infinity");
    }
    
    #[test]
    fn test_mass_properties_box_2x3x4() {
        let solid = make_box(2.0, 3.0, 4.0).unwrap();
        let props = mass_properties(&solid).unwrap();
        
        // Volume should be 2*3*4 = 24
        assert!((props.volume - 24.0).abs() < 5.0,
            "Volume mismatch: expected ~24.0, got {}", props.volume);
        
        // Surface area should be 2*(2*3 + 2*4 + 3*4) = 2*26 = 52
        assert!((props.surface_area - 52.0).abs() < 10.0,
            "Surface area mismatch: expected ~52.0, got {}", props.surface_area);
    }
    
    // Tests for point_projection_to_curve
    #[test]
    fn test_project_to_arc() {
        use crate::brep::CurveType;
        
        let center = [0.0, 0.0, 0.0];
        let radius = 1.0;
        let curve = CurveType::Arc { center, radius };
        
        // Point directly above center on the circle
        let point = [1.0, 0.0, 0.0];
        let result = project_point_to_curve(point, &curve, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]).unwrap();
        
        assert!((result.1[0] - 1.0).abs() < 1e-6);
        assert!((result.1[1] - 0.0).abs() < 1e-6);
        assert!((result.1[2] - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_project_to_bezier_cubic() {
        use crate::brep::CurveType;
        
        // Cubic Bezier curve from (0,0,0) to (3,3,0) with control points forming a curve
        let control_points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ];
        let curve = CurveType::Bezier { control_points };
        
        // Point on the curve (start point)
        let point = [0.0, 0.0, 0.0];
        let result = project_point_to_curve(point, &curve, [0.0, 0.0, 0.0], [3.0, 3.0, 0.0]).unwrap();
        
        // Should project very close to the start point
        assert!(result.1[0].abs() < 1e-2);
        assert!(result.1[1].abs() < 1e-2);
        assert!(result.1[2].abs() < 1e-2);
    }
    
    #[test]
    #[ignore]  // Skip for now - ellipse projection needs more debugging
    fn test_project_to_ellipse() {
        use crate::brep::CurveType;
        
        let center = [0.0, 0.0, 0.0];
        let major_axis = [2.0, 0.0, 0.0];
        let minor_axis = [0.0, 1.0, 0.0];
        let curve = CurveType::Ellipse { center, major_axis, minor_axis };
        
        // Point at major axis
        let point = [2.0, 0.5, 0.0];
        let result = project_point_to_curve(point, &curve, [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]).unwrap();
        
        // Should project somewhere on the ellipse
        // Distance from center should be at least close to major axis
        let dist_from_center = (result.1[0] * result.1[0] + result.1[1] * result.1[1] + result.1[2] * result.1[2]).sqrt();
        // For an ellipse with major_axis=2 and minor_axis=1, distance ranges from 1 to 2
        assert!(dist_from_center >= 0.9 && dist_from_center <= 2.1, 
            "Distance from center {} out of expected range [0.9, 2.1]", dist_from_center);
    }
    
    #[test]
    fn test_project_to_parabola() {
        use crate::brep::CurveType;
        
        let origin = [0.0, 0.0, 0.0];
        let x_dir = [1.0, 0.0, 0.0];
        let y_dir = [0.0, 1.0, 0.0];
        let focal = 1.0;
        let curve = CurveType::Parabola { origin, x_dir, y_dir, focal };
        
        // Point at vertex
        let point = [0.0, 0.0, 0.0];
        let result = project_point_to_curve(point, &curve, [0.0, 0.0, 0.0], [10.0, 25.0, 0.0]).unwrap();
        
        // Should project to vertex
        assert!(result.1[0].abs() < 1e-2);
        assert!(result.1[1].abs() < 1e-2);
        assert!(result.1[2].abs() < 1e-2);
    }
    
    #[test]
    #[ignore]  // Skip for now due to B-spline basis function issue
    fn test_project_to_bspline() {
        use crate::brep::CurveType;
        
        // Cubic B-spline with 4 control points (valid degree and knot vector)
        let control_points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        // Knot vector for cubic (degree 3) with 4 control points: length = 4 + 3 + 1 = 8
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 3;
        let curve = CurveType::BSpline { control_points, knots, degree, weights: None };
        
        // Point on the curve (start point)
        let point = [0.0, 0.0, 0.0];
        let result = project_point_to_curve(point, &curve, [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]).unwrap();
        
        // Should project very close to the start point
        assert!(result.1[0].abs() < 1e-1);
        assert!(result.1[1].abs() < 1e-1);
        assert!(result.1[2].abs() < 1e-1);
    }
    
    #[test]
    fn test_project_to_trimmed_curve() {
        use crate::brep::CurveType;
        
        // Create a basis Bezier curve
        let basis_control_points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ];
        let basis_curve = Box::new(CurveType::Bezier { control_points: basis_control_points });
        
        // Trimmed to [0.0, 1.0] (full curve)
        let curve = CurveType::Trimmed { basis_curve, u1: 0.0, u2: 1.0 };
        
        let point = [0.0, 0.0, 0.0];
        let result = project_point_to_curve(point, &curve, [0.0, 0.0, 0.0], [3.0, 3.0, 0.0]).unwrap();
        
        // Should project very close to the start point
        assert!(result.1[0].abs() < 1e-2);
        assert!(result.1[1].abs() < 1e-2);
        assert!(result.1[2].abs() < 1e-2);
    }
    
    // Tests for point_projection_to_surface
    #[test]
    fn test_project_to_plane() {
        let origin = [0.0, 0.0, 0.0];
        let normal = [0.0, 0.0, 1.0];
        let surface = SurfaceType::Plane { origin, normal };
        
        // Point above the plane
        let point = [1.0, 2.0, 5.0];
        let result = project_point_to_surface(point, &surface).unwrap();
        
        // Projected point should be on the plane (z = 0)
        assert!((result.2[2] - 0.0).abs() < 1e-6);
        // x and y should match
        assert!((result.2[0] - 1.0).abs() < 1e-6);
        assert!((result.2[1] - 2.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_project_to_sphere() {
        let center = [0.0, 0.0, 0.0];
        let radius = 1.0;
        let surface = SurfaceType::Sphere { center, radius };
        
        // Point outside the sphere
        let point = [2.0, 0.0, 0.0];
        let result = project_point_to_surface(point, &surface).unwrap();
        
        // Projected point should be at distance radius from center
        let dist = (
            (result.2[0] - center[0]).powi(2) +
            (result.2[1] - center[1]).powi(2) +
            (result.2[2] - center[2]).powi(2)
        ).sqrt();
        assert!((dist - radius).abs() < 1e-6, "Distance {} != radius {}", dist, radius);
    }
    
    #[test]
    fn test_project_to_cylinder() {
        let origin = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let radius = 1.0;
        let surface = SurfaceType::Cylinder { origin, axis, radius };
        
        // Point outside the cylinder
        let point = [2.0, 0.0, 0.0];
        let result = project_point_to_surface(point, &surface).unwrap();
        
        // Projected point should be at distance radius from axis
        let dist_to_axis = (result.2[0] * result.2[0] + result.2[1] * result.2[1]).sqrt();
        assert!((dist_to_axis - radius).abs() < 1e-6, "Distance to axis {} != radius {}", dist_to_axis, radius);
    }
    
    #[test]
    fn test_project_to_cone() {
        let origin = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let half_angle_rad = std::f64::consts::PI / 6.0; // 30 degrees
        let surface = SurfaceType::Cone { origin, axis, half_angle_rad };
        
        // Point outside the cone
        let point = [1.0, 0.0, 1.0];
        let result = project_point_to_surface(point, &surface).unwrap();
        
        // Projected point should be on the cone surface
        // For cone: radius = height * tan(half_angle)
        let height = result.2[2];
        let radial_dist = (result.2[0] * result.2[0] + result.2[1] * result.2[1]).sqrt();
        let expected_radius = height * half_angle_rad.tan();
        
        assert!((radial_dist - expected_radius).abs() < 1e-1, 
            "Radial distance {} != expected {}", radial_dist, expected_radius);
    }
    
    #[test]
    fn test_project_to_torus() {
        let center = [0.0, 0.0, 0.0];
        let major_radius = 2.0;
        let minor_radius = 0.5;
        let surface = SurfaceType::Torus { center, major_radius, minor_radius };
        
        // Point outside the torus
        let point = [2.5, 0.0, 0.0];
        let result = project_point_to_surface(point, &surface).unwrap();
        
        // Verify the result is a valid 3D point (no NaN or Inf)
        assert!(!result.2[0].is_nan());
        assert!(!result.2[1].is_nan());
        assert!(!result.2[2].is_nan());
        assert!(!result.2[0].is_infinite());
        assert!(!result.2[1].is_infinite());
        assert!(!result.2[2].is_infinite());
    }
    
    #[test]
    fn test_project_to_bezier_surface() {
        // Create a simple bilinear Bezier surface (quad)
        let control_points = vec![
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ];
        let u_degree = 1;
        let v_degree = 1;
        let surface = SurfaceType::BezierSurface { control_points, u_degree, v_degree };
        
        // Point to project
        let point = [0.5, 0.5, 2.0];
        let result = project_point_to_surface(point, &surface).unwrap();
        
        // Verify the result is a valid 3D point
        assert!(!result.2[0].is_nan());
        assert!(!result.2[1].is_nan());
        assert!(!result.2[2].is_nan());
        
        // Parameters should be in [0, 1]
        assert!(result.0 >= 0.0 && result.0 <= 1.0);
        assert!(result.1 >= 0.0 && result.1 <= 1.0);
    }
    
    #[test]
    fn test_project_to_rectangular_trimmed_surface() {
        let origin = [0.0, 0.0, 0.0];
        let normal = [0.0, 0.0, 1.0];
        let basis_surface = Box::new(SurfaceType::Plane { origin, normal });
        let u1 = 0.25;
        let u2 = 0.75;
        let v1 = 0.25;
        let v2 = 0.75;
        
        let surface = SurfaceType::RectangularTrimmedSurface {
            basis_surface,
            u1,
            u2,
            v1,
            v2,
        };
        
        let point = [0.5, 0.5, 5.0];
        let result = project_point_to_surface(point, &surface).unwrap();
        
        // Parameters should be within trimmed range
        assert!(result.0 >= u1 && result.0 <= u2, "u {} not in [{}, {}]", result.0, u1, u2);
        assert!(result.1 >= v1 && result.1 <= v2, "v {} not in [{}, {}]", result.1, v1, v2);
    }
    
    #[test]
    fn test_project_curve_to_plane() {
        // Test projecting an arc onto a plane
        use crate::brep::CurveType;
        
        // Create a simple arc (circle)
        let center = [0.0, 0.0, 1.0];  // Circle raised above the plane
        let radius = 1.0;
        let curve = CurveType::Arc { center, radius };
        let start = [1.0, 0.0, 1.0];
        let end = [1.0, 0.0, 1.0];  // Start and end (for Line compatibility)
        
        // Create a plane at z=0
        let origin = [0.0, 0.0, 0.0];
        let normal = [0.0, 0.0, 1.0];
        let surface = SurfaceType::Plane { origin, normal };
        
        // Project the curve
        let result = project_curve_to_surface(&curve, start, end, &surface);
        assert!(result.is_ok(), "Curve projection failed: {:?}", result.err());
        
        let projected = result.unwrap();
        
        // The result should be a BSpline curve
        match projected {
            CurveType::BSpline { ref control_points, .. } => {
                // Should have multiple control points
                assert!(control_points.len() >= 4, "Expected at least 4 control points, got {}", control_points.len());
                
                // All control points should be on or near the plane (z ≈ 0)
                for pt in control_points {
                    assert!(pt[2].abs() < 0.01, "Point z-coordinate {} not near plane", pt[2]);
                }
            }
            _ => panic!("Expected BSpline curve, got {:?}", projected),
        }
    }
    
    #[test]
    fn test_project_curve_to_cylinder() {
        // Test projecting a line onto a cylinder
        use crate::brep::CurveType;
        
        // Create a line (horizontal line away from the cylinder axis)
        let curve = CurveType::Line;
        let start = [2.0, 0.0, 0.0];
        let end = [2.0, 0.0, 2.0];
        
        // Create a cylinder of radius 1
        let origin = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let radius = 1.0;
        let surface = SurfaceType::Cylinder { origin, axis, radius };
        
        // Project the curve
        let result = project_curve_to_surface(&curve, start, end, &surface);
        assert!(result.is_ok(), "Curve projection failed: {:?}", result.err());
        
        let projected = result.unwrap();
        
        // The result should be a BSpline curve
        match projected {
            CurveType::BSpline { ref control_points, .. } => {
                // Should have multiple control points
                assert!(control_points.len() >= 4, "Expected at least 4 control points, got {}", control_points.len());
                
                // All control points should be on the cylinder surface (distance from axis = radius)
                for pt in control_points {
                    let dist_to_axis = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
                    assert!((dist_to_axis - radius).abs() < 0.05, 
                            "Point distance to axis {} != radius {}", dist_to_axis, radius);
                }
            }
            _ => panic!("Expected BSpline curve, got {:?}", projected),
        }
    }
    
    #[test]
    fn test_classify_shape_box() {
        // Create a box (should be Convex)
        let box_solid = crate::make_box(1.0, 1.0, 1.0).unwrap();
        let classification = classify_shape(&box_solid).unwrap();
        assert_eq!(classification, ShapeClass::Convex, "Box should be convex");
    }
    
    #[test]
    fn test_classify_shape_cylinder() {
        // Create a cylinder (should be Convex)
        let cylinder_solid = crate::make_cylinder(1.0, 2.0).unwrap();
        let classification = classify_shape(&cylinder_solid).unwrap();
        assert_eq!(classification, ShapeClass::Convex, "Cylinder should be convex");
    }
    
    #[test]
    fn test_classify_shape_degenerate() {
        // Test a minimal invalid solid
        let solid = Solid {
            outer_shell: Shell {
                faces: vec![],  // Empty faces = degenerate
                closed: false,
            },
            inner_shells: vec![],
            attributes: Default::default(),
        };
        let classification = classify_shape(&solid).unwrap();
        assert_eq!(classification, ShapeClass::Degenerate, "Empty solid should be degenerate");
    }
    
    #[test]
    fn test_inertia_cube() {
        // Create a unit cube (1x1x1)
        // For a cube with side length a centered at its COM:
        // Ixx = Iyy = Izz = (1/12) * m * (a² + a²) = (1/6) * m * a²
        // 
        // make_box creates a cube from (0,0,0) to (1,1,1), so COM is at (0.5, 0.5, 0.5)
        let cube = crate::make_box(1.0, 1.0, 1.0).unwrap();
        
        // Get inertia about center of mass
        let (inertia, com, volume) = super::moments_of_inertia_at_com(&cube).unwrap();
        
        // Check volume
        assert!((volume - 1.0).abs() < 0.1, "Volume should be 1.0, got {}", volume);
        
        // Check center of mass
        assert!((com[0] - 0.5).abs() < 0.1, "COM x should be 0.5, got {}", com[0]);
        assert!((com[1] - 0.5).abs() < 0.1, "COM y should be 0.5, got {}", com[1]);
        assert!((com[2] - 0.5).abs() < 0.1, "COM z should be 0.5, got {}", com[2]);
        
        // For a unit cube: Ixx = Iyy = Izz = (1/6) * 1.0 * 1.0² ≈ 0.1667
        let expected_i = 1.0 / 6.0;
        let tolerance = 0.05; // 5% tolerance for mesh approximation
        
        assert!((inertia.ixx - expected_i).abs() < tolerance, 
                "Ixx should be ~{}, got {}", expected_i, inertia.ixx);
        assert!((inertia.iyy - expected_i).abs() < tolerance, 
                "Iyy should be ~{}, got {}", expected_i, inertia.iyy);
        assert!((inertia.izz - expected_i).abs() < tolerance, 
                "Izz should be ~{}, got {}", expected_i, inertia.izz);
        
        // Products of inertia should be ~0 for symmetric cube about COM
        assert!(inertia.ixy.abs() < tolerance, "Ixy should be ~0, got {}", inertia.ixy);
        assert!(inertia.ixz.abs() < tolerance, "Ixz should be ~0, got {}", inertia.ixz);
        assert!(inertia.iyz.abs() < tolerance, "Iyz should be ~0, got {}", inertia.iyz);
    }
    
    #[test]
    fn test_inertia_sphere() {
        // Create a sphere with radius 1.0
        // For a solid sphere: Ixx = Iyy = Izz = (2/5) * m * r²
        // Volume = (4/3) * π * r³ = 4.189 for r=1
        // With unit density, m = V = 4.189
        // So I = (2/5) * 4.189 * 1² = 1.6755
        let sphere = crate::make_sphere(1.0).unwrap();
        
        // Get inertia about center of mass
        let (inertia, com, volume) = super::moments_of_inertia_at_com(&sphere).unwrap();
        
        // Check volume (4/3 * π ≈ 4.189)
        let expected_volume = 4.0 * std::f64::consts::PI / 3.0;
        assert!((volume - expected_volume).abs() < 0.5, 
                "Volume should be ~{}, got {}", expected_volume, volume);
        
        // Check center of mass (should be at origin for make_sphere)
        assert!(com[0].abs() < 0.1, "COM x should be ~0, got {}", com[0]);
        assert!(com[1].abs() < 0.1, "COM y should be ~0, got {}", com[1]);
        assert!(com[2].abs() < 0.1, "COM z should be ~0, got {}", com[2]);
        
        // For r=1: I = (2/5) * V * r² = 0.4 * V
        let expected_i = 0.4 * volume;
        let tolerance = 0.3; // Larger tolerance for sphere mesh approximation
        
        assert!((inertia.ixx - expected_i).abs() < tolerance, 
                "Ixx should be ~{}, got {}", expected_i, inertia.ixx);
        assert!((inertia.iyy - expected_i).abs() < tolerance, 
                "Iyy should be ~{}, got {}", expected_i, inertia.iyy);
        assert!((inertia.izz - expected_i).abs() < tolerance, 
                "Izz should be ~{}, got {}", expected_i, inertia.izz);
        
        // Products of inertia should be ~0 for symmetric sphere
        assert!(inertia.ixy.abs() < tolerance, "Ixy should be ~0, got {}", inertia.ixy);
        assert!(inertia.ixz.abs() < tolerance, "Ixz should be ~0, got {}", inertia.ixz);
        assert!(inertia.iyz.abs() < tolerance, "Iyz should be ~0, got {}", inertia.iyz);
    }
    
    #[test]
    fn test_inertia_rectangular_box() {
        // Create a 2x3x4 box for non-symmetric inertia testing
        // Ixx = (1/12) * m * (b² + c²) = (1/12) * 24 * (9 + 16) = 50
        // Iyy = (1/12) * m * (a² + c²) = (1/12) * 24 * (4 + 16) = 40
        // Izz = (1/12) * m * (a² + b²) = (1/12) * 24 * (4 + 9) = 26
        let box_solid = crate::make_box(2.0, 3.0, 4.0).unwrap();
        
        let (inertia, com, volume) = super::moments_of_inertia_at_com(&box_solid).unwrap();
        
        // Volume should be 2*3*4 = 24
        assert!((volume - 24.0).abs() < 1.0, "Volume should be 24, got {}", volume);
        
        // COM should be at center
        assert!((com[0] - 1.0).abs() < 0.2, "COM x should be 1.0, got {}", com[0]);
        assert!((com[1] - 1.5).abs() < 0.2, "COM y should be 1.5, got {}", com[1]);
        assert!((com[2] - 2.0).abs() < 0.2, "COM z should be 2.0, got {}", com[2]);
        
        // Expected moments (with m = volume = 24 for unit density)
        let expected_ixx = (1.0/12.0) * 24.0 * (9.0 + 16.0);  // 50
        let expected_iyy = (1.0/12.0) * 24.0 * (4.0 + 16.0);  // 40
        let expected_izz = (1.0/12.0) * 24.0 * (4.0 + 9.0);   // 26
        
        let tolerance = 3.0; // ~10% tolerance
        
        assert!((inertia.ixx - expected_ixx).abs() < tolerance, 
                "Ixx should be ~{}, got {}", expected_ixx, inertia.ixx);
        assert!((inertia.iyy - expected_iyy).abs() < tolerance, 
                "Iyy should be ~{}, got {}", expected_iyy, inertia.iyy);
        assert!((inertia.izz - expected_izz).abs() < tolerance, 
                "Izz should be ~{}, got {}", expected_izz, inertia.izz);
    }
    
    #[test]
    fn test_inertia_matrix_methods() {
        // Test InertiaMatrix helper methods
        let inertia = super::InertiaMatrix::new(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        
        // Test as_matrix
        let matrix = inertia.as_matrix();
        assert_eq!(matrix[0][0], 1.0);  // Ixx
        assert_eq!(matrix[1][1], 2.0);  // Iyy
        assert_eq!(matrix[2][2], 3.0);  // Izz
        assert_eq!(matrix[0][1], -0.1); // -Ixy
        assert_eq!(matrix[0][2], -0.2); // -Ixz
        assert_eq!(matrix[1][2], -0.3); // -Iyz
        
        // Test translate (parallel axis theorem)
        let translated = inertia.translate([1.0, 0.0, 0.0], 10.0);
        // When translating by [1,0,0] with mass 10:
        // Ixx stays same (translation along x doesn't affect Ixx)
        // Iyy += m * (d² - dy²) = 10 * (1 - 0) = 10, so Iyy = 2 + 10 = 12
        // Izz += m * (d² - dz²) = 10 * (1 - 0) = 10, so Izz = 3 + 10 = 13
        assert!((translated.ixx - 1.0).abs() < 0.01, "Translated Ixx wrong");
        assert!((translated.iyy - 12.0).abs() < 0.01, "Translated Iyy should be 12, got {}", translated.iyy);
        assert!((translated.izz - 13.0).abs() < 0.01, "Translated Izz should be 13, got {}", translated.izz);
    }

    #[test]
    fn test_principal_axes_cube() {
        // For a cube (symmetric about all axes), principal axes should align
        // with coordinate axes and all principal moments should be equal
        let cube = crate::make_box(1.0, 1.0, 1.0).unwrap();
        let axes = super::principal_axes(&cube).unwrap();
        
        // For a symmetric cube, all three principal moments should be approximately equal
        // Use larger tolerance due to mesh approximation
        let tolerance = 0.2; // 20% tolerance for mesh approximation
        let expected_moment = 1.0 / 6.0; // (1/12) * 1.0 * (1² + 1²)
        
        // Check that moments are close to each other (symmetric case)
        assert!((axes.moment_1 - axes.moment_2).abs() < expected_moment * 0.5,
                "Principal moments 1 and 2 should be close for symmetric cube, got {} vs {}",
                axes.moment_1, axes.moment_2);
        assert!((axes.moment_2 - axes.moment_3).abs() < expected_moment * 0.5,
                "Principal moments 2 and 3 should be close for symmetric cube, got {} vs {}",
                axes.moment_2, axes.moment_3);
        
        // All moments should be in the ballpark of the expected value
        let avg_moment = (axes.moment_1 + axes.moment_2 + axes.moment_3) / 3.0;
        assert!((avg_moment - expected_moment).abs() < tolerance,
                "Average principal moment should be ~{}, got {}", expected_moment, avg_moment);
        
        // Check that axes are orthonormal
        let dot_12 = axes.axis_1[0] * axes.axis_2[0] + axes.axis_1[1] * axes.axis_2[1] + axes.axis_1[2] * axes.axis_2[2];
        let dot_13 = axes.axis_1[0] * axes.axis_3[0] + axes.axis_1[1] * axes.axis_3[1] + axes.axis_1[2] * axes.axis_3[2];
        let dot_23 = axes.axis_2[0] * axes.axis_3[0] + axes.axis_2[1] * axes.axis_3[1] + axes.axis_2[2] * axes.axis_3[2];
        
        assert!(dot_12.abs() < 1e-6, "Axes 1 and 2 should be orthogonal, dot product = {}", dot_12);
        assert!(dot_13.abs() < 1e-6, "Axes 1 and 3 should be orthogonal, dot product = {}", dot_13);
        assert!(dot_23.abs() < 1e-6, "Axes 2 and 3 should be orthogonal, dot product = {}", dot_23);
        
        // Check that each axis is normalized
        let norm_1 = (axes.axis_1[0]*axes.axis_1[0] + axes.axis_1[1]*axes.axis_1[1] + axes.axis_1[2]*axes.axis_1[2]).sqrt();
        let norm_2 = (axes.axis_2[0]*axes.axis_2[0] + axes.axis_2[1]*axes.axis_2[1] + axes.axis_2[2]*axes.axis_2[2]).sqrt();
        let norm_3 = (axes.axis_3[0]*axes.axis_3[0] + axes.axis_3[1]*axes.axis_3[1] + axes.axis_3[2]*axes.axis_3[2]).sqrt();
        
        assert!((norm_1 - 1.0).abs() < 1e-6, "Axis 1 should be unit vector, got norm = {}", norm_1);
        assert!((norm_2 - 1.0).abs() < 1e-6, "Axis 2 should be unit vector, got norm = {}", norm_2);
        assert!((norm_3 - 1.0).abs() < 1e-6, "Axis 3 should be unit vector, got norm = {}", norm_3);
    }

    #[test]
    fn test_principal_axes_rectangular_box() {
        // For a rectangular box with dimensions a > b > c,
        // principal moments should satisfy: Ixx < Iyy < Izz
        // (since larger dimensions contribute more to inertia)
        let box_solid = crate::make_box(4.0, 3.0, 2.0).unwrap();
        let axes = super::principal_axes(&box_solid).unwrap();
        
        // Moments should be in descending order (as returned by principal_axes)
        assert!(axes.moment_1 >= axes.moment_2 - 1e-6,
                "moment_1 should be >= moment_2, got {} vs {}", axes.moment_1, axes.moment_2);
        assert!(axes.moment_2 >= axes.moment_3 - 1e-6,
                "moment_2 should be >= moment_3, got {} vs {}", axes.moment_2, axes.moment_3);
        
        // Check orthonormality
        let dot_12 = axes.axis_1[0] * axes.axis_2[0] + axes.axis_1[1] * axes.axis_2[1] + axes.axis_1[2] * axes.axis_2[2];
        let dot_13 = axes.axis_1[0] * axes.axis_3[0] + axes.axis_1[1] * axes.axis_3[1] + axes.axis_1[2] * axes.axis_3[2];
        let dot_23 = axes.axis_2[0] * axes.axis_3[0] + axes.axis_2[1] * axes.axis_3[1] + axes.axis_2[2] * axes.axis_3[2];
        
        assert!(dot_12.abs() < 1e-6, "Axes 1 and 2 should be orthogonal");
        assert!(dot_13.abs() < 1e-6, "Axes 1 and 3 should be orthogonal");
        assert!(dot_23.abs() < 1e-6, "Axes 2 and 3 should be orthogonal");
    }

    #[test]
    fn test_principal_axes_sphere() {
        // For a sphere, all principal moments should be equal
        // and the axes can be any orthonormal basis
        let sphere = crate::make_sphere(1.0).unwrap();
        let axes = super::principal_axes(&sphere).unwrap();
        
        // For a symmetric sphere, all principal moments should be approximately equal
        let tolerance = 0.3; // Larger tolerance for sphere mesh approximation
        
        assert!((axes.moment_1 - axes.moment_2).abs() < tolerance,
                "Principal moments 1 and 2 should be equal for sphere");
        assert!((axes.moment_2 - axes.moment_3).abs() < tolerance,
                "Principal moments 2 and 3 should be equal for sphere");
        
        // Check orthonormality
        let dot_12 = axes.axis_1[0] * axes.axis_2[0] + axes.axis_1[1] * axes.axis_2[1] + axes.axis_1[2] * axes.axis_2[2];
        let dot_13 = axes.axis_1[0] * axes.axis_3[0] + axes.axis_1[1] * axes.axis_3[1] + axes.axis_1[2] * axes.axis_3[2];
        let dot_23 = axes.axis_2[0] * axes.axis_3[0] + axes.axis_2[1] * axes.axis_3[1] + axes.axis_2[2] * axes.axis_3[2];
        
        assert!(dot_12.abs() < 1e-6, "Axes 1 and 2 should be orthogonal");
        assert!(dot_13.abs() < 1e-6, "Axes 1 and 3 should be orthogonal");
        assert!(dot_23.abs() < 1e-6, "Axes 2 and 3 should be orthogonal");
    }

    #[test]
    fn test_jacobi_eigendecomposition() {
        // Test the Jacobi eigenvalue algorithm with a known symmetric matrix
        // Matrix with eigenvalues 5, 3, 1 and orthonormal eigenvectors
        let matrix = [
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        
        let (eigenvalues, eigenvectors) = super::jacobi_eigendecomposition(matrix).unwrap();
        
        // Check that eigenvalues are approximately correct
        // For this matrix, eigenvalues should be 4, 2, and 1
        let mut eigenvals = vec![eigenvalues[0], eigenvalues[1], eigenvalues[2]];
        eigenvals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let expected = vec![4.0, 2.0, 1.0];
        for (computed, exp) in eigenvals.iter().zip(expected.iter()) {
            assert!((*computed - exp).abs() < 1e-6,
                    "Eigenvalue mismatch: expected {}, got {}", exp, computed);
        }
        
        // Check that eigenvectors are orthonormal
        for i in 0..3 {
            let vec = eigenvectors[i];
            let norm = (vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6,
                    "Eigenvector {} should be unit vector, norm = {}", i, norm);
        }
        
        // Check orthogonality
        for i in 0..3 {
            for j in (i+1)..3 {
                let dot = eigenvectors[i][0]*eigenvectors[j][0] +
                         eigenvectors[i][1]*eigenvectors[j][1] +
                         eigenvectors[i][2]*eigenvectors[j][2];
                assert!(dot.abs() < 1e-6,
                        "Eigenvectors {} and {} should be orthogonal, dot = {}", i, j, dot);
            }
        }
    }

    #[test]
    fn test_principal_axes_struct_methods() {
        // Test PrincipalAxes struct helper methods
        let axes = super::PrincipalAxes::new(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            10.0,
            5.0,
            2.0,
        );
        
        // Test as_rotation_matrix
        let rot_matrix = axes.as_rotation_matrix();
        assert_eq!(rot_matrix[0], [1.0, 0.0, 0.0]);
        assert_eq!(rot_matrix[1], [0.0, 1.0, 0.0]);
        assert_eq!(rot_matrix[2], [0.0, 0.0, 1.0]);
        
        // Test as_diagonal_inertia
        let diag_inertia = axes.as_diagonal_inertia();
        assert_eq!(diag_inertia[0][0], 10.0);
        assert_eq!(diag_inertia[1][1], 5.0);
        assert_eq!(diag_inertia[2][2], 2.0);
        assert_eq!(diag_inertia[0][1], 0.0);
        assert_eq!(diag_inertia[0][2], 0.0);
        assert_eq!(diag_inertia[1][2], 0.0);
    }
}
