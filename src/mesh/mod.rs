//! Tessellation and meshing

use crate::brep::{Face, SurfaceType, Wire};
use crate::brep::Solid;
use crate::{Result, CascadeError};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleMesh {
    pub vertices: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

/// Parametric domain for surface tessellation
/// 
/// Represents a rectangular region in parametric space (u, v) used for
/// domain-aware tessellation of surfaces. Allows specifying custom parameter ranges
/// and subdivision counts for fine control over mesh generation.
/// 
/// # Example
/// ```ignore
/// use cascade_rs::mesh::MeshDomain;
/// 
/// // Create a domain spanning [0, 1] in both u and v with 10x10 subdivisions
/// let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (10, 10))?;
/// 
/// // Sample parametric points
/// let points = domain.sample_points();
/// assert_eq!(points.len(), 11 * 11); // (subdivisions+1)^2 points
/// 
/// // Check if a point is in the domain
/// assert!(domain.contains(0.5, 0.5));
/// assert!(!domain.contains(1.5, 0.5));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshDomain {
    /// Parameter range in u direction: (u_min, u_max)
    pub u_range: (f64, f64),
    /// Parameter range in v direction: (v_min, v_max)
    pub v_range: (f64, f64),
    /// Number of subdivisions in (u, v) directions
    pub subdivisions: (usize, usize),
}

impl MeshDomain {
    /// Create a new mesh domain with the specified parameter ranges and subdivisions
    ///
    /// # Arguments
    /// * `u_range` - Tuple (u_min, u_max) for u parameter range
    /// * `v_range` - Tuple (v_min, v_max) for v parameter range
    /// * `subdivisions` - Tuple (u_subdivisions, v_subdivisions) for grid density
    ///
    /// # Returns
    /// * `Result<MeshDomain>` - A new domain or error if parameters are invalid
    ///
    /// # Errors
    /// Returns error if:
    /// - u_min >= u_max or v_min >= v_max (invalid ranges)
    /// - subdivisions are zero
    pub fn new(u_range: (f64, f64), v_range: (f64, f64), subdivisions: (usize, usize)) -> Result<Self> {
        if u_range.0 >= u_range.1 {
            return Err(CascadeError::InvalidGeometry(
                format!("Invalid u_range: u_min ({}) must be < u_max ({})", u_range.0, u_range.1)
            ));
        }
        if v_range.0 >= v_range.1 {
            return Err(CascadeError::InvalidGeometry(
                format!("Invalid v_range: v_min ({}) must be < v_max ({})", v_range.0, v_range.1)
            ));
        }
        if subdivisions.0 == 0 || subdivisions.1 == 0 {
            return Err(CascadeError::InvalidGeometry(
                "Subdivisions must be positive (non-zero)".into()
            ));
        }
        
        Ok(MeshDomain {
            u_range,
            v_range,
            subdivisions,
        })
    }

    /// Check if a parametric point (u, v) is contained within this domain
    ///
    /// # Arguments
    /// * `u` - U parameter value
    /// * `v` - V parameter value
    ///
    /// # Returns
    /// True if the point is within the domain bounds, false otherwise
    pub fn contains(&self, u: f64, v: f64) -> bool {
        u >= self.u_range.0 && u <= self.u_range.1 &&
        v >= self.v_range.0 && v <= self.v_range.1
    }

    /// Generate a grid of parametric sample points in the domain
    ///
    /// Creates a regular grid of (u, v) parameter points based on the domain's
    /// subdivisions. Returns (subdivisions+1)^2 points total, including boundary points.
    ///
    /// # Returns
    /// Vector of (u, v) parametric coordinates
    ///
    /// # Example
    /// ```ignore
    /// let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (2, 2))?;
    /// let points = domain.sample_points();
    /// // Points: (0,0), (0,0.5), (0,1), (0.5,0), (0.5,0.5), (0.5,1), (1,0), (1,0.5), (1,1)
    /// assert_eq!(points.len(), 9); // 3x3 grid
    /// ```
    pub fn sample_points(&self) -> Vec<(f64, f64)> {
        let mut points = Vec::new();
        let (u_subdivisions, v_subdivisions) = self.subdivisions;
        let (u_min, u_max) = self.u_range;
        let (v_min, v_max) = self.v_range;
        
        // Generate grid points
        for u_idx in 0..=u_subdivisions {
            let u = u_min + (u_idx as f64) / (u_subdivisions as f64) * (u_max - u_min);
            
            for v_idx in 0..=v_subdivisions {
                let v = v_min + (v_idx as f64) / (v_subdivisions as f64) * (v_max - v_min);
                points.push((u, v));
            }
        }
        
        points
    }
}

/// Triangulate a parametric surface over a specified mesh domain
///
/// Creates a triangle mesh by evaluating the given surface at parametric points
/// defined by the mesh domain. Generates a regular grid tessellation of the domain.
///
/// # Arguments
/// * `surface` - The parametric surface to tessellate
/// * `domain` - The parametric domain defining the region and resolution
///
/// # Returns
/// * `Result<TriangleMesh>` - The tessellated mesh with vertices, normals, and triangle indices
///
/// # Example
/// ```ignore
/// use cascade_rs::mesh::{MeshDomain, triangulate_domain};
/// use cascade_rs::brep::SurfaceType;
/// 
/// let surface = SurfaceType::Sphere {
///     center: [0.0, 0.0, 0.0],
///     radius: 5.0,
/// };
/// 
/// let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (16, 32))?;
/// let mesh = triangulate_domain(&surface, &domain)?;
/// 
/// assert!(mesh.vertices.len() > 0);
/// assert!(mesh.triangles.len() > 0);
/// assert_eq!(mesh.normals.len(), mesh.vertices.len());
/// ```
pub fn triangulate_domain(surface: &SurfaceType, domain: &MeshDomain) -> Result<TriangleMesh> {
    let (u_subdivisions, v_subdivisions) = domain.subdivisions;
    let (u_min, u_max) = domain.u_range;
    let (v_min, v_max) = domain.v_range;
    
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut triangles = Vec::new();
    
    // Create vertex grid
    for u_idx in 0..=u_subdivisions {
        let u = u_min + (u_idx as f64) / (u_subdivisions as f64) * (u_max - u_min);
        
        for v_idx in 0..=v_subdivisions {
            let v = v_min + (v_idx as f64) / (v_subdivisions as f64) * (v_max - v_min);
            
            // Evaluate point and normal on surface
            let pt = surface.point_at(u, v);
            let normal = surface.normal_at(u, v);
            
            vertices.push(pt);
            normals.push(normal);
        }
    }
    
    // Create triangles by connecting grid points
    // Each quad is subdivided into two triangles
    for u_idx in 0..u_subdivisions {
        for v_idx in 0..v_subdivisions {
            let v0 = u_idx * (v_subdivisions + 1) + v_idx;
            let v1 = u_idx * (v_subdivisions + 1) + v_idx + 1;
            let v2 = (u_idx + 1) * (v_subdivisions + 1) + v_idx;
            let v3 = (u_idx + 1) * (v_subdivisions + 1) + v_idx + 1;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(TriangleMesh {
        vertices,
        normals,
        triangles,
    })
}

/// Incremental mesh builder that tessellates a solid progressively by adding faces one at a time
///
/// This is useful for partial updates when only some faces of a solid change,
/// or when you want to progressively build a mesh for streaming/rendering.
///
/// # Example
/// ```ignore
/// let solid = make_box(10.0, 10.0, 10.0)?;
/// let mut incremental = IncrementalMesh::new(&solid, 0.1)?;
///
/// // Add first face
/// incremental.add_face(0)?;
/// let partial_mesh = incremental.build()?;
/// println!("First face: {} triangles", partial_mesh.triangles.len());
///
/// // Add more faces
/// for i in 1..solid.outer_shell.faces.len() {
///     incremental.add_face(i)?;
/// }
/// let full_mesh = incremental.build()?;
/// println!("Full mesh: {} triangles", full_mesh.triangles.len());
/// ```
#[derive(Debug, Clone)]
pub struct IncrementalMesh {
    /// Reference to the solid being tessellated
    solid: Solid,
    /// Tolerance for tessellation
    tolerance: f64,
    /// Set of face indices (outer shell) that have been added
    outer_faces: HashSet<usize>,
    /// Mapping from (shell_idx, face_idx_in_shell) to which inner shell faces are added
    inner_faces: HashSet<(usize, usize)>,
}

impl IncrementalMesh {
    /// Create a new incremental mesh builder for a solid
    ///
    /// # Arguments
    /// * `solid` - The solid to tessellate
    /// * `tolerance` - Tolerance for tessellation (deflection)
    ///
    /// # Returns
    /// * `Result<IncrementalMesh>` - A new incremental mesh builder
    ///
    /// # Example
    /// ```ignore
    /// let solid = make_sphere(5.0)?;
    /// let incremental = IncrementalMesh::new(&solid, 0.1)?;
    /// ```
    pub fn new(solid: &Solid, tolerance: f64) -> Result<Self> {
        if tolerance <= 0.0 {
            return Err(CascadeError::InvalidGeometry(
                "Tolerance must be positive".into(),
            ));
        }
        
        Ok(IncrementalMesh {
            solid: solid.clone(),
            tolerance,
            outer_faces: HashSet::new(),
            inner_faces: HashSet::new(),
        })
    }
    
    /// Get the total number of faces available in the solid
    ///
    /// This includes all faces from the outer shell and all inner shells.
    ///
    /// # Returns
    /// The total count of faces
    pub fn total_faces(&self) -> usize {
        let mut count = self.solid.outer_shell.faces.len();
        for shell in &self.solid.inner_shells {
            count += shell.faces.len();
        }
        count
    }
    
    /// Add a single face to the mesh by its global index
    ///
    /// The face index is global across the solid:
    /// - Indices 0..(outer_faces_len) refer to outer shell faces
    /// - Indices outer_faces_len.. refer to inner shell faces (shell_idx, face_idx)
    ///
    /// # Arguments
    /// * `face_idx` - Global index of the face to add
    ///
    /// # Returns
    /// * `Result<()>` - Ok if the face was added, error if index out of bounds
    pub fn add_face(&mut self, face_idx: usize) -> Result<()> {
        let outer_count = self.solid.outer_shell.faces.len();
        
        if face_idx < outer_count {
            // Outer shell face
            self.outer_faces.insert(face_idx);
            Ok(())
        } else {
            // Inner shell face
            let idx = face_idx - outer_count;
            let mut remaining = idx;
            
            for shell_idx in 0..self.solid.inner_shells.len() {
                let shell_size = self.solid.inner_shells[shell_idx].faces.len();
                if remaining < shell_size {
                    self.inner_faces.insert((shell_idx, remaining));
                    return Ok(());
                }
                remaining -= shell_size;
            }
            
            Err(CascadeError::InvalidGeometry(
                format!("Face index {} out of bounds", face_idx),
            ))
        }
    }
    
    /// Add multiple faces by their global indices
    ///
    /// # Arguments
    /// * `face_indices` - Slice of global face indices to add
    ///
    /// # Returns
    /// * `Result<()>` - Ok if all faces were added successfully
    pub fn add_faces(&mut self, face_indices: &[usize]) -> Result<()> {
        for &idx in face_indices {
            self.add_face(idx)?;
        }
        Ok(())
    }
    
    /// Check if a face has been added to the mesh
    ///
    /// # Arguments
    /// * `face_idx` - Global index of the face to check
    ///
    /// # Returns
    /// Whether the face is included in the current mesh
    pub fn contains_face(&self, face_idx: usize) -> bool {
        let outer_count = self.solid.outer_shell.faces.len();
        
        if face_idx < outer_count {
            self.outer_faces.contains(&face_idx)
        } else {
            let idx = face_idx - outer_count;
            let mut remaining = idx;
            
            for shell_idx in 0..self.solid.inner_shells.len() {
                let shell_size = self.solid.inner_shells[shell_idx].faces.len();
                if remaining < shell_size {
                    return self.inner_faces.contains(&(shell_idx, remaining));
                }
                remaining -= shell_size;
            }
            
            false
        }
    }
    
    /// Get the number of faces that have been added
    ///
    /// # Returns
    /// The count of added faces
    pub fn added_faces_count(&self) -> usize {
        self.outer_faces.len() + self.inner_faces.len()
    }
    
    /// Clear all added faces
    pub fn clear(&mut self) {
        self.outer_faces.clear();
        self.inner_faces.clear();
    }
    
    /// Build and return the incremental triangle mesh
    ///
    /// This tessellates only the faces that have been added via `add_face()` or `add_faces()`.
    /// The final mesh respects vertex deduplication and triangle connectivity for added faces.
    ///
    /// # Returns
    /// * `Result<TriangleMesh>` - The tessellated mesh containing only added faces
    ///
    /// # Example
    /// ```ignore
    /// let mut incremental = IncrementalMesh::new(&solid, 0.1)?;
    /// incremental.add_face(0)?;
    /// incremental.add_face(1)?;
    /// let mesh = incremental.build()?;
    /// ```
    pub fn build(&self) -> Result<TriangleMesh> {
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut triangles = Vec::new();
        
        // Process added outer shell faces
        for &face_idx in &self.outer_faces {
            if face_idx < self.solid.outer_shell.faces.len() {
                let face = &self.solid.outer_shell.faces[face_idx];
                triangulate_face(
                    face,
                    self.tolerance,
                    &mut vertices,
                    &mut normals,
                    &mut triangles,
                )?;
            }
        }
        
        // Process added inner shell faces
        for &(shell_idx, face_idx) in &self.inner_faces {
            if shell_idx < self.solid.inner_shells.len() {
                let shell = &self.solid.inner_shells[shell_idx];
                if face_idx < shell.faces.len() {
                    let face = &shell.faces[face_idx];
                    triangulate_face(
                        face,
                        self.tolerance,
                        &mut vertices,
                        &mut normals,
                        &mut triangles,
                    )?;
                }
            }
        }
        
        Ok(TriangleMesh {
            vertices,
            normals,
            triangles,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolygonMesh {
    pub vertices: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
}

impl PolygonMesh {
    /// Create a new empty polygon mesh
    pub fn new() -> Self {
        PolygonMesh {
            vertices: Vec::new(),
            normals: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Add a vertex to the mesh
    ///
    /// # Arguments
    /// * `vertex` - The 3D coordinate of the vertex
    /// * `normal` - The normal vector at the vertex
    ///
    /// # Returns
    /// The index of the newly added vertex
    pub fn add_vertex(&mut self, vertex: [f64; 3], normal: [f64; 3]) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(vertex);
        self.normals.push(normal);
        idx
    }

    /// Add a face with variable vertex count
    ///
    /// # Arguments
    /// * `indices` - Slice of vertex indices that form the face (can be triangle, quad, or n-gon)
    ///
    /// # Panics
    /// If any index is out of bounds or if fewer than 3 vertices are provided
    pub fn add_face(&mut self, indices: &[usize]) {
        assert!(
            indices.len() >= 3,
            "Face must have at least 3 vertices"
        );
        assert!(
            indices.iter().all(|&idx| idx < self.vertices.len()),
            "Face index out of bounds"
        );
        self.faces.push(indices.to_vec());
    }

    /// Convert this polygon mesh to a triangle mesh via fan triangulation
    ///
    /// Each polygon with N vertices (N >= 3) is triangulated by connecting all
    /// vertices to the first vertex (fan triangulation).
    ///
    /// # Returns
    /// A TriangleMesh with all polygons decomposed into triangles
    pub fn to_triangle_mesh(&self) -> TriangleMesh {
        let mut triangles = Vec::new();

        for face in &self.faces {
            // Fan triangulation: use first vertex as the center
            // For a polygon with n vertices at indices [i0, i1, i2, ..., i(n-1)],
            // create triangles: (i0, i1, i2), (i0, i2, i3), ..., (i0, i(n-2), i(n-1))
            for i in 1..(face.len() - 1) {
                triangles.push([face[0], face[i], face[i + 1]]);
            }
        }

        TriangleMesh {
            vertices: self.vertices.clone(),
            normals: self.normals.clone(),
            triangles,
        }
    }
}

impl Default for PolygonMesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Triangulate a solid into a mesh
///
/// For planar faces: simple triangulation
/// For curved faces: subdivide based on tolerance
pub fn triangulate(solid: &Solid, tolerance: f64) -> Result<TriangleMesh> {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut triangles = Vec::new();
    
    // Process outer shell
    for face in &solid.outer_shell.faces {
        triangulate_face(face, tolerance, &mut vertices, &mut normals, &mut triangles)?;
    }
    
    // Process inner shells (voids)
    for shell in &solid.inner_shells {
        for face in &shell.faces {
            triangulate_face(face, tolerance, &mut vertices, &mut normals, &mut triangles)?;
        }
    }
    
    Ok(TriangleMesh { vertices, normals, triangles })
}

/// Triangulate a solid into a mesh with deflection (chord height) control
///
/// This function adaptively subdivides the mesh based on deflection tolerance.
/// Deflection = the maximum allowed distance between the mesh edge (chord) and the actual curved surface.
///
/// # Arguments
/// * `solid` - The solid to triangulate
/// * `deflection` - Maximum chord height tolerance (smaller = finer mesh = more triangles)
///
/// # Example
/// ```ignore
/// let sphere = make_sphere(10.0)?;
/// 
/// // Coarse mesh with 1.0 deflection
/// let mesh_coarse = triangulate_with_deflection(&sphere, 1.0)?;
/// 
/// // Fine mesh with 0.1 deflection
/// let mesh_fine = triangulate_with_deflection(&sphere, 0.1)?;
/// 
/// // Fine mesh will have significantly more triangles
/// assert!(mesh_fine.triangles.len() > mesh_coarse.triangles.len());
/// ```
pub fn triangulate_with_deflection(solid: &Solid, deflection: f64) -> Result<TriangleMesh> {
    if deflection <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Deflection must be positive".into(),
        ));
    }
    
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut triangles = Vec::new();
    
    // Process outer shell
    for face in &solid.outer_shell.faces {
        triangulate_face(face, deflection, &mut vertices, &mut normals, &mut triangles)?;
    }
    
    // Process inner shells (voids)
    for shell in &solid.inner_shells {
        for face in &shell.faces {
            triangulate_face(face, deflection, &mut vertices, &mut normals, &mut triangles)?;
        }
    }
    
    Ok(TriangleMesh { vertices, normals, triangles })
}

/// Triangulate a solid into a mesh with angle control
///
/// This function adaptively subdivides the mesh based on the maximum allowed angle
/// between adjacent triangle normals. This is a measure of surface curvature.
///
/// For a curved surface, a smaller max_angle means the mesh must have finer detail
/// to keep the angle between adjacent triangle normals below the threshold.
///
/// # Arguments
/// * `solid` - The solid to triangulate
/// * `max_angle` - Maximum allowed angle between adjacent triangle normals (in radians)
///
/// # Example
/// ```ignore
/// let sphere = make_sphere(10.0)?;
/// 
/// // Coarse mesh with 0.2 rad angle control (about 11.5°)
/// let mesh_coarse = triangulate_with_angle(&sphere, 0.2)?;
/// 
/// // Fine mesh with 0.05 rad angle control (about 2.9°)
/// let mesh_fine = triangulate_with_angle(&sphere, 0.05)?;
/// 
/// // Fine mesh will have significantly more triangles
/// assert!(mesh_fine.triangles.len() > mesh_coarse.triangles.len());
/// ```
pub fn triangulate_with_angle(solid: &Solid, max_angle: f64) -> Result<TriangleMesh> {
    if max_angle <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "max_angle must be positive".into(),
        ));
    }
    
    if max_angle > std::f64::consts::PI {
        return Err(CascadeError::InvalidGeometry(
            "max_angle must be less than π radians".into(),
        ));
    }
    
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut triangles = Vec::new();
    
    // Process outer shell
    for face in &solid.outer_shell.faces {
        triangulate_face_with_angle(face, max_angle, &mut vertices, &mut normals, &mut triangles)?;
    }
    
    // Process inner shells (voids)
    for shell in &solid.inner_shells {
        for face in &shell.faces {
            triangulate_face_with_angle(face, max_angle, &mut vertices, &mut normals, &mut triangles)?;
        }
    }
    
    Ok(TriangleMesh { vertices, normals, triangles })
}

/// Triangulate a single face and collect its triangles
fn triangulate_face(
    face: &Face,
    tolerance: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            triangulate_planar_face(face, origin, normal, vertices, normals, triangles)?;
        }
        SurfaceType::Cylinder { origin, axis, radius } => {
            triangulate_cylindrical_face(face, origin, axis, *radius, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::Sphere { center, radius } => {
            triangulate_spherical_face(face, center, *radius, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::Cone { origin, axis, half_angle_rad } => {
            triangulate_conical_face(face, origin, axis, *half_angle_rad, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::Torus { center, major_radius, minor_radius } => {
            triangulate_toroidal_face(face, center, *major_radius, *minor_radius, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::BSpline {
            u_degree,
            v_degree,
            u_knots,
            v_knots,
            control_points,
            weights,
        } => {
            triangulate_bspline_face(
                face,
                *u_degree,
                *v_degree,
                u_knots,
                v_knots,
                control_points,
                weights.as_ref(),
                tolerance,
                vertices,
                normals,
                triangles,
            )?;
        }
        SurfaceType::BezierSurface { .. } => {
            triangulate_parametric_surface(face, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::SurfaceOfRevolution { .. } => {
            triangulate_parametric_surface(face, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::SurfaceOfLinearExtrusion { .. } => {
            triangulate_parametric_surface(face, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::RectangularTrimmedSurface { .. } => {
            triangulate_parametric_surface(face, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::OffsetSurface { .. } => {
            triangulate_parametric_surface(face, tolerance, vertices, normals, triangles)?;
        }
        SurfaceType::PlateSurface { .. } => {
            // PlateSurface is for plate/filling surfaces - use parametric surface handler
            triangulate_parametric_surface(face, tolerance, vertices, normals, triangles)?;
        }
    }
    Ok(())
}

/// Triangulate a single face with angle control
///
/// The angle parameter controls the maximum allowed angle between adjacent triangle normals.
/// For a sphere, this is converted to an equivalent deflection using the formula:
/// deflection = radius * (1 - cos(max_angle/2))
///
/// For other surface types, we estimate the local radius of curvature and apply a similar conversion.
fn triangulate_face_with_angle(
    face: &Face,
    max_angle: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            // Planar faces don't need angle control
            triangulate_planar_face(face, origin, normal, vertices, normals, triangles)?;
        }
        SurfaceType::Sphere { center, radius } => {
            // For sphere, convert max_angle to deflection
            // For a chord subtending angle θ at center: deflection ≈ R * (1 - cos(θ/2))
            let deflection = radius * (1.0 - (max_angle / 2.0).cos());
            triangulate_spherical_face(face, center, *radius, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::Cylinder { origin, axis, radius } => {
            // For cylinder, the angle control is primarily in the circumferential direction
            // The relationship between max_angle and deflection is: deflection ≈ radius * max_angle
            let deflection = radius * max_angle;
            triangulate_cylindrical_face(face, origin, axis, *radius, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::Cone { origin, axis, half_angle_rad } => {
            // For cone, similar to cylinder but with varying radius
            // Use a conservative estimate with the half-angle taken into account
            let deflection = (half_angle_rad.tan().abs() + 1.0) * max_angle;
            triangulate_conical_face(face, origin, axis, *half_angle_rad, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::Torus { center, major_radius, minor_radius } => {
            // For torus, the curvature varies between 1/major_radius and 1/minor_radius
            // Use the smaller radius (more curved) for conservative deflection
            let effective_radius = if minor_radius < major_radius { *minor_radius } else { *major_radius };
            let deflection = effective_radius * (1.0 - (max_angle / 2.0).cos());
            triangulate_toroidal_face(face, center, *major_radius, *minor_radius, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::BSpline {
            u_degree,
            v_degree,
            u_knots,
            v_knots,
            control_points,
            weights,
        } => {
            // For BSpline, estimate the extent and use a conservative deflection
            // Flatten the control points for the angle estimation
            let flat_points: Vec<[f64; 3]> = control_points.iter().flat_map(|row| row.iter().copied()).collect();
            let deflection = estimate_deflection_from_angle(max_angle, &flat_points);
            triangulate_bspline_face(
                face,
                *u_degree,
                *v_degree,
                u_knots,
                v_knots,
                control_points,
                weights.as_ref(),
                deflection,
                vertices,
                normals,
                triangles,
            )?;
        }
        SurfaceType::BezierSurface { .. } => {
            let deflection = max_angle * 0.1; // Conservative estimate
            triangulate_parametric_surface(face, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::SurfaceOfRevolution { .. } => {
            let deflection = max_angle * 0.1;
            triangulate_parametric_surface(face, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::SurfaceOfLinearExtrusion { .. } => {
            let deflection = max_angle * 0.1;
            triangulate_parametric_surface(face, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::RectangularTrimmedSurface { .. } => {
            let deflection = max_angle * 0.1;
            triangulate_parametric_surface(face, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::OffsetSurface { .. } => {
            let deflection = max_angle * 0.1;
            triangulate_parametric_surface(face, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::PlateSurface { .. } => {
            let deflection = max_angle * 0.1;
            triangulate_parametric_surface(face, deflection, vertices, normals, triangles)?;
        }
    }
    Ok(())
}

/// Triangulate a single face with deflection control
fn triangulate_face_with_deflection(
    face: &Face,
    deflection: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            // Planar faces don't need deflection control
            triangulate_planar_face(face, origin, normal, vertices, normals, triangles)?;
        }
        SurfaceType::Sphere { center, radius } => {
            // For sphere, use deflection to control subdivision
            // TODO: Implement triangulate_spherical_face_deflection
            // For now, use planar approximation
            if !face.outer_wire.edges.is_empty() {
                let v0 = &face.outer_wire.edges[0].start.point;
                let normal = [
                    center[0] - v0[0],
                    center[1] - v0[1],
                    center[2] - v0[2],
                ];
                let norm_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                let normal = if norm_len > 1e-6 {
                    [normal[0] / norm_len, normal[1] / norm_len, normal[2] / norm_len]
                } else {
                    [0.0, 0.0, 1.0]
                };
                triangulate_planar_face(face, center, &normal, vertices, normals, triangles)?;
            }
        }
        SurfaceType::Cylinder { origin, axis, radius } => {
            // Cylinder is developable, use deflection
            triangulate_cylindrical_face_deflection(face, origin, axis, *radius, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::Cone { origin, axis, half_angle_rad } => {
            // Cone is developable, use deflection
            triangulate_conical_face_deflection(face, origin, axis, *half_angle_rad, deflection, vertices, normals, triangles)?;
        }
        SurfaceType::Torus { center, major_radius, minor_radius } => {
            // Torus is curved, use deflection
            triangulate_toroidal_face_deflection(face, center, *major_radius, *minor_radius, deflection, vertices, normals, triangles)?;
        }
        _ => {
            // For other surface types, fall back to fixed subdivision
            // (could be enhanced for BSpline, Bezier, etc.)
            triangulate_parametric_surface(face, deflection, vertices, normals, triangles)?;
        }
    }
    Ok(())
}

/// Triangulate a planar face using ear clipping algorithm
fn triangulate_planar_face(
    face: &Face,
    _origin: &[f64; 3],
    normal: &[f64; 3],
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    // Get 2D projection of the outer wire
    let outer_points = wire_to_points(&face.outer_wire);
    
    if outer_points.len() < 3 {
        return Err(CascadeError::InvalidGeometry(
            "Face must have at least 3 vertices".into(),
        ));
    }
    
    // Simple triangulation: fan triangulation from first vertex
    let base_idx = vertices.len();
    let normal_vec = normalize(normal);
    
    // Add all outer wire vertices
    for point in &outer_points {
        vertices.push(*point);
        normals.push(normal_vec);
    }
    
    // Fan triangulation
    for i in 1..(outer_points.len() - 1) {
        triangles.push([
            base_idx,
            base_idx + i,
            base_idx + i + 1,
        ]);
    }
    
    // TODO: Handle holes (inner wires) if needed
    Ok(())
}

/// Triangulate a cylindrical face
fn triangulate_cylindrical_face(
    face: &Face,
    origin: &[f64; 3],
    axis: &[f64; 3],
    radius: f64,
    tolerance: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    let axis_normalized = normalize(axis);
    
    // Create perpendicular vectors
    let perp1 = perpendicular_to(&axis_normalized);
    let perp2 = cross(&axis_normalized, &perp1);
    
    // Estimate angle subdivisions needed
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let angle_subdivisions = ((circumference / tolerance).ceil() as usize).max(8);
    
    // Get the height range from the wire
    let outer_points = wire_to_points(&face.outer_wire);
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    
    for point in &outer_points {
        let z = dot(&[point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]], &axis_normalized);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    
    let height_subdivisions = ((max_z - min_z) / tolerance).ceil() as usize + 1;
    
    let base_idx = vertices.len();
    
    // Create vertex grid
    for h in 0..=height_subdivisions {
        let z_param = min_z + (h as f64) / (height_subdivisions as f64) * (max_z - min_z);
        let z_offset = scale_vec(&axis_normalized, z_param);
        
        for a in 0..angle_subdivisions {
            let angle = (a as f64) / (angle_subdivisions as f64) * 2.0 * std::f64::consts::PI;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            
            let x = origin[0] + z_offset[0] + (cos_a * perp1[0] + sin_a * perp2[0]) * radius;
            let y = origin[1] + z_offset[1] + (cos_a * perp1[1] + sin_a * perp2[1]) * radius;
            let z = origin[2] + z_offset[2] + (cos_a * perp1[2] + sin_a * perp2[2]) * radius;
            
            vertices.push([x, y, z]);
            
            // Normal points outward from cylinder axis
            let normal = normalize(&[
                x - origin[0] - z_offset[0],
                y - origin[1] - z_offset[1],
                z - origin[2] - z_offset[2],
            ]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for h in 0..height_subdivisions {
        for a in 0..angle_subdivisions {
            let a_next = (a + 1) % angle_subdivisions;
            
            let v0 = base_idx + h * angle_subdivisions + a;
            let v1 = base_idx + h * angle_subdivisions + a_next;
            let v2 = base_idx + (h + 1) * angle_subdivisions + a;
            let v3 = base_idx + (h + 1) * angle_subdivisions + a_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a cylindrical face with deflection control
fn triangulate_cylindrical_face_deflection(
    face: &Face,
    origin: &[f64; 3],
    axis: &[f64; 3],
    radius: f64,
    deflection: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    let axis_normalized = normalize(axis);
    
    // Create perpendicular vectors
    let perp1 = perpendicular_to(&axis_normalized);
    let perp2 = cross(&axis_normalized, &perp1);
    
    // Get the height range from the wire
    let outer_points = wire_to_points(&face.outer_wire);
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    
    for point in &outer_points {
        let z = dot(&[point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]], &axis_normalized);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    
    // For cylinder circumference, use deflection formula for circle: 
    // chord height s = r * (1 - cos(θ/2)), solve for angle
    let sagitta = deflection.min(radius);
    let cos_half_angle = 1.0 - sagitta / radius;
    let cos_half_angle = cos_half_angle.clamp(-1.0, 1.0);
    let half_angle = cos_half_angle.acos();
    let angle_per_edge = 2.0 * half_angle;
    
    let two_pi = 2.0 * std::f64::consts::PI;
    let angle_subdivisions = (two_pi / angle_per_edge).ceil() as usize;
    let angle_subdivisions = angle_subdivisions.max(3);
    
    // Height subdivisions: use deflection as tolerance
    let height_subdivisions = ((max_z - min_z) / deflection).ceil() as usize + 1;
    let height_subdivisions = height_subdivisions.max(2);
    
    let base_idx = vertices.len();
    
    // Create vertex grid
    for h in 0..=height_subdivisions {
        let z_param = min_z + (h as f64) / (height_subdivisions as f64) * (max_z - min_z);
        let z_offset = scale_vec(&axis_normalized, z_param);
        
        for a in 0..angle_subdivisions {
            let angle = (a as f64) / (angle_subdivisions as f64) * two_pi;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            
            let x = origin[0] + z_offset[0] + (cos_a * perp1[0] + sin_a * perp2[0]) * radius;
            let y = origin[1] + z_offset[1] + (cos_a * perp1[1] + sin_a * perp2[1]) * radius;
            let z = origin[2] + z_offset[2] + (cos_a * perp1[2] + sin_a * perp2[2]) * radius;
            
            vertices.push([x, y, z]);
            
            // Normal points outward from cylinder axis
            let normal = normalize(&[
                x - origin[0] - z_offset[0],
                y - origin[1] - z_offset[1],
                z - origin[2] - z_offset[2],
            ]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for h in 0..height_subdivisions {
        for a in 0..angle_subdivisions {
            let a_next = (a + 1) % angle_subdivisions;
            
            let v0 = base_idx + h * angle_subdivisions + a;
            let v1 = base_idx + h * angle_subdivisions + a_next;
            let v2 = base_idx + (h + 1) * angle_subdivisions + a;
            let v3 = base_idx + (h + 1) * angle_subdivisions + a_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a spherical face
fn triangulate_spherical_face(
    _face: &Face,
    center: &[f64; 3],
    radius: f64,
    tolerance: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    // Estimate subdivisions based on tolerance
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let subdivisions = ((circumference / tolerance).ceil() as usize).max(4);
    
    let base_idx = vertices.len();
    
    // Latitude loops
    for lat in 0..=subdivisions {
        let lat_angle = -std::f64::consts::PI / 2.0 + (lat as f64) / (subdivisions as f64) * std::f64::consts::PI;
        let lat_sin = lat_angle.sin();
        let lat_cos = lat_angle.cos();
        
        // Longitude points
        for lon in 0..subdivisions {
            let lon_angle = (lon as f64) / (subdivisions as f64) * 2.0 * std::f64::consts::PI;
            let lon_cos = lon_angle.cos();
            let lon_sin = lon_angle.sin();
            
            let x = center[0] + radius * lat_cos * lon_cos;
            let y = center[1] + radius * lat_cos * lon_sin;
            let z = center[2] + radius * lat_sin;
            
            vertices.push([x, y, z]);
            
            // Normal is radial direction
            let normal = normalize(&[x - center[0], y - center[1], z - center[2]]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for lat in 0..subdivisions {
        for lon in 0..subdivisions {
            let lon_next = (lon + 1) % subdivisions;
            
            let v0 = base_idx + lat * subdivisions + lon;
            let v1 = base_idx + lat * subdivisions + lon_next;
            let v2 = base_idx + (lat + 1) * subdivisions + lon;
            let v3 = base_idx + (lat + 1) * subdivisions + lon_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a spherical face with deflection control
fn triangulate_spherical_face_deflection(
    _face: &Face,
    center: &[f64; 3],
    radius: f64,
    deflection: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    // For a sphere, calculate the angle subtended by a chord of given deflection
    // Chord height (sagitta) formula: s = r * (1 - cos(θ/2))
    // Solving for θ: θ = 2 * arccos(1 - s/r)
    // where s is the sagitta (deflection) and r is the radius
    
    let sagitta = deflection.min(radius); // Ensure we don't exceed radius
    let cos_half_angle = 1.0 - sagitta / radius;
    let cos_half_angle = cos_half_angle.clamp(-1.0, 1.0); // Ensure valid range for arccos
    let half_angle = cos_half_angle.acos();
    let angle_per_edge = 2.0 * half_angle;
    
    // Calculate number of subdivisions needed
    let two_pi = 2.0 * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    
    // Latitude subdivisions: from -π/2 to π/2 = π total
    let lat_subdivisions = (pi / angle_per_edge).ceil() as usize;
    let lat_subdivisions = lat_subdivisions.max(2);
    
    // Longitude subdivisions: from 0 to 2π = 2π total
    let lon_subdivisions = (two_pi / angle_per_edge).ceil() as usize;
    let lon_subdivisions = lon_subdivisions.max(3);
    
    let base_idx = vertices.len();
    
    // Latitude loops
    for lat in 0..=lat_subdivisions {
        let lat_angle = -pi / 2.0 + (lat as f64) / (lat_subdivisions as f64) * pi;
        let lat_sin = lat_angle.sin();
        let lat_cos = lat_angle.cos();
        
        // Longitude points
        for lon in 0..lon_subdivisions {
            let lon_angle = (lon as f64) / (lon_subdivisions as f64) * two_pi;
            let lon_cos = lon_angle.cos();
            let lon_sin = lon_angle.sin();
            
            let x = center[0] + radius * lat_cos * lon_cos;
            let y = center[1] + radius * lat_cos * lon_sin;
            let z = center[2] + radius * lat_sin;
            
            vertices.push([x, y, z]);
            
            // Normal is radial direction
            let normal = normalize(&[x - center[0], y - center[1], z - center[2]]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for lat in 0..lat_subdivisions {
        for lon in 0..lon_subdivisions {
            let lon_next = (lon + 1) % lon_subdivisions;
            
            let v0 = base_idx + lat * lon_subdivisions + lon;
            let v1 = base_idx + lat * lon_subdivisions + lon_next;
            let v2 = base_idx + (lat + 1) * lon_subdivisions + lon;
            let v3 = base_idx + (lat + 1) * lon_subdivisions + lon_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a conical face
fn triangulate_conical_face(
    face: &Face,
    origin: &[f64; 3],
    axis: &[f64; 3],
    half_angle_rad: f64,
    tolerance: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    let axis_normalized = normalize(axis);
    
    // Create perpendicular vectors
    let perp1 = perpendicular_to(&axis_normalized);
    let perp2 = cross(&axis_normalized, &perp1);
    
    // Get the height range from the wire
    let outer_points = wire_to_points(&face.outer_wire);
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    
    for point in &outer_points {
        let z = dot(&[point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]], &axis_normalized);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    
    // Calculate radii at min and max height
    let radius_at_min = min_z * half_angle_rad.tan();
    let radius_at_max = max_z * half_angle_rad.tan();
    let max_radius = radius_at_min.abs().max(radius_at_max.abs());
    
    // Estimate angle subdivisions needed based on largest radius
    let circumference = 2.0 * std::f64::consts::PI * max_radius;
    let angle_subdivisions = ((circumference / tolerance).ceil() as usize).max(8);
    
    let height_subdivisions = ((max_z - min_z) / tolerance).ceil() as usize + 1;
    
    let base_idx = vertices.len();
    
    // Create vertex grid
    for h in 0..=height_subdivisions {
        let z_param = min_z + (h as f64) / (height_subdivisions as f64) * (max_z - min_z);
        let radius = z_param * half_angle_rad.tan();
        let z_offset = scale_vec(&axis_normalized, z_param);
        
        for a in 0..angle_subdivisions {
            let angle = (a as f64) / (angle_subdivisions as f64) * 2.0 * std::f64::consts::PI;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            
            let x = origin[0] + z_offset[0] + (cos_a * perp1[0] + sin_a * perp2[0]) * radius;
            let y = origin[1] + z_offset[1] + (cos_a * perp1[1] + sin_a * perp2[1]) * radius;
            let z = origin[2] + z_offset[2] + (cos_a * perp1[2] + sin_a * perp2[2]) * radius;
            
            vertices.push([x, y, z]);
            
            // Normal for cone: perpendicular to cone surface
            // The surface normal makes angle (π/2 - half_angle) with axis
            let radial_dir = normalize(&[
                cos_a * perp1[0] + sin_a * perp2[0],
                cos_a * perp1[1] + sin_a * perp2[1],
                cos_a * perp1[2] + sin_a * perp2[2],
            ]);
            let cos_normal_angle = half_angle_rad.cos();
            let sin_normal_angle = half_angle_rad.sin();
            let normal = normalize(&[
                radial_dir[0] * cos_normal_angle - axis_normalized[0] * sin_normal_angle,
                radial_dir[1] * cos_normal_angle - axis_normalized[1] * sin_normal_angle,
                radial_dir[2] * cos_normal_angle - axis_normalized[2] * sin_normal_angle,
            ]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for h in 0..height_subdivisions {
        for a in 0..angle_subdivisions {
            let a_next = (a + 1) % angle_subdivisions;
            
            let v0 = base_idx + h * angle_subdivisions + a;
            let v1 = base_idx + h * angle_subdivisions + a_next;
            let v2 = base_idx + (h + 1) * angle_subdivisions + a;
            let v3 = base_idx + (h + 1) * angle_subdivisions + a_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a conical face with deflection control
fn triangulate_conical_face_deflection(
    face: &Face,
    origin: &[f64; 3],
    axis: &[f64; 3],
    half_angle_rad: f64,
    deflection: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    let axis_normalized = normalize(axis);
    
    // Create perpendicular vectors
    let perp1 = perpendicular_to(&axis_normalized);
    let perp2 = cross(&axis_normalized, &perp1);
    
    // Get the height range from the wire
    let outer_points = wire_to_points(&face.outer_wire);
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    
    for point in &outer_points {
        let z = dot(&[point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]], &axis_normalized);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    
    // Calculate radii at min and max height
    let radius_at_min = (min_z.abs()) * half_angle_rad.tan();
    let radius_at_max = (max_z.abs()) * half_angle_rad.tan();
    let max_radius = radius_at_min.max(radius_at_max);
    
    // Use deflection for angle subdivisions
    let sagitta = deflection.min(max_radius);
    let cos_half_angle = if max_radius > 0.0 {
        (1.0 - sagitta / max_radius).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let half_angle = cos_half_angle.acos();
    let angle_per_edge = 2.0 * half_angle;
    
    let two_pi = 2.0 * std::f64::consts::PI;
    let angle_subdivisions = (two_pi / angle_per_edge).ceil() as usize;
    let angle_subdivisions = angle_subdivisions.max(3);
    
    let height_subdivisions = ((max_z - min_z) / deflection).ceil() as usize + 1;
    let height_subdivisions = height_subdivisions.max(2);
    
    let base_idx = vertices.len();
    
    // Create vertex grid
    for h in 0..=height_subdivisions {
        let z_param = min_z + (h as f64) / (height_subdivisions as f64) * (max_z - min_z);
        let radius = z_param.abs() * half_angle_rad.tan();
        let z_offset = scale_vec(&axis_normalized, z_param);
        
        for a in 0..angle_subdivisions {
            let angle = (a as f64) / (angle_subdivisions as f64) * two_pi;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            
            let x = origin[0] + z_offset[0] + (cos_a * perp1[0] + sin_a * perp2[0]) * radius;
            let y = origin[1] + z_offset[1] + (cos_a * perp1[1] + sin_a * perp2[1]) * radius;
            let z = origin[2] + z_offset[2] + (cos_a * perp1[2] + sin_a * perp2[2]) * radius;
            
            vertices.push([x, y, z]);
            
            // Normal for cone: perpendicular to cone surface
            let radial_dir = normalize(&[
                cos_a * perp1[0] + sin_a * perp2[0],
                cos_a * perp1[1] + sin_a * perp2[1],
                cos_a * perp1[2] + sin_a * perp2[2],
            ]);
            let cos_normal_angle = half_angle_rad.cos();
            let sin_normal_angle = half_angle_rad.sin();
            let normal = normalize(&[
                radial_dir[0] * cos_normal_angle - axis_normalized[0] * sin_normal_angle,
                radial_dir[1] * cos_normal_angle - axis_normalized[1] * sin_normal_angle,
                radial_dir[2] * cos_normal_angle - axis_normalized[2] * sin_normal_angle,
            ]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for h in 0..height_subdivisions {
        for a in 0..angle_subdivisions {
            let a_next = (a + 1) % angle_subdivisions;
            
            let v0 = base_idx + h * angle_subdivisions + a;
            let v1 = base_idx + h * angle_subdivisions + a_next;
            let v2 = base_idx + (h + 1) * angle_subdivisions + a;
            let v3 = base_idx + (h + 1) * angle_subdivisions + a_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a toroidal face
fn triangulate_toroidal_face(
    _face: &Face,
    center: &[f64; 3],
    major_radius: f64,
    minor_radius: f64,
    tolerance: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    // Estimate subdivisions based on tolerance
    let major_circumference = 2.0 * std::f64::consts::PI * major_radius;
    let minor_circumference = 2.0 * std::f64::consts::PI * minor_radius;
    
    let major_subdivisions = ((major_circumference / tolerance).ceil() as usize).max(4);
    let minor_subdivisions = ((minor_circumference / tolerance).ceil() as usize).max(4);
    
    let base_idx = vertices.len();
    
    // Generate torus surface by sweeping a circle (minor) around another circle (major)
    // u: angle around major circle (0 to 2π)
    // v: angle around minor circle (0 to 2π)
    for u in 0..=major_subdivisions {
        let u_angle = (u as f64) / (major_subdivisions as f64) * 2.0 * std::f64::consts::PI;
        let u_cos = u_angle.cos();
        let u_sin = u_angle.sin();
        
        for v in 0..=minor_subdivisions {
            let v_angle = (v as f64) / (minor_subdivisions as f64) * 2.0 * std::f64::consts::PI;
            let v_cos = v_angle.cos();
            let v_sin = v_angle.sin();
            
            // Position on torus: major circle offset + minor circle radius in that direction
            let x = center[0] + (major_radius + minor_radius * v_cos) * u_cos;
            let y = center[1] + (major_radius + minor_radius * v_cos) * u_sin;
            let z = center[2] + minor_radius * v_sin;
            
            vertices.push([x, y, z]);
            
            // Normal points outward from the torus surface
            // It's the direction from the major circle to the point
            let normal = normalize(&[
                (major_radius + minor_radius * v_cos) * u_cos,
                (major_radius + minor_radius * v_cos) * u_sin,
                minor_radius * v_sin,
            ]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for u in 0..major_subdivisions {
        for v in 0..minor_subdivisions {
            let v_next = v + 1;
            
            let v0 = base_idx + u * (minor_subdivisions + 1) + v;
            let v1 = base_idx + u * (minor_subdivisions + 1) + v_next;
            let v2 = base_idx + (u + 1) * (minor_subdivisions + 1) + v;
            let v3 = base_idx + (u + 1) * (minor_subdivisions + 1) + v_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a toroidal face with deflection control
fn triangulate_toroidal_face_deflection(
    _face: &Face,
    center: &[f64; 3],
    major_radius: f64,
    minor_radius: f64,
    deflection: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    // Calculate subdivisions based on deflection
    // For torus, we need to handle both major and minor circle deflections
    
    // Major circle: from radius major_radius
    let major_sagitta = deflection.min(major_radius);
    let major_cos_half = (1.0 - major_sagitta / major_radius).clamp(-1.0, 1.0);
    let major_half_angle = major_cos_half.acos();
    let major_angle_per_edge = 2.0 * major_half_angle;
    
    // Minor circle: from radius minor_radius
    let minor_sagitta = deflection.min(minor_radius);
    let minor_cos_half = (1.0 - minor_sagitta / minor_radius).clamp(-1.0, 1.0);
    let minor_half_angle = minor_cos_half.acos();
    let minor_angle_per_edge = 2.0 * minor_half_angle;
    
    let two_pi = 2.0 * std::f64::consts::PI;
    let major_subdivisions = (two_pi / major_angle_per_edge).ceil() as usize;
    let major_subdivisions = major_subdivisions.max(3);
    
    let minor_subdivisions = (two_pi / minor_angle_per_edge).ceil() as usize;
    let minor_subdivisions = minor_subdivisions.max(3);
    
    let base_idx = vertices.len();
    
    // Generate torus surface by sweeping a circle (minor) around another circle (major)
    for u in 0..=major_subdivisions {
        let u_angle = (u as f64) / (major_subdivisions as f64) * two_pi;
        let u_cos = u_angle.cos();
        let u_sin = u_angle.sin();
        
        for v in 0..=minor_subdivisions {
            let v_angle = (v as f64) / (minor_subdivisions as f64) * two_pi;
            let v_cos = v_angle.cos();
            let v_sin = v_angle.sin();
            
            // Position on torus
            let x = center[0] + (major_radius + minor_radius * v_cos) * u_cos;
            let y = center[1] + (major_radius + minor_radius * v_cos) * u_sin;
            let z = center[2] + minor_radius * v_sin;
            
            vertices.push([x, y, z]);
            
            // Normal points outward from the torus surface
            let normal = normalize(&[
                (major_radius + minor_radius * v_cos) * u_cos,
                (major_radius + minor_radius * v_cos) * u_sin,
                minor_radius * v_sin,
            ]);
            normals.push(normal);
        }
    }
    
    // Create triangles
    for u in 0..major_subdivisions {
        for v in 0..minor_subdivisions {
            let v_next = v + 1;
            
            let v0 = base_idx + u * (minor_subdivisions + 1) + v;
            let v1 = base_idx + u * (minor_subdivisions + 1) + v_next;
            let v2 = base_idx + (u + 1) * (minor_subdivisions + 1) + v;
            let v3 = base_idx + (u + 1) * (minor_subdivisions + 1) + v_next;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a BSpline surface face
fn triangulate_bspline_face(
    _face: &Face,
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    control_points: &[Vec<[f64; 3]>],
    weights: Option<&Vec<Vec<f64>>>,
    tolerance: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    // Get knot span ranges
    let u_min = *u_knots.first().unwrap_or(&0.0);
    let u_max = *u_knots.last().unwrap_or(&1.0);
    let v_min = *v_knots.first().unwrap_or(&0.0);
    let v_max = *v_knots.last().unwrap_or(&1.0);
    
    // Estimate subdivisions based on control point density and tolerance
    // Use at least 4 subdivisions per direction, more if needed for accuracy
    let u_subdivisions = ((control_points.len() as f64).sqrt().ceil() as usize * 2).max(4);
    let v_subdivisions = if control_points.is_empty() {
        4
    } else {
        ((control_points[0].len() as f64).sqrt().ceil() as usize * 2).max(4)
    };
    
    let base_idx = vertices.len();
    
    // Create vertex grid
    for u_idx in 0..=u_subdivisions {
        let u = u_min + (u_idx as f64) / (u_subdivisions as f64) * (u_max - u_min);
        
        for v_idx in 0..=v_subdivisions {
            let v = v_min + (v_idx as f64) / (v_subdivisions as f64) * (v_max - v_min);
            
            // Evaluate point on surface
            let pt = crate::brep::SurfaceType::BSpline {
                u_degree,
                v_degree,
                u_knots: u_knots.to_vec(),
                v_knots: v_knots.to_vec(),
                control_points: control_points.to_vec(),
                weights: weights.map(|w| w.to_vec()),
            }
            .point_at(u, v);
            
            vertices.push(pt);
            
            // Evaluate normal on surface
            let normal = crate::brep::SurfaceType::BSpline {
                u_degree,
                v_degree,
                u_knots: u_knots.to_vec(),
                v_knots: v_knots.to_vec(),
                control_points: control_points.to_vec(),
                weights: weights.map(|w| w.to_vec()),
            }
            .normal_at(u, v);
            
            normals.push(normal);
        }
    }
    
    // Create triangles by connecting grid points
    for u_idx in 0..u_subdivisions {
        for v_idx in 0..v_subdivisions {
            let v0 = base_idx + u_idx * (v_subdivisions + 1) + v_idx;
            let v1 = base_idx + u_idx * (v_subdivisions + 1) + v_idx + 1;
            let v2 = base_idx + (u_idx + 1) * (v_subdivisions + 1) + v_idx;
            let v3 = base_idx + (u_idx + 1) * (v_subdivisions + 1) + v_idx + 1;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Triangulate a parametric surface (uses surface_type.point_at/normal_at)
/// Works for SurfaceOfRevolution and any surface with proper parametric evaluation
fn triangulate_parametric_surface(
    face: &Face,
    _tolerance: f64,
    vertices: &mut Vec<[f64; 3]>,
    normals: &mut Vec<[f64; 3]>,
    triangles: &mut Vec<[usize; 3]>,
) -> Result<()> {
    // Use fixed grid subdivision for parametric surfaces
    // u ∈ [0, 1] for curve parameter
    // v ∈ [0, 2π] for rotation angle (for SurfaceOfRevolution)
    let u_subdivisions = 16;
    let v_subdivisions = 32;
    
    let u_min = 0.0;
    let u_max = 1.0;
    let v_min = 0.0;
    let v_max = 2.0 * std::f64::consts::PI;
    
    let base_idx = vertices.len();
    
    // Create vertex grid
    for u_idx in 0..=u_subdivisions {
        let u = u_min + (u_idx as f64) / (u_subdivisions as f64) * (u_max - u_min);
        
        for v_idx in 0..=v_subdivisions {
            let v = v_min + (v_idx as f64) / (v_subdivisions as f64) * (v_max - v_min);
            
            // Evaluate point and normal on surface
            let pt = face.surface_type.point_at(u, v);
            let normal = face.surface_type.normal_at(u, v);
            
            vertices.push(pt);
            normals.push(normal);
        }
    }
    
    // Create triangles by connecting grid points
    for u_idx in 0..u_subdivisions {
        for v_idx in 0..v_subdivisions {
            let v0 = base_idx + u_idx * (v_subdivisions + 1) + v_idx;
            let v1 = base_idx + u_idx * (v_subdivisions + 1) + v_idx + 1;
            let v2 = base_idx + (u_idx + 1) * (v_subdivisions + 1) + v_idx;
            let v3 = base_idx + (u_idx + 1) * (v_subdivisions + 1) + v_idx + 1;
            
            triangles.push([v0, v1, v2]);
            triangles.push([v1, v3, v2]);
        }
    }
    
    Ok(())
}

/// Export mesh to STL format (ASCII)
pub fn export_stl(mesh: &TriangleMesh, path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    
    // Write ASCII STL header
    writeln!(file, "solid mesh")?;
    
    // Write each triangle
    for triangle in &mesh.triangles {
        let [i0, i1, i2] = triangle;
        let v0 = mesh.vertices[*i0];
        let v1 = mesh.vertices[*i1];
        let v2 = mesh.vertices[*i2];
        
        // Calculate normal from vertices if not using precomputed normals
        let normal = calculate_triangle_normal(&v0, &v1, &v2);
        
        writeln!(file, "  facet normal {} {} {}", normal[0], normal[1], normal[2])?;
        writeln!(file, "    outer loop")?;
        writeln!(file, "      vertex {} {} {}", v0[0], v0[1], v0[2])?;
        writeln!(file, "      vertex {} {} {}", v1[0], v1[1], v1[2])?;
        writeln!(file, "      vertex {} {} {}", v2[0], v2[1], v2[2])?;
        writeln!(file, "    endloop")?;
        writeln!(file, "  endfacet")?;
    }
    
    // Write footer
    writeln!(file, "endsolid mesh")?;
    
    Ok(())
}

/// Export mesh to STL format (binary)
///
/// Binary STL format:
/// - 80-byte header (arbitrary text, we use "cascade-rs binary STL")
/// - 4-byte little-endian u32: number of triangles
/// - For each triangle (50 bytes):
///   - Normal: 3 × f32 little-endian (12 bytes)
///   - Vertex 1: 3 × f32 little-endian (12 bytes)
///   - Vertex 2: 3 × f32 little-endian (12 bytes)
///   - Vertex 3: 3 × f32 little-endian (12 bytes)
///   - Attribute byte count: u16 little-endian (2 bytes, usually 0)
pub fn write_stl_binary(mesh: &TriangleMesh, path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    
    // Write 80-byte header
    let header = b"cascade-rs binary STL export";
    let mut header_buf = [0u8; 80];
    let copy_len = header.len().min(80);
    header_buf[..copy_len].copy_from_slice(&header[..copy_len]);
    file.write_all(&header_buf)?;
    
    // Write triangle count (4 bytes, little-endian u32)
    let triangle_count = mesh.triangles.len() as u32;
    file.write_all(&triangle_count.to_le_bytes())?;
    
    // Write each triangle (50 bytes each)
    for triangle in &mesh.triangles {
        let [i0, i1, i2] = triangle;
        let v0 = mesh.vertices[*i0];
        let v1 = mesh.vertices[*i1];
        let v2 = mesh.vertices[*i2];
        
        // Calculate normal from vertices
        let normal = calculate_triangle_normal(&v0, &v1, &v2);
        
        // Write normal (3 × f32)
        file.write_all(&(normal[0] as f32).to_le_bytes())?;
        file.write_all(&(normal[1] as f32).to_le_bytes())?;
        file.write_all(&(normal[2] as f32).to_le_bytes())?;
        
        // Write vertex 1 (3 × f32)
        file.write_all(&(v0[0] as f32).to_le_bytes())?;
        file.write_all(&(v0[1] as f32).to_le_bytes())?;
        file.write_all(&(v0[2] as f32).to_le_bytes())?;
        
        // Write vertex 2 (3 × f32)
        file.write_all(&(v1[0] as f32).to_le_bytes())?;
        file.write_all(&(v1[1] as f32).to_le_bytes())?;
        file.write_all(&(v1[2] as f32).to_le_bytes())?;
        
        // Write vertex 3 (3 × f32)
        file.write_all(&(v2[0] as f32).to_le_bytes())?;
        file.write_all(&(v2[1] as f32).to_le_bytes())?;
        file.write_all(&(v2[2] as f32).to_le_bytes())?;
        
        // Write attribute byte count (u16, typically 0)
        file.write_all(&0u16.to_le_bytes())?;
    }
    
    Ok(())
}

// ===== Helper Functions =====

/// Estimate an appropriate deflection value from a max angle and control points
/// This is used for parametric surfaces where radius of curvature is not easily available
fn estimate_deflection_from_angle(max_angle: f64, control_points: &[[f64; 3]]) -> f64 {
    if control_points.is_empty() {
        return max_angle * 0.1;
    }
    
    // Estimate the extent of the surface from control points
    let mut min = control_points[0];
    let mut max = control_points[0];
    
    for point in control_points {
        for i in 0..3 {
            if point[i] < min[i] {
                min[i] = point[i];
            }
            if point[i] > max[i] {
                max[i] = point[i];
            }
        }
    }
    
    // Estimate the typical size of the surface
    let size_x = (max[0] - min[0]).abs();
    let size_y = (max[1] - min[1]).abs();
    let size_z = (max[2] - min[2]).abs();
    let typical_size = (size_x * size_x + size_y * size_y + size_z * size_z).sqrt();
    
    // Estimate radius from typical size
    let estimated_radius = typical_size.max(1.0);
    
    // Convert angle to deflection
    estimated_radius * (1.0 - (max_angle / 2.0).cos())
}

/// Normalize a 3D vector
fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Scale a vector
fn scale_vec(v: &[f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Dot product
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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

/// Extract points from a wire
fn wire_to_points(wire: &Wire) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for edge in &wire.edges {
        points.push(edge.start.point);
    }
    points
}

/// Calculate normal of a triangle given 3 vertices
fn calculate_triangle_normal(v0: &[f64; 3], v1: &[f64; 3], v2: &[f64; 3]) -> [f64; 3] {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    normalize(&cross(&e1, &e2))
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_mesh_create() {
        let mesh = PolygonMesh::new();
        assert_eq!(mesh.vertices.len(), 0);
        assert_eq!(mesh.normals.len(), 0);
        assert_eq!(mesh.faces.len(), 0);
    }

    #[test]
    fn test_polygon_mesh_add_vertices() {
        let mut mesh = PolygonMesh::new();
        
        let idx0 = mesh.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let idx1 = mesh.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let idx2 = mesh.add_vertex([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let idx3 = mesh.add_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(idx3, 3);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.normals.len(), 4);
    }

    #[test]
    fn test_polygon_mesh_add_triangle_face() {
        let mut mesh = PolygonMesh::new();
        
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v2 = mesh.add_vertex([0.5, 1.0, 0.0], [0.0, 0.0, 1.0]);
        
        mesh.add_face(&[v0, v1, v2]);
        
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.faces[0].len(), 3);
    }

    #[test]
    fn test_polygon_mesh_add_quad_face() {
        let mut mesh = PolygonMesh::new();
        
        // Create a unit square: (0,0) -> (1,0) -> (1,1) -> (0,1)
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v2 = mesh.add_vertex([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let v3 = mesh.add_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        
        mesh.add_face(&[v0, v1, v2, v3]);
        
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.faces[0].len(), 4);
        assert_eq!(mesh.vertices.len(), 4);
    }

    #[test]
    fn test_polygon_mesh_add_pentagon_face() {
        let mut mesh = PolygonMesh::new();
        
        // Create a pentagon (5-gon) in the XY plane
        let mut indices = Vec::new();
        for i in 0..5 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 5.0;
            let x = angle.cos();
            let y = angle.sin();
            indices.push(mesh.add_vertex([x, y, 0.0], [0.0, 0.0, 1.0]));
        }
        
        mesh.add_face(&indices);
        
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.faces[0].len(), 5);
        assert_eq!(mesh.vertices.len(), 5);
    }

    #[test]
    fn test_polygon_mesh_triangle_to_triangle_conversion() {
        let mut mesh = PolygonMesh::new();
        
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v2 = mesh.add_vertex([0.5, 1.0, 0.0], [0.0, 0.0, 1.0]);
        
        mesh.add_face(&[v0, v1, v2]);
        
        let tri_mesh = mesh.to_triangle_mesh();
        
        // A single triangle should produce exactly 1 triangle
        assert_eq!(tri_mesh.triangles.len(), 1);
        assert_eq!(tri_mesh.triangles[0], [v0, v1, v2]);
        assert_eq!(tri_mesh.vertices.len(), 3);
    }

    #[test]
    fn test_polygon_mesh_quad_to_triangle_conversion() {
        let mut mesh = PolygonMesh::new();
        
        // Unit square
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v2 = mesh.add_vertex([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let v3 = mesh.add_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        
        mesh.add_face(&[v0, v1, v2, v3]);
        
        let tri_mesh = mesh.to_triangle_mesh();
        
        // A quad should be triangulated into 2 triangles via fan triangulation
        // Triangles: (v0, v1, v2) and (v0, v2, v3)
        assert_eq!(tri_mesh.triangles.len(), 2);
        assert_eq!(tri_mesh.triangles[0], [v0, v1, v2]);
        assert_eq!(tri_mesh.triangles[1], [v0, v2, v3]);
        assert_eq!(tri_mesh.vertices.len(), 4);
    }

    #[test]
    fn test_polygon_mesh_pentagon_to_triangle_conversion() {
        let mut mesh = PolygonMesh::new();
        
        // Create a pentagon
        let mut indices = Vec::new();
        for i in 0..5 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 5.0;
            let x = angle.cos();
            let y = angle.sin();
            indices.push(mesh.add_vertex([x, y, 0.0], [0.0, 0.0, 1.0]));
        }
        
        mesh.add_face(&indices);
        
        let tri_mesh = mesh.to_triangle_mesh();
        
        // A pentagon (5 vertices) should be triangulated into (5 - 2) = 3 triangles
        assert_eq!(tri_mesh.triangles.len(), 3);
        assert_eq!(tri_mesh.vertices.len(), 5);
    }

    #[test]
    fn test_polygon_mesh_multiple_faces() {
        let mut mesh = PolygonMesh::new();
        
        // Create two quads
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v2 = mesh.add_vertex([1.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let v3 = mesh.add_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        
        let v4 = mesh.add_vertex([2.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v5 = mesh.add_vertex([3.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let v6 = mesh.add_vertex([3.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let v7 = mesh.add_vertex([2.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        
        mesh.add_face(&[v0, v1, v2, v3]);
        mesh.add_face(&[v4, v5, v6, v7]);
        
        assert_eq!(mesh.faces.len(), 2);
        
        let tri_mesh = mesh.to_triangle_mesh();
        
        // Two quads should produce 4 triangles total
        assert_eq!(tri_mesh.triangles.len(), 4);
        assert_eq!(tri_mesh.vertices.len(), 8);
    }

    #[test]
    fn test_polygon_mesh_mixed_faces() {
        let mut mesh = PolygonMesh::new();
        
        // Add triangle
        let t0 = mesh.add_vertex([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let t1 = mesh.add_vertex([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let t2 = mesh.add_vertex([0.5, 1.0, 0.0], [0.0, 0.0, 1.0]);
        mesh.add_face(&[t0, t1, t2]);
        
        // Add quad
        let q0 = mesh.add_vertex([2.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let q1 = mesh.add_vertex([3.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let q2 = mesh.add_vertex([3.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let q3 = mesh.add_vertex([2.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        mesh.add_face(&[q0, q1, q2, q3]);
        
        // Add pentagon
        let mut pent_indices = Vec::new();
        for i in 0..5 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 5.0;
            let x = 5.0 + angle.cos();
            let y = angle.sin();
            pent_indices.push(mesh.add_vertex([x, y, 0.0], [0.0, 0.0, 1.0]));
        }
        mesh.add_face(&pent_indices);
        
        assert_eq!(mesh.faces.len(), 3);
        
        let tri_mesh = mesh.to_triangle_mesh();
        
        // Triangle (1) + Quad (2) + Pentagon (3) = 6 triangles total
        assert_eq!(tri_mesh.triangles.len(), 6);
        assert_eq!(tri_mesh.vertices.len(), 12); // 3 + 4 + 5
    }

    #[test]
    fn test_polygon_mesh_normals_preserved() {
        let mut mesh = PolygonMesh::new();
        
        let normal = normalize(&[1.0, 1.0, 1.0]);
        
        let v0 = mesh.add_vertex([0.0, 0.0, 0.0], normal);
        let v1 = mesh.add_vertex([1.0, 0.0, 0.0], normal);
        let v2 = mesh.add_vertex([1.0, 1.0, 0.0], normal);
        let v3 = mesh.add_vertex([0.0, 1.0, 0.0], normal);
        
        mesh.add_face(&[v0, v1, v2, v3]);
        
        let tri_mesh = mesh.to_triangle_mesh();
        
        // Normals should be preserved and in same order as vertices
        assert_eq!(tri_mesh.normals.len(), 4);
        for &normal_val in &tri_mesh.normals {
            assert!((normal_val[0] - normal[0]).abs() < 1e-10);
            assert!((normal_val[1] - normal[1]).abs() < 1e-10);
            assert!((normal_val[2] - normal[2]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_polygon_mesh_default() {
        let mesh = PolygonMesh::default();
        assert_eq!(mesh.vertices.len(), 0);
        assert_eq!(mesh.normals.len(), 0);
        assert_eq!(mesh.faces.len(), 0);
    }

    // ===== IncrementalMesh Tests =====

    /// Test creating an incremental mesh builder
    #[test]
    fn test_incremental_mesh_new() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        assert_eq!(incremental.added_faces_count(), 0);
        assert!(incremental.total_faces() > 0);
    }

    /// Test invalid tolerance
    #[test]
    fn test_incremental_mesh_invalid_tolerance() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        
        let result = IncrementalMesh::new(&box_solid, -0.1);
        assert!(result.is_err(), "Should reject negative tolerance");
        
        let result = IncrementalMesh::new(&box_solid, 0.0);
        assert!(result.is_err(), "Should reject zero tolerance");
    }

    /// Test adding a single face
    #[test]
    fn test_incremental_mesh_add_single_face() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        incremental.add_face(0).expect("Failed to add face 0");
        assert_eq!(incremental.added_faces_count(), 1);
        assert!(incremental.contains_face(0));
        
        let mesh = incremental.build().expect("Failed to build mesh");
        assert!(mesh.triangles.len() > 0, "Single face should have triangles");
        assert!(mesh.vertices.len() > 0, "Single face should have vertices");
    }

    /// Test adding multiple faces incrementally
    #[test]
    fn test_incremental_mesh_add_multiple_faces() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        let total_faces = incremental.total_faces();
        assert_eq!(total_faces, 6, "Box should have 6 faces");
        
        // Add faces one by one
        let mut prev_triangles = 0;
        for i in 0..total_faces {
            incremental.add_face(i).expect("Failed to add face");
            let mesh = incremental.build().expect("Failed to build mesh");
            
            assert!(mesh.triangles.len() >= prev_triangles, "Adding faces should not decrease triangle count");
            prev_triangles = mesh.triangles.len();
            assert_eq!(incremental.added_faces_count(), i + 1);
        }
    }

    /// Test adding all faces matches full tessellation
    #[test]
    fn test_incremental_mesh_all_faces_match_tessellation() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        
        // Full tessellation
        let full_mesh = triangulate(&box_solid, 0.1).expect("Failed to tessellate");
        
        // Incremental tessellation adding all faces
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        for i in 0..incremental.total_faces() {
            incremental.add_face(i).expect("Failed to add face");
        }
        let incremental_mesh = incremental.build().expect("Failed to build incremental mesh");
        
        // Should have same number of triangles
        assert_eq!(
            incremental_mesh.triangles.len(),
            full_mesh.triangles.len(),
            "Incremental with all faces should match full tessellation"
        );
        
        // Should have same number of vertices
        assert_eq!(
            incremental_mesh.vertices.len(),
            full_mesh.vertices.len(),
            "Incremental vertices should match full tessellation"
        );
    }

    /// Test adding faces in different orders
    #[test]
    fn test_incremental_mesh_order_independence() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let total_faces = {
            let inc = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
            inc.total_faces()
        };
        
        // Add faces in order 0, 1, 2, ...
        let mut inc1 = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        for i in 0..total_faces {
            inc1.add_face(i).expect("Failed to add face");
        }
        let mesh1 = inc1.build().expect("Failed to build mesh1");
        
        // Add faces in reverse order
        let mut inc2 = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        for i in (0..total_faces).rev() {
            inc2.add_face(i).expect("Failed to add face");
        }
        let mesh2 = inc2.build().expect("Failed to build mesh2");
        
        // Both should have same triangle count
        assert_eq!(mesh1.triangles.len(), mesh2.triangles.len());
        assert_eq!(mesh1.vertices.len(), mesh2.vertices.len());
    }

    /// Test contains_face predicate
    #[test]
    fn test_incremental_mesh_contains_face() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        assert!(!incremental.contains_face(0));
        incremental.add_face(0).expect("Failed to add face");
        assert!(incremental.contains_face(0));
        
        assert!(!incremental.contains_face(1));
        incremental.add_face(1).expect("Failed to add face");
        assert!(incremental.contains_face(1));
    }

    /// Test add_faces method (batch adding)
    #[test]
    fn test_incremental_mesh_add_multiple_at_once() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        incremental.add_faces(&[0, 1, 2]).expect("Failed to add faces");
        assert_eq!(incremental.added_faces_count(), 3);
        assert!(incremental.contains_face(0));
        assert!(incremental.contains_face(1));
        assert!(incremental.contains_face(2));
        
        let mesh = incremental.build().expect("Failed to build mesh");
        assert!(mesh.triangles.len() > 0);
    }

    /// Test invalid face index
    #[test]
    fn test_incremental_mesh_invalid_face_index() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        let result = incremental.add_face(999);
        assert!(result.is_err(), "Should reject invalid face index");
    }

    /// Test clear method
    #[test]
    fn test_incremental_mesh_clear() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        incremental.add_face(0).expect("Failed to add face");
        assert_eq!(incremental.added_faces_count(), 1);
        
        incremental.clear();
        assert_eq!(incremental.added_faces_count(), 0);
        assert!(!incremental.contains_face(0));
        
        let mesh = incremental.build().expect("Failed to build mesh");
        assert_eq!(mesh.triangles.len(), 0, "Empty mesh should have no triangles");
    }

    /// Test sphere tessellation incrementally
    #[test]
    fn test_incremental_mesh_sphere() {
        let sphere = crate::primitive::make_sphere(5.0).expect("Failed to make sphere");
        
        // Sphere typically has 1 face for the parametric surface
        let mut incremental = IncrementalMesh::new(&sphere, 0.1).expect("Failed to create incremental mesh");
        let total = incremental.total_faces();
        
        incremental.add_face(0).expect("Failed to add face");
        let mesh = incremental.build().expect("Failed to build mesh");
        
        // Check mesh validity
        assert!(mesh.vertices.len() > 0);
        assert!(mesh.triangles.len() > 0);
        assert_eq!(mesh.normals.len(), mesh.vertices.len(), "Should have normal for each vertex");
        
        // Verify normals are normalized
        for &normal in &mesh.normals {
            let len = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "Normals should be unit vectors");
        }
    }

    /// Test cylinder with different deflection values
    #[test]
    fn test_incremental_mesh_cylinder_deflection() {
        let cylinder = crate::primitive::make_cylinder(5.0, 10.0).expect("Failed to make cylinder");
        
        // Coarse mesh
        let mut inc_coarse = IncrementalMesh::new(&cylinder, 0.5).expect("Failed to create incremental mesh");
        for i in 0..inc_coarse.total_faces() {
            inc_coarse.add_face(i).expect("Failed to add face");
        }
        let mesh_coarse = inc_coarse.build().expect("Failed to build coarse mesh");
        
        // Fine mesh
        let mut inc_fine = IncrementalMesh::new(&cylinder, 0.05).expect("Failed to create incremental mesh");
        for i in 0..inc_fine.total_faces() {
            inc_fine.add_face(i).expect("Failed to add face");
        }
        let mesh_fine = inc_fine.build().expect("Failed to build fine mesh");
        
        // Fine mesh should have more triangles
        assert!(
            mesh_fine.triangles.len() > mesh_coarse.triangles.len(),
            "Finer deflection should produce more triangles"
        );
    }

    /// Test partial mesh building for progressive rendering
    #[test]
    fn test_incremental_mesh_progressive_building() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let mut incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        let total = incremental.total_faces();
        let mut prev_mesh = None;
        
        for i in 0..total {
            incremental.add_face(i).expect("Failed to add face");
            let mesh = incremental.build().expect("Failed to build mesh");
            
            // Each iteration should have more or equal triangles
            if let Some(prev) = prev_mesh {
                assert!(mesh.triangles.len() >= prev);
            }
            prev_mesh = Some(mesh.triangles.len());
        }
    }

    /// Test that empty incremental mesh builds successfully
    #[test]
    fn test_incremental_mesh_empty_build() {
        let box_solid = crate::primitive::make_box(10.0, 10.0, 10.0).expect("Failed to make box");
        let incremental = IncrementalMesh::new(&box_solid, 0.1).expect("Failed to create incremental mesh");
        
        let mesh = incremental.build().expect("Failed to build empty mesh");
        assert_eq!(mesh.triangles.len(), 0);
        assert_eq!(mesh.vertices.len(), 0);
        assert_eq!(mesh.normals.len(), 0);
    }

    // ============== MeshDomain Tests ==============

    /// Test basic mesh domain creation
    #[test]
    fn test_mesh_domain_new() {
        let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (10, 10))
            .expect("Failed to create MeshDomain");
        
        assert_eq!(domain.u_range, (0.0, 1.0));
        assert_eq!(domain.v_range, (0.0, 1.0));
        assert_eq!(domain.subdivisions, (10, 10));
    }

    /// Test invalid u_range (u_min >= u_max)
    #[test]
    fn test_mesh_domain_invalid_u_range() {
        let result = MeshDomain::new((1.0, 0.0), (0.0, 1.0), (10, 10));
        assert!(result.is_err(), "Should reject invalid u_range");
    }

    /// Test invalid v_range (v_min >= v_max)
    #[test]
    fn test_mesh_domain_invalid_v_range() {
        let result = MeshDomain::new((0.0, 1.0), (1.0, 0.0), (10, 10));
        assert!(result.is_err(), "Should reject invalid v_range");
    }

    /// Test invalid subdivisions (zero)
    #[test]
    fn test_mesh_domain_zero_subdivisions() {
        let result1 = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (0, 10));
        assert!(result1.is_err(), "Should reject zero u_subdivisions");
        
        let result2 = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (10, 0));
        assert!(result2.is_err(), "Should reject zero v_subdivisions");
    }

    /// Test contains() method for points within domain
    #[test]
    fn test_mesh_domain_contains() {
        let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (10, 10))
            .expect("Failed to create MeshDomain");
        
        // Points inside domain
        assert!(domain.contains(0.0, 0.0));
        assert!(domain.contains(0.5, 0.5));
        assert!(domain.contains(1.0, 1.0));
        assert!(domain.contains(0.25, 0.75));
        
        // Points outside domain
        assert!(!domain.contains(-0.1, 0.5));
        assert!(!domain.contains(1.1, 0.5));
        assert!(!domain.contains(0.5, -0.1));
        assert!(!domain.contains(0.5, 1.1));
    }

    /// Test contains() with negative ranges
    #[test]
    fn test_mesh_domain_contains_negative_range() {
        let domain = MeshDomain::new((-1.0, 1.0), (-2.0, 2.0), (10, 10))
            .expect("Failed to create MeshDomain");
        
        assert!(domain.contains(0.0, 0.0));
        assert!(domain.contains(-0.5, -1.5));
        assert!(domain.contains(1.0, 2.0));
        assert!(!domain.contains(-1.1, 0.0));
        assert!(!domain.contains(1.1, 0.0));
    }

    /// Test sample_points() generates correct grid
    #[test]
    fn test_mesh_domain_sample_points() {
        let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (2, 2))
            .expect("Failed to create MeshDomain");
        
        let points = domain.sample_points();
        
        // Should have (subdivisions+1)^2 points
        assert_eq!(points.len(), 9, "Expected 3x3 = 9 points");
        
        // Check specific points
        assert!(points.contains(&(0.0, 0.0)));
        assert!(points.contains(&(0.5, 0.5)));
        assert!(points.contains(&(1.0, 1.0)));
        
        // All points should be in domain
        for &(u, v) in &points {
            assert!(domain.contains(u, v), "Point ({}, {}) should be in domain", u, v);
        }
    }

    /// Test sample_points() with different subdivisions
    #[test]
    fn test_mesh_domain_sample_points_grid_structure() {
        let domain = MeshDomain::new((0.0, 2.0), (0.0, 4.0), (4, 8))
            .expect("Failed to create MeshDomain");
        
        let points = domain.sample_points();
        
        // Should have (4+1) * (8+1) = 45 points
        assert_eq!(points.len(), 5 * 9, "Expected 5x9 = 45 points");
        
        // Verify corners
        assert!(points.contains(&(0.0, 0.0)));
        assert!(points.contains(&(2.0, 4.0)));
        
        // Verify some interior points
        assert!(points.contains(&(1.0, 2.0)));
    }

    /// Test sample_points() returns all points in domain
    #[test]
    fn test_mesh_domain_sample_points_bounds_check() {
        let domain = MeshDomain::new((0.5, 1.5), (1.0, 3.0), (3, 5))
            .expect("Failed to create MeshDomain");
        
        let points = domain.sample_points();
        
        // All points should be within domain bounds
        for &(u, v) in &points {
            assert!(u >= 0.5 && u <= 1.5, "u={} out of range [0.5, 1.5]", u);
            assert!(v >= 1.0 && v <= 3.0, "v={} out of range [1.0, 3.0]", v);
        }
        
        // Check that we have the expected grid structure
        // For 3 u-subdivisions and 5 v-subdivisions, we expect (3+1) * (5+1) = 24 points
        assert_eq!(points.len(), 4 * 6, "Should have 4x6 = 24 points");
        
        // Verify corners
        assert!(points.contains(&(0.5, 1.0)), "Should contain corner (0.5, 1.0)");
        assert!(points.contains(&(1.5, 3.0)), "Should contain corner (1.5, 3.0)");
    }

    /// Test triangulate_domain with a sphere
    #[test]
    fn test_triangulate_domain_sphere() {
        let surface = SurfaceType::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 5.0,
        };
        
        let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (8, 16))
            .expect("Failed to create MeshDomain");
        
        let mesh = triangulate_domain(&surface, &domain)
            .expect("Failed to triangulate domain");
        
        // Check mesh structure
        assert!(mesh.vertices.len() > 0, "Mesh should have vertices");
        assert!(mesh.triangles.len() > 0, "Mesh should have triangles");
        assert_eq!(mesh.normals.len(), mesh.vertices.len(), "Should have normal for each vertex");
        
        // Expected: (8+1) * (16+1) = 153 vertices
        assert_eq!(mesh.vertices.len(), 9 * 17, "Should have 9x17 = 153 vertices");
        
        // Expected: 8 * 16 * 2 = 256 triangles
        assert_eq!(mesh.triangles.len(), 8 * 16 * 2, "Should have 8x16x2 = 256 triangles");
    }

    /// Test triangulate_domain with a cylinder
    #[test]
    fn test_triangulate_domain_cylinder() {
        let surface = SurfaceType::Cylinder {
            origin: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            radius: 3.0,
        };
        
        let domain = MeshDomain::new((0.0, 1.0), (-5.0, 5.0), (6, 10))
            .expect("Failed to create MeshDomain");
        
        let mesh = triangulate_domain(&surface, &domain)
            .expect("Failed to triangulate domain");
        
        // Check mesh validity
        assert_eq!(mesh.vertices.len(), 7 * 11, "Should have 7x11 = 77 vertices");
        assert_eq!(mesh.triangles.len(), 6 * 10 * 2, "Should have 6x10x2 = 120 triangles");
        assert_eq!(mesh.normals.len(), mesh.vertices.len());
        
        // Verify normals are normalized
        for &normal in &mesh.normals {
            let len_sq = normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2];
            assert!((len_sq - 1.0).abs() < 1e-6, "Normals should be unit vectors");
        }
    }

    /// Test triangulate_domain with partial domain (subset of parameter space)
    #[test]
    fn test_triangulate_domain_partial() {
        let surface = SurfaceType::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 2.0,
        };
        
        // Only tessellate upper hemisphere
        let domain = MeshDomain::new((0.0, 1.0), (0.5, 1.0), (4, 4))
            .expect("Failed to create MeshDomain");
        
        let mesh = triangulate_domain(&surface, &domain)
            .expect("Failed to triangulate domain");
        
        // Check mesh
        assert_eq!(mesh.vertices.len(), 5 * 5, "Should have 5x5 = 25 vertices");
        assert_eq!(mesh.triangles.len(), 4 * 4 * 2, "Should have 4x4x2 = 32 triangles");
    }

    /// Test triangulate_domain triangle connectivity
    #[test]
    fn test_triangulate_domain_triangle_indices() {
        let surface = SurfaceType::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        let domain = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (2, 2))
            .expect("Failed to create MeshDomain");
        
        let mesh = triangulate_domain(&surface, &domain)
            .expect("Failed to triangulate domain");
        
        // Verify all triangle indices are valid
        for &triangle in &mesh.triangles {
            for &idx in &triangle {
                assert!(idx < mesh.vertices.len(), "Triangle index {} out of bounds", idx);
            }
        }
        
        // Verify connectivity: should form a grid
        // For 2x2 subdivisions, we expect a 3x3 vertex grid
        assert_eq!(mesh.vertices.len(), 9);
        
        // Each interior vertex should be used by 4 quads (8 triangles)
        // but that's complex to verify, so we just check basic connectivity
        let mut vertex_usage_count = vec![0; mesh.vertices.len()];
        for &triangle in &mesh.triangles {
            for &idx in &triangle {
                vertex_usage_count[idx] += 1;
            }
        }
        
        // No vertex should be unused
        for (i, &count) in vertex_usage_count.iter().enumerate() {
            assert!(count > 0, "Vertex {} is not referenced by any triangle", i);
        }
    }

    /// Test triangulate_domain with coarse vs fine resolution
    #[test]
    fn test_triangulate_domain_resolution_comparison() {
        let surface = SurfaceType::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        
        // Coarse mesh
        let domain_coarse = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (2, 2))
            .expect("Failed to create coarse domain");
        let mesh_coarse = triangulate_domain(&surface, &domain_coarse)
            .expect("Failed to triangulate coarse");
        
        // Fine mesh
        let domain_fine = MeshDomain::new((0.0, 1.0), (0.0, 1.0), (8, 8))
            .expect("Failed to create fine domain");
        let mesh_fine = triangulate_domain(&surface, &domain_fine)
            .expect("Failed to triangulate fine");
        
        // Fine mesh should have more triangles and vertices
        assert!(mesh_fine.vertices.len() > mesh_coarse.vertices.len());
        assert!(mesh_fine.triangles.len() > mesh_coarse.triangles.len());
    }
}
