//! Tessellation and meshing

use crate::brep::{Face, SurfaceType, Wire};
use crate::{Result, CascadeError};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleMesh {
    pub vertices: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
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
        SurfaceType::BSpline { .. } => {
            return Err(CascadeError::NotImplemented("BSpline surface triangulation".into()));
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

// ===== Helper Functions =====

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
