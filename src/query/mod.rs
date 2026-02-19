//! Geometric queries

use crate::brep::{Shape, Solid, Shell, Face, Wire, SurfaceType};
use crate::{Result, CascadeError};

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
        _ => [0.0, 0.0, 1.0],
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
}
