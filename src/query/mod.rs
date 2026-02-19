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
    }
    
    // Normalize center of mass by volume
    if volume.abs() > 1e-10 {
        for i in 0..3 {
            center_of_mass[i] /= volume.abs();
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
        // V = 1/6 * sum(face_center · face_normal * face_area)
        let volume_contribution = (face_center[0] * face_normal[0] + 
                                  face_center[1] * face_normal[1] + 
                                  face_center[2] * face_normal[2]) * face_area / 6.0;
        *volume += volume_contribution;
        
        // Center of mass contribution (weighted by face area)
        for i in 0..3 {
            center_of_mass[i] += face_center[i] * face_area;
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
