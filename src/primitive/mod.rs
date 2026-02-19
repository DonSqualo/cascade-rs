//! Primitive shape creation
//!
//! These functions create basic solid shapes that can be used
//! as building blocks for more complex geometry.

use crate::brep::{Solid, Shape, Vertex, Edge, Wire, Face, Shell, CurveType, SurfaceType};
use crate::{Result, CascadeError};

/// Create a box (cuboid) solid
///
/// # Arguments
/// * `dx` - Size in X direction
/// * `dy` - Size in Y direction  
/// * `dz` - Size in Z direction
///
/// # Returns
/// A Solid representing a box centered at origin
pub fn make_box(dx: f64, dy: f64, dz: f64) -> Result<Solid> {
    if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Box dimensions must be positive".into()
        ));
    }
    
    // Create 8 vertices at the corners of the box
    // Bottom face (z=0)
    let v0 = Vertex::new(0.0, 0.0, 0.0);
    let v1 = Vertex::new(dx, 0.0, 0.0);
    let v2 = Vertex::new(dx, dy, 0.0);
    let v3 = Vertex::new(0.0, dy, 0.0);
    
    // Top face (z=dz)
    let v4 = Vertex::new(0.0, 0.0, dz);
    let v5 = Vertex::new(dx, 0.0, dz);
    let v6 = Vertex::new(dx, dy, dz);
    let v7 = Vertex::new(0.0, dy, dz);
    
    // Create 12 edges
    // Bottom face edges (z=0)
    let e0 = Edge { start: v0.clone(), end: v1.clone(), curve_type: CurveType::Line };
    let e1 = Edge { start: v1.clone(), end: v2.clone(), curve_type: CurveType::Line };
    let e2 = Edge { start: v2.clone(), end: v3.clone(), curve_type: CurveType::Line };
    let e3 = Edge { start: v3.clone(), end: v0.clone(), curve_type: CurveType::Line };
    
    // Top face edges (z=dz)
    let e4 = Edge { start: v4.clone(), end: v5.clone(), curve_type: CurveType::Line };
    let e5 = Edge { start: v5.clone(), end: v6.clone(), curve_type: CurveType::Line };
    let e6 = Edge { start: v6.clone(), end: v7.clone(), curve_type: CurveType::Line };
    let e7 = Edge { start: v7.clone(), end: v4.clone(), curve_type: CurveType::Line };
    
    // Vertical edges
    let e8 = Edge { start: v0.clone(), end: v4.clone(), curve_type: CurveType::Line };
    let e9 = Edge { start: v1.clone(), end: v5.clone(), curve_type: CurveType::Line };
    let e10 = Edge { start: v2.clone(), end: v6.clone(), curve_type: CurveType::Line };
    let e11 = Edge { start: v3.clone(), end: v7.clone(), curve_type: CurveType::Line };
    
    // Create 6 faces
    
    // Bottom face (z=0) - normal pointing down
    let bottom_wire = Wire {
        edges: vec![e0.clone(), e1.clone(), e2.clone(), e3.clone()],
        closed: true,
    };
    let bottom_face = Face {
        outer_wire: bottom_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
        },
    };
    
    // Top face (z=dz) - normal pointing up
    let top_wire = Wire {
        edges: vec![e4.clone(), e5.clone(), e6.clone(), e7.clone()],
        closed: true,
    };
    let top_face = Face {
        outer_wire: top_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, dz],
            normal: [0.0, 0.0, 1.0],
        },
    };
    
    // Front face (y=0) - normal pointing back (negative Y)
    let front_wire = Wire {
        edges: vec![e0.clone(), e9.clone(), e4.clone(), e8.clone()],
        closed: true,
    };
    let front_face = Face {
        outer_wire: front_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [0.0, -1.0, 0.0],
        },
    };
    
    // Back face (y=dy) - normal pointing forward (positive Y)
    let back_wire = Wire {
        edges: vec![e3.clone(), e7.clone(), e6.clone(), e2.clone()],
        closed: true,
    };
    let back_face = Face {
        outer_wire: back_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, dy, 0.0],
            normal: [0.0, 1.0, 0.0],
        },
    };
    
    // Left face (x=0) - normal pointing left (negative X)
    let left_wire = Wire {
        edges: vec![e8.clone(), e7.clone(), e3.clone(), e11.clone()],
        closed: true,
    };
    let left_face = Face {
        outer_wire: left_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [0.0, 0.0, 0.0],
            normal: [-1.0, 0.0, 0.0],
        },
    };
    
    // Right face (x=dx) - normal pointing right (positive X)
    let right_wire = Wire {
        edges: vec![e1.clone(), e10.clone(), e5.clone(), e9.clone()],
        closed: true,
    };
    let right_face = Face {
        outer_wire: right_wire,
        inner_wires: vec![],
        surface_type: SurfaceType::Plane {
            origin: [dx, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
        },
    };
    
    // Create shell with all 6 faces
    let shell = Shell {
        faces: vec![bottom_face, top_face, front_face, back_face, left_face, right_face],
        closed: true,
    };
    
    // Create solid
    let solid = Solid {
        outer_shell: shell,
        inner_shells: vec![],
    };
    
    Ok(solid)
}

/// Create a sphere solid
///
/// # Arguments
/// * `radius` - Sphere radius
///
/// # Returns
/// A Solid representing a sphere centered at origin
pub fn make_sphere(radius: f64) -> Result<Solid> {
    if radius <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Sphere radius must be positive".into()
        ));
    }
    
    // TODO: Implement sphere creation
    // Should create a solid with spherical surface
    
    Err(CascadeError::NotImplemented("primitive::sphere".into()))
}

/// Create a cylinder solid
///
/// # Arguments
/// * `radius` - Cylinder radius
/// * `height` - Cylinder height
///
/// # Returns
/// A Solid representing a cylinder along Z axis, base at origin
pub fn make_cylinder(radius: f64, height: f64) -> Result<Solid> {
    if radius <= 0.0 || height <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Cylinder dimensions must be positive".into()
        ));
    }
    
    // TODO: Implement cylinder creation
    
    Err(CascadeError::NotImplemented("primitive::cylinder".into()))
}

/// Create a cone solid
///
/// # Arguments
/// * `radius1` - Base radius
/// * `radius2` - Top radius (0 for pointed cone)
/// * `height` - Cone height
///
/// # Returns
/// A Solid representing a cone along Z axis
pub fn make_cone(radius1: f64, radius2: f64, height: f64) -> Result<Solid> {
    if radius1 < 0.0 || radius2 < 0.0 || height <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Cone dimensions must be non-negative (height positive)".into()
        ));
    }
    if radius1 == 0.0 && radius2 == 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "At least one cone radius must be positive".into()
        ));
    }
    
    // TODO: Implement cone creation
    
    Err(CascadeError::NotImplemented("primitive::cone".into()))
}

/// Create a torus solid
///
/// # Arguments
/// * `major_radius` - Distance from center to tube center
/// * `minor_radius` - Tube radius
///
/// # Returns
/// A Solid representing a torus centered at origin, in XY plane
pub fn make_torus(major_radius: f64, minor_radius: f64) -> Result<Solid> {
    if major_radius <= 0.0 || minor_radius <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Torus radii must be positive".into()
        ));
    }
    if minor_radius >= major_radius {
        return Err(CascadeError::InvalidGeometry(
            "Minor radius must be less than major radius".into()
        ));
    }
    
    // TODO: Implement torus creation
    
    Err(CascadeError::NotImplemented("primitive::torus".into()))
}

/// Create a wedge (prism) solid
///
/// # Arguments
/// * `dx` - Size in X
/// * `dy` - Size in Y
/// * `dz` - Size in Z
/// * `ltx` - Top X size (for tapering)
///
/// # Returns
/// A Solid representing a wedge
pub fn make_wedge(dx: f64, dy: f64, dz: f64, ltx: f64) -> Result<Solid> {
    if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
        return Err(CascadeError::InvalidGeometry(
            "Wedge dimensions must be positive".into()
        ));
    }
    
    // TODO: Implement wedge creation
    
    Err(CascadeError::NotImplemented("primitive::wedge".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_box_invalid_dimensions() {
        assert!(make_box(-1.0, 1.0, 1.0).is_err());
        assert!(make_box(0.0, 1.0, 1.0).is_err());
    }
    
    #[test]
    fn test_sphere_invalid_radius() {
        assert!(make_sphere(-1.0).is_err());
        assert!(make_sphere(0.0).is_err());
    }
}
