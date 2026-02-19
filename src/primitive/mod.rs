//! Primitive shape creation
//!
//! These functions create basic solid shapes that can be used
//! as building blocks for more complex geometry.

use crate::brep::{Solid, Shape};
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
    
    // TODO: Implement box creation
    // Should create:
    // - 8 vertices at corners
    // - 12 edges connecting vertices
    // - 6 faces (planes)
    // - 1 shell (closed)
    // - 1 solid
    
    Err(CascadeError::NotImplemented("primitive::box".into()))
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
