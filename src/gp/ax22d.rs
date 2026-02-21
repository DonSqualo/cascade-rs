//! 2D coordinate system (point + two orthogonal directions).
//!
//! Port of OCCT's gp_Ax22d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Ax22d.hxx

use crate::gp::{Pnt2d, Dir2d};

/// A 2D coordinate system defined by an origin and two orthogonal directions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ax22d {
    origin: Pnt2d,
    x_direction: Dir2d,
    y_direction: Dir2d,
}

impl Ax22d {
    /// Creates a coordinate system at origin with X and Y axes.
    #[inline]
    pub fn new() -> Self {
        Self {
            origin: Pnt2d::new(),
            x_direction: Dir2d::new(),
            y_direction: Dir2d::from_coords(0.0, 1.0),
        }
    }

    /// Creates a coordinate system from origin and X direction.
    /// Y direction is computed as perpendicular.
    #[inline]
    pub fn from_origin_x_direction(origin: Pnt2d, x_direction: Dir2d) -> Self {
        // Y is 90° counterclockwise from X
        let y_direction = Dir2d::from_coords(-x_direction.y(), x_direction.x());
        Self { origin, x_direction, y_direction }
    }

    /// Creates a coordinate system from origin and both directions.
    #[inline]
    pub fn from_origin_xy_directions(origin: Pnt2d, x_direction: Dir2d, y_direction: Dir2d) -> Self {
        Self { origin, x_direction, y_direction }
    }

    /// Returns the origin.
    #[inline]
    pub fn origin(&self) -> Pnt2d {
        self.origin
    }

    /// Sets the origin.
    #[inline]
    pub fn set_origin(&mut self, origin: Pnt2d) {
        self.origin = origin;
    }

    /// Returns the X direction.
    #[inline]
    pub fn x_direction(&self) -> Dir2d {
        self.x_direction
    }

    /// Returns the Y direction.
    #[inline]
    pub fn y_direction(&self) -> Dir2d {
        self.y_direction
    }

    /// Sets the X direction and recomputes Y as perpendicular.
    #[inline]
    pub fn set_x_direction(&mut self, x_direction: Dir2d) {
        self.x_direction = x_direction;
        self.y_direction = Dir2d::from_coords(-x_direction.y(), x_direction.x());
    }

    /// Sets the Y direction and recomputes X as perpendicular.
    #[inline]
    pub fn set_y_direction(&mut self, y_direction: Dir2d) {
        self.y_direction = y_direction;
        self.x_direction = Dir2d::from_coords(y_direction.y(), -y_direction.x());
    }
}

impl Default for Ax22d {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ax22d_new() {
        let ax = Ax22d::new();
        assert_eq!(ax.x_direction().x(), 1.0);
        assert_eq!(ax.y_direction().y(), 1.0);
    }

    #[test]
    fn test_ax22d_from_origin_x_direction() {
        let origin = Pnt2d::from_coords(1.0, 2.0);
        let x_dir = Dir2d::from_coords(3.0, 4.0);
        let ax = Ax22d::from_origin_x_direction(origin, x_dir);
        assert_eq!(ax.origin().x(), 1.0);
        
        // Check that Y is perpendicular to X
        let dot = ax.x_direction().dot(&ax.y_direction());
        assert!(dot.abs() < 1e-10);
    }

    #[test]
    fn test_ax22d_orthogonality() {
        let ax = Ax22d::new();
        let dot = ax.x_direction().dot(&ax.y_direction());
        assert!(dot.abs() < 1e-10);
    }
}
