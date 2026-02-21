//! 2D parabola.
//!
//! Port of OCCT's gp_Parab2d class.
//! Source: src/FoundationClasses/TKMath/gp/gp_Parab2d.hxx

use crate::gp::{Pnt2d, Dir2d, Ax22d};

/// A 2D parabola defined by vertex, focus direction, and focal distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Parab2d {
    position: Ax22d,
    focal_length: f64,
}

impl Parab2d {
    /// Creates a parabola with given vertex (apex), direction, and focal distance.
    #[inline]
    pub fn new(vertex: Pnt2d, direction: Dir2d, focal_length: f64) -> Self {
        Self {
            position: Ax22d::from_origin_x_direction(vertex, direction),
            focal_length: focal_length.abs(),
        }
    }

    /// Creates a parabola with axis.
    #[inline]
    pub fn from_axis22d(axis: Ax22d, focal_length: f64) -> Self {
        Self {
            position: axis,
            focal_length: focal_length.abs(),
        }
    }

    /// Returns the vertex (apex) of the parabola.
    #[inline]
    pub fn vertex(&self) -> Pnt2d {
        self.position.origin()
    }

    /// Returns the focal length (distance from vertex to focus).
    #[inline]
    pub fn focal_length(&self) -> f64 {
        self.focal_length
    }

    /// Returns the focus point.
    #[inline]
    pub fn focus(&self) -> Pnt2d {
        let dir = self.position.x_direction();
        Pnt2d::from_coords(
            self.vertex().x() + self.focal_length * dir.x(),
            self.vertex().y() + self.focal_length * dir.y(),
        )
    }

    /// Returns the directrix line.
    #[inline]
    pub fn directrix(&self) -> Pnt2d {
        let dir = self.position.x_direction();
        Pnt2d::from_coords(
            self.vertex().x() - self.focal_length * dir.x(),
            self.vertex().y() - self.focal_length * dir.y(),
        )
    }

    /// Sets the vertex.
    #[inline]
    pub fn set_vertex(&mut self, vertex: Pnt2d) {
        let mut pos = self.position;
        pos.set_origin(vertex);
        self.position = pos;
    }

    /// Sets the focal length.
    #[inline]
    pub fn set_focal_length(&mut self, focal_length: f64) {
        self.focal_length = focal_length.abs();
    }

    /// Returns position axis.
    #[inline]
    pub fn position(&self) -> Ax22d {
        self.position
    }

    /// Sets position axis.
    #[inline]
    pub fn set_position(&mut self, axis: Ax22d) {
        self.position = axis;
    }

    /// Computes the parameter (distance along the parabola from vertex).
    /// For a parabola y^2 = 4*p*x, parameter t gives point (t^2/(4*p), t).
    pub fn parameter(&self, point: Pnt2d) -> f64 {
        let v = point - self.vertex();
        let dir = self.position.x_direction();
        
        // Project point onto parabola axis
        let proj_x = v.xy().dot(&dir.xy());
        
        // Compute parameter using parabola equation y^2 = 4*p*x
        if proj_x < 0.0 {
            return 0.0;
        }
        (4.0 * self.focal_length * proj_x).sqrt()
    }

    /// Returns the directrix distance (2 * focal_length).
    #[inline]
    pub fn directrix_distance(&self) -> f64 {
        2.0 * self.focal_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parab2d_new() {
        let vertex = Pnt2d::from_coords(1.0, 2.0);
        let direction = Dir2d::new();
        let parabola = Parab2d::new(vertex, direction, 2.0);
        assert_eq!(parabola.focal_length(), 2.0);
        assert_eq!(parabola.vertex().x(), 1.0);
    }

    #[test]
    fn test_parab2d_focus() {
        let vertex = Pnt2d::from_coords(0.0, 0.0);
        let direction = Dir2d::new();
        let parabola = Parab2d::new(vertex, direction, 1.0);
        let focus = parabola.focus();
        assert_eq!(focus.x(), 1.0);
        assert_eq!(focus.y(), 0.0);
    }

    #[test]
    fn test_parab2d_directrix_distance() {
        let vertex = Pnt2d::new();
        let direction = Dir2d::new();
        let parabola = Parab2d::new(vertex, direction, 1.0);
        assert_eq!(parabola.directrix_distance(), 2.0);
    }
}
