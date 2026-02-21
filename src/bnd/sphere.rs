//! Bnd_Sphere - Bounding Sphere

use crate::gp::XYZ;

/// Represents a bounding sphere
#[derive(Clone, Copy, Debug)]
pub struct BndSphere {
    center: XYZ,
    radius: f64,
    u: i32,
    v: i32,
    is_valid: bool,
}

impl Default for BndSphere {
    fn default() -> Self {
        Self::new()
    }
}

impl BndSphere {
    /// Creates an empty sphere
    pub fn new() -> Self {
        BndSphere {
            center: XYZ::new(0.0, 0.0, 0.0),
            radius: 0.0,
            u: 0,
            v: 0,
            is_valid: false,
        }
    }

    /// Creates a sphere from center and radius
    pub fn from_center_radius(center: XYZ, radius: f64, u: i32, v: i32) -> Self {
        BndSphere {
            center,
            radius,
            u,
            v,
            is_valid: false,
        }
    }

    /// Returns the center
    pub fn center(&self) -> XYZ {
        self.center
    }

    /// Returns the radius
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns U parameter
    pub fn u(&self) -> i32 {
        self.u
    }

    /// Returns V parameter
    pub fn v(&self) -> i32 {
        self.v
    }

    /// Returns true if sphere is valid
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    /// Sets validity
    pub fn set_valid(&mut self, valid: bool) {
        self.is_valid = valid;
    }

    /// Returns the distance to a point
    pub fn distance(&self, point: XYZ) -> f64 {
        let dx = self.center.x() - point.x();
        let dy = self.center.y() - point.y();
        let dz = self.center.z() - point.z();
        let dist_center = (dx * dx + dy * dy + dz * dz).sqrt();
        (dist_center - self.radius).abs()
    }

    /// Returns the square distance to a point
    pub fn square_distance(&self, point: XYZ) -> f64 {
        let dx = self.center.x() - point.x();
        let dy = self.center.y() - point.y();
        let dz = self.center.z() - point.z();
        let dist_center_sq = dx * dx + dy * dy + dz * dz;
        let dist = dist_center_sq.sqrt() - self.radius;
        dist * dist
    }

    /// Returns the square extent
    pub fn square_extent(&self) -> f64 {
        (2.0 * self.radius) * (2.0 * self.radius)
    }

    /// Adds another sphere, creating the enclosing sphere
    pub fn add(&mut self, other: &BndSphere) {
        let dx = other.center.x() - self.center.x();
        let dy = other.center.y() - self.center.y();
        let dz = other.center.z() - self.center.z();
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        
        let r1 = self.radius;
        let r2 = other.radius;
        
        if dist + r2 <= r1 {
            // other is inside self
            return;
        }
        if dist + r1 <= r2 {
            // self is inside other
            *self = *other;
            return;
        }
        
        // Compute the new radius and center
        let new_radius = (dist + r1 + r2) / 2.0;
        let alpha = (new_radius - r1) / dist;
        
        self.center = XYZ::new(
            self.center.x() + alpha * dx,
            self.center.y() + alpha * dy,
            self.center.z() + alpha * dz,
        );
        self.radius = new_radius;
    }

    /// Returns minimum and maximum distances to a point
    pub fn distances(&self, point: XYZ) -> (f64, f64) {
        let dx = point.x() - self.center.x();
        let dy = point.y() - self.center.y();
        let dz = point.z() - self.center.z();
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let min_dist = (dist - self.radius).max(0.0);
        let max_dist = dist + self.radius;
        (min_dist, max_dist)
    }

    /// Returns minimum and maximum square distances to a point
    pub fn square_distances(&self, point: XYZ) -> (f64, f64) {
        let dx = point.x() - self.center.x();
        let dy = point.y() - self.center.y();
        let dz = point.z() - self.center.z();
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let dist = dist_sq.sqrt();
        let r = self.radius;
        
        let min_dist_sq = if dist > r {
            (dist - r) * (dist - r)
        } else {
            0.0
        };
        let max_dist_sq = (dist + r) * (dist + r);
        (min_dist_sq, max_dist_sq)
    }

    /// Returns true if another sphere is outside this sphere
    pub fn is_out(&self, other: &BndSphere) -> bool {
        let dx = other.center.x() - self.center.x();
        let dy = other.center.y() - self.center.y();
        let dz = other.center.z() - self.center.z();
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        dist > self.radius + other.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constructor() {
        let s = BndSphere::new();
        assert!((s.center().x() - 0.0).abs() < f64::EPSILON);
        assert!((s.center().y() - 0.0).abs() < f64::EPSILON);
        assert!((s.center().z() - 0.0).abs() < f64::EPSILON);
        assert!((s.radius() - 0.0).abs() < f64::EPSILON);
        assert!(!s.is_valid());
        assert_eq!(s.u(), 0);
        assert_eq!(s.v(), 0);
    }
}
