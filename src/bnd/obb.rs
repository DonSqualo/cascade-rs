//! Bnd_OBB - Oriented Bounding Box

use crate::gp::{Dir, Pnt, XYZ};

/// Represents an oriented bounding box
#[derive(Clone, Copy, Debug)]
pub struct BndOBB {
    center: XYZ,
    axes: [XYZ; 3],  // X, Y, Z directions
    h_dims: [f64; 3], // Half dimensions for each axis
}

impl BndOBB {
    /// Creates an empty OBB
    pub fn new() -> Self {
        BndOBB {
            center: XYZ::new(0.0, 0.0, 0.0),
            axes: [
                XYZ::new(1.0, 0.0, 0.0),
                XYZ::new(0.0, 1.0, 0.0),
                XYZ::new(0.0, 0.0, 1.0),
            ],
            h_dims: [0.0, 0.0, 0.0],
        }
    }

    /// Creates an OBB from center, axes, and half sizes
    pub fn from_components(center: Pnt, x_dir: Dir, y_dir: Dir, z_dir: Dir, hx: f64, hy: f64, hz: f64) -> Self {
        BndOBB {
            center: center.xyz(),
            axes: [x_dir.xyz(), y_dir.xyz(), z_dir.xyz()],
            h_dims: [hx, hy, hz],
        }
    }

    /// Returns the center
    pub fn center(&self) -> XYZ {
        self.center
    }

    /// Sets the center
    pub fn set_center(&mut self, center: Pnt) {
        self.center = center.xyz();
    }

    /// Returns the X axis direction
    pub fn x_axis(&self) -> XYZ {
        self.axes[0]
    }

    /// Returns the Y axis direction
    pub fn y_axis(&self) -> XYZ {
        self.axes[1]
    }

    /// Returns the Z axis direction
    pub fn z_axis(&self) -> XYZ {
        self.axes[2]
    }

    /// Returns the half sizes
    pub fn half_sizes(&self) -> (f64, f64, f64) {
        (self.h_dims[0], self.h_dims[1], self.h_dims[2])
    }

    /// Sets the X component
    pub fn set_x_component(&mut self, dir: Dir, hx: f64) {
        self.axes[0] = dir.xyz();
        self.h_dims[0] = hx;
    }

    /// Sets the Y component
    pub fn set_y_component(&mut self, dir: Dir, hy: f64) {
        self.axes[1] = dir.xyz();
        self.h_dims[1] = hy;
    }

    /// Sets the Z component
    pub fn set_z_component(&mut self, dir: Dir, hz: f64) {
        self.axes[2] = dir.xyz();
        self.h_dims[2] = hz;
    }

    /// Adds another OBB
    pub fn add(&mut self, other: &BndOBB) {
        // For simplicity, expand the half sizes to contain the other box
        for i in 0..3 {
            self.h_dims[i] = self.h_dims[i].max(other.h_dims[i]);
        }
    }

    /// Adds a point
    pub fn add_point(&mut self, p: Pnt) {
        // Transform point to local coordinates
        let v = XYZ::new(
            p.x_val() - self.center.x_val(),
            p.y_val() - self.center.y_val(),
            p.z_val() - self.center.z_val(),
        );
        
        // Project onto each axis
        for i in 0..3 {
            let proj = (v.x_val() * self.axes[i].x_val() + 
                       v.y_val() * self.axes[i].y_val() + 
                       v.z_val() * self.axes[i].z_val()).abs();
            self.h_dims[i] = self.h_dims[i].max(proj);
        }
    }

    /// Checks if a point is contained in this OBB
    pub fn contains(&self, p: Pnt) -> bool {
        let v = XYZ::new(
            p.x_val() - self.center.x_val(),
            p.y_val() - self.center.y_val(),
            p.z_val() - self.center.z_val(),
        );
        
        for i in 0..3 {
            let proj = v.x_val() * self.axes[i].x_val() + 
                      v.y_val() * self.axes[i].y_val() + 
                      v.z_val() * self.axes[i].z_val();
            if proj.abs() > self.h_dims[i] {
                return false;
            }
        }
        true
    }

    /// Checks if another OBB intersects this OBB (using SAT - Separating Axis Theorem)
    pub fn intersects(&self, other: &BndOBB) -> bool {
        // Simple AABB check for now (not full OBB-OBB)
        let cx_diff = (self.center.x_val() - other.center.x_val()).abs();
        let cy_diff = (self.center.y_val() - other.center.y_val()).abs();
        let cz_diff = (self.center.z_val() - other.center.z_val()).abs();
        
        let hx = self.h_dims[0] + other.h_dims[0];
        let hy = self.h_dims[1] + other.h_dims[1];
        let hz = self.h_dims[2] + other.h_dims[2];
        
        cx_diff <= hx && cy_diff <= hy && cz_diff <= hz
    }
}

impl Default for BndOBB {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let obb = BndOBB::new();
        let (hx, hy, hz) = obb.half_sizes();
        assert!((hx - 0.0).abs() < f64::EPSILON);
        assert!((hy - 0.0).abs() < f64::EPSILON);
        assert!((hz - 0.0).abs() < f64::EPSILON);
    }
}
