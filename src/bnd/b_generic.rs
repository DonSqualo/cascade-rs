//! Generic bounding boxes with 2D and 3D variants for double and float precision

use crate::gp::{Pnt, Pnt2d, XYZ, XY};

/// Generic 2D bounding box with configurable precision
#[derive(Clone, Copy, Debug)]
pub struct BndB2d {
    center: [f64; 2],
    h_size: [f64; 2],
}

impl BndB2d {
    /// Creates an empty bounding box
    pub fn new() -> Self {
        BndB2d {
            center: [0.0, 0.0],
            h_size: [0.0, 0.0],
        }
    }

    /// Creates a bounding box from center and half size
    pub fn from_center_hsize(center: XY, h_size: XY) -> Self {
        BndB2d {
            center: [center.x(), center.y()],
            h_size: [h_size.x(), h_size.y()],
        }
    }

    /// Returns true if the box is void
    pub fn is_void(&self) -> bool {
        self.h_size[0] < 0.0 || self.h_size[1] < 0.0
    }

    /// Clears the box
    pub fn clear(&mut self) {
        self.h_size = [-1.0, -1.0];
    }

    /// Adds a point
    pub fn add_point(&mut self, p: XY) {
        let dx = (p.x() - self.center[0]).abs();
        let dy = (p.y() - self.center[1]).abs();
        
        if self.is_void() {
            self.center = [p.x(), p.y()];
            self.h_size = [0.0, 0.0];
        } else {
            if dx > self.h_size[0] || dy > self.h_size[1] {
                let nhx = (self.h_size[0] + dx) / 2.0;
                let nhy = (self.h_size[1] + dy) / 2.0;
                self.center = [
                    self.center[0] + (p.x() - self.center[0]).signum() * (nhx - self.h_size[0]),
                    self.center[1] + (p.y() - self.center[1]).signum() * (nhy - self.h_size[1]),
                ];
                self.h_size = [nhx, nhy];
            }
        }
    }

    /// Adds a Pnt2d
    pub fn add_pnt2d(&mut self, p: Pnt2d) {
        self.add_point(XY::new(p.x(), p.y()));
    }

    /// Adds another bounding box
    pub fn add_box(&mut self, other: &BndB2d) {
        if other.is_void() {
            return;
        }
        if self.is_void() {
            *self = *other;
            return;
        }
        
        let min1_x = self.center[0] - self.h_size[0];
        let max1_x = self.center[0] + self.h_size[0];
        let min1_y = self.center[1] - self.h_size[1];
        let max1_y = self.center[1] + self.h_size[1];
        
        let min2_x = other.center[0] - other.h_size[0];
        let max2_x = other.center[0] + other.h_size[0];
        let min2_y = other.center[1] - other.h_size[1];
        let max2_y = other.center[1] + other.h_size[1];
        
        let new_min_x = min1_x.min(min2_x);
        let new_max_x = max1_x.max(max2_x);
        let new_min_y = min1_y.min(min2_y);
        let new_max_y = max1_y.max(max2_y);
        
        self.center[0] = (new_min_x + new_max_x) / 2.0;
        self.center[1] = (new_min_y + new_max_y) / 2.0;
        self.h_size[0] = (new_max_x - new_min_x) / 2.0;
        self.h_size[1] = (new_max_y - new_min_y) / 2.0;
    }

    /// Returns the minimum corner
    pub fn corner_min(&self) -> XY {
        XY::new(self.center[0] - self.h_size[0], self.center[1] - self.h_size[1])
    }

    /// Returns the maximum corner
    pub fn corner_max(&self) -> XY {
        XY::new(self.center[0] + self.h_size[0], self.center[1] + self.h_size[1])
    }

    /// Returns the square extent
    pub fn square_extent(&self) -> f64 {
        let dx = 2.0 * self.h_size[0];
        let dy = 2.0 * self.h_size[1];
        dx * dx + dy * dy
    }

    /// Enlarges the box
    pub fn enlarge(&mut self, diff: f64) {
        if !self.is_void() {
            self.h_size[0] += diff;
            self.h_size[1] += diff;
        }
    }

    /// Checks if a point is outside
    pub fn is_out_point(&self, p: XY) -> bool {
        let dx = (p.x() - self.center[0]).abs();
        let dy = (p.y() - self.center[1]).abs();
        dx > self.h_size[0] || dy > self.h_size[1]
    }

    /// Checks if another box is outside
    pub fn is_out_box(&self, other: &BndB2d) -> bool {
        let dx = (self.center[0] - other.center[0]).abs();
        let dy = (self.center[1] - other.center[1]).abs();
        dx > self.h_size[0] + other.h_size[0] ||
        dy > self.h_size[1] + other.h_size[1]
    }

    /// Limits this box by another
    pub fn limit(&mut self, other: &BndB2d) -> bool {
        if self.is_out_box(other) {
            return false;
        }
        
        let min1_x = self.center[0] - self.h_size[0];
        let max1_x = self.center[0] + self.h_size[0];
        let min1_y = self.center[1] - self.h_size[1];
        let max1_y = self.center[1] + self.h_size[1];
        
        let min2_x = other.center[0] - other.h_size[0];
        let max2_x = other.center[0] + other.h_size[0];
        let min2_y = other.center[1] - other.h_size[1];
        let max2_y = other.center[1] + other.h_size[1];
        
        let new_min_x = min1_x.max(min2_x);
        let new_max_x = max1_x.min(max2_x);
        let new_min_y = min1_y.max(min2_y);
        let new_max_y = max1_y.min(max2_y);
        
        if new_min_x > new_max_x || new_min_y > new_max_y {
            return false;
        }
        
        self.center[0] = (new_min_x + new_max_x) / 2.0;
        self.center[1] = (new_min_y + new_max_y) / 2.0;
        self.h_size[0] = (new_max_x - new_min_x) / 2.0;
        self.h_size[1] = (new_max_y - new_min_y) / 2.0;
        
        true
    }

    /// Checks if contained in another box
    pub fn is_in(&self, other: &BndB2d) -> bool {
        let min1_x = self.center[0] - self.h_size[0];
        let max1_x = self.center[0] + self.h_size[0];
        let min1_y = self.center[1] - self.h_size[1];
        let max1_y = self.center[1] + self.h_size[1];
        
        let min2_x = other.center[0] - other.h_size[0];
        let max2_x = other.center[0] + other.h_size[0];
        let min2_y = other.center[1] - other.h_size[1];
        let max2_y = other.center[1] + other.h_size[1];
        
        min1_x >= min2_x && max1_x <= max2_x && min1_y >= min2_y && max1_y <= max2_y
    }

    /// Sets the center
    pub fn set_center(&mut self, center: XY) {
        self.center = [center.x(), center.y()];
    }

    /// Sets the half size
    pub fn set_hsize(&mut self, hsize: XY) {
        self.h_size = [hsize.x(), hsize.y()];
    }
}

impl Default for BndB2d {
    fn default() -> Self {
        Self::new()
    }
}

/// 2D Bounding Box with float precision
pub type BndB2f = BndB2d;

/// Generic 3D bounding box with configurable precision
#[derive(Clone, Copy, Debug)]
pub struct BndB3d {
    center: [f64; 3],
    h_size: [f64; 3],
}

impl BndB3d {
    /// Creates an empty bounding box
    pub fn new() -> Self {
        BndB3d {
            center: [0.0, 0.0, 0.0],
            h_size: [0.0, 0.0, 0.0],
        }
    }

    /// Creates a bounding box from center and half size
    pub fn from_center_hsize(center: XYZ, h_size: XYZ) -> Self {
        BndB3d {
            center: [center.x(), center.y(), center.z()],
            h_size: [h_size.x(), h_size.y(), h_size.z()],
        }
    }

    /// Returns true if the box is void
    pub fn is_void(&self) -> bool {
        self.h_size[0] < 0.0 || self.h_size[1] < 0.0 || self.h_size[2] < 0.0
    }

    /// Clears the box
    pub fn clear(&mut self) {
        self.h_size = [-1.0, -1.0, -1.0];
    }

    /// Adds a point
    pub fn add_point(&mut self, p: XYZ) {
        let dx = (p.x() - self.center[0]).abs();
        let dy = (p.y() - self.center[1]).abs();
        let dz = (p.z() - self.center[2]).abs();
        
        if self.is_void() {
            self.center = [p.x(), p.y(), p.z()];
            self.h_size = [0.0, 0.0, 0.0];
        } else {
            if dx > self.h_size[0] || dy > self.h_size[1] || dz > self.h_size[2] {
                let nhx = (self.h_size[0] + dx) / 2.0;
                let nhy = (self.h_size[1] + dy) / 2.0;
                let nhz = (self.h_size[2] + dz) / 2.0;
                self.center[0] += (p.x() - self.center[0]).signum() * (nhx - self.h_size[0]);
                self.center[1] += (p.y() - self.center[1]).signum() * (nhy - self.h_size[1]);
                self.center[2] += (p.z() - self.center[2]).signum() * (nhz - self.h_size[2]);
                self.h_size = [nhx, nhy, nhz];
            }
        }
    }

    /// Adds a Pnt
    pub fn add_pnt(&mut self, p: Pnt) {
        self.add_point(p.xyz());
    }

    /// Adds another bounding box
    pub fn add_box(&mut self, other: &BndB3d) {
        if other.is_void() {
            return;
        }
        if self.is_void() {
            *self = *other;
            return;
        }
        
        let min1_x = self.center[0] - self.h_size[0];
        let max1_x = self.center[0] + self.h_size[0];
        let min1_y = self.center[1] - self.h_size[1];
        let max1_y = self.center[1] + self.h_size[1];
        let min1_z = self.center[2] - self.h_size[2];
        let max1_z = self.center[2] + self.h_size[2];
        
        let min2_x = other.center[0] - other.h_size[0];
        let max2_x = other.center[0] + other.h_size[0];
        let min2_y = other.center[1] - other.h_size[1];
        let max2_y = other.center[1] + other.h_size[1];
        let min2_z = other.center[2] - other.h_size[2];
        let max2_z = other.center[2] + other.h_size[2];
        
        let new_min_x = min1_x.min(min2_x);
        let new_max_x = max1_x.max(max2_x);
        let new_min_y = min1_y.min(min2_y);
        let new_max_y = max1_y.max(max2_y);
        let new_min_z = min1_z.min(min2_z);
        let new_max_z = max1_z.max(max2_z);
        
        self.center[0] = (new_min_x + new_max_x) / 2.0;
        self.center[1] = (new_min_y + new_max_y) / 2.0;
        self.center[2] = (new_min_z + new_max_z) / 2.0;
        self.h_size[0] = (new_max_x - new_min_x) / 2.0;
        self.h_size[1] = (new_max_y - new_min_y) / 2.0;
        self.h_size[2] = (new_max_z - new_min_z) / 2.0;
    }

    /// Returns the minimum corner
    pub fn corner_min(&self) -> XYZ {
        XYZ::new(
            self.center[0] - self.h_size[0],
            self.center[1] - self.h_size[1],
            self.center[2] - self.h_size[2],
        )
    }

    /// Returns the maximum corner
    pub fn corner_max(&self) -> XYZ {
        XYZ::new(
            self.center[0] + self.h_size[0],
            self.center[1] + self.h_size[1],
            self.center[2] + self.h_size[2],
        )
    }

    /// Returns the square extent
    pub fn square_extent(&self) -> f64 {
        let dx = 2.0 * self.h_size[0];
        let dy = 2.0 * self.h_size[1];
        let dz = 2.0 * self.h_size[2];
        dx * dx + dy * dy + dz * dz
    }

    /// Enlarges the box
    pub fn enlarge(&mut self, diff: f64) {
        if !self.is_void() {
            self.h_size[0] += diff;
            self.h_size[1] += diff;
            self.h_size[2] += diff;
        }
    }

    /// Checks if a point is outside
    pub fn is_out_point(&self, p: XYZ) -> bool {
        let dx = (p.x() - self.center[0]).abs();
        let dy = (p.y() - self.center[1]).abs();
        let dz = (p.z() - self.center[2]).abs();
        dx > self.h_size[0] || dy > self.h_size[1] || dz > self.h_size[2]
    }

    /// Checks if another box is outside
    pub fn is_out_box(&self, other: &BndB3d) -> bool {
        let dx = (self.center[0] - other.center[0]).abs();
        let dy = (self.center[1] - other.center[1]).abs();
        let dz = (self.center[2] - other.center[2]).abs();
        dx > self.h_size[0] + other.h_size[0] ||
        dy > self.h_size[1] + other.h_size[1] ||
        dz > self.h_size[2] + other.h_size[2]
    }

    /// Limits this box by another
    pub fn limit(&mut self, other: &BndB3d) -> bool {
        if self.is_out_box(other) {
            return false;
        }
        
        let min1_x = self.center[0] - self.h_size[0];
        let max1_x = self.center[0] + self.h_size[0];
        let min1_y = self.center[1] - self.h_size[1];
        let max1_y = self.center[1] + self.h_size[1];
        let min1_z = self.center[2] - self.h_size[2];
        let max1_z = self.center[2] + self.h_size[2];
        
        let min2_x = other.center[0] - other.h_size[0];
        let max2_x = other.center[0] + other.h_size[0];
        let min2_y = other.center[1] - other.h_size[1];
        let max2_y = other.center[1] + other.h_size[1];
        let min2_z = other.center[2] - other.h_size[2];
        let max2_z = other.center[2] + other.h_size[2];
        
        let new_min_x = min1_x.max(min2_x);
        let new_max_x = max1_x.min(max2_x);
        let new_min_y = min1_y.max(min2_y);
        let new_max_y = max1_y.min(max2_y);
        let new_min_z = min1_z.max(min2_z);
        let new_max_z = max1_z.min(max2_z);
        
        if new_min_x > new_max_x || new_min_y > new_max_y || new_min_z > new_max_z {
            return false;
        }
        
        self.center[0] = (new_min_x + new_max_x) / 2.0;
        self.center[1] = (new_min_y + new_max_y) / 2.0;
        self.center[2] = (new_min_z + new_max_z) / 2.0;
        self.h_size[0] = (new_max_x - new_min_x) / 2.0;
        self.h_size[1] = (new_max_y - new_min_y) / 2.0;
        self.h_size[2] = (new_max_z - new_min_z) / 2.0;
        
        true
    }

    /// Checks if contained in another box
    pub fn is_in(&self, other: &BndB3d) -> bool {
        let min1_x = self.center[0] - self.h_size[0];
        let max1_x = self.center[0] + self.h_size[0];
        let min1_y = self.center[1] - self.h_size[1];
        let max1_y = self.center[1] + self.h_size[1];
        let min1_z = self.center[2] - self.h_size[2];
        let max1_z = self.center[2] + self.h_size[2];
        
        let min2_x = other.center[0] - other.h_size[0];
        let max2_x = other.center[0] + other.h_size[0];
        let min2_y = other.center[1] - other.h_size[1];
        let max2_y = other.center[1] + other.h_size[1];
        let min2_z = other.center[2] - other.h_size[2];
        let max2_z = other.center[2] + other.h_size[2];
        
        min1_x >= min2_x && max1_x <= max2_x &&
        min1_y >= min2_y && max1_y <= max2_y &&
        min1_z >= min2_z && max1_z <= max2_z
    }

    /// Sets the center
    pub fn set_center(&mut self, center: XYZ) {
        self.center = [center.x(), center.y(), center.z()];
    }

    /// Sets the half size
    pub fn set_hsize(&mut self, hsize: XYZ) {
        self.h_size = [hsize.x(), hsize.y(), hsize.z()];
    }
}

impl Default for BndB3d {
    fn default() -> Self {
        Self::new()
    }
}

/// 3D Bounding Box with float precision
pub type BndB3f = BndB3d;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bnd_b2d_default() {
        let b = BndB2d::new();
        assert!(b.is_void());
    }

    #[test]
    fn test_bnd_b3d_default() {
        let b = BndB3d::new();
        assert!(b.is_void());
    }
}
