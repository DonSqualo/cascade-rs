//! Bnd_Box - 3D Bounding Box
//! 
//! Describes a bounding box in 3D space. A bounding box is parallel to the axes
//! of the coordinate system. It may be infinite (open) in one or more directions.

use crate::gp::{Dir, Pnt, Trsf};

/// Represents a 3D bounding box with optional gaps and open directions
#[derive(Clone, Copy, Debug)]
pub struct BndBox {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
    gap: f64,
    flags: u8,
}

// Flag constants
const VOID_MASK: u8 = 0x80;      // Empty box
const WHOLE_MASK: u8 = 0x7F;     // All directions open
const XMIN_MASK: u8 = 0x01;      // Open Xmin
const XMAX_MASK: u8 = 0x02;      // Open Xmax
const YMIN_MASK: u8 = 0x04;      // Open Ymin
const YMAX_MASK: u8 = 0x08;      // Open Ymax
const ZMIN_MASK: u8 = 0x10;      // Open Zmin
const ZMAX_MASK: u8 = 0x20;      // Open Zmax

impl Default for BndBox {
    fn default() -> Self {
        Self::new()
    }
}

impl BndBox {
    /// Creates an empty bounding box (void)
    pub fn new() -> Self {
        BndBox {
            xmin: f64::MAX,
            xmax: f64::NEG_INFINITY,
            ymin: f64::MAX,
            ymax: f64::NEG_INFINITY,
            zmin: f64::MAX,
            zmax: f64::NEG_INFINITY,
            gap: 0.0,
            flags: VOID_MASK,
        }
    }

    /// Creates a bounding box from minimum and maximum points
    pub fn from_points(min: Pnt, max: Pnt) -> Self {
        BndBox {
            xmin: min.x(),
            xmax: max.x(),
            ymin: min.y(),
            ymax: max.y(),
            zmin: min.z(),
            zmax: max.z(),
            gap: 0.0,
            flags: 0,
        }
    }

    /// Returns true if this box is void (empty)
    pub fn is_void(&self) -> bool {
        (self.flags & VOID_MASK) != 0
    }

    /// Returns true if this box covers all of 3D space
    pub fn is_whole(&self) -> bool {
        (self.flags & WHOLE_MASK) == WHOLE_MASK
    }

    /// Sets this box as void (empty)
    pub fn set_void(&mut self) {
        self.xmin = f64::MAX;
        self.xmax = f64::NEG_INFINITY;
        self.ymin = f64::MAX;
        self.ymax = f64::NEG_INFINITY;
        self.zmin = f64::MAX;
        self.zmax = f64::NEG_INFINITY;
        self.gap = 0.0;
        self.flags = VOID_MASK;
    }

    /// Sets this box to cover all of 3D space
    pub fn set_whole(&mut self) {
        self.flags = WHOLE_MASK;
    }

    /// Sets this box to contain only the given point
    pub fn set_point(&mut self, p: Pnt) {
        self.set_void();
        self.add_point(p);
    }

    /// Sets this box to bound the half-line from point P in direction D
    pub fn set_half_line(&mut self, p: Pnt, d: Dir) {
        self.set_void();
        self.add_point(p);
        self.add_direction(d);
    }

    /// Updates this box to contain the given interval
    pub fn update_interval(&mut self, xmin: f64, ymin: f64, zmin: f64, xmax: f64, ymax: f64, zmax: f64) {
        if self.is_void() {
            self.xmin = xmin;
            self.xmax = xmax;
            self.ymin = ymin;
            self.ymax = ymax;
            self.zmin = zmin;
            self.zmax = zmax;
            self.flags = 0;
        } else {
            self.xmin = self.xmin.min(xmin);
            self.xmax = self.xmax.max(xmax);
            self.ymin = self.ymin.min(ymin);
            self.ymax = self.ymax.max(ymax);
            self.zmin = self.zmin.min(zmin);
            self.zmax = self.zmax.max(zmax);
        }
    }

    /// Updates this box to contain the given point
    pub fn update_point(&mut self, x: f64, y: f64, z: f64) {
        self.update_interval(x, y, z, x, y, z);
    }

    /// Returns the gap of this box
    pub fn gap(&self) -> f64 {
        self.gap
    }

    /// Sets the gap of this box
    pub fn set_gap(&mut self, tol: f64) {
        self.gap = tol.abs();
    }

    /// Enlarges the box gap
    pub fn enlarge(&mut self, tol: f64) {
        self.gap = self.gap.max(tol.abs());
    }

    /// Returns the bounds including gap: (xmin, ymin, zmin, xmax, ymax, zmax)
    pub fn get(&self) -> (f64, f64, f64, f64, f64, f64) {
        let xmin = if self.is_open_xmin() {
            INFINITY * -1.0
        } else {
            self.xmin - self.gap
        };
        let xmax = if self.is_open_xmax() {
            INFINITY
        } else {
            self.xmax + self.gap
        };
        let ymin = if self.is_open_ymin() {
            INFINITY * -1.0
        } else {
            self.ymin - self.gap
        };
        let ymax = if self.is_open_ymax() {
            INFINITY
        } else {
            self.ymax + self.gap
        };
        let zmin = if self.is_open_zmin() {
            INFINITY * -1.0
        } else {
            self.zmin - self.gap
        };
        let zmax = if self.is_open_zmax() {
            INFINITY
        } else {
            self.zmax + self.gap
        };
        (xmin, ymin, zmin, xmax, ymax, zmax)
    }

    /// Returns the minimum corner
    pub fn corner_min(&self) -> Pnt {
        let (xmin, ymin, zmin, _, _, _) = self.get();
        Pnt::new(xmin, ymin, zmin)
    }

    /// Returns the maximum corner
    pub fn corner_max(&self) -> Pnt {
        let (_, _, _, xmax, ymax, zmax) = self.get();
        Pnt::new(xmax, ymax, zmax)
    }

    /// Returns the center of this box, or None if void
    pub fn center(&self) -> Option<Pnt> {
        if self.is_void() {
            None
        } else {
            let (xmin, ymin, zmin, xmax, ymax, zmax) = self.get();
            Some(Pnt::new(
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                (zmin + zmax) / 2.0,
            ))
        }
    }

    /// Opens the box in Xmin direction (makes it infinite)
    pub fn open_xmin(&mut self) {
        self.flags |= XMIN_MASK;
    }

    /// Opens the box in Xmax direction (makes it infinite)
    pub fn open_xmax(&mut self) {
        self.flags |= XMAX_MASK;
    }

    /// Opens the box in Ymin direction (makes it infinite)
    pub fn open_ymin(&mut self) {
        self.flags |= YMIN_MASK;
    }

    /// Opens the box in Ymax direction (makes it infinite)
    pub fn open_ymax(&mut self) {
        self.flags |= YMAX_MASK;
    }

    /// Opens the box in Zmin direction (makes it infinite)
    pub fn open_zmin(&mut self) {
        self.flags |= ZMIN_MASK;
    }

    /// Opens the box in Zmax direction (makes it infinite)
    pub fn open_zmax(&mut self) {
        self.flags |= ZMAX_MASK;
    }

    /// Returns true if this box is open in any direction
    pub fn is_open(&self) -> bool {
        (self.flags & WHOLE_MASK) != 0
    }

    /// Returns true if this box is open in Xmin direction
    pub fn is_open_xmin(&self) -> bool {
        (self.flags & XMIN_MASK) != 0
    }

    /// Returns true if this box is open in Xmax direction
    pub fn is_open_xmax(&self) -> bool {
        (self.flags & XMAX_MASK) != 0
    }

    /// Returns true if this box is open in Ymin direction
    pub fn is_open_ymin(&self) -> bool {
        (self.flags & YMIN_MASK) != 0
    }

    /// Returns true if this box is open in Ymax direction
    pub fn is_open_ymax(&self) -> bool {
        (self.flags & YMAX_MASK) != 0
    }

    /// Returns true if this box is open in Zmin direction
    pub fn is_open_zmin(&self) -> bool {
        (self.flags & ZMIN_MASK) != 0
    }

    /// Returns true if this box is open in Zmax direction
    pub fn is_open_zmax(&self) -> bool {
        (self.flags & ZMAX_MASK) != 0
    }

    /// Returns true if the X range is smaller than tolerance
    pub fn is_x_thin(&self, tol: f64) -> bool {
        if self.is_void() || self.is_open_xmin() || self.is_open_xmax() {
            false
        } else {
            (self.xmax - self.xmin) < tol
        }
    }

    /// Returns true if the Y range is smaller than tolerance
    pub fn is_y_thin(&self, tol: f64) -> bool {
        if self.is_void() || self.is_open_ymin() || self.is_open_ymax() {
            false
        } else {
            (self.ymax - self.ymin) < tol
        }
    }

    /// Returns true if the Z range is smaller than tolerance
    pub fn is_z_thin(&self, tol: f64) -> bool {
        if self.is_void() || self.is_open_zmin() || self.is_open_zmax() {
            false
        } else {
            (self.zmax - self.zmin) < tol
        }
    }

    /// Returns true if box is thin in all three dimensions
    pub fn is_thin(&self, tol: f64) -> bool {
        self.is_x_thin(tol) && self.is_y_thin(tol) && self.is_z_thin(tol)
    }

    /// Returns the square extent (length of diagonal)
    pub fn square_extent(&self) -> f64 {
        let (xmin, ymin, zmin, xmax, ymax, zmax) = self.get();
        let dx = xmax - xmin;
        let dy = ymax - ymin;
        let dz = zmax - zmin;
        dx * dx + dy * dy + dz * dz
    }

    /// Checks if a point is outside this box
    pub fn is_out_point(&self, p: Pnt) -> bool {
        let (xmin, ymin, zmin, xmax, ymax, zmax) = self.get();
        p.x() < xmin || p.x() > xmax ||
        p.y() < ymin || p.y() > ymax ||
        p.z() < zmin || p.z() > zmax
    }

    /// Checks if another box is outside this box
    pub fn is_out_box(&self, other: &BndBox) -> bool {
        if self.is_void() || other.is_void() {
            return true;
        }
        let (xmin1, ymin1, zmin1, xmax1, ymax1, zmax1) = self.get();
        let (xmin2, ymin2, zmin2, xmax2, ymax2, zmax2) = other.get();
        
        xmax2 < xmin1 || xmin2 > xmax1 ||
        ymax2 < ymin1 || ymin2 > ymax1 ||
        zmax2 < zmin1 || zmin2 > zmax1
    }

    /// Returns the distance between two boxes
    pub fn distance(&self, other: &BndBox) -> f64 {
        if !self.is_out_box(other) {
            return 0.0;
        }
        let (xmin1, ymin1, zmin1, xmax1, ymax1, zmax1) = self.get();
        let (xmin2, ymin2, zmin2, xmax2, ymax2, zmax2) = other.get();
        
        let dx = if xmax2 < xmin1 {
            xmin1 - xmax2
        } else if xmin2 > xmax1 {
            xmin2 - xmax1
        } else {
            0.0
        };
        
        let dy = if ymax2 < ymin1 {
            ymin1 - ymax2
        } else if ymin2 > ymax1 {
            ymin2 - ymax1
        } else {
            0.0
        };
        
        let dz = if zmax2 < zmin1 {
            zmin1 - zmax2
        } else if zmin2 > zmax1 {
            zmin2 - zmax1
        } else {
            0.0
        };
        
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Returns true if a point is inside this box
    pub fn contains(&self, p: Pnt) -> bool {
        !self.is_out_point(p)
    }

    /// Returns true if another box intersects this box
    pub fn intersects(&self, other: &BndBox) -> bool {
        !self.is_out_box(other)
    }

    /// Adds another box to this box
    pub fn add_box(&mut self, other: &BndBox) {
        if other.is_void() {
            return;
        }
        if self.is_void() {
            *self = *other;
            return;
        }
        let gap = self.gap.max(other.gap);
        self.xmin = self.xmin.min(other.xmin - other.gap);
        self.xmax = self.xmax.max(other.xmax + other.gap);
        self.ymin = self.ymin.min(other.ymin - other.gap);
        self.ymax = self.ymax.max(other.ymax + other.gap);
        self.zmin = self.zmin.min(other.zmin - other.gap);
        self.zmax = self.zmax.max(other.zmax + other.gap);
        self.gap = gap;
        self.flags |= other.flags;
    }

    /// Adds a point to this box
    pub fn add_point(&mut self, p: Pnt) {
        self.update_point(p.x(), p.y(), p.z());
    }

    /// Adds a direction to this box
    pub fn add_direction(&mut self, d: Dir) {
        let x = d.x();
        let y = d.y();
        let z = d.z();
        if x > 0.0 {
            self.open_xmax();
        } else if x < 0.0 {
            self.open_xmin();
        }
        if y > 0.0 {
            self.open_ymax();
        } else if y < 0.0 {
            self.open_ymin();
        }
        if z > 0.0 {
            self.open_zmax();
        } else if z < 0.0 {
            self.open_zmin();
        }
    }

    /// Returns the transformed box
    pub fn transformed(&self, trsf: &Trsf) -> BndBox {
        if self.is_void() {
            return BndBox::new();
        }
        
        // Transform all 8 corners
        let (xmin, ymin, zmin, xmax, ymax, zmax) = self.get();
        let corners = [
            Pnt::new(xmin, ymin, zmin),
            Pnt::new(xmax, ymin, zmin),
            Pnt::new(xmin, ymax, zmin),
            Pnt::new(xmin, ymin, zmax),
            Pnt::new(xmax, ymax, zmin),
            Pnt::new(xmax, ymin, zmax),
            Pnt::new(xmin, ymax, zmax),
            Pnt::new(xmax, ymax, zmax),
        ];
        
        let mut result = BndBox::new();
        for corner in &corners {
            result.add_point(trsf.transform_point(corner));
        }
        
        if self.is_open() {
            // Handle open directions
            if self.is_open_xmin() || self.is_open_xmax() {
                result.flags |= XMIN_MASK | XMAX_MASK;
            }
            if self.is_open_ymin() || self.is_open_ymax() {
                result.flags |= YMIN_MASK | YMAX_MASK;
            }
            if self.is_open_zmin() || self.is_open_zmax() {
                result.flags |= ZMIN_MASK | ZMAX_MASK;
            }
        }
        
        result.gap = self.gap;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constructor() {
        let b = BndBox::new();
        assert!(b.is_void());
        assert!(!b.is_whole());
        assert!((b.gap() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_points() {
        let p_min = Pnt::new(1.0, 2.0, 3.0);
        let p_max = Pnt::new(4.0, 5.0, 6.0);
        let b = BndBox::from_points(p_min, p_max);
        assert!(!b.is_void());
        let (xmin, ymin, zmin, xmax, ymax, zmax) = b.get();
        assert!((xmin - 1.0).abs() < PRECISION_CONFUSION);
        assert!((ymin - 2.0).abs() < PRECISION_CONFUSION);
        assert!((zmin - 3.0).abs() < PRECISION_CONFUSION);
        assert!((xmax - 4.0).abs() < PRECISION_CONFUSION);
        assert!((ymax - 5.0).abs() < PRECISION_CONFUSION);
        assert!((zmax - 6.0).abs() < PRECISION_CONFUSION);
    }

    #[test]
    fn test_set_void() {
        let mut b = BndBox::new();
        b.update_point(1.0, 2.0, 3.0);
        assert!(!b.is_void());
        b.set_void();
        assert!(b.is_void());
    }
}
