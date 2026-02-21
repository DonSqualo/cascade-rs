//! Bnd_Box2d - 2D Bounding Box

use crate::gp::{Pnt2d, XY};

/// Represents a 2D bounding box with optional gaps and open directions
#[derive(Clone, Copy, Debug)]
pub struct BndBox2d {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    gap: f64,
    flags: u8,
}

// Flag constants
const VOID_MASK: u8 = 0x10;   // Empty box
const WHOLE_MASK: u8 = 0x0F;  // All directions open
const XMIN_MASK: u8 = 0x01;   // Open Xmin
const XMAX_MASK: u8 = 0x02;   // Open Xmax
const YMIN_MASK: u8 = 0x04;   // Open Ymin
const YMAX_MASK: u8 = 0x08;   // Open Ymax

impl Default for BndBox2d {
    fn default() -> Self {
        Self::new()
    }
}

impl BndBox2d {
    /// Creates an empty bounding box (void)
    pub fn new() -> Self {
        BndBox2d {
            xmin: f64::MAX,
            xmax: f64::NEG_INFINITY,
            ymin: f64::MAX,
            ymax: f64::NEG_INFINITY,
            gap: 0.0,
            flags: VOID_MASK,
        }
    }

    /// Returns true if this box is void (empty)
    pub fn is_void(&self) -> bool {
        (self.flags & VOID_MASK) != 0
    }

    /// Returns true if this box covers all of 2D space
    pub fn is_whole(&self) -> bool {
        (self.flags & WHOLE_MASK) == WHOLE_MASK
    }

    /// Sets this box as void (empty)
    pub fn set_void(&mut self) {
        self.xmin = f64::MAX;
        self.xmax = f64::NEG_INFINITY;
        self.ymin = f64::MAX;
        self.ymax = f64::NEG_INFINITY;
        self.gap = 0.0;
        self.flags = VOID_MASK;
    }

    /// Sets this box to cover all of 2D space
    pub fn set_whole(&mut self) {
        self.flags = WHOLE_MASK;
    }

    /// Sets this box to contain only the given point
    pub fn set_point(&mut self, p: Pnt2d) {
        self.set_void();
        self.update(p.x(), p.y());
    }

    /// Updates this box to contain the given interval
    pub fn update(&mut self, xmin: f64, ymin: f64, xmax: f64, ymax: f64) {
        if self.is_void() {
            self.xmin = xmin;
            self.xmax = xmax;
            self.ymin = ymin;
            self.ymax = ymax;
            self.flags = 0;
        } else {
            self.xmin = self.xmin.min(xmin);
            self.xmax = self.xmax.max(xmax);
            self.ymin = self.ymin.min(ymin);
            self.ymax = self.ymax.max(ymax);
        }
    }

    /// Updates this box with a single point
    pub fn update_point(&mut self, x: f64, y: f64) {
        self.update(x, y, x, y);
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

    /// Returns the bounds including gap: (xmin, ymin, xmax, ymax)
    pub fn get(&self) -> (f64, f64, f64, f64) {
        let xmin = if self.is_open_xmin() {
            f64::NEG_INFINITY
        } else {
            self.xmin - self.gap
        };
        let xmax = if self.is_open_xmax() {
            f64::INFINITY
        } else {
            self.xmax + self.gap
        };
        let ymin = if self.is_open_ymin() {
            f64::NEG_INFINITY
        } else {
            self.ymin - self.gap
        };
        let ymax = if self.is_open_ymax() {
            f64::INFINITY
        } else {
            self.ymax + self.gap
        };
        (xmin, ymin, xmax, ymax)
    }

    /// Returns the minimum corner
    pub fn corner_min(&self) -> XY {
        let (xmin, ymin, _, _) = self.get();
        XY::new(xmin, ymin)
    }

    /// Returns the maximum corner
    pub fn corner_max(&self) -> XY {
        let (_, _, xmax, ymax) = self.get();
        XY::new(xmax, ymax)
    }

    /// Returns the center of this box, or None if void
    pub fn center(&self) -> Option<XY> {
        if self.is_void() {
            None
        } else {
            let (xmin, ymin, xmax, ymax) = self.get();
            Some(XY::new((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))
        }
    }

    /// Opens the box in Xmin direction
    pub fn open_xmin(&mut self) {
        self.flags |= XMIN_MASK;
    }

    /// Opens the box in Xmax direction
    pub fn open_xmax(&mut self) {
        self.flags |= XMAX_MASK;
    }

    /// Opens the box in Ymin direction
    pub fn open_ymin(&mut self) {
        self.flags |= YMIN_MASK;
    }

    /// Opens the box in Ymax direction
    pub fn open_ymax(&mut self) {
        self.flags |= YMAX_MASK;
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

    /// Returns the square extent
    pub fn square_extent(&self) -> f64 {
        let (xmin, ymin, xmax, ymax) = self.get();
        let dx = xmax - xmin;
        let dy = ymax - ymin;
        dx * dx + dy * dy
    }

    /// Returns the distance between two boxes
    pub fn distance(&self, other: &BndBox2d) -> f64 {
        if !self.is_out(other) {
            return 0.0;
        }
        let (xmin1, ymin1, xmax1, ymax1) = self.get();
        let (xmin2, ymin2, xmax2, ymax2) = other.get();
        
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
        
        (dx * dx + dy * dy).sqrt()
    }

    /// Checks if a point is outside this box
    pub fn is_out_point(&self, p: XY) -> bool {
        let (xmin, ymin, xmax, ymax) = self.get();
        p.x() < xmin || p.x() > xmax || p.y() < ymin || p.y() > ymax
    }

    /// Checks if another box is outside this box
    pub fn is_out_box(&self, other: &BndBox2d) -> bool {
        if self.is_void() || other.is_void() {
            return true;
        }
        let (xmin1, ymin1, xmax1, ymax1) = self.get();
        let (xmin2, ymin2, xmax2, ymax2) = other.get();
        
        xmax2 < xmin1 || xmin2 > xmax1 || ymax2 < ymin1 || ymin2 > ymax1
    }

    /// Checks if a point is outside this box
    pub fn is_out(&self, other: &BndBox2d) -> bool {
        self.is_out_box(other)
    }

    /// Returns true if a point is inside this box
    pub fn contains(&self, p: Pnt2d) -> bool {
        !self.is_out_point(XY::new(p.x(), p.y()))
    }

    /// Returns true if another box intersects this box
    pub fn intersects(&self, other: &BndBox2d) -> bool {
        !self.is_out_box(other)
    }

    /// Adds another box to this box
    pub fn add(&mut self, other: &BndBox2d) {
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
        self.gap = gap;
        self.flags |= other.flags;
    }

    /// Adds a point to this box
    pub fn add_point(&mut self, p: Pnt2d) {
        self.update_point(p.x(), p.y());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constructor() {
        let b = BndBox2d::new();
        assert!(b.is_void());
        assert!(!b.is_whole());
        assert!((b.gap() - 0.0).abs() < f64::EPSILON);
    }
}
