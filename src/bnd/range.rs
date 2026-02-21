//! Bnd_Range - 1D Range

/// Status of intersection test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntersectStatus {
    /// Value is outside the range
    Out,
    /// Value is inside the range
    In,
    /// Value is on the boundary of the range
    Boundary,
}

/// Represents a 1D range [min, max]
#[derive(Clone, Copy, Debug)]
pub struct BndRange {
    min: f64,
    max: f64,
    is_void: bool,
}

impl BndRange {
    /// Creates an empty range
    pub fn new() -> Self {
        BndRange {
            min: 0.0,
            max: 0.0,
            is_void: true,
        }
    }

    /// Creates a range [min, max]
    pub fn from_bounds(min: f64, max: f64) -> Self {
        BndRange {
            min,
            max,
            is_void: false,
        }
    }

    /// Returns true if this range is empty
    pub fn is_void(&self) -> bool {
        self.is_void
    }

    /// Returns the delta (max - min)
    pub fn delta(&self) -> f64 {
        if self.is_void {
            -1.0
        } else {
            self.max - self.min
        }
    }

    /// Returns the minimum value
    pub fn min(&self) -> Option<f64> {
        if self.is_void {
            None
        } else {
            Some(self.min)
        }
    }

    /// Returns the maximum value
    pub fn max(&self) -> Option<f64> {
        if self.is_void {
            None
        } else {
            Some(self.max)
        }
    }

    /// Returns (min, max)
    pub fn bounds(&self) -> Option<(f64, f64)> {
        if self.is_void {
            None
        } else {
            Some((self.min, self.max))
        }
    }

    /// Returns the center of the range
    pub fn center(&self) -> Option<f64> {
        if self.is_void {
            None
        } else {
            Some((self.min + self.max) / 2.0)
        }
    }

    /// Gets min value (returns true if not void)
    pub fn get_min(&self, min: &mut f64) -> bool {
        if self.is_void {
            false
        } else {
            *min = self.min;
            true
        }
    }

    /// Gets max value (returns true if not void)
    pub fn get_max(&self, max: &mut f64) -> bool {
        if self.is_void {
            false
        } else {
            *max = self.max;
            true
        }
    }

    /// Gets bounds (returns true if not void)
    pub fn get_bounds(&self, min: &mut f64, max: &mut f64) -> bool {
        if self.is_void {
            false
        } else {
            *min = self.min;
            *max = self.max;
            true
        }
    }

    /// Gets intermediate point: p = min + lambda * (max - min)
    pub fn get_intermediate_point(&self, lambda: f64, par: &mut f64) -> bool {
        if self.is_void {
            false
        } else {
            *par = self.min + lambda * (self.max - self.min);
            true
        }
    }

    /// Sets this range as void
    pub fn set_void(&mut self) {
        self.is_void = true;
    }

    /// Adds a value to this range
    pub fn add(&mut self, val: f64) {
        if self.is_void {
            self.min = val;
            self.max = val;
            self.is_void = false;
        } else {
            self.min = self.min.min(val);
            self.max = self.max.max(val);
        }
    }

    /// Adds another range to this range
    pub fn add_range(&mut self, other: &BndRange) {
        if other.is_void {
            return;
        }
        if self.is_void {
            *self = *other;
        } else {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    /// Computes the intersection with another range
    pub fn common(&mut self, other: &BndRange) {
        if self.is_void || other.is_void {
            self.is_void = true;
            return;
        }
        let new_min = self.min.max(other.min);
        let new_max = self.max.min(other.max);
        if new_min > new_max {
            self.is_void = true;
        } else {
            self.min = new_min;
            self.max = new_max;
        }
    }

    /// Computes the union with another range
    pub fn union(&mut self, other: &BndRange) -> bool {
        if other.is_void {
            return false;
        }
        if self.is_void {
            *self = *other;
            return true;
        }
        // Check if ranges overlap or touch
        if other.min <= self.max + 1e-7 && other.max + 1e-7 >= self.min {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
            true
        } else {
            false
        }
    }

    /// Tests if a value is intersected with this range
    pub fn is_intersected(&self, val: f64) -> IntersectStatus {
        if self.is_void {
            IntersectStatus::Out
        } else if val < self.min {
            IntersectStatus::Out
        } else if val > self.max {
            IntersectStatus::Out
        } else if (val - self.min).abs() < 1e-7 || (val - self.max).abs() < 1e-7 {
            IntersectStatus::Boundary
        } else {
            IntersectStatus::In
        }
    }

    /// Enlarges this range
    pub fn enlarge(&mut self, delta: f64) {
        if !self.is_void {
            self.min -= delta;
            self.max += delta;
        }
    }

    /// Shifts this range by a value
    pub fn shift(&mut self, val: f64) {
        if !self.is_void {
            self.min += val;
            self.max += val;
        }
    }

    /// Trims from min
    pub fn trim_from(&mut self, val: f64) {
        if self.is_void {
            return;
        }
        if val > self.max {
            self.is_void = true;
        } else {
            self.min = self.min.max(val);
        }
    }

    /// Trims to max
    pub fn trim_to(&mut self, val: f64) {
        if self.is_void {
            return;
        }
        if val < self.min {
            self.is_void = true;
        } else {
            self.max = self.max.min(val);
        }
    }

    /// Returns true if a value is outside this range
    pub fn is_out(&self, val: f64) -> bool {
        if self.is_void {
            true
        } else {
            val < self.min || val > self.max
        }
    }

    /// Returns true if another range is outside this range
    pub fn is_out_range(&self, other: &BndRange) -> bool {
        if self.is_void || other.is_void {
            true
        } else {
            other.max < self.min || other.min > self.max
        }
    }

    /// Returns true if another range intersects this range
    pub fn intersects(&self, other: &BndRange) -> bool {
        !self.is_out_range(other)
    }

    /// Returns true if this range equals another
    pub fn equals(&self, other: &BndRange) -> bool {
        if self.is_void && other.is_void {
            true
        } else if self.is_void || other.is_void {
            false
        } else {
            (self.min - other.min).abs() < 1e-7 && (self.max - other.max).abs() < 1e-7
        }
    }

    /// Returns true if this range contains a value
    pub fn contains(&self, val: f64) -> bool {
        !self.is_void && val >= self.min && val <= self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_constructor() {
        let r = BndRange::new();
        assert!(r.is_void());
    }
}
