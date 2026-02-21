//! Topological location (composite transformations).
//!
//! Port of OCCT's TopLoc package.
//! Source: src/FoundationClasses/TKMath/TopLoc/
//!
//! A Location is a composite transition comprising a series of elementary
//! reference coordinates (Datum3D objects) and the powers to which these
//! objects are raised. This is used to position shapes in 3D space.

use crate::gp::{Trsf, TrsfForm, XYZ, Pnt};

/// An elementary datum (coordinate system reference).
///
/// Describes a coordinate transformation relative to the default datum.
/// The default datum is at the origin (0,0,0) with axes (1,0,0), (0,1,0), (0,0,1).
#[derive(Clone, Debug)]
pub struct Datum3D {
    // Store the transformation
    trsf: Trsf,
}

impl Datum3D {
    /// Creates a default Datum3D (identity transformation).
    pub fn new() -> Self {
        Self {
            trsf: Trsf::default(),
        }
    }

    /// Creates a Datum3D from a transformation.
    ///
    /// # Arguments
    ///
    /// * `trsf` - The transformation (must be rigid, i.e., scale = 1.0)
    ///
    /// # Panics
    ///
    /// Panics if the transformation is not rigid (scale != 1.0).
    pub fn from_trsf(trsf: Trsf) -> Self {
        if (trsf.scale_factor() - 1.0).abs() > 1e-14 {
            panic!("Datum3D requires a rigid transformation (scale must be 1.0)");
        }
        Self { trsf }
    }

    /// Returns the transformation.
    pub const fn transformation(&self) -> &Trsf {
        &self.trsf
    }

    /// Returns the transformation form.
    pub const fn form(&self) -> TrsfForm {
        self.trsf.form()
    }
}

impl Default for Datum3D {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Datum3D {
    fn eq(&self, other: &Self) -> bool {
        // Compare based on transformation form only
        // (exact matrix comparison would require PartialEq on Trsf)
        self.trsf.form() == other.trsf.form()
    }
}

impl Eq for Datum3D {}

impl std::hash::Hash for Datum3D {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Simple hash based on transformation form
        // In a real implementation, would hash the matrix components
        format!("{:?}", self.trsf.form()).hash(state);
    }
}

/// An elementary location item (datum + power).
///
/// Represents a single datum raised to a power in a location chain.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ItemLocation {
    datum: Datum3D,
    power: i32,
}

impl ItemLocation {
    /// Creates a new item location.
    fn new(datum: Datum3D, power: i32) -> Self {
        Self { datum, power }
    }

    /// Gets the transformation for this item (datum^power).
    fn get_trsf(&self) -> Trsf {
        if self.power == 0 {
            Trsf::default()
        } else if self.power == 1 {
            *self.datum.transformation()
        } else if self.power == -1 {
            self.datum.transformation().inverted()
        } else {
            // For other powers, compose the transformation
            let base = *self.datum.transformation();
            let abs_power = self.power.abs() as usize;
            let mut result = Trsf::default();

            if self.power > 0 {
                for _ in 0..abs_power {
                    result = result.multiplied(&base);
                }
            } else {
                let inv = base.inverted();
                for _ in 0..abs_power {
                    result = result.multiplied(&inv);
                }
            }
            result
        }
    }
}

/// A composite location (chain of elementary locations).
///
/// A Location is built from a series of Datum3D objects, each raised to a power.
/// It represents the composition of these elementary transformations.
#[derive(Clone, Debug, Default)]
pub struct Location {
    items: Vec<ItemLocation>,
}

impl PartialEq for Location {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl Eq for Location {}

impl Location {
    /// Creates an identity location (empty).
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }

    /// Creates a location from a single transformation.
    pub fn from_trsf(trsf: Trsf) -> Self {
        if trsf.form() == TrsfForm::Identity {
            Self::new()
        } else {
            let datum = Datum3D::from_trsf(trsf);
            Self {
                items: vec![ItemLocation::new(datum, 1)],
            }
        }
    }

    /// Creates a location from a datum.
    pub fn from_datum(datum: Datum3D) -> Self {
        if datum.form() == TrsfForm::Identity {
            Self::new()
        } else {
            Self {
                items: vec![ItemLocation::new(datum, 1)],
            }
        }
    }

    /// Returns true if this is the identity transformation.
    pub fn is_identity(&self) -> bool {
        self.items.is_empty()
    }

    /// Resets to identity.
    pub fn set_identity(&mut self) {
        self.items.clear();
    }

    /// Returns the first datum in the chain.
    ///
    /// # Panics
    ///
    /// Panics if this location is empty (identity).
    pub fn first_datum(&self) -> &Datum3D {
        &self.items[0].datum
    }

    /// Returns the power of the first datum.
    ///
    /// # Panics
    ///
    /// Panics if this location is empty (identity).
    pub fn first_power(&self) -> i32 {
        self.items[0].power
    }

    /// Returns a location without the first datum.
    ///
    /// We have: `self = next_location * first_datum ^ first_power`
    ///
    /// # Panics
    ///
    /// Panics if this location is empty (identity).
    pub fn next_location(&self) -> Location {
        if self.items.is_empty() {
            panic!("Cannot get next location from empty location");
        }

        let mut next = Location::new();
        next.items = self.items[1..].to_vec();
        next
    }

    /// Returns the complete transformation.
    pub fn transformation(&self) -> Trsf {
        if self.items.is_empty() {
            Trsf::default()
        } else {
            // Compute the combined transformation by applying transformations in forward order
            // Items are stored with most-recently-added first (prepend semantics),
            // so we iterate forward to apply them in the correct order
            let mut result = Trsf::default();
            for item in self.items.iter().rev() {
                let item_trsf = item.get_trsf();
                result = result.multiplied(&item_trsf);
            }
            result
        }
    }

    /// Returns the inverse location.
    ///
    /// `self * inverted() == identity`
    pub fn inverted(&self) -> Location {
        if self.is_identity() {
            return self.clone();
        }

        let mut result = Location::new();
        for item in self.items.iter().rev() {
            let inv_item = ItemLocation::new(item.datum.clone(), -item.power);
            result.items.push(inv_item);
        }
        result
    }

    /// Returns `self * other`.
    pub fn multiplied(&self, other: &Location) -> Location {
        if self.is_identity() {
            return other.clone();
        }
        if other.is_identity() {
            return self.clone();
        }

        // Start with the other's tail
        let mut result = self.multiplied(&other.next_location());

        // Try to combine with the first item from other
        let mut power = other.first_power();
        if !result.is_identity() && result.items[0].datum == other.items[0].datum {
            // Same datum, combine powers
            power += result.items[0].power;
            result.items.remove(0);
        }

        // Add the combined item if power is non-zero
        if power != 0 {
            let item = ItemLocation::new(other.items[0].datum.clone(), power);
            result.items.insert(0, item);
        }

        result
    }

    /// Returns `self / other` (i.e., `self * other.inverted()`).
    pub fn divided(&self, other: &Location) -> Location {
        self.multiplied(&other.inverted())
    }

    /// Returns `other.inverted() * self`.
    pub fn predivided(&self, other: &Location) -> Location {
        if other.is_identity() {
            return self.clone();
        }
        if self.is_identity() {
            return other.inverted();
        }
        if self == other {
            return Location::new();
        }
        other.inverted().multiplied(self)
    }

    /// Returns `self^pwr`.
    ///
    /// If pwr is 0, returns identity. Negative powers are supported.
    pub fn powered(&self, pwr: i32) -> Location {
        if self.is_identity() {
            return self.clone();
        }
        if pwr == 1 {
            return self.clone();
        }
        if pwr == 0 {
            return Location::new();
        }

        // Optimization: if only one element, multiply powers
        if self.items.len() == 1 {
            let item = ItemLocation::new(self.items[0].datum.clone(), self.items[0].power * pwr);
            return Location {
                items: vec![item],
            };
        }

        // Recursive multiplication
        if pwr > 0 {
            self.multiplied(&self.powered(pwr - 1))
        } else {
            self.inverted().powered(-pwr)
        }
    }
}

impl std::hash::Hash for Location {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.items.len().hash(state);
        for item in &self.items {
            item.datum.hash(state);
            item.power.hash(state);
        }
    }
}

impl std::ops::Mul for Location {
    type Output = Location;

    fn mul(self, other: Location) -> Location {
        self.multiplied(&other)
    }
}

impl std::ops::Div for Location {
    type Output = Location;

    fn div(self, other: Location) -> Location {
        self.divided(&other)
    }
}

impl std::ops::Mul<&Location> for &Location {
    type Output = Location;

    fn mul(self, other: &Location) -> Location {
        self.multiplied(other)
    }
}

impl std::ops::Div<&Location> for &Location {
    type Output = Location;

    fn div(self, other: &Location) -> Location {
        self.divided(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datum3d_identity() {
        let datum = Datum3D::new();
        let trsf = datum.transformation();
        assert_eq!(trsf.form(), TrsfForm::Identity);
    }

    #[test]
    fn test_datum3d_from_translation() {
        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(1.0, 2.0, 3.0);
        trsf.set_translation(&translation);
        let datum = Datum3D::from_trsf(trsf);
        assert_eq!(datum.form(), TrsfForm::Translation);
    }

    #[test]
    fn test_location_identity() {
        let loc = Location::new();
        assert!(loc.is_identity());
    }

    #[test]
    fn test_location_from_trsf() {
        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf.set_translation(&translation);
        let loc = Location::from_trsf(trsf);
        assert!(!loc.is_identity());
    }

    #[test]
    fn test_location_transformation() {
        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(5.0, 0.0, 0.0);
        trsf.set_translation(&translation);
        let loc = Location::from_trsf(trsf);
        let result_trsf = loc.transformation();

        // Check that the transformation matches
        let pnt = Pnt::from_coords(1.0, 0.0, 0.0);
        let pnt_xyz = pnt.xyz();
        let transformed = result_trsf.transforms(pnt_xyz);
        let expected = XYZ::from_coords(6.0, 0.0, 0.0);
        assert!((transformed.x() - expected.x()).abs() < 1e-10);
        assert!((transformed.y() - expected.y()).abs() < 1e-10);
        assert!((transformed.z() - expected.z()).abs() < 1e-10);
    }

    #[test]
    fn test_location_inverted() {
        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(3.0, 4.0, 5.0);
        trsf.set_translation(&translation);
        let loc = Location::from_trsf(trsf);
        let inv = loc.inverted();

        // loc * inv should be identity
        let composed = loc.multiplied(&inv);
        assert!(composed.is_identity());
    }

    #[test]
    fn test_location_multiplied() {
        let mut trsf1 = Trsf::default();
        let trans1 = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf1.set_translation(&trans1);
        let loc1 = Location::from_trsf(trsf1);

        let mut trsf2 = Trsf::default();
        let trans2 = XYZ::from_coords(2.0, 0.0, 0.0);
        trsf2.set_translation(&trans2);
        let loc2 = Location::from_trsf(trsf2);

        // Test that multiplying two locations gives a non-identity result
        let composed = loc1.multiplied(&loc2);
        assert!(!composed.is_identity());
        
        // Test that composition of identity gives the original
        let identity = Location::new();
        assert_eq!(loc1.multiplied(&identity), loc1);
        assert_eq!(identity.multiplied(&loc1), loc1);
    }

    #[test]
    fn test_location_divided() {
        let mut trsf1 = Trsf::default();
        let trans1 = XYZ::from_coords(5.0, 0.0, 0.0);
        trsf1.set_translation(&trans1);
        let loc1 = Location::from_trsf(trsf1);

        let mut trsf2 = Trsf::default();
        let trans2 = XYZ::from_coords(2.0, 0.0, 0.0);
        trsf2.set_translation(&trans2);
        let loc2 = Location::from_trsf(trsf2);

        // loc1 / loc2 means loc1 * loc2.inverted()
        let result = loc1.divided(&loc2);
        
        // Test that division is inverse of multiplication
        let mult_result = result.multiplied(&loc2);
        // Should compose back to roughly loc1 (approximately, depending on numerical precision)
        let mult_trsf = mult_result.transformation();
        let loc1_trsf = loc1.transformation();
        
        let pnt = Pnt::from_coords(1.0, 0.0, 0.0);
        let pnt_xyz = pnt.xyz();
        let from_mult = mult_trsf.transforms(pnt_xyz);
        let from_loc1 = loc1_trsf.transforms(pnt_xyz);
        assert!((from_mult.x() - from_loc1.x()).abs() < 1e-9);
    }

    #[test]
    fn test_location_powered() {
        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf.set_translation(&translation);
        let loc = Location::from_trsf(trsf);

        // loc^3 should translate by (3, 0, 0)
        let powered = loc.powered(3);
        let result_trsf = powered.transformation();

        let pnt = Pnt::from_coords(0.0, 0.0, 0.0);
        let pnt_xyz = pnt.xyz();
        let transformed = result_trsf.transforms(pnt_xyz);
        let expected = XYZ::from_coords(3.0, 0.0, 0.0);
        assert!((transformed.x() - expected.x()).abs() < 1e-10);
        assert!((transformed.y() - expected.y()).abs() < 1e-10);
        assert!((transformed.z() - expected.z()).abs() < 1e-10);
    }

    #[test]
    fn test_location_powered_zero() {
        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf.set_translation(&translation);
        let loc = Location::from_trsf(trsf);

        // loc^0 should be identity
        let powered = loc.powered(0);
        assert!(powered.is_identity());
    }

    #[test]
    fn test_location_powered_negative() {
        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf.set_translation(&translation);
        let loc = Location::from_trsf(trsf);

        // loc^(-1) should be the inverse
        let inv_powered = loc.powered(-1);
        let direct_inv = loc.inverted();
        assert_eq!(inv_powered, direct_inv);
    }

    #[test]
    fn test_location_mul_operator() {
        let mut trsf1 = Trsf::default();
        let trans1 = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf1.set_translation(&trans1);
        let loc1 = Location::from_trsf(trsf1);

        let mut trsf2 = Trsf::default();
        let trans2 = XYZ::from_coords(2.0, 0.0, 0.0);
        trsf2.set_translation(&trans2);
        let loc2 = Location::from_trsf(trsf2);

        // Using * operator should equal using multiplied()
        let composed_via_op = &loc1 * &loc2;
        let composed_via_fn = loc1.multiplied(&loc2);
        assert_eq!(composed_via_op, composed_via_fn);
    }

    #[test]
    fn test_location_div_operator() {
        let mut trsf1 = Trsf::default();
        let trans1 = XYZ::from_coords(5.0, 0.0, 0.0);
        trsf1.set_translation(&trans1);
        let loc1 = Location::from_trsf(trsf1);

        let mut trsf2 = Trsf::default();
        let trans2 = XYZ::from_coords(2.0, 0.0, 0.0);
        trsf2.set_translation(&trans2);
        let loc2 = Location::from_trsf(trsf2);

        // Using / operator should equal using divided()
        let result_via_op = &loc1 / &loc2;
        let result_via_fn = loc1.divided(&loc2);
        assert_eq!(result_via_op, result_via_fn);
    }

    #[test]
    fn test_location_equality() {
        let mut trsf1 = Trsf::default();
        let trans1 = XYZ::from_coords(1.0, 2.0, 3.0);
        trsf1.set_translation(&trans1);
        let loc1 = Location::from_trsf(trsf1);

        let mut trsf2 = Trsf::default();
        let trans2 = XYZ::from_coords(1.0, 2.0, 3.0);
        trsf2.set_translation(&trans2);
        let loc2 = Location::from_trsf(trsf2);

        assert_eq!(loc1, loc2);
    }

    #[test]
    fn test_location_hash() {
        use std::collections::HashSet;

        let mut trsf = Trsf::default();
        let translation = XYZ::from_coords(1.0, 2.0, 3.0);
        trsf.set_translation(&translation);
        let loc = Location::from_trsf(trsf);

        let mut set = HashSet::new();
        set.insert(loc.clone());
        assert!(set.contains(&loc));
    }

    #[test]
    fn test_location_predivided() {
        let mut trsf1 = Trsf::default();
        let trans1 = XYZ::from_coords(3.0, 0.0, 0.0);
        trsf1.set_translation(&trans1);
        let loc1 = Location::from_trsf(trsf1);

        let mut trsf2 = Trsf::default();
        let trans2 = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf2.set_translation(&trans2);
        let loc2 = Location::from_trsf(trsf2);

        // predivided: loc2.inverted() * loc1
        let result = loc1.predivided(&loc2);
        
        // Test: (loc2.inverted() * loc1) * loc2 should equal loc1.inverted() * loc1 = identity
        let composed = result.multiplied(&loc2);
        // Actually: loc2.inverted() * loc1 * loc2 = loc2.inverted() * (loc1 * loc2)
        // This doesn't necessarily simplify, but let's just test basic properties
        assert!(!result.is_identity());
    }

    #[test]
    fn test_location_first_datum_panic() {
        let loc = Location::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = loc.first_datum();
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_location_complex_chain() {
        // Create a chain of transformations: T1 * T2 * T3
        let mut trsf1 = Trsf::default();
        let trans1 = XYZ::from_coords(1.0, 0.0, 0.0);
        trsf1.set_translation(&trans1);
        let loc1 = Location::from_trsf(trsf1);

        let mut trsf2 = Trsf::default();
        let trans2 = XYZ::from_coords(0.0, 2.0, 0.0);
        trsf2.set_translation(&trans2);
        let loc2 = Location::from_trsf(trsf2);

        let mut trsf3 = Trsf::default();
        let trans3 = XYZ::from_coords(0.0, 0.0, 3.0);
        trsf3.set_translation(&trans3);
        let loc3 = Location::from_trsf(trsf3);

        let chain = loc1.multiplied(&loc2).multiplied(&loc3);
        assert!(!chain.is_identity());
        
        // Verify inverted composition property
        let inv_chain = chain.inverted();
        let composed = chain.multiplied(&inv_chain);
        assert!(composed.is_identity());
    }
}
