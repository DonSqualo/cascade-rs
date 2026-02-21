//! Precision constants for geometric comparisons.
//!
//! Port of OCCT's Precision package.
//! Source: src/FoundationClasses/TKernel/Precision/Precision.hxx
//!
//! These values are the ground truth - do not change without
//! understanding the implications for all OCCT algorithms.

/// Angular tolerance for checking equality of angles (radians).
/// Used for parallelism checks on vectors.
/// Value: 1.0e-12
pub const ANGULAR: f64 = 1.0e-12;

/// Confusion tolerance for checking coincidence of two points in real space.
/// Two points are coincident if their distance < CONFUSION.
/// Value: 1.0e-7
pub const CONFUSION: f64 = 1.0e-7;

/// Square of CONFUSION for performance.
pub const SQUARE_CONFUSION: f64 = CONFUSION * CONFUSION;

/// Computational tolerance at machine epsilon level.
/// For low-level numerical comparisons, NOT geometric comparisons.
pub const COMPUTATIONAL: f64 = f64::EPSILON;

/// Square of COMPUTATIONAL for performance.
pub const SQUARE_COMPUTATIONAL: f64 = COMPUTATIONAL * COMPUTATIONAL;

/// Intersection tolerance for iterative intersection algorithms.
/// Value: CONFUSION / 100 = 1.0e-9
pub const INTERSECTION: f64 = CONFUSION * 0.01;

/// Approximation tolerance for approximation algorithms.
/// Value: CONFUSION * 10 = 1.0e-6
pub const APPROXIMATION: f64 = CONFUSION * 10.0;

/// Convert real space precision to parametric space precision.
/// Returns p / t where t is mean tangent length.
#[inline]
pub const fn parametric(p: f64, t: f64) -> f64 {
    p / t
}

/// Default parametric confusion (assumes tangent length ~100).
/// Value: CONFUSION * 100 = 1.0e-5
pub const PARAMETRIC_CONFUSION: f64 = CONFUSION * 100.0;

/// gp::Resolution() - fundamental geometric resolution.
/// Used for zero-length checks in normalization.
/// Value: DBL_MIN (~2.2e-308)
/// 
/// NOTE: This is different from CONFUSION (1e-7).
/// Resolution is for numerical zero checks.
/// Confusion is for geometric tolerance.
pub const RESOLUTION: f64 = f64::MIN_POSITIVE;  // DBL_MIN

/// "Infinite" value for algorithms that need infinity bounds.
/// Value: 1.0e100 (not f64::INFINITY to avoid NaN in arithmetic)
pub const INFINITE: f64 = 1.0e100;

/// Check if a value is considered infinite.
#[inline]
pub const fn is_infinite(value: f64) -> bool {
    value.abs() >= INFINITE * 0.5
}

/// Check if a value is considered positive infinite.
#[inline]
pub const fn is_positive_infinite(value: f64) -> bool {
    value >= INFINITE * 0.5
}

/// Check if a value is considered negative infinite.
#[inline]
pub const fn is_negative_infinite(value: f64) -> bool {
    value <= -INFINITE * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_values() {
        // These are the OCCT ground truth values
        assert_eq!(ANGULAR, 1.0e-12);
        assert_eq!(CONFUSION, 1.0e-7);
        assert_eq!(INTERSECTION, 1.0e-9);
        assert_eq!(APPROXIMATION, 1.0e-6);
    }

    #[test]
    fn test_infinite() {
        assert!(is_infinite(INFINITE));
        assert!(is_infinite(-INFINITE));
        assert!(!is_infinite(1.0e99));
        assert!(is_positive_infinite(INFINITE));
        assert!(!is_positive_infinite(-INFINITE));
        assert!(is_negative_infinite(-INFINITE));
    }
}
