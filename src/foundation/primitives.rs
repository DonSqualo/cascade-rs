//! OCCT-Compatible Primitive Type Wrappers
//!
//! Provides explicit primitive type wrappers for OCCT compatibility:
//! - `Standard_Boolean` - wrapper around bool
//! - `Standard_Integer` - wrapper around i32
//! - `Standard_Real` - wrapper around f64
//! - `Standard_CString` - wrapper around String

use std::fmt;

/// OCCT-compatible wrapper for boolean values
/// Equivalent to OCCT's `Standard_Boolean`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Standard_Boolean(bool);

impl Standard_Boolean {
    /// Create a new Standard_Boolean
    pub fn new(value: bool) -> Self {
        Standard_Boolean(value)
    }

    /// Get the inner boolean value
    pub fn value(self) -> bool {
        self.0
    }

    /// Convert to bool (alias for value())
    pub fn to_bool(self) -> bool {
        self.0
    }

    /// Check if the value is true
    pub fn is_true(self) -> bool {
        self.0
    }

    /// Check if the value is false
    pub fn is_false(self) -> bool {
        !self.0
    }
}

impl From<bool> for Standard_Boolean {
    fn from(value: bool) -> Self {
        Standard_Boolean(value)
    }
}

impl From<Standard_Boolean> for bool {
    fn from(value: Standard_Boolean) -> Self {
        value.0
    }
}

impl fmt::Display for Standard_Boolean {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Standard_Boolean {
    fn default() -> Self {
        Standard_Boolean(false)
    }
}

/// OCCT-compatible wrapper for 32-bit signed integers
/// Equivalent to OCCT's `Standard_Integer`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Standard_Integer(i32);

impl Standard_Integer {
    /// Create a new Standard_Integer
    pub fn new(value: i32) -> Self {
        Standard_Integer(value)
    }

    /// Get the inner integer value
    pub fn value(self) -> i32 {
        self.0
    }

    /// Convert to i32 (alias for value())
    pub fn to_i32(self) -> i32 {
        self.0
    }

    /// Check if the integer is positive
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Check if the integer is negative
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Check if the integer is zero
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Get the absolute value
    pub fn abs(self) -> Standard_Integer {
        Standard_Integer(self.0.abs())
    }
}

impl From<i32> for Standard_Integer {
    fn from(value: i32) -> Self {
        Standard_Integer(value)
    }
}

impl From<Standard_Integer> for i32 {
    fn from(value: Standard_Integer) -> Self {
        value.0
    }
}

impl fmt::Display for Standard_Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Standard_Integer {
    fn default() -> Self {
        Standard_Integer(0)
    }
}

/// OCCT-compatible wrapper for 64-bit floating point numbers
/// Equivalent to OCCT's `Standard_Real`
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Standard_Real(f64);

impl Standard_Real {
    /// Create a new Standard_Real
    pub fn new(value: f64) -> Self {
        Standard_Real(value)
    }

    /// Get the inner float value
    pub fn value(self) -> f64 {
        self.0
    }

    /// Convert to f64 (alias for value())
    pub fn to_f64(self) -> f64 {
        self.0
    }

    /// Check if the value is positive
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    /// Check if the value is negative
    pub fn is_negative(self) -> bool {
        self.0 < 0.0
    }

    /// Check if the value is zero (with tolerance)
    pub fn is_zero_with_tolerance(self, tolerance: f64) -> bool {
        self.0.abs() < tolerance
    }

    /// Get the absolute value
    pub fn abs(self) -> Standard_Real {
        Standard_Real(self.0.abs())
    }

    /// Get the square root
    pub fn sqrt(self) -> Standard_Real {
        Standard_Real(self.0.sqrt())
    }

    /// Square the value
    pub fn square(self) -> Standard_Real {
        Standard_Real(self.0 * self.0)
    }

    /// Raise to a power
    pub fn pow(self, power: i32) -> Standard_Real {
        Standard_Real(self.0.powi(power))
    }
}

impl From<f64> for Standard_Real {
    fn from(value: f64) -> Self {
        Standard_Real(value)
    }
}

impl From<Standard_Real> for f64 {
    fn from(value: Standard_Real) -> Self {
        value.0
    }
}

impl From<i32> for Standard_Real {
    fn from(value: i32) -> Self {
        Standard_Real(value as f64)
    }
}

impl From<Standard_Integer> for Standard_Real {
    fn from(value: Standard_Integer) -> Self {
        Standard_Real(value.value() as f64)
    }
}

impl fmt::Display for Standard_Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Standard_Real {
    fn default() -> Self {
        Standard_Real(0.0)
    }
}

/// OCCT-compatible wrapper for C-style strings
/// Equivalent to OCCT's `Standard_CString`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Standard_CString(String);

impl Standard_CString {
    /// Create a new Standard_CString
    pub fn new(value: impl Into<String>) -> Self {
        Standard_CString(value.into())
    }

    /// Get the inner string value
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Convert to String
    pub fn to_string_value(&self) -> String {
        self.0.clone()
    }

    /// Get the length of the string
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the string is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if the string is null (empty string)
    pub fn is_null(&self) -> bool {
        self.0.is_empty()
    }

    /// Convert to uppercase
    pub fn to_uppercase(&self) -> Standard_CString {
        Standard_CString(self.0.to_uppercase())
    }

    /// Convert to lowercase
    pub fn to_lowercase(&self) -> Standard_CString {
        Standard_CString(self.0.to_lowercase())
    }

    /// Trim whitespace
    pub fn trim(&self) -> Standard_CString {
        Standard_CString(self.0.trim().to_string())
    }
}

impl From<String> for Standard_CString {
    fn from(value: String) -> Self {
        Standard_CString(value)
    }
}

impl From<&str> for Standard_CString {
    fn from(value: &str) -> Self {
        Standard_CString(value.to_string())
    }
}

impl From<Standard_CString> for String {
    fn from(value: Standard_CString) -> Self {
        value.0
    }
}

impl fmt::Display for Standard_CString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Standard_CString {
    fn default() -> Self {
        Standard_CString(String::new())
    }
}

impl AsRef<str> for Standard_CString {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_boolean_creation() {
        let true_bool = Standard_Boolean::new(true);
        let false_bool = Standard_Boolean::new(false);

        assert_eq!(true_bool.value(), true);
        assert_eq!(false_bool.value(), false);
    }

    #[test]
    fn test_standard_boolean_from_bool() {
        let bool_val = true;
        let std_bool: Standard_Boolean = bool_val.into();
        assert_eq!(std_bool.value(), true);

        let false_val = false;
        let std_bool: Standard_Boolean = false_val.into();
        assert_eq!(std_bool.value(), false);
    }

    #[test]
    fn test_standard_boolean_to_bool() {
        let std_bool = Standard_Boolean::new(true);
        let bool_val: bool = std_bool.into();
        assert_eq!(bool_val, true);
    }

    #[test]
    fn test_standard_boolean_predicates() {
        let true_bool = Standard_Boolean::new(true);
        let false_bool = Standard_Boolean::new(false);

        assert!(true_bool.is_true());
        assert!(!true_bool.is_false());

        assert!(!false_bool.is_true());
        assert!(false_bool.is_false());
    }

    #[test]
    fn test_standard_boolean_display() {
        let true_bool = Standard_Boolean::new(true);
        let false_bool = Standard_Boolean::new(false);

        assert_eq!(format!("{}", true_bool), "true");
        assert_eq!(format!("{}", false_bool), "false");
    }

    #[test]
    fn test_standard_boolean_default() {
        let default_bool = Standard_Boolean::default();
        assert_eq!(default_bool.value(), false);
    }

    #[test]
    fn test_standard_integer_creation() {
        let int_val = Standard_Integer::new(42);
        assert_eq!(int_val.value(), 42);
    }

    #[test]
    fn test_standard_integer_from_i32() {
        let int_val: i32 = 100;
        let std_int: Standard_Integer = int_val.into();
        assert_eq!(std_int.value(), 100);
    }

    #[test]
    fn test_standard_integer_to_i32() {
        let std_int = Standard_Integer::new(200);
        let int_val: i32 = std_int.into();
        assert_eq!(int_val, 200);
    }

    #[test]
    fn test_standard_integer_predicates() {
        let positive = Standard_Integer::new(10);
        let negative = Standard_Integer::new(-10);
        let zero = Standard_Integer::new(0);

        assert!(positive.is_positive());
        assert!(!positive.is_negative());
        assert!(!positive.is_zero());

        assert!(!negative.is_positive());
        assert!(negative.is_negative());
        assert!(!negative.is_zero());

        assert!(!zero.is_positive());
        assert!(!zero.is_negative());
        assert!(zero.is_zero());
    }

    #[test]
    fn test_standard_integer_abs() {
        let positive = Standard_Integer::new(10);
        let negative = Standard_Integer::new(-10);

        assert_eq!(positive.abs().value(), 10);
        assert_eq!(negative.abs().value(), 10);
    }

    #[test]
    fn test_standard_integer_display() {
        let int_val = Standard_Integer::new(42);
        assert_eq!(format!("{}", int_val), "42");
    }

    #[test]
    fn test_standard_integer_default() {
        let default_int = Standard_Integer::default();
        assert_eq!(default_int.value(), 0);
    }

    #[test]
    fn test_standard_real_creation() {
        let real_val = Standard_Real::new(3.14);
        assert!((real_val.value() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_standard_real_from_f64() {
        let f_val: f64 = 2.718;
        let std_real: Standard_Real = f_val.into();
        assert!((std_real.value() - 2.718).abs() < 1e-10);
    }

    #[test]
    fn test_standard_real_to_f64() {
        let std_real = Standard_Real::new(1.414);
        let f_val: f64 = std_real.into();
        assert!((f_val - 1.414).abs() < 1e-10);
    }

    #[test]
    fn test_standard_real_from_i32() {
        let int_val: i32 = 42;
        let std_real: Standard_Real = int_val.into();
        assert_eq!(std_real.value(), 42.0);
    }

    #[test]
    fn test_standard_real_from_standard_integer() {
        let std_int = Standard_Integer::new(100);
        let std_real: Standard_Real = std_int.into();
        assert_eq!(std_real.value(), 100.0);
    }

    #[test]
    fn test_standard_real_predicates() {
        let positive = Standard_Real::new(5.5);
        let negative = Standard_Real::new(-5.5);
        let zero = Standard_Real::new(0.0);

        assert!(positive.is_positive());
        assert!(!positive.is_negative());

        assert!(!negative.is_positive());
        assert!(negative.is_negative());

        assert!(!zero.is_positive());
        assert!(!zero.is_negative());
    }

    #[test]
    fn test_standard_real_is_zero_with_tolerance() {
        let almost_zero = Standard_Real::new(1e-7);
        let definitely_not_zero = Standard_Real::new(0.1);

        assert!(almost_zero.is_zero_with_tolerance(1e-6));
        assert!(!definitely_not_zero.is_zero_with_tolerance(1e-6));
    }

    #[test]
    fn test_standard_real_abs() {
        let positive = Standard_Real::new(5.5);
        let negative = Standard_Real::new(-5.5);

        assert!((positive.abs().value() - 5.5).abs() < 1e-10);
        assert!((negative.abs().value() - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_standard_real_sqrt() {
        let val = Standard_Real::new(4.0);
        assert!((val.sqrt().value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_real_square() {
        let val = Standard_Real::new(3.0);
        assert!((val.square().value() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_real_pow() {
        let val = Standard_Real::new(2.0);
        assert!((val.pow(3).value() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_real_display() {
        let real_val = Standard_Real::new(3.14);
        assert!(format!("{}", real_val).contains("3.14"));
    }

    #[test]
    fn test_standard_real_default() {
        let default_real = Standard_Real::default();
        assert_eq!(default_real.value(), 0.0);
    }

    #[test]
    fn test_standard_cstring_creation() {
        let cstr = Standard_CString::new("hello");
        assert_eq!(cstr.value(), "hello");
    }

    #[test]
    fn test_standard_cstring_from_string() {
        let string_val = String::from("world");
        let cstr: Standard_CString = string_val.into();
        assert_eq!(cstr.value(), "world");
    }

    #[test]
    fn test_standard_cstring_from_str() {
        let str_val = "test";
        let cstr: Standard_CString = str_val.into();
        assert_eq!(cstr.value(), "test");
    }

    #[test]
    fn test_standard_cstring_to_string() {
        let cstr = Standard_CString::new("cascade");
        let string_val: String = cstr.into();
        assert_eq!(string_val, "cascade");
    }

    #[test]
    fn test_standard_cstring_len() {
        let cstr = Standard_CString::new("hello");
        assert_eq!(cstr.len(), 5);
    }

    #[test]
    fn test_standard_cstring_is_empty() {
        let empty = Standard_CString::new("");
        let non_empty = Standard_CString::new("text");

        assert!(empty.is_empty());
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_standard_cstring_is_null() {
        let null = Standard_CString::new("");
        let non_null = Standard_CString::new("text");

        assert!(null.is_null());
        assert!(!non_null.is_null());
    }

    #[test]
    fn test_standard_cstring_to_uppercase() {
        let cstr = Standard_CString::new("hello");
        assert_eq!(cstr.to_uppercase().value(), "HELLO");
    }

    #[test]
    fn test_standard_cstring_to_lowercase() {
        let cstr = Standard_CString::new("HELLO");
        assert_eq!(cstr.to_lowercase().value(), "hello");
    }

    #[test]
    fn test_standard_cstring_trim() {
        let cstr = Standard_CString::new("  hello world  ");
        assert_eq!(cstr.trim().value(), "hello world");
    }

    #[test]
    fn test_standard_cstring_display() {
        let cstr = Standard_CString::new("cascade-rs");
        assert_eq!(format!("{}", cstr), "cascade-rs");
    }

    #[test]
    fn test_standard_cstring_default() {
        let default_cstr = Standard_CString::default();
        assert!(default_cstr.is_empty());
    }

    #[test]
    fn test_standard_cstring_as_ref() {
        let cstr = Standard_CString::new("test");
        let str_ref: &str = cstr.as_ref();
        assert_eq!(str_ref, "test");
    }

    #[test]
    fn test_conversions_chain() {
        // Test converting between types
        let int_val: i32 = 42;
        let std_int: Standard_Integer = int_val.into();
        let std_real: Standard_Real = std_int.into();

        assert_eq!(std_real.value(), 42.0);
    }

    #[test]
    fn test_standard_types_clone_copy() {
        let bool_val = Standard_Boolean::new(true);
        let bool_clone = bool_val.clone();
        let bool_copy = bool_val;

        assert_eq!(bool_clone, bool_copy);
        assert_eq!(bool_clone, bool_val);

        let int_val = Standard_Integer::new(100);
        let int_clone = int_val.clone();
        let int_copy = int_val;

        assert_eq!(int_clone, int_copy);
        assert_eq!(int_clone, int_val);

        let real_val = Standard_Real::new(3.14);
        let real_clone = real_val.clone();
        let real_copy = real_val;

        assert!((real_clone.value() - real_copy.value()).abs() < 1e-10);
    }
}
