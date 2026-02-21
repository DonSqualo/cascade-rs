//! OCCT-Style Smart Pointers (Memory Management)
//!
//! This module provides `Handle<T>`, a reference-counted smart pointer wrapper
//! compatible with OpenCASCADE Technology (OCCT) patterns.
//!
//! # Overview
//!
//! The `Handle<T>` type provides:
//! - **Reference counting**: Multiple references share one allocation via `Arc<T>`
//! - **Thread-safety**: Safe to share across threads
//! - **Null handling**: Optional semantics with `is_null()` checks
//! - **Transparent access**: Implements `Deref` for seamless value access
//! - **OCCT compatibility**: `NCollection_Handle<T>` alias for OCCT code patterns
//!
//! # Example
//!
//! ```rust,ignore
//! use cascade::foundation::Handle;
//!
//! // Create a handle
//! let handle = Handle::new(42);
//! assert_eq!(handle.get(), Some(&42));
//!
//! // Clone increments reference count (cheap operation)
//! let h2 = handle.clone();
//! assert_eq!(h2.get(), Some(&42));
//!
//! // Dereferencing for transparent access
//! assert_eq!(*h2, 42);
//!
//! // Check for null
//! let null_handle: Handle<i32> = Handle::null();
//! assert!(null_handle.is_null());
//! ```

use std::ops::Deref;
use std::sync::Arc;
use std::fmt;

/// A reference-counted smart pointer for OCCT-compatible memory management.
///
/// `Handle<T>` wraps `Arc<T>` to provide:
/// - **Shared ownership**: All clones share the same underlying data
/// - **Automatic cleanup**: Data is deallocated when last Handle is dropped
/// - **Null semantics**: Can represent absence of value
/// - **Transparent access**: Derefs to `&T` automatically
///
/// # Design
///
/// This is modeled after OpenCASCADE's `opencascade::handle<T>`, but uses Rust's
/// `Arc<T>` for thread-safe, lock-free reference counting.
#[derive(Clone)]
pub struct Handle<T: ?Sized> {
    // Option<Arc<T>> allows us to have null handles
    inner: Option<Arc<T>>,
}

impl<T: ?Sized> Handle<T> {
    /// Returns `true` if this handle is null (contains no value).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let h = Handle::new(42);
    /// assert!(!h.is_null());
    ///
    /// let null_h: Handle<i32> = Handle::null();
    /// assert!(null_h.is_null());
    /// ```
    pub fn is_null(&self) -> bool {
        self.inner.is_none()
    }

    /// Returns a reference to the contained value, or `None` if null.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let h = Handle::new(vec![1, 2, 3]);
    /// assert_eq!(h.get(), Some(&vec![1, 2, 3]));
    ///
    /// let null: Handle<Vec<i32>> = Handle::null();
    /// assert_eq!(null.get(), None);
    /// ```
    pub fn get(&self) -> Option<&T> {
        self.inner.as_ref().map(|arc| arc.as_ref())
    }

    /// Returns the number of strong references to the contained value.
    ///
    /// Returns 0 if the handle is null.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let h1 = Handle::new(42);
    /// assert_eq!(h1.strong_count(), 1);
    ///
    /// let h2 = h1.clone();
    /// assert_eq!(h1.strong_count(), 2); // Both h1 and h2 share the reference
    /// ```
    pub fn strong_count(&self) -> usize {
        self.inner.as_ref().map(|arc| Arc::strong_count(arc)).unwrap_or(0)
    }

    /// Creates a new handle pointing to the same allocation (clones the Arc).
    ///
    /// This operation is O(1) and atomically increments the reference count.
    /// It's the idiomatic way to share ownership of data across threads or contexts.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let h1 = Handle::new(vec![1, 2, 3]);
    /// let h2 = h1.clone();  // Cheap operation, increments refcount
    ///
    /// // Both point to the same data
    /// assert_eq!(h1.get(), h2.get());
    /// ```
    pub fn clone(&self) -> Self {
        Handle {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Handle<T> {
    /// Creates a new handle containing the given value.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let handle = Handle::new(String::from("hello"));
    /// assert!(!handle.is_null());
    /// ```
    pub fn new(value: T) -> Self {
        Handle {
            inner: Some(Arc::new(value)),
        }
    }

    /// Creates a null handle (representing no value).
    ///
    /// Useful for optional scenarios where a handle may not contain a value.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let null: Handle<i32> = Handle::null();
    /// assert!(null.is_null());
    /// assert_eq!(null.get(), None);
    /// ```
    pub fn null() -> Self {
        Handle { inner: None }
    }
}

impl<T: ?Sized> Deref for Handle<T> {
    type Target = T;

    /// Dereferences the handle to access the contained value.
    ///
    /// # Panics
    ///
    /// Panics if the handle is null. Use `get()` for fallible access.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let h = Handle::new(42);
    /// assert_eq!(*h, 42);
    /// ```
    fn deref(&self) -> &Self::Target {
        self.inner
            .as_ref()
            .expect("Cannot dereference a null Handle")
            .as_ref()
    }
}

impl<T> Default for Handle<T> {
    /// Returns a null handle.
    fn default() -> Self {
        Handle::null()
    }
}

impl<T: fmt::Debug> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            Some(arc) => f.debug_tuple("Handle").field(arc).finish(),
            None => f.write_str("Handle(null)"),
        }
    }
}

impl<T: PartialEq> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self.get(), other.get()) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false,
        }
    }
}

impl<T: Eq> Eq for Handle<T> {}

/// OCCT-compatible type alias for `Handle<T>`.
///
/// Used when interfacing with OpenCASCADE code or maintaining naming consistency.
///
/// # Example
///
/// ```rust,ignore
/// use cascade::foundation::NCollection_Handle;
///
/// let handle: NCollection_Handle<i32> = Handle::new(42);
/// assert_eq!(handle.get(), Some(&42));
/// ```
pub type NCollection_Handle<T> = Handle<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_creation() {
        let h = Handle::new(42);
        assert!(!h.is_null());
        assert_eq!(h.get(), Some(&42));
    }

    #[test]
    fn test_handle_null() {
        let h: Handle<i32> = Handle::null();
        assert!(h.is_null());
        assert_eq!(h.get(), None);
    }

    #[test]
    fn test_handle_clone() {
        let h1 = Handle::new(vec![1, 2, 3]);
        assert_eq!(h1.strong_count(), 1);

        let h2 = h1.clone();
        assert_eq!(h1.strong_count(), 2);
        assert_eq!(h2.strong_count(), 2);

        // Both point to same data
        assert_eq!(h1.get(), h2.get());
    }

    #[test]
    fn test_handle_deref() {
        let h = Handle::new(42);
        assert_eq!(*h, 42);
    }

    #[test]
    fn test_handle_deref_struct() {
        struct Point {
            x: f64,
            y: f64,
        }

        let h = Handle::new(Point { x: 1.0, y: 2.0 });
        assert_eq!(h.x, 1.0);
        assert_eq!(h.y, 2.0);
    }

    #[test]
    #[should_panic(expected = "Cannot dereference a null Handle")]
    fn test_handle_deref_null_panics() {
        let h: Handle<i32> = Handle::null();
        let _ = *h;
    }

    #[test]
    fn test_handle_strong_count() {
        let h1 = Handle::new(42);
        assert_eq!(h1.strong_count(), 1);

        let h2 = h1.clone();
        assert_eq!(h1.strong_count(), 2);

        let h3 = h2.clone();
        assert_eq!(h1.strong_count(), 3);
        assert_eq!(h2.strong_count(), 3);
        assert_eq!(h3.strong_count(), 3);

        drop(h2);
        assert_eq!(h1.strong_count(), 2);
    }

    #[test]
    fn test_handle_strong_count_null() {
        let h: Handle<i32> = Handle::null();
        assert_eq!(h.strong_count(), 0);
    }

    #[test]
    fn test_nCollection_handle_alias() {
        let h: NCollection_Handle<String> = Handle::new(String::from("test"));
        assert_eq!(h.get(), Some(&String::from("test")));
    }

    #[test]
    fn test_handle_default() {
        let h: Handle<i32> = Default::default();
        assert!(h.is_null());
    }

    #[test]
    fn test_handle_debug() {
        let h = Handle::new(42);
        let debug_str = format!("{:?}", h);
        assert!(debug_str.contains("Handle"));
        assert!(debug_str.contains("42"));

        let null_h: Handle<i32> = Handle::null();
        let null_debug = format!("{:?}", null_h);
        assert_eq!(null_debug, "Handle(null)");
    }

    #[test]
    fn test_handle_equality() {
        let h1 = Handle::new(42);
        let h2 = Handle::new(42);
        let h3 = h1.clone();

        assert_eq!(h1, h3);
        assert_eq!(h1, h2); // Same value, different allocations

        let h4: Handle<i32> = Handle::null();
        let h5: Handle<i32> = Handle::null();
        assert_eq!(h4, h5); // Both null

        assert_ne!(h1, h4);
    }

    #[test]
    fn test_handle_with_complex_types() {
        #[derive(Clone, PartialEq, Debug)]
        struct ComplexData {
            name: String,
            values: Vec<f64>,
        }

        let data = ComplexData {
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0],
        };

        let h1 = Handle::new(data.clone());
        let h2 = h1.clone();

        assert_eq!(h1, h2);
        assert_eq!(h1.get().unwrap().name, "test");
        assert_eq!(h1.get().unwrap().values.len(), 3);
    }

    #[test]
    fn test_handle_with_trait_objects() {
        trait MyTrait {
            fn value(&self) -> i32;
        }

        struct Impl(i32);

        impl MyTrait for Impl {
            fn value(&self) -> i32 {
                self.0
            }
        }

        let boxed: Box<dyn MyTrait> = Box::new(Impl(42));
        let h = Handle::new(boxed);
        assert!(!h.is_null());
        assert_eq!(h.get().unwrap().value(), 42);
    }

    #[test]
    fn test_handle_clone_reference_sharing() {
        let h1 = Handle::new(vec![1, 2, 3]);

        let handles: Vec<_> = (0..5).map(|_| h1.clone()).collect();

        // All handles point to the same data
        for h in &handles {
            assert_eq!(h.get(), Some(&vec![1, 2, 3]));
        }

        // Reference count reflects all clones
        assert_eq!(h1.strong_count(), 6); // h1 + 5 clones in handles vec
    }

    #[test]
    fn test_handle_null_clone() {
        let h1: Handle<i32> = Handle::null();
        let h2 = h1.clone();

        assert!(h2.is_null());
        assert_eq!(h2.get(), None);
    }

    #[test]
    fn test_handle_multithread_safety() {
        use std::thread;

        let h = Handle::new(42);

        // Move handle to multiple threads
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let handle = h.clone();
                thread::spawn(move || handle.get().copied())
            })
            .collect();

        // All threads successfully accessed the value
        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(result, Some(42));
        }
    }
}
