//! Runtime Type Information (RTTI) System
//!
//! Provides runtime type information for CASCADE-RS types, enabling dynamic type checking
//! and safe downcasting. This system is essential for heterogeneous shape collections
//! and polymorphic operations.
//!
//! # Architecture
//!
//! The RTTI system consists of three core components:
//!
//! 1. **TypeInfo** - Immutable type metadata containing:
//!    - `type_name`: Human-readable type name (e.g., "Solid")
//!    - `type_id`: Unique identifier via Rust's `TypeId` for safe comparisons
//!
//! 2. **Typed Trait** - Base trait implemented by all types that support RTTI:
//!    - `type_info() -> TypeInfo` - Returns metadata (dyn-safe)
//!    - `is_type_id(TypeId) -> bool` - Type checking by id (dyn-safe)
//!
//! 3. **TypedExt Trait** - Generic extension methods for concrete types:
//!    - `is_type<T>() -> bool` - Check if object is of type T
//!    - `downcast_ref<T>(&self) -> Option<&T>` - Safe reference downcast
//!    - `downcast_mut<T>(&mut self) -> Option<&mut T>` - Safe mutable downcast
//!
//! # Example
//!
//! ```ignore
//! let solid = Solid::new(...);
//! assert!(solid.is_type::<Solid>()); // Via TypedExt
//! assert!(!solid.is_type::<Face>());
//!
//! if let Some(face_ref) = solid.downcast_ref::<Face>() {
//!     // Work with face
//! }
//!
//! // With trait objects (dyn Typed):
//! let shapes: Vec<Box<dyn Typed>> = vec![...];
//! for shape in &shapes {
//!     if shape.is_type_id(TypeId::of::<Face>()) {
//!         // This is a Face
//!     }
//! }
//! ```

use std::any::TypeId;
use std::fmt;

/// Runtime type information for a CASCADE-RS type
///
/// TypeInfo is a lightweight, immutable descriptor containing the type's name
/// and its unique `TypeId`. It can be cheaply copied and compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeInfo {
    /// Human-readable name of the type (e.g., "Solid", "Face")
    type_name: &'static str,
    /// Unique identifier for this type (from Rust's TypeId)
    type_id: TypeId,
}

impl TypeInfo {
    /// Create new TypeInfo for a type
    ///
    /// # Arguments
    /// * `type_name` - Human-readable name of the type
    /// * `type_id` - Unique identifier (typically from std::any::TypeId::of::<T>())
    pub fn new(type_name: &'static str, type_id: TypeId) -> Self {
        TypeInfo { type_name, type_id }
    }

    /// Get the human-readable type name
    pub fn name(&self) -> &'static str {
        self.type_name
    }

    /// Get the unique type identifier
    pub fn id(&self) -> TypeId {
        self.type_id
    }

    /// Check if this type matches another type's TypeId
    pub fn is(&self, other: TypeId) -> bool {
        self.type_id == other
    }

    /// Check if this type matches a specific generic type T
    pub fn is_type<T: 'static>(&self) -> bool {
        self.type_id == TypeId::of::<T>()
    }
}

impl fmt::Display for TypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.type_name)
    }
}

/// Base trait for types that support runtime type information
///
/// This is the dyn-compatible base trait. It provides methods to query type information
/// and is safe to use as a trait object (`dyn Typed`).
///
/// For concrete types, you can also use the `TypedExt` trait for generic downcast methods.
pub trait Typed {
    /// Get the TypeInfo for this object's type
    fn type_info(&self) -> TypeInfo;

    /// Check if this object is of the given type by TypeId
    /// This is dyn-safe and can be called through a trait object.
    fn is_type_id(&self, type_id: TypeId) -> bool {
        self.type_info().is(type_id)
    }
}

/// Extension trait for generic type checking and downcasting
///
/// This trait provides generic methods for concrete types that implement `Typed`.
/// It is automatically implemented for all concrete types that implement `Typed`.
///
/// Note: These methods are only available on concrete types, not on trait objects.
/// For trait objects (`dyn Typed`), use `is_type_id()` which is available on the
/// base `Typed` trait.
///
/// # Example
///
/// ```ignore
/// use cascade_rs::foundation::TypedExt;
///
/// let solid = MockSolid { volume: 100.0 };
///
/// // These methods are available on concrete types
/// assert!(solid.is_type::<MockSolid>());
/// if let Some(solid_ref) = solid.downcast_ref::<MockSolid>() {
///     println!("Volume: {}", solid_ref.volume);
/// }
///
/// // For trait objects, use the base Typed trait
/// let typed: &dyn Typed = &solid;
/// if typed.is_type_id(TypeId::of::<MockSolid>()) {
///     // This is a MockSolid
/// }
/// ```
pub trait TypedExt: Typed {
    /// Check if this object is of type T
    ///
    /// # Example
    /// ```ignore
    /// if solid.is_type::<Solid>() {
    ///     println!("This is a solid");
    /// }
    /// ```
    fn is_type<T: 'static>(&self) -> bool {
        self.is_type_id(TypeId::of::<T>())
    }

    /// Safely downcast by immutable reference to type T
    ///
    /// Returns `Some(&T)` if the object is of type T, otherwise `None`.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(solid_ref) = shape.downcast_ref::<Solid>() {
    ///     println!("Solid volume: {}", solid_ref.volume());
    /// }
    /// ```
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;

    /// Safely downcast by mutable reference to type T
    ///
    /// Returns `Some(&mut T)` if the object is of type T, otherwise `None`.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(solid_mut) = shape_mut.downcast_mut::<Solid>() {
    ///     solid_mut.transform(&matrix);
    /// }
    /// ```
    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T>;
}

/// Automatically implement TypedExt for all concrete types that implement Typed
impl<T: Typed> TypedExt for T {
    fn downcast_ref<U: 'static>(&self) -> Option<&U> {
        if self.is_type::<U>() {
            // SAFETY: We've verified the type match above using TypeId
            unsafe { Some(&*(self as *const T as *const U)) }
        } else {
            None
        }
    }

    fn downcast_mut<U: 'static>(&mut self) -> Option<&mut U> {
        if self.is_type::<U>() {
            // SAFETY: We've verified the type match above using TypeId
            unsafe { Some(&mut *(self as *mut T as *mut U)) }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock types for testing
    #[derive(Debug, Clone)]
    struct MockSolid {
        volume: f64,
    }

    #[derive(Debug, Clone)]
    struct MockFace {
        area: f64,
    }

    #[derive(Debug, Clone)]
    struct MockEdge {
        length: f64,
    }

    // Implement Typed for mock types
    impl Typed for MockSolid {
        fn type_info(&self) -> TypeInfo {
            TypeInfo::new("MockSolid", TypeId::of::<MockSolid>())
        }
    }

    impl Typed for MockFace {
        fn type_info(&self) -> TypeInfo {
            TypeInfo::new("MockFace", TypeId::of::<MockFace>())
        }
    }

    impl Typed for MockEdge {
        fn type_info(&self) -> TypeInfo {
            TypeInfo::new("MockEdge", TypeId::of::<MockEdge>())
        }
    }

    #[test]
    fn test_typeinfo_creation() {
        let type_id = TypeId::of::<MockSolid>();
        let info = TypeInfo::new("MockSolid", type_id);

        assert_eq!(info.name(), "MockSolid");
        assert_eq!(info.id(), type_id);
    }

    #[test]
    fn test_typeinfo_equality() {
        let type_id_1 = TypeId::of::<MockSolid>();
        let info1 = TypeInfo::new("MockSolid", type_id_1);
        let info2 = TypeInfo::new("MockSolid", type_id_1);

        assert_eq!(info1, info2);
    }

    #[test]
    fn test_typeinfo_inequality() {
        let type_id_solid = TypeId::of::<MockSolid>();
        let type_id_face = TypeId::of::<MockFace>();
        let info_solid = TypeInfo::new("MockSolid", type_id_solid);
        let info_face = TypeInfo::new("MockFace", type_id_face);

        assert_ne!(info_solid, info_face);
    }

    #[test]
    fn test_typeinfo_is_type() {
        let info = TypeInfo::new("MockSolid", TypeId::of::<MockSolid>());

        assert!(info.is_type::<MockSolid>());
        assert!(!info.is_type::<MockFace>());
    }

    #[test]
    fn test_typeinfo_is_method() {
        let info = TypeInfo::new("MockSolid", TypeId::of::<MockSolid>());
        let solid_id = TypeId::of::<MockSolid>();
        let face_id = TypeId::of::<MockFace>();

        assert!(info.is(solid_id));
        assert!(!info.is(face_id));
    }

    #[test]
    fn test_typeinfo_display() {
        let info = TypeInfo::new("MockSolid", TypeId::of::<MockSolid>());
        assert_eq!(format!("{}", info), "MockSolid");
    }

    #[test]
    fn test_typed_type_info() {
        let solid = MockSolid { volume: 100.0 };
        let info = solid.type_info();

        assert_eq!(info.name(), "MockSolid");
        assert_eq!(info.id(), TypeId::of::<MockSolid>());
    }

    #[test]
    fn test_typed_is_type_positive() {
        let solid = MockSolid { volume: 100.0 };

        assert!(solid.is_type::<MockSolid>());
    }

    #[test]
    fn test_typed_is_type_negative() {
        let solid = MockSolid { volume: 100.0 };

        assert!(!solid.is_type::<MockFace>());
        assert!(!solid.is_type::<MockEdge>());
    }

    #[test]
    fn test_typed_is_type_id() {
        let solid = MockSolid { volume: 100.0 };
        let solid_id = TypeId::of::<MockSolid>();
        let face_id = TypeId::of::<MockFace>();

        assert!(solid.is_type_id(solid_id));
        assert!(!solid.is_type_id(face_id));
    }

    #[test]
    fn test_downcast_ref_success() {
        let solid = MockSolid { volume: 42.0 };

        if let Some(solid_ref) = solid.downcast_ref::<MockSolid>() {
            assert_eq!(solid_ref.volume, 42.0);
        } else {
            panic!("Downcast should have succeeded");
        }
    }

    #[test]
    fn test_downcast_ref_failure() {
        let solid = MockSolid { volume: 42.0 };
        let result = solid.downcast_ref::<MockFace>();

        assert!(result.is_none());
    }

    #[test]
    fn test_dyn_trait_object_type_info() {
        let solid = MockSolid { volume: 100.0 };
        let typed: &dyn Typed = &solid;

        let info = typed.type_info();
        assert_eq!(info.name(), "MockSolid");
    }

    #[test]
    fn test_dyn_trait_object_is_type_id() {
        let solid = MockSolid { volume: 100.0 };
        let typed: &dyn Typed = &solid;

        assert!(typed.is_type_id(TypeId::of::<MockSolid>()));
        assert!(!typed.is_type_id(TypeId::of::<MockFace>()));
    }

    #[test]
    fn test_downcast_mut_success() {
        let mut solid = MockSolid { volume: 42.0 };

        if let Some(solid_mut) = solid.downcast_mut::<MockSolid>() {
            solid_mut.volume = 100.0;
            assert_eq!(solid_mut.volume, 100.0);
        } else {
            panic!("Downcast should have succeeded");
        }
    }

    #[test]
    fn test_downcast_mut_failure() {
        let mut solid = MockSolid { volume: 42.0 };
        let result = solid.downcast_mut::<MockFace>();

        assert!(result.is_none());
    }

    #[test]
    fn test_heterogeneous_collection() {
        // Create a heterogeneous collection using dyn trait objects
        let shapes: Vec<Box<dyn Typed>> = vec![
            Box::new(MockSolid { volume: 100.0 }),
            Box::new(MockFace { area: 50.0 }),
            Box::new(MockEdge { length: 10.0 }),
        ];

        // Count by type using is_type_id (which works with dyn objects)
        let mut solid_count = 0;
        let mut face_count = 0;
        let mut edge_count = 0;

        let solid_id = TypeId::of::<MockSolid>();
        let face_id = TypeId::of::<MockFace>();
        let edge_id = TypeId::of::<MockEdge>();

        for shape in &shapes {
            if shape.is_type_id(solid_id) {
                solid_count += 1;
            } else if shape.is_type_id(face_id) {
                face_count += 1;
            } else if shape.is_type_id(edge_id) {
                edge_count += 1;
            }
        }

        assert_eq!(solid_count, 1);
        assert_eq!(face_count, 1);
        assert_eq!(edge_count, 1);
    }

    #[test]
    fn test_concrete_type_downcast_chain() {
        // Test downcasting on concrete types
        let solids: Vec<MockSolid> = vec![
            MockSolid { volume: 100.0 },
            MockSolid { volume: 200.0 },
        ];

        // Use TypedExt to downcast concrete types
        let mut total_volume = 0.0;
        for solid in &solids {
            if let Some(solid_ref) = solid.downcast_ref::<MockSolid>() {
                total_volume += solid_ref.volume;
            }
        }

        assert_eq!(total_volume, 300.0);
    }

    #[test]
    fn test_concrete_type_mut_downcast() {
        // Test mutable downcasting on concrete types
        let mut solids: Vec<MockSolid> = vec![
            MockSolid { volume: 100.0 },
            MockSolid { volume: 50.0 },
        ];

        // Mutate using TypedExt
        for solid in &mut solids {
            if let Some(solid_mut) = solid.downcast_mut::<MockSolid>() {
                solid_mut.volume *= 2.0;
            }
        }

        // Verify mutation
        assert_eq!(solids[0].volume, 200.0);
        assert_eq!(solids[1].volume, 100.0);
    }

    #[test]
    fn test_typed_ext_auto_implementation() {
        // Verify that TypedExt is automatically implemented
        let solid = MockSolid { volume: 75.0 };

        // These methods should work via TypedExt
        assert!(solid.is_type::<MockSolid>());
        assert_eq!(solid.downcast_ref::<MockSolid>().unwrap().volume, 75.0);
    }
}
