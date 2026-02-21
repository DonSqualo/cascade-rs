//! Foundation Classes for CASCADE-RS
//!
//! Provides OCCT-compatible primitive types, collection classes, mathematical utilities,
//! and smart pointer wrappers for interoperability with OpenCASCADE-based systems.

#![allow(non_camel_case_types)]

pub mod math;
pub mod primitives;
pub mod collections;
pub mod handles;
pub mod rtti;

pub use math::{
    Matrix3x3, Matrix4x4,
    solve_linear_system_3x3, eigenvalues_3x3,
    VectorOps,
};

pub use primitives::{
    Standard_Boolean, Standard_Integer, Standard_Real, Standard_CString,
};

pub use collections::{
    Array1, List, Map, Set,
    TColStd_Array1, TColStd_List, TColStd_Map, TColStd_Set,
};

pub use handles::{
    Handle, NCollection_Handle,
};

pub use rtti::{
    TypeInfo, Typed, TypedExt,
};
