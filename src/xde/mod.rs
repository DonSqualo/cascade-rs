//! XDE (Extended Data Exchange) - Metadata attributes for shapes
//! 
//! Supports storing name, color, layer, and material information alongside geometry.
//! Also supports hierarchical assembly structures with part instances and transforms.

use serde::{Deserialize, Serialize};
use crate::brep::Solid;

/// Shape attributes for XDE metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShapeAttributes {
    /// Optional name identifier
    pub name: Option<String>,
    /// Optional RGB color [r, g, b] with values in 0.0..=1.0
    pub color: Option<[f64; 3]>,
    /// Optional layer identifier
    pub layer: Option<String>,
    /// Optional material identifier
    pub material: Option<String>,
}

impl ShapeAttributes {
    /// Create a new empty set of attributes
    pub fn new() -> Self {
        Self {
            name: None,
            color: None,
            layer: None,
            material: None,
        }
    }

    /// Create attributes with just a name
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            color: None,
            layer: None,
            material: None,
        }
    }

    /// Check if any attribute is set
    pub fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.color.is_none()
            && self.layer.is_none()
            && self.material.is_none()
    }
}

/// Set the name attribute of a solid
pub fn set_shape_name(solid: &mut Solid, name: &str) {
    solid.attributes.name = Some(name.to_string());
}

/// Get the name attribute of a solid
pub fn get_shape_name(solid: &Solid) -> Option<&str> {
    solid.attributes.name.as_deref()
}

/// Set all attributes of a solid
pub fn set_shape_attributes(solid: &mut Solid, attrs: &ShapeAttributes) {
    solid.attributes = attrs.clone();
}

/// Get all attributes of a solid
pub fn get_shape_attributes(solid: &Solid) -> ShapeAttributes {
    solid.attributes.clone()
}

/// Set the color attribute of a solid
pub fn set_shape_color(solid: &mut Solid, color: [f64; 3]) {
    solid.attributes.color = Some(color);
}

/// Get the color attribute of a solid
pub fn get_shape_color(solid: &Solid) -> Option<[f64; 3]> {
    solid.attributes.color
}

/// Set the layer attribute of a solid
pub fn set_shape_layer(solid: &mut Solid, layer: &str) {
    solid.attributes.layer = Some(layer.to_string());
}

/// Get the layer attribute of a solid
pub fn get_shape_layer(solid: &Solid) -> Option<&str> {
    solid.attributes.layer.as_deref()
}

/// Set the material attribute of a solid
pub fn set_shape_material(solid: &mut Solid, material: &str) {
    solid.attributes.material = Some(material.to_string());
}

/// Get the material attribute of a solid
pub fn get_shape_material(solid: &Solid) -> Option<&str> {
    solid.attributes.material.as_deref()
}

/// An assembly is a hierarchical structure for managing collections of parts and sub-assemblies
/// 
/// Supports:
/// - Direct parts (Solid geometry)
/// - Nested sub-assemblies for hierarchical organization
/// - Part instances with transformation matrices for reuse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assembly {
    /// Name identifier for the assembly
    pub name: String,
    /// Child nodes (parts, sub-assemblies, or instances)
    pub children: Vec<AssemblyNode>,
}

/// A node in an assembly hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssemblyNode {
    /// A direct part with its geometry
    Part(Solid),
    /// A sub-assembly nested within this assembly
    SubAssembly(Box<Assembly>),
    /// An instance referencing a part from a library with a transformation
    /// 
    /// The reference is an index that can be used to look up the part in an external library.
    /// The transform is a 4x4 homogeneous transformation matrix [row][col].
    Instance {
        /// Index/ID of the referenced part (in an external parts library)
        reference: usize,
        /// 4x4 transformation matrix as row-major array
        transform: [[f64; 4]; 4],
    },
}

impl Assembly {
    /// Create a new empty assembly with a given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            children: Vec::new(),
        }
    }

    /// Get the number of direct children
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Check if the assembly is empty (no children)
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}

/// Create a new assembly with the given name
pub fn create_assembly(name: &str) -> Assembly {
    Assembly::new(name)
}

/// Add a part (solid) to an assembly
pub fn add_part(assembly: &mut Assembly, solid: Solid) {
    assembly.children.push(AssemblyNode::Part(solid));
}

/// Add a sub-assembly to an assembly
pub fn add_subassembly(assembly: &mut Assembly, sub: Assembly) {
    assembly.children.push(AssemblyNode::SubAssembly(Box::new(sub)));
}

/// Add an instance reference to an assembly
/// 
/// The reference is an index into an external parts library.
/// The transform is a 4x4 homogeneous transformation matrix.
pub fn add_instance(assembly: &mut Assembly, reference: usize, transform: [[f64; 4]; 4]) {
    assembly.children.push(AssemblyNode::Instance {
        reference,
        transform,
    });
}

/// Flatten an assembly into a list of solids with their transformations
/// 
/// Recursively traverses the assembly hierarchy, collecting all Part nodes
/// with their accumulated transformations. Sub-assemblies are expanded,
/// instances are skipped (as they reference external geometry).
/// 
/// Returns a vector of (Solid, transformation_matrix) pairs, where the
/// transformation is the accumulated transform from the assembly root to that part.
pub fn flatten_assembly(assembly: &Assembly) -> Vec<(Solid, [[f64; 4]; 4])> {
    let identity = identity_matrix();
    flatten_assembly_recursive(assembly, &identity)
}

/// Recursively flatten an assembly with accumulated transformation
fn flatten_assembly_recursive(
    assembly: &Assembly,
    parent_transform: &[[f64; 4]; 4],
) -> Vec<(Solid, [[f64; 4]; 4])> {
    let mut result = Vec::new();

    for child in &assembly.children {
        match child {
            AssemblyNode::Part(solid) => {
                // Add the part with the accumulated transformation
                result.push((solid.clone(), *parent_transform));
            }
            AssemblyNode::SubAssembly(sub) => {
                // Recursively flatten the sub-assembly
                let flattened = flatten_assembly_recursive(sub, parent_transform);
                result.extend(flattened);
            }
            AssemblyNode::Instance {
                reference: _,
                transform: _,
            } => {
                // Skip instances as they reference external geometry
                // In a full implementation, these would be resolved from a parts library
            }
        }
    }

    result
}

/// Create an identity transformation matrix (4x4)
fn identity_matrix() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;

    #[test]
    fn test_shape_attributes_creation() {
        let attrs = ShapeAttributes::new();
        assert!(attrs.is_empty());

        let attrs = ShapeAttributes::with_name("my_box");
        assert!(!attrs.is_empty());
        assert_eq!(attrs.name, Some("my_box".to_string()));
    }

    #[test]
    fn test_set_and_get_name() {
        let mut solid = make_box(1.0, 2.0, 3.0).unwrap();
        assert_eq!(get_shape_name(&solid), None);

        set_shape_name(&mut solid, "test_box");
        assert_eq!(get_shape_name(&solid), Some("test_box"));
    }

    #[test]
    fn test_set_and_get_attributes() {
        let mut solid = make_box(1.0, 2.0, 3.0).unwrap();

        let attrs = ShapeAttributes {
            name: Some("my_box".to_string()),
            color: Some([1.0, 0.0, 0.0]),
            layer: Some("Layer1".to_string()),
            material: Some("Steel".to_string()),
        };

        set_shape_attributes(&mut solid, &attrs);

        let retrieved = get_shape_attributes(&solid);
        assert_eq!(retrieved.name, Some("my_box".to_string()));
        assert_eq!(retrieved.color, Some([1.0, 0.0, 0.0]));
        assert_eq!(retrieved.layer, Some("Layer1".to_string()));
        assert_eq!(retrieved.material, Some("Steel".to_string()));
    }

    #[test]
    fn test_set_and_get_color() {
        let mut solid = make_box(1.0, 2.0, 3.0).unwrap();
        assert_eq!(get_shape_color(&solid), None);

        let red = [1.0, 0.0, 0.0];
        set_shape_color(&mut solid, red);
        assert_eq!(get_shape_color(&solid), Some(red));
    }

    #[test]
    fn test_set_and_get_layer() {
        let mut solid = make_box(1.0, 2.0, 3.0).unwrap();
        assert_eq!(get_shape_layer(&solid), None);

        set_shape_layer(&mut solid, "Layer1");
        assert_eq!(get_shape_layer(&solid), Some("Layer1"));
    }

    #[test]
    fn test_set_and_get_material() {
        let mut solid = make_box(1.0, 2.0, 3.0).unwrap();
        assert_eq!(get_shape_material(&solid), None);

        set_shape_material(&mut solid, "Steel");
        assert_eq!(get_shape_material(&solid), Some("Steel"));
    }

    #[test]
    fn test_create_assembly() {
        let assembly = create_assembly("MainAssembly");
        assert_eq!(assembly.name, "MainAssembly");
        assert!(assembly.is_empty());
        assert_eq!(assembly.child_count(), 0);
    }

    #[test]
    fn test_add_part_to_assembly() {
        let mut assembly = create_assembly("TestAssembly");
        let solid = make_box(1.0, 2.0, 3.0).unwrap();

        assert!(assembly.is_empty());
        add_part(&mut assembly, solid);
        assert!(!assembly.is_empty());
        assert_eq!(assembly.child_count(), 1);

        // Verify it's a Part node
        match &assembly.children[0] {
            AssemblyNode::Part(_) => (),
            _ => panic!("Expected Part node"),
        }
    }

    #[test]
    fn test_add_subassembly() {
        let mut parent = create_assembly("Parent");
        let child = create_assembly("Child");

        assert_eq!(parent.child_count(), 0);
        add_subassembly(&mut parent, child);
        assert_eq!(parent.child_count(), 1);

        // Verify it's a SubAssembly node
        match &parent.children[0] {
            AssemblyNode::SubAssembly(sub) => assert_eq!(sub.name, "Child"),
            _ => panic!("Expected SubAssembly node"),
        }
    }

    #[test]
    fn test_add_instance() {
        let mut assembly = create_assembly("WithInstances");
        let identity = identity_matrix();

        add_instance(&mut assembly, 42, identity);
        assert_eq!(assembly.child_count(), 1);

        match &assembly.children[0] {
            AssemblyNode::Instance { reference, transform } => {
                assert_eq!(*reference, 42);
                assert_eq!(*transform, identity);
            }
            _ => panic!("Expected Instance node"),
        }
    }

    #[test]
    fn test_flatten_assembly_with_parts() {
        let mut assembly = create_assembly("MainAssembly");
        let solid1 = make_box(1.0, 1.0, 1.0).unwrap();
        let solid2 = make_box(2.0, 2.0, 2.0).unwrap();

        add_part(&mut assembly, solid1);
        add_part(&mut assembly, solid2);

        let flattened = flatten_assembly(&assembly);
        assert_eq!(flattened.len(), 2);

        // Verify all parts have identity transformation
        for (_, transform) in &flattened {
            assert_eq!(*transform, identity_matrix());
        }
    }

    #[test]
    fn test_flatten_assembly_with_subassembly() {
        let mut parent = create_assembly("Parent");
        let mut child = create_assembly("Child");

        let solid1 = make_box(1.0, 1.0, 1.0).unwrap();
        let solid2 = make_box(2.0, 2.0, 2.0).unwrap();

        add_part(&mut child, solid1);
        add_part(&mut parent, solid2);
        add_subassembly(&mut parent, child);

        let flattened = flatten_assembly(&parent);
        assert_eq!(flattened.len(), 2);
    }

    #[test]
    fn test_flatten_assembly_skips_instances() {
        let mut assembly = create_assembly("WithInstances");
        let solid = make_box(1.0, 1.0, 1.0).unwrap();

        add_part(&mut assembly, solid);
        add_instance(&mut assembly, 99, identity_matrix());

        // Should only have 1 solid (the instance is skipped)
        let flattened = flatten_assembly(&assembly);
        assert_eq!(flattened.len(), 1);
    }

    #[test]
    fn test_assembly_hierarchy() {
        let mut main = create_assembly("Main");
        let mut sub1 = create_assembly("Sub1");
        let mut sub2 = create_assembly("Sub2");

        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let box2 = make_box(2.0, 2.0, 2.0).unwrap();
        let box3 = make_box(3.0, 3.0, 3.0).unwrap();

        add_part(&mut sub1, box1);
        add_part(&mut sub2, box2);
        add_part(&mut main, box3);
        add_subassembly(&mut main, sub1);
        add_subassembly(&mut main, sub2);

        let flattened = flatten_assembly(&main);
        assert_eq!(flattened.len(), 3);
    }
}
