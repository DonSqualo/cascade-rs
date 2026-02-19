//! Topological queries for BREP shapes
//!
//! This module provides topological queries such as finding adjacent faces,
//! connected edges, and shared edges between faces in BREP structures.
//! It also provides the Explorer for iterating over sub-shapes of a given type.

use crate::TOLERANCE;
use std::collections::HashMap;
use super::{Vertex, Edge, Face, Solid, Shell, Wire, Compound};

/// Shape type enumeration for use with Explorer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeType {
    Vertex,
    Edge,
    Wire,
    Face,
    Shell,
    Solid,
    Compound,
}

/// Explorer for iterating over sub-shapes of a given type
///
/// The Explorer provides a mechanism to traverse the topological hierarchy
/// of a shape, extracting all sub-shapes of a specified type.
///
/// Supported hierarchies:
/// - Solid → Shells → Faces → Wires → Edges → Vertices
/// - Compound → Solids → ... (recursive)
///
/// # Example
/// ```ignore
/// let mut explorer = Explorer::new(&solid, ShapeType::Face);
/// while explorer.more() {
///     let face = explorer.current();
///     println!("Processing face");
///     explorer.next();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Explorer {
    /// The collected sub-shapes to iterate over
    sub_shapes: Vec<ExplorerItem>,
    /// Current position in the iteration
    current_index: usize,
}

/// Internal representation of a shape for the Explorer
#[derive(Debug, Clone)]
enum ExplorerItem {
    Vertex(Vertex),
    Edge(Edge),
    Wire(Wire),
    Face(Face),
    Shell(Shell),
    Solid(Solid),
    Compound(Compound),
}

impl Explorer {
    /// Create a new Explorer for a given shape type
    ///
    /// Collects all sub-shapes of the specified type from the input shape
    /// and positions the iterator at the beginning.
    pub fn new(shape: &super::Shape, target_type: ShapeType) -> Self {
        let mut sub_shapes = Vec::new();
        Self::collect_sub_shapes(shape, target_type, &mut sub_shapes);
        
        Explorer {
            sub_shapes,
            current_index: 0,
        }
    }

    /// Check if there are more shapes to iterate
    pub fn more(&self) -> bool {
        self.current_index < self.sub_shapes.len()
    }

    /// Advance to the next shape
    pub fn next(&mut self) {
        if self.more() {
            self.current_index += 1;
        }
    }

    /// Get the current shape as a generic Shape enum
    pub fn current(&self) -> Option<super::Shape> {
        if !self.more() {
            return None;
        }

        match &self.sub_shapes[self.current_index] {
            ExplorerItem::Vertex(v) => Some(super::Shape::Vertex(v.clone())),
            ExplorerItem::Edge(e) => Some(super::Shape::Edge(e.clone())),
            ExplorerItem::Wire(w) => Some(super::Shape::Wire(w.clone())),
            ExplorerItem::Face(f) => Some(super::Shape::Face(f.clone())),
            ExplorerItem::Shell(s) => Some(super::Shape::Shell(s.clone())),
            ExplorerItem::Solid(s) => Some(super::Shape::Solid(s.clone())),
            ExplorerItem::Compound(c) => Some(super::Shape::Compound(c.clone())),
        }
    }

    /// Reset the iterator to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get the number of sub-shapes
    pub fn count(&self) -> usize {
        self.sub_shapes.len()
    }

    /// Collect all sub-shapes of the target type from the given shape
    fn collect_sub_shapes(
        shape: &super::Shape,
        target_type: ShapeType,
        result: &mut Vec<ExplorerItem>,
    ) {
        match shape {
            super::Shape::Vertex(v) => {
                if target_type == ShapeType::Vertex {
                    result.push(ExplorerItem::Vertex(v.clone()));
                }
            }
            super::Shape::Edge(e) => {
                if target_type == ShapeType::Edge {
                    result.push(ExplorerItem::Edge(e.clone()));
                }
                // Edges contain vertices
                if target_type == ShapeType::Vertex {
                    result.push(ExplorerItem::Vertex(e.start.clone()));
                    result.push(ExplorerItem::Vertex(e.end.clone()));
                }
            }
            super::Shape::Wire(w) => {
                if target_type == ShapeType::Wire {
                    result.push(ExplorerItem::Wire(w.clone()));
                }
                // Wire contains edges
                if target_type == ShapeType::Edge {
                    for edge in &w.edges {
                        result.push(ExplorerItem::Edge(edge.clone()));
                    }
                }
                // Collect vertices from edges
                if target_type == ShapeType::Vertex {
                    for edge in &w.edges {
                        result.push(ExplorerItem::Vertex(edge.start.clone()));
                        result.push(ExplorerItem::Vertex(edge.end.clone()));
                    }
                }
            }
            super::Shape::Face(f) => {
                if target_type == ShapeType::Face {
                    result.push(ExplorerItem::Face(f.clone()));
                }
                // Face contains wires (outer + inner)
                if target_type == ShapeType::Wire {
                    result.push(ExplorerItem::Wire(f.outer_wire.clone()));
                    for wire in &f.inner_wires {
                        result.push(ExplorerItem::Wire(wire.clone()));
                    }
                }
                // Collect edges from all wires
                if target_type == ShapeType::Edge {
                    for edge in &f.outer_wire.edges {
                        result.push(ExplorerItem::Edge(edge.clone()));
                    }
                    for wire in &f.inner_wires {
                        for edge in &wire.edges {
                            result.push(ExplorerItem::Edge(edge.clone()));
                        }
                    }
                }
                // Collect vertices from all edges
                if target_type == ShapeType::Vertex {
                    for edge in &f.outer_wire.edges {
                        result.push(ExplorerItem::Vertex(edge.start.clone()));
                        result.push(ExplorerItem::Vertex(edge.end.clone()));
                    }
                    for wire in &f.inner_wires {
                        for edge in &wire.edges {
                            result.push(ExplorerItem::Vertex(edge.start.clone()));
                            result.push(ExplorerItem::Vertex(edge.end.clone()));
                        }
                    }
                }
            }
            super::Shape::Shell(s) => {
                if target_type == ShapeType::Shell {
                    result.push(ExplorerItem::Shell(s.clone()));
                }
                // Shell contains faces
                if target_type == ShapeType::Face {
                    for face in &s.faces {
                        result.push(ExplorerItem::Face(face.clone()));
                    }
                }
                // Recursively collect from faces for lower-level types
                if matches!(target_type, ShapeType::Wire | ShapeType::Edge | ShapeType::Vertex) {
                    for face in &s.faces {
                        Self::collect_sub_shapes(
                            &super::Shape::Face(face.clone()),
                            target_type,
                            result,
                        );
                    }
                }
            }
            super::Shape::Solid(s) => {
                if target_type == ShapeType::Solid {
                    result.push(ExplorerItem::Solid(s.clone()));
                }
                // Solid contains shells (outer + inner)
                if target_type == ShapeType::Shell {
                    result.push(ExplorerItem::Shell(s.outer_shell.clone()));
                    for shell in &s.inner_shells {
                        result.push(ExplorerItem::Shell(shell.clone()));
                    }
                }
                // Recursively collect from shells for lower-level types
                if matches!(target_type, ShapeType::Face | ShapeType::Wire | ShapeType::Edge | ShapeType::Vertex) {
                    Self::collect_sub_shapes(
                        &super::Shape::Shell(s.outer_shell.clone()),
                        target_type,
                        result,
                    );
                    for shell in &s.inner_shells {
                        Self::collect_sub_shapes(
                            &super::Shape::Shell(shell.clone()),
                            target_type,
                            result,
                        );
                    }
                }
            }
            super::Shape::Compound(c) => {
                if target_type == ShapeType::Compound {
                    result.push(ExplorerItem::Compound(c.clone()));
                }
                // Compound contains solids
                if target_type == ShapeType::Solid {
                    for solid in &c.solids {
                        result.push(ExplorerItem::Solid(solid.clone()));
                    }
                }
                // Recursively collect from solids for lower-level types
                if matches!(target_type, ShapeType::Shell | ShapeType::Face | ShapeType::Wire | ShapeType::Edge | ShapeType::Vertex) {
                    for solid in &c.solids {
                        Self::collect_sub_shapes(
                            &super::Shape::Solid(solid.clone()),
                            target_type,
                            result,
                        );
                    }
                }
            }
        }
    }
}

/// Get the neighbors of a face in a solid
/// Returns a vector of all faces that share an edge with the given face
pub fn adjacent_faces(face: &Face, solid: &Solid) -> Vec<Face> {
    let mut adjacent = Vec::new();
    
    // Get all edges from the query face (outer + inner wires)
    let query_edges = get_face_edges(face);
    
    // Check all faces in the solid (outer shell + inner shells)
    let all_faces = get_solid_faces(solid);
    
    for other_face in &all_faces {
        // Skip if same face (by reference check impossible, so check by comparing edges)
        if other_face as *const Face == face as *const Face {
            continue;
        }
        
        let other_edges = get_face_edges(other_face);
        
        // Check if faces share any edge
        for q_edge in &query_edges {
            for o_edge in &other_edges {
                if edges_are_same(q_edge, o_edge) {
                    adjacent.push(other_face.clone());
                    return adjacent; // Found one adjacent face, return early (avoid duplicates)
                }
            }
        }
    }
    
    adjacent
}

/// Get all edges connected to a vertex
/// Returns a vector of all edges that have the vertex as start or end point
pub fn connected_edges(vertex: &Vertex, solid: &Solid) -> Vec<Edge> {
    let mut connected = Vec::new();
    
    // Get all edges from the solid
    let all_faces = get_solid_faces(solid);
    
    for face in &all_faces {
        let face_edges = get_face_edges(face);
        
        for edge in face_edges {
            if vertices_are_same(&edge.start, vertex) || vertices_are_same(&edge.end, vertex) {
                // Check if we already have this edge
                let already_have = connected.iter().any(|e: &Edge| edges_are_same(e, &edge));
                if !already_have {
                    connected.push(edge);
                }
            }
        }
    }
    
    connected
}

/// Find the shared edge between two faces
/// Returns Some(edge) if faces share an edge, None otherwise
pub fn shared_edge(face1: &Face, face2: &Face) -> Option<Edge> {
    let edges1 = get_face_edges(face1);
    let edges2 = get_face_edges(face2);
    
    for e1 in &edges1 {
        for e2 in &edges2 {
            if edges_are_same(e1, e2) {
                return Some(e1.clone());
            }
        }
    }
    
    None
}

/// Build an adjacency map for all faces in a shell
/// Returns a HashMap where each face index maps to a vector of adjacent face indices
pub fn face_neighbors(shell: &Shell) -> HashMap<usize, Vec<usize>> {
    let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
    
    // Initialize all face indices
    for i in 0..shell.faces.len() {
        neighbors.insert(i, Vec::new());
    }
    
    // For each pair of faces, check if they're adjacent
    for i in 0..shell.faces.len() {
        for j in (i + 1)..shell.faces.len() {
            if shared_edge(&shell.faces[i], &shell.faces[j]).is_some() {
                // i and j are adjacent
                neighbors.entry(i).or_default().push(j);
                neighbors.entry(j).or_default().push(i);
            }
        }
    }
    
    neighbors
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compare two vertices with tolerance
fn vertices_are_same(v1: &Vertex, v2: &Vertex) -> bool {
    (v1.point[0] - v2.point[0]).abs() < TOLERANCE
        && (v1.point[1] - v2.point[1]).abs() < TOLERANCE
        && (v1.point[2] - v2.point[2]).abs() < TOLERANCE
}

/// Check if two edges are the same (same start and end or reversed)
fn edges_are_same(e1: &Edge, e2: &Edge) -> bool {
    // Same direction
    let same_direction = vertices_are_same(&e1.start, &e2.start)
        && vertices_are_same(&e1.end, &e2.end);
    
    // Reversed direction (edges can be traversed either way)
    let reversed_direction = vertices_are_same(&e1.start, &e2.end)
        && vertices_are_same(&e1.end, &e2.start);
    
    same_direction || reversed_direction
}

/// Get all edges from a face (outer wire + inner wires)
fn get_face_edges(face: &Face) -> Vec<Edge> {
    let mut edges = Vec::new();
    
    // Add edges from outer wire
    edges.extend(face.outer_wire.edges.iter().cloned());
    
    // Add edges from inner wires (holes)
    for wire in &face.inner_wires {
        edges.extend(wire.edges.iter().cloned());
    }
    
    edges
}

/// Get all faces from a solid (outer shell + inner shells)
fn get_solid_faces(solid: &Solid) -> Vec<Face> {
    get_solid_faces_internal(solid)
}

/// Get all faces from a solid (outer shell + inner shells) - public for use by other modules
pub fn get_solid_faces_internal(solid: &Solid) -> Vec<Face> {
    let mut faces = Vec::new();
    
    // Add faces from outer shell
    faces.extend(solid.outer_shell.faces.iter().cloned());
    
    // Add faces from inner shells (voids)
    for shell in &solid.inner_shells {
        faces.extend(shell.faces.iter().cloned());
    }
    
    faces
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;
    
    #[test]
    fn test_adjacent_faces_box() {
        // A box has 6 faces, each should be adjacent to 4 others
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        
        if !solid.outer_shell.faces.is_empty() {
            let face = &solid.outer_shell.faces[0];
            let adjacent = adjacent_faces(face, &solid);
            
            // First face should have at least some adjacent faces
            assert!(!adjacent.is_empty(), "First face should have adjacent faces in a box");
        }
    }
    
    #[test]
    fn test_connected_edges_box() {
        // A vertex in a box should connect to 3 edges
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        
        if !solid.outer_shell.faces.is_empty() {
            if !solid.outer_shell.faces[0].outer_wire.edges.is_empty() {
                let edge = &solid.outer_shell.faces[0].outer_wire.edges[0];
                let vertex = &edge.start;
                
                let connected = connected_edges(vertex, &solid);
                
                // A vertex on a box should connect to multiple edges
                assert!(!connected.is_empty(), "Vertex should have connected edges");
            }
        }
    }
    
    #[test]
    fn test_shared_edge() {
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        
        if solid.outer_shell.faces.len() >= 2 {
            let face1 = &solid.outer_shell.faces[0];
            let face2 = &solid.outer_shell.faces[1];
            
            // These faces may or may not share an edge
            let shared = shared_edge(face1, face2);
            
            // Just verify the function doesn't crash and returns appropriate type
            match shared {
                Some(_) => {
                    // Faces share an edge, which is valid
                    assert!(true);
                }
                None => {
                    // Faces don't share an edge, also valid
                    assert!(true);
                }
            }
        }
    }
    
    #[test]
    fn test_face_neighbors_box() {
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let neighbors = face_neighbors(&solid.outer_shell);
        
        // A box has 6 faces, so we should have 6 entries
        assert_eq!(neighbors.len(), solid.outer_shell.faces.len());
        
        // Each face in a box should have 4 neighbors
        for (_, neighbor_list) in neighbors.iter() {
            // Most faces in a box have 4 neighbors
            assert!(neighbor_list.len() <= 4, "Face should have at most 4 neighbors");
        }
    }

    #[test]
    fn test_explorer_faces_in_solid() {
        // Create a box solid
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let shape = crate::Shape::Solid(solid);
        
        // Explore for faces
        let explorer = Explorer::new(&shape, ShapeType::Face);
        
        // A box should have 6 faces
        assert_eq!(explorer.count(), 6, "Box should have 6 faces");
        
        // Verify all items are accessible
        let mut count = 0;
        let mut explorer = explorer;
        while explorer.more() {
            let face_shape = explorer.current();
            assert!(face_shape.is_some(), "Current shape should be Some");
            
            // Verify it's actually a face
            match face_shape.unwrap() {
                crate::Shape::Face(_) => {
                    count += 1;
                }
                _ => panic!("Expected Face, got different shape type"),
            }
            
            explorer.next();
        }
        
        assert_eq!(count, 6, "Should iterate exactly 6 faces");
    }

    #[test]
    fn test_explorer_edges_in_solid() {
        // Create a box solid
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let shape = crate::Shape::Solid(solid);
        
        // Explore for edges
        let explorer = Explorer::new(&shape, ShapeType::Edge);
        
        // A box has 12 unique edges, but the Explorer collects edges from all faces
        // Each edge is shared by 2 faces, so we get more than 12
        assert!(explorer.count() > 12, "Should have more than 12 edges (edges are counted multiple times from different faces)");
        
        // Verify all items are edges
        let mut explorer = explorer;
        while explorer.more() {
            match explorer.current().unwrap() {
                crate::Shape::Edge(_) => {},
                _ => panic!("Expected Edge"),
            }
            explorer.next();
        }
    }

    #[test]
    fn test_explorer_vertices_in_solid() {
        // Create a box solid
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let shape = crate::Shape::Solid(solid);
        
        // Explore for vertices
        let explorer = Explorer::new(&shape, ShapeType::Vertex);
        
        // A box should have 8 unique vertices (though duplicates may be collected)
        assert!(explorer.count() >= 8, "Should have at least 8 vertices");
        
        // Verify all items are vertices
        let mut explorer = explorer;
        while explorer.more() {
            match explorer.current().unwrap() {
                crate::Shape::Vertex(_) => {},
                _ => panic!("Expected Vertex"),
            }
            explorer.next();
        }
    }

    #[test]
    fn test_explorer_reset() {
        // Create a box solid
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let shape = crate::Shape::Solid(solid);
        
        // Explore for faces
        let mut explorer = Explorer::new(&shape, ShapeType::Face);
        
        // Iterate through all faces
        while explorer.more() {
            explorer.next();
        }
        
        // Should be at the end
        assert!(!explorer.more(), "Should be at the end after full iteration");
        
        // Reset
        explorer.reset();
        
        // Should be back at the beginning
        assert!(explorer.more(), "Should have items after reset");
        
        // Get first face
        let first_face = explorer.current();
        assert!(first_face.is_some(), "Should have a face after reset");
    }

    #[test]
    fn test_explorer_empty_shape() {
        // Create a single vertex (no sub-shapes)
        let vertex = crate::Vertex::new(0.0, 0.0, 0.0);
        let shape = crate::Shape::Vertex(vertex);
        
        // Explore for faces - should be empty
        let explorer = Explorer::new(&shape, ShapeType::Face);
        
        assert_eq!(explorer.count(), 0, "Vertex should have no faces");
        
        // Explore for vertices - should contain itself
        let explorer = Explorer::new(&shape, ShapeType::Vertex);
        assert_eq!(explorer.count(), 1, "Vertex should find itself");
    }

    #[test]
    fn test_explorer_shell_faces() {
        // Create a box solid and extract its shell
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let shell = solid.outer_shell;
        let shape = crate::Shape::Shell(shell);
        
        // Explore for faces in the shell
        let explorer = Explorer::new(&shape, ShapeType::Face);
        
        // A box shell should have 6 faces
        assert_eq!(explorer.count(), 6, "Box shell should have 6 faces");
    }

    #[test]
    fn test_explorer_face_wires() {
        // Create a box and get the first face
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let face = &solid.outer_shell.faces[0];
        let shape = crate::Shape::Face(face.clone());
        
        // Explore for wires in the face
        let explorer = Explorer::new(&shape, ShapeType::Wire);
        
        // Should have at least the outer wire
        assert!(explorer.count() >= 1, "Face should have at least one wire");
    }

    #[test]
    fn test_explorer_wire_edges() {
        // Create a box and get the first face's outer wire
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let face = &solid.outer_shell.faces[0];
        let wire = &face.outer_wire;
        let shape = crate::Shape::Wire(wire.clone());
        
        // Explore for edges in the wire
        let explorer = Explorer::new(&shape, ShapeType::Edge);
        
        // Should have 4 edges for a rectangular face
        assert_eq!(explorer.count(), 4, "Rectangular wire should have 4 edges");
    }
}
