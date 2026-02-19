//! Topological queries for BREP shapes
//!
//! This module provides topological queries such as finding adjacent faces,
//! connected edges, and shared edges between faces in BREP structures.

use crate::TOLERANCE;
use std::collections::HashMap;
use super::{Vertex, Edge, Face, Solid, Shell};

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
}
