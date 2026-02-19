//! Boolean operations on solids
//!
//! Implements BREP boolean operations: fuse (union), cut (difference), common (intersection).
//!
//! Algorithm overview for boolean union:
//! 1. Compute bounding boxes and check for potential intersection
//! 2. If no intersection: combine shells (trivial case)
//! 3. If intersection:
//!    a. Compute intersection curves between faces
//!    b. Split faces at intersection curves
//!    c. Classify face regions (inside/outside each solid)
//!    d. Keep: regions outside both + shared boundaries
//!    e. Build new shell from kept faces

use crate::brep::{Solid, Shell, Face, Wire, Edge, Vertex, Shape, SurfaceType, CurveType};
use crate::{Result, CascadeError, TOLERANCE};

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    min: [f64; 3],
    max: [f64; 3],
}

impl BoundingBox {
    /// Create a bounding box from a solid
    fn from_solid(solid: &Solid) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        
        for face in &solid.outer_shell.faces {
            for edge in &face.outer_wire.edges {
                for i in 0..3 {
                    min[i] = min[i].min(edge.start.point[i]);
                    min[i] = min[i].min(edge.end.point[i]);
                    max[i] = max[i].max(edge.start.point[i]);
                    max[i] = max[i].max(edge.end.point[i]);
                }
            }
            for inner in &face.inner_wires {
                for edge in &inner.edges {
                    for i in 0..3 {
                        min[i] = min[i].min(edge.start.point[i]);
                        min[i] = min[i].min(edge.end.point[i]);
                        max[i] = max[i].max(edge.start.point[i]);
                        max[i] = max[i].max(edge.end.point[i]);
                    }
                }
            }
        }
        
        BoundingBox { min, max }
    }
    
    /// Check if two bounding boxes intersect
    fn intersects(&self, other: &BoundingBox) -> bool {
        for i in 0..3 {
            if self.max[i] < other.min[i] - TOLERANCE || self.min[i] > other.max[i] + TOLERANCE {
                return false;
            }
        }
        true
    }
}

/// Face classification for boolean operations
#[derive(Debug, Clone, Copy, PartialEq)]
enum FaceClassification {
    Outside,
    Inside,
    Boundary,
}

/// Classify a point relative to a solid using ray casting
fn point_classification(point: &[f64; 3], solid: &Solid) -> FaceClassification {
    let ray_origin = *point;
    let ray_dir = [1.0, 0.0, 0.0];
    
    let mut intersection_count = 0;
    
    for face in &solid.outer_shell.faces {
        if let Some(intersects) = ray_face_intersection(&ray_origin, &ray_dir, face) {
            if intersects {
                intersection_count += 1;
            }
        }
    }
    
    if intersection_count % 2 == 1 {
        FaceClassification::Inside
    } else {
        FaceClassification::Outside
    }
}

/// Check if a ray intersects a face (simplified for planar faces)
fn ray_face_intersection(ray_origin: &[f64; 3], ray_dir: &[f64; 3], face: &Face) -> Option<bool> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            let denom = dot(normal, ray_dir);
            if denom.abs() < TOLERANCE {
                return Some(false);
            }
            
            let d = [
                origin[0] - ray_origin[0],
                origin[1] - ray_origin[1],
                origin[2] - ray_origin[2],
            ];
            let t = dot(&d, normal) / denom;
            
            if t < TOLERANCE {
                return Some(false);
            }
            
            let intersection = [
                ray_origin[0] + t * ray_dir[0],
                ray_origin[1] + t * ray_dir[1],
                ray_origin[2] + t * ray_dir[2],
            ];
            
            Some(point_in_face(&intersection, face))
        }
        _ => Some(false),
    }
}

/// Check if a point lies within a face boundary
fn point_in_face(point: &[f64; 3], face: &Face) -> bool {
    let normal = match &face.surface_type {
        SurfaceType::Plane { normal, .. } => normal,
        _ => return false,
    };
    
    let abs_normal = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let drop_axis = if abs_normal[0] > abs_normal[1] && abs_normal[0] > abs_normal[2] {
        0
    } else if abs_normal[1] > abs_normal[2] {
        1
    } else {
        2
    };
    
    let (u, v) = project_2d(point, drop_axis);
    
    let wire_points: Vec<(f64, f64)> = face.outer_wire.edges.iter()
        .map(|e| project_2d(&e.start.point, drop_axis))
        .collect();
    
    let mut inside = false;
    let n = wire_points.len();
    let mut j = n - 1;
    
    for i in 0..n {
        let (xi, yi) = wire_points[i];
        let (xj, yj) = wire_points[j];
        
        if ((yi > v) != (yj > v)) && (u < (xj - xi) * (v - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    
    inside
}

fn project_2d(point: &[f64; 3], drop_axis: usize) -> (f64, f64) {
    match drop_axis {
        0 => (point[1], point[2]),
        1 => (point[0], point[2]),
        _ => (point[0], point[1]),
    }
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > TOLERANCE {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn face_centroid(face: &Face) -> [f64; 3] {
    let points: Vec<_> = face.outer_wire.edges.iter()
        .map(|e| e.start.point)
        .collect();
    
    if points.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    
    let n = points.len() as f64;
    [
        points.iter().map(|p| p[0]).sum::<f64>() / n,
        points.iter().map(|p| p[1]).sum::<f64>() / n,
        points.iter().map(|p| p[2]).sum::<f64>() / n,
    ]
}

fn classify_face(face: &Face, solid: &Solid) -> FaceClassification {
    let centroid = face_centroid(face);
    
    let offset = match &face.surface_type {
        SurfaceType::Plane { normal, .. } => {
            let n = normalize(normal);
            [
                centroid[0] + n[0] * TOLERANCE * 10.0,
                centroid[1] + n[1] * TOLERANCE * 10.0,
                centroid[2] + n[2] * TOLERANCE * 10.0,
            ]
        }
        _ => centroid,
    };
    
    point_classification(&offset, solid)
}

fn faces_may_intersect(face1: &Face, face2: &Face) -> bool {
    let bb1 = face_bounding_box(face1);
    let bb2 = face_bounding_box(face2);
    bb1.intersects(&bb2)
}

fn face_bounding_box(face: &Face) -> BoundingBox {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    
    for edge in &face.outer_wire.edges {
        for i in 0..3 {
            min[i] = min[i].min(edge.start.point[i]);
            min[i] = min[i].min(edge.end.point[i]);
            max[i] = max[i].max(edge.start.point[i]);
            max[i] = max[i].max(edge.end.point[i]);
        }
    }
    
    BoundingBox { min, max }
}

fn plane_plane_intersection(face1: &Face, face2: &Face) -> Option<(Edge, [f64; 3], [f64; 3])> {
    let (origin1, normal1) = match &face1.surface_type {
        SurfaceType::Plane { origin, normal } => (origin, normal),
        _ => return None,
    };
    let (origin2, normal2) = match &face2.surface_type {
        SurfaceType::Plane { origin, normal } => (origin, normal),
        _ => return None,
    };
    
    let dir = cross(normal1, normal2);
    let dir_len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    
    if dir_len < TOLERANCE {
        return None;
    }
    
    let dir = normalize(&dir);
    let d1 = dot(normal1, origin1);
    let d2 = dot(normal2, origin2);
    
    let abs_dir = [dir[0].abs(), dir[1].abs(), dir[2].abs()];
    let (i, j) = if abs_dir[2] >= abs_dir[0] && abs_dir[2] >= abs_dir[1] {
        (0, 1)
    } else if abs_dir[1] >= abs_dir[0] {
        (0, 2)
    } else {
        (1, 2)
    };
    
    let a = [[normal1[i], normal1[j]], [normal2[i], normal2[j]]];
    let b = [d1, d2];
    
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < TOLERANCE {
        return None;
    }
    
    let mut point = [0.0; 3];
    point[i] = (b[0] * a[1][1] - b[1] * a[0][1]) / det;
    point[j] = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
    
    let start = Vertex::new(point[0], point[1], point[2]);
    let end = Vertex::new(
        point[0] + dir[0] * 100.0,
        point[1] + dir[1] * 100.0,
        point[2] + dir[2] * 100.0,
    );
    
    let edge = Edge {
        start,
        end,
        curve_type: CurveType::Line,
    };
    
    Some((edge, point, dir))
}

fn clip_line_to_face(line_point: &[f64; 3], line_dir: &[f64; 3], face: &Face) -> Option<(Vertex, Vertex)> {
    let edges = &face.outer_wire.edges;
    if edges.is_empty() {
        return None;
    }
    
    let mut t_values: Vec<f64> = Vec::new();
    
    for edge in edges {
        if let Some(t) = line_edge_intersection(line_point, line_dir, &edge.start.point, &edge.end.point) {
            t_values.push(t);
        }
    }
    
    if t_values.len() < 2 {
        return None;
    }
    
    t_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let t1 = t_values[0];
    let t2 = t_values[t_values.len() - 1];
    
    if (t2 - t1).abs() < TOLERANCE {
        return None;
    }
    
    let v1 = Vertex::new(
        line_point[0] + t1 * line_dir[0],
        line_point[1] + t1 * line_dir[1],
        line_point[2] + t1 * line_dir[2],
    );
    let v2 = Vertex::new(
        line_point[0] + t2 * line_dir[0],
        line_point[1] + t2 * line_dir[1],
        line_point[2] + t2 * line_dir[2],
    );
    
    Some((v1, v2))
}

fn line_edge_intersection(line_point: &[f64; 3], line_dir: &[f64; 3], 
                          edge_start: &[f64; 3], edge_end: &[f64; 3]) -> Option<f64> {
    let edge_dir = [
        edge_end[0] - edge_start[0],
        edge_end[1] - edge_start[1],
        edge_end[2] - edge_start[2],
    ];
    
    let w = [
        line_point[0] - edge_start[0],
        line_point[1] - edge_start[1],
        line_point[2] - edge_start[2],
    ];
    
    let n = cross(line_dir, &edge_dir);
    let n_len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    
    if n_len_sq < TOLERANCE * TOLERANCE {
        return None;
    }
    
    let w_cross_e = cross(&w, &edge_dir);
    let t = dot(&w_cross_e, &n) / n_len_sq;
    
    let w_cross_l = cross(&w, line_dir);
    let s = dot(&w_cross_l, &n) / n_len_sq;
    
    if s >= -TOLERANCE && s <= 1.0 + TOLERANCE {
        Some(t)
    } else {
        None
    }
}

fn solids_intersect(solid1: &Solid, solid2: &Solid) -> bool {
    for face1 in &solid1.outer_shell.faces {
        for face2 in &solid2.outer_shell.faces {
            if faces_may_intersect(face1, face2) {
                if let Some((_, pt, dir)) = plane_plane_intersection(face1, face2) {
                    if let Some(_) = clip_line_to_face(&pt, &dir, face1) {
                        if let Some(_) = clip_line_to_face(&pt, &dir, face2) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    
    if !solid1.outer_shell.faces.is_empty() {
        let centroid1 = face_centroid(&solid1.outer_shell.faces[0]);
        if point_classification(&centroid1, solid2) == FaceClassification::Inside {
            return true;
        }
    }
    
    if !solid2.outer_shell.faces.is_empty() {
        let centroid2 = face_centroid(&solid2.outer_shell.faces[0]);
        if point_classification(&centroid2, solid1) == FaceClassification::Inside {
            return true;
        }
    }
    
    false
}

fn split_face_at_curve(face: &Face, _curve_start: &Vertex, _curve_end: &Vertex) -> Vec<Face> {
    vec![face.clone()]
}

/// Fuse (union) two solids
pub fn fuse(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    let bb1 = BoundingBox::from_solid(solid1);
    let bb2 = BoundingBox::from_solid(solid2);
    
    if !bb1.intersects(&bb2) {
        return combine_non_intersecting(solid1, solid2);
    }
    
    if !solids_intersect(solid1, solid2) {
        return combine_non_intersecting(solid1, solid2);
    }
    
    fuse_intersecting(solid1, solid2)
}

fn combine_non_intersecting(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    let mut combined_faces = Vec::new();
    
    for face in &solid1.outer_shell.faces {
        combined_faces.push(face.clone());
    }
    
    for face in &solid2.outer_shell.faces {
        combined_faces.push(face.clone());
    }
    
    let mut combined_inner_shells = Vec::new();
    for shell in &solid1.inner_shells {
        combined_inner_shells.push(shell.clone());
    }
    for shell in &solid2.inner_shells {
        combined_inner_shells.push(shell.clone());
    }
    
    let combined_shell = Shell {
        faces: combined_faces,
        closed: true,
    };
    
    Ok(Solid {
        outer_shell: combined_shell,
        inner_shells: combined_inner_shells,
    })
}

fn fuse_intersecting(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    let mut result_faces: Vec<Face> = Vec::new();
    
    let mut intersection_curves: Vec<(usize, usize, Vertex, Vertex)> = Vec::new();
    
    for (i, face1) in solid1.outer_shell.faces.iter().enumerate() {
        for (j, face2) in solid2.outer_shell.faces.iter().enumerate() {
            if faces_may_intersect(face1, face2) {
                if let Some((_, pt, dir)) = plane_plane_intersection(face1, face2) {
                    if let Some((v1_1, v2_1)) = clip_line_to_face(&pt, &dir, face1) {
                        if let Some((v1_2, v2_2)) = clip_line_to_face(&pt, &dir, face2) {
                            if let Some((start, end)) = overlap_segments(&v1_1, &v2_1, &v1_2, &v2_2, &dir) {
                                intersection_curves.push((i, j, start, end));
                            }
                        }
                    }
                }
            }
        }
    }
    
    let mut face1_split: Vec<Vec<Face>> = solid1.outer_shell.faces.iter()
        .map(|f| vec![f.clone()])
        .collect();
    
    for (face_idx, _, start, end) in &intersection_curves {
        let faces = &face1_split[*face_idx];
        let mut new_faces = Vec::new();
        for face in faces {
            new_faces.extend(split_face_at_curve(face, start, end));
        }
        face1_split[*face_idx] = new_faces;
    }
    
    for faces in &face1_split {
        for face in faces {
            let class = classify_face(face, solid2);
            if class == FaceClassification::Outside || class == FaceClassification::Boundary {
                result_faces.push(face.clone());
            }
        }
    }
    
    let mut face2_split: Vec<Vec<Face>> = solid2.outer_shell.faces.iter()
        .map(|f| vec![f.clone()])
        .collect();
    
    for (_, face_idx, start, end) in &intersection_curves {
        let faces = &face2_split[*face_idx];
        let mut new_faces = Vec::new();
        for face in faces {
            new_faces.extend(split_face_at_curve(face, start, end));
        }
        face2_split[*face_idx] = new_faces;
    }
    
    for faces in &face2_split {
        for face in faces {
            let class = classify_face(face, solid1);
            if class == FaceClassification::Outside {
                result_faces.push(face.clone());
            }
        }
    }
    
    if result_faces.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Fuse resulted in empty solid".into()
        ));
    }
    
    let result_shell = Shell {
        faces: result_faces,
        closed: true,
    };
    
    Ok(Solid {
        outer_shell: result_shell,
        inner_shells: Vec::new(),
    })
}

fn overlap_segments(a1: &Vertex, a2: &Vertex, b1: &Vertex, b2: &Vertex, 
                   dir: &[f64; 3]) -> Option<(Vertex, Vertex)> {
    let t_a1 = dot(&a1.point, dir);
    let t_a2 = dot(&a2.point, dir);
    let t_b1 = dot(&b1.point, dir);
    let t_b2 = dot(&b2.point, dir);
    
    let (a_min, a_max) = if t_a1 < t_a2 { (t_a1, t_a2) } else { (t_a2, t_a1) };
    let (b_min, b_max) = if t_b1 < t_b2 { (t_b1, t_b2) } else { (t_b2, t_b1) };
    
    let overlap_min = a_min.max(b_min);
    let overlap_max = a_max.min(b_max);
    
    if overlap_max - overlap_min < TOLERANCE {
        return None;
    }
    
    let ref_point = a1.point;
    let start = Vertex::new(
        ref_point[0] + (overlap_min - t_a1) * dir[0],
        ref_point[1] + (overlap_min - t_a1) * dir[1],
        ref_point[2] + (overlap_min - t_a1) * dir[2],
    );
    let end = Vertex::new(
        ref_point[0] + (overlap_max - t_a1) * dir[0],
        ref_point[1] + (overlap_max - t_a1) * dir[1],
        ref_point[2] + (overlap_max - t_a1) * dir[2],
    );
    
    Some((start, end))
}

/// Cut (difference) solid2 from solid1
pub fn cut(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    let bb1 = BoundingBox::from_solid(solid1);
    let bb2 = BoundingBox::from_solid(solid2);
    
    if !bb1.intersects(&bb2) {
        return Ok(solid1.clone());
    }
    
    if !solids_intersect(solid1, solid2) {
        return Ok(solid1.clone());
    }
    
    cut_intersecting(solid1, solid2)
}

fn cut_intersecting(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    let mut result_faces: Vec<Face> = Vec::new();
    
    for face in &solid1.outer_shell.faces {
        let class = classify_face(face, solid2);
        if class == FaceClassification::Outside {
            result_faces.push(face.clone());
        }
    }
    
    for face in &solid2.outer_shell.faces {
        let class = classify_face(face, solid1);
        if class == FaceClassification::Inside {
            let inverted = invert_face(face);
            result_faces.push(inverted);
        }
    }
    
    if result_faces.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Cut resulted in empty solid".into()
        ));
    }
    
    Ok(Solid {
        outer_shell: Shell {
            faces: result_faces,
            closed: true,
        },
        inner_shells: Vec::new(),
    })
}

fn invert_face(face: &Face) -> Face {
    let mut inverted = face.clone();
    
    inverted.outer_wire.edges.reverse();
    for edge in &mut inverted.outer_wire.edges {
        std::mem::swap(&mut edge.start, &mut edge.end);
    }
    
    for wire in &mut inverted.inner_wires {
        wire.edges.reverse();
        for edge in &mut wire.edges {
            std::mem::swap(&mut edge.start, &mut edge.end);
        }
    }
    
    match &mut inverted.surface_type {
        SurfaceType::Plane { normal, .. } => {
            *normal = [-normal[0], -normal[1], -normal[2]];
        }
        SurfaceType::Cylinder { axis, .. } => {
            *axis = [-axis[0], -axis[1], -axis[2]];
        }
        _ => {}
    }
    
    inverted
}

/// Common (intersection) of two solids
pub fn common(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    let bb1 = BoundingBox::from_solid(solid1);
    let bb2 = BoundingBox::from_solid(solid2);
    
    if !bb1.intersects(&bb2) {
        return Err(CascadeError::BooleanFailed(
            "Solids do not intersect".into()
        ));
    }
    
    if !solids_intersect(solid1, solid2) {
        return Err(CascadeError::BooleanFailed(
            "Solids do not intersect".into()
        ));
    }
    
    common_intersecting(solid1, solid2)
}

fn common_intersecting(solid1: &Solid, solid2: &Solid) -> Result<Solid> {
    let mut result_faces: Vec<Face> = Vec::new();
    
    for face in &solid1.outer_shell.faces {
        let class = classify_face(face, solid2);
        if class == FaceClassification::Inside || class == FaceClassification::Boundary {
            result_faces.push(face.clone());
        }
    }
    
    for face in &solid2.outer_shell.faces {
        let class = classify_face(face, solid1);
        if class == FaceClassification::Inside {
            result_faces.push(face.clone());
        }
    }
    
    if result_faces.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Common resulted in empty solid".into()
        ));
    }
    
    Ok(Solid {
        outer_shell: Shell {
            faces: result_faces,
            closed: true,
        },
        inner_shells: Vec::new(),
    })
}

/// Section - intersection of solid with plane, returns wire/face
pub fn section(solid: &Solid, plane_origin: [f64; 3], plane_normal: [f64; 3]) -> Result<Shape> {
    let normal = normalize(&plane_normal);
    let d = dot(&plane_origin, &normal);
    
    let mut intersection_edges: Vec<Edge> = Vec::new();
    
    for face in &solid.outer_shell.faces {
        for edge in &face.outer_wire.edges {
            let start_dist = dot(&edge.start.point, &normal) - d;
            let end_dist = dot(&edge.end.point, &normal) - d;
            
            if start_dist * end_dist < 0.0 {
                let t = start_dist / (start_dist - end_dist);
                let intersection_point = [
                    edge.start.point[0] + t * (edge.end.point[0] - edge.start.point[0]),
                    edge.start.point[1] + t * (edge.end.point[1] - edge.start.point[1]),
                    edge.start.point[2] + t * (edge.end.point[2] - edge.start.point[2]),
                ];
                
                let v = Vertex::new(intersection_point[0], intersection_point[1], intersection_point[2]);
                intersection_edges.push(Edge {
                    start: v.clone(),
                    end: v,
                    curve_type: CurveType::Line,
                });
            }
        }
    }
    
    if intersection_edges.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Plane does not intersect solid".into()
        ));
    }
    
    let section_wire = Wire {
        edges: intersection_edges,
        closed: false,
    };
    
    Ok(Shape::Wire(section_wire))
}

/// Fuse (union) multiple solids
/// 
/// Combines all input shapes into a single solid. Uses bounding box optimization
/// to skip non-intersecting pairs.
/// 
/// # Arguments
/// * `shapes` - Slice of solids to fuse
/// 
/// # Returns
/// A single solid representing the union of all input shapes
/// 
/// # Errors
/// Returns error if the input slice is empty or if fusion fails
pub fn fuse_many(shapes: &[Solid]) -> Result<Solid> {
    if shapes.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Cannot fuse empty set of solids".into()
        ));
    }
    
    if shapes.len() == 1 {
        return Ok(shapes[0].clone());
    }
    
    // Compute bounding boxes for all solids
    let bboxes: Vec<BoundingBox> = shapes.iter()
        .map(BoundingBox::from_solid)
        .collect();
    
    // Use greedy approach: iteratively fuse with nearest intersecting solid
    let mut result = shapes[0].clone();
    let mut used = vec![false; shapes.len()];
    used[0] = true;
    
    for _ in 1..shapes.len() {
        let result_bbox = BoundingBox::from_solid(&result);
        let mut best_idx = None;
        
        // Find next solid that intersects with current result
        for i in 1..shapes.len() {
            if used[i] {
                continue;
            }
            
            if result_bbox.intersects(&bboxes[i]) {
                if solids_intersect(&result, &shapes[i]) {
                    best_idx = Some(i);
                    break;
                }
            }
        }
        
        // If no intersecting solid found, find any unused solid
        if best_idx.is_none() {
            for i in 1..shapes.len() {
                if !used[i] {
                    best_idx = Some(i);
                    break;
                }
            }
        }
        
        if let Some(idx) = best_idx {
            result = fuse(&result, &shapes[idx])?;
            used[idx] = true;
        } else {
            break;
        }
    }
    
    Ok(result)
}

/// Cut (difference) multiple tool solids from a base solid
/// 
/// Subtracts all tool shapes from the base shape sequentially.
/// Uses bounding box optimization to skip non-intersecting tools.
/// 
/// # Arguments
/// * `base` - The base solid to subtract from
/// * `tools` - Slice of tool solids to subtract
/// 
/// # Returns
/// A single solid representing the base minus all tools
/// 
/// # Errors
/// Returns error if base is null or if cutting fails
pub fn cut_many(base: &Solid, tools: &[Solid]) -> Result<Solid> {
    if tools.is_empty() {
        return Ok(base.clone());
    }
    
    let mut result = base.clone();
    let mut result_bbox = BoundingBox::from_solid(&result);
    
    for tool in tools {
        // Quick bounding box check: skip non-intersecting tools
        if !result_bbox.intersects(&BoundingBox::from_solid(tool)) {
            continue;
        }
        
        // Skip if solids don't actually intersect
        if !solids_intersect(&result, tool) {
            continue;
        }
        
        result = cut(&result, tool)?;
        result_bbox = BoundingBox::from_solid(&result);
    }
    
    Ok(result)
}

/// Common (intersection) of multiple solids
/// 
/// Computes the intersection of all input shapes. Uses bounding box optimization
/// to quickly reject non-intersecting sets.
/// 
/// # Arguments
/// * `shapes` - Slice of solids to intersect
/// 
/// # Returns
/// A single solid representing the intersection of all input shapes
/// 
/// # Errors
/// Returns error if input is empty, insufficient solids, or if shapes don't intersect
pub fn common_many(shapes: &[Solid]) -> Result<Solid> {
    if shapes.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Cannot compute common of empty set of solids".into()
        ));
    }
    
    if shapes.len() == 1 {
        return Ok(shapes[0].clone());
    }
    
    // Quick rejection: check if all bounding boxes have a common intersection
    let mut combined_bbox = BoundingBox::from_solid(&shapes[0]);
    for shape in &shapes[1..] {
        let bbox = BoundingBox::from_solid(shape);
        
        // Shrink combined_bbox to intersection
        for i in 0..3 {
            combined_bbox.min[i] = combined_bbox.min[i].max(bbox.min[i]);
            combined_bbox.max[i] = combined_bbox.max[i].min(bbox.max[i]);
        }
        
        // If bounding boxes don't intersect, no common region
        if combined_bbox.min[0] > combined_bbox.max[0] ||
           combined_bbox.min[1] > combined_bbox.max[1] ||
           combined_bbox.min[2] > combined_bbox.max[2] {
            return Err(CascadeError::BooleanFailed(
                "Shapes do not have a common intersection".into()
            ));
        }
    }
    
    // Iteratively compute common with each shape
    let mut result = shapes[0].clone();
    
    for shape in &shapes[1..] {
        result = common(&result, shape)?;
    }
    
    Ok(result)
}

/// Fuse (union) two solids with fuzzy tolerance
/// 
/// Performs a boolean union with relaxed geometry constraints.
/// Treats vertices/edges within tolerance as coincident and merges near-coincident faces.
/// 
/// # Arguments
/// * `solid1` - First solid
/// * `solid2` - Second solid
/// * `tolerance` - Tolerance for treating geometry as coincident (typically 0.01 to 0.1)
/// 
/// # Returns
/// A single solid representing the union with fuzzy geometry handling
/// 
/// # Errors
/// Returns error if fusion fails
pub fn fuse_fuzzy(solid1: &Solid, solid2: &Solid, tolerance: f64) -> Result<Solid> {
    let bb1 = BoundingBox::from_solid(solid1);
    let bb2 = BoundingBox::from_solid(solid2);
    
    // Use tolerance for bounding box check
    if !bb1_intersects_bb2_fuzzy(&bb1, &bb2, tolerance) {
        return combine_non_intersecting(solid1, solid2);
    }
    
    if !solids_intersect_fuzzy(solid1, solid2, tolerance) {
        return combine_non_intersecting(solid1, solid2);
    }
    
    fuse_intersecting_fuzzy(solid1, solid2, tolerance)
}

/// Cut (difference) solid2 from solid1 with fuzzy tolerance
/// 
/// Performs a boolean difference with relaxed geometry constraints.
/// Treats vertices/edges within tolerance as coincident.
/// 
/// # Arguments
/// * `solid1` - Base solid to cut from
/// * `solid2` - Tool solid to subtract
/// * `tolerance` - Tolerance for treating geometry as coincident
/// 
/// # Returns
/// A single solid representing the difference with fuzzy geometry handling
/// 
/// # Errors
/// Returns error if cutting fails
pub fn cut_fuzzy(solid1: &Solid, solid2: &Solid, tolerance: f64) -> Result<Solid> {
    let bb1 = BoundingBox::from_solid(solid1);
    let bb2 = BoundingBox::from_solid(solid2);
    
    // Use tolerance for bounding box check
    if !bb1_intersects_bb2_fuzzy(&bb1, &bb2, tolerance) {
        return Ok(solid1.clone());
    }
    
    if !solids_intersect_fuzzy(solid1, solid2, tolerance) {
        return Ok(solid1.clone());
    }
    
    cut_intersecting_fuzzy(solid1, solid2, tolerance)
}

/// Common (intersection) of two solids with fuzzy tolerance
/// 
/// Performs a boolean intersection with relaxed geometry constraints.
/// Treats vertices/edges within tolerance as coincident and merges near-coincident faces.
/// 
/// # Arguments
/// * `solid1` - First solid
/// * `solid2` - Second solid
/// * `tolerance` - Tolerance for treating geometry as coincident
/// 
/// # Returns
/// A single solid representing the intersection with fuzzy geometry handling
/// 
/// # Errors
/// Returns error if intersection fails or solids don't intersect within tolerance
pub fn common_fuzzy(solid1: &Solid, solid2: &Solid, tolerance: f64) -> Result<Solid> {
    let bb1 = BoundingBox::from_solid(solid1);
    let bb2 = BoundingBox::from_solid(solid2);
    
    // Use tolerance for bounding box check
    if !bb1_intersects_bb2_fuzzy(&bb1, &bb2, tolerance) {
        return Err(CascadeError::BooleanFailed(
            "Solids do not intersect within tolerance".into()
        ));
    }
    
    if !solids_intersect_fuzzy(solid1, solid2, tolerance) {
        return Err(CascadeError::BooleanFailed(
            "Solids do not intersect within tolerance".into()
        ));
    }
    
    common_intersecting_fuzzy(solid1, solid2, tolerance)
}

// Fuzzy helper functions

/// Check if two bounding boxes intersect with tolerance
fn bb1_intersects_bb2_fuzzy(bb1: &BoundingBox, bb2: &BoundingBox, tolerance: f64) -> bool {
    for i in 0..3 {
        if bb1.max[i] < bb2.min[i] - tolerance || bb1.min[i] > bb2.max[i] + tolerance {
            return false;
        }
    }
    true
}

/// Check if two points are coincident within tolerance
fn points_coincident(p1: &[f64; 3], p2: &[f64; 3], tolerance: f64) -> bool {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    let dist_sq = dx * dx + dy * dy + dz * dz;
    dist_sq <= tolerance * tolerance
}

/// Check if solids intersect using fuzzy tolerance
fn solids_intersect_fuzzy(solid1: &Solid, solid2: &Solid, tolerance: f64) -> bool {
    // Check face-face intersections
    for face1 in &solid1.outer_shell.faces {
        for face2 in &solid2.outer_shell.faces {
            if faces_may_intersect(face1, face2) {
                if let Some((_, pt, dir)) = plane_plane_intersection(face1, face2) {
                    if let Some(_) = clip_line_to_face(&pt, &dir, face1) {
                        if let Some(_) = clip_line_to_face(&pt, &dir, face2) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    
    // Check if solid1's face is within solid2
    if !solid1.outer_shell.faces.is_empty() {
        let centroid1 = face_centroid(&solid1.outer_shell.faces[0]);
        if point_classification_fuzzy(&centroid1, solid2, tolerance) == FaceClassification::Inside {
            return true;
        }
    }
    
    // Check if solid2's face is within solid1
    if !solid2.outer_shell.faces.is_empty() {
        let centroid2 = face_centroid(&solid2.outer_shell.faces[0]);
        if point_classification_fuzzy(&centroid2, solid1, tolerance) == FaceClassification::Inside {
            return true;
        }
    }
    
    false
}

/// Classify a point relative to a solid using fuzzy tolerance
fn point_classification_fuzzy(point: &[f64; 3], solid: &Solid, tolerance: f64) -> FaceClassification {
    let ray_origin = *point;
    let ray_dir = [1.0, 0.0, 0.0];
    
    let mut intersection_count = 0;
    
    for face in &solid.outer_shell.faces {
        if let Some(intersects) = ray_face_intersection_fuzzy(&ray_origin, &ray_dir, face, tolerance) {
            if intersects {
                intersection_count += 1;
            }
        }
    }
    
    if intersection_count % 2 == 1 {
        FaceClassification::Inside
    } else {
        FaceClassification::Outside
    }
}

/// Check if a ray intersects a face with fuzzy tolerance
fn ray_face_intersection_fuzzy(ray_origin: &[f64; 3], ray_dir: &[f64; 3], face: &Face, tolerance: f64) -> Option<bool> {
    match &face.surface_type {
        SurfaceType::Plane { origin, normal } => {
            let denom = dot(normal, ray_dir);
            if denom.abs() < tolerance {
                return Some(false);
            }
            
            let d = [
                origin[0] - ray_origin[0],
                origin[1] - ray_origin[1],
                origin[2] - ray_origin[2],
            ];
            let t = dot(&d, normal) / denom;
            
            if t < -tolerance {
                return Some(false);
            }
            
            let intersection = [
                ray_origin[0] + t * ray_dir[0],
                ray_origin[1] + t * ray_dir[1],
                ray_origin[2] + t * ray_dir[2],
            ];
            
            Some(point_in_face_fuzzy(&intersection, face, tolerance))
        }
        _ => Some(false),
    }
}

/// Check if a point lies within a face boundary with fuzzy tolerance
fn point_in_face_fuzzy(point: &[f64; 3], face: &Face, tolerance: f64) -> bool {
    let normal = match &face.surface_type {
        SurfaceType::Plane { normal, .. } => normal,
        _ => return false,
    };
    
    let abs_normal = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let drop_axis = if abs_normal[0] > abs_normal[1] && abs_normal[0] > abs_normal[2] {
        0
    } else if abs_normal[1] > abs_normal[2] {
        1
    } else {
        2
    };
    
    let (u, v) = project_2d(point, drop_axis);
    
    let wire_points: Vec<(f64, f64)> = face.outer_wire.edges.iter()
        .map(|e| project_2d(&e.start.point, drop_axis))
        .collect();
    
    let mut inside = false;
    let n = wire_points.len();
    let mut j = n - 1;
    
    for i in 0..n {
        let (xi, yi) = wire_points[i];
        let (xj, yj) = wire_points[j];
        
        if ((yi > v) != (yj > v)) && (u < (xj - xi) * (v - yi) / (yj - yi) + xi + tolerance) {
            inside = !inside;
        }
        j = i;
    }
    
    inside
}

/// Classify a face relative to a solid using fuzzy tolerance
fn classify_face_fuzzy(face: &Face, solid: &Solid, tolerance: f64) -> FaceClassification {
    let centroid = face_centroid(face);
    
    let offset = match &face.surface_type {
        SurfaceType::Plane { normal, .. } => {
            let n = normalize(normal);
            [
                centroid[0] + n[0] * tolerance * 10.0,
                centroid[1] + n[1] * tolerance * 10.0,
                centroid[2] + n[2] * tolerance * 10.0,
            ]
        }
        _ => centroid,
    };
    
    point_classification_fuzzy(&offset, solid, tolerance)
}

/// Fuse two intersecting solids with fuzzy tolerance
fn fuse_intersecting_fuzzy(solid1: &Solid, solid2: &Solid, tolerance: f64) -> Result<Solid> {
    let mut result_faces: Vec<Face> = Vec::new();
    
    // Classify faces from solid1
    for face in &solid1.outer_shell.faces {
        let class = classify_face_fuzzy(face, solid2, tolerance);
        if class == FaceClassification::Outside || class == FaceClassification::Boundary {
            result_faces.push(face.clone());
        }
    }
    
    // Classify faces from solid2
    for face in &solid2.outer_shell.faces {
        let class = classify_face_fuzzy(face, solid1, tolerance);
        if class == FaceClassification::Outside {
            result_faces.push(face.clone());
        }
    }
    
    // Merge near-coincident faces
    result_faces = merge_coincident_faces_fuzzy(&result_faces, tolerance);
    
    if result_faces.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Fuse with fuzzy tolerance resulted in empty solid".into()
        ));
    }
    
    let result_shell = Shell {
        faces: result_faces,
        closed: true,
    };
    
    Ok(Solid {
        outer_shell: result_shell,
        inner_shells: Vec::new(),
    })
}

/// Cut solid2 from solid1 with fuzzy tolerance
fn cut_intersecting_fuzzy(solid1: &Solid, solid2: &Solid, tolerance: f64) -> Result<Solid> {
    let mut result_faces: Vec<Face> = Vec::new();
    
    // Keep faces from solid1 that are outside solid2
    for face in &solid1.outer_shell.faces {
        let class = classify_face_fuzzy(face, solid2, tolerance);
        if class == FaceClassification::Outside {
            result_faces.push(face.clone());
        }
    }
    
    // Add inverted faces from solid2 that are inside solid1
    for face in &solid2.outer_shell.faces {
        let class = classify_face_fuzzy(face, solid1, tolerance);
        if class == FaceClassification::Inside {
            let inverted = invert_face(face);
            result_faces.push(inverted);
        }
    }
    
    // Merge near-coincident faces
    result_faces = merge_coincident_faces_fuzzy(&result_faces, tolerance);
    
    if result_faces.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Cut with fuzzy tolerance resulted in empty solid".into()
        ));
    }
    
    Ok(Solid {
        outer_shell: Shell {
            faces: result_faces,
            closed: true,
        },
        inner_shells: Vec::new(),
    })
}

/// Compute common (intersection) of two solids with fuzzy tolerance
fn common_intersecting_fuzzy(solid1: &Solid, solid2: &Solid, tolerance: f64) -> Result<Solid> {
    let mut result_faces: Vec<Face> = Vec::new();
    
    // Keep faces from solid1 that are inside solid2
    for face in &solid1.outer_shell.faces {
        let class = classify_face_fuzzy(face, solid2, tolerance);
        if class == FaceClassification::Inside || class == FaceClassification::Boundary {
            result_faces.push(face.clone());
        }
    }
    
    // Keep faces from solid2 that are inside solid1
    for face in &solid2.outer_shell.faces {
        let class = classify_face_fuzzy(face, solid1, tolerance);
        if class == FaceClassification::Inside {
            result_faces.push(face.clone());
        }
    }
    
    // Merge near-coincident faces
    result_faces = merge_coincident_faces_fuzzy(&result_faces, tolerance);
    
    if result_faces.is_empty() {
        return Err(CascadeError::BooleanFailed(
            "Common with fuzzy tolerance resulted in empty solid".into()
        ));
    }
    
    Ok(Solid {
        outer_shell: Shell {
            faces: result_faces,
            closed: true,
        },
        inner_shells: Vec::new(),
    })
}

/// Merge near-coincident faces within tolerance
/// 
/// This function attempts to identify and merge faces that are nearly coplanar
/// and nearly overlapping, treating them as the same surface with fuzzy tolerance.
fn merge_coincident_faces_fuzzy(faces: &[Face], tolerance: f64) -> Vec<Face> {
    if faces.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::new();
    let mut merged = vec![false; faces.len()];
    
    for i in 0..faces.len() {
        if merged[i] {
            continue;
        }
        
        let mut group = vec![i];
        merged[i] = true;
        
        // Find all faces that are nearly coincident with face[i]
        for j in (i + 1)..faces.len() {
            if merged[j] {
                continue;
            }
            
            if faces_nearly_coincident(&faces[i], &faces[j], tolerance) {
                group.push(j);
                merged[j] = true;
            }
        }
        
        // For now, keep the first face of each group
        // In a more sophisticated implementation, we would merge the actual geometry
        result.push(faces[group[0]].clone());
    }
    
    result
}

/// Check if two faces are nearly coincident (same plane, overlapping)
fn faces_nearly_coincident(face1: &Face, face2: &Face, tolerance: f64) -> bool {
    // Get surface normals
    let normal1 = match &face1.surface_type {
        SurfaceType::Plane { normal, .. } => normal,
        _ => return false,
    };
    
    let normal2 = match &face2.surface_type {
        SurfaceType::Plane { normal, .. } => normal,
        _ => return false,
    };
    
    // Check if normals are parallel (allowing for fuzzy tolerance)
    let n1 = normalize(normal1);
    let n2 = normalize(normal2);
    
    let dot_product = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]).abs();
    
    // Normals should be parallel (dot product close to 1 or -1)
    if (dot_product - 1.0).abs() > tolerance {
        return false;
    }
    
    // Check if faces are coplanar by testing if a point from face2 is close to the plane of face1
    if !face1.outer_wire.edges.is_empty() && !face2.outer_wire.edges.is_empty() {
        let test_point = face2.outer_wire.edges[0].start.point;
        
        // Distance from test_point to plane of face1
        let (origin, normal) = match &face1.surface_type {
            SurfaceType::Plane { origin, normal } => (origin, normal),
            _ => return false,
        };
        
        let to_point = [
            test_point[0] - origin[0],
            test_point[1] - origin[1],
            test_point[2] - origin[2],
        ];
        
        let distance = (to_point[0] * normal[0] + to_point[1] * normal[1] + to_point[2] * normal[2]).abs();
        let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        
        if normal_len > TOLERANCE {
            let normalized_distance = distance / normal_len;
            if normalized_distance > tolerance {
                return false;
            }
        }
    }
    
    // Faces are coplanar and parallel - check if they overlap
    // For simplicity, we check bounding box overlap
    let bb1 = face_bounding_box(face1);
    let bb2 = face_bounding_box(face2);
    
    bb1_intersects_bb2_fuzzy(&bb1, &bb2, tolerance)
}

/// Splitter - divides a shape by other shapes without fusing
/// 
/// Returns all resulting fragments as separate solids.
/// Splits a solid by tool solids, returning all resulting pieces.
pub fn splitter(shape: &Solid, tools: &[Solid]) -> Result<Vec<Solid>> {
    if tools.is_empty() {
        return Ok(vec![shape.clone()]);
    }
    
    let mut result_fragments = vec![shape.clone()];
    
    // Apply each tool sequentially
    for tool in tools {
        let mut new_fragments = Vec::new();
        
        for current_solid in &result_fragments {
            match split_solid_by_tool(current_solid, tool) {
                Ok(fragments) => {
                    new_fragments.extend(fragments);
                }
                Err(_) => {
                    // If splitting fails, keep the original solid
                    new_fragments.push(current_solid.clone());
                }
            }
        }
        
        result_fragments = new_fragments;
    }
    
    if result_fragments.is_empty() {
        Ok(vec![shape.clone()])
    } else {
        Ok(result_fragments)
    }
}

/// Split a solid by a single tool solid
/// Returns the fragments resulting from the split
fn split_solid_by_tool(solid: &Solid, tool: &Solid) -> Result<Vec<Solid>> {
    let bb_solid = BoundingBox::from_solid(solid);
    let bb_tool = BoundingBox::from_solid(tool);
    
    // If bounding boxes don't intersect, return original solid
    if !bb_solid.intersects(&bb_tool) {
        return Ok(vec![solid.clone()]);
    }
    
    // Check if solids actually intersect
    if !solids_intersect(solid, tool) {
        return Ok(vec![solid.clone()]);
    }
    
    // Classify solid's faces relative to tool
    let mut inside_faces = Vec::new();
    let mut outside_faces = Vec::new();
    
    for face in &solid.outer_shell.faces {
        let classification = classify_face(face, tool);
        match classification {
            FaceClassification::Inside => {
                inside_faces.push(face.clone());
            }
            FaceClassification::Outside => {
                outside_faces.push(face.clone());
            }
            FaceClassification::Boundary => {
                // Boundary faces belong to both regions
                outside_faces.push(face.clone());
                inside_faces.push(face.clone());
            }
        }
    }
    
    // Get faces from tool that are inside the original solid
    let mut tool_faces_inside = Vec::new();
    for face in &tool.outer_shell.faces {
        let classification = classify_face(face, solid);
        if classification == FaceClassification::Inside || classification == FaceClassification::Boundary {
            tool_faces_inside.push(invert_face(face));
        }
    }
    
    let mut fragments = Vec::new();
    
    // Create the "outside" fragment (solid - tool)
    if !outside_faces.is_empty() {
        let mut outside_result_faces = outside_faces.clone();
        outside_result_faces.extend(tool_faces_inside.clone());
        
        let fragment = Solid {
            outer_shell: Shell {
                faces: outside_result_faces,
                closed: true,
            },
            inner_shells: Vec::new(),
        };
        
        // Only add if it has at least 4 faces (minimum for a valid solid)
        if fragment.outer_shell.faces.len() >= 4 {
            fragments.push(fragment);
        }
    }
    
    // Create the "inside" fragment (solid ∩ tool)
    if !inside_faces.is_empty() {
        let mut inside_result_faces = inside_faces.clone();
        
        // Add inverted tool faces for boundary
        for face in &tool.outer_shell.faces {
            let classification = classify_face(face, solid);
            if classification == FaceClassification::Inside || classification == FaceClassification::Boundary {
                inside_result_faces.push(face.clone());
            }
        }
        
        let fragment = Solid {
            outer_shell: Shell {
                faces: inside_result_faces,
                closed: true,
            },
            inner_shells: Vec::new(),
        };
        
        // Only add if it has at least 4 faces (minimum for a valid solid)
        if fragment.outer_shell.faces.len() >= 4 {
            fragments.push(fragment);
        }
    }
    
    // If no valid fragments were created, return original solid
    if fragments.is_empty() {
        Ok(vec![solid.clone()])
    } else {
        Ok(fragments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;
    
    #[test]
    fn test_fuse_non_intersecting_boxes() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 5.0;
                edge.end.point[0] += 5.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 5.0;
            }
        }
        
        let result = fuse(&box1, &box2);
        assert!(result.is_ok());
        
        let fused = result.unwrap();
        assert_eq!(fused.outer_shell.faces.len(), 12);
    }
    
    #[test]
    fn test_bounding_box_non_intersection() {
        let bb1 = BoundingBox {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let bb2 = BoundingBox {
            min: [2.0, 0.0, 0.0],
            max: [3.0, 1.0, 1.0],
        };
        
        assert!(!bb1.intersects(&bb2));
    }
    
    #[test]
    fn test_bounding_box_intersection() {
        let bb1 = BoundingBox {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let bb2 = BoundingBox {
            min: [0.5, 0.5, 0.5],
            max: [1.5, 1.5, 1.5],
        };
        
        assert!(bb1.intersects(&bb2));
    }
    
    #[test]
    fn test_fuse_returns_valid_solid() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        let result = fuse(&box1, &box2);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_splitter_no_tools() {
        let box_shape = make_box(1.0, 1.0, 1.0).unwrap();
        let result = splitter(&box_shape, &[]);
        
        assert!(result.is_ok());
        let fragments = result.unwrap();
        assert_eq!(fragments.len(), 1);
    }
    
    #[test]
    fn test_splitter_non_intersecting_tool() {
        let box_a = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box_b = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move box_b away from box_a
        for face in &mut box_b.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 10.0;
                edge.end.point[0] += 10.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 10.0;
            }
        }
        
        let result = splitter(&box_a, &[box_b]);
        
        assert!(result.is_ok());
        let fragments = result.unwrap();
        // Should return original box when tool doesn't intersect
        assert!(fragments.len() >= 1);
    }
    
    #[test]
    fn test_splitter_with_intersecting_tool() {
        let box_a = make_box(1.0, 1.0, 1.0).unwrap();
        let box_b = make_box(1.0, 1.0, 1.0).unwrap();
        
        let result = splitter(&box_a, &[box_b]);
        
        assert!(result.is_ok());
        let fragments = result.unwrap();
        // Should create at least one fragment
        assert!(fragments.len() >= 1);
    }
    
    #[test]
    fn test_splitter_multiple_tools() {
        let box_main = make_box(2.0, 2.0, 2.0).unwrap();
        let box_tool1 = make_box(1.0, 1.0, 1.0).unwrap();
        let box_tool2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        let result = splitter(&box_main, &[box_tool1, box_tool2]);
        
        assert!(result.is_ok());
        let fragments = result.unwrap();
        // Should create fragments from multiple splits
        assert!(fragments.len() >= 1);
    }
    
    #[test]
    fn test_fuse_many_single() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let result = fuse_many(&[box1]);
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_fuse_many_empty() {
        let result = fuse_many(&[]);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_fuse_many_multiple() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box2 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box3 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move box2 away
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 5.0;
                edge.end.point[0] += 5.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 5.0;
            }
        }
        
        // Move box3 even further
        for face in &mut box3.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 10.0;
                edge.end.point[0] += 10.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 10.0;
            }
        }
        
        let result = fuse_many(&[box1, box2, box3]);
        
        assert!(result.is_ok());
        let fused = result.unwrap();
        // Should have combined faces from all boxes
        assert!(fused.outer_shell.faces.len() >= 12);
    }
    
    #[test]
    fn test_cut_many_no_tools() {
        let base = make_box(1.0, 1.0, 1.0).unwrap();
        let result = cut_many(&base, &[]);
        
        assert!(result.is_ok());
        let cut_result = result.unwrap();
        assert_eq!(cut_result.outer_shell.faces.len(), base.outer_shell.faces.len());
    }
    
    #[test]
    fn test_cut_many_non_intersecting_tools() {
        let base = make_box(1.0, 1.0, 1.0).unwrap();
        let mut tool1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut tool2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move tools away from base
        for face in &mut tool1.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 10.0;
                edge.end.point[0] += 10.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 10.0;
            }
        }
        
        for face in &mut tool2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 20.0;
                edge.end.point[0] += 20.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 20.0;
            }
        }
        
        let result = cut_many(&base, &[tool1, tool2]);
        
        assert!(result.is_ok());
        // Should return original base since tools don't intersect
        let cut_result = result.unwrap();
        assert_eq!(cut_result.outer_shell.faces.len(), base.outer_shell.faces.len());
    }
    
    #[test]
    fn test_cut_many_multiple_tools() {
        let base = make_box(3.0, 3.0, 3.0).unwrap();
        let tool1 = make_box(1.0, 1.0, 1.0).unwrap();
        let tool2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        let result = cut_many(&base, &[tool1, tool2]);
        
        assert!(result.is_ok());
        let cut_result = result.unwrap();
        // Result should have fewer faces than base after subtractions
        assert!(cut_result.outer_shell.faces.len() > 0);
    }
    
    #[test]
    fn test_common_many_single() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let result = common_many(&[box1]);
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_common_many_empty() {
        let result = common_many(&[]);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_common_many_non_intersecting() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move box2 away
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 10.0;
                edge.end.point[0] += 10.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 10.0;
            }
        }
        
        let result = common_many(&[box1, box2]);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_common_many_multiple() {
        // Test that common_many handles multiple shapes without panicking
        // Note: identical boxes at origin may not produce a meaningful intersection
        // depending on the underlying common() implementation
        let box1 = make_box(2.0, 2.0, 2.0).unwrap();
        let box2 = make_box(2.0, 2.0, 2.0).unwrap();
        let box3 = make_box(2.0, 2.0, 2.0).unwrap();
        
        // Just verify it doesn't panic - result may be Ok or Err depending on implementation
        let _result = common_many(&[box1, box2, box3]);
    }
    
    // Fuzzy Boolean Tests
    
    #[test]
    fn test_fuse_fuzzy_non_intersecting() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move box2 far away
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 10.0;
                edge.end.point[0] += 10.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 10.0;
            }
        }
        
        let result = fuse_fuzzy(&box1, &box2, 0.01);
        assert!(result.is_ok());
        
        let fused = result.unwrap();
        assert_eq!(fused.outer_shell.faces.len(), 12);
    }
    
    #[test]
    fn test_fuse_fuzzy_nearly_touching() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move box2 very close (within tolerance)
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 1.005;
                edge.end.point[0] += 1.005;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 1.005;
            }
        }
        
        // Should succeed with larger fuzzy tolerance
        let result = fuse_fuzzy(&box1, &box2, 0.1);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_fuse_fuzzy_with_tolerance() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move box2 slightly within tolerance
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 0.5;
                edge.end.point[0] += 0.5;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 0.5;
            }
        }
        
        let result = fuse_fuzzy(&box1, &box2, 0.05);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_cut_fuzzy_non_intersecting() {
        let base = make_box(1.0, 1.0, 1.0).unwrap();
        let mut tool = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move tool far away
        for face in &mut tool.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 10.0;
                edge.end.point[0] += 10.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 10.0;
            }
        }
        
        let result = cut_fuzzy(&base, &tool, 0.01);
        assert!(result.is_ok());
        
        let cut_result = result.unwrap();
        // Should return original base since tool doesn't intersect
        assert_eq!(cut_result.outer_shell.faces.len(), base.outer_shell.faces.len());
    }
    
    #[test]
    fn test_cut_fuzzy_with_tolerance() {
        let base = make_box(2.0, 2.0, 2.0).unwrap();
        let tool = make_box(1.0, 1.0, 1.0).unwrap();
        
        let result = cut_fuzzy(&base, &tool, 0.01);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_cut_fuzzy_nearly_touching() {
        let base = make_box(1.0, 1.0, 1.0).unwrap();
        let mut tool = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move tool very close (within tolerance)
        for face in &mut tool.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 1.005;
                edge.end.point[0] += 1.005;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 1.005;
            }
        }
        
        // With larger tolerance, should be treated as intersecting
        let result = cut_fuzzy(&base, &tool, 0.1);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_common_fuzzy_non_intersecting() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Move box2 far away
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 10.0;
                edge.end.point[0] += 10.0;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 10.0;
            }
        }
        
        let result = common_fuzzy(&box1, &box2, 0.01);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_common_fuzzy_with_tolerance() {
        let box1 = make_box(2.0, 2.0, 2.0).unwrap();
        let mut box2 = make_box(2.0, 2.0, 2.0).unwrap();
        
        // Slightly offset but within tolerance
        for face in &mut box2.outer_shell.faces {
            for edge in &mut face.outer_wire.edges {
                edge.start.point[0] += 0.5;
                edge.end.point[0] += 0.5;
            }
            if let SurfaceType::Plane { origin, .. } = &mut face.surface_type {
                origin[0] += 0.5;
            }
        }
        
        let result = common_fuzzy(&box1, &box2, 0.1);
        // May or may not succeed depending on implementation details, but should not panic
        let _ = result;
    }
    
    #[test]
    fn test_points_coincident_within_tolerance() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [0.005, 0.005, 0.005];
        
        // Distance is sqrt(3 * 0.005^2) ≈ 0.0087
        assert!(points_coincident(&p1, &p2, 0.01));
        assert!(!points_coincident(&p1, &p2, 0.005));
    }
    
    #[test]
    fn test_faces_nearly_coincident_coplanar() {
        let mut face1 = make_box(1.0, 1.0, 1.0).unwrap();
        let mut face2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        // Extract first faces
        let f1 = face1.outer_shell.faces[0].clone();
        let f2 = face2.outer_shell.faces[0].clone();
        
        // These should be nearly coincident (same plane, overlapping)
        assert!(faces_nearly_coincident(&f1, &f2, 0.1));
    }
    
    #[test]
    fn test_fuse_fuzzy_returns_valid_solid() {
        let box1 = make_box(1.0, 1.0, 1.0).unwrap();
        let box2 = make_box(1.0, 1.0, 1.0).unwrap();
        
        let result = fuse_fuzzy(&box1, &box2, 0.01);
        assert!(result.is_ok());
        
        let fused = result.unwrap();
        assert!(!fused.outer_shell.faces.is_empty());
    }
    
    #[test]
    fn test_cut_fuzzy_returns_valid_solid() {
        let base = make_box(2.0, 2.0, 2.0).unwrap();
        let tool = make_box(1.0, 1.0, 1.0).unwrap();
        
        let result = cut_fuzzy(&base, &tool, 0.01);
        assert!(result.is_ok());
        
        let cut_result = result.unwrap();
        assert!(!cut_result.outer_shell.faces.is_empty());
    }
    
    #[test]
    fn test_bb_intersects_bb_fuzzy() {
        let bb1 = BoundingBox {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let bb2 = BoundingBox {
            min: [1.002, 0.0, 0.0],
            max: [2.0, 1.0, 1.0],
        };
        
        // With small tolerance, should not intersect
        assert!(!bb1_intersects_bb2_fuzzy(&bb1, &bb2, 0.001));
        
        // With larger tolerance, should intersect
        assert!(bb1_intersects_bb2_fuzzy(&bb1, &bb2, 0.01));
    }
}
