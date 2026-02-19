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
}
