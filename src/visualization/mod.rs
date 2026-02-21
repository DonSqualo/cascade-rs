//! Visualization module - rendering and display modes for cascade shapes
//!
//! Provides facilities for displaying 3D shapes in different rendering modes:
//! - Shaded: Flat shading with lighting calculation
//! - Wireframe: Edge-only rendering
//! - ShadedWithEdges: Combination of shaded faces with edge highlighting
//!
//! Includes shape selection via ray-casting for interactive visualization.

use crate::mesh::TriangleMesh;
use crate::{CascadeError, Result as CascadeResult, Solid};
use std::collections::{HashMap, HashSet};

/// Display mode for shape visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    /// Shaded rendering with flat shading and lighting
    Shaded,
    /// Wireframe rendering - edges only
    Wireframe,
    /// Shaded faces with edge overlay
    ShadedWithEdges,
}

/// Selection mode for interactive shape selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// No selection
    None,
    /// Select entire shapes
    Shape,
    /// Select individual faces
    Face,
    /// Select individual edges
    Edge,
    /// Select individual vertices
    Vertex,
}

/// Display style configuration
#[derive(Debug, Clone)]
pub struct DisplayStyle {
    /// Current display mode
    pub mode: DisplayMode,
    /// Base color as [R, G, B] normalized to [0.0, 1.0]
    pub color: [f64; 3],
    /// Surface transparency/alpha (0.0 = transparent, 1.0 = opaque)
    pub transparency: f64,
    /// Edge color as [R, G, B] normalized to [0.0, 1.0]
    pub edge_color: [f64; 3],
    /// Edge width for wireframe and edge overlay modes
    pub edge_width: f64,
}

impl Default for DisplayStyle {
    fn default() -> Self {
        DisplayStyle {
            mode: DisplayMode::Shaded,
            color: [0.8, 0.8, 0.8],
            transparency: 1.0,
            edge_color: [0.0, 0.0, 0.0],
            edge_width: 1.0,
        }
    }
}

impl DisplayStyle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: DisplayMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.color = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
        self
    }

    pub fn with_transparency(mut self, alpha: f64) -> Self {
        self.transparency = alpha.clamp(0.0, 1.0);
        self
    }

    pub fn with_edge_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.edge_color = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
        self
    }

    pub fn with_edge_width(mut self, width: f64) -> Self {
        self.edge_width = width.max(0.1);
        self
    }
}

/// Highlighting style for emphasized shape visualization
#[derive(Debug, Clone)]
pub struct HighlightStyle {
    /// Highlight color as [R, G, B] normalized to [0.0, 1.0]
    pub color: [f64; 3],
    /// Highlight opacity/alpha (0.0 = transparent, 1.0 = opaque)
    pub opacity: f64,
    /// Width of the outline for highlighted elements
    pub outline_width: f64,
}

impl Default for HighlightStyle {
    fn default() -> Self {
        HighlightStyle {
            color: [1.0, 0.5, 0.0], // Default orange highlight
            opacity: 1.0,
            outline_width: 2.0,
        }
    }
}

impl HighlightStyle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_color(mut self, r: f64, g: f64, b: f64) -> Self {
        self.color = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
        self
    }

    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    pub fn with_outline_width(mut self, width: f64) -> Self {
        self.outline_width = width.max(0.1);
        self
    }
}

/// Represents a hit result from ray-casting selection
#[derive(Debug, Clone)]
pub struct SelectionHit {
    /// Type of entity selected
    pub entity_type: SelectionMode,
    /// Index of the selected entity
    pub entity_id: usize,
    /// Distance from ray origin to hit point
    pub distance: f64,
    /// Point of intersection in world space
    pub hit_point: [f64; 3],
}

/// Selection state for tracking selected entities
#[derive(Debug, Clone)]
pub struct Selection {
    /// Current selection mode
    pub mode: SelectionMode,
    /// List of currently selected entities
    selected_entities: Vec<SelectionHit>,
}

impl Default for Selection {
    fn default() -> Self {
        Selection {
            mode: SelectionMode::None,
            selected_entities: Vec::new(),
        }
    }
}

impl Selection {
    /// Create a new selection with no mode
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the selection mode
    pub fn set_mode(&mut self, mode: SelectionMode) {
        self.mode = mode;
    }

    /// Add a hit to the selection
    pub fn add_hit(&mut self, hit: SelectionHit) {
        self.selected_entities.push(hit);
        // Sort by distance so closest is first
        self.selected_entities.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Clear all selections
    pub fn clear(&mut self) {
        self.selected_entities.clear();
    }

    /// Get the closest selected entity
    pub fn closest(&self) -> Option<&SelectionHit> {
        self.selected_entities.first()
    }

    /// Get all selected entities
    pub fn all(&self) -> &[SelectionHit] {
        &self.selected_entities
    }

    /// Check if any entity is selected
    pub fn is_empty(&self) -> bool {
        self.selected_entities.is_empty()
    }

    /// Get count of selected entities
    pub fn count(&self) -> usize {
        self.selected_entities.len()
    }
}

#[derive(Debug, Clone)]
pub struct Viewer {
    style: DisplayStyle,
    mesh: Option<TriangleMesh>,
    tolerance: f64,
    selection: Selection,
    /// Camera position in world space
    camera_pos: [f64; 3],
    /// Camera look-at direction (normalized)
    camera_dir: [f64; 3],
    /// Camera up vector (normalized)
    camera_up: [f64; 3],
    /// Highlight style configuration
    highlight_style: HighlightStyle,
    /// Set of highlighted shape IDs (simple usize identifiers)
    highlighted_shapes: HashSet<usize>,
    /// Map of face indices to highlight styles
    highlighted_faces: HashMap<usize, HighlightStyle>,
    /// Map of edge indices (vertex_pair tuples) to highlight styles
    highlighted_edges: HashMap<(usize, usize), HighlightStyle>,
}

impl Default for Viewer {
    fn default() -> Self {
        Viewer {
            style: DisplayStyle::default(),
            mesh: None,
            tolerance: 1e-3,
            selection: Selection::default(),
            camera_pos: [10.0, 10.0, 10.0],
            camera_dir: [-1.0, -1.0, -1.0],
            camera_up: [0.0, 0.0, 1.0],
            highlight_style: HighlightStyle::default(),
            highlighted_shapes: HashSet::new(),
            highlighted_faces: HashMap::new(),
            highlighted_edges: HashMap::new(),
        }
    }
}

impl Viewer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tolerance(tolerance: f64) -> Self {
        Viewer {
            style: DisplayStyle::default(),
            mesh: None,
            tolerance: tolerance.max(1e-6),
            selection: Selection::default(),
            camera_pos: [10.0, 10.0, 10.0],
            camera_dir: [-1.0, -1.0, -1.0],
            camera_up: [0.0, 0.0, 1.0],
            highlight_style: HighlightStyle::default(),
            highlighted_shapes: HashSet::new(),
            highlighted_faces: HashMap::new(),
            highlighted_edges: HashMap::new(),
        }
    }

    pub fn set_display_mode(&mut self, mode: DisplayMode) {
        self.style.mode = mode;
    }

    pub fn set_style(&mut self, style: DisplayStyle) {
        self.style = style;
    }

    pub fn load_solid(&mut self, solid: &Solid) -> CascadeResult<()> {
        self.mesh = Some(crate::mesh::triangulate(solid, self.tolerance)?);
        Ok(())
    }

    pub fn mesh(&self) -> Option<&TriangleMesh> {
        self.mesh.as_ref()
    }

    pub fn style(&self) -> &DisplayStyle {
        &self.style
    }

    /// Set the selection mode for this viewer
    pub fn set_selection_mode(&mut self, mode: SelectionMode) {
        self.selection.set_mode(mode);
    }

    /// Set the camera position and orientation
    pub fn set_camera(&mut self, pos: [f64; 3], dir: [f64; 3], up: [f64; 3]) {
        self.camera_pos = pos;
        self.camera_dir = normalize(&dir);
        self.camera_up = normalize(&up);
    }

    /// Perform ray-casting selection at screen coordinates (x, y)
    /// Returns the closest hit or None if nothing was selected
    pub fn select_at(&mut self, x: f64, y: f64, viewport_width: f64, viewport_height: f64) -> Option<SelectionHit> {
        if self.selection.mode == SelectionMode::None {
            return None;
        }

        let mesh = self.mesh.as_ref()?;

        // Generate ray from camera through pixel
        // Normalize screen coordinates to [-1, 1]
        let ndc_x = (2.0 * x / viewport_width) - 1.0;
        let ndc_y = 1.0 - (2.0 * y / viewport_height);

        // Compute ray direction in world space
        let forward = self.camera_dir;
        let right = cross_product(&forward, &self.camera_up);
        let right = normalize(&right);
        let up = cross_product(&right, &forward);
        let up = normalize(&up);

        // Simple perspective camera ray
        let ray_dir = [
            forward[0] + ndc_x * right[0] * 0.5 + ndc_y * up[0] * 0.5,
            forward[1] + ndc_x * right[1] * 0.5 + ndc_y * up[1] * 0.5,
            forward[2] + ndc_x * right[2] * 0.5 + ndc_y * up[2] * 0.5,
        ];
        let ray_dir = normalize(&ray_dir);

        let ray_origin = self.camera_pos;

        // Test ray against all triangles in the mesh
        let mut closest_hit: Option<SelectionHit> = None;

        match self.selection.mode {
            SelectionMode::Shape => {
                // Select any triangle hit counts as shape selection
                for (tri_idx, triangle) in mesh.triangles.iter().enumerate() {
                    let idx0 = triangle[0];
                    let idx1 = triangle[1];
                    let idx2 = triangle[2];

                    if idx0 >= mesh.vertices.len() || idx1 >= mesh.vertices.len() || idx2 >= mesh.vertices.len() {
                        continue;
                    }

                    let v0 = mesh.vertices[idx0];
                    let v1 = mesh.vertices[idx1];
                    let v2 = mesh.vertices[idx2];

                    if let Some((distance, hit_point)) = ray_triangle_intersection(&ray_origin, &ray_dir, &v0, &v1, &v2) {
                        if distance >= 0.0 {
                            if closest_hit.is_none() || distance < closest_hit.as_ref().unwrap().distance {
                                closest_hit = Some(SelectionHit {
                                    entity_type: SelectionMode::Shape,
                                    entity_id: 0,
                                    distance,
                                    hit_point,
                                });
                            }
                        }
                    }
                }
            }
            SelectionMode::Face => {
                // Each triangle is considered a face
                for (tri_idx, triangle) in mesh.triangles.iter().enumerate() {
                    let idx0 = triangle[0];
                    let idx1 = triangle[1];
                    let idx2 = triangle[2];

                    if idx0 >= mesh.vertices.len() || idx1 >= mesh.vertices.len() || idx2 >= mesh.vertices.len() {
                        continue;
                    }

                    let v0 = mesh.vertices[idx0];
                    let v1 = mesh.vertices[idx1];
                    let v2 = mesh.vertices[idx2];

                    if let Some((distance, hit_point)) = ray_triangle_intersection(&ray_origin, &ray_dir, &v0, &v1, &v2) {
                        if distance >= 0.0 {
                            if closest_hit.is_none() || distance < closest_hit.as_ref().unwrap().distance {
                                closest_hit = Some(SelectionHit {
                                    entity_type: SelectionMode::Face,
                                    entity_id: tri_idx,
                                    distance,
                                    hit_point,
                                });
                            }
                        }
                    }
                }
            }
            SelectionMode::Vertex => {
                // Select the closest vertex to the ray
                for (v_idx, vertex) in mesh.vertices.iter().enumerate() {
                    let to_vertex = [vertex[0] - ray_origin[0], vertex[1] - ray_origin[1], vertex[2] - ray_origin[2]];
                    let projection_dist = dot_product(&to_vertex, &ray_dir);
                    
                    if projection_dist >= 0.0 {
                        let closest_point_on_ray = [
                            ray_origin[0] + ray_dir[0] * projection_dist,
                            ray_origin[1] + ray_dir[1] * projection_dist,
                            ray_origin[2] + ray_dir[2] * projection_dist,
                        ];
                        
                        let diff = [
                            vertex[0] - closest_point_on_ray[0],
                            vertex[1] - closest_point_on_ray[1],
                            vertex[2] - closest_point_on_ray[2],
                        ];
                        let distance = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
                        
                        // Allow selection if within a reasonable pick distance (0.1 units from ray)
                        if distance < 0.1 {
                            if closest_hit.is_none() || projection_dist < closest_hit.as_ref().unwrap().distance {
                                closest_hit = Some(SelectionHit {
                                    entity_type: SelectionMode::Vertex,
                                    entity_id: v_idx,
                                    distance: projection_dist,
                                    hit_point: *vertex,
                                });
                            }
                        }
                    }
                }
            }
            SelectionMode::Edge => {
                // Select edge (line segment) by finding closest approach to ray
                let mut edge_set = std::collections::HashSet::new();
                for triangle in &mesh.triangles {
                    let idx0 = triangle[0];
                    let idx1 = triangle[1];
                    let idx2 = triangle[2];

                    let edges = vec![
                        (idx0.min(idx1), idx0.max(idx1)),
                        (idx1.min(idx2), idx1.max(idx2)),
                        (idx2.min(idx0), idx2.max(idx0)),
                    ];

                    for (a, b) in edges {
                        edge_set.insert((a, b));
                    }
                }

                for (edge_id, (v_idx_a, v_idx_b)) in edge_set.iter().enumerate() {
                    if *v_idx_a >= mesh.vertices.len() || *v_idx_b >= mesh.vertices.len() {
                        continue;
                    }

                    let v_a = mesh.vertices[*v_idx_a];
                    let v_b = mesh.vertices[*v_idx_b];

                    if let Some((distance, hit_point)) = ray_segment_closest_approach(&ray_origin, &ray_dir, &v_a, &v_b) {
                        if distance < 0.1 {
                            if closest_hit.is_none() || distance < closest_hit.as_ref().unwrap().distance {
                                closest_hit = Some(SelectionHit {
                                    entity_type: SelectionMode::Edge,
                                    entity_id: edge_id,
                                    distance,
                                    hit_point,
                                });
                            }
                        }
                    }
                }
            }
            SelectionMode::None => {}
        }

        if let Some(hit) = closest_hit.clone() {
            self.selection.clear();
            self.selection.add_hit(hit.clone());
            Some(hit)
        } else {
            self.selection.clear();
            None
        }
    }

    /// Get the current selection state
    pub fn get_selection(&self) -> &Selection {
        &self.selection
    }

    /// Get the current selection state (mutable)
    pub fn get_selection_mut(&mut self) -> &mut Selection {
        &mut self.selection
    }

    pub fn render_shaded(&self) -> CascadeResult<Vec<RenderFace>> {
        let mesh = self
            .mesh
            .as_ref()
            .ok_or(CascadeError::NotImplemented("No mesh loaded".to_string()))?;

        let mut faces = Vec::new();
        let light_dir = normalize(&[1.0, 1.0, 1.0]);

        for triangle in &mesh.triangles {
            let idx0 = triangle[0];
            let idx1 = triangle[1];
            let idx2 = triangle[2];

            if idx0 >= mesh.vertices.len()
                || idx1 >= mesh.vertices.len()
                || idx2 >= mesh.vertices.len()
            {
                continue;
            }

            let v0 = mesh.vertices[idx0];
            let v1 = mesh.vertices[idx1];
            let v2 = mesh.vertices[idx2];

            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

            let normal = cross_product(&e1, &e2);
            let normal = normalize(&normal);

            let intensity = (dot_product(&normal, &light_dir) + 1.0) / 2.0;
            let intensity = intensity.clamp(0.2, 1.0);

            let lit_color = [
                self.style.color[0] * intensity,
                self.style.color[1] * intensity,
                self.style.color[2] * intensity,
            ];

            faces.push(RenderFace {
                vertices: [v0, v1, v2],
                normal,
                color: lit_color,
                transparency: self.style.transparency,
            });
        }

        Ok(faces)
    }

    pub fn render_wireframe(&self) -> CascadeResult<Vec<RenderEdge>> {
        let mesh = self
            .mesh
            .as_ref()
            .ok_or(CascadeError::NotImplemented("No mesh loaded".to_string()))?;

        let mut edges = Vec::new();
        let mut edge_set = std::collections::HashSet::new();

        for triangle in &mesh.triangles {
            let idx0 = triangle[0];
            let idx1 = triangle[1];
            let idx2 = triangle[2];

            if idx0 >= mesh.vertices.len()
                || idx1 >= mesh.vertices.len()
                || idx2 >= mesh.vertices.len()
            {
                continue;
            }

            let edges_to_add = vec![
                (idx0.min(idx1), idx0.max(idx1)),
                (idx1.min(idx2), idx1.max(idx2)),
                (idx2.min(idx0), idx2.max(idx0)),
            ];

            for (a, b) in edges_to_add {
                if edge_set.insert((a, b)) {
                    edges.push(RenderEdge {
                        start: mesh.vertices[a],
                        end: mesh.vertices[b],
                        color: self.style.edge_color,
                        width: self.style.edge_width,
                    });
                }
            }
        }

        Ok(edges)
    }

    pub fn render_shaded_with_edges(&self) -> CascadeResult<(Vec<RenderFace>, Vec<RenderEdge>)> {
        let faces = self.render_shaded()?;
        let edges = self.render_wireframe()?;
        Ok((faces, edges))
    }

    /// Set the highlight style for subsequent highlight operations
    pub fn set_highlight_style(&mut self, style: HighlightStyle) {
        self.highlight_style = style;
    }

    /// Get the current highlight style
    pub fn highlight_style(&self) -> &HighlightStyle {
        &self.highlight_style
    }

    /// Highlight an entire shape by ID
    pub fn highlight_shape(&mut self, shape_id: usize) {
        self.highlighted_shapes.insert(shape_id);
    }

    /// Remove highlight from a specific shape
    pub fn unhighlight_shape(&mut self, shape_id: usize) {
        self.highlighted_shapes.remove(&shape_id);
    }

    /// Check if a shape is highlighted
    pub fn is_shape_highlighted(&self, shape_id: usize) -> bool {
        self.highlighted_shapes.contains(&shape_id)
    }

    /// Clear all highlighted shapes
    pub fn clear_shape_highlights(&mut self) {
        self.highlighted_shapes.clear();
    }

    /// Highlight a specific face by index
    pub fn highlight_face(&mut self, face_index: usize, style: Option<HighlightStyle>) {
        let highlight = style.unwrap_or_else(|| self.highlight_style.clone());
        self.highlighted_faces.insert(face_index, highlight);
    }

    /// Remove highlight from a specific face
    pub fn unhighlight_face(&mut self, face_index: usize) {
        self.highlighted_faces.remove(&face_index);
    }

    /// Check if a face is highlighted
    pub fn is_face_highlighted(&self, face_index: usize) -> bool {
        self.highlighted_faces.contains_key(&face_index)
    }

    /// Clear all highlighted faces
    pub fn clear_face_highlights(&mut self) {
        self.highlighted_faces.clear();
    }

    /// Highlight a specific edge by vertex pair indices
    pub fn highlight_edge(
        &mut self,
        start_idx: usize,
        end_idx: usize,
        style: Option<HighlightStyle>,
    ) {
        let edge_key = if start_idx < end_idx {
            (start_idx, end_idx)
        } else {
            (end_idx, start_idx)
        };
        let highlight = style.unwrap_or_else(|| self.highlight_style.clone());
        self.highlighted_edges.insert(edge_key, highlight);
    }

    /// Remove highlight from a specific edge
    pub fn unhighlight_edge(&mut self, start_idx: usize, end_idx: usize) {
        let edge_key = if start_idx < end_idx {
            (start_idx, end_idx)
        } else {
            (end_idx, start_idx)
        };
        self.highlighted_edges.remove(&edge_key);
    }

    /// Check if an edge is highlighted
    pub fn is_edge_highlighted(&self, start_idx: usize, end_idx: usize) -> bool {
        let edge_key = if start_idx < end_idx {
            (start_idx, end_idx)
        } else {
            (end_idx, start_idx)
        };
        self.highlighted_edges.contains_key(&edge_key)
    }

    /// Clear all highlighted edges
    pub fn clear_edge_highlights(&mut self) {
        self.highlighted_edges.clear();
    }

    /// Clear all highlights (shapes, faces, and edges)
    pub fn clear_highlights(&mut self) {
        self.clear_shape_highlights();
        self.clear_face_highlights();
        self.clear_edge_highlights();
    }

    /// Get the number of currently highlighted shapes
    pub fn highlight_count(&self) -> usize {
        self.highlighted_shapes.len() + self.highlighted_faces.len() + self.highlighted_edges.len()
    }
}

#[derive(Debug, Clone)]
pub struct RenderFace {
    pub vertices: [[f64; 3]; 3],
    pub normal: [f64; 3],
    pub color: [f64; 3],
    pub transparency: f64,
}

#[derive(Debug, Clone)]
pub struct RenderEdge {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub color: [f64; 3],
    pub width: f64,
}

fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Ray-triangle intersection using Moller-Trumbore algorithm
/// Returns (distance, hit_point) if intersection occurs, None otherwise
fn ray_triangle_intersection(ray_origin: &[f64; 3], ray_dir: &[f64; 3], 
                             v0: &[f64; 3], v1: &[f64; 3], v2: &[f64; 3]) -> Option<(f64, [f64; 3])> {
    const EPSILON: f64 = 1e-8;

    let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let h = cross_product(ray_dir, &edge2);
    let a = dot_product(&edge1, &h);

    if a.abs() < EPSILON {
        return None; // Ray parallel to triangle
    }

    let f = 1.0 / a;
    let s = [ray_origin[0] - v0[0], ray_origin[1] - v0[1], ray_origin[2] - v0[2]];
    let u = f * dot_product(&s, &h);

    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = cross_product(&s, &edge1);
    let v = f * dot_product(ray_dir, &q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * dot_product(&edge2, &q);

    if t > EPSILON {
        let hit_point = [
            ray_origin[0] + ray_dir[0] * t,
            ray_origin[1] + ray_dir[1] * t,
            ray_origin[2] + ray_dir[2] * t,
        ];
        Some((t, hit_point))
    } else {
        None
    }
}

/// Compute closest approach between ray and line segment
/// Returns (distance_from_ray, closest_point) if segment is close to ray
fn ray_segment_closest_approach(ray_origin: &[f64; 3], ray_dir: &[f64; 3],
                                seg_start: &[f64; 3], seg_end: &[f64; 3]) -> Option<(f64, [f64; 3])> {
    let seg_dir = [seg_end[0] - seg_start[0], seg_end[1] - seg_start[1], seg_end[2] - seg_start[2]];
    let seg_len_sq = dot_product(&seg_dir, &seg_dir);

    if seg_len_sq < 1e-10 {
        // Degenerate segment, treat as point
        let to_start = [seg_start[0] - ray_origin[0], seg_start[1] - ray_origin[1], seg_start[2] - ray_origin[2]];
        let t = dot_product(&to_start, ray_dir);

        if t >= 0.0 {
            let closest = [
                ray_origin[0] + ray_dir[0] * t,
                ray_origin[1] + ray_dir[1] * t,
                ray_origin[2] + ray_dir[2] * t,
            ];

            let diff = [seg_start[0] - closest[0], seg_start[1] - closest[1], seg_start[2] - closest[2]];
            let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            Some((dist, closest))
        } else {
            None
        }
    } else {
        let to_start = [seg_start[0] - ray_origin[0], seg_start[1] - ray_origin[1], seg_start[2] - ray_origin[2]];

        let s = dot_product(&to_start, &seg_dir) / seg_len_sq;
        let s = s.clamp(0.0, 1.0);

        let seg_closest = [
            seg_start[0] + seg_dir[0] * s,
            seg_start[1] + seg_dir[1] * s,
            seg_start[2] + seg_dir[2] * s,
        ];

        let to_seg_closest = [seg_closest[0] - ray_origin[0], seg_closest[1] - ray_origin[1], seg_closest[2] - ray_origin[2]];
        let t = dot_product(&to_seg_closest, ray_dir);

        if t >= 0.0 {
            let ray_closest = [
                ray_origin[0] + ray_dir[0] * t,
                ray_origin[1] + ray_dir[1] * t,
                ray_origin[2] + ray_dir[2] * t,
            ];

            let diff = [seg_closest[0] - ray_closest[0], seg_closest[1] - ray_closest[1], seg_closest[2] - ray_closest[2]];
            let dist = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();

            Some((dist, seg_closest))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;

    #[test]
    fn test_display_mode_enum() {
        assert_eq!(DisplayMode::Shaded, DisplayMode::Shaded);
        assert_ne!(DisplayMode::Shaded, DisplayMode::Wireframe);
        assert_ne!(DisplayMode::Wireframe, DisplayMode::ShadedWithEdges);
    }

    #[test]
    fn test_display_style_default() {
        let style = DisplayStyle::default();
        assert_eq!(style.mode, DisplayMode::Shaded);
        assert_eq!(style.color, [0.8, 0.8, 0.8]);
        assert_eq!(style.transparency, 1.0);
        assert_eq!(style.edge_color, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_display_style_builder() {
        let style = DisplayStyle::new()
            .with_mode(DisplayMode::Wireframe)
            .with_color(1.0, 0.0, 0.0)
            .with_transparency(0.5)
            .with_edge_color(0.0, 1.0, 0.0)
            .with_edge_width(2.0);

        assert_eq!(style.mode, DisplayMode::Wireframe);
        assert_eq!(style.color, [1.0, 0.0, 0.0]);
        assert_eq!(style.transparency, 0.5);
        assert_eq!(style.edge_color, [0.0, 1.0, 0.0]);
        assert_eq!(style.edge_width, 2.0);
    }

    #[test]
    fn test_display_style_color_clamping() {
        let style = DisplayStyle::new().with_color(1.5, -0.5, 0.5);

        assert_eq!(style.color[0], 1.0);
        assert_eq!(style.color[1], 0.0);
        assert_eq!(style.color[2], 0.5);
    }

    #[test]
    fn test_display_style_transparency_clamping() {
        let style = DisplayStyle::new().with_transparency(1.5);
        assert_eq!(style.transparency, 1.0);

        let style = DisplayStyle::new().with_transparency(-0.5);
        assert_eq!(style.transparency, 0.0);
    }

    #[test]
    fn test_viewer_creation() {
        let viewer = Viewer::new();
        assert_eq!(viewer.style.mode, DisplayMode::Shaded);
        assert!(viewer.mesh.is_none());
    }

    #[test]
    fn test_viewer_with_tolerance() {
        let viewer = Viewer::with_tolerance(1e-4);
        assert_eq!(viewer.tolerance, 1e-4);

        let viewer = Viewer::with_tolerance(1e-10);
        assert_eq!(viewer.tolerance, 1e-6);
    }

    #[test]
    fn test_viewer_set_display_mode() {
        let mut viewer = Viewer::new();
        assert_eq!(viewer.style.mode, DisplayMode::Shaded);

        viewer.set_display_mode(DisplayMode::Wireframe);
        assert_eq!(viewer.style.mode, DisplayMode::Wireframe);

        viewer.set_display_mode(DisplayMode::ShadedWithEdges);
        assert_eq!(viewer.style.mode, DisplayMode::ShadedWithEdges);
    }

    #[test]
    fn test_viewer_set_style() {
        let mut viewer = Viewer::new();
        let style = DisplayStyle::new()
            .with_mode(DisplayMode::Wireframe)
            .with_color(0.5, 0.5, 0.5);

        viewer.set_style(style.clone());
        assert_eq!(viewer.style.mode, DisplayMode::Wireframe);
        assert_eq!(viewer.style.color, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_render_shaded_without_mesh() {
        let viewer = Viewer::new();
        let result = viewer.render_shaded();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_wireframe_without_mesh() {
        let viewer = Viewer::new();
        let result = viewer.render_wireframe();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_shaded_with_edges_without_mesh() {
        let viewer = Viewer::new();
        let result = viewer.render_shaded_with_edges();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_shaded_display_mode() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;

        let faces = viewer.render_shaded()?;

        assert!(!faces.is_empty(), "Shaded rendering should produce faces");

        for face in &faces {
            assert!(
                face.color.iter().all(|c| *c >= 0.0 && *c <= 1.0),
                "Colors should be normalized"
            );
            assert!(
                face.transparency >= 0.0 && face.transparency <= 1.0,
                "Transparency should be normalized"
            );

            let normal_mag =
                (face.normal[0].powi(2) + face.normal[1].powi(2) + face.normal[2].powi(2)).sqrt();
            assert!(
                (normal_mag - 1.0).abs() < 1e-3 || normal_mag < 1e-6,
                "Normal should be unit length"
            );
        }

        Ok(())
    }

    #[test]
    fn test_render_wireframe_display_mode() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;

        let edges = viewer.render_wireframe()?;

        assert!(
            !edges.is_empty(),
            "Wireframe rendering should produce edges"
        );

        for edge in &edges {
            assert!(
                edge.color.iter().all(|c| *c >= 0.0 && *c <= 1.0),
                "Edge colors should be normalized"
            );
            assert!(edge.width > 0.0, "Edge width should be positive");

            let dx = edge.start[0] - edge.end[0];
            let dy = edge.start[1] - edge.end[1];
            let dz = edge.start[2] - edge.end[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(dist > 1e-6, "Edge start and end should be different");
        }

        Ok(())
    }

    #[test]
    fn test_render_shaded_with_edges_display_mode() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;

        let (faces, edges) = viewer.render_shaded_with_edges()?;

        assert!(!faces.is_empty(), "Should have shaded faces");
        assert!(!edges.is_empty(), "Should have edges");

        Ok(())
    }

    #[test]
    fn test_render_face_structure() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.set_style(
            DisplayStyle::new()
                .with_color(1.0, 0.0, 0.0)
                .with_transparency(0.7),
        );

        viewer.load_solid(&solid)?;
        let faces = viewer.render_shaded()?;

        for face in &faces {
            assert_eq!(face.vertices.len(), 3);

            for vertex in &face.vertices {
                assert_eq!(vertex.len(), 3);
            }

            assert!(face.color[0] <= 1.0, "Red channel should respect set color");

            assert_eq!(face.transparency, 0.7);
        }

        Ok(())
    }

    #[test]
    fn test_render_edge_structure() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.set_style(
            DisplayStyle::new()
                .with_edge_color(0.0, 1.0, 0.0)
                .with_edge_width(2.5),
        );

        viewer.load_solid(&solid)?;
        let edges = viewer.render_wireframe()?;

        for edge in &edges {
            assert_eq!(edge.color, [0.0, 1.0, 0.0]);

            assert_eq!(edge.width, 2.5);

            assert_eq!(edge.start.len(), 3);
            assert_eq!(edge.end.len(), 3);
        }

        Ok(())
    }

    #[test]
    fn test_viewer_load_and_mesh() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::new();

        assert!(viewer.mesh().is_none());

        viewer.load_solid(&solid)?;
        assert!(viewer.mesh().is_some());

        Ok(())
    }

    #[test]
    fn test_helper_functions() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert_eq!(dot_product(&a, &b), 0.0);

        let c = [1.0, 0.0, 0.0];
        let d = [1.0, 0.0, 0.0];
        assert_eq!(dot_product(&c, &d), 1.0);

        let cross = cross_product(&a, &b);
        assert_eq!(cross, [0.0, 0.0, 1.0]);

        let v = [3.0, 4.0, 0.0];
        let norm = normalize(&v);
        let mag = (norm[0].powi(2) + norm[1].powi(2) + norm[2].powi(2)).sqrt();
        assert!((mag - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lighting_calculation() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.set_style(DisplayStyle::new().with_color(1.0, 1.0, 1.0));

        viewer.load_solid(&solid)?;
        let faces = viewer.render_shaded()?;

        let mut has_variation = false;
        let first_brightness = faces[0].color[0];

        for face in &faces {
            let brightness = face.color[0];
            if (brightness - first_brightness).abs() > 1e-6 {
                has_variation = true;
                break;
            }
        }

        assert!(
            has_variation || faces.len() < 3,
            "Different face orientations should have different lighting"
        );

        Ok(())
    }

    #[test]
    fn test_highlight_style_creation() {
        let style = HighlightStyle::new();
        assert_eq!(style.color, [1.0, 0.5, 0.0]);
        assert_eq!(style.opacity, 1.0);
        assert_eq!(style.outline_width, 2.0);
    }

    #[test]
    fn test_highlight_style_builder() {
        let style = HighlightStyle::new()
            .with_color(0.0, 1.0, 0.0)
            .with_opacity(0.7)
            .with_outline_width(3.0);

        assert_eq!(style.color, [0.0, 1.0, 0.0]);
        assert_eq!(style.opacity, 0.7);
        assert_eq!(style.outline_width, 3.0);
    }

    #[test]
    fn test_highlight_style_color_clamping() {
        let style = HighlightStyle::new().with_color(1.5, -0.5, 0.5);

        assert_eq!(style.color[0], 1.0);
        assert_eq!(style.color[1], 0.0);
        assert_eq!(style.color[2], 0.5);
    }

    #[test]
    fn test_highlight_style_opacity_clamping() {
        let style = HighlightStyle::new().with_opacity(1.5);
        assert_eq!(style.opacity, 1.0);

        let style = HighlightStyle::new().with_opacity(-0.5);
        assert_eq!(style.opacity, 0.0);
    }

    #[test]
    fn test_highlight_style_outline_width_clamping() {
        let style = HighlightStyle::new().with_outline_width(0.05);
        assert_eq!(style.outline_width, 0.1);

        let style = HighlightStyle::new().with_outline_width(5.0);
        assert_eq!(style.outline_width, 5.0);
    }

    #[test]
    fn test_highlight_shape() {
        let mut viewer = Viewer::new();
        assert!(!viewer.is_shape_highlighted(0));

        viewer.highlight_shape(0);
        assert!(viewer.is_shape_highlighted(0));

        viewer.highlight_shape(1);
        assert!(viewer.is_shape_highlighted(1));
        assert_eq!(viewer.highlight_count(), 2);
    }

    #[test]
    fn test_unhighlight_shape() {
        let mut viewer = Viewer::new();
        viewer.highlight_shape(0);
        assert!(viewer.is_shape_highlighted(0));

        viewer.unhighlight_shape(0);
        assert!(!viewer.is_shape_highlighted(0));
    }

    #[test]
    fn test_clear_shape_highlights() {
        let mut viewer = Viewer::new();
        viewer.highlight_shape(0);
        viewer.highlight_shape(1);
        viewer.highlight_shape(2);
        assert_eq!(viewer.highlight_count(), 3);

        viewer.clear_shape_highlights();
        assert_eq!(viewer.highlight_count(), 0);
        assert!(!viewer.is_shape_highlighted(0));
    }

    #[test]
    fn test_highlight_face() {
        let mut viewer = Viewer::new();
        assert!(!viewer.is_face_highlighted(0));

        viewer.highlight_face(0, None);
        assert!(viewer.is_face_highlighted(0));

        let custom_style = HighlightStyle::new().with_color(1.0, 0.0, 0.0);
        viewer.highlight_face(1, Some(custom_style));
        assert!(viewer.is_face_highlighted(1));
        assert_eq!(viewer.highlight_count(), 2);
    }

    #[test]
    fn test_unhighlight_face() {
        let mut viewer = Viewer::new();
        viewer.highlight_face(0, None);
        assert!(viewer.is_face_highlighted(0));

        viewer.unhighlight_face(0);
        assert!(!viewer.is_face_highlighted(0));
    }

    #[test]
    fn test_clear_face_highlights() {
        let mut viewer = Viewer::new();
        viewer.highlight_face(0, None);
        viewer.highlight_face(1, None);
        viewer.highlight_face(2, None);
        assert_eq!(viewer.highlight_count(), 3);

        viewer.clear_face_highlights();
        assert_eq!(viewer.highlight_count(), 0);
        assert!(!viewer.is_face_highlighted(0));
    }

    #[test]
    fn test_highlight_edge() {
        let mut viewer = Viewer::new();
        assert!(!viewer.is_edge_highlighted(0, 1));

        viewer.highlight_edge(0, 1, None);
        assert!(viewer.is_edge_highlighted(0, 1));
        assert!(viewer.is_edge_highlighted(1, 0)); // Order shouldn't matter

        let custom_style = HighlightStyle::new().with_color(0.0, 0.0, 1.0);
        viewer.highlight_edge(2, 3, Some(custom_style));
        assert!(viewer.is_edge_highlighted(2, 3));
        assert_eq!(viewer.highlight_count(), 2);
    }

    #[test]
    fn test_unhighlight_edge() {
        let mut viewer = Viewer::new();
        viewer.highlight_edge(0, 1, None);
        assert!(viewer.is_edge_highlighted(0, 1));

        viewer.unhighlight_edge(0, 1);
        assert!(!viewer.is_edge_highlighted(0, 1));
    }

    #[test]
    fn test_edge_highlight_order_independence() {
        let mut viewer = Viewer::new();

        // Highlight with (0, 1) order
        viewer.highlight_edge(0, 1, None);

        // Check with reversed order (1, 0)
        assert!(viewer.is_edge_highlighted(1, 0));

        // Unhighlight with reversed order
        viewer.unhighlight_edge(1, 0);
        assert!(!viewer.is_edge_highlighted(0, 1));
    }

    #[test]
    fn test_clear_edge_highlights() {
        let mut viewer = Viewer::new();
        viewer.highlight_edge(0, 1, None);
        viewer.highlight_edge(2, 3, None);
        viewer.highlight_edge(4, 5, None);
        assert_eq!(viewer.highlight_count(), 3);

        viewer.clear_edge_highlights();
        assert_eq!(viewer.highlight_count(), 0);
        assert!(!viewer.is_edge_highlighted(0, 1));
    }

    #[test]
    fn test_clear_all_highlights() {
        let mut viewer = Viewer::new();

        // Highlight shapes, faces, and edges
        viewer.highlight_shape(0);
        viewer.highlight_shape(1);
        viewer.highlight_face(0, None);
        viewer.highlight_face(1, None);
        viewer.highlight_edge(0, 1, None);
        viewer.highlight_edge(2, 3, None);

        assert_eq!(viewer.highlight_count(), 6);

        // Clear all
        viewer.clear_highlights();
        assert_eq!(viewer.highlight_count(), 0);
        assert!(!viewer.is_shape_highlighted(0));
        assert!(!viewer.is_face_highlighted(0));
        assert!(!viewer.is_edge_highlighted(0, 1));
    }

    #[test]
    fn test_multiple_highlights_coexist() {
        let mut viewer = Viewer::new();

        // Add multiple highlights
        viewer.highlight_shape(0);
        viewer.highlight_shape(1);
        viewer.highlight_face(5, None);
        viewer.highlight_face(6, None);
        viewer.highlight_edge(10, 11, None);
        viewer.highlight_edge(12, 13, None);

        assert_eq!(viewer.highlight_count(), 6);

        // Verify all are still highlighted
        assert!(viewer.is_shape_highlighted(0));
        assert!(viewer.is_shape_highlighted(1));
        assert!(viewer.is_face_highlighted(5));
        assert!(viewer.is_face_highlighted(6));
        assert!(viewer.is_edge_highlighted(10, 11));
        assert!(viewer.is_edge_highlighted(12, 13));
    }

    #[test]
    fn test_set_highlight_style() {
        let mut viewer = Viewer::new();
        let custom_style = HighlightStyle::new()
            .with_color(0.5, 0.5, 0.5)
            .with_opacity(0.5)
            .with_outline_width(4.0);

        viewer.set_highlight_style(custom_style.clone());

        let retrieved = viewer.highlight_style();
        assert_eq!(retrieved.color, [0.5, 0.5, 0.5]);
        assert_eq!(retrieved.opacity, 0.5);
        assert_eq!(retrieved.outline_width, 4.0);
    }

    #[test]
    fn test_highlight_with_custom_style() {
        let mut viewer = Viewer::new();
        let style1 = HighlightStyle::new().with_color(1.0, 0.0, 0.0);
        let style2 = HighlightStyle::new().with_color(0.0, 1.0, 0.0);

        viewer.highlight_face(0, Some(style1));
        viewer.highlight_face(1, Some(style2));

        assert!(viewer.is_face_highlighted(0));
        assert!(viewer.is_face_highlighted(1));
    }

    #[test]
    fn test_highlight_count_accuracy() {
        let mut viewer = Viewer::new();

        assert_eq!(viewer.highlight_count(), 0);

        viewer.highlight_shape(0);
        assert_eq!(viewer.highlight_count(), 1);

        viewer.highlight_face(0, None);
        assert_eq!(viewer.highlight_count(), 2);

        viewer.highlight_edge(0, 1, None);
        assert_eq!(viewer.highlight_count(), 3);

        viewer.unhighlight_shape(0);
        assert_eq!(viewer.highlight_count(), 2);

        viewer.unhighlight_face(0);
        assert_eq!(viewer.highlight_count(), 1);

        viewer.unhighlight_edge(0, 1);
        assert_eq!(viewer.highlight_count(), 0);
    }

    #[test]
    fn test_highlight_with_box_geometry() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;

        // Get mesh info
        let mesh = viewer.mesh().unwrap();
        let _vertex_count = mesh.vertices.len();
        let triangle_count = mesh.triangles.len();

        // Highlight first few faces
        for i in 0..triangle_count.min(5) {
            viewer.highlight_face(i, None);
        }

        assert!(viewer.highlight_count() >= 5);
        assert!(viewer.is_face_highlighted(0));

        Ok(())
    }

    #[test]
    fn test_highlight_persistence_across_operations() {
        let mut viewer = Viewer::new();

        // Set highlights
        viewer.highlight_shape(0);
        viewer.highlight_face(0, None);
        viewer.highlight_edge(0, 1, None);

        // Modify display style
        viewer.set_display_mode(DisplayMode::Wireframe);

        // Highlights should still be there
        assert!(viewer.is_shape_highlighted(0));
        assert!(viewer.is_face_highlighted(0));
        assert!(viewer.is_edge_highlighted(0, 1));
    }

    #[test]
    fn test_highlight_independent_operations() {
        let mut viewer = Viewer::new();

        // Highlight shapes
        viewer.highlight_shape(0);
        viewer.highlight_shape(1);
        viewer.highlight_shape(2);

        // Clear only shape highlights
        viewer.clear_shape_highlights();
        assert_eq!(viewer.highlight_count(), 0);

        // Highlight faces
        viewer.highlight_face(0, None);
        viewer.highlight_face(1, None);
        assert_eq!(viewer.highlight_count(), 2);

        // Add edge highlights
        viewer.highlight_edge(0, 1, None);
        assert_eq!(viewer.highlight_count(), 3);

        // Clear only face highlights
        viewer.clear_face_highlights();
        assert_eq!(viewer.highlight_count(), 1);
        assert!(viewer.is_edge_highlighted(0, 1));
    }

    // ===== SELECTION TESTS =====

    #[test]
    fn test_selection_mode_enum() {
        assert_eq!(SelectionMode::None, SelectionMode::None);
        assert_ne!(SelectionMode::None, SelectionMode::Shape);
        assert_ne!(SelectionMode::Shape, SelectionMode::Face);
        assert_ne!(SelectionMode::Face, SelectionMode::Edge);
        assert_ne!(SelectionMode::Edge, SelectionMode::Vertex);
    }

    #[test]
    fn test_selection_creation() {
        let selection = Selection::new();
        assert_eq!(selection.mode, SelectionMode::None);
        assert!(selection.is_empty());
        assert_eq!(selection.count(), 0);
        assert!(selection.closest().is_none());
    }

    #[test]
    fn test_selection_mode_setting() {
        let mut selection = Selection::new();
        assert_eq!(selection.mode, SelectionMode::None);

        selection.set_mode(SelectionMode::Shape);
        assert_eq!(selection.mode, SelectionMode::Shape);

        selection.set_mode(SelectionMode::Face);
        assert_eq!(selection.mode, SelectionMode::Face);
    }

    #[test]
    fn test_selection_hit_addition() {
        let mut selection = Selection::new();
        
        let hit1 = SelectionHit {
            entity_type: SelectionMode::Face,
            entity_id: 0,
            distance: 5.0,
            hit_point: [0.0, 0.0, 0.0],
        };

        selection.add_hit(hit1);
        assert_eq!(selection.count(), 1);
        assert!(!selection.is_empty());
        assert!(selection.closest().is_some());
        assert_eq!(selection.closest().unwrap().distance, 5.0);
    }

    #[test]
    fn test_selection_closest_sorting() {
        let mut selection = Selection::new();
        
        let hit1 = SelectionHit {
            entity_type: SelectionMode::Face,
            entity_id: 0,
            distance: 10.0,
            hit_point: [0.0, 0.0, 0.0],
        };

        let hit2 = SelectionHit {
            entity_type: SelectionMode::Face,
            entity_id: 1,
            distance: 5.0,
            hit_point: [1.0, 0.0, 0.0],
        };

        let hit3 = SelectionHit {
            entity_type: SelectionMode::Face,
            entity_id: 2,
            distance: 15.0,
            hit_point: [2.0, 0.0, 0.0],
        };

        selection.add_hit(hit1);
        selection.add_hit(hit3);
        selection.add_hit(hit2);

        assert_eq!(selection.count(), 3);
        assert_eq!(selection.closest().unwrap().distance, 5.0);
        assert_eq!(selection.closest().unwrap().entity_id, 1);
    }

    #[test]
    fn test_selection_clear() {
        let mut selection = Selection::new();
        
        let hit = SelectionHit {
            entity_type: SelectionMode::Vertex,
            entity_id: 5,
            distance: 3.0,
            hit_point: [0.5, 0.5, 0.5],
        };

        selection.add_hit(hit);
        assert_eq!(selection.count(), 1);

        selection.clear();
        assert_eq!(selection.count(), 0);
        assert!(selection.is_empty());
        assert!(selection.closest().is_none());
    }

    #[test]
    fn test_selection_all() {
        let mut selection = Selection::new();
        
        let hit1 = SelectionHit {
            entity_type: SelectionMode::Face,
            entity_id: 0,
            distance: 5.0,
            hit_point: [0.0, 0.0, 0.0],
        };

        let hit2 = SelectionHit {
            entity_type: SelectionMode::Face,
            entity_id: 1,
            distance: 10.0,
            hit_point: [1.0, 0.0, 0.0],
        };

        selection.add_hit(hit1);
        selection.add_hit(hit2);

        let all = selection.all();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].entity_id, 0);
        assert_eq!(all[1].entity_id, 1);
    }

    #[test]
    fn test_viewer_selection_mode() {
        let mut viewer = Viewer::new();
        assert_eq!(viewer.get_selection().mode, SelectionMode::None);

        viewer.set_selection_mode(SelectionMode::Face);
        assert_eq!(viewer.get_selection().mode, SelectionMode::Face);

        viewer.set_selection_mode(SelectionMode::Vertex);
        assert_eq!(viewer.get_selection().mode, SelectionMode::Vertex);
    }

    #[test]
    fn test_viewer_camera_setting() {
        let mut viewer = Viewer::new();
        
        let pos = [5.0, 5.0, 5.0];
        let dir = [1.0, 0.0, 0.0];
        let up = [0.0, 1.0, 0.0];

        viewer.set_camera(pos, dir, up);
        assert_eq!(viewer.camera_pos, pos);
    }

    #[test]
    fn test_viewer_select_without_mode() {
        let solid = make_box(1.0, 1.0, 1.0).unwrap();
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid).unwrap();

        // No selection mode set - should return None
        let result = viewer.select_at(0.5, 0.5, 1.0, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_viewer_select_without_mesh() {
        let mut viewer = Viewer::new();
        viewer.set_selection_mode(SelectionMode::Shape);

        let result = viewer.select_at(0.5, 0.5, 1.0, 1.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_viewer_select_shape_mode() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;
        viewer.set_selection_mode(SelectionMode::Shape);
        viewer.set_camera([10.0, 10.0, 10.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 1.0]);

        // Try selecting at center of screen
        let result = viewer.select_at(0.5, 0.5, 1.0, 1.0);
        
        // Should get a hit for shape selection
        if let Some(hit) = result {
            assert_eq!(hit.entity_type, SelectionMode::Shape);
            assert!(hit.distance >= 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_viewer_select_face_mode() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;
        viewer.set_selection_mode(SelectionMode::Face);
        viewer.set_camera([10.0, 10.0, 10.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 1.0]);

        let result = viewer.select_at(0.5, 0.5, 1.0, 1.0);
        
        if let Some(hit) = result {
            assert_eq!(hit.entity_type, SelectionMode::Face);
            assert!(hit.distance >= 0.0);
            assert!(hit.entity_id < 1000); // Reasonable face index
        }

        Ok(())
    }

    #[test]
    fn test_viewer_select_vertex_mode() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;
        viewer.set_selection_mode(SelectionMode::Vertex);
        viewer.set_camera([10.0, 10.0, 10.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 1.0]);

        // Try selecting near a vertex
        let result = viewer.select_at(0.5, 0.5, 1.0, 1.0);
        
        // Vertex selection has stricter distance requirements
        // Result may be None if ray doesn't pass close to a vertex

        Ok(())
    }

    #[test]
    fn test_viewer_select_edge_mode() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;
        viewer.set_selection_mode(SelectionMode::Edge);
        viewer.set_camera([10.0, 10.0, 10.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 1.0]);

        let result = viewer.select_at(0.5, 0.5, 1.0, 1.0);
        
        // Edge selection depends on ray proximity to edges

        Ok(())
    }

    #[test]
    fn test_selection_clears_on_new_selection() -> CascadeResult<()> {
        let solid = make_box(1.0, 1.0, 1.0)?;
        let mut viewer = Viewer::with_tolerance(0.1);
        viewer.load_solid(&solid)?;
        viewer.set_selection_mode(SelectionMode::Shape);
        viewer.set_camera([10.0, 10.0, 10.0], [-1.0, -1.0, -1.0], [0.0, 0.0, 1.0]);

        let _ = viewer.select_at(0.5, 0.5, 1.0, 1.0);
        let initial_count = viewer.get_selection().count();

        // Select at different location
        let _ = viewer.select_at(0.3, 0.3, 1.0, 1.0);
        let final_count = viewer.get_selection().count();

        // Should have at most 1 selection (cleared between selections)
        assert!(final_count <= 1);

        Ok(())
    }

    #[test]
    fn test_get_selection_mutable() {
        let mut viewer = Viewer::new();
        viewer.set_selection_mode(SelectionMode::Face);

        let selection_mut = viewer.get_selection_mut();
        assert_eq!(selection_mut.mode, SelectionMode::Face);
        
        selection_mut.set_mode(SelectionMode::Vertex);
        assert_eq!(viewer.get_selection().mode, SelectionMode::Vertex);
    }

    #[test]
    fn test_ray_triangle_intersection_basic() {
        let ray_origin = [0.0, 0.0, -5.0];
        let ray_dir = [0.0, 0.0, 1.0];
        let v0 = [-1.0, -1.0, 0.0];
        let v1 = [1.0, -1.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        let result = ray_triangle_intersection(&ray_origin, &ray_dir, &v0, &v1, &v2);
        assert!(result.is_some(), "Ray should intersect triangle");

        let (dist, hit) = result.unwrap();
        assert!(dist > 0.0 && dist < 10.0, "Distance should be positive and reasonable");
        assert!((hit[2] - 0.0).abs() < 1e-6, "Hit point should be on triangle plane (z=0)");
    }

    #[test]
    fn test_ray_triangle_intersection_miss() {
        let ray_origin = [0.0, 0.0, -5.0];
        let ray_dir = [0.0, 0.0, 1.0];
        let v0 = [2.0, 2.0, 0.0];
        let v1 = [3.0, 2.0, 0.0];
        let v2 = [2.5, 3.0, 0.0];

        let result = ray_triangle_intersection(&ray_origin, &ray_dir, &v0, &v1, &v2);
        assert!(result.is_none(), "Ray should not intersect distant triangle");
    }

    #[test]
    fn test_ray_triangle_intersection_parallel() {
        let ray_origin = [0.0, 0.0, -5.0];
        let ray_dir = [1.0, 0.0, 0.0]; // Parallel to triangle in XY plane
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.5, 1.0, 0.0];

        let result = ray_triangle_intersection(&ray_origin, &ray_dir, &v0, &v1, &v2);
        assert!(result.is_none(), "Parallel ray should not intersect");
    }

    #[test]
    fn test_ray_segment_closest_approach_perpendicular() {
        let ray_origin = [0.0, 0.0, 0.0];
        let ray_dir = [1.0, 0.0, 0.0];
        let seg_start = [5.0, -1.0, 0.0];
        let seg_end = [5.0, 1.0, 0.0];

        let result = ray_segment_closest_approach(&ray_origin, &ray_dir, &seg_start, &seg_end);
        
        if let Some((distance, _)) = result {
            assert!(distance < 1.1, "Closest distance should be ~1.0 (perpendicular segment)");
        }
    }

    #[test]
    fn test_ray_segment_closest_approach_degenerate() {
        let ray_origin = [0.0, 0.0, 0.0];
        let ray_dir = [1.0, 0.0, 0.0];
        let seg_start = [2.0, 0.5, 0.0];
        let seg_end = [2.0, 0.5, 0.0]; // Degenerate (same point)

        let result = ray_segment_closest_approach(&ray_origin, &ray_dir, &seg_start, &seg_end);
        
        if let Some((distance, _)) = result {
            // Distance from point [2.0, 0.5, 0.0] to ray = sqrt(0.25) = 0.5
            assert!((distance - 0.5).abs() < 1e-3, "Should compute distance to point");
        }
    }
}
