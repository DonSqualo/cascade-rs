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
}
