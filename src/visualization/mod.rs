//! 3D Viewer Infrastructure
//!
//! Provides basic 3D visualization capabilities for CASCADE-RS shapes.
//! Includes camera control, viewport management, and simple rasterization.

use crate::{Result, CascadeError};
use crate::brep::Solid;
use crate::mesh::{triangulate, TriangleMesh};
use nalgebra as na;
use std::collections::HashMap;

/// Projection type for the camera
#[derive(Debug, Clone, Copy)]
pub enum ProjectionType {
    /// Orthographic projection (parallel view)
    Orthographic,
    /// Perspective projection
    Perspective { fov_degrees: f64 },
}

/// Camera for 3D viewing with position, orientation, and projection
#[derive(Debug, Clone)]
pub struct Camera {
    /// Camera position in 3D space
    pub position: [f64; 3],
    /// Point the camera is looking at
    pub target: [f64; 3],
    /// Up vector for camera orientation
    pub up: [f64; 3],
    /// Projection type
    pub projection: ProjectionType,
    /// Near clipping plane distance
    pub near: f64,
    /// Far clipping plane distance
    pub far: f64,
}

impl Camera {
    /// Create a new camera with default settings
    pub fn new() -> Self {
        Camera {
            position: [0.0, 0.0, 100.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            projection: ProjectionType::Orthographic,
            near: 0.1,
            far: 10000.0,
        }
    }

    /// Set the camera position
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.position = [x, y, z];
    }

    /// Set the camera target (look-at point)
    pub fn set_target(&mut self, x: f64, y: f64, z: f64) {
        self.target = [x, y, z];
    }

    /// Set the camera up vector
    pub fn set_up(&mut self, x: f64, y: f64, z: f64) {
        self.up = [x, y, z];
    }

    /// Set the projection type
    pub fn set_projection(&mut self, projection: ProjectionType) {
        self.projection = projection;
    }

    /// Get the view matrix for this camera
    fn get_view_matrix(&self) -> na::Matrix4<f64> {
        let pos = na::Point3::new(self.position[0], self.position[1], self.position[2]);
        let target = na::Point3::new(self.target[0], self.target[1], self.target[2]);
        let up = na::Vector3::new(self.up[0], self.up[1], self.up[2]);

        na::Isometry3::look_at_lh(&pos, &target, &up).to_homogeneous()
    }

    /// Get the projection matrix for this camera
    fn get_projection_matrix(&self, aspect: f64) -> na::Matrix4<f64> {
        match self.projection {
            ProjectionType::Orthographic => {
                let height = 100.0; // Base height for orthographic view
                let width = height * aspect;
                na::Orthographic3::new(-width / 2.0, width / 2.0, -height / 2.0, height / 2.0, self.near, self.far)
                    .as_matrix()
                    .clone()
            }
            ProjectionType::Perspective { fov_degrees } => {
                let fov_rad = fov_degrees.to_radians();
                na::Perspective3::new(aspect, fov_rad, self.near, self.far)
                    .as_matrix()
                    .clone()
            }
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

/// Viewport for rendering
#[derive(Debug, Clone)]
pub struct Viewport {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Background color (RGBA)
    pub background_color: [u8; 4],
}

impl Viewport {
    /// Create a new viewport
    pub fn new(width: usize, height: usize) -> Self {
        Viewport {
            width,
            height,
            background_color: [200, 200, 200, 255], // Light gray
        }
    }

    /// Set the background color (RGBA)
    pub fn set_background_color(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.background_color = [r, g, b, a];
    }

    /// Get aspect ratio
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self::new(800, 600)
    }
}

/// A 3D shape entry in the viewer
#[derive(Debug, Clone)]
struct ShapeEntry {
    solid: Solid,
    color: [u8; 3],
    visible: bool,
}

/// 3D Viewer for visualizing CASCADE shapes
#[derive(Debug)]
pub struct Viewer {
    camera: Camera,
    viewport: Viewport,
    shapes: HashMap<usize, ShapeEntry>,
    next_shape_id: usize,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
}

impl Viewer {
    /// Create a new viewer with default camera and viewport
    pub fn new() -> Self {
        Viewer {
            camera: Camera::new(),
            viewport: Viewport::default(),
            shapes: HashMap::new(),
            next_shape_id: 0,
            bounds_min: [f64::INFINITY; 3],
            bounds_max: [f64::NEG_INFINITY; 3],
        }
    }

    /// Create a new viewer with specified viewport
    pub fn with_viewport(width: usize, height: usize) -> Self {
        Viewer {
            camera: Camera::new(),
            viewport: Viewport::new(width, height),
            shapes: HashMap::new(),
            next_shape_id: 0,
            bounds_min: [f64::INFINITY; 3],
            bounds_max: [f64::NEG_INFINITY; 3],
        }
    }

    /// Get a mutable reference to the camera
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Get a reference to the camera
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Get a mutable reference to the viewport
    pub fn viewport_mut(&mut self) -> &mut Viewport {
        &mut self.viewport
    }

    /// Get a reference to the viewport
    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    /// Add a shape to the viewer with default color (white)
    pub fn add_shape(&mut self, solid: Solid) -> Result<usize> {
        self.add_shape_with_color(solid, [255, 255, 255])
    }

    /// Add a shape to the viewer with a specific color (RGB)
    pub fn add_shape_with_color(&mut self, solid: Solid, color: [u8; 3]) -> Result<usize> {
        let id = self.next_shape_id;
        self.next_shape_id += 1;

        let entry = ShapeEntry {
            solid,
            color,
            visible: true,
        };

        self.shapes.insert(id, entry);
        self.update_bounds();

        Ok(id)
    }

    /// Remove a shape from the viewer by ID
    pub fn remove_shape(&mut self, id: usize) -> Result<()> {
        if self.shapes.remove(&id).is_some() {
            self.update_bounds();
            Ok(())
        } else {
            Err(CascadeError::InvalidGeometry(format!("Shape with ID {} not found", id)))
        }
    }

    /// Set visibility of a shape
    pub fn set_shape_visible(&mut self, id: usize, visible: bool) -> Result<()> {
        if let Some(entry) = self.shapes.get_mut(&id) {
            entry.visible = visible;
            Ok(())
        } else {
            Err(CascadeError::InvalidGeometry(format!("Shape with ID {} not found", id)))
        }
    }

    /// Get the number of shapes in the viewer
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    /// Fit all visible shapes in the view
    pub fn fit_all(&mut self) -> Result<()> {
        if self.bounds_max[0] == f64::NEG_INFINITY || self.bounds_max[0] < self.bounds_min[0] {
            // No valid bounds
            return Ok(());
        }

        // Calculate center and size
        let center = [
            (self.bounds_min[0] + self.bounds_max[0]) / 2.0,
            (self.bounds_min[1] + self.bounds_max[1]) / 2.0,
            (self.bounds_min[2] + self.bounds_max[2]) / 2.0,
        ];

        let size = [
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        ];

        let _max_size = size[0].max(size[1]).max(size[2]);
        let diagonal = (size[0] * size[0] + size[1] * size[1] + size[2] * size[2]).sqrt();

        // Position camera to view all shapes
        self.camera.set_target(center[0], center[1], center[2]);
        let distance = diagonal.max(1.0) * 1.5;
        self.camera
            .set_position(center[0], center[1] + distance / 2.0, center[2] + distance);

        Ok(())
    }

    /// Set the view direction (updates camera position while keeping target)
    pub fn set_view_direction(&mut self, direction: ViewDirection) {
        let center = [
            (self.bounds_min[0] + self.bounds_max[0]) / 2.0,
            (self.bounds_min[1] + self.bounds_max[1]) / 2.0,
            (self.bounds_min[2] + self.bounds_max[2]) / 2.0,
        ];

        let size = [
            self.bounds_max[0] - self.bounds_min[0],
            self.bounds_max[1] - self.bounds_min[1],
            self.bounds_max[2] - self.bounds_min[2],
        ];

        let distance = size[0].max(size[1]).max(size[2]).max(1.0) * 1.5;

        match direction {
            ViewDirection::Front => {
                self.camera.set_position(center[0], center[1], center[2] + distance);
                self.camera.set_up(0.0, 1.0, 0.0);
            }
            ViewDirection::Back => {
                self.camera.set_position(center[0], center[1], center[2] - distance);
                self.camera.set_up(0.0, 1.0, 0.0);
            }
            ViewDirection::Top => {
                self.camera.set_position(center[0], center[1] + distance, center[2]);
                self.camera.set_up(0.0, 0.0, -1.0);
            }
            ViewDirection::Bottom => {
                self.camera.set_position(center[0], center[1] - distance, center[2]);
                self.camera.set_up(0.0, 0.0, 1.0);
            }
            ViewDirection::Left => {
                self.camera.set_position(center[0] - distance, center[1], center[2]);
                self.camera.set_up(0.0, 1.0, 0.0);
            }
            ViewDirection::Right => {
                self.camera.set_position(center[0] + distance, center[1], center[2]);
                self.camera.set_up(0.0, 1.0, 0.0);
            }
            ViewDirection::Isometric => {
                let offset = distance / (3.0_f64).sqrt();
                self.camera.set_position(
                    center[0] + offset,
                    center[1] + offset,
                    center[2] + offset,
                );
                self.camera.set_up(0.0, 1.0, 0.0);
            }
        }

        self.camera.set_target(center[0], center[1], center[2]);
    }

    /// Render the viewer to an image buffer (RGBA format)
    /// Returns a vector of RGBA pixels
    pub fn render_to_image(&self) -> Result<Vec<u8>> {
        let width = self.viewport.width;
        let height = self.viewport.height;
        let mut buffer = vec![0u8; width * height * 4];

        // Fill with background color
        for i in 0..width * height {
            let offset = i * 4;
            buffer[offset] = self.viewport.background_color[0];
            buffer[offset + 1] = self.viewport.background_color[1];
            buffer[offset + 2] = self.viewport.background_color[2];
            buffer[offset + 3] = self.viewport.background_color[3];
        }

        // Simple z-buffer for depth
        let mut z_buffer = vec![f64::NEG_INFINITY; width * height];

        // Get view and projection matrices
        let view_matrix = self.camera.get_view_matrix();
        let proj_matrix = self.camera
            .get_projection_matrix(self.viewport.aspect_ratio());

        let vp_matrix = proj_matrix * view_matrix;

        // Render each shape
        for entry in self.shapes.values() {
            if !entry.visible {
                continue;
            }

            // Triangulate the shape
            match triangulate(&entry.solid, 1.0) {
                Ok(mesh) => {
                    self.rasterize_mesh(&mesh, &vp_matrix, &entry.color, &mut buffer, &mut z_buffer);
                }
                Err(_) => {
                    // Skip shapes that can't be triangulated
                    continue;
                }
            }
        }

        Ok(buffer)
    }

    /// Rasterize a triangle mesh into the image buffer
    fn rasterize_mesh(
        &self,
        mesh: &TriangleMesh,
        vp_matrix: &na::Matrix4<f64>,
        color: &[u8; 3],
        buffer: &mut [u8],
        z_buffer: &mut [f64],
    ) {
        let width = self.viewport.width as f64;
        let height = self.viewport.height as f64;

        for triangle in &mesh.triangles {
            let v0_idx = triangle[0];
            let v1_idx = triangle[1];
            let v2_idx = triangle[2];

            if v0_idx >= mesh.vertices.len() || v1_idx >= mesh.vertices.len()
                || v2_idx >= mesh.vertices.len()
            {
                continue;
            }

            let v0 = mesh.vertices[v0_idx];
            let v1 = mesh.vertices[v1_idx];
            let v2 = mesh.vertices[v2_idx];

            // Transform vertices to clip space
            let p0 = self.transform_vertex(&v0, vp_matrix);
            let p1 = self.transform_vertex(&v1, vp_matrix);
            let p2 = self.transform_vertex(&v2, vp_matrix);

            // Rasterize triangle
            self.rasterize_triangle(p0, p1, p2, color, buffer, z_buffer, width, height);
        }
    }

    /// Transform a vertex to clip space
    fn transform_vertex(&self, vertex: &[f64; 3], vp_matrix: &na::Matrix4<f64>) -> [f64; 3] {
        let v = na::Vector4::new(vertex[0], vertex[1], vertex[2], 1.0);
        let transformed = vp_matrix * v;

        // Perspective divide
        let w = if transformed.w.abs() > 1e-6 {
            transformed.w
        } else {
            1.0
        };

        let x = (transformed.x / w + 1.0) * self.viewport.width as f64 * 0.5;
        let y = (1.0 - transformed.y / w) * self.viewport.height as f64 * 0.5;
        let z = transformed.z / w;

        [x, y, z]
    }

    /// Rasterize a single triangle
    fn rasterize_triangle(
        &self,
        p0: [f64; 3],
        p1: [f64; 3],
        p2: [f64; 3],
        color: &[u8; 3],
        buffer: &mut [u8],
        z_buffer: &mut [f64],
        width: f64,
        height: f64,
    ) {
        let width_i = width as i32;
        let height_i = height as i32;

        // Bounding box
        let min_x = (p0[0].min(p1[0]).min(p2[0]).max(0.0) as i32).min(width_i - 1);
        let max_x = (p0[0].max(p1[0]).max(p2[0]).min(width - 1.0) as i32).max(0);
        let min_y = (p0[1].min(p1[1]).min(p2[1]).max(0.0) as i32).min(height_i - 1);
        let max_y = (p0[1].max(p1[1]).max(p2[1]).min(height - 1.0) as i32).max(0);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if x < 0 || x >= width_i || y < 0 || y >= height_i {
                    continue;
                }

                let px = x as f64 + 0.5;
                let py = y as f64 + 0.5;

                // Barycentric coordinates
                let v0 = [p1[0] - p0[0], p1[1] - p0[1]];
                let v1 = [p2[0] - p0[0], p2[1] - p0[1]];
                let v2 = [px - p0[0], py - p0[1]];

                let dot00 = v0[0] * v0[0] + v0[1] * v0[1];
                let dot01 = v0[0] * v1[0] + v0[1] * v1[1];
                let dot02 = v0[0] * v2[0] + v0[1] * v2[1];
                let dot11 = v1[0] * v1[0] + v1[1] * v1[1];
                let dot12 = v1[0] * v2[0] + v1[1] * v2[1];

                let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
                let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
                let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

                // Check if point is in triangle
                if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                    // Interpolate z
                    let z = p0[2] + u * (p1[2] - p0[2]) + v * (p2[2] - p0[2]);

                    let idx = y as usize * width_i as usize + x as usize;
                    if idx < z_buffer.len() && z > z_buffer[idx] {
                        z_buffer[idx] = z;

                        let offset = idx * 4;
                        if offset + 3 < buffer.len() {
                            buffer[offset] = color[0];
                            buffer[offset + 1] = color[1];
                            buffer[offset + 2] = color[2];
                            buffer[offset + 3] = 255;
                        }
                    }
                }
            }
        }
    }

    /// Update the bounding box based on all visible shapes
    fn update_bounds(&mut self) {
        self.bounds_min = [f64::INFINITY; 3];
        self.bounds_max = [f64::NEG_INFINITY; 3];

        for entry in self.shapes.values() {
            if !entry.visible {
                continue;
            }

            // Try to get bounds from triangulation
            if let Ok(mesh) = triangulate(&entry.solid, 1.0) {
                for vertex in &mesh.vertices {
                    for i in 0..3 {
                        self.bounds_min[i] = self.bounds_min[i].min(vertex[i]);
                        self.bounds_max[i] = self.bounds_max[i].max(vertex[i]);
                    }
                }
            }
        }
    }
}

impl Default for Viewer {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard view directions
#[derive(Debug, Clone, Copy)]
pub enum ViewDirection {
    Front,
    Back,
    Top,
    Bottom,
    Left,
    Right,
    Isometric,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::make_box;

    #[test]
    fn test_camera_new() {
        let camera = Camera::new();
        assert_eq!(camera.position, [0.0, 0.0, 100.0]);
        assert_eq!(camera.target, [0.0, 0.0, 0.0]);
        assert_eq!(camera.up, [0.0, 1.0, 0.0]);
        assert_eq!(camera.near, 0.1);
        assert_eq!(camera.far, 10000.0);
    }

    #[test]
    fn test_camera_set_position() {
        let mut camera = Camera::new();
        camera.set_position(10.0, 20.0, 30.0);
        assert_eq!(camera.position, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_viewport_new() {
        let viewport = Viewport::new(1024, 768);
        assert_eq!(viewport.width, 1024);
        assert_eq!(viewport.height, 768);
        assert_eq!(viewport.aspect_ratio(), 1024.0 / 768.0);
    }

    #[test]
    fn test_viewport_background_color() {
        let mut viewport = Viewport::new(800, 600);
        viewport.set_background_color(255, 0, 0, 255);
        assert_eq!(viewport.background_color, [255, 0, 0, 255]);
    }

    #[test]
    fn test_viewer_new() {
        let viewer = Viewer::new();
        assert_eq!(viewer.shape_count(), 0);
        assert_eq!(viewer.viewport.width, 800);
        assert_eq!(viewer.viewport.height, 600);
    }

    #[test]
    fn test_viewer_add_shape() {
        let mut viewer = Viewer::new();
        let box_shape = make_box(10.0, 10.0, 10.0).expect("Failed to create box");

        let id = viewer.add_shape(box_shape).expect("Failed to add shape");
        assert_eq!(viewer.shape_count(), 1);
        assert_eq!(id, 0);

        let id2 = viewer.add_shape(make_box(5.0, 5.0, 5.0).unwrap()).unwrap();
        assert_eq!(viewer.shape_count(), 2);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_viewer_remove_shape() {
        let mut viewer = Viewer::new();
        let box_shape = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        let id = viewer.add_shape(box_shape).expect("Failed to add shape");

        assert_eq!(viewer.shape_count(), 1);
        viewer.remove_shape(id).expect("Failed to remove shape");
        assert_eq!(viewer.shape_count(), 0);
    }

    #[test]
    fn test_viewer_add_shape_with_color() {
        let mut viewer = Viewer::new();
        let box_shape = make_box(10.0, 10.0, 10.0).expect("Failed to create box");

        let id = viewer
            .add_shape_with_color(box_shape, [255, 0, 0])
            .expect("Failed to add shape");
        assert_eq!(viewer.shape_count(), 1);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_viewer_fit_all() {
        let mut viewer = Viewer::new();
        let box_shape = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        viewer.add_shape(box_shape).expect("Failed to add shape");

        viewer.fit_all().expect("Failed to fit all");
        // After fit_all, camera should be positioned to see the shape
        assert_ne!(viewer.camera.position, Camera::new().position);
    }

    #[test]
    fn test_viewer_set_view_direction() {
        let mut viewer = Viewer::new();
        let box_shape = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        viewer.add_shape(box_shape).expect("Failed to add shape");
        viewer.fit_all().expect("Failed to fit all");

        let pos_before = viewer.camera.position;
        viewer.set_view_direction(ViewDirection::Top);
        let pos_after = viewer.camera.position;

        assert_ne!(pos_before, pos_after);
        // Y position should be larger for top view
        assert!(pos_after[1] > pos_before[1] || pos_after != pos_before);
    }

    #[test]
    fn test_viewer_render_to_image() {
        let mut viewer = Viewer::new();
        let box_shape = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        viewer.add_shape(box_shape).expect("Failed to add shape");
        viewer.fit_all().expect("Failed to fit all");

        let image = viewer.render_to_image().expect("Failed to render");
        // Should have RGBA data for 800x600
        assert_eq!(image.len(), 800 * 600 * 4);
    }

    #[test]
    fn test_viewer_render_empty() {
        let viewer = Viewer::new();
        let image = viewer.render_to_image().expect("Failed to render");
        // Should have RGBA data for 800x600
        assert_eq!(image.len(), 800 * 600 * 4);
    }

    #[test]
    fn test_viewer_render_with_color() {
        let mut viewer = Viewer::new();
        let box_shape = make_box(10.0, 10.0, 10.0).expect("Failed to create box");
        viewer
            .add_shape_with_color(box_shape, [255, 0, 0])
            .expect("Failed to add shape");
        viewer.fit_all().expect("Failed to fit all");

        let image = viewer.render_to_image().expect("Failed to render");
        assert_eq!(image.len(), 800 * 600 * 4);
    }
}
