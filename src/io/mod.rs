//! File I/O for CAD formats

use crate::brep::{Shape, Solid, Shell, Face, Wire, Edge, Vertex, CurveType, SurfaceType, Compound};
use crate::{Result, CascadeError};
use std::io::Write;

pub fn read_step(path: &str) -> Result<Shape> {
    Err(CascadeError::NotImplemented("io::step_read".into()))
}

pub fn write_step(shape: &Shape, path: &str) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut step_writer = StepWriter::new(writer);
    step_writer.write_shape(shape)
}

pub fn read_brep(path: &str) -> Result<Shape> {
    Err(CascadeError::NotImplemented("io::brep_read".into()))
}

pub fn write_brep(shape: &Shape, path: &str) -> Result<()> {
    Err(CascadeError::NotImplemented("io::brep_write".into()))
}

/// STEP file writer (ISO 10303-21 format)
struct StepWriter<W: Write> {
    writer: W,
    entity_id: usize,
    entities: Vec<String>,
}

impl<W: Write> StepWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            entity_id: 0,
            entities: Vec::new(),
        }
    }
    
    fn next_id(&mut self) -> usize {
        self.entity_id += 1;
        self.entity_id
    }
    
    fn write_shape(&mut self, shape: &Shape) -> Result<()> {
        match shape {
            Shape::Vertex(v) => {
                self.add_vertex(v);
            }
            Shape::Edge(e) => {
                self.add_edge(e);
            }
            Shape::Wire(w) => {
                self.add_wire(w);
            }
            Shape::Face(f) => {
                self.add_face(f);
            }
            Shape::Shell(s) => {
                self.add_shell(s);
            }
            Shape::Solid(s) => {
                self.add_solid(s);
            }
            Shape::Compound(c) => {
                self.add_compound(c);
            }
        }
        
        self.write_file()?;
        Ok(())
    }
    
    fn add_vertex(&mut self, v: &Vertex) {
        let id = self.next_id();
        let entity = format!(
            "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
            id, v.point[0], v.point[1], v.point[2]
        );
        self.entities.push(entity);
    }
    
    fn add_edge(&mut self, e: &Edge) -> usize {
        let start_pt_id = self.next_id();
        let start_entity = format!(
            "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
            start_pt_id, e.start.point[0], e.start.point[1], e.start.point[2]
        );
        self.entities.push(start_entity);
        
        let end_pt_id = self.next_id();
        let end_entity = format!(
            "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
            end_pt_id, e.end.point[0], e.end.point[1], e.end.point[2]
        );
        self.entities.push(end_entity);
        
        let start_v_id = self.next_id();
        let start_v_entity = format!("#{} = VERTEX_POINT('', #{});", start_v_id, start_pt_id);
        self.entities.push(start_v_entity);
        
        let end_v_id = self.next_id();
        let end_v_entity = format!("#{} = VERTEX_POINT('', #{});", end_v_id, end_pt_id);
        self.entities.push(end_v_entity);
        
        let curve_id = match &e.curve_type {
            CurveType::Line => {
                let line_id = self.next_id();
                let dir_id = self.next_id();
                
                let dx = e.end.point[0] - e.start.point[0];
                let dy = e.end.point[1] - e.start.point[1];
                let dz = e.end.point[2] - e.start.point[2];
                
                let dir_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    dir_id, dx, dy, dz
                );
                self.entities.push(dir_entity);
                
                let line_entity = format!(
                    "#{} = LINE('', #{}, #{});",
                    line_id, start_pt_id, dir_id
                );
                self.entities.push(line_entity);
                line_id
            }
            CurveType::Arc { center, radius } => {
                let arc_id = self.next_id();
                let center_id = self.next_id();
                let axis_id = self.next_id();
                
                let center_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    center_id, center[0], center[1], center[2]
                );
                self.entities.push(center_entity);
                
                let axis_entity = format!("#{} = DIRECTION('', (0.0, 0.0, 1.0));", axis_id);
                self.entities.push(axis_entity);
                
                let arc_entity = format!(
                    "#{} = CIRCLE('', #{}, {:.6});",
                    arc_id, center_id, radius
                );
                self.entities.push(arc_entity);
                arc_id
            }
            CurveType::Ellipse { center, major_axis, minor_axis } => {
                // Ellipse support in STEP
                let ellipse_id = self.next_id();
                let center_id = self.next_id();
                let axis_id = self.next_id();
                let ref_axis_id = self.next_id();
                
                let center_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    center_id, center[0], center[1], center[2]
                );
                self.entities.push(center_entity);
                
                // Calculate magnitudes of axes
                let major_len = (major_axis[0].powi(2) + major_axis[1].powi(2) + major_axis[2].powi(2)).sqrt();
                let minor_len = (minor_axis[0].powi(2) + minor_axis[1].powi(2) + minor_axis[2].powi(2)).sqrt();
                
                let axis_entity = format!("#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));", 
                    axis_id, major_axis[0], major_axis[1], major_axis[2]);
                self.entities.push(axis_entity);
                
                let ref_axis_entity = format!("#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));", 
                    ref_axis_id, minor_axis[0], minor_axis[1], minor_axis[2]);
                self.entities.push(ref_axis_entity);
                
                let ellipse_entity = format!(
                    "#{} = ELLIPSE('', #{}, {:.6}, {:.6});",
                    ellipse_id, center_id, major_len, minor_len
                );
                self.entities.push(ellipse_entity);
                ellipse_id
            }
            CurveType::Bezier { control_points } => {
                // Bezier curve - write as B-spline with appropriate knots
                let bezier_id = self.next_id();
                let degree = if control_points.len() > 1 {
                    (control_points.len() - 1).min(3) as i32
                } else {
                    0
                };
                
                // For now, approximate as line
                let line_id = self.next_id();
                let dir_id = self.next_id();
                
                let dx = e.end.point[0] - e.start.point[0];
                let dy = e.end.point[1] - e.start.point[1];
                let dz = e.end.point[2] - e.start.point[2];
                
                let dir_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    dir_id, dx, dy, dz
                );
                self.entities.push(dir_entity);
                
                let line_entity = format!(
                    "#{} = LINE('', #{}, #{});",
                    line_id, start_pt_id, dir_id
                );
                self.entities.push(line_entity);
                line_id
            }
            CurveType::BSpline { .. } => {
                // For now, treat B-spline as a line approximation
                let line_id = self.next_id();
                let dir_id = self.next_id();
                
                let dx = e.end.point[0] - e.start.point[0];
                let dy = e.end.point[1] - e.start.point[1];
                let dz = e.end.point[2] - e.start.point[2];
                
                let dir_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    dir_id, dx, dy, dz
                );
                self.entities.push(dir_entity);
                
                let line_entity = format!(
                    "#{} = LINE('', #{}, #{});",
                    line_id, start_pt_id, dir_id
                );
                self.entities.push(line_entity);
                line_id
            }
        };
        
        let edge_curve_id = self.next_id();
        let edge_curve_entity = format!(
            "#{} = EDGE_CURVE('', #{}, #{}, #{}, .T.);",
            edge_curve_id, start_v_id, end_v_id, curve_id
        );
        self.entities.push(edge_curve_entity);
        
        edge_curve_id
    }
    
    fn add_wire(&mut self, w: &Wire) -> Vec<usize> {
        let mut edge_ids = Vec::new();
        for edge in &w.edges {
            let edge_id = self.add_edge(edge);
            edge_ids.push(edge_id);
        }
        edge_ids
    }
    
    fn add_face(&mut self, f: &Face) -> usize {
        let outer_edge_ids = self.add_wire(&f.outer_wire);
        
        let outer_loop_id = self.next_id();
        let edges_list = outer_edge_ids.iter()
            .map(|id| format!("#{}", id))
            .collect::<Vec<_>>()
            .join(", ");
        let outer_loop_entity = format!(
            "#{} = EDGE_LOOP('', ({}));",
            outer_loop_id, edges_list
        );
        self.entities.push(outer_loop_entity);
        
        let mut loop_ids = vec![outer_loop_id];
        
        // Handle inner wires (holes)
        for inner_wire in &f.inner_wires {
            let inner_edge_ids = self.add_wire(inner_wire);
            let inner_loop_id = self.next_id();
            let inner_edges_list = inner_edge_ids.iter()
                .map(|id| format!("#{}", id))
                .collect::<Vec<_>>()
                .join(", ");
            let inner_loop_entity = format!(
                "#{} = EDGE_LOOP('', ({}));",
                inner_loop_id, inner_edges_list
            );
            self.entities.push(inner_loop_entity);
            loop_ids.push(inner_loop_id);
        }
        
        // Create surface
        let surface_id = match &f.surface_type {
            SurfaceType::Plane { origin, normal } => {
                let plane_id = self.next_id();
                let origin_id = self.next_id();
                let normal_id = self.next_id();
                
                let origin_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    origin_id, origin[0], origin[1], origin[2]
                );
                self.entities.push(origin_entity);
                
                let normal_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    normal_id, normal[0], normal[1], normal[2]
                );
                self.entities.push(normal_entity);
                
                let plane_entity = format!("#{} = PLANE('', #{}, #{});", plane_id, origin_id, normal_id);
                self.entities.push(plane_entity);
                
                plane_id
            }
            SurfaceType::Cylinder { origin, axis, radius } => {
                let cyl_id = self.next_id();
                let origin_id = self.next_id();
                let axis_id = self.next_id();
                
                let origin_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    origin_id, origin[0], origin[1], origin[2]
                );
                self.entities.push(origin_entity);
                
                let axis_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    axis_id, axis[0], axis[1], axis[2]
                );
                self.entities.push(axis_entity);
                
                let cyl_entity = format!(
                    "#{} = CYLINDRICAL_SURFACE('', #{}, {:.6}, #{});",
                    cyl_id, origin_id, radius, axis_id
                );
                self.entities.push(cyl_entity);
                
                cyl_id
            }
            SurfaceType::Sphere { center, radius } => {
                let sph_id = self.next_id();
                let center_id = self.next_id();
                
                let center_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    center_id, center[0], center[1], center[2]
                );
                self.entities.push(center_entity);
                
                let sph_entity = format!(
                    "#{} = SPHERICAL_SURFACE('', #{}, {:.6});",
                    sph_id, center_id, radius
                );
                self.entities.push(sph_entity);
                
                sph_id
            }
            SurfaceType::Cone { origin, axis, half_angle_rad } => {
                let cone_id = self.next_id();
                let origin_id = self.next_id();
                let axis_id = self.next_id();
                
                let origin_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    origin_id, origin[0], origin[1], origin[2]
                );
                self.entities.push(origin_entity);
                
                let axis_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    axis_id, axis[0], axis[1], axis[2]
                );
                self.entities.push(axis_entity);
                
                let cone_entity = format!(
                    "#{} = CONICAL_SURFACE('', #{}, {:.6}, #{});",
                    cone_id, origin_id, half_angle_rad, axis_id
                );
                self.entities.push(cone_entity);
                
                cone_id
            }
            SurfaceType::Torus { center, major_radius, minor_radius } => {
                let torus_id = self.next_id();
                let center_id = self.next_id();
                
                let center_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    center_id, center[0], center[1], center[2]
                );
                self.entities.push(center_entity);
                
                let torus_entity = format!(
                    "#{} = TOROIDAL_SURFACE('', #{}, {:.6}, {:.6});",
                    torus_id, center_id, major_radius, minor_radius
                );
                self.entities.push(torus_entity);
                
                torus_id
            }
            SurfaceType::BSpline { .. } => {
                // Fallback to plane for B-spline surfaces
                let plane_id = self.next_id();
                let origin_id = self.next_id();
                let normal_id = self.next_id();
                
                let origin_entity = format!(
                    "#{} = CARTESIAN_POINT('', (0.0, 0.0, 0.0));",
                    origin_id
                );
                self.entities.push(origin_entity);
                
                let normal_entity = format!(
                    "#{} = DIRECTION('', (0.0, 0.0, 1.0));",
                    normal_id
                );
                self.entities.push(normal_entity);
                
                let plane_entity = format!("#{} = PLANE('', #{}, #{});", plane_id, origin_id, normal_id);
                self.entities.push(plane_entity);
                
                plane_id
            }
        };
        
        // Create face
        let face_id = self.next_id();
        let loops_list = loop_ids.iter()
            .map(|id| format!("#{}", id))
            .collect::<Vec<_>>()
            .join(", ");
        let face_entity = format!(
            "#{} = FACE('', ({}), #{}, .T.);",
            face_id, loops_list, surface_id
        );
        self.entities.push(face_entity);
        
        face_id
    }
    
    fn add_shell(&mut self, s: &Shell) -> usize {
        let mut face_ids = Vec::new();
        for face in &s.faces {
            let face_id = self.add_face(face);
            face_ids.push(face_id);
        }
        
        let shell_id = self.next_id();
        let faces_list = face_ids.iter()
            .map(|id| format!("#{}", id))
            .collect::<Vec<_>>()
            .join(", ");
        let shell_entity = format!(
            "#{} = CLOSED_SHELL('', ({}));",
            shell_id, faces_list
        );
        self.entities.push(shell_entity);
        
        shell_id
    }
    
    fn add_solid(&mut self, s: &Solid) -> usize {
        let outer_shell_id = self.add_shell(&s.outer_shell);
        
        let mut shell_ids = vec![outer_shell_id];
        for inner_shell in &s.inner_shells {
            let inner_shell_id = self.add_shell(inner_shell);
            shell_ids.push(inner_shell_id);
        }
        
        let solid_id = self.next_id();
        let solid_entity = format!(
            "#{} = MANIFOLD_SOLID_BREP('', #{});",
            solid_id, shell_ids[0]
        );
        self.entities.push(solid_entity);
        
        solid_id
    }
    
    fn add_compound(&mut self, c: &Compound) {
        for solid in &c.solids {
            self.add_solid(solid);
        }
    }
    
    fn write_file(&mut self) -> Result<()> {
        writeln!(self.writer, "ISO-10303-21;")?;
        writeln!(self.writer, "HEADER;")?;
        writeln!(self.writer, "FILE_DESCRIPTION(('cascade-rs STEP export'), '2', '2', '', '', 1.0, '');")?;
        writeln!(self.writer, "FILE_NAME('cascade-rs export', '', (''), (''), 'cascade-rs', '', '');")?;
        writeln!(self.writer, "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")?;
        writeln!(self.writer, "ENDSEC;")?;
        writeln!(self.writer, "DATA;")?;
        
        for entity in &self.entities {
            writeln!(self.writer, "{}", entity)?;
        }
        
        writeln!(self.writer, "ENDSEC;")?;
        writeln!(self.writer, "END-ISO-10303-21;")?;
        
        Ok(())
    }
}
