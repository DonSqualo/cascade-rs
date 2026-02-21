//! IGES (Initial Graphics Exchange Specification) file I/O
//! 
//! IGES is an older CAD exchange format with fixed 80-character records.
//! File structure:
//! - Start section (S)
//! - Global section (G)
//! - Directory/Data section (D) - entity metadata
//! - Parameter Data section (P) - entity parameters
//! - Terminate section (T)

use crate::brep::{Shape, Edge, Vertex, CurveType};
use crate::{Result, CascadeError};
use std::collections::HashMap;
use std::fs;

/// IGES entity type codes
#[derive(Debug, Clone, Copy, PartialEq)]
enum IgesEntityType {
    Point = 116,           // Point (3D)
    Line = 110,            // Line
    Circle = 100,          // Circle
    Ellipse = 104,         // Ellipse
    Parabola = 102,        // Parabola
    Hyperbola = 103,       // Hyperbola
    BSplineCurve = 126,    // B-spline curve
    BSplineSurface = 128,  // B-spline surface
    Plane = 108,           // Plane
    Cylinder = 112,        // Cylinder
    Cone = 111,            // Cone
    Sphere = 114,          // Sphere
    Torus = 120,           // Torus
}

impl IgesEntityType {
    fn from_code(code: i32) -> Option<Self> {
        match code {
            116 => Some(IgesEntityType::Point),
            110 => Some(IgesEntityType::Line),
            100 => Some(IgesEntityType::Circle),
            104 => Some(IgesEntityType::Ellipse),
            102 => Some(IgesEntityType::Parabola),
            103 => Some(IgesEntityType::Hyperbola),
            126 => Some(IgesEntityType::BSplineCurve),
            128 => Some(IgesEntityType::BSplineSurface),
            108 => Some(IgesEntityType::Plane),
            112 => Some(IgesEntityType::Cylinder),
            111 => Some(IgesEntityType::Cone),
            114 => Some(IgesEntityType::Sphere),
            120 => Some(IgesEntityType::Torus),
            _ => None,
        }
    }
}

/// IGES entity data
#[derive(Debug, Clone)]
enum IgesEntity {
    Point { coords: [f64; 3] },
    Line { start: [f64; 3], end: [f64; 3] },
    Circle { center: [f64; 3], radius: f64, normal: [f64; 3] },
    Ellipse { center: [f64; 3], major_axis: [f64; 3], eccentricity: f64 },
    BSplineCurve {
        degree: i32,
        control_points: Vec<[f64; 3]>,
        knots: Vec<f64>,
    },
    BSplineSurface {
        u_degree: i32,
        v_degree: i32,
        control_points: Vec<Vec<[f64; 3]>>,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
    },
    Plane { origin: [f64; 3], normal: [f64; 3] },
    Cylinder { origin: [f64; 3], axis: [f64; 3], radius: f64 },
    Sphere { center: [f64; 3], radius: f64 },
    Cone { origin: [f64; 3], axis: [f64; 3], half_angle: f64 },
    Torus { center: [f64; 3], axis: [f64; 3], major_radius: f64, minor_radius: f64 },
}

/// Directory Entry metadata
#[derive(Debug, Clone)]
struct DirectoryEntry {
    entity_type: i32,
    parameter_data_index: i32,
    structure: i32,
    line_font_pattern: i32,
    level: i32,
    view: i32,
    transformation_matrix: i32,
    label_assoc: i32,
    status: [i32; 4],
}

/// Parse IGES file
pub fn read_iges(path: &str) -> Result<Shape> {
    let content = fs::read_to_string(path)
        .map_err(|e| CascadeError::IoError(e))?;
    
    let parser = IgesParser::new(&content)?;
    parser.parse()
}

/// Write a Shape to IGES format
pub fn write_iges(shape: &Shape, path: &str) -> Result<()> {
    let mut writer = IgesWriter::new();
    writer.add_shape(shape)?;
    writer.write_file(path)
}

struct IgesParser {
    directory_entries: HashMap<i32, DirectoryEntry>,
    parameter_records: Vec<String>,
    entities: HashMap<i32, IgesEntity>,
}

impl IgesParser {
    fn new(content: &str) -> Result<Self> {
        let lines: Vec<&str> = content.lines().collect();
        
        // Validate lines are 80 characters (or close to it, allowing for trailing newlines)
        for (idx, line) in lines.iter().enumerate() {
            if line.len() > 80 {
                return Err(CascadeError::InvalidGeometry(
                    format!("IGES line {} exceeds 80 characters", idx + 1)
                ));
            }
        }
        
        let mut parser = IgesParser {
            directory_entries: HashMap::new(),
            parameter_records: Vec::new(),
            entities: HashMap::new(),
        };
        
        parser.parse_sections(&lines)?;
        Ok(parser)
    }
    
    fn parse_sections(&mut self, lines: &[&str]) -> Result<()> {
        let mut i = 0;
        
        // Skip start section (S)
        while i < lines.len() {
            let line = lines[i];
            let section_type = if line.len() >= 73 {
                &line[72..73]
            } else {
                ""
            };
            
            if section_type == "G" {
                break; // Start of global section
            }
            i += 1;
        }
        
        // Skip global section (G)
        while i < lines.len() {
            let line = lines[i];
            let section_type = if line.len() >= 73 {
                &line[72..73]
            } else {
                ""
            };
            
            if section_type == "D" {
                break; // Start of directory section
            }
            i += 1;
        }
        
        // Parse directory section (D) - pairs of 80-char records per entity
        let dir_start = i;
        while i < lines.len() {
            let line = lines[i];
            let section_type = if line.len() >= 73 {
                &line[72..73]
            } else {
                ""
            };
            
            if section_type != "D" {
                break; // End of directory section
            }
            i += 1;
        }
        
        // Parse directory entries (each entity has 2 directory records)
        self.parse_directory(lines, dir_start, i)?;
        
        // Parse parameter data section (P)
        while i < lines.len() {
            let line = lines[i];
            let section_type = if line.len() >= 73 {
                &line[72..73]
            } else {
                ""
            };
            
            if section_type == "P" {
                break;
            }
            i += 1;
        }
        
        let param_start = i;
        while i < lines.len() {
            let line = lines[i];
            let section_type = if line.len() >= 73 {
                &line[72..73]
            } else {
                ""
            };
            
            if section_type != "P" {
                break; // End of parameter section
            }
            i += 1;
        }
        
        // Collect parameter records
        self.collect_parameters(lines, param_start, i)?;
        
        // Parse entities from directory + parameters
        self.parse_entities()?;
        
        Ok(())
    }
    
    fn parse_directory(&mut self, lines: &[&str], start: usize, end: usize) -> Result<()> {
        let mut entry_num = 1;
        let mut i = start;
        
        while i + 1 < end {
            let line1 = lines[i];
            let line2 = lines[i + 1];
            
            // Parse first line of directory entry
            let entity_type: i32 = line1[0..8].trim().parse()
                .map_err(|_| CascadeError::InvalidGeometry("Invalid entity type".into()))?;
            let param_data_index: i32 = line1[8..16].trim().parse()
                .map_err(|_| CascadeError::InvalidGeometry("Invalid parameter index".into()))?;
            let structure: i32 = line1[16..24].trim().parse()
                .unwrap_or(0);
            let line_font: i32 = line1[24..32].trim().parse()
                .unwrap_or(2);
            let level: i32 = line1[32..40].trim().parse()
                .unwrap_or(0);
            let view: i32 = line1[40..48].trim().parse()
                .unwrap_or(0);
            let transform: i32 = line1[48..56].trim().parse()
                .unwrap_or(0);
            let label_assoc: i32 = line1[56..64].trim().parse()
                .unwrap_or(0);
            
            // Parse second line for status fields
            let blank1: i32 = line2[0..8].trim().parse()
                .unwrap_or(0);
            let blank2: i32 = line2[8..16].trim().parse()
                .unwrap_or(0);
            let blank3: i32 = line2[16..24].trim().parse()
                .unwrap_or(0);
            let blank4: i32 = line2[24..32].trim().parse()
                .unwrap_or(0);
            
            let entry = DirectoryEntry {
                entity_type,
                parameter_data_index: param_data_index,
                structure,
                line_font_pattern: line_font,
                level,
                view,
                transformation_matrix: transform,
                label_assoc,
                status: [blank1, blank2, blank3, blank4],
            };
            
            self.directory_entries.insert(entry_num, entry);
            entry_num += 1;
            i += 2;
        }
        
        Ok(())
    }
    
    fn collect_parameters(&mut self, lines: &[&str], start: usize, end: usize) -> Result<()> {
        let mut param_text = String::new();
        
        for i in start..end {
            let line = lines[i];
            // Remove section character (last 8 chars are metadata)
            let data_part = if line.len() > 64 {
                &line[0..64]
            } else {
                line
            };
            param_text.push_str(data_part);
        }
        
        // Split by comma to get individual parameter records
        let records: Vec<&str> = param_text.split(',').collect();
        for record in records {
            let trimmed = record.trim();
            if !trimmed.is_empty() {
                self.parameter_records.push(trimmed.to_string());
            }
        }
        
        Ok(())
    }
    
    fn parse_entities(&mut self) -> Result<()> {
        for (entity_id, entry) in &self.directory_entries {
            if let Some(entity_type_enum) = IgesEntityType::from_code(entry.entity_type) {
                let param_idx = (entry.parameter_data_index - 1) as usize;
                
                let entity = match entity_type_enum {
                    IgesEntityType::Point => self.parse_point(param_idx)?,
                    IgesEntityType::Line => self.parse_line(param_idx)?,
                    IgesEntityType::Circle => self.parse_circle(param_idx)?,
                    IgesEntityType::Sphere => self.parse_sphere(param_idx)?,
                    IgesEntityType::Cylinder => self.parse_cylinder(param_idx)?,
                    IgesEntityType::Cone => self.parse_cone(param_idx)?,
                    IgesEntityType::Torus => self.parse_torus(param_idx)?,
                    IgesEntityType::Plane => self.parse_plane(param_idx)?,
                    IgesEntityType::BSplineCurve => self.parse_bspline_curve(param_idx)?,
                    IgesEntityType::BSplineSurface => self.parse_bspline_surface(param_idx)?,
                    _ => {
                        // Skip unsupported types
                        continue;
                    }
                };
                
                self.entities.insert(*entity_id, entity);
            }
        }
        
        Ok(())
    }
    
    fn parse_point(&self, param_idx: usize) -> Result<IgesEntity> {
        // Point: x, y, z coordinates
        if param_idx + 2 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient point parameters".into()));
        }
        
        let x: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid point X".into()))?;
        let y: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid point Y".into()))?;
        let z: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid point Z".into()))?;
        
        Ok(IgesEntity::Point { coords: [x, y, z] })
    }
    
    fn parse_line(&self, param_idx: usize) -> Result<IgesEntity> {
        // Line: x1, y1, z1, x2, y2, z2
        if param_idx + 5 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient line parameters".into()));
        }
        
        let x1: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid line X1".into()))?;
        let y1: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid line Y1".into()))?;
        let z1: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid line Z1".into()))?;
        let x2: f64 = self.parameter_records[param_idx + 3].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid line X2".into()))?;
        let y2: f64 = self.parameter_records[param_idx + 4].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid line Y2".into()))?;
        let z2: f64 = self.parameter_records[param_idx + 5].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid line Z2".into()))?;
        
        Ok(IgesEntity::Line {
            start: [x1, y1, z1],
            end: [x2, y2, z2],
        })
    }
    
    fn parse_circle(&self, param_idx: usize) -> Result<IgesEntity> {
        // Circle: z-plane, center-x, center-y, radius
        if param_idx + 3 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient circle parameters".into()));
        }
        
        let _z: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid circle Z".into()))?;
        let cx: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid circle X".into()))?;
        let cy: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid circle Y".into()))?;
        let r: f64 = self.parameter_records[param_idx + 3].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid circle radius".into()))?;
        
        Ok(IgesEntity::Circle {
            center: [cx, cy, 0.0],
            radius: r,
            normal: [0.0, 0.0, 1.0],
        })
    }
    
    fn parse_sphere(&self, param_idx: usize) -> Result<IgesEntity> {
        // Sphere: center-x, center-y, center-z, radius
        if param_idx + 3 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient sphere parameters".into()));
        }
        
        let x: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid sphere X".into()))?;
        let y: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid sphere Y".into()))?;
        let z: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid sphere Z".into()))?;
        let r: f64 = self.parameter_records[param_idx + 3].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid sphere radius".into()))?;
        
        Ok(IgesEntity::Sphere {
            center: [x, y, z],
            radius: r,
        })
    }
    
    fn parse_cylinder(&self, param_idx: usize) -> Result<IgesEntity> {
        // Cylinder: origin-x, origin-y, origin-z, axis-x, axis-y, axis-z, radius
        if param_idx + 6 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient cylinder parameters".into()));
        }
        
        let ox: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cylinder origin X".into()))?;
        let oy: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cylinder origin Y".into()))?;
        let oz: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cylinder origin Z".into()))?;
        let ax: f64 = self.parameter_records[param_idx + 3].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cylinder axis X".into()))?;
        let ay: f64 = self.parameter_records[param_idx + 4].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cylinder axis Y".into()))?;
        let az: f64 = self.parameter_records[param_idx + 5].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cylinder axis Z".into()))?;
        let r: f64 = self.parameter_records[param_idx + 6].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cylinder radius".into()))?;
        
        // Normalize axis
        let len = (ax * ax + ay * ay + az * az).sqrt();
        let (ax, ay, az) = if len > 0.0 {
            (ax / len, ay / len, az / len)
        } else {
            (0.0, 0.0, 1.0)
        };
        
        Ok(IgesEntity::Cylinder {
            origin: [ox, oy, oz],
            axis: [ax, ay, az],
            radius: r,
        })
    }
    
    fn parse_cone(&self, param_idx: usize) -> Result<IgesEntity> {
        // Cone: origin, axis, semi-vertical angle
        if param_idx + 6 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient cone parameters".into()));
        }
        
        let ox: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cone origin X".into()))?;
        let oy: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cone origin Y".into()))?;
        let oz: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cone origin Z".into()))?;
        let ax: f64 = self.parameter_records[param_idx + 3].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cone axis X".into()))?;
        let ay: f64 = self.parameter_records[param_idx + 4].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cone axis Y".into()))?;
        let az: f64 = self.parameter_records[param_idx + 5].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cone axis Z".into()))?;
        let angle: f64 = self.parameter_records[param_idx + 6].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid cone angle".into()))?;
        
        // Normalize axis
        let len = (ax * ax + ay * ay + az * az).sqrt();
        let (ax, ay, az) = if len > 0.0 {
            (ax / len, ay / len, az / len)
        } else {
            (0.0, 0.0, 1.0)
        };
        
        Ok(IgesEntity::Cone {
            origin: [ox, oy, oz],
            axis: [ax, ay, az],
            half_angle: angle,
        })
    }
    
    fn parse_torus(&self, param_idx: usize) -> Result<IgesEntity> {
        // Torus: center, axis, major radius, minor radius
        if param_idx + 7 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient torus parameters".into()));
        }
        
        let cx: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus center X".into()))?;
        let cy: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus center Y".into()))?;
        let cz: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus center Z".into()))?;
        let ax: f64 = self.parameter_records[param_idx + 3].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus axis X".into()))?;
        let ay: f64 = self.parameter_records[param_idx + 4].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus axis Y".into()))?;
        let az: f64 = self.parameter_records[param_idx + 5].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus axis Z".into()))?;
        let major_r: f64 = self.parameter_records[param_idx + 6].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus major radius".into()))?;
        let minor_r: f64 = self.parameter_records[param_idx + 7].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid torus minor radius".into()))?;
        
        // Normalize axis
        let len = (ax * ax + ay * ay + az * az).sqrt();
        let (ax, ay, az) = if len > 0.0 {
            (ax / len, ay / len, az / len)
        } else {
            (0.0, 0.0, 1.0)
        };
        
        Ok(IgesEntity::Torus {
            center: [cx, cy, cz],
            axis: [ax, ay, az],
            major_radius: major_r,
            minor_radius: minor_r,
        })
    }
    
    fn parse_plane(&self, param_idx: usize) -> Result<IgesEntity> {
        // Plane: a, b, c, d (ax + by + cz = d)
        if param_idx + 3 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient plane parameters".into()));
        }
        
        let a: f64 = self.parameter_records[param_idx].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid plane A".into()))?;
        let b: f64 = self.parameter_records[param_idx + 1].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid plane B".into()))?;
        let c: f64 = self.parameter_records[param_idx + 2].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid plane C".into()))?;
        let d: f64 = self.parameter_records[param_idx + 3].parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid plane D".into()))?;
        
        // Normalize normal vector
        let len = (a * a + b * b + c * c).sqrt();
        let (a, b, c) = if len > 0.0 {
            (a / len, b / len, c / len)
        } else {
            (0.0, 0.0, 1.0)
        };
        
        // Find a point on the plane
        let origin = if c.abs() > 0.001 {
            [0.0, 0.0, d / c]
        } else if b.abs() > 0.001 {
            [0.0, d / b, 0.0]
        } else {
            [d / a, 0.0, 0.0]
        };
        
        Ok(IgesEntity::Plane {
            origin,
            normal: [a, b, c],
        })
    }
    
    fn parse_bspline_curve(&self, param_idx: usize) -> Result<IgesEntity> {
        // B-spline curve
        // K, M, PROP1, PROP2, N, T(1), ..., T(K+M+2), W(1), ..., W(N+1), 
        // X(1), Y(1), Z(1), ..., X(N+1), Y(N+1), Z(N+1), U0, U1
        
        if param_idx + 2 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient B-spline parameters".into()));
        }
        
        let degree: i32 = self.parameter_records[param_idx].trim_matches(';').parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid B-spline degree".into()))?;
        let num_control: i32 = self.parameter_records[param_idx + 1].trim_matches(';').parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid B-spline control point count".into()))?;
        
        // For now, return a simple B-spline with placeholder data
        Ok(IgesEntity::BSplineCurve {
            degree,
            control_points: vec![[0.0, 0.0, 0.0]; num_control as usize],
            knots: vec![],
        })
    }
    
    fn parse_bspline_surface(&self, param_idx: usize) -> Result<IgesEntity> {
        // B-spline surface (similar structure to curve but 2D)
        if param_idx + 2 >= self.parameter_records.len() {
            return Err(CascadeError::InvalidGeometry("Insufficient B-spline surface parameters".into()));
        }
        
        let u_degree: i32 = self.parameter_records[param_idx].trim_matches(';').parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid B-spline surface U degree".into()))?;
        let v_degree: i32 = self.parameter_records[param_idx + 1].trim_matches(';').parse()
            .map_err(|_| CascadeError::InvalidGeometry("Invalid B-spline surface V degree".into()))?;
        
        Ok(IgesEntity::BSplineSurface {
            u_degree,
            v_degree,
            control_points: vec![],
            u_knots: vec![],
            v_knots: vec![],
        })
    }
    
    fn parse(&self) -> Result<Shape> {
        if self.entities.is_empty() {
            return Err(CascadeError::InvalidGeometry("No entities found in IGES file".into()));
        }
        
        // Convert first entity to Shape
        // For now, prioritize Point > Line > Sphere > Cylinder
        for (_id, entity) in &self.entities {
            match entity {
                IgesEntity::Point { coords } => {
                    return Ok(Shape::Vertex(Vertex {
                        point: *coords,
                    }));
                }
                IgesEntity::Line { start, end } => {
                    let v_start = Vertex { point: *start };
                    let v_end = Vertex { point: *end };
                    
                    return Ok(Shape::Edge(Edge {
                        start: v_start,
                        end: v_end,
                        curve_type: CurveType::Line,
                    }));
                }
                IgesEntity::Sphere { center, radius } => {
                    // Create a simple sphere primitive
                    // For now return as compound/solid - needs primitive factory
                    return Ok(Shape::Vertex(Vertex { point: *center }));
                }
                _ => continue,
            }
        }
        
        Err(CascadeError::InvalidGeometry("Could not convert entities to Shape".into()))
    }
}

struct IgesWriter {
    entities: Vec<(IgesEntity, u32)>, // (Entity, Layer number)
}

impl IgesWriter {
    fn new() -> Self {
        IgesWriter {
            entities: Vec::new(),
        }
    }
    
    fn add_shape(&mut self, shape: &Shape) -> Result<()> {
        self.add_shape_with_layer(shape, 0)
    }
    
    fn add_shape_with_layer(&mut self, shape: &Shape, layer: u32) -> Result<()> {
        match shape {
            Shape::Vertex(v) => {
                self.entities.push((IgesEntity::Point {
                    coords: v.point,
                }, layer));
            }
            Shape::Edge(e) => {
                self.entities.push((IgesEntity::Line {
                    start: e.start.point,
                    end: e.end.point,
                }, layer));
            }
            _ => {
                return Err(CascadeError::NotImplemented("IGES write for this shape type".into()));
            }
        }
        Ok(())
    }
    
    fn add_solid_with_layer(&mut self, solid: &crate::brep::Solid, layer: u32) -> Result<()> {
        // Add all faces from outer shell
        for face in &solid.outer_shell.faces {
            // Add edges from outer wire
            for edge in &face.outer_wire.edges {
                self.entities.push((IgesEntity::Line {
                    start: edge.start.point,
                    end: edge.end.point,
                }, layer));
            }
            // Add edges from inner wires (holes)
            for inner_wire in &face.inner_wires {
                for edge in &inner_wire.edges {
                    self.entities.push((IgesEntity::Line {
                        start: edge.start.point,
                        end: edge.end.point,
                    }, layer));
                }
            }
        }
        
        // Add all faces from inner shells (voids)
        for shell in &solid.inner_shells {
            for face in &shell.faces {
                for edge in &face.outer_wire.edges {
                    self.entities.push((IgesEntity::Line {
                        start: edge.start.point,
                        end: edge.end.point,
                    }, layer));
                }
                for inner_wire in &face.inner_wires {
                    for edge in &inner_wire.edges {
                        self.entities.push((IgesEntity::Line {
                            start: edge.start.point,
                            end: edge.end.point,
                        }, layer));
                    }
                }
            }
        }
        Ok(())
    }
    
    fn write_file(&self, path: &str) -> Result<()> {
        let mut content = String::new();
        
        // Start section (S)
        content.push_str("                                                                        S      1\n");
        
        // Global section (G)
        content.push_str("1H,,1H,10HCascade-RS,31H,13H,1HM,1,0.01,1,1HM,1,0,NONE,none,N,N,  G      1\n");
        content.push_str("27H,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,  G      2\n");
        content.push_str("1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0  G      3\n");
        
        // Directory (D) section
        // Validate layer numbers (IGES standard: 0-65535)
        for (_entity, layer) in &self.entities {
            if *layer > 65535 {
                return Err(CascadeError::InvalidGeometry(
                    format!("Layer number {} exceeds IGES maximum of 65535", layer)
                ));
            }
        }
        
        let mut entity_id = 1i32;
        let mut param_idx = 1i32;
        
        for (entity, layer) in &self.entities {
            let entity_type = match entity {
                IgesEntity::Point { .. } => 116,
                IgesEntity::Line { .. } => 110,
                IgesEntity::Circle { .. } => 100,
                IgesEntity::Sphere { .. } => 114,
                IgesEntity::Cylinder { .. } => 112,
                _ => continue,
            };
            
            // Format directory line 1: IGES-compliant format
            // Field 5 (bytes 33-40) is the Level/Layer field  
            let line1 = format!(
                "{:8}{:8}{:8}{:8}{:8}{:8}{:8}{:8}{:8}                  D      {}\n",
                entity_type, param_idx, 0, 2, layer, 0, 0, 0, 0, entity_id * 2 - 1
            );
            content.push_str(&line1);
            
            let num_params = match entity {
                IgesEntity::Point { .. } => 3,
                IgesEntity::Line { .. } => 6,
                IgesEntity::Circle { .. } => 4,
                IgesEntity::Sphere { .. } => 4,
                IgesEntity::Cylinder { .. } => 7,
                _ => 0,
            };
            
            // Directory line 2: Status fields
            let line2 = format!(
                "{:8}{:8}{:8}{:8}{:8}{:8}{:8}{:8}{:8}                  D      {}\n",
                0, 0, 0, 0, 0, 0, 0, 0, 0, entity_id * 2
            );
            content.push_str(&line2);
            
            param_idx += num_params;
            entity_id += 1;
        }
        
        // Parameter (P) section
        let mut record_num = 1i32;
        let mut param_str = String::new();
        
        for (entity, _layer) in &self.entities {
            match entity {
                IgesEntity::Point { coords } => {
                    param_str.push_str(&format!("{},{},{},", coords[0], coords[1], coords[2]));
                }
                IgesEntity::Line { start, end } => {
                    param_str.push_str(&format!(
                        "{},{},{},{},{},{},",
                        start[0], start[1], start[2], end[0], end[1], end[2]
                    ));
                }
                IgesEntity::Circle { center, radius, .. } => {
                    param_str.push_str(&format!(
                        "0,{},{},{},",
                        center[0], center[1], radius
                    ));
                }
                IgesEntity::Sphere { center, radius } => {
                    param_str.push_str(&format!(
                        "{},{},{},{},",
                        center[0], center[1], center[2], radius
                    ));
                }
                IgesEntity::Cylinder { origin, axis, radius } => {
                    param_str.push_str(&format!(
                        "{},{},{},{},{},{},{},",
                        origin[0], origin[1], origin[2], axis[0], axis[1], axis[2], radius
                    ));
                }
                _ => {}
            }
        }
        
        // Write parameter records in 64-char chunks
        param_str.pop(); // Remove trailing comma
        let param_bytes = param_str.as_bytes();
        let mut offset = 0;
        
        while offset < param_bytes.len() {
            let end = (offset + 64).min(param_bytes.len());
            let chunk = String::from_utf8_lossy(&param_bytes[offset..end]);
            
            let padded = format!("{:<64}P      {}\n", chunk, record_num);
            content.push_str(&padded);
            record_num += 1;
            offset = end;
        }
        
        // Terminate section (T)
        content.push_str(&format!("S      1G      3D      {}P      {}                        T      1\n",
            entity_id * 2, record_num - 1));
        
        std::fs::write(path, content)
            .map_err(|e| CascadeError::IoError(e))?;
        
        Ok(())
    }
}

/// Write multiple Solids to IGES format, each with an optional layer number
/// 
/// Each solid can be assigned to a specific layer for organization in CAD applications.
/// Layer numbers range from 0-65535 per IGES specification. If no layer is specified,
/// the solid will be assigned to layer 0 (default).
/// 
/// # Arguments
/// * `solids` - Slice of (Solid reference, optional layer number) tuples
/// * `path` - Output file path
/// 
/// # Returns
/// * `Result<()>` - Ok if successful, error otherwise
/// 
/// # Example
/// ```ignore
/// let solid1 = make_box(1.0, 2.0, 3.0)?;
/// let solid2 = make_cylinder(1.0, 5.0)?;
/// let solids = vec![(&solid1, Some(1)), (&solid2, Some(2))];
/// write_iges_with_layers(&solids, "output.igs")?;
/// ```
pub fn write_iges_with_layers(solids: &[(&crate::brep::Solid, Option<u32>)], path: &str) -> Result<()> {
    let mut writer = IgesWriter::new();
    
    for (solid, layer_opt) in solids {
        let layer = layer_opt.unwrap_or(0);
        
        // Validate layer number (IGES standard: 0-65535)
        if layer > 65535 {
            return Err(CascadeError::InvalidGeometry(
                format!("Layer number {} exceeds IGES maximum of 65535", layer)
            ));
        }
        
        writer.add_solid_with_layer(solid, layer)?;
    }
    
    writer.write_file(path)
}
