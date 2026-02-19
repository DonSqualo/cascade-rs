//! STEP (ISO 10303-21) file parser
//! 
//! Parses STEP format files and converts them to cascade-rs BREP geometry.

use crate::brep::{Shape, Solid, Shell, Face, Wire, Edge, Vertex, CurveType, SurfaceType};
use crate::{Result, CascadeError};
use std::collections::HashMap;
use std::fs;

/// STEP entity type discriminator
#[derive(Debug, Clone)]
enum StepEntity {
    CartesianPoint { coords: [f64; 3] },
    Direction { ratios: [f64; 3] },
    Axis2Placement3D { location: [f64; 3], axis: [f64; 3], ref_direction: [f64; 3] },
    VertexPoint { point_id: usize },
    Line { start: [f64; 3], direction: [f64; 3] },
    Circle { center: [f64; 3], radius: f64, axis: [f64; 3] },
    EdgeCurve { start_id: usize, end_id: usize, curve_id: usize },
    OrientedEdge { edge_id: usize },
    EdgeLoop { edge_ids: Vec<usize> },
    FaceBound { loop_id: usize },
    FaceOuterBound { loop_id: usize },
    AdvancedFace { bounds: Vec<usize>, surface_id: usize },
    Plane { location: [f64; 3], normal: [f64; 3] },
    Cylinder { location: [f64; 3], axis: [f64; 3], radius: f64 },
    Sphere { center: [f64; 3], radius: f64 },
    ClosedShell { face_ids: Vec<usize> },
    ManifoldSolidBrep { shell_id: usize },
}

/// Parse a STEP file and return a Shape
pub fn read_step(path: &str) -> Result<Shape> {
    let contents = fs::read_to_string(path)
        .map_err(|e| CascadeError::IoError(e))?;
    
    let parser = StepParser::new(&contents)?;
    parser.parse()
}

struct StepParser {
    entities: HashMap<usize, StepEntity>,
    points: HashMap<usize, [f64; 3]>,
    raw_entities: HashMap<usize, String>,  // Store raw entity strings for deferred parsing
}

impl StepParser {
    fn new(content: &str) -> Result<Self> {
        let mut parser = StepParser {
            entities: HashMap::new(),
            points: HashMap::new(),
            raw_entities: HashMap::new(),
        };
        
        parser.parse_content(content)?;
        parser.finalize()?;
        Ok(parser)
    }
    
    fn finalize(&mut self) -> Result<()> {
        // Second pass: parse all other entities now that points are available
        let raw_copy: Vec<(usize, String)> = self.raw_entities.iter().map(|(k, v)| (*k, v.clone())).collect();
        
        for (id, raw) in raw_copy.iter() {
            let entity_type_name = raw.split('(').next().unwrap_or("").trim();
            // Skip entities that were already parsed
            if entity_type_name == "CARTESIAN_POINT" || entity_type_name == "DIRECTION" {
                continue;
            }
            
            if let Ok(entity) = self.parse_entity(&raw, *id) {
                self.entities.insert(*id, entity);
            }
        }
        Ok(())
    }
    
    fn parse_content(&mut self, content: &str) -> Result<()> {
        // Find the data section between "DATA;" and "ENDSEC;"
        let data_start = content.find("DATA;")
            .ok_or_else(|| CascadeError::InvalidGeometry("No DATA section found".into()))?;
        
        // Find ENDSEC that comes after DATA (skip header's ENDSEC)
        let search_after = data_start + 5;
        let data_end = content[search_after..].find("ENDSEC;")
            .ok_or_else(|| CascadeError::InvalidGeometry("No ENDSEC found".into()))?
            + search_after;
        
        let data = &content[data_start + 5..data_end];
        
        // Handle multi-line entities by accumulating lines until we find a complete statement ending with ;
        let mut current_entity = String::new();
        
        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            current_entity.push(' ');
            current_entity.push_str(line);
            
            // Check if this line ends with semicolon (complete entity)
            if line.ends_with(';') {
                let entity_line = current_entity.trim();
                
                // Remove trailing semicolon
                let entity_line = if entity_line.ends_with(';') {
                    &entity_line[..entity_line.len() - 1]
                } else {
                    entity_line
                };
                
                self.parse_entity_line(entity_line)?;
                current_entity.clear();
            }
        }
        
        Ok(())
    }
    
    fn parse_entity_line(&mut self, line: &str) -> Result<()> {
        // Format: #id = ENTITY_TYPE(...);
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Ok(()); // Skip malformed lines
        }
        
        let id_str = parts[0].trim();
        let entity_str = parts[1].trim();
        
        // Extract ID
        let id: usize = id_str.strip_prefix('#')
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| CascadeError::InvalidGeometry(format!("Invalid entity ID: {}", id_str)))?;
        
        // Store raw entity for deferred parsing
        self.raw_entities.insert(id, entity_str.to_string());
        
        // First pass: only parse CARTESIAN_POINT, DIRECTION immediately
        // Other entities will be parsed in finalize() after points are available
        let entity_type_name = entity_str.split('(').next().unwrap_or("").trim();
        if entity_type_name == "CARTESIAN_POINT" || entity_type_name == "DIRECTION" {
            if let Ok(entity) = self.parse_entity(entity_str, id) {
                self.entities.insert(id, entity);
            }
        }
        
        Ok(())
    }
    
    fn skip_name_and_parse_args(&self, args_str: &str) -> String {
        // Skip the name field which is typically '' or a string like 'vertex0'
        // The pattern is: '<name>', <actual_args>
        let trimmed = args_str.trim();
        
        // Find the first quote
        if let Some(first_quote) = trimmed.find('\'') {
            // Find the closing quote
            if let Some(closing_quote) = trimmed[first_quote + 1..].find('\'') {
                let after_name = first_quote + 1 + closing_quote + 1;
                // Skip the comma after the name if it exists
                let rest = trimmed[after_name..].trim_start();
                if rest.starts_with(',') {
                    return rest[1..].trim_start().to_string();
                }
                return rest.to_string();
            }
        }
        
        trimmed.to_string()
    }
    
    fn parse_entity(&mut self, entity_str: &str, id: usize) -> Result<StepEntity> {
        // Extract entity type and arguments
        if let Some(paren_idx) = entity_str.find('(') {
            let entity_type = &entity_str[..paren_idx].trim();
            let args_end = entity_str.rfind(')')
                .ok_or_else(|| CascadeError::InvalidGeometry("Missing closing paren".into()))?;
            let args_str = &entity_str[paren_idx + 1..args_end];
            
            // Skip the name field
            let args_to_parse = self.skip_name_and_parse_args(args_str);
            
            let entity = match *entity_type {
                "CARTESIAN_POINT" => {
                    let coords = self.parse_coords(&args_to_parse)?;
                    self.points.insert(id, coords);
                    StepEntity::CartesianPoint { coords }
                },
                "FACE_SURFACE" => {
                    // FACE_SURFACE has same structure as ADVANCED_FACE
                    let parts = self.parse_list(&args_to_parse)?;
                    let bounds = if parts.len() > 0 {
                        let bound_list = self.parse_list(&parts[0])?;
                        bound_list.iter()
                            .map(|p| self.resolve_id(p))
                            .collect::<Result<Vec<_>>>()?
                    } else {
                        vec![]
                    };
                    let surface_id = if parts.len() > 1 {
                        self.resolve_id(&parts[1])?
                    } else {
                        0
                    };
                    StepEntity::AdvancedFace { bounds, surface_id }
                },
                "DIRECTION" => {
                    let ratios = self.parse_coords(&args_to_parse)?;
                    StepEntity::Direction { ratios }
                },
                "AXIS2_PLACEMENT_3D" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let location = if parts.len() > 0 {
                        self.resolve_point(&parts[0])?
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let axis = if parts.len() > 1 {
                        self.resolve_direction(&parts[1])?
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    let ref_direction = if parts.len() > 2 {
                        self.resolve_direction(&parts[2])?
                    } else {
                        [1.0, 0.0, 0.0]
                    };
                    StepEntity::Axis2Placement3D { location, axis, ref_direction }
                },
                "VERTEX_POINT" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let point_id = self.resolve_id(&parts[parts.len() - 1])?;
                    StepEntity::VertexPoint { point_id }
                },
                "LINE" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let start = if parts.len() > 0 {
                        self.resolve_point(&parts[0])?
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let direction = if parts.len() > 1 {
                        self.resolve_direction(&parts[1])?
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    StepEntity::Line { start, direction }
                },
                "CIRCLE" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let center = if parts.len() > 0 {
                        self.resolve_point(&parts[0])?
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let radius = if parts.len() > 1 {
                        self.parse_float(&parts[1])?
                    } else {
                        1.0
                    };
                    let axis = if parts.len() > 2 {
                        self.resolve_direction(&parts[2])?
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    StepEntity::Circle { center, radius, axis }
                },
                "EDGE_CURVE" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let start_id = self.resolve_id(&parts[0])?;
                    let end_id = self.resolve_id(&parts[1])?;
                    let curve_id = self.resolve_id(&parts[2])?;
                    StepEntity::EdgeCurve { start_id, end_id, curve_id }
                },
                "ORIENTED_EDGE" => {
                    // Format: ORIENTED_EDGE('', *, *, edge_ref, .T./.F.)
                    // The edge_ref is typically the fourth argument, after name and two wildcards
                    let parts = self.parse_list(&args_to_parse)?;
                    let edge_id = if parts.len() > 2 {
                        // Skip the first two parts (usually wildcards), use the third
                        self.resolve_id(&parts[2])?
                    } else if parts.len() > 0 {
                        self.resolve_id(&parts[0])?
                    } else {
                        0
                    };
                    StepEntity::OrientedEdge { edge_id }
                },
                "EDGE_LOOP" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let edge_ids = parts.iter()
                        .map(|p| self.resolve_id(p))
                        .collect::<Result<Vec<_>>>()?;
                    StepEntity::EdgeLoop { edge_ids }
                },
                "FACE_BOUND" | "FACE_OUTER_BOUND" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let loop_id = self.resolve_id(&parts[0])?;
                    if *entity_type == "FACE_BOUND" {
                        StepEntity::FaceBound { loop_id }
                    } else {
                        StepEntity::FaceOuterBound { loop_id }
                    }
                },
                "ADVANCED_FACE" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let bounds = if parts.len() > 0 {
                        let bound_list = self.parse_list(&parts[0])?;
                        bound_list.iter()
                            .map(|p| self.resolve_id(p))
                            .collect::<Result<Vec<_>>>()?
                    } else {
                        vec![]
                    };
                    let surface_id = if parts.len() > 1 {
                        self.resolve_id(&parts[1])?
                    } else {
                        0
                    };
                    StepEntity::AdvancedFace { bounds, surface_id }
                },
                "PLANE" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let location = if parts.len() > 0 {
                        self.resolve_point(&parts[0])?
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let normal = if parts.len() > 1 {
                        self.resolve_direction(&parts[1])?
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    StepEntity::Plane { location, normal }
                },
                "CYLINDRICAL_SURFACE" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let location = if parts.len() > 0 {
                        self.resolve_point(&parts[0])?
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let axis = if parts.len() > 1 {
                        self.resolve_direction(&parts[1])?
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    let radius = if parts.len() > 2 {
                        self.parse_float(&parts[2])?
                    } else {
                        1.0
                    };
                    StepEntity::Cylinder { location, axis, radius }
                },
                "SPHERICAL_SURFACE" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let center = if parts.len() > 0 {
                        self.resolve_point(&parts[0])?
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                    let radius = if parts.len() > 1 {
                        self.parse_float(&parts[1])?
                    } else {
                        1.0
                    };
                    StepEntity::Sphere { center, radius }
                },
                "CLOSED_SHELL" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let face_ids = parts.iter()
                        .map(|p| self.resolve_id(p))
                        .collect::<Result<Vec<_>>>()?;
                    StepEntity::ClosedShell { face_ids }
                },
                "MANIFOLD_SOLID_BREP" => {
                    let parts = self.parse_list(&args_to_parse)?;
                    let shell_id = if parts.len() > 0 {
                        self.resolve_id(&parts[0])?
                    } else {
                        0
                    };
                    StepEntity::ManifoldSolidBrep { shell_id }
                },
                _ => {
                    // Unknown entity, skip
                    return Ok(StepEntity::CartesianPoint { coords: [0.0, 0.0, 0.0] });
                }
            };
            
            Ok(entity)
        } else {
            Err(CascadeError::InvalidGeometry("Invalid entity format".into()))
        }
    }
    
    fn parse_list(&self, args_str: &str) -> Result<Vec<String>> {
        let mut args_str = args_str.trim();
        
        // Handle parenthesized lists: unwrap (#1, #2, ...) to #1, #2, ...
        if args_str.starts_with('(') && args_str.ends_with(')') {
            args_str = &args_str[1..args_str.len()-1];
        }
        
        let mut result = vec![];
        let mut current = String::new();
        let mut depth = 0;
        let mut in_string = false;
        
        for ch in args_str.chars() {
            match ch {
                '(' | '[' => {
                    depth += 1;
                    current.push(ch);
                },
                ')' | ']' => {
                    depth -= 1;
                    current.push(ch);
                },
                '\'' => {
                    in_string = !in_string;
                    current.push(ch);
                },
                ',' if depth == 0 && !in_string => {
                    if !current.trim().is_empty() {
                        result.push(current.trim().to_string());
                    }
                    current.clear();
                },
                _ => {
                    current.push(ch);
                }
            }
        }
        
        if !current.trim().is_empty() {
            result.push(current.trim().to_string());
        }
        
        Ok(result)
    }
    
    fn parse_coords(&self, coord_str: &str) -> Result<[f64; 3]> {
        let parts = self.parse_list(coord_str)?;
        if parts.len() == 3 {
            Ok([
                self.parse_float(&parts[0])?,
                self.parse_float(&parts[1])?,
                self.parse_float(&parts[2])?,
            ])
        } else {
            Err(CascadeError::InvalidGeometry(format!("Expected 3 coordinates, got {}", parts.len())))
        }
    }
    
    fn parse_float(&self, s: &str) -> Result<f64> {
        s.trim().parse::<f64>()
            .map_err(|_| CascadeError::InvalidGeometry(format!("Cannot parse float: {}", s)))
    }
    
    fn resolve_id(&self, s: &str) -> Result<usize> {
        let s = s.trim();
        if let Some(id_str) = s.strip_prefix('#') {
            id_str.parse::<usize>()
                .map_err(|_| CascadeError::InvalidGeometry(format!("Invalid ID: {}", s)))
        } else {
            Err(CascadeError::InvalidGeometry(format!("Expected reference, got: {}", s)))
        }
    }
    
    fn resolve_point(&self, s: &str) -> Result<[f64; 3]> {
        let id = self.resolve_id(s)?;
        
        // Try to get direct point first
        if let Some(point) = self.points.get(&id) {
            return Ok(*point);
        }
        
        // Try to resolve via entity reference (e.g., AXIS2_PLACEMENT_3D location)
        if let Some(entity) = self.entities.get(&id) {
            match entity {
                StepEntity::Axis2Placement3D { location, .. } => {
                    return Ok(*location);
                },
                _ => {}
            }
        }
        
        Err(CascadeError::InvalidGeometry(format!("Point not found: #{}", id)))
    }
    
    fn resolve_direction(&self, s: &str) -> Result<[f64; 3]> {
        let id = self.resolve_id(s)?;
        
        if let Some(point) = self.points.get(&id) {
            // It's a point, use as direction (normalize)
            let mag = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
            if mag > 1e-10 {
                Ok([point[0] / mag, point[1] / mag, point[2] / mag])
            } else {
                Ok([0.0, 0.0, 1.0])
            }
        } else if let Some(entity) = self.entities.get(&id) {
            match entity {
                StepEntity::Direction { ratios } => Ok(*ratios),
                StepEntity::Axis2Placement3D { axis, .. } => Ok(*axis),
                _ => Ok([0.0, 0.0, 1.0]),
            }
        } else {
            Ok([0.0, 0.0, 1.0])
        }
    }
    
    fn parse(&self) -> Result<Shape> {
        // Find the root solid or shape
        let mut root_solid = None;
        
        for (id, entity) in &self.entities {
            if let StepEntity::ManifoldSolidBrep { shell_id } = entity {
                root_solid = Some((*id, *shell_id));
                break;
            }
        }
        
        if let Some((_, shell_id)) = root_solid {
            let solid = self.build_solid(shell_id)?;
            Ok(Shape::Solid(solid))
        } else {
            // Try to find a closed shell
            for (id, entity) in &self.entities {
                if let StepEntity::ClosedShell { .. } = entity {
                    let shell = self.build_shell(*id)?;
                    return Ok(Shape::Shell(shell));
                }
            }
            
            Err(CascadeError::InvalidGeometry("No solid or shell found in STEP file".into()))
        }
    }
    
    fn build_solid(&self, shell_id: usize) -> Result<Solid> {
        let outer_shell = self.build_shell(shell_id)?;
        Ok(Solid {
            outer_shell,
            inner_shells: vec![],
        })
    }
    
    fn build_shell(&self, shell_id: usize) -> Result<Shell> {
        if let Some(StepEntity::ClosedShell { face_ids }) = self.entities.get(&shell_id) {
            let mut faces = vec![];
            for face_id in face_ids {
                faces.push(self.build_face(*face_id)?);
            }
            Ok(Shell {
                faces,
                closed: true,
            })
        } else {
            Err(CascadeError::InvalidGeometry(format!("Shell not found: #{}", shell_id)))
        }
    }
    
    fn build_face(&self, face_id: usize) -> Result<Face> {
        if let Some(StepEntity::AdvancedFace { bounds, surface_id }) = self.entities.get(&face_id) {
            let surface_type = self.build_surface(*surface_id)?;
            
            let mut outer_wire = None;
            let mut inner_wires = vec![];
            
            for (i, bound_id) in bounds.iter().enumerate() {
                let wire = self.build_wire_from_bound(*bound_id)?;
                if i == 0 {
                    outer_wire = Some(wire);
                } else {
                    inner_wires.push(wire);
                }
            }
            
            Ok(Face {
                outer_wire: outer_wire.unwrap_or_else(|| Wire {
                    edges: vec![],
                    closed: false,
                }),
                inner_wires,
                surface_type,
            })
        } else {
            Err(CascadeError::InvalidGeometry(format!("Face not found: #{}", face_id)))
        }
    }
    
    fn build_wire_from_bound(&self, bound_id: usize) -> Result<Wire> {
        let loop_id = match self.entities.get(&bound_id) {
            Some(StepEntity::FaceOuterBound { loop_id }) | 
            Some(StepEntity::FaceBound { loop_id }) => *loop_id,
            _ => return Err(CascadeError::InvalidGeometry(format!("Bound not found: #{}", bound_id))),
        };
        
        self.build_wire(loop_id)
    }
    
    fn build_wire(&self, loop_id: usize) -> Result<Wire> {
        if let Some(StepEntity::EdgeLoop { edge_ids }) = self.entities.get(&loop_id) {
            let mut edges = vec![];
            for edge_id in edge_ids {
                edges.push(self.build_edge(*edge_id)?);
            }
            Ok(Wire {
                edges,
                closed: true,
            })
        } else {
            Err(CascadeError::InvalidGeometry(format!("Loop not found: #{}", loop_id)))
        }
    }
    
    fn build_edge(&self, edge_id: usize) -> Result<Edge> {
        if let Some(StepEntity::EdgeCurve { start_id: _, end_id: _, curve_id }) = self.entities.get(&edge_id) {
            let start = Vertex::new(0.0, 0.0, 0.0);
            let end = Vertex::new(0.0, 0.0, 0.0);
            
            let curve_type = self.build_curve_type(*curve_id)?;
            
            Ok(Edge {
                start,
                end,
                curve_type,
            })
        } else {
            Err(CascadeError::InvalidGeometry(format!("Edge not found: #{}", edge_id)))
        }
    }
    
    fn build_curve_type(&self, curve_id: usize) -> Result<CurveType> {
        match self.entities.get(&curve_id) {
            Some(StepEntity::Line { .. }) => Ok(CurveType::Line),
            Some(StepEntity::Circle { center, radius, .. }) => {
                Ok(CurveType::Arc {
                    center: *center,
                    radius: *radius,
                })
            },
            _ => Ok(CurveType::Line),
        }
    }
    
    fn build_surface(&self, surface_id: usize) -> Result<SurfaceType> {
        match self.entities.get(&surface_id) {
            Some(StepEntity::Plane { location, normal }) => {
                Ok(SurfaceType::Plane {
                    origin: *location,
                    normal: *normal,
                })
            },
            Some(StepEntity::Cylinder { location, axis, radius }) => {
                Ok(SurfaceType::Cylinder {
                    origin: *location,
                    axis: *axis,
                    radius: *radius,
                })
            },
            Some(StepEntity::Sphere { center, radius }) => {
                Ok(SurfaceType::Sphere {
                    center: *center,
                    radius: *radius,
                })
            },
            _ => {
                Ok(SurfaceType::Plane {
                    origin: [0.0, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_step_box() {
        // This test will be validated when we have a real STEP file
        let _ = read_step("test_data/box.step");
    }
}
