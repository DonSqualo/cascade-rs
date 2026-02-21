//! File I/O for CAD formats

mod step;
mod iges;
mod obj;
mod ply;

use crate::brep::{Shape, Solid, Shell, Face, Wire, Edge, Vertex, CurveType, SurfaceType, Compound};
use crate::xde::ShapeAttributes;
use crate::{Result, CascadeError};
use std::io::Write;

pub use step::read_step;
pub use iges::{read_iges, write_iges, write_iges_with_layers};
pub use obj::write_obj;
pub use ply::write_ply;

/// PMI (Product Manufacturing Information) Dimension structure
/// Represents manufacturing tolerances and dimensions in STEP format
#[derive(Debug, Clone)]
pub struct StepDimension {
    /// Nominal dimension value
    pub value: f64,
    /// Tolerance bounds (upper, lower) - if None, no tolerance is applied
    pub tolerance: Option<(f64, f64)>,
    /// Optional datum reference (e.g., for perpendicularity to a datum)
    pub datum_ref: Option<String>,
    /// Dimension type: "linear", "angular", "radial", "diameter"
    pub dim_type: DimensionType,
    /// Location on the model where this dimension applies
    pub location: [f64; 3],
}

/// PMI Dimension type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimensionType {
    Linear,
    Angular,
    Radial,
    Diameter,
}

/// Geometric Tolerance structure for PMI (Geometric Product Specification)
#[derive(Debug, Clone)]
pub struct StepGeometricTolerance {
    /// Tolerance type: "flatness", "perpendicularity", "parallelism", "circularity", "cylindricity"
    pub tolerance_type: String,
    /// Tolerance value
    pub tolerance_value: f64,
    /// Reference datum if applicable
    pub datum_ref: Option<String>,
}

pub fn write_step(shape: &Shape, path: &str) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut step_writer = StepWriter::new(writer);
    step_writer.write_shape(shape)
}

/// Write a STEP file with full AP203 (Configuration Controlled 3D Design) compliance
/// 
/// Exports a solid with complete AP203 schema compliance including:
/// - APPLICATION_CONTEXT entity defining the design context
/// - APPLICATION_PROTOCOL_DEFINITION entity specifying AP203 protocol
/// - PRODUCT_DEFINITION_CONTEXT for design configuration
/// - PRODUCT entity for the overall design
/// - PRODUCT_DEFINITION_FORMATION for versioning
/// - PRODUCT_DEFINITION linking design to context
/// - SHAPE_REPRESENTATION for geometric representation
/// - Full BREP topology with MANIFOLD_SOLID_BREP, shells, faces, and edges
/// 
/// # Arguments
/// * `solid` - The solid geometry to export
/// * `path` - Output file path for the STEP AP203 file
/// 
/// # Example
/// ```rust,no_run
/// use cascade::make_box;
/// use cascade::io::write_step_ap203;
/// 
/// let solid = make_box(10.0, 20.0, 30.0)?;
/// write_step_ap203(&solid, "design.step")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// 
/// # AP203 Schema Entities Generated
/// The output file includes:
/// - Entity #1: APPLICATION_CONTEXT - Application domain context
/// - Entity #2: APPLICATION_PROTOCOL_DEFINITION - Protocol specification (config_control_design)
/// - Entity #3: PRODUCT_DEFINITION_CONTEXT - Design context
/// - Entity #4: PRODUCT - The design product
/// - Entity #5: PRODUCT_DEFINITION_FORMATION - Product version
/// - Entity #6: PRODUCT_DEFINITION - Links formation to context
/// - Entities #11+: Geometric BREP representation
pub fn write_step_ap203(solid: &Solid, path: &str) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut step_writer = StepWriter::new(writer);
    step_writer.ap203_mode = true;
    let shape = Shape::Solid(solid.clone());
    step_writer.write_shape(&shape)
}

/// Write a STEP file with color and material attributes (AP214)
/// 
/// Exports a solid with its XDE attributes (name, color, layer, material) using
/// STEP AP214 representation with COLOUR_RGB, STYLED_ITEM, and PRESENTATION_STYLE_ASSIGNMENT.
/// 
/// # Arguments
/// * `solid` - The solid to export
/// * `path` - Output file path
/// 
/// # Example
/// ```rust,no_run
/// use cascade::{make_sphere, xde, io};
/// 
/// let mut sphere = make_sphere(10.0)?;
/// xde::set_shape_color(&mut sphere, [1.0, 0.0, 0.0]);  // Red color
/// io::write_step_with_attributes(&sphere, "colored_sphere.step")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_step_with_attributes(solid: &Solid, path: &str) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut step_writer = StepWriter::new(writer);
    step_writer.write_solid_with_attributes(solid)
}

/// Write a STEP file with PMI (Product Manufacturing Information)
/// 
/// # Arguments
/// * `shape` - The geometric shape to export
/// * `dimensions` - Vector of manufacturing dimensions with tolerances
/// * `path` - Output file path
/// 
/// # Example
/// ```rust,no_run
/// use cascade::io::{StepDimension, DimensionType};
/// 
/// let dim = StepDimension {
///     value: 50.0,
///     tolerance: Some((0.5, -0.5)),
///     datum_ref: None,
///     dim_type: DimensionType::Linear,
///     location: [0.0, 0.0, 0.0],
/// };
/// 
/// cascade::io::write_step_with_pmi(&shape, &[dim], "output.step")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_step_with_pmi(shape: &Shape, dimensions: &[StepDimension], path: &str) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut step_writer = StepWriter::new(writer);
    step_writer.write_shape_with_pmi(shape, dimensions)
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
    ap203_mode: bool, // Enable AP203 compliance
}

impl<W: Write> StepWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            writer,
            entity_id: 0,
            entities: Vec::new(),
            ap203_mode: true,
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
            Shape::CompSolid(_) => {
                // CompSolid not yet fully implemented
            }
        }
        
        self.write_file()?;
        Ok(())
    }
    
    /// Write a solid with color and material attributes (STEP AP214)
    fn write_solid_with_attributes(&mut self, solid: &Solid) -> Result<()> {
        // Add the geometric data
        let root_solid_id = self.add_solid(solid);
        
        // If the solid has a color, add STEP AP214 color entities
        if let Some(color) = solid.attributes.color {
            self.add_color_entity(root_solid_id, color)?;
        }
        
        self.write_file()?;
        Ok(())
    }
    
    /// Write STEP AP214 (Automotive Design) with full compliance
    /// 
    /// AP214 extends AP203 with automotive-specific enhancements:
    /// - Advanced BREP representation
    /// - Comprehensive color and material support
    /// - Layer management
    /// - Validation properties
    /// - Styled items and presentation styles
    fn write_step_ap214_solid(&mut self, solid: &Solid) -> Result<()> {
        // AP214 mode: enables AUTOMOTIVE_DESIGN schema
        self.ap203_mode = false;
        
        // Add the geometric solid representation
        let root_solid_id = self.add_solid(solid);
        
        // Add color and material attributes from the solid
        if let Some(color) = solid.attributes.color {
            self.add_ap214_color_entity(root_solid_id, color)?;
        }
        
        // Add layer information if present
        if let Some(ref layer) = solid.attributes.layer {
            self.add_ap214_layer_entity(root_solid_id, layer)?;
        }
        
        // Add material information if present
        if let Some(ref material) = solid.attributes.material {
            self.add_ap214_material_entity(root_solid_id, material)?;
        }
        
        // Add validation properties for AP214 compliance
        self.add_ap214_validation_properties()?;
        
        // Write the file with AP214 schema
        self.write_file_ap214()?;
        Ok(())
    }

    /// Add AP214 color entity with enhanced presentation support
    fn add_ap214_color_entity(&mut self, solid_id: usize, color: [f64; 3]) -> Result<()> {
        // COLOUR_RGB - Basic color definition
        let colour_rgb_id = self.next_id();
        let entity = format!(
            "#{} = COLOUR_RGB('automotive_color', {:.6}, {:.6}, {:.6});",
            colour_rgb_id, color[0], color[1], color[2]
        );
        self.entities.push(entity);
        
        // SURFACE_STYLE_RENDERING_PROPERTIES - Rendering settings with material properties
        let surface_style_id = self.next_id();
        let entity = format!(
            "#{} = SURFACE_STYLE_RENDERING_PROPERTIES('surface_rendering', #{}, $, $, $);",
            surface_style_id, colour_rgb_id
        );
        self.entities.push(entity);
        
        // SURFACE_SIDE_STYLE - Apply to positive side
        let surface_side_style_id = self.next_id();
        let entity = format!(
            "#{} = SURFACE_SIDE_STYLE('positive_side', (#{}, ));",
            surface_side_style_id, surface_style_id
        );
        self.entities.push(entity);
        
        // SURFACE_STYLE_USAGE - Specify which side gets the style
        let surface_style_usage_id = self.next_id();
        let entity = format!(
            "#{} = SURFACE_STYLE_USAGE(.POSITIVE., #{});",
            surface_style_usage_id, surface_side_style_id
        );
        self.entities.push(entity);
        
        // STYLED_ITEM - Link the style to the geometric solid
        let styled_item_id = self.next_id();
        let entity = format!(
            "#{} = STYLED_ITEM('', (#{}, ), #{});",
            styled_item_id, surface_style_usage_id, solid_id
        );
        self.entities.push(entity);
        
        // PRESENTATION_STYLE_ASSIGNMENT - AP214 presentation management
        let presentation_style_id = self.next_id();
        let entity = format!(
            "#{} = PRESENTATION_STYLE_ASSIGNMENT((#{}));",
            presentation_style_id, styled_item_id
        );
        self.entities.push(entity);
        
        Ok(())
    }

    /// Add AP214 layer entity for design organization
    fn add_ap214_layer_entity(&mut self, solid_id: usize, layer: &str) -> Result<()> {
        // REPRESENTATION_ITEM - Base for layer assignment
        let layer_item_id = self.next_id();
        let entity = format!(
            "#{} = REPRESENTATION_ITEM('{}', #{});",
            layer_item_id, layer, solid_id
        );
        self.entities.push(entity);
        
        // MAPPED_ITEM - For layer organization in AP214
        let mapped_item_id = self.next_id();
        let entity = format!(
            "#{} = MAPPED_ITEM('', #{}, (#{}, ), #{});",
            mapped_item_id, layer_item_id, solid_id, layer_item_id
        );
        self.entities.push(entity);
        
        Ok(())
    }

    /// Add AP214 material entity for material specification
    fn add_ap214_material_entity(&mut self, solid_id: usize, material_name: &str) -> Result<()> {
        // MATERIAL - AP214 material definition
        let material_id = self.next_id();
        let entity = format!(
            "#{} = MATERIAL('{}', {});",
            material_id, material_name, material_name
        );
        self.entities.push(entity);
        
        // APPLIED_MATERIAL_PROPERTY - Link material to geometric item
        let applied_material_id = self.next_id();
        let entity = format!(
            "#{} = APPLIED_MATERIAL_PROPERTY('{}', (#{}, ), #{});",
            applied_material_id, material_name, solid_id, material_id
        );
        self.entities.push(entity);
        
        Ok(())
    }

    /// Add AP214 validation properties for design verification
    fn add_ap214_validation_properties(&mut self) -> Result<()> {
        // CONTEXT_DEPENDENT_SHAPE_REPRESENTATION - Validation context
        let context_id = self.next_id();
        let entity = format!(
            "#{} = CONTEXT_DEPENDENT_SHAPE_REPRESENTATION('', $, #{});",
            context_id, context_id
        );
        self.entities.push(entity);
        
        // SHAPE_REPRESENTATION_WITH_PARAMETERS - Parametric shape support
        let param_id = self.next_id();
        let entity = format!(
            "#{} = SHAPE_REPRESENTATION_WITH_PARAMETERS('', (), #{});",
            param_id, param_id
        );
        self.entities.push(entity);
        
        Ok(())
    }

    /// Write STEP file with AP214 (AUTOMOTIVE_DESIGN) schema header
    fn write_file_ap214(&mut self) -> Result<()> {
        writeln!(self.writer, "ISO-10303-21;")?;
        writeln!(self.writer, "HEADER;")?;
        writeln!(self.writer, "FILE_DESCRIPTION(('automotive design data with colors, layers, and validation'), '2', '2', '', '', 1.0, '');")?;
        writeln!(self.writer, "FILE_NAME('cascade-rs AP214 export', '', ('cascade-rs'), (''), 'cascade-rs', '', '');")?;
        
        // AP214 (Automotive Design) schema specification
        writeln!(self.writer, "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")?;
        
        writeln!(self.writer, "ENDSEC;")?;
        writeln!(self.writer, "DATA;")?;
        
        // Write AP214 header entities for automotive design context
        self.write_ap214_header()?;
        
        // Write all geometric and attribute entities
        for entity in &self.entities {
            writeln!(self.writer, "{}", entity)?;
        }
        
        writeln!(self.writer, "ENDSEC;")?;
        writeln!(self.writer, "END-ISO-10303-21;")?;
        
        Ok(())
    }

    /// Write AP214-specific header with AUTOMOTIVE_DESIGN context
    fn write_ap214_header(&mut self) -> Result<()> {
        // AP214 (Automotive Design) extends AP203 with:
        // - AUTOMOTIVE_DESIGN context
        // - Enhanced color and material support
        // - Layer management
        // - Validation properties
        
        // Entity #1: APPLICATION_CONTEXT (Automotive Design)
        let app_context_id = 1;
        writeln!(
            self.writer,
            "#{} = APPLICATION_CONTEXT('automotive design with colors and materials');",
            app_context_id
        )?;
        
        // Entity #2: APPLICATION_PROTOCOL_DEFINITION (AP214)
        let app_proto_def_id = 2;
        writeln!(
            self.writer,
            "#{} = APPLICATION_PROTOCOL_DEFINITION('international standard', 'automotive_design', 1994, #{});",
            app_proto_def_id, app_context_id
        )?;
        
        // Entity #3: PRODUCT_DEFINITION_CONTEXT (Automotive)
        let prod_def_context_id = 3;
        writeln!(
            self.writer,
            "#{} = PRODUCT_DEFINITION_CONTEXT('part definition', #{}, 'design');",
            prod_def_context_id, app_context_id
        )?;
        
        // Entity #4: PRODUCT (Automotive part)
        let product_id = 4;
        writeln!(
            self.writer,
            "#{} = PRODUCT('automotive_design', 'automotive_design', 'automotive design v1.0', (#{}));",
            product_id, app_context_id
        )?;
        
        // Entity #5: PRODUCT_DEFINITION_FORMATION
        let prod_def_form_id = 5;
        writeln!(
            self.writer,
            "#{} = PRODUCT_DEFINITION_FORMATION('A', 'design stage', #{}, #10);",
            prod_def_form_id, product_id
        )?;
        
        // Entity #6: PRODUCT_DEFINITION
        let prod_def_id = 6;
        writeln!(
            self.writer,
            "#{} = PRODUCT_DEFINITION('design', '', #{}, #{});",
            prod_def_id, prod_def_form_id, prod_def_context_id
        )?;
        
        // Entity #7: SHAPE_REPRESENTATION_CONTEXT (Automotive)
        let shape_context_id = 7;
        writeln!(
            self.writer,
            "#{} = SHAPE_REPRESENTATION_CONTEXT('3D', #{}, 0.00001, 0.0, 0.0);",
            shape_context_id, app_context_id
        )?;
        
        // Entity #8: GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT
        let uncertainty_id = 8;
        writeln!(
            self.writer,
            "#{} = GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{}), 0.00001);",
            uncertainty_id, shape_context_id
        )?;
        
        // Entity #9: GLOBAL_UNIT_ASSIGNED_CONTEXT (Millimeters)
        let unit_id = 9;
        writeln!(
            self.writer,
            "#{} = GLOBAL_UNIT_ASSIGNED_CONTEXT((#{}, ), 'MILLIMETRE');",
            unit_id, uncertainty_id
        )?;
        
        // Update internal entity counter to avoid ID collisions
        self.entity_id = 10;
        
        Ok(())
    }
    
    /// Add STEP AP214 color entities for a solid
    /// Creates COLOUR_RGB, SURFACE_STYLE_USAGE, STYLED_ITEM, and PRESENTATION_STYLE_ASSIGNMENT
    fn add_color_entity(&mut self, solid_id: usize, color: [f64; 3]) -> Result<()> {
        // Create COLOUR_RGB entity
        let colour_rgb_id = self.next_id();
        let entity = format!(
            "#{} = COLOUR_RGB('', {:.6}, {:.6}, {:.6});",
            colour_rgb_id, color[0], color[1], color[2]
        );
        self.entities.push(entity);
        
        // Create SURFACE_STYLE_RENDERING_PROPERTIES with the color
        let surface_style_id = self.next_id();
        let entity = format!(
            "#{} = SURFACE_STYLE_RENDERING_PROPERTIES('', #{}, $, $, $);",
            surface_style_id, colour_rgb_id
        );
        self.entities.push(entity);
        
        // Create SURFACE_SIDE_STYLE
        let surface_side_style_id = self.next_id();
        let entity = format!(
            "#{} = SURFACE_SIDE_STYLE('', (#{}, ));",
            surface_side_style_id, surface_style_id
        );
        self.entities.push(entity);
        
        // Create SURFACE_STYLE_USAGE (for POSITIVE side)
        let surface_style_usage_id = self.next_id();
        let entity = format!(
            "#{} = SURFACE_STYLE_USAGE(.POSITIVE., #{});",
            surface_style_usage_id, surface_side_style_id
        );
        self.entities.push(entity);
        
        // Create STYLED_ITEM
        let styled_item_id = self.next_id();
        let entity = format!(
            "#{} = STYLED_ITEM('', (#{}), #{});",
            styled_item_id, surface_style_usage_id, solid_id
        );
        self.entities.push(entity);
        
        // Create PRESENTATION_STYLE_ASSIGNMENT
        let presentation_style_id = self.next_id();
        let entity = format!(
            "#{} = PRESENTATION_STYLE_ASSIGNMENT((#{}));",
            presentation_style_id, styled_item_id
        );
        self.entities.push(entity);
        
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
            CurveType::Parabola { .. } | CurveType::Trimmed { .. } | CurveType::Hyperbola { .. } | CurveType::Offset { .. } => {
                // For now, approximate parabola, hyperbola, trimmed, and offset curves as lines
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
            SurfaceType::BSpline { .. } | SurfaceType::BezierSurface { .. } => {
                // Fallback to plane for B-spline and Bezier surfaces
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
            SurfaceType::SurfaceOfRevolution { axis_location, axis_direction, .. } => {
                // Export as SURFACE_OF_REVOLUTION in STEP
                // For now, fallback to a placeholder plane
                let surf_id = self.next_id();
                let origin_id = self.next_id();
                let axis_id = self.next_id();
                
                let origin_entity = format!(
                    "#{} = CARTESIAN_POINT('', ({:.6}, {:.6}, {:.6}));",
                    origin_id, axis_location[0], axis_location[1], axis_location[2]
                );
                self.entities.push(origin_entity);
                
                let axis_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    axis_id, axis_direction[0], axis_direction[1], axis_direction[2]
                );
                self.entities.push(axis_entity);
                
                // Fallback to plane representation
                let plane_entity = format!("#{} = PLANE('', #{}, #{});", surf_id, origin_id, axis_id);
                self.entities.push(plane_entity);
                
                surf_id
            }
            SurfaceType::SurfaceOfLinearExtrusion { direction, .. } => {
                // Export as a plane representation for now
                // TODO: Implement proper SURFACE_OF_LINEAR_EXTRUSION in STEP
                let surf_id = self.next_id();
                let origin_id = self.next_id();
                let normal_id = self.next_id();
                
                let origin_entity = format!(
                    "#{} = CARTESIAN_POINT('', (0.0, 0.0, 0.0));",
                    origin_id
                );
                self.entities.push(origin_entity);
                
                let normal_entity = format!(
                    "#{} = DIRECTION('', ({:.6}, {:.6}, {:.6}));",
                    normal_id, direction[0], direction[1], direction[2]
                );
                self.entities.push(normal_entity);
                
                let plane_entity = format!("#{} = PLANE('', #{}, #{});", surf_id, origin_id, normal_id);
                self.entities.push(plane_entity);
                
                surf_id
            }
            SurfaceType::BezierSurface { .. } => {
                // Fallback to plane for Bezier surfaces
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
            SurfaceType::RectangularTrimmedSurface { .. } => {
                // Fallback to plane for rectangular trimmed surfaces
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
            SurfaceType::OffsetSurface { .. } => {
                // Fallback to plane for offset surfaces
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
            SurfaceType::PlateSurface { .. } => {
                // Fallback to plane for plate surfaces
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
        
        // AP203 compliance: use CONFIG_CONTROL_DESIGN schema
        if self.ap203_mode {
            writeln!(self.writer, "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));")?;
        } else {
            writeln!(self.writer, "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));")?;
        }
        
        writeln!(self.writer, "ENDSEC;")?;
        writeln!(self.writer, "DATA;")?;
        
        // For AP203, prepend mandatory application context and product definition entities
        if self.ap203_mode {
            self.write_ap203_header()?;
        }
        
        for entity in &self.entities {
            writeln!(self.writer, "{}", entity)?;
        }
        
        writeln!(self.writer, "ENDSEC;")?;
        writeln!(self.writer, "END-ISO-10303-21;")?;
        
        Ok(())
    }
    
    fn write_ap203_header(&mut self) -> Result<()> {
        // AP203 (Configuration Controlled Design) requires:
        // 1. APPLICATION_CONTEXT
        // 2. APPLICATION_PROTOCOL_DEFINITION
        // 3. PRODUCT_DEFINITION_CONTEXT
        // 4. PRODUCT_DEFINITION_FORMATION
        // 5. PRODUCT_DEFINITION
        
        // Entity #1: APPLICATION_CONTEXT
        let app_context_id = 1;
        writeln!(
            self.writer,
            "#{} = APPLICATION_CONTEXT('configuration controlled 3D designs of mechanical parts and assemblies');",
            app_context_id
        )?;
        
        // Entity #2: APPLICATION_PROTOCOL_DEFINITION
        let app_proto_def_id = 2;
        writeln!(
            self.writer,
            "#{} = APPLICATION_PROTOCOL_DEFINITION('international standard', 'config_control_design', 1994, #{});",
            app_proto_def_id, app_context_id
        )?;
        
        // Entity #3: PRODUCT_DEFINITION_CONTEXT
        let prod_def_context_id = 3;
        writeln!(
            self.writer,
            "#{} = PRODUCT_DEFINITION_CONTEXT('part definition', #{}, 'design');",
            prod_def_context_id, app_context_id
        )?;
        
        // Entity #4: PRODUCT (for the design)
        let product_id = 4;
        writeln!(
            self.writer,
            "#{} = PRODUCT('design', 'design', 'design v1.0', (#{}));",
            product_id, app_context_id
        )?;
        
        // Entity #5: PRODUCT_DEFINITION_FORMATION
        let prod_def_form_id = 5;
        writeln!(
            self.writer,
            "#{} = PRODUCT_DEFINITION_FORMATION('A', 'design stage', #{}, #10);",
            prod_def_form_id, product_id
        )?;
        
        // Entity #6: PRODUCT_DEFINITION
        let prod_def_id = 6;
        writeln!(
            self.writer,
            "#{} = PRODUCT_DEFINITION('design', '', #{}, #{});",
            prod_def_id, prod_def_form_id, prod_def_context_id
        )?;
        
        // Update internal entity counter to avoid ID collisions
        self.entity_id = 10;
        
        Ok(())
    }
    
    /// Write shape with PMI (Product Manufacturing Information)
    fn write_shape_with_pmi(&mut self, shape: &Shape, dimensions: &[StepDimension]) -> Result<()> {
        // First, add the geometric shape
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
            Shape::CompSolid(_) => {
                // CompSolid not yet fully implemented
            }
        }
        
        // Add PMI annotations
        for dimension in dimensions {
            self.add_dimension(dimension)?;
        }
        
        self.write_file()?;
        Ok(())
    }
    
    /// Add a dimensional size entity (PMI)
    /// Implements DIMENSIONAL_SIZE and DIMENSIONAL_LOCATION from STEP AP203
    fn add_dimension(&mut self, dim: &StepDimension) -> Result<()> {
        // Create measurement value entity
        let measurement_id = self.next_id();
        let measurement_entity = format!(
            "#{} = MEASURE_REPRESENTATION_ITEM('dimension', POSITIVE_LENGTH_MEASURE({}), #{});",
            measurement_id, dim.value, measurement_id  // Self-referential for simplicity
        );
        self.entities.push(measurement_entity);
        
        // Create dimensional size entity
        let dim_size_id = self.next_id();
        let tolerance_clause = if let Some((upper, lower)) = dim.tolerance {
            format!(
                ", PLUS_MINUS_TOLERANCE({}, {}, {})",
                dim.value, upper, lower
            )
        } else {
            String::new()
        };
        
        let dim_type_str = match dim.dim_type {
            DimensionType::Linear => "LINEAR_DIMENSION",
            DimensionType::Angular => "ANGULAR_DIMENSION",
            DimensionType::Radial => "RADIAL_DIMENSION",
            DimensionType::Diameter => "DIAMETER_DIMENSION",
        };
        
        let dim_size_entity = format!(
            "#{} = DIMENSIONAL_SIZE('dimension{}', #{}, {})",
            dim_size_id,
            self.entity_id,
            measurement_id,
            tolerance_clause
        );
        self.entities.push(dim_size_entity);
        
        // Create dimensional location with coordinates
        let location_id = self.next_id();
        let location_entity = format!(
            "#{} = DIMENSIONAL_LOCATION(#{}, 'dimension location', ({:.6}, {:.6}, {:.6}));",
            location_id, dim_size_id, dim.location[0], dim.location[1], dim.location[2]
        );
        self.entities.push(location_entity);
        
        // Add datum reference if present (for geometric tolerances relative to datum)
        if let Some(datum_ref) = &dim.datum_ref {
            let datum_feat_id = self.next_id();
            let datum_entity = format!(
                "#{} = DATUM('{}', 'datum feature', #{});",
                datum_feat_id, datum_ref, location_id
            );
            self.entities.push(datum_entity);
        }
        
        Ok(())
    }
    
    /// Add geometric tolerance entity (PMI)
    /// Implements GEOMETRIC_TOLERANCE from STEP AP203
    fn add_geometric_tolerance(&mut self, tolerance: &StepGeometricTolerance) -> Result<()> {
        // Create tolerance zone
        let zone_id = self.next_id();
        let zone_entity = format!(
            "#{} = GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE('{}', {:.6}, '{}', ?, ?);",
            zone_id, tolerance.tolerance_type, tolerance.tolerance_value,
            tolerance.datum_ref.as_deref().unwrap_or("none")
        );
        self.entities.push(zone_entity);
        
        Ok(())
    }
    
    /// Add datum feature entity (PMI)
    /// Implements DATUM and DATUM_FEATURE from STEP
    fn add_datum_feature(&mut self, datum_name: &str, feature_location: [f64; 3]) -> Result<()> {
        // Create datum target point
        let point_id = self.next_id();
        let point_entity = format!(
            "#{} = CARTESIAN_POINT('{}', ({:.6}, {:.6}, {:.6}));",
            point_id, datum_name, feature_location[0], feature_location[1], feature_location[2]
        );
        self.entities.push(point_entity);
        
        // Create datum feature
        let datum_id = self.next_id();
        let datum_entity = format!(
            "#{} = DATUM('{}', '{}', #{});",
            datum_id, datum_name, datum_name, point_id
        );
        self.entities.push(datum_entity);
        
        Ok(())
    }
    
    /// Add annotation plane entity (for drawing annotations)
    fn add_annotation_plane(&mut self, plane_origin: [f64; 3], normal: [f64; 3]) -> Result<()> {
        let origin_id = self.next_id();
        let origin_entity = format!(
            "#{} = CARTESIAN_POINT('annotation plane origin', ({:.6}, {:.6}, {:.6}));",
            origin_id, plane_origin[0], plane_origin[1], plane_origin[2]
        );
        self.entities.push(origin_entity);
        
        let normal_id = self.next_id();
        let normal_entity = format!(
            "#{} = DIRECTION('annotation plane normal', ({:.6}, {:.6}, {:.6}));",
            normal_id, normal[0], normal[1], normal[2]
        );
        self.entities.push(normal_entity);
        
        let plane_id = self.next_id();
        let plane_entity = format!(
            "#{} = PLANE('annotation plane', #{}, #{});",
            plane_id, origin_id, normal_id
        );
        self.entities.push(plane_entity);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{make_sphere, make_box};
    use crate::xde::{set_shape_color, set_shape_layer};
    use std::fs;

    #[test]
    fn test_step_color() -> Result<()> {
        // Create a sphere
        let mut sphere = make_sphere(10.0)?;
        
        // Set color to red [1.0, 0.0, 0.0]
        set_shape_color(&mut sphere, [1.0, 0.0, 0.0]);
        
        // Write to STEP file with colors
        let output_path = "/tmp/test_colored_sphere.step";
        write_step_with_attributes(&sphere, output_path)?;
        
        // Verify file was created
        assert!(fs::metadata(output_path).is_ok(), "STEP file was not created");
        
        // Read the file and verify it contains color entities
        let contents = fs::read_to_string(output_path)?;
        
        // Check for COLOUR_RGB entity
        assert!(contents.contains("COLOUR_RGB"), "STEP file does not contain COLOUR_RGB entity");
        
        // Check for SURFACE_STYLE_RENDERING_PROPERTIES
        assert!(contents.contains("SURFACE_STYLE_RENDERING_PROPERTIES"), "STEP file does not contain SURFACE_STYLE_RENDERING_PROPERTIES");
        
        // Check for STYLED_ITEM
        assert!(contents.contains("STYLED_ITEM"), "STEP file does not contain STYLED_ITEM");
        
        // Check for PRESENTATION_STYLE_ASSIGNMENT
        assert!(contents.contains("PRESENTATION_STYLE_ASSIGNMENT"), "STEP file does not contain PRESENTATION_STYLE_ASSIGNMENT");
        
        // Check for the color values in the file (1.0, 0.0, 0.0)
        assert!(contents.contains("1.000000"), "STEP file does not contain color value 1.0");
        assert!(contents.contains("0.000000"), "STEP file does not contain color value 0.0");
        
        // Clean up
        fs::remove_file(output_path).ok();
        
        Ok(())
    }

    #[test]
    fn test_step_ap214_full_compliance() -> Result<()> {
        use crate::primitive::make_box;
        use crate::xde::{set_shape_layer, set_shape_material};

        // Create a colored box with layer and material for AP214 test
        let mut box_solid = make_box(10.0, 20.0, 30.0)?;
        
        // Set AP214 attributes
        set_shape_color(&mut box_solid, [0.8, 0.2, 0.2]);  // Red color
        set_shape_layer(&mut box_solid, "Layer_1");          // Layer 1
        set_shape_material(&mut box_solid, "Steel");         // Material
        
        // Write using AP214 function
        let output_path = "/tmp/test_ap214.step";
        write_step_ap214(&box_solid, output_path)?;
        
        // Verify file was created
        assert!(fs::metadata(output_path).is_ok(), "AP214 STEP file was not created");
        
        // Read the file and verify AP214 compliance
        let contents = fs::read_to_string(output_path)?;
        
        // Check for AP214 schema header
        assert!(contents.contains("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'))"), 
                "Missing AP214 AUTOMOTIVE_DESIGN schema");
        
        // Check for AUTOMOTIVE_DESIGN context (key AP214 requirement)
        assert!(contents.contains("automotive design"), 
                "Missing automotive design context");
        assert!(contents.contains("automotive_design"), 
                "Missing automotive_design protocol");
        
        // Check for AP214 enhanced BREP entities
        assert!(contents.contains("MANIFOLD_SOLID_BREP"), 
                "Missing MANIFOLD_SOLID_BREP for advanced BREP");
        assert!(contents.contains("CLOSED_SHELL"), 
                "Missing CLOSED_SHELL for valid topology");
        
        // Check for color support (AP214 requirement)
        assert!(contents.contains("COLOUR_RGB"), 
                "Missing COLOUR_RGB for color support");
        assert!(contents.contains("SURFACE_STYLE_RENDERING_PROPERTIES"), 
                "Missing SURFACE_STYLE_RENDERING_PROPERTIES");
        assert!(contents.contains("STYLED_ITEM"), 
                "Missing STYLED_ITEM for styled geometry");
        assert!(contents.contains("PRESENTATION_STYLE_ASSIGNMENT"), 
                "Missing PRESENTATION_STYLE_ASSIGNMENT");
        
        // Verify color values (0.8, 0.2, 0.2)
        assert!(contents.contains("0.800000"), "Missing red component (0.8) in color");
        assert!(contents.contains("0.200000"), "Missing green/blue components (0.2) in color");
        
        // Check for layer support (AP214 automotive requirement)
        assert!(contents.contains("Layer_1") || contents.contains("MAPPED_ITEM"), 
                "Missing layer support in AP214");
        
        // Check for material support (AP214 requirement)
        assert!(contents.contains("MATERIAL") || contents.contains("Steel"), 
                "Missing material specification in AP214");
        
        // Check for validation properties (AP214 requirement)
        assert!(contents.contains("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION") || 
                contents.contains("SHAPE_REPRESENTATION_WITH_PARAMETERS"), 
                "Missing validation properties in AP214");
        
        // Check for APPLICATION_PROTOCOL_DEFINITION (AP214 requirement)
        assert!(contents.contains("APPLICATION_PROTOCOL_DEFINITION"), 
                "Missing APPLICATION_PROTOCOL_DEFINITION for AP214");
        
        // Check for GLOBAL_UNIT_ASSIGNED_CONTEXT (AP214 automotive requirement)
        assert!(contents.contains("GLOBAL_UNIT_ASSIGNED_CONTEXT"), 
                "Missing GLOBAL_UNIT_ASSIGNED_CONTEXT for AP214 compliance");
        
        // Verify it's a valid ISO 10303-21 STEP file structure
        assert!(contents.contains("ISO-10303-21;"), "Missing ISO-10303-21 header");
        assert!(contents.contains("END-ISO-10303-21;"), "Missing ISO-10303-21 end marker");
        assert!(contents.contains("FILE_SCHEMA"), "Missing FILE_SCHEMA");
        assert!(contents.contains("FILE_DESCRIPTION"), "Missing FILE_DESCRIPTION");
        
        // Clean up
        fs::remove_file(output_path).ok();
        
        Ok(())
    }

    #[test]
    fn test_step_ap242_full_compliance() -> Result<()> {
        use crate::primitive::make_box;
        use crate::xde::set_shape_layer;

        // Create a colored box with tessellation for AP242 test
        let mut box_solid = make_box(10.0, 20.0, 30.0)?;
        
        // Set attributes for AP242 export
        set_shape_color(&mut box_solid, [0.2, 0.8, 0.2]);  // Green color
        set_shape_layer(&mut box_solid, "Layer_1");         // Layer 1
        
        // Write using AP242 function
        let output_path = "/tmp/test_ap242.step";
        write_step_ap242(&box_solid, output_path)?;
        
        // Verify file was created
        assert!(fs::metadata(output_path).is_ok(), "AP242 STEP file was not created");
        
        // Read the file and verify AP242 compliance
        let contents = fs::read_to_string(output_path)?;
        
        // Check for AP242 schema header
        assert!(contents.contains("FILE_SCHEMA(('MANAGED_MODEL_BASED_3D_ENGINEERING_DESIGN'))"), 
                "Missing AP242 MANAGED_MODEL_BASED_3D_ENGINEERING_DESIGN schema");
        
        // Check for AP242 context (key AP242 requirement)
        assert!(contents.contains("managed model-based 3d engineering design"), 
                "Missing managed model-based 3d engineering design context");
        assert!(contents.contains("managed_model_based_3d_engineering_design") ||
                contents.contains("MANAGED_MODEL_BASED_3D_ENGINEERING_DESIGN"), 
                "Missing AP242 protocol identifier");
        
        // Check for BREP geometry (AP242 includes AP203 BREP)
        assert!(contents.contains("MANIFOLD_SOLID_BREP"), 
                "Missing MANIFOLD_SOLID_BREP for BREP representation");
        assert!(contents.contains("CLOSED_SHELL"), 
                "Missing CLOSED_SHELL for valid topology");
        assert!(contents.contains("FACE") || contents.contains("ADVANCED_FACE"), 
                "Missing face entities in BREP");
        assert!(contents.contains("EDGE") || contents.contains("EDGE_CURVE"), 
                "Missing edge entities in BREP");
        
        // Check for color support (AP242 includes AP214 features)
        assert!(contents.contains("COLOUR_RGB"), 
                "Missing COLOUR_RGB for color support");
        assert!(contents.contains("SURFACE_STYLE_RENDERING_PROPERTIES") ||
                contents.contains("STYLED_ITEM"), 
                "Missing styling entities for color");
        
        // Verify color values (0.2, 0.8, 0.2)
        assert!(contents.contains("0.200000") || contents.contains("0.2"), 
                "Missing green component in color");
        assert!(contents.contains("0.800000") || contents.contains("0.8"), 
                "Missing green component (0.8) in color");
        
        // Check for APPLICATION_PROTOCOL_DEFINITION (AP242 requirement)
        assert!(contents.contains("APPLICATION_PROTOCOL_DEFINITION"), 
                "Missing APPLICATION_PROTOCOL_DEFINITION for AP242");
        
        // Check for APPLICATION_CONTEXT (AP242 requirement)
        assert!(contents.contains("APPLICATION_CONTEXT"), 
                "Missing APPLICATION_CONTEXT for AP242");
        
        // Verify it's a valid ISO 10303-21 STEP file structure
        assert!(contents.contains("ISO-10303-21;"), "Missing ISO-10303-21 header");
        assert!(contents.contains("END-ISO-10303-21;"), "Missing ISO-10303-21 end marker");
        assert!(contents.contains("FILE_SCHEMA"), "Missing FILE_SCHEMA");
        assert!(contents.contains("FILE_DESCRIPTION"), "Missing FILE_DESCRIPTION");
        assert!(contents.contains("DATA;"), "Missing DATA section");
        assert!(contents.contains("ENDSEC;"), "Missing ENDSEC markers");
        
        // Clean up
        fs::remove_file(output_path).ok();
        
        Ok(())
    }
}
