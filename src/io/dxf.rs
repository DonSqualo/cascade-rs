//! DXF (AutoCAD Drawing Exchange Format) export for 2D geometry

use crate::brep::{Wire, CurveType};
use crate::Result;
use std::fs::File;
use std::io::Write;

/// Write wires to DXF ASCII format
///
/// Exports a collection of 2D wires to DXF format with proper header and ENTITIES section.
/// Each edge in the wires is converted to the appropriate DXF entity:
/// - LINE for linear edges
/// - ARC for circular edges
/// - SPLINE for BSpline curves
///
/// # Arguments
/// * `wires` - Array of wires to export
/// * `path` - Output file path
///
/// # Example
/// ```rust,no_run
/// use cascade::brep::Wire;
/// use cascade::io::write_dxf;
///
/// // Create a rectangle wire with 4 edges
/// let wire = create_rectangle_wire();
/// let wires = vec![wire];
/// write_dxf(&wires, "output.dxf").unwrap();
/// ```
pub fn write_dxf(wires: &[Wire], path: &str) -> Result<()> {
    let mut file = File::create(path)?;

    // Write DXF header
    write_dxf_header(&mut file)?;

    // Write ENTITIES section
    write_dxf_entities(&mut file, wires)?;

    // Write EOF
    writeln!(file, "  0")?;
    writeln!(file, "ENDSEC")?;
    writeln!(file, "  0")?;
    writeln!(file, "EOF")?;

    Ok(())
}

/// Write DXF header with required sections
fn write_dxf_header(file: &mut File) -> Result<()> {
    // HEADER section
    writeln!(file, "  0")?;
    writeln!(file, "SECTION")?;
    writeln!(file, "  2")?;
    writeln!(file, "HEADER")?;
    writeln!(file, "  9")?;
    writeln!(file, "$ACADVER")?;
    writeln!(file, "  1")?;
    writeln!(file, "AC1021")?; // DXF R2007 version
    writeln!(file, "  0")?;
    writeln!(file, "ENDSEC")?;

    // CLASSES section (minimal)
    writeln!(file, "  0")?;
    writeln!(file, "SECTION")?;
    writeln!(file, "  2")?;
    writeln!(file, "CLASSES")?;
    writeln!(file, "  0")?;
    writeln!(file, "ENDSEC")?;

    // TABLES section (minimal)
    writeln!(file, "  0")?;
    writeln!(file, "SECTION")?;
    writeln!(file, "  2")?;
    writeln!(file, "TABLES")?;

    // LAYER table
    writeln!(file, "  0")?;
    writeln!(file, "TABLE")?;
    writeln!(file, "  2")?;
    writeln!(file, "LAYER")?;
    writeln!(file, " 70")?;
    writeln!(file, "1")?;
    writeln!(file, "  0")?;
    writeln!(file, "LAYER")?;
    writeln!(file, "  2")?;
    writeln!(file, "0")?;
    writeln!(file, " 70")?;
    writeln!(file, "0")?;
    writeln!(file, " 62")?;
    writeln!(file, "7")?;
    writeln!(file, "  6")?;
    writeln!(file, "CONTINUOUS")?;
    writeln!(file, "  0")?;
    writeln!(file, "ENDTAB")?;

    writeln!(file, "  0")?;
    writeln!(file, "ENDSEC")?;

    // BLOCKS section (minimal)
    writeln!(file, "  0")?;
    writeln!(file, "SECTION")?;
    writeln!(file, "  2")?;
    writeln!(file, "BLOCKS")?;
    writeln!(file, "  0")?;
    writeln!(file, "ENDSEC")?;

    Ok(())
}

/// Write ENTITIES section with wire geometry
fn write_dxf_entities(file: &mut File, wires: &[Wire]) -> Result<()> {
    writeln!(file, "  0")?;
    writeln!(file, "SECTION")?;
    writeln!(file, "  2")?;
    writeln!(file, "ENTITIES")?;

    // Write each edge as a DXF entity
    for wire in wires {
        for edge in &wire.edges {
            match &edge.curve_type {
                CurveType::Line => {
                    write_dxf_line(file, edge.start.point, edge.end.point)?;
                }
                CurveType::Arc { center, radius } => {
                    write_dxf_arc(file, edge.start.point, edge.end.point, *center, *radius)?;
                }
                CurveType::BSpline {
                    control_points, ..
                } => {
                    write_dxf_spline(file, control_points)?;
                }
                CurveType::Bezier { control_points } => {
                    write_dxf_spline(file, control_points)?;
                }
                _ => {
                    // For other curve types, approximate with a line for now
                    write_dxf_line(file, edge.start.point, edge.end.point)?;
                }
            }
        }
    }

    Ok(())
}

/// Write a LINE entity
fn write_dxf_line(file: &mut File, start: [f64; 3], end: [f64; 3]) -> Result<()> {
    writeln!(file, "  0")?;
    writeln!(file, "LINE")?;
    writeln!(file, "  8")?;
    writeln!(file, "0")?; // Layer
    writeln!(file, " 10")?;
    writeln!(file, "{:.6}", start[0])?;
    writeln!(file, " 20")?;
    writeln!(file, "{:.6}", start[1])?;
    writeln!(file, " 30")?;
    writeln!(file, "{:.6}", start[2])?;
    writeln!(file, " 11")?;
    writeln!(file, "{:.6}", end[0])?;
    writeln!(file, " 21")?;
    writeln!(file, "{:.6}", end[1])?;
    writeln!(file, " 31")?;
    writeln!(file, "{:.6}", end[2])?;
    Ok(())
}

/// Write an ARC entity
fn write_dxf_arc(
    file: &mut File,
    start: [f64; 3],
    end: [f64; 3],
    center: [f64; 3],
    radius: f64,
) -> Result<()> {
    // Calculate start and end angles
    let start_angle = calculate_angle(center, start);
    let end_angle = calculate_angle(center, end);

    writeln!(file, "  0")?;
    writeln!(file, "ARC")?;
    writeln!(file, "  8")?;
    writeln!(file, "0")?; // Layer
    writeln!(file, " 10")?;
    writeln!(file, "{:.6}", center[0])?;
    writeln!(file, " 20")?;
    writeln!(file, "{:.6}", center[1])?;
    writeln!(file, " 30")?;
    writeln!(file, "{:.6}", center[2])?;
    writeln!(file, " 40")?;
    writeln!(file, "{:.6}", radius)?;
    writeln!(file, " 50")?;
    writeln!(file, "{:.6}", start_angle)?;
    writeln!(file, " 51")?;
    writeln!(file, "{:.6}", end_angle)?;
    Ok(())
}

/// Write a SPLINE entity
fn write_dxf_spline(file: &mut File, control_points: &[[f64; 3]]) -> Result<()> {
    if control_points.is_empty() {
        return Ok(());
    }

    writeln!(file, "  0")?;
    writeln!(file, "SPLINE")?;
    writeln!(file, "  8")?;
    writeln!(file, "0")?; // Layer
    writeln!(file, " 70")?;
    writeln!(file, "8")?; // Spline type: cubic B-spline
    writeln!(file, " 71")?;
    writeln!(file, "3")?; // Degree
    writeln!(file, " 72")?;
    writeln!(file, "{}", control_points.len())?; // Number of knots (will be computed)
    writeln!(file, " 73")?;
    writeln!(file, "{}", control_points.len())?; // Number of control points

    // Write control points
    for (i, point) in control_points.iter().enumerate() {
        writeln!(file, " 1{}", if i == 0 { "0" } else { "1" })?; // 10 or 11
        writeln!(file, "{:.6}", point[0])?;
        writeln!(file, " 2{}", if i == 0 { "0" } else { "1" })?; // 20 or 21
        writeln!(file, "{:.6}", point[1])?;
        writeln!(file, " 3{}", if i == 0 { "0" } else { "1" })?; // 30 or 31
        writeln!(file, "{:.6}", point[2])?;
    }

    Ok(())
}

/// Calculate angle from center to point in degrees
fn calculate_angle(center: [f64; 3], point: [f64; 3]) -> f64 {
    let dx = point[0] - center[0];
    let dy = point[1] - center[1];
    let angle_rad = dy.atan2(dx);
    let angle_deg = angle_rad.to_degrees();
    if angle_deg < 0.0 {
        angle_deg + 360.0
    } else {
        angle_deg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::{Vertex, Edge};
    use std::path::Path;

    #[test]
    fn test_write_dxf_rectangle_wire() {
        // Create a rectangle wire with 4 LINE entities
        let edges = vec![
            Edge {
                start: Vertex {
                    point: [0.0, 0.0, 0.0],
                },
                end: Vertex {
                    point: [10.0, 0.0, 0.0],
                },
                curve_type: CurveType::Line,
            },
            Edge {
                start: Vertex {
                    point: [10.0, 0.0, 0.0],
                },
                end: Vertex {
                    point: [10.0, 5.0, 0.0],
                },
                curve_type: CurveType::Line,
            },
            Edge {
                start: Vertex {
                    point: [10.0, 5.0, 0.0],
                },
                end: Vertex {
                    point: [0.0, 5.0, 0.0],
                },
                curve_type: CurveType::Line,
            },
            Edge {
                start: Vertex {
                    point: [0.0, 5.0, 0.0],
                },
                end: Vertex {
                    point: [0.0, 0.0, 0.0],
                },
                curve_type: CurveType::Line,
            },
        ];

        let wire = Wire {
            edges,
            closed: true,
        };

        let wires = vec![wire];
        let output_path = "/tmp/test_dxf_rectangle.dxf";

        write_dxf(&wires, output_path).expect("Failed to write DXF");

        // Verify file exists
        assert!(
            Path::new(output_path).exists(),
            "DXF file should be created"
        );

        // Check file content
        let content = std::fs::read_to_string(output_path).expect("Failed to read DXF");

        // Check header
        assert!(content.contains("SECTION"), "DXF should have SECTION");
        assert!(content.contains("HEADER"), "DXF should have HEADER section");

        // Check entities
        assert!(content.contains("ENTITIES"), "DXF should have ENTITIES section");

        // Check for 4 LINE entities (count lines that match the pattern for entity declarations)
        let lines: Vec<&str> = content.lines().collect();
        let mut line_count = 0;
        for i in 0..lines.len() - 1 {
            if lines[i] == "  0" && lines[i + 1] == "LINE" {
                line_count += 1;
            }
        }
        assert_eq!(line_count, 4, "DXF should contain 4 LINE entities for rectangle");

        // Check for EOF
        assert!(content.contains("EOF"), "DXF should have EOF marker");
    }

    #[test]
    fn test_write_dxf_with_arc() {
        let edges = vec![
            Edge {
                start: Vertex {
                    point: [0.0, 0.0, 0.0],
                },
                end: Vertex {
                    point: [5.0, 0.0, 0.0],
                },
                curve_type: CurveType::Arc {
                    center: [0.0, 0.0, 0.0],
                    radius: 5.0,
                },
            },
        ];

        let wire = Wire {
            edges,
            closed: false,
        };

        let wires = vec![wire];
        let output_path = "/tmp/test_dxf_arc.dxf";

        write_dxf(&wires, output_path).expect("Failed to write DXF");

        let content = std::fs::read_to_string(output_path).expect("Failed to read DXF");

        // Check for ARC entity
        assert!(content.contains("ARC"), "DXF should contain ARC entity");

        // Check arc properties
        assert!(content.contains("0.000000")); // Center X
    }

    #[test]
    fn test_dxf_file_has_proper_structure() {
        let edges = vec![Edge {
            start: Vertex {
                point: [0.0, 0.0, 0.0],
            },
            end: Vertex {
                point: [1.0, 1.0, 0.0],
            },
            curve_type: CurveType::Line,
        }];

        let wire = Wire {
            edges,
            closed: false,
        };

        let wires = vec![wire];
        let output_path = "/tmp/test_dxf_structure.dxf";

        write_dxf(&wires, output_path).expect("Failed to write DXF");

        let content = std::fs::read_to_string(output_path).expect("Failed to read DXF");
        let lines: Vec<&str> = content.lines().collect();

        // Check sections appear in order
        let mut header_idx = 0;
        let mut entities_idx = 0;
        let mut eof_idx = 0;

        for (i, line) in lines.iter().enumerate() {
            if *line == "HEADER" {
                header_idx = i;
            }
            if *line == "ENTITIES" {
                entities_idx = i;
            }
            if *line == "EOF" {
                eof_idx = i;
            }
        }

        assert!(
            header_idx < entities_idx,
            "HEADER should appear before ENTITIES"
        );
        assert!(
            entities_idx < eof_idx,
            "ENTITIES should appear before EOF"
        );
    }

    #[test]
    fn test_write_dxf_empty_wire() {
        let wires: Vec<Wire> = vec![];
        let output_path = "/tmp/test_dxf_empty.dxf";

        write_dxf(&wires, output_path).expect("Failed to write empty DXF");

        let content = std::fs::read_to_string(output_path).expect("Failed to read DXF");

        // Should still have proper structure
        assert!(content.contains("HEADER"), "Empty DXF should have HEADER");
        assert!(content.contains("ENTITIES"), "Empty DXF should have ENTITIES");
        assert!(content.contains("EOF"), "Empty DXF should have EOF");
    }
}
