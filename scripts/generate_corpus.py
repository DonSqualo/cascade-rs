#!/usr/bin/env python3
"""
Golden corpus generator for cascade-rs

Uses OpenCASCADE (via python-occ or FreeCAD) to generate test cases.
Each test case includes:
- input.json: Input parameters
- expected.json: Expected output (BREP topology + mesh)
- metadata.json: OCCT version, tolerance, etc.

Usage:
    pip install numpy
    # Install FreeCAD or python-occ for OCCT bindings
    python scripts/generate_corpus.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Try to import OCCT bindings
try:
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeBox, 
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeTorus,
    )
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
    HAVE_OCC = True
except ImportError:
    HAVE_OCC = False
    print("Warning: python-occ not found. Install with: conda install -c conda-forge pythonocc-core")

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
TOLERANCE = 1e-6

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def shape_to_json(shape):
    """Convert OCCT shape to JSON-serializable dict"""
    if not HAVE_OCC:
        return {}
    
    # Count topology
    n_vertices = 0
    n_edges = 0
    n_faces = 0
    
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        n_vertices += 1
        exp.Next()
    
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        n_edges += 1
        exp.Next()
        
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n_faces += 1
        exp.Next()
    
    # Get volume and surface area
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    volume = props.Mass()
    center = props.CentreOfMass()
    
    surf_props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, surf_props)
    surface_area = surf_props.Mass()
    
    # Mesh for comparison
    mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh.Perform()
    
    # Extract mesh vertices and triangles
    # (simplified - real implementation would extract from mesh)
    
    return {
        "topology": {
            "n_vertices": n_vertices,
            "n_edges": n_edges,
            "n_faces": n_faces,
        },
        "properties": {
            "volume": volume,
            "surface_area": surface_area,
            "center_of_mass": [center.X(), center.Y(), center.Z()],
        }
    }

def generate_primitive_tests():
    """Generate test cases for primitive shapes"""
    if not HAVE_OCC:
        print("Skipping primitives (no OCC)")
        return
    
    primitives_dir = ensure_dir(CORPUS_DIR / "primitives")
    
    # Box tests
    for name, (dx, dy, dz) in [
        ("unit_cube", (1, 1, 1)),
        ("rectangular", (2, 3, 4)),
        ("thin_plate", (10, 10, 0.1)),
    ]:
        test_dir = ensure_dir(primitives_dir / "box" / name)
        
        shape = BRepPrimAPI_MakeBox(dx, dy, dz).Shape()
        
        with open(test_dir / "input.json", "w") as f:
            json.dump({"dx": dx, "dy": dy, "dz": dz}, f, indent=2)
        
        with open(test_dir / "expected.json", "w") as f:
            json.dump(shape_to_json(shape), f, indent=2)
        
        print(f"  Generated: primitives/box/{name}")
    
    # Sphere tests
    for name, radius in [("unit", 1.0), ("small", 0.1), ("large", 100.0)]:
        test_dir = ensure_dir(primitives_dir / "sphere" / name)
        
        shape = BRepPrimAPI_MakeSphere(radius).Shape()
        
        with open(test_dir / "input.json", "w") as f:
            json.dump({"radius": radius}, f, indent=2)
        
        with open(test_dir / "expected.json", "w") as f:
            json.dump(shape_to_json(shape), f, indent=2)
        
        print(f"  Generated: primitives/sphere/{name}")
    
    # Cylinder tests
    for name, (r, h) in [("unit", (1, 1)), ("tall", (0.5, 10)), ("wide", (5, 0.5))]:
        test_dir = ensure_dir(primitives_dir / "cylinder" / name)
        
        shape = BRepPrimAPI_MakeCylinder(r, h).Shape()
        
        with open(test_dir / "input.json", "w") as f:
            json.dump({"radius": r, "height": h}, f, indent=2)
        
        with open(test_dir / "expected.json", "w") as f:
            json.dump(shape_to_json(shape), f, indent=2)
        
        print(f"  Generated: primitives/cylinder/{name}")
    
    # Add cone, torus similarly...

def generate_boolean_tests():
    """Generate test cases for boolean operations"""
    if not HAVE_OCC:
        print("Skipping booleans (no OCC)")
        return
    
    bool_dir = ensure_dir(CORPUS_DIR / "booleans")
    
    # Box-box union
    test_dir = ensure_dir(bool_dir / "fuse" / "box_box_overlap")
    
    box1 = BRepPrimAPI_MakeBox(2, 2, 2).Shape()
    box2 = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    # TODO: translate box2
    
    fused = BRepAlgoAPI_Fuse(box1, box2).Shape()
    
    with open(test_dir / "input.json", "w") as f:
        json.dump({
            "shape1": {"type": "box", "dx": 2, "dy": 2, "dz": 2},
            "shape2": {"type": "box", "dx": 1, "dy": 1, "dz": 1, "translate": [1, 1, 1]},
        }, f, indent=2)
    
    with open(test_dir / "expected.json", "w") as f:
        json.dump(shape_to_json(fused), f, indent=2)
    
    print(f"  Generated: booleans/fuse/box_box_overlap")

def generate_metadata():
    """Generate corpus metadata"""
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "occt_available": HAVE_OCC,
        "tolerance": TOLERANCE,
        "version": "0.1.0",
    }
    
    if HAVE_OCC:
        from OCC.Core.Standard import Standard_Version
        metadata["occt_version"] = "7.x"  # Would extract actual version
    
    with open(CORPUS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    print("Generating cascade-rs test corpus...")
    print(f"Output directory: {CORPUS_DIR}")
    print(f"OCCT available: {HAVE_OCC}")
    print()
    
    ensure_dir(CORPUS_DIR)
    
    print("Generating primitive tests...")
    generate_primitive_tests()
    
    print("\nGenerating boolean tests...")
    generate_boolean_tests()
    
    print("\nGenerating metadata...")
    generate_metadata()
    
    print("\n✓ Corpus generation complete")

if __name__ == "__main__":
    main()
