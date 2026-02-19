# OpenCASCADE Feature Parity Tracker

**Goal:** 100% feature parity with OpenCASCADE Technology (OCCT)

This is the REAL feature list. OCCT has 7 major modules with hundreds of features.

## Module 1: Foundation Classes
- [ ] Primitive types (Boolean, Integer, Real, String)
- [ ] Collection classes (arrays, lists, maps, sets)
- [ ] Math (vectors, matrices, linear algebra)
- [ ] Memory management (smart pointers)
- [ ] RTTI (runtime type info)

## Module 2: Modeling Data (BREP)

### 2.1 Geometry - Points & Vectors
- [x] gp_Pnt (3D point)
- [x] gp_Vec (3D vector)
- [x] gp_Pnt2d (2D point)
- [x] gp_Vec2d (2D vector)
- [x] gp_Dir (unit vector)
- [x] gp_XYZ (coordinates)

### 2.2 Geometry - Curves
- [x] Line
- [x] Circle
- [x] Ellipse
- [x] Parabola
- [x] Hyperbola
- [x] BezierCurve
- [x] BSplineCurve
- [x] OffsetCurve
- [x] TrimmedCurve

### 2.3 Geometry - Surfaces
- [x] Plane
- [x] CylindricalSurface
- [x] SphericalSurface
- [x] ConicalSurface
- [x] ToroidalSurface
- [x] BezierSurface
- [x] BSplineSurface
- [x] RectangularTrimmedSurface
- [x] OffsetSurface
- [x] SurfaceOfRevolution
- [x] SurfaceOfLinearExtrusion
- [ ] PlateSurface

### 2.4 Topology
- [x] Vertex
- [x] Edge
- [x] Wire
- [x] Face
- [x] Shell
- [x] Solid
- [x] Compound
- [x] CompSolid

### 2.5 Topological Operations
- [x] Adjacent faces query
- [x] Connected edges query
- [x] Shared edge query
- [x] Face neighbors map
- [x] Explorer (iterate sub-shapes)
- [x] Builder (construct topology)
- [x] Modifier (edit topology)

## Module 3: Modeling Algorithms

### 3.1 Primitives
- [x] Box/Cuboid
- [x] Sphere
- [x] Cylinder
- [x] Cone
- [x] Torus
- [x] Wedge/Prism
- [x] Half-space (infinite solid)

### 3.2 Boolean Operations
- [x] Fuse (union)
- [x] Cut (difference)
- [x] Common (intersection)
- [x] Section (solid/plane)
- [x] Splitter
- [x] Boolean with multiple arguments
- [ ] Fuzzy boolean (tolerance-based)

### 3.3 Sweeps & Extrusions
- [x] Prism (linear sweep)
- [x] Revol (rotational sweep)
- [x] Pipe (path sweep)
- [ ] Evolved (complex sweep)
- [ ] Draft prism (tapered extrusion)

### 3.4 Lofting & Skinning
- [x] ThruSections (loft) - `make_loft(profiles: &[Wire], ruled: bool) -> Result<Solid>`
- [x] Ruled surface - supported via `ruled=true` parameter
- [x] Skinning - smooth BSpline surfaces via `ruled=false` parameter

### 3.5 Filleting & Chamfering
- [x] Fillet (constant radius)
- [x] Fillet (variable radius)
- [x] Chamfer (planar bevel)
- [ ] Blend (rolling ball)

### 3.6 Offset Operations
- [ ] Offset surface
- [ ] Offset curve
- [x] Thick solid
- [x] Shell (hollow solid)

### 3.7 Draft & Taper
- [x] Draft angle - `add_draft(solid: &Solid, face: &Face, angle: f64, neutral_plane: &Face) -> Result<Solid>`
- [ ] Taper

### 3.8 Feature Operations
- [x] Hole (simple) - `make_hole(solid: &Solid, center: [f64; 3], direction: [f64; 3], diameter: f64, depth: f64) -> Result<Solid>`
- [ ] Hole (countersunk/counterbore)
- [x] Slot - `make_slot(solid: &Solid, path: &Wire, width: f64, depth: f64) -> Result<Solid>`
- [x] Rib - `make_rib(solid: &Solid, profile: &Wire, direction: [f64; 3], thickness: f64) -> Result<Solid>`
- [ ] Groove
- [ ] Linear pattern
- [ ] Circular pattern

### 3.9 Local Operations
- [ ] Split face
- [ ] Split edge
- [ ] Remove face
- [ ] Replace face
- [ ] Thicken face

### 3.10 Healing & Repair
- [x] Sewing (connect faces) - sew_faces()
- [x] Fix shape - fix_shape()
- [x] Fix wire - clean_wire()
- [x] Fix face - fix_face()
- [x] Remove degenerate edges
- [x] Remove degenerate faces
- [ ] Drop small edges (threshold-based)

### 3.11 Geometric Algorithms
- [ ] Curve-curve intersection (2D)
- [ ] Curve-surface intersection
- [x] Surface-surface intersection (plane-plane, plane-cylinder, cylinder-cylinder)
- [ ] Point projection to curve
- [ ] Point projection to surface
- [ ] Curve projection to surface
- [ ] Extrema (distance) calculation
- [ ] Curve approximation
- [ ] Surface approximation
- [ ] Curve interpolation
- [ ] Surface interpolation

### 3.12 Constraints-based Construction
- [ ] Circle tangent to 3 elements
- [ ] Circle tangent to 2 elements + radius
- [ ] Line tangent to 2 elements
- [ ] Bisector curves

## Module 4: Mesh

### 4.1 Tessellation
- [x] Triangulate solid
- [ ] Incremental mesh
- [ ] Mesh with deflection control
- [ ] Mesh with angle control

### 4.2 Mesh Data
- [x] Triangle mesh structure
- [ ] Polygon mesh
- [ ] Mesh domain

### 4.3 Mesh Export
- [x] STL (ASCII)
- [ ] STL (binary)
- [ ] OBJ
- [ ] PLY

## Module 5: Data Exchange

### 5.1 STEP
- [x] STEP read (basic)
- [x] STEP write (basic)
- [ ] STEP AP203 full compliance
- [ ] STEP AP214 full compliance
- [ ] STEP AP242 full compliance
- [ ] STEP with colors/materials
- [ ] STEP with assemblies
- [ ] STEP with PMI (annotations)

### 5.2 IGES
- [x] IGES read (basic geometry support: points, lines, circles, spheres, cylinders, cones, tori, planes, B-splines)
- [x] IGES write (basic geometry support: points, lines, circles, spheres, cylinders)
- [ ] IGES with layers
- [ ] IGES with colors

### 5.3 Other Formats
- [ ] BREP native format
- [ ] glTF read
- [ ] glTF write
- [ ] VRML write
- [ ] DXF (2D)

### 5.4 XDE (Extended Data Exchange)
- [ ] Color attributes
- [ ] Layer attributes
- [ ] Material attributes
- [ ] Name attributes
- [ ] Assembly structure

## Module 6: Queries & Analysis

### 6.1 Geometric Properties
- [x] Bounding box
- [x] Volume
- [x] Surface area
- [x] Center of mass
- [ ] Moments of inertia
- [ ] Principal axes
- [ ] Radius of gyration

### 6.2 Topological Queries
- [ ] Point inside solid
- [ ] Shape classification
- [ ] Distance to shape
- [ ] Closest point on shape

### 6.3 Validity Checks
- [x] Check shape validity (check_valid)
- [x] Check watertight (check_watertight)
- [x] Check self-intersection (check_self_intersection)
- [x] Check degeneracies (degenerate edges/faces detection)

## Module 7: Visualization (Lower Priority)
- [ ] 3D viewer
- [ ] Shaded display
- [ ] Wireframe display
- [ ] Selection
- [ ] Highlighting

---

## Current Progress

**Implemented:** 83 features  
**Remaining:** 83 features  
**Total:** 166 features  
**Completion:** 50%

**Priority Order:**
1. Curve-surface intersection (for boolean operations)
2. RectangularTrimmedSurface, OffsetSurface, PlateSurface
3. Full STEP/IGES compliance
4. Point/curve/surface projection operations
5. Half-space and CompSolid support

---

*Last updated: 2026-02-24 - Implemented Half-space primitive (infinite solid)*
