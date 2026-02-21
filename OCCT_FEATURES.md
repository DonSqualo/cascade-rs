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
- [x] RectangularTrimmedSurface - `rectangular_trimmed()` method in SurfaceType enum
- [x] OffsetSurface - SurfaceType::OffsetSurface enum variant with offset_distance support
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
- [x] Fuzzy boolean (tolerance-based)

### 3.3 Sweeps & Extrusions
- [x] Prism (linear sweep)
- [x] Revol (rotational sweep)
- [x] Pipe (path sweep)
- [x] Evolved (complex sweep) - `evolved(profile: &Wire, spine: &Wire, options: &EvolveOptions) -> Result<Solid>` with scaling and rotation
- [x] Draft prism (tapered extrusion)

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
- [x] Offset surface - SurfaceType::OffsetSurface with basis_surface and offset_distance
- [x] Offset curve - OffsetCurve struct with new(), offset(), point_at(), tangent() methods  
- [x] Thick solid
- [x] Shell (hollow solid)

### 3.7 Draft & Taper
- [x] Draft angle - `add_draft(solid: &Solid, face: &Face, angle: f64, neutral_plane: &Face) -> Result<Solid>`
- [x] Taper - `taper(solid: &Solid, face_indices: &[usize], angle: f64, neutral_plane: &Face) -> Result<Solid>`

### 3.8 Feature Operations
- [x] Hole (simple) - `make_hole(solid: &Solid, center: [f64; 3], direction: [f64; 3], diameter: f64, depth: f64) -> Result<Solid>`
- [x] Hole (countersunk/counterbore) - `make_hole_countersunk(...)` and `make_hole_counterbore(...)`
- [x] Slot - `make_slot(solid: &Solid, path: &Wire, width: f64, depth: f64) -> Result<Solid>`
- [x] Rib - `make_rib(solid: &Solid, profile: &Wire, direction: [f64; 3], thickness: f64) -> Result<Solid>`
- [x] Groove - `make_groove(solid: &Solid, path: &Wire, profile: &Wire) -> Result<Solid>`
- [x] Linear pattern - `linear_pattern(shape: &Solid, direction: [f64;3], count: usize, spacing: f64) -> Result<Vec<Solid>>` and `linear_pattern_fused(...)` -> Result<Solid>`
- [x] Circular pattern - `circular_pattern(shape: &Solid, axis_point: [f64;3], axis_dir: [f64;3], count: usize, angle: f64) -> Result<Vec<Solid>>` and `circular_pattern_fused(...) -> Result<Solid>`

### 3.9 Local Operations
- [x] Split face - `split_face(face: &Face, splitting_curve: &Edge) -> Result<Vec<Face>>`
- [x] Split edge - `split_edge(edge: &Edge, parameter: f64) -> Result<(Edge, Edge)>` and `split_edge_at_point(edge: &Edge, point: [f64;3]) -> Result<(Edge, Edge)>`
- [x] Remove face - `remove_face(solid: &Solid, face_index: usize) -> Result<Solid>`
- [x] Replace face - `replace_face(solid: &Solid, face_index: usize, new_face: &Face) -> Result<Solid>`
- [x] Thicken face - `thicken_face(face: &Face, thickness: f64) -> Result<Solid>`

### 3.10 Healing & Repair
- [x] Sewing (connect faces) - sew_faces()
- [x] Fix shape - fix_shape()
- [x] Fix wire - clean_wire()
- [x] Fix face - fix_face()
- [x] Remove degenerate edges
- [x] Remove degenerate faces
- [x] Drop small edges (threshold-based) - `drop_small_edges(solid: &Solid, threshold: f64) -> Result<Solid>`

### 3.11 Geometric Algorithms
- [x] Curve-curve intersection (2D)
- [x] Curve-surface intersection - `intersect_curve_surface(curve: &CurveType, start: [f64;3], end: [f64;3], surface: &SurfaceType) -> Result<Vec<[f64;3]>>` - Analytical line-plane, line-sphere, line-cylinder, line-cone intersections; numerical marching for complex curves
- [x] Surface-surface intersection (plane-plane, plane-cylinder, cylinder-cylinder)
- [x] Point projection to curve - `project_point_to_curve(point: [f64;3], curve: &CurveType, start: [f64;3], end: [f64;3]) -> Result<(f64, [f64;3])>` - Handles line, arc, bezier, bspline, parabola, hyperbola, ellipse, trimmed, and offset curves
- [x] Point projection to surface
- [x] Curve projection to surface - `project_curve_to_surface(curve: &CurveType, start: [f64;3], end: [f64;3], surface: &SurfaceType) -> Result<CurveType>` - Samples curve, projects each point to surface, fits BSpline result with graceful error handling
- [x] Extrema (distance) calculation - `extrema_curve_curve(c1, s1, e1, c2, s2, e2)` and `extrema_point_solid(point, solid)`
- [x] Curve approximation - `approximate_curve(points: &[[f64;3]], degree: usize, tolerance: f64) -> Result<CurveType>` - Least squares BSpline fitting with uniform knot vector
- [x] Surface approximation - `approximate_surface(points: &[Vec<[f64;3]>], u_degree: usize, v_degree: usize, tolerance: f64) -> Result<SurfaceType>` - Least squares BSpline fitting in both U and V directions with adaptive control point count
- [x] Curve interpolation - `interpolate_curve(points: &[[f64;3]], degree: usize) -> Result<CurveType>` - Creates a BSpline curve passing exactly through all points using chord-length parameterization and tridiagonal system solver
- [x] Surface interpolation - `interpolate_surface(points: &[Vec<[f64;3]>], u_degree: usize, v_degree: usize) -> Result<SurfaceType>` - BSpline surface passing through 2D grid of points, interpolates rows first (u-direction) then columns (v-direction) with chord-length parameterization

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
- [x] STL (binary) - `write_stl_binary(mesh: &TriangleMesh, path: &str) -> Result<()>` with 80-byte header, little-endian format
- [x] OBJ
- [x] PLY - `write_ply(mesh: &TriangleMesh, path: &str) -> Result<()>` ASCII format with proper PLY header and triangle face indices

## Module 5: Data Exchange

### 5.1 STEP
- [x] STEP read (basic)
- [x] STEP write (basic)
- [ ] STEP AP203 full compliance
- [ ] STEP AP214 full compliance
- [ ] STEP AP242 full compliance
- [ ] STEP with colors/materials
- [ ] STEP with assemblies
- [x] STEP with PMI (annotations)

### 5.2 IGES
- [x] IGES read (basic geometry support: points, lines, circles, spheres, cylinders, cones, tori, planes, B-splines)
- [x] IGES write (basic geometry support: points, lines, circles, spheres, cylinders)
- [x] IGES with layers - `write_iges_with_layers()` with Directory Entry Level field (bytes 33-40) for layer assignment; layer numbers 0-65535
- [x] IGES with colors

### 5.3 Other Formats
- [x] BREP native format
- [x] glTF read - `read_gltf(path: &str) -> Result<TriangleMesh>` reads glTF 2.0 JSON with external binary buffer; `read_glb(path: &str) -> Result<TriangleMesh>` reads binary GLB format; extracts vertex positions, normals (if available), and triangle indices
- [x] glTF write
- [ ] VRML write
- [ ] DXF (2D)

### 5.4 XDE (Extended Data Exchange)
- [x] Color attributes - set_shape_color(), get_shape_color() functions storing [f64; 3] RGB
- [x] Layer attributes - set_shape_layer(), get_shape_layer() functions storing layer identifier
- [x] Material attributes - set_shape_material(), get_shape_material() functions storing material identifier
- [x] Name attributes - ShapeAttributes struct with name, color, layer, material fields; set_shape_name(), get_shape_name(), set_shape_attributes(), get_shape_attributes() functions
- [x] Assembly structure - Assembly struct with hierarchical organization; AssemblyNode enum supporting Part (Solid), SubAssembly (nested Assembly), and Instance (with reference and transform); create_assembly(), add_part(), add_subassembly(), add_instance(), flatten_assembly() functions

## Module 6: Queries & Analysis

### 6.1 Geometric Properties
- [x] Bounding box
- [x] Volume
- [x] Surface area
- [x] Center of mass
- [x] Moments of inertia
- [x] Principal axes
- [ ] Radius of gyration

### 6.2 Topological Queries
- [x] Point inside solid - `point_inside(solid: &Solid, point: [f64; 3]) -> Result<bool>` ray-casting algorithm
- [x] Shape classification - `classify_shape(solid: &Solid) -> Result<ShapeClass>` returns Convex/Concave/Mixed
- [x] Distance to shape - `distance(shape1: &Shape, shape2: &Shape) -> Result<f64>` Euclidean distance
- [x] Closest point on shape - Implemented via project_point_to_curve() and project_point_to_surface()

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

**Implemented:** 134 features  
**Remaining:** 32 features  
**Total:** 166 features  
**Completion:** 80.7%

**Priority Order:**
1. RectangularTrimmedSurface, OffsetSurface, PlateSurface
2. Full STEP/IGES compliance
3. Point/curve/surface projection operations
4. Half-space and CompSolid support
5. More complex intersection cases (BSpline curves/surfaces)

---

*Last updated: 2026-02-21 - Implemented Module 4.3 PLY mesh export (write_ply function in ASCII format with proper PLY header format). All tests passing (402 test suite).*
