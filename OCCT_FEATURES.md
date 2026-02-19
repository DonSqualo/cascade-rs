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
- [ ] gp_Pnt2d (2D point)
- [ ] gp_Vec2d (2D vector)
- [ ] gp_Dir (unit vector)
- [ ] gp_XYZ (coordinates)

### 2.2 Geometry - Curves
- [x] Line
- [x] Circle
- [x] Ellipse
- [ ] Parabola
- [ ] Hyperbola
- [x] BezierCurve
- [x] BSplineCurve
- [ ] OffsetCurve
- [ ] TrimmedCurve

### 2.3 Geometry - Surfaces
- [x] Plane
- [x] CylindricalSurface
- [x] SphericalSurface
- [x] ConicalSurface
- [x] ToroidalSurface
- [ ] BezierSurface
- [x] BSplineSurface
- [ ] RectangularTrimmedSurface
- [ ] OffsetSurface
- [ ] SurfaceOfRevolution
- [ ] SurfaceOfLinearExtrusion
- [ ] PlateSurface

### 2.4 Topology
- [x] Vertex
- [x] Edge
- [x] Wire
- [x] Face
- [x] Shell
- [x] Solid
- [x] Compound
- [ ] CompSolid

### 2.5 Topological Operations
- [x] Adjacent faces query
- [x] Connected edges query
- [x] Shared edge query
- [x] Face neighbors map
- [ ] Explorer (iterate sub-shapes)
- [ ] Builder (construct topology)
- [ ] Modifier (edit topology)

## Module 3: Modeling Algorithms

### 3.1 Primitives
- [x] Box/Cuboid
- [x] Sphere
- [x] Cylinder
- [x] Cone
- [x] Torus
- [x] Wedge/Prism
- [ ] Half-space (infinite solid)

### 3.2 Boolean Operations
- [x] Fuse (union)
- [x] Cut (difference)
- [x] Common (intersection)
- [x] Section (solid/plane)
- [ ] Splitter
- [ ] Boolean with multiple arguments
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
- [ ] Fillet (variable radius)
- [x] Chamfer (planar bevel)
- [ ] Blend (rolling ball)

### 3.6 Offset Operations
- [ ] Offset surface
- [ ] Offset curve
- [ ] Thick solid
- [x] Shell (hollow solid)

### 3.7 Draft & Taper
- [ ] Draft angle
- [ ] Taper

### 3.8 Feature Operations
- [ ] Hole (simple)
- [ ] Hole (countersunk/counterbore)
- [ ] Slot
- [ ] Rib
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
- [ ] Sewing (connect faces)
- [ ] Fix shape
- [ ] Fix wire
- [ ] Fix face
- [ ] Remove small edges
- [ ] Remove small faces
- [ ] Drop small edges

### 3.11 Geometric Algorithms
- [ ] Curve-curve intersection (2D)
- [ ] Curve-surface intersection
- [ ] Surface-surface intersection
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
- [ ] IGES read
- [ ] IGES write
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
- [ ] Check shape validity
- [ ] Check watertight
- [ ] Check self-intersection
- [ ] Check degeneracies

## Module 7: Visualization (Lower Priority)
- [ ] 3D viewer
- [ ] Shaded display
- [ ] Wireframe display
- [ ] Selection
- [ ] Highlighting

---

## Current Progress

**Implemented:** ~31 features (chamfer added)
**Total OCCT Features:** ~200+ core features

**Priority Order:**
1. Sweeps & Extrusions (prism, revol, pipe)
2. Filleting & Chamfering
3. More curve/surface types (BSpline)
4. Shape healing
5. Full STEP compliance
6. IGES support

---

*Last updated: 2026-02-19 - Added Ellipse, Bezier, and BSpline curve types*
