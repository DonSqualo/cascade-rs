# CASCADE-RS: OCCT Line-by-Line Port

**Method:** Read C++ source + GTests as baseline → Write comprehensive Rust tests → Implement
**Source:** /home/heim/projects/occt-source
**Tracking:** This file is the memory system. Sub-agents update here.

---

## Current Status

**965 tests passing** (5 pre-existing failures in construct)

---

## Layer 0: Foundation ✅ COMPLETE

### precision ✅
- Tests: 2/2

### gp ✅ COMPLETE (269 tests)
- 3D Core: XYZ, Pnt, Vec, Dir, Mat, Ax1, Ax2, Ax3, Trsf, GTrsf
- 2D Core: XY, Pnt2d, Vec2d, Dir2d, Mat2d, Ax2d, Ax22d, Trsf2d
- 2D Curves: Lin2d, Circ2d, Elips2d, Hypr2d, Parab2d
- 3D Geometry: Lin, Pln, Circ, Elips, Hypr, Parab, Cylinder, Cone, Sphere, Torus

### bnd ✅ COMPLETE (9 tests)
- BndBox, BndBox2d, BndSphere, BndOBB, BndRange
- BndB2d, BndB2f, BndB3d, BndB3f

---

## Layer 2: Math & Location 🔴 NOT STARTED

### TopLoc (Transformations)
- Source: `TKMath/TopLoc/`
- Classes: Location, Datum3D, ItemLocation
- Purpose: Composite transformations for positioning shapes
- GTests: None dedicated, tested via integration

### math (Numerical methods)
- Source: `TKMath/math/`
- Classes: 54 headers - vectors, matrices, solvers, minimizers
- GTests: ~40 test files
- Large package - consider sub-agents

### ElCLib (Elementary curves)
- Source: `TKMath/ElCLib/`
- Purpose: Parametric curve utilities (Lin, Circ, Elips, etc.)
- GTests: ElCLib_Test.cxx

### ElSLib (Elementary surfaces)
- Source: `TKMath/ElSLib/`
- Purpose: Parametric surface utilities (Cylinder, Cone, Sphere, etc.)

---

## Layer 3: Geometry 🔴 NOT STARTED

Located in: `src/ModelingData/TKG3d/`
- Geom_* (3D curves and surfaces)
- Geom2d_* (2D curves)

---

## Commands

```bash
cd /home/heim/projects/cascade-rs
cargo test --lib              # 965 tests
cargo test --lib -- gp        # gp tests
cargo test --lib -- bnd       # bnd tests
cargo check                   # Check compilation
```
