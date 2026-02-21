# CASCADE-RS: OCCT Line-by-Line Port

**Method:** Read C++ source + GTests as baseline → Write comprehensive Rust tests → Implement
**Source:** /home/heim/projects/occt-source
**Tracking:** This file is the memory system. Sub-agents update here.

---

## Current Status

**170 gp tests passing** (2D types integrated)

### What's Done
- precision: Complete (2 tests)
- gp 3D core: Complete (XYZ, Pnt, Vec, Dir, Mat, Ax1, Ax2, Ax3, Trsf, GTrsf)
- gp 2D core: Complete (XY, Pnt2d, Vec2d, Dir2d, Mat2d, Ax2d, Ax22d, Trsf2d, Lin2d, Circ2d, Elips2d, Hypr2d, Parab2d)

### What's Written But Needs Integration
- gp 3D geometry: Pln, Lin, Circ, Elips, Hypr, Parab, Cylinder, Cone, Sphere, Torus
- bnd package: BndBox, BndBox2d, BndSphere, BndOBB, BndRange (268 tests extracted)

---

## Layer 0: Foundation

### precision ✅ COMPLETE
- Tests: 2/2

### gp - 3D Core ✅ COMPLETE
| Type | Status | Tests |
|------|--------|-------|
| XYZ | ✅ | 28 |
| Pnt | ✅ | 5 |
| Vec | ✅ | 4 |
| Dir | ✅ | 7 |
| Mat | ✅ | 3 |
| Ax1 | ✅ | 3 |
| Ax2 | ✅ | 2 |
| Ax3 | ✅ | 3 |
| Trsf | ✅ | 5 |
| GTrsf | ✅ | 10 |

### gp - 2D Core ✅ COMPLETE
| Type | Status | Tests |
|------|--------|-------|
| XY | ✅ | 30 |
| Pnt2d | ✅ | 9 |
| Vec2d | ✅ | 14 |
| Dir2d | ✅ | 5 |
| Mat2d | ✅ | 10 |
| Ax2d | ✅ | 3 |
| Ax22d | ✅ | 3 |
| Trsf2d | ✅ | 5 |
| Lin2d | ✅ | 5 |
| Circ2d | ✅ | 5 |
| Elips2d | ✅ | 4 |
| Hypr2d | ✅ | 3 |
| Parab2d | ✅ | 3 |

### gp - 3D Geometry 🟡 CODE EXISTS (needs integration)
| Type | Status | Notes |
|------|--------|-------|
| Pln | 🟡 | API fixes needed |
| Lin | 🟡 | API fixes needed |
| Circ | 🟡 | API fixes needed |
| Elips | 🟡 | API fixes needed |
| Hypr | 🟡 | API fixes needed |
| Parab | 🟡 | API fixes needed |
| Cylinder | 🟡 | API fixes needed |
| Cone | 🟡 | API fixes needed |
| Sphere | 🟡 | API fixes needed |
| Torus | 🟡 | API fixes needed |

---

## Layer 1: Math & Bounds

### Bnd 🟡 CODE EXISTS (needs integration)
| Class | Status | Notes |
|-------|--------|-------|
| BndBox | 🟡 | 59 API errors |
| BndBox2d | 🟡 | Needs Pnt2d fixes |
| BndSphere | 🟡 | Needs Pnt fixes |
| BndOBB | 🟡 | Needs Dir fixes |
| BndRange | 🟡 | Should be simple |

---

## Next Steps

1. **Quick Win:** Integrate BndRange (simple 1D range, minimal deps)
2. **Then:** Fix BndBox API issues (Pnt constructor, Dir methods)
3. **Then:** Integrate 3D geometry types (Pln, Lin, etc.)

---

## Scripts

```bash
# Run gp tests
cargo test --lib -- gp

# Check compilation
cargo check

# Extract tests for a package
python3 scripts/extract_tests.py <PackageName>
```
