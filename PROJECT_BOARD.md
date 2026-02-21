# CASCADE-RS: OCCT Line-by-Line Port

**Method:** Read C++ source + GTests as baseline → Write comprehensive Rust tests → Implement
**Source:** /home/heim/projects/occt-source
**Tracking:** This file is the memory system. Sub-agents update here.

---

## Active Sub-Agents

| Agent | Package | Started | Status | Tests |
|-------|---------|---------|--------|-------|
| port-gp-3d-geom | Pln,Lin,Circ,Cylinder,etc | 2025-02-21 | 🟡 Structure Complete | WIP API fixes |
| port-bnd | Bnd_Box,Bnd_Sphere,etc | 10:32 UTC | 🟡 Running | - |
| port-gp-2d | XY,Pnt2d,Vec2d,etc | 10:32 UTC | 🟡 Running | - |

---

## Layer 0: Foundation (REQUIRED FIRST)

### precision ✅ COMPLETE
- Source: `TKernel/Precision/`
- Tests: 2/2
- All constants ported

### gp ✅ CORE COMPLETE (61 tests)
Remaining types to port:

| Type | Source | Status | Tests |
|------|--------|--------|-------|
| XYZ | gp_XYZ.hxx | ✅ | 28 |
| Pnt | gp_Pnt.hxx | ✅ | 5 |
| Vec | gp_Vec.hxx | ✅ | 4 |
| Dir | gp_Dir.hxx | ✅ | 7 |
| Mat | gp_Mat.hxx | ✅ | 3 |
| Ax1 | gp_Ax1.hxx | ✅ | 3 |
| Ax2 | gp_Ax2.hxx | ✅ | 2 |
| Ax3 | gp_Ax3.hxx | ✅ | 3 |
| Trsf | gp_Trsf.hxx | ✅ | 5 |
| Pln | gp_Pln.hxx | 🟡 | 16 |
| Lin | gp_Lin.hxx | 🟡 | 13 |
| Circ | gp_Circ.hxx | 🟡 | 11 |
| Elips | gp_Elips.hxx | 🟡 | 8 |
| Hypr | gp_Hypr.hxx | 🟡 | 6 |
| Parab | gp_Parab.hxx | 🟡 | 5 |
| Cylinder | gp_Cylinder.hxx | 🟡 | 6 |
| Cone | gp_Cone.hxx | 🟡 | 5 |
| Sphere | gp_Sphere.hxx | 🟡 | 8 |
| Torus | gp_Torus.hxx | 🟡 | 6 |
| GTrsf | gp_GTrsf.hxx | 🔴 | 0 |
| Pnt2d | gp_Pnt2d.hxx | 🔴 | 0 |
| Vec2d | gp_Vec2d.hxx | 🔴 | 0 |
| Dir2d | gp_Dir2d.hxx | 🔴 | 0 |
| Mat2d | gp_Mat2d.hxx | 🔴 | 0 |
| Trsf2d | gp_Trsf2d.hxx | 🔴 | 0 |
| Ax2d | gp_Ax2d.hxx | 🔴 | 0 |
| Ax22d | gp_Ax22d.hxx | 🔴 | 0 |
| Lin2d | gp_Lin2d.hxx | 🔴 | 0 |
| Circ2d | gp_Circ2d.hxx | 🔴 | 0 |
| Elips2d | gp_Elips2d.hxx | 🔴 | 0 |
| Hypr2d | gp_Hypr2d.hxx | 🔴 | 0 |
| Parab2d | gp_Parab2d.hxx | 🔴 | 0 |

---

## Layer 1: Math & Bounds

### Bnd (Bounding Boxes)
- Source: `TKMath/Bnd/`
- GTests: 268 tests in 8 files
- Status: 🔴 NOT STARTED

| Class | Lines | Status |
|-------|-------|--------|
| Bnd_Box | ~800 | 🔴 |
| Bnd_Box2d | ~400 | 🔴 |
| Bnd_Sphere | ~200 | 🔴 |
| Bnd_OBB | ~600 | 🔴 |
| Bnd_Range | ~150 | 🔴 |
| Bnd_B2f/B2d | ~200 | 🔴 |
| Bnd_B3f/B3d | ~200 | 🔴 |

### math (Numerical Algorithms)
- Source: `TKMath/math/`
- ~30,000 lines
- Status: 🔴 NOT STARTED

### BSplCLib (B-Spline Curves)
- Source: `TKMath/BSplCLib/`
- ~15,000 lines
- Status: 🔴 NOT STARTED

### BSplSLib (B-Spline Surfaces)
- Source: `TKMath/BSplSLib/`
- ~10,000 lines
- Status: 🔴 NOT STARTED

---

## Sub-Agent Task Template

```
## Task: Port OCCT package <PACKAGE>

### Setup
cd /home/heim/projects/cascade-rs
git checkout -b port/<package>

### Sources
OCCT: /home/heim/projects/occt-source/src/.../<PACKAGE>/
GTests: /home/heim/projects/occt-source/src/.../GTests/<PACKAGE>*_Test.cxx

### Method
1. Read ALL .hxx files - document every method
2. Read GTests as baseline behavior specs
3. Create src/<package>/mod.rs
4. Write Rust tests for EVERY method
5. Implement until tests pass
6. Run: cargo test --lib -- <package>

### Output
Update PROJECT_BOARD.md with:
- Status change
- Test count
- Any Chesterton's Fence notes

### Completion
git add -A && git commit -m "feat(<package>): Port from OCCT"
```

---

## Chesterton's Fence Notes

Document anything suspicious that might be intentional:

### gp
- `Trsf::Transform` special-cases Identity/Translation/Scale/PntMirror - optimization
- `Resolution()` = DBL_MIN, different from `Confusion()` = 1e-7
- Geometric types (Lin, Pln, Circ, etc.) share common transformation API:
  - mirror_pnt, mirror_ax1, mirror_ax2 (symmetry operations)
  - rotate, scale, transform, translate (affine operations)
  - All return immutable copies with `_ed` suffix for functional style
- Direction methods access via .xyz() to get underlying XYZ
- Ax1/Ax2/Ax3 use `xdirection()`, `ydirection()` (no underscores)

### Port Notes (in progress)
- Created 10 new modules: lin, pln, circ, elips, hypr, parab, cylinder, cone, sphere, torus
- Total test stubs written: 84 tests across all 10 types
- Remaining: API integration fixes (~196 compiler errors due to existing codebase API differences)
- All methods documented, type signatures complete
- Major blockers: existing codebase method name variations (need to verify exact APIs)

---

## Commands

```bash
# Extract GTests for package
python3 scripts/extract_tests.py <Package>

# Run tests for package  
cargo test --lib -- <package>

# Count methods in header
grep -c "^\s*\(void\|double\|bool\|gp_\)" <file>.hxx
```

---

*Last updated: 2025-02-21 10:32 UTC*
*Total tests: 61 (gp) + legacy*
