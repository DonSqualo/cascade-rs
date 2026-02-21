# CASCADE-RS: OCCT Test-Driven Port

**Method:** Extract OCCT GTests → Implement Rust that passes them
**Source:** /home/heim/projects/occt-source
**Tests:** 251 test files, 100K+ lines of test specifications

---

## Progress Overview

| Layer | Package | OCCT Lines | Rust Lines | Tests | Status |
|-------|---------|------------|------------|-------|--------|
| 0 | precision | ~400 | 91 | 2/2 | ✅ Complete |
| 0 | gp (primitives) | ~15,000 | 5,500+ | 48/48 | ✅ Core Complete |
| 1 | Bnd | ~5,000 | - | 0/268 | 🔴 Not started |
| 1 | math | ~30,000 | - | 0/? | 🔴 Not started |
| 2 | Geom | ~50,000 | - | 0/? | 🔴 Not started |

---

## Completed (Layer 0)

### precision ✅
Port of: `src/FoundationClasses/TKernel/Precision/`
- All OCCT precision constants (ANGULAR, CONFUSION, etc.)
- 2 tests passing

### gp (geometric primitives) ✅
Port of: `src/FoundationClasses/TKMath/gp/`

| Class | Source | Rust | Tests | Status |
|-------|--------|------|-------|--------|
| XYZ | gp_XYZ.hxx | xyz.rs | 18/18 | ✅ |
| Pnt | gp_Pnt.hxx | pnt.rs | 5/5 | ✅ |
| Vec | gp_Vec.hxx | vec.rs | 4/4 | ✅ |
| Dir | gp_Dir.hxx | dir.rs | 7/7 | ✅ |
| Mat | gp_Mat.hxx | mat.rs | 3/3 | ✅ |
| Ax1 | gp_Ax1.hxx | ax1.rs | 3/3 | ✅ |
| Ax2 | gp_Ax2.hxx | ax2.rs | 2/2 | ✅ |
| Ax3 | gp_Ax3.hxx | ax3.rs | 3/3 | ✅ |
| Trsf | gp_Trsf.hxx | trsf.rs | 5/5 | ✅ |

**OCC23361 test case passing:** Transformation composition verified.

---

## Next Up (Layer 0-1)

### gp remaining types
- [ ] gp_Pln (Plane)
- [ ] gp_Lin (Line)
- [ ] gp_Circ (Circle)
- [ ] gp_Elips (Ellipse)
- [ ] gp_Hypr (Hyperbola)
- [ ] gp_Parab (Parabola)
- [ ] gp_Cylinder
- [ ] gp_Cone
- [ ] gp_Sphere
- [ ] gp_Torus
- [ ] gp_GTrsf (General transformation)
- [ ] 2D variants (Pnt2d, Vec2d, Dir2d, etc.)

### Bnd (Bounding boxes)
Source: `src/FoundationClasses/TKMath/Bnd/`
- [ ] Bnd_Box
- [ ] Bnd_Box2d
- [ ] Bnd_Sphere
- [ ] Bnd_OBB
- [ ] Bnd_Range
- [ ] BndLib

### math (Numerical algorithms)
Source: `src/FoundationClasses/TKMath/math/`
Heavy lifting - will need sub-agents.

---

## Test Extraction

```bash
# Extract tests for a package
python3 scripts/extract_tests.py <Package>

# Example: Bnd tests
python3 scripts/extract_tests.py Bnd
# Found 268 tests

# Run gp tests
cargo test --lib -- gp
```

---

## Sub-Agent Tasks

For parallel porting, spawn sub-agents with:

```
Task: Port OCCT package <X> to cascade-rs

1. Read OCCT source: /home/heim/projects/occt-source/src/.../X/
2. Extract tests: python3 scripts/extract_tests.py X
3. Create src/<x>/mod.rs matching OCCT structure
4. Implement until tests pass: cargo test --lib -- <x>
5. Update PROJECT_BOARD.md with progress

Workspace: /home/heim/projects/cascade-rs
Branch: git checkout -b port/<package>
```

---

## Notes (Chesterton's Fence)

Document anything that looks weird but might be intentional:

### gp package
- **gp_Trsf::Transform**: Has special-case handling for Identity, Translation, Scale, PntMirror - optimization, keep it
- **Dir normalization**: Always normalizes on construction - this is intentional for unit vector guarantee

### Bnd package
- (pending)

---

## Integration Test: STEP Files

Final validation: load real-world STEP files.

Location: `test_corpus/`

---

*Last updated: 2026-02-21*
*Layer 0 (gp, precision) complete: 50 tests passing*
