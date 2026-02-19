# CASCADE-RS: 2-Week OCCT Parity Sprint

**Start:** 2026-02-19
**Deadline:** 2026-03-05
**Goal:** 100% OCCT feature parity, tested against real CAD files

## Daily Targets

| Day | Focus | Features |
|-----|-------|----------|
| 1 | Sweeps + Curves | prism, revol, pipe, BSpline curves |
| 2 | Surfaces | BSpline surface, offset surface, ruled surface |
| 3 | Fillets | constant fillet, variable fillet, chamfer |
| 4 | Booleans+ | splitter, fuzzy boolean, multi-arg boolean |
| 5 | Healing | sewing, fix shape, remove degeneracies |
| 6 | STEP full | AP203/214/242, colors, assemblies |
| 7 | IGES | read/write, layers, colors |
| 8 | Queries | point-in-solid, distance, validity checks |
| 9 | Features | holes, slots, ribs, patterns |
| 10 | Offsets | thick solid, shell, draft |
| 11 | Lofting | thru-sections, skinning |
| 12 | Advanced | constraints, intersections |
| 13 | Testing | real file validation suite |
| 14 | Polish | edge cases, performance |

## Test Corpus

Need real STEP/IGES files from:
- [ ] FreeCAD exports
- [ ] Fusion360 exports
- [ ] SolidWorks samples
- [ ] GrabCAD models
- [ ] NIST CAD test files

## Parallel Agent Strategy

Run 5-10 sub-agents continuously:
- Each agent: 1 feature implementation
- Verification: must pass cargo test + real file test
- Auto-spawn next feature on completion

## Progress Tracking

Updated in OCCT_FEATURES.md with:
- [x] = implemented and tested
- [~] = implemented, needs testing
- [ ] = not started

## Commands

```bash
# Check progress
cargo run --bin validate-features

# Run full test suite
cargo test

# Test against real STEP file
cargo run --example step_roundtrip -- test_data/real_part.step
```

---

**This is a sprint. No sleep for the agents.**
