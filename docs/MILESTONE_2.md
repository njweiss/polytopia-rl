# Milestone 2: RNG and map generation

Spec milestone 3 (plus the RNG it needs). Depends on milestone 1 only for the repo layout.

## Goal
Implement the game's deterministic RNG and the seeded Tribes-style map generator so that the
pre-written tests in `crates/sim-core/tests/rng.rs` and `tests/mapgen.rs` pass.

## In scope
1. `src/rng.rs`: PCG-XSH-RR 64/32 (`pcg32`), seeded exactly like the reference
   `pcg32_srandom_r(&rng, seed, 54)`. Implement `new`, `next_u32`, `gen_f32`, `gen_range`.
   No external crates. The published test vector pins it.
2. `src/map.rs`: implement `generate_map` per "Map generation > Pipeline" in `docs/SPEC.md`.
   `grid.rs` and the types, accessors and `MapParams::default()` in `map.rs` are already done.

## Out of scope (do not build)
Game state, seat assignment (which player gets which capital), movement, fog, rendering,
Python bindings, benchmarks, any dependency (rand, etc.).

## Draw order (replay stability)
All randomness comes from the `&mut Rng` passed in. Use this order, so a seed always means the
same map for any correct implementation:
1. Capital 1: `gen_range(k)` over the inland tiles (row and col in `2..=size-3`), listed in
   row-major order.
2. Capital 2: `gen_range(k)` over the inland tiles at maximum Chebyshev distance from capital 1,
   in row-major order.
3. Terrain: for every non-capital tile in row-major order, one `gen_f32()` = `u`. Forest if
   `u < forest_prob`; mountain if `u > 1 - mountain_prob`; else field.
4. Villages: no randomness. Single row-major pass; a tile becomes a village if it is not on the
   map edge, not a mountain, and farther than 2 (Chebyshev) from every capital and every village
   already placed. A village tile becomes Field.
5. Resources: for every tile in row-major order that is not a site and lies within 2 of a site:
   probabilities are multiplied by `outer_ring_factor` when the tile is not within 1 of a site.
   - Field: one `gen_f32()` against `fruit_prob * (1 - crop_prob / 2) * factor`; only if that
     fails, a second `gen_f32()` against `crop_prob * (1 - fruit_prob / 2) * factor`.
   - Forest: one draw against `animal_prob * factor`. Mountain: one draw against
     `metal_prob * factor`.
6. Imperius guarantee: for each capital in generation order, while it has fewer than
   `min_capital_fruit` fruit among its 8 neighbors: `gen_range(8)` picks a neighbor (row-major
   order of the 8), which becomes Field with Fruit.

## Clarifications to the spec (owner: please confirm or overrule, then update the doc)
- **C1: RNG.** pcg32 as above. The spec said only "the game's own RNG". - Confirmed
- **C2: site tiles.** Capital and village tiles are `Terrain::Field` with no resource
  ("a village on a forest replaces the forest"). - Confirmed
- **C3: village test wording.** The spec test says "no unmarked tile remains". Mountain and
  edge tiles are never village-eligible, so the correct statement is "no *eligible* tile (non-edge,
  non-mountain) remains farther than 2 from every site". The tests check the full greedy replay. - Confirmed
- **C4: terrain shares.** The spec says shares come out "near 40% forest, 15% mountain, 45%
  field". Measured on this pipeline (20,000 seeds): 36.1% / 14.5% / 49.4%. Villages replace forests
  and the guarantee overwrites ring tiles with field. The tests use the measured values. - Confirmed
- **C5: village rings overlap.** A tile within 1 of one site and 2 of another counts as inner
  (the larger mark wins). Marks never downgrade. - Confirmed

## Gate
`./scripts/check.sh` passes, including 7 tests in `tests/rng.rs` and 15 in `tests/mapgen.rs`.
(The mapgen statistics tests take about a second; that is expected.)

## Notes for the implementer
- Use `f32` for all probability arithmetic in the order written above.
- Do not iterate any `HashMap`/`HashSet` while generating.
- If the tests seem to contradict the spec, stop and write it in `docs/QUESTIONS.md`.
