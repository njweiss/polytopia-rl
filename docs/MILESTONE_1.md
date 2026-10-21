# Milestone 1: data tables and combat

Covers spec milestones 1 and 2 (repo conventions is already done; this is the first real code).

## Goal
Load the unit and tech tables from `data/*.json` and implement the combat function, so that the
pre-written tests pass.

## In scope
1. `crates/sim-core/src/data.rs`: implement `Tables::load`, `Tables::from_json`, `unit`, `tech`,
   `tech_cost`. Embed the JSON with `include_str!`. `from_json` must return `Err` for malformed
   JSON and for any missing or duplicated `UnitType`/`Tech` (every variant exactly once).
   You may add private fields and helper types; keep the public signatures as they are.
2. `crates/sim-core/src/combat.rs`: implement `round_half_up`, `compute_damage`,
   `resolve_attack`. Follow "Combat" in `docs/SPEC.md` literally: `f32`, formula order as
   written, both forces use pre-attack health, retaliation only if the defender survives and
   the attacker is in the defender's range.

## Out of scope (do not build)
Map, state structs, movement, legality, kill-move, veterans/promotion, defense-bonus lookup
(the caller passes `defense_bonus`), RNG, Python bindings, bots, rendering.

## Interfaces (already stubbed, signatures fixed)
See the stubs in `data.rs` and `combat.rs`. Do not change public signatures; if you believe one
is wrong, record it in `docs/QUESTIONS.md` and stop.

## Gate
`./scripts/check.sh` passes: the 8 tests in `tests/combat_golden.rs` and the 8 in
`tests/tables.rs` (16 total), clippy clean with `-D warnings`.

## Notes for the implementer
- Case 4 in the golden table uses a veteran Rider: max health 15 is passed explicitly.
- Golden case 1 is an exact .5 tie; `f32::round` rounds halves away from zero, which is
  half-up for non-negative inputs.
- The tests include shape checks beyond the five wiki rows (no retaliation out of range or on a
  kill, floor at zero); they follow directly from the spec text.
