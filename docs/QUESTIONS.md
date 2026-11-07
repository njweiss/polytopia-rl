# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q41 are there with pointers to their full text). Number new entries from **Q42**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q42: v2 techs in the state digest

- Needed for: milestone 15 "Ruleset flag" (digest) and "v2 tech research" (TechSet bits 9 to 15);
  docs/MILESTONE_7 "State digest" with Q6 ("`TechSet` is its `u16` bitset").
- Ambiguity: the brief only says the digest appends the ruleset byte. Today's digest rebuilds the
  bitset from `ALL_TECHS`, which (kept at its v1 contents) would silently drop bits 9 to 15, so two v2
  states differing only in a v2 tech would share a digest and `verify` could not see the divergence.
- Options: A) the digest hashes the whole `u16` (bits 9 to 15 included); identical for every v1 state,
  since a v1 set never holds a v2 tech. B) keep folding `ALL_TECHS` (v2 techs invisible to the digest).
- Agent's pick and why: A, the Q6 definition and the only option under which replay verification
  covers v2 research. Encoded in `crates/sim-tools/tests/v2_digest.rs`
  (`v2_techs_are_part_of_the_digest`, and the reference encoder's `techset_bits` now folds all 16
  techs).
- Owner answer:

## Q43: how the Rust `Env` refuses v2

- Needed for: milestone 15 "Env" ("`sim_env::Env::new` ... returns an error ... naming milestone 20").
- Ambiguity: `Env::new(cfg) -> Env` returns no `Result` (it panics on a map below `MIN_SIZE`), and the
  brief's Public API does not change its signature.
- Options: A) `Env::new` panics with a message containing "milestone 20" (like the size check; the
  Python layer checks first and raises `ValueError`); B) `Env::new` returns `Result` (changes every
  caller and earlier test); C) a new `Env::try_new -> Result`.
- Agent's pick and why: A, no signature change and the existing precedent. Encoded in
  `crates/sim-env/tests/v2_hash.rs` (`env_refuses_v2_naming_milestone_20`, via `catch_unwind`).
- Owner answer:

## Q44: the Python `ruleset` keyword

- Needed for: milestone 15 "Env" ("Python `polytopia_sim.Env` ... `ValueError` naming milestone 20").
- Ambiguity: `polytopia_sim.Env` has no way to pass a ruleset today; the brief does not name one.
- Options: A) keyword `ruleset="v1"` (strings as in the `Config` JSON): `"v1"` is today's behavior,
  `"v2"` raises `ValueError` naming milestone 20, anything else raises `ValueError`; B) no keyword
  (nothing to refuse).
- Agent's pick and why: A, the smallest surface that makes the refusal testable; the stub already
  parses it. Encoded in `python/tests/rl/test_ruleset.py`.
- Owner answer:

## Q45: `Combatant.attack` as `f32`

- Needed for: milestone 15 V2 (`UnitStats.attack: f32`) and the mutant "attack truncated to an
  integer (Knight 3)".
- Ambiguity: the brief changes `UnitStats.attack` only; `sim_core::combat::Combatant.attack` is `u32`,
  so feeding a Knight into combat would need `as u32`, truncating 3.5 to 3.
- Options: A) `Combatant.attack` becomes `f32` too (the formulas already cast it to `f32` first, so v1
  results are bit-identical; `Combatant` loses `Eq`); B) keep `u32` until milestone 19.
- Agent's pick and why: A, so the 3.5 reaches the spec formulas unrounded. The stub makes the change;
  `crates/sim-core/tests/combat_golden.rs` literals became `2.0`-style (a type adjustment) and
  `v2_tables.rs::the_knights_attack_is_not_truncated_in_combat` checks Knight vs Warrior = (10, 3).
- Owner answer:

## Q46: shape of the v2 building table and of the v2 test entry point

- Needed for: milestone 15 "Data files" and "Tests" item 2 ("a `from_json`-style entry point the
  implementer adds for v2, named in the stub").
- Ambiguity: (1) the brief says `adjacent` "names improvements or buildings" and that load validates
  "that `adjacent` names exist", without fixing a type; (2) "`deny_unknown_fields` on every row" cannot
  apply to rows in the v1 `units.json` / `techs.json` / `site_actions.json` format, which use
  `#[serde(flatten)]` (serde does not support both).
- Options and agent's pick: (1) `BuildingInfo.adjacent: Vec<String>` of snake-case names, validated
  against `lumber_hut`, `farm`, `mine`, `sawmill`, `windmill`, `forge`, `market`; a later milestone may
  type it. (2) Unknown fields are tested only in `buildings.json`, `monuments.json` and `rules.json`;
  unknown unit/tech/site-action rows (bad variant names) are tested in all three. The entry point is
  `Tables::from_json_v2(units, techs, site_actions, buildings, monuments, rules)` with the v1 files
  always embedded.
- Owner answer:
