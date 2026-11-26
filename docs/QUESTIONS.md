# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q51 are there with pointers to their full text). Number new entries from **Q52**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q52: M15 `ruleset.rs::new_game_reads_the_tables_of_the_configs_ruleset` asserts whole-state V2 == V1 at start

- Needed for: Milestone 18 ("Lighthouses: Placement at game start (`new_game`)") and `crates/sim-core/tests/ruleset.rs` (`new_game_reads_the_tables_of_the_configs_ruleset`, lines 174–179).
- Ambiguity: In M15 (when no other v2 rules were active yet), `ruleset.rs` lines 174–179 assert that `new_game_from_seed(seed, ..., &v2_config())` equals the `V1` starting state once `config.ruleset` and `tables` are swapped back (`assert_eq!(b1, a, "seed {seed}")`). Under M18, `new_game` under `V2` sets the four corner tiles to `lighthouse: true`, `terrain: Terrain::Field`, `resource: None`, and `explored = false` when `!full_observability`, so `b1` differs from `a` on those four corners. M6 ("Changes to earlier tests") approved updating `ruleset.rs` for `BuildMonument` legality, but did not explicitly list `new_game_reads_the_tables_of_the_configs_ruleset`.
- Options:
  - A: Approve updating `new_game_reads_the_tables_of_the_configs_ruleset` in `crates/sim-core/tests/ruleset.rs` to assert that `V2` `new_game` matches `V1` on RNG, seats, cities, units, players (except corner `explored`), and all non-corner tiles, while verifying the four corner tiles have `lighthouse: true`, `Terrain::Field`, `resource: None`, and `explored == false`. (Zero v1 expectations changed.)
  - B: Remove lines 174–179 from `new_game_reads_the_tables_of_the_configs_ruleset` and rely on M18's lighthouse setup tests instead.
- Agent's pick and why: Option A — it preserves the M15 regression check that `generate_map`, seat assignment, RNG consumption, and all non-corner tiles remain byte-identical between `V1` and `V2`, while accommodating M18's four corner lighthouses.
- Owner answer: A (owner, 2026-11-26) — verify RNG, seats, cities, units, and non-corner tiles/explored match `V1` while checking the four corners are fogged `Field` lighthouses with no resource.

## Q53: M16 `connections.rs` playout vacuity guard (`losses > 2`) drops to 2 under M18's shifted action stream

- Needed for: Milestone 18 and `crates/sim-core/tests/connections.rs` (`random_v2_playouts_follow_the_update_and_replay_identically`, line 611).
- Ambiguity: With M18 implemented, all 30 seeds (`0..30`) of `random_v2_playouts_follow_the_update_and_replay_identically` pass every `check_update` and replay assertion (and `built > 1000`, `gains > 30`), but because M18's corner lighthouses (blocking `BuildRoad` on corners and changing Explorer walks / capital population) and `BuildMonument` shift the random policy's action draws, the 30 fixed seeds happen to lose 2 connections instead of >= 3, failing the vacuity guard `assert!(losses > 2, "only {losses} connections lost")`.
- Options:
  - A: Increase the seed range in `connections.rs` (`0..30` -> `0..45`) so the vacuity guard `losses > 2` stays untouched and strong without weakening any threshold.
  - B: Relax the vacuity guard from `losses > 2` to `losses >= 2`.
- Agent's pick and why: Option A — preserves every assertion threshold intact while exercising more v2 playout seeds (~4–5s).
- Owner answer: A (owner, 2026-11-26) — expand seed range `0..30` -> `0..45` in `connections.rs`, keeping `losses > 2` unchanged.
