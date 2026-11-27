# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q54 are there with pointers to their full text). Number new entries from **Q55**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q55: `v2_m19_playouts` random policy never triggers a Persist chain follow-up (`persist_chains == 0`)

- Needed for: Milestone 19 (`crates/sim-core/tests/v2_m19_playouts.rs`, `docs/MILESTONE_19.md` "Tests" item 5)
- Ambiguity: `random_v2_playouts_train_all_new_units_trigger_persist_and_replay_identically` asserts `persist_chains >= 3` across 24 seeds (`0..24`, `turn_cap: 18`) on the default 11x11 fogged map with `ALL_V2_TECHS` unlocked. Because `pick_m19` uses undirected random moves mixed uniformly into `others` alongside dozens of `BuildRoad`/`Site`/`BuildBuilding`/`Destroy`/`Disband` actions (plus a 1/8 `EndTurn` chance whenever `priority` is empty), units rarely cross the 11x11 map or cluster adjacently before being disbanded or hitting the turn cap. In practice `trained` (`>= 10` each for Swordsman, Catapult, Knight), `check_m19_invariants`, and replay determinism all succeed, but `persist_chains` is `0`, failing line 268. Meanwhile all 20 tests in `crates/sim-core/tests/v2_units.rs` (including 5-kill Persist chains, mid-chain `Promote`, non-lethal chain termination, mountain kill without kill-move, lighthouse `pending_reward` resumption, and retaliation) and all 16 tests in `crates/bots/tests/v2_bots.rs` (including `rush_v2_chains_knight_persist_kills_and_promotes_mid_chain`) pass.
- Options:
  - A: Adjust `pick_m19` / map params in `crates/sim-core/tests/v2_m19_playouts.rs` (e.g., smaller map `size: 6` or `7` with `full_observability: true`, avoiding `Disband` and biasing `Move` over `BuildRoad`/`Destroy` spam or toward enemies) so random playouts reliably produce `persist_chains >= 3` and `stiff_defenses >= 3`.
  - B: Keep the 11x11 random playout for broad action/invariant/replay coverage (asserting `trained[0..3] >= 10`) and add a second smaller-board or enemy-seeking playout loop in `v2_m19_playouts.rs` to assert `persist_chains >= 3` and `stiff_defenses >= 3`.
- Agent's pick and why: Option A (or B) — the failure is a Phase A playout-driver calibration issue (written before v2 unit training was implemented), not a simulator bug. Updating `crates/sim-core/tests/v2_m19_playouts.rs` and updating `.git/tests-lock` will let `pixi run check` pass without weakening any invariant or vacuity check.
- Owner answer: A (owner, 2026-11-27) — calibrate `pick_m19` in `crates/sim-core/tests/v2_m19_playouts.rs` to bias enemy-directed moves and avoid `Disband` spam while keeping the 11x11 map and all invariant/vacuity assertions intact.

