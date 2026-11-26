# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q49 are there with pointers to their full text). Number new entries from **Q50**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)


## Q50: Gated `update_connections` vs M16 `connections.rs` tests triggering via `EndTurn`

- Needed for: Milestone 17 ("Post-apply sync (`sync_buildings` + `update_connections`)", P7) and `crates/sim-core/tests/connections.rs`.
- Ambiguity: `docs/MILESTONE_17.md` specifies running `update_connections` only on actions that can affect roads, cities, or territory (`BuildRoad`, `Capture`, `ChooseReward { reward: Reward::BorderGrowth, .. }`). However, seven M16 tests in `crates/sim-core/tests/connections.rs` (`a_path_may_pass_through_an_own_city`, `an_adjacent_city_and_capital_are_connected_without_roads`, `capturing_a_city_on_the_path_breaks_the_connection`, `capturing_the_connected_city_moves_its_connection_to_the_new_owner`, `no_connection_changes_after_a_capital_capture_ends_the_game`, `capturing_a_village_removes_its_road_and_the_new_city_still_links`, `an_enemy_village_capture_removes_the_road_and_breaks_the_path`) pre-place roads/cities on the board with `StateBuilder` and call `Action::EndTurn` to trigger the initial connection update, and `connections.rs` is not listed under P6 "Changes to earlier tests".
- Options:
  - A: Keep the M17 gate (`BuildRoad`, `Capture`, `BorderGrowth`) and approve updating those 7 tests in `connections.rs` to trigger the connection update via `BuildRoad` or `Capture` (and for `an_adjacent_city_and_capital_are_connected_without_roads`, founding the adjacent city via `Capture` of an adjacent village or building an unrelated road).
  - B: Include `Action::EndTurn` in the `update_connections` gate (`BuildRoad | Capture | BorderGrowth | EndTurn`), keeping all 7 M16 scripted tests green unchanged while still skipping BFS on Move/Attack/Site/Train/BuildBuilding/etc.
  - C: Keep `update_connections` running after every non-terminal v2 action (single-BFS speedup only, no action-kind gating).
- Agent's pick and why: Option A (or B if zero edits to the 7 scripted tests are preferred). Option A keeps the exact M17 gate (`BuildRoad | Capture | BorderGrowth`) so `EndTurn` incurs no BFS cost and only updates those M16 test setups to use real connection-triggering actions.
- Owner answer: C (owner, 2026-11-26) — single-BFS speedup only, keep `update_connections` running after every non-terminal v2 action (no action-kind gating; 7 M16 tests in `connections.rs` stay untouched).

## Q51: `connections.rs::check_update` population assertion on M17 `BuildBuilding` / `Destroy`

- Needed for: Milestone 17 (`BuildBuilding`, `Destroy`) and `crates/sim-core/tests/connections.rs` (`random_v2_playouts_follow_the_update_and_replay_identically`).
- Ambiguity: `crates/sim-core/tests/connections.rs` line 486 checks `if !matches!(a, Action::Site { .. } | Action::ChooseReward { .. })` that city population changed only by connection deltas (`delta[i]`). Once M17 unlocks `BuildBuilding` and `Destroy` under `V2`, `random_v2_playouts_follow_the_update_and_replay_identically` picks them via `pick_growing` (`pick_weighted`) and fails on line 493 when a building changes a city's population. (`Action::Site` can also change a *different* friendly city's population via `sync_buildings`, though `Site` is already excluded by line 486.) `connections.rs` is not listed under P6 "Changes to earlier tests".
- Options:
  - A: Approve adding `Action::BuildBuilding { .. } | Action::Destroy { .. }` to the line 486 skip guard in `crates/sim-core/tests/connections.rs` (alongside `Site` and `ChooseReward`), while keeping the `connected_to` check running on every non-terminal action.
  - B: Filter `BuildBuilding` and `Destroy` out of `pick_growing` in `connections.rs` (weaker than A because `Site` already uses the line 486 guard).
- Agent's pick and why: Option A — `BuildBuilding` and `Destroy` modify city population just like `Site` and `ChooseReward`, and keeping `connected_to` checked across them exercises connections alongside M17 actions too.
- Owner answer: A (owner, 2026-11-26) — add `Action::BuildBuilding { .. } | Action::Destroy { .. }` to the line 486 population-delta skip guard in `crates/sim-core/tests/connections.rs`.
