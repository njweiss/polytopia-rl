# Milestone 19: v2 units, skills, bots and renderer (v2)

Fifth milestone of game v2 — the final rules milestone before the v2 RL observation/action space (M20).
The rules are in `docs/SPEC.md` "Game v2 rules" ("Units"). Under `ruleset: v2` this milestone unlocks
training and combat for the three v2 units (`Swordsman`, `Catapult`, `Knight`), implements the `Stiff`
and `Persist` skills, updates `RushBot` and `EconomyBot` to play v2 cleanly and use v2 units/economy, and
removes the last v2 rendering gaps (v2 unit glyphs, v2 techs in the text panel, v2 HTML unit/tile data,
and `--ruleset v2` in the CLI).

**The hard constraint (as in M15–M18):** under `ruleset: v1` everything is byte-identical: legal actions,
`apply`, combat, movement, bots (including RNG draws and every v1 bot test), text renderer, `describe_tile`,
HTML `frames_json`, digests, replay files, `Config` JSON, `rules_hash`, `obs_hash`, and observations.
Every v1 identity test and replay fixture passes unchanged.

## Goal

1. Under `V2`, `Action::Train` unlocks `Swordsman` (Smithery, 5★), `Catapult` (Mathematics, 8★), and
   `Knight` (Chivalry, 8★) with their table stats, skills, movement, score (25 / 40 / 40), and disband
   refunds (2 / 4 / 4).
2. `Skill::Stiff` suppresses retaliation unconditionally; `Skill::Persist` restores an attack on every
   attack kill with no chain limit (without restoring movement, capture, or recover).
3. `RandomBot`, `RushBot`, and `EconomyBot` play full `V2` games legally, pass the no-cheating property
   under `V2`, and beat `RandomBot` under `V2` (`RushBot` and `EconomyBot` gain lightweight `V2`-only
   extensions while remaining 100% byte-identical under `V1`).
4. `sim_tools` renders v2 units (`Swordsman`, `Catapult`, `Knight`), v2 techs, and v2 tile features in
   the text renderer and HTML viewer, and the `polytopia` CLI accepts `--ruleset v1|v2`.

## Decisions already made (owner)

- All v2 unit stats and skill rules in `docs/SPEC.md` "Game v2 rules" ("Units", owner 2026-11-01/02) and
  the M15 data table `data/v2/units.json`:
  - `Swordsman`: cost 5, health 15, attack 3, defense 3, movement 1, range 1, `[Dash, Fortify]`, `Smithery`.
  - `Catapult`: cost 8, health 10, attack 4, defense 0, movement 1, range 3, `[Stiff]`, `Mathematics`.
  - `Knight`: cost 8, health 10, attack 3.5, defense 1, movement 3, range 1, `[Dash, Fortify, Persist]`, `Chivalry`.
- `Stiff`: unit never retaliates; with no `Dash`, a Catapult cannot attack after moving.
- `Persist`: each kill by the unit's attack restores a full attack with no limit; does not restore
  movement apart from the melee kill-move onto the target's tile.
- Unit score stays `5 × cost` (Swordsman 25, Catapult 40, Knight 40); Super Unit stays 50.
- Bots must at least handle v2 units and actions legally; how much they use v2 content is decided here.

## In scope

1. `sim_core::actions`:
   - `legal_actions` and `train_legal`: iterate/check `units(state.config.ruleset)` instead of `ALL_UNITS`.
   - `apply_attack`: suppress retaliation when `sd.skills.contains(&Skill::Stiff)`; on a surviving
     attacker's kill (`defender_died && !attacker_died`) when `sa.skills.contains(&Skill::Persist)`, set
     `u.phase = Phase::Moved` instead of `Phase::Attacked` (U1).
2. `sim_core::view`:
   - Add `PlayerView::ruleset(&self) -> Ruleset` (public config value, no hidden-info leak).
3. `bots`:
   - `attack_score` (`crates/bots/src/lib.rs`): skip retaliation damage when the defender has `Skill::Stiff`.
   - `RushBot` and `EconomyBot`: `V2`-only strategy extensions gated on `view.ruleset() == Ruleset::V2`
     (U2), leaving `V1` 100% untouched.
4. `sim_tools`:
   - `render.rs`: v2 unit glyphs (replacing `unimplemented!("v2 unit glyphs")`, U3); `panel_line` iterates
     `techs(state.config.ruleset)` instead of `ALL_TECHS`; `describe_tile` appends v2-only tile details
     under `V2` (U4).
   - `html.rs`: add v2 units to `KIND` and legend; emit optional v2 tile keys in `tile_json` only when
     present/true so v1 `frames_json` is byte-identical (U4).
   - `cli.rs`: add optional `--ruleset v1|v2` (default `v1`) to `play-random`, `play`, and `tournament` (U5).

## Out of scope

v2 observation channels, factored action slots for v2 actions/units, `sim_env::Env` and Python `Env`
support for `V2`, and the damage preview observation feature (all M20); v1 -> v2 policy transfer (M21);
water/naval units and ruins (v3).

## Public API (fixed so tests can import it)

```rust
// sim_core::view
impl<'a> PlayerView<'a> {
    pub fn ruleset(&self) -> Ruleset;
}
```

No existing public signatures change in `sim_core`, `bots`, or `sim_tools`. `ALL_UNITS` (5) and
`ALL_TECHS` (9) stay unchanged so `sim_env`, `obs_hash`, and `v1` stay byte-identical.

## Rules

### Training v2 units (`V2` only; illegal under `V1`)

- `legal_actions` emits `Action::Train { city, kind }` per owned city in ascending `CityId` order, then
  `units(state.config.ruleset)` order:
  - Under `V1`: `Warrior, Rider, Archer, Defender` (`SuperUnit` has `cost: None`).
  - Under `V2`: `Warrior, Rider, Archer, Defender, Swordsman, Catapult, Knight`.
- `train_legal` checks `units(state.config.ruleset).contains(&kind)` and reads `cost` and `unlocked_by`
  from `state.tables.unit(kind)` as before:
  - `Swordsman`: requires `Tech::Smithery`, costs 5 stars.
  - `Catapult`: requires `Tech::Mathematics`, costs 8 stars.
  - `Knight`: requires `Tech::Chivalry`, costs 8 stars.
- City occupancy, capacity (`supported_units < city.level + 1`), origin assignment, `Phase::Done` on
  spawn, unit score (`5 × cost`), and Disband refund (`cost / disband_refund_divisor`) all follow the
  existing table-driven rules.

### Combat and skills (`Stiff`, `Persist`, Catapult `defense == 0`, Knight `attack == 3.5`)

- **`Stiff`:** in `apply_attack`, retaliation runs only if `defender_health > 0 && in_range &&
  !sd.skills.contains(&Skill::Stiff)`. A `Stiff` defender never deals retaliation damage (even if in
  range and even if a custom table gives it `defense > 0`).
  - Catapult has `range: 3`, `defense: 0`, `skills: [Stiff]` (no `Dash`, no `Escape`, no `Fortify`).
  - With no `Dash`, a Catapult cannot attack after moving (`Fresh -> Move -> Moved`, no `Attack` legal).
  - With `range == 3 > 1`, a Catapult attacks at Chebyshev distance 1..=3, never kill-moves on kill, and
    sets `last_dir = Some(direction(a.pos, d.pos))` on attack.
  - With `defense == 0`, `defense_force == 0.0`, so any living attacker deals
    `round_half_up(attacker.attack * 4.5)` regardless of terrain/city defense bonus.
- **`Persist`:** in `apply_attack`, when `!attacker_died`:
  - If `defender_died && sa.skills.contains(&Skill::Persist)`, set `u.phase = Phase::Moved` (U1);
    otherwise set `u.phase = Phase::Attacked`.
  - Because `Knight` has `Dash` and no `Escape`, `Phase::Moved` allows chaining `Attack` from the
    Knight's new position (after melee kill-move) with no limit on chain length, while disallowing
    `Move`, `Capture`, and `Recover`.
  - `Promote` (on reaching 3 kills) and `Disband` remain legal in `Phase::Moved` mid-chain, and
    `Promote` leaves `u.phase == Phase::Moved` so the newly veteran Knight (15/15 HP) can keep chaining.
  - If a Knight kills a defender standing on a mountain when the Knight's owner lacks `Climbing`, the
    kill-move is skipped (`terrain_allows == false`), `u.phase` still becomes `Phase::Moved`, and the
    Knight may attack another adjacent enemy from its current tile.
  - If a kill-move reveals a corner lighthouse that triggers a capital `pending_reward`, `ChooseReward`
    resolves first while the Knight retains `Phase::Moved`, and the Persist chain can continue afterward.
  - A non-lethal attack ends the chain (`u.phase = Phase::Attacked`, no further attacks or moves).
  - A retaliation kill never triggers `Persist` (only the attacker's phase transitions).
- **Knight movement:** `movement == 3` -> 6 half-points in `reachable_tiles` (up to 3 plain tiles, or up
  to 6 tiles along usable v2 roads, subject to ZoC and rough-terrain stops).
- **Swordsman:** `Dash` (can attack after moving), `Fortify` (receives city 1.5x and wall 4.0x defense
  bonuses), 15 max HP (20 veteran), 3 attack, 3 defense.

### Bots (`V1` byte-identical; `V2` legal + lightweight v2 play, U2)

- **`attack_score` (`crates/bots/src/lib.rs`):** `taken = if in_range && !stiff { damage.to_attacker }
  else { 0 }`, where `stiff = view.tables().unit(d.kind).skills.contains(&Skill::Stiff)`. (No-op under
  `V1` since no v1 unit has `Stiff`.)
- **Under `Ruleset::V1`:** `RandomBot`, `RushBot`, and `EconomyBot` execute the exact v1 rules and RNG
  draws unchanged.
- **Under `Ruleset::V2` (`view.ruleset() == Ruleset::V2`, U2):**
  - **`RushBot`:**
    1. `choose_reward` (`REWARDS`)
    2. `promote`
    3. `capture`
    4. `best_attack(0)` (automatically chains `Knight` `Persist` kills!)
    5. `build_monument` (first legal `Action::BuildMonument`, free +3 pop & 400 score)
    6. `train`: per owned city in ascending id, first legal of `[Knight, Rider, Warrior]`
    7. `research`: first legal of `[Riding, FreeSpirit, Chivalry]`
    8. `HarvestFruit` -> `HarvestAnimal`
    9. `recover`
    10. `move_toward` (enemy capital -> nearest visible enemy city or unit -> explore)
    11. `EndTurn`
  - **`EconomyBot`:**
    1. `choose_reward` (`REWARDS`)
    2. `promote`
    3. `capture`
    4. `best_attack(3)`
    5. `build_monument` (first legal `Action::BuildMonument`)
    6. `SITES` (`Farm`, `Mine`, `HarvestFruit`, `HarvestAnimal`, `LumberHut`)
    7. `build_building`: first legal `Action::BuildBuilding` in `[Market, Forge, Sawmill, Windmill]`
       order (tiles row-major within each building kind, matching `site_of_kind`)
    8. `research`: first legal of the 8 v1 techs followed by `[Roads, Mathematics, Construction,
       Smithery, Trade, FreeSpirit, Chivalry]`
    9. `train`: while `my_units().len() < 2 * my_cities().len()`, per owned city in ascending id, first
       legal of `[Swordsman, Catapult, Warrior]`
    10. `recover`
    11. `move_toward` (nearest visible unoccupied village -> explore)
    12. `EndTurn`
- **No cheating:** `PlayerView` never leaks hidden opponent/fog state under `V2` either (including
  hidden v2 fields: opponent's `kills`, `lighthouses_discovered`, `monuments_unlocked`, `monuments_placed`,
  and unexplored tiles' roads/buildings/monuments).

### Renderer, HTML viewer, and CLI (`V1` byte-identical; U3–U5)

- **Unit glyphs (U3):** in `sim_tools::render::cell` (uppercase P0, lowercase P1) and `sim_tools::html`
  `KIND` map + legend:
  - `Swordsman` -> `'X'` / `'x'`
  - `Catapult` -> `'C'` / `'c'`
  - `Knight` -> `'K'` / `'k'`
- **Text renderer panel (`render.rs:138`):** iterate `techs(state.config.ruleset)` instead of `ALL_TECHS`
  so v2 techs appear in declaration order under `V2` while `V1` stays identical.
- **`describe_tile` (`render.rs`, U4):** under `V1`, exact existing format unchanged. Under `V2`, append
  space-separated optional tags at the end of the line (only for non-default v2 tile fields on explored
  tiles, in fixed order):
  - `road` if `t.road`
  - `bldg=<snake>(&b)(<level>)` if `let Some(b) = t.building` (e.g. `bldg=sawmill(2)`)
  - `monument=<snake>(&m)` if `let Some(m) = t.monument`
  - `lighthouse` if `t.lighthouse`
- **HTML viewer (`html.rs`, U4):**
  - Add `swordsman: "X"`, `catapult: "C"`, `knight: "K"` to `KIND` and update the legend text.
  - In `tile_json`, omit v2 keys when false/None (`road`, `building`, `building_level` when
    `building.is_some()`, `monument`, `lighthouse`), so every v1 `frames_json` output is byte-identical.
- **CLI (`cli.rs`, U5):**
  - Accept optional `--ruleset v1|v2` (default `v1`; invalid value exits 2 with usage error) on
    `play-random`, `play`, and `tournament`, setting `config.ruleset`.

### Digest & Hashes

- `state_digest`: already hashes `u.kind as u8` (0..=7) and all v2 fields; no digest format change in M19.
- `rules_hash_for(V1)` (`a8b52b91eae34503`), `rules_hash_for(V2)`, and `obs_hash(11)` (`d5f9b1033cfd55ca`)
  are all unchanged (`data/*.json` unchanged; `RULES_VERSION` not bumped).

## Clarifications (owner: confirm or overrule)

- **U1 (Persist phase):** Represent a `Persist` attack-kill reset as `u.phase = Phase::Moved` (keeping
  the 4-variant `Phase` enum unchanged, since `Knight` has `Dash` and no `Escape`, so `Phase::Moved`
  permits unlimited chained attacks + `Promote`/`Disband` while blocking `Move`/`Capture`/`Recover`). — Confirmed (owner, 2026-11-27)
- **U2 (Bot v2 behavior):** Keep `V1` bots 100% byte-identical and add the lightweight `V2`-only
  extensions above (`PlayerView::ruleset()`, `RushBot` using monuments + `Riding -> FreeSpirit -> Chivalry`
  + `Knight`, `EconomyBot` using monuments + economy buildings + v2 techs + `Swordsman`/`Catapult`, plus
  `Stiff` awareness in `attack_score`) so M20/M21 training and evaluation against bots exercise v2 content. — Confirmed (owner, 2026-11-27)
- **U3 (v2 unit glyphs):** Use `X` (Swordsman), `C` (Catapult), `K` (Knight) — lowercase `x`/`c`/`k` for
  P1 — in `render.rs` and `html.rs` (alternatives: `N` for Knight like chess, `M` or `B` for Swordsman). — Confirmed (owner, 2026-11-27)
- **U4 (Renderer & HTML v2 fields):** List `techs(state.config.ruleset)` in `panel_line`, append optional
  v2 tags in `describe_tile` under `V2` only, and emit non-default v2 tile keys in `html::tile_json` (all
  v1 outputs byte-identical). — Confirmed (owner, 2026-11-27)
- **U5 (CLI `--ruleset`):** Add optional `--ruleset v1|v2` (default `v1`) to `play-random`, `play`, and
  `tournament`. — Confirmed (owner, 2026-11-27)
- **U6 (Earlier test edits):** Update the two M15 tests in `crates/sim-core/tests/ruleset.rs` that
  previously asserted `NEW_UNITS` (`Swordsman`, `Catapult`, `Knight`) were still illegal under `V2`, so
  they now assert `NEW_UNITS` are legal under `V2` (while all `under_v1_*` tests stay unchanged). — Confirmed (owner, 2026-11-27)

## Changes to earlier tests

Allowed only for the following (no v1 expected value may change; no test may be removed or weakened; list
every edit in the phase A commit message):

1. **M15 `crates/sim-core/tests/ruleset.rs` (U6):**
   - `under_v2_with_every_tech_no_new_unit_or_burn_forest_is_legal` (lines 412–435): assert `Swordsman`,
     `Catapult`, and `Knight` ARE legal `Action::Train` actions under `V2` when all techs and stars are
     present (all `under_v1_*` tests untouched).
   - `random_v2_playouts_research_v2_techs_and_nothing_else` (lines 458–463): remove the M15 pre-M19
     assertion that `NEW_UNITS` never appear in `V2` legal actions (keeping the `expected_research` and
     `researched_v2 >= 5` checks).

## Tests (phase A guidance)

1. **v1 identity:** all existing v1 tests in `sim-core`, `bots`, `sim-tools`, `sim-env`, and Python pass
   unchanged; `rules_hash_for(V1) == "a8b52b91eae34503"`, `obs_hash(11) == "d5f9b1033cfd55ca"`, v1 replay
   fixture verifies unchanged.
2. **Training v2 units (`crates/sim-core/tests/v2_units.rs`):**
   - Under `V2`: `Swordsman` (Smithery, 5★), `Catapult` (Mathematics, 8★), `Knight` (Chivalry, 8★) are
     legal iff tech researched, enough stars, city tile empty, and below unit cap; canonical order in
     `legal_actions` (`Warrior, Rider, Archer, Defender, Swordsman, Catapult, Knight`).
   - Spawned with table HP (15 / 10 / 10), `Phase::Done`, city origin; stars deducted; score increases by
     25 / 40 / 40; Disband refunds 2 / 4 / 4.
   - Under `V1`: never legal or applicable even with all v1 techs and 99 stars.
3. **Combat, `Stiff`, `Persist`, and movement (`crates/sim-core/tests/v2_units.rs`):**
   - **Swordsman:** Dash (move then attack), Fortify (1.5x city, 4.0x wall defense bonus), 15 HP / 3 atk /
     3 def combat values verified against hand-computed `f32` formulas.
   - **Catapult:** range 3 attack (distance 1, 2, 3 legal; distance 4 illegal); ranged kill does NOT
     kill-move and sets `last_dir`; no Dash (cannot attack after moving); `Stiff` never retaliates
     (including a test with a custom v2 table giving a `Stiff` unit `defense > 0`, so `Stiff` itself is
     tested independently of `defense == 0`!); attacking a Catapult (`defense == 0`) deals
     `round_half_up(attack * 4.5)` even in a walled city (no Fortify, `defense_force == 0`).
   - **Knight:** 3.5 attack combat math (hand-computed against Warrior/Defender/Swordsman); movement 3
     (3 plain tiles, up to 6 tiles on v2 roads, stopped by ZoC/forest without road); Dash (move then
     attack); Fortify (city/wall bonus); **Persist**:
     - Multi-kill chain (e.g. 4+ consecutive kills in one turn with kill-moves) — no limit.
     - Mid-chain `Promote` on 3rd kill restores HP to 15/15 and keeps `Phase::Moved` so chain continues.
     - Mid-chain non-lethal attack transitions to `Phase::Attacked` and ends the chain (no more attacks or
       moves).
     - After a Persist kill, `Move`, `Capture` (even if kill-moved onto a village or enemy city!), and
       `Recover` are illegal.
     - Persist kill against an enemy on a mountain without `Climbing`: no kill-move, stays in place with
       `Phase::Moved`, can still attack another adjacent enemy.
     - Mid-chain kill-move discovering a corner lighthouse that triggers capital `pending_reward`:
       `ChooseReward` resolves, then Knight resumes its Persist chain.
     - Retaliation kill by a defending Knight does NOT grant Persist or change anything improperly.
4. **v2 bots (`crates/bots/tests/v2_bots.rs`):**
   - `PlayerView::ruleset()` returns `V1` / `V2` appropriately.
   - Scripted unit tests: `RushBot` under `V2` places free monuments, researches `FreeSpirit`/`Chivalry`,
     trains `Knight` over `Rider`, chains `Knight` Persist kills; `EconomyBot` under `V2` places free
     monuments, builds `Market`/`Forge`/`Sawmill`/`Windmill` in priority order, researches v2 techs,
     trains `Swordsman`/`Catapult`; `attack_score` accounts for `Stiff` (0 retaliation).
   - Property tests: all bot pairs (`Random`, `Rush`, `Economy`, bare and epsilon) play full `V2` games
     legally, deterministically, and end turns within 200 actions.
   - Behavior floor under `V2`: `RushBot` and `EconomyBot` each win >= 75% of 40 games (seeds 0..20,
     swapped seats, turn cap 30) against `RandomBot`.
   - No-cheating under `V2`: twin states differing in hidden v2 fields (opponent `kills`,
     `lighthouses_discovered`, `monuments_unlocked`, `monuments_placed`, hidden v2 techs, unexplored
     roads/buildings/monuments/units) produce identical `PlayerView` outputs and bot choices.
5. **Renderer, HTML, CLI, and v2 playouts (`crates/sim-tools/tests/v2_render.rs`, `crates/sim-core/tests/v2_m19_playouts.rs`):**
   - `render` shows `X`/`x`, `C`/`c`, `K`/`k` for P0/P1 Swordsman/Catapult/Knight and lists v2 techs in
     `panel_line`; `describe_tile` appends v2 tags under `V2` and leaves `V1` unchanged.
   - `html_page` and `frames_json` work on v2 replays containing v2 units, buildings, roads, lighthouses,
     and monuments (and v1 `frames_json` is byte-identical).
   - `run_cli` accepts `--ruleset v2` on `play-random`, `play`, and `tournament`, rejects invalid rulesets
     with exit code 2, and produces verifiable v2 replays.
   - Multi-seed random v2 playouts train all 3 new units, trigger Persist chains, maintain invariants, and
     replay identically.

Mutants to aim for: `train_legal` still gating on `ALL_UNITS`; `legal_actions` omitting v2 units or wrong
order; `Stiff` still retaliating when `defense > 0`; `Persist` setting `Phase::Fresh` (allowing Move/Capture/
Recover after a kill) or `Phase::Attacked` (no chain) or capping chain length; `Persist` firing on a
non-lethal attack or on retaliation; `Promote` overwriting `Phase::Moved` to `Phase::Done`; Catapult
allowed to attack after moving or kill-moving on kill; `Knight` attack truncated to 3.0; `render.rs`
panicking on v2 unit glyphs or omitting v2 techs; `html.rs` emitting `?` for v2 units or adding v2 keys
to v1 `tile_json`; `RushBot`/`EconomyBot` changing any `V1` action or leaking hidden v2 state.

## Gate

`pixi run check` exits 0 (fmt, clippy `-D warnings`, all Rust, Python, and RL tests green). `rules_hash()`
== `"a8b52b91eae34503"`, `obs_hash(11)` == `"d5f9b1033cfd55ca"`, and `polytopia verify` accepts the v1
replay fixture unchanged. Rules milestone: a `/review 19` phase is recommended.

## Notes for the implementer

- Read all unit stats and skills from `state.tables.unit(kind)`; never hard-code unit numbers or check
  `u.kind == UnitType::Knight` where `skills.contains(&Skill::Persist)` or `Skill::Stiff` applies.
- Keep `ALL_UNITS` (5) and `ALL_TECHS` (9) untouched; use `units(state.config.ruleset)` and
  `techs(state.config.ruleset)` where v2 content should be visible.
- Keep all `V1` bot, renderer, HTML, CLI, and core behavior byte-identical.
- If a test contradicts this brief or the spec, stop and add a `docs/QUESTIONS.md` entry (next number
  **Q55**).
