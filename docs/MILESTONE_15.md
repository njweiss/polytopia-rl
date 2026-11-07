# Milestone 15: v2 foundation (ruleset flag, v2 data tables, the new tech tree)

First milestone of game v2. The v2 rules are approved in the Claude Doc tab "v2 rules (draft)"
(https://claude.ai/artifact/3yVekWM5SAoUVxjZVmomTm#f868c7c8-c04b; working notes `docs/V2_RULES_NOTES.md`).
This milestone lays the foundation every later v2 milestone builds on: a ruleset flag, the v2 data tables,
the enum variants, and the seven new technologies, researchable under v2. No other v2 rule is active yet.

**The hard constraint:** under `ruleset: v1` everything is byte-identical to today: legal actions,
`apply`, digests, replay files, `Config` JSON, `rules_hash`, `obs_hash`, observations, bots. Every
existing test passes with no expected value changed.

## Goal

1. A `Ruleset` (`V1` | `V2`) in `Config`, defaulting to `V1`.
2. The v2 tables in `data/v2/*.json`, loaded and validated through `Tables`, with every number in them
   matching the approved v2 rules.
3. Under `V2`: the 7 new techs are researchable with their tiers and prerequisites, score as techs, and
   fit the tech set. Nothing else from v2 is legal yet (no new units, buildings, roads, actions,
   monuments or lighthouses).
4. Hashes: `rules_hash` for v1 unchanged; a distinct v2 `rules_hash`.

## Decisions already made (owner)

- The v2 rules (Claude Doc tab), including the approved proposals of 2026-11-02, notably "the ruleset
  flag is part of `rules_hash`".
- v1 stays runnable and unchanged behind the flag.
- v2 milestones after this one (plan): M16 roads and city connections; M17 buildings and tile actions
  (sawmill, windmill, forge, market, burn forest, disband, destroy) and population loss; M18 monuments
  and lighthouses; M19 the new units and skills, and the bots; M20 the v2 observation and action space
  (including the damage preview); M21 the v1 to v2 transfer.

## In scope

1. `sim_core`: `Ruleset`, the `Config.ruleset` field, new enum variants, `Tables::load_for(ruleset)`,
   the v2 data files, v2 tech research legality, the tech set holding 16 techs, `UnitStats.attack` as
   `f32`.
2. `sim_env::hash`: `rules_hash_for(ruleset)`; `obs_hash` unchanged.
3. `sim_env::Env` and the Python `Env` reject `ruleset = v2` with a clear error until M20.

## Out of scope

Every v2 rule other than tech research: roads, buildings, burn forest, disband, destroy, monuments,
lighthouses, the Explorer's lighthouse step, population loss (and `City.population` as a signed type;
that change comes with M17), training the new units, Stiff and Persist, bot changes, observation or
action-space changes, the damage preview.

## Public API (fixed so tests can import it)

```rust
// sim_core::state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ruleset { #[default] V1, V2 }
pub struct Config { pub full_observability: bool, pub turn_cap: u32, pub score_weights: ScoreWeights,
                    pub ruleset: Ruleset }   // serde: default V1, not serialized when V1

// sim_core::data (new variants are appended after the existing ones; existing order unchanged)
pub enum UnitType { Warrior, Rider, Archer, Defender, SuperUnit, Swordsman, Catapult, Knight }
pub enum Skill { Dash, Escape, Fortify, Static, Stiff, Persist }
pub enum Tech { Climbing, Hunting, Organization, Riding, Archery, Forestry, Farming, Strategy, Mining,
                Roads, FreeSpirit, Trade, Chivalry, Mathematics, Construction, Smithery }
pub enum SiteAction { HarvestAnimal, HarvestFruit, LumberHut, ClearForest, Farm, Mine, BurnForest }
pub enum Building { Sawmill, Windmill, Forge, Market }       // new
pub enum Monument { ParkOfFortune, GateOfPower, EyeOfGod }   // new

pub const ALL_UNITS / ALL_TECHS / ALL_SITE_ACTIONS / ALL_REWARDS   // unchanged: the v1 lists
pub fn units(ruleset: Ruleset) -> &'static [UnitType]              // v1: ALL_UNITS; v2: all 8
pub fn techs(ruleset: Ruleset) -> &'static [Tech]                  // v1: ALL_TECHS; v2: all 16
pub fn site_actions(ruleset: Ruleset) -> &'static [SiteAction]     // v1: ALL_SITE_ACTIONS; v2: all 7

pub struct UnitStats { pub cost: Option<u32>, pub health: u32, pub attack: f32, pub defense: u32,
                       pub movement: u32, pub range: u32, pub skills: Vec<Skill>, pub unlocked_by: Option<Tech> }

impl Tables {
    pub fn load() -> Result<Tables, String>                        // unchanged: the v1 tables
    pub fn load_for(ruleset: Ruleset) -> Result<Tables, String>    // V1 == load(); V2 adds data/v2/*
    pub fn ruleset(&self) -> Ruleset
    pub fn building(&self, b: Building) -> &BuildingInfo           // panics under V1
    pub fn monument(&self, m: Monument) -> &MonumentInfo           // panics under V1
    pub fn v2_rules(&self) -> &RulesV2                             // panics under V1
}

// sim_env::hash
pub fn rules_hash() -> String                                       // unchanged (= v1)
pub fn rules_hash_for(ruleset: Ruleset) -> String                   // V1 == rules_hash()
pub fn embedded_v2_data_files() -> Vec<(&'static str, &'static str)>  // "v2/<name>", file-name order
```

`ALL_UNITS`, `ALL_TECHS`, `ALL_SITE_ACTIONS` keep their v1 contents because the observation encoding,
`obs_hash`, the action heads and the bots iterate them; under v1 nothing may see a v2 variant.

## Rules

### Data files (new, `data/v2/`)

All JSON, embedded with `include_str!`, `deny_unknown_fields` on every row and file. Every number below
is from the approved v2 rules.

- `units.json`: rows in the v1 `units.json` format for `swordsman` (5, 15, 3, 3, 1, 1, [dash, fortify],
  smithery), `catapult` (8, 10, 4, 0, 1, 3, [stiff], mathematics), `knight` (8, 10, 3.5, 1, 3, 1, [dash,
  fortify, persist], chivalry) in the order cost, health, attack, defense, movement, range, skills,
  unlocked_by.
- `techs.json`: rows in the v1 format: `roads` (2, riding), `free_spirit` (2, riding), `trade` (3, roads),
  `chivalry` (3, free_spirit), `mathematics` (3, forestry), `construction` (3, farming), `smithery` (3,
  mining).
- `site_actions.json`: one row in the v1 format: `burn_forest` (cost 3, tech construction, population 0,
  stars 0). Its effect (forest to field with crop, animal forfeited) is M17's.
- `buildings.json`: rows `{ "building", "cost", "tech", "adjacent", "level_from", "population_per_level",
  "stars_per_level", "max_output", "one_per_city" }`:
  - `sawmill`: 5, mathematics, ["lumber_hut"], "count", 1, 0, null, true
  - `windmill`: 5, construction, ["farm"], "count", 1, 0, null, true
  - `forge`: 5, smithery, ["mine"], "count", 2, 0, null, true
  - `market`: 5, trade, ["sawmill", "windmill", "forge"], "levels", 0, 1, 8, true

  `adjacent` names improvements or buildings; `level_from` is "count" (number of adjacent matches) or
  "levels" (sum of the adjacent buildings' levels); `max_output` caps the output (stars for the market).
- `monuments.json`: rows `{ "monument", "task", "threshold" }`: `park_of_fortune` ("city_level", 5),
  `gate_of_power` ("kills", 10), `eye_of_god` ("lighthouses", 4).
- `rules.json` (v2 scalars): `{ "road_cost": 3, "road_tech": "roads", "road_step_cost": 0.5,
  "connection_population": 1, "disband_tech": "free_spirit", "destroy_tech": "chivalry",
  "disband_refund_divisor": 2, "super_unit_value": 10, "lighthouse_population": 1,
  "explorer_lighthouse_bonus": 3.5, "monument_population": 3, "monument_score": 400 }`.

`Tables::load_for(V2)` loads the v1 files and the v2 files and validates that every variant of the v2
lists appears exactly once in its table (units, techs, site actions, buildings, monuments), that every
tech prerequisite is a tech of the same ruleset with a lower tier, and that `adjacent` names exist.
Malformed or incomplete data is an `Err`, never a panic (as in v1).

### Ruleset flag

- `Config.ruleset` defaults to `V1`. Serialization: the field is omitted when `V1`, so the v1 `Config`
  JSON, and every replay file and test fixture containing it, is unchanged; `"ruleset": "v2"` when `V2`.
  Deserializing a JSON without the field gives `V1`.
- The digest (milestone 7) appends `ruleset` as one byte (`V2` = 1) after the existing `Config` bytes
  only when `V2`; v1 digests are unchanged.
- `new_game` and every rules function read the tables of `config.ruleset` (`Tables::load_for`, shared
  once per ruleset per process as v1 does today).

### v2 tech research

- Under `V2`, Research is legal for every tech of `techs(V2)` not yet researched whose prerequisite is
  researched and whose cost `4 + cities x tier` the player can pay (the v1 rule, with the v2 table).
  Under `V1`, v2 techs never appear in legal actions and Research of one is illegal.
- `TechSet` holds 16 techs: bit `Tech as u16`, so the 7 new techs use bits 9 to 15. v1 sets are unchanged.
- Score: researched v2 techs count `100 x tier` like v1 techs (the existing Technology term).
- Researching a v2 tech in M15 unlocks nothing yet: Train of `Swordsman`, `Catapult`, `Knight`,
  `BurnForest` and every other v2 action stay illegal until their milestones, even under `V2`.

### Hashes

- `rules_hash()` and `rules_hash_for(V1)` return exactly today's value (`a8b52b91eae34503` for the
  current data).
- `rules_hash_for(V2)` = FNV-1a 64 of: the v1 canonical bytes (as today, `RULES_VERSION` included), then
  for each `data/v2/*.json` in file-name order its name `"v2/<file>"`, `\0`, its contents, then
  `RULESET`, `\0`, `v2`.
- `obs_hash` is unchanged (`d5f9b1033cfd55ca` at size 11): the observation does not change in M15.

### Env

`sim_env::Env::new` (and Python `polytopia_sim.Env`) with a `Config` whose ruleset is `V2` returns an
error (Python `ValueError`) naming milestone 20, until the observation supports v2.

## Clarifications (owner: confirm or overrule)

- **V1:** the v2 data lives in separate files under `data/v2/` rather than in the v1 files, so the v1
  `rules_hash` stays exactly as it is and every v1 checkpoint keeps loading without a warning. - Confirmed (owner, 2026-11-07)
- **V2:** `UnitStats.attack` becomes `f32` (the Knight's 3.5). v1's integer attacks convert exactly, so
  every combat result is unchanged. - Confirmed (owner, 2026-11-07)
- **V3:** the signed population type (needed for population loss) is deferred to M17, where population
  loss first happens, to keep M15's changes to earlier tests minimal. - Confirmed (owner, 2026-11-07)
- **V4:** under `V2` in this milestone, researching a v2 tech unlocks nothing yet; each later milestone
  turns on its own content under `V2`, so v2 games are playable (as v1 plus a bigger tech tree) at every
  step. - Confirmed (owner, 2026-11-07)
- **V5:** the observation, action heads and bots keep seeing only the v1 lists until M20 (observation) and
  M19 (bots); `Env` refuses v2 until then, so no RL code can touch a half-built v2. - Confirmed (owner, 2026-11-07)

## Changes to earlier tests

- Allowed only where a type changes: `UnitStats.attack` is now `f32` (comparisons and constructions in
  tests may need `as f32` or float literals) and `Config` has a new field (struct literals need
  `ruleset: Ruleset::V1` or `..Config::default()`). No expected value may change, no test may be removed
  or weakened. Each such edit is listed in the phase A commit message.
- `default_config_json_is_exact` (serde) must still pass unchanged: the v1 JSON omits `ruleset`.

## Tests (phase A guidance)

1. **v1 byte-identity (the core):** `rules_hash_for(V1) == rules_hash() == "a8b52b91eae34503"`;
   `obs_hash(11) == "d5f9b1033cfd55ca"`; the v1 default `Config` JSON is unchanged; a v1 digest of a
   fixed game (an existing golden) is unchanged; all existing Rust and Python tests pass.
2. **Tables v2:** `load_for(V2)` loads; every v2 value equals the table above (units, techs, site action,
   buildings, monuments, v2 rules); a missing, duplicated or unknown v2 row, a prerequisite of equal or
   higher tier, or an unknown `adjacent` name is an `Err` (hand-written malformed JSON through a
   `from_json`-style entry point the implementer adds for v2, named in the stub).
3. **Ruleset plumbing:** `Config` JSON round trip with `V2` contains `"ruleset":"v2"`, without it gives
   `V1`; the v2 digest of a state differs from the v1 digest of the same state only by the appended byte.
4. **Research under V2:** a tier-2 v2 tech (Roads) is legal after Riding and illegal before; a tier-3 one
   (Trade) needs Roads; costs follow `4 + cities x tier`; stars are deducted; the tech set records bits
   9 to 15; score counts `100 x tier`. Under `V1` no v2 tech is ever legal (property test over random
   playouts) and the legal-action lists of random v1 playouts are unchanged (compare with a recorded run).
5. **Nothing else unlocked:** under `V2` with every tech researched, Train of the three new units and
   `BurnForest` are not legal actions.
6. **Hashes:** `rules_hash_for(V2)` is the documented canonical (golden computed independently in the
   test), changes when a `data/v2` byte changes, and differs from v1.
7. **Env refuses v2:** Rust error and Python `ValueError` naming milestone 20.

Mutants to aim for: `ruleset` serialized under v1 (JSON or digest); v2 data mixed into the v1
`rules_hash`; `ALL_TECHS` extended (changes `obs_hash`); a v2 tech researchable under v1; prerequisites
ignored for a tier-3 tech; a TechSet bit collision; v2 unit trainable in M15; attack truncated to an
integer (Knight 3); the v2 tables accepting a duplicate row.

## Gate

`./scripts/check.sh` passes (all existing tests unchanged in value). Plus: `rules_hash` and `obs_hash` of
the build equal the recorded v1 values, and `sim-tools verify` accepts an existing v1 replay file
unchanged. Rules milestone: a `/review 15` phase is recommended (the v1 byte-identity is easy to break
quietly).

## Notes for the implementer

- Append enum variants; never reorder existing ones (their discriminants are bits and indices).
- Keep every v1 code path's behavior identical; prefer reading `config.ruleset` at the few points that
  branch (research legality, table loading) over threading v2 conditions through the rules.
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry (next number Q42).
