# Milestone 8: heuristic bots and a tournament

The spec's milestone 8. Depends on milestones 1 to 7. These bots are the first opponents for the RL
agent and the stress drivers for the simulator, so they must be deterministic, always legal, and must
not use information a real player would not have.

## Goal

Two scripted strategies, a Rush bot and an Economy bot, plus a random baseline, that play complete games
through a fog-restricted `PlayerView`; and a tournament tool that plays them against each other and
reports win rates and development statistics.

## Decisions already made (owner)

- Bots see a fog-restricted view, not the full state.
- Hard gate: each of Rush and Economy beats the random policy in at least 80% of side-swapped games.
  Rush vs Economy balance is reported, not gated (the implementer tries to keep each side between 25%
  and 75% only by choosing among the options this brief leaves open; it must not change a rule below
  to get there. If it cannot, it reports the numbers in its final message and in `docs/QUESTIONS.md`).

## In scope

1. `sim-core`: new module `view` with `PlayerView` (the only thing bots may read).
2. New crate `crates/bots` (library `bots`): the `Bot` trait, `RandomBot`, `RushBot`, `EconomyBot`, an
   epsilon wrapper, `make_bot`, and the match runner/statistics.
3. `sim-tools`: new CLI subcommands `play` and `tournament`, and `scripts/bots_gate.sh`.

## Out of scope (do not build)

Learning, search/lookahead, any use of `State` by bots, observation tensors or masks (milestone 9),
tuning beyond what is written below, new dependencies (only `sim-core`, `serde`, `serde_json`; `bots`
depends on `sim-core` only; `sim-tools` now also depends on `bots`).

## Public API (fixed so tests can import it)

```rust
// sim_core::view   (TechSet gains Clone, Copy, PartialEq, Eq, Debug if it lacks them)
pub struct CityView { pub id: CityId, pub owner: PlayerId, pub pos: Pos, pub level: u32, pub is_capital: bool }
pub struct UnitView { pub id: Option<UnitId> /* Some only for the viewer's own units */,
                      pub kind: UnitType, pub owner: PlayerId, pub pos: Pos,
                      pub health: u32, pub max_health: u32, pub veteran: bool, pub phase: Phase }
pub struct VisibleTile { pub terrain: Terrain, pub resource: Option<Resource>, pub improvement: Option<Improvement>,
                         pub village: bool, pub city: Option<CityView>, pub territory: Option<PlayerId>,
                         pub unit: Option<UnitView> }          // all: Clone, Debug, PartialEq, Eq
pub struct PlayerView<'a> { /* private: &State + player */ }
impl<'a> PlayerView<'a> {
    pub fn new(state: &'a State, player: PlayerId) -> PlayerView<'a>;
    pub fn player(&self) -> PlayerId;      pub fn size(&self) -> u8;
    pub fn turn(&self) -> u32;             pub fn turn_cap(&self) -> u32;
    pub fn current(&self) -> PlayerId;     pub fn is_terminal(&self) -> bool;
    pub fn is_explored(&self, p: Pos) -> bool;
    pub fn tile(&self, p: Pos) -> Option<VisibleTile>;        // None if unexplored
    pub fn stars(&self) -> u32;            pub fn techs(&self) -> TechSet;
    pub fn income(&self) -> u32;           pub fn score(&self, player: PlayerId) -> u32;   // both are public
    pub fn my_units(&self) -> Vec<&'a Unit>;                  // ascending id
    pub fn my_cities(&self) -> Vec<&'a City>;                 // ascending id
    pub fn visible_units(&self) -> Vec<UnitView>;             // every unit on an explored tile, row-major
    pub fn visible_cities(&self) -> Vec<CityView>;            // own cities + enemy cities on explored tiles, by id
    pub fn tables(&self) -> &'a Tables;
    pub fn legal_actions(&self) -> Vec<Action>;               // legal_actions(state) if it is this player's turn, else empty
    pub fn reachable_tiles(&self, unit: UnitId) -> Vec<Pos>;  // own units only, panics otherwise
    pub fn pending_reward_city(&self) -> Option<CityId>;      // own city with a pending reward
}

// bots
pub trait Bot {                                               // bots are stateless: a pure function of (view, rng)
    fn name(&self) -> &'static str;
    fn choose(&self, view: &PlayerView, rng: &mut Rng) -> Action;   // must be in view.legal_actions()
}
pub enum BotKind { Random, Rush, Economy }                    // Copy, Eq, Debug; name() -> "random"|"rush"|"economy"
pub fn make_bot(kind: BotKind, epsilon: f32) -> Box<dyn Bot>; // epsilon in [0, 1]
pub fn bot_rng(seed: u64, player: PlayerId) -> Rng;           // Rng::new(seed ^ SALT[player]); SALT[0]=0x9e3779b97f4a7c15, SALT[1]=0xc2b2ae3d27d4eb4f

pub struct Snapshot { pub turn: u32, pub cities: [u32; 2], pub units: [u32; 2], pub stars: [u32; 2], pub score: [u32; 2] }
pub struct GameResult { pub seed: u64, pub bots: [BotKind; 2] /* by player id */, pub outcome: Outcome,
                        pub turns: u32, pub scores: [u32; 2], pub captured_capital: bool,
                        pub snapshots: Vec<Snapshot> }          // snapshots at turns 10, 20, 30 (those that were reached)
pub fn play_game(seed: u64, bots: [BotKind; 2], epsilons: [f32; 2], map: &MapParams, config: &Config) -> GameResult;
pub struct MatchStats { pub a: BotKind, pub b: BotKind, pub games: u32, pub a_wins: u32, pub b_wins: u32, pub draws: u32,
                        pub a_capture_wins: u32, pub b_capture_wins: u32, pub mean_score_diff: f64 /* a - b */,
                        pub checkpoints: Vec<CheckpointStats> }
pub struct CheckpointStats { pub turn: u32, pub games: u32,
                             pub a: PlayerMeans, pub b: PlayerMeans }   // means over games that reached `turn`
pub struct PlayerMeans { pub cities: f64, pub units: f64, pub stars: f64, pub score: f64 }
impl MatchStats { pub fn from_results(a: BotKind, b: BotKind, results: &[GameResult]) -> MatchStats;
                  pub fn win_rate_a(&self) -> f64; pub fn win_rate_b(&self) -> f64; }   // wins / games (draws count as no win)
pub fn run_match(a: BotKind, b: BotKind, seeds: std::ops::Range<u64>, map: &MapParams, config: &Config) -> MatchStats;
//   each seed is played twice: a as player 0, then b as player 0 (same map, seats swapped), epsilons 0

// sim_tools::cli  new subcommands (see below)
```

`PlayerView` must never hand out `&State`, an unexplored tile's contents, the opponent's stars, techs,
income, unit ids, kills, origins, last directions, or an enemy city's population/buildings, and must
never expose the rng.

## Rules

### PlayerView visibility

- `tile(p)` is `None` if the viewer has not explored `p`. Otherwise the live tile with the same hiding
  as the milestone 7 player view: a crop is hidden (`resource = None`) unless the viewer has Organization,
  a metal unless it has Climbing. `territory` is the owner of the tile's `owner_city`.
- `unit` and `city` on a tile are present only for explored tiles. `UnitView.id` is `Some` only for the
  viewer's own units.
- `visible_units`/`visible_cities` follow the same rule; own cities are always listed.
- `legal_actions` is empty unless `view.player() == current` and the game is not terminal.

### Common bot machinery (used by Rush and Economy)

- `dist(a, b)` is Chebyshev distance. "Row-major order" breaks ties on positions.
- **Vision gain** of tile `t`: the number of in-bounds tiles within Chebyshev radius `r` of `t` that the
  viewer has not explored, where `r = 2` if `t` is a mountain and 1 otherwise.
- **`greedy_step(unit, target)`**: among `view.reachable_tiles(unit)` with `dist(t, target) <
  dist(unit.pos, target)`, pick the one with the smallest `dist(t, target)`, ties smallest `(row, col)`;
  `None` if there is none.
- **`explore_step(unit)`**: among `reachable_tiles(unit)` with vision gain > 0, pick the largest gain,
  ties smallest `(row, col)`. If no reachable tile has vision gain > 0 (Q10): let `f` be the nearest
  explored tile (by `dist` from the unit, any occupant or terrain) with vision gain > 0, ties
  smallest `(row, col)`, and return `greedy_step(unit, f)`; `None` if there is no such tile or
  `greedy_step` gives `None`.
- **Attack score** for a legal `Attack{unit, target}`: let `d` be the visible enemy unit at `target`.
  With defense bonus fixed at `1.0` (enemy techs are hidden), compute
  `sim_core::combat::compute_damage(attacker, defender, 1.0)` from the attacker's real health and the
  defender's visible health, both `max_health`s and the table attack/defense. `dealt = to_defender`;
  `taken = to_attacker` if `dealt < d.health` and `dist(attacker, target) <= d's range stat`, else 0.
  Score = `1000 + dealt` if `dealt >= d.health` (a kill), else `dealt - taken` as a signed integer.
  An attack is acceptable if its score is a kill, or its (signed) score `>= T` (the bot's threshold).
  The best attack is the highest score; ties: lowest attacker id, then target row-major.
- **Reward choice:** the bot's preference table gives an ordered list per menu level; take the first one
  that is in the pending city's legal `ChooseReward` actions, else the first legal `ChooseReward`.
- **Low health:** an own unit is "hurt" if `health * 2 < max_health`.

### Rush bot

Evaluate top to bottom, return the first rule that yields a legal action:

1. **Pending reward:** preferences L2 `Explorer`; L3 `Resources`; L4 `PopulationGrowth`; L5+ `SuperUnit`.
2. **Promote:** the legal `Promote` of the lowest unit id.
3. **Capture:** the legal `Capture` of the lowest unit id.
4. **Attack** with threshold `T = 0`.
5. **Train:** for cities in ascending id, the first legal `Train` of a Rider if any Rider train is legal for
   that city, else a Warrior.
6. **Research** `Riding` if that action is legal.
7. **Site actions:** the first legal `Site` of kind `HarvestFruit`, else of kind `HarvestAnimal`
   (tile row-major).
8. **Recover:** the lowest-id own unit that is hurt, has a legal `Recover`, and has no visible enemy unit
   within distance 2.
9. **Move:** for own Fresh units in ascending id, the first that has a move: its target is the known
   enemy capital if one is visible, else the nearest (ties: row-major position) visible enemy city or
   enemy unit; if there is a target, `greedy_step`, and if that gives `None` or there is no target,
   `explore_step`. Return `Move{unit, to}` for the first unit that yields a tile.
10. **EndTurn.**

### Economy bot

1. **Pending reward:** preferences L2 `Workshop`; L3 `Resources`; L4 `PopulationGrowth`; L5+ `Park`.
2. **Promote.** 3. **Capture** (as above).
3. **Attack** with threshold `T = 3`.
4. **Site actions:** the first legal `Site` in this kind order: `Farm`, `Mine`, `HarvestFruit`,
   `HarvestAnimal`, `LumberHut` (tile row-major within a kind).
5. **Research:** the first legal `Research` in the order Hunting, Farming, Climbing, Mining, Forestry,
   Riding, Strategy, Archery.
6. **Train a Warrior** (cities ascending) while `my_units().len() < 2 * my_cities().len()`.
7. **Recover** (as above).
8. **Move:** for own Fresh units in ascending id, the first that has a move: target = the nearest
   (ties row-major) known village tile with no unit on it; if there is a target, `greedy_step`; if that
   gives `None` or there is no target, `explore_step`.
9. **EndTurn.**

### Random bot and epsilon wrapper

`RandomBot::choose` replicates `sim_tools::policy::random_action` exactly (EndTurn with probability 0.15
when other actions exist; `gen_f32` first, then `gen_range`), but takes its actions from
`view.legal_actions()`. `make_bot(kind, epsilon)` with `epsilon > 0` wraps the bot: first
`rng.gen_f32()`; if it is `< epsilon`, return a `RandomBot` action drawn from the same rng; else the
wrapped bot's action. With `epsilon == 0` no draw is made (so `make_bot(k, 0.0)` is the bare bot).

### Progress and termination

Every non-EndTurn action a bot chooses must change the state in a way that cannot repeat forever (it
consumes a unit's phase, stars, a resource, a reward, a promotion, ...). From any reachable state a bot
reaches `EndTurn` within 200 of its own actions in one turn.

### Match running and statistics

- `play_game` creates `new_game_from_seed(seed, map, config)`, gives each player its own
  `bot_rng(seed, player)`, and loops: the bot of `state.current` chooses from `PlayerView::new(&state,
  current)` and the action is applied, until the state is terminal. `turns` is the final `state.turn`.
  `captured_capital` is true iff the game ended by capital capture (`outcome` is a `Win` and the turn cap
  was not what ended it: detect by the loser still owning its capital city / by the winner owning both
  capitals; tests pin this). `snapshots` record, at the first moment `state.turn == t && state.current == 0`
  for `t` in `{10, 20, 30}`, the per-player city count, unit count, stars and score.
- `run_match(a, b, seeds, ..)` plays each seed twice (a as player 0, then b as player 0). `from_results`
  attributes each result to a or b through `result.bots`. `a_wins + b_wins + draws == games`.
  `checkpoints` has one entry per snapshot turn that at least one game reached; means are over the
  games that reached it.

### CLI additions (sim-tools)

- `play --bots A,B --seed N --out FILE [--turn-cap K] [--max-actions M]`: `A` and `B` are
  `random|rush|economy` (player 0, player 1). Records a replay log using the bots (same
  rng derivation as `play_game`) and prints `wrote FILE: {n} actions, outcome {..}`.
- `tournament --bots LIST [--games N] [--turn-cap K] [--seed S] [--gate-min R] [--save-replays DIR --save-count C]`:
  `LIST` is a comma list of at least two kinds; defaults `N = 100` (seeds per pair, so `2N` games per
  pair), `K = 50`, `S = 0`. For every pair `(x, y)` with `x` before `y` in the list it runs
  `run_match(x, y, S..S+N)` and prints:

  ```Rust
  tournament: {N} seeds per pair ({2N} games), turn cap {K}
  {x} vs {y}: {x} {xw} win, {y} {yw} win, {d} draw | by capture {xc}/{yc} | mean score diff {sd:+.1}
    turn {t}: {x} cities {..} units {..} stars {..} score {..} | {y} cities {..} units {..} stars {..} score {..}
  ```

  (one `turn` line per checkpoint, numbers with one decimal, `{x}`/`{y}` the bot names). With `--gate-min R`:
  exit 1 and print `GATE FAILED: ...` if, for any pair involving `random` and one of `rush`/`economy`, that
  bot's win rate is below `R`; otherwise exit 0. `--save-replays DIR --save-count C` writes the replay logs
  (milestone 7 format) of the first `C` games of each pair as `DIR/{x}-vs-{y}-{i}.json`. Usage errors exit 2.
- `scripts/bots_gate.sh`: release build, then
  `polytopia tournament --bots random,rush,economy --games 100 --turn-cap 30 --gate-min 0.8 --save-replays target/bots --save-count 2`.

## Clarifications to the spec (owner: confirm or overrule)

- **H1: bots are stateless** (pure functions of the view and their rng stream). Exploration therefore
  is "move toward unexplored space", not a remembered plan. - Confirmed
- **H2: bots estimate fights with defense bonus 1.0** since enemy techs are hidden; they may
  underestimate forest/mountain/city bonuses. - Confirmed
- **H3: a "win" in the gate** is a win in the outcome (capture or higher score at the cap); draws are
  not wins. - Confirmed
- **H4: `RandomBot` duplicates `random_action`'s logic** and a parity test in `sim-tools` checks they agree. - Confirmed
- **H5: the Explorer reward is preferred by Rush** (to find the enemy), the Super Unit at level 5.
  Economy never takes Explorer or Super Unit. - Confirmed
- **H6: the rules above are the specification of the bots.** Behavior changes are spec changes and need
  the owner's approval; tuning thresholds is not part of this milestone. - Confirmed

## Changes to earlier tests

`TechSet` derives may change (Clone/Copy/PartialEq/Debug) and `sim-tools` usage text lists the new
subcommands; no behavior changes expected. List anything else in the report.

## Tests (phase A guidance)

Golden expectations are hand-derived from the rules above on small boards; statistics checks use
independent counting.

1. **PlayerView:** per-field goldens on a scripted board: unexplored tiles are `None`; crop/metal hiding
   with/without Organization/Climbing; enemy unit/city on an explored tile visible without id/internals,
   own unit with id; enemy unit on an unexplored tile absent; `legal_actions` empty when it is not the
   viewer's turn; `income`, `score` for both; `pending_reward_city`.
   **No-cheating (metamorphic) property:** build many random states (playout snapshots) and for each make
   a twin that differs only in hidden information: contents of unexplored tiles (terrain, resources,
   units, villages, cities incl. adding/removing units there), the opponent's stars/techs, enemy unit
   kills/origin/last direction, enemy city population/buildings, the state's rng and `next_unit_id`.
   Every `PlayerView` method must return equal results on the twins, and every bot must choose the same
   action from the twins with equal bot rngs (also with a few random epsilons).
2. **Bot rule goldens** (one scripted board per rule per bot, written from the rule text): each reward
   preference per level for both bots; Promote and Capture priority; attack choice with
   kill/score/tie-breaks and the T=0 vs T=3 difference (an attack with advantage 1 is taken by Rush but
   not Economy); Rush Rider-vs-Warrior training, Riding research, harvest kinds, recover condition, target
   priority (capital > nearest city/unit) and the greedy/explore fallbacks incl. tie-breaks;
   Economy site-action kind order, research order, train rule, village target and fallbacks; both end
   the turn when nothing applies. Each case asserts the exact `Action`.
3. **Bot properties:** over many random games (all bot pairings, epsilon 0 and > 0): the chosen action is
   always in `legal_actions`; same seed -> same game; the 200-action bound per turn; `bot_rng` values;
   `make_bot` epsilon wrapper draw order and the `epsilon == 0` no-draw case.
4. **Match runner and stats:** `from_results` on hand-made `GameResult`s (win attribution through
   `bots`, capture wins, mean score diff sign, checkpoint means over only the games that reached the
   turn, win rates); `play_game` determinism; captured_capital correctness on a scripted capture game and
   a cap-ended game; seat swapping (each seed twice with swapped bots); snapshot turns.
5. **Behavior floor (debug-friendly size):** Rush and Economy each win at least 75% of 40 games (20 seeds
   x 2 seats, turn cap 30) against `random`. The full 200-game 80% gate is `scripts/bots_gate.sh`.
6. **sim-tools:** `play` writes a log that `verify` accepts; `tournament` output matches the exact format
   built from an independent `run_match` call (small N, K); `--gate-min 1.01` exits 1 and prints
   `GATE FAILED`, `--gate-min 0.0` exits 0; `--save-replays` writes the right file names and each file
   verifies; usage errors exit 2. Parity test: `RandomBot` and `random_action` choose the same action
   from the same states with equal rngs.

Mutants to aim for: view leaking unexplored data, enemy ids/stars/techs or the rng; hidden-crop rule
inverted; rule order swapped in a bot; wrong preference at a level; threshold `T` swapped; attack
tie-break wrong; greedy_step moving away or not strictly closer; explore ignoring the mountain radius;
Rush targeting the nearest city before the capital; Economy training without the unit bound; epsilon
draw made when 0; seats not swapped; captured_capital wrong; checkpoint means over all games.

## Gate

`./scripts/check.sh` passes, and the owner has run `./scripts/bots_gate.sh` (it must exit 0) and read
the printed table. The implementer pastes that table in the final report.

## Notes for the implementer

- Do not give bots a way to reach `State`. If a rule needs information the view lacks, stop and ask.
- No `HashMap`/`HashSet` where order can affect a choice.
- If the tests contradict this brief or the spec, stop and add a QUESTIONS.md entry. If a bot fails the
  80% floor, do not edit the rules: report the numbers.
