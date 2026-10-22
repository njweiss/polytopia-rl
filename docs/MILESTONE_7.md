# Milestone 7: tooling (text renderer, replay log, viewer)

The spec's milestone 7. Depends on milestones 1 to 6 (complete games with outcomes exist). This is
tooling around the simulator; it must not change any game rule.

## Goal

Be able to play a game (random for now), save it as a replay log, verify that the log replays to
bit-identical states, and look at any moment of it: as text in the terminal, or in a self-contained
HTML page with a slider.

## In scope

1. A new crate `crates/sim-tools` (library `sim_tools` + binary `polytopia`).
2. `sim-core` gains `serde` derives on its public data types (nothing else changes there).
3. Text renderer: `render`, `describe_tile`, `action_label`.
4. State digest and the replay log: record, serialize, replay with verification.
5. A random action policy (`random_action`), used by `play-random`; bots are milestone 8.
6. The `polytopia` CLI: `play-random`, `verify`, `view`, `html`.
7. The HTML replay page.
8. `scripts/demo.sh`: builds a sample replay and its HTML page.

## Out of scope (do not build)

Bots/heuristics beyond the random policy, observation or mask encoding, PyO3, ANSI colors, interactive
TUIs, any network access or external web resources, performance work, new dependencies (use only
`sim-core`, `serde`, `serde_json`; no `clap`, no `tempfile`: hand-parse arguments and use
`std::env::temp_dir()` with unique names in tests).

## Changes to sim-core

Add `#[derive(Serialize, Deserialize)]` (keep all existing derives and the existing snake_case renames)
to: `Pos`, `UnitId`, `CityId`, `UnitType`, `Tech`, `SiteAction`, `Reward`, `Action`, `Config`,
`ScoreWeights`, `MapParams`, `Outcome`. Use serde's default representation (externally tagged enums,
newtype structs as their inner value), which fixes this JSON, and tests pin it:
`Action::EndTurn` is `"EndTurn"`; `Action::Move{unit: UnitId(3), to: Pos{row:2,col:3}}` is
`{"Move":{"unit":3,"to":{"row":2,"col":3}}}`; `Action::Train{city: CityId(0), kind: UnitType::Warrior}`
is `{"Train":{"city":0,"kind":"warrior"}}`; `Outcome::Win(1)` is `{"Win":1}`; `Outcome::Draw` is `"Draw"`.
`State` itself is NOT serialized anywhere (replays store seed + actions).

## Public API (fixed so tests can import it)

```rust
// sim_tools::render
pub enum View { Omniscient, Player(PlayerId) }
pub fn render(state: &State, view: View) -> String;
pub fn describe_tile(state: &State, view: View, pos: Pos) -> String;
pub fn action_label(action: &Action) -> String;

// sim_tools::digest
pub fn state_digest(state: &State) -> u64;

// sim_tools::policy
pub fn random_action(state: &State, rng: &mut Rng) -> Action;

// sim_tools::replay
pub struct ReplayLog {                       // Clone, Debug, PartialEq, Serialize, Deserialize
    pub version: u32,                        // 1
    pub seed: u64,
    pub map_params: MapParams,
    pub config: Config,
    pub actions: Vec<Action>,
    pub digests: Vec<String>,                // 16 lowercase hex digits; len == actions.len() + 1
    pub outcome: Option<Outcome>,            // outcome of the final state (None if unfinished)
}
pub enum ReplayError {                       // Debug, PartialEq
    Json(String), BadVersion(u32), LengthMismatch,
    IllegalAction { index: usize },          // index of the offending action
    DigestMismatch { index: usize },         // 0 = the initial state, i+1 = after action i
    OutcomeMismatch,
}
impl ReplayLog {
    pub fn to_json(&self) -> String;                                 // compact JSON, deterministic
    pub fn from_json(s: &str) -> Result<ReplayLog, ReplayError>;     // Json(..) on failure
}
pub fn record_game(seed: u64, map_params: &MapParams, config: &Config, max_actions: usize,
                   policy: &mut dyn FnMut(&State) -> Action) -> ReplayLog;
pub fn replay(log: &ReplayLog) -> Result<State, ReplayError>;        // verifies everything, returns final state
pub fn replay_each(log: &ReplayLog, f: &mut dyn FnMut(usize, &State)) -> Result<(), ReplayError>;
//   calls f(0, initial) then f(i + 1, state after action i); same verification as `replay`

// sim_tools::html
pub fn frames_json(log: &ReplayLog) -> Result<String, ReplayError>;
pub fn html_page(log: &ReplayLog) -> Result<String, ReplayError>;

// sim_tools::cli
pub fn run_cli(args: &[String], out: &mut dyn std::io::Write, err: &mut dyn std::io::Write) -> i32;
```

`src/main.rs` is only: collect `std::env::args().skip(1)`, call `run_cli` with stdout/stderr, exit.

## Rules

### State digest

`state_digest` is FNV-1a 64-bit (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`) over this byte
stream. All integers are fixed-width little-endian; `bool` is one byte 0/1; a fieldless enum is its
declaration index as `u8`; `Option<T>` is a tag byte 0/1 followed by the payload if present; `Vec`s
are a `u32` length then elements. Order: every field of `State` in declaration order, EXCEPT `tables`
(constant rules data) and with `rng` replaced by the next two `u32` outputs of a clone of it. Inside
each struct, fields in declaration order. `config` is included. The digest never mutates the state.

### Replay log

- `digests[0]` is the digest of `new_game_from_seed(seed, &map_params, &config)`; `digests[i+1]` is the
  digest after applying `actions[i]`. Digests are formatted `{:016x}`.
- `record_game` creates the game, repeatedly asks `policy` for an action, applies it (a policy action
  that is not legal is a bug: panic), stops after the game becomes terminal or after `max_actions`
  actions, and fills `digests` and `outcome`.
- `replay` rebuilds the game from `seed`, `map_params`, `config` and applies the actions. Errors, in
  priority order: `BadVersion(v)` if `version != 1`; `LengthMismatch` if `digests.len() !=
  actions.len() + 1`; `DigestMismatch{0}` if the initial digest differs; then for each action in order
  `IllegalAction{index}` if `apply` returns `Err`, then `DigestMismatch{index: i+1}` if its digest
  differs; finally `OutcomeMismatch` if `log.outcome != final_state.outcome`.
- `to_json` output is deterministic (struct field order as declared), so the same log always yields
  the same bytes, and `from_json(to_json(x)) == Ok(x)`.

### Random policy

`random_action(state, rng)`: let `legal = legal_actions(state)` (non-empty if the game is not terminal;
panic on a terminal state). If `EndTurn` is not in `legal` (a reward is pending), return
`legal[rng.gen_range(legal.len() as u32)]` with no `gen_f32` draw (Q7). Let `others` be `legal`
without `EndTurn`. If `others` is empty, return `EndTurn`. Otherwise draw `u = rng.gen_f32()`; if `u < 0.15` return `EndTurn`, else
`others[rng.gen_range(others.len() as u32)]`. Draw order is exactly: the `gen_f32` first, then (only in
the second case) the `gen_range`.

### Text renderer

`render(state, view)` returns the board, a blank line, then the panel; every line has trailing spaces
removed; the string ends with one `\n`.

**Board.** Header line: three spaces, then for each column `c` the text `format!("{c:<4}")`, joined by a
single space. Then one line per row `r`: `format!("{r:>2} ")` followed by the row's cells joined by a
single space. Each cell is exactly 4 characters:

1. `base`: if the view is `Player(p)` and the tile is not explored by `p`, the whole cell is `????`.
   Otherwise, in priority order: improvement (`H` lumber hut, `F` farm, `M` mine); resource (`o` fruit,
   `+` crop, `a` animal, `$` metal); terrain (`.` field, `f` forest, `^` mountain).
   In a `Player(p)` view a crop is hidden (shown as its terrain) unless `p` has Organization and a
   metal is hidden unless `p` has Climbing. `Omniscient` shows everything.
2. `territory`: `0` or `1` = the owner of the city that owns the tile (`owner_city`); `.` if unowned.
3. `site`: `V` village, `K` capital city, `C` any other city, `.` otherwise.
4. `unit`: `W` warrior, `R` rider, `A` archer, `D` defender, `S` super unit; uppercase for player 0,
   lowercase for player 1; `.` if empty.

**Panel.** First line: `turn {turn} | P{current} to move`, followed by `| game over: P{w} wins` or
`| game over: draw` when the state has an outcome. Then one line per player p = 0, 1:

- if the view is `Omniscient` or `Player(p)`:
  `P{p}: stars {stars} | income {income} | score {score} | cities {n} | units {n} | techs {list}`
  where `list` is the researched techs in the `ALL_TECHS` order, lowercase snake_case, joined by `", "`,
  or `-` if none;
- otherwise (the opponent in a player view): `P{p}: score {score}`.

**`describe_tile`**: one line, exact format
`({row},{col}) {terrain} res={res} imp={imp} territory={ter} site={site} unit={unit}` with
`terrain` in `field|forest|mountain`; `res` in `fruit|crop|animal|metal|-` (hidden as `-` in a player
view lacking the tech, as above); `imp` in `lumber_hut|farm|mine|-`; `ter` = `P0|P1|-`; `site` =
`village|city#{id}(P{owner})|capital#{id}(P{owner})|-`; `unit` =
`#{id} P{owner} {kind} hp {health}/{max} {phase}{vet}{kills}` where `kind` is the snake_case type,
`phase` is `Fresh|Moved|Attacked|Done`, `vet` is `veteran` or empty, and `kills` is `kills {n}` when
`n > 0` else empty; `-` when empty. For an unexplored tile in a player view the whole line is
`({row},{col}) unexplored`.

**`action_label`**: `Move u{id} -> ({r},{c})`, `Attack u{id} -> ({r},{c})`, `Capture u{id}`,
`Recover u{id}`, `Promote u{id}`, `Train c{id} {kind}`, `Site ({r},{c}) {site_action}`,
`Research {tech}`, `Reward c{id} {reward}`, `End turn` (kinds, site actions, techs, rewards in
snake_case).

### CLI

Exit codes: 0 success, 1 a verify/replay failure, 2 usage error. Messages go to `out`, errors/usage to
`err`.

- `play-random --seed N --out FILE [--turn-cap K] [--max-actions M] [--full-observability]`: defaults
  `K = 50`, `M = 5000`, default `MapParams`. Plays with `random_action` driven by an `Rng` seeded with
  `N` XOR `0x9e3779b97f4a7c15` (so the policy stream differs from the map stream), writes the log, and
  prints `wrote FILE: {n} actions, outcome {Win(p)|Draw|unfinished}`.
- `verify FILE`: `replay`; prints `OK: {n} actions, outcome {..}`; on error prints the error and exits 1.
- `view FILE [--step I | --all] [--player P]`: replays and prints `render` of the state after `I` actions
  (default: the last state; `I > n` is a usage error) with `Omniscient` unless `--player P` (0 or 1);
  `--all` (instead of `--step`) prints every state, each preceded by the line `== step {i}: {action_label of action i-1} ==`
  (step 0 has the header `== step 0: start ==`).
- `html FILE OUT.html`: writes `html_page`.
- No arguments, an unknown subcommand, or missing/invalid options: usage on `err`, exit 2.

### HTML page

`html_page` returns a single self-contained document: inline CSS and JavaScript, no external
resources and no `http://`/`https://` strings anywhere. It embeds the data as
`const REPLAY = <frames_json>;`. `frames_json` is
`{"size": n, "labels": [...], "frames": [...]}` with `labels.len() == actions.len()` (`labels[i]` is
`action_label(actions[i])`) and `frames.len() == actions.len() + 1`. Each frame is an object with at
least: `turn`, `current`, `outcome` (`null`, `"p0"`, `"p1"` or `"draw"`), `players` (array of two
objects with `stars`, `score`, `cities`, `units`), `tiles` (row-major array of objects with enough to
draw: terrain, resource, improvement, village flag, city id or null, territory owner or null, unit id
or null), `units` (id, kind, owner, row, col, health, max_health, phase, veteran) and `cities` (id,
owner, row, col, level, capital). The page shows the board (CSS grid or canvas), a slider over frames,
play/pause, previous/next buttons, left/right arrow keys, the action label that led to the current
frame, and the player panel. All drawing is done by the page from the frame data; it contains no game
rules.

### scripts/demo.sh

Builds in release, plays seed 1 with `--turn-cap 20`, verifies it, and writes
`target/demo/replay.json` and `target/demo/replay.html`, printing both paths. Exits non-zero on any
failure.

## Clarifications to the spec (owner: confirm or overrule)

- **G1: the digest** covers all state except the constant tables, uses the next two rng outputs in place
  of private rng fields, and includes `config`. It identifies a state for verification; it is not a
  security hash. - Confirmed
- **G2: replay version 1,** digests as hex strings (so JavaScript never sees a lossy 64-bit number). - Confirmed
- **G3: step numbering** in `view`/`--all`: step `i` is the state after `i` actions (0 = initial). - Confirmed
- **G4: views follow the observation rules.** The player view hides unexplored tiles, hidden crops and
  metal, and the opponent's stars/techs/income; it shows live state on explored tiles and the public
  opponent score.
- **G5: improvements hide the resource symbol** on the board (an improvement outranks a resource); - Confirmed
  use `describe_tile` for the full detail.
- **G6: the random policy's EndTurn weight is 0.15** and games stop at `max_actions` even if
  unfinished. It exists for demos and tests; the milestone 8 bots replace it. - Confirmed
- **G7: hand-written argument parsing,** no CLI crate. - Confirmed

## Changes to earlier tests

None expected: the sim-core change is derives only. If serde derives force any test-side change,
list it in the report.

## Tests (phase A guidance)

Golden strings are written by hand from the formats above, never captured from the implementation.

1. **serde:** the exact JSON strings above for each listed value; round trips of every `Action`
   variant, `Config` (including `ScoreWeights`), `MapParams` with `f32` fields exactly equal after a
   round trip (include `1.0/3.0`); `Outcome`.
2. **digest:** equal states built two independent ways give equal digests; `clone` equal; digests do not
   depend on `tables` identity. Field sensitivity: a table of single-field edits covering every `State`
   field, and the nested fields of units/cities/tiles/players, each changes the digest (use an
   exhaustive destructuring `let State { .. } = &s;` of every struct in the test so adding a field
   breaks compilation until the test is updated). Differences only in rng state change it. Ordering
   matters (swap two units). No panics along random playouts; consecutive states of a playout have
   different digests.
3. **render goldens:** small hand-drawn boards (size 5 and one 7) with `StateBuilder`: every symbol of
   the legend (all terrain, resources, improvements, village/city/capital, both unit cases, territory
   0/1/none), `Omniscient`, `Player(0)`, `Player(1)` views with fog `????`, crop hidden without
   Organization and metal hidden without Climbing, panel lines with hand-computed income and score,
   opponent line reduced to the score in a player view, techs `-` and multiple, terminal outcome
   line, trailing-space and final-newline rules. `describe_tile` for each field combination and for an
   unexplored tile. `action_label` for every `Action` variant.
4. **replay:** record many random games (varied seeds, small `turn_cap` values, `full_observability`
   both ways) until terminal; `to_json`/`from_json` round trip; `replay` returns a state equal to the one
   `record_game` ended with and to an independent re-simulation done in the test with `sim_core`
   calls; `replay_each` visits `n + 1` states in order and each equals the independent one. Tamper
   tests with exact expected errors: edited action (becomes illegal -> `IllegalAction{index}`, or still
   legal -> `DigestMismatch{index+1}`), changed seed (`DigestMismatch{0}`), altered digest string,
   truncated/extended digests (`LengthMismatch`), `version = 2` (`BadVersion(2)`), altered `outcome`
   (`OutcomeMismatch`), garbage JSON (`Json`), an action appended after the game ended
   (`IllegalAction{index: n}`). Byte-for-byte determinism of `to_json`.
5. **policy:** always returns an action in `legal_actions`; same seed -> same sequence; both `EndTurn` and
   other actions occur in a long game; `EndTurn` frequency near 0.15 when others exist; panics on a
   terminal state; every game with a small `turn_cap` terminates well within `max_actions`.
6. **CLI** (via `run_cli` with `Vec<u8>` buffers and files under unique `std::env::temp_dir()` paths,
   removed afterwards): `play-random` then `verify` succeed and print the exact messages; a corrupted file
   makes `verify` exit 1 with the error; `view --step i` equals `render` of an independently replayed
   state, with and without `--player`; `view --all` has `n + 1` frames with the exact headers;
   `html` writes a file; every usage error path returns 2 and writes to `err`, not `out`.
7. **HTML:** `frames_json` parses; `frames.len() == n + 1`, `labels.len() == n`; frame 0 and the last
   frame's units/cities/turn/players equal values from the independently replayed states; the page
   contains `const REPLAY =`, a slider (`<input type="range"`) and no `http://`/`https://`; the embedded
   JSON equals `frames_json`; size stays reasonable (a 50-turn random game under 10 MB; Q8).
   The visual quality is checked by the owner opening `scripts/demo.sh`'s output; if a headless
   browser is available the implementer should additionally load the page and report any console
   errors (informational, not part of the gate).
8. **demo.sh:** runs to completion in the review phase; its outputs exist.

Mutants to aim for: digest ignoring `explored`, units order, rng, a unit field, `config`; digest index
off by one; replay skipping digest or outcome checks; wrong error priority; renderer swapping owner case
or territory vs unit owner, showing hidden crops/metal or opponent stars, wrong fog cell, wrong header
widths, trailing whitespace; `view --step` off by one; policy ignoring the draw order or EndTurn
weight; serialization dropping a field; HTML frame count off by one.

## Gate

`./scripts/check.sh` passes (including the new crate), `./scripts/demo.sh` succeeds, and the owner has
opened `target/demo/replay.html` and stepped through a game.

## Notes for the implementer

- Do not touch game rules in `sim-core`; only add the serde derives.
- No `HashMap`/`HashSet` where output order matters (JSON, labels, panels).
- If the tests contradict this brief or the spec, stop and add a QUESTIONS.md entry.
