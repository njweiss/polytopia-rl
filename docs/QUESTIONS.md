# Open questions from agents

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q1: Starting value of the turn counter

- Needed for: milestone 3, "new_game" rule 6 and C10; spec "Observation > Global features"
- Ambiguity: the brief's new_game rule 6 says `turn = 1`, but C10 (and the spec) say the round
  counter starts at 0 and increments when player 1 ends.
- Options: A) `turn = 0` at the start (C10, spec). B) `turn = 1` (rule 6).
- Agent's pick and why: A, it matches C10 and the latest spec text. The tests encode A
  (`setup.rs::globals_and_rng`, `actions.rs::end_turn_alternates_players_and_counts_rounds_from_zero`).
- Owner answer: A (`turn = 0`), given during phase A on 2026-10-21. Rule 6 in docs/MILESTONE_3
  still says `turn = 1` and should be updated.

## Q2: Tile actions after D4 was overruled (Farm, Mine, Lumber hut, Clear forest)

- Needed for: milestone 4, "Site actions" table and D4; spec "Data tables > Tile actions" notes
- Ambiguity: D4 says "Overruled, see spec", but the brief's site-action table still has the old
  effects. The spec notes say lumber hut, farm and mine "do not consume forest, crop, or mountain"
  and only one improvement per tile; lumber hut may be built on an animal; clear forest may be done
  on an animal and forfeits it. Not covered: whether "mountain" for mines means the metal stays;
  whether an existing lumber hut blocks Clear forest; whether an animal under a lumber hut can
  still be harvested.
- Options: A) Farm keeps `Crop`, Mine keeps `Metal` (improvement set, resource unchanged); Lumber
  hut needs forest + no improvement (an animal stays); Clear forest needs forest + no improvement,
  sets Field and `resource = None`; harvests need only their resource (so hunting under a hut is
  legal and leaves the hut). B) the brief's table as written (Farm/Mine consume the resource; hut
  and clear forest need `resource == None`).
- Agent's pick and why: A, the spec is the newer source and D4 points to it. The tests encode A
  (`economy.rs`: `farm_golden_keeps_the_crop`, `mine_golden_keeps_the_metal`,
  `lumber_hut_on_an_animal_keeps_the_animal`, `clear_forest_forfeits_the_animal`,
  `an_improvement_blocks_a_second_one`; `common::exp_site_legal` / `exp_after`). The brief's
  table should be updated to match.
- Owner answer: A, given after phase A on 2026-10-21. The brief's site-action table in
  docs/MILESTONE_4 should be updated to match.

## Q3: `Tables::from_json` signature and reward-menu validation

- Needed for: milestone 4, "Public API additions" ("from_json/load must now also take the two new
  JSON files") and test item 7
- Ambiguity: the argument order is not written, nor what counts as a "missing" or "duplicated"
  reward menu, nor what `Tables::reward_menu(level)` returns below level 2.
- Options: argument order `(units, techs, site_actions, rewards)`; menus must cover every level
  from 2 up exactly once (a gap, an overlap or no open-ended last menu is `Err`); unknown names and
  missing keys are `Err`. Below level 2 is left to the implementer (no test calls it).
- Agent's pick and why: as listed; it mirrors the milestone 1 rule "every entry exactly once".
  Tests: `tables.rs::reward_menus_must_cover_every_level_from_2_up_exactly_once`,
  `duplicate_or_missing_site_actions_are_rejected`, `malformed_site_or_reward_json_is_an_error`.
- Owner answer: approved as listed, given after phase A on 2026-10-21.

## Q4: The Explorer's 0.5 visited penalty can never change a choice

- Needed for: milestone 5, "Explorer walk" step 4, and test item 1 ("the 0.5 visited penalty
  flips a choice"); spec "Explorer walk" step 3
- Ambiguity: every visited tile was revealed radius 1 when the explorer stood on it (step 0 for
  the start, E1; step 7 for each move), so `fog(t) = 0` for every visited tile and no visited tile
  is ever a candidate (step 3 needs `fog(t) >= 1`). The penalty only ever applies to tiles that are
  not candidates, so it never changes the target. A golden where it "flips a choice" cannot be
  drawn, and the mutants "penalty dropped" and "penalty applied to the wrong tile" (all candidates
  are unvisited, so a penalty on every candidate keeps their order) are equivalent: no test can
  catch them.
- Options: A) keep the rule as written (a harmless no-op; tests do not cover it). B) change the
  rule so it can matter, e.g. penalize candidates adjacent to a visited tile, or count fog with a
  larger radius than the explorer's vision.
- Agent's pick and why: A. The walk's observable behavior is fully specified without it, and
  "backtracks only when nothing else is open" is already carried by the wander rule (E3). No test
  depends on the choice; `common::exp_explorer_walk` implements the penalty as written.
- Owner answer: A on 2026-10-21

## Q5: The Explorer differential test cannot compare paths

- Needed for: milestone 5, test item 2 ("comparing path, mask and final rng state")
- Ambiguity: the fixed API `explorer_walk(state: &mut State, city: CityId)` returns nothing, and
  the explorer never appears in the state, so the path is not observable.
- Options: A) compare the whole state after the walk (owner's mask and rng state included) and
  check paths only through the mask they reveal. B) change the API to return the path
  (`-> Vec<Pos>`).
- Agent's pick and why: A, the API is fixed by the brief. The goldens use long corridors and
  mostly fogged boards so the revealed mask pins the path, and the rng state pins the number and
  bounds of the draws. Tests: `explorer.rs::explorer_walk_matches_an_independent_replay` and every
  golden there.
- Owner answer: A on 2026-10-21

## Q6: Digest byte stream details the brief leaves open

- Needed for: milestone 7, "State digest" (G1) and test item 2
- Ambiguity: three encodings are not written down. (1) `players: [PlayerState; 2]` is a fixed-size
  array, not a `Vec`: with or without a `u32` length prefix? (2) `TechSet` has one private `u16`
  field: is it hashed as that `u16`, and which bit is which tech? (3) `Outcome` is not a fieldless
  enum: how is `Win(p)` encoded?
- Options: A) no length prefix for the array; `TechSet` = its `u16` LE with bit `i` set for the
  `i`-th declared `Tech` (bit = `Tech as u16`, as the type's own doc says); `Outcome` = variant index
  as `u8` (`Win` 0, `Draw` 1) followed by the `Win` payload `u8`. B) any other choice.
- Agent's pick and why: A, it is what a field-by-field encoder writes naturally. The tests pin (1)
  and (2) exactly (`digest.rs::digest_is_fnv1a_over_the_brief_byte_stream`,
  `digest_matches_the_reference_along_playouts`, and every replay/CLI test that compares log digests
  with `common::my_digest`). (3) is only checked by property tests (the three outcomes give three
  different digests); terminal states are excluded from the exact comparison, but the replay/CLI
  tests do compare the digests of finished games, so they assume A for (3) as well.
- Owner answer: A on 2026-10-21

## Q7: The random policy returns an illegal EndTurn when a reward is pending

- Needed for: milestone 7, "Random policy" (G6) and test item 5 ("always returns an action in
  `legal_actions`")
- Ambiguity: with a pending reward `legal_actions` is only the city's `ChooseReward` options, so
  `others` is non-empty and the rule as written returns `EndTurn` 15% of the time, which is illegal.
  `record_game` then panics. In the throwaway reference this broke about 60% of default-config games
  (35 of 60 seeds).
- Options: A) if `EndTurn` is not in `legal`, return `legal[rng.gen_range(legal.len())]` with no
  `gen_f32` draw; otherwise the rule as written. B) always draw `gen_f32` first and fall through to
  `others[gen_range]` when `u < 0.15` but `EndTurn` is illegal. C) something else.
- Agent's pick and why: A. It is the smallest change, and it keeps the brief's draw order exactly
  whenever `EndTurn` is legal. The tests encode A (`common::my_random_action`;
  `policy.rs::pending_reward_never_returns_end_turn`, `default_games_with_rewards_stay_legal`, and
  every test that replays the policy, including the CLI log tests). The brief's "Random policy"
  paragraph should be updated.
- Owner answer: A on 2026-10-21

## Q8: The 5 MB HTML bound effectively requires a compact frame encoding

- Needed for: milestone 7, test item 7 ("a 50-turn random game under about 5 MB")
- Ambiguity: the brief does not fix tile key names. The throwaway reference used one-letter tile keys
  and still produced 4.3 MB for the test's game (seed 1, policy seed `1 ^ 0x9e3779b97f4a7c15`,
  default config, 439 actions, about 9.9 KB per frame). Spelled-out keys (`"terrain"`,
  `"improvement"`, ...) would go over 5 MB.
- Options: A) keep the bound (the test checks `< 5 * 1024 * 1024` bytes); the implementer must use
  short tile keys, arrays, or omit null fields. B) raise the bound (e.g. 10 MB). C) store tiles only
  when they change.
- Agent's pick and why: A, it is the brief's number, and compact frames load faster in the browser.
  Test: `html.rs::page_for_a_50_turn_game_stays_under_5_mb`.
- Owner answer: B on 2026-10-21

## Q9: Formatting details the milestone 7 tests assume

- Needed for: milestone 7, "Text renderer", "CLI", "HTML page", "Replay log"
- Ambiguity: these small points are not fully written in the brief. The tests encode the choices
  below.
  1. `describe_tile` unit text `{phase}{vet}{kills}`: parts joined by single spaces, empty parts
     left out (`... hp 12/15 Moved veteran kills 3`, `... hp 15/15 Done veteran`).
  2. Panel game-over suffix: separated by one space, `turn 50 | P0 to move | game over: P1 wins`.
  3. CLI messages (`wrote ...`, `OK: ...`) are compared ignoring trailing newlines. `view` prints
     exactly `render(..)` on `out`, and `view --all` prints exactly the header line plus `render`
     per step, with nothing in between. Successful commands write nothing to `err`.
  4. Usage errors (exit 2, text only on `err`, nothing on `out`, output file not written) also
     include `--seed -1`, `--player 2`, a missing option value, an unknown option, and an
     unknown flag. Not tested (left open): `--step` together with `--all`, `--turn-cap 0`, and a
     missing input file (exit 1 or 2).
  5. `verify` failures: exit 1, error on `err`, no `OK` on `out`. The error text is not pinned.
  6. Frames: `players[i].cities`/`units` are counts; unit and city keys `id`, `owner`, `row`, `col`,
     `health`, `max_health`, `veteran`, `level`, `capital` hold plain numbers/bools. The encodings
     of `kind` and `phase` and the tile keys are not tested (only that every tile is an object).
  7. `replay_each` on a mismatch: the states before it are visited in order. Whether the
     mismatching state is visited before its check is left open.
- Options: A) as listed. B) owner changes any item (the matching tests change in a new phase A
  commit).
- Agent's pick and why: A, the most natural reading of each format.
- Owner answer: A on 2026-10-21

## Q10: As written, the Economy bot misses the behavior floor (its first Warrior never moves in about a quarter of games)

- Needed for: milestone 8, "Economy bot" rule 8 and "Common bot machinery" (`explore_step`), and
  test item 5 (`bots/tests/floor.rs::economy_beats_random`, at least 30 of 40 wins)
- Ambiguity: the start reveals radius 2 around the capital, and villages are at least 3 from any
  city. So the Economy bot has no village target, and every tile its Warrior can reach (radius 1)
  has vision gain 0 unless a mountain is next to it and it has Climbing. Then `explore_step` is
  `None`, the Warrior stays on the capital, and since a unit on the city tile makes `Train`
  illegal, the bot never trains either. It researches and harvests, then hoards stars.
  In the throwaway reference (a literal reading of the rules) this happens in 26 of 100 games
  against `random` (seeds 0..100, Economy as player 0, turn cap 30). Economy wins 27 of 40 on the
  floor test's games (needs 30) and 127 of 200 on the gate's games (needs 160). Rush is not hit
  as hard (194 of 200) because it takes Explorer at level 2, which reveals targets.
- Options: A) when no reachable tile has vision gain, `explore_step` steps with `greedy_step`
  toward the nearest (ties row-major) explored tile with vision gain > 0. B) add a rule before
  Move: a unit standing on an own city tile where `Train` would otherwise be legal moves off it
  (first reachable tile row-major). C) keep the rules and lower or drop the Economy floor.
- Agent's pick and why: A. It fixes the cause (no way to reach the fog) for both bots and keeps
  them stateless. The tests encode the brief as written, so `economy_beats_random` fails against
  the literal rules. The implementer cannot get `check.sh` green until this is answered and the
  brief, and if needed the goldens, are updated in a new phase A commit.
- Owner answer: A on 2026-10-21

## Q11: Zone of Control from units on unexplored tiles reaches `reachable_tiles` and `legal_actions`

- Needed for: milestone 8, `PlayerView` must not leak unexplored contents (test item 1, the
  no-cheating property)
- Ambiguity: milestone 3 movement stops a unit that arrives next to any enemy unit. That includes
  an enemy on an unexplored tile next to an explored one. So a Rider's reachable set, and the
  legal Move list, change when a hidden enemy unit is added or removed there. The view passes
  `legal_actions` and `reachable_tiles` through, so it can leak that a hidden unit exists.
- Options: A) keep the rule. The no-cheating test only adds or removes units on "deep" unexplored
  tiles (all eight neighbors unexplored too), where they cannot affect movement. B) change the
  movement rule so only enemy units on tiles the mover's owner has explored exert Zone of Control
  (a milestone 3 rule change, with new movement tests). C) keep the rule but have the view filter
  it out (then the view's legal list differs from `legal_actions`, which breaks the brief's API).
- Agent's pick and why: A for milestone 8, since it leaves the existing rules alone. B is worth
  considering before RL, because the learner's legal mask will carry the same signal. The test encodes A (`bots/tests/no_cheating.rs`, header comment).
- Owner answer: A on 2026-10-21

## Q12: Details the milestone 8 tests assume

- Needed for: milestone 8, "Match running and statistics", "CLI additions", "Public API"
- Ambiguity: these points are not fully pinned by the brief. The tests encode the choices below.
  1. Snapshots: "the first moment `state.turn == t && state.current == 0`" includes the terminal
     state. A game ended by the turn cap 30 has its turn-30 snapshot taken from the final state
     (`runner.rs::snapshot_turns`, and every comparison with `common::my_play_game`).
  2. A bare bot's `name()` equals its kind's name (`"random"`, `"rush"`, `"economy"`).
  3. `captured_capital` is `Win(w)` with `w` owning every `is_capital` city.
  4. `play`: the message is `wrote FILE: {n} actions, outcome {..}` with the same outcome text as
     `play-random`; the replay log uses `MapParams::default()` and `Config::default()` with the
     given turn cap. Its default turn cap and max-actions are not tested.
  5. `tournament`: `MapParams::default()` and `Config::default()` with turn cap K. On success the
     table goes to `out` exactly as built from the brief's format strings (one line per pair and
     checkpoint, no blank lines, "win"/"draw" always singular) and nothing goes to `err`. `GATE
     FAILED` may be printed on either stream; the exit code is 1.
  6. `--save-replays`: `{i}` counts from 0, in `run_match`'s game order (seed S with x as player
     0, seed S swapped, seed S+1, ...); the saved games are complete (no action limit).
  7. Usage errors (exit 2) also include: `play` with one or three bots, an unknown bot name, a
     missing option value; `tournament --seed -3`, `--gate-min high`.
  8. `PlayerView::reachable_tiles` panics for any unit that is not the viewer's, hidden or not.
  9. Not tested: the reward fallback ("else the first legal ChooseReward"), since every preference
     is always on its menu with the current tables.
- Options: A) as listed. B) owner changes any item (the matching tests change in a new phase A
  commit).
- Agent's pick and why: A, the most direct reading of each sentence.
- Owner answer: A on 2026-10-21

## Q13: Rush beats Economy 84% of the time, outside the brief's 25%-75% target

- Needed for: milestone 8, "Decisions already made" (Rush vs Economy balance is reported, not
  gated; the implementer reports if it cannot stay between 25% and 75% using only the open options)
- Result: `scripts/bots_gate.sh` (seeds 0..100, both seats, turn cap 30):
  `rush vs economy: rush 168 win, economy 32 win, 0 draw | by capture 168/26 | mean score diff -267.2`.
  Every Rush win is a capital capture. Economy is ahead on score at every checkpoint
  (turn 20: 3337.7 vs 1309.8), but it trains only up to 2 units per city and only Warriors, and
  its attack threshold is 3. My guess (not measured) is that it rarely defends its capital in time.
- Open options I had: almost none. H6 makes the rules the specification. The only choices left
  to me (several visible enemy capitals: nearest, ties row-major; the reward fallback) never
  come up in these games, so they cannot move the balance.
- Options: A) accept the imbalance for now (both bots pass the 80% gate against random, which is
  what RL needs first). B) owner changes Economy's rules (for example a higher unit bound, or a
  defend-the-capital rule), with new goldens in a new phase A commit.
- Agent's pick and why: A. The milestone's hard gate passes, and any rebalance is a behavior change
  that needs the owner (H6).
- Owner answer: A on 2026-10-22

## Q14: `sim-py` must enable `extension-module` for the cargo gate to link

- Needed for: milestone 9, "Changes to earlier tests" (`sim-py` builds without the
  `extension-module` feature under `cargo clippy/test --workspace`)
- Ambiguity: it works the other way round. Without `extension-module`, PyO3 links the cdylib
  against `libpython3.12.so`. That library is not installed on this machine (only
  `libpython3.12.so.1`), so `cargo build --workspace` fails with
  `rust-lld: unable to find library -lpython3.12`. With the feature on, the cdylib leaves the
  Python symbols undefined, as an extension module should, and links without libpython. Either
  way, PyO3's build script runs `python3` once to read the interpreter config.
- Options: A) make `extension-module` a default feature of `sim-py` and set `test = false`,
  `doctest = false` on its lib (it has no Rust tests, J1). This is what the stub commit does.
  B) keep the feature off and install the Python dev package (`libpython3.12-dev`) wherever the
  gate runs. C) set `PYO3_BUILD_EXTENSION_MODULE=1` in `.cargo/config.toml`.
- Agent's pick and why: A. The gate stays self-contained and matches what maturin builds.
  `pyo3` macros compile under the workspace `unsafe_code = "forbid"` lint, so `sim-py` needs no
  exception.
- Owner answer: B on 2026-10-22

## Q15: Details the milestone 9 tests assume

- Needed for: milestone 9, "Public API", "Observation", "Factored action space", "Env",
  "Python layer"
- Ambiguity: these points are not fully pinned by the brief. The tests encode the choices below.
  1. Names: the phase group is `phase.fresh` .. `phase.done`, and the tech flags in
     `global_names` are `tech.climbing` .. `tech.mining`. Python `spec()` lists are snake_case
     data names: `action_types` `move` .. `end_turn` (`choose_reward`, `end_turn`), and
     `unit_types`, `rewards`, `techs`, `site_actions` as in `data/*.json`.
  2. `can_promote`, `can_harvest` and `can_build` come from `view.legal_actions()`, so all three
     are 0 everywhere when it is not the viewer's turn and while a reward is pending.
  3. `pending_reward` is 1 only when the observer has a pending city.
  4. `DecodeError` is `ActorOutOfRange | NoActor | WrongKind | BadArg` (each test case breaks
     one rule only). `step_factored` with `actor >= num_actors` may return `BadIndex` or
     `Decode(ActorOutOfRange)`; both are accepted. In Python a `kind >= 10` is a `ValueError`.
  5. `ActionMasks::arg(a, k)` has length `arg_len(k)` for every actor, all false when the slot
     has no such action.
  6. Rust `Env` has no `replay_json` (the brief's API leaves it out). The Rust tests rebuild a
     replay log from `action_log()` with `sim_tools::replay::record_game` (so `sim-env` has a
     dev-dependency on `sim-tools`). The pytest suite checks `replay_json()` with
     `polytopia verify` (`$POLYTOPIA_BIN`, else `cargo run -p sim-tools --bin polytopia`).
     Where the replay-building code lives is left to the implementer. "No game logic in
     `sim-py`" suggests `sim-env`.
  7. `info()["agent_actions"]` counts the caller's successful steps since `reset`, and
     `info()["scores"]` is a `(s0, s1)` tuple. The default of Python's `agent_seat` is not
     tested.
  8. In two-seat mode the "acting player" of a step is `state.current` before the step.
  9. After truncation only the `Finished` errors are tested. `is_terminal()` and
     `legal_actions()` are not.
  10. `Env::new`'s size assertion is not tested. `MapParams` has no explicit size check, and
      sizes below 5 underflow in `generate_map`.
  11. Pinned versions: Rust `pyo3 =0.30.1` and `numpy =0.30.0` (`crates/sim-py`); Python
      `maturin==1.15.0`, `numpy==2.5.4`, `pytest==9.1.1` (`python/pyproject.toml`).
  12. The pytest files live in `python/tests/`, which the `check.sh` test lock does not cover
      (it only compares `crates/*/tests/`).
- Options: A) as listed. B) owner changes any item (the matching tests change in a new phase A
  commit).
- Agent's pick and why: A, the most direct reading of each sentence. For 12, extending the
  lock to `python/tests/` would be a one-line change to `check.sh`.
- Owner answer: A on 2026-10-22

## Q16: Which map sizes `Env::new` accepts

- Needed for: milestone 9, "Public API" (`Env::new` asserts `size` is accepted by `MapParams`);
  Q15.10
- Ambiguity: `MapParams` had no size check, so "accepted" was undefined. `generate_map` picks
  both capitals from the inland tiles (rows and cols `2..=size-3`). Sizes 0 to 4 have no
  inland tile and panic (or underflow `size - 3`). Size 5 has one, so both capitals land on the
  same tile for every seed (checked on seeds 0..300). Sizes 6 to 255 generate two distinct
  capitals and a full game for every seed checked.
- Options: A) `MapParams::MIN_SIZE = 6`; `Env::new` asserts `size >= MIN_SIZE`, and Python's
  `Env(size=...)` raises `ValueError` below it. B) a larger minimum (for example 8, so each
  capital's inner ring fits away from the other). C) validate inside `generate_map` instead.
- Agent's pick and why: A, the smallest size the generator handles correctly. No upper bound:
  `u8` already caps it at 255, which generates fine.
- Owner answer: A on 2026-10-22 (fix requested after the milestone 9 review)

## Q17: pixi as the Python environment manager

- Needed for: milestone 9, "Out of scope" (new dependencies: Python `maturin`, `numpy`,
  `pytest` only, versions pinned and recorded)
- Ambiguity: the brief approves the Python packages but not how the interpreter and tools are
  installed. Commit 654b518 added `pixi.toml`, `pixi.lock`, `.gitattributes` and the pixi
  tasks `check` and `py-gate`.
- Decision: pixi (tested with 0.76.1, conda-forge channel, `linux-64`) is the owner-approved
  manager for the Python side. `pixi.toml` gives ranges; `pixi.lock` pins the resolved
  versions: `python 3.15.1`, `numpy 2.5.4`, `maturin 1.15.0`, `pytest 9.1.1`, `pip 26.2.2`.
  These match the exact pins in `python/pyproject.toml` (Q15.11). `pip` is approved as part of
  the environment (maturin uses it in `maturin develop`). Run the full gate, Python tests
  included, with `pixi run check`; plain `./scripts/check.sh` outside the pixi env skips them.
  Add a Python dependency only with the owner's approval, via `pixi add`, and commit the
  updated `pixi.lock`.
- Owner answer: approved on 2026-10-22
