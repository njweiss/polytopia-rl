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

## Q18: The pixi env's numpy cannot load MKL

- Needed for: milestone 10 (and the milestone 9 pytest suite), Q17
- Ambiguity: the uncommitted `pixi.toml` change (python 3.12, `pytorch`, `matplotlib`) resolved
  conda-forge `libblas 3.11.0 *_mkl` with `mkl 2026.2.0`. Its activation script sets
  `MKL_INTERFACE_LAYER=LP64,GNU`, but that mkl build has no `libmkl_gf_lp64.so.3`, so a plain
  `import numpy` followed by any linear algebra (and `import polytopia_sim`) aborts with
  `Intel oneMKL FATAL ERROR: Cannot load libmkl_gf_lp64.so.3`. Importing torch first hides it.
- Options: A) pin the BLAS to OpenBLAS (`pixi add "libblas=*=*openblas"`). B) pin an `mkl` build
  that ships the GNU interface layer. C) set `MKL_INTERFACE_LAYER=LP64` in a pixi
  `[activation.env]`.
- Agent's pick and why: A. It removes the MKL interface-layer problem entirely, and torch's CPU
  build does not need MKL-backed numpy. The milestone 10 tests were validated with
  `MKL_INTERFACE_LAYER=LP64` set by hand (option C as a stopgap). Nothing was changed in
  `pixi.toml`.
- Owner answer: A on 2026-10-22. Done: `pixi.toml` pins `libblas` to the `*openblas` build
  (resolved `libblas 3.11.0 *_openblas`, `libopenblas 0.3.34`, `nomkl`; `pytorch 2.14.1` moved to
  its `cpu_generic` build). The milestone 9 pytest suite passes without `MKL_INTERFACE_LAYER`.

## Q19: Details the milestone 10 tests assume

- Needed for: milestone 10, "Public API" (several pieces are named but not specified), "Rules"
- Ambiguity and the choices the tests fix (stubs in `python/polytopia_rl/` document each):
  1. API additions the tests need: `vec_env.make_vec_env(cfg, seed_base, seed_stride=1)`;
     `SyncVecEnv(..., opponents=None)` (labels for `EpisodeStat.opponent`, since `Env` cannot
     report its opponent); `SyncVecEnv.state_dict()`/`load_state_dict()` (exact resume: per-env
     episode index and seed, plus the agent actions of each episode in progress, restored by
     reset and replay); `vec_env.TRAIN_SEED_LIMIT`; `evaluate.EVAL_SEED_BASE`,
     `evaluate.wilson_interval`, `evaluate.net_policy(net, generator=None, device="cpu")`;
     `curriculum.StageHistory(steps_in_stage, outcomes)` as `next_stage`'s `history`;
     `ppo.collect_rollout(net, vec_env, buffer, generator)`; `train.run(cfg, run_dir,
     resume=None) -> PolicyNet`, `train.load_policy(path, device) -> (net, cfg)`,
     `train.main(argv)` with `--config`, `--run-dir`, `--resume` and `key=value` overrides;
     `plot.plot_run(run_dir) -> list[str]` (PNG paths under the run dir).
  2. `ActOut` also returns the three masks it fetched (`actor_mask`, `kind_mask`, `arg_mask`) so
     the rollout buffer can store them. `Features` has exactly `fmap` (B, width, H, W) and `g`; the
     layout tests build `Features` by hand, perturb `fmap` at one tile and hold `g` fixed.
  3. Heads are the submodules `actor_head`, `kind_head`, `arg_head`, `value_head`, sharing no
     parameters: the kind and argument heads each own their global-actor vector and family
     embedding ("no weight sharing between the three stage heads beyond the trunk").
  4. `masks_fn("actor")`, `masks_fn("kind", actors)`, `masks_fn("arg", actors, kinds)` with numpy
     int64 arrays, returning bool arrays. A row whose kind has no argument stores `arg = 0` and an
     all-false arg mask.
  5. `PPOConfig` field names beyond the table: `size`, `turn_cap`, `full_observability`,
     `opponents`, `opponent_epsilon`, `clip_vloss` (default true, CleanRL), `stages`,
     `min_stage_steps` (default 1_000_000, my pick), `stage_window`, `stage_win_rate`,
     `eval_every`, `eval_seeds` (default 50). A stage dict overrides config fields (only
     `opponents` is used). `from_json` and overrides reject unknown keys with `ValueError`;
     override values parse as JSON, else as the raw string.
  6. Advantage normalization lives in `ppo_loss` (unbiased std, `+ 1e-8`), so it is per
     minibatch by construction. `approx_kl = mean((r - 1) - log r)`, `clipfrac =
     mean(|r - 1| > clip_eps)`; the value clip uses `clip_eps`.
  7. GAE's `next_values[t]` is read only where `truncated[t]`; if both flags were set, the
     terminated rule (no bootstrap) wins. `RolloutBuffer` field names are fixed (the
     `collect_rollout` test reads `values`, `next_values`, `last_value`, `truncated`, ...).
  8. LR annealing is CleanRL's `lr * (1 - (update - 1) / num_updates)` with update 1-based and
     `num_updates = total_steps // (num_envs * num_steps)`; it never reaches exactly 0.
  9. `EpisodeStat.outcome` is `"draw"` when the episode has no outcome (truncation). Episodes that
     finish on the same step are reported in env-index order. `length` counts agent actions.
  10. `evaluate(..., seeds: int)` plays `EVAL_SEED_BASE + j`, `j < seeds`; `per_game` entries are
      dicts `{seed, seat, outcome, length, score_diff}` in seed order, seat 0 first; `length` is
      agent actions; `by_seat = {0: rate, 1: rate}`; Wilson uses z = 1.96. `env_kwargs` may not
      override the opponent, its epsilon (forced to 0) or the seat.
  11. Checkpoints are written at every evaluation (each time `global_step` crosses a multiple of
      `eval_every`) and at the end, and hold at least `model`, `optimizer`, `config` (JSON),
      `global_step`, `stage`, `vec_env`, `rng`. The resume and repeatability tests compare
      metrics with timing keys removed (any key containing `sps` or `time`).
  12. Metric names: per-head entropies `entropy_actor`, `entropy_kind`, `entropy_arg`;
      `kind_freq` is a dict over `spec()["action_types"]` summing to 1; `mean_return`,
      `train_win_rate` and `mean_episode_len` are over the same last-`stage_window` window and
      are `null` until an episode has finished.
  13. `python/configs/stage1_random.json` has exactly one stage (`random`), and
      `python/configs/curriculum.json` the brief's three.
  14. The vector env and the model use only these `Env` methods: `reset`, `observation`,
      `actor_mask`, `kind_mask`, `arg_mask`, `step_factored`, `info` (`agent_seat`,
      `outcome`), `spec` (`height`, `width`, `channels`, `globals`, `num_actors`, `arg_lens`,
      `action_types`). The learning tests' `FakeEnv` implements only these.
  15. The `check.sh` test lock now also covers `python/tests/rl/`. `py_check.sh` still runs
      `pytest python/tests`, which includes `python/tests/rl/`; the implementer's `rl_check.sh`
      wiring should keep the two suites apart (for example `--ignore=python/tests/rl`).
- Options: A) as listed. B) the owner changes any item (the matching tests change in a new phase A
  commit).
- Agent's pick and why: A, the most direct reading of the brief plus the smallest additions that
  make each listed test possible.
- Owner answer: A on 2026-10-22

## Q20: How `evaluate` passes `greedy` to `policy_fn`

- Needed for: milestone 10, "Public API" (`evaluate(policy_fn, opponent, seeds, env_kwargs,
  greedy)`) and "Evaluation" (`policy_fn(env) -> (actor, kind, arg)`)
- Ambiguity: `evaluate` takes `greedy`, but a `policy_fn(env)` has no way to receive it.
- Options: A) `evaluate` calls `policy_fn(env, greedy)`; scripted policies ignore the flag.
  B) keep `policy_fn(env)`, and `greedy` is only used when `evaluate` wraps a net itself.
  C) drop `greedy` from `evaluate`, and `net_policy(net, greedy)` takes it instead.
- Agent's pick and why: A. It keeps both signatures close to the brief, and one `net_policy`
  serves greedy and sampled evaluation. The tests use A.
- Owner answer: A on 2026-10-22

## Q21: The pixi env's torch has no CUDA

- Needed for: milestone 10, "Decisions already made" (compute: training must run on the RTX 4060)
  and "Out of scope" (pin versions and record the CUDA build used)
- Ambiguity: since Q18 (OpenBLAS pin) the pixi env resolves `pytorch 2.14.1 cpu_generic`
  (`torch.version.cuda` is `None`, `torch.cuda.is_available()` is `False`), so `device="auto"`
  falls back to the CPU. Switching to a CUDA build is a dependency change I may not make.
- Options: A) the owner adds a CUDA build in `pixi.toml` (for example a conda-forge
  `pytorch=*=cuda*` build plus the `cuda-version` it needs, checked against the OpenBLAS pin) and
  commits `pixi.lock`; `python/requirements-rl.txt` then records that build. B) stay on CPU for
  milestone 10 (the 10a smoke run was done on CPU) and decide on GPU when 10b needs it.
- Agent's pick and why: A before 10b (20M steps). The code is device-agnostic (`device="auto"`,
  sampling uses a CPU generator so it also works with a CUDA model); nothing else changes. The
  RL_REPORT timing split shows how much the GPU can help (the forward pass share).
- Owner answer: A on 2026-10-22. Done: `pixi.toml` declares `cuda = "12.9"` on the linux-64 platform
  and pins `pytorch` to the `cuda129_generic*` build (`pytorch 2.14.1 cuda129_generic_py312`,
  `cuda-version 12.9`; the conda-forge `cuda129_mkl` builds would break the Q18 OpenBLAS pin). It
  also adds `cuda-cudart-dev 12.9`: torch 2.14 runs one batched-matmul backward kernel through
  Triton, which compiles a C helper at run time against `cuda.h` (without it, `train_update` on
  CUDA fails with `cuda.h: No such file or directory`). Measured on the RTX 4060 (default config):
  ~1,550 sps vs ~255 sps on CPU.

## Q22: The exact bytes behind `obs_hash` and `rules_hash`

- Needed for: milestone 11, "Public API" (`obs_hash`, `rules_hash`) and L9 (`RULES_VERSION` is part
  of `rules_hash`); the tests pin a golden canonical string for `size = 3`.
- Ambiguity: the brief lists the ingredients but not the separators, whether lists are delimited,
  how numbers are written, or where `RULES_VERSION` goes in the rules bytes.
- Options: A) one flat list of items joined with `\n` (no section markers, no trailing newline):
  `size`, channel names, global names, the 10 action-type names, `arg_len` per action type in
  decimal, unit types, rewards, techs, site actions. Rules bytes: for each embedded file in file-name
  order, its bare name (`units.json`), `\0`, its contents; then `RULES_VERSION`, `\0`, the version in
  decimal. B) the same with a header line per section (`[channels]`, ...), which rules out an item
  moving across a section boundary without a hash change. C) `RULES_VERSION` first instead of last.
- Agent's pick and why: A, the literal reading of "each item joined with `\n`". The section-boundary
  case of B cannot happen without also changing a name. Test hooks (public, in `sim_env::hash`):
  `hex64(u64) -> String`, `obs_canonical(size) -> String`, `embedded_data_files()`,
  `rules_canonical(files, version) -> Vec<u8>`. The Python test assumes `RULES_VERSION == 1`; bumping
  it means updating that constant in `python/tests/rl/sp_helpers.py` and the Rust test.
- Owner answer: A on 2026-10-23

## Q23: A hook to play the Rust bots' moves from Python (`Env.bot_action`)

- Needed for: milestone 11 "Tests" 3 ("a policy that plays the Rust bot's exact choices (through
  `step_legal` lookup)") and 7 ("a scripted policy equal to the Rust economy bot (via the two-seat
  path)"), and the tournament's bot-vs-bot games.
- Ambiguity: nothing in the current API lets Python ask what a bot would play, so neither test can be
  written; the brief does not name the mechanism.
- Options: A) `sim_env::env::Env::bot_action(&mut self, kind: BotKind) -> Result<Factored,
  EnvError>`, exposed as `Env.bot_action(name) -> (actor, kind, arg)`: the move bot `name` (epsilon
  0) picks for the player to move from that player's `PlayerView`, with one rng stream per seat
  (`bot_rng(seed, seat)`, created at `reset`, advanced only by this call), so driving a seat with it
  reproduces `bots::play_game`. `Finished` (Python `RuntimeError`) once over, `ValueError` for an
  unknown name. The env's own opponent is unaffected. B) return an index into `legal_actions()`
  plus a new `step_legal` binding; C) expose `legal_actions` as JSON and match in Python.
- Agent's pick and why: A. One call, no new action encoding in Python, and it works with
  `step_factored`, which every policy already uses. It reads only the mover's own view, so it does not
  cheat. Tested in `crates/sim-env/tests/bot_action.rs` against `bots::bot_policy`.
- Owner answer: A on 2026-10-23

## Q24: Loss review definitions

- Needed for: milestone 11 "Loss review" (`collect_losses`, `LossRecord`).
- Ambiguity: which seeds, greedy or not, the record fields, how to tell a capital capture from a
  score loss, and what "the turn the first city was lost" means when the agent only observes at its
  own decisions.
- Options and the tests' choices: seeds are the suite seeds `SUITE_SEED_BASE + j`, `j <
  seeds`, seat 0 then seat 1, greedy, epsilon 0, stopping at `limit` losses. `LossRecord(seed, seat,
  final_turn, loss_type, scores, first_city_lost_turn, path)`, `scores` seat-indexed from `info()`.
  `loss_type` is `capital_captured` when the game ended before the turn cap (`info()["turn"] <
  turn_cap`), else `score`. `first_city_lost_turn` is `info()["turn"]` at the first agent
  observation (a decision or the final observation) where a tile that showed `city.mine` at its
  previous observation no longer does, or None. Alternative: the exact turn of the capturing move,
  which needs the opponent's moves decoded in Python (not possible with the current API). Note: for a
  seat-0 agent, a city lost during the opponent's turn `t` is first seen at turn `t + 1`, except when
  it ends the game.
- Agent's pick and why: as above; observation-based, so it uses only what a policy may see. The
  scripted test agent never holds a second city, so the tests check only "capital capture: equals
  the final turn" and "score loss: None".
- Owner answer:  As above on 2026-10-23

## Q25: Pool thinning and the snapshot draw

- Needed for: milestone 11 "Self-play" (snapshots, opponent per episode), L4.
- Ambiguity: "when full, thin by dropping every second entry except the first and the newest" does
  not say whether thinning happens before or after the new entry is added, nor whether "uniformly
  drawn entry of the whole pool" includes the newest.
- Options: A) `add` appends, then while `len > pool_max` keep positions 0, 2, 4, ... and the newest
  (pool_max 4, adding steps 0..4 gives `[0, 2, 4]`). B) thin a full pool before appending (gives
  `[0, 2, 3, 4]`). The uniform draw includes the newest (so it has probability `p_latest + (1 -
  p_latest) / K`).
- Agent's pick and why: A, and the newest included in the uniform draw ("the whole pool"). The tests
  pin both. Also fixed by the tests: `SnapshotPool(pool_max=40, p_latest=0.5)`, `Entry(step, state,
  name="snap_{step}")`, and `selfplay.draw_opponent(pool, rng, p_bot)` (bot with probability `p_bot`,
  uniform over `BOT_OPPONENTS = ("economy", "rush")`, else `pool.sample(rng)`).
- Owner answer:  A on 2026-10-23

## Q26: Truncation and the final observation in two-seat slots

- Needed for: milestone 11 "Tests" 7 ("an opponent that is mid-turn at the learner's truncation is
  handled") and "Execution".
- Ambiguity: a two-seat `Env` counts both seats' actions towards `max_agent_actions`, so it can
  truncate on an opponent move; and in two-seat mode `observation()` is the mover's (or the last
  actor's) view, so after the opponent's move the learner's view of the final state is not available.
- Options: A) a slot ends whenever its env reports `terminated` or `truncated` on any sub-step (the
  learner's or the opponent's); the episode is then truncated for the learner, and the final
  observation is defined (and tested) only when the learner's own action ended the episode.
  B) count learner actions in Python and truncate only after a complete opponent turn (needs the
  two-seat env's own limit raised). C) add `observation(seat)` to `Env`.
- Agent's pick and why: A, no new API; truncation in two-seat slots is rare at the 5000-action
  default, and a terminated episode does not bootstrap. The tests also fix: the learner's reward
  equals the M9 formula exactly in float32 (`tw * t + sw * (gamma * phi' - phi)`), and opponent calls
  are batched so that, in a step where no episode ends, each snapshot gets exactly one call per round.
  A truncation inside the opponent's opening moves right after a reset is not tested.
- Owner answer:  A on 2026-10-23

## Q27: Details the milestone 11 tests assume

- Needed for: milestone 11, "Public API" (several types are named but not specified), "Rules".
- Ambiguity and the choices the tests fix (stubs in `python/polytopia_rl/` document each):
  1. `checkpoint`: `SpecWarning(UserWarning)` raised via `warnings.warn` for every non-strict
     mismatch and for a file without `spec`; `Checkpoint(model, optimizer, cfg, global_step, stage,
     seed_index, rng_state, spec, extra)` with `model`/`optimizer` state dicts and `cfg` a
     `PPOConfig`; `save_checkpoint` takes the module and optimizer; the file keeps `model`,
     `optimizer`, `config` (JSON), `global_step`, `stage` and adds `spec`, so `train.load_policy`
     still reads it.
  2. `suite`: `SUITE_SEED_LIMIT = 1_500_000_000` (a range reaching it raises `ValueError`);
     `run_suite(..., *, step=None)`; `SuiteResult(conditions: dict name -> ConditionResult,
     obs_hash, rules_hash, step)` with the hashes of the running build; `ConditionResult` holds
     `evaluate`'s fields plus `loss_capital_share`, `loss_score_share` (0 without losses) and a
     per-game `loss_type`; `Comparison(name, win_rate_a, win_rate_b, diff = b - a, z, significant)`,
     z with the pooled proportion, 0 when the pooled sd is 0; `compare` covers the conditions in
     both, in `a`'s order. The suite CLI's `--out DIR` writes `suite.json` (with `obs_hash`,
     `step`) and `suite.md`; `--compare` prints one line per condition. `python/configs/suite.json`
     has `"seeds": 200`.
  3. `tournament`: `Player(name, bot=None, policy_fn=None, greedy=True)`; `PairResult(a, b, games,
     a_wins, b_wins, draws, seat_wins, per_game)`, per-game dicts `{seed, a_seat, winner: "a" |
     "b" | None, length}`; `TournamentResult(players, wins (decisive), draws, games, seat_win_rate
     (share of all games won by each seat), ratings)`, anchored at the player named `random` (else
     player 0); `tournament_markdown`; `tournament.json` holds the same fields (matrices as lists,
     `seat_win_rate` keyed `"0"`/`"1"`).
  4. `selfplay`: `SelfPlayVecEnv(env_fn, num_envs, seed_base, seed_stride=1, *, pool, make_policy,
     rng, p_bot=0.2, reward)`: `env_fn(opponent)` builds an env (`"none"` for two-seat),
     `make_policy(entry)` returns a batch policy `policy(envs) -> actions`, `rng` is the dedicated
     NumPy generator, `reward` the four M9 weights; `envs[i]` is slot `i`'s current env;
     `EpisodeStat.opponent` is the bot name or `snap_{step}` (this is the per-episode opponent log).
     `net_batch_policy(net, generator, greedy=False, device="cpu")` makes one `trunk` call per
     batch. `regression(win_rates, floor)`: any rate `< floor`.
  5. Self-play trainer (`train.run(cfg, run_dir, resume=None, init_checkpoint=None)`, CLI
     `--init-checkpoint`), switched on by the config key `selfplay`. New `PPOConfig` keys and
     defaults: `selfplay=False`, `snapshot_every=1_000_000`, `pool_max=40`, `p_bot=0.2`,
     `p_latest=0.5`, `opponent_greedy=False`, `rating_every=2_000_000`, `rating_seeds=50`,
     `regression_floor=0.9`, `tournament_seeds=100`; the suite subset uses `eval_seeds`. A snapshot
     is added each time `global_step` crosses a multiple of `snapshot_every` (so also at the last
     update when it lands on one). `eval.jsonl` rows carry `"type": "suite"` (`suite: {name:
     {win_rate, ...}}`, `regression`) or `"type": "rating"` (`rating: {learner, snapshot_0}`);
     metrics rows add `pool_size`. Checkpoints `ckpt_{step}.pt` and `latest.pt` at every evaluation
     as in milestone 10, plus `ckpt_final.pt`; `pool/snap_{step}.pt` are checkpoints whose
     `global_step` is the snapshot's step; `train.log` is written. Resume restores the pool, the
     opponent RNG and sampling generator, and the env state exactly.
  6. CLIs that take a checkpoint print its `obs_hash`, `rules_hash` and step on the first line.
     `losses` CLI: `--checkpoint`, `--opponent`, `--count`, `--out`.
  7. Changed milestone 10 test: `CKPT_KEYS` in `test_train.py` gains `spec` (the brief's "Changes
     to earlier tests"). No other earlier test changes.
- Options: A) as listed. B) the owner changes any item (the matching tests change in a new phase A
  commit).
- Agent's pick and why: A, the most direct reading of the brief plus the smallest additions that
  make each listed test possible.
- Owner answer: A on 2026-10-23

## Q28: Self-play rollout throughput (the opponent rounds)

- Needed for: milestone 11 gate 11b (20M self-play steps), "Execution" and "Notes for the
  implementer" (batch the opponents' forward passes).
- Ambiguity: none in the rules; the implementation follows "Execution" exactly. But measured on the
  RTX 4060 (size 11, 32 envs, 2-update smoke run from the M10 `latest.pt`), self-play runs at
  ~200-240 sps overall (rollout ~220-280 sps, update ~1,600-2,500 sps), against ~1,550 sps for the M10
  bot-mode training. A cProfile of one 4,096-step rollout: 13.1 s, of which 11.5 s is 1,369 opponent
  batch-policy calls (~10 rounds per learner step, ~8 ms each, mostly small-batch launch overhead).
  The rounds wait for the longest opponent turn among the slots whose learner just ended its turn. At
  this rate 20M steps take ~25 h, not the "several hours" the brief expects.
- Options: A) accept it (11b is an overnight-plus run). B) a later speed milestone: e.g. step the
  learner's and the opponents' moves of all slots in one forward pass per snapshot (slots that wait
  on an opponent are not stepped by the learner meanwhile), or more envs per batch, or the M10 note's
  batched mask fetch. C) cut 11b to fewer steps.
- Agent's pick and why: A for now, B after profiling as CLAUDE.md asks. Nothing changed here; the
  metrics rows now also log `rollout_sps` and `update_sps` so the owner's runs show the split.
- Owner answer: A on 2026-10-23

## Q29: Speed after the 3M smoke run (opponent calls, tournament, warm start)

- Needed for: milestone 11 gates 11a step 4 (the round robin) and 11b (20M steps plus the final
  tournament), "At the end" and "Notes for the implementer".
- Ambiguity: none in the rules. The 3M smoke run (`runs/selfplay_smoke`) measured: rollout ~300 sps
  with only snap_0 in the pool, ~175 sps from the 1M snapshot on. A profile of one 4,096-step rollout
  with the 3-snapshot pool: 23 s, of which ~21 s is opponent policy calls (~20 calls per learner step,
  mean batch 1.5 envs, ~7 ms fixed cost per call: kernel launches, device copies and three GPU syncs).
  The Rust env calls take < 0.3 s and the Python slot loop < 0.2 s. `act` also computes `entropy` and
  `log_prob` for opponents (~20% of policy time), which they do not use. Opponents on CPU: 39 sps;
  128 envs: 285 sps. The final tournament plays one game at a time (net vs net ~5.8 s, net vs bot
  ~1.2 s per game): ~1.6 h for the smoke pool, ~70 h for a 20M pool of ~21 snapshots. Separately, the
  warm start's first updates had approx_kl ~0.2-0.38 (M10 ~0.05), and rush_e0 fell from ~0.97 (M10)
  to 0.79 at 1M and 0.85 at 2M.
- Options: R1) skip `entropy`/`log_prob` in opponent calls (flag on `act`; sampled actions
  bit-identical). R2) batch the tournament and rating games across seeds inside `evaluate`'s
  internals, so `evaluate` stays the one place that plays an evaluation game. R3) `num_envs` 128
  (or 128 x `num_steps` 32 to keep the 4,096 batch). R4) interleave learner and opponent moves in the
  rollout (restructures `SelfPlayVecEnv` and the buffer). R5) order: R1 and R2, then finish 11a, then
  launch 11b. R6) note the warm-start KL and rush drop in RL_REPORT; if 11b repeats it, try a lower
  `lr` or a warm-up (config only).
- Agent's pick and why: R1 and R2 now (cheap; without R2 the 11b tournament is impractical), R3
  after 11a with numbers, R4 deferred, R5 as listed, R6 as listed.
- Owner answer: R1 yes, R2 yes, R3 later, R4 defer, R5 agreed, R6 yes, on 2026-10-23
- Implemented (R1, R2): `PolicyNet.act(..., stats=False)` skips `log_prob`, `entropy` and `value`
  (`logp`/`entropy`/`value` are None); `net_batch_policy` and `evaluate.net_policy` use it, the PPO
  rollout does not. Checked on the smoke learner: 0 of 14,400 sampled action components differ with
  the same generator seed; profile rollout 177 -> 215 sps. `Player.batch_fn(envs)` plus a lockstep
  `play_pair` (all of a pair's games at once, one call per player per round); the batching lives in
  `play_pair`, which already had its own game loop, not in `evaluate` as the R2 text said; `policy_fn`
  players and bots play as before. Greedy net games are identical to the sequential ones (40 games
  checked); 100 seeds: net vs net 5.8 -> 0.09 s/game, net vs rush 1.2 -> 0.02 s/game, so a 20M pool's
  round robin is ~1.2 h instead of ~70 h.

## Q30: Per-call cost of play-only policy calls (after Q29 R1)

- Needed for: milestone 11 gate 11b throughput (the 20M run `runs/selfplay_lr1e-4`, started
  2026-10-24 with `lr=1e-4`). Speed work outside the brief, so the owner decides.
- Ambiguity: none in the rules. `torch.profiler` on `net_batch_policy` (smoke-run 1M snapshot,
  batch 1, two-seat mid-game states): ~5.4 ms wall per call, of which GPU kernel time is only
  ~0.3-0.7 ms. Per call: ~124 `cudaLaunchKernel` (~22 us each, ~2.7 ms), ~19
  `cudaStreamSynchronize` (~1.1 ms) and ~27 `cudaMemcpyAsync` (~1 ms). The calls are bound by
  kernel launches and host-device syncs, not by the network. The syncs come from the lazy mask fetch
  (actors, then kinds, to the CPU), the mask validity check in `categorical.masked_logits`
  (`bool(mask.any(-1).all())`, once per stage), sampling with a CPU generator (probabilities to the
  CPU and the index back, once per stage) and `tolist`. Timing candidate fixes by monkeypatching,
  while the 20M run shared the GPU (so noisy: baselines 5.4-8.4 ms for the same call):
  skip the check -6% to +14%; plus `torch.inference_mode` -3% to +24%; plus a CUDA generator
  +17% to +35%.
- Options: A) result-identical changes, safe to swap into a running experiment by stopping and
  `--resume`: skip the mask check on the play-only path (`act(stats=False)`; Rust masks always have
  a legal option, and the training path keeps the check) and run play-only calls under
  `torch.inference_mode`. B) sample opponents with a CUDA generator: removes three round trips per
  call, but changes the opponents' random stream, so fresh runs only (still deterministic, not
  identical to earlier runs). C) CUDA graphs or `torch.compile(mode="reduce-overhead")` for the
  opponent forward: attacks the ~124 launches (the largest cost), but variable batch sizes need a
  graph per size; a larger change, fresh runs only until shown identical. D) nothing now.
- Agent's pick and why: A now (small, cannot change any result, and the 20M run can pick it up at
  its 1M checkpoint through `--resume`), measured again on an idle GPU before claiming a number; B
  and C for a later speed milestone with the larger structural options (Q29 R3, R4). The noisy
  measurements above do not support a firm estimate: A is somewhere between 0 and ~25% of the
  opponent call cost.
- Owner answer: A + B, plus Q29 R3 as `num_envs=128 num_steps=32` (batch stays 4,096), and
  `lr=1e-4` for the 11b run, on 2026-10-24. C not now. The 20M run `runs/selfplay_lr1e-4` was
  stopped at step 180k (no checkpoint) to restart with these.
- Implemented: `categorical.masked_logits`/`sample` take `check` (default True); `act(stats=False)`
  passes `check=False`. `net_batch_policy` and `evaluate.net_policy` run under
  `torch.inference_mode`. The trainer's opponent generator is created on the model's device. Config
  `python/configs/selfplay.json`: `num_envs 128`, `num_steps 32`, `lr 1e-4`. Checked on the GPU:
  A changes 0 of 14,400 sampled action components (same CPU seed); the CUDA generator repeats for a
  seed and after `get_state`/`set_state`. Rollout with the 3-snapshot smoke pool (idle GPU): 243 sps
  (32 x 128, before A+B) -> 264 (32 x 128) -> 407 (128 x 32). Real trainer, 10 updates from the M10
  checkpoint (pool of snap_0 only): 621 sps, against 307-374 for the stopped run at the same stage.
- Follow-up (2026-10-24, owner approved): the trainer kept 11 CPU worker threads spinning at
  ~90-100% each (kernel time 68%, ~480k context switches/s) on tiny CPU tensor ops. With
  `OMP_NUM_THREADS=1` the 128 x 32 rollout ran 315 -> 395 and 358 -> 421 sps (two alternating
  tests, each beside the running trainer). `scripts/selfplay.sh` now defaults `OMP_NUM_THREADS` to 1
  (an explicit value wins). Results do not change. The 11b run `runs/selfplay_seed1` was stopped at
  ~1.15M and resumed from its 1M `latest.pt` with it; its `metrics.jsonl` and `train.log` were cut
  back to step 1,003,520 before the resume (originals in `pre_resume_backup/`).
