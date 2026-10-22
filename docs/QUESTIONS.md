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
