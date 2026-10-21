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
