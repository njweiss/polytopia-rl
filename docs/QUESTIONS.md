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
