# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q35 are there with pointers to their full text). Number new entries from **Q36**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q36: which error `tick` raises for an action of a slot that is not on its learner's move

- Needed for: milestone 13, Public API (`tick`: "all must be in learner_slots()") and Tests 2 ("tick
  rejects an action for a slot not in it").
- Ambiguity: the brief says the action is rejected but not how.
- Options: A) `ValueError` (as the config and seed checks do); B) `AssertionError` / any exception.
- Agent's pick and why: A. It is a caller error like the others in the package, and a test cannot
  accept "any exception" without also accepting the phase A stub's `NotImplementedError`. The tests
  (`test_interleaved_tick.py`) require `ValueError`, also for a slot index outside `range(num_envs)`.
- Owner answer:

## Q37: the equivalence test fixes each episode's opponent as a function of its seed

- Needed for: milestone 13, "Equivalence" and Tests 1 ("the same opponent decisions").
- Ambiguity: the opponent of each new episode is drawn from the vec env's one shared stream
  (`draw_opponent(pool, rng, p_bot)` in `_reset`), in reset order. The two collectors reset slots in
  different orders, so with that stream the per-slot opponents (and so the games) differ and no
  per-slot comparison is possible. The brief accepts this ("which random draw goes to which decision
  differs") but asks for a comparison with the same opponent decisions.
- Options: A) a test subclass (`il_helpers.KeyedVec`) overrides `_start(i, seed, opp)` and replaces
  `opp` by a fixed function of `seed` (two snapshot entries and both bots), so the shared draw is still
  consumed but ignored; B) patch `draw_opponent` and read the seed from the caller's frame; C) compare
  only with a single opponent (no pool mix).
- Agent's pick and why: A. It keeps the brief's mix (two snapshots plus bots, both seats) and depends
  only on today's `_reset -> _start` path, which the brief says to reuse. Consequence for phase B: every
  episode start, in both collectors, must keep going through `_start` (as `_reset` does today).
- Owner answer:
