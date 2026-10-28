# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q37 are there with pointers to their full text). Number new entries from **Q38**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q38: a slot still playing a snapshot that thinning removed from the pool

- Needed for: milestone 13b, "Public API" (`SelfPlayVecEnv`: `sync(pool.entries)` before the stacked call)
  and "Rules / Stacked forward".
- Ambiguity: a slot keeps its episode's opponent `Entry` until the episode ends, and episodes span
  rollouts. Once the pool is full (`pool_max`), `pool.add` thins it between rollouts, so a slot can be
  mid-episode against an entry that is no longer in `pool.entries` (M13 keeps such policies alive in
  `_policies`, and `state_dict` saves their weights under `entries`). With the brief as written, the
  stack built from `pool.entries` has no row for that entry and `index(name)` cannot answer.
- Options: A) the vec env syncs `pool.entries` followed by the entries of slots whose opponent left
  the pool (first appearance by slot, no duplicates); the stack then restacks once more when the
  last such episode ends. B) groups whose entry is not in the pool fall back to the per-snapshot
  `make_policy` call in that tick (two kinds of calls in one tick). C) keep the brief literally
  (crash once thinning starts, ~40M steps into a run with the defaults).
- Agent's pick and why: A. It keeps the "one stacked call per tick" goal, needs no second code path,
  and the extra restacks are as rare as thinning. The phase A tests do not exercise thinning: they
  assert `sync` receives exactly `pool.entries` only in fixtures where every slot's opponent is in the
  pool, which A, B and C all satisfy.
- Owner answer: A on 2026-10-28 (sync the pool entries plus the entries of slots whose opponent left the pool)
