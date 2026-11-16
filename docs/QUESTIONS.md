# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q46 are there with pointers to their full text). Number new entries from **Q47**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q47: two M15 v2-digest tests contradict the M16 digest rule

- Needed for: milestone 16 "Digest (under V2 only)"; tests `crates/sim-tools/tests/v2_digest.rs`
  `the_v2_digest_is_the_v1_stream_plus_one_byte_after_the_config` and
  `v2_replay_files_carry_the_ruleset_and_verify`.
- Ambiguity: both M15 tests assert that every v2 digest is the v1 byte stream plus only the ruleset
  byte (their `expected_v2_digest`). M16 adds a trailing `road` byte to every tile record and a
  trailing `connected_to` option to every city record under v2, for every v2 state (also when all
  roads are false). So a correct M16 implementation fails both tests (confirmed with the phase A
  reference implementation: these two are the only existing tests that fail). The brief's
  "Changes to earlier tests" does not allow changing their expectation, so phase A left them as
  they are.
- Options: A. update `expected_v2_digest` in `v2_digest.rs` to the M16 encoding (for example by
  reusing the byte insertion of `crates/sim-tools/tests/road_digest.rs::expected_v2_bytes`); v1
  expectations unchanged, only the v2 digest expectation changes. B. change the M16 rule so the new
  bytes appear only when a road or connection exists (keeps the M15 expectation but makes the
  encoding data-dependent).
- Agent's pick and why: A. The brief's per-record encoding is simpler and fixed-width per record,
  and no v2 replay file or checkpoint exists yet (R5), so nothing persisted depends on the M15 v2
  digest. This needs an owner-approved phase A edit before phase B can reach a green check.
- Owner answer: A (owner, 2026-11-12). Phase A updates `expected_v2_digest` in `v2_digest.rs` to the M16 encoding; v1 expectations unchanged.

## Q48: connections when a player owns two capitals (a capital capture)

- Needed for: milestone 16 "City connections" (`is_connected`, "its owner's capital").
- Ambiguity: capturing the enemy capital ends the game, but `apply` still runs the connection update
  on that terminal state, where the winner owns two cities with `is_capital`. "Its owner's capital"
  is then not unique, and the loser has none.
- Options: A. the owner's lowest-id capital; B. the capital the owner started with; C. skip the
  connection update once the game has an outcome.
- Agent's pick and why: A (simplest, deterministic). The tests do not pin it: the playout property
  test skips terminal states, and no scripted test captures a capital.
- Owner answer: C (owner, 2026-11-12). The connection update is skipped once the game has an outcome; phase A adds a test that pins it.

## Q49: a level-up reward for the opponent's city during the current player's turn

- Needed for: spec "Forced rewards" (SPEC.md: "If any city has a pending reward, the only legal action
  is that city's Choose reward") with milestone 16 "Neutral roads work for both" (found by /review 16).
- Ambiguity: player P builds a road on a neutral tile that completes player Q's connection. Q's city
  and capital each gain +1 during P's turn; if that crosses a threshold, Q's city levels up with
  `pending_reward = true` while P is moving. Read literally, the spec would block P on a reward P
  cannot choose.
- What the M16 code does: only the current player's pending rewards block (`pending_city` looks at
  the current player's cities). Q chooses the reward at the start of Q's turn. If P captures that city
  first, P inherits the open reward and must choose it at once.
- Options: A. keep the code's behaviour and write it into the spec (rewards block only their owner;
  a captured city keeps its pending reward). B. as A, but a capture clears the pending reward.
- Orchestrator's pick: A (matches the code, no new rule). Needs a scripted test in a later phase A.
- Owner answer:
