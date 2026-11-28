# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q57 are there with pointers to their full text). Number new entries from **Q58**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q58: `test_arg_head_v2_relative_position_bias` checks `expected_atk_shift.sum() > 0` for negative `delta2`

- Needed for: Milestone 21 (`python/tests/rl/test_arch_v2.py`, lines 587-602)
- Ambiguity: In `test_arg_head_v2_relative_position_bias` (parametrized over `size in [6, 11]`), the test sets `delta2 = -2.25`, builds `expected_atk_shift` filled with `0.0` and `-2.25` entries (independent of the model under test), and then asserts `expected_atk_shift.sum().item() > 0` on line 601 before comparing `atk_after2 - atk_base` against `expected_atk_shift` on line 602. Because `delta2` is negative and every non-zero entry in `expected_atk_shift` equals `-2.25`, the sum is `-2.25` (`size=6`) or `-4.5` (`size=11`), so line 601 fails unconditionally for any implementation (even though `atk_after2 - atk_base` matches `expected_atk_shift` to `atol=1e-5`).
- Options:
  - A: Change line 601 of `python/tests/rl/test_arch_v2.py` to `assert expected_atk_shift.abs().sum().item() > 0` (or `assert expected_atk_shift.sum().item() < 0`) so the guard checks that at least one in-bounds target tile exists without contradicting `delta2 = -2.25`.
  - B: Change `delta2` on line 588 to a positive value (e.g. `2.25`), keeping line 601 as `> 0` (loses coverage of negative relative-position biases).
- Agent's pick and why: Option A (`expected_atk_shift.abs().sum().item() > 0`), because keeping `delta2 = -2.25` preserves test coverage of negative relative-position biases while fixing the sign typo in the non-emptiness guard.
- Owner answer: A (owner, 2026-11-27) — change line 601 to `assert expected_atk_shift.abs().sum().item() > 0`.
