# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q55 are there with pointers to their full text). Number new entries from **Q56**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q56: Phase A tests assert Warrior vs Warrior forest Archery strictly increases retaliation (`5 > 5`)

- Needed for: Milestone 20 (`crates/sim-core/tests/v2_view.rs:334` and `crates/sim-env/tests/v2_env.rs:1974`)
- Ambiguity: Both locked Phase A tests pit a full-health Warrior (10/10, atk=2, def=2) against a full-health Warrior (10/10, atk=2, def=2) on a forest tile, comparing defense bonus `1.0` (no Archery) vs `1.5` (hidden enemy Archery), and assert that retaliation strictly increases (`p_arch.retaliation > p_no_arch.retaliation` in `v2_view.rs:334`, and `ba_yes[PREV_TARGET_MIN_RETAL] > ba_no[PREV_TARGET_MIN_RETAL]` in `v2_env.rs:1974`). Under the spec combat formula (`docs/SPEC.md` "Combat", half-up rounding):
  - At `1.0x`: `attackForce = 2.0`, `defenseForce = 2.0`, `totalDamage = 4.0`, `defenseResult = (2.0 / 4.0) * 2.0 * 4.5 = 4.5` -> `round_half_up(4.5) = 5` (and `dealt = 5`).
  - At `1.5x`: `attackForce = 2.0`, `defenseForce = 3.0`, `totalDamage = 5.0`, `defenseResult = (3.0 / 5.0) * 2.0 * 4.5 = 5.4` -> `round_half_up(5.4) = 5` (and `dealt = 4`).
  Both `apply_attack` and `attack_preview` return `retaliation == 5` in both cases (`assert_preview_matches_apply` passes for both), so `5 > 5` fails in both locked tests.
- Options:
  - A: Amend the two locked tests (and refresh `.git/tests-lock`) to use a matchup whose rounded retaliation strictly increases at `1.5x` — e.g., Swordsman (atk=3, def=3) attacking Warrior on forest (`dealt` drops `8 -> 7`, `retaliation` rises `4 -> 5`) or Warrior attacking Defender on forest (`dealt` drops `4 -> 3`, `retaliation` rises `8 -> 9`).
  - B: Change the two assertions from `>` to `>=` (keeping Warrior vs Warrior, where `dealt` already strictly drops `5 -> 4` and `defense_bonus` rises `1.0 -> 1.5`).
- Agent's pick and why: Option A (e.g., Swordsman attacking Warrior, or Warrior attacking Defender) — it tests both strict decrease in `dealt` and strict increase in `retaliation` under hidden Archery without weakening the assertion.
- Owner answer: A (owner, 2026-11-27) — use Swordsman attacking Warrior on forest (`dealt` 8 -> 7, `retaliation` 4 -> 5) in both tests.

## Q57: Phase A test `encode_board_v2_damage_preview_aggregation_and_zero_cases` puts own Archer in range of Target 1

- Needed for: Milestone 20 (`crates/sim-env/tests/v2_env.rs:593,622,629-633`)
- Ambiguity: In `encode_board_v2_damage_preview_aggregation_and_zero_cases`, Target 1 is an enemy Defender (9/15) at `p(2, 2)`, and own units include `u0` (Warrior 9/10 at `p(2, 1)`), `u1` (Swordsman 15/15 at `p(1, 2)`), and `u2` (Archer 10/10, range 2, at `p(4, 2)` with comment `// u2 (hits (4,4) and (2,4))`). Because `chebyshev(p(4, 2), p(2, 2)) == 2 <= 2`, `Action::Attack { unit: UnitId(2), target: p(2, 2) }` is also a legal attack, and since Defender has range 1, `u2` takes `0` retaliation. Consequently, `preview.target_min_retal` at `p(2, 2)` across all legal attackers is `0.0`, while line 622 computes `t1_min_retal` from `u0` and `u1` only (`6.0 / 15.0 = 0.4`), failing line 629 (`got 0, expected 0.4`).
- Options:
  - A: Amend `crates/sim-env/tests/v2_env.rs` (and refresh `.git/tests-lock`) so `u2` is at Chebyshev distance `>= 3` from `p(2, 2)` (e.g., place Target 2 at `p(5, 2)`, `u3` at `p(5, 3)`, `u2` at `p(5, 4)`, Target 4 at `p(3, 4)`, and Target 3 at `p(7, 7)`, keeping `u2` at range 2 from Target 2 and Target 4 and out of range of Target 1 and Target 3).
  - B: Include `u2` in `t1_min_retal` (which would reduce `t1_min_retal` to `0.0` and lose coverage of non-zero multi-attacker `min_retal`).
- Agent's pick and why: Option A — moving `u2` out of range of Target 1 preserves the intended test of non-zero multi-attacker `min_retal` aggregation between `u0` (`8/9`) and `u1` (`6/15`) on Target 1.
- Owner answer: A (owner, 2026-11-27) — relocate `u2` (and Target 2/3/4, `u3`) so `u2` is at distance 3 from Target 1 while keeping distance 2 from Target 2 and Target 4.
