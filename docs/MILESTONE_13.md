# Milestone 13: interleaved self-play rollout (speed)

Depends on milestones 1 to 12c. Self-play trains at ~200-300 sps once the snapshot pool has grown, so a
10M-step run takes ~10 h and every experiment costs a day or more. M14 (architecture and scaling) will
multiply the number of runs. This milestone removes the main cost the profile below found, without
changing what the learner learns from.

## Profile (orchestrator, 2026-10-27, before any change)

Resuming `runs/exp/selfplay12b/pbot40/seed1/latest.pt` (pool of 10 snapshots, `p_bot` 0.4, 128 envs x 32
steps) for 3 rollouts, RTX 4060, `OMP_NUM_THREADS=1`:

| measure | value |
| --- | --- |
| rollout time in opponent policy calls | 33.4 s of 36.4 s (92%) |
| opponent calls | 5,604 = 58 per learner step |
| cost per call | 6.0 ms (Q30: ~124 kernel launches and ~19 host-device syncs, ~0.5 ms of GPU work) |
| batch per call | mean 1.7 envs, median 1 |
| rounds of opponent moves per learner step | ~26 |
| snapshot groups per round | mean 2.2 (1 group in 40% of rounds, max 7) |
| update (PPO epochs) | 4.8 s of 41.2 s |

**The cause is the per-step barrier.** After each learner step `SelfPlayVecEnv.step` plays every
snapshot opponent's whole turn before returning (`_advance`). All 128 slots wait for the longest
opponent turn among the slots that just ended their turn (~26 rounds), and each round makes one tiny
call per snapshot group. Grouping all snapshots into one call (stacking) would cut calls only ~2.2x;
removing the barrier cuts them by about an order of magnitude, because learner and opponent moves of
different slots then share the same ticks and batches.

## Goal

1. A self-play rollout in which, at every tick, each slot makes exactly one move, the learner's or its
   opponent's, so no slot waits for another slot's opponent turn.
2. Per slot, the same game, rewards, observations and transitions as today; only the order across slots
   changes.
3. Measure the speed-up on the profile fixture above and re-profile to choose the next speed step.

## Decisions already made (owner)

- Speed work comes before M14 (roadmap, 2026-10-27). Profile first (CLAUDE.md); the profile above is
  the justification.
- Learning is unchanged: the learner still trains only on its own transitions, rewards are the M9
  formula over the learner's action and the whole opponent turn (M11 L11), PPO and GAE are unchanged.
- No new dependencies.

## In scope

1. `SelfPlayVecEnv` gains the tick interface (Public API) next to its existing `step`.
2. `ppo.collect_rollout_interleaved` fills the same `RolloutBuffer` ([T, N] layout) from it.
3. Config key `rollout: "interleaved" | "lockstep"` (`PPOConfig`, default `"interleaved"`); the trainer
   uses the interleaved collector for self-play when it is `"interleaved"`, today's path otherwise. Bot
   training (non-self-play) is unchanged.
4. The before/after measurement and a re-profile in RL_REPORT.

## Out of scope

Stacking snapshot weights into one batched forward (`torch.func`), CUDA graphs, `torch.compile`, moving
mask fetching into Rust, subprocess workers, any change to matchmaking, the pool, rewards, PPO, GAE,
the buffer layout, the bots, or the game. These are the candidates for a later speed step, chosen from
the re-profile.

## Public API (fixed so tests can import it)

```python
# polytopia_rl.selfplay.SelfPlayVecEnv (new methods; reset_all, step, state_dict, ... unchanged)
def learner_slots(self) -> list[int]              # slots whose learner is to move, ascending
def tick(self, learner_actions: dict[int, tuple[int, int, int]]) -> list[Transition]
#   applies one learner action to every slot in learner_actions (all must be in learner_slots()),
#   then one opponent move to every other slot whose snapshot opponent is to move (one batch-policy
#   call per snapshot, as today), handles episode ends and auto-resets, and returns the transitions
#   completed during this tick, in ascending slot order.

@dataclass
class Transition:
    slot: int
    reward: float          # the M9 formula over the learner's action and everything up to the
                           # learner's next move or the episode end (float32 arithmetic as today)
    terminated: bool
    truncated: bool
    final_board: np.ndarray | None    # as VecStep: the final observation when the episode ended
    final_globals: np.ndarray | None

# polytopia_rl.ppo
def collect_rollout_interleaved(net, vec_env, buffer, generator) -> list[EpisodeStat]

# polytopia_rl.config.PPOConfig
rollout: str = "interleaved"   # "interleaved" | "lockstep"; anything else raises ValueError on load
```

## Rules

### Ticks

- A slot is in one of three states: learner to move; opponent (snapshot) to move; or waiting, which
  means it has a learner action whose transition is not complete yet (its opponent is still moving).
  Bot-opponent slots never wait: the Rust env plays the bot's turn inside the learner's step, as today.
- `tick(actions)`: first apply each given learner action (in ascending slot order; record the scores
  before it for snapshot slots, as `step` does today). Then, for every slot (including ones that just
  received a learner action) whose opponent is to move, make exactly one opponent move, grouped by
  snapshot, one batch-policy call per group, groups in order of first appearance by slot. A slot's
  transition completes when, after this, it is on the learner's move or its episode ended. Bot slots
  complete in the same tick as their learner action.
- Episode end: as today (`_ended`, `final_board/globals`, `EpisodeStat`, then `_reset`). After a reset
  the slot's opponent may be to move (learner seat 1); it is then advanced by later ticks like any
  other slot, never inside the reset.
- Rewards: a snapshot slot's reward is `_snapshot_reward` with the scores before the learner's action
  and the scores when the transition completes, exactly as today; a bot slot's reward is the env's.

### Collecting a rollout

- `collect_rollout_interleaved` keeps a per-slot transition count `t_i`. Each loop iteration: the slots
  in `learner_slots()` with `t_i < T` get one learner forward pass together (`net.act` with stats, the
  training path, sampling with `generator`); their observation, masks, action, log-prob and value go to
  `buffer[t_i, i]`; then `tick(...)` runs; each returned transition writes reward, terminated, truncated
  (and `next_values` from the final observation for truncation, as today) at `buffer[t_i, i]` and
  increments `t_i`. Slots with `t_i == T` receive no learner actions, but their opponents keep moving
  until they are on the learner's move.
- The rollout ends when every slot has `t_i == T` and is on the learner's move. The bootstrap value of
  each slot is the value of its current observation, as today.
- Episode stats are returned in completion order (the trainer's metrics treat them as a set).

### Equivalence (the main correctness rule)

With the same learner decisions and the same opponent decisions, every slot's sequence of
(observation, masks, action, reward, terminated, truncated, final observation) under
`collect_rollout_interleaved` equals its sequence under today's `collect_rollout`. Only the interleaving
across slots, and therefore which random draw goes to which decision, differs. Runs with
`rollout: "interleaved"` are therefore not bit-identical to earlier runs, but are deterministic on the CPU
for a fixed seed, and resume exactly (the vec env is on the learner's move in every slot at a rollout
boundary, as today).

## Clarifications (owner: confirm or overrule)

- **S1:** the default becomes `"interleaved"`. Runs before M13 (11b, 12b, 12c) used the lockstep path;
  `rollout: "lockstep"` reproduces them as before. Their results stay comparable in distribution, since
  the per-slot game is unchanged. - Confirmed (owner, 2026-10-27)
- **S2:** slots that finished their `T` transitions idle until the slowest slot finishes. The barrier
  moves from every step to the end of each rollout (once per 32 steps), which is the trade. - Confirmed (owner, 2026-10-27)
- **S3:** stacking snapshots (`torch.func`) and CUDA graphs wait for the re-profile, because the profile
  shows the barrier, not the per-snapshot split, as the main cost. - Confirmed (owner, 2026-10-27)

## Changes to earlier tests

None expected. The existing `SelfPlayVecEnv` tests exercise `step` (unchanged) and the lockstep path. If
an existing trainer test depends on the default rollout path in a way the new default breaks, that is a
QUESTIONS.md entry, not a test edit.

## Tests (phase A guidance)

1. **Equivalence (the core test):** CPU, both seats, shaping on and off, a pool with two snapshot entries
   plus bots. Learner = a deterministic scripted policy (for example a stub net whose `act` returns
   `Env.bot_action("rush")` for each slot's learner seat, with fixed log-probs and values); snapshot opponents = scripted batch policies built on `Env.bot_action`. Run today's
   `collect_rollout` and `collect_rollout_interleaved` on identical vec envs and compare, per slot, the
   full sequence of buffer rows and episode stats (as multisets). Include episodes that end by capital
   capture, by the turn cap, and by truncation (a small `max_agent_actions`), and an opponent-first
   reset (learner seat 1).
2. **Tick mechanics:** `learner_slots()` content; `tick` rejects an action for a slot not in it; each
   tick gives every non-waiting slot exactly one move; one batch-policy call per snapshot group per tick
   (mock counting calls); bot slots complete in the same tick; transitions are returned in ascending slot
   order; an opponent-first reset is advanced by ticks, not inside `_reset`.
3. **Rollout end:** every slot has exactly `T` rows; finished slots get no learner actions; the rollout
   ends only when every slot is on the learner's move; bootstrap values come from those observations.
4. **Determinism and resume (CPU, tiny config):** two identical interleaved runs are identical; a resume
   at a rollout boundary continues exactly.
5. **Config:** `rollout` defaults to `"interleaved"`, `"lockstep"` uses the old collector (spy), an
   unknown value raises on load; bot (non-self-play) training never uses the interleaved collector.
6. **Call count:** on a fixture where the old path needs `R` rounds of opponent calls per learner step,
   the interleaved path makes at most (1 learner call + one call per snapshot group) per tick (mock
   counters), so the number of calls per rollout drops; assert the counts, not wall time.

Mutants to aim for: a slot making two moves in one tick; the learner acting while its slot waits; the
reward computed from the scores at the opponent's first move instead of the turn end; the transition
written to the wrong `t_i`; a reset advancing the opponent inside `_reset`; the rollout ending while a
slot's opponent is mid-turn; finished slots still acting; truncation's `next_values` missing; groups
merged across snapshots.

## Gate

**13a (implementation):** `./scripts/check.sh` passes. Then, on the owner's machine:
1. The profile fixture above (resume `runs/exp/selfplay12b/pbot40/seed1/latest.pt` for 3 rollouts into a
   scratch run directory) with `rollout=lockstep` and `rollout=interleaved`: rollout sps, update sps,
   calls per learner step, mean batch per call, groups per tick, share of rollout time in policy calls.
2. A 2M-step self-play smoke run from the M10 checkpoint with the interleaved path (seed 1, 128 x 32):
   it completes, and its in-training rows at 1M and 2M are within the 12b `pbot40` spread at the same
   steps (a sanity check that learning is not broken, not a statistical test).
3. The concurrency table of M12 (1, 2, 3, 4 trainers) with the interleaved path, to set
   `--parallel` in `scripts/experiment.sh`.
4. A short re-profile: where the remaining rollout time goes, and a recommendation between stacking
   snapshots, CUDA graphs and nothing.
Paste the tables into RL_REPORT. No performance threshold; the expectation from the profile is a
rollout several times faster, and a much smaller gain should be explained.

## Notes for the implementer

- Keep `step` and `collect_rollout` as they are; the interleaved path is new code next to them, and the
  lockstep path stays the reference for the equivalence test.
- Reuse `_snapshot_reward`, `_episode_stat`, `_reset` and the per-snapshot policy cache.
- Do not read anything from `Env.state`; policies get `observation()` and masks only.
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry (next number Q36).
