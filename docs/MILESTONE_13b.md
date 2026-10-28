# Milestone 13b: one stacked forward pass for all snapshot opponents

Depends on milestones 1 to 13. Gate 13a measured the interleaved rollout at a pool of 10 snapshots: the
opponent forward passes are 51% of rollout time, at ~4.4 calls per tick (one per snapshot group), each
costing ~6-7 ms almost regardless of batch size (171 kernel launches, 10 host-device syncs, ~1.1 ms of
GPU work). This milestone makes it one call per tick: the weights of the pool's snapshots are stacked,
and every snapshot's opponents are evaluated together with `torch.func`.

## Goal

1. One opponent forward per tick (per stage: actor, kind, argument), whatever the number of snapshot
   groups, with the same per-snapshot policies as today.
2. Greedy decisions identical to today's per-snapshot calls (up to float tolerance, see Rules); sampled
   decisions from the same distributions.
3. Measure the gain on the 13a fixture.

## Decisions already made (owner)

- Do the stacked forward (option (b) of the 13a recommendation), as its own milestone (2026-10-28).
- Batching the rollout buffer writes (option (a)) and the opening-turn truncation safeguard are not part
  of this milestone.
- No new dependencies (`torch.func` is part of torch).

## In scope

1. `polytopia_rl.stacked`: the stacked weights cache and the stacked batch policy (Public API).
2. `SelfPlayVecEnv.tick` uses the stacked policy when the config key `opponent_forward` is
   `"stacked"` (default), today's per-snapshot calls when it is `"per_snapshot"`.
3. The 13b measurement in RL_REPORT.

## Out of scope

The lockstep path (`step` / `_advance`, unchanged: it keeps per-snapshot calls), the learner's forward,
CUDA graphs, `torch.compile`, the buffer writes, the bots, matchmaking, the pool, PPO.

## Public API (fixed so tests can import it)

```python
# polytopia_rl.stacked
class StackedSnapshots:
    def __init__(self, base_net: PolicyNet, device: str): ...
    def sync(self, entries: list[Entry]) -> None
    #   (re)stack the parameters of `entries` on `device` when their names differ from the last sync;
    #   no-op otherwise. Rows are in the order of `entries`.
    def index(self, name: str) -> int     # row of an entry in the current stack

def stacked_batch_policy(stack: StackedSnapshots, generator, greedy: bool = False) -> Callable
#   policy(groups) where groups is a list of (entry_name, envs): returns one list of actions per group,
#   aligned with that group's envs. One vmapped call per stage over all groups.

# polytopia_rl.selfplay
SelfPlayVecEnv(..., stacked_policy: Callable | None = None)   # new keyword; None keeps the M13 calls
#   when given, tick() makes one stacked_policy(groups) call per tick instead of one make_policy(entry)
#   call per group, after calling stacked_policy.sync(pool.entries) (the policy exposes `sync`)
make_selfplay_vec_env(cfg, seed_base, *, pool, make_policy, rng, stacked_policy=None)

# polytopia_rl.config.PPOConfig
opponent_forward: str = "stacked"   # "stacked" | "per_snapshot"; anything else raises ValueError on load
#   the trainer builds a StackedSnapshots on its device from a copy of the learner's architecture and
#   passes stacked_batch_policy(stack, opponent generator, cfg.opponent_greedy) when "stacked"
```

The callable returned by `stacked_batch_policy` has a `sync(entries)` attribute that forwards to its
`StackedSnapshots`.

## Rules

### Stacked forward

- `sync` builds the stack with `torch.func.stack_module_state` over copies of `base_net` loaded with each
  entry's weights (frozen, no grad), moved to `device`. The vec env calls `sync(pool.entries)` at the
  start of every tick that has snapshot opponents to move; since the pool changes only when a snapshot is
  added (every `snapshot_every` steps), restacking is rare.
- A stacked call for groups `g = 0..G-1` with batch sizes `b_g`: gather the stacked parameters of the G
  entries (`index_select` on each stacked tensor), pad each group's observations to `B = max b_g` by
  repeating the group's first row, and evaluate `trunk`, then the actor logits, with
  `torch.func.vmap(functional_call(...))` over the G parameter sets. The valid rows (group order, then
  slot order inside the group) are flattened to `R = sum b_g` rows; masks are fetched for those rows
  exactly as `net_batch_policy` does; `categorical.sample` (with `check=False`, as on the play path)
  picks the actors for all R rows at once. The kind and argument stages repeat this with the chosen
  actors (and kinds), padded the same way. Padded rows never reach a mask, a sample, or an env.
- Play-only, as today: `torch.inference_mode`, no log-probs, entropies or values.
- Sampling: one `categorical.sample` call per stage over all R rows with the tick's opponent generator,
  so draws are assigned in (group, slot) order. Sampled runs are therefore not bit-identical to
  per-snapshot runs, but each row is drawn from the same distribution and runs are deterministic on the
  CPU for a fixed seed.
- Greedy: argmax per row; identical to the per-snapshot path whenever the logits agree, which they do to
  float tolerance (vmap may sum in a different order; on the GPU, TF32 adds the batch-shape effect
  recorded in Q34).

### In the vec env

- With `opponent_forward == "stacked"`, `tick` builds the groups exactly as today (snapshot groups in
  order of first appearance by slot, slots ascending inside a group) and makes one
  `stacked_batch_policy` call for all of them; actions go back to the same slots as today.
- With `"per_snapshot"`, `tick` is unchanged from M13.

## Clarifications (owner: confirm or overrule)

- **T1:** the default becomes `"stacked"`; `"per_snapshot"` reproduces M13 runs. - Confirmed (owner, 2026-10-28)
- **T2:** padding repeats a group's first row (always a valid observation), so no special handling of
  empty or zero rows is needed in the network; the padded outputs are discarded. - Confirmed (owner, 2026-10-28)
- **T3:** the stack holds the whole pool (at most `pool_max` = 40 entries, ~2 MB each at width 64, so
  ~80 MB on the GPU at most) and per tick gathers only the groups that move. - Confirmed (owner, 2026-10-28)

## Changes to earlier tests

None. The M13 tests use `per_snapshot`-style scripted policies through `make_policy`; if any of them
depends on the default opponent path in a way the new default breaks, that is a QUESTIONS.md entry
(next number Q38), not a test edit.

## Tests (phase A guidance)

1. **Stacked equals per snapshot (CPU):** for G = 1..4 tiny `PolicyNet`s with different random weights
   and groups of different sizes (including size 1), the stacked trunk features and the actor, kind
   and argument logits of every valid row equal each snapshot's own forward within `atol 1e-5`.
2. **Greedy decisions:** on real env positions (tiny map), the stacked greedy policy returns the same
   actions, per slot, as calling `net_batch_policy(net_g, ..., greedy=True)` per group (choose positions
   whose top-two logit gap exceeds 1e-3, or assert equality and check the gap).
3. **Alignment:** actions return to the right group and slot when groups have different sizes and
   orders; padded rows never fetch masks (mock envs that fail if asked) and never reach an env.
4. **Sampled distribution:** on a fixed position, the stacked sampled policy's action frequencies over
   many draws match the per-snapshot policy's within a loose chi-square bound; a fixed seed repeats.
5. **Cache:** `sync` with the same entries does not restack (spy); adding an entry restacks; later
   changes to the learner's weights do not change the stack (frozen copies); `index` follows the order.
6. **Vec env:** with `opponent_forward = "stacked"`, exactly one stacked call per tick that has snapshot
   moves (mock count), and with greedy snapshot opponents the per-slot trajectories of a short
   interleaved rollout equal those of `"per_snapshot"` (CPU, tiny nets, tie-free check as in 2).
7. **Config:** default `"stacked"`, `"per_snapshot"` selects the old calls, an unknown value raises on
   load; the lockstep path never uses the stacked policy.

Mutants to aim for: padded rows leaking into the sampled rows; actions returned to the wrong slot or
group; the wrong snapshot's parameters gathered (index off by one); a stale stack after a snapshot is
added; masks fetched for the wrong rows; greedy ignored; sampling per group instead of once per stage
(caught by the call/draw count); the learner's live weights aliased into the stack.

## Gate

**13b-a:** `./scripts/check.sh` passes. Then on the owner's machine (GPU idle): the 13a fixture
(`runs/gate13a/fixture.py`, resume `runs/exp/selfplay12b/pbot40/seed1/latest.pt`, pool of 10, 3
rollouts) with `opponent_forward=per_snapshot` and `stacked`, two runs each alternated: rollout and total
sps, opponent calls per rollout, time per opponent call, share of rollout time in opponent calls; plus
the same with a pool of 1 (the M10 checkpoint, a fresh run) to confirm no slowdown at the start of a
run. A 1M-step smoke run with the default config completes. Results in RL_REPORT "Milestone 13b". No
performance threshold; the 13a estimate is ~1.5x on the rollout at pool 10.

## Notes for the implementer

- `torch.func.stack_module_state`, `functional_call` and `vmap` work on this network: no batch norm,
  and every op in the trunk and heads (conv, gather, where, embedding, einsum) is vmappable. The heads'
  Python loops over kinds are fixed-length and fine under vmap.
- Keep `net_batch_policy` and `make_policy` unchanged; they remain the reference and the
  `per_snapshot` path.
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry (next number Q38).
