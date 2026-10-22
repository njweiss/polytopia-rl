# Milestone 10: PPO training against the bots

The spec's milestone 10 ("PPO against the bots, then self-play"). Depends on milestones 1 to 9. This
milestone is Python and PyTorch only: no Rust changes unless the Python layer finds a real bug in
milestone 9 (then stop and record it in `docs/QUESTIONS.md`). Self-play and a flat-action baseline are
follow-ups, not part of this milestone.

## Goal

A CleanRL-style PPO trainer with the spec's factored, conditionally masked action heads, a vectorized
environment, an opponent curriculum (random, then economy bot, then a mix), a rigorous side-swapped
evaluator, and logging good enough to see whether and why learning works. The owner's priority is
understanding RL, so every piece is small, readable and unit-tested on its own.

## Decisions already made (owner)

- **Library:** none. A hand-written single-file-style PPO in PyTorch (this closes open question 2 of the
  spec). Library PPOs do not support the conditional actor -> type -> argument masks.
- **Compute:** one machine with an RTX 4060 (8 GB) and a Ryzen 5 5600X3D (6 cores). Training must run
  there; CPU-only must work for tests and smoke runs.
- **Scope and gate:** PPO against the bots with a curriculum; the gate is beating random and the
  economy bot in side-swapped evaluation. Rush is reported, not gated.

## In scope

1. Python package `python/polytopia_rl/` (pure Python on top of `polytopia_sim`):
   `config.py`, `vec_env.py`, `categorical.py`, `model.py`, `gae.py`, `ppo.py`, `curriculum.py`,
   `evaluate.py`, `train.py`, `plot.py`.
2. JSON configs in `python/configs/`, scripts `scripts/rl_check.sh`, `scripts/train.sh`, `scripts/rl_gate.sh`.
3. pytest tests in `python/tests/rl/` (CPU, fast).
4. A training report `docs/RL_REPORT.md` written by the implementer after the smoke run.

## Out of scope (do not build)

Self-play, league or Elo, a flat-index policy, recurrent or transformer models, multi-GPU, subprocess
vector envs (measure first; a `SyncVecEnv` in one process is the baseline), mixed precision, `torch.compile`,
hyperparameter sweeps, W&B or TensorBoard, imitation learning. New dependencies approved by this brief
and no others: Python `torch`, `numpy`, `matplotlib` (plots only), and `pytest` (dev). Pin versions in
`python/requirements-rl.txt` and record the CUDA build used.

## Public API (fixed so tests can import it)

```python
# polytopia_rl.config
@dataclass
class PPOConfig:                      # defaults in the table below; to_json()/from_json(); override via "key=value"
    ...
def load_config(path: str, overrides: list[str] = ()) -> PPOConfig

# polytopia_rl.vec_env
class SyncVecEnv:
    def __init__(self, env_fns: list[Callable[[], Env]], seed_base: int, seed_stride: int = 1)
    num_envs: int
    def reset_all(self) -> VecState                       # resets every env with the next seeds
    def step(self, actors, kinds, args) -> VecStep        # int arrays of length num_envs; auto-resets finished envs
    def observations(self) -> tuple[np.ndarray, np.ndarray]   # board (E,C,H,W) f32, globals (E,G) f32
    def actor_masks(self) -> np.ndarray                   # (E, 3N+1) bool
    def kind_masks(self, actors) -> np.ndarray            # (E, 10) bool
    def arg_masks(self, actors, kinds) -> np.ndarray      # (E, N) bool, padded with False beyond arg_len; no-arg kinds: all False
    def episode_stats(self) -> list[EpisodeStat]          # finished episodes since the last call
@dataclass VecStep: reward (E,) f32, terminated (E,) bool, truncated (E,) bool,
                    final_board/final_globals for envs that just finished (E,...) — the observation BEFORE the auto-reset
@dataclass EpisodeStat: env_index, seed, opponent, agent_seat, outcome ("win"|"loss"|"draw"), length (agent actions), ret (undiscounted return)

# polytopia_rl.categorical
def masked_logits(logits, mask) -> Tensor                 # false -> a large negative constant; raises if a row has no true entry
def sample(logits, mask, generator, greedy=False) -> LongTensor
def log_prob(logits, mask, index) -> Tensor
def entropy(logits, mask) -> Tensor                       # exact entropy of the masked distribution; 0 when one option

# polytopia_rl.model
class PolicyNet(nn.Module):
    def __init__(self, spec: dict, width: int = 64, blocks: int = 4)   # spec = Env.spec()
    def trunk(self, board, globals) -> Features                       # computed once per batch
    def actor_logits(self, feats) -> Tensor                           # (B, 3N+1)
    def kind_logits(self, feats, actors) -> Tensor                    # (B, 10), conditioned on the chosen actor
    def arg_logits(self, feats, actors, kinds) -> Tensor              # (B, N) for tile kinds, (B, A) for enum kinds, padded to (B, N); conditioned on actor and kind
    def value(self, feats) -> Tensor                                  # (B,)
    def act(self, feats, masks_fn, generator, greedy=False) -> ActOut # staged sampling; masks_fn(stage, ...) fetches masks lazily
    def evaluate(self, board, globals, actors, kinds, args, actor_mask, kind_mask, arg_mask) -> EvalOut   # logp, entropy (total and per head), value
@dataclass ActOut: actors, kinds, args, logp, entropy, value
@dataclass EvalOut: logp, entropy, head_entropy {"actor","kind","arg"}, value

# polytopia_rl.gae
def compute_gae(rewards, values, terminated, truncated, next_values, last_value, gamma, lam) -> (advantages, returns)

# polytopia_rl.ppo
def ppo_loss(new_logp, old_logp, adv, new_value, old_value, returns, entropy, cfg) -> LossOut   # pg, value, entropy terms, total, approx_kl, clipfrac
def minibatch_indices(n, num_minibatches, generator) -> list[LongTensor]
class RolloutBuffer, def train_update(...)

# polytopia_rl.curriculum
def next_stage(stage, history, cfg) -> int                # pure function

# polytopia_rl.evaluate
def evaluate(policy_fn, opponent, seeds, env_kwargs, greedy) -> EvalResult    # each seed played twice, seat 0 then seat 1
@dataclass EvalResult: games, wins, losses, draws, win_rate, wilson_lo, wilson_hi, by_seat, mean_length, mean_score_diff, per_game list

# CLI
python -m polytopia_rl.train --config python/configs/stage1_random.json [key=value ...]
python -m polytopia_rl.evaluate --checkpoint CKPT --opponents random,economy,rush --seeds 200
python -m polytopia_rl.plot --run RUNDIR
```

## Rules

### Environment wiring

- Every training env is `polytopia_sim.Env(opponent=<kind>, agent_seat="seed_parity",
  shaping_weight=cfg.shaping_weight, gamma=cfg.gamma, ...)` with the map and turn cap from the config
  (defaults `size=11`, `turn_cap=30`, `full_observability=False`).
- `SyncVecEnv` gives env `i` the episode seeds `seed_base + i * seed_stride + k * num_envs * seed_stride`
  for its k-th episode (so seeds never repeat within a run). Auto-reset happens inside `step`, after
  recording the final observation and the episode stat. A finished env's next observation is the reset
  one; `VecStep.final_*` holds the one before.
- Training seeds are `< 1_000_000_000`; evaluation seeds are `>= 1_000_000_000`. Never mixed.
- Per-env opponents: `cfg.opponents` is a list assigned round-robin to envs (stage 3 uses
  `["economy", "rush"]`), `cfg.opponent_epsilon` applies to all bot opponents.

### Policy (spec: factored heads, four actor slots)

- **Trunk:** a 3x3 conv stem (`C -> width`), `blocks` residual blocks (conv-BN-free: conv, ReLU, conv,
  add, ReLU), producing a feature map `F` of shape `(width, H, W)`. The global vector goes through an MLP
  and is concatenated with spatial mean- and max-pooled `F` into a context vector `g`.
- **Actor head (stage 1):** a 1x1 conv gives 3 logit maps (unit, city, site families), flattened
  family-major to match the actor index `fam * N + tile`, plus one global logit from a linear layer on
  `g`: `3N + 1` logits, masked by `actor_mask`.
- **Kind head (stage 2):** context `z = [F at the actor's tile (or a learned vector for the global actor),
  g, family embedding]` -> MLP -> 10 logits, masked by `kind_mask(actor)`.
- **Argument head (stage 3):** `c = MLP([z, kind embedding])`. For `Move` and `Attack` the tile logits are
  `(1x1 conv of F) . c / sqrt(width) + bias_kind` per tile (`N` logits). For `Train`, `ChooseReward`,
  `Site`, `Research` a separate linear head per kind gives `arg_len(kind)` logits. Kinds with no argument
  (`Capture, Recover, Promote, EndTurn`) have no argument stage. Logits are padded to `N` and masked by
  `arg_mask(actor, kind)`.
- **Value head:** MLP on `g`.
- Orthogonal init (gain `sqrt(2)`; policy heads 0.01; value 1.0), as in CleanRL. No weight sharing
  between the three stage heads beyond the trunk.
- **Log-probability** of an action is the sum over the stages that fired (an argument-less kind adds no
  argument term). **Entropy** is the sum of the conditional entropies along the sampled path, i.e. the
  entropy of each fired stage's masked distribution given the earlier choices (spec). `evaluate` also
  returns the three per-head entropies (zero for an unfired stage).
- `act` runs the trunk once, then samples the three stages in order, fetching masks lazily through
  `masks_fn` after each stage. `evaluate` recomputes everything from stored `(obs, actor, kind, arg)` and
  the stored masks and must agree with `act` to float tolerance.

### PPO (CleanRL conventions)

- Rollout `num_steps x num_envs`; one step is one agent action. Store board, globals, the three chosen
  indices, the three stage masks (actor, kind, arg padded), old logp, value, reward, terminated,
  truncated.
- **GAE** with per-env bootstrapping. A terminated step has no bootstrap. A truncated step bootstraps from
  the value of its `final` observation (computed in a batch after the step). The rollout end bootstraps
  from the value of the current observations. Formula: `delta_t = r_t + gamma * V_next * (1 - term_t) -
  V_t`, `A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}` where `done = terminated or truncated` and
  `V_next` is the final-observation value on a truncation. `returns = advantages + values`.
- **Loss:** clipped surrogate `max(-A * r, -A * clip(r, 1-eps, 1+eps))` with advantages normalized per
  minibatch (config flag), value loss `0.5 * (V - R)^2` (optionally clipped, config flag), entropy bonus,
  `total = pg + vf_coef * v - ent_coef * entropy`. Global grad-norm clipping, Adam with `eps=1e-5`,
  linear LR annealing, minibatches by shuffling all `num_steps * num_envs` samples, `update_epochs`
  epochs, optional early stop when `approx_kl > target_kl`.
- Metrics per update (one JSON line to `RUNDIR/metrics.jsonl`): `global_step`, `update`, `sps`, `stage`,
  `lr`, `pg_loss`, `v_loss`, `entropy` (total and per head), `approx_kl`, `clipfrac`,
  `explained_variance`, `mean_return`, `train_win_rate` (last 200 finished episodes), `mean_episode_len`,
  `kind_freq` (fraction of each action kind chosen), `end_turn_freq`, `adv_mean/std`.

### Default hyperparameters (all in `PPOConfig`, owner may change)

| name | default | | name | default |
| --- | --- | --- | --- | --- |
| `num_envs` | 32 | | `gamma` | 0.999 |
| `num_steps` | 128 | | `gae_lambda` | 0.95 |
| `num_minibatches` | 8 | | `clip_eps` | 0.2 |
| `update_epochs` | 4 | | `ent_coef` | 0.01 |
| `lr` | 3e-4 (annealed to 0) | | `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 | | `target_kl` | none |
| `width` / `blocks` | 64 / 4 | | `norm_adv` | true |
| `shaping_weight` | 1.0 | | `potential_scale` | 1000 |
| `terminal_weight` | 1.0 | | `total_steps` | 20_000_000 |
| `seed` | 1 | | `device` | `"auto"` (cuda if available) |

### Curriculum

Stages in the config: `[{"opponents": ["random"]}, {"opponents": ["economy"]}, {"opponents": ["economy",
"rush"]}]`. `next_stage(stage, history, cfg)` advances when at least `cfg.min_stage_steps` have been
trained in the stage and the win rate over the last `cfg.stage_window` (200) finished episodes is
`>= cfg.stage_win_rate` (0.8 for stages 1 and 2); the last stage never advances. On advance, the envs are
rebuilt with the new opponents and fresh seeds, the optimizer and policy persist, and the LR schedule
continues. The stage index is logged and saved in checkpoints.

### Evaluation

- `evaluate` plays `seeds` distinct evaluation seeds, each **twice with sides swapped** (one game as seat
  0 and one as seat 1 on the same map, via two `Env`s with `Fixed` seats), against one opponent
  (`epsilon 0`), for `greedy` (argmax of every masked head) or sampled actions. It reports counts, win
  rate and the Wilson 95% interval, win rate per seat, mean game length and mean final score difference.
- `policy_fn(env) -> (actor, kind, arg)` is the interface for any policy (the net, or a scripted test
  policy); the net wrapper lives in `evaluate.py`.
- Evaluation during training runs every `eval_every` steps (default 500_000) against the current stage's
  opponents with 50 seeds, greedy, and logs to `eval.jsonl`; the best checkpoint by the final-stage
  score is kept as `best.pt`.
- Checkpoints (`ckpt_{step}.pt`, `latest.pt`, `best.pt`) contain model and optimizer state, config JSON,
  `global_step`, stage, the next training seed index, and RNG states for a clean resume.

### Reproducibility

Seeding sets Python, NumPy, torch and a dedicated `torch.Generator` for sampling. On CPU with a fixed
seed a run is bit-for-bit repeatable (tested on a tiny config). GPU runs are statistically repeatable
only.

## Clarifications to the spec (owner: confirm or overrule, then update the doc)

- **K1: library choice** is "write it yourself, CleanRL-style". - Confirmed
- **K2: gate thresholds** (my proposal): the final greedy evaluation over 200 seeds (400 games) wins
  at least **90% vs random** and at least **60% vs the economy bot**; rush is reported. Win means the
  game outcome (capture or higher score at the cap); draws are not wins. - Confirmed
- **K3: shaping default is on** (`shaping_weight 1.0` on score difference) because terminal-only reward
  is very sparse over ~100s of steps; the implementer's smoke run includes a `shaping_weight=0` ablation
  for the report. - Confirmed
- **K4: `gamma = 0.999`** since an episode is hundreds of single-action steps and the winner is decided
  at the end. Shaping uses the same gamma, so it stays policy-invariant. - Confirmed
- **K5: turn cap 30** and map size 11 stay at their M9 defaults here. Spec open question 1 (30/40/50) is
  decided by the owner after seeing the bot stats; changing it only changes a config value. - Confirmed
- **K6: eval seeds** start at 1e9, disjoint from training seeds. - Confirmed
- **K7: stage promotion uses training win rate** (opponent fixed within a stage, epsilon 0 for the
  scripted bots), not the periodic eval. - Confirmed
- **K8: entropy** is the sum of conditional entropies along the sampled path, as the spec says; it is an
  approximation of the joint entropy and is what the entropy bonus uses. - Confirmed
- **K9: `SyncVecEnv` first.** Subprocess workers only if profiling shows the Rust env, not the network,
  is the bottleneck. - Confirmed

## Changes to earlier tests

None. `scripts/check.sh` calls `rl_check.sh` when `torch` is importable and prints `SKIPPED rl tests
(torch not found)` otherwise; the test-lock list gains `python/tests/rl/`.

## Tests (phase A guidance; pytest, CPU, whole suite well under 60 s)

Golden values are computed by hand or by independent NumPy reference code in the test file, never
imported from the module under test.

1. **categorical:** masked softmax on hand-worked logits; sampled indices never masked; `log_prob` and
   `entropy` equal a NumPy reference; one valid option gives entropy 0 and logp 0; an all-false row
   raises; greedy picks the masked argmax; sampling frequencies match probabilities (large n, tolerance);
   a fixed generator gives a fixed sample.
2. **vec_env:** shapes and dtypes; the episode seed sequence is exactly the formula above and never
   repeats; auto-reset puts the reset observation in the next step and the pre-reset one in `final_*`;
   `terminated` vs `truncated`; per-env opponent assignment; `EpisodeStat` fields on a scripted finished
   episode (outcome from the seat's perspective); mask helpers agree with direct `Env` calls; two vec envs
   with the same seeds evolve identically.
3. **model structure:** output shapes; actor index layout (a unit tile and the same tile's city slot
   are different logits); kind logits depend on the chosen actor (changing the actor changes the logits);
   argument logits depend on the kind; a no-argument kind adds nothing to logp (perturbing every
   argument head's weights leaves its logp unchanged); gradients reach only the fired heads;
   `evaluate` equals the logp, entropy and value recorded by `act` for the same actions (tolerance
   1e-5), over a batch of real env states; batch rows are independent (permutation invariance).
4. **action validity (mask correctness through the net):** on real `Env`s with random-weight nets, 2000
   sampled `(actor, kind, arg)` paths all step without `ValueError`, including pending-reward states;
   no sampled path ever picks a masked option.
5. **GAE:** hand-computed small cases (single env; two envs; a terminated step; a truncated step with
   its final-observation bootstrap; the rollout-end bootstrap); an independent reference loop in the test
   file on random inputs; `returns = adv + values`; gamma = 0 and lam = 0 reduce to one-step TD.
6. **PPO loss:** hand-computed tiny batches for the clipped surrogate both signs of advantage, inside
   and outside the clip; value loss; entropy sign; `approx_kl` and `clipfrac` definitions; at ratio 1 the
   gradient of the surrogate equals `-A * grad logp`; old logp carries no gradient;
   `minibatch_indices` covers every sample exactly once; per-minibatch advantage normalization.
7. **Learning smoke (the pipeline's end-to-end test):** a `FakeEnv` implementing the same `Env` methods
   (small board, a few legal actions with masks, reward +1 only when a specific factored action path is
   chosen) trains with the real `train_update` and reaches a high success rate within a bounded number of
   updates on CPU, for a fixed seed. A second variant needs a two-stage choice (the right kind AND the
   right argument), so it exercises the conditional heads.
8. **Curriculum:** `next_stage` truth table (not enough steps, window not full, below/at/above the
   threshold, last stage); env rebuild keeps the policy and optimizer.
9. **Evaluate:** each seed played twice with swapped seats on the same map; seeds are `>= 1e9`;
   counts add up; the Wilson interval on known counts; a scripted always-EndTurn policy loses almost
   every game to `economy`; a scripted policy imitating the random bot wins about half against
   `random` (loose tolerance); greedy evaluation is deterministic; `by_seat` correct.
10. **Checkpoint and resume:** save, load, same outputs; resume continues the exact seed sequence and
    (on CPU, tiny config) matches an uninterrupted run's metrics; config round trip and `key=value`
    overrides (bad keys error).
11. **Train driver (tiny config, CPU):** a few updates run, `metrics.jsonl` has the documented keys and
    finite values, stage fields present, `plot` runs and writes files.

Mutants to aim for: actor family offsets mixed up; kind or argument head not conditioned on the actor;
masks not applied (or applied after softmax); fired-head rule wrong (argument term added for a
no-argument kind); entropy summed over unfired heads; `evaluate` and `act` disagreeing; GAE off by one,
bootstrapping on terminal or not on truncation, `done` mask reversed; ratio computed with detached new
logp; advantage normalization across the whole rollout instead of per minibatch; minibatches not
covering the data; auto-reset losing the final observation; seeds repeating across episodes or
overlapping eval seeds; evaluation not swapping seats or reusing the training seed range; stage advancing
on too few steps; LR annealing off by one; checkpoint missing optimizer state or the seed index.

## Gate

**10a (implementation, done in Claude Code):** `./scripts/check.sh` passes (Rust, Python M9 tests, and
`rl_check.sh` with the suite above), then a smoke run of the implementer's own on the owner's machine:
`scripts/train.sh python/configs/stage1_random.json total_steps=2000000`. Report the metrics table
(sps, entropy, kl, clipfrac, explained variance, win rate vs random over time) and the greedy 200-seed
evaluation vs random. **The pipeline passes if win rate vs random reaches at least 75%.** If it does not,
do not change the rules or tests: diagnose with the logged metrics (entropy collapse, `end_turn_freq`,
value explained variance, KL) and write the findings and a proposed fix in `docs/RL_REPORT.md`.
Also run `shaping_weight=0` for the same budget and record the comparison.

**10b (the real result, run by the owner):** `scripts/train.sh python/configs/curriculum.json`, then
`scripts/rl_gate.sh` (greedy, 200 seeds side-swapped vs random, economy and rush) must show >= 90% vs
random and >= 60% vs economy (K2). The owner reads the report and decides next steps (turn cap,
self-play).

## Notes for the implementer

- Run the trunk once per rollout step and the three heads staged; fetch masks lazily per stage so you
  never build the full dense (actor x kind x argument) mask.
- Keep each file short and commented for a reader new to PPO: say what each loss term does and why.
- Do not read anything from `Env.state`; the policy only gets `observation()` and masks.
- Measure `sps` for env stepping and for the forward pass separately and put both in the report; if
  the Rust env is under 20% of the step time, note that subprocess workers are not worth it.
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry.
