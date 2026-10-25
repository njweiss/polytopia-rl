# Milestone 11: evaluation suite, versioning and snapshot-pool self-play

Beyond the spec's ten milestones (the spec's "then self-play"). Depends on milestones 1 to 10. The M10
agent beats every scripted bot at 97 to 100%, so the bots can no longer measure progress. This
milestone builds the measuring instruments first (a fixed evaluation suite, spec hashes in checkpoints,
Elo ratings), then self-play that uses them. The combat-preview observation change is a separate later
milestone and will bump the observation hash introduced here.

## Goal

1. Make every result comparable: one fixed evaluation suite, one rating tool, and checkpoints that refuse
   to load into a game or observation they were not trained for.
2. Run a baseline on the M10 checkpoint that answers the robustness questions (noisy opponents, full
   observability, how it loses to rush).
3. Train by self-play against a pool of past snapshots mixed with the scripted bots, starting from the M10
   checkpoint, and show whether it improves.

## Decisions already made (owner)

- Self-play uses a snapshot pool plus a share of bot games (not pure self-play).
- Keep the M10 architecture and observation unchanged; the combat preview and any architecture or
  capacity experiment are later milestones.
- Compute stays one RTX 4060; runs of several hours are acceptable.

## In scope

1. **Rust (`sim-env`, `sim-py`):** `obs_hash` and `rules_hash`, exposed in `Env.spec()`.
2. **Python (`polytopia_rl/`):** `checkpoint.py` (factor save/load out of `train.py` if needed; the file
   format stays a superset), `suite.py`, `losses.py`, `rating.py`, `tournament.py`, `selfplay.py`,
   `selfplay_train.py` (or a `--selfplay` mode of `train.py`; keep one trainer), configs
   `suite.json`, `selfplay.json`.
3. Scripts `scripts/suite.sh`, `scripts/selfplay.sh`, `scripts/tournament.sh`; a report section in
   `docs/RL_REPORT.md`.

## Out of scope (do not build)

Combat-preview features, architecture changes, distillation, learning from the opponent's moves, PFSP
(prioritized matchmaking), league exploiters, population-based training, new dependencies (numpy and
torch only), subprocess envs, changes to any game rule or to the bots.

## Public API (fixed so tests can import it)

```rust
// sim_env::hash
pub fn fnv1a64(bytes: &[u8]) -> u64;                       // FNV-1a, offset 0xcbf29ce484222325, prime 0x100000001b3
pub fn obs_hash(size: u8) -> String;                       // 16 lowercase hex digits
pub fn rules_hash() -> String;                             // 16 lowercase hex digits
```

`obs_hash(size)` hashes the canonical string: `size`, then in order `channel_names`, `global_names`,
the 10 action-type names, `arg_len` per kind, and the enum name lists for unit types, rewards, techs and
site actions, each item joined with `\n` (the implementer writes it down in a code comment and pins it with
a golden test). `rules_hash()` hashes the concatenated bytes of every `data/*.json` the simulator
embeds, in file-name order, each preceded by its file name and a `\0`. `Env.spec()` gains the keys
`obs_hash`, `rules_hash`.

```python
# polytopia_rl.checkpoint
def save_checkpoint(path, *, model, optimizer, cfg, global_step, stage, seed_index, rng_state, spec, extra=None) -> None
def load_checkpoint(path, *, spec, strict_obs=True, strict_rules=False) -> Checkpoint
#   raises SpecMismatch if obs_hash differs (strict_obs) ; rules_hash mismatch warns (or raises if strict_rules)

# polytopia_rl.suite
SUITE_SEED_BASE = 1_000_000_000                             # suite seeds: [1e9, 1.5e9)
@dataclass Condition: name, opponent ("random"|"economy"|"rush"), epsilon, full_observability, greedy
DEFAULT_SUITE: list[Condition]                              # see Rules
def run_suite(policy_fn, conditions, seeds, env_kwargs) -> SuiteResult      # evaluate() per condition
def compare(a: SuiteResult, b: SuiteResult) -> list[Comparison]            # per-condition difference with a two-proportion z test
def suite_markdown(result) -> str

# polytopia_rl.losses
def collect_losses(policy_fn, opponent, seeds, env_kwargs, out_dir, limit) -> list[LossRecord]   # writes replay JSONs

# polytopia_rl.rating
def fit_ratings(wins: np.ndarray, games: np.ndarray, anchor: int, anchor_rating: float = 0.0) -> np.ndarray   # Bradley-Terry in Elo units
def win_prob(r_a, r_b) -> float                              # 1 / (1 + 10 ** ((r_b - r_a) / 400))

# polytopia_rl.tournament
RATING_SEED_BASE = 2_000_000_000
def play_pair(a, b, seeds, env_kwargs) -> PairResult         # both players may be nets or bots; each seed twice, seats swapped
def round_robin(players, seeds, env_kwargs) -> TournamentResult   # win matrix, games matrix, seat win rates, ratings

# polytopia_rl.selfplay
class SnapshotPool:                                          # add(model_state, step), sample(rng) -> Entry, entries, thin()
class SelfPlayVecEnv:                                        # same interface as SyncVecEnv (M10); see Rules
```

CLI: `python -m polytopia_rl.suite --checkpoint CKPT [--seeds 200] [--out DIR]`,
`python -m polytopia_rl.suite --compare A.json B.json`, `python -m polytopia_rl.losses --checkpoint CKPT
--opponent rush --count 5 --out DIR`, `python -m polytopia_rl.tournament --run RUNDIR [--seeds 100]`,
`python -m polytopia_rl.train --config python/configs/selfplay.json --init-checkpoint CKPT [k=v ...]`.

## Rules

### Hashes and checkpoints

- Every checkpoint stores `spec` (the dict from `Env.spec()` including both hashes) next to the config.
- `load_checkpoint` raises `SpecMismatch` when `obs_hash` differs (unless `strict_obs=False`, used only
  by an explicit `--allow-obs-mismatch` flag that prints a loud warning). A different `rules_hash` prints a
  warning (the network still runs, but results belong to a different game); `strict_rules=True` raises.
- Old M10 checkpoints (no `spec`) load with a warning and are stamped on first resave; the baseline suite
  run on the M10 checkpoint records the hashes of the current build.
- All CLIs that take a checkpoint report its hashes and training step in their first output line.

### Evaluation suite

`DEFAULT_SUITE` (all greedy unless stated), `seeds` per condition default 200, each seed twice with seats
swapped (M10 `evaluate`), seeds taken from `[SUITE_SEED_BASE, +seeds)` and the same seeds for every
condition and checkpoint:

| name | opponent | epsilon | full_observability |
| --- | --- | --- | --- |
| `random_e0`, `economy_e0`, `rush_e0` | random / economy / rush | 0 | false |
| `economy_e20`, `rush_e20` | economy / rush | 0.2 | false |
| `economy_fullobs`, `rush_fullobs` | economy / rush | 0 | true |
| `rush_e0_sampled` | rush | 0 | false, sampled (not greedy) |

`SuiteResult` stores per condition: games, W-L-D, win rate, Wilson 95% interval, win rate per seat,
mean length, mean score diff, share of losses by capital capture vs by score at the cap, plus
checkpoint hashes and step. `compare` reports the difference of win rates and flags it when a
two-proportion z test gives `|z| > 1.96`. Under `full_observability` the same policy is used with the
same encoding (explored all 1); this condition is a **diagnostic** of distribution shift, not a target.

### Loss review

`collect_losses` plays the seed pairs of one opponent, keeps games the policy lost (up to `limit`),
writes each as `OUT/{opponent}-{seed}-seat{s}.json` (`Env.replay_json()`, verifiable with
`sim-tools verify` and viewable with the M7 viewer) and returns for each: seed, seat, final turn, loss
type (`capital_captured` or `score`), final scores, and the turn the first city was lost. It prints a
summary table and the share of losses per seat.

### Ratings

`fit_ratings(wins, games, anchor)` fits Bradley-Terry strengths by maximum likelihood (iterative MM
updates until the largest change is below `1e-9`, at most 10,000 iterations) from `wins[i, j]` (games
won by `i` against `j`, draws counting 0.5 for each) and `games[i, j]`; ratings are `400 * log10(strength)`
shifted so that `ratings[anchor] == anchor_rating`. A player with no games, or a disconnected graph,
raises. Players that won or lost every game would diverge; apply a prior of half a win and half a loss
to every pair that has played (one virtual game split as 0.5-0.5), and document it. The anchor in all
reports is the `random` bot at 0.

### Self-play

- **Players in a pool:** snapshots (frozen copies of the learner's weights at `global_step`) and the three
  scripted bots. The learner is never in the pool until a snapshot is taken.
- **Snapshots:** the initial weights are snapshot 0 (that is the M10 checkpoint); a new snapshot is added
  every `snapshot_every` steps (default 1,000,000). Pool capacity `pool_max` (default 40): when full, thin by
  dropping every second entry except the first and the newest.
- **Opponent per episode:** at each episode start for each env: with probability `p_bot` (default 0.2) a bot
  drawn uniformly from `{economy, rush}`; otherwise a snapshot, which is the newest with probability
  `p_latest` (default 0.5) and a uniformly drawn entry of the whole pool otherwise. Draws use a dedicated
  RNG stream. Opponent identity is logged per finished episode.
- **Snapshot opponents act** with sampled (not greedy) actions, `opponent_greedy=false` by default, through
  the same `PolicyNet.act` and masks as the learner, from their own seat's observation.
- **Execution:** `SelfPlayVecEnv` keeps one `Env` per slot. Bot opponents use the Rust-side opponent
  (bot mode). Snapshot opponents use a two-seat env (`opponent="none"`). After the learner's action, all
  slots whose opponent has to move are advanced in rounds: in each round the slots are grouped by
  snapshot, each group does one batched forward pass, and every slot applies one action; the loop ends when
  every slot is either terminal or on the learner's move. The learner's seat is `seed % 2`; if it is
  seat 1 the opponent moves first at reset.
- **Reward consistency:** the learner's reward in a snapshot slot is computed from the scores before the
  learner's action and after the whole opponent turn, and the terminal result, with exactly the M9 formula
  (`terminal_weight`, `shaping_weight`, `gamma`, `potential_scale`, terminal potential 0). So a snapshot
  slot whose opponent is a scripted policy equal to a Rust bot returns identical rewards, dones and
  observations to the bot-mode env for the same seed.
- **Learning:** PPO updates use only the learner's own transitions, exactly the M10 update. The opponent's
  moves are not learned from.
- **Warm start:** `--init-checkpoint` loads model weights (obs hash enforced) into a fresh optimizer and a
  fresh LR schedule (config `lr` annealed over `total_steps`); the M10 optimizer state is not reused.
- **Monitoring** (all to `eval.jsonl`/`metrics.jsonl`): every `eval_every` (default 1,000,000) steps run the
  suite subset `{economy_e0, rush_e0, rush_e20}` with 50 seeds for the current weights, and flag
  `regression: true` if any win rate is below `regression_floor` (default 0.90); no automatic stop. Every
  `rating_every` (default 2,000,000) steps play the current weights (as a temporary player) against
  snapshot 0, the newest snapshot, three snapshots drawn evenly from the pool and the three bots, 50 seeds
  each (seats swapped), and log the fitted rating of the learner and of snapshot 0. `train.log` lists the
  pool and the opponent mix over the last 200 episodes.
- At the end the trainer writes `ckpt_final.pt`, the pool (`pool/snap_{step}.pt`) and a full round-robin
  of all pool entries plus the bots (`tournament.json`, `tournament.md`: win matrix, Elo, win rate by seat).

### Seed ranges

Training `< 1e9` (as before); suite `[1e9, 1.5e9)`; ratings and tournaments `>= 2e9`; no overlap anywhere.

## Clarifications to the spec (owner: confirm or overrule, then update the doc)

- **L1:** opponent mix defaults: 20% bots (economy/rush), 50% of the rest the newest snapshot, 50% uniform
  over the pool. Bots stay in the mix so the learner cannot forget how to beat simple strategies. - Confirmed
- **L2:** snapshot opponents sample actions (stochastic), for diversity; greedy is a config switch. - Confirmed
- **L3:** the learner learns from its own seat's transitions only (simplest; not "true" two-seat
  self-play with shared experience). - Confirmed
- **L4:** snapshot every 1M steps, pool of at most 40. - Confirmed
- **L5:** Elo is Bradley-Terry fitted from side-swapped matches, random bot anchored at 0, draws half a
  win, one virtual 0.5-0.5 game per pair as a prior. - Confirmed
- **L6:** the suite conditions above are the standing benchmark for every later run; adding a condition
  later is a spec change and the older results then lack that column. - Confirmed
- **L7:** `full_observability` is a diagnostic of how brittle the policy is to a different observation
  distribution; a weak result there is not a failure. - Confirmed
- **L8:** regression floor 0.90 against economy and rush (eps 0 and 0.2) during self-play is a flag, not a
  stop. The real regression check is the baseline comparison in the gate. - Confirmed
- **L9:** `obs_hash` includes everything the network depends on (layout, action space and enum orders);
  `rules_hash` covers the data tables only. A rules change that is not in `data/*.json` (code) is not
  detected; bump a `RULES_VERSION` constant (included in `rules_hash`) by hand for those.  - Confirmed
- **L10:** first-player advantage is measured, not corrected: reports always show win rate by seat.  - Confirmed
- **L11:** the M9 reward definition is the single source of truth; self-play recomputes it in Python and a
  test pins it against the Rust env.  - Confirmed

## Changes to earlier tests

`Env.spec()` gains two keys; M9 spec tests that compare the full key set update. Checkpoint tests from
M10 gain the `spec` field; old-format loading stays supported with a warning. No other changes.
`scripts/check.sh` is unchanged (the new tests run inside `rl_check.sh`).

## Tests (phase A guidance)

Golden values by hand or independent reference code in the test file.

1. **Hashes (Rust):** `fnv1a64` on the known vectors (`""` -> `cbf29ce484222325`, `"a"` ->
   `af63dc4c8601ec8c`, `"foobar"` -> `85944171f73967e8`); output is 16 lowercase hex digits; deterministic;
   `obs_hash` changes when any one ingredient changes (size, a channel name, an action-type order, an arg
   length, an enum order) via a test hook that hashes a modified canonical string; `rules_hash` changes with
   a data byte and is independent of anything else; the canonical string for `size = 3` matches a golden
   string in the test. `Env.spec()` exposes both and Python sees the same values.
2. **Checkpoint:** round trip keeps `spec`; mismatching `obs_hash` raises `SpecMismatch`; `strict_obs=False`
   loads with a warning; rules mismatch warns, `strict_rules` raises; old-format file loads with a warning;
   weights restore bit-exactly.
3. **Suite:** the default suite has exactly the conditions in the table; seed ranges and the "same seeds
   for every condition" rule; seeds twice with seats swapped; a scripted always-EndTurn policy loses nearly
   everything in every condition; a policy that plays the Rust bot's exact choices (through `step_legal`
   lookup) against that same bot at eps 0 wins about half (loose bounds) and exactly mirrors seats;
   `compare` flags a known large difference and not a known tiny one (z-test against a hand
   computation); `suite_markdown` contains each condition once; results embed hashes and step.
4. **Losses:** a scripted losing policy produces replay files that `sim-tools verify` accepts (subprocess);
   `loss type` and `first city lost` correct on two scripted games (a capital capture and a cap-ended
   loss); `limit` respected; seat shares add up.
5. **Ratings:** `win_prob` identities (equal ratings -> 0.5, +400 -> 10/11); `fit_ratings` recovers known
   ratings from expected-count matrices (noise-free) to within 1 Elo; anchor respected; draws as half wins;
   all-wins pairs finite thanks to the prior; no-games player raises; permuting players permutes ratings;
   a transitive three-player chain orders correctly.
6. **Tournament:** `play_pair` plays each seed twice with swapped seats; seeds are `>= 2e9`; win matrix
   and game counts are consistent (`wins[i,j] + wins[j,i] + draws = games`); a net vs itself is about
   50% with seat win rates reported; round robin of three scripted players gives the expected ordering.
7. **SelfPlayVecEnv:** with the opponent forced to a scripted policy equal to the Rust economy bot (via
   the two-seat path), observations, masks, rewards, dones and episode stats are identical to the bot-mode
   `SyncVecEnv` for the same seeds, for both seats and with shaping on and off; opponent actions are
   batched per snapshot (count forward calls on a mock); learner seat from seed parity and the opponent
   moving first when the learner is seat 1; auto-reset and final observations as in M10; an opponent
   that is mid-turn at the learner's truncation is handled (env finishes without a stuck loop); two
   identical runs are identical on CPU.
8. **Pool and matchmaking:** `SnapshotPool.add`, capacity thinning keeps first and newest; sampling
   frequencies match `p_bot`, `p_latest` and uniform (large n, tolerance); the opponent drawn per episode
   is logged; snapshots are frozen copies (training the learner later does not change them);
   `opponent_greedy` switch.
9. **Self-play smoke (tiny, CPU):** with the M10 `FakeEnv`-style tiny game and a tiny net, a few
   updates run end to end with a pool, snapshots are added on schedule, metrics have the documented keys,
   `regression` flag logic works on a mocked suite, the final tournament files are written, and warm start
   from a checkpoint with a wrong `obs_hash` fails.
10. **Train driver:** `--init-checkpoint` loads weights, a fresh optimizer and a fresh LR schedule (the
    first update's LR equals `lr`); resume from a self-play checkpoint continues pool, opponent RNG and seed
    index exactly (CPU, tiny config).

Mutants to aim for: hash ignoring one ingredient or the item order; `SpecMismatch` not raised; suite
conditions using different seeds, one condition missing, seats not swapped, epsilon or full-observability
flag not applied, greedy/sampled swapped; Elo scale wrong (natural log), anchor ignored, draws counted as
losses, prior missing; snapshot aliasing the live weights; pool thinning dropping the newest; `p_bot`/
`p_latest` inverted; opponent moving first missing for seat 1; learner reward computed from the opponent's
perspective, or only from the learner's own sub-step; opponent turn not completed before returning; seed
ranges overlapping; warm start reusing the old optimizer or LR; replay files missing the opponent's moves.

## Gate

**11a (implementation, Claude Code):** `./scripts/check.sh` passes (Rust, M9 and M10 tests, and the new
tests). Then, on the owner's machine: (1) `scripts/suite.sh` on the M10 `latest.pt` (200 seeds, all
conditions); (2) `losses` for 5 lost rush games of the M10 checkpoint, with the summary; (3) a 3M-step self-play smoke
run from that checkpoint; (4) the round-robin of its pool and the bots. Paste all tables into
`docs/RL_REPORT.md` and add a short "what we learned" section: how the agent loses to rush (seat, capture
vs score, how early), how it does against noisy opponents and under full observability, and the Elo
curve. The pipeline passes if every step runs and the numbers are internally consistent (Elo ordering
agrees with the head-to-head win rates, the `random` bot is the lowest rated). No performance
threshold; if self-play does not improve in 3M steps, report it, do not change rules or tests.

**11b (the real result, run by the owner):** `scripts/selfplay.sh python/configs/selfplay.json` for 20M
steps. Passes if (i) the final weights beat snapshot 0 (the M10 checkpoint) in at least 60% of 200
side-swapped games, and (ii) in the suite no condition falls more than 5 percentage points below the 11a
baseline unless the difference is not significant (`compare`). The owner reads the Elo curve, seat win
rates and the loss review, then decides on the next step (combat preview, capacity experiment, turn cap,
game v2).

## Notes for the implementer

- Keep `evaluate` (M10) as the one place that plays an evaluation game; the suite, tournament and loss
  collector call it or share its internals.
- Batch the opponents' forward passes; Python-side per-slot GPU calls will be the bottleneck otherwise.
  Report sps for rollout (learner plus opponents) and update separately, as in M10.
- Do not read anything from `Env.state`; policies get `observation()` and masks only.
- `RULES_VERSION` starts at 1 in `sim-env`; mention in the report how to bump it.
- If a test contradicts this brief or the spec, stop and add a QUESTIONS.md entry.
