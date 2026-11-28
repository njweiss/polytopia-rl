# Milestone 22: v1 -> v2 policy transfer (prefix warm-start and policy distillation)

Pure-Python RL milestone providing both v1 -> v2 transfer paths on top of the completed v2 RL stack
(M20) and `arch="v2"` network (M21), plus ready-to-run v2 configs and a 4-arm transfer experiment spec:

1. **Prefix warm-start (`warm_start_v1_to_v2`):** transfers weights directly from a trained `v1`
   checkpoint into a `v2` `arch="v1"` `PolicyNet` (same `width`, `blocks`, `size`), zero-initializing
   the new v2 input columns so step-0 behavior on v1 features is preserved exactly while gradients flow
   into all new v2 inputs/outputs on update 1.
2. **Policy distillation (`distill_kl` + frozen teacher in `ppo.train_update` / `train.run`):**
   architecture-agnostic KL distillation from a frozen `v1` teacher checkpoint into **any** `v2` student
   (including `arch="v2"` of any `width`/`blocks`), projecting v2 observations and factored masks onto
   the shared v1 prefix, with linear annealing of `distill_coef` to `0.0` over `distill_steps`.

**The hard constraint:** zero Rust, `obs_hash`, `rules_hash`, or existing-test changes. When
`warm_start_checkpoint is None` and `teacher_checkpoint is None` (defaults), `train.run` and
`ppo.train_update` behave 100% identically to M21.

## Goal

1. Implement `python/polytopia_rl/transfer.py` with:
   - `warm_start_v1_to_v2(dst_net, v1_state, v1_spec, v2_spec)` (exact prefix copy + zero-init new input
     columns, validating `dst_net.arch == "v1"`, same `size`/`width`/`blocks`, and `v1 -> v2` spec).
   - `TeacherDistiller` (or `compute_teacher_logits` + `distill_kl`): computes frozen teacher logits
     **once per rollout** (not per PPO epoch), projects v2 obs & masks to the v1 prefix (or identity for
     same-spec `v2 -> v2` distillation), and computes numerically stable temperature-scaled masked KL
     across all three factored stages (`actor`, `kind`, `arg`), cleanly zeroing out rows with no
     v1-legal options at Stage 2 or Stage 3.
2. Wire transfer into `PPOConfig`, `ppo.train_update`, and `train.run` (including exact resume of the
   annealing schedule from `latest.pt`, and logging `distill_kl` and `distill_coef` in `metrics.jsonl`).
3. Add shipped v2 configs (`python/configs/v2_curriculum.json`, `python/configs/v2_selfplay.json`) and
   an M12 experiment spec (`python/experiments/transfer22.json`) comparing `scratch_v1`, `scratch_v2`,
   `warmstart_v1`, and `distill_v2` out of the box with `python -m polytopia_rl.experiment`.

## Decisions already made (owner)

- Support both warm-start and distillation as comparable arms in the M12 experiment harness
  (`docs/STATUS.md` "Waiting on the owner").
- M20 deliberately kept the exact 37-channel / 18-global / 10-kind v1 prefix in `v2` observations and
  actions specifically to enable M22 transfer.

## In scope

1. `python/polytopia_rl/transfer.py` (new module):
   - `warm_start_v1_to_v2(dst_net, v1_state, v1_spec, v2_spec) -> None`
   - `load_warm_start(dst_net, ckpt_path, v2_spec) -> Checkpoint`
   - `masked_kl(teacher_logits, student_logits, mask, temp=1.0) -> Tensor`
   - `project_v2_to_v1_obs(board, globals, v1_spec) -> tuple[Tensor, Tensor]`
   - `compute_teacher_logits(teacher, teacher_spec, student_spec, board, globals, actors, kinds) -> tuple[Tensor, Tensor, Tensor]`
   - `distill_kl(teacher_logits, student_eval_logits, actors, kinds, actor_mask, kind_mask, arg_mask, teacher_spec, student_spec, temp=1.0) -> tuple[Tensor, dict[str, Tensor]]`
2. `python/polytopia_rl/config.py`:
   - Add `warm_start_checkpoint: str | None = None`, `teacher_checkpoint: str | None = None`,
     `distill_coef: float = 0.0`, `distill_steps: int | None = None`, `distill_temp: float = 1.0`.
   - Validate in `PPOConfig.from_json`: `distill_coef >= 0.0`; `distill_coef > 0.0` requires
     `teacher_checkpoint is not None`; `distill_steps is None or distill_steps >= 0`; `distill_temp > 0.0`.
3. `python/polytopia_rl/model.py`:
   - Add optional helper (or extend `EvalOut` / add `evaluate_with_logits` or include logits on `EvalOut` —
     T3) so `train_update` reuses the student's `actor_logits`, `kind_logits`, `arg_logits` from the
     same forward pass without running `PolicyNet` twice per minibatch!
4. `python/polytopia_rl/ppo.py`:
   - Extend `train_update(net, optimizer, buffer, cfg, generator, *, teacher=None, teacher_spec=None,
     student_spec=None, distill_coef=0.0)`: when `teacher is not None and distill_coef > 0.0`, compute
     frozen teacher logits **once** before the `update_epochs` loop, add `distill_coef * kl.mean()` to
     each minibatch's total loss, and include `distill_kl` and `distill_coef` in the returned metrics
     dict (default kwargs keep all existing callers 100% unchanged!).
5. `python/polytopia_rl/train.py`:
   - Support `cfg.warm_start_checkpoint` on fresh start (`resume is None`, rejecting simultaneous
     `init_checkpoint` + `cfg.warm_start_checkpoint` with `ValueError`).
   - Load frozen `teacher` once when `cfg.teacher_checkpoint is not None` (using `checkpoint.load_net`
     against the teacher's own ruleset/size, set `.eval().requires_grad_(False)`), compute annealed
     `current_distill_coef` deterministically from `update`, and pass to `train_update`.
6. `python/configs/v2_curriculum.json`, `python/configs/v2_selfplay.json`, and
   `python/experiments/transfer22.json`.

## Out of scope

Running multi-million-step GPU training experiments (done by the owner on their RTX 4060 machine);
changing Rust, `sim-env`, `sim-py`, `obs_hash`, `rules_hash`, or `experiment.py`; value distillation
(`distill_vf_coef`, T2).

## Public API (fixed so tests can import it)

```python
# python/polytopia_rl/transfer.py
def warm_start_v1_to_v2(
    dst_net: PolicyNet,
    v1_state: dict[str, Tensor],
    v1_spec: dict,
    v2_spec: dict,
) -> None: ...

def load_warm_start(dst_net: PolicyNet, ckpt_path: str, v2_spec: dict) -> Checkpoint: ...

def masked_kl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    mask: Tensor,
    temp: float = 1.0,
) -> Tensor: ...

def distill_schedule(
    update: int,
    base_coef: float,
    total_updates: int,
    distill_updates: int | None,
) -> float: ...

def compute_teacher_logits(
    teacher: PolicyNet,
    teacher_spec: dict,
    student_spec: dict,
    board: Tensor,
    globals: Tensor,
    actors: Tensor,
    kinds: Tensor,
) -> tuple[Tensor, Tensor, Tensor]: ...

def distill_kl(
    teacher_logits: tuple[Tensor, Tensor, Tensor],
    student_logits: tuple[Tensor, Tensor, Tensor],
    kinds: Tensor,
    actor_mask: Tensor,
    kind_mask: Tensor,
    arg_mask: Tensor,
    teacher_spec: dict,
    student_spec: dict,
    temp: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]: ...
```

## Rules

### Prefix Warm-Start (`warm_start_v1_to_v2`, T1)

- Preconditions (raise `ValueError` on any violation):
  - `dst_net.arch == "v1"` (prefix copying requires identical `arch="v1"` trunk/head modules; use
    distillation for `arch="v2"`).
  - `v1_spec["height"] == v2_spec["height"]` and `v1_spec["width"] == v2_spec["width"]`.
  - `v2_spec["channel_names"][:c1] == v1_spec["channel_names"]` and
    `v2_spec["global_names"][:g1] == v1_spec["global_names"]` and
    `v2_spec["action_types"][:k1] == v1_spec["action_types"]`.
  - `v1_state["stem.weight"].shape[0] == dst_net.stem.weight.shape[0]` (same `width`) and same number
    of `blocks`.
- Weight copy rules (inside `torch.no_grad()`):
  - **`stem.weight`**: copy `[:, :c1, :, :]` from `v1_state`; **zero** `[:, c1:, :, :]` so non-zero v2
    channels (`lighthouse`, `preview.*`, `road`) inject zero noise at step 0. Copy `stem.bias` verbatim.
  - **`global_mlp.0.0.weight`**: copy `[:, :g1]` from `v1_state`; **zero** `[:, g1:]`. Copy all other
    `global_mlp.*` tensors verbatim.
  - **`blocks.*`, `actor_head.*`, `value_head.*`, `kind_head.context.*`, `kind_head.mlp.0.*`,
    `arg_head.context.*`, `arg_head.c.*`, `arg_head.tile_keys.*`**: exact copy.
  - **`kind_head.mlp.2.{weight,bias}`**: copy `[:k1]` from `v1_state`; leave `[k1:]` at `dst_net`'s
    orthogonal init (`gain=0.01`, bias `0.0`).
  - **`arg_head.kind_embed.weight` & `arg_head.tile_bias`**: copy `[:k1]` from `v1_state`; leave `[k1:]`
    at `dst_net`'s fresh init.
  - **`arg_head.enum.{i}.{weight,bias}`**: match each v1 enum kind by name (`train`, `choose_reward`,
    `site`, `research`): copy `[:v1_arg_len]` into the corresponding `dst_net` enum head; leave new
    output rows (`[v1_arg_len:]`) and brand-new enum heads (`build_building`, `build_monument`) at
    `dst_net`'s fresh init.
- Result: on any state where only v1-prefix channels/globals are active (even with arbitrary values on
  new v2 channels!), `fmap`, `g`, `value`, `actor_logits`, `kind_logits[:, :10]`, and v1-prefix
  `arg_logits` match the original v1 checkpoint to `atol=1e-6` on step 0!

### Policy Distillation (`masked_kl`, `compute_teacher_logits`, `distill_kl`, `distill_schedule`, T2–T4)

- **Supported pairs (`teacher_spec` -> `student_spec`):**
  - Same map size (`height`, `width` equal); `teacher_spec` is either identical to `student_spec`
    (`v2 -> v2` or `v1 -> v1`) OR a strict prefix of `student_spec` (`v1 -> v2`). Otherwise raise
    `ValueError`.
- **`masked_kl(teacher_logits, student_logits, mask, temp=1.0)`:**
  - `t = masked_logits(teacher_logits.detach() / temp, mask, check=False)`
  - `s = masked_logits(student_logits / temp, mask, check=False)`
  - `p_t = F.softmax(t, dim=-1)`
  - `log_p_t = F.log_softmax(t, dim=-1)`
  - `log_p_s = F.log_softmax(s, dim=-1)`
  - `kl = torch.where(mask, p_t * (log_p_t - log_p_s), 0.0).sum(dim=-1).clamp_min(0.0) * (temp * temp)`
  - Guaranteed finite (zero `NaN`s even when invalid entries are masked out), `0.0` when distributions
    match or when `mask` has 1 legal option, and gradients flow only through `student_logits`!
- **Three-stage factored distillation (`distill_kl`):**
  - **Stage 1 (`actor`):** `kl_actor = masked_kl(t_actor, s_actor, actor_mask, temp)` (all rows valid).
  - **Stage 2 (`kind`):** `k1 = len(teacher_spec["action_types"])`; `km_t = kind_mask[:, :k1]`;
    `has_k = km_t.any(dim=1)`; `safe_km = km_t | ~has_k.unsqueeze(1)`;
    `kl_kind = torch.where(has_k, masked_kl(t_kind, s_kind[:, :k1], safe_km, temp), 0.0)`.
    (Zero for rows whose chosen actor had only v2-only kinds, e.g., neutral `BuildRoad` or `Disband`!)
  - **Stage 3 (`arg`):** let `v1_arg_lens` be the teacher's `arg_lens` padded to length
    `len(student_spec["action_types"])` with `0` for v2-only kinds. For each row `b`, `len_b =
    v1_arg_lens[kinds[b]]` (`0` for v2-only kinds or no-arg kinds); `am_t = arg_mask & (arange(N) <
    len_b[:, None])`; `has_a = am_t.any(dim=1)`; `safe_am = am_t | ~has_a.unsqueeze(1)`;
    `kl_arg = torch.where(has_a, masked_kl(t_arg, s_arg, safe_am, temp), 0.0)`.
    (Note: when calling `teacher.arg_logits(t_feats, actors, safe_kinds)`, clamp `safe_kinds =
    kinds.clamp(max=k1 - 1)` so `kinds >= k1` never out-of-bounds-index the teacher's `kind_embed`,
    since those rows are masked out to `0.0` by `has_a` anyway!)
  - Return `(kl_actor + kl_kind + kl_arg, {"actor": kl_actor, "kind": kl_kind, "arg": kl_arg})`.
- **Linear annealing (`distill_schedule`):**
  - If `base_coef == 0.0`: return `0.0`.
  - If `distill_updates == 0`: return `base_coef` (constant).
  - Otherwise `horizon = total_updates if distill_updates is None else distill_updates`:
    return `base_coef * max(0.0, 1.0 - (update - 1) / horizon)` (full `base_coef` on update 1, linearly
    reaches `0.0` at `update == horizon + 1`, exact same convention as `lr` annealing in `train.py:223`!).
- **Single teacher pass per rollout:** in `train_update`, when `teacher is not None and distill_coef > 0.0`,
  compute `(t_actor, t_kind, t_arg)` once over the flattened rollout `(T * E, ...)` under
  `torch.no_grad()`, then slice `[idx]` per minibatch across all `update_epochs`!

### Reusing Student Logits in `train_update` (T3)

- Add optional `return_logits: bool = False` (or `logits: tuple[Tensor, Tensor, Tensor] | None = None`
  field on `EvalOut`, populated when `return_logits=True`) on `PolicyNet.evaluate(...)` so existing
  callers see identical `EvalOut` behavior while `train_update` with distillation reuses the exact
  student `(actor_logits, kind_logits, arg_logits)` from the same forward pass (zero duplicate trunk
  passes!).

### Shipped Configs & Experiment Spec (T5)

- `python/configs/v2_curriculum.json`: `ruleset: "v2"`, `arch: "v2"`, 3-stage curriculum matching
  `curriculum.json`.
- `python/configs/v2_selfplay.json`: `ruleset: "v2"`, `arch: "v2"`, matching `selfplay.json` settings
  (`selfplay: true`, `p_bot: 0.4`, `128x32`, `lr: 1e-4`, `opponent_forward: "per_snapshot"`).
- `python/experiments/transfer22.json`: M12 `Spec` (`base_config: "python/configs/v2_curriculum.json"`,
  `seeds: [1, 2, 3]`, `total_steps: 10000000`, `init_checkpoint: null`, `reference: null`, `primary:
  ["rush_e0", "economy_e0"]`) with 4 arms:
  - `scratch_v1`: `{"arch": "v1"}`
  - `scratch_v2`: `{"arch": "v2"}`
  - `warmstart_v1`: `{"arch": "v1", "warm_start_checkpoint": "runs/curriculum_seed1/latest.pt"}`
  - `distill_v2`: `{"arch": "v2", "teacher_checkpoint": "runs/curriculum_seed1/latest.pt", "distill_coef": 0.1, "distill_steps": 3000000}`

## Clarifications (owner: confirm or overrule)

- **T1 (Warm-start input zeroing):** In `warm_start_v1_to_v2`, zero-initialize the new v2 input columns
  (`stem.weight[:, 37:] = 0` and `global_mlp.0.0.weight[:, 18:] = 0`) while keeping fresh small-gain
  (`0.01`) orthogonal init on new output rows (`kind_head`, `arg_head.enum`), so step-0 v1-prefix
  outputs are preserved bit-for-bit even on v2 boards with lighthouses/preview channels. — Confirmed (owner, 2026-11-28)
- **T2 (Policy KL distillation only, no value distillation):** Distill factored policy logits
  (`actor` + `kind` + `arg`) with temperature-scaled `T^2 * KL(p_teacher || p_student)` annealed linearly
  to `0.0` over `distill_steps`; do not distill `value_head` since v2 score/reward distributions differ
  from v1. — Confirmed (owner, 2026-11-28)
- **T3 (Reusing student logits in `PolicyNet.evaluate`):** Add `return_logits: bool = False` on
  `PolicyNet.evaluate` (attaching `logits=(actor_logits, kind_logits, arg_logits)` on `EvalOut` when
  `True`, `None` by default) so distillation adds zero extra student forward passes and computes teacher
  logits only once per rollout. — Confirmed (owner, 2026-11-28)
- **T4 (`PPOConfig` fields for per-arm transfer in `experiment.py`):** Add `warm_start_checkpoint`,
  `teacher_checkpoint`, `distill_coef`, `distill_steps`, and `distill_temp` directly to `PPOConfig` so
  `python/experiments/transfer22.json` works with `experiment.py` out of the box (zero changes to
  `experiment.py`). — Confirmed (owner, 2026-11-28)
- **T5 (Shipped v2 configs & transfer experiment spec):** Include `python/configs/v2_curriculum.json`,
  `python/configs/v2_selfplay.json`, and `python/experiments/transfer22.json` (4 arms: `scratch_v1`,
  `scratch_v2`, `warmstart_v1`, `distill_v2`) and verify they parse cleanly via `load_config` / `load_spec`. — Confirmed (owner, 2026-11-28)
- **T6 (Pure-Python scope):** Zero Rust changes, zero `obs_hash`/`rules_hash` changes, zero edits to
  existing tests; all M22 tests live in `python/tests/rl/test_transfer.py`. — Confirmed (owner, 2026-11-28)

## Changes to earlier tests

None. Every existing Rust, Python, and RL test passes unchanged.

## Tests (phase A guidance)

All in `python/tests/rl/test_transfer.py`:

1. **Config validation:** defaults (`warm_start_checkpoint=None`, `teacher_checkpoint=None`,
   `distill_coef=0.0`, `distill_steps=None`, `distill_temp=1.0`), JSON round-trip & overrides,
   `ValueError` on `distill_coef < 0`, `distill_coef > 0` without `teacher_checkpoint`,
   `distill_steps < 0`, `distill_temp <= 0`; shipped `v2_curriculum.json`, `v2_selfplay.json`, and
   `transfer22.json` load and validate cleanly.
2. **Prefix warm-start (`warm_start_v1_to_v2` & `load_warm_start`):**
   - Exact step-0 functional preservation: save a trained/randomized `v1` `PolicyNet` (`size=6, width=16,
     blocks=2`), warm-start a `v2` `arch="v1"` `PolicyNet` from it; on real v2 observations (with
     non-zero v2 channels like `lighthouse` and `preview.*`!), `fmap`, `g`, `value`, `actor_logits`,
     `kind_logits[:, :10]`, and v1-prefix `arg_logits` match the v1 network on `(board[:, :37],
     globals[:, :18])` to `atol=1e-5`!
   - Non-zero gradients flow immediately into the zero-initialized new input columns (`stem.weight[:, 37:]`,
     `global_mlp.0.0.weight[:, 18:]`) and new output rows on the very first backward pass.
   - Validation errors (`ValueError`): `dst_net.arch == "v2"`, mismatched `width`, mismatched `blocks`,
     mismatched `size`, or non-v1 source checkpoint.
3. **Masked KL & distillation schedule (`masked_kl`, `distill_schedule`):**
   - `masked_kl` equals `0.0` when teacher == student (even with invalid masked logits = `1e6`!),
     matches an independent numpy reference across temperatures `T = 1.0, 2.0, 0.5`, is strictly `>= 0`
     and finite (no `NaN`s), gives `0.0` when only 1 option is legal, and puts gradients ONLY on
     `student_logits` (never `teacher_logits`).
   - `distill_schedule`: update 1 = `base_coef`, midpoint = `0.5 * base_coef`, update `>= horizon + 1` =
     `0.0`, `distill_updates=0` stays constant, `base_coef=0.0` stays `0.0`.
4. **Factored distillation (`compute_teacher_logits`, `distill_kl`):**
   - `v1 -> v2` distillation into an **`arch="v2"`** student with a **different** `width` & `blocks`
     (e.g., v1 teacher `arch="v1", width=8, blocks=1` -> v2 student `arch="v2", width=16, blocks=2`):
     - Handles mixed batches containing: (a) normal v1 actions (`move`, `attack`, `train`, `research`,
       `end_turn`), (b) v2-only chosen kinds (`build_road`, `build_building`, `build_monument`,
       `disband`, `destroy`) — verifies Stage-3 `kl_arg == 0.0` and no index crash on `kinds >= 10`!,
       (c) rows where `kind_mask[:, :10]` is all-False — verifies Stage-2 `kl_kind == 0.0`!, and (d)
       `research` rows where only v2 techs (`9..16`) are legal — verifies Stage-3 `kl_arg == 0.0`!
     - Gradient descent on `distill_kl` alone strictly decreases KL from teacher to student over a few
       steps while leaving all teacher parameters unchanged (`requires_grad == False` / bit-identical).
   - Same-spec `v2 -> v2` distillation works too (no prefix truncation); size mismatch raises `ValueError`.
5. **Single teacher pass & `train_update` / `train.run` end-to-end:**
   - Spy on `teacher.trunk` inside `train_update(..., update_epochs=4, num_minibatches=2)`: assert
     `teacher.trunk` is called **once** per `train_update` (not `4` or `8` times!).
   - End-to-end CPU `train.run`:
     - `warmstart_v1`: `ruleset="v2", arch="v1", warm_start_checkpoint=<v1_ckpt>` runs cleanly, saves
       v2 checkpoint, and refuses simultaneous `init_checkpoint` + `warm_start_checkpoint`.
     - `distill_v2`: `ruleset="v2", arch="v2", teacher_checkpoint=<v1_ckpt>, distill_coef=0.1,
       distill_steps=16` runs cleanly, logs finite `distill_kl` & decaying `distill_coef` in
       `metrics.jsonl` (decaying to `0.0` after `distill_steps`), teacher weights untouched, and
       resuming from mid-run `latest.pt` produces bit-identical weights & metrics to an uninterrupted run!
     - Verify `rules_hash() == "a8b52b91eae34503"` and `obs_hash(11) == "d5f9b1033cfd55ca"` unchanged.

Mutants to aim for: forgetting to zero `stem.weight[:, 37:]` or `global_mlp.0.0.weight[:, 18:]` (breaks
step-0 preservation on v2 boards); copying `arg_head.enum` by index instead of by kind name; allowing
`warm_start_v1_to_v2` on `arch="v2"`; `NaN` in `masked_kl` from `0 * (-inf)`; forgetting `temp ** 2`
scaling; crashing on `kinds >= 10` when calling `teacher.arg_logits`; computing non-zero Stage-2/3 KL on
v2-only rows; running teacher forward once per minibatch/epoch instead of once per rollout; `train.run`
mutating teacher weights or failing to resume the distillation schedule identically.

## Gate

`pixi run check` exits 0 (fmt, clippy `-D warnings`, all Rust, Python, and RL tests green). `rules_hash()`
== `"a8b52b91eae34503"` and `obs_hash(11)` == `"d5f9b1033cfd55ca"` unchanged.

## Notes for the implementer

- Keep `train_update`'s existing positional signature untouched (all distillation args keyword-only with
  defaults) so zero existing tests or scripts break.
- Load `teacher` with `checkpoint.load_net(cfg.teacher_checkpoint, device=device, strict_obs=True)` so
  its own `obs_hash` and `arch` are verified automatically.
- If a test contradicts this brief, stop and add a `docs/QUESTIONS.md` entry (next number **Q59**).
