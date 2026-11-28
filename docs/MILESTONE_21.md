# Milestone 21: `arch="v2"` PolicyNet architecture (global-conditioned trunk, normalized ResBlocks, relative-position ArgHead)

Pure-Python RL architecture milestone preparing the policy network for v2 training and M22 (v1 -> v2
policy transfer). Milestone 14b showed that the unnormalized post-ReLU `ResBlock` degrades at depth 8
(`w64b8` = -118 Elo vs `w64b4` = +147 Elo), and v2 exposes two additional representational limits of
the v1 network: `fmap` (and therefore `ActorHead`) never sees `globals` (stars, techs, turn, monuments),
and `ArgHead` shares one `tile_keys` projection across `move` and `attack` with no relative `(dr, dc)`
awareness for 3-tile Catapult shots or 6-tile Knight road runs.

This milestone adds `arch="v2"` to `PolicyNet` and `PPOConfig` behind a backward-compatible default
`arch="v1"`.

**The hard constraint:** `arch="v1"` (the default) is 100% bit-for-bit identical to today — same
submodules, parameter names, initialization, forward outputs, and checkpoints. Zero Rust code, zero
`obs_hash`/`rules_hash` values, and zero existing tests change.

## Goal

1. Add `arch: str = "v1"` (`"v1"` | `"v2"`) to `PPOConfig` and `PolicyNet`, wired through `train.py` and
   `checkpoint.py` (`load_net` automatically restores the checkpoint's architecture from `cfg.arch`).
2. Implement `arch="v2"` in `python/polytopia_rl/model.py`:
   - **CoordConv** (2 static `(row, col)` channels in `[-1, 1]` inside `trunk`, no `obs_hash` change) so
     the CNN directly knows corners (lighthouses) and map edges.
   - **Pre-activation `GroupNorm` + scaled `ResBlockV2`** with **KataGo-style pooled-context + globals
     FiLM** (`[global_mlp(globals), mean_pool(x)]` -> `(1 + delta_gamma, beta)`, zero-initialized to
     identity; residual branch scaled by `1 / sqrt(2 * blocks)`), fixing deep-trunk optimization and
     giving every tile (and `ActorHead`) full global and full-board context in one hop.
   - **`ArgHeadV2`** with separate `1×1` `tile_keys` per tile kind (`move` vs `attack`), QK `LayerNorm`,
     and learned `(2H-1) × (2W-1)` 2D relative-position bias per tile kind (zero-initialized).
3. Verify full compatibility with `StackedSnapshots` (`torch.func.vmap`), `SyncVecEnv`,
   `SelfPlayVecEnv`, `collect_rollout_interleaved`, PPO updates, evaluation, and checkpointing on both
   `ruleset="v1"` and `ruleset="v2"`.

## Decisions already made (owner)

- Shift v1 -> v2 policy transfer (warm-start & distillation) to **Milestone 22** so M21 lands `arch="v2"`
  first (distillation in M22 can transfer directly from the v1 teacher into `arch="v2"`).
- Include all 5 recommended items (CoordConv, pooled-context + globals FiLM, pre-norm `GroupNorm` +
  scaled `ResBlock`, separate `move`/`attack` keys + 2D relative-position bias + QK-Norm in `ArgHead`,
  and `PPOConfig.arch`).

## In scope

1. `python/polytopia_rl/config.py`:
   - Add `arch: str = "v1"` on `PPOConfig`; validate `arch in ("v1", "v2")` in `PPOConfig.from_json`
     (raising `ValueError` otherwise).
2. `python/polytopia_rl/model.py`:
   - Extend `PolicyNet.__init__(self, spec: dict, width: int = 64, blocks: int = 4, arch: str = "v1")`
     (validating `arch in ("v1", "v2")`, raising `ValueError` otherwise).
   - Keep `arch == "v1"` 100% unchanged.
   - Implement `ResBlockV2` and `ArgHeadV2` for `arch == "v2"`, preserving `Features(fmap, g)`, the 4
     head submodule names (`actor_head`, `kind_head`, `arg_head`, `value_head`), and all `PolicyNet`
     public methods (`trunk`, `actor_logits`, `kind_logits`, `arg_logits`, `value`, `act`, `evaluate`).
3. `python/polytopia_rl/train.py` and `python/polytopia_rl/checkpoint.py`:
   - Pass `arch=cfg.arch` when constructing `PolicyNet` in `train.run` and `checkpoint.load_net`.

## Out of scope

2D self-attention across tiles; separate policy/value trunks; recurrent/LSTM memory; Rust/`sim-env`/
`sim-py` changes; `obs_hash`/`rules_hash` changes; long GPU training runs and v1 -> v2 transfer
experiments (M22).

## Public API (fixed so tests can import it)

```python
# python/polytopia_rl/config.py
@dataclass
class PPOConfig:
    ...
    arch: str = "v1"  # "v1" | "v2"

# python/polytopia_rl/model.py
class PolicyNet(nn.Module):
    def __init__(self, spec: dict, width: int = 64, blocks: int = 4, arch: str = "v1"): ...
    # Same public methods and head submodule names (`actor_head`, `kind_head`, `arg_head`, `value_head`)
```

## Architecture specification (`arch="v2"`)

### Helper: `GroupNorm` group count

- `groups = 8 if width % 8 == 0 else 1` (`nn.GroupNorm(groups, width)`). Zero running stats, batch-size
  independent, compatible with both small test widths (`width=8`) and production widths (`32, 64, 128`),
  and 100% `torch.func.vmap`-safe.

### Trunk (`arch="v2"`)

1. **CoordConv buffer (non-persistent):** `coords` of shape `(1, 2, H, W)` (`float32`):
   - Channel 0: row coordinate `torch.linspace(-1.0, 1.0, H)` broadcast to `(H, W)`.
   - Channel 1: col coordinate `torch.linspace(-1.0, 1.0, W)` broadcast to `(H, W)`.
   - In `trunk(board, globals)`: `x = torch.cat([board, self.coords.expand(B, -1, -1, -1)], dim=1)` ->
     `(B, C + 2, H, W)`.
2. **Stem:** `self.stem = _init(nn.Conv2d(channels + 2, width, 3, padding=1), math.sqrt(2))` (linear
   projection into the pre-activation residual stream; no activation here).
3. **Global MLP:** `self.global_mlp = nn.Sequential(_mlp([num_globals, width, width], math.sqrt(2)),
   nn.SiLU())` -> `g_emb` of shape `(B, width)`.
4. **`blocks` × `ResBlockV2(width, blocks)`** (in `self.blocks = nn.ModuleList(...)`):
   - `cond = torch.cat([g_emb, x.mean(dim=(2, 3))], dim=1)` (`(B, 2 * width)`).
   - `h = self.conv1(F.silu(self.norm1(x)))`
   - `delta_gamma, beta = self.film(cond).chunk(2, dim=1)` (`self.film = nn.Linear(2 * width, 2 * width)`
     initialized with **zero weight and zero bias** so step 0 starts at exact identity `1 * norm2(h) + 0`).
   - `h = self.norm2(h) * (1.0 + delta_gamma[:, :, None, None]) + beta[:, :, None, None]`
   - `h = self.conv2(F.silu(h))`
   - `return x + self.res_scale * h`, where `self.res_scale = 1.0 / math.sqrt(2 * blocks)`.
   - `conv1` and `conv2` use orthogonal init (`gain = math.sqrt(2)`).
5. **Final trunk norm + activation:** `fmap = F.silu(self.trunk_norm(x))` where
   `self.trunk_norm = nn.GroupNorm(groups, width)`.
6. **Context vector `g`:** `torch.cat([g_emb, fmap.mean(dim=(2, 3)), fmap.amax(dim=(2, 3))], dim=1)`
   (`(B, 3 * width)`, identical shape to `v1`).

### Heads (`arch="v2"`)

- `actor_head`: `ActorHead(width, ctx_dim)` (unchanged; now receives FiLM-conditioned `fmap`).
- `kind_head`: `KindHead(width, ctx_dim, num_tiles, len(kinds))` (unchanged).
- `value_head`: `_mlp([ctx_dim, width, 1], 1.0)` (unchanged).
- `arg_head`: `ArgHeadV2(width, ctx_dim, height, width_board, kinds, arg_lens)`:
  - Same `ActorContext`, `kind_embed`, `c` MLP (`(B, width)`), and `enum` heads (`ModuleList` of linear
    layers for all enum kinds).
  - **Separate `tile_keys` per tile kind (`move`, `attack`):**
    `self.tile_keys = nn.ModuleList(_init(nn.Conv2d(width, width, 1), math.sqrt(2)) for _ in self.tile_kinds)`.
  - **QK `LayerNorm`:** `self.q_norm = nn.LayerNorm(width)` on `c` (`q = self.q_norm(c)`) and
    `self.k_norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.tile_kinds)` applied per tile on
    `keys` (`(B, N, width)`) before `torch.einsum("bnw,bw->bn", k, q) / math.sqrt(self.width)`.
  - **2D relative-position bias per tile kind:**
    - `self.rel_bias = nn.Parameter(torch.zeros(len(self.tile_kinds), (2 * H - 1) * (2 * W - 1)))`
      (zero-initialized).
    - Non-persistent integer buffer `rel_index` of shape `(N, N)`:
      `rel_index[a, t] = (r_t - r_a + H - 1) * (2 * W - 1) + (c_t - c_a + W - 1)`.
    - For each row, `actor_tile = torch.where(actors < 3 * N, actors % N, torch.zeros_like(actors))`,
      `idx = self.rel_index[actor_tile]` (`(B, N)`), and for tile kind `ti`:
      `bias = self.rel_bias[ti][idx]` (`(B, N)`), added to `tile_logits + self.tile_bias[ti]`.

## Clarifications (owner: confirm or overrule)

- **A1 (`arch` flag & default):** Add `arch: str = "v1"` (`"v1"` | `"v2"`) to `PPOConfig` and
  `PolicyNet`, wired through `train.py` and `checkpoint.py`, leaving `arch="v1"` 100% bit-for-bit
  identical. — Confirmed (owner, 2026-11-27)
- **A2 (Trunk normalization, FiLM, activation, scaling):** CoordConv `(row, col)` buffer in `[-1, 1]`;
  pre-activation `ResBlockV2` with `GroupNorm(8 if width % 8 == 0 else 1, width)`, `SiLU` activation,
  zero-initialized FiLM conditioned on `[global_mlp(globals), mean_pool(x)]`, residual scale
  `1 / sqrt(2 * blocks)`, and final `GroupNorm` + `SiLU`. — Confirmed (owner, 2026-11-27)
- **A3 (`ArgHeadV2`):** Separate `1×1` `tile_keys` conv + QK `LayerNorm` + zero-initialized 2D
  relative-position bias `(2H-1) × (2W-1)` per tile kind (`move`, `attack`). — Confirmed (owner, 2026-11-27)
- **A4 (`vmap` / `StackedSnapshots` & head submodule names):** Preserve `actor_head`, `kind_head`,
  `arg_head`, `value_head` submodule names and `Features(fmap, g)` contract; guarantee exact parity
  under `StackedSnapshots` (`torch.func.vmap`) on both `v1` and `v2` rulesets. — Confirmed (owner, 2026-11-27)
- **A5 (No edits to earlier tests or Rust):** Zero Rust changes, zero `obs_hash`/`rules_hash` changes,
  zero edits to existing tests; all M21 tests live in `python/tests/rl/test_arch_v2.py`. — Confirmed (owner, 2026-11-27)

## Changes to earlier tests

None. Every existing Rust, Python, and RL test passes unchanged.

## Tests (phase A guidance)

All in `python/tests/rl/test_arch_v2.py`:

1. **Config & constructor validation:** `PPOConfig.from_json("{}").arch == "v1"`, JSON round-trip and
   override with `arch="v2"`, invalid `arch="v3"` raises `ValueError` in both `PPOConfig.from_json` and
   `PolicyNet(..., arch="v3")`.
2. **v1 bit-for-bit identity:** `PolicyNet(spec, width=16, blocks=2)` and `PolicyNet(spec, width=16,
   blocks=2, arch="v1")` with the same seed produce bit-identical weights and forward outputs; existing
   `arch="v1"` checkpoints load cleanly when `arch` key is absent from saved config JSON.
3. **Shapes, submodule grouping & spatial/global properties (`arch="v2"` on both `v1` and `v2` specs,
   sizes `6` and `11`, widths `8` and `16`, blocks `1`, `2`, `8`):**
   - Exact output shapes for `trunk`, `actor_logits`, `kind_logits`, `arg_logits`, `value`, `act`,
     `evaluate`.
   - Submodule names `actor_head`, `kind_head`, `arg_head`, `value_head` exist and cover heads cleanly.
   - **Globals reach `fmap` and `actor_logits`:** perturbing `globals` changes `fmap` (and tile actor
     logits, not just `global_logit`!) once `film` has non-zero weights (and also verify non-zero
     gradients flow from `actor_logits[:, :3*N]` back into `global_mlp` and `film` even at step 0!).
   - **Full-board spatial communication:** perturbing a distant corner of `board` reaches the opposite
     corner of `fmap` in 1 `ResBlockV2` via pooled-context FiLM (unlike `arch="v1"` with 1 block!).
   - **CoordConv breaks translation symmetry:** an all-zero board produces spatially non-uniform `fmap`
     at corners vs center.
   - **Deep-trunk numerical stability (`blocks=8`):** forward activation norms and backward gradient norms
     stay bounded and finite (no explosion compared to `blocks=1`).
   - **`ArgHeadV2` properties:**
     - Perturbing `fmap` at tile `t` (other than actor tile) changes ONLY `arg_logits[:, t]` (preserving
       per-tile locality via per-tile `LayerNorm`).
     - Separate `move` vs `attack` keys: perturbing `tile_keys[0]` (`move`) changes `move` logits and
       leaves `attack` logits bit-identical (and vice versa).
     - Relative-position bias: mutating `rel_bias[0]` at offset `(dr, dc)` shifts `arg_logits[b, t]`
       ONLY where `(r_t - r_actor, c_t - c_actor) == (dr, dc)` (across multiple actors and map sizes)
       and leaves `attack` logits untouched!
     - Full gradient flow through all parameters (including `film`, `rel_bias`, `q_norm`, `k_norms`,
       `tile_keys`, `global_mlp`, `stem`, `blocks`) during `evaluate()` + `backward()`.
4. **`StackedSnapshots` (`torch.func.vmap`) parity on `arch="v2"`:** stacked forward (`trunk`,
   `actor_logits`, `kind_logits`, `arg_logits`) and `stacked_batch_policy` match individual
   `PolicyNet(..., arch="v2")` outputs (atol `1e-5`) and pick identical greedy/sampled actions across
   multiple snapshots on `ruleset="v2"`.
5. **End-to-end training, self-play (`opponent_forward="stacked"`), evaluation, and checkpointing:**
   short CPU `train.run` with `ruleset="v2", arch="v2", selfplay=True, opponent_forward="stacked"`
   (e.g., `size=6, width=8, blocks=2, num_envs=2, num_steps=8, total_steps=32`) + `checkpoint.load_net`
   restoring `arch="v2"` automatically + `evaluate` + verifying `load_net` still loads an old v1
   checkpoint without `"arch"` in its JSON.

Mutants to aim for: `arch="v2"` ignored (falls back to v1); `film` missing `mean_pool(x)` (no full-board
1-block communication) or missing `g_emb` (no gradient from spatial actors to `global_mlp`); `CoordConv`
omitted (translation-invariant on zero board); `GroupNorm` failing on `width=8` or non-divisible width;
missing residual scaling; `move` and `attack` still sharing `tile_keys`; `rel_bias` indexing `(actor - target)`
backwards or leaking across `move`/`attack`; `StackedSnapshots` crashing under `vmap` on non-persistent
buffers; `checkpoint.load_net` or `train.run` forgetting to pass `arch=cfg.arch`.

## Gate

`pixi run check` exits 0 (fmt, clippy `-D warnings`, all Rust, Python, and RL tests green). `rules_hash()`
== `"a8b52b91eae34503"` and `obs_hash(11)` == `"d5f9b1033cfd55ca"` unchanged.

## Notes for the implementer

- Keep `arch="v1"` completely untouched so every existing checkpoint and `test_model.py` assertion stays
  bit-for-bit identical.
- Keep all buffers (`coords`, `rel_index`, `has_arg`) `persistent=False` and stateless (no running stats)
  so `torch.func.vmap` in `stacked.py` works out of the box and state dicts stay clean.
- If a test contradicts this brief, stop and add a `docs/QUESTIONS.md` entry (next number **Q58**).
