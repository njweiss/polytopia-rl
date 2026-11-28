"""Milestone 22 Phase A tests: v1 -> v2 policy transfer (prefix warm-start & policy distillation).

Covers:
1. PPOConfig transfer fields, validation, and shipped v2 configs / transfer22 experiment spec.
2. Prefix warm-start (`warm_start_v1_to_v2`, `load_warm_start`): zero-init of new v2 input columns,
   fresh orthogonal init on new output rows/heads, name-based enum matching, exact step-0 functional
   preservation on v2 observations with non-zero v2 channels/globals, immediate gradient flow, and
   precondition validation.
3. Numerically stable temperature-scaled `masked_kl` (vs independent NumPy float64 reference,
   gradient isolation) and linear annealing `distill_schedule`.
4. Factored policy distillation (`compute_teacher_logits`, `distill_kl`): v1 -> v2 into `arch="v2"`
   of different width/blocks across mixed batches (v1 kinds, v2-only kinds, empty v1 kind-mask rows,
   v2-only enum-arg rows), gradient descent convergence with frozen teacher, same-spec v2 -> v2
   without truncation, and spec/size mismatch errors.
5. `PolicyNet.evaluate(return_logits=True)`, single teacher pass per rollout in `train_update`, and
   end-to-end CPU `train.run` for both `warmstart_v1` and `distill_v2` (including mid-decay exact
   resume and unchanged v1 rules/obs hashes).
"""

from __future__ import annotations

import copy
import json
import math
import pathlib

import numpy as np
import polytopia_sim as ps
import pytest
import torch

from polytopia_rl import experiment, train
from polytopia_rl.checkpoint import Checkpoint, load_net, save_checkpoint
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.model import Features, PolicyNet
from polytopia_rl.ppo import RolloutBuffer, collect_rollout, train_update
from polytopia_rl.transfer import (
    compute_teacher_logits,
    distill_kl,
    distill_schedule,
    load_warm_start,
    masked_kl,
    warm_start_v1_to_v2,
)
from polytopia_rl.vec_env import make_vec_env

REPO = pathlib.Path(__file__).resolve().parents[3]

# v1 action-type indices (0..9) and v2-only action-type indices (10..14).
MOVE = 0
ATTACK = 1
CAPTURE = 2
TRAIN = 5
CHOOSE_REWARD = 6
SITE = 7
RESEARCH = 8
END_TURN = 9
DISBAND = 10
BUILD_ROAD = 11
BUILD_BUILDING = 12
BUILD_MONUMENT = 13
DESTROY = 14


def _spec(size: int = 6, ruleset: str = "v2") -> dict:
    return ps.Env(size=size, opponent="random", agent_seat="seed_parity", turn_cap=4, ruleset=ruleset).spec()


def _randomized_net(spec: dict, seed: int, width: int = 16, blocks: int = 2, arch: str = "v1", std: float = 0.25) -> PolicyNet:
    """PolicyNet with all weights and biases randomized (non-zero biases catch missing bias copies)."""
    torch.manual_seed(seed)
    net = PolicyNet(spec, width=width, blocks=blocks, arch=arch)
    gen = torch.Generator().manual_seed(seed + 10_000)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith(".weight") and p.ndim == 1:
                p.copy_(1.0 + torch.randn(p.shape, generator=gen) * std)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * std)
    return net


def _real_v2_obs(size: int = 6, b: int = 4, base: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    boards, globs = [], []
    for s in range(b):
        env = ps.Env(size=size, opponent="random", agent_seat="seed_parity", turn_cap=4, ruleset="v2")
        env.reset(base + s)
        x, y = env.observation()
        boards.append(x)
        globs.append(y)
    return torch.from_numpy(np.stack(boards)), torch.from_numpy(np.stack(globs))


def _save_ckpt(path: pathlib.Path, net: PolicyNet, cfg: PPOConfig, spec: dict, step: int = 16) -> str:
    save_checkpoint(
        str(path),
        model=net,
        optimizer=None,
        cfg=cfg,
        global_step=step,
        stage=0,
        seed_index={},
        rng_state={},
        spec=spec,
    )
    return str(path)


def _np_masked_kl(t_logits: np.ndarray, s_logits: np.ndarray, mask: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Independent float64 reference for `T^2 * sum_i p_t[i] * (log p_t[i] - log p_s[i])`."""
    t = np.where(mask, np.asarray(t_logits, dtype=np.float64) / temp, -np.inf)
    s = np.where(mask, np.asarray(s_logits, dtype=np.float64) / temp, -np.inf)
    t_max = t.max(axis=-1, keepdims=True)
    s_max = s.max(axis=-1, keepdims=True)
    log_pt = t - (t_max + np.log(np.exp(t - t_max).sum(axis=-1, keepdims=True)))
    log_ps = s - (s_max + np.log(np.exp(s - s_max).sum(axis=-1, keepdims=True)))
    pt = np.exp(log_pt)
    kl = np.where(mask, pt * np.where(mask, log_pt - log_ps, 0.0), 0.0).sum(axis=-1)
    return np.maximum(kl, 0.0) * (temp * temp)


def _metrics(run_dir: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]


def _untimed(row: dict) -> dict:
    return {k: v for k, v in row.items() if "sps" not in k and "time" not in k}


def _tiny_v2_cfg(**kw) -> PPOConfig:
    base = dict(
        size=6,
        turn_cap=4,
        ruleset="v2",
        arch="v1",
        width=8,
        blocks=1,
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=2,
        total_steps=32,
        eval_every=16,
        eval_seeds=1,
        device="cpu",
        seed=1,
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    return PPOConfig(**{**base, **kw})


# ---------------------------------------------------------------- 1. Config & shipped specs


def test_ppo_config_transfer_defaults_round_trip_and_overrides(tmp_path):
    default = PPOConfig()
    assert default.warm_start_checkpoint is None
    assert default.teacher_checkpoint is None
    assert default.distill_coef == 0.0
    assert default.distill_steps is None
    assert default.distill_temp == 1.0

    # Legacy JSON without transfer keys keeps defaults.
    legacy = json.loads(default.to_json())
    for k in ("warm_start_checkpoint", "teacher_checkpoint", "distill_coef", "distill_steps", "distill_temp"):
        legacy.pop(k, None)
    restored_legacy = PPOConfig.from_json(json.dumps(legacy))
    assert restored_legacy == default

    # Non-default distillation and warm-start configs round-trip cleanly.
    distill_cfg = PPOConfig(
        ruleset="v2",
        arch="v2",
        teacher_checkpoint="runs/curriculum_seed1/latest.pt",
        distill_coef=0.1,
        distill_steps=3_000_000,
        distill_temp=2.0,
    )
    assert PPOConfig.from_json(distill_cfg.to_json()) == distill_cfg

    const_distill = PPOConfig(
        ruleset="v2",
        arch="v2",
        teacher_checkpoint="runs/curriculum_seed1/latest.pt",
        distill_coef=0.05,
        distill_steps=0,
    )
    assert PPOConfig.from_json(const_distill.to_json()) == const_distill

    ws_cfg = PPOConfig(ruleset="v2", arch="v1", warm_start_checkpoint="runs/curriculum_seed1/latest.pt")
    assert PPOConfig.from_json(ws_cfg.to_json()) == ws_cfg

    cfg_path = tmp_path / "base.json"
    cfg_path.write_text(json.dumps({"ruleset": "v2", "arch": "v2"}))
    loaded = load_config(
        str(cfg_path),
        [
            "teacher_checkpoint=runs/v1/latest.pt",
            "distill_coef=0.15",
            "distill_steps=2000000",
            "distill_temp=1.5",
        ],
    )
    assert loaded.teacher_checkpoint == "runs/v1/latest.pt"
    assert loaded.distill_coef == pytest.approx(0.15)
    assert loaded.distill_steps == 2_000_000
    assert loaded.distill_temp == pytest.approx(1.5)


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"distill_coef": -0.01},
        {"distill_coef": -1.0, "teacher_checkpoint": "runs/v1/latest.pt"},
        {"distill_coef": 0.1, "teacher_checkpoint": None},
        {"distill_coef": 0.001},
        {"distill_steps": -1},
        {"distill_steps": -100, "teacher_checkpoint": "runs/v1/latest.pt", "distill_coef": 0.1},
        {"distill_temp": 0.0},
        {"distill_temp": -1.0},
        {"distill_temp": -0.5, "teacher_checkpoint": "runs/v1/latest.pt", "distill_coef": 0.1},
    ],
)
def test_ppo_config_transfer_validation_errors(tmp_path, bad_fields):
    data = json.loads(PPOConfig().to_json())
    data.update(bad_fields)
    with pytest.raises(ValueError) as exc_json:
        PPOConfig.from_json(json.dumps(data))
    assert "unknown config keys" not in str(exc_json.value)

    cfg_path = tmp_path / "base.json"
    cfg_path.write_text(json.dumps({"ruleset": "v2"}))
    overrides = [f"{k}={json.dumps(v)}" for k, v in bad_fields.items()]
    with pytest.raises(ValueError) as exc_load:
        load_config(str(cfg_path), overrides)
    assert "unknown config keys" not in str(exc_load.value)


def test_shipped_v2_configs_and_transfer22_experiment_spec():
    v2_curr_path = REPO / "python" / "configs" / "v2_curriculum.json"
    v2_sp_path = REPO / "python" / "configs" / "v2_selfplay.json"
    exp_path = REPO / "python" / "experiments" / "transfer22.json"

    assert v2_curr_path.exists(), f"missing {v2_curr_path}"
    assert v2_sp_path.exists(), f"missing {v2_sp_path}"
    assert exp_path.exists(), f"missing {exp_path}"

    v1_curr = load_config(str(REPO / "python" / "configs" / "curriculum.json"))
    v2_curr = load_config(str(v2_curr_path))
    assert v2_curr.ruleset == "v2"
    assert v2_curr.arch == "v2"
    assert v2_curr.selfplay is False
    assert v2_curr.stages == v1_curr.stages
    assert len(v2_curr.stages) == 3

    v2_sp = load_config(str(v2_sp_path))
    assert v2_sp.ruleset == "v2"
    assert v2_sp.arch == "v2"
    assert v2_sp.selfplay is True
    assert v2_sp.p_bot == pytest.approx(0.4)
    assert v2_sp.num_envs == 128
    assert v2_sp.num_steps == 32
    assert v2_sp.lr == pytest.approx(1e-4)
    assert v2_sp.opponent_forward == "per_snapshot"

    spec = experiment.load_spec(str(exp_path))
    assert spec.base_config == "python/configs/v2_curriculum.json"
    assert spec.seeds == [1, 2, 3]
    assert spec.total_steps == 10_000_000
    assert spec.init_checkpoint is None
    assert spec.reference is None
    assert spec.primary == ["rush_e0", "economy_e0"]
    assert list(spec.arms.keys()) == ["scratch_v1", "scratch_v2", "warmstart_v1", "distill_v2"]

    arm_cfgs = {}
    for arm, overrides in spec.arms.items():
        cli_overrides = [f"seed={spec.seeds[0]}", f"total_steps={spec.total_steps}"] + [
            f"{k}={json.dumps(v)}" for k, v in overrides.items()
        ]
        arm_cfgs[arm] = load_config(str(REPO / spec.base_config), cli_overrides)
        assert arm_cfgs[arm].ruleset == "v2"

    assert arm_cfgs["scratch_v1"].arch == "v1"
    assert arm_cfgs["scratch_v1"].warm_start_checkpoint is None
    assert arm_cfgs["scratch_v1"].teacher_checkpoint is None
    assert arm_cfgs["scratch_v1"].distill_coef == 0.0

    assert arm_cfgs["scratch_v2"].arch == "v2"
    assert arm_cfgs["scratch_v2"].warm_start_checkpoint is None
    assert arm_cfgs["scratch_v2"].teacher_checkpoint is None
    assert arm_cfgs["scratch_v2"].distill_coef == 0.0

    assert arm_cfgs["warmstart_v1"].arch == "v1"
    assert arm_cfgs["warmstart_v1"].warm_start_checkpoint is not None
    assert arm_cfgs["warmstart_v1"].distill_coef == 0.0

    assert arm_cfgs["distill_v2"].arch == "v2"
    assert arm_cfgs["distill_v2"].teacher_checkpoint is not None
    assert arm_cfgs["distill_v2"].distill_coef > 0.0
    assert arm_cfgs["distill_v2"].distill_steps is not None and arm_cfgs["distill_v2"].distill_steps > 0


# ---------------------------------------------------------------- 2. Prefix warm-start


def test_warm_start_v1_to_v2_weights_load_and_step0_exact_preservation(tmp_path):
    size = 6
    n_tiles = size * size
    v1_spec = _spec(size, "v1")
    v2_spec = _spec(size, "v2")
    c1, g1, k1 = v1_spec["channels"], v1_spec["globals"], len(v1_spec["action_types"])
    assert (c1, g1, k1) == (37, 18, 10)

    v1_net = _randomized_net(v1_spec, seed=11, width=16, blocks=2, arch="v1")
    v1_state = copy.deepcopy(v1_net.state_dict())

    torch.manual_seed(77)
    dst_net = PolicyNet(v2_spec, width=16, blocks=2, arch="v1")
    pre_kind_w = dst_net.kind_head.mlp[2].weight.detach().clone()
    pre_kind_b = dst_net.kind_head.mlp[2].bias.detach().clone()
    pre_kind_emb = dst_net.arg_head.kind_embed.weight.detach().clone()
    pre_tile_bias = dst_net.arg_head.tile_bias.detach().clone()
    pre_enum = [(m.weight.detach().clone(), m.bias.detach().clone()) for m in dst_net.arg_head.enum]

    warm_start_v1_to_v2(dst_net, v1_state, v1_spec, v2_spec)

    # New v2 input columns are strictly zero; v1 prefix columns & biases match verbatim.
    assert torch.equal(dst_net.stem.weight[:, :c1], v1_net.stem.weight)
    assert torch.all(dst_net.stem.weight[:, c1:] == 0)
    assert torch.equal(dst_net.stem.bias, v1_net.stem.bias)

    assert torch.equal(dst_net.global_mlp[0][0].weight[:, :g1], v1_net.global_mlp[0][0].weight)
    assert torch.all(dst_net.global_mlp[0][0].weight[:, g1:] == 0)
    assert torch.equal(dst_net.global_mlp[0][0].bias, v1_net.global_mlp[0][0].bias)
    assert torch.equal(dst_net.global_mlp[0][2].weight, v1_net.global_mlp[0][2].weight)
    assert torch.equal(dst_net.global_mlp[0][2].bias, v1_net.global_mlp[0][2].bias)

    # New output rows retain dst_net's non-zero orthogonal init.
    assert torch.equal(dst_net.kind_head.mlp[2].weight[:k1], v1_net.kind_head.mlp[2].weight)
    assert torch.equal(dst_net.kind_head.mlp[2].bias[:k1], v1_net.kind_head.mlp[2].bias)
    assert torch.equal(dst_net.kind_head.mlp[2].weight[k1:], pre_kind_w[k1:])
    assert torch.equal(dst_net.kind_head.mlp[2].bias[k1:], pre_kind_b[k1:])
    assert dst_net.kind_head.mlp[2].weight[k1:].abs().sum().item() > 0

    assert torch.equal(dst_net.arg_head.kind_embed.weight[:k1], v1_net.arg_head.kind_embed.weight)
    assert torch.equal(dst_net.arg_head.kind_embed.weight[k1:], pre_kind_emb[k1:])
    assert torch.equal(dst_net.arg_head.tile_bias[:k1], v1_net.arg_head.tile_bias)
    assert torch.equal(dst_net.arg_head.tile_bias[k1:], pre_tile_bias[k1:])

    # Enum heads: v1 rows match v1_net; new rows & brand-new v2 heads retain non-zero init.
    v1_enum_by_name = {v1_spec["action_types"][k]: m for k, m in zip(v1_net.arg_head.enum_kinds, v1_net.arg_head.enum)}
    for idx, (k, m) in enumerate(zip(dst_net.arg_head.enum_kinds, dst_net.arg_head.enum)):
        name = v2_spec["action_types"][k]
        w_pre, b_pre = pre_enum[idx]
        if name in v1_enum_by_name:
            src_m = v1_enum_by_name[name]
            v1_len = src_m.weight.shape[0]
            assert torch.equal(m.weight[:v1_len], src_m.weight)
            assert torch.equal(m.bias[:v1_len], src_m.bias)
            if m.weight.shape[0] > v1_len:
                assert torch.equal(m.weight[v1_len:], w_pre[v1_len:])
                assert torch.equal(m.bias[v1_len:], b_pre[v1_len:])
                assert m.weight[v1_len:].abs().sum().item() > 0, f"{name} new rows zeroed"
        else:
            assert torch.equal(m.weight, w_pre)
            assert torch.equal(m.bias, b_pre)
            assert m.weight.abs().sum().item() > 0, f"brand-new enum head {name} zeroed"

    # Exact step-0 functional preservation on real v2 observations WITH non-zero v2 channels & globals.
    board_v2, glob_v2 = _real_v2_obs(size=size, b=4, base=3)
    assert board_v2[:, c1:].abs().sum().item() > 0, "real v2 obs expected to have non-zero v2 channels (e.g., lighthouse)"
    # Inject strong signal into v2-only channels (road/preview/etc.) and v2-only globals too.
    gen = torch.Generator().manual_seed(999)
    board_v2 = board_v2.clone()
    glob_v2 = glob_v2.clone()
    board_v2[:, c1:] += torch.randn(board_v2[:, c1:].shape, generator=gen) * 2.5
    glob_v2[:, g1:] += torch.randn(glob_v2[:, g1:].shape, generator=gen) * 2.5

    board_v1 = board_v2[:, :c1]
    glob_v1 = glob_v2[:, :g1]

    with torch.no_grad():
        f1 = v1_net.trunk(board_v1, glob_v1)
        f2 = dst_net.trunk(board_v2, glob_v2)
        torch.testing.assert_close(f2.fmap, f1.fmap, atol=1e-5, rtol=0)
        torch.testing.assert_close(f2.g, f1.g, atol=1e-5, rtol=0)
        torch.testing.assert_close(dst_net.value(f2), v1_net.value(f1), atol=1e-5, rtol=0)
        torch.testing.assert_close(dst_net.actor_logits(f2), v1_net.actor_logits(f1), atol=1e-5, rtol=0)

        actors = torch.tensor([5, n_tiles + 7, 2 * n_tiles + 11, 3 * n_tiles], dtype=torch.int64)
        torch.testing.assert_close(dst_net.kind_logits(f2, actors)[:, :k1], v1_net.kind_logits(f1, actors), atol=1e-5, rtol=0)

        for kind, actor_val, v1_len in [
            (MOVE, 5, n_tiles),
            (ATTACK, 9, n_tiles),
            (TRAIN, n_tiles + 4, 5),
            (CHOOSE_REWARD, n_tiles + 4, 8),
            (SITE, 2 * n_tiles + 11, 6),
            (RESEARCH, 3 * n_tiles, 9),
        ]:
            a_t = torch.full((4,), actor_val, dtype=torch.int64)
            k_t = torch.full((4,), kind, dtype=torch.int64)
            torch.testing.assert_close(
                dst_net.arg_logits(f2, a_t, k_t)[:, :v1_len],
                v1_net.arg_logits(f1, a_t, k_t)[:, :v1_len],
                atol=1e-5,
                rtol=0,
            )

    # `load_warm_start` loads from a saved v1 checkpoint and produces identical weights.
    v1_cfg = PPOConfig(size=size, ruleset="v1", arch="v1", width=16, blocks=2, device="cpu")
    ckpt_path = _save_ckpt(tmp_path / "v1_src.pt", v1_net, v1_cfg, v1_spec)
    torch.manual_seed(77)
    dst_from_file = PolicyNet(v2_spec, width=16, blocks=2, arch="v1")
    ck = load_warm_start(dst_from_file, ckpt_path, v2_spec)
    assert isinstance(ck, Checkpoint)
    for k, v in dst_net.state_dict().items():
        assert torch.equal(v, dst_from_file.state_dict()[k]), f"load_warm_start mismatch at {k}"


def test_warm_start_v1_to_v2_matches_enum_heads_by_kind_name():
    """Mutant catcher: copying `arg_head.enum` by positional index fails if an earlier kind gains an enum arg."""
    v1_spec = _spec(6, "v1")
    v2_spec = copy.deepcopy(_spec(6, "v2"))
    # Give `capture` (index 2, no-arg in v1) a synthetic enum arg of length 2 in v2 so all v1 enum
    # heads shift by +1 in `dst_net.arg_head.enum` while `v2_spec["action_types"][:10] == v1_spec["action_types"]`.
    v2_spec["arg_lens"][CAPTURE] = 2

    v1_net = _randomized_net(v1_spec, seed=19, width=8, blocks=1, arch="v1")
    dst_net = PolicyNet(v2_spec, width=8, blocks=1, arch="v1")
    pre_capture_w = dst_net.arg_head.enum[0].weight.detach().clone()

    warm_start_v1_to_v2(dst_net, v1_net.state_dict(), v1_spec, v2_spec)

    assert dst_net.arg_head.enum_kinds[0] == CAPTURE
    assert torch.equal(dst_net.arg_head.enum[0].weight, pre_capture_w)

    v1_by_name = {v1_spec["action_types"][k]: m for k, m in zip(v1_net.arg_head.enum_kinds, v1_net.arg_head.enum)}
    dst_by_name = {v2_spec["action_types"][k]: m for k, m in zip(dst_net.arg_head.enum_kinds, dst_net.arg_head.enum)}
    for name, src_m in v1_by_name.items():
        dst_m = dst_by_name[name]
        n = src_m.weight.shape[0]
        assert torch.equal(dst_m.weight[:n], src_m.weight), f"enum head {name} not matched by name"
        assert torch.equal(dst_m.bias[:n], src_m.bias), f"enum head {name} bias not matched by name"


def test_warm_start_v1_to_v2_immediate_gradient_flow():
    size = 6
    n_tiles = size * size
    v1_spec = _spec(size, "v1")
    v2_spec = _spec(size, "v2")
    c1, g1, k1 = v1_spec["channels"], v1_spec["globals"], len(v1_spec["action_types"])

    v1_net = _randomized_net(v1_spec, seed=23, width=16, blocks=2, arch="v1")
    dst_net = PolicyNet(v2_spec, width=16, blocks=2, arch="v1")
    warm_start_v1_to_v2(dst_net, v1_net.state_dict(), v1_spec, v2_spec)

    # Batch exercising v2-extended enum rows and brand-new v2 kinds/enum heads on non-zero v2 inputs.
    rows = [
        (n_tiles + 4, TRAIN, 5, 8),
        (2 * n_tiles + 7, SITE, 6, 7),
        (3 * n_tiles, RESEARCH, 10, 16),
        (2 * n_tiles + 9, BUILD_BUILDING, 1, 4),
        (2 * n_tiles + 12, BUILD_MONUMENT, 2, 3),
        (5, DISBAND, 0, 0),
    ]
    b = len(rows)
    board, glob = _real_v2_obs(size=size, b=b, base=5)
    gen = torch.Generator().manual_seed(123)
    board = board.clone()
    glob = glob.clone()
    board[:, c1:] += torch.randn(board[:, c1:].shape, generator=gen)
    glob[:, g1:] += torch.randn(glob[:, g1:].shape, generator=gen)

    actors = torch.tensor([r[0] for r in rows], dtype=torch.int64)
    kinds = torch.tensor([r[1] for r in rows], dtype=torch.int64)
    args = torch.tensor([r[2] for r in rows], dtype=torch.int64)
    am = torch.ones(b, 3 * n_tiles + 1, dtype=torch.bool)
    km = torch.ones(b, len(v2_spec["action_types"]), dtype=torch.bool)
    xm = torch.zeros(b, n_tiles, dtype=torch.bool)
    for i, (_, _, _, alen) in enumerate(rows):
        if alen > 0:
            xm[i, :alen] = True

    dst_net.zero_grad(set_to_none=True)
    ev = dst_net.evaluate(board, glob, actors, kinds, args, am, km, xm)
    (-(ev.logp.mean()) + ev.value.pow(2).mean()).backward()

    assert dst_net.stem.weight.grad is not None and torch.isfinite(dst_net.stem.weight.grad).all()
    assert dst_net.stem.weight.grad[:, c1:].abs().sum().item() > 0, "no gradient into new stem columns"

    g_w_grad = dst_net.global_mlp[0][0].weight.grad
    assert g_w_grad is not None and torch.isfinite(g_w_grad).all()
    assert g_w_grad[:, g1:].abs().sum().item() > 0, "no gradient into new global_mlp columns"

    k_w_grad = dst_net.kind_head.mlp[2].weight.grad
    assert k_w_grad is not None and k_w_grad[k1:].abs().sum().item() > 0, "no gradient into new kind_head rows"

    dst_by_name = {v2_spec["action_types"][k]: m for k, m in zip(dst_net.arg_head.enum_kinds, dst_net.arg_head.enum)}
    assert dst_by_name["train"].weight.grad[5:].abs().sum().item() > 0
    assert dst_by_name["site"].weight.grad[6:].abs().sum().item() > 0
    assert dst_by_name["research"].weight.grad[9:].abs().sum().item() > 0
    assert dst_by_name["build_building"].weight.grad.abs().sum().item() > 0
    assert dst_by_name["build_monument"].weight.grad.abs().sum().item() > 0


def test_warm_start_v1_to_v2_validation_errors(tmp_path):
    v1_spec6 = _spec(6, "v1")
    v2_spec6 = _spec(6, "v2")
    v2_spec11 = _spec(11, "v2")

    v1_net = _randomized_net(v1_spec6, seed=1, width=16, blocks=2, arch="v1")
    v1_state = v1_net.state_dict()

    # 1. dst_net.arch == "v2" rejected.
    with pytest.raises(ValueError):
        warm_start_v1_to_v2(PolicyNet(v2_spec6, width=16, blocks=2, arch="v2"), v1_state, v1_spec6, v2_spec6)

    # 2. Mismatched width rejected.
    with pytest.raises(ValueError):
        warm_start_v1_to_v2(PolicyNet(v2_spec6, width=8, blocks=2, arch="v1"), v1_state, v1_spec6, v2_spec6)

    # 3. Mismatched blocks rejected.
    with pytest.raises(ValueError):
        warm_start_v1_to_v2(PolicyNet(v2_spec6, width=16, blocks=1, arch="v1"), v1_state, v1_spec6, v2_spec6)

    # 4. Mismatched map size rejected.
    with pytest.raises(ValueError):
        warm_start_v1_to_v2(PolicyNet(v2_spec11, width=16, blocks=2, arch="v1"), v1_state, v1_spec6, v2_spec11)

    # 5. Non-prefix channel_names / global_names / action_types rejected.
    bad_v1 = copy.deepcopy(v1_spec6)
    bad_v1["channel_names"][0] = "not_explored"
    with pytest.raises(ValueError):
        warm_start_v1_to_v2(PolicyNet(v2_spec6, width=16, blocks=2, arch="v1"), v1_state, bad_v1, v2_spec6)

    # 6. Non-v1 source checkpoint (v2 ruleset checkpoint) rejected by load_warm_start.
    v2_net = PolicyNet(v2_spec6, width=16, blocks=2, arch="v1")
    v2_cfg = PPOConfig(size=6, ruleset="v2", arch="v1", width=16, blocks=2, device="cpu")
    v2_ckpt = _save_ckpt(tmp_path / "v2_src.pt", v2_net, v2_cfg, v2_spec6)
    with pytest.raises(ValueError):
        load_warm_start(PolicyNet(v2_spec6, width=16, blocks=2, arch="v1"), v2_ckpt, v2_spec6)

    # 7. v1 ruleset checkpoint with arch="v2" rejected by load_warm_start.
    v1_arch2 = PolicyNet(v1_spec6, width=16, blocks=2, arch="v2")
    v1_arch2_cfg = PPOConfig(size=6, ruleset="v1", arch="v2", width=16, blocks=2, device="cpu")
    v1_arch2_ckpt = _save_ckpt(tmp_path / "v1_arch2.pt", v1_arch2, v1_arch2_cfg, v1_spec6)
    with pytest.raises(ValueError):
        load_warm_start(PolicyNet(v2_spec6, width=16, blocks=2, arch="v1"), v1_arch2_ckpt, v2_spec6)


# ---------------------------------------------------------------- 3. masked_kl & distill_schedule


@pytest.mark.parametrize("temp", [0.5, 1.0, 2.0])
def test_masked_kl_matches_numpy_reference_and_handles_extremes(temp):
    gen = torch.Generator().manual_seed(42)
    b, k = 8, 12
    t_logits = torch.randn(b, k, generator=gen) * 2.0
    s_logits = torch.randn(b, k, generator=gen) * 2.0
    mask = torch.zeros(b, k, dtype=torch.bool)
    for i in range(b):
        if i == 0:
            mask[i, 3] = True  # single legal option
        else:
            legal = torch.randperm(k, generator=gen)[: (i % 5) + 2]
            mask[i, legal] = True

    # Inject extreme values into masked-out entries to catch 0 * (-inf) / NaN or mask leakage.
    t_extreme = torch.where(mask, t_logits, torch.tensor(1e6))
    s_extreme = torch.where(mask, s_logits, torch.tensor(-1e6))

    got = masked_kl(t_extreme, s_extreme, mask, temp=temp)
    want = torch.from_numpy(_np_masked_kl(t_logits.numpy(), s_logits.numpy(), mask.numpy(), temp=temp)).float()

    assert got.shape == (b,)
    assert torch.isfinite(got).all()
    assert (got >= 0.0).all()
    assert got[0].item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    # Identical distributions on legal options -> exact 0.0 even with extreme masked-out entries.
    s_same = torch.where(mask, t_logits, torch.tensor(-1e6))
    kl_same = masked_kl(t_extreme, s_same, mask, temp=temp)
    assert torch.isfinite(kl_same).all()
    torch.testing.assert_close(kl_same, torch.zeros(b), atol=1e-6, rtol=0)


def test_masked_kl_temperature_scaling_and_gradient_isolation():
    t_logits = torch.tensor([[2.0, 0.5, -1.0, 1e6]], requires_grad=True)
    s_logits = torch.tensor([[-0.5, 1.5, 0.2, -1e6]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False]])

    # Catch missing T^2 scaling explicitly by comparing against reference at T=2.0.
    kl_t2 = masked_kl(t_logits, s_logits, mask, temp=2.0)
    ref_t2 = _np_masked_kl(t_logits.detach().numpy(), s_logits.detach().numpy(), mask.numpy(), temp=2.0)[0]
    assert kl_t2.item() == pytest.approx(ref_t2, abs=1e-5)
    assert kl_t2.item() != pytest.approx(ref_t2 / 4.0, rel=1e-2)

    kl_t2.sum().backward()
    assert t_logits.grad is None or torch.all(t_logits.grad == 0), "gradient leaked into teacher_logits"
    assert s_logits.grad is not None and torch.isfinite(s_logits.grad).all()
    assert s_logits.grad[0, :3].abs().sum().item() > 0
    assert s_logits.grad[0, 3].item() == 0.0


def test_distill_schedule_annealing_and_edge_cases():
    # Standard linear decay: update 1 -> base_coef, midpoint -> 0.5 * base_coef, horizon+1 -> 0.0.
    assert distill_schedule(1, 0.1, total_updates=100, distill_updates=40) == pytest.approx(0.1)
    assert distill_schedule(21, 0.1, total_updates=100, distill_updates=40) == pytest.approx(0.05)
    assert distill_schedule(40, 0.1, total_updates=100, distill_updates=40) == pytest.approx(0.1 * (1.0 - 39 / 40))
    assert distill_schedule(41, 0.1, total_updates=100, distill_updates=40) == pytest.approx(0.0)
    assert distill_schedule(100, 0.1, total_updates=100, distill_updates=40) == pytest.approx(0.0)

    # distill_updates=None uses total_updates as horizon.
    assert distill_schedule(1, 0.2, total_updates=10, distill_updates=None) == pytest.approx(0.2)
    assert distill_schedule(6, 0.2, total_updates=10, distill_updates=None) == pytest.approx(0.1)
    assert distill_schedule(11, 0.2, total_updates=10, distill_updates=None) == pytest.approx(0.0)

    # distill_updates=0 keeps base_coef constant.
    for u in (1, 5, 100, 1000):
        assert distill_schedule(u, 0.1, total_updates=10, distill_updates=0) == pytest.approx(0.1)

    # base_coef=0.0 always returns 0.0.
    for du in (None, 0, 10):
        assert distill_schedule(1, 0.0, total_updates=10, distill_updates=du) == 0.0
        assert distill_schedule(5, 0.0, total_updates=10, distill_updates=du) == 0.0


# ---------------------------------------------------------------- 4. Factored distillation


def test_distill_v1_to_v2_arch_v2_mixed_batch_and_all_edge_cases():
    size = 6
    n_tiles = size * size
    g_actor = 3 * n_tiles
    v1_spec = _spec(size, "v1")
    v2_spec = _spec(size, "v2")
    k1 = len(v1_spec["action_types"])
    k2 = len(v2_spec["action_types"])

    # Frozen v1 teacher (arch="v1", width=8, blocks=1) -> v2 student (arch="v2", width=16, blocks=2).
    teacher = _randomized_net(v1_spec, seed=31, width=8, blocks=1, arch="v1").eval().requires_grad_(False)
    student = _randomized_net(v2_spec, seed=32, width=16, blocks=2, arch="v2")

    # 14 carefully crafted rows covering every v1 kind, all 5 v2-only kinds, empty v1 kind_mask,
    # and v2-only legal enum args for train / site / research.
    # (actor, kind, legal_kinds, legal_args)
    spec_rows = [
        (5, MOVE, [MOVE, ATTACK, END_TURN], [6, 11, 12]),  # 0: v1 tile MOVE
        (8, ATTACK, [MOVE, ATTACK], [9, 14]),  # 1: v1 tile ATTACK
        (n_tiles + 4, TRAIN, [TRAIN, END_TURN], [0, 1, 2, 5, 6]),  # 2: TRAIN mixed v1+v2 args
        (g_actor, RESEARCH, [RESEARCH, END_TURN], [0, 2, 10, 12]),  # 3: RESEARCH mixed v1+v2 args
        (g_actor, END_TURN, [RESEARCH, END_TURN], []),  # 4: v1 no-arg END_TURN
        (7, DISBAND, [MOVE, DISBAND], []),  # 5: v2-only DISBAND
        (2 * n_tiles + 3, BUILD_ROAD, [SITE, BUILD_ROAD], []),  # 6: v2-only BUILD_ROAD
        (2 * n_tiles + 4, BUILD_BUILDING, [SITE, BUILD_BUILDING], [0, 1, 2, 3]),  # 7: v2-only BUILD_BUILDING (has arg!)
        (2 * n_tiles + 5, BUILD_MONUMENT, [SITE, BUILD_MONUMENT], [0, 1, 2]),  # 8: v2-only BUILD_MONUMENT (has arg!)
        (2 * n_tiles + 6, DESTROY, [SITE, DESTROY], []),  # 9: v2-only DESTROY
        (2 * n_tiles + 8, BUILD_ROAD, [BUILD_ROAD, BUILD_BUILDING], []),  # 10: kind_mask[:, :10] all-False!
        (g_actor, RESEARCH, [RESEARCH, END_TURN], [9, 10, 14, 15]),  # 11: RESEARCH with ONLY v2 techs (9..15)
        (n_tiles + 4, TRAIN, [TRAIN, END_TURN], [5, 6, 7]),  # 12: TRAIN with ONLY v2 units (5..7)
        (2 * n_tiles + 9, SITE, [SITE, BUILD_ROAD], [6]),  # 13: SITE with ONLY v2 burn_forest (6)
    ]
    b = len(spec_rows)
    board, glob = _real_v2_obs(size=size, b=b, base=10)
    actors = torch.tensor([r[0] for r in spec_rows], dtype=torch.int64)
    kinds = torch.tensor([r[1] for r in spec_rows], dtype=torch.int64)

    actor_mask = torch.zeros(b, 3 * n_tiles + 1, dtype=torch.bool)
    kind_mask = torch.zeros(b, k2, dtype=torch.bool)
    arg_mask = torch.zeros(b, n_tiles, dtype=torch.bool)
    for i, (a, _, lks, las) in enumerate(spec_rows):
        actor_mask[i, a] = True
        actor_mask[i, (a + 1) % (3 * n_tiles + 1)] = True
        for lk in lks:
            kind_mask[i, lk] = True
        for la in las:
            arg_mask[i, la] = True

    t_logits = compute_teacher_logits(teacher, v1_spec, v2_spec, board, glob, actors, kinds)
    assert len(t_logits) == 3
    assert t_logits[0].shape == (b, 3 * n_tiles + 1)
    assert t_logits[1].shape == (b, k1)
    assert t_logits[2].shape == (b, n_tiles)

    # Teacher logits on v1 rows match direct teacher forward on projected v1 obs.
    with torch.no_grad():
        tf = teacher.trunk(board[:, : v1_spec["channels"]], glob[:, : v1_spec["globals"]])
        torch.testing.assert_close(t_logits[0], teacher.actor_logits(tf), atol=1e-6, rtol=0)
        torch.testing.assert_close(t_logits[1], teacher.kind_logits(tf, actors), atol=1e-6, rtol=0)
        tf5 = Features(tf.fmap[:5], tf.g[:5])
        torch.testing.assert_close(t_logits[2][:5], teacher.arg_logits(tf5, actors[:5], kinds[:5]), atol=1e-6, rtol=0)

    sf = student.trunk(board, glob)
    s_logits = (student.actor_logits(sf), student.kind_logits(sf, actors), student.arg_logits(sf, actors, kinds))

    total, parts = distill_kl(t_logits, s_logits, kinds, actor_mask, kind_mask, arg_mask, v1_spec, v2_spec, temp=1.5)
    assert set(parts.keys()) == {"actor", "kind", "arg"}
    assert total.shape == (b,)
    assert torch.isfinite(total).all()
    assert (total >= 0.0).all()
    torch.testing.assert_close(total, parts["actor"] + parts["kind"] + parts["arg"], atol=1e-6, rtol=0)

    # Active v1 rows have positive Stage-3 KL.
    for i in (0, 1, 2, 3):
        assert parts["arg"][i].item() > 0.0, f"row {i} should have positive Stage-3 KL"

    # No-arg v1 row (4), all 5 v2-only kinds (5..9), row 10, and v2-only-arg rows (11..13) have exact 0.0 Stage-3 KL.
    for i in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        assert parts["arg"][i].item() == 0.0, f"row {i} must have zero Stage-3 KL, got {parts['arg'][i].item()}"

    # Row 10 has kind_mask[:, :10] all-False -> exact 0.0 Stage-2 KL.
    assert parts["kind"][10].item() == 0.0, f"row 10 must have zero Stage-2 KL, got {parts['kind'][10].item()}"
    assert parts["actor"][10].item() > 0.0

    # Perturbing student's v2-only arg logits on mixed rows (TRAIN[5:8], RESEARCH[9:16]) must NOT change Stage-3 KL.
    s_arg_pert = s_logits[2].clone()
    s_arg_pert[2, 5:8] += 50.0
    s_arg_pert[3, 9:16] -= 50.0
    _, parts_pert = distill_kl(
        t_logits, (s_logits[0], s_logits[1], s_arg_pert), kinds, actor_mask, kind_mask, arg_mask, v1_spec, v2_spec, temp=1.5
    )
    torch.testing.assert_close(parts_pert["arg"][[2, 3]], parts["arg"][[2, 3]], atol=1e-6, rtol=0)


def test_distill_gradient_descent_reduces_kl_and_leaves_teacher_untouched():
    size = 6
    n_tiles = size * size
    v1_spec = _spec(size, "v1")
    v2_spec = _spec(size, "v2")

    teacher = _randomized_net(v1_spec, seed=41, width=8, blocks=1, arch="v1").eval().requires_grad_(False)
    teacher_before = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    student = PolicyNet(v2_spec, width=16, blocks=2, arch="v2")
    opt = torch.optim.Adam(student.parameters(), lr=5e-3)

    board, glob = _real_v2_obs(size=size, b=6, base=20)
    actors = torch.tensor([5, 8, n_tiles + 4, 3 * n_tiles, 2 * n_tiles + 3, 2 * n_tiles + 5], dtype=torch.int64)
    kinds = torch.tensor([MOVE, ATTACK, TRAIN, RESEARCH, SITE, BUILD_BUILDING], dtype=torch.int64)
    actor_mask = torch.ones(6, 3 * n_tiles + 1, dtype=torch.bool)
    kind_mask = torch.ones(6, len(v2_spec["action_types"]), dtype=torch.bool)
    arg_mask = torch.zeros(6, n_tiles, dtype=torch.bool)
    arg_mask[0, :12] = True
    arg_mask[1, :12] = True
    arg_mask[2, :8] = True
    arg_mask[3, :16] = True
    arg_mask[4, :7] = True
    arg_mask[5, :4] = True

    t_logits = compute_teacher_logits(teacher, v1_spec, v2_spec, board, glob, actors, kinds)

    kls = []
    for _ in range(15):
        sf = student.trunk(board, glob)
        s_logits = (student.actor_logits(sf), student.kind_logits(sf, actors), student.arg_logits(sf, actors, kinds))
        kl, _ = distill_kl(t_logits, s_logits, kinds, actor_mask, kind_mask, arg_mask, v1_spec, v2_spec, temp=1.0)
        loss = kl.mean()
        kls.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert kls[-1] < 0.5 * kls[0], f"KL did not decrease meaningfully: start={kls[0]:.4f}, end={kls[-1]:.4f}"
    for k, before in teacher_before.items():
        assert torch.equal(teacher.state_dict()[k], before), f"teacher weight mutated at {k}"
    for p in teacher.parameters():
        assert p.grad is None


def test_distill_same_spec_v2_to_v2_and_incompatible_specs_raise():
    v1_spec6 = _spec(6, "v1")
    v2_spec6 = _spec(6, "v2")
    v2_spec11 = _spec(11, "v2")
    n_tiles = 36

    t_v2 = _randomized_net(v2_spec6, seed=51, width=8, blocks=1, arch="v2").eval().requires_grad_(False)
    s_v2_same = copy.deepcopy(t_v2)
    s_v2_diff = _randomized_net(v2_spec6, seed=52, width=16, blocks=2, arch="v2")

    board, glob = _real_v2_obs(size=6, b=4, base=1)
    actors = torch.tensor([2 * n_tiles + 4, 2 * n_tiles + 5, 3 * n_tiles, 5], dtype=torch.int64)
    kinds = torch.tensor([BUILD_BUILDING, BUILD_MONUMENT, RESEARCH, MOVE], dtype=torch.int64)
    actor_mask = torch.ones(4, 3 * n_tiles + 1, dtype=torch.bool)
    kind_mask = torch.zeros(4, len(v2_spec6["action_types"]), dtype=torch.bool)
    kind_mask[0, [BUILD_ROAD, BUILD_BUILDING, DESTROY]] = True  # v2-only kinds
    kind_mask[1, [BUILD_MONUMENT, DESTROY]] = True
    kind_mask[2, [RESEARCH, END_TURN]] = True
    kind_mask[3, [MOVE, ATTACK]] = True
    arg_mask = torch.zeros(4, n_tiles, dtype=torch.bool)
    arg_mask[0, :4] = True
    arg_mask[1, :3] = True
    arg_mask[2, 9:16] = True  # v2-only techs
    arg_mask[3, :10] = True

    t_logits = compute_teacher_logits(t_v2, v2_spec6, v2_spec6, board, glob, actors, kinds)
    assert t_logits[1].shape == (4, len(v2_spec6["action_types"]))

    # Identical v2 teacher & student -> exact zero KL across all stages.
    with torch.no_grad():
        sf_same = s_v2_same.trunk(board, glob)
        s_same_logits = (
            s_v2_same.actor_logits(sf_same),
            s_v2_same.kind_logits(sf_same, actors),
            s_v2_same.arg_logits(sf_same, actors, kinds),
        )
        kl_zero, _ = distill_kl(t_logits, s_same_logits, kinds, actor_mask, kind_mask, arg_mask, v2_spec6, v2_spec6)
        torch.testing.assert_close(kl_zero, torch.zeros(4), atol=1e-5, rtol=0)

        # Distinct v2 student -> positive KL on v2-only kinds and v2-only args (no prefix truncation!).
        sf_diff = s_v2_diff.trunk(board, glob)
        s_diff_logits = (
            s_v2_diff.actor_logits(sf_diff),
            s_v2_diff.kind_logits(sf_diff, actors),
            s_v2_diff.arg_logits(sf_diff, actors, kinds),
        )
        _, parts_diff = distill_kl(t_logits, s_diff_logits, kinds, actor_mask, kind_mask, arg_mask, v2_spec6, v2_spec6)
        assert parts_diff["kind"][0].item() > 0.0
        assert parts_diff["arg"][0].item() > 0.0  # BUILD_BUILDING
        assert parts_diff["arg"][1].item() > 0.0  # BUILD_MONUMENT
        assert parts_diff["arg"][2].item() > 0.0  # v2-only RESEARCH techs

    # Mismatched map sizes -> ValueError.
    with pytest.raises(ValueError):
        compute_teacher_logits(t_v2, v2_spec6, v2_spec11, board, glob, actors, kinds)
    with pytest.raises(ValueError):
        distill_kl(t_logits, s_diff_logits, kinds, actor_mask, kind_mask, arg_mask, v2_spec6, v2_spec11)

    # Incompatible reverse direction (v2 teacher -> v1 student) -> ValueError.
    with pytest.raises(ValueError):
        compute_teacher_logits(t_v2, v2_spec6, v1_spec6, board[:, :37], glob[:, :18], actors, kinds.clamp(max=9))


# ---------------------------------------------------------------- 5. evaluate(return_logits), single teacher pass, train.run


@pytest.mark.parametrize("arch", ["v1", "v2"])
def test_policy_net_evaluate_return_logits(arch):
    size = 6
    n_tiles = size * size
    spec = _spec(size, "v2")
    net = _randomized_net(spec, seed=61, width=16, blocks=2, arch=arch)

    board, glob = _real_v2_obs(size=size, b=4, base=2)
    actors = torch.tensor([5, n_tiles + 4, 2 * n_tiles + 8, 3 * n_tiles], dtype=torch.int64)
    kinds = torch.tensor([MOVE, TRAIN, BUILD_BUILDING, END_TURN], dtype=torch.int64)
    args = torch.tensor([6, 1, 2, 0], dtype=torch.int64)
    am = torch.ones(4, 3 * n_tiles + 1, dtype=torch.bool)
    km = torch.ones(4, len(spec["action_types"]), dtype=torch.bool)
    xm = torch.ones(4, n_tiles, dtype=torch.bool)

    ev_default = net.evaluate(board, glob, actors, kinds, args, am, km, xm)
    assert ev_default.logits is None

    ev_with = net.evaluate(board, glob, actors, kinds, args, am, km, xm, return_logits=True)
    assert ev_with.logits is not None and len(ev_with.logits) == 3
    assert torch.equal(ev_with.logp, ev_default.logp)
    assert torch.equal(ev_with.entropy, ev_default.entropy)
    assert torch.equal(ev_with.value, ev_default.value)

    with torch.no_grad():
        f = net.trunk(board, glob)
        torch.testing.assert_close(ev_with.logits[0], net.actor_logits(f), atol=1e-6, rtol=0)
        torch.testing.assert_close(ev_with.logits[1], net.kind_logits(f, actors), atol=1e-6, rtol=0)
        torch.testing.assert_close(ev_with.logits[2], net.arg_logits(f, actors, kinds), atol=1e-6, rtol=0)

    # Returned logits remain attached to the student's autograd graph.
    net.zero_grad(set_to_none=True)
    (ev_with.logits[0].sum() + ev_with.logits[1].sum() + ev_with.logits[2].sum()).backward()
    assert net.stem.weight.grad is not None and net.stem.weight.grad.abs().sum().item() > 0


def test_train_update_single_teacher_pass_per_rollout(monkeypatch):
    size = 6
    v1_spec = _spec(size, "v1")
    v2_spec = _spec(size, "v2")
    cfg = _tiny_v2_cfg(arch="v2", update_epochs=4, num_minibatches=2)

    teacher = _randomized_net(v1_spec, seed=71, width=8, blocks=1, arch="v1").eval().requires_grad_(False)
    teacher_before = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
    student = PolicyNet(v2_spec, width=cfg.width, blocks=cfg.blocks, arch="v2")
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)

    vec = make_vec_env(cfg, seed_base=0)
    vec.reset_all()
    buf = RolloutBuffer(cfg.num_steps, cfg.num_envs, v2_spec, device="cpu")
    collect_rollout(student, vec, buf, gen)

    trunk_calls = 0
    orig_trunk = teacher.trunk

    def spy_trunk(*args, **kwargs):
        nonlocal trunk_calls
        trunk_calls += 1
        return orig_trunk(*args, **kwargs)

    monkeypatch.setattr(teacher, "trunk", spy_trunk)

    # Active distillation: update_epochs=4, num_minibatches=2 (8 minibatches) -> teacher.trunk called ONCE!
    m = train_update(
        student,
        opt,
        buf,
        cfg,
        gen,
        teacher=teacher,
        teacher_spec=v1_spec,
        student_spec=v2_spec,
        distill_coef=0.1,
    )
    assert trunk_calls == 1, f"expected 1 teacher.trunk call per train_update, got {trunk_calls}"
    assert "distill_kl" in m and math.isfinite(m["distill_kl"]) and m["distill_kl"] >= 0.0
    assert m["distill_coef"] == pytest.approx(0.1)
    for k, before in teacher_before.items():
        assert torch.equal(teacher.state_dict()[k], before)

    # distill_coef=0.0 skips teacher forward completely.
    m_zero = train_update(
        student,
        opt,
        buf,
        cfg,
        gen,
        teacher=teacher,
        teacher_spec=v1_spec,
        student_spec=v2_spec,
        distill_coef=0.0,
    )
    assert trunk_calls == 1, f"teacher.trunk should not run when distill_coef == 0.0 (got {trunk_calls})"
    assert m_zero["distill_coef"] == pytest.approx(0.0)

    # Default M21 call (teacher=None) does not add distillation keys.
    m_default = train_update(student, opt, buf, cfg, gen)
    assert "distill_kl" not in m_default and "distill_coef" not in m_default


def test_train_run_warmstart_v1_end_to_end_and_conflict_error(tmp_path):
    v1_spec = _spec(6, "v1")
    v2_spec = _spec(6, "v2")
    v1_net = _randomized_net(v1_spec, seed=81, width=8, blocks=1, arch="v1")
    v1_cfg = PPOConfig(size=6, ruleset="v1", arch="v1", width=8, blocks=1, device="cpu")
    v1_ckpt = _save_ckpt(tmp_path / "v1_teacher.pt", v1_net, v1_cfg, v1_spec)

    # Simultaneous init_checkpoint + warm_start_checkpoint must raise ValueError.
    v2_net = PolicyNet(v2_spec, width=8, blocks=1, arch="v1")
    v2_init_ckpt = _save_ckpt(tmp_path / "v2_init.pt", v2_net, _tiny_v2_cfg(), v2_spec)
    cfg_conflict = _tiny_v2_cfg(warm_start_checkpoint=v1_ckpt)
    with pytest.raises(ValueError):
        train.run(cfg_conflict, str(tmp_path / "conflict"), init_checkpoint=v2_init_ckpt)

    # Clean end-to-end warm-started v2 run.
    cfg = _tiny_v2_cfg(arch="v1", warm_start_checkpoint=v1_ckpt, total_steps=32, eval_every=16)
    run_dir = tmp_path / "ws_run"
    net = train.run(cfg, str(run_dir))

    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "ckpt_final.pt").exists()
    rows = _metrics(run_dir)
    assert len(rows) == 2
    assert all(math.isfinite(r["pg_loss"]) and math.isfinite(r["v_loss"]) for r in rows)

    loaded, ck = load_net(str(run_dir / "latest.pt"), device="cpu")
    assert ck.cfg.ruleset == "v2" and ck.cfg.arch == "v1"
    assert loaded.stem.in_channels == v2_spec["channels"]
    for k, v in net.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_train_run_distill_v2_end_to_end_schedule_resume_and_v1_hashes(tmp_path):
    # Hard constraint: v1 rules_hash and obs_hash(11) remain completely untouched.
    v1_spec_11 = _spec(11, "v1")
    assert v1_spec_11["rules_hash"] == "a8b52b91eae34503"
    assert v1_spec_11["obs_hash"] == "d5f9b1033cfd55ca"

    v1_spec = _spec(6, "v1")
    v1_net = _randomized_net(v1_spec, seed=91, width=8, blocks=1, arch="v1")
    v1_cfg = PPOConfig(size=6, ruleset="v1", arch="v1", width=8, blocks=1, device="cpu")
    teacher_ckpt = _save_ckpt(tmp_path / "v1_teacher.pt", v1_net, v1_cfg, v1_spec)
    teacher_before = {k: v.detach().clone() for k, v in v1_net.state_dict().items()}

    # batch = 2 * 8 = 16; total_steps = 48 (3 updates); distill_steps = 32 (2 updates):
    # update 1 -> 0.1, update 2 -> 0.05, update 3 -> 0.0.
    cfg = _tiny_v2_cfg(
        arch="v2",
        width=8,
        blocks=1,
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=2,
        total_steps=48,
        eval_every=16,
        eval_seeds=1,
        teacher_checkpoint=teacher_ckpt,
        distill_coef=0.1,
        distill_steps=32,
        distill_temp=1.5,
    )

    run_full = tmp_path / "distill_full"
    net_full = train.run(cfg, str(run_full))
    rows_full = _metrics(run_full)
    assert len(rows_full) == 3

    assert rows_full[0]["distill_coef"] == pytest.approx(0.1)
    assert rows_full[1]["distill_coef"] == pytest.approx(0.05)
    assert rows_full[2]["distill_coef"] == pytest.approx(0.0)

    for i in (0, 1):
        assert math.isfinite(rows_full[i]["distill_kl"]) and rows_full[i]["distill_kl"] > 0.0
    assert math.isfinite(rows_full[2]["distill_kl"]) and rows_full[2]["distill_kl"] >= 0.0

    # Teacher checkpoint on disk was not modified.
    teacher_after, _ = load_net(teacher_ckpt, device="cpu")
    for k, before in teacher_before.items():
        assert torch.equal(teacher_after.state_dict()[k], before)

    loaded_full, ck_full = load_net(str(run_full / "ckpt_final.pt"), device="cpu")
    assert ck_full.cfg.ruleset == "v2" and ck_full.cfg.arch == "v2"
    for k, v in net_full.state_dict().items():
        assert torch.equal(v, loaded_full.state_dict()[k])

    # Mid-decay exact resume from ckpt_16.pt reproduces bit-identical weights & metrics rows.
    run_resumed = tmp_path / "distill_resumed"
    net_resumed = train.run(cfg, str(run_resumed), resume=str(run_full / "ckpt_16.pt"))
    rows_resumed = _metrics(run_resumed)

    assert [r["update"] for r in rows_resumed] == [2, 3]
    assert [_untimed(r) for r in rows_resumed] == [_untimed(r) for r in rows_full[1:]]
    for k, v in net_full.state_dict().items():
        assert torch.equal(v, net_resumed.state_dict()[k]), f"resume weight drift at {k}"
