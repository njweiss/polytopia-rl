"""Milestone 23 Phase A tests: Revised Self-Play Environment (PFSP, live two-sided self-play,
barrier-free stream rollouts, pinned cross-arch anchors, win-rate gating, shaping annealing &
paired seeds).

Covers:
1. `pfsp.py` unit tests: `parse_bot_specs`, `pfsp_weights`, `WinRateTracker`, `shaping_schedule`.
2. `compute_gae_segments`: lane-major equivalence to `compute_gae`, hand-computed multi-segment
   cases (terminated, truncated, ongoing end, implicit final segment_end), and empty `(0,)` input.
3. `PPOConfig` M23 fields, validation errors, and shipped `v2_selfplay_revised.json` /
   `selfplay23.json` (while confirming `selfplay.json` and `v2_selfplay.json` stay untouched).
4. `SnapshotPool` pinned anchors (immune to `thin()`, ignored by `latest_snapshot()`), legacy
   `draw_opponent` compatibility, and PFSP sampling frequencies (5-sigma bounds).
5. `SelfPlayVecEnv` mechanics: live two-sided `"self"` hand-verified trace with zero opponent
   calls, `paired_seeds` FIFO mirror queue + mid-queue `state_dict()` round-trip, Python-side M9
   `float32` bot shaping (`set_shaping_weight`), and `opponent_forward="auto"` with cross-arch
   pinned anchors.
6. Barrier-free `collect_rollout_stream`: exact `batch` transitions per rollout, zero drain ticks,
   independent carryover/pending/current-obs bootstrap reference, `StreamState` serialization, and
   `train_update` compatibility (plain PPO + M22 distillation).
7. End-to-end CPU `train.run` (revised self-play): cross-arch anchor + `SpecMismatch`, win-rate
   gating + `snapshot_max_steps`, shaping annealing, `metrics.jsonl` fields, non-regressed vs
   regressed `best.pt`, `tournament.json` + `tournament.main` with `anchor_0.pt`, exact mid-run
   bit-for-bit resume, and unchanged v1 `rules_hash` / `obs_hash(11)`.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import pathlib
import shutil

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from il_helpers import KeyedVec, StubNet, learner_path, pool_of, reward_cfg, stub_logp, stub_value
from rl_helpers import END_TURN, MOVE
from sp_helpers import REPO, FakeTwoSeatEnv, fake_scorer

from polytopia_rl import experiment, tournament, train
from polytopia_rl.checkpoint import SpecMismatch, save_checkpoint
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.gae import compute_gae, compute_gae_segments
from polytopia_rl.model import PolicyNet
from polytopia_rl.pfsp import BotSpec, WinRateTracker, parse_bot_specs, pfsp_weights, shaping_schedule
from polytopia_rl.ppo import RolloutBuffer, StreamState, collect_rollout_stream, train_update
from polytopia_rl.selfplay import (
    LIVE_SELF,
    Entry,
    SelfPlayVecEnv,
    SnapshotPool,
    draw_opponent,
    make_selfplay_vec_env,
    net_batch_policy,
)
from polytopia_rl.stacked import StackedSnapshots, stacked_batch_policy

f32 = np.float32
FAKE_G = 3 * 16
FAKE_REWARD = reward_cfg(shaping=1.0, gamma=0.9, scale=10.0)


def _m9_seat(before, after, seat: int, *, terminal: float = 0.0, terminated: bool = False, reward=FAKE_REWARD) -> np.float32:
    scale = f32(reward["potential_scale"])
    gamma = f32(reward["gamma"])
    tw = f32(reward["terminal_weight"])
    sw = f32(reward["shaping_weight"])
    phi_before = (f32(before[seat]) - f32(before[1 - seat])) / scale
    phi_after = f32(0.0) if terminated else (f32(after[seat]) - f32(after[1 - seat])) / scale
    return tw * f32(terminal) + sw * (gamma * phi_after - phi_before)


def _ref_gae_segments(rewards, values, terminated, truncated, next_values, segment_end, gamma, lam):
    """Independent NumPy float64 reference for `compute_gae_segments`."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    a_next = 0.0
    for i in range(n - 1, -1, -1):
        is_end = bool(segment_end[i]) or (i == n - 1)
        v_next = float(next_values[i]) if (bool(truncated[i]) or is_end) else float(values[i + 1])
        not_term = 0.0 if bool(terminated[i]) else 1.0
        delta = float(rewards[i]) + gamma * v_next * not_term - float(values[i])
        cont = 0.0 if (bool(terminated[i]) or bool(truncated[i]) or is_end) else 1.0
        a_next = delta + gamma * lam * cont * a_next
        adv[i] = a_next
    return adv, adv + np.asarray(values, dtype=np.float64)


def _spec(size: int = 6, ruleset: str = "v2") -> dict:
    return dict(ps.Env(size=size, ruleset=ruleset).spec())


def _tiny_net(spec: dict, seed: int = 0, arch: str = "v2", width: int = 8, blocks: int = 1) -> PolicyNet:
    torch.manual_seed(seed)
    return PolicyNet(spec, width=width, blocks=blocks, arch=arch)


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _untimed(row: dict) -> dict:
    return {k: v for k, v in row.items() if "sps" not in k and "time" not in k}


def _same_state(a: dict, b: dict) -> bool:
    assert a.keys() == b.keys()
    return all(torch.equal(a[k], b[k]) for k in a)


def _save_anchor(path: pathlib.Path, *, size: int = 6, ruleset: str = "v2", arch: str = "v1", seed: int = 77, obs_hash: str | None = None) -> PolicyNet:
    sp = _spec(size=size, ruleset=ruleset)
    net = _tiny_net(sp, seed=seed, arch=arch, width=8, blocks=1)
    cfg = PPOConfig(size=size, turn_cap=3, ruleset=ruleset, arch=arch, width=8, blocks=1, device="cpu")
    if obs_hash is not None:
        sp = dict(sp)
        sp["obs_hash"] = obs_hash
    save_checkpoint(
        str(path),
        model=net,
        optimizer=None,
        cfg=cfg,
        global_step=0,
        stage=0,
        seed_index=None,
        rng_state=None,
        spec=sp,
    )
    return net


# ---------------------------------------------------------------- 1. pfsp.py


def test_parse_bot_specs_defaults_and_custom():
    default = parse_bot_specs(None)
    assert default == [
        BotSpec("economy", "economy", 0.0, 1.0),
        BotSpec("rush", "rush", 0.0, 1.0),
    ]
    with_eps = parse_bot_specs(None, default_epsilon=0.25)
    assert with_eps == [
        BotSpec("economy", "economy", 0.25, 1.0),
        BotSpec("rush", "rush", 0.25, 1.0),
    ]

    custom = parse_bot_specs(
        [
            {"name": "rush_e0", "bot": "rush"},
            {"name": "rush_e20", "bot": "rush", "epsilon": 0.2, "weight": 2.5},
            {"name": "economy_e0", "bot": "economy", "epsilon": 0.0, "weight": 0.5},
            {"name": "rand_e100", "bot": "random", "epsilon": 1.0},
        ]
    )
    assert custom == [
        BotSpec("rush_e0", "rush", 0.0, 1.0),
        BotSpec("rush_e20", "rush", 0.2, 2.5),
        BotSpec("economy_e0", "economy", 0.0, 0.5),
        BotSpec("rand_e100", "random", 1.0, 1.0),
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        custom[0].weight = 3.0


@pytest.mark.parametrize(
    "raw",
    [
        [],
        "rush",
        ["rush"],
        [{}],
        [{"name": "only_name"}],
        [{"bot": "rush"}],
        [{"name": "", "bot": "rush"}],
        [{"name": 123, "bot": "rush"}],
        [{"name": "self", "bot": "rush"}],
        [{"name": "snap_0", "bot": "rush"}],
        [{"name": "snap_custom", "bot": "economy"}],
        [{"name": "anchor_0", "bot": "rush"}],
        [{"name": "anchor_v1", "bot": "economy"}],
        [{"name": "dup", "bot": "rush"}, {"name": "dup", "bot": "economy"}],
        [{"name": "bad_bot", "bot": "greedy"}],
        [{"name": "bad_bot", "bot": "none"}],
        [{"name": "bad_key", "bot": "rush", "extra": 1}],
        [{"name": "neg_eps", "bot": "rush", "epsilon": -0.01}],
        [{"name": "big_eps", "bot": "rush", "epsilon": 1.01}],
        [{"name": "zero_w", "bot": "rush", "weight": 0.0}],
        [{"name": "neg_w", "bot": "rush", "weight": -1.0}],
    ],
)
def test_parse_bot_specs_validation_errors(raw):
    with pytest.raises(ValueError):
        parse_bot_specs(raw)


def test_pfsp_weights_exact_hand_calculations():
    # "off": proportional to base_weights (or uniform when None).
    w_off = pfsp_weights([0.1, 0.5, 0.9], "off")
    assert w_off.dtype == np.float64 and w_off.shape == (3,)
    np.testing.assert_allclose(w_off, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)

    w_off_base = pfsp_weights([0.1, 0.5, 0.9], "off", base_weights=[1.0, 2.0, 5.0])
    np.testing.assert_allclose(w_off_base, [0.125, 0.25, 0.625], atol=1e-12)

    # "hard": b_i * ((1 - w_i)^power + eps), normalized.
    # For w = [0.0, 0.5, 1.0], power=2.0, eps=1e-3: raw = [1.001, 0.251, 0.001], sum = 1.253.
    w_hard = pfsp_weights([0.0, 0.5, 1.0], "hard", power=2.0, eps=1e-3)
    assert w_hard.dtype == np.float64 and w_hard.shape == (3,)
    np.testing.assert_allclose(w_hard, np.array([1.001, 0.251, 0.001]) / 1.253, atol=1e-12)
    assert math.isclose(float(w_hard.sum()), 1.0, abs_tol=1e-12)

    # "hard" with base_weights and eps=0.0: [1.0 * 0.8, 3.0 * 0.4] = [0.8, 1.2] -> [0.4, 0.6].
    w_hard_base = pfsp_weights([0.2, 0.6], "hard", power=1.0, eps=0.0, base_weights=[1.0, 3.0])
    np.testing.assert_allclose(w_hard_base, [0.4, 0.6], atol=1e-12)

    # "variance": b_i * ((w_i * (1 - w_i))^power + eps), normalized.
    # For w = [0.0, 0.25, 0.5, 1.0], power=2.0, eps=1e-3:
    # w*(1-w) = [0, 3/16, 1/4, 0] -> sq + 1e-3 = [0.001, 0.03615625, 0.0635, 0.001], sum = 0.10165625.
    w_var = pfsp_weights([0.0, 0.25, 0.5, 1.0], "variance", power=2.0, eps=1e-3)
    want_var = np.array([0.001, (3 / 16) ** 2 + 1e-3, 0.25**2 + 1e-3, 0.001], dtype=np.float64)
    want_var /= want_var.sum()
    np.testing.assert_allclose(w_var, want_var, atol=1e-12)
    assert int(np.argmax(w_var)) == 2

    # Out-of-range win rates are clamped to [0, 1].
    w_clamped = pfsp_weights([-0.5, 1.5], "hard", power=2.0, eps=1e-3)
    w_bounds = pfsp_weights([0.0, 1.0], "hard", power=2.0, eps=1e-3)
    np.testing.assert_allclose(w_clamped, w_bounds, atol=1e-12)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (([], "hard"), {}),
        (([0.5], "linear"), {}),
        (([0.5], ""), {}),
        (([0.5], "hard"), {"power": 0.0}),
        (([0.5], "hard"), {"power": -2.0}),
        (([0.5], "hard"), {"eps": -1e-4}),
        (([0.5, 0.5], "hard"), {"base_weights": [1.0]}),
        (([0.5, 0.5], "hard"), {"base_weights": [1.0, 0.0]}),
        (([0.5, 0.5], "hard"), {"base_weights": [-1.0, 1.0]}),
    ],
)
def test_pfsp_weights_validation_errors(args, kwargs):
    with pytest.raises(ValueError):
        pfsp_weights(*args, **kwargs)


def test_win_rate_tracker_warmup_ema_and_serialization():
    tr = WinRateTracker(alpha=0.25, prior=0.4)
    assert tr.win_rate("rush_e0") == pytest.approx(0.4)
    assert tr.games("rush_e0") == 0

    # Warmup regime (1/n >= 0.25 for n=1..4 gives exact running mean), then EMA (eta=0.25 for n>=5).
    expected = [
        ("win", 1.0),
        ("loss", 0.5),
        ("draw", 0.5),
        ("win", 0.625),
        ("loss", 0.46875),
        ("win", 0.6015625),
    ]
    for idx, (outcome, want) in enumerate(expected, start=1):
        got = tr.record("rush_e0", outcome)
        assert got == pytest.approx(want, abs=1e-12)
        assert tr.win_rate("rush_e0") == pytest.approx(want, abs=1e-12)
        assert tr.games("rush_e0") == idx

    # Independent opponent tracking.
    assert tr.record("anchor_0", "draw") == pytest.approx(0.5)
    assert tr.games("anchor_0") == 1
    assert tr.win_rate("rush_e0") == pytest.approx(0.6015625)

    # state_dict round-trip (sorted keys, exact continuation).
    sd = tr.state_dict()
    assert sd["alpha"] == pytest.approx(0.25)
    assert sd["prior"] == pytest.approx(0.4)
    # Check deterministic sorted opponent ordering in serialized mapping.
    opp_map = next(v for v in sd.values() if isinstance(v, dict))
    assert list(opp_map.keys()) == sorted(opp_map.keys())

    tr2 = WinRateTracker(alpha=0.05, prior=0.5)
    tr2.load_state_dict(copy.deepcopy(sd))
    assert tr2.win_rate("unseen") == pytest.approx(0.4)
    assert tr2.games("unseen") == 0
    assert tr2.win_rate("rush_e0") == pytest.approx(0.6015625)
    assert tr2.games("rush_e0") == 6
    assert tr2.win_rate("anchor_0") == pytest.approx(0.5)
    assert tr2.games("anchor_0") == 1
    assert tr2.record("rush_e0", "loss") == pytest.approx(tr.record("rush_e0", "loss"), abs=1e-12)


def test_win_rate_tracker_validation_errors():
    for bad_alpha in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError):
            WinRateTracker(alpha=bad_alpha)
    for bad_prior in (-0.01, 1.01):
        with pytest.raises(ValueError):
            WinRateTracker(prior=bad_prior)
    tr = WinRateTracker()
    for bad_outcome in ("tie", "", "win_0", "WIN"):
        with pytest.raises(ValueError):
            tr.record("rush", bad_outcome)


def test_shaping_schedule_decay_clamp_none_and_errors():
    assert shaping_schedule(1, 1.0, 0.0, None) == pytest.approx(1.0)
    assert shaping_schedule(500, 0.75, 0.0, None) == pytest.approx(0.75)

    # Linear decay from 1.0 to 0.2 over 4 updates: frac = clamp((update - 1) / 4, 0, 1).
    expected = [(0, 1.0), (1, 1.0), (2, 0.8), (3, 0.6), (4, 0.4), (5, 0.2), (9, 0.2)]
    for u, want in expected:
        assert shaping_schedule(u, 1.0, 0.2, 4) == pytest.approx(want, abs=1e-12)

    # Increasing schedule works too.
    assert shaping_schedule(3, 0.0, 1.0, 4) == pytest.approx(0.5, abs=1e-12)

    for bad in (0, -1, -100):
        with pytest.raises(ValueError):
            shaping_schedule(1, 1.0, 0.0, bad)


# ---------------------------------------------------------------- 2. compute_gae_segments


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_gae_segments_matches_compute_gae_lane_major(seed):
    rng = np.random.default_rng(seed)
    T, E = 17, 5
    r = rng.normal(size=(T, E)).astype(np.float32)
    v = rng.normal(size=(T, E)).astype(np.float32)
    nv = rng.normal(size=(T, E)).astype(np.float32)
    last = rng.normal(size=E).astype(np.float32)
    u = rng.random((T, E))
    term = u < 0.15
    trunc = (u >= 0.15) & (u < 0.25)
    gamma, lam = 0.97, 0.9

    adv_2d, ret_2d = compute_gae(
        torch.from_numpy(r),
        torch.from_numpy(v),
        torch.from_numpy(term),
        torch.from_numpy(trunc),
        torch.from_numpy(nv),
        torch.from_numpy(last),
        gamma,
        lam,
    )

    # Flatten in lane-major (E, T) order; set segment_end at each lane's final step and populate
    # next_values with last_value on non-truncated segment ends.
    seg_end = np.zeros((E, T), dtype=bool)
    seg_end[:, -1] = True
    nv_lm = nv.T.copy()
    trunc_lm = trunc.T
    for e in range(E):
        if not trunc_lm[e, -1]:
            nv_lm[e, -1] = last[e]

    adv_1d, ret_1d = compute_gae_segments(
        torch.from_numpy(r.T.reshape(-1)),
        torch.from_numpy(v.T.reshape(-1)),
        torch.from_numpy(term.T.reshape(-1)),
        torch.from_numpy(trunc_lm.reshape(-1)),
        torch.from_numpy(nv_lm.reshape(-1)),
        torch.from_numpy(seg_end.reshape(-1)),
        gamma,
        lam,
    )
    assert adv_1d.shape == (T * E,) and ret_1d.shape == (T * E,)
    np.testing.assert_allclose(adv_1d.reshape(E, T).T.numpy(), adv_2d.numpy(), atol=1e-6)
    np.testing.assert_allclose(ret_1d.reshape(E, T).T.numpy(), ret_2d.numpy(), atol=1e-6)


def test_compute_gae_segments_hand_cases_and_implicit_last_end():
    gamma, lam = 0.9, 0.8
    # Segments: [0..2] (ongoing end bootstrap 0.4), [3..4] (term at 3, trunc at 4 w/ 1.5),
    # [5] (single ongoing step with segment_end[-1]=False explicitly -> treated as True).
    rewards = torch.tensor([1.0, 0.0, 2.0, 3.0, -1.0, 0.5], dtype=torch.float32)
    values = torch.tensor([0.5, 0.2, 0.1, 0.5, 0.4, 0.25], dtype=torch.float32)
    terminated = torch.tensor([False, False, False, True, False, False], dtype=torch.bool)
    truncated = torch.tensor([False, False, False, False, True, False], dtype=torch.bool)
    next_values = torch.tensor([99.0, 99.0, 0.4, 7.0, 1.5, 2.0], dtype=torch.float32)
    segment_end = torch.tensor([False, False, True, False, True, False], dtype=torch.bool)

    adv, ret = compute_gae_segments(rewards, values, terminated, truncated, next_values, segment_end, gamma, lam)
    want_adv = [1.772384, 1.5172, 2.26, 2.5, -0.05, 2.05]
    assert adv.tolist() == pytest.approx(want_adv, abs=1e-5)
    assert ret.tolist() == pytest.approx([a + v for a, v in zip(want_adv, values.tolist())], abs=1e-5)

    # Terminated AT segment_end ignores next_values.
    term_end = torch.tensor([False, False, True, True, False, True], dtype=torch.bool)
    adv2, _ = compute_gae_segments(rewards, values, term_end, truncated, torch.full((6,), 100.0), segment_end, gamma, lam)
    assert adv2[2].item() == pytest.approx(1.9, abs=1e-6)
    assert adv2[5].item() == pytest.approx(0.25, abs=1e-6)

    # Random variable-length segments against independent NumPy float64 reference.
    rng = np.random.default_rng(42)
    n = 37
    r = rng.normal(size=n).astype(np.float32)
    v = rng.normal(size=n).astype(np.float32)
    nv = rng.normal(size=n).astype(np.float32)
    u = rng.random(n)
    term = u < 0.15
    trunc = (u >= 0.15) & (u < 0.30)
    seg = rng.random(n) < 0.25
    adv3, ret3 = compute_gae_segments(*(torch.from_numpy(x) for x in (r, v, term, trunc, nv, seg)), 0.95, 0.9)
    ref_adv, ref_ret = _ref_gae_segments(r, v, term, trunc, nv, seg, 0.95, 0.9)
    np.testing.assert_allclose(adv3.numpy(), ref_adv, atol=1e-5)
    np.testing.assert_allclose(ret3.numpy(), ref_ret, atol=1e-5)


def test_compute_gae_segments_empty_input():
    for dtype in (torch.float32, torch.float64):
        empty_f = torch.empty(0, dtype=dtype)
        empty_b = torch.empty(0, dtype=torch.bool)
        adv, ret = compute_gae_segments(empty_f, empty_f, empty_b, empty_b, empty_f, empty_b, 0.99, 0.95)
        assert adv.shape == (0,) and ret.shape == (0,)
        assert adv.dtype == dtype and ret.dtype == dtype
        assert adv.device == empty_f.device and ret.device == empty_f.device


# ---------------------------------------------------------------- 3. PPOConfig & shipped configs


def test_ppo_config_m23_defaults_and_round_trip():
    default = PPOConfig()
    assert default.p_self == 0.0
    assert default.pfsp_mode == "off"
    assert default.pfsp_power == 2.0
    assert default.pfsp_ema_alpha == 0.05
    assert default.selfplay_bots is None
    assert default.anchor_checkpoints == []
    assert default.snapshot_win_rate is None
    assert default.snapshot_min_games == 20
    assert default.snapshot_max_steps is None
    assert default.shaping_anneal_steps is None
    assert default.shaping_end == 0.0
    assert default.paired_seeds is False

    # Legacy JSON without any M23 keys restores to exact defaults.
    legacy = json.loads(default.to_json())
    for k in (
        "p_self",
        "pfsp_mode",
        "pfsp_power",
        "pfsp_ema_alpha",
        "selfplay_bots",
        "anchor_checkpoints",
        "snapshot_win_rate",
        "snapshot_min_games",
        "snapshot_max_steps",
        "shaping_anneal_steps",
        "shaping_end",
        "paired_seeds",
    ):
        legacy.pop(k, None)
    assert PPOConfig.from_json(json.dumps(legacy)) == default

    # Non-default M23 config round-trips cleanly.
    revised = PPOConfig(
        ruleset="v2",
        arch="v2",
        selfplay=True,
        rollout="stream",
        opponent_forward="auto",
        p_bot=0.25,
        p_self=0.35,
        p_latest=0.25,
        pfsp_mode="hard",
        pfsp_power=2.5,
        pfsp_ema_alpha=0.1,
        selfplay_bots=[
            {"name": "rush_e0", "bot": "rush", "epsilon": 0.0},
            {"name": "rush_e20", "bot": "rush", "epsilon": 0.2, "weight": 0.5},
        ],
        anchor_checkpoints=["runs/anchor_v1.pt"],
        snapshot_every=500_000,
        snapshot_win_rate=0.55,
        snapshot_min_games=15,
        snapshot_max_steps=2_000_000,
        shaping_anneal_steps=5_000_000,
        shaping_end=0.1,
        paired_seeds=True,
    )
    assert PPOConfig.from_json(revised.to_json()) == revised


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"rollout": "bad"},
        {"opponent_forward": "bad"},
        {"p_bot": -0.01},
        {"p_bot": 1.01},
        {"p_self": -0.01, "rollout": "stream"},
        {"p_self": 1.01, "rollout": "stream"},
        {"p_bot": 0.6, "p_self": 0.5, "rollout": "stream"},
        {"p_latest": -0.01},
        {"p_latest": 1.01},
        {"p_self": 0.2, "rollout": "interleaved"},
        {"p_self": 0.2, "rollout": "lockstep"},
        {"pfsp_mode": "linear"},
        {"pfsp_power": 0.0},
        {"pfsp_power": -1.0},
        {"pfsp_ema_alpha": 0.0},
        {"pfsp_ema_alpha": -0.05},
        {"pfsp_ema_alpha": 1.01},
        {"selfplay_bots": []},
        {"selfplay_bots": [{"name": "self", "bot": "rush"}]},
        {"selfplay_bots": [{"name": "x", "bot": "invalid"}]},
        {"anchor_checkpoints": [""]},
        {"anchor_checkpoints": [123]},
        {"anchor_checkpoints": "not_a_list.pt"},
        {"snapshot_win_rate": -0.01},
        {"snapshot_win_rate": 1.01},
        {"snapshot_min_games": 0},
        {"snapshot_min_games": -5},
        {"snapshot_every": 1000, "snapshot_max_steps": 500},
        {"shaping_anneal_steps": 0},
        {"shaping_anneal_steps": -100},
        {"shaping_end": -0.01},
    ],
)
def test_ppo_config_m23_validation_errors(bad_fields):
    data = json.loads(PPOConfig().to_json())
    data.update({"rollout": "stream", "opponent_forward": "auto", "p_self": 0.0})
    data.update(bad_fields)
    with pytest.raises(ValueError) as exc:
        PPOConfig.from_json(json.dumps(data))
    assert "unknown config keys" not in str(exc.value)


def test_shipped_m23_configs_and_legacy_configs_untouched():
    revised_path = REPO / "python" / "configs" / "v2_selfplay_revised.json"
    exp_path = REPO / "python" / "experiments" / "selfplay23.json"
    assert revised_path.exists(), f"missing {revised_path}"
    assert exp_path.exists(), f"missing {exp_path}"

    revised = load_config(str(revised_path))
    assert revised.ruleset == "v2"
    assert revised.arch == "v2"
    assert revised.selfplay is True
    assert revised.rollout == "stream"
    assert revised.opponent_forward == "auto"
    assert revised.total_steps == 20_000_000
    assert revised.num_envs == 128 and revised.num_steps == 32
    assert revised.lr == pytest.approx(1e-4)
    assert revised.p_bot == pytest.approx(0.25)
    assert revised.p_self == pytest.approx(0.35)
    assert revised.p_latest == pytest.approx(0.25)
    assert revised.pfsp_mode == "hard"
    assert revised.pfsp_power == pytest.approx(2.0)
    assert revised.pfsp_ema_alpha == pytest.approx(0.05)
    assert parse_bot_specs(revised.selfplay_bots) == [
        BotSpec("rush_e0", "rush", 0.0, 1.0),
        BotSpec("rush_e20", "rush", 0.2, 1.0),
        BotSpec("economy_e0", "economy", 0.0, 1.0),
    ]
    assert revised.snapshot_every == 500_000
    assert revised.snapshot_win_rate == pytest.approx(0.55)
    assert revised.snapshot_min_games == 20
    assert revised.snapshot_max_steps == 2_000_000
    assert revised.pool_max == 40
    assert revised.opponent_greedy is False
    assert revised.shaping_anneal_steps == 5_000_000
    assert revised.shaping_end == pytest.approx(0.0)
    assert revised.paired_seeds is True

    spec = experiment.load_spec(str(exp_path))
    assert list(spec.arms.keys()) == ["legacy", "revised"]
    assert spec.seeds == [1, 2, 3]
    assert spec.total_steps == 10_000_000
    assert spec.primary == ["rush_e0", "elo_vs_init"]
    assert spec.init_checkpoint == "runs/exp/transfer22/distill_v2/seed1/ckpt_final.pt"

    base_path = REPO / spec.base_config
    assert base_path.exists(), f"missing base_config {base_path}"
    arm_cfgs = {}
    for arm, overrides in spec.arms.items():
        cli = [f"total_steps={spec.total_steps}"] + [f"{k}={json.dumps(v)}" for k, v in overrides.items()]
        arm_cfgs[arm] = load_config(str(base_path), cli)
    assert arm_cfgs["legacy"].rollout == "interleaved"
    assert arm_cfgs["legacy"].p_self == 0.0
    assert arm_cfgs["legacy"].pfsp_mode == "off"
    assert arm_cfgs["legacy"].paired_seeds is False
    assert arm_cfgs["legacy"].shaping_anneal_steps is None
    assert arm_cfgs["revised"].rollout == "stream"
    assert arm_cfgs["revised"].opponent_forward == "auto"
    assert arm_cfgs["revised"].p_self > 0.0
    assert arm_cfgs["revised"].pfsp_mode in ("hard", "variance")
    assert arm_cfgs["revised"].paired_seeds is True
    assert arm_cfgs["revised"].shaping_anneal_steps is not None
    assert arm_cfgs["revised"].snapshot_win_rate is not None

    # Confirm legacy shipped self-play configs are completely untouched.
    for rel, ruleset, arch in (
        ("python/configs/selfplay.json", "v1", "v1"),
        ("python/configs/v2_selfplay.json", "v2", "v2"),
    ):
        cfg = load_config(str(REPO / rel))
        assert cfg.ruleset == ruleset and cfg.arch == arch
        assert cfg.rollout == "interleaved"
        assert cfg.opponent_forward == "per_snapshot"
        assert cfg.p_bot == pytest.approx(0.4)
        assert cfg.p_self == 0.0
        assert cfg.p_latest == pytest.approx(0.5)
        assert cfg.pfsp_mode == "off"
        assert cfg.selfplay_bots is None
        assert cfg.snapshot_win_rate is None
        assert cfg.shaping_anneal_steps is None
        assert cfg.paired_seeds is False


# ---------------------------------------------------------------- 4. Pool, PFSP & anchors


def _dummy_state(val: float) -> dict:
    return {"w": torch.full((2,), float(val))}


def test_snapshot_pool_pinned_anchors_immune_to_thin_and_latest_snapshot():
    pool = SnapshotPool(pool_max=4, p_latest=0.5)
    assert pool.latest_snapshot() is None

    sentinel_tpl = object()
    a0 = pool.add(_dummy_state(-1), 0, pinned=True, name="anchor_0", net_template=sentinel_tpl)
    a1 = pool.add(_dummy_state(-2), 0, pinned=True, name="anchor_1", net_template=sentinel_tpl)
    assert a0.pinned and a0.name == "anchor_0" and a0.net_template is sentinel_tpl
    assert pool.latest_snapshot() is a1  # all entries pinned -> entries[-1]

    for s in range(20):
        pool.add(_dummy_state(s), s)
        pinned = [e for e in pool.entries if e.pinned]
        unpinned = [e for e in pool.entries if not e.pinned]
        assert [e.name for e in pinned] == ["anchor_0", "anchor_1"]
        assert len(unpinned) <= 4
        assert unpinned[0].step == 0 and unpinned[-1].step == s
        assert pool.latest_snapshot() is unpinned[-1]

    # Even if a pinned entry is added AFTER unpinned snapshots, latest_snapshot() returns newest unpinned.
    pool.add(_dummy_state(-3), 0, pinned=True, name="anchor_2")
    assert pool.latest_snapshot().name == "snap_19"
    pool.thin()
    assert [e.name for e in pool.entries if e.pinned] == ["anchor_0", "anchor_1", "anchor_2"]


def test_draw_opponent_pfsp_frequencies_and_legacy_compatibility():
    # Legacy compatibility: identical draw sequence to M11 when M23 options are at defaults.
    pool_leg = SnapshotPool(pool_max=10, p_latest=0.6)
    for s in range(4):
        pool_leg.add(_dummy_state(s), s)
    rng_actual = np.random.default_rng(123)
    rng_ref = np.random.default_rng(123)
    for _ in range(100):
        got = draw_opponent(pool_leg, rng_actual, p_bot=0.3)
        if rng_ref.random() < 0.3:
            want = ("economy", "rush")[rng_ref.integers(2)]
        elif rng_ref.random() < 0.6:
            want = pool_leg.entries[-1]
        else:
            want = pool_leg.entries[rng_ref.integers(len(pool_leg.entries))]
        assert got == want

    # Revised PFSP "hard" mode with weighted bots + live "self" + pinned anchor + snapshots.
    pool = SnapshotPool(pool_max=10, p_latest=0.25)
    pool.add(_dummy_state(-1), 0, pinned=True, name="anchor_0")
    pool.add(_dummy_state(0), 0)
    pool.add(_dummy_state(100), 100)
    bots = parse_bot_specs(
        [
            {"name": "rush_e0", "bot": "rush", "epsilon": 0.0, "weight": 2.0},
            {"name": "rush_e20", "bot": "rush", "epsilon": 0.2, "weight": 1.0},
            {"name": "economy_e0", "bot": "economy", "epsilon": 0.0, "weight": 1.0},
        ]
    )
    tracker = WinRateTracker(alpha=0.05, prior=0.5)
    # rush_e0: 1 win, 4 losses -> warmup exact mean 0.2
    tracker.record("rush_e0", "win")
    for _ in range(4):
        tracker.record("rush_e0", "loss")
    # rush_e20: 4 wins, 1 loss -> warmup exact mean 0.8
    for _ in range(4):
        tracker.record("rush_e20", "win")
    tracker.record("rush_e20", "loss")
    # economy_e0: unseen -> 0.5
    # anchor_0: 1 loss -> 0.0; snap_0: 1 win -> 1.0; snap_100 (latest): unseen -> 0.5
    tracker.record("anchor_0", "loss")
    tracker.record("snap_0", "win")

    # Hand-computed exact expected probabilities (N = 60_000):
    # Bot raw (hard, power=2, eps=1e-3): rush_e0=2*(0.8^2+1e-3)=1.282, rush_e20=0.041, economy_e0=0.251; sum=1.574
    # Pool raw: anchor_0=1.001, snap_0=0.001, snap_100=0.251; sum=1.253
    # p_bot=0.25, p_self=0.35, p_pool=0.40, p_latest=0.25 (latest is snap_100, NOT anchor_0).
    want_hard = {
        "rush_e0": 0.25 * (1.282 / 1.574),
        "rush_e20": 0.25 * (0.041 / 1.574),
        "economy_e0": 0.25 * (0.251 / 1.574),
        LIVE_SELF: 0.35,
        "anchor_0": 0.40 * 0.75 * (1.001 / 1.253),
        "snap_0": 0.40 * 0.75 * (0.001 / 1.253),
        "snap_100": 0.40 * (0.25 + 0.75 * (0.251 / 1.253)),
    }
    n_draws = 60_000
    rng = np.random.default_rng(7)
    counts = {k: 0 for k in want_hard}
    for _ in range(n_draws):
        o = draw_opponent(
            pool,
            rng,
            p_bot=0.25,
            p_self=0.35,
            bots=bots,
            tracker=tracker,
            pfsp_mode="hard",
            pfsp_power=2.0,
        )
        if o == LIVE_SELF:
            counts[LIVE_SELF] += 1
        elif isinstance(o, BotSpec):
            counts[o.name] += 1
        else:
            assert isinstance(o, Entry)
            counts[o.name] += 1
    for k, p in want_hard.items():
        emp = counts[k] / n_draws
        sd = math.sqrt(p * (1.0 - p) / n_draws)
        # 5-sigma bound (+ 5e-4 floor for near-zero probability snap_0)
        assert abs(emp - p) <= 5.0 * sd + 5e-4, (k, emp, p, sd)

    # PFSP "variance" mode: peaks at w=0.5 (snap_100) over w=0.0 (anchor_0) and w=1.0 (snap_0).
    # With p_bot=0, p_self=0, p_latest=0.25:
    # raw = [0.001, 0.001, 0.0635], sum = 0.0655.
    want_var = {
        "anchor_0": 0.75 * (0.001 / 0.0655),
        "snap_0": 0.75 * (0.001 / 0.0655),
        "snap_100": 0.25 + 0.75 * (0.0635 / 0.0655),
    }
    rng_v = np.random.default_rng(9)
    vcounts = {k: 0 for k in want_var}
    for _ in range(n_draws):
        e = draw_opponent(pool, rng_v, p_bot=0.0, p_self=0.0, tracker=tracker, pfsp_mode="variance", pfsp_power=2.0)
        vcounts[e.name] += 1
    for k, p in want_var.items():
        emp = vcounts[k] / n_draws
        sd = math.sqrt(p * (1.0 - p) / n_draws)
        assert abs(emp - p) <= 5.0 * sd + 5e-4, (k, emp, p, sd)


# ---------------------------------------------------------------- 5. SelfPlayVecEnv mechanics


def test_live_self_two_seat_hand_trace_and_zero_opponent_calls():
    pool = pool_of(1)

    def forbidden_policy(entry):
        raise AssertionError("live self episodes must never invoke opponent make_policy")

    vec = SelfPlayVecEnv(
        lambda opp: FakeTwoSeatEnv(k=3, turns=2),
        num_envs=1,
        seed_base=0,
        pool=pool,
        make_policy=forbidden_policy,
        rng=np.random.default_rng(0),
        p_bot=0.0,
        p_self=1.0,
        reward=FAKE_REWARD,
    )
    vec.reset_all()

    # Scripted 12-action game: seat 0 scores +2 per turn, seat 1 scores +1 per turn -> seat 0 wins 4-2.
    script = [
        # Turn 0, seat 0
        (0, (0, MOVE, 1), [(0, _m9_seat((0, 0), (1, 0), 0), False, False)]),
        (0, (0, MOVE, 1), [(0, _m9_seat((1, 0), (2, 0), 0), False, False)]),
        (0, (FAKE_G, END_TURN, 0), []),
        # Turn 0, seat 1
        (1, (0, MOVE, 1), [(1, _m9_seat((2, 0), (2, 1), 1), False, False)]),
        (1, (0, MOVE, 0), [(1, _m9_seat((2, 1), (2, 1), 1), False, False)]),
        (1, (FAKE_G, END_TURN, 0), [(0, _m9_seat((2, 0), (2, 1), 0), False, False)]),
        # Turn 1, seat 0
        (0, (0, MOVE, 1), [(0, _m9_seat((2, 1), (3, 1), 0), False, False)]),
        (0, (0, MOVE, 1), [(0, _m9_seat((3, 1), (4, 1), 0), False, False)]),
        (0, (FAKE_G, END_TURN, 0), [(1, _m9_seat((2, 1), (4, 1), 1), False, False)]),
        # Turn 1, seat 1 -> terminal win_0; emits BOTH seats sorted by (slot, seat).
        (1, (0, MOVE, 1), [(1, _m9_seat((4, 1), (4, 2), 1), False, False)]),
        (1, (0, MOVE, 0), [(1, _m9_seat((4, 2), (4, 2), 1), False, False)]),
        (
            1,
            (FAKE_G, END_TURN, 0),
            [
                (0, _m9_seat((4, 1), (4, 2), 0, terminal=1.0, terminated=True), True, False),
                (1, _m9_seat((4, 2), (4, 2), 1, terminal=-1.0, terminated=True), True, False),
            ],
        ),
    ]

    seat0_ret = 0.0
    for step_idx, (want_seat, action, want_trs) in enumerate(script):
        assert vec.learner_slots() == [0], step_idx
        assert vec.learner_seat(0) == want_seat, step_idx
        trs = vec.tick({0: action})
        assert len(trs) == len(want_trs), (step_idx, trs)
        for tr, (ws, wr, wterm, wtrunc) in zip(trs, want_trs):
            assert tr.slot == 0 and tr.seat == ws, (step_idx, tr)
            assert tr.reward == pytest.approx(float(wr), abs=1e-6), (step_idx, tr.reward, float(wr))
            assert tr.terminated is wterm and tr.truncated is wtrunc, (step_idx, tr)
            if wterm:
                assert tr.final_board is not None and tr.final_globals is not None
            else:
                assert tr.final_board is None and tr.final_globals is None
            if ws == 0:
                seat0_ret += float(wr)

    stats = vec.episode_stats()
    assert len(stats) == 1
    st = stats[0]
    assert st.env_index == 0 and st.seed == 0
    assert st.opponent == LIVE_SELF
    assert st.agent_seat == 0
    assert st.outcome == "win"
    assert st.length == 12
    assert st.ret == pytest.approx(seat0_ret, abs=1e-5)


def test_paired_seeds_mirror_queue_and_mid_queue_state_round_trip():
    # Part A: FakeTwoSeatEnv with snapshots + live "self", verifying FIFO mirror queue & exact state_dict resume.
    pool = pool_of(2)

    def make_vec():
        return SelfPlayVecEnv(
            lambda opp: FakeTwoSeatEnv(k=2, turns=1),
            num_envs=3,
            seed_base=10,
            pool=pool,
            make_policy=lambda entry: fake_scorer,
            rng=np.random.default_rng(17),
            p_bot=0.0,
            p_self=0.35,
            reward=FAKE_REWARD,
            paired_seeds=True,
        )

    vec = make_vec()
    vec.reset_all()
    all_stats = []
    saved_state = None
    saved_rng = None
    ref_post_stats = []
    ref_post_trs = []

    for tick_idx in range(120):
        acts = {i: learner_path(vec.envs[i]) for i in vec.learner_slots()}
        trs = vec.tick(acts)
        st = vec.episode_stats()
        all_stats.extend(st)
        if tick_idx == 25:
            saved_state = copy.deepcopy(vec.state_dict())
            saved_rng = copy.deepcopy(vec.rng.bit_generator.state)
        elif saved_state is not None:
            ref_post_trs.append([(t.slot, t.seat, t.reward, t.terminated, t.truncated) for t in trs])
            ref_post_stats.extend(st)

    # Restore mid-queue into a fresh SelfPlayVecEnv and verify bit-for-bit identical continuation.
    vec2 = make_vec()
    vec2.rng.bit_generator.state = saved_rng
    vec2.load_state_dict(saved_state)
    got_post_trs = []
    got_post_stats = []
    for _ in range(120 - 26):
        acts = {i: learner_path(vec2.envs[i]) for i in vec2.learner_slots()}
        trs = vec2.tick(acts)
        got_post_trs.append([(t.slot, t.seat, t.reward, t.terminated, t.truncated) for t in trs])
        got_post_stats.extend(vec2.episode_stats())
    assert got_post_trs == ref_post_trs
    assert got_post_stats == ref_post_stats

    # Every completed non-"self" seed with >=2 completions must have played BOTH seat 0 and seat 1
    # against the SAME opponent; "self" seeds must never be duplicated.
    by_seed: dict[int, list] = {}
    for s in all_stats:
        by_seed.setdefault(s.seed, []).append((s.opponent, s.agent_seat))
    non_self_paired = 0
    self_seen = 0
    for seed, plays in by_seed.items():
        if plays[0][0] == LIVE_SELF:
            assert len(plays) == 1, f"live self seed {seed} should not be mirrored: {plays}"
            self_seen += 1
        elif len(plays) == 2:
            assert plays[0][0] == plays[1][0], f"mirror changed opponent on seed {seed}: {plays}"
            assert {plays[0][1], plays[1][1]} == {0, 1}, f"mirror did not swap seats on seed {seed}: {plays}"
            non_self_paired += 1
        else:
            assert len(plays) == 1 and plays[0][1] == 0, (seed, plays)
    assert non_self_paired >= 5 and self_seen >= 2, (non_self_paired, self_seen)

    # Part B: Real polytopia_sim.Env with bot opponents under paired_seeds=True plays both seats per seed.
    cfg = PPOConfig(
        size=6,
        turn_cap=2,
        ruleset="v2",
        num_envs=4,
        p_bot=1.0,
        p_self=0.0,
        paired_seeds=True,
        selfplay_bots=[{"name": "rush_e0", "bot": "rush", "epsilon": 0.0}],
    )
    bot_vec = make_selfplay_vec_env(
        cfg,
        seed_base=0,
        pool=pool_of(1),
        make_policy=lambda e: fake_scorer,
        rng=np.random.default_rng(3),
    )
    bot_vec.reset_all()
    bot_stats = []
    for _ in range(120):
        acts = {i: tuple(int(x) for x in bot_vec.envs[i].bot_action("economy")) for i in bot_vec.learner_slots()}
        bot_vec.tick(acts)
        bot_stats.extend(bot_vec.episode_stats())
        if len(bot_stats) >= 12:
            break
    bot_by_seed: dict[int, set[int]] = {}
    for s in bot_stats:
        assert s.opponent == "rush_e0"
        bot_by_seed.setdefault(s.seed, set()).add(s.agent_seat)
    paired_bot_seeds = [sd for sd, seats in bot_by_seed.items() if seats == {0, 1}]
    assert len(paired_bot_seeds) >= 3, f"expected paired bot seeds with both seats 0 and 1, got {bot_by_seed}"


def test_python_bot_shaping_matches_rust_env_float32():
    cfg = PPOConfig(
        size=6,
        turn_cap=4,
        ruleset="v2",
        num_envs=1,
        p_bot=1.0,
        p_self=0.0,
        shaping_weight=1.0,
        shaping_anneal_steps=100,
        shaping_end=0.0,
        selfplay_bots=[{"name": "rush_e0", "bot": "rush", "epsilon": 0.0}],
    )
    rewards_by_w = {}
    for w in (1.0, 0.5, 0.0):
        vec = make_selfplay_vec_env(
            cfg,
            seed_base=5,
            pool=pool_of(1),
            make_policy=lambda e: fake_scorer,
            rng=np.random.default_rng(0),
        )
        vec.set_shaping_weight(w)
        vec.reset_all()
        ref_env = ps.Env(
            size=6,
            turn_cap=4,
            ruleset="v2",
            opponent="rush",
            opponent_epsilon=0.0,
            agent_seat="seed_parity",
            terminal_weight=cfg.terminal_weight,
            shaping_weight=w,
            gamma=cfg.gamma,
            potential_scale=cfg.potential_scale,
        )
        ref_env.reset(5)
        got_rewards = []
        ref_rewards = []
        for _ in range(200):
            action = tuple(int(x) for x in ref_env.bot_action("economy"))
            trs = vec.tick({0: action})
            ref_r, _, ref_term, ref_trunc = ref_env.step_factored(*action)
            assert len(trs) == 1
            got_rewards.append(np.float32(trs[0].reward))
            ref_rewards.append(np.float32(ref_r))
            assert trs[0].terminated == ref_term and trs[0].truncated == ref_trunc
            if ref_term or ref_trunc:
                break
        assert len(got_rewards) >= 4
        np.testing.assert_allclose(got_rewards, ref_rewards, atol=1e-6, rtol=0.0)
        rewards_by_w[w] = got_rewards

    # Non-terminal rewards under w=0.0 must all be 0.0, and differ from w=1.0.
    assert all(r == 0.0 for r in rewards_by_w[0.0][:-1])
    assert any(abs(a - b) > 1e-4 for a, b in zip(rewards_by_w[1.0][:-1], rewards_by_w[0.0][:-1]))


def test_opponent_forward_auto_dispatch_and_cross_arch_anchor():
    # Part A: FakeTwoSeatEnv + MockStacked verifying auto_stack=True dispatch & non-stackable anchor isolation.
    pool = SnapshotPool(pool_max=10)
    e0 = pool.add(_dummy_state(0), 0)
    e1 = pool.add(_dummy_state(1), 100)
    anc = pool.add(_dummy_state(9), 0, pinned=True, name="anchor_0", net_template=object())

    per_snap_calls = []
    stacked_calls = []
    stacked_syncs = []

    def make_policy(entry):
        def policy(envs):
            per_snap_calls.append((entry.name, len(envs)))
            return fake_scorer(envs)

        return policy

    class MockStacked:
        def sync(self, entries):
            stacked_syncs.append([e.name for e in entries])

        def __call__(self, groups):
            groups = list(groups)
            stacked_calls.append([(name, len(envs)) for name, envs in groups])
            return [fake_scorer(envs) for _, envs in groups]

    # Case 1: 1 stackable group (e0 on slot 0, seed 1 => opponent moves first after tick on learner)
    # Use seed=1 (seat 1 is learner, so seat 0 opponent moves first after reset_all? No, reset_all advances
    # opponent's opening turn, so learner moves then passes turn back to opponent!).
    def run_case(slot_opps):
        per_snap_calls.clear()
        stacked_calls.clear()
        stacked_syncs.clear()
        n = len(slot_opps)
        vec = KeyedVec(
            lambda opp: FakeTwoSeatEnv(k=2, turns=2),
            n,
            seed_base=0,
            seed_stride=2,  # all even seeds => learner is seat 0 on all slots!
            pool=pool,
            make_policy=make_policy,
            rng=np.random.default_rng(0),
            p_bot=0.0,
            reward=FAKE_REWARD,
            choices=slot_opps,
            pick=lambda seed: (seed // 2) % n,
            stacked_policy=MockStacked(),
            auto_stack=True,
        )
        vec.reset_all()
        per_snap_calls.clear()
        # Tick 1: learner MOVE on all slots (still seat 0).
        vec.tick({i: (0, MOVE, 1) for i in range(n)})
        # Tick 2: learner END_TURN on all slots -> all slots switch to seat 1 (opponent) and opponent moves!
        vec.tick({i: (FAKE_G, END_TURN, 0) for i in range(n)})
        return list(per_snap_calls), list(stacked_calls), list(stacked_syncs)

    # 1 stackable group -> per-snapshot only, zero stacked calls.
    ps1, st1, sy1 = run_case([e0, e0])
    assert ps1 == [("snap_0", 2)] and st1 == [] and sy1 == []

    # 2 stackable groups -> stacked call, sync contains only stackable entries (never anchor_0).
    ps2, st2, sy2 = run_case([e0, e1])
    assert ps2 == []
    assert st2 == [[("snap_0", 1), ("snap_100", 1)]]
    assert sy2 == [["snap_0", "snap_100"]]

    # 1 stackable group + 1 cross-arch anchor -> BOTH via per-snapshot (only 1 stackable group).
    ps3, st3, sy3 = run_case([e0, anc])
    assert sorted(ps3) == [("anchor_0", 1), ("snap_0", 1)]
    assert st3 == [] and sy3 == []

    # 2 stackable groups + 1 cross-arch anchor -> 2 stackable via stacked, anchor via per-snapshot,
    # sync NEVER includes anchor_0.
    ps4, st4, sy4 = run_case([e0, e1, anc])
    assert ps4 == [("anchor_0", 1)]
    assert st4 == [[("snap_0", 1), ("snap_100", 1)]]
    assert sy4 == [["snap_0", "snap_100"]]

    # Part B: Real v2 env with arch="v2" learner + arch="v1" pinned anchor + real StackedSnapshots under "auto".
    sp = _spec(size=6, ruleset="v2")
    learner = _tiny_net(sp, seed=1, arch="v2")
    anchor_v1 = _tiny_net(sp, seed=2, arch="v1")
    real_pool = SnapshotPool(pool_max=10)
    real_pool.add(anchor_v1.state_dict(), 0, pinned=True, name="anchor_0", net_template=anchor_v1)
    real_pool.add(learner.state_dict(), 0)
    real_pool.add(_tiny_net(sp, seed=3, arch="v2").state_dict(), 100)

    opp_gen = torch.Generator().manual_seed(0)

    def real_make_policy(entry):
        base = entry.net_template if entry.net_template is not None else learner
        snap = copy.deepcopy(base)
        snap.load_state_dict(entry.state)
        return net_batch_policy(snap, opp_gen, greedy=True, device="cpu")

    stack = StackedSnapshots(copy.deepcopy(learner), "cpu")
    cfg = PPOConfig(size=6, turn_cap=3, ruleset="v2", arch="v2", num_envs=4, p_bot=0.0, p_self=0.0, opponent_forward="auto")
    real_vec = make_selfplay_vec_env(
        cfg,
        seed_base=0,
        pool=real_pool,
        make_policy=real_make_policy,
        rng=np.random.default_rng(4),
        stacked_policy=stacked_batch_policy(stack, opp_gen, greedy=True),
    )
    real_vec.reset_all()
    for _ in range(40):
        acts = {i: tuple(int(x) for x in real_vec.envs[i].bot_action("economy")) for i in real_vec.learner_slots()}
        real_vec.tick(acts)


# ---------------------------------------------------------------- 6. collect_rollout_stream


def test_collect_rollout_stream_exact_batch_no_drain_and_bootstrap_reference():
    pool = pool_of(1)

    def build_vec():
        # Slot 0: LIVE_SELF, Slot 1: snap_0, Slot 2: LIVE_SELF; truncate_at=9 exercises truncation too!
        return KeyedVec(
            lambda opp: FakeTwoSeatEnv(k=3, turns=2, truncate_at=9),
            num_envs=3,
            seed_base=0,
            seed_stride=1,
            pool=pool,
            make_policy=lambda entry: fake_scorer,
            rng=np.random.default_rng(0),
            p_bot=0.0,
            p_self=0.5,
            reward=FAKE_REWARD,
            choices=[LIVE_SELF, pool.entries[0], LIVE_SELF],
            pick=lambda seed: seed % 3,
        )

    vec = build_vec()
    vec.reset_all()
    net = StubNet()
    T, E = 4, 3
    batch = T * E
    buf = RolloutBuffer(T, E, FakeTwoSeatEnv().spec())
    state = StreamState()

    # Independent shadow observer wrapping vec.tick to record exact (slot, seat) records and verify
    # every single transition, segment_end, and bootstrap value independently!
    shadow_pending: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, tuple[int, int, int], float, float]] = {}
    shadow_carry: list[tuple[tuple[int, int], dict]] = []
    orig_tick = vec.tick
    ticks_per_rollout = []

    def traced_tick(learner_actions):
        ticks_per_rollout[-1] += 1
        for slot, act in learner_actions.items():
            seat = vec.learner_seat(slot)
            b, g = vec.envs[slot].observation()
            shadow_pending[(slot, seat)] = (b.copy(), g.copy(), act, float(stub_logp(b, g)), float(stub_value(b, g)))
        trs = orig_tick(learner_actions)
        for tr in trs:
            b, g, act, lp, val = shadow_pending.pop((tr.slot, tr.seat))
            shadow_carry.append(
                (
                    (tr.slot, tr.seat),
                    {
                        "board": b,
                        "globals": g,
                        "action": act,
                        "logp": lp,
                        "value": val,
                        "reward": float(tr.reward),
                        "terminated": bool(tr.terminated),
                        "truncated": bool(tr.truncated),
                        "final_board": None if tr.final_board is None else tr.final_board.copy(),
                        "final_globals": None if tr.final_globals is None else tr.final_globals.copy(),
                    },
                )
            )
        return trs

    vec.tick = traced_tick

    saw_mid_turn_boundary = False
    saved_snap = None
    rollouts_after_save = []

    for r_idx in range(3):
        ticks_per_rollout.append(0)
        collect_rollout_stream(net, vec, buf, torch.Generator().manual_seed(r_idx), state)

        # Pop exact `batch` records from shadow_carry into lanes sorted by (slot, seat).
        taken = shadow_carry[:batch]
        del shadow_carry[:batch]
        assert len(taken) == batch

        lanes: dict[tuple[int, int], list[dict]] = {}
        for lane_id, rec in taken:
            lanes.setdefault(lane_id, []).append(rec)

        ordered = []
        want_seg_end = []
        want_next_val = []
        for lane_id in sorted(lanes):
            recs = lanes[lane_id]
            for j, rec in enumerate(recs):
                is_end = j == len(recs) - 1
                ordered.append(rec)
                want_seg_end.append(is_end)
                if rec["truncated"]:
                    want_next_val.append(float(stub_value(rec["final_board"], rec["final_globals"])))
                elif is_end and not rec["terminated"]:
                    # Priority 1: first carryover record for lane_id.
                    carry_match = [c_rec for c_lane, c_rec in shadow_carry if c_lane == lane_id]
                    if carry_match:
                        want_next_val.append(carry_match[0]["value"])
                    elif lane_id in shadow_pending:
                        # Priority 2: pending step for lane_id.
                        want_next_val.append(shadow_pending[lane_id][4])
                    else:
                        # Priority 3: current observation of vec.envs[slot].
                        cb, cg = vec.envs[lane_id[0]].observation()
                        want_next_val.append(float(stub_value(cb, cg)))
                else:
                    want_next_val.append(0.0)

        assert buf.segment_end is not None
        assert buf.segment_end.shape == (batch,) and buf.segment_end.dtype == torch.bool
        assert buf.segment_end.tolist() == want_seg_end, f"rollout {r_idx}: segment_end mismatch"
        assert buf.segment_end[-1].item() is True

        flat_b = buf.board.reshape(batch, *buf.board.shape[2:])
        flat_g = buf.globals.reshape(batch, -1)
        flat_a = buf.actors.reshape(batch)
        flat_k = buf.kinds.reshape(batch)
        flat_x = buf.args.reshape(batch)
        flat_lp = buf.logp.reshape(batch)
        flat_v = buf.values.reshape(batch)
        flat_r = buf.rewards.reshape(batch)
        flat_term = buf.terminated.reshape(batch)
        flat_trunc = buf.truncated.reshape(batch)
        flat_nv = buf.next_values.reshape(batch)

        for idx, rec in enumerate(ordered):
            assert torch.equal(flat_b[idx], torch.from_numpy(rec["board"])), (r_idx, idx)
            assert torch.equal(flat_g[idx], torch.from_numpy(rec["globals"])), (r_idx, idx)
            assert (flat_a[idx].item(), flat_k[idx].item(), flat_x[idx].item()) == rec["action"], (r_idx, idx)
            assert flat_lp[idx].item() == pytest.approx(rec["logp"], abs=1e-6), (r_idx, idx)
            assert flat_v[idx].item() == pytest.approx(rec["value"], abs=1e-6), (r_idx, idx)
            assert flat_r[idx].item() == pytest.approx(rec["reward"], abs=1e-6), (r_idx, idx)
            assert bool(flat_term[idx]) is rec["terminated"], (r_idx, idx)
            assert bool(flat_trunc[idx]) is rec["truncated"], (r_idx, idx)
            assert flat_nv[idx].item() == pytest.approx(want_next_val[idx], abs=1e-6), (r_idx, idx)

        # Verify barrier-free property: at least one slot stays pending or has carryover across boundaries.
        if shadow_pending or shadow_carry:
            saw_mid_turn_boundary = True

        if r_idx == 0:
            saved_snap = (copy.deepcopy(vec.state_dict()), copy.deepcopy(state.state_dict()))
        else:
            rollouts_after_save.append(
                (
                    buf.board.clone(),
                    buf.rewards.clone(),
                    buf.values.clone(),
                    buf.next_values.clone(),
                    buf.segment_end.clone(),
                )
            )

    assert saw_mid_turn_boundary, "stream collector should not drain pending/carryover state at rollout boundary"

    # Verify exact StreamState + SelfPlayVecEnv resume mid-stream reproduces rollouts 1 & 2 bit-for-bit.
    vec2 = build_vec()
    vec2.load_state_dict(saved_snap[0])
    state2 = StreamState()
    state2.load_state_dict(saved_snap[1])
    buf2 = RolloutBuffer(T, E, FakeTwoSeatEnv().spec())
    net2 = StubNet()
    for r_idx, want_tensors in zip((1, 2), rollouts_after_save):
        collect_rollout_stream(net2, vec2, buf2, torch.Generator().manual_seed(r_idx), state2)
        got_tensors = (buf2.board, buf2.rewards, buf2.values, buf2.next_values, buf2.segment_end)
        for g_t, w_t in zip(got_tensors, want_tensors):
            assert torch.equal(g_t, w_t)


def test_train_update_with_stream_buffer_plain_and_distill():
    sp_v2 = _spec(size=6, ruleset="v2")
    sp_v1 = _spec(size=6, ruleset="v1")
    student = _tiny_net(sp_v2, seed=10, arch="v2")
    teacher = _tiny_net(sp_v1, seed=20, arch="v1")
    teacher.eval().requires_grad_(False)

    cfg = PPOConfig(
        size=6,
        turn_cap=3,
        ruleset="v2",
        arch="v2",
        width=8,
        blocks=1,
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        selfplay=True,
        rollout="stream",
        p_bot=0.25,
        p_self=0.5,
        device="cpu",
    )
    pool = SnapshotPool(pool_max=4)
    pool.add(student.state_dict(), 0)
    opp_gen = torch.Generator().manual_seed(0)

    def make_policy(entry):
        s = copy.deepcopy(student)
        s.load_state_dict(entry.state)
        return net_batch_policy(s, opp_gen, greedy=False, device="cpu")

    vec = make_selfplay_vec_env(cfg, seed_base=0, pool=pool, make_policy=make_policy, rng=np.random.default_rng(1))
    vec.reset_all()
    buf = RolloutBuffer(cfg.num_steps, cfg.num_envs, sp_v2, device="cpu")
    stream_state = StreamState()
    collect_rollout_stream(student, vec, buf, torch.Generator().manual_seed(0), stream_state)
    assert buf.segment_end is not None

    opt = torch.optim.Adam(student.parameters(), lr=1e-3, eps=1e-5)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    m_plain = train_update(student, opt, buf, cfg, torch.Generator().manual_seed(1))
    assert math.isfinite(m_plain["pg_loss"]) and math.isfinite(m_plain["v_loss"])
    assert any(not torch.equal(v, student.state_dict()[k]) for k, v in before.items())

    # Now with M22 distillation on a stream buffer!
    collect_rollout_stream(student, vec, buf, torch.Generator().manual_seed(2), stream_state)
    before2 = {k: v.clone() for k, v in student.state_dict().items()}
    m_distill = train_update(
        student,
        opt,
        buf,
        cfg,
        torch.Generator().manual_seed(3),
        teacher=teacher,
        teacher_spec=sp_v1,
        student_spec=sp_v2,
        distill_coef=0.1,
    )
    assert math.isfinite(m_distill["distill_kl"]) and m_distill["distill_coef"] == pytest.approx(0.1)
    assert any(not torch.equal(v, student.state_dict()[k]) for k, v in before2.items())


# ---------------------------------------------------------------- 7. End-to-end CPU train.run


def _revised_cfg(anchor_path: str, **kw) -> PPOConfig:
    batch = 4 * 8  # 32
    base = dict(
        size=6,
        turn_cap=3,
        ruleset="v2",
        arch="v2",
        width=8,
        blocks=1,
        num_envs=4,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        total_steps=4 * batch,  # 128
        selfplay=True,
        rollout="stream",
        opponent_forward="auto",
        p_bot=0.25,
        p_self=0.40,
        p_latest=0.25,
        pfsp_mode="hard",
        pfsp_power=2.0,
        pfsp_ema_alpha=0.1,
        selfplay_bots=[
            {"name": "rush_e0", "bot": "rush", "epsilon": 0.0},
            {"name": "economy_e20", "bot": "economy", "epsilon": 0.2},
        ],
        anchor_checkpoints=[anchor_path],
        snapshot_every=2 * batch,  # 64
        snapshot_win_rate=0.99,  # high gate blocks step 64
        snapshot_min_games=50,
        snapshot_max_steps=4 * batch,  # forces snapshot at step 128
        shaping_weight=1.0,
        shaping_anneal_steps=3 * batch,  # anneal_updates = 3 -> [1.0, 2/3, 1/3, 0.0]
        shaping_end=0.0,
        paired_seeds=True,
        eval_every=2 * batch,  # 64
        eval_seeds=1,
        regression_floor=0.0,  # guaranteed non-regressed -> best.pt saved
        rating_every=3 * batch,  # 96 (tests last_regression persistence across ckpt_64 resume!)
        rating_seeds=1,
        tournament_seeds=1,
        device="cpu",
        seed=5,
        opponents=["economy", "rush"],
        stages=[{"opponents": ["economy", "rush"]}],
    )
    return PPOConfig(**{**base, **kw})


def test_train_run_revised_selfplay_end_to_end_and_exact_resume(tmp_path):
    # Cross-arch anchor: v2 ruleset, arch="v1" checkpoint loaded into arch="v2" learner run.
    anchor_path = tmp_path / "anchor_v1.pt"
    anchor_net = _save_anchor(anchor_path, size=6, ruleset="v2", arch="v1", seed=77)

    cfg = _revised_cfg(str(anchor_path))
    run_dir = tmp_path / "run"
    train.run(cfg, str(run_dir))

    # 1. metrics.jsonl fields & shaping_weight decay.
    mrows = _jsonl(run_dir / "metrics.jsonl")
    assert [r["update"] for r in mrows] == [1, 2, 3, 4]
    assert [r["shaping_weight"] for r in mrows] == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0], abs=1e-6)
    for r in mrows:
        assert isinstance(r["sp_win_rate"], dict)
        assert "win_rate_bots" in r and "win_rate_pool" in r and "win_rate_latest" in r
        for k in ("win_rate_bots", "win_rate_pool", "win_rate_latest"):
            assert r[k] is None or (0.0 <= r[k] <= 1.0)

    # 2. Win-rate gate blocked step 64, snapshot_max_steps forced step 128; pinned anchor_0 saved.
    pool_files = sorted(p.name for p in (run_dir / "pool").iterdir())
    assert pool_files == ["anchor_0.pt", "snap_0.pt", "snap_128.pt"]
    saved_anchor = torch.load(run_dir / "pool" / "anchor_0.pt", map_location="cpu", weights_only=False)
    assert _same_state(saved_anchor["model"], anchor_net.state_dict())

    # 3. Non-regressed best.pt saved at rating_every (step 96).
    assert (run_dir / "best.pt").exists()
    erows = _jsonl(run_dir / "eval.jsonl")
    assert [r["global_step"] for r in erows if r["type"] == "suite"] == [64, 128]
    assert [r["global_step"] for r in erows if r["type"] == "rating"] == [96]

    # 4. Tournament includes anchor_0 and rebuilds cleanly via tournament.main (cross-arch!).
    t_orig = json.loads((run_dir / "tournament.json").read_text())
    assert set(t_orig["players"]) == {"random", "economy", "rush", "anchor_0", "snap_0", "snap_128"}
    rebuilt = tmp_path / "rebuilt"
    shutil.copytree(run_dir / "pool", rebuilt / "pool")
    tournament.main(["--run", str(rebuilt), "--seeds", "1"])
    t_rebuilt = json.loads((rebuilt / "tournament.json").read_text())
    assert set(t_rebuilt["players"]) == {"random", "economy", "rush", "anchor_0", "snap_0", "snap_128"}

    # 5. Exact bit-for-bit mid-run resume from ckpt_64.pt (mid-stream, mid-anneal, mid-mirror-queue,
    # with tracker + carryover + pending + last_regression + anchor_0).
    resumed_dir = tmp_path / "resumed"
    train.run(cfg, str(resumed_dir), resume=str(run_dir / "ckpt_64.pt"))
    res_mrows = _jsonl(resumed_dir / "metrics.jsonl")
    assert [r["update"] for r in res_mrows] == [3, 4]
    assert [_untimed(r) for r in res_mrows] == [_untimed(r) for r in mrows[2:]]
    res_erows = [_untimed(r) for r in _jsonl(resumed_dir / "eval.jsonl")]
    assert res_erows == [_untimed(r) for r in erows if r["global_step"] > 64]
    assert (resumed_dir / "best.pt").exists()
    assert _same_state(
        torch.load(resumed_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)["model"],
        torch.load(run_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)["model"],
    )
    for name in ("anchor_0.pt", "snap_0.pt", "snap_128.pt"):
        assert _same_state(
            torch.load(resumed_dir / "pool" / name, map_location="cpu", weights_only=False)["model"],
            torch.load(run_dir / "pool" / name, map_location="cpu", weights_only=False)["model"],
        ), name

    # 6. Unchanged v1 rules_hash and obs_hash(11).
    v1_spec = ps.Env(size=11, ruleset="v1").spec()
    assert v1_spec["rules_hash"] == "a8b52b91eae34503"
    assert v1_spec["obs_hash"] == "d5f9b1033cfd55ca"


def test_train_run_best_pt_regressed_win_rate_gate_trigger_and_anchor_spec_mismatch(tmp_path):
    anchor_path = tmp_path / "anchor_v1.pt"
    _save_anchor(anchor_path, size=6, ruleset="v2", arch="v1", seed=88)

    # 1. Regressed run (regression_floor=1.01) must NOT write best.pt; win-rate gate at 0.0 with
    # min_games=1 against snap_0 triggers snap_32.pt BEFORE snapshot_max_steps=640.
    cfg = _revised_cfg(
        str(anchor_path),
        total_steps=64,
        p_bot=0.0,
        p_self=0.0,
        p_latest=1.0,  # every game is vs latest unpinned snapshot (snap_0)
        snapshot_every=32,
        snapshot_win_rate=0.0,
        snapshot_min_games=1,
        snapshot_max_steps=640,
        eval_every=32,
        rating_every=32,
        regression_floor=1.01,  # guaranteed regression=True
        shaping_anneal_steps=None,
    )
    reg_dir = tmp_path / "regressed"
    train.run(cfg, str(reg_dir))
    assert not (reg_dir / "best.pt").exists(), "best.pt must not be saved when latest suite regressed"
    pool_files = sorted(p.name for p in (reg_dir / "pool").iterdir())
    assert "snap_32.pt" in pool_files, f"win-rate gate should have triggered at step 32: {pool_files}"
    assert "anchor_0.pt" in pool_files

    # 2. Anchor with mismatched obs_hash raises SpecMismatch at startup.
    bad_anchor = tmp_path / "bad_anchor.pt"
    _save_anchor(bad_anchor, size=6, ruleset="v2", arch="v1", seed=99, obs_hash="0000000000000000")
    bad_cfg = _revised_cfg(str(bad_anchor), total_steps=32)
    with pytest.raises(SpecMismatch):
        train.run(bad_cfg, str(tmp_path / "bad_run"))
