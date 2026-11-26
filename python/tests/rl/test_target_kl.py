"""Milestone 16b tests 5: epoch-mean `target_kl` early stopping in `ppo.train_update`.

Checks (docs/MILESTONE_16b.md, Rule 5 & Clarification B4):
- `train_update` accumulates `approx_kl` across all minibatches of an epoch and stops early iff
  the epoch mean (`epoch_kl / num_minibatches`) exceeds `cfg.target_kl`.
- High-then-low minibatches (epoch mean > target_kl, last minibatch < target_kl) MUST stop early.
- Low-then-high minibatches (epoch mean < target_kl, last minibatch > target_kl) MUST NOT stop early.
- The per-epoch accumulator resets between epochs.
- `cfg.target_kl is None` never stops early.
"""

import math

import polytopia_sim as ps
import pytest
import torch
from torch import nn

from polytopia_rl.config import PPOConfig
from polytopia_rl.model import EvalOut, PolicyNet
from polytopia_rl.ppo import RolloutBuffer, train_update


def _kl_of_ratio(r: float) -> float:
    return (r - 1.0) - math.log(r)


class _ScriptedKLNet(nn.Module):
    """Minimal net whose `evaluate` produces exact per-minibatch probability ratios `ratios[call]`,
    given `buffer.logp == 0`, so `approx_kl` per minibatch is `(r - 1) - ln(r)` exactly."""

    def __init__(self, ratios: list[float]):
        super().__init__()
        self.ratios = list(ratios)
        self.param = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def evaluate(self, board, globals, actors, kinds, args, actor_mask, kind_mask, arg_mask) -> EvalOut:
        r = self.ratios[self.calls]
        self.calls += 1
        b = board.shape[0]
        logp = torch.full((b,), math.log(r), dtype=torch.float32) + 0.0 * self.param
        zero = torch.zeros(b, dtype=torch.float32) + 0.0 * self.param
        return EvalOut(
            logp=logp,
            entropy=zero,
            head_entropy={"actor": zero, "kind": zero, "arg": zero},
            value=zero,
        )


def _tiny_buffer(n_steps=2, n_envs=2):
    spec = {
        "height": 2,
        "width": 2,
        "channels": 1,
        "globals": 1,
        "num_actors": 13,
        "action_types": ["move"] * 10,
    }
    return RolloutBuffer(n_steps, n_envs, spec)


def _run_scripted(ratios: list[float], target_kl: float | None, update_epochs: int = 3, num_minibatches: int = 2):
    net = _ScriptedKLNet(ratios)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    buf = _tiny_buffer(n_steps=2, n_envs=2)
    cfg = PPOConfig(
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        target_kl=target_kl,
        norm_adv=False,
    )
    metrics = train_update(net, opt, buf, cfg, torch.Generator().manual_seed(0))
    return net.calls, metrics


def test_stops_when_epoch_mean_exceeds_target_kl_even_if_last_minibatch_is_below():
    # Epoch 1: mb1 r=2.0 (kl ~ 0.3069), mb2 r=1.0 (kl = 0.0) -> epoch mean ~ 0.1534 > 0.10,
    # while last minibatch kl = 0.0 < 0.10.
    kl_hi = _kl_of_ratio(2.0)
    assert kl_hi / 2 > 0.10
    calls, m = _run_scripted([2.0, 1.0, 1.0, 1.0, 1.0, 1.0], target_kl=0.10, update_epochs=3, num_minibatches=2)
    assert calls == 2, f"expected early stop after epoch 1 (2 minibatches), got {calls}"
    assert m["approx_kl"] == pytest.approx(kl_hi / 2, abs=1e-5)


def test_continues_when_epoch_mean_is_below_target_kl_even_if_last_minibatch_exceeds_it():
    # Epoch 1: mb1 r=1.0 (kl = 0.0), mb2 r=2.0 (kl ~ 0.3069) -> epoch mean ~ 0.1534 < 0.20,
    # while last minibatch kl ~ 0.3069 > 0.20. Must NOT stop early.
    kl_hi = _kl_of_ratio(2.0)
    assert kl_hi / 2 < 0.20 < kl_hi
    calls, m = _run_scripted([1.0, 2.0, 1.0, 1.0, 1.0, 1.0], target_kl=0.20, update_epochs=3, num_minibatches=2)
    assert calls == 6, f"expected all 3 epochs (6 minibatches), got {calls}"
    assert m["approx_kl"] == pytest.approx(kl_hi / 6, abs=1e-5)


def test_epoch_kl_accumulator_resets_each_epoch():
    # r=1.5 -> kl ~ 0.09453. With target_kl = 0.10:
    # Epoch 1: [1.5, 1.5] -> mean 0.09453 < 0.10 (continue).
    # Epoch 2: [1.5, 1.0] -> mean 0.04727 < 0.10 (continue; cumulative sum / 2 would be 0.1418 > 0.10!).
    # Epoch 3: [1.0, 1.0] -> mean 0.0 < 0.10.
    kl_mid = _kl_of_ratio(1.5)
    assert kl_mid < 0.10 < 1.5 * kl_mid
    calls, _ = _run_scripted([1.5, 1.5, 1.5, 1.0, 1.0, 1.0], target_kl=0.10, update_epochs=3, num_minibatches=2)
    assert calls == 6, f"epoch KL must reset between epochs (expected 6 minibatches, got {calls})"

    # Now stop on epoch 2 specifically: epoch 1 mean ~ 0.0473 < 0.08, epoch 2 mean ~ 0.0945 > 0.08.
    calls2, _ = _run_scripted([1.0, 1.5, 1.5, 1.5, 1.0, 1.0], target_kl=0.08, update_epochs=3, num_minibatches=2)
    assert calls2 == 4, f"expected early stop after epoch 2 (4 minibatches), got {calls2}"


def test_target_kl_none_never_stops_early():
    calls, _ = _run_scripted([5.0] * 6, target_kl=None, update_epochs=3, num_minibatches=2)
    assert calls == 6


def test_real_policynet_target_kl_smoke():
    spec = ps.Env(size=6).spec()
    torch.manual_seed(0)
    net = PolicyNet(spec, width=8, blocks=1)
    buf = RolloutBuffer(4, 2, spec)
    # Give valid all-true masks so evaluate runs cleanly.
    buf.actor_mask.fill_(True)
    buf.kind_mask.fill_(True)
    buf.arg_mask.fill_(True)
    buf.rewards.fill_(1.0)
    buf.logp.fill_(-5.0)  # forces large positive approx_kl on epoch 1

    calls = 0
    real_eval = net.evaluate

    def spy_eval(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_eval(*args, **kwargs)

    net.evaluate = spy_eval
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    cfg = PPOConfig(size=6, width=8, blocks=1, update_epochs=4, num_minibatches=2, target_kl=0.001)
    train_update(net, opt, buf, cfg, torch.Generator().manual_seed(0))
    assert calls == 2, f"real PolicyNet should stop after epoch 1 (2 minibatches), got {calls}"
