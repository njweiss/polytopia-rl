"""Milestone 10 tests 6: PPO loss and minibatching (CleanRL conventions; hand-computed values).

Surrogate cases (eps 0.2, term = max(-A r, -A clip(r))):
  r=1.0 A=+1 -> -1.0     r=1.5 A=+1 -> -1.2 (clipped)   r=0.5 A=-1 -> 0.8 (clipped)
  r=1.1 A=-2 -> 2.2      r=0.5 A=+1 -> -0.5 (unclipped) r=1.5 A=-1 -> 1.5 (unclipped)
  mean = 1.8 / 6 = 0.3; clipfrac = 4/6 (|r - 1| > 0.2 for r = 1.5, 0.5, 0.5, 1.5).
"""

import math

import numpy as np
import pytest
import torch

from polytopia_rl.config import PPOConfig
from polytopia_rl.ppo import minibatch_indices, ppo_loss

RATIOS = [1.0, 1.5, 0.5, 1.1, 0.5, 1.5]
ADV = [1.0, 1.0, -1.0, -2.0, 1.0, -1.0]


def cfg(**kw):
    base = dict(clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, norm_adv=False, clip_vloss=False)
    return PPOConfig(**{**base, **kw})


def f32(x):
    return torch.tensor(x, dtype=torch.float32)


def loss(ratios=RATIOS, adv=ADV, new_value=None, old_value=None, returns=None, entropy=None, **kw):
    n = len(ratios)
    zeros = [0.0] * n
    return ppo_loss(
        torch.log(f32(ratios)),
        f32(zeros),
        f32(adv),
        f32(new_value or zeros),
        f32(old_value or zeros),
        f32(returns or zeros),
        f32(entropy or zeros),
        cfg(**kw),
    )


def test_clipped_surrogate_both_signs_inside_and_outside_the_clip():
    out = loss()
    assert out.pg.item() == pytest.approx(0.3, abs=1e-6)
    assert out.clipfrac.item() == pytest.approx(4 / 6, abs=1e-6)


def test_approx_kl_definition():
    r = np.array(RATIOS)
    assert loss().approx_kl.item() == pytest.approx(np.mean((r - 1) - np.log(r)), abs=1e-6)


def test_value_loss_unclipped_and_clipped():
    # Unclipped: 0.5 * mean((V - R)^2) with V = [1, 1], R = [2, 0.5] -> 0.5 * (1 + 0.25) / 2 = 0.3125.
    v = loss(ratios=[1.0, 1.0], adv=[0.0, 0.0], new_value=[1.0, 1.0], old_value=[0.0, 0.0], returns=[2.0, 0.5])
    assert v.value.item() == pytest.approx(0.3125, abs=1e-6)
    # Clipped (old V = 0, eps 0.2 -> V_clip = 0.2): sample 1 max(1, 3.24) = 3.24; sample 2
    # max(0.25, 0.09) = 0.25 -> 0.5 * (3.24 + 0.25) / 2 = 0.8725.
    c = loss(
        ratios=[1.0, 1.0],
        adv=[0.0, 0.0],
        new_value=[1.0, 1.0],
        old_value=[0.0, 0.0],
        returns=[2.0, 0.5],
        clip_vloss=True,
    )
    assert c.value.item() == pytest.approx(0.8725, abs=1e-6)


def test_total_combines_terms_with_entropy_bonus_sign():
    out = loss(new_value=[1.0] * 6, returns=[0.0] * 6, entropy=[2.0] * 6)
    # pg 0.3, value 0.5, entropy 2.0: total = 0.3 + 0.5 * 0.5 - 0.01 * 2.0 = 0.53.
    assert out.entropy.item() == pytest.approx(2.0, abs=1e-6)
    assert out.total.item() == pytest.approx(0.53, abs=1e-6)
    more_entropy = loss(new_value=[1.0] * 6, returns=[0.0] * 6, entropy=[3.0] * 6)
    assert more_entropy.total.item() < out.total.item()


def test_gradient_at_ratio_one_is_minus_advantage_times_grad_logp():
    old = f32([-0.3, -1.2, -2.0, -0.7])
    new = old.clone().requires_grad_(True)
    old_req = old.clone().requires_grad_(True)
    adv = f32([1.5, -0.5, 2.0, -1.0])
    z = torch.zeros(4)
    out = ppo_loss(new, old_req, adv, z, z, z, z, cfg())
    out.pg.backward()
    np.testing.assert_allclose(new.grad.numpy(), -adv.numpy() / 4, atol=1e-6)
    assert old_req.grad is None or not old_req.grad.abs().sum().item()


def test_advantages_normalized_per_batch():
    # adv [1, 2, 3, 4]: mean 2.5, unbiased std sqrt(5/3); ratios inside the clip -> pg = mean(-A_n r).
    adv = np.array([1.0, 2.0, 3.0, 4.0])
    r = np.array([1.1, 0.9, 1.0, 1.05])
    a_n = (adv - 2.5) / (math.sqrt(5 / 3) + 1e-8)
    out = loss(ratios=r.tolist(), adv=adv.tolist(), norm_adv=True)
    assert out.pg.item() == pytest.approx(np.mean(-a_n * r), abs=1e-6)
    # Normalization is over the given batch: shifting and scaling it changes nothing.
    out2 = loss(ratios=r.tolist(), adv=(adv * 10 + 7).tolist(), norm_adv=True)
    assert out2.pg.item() == pytest.approx(out.pg.item(), abs=1e-5)


@pytest.mark.parametrize("n,m", [(12, 4), (1024, 8), (7, 1)])
def test_minibatch_indices_cover_every_sample_once(n, m):
    parts = minibatch_indices(n, m, torch.Generator().manual_seed(0))
    assert len(parts) == m
    assert all(len(p) == n // m for p in parts)
    assert all(p.dtype == torch.int64 for p in parts)
    assert sorted(torch.cat(parts).tolist()) == list(range(n))


def test_minibatch_indices_are_shuffled_and_seeded():
    a = torch.cat(minibatch_indices(64, 4, torch.Generator().manual_seed(1)))
    b = torch.cat(minibatch_indices(64, 4, torch.Generator().manual_seed(1)))
    assert torch.equal(a, b)
    assert a.tolist() != list(range(64))
