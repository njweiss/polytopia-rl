"""Milestone 10 tests 5: GAE ("PPO (CleanRL conventions)").

Hand cases use r = [1, 0, 2], V = [0.5, 0.2, 0.1], gamma = 0.9, lam = 0.8 (gamma * lam = 0.72):
  no done, last_value 0.4: delta = [0.68, -0.11, 2.26], A = [1.772384, 1.5172, 2.26]
  terminated at t=1:       delta_1 = -0.2, A = [0.536, -0.2, 2.26]
  truncated at t=1, final value 1.0: delta_1 = 0.9 - 0.2 = 0.7, A = [1.184, 0.7, 2.26]
  terminated at t=2:       A_2 = 2 - 0.1 = 1.9 (last_value unused)
  truncated at t=2, final value 3.0: A_2 = 2 + 2.7 - 0.1 = 4.6 (final value, not last_value)
"""

import numpy as np
import pytest
import torch

from polytopia_rl.gae import compute_gae

R = [1.0, 0.0, 2.0]
V = [0.5, 0.2, 0.1]


def col(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype).reshape(-1, 1)


def gae1(term=(False,) * 3, trunc=(False,) * 3, next_values=(0.0,) * 3, last_value=0.4, gamma=0.9, lam=0.8):
    adv, ret = compute_gae(
        col(R),
        col(V),
        col(term, torch.bool),
        col(trunc, torch.bool),
        col(next_values),
        torch.tensor([last_value]),
        gamma,
        lam,
    )
    return adv[:, 0].tolist(), ret[:, 0].tolist()


def test_single_env_no_done():
    adv, ret = gae1()
    assert adv == pytest.approx([1.772384, 1.5172, 2.26], abs=1e-5)
    assert ret == pytest.approx([2.272384, 1.7172, 2.36], abs=1e-5)


def test_terminated_step_has_no_bootstrap():
    adv, _ = gae1(term=(False, True, False))
    assert adv == pytest.approx([0.536, -0.2, 2.26], abs=1e-5)


def test_truncated_step_bootstraps_from_its_final_observation():
    adv, _ = gae1(trunc=(False, True, False), next_values=(0.0, 1.0, 0.0))
    assert adv == pytest.approx([1.184, 0.7, 2.26], abs=1e-5)


def test_rollout_end_bootstrap():
    assert gae1(last_value=0.0)[0][2] == pytest.approx(1.9, abs=1e-6)
    assert gae1(term=(False, False, True))[0][2] == pytest.approx(1.9, abs=1e-6)
    assert gae1(trunc=(False, False, True), next_values=(0.0, 0.0, 3.0))[0][2] == pytest.approx(4.6, abs=1e-5)


def test_two_envs_are_independent():
    rewards = torch.tensor([R, R], dtype=torch.float32).T
    values = torch.tensor([V, V], dtype=torch.float32).T
    term = torch.tensor([[False, False, False], [False, True, False]]).T
    trunc = torch.zeros(3, 2, dtype=torch.bool)
    adv, ret = compute_gae(rewards, values, term, trunc, torch.zeros(3, 2), torch.tensor([0.4, 0.4]), 0.9, 0.8)
    assert adv[:, 0].tolist() == pytest.approx([1.772384, 1.5172, 2.26], abs=1e-5)
    assert adv[:, 1].tolist() == pytest.approx([0.536, -0.2, 2.26], abs=1e-5)
    np.testing.assert_allclose(ret.numpy(), (adv + values).numpy(), atol=1e-6)


def reference_gae(r, v, term, trunc, nv, last, gamma, lam):
    t_len, e_len = r.shape
    adv = np.zeros((t_len, e_len))
    for e in range(e_len):
        a_next = 0.0
        for t in reversed(range(t_len)):
            if trunc[t, e]:
                v_next = nv[t, e]
            elif t == t_len - 1:
                v_next = last[e]
            else:
                v_next = v[t + 1, e]
            delta = r[t, e] + gamma * v_next * (0.0 if term[t, e] else 1.0) - v[t, e]
            done = term[t, e] or trunc[t, e]
            a_next = delta + gamma * lam * (0.0 if done else 1.0) * a_next
            adv[t, e] = a_next
    return adv


@pytest.mark.parametrize("seed", range(4))
def test_matches_reference_loop_on_random_inputs(seed):
    rng = np.random.default_rng(seed)
    t_len, e_len = 17, 5
    r = rng.normal(size=(t_len, e_len)).astype(np.float32)
    v = rng.normal(size=(t_len, e_len)).astype(np.float32)
    nv = rng.normal(size=(t_len, e_len)).astype(np.float32)
    last = rng.normal(size=e_len).astype(np.float32)
    u = rng.random((t_len, e_len))
    term, trunc = u < 0.15, (u >= 0.15) & (u < 0.25)
    gamma, lam = 0.97, 0.9
    adv, ret = compute_gae(
        *(torch.from_numpy(x) for x in (r, v, term, trunc, nv, last)), gamma, lam
    )
    ref = reference_gae(r, v, term, trunc, nv, last, gamma, lam)
    np.testing.assert_allclose(adv.numpy(), ref, atol=1e-4)
    np.testing.assert_allclose(ret.numpy(), ref + v, atol=1e-4)


def test_lambda_zero_is_one_step_td_and_gamma_zero_is_r_minus_v():
    adv, _ = gae1(lam=0.0)
    assert adv == pytest.approx([0.68, -0.11, 2.26], abs=1e-6)
    adv, _ = gae1(gamma=0.0, lam=0.0)
    assert adv == pytest.approx([0.5, -0.2, 1.9], abs=1e-6)
