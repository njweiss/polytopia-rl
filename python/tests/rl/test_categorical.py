"""Milestone 10 tests 1: masked categorical distributions.

Expected values: hand-worked softmax (exp(1) vs exp(3)), and the NumPy reference in rl_helpers.
"""

import math

import numpy as np
import pytest
import torch
from rl_helpers import np_masked_entropy, np_masked_logp

from polytopia_rl.categorical import entropy, log_prob, masked_logits, sample


def t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


def random_case(seed, b=64, k=13):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, size=(b, k)).astype(np.float32)
    mask = rng.random((b, k)) < 0.4
    mask[np.arange(b), rng.integers(k, size=b)] = True  # at least one true per row
    return logits, mask


def test_masked_softmax_hand_worked():
    # Valid entries 1 and 3: p = (1, e^2) / (1 + e^2) = (0.1192029, 0.8807971); the masked one is 0.
    p = torch.softmax(masked_logits(t([[1.0, 2.0, 3.0]]), t([[True, False, True]], torch.bool)), -1)
    e2 = math.exp(2.0)
    assert p[0, 0].item() == pytest.approx(1 / (1 + e2), abs=1e-6)
    assert p[0, 2].item() == pytest.approx(e2 / (1 + e2), abs=1e-6)
    assert p[0, 1].item() == 0.0


def test_masked_logits_keeps_valid_entries():
    out = masked_logits(t([[0.5, -2.0, 7.0]]), t([[True, True, False]], torch.bool))
    assert out[0, 0].item() == 0.5 and out[0, 1].item() == -2.0
    assert out[0, 2].item() < -1e6


def test_mask_applied_before_softmax_even_for_huge_logits():
    # A masked entry with a huge logit must not take any mass from the valid ones.
    logits = t([[0.0, 1e4, math.log(3.0)]])
    mask = t([[True, False, True]], torch.bool)
    assert torch.softmax(masked_logits(logits, mask), -1)[0].tolist() == pytest.approx(
        [0.25, 0.0, 0.75], abs=1e-6
    )
    assert log_prob(logits, mask, torch.tensor([2]))[0].item() == pytest.approx(math.log(0.75), abs=1e-5)
    assert entropy(logits, mask)[0].item() == pytest.approx(
        -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)), abs=1e-5
    )


def test_all_false_row_raises():
    mask = t([[True, False], [False, False]], torch.bool)
    with pytest.raises(ValueError):
        masked_logits(torch.zeros(2, 2), mask)


@pytest.mark.parametrize("seed", range(3))
def test_log_prob_and_entropy_match_numpy_reference(seed):
    logits, mask = random_case(seed)
    rng = np.random.default_rng(100 + seed)
    index = np.array([rng.choice(np.flatnonzero(row)) for row in mask])
    ref_lp = np_masked_logp(logits, mask)[np.arange(len(index)), index]
    lp = log_prob(torch.from_numpy(logits), torch.from_numpy(mask), torch.from_numpy(index))
    np.testing.assert_allclose(lp.numpy(), ref_lp, atol=1e-5)
    ent = entropy(torch.from_numpy(logits), torch.from_numpy(mask))
    np.testing.assert_allclose(ent.numpy(), np_masked_entropy(logits, mask), atol=1e-5)


def test_one_valid_option_gives_zero_logp_and_entropy():
    logits = t([[3.0, -50.0, 1e4, 0.2]])
    mask = t([[False, True, False, False]], torch.bool)
    assert abs(log_prob(logits, mask, torch.tensor([1]))[0].item()) < 1e-6
    assert abs(entropy(logits, mask)[0].item()) < 1e-6


def test_entropy_of_uniform_is_log_k():
    mask = t([[True, True, False, True, True]], torch.bool)
    assert entropy(torch.zeros(1, 5), mask)[0].item() == pytest.approx(math.log(4), abs=1e-6)


def test_samples_never_masked():
    logits, mask = random_case(7, b=500, k=9)
    lg, mk = torch.from_numpy(logits), torch.from_numpy(mask)
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        idx = sample(lg, mk, gen).numpy()
        assert idx.dtype == np.int64 and idx.shape == (500,)
        assert mask[np.arange(500), idx].all()


def test_greedy_picks_masked_argmax():
    logits = t([[5.0, 1.0, 3.0, 2.0], [0.0, -1.0, -3.0, 4.0]])
    mask = t([[False, True, True, True], [True, True, True, False]], torch.bool)
    gen = torch.Generator().manual_seed(0)
    assert sample(logits, mask, gen, greedy=True).tolist() == [2, 0]


def test_sampling_frequencies_match_probabilities():
    # p over the valid entries {0, 2, 3} = softmax(1, 3, 0) = (0.1141952, 0.8437947, 0.0420101).
    n = 200_000
    logits = t([[1.0, 2.0, 3.0, 0.0]]).expand(n, 4)
    mask = t([[True, False, True, True]], torch.bool).expand(n, 4)
    gen = torch.Generator().manual_seed(1)
    counts = np.bincount(sample(logits, mask, gen).numpy(), minlength=4) / n
    z = math.exp(1) + math.exp(3) + 1.0
    expected = np.array([math.exp(1) / z, 0.0, math.exp(3) / z, 1.0 / z])
    sigma = np.sqrt(expected * (1 - expected) / n)
    assert counts[1] == 0.0
    # 5 sigma: at most ~0.004 here.
    assert np.all(np.abs(counts - expected) <= 5 * sigma + 1e-12)


def test_fixed_generator_gives_fixed_sample():
    logits, mask = random_case(3, b=200, k=11)
    lg, mk = torch.from_numpy(logits), torch.from_numpy(mask)
    a = sample(lg, mk, torch.Generator().manual_seed(42))
    b = sample(lg, mk, torch.Generator().manual_seed(42))
    c = sample(lg, mk, torch.Generator().manual_seed(43))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
