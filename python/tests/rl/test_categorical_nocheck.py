"""Milestone 16b tests 3: `categorical` `check=False` and `PolicyNet` mask-check sync removal.

Checks (docs/MILESTONE_16b.md, Rule 3 & Clarification B1):
- `categorical.log_prob` and `categorical.entropy` accept `check: bool = True` and forward it to
  `masked_logits(logits, mask, check=check)`.
- On valid masks, `check=False` is bit-identical to `check=True` for both outputs and gradients.
- Default `check=True` still raises `ValueError` on an all-false mask row; `check=False` does not.
- `PolicyNet.act` (both `stats=True` and `stats=False`) and `PolicyNet.evaluate` pass `check=False`
  on every `masked_logits` call, with bit-identical outputs and gradients compared to `check=True`.
"""

import numpy as np
import polytopia_sim as ps
import pytest
import torch

from polytopia_rl import categorical
from polytopia_rl.model import PolicyNet
from polytopia_rl.vec_env import SyncVecEnv

SIZE = 6
KW = dict(size=SIZE, opponent="random", agent_seat="seed_parity", turn_cap=4)


def _random_case(seed, b=48, k=13):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, size=(b, k)).astype(np.float32)
    mask = rng.random((b, k)) < 0.4
    mask[np.arange(b), rng.integers(k, size=b)] = True
    # Include a single-valid-option row and a row with a huge masked logit.
    mask[0, :] = False
    mask[0, 2] = True
    logits[1, :] = 0.0
    mask[1, :] = False
    mask[1, [0, 3]] = True
    logits[1, 1] = 1e4
    idx = np.array([rng.choice(np.flatnonzero(row)) for row in mask], dtype=np.int64)
    return torch.from_numpy(logits), torch.from_numpy(mask), torch.from_numpy(idx)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_log_prob_and_entropy_check_false_match_check_true_bit_for_bit(seed):
    logits, mask, idx = _random_case(seed)
    lp_true = categorical.log_prob(logits, mask, idx, check=True)
    lp_false = categorical.log_prob(logits, mask, idx, check=False)
    assert torch.equal(lp_false, lp_true)

    ent_true = categorical.entropy(logits, mask, check=True)
    ent_false = categorical.entropy(logits, mask, check=False)
    assert torch.equal(ent_false, ent_true)

    # Gradients must also match bit-for-bit.
    l1 = logits.clone().requires_grad_(True)
    l2 = logits.clone().requires_grad_(True)
    (categorical.log_prob(l1, mask, idx, check=True).sum() + categorical.entropy(l1, mask, check=True).sum()).backward()
    (categorical.log_prob(l2, mask, idx, check=False).sum() + categorical.entropy(l2, mask, check=False).sum()).backward()
    assert torch.equal(l1.grad, l2.grad)


def test_log_prob_and_entropy_default_check_raises_on_all_false_row_and_check_false_skips():
    logits = torch.zeros(2, 3)
    mask = torch.tensor([[True, False, True], [False, False, False]], dtype=torch.bool)
    idx = torch.tensor([0, 0], dtype=torch.int64)

    with pytest.raises(ValueError):
        categorical.log_prob(logits, mask, idx)
    with pytest.raises(ValueError):
        categorical.entropy(logits, mask)

    # check=False skips the sync/validation check instead of raising ValueError.
    lp = categorical.log_prob(logits, mask, idx, check=False)
    ent = categorical.entropy(logits, mask, check=False)
    assert lp.shape == (2,) and ent.shape == (2,)


def _real_batch(num_envs=6):
    vec = SyncVecEnv([lambda: ps.Env(**KW) for _ in range(num_envs)], seed_base=10)
    vec.reset_all()
    fetch = {"actor": vec.actor_masks, "kind": vec.kind_masks, "arg": vec.arg_masks}
    board, glob = (torch.from_numpy(x) for x in vec.observations())
    return board, glob, lambda stage, *a: fetch[stage](*a)


def test_policynet_act_and_evaluate_pass_check_false_and_match_reference(monkeypatch):
    spec = ps.Env(**KW).spec()
    torch.manual_seed(42)
    net = PolicyNet(spec, width=16, blocks=2)
    board, glob, masks_fn = _real_batch()

    real_masked_logits = categorical.masked_logits
    seen_checks: list[bool] = []

    def spy_masked_logits(logits, mask, check=True):
        seen_checks.append(check)
        return real_masked_logits(logits, mask, check=check)

    monkeypatch.setattr(categorical, "masked_logits", spy_masked_logits)

    # 1. PolicyNet.act(stats=True): every masked_logits call uses check=False (Clarification B1).
    seen_checks.clear()
    with torch.no_grad():
        feats = net.trunk(board, glob)
        act_fast = net.act(feats, masks_fn, torch.Generator().manual_seed(3), stats=True)
    assert len(seen_checks) >= 9, "3 stages x (sample + log_prob + entropy)"
    assert all(c is False for c in seen_checks), f"expected all check=False in act(stats=True), got {seen_checks}"

    # 2. PolicyNet.act(stats=False): every masked_logits call uses check=False.
    seen_checks.clear()
    with torch.no_grad():
        act_play = net.act(feats, masks_fn, torch.Generator().manual_seed(3), stats=False)
    assert len(seen_checks) == 3
    assert all(c is False for c in seen_checks), f"expected all check=False in act(stats=False), got {seen_checks}"
    assert torch.equal(act_play.actors, act_fast.actors)
    assert torch.equal(act_play.kinds, act_fast.kinds)
    assert torch.equal(act_play.args, act_fast.args)

    # 3. PolicyNet.evaluate(): every masked_logits call uses check=False.
    seen_checks.clear()
    net.zero_grad(set_to_none=True)
    ev_fast = net.evaluate(
        board, glob, act_fast.actors, act_fast.kinds, act_fast.args, act_fast.actor_mask, act_fast.kind_mask, act_fast.arg_mask
    )
    assert len(seen_checks) == 6, "3 stages x (log_prob + entropy)"
    assert all(c is False for c in seen_checks), f"expected all check=False in evaluate(), got {seen_checks}"
    (ev_fast.logp.sum() + ev_fast.entropy.sum() + ev_fast.value.sum()).backward()
    grads_fast = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}

    # 4. Reference with check=True forced on every call: bit-identical outputs and gradients.
    def force_check_true(logits, mask, check=True):
        return real_masked_logits(logits, mask, check=True)

    monkeypatch.setattr(categorical, "masked_logits", force_check_true)
    with torch.no_grad():
        act_ref = net.act(net.trunk(board, glob), masks_fn, torch.Generator().manual_seed(3), stats=True)
    assert torch.equal(act_fast.actors, act_ref.actors)
    assert torch.equal(act_fast.kinds, act_ref.kinds)
    assert torch.equal(act_fast.args, act_ref.args)
    assert torch.equal(act_fast.logp, act_ref.logp)
    assert torch.equal(act_fast.entropy, act_ref.entropy)
    assert torch.equal(act_fast.value, act_ref.value)

    net.zero_grad(set_to_none=True)
    ev_ref = net.evaluate(
        board, glob, act_fast.actors, act_fast.kinds, act_fast.args, act_fast.actor_mask, act_fast.kind_mask, act_fast.arg_mask
    )
    assert torch.equal(ev_fast.logp, ev_ref.logp)
    assert torch.equal(ev_fast.entropy, ev_ref.entropy)
    assert torch.equal(ev_fast.value, ev_ref.value)
    for head in ("actor", "kind", "arg"):
        assert torch.equal(ev_fast.head_entropy[head], ev_ref.head_entropy[head])
    (ev_ref.logp.sum() + ev_ref.entropy.sum() + ev_ref.value.sum()).backward()
    grads_ref = {name: p.grad.clone() for name, p in net.named_parameters() if p.grad is not None}
    assert grads_fast.keys() == grads_ref.keys() and grads_fast
    for name in grads_ref:
        assert torch.equal(grads_fast[name], grads_ref[name]), f"grad mismatch for {name}"
