"""Milestone 10 tests 7: the real rollout + `train_update` pipeline learns on a `FakeEnv`, and
`collect_rollout` stores the truncation bootstrap.

The FakeEnv (rl_helpers) pays +1 only for one factored path at its choice step. With a uniform
initial policy the target's probability is 1/6 (actor 1/3, kind 1/2) or 1/18 (actor 1/3, kind 1/2,
argument 1/3); a working pipeline drives it above 0.9 within the update budget.
"""

import numpy as np
import pytest
import torch
from rl_helpers import ATTACK, CAPTURE, END_TURN, MOVE, TRAIN, FakeEnv

from polytopia_rl.config import PPOConfig
from polytopia_rl.model import PolicyNet
from polytopia_rl.ppo import RolloutBuffer, collect_rollout, train_update
from polytopia_rl.vec_env import SyncVecEnv

N = 16
G = 3 * N

# Two-stage choice (kind AND argument): Attack on 9 and Move to 2 or 10 pay nothing.
KIND_AND_ARG = {(5, MOVE): [2, 9, 10], (5, ATTACK): [9, 10], (N + 3, TRAIN): [0, 1, 2], (G, END_TURN): [0]}
# Actor-and-kind choice with an argument-less target.
ACTOR_AND_KIND = {(5, CAPTURE): [0], (5, MOVE): [1, 2], (N + 3, TRAIN): [0, 1], (G, END_TURN): [0]}

CASES = [
    ("capture", ACTOR_AND_KIND, (5, CAPTURE, 0)),
    ("move_9", KIND_AND_ARG, (5, MOVE, 9)),
    ("train_2", KIND_AND_ARG, (N + 3, TRAIN, 2)),
]


def smoke_cfg():
    return PPOConfig(
        num_envs=8,
        num_steps=16,
        num_minibatches=4,
        update_epochs=4,
        lr=1e-3,
        ent_coef=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        width=16,
        blocks=1,
        device="cpu",
    )


@pytest.mark.parametrize("name,legal,target", CASES, ids=[c[0] for c in CASES])
def test_ppo_learns_the_paying_path(name, legal, target):
    cfg = smoke_cfg()
    torch.manual_seed(0)
    make = lambda: FakeEnv(legal=legal, target=target, pre_steps=1)  # noqa: E731
    vec = SyncVecEnv([make for _ in range(cfg.num_envs)], seed_base=0)
    vec.reset_all()
    spec = make().spec()
    net = PolicyNet(spec, width=cfg.width, blocks=cfg.blocks)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    buf = RolloutBuffer(cfg.num_steps, cfg.num_envs, spec)
    gen = torch.Generator().manual_seed(0)
    rates = []
    for _ in range(30):
        stats = collect_rollout(net, vec, buf, gen)
        rates.append(np.mean([s.ret for s in stats]))
        if rates[-1] >= 0.95:
            break
        train_update(net, opt, buf, cfg, gen)
    assert rates[0] < 0.5, f"{name}: the untrained policy should rarely find the target ({rates[0]})"
    assert rates[-1] >= 0.95, f"{name}: success rates {[round(r, 2) for r in rates]}"
    # The greedy choice at the choice step is the target path.
    env = make()
    env.reset(0)
    env.step_factored(G, END_TURN, 0)
    board, glob = env.observation()
    fetch = {
        "actor": lambda: env.actor_mask()[None],
        "kind": lambda a: env.kind_mask(int(a[0]))[None],
        "arg": lambda a, k: np.pad(env.arg_mask(int(a[0]), int(k[0])), (0, N - env.arg_mask(int(a[0]), int(k[0])).size))[None],
    }
    with torch.no_grad():
        out = net.act(
            net.trunk(torch.from_numpy(board)[None], torch.from_numpy(glob)[None]),
            lambda stage, *a: fetch[stage](*a),
            gen,
            greedy=True,
        )
    assert (out.actors.item(), out.kinds.item(), out.args.item()) == target


def test_collect_rollout_stores_truncation_bootstrap_and_last_value():
    # Only EndTurn is ever legal; each episode is truncated after 3 steps, so with 7 steps the
    # step counters are 0,1,2 | 0,1,2 | 0 and the current observation at the end has t = 1.
    num_envs, num_steps = 2, 7
    make = lambda: FakeEnv(pre_steps=10, truncate_at=3)  # noqa: E731
    vec = SyncVecEnv([make for _ in range(num_envs)], seed_base=4)  # seeds 4, 5 | 6, 7 | 8, 9
    vec.reset_all()
    spec = make().spec()
    torch.manual_seed(0)
    net = PolicyNet(spec, width=8, blocks=1)
    buf = RolloutBuffer(num_steps, num_envs, spec)
    stats = collect_rollout(net, vec, buf, torch.Generator().manual_seed(0))
    assert [(s.env_index, s.seed, s.length) for s in stats] == [(0, 4, 3), (1, 5, 3), (0, 6, 3), (1, 7, 3)]

    def value_of(t, seed):
        board = np.zeros((1, 2, 4, 4), dtype=np.float32)
        board[0, 0] = t
        board[0, 1] = seed % 2
        glob = np.array([[t, seed % 2, 1.0]], dtype=np.float32)
        with torch.no_grad():
            return net.value(net.trunk(torch.from_numpy(board), torch.from_numpy(glob))).item()

    trunc = torch.as_tensor(buf.truncated).bool()
    expected = torch.zeros(num_steps, num_envs, dtype=torch.bool)
    expected[[2, 5]] = True
    assert torch.equal(trunc, expected)
    assert not torch.as_tensor(buf.terminated).bool().any()
    seeds = [[4, 5], [6, 7], [8, 9]]
    for t in range(num_steps):
        for e in range(num_envs):
            seed = seeds[t // 3][e]
            assert buf.values[t, e].item() == pytest.approx(value_of(t % 3, seed), abs=1e-5)
            assert buf.actors[t, e].item() == G and buf.kinds[t, e].item() == END_TURN
            assert buf.logp[t, e].item() == pytest.approx(0.0, abs=1e-6)
    for t in (2, 5):
        for e in range(num_envs):
            # The final observation of a truncated episode has t = 3 (after its third step).
            assert buf.next_values[t, e].item() == pytest.approx(value_of(3, seeds[t // 3][e]), abs=1e-5)
    for e in range(num_envs):
        assert buf.last_value[e].item() == pytest.approx(value_of(1, seeds[2][e]), abs=1e-5)
