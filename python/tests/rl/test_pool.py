"""Milestone 11 tests 8: snapshot pool and matchmaking ("Self-play": snapshots, opponent per
episode, snapshot opponents act), plus the regression flag ("Monitoring").

Thinning (docs/QUESTIONS.md Q25): `add` appends, then while there are more than `pool_max`
entries keep positions 0, 2, 4, ... and the newest.

Sampling frequencies, with K entries: `sample` gives the newest with p_latest + (1 - p_latest) / K
and every other entry (1 - p_latest) / K; `draw_opponent` gives each of economy and rush p_bot / 2
and scales the snapshot probabilities by 1 - p_bot. Tolerances are about 5 standard deviations of
a frequency over n draws (sd = sqrt(p (1 - p) / n)).
"""

import numpy as np
import polytopia_sim as ps
import pytest
import torch

from polytopia_rl.evaluate import net_policy
from polytopia_rl.model import PolicyNet
from polytopia_rl.selfplay import BOT_OPPONENTS, Entry, SnapshotPool, draw_opponent, net_batch_policy, regression

SIZE = 6


def tiny_net(seed=0):
    torch.manual_seed(seed)
    return PolicyNet(ps.Env(size=SIZE).spec(), width=8, blocks=1)


def state(x):
    return {"w": torch.full((2,), float(x))}


def steps(pool):
    return [e.step for e in pool.entries]


def test_add_records_step_name_and_order():
    pool = SnapshotPool(pool_max=10)
    e = pool.add(state(0), 0)
    assert isinstance(e, Entry) and e.step == 0 and e.name == "snap_0"
    pool.add(state(1), 1000)
    assert steps(pool) == [0, 1000]
    assert [en.name for en in pool.entries] == ["snap_0", "snap_1000"]
    assert torch.equal(pool.entries[1].state["w"], torch.full((2,), 1.0))


def test_thin_keeps_even_positions_and_the_newest():
    pool = SnapshotPool(pool_max=100)
    for s in range(6):
        pool.add(state(s), s)
    pool.thin()
    assert steps(pool) == [0, 2, 4, 5]
    pool = SnapshotPool(pool_max=100)
    for s in range(7):
        pool.add(state(s), s)
    pool.thin()
    assert steps(pool) == [0, 2, 4, 6]


def test_capacity_thinning_keeps_first_and_newest():
    pool = SnapshotPool(pool_max=4)
    for s in range(4):
        pool.add(state(s), s)
    assert steps(pool) == [0, 1, 2, 3]
    pool.add(state(4), 4)  # 5 > 4: keep positions 0, 2, 4
    assert steps(pool) == [0, 2, 4]
    pool.add(state(5), 5)
    pool.add(state(6), 6)  # [0, 2, 4, 5, 6] -> [0, 4, 6]
    assert steps(pool) == [0, 4, 6]
    for s in range(7, 40):
        pool.add(state(s), s)
        assert len(pool.entries) <= 4
        assert pool.entries[0].step == 0 and pool.entries[-1].step == s
        assert steps(pool) == sorted(steps(pool))


def test_snapshots_are_frozen_copies():
    net = tiny_net()
    pool = SnapshotPool()
    pool.add(net.state_dict(), 0)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    opt.zero_grad()
    sum(p.sum() for p in net.parameters()).backward()
    opt.step()
    with torch.no_grad():
        next(net.parameters()).add_(1.0)
    snap = pool.entries[0].state
    assert any(not torch.equal(v, net.state_dict()[k]) for k, v in before.items())
    for k, v in before.items():
        assert torch.equal(snap[k], v), k


N = 100_000


def test_sample_frequencies():
    pool = SnapshotPool(p_latest=0.6)
    for s in range(4):
        pool.add(state(s), s)
    rng = np.random.default_rng(0)
    counts = np.zeros(4)
    for _ in range(N):
        counts[pool.sample(rng).step] += 1
    freq = counts / N
    # newest 0.6 + 0.4 / 4 = 0.7 (sd 0.0014), others 0.1 (sd 0.00095)
    np.testing.assert_allclose(freq, [0.1, 0.1, 0.1, 0.7], atol=0.006)


def test_draw_opponent_frequencies():
    pool = SnapshotPool(p_latest=0.6)
    for s in range(4):
        pool.add(state(s), s)
    assert BOT_OPPONENTS == ("economy", "rush")
    rng = np.random.default_rng(1)
    counts = {"economy": 0, "rush": 0, 0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(N):
        o = draw_opponent(pool, rng, p_bot=0.3)
        counts[o if isinstance(o, str) else o.step] += 1
    freq = {k: v / N for k, v in counts.items()}
    # bots 0.15 each (sd 0.0011); newest 0.7 * 0.7 = 0.49 (sd 0.0016); others 0.7 * 0.1 = 0.07.
    want = {"economy": 0.15, "rush": 0.15, 0: 0.07, 1: 0.07, 2: 0.07, 3: 0.49}
    for k in want:
        assert freq[k] == pytest.approx(want[k], abs=0.007), (k, freq)


def test_draws_are_reproducible_from_the_rng():
    pool = SnapshotPool()
    for s in range(5):
        pool.add(state(s), s)

    def seq(seed):
        rng = np.random.default_rng(seed)
        return [(o if isinstance(o, str) else o.name) for o in (draw_opponent(pool, rng, 0.2) for _ in range(50))]

    assert seq(3) == seq(3)
    assert seq(3) != seq(4)


def envs_at(seeds, opponent="none"):
    out = []
    for s in seeds:
        env = ps.Env(size=SIZE, turn_cap=5, opponent=opponent)
        env.reset(s)
        out.append(env)
    return out


def test_net_batch_policy_greedy_equals_net_policy_in_one_forward_pass():
    net = tiny_net(1)
    envs = envs_at([3, 4, 5, 6])
    calls = []
    trunk = net.trunk
    net.trunk = lambda b, g: (calls.append(b.shape[0]), trunk(b, g))[1]
    acts = net_batch_policy(net, torch.Generator().manual_seed(0), greedy=True)(envs)
    assert calls == [4], "one batched trunk call for the four envs"
    net.trunk = trunk
    single = net_policy(net)
    assert [tuple(map(int, a)) for a in acts] == [single(env, True) for env in envs]


def test_net_batch_policy_samples_unless_greedy():
    net = tiny_net(2)
    envs = envs_at(range(20, 32))
    greedy = [tuple(map(int, a)) for a in net_batch_policy(net, torch.Generator().manual_seed(0), greedy=True)(envs)]
    g2 = [tuple(map(int, a)) for a in net_batch_policy(net, torch.Generator().manual_seed(9), greedy=True)(envs)]
    assert greedy == g2, "greedy ignores the generator"
    s1 = [tuple(map(int, a)) for a in net_batch_policy(net, torch.Generator().manual_seed(5))(envs)]
    s2 = [tuple(map(int, a)) for a in net_batch_policy(net, torch.Generator().manual_seed(5))(envs)]
    s3 = [tuple(map(int, a)) for a in net_batch_policy(net, torch.Generator().manual_seed(6))(envs)]
    assert s1 == s2, "sampling is reproducible from the generator"
    # An untrained net is close to uniform over dozens of legal actions: sampled play differs.
    assert s1 != greedy and s1 != s3
    for env, (a, k, x) in zip(envs, s1):
        assert env.actor_mask()[a] and env.kind_mask(a)[k]


def test_regression_flag():
    assert not regression({"economy_e0": 0.95, "rush_e0": 0.91, "rush_e20": 0.90}, 0.90)
    assert regression({"economy_e0": 0.95, "rush_e0": 0.89, "rush_e20": 0.99}, 0.90)
    assert not regression({"economy_e0": 0.0}, 0.0)
    assert regression({"economy_e0": 1.0}, 1.01)
