"""Milestone 12 tests 1-2: the batched `evaluate` path and the batched suite ("Batched evaluation").

The scripted policies are pure functions of (seed, seat, agent actions so far), so the sequential and
the batched path must produce the same games: greedy batched results equal the sequential ones game
for game. Their batch form is `[policy(env, greedy) for env in envs]`. Expected round structure,
from the brief: every game is built first (seed order, seat 0 first); round r makes one `batch_fn`
call over the games still running, in game order, and steps each once; a game of length L therefore
appears in rounds 0 .. L - 1 and nowhere else.
"""

import polytopia_sim as ps
import pytest
import torch
from sp_helpers import end_turn_policy, keyed_random_policy

from polytopia_rl import suite, train
from polytopia_rl.checkpoint import save_checkpoint
from polytopia_rl.config import PPOConfig
from polytopia_rl.evaluate import EVAL_SEED_BASE, evaluate, net_policy
from polytopia_rl.model import PolicyNet
from polytopia_rl.selfplay import net_batch_policy
from polytopia_rl.suite import DEFAULT_SUITE, SUITE_SEED_BASE, run_suite

KW = dict(size=8, turn_cap=6)


def batch_of(policy, greedy=True):
    return lambda envs: [policy(env, greedy) for env in envs]


def no_policy(env, greedy):
    raise AssertionError("policy_fn must be ignored when batch_fn is given")


def mixed_policy(env, greedy):
    """EndTurn only on even seeds (short games), the keyed random policy on odd seeds."""
    if env.info()["seed"] % 2 == 0:
        return end_turn_policy(env, greedy)
    return keyed_random_policy(salt=3)(env, greedy)


@pytest.mark.parametrize(
    "opponent,epsilon,seed_base",
    [("random", 0.0, EVAL_SEED_BASE), ("economy", 0.0, EVAL_SEED_BASE), ("rush", 0.0, EVAL_SEED_BASE), ("rush", 0.2, 123_456)],
)
def test_greedy_batched_equals_sequential_game_for_game(opponent, epsilon, seed_base):
    policy = keyed_random_policy()
    seq = evaluate(policy, opponent, 3, KW, True, seed_base=seed_base, epsilon=epsilon)
    bat = evaluate(no_policy, opponent, 3, KW, True, seed_base=seed_base, epsilon=epsilon, batch_fn=batch_of(policy))
    assert [(g["seed"], g["seat"]) for g in bat.per_game] == [(seed_base + j, s) for j in range(3) for s in (0, 1)]
    assert bat.per_game == seq.per_game
    assert bat == seq


class RoundRecorder:
    """Batch form of `mixed_policy` that records, per call, which games it was asked to move."""

    def __init__(self):
        self.calls = []

    def __call__(self, envs):
        keys = [(env.info()["seed"], env.info()["agent_seat"]) for env in envs]
        assert len({id(env) for env in envs}) == len(envs), "one env per game"
        self.calls.append(keys)
        return [mixed_policy(env, True) for env in envs]


def test_one_batch_call_per_round_over_exactly_the_live_games():
    seeds = 4
    rec = RoundRecorder()
    res = evaluate(no_policy, "rush", seeds, KW, True, batch_fn=rec)
    order = [(EVAL_SEED_BASE + j, s) for j in range(seeds) for s in (0, 1)]
    assert [(g["seed"], g["seat"]) for g in res.per_game] == order
    length = {(g["seed"], g["seat"]): g["length"] for g in res.per_game}
    assert len(set(length.values())) > 1, "the games must end at different rounds for this test"
    assert len(rec.calls) == max(length.values())
    for r, keys in enumerate(rec.calls):
        assert keys == [k for k in order if length[k] > r], f"round {r}"
    # The same games as the sequential path.
    seq = evaluate(mixed_policy, "rush", seeds, KW, True)
    assert res.per_game == seq.per_game


def test_finished_games_leave_the_round_set_early():
    # Even seeds only end their turn; with turn cap 6 those games are over after at most 6 agent
    # actions while the odd seeds' games go on.
    rec = RoundRecorder()
    res = evaluate(no_policy, "random", 4, KW, True, batch_fn=rec)
    short = {(g["seed"], g["seat"]) for g in res.per_game if g["seed"] % 2 == 0}
    assert all(g["length"] <= KW["turn_cap"] for g in res.per_game if (g["seed"], g["seat"]) in short)
    late = rec.calls[KW["turn_cap"] :]
    assert late, "the odd seeds' games are longer than the turn cap in agent actions"
    for keys in late:
        assert not short & set(keys)


def tiny_net(seed=0):
    torch.manual_seed(seed)
    return PolicyNet(ps.Env(size=8).spec(), width=8, blocks=1)


def test_greedy_net_batched_equals_sequential_and_sampled_is_repeatable():
    net = tiny_net()
    kw = dict(size=8, turn_cap=4)
    seq = evaluate(net_policy(net), "rush", 3, kw, True)
    bat = evaluate(no_policy, "rush", 3, kw, True, batch_fn=net_batch_policy(net, torch.Generator(), greedy=True))
    assert bat.per_game == seq.per_game
    a = evaluate(no_policy, "random", 3, kw, False, batch_fn=net_batch_policy(net, torch.Generator().manual_seed(5), greedy=False))
    b = evaluate(no_policy, "random", 3, kw, False, batch_fn=net_batch_policy(net, torch.Generator().manual_seed(5), greedy=False))
    assert a.per_game == b.per_game
    assert [(g["seed"], g["seat"]) for g in a.per_game] == [(EVAL_SEED_BASE + j, s) for j in range(3) for s in (0, 1)]


# ---------------------------------------------------------------- suite


def test_make_batch_fn_gets_each_conditions_greedy_flag_and_greedy_results_match():
    policy = keyed_random_policy(salt=1)
    flags = []

    def make_batch_fn(greedy):
        flags.append(greedy)
        return batch_of(policy, greedy)

    bat = run_suite(no_policy, DEFAULT_SUITE, 2, KW, step=7, make_batch_fn=make_batch_fn)
    assert flags == [c.greedy for c in DEFAULT_SUITE]
    seq = run_suite(policy, DEFAULT_SUITE, 2, KW, step=7)
    assert list(bat.conditions) == [c.name for c in DEFAULT_SUITE]
    assert (bat.obs_hash, bat.rules_hash, bat.step) == (seq.obs_hash, seq.rules_hash, seq.step)
    for c in DEFAULT_SUITE:
        if c.greedy:
            assert bat.conditions[c.name] == seq.conditions[c.name], c.name
        pg = bat.conditions[c.name].per_game
        assert [(g["seed"], g["seat"]) for g in pg] == [(SUITE_SEED_BASE + j, s) for j in range(2) for s in (0, 1)]


def test_suite_cli_plays_batched_with_net_batch_policy(tmp_path, monkeypatch):
    cfg = PPOConfig(size=6, turn_cap=3, width=8, blocks=1, device="cpu")
    torch.manual_seed(2)
    net = PolicyNet(ps.Env(size=6).spec(), width=8, blocks=1)
    path = tmp_path / "ck.pt"
    save_checkpoint(str(path), model=net, optimizer=None, cfg=cfg, global_step=64, stage=0, seed_index=None, rng_state=None, spec=ps.Env(size=6).spec())
    seen = []
    real = suite.run_suite

    def spy(*args, **kwargs):
        seen.append(kwargs.get("make_batch_fn"))
        return real(*args, **kwargs)

    monkeypatch.setattr(suite, "run_suite", spy)
    monkeypatch.setattr(train, "resolve_device", lambda device: "cpu")
    suite.main(["--checkpoint", str(path), "--seeds", "1", "--out", str(tmp_path / "out")])
    assert len(seen) == 1 and seen[0] is not None, "the suite CLI passes make_batch_fn"
    assert (tmp_path / "out" / "suite.json").exists()
    # The batch function is the net's batched greedy policy: same actions as net_batch_policy.
    envs = []
    for s in range(3):
        env = ps.Env(size=6, turn_cap=3, opponent="rush", agent_seat=s % 2)
        env.reset(SUITE_SEED_BASE + s)
        envs.append(env)
    want = net_batch_policy(net, torch.Generator(), greedy=True)(envs)
    got = seen[0](True)(envs)
    assert [tuple(int(x) for x in a) for a in got] == [tuple(int(x) for x in a) for a in want]
    assert callable(seen[0](False))
