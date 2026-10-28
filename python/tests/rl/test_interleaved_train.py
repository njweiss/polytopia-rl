"""Milestone 13 tests 4 and 5: the `rollout` config key and the trainer's choice of collector
("In scope" 3), and an interleaved self-play run that is deterministic and resumes exactly.

The collector spies wrap `polytopia_rl.ppo.collect_rollout` / `collect_rollout_interleaved` and the
names `polytopia_rl.train` may have imported from it. The switch tests stop the run at the first
rollout (the spy raises `Stop`), so they cost one env reset.
"""

import json
import pathlib

import pytest
import torch

from polytopia_rl import ppo, train
from polytopia_rl.config import PPOConfig, load_config

BATCH = 4 * 16
UPDATES = 4


def tiny_cfg(**kw):
    base = dict(
        size=6,
        turn_cap=3,
        num_envs=4,
        num_steps=16,
        num_minibatches=2,
        update_epochs=1,
        width=8,
        blocks=1,
        total_steps=UPDATES * BATCH,
        eval_every=2 * BATCH,
        eval_seeds=1,
        device="cpu",
        seed=3,
        opponents=["economy", "rush"],
        stages=[{"opponents": ["economy", "rush"]}],
    )
    return PPOConfig(**{**base, **kw})


def sp_cfg(**kw):
    sp = dict(selfplay=True, snapshot_every=2 * BATCH, rating_every=100 * BATCH, rating_seeds=1, tournament_seeds=1, p_bot=0.5)
    return tiny_cfg(**{**sp, **kw})


# ---------------------------------------------------------------- config


def test_rollout_key_defaults_to_interleaved_and_rejects_unknown_values(tmp_path):
    assert PPOConfig().rollout == "interleaved"
    assert PPOConfig.from_json("{}").rollout == "interleaved"
    assert PPOConfig.from_json('{"rollout": "lockstep"}').rollout == "lockstep"
    assert PPOConfig.from_json('{"rollout": "interleaved"}').rollout == "interleaved"
    for bad in ("Lockstep", "async", "", None, 1):
        with pytest.raises(ValueError):
            PPOConfig.from_json(json.dumps({"rollout": bad}))
    path = tmp_path / "c.json"
    path.write_text("{}")
    assert load_config(str(path), ["rollout=lockstep"]).rollout == "lockstep"
    with pytest.raises(ValueError):
        load_config(str(path), ["rollout=sideways"])


# ---------------------------------------------------------------- trainer switch


class Stop(Exception):
    pass


def spy(monkeypatch, stop: bool):
    """Replace both collectors (in `ppo` and in `train`); returns the list of collector names called."""
    called = []
    originals = {name: getattr(ppo, name) for name in ("collect_rollout", "collect_rollout_interleaved")}

    def make(name):
        def wrapper(*a, **kw):
            called.append(name)
            if stop:
                raise Stop
            return originals[name](*a, **kw)

        return wrapper

    for name in originals:
        w = make(name)
        monkeypatch.setattr(ppo, name, w)
        monkeypatch.setattr(train, name, w, raising=False)
    return called


CASES = {
    "selfplay-default": (sp_cfg(), "collect_rollout_interleaved"),
    "selfplay-interleaved": (sp_cfg(rollout="interleaved"), "collect_rollout_interleaved"),
    "selfplay-lockstep": (sp_cfg(rollout="lockstep"), "collect_rollout"),
    "bots-default": (tiny_cfg(), "collect_rollout"),  # bot training never uses the interleaved collector
    "bots-interleaved": (tiny_cfg(rollout="interleaved"), "collect_rollout"),
    "bots-lockstep": (tiny_cfg(rollout="lockstep"), "collect_rollout"),
}


def test_trainer_picks_the_collector_from_the_config(monkeypatch, tmp_path):
    got = {}
    for name, (cfg, _) in CASES.items():
        with monkeypatch.context() as mp:
            called = spy(mp, stop=True)
            with pytest.raises(Stop):
                train.run(cfg, str(tmp_path / name))
            got[name] = called
    assert got == {name: [want] for name, (_, want) in CASES.items()}


# ---------------------------------------------------------------- determinism and resume


def jsonl(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines()]


def untimed(row):
    return {k: v for k, v in row.items() if "sps" not in k and "time" not in k}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        called = spy(mp, stop=False)
        d = tmp_path_factory.mktemp("il")
        train.run(sp_cfg(), str(d / "a"))
        n_a = called.count("collect_rollout_interleaved")
        train.run(sp_cfg(), str(d / "b"))
        train.run(sp_cfg(), str(d / "r"), resume=str(d / "a" / f"ckpt_{2 * BATCH}.pt"))
        return d, n_a, list(called)
    finally:
        mp.undo()


def test_interleaved_selfplay_runs_use_the_interleaved_collector(runs):
    _, n_a, called = runs
    assert n_a == UPDATES
    assert set(called) == {"collect_rollout_interleaved"}
    assert len(called) == 2 * UPDATES + UPDATES // 2


def test_two_identical_interleaved_runs_are_identical(runs):
    d, _, called = runs
    assert set(called) == {"collect_rollout_interleaved"}
    a, b = jsonl(d / "a" / "metrics.jsonl"), jsonl(d / "b" / "metrics.jsonl")
    assert len(a) == UPDATES
    assert [untimed(r) for r in a] == [untimed(r) for r in b]
    assert same_state(load(d / "a" / "ckpt_final.pt")["model"], load(d / "b" / "ckpt_final.pt")["model"])


def test_an_interleaved_run_resumed_at_a_rollout_boundary_continues_exactly(runs):
    d, _, called = runs
    assert set(called) == {"collect_rollout_interleaved"}
    rows = jsonl(d / "r" / "metrics.jsonl")
    assert [r["update"] for r in rows] == [3, 4]
    assert [untimed(r) for r in rows] == [untimed(r) for r in jsonl(d / "a" / "metrics.jsonl")[2:]]
    assert same_state(load(d / "r" / "ckpt_final.pt")["model"], load(d / "a" / "ckpt_final.pt")["model"])
