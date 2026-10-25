"""Milestone 12 tests 3 and 8: seed-varied training maps, the in-training batched suite subset and the
rating gap ("Seed-varied training maps", "In scope" 1 and 3).

Expected seed bases from the brief: `train_seed_base(s) = (s - 1) * 10_000_000` for 1 <= s <= 100,
so seed 1 keeps base 0 and seed 100 starts at 990,000,000 (below the 1e9 evaluation range). The vec
env's episode seeds follow the M10 formula `seed_base + i + k * num_envs` (stride 1).

The overlap test shrinks `train.TRAIN_SEEDS_PER_RUN` to 6 seeds per run: with 4 envs, seed 2 owns
seeds 6..11, the first resets use 6..9, and the third episode of env 0 would need seed 14 (> 11), so a
tiny run (dozens of episodes) must raise. The rating test replaces `train.fit_ratings` with fixed
ratings `R0 + STEP * i` per player index; in `learner_ratings` the learner is player 0 and snapshot 0
(the first pick) is player 1, so `gap_snapshot_0 = -STEP`.
"""

import json
import pathlib

import numpy as np
import pytest
import torch

from polytopia_rl import train
from polytopia_rl.config import PPOConfig
from polytopia_rl.vec_env import TRAIN_SEED_LIMIT

BATCH = 128
UPDATES = 4
R0, STEP = 921.278, 61.5


def tiny_cfg(**kw):
    base = dict(
        size=6,
        turn_cap=3,
        num_envs=4,
        num_steps=32,
        num_minibatches=4,
        update_epochs=2,
        width=8,
        blocks=1,
        total_steps=UPDATES * BATCH,
        eval_every=2 * BATCH,
        eval_seeds=1,
        device="cpu",
        seed=2,
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    return PPOConfig(**{**base, **kw})


def sp_cfg(**kw):
    sp = dict(
        seed=3,
        opponents=["economy", "rush"],
        stages=[{"opponents": ["economy", "rush"]}],
        selfplay=True,
        snapshot_every=2 * BATCH,
        rating_every=4 * BATCH,
        rating_seeds=1,
        tournament_seeds=1,
        p_bot=0.5,
    )
    return tiny_cfg(**{**sp, **kw})


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def jsonl(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines()]


def check_seeds(state, base, num_envs=4):
    assert state["seed_base"] == base
    for i in range(num_envs):
        k = state["next_k"][i] - 1  # the episode in progress
        assert state["seeds"][i] == base + i + k * num_envs, i
        assert base <= state["seeds"][i] < base + train.TRAIN_SEEDS_PER_RUN


# ---------------------------------------------------------------- train_seed_base


def test_train_seed_base_values():
    assert train.TRAIN_SEEDS_PER_RUN == 10_000_000
    assert train.train_seed_base(1) == 0
    assert train.train_seed_base(2) == 10_000_000
    assert train.train_seed_base(3) == 20_000_000
    assert train.train_seed_base(100) == 990_000_000
    assert train.train_seed_base(100) + train.TRAIN_SEEDS_PER_RUN <= TRAIN_SEED_LIMIT


@pytest.mark.parametrize("seed", [0, -1, 101, 1000])
def test_train_seed_base_rejects_out_of_range_seeds(seed):
    with pytest.raises(ValueError):
        train.train_seed_base(seed)


# ---------------------------------------------------------------- the trainer


@pytest.fixture(scope="module")
def run2(tmp_path_factory):
    d = tmp_path_factory.mktemp("seed2")
    train.run(tiny_cfg(), str(d))
    return d


def test_trainer_starts_at_the_seed_base_and_seed_1_keeps_base_0(run2, tmp_path):
    for name in ["ckpt_256.pt", "latest.pt"]:
        check_seeds(load(run2 / name)["vec_env"], 10_000_000)
    train.run(tiny_cfg(seed=1, total_steps=BATCH, eval_every=BATCH), str(tmp_path))
    check_seeds(load(tmp_path / "latest.pt")["vec_env"], 0)


def test_a_run_that_would_cross_into_the_next_base_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "TRAIN_SEEDS_PER_RUN", 6)
    assert train.train_seed_base(2) == 6
    with pytest.raises(ValueError):
        train.run(tiny_cfg(seed=2, eval_every=10**9), str(tmp_path))


def test_resume_keeps_the_stored_seed_base(run2, tmp_path):
    # Move the stored base: a resume must keep it, not recompute it from the seed.
    ck = load(run2 / "ckpt_256.pt")
    assert ck["vec_env"]["seed_base"] == 10_000_000
    moved = 10_000_000 + 5_000
    ck["vec_env"]["seed_base"] = moved
    torch.save(ck, tmp_path / "moved.pt")
    out = tmp_path / "run"
    train.run(tiny_cfg(), str(out), resume=str(tmp_path / "moved.pt"))
    state = load(out / "latest.pt")["vec_env"]
    assert state["seed_base"] == moved
    old = ck["vec_env"]
    for i in range(4):
        if state["next_k"][i] > old["next_k"][i]:  # a new episode started after the resume
            assert state["seeds"][i] == moved + i + (state["next_k"][i] - 1) * 4


# ---------------------------------------------------------------- self-play: seeds, batched subset, rating gap


def fixed_ratings(wins, games, anchor, anchor_rating=0.0):
    return R0 + STEP * np.arange(len(wins), dtype=np.float64)


@pytest.fixture(scope="module")
def sp_runs(tmp_path_factory):
    mocked = tmp_path_factory.mktemp("sp_mocked")
    plain = tmp_path_factory.mktemp("sp_plain")
    calls = []
    real = train.run_suite

    def spy(*args, **kwargs):
        calls.append(kwargs.get("make_batch_fn"))
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "run_suite", spy)
        mp.setattr(train, "fit_ratings", fixed_ratings)
        train.run(sp_cfg(), str(mocked))
    train.run(sp_cfg(), str(plain))
    return mocked, plain, calls


def test_selfplay_trainer_starts_at_the_seed_base(sp_runs):
    mocked, plain, _ = sp_runs
    state = load(plain / "ckpt_final.pt")["vec_env"]
    assert state["seed_base"] == 20_000_000
    for i in range(4):
        assert 20_000_000 <= state["seeds"][i] < 30_000_000


def test_in_training_suite_subset_plays_batched(sp_runs):
    _, _, calls = sp_runs
    assert len(calls) == 2  # suite evaluations at 256 and 512
    assert all(c is not None for c in calls)
    assert all(callable(c(True)) for c in calls)


def test_rating_row_has_the_gap_to_snapshot_0(sp_runs):
    mocked, plain, _ = sp_runs
    rows = [r for r in jsonl(mocked / "eval.jsonl") if r["type"] == "rating"]
    assert [r["global_step"] for r in rows] == [512]
    r = rows[0]
    assert r["rating"]["learner"] == pytest.approx(R0)
    assert r["rating"]["snapshot_0"] == pytest.approx(R0 + STEP)
    assert r["gap_snapshot_0"] == pytest.approx(-STEP)
    for r in (x for x in jsonl(plain / "eval.jsonl") if x["type"] == "rating"):
        assert r["gap_snapshot_0"] == pytest.approx(r["rating"]["learner"] - r["rating"]["snapshot_0"])
