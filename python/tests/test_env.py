"""Milestone 9 "Tests" item 10: the Python layer of `polytopia_sim`.

Expected values come from docs/MILESTONE_9 (shapes from the channel and global tables, head
sizes from "Factored action space", names as documented in docs/QUESTIONS.md Q15), and from
the spec's starting score of 515 per player.
"""

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

import polytopia_sim as ps

REPO = pathlib.Path(__file__).resolve().parents[2]
OPPONENTS = ["random", "rush", "economy", "none"]

UNIT_TYPES = ["warrior", "rider", "archer", "defender", "super_unit"]
TECHS = [
    "climbing",
    "hunting",
    "organization",
    "riding",
    "archery",
    "forestry",
    "farming",
    "strategy",
    "mining",
]
REWARDS = [
    "workshop",
    "explorer",
    "city_wall",
    "resources",
    "population_growth",
    "border_growth",
    "park",
    "super_unit",
]
SITE_ACTIONS = ["harvest_animal", "harvest_fruit", "lumber_hut", "clear_forest", "farm", "mine"]
ACTION_TYPES = [
    "move",
    "attack",
    "capture",
    "recover",
    "promote",
    "train",
    "choose_reward",
    "site",
    "research",
    "end_turn",
]
CHANNEL_NAMES = (
    ["explored"]
    + [f"terrain.{t}" for t in ["field", "forest", "mountain"]]
    + [f"resource.{r}" for r in ["fruit", "crop", "animal", "metal"]]
    + [f"improvement.{i}" for i in ["lumber_hut", "farm", "mine"]]
    + ["village", "territory.mine", "territory.enemy"]
    + [f"city.{c}" for c in ["mine", "enemy", "level", "capital"]]
    + [f"my_unit.{u}" for u in UNIT_TYPES]
    + [f"enemy_unit.{u}" for u in UNIT_TYPES]
    + ["health"]
    + [f"phase.{p}" for p in ["fresh", "moved", "attacked", "done"]]
    + ["veteran", "can_promote", "can_harvest", "can_build"]
)
GLOBAL_NAMES = [
    "stars",
    "income",
    "turn",
    "pending_reward",
    "my_score",
    "opp_score",
    "score_diff",
    "my_cities",
    "my_units",
] + [f"tech.{t}" for t in TECHS]
C, G = 37, 18
RESEARCH, END_TURN = 8, 9


def make(opponent="rush", agent_seat=0, **kw):
    return ps.Env(opponent=opponent, agent_seat=agent_seat, **kw)


def n_tiles(size=11):
    return size * size


def arg_lens(size=11):
    n = n_tiles(size)
    return [n, n, 0, 0, 0, 5, 8, 6, 9, 0]


def sample_factored(env, rng):
    """Samples actor, kind and argument from the masks, as a learner would."""
    actors = np.flatnonzero(env.actor_mask())
    actor = int(rng.choice(actors))
    kinds = np.flatnonzero(env.kind_mask(actor))
    kind = int(rng.choice(kinds))
    am = env.arg_mask(actor, kind)
    arg = int(rng.choice(np.flatnonzero(am))) if am.size else 0
    return actor, kind, arg


def play_random(env, seed, rng):
    """Random legal-index agent until the episode ends; returns the last StepResult."""
    env.reset(seed)
    while True:
        out = env.step_legal(int(rng.integers(env.legal_count())))
        if out[2] or out[3]:
            return out


def snapshot(env):
    board, glob = env.observation()
    return (board.tobytes(), glob.tobytes(), env.actor_mask().tobytes(), env.legal_count(), env.info())


@pytest.mark.parametrize("opponent", OPPONENTS)
def test_shapes_dtypes_and_contiguity(opponent):
    env = make(opponent)
    env.reset(3)
    board, glob = env.observation()
    assert board.shape == (C, 11, 11) and board.dtype == np.float32
    assert glob.shape == (G,) and glob.dtype == np.float32
    assert board.flags["C_CONTIGUOUS"] and glob.flags["C_CONTIGUOUS"]
    am = env.actor_mask()
    assert am.shape == (3 * 121 + 1,) and am.dtype == np.bool_ and am.flags["C_CONTIGUOUS"]
    actor = int(np.flatnonzero(am)[0])
    km = env.kind_mask(actor)
    assert km.shape == (10,) and km.dtype == np.bool_
    for kind, length in enumerate(arg_lens()):
        m = env.arg_mask(actor, kind)
        assert m.shape == (length,) and m.dtype == np.bool_ and m.flags["C_CONTIGUOUS"]
    reward, rewards, terminated, truncated = env.step_legal(0)
    assert isinstance(reward, float)
    assert isinstance(rewards, np.ndarray) and rewards.shape == (2,) and rewards.dtype == np.float32
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_arrays_are_fresh_per_call():
    env = make("rush")
    env.reset(1)
    board, glob = env.observation()
    keep_b, keep_g = board.copy(), glob.copy()
    board[...] = 9.0
    glob[...] = 9.0
    am = env.actor_mask()
    am[...] = True
    b2, g2 = env.observation()
    assert np.array_equal(b2, keep_b) and np.array_equal(g2, keep_g)
    assert env.actor_mask().sum() < am.size


def test_map_size_sets_the_shapes():
    env = make("none", size=8)
    env.reset(0)
    board, glob = env.observation()
    assert board.shape == (C, 8, 8)
    assert env.actor_mask().shape == (3 * 64 + 1,)
    spec = env.spec()
    assert (spec["height"], spec["width"], spec["num_actors"]) == (8, 8, 193)
    assert list(spec["arg_lens"]) == arg_lens(8)


def test_spec_matches_the_rust_layout():
    env = make("rush")
    spec = env.spec()
    assert spec["height"] == 11 and spec["width"] == 11
    assert spec["channels"] == C and spec["globals"] == G
    assert list(spec["channel_names"]) == CHANNEL_NAMES
    assert list(spec["global_names"]) == GLOBAL_NAMES
    assert list(spec["action_types"]) == ACTION_TYPES
    assert spec["num_actors"] == 364
    assert list(spec["arg_lens"]) == arg_lens()
    assert list(spec["unit_types"]) == UNIT_TYPES
    assert list(spec["rewards"]) == REWARDS
    assert list(spec["techs"]) == TECHS
    assert list(spec["site_actions"]) == SITE_ACTIONS


def test_starting_observation_and_info():
    env = make("rush", agent_seat=0)
    env.reset(4)
    info = env.info()
    assert info["seed"] == 4
    assert info["turn"] == 0
    assert info["current_player"] == 0
    assert info["observer"] == 0
    assert info["agent_seat"] == 0
    assert tuple(info["scores"]) == (515, 515)
    assert info["outcome"] is None
    assert info["agent_actions"] == 0
    _, glob = env.observation()
    # stars 5/50, income 2/20, turn 0, no pending, 515/2000 twice, diff 0, 1/8 cities,
    # 1/16 units, Organization only.
    want = [0.1, 0.1, 0.0, 0.0, 0.2575, 0.2575, 0.0, 0.125, 0.0625] + [
        1.0 if t == "organization" else 0.0 for t in TECHS
    ]
    np.testing.assert_allclose(glob, np.array(want, dtype=np.float32), atol=1e-6)
    env.step_legal(0)
    assert env.info()["agent_actions"] == 1


def test_seed_parity_and_bot_mode_seats():
    for seed in range(6):
        env = make("economy", agent_seat="seed_parity")
        env.reset(seed)
        info = env.info()
        assert info["agent_seat"] == seed % 2
        assert info["current_player"] == seed % 2
        assert info["observer"] == seed % 2
    env = make("economy", agent_seat=1)
    env.reset(0)
    assert env.info()["current_player"] == 1


def test_two_seat_observer_follows_the_mover():
    env = make("none")
    env.reset(2)
    assert env.info()["observer"] == 0
    env.step_factored(363, END_TURN, 0)
    info = env.info()
    assert info["observer"] == 1 and info["current_player"] == 1
    assert info["agent_actions"] == 1


def test_reward_parameters_reach_the_env():
    # Two-seat, both at 515; researching Climbing (5 stars, +100 score) gives
    # shaping 1 x (0.9 x 100/1000 - 0) = 0.09 to the acting player 0.
    env = make("none", shaping_weight=1.0, gamma=0.9)
    env.reset(6)
    reward, rewards, terminated, truncated = env.step_factored(363, RESEARCH, 0)
    assert reward == pytest.approx(0.09, abs=1e-6)
    np.testing.assert_allclose(rewards, [0.09, -0.09], atol=1e-6)
    assert not terminated and not truncated
    assert tuple(env.info()["scores"]) == (615, 515)


@pytest.mark.parametrize("opponent", OPPONENTS)
def test_random_agent_finishes_200_games(opponent):
    env = make(opponent, agent_seat="seed_parity")
    rng = np.random.default_rng(1234)
    for game in range(200):
        reward, rewards, terminated, truncated = play_random(env, game, rng)
        info = env.info()
        assert terminated and not truncated, f"game {game}"
        assert info["outcome"] in ("win_0", "win_1", "draw")
        seat = info["agent_seat"] if opponent != "none" else info["observer"]
        if info["outcome"] == "draw":
            assert reward == 0.0
        else:
            assert reward == (1.0 if info["outcome"] == f"win_{seat}" else -1.0)
        assert env.legal_count() == 0
        assert not env.actor_mask().any()


def test_error_types():
    env = make("rush")
    with pytest.raises(RuntimeError):
        env.step_legal(0)
    with pytest.raises(RuntimeError):
        env.step_factored(363, END_TURN, 0)
    env.reset(0)
    before = snapshot(env)
    with pytest.raises(ValueError):
        env.step_legal(env.legal_count())
    with pytest.raises(ValueError):
        env.step_factored(363, 0, 0)  # Move on the global actor: decode error
    with pytest.raises(ValueError):
        env.step_factored(363, END_TURN, 1)  # nonzero argument
    with pytest.raises(ValueError):
        env.step_factored(363, RESEARCH, 8)  # Mining without Climbing: illegal
    with pytest.raises(ValueError):
        env.step_factored(364, END_TURN, 0)  # actor out of range
    with pytest.raises(ValueError):
        env.step_factored(363, 10, 0)  # no such kind
    assert snapshot(env) == before, "a failed call changed the env"
    play_random(env, 0, np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        env.step_legal(0)
    with pytest.raises(RuntimeError):
        env.step_factored(363, END_TURN, 0)


def test_truncation():
    env = make("economy", max_agent_actions=3)
    env.reset(0)
    assert env.step_factored(363, END_TURN, 0)[3] is False
    assert env.step_factored(363, END_TURN, 0)[3] is False
    _, _, terminated, truncated = env.step_factored(363, END_TURN, 0)
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step_legal(0)


def episode(env, seed, policy_seed):
    rng = np.random.default_rng(policy_seed)
    env.reset(seed)
    trace = []
    while True:
        board, glob = env.observation()
        out = env.step_legal(int(rng.integers(env.legal_count())))
        trace.append((board.tobytes(), glob.tobytes(), out[0], out[1].tobytes(), out[2], out[3]))
        if out[2] or out[3]:
            return trace, env.info(), env.replay_json()


@pytest.mark.parametrize("opponent", ["rush", "none"])
def test_seed_determinism_across_two_envs(opponent):
    a = make(opponent, opponent_epsilon=0.3, turn_cap=12)
    b = make(opponent, opponent_epsilon=0.3, turn_cap=12)
    for seed in range(3):
        assert episode(a, seed, 77) == episode(b, seed, 77)


def polytopia_cli():
    exe = os.environ.get("POLYTOPIA_BIN")
    if exe:
        return [exe]
    return ["cargo", "run", "-q", "-p", "sim-tools", "--bin", "polytopia", "--"]


@pytest.mark.parametrize("opponent,seat", [("rush", 1), ("random", 0), ("none", 0)])
def test_replay_json_verifies_with_the_cli(tmp_path, opponent, seat):
    env = make(opponent, agent_seat=seat, turn_cap=10, opponent_epsilon=0.2)
    play_random(env, 5, np.random.default_rng(5))
    info = env.info()
    text = env.replay_json()
    log = json.loads(text)
    assert log["seed"] == 5
    assert log["config"]["turn_cap"] == 10
    n = len(log["actions"])
    if opponent == "none":
        assert n == info["agent_actions"]
    else:
        assert n > info["agent_actions"], "the replay must include the opponent's actions"
    path = tmp_path / "game.json"
    path.write_text(text)
    r = subprocess.run(polytopia_cli() + ["verify", str(path)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    outcome = {"win_0": "Win(0)", "win_1": "Win(1)", "draw": "Draw"}[info["outcome"]]
    assert r.stdout.strip() == f"OK: {n} actions, outcome {outcome}"


def test_replay_json_mid_episode_verifies(tmp_path):
    env = make("economy", agent_seat=1)
    env.reset(8)
    rng = np.random.default_rng(8)
    for _ in range(10):
        env.step_legal(int(rng.integers(env.legal_count())))
    assert env.info()["outcome"] is None, "test premise: the game is still running"
    text = env.replay_json()
    n = len(json.loads(text)["actions"])
    assert n > 10
    path = tmp_path / "partial.json"
    path.write_text(text)
    r = subprocess.run(polytopia_cli() + ["verify", str(path)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"OK: {n} actions, outcome unfinished"


def test_factored_sampling_always_steps():
    rng = np.random.default_rng(42)
    for game in range(50):
        env = make(OPPONENTS[game % 4], agent_seat="seed_parity", turn_cap=15)
        env.reset(game)
        while True:
            actor, kind, arg = sample_factored(env, rng)
            out = env.step_factored(actor, kind, arg)
            if out[2] or out[3]:
                break
        assert out[2] and not out[3]
