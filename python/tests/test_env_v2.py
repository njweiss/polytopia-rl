"""Milestone 20 Phase A: `polytopia_sim.Env` under `ruleset="v2"` (docs/MILESTONE_20 "Python
`polytopia_sim` & RL end-to-end")."""

import json
import os
import pathlib
import subprocess

import numpy as np
import polytopia_sim as ps
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
OPPONENTS = ["random", "rush", "economy", "none"]

V1_UNIT_TYPES = ["warrior", "rider", "archer", "defender", "super_unit"]
V2_UNIT_TYPES = V1_UNIT_TYPES + ["swordsman", "catapult", "knight"]
V1_TECHS = [
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
V2_TECHS = V1_TECHS + [
    "roads",
    "free_spirit",
    "trade",
    "chivalry",
    "mathematics",
    "construction",
    "smithery",
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
V2_SITE_ACTIONS = [
    "harvest_animal",
    "harvest_fruit",
    "lumber_hut",
    "clear_forest",
    "farm",
    "mine",
    "burn_forest",
]
V2_BUILDINGS = ["sawmill", "windmill", "forge", "market"]
V2_MONUMENTS = ["park_of_fortune", "gate_of_power", "eye_of_god"]
V2_ACTION_TYPES = [
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
    "disband",
    "build_road",
    "build_building",
    "build_monument",
    "destroy",
]
V1_CHANNEL_NAMES = (
    ["explored"]
    + [f"terrain.{t}" for t in ["field", "forest", "mountain"]]
    + [f"resource.{r}" for r in ["fruit", "crop", "animal", "metal"]]
    + [f"improvement.{i}" for i in ["lumber_hut", "farm", "mine"]]
    + ["village", "territory.mine", "territory.enemy"]
    + [f"city.{c}" for c in ["mine", "enemy", "level", "capital"]]
    + [f"my_unit.{u}" for u in V1_UNIT_TYPES]
    + [f"enemy_unit.{u}" for u in V1_UNIT_TYPES]
    + ["health"]
    + [f"phase.{p}" for p in ["fresh", "moved", "attacked", "done"]]
    + ["veteran", "can_promote", "can_harvest", "can_build"]
)
V2_CHANNEL_NAMES = V1_CHANNEL_NAMES + [
    "my_unit.swordsman",
    "my_unit.catapult",
    "my_unit.knight",
    "enemy_unit.swordsman",
    "enemy_unit.catapult",
    "enemy_unit.knight",
    "road",
    "city.connected",
    "building.sawmill",
    "building.windmill",
    "building.forge",
    "building.market",
    "building.level",
    "lighthouse",
    "monument.park_of_fortune",
    "monument.gate_of_power",
    "monument.eye_of_god",
    "can_build_road",
    "can_build_building",
    "can_build_monument",
    "can_destroy",
    "preview.target_can_kill",
    "preview.target_max_dmg",
    "preview.target_min_retal",
    "preview.target_def_bonus",
    "preview.attacker_can_kill",
    "preview.attacker_max_dmg",
]
V2_GLOBAL_NAMES = (
    [
        "stars",
        "income",
        "turn",
        "pending_reward",
        "my_score",
        "opp_score",
        "score_diff",
        "my_cities",
        "my_units",
    ]
    + [f"tech.{t}" for t in V2_TECHS]
    + ["my_kills", "my_lighthouses"]
    + [f"monument_unlocked.{m}" for m in V2_MONUMENTS]
    + [f"monument_placed.{m}" for m in V2_MONUMENTS]
)


def v2_arg_lens(size=11):
    n = size * size
    return [n, n, 0, 0, 0, 8, 8, 7, 16, 0, 0, 0, 4, 3, 0]


def sample_factored(env, rng):
    actors = np.flatnonzero(env.actor_mask())
    actor = int(rng.choice(actors))
    kinds = np.flatnonzero(env.kind_mask(actor))
    kind = int(rng.choice(kinds))
    am = env.arg_mask(actor, kind)
    arg = int(rng.choice(np.flatnonzero(am))) if am.size else 0
    return actor, kind, arg


def polytopia_cli():
    exe = os.environ.get("POLYTOPIA_BIN")
    if exe:
        return [exe]
    return ["cargo", "run", "-q", "-p", "sim-tools", "--bin", "polytopia", "--"]


@pytest.mark.parametrize("size", [6, 11])
def test_v2_spec_and_shapes_and_v1_identity(size):
    v1 = ps.Env(size=size, ruleset="v1").spec()
    assert "buildings" not in v1 and "monuments" not in v1 and "ruleset" not in v1
    if size == 11:
        assert v1["obs_hash"] == "d5f9b1033cfd55ca"
        assert v1["rules_hash"] == "a8b52b91eae34503"

    env = ps.Env(size=size, opponent="rush", ruleset="v2")
    spec = env.spec()
    assert spec["height"] == size and spec["width"] == size
    assert spec["channels"] == 64 and spec["globals"] == 33
    assert list(spec["channel_names"]) == V2_CHANNEL_NAMES
    assert list(spec["global_names"]) == V2_GLOBAL_NAMES
    assert list(spec["action_types"]) == V2_ACTION_TYPES
    assert spec["num_actors"] == 3 * size * size + 1
    assert list(spec["arg_lens"]) == v2_arg_lens(size)
    assert list(spec["unit_types"]) == V2_UNIT_TYPES
    assert list(spec["rewards"]) == REWARDS
    assert list(spec["techs"]) == V2_TECHS
    assert list(spec["site_actions"]) == V2_SITE_ACTIONS
    assert list(spec["buildings"]) == V2_BUILDINGS
    assert list(spec["monuments"]) == V2_MONUMENTS
    assert spec["ruleset"] == "v2"
    assert spec["obs_hash"] != v1["obs_hash"]
    assert spec["rules_hash"] != v1["rules_hash"]

    env.reset(3)
    board, glob = env.observation()
    assert board.shape == (64, size, size) and board.dtype == np.float32 and board.flags["C_CONTIGUOUS"]
    assert glob.shape == (33,) and glob.dtype == np.float32 and glob.flags["C_CONTIGUOUS"]
    am = env.actor_mask()
    assert am.shape == (3 * size * size + 1,) and am.dtype == np.bool_ and am.flags["C_CONTIGUOUS"]
    actor = int(np.flatnonzero(am)[0])
    km = env.kind_mask(actor)
    assert km.shape == (15,) and km.dtype == np.bool_ and km.flags["C_CONTIGUOUS"]
    for kind, length in enumerate(v2_arg_lens(size)):
        m = env.arg_mask(actor, kind)
        assert m.shape == (length,) and m.dtype == np.bool_ and m.flags["C_CONTIGUOUS"]


def test_v2_factored_sampling_finishes_50_games():
    rng = np.random.default_rng(2026)
    for game in range(50):
        env = ps.Env(
            opponent=OPPONENTS[game % 4],
            agent_seat="seed_parity",
            turn_cap=12,
            ruleset="v2",
        )
        env.reset(game)
        while True:
            actor, kind, arg = sample_factored(env, rng)
            out = env.step_factored(actor, kind, arg)
            if out[2] or out[3]:
                break
        assert out[2] and not out[3], f"v2 game {game} truncated"
        assert env.legal_count() == 0
        assert not env.actor_mask().any()


@pytest.mark.parametrize("bot_name", ["random", "rush", "economy"])
def test_v2_bot_action_and_replay_verifies_with_cli(tmp_path, bot_name):
    env = ps.Env(size=8, opponent="rush", opponent_epsilon=0.1, agent_seat=0, turn_cap=8, ruleset="v2")
    env.reset(5)
    while True:
        actor, kind, arg = env.bot_action(bot_name)
        assert 0 <= kind < 15
        out = env.step_factored(actor, kind, arg)
        if out[2] or out[3]:
            break
    assert out[2] and not out[3]
    text = env.replay_json()
    log = json.loads(text)
    assert log["config"]["ruleset"] == "v2"
    path = tmp_path / f"v2_{bot_name}.json"
    path.write_text(text)
    r = subprocess.run(polytopia_cli() + ["verify", str(path)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    info = env.info()
    outcome = {"win_0": "Win(0)", "win_1": "Win(1)", "draw": "Draw"}[info["outcome"]]
    assert r.stdout.strip() == f"OK: {len(log['actions'])} actions, outcome {outcome}"


def test_v2_and_v1_kind_bounds_raise_value_error():
    env_v2 = ps.Env(size=6, opponent="none", ruleset="v2")
    env_v2.reset(0)
    global_actor = 3 * 36
    with pytest.raises(ValueError):
        env_v2.step_factored(global_actor, 15, 0)
    with pytest.raises(ValueError):
        env_v2.arg_mask(global_actor, 15)
    with pytest.raises(ValueError):
        env_v2.step_factored(global_actor + 1, 9, 0)

    # Under v1, kind 10..14 (v2 kinds) MUST still raise ValueError.
    env_v1 = ps.Env(size=6, opponent="none", ruleset="v1")
    env_v1.reset(0)
    for bad_kind in range(10, 15):
        with pytest.raises(ValueError):
            env_v1.step_factored(global_actor, bad_kind, 0)
        with pytest.raises(ValueError):
            env_v1.arg_mask(global_actor, bad_kind)
