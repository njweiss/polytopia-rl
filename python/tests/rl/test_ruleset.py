"""Milestone 15 tests 1 and 7 (Python part): `polytopia_sim.Env` refuses `ruleset="v2"` with a
`ValueError` naming milestone 20, and v1 is byte-identical (docs/MILESTONE_15 "Env", "Hashes").
The `ruleset` keyword (default "v1") is docs/QUESTIONS.md Q44.

The recorded hashes are the brief's values for the current data (also docs/RL_REPORT.md).
"""

import numpy as np
import polytopia_sim as ps
import pytest


def test_v2_is_refused_with_a_value_error_naming_milestone_20():
    with pytest.raises(ValueError, match="milestone 20"):
        ps.Env(size=6, ruleset="v2")
    with pytest.raises(ValueError, match="milestone 20"):
        ps.Env(size=11, opponent="none", ruleset="v2")


def test_an_unknown_ruleset_is_a_value_error():
    for bad in ("v3", "V1", ""):
        with pytest.raises(ValueError):
            ps.Env(size=6, ruleset=bad)


def test_v1_hashes_are_the_recorded_values():
    spec = ps.Env(size=11).spec()
    assert spec["obs_hash"] == "d5f9b1033cfd55ca"
    assert spec["rules_hash"] == "a8b52b91eae34503"
    assert ps.Env(size=11, ruleset="v1").spec() == spec


def test_explicit_v1_plays_the_same_game_as_the_default():
    a, b = ps.Env(size=8, turn_cap=3), ps.Env(size=8, turn_cap=3, ruleset="v1")
    for env in (a, b):
        env.reset(4)
    for _ in range(60):
        oa, ob = a.observation(), b.observation()
        assert np.array_equal(oa[0], ob[0]) and np.array_equal(oa[1], ob[1])
        assert a.info() == b.info()
        if a.info()["outcome"] is not None:
            break
        step = a.bot_action("economy")
        assert b.bot_action("economy") == step
        ra, rb = a.step_factored(*step), b.step_factored(*step)
        assert ra[0] == rb[0] and np.array_equal(ra[1], rb[1]) and ra[2:] == rb[2:]
        if ra[2] or ra[3]:
            break
    assert a.replay_json() == b.replay_json()
    assert "ruleset" not in a.replay_json()
