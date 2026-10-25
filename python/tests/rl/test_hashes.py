"""Milestone 11 tests 1 (Python part): `Env.spec()` exposes `obs_hash` and `rules_hash`, and the
`Env.bot_action` test hook (docs/QUESTIONS.md Q23).

Expected hashes come from an independent FNV-1a over the brief's canonical string (built here from
the spec's own name lists, docs/QUESTIONS.md Q22) and over the `data/*.json` files read from disk.
"""

import re

import numpy as np
import polytopia_sim as ps
import pytest
from rl_helpers import END_TURN
from sp_helpers import hex16, obs_canonical_from_spec, rules_bytes

HEX16 = re.compile(r"^[0-9a-f]{16}$")


@pytest.mark.parametrize("size", [6, 8, 11])
def test_spec_hashes_match_an_independent_fnv(size):
    spec = ps.Env(size=size).spec()
    assert HEX16.match(spec["obs_hash"]), spec["obs_hash"]
    assert HEX16.match(spec["rules_hash"]), spec["rules_hash"]
    assert spec["obs_hash"] == hex16(obs_canonical_from_spec(spec, size).encode())
    assert spec["rules_hash"] == hex16(rules_bytes())


def test_obs_hash_depends_on_size_and_rules_hash_does_not():
    a, b = ps.Env(size=6).spec(), ps.Env(size=11).spec()
    assert a["obs_hash"] != b["obs_hash"]
    assert a["rules_hash"] == b["rules_hash"]
    # Opponent, seat and rule switches that are not part of the observation leave both unchanged.
    c = ps.Env(size=11, opponent="none", turn_cap=7, full_observability=True).spec()
    assert (c["obs_hash"], c["rules_hash"]) == (b["obs_hash"], b["rules_hash"])


def test_bot_action_returns_a_legal_path():
    env = ps.Env(size=8, opponent="none", turn_cap=4)
    env.reset(3)
    for _ in range(40):
        if env.info()["outcome"] is not None:
            break
        for name in ("random", "economy", "rush"):
            a, k, x = env.bot_action(name)
            assert env.actor_mask()[a] and env.kind_mask(a)[k]
            xm = env.arg_mask(a, k)
            assert (xm.size == 0 and x == 0) or xm[x]
        env.step_factored(*env.bot_action("economy"))
    with pytest.raises(ValueError):
        env.bot_action("nobody")


def test_bot_action_after_the_end_is_a_runtime_error():
    env = ps.Env(size=6, opponent="none", turn_cap=1)
    env.reset(1)
    done = False
    while not done:
        am = env.actor_mask()
        g = len(am) - 1
        _, _, done, trunc = env.step_factored(g, END_TURN, 0)
        done = done or trunc
    with pytest.raises(RuntimeError):
        env.bot_action("rush")
    assert np.all(~env.actor_mask())
