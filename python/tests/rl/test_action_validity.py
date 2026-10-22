"""Milestone 10 tests 4: every path sampled through the net is legal on real Envs (spec invariant
"Every legal-action path through the factored heads is a legal action in the sim")."""

import numpy as np
import polytopia_sim as ps
import torch
from rl_helpers import NO_ARG_KINDS

from polytopia_rl.model import PolicyNet
from polytopia_rl.vec_env import SyncVecEnv

NUM_ENVS = 16
PATHS = 2000


def test_sampled_paths_always_step_without_error():
    # Default map size and turn cap, as in training. A ValueError from `step` fails the test.
    kw = dict(opponent="random", agent_seat="seed_parity")
    spec = ps.Env(**kw).spec()
    pending = spec["global_names"].index("pending_reward")
    torch.manual_seed(0)
    net = PolicyNet(spec, width=8, blocks=1)
    vec = SyncVecEnv([lambda: ps.Env(**kw) for _ in range(NUM_ENVS)], seed_base=1000)
    vec.reset_all()
    gen = torch.Generator().manual_seed(0)
    fetch = {"actor": vec.actor_masks, "kind": vec.kind_masks, "arg": vec.arg_masks}
    pending_paths = 0
    paths = 0
    while paths < PATHS:
        board, glob = vec.observations()
        with torch.no_grad():
            out = net.act(
                net.trunk(torch.from_numpy(board), torch.from_numpy(glob)),
                lambda stage, *a: fetch[stage](*a),
                gen,
            )
        a, k, x = out.actors.numpy(), out.kinds.numpy(), out.args.numpy()
        # No sampled stage ever picks a masked option (masks re-fetched from the env).
        b = np.arange(NUM_ENVS)
        assert vec.actor_masks()[b, a].all()
        assert vec.kind_masks(a)[b, k].all()
        xm = vec.arg_masks(a, k)
        fired = ~np.isin(k, NO_ARG_KINDS)
        assert xm[b[fired], x[fired]].all()
        assert (x[~fired] == 0).all()
        pending_paths += int((glob[:, pending] == 1).sum())
        vec.step(a, k, x)
        paths += NUM_ENVS
    assert pending_paths > 0, "no pending-reward state was reached"
