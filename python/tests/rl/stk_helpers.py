"""Shared helpers for the milestone 13b tests (one stacked forward for all snapshot opponents).

The reference for every stacked result is each snapshot's own `PolicyNet` (its `trunk`, then
`actor_logits`, `kind_logits`, `arg_logits`) or `net_batch_policy` per group; nothing here calls the
code under test to compute an expectation.

- `random_net`: a tiny `PolicyNet` whose every parameter is redrawn N(0, std^2) from its own seed, so
  snapshots differ clearly and the heads' logits are spread enough (the default init gives the heads
  gain 0.01, which leaves top-two gaps far below the float tolerance).
- `positions`: real two-seat env positions (tiny map) after a few uniformly random legal actions.
- `ref_stage_logits`: one snapshot's logits per stage for a group, conditioned on given actors/kinds.
- `SampleSpy`: records every `categorical.sample` call (logits, mask, greedy, check, result).
- `MaskEnv`: a `FakeEnv` with exactly one legal path, that records every mask request and fails if
  it is stepped.
"""

import numpy as np
import polytopia_sim as ps
import torch
from rl_helpers import FakeEnv, random_path

from polytopia_rl import categorical
from polytopia_rl.model import PolicyNet
from polytopia_rl.selfplay import Entry

SIZE = 6
WIDTH, BLOCKS = 8, 1
STD = 0.2


def real_spec():
    return ps.Env(size=SIZE).spec()


def random_net(seed, spec=None, std=STD):
    spec = spec or real_spec()
    net = PolicyNet(spec, width=WIDTH, blocks=BLOCKS)
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return net


def entry_of(net, step):
    """A pool entry holding a frozen copy of `net`'s weights."""
    return Entry(step, {k: v.detach().clone() for k, v in net.state_dict().items()}, f"snap_{step}")


def positions(n, seed, max_steps=10):
    """`n` two-seat envs (size 6) on seeds `seed * 1000 + j`, each after 0..max_steps-1 random legal
    actions; none of them finished."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        env = ps.Env(size=SIZE, opponent="none", agent_seat="seed_parity")
        env.reset(seed * 1000 + j)
        for _ in range(int(rng.integers(max_steps))):
            _, _, term, trunc = env.step_factored(*random_path(env, rng))
            assert not (term or trunc)
        out.append(env)
    return out


def obs_tensors(envs):
    obs = [env.observation() for env in envs]
    return torch.from_numpy(np.stack([b for b, _ in obs])), torch.from_numpy(np.stack([g for _, g in obs]))


def padded_arg_mask(env, actor, kind, n):
    m = np.zeros(n, dtype=bool)
    xm = env.arg_mask(int(actor), int(kind))
    m[: xm.size] = xm
    return m


def ref_stage_logits(net, envs, actors=None, kinds=None):
    """`(actor_logits, kind_logits | None, arg_logits | None)` of `net` for `envs`, the kind stage
    conditioned on `actors`, the argument stage on `actors` and `kinds`."""
    board, glob = obs_tensors(envs)
    with torch.no_grad():
        feats = net.trunk(board, glob)
        a = net.actor_logits(feats)
        k = x = None
        if actors is not None:
            k = net.kind_logits(feats, torch.as_tensor(actors, dtype=torch.int64))
        if kinds is not None:
            x = net.arg_logits(feats, torch.as_tensor(actors, dtype=torch.int64), torch.as_tensor(kinds, dtype=torch.int64))
    return a, k, x


def has_arg(spec):
    return torch.tensor([n > 0 for n in spec["arg_lens"]])


def top_two_gap(logits, mask):
    """The gap between the two largest logits among the true entries of `mask` (inf with one)."""
    v = np.sort(np.asarray(logits, dtype=np.float64)[np.asarray(mask, dtype=bool)])
    return np.inf if v.size < 2 else float(v[-1] - v[-2])


def greedy_gaps(net, envs):
    """The smallest top-two gap over the three stages of `net`'s greedy path for each env (the
    argument stage only when the chosen kind has an argument)."""
    n = envs[0].spec()["height"] * envs[0].spec()["width"]
    a_logits, _, _ = ref_stage_logits(net, envs)
    a_masks = [env.actor_mask() for env in envs]
    actors = [int(np.argmax(np.where(m, a_logits[i].numpy(), -np.inf))) for i, m in enumerate(a_masks)]
    _, k_logits, _ = ref_stage_logits(net, envs, actors)
    k_masks = [env.kind_mask(a) for env, a in zip(envs, actors)]
    kinds = [int(np.argmax(np.where(m, k_logits[i].numpy(), -np.inf))) for i, m in enumerate(k_masks)]
    _, _, x_logits = ref_stage_logits(net, envs, actors, kinds)
    fired = has_arg(envs[0].spec())
    gaps = []
    for i, env in enumerate(envs):
        g = min(top_two_gap(a_logits[i], a_masks[i]), top_two_gap(k_logits[i], k_masks[i]))
        if fired[kinds[i]]:
            g = min(g, top_two_gap(x_logits[i], padded_arg_mask(env, actors[i], kinds[i], n)))
        gaps.append(g)
    return gaps


class SampleSpy:
    """Wraps `categorical.sample` (the module attribute, and the name `polytopia_rl.stacked` may have
    imported) and records each call; `calls` can be cleared between policy calls."""

    def __init__(self, monkeypatch):
        import polytopia_rl.stacked as stacked

        self.calls = []
        original = categorical.sample

        def spy(logits, mask, generator, greedy=False, check=True):
            out = original(logits, mask, generator, greedy, check)
            self.calls.append(
                dict(logits=logits.detach().clone(), mask=torch.as_tensor(mask).clone(), greedy=greedy, check=check, out=out.clone(), generator=generator)
            )
            return out

        monkeypatch.setattr(categorical, "sample", spy)
        monkeypatch.setattr(stacked, "sample", spy, raising=False)


class MaskEnv(FakeEnv):
    """The choice step of a `FakeEnv` whose only legal path is `path`; records every mask request as
    `("actor",)`, `("kind", actor)`, `("arg", actor, kind)`, and raises if stepped or reset."""

    def __init__(self, path, seed):
        a, k, x = path
        super().__init__(legal={(a, k): [x]}, pre_steps=0)
        super().reset(seed)
        self.path = path
        self.requests = []

    def actor_mask(self):
        self.requests.append(("actor",))
        return super().actor_mask()

    def kind_mask(self, actor):
        self.requests.append(("kind", int(actor)))
        return super().kind_mask(actor)

    def arg_mask(self, actor, kind):
        self.requests.append(("arg", int(actor), int(kind)))
        return super().arg_mask(actor, kind)

    def step_factored(self, *a):
        raise AssertionError("a policy stepped an env")

    def reset(self, seed):
        raise AssertionError("a policy reset an env")
