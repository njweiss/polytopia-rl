"""Milestone 10 tests 3: PolicyNet structure ("Policy (spec: factored heads, four actor slots)").

The layout tests perturb the trunk's feature map `F` at one tile while holding the context vector
`g` fixed, so each head's dependence on `F` is visible exactly: the actor head is a 1x1 conv of F
(+ one logit from g), the kind head reads F at the actor's tile only, and the tile argument head is
a 1x1 conv of F dotted with a context vector.
"""

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from rl_helpers import (
    ATTACK,
    CAPTURE,
    END_TURN,
    MOVE,
    NO_ARG_KINDS,
    RESEARCH,
    TRAIN,
    np_masked_entropy,
    np_masked_logp,
)

from polytopia_rl.model import Features, PolicyNet
from polytopia_rl.vec_env import SyncVecEnv

SIZE = 6
N = SIZE * SIZE
G_ACTOR = 3 * N
KW = dict(size=SIZE, opponent="random", agent_seat="seed_parity", turn_cap=4)


def spec():
    return ps.Env(**KW).spec()


def make_net(seed=0, width=16, blocks=2):
    torch.manual_seed(seed)
    return PolicyNet(spec(), width=width, blocks=blocks)


def real_obs(b=4, base=0):
    boards, globs = [], []
    for s in range(b):
        e = ps.Env(**KW)
        e.reset(base + s)
        x, y = e.observation()
        boards.append(x)
        globs.append(y)
    return torch.from_numpy(np.stack(boards)), torch.from_numpy(np.stack(globs))


def feats_of(net, b=4):
    with torch.no_grad():
        return net.trunk(*real_obs(b))


def perturbed(feats, tile, scale=1.0):
    fmap = feats.fmap.clone()
    r, c = divmod(tile, SIZE)
    gen = torch.Generator().manual_seed(tile)
    fmap[:, :, r, c] += scale * torch.randn(fmap.shape[0], fmap.shape[1], generator=gen)
    return Features(fmap=fmap, g=feats.g)


def changed(a, b):
    """Boolean per column: did any row change?"""
    return (a != b).any(dim=0)


def full(b, v):
    return torch.full((b,), v, dtype=torch.int64)


def test_output_shapes():
    net = make_net()
    board, glob = real_obs(5)
    with torch.no_grad():
        f = net.trunk(board, glob)
        assert f.fmap.shape == (5, 16, SIZE, SIZE)
        assert f.g.dim() == 2 and f.g.shape[0] == 5
        assert net.actor_logits(f).shape == (5, 3 * N + 1)
        assert net.kind_logits(f, full(5, G_ACTOR)).shape == (5, 10)
        assert net.arg_logits(f, full(5, 3), full(5, MOVE)).shape == (5, N)
        assert net.arg_logits(f, full(5, N + 3), full(5, TRAIN)).shape == (5, N)
        assert net.value(f).shape == (5,)


def test_actor_index_layout_is_family_major_row_major():
    net = make_net()
    f = feats_of(net)
    base = net.actor_logits(f)
    for tile in (0, 10, 2 * SIZE + 5, N - 1):  # (1, 4) and (2, 5) are not symmetric
        with torch.no_grad():
            ch = changed(base, net.actor_logits(perturbed(f, tile)))
        expected = torch.zeros(3 * N + 1, dtype=torch.bool)
        expected[[tile, N + tile, 2 * N + tile]] = True
        assert torch.equal(ch, expected), f"tile {tile}: changed {torch.nonzero(ch).flatten().tolist()}"
    # The unit and city slots of the same tile are different logits.
    assert not torch.equal(base[:, 10], base[:, N + 10])


def test_global_actor_logit_comes_from_g_only():
    net = make_net()
    f = feats_of(net)
    with torch.no_grad():
        base = net.actor_logits(f)
        g2 = Features(fmap=f.fmap, g=f.g + torch.randn(f.g.shape, generator=torch.Generator().manual_seed(1)))
        ch = changed(base, net.actor_logits(g2))
    expected = torch.zeros(3 * N + 1, dtype=torch.bool)
    expected[G_ACTOR] = True
    assert torch.equal(ch, expected)


def test_kind_logits_conditioned_on_actor():
    net = make_net()
    f = feats_of(net)
    t = 2 * SIZE + 5
    with torch.no_grad():
        rows = [net.kind_logits(f, full(4, a)) for a in (t, N + t, 2 * N + t, G_ACTOR, t - 1)]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            assert not torch.equal(rows[i], rows[j]), f"actor choices {i} and {j} give the same kind logits"


@pytest.mark.parametrize("fam", [0, 1, 2])
def test_kind_logits_read_the_feature_map_at_the_actor_tile_only(fam):
    net = make_net()
    f = feats_of(net)
    t = 2 * SIZE + 5
    actors = full(4, fam * N + t)
    with torch.no_grad():
        base = net.kind_logits(f, actors)
        assert not torch.equal(base, net.kind_logits(perturbed(f, t), actors))
        for u in (0, 7, t - 1, t + SIZE, N - 1):
            assert torch.equal(base, net.kind_logits(perturbed(f, u), actors)), f"tile {u} leaked in"


def test_global_actor_kind_logits_ignore_the_feature_map():
    net = make_net()
    f = feats_of(net)
    with torch.no_grad():
        base = net.kind_logits(f, full(4, G_ACTOR))
        for u in (0, 13, N - 1):
            assert torch.equal(base, net.kind_logits(perturbed(f, u), full(4, G_ACTOR)))


@pytest.mark.parametrize("kind", [MOVE, ATTACK])
def test_tile_argument_logits_layout(kind):
    # Actor at t; perturbing F at u != t changes exactly argument logit u.
    net = make_net()
    f = feats_of(net)
    t = 9
    actors, kinds = full(4, t), full(4, kind)
    with torch.no_grad():
        base = net.arg_logits(f, actors, kinds)
        for u in (0, 2 * SIZE + 5, N - 1):
            ch = changed(base, net.arg_logits(perturbed(f, u), actors, kinds))
            expected = torch.zeros(N, dtype=torch.bool)
            expected[u] = True
            assert torch.equal(ch, expected), f"tile {u}: changed {torch.nonzero(ch).flatten().tolist()}"


def test_argument_logits_depend_on_kind_and_actor():
    net = make_net()
    f = feats_of(net)
    with torch.no_grad():
        move = net.arg_logits(f, full(4, 9), full(4, MOVE))
        attack = net.arg_logits(f, full(4, 9), full(4, ATTACK))
        move_other_actor = net.arg_logits(f, full(4, 10), full(4, MOVE))
        assert not torch.equal(move, attack)
        assert not torch.equal(move[:, :9], move_other_actor[:, :9])
        train_a = net.arg_logits(f, full(4, N + 9), full(4, TRAIN))[:, :5]
        train_b = net.arg_logits(f, full(4, N + 10), full(4, TRAIN))[:, :5]
        assert not torch.equal(train_a, train_b)
        # An enum head reads F at the actor's tile only (through z), not the whole map.
        assert torch.equal(train_a, net.arg_logits(perturbed(f, 0), full(4, N + 9), full(4, TRAIN))[:, :5])
        research = net.arg_logits(f, full(4, G_ACTOR), full(4, RESEARCH))[:, :9]
        assert torch.equal(research, net.arg_logits(perturbed(f, 9), full(4, G_ACTOR), full(4, RESEARCH))[:, :9])


# ------------------------------------------------------------ evaluate / act on real states


def collect_states(net, steps=12, num_envs=6, seed=0):
    """Run `act` on a vec env for a few steps; return the stored batch of (obs, actions, masks, ActOut)."""
    vec = SyncVecEnv([lambda: ps.Env(**KW) for _ in range(num_envs)], seed_base=seed)
    vec.reset_all()
    gen = torch.Generator().manual_seed(seed)
    fetch = {"actor": vec.actor_masks, "kind": vec.kind_masks, "arg": vec.arg_masks}
    batches = []
    for _ in range(steps):
        board, glob = (torch.from_numpy(x) for x in vec.observations())
        with torch.no_grad():
            out = net.act(net.trunk(board, glob), lambda stage, *a: fetch[stage](*a), gen)
        batches.append((board, glob, out))
        vec.step(out.actors.numpy(), out.kinds.numpy(), out.args.numpy())
    return batches


def stack(batches):
    cat = lambda xs: torch.cat(xs, dim=0)  # noqa: E731
    return dict(
        board=cat([b for b, _, _ in batches]),
        globals=cat([g for _, g, _ in batches]),
        actors=cat([o.actors for _, _, o in batches]),
        kinds=cat([o.kinds for _, _, o in batches]),
        args=cat([o.args for _, _, o in batches]),
        actor_mask=cat([o.actor_mask for _, _, o in batches]),
        kind_mask=cat([o.kind_mask for _, _, o in batches]),
        arg_mask=cat([o.arg_mask for _, _, o in batches]),
        logp=cat([o.logp for _, _, o in batches]),
        entropy=cat([o.entropy for _, _, o in batches]),
        value=cat([o.value for _, _, o in batches]),
    )


def run_evaluate(net, d, perm=None):
    keys = ["board", "globals", "actors", "kinds", "args", "actor_mask", "kind_mask", "arg_mask"]
    xs = [d[k] if perm is None else d[k][perm] for k in keys]
    with torch.no_grad():
        return net.evaluate(*xs)


def reference_logp_entropy(net, d):
    """The fired-stage rule, recomputed with the NumPy reference from the net's own head logits."""
    with torch.no_grad():
        f = net.trunk(d["board"], d["globals"])
        al = net.actor_logits(f).numpy()
        kl = net.kind_logits(f, d["actors"]).numpy()
        xl = net.arg_logits(f, d["actors"], d["kinds"]).numpy()
    b = np.arange(len(al))
    a, k, x = d["actors"].numpy(), d["kinds"].numpy(), d["args"].numpy()
    am, km, xm = d["actor_mask"].numpy(), d["kind_mask"].numpy(), d["arg_mask"].numpy()
    fired = ~np.isin(k, NO_ARG_KINDS)
    lp = np_masked_logp(al, am)[b, a] + np_masked_logp(kl, km)[b, k]
    h_actor, h_kind = np_masked_entropy(al, am), np_masked_entropy(kl, km)
    h_arg = np.zeros(len(al))
    xm_safe = np.where(fired[:, None], xm, True)  # unfired rows: never evaluated
    lp_arg = np_masked_logp(xl, xm_safe)[b, x]
    lp = lp + np.where(fired, lp_arg, 0.0)
    h_arg = np.where(fired, np_masked_entropy(xl, xm_safe), 0.0)
    return lp, h_actor, h_kind, h_arg


def test_act_respects_masks_and_evaluate_matches_act():
    net = make_net(seed=1)
    d = stack(collect_states(net))
    b = np.arange(len(d["actors"]))
    a, k, x = d["actors"].numpy(), d["kinds"].numpy(), d["args"].numpy()
    assert d["actor_mask"].numpy()[b, a].all()
    assert d["kind_mask"].numpy()[b, k].all()
    fired = ~np.isin(k, NO_ARG_KINDS)
    assert d["arg_mask"].numpy()[b[fired], x[fired]].all()
    assert (x[~fired] == 0).all()
    assert not d["arg_mask"].numpy()[~fired].any()
    assert fired.any() and (~fired).any()
    ev = run_evaluate(net, d)
    np.testing.assert_allclose(ev.logp.numpy(), d["logp"].numpy(), atol=1e-5)
    np.testing.assert_allclose(ev.entropy.numpy(), d["entropy"].numpy(), atol=1e-5)
    np.testing.assert_allclose(ev.value.numpy(), d["value"].numpy(), atol=1e-5)


def test_logp_and_entropy_follow_the_fired_stage_rule():
    net = make_net(seed=2)
    d = stack(collect_states(net, seed=3))
    lp, h_actor, h_kind, h_arg = reference_logp_entropy(net, d)
    ev = run_evaluate(net, d)
    np.testing.assert_allclose(ev.logp.numpy(), lp, atol=1e-5)
    np.testing.assert_allclose(ev.head_entropy["actor"].numpy(), h_actor, atol=1e-5)
    np.testing.assert_allclose(ev.head_entropy["kind"].numpy(), h_kind, atol=1e-5)
    np.testing.assert_allclose(ev.head_entropy["arg"].numpy(), h_arg, atol=1e-5)
    np.testing.assert_allclose(ev.entropy.numpy(), h_actor + h_kind + h_arg, atol=1e-5)
    fired = ~np.isin(d["kinds"].numpy(), NO_ARG_KINDS)
    assert (ev.head_entropy["arg"].numpy()[~fired] == 0).all()


def test_greedy_act_takes_every_masked_argmax():
    net = make_net(seed=4)
    vec = SyncVecEnv([lambda: ps.Env(**KW) for _ in range(5)], seed_base=7)
    vec.reset_all()
    fetch = {"actor": vec.actor_masks, "kind": vec.kind_masks, "arg": vec.arg_masks}
    board, glob = (torch.from_numpy(x) for x in vec.observations())
    with torch.no_grad():
        f = net.trunk(board, glob)
        out = net.act(f, lambda stage, *a: fetch[stage](*a), torch.Generator().manual_seed(0), greedy=True)
        al = net.actor_logits(f).numpy()
        exp_a = np.where(vec.actor_masks(), al, -np.inf).argmax(1)
        assert out.actors.tolist() == exp_a.tolist()
        kl = net.kind_logits(f, out.actors).numpy()
        exp_k = np.where(vec.kind_masks(exp_a), kl, -np.inf).argmax(1)
        assert out.kinds.tolist() == exp_k.tolist()
        xl = net.arg_logits(f, out.actors, out.kinds).numpy()
        xm = vec.arg_masks(exp_a, exp_k)
        for i in range(5):
            if xm[i].any():
                assert out.args[i].item() == np.where(xm[i], xl[i], -np.inf).argmax()
            else:
                assert out.args[i].item() == 0


def test_batch_rows_are_independent():
    net = make_net(seed=5)
    d = stack(collect_states(net, steps=4, seed=6))
    perm = torch.from_numpy(np.random.default_rng(0).permutation(len(d["actors"])))
    ev, evp = run_evaluate(net, d), run_evaluate(net, d, perm)
    for x, y in ((ev.logp, evp.logp), (ev.entropy, evp.entropy), (ev.value, evp.value)):
        np.testing.assert_allclose(x[perm].numpy(), y.numpy(), atol=1e-5)


def synthetic_batch(kind, b=6):
    """A batch where every row uses `kind` with a fitting actor and all-true masks."""
    board, glob = real_obs(b)
    actor = {MOVE: 9, ATTACK: 9, CAPTURE: 9, TRAIN: N + 9, RESEARCH: G_ACTOR, END_TURN: G_ACTOR}[kind]
    arg_len = {MOVE: N, ATTACK: N, TRAIN: 5, RESEARCH: 9}.get(kind, 0)
    arg_mask = torch.zeros(b, N, dtype=torch.bool)
    arg_mask[:, :arg_len] = True
    return dict(
        board=board,
        globals=glob,
        actors=full(b, actor),
        kinds=full(b, kind),
        args=full(b, 2 if arg_len else 0),
        actor_mask=torch.ones(b, 3 * N + 1, dtype=torch.bool),
        kind_mask=torch.ones(b, 10, dtype=torch.bool),
        arg_mask=arg_mask,
    )


@pytest.mark.parametrize("kind", [CAPTURE, END_TURN])
def test_no_argument_kind_ignores_every_argument_head_weight(kind):
    net = make_net(seed=6)
    d = synthetic_batch(kind)
    before = run_evaluate(net, d)
    with torch.no_grad():
        for p in net.arg_head.parameters():
            p.add_(torch.randn_like(p))
    after = run_evaluate(net, d)
    assert torch.equal(before.logp, after.logp)
    assert torch.equal(before.entropy, after.entropy)


@pytest.mark.parametrize("kind", [MOVE, TRAIN, RESEARCH])
def test_argument_kind_depends_on_argument_head(kind):
    net = make_net(seed=6)
    d = synthetic_batch(kind)
    before = run_evaluate(net, d)
    with torch.no_grad():
        for p in net.arg_head.parameters():
            p.add_(torch.randn_like(p))
    assert not torch.equal(before.logp, run_evaluate(net, d).logp)


def grads(module):
    return [p.grad for p in module.parameters()]


def nonzero(gs):
    return any(g is not None and g.abs().sum().item() > 0 for g in gs)


def backward(net, d, loss_of):
    net.zero_grad(set_to_none=True)
    keys = ["board", "globals", "actors", "kinds", "args", "actor_mask", "kind_mask", "arg_mask"]
    loss_of(net.evaluate(*[d[k] for k in keys])).backward()


def test_gradients_reach_only_the_fired_heads():
    net = make_net(seed=7)
    policy_loss = lambda ev: ev.logp.sum() + ev.entropy.sum()  # noqa: E731
    backward(net, synthetic_batch(END_TURN), policy_loss)
    assert nonzero(grads(net.actor_head)) and nonzero(grads(net.kind_head))
    assert not nonzero(grads(net.arg_head))
    assert not nonzero(grads(net.value_head))
    backward(net, synthetic_batch(MOVE), policy_loss)
    assert nonzero(grads(net.actor_head)) and nonzero(grads(net.kind_head)) and nonzero(grads(net.arg_head))
    assert not nonzero(grads(net.value_head))
    backward(net, synthetic_batch(MOVE), lambda ev: ev.value.sum())
    assert nonzero(grads(net.value_head))
    for head in (net.actor_head, net.kind_head, net.arg_head):
        assert not nonzero(grads(head))


def test_heads_share_no_parameters():
    net = make_net()
    groups = [net.actor_head, net.kind_head, net.arg_head, net.value_head]
    ids = [{id(p) for p in m.parameters()} for m in groups]
    for i in range(4):
        assert ids[i], "every head has parameters"
        for j in range(i + 1, 4):
            assert not ids[i] & ids[j]
