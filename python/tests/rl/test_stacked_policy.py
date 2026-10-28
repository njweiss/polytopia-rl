"""Milestone 13b tests 1 to 5: `polytopia_rl.stacked` ("Stacked forward").

The stacked policy is observed through what it hands to `categorical.sample` (one call per stage over
the R valid rows, `SampleSpy`), through the masks it asks the envs for (`MaskEnv`), and through the
actions it returns. Every expected number is each snapshot's own `PolicyNet` forward on its own
group (stk_helpers.ref_stage_logits) or `net_batch_policy` per group; logits are compared with
atol 1e-5 (the vmapped forward may sum in a different order). Tiny nets (width 8, one block) with
all parameters redrawn N(0, 0.2^2) per snapshot; real positions on a size-6 map.

The public API exposes no trunk features, so test 1 checks them through the three stages' logits
(the actor logits read F and g directly; the kind and argument logits read F at the actor's tile
and g).
"""

import copy
from collections import Counter

import numpy as np
import pytest
import torch
from rl_helpers import ATTACK, CAPTURE, END_TURN, MOVE, RESEARCH, TRAIN, FakeEnv, all_paths, np_masked_logp
from stk_helpers import MaskEnv, SampleSpy, entry_of, greedy_gaps, has_arg, padded_arg_mask, positions, random_net, real_spec, ref_stage_logits

from polytopia_rl import categorical
from polytopia_rl.selfplay import Entry, net_batch_policy
from polytopia_rl.stacked import StackedSnapshots, stacked_batch_policy

POOL = 5
ATOL = 1e-5


@pytest.fixture(scope="module")
def pool():
    """Five snapshots with different weights: `(nets, entries)`, entry i is `snap_{100 * i}`."""
    nets = [random_net(10 + i) for i in range(POOL)]
    return nets, [entry_of(net, 100 * i) for i, net in enumerate(nets)]


def new_stack(entries, base_seed=99):
    """A stack whose base net has yet other weights (they must never be used)."""
    stack = StackedSnapshots(random_net(base_seed), "cpu")
    stack.sync(entries)
    return stack


def split(envs, sizes):
    out, k = [], 0
    for b in sizes:
        out.append(envs[k : k + b])
        k += b
    return out


def make_groups(entries, which, env_groups):
    return [(entries[w].name, envs) for w, envs in zip(which, env_groups)]


def flat_actions(actions):
    return [tuple(int(x) for x in a) for acts in actions for a in acts]


def close(got, want, what):
    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=0, msg=lambda m: f"{what}: {m}")


# ---------------------------------------------------------------- 1. stacked logits equal each snapshot's forward

# (pool entries used, in group order; group sizes): groups use a subset of the stack in an order
# that is not the stack's, so the gathered rows are not 0..G-1.
CASES = {
    "G1-size1": ([3], [1]),
    "G1-size3": ([0], [3]),
    "G2": ([4, 1], [3, 1]),
    "G3": ([2, 0, 3], [1, 4, 2]),
    "G4": ([3, 1, 4, 0], [2, 1, 5, 3]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_stacked_logits_and_masks_equal_each_snapshots_own_forward(case, pool, monkeypatch):
    nets, entries = pool
    which, sizes = CASES[case]
    envs = positions(sum(sizes), seed=3 + len(which))
    env_groups = split(envs, sizes)
    stack = new_stack(entries)
    spy = SampleSpy(monkeypatch)
    policy = stacked_batch_policy(stack, torch.Generator().manual_seed(1), greedy=False)
    actions = policy(make_groups(entries, which, env_groups))

    assert [len(a) for a in actions] == sizes, "one list of actions per group, aligned with its envs"
    flat = flat_actions(actions)
    R = sum(sizes)
    assert len(spy.calls) == 3, "one categorical.sample call per stage for all groups"
    for c in spy.calls:
        assert c["logits"].shape[0] == R and c["mask"].shape[0] == R, "only the valid rows reach a sample"
        assert c["check"] is False and c["greedy"] is False
    actors = np.array([a for a, _, _ in flat])
    kinds = np.array([k for _, k, _ in flat])
    args = np.array([x for _, _, x in flat])
    n = envs[0].spec()["height"] * envs[0].spec()["width"]
    fired = has_arg(real_spec())[torch.as_tensor(kinds)]

    # Reference: each group's own snapshot, conditioned on the actors and kinds the stacked policy chose.
    want = [[], [], []]
    k0 = 0
    for w, g_envs in zip(which, env_groups):
        b = len(g_envs)
        for stage, t in enumerate(ref_stage_logits(nets[w], g_envs, actors[k0 : k0 + b], kinds[k0 : k0 + b])):
            want[stage].append(t)
        k0 += b
    close(spy.calls[0]["logits"], torch.cat(want[0]), "actor logits")
    close(spy.calls[1]["logits"], torch.cat(want[1]), "kind logits")
    close(spy.calls[2]["logits"], torch.cat(want[2]), "arg logits")

    # Masks: fetched for the valid rows in (group, slot) order, as net_batch_policy does.
    assert np.array_equal(spy.calls[0]["mask"].numpy(), np.stack([e.actor_mask() for e in envs]))
    assert np.array_equal(spy.calls[1]["mask"].numpy(), np.stack([e.kind_mask(int(a)) for e, a in zip(envs, actors)]))
    arg_mask = spy.calls[2]["mask"].numpy()
    for r, e in enumerate(envs):
        if fired[r]:
            assert np.array_equal(arg_mask[r], padded_arg_mask(e, actors[r], kinds[r], n)), r
            assert args[r] == int(spy.calls[2]["out"][r])
        else:
            assert arg_mask[r].any() and args[r] == 0, "a kind without an argument gets arg 0"
    assert spy.calls[0]["out"].tolist() == actors.tolist()
    assert spy.calls[1]["out"].tolist() == kinds.tolist()
    # Every action is legal in its own env.
    for e, a in zip(envs, flat):
        assert a in all_paths(e), a


# ---------------------------------------------------------------- 2. greedy decisions


def test_greedy_actions_equal_net_batch_policy_per_group(pool, monkeypatch):
    nets, entries = pool
    which, sizes = [2, 4, 0, 1], [4, 1, 3, 6]
    env_groups = split(positions(sum(sizes), seed=21), sizes)
    # The fixture is tie-free: every stage's top-two gap on each greedy path exceeds 1e-3, so float
    # order differences cannot flip an argmax.
    for w, g_envs in zip(which, env_groups):
        assert min(greedy_gaps(nets[w], g_envs)) > 1e-3
    want = [[tuple(int(x) for x in a) for a in net_batch_policy(nets[w], torch.Generator().manual_seed(0), greedy=True)(g_envs)] for w, g_envs in zip(which, env_groups)]
    stack = new_stack(entries)
    spy = SampleSpy(monkeypatch)
    for seed in (0, 1, 2):
        spy.calls.clear()
        got = stacked_batch_policy(stack, torch.Generator().manual_seed(seed), greedy=True)(make_groups(entries, which, env_groups))
        assert [[tuple(int(x) for x in a) for a in acts] for acts in got] == want, f"generator seed {seed}"
        assert len(spy.calls) == 3 and all(c["greedy"] is True for c in spy.calls)


# ---------------------------------------------------------------- 3. alignment and mask requests


def unique_path(j):
    """A distinct single legal path for env j on the size-4 fake (N = 16, global actor 48), cycling
    through tile, enum and argument-free kinds."""
    r = j % 5
    if r == 0:
        return (j % 16, MOVE, (3 * j + 1) % 16)
    if r == 1:
        return (16 + j % 16, TRAIN, j % 5)
    if r == 2:
        return (32 + j % 16, CAPTURE, 0)
    if r == 3:
        return (48, RESEARCH, j % 9)
    return (j % 16, ATTACK, (5 * j + 2) % 16)


@pytest.mark.parametrize("greedy", [False, True])
def test_actions_return_to_their_group_and_slot_and_masks_are_fetched_once_per_valid_row(greedy, monkeypatch):
    spec = FakeEnv().spec()
    nets = [random_net(40 + i, spec) for i in range(POOL)]
    entries = [entry_of(net, 100 * i) for i, net in enumerate(nets)]
    which, sizes = [3, 0, 4, 1], [3, 1, 5, 2]
    paths = [unique_path(j) for j in range(sum(sizes))]
    assert len(set(paths)) == len(paths)
    envs = [MaskEnv(p, seed=j) for j, p in enumerate(paths)]
    env_groups = split(envs, sizes)
    stack = StackedSnapshots(random_net(99, spec), "cpu")
    stack.sync(entries)
    spy = SampleSpy(monkeypatch)
    got = stacked_batch_policy(stack, torch.Generator().manual_seed(5), greedy=greedy)(make_groups(entries, which, env_groups))
    assert [[tuple(int(x) for x in a) for a in acts] for acts in got] == [[e.path for e in g] for g in env_groups]
    for e in envs:
        a, k, _ = e.path
        assert e.requests == [("actor",), ("kind", a), ("arg", a, k)], (e.path, e.requests)
    assert [c["logits"].shape[0] for c in spy.calls] == [len(envs)] * 3
    # The expectation is what net_batch_policy does per group.
    for w, g in zip(which, env_groups):
        fresh = [MaskEnv(e.path, seed=0) for e in g]
        ref = net_batch_policy(nets[w], torch.Generator().manual_seed(5), greedy=greedy)(fresh)
        assert [tuple(int(x) for x in a) for a in ref] == [e.path for e in g]
        assert [e.requests for e in fresh] == [e.requests for e in g]


def test_a_global_actor_with_end_turn_and_a_single_group_of_one(monkeypatch):
    spec = FakeEnv().spec()
    net = random_net(60, spec)
    entries = [entry_of(random_net(61, spec), 0), entry_of(net, 100)]
    env = MaskEnv((48, END_TURN, 0), seed=1)
    stack = StackedSnapshots(random_net(99, spec), "cpu")
    stack.sync(entries)
    got = stacked_batch_policy(stack, torch.Generator().manual_seed(0))([("snap_100", [env])])
    assert [[tuple(int(x) for x in a) for a in acts] for acts in got] == [[(48, END_TURN, 0)]]
    assert env.requests == [("actor",), ("kind", 48), ("arg", 48, END_TURN)]


# ---------------------------------------------------------------- 4. sampling


def exact_path_probs(net, env):
    """{path: probability} of `net`'s factored policy on `env`, from its own forward (float64)."""
    paths = all_paths(env)
    envs = [env] * len(paths)
    actors = [a for a, _, _ in paths]
    kinds = [k for _, k, _ in paths]
    a_logits, k_logits, x_logits = ref_stage_logits(net, envs, actors, kinds)
    n = env.spec()["height"] * env.spec()["width"]
    fired = has_arg(env.spec())
    out = {}
    lp_a = np_masked_logp(a_logits[0].numpy(), env.actor_mask())
    for r, (a, k, x) in enumerate(paths):
        lp = lp_a[a] + np_masked_logp(k_logits[r].numpy(), env.kind_mask(a))[k]
        if fired[k]:
            lp += np_masked_logp(x_logits[r].numpy(), padded_arg_mask(env, a, k, n))[x]
        out[(a, k, x)] = float(np.exp(lp))
    assert abs(sum(out.values()) - 1.0) < 1e-6
    return out


def chi_square(counts: Counter, probs: dict, n: int):
    """Pearson's statistic and degrees of freedom; bins with an expected count below 5 are pooled."""
    assert set(counts) <= set(probs), "a sampled action outside the legal paths"
    big = [p for p in probs if n * probs[p] >= 5]
    obs = [counts[p] for p in big]
    exp = [n * probs[p] for p in big]
    rest = n - sum(exp)
    if rest >= 5:
        obs.append(n - sum(obs))
        exp.append(rest)
    stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    return stat, len(exp) - 1


def loose_bound(df):
    # chi^2(df) has mean df and variance 2 df: about 6 sigma above the mean (the brief asks for a
    # loose bound; a wrong distribution here gives statistics in the hundreds).
    return df + 6 * np.sqrt(2 * df)


def test_sampled_actions_follow_each_snapshots_distribution(pool):
    nets, entries = pool
    env_a, env_b = positions(2, seed=30, max_steps=4)
    M, CALLS = 500, 4
    stack = new_stack(entries)
    policy = stacked_batch_policy(stack, torch.Generator().manual_seed(17))
    counts = {"a": Counter(), "b": Counter()}
    for _ in range(CALLS):
        got = policy([(entries[3].name, [env_a] * M), (entries[1].name, [env_b] * M)])
        counts["a"].update(tuple(int(x) for x in a) for a in got[0])
        counts["b"].update(tuple(int(x) for x in a) for a in got[1])
    for key, net, env in (("a", nets[3], env_a), ("b", nets[1], env_b)):
        probs = exact_path_probs(net, env)
        stat, df = chi_square(counts[key], probs, M * CALLS)
        assert df >= 3, "the fixture needs a spread distribution"
        assert stat < loose_bound(df), (key, stat, df)
        # The per-snapshot policy on the same position passes the same check (the reference).
        ref = net_batch_policy(net, torch.Generator().manual_seed(18))([env] * (M * CALLS))
        stat_ref, _ = chi_square(Counter(tuple(int(x) for x in a) for a in ref), probs, M * CALLS)
        assert stat_ref < loose_bound(df), (key, stat_ref, df)
    # The two snapshots' distributions really differ (the groups did not share parameters).
    pa, pb = exact_path_probs(nets[3], env_a), exact_path_probs(nets[1], env_a)
    assert max(abs(pa[p] - pb[p]) for p in pa) > 0.05


def test_a_fixed_seed_repeats_and_another_seed_differs(pool):
    nets, entries = pool
    envs = positions(6, seed=33)
    groups = [(entries[0].name, envs[:2]), (entries[2].name, envs[2:3] * 40), (entries[4].name, envs[3:])]
    stack = new_stack(entries)

    def draw(seed):
        return stacked_batch_policy(stack, torch.Generator().manual_seed(seed))(groups)

    first = draw(5)
    assert [[tuple(int(x) for x in a) for a in g] for g in draw(5)] == [[tuple(int(x) for x in a) for a in g] for g in first]
    assert flat_actions(draw(6)) != flat_actions(first)


def test_draws_come_from_one_sample_per_stage_over_all_rows_in_group_then_slot_order(pool):
    """Replay with the documented procedure on each snapshot's own logits: concatenate the groups'
    rows, then one `categorical.sample` per stage with the same generator state."""
    nets, entries = pool
    which, sizes = [1, 3, 0], [3, 2, 4]
    envs = positions(sum(sizes), seed=35)
    env_groups = split(envs, sizes)
    gen = torch.Generator().manual_seed(23)
    state = gen.get_state()
    stack = new_stack(entries)
    got = flat_actions(stacked_batch_policy(stack, gen)(make_groups(entries, which, env_groups)))

    replay = torch.Generator()
    replay.set_state(state)
    n = envs[0].spec()["height"] * envs[0].spec()["width"]
    a_logits = torch.cat([ref_stage_logits(nets[w], g)[0] for w, g in zip(which, env_groups)])
    actors = categorical.sample(a_logits, torch.from_numpy(np.stack([e.actor_mask() for e in envs])), replay, False, check=False).numpy()
    k_logits, k0 = [], 0
    for w, g in zip(which, env_groups):
        k_logits.append(ref_stage_logits(nets[w], g, actors[k0 : k0 + len(g)])[1])
        k0 += len(g)
    kmask = torch.from_numpy(np.stack([e.kind_mask(int(a)) for e, a in zip(envs, actors)]))
    kinds = categorical.sample(torch.cat(k_logits), kmask, replay, False, check=False).numpy()
    x_logits, k0 = [], 0
    for w, g in zip(which, env_groups):
        x_logits.append(ref_stage_logits(nets[w], g, actors[k0 : k0 + len(g)], kinds[k0 : k0 + len(g)])[2])
        k0 += len(g)
    fired = has_arg(real_spec())[torch.as_tensor(kinds)]
    xmask = torch.from_numpy(np.stack([padded_arg_mask(e, a, k, n) for e, a, k in zip(envs, actors, kinds)]))
    args = torch.where(fired, categorical.sample(torch.cat(x_logits), xmask | ~fired.unsqueeze(1), replay, False, check=False), 0).numpy()
    assert got == [(int(a), int(k), int(x)) for a, k, x in zip(actors, kinds, args)]
    assert gen.get_state().equal(replay.get_state()), "the policy draws exactly once per stage"


# ---------------------------------------------------------------- 5. the stack cache


class StackCount:
    """Counts `torch.func.stack_module_state` calls (the module attribute and the name
    `polytopia_rl.stacked` may have imported)."""

    def __init__(self, monkeypatch):
        import torch.func

        import polytopia_rl.stacked as stacked

        self.n = 0
        original = torch.func.stack_module_state

        def spy(*a, **kw):
            self.n += 1
            return original(*a, **kw)

        monkeypatch.setattr(torch.func, "stack_module_state", spy)
        monkeypatch.setattr(stacked, "stack_module_state", spy, raising=False)


def test_sync_restacks_only_when_the_names_change_and_index_follows_the_order(pool, monkeypatch):
    _, entries = pool
    count = StackCount(monkeypatch)
    stack = StackedSnapshots(random_net(99), "cpu")
    policy = stacked_batch_policy(stack, torch.Generator())
    first = entries[:3]
    stack.sync(first)
    assert count.n == 1
    assert [stack.index(e.name) for e in first] == [0, 1, 2]
    stack.sync(first)
    stack.sync(list(first))
    policy.sync(first)  # the policy's sync forwards to the stack
    assert count.n == 1, "same names: no restack"
    policy.sync(entries[:4])  # a snapshot was added
    assert count.n == 2
    assert [stack.index(e.name) for e in entries[:4]] == [0, 1, 2, 3]
    thinned = [entries[0], entries[2], entries[3], entries[4]]  # same length, different names
    stack.sync(thinned)
    assert count.n == 3
    assert [stack.index(e.name) for e in thinned] == [0, 1, 2, 3]
    reordered = [entries[4], entries[0], entries[3], entries[2]]
    stack.sync(reordered)
    assert [stack.index(e.name) for e in reordered] == [0, 1, 2, 3]


def stacked_actor_logits(stack, name, envs, monkeypatch):
    with monkeypatch.context() as mp:
        spy = SampleSpy(mp)
        stacked_batch_policy(stack, torch.Generator().manual_seed(0), greedy=True)([(name, envs)])
        return spy.calls[0]["logits"]


def test_a_new_entry_is_picked_up_and_same_names_keep_the_stacked_copies(monkeypatch):
    nets = [random_net(70 + i) for i in range(4)]
    entries = [entry_of(net, 100 * i) for i, net in enumerate(nets)]
    envs = positions(3, seed=37)
    stack = StackedSnapshots(random_net(99), "cpu")
    policy = stacked_batch_policy(stack, torch.Generator())
    policy.sync(entries[:2])
    close(stacked_actor_logits(stack, entries[1].name, envs, monkeypatch), ref_stage_logits(nets[1], envs)[0], "snap_100")
    policy.sync(entries[:3])
    close(stacked_actor_logits(stack, entries[2].name, envs, monkeypatch), ref_stage_logits(nets[2], envs)[0], "added snap_200")
    close(stacked_actor_logits(stack, entries[0].name, envs, monkeypatch), ref_stage_logits(nets[0], envs)[0], "snap_0 after restack")
    # Overwrite snap_0's stored weights in place (with another net's): a sync with the same names
    # keeps the stacked copies...
    old = copy.deepcopy(nets[0])
    replacement = random_net(77).state_dict()
    with torch.no_grad():
        for k, v in entries[0].state.items():
            v.copy_(replacement[k])
    policy.sync(entries[:3])
    close(stacked_actor_logits(stack, entries[0].name, envs, monkeypatch), ref_stage_logits(old, envs)[0], "no restack on same names")
    # ...and the next restack (a snapshot added) reads the entries again.
    policy.sync(entries[:4])
    changed = copy.deepcopy(nets[0])
    changed.load_state_dict(entries[0].state)
    close(stacked_actor_logits(stack, entries[0].name, envs, monkeypatch), ref_stage_logits(changed, envs)[0], "restacked snap_0")


def test_the_stack_holds_frozen_copies_and_never_changes_the_base_net(monkeypatch):
    learner = random_net(80)
    before = {k: v.clone() for k, v in learner.state_dict().items()}
    frozen = copy.deepcopy(learner)
    live = Entry(0, learner.state_dict(), "snap_0")  # the learner's live tensors, not copies
    other = entry_of(random_net(81), 100)
    envs = positions(3, seed=39)
    two = StackedSnapshots(learner, "cpu")
    two.sync([other, live])
    one = StackedSnapshots(learner, "cpu")
    one.sync([live])
    after = learner.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "sync changed the base net's weights"
    # Later changes to the learner's weights do not reach the stacks.
    with torch.no_grad():
        for p in learner.parameters():
            p.add_(1.0)
    want = ref_stage_logits(frozen, envs)[0]
    close(stacked_actor_logits(two, "snap_0", envs, monkeypatch), want, "two-entry stack")
    close(stacked_actor_logits(one, "snap_0", envs, monkeypatch), want, "one-entry stack")
