"""Milestone 21 Phase A tests: `arch="v2"` PolicyNet architecture.

Covers config & constructor validation, `arch="v1"` bit-for-bit identity and legacy checkpoint
compatibility, `arch="v2"` trunk (CoordConv, GroupNorm, pre-norm scaled ResBlockV2, pooled-context
+ globals FiLM), `ArgHeadV2` (per-tile QK-Norm locality, separate move/attack keys, 2D relative
position bias), full gradient flow, `StackedSnapshots` (`torch.func.vmap`) parity, and end-to-end
training / stacked self-play / checkpointing / evaluation on `ruleset="v2"`.
"""

import json
import math

import numpy as np
import polytopia_sim as ps
import pytest
import torch
from rl_helpers import ATTACK, END_TURN, MOVE, TRAIN, all_paths
from stk_helpers import SampleSpy, entry_of, obs_tensors, ref_stage_logits
from torch.func import functional_call, vmap

from polytopia_rl import train
from polytopia_rl.checkpoint import load_net, save_checkpoint
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.evaluate import env_kwargs_of, evaluate, net_policy
from polytopia_rl.model import Features, PolicyNet
from polytopia_rl.selfplay import net_batch_policy
from polytopia_rl.stacked import StackedSnapshots, stacked_batch_policy
from polytopia_rl.vec_env import env_masks_fn


def _spec(size: int = 6, ruleset: str = "v2") -> dict:
    return ps.Env(size=size, opponent="random", agent_seat="seed_parity", turn_cap=4, ruleset=ruleset).spec()


def _real_obs(size: int = 6, ruleset: str = "v2", b: int = 4, base: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    boards, globs = [], []
    for s in range(b):
        env = ps.Env(size=size, opponent="random", agent_seat="seed_parity", turn_cap=4, ruleset=ruleset)
        env.reset(base + s)
        x, y = env.observation()
        boards.append(x)
        globs.append(y)
    return torch.from_numpy(np.stack(boards)), torch.from_numpy(np.stack(globs))


def _full(b: int, v: int) -> torch.Tensor:
    return torch.full((b,), v, dtype=torch.int64)


def _perturbed_fmap(feats: Features, tile: int, size: int, scale: float = 1.0) -> Features:
    fmap = feats.fmap.clone()
    r, c = divmod(tile, size)
    gen = torch.Generator().manual_seed(tile + 100)
    fmap[:, :, r, c] += scale * torch.randn(fmap.shape[0], fmap.shape[1], generator=gen)
    return Features(fmap=fmap, g=feats.g)


def _changed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a != b).any(dim=0)


def _positions(n: int, seed: int, size: int = 6, ruleset: str = "v2", max_steps: int = 8) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        env = ps.Env(size=size, opponent="none", agent_seat="seed_parity", ruleset=ruleset)
        env.reset(seed * 1000 + j)
        for _ in range(int(rng.integers(max_steps))):
            a = int(rng.choice(np.flatnonzero(env.actor_mask())))
            k = int(rng.choice(np.flatnonzero(env.kind_mask(a))))
            xm = env.arg_mask(a, k)
            x = int(rng.choice(np.flatnonzero(xm))) if xm.size else 0
            _, _, term, trunc = env.step_factored(a, k, x)
            assert not (term or trunc)
        out.append(env)
    return out


def _random_net_v2(spec: dict, seed: int, width: int = 8, blocks: int = 2, std: float = 0.25) -> PolicyNet:
    torch.manual_seed(seed)
    net = PolicyNet(spec, width=width, blocks=blocks, arch="v2")
    gen = torch.Generator().manual_seed(seed + 10_000)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith(".weight") and p.ndim == 1:
                p.copy_(1.0 + torch.randn(p.shape, generator=gen) * std)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * std)
    return net


# ---------------------------------------------------------------- 1. Config & constructor validation


def test_ppo_config_arch_default_round_trip_override_and_validation(tmp_path):
    default = PPOConfig()
    assert default.arch == "v1"
    assert PPOConfig.from_json("{}").arch == "v1"

    legacy = json.loads(default.to_json())
    legacy.pop("arch", None)
    assert PPOConfig.from_json(json.dumps(legacy)).arch == "v1"

    v2 = PPOConfig(arch="v2")
    assert v2.arch == "v2"
    assert PPOConfig.from_json(v2.to_json()) == v2

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"size": 6, "ruleset": "v2"}))
    loaded = load_config(str(cfg_path), ["arch=v2"])
    assert loaded.arch == "v2" and loaded.ruleset == "v2" and loaded.size == 6

    for bad in ("v3", "V2", "", "v0"):
        data = json.loads(default.to_json())
        data["arch"] = bad
        with pytest.raises(ValueError):
            PPOConfig.from_json(json.dumps(data))
        with pytest.raises(ValueError):
            load_config(str(cfg_path), [f"arch={bad}"])


@pytest.mark.parametrize("bad", ["v3", "V2", "", "v0"])
def test_policy_net_invalid_arch_raises_value_error(bad):
    with pytest.raises(ValueError):
        PolicyNet(_spec(6, "v1"), width=8, blocks=1, arch=bad)


# ---------------------------------------------------------------- 2. v1 bit-for-bit identity & legacy checkpoints


@pytest.mark.parametrize("ruleset", ["v1", "v2"])
def test_v1_bit_for_bit_identity(ruleset):
    spec = _spec(6, ruleset)
    board, glob = _real_obs(6, ruleset, b=4)

    torch.manual_seed(42)
    net_default = PolicyNet(spec, width=16, blocks=2)
    torch.manual_seed(42)
    net_v1 = PolicyNet(spec, width=16, blocks=2, arch="v1")

    sd_default = net_default.state_dict()
    sd_v1 = net_v1.state_dict()
    assert list(sd_default.keys()) == list(sd_v1.keys())
    for k in sd_default:
        assert torch.equal(sd_default[k], sd_v1[k]), f"state_dict mismatch at {k}"

    with torch.no_grad():
        f_d = net_default.trunk(board, glob)
        f_1 = net_v1.trunk(board, glob)
        assert torch.equal(f_d.fmap, f_1.fmap)
        assert torch.equal(f_d.g, f_1.g)
        assert torch.equal(net_default.actor_logits(f_d), net_v1.actor_logits(f_1))
        actors = _full(4, 9)
        kinds = _full(4, MOVE)
        assert torch.equal(net_default.kind_logits(f_d, actors), net_v1.kind_logits(f_1, actors))
        assert torch.equal(net_default.arg_logits(f_d, actors, kinds), net_v1.arg_logits(f_1, actors, kinds))
        assert torch.equal(net_default.value(f_d), net_v1.value(f_1))


def test_load_legacy_checkpoint_without_arch_key_restores_v1(tmp_path):
    spec = _spec(6, "v1")
    cfg = PPOConfig(size=6, width=8, blocks=1, ruleset="v1", device="cpu")
    torch.manual_seed(7)
    net = PolicyNet(spec, width=8, blocks=1, arch="v1")

    ckpt_path = tmp_path / "legacy_v1.pt"
    save_checkpoint(
        str(ckpt_path),
        model=net,
        optimizer=None,
        cfg=cfg,
        global_step=16,
        stage=0,
        seed_index={},
        rng_state={},
        spec=spec,
    )
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = json.loads(raw["config"])
    cfg_dict.pop("arch", None)
    raw["config"] = json.dumps(cfg_dict)
    torch.save(raw, str(ckpt_path))

    loaded, ck = load_net(str(ckpt_path), device="cpu")
    assert ck.cfg.arch == "v1"
    assert loaded.stem.in_channels == spec["channels"]
    assert list(loaded.state_dict().keys()) == list(net.state_dict().keys())
    for k, v in net.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), f"weight mismatch at {k}"

    board, glob = _real_obs(6, "v1", b=3)
    with torch.no_grad():
        assert torch.equal(loaded.actor_logits(loaded.trunk(board, glob)), net.actor_logits(net.trunk(board, glob)))


# ---------------------------------------------------------------- 3. Shapes, GroupNorm, submodule grouping, buffers


@pytest.mark.parametrize(
    ("ruleset", "size", "width", "blocks"),
    [
        ("v1", 6, 8, 1),
        ("v1", 6, 6, 2),
        ("v2", 6, 8, 2),
        ("v2", 6, 6, 1),
        ("v2", 11, 16, 2),
        ("v1", 11, 16, 8),
    ],
)
def test_v2_shapes_groupnorm_and_act_evaluate_parity(ruleset, size, width, blocks):
    spec = _spec(size, ruleset)
    n_tiles = size * size
    n_kinds = len(spec["action_types"])

    torch.manual_seed(0)
    net = PolicyNet(spec, width=width, blocks=blocks, arch="v2")
    assert net.stem.in_channels == spec["channels"] + 2

    expected_groups = 8 if width % 8 == 0 else 1
    gn_layers = [m for m in net.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert len(gn_layers) == 2 * blocks + 1
    for gn in gn_layers:
        assert gn.num_groups == expected_groups and gn.num_channels == width

    envs = _positions(3, seed=5, size=size, ruleset=ruleset, max_steps=3)
    board, glob = obs_tensors(envs)
    with torch.no_grad():
        f = net.trunk(board, glob)
        assert f.fmap.shape == (3, width, size, size)
        assert f.g.shape == (3, 3 * width)
        assert net.actor_logits(f).shape == (3, 3 * n_tiles + 1)
        assert net.kind_logits(f, _full(3, 3 * n_tiles)).shape == (3, n_kinds)
        assert net.arg_logits(f, _full(3, 1), _full(3, MOVE)).shape == (3, n_tiles)
        assert net.arg_logits(f, _full(3, n_tiles + 1), _full(3, TRAIN)).shape == (3, n_tiles)
        assert net.value(f).shape == (3,)

        out = net.act(f, env_masks_fn(envs, n_tiles), torch.Generator().manual_seed(11))
        assert torch.isfinite(out.logp).all()
        assert torch.isfinite(out.entropy).all()
        assert torch.isfinite(out.value).all()
        for i, env in enumerate(envs):
            action = (int(out.actors[i]), int(out.kinds[i]), int(out.args[i]))
            assert action in all_paths(env)

        ev = net.evaluate(board, glob, out.actors, out.kinds, out.args, out.actor_mask, out.kind_mask, out.arg_mask)
        torch.testing.assert_close(ev.logp, out.logp, atol=1e-5, rtol=0)
        torch.testing.assert_close(ev.entropy, out.entropy, atol=1e-5, rtol=0)
        torch.testing.assert_close(ev.value, out.value, atol=1e-5, rtol=0)


def test_v2_head_submodule_grouping_and_non_persistent_buffers():
    size = 6
    spec = _spec(size, "v2")
    net = PolicyNet(spec, width=16, blocks=2, arch="v2")

    heads = [net.actor_head, net.kind_head, net.arg_head, net.value_head]
    head_ids = [{id(p) for p in h.parameters()} for h in heads]
    for i in range(4):
        assert head_ids[i], f"head {i} must have parameters"
        for j in range(i + 1, 4):
            assert not (head_ids[i] & head_ids[j]), f"heads {i} and {j} share parameters"

    all_head_ids = set().union(*head_ids)
    trunk_ids = {id(p) for p in net.parameters()} - all_head_ids
    assert trunk_ids, "trunk must have parameters"

    buffers = dict(net.named_buffers())
    state_keys = set(net.state_dict().keys())
    for buf_name in ("coords", "has_arg", "arg_head.rel_index"):
        assert buf_name in buffers, f"missing buffer {buf_name}"
        assert buf_name not in state_keys, f"buffer {buf_name} must be non-persistent"

    coords = buffers["coords"]
    assert coords.shape == (1, 2, size, size) and coords.dtype == torch.float32
    expected_rows = torch.linspace(-1.0, 1.0, size).view(size, 1).expand(size, size)
    expected_cols = torch.linspace(-1.0, 1.0, size).view(1, size).expand(size, size)
    torch.testing.assert_close(coords[0, 0], expected_rows, atol=0, rtol=0)
    torch.testing.assert_close(coords[0, 1], expected_cols, atol=0, rtol=0)


# ---------------------------------------------------------------- 4. Globals reach fmap & spatial actor_logits via FiLM


def test_v2_globals_reach_fmap_and_spatial_actor_logits():
    size = 6
    n_tiles = size * size
    width = 16
    spec = _spec(size, "v2")
    torch.manual_seed(3)
    net = PolicyNet(spec, width=width, blocks=2, arch="v2")
    board, glob = _real_obs(size, "v2", b=4, base=2)
    glob2 = glob + torch.randn(glob.shape, generator=torch.Generator().manual_seed(99))

    # Step 0: FiLM is zero-initialized, so trunk starts at exact identity w.r.t. globals.
    for blk in net.blocks:
        assert torch.all(blk.film.weight == 0)
        assert torch.all(blk.film.bias == 0)
    with torch.no_grad():
        assert torch.equal(net.trunk(board, glob).fmap, net.trunk(board, glob2).fmap)

    # Step 0 backward from spatial actor logits ONLY: gradients flow into both halves of film.weight
    # (g_emb columns :width and mean_pool(x) columns width:) and film.bias.
    net.zero_grad(set_to_none=True)
    spatial = net.actor_logits(net.trunk(board, glob))[:, : 3 * n_tiles]
    weights = torch.randn(spatial.shape, generator=torch.Generator().manual_seed(17))
    (spatial * weights).sum().backward()
    for i, blk in enumerate(net.blocks):
        assert blk.film.weight.grad is not None and torch.isfinite(blk.film.weight.grad).all()
        assert blk.film.bias.grad is not None and torch.isfinite(blk.film.bias.grad).all()
        assert blk.film.weight.grad[:, :width].abs().sum().item() > 0, f"block {i} film missing g_emb gradient"
        assert blk.film.weight.grad[:, width:].abs().sum().item() > 0, f"block {i} film missing mean_pool gradient"
        assert blk.film.bias.grad.abs().sum().item() > 0, f"block {i} film bias zero gradient"

    # After 1 gradient step on film (making film weights non-zero), spatial actor logits send non-zero
    # gradients directly back into global_mlp, and perturbing globals changes fmap and spatial actor_logits.
    with torch.no_grad():
        for blk in net.blocks:
            blk.film.weight.add_(-0.1 * blk.film.weight.grad)
            blk.film.bias.add_(-0.1 * blk.film.bias.grad)

    net.zero_grad(set_to_none=True)
    spatial_after = net.actor_logits(net.trunk(board, glob))[:, : 3 * n_tiles]
    (spatial_after * weights).sum().backward()
    for name, p in net.global_mlp.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"global_mlp.{name} got zero gradient from spatial actor_logits"

    with torch.no_grad():
        f1 = net.trunk(board, glob)
        f2 = net.trunk(board, glob2)
        assert not torch.allclose(f1.fmap, f2.fmap, atol=1e-6), "globals did not reach fmap"
        a1 = net.actor_logits(f1)
        a2 = net.actor_logits(f2)
        assert not torch.allclose(a1[:, : 3 * n_tiles], a2[:, : 3 * n_tiles], atol=1e-6), (
            "globals did not reach spatial tile actor_logits"
        )


# ---------------------------------------------------------------- 5. Full-board 1-block communication via pooled FiLM


def test_v2_full_board_one_block_communication_via_pooled_film():
    size = 11
    width = 16
    spec = _spec(size, "v2")
    board, glob = _real_obs(size, "v2", b=2, base=1)
    board2 = board.clone()
    board2[:, :, 0, 0] += 5.0

    # Baseline contrast: v1 with 1 block (3x3 stem + two 3x3 convs = radius 3) cannot reach (10, 10) from (0, 0).
    torch.manual_seed(0)
    net_v1 = PolicyNet(spec, width=width, blocks=1, arch="v1")
    with torch.no_grad():
        assert torch.equal(net_v1.trunk(board, glob).fmap[:, :, 10, 10], net_v1.trunk(board2, glob).fmap[:, :, 10, 10])

    # v2 with 1 block: hook film input to verify [g_emb, mean_pool(x)] wiring directly, and check
    # opposite-corner (10, 10) changes via pooled-context FiLM (beyond GroupNorm alone).
    torch.manual_seed(0)
    net_v2 = PolicyNet(spec, width=width, blocks=1, arch="v2")
    captured = []
    handle = net_v2.blocks[0].film.register_forward_hook(lambda _m, inp, _out: captured.append(inp[0].detach().clone()))
    try:
        with torch.no_grad():
            f1_zero = net_v2.trunk(board, glob).fmap
            f2_zero = net_v2.trunk(board2, glob).fmap
            glob2 = glob + torch.randn(glob.shape, generator=torch.Generator().manual_seed(5))
            net_v2.trunk(board, glob2)
    finally:
        handle.remove()

    cond_b1, cond_b2, cond_g2 = captured
    assert cond_b1.shape == (2, 2 * width)
    # Perturbing board@(0,0) changes mean_pool(x) slice (width:) while keeping g_emb slice (:width) identical.
    assert torch.equal(cond_b1[:, :width], cond_b2[:, :width])
    assert not torch.allclose(cond_b1[:, width:], cond_b2[:, width:], atol=1e-6)
    # Perturbing globals changes g_emb slice (:width) while keeping mean_pool(x) slice (width:) identical.
    assert not torch.allclose(cond_b1[:, :width], cond_g2[:, :width], atol=1e-6)
    assert torch.equal(cond_b1[:, width:], cond_g2[:, width:])

    # Activate ONLY the mean_pool(x) columns of film.weight and verify opposite corner (10, 10) shifts
    # differently from the zero-film baseline when tile (0, 0) is perturbed.
    with torch.no_grad():
        gen = torch.Generator().manual_seed(13)
        net_v2.blocks[0].film.weight[:, :width].zero_()
        net_v2.blocks[0].film.weight[:, width:].copy_(torch.randn(2 * width, width, generator=gen) * 0.5)
        f1_film = net_v2.trunk(board, glob).fmap
        f2_film = net_v2.trunk(board2, glob).fmap

    assert not torch.allclose(f1_film[:, :, 10, 10], f2_film[:, :, 10, 10], atol=1e-5)
    delta_zero = f2_zero[:, :, 10, 10] - f1_zero[:, :, 10, 10]
    delta_film = f2_film[:, :, 10, 10] - f1_film[:, :, 10, 10]
    assert not torch.allclose(delta_film, delta_zero, atol=1e-5), (
        "pooled-context FiLM did not contribute to opposite-corner (10, 10) communication"
    )


# ---------------------------------------------------------------- 6. CoordConv breaks translation symmetry


@pytest.mark.parametrize("size", [6, 11])
def test_v2_coordconv_breaks_translation_symmetry(size):
    spec = _spec(size, "v2")
    torch.manual_seed(4)
    net = PolicyNet(spec, width=16, blocks=2, arch="v2")
    assert "coords" not in net.state_dict()

    board = torch.zeros(2, spec["channels"], size, size)
    glob = torch.zeros(2, spec["globals"])
    with torch.no_grad():
        fmap = net.trunk(board, glob).fmap

    tl = fmap[:, :, 0, 0]
    tr = fmap[:, :, 0, size - 1]
    bl = fmap[:, :, size - 1, 0]
    br = fmap[:, :, size - 1, size - 1]
    mid = fmap[:, :, size // 2, size // 2]

    assert not torch.allclose(tl, mid, atol=1e-5), "corner vs center identical on zero board"
    assert not torch.allclose(tl, br, atol=1e-5), "opposite corners identical on zero board"
    assert not torch.allclose(tl, tr, atol=1e-5), "col coordinate channel missing from CoordConv"
    assert not torch.allclose(tl, bl, atol=1e-5), "row coordinate channel missing from CoordConv"


# ---------------------------------------------------------------- 7. Deep-trunk numerical stability (blocks=8)


def test_v2_deep_trunk_numerical_stability_and_res_scale():
    size = 11
    n_tiles = size * size
    spec = _spec(size, "v2")
    board, glob = _real_obs(size, "v2", b=4, base=3)

    torch.manual_seed(10)
    net1 = PolicyNet(spec, width=16, blocks=1, arch="v2")
    torch.manual_seed(10)
    net8 = PolicyNet(spec, width=16, blocks=8, arch="v2")

    assert net1.blocks[0].res_scale == pytest.approx(1.0 / math.sqrt(2 * 1))
    for blk in net8.blocks:
        assert blk.res_scale == pytest.approx(1.0 / math.sqrt(2 * 8))

    actors = _full(4, 5)
    kinds = _full(4, MOVE)
    args = _full(4, 6)
    am = torch.ones(4, 3 * n_tiles + 1, dtype=torch.bool)
    km = torch.ones(4, len(spec["action_types"]), dtype=torch.bool)
    xm = torch.ones(4, n_tiles, dtype=torch.bool)

    norms = {}
    stem_grads = {}
    for label, net in (("b1", net1), ("b8", net8)):
        net.zero_grad(set_to_none=True)
        feats = net.trunk(board, glob)
        assert torch.isfinite(feats.fmap).all() and torch.isfinite(feats.g).all()
        norms[label] = feats.fmap.detach().pow(2).mean().sqrt().item()

        ev = net.evaluate(board, glob, actors, kinds, args, am, km, xm)
        assert torch.isfinite(ev.logp).all() and torch.isfinite(ev.entropy).all() and torch.isfinite(ev.value).all()
        (-(ev.logp.mean()) + ev.value.pow(2).mean()).backward()
        for name, p in net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{label} non-finite grad at {name}"
        stem_grads[label] = net.stem.weight.grad.norm().item()

    assert 0.1 * norms["b1"] < norms["b8"] < 5.0 * norms["b1"], f"activation norm blew up: {norms}"
    assert 0.0 < stem_grads["b8"] < 10.0 * stem_grads["b1"] and stem_grads["b8"] < 100.0, (
        f"stem gradient blew up at depth 8: {stem_grads}"
    )


# ---------------------------------------------------------------- 8. ArgHeadV2 structural properties & gradient flow


@pytest.mark.parametrize("kind", [MOVE, ATTACK])
def test_arg_head_v2_per_tile_locality(kind):
    size = 6
    n_tiles = size * size
    spec = _spec(size, "v2")
    torch.manual_seed(2)
    net = PolicyNet(spec, width=16, blocks=2, arch="v2")
    board, glob = _real_obs(size, "v2", b=4)
    with torch.no_grad():
        f = net.trunk(board, glob)
        actor_tile = 9
        actors, kinds = _full(4, actor_tile), _full(4, kind)
        base = net.arg_logits(f, actors, kinds)
        for u in (0, 2 * size + 5, n_tiles - 1):
            ch = _changed(base, net.arg_logits(_perturbed_fmap(f, u, size), actors, kinds))
            expected = torch.zeros(n_tiles, dtype=torch.bool)
            expected[u] = True
            assert torch.equal(ch, expected), f"tile {u}: changed {torch.nonzero(ch).flatten().tolist()}"

        # Enum heads still ignore tiles other than the actor's tile.
        train_base = net.arg_logits(f, _full(4, n_tiles + actor_tile), _full(4, TRAIN))[:, :8]
        train_pert = net.arg_logits(_perturbed_fmap(f, 0, size), _full(4, n_tiles + actor_tile), _full(4, TRAIN))[:, :8]
        assert torch.equal(train_base, train_pert)


def test_arg_head_v2_separate_move_and_attack_keys():
    size = 6
    spec = _spec(size, "v2")
    torch.manual_seed(6)
    net = PolicyNet(spec, width=16, blocks=2, arch="v2")
    assert len(net.arg_head.tile_keys) == 2
    assert len(net.arg_head.k_norms) == 2

    ti_move = net.arg_head.tile_kinds.index(MOVE)
    ti_atk = net.arg_head.tile_kinds.index(ATTACK)

    board, glob = _real_obs(size, "v2", b=4)
    actors = _full(4, 14)
    with torch.no_grad():
        f = net.trunk(board, glob)
        move_0 = net.arg_logits(f, actors, _full(4, MOVE))
        atk_0 = net.arg_logits(f, actors, _full(4, ATTACK))

        # Perturb move key conv -> move logits change, attack logits stay bit-identical.
        gen = torch.Generator().manual_seed(21)
        net.arg_head.tile_keys[ti_move].weight.add_(torch.randn(net.arg_head.tile_keys[ti_move].weight.shape, generator=gen))
        move_1 = net.arg_logits(f, actors, _full(4, MOVE))
        atk_1 = net.arg_logits(f, actors, _full(4, ATTACK))
        assert not torch.equal(move_1, move_0)
        assert torch.equal(atk_1, atk_0)

        # Perturb attack key conv -> attack logits change, move logits stay bit-identical.
        net.arg_head.tile_keys[ti_atk].weight.add_(torch.randn(net.arg_head.tile_keys[ti_atk].weight.shape, generator=gen))
        move_2 = net.arg_logits(f, actors, _full(4, MOVE))
        atk_2 = net.arg_logits(f, actors, _full(4, ATTACK))
        assert torch.equal(move_2, move_1)
        assert not torch.equal(atk_2, atk_1)

        # Mixed batch in one forward call matches per-kind logits.
        mixed_kinds = torch.tensor([MOVE, ATTACK, MOVE, ATTACK], dtype=torch.int64)
        mixed = net.arg_logits(f, actors, mixed_kinds)
        torch.testing.assert_close(mixed[[0, 2]], move_2[[0, 2]], atol=1e-6, rtol=0)
        torch.testing.assert_close(mixed[[1, 3]], atk_2[[1, 3]], atol=1e-6, rtol=0)


@pytest.mark.parametrize("size", [6, 11])
def test_arg_head_v2_relative_position_bias(size):
    n_tiles = size * size
    spec = _spec(size, "v2")
    torch.manual_seed(8)
    net = PolicyNet(spec, width=16, blocks=1, arch="v2")

    assert "arg_head.rel_index" not in net.state_dict()
    assert net.arg_head.rel_bias.shape == (2, (2 * size - 1) * (2 * size - 1))
    assert torch.all(net.arg_head.rel_bias == 0)

    ti_move = net.arg_head.tile_kinds.index(MOVE)
    ti_atk = net.arg_head.tile_kinds.index(ATTACK)

    # 4 distinct actors so each tests both in-bounds and boundary targets.
    actor_coords = [(1, 2), (3, 4), (0, 1), (size - 1, size - 2)]
    actors = torch.tensor([r * size + c for r, c in actor_coords], dtype=torch.int64)
    board, glob = _real_obs(size, "v2", b=4)

    with torch.no_grad():
        f = net.trunk(board, glob)
        move_base = net.arg_logits(f, actors, _full(4, MOVE))
        atk_base = net.arg_logits(f, actors, _full(4, ATTACK))

        # Mutate move rel_bias at asymmetric offset (dr=+2, dc=-1): target - actor == (2, -1).
        dr, dc = 2, -1
        delta = 3.5
        offset_idx = (dr + size - 1) * (2 * size - 1) + (dc + size - 1)
        net.arg_head.rel_bias[ti_move, offset_idx] = delta

        move_after = net.arg_logits(f, actors, _full(4, MOVE))
        atk_after = net.arg_logits(f, actors, _full(4, ATTACK))
        assert torch.equal(atk_after, atk_base), "move rel_bias leaked into attack logits"

        expected_shift = torch.zeros(4, n_tiles)
        for b, (ra, ca) in enumerate(actor_coords):
            rt, ct = ra + dr, ca + dc
            if 0 <= rt < size and 0 <= ct < size:
                expected_shift[b, rt * size + ct] = delta
        assert expected_shift.sum().item() > 0
        torch.testing.assert_close(move_after - move_base, expected_shift, atol=1e-5, rtol=0)

        # Explicitly verify reversed offset (actor - target) == (dr, dc) was NOT shifted.
        for b, (ra, ca) in enumerate(actor_coords):
            rt_rev, ct_rev = ra - dr, ca - dc
            if 0 <= rt_rev < size and 0 <= ct_rev < size:
                assert abs((move_after - move_base)[b, rt_rev * size + ct_rev].item()) < 1e-5

        # Mutate attack rel_bias at 3-tile offset (dr=-1, dc=+3): shifts only attack logits.
        dr2, dc2 = -1, 3
        delta2 = -2.25
        offset_idx2 = (dr2 + size - 1) * (2 * size - 1) + (dc2 + size - 1)
        net.arg_head.rel_bias[ti_atk, offset_idx2] = delta2

        move_after2 = net.arg_logits(f, actors, _full(4, MOVE))
        atk_after2 = net.arg_logits(f, actors, _full(4, ATTACK))
        assert torch.equal(move_after2, move_after), "attack rel_bias leaked into move logits"

        expected_atk_shift = torch.zeros(4, n_tiles)
        for b, (ra, ca) in enumerate(actor_coords):
            rt, ct = ra + dr2, ca + dc2
            if 0 <= rt < size and 0 <= ct < size:
                expected_atk_shift[b, rt * size + ct] = delta2
        assert expected_atk_shift.sum().item() > 0
        torch.testing.assert_close(atk_after2 - atk_base, expected_atk_shift, atol=1e-5, rtol=0)


@pytest.mark.parametrize("ruleset", ["v1", "v2"])
def test_v2_full_gradient_flow_and_head_isolation(ruleset):
    size = 6
    n_tiles = size * size
    g_actor = 3 * n_tiles
    spec = _spec(size, ruleset)
    n_kinds = len(spec["action_types"])
    arg_lens = spec["arg_lens"]

    torch.manual_seed(9)
    net = PolicyNet(spec, width=16, blocks=2, arch="v2")

    # Build one batch covering MOVE, ATTACK, every enum kind across all 4 actor families, and END_TURN.
    rows = [
        (7, MOVE, 13, n_tiles),
        (14, ATTACK, 20, n_tiles),
        (g_actor, END_TURN, 0, 0),
    ]
    for k, alen in enumerate(arg_lens):
        if alen > 0 and k not in (MOVE, ATTACK):
            name = spec["action_types"][k]
            if name in ("train", "choose_reward"):
                actor = n_tiles + 9  # city family (1)
            elif name == "research":
                actor = g_actor  # global family (3)
            else:
                actor = 2 * n_tiles + 11  # site family (2)
            rows.append((actor, k, min(1, alen - 1), alen))

    b = len(rows)
    board, glob = _real_obs(size, ruleset, b=b)
    actors = torch.tensor([r[0] for r in rows], dtype=torch.int64)
    kinds = torch.tensor([r[1] for r in rows], dtype=torch.int64)
    args = torch.tensor([r[2] for r in rows], dtype=torch.int64)
    actor_mask = torch.ones(b, 3 * n_tiles + 1, dtype=torch.bool)
    kind_mask = torch.ones(b, n_kinds, dtype=torch.bool)
    arg_mask = torch.zeros(b, n_tiles, dtype=torch.bool)
    for i, (_, _, _, alen) in enumerate(rows):
        if alen > 0:
            arg_mask[i, :alen] = True

    net.zero_grad(set_to_none=True)
    ev = net.evaluate(board, glob, actors, kinds, args, actor_mask, kind_mask, arg_mask)
    loss = -(ev.logp.mean()) + 0.1 * ev.entropy.mean() + ev.value.pow(2).mean()
    loss.backward()

    for name, p in net.named_parameters():
        assert p.grad is not None, f"missing grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        # An additive scalar broadcast across all logits of a softmax is mathematically shift-invariant.
        if name == "arg_head.tile_bias":
            continue
        assert p.grad.abs().sum().item() > 0, f"zero grad on {name}"

    # Verify both move and attack slices of rel_bias received non-zero gradients.
    assert net.arg_head.rel_bias.grad[0].abs().sum().item() > 0
    assert net.arg_head.rel_bias.grad[1].abs().sum().item() > 0

    # No-arg kind (END_TURN) isolates arg_head and value_head from policy loss gradients.
    net.zero_grad(set_to_none=True)
    ev_end = net.evaluate(
        board[:2],
        glob[:2],
        _full(2, g_actor),
        _full(2, END_TURN),
        _full(2, 0),
        actor_mask[:2],
        kind_mask[:2],
        torch.zeros(2, n_tiles, dtype=torch.bool),
    )
    (ev_end.logp.sum() + ev_end.entropy.sum()).backward()
    assert not any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in net.arg_head.parameters())
    assert not any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in net.value_head.parameters())


# ---------------------------------------------------------------- 9. StackedSnapshots (vmap) parity on arch="v2"


@pytest.mark.parametrize("ruleset", ["v1", "v2"])
def test_v2_stacked_snapshots_vmap_parity(ruleset, monkeypatch):
    size = 6
    spec = _spec(size, ruleset)
    nets = [_random_net_v2(spec, seed=30 + i, width=8, blocks=2) for i in range(3)]
    entries = [entry_of(net, 100 * i) for i, net in enumerate(nets)]

    stack = StackedSnapshots(_random_net_v2(spec, seed=99, width=8, blocks=2), "cpu")
    stack.sync(entries)

    # Groups in non-sorted snapshot order with uneven sizes.
    which = [2, 0, 1]
    sizes = [2, 1, 3]
    envs = _positions(sum(sizes), seed=12, size=size, ruleset=ruleset, max_steps=6)
    env_groups = []
    k0 = 0
    for s in sizes:
        env_groups.append(envs[k0 : k0 + s])
        k0 += s
    groups = [(entries[w].name, g_envs) for w, g_envs in zip(which, env_groups)]

    # 1. Direct vmapped trunk parity against each snapshot's own PolicyNet.trunk.
    with torch.inference_mode():
        b_max = max(sizes)
        padded_boards, padded_globs = [], []
        for g_envs in env_groups:
            gb, gg = obs_tensors(g_envs)
            if len(g_envs) < b_max:
                pad_n = b_max - len(g_envs)
                gb = torch.cat([gb, gb[:1].expand(pad_n, -1, -1, -1)], dim=0)
                gg = torch.cat([gg, gg[:1].expand(pad_n, -1)], dim=0)
            padded_boards.append(gb)
            padded_globs.append(gg)
        idx = torch.as_tensor([stack.index(entries[w].name) for w in which])
        params = {k: v.index_select(0, idx) for k, v in stack.params.items()}
        buffers = {k: v.index_select(0, idx) for k, v in stack.buffers.items()}
        fmap_v, g_v = vmap(lambda p, b, x, y: functional_call(stack.module, (p, b), ("trunk", x, y)))(
            params, buffers, torch.stack(padded_boards), torch.stack(padded_globs)
        )
        for gi, (w, g_envs) in enumerate(zip(which, env_groups)):
            ref_feats = nets[w].trunk(*obs_tensors(g_envs))
            torch.testing.assert_close(fmap_v[gi, : len(g_envs)], ref_feats.fmap, atol=1e-5, rtol=0)
            torch.testing.assert_close(g_v[gi, : len(g_envs)], ref_feats.g, atol=1e-5, rtol=0)

    # 2. Sampled stacked_batch_policy stage logits match each snapshot's individual forward.
    spy = SampleSpy(monkeypatch)
    policy = stacked_batch_policy(stack, torch.Generator().manual_seed(7), greedy=False)
    actions = policy(groups)
    assert [len(a) for a in actions] == sizes
    flat = [tuple(int(x) for x in a) for acts in actions for a in acts]
    for env, act in zip(envs, flat):
        assert act in all_paths(env)

    actors = np.array([a for a, _, _ in flat])
    kinds = np.array([k for _, k, _ in flat])
    want = [[], [], []]
    k0 = 0
    for w, g_envs in zip(which, env_groups):
        b = len(g_envs)
        for stage, t in enumerate(ref_stage_logits(nets[w], g_envs, actors[k0 : k0 + b], kinds[k0 : k0 + b])):
            want[stage].append(t)
        k0 += b
    torch.testing.assert_close(spy.calls[0]["logits"], torch.cat(want[0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(spy.calls[1]["logits"], torch.cat(want[1]), atol=1e-5, rtol=0)
    torch.testing.assert_close(spy.calls[2]["logits"], torch.cat(want[2]), atol=1e-5, rtol=0)

    # 3. Greedy actions match per-snapshot net_batch_policy identically.
    want_greedy = [
        [tuple(int(x) for x in a) for a in net_batch_policy(nets[w], torch.Generator().manual_seed(0), greedy=True)(g_envs)]
        for w, g_envs in zip(which, env_groups)
    ]
    got_greedy = stacked_batch_policy(stack, torch.Generator().manual_seed(0), greedy=True)(groups)
    assert [[tuple(int(x) for x in a) for a in acts] for acts in got_greedy] == want_greedy


# ---------------------------------------------------------------- 10. End-to-end training, self-play, checkpoint & eval


def test_v2_end_to_end_train_stacked_selfplay_checkpoint_and_eval(tmp_path):
    # Verify v1 hashes remain completely untouched.
    v1_spec_11 = _spec(11, "v1")
    assert v1_spec_11["rules_hash"] == "a8b52b91eae34503"
    assert v1_spec_11["obs_hash"] == "d5f9b1033cfd55ca"

    cfg = PPOConfig(
        size=6,
        turn_cap=4,
        ruleset="v2",
        arch="v2",
        selfplay=True,
        rollout="interleaved",
        opponent_forward="stacked",
        p_bot=0.0,
        width=8,
        blocks=2,
        num_envs=2,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        total_steps=32,
        snapshot_every=16,
        eval_every=32,
        eval_seeds=1,
        rating_every=64,
        tournament_seeds=1,
        device="cpu",
        seed=1,
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    v2_spec = _spec(6, "v2")
    run_dir = tmp_path / "arch_v2_sp"

    net = train.run(cfg, str(run_dir))
    assert net.stem.in_channels == v2_spec["channels"] + 2
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "ckpt_final.pt").exists()
    assert (run_dir / "pool" / "snap_0.pt").exists()

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert np.isfinite(row["pg_loss"]) and np.isfinite(row["v_loss"]) and np.isfinite(row["entropy"])
        assert len(row["kind_freq"]) == 15

    loaded, ck = load_net(str(run_dir / "latest.pt"), device="cpu")
    assert ck.cfg.arch == "v2"
    assert ck.cfg.ruleset == "v2"
    assert loaded.stem.in_channels == v2_spec["channels"] + 2
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in net.parameters())
    assert list(loaded.state_dict().keys()) == list(net.state_dict().keys())
    for k, v in net.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), f"weight mismatch at {k}"

    ev = evaluate(net_policy(loaded), "random", seeds=1, env_kwargs=env_kwargs_of(cfg), greedy=True)
    assert ev.games == 2
