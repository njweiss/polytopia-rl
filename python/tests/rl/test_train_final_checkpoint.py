"""Every training run ends with `ckpt_final.pt`, bot training included (owner decision 2026-10-28).

The experiment harness (milestones 12 and 14) finds finished runs, plays the round robin, reads the
cost columns and resolves `init_from` through each run's `ckpt_final.pt`. Before this fix only self-play
runs wrote it, so the 14a smoke's bot-curriculum phase (`scale14_bots`) failed. Tiny CPU config as in
`test_train.py`: batch 4 x 32 = 128 steps, 4 updates, an eval (and checkpoint) every 256 steps.
"""

import torch

from polytopia_rl import train
from polytopia_rl.config import PPOConfig

BATCH = 128
UPDATES = 4


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
        seed=3,
        opponents=["random"],
        stages=[{"opponents": ["random"]}],
    )
    return PPOConfig(**{**base, **kw})


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same_state(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[k], b[k]) for k in a)


def test_bot_training_writes_ckpt_final_with_the_final_weights(tmp_path):
    net = train.run(tiny_cfg(), str(tmp_path))
    final = load(tmp_path / "ckpt_final.pt")
    assert final["global_step"] == UPDATES * BATCH
    assert same_state(final["model"], net.state_dict())
    assert same_state(final["model"], load(tmp_path / "latest.pt")["model"])
    assert same_state(final["model"], load(tmp_path / f"ckpt_{UPDATES * BATCH}.pt")["model"])


def test_bot_training_ckpt_final_loads_with_load_net(tmp_path):
    from polytopia_rl.checkpoint import load_net

    train.run(tiny_cfg(width=16, blocks=2), str(tmp_path))
    net, ck = load_net(str(tmp_path / "ckpt_final.pt"), "cpu")
    assert (ck.cfg.width, ck.cfg.blocks) == (16, 2)
    assert ck.global_step == UPDATES * BATCH
