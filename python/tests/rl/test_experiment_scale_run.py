"""Milestone 14 tests 2, 3 (launcher side) and 4: `init_from`, `reference`, mixed architectures, and
specs without the new fields ("Rules", "Changes to earlier tests").

As in test_experiment_run.py, a fake `subprocess.Popen` stands in for the trainer: a `python -m
polytopia_rl.train` command stays running for `POLLS - 1` polls, then writes tiny real checkpoints
(6x6 map, turn cap 3) at steps 100 to 400 plus `latest.pt`, `ckpt_final.pt` and a `metrics.jsonl`.
Unlike there, the checkpoints have the run's own `width` and `blocks` (from the command's config and
overrides), so arms of different sizes give networks of different sizes. Like the real trainer, the
fake fails (exit 1, message in its log) when `--init-checkpoint` names a file that does not exist.

`load_net` and `play_pair` are wrapped (the real functions still run) to record which checkpoint
files were loaded and which players met.

Expected values from the brief: with `init_from` the command of run (arm, seed) carries
`--init-checkpoint runs/exp/{init_from}/{arm}/seed{seed}/ckpt_final.pt` in place of an init
checkpoint, `--resume latest.pt` still wins for a partly trained run, and a missing source fails only
that run with a message naming the path; with `reference` the head-to-heads (`h2h_{step}.json`) are
against it (player "ref") and not against the init checkpoint, and the round robin has "ref" and no
"init"; the reference may be another architecture. Without the new fields the commands, files and
report keys are the milestone 12 ones, enumerated here by hand.
"""

import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import polytopia_sim as ps
import pytest
import torch

import polytopia_rl.checkpoint
import polytopia_rl.tournament
from polytopia_rl import experiment
from polytopia_rl.checkpoint import save_checkpoint
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.model import PolicyNet
from polytopia_rl.suite import DEFAULT_SUITE

STEPS = [100, 200, 300, 400]
EVAL_STEPS = [200, 300, 400]
POLLS = 3
REAL_POPEN = subprocess.Popen
BOTS = ["random", "economy", "rush"]
SPEC6 = ps.Env(size=6).spec()


def tiny_base():
    return PPOConfig(size=6, turn_cap=3, width=8, blocks=1, device="cpu", selfplay=True, opponents=["economy", "rush"], stages=[{"opponents": ["economy", "rush"]}])


def write_ckpt(path, cfg, step):
    torch.manual_seed(1000 * cfg.seed + step)
    net = PolicyNet(SPEC6, width=cfg.width, blocks=cfg.blocks)
    save_checkpoint(str(path), model=net, optimizer=None, cfg=cfg, global_step=step, stage=0, seed_index=None, rng_state=None, spec=SPEC6)


def sized(width, blocks, seed=0):
    return dataclasses.replace(tiny_base(), width=width, blocks=blocks, seed=seed)


def fake_train_outputs(run_dir, cfg, steps=STEPS, final=True):
    os.makedirs(run_dir, exist_ok=True)
    for s in steps:
        p = os.path.join(run_dir, f"ckpt_{s}.pt")
        if not os.path.exists(p):
            write_ckpt(p, cfg, s)
    write_ckpt(os.path.join(run_dir, "latest.pt"), cfg, steps[-1])
    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
        for i, s in enumerate(steps):
            f.write(json.dumps({"global_step": s, "update": i + 1, "sps": 500.0 + 10 * i}) + "\n")
    if final:
        write_ckpt(os.path.join(run_dir, "ckpt_final.pt"), cfg, steps[-1])


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


class Lab:
    """What the fake subprocesses, `load_net` and `play_pair` saw."""

    def __init__(self):
        self.launches = []  # argv in launch order
        self.loaded = []  # normalized checkpoint paths given to load_net
        self.pairs = []  # (a.name, b.name) of every play_pair call

    def popen_class(lab):
        class FakePopen:
            def __init__(self, args, **kwargs):
                args = [str(a) for a in args]
                if "polytopia_rl.train" not in args:
                    self.real = REAL_POPEN(args, **kwargs)
                    return
                self.real = None
                self.args = args
                self.returncode = None
                self.pid = 4242
                self.polls = 0
                self.log = kwargs.get("stdout")
                self.run_dir = flag(args, "--run-dir")
                cmd = os.path.join(self.run_dir, "command.txt")
                assert os.path.exists(cmd), "command.txt is written before the run starts"
                with open(cmd) as f:
                    assert shlex.split(f.readline()) == args, "command.txt starts with the exact command"
                lab.launches.append(args)

            def _finish(self):
                init = flag(self.args, "--init-checkpoint")
                if init is not None and "--resume" not in self.args and not os.path.exists(init):
                    if hasattr(self.log, "write"):
                        self.log.write(f"FileNotFoundError: {init}\n")
                    self.returncode = 1
                    return
                overrides = [a for a in self.args if "=" in a and not a.startswith("-")]
                fake_train_outputs(self.run_dir, load_config(flag(self.args, "--config"), overrides))
                self.returncode = 0

            def poll(self):
                if self.real is not None:
                    return self.real.poll()
                if self.returncode is None:
                    self.polls += 1
                    if self.polls >= POLLS:
                        self._finish()
                return self.returncode

            def wait(self, timeout=None):
                if self.real is not None:
                    return self.real.wait(timeout)
                if self.returncode is None:
                    self._finish()
                return self.returncode

            def kill(self):
                pass

            terminate = kill

            def communicate(self, *a, **k):
                if self.real is not None:
                    return self.real.communicate(*a, **k)
                self.wait()
                return None, None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.wait()

            def __getattr__(self, name):
                real = self.__dict__.get("real")
                if real is None:
                    raise AttributeError(name)
                return getattr(real, name)

        return FakePopen


def install(mp, lab):
    cls = lab.popen_class()
    mp.setattr(subprocess, "Popen", cls)
    if hasattr(experiment, "Popen"):
        mp.setattr(experiment, "Popen", cls)
    mp.setattr(time, "sleep", lambda s: None)
    real_load_net = polytopia_rl.checkpoint.load_net
    real_play_pair = polytopia_rl.tournament.play_pair

    def load_net(path, *a, **k):
        lab.loaded.append(os.path.normpath(str(path)))
        return real_load_net(path, *a, **k)

    def play_pair(a, b, *rest, **k):
        lab.pairs.append((a.name, b.name))
        return real_play_pair(a, b, *rest, **k)

    mp.setattr(polytopia_rl.checkpoint, "load_net", load_net)
    mp.setattr(polytopia_rl.tournament, "play_pair", play_pair)
    if hasattr(experiment, "load_net"):
        mp.setattr(experiment, "load_net", load_net)
    if hasattr(experiment, "play_pair"):
        mp.setattr(experiment, "play_pair", play_pair)


def make_world(root, arms, seeds, init=False, reference=None, init_from=None, name="t"):
    """Base config, optional init (width 8, 1 block) and reference (width 4, 3 blocks: an
    architecture no arm has) checkpoints, and the spec file under `root` (the working directory)."""
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "tiny.json").write_text(tiny_base().to_json())
    if init:
        write_ckpt(root / "init.pt", tiny_base(), 5_000_000)
    if reference is not None:
        write_ckpt(root / reference, sized(4, 3), 20_000_000)
    d = dict(
        name=name,
        base_config="configs/tiny.json",
        seeds=seeds,
        total_steps=400,
        arms=arms,
        primary=["rush_e0"],
        suite_seeds=1,
        tournament_seeds=1,
        eval_checkpoints=3,
        h2h_seeds=1,
    )
    if init:
        d["init_checkpoint"] = "init.pt"
    if reference is not None:
        d["reference"] = reference
    if init_from is not None:
        d["init_from"] = init_from
    (root / "spec.json").write_text(json.dumps(d))
    return str(root / "spec.json")


def rd(arm, seed, name="t"):
    return os.path.join("runs", "exp", name, arm, f"seed{seed}")


def source(arm, seed, src="src"):
    return os.path.join("runs", "exp", src, arm, f"seed{seed}", "ckpt_final.pt")


def normalized(argv):
    out = list(argv)
    for name in ("--config", "--run-dir", "--init-checkpoint", "--resume"):
        if name in out:
            i = out.index(name) + 1
            out[i] = os.path.normpath(out[i])
    return out


def fresh_argv(arm, seed, overrides, init=None):
    out = ["-m", "polytopia_rl.train", "--config", os.path.join("configs", "tiny.json"), "--run-dir", rd(arm, seed)]
    if init is not None:
        out += ["--init-checkpoint", init]
    return out + [f"seed={seed}", "total_steps=400"] + overrides


def check_python(argv):
    assert argv[0] in (sys.executable, "python", "python3"), argv[0]


def tournament(root, name="t"):
    return json.loads((root / "runs" / "exp" / name / "tournament.json").read_text())


def run_world(root, spec_path, parallel=2):
    lab = Lab()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        install(mp, lab)
        spec = experiment.load_spec(spec_path)
        experiment.run(spec, parallel=parallel)
    return lab, spec


# ---------------------------------------------------------------- reference, mixed architectures

MIXED = {"w8b1": {"width": 8, "blocks": 1}, "w16b2": {"width": 16, "blocks": 2}}


@pytest.fixture(scope="module")
def phase1(tmp_path_factory):
    """Phase 1 shape: no init, a reference of another architecture, arms of two sizes."""
    root = tmp_path_factory.mktemp("p1")
    lab, spec = run_world(root, make_world(root, MIXED, [1], reference="ref.pt"))
    return root, lab, spec


def test_reference_does_not_enter_the_train_command(phase1):
    _, lab, _ = phase1
    for a in lab.launches:
        check_python(a)
    assert [normalized(a)[1:] for a in lab.launches] == [
        fresh_argv("w8b1", 1, ["width=8", "blocks=1"]),
        fresh_argv("w16b2", 1, ["width=16", "blocks=2"]),
    ]


def test_runs_have_their_arms_architecture(phase1):
    # Sanity of the fixture: the two arms and the reference are three different networks.
    root, _, _ = phase1
    shapes = {}
    for arm in MIXED:
        ck = torch.load(root / rd(arm, 1) / "ckpt_final.pt", weights_only=False)
        cfg = PPOConfig.from_json(ck["config"])
        shapes[arm] = (cfg.width, cfg.blocks)
    ref = PPOConfig.from_json(torch.load(root / "ref.pt", weights_only=False)["config"])
    assert shapes == {"w8b1": (8, 1), "w16b2": (16, 2)}
    assert (ref.width, ref.blocks) == (4, 3)


def test_head_to_heads_are_against_the_reference(phase1):
    root, lab, _ = phase1
    assert os.path.normpath("ref.pt") in lab.loaded, "the reference checkpoint is loaded"
    for arm in MIXED:
        d = root / rd(arm, 1)
        for s in EVAL_STEPS:
            h = json.loads((d / f"h2h_{s}.json").read_text())
            assert h["b"] == "ref"
            assert h["games"] == 2 and h["a_wins"] + h["b_wins"] + h["draws"] == 2
        assert not (d / "h2h_100.json").exists(), "only the evaluated checkpoints"
        assert sorted(p.name for p in d.glob("h2h_*.json")) == sorted(f"h2h_{s}.json" for s in EVAL_STEPS)
        assert sum(1 for p in lab.pairs if p == (f"{arm}_seed1", "ref")) >= len(EVAL_STEPS)


def test_round_robin_has_ref_and_mixed_sizes(phase1):
    root, lab, spec = phase1
    t = tournament(root)
    want = BOTS + ["ref", "w8b1_seed1", "w16b2_seed1"]
    assert t["players"] == want
    assert experiment.tournament_players(spec) == want
    games = np.array(t["games"])
    n = len(want)
    assert np.all(games[~np.eye(n, dtype=bool)] == 2), "every pair played, mixed sizes included"
    assert all(np.isfinite(t["ratings"]))
    assert ("w8b1_seed1", "w16b2_seed1") in lab.pairs
    assert ("ref", "w8b1_seed1") in lab.pairs and ("ref", "w16b2_seed1") in lab.pairs


def test_suites_are_written_with_a_reference(phase1):
    root, _, _ = phase1
    for arm in MIXED:
        for s in EVAL_STEPS:
            assert (root / rd(arm, 1) / f"suite_{s}" / "suite.json").exists(), (arm, s)


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    """An init checkpoint and a reference: training starts from init, evaluation is against ref."""
    root = tmp_path_factory.mktemp("both")
    lab, spec = run_world(root, make_world(root, {"base": {}}, [1], init=True, reference="ref.pt"))
    return root, lab, spec


def test_init_checkpoint_still_starts_training_with_a_reference(both):
    _, lab, _ = both
    assert [normalized(a)[1:] for a in lab.launches] == [fresh_argv("base", 1, [], init="init.pt")]


def test_reference_replaces_init_in_the_head_to_head(both):
    root, lab, _ = both
    assert os.path.normpath("init.pt") not in lab.loaded, "the init checkpoint is not an opponent"
    assert os.path.normpath("ref.pt") in lab.loaded
    assert all("init" not in p for p in lab.pairs)
    for s in EVAL_STEPS:
        h = json.loads((root / rd("base", 1) / f"h2h_{s}.json").read_text())
        assert h["b"] == "ref"


def test_reference_replaces_init_in_the_round_robin(both):
    root, _, spec = both
    want = BOTS + ["ref", "base_seed1"]
    assert tournament(root)["players"] == want
    assert experiment.tournament_players(spec) == want


# ---------------------------------------------------------------- init_from

SIZES = {"w8b1": (8, 1), "w16b2": (16, 2)}


def write_sources(root, pairs, src="src"):
    for arm, seed in pairs:
        path = root / source(arm, seed, src)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_ckpt(path, sized(*SIZES[arm], seed=seed), 10_000_000)


def test_train_command_with_init_from(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = experiment.load_spec(make_world(tmp_path, MIXED, [1, 2], init_from="scale14_bots"))
    for arm, ov in (("w8b1", ["width=8", "blocks=1"]), ("w16b2", ["width=16", "blocks=2"])):
        for seed in (1, 2):
            got = normalized(experiment.train_command(spec, arm, seed))
            check_python(got)
            assert got[1:] == fresh_argv(arm, seed, ov, init=source(arm, seed, "scale14_bots")), (arm, seed)


@pytest.fixture(scope="module")
def phase2(tmp_path_factory):
    """Phase 2 shape: every run warm-starts from its own (arm, seed) of experiment `src`."""
    root = tmp_path_factory.mktemp("p2")
    write_sources(root, [(a, s) for a in SIZES for s in (1, 2)])
    lab, spec = run_world(root, make_world(root, MIXED, [1, 2], init_from="src"))
    return root, lab, spec


def test_init_from_commands_name_each_runs_own_source(phase2):
    _, lab, _ = phase2
    got = [normalized(a) for a in lab.launches]
    for a in got:
        check_python(a)
    assert [a[1:] for a in got] == [
        fresh_argv("w8b1", 1, ["width=8", "blocks=1"], init=source("w8b1", 1)),
        fresh_argv("w8b1", 2, ["width=8", "blocks=1"], init=source("w8b1", 2)),
        fresh_argv("w16b2", 1, ["width=16", "blocks=2"], init=source("w16b2", 1)),
        fresh_argv("w16b2", 2, ["width=16", "blocks=2"], init=source("w16b2", 2)),
    ]


def test_init_from_command_txt_records_the_source(phase2):
    root, _, _ = phase2
    for arm in SIZES:
        for seed in (1, 2):
            argv = normalized(shlex.split((root / rd(arm, seed) / "command.txt").read_text().splitlines()[0]))
            assert flag(argv, "--init-checkpoint") == source(arm, seed), (arm, seed)


def test_init_from_runs_finish_and_are_evaluated(phase2):
    root, _, _ = phase2
    for arm in SIZES:
        for seed in (1, 2):
            d = root / rd(arm, seed)
            assert (d / "ckpt_final.pt").exists()
            for s in EVAL_STEPS:
                assert (d / f"suite_{s}" / "suite.json").exists(), (arm, seed, s)
    assert set(tournament(root)["players"]) >= {f"{a}_seed{s}" for a in SIZES for s in (1, 2)}


def test_init_from_a_missing_source_fails_that_run_only(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    lab = Lab()
    install(monkeypatch, lab)
    write_sources(tmp_path, [("w8b1", 1), ("w8b1", 3)])
    path = make_world(tmp_path, {"w8b1": {"width": 8, "blocks": 1}}, [1, 2, 3], init_from="src")
    with pytest.raises(BaseException) as err:
        experiment.main(["run", path])
    if isinstance(err.value, SystemExit):
        assert err.value.code not in (0, None)
    for seed in (1, 3):
        d = tmp_path / rd("w8b1", seed)
        assert (d / "ckpt_final.pt").exists(), seed
        for s in EVAL_STEPS:
            assert (d / f"suite_{s}" / "suite.json").exists(), (seed, s)
    d2 = tmp_path / rd("w8b1", 2)
    assert not (d2 / "ckpt_final.pt").exists()
    argv = normalized(shlex.split((d2 / "command.txt").read_text().splitlines()[0]))
    assert flag(argv, "--init-checkpoint") == source("w8b1", 2), "command.txt records the actual path"
    out = capsys.readouterr()
    missing = source("w8b1", 2).replace(os.sep, "/")
    lines = [line for line in (out.out + out.err).replace(os.sep, "/").splitlines() if missing in line]
    # The echoed command (`start ...: python -m polytopia_rl.train ...`) names the path too; it is not
    # the failure message. The launcher itself says which source is missing (docs/QUESTIONS.md Q41).
    assert [line for line in lines if "polytopia_rl.train" not in line], "the launcher's message names the missing source"


def test_init_from_resume_still_uses_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lab = Lab()
    install(monkeypatch, lab)
    write_sources(tmp_path, [("w8b1", 1)])
    spec = experiment.load_spec(make_world(tmp_path, {"w8b1": {"width": 8, "blocks": 1}}, [1], init_from="src"))
    fake_train_outputs(rd("w8b1", 1), sized(8, 1, seed=1), steps=[100, 200], final=False)  # crashed
    experiment.run(spec)
    (resumed,) = [normalized(a) for a in lab.launches]
    i = resumed.index("--resume")
    assert resumed[i + 1] == os.path.join(rd("w8b1", 1), "latest.pt")
    rest = resumed[:i] + resumed[i + 2 :]
    if "--init-checkpoint" in rest:  # ignored by the trainer with --resume; tolerated as in milestone 12
        j = rest.index("--init-checkpoint")
        assert rest[j + 1] == source("w8b1", 1)
        rest = rest[:j] + rest[j + 2 :]
    assert rest[1:] == fresh_argv("w8b1", 1, ["width=8", "blocks=1"])
    assert (tmp_path / rd("w8b1", 1) / "ckpt_final.pt").exists()


# ---------------------------------------------------------------- specs without the new fields

OLD_ARMS = {"base": {"p_bot": 0.2}, "ent003": {"ent_coef": 0.003}}
OLD_RUN_KEYS = {"arm", "seed", "steps", "suite", "suite_mean", "elo_vs_init", "elo_vs_init_mean", "obs_hash", "rules_hash"}
OLD_REPORT_KEYS = {"name", "arms", "seeds", "runs", "tables", "comparisons", "n_flagged", "elo", "obs_hash"}


@pytest.fixture(scope="module")
def old_style(tmp_path_factory):
    """A 12b-style spec (init checkpoint, no reference, no init_from), run then reported."""
    root = tmp_path_factory.mktemp("old")
    lab, spec = run_world(root, make_world(root, OLD_ARMS, [1, 2], init=True))
    cwd = os.getcwd()
    try:
        os.chdir(root)
        rep = experiment.report(spec)
    finally:
        os.chdir(cwd)
    return root, lab, spec, rep


def test_old_spec_commands_are_unchanged(old_style):
    _, lab, spec, _ = old_style
    assert (spec.reference, spec.init_from) == (None, None)
    assert [normalized(a)[1:] for a in lab.launches] == [
        fresh_argv("base", 1, ["p_bot=0.2"], init="init.pt"),
        fresh_argv("base", 2, ["p_bot=0.2"], init="init.pt"),
        fresh_argv("ent003", 1, ["ent_coef=0.003"], init="init.pt"),
        fresh_argv("ent003", 2, ["ent_coef=0.003"], init="init.pt"),
    ]


def test_old_spec_files_are_unchanged(old_style):
    root, _, _, _ = old_style
    exp = root / "runs" / "exp" / "t"
    run_files = {"command.txt", "train.out", "metrics.jsonl", "latest.pt", "ckpt_final.pt"}
    run_files |= {f"ckpt_{s}.pt" for s in STEPS} | {f"h2h_{s}.json" for s in EVAL_STEPS}
    run_files |= {os.path.join(f"suite_{s}", n) for s in EVAL_STEPS for n in ("suite.json", "suite.md")}
    want = {os.path.join(arm, f"seed{seed}", f) for arm in OLD_ARMS for seed in (1, 2) for f in run_files}
    want |= {"tournament.json", "tournament.md", "report.json", "report.md", "report_curves.png"}
    got = {os.path.relpath(os.path.join(dp, f), exp) for dp, _, fs in os.walk(exp) for f in fs}
    assert got == want
    for arm in OLD_ARMS:
        for seed in (1, 2):
            for s in EVAL_STEPS:
                h = json.loads((exp / arm / f"seed{seed}" / f"h2h_{s}.json").read_text())
                assert h["b"] == "init"


def test_old_spec_round_robin_is_unchanged(old_style):
    root, lab, spec, _ = old_style
    want = BOTS + ["init"] + [f"{a}_seed{s}" for a in OLD_ARMS for s in (1, 2)]
    assert tournament(root)["players"] == want
    assert experiment.tournament_players(spec) == want
    assert all("ref" not in p for p in lab.pairs)


def test_old_spec_report_keys_are_unchanged(old_style):
    root, _, _, rep = old_style
    conds = [c.name for c in DEFAULT_SUITE]
    assert list(rep.tables) == conds + ["elo_vs_init"]
    for r in rep.runs:
        assert set(r) == OLD_RUN_KEYS
        assert r["elo_vs_init"] is not None and len(r["elo_vs_init"]) == len(EVAL_STEPS)
    assert rep.elo is not None and "init" in rep.elo and "ref" not in rep.elo
    got = set(json.loads((root / "runs" / "exp" / "t" / "report.json").read_text()))
    assert got == OLD_REPORT_KEYS | {"cost"}, "the cost columns are the only new report key"
