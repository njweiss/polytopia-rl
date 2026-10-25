"""Milestone 12 tests 6: the experiment launcher ("Experiment runs"), with a fake trainer subprocess.

`FakePopen` replaces `subprocess.Popen`. A `python -m polytopia_rl.train` command does not train: it
stays running for `POLLS - 1` calls of `poll()`, then "finishes" by writing tiny real checkpoints
(6x6 map, turn cap 3, width 8, one block) at steps 100, 200, 300 and 400 plus `latest.pt` and
`ckpt_final.pt` (a resumed command writes only the steps it does not have yet). Any other command is
passed to the real `Popen`. The launcher's evaluations then play real (tiny) games: the suite with
1 seed (2 games per condition), the head-to-head with 1 seed, the round robin with 1 seed per pair.

Expected values from the brief: commands `python -m polytopia_rl.train --config BASE --run-dir DIR
[--init-checkpoint CKPT] seed=S total_steps=N k=v ...` in spec order (arms outer, seeds inner),
`OMP_NUM_THREADS=1` unless set, `command.txt` written first; with `eval_checkpoints` 3 the evaluated
steps are 200, 300 and 400; the round robin's players are the bots, `init` and every final policy
(named `{arm}_seed{seed}`, docs/QUESTIONS.md Q31).
"""

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

from polytopia_rl import experiment
from polytopia_rl.checkpoint import save_checkpoint
from polytopia_rl.config import PPOConfig, load_config
from polytopia_rl.model import PolicyNet
from polytopia_rl.suite import DEFAULT_SUITE, suite_from_json
from polytopia_rl.tournament import RATING_SEED_BASE

STEPS = [100, 200, 300, 400]
EVAL_STEPS = [200, 300, 400]
POLLS = 3
REAL_POPEN = subprocess.Popen
BOTS = {"random", "economy", "rush"}


def tiny_base():
    return PPOConfig(size=6, turn_cap=3, width=8, blocks=1, device="cpu", selfplay=True, opponents=["economy", "rush"], stages=[{"opponents": ["economy", "rush"]}])


def write_ckpt(path, cfg, step):
    torch.manual_seed(1000 * cfg.seed + step)
    net = PolicyNet(ps.Env(size=6).spec(), width=8, blocks=1)
    save_checkpoint(str(path), model=net, optimizer=None, cfg=cfg, global_step=step, stage=0, seed_index=None, rng_state=None, spec=ps.Env(size=6).spec())


def fake_train_outputs(run_dir, cfg, steps=STEPS, final=True):
    os.makedirs(run_dir, exist_ok=True)
    for s in steps:
        p = os.path.join(run_dir, f"ckpt_{s}.pt")
        if not os.path.exists(p):
            write_ckpt(p, cfg, s)
    write_ckpt(os.path.join(run_dir, "latest.pt"), cfg, steps[-1])
    if final:
        write_ckpt(os.path.join(run_dir, "ckpt_final.pt"), cfg, steps[-1])


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


class Lab:
    """What the fake subprocesses saw."""

    def __init__(self, fail=None):
        self.launches = []  # (argv, effective env) in launch order
        self.live = 0
        self.max_live = 0
        self.fail = fail or {}  # normalized run dir -> exit code

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
                self.run_dir = flag(args, "--run-dir")
                cmd = os.path.join(self.run_dir, "command.txt")
                assert os.path.exists(cmd), "command.txt is written before the run starts"
                with open(cmd) as f:
                    assert shlex.split(f.readline()) == args, "command.txt starts with the exact command"
                env = kwargs.get("env")
                lab.launches.append((args, dict(os.environ if env is None else env)))
                lab.live += 1
                lab.max_live = max(lab.max_live, lab.live)

            def _finish(self):
                lab.live -= 1
                code = lab.fail.get(os.path.normpath(self.run_dir), 0)
                overrides = [a for a in self.args if "=" in a and not a.startswith("-")]
                cfg = load_config(flag(self.args, "--config"), overrides)
                if code == 0:
                    fake_train_outputs(self.run_dir, cfg)
                self.returncode = code

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


def install(monkeypatch, lab):
    cls = lab.popen_class()
    monkeypatch.setattr(subprocess, "Popen", cls)
    if hasattr(experiment, "Popen"):
        monkeypatch.setattr(experiment, "Popen", cls)
    monkeypatch.setattr(time, "sleep", lambda s: None)


def make_world(root, arms, seeds, init=True, **spec_kw):
    """Base config, init checkpoint and spec file under `root` (the working directory)."""
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "tiny.json").write_text(tiny_base().to_json())
    if init:
        write_ckpt(root / "init.pt", tiny_base(), 5_000_000)
    d = dict(
        name="t",
        base_config="configs/tiny.json",
        init_checkpoint="init.pt" if init else None,
        seeds=seeds,
        total_steps=400,
        arms=arms,
        primary=["rush_e0"],
        suite_seeds=1,
        tournament_seeds=1,
        eval_checkpoints=3,
        h2h_seeds=1,
    )
    d.update(spec_kw)
    (root / "spec.json").write_text(json.dumps(d))
    return str(root / "spec.json")


def rd(arm, seed):
    return os.path.join("runs", "exp", "t", arm, f"seed{seed}")


def normalized(argv):
    out = list(argv)
    for name in ("--config", "--run-dir", "--init-checkpoint", "--resume"):
        if name in out:
            i = out.index(name) + 1
            out[i] = os.path.normpath(out[i])
    return out


def fresh_argv(arm, seed, overrides, init=True):
    out = ["-m", "polytopia_rl.train", "--config", os.path.join("configs", "tiny.json"), "--run-dir", rd(arm, seed)]
    if init:
        out += ["--init-checkpoint", "init.pt"]
    return out + [f"seed={seed}", "total_steps=400"] + overrides


def check_python(argv):
    assert argv[0] in (sys.executable, "python", "python3"), argv[0]


ARMS = {"base": {}, "pbot40": {"p_bot": 0.4}}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    lab = Lab()
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        mp.delenv("OMP_NUM_THREADS", raising=False)
        install(mp, lab)
        spec = experiment.load_spec(make_world(root, ARMS, [1, 2]))
        experiment.run(spec, parallel=2)
        again = Lab()
        install(mp, again)
        experiment.run(spec, parallel=2)
    return root, lab, again


def test_commands_in_spec_order_with_exact_argv(world):
    _, lab, _ = world
    got = [normalized(a) for a, _ in lab.launches]
    want = [
        fresh_argv("base", 1, []),
        fresh_argv("base", 2, []),
        fresh_argv("pbot40", 1, ["p_bot=0.4"]),
        fresh_argv("pbot40", 2, ["p_bot=0.4"]),
    ]
    for a in got:
        check_python(a)
    assert [a[1:] for a in got] == want


def test_one_thread_per_trainer_by_default(world):
    _, lab, _ = world
    for _, env in lab.launches:
        assert env.get("OMP_NUM_THREADS") == "1"


def test_parallel_runs_at_most_k_and_uses_k(world):
    _, lab, _ = world
    assert lab.max_live == 2


def test_every_evaluated_checkpoint_gets_a_suite_and_a_head_to_head(world):
    root, _, _ = world
    for arm in ARMS:
        for seed in (1, 2):
            d = root / rd(arm, seed)
            assert (d / "command.txt").exists()
            assert not (d / "suite_100").exists() and not (d / "h2h_100.json").exists(), "only the last 3 checkpoints"
            for s in EVAL_STEPS:
                res = suite_from_json((d / f"suite_{s}" / "suite.json").read_text())
                assert res.step == s
                assert list(res.conditions) == [c.name for c in DEFAULT_SUITE]
                assert all(r.games == 2 for r in res.conditions.values())
                assert res.obs_hash == ps.Env(size=6).spec()["obs_hash"]
                h = json.loads((d / f"h2h_{s}.json").read_text())
                assert h["b"] == "init"
                assert h["games"] == 2 and h["a_wins"] + h["b_wins"] + h["draws"] == 2
                assert [g["seed"] for g in h["per_game"]] == [RATING_SEED_BASE, RATING_SEED_BASE]


def test_round_robin_of_final_policies_bots_and_init(world):
    root, _, _ = world
    t = json.loads((root / "runs" / "exp" / "t" / "tournament.json").read_text())
    finals = {f"{arm}_seed{s}" for arm in ARMS for s in (1, 2)}
    assert set(t["players"]) == BOTS | {"init"} | finals
    games = np.array(t["games"])
    n = len(t["players"])
    assert np.all(games[~np.eye(n, dtype=bool)] == 2)
    assert (root / "runs" / "exp" / "t" / "tournament.md").exists()


def test_a_repeated_run_launches_nothing(world):
    _, _, again = world
    assert again.launches == []


def test_skip_resume_and_fresh_by_the_files_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    lab = Lab()
    install(monkeypatch, lab)
    spec = experiment.load_spec(make_world(tmp_path, {"base": {}}, [1, 2, 3]))
    cfg = tiny_base()
    fake_train_outputs(rd("base", 1), cfg)  # done: ckpt_final.pt
    fake_train_outputs(rd("base", 2), cfg, steps=[100, 200], final=False)  # crashed: latest.pt only
    experiment.run(spec)
    got = [normalized(a) for a, _ in lab.launches]
    assert len(got) == 2, "the finished run is skipped"
    resumed, fresh = got
    i = resumed.index("--resume")
    assert resumed[i + 1] == os.path.join(rd("base", 2), "latest.pt")
    rest = resumed[:i] + resumed[i + 2 :]
    if "--init-checkpoint" in rest:
        j = rest.index("--init-checkpoint")
        rest = rest[:j] + rest[j + 2 :]
    assert rest[1:] == fresh_argv("base", 2, [], init=False)
    assert fresh[1:] == fresh_argv("base", 3, [])
    for seed in (1, 2, 3):
        assert (tmp_path / rd("base", seed) / "ckpt_final.pt").exists()
    for seed in (2, 3):
        for s in EVAL_STEPS:
            assert (tmp_path / rd("base", seed) / f"suite_{s}" / "suite.json").exists(), (seed, s)


def test_without_init_checkpoint_no_head_to_head_and_omp_kept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    lab = Lab()
    install(monkeypatch, lab)
    spec = experiment.load_spec(make_world(tmp_path, {"solo": {"ent_coef": 0.003}}, [5], init=False))
    experiment.run(spec)
    ((argv, env),) = lab.launches
    assert normalized(argv)[1:] == fresh_argv("solo", 5, ["ent_coef=0.003"], init=False)
    assert env["OMP_NUM_THREADS"] == "4"
    d = tmp_path / rd("solo", 5)
    for s in EVAL_STEPS:
        assert (d / f"suite_{s}" / "suite.json").exists()
    assert not list(d.glob("h2h_*.json"))
    t = json.loads((tmp_path / "runs" / "exp" / "t" / "tournament.json").read_text())
    assert set(t["players"]) == BOTS | {"solo_seed5"}


def test_a_failing_run_does_not_stop_the_others_and_the_exit_is_non_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    lab = Lab(fail={os.path.normpath(rd("base", 2)): 77})
    install(monkeypatch, lab)
    path = make_world(tmp_path, {"base": {}}, [1, 2, 3])
    with pytest.raises(BaseException) as err:
        experiment.main(["run", path])
    if isinstance(err.value, SystemExit):
        assert err.value.code not in (0, None)
    assert [flag(normalized(a), "--run-dir") for a, _ in lab.launches] == [rd("base", s) for s in (1, 2, 3)]
    for seed in (1, 3):
        d = tmp_path / rd("base", seed)
        assert (d / "ckpt_final.pt").exists()
        for s in EVAL_STEPS:
            assert (d / f"suite_{s}" / "suite.json").exists(), (seed, s)
    out = capsys.readouterr()
    text = out.out + out.err
    assert "77" in text and "seed2" in text, "the failure is reported with its exit code and log path"


def test_cli_passes_parallel_and_report(tmp_path, monkeypatch):
    path = make_world(tmp_path, ARMS, [1, 2], init=False)
    calls = []
    monkeypatch.setattr(experiment, "run", lambda spec, parallel=1: calls.append(("run", spec.name, parallel)))
    monkeypatch.setattr(experiment, "report", lambda spec: calls.append(("report", spec.name)))
    experiment.main(["run", path, "--parallel", "3"])
    experiment.main(["report", path])
    assert calls == [("run", "t", 3), ("report", "t")]
