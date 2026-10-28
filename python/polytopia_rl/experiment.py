"""Multi-seed experiments (docs/MILESTONE_12 "Experiment runs", "Report").

`python -m polytopia_rl.experiment run SPEC [--parallel K] [--only ARM]`,
`python -m polytopia_rl.experiment report SPEC`.

Layout (paths relative to the working directory): run `(arm, seed)` lives in
`runs/exp/{name}/{arm}/seed{seed}` with `command.txt`, the trainer's files, `suite_{step}/suite.json`
(and `suite.md`) per evaluated checkpoint and, with a reference or an init checkpoint,
`h2h_{step}.json`: `dataclasses.asdict(play_pair(run, opponent, h2h_seeds, env_kwargs))` with the run's
checkpoint as player `a` and the opponent as `b`: the reference (named `"ref"`) if the spec has one,
else the init checkpoint (named `"init"`). The experiment round robin is
`runs/exp/{name}/tournament.json` and `.md` (`tournament.write_tournament`); its players are the bots
(`random`, `economy`, `rush`), that opponent (if any) and each run's `ckpt_final.pt` as
`{arm}_seed{seed}`. With `init_from`, run `(arm, seed)` warm-starts from
`runs/exp/{init_from}/{arm}/seed{seed}/ckpt_final.pt`.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_TOTAL_STEPS = 10_000_000


@dataclass(kw_only=True)
class Spec:
    name: str
    base_config: str
    seeds: list
    arms: dict  # arm name -> config overrides (JSON values), in spec order
    total_steps: int = DEFAULT_TOTAL_STEPS
    init_checkpoint: str | None = None
    primary: list = field(default_factory=list)  # suite condition names and/or the Elo metric
    suite_seeds: int
    tournament_seeds: int
    eval_checkpoints: int = 3
    h2h_seeds: int = 100
    # A checkpoint every evaluated checkpoint plays head to head instead of the init checkpoint
    # (`h2h_{step}.json`, metric "elo_vs_ref", round robin player "ref"), and the name of an earlier
    # experiment whose `runs/exp/{init_from}/{arm}/seed{seed}/ckpt_final.pt` warm-starts run (arm, seed)
    # (exclusive with `init_checkpoint`).
    reference: str | None = None
    init_from: str | None = None


@dataclass
class ExperimentReport:
    name: str
    arms: list  # spec order
    seeds: list  # spec order
    # One dict per (arm, seed), arms outer and seeds inner: {"arm", "seed", "steps" (evaluated steps,
    # ascending), "suite": {condition: [win rate per step]}, "suite_mean": {condition: mean over the
    # steps}, "elo_vs_init": [Elo per step] or None, "elo_vs_init_mean": float or None, "obs_hash",
    # "rules_hash"}; with a reference "elo_vs_ref" and "elo_vs_ref_mean" replace the "elo_vs_init" keys.
    runs: list
    # metric -> arm -> {"per_seed": [run value per seed, spec order], "mean", "sd" (ddof=1), "min",
    # "max"}; metrics are the suite conditions, then the Elo metric (`elo_metric(spec)`) if any.
    tables: dict
    # One dict per (pair of arms in spec order, metric): {"a", "b", "metric", "primary", "mean_a",
    # "mean_b", "diff", "t", "df", "p", "significant"} from `welch(per_seed[a], per_seed[b])`;
    # "significant" is a bool for primary metrics and None for exploratory ones.
    comparisons: list
    n_flagged: int  # comparisons that carry a significance flag (primary ones)
    elo: dict | None  # player -> Elo from the experiment round robin (None if not played yet)
    obs_hash: str
    # arm -> {"params": parameter count of the arm's network (from its first seed's final checkpoint),
    # "sps": `_summary` over seeds of each run's mean `sps` in `metrics.jsonl`}; None where a checkpoint
    # cannot be loaded or a metrics file is missing.
    cost: dict | None = None


def eval_checkpoints(run_dir, k) -> list[str]:
    """Paths of the last `k` distinct steps among `run_dir/ckpt_{step}.pt` (numeric order, oldest
    first); fewer when the run has fewer."""
    steps = {}
    for path in glob.glob(os.path.join(run_dir, "ckpt_*.pt")):
        m = re.fullmatch(r"ckpt_(\d+)\.pt", os.path.basename(path))
        if m:
            steps[int(m.group(1))] = path
    return [steps[s] for s in sorted(steps)[-k:]] if k > 0 else []


def load_spec(path) -> Spec:
    """Read a spec JSON. Unknown keys, and `seed` or `total_steps` in an arm, raise `ValueError`."""
    with open(path) as f:
        d = json.load(f)
    known = {f.name for f in dataclasses.fields(Spec)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"{path}: unknown spec keys {unknown}")
    missing = sorted(f.name for f in dataclasses.fields(Spec) if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in d)
    if missing:
        raise ValueError(f"{path}: missing spec keys {missing}")
    for arm, overrides in d["arms"].items():
        bad = sorted({"seed", "total_steps"} & set(overrides))
        if bad:
            raise ValueError(f"{path}: arm {arm!r} sets {bad}, which the launcher sets")
    if d.get("init_from") is not None and d.get("init_checkpoint") is not None:
        raise ValueError(f"{path}: init_from and init_checkpoint are mutually exclusive")
    return Spec(**d)


def run_dir(spec, arm, seed) -> str:
    """`runs/exp/{name}/{arm}/seed{seed}`."""
    return os.path.join("runs", "exp", spec.name, arm, f"seed{seed}")


def exp_dir(spec) -> str:
    return os.path.join("runs", "exp", spec.name)


def opponent(spec) -> tuple[str, str] | None:
    """`(name, checkpoint)` of the head-to-head opponent: the reference, else the init checkpoint."""
    if spec.reference is not None:
        return "ref", spec.reference
    if spec.init_checkpoint is not None:
        return "init", spec.init_checkpoint
    return None


def elo_metric(spec) -> str | None:
    """The metric of the head-to-heads: `elo_vs_ref` or `elo_vs_init` (None without an opponent)."""
    opp = opponent(spec)
    return None if opp is None else f"elo_vs_{opp[0]}"


def init_source(spec, arm, seed) -> str:
    """`runs/exp/{init_from}/{arm}/seed{seed}/ckpt_final.pt`."""
    return os.path.join("runs", "exp", spec.init_from, arm, f"seed{seed}", "ckpt_final.pt")


def tournament_players(spec) -> list[str]:
    """The experiment round robin's players: the bots, the opponent (`ref` or `init`, if any), then
    `{arm}_seed{seed}`."""
    from polytopia_rl.tournament import BOTS

    opp = opponent(spec)
    return list(BOTS) + ([opp[0]] if opp is not None else []) + [f"{a}_seed{s}" for a in spec.arms for s in spec.seeds]


def train_command(spec, arm, seed) -> list[str]:
    """The trainer command for run `(arm, seed)`: resumed from `latest.pt` if the run has one."""
    d = run_dir(spec, arm, seed)
    argv = [sys.executable, "-m", "polytopia_rl.train", "--config", spec.base_config, "--run-dir", d]
    latest = os.path.join(d, "latest.pt")
    if os.path.exists(latest):
        argv += ["--resume", latest]
    elif spec.init_checkpoint is not None:
        argv += ["--init-checkpoint", spec.init_checkpoint]
    elif spec.init_from is not None:
        argv += ["--init-checkpoint", init_source(spec, arm, seed)]
    argv += [f"seed={seed}", f"total_steps={spec.total_steps}"]
    return argv + [f"{k}={json.dumps(v)}" for k, v in spec.arms[arm].items()]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def run(spec, parallel=1) -> None:
    """Train every (arm, seed) not yet done (at most `parallel` subprocesses at once), evaluate the
    last `eval_checkpoints` checkpoints of each finished run, then play the experiment round robin.
    Raises `RunsFailed` at the end if any run failed (the others are trained and evaluated)."""
    pairs = [(arm, seed) for arm in spec.arms for seed in spec.seeds]
    pending = [p for p in pairs if not os.path.exists(os.path.join(run_dir(spec, *p), "ckpt_final.pt"))]
    finished = [p for p in pairs if p not in pending]  # done before this call: evaluate what is missing
    commit = _git_commit()
    running = []  # (pair, process, log file)
    failed = []
    evaluator = _Evaluator(spec)
    for p in finished:
        evaluator.evaluate(*p)

    def launch():
        while pending and len(running) < parallel:
            arm, seed = pending.pop(0)
            d = run_dir(spec, arm, seed)
            os.makedirs(d, exist_ok=True)
            argv = train_command(spec, arm, seed)
            with open(os.path.join(d, "command.txt"), "w") as f:
                f.write(shlex.join(argv) + "\n" + f"git {commit}\n")
            if spec.init_from is not None and "--resume" not in argv and not os.path.exists(init_source(spec, arm, seed)):
                failed.append((arm, seed))
                print(f"FAILED {d}: init_from source {init_source(spec, arm, seed)} does not exist", file=sys.stderr, flush=True)
                continue
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "1")
            log = open(os.path.join(d, "train.out"), "a")
            print(f"start {d}: {shlex.join(argv)}", flush=True)
            running.append(((arm, seed), subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    while pending or running:
        launch()
        time.sleep(10)
        status = [(r, r[1].poll()) for r in running]
        running[:] = [r for r, code in status if code is None]
        ok = []
        for ((arm, seed), _, log), code in status:
            if code is None:
                continue
            log.close()
            d = run_dir(spec, arm, seed)
            if code != 0:
                failed.append((arm, seed))
                print(f"FAILED {d}: exit code {code}, log {log.name}", file=sys.stderr, flush=True)
            else:
                print(f"done {d}", flush=True)
                ok.append((arm, seed))
        launch()  # refill the free slots before evaluating, so the other trainers keep going
        for arm, seed in ok:
            evaluator.evaluate(arm, seed)
    if failed:
        raise RunsFailed(f"{len(failed)} run(s) failed: " + ", ".join(run_dir(spec, *p) for p in failed))
    evaluator.round_robin(pairs)


class RunsFailed(RuntimeError):
    """Some trainer subprocesses exited non-zero (the others were trained and evaluated)."""


class _Evaluator:
    """The launcher's evaluations, in its own process: per-checkpoint suites and head-to-heads, and
    the round robin of the final weights."""

    def __init__(self, spec):
        self.spec = spec
        self._opponent = None

    def _device(self):
        from polytopia_rl.train import resolve_device

        return resolve_device("auto")

    def _player(self, name, path):
        import torch

        from polytopia_rl.checkpoint import load_net
        from polytopia_rl.selfplay import net_batch_policy
        from polytopia_rl.tournament import Player

        device = self._device()
        net, ck = load_net(path, device)
        return Player(name, batch_fn=net_batch_policy(net, torch.Generator(), greedy=True, device=device)), net, ck

    def opponent_player(self):
        if self._opponent is None:
            self._opponent = self._player(*opponent(self.spec))[0]
        return self._opponent

    def evaluate(self, arm, seed) -> None:
        """Suite and head-to-head of each of the run's last checkpoints that lack them."""
        import torch

        from polytopia_rl.evaluate import net_policy
        from polytopia_rl.selfplay import net_batch_policy
        from polytopia_rl.suite import DEFAULT_SUITE, run_suite, suite_markdown, suite_to_json
        from polytopia_rl.tournament import play_pair
        from polytopia_rl.vec_env import env_kwargs_of

        spec, d = self.spec, run_dir(self.spec, arm, seed)
        device = self._device()
        for path in eval_checkpoints(d, spec.eval_checkpoints):
            step = int(os.path.basename(path)[5:-3])
            suite_dir = os.path.join(d, f"suite_{step}")
            h2h_path = os.path.join(d, f"h2h_{step}.json")
            need_h2h = opponent(spec) is not None and not os.path.exists(h2h_path)
            if os.path.exists(os.path.join(suite_dir, "suite.json")) and not need_h2h:
                continue
            player, net, ck = self._player(f"{arm}_seed{seed}", path)
            env_kwargs = env_kwargs_of(ck.cfg)
            if not os.path.exists(os.path.join(suite_dir, "suite.json")):
                print(f"suite {path} ({spec.suite_seeds} seeds)", flush=True)
                res = run_suite(
                    net_policy(net, device=device),
                    DEFAULT_SUITE,
                    spec.suite_seeds,
                    env_kwargs,
                    step=ck.global_step,
                    make_batch_fn=lambda greedy: net_batch_policy(net, torch.Generator().manual_seed(0), greedy=greedy, device=device),
                )
                os.makedirs(suite_dir, exist_ok=True)
                with open(os.path.join(suite_dir, "suite.md"), "w") as f:
                    f.write(suite_markdown(res))
                with open(os.path.join(suite_dir, "suite.json"), "w") as f:  # last: marks the suite done
                    f.write(suite_to_json(res))
            if need_h2h:
                print(f"head-to-head {path} against {opponent(spec)[0]} ({spec.h2h_seeds} seeds)", flush=True)
                pair = play_pair(player, self.opponent_player(), spec.h2h_seeds, env_kwargs)
                with open(h2h_path, "w") as f:
                    json.dump(dataclasses.asdict(pair), f)

    def round_robin(self, pairs) -> None:
        """The bots, the opponent and every final policy, unless `tournament.json` already has them."""
        from polytopia_rl.tournament import BOTS, Player, round_robin, write_tournament
        from polytopia_rl.vec_env import env_kwargs_of

        spec = self.spec
        names = tournament_players(spec)
        out = os.path.join(exp_dir(spec), "tournament.json")
        if os.path.exists(out):
            with open(out) as f:
                if json.load(f)["players"] == names:
                    return
        players = [Player(b, bot=b) for b in BOTS]
        if opponent(spec) is not None:
            players.append(self.opponent_player())
        cfg = None
        for arm, seed in pairs:
            player, _, ck = self._player(f"{arm}_seed{seed}", os.path.join(run_dir(spec, arm, seed), "ckpt_final.pt"))
            players.append(player)
            cfg = ck.cfg
        print(f"round robin of {len(players)} players ({spec.tournament_seeds} seeds per pair)", flush=True)
        res = round_robin(players, spec.tournament_seeds, env_kwargs_of(cfg))
        write_tournament(res, exp_dir(spec))


# ---------------------------------------------------------------- report


def _elo(pair: dict) -> float:
    """`400 log10(s / (1 - s))` of the head-to-head score of `a`, clipped to `[0.5 / n, 1 - 0.5 / n]`."""
    n = pair["games"]
    s = (pair["a_wins"] + 0.5 * pair["draws"]) / n
    s = min(max(s, 0.5 / n), 1 - 0.5 / n)
    return 400 * math.log10(s / (1 - s))


def _summary(values: list) -> dict:
    return {
        "per_seed": values,
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _read_run(spec, arm, seed) -> dict:
    from polytopia_rl.suite import suite_from_json

    d = run_dir(spec, arm, seed)
    steps = sorted(int(m.group(1)) for p in glob.glob(os.path.join(d, "suite_*", "suite.json")) if (m := re.fullmatch(r"suite_(\d+)", os.path.basename(os.path.dirname(p)))))
    if not steps:
        raise ValueError(f"{d} has no evaluated checkpoint (suite_{{step}}/suite.json); run it first")
    suites = []
    for step in steps:
        with open(os.path.join(d, f"suite_{step}", "suite.json")) as f:
            suites.append(suite_from_json(f.read()))
    obs = {r.obs_hash for r in suites}
    if len(obs) > 1:
        raise ValueError(f"{d}: suites with different obs_hash values {sorted(obs)}")
    conditions = list(suites[0].conditions)
    suite = {c: [float(r.conditions[c].win_rate) for r in suites] for c in conditions}
    elo, metric = None, elo_metric(spec) or "elo_vs_init"
    if opponent(spec) is not None:
        elo = []
        for step in steps:
            with open(os.path.join(d, f"h2h_{step}.json")) as f:
                elo.append(_elo(json.load(f)))
    return {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "suite": suite,
        "suite_mean": {c: float(np.mean(v)) for c, v in suite.items()},
        metric: elo,
        f"{metric}_mean": float(np.mean(elo)) if elo is not None else None,
        "obs_hash": suites[0].obs_hash,
        "rules_hash": suites[0].rules_hash,
    }


def report(spec) -> ExperimentReport:
    """Aggregate the experiment; also writes `runs/exp/{name}/report.md`, `report.json`
    (`dataclasses.asdict` of the result) and `report_curves.png`."""
    from polytopia_rl.stats import welch

    arms, seeds = list(spec.arms), list(spec.seeds)
    runs = [_read_run(spec, arm, seed) for arm in arms for seed in seeds]
    obs = sorted({r["obs_hash"] for r in runs})
    if len(obs) > 1:
        raise ValueError(f"experiment {spec.name}: runs with different obs_hash values {obs}")
    elo_m = elo_metric(spec)
    metrics = list(runs[0]["suite_mean"]) + ([elo_m] if elo_m is not None else [])

    def value(r, metric):
        return r[f"{metric}_mean"] if metric == elo_m else r["suite_mean"][metric]

    by_run = {(r["arm"], r["seed"]): r for r in runs}
    tables = {m: {arm: _summary([value(by_run[arm, s], m) for s in seeds]) for arm in arms} for m in metrics}
    comparisons = []
    if len(seeds) >= 2:
        for a, b in itertools.combinations(arms, 2):
            for m in metrics:
                w = welch(tables[m][a]["per_seed"], tables[m][b]["per_seed"])
                primary = m in spec.primary
                comparisons.append(
                    {
                        "a": a,
                        "b": b,
                        "metric": m,
                        "primary": primary,
                        "mean_a": w.mean_a,
                        "mean_b": w.mean_b,
                        "diff": w.diff,
                        "t": w.t,
                        "df": w.df,
                        "p": w.p,
                        "significant": w.significant if primary else None,
                    }
                )
    elo = None
    tpath = os.path.join(exp_dir(spec), "tournament.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            t = json.load(f)
        if t["players"] == tournament_players(spec):  # not a round robin of only some arms (--only)
            elo = {name: float(r) for name, r in zip(t["players"], t["ratings"])}
    rep = ExperimentReport(
        name=spec.name,
        arms=arms,
        seeds=seeds,
        runs=runs,
        tables=tables,
        comparisons=comparisons,
        n_flagged=sum(c["primary"] for c in comparisons),
        elo=elo,
        obs_hash=obs[0],
        cost={arm: _cost(spec, arm) for arm in arms},
    )
    out = exp_dir(spec)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(dataclasses.asdict(rep), f, indent=1)
    _plot_curves(spec, os.path.join(out, "report_curves.png"))
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(report_markdown(rep))
    return rep


def _cost(spec, arm) -> dict:
    """Parameter count of the arm's network (its first seed's `ckpt_final.pt`) and `_summary` over
    seeds of each run's mean `sps` over its `metrics.jsonl` rows; None where a file is unreadable."""
    from polytopia_rl.checkpoint import load_net

    try:
        net, _ = load_net(os.path.join(run_dir(spec, arm, spec.seeds[0]), "ckpt_final.pt"), "cpu")
        params = sum(p.numel() for p in net.parameters())
    except Exception:  # placeholder or missing checkpoint
        params = None
    per_seed = []
    for seed in spec.seeds:
        p = os.path.join(run_dir(spec, arm, seed), "metrics.jsonl")
        if not os.path.exists(p):
            per_seed = None
            break
        with open(p) as f:
            sps = [r["sps"] for line in f if line.strip() and "sps" in (r := json.loads(line))]
        if not sps:
            per_seed = None
            break
        per_seed.append(float(np.mean(sps)))
    return {"params": params, "sps": _summary(per_seed) if per_seed is not None else None}


def _fmt(x, digits=3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def report_markdown(rep: ExperimentReport) -> str:
    n_seeds = len(rep.seeds)
    elo_m = "elo_vs_ref" if "elo_vs_ref" in rep.tables else "elo_vs_init"
    lines = [
        f"# Experiment {rep.name}",
        "",
        f"Arms {', '.join(rep.arms)}; seeds {', '.join(map(str, rep.seeds))}; obs_hash `{rep.obs_hash}`.",
        f"A run's value is the mean over its evaluated checkpoints (suite win rate, or `{elo_m}`).",
        f"With {n_seeds} seeds per arm Welch's test detects only large effects; a primary comparison without a "
        "flag is \"not detected\", not \"no effect\". Non-primary comparisons are exploratory (no flag).",
        "",
    ]
    for metric, table in rep.tables.items():
        digits = 1 if metric == elo_m else 3
        lines += [f"## {metric}", "", "| arm | " + " | ".join(f"seed {s}" for s in rep.seeds) + " | mean | sd | min to max |", "| --- |" + " --- |" * (n_seeds + 3)]
        for arm, row in table.items():
            cells = [_fmt(v, digits) for v in row["per_seed"]]
            lines.append(f"| {arm} | " + " | ".join(cells) + f" | {_fmt(row['mean'], digits)} | {_fmt(row['sd'], digits)} | {_fmt(row['min'], digits)} to {_fmt(row['max'], digits)} |")
        lines += ["", "Per checkpoint (step: value):", "", "| arm | seed | checkpoints |", "| --- | --- | --- |"]
        for r in rep.runs:
            vals = r[elo_m] if metric == elo_m else r["suite"][metric]
            lines.append(f"| {r['arm']} | {r['seed']} | " + ", ".join(f"{s}: {_fmt(v, digits)}" for s, v in zip(r["steps"], vals)) + " |")
        lines.append("")
    lines += [
        "## Comparisons (Welch, b minus a)",
        "",
        f"Flagged comparisons (primary metrics {', '.join(m for m in rep.tables if any(c['metric'] == m and c['primary'] for c in rep.comparisons)) or 'none'}): {rep.n_flagged}.",
        "",
        "| a | b | metric | mean a | mean b | diff | t | df | p | flag |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in rep.comparisons:
        flag = "exploratory" if c["significant"] is None else ("significant" if c["significant"] else "not detected")
        lines.append(f"| {c['a']} | {c['b']} | {c['metric']} | {c['mean_a']:.3f} | {c['mean_b']:.3f} | {c['diff']:+.3f} | {c['t']:+.2f} | {c['df']:.1f} | {c['p']:.3f} | {flag} |")
    lines += ["", "## Round robin of the final weights (exploratory)", ""]
    if rep.elo is None:
        lines.append("Not played yet.")
    else:
        lines += ["| player | Elo |", "| --- | --- |"] + [f"| {name} | {r:.0f} |" for name, r in sorted(rep.elo.items(), key=lambda kv: -kv[1])]
    if rep.cost is not None:
        lines += [
            "",
            "## Cost (reported, not tested)",
            "",
            "Parameters of the arm's network; training sps: each run's mean over its updates.",
            "",
            "| arm | parameters | " + " | ".join(f"sps seed {s}" for s in rep.seeds) + " | sps mean | sps sd |",
            "| --- | --- |" + " --- |" * (n_seeds + 2),
        ]
        for arm, c in rep.cost.items():
            sps = c["sps"]
            cells = [_fmt(v, 0) for v in sps["per_seed"]] + [_fmt(sps["mean"], 0), _fmt(sps["sd"], 0)] if sps is not None else ["-"] * (n_seeds + 2)
            lines.append(f"| {arm} | {'-' if c['params'] is None else c['params']} | " + " | ".join(cells) + " |")
    lines += ["", "## Runs", "", "| arm | seed | evaluated steps | obs_hash | rules_hash |", "| --- | --- | --- | --- | --- |"]
    for r in rep.runs:
        lines.append(f"| {r['arm']} | {r['seed']} | {', '.join(map(str, r['steps']))} | `{r['obs_hash']}` | `{r['rules_hash']}` |")
    lines += ["", "Learning curves: `report_curves.png`."]
    return "\n".join(lines) + "\n"


def _plot_curves(spec, path) -> None:
    """Per in-training suite-subset condition: each arm's mean win rate over seeds against the step,
    with a min to max band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {}  # arm -> seed -> condition -> {step: win rate}
    for arm in spec.arms:
        for seed in spec.seeds:
            rows = {}
            p = os.path.join(run_dir(spec, arm, seed), "eval.jsonl")
            if os.path.exists(p):
                with open(p) as f:
                    for line in f:
                        r = json.loads(line) if line.strip() else {}
                        if r.get("type") == "suite":
                            for c, v in r["suite"].items():
                                rows.setdefault(c, {})[r["global_step"]] = v["win_rate"]
            curves.setdefault(arm, {})[seed] = rows
    conditions = sorted({c for per_seed in curves.values() for rows in per_seed.values() for c in rows})
    fig, axes = plt.subplots(1, max(1, len(conditions)), figsize=(6 * max(1, len(conditions)), 4.5), squeeze=False)
    for ax, c in zip(axes[0], conditions):
        for arm, per_seed in curves.items():
            series = [rows.get(c, {}) for rows in per_seed.values()]
            steps = sorted(set.intersection(*(set(s) for s in series))) if series else []
            if not steps:
                continue
            ys = np.array([[s[x] for x in steps] for s in series])
            ax.plot(steps, ys.mean(axis=0), marker="o", label=arm)
            ax.fill_between(steps, ys.min(axis=0), ys.max(axis=0), alpha=0.2)
        ax.set_title(f"{c} win rate (mean over seeds, min to max)")
        ax.set_xlabel("global step")
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Multi-seed experiments: run or report a spec.")
    p.add_argument("command", choices=["run", "report"])
    p.add_argument("spec")
    p.add_argument("--parallel", type=int, default=1, help="trainers at once (run)")
    p.add_argument("--only", help="run only this arm (run)")
    a = p.parse_args(argv)
    spec = load_spec(a.spec)
    if a.command == "report":
        report(spec)
        out = exp_dir(spec)
        print(f"wrote {os.path.join(out, 'report.md')}, report.json and report_curves.png", flush=True)
        return
    if a.only is not None:
        if a.only not in spec.arms:
            p.error(f"--only {a.only}: not an arm of {a.spec} ({', '.join(spec.arms)})")
        spec = dataclasses.replace(spec, arms={a.only: spec.arms[a.only]})
    try:
        run(spec, parallel=a.parallel)
    except RunsFailed as e:
        print(f"experiment {spec.name}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
