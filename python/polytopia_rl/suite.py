"""The fixed evaluation suite (docs/MILESTONE_11 "Evaluation suite").

`python -m polytopia_rl.suite --checkpoint CKPT [--seeds 200] [--out DIR]` (writes `DIR/suite.json`
and `DIR/suite.md`), `python -m polytopia_rl.suite --compare A.json B.json`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass

import polytopia_sim

from polytopia_rl.evaluate import evaluate

SUITE_SEED_BASE = 1_000_000_000  # suite seeds: [SUITE_SEED_BASE, SUITE_SEED_LIMIT)
SUITE_SEED_LIMIT = 1_500_000_000
DEFAULT_TURN_CAP = 30  # polytopia_sim.Env's default, for env_kwargs without "turn_cap"
CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "suite.json")


@dataclass
class Condition:
    name: str
    opponent: str  # "random" | "economy" | "rush"
    epsilon: float = 0.0
    full_observability: bool = False
    greedy: bool = True


DEFAULT_SUITE: list[Condition] = [  # the brief's table, in table order
    Condition("random_e0", "random"),
    Condition("economy_e0", "economy"),
    Condition("rush_e0", "rush"),
    Condition("economy_e20", "economy", epsilon=0.2),
    Condition("rush_e20", "rush", epsilon=0.2),
    Condition("economy_fullobs", "economy", full_observability=True),
    Condition("rush_fullobs", "rush", full_observability=True),
    Condition("rush_e0_sampled", "rush", greedy=False),
]


@dataclass
class ConditionResult:
    name: str
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    wilson_lo: float
    wilson_hi: float
    by_seat: dict  # {0: win rate as seat 0, 1: win rate as seat 1}
    mean_length: float
    mean_score_diff: float
    loss_capital_share: float  # share of the losses that were capital captures (0 without losses)
    loss_score_share: float  # share of the losses decided by score at the cap (0 without losses)
    per_game: list  # evaluate()'s per-game dicts plus "loss_type" (None unless lost)


@dataclass
class SuiteResult:
    conditions: dict  # name -> ConditionResult, in the order run
    obs_hash: str  # of the build that ran the suite (Env.spec())
    rules_hash: str
    step: int | None  # training step of the evaluated weights, if known


@dataclass
class Comparison:
    name: str
    win_rate_a: float
    win_rate_b: float
    diff: float  # win_rate_b - win_rate_a
    z: float  # two-proportion z statistic with the pooled proportion (0 when the pooled sd is 0)
    significant: bool  # |z| > 1.96


def run_suite(policy_fn, conditions, seeds, env_kwargs, *, step=None, make_batch_fn=None) -> SuiteResult:
    """Each condition plays seeds `SUITE_SEED_BASE + j` for `j in range(seeds)`, each seed as seat 0
    then seat 1, against the condition's opponent with its epsilon and full observability, calling
    `policy_fn(env, condition.greedy)`. `env_kwargs` give the map and turn cap. A seed range that
    reaches `SUITE_SEED_LIMIT` raises `ValueError`.

    `make_batch_fn(greedy) -> batch_fn` (docs/MILESTONE_12): when given, each condition plays
    batched through `evaluate(..., batch_fn=make_batch_fn(condition.greedy))`."""
    if make_batch_fn is not None:
        raise NotImplementedError("milestone 12: batched suite")
    if SUITE_SEED_BASE + seeds > SUITE_SEED_LIMIT:
        raise ValueError(f"{seeds} suite seeds reach {SUITE_SEED_LIMIT}")
    spec = polytopia_sim.Env(size=env_kwargs.get("size", 11)).spec()
    turn_cap = env_kwargs.get("turn_cap", DEFAULT_TURN_CAP)
    out = {}
    for c in conditions:
        kw = {**env_kwargs, "full_observability": c.full_observability}
        res = evaluate(policy_fn, c.opponent, seeds, kw, c.greedy, seed_base=SUITE_SEED_BASE, epsilon=c.epsilon)
        per_game = [{**g, "loss_type": loss_type(g, turn_cap)} for g in res.per_game]
        cap = sum(g["loss_type"] == "capital_captured" for g in per_game)
        score = sum(g["loss_type"] == "score" for g in per_game)
        fields = {k: v for k, v in dataclasses.asdict(res).items() if k != "per_game"}
        out[c.name] = ConditionResult(
            name=c.name,
            **fields,
            loss_capital_share=cap / res.losses if res.losses else 0.0,
            loss_score_share=score / res.losses if res.losses else 0.0,
            per_game=per_game,
        )
    return SuiteResult(conditions=out, obs_hash=spec["obs_hash"], rules_hash=spec["rules_hash"], step=step)


def loss_type(game: dict, turn_cap: int) -> str | None:
    """A lost game that ended before the turn cap was a capital capture; one that reached it was
    decided by score. None unless lost."""
    if game["outcome"] != "loss":
        return None
    return "score" if game["turn"] >= turn_cap else "capital_captured"


def compare(a: SuiteResult, b: SuiteResult) -> list[Comparison]:
    """One `Comparison` per condition present in both, in `a`'s order."""
    out = []
    for name, ra in a.conditions.items():
        rb = b.conditions.get(name)
        if rb is None:
            continue
        p = (ra.wins + rb.wins) / (ra.games + rb.games)
        se = math.sqrt(p * (1 - p) * (1 / ra.games + 1 / rb.games))
        diff = rb.win_rate - ra.win_rate
        z = diff / se if se > 0 else 0.0
        out.append(Comparison(name, ra.win_rate, rb.win_rate, diff, z, abs(z) > 1.96))
    return out


def suite_markdown(result) -> str:
    lines = [
        f"Suite: obs_hash `{result.obs_hash}`, rules_hash `{result.rules_hash}`, step {result.step}",
        "",
        "| condition | games | W-L-D | win rate | 95% CI | seat 0 | seat 1 | mean len | score diff | losses capital / score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result.conditions.values():
        lines.append(
            f"| {r.name} | {r.games} | {r.wins}-{r.losses}-{r.draws} | {r.win_rate:.3f} | "
            f"[{r.wilson_lo:.3f}, {r.wilson_hi:.3f}] | {r.by_seat[0]:.3f} | {r.by_seat[1]:.3f} | "
            f"{r.mean_length:.1f} | {r.mean_score_diff:+.1f} | {r.loss_capital_share:.2f} / {r.loss_score_share:.2f} |"
        )
    return "\n".join(lines) + "\n"


def suite_to_json(result: SuiteResult) -> str:
    return json.dumps(dataclasses.asdict(result))


def suite_from_json(text: str) -> SuiteResult:
    d = json.loads(text)
    conditions = {}
    for name, c in d["conditions"].items():
        c["by_seat"] = {int(k): v for k, v in c["by_seat"].items()}
        conditions[name] = ConditionResult(**c)
    return SuiteResult(conditions=conditions, obs_hash=d["obs_hash"], rules_hash=d["rules_hash"], step=d["step"])


def main(argv: list[str] | None = None) -> None:
    from polytopia_rl.checkpoint import describe, load_net
    from polytopia_rl.evaluate import net_policy
    from polytopia_rl.train import resolve_device
    from polytopia_rl.vec_env import env_kwargs_of

    with open(CONFIG) as f:
        default_seeds = json.load(f)["seeds"]
    p = argparse.ArgumentParser(description="The fixed evaluation suite.")
    p.add_argument("--checkpoint")
    p.add_argument("--seeds", type=int, default=default_seeds)
    p.add_argument("--out", help="directory for suite.json and suite.md")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two suite.json files")
    p.add_argument("--allow-obs-mismatch", action="store_true")
    a = p.parse_args(argv)
    if a.compare:
        with open(a.compare[0]) as fa, open(a.compare[1]) as fb:
            ra, rb = suite_from_json(fa.read()), suite_from_json(fb.read())
        print(f"A {a.compare[0]} (step {ra.step}, obs_hash {ra.obs_hash})  B {a.compare[1]} (step {rb.step}, obs_hash {rb.obs_hash})")
        for c in compare(ra, rb):
            flag = "  SIGNIFICANT" if c.significant else ""
            print(f"{c.name:16s} A {c.win_rate_a:.3f}  B {c.win_rate_b:.3f}  diff {c.diff:+.3f}  z {c.z:+.2f}{flag}")
        return
    if not a.checkpoint:
        p.error("--checkpoint or --compare is required")
    device = resolve_device("auto")
    net, ck = load_net(a.checkpoint, device, strict_obs=not a.allow_obs_mismatch)
    print(describe(a.checkpoint, ck), flush=True)
    res = run_suite(net_policy(net, device=device), DEFAULT_SUITE, a.seeds, env_kwargs_of(ck.cfg), step=ck.global_step)
    md = suite_markdown(res)
    print(md, end="")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "suite.json"), "w") as f:
            f.write(suite_to_json(res))
        with open(os.path.join(a.out, "suite.md"), "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
