"""Loss review: replays of lost games (docs/MILESTONE_11 "Loss review").

`python -m polytopia_rl.losses --checkpoint CKPT --opponent rush --count 5 --out DIR`
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossRecord:
    seed: int
    seat: int
    final_turn: int  # info()["turn"] at the end
    loss_type: str  # "capital_captured" (game ended before the turn cap) or "score"
    scores: tuple  # info()["scores"] at the end, (seat 0, seat 1)
    first_city_lost_turn: int | None  # see collect_losses
    path: str  # the replay file


def collect_losses(policy_fn, opponent, seeds, env_kwargs, out_dir, limit) -> list[LossRecord]:
    """Play seeds `SUITE_SEED_BASE + j` for `j in range(seeds)`, seat 0 then seat 1, greedy, against
    `opponent` (epsilon 0); keep the lost games in that order, stopping after `limit`. Each is
    written to `out_dir/{opponent}-{seed}-seat{s}.json` (`Env.replay_json()`).

    `first_city_lost_turn`: `info()["turn"]` at the first observation of the agent (a decision or
    the final observation) where a tile that held one of its cities at its previous observation no
    longer does; None if that never happens."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
