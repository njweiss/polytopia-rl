"""Snapshot pool, matchmaking and the self-play vector env (docs/MILESTONE_11 "Self-play").

An env is anything with the `polytopia_sim.Env` methods `reset(seed)`, `observation()`,
`actor_mask()`, `kind_mask(actor)`, `arg_mask(actor, kind)`, `step_factored(actor, kind, arg)`,
`info()` (keys `seed`, `turn`, `current_player`, `agent_seat`, `scores`, `outcome`,
`agent_actions`) and `spec()`.

A batch policy is a callable `policy(envs) -> list[(actor, kind, arg)]` that chooses one action for
each env (all on the move of the seat it plays) in a single batched forward pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from polytopia_rl.vec_env import EpisodeStat, SyncVecEnv, VecState, VecStep, agent_outcome, env_kwargs_of

BOT_OPPONENTS = ("economy", "rush")


@dataclass
class Transition:
    """A learner transition completed during `SelfPlayVecEnv.tick` (docs/MILESTONE_13 "Ticks")."""

    slot: int
    reward: float  # the M9 formula over the learner's action and everything up to its next move or the episode end
    terminated: bool
    truncated: bool
    final_board: np.ndarray | None  # as VecStep: the final observation when the episode ended
    final_globals: np.ndarray | None


@dataclass
class Entry:
    step: int
    state: dict  # a frozen copy of the learner's state dict at `step` (CPU tensors)
    name: str  # f"snap_{step}"


class SnapshotPool:
    """Snapshots in the order added. `add` appends, then thins while there are more than
    `pool_max` entries."""

    def __init__(self, pool_max: int = 40, p_latest: float = 0.5):
        if pool_max < 2:
            raise ValueError("pool_max must be at least 2 (thinning keeps the first and the newest)")
        self.pool_max = pool_max
        self.p_latest = p_latest
        self.entries: list[Entry] = []

    def add(self, model_state: dict, step: int) -> Entry:
        """Append a deep copy of `model_state` (later changes to the learner do not reach it)."""
        entry = Entry(step, {k: v.detach().cpu().clone() for k, v in model_state.items()}, f"snap_{step}")
        self.entries.append(entry)
        while len(self.entries) > self.pool_max:
            self.thin()
        return entry

    def thin(self) -> None:
        """Keep the entries at even positions (0, 2, 4, ...) and the newest; drop the rest."""
        keep = self.entries[::2]
        if keep[-1] is not self.entries[-1]:
            keep.append(self.entries[-1])
        self.entries = keep

    def sample(self, rng: np.random.Generator) -> Entry:
        """The newest entry with probability `p_latest`, else an entry drawn uniformly from all of
        them (the newest included)."""
        if rng.random() < self.p_latest:
            return self.entries[-1]
        return self.entries[rng.integers(len(self.entries))]


def draw_opponent(pool: SnapshotPool, rng: np.random.Generator, p_bot: float):
    """With probability `p_bot` a bot name drawn uniformly from `BOT_OPPONENTS`, else
    `pool.sample(rng)` (an `Entry`)."""
    if rng.random() < p_bot:
        return BOT_OPPONENTS[rng.integers(len(BOT_OPPONENTS))]
    return pool.sample(rng)


def net_batch_policy(net, generator, greedy: bool = False, device: str = "cpu") -> Callable:
    """A batch policy for a `PolicyNet`: one `trunk` + `act` call for all the envs, masks from
    each env's own observation and masks; sampled with `generator`, or argmax when `greedy`."""

    def policy(envs):
        obs = [env.observation() for env in envs]
        board = torch.from_numpy(np.stack([b for b, _ in obs])).to(device)
        glob = torch.from_numpy(np.stack([g for _, g in obs])).to(device)
        n = board.shape[2] * board.shape[3]

        def masks_fn(stage, actors=None, kinds=None):
            if stage == "actor":
                return np.stack([env.actor_mask() for env in envs])
            if stage == "kind":
                return np.stack([env.kind_mask(int(a)) for env, a in zip(envs, actors)])
            m = np.zeros((len(envs), n), dtype=bool)
            for i, env in enumerate(envs):
                xm = env.arg_mask(int(actors[i]), int(kinds[i]))
                m[i, : xm.size] = xm
            return m

        with torch.inference_mode():
            out = net.act(net.trunk(board, glob), masks_fn, generator, greedy=greedy, stats=False)
        return list(zip(out.actors.tolist(), out.kinds.tolist(), out.args.tolist()))

    return policy


def regression(win_rates: dict, floor: float) -> bool:
    """True when any win rate is below `floor`."""
    return any(r < floor for r in win_rates.values())


class SelfPlayVecEnv(SyncVecEnv):
    """The `SyncVecEnv` interface (`num_envs`, `reset_all`, `step`, `observations`, `actor_masks`,
    `kind_masks`, `arg_masks`, `episode_stats`, `state_dict`, `load_state_dict`) for a learner
    playing against bots and pool snapshots.

    `env_fn(opponent)` builds an env: `opponent` is a bot name (bot mode, agent seat from seed
    parity) or `"none"` (two-seat, for snapshot opponents). Seeds follow `SyncVecEnv`; the learner's
    seat is `seed % 2`. At each episode start the slot's opponent is `draw_opponent(pool, rng,
    p_bot)`. Snapshot slots play `make_policy(entry)(envs)` for their opponent moves, grouped by
    snapshot, one call per group per round, until every slot is terminal or on the learner's move.
    The learner's reward in a snapshot slot is the M9 formula over the learner's action and the whole
    opponent turn, with `reward` = {terminal_weight, shaping_weight, gamma, potential_scale}.
    `EpisodeStat.opponent` is the bot name or the entry's name. `envs[i]` is the env slot `i` is
    currently playing in.
    """

    def __init__(
        self,
        env_fn: Callable,
        num_envs: int,
        seed_base: int,
        seed_stride: int = 1,
        *,
        pool: SnapshotPool,
        make_policy: Callable,
        rng: np.random.Generator,
        p_bot: float = 0.2,
        reward: dict,
    ):
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.seed_base = seed_base
        self.seed_stride = seed_stride
        self.pool = pool
        self.make_policy = make_policy
        self.rng = rng
        self.p_bot = p_bot
        self.reward = {k: np.float32(reward[k]) for k in ("terminal_weight", "shaping_weight", "gamma", "potential_scale")}
        self._env_cache = [{} for _ in range(num_envs)]  # per slot: "none" or bot name -> env
        self._policies = {}  # entry name -> batch policy
        self.envs = [self._env(i, "none") for i in range(num_envs)]
        spec = self.envs[0].spec()
        self.num_tiles = spec["height"] * spec["width"]
        self.board_shape = (spec["channels"], spec["height"], spec["width"])
        self.num_globals = spec["globals"]
        self._next_k = [0] * num_envs
        self._seed = [None] * num_envs
        self._opp = [None] * num_envs  # bot name or Entry
        self._actions = [[] for _ in range(num_envs)]  # the learner's actions of the episode
        self._log = [[] for _ in range(num_envs)]  # snapshot slots: every action of the episode, both seats
        self._ended = [None] * num_envs  # (terminated, truncated) once the slot's episode is over
        self._ret = [0.0] * num_envs
        self._stats: list[EpisodeStat] = []
        for i in range(num_envs):
            self._check_seed(self._episode_seed(i, 0))

    def _env(self, i: int, kind: str):
        if kind not in self._env_cache[i]:
            self._env_cache[i][kind] = self.env_fn(kind)
        return self._env_cache[i][kind]

    def _policy(self, entry: Entry):
        if entry.name not in self._policies:
            self._policies[entry.name] = self.make_policy(entry)
        return self._policies[entry.name]

    def _start(self, i: int, seed: int, opp) -> None:
        """Begin slot `i`'s episode on `seed` against `opp` (the opponent's opening moves, if any,
        are left to `_advance`)."""
        env = self._env(i, opp if isinstance(opp, str) else "none")
        env.reset(seed)
        self.envs[i] = env
        self._opp[i] = opp
        self._seed[i] = seed
        self._actions[i] = []
        self._log[i] = []
        self._ended[i] = None
        self._ret[i] = 0.0

    def _reset(self, i: int) -> None:
        seed = self._episode_seed(i, self._next_k[i])
        self._check_seed(seed)
        self._start(i, seed, draw_opponent(self.pool, self.rng, self.p_bot))
        self._next_k[i] += 1
        # Forget the policies of snapshots that left the pool and that no slot plays any more.
        live = {e.name for e in self.pool.entries} | {o.name for o in self._opp if isinstance(o, Entry)}
        self._policies = {k: v for k, v in self._policies.items() if k in live}

    def _opponent_to_move(self, i: int) -> bool:
        return isinstance(self._opp[i], Entry) and self._ended[i] is None and self.envs[i].info()["current_player"] != self._seed[i] % 2

    def _advance(self, slots) -> None:
        """Play the snapshot opponents of `slots` in rounds until each slot is over or on the
        learner's move. Each round: one batch-policy call per snapshot, one action per slot."""
        pending = [i for i in slots if self._opponent_to_move(i)]
        while pending:
            groups: dict[str, list[int]] = {}
            for i in pending:
                groups.setdefault(self._opp[i].name, []).append(i)
            pending = []
            for idx in groups.values():
                actions = self._policy(self._opp[idx[0]])([self.envs[i] for i in idx])
                for i, a in zip(idx, actions):
                    a = tuple(int(x) for x in a)
                    _, _, terminated, truncated = self.envs[i].step_factored(*a)
                    self._log[i].append(a)
                    if terminated or truncated:
                        self._ended[i] = (terminated, truncated)
                    elif self._opponent_to_move(i):
                        pending.append(i)
            pending.sort()

    def _potential(self, scores, seat: int) -> np.float32:
        return (np.float32(scores[seat]) - np.float32(scores[1 - seat])) / self.reward["potential_scale"]

    def _snapshot_reward(self, i: int, before) -> np.float32:
        """The M9 reward of the learner from the scores `before` its action to the state after the
        opponent's turn (potential 0 when terminated)."""
        r = self.reward
        seat = self._seed[i] % 2
        info = self.envs[i].info()
        terminated = self._ended[i] is not None and self._ended[i][0]
        t = np.float32({f"win_{seat}": 1.0, f"win_{1 - seat}": -1.0}.get(info["outcome"], 0.0))
        after = np.float32(0.0) if terminated else self._potential(info["scores"], seat)
        return r["terminal_weight"] * t + r["shaping_weight"] * (r["gamma"] * after - self._potential(before, seat))

    def reset_all(self) -> VecState:
        for i in range(self.num_envs):
            self._reset(i)
        self._advance(range(self.num_envs))
        return VecState(*self.observations())

    def step(self, actors, kinds, args) -> VecStep:
        e = self.num_envs
        out = VecStep(
            reward=np.zeros(e, dtype=np.float32),
            terminated=np.zeros(e, dtype=bool),
            truncated=np.zeros(e, dtype=bool),
            final_board=np.zeros((e, *self.board_shape), dtype=np.float32),
            final_globals=np.zeros((e, self.num_globals), dtype=np.float32),
        )
        before = {}  # snapshot slots: scores before the learner's action
        for i, env in enumerate(self.envs):
            action = (int(actors[i]), int(kinds[i]), int(args[i]))
            snapshot = isinstance(self._opp[i], Entry)
            if snapshot:
                before[i] = env.info()["scores"]
            reward, _, terminated, truncated = env.step_factored(*action)
            self._actions[i].append(action)
            if snapshot:
                self._log[i].append(action)
            else:
                out.reward[i] = reward
            if terminated or truncated:
                self._ended[i] = (terminated, truncated)
        self._advance(list(before))
        restarted = []
        for i, env in enumerate(self.envs):
            if i in before:
                out.reward[i] = self._snapshot_reward(i, before[i])
            self._ret[i] += float(out.reward[i])
            if self._ended[i] is not None:
                out.terminated[i], out.truncated[i] = self._ended[i]
                out.final_board[i], out.final_globals[i] = env.observation()
                self._stats.append(self._episode_stat(i))
                self._reset(i)
                restarted.append(i)
        self._advance(restarted)
        return out

    def learner_slots(self) -> list[int]:
        """Slots whose learner is to move, ascending (docs/MILESTONE_13 "Ticks")."""
        raise NotImplementedError("milestone 13")

    def tick(self, learner_actions: dict[int, tuple[int, int, int]]) -> list[Transition]:
        """Apply one learner action to every slot in `learner_actions` (all must be in
        `learner_slots()`), then one opponent move to every other slot whose snapshot opponent is to
        move (one batch-policy call per snapshot), handle episode ends and auto-resets, and return the
        transitions completed during this tick, in ascending slot order (docs/MILESTONE_13 "Ticks")."""
        raise NotImplementedError("milestone 13")

    def _episode_stat(self, i: int) -> EpisodeStat:
        seat = self._seed[i] % 2
        opp = self._opp[i]
        return EpisodeStat(
            env_index=i,
            seed=self._seed[i],
            opponent=opp if isinstance(opp, str) else opp.name,
            agent_seat=seat,
            outcome=agent_outcome(self.envs[i].info()["outcome"], seat),
            length=len(self._actions[i]),
            ret=self._ret[i],
        )

    def state_dict(self) -> dict:
        """Per slot: episode index, seed, opponent, the episode's actions so far (replayed on load)
        and return; plus the weights of slot opponents that are no longer in the pool."""
        in_pool = {e.name for e in self.pool.entries}
        gone = {o.name: {"step": o.step, "state": o.state} for o in self._opp if isinstance(o, Entry) and o.name not in in_pool}
        return {
            "seed_base": self.seed_base,
            "seed_stride": self.seed_stride,
            "next_k": list(self._next_k),
            "seeds": list(self._seed),
            "opponents": [o if isinstance(o, str) else o.name for o in self._opp],
            "actions": [[list(a) for a in acts] for acts in self._actions],
            "log": [[list(a) for a in acts] for acts in self._log],
            "rets": list(self._ret),
            "entries": gone,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore after the pool has been restored."""
        self.seed_base = state["seed_base"]
        self.seed_stride = state["seed_stride"]
        entries = {e.name: e for e in self.pool.entries}
        for name, e in state["entries"].items():
            entries[name] = Entry(e["step"], e["state"], name)
        for i in range(self.num_envs):
            name = state["opponents"][i]
            self._start(i, state["seeds"][i], name if name in BOT_OPPONENTS else entries[name])
            self._next_k[i] = state["next_k"][i]
            replay = state["log"][i] if name not in BOT_OPPONENTS else state["actions"][i]
            for action in replay:
                _, _, terminated, truncated = self.envs[i].step_factored(*action)
                assert not (terminated or truncated), "a replayed episode ended early"
            self._actions[i] = [tuple(a) for a in state["actions"][i]]
            self._log[i] = [tuple(a) for a in state["log"][i]]
            self._ret[i] = state["rets"][i]
        self._stats = []


def make_selfplay_vec_env(cfg, seed_base: int, *, pool: SnapshotPool, make_policy: Callable, rng: np.random.Generator) -> SelfPlayVecEnv:
    """`cfg.num_envs` slots of `polytopia_sim.Env(size, turn_cap, full_observability, opponent,
    opponent_epsilon, agent_seat="seed_parity", reward weights)` from `cfg`."""
    import polytopia_sim

    reward = dict(terminal_weight=cfg.terminal_weight, shaping_weight=cfg.shaping_weight, gamma=cfg.gamma, potential_scale=cfg.potential_scale)

    def env_fn(opponent):
        return polytopia_sim.Env(
            **env_kwargs_of(cfg),
            opponent=opponent,
            opponent_epsilon=cfg.opponent_epsilon,
            agent_seat="seed_parity",
            **reward,
        )

    return SelfPlayVecEnv(env_fn, cfg.num_envs, seed_base, pool=pool, make_policy=make_policy, rng=rng, p_bot=cfg.p_bot, reward=reward)
