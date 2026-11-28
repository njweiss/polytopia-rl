"""Snapshot pool, matchmaking and the self-play vector env (docs/MILESTONE_11, MILESTONE_23).

An env is anything with the `polytopia_sim.Env` methods `reset(seed)`, `observation()`,
`actor_mask()`, `kind_mask(actor)`, `arg_mask(actor, kind)`, `step_factored(actor, kind, arg)`,
`info()` (keys `seed`, `turn`, `current_player`, `agent_seat`, `scores`, `outcome`,
`agent_actions`) and `spec()`.

A batch policy is a callable `policy(envs) -> list[(actor, kind, arg)]` that chooses one action for
each env (all on the move of the seat it plays) in a single batched forward pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from polytopia_rl.pfsp import BotSpec, WinRateTracker, parse_bot_specs, pfsp_weights
from polytopia_rl.vec_env import EpisodeStat, SyncVecEnv, VecState, VecStep, agent_outcome, env_kwargs_of, env_masks_fn

BOT_OPPONENTS = ("economy", "rush")
LIVE_SELF = "self"


def _current_player(env) -> int:
    if hasattr(env, "current_player"):
        return env.current_player()
    return env.info()["current_player"]


def _is_bot(opp) -> bool:
    return isinstance(opp, BotSpec) or (isinstance(opp, str) and opp != LIVE_SELF)


def _opp_name(opp) -> str:
    if isinstance(opp, (Entry, BotSpec)):
        return opp.name
    return opp


@dataclass
class Transition:
    """A learner transition completed during `SelfPlayVecEnv.tick` (docs/MILESTONE_13 "Ticks")."""

    slot: int
    reward: float  # the M9 formula over the learner's action and everything up to its next move or the episode end
    terminated: bool
    truncated: bool
    final_board: np.ndarray | None  # as VecStep: the final observation when the episode ended
    final_globals: np.ndarray | None
    seat: int = 0


@dataclass
class Entry:
    step: int
    state: dict  # a frozen copy of the learner's state dict at `step` (CPU tensors)
    name: str  # f"snap_{step}" or f"anchor_{i}"
    pinned: bool = False
    net_template: Any | None = None


class SnapshotPool:
    """Snapshots in the order added. `add` appends, then thins unpinned entries while there are
    more than `pool_max` unpinned entries."""

    def __init__(self, pool_max: int = 40, p_latest: float = 0.5):
        if pool_max < 2:
            raise ValueError("pool_max must be at least 2 (thinning keeps the first and the newest)")
        self.pool_max = pool_max
        self.p_latest = p_latest
        self.entries: list[Entry] = []

    def add(
        self,
        model_state: dict,
        step: int,
        *,
        pinned: bool = False,
        name: str | None = None,
        net_template=None,
    ) -> Entry:
        """Append a deep copy of `model_state` (later changes to the learner do not reach it)."""
        entry_name = name if name is not None else f"snap_{step}"
        entry = Entry(
            step,
            {k: v.detach().cpu().clone() for k, v in model_state.items()},
            entry_name,
            pinned=pinned,
            net_template=net_template,
        )
        self.entries.append(entry)
        while sum(1 for e in self.entries if not e.pinned) > self.pool_max:
            self.thin()
        return entry

    def thin(self) -> None:
        """Keep every pinned entry and unpinned entries at even positions (0, 2, 4, ...) plus the
        newest unpinned entry; drop the rest."""
        unpinned = [e for e in self.entries if not e.pinned]
        if not unpinned:
            return
        keep_unpinned = unpinned[::2]
        if keep_unpinned[-1] is not unpinned[-1]:
            keep_unpinned.append(unpinned[-1])
        keep_ids = {id(e) for e in keep_unpinned}
        self.entries = [e for e in self.entries if e.pinned or id(e) in keep_ids]

    def latest_snapshot(self) -> Entry | None:
        """The newest unpinned entry, else `entries[-1]` if all entries are pinned, else None."""
        if not self.entries:
            return None
        for e in reversed(self.entries):
            if not e.pinned:
                return e
        return self.entries[-1]

    def sample(
        self,
        rng: np.random.Generator,
        *,
        tracker: WinRateTracker | None = None,
        pfsp_mode: str = "off",
        pfsp_power: float = 2.0,
    ) -> Entry:
        """The newest unpinned entry with probability `p_latest`, else an entry drawn from all
        entries (uniformly when `pfsp_mode == "off"`, or PFSP-weighted otherwise)."""
        if not self.entries:
            raise ValueError("cannot sample from an empty SnapshotPool")
        if rng.random() < self.p_latest:
            return self.latest_snapshot()
        if pfsp_mode == "off":
            return self.entries[rng.integers(len(self.entries))]
        wrs = [tracker.win_rate(e.name) if tracker is not None else 0.5 for e in self.entries]
        probs = pfsp_weights(wrs, pfsp_mode, power=pfsp_power)
        return self.entries[int(rng.choice(len(self.entries), p=probs))]


def draw_opponent(
    pool: SnapshotPool,
    rng: np.random.Generator,
    p_bot: float,
    *,
    p_self: float = 0.0,
    bots: Sequence[BotSpec] | None = None,
    tracker: WinRateTracker | None = None,
    pfsp_mode: str = "off",
    pfsp_power: float = 2.0,
):
    """Draw a bot (`BotSpec` when `bots` is given, else legacy `str`), `LIVE_SELF`, or pool `Entry`."""
    u = rng.random()
    if u < p_bot:
        if bots is None:
            if pfsp_mode == "off":
                return BOT_OPPONENTS[rng.integers(len(BOT_OPPONENTS))]
            wrs = [tracker.win_rate(b) if tracker is not None else 0.5 for b in BOT_OPPONENTS]
            probs = pfsp_weights(wrs, pfsp_mode, power=pfsp_power)
            return BOT_OPPONENTS[int(rng.choice(len(BOT_OPPONENTS), p=probs))]
        wrs = [tracker.win_rate(b.name) if tracker is not None else 0.5 for b in bots]
        probs = pfsp_weights(wrs, pfsp_mode, power=pfsp_power, base_weights=[b.weight for b in bots])
        return bots[int(rng.choice(len(bots), p=probs))]
    if u < p_bot + p_self:
        return LIVE_SELF
    return pool.sample(rng, tracker=tracker, pfsp_mode=pfsp_mode, pfsp_power=pfsp_power)


def net_batch_policy(net, generator, greedy: bool = False, device: str = "cpu") -> Callable:
    """A batch policy for a `PolicyNet`: one `trunk` + `act` call for all the envs, masks from
    each env's own observation and masks; sampled with `generator`, or argmax when `greedy`."""

    def policy(envs):
        obs = [env.observation() for env in envs]
        board = torch.from_numpy(np.stack([b for b, _ in obs])).to(device)
        glob = torch.from_numpy(np.stack([g for _, g in obs])).to(device)
        masks_fn = env_masks_fn(envs, board.shape[2] * board.shape[3])
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
    playing against bots, pool snapshots/anchors, and live two-sided self-play."""

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
        stacked_policy: Callable | None = None,
        p_self: float = 0.0,
        bots: Sequence[BotSpec] | None = None,
        tracker: WinRateTracker | None = None,
        pfsp_mode: str = "off",
        pfsp_power: float = 2.0,
        auto_stack: bool = False,
        python_bot_shaping: bool = False,
        paired_seeds: bool = False,
    ):
        self.env_fn = env_fn
        self.num_envs = num_envs
        self.seed_base = seed_base
        self.seed_stride = seed_stride
        self.pool = pool
        self.make_policy = make_policy
        self.stacked_policy = stacked_policy
        self.rng = rng
        self.p_bot = p_bot
        self.p_self = p_self
        self.bots = list(bots) if bots is not None else None
        self._bot_map = {b.name: b for b in self.bots} if self.bots is not None else {}
        self.tracker = tracker
        self.pfsp_mode = pfsp_mode
        self.pfsp_power = pfsp_power
        self.auto_stack = auto_stack
        self.python_bot_shaping = python_bot_shaping
        self.paired_seeds = paired_seeds
        self.reward = {k: np.float32(reward[k]) for k in ("terminal_weight", "shaping_weight", "gamma", "potential_scale")}
        self._env_cache = [{} for _ in range(num_envs)]
        self._policies = {}
        self.envs = [self._env(i, "none") for i in range(num_envs)]
        spec = self.envs[0].spec()
        self.num_tiles = spec["height"] * spec["width"]
        self.board_shape = (spec["channels"], spec["height"], spec["width"])
        self.num_globals = spec["globals"]
        self._next_k = [0] * num_envs
        self._seed = [None] * num_envs
        self._seat = [0] * num_envs
        self._opp = [None] * num_envs
        self._actions = [[] for _ in range(num_envs)]
        self._log = [[] for _ in range(num_envs)]
        self._ended = [None] * num_envs
        self._ret = [0.0] * num_envs
        self._before = [None] * num_envs
        self._self_before = [[None, None] for _ in range(num_envs)]
        self._self_ret = [[0.0, 0.0] for _ in range(num_envs)]
        self._self_len = [[0, 0] for _ in range(num_envs)]
        self._mirror_queue: list[tuple[int, object, int]] = []
        self._next_seed_idx = 0
        self._stats: list[EpisodeStat] = []
        for i in range(num_envs):
            self._check_seed(self._episode_seed(i, 0))

    def set_shaping_weight(self, w: float) -> None:
        self.reward["shaping_weight"] = np.float32(w)

    def learner_seat(self, i: int) -> int:
        if self._opp[i] == LIVE_SELF:
            return _current_player(self.envs[i])
        return self._seat[i]

    def _env_key(self, opp, seat: int):
        if isinstance(opp, Entry) or opp == LIVE_SELF:
            return "none"
        if isinstance(opp, BotSpec):
            return (opp.bot, opp.epsilon, seat if self.paired_seeds else "seed_parity")
        if self.paired_seeds:
            return (opp, None, seat)
        return opp

    def _env(self, i: int, kind):
        if kind not in self._env_cache[i]:
            self._env_cache[i][kind] = self.env_fn(kind)
        return self._env_cache[i][kind]

    def _policy(self, entry: Entry):
        if entry.name not in self._policies:
            self._policies[entry.name] = self.make_policy(entry)
        return self._policies[entry.name]

    def _stack_entries(self) -> list[Entry]:
        """Stackable pool entries (`net_template is None`), then stackable slot opponents that
        left the pool, in slot order of first appearance (docs/QUESTIONS.md Q38)."""
        entries = [e for e in self.pool.entries if e.net_template is None]
        names = {e.name for e in entries}
        for opp in self._opp:
            if isinstance(opp, Entry) and opp.net_template is None and opp.name not in names:
                entries.append(opp)
                names.add(opp.name)
        return entries

    def _draw_opp(self):
        return draw_opponent(
            self.pool,
            self.rng,
            self.p_bot,
            p_self=self.p_self,
            bots=self.bots,
            tracker=self.tracker,
            pfsp_mode=self.pfsp_mode,
            pfsp_power=self.pfsp_power,
        )

    def _start(self, i: int, seed: int, opp, seat: int | None = None) -> None:
        """Begin slot `i`'s episode on `seed` against `opp` (the opponent's opening moves, if any,
        are left to `_advance`)."""
        resolved_seat = (seed % 2) if seat is None else int(seat)
        env = self._env(i, self._env_key(opp, resolved_seat))
        env.reset(seed)
        self.envs[i] = env
        self._opp[i] = opp
        self._seed[i] = seed
        self._seat[i] = resolved_seat
        self._actions[i] = []
        self._log[i] = []
        self._ended[i] = None
        self._ret[i] = 0.0
        self._before[i] = None
        self._self_before[i] = [None, None]
        self._self_ret[i] = [0.0, 0.0]
        self._self_len[i] = [0, 0]

    def _reset(self, i: int) -> None:
        if self.paired_seeds:
            if self._mirror_queue:
                seed, opp, seat = self._mirror_queue.pop(0)
                self._start(i, seed, opp, seat)
            else:
                seed = self.seed_base + self._next_seed_idx * self.seed_stride
                self._check_seed(seed)
                self._next_seed_idx += 1
                opp = self._draw_opp()
                self._start(i, seed, opp, 0)
                if opp != LIVE_SELF:
                    self._mirror_queue.append((seed, opp, 1))
        else:
            seed = self._episode_seed(i, self._next_k[i])
            self._check_seed(seed)
            self._start(i, seed, self._draw_opp())
            self._next_k[i] += 1
        live = (
            {e.name for e in self.pool.entries}
            | {o.name for o in self._opp if isinstance(o, Entry)}
            | {o.name for _, o, _ in self._mirror_queue if isinstance(o, Entry)}
        )
        self._policies = {k: v for k, v in self._policies.items() if k in live}

    def _opponent_to_move(self, i: int) -> bool:
        return isinstance(self._opp[i], Entry) and self._ended[i] is None and _current_player(self.envs[i]) != self._seat[i]

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

    def _seat_reward(self, i: int, seat: int, before) -> np.float32:
        """The M9 reward of `seat` from `before` scores to the current state (potential 0 when terminated)."""
        r = self.reward
        info = self.envs[i].info()
        terminated = self._ended[i] is not None and self._ended[i][0]
        t = np.float32({f"win_{seat}": 1.0, f"win_{1 - seat}": -1.0}.get(info["outcome"], 0.0))
        after = np.float32(0.0) if terminated else self._potential(info["scores"], seat)
        return r["terminal_weight"] * t + r["shaping_weight"] * (r["gamma"] * after - self._potential(before, seat))

    def _snapshot_reward(self, i: int, before) -> np.float32:
        return self._seat_reward(i, self._seat[i], before)

    def _bot_shaping_delta(self, i: int, before, terminated: bool) -> np.float32:
        r = self.reward
        seat = self._seat[i]
        after = np.float32(0.0) if terminated else self._potential(self.envs[i].info()["scores"], seat)
        return r["shaping_weight"] * (r["gamma"] * after - self._potential(before, seat))

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
        before = {}
        for i, env in enumerate(self.envs):
            action = (int(actors[i]), int(kinds[i]), int(args[i]))
            snapshot = isinstance(self._opp[i], Entry)
            bot_before = None
            if snapshot:
                before[i] = env.info()["scores"]
            elif self.python_bot_shaping:
                bot_before = env.info()["scores"]
            reward, _, terminated, truncated = env.step_factored(*action)
            self._actions[i].append(action)
            if snapshot:
                self._log[i].append(action)
            else:
                r_val = np.float32(reward)
                if bot_before is not None:
                    r_val = r_val + self._bot_shaping_delta(i, bot_before, terminated)
                out.reward[i] = r_val
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
        return [
            i
            for i, env in enumerate(self.envs)
            if self._ended[i] is None and (self._opp[i] == LIVE_SELF or _current_player(env) == self._seat[i])
        ]

    def tick(self, learner_actions: dict[int, tuple[int, int, int]]) -> list[Transition]:
        """Apply one learner action to every slot in `learner_actions` (all must be in
        `learner_slots()`), then one opponent move to every other slot whose snapshot opponent is to
        move, handle episode ends and auto-resets, and return completed transitions sorted by
        `(slot, seat)`."""
        bad = sorted(set(learner_actions) - set(self.learner_slots()))
        if bad:
            raise ValueError(f"slots {bad} are not on their learner's move")
        bot_reward = {}
        self_trs: dict[int, list[Transition]] = {}
        for i in sorted(learner_actions):
            action = tuple(int(x) for x in learner_actions[i])
            env = self.envs[i]
            opp = self._opp[i]
            if opp == LIVE_SELF:
                seat = _current_player(env)
                before_scores = env.info()["scores"]
                _, _, terminated, truncated = env.step_factored(*action)
                self._log[i].append(action)
                self._self_len[i][seat] += 1
                if terminated or truncated:
                    self._ended[i] = (terminated, truncated)
                    fb, fg = env.observation()
                    trs_i = []
                    r_s = self._seat_reward(i, seat, before_scores)
                    self._self_ret[i][seat] += float(r_s)
                    trs_i.append(
                        Transition(
                            slot=i,
                            seat=seat,
                            reward=float(r_s),
                            terminated=terminated,
                            truncated=truncated,
                            final_board=fb.copy(),
                            final_globals=fg.copy(),
                        )
                    )
                    other = 1 - seat
                    if self._self_before[i][other] is not None:
                        r_o = self._seat_reward(i, other, self._self_before[i][other])
                        self._self_ret[i][other] += float(r_o)
                        self._self_before[i][other] = None
                        trs_i.append(
                            Transition(
                                slot=i,
                                seat=other,
                                reward=float(r_o),
                                terminated=terminated,
                                truncated=truncated,
                                final_board=fb.copy(),
                                final_globals=fg.copy(),
                            )
                        )
                    trs_i.sort(key=lambda t: (t.slot, t.seat))
                    self_trs[i] = trs_i
                elif _current_player(env) == seat:
                    r_s = self._seat_reward(i, seat, before_scores)
                    self._self_ret[i][seat] += float(r_s)
                    self_trs[i] = [
                        Transition(
                            slot=i,
                            seat=seat,
                            reward=float(r_s),
                            terminated=False,
                            truncated=False,
                            final_board=None,
                            final_globals=None,
                        )
                    ]
                else:
                    other = 1 - seat
                    self._self_before[i][seat] = before_scores
                    if self._self_before[i][other] is not None:
                        r_o = self._seat_reward(i, other, self._self_before[i][other])
                        self._self_ret[i][other] += float(r_o)
                        self._self_before[i][other] = None
                        self_trs[i] = [
                            Transition(
                                slot=i,
                                seat=other,
                                reward=float(r_o),
                                terminated=False,
                                truncated=False,
                                final_board=None,
                                final_globals=None,
                            )
                        ]
                continue

            snapshot = isinstance(opp, Entry)
            bot_before = None
            if snapshot:
                self._before[i] = env.info()["scores"]
            elif self.python_bot_shaping:
                bot_before = env.info()["scores"]
            reward, _, terminated, truncated = env.step_factored(*action)
            self._actions[i].append(action)
            if snapshot:
                self._log[i].append(action)
            else:
                r_val = np.float32(reward)
                if bot_before is not None:
                    r_val = r_val + self._bot_shaping_delta(i, bot_before, terminated)
                bot_reward[i] = r_val
            if terminated or truncated:
                self._ended[i] = (terminated, truncated)

        # One opponent move per slot whose snapshot/anchor opponent is to move, grouped by entry
        # name in order of first appearance by slot.
        groups: dict[str, list[int]] = {}
        for i in range(self.num_envs):
            if self._opponent_to_move(i):
                groups.setdefault(self._opp[i].name, []).append(i)

        group_actions: dict[str, list] = {}
        if groups:
            if self.stacked_policy is None:
                for name, idx in groups.items():
                    group_actions[name] = self._policy(self._opp[idx[0]])([self.envs[i] for i in idx])
            else:
                stackable: list[tuple[str, list[int]]] = []
                non_stackable: list[tuple[str, list[int]]] = []
                for name, idx in groups.items():
                    if self._opp[idx[0]].net_template is not None:
                        non_stackable.append((name, idx))
                    else:
                        stackable.append((name, idx))
                for name, idx in non_stackable:
                    group_actions[name] = self._policy(self._opp[idx[0]])([self.envs[i] for i in idx])
                use_stacked = len(stackable) > 0 and (not self.auto_stack or len(stackable) >= 2)
                if use_stacked:
                    self.stacked_policy.sync(self._stack_entries())
                    stacked_out = self.stacked_policy([(name, [self.envs[i] for i in idx]) for name, idx in stackable])
                    for (name, _), acts in zip(stackable, stacked_out):
                        group_actions[name] = acts
                else:
                    for name, idx in stackable:
                        group_actions[name] = self._policy(self._opp[idx[0]])([self.envs[i] for i in idx])

        for name, idx in groups.items():
            for i, a in zip(idx, group_actions[name]):
                a = tuple(int(x) for x in a)
                _, _, terminated, truncated = self.envs[i].step_factored(*a)
                self._log[i].append(a)
                if terminated or truncated:
                    self._ended[i] = (terminated, truncated)

        out = []
        for i in range(self.num_envs):
            if self._opp[i] == LIVE_SELF:
                if i in self_trs:
                    out.extend(self_trs[i])
                if self._ended[i] is not None:
                    self._stats.append(self._episode_stat(i))
                    self._reset(i)
                continue
            if i in bot_reward:
                reward = np.float32(bot_reward[i])
            elif self._before[i] is not None and (self._ended[i] is not None or not self._opponent_to_move(i)):
                reward = self._snapshot_reward(i, self._before[i])
                self._before[i] = None
            elif self._ended[i] is not None:
                # Episode ended during the opponent's opening turn before the learner ever acted.
                self._stats.append(self._episode_stat(i))
                self._reset(i)
                continue
            else:
                continue
            self._ret[i] += float(reward)
            tr = Transition(
                slot=i,
                seat=self._seat[i],
                reward=float(reward),
                terminated=False,
                truncated=False,
                final_board=None,
                final_globals=None,
            )
            if self._ended[i] is not None:
                tr.terminated, tr.truncated = self._ended[i]
                tr.final_board, tr.final_globals = self.envs[i].observation()
                self._stats.append(self._episode_stat(i))
                self._reset(i)
            out.append(tr)
        out.sort(key=lambda tr: (tr.slot, tr.seat))
        return out

    def _episode_stat(self, i: int) -> EpisodeStat:
        opp = self._opp[i]
        if opp == LIVE_SELF:
            return EpisodeStat(
                env_index=i,
                seed=self._seed[i],
                opponent=LIVE_SELF,
                agent_seat=0,
                outcome=agent_outcome(self.envs[i].info()["outcome"], 0),
                length=sum(self._self_len[i]),
                ret=self._self_ret[i][0],
            )
        seat = self._seat[i]
        return EpisodeStat(
            env_index=i,
            seed=self._seed[i],
            opponent=_opp_name(opp),
            agent_seat=seat,
            outcome=agent_outcome(self.envs[i].info()["outcome"], seat),
            length=len(self._actions[i]),
            ret=self._ret[i],
        )

    def state_dict(self) -> dict:
        """Per slot: episode index, seed, opponent, seat, actions/log so far (replayed on load),
        mid-turn pending scores, live-self state, paired-seed mirror queue, and gone pool entries."""
        in_pool = {e.name for e in self.pool.entries}
        gone = {}
        for o in self._opp:
            if isinstance(o, Entry) and o.name not in in_pool and o.name not in gone:
                gone[o.name] = {"step": o.step, "state": o.state, "pinned": o.pinned}
        for _, o, _ in self._mirror_queue:
            if isinstance(o, Entry) and o.name not in in_pool and o.name not in gone:
                gone[o.name] = {"step": o.step, "state": o.state, "pinned": o.pinned}
        sd = {
            "seed_base": self.seed_base,
            "seed_stride": self.seed_stride,
            "next_k": list(self._next_k),
            "seeds": list(self._seed),
            "opponents": [_opp_name(o) for o in self._opp],
            "actions": [[list(a) for a in acts] for acts in self._actions],
            "log": [[list(a) for a in acts] for acts in self._log],
            "rets": list(self._ret),
            "entries": gone,
            "before": [None if b is None else list(b) for b in self._before],
            "self_before": [[None if s is None else list(s) for s in sb] for sb in self._self_before],
            "self_ret": [list(sr) for sr in self._self_ret],
            "self_len": [list(sl) for sl in self._self_len],
            "mirror_queue": [[seed, _opp_name(opp), seat] for seed, opp, seat in self._mirror_queue],
            "next_seed_idx": self._next_seed_idx,
        }
        if self.paired_seeds:
            sd["seats"] = list(self._seat)
        return sd

    def load_state_dict(self, state: dict) -> None:
        """Restore after the pool has been restored."""
        self.seed_base = state["seed_base"]
        self.seed_stride = state["seed_stride"]
        entries = {e.name: e for e in self.pool.entries}
        for name, e in state["entries"].items():
            entries[name] = Entry(e["step"], e["state"], name, pinned=bool(e.get("pinned", False)))

        def resolve(name: str):
            if name == LIVE_SELF:
                return LIVE_SELF
            if name in self._bot_map:
                return self._bot_map[name]
            if name in BOT_OPPONENTS:
                return name
            return entries[name]

        seats = state.get("seats")
        befores = state.get("before")
        self_befores = state.get("self_before")
        self_rets = state.get("self_ret")
        self_lens = state.get("self_len")

        for i in range(self.num_envs):
            name = state["opponents"][i]
            opp = resolve(name)
            if self.paired_seeds and seats is not None:
                self._start(i, state["seeds"][i], opp, seats[i])
            else:
                self._start(i, state["seeds"][i], opp)
            self._next_k[i] = state["next_k"][i]
            replay = state["actions"][i] if _is_bot(opp) else state["log"][i]
            for action in replay:
                _, _, terminated, truncated = self.envs[i].step_factored(*action)
                assert not (terminated or truncated), "a replayed episode ended early"
            self._actions[i] = [tuple(a) for a in state["actions"][i]]
            self._log[i] = [tuple(a) for a in state["log"][i]]
            self._ret[i] = state["rets"][i]
            if befores is not None and befores[i] is not None:
                self._before[i] = tuple(befores[i])
            if self_befores is not None:
                self._self_before[i] = [None if s is None else tuple(s) for s in self_befores[i]]
            if self_rets is not None:
                self._self_ret[i] = [float(x) for x in self_rets[i]]
            if self_lens is not None:
                self._self_len[i] = [int(x) for x in self_lens[i]]

        self._mirror_queue = [(int(seed), resolve(name), int(seat)) for seed, name, seat in state.get("mirror_queue", [])]
        self._next_seed_idx = int(state.get("next_seed_idx", 0))
        self._stats = []


def make_selfplay_vec_env(
    cfg,
    seed_base: int,
    *,
    pool: SnapshotPool,
    make_policy: Callable,
    rng: np.random.Generator,
    stacked_policy: Callable | None = None,
    tracker: WinRateTracker | None = None,
) -> SelfPlayVecEnv:
    """`cfg.num_envs` slots of `polytopia_sim.Env` configured from `cfg`."""
    import polytopia_sim

    reward = dict(
        terminal_weight=cfg.terminal_weight,
        shaping_weight=cfg.shaping_weight,
        gamma=cfg.gamma,
        potential_scale=cfg.potential_scale,
    )
    python_bot_shaping = getattr(cfg, "shaping_anneal_steps", None) is not None
    bot_reward = {**reward, "shaping_weight": 0.0} if python_bot_shaping else reward

    def env_fn(key):
        if key == "none":
            return polytopia_sim.Env(
                **env_kwargs_of(cfg),
                opponent="none",
                opponent_epsilon=cfg.opponent_epsilon,
                agent_seat="seed_parity",
                **reward,
            )
        if isinstance(key, tuple):
            bot_name, epsilon, seat_key = key
            return polytopia_sim.Env(
                **env_kwargs_of(cfg),
                opponent=bot_name,
                opponent_epsilon=cfg.opponent_epsilon if epsilon is None else epsilon,
                agent_seat=seat_key,
                **bot_reward,
            )
        return polytopia_sim.Env(
            **env_kwargs_of(cfg),
            opponent=key,
            opponent_epsilon=cfg.opponent_epsilon,
            agent_seat="seed_parity",
            **bot_reward,
        )

    raw_bots = getattr(cfg, "selfplay_bots", None)
    bots = parse_bot_specs(raw_bots, default_epsilon=cfg.opponent_epsilon) if raw_bots is not None else None
    return SelfPlayVecEnv(
        env_fn,
        cfg.num_envs,
        seed_base,
        pool=pool,
        make_policy=make_policy,
        rng=rng,
        p_bot=cfg.p_bot,
        reward=reward,
        stacked_policy=stacked_policy,
        p_self=getattr(cfg, "p_self", 0.0),
        bots=bots,
        tracker=tracker,
        pfsp_mode=getattr(cfg, "pfsp_mode", "off"),
        pfsp_power=getattr(cfg, "pfsp_power", 2.0),
        auto_stack=(getattr(cfg, "opponent_forward", "stacked") == "auto"),
        python_bot_shaping=python_bot_shaping,
        paired_seeds=getattr(cfg, "paired_seeds", False),
    )
