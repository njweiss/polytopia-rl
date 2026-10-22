//! Milestone 9 "Tests" item 7: `Env` behavior. Reset seats, the Bot-mode opponent loop, the
//! mirror check against `bots::play_game` (and an independent replica of it that also yields the
//! action log), two-seat mode, the three step entry points agreeing, illegal calls changing
//! nothing, truncation and termination.

mod common;

use bots::{make_bot, play_game, BotKind};
use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::Tech;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{PlayerId, State};
use sim_core::view::PlayerView;
use sim_env::actions::{num_actors, ActionMasks, ActionType, DecodeError, Factored};
use sim_env::env::{AgentSeat, Env, EnvConfig, EnvError, Opponent, StepOut};

fn env(opponent: Opponent, seat: AgentSeat, turn_cap: u32) -> Env {
    Env::new(env_cfg(opponent, seat, turn_cap))
}

fn with_max(mut c: EnvConfig, max: u32) -> EnvConfig {
    c.max_agent_actions = max;
    c
}

/// Replays `log` from the seeded start with `sim_core::actions::apply`, checking that each entry
/// was applied by the player to move.
fn replay_log(seed: u64, turn_cap: u32, log: &[(PlayerId, Action)]) -> State {
    let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg(turn_cap));
    for (i, (player, a)) in log.iter().enumerate() {
        assert_eq!(*player, s.current, "log entry {i}: wrong player");
        apply(&mut s, a).unwrap_or_else(|_| panic!("log entry {i} {a:?} is illegal"));
    }
    s
}

fn actor_mask(e: &Env) -> Vec<bool> {
    e.masks().actor().to_vec()
}

/// Everything observable about an env, for "changed nothing" checks.
#[derive(Debug, PartialEq)]
struct Snapshot {
    state: State,
    log: Vec<(PlayerId, Action)>,
    legal: Vec<Action>,
    actors: Vec<bool>,
    observer: PlayerId,
    terminal: bool,
}

fn snapshot(e: &Env) -> Snapshot {
    Snapshot {
        state: e.state().clone(),
        log: e.action_log().to_vec(),
        legal: e.legal_actions().to_vec(),
        actors: actor_mask(e),
        observer: e.observer(),
        terminal: e.is_terminal(),
    }
}

#[test]
fn reset_in_bot_mode_leaves_the_agent_to_move() {
    for seed in 0..6u64 {
        for kind in KINDS {
            for seat in [
                AgentSeat::Fixed(0),
                AgentSeat::Fixed(1),
                AgentSeat::SeedParity,
            ] {
                let mut e = env(bot(kind, 0.0), seat, 30);
                e.reset(seed);
                let agent = match seat {
                    AgentSeat::Fixed(p) => p,
                    AgentSeat::SeedParity => (seed % 2) as u8,
                };
                let ctx = format!("seed {seed} {kind:?} {seat:?}");
                assert_eq!(e.seed(), seed, "{ctx}");
                assert_eq!(e.agent_seat(), agent, "{ctx}");
                assert_eq!(e.observer(), agent, "{ctx}");
                assert_eq!(e.view().player(), agent, "{ctx}");
                assert!(!e.is_terminal(), "{ctx}");
                assert_eq!(e.state().current, agent, "{ctx}");
                assert_eq!(e.legal_actions(), legal_actions(e.state()).as_slice());
                let start = new_game_from_seed(seed, &MapParams::default(), &cfg(30));
                if agent == 0 {
                    assert!(e.action_log().is_empty(), "{ctx}");
                    assert_eq!(e.state(), &start, "{ctx}");
                } else {
                    let log = e.action_log();
                    assert!(!log.is_empty(), "{ctx}: the opponent did not play");
                    assert!(log.iter().all(|(pl, _)| *pl == 0), "{ctx}");
                    assert_eq!(log.last().unwrap().1, Action::EndTurn, "{ctx}");
                    assert_eq!(&replay_log(seed, 30, log), e.state(), "{ctx}");
                }
            }
        }
    }
}

#[test]
fn reset_in_two_seat_mode_plays_nothing() {
    for seed in 0..4u64 {
        let mut e = env(Opponent::TwoSeat, AgentSeat::SeedParity, 30);
        e.reset(seed);
        let start = new_game_from_seed(seed, &MapParams::default(), &cfg(30));
        assert_eq!(e.state(), &start);
        assert!(e.action_log().is_empty());
        assert_eq!(e.observer(), 0);
        assert_eq!(e.agent_seat(), (seed % 2) as u8);
        assert_eq!(e.legal_actions(), legal_actions(&start).as_slice());
    }
}

#[test]
fn reset_starts_a_fresh_episode() {
    let mut e = env(bot(BotKind::Rush, 0.0), AgentSeat::Fixed(0), 30);
    e.reset(3);
    for _ in 0..3 {
        e.step(&Action::EndTurn).unwrap();
    }
    e.reset(4);
    let fresh = {
        let mut f = env(bot(BotKind::Rush, 0.0), AgentSeat::Fixed(0), 30);
        f.reset(4);
        snapshot(&f)
    };
    assert_eq!(snapshot(&e), fresh);
    assert_eq!(e.seed(), 4);
}

#[test]
fn bot_step_runs_the_opponent_exactly_until_the_agent_is_to_move() {
    let mut opponent_turns = 0;
    for seed in 0..6u64 {
        for kind in KINDS {
            let seat = AgentSeat::SeedParity;
            let mut e = env(bot(kind, 0.1), seat, 12);
            e.reset(seed);
            let agent = (seed % 2) as u8;
            let mut rng = Rng::new(seed + 7);
            while !e.is_terminal() {
                let before = e.action_log().len();
                let i = random_index(e.legal_actions(), &mut rng);
                let a = e.legal_actions()[i];
                let out = e.step(&a).unwrap();
                let log = e.action_log();
                assert_eq!(
                    log[before],
                    (agent, a),
                    "the agent's action is logged first"
                );
                let opp: Vec<&(PlayerId, Action)> = log[before + 1..].iter().collect();
                assert!(opp.iter().all(|(pl, _)| *pl != agent), "agent acted twice");
                if e.is_terminal() {
                    assert!(out.terminated);
                    break;
                }
                assert_eq!(
                    e.state().current,
                    agent,
                    "not the agent's turn after a step"
                );
                if a == Action::EndTurn {
                    assert!(!opp.is_empty(), "the opponent's turn was skipped");
                    assert_eq!(opp.last().unwrap().1, Action::EndTurn);
                    opponent_turns += 1;
                } else {
                    assert!(
                        opp.is_empty(),
                        "the opponent played during the agent's turn"
                    );
                }
                assert!(!out.terminated && !out.truncated);
            }
            assert_eq!(&replay_log(seed, 12, e.action_log()), e.state());
        }
    }
    assert!(opponent_turns > 100);
}

/// An independent replica of `bots::play_game` that also returns the `(player, action)` log.
fn my_play_game(
    seed: u64,
    kinds: [BotKind; 2],
    eps: [f32; 2],
    turn_cap: u32,
) -> (State, Vec<(PlayerId, Action)>) {
    let bots = [make_bot(kinds[0], eps[0]), make_bot(kinds[1], eps[1])];
    let mut rngs = [Rng::new(seed ^ SALT[0]), Rng::new(seed ^ SALT[1])];
    let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg(turn_cap));
    let mut log = Vec::new();
    while !s.is_terminal() {
        let cur = s.current;
        let a = bots[cur as usize].choose(&PlayerView::new(&s, cur), &mut rngs[cur as usize]);
        apply(&mut s, &a).expect("bots choose legal actions");
        log.push((cur, a));
    }
    (s, log)
}

#[test]
fn an_agent_mirroring_a_bot_reproduces_play_game() {
    let combos = [
        (BotKind::Rush, 0.0, BotKind::Economy, 0.0),
        (BotKind::Economy, 0.0, BotKind::Rush, 0.0),
        (BotKind::Random, 0.0, BotKind::Random, 0.0),
        (BotKind::Rush, 0.3, BotKind::Rush, 0.3),
        (BotKind::Economy, 0.2, BotKind::Random, 0.5),
    ];
    for seed in 0..4u64 {
        for (ak, ae, ok, oe) in combos {
            for seat in [0u8, 1] {
                let ctx = format!("seed {seed} agent {ak:?}/{ae} opp {ok:?}/{oe} seat {seat}");
                let mut e = env(bot(ok, oe), AgentSeat::Fixed(seat), 30);
                e.reset(seed);
                let agent = make_bot(ak, ae);
                let mut rng = Rng::new(seed ^ SALT[seat as usize]);
                let mut steps = 0;
                while !e.is_terminal() {
                    let a = agent.choose(&e.view(), &mut rng);
                    e.step(&a).unwrap();
                    steps += 1;
                }
                let (kinds, eps) = if seat == 0 {
                    ([ak, ok], [ae, oe])
                } else {
                    ([ok, ak], [oe, ae])
                };
                let r = play_game(seed, kinds, eps, &MapParams::default(), &cfg(30));
                let s = e.state();
                assert_eq!(s.outcome, Some(r.outcome), "{ctx}");
                assert_eq!(s.turn, r.turns, "{ctx}");
                assert_eq!([score(s, 0), score(s, 1)], r.scores, "{ctx}");
                let (mine, log) = my_play_game(seed, kinds, eps, 30);
                assert_eq!(e.action_log(), log.as_slice(), "{ctx}");
                assert_eq!(s, &mine, "{ctx}");
                let agent_moves = log.iter().filter(|(pl, _)| *pl == seat).count();
                assert_eq!(steps, agent_moves, "{ctx}: one step per agent action");
            }
        }
    }
}

#[test]
fn two_seat_mode_applies_one_action_per_step_and_follows_the_mover() {
    for seed in 0..4u64 {
        let mut e = env(Opponent::TwoSeat, AgentSeat::Fixed(0), 6);
        e.reset(seed);
        let mut rng = Rng::new(seed + 11);
        let mut flips = 0;
        while !e.is_terminal() {
            let mover = e.state().current;
            assert_eq!(e.observer(), mover);
            assert_eq!(e.view().player(), mover);
            assert_eq!(e.legal_actions(), legal_actions(e.state()).as_slice());
            let before = e.action_log().len();
            let i = random_index(e.legal_actions(), &mut rng);
            let a = e.legal_actions()[i];
            e.step_legal(i).unwrap();
            assert_eq!(e.action_log().len(), before + 1);
            assert_eq!(e.action_log()[before], (mover, a));
            if !e.is_terminal() {
                assert_eq!(e.observer(), e.state().current);
                flips += usize::from(e.observer() != mover);
            }
        }
        assert!(flips >= 10, "seed {seed}: observer flipped {flips} times");
        assert_eq!(&replay_log(seed, 6, e.action_log()), e.state());
    }
}

#[test]
fn two_seat_observer_is_the_last_actor_once_terminal() {
    // Turn cap 1: player 0 ends, player 1 ends -> decided by score; state.current is 0 then.
    let mut e = env(Opponent::TwoSeat, AgentSeat::Fixed(0), 1);
    e.reset(2);
    e.step(&Action::EndTurn).unwrap();
    assert_eq!(e.observer(), 1);
    let out = e.step(&Action::EndTurn).unwrap();
    assert!(out.terminated && !out.truncated);
    assert!(e.is_terminal());
    assert_eq!(
        e.state().current,
        0,
        "test premise: the cap leaves player 0 current"
    );
    assert_eq!(e.observer(), 1);
    assert_eq!(e.view().player(), 1);
}

#[test]
fn the_three_step_entry_points_agree() {
    for (opponent, seat) in [
        (bot(BotKind::Random, 0.0), AgentSeat::Fixed(1)),
        (bot(BotKind::Rush, 0.25), AgentSeat::Fixed(0)),
        (Opponent::TwoSeat, AgentSeat::Fixed(0)),
    ] {
        for seed in 0..3u64 {
            let mk = || {
                let mut e = env(opponent, seat, 10);
                e.reset(seed);
                e
            };
            let (mut a, mut b, mut c) = (mk(), mk(), mk());
            let mut rng = Rng::new(seed + 3);
            while !a.is_terminal() {
                let i = random_index(a.legal_actions(), &mut rng);
                let act = a.legal_actions()[i];
                let (actor, kind, arg) = my_encode(a.state(), &act);
                let oa = a.step_legal(i).unwrap();
                let ob = b.step(&act).unwrap();
                let oc = c.step_factored(&Factored { actor, kind, arg }).unwrap();
                assert_eq!(oa, ob);
                assert_eq!(oa, oc);
                assert_eq!(snapshot(&a), snapshot(&b));
                assert_eq!(snapshot(&a), snapshot(&c));
            }
            assert!(b.is_terminal() && c.is_terminal());
        }
    }
}

#[test]
fn stepping_before_reset_is_not_reset() {
    let mut e = env(bot(BotKind::Rush, 0.0), AgentSeat::Fixed(0), 30);
    assert_eq!(e.step(&Action::EndTurn), Err(EnvError::NotReset));
    assert_eq!(e.step_legal(0), Err(EnvError::NotReset));
    let f = Factored {
        actor: num_actors(11) - 1,
        kind: ActionType::EndTurn,
        arg: 0,
    };
    assert_eq!(e.step_factored(&f), Err(EnvError::NotReset));
}

#[test]
fn illegal_calls_change_nothing() {
    for (opponent, seat) in [
        (bot(BotKind::Economy, 0.0), AgentSeat::Fixed(1)),
        (Opponent::TwoSeat, AgentSeat::Fixed(0)),
    ] {
        let mut e = env(opponent, seat, 30);
        e.reset(1);
        // Play a little first so the state is not the start.
        e.step(&Action::EndTurn).unwrap();
        let before = snapshot(&e);
        let n = 11usize * 11;
        let unit = e
            .state()
            .units
            .iter()
            .find(|u| u.owner == e.state().current)
            .unwrap();
        let unit_tile = ti(11, unit.pos);
        let far = (0..n)
            .find(|&t| cheb(tile_of(11, t), unit.pos) > 3)
            .unwrap();

        // Illegal: researching Mining without Climbing.
        assert_eq!(
            e.step(&Action::Research { tech: Tech::Mining }),
            Err(EnvError::Illegal)
        );
        // Illegal through the factored path: a decodable move to a far tile.
        assert_eq!(
            e.step_factored(&Factored {
                actor: unit_tile,
                kind: ActionType::Move,
                arg: far
            }),
            Err(EnvError::Illegal)
        );
        // Bad index.
        let len = e.legal_actions().len();
        assert_eq!(e.step_legal(len), Err(EnvError::BadIndex));
        assert_eq!(e.step_legal(usize::MAX), Err(EnvError::BadIndex));
        // Decode errors.
        assert_eq!(
            e.step_factored(&Factored {
                actor: 3 * n,
                kind: ActionType::Move,
                arg: 0
            }),
            Err(EnvError::Decode(DecodeError::WrongKind))
        );
        assert_eq!(
            e.step_factored(&Factored {
                actor: 3 * n,
                kind: ActionType::EndTurn,
                arg: 1
            }),
            Err(EnvError::Decode(DecodeError::BadArg))
        );
        let out_of_range = e.step_factored(&Factored {
            actor: 3 * n + 1,
            kind: ActionType::EndTurn,
            arg: 0,
        });
        assert!(
            matches!(
                out_of_range,
                Err(EnvError::BadIndex) | Err(EnvError::Decode(DecodeError::ActorOutOfRange))
            ),
            "{out_of_range:?}"
        );
        assert_eq!(snapshot(&e), before, "{opponent:?}");
    }
}

#[test]
fn illegal_calls_do_not_count_toward_truncation() {
    let c = with_max(
        env_cfg(bot(BotKind::Economy, 0.0), AgentSeat::Fixed(0), 30),
        3,
    );
    let mut e = Env::new(c);
    e.reset(0);
    assert!(!e.step(&Action::EndTurn).unwrap().truncated);
    for _ in 0..5 {
        assert!(e.step(&Action::Research { tech: Tech::Mining }).is_err());
        assert!(e.step_legal(10_000).is_err());
    }
    assert!(!e.step(&Action::EndTurn).unwrap().truncated);
    let out = e.step(&Action::EndTurn).unwrap();
    assert!(out.truncated && !out.terminated);
}

#[test]
fn truncation_counts_agent_actions_only() {
    for seed in 0..3u64 {
        let c = with_max(
            env_cfg(bot(BotKind::Economy, 0.0), AgentSeat::Fixed(0), 30),
            4,
        );
        let mut e = Env::new(c);
        e.reset(seed);
        for k in 1..=3 {
            let out = e.step(&Action::EndTurn).unwrap();
            assert!(!out.truncated && !out.terminated, "seed {seed} step {k}");
        }
        assert!(
            e.action_log().len() >= 6,
            "test premise: the opponent played several actions"
        );
        let out = e.step(&Action::EndTurn).unwrap();
        assert!(out.truncated, "seed {seed}");
        assert!(!out.terminated);
        assert!(!e.state().is_terminal());
        // The episode is finished.
        let before = snapshot(&e);
        assert_eq!(e.step(&Action::EndTurn), Err(EnvError::Finished));
        assert_eq!(e.step_legal(0), Err(EnvError::Finished));
        let f = Factored {
            actor: 3 * 121,
            kind: ActionType::EndTurn,
            arg: 0,
        };
        assert_eq!(e.step_factored(&f), Err(EnvError::Finished));
        assert_eq!(snapshot(&e), before);
        // Reset clears the counter.
        e.reset(seed);
        for _ in 1..=3 {
            assert!(!e.step(&Action::EndTurn).unwrap().truncated);
        }
        assert!(e.step(&Action::EndTurn).unwrap().truncated);
    }
}

#[test]
fn two_seat_truncation_counts_every_caller_action() {
    let c = with_max(env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), 30), 5);
    let mut e = Env::new(c);
    e.reset(9);
    for _ in 0..4 {
        assert!(!e.step(&Action::EndTurn).unwrap().truncated);
    }
    assert!(e.step(&Action::EndTurn).unwrap().truncated);
    assert_eq!(e.action_log().len(), 5);
}

#[test]
fn reaching_the_turn_cap_is_terminated_not_truncated() {
    // Two-seat, cap 2: four EndTurns end the game, exactly when max_agent_actions is reached.
    let c = with_max(env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), 2), 4);
    let mut e = Env::new(c);
    e.reset(5);
    for _ in 0..3 {
        let out = e.step(&Action::EndTurn).unwrap();
        assert!(!out.terminated && !out.truncated);
    }
    let out = e.step(&Action::EndTurn).unwrap();
    assert!(out.terminated, "the cap terminates");
    assert!(!out.truncated, "a terminal step is never truncated");
    assert!(e.is_terminal());

    // Bot mode with a random agent, cap 3.
    for seed in 0..4u64 {
        let mut e = env(bot(BotKind::Random, 0.0), AgentSeat::SeedParity, 3);
        e.reset(seed);
        let mut rng = Rng::new(seed);
        let mut last = None;
        while !e.is_terminal() {
            let i = random_index(e.legal_actions(), &mut rng);
            last = Some(e.step_legal(i).unwrap());
        }
        let last: StepOut = last.unwrap();
        assert!(last.terminated && !last.truncated, "seed {seed}");
        assert!(e.legal_actions().is_empty());
        assert!(e.masks().actor().iter().all(|&x| !x));
        assert_eq!(e.observer(), (seed % 2) as u8);
        assert_eq!(e.step(&Action::EndTurn), Err(EnvError::Finished));
        assert_eq!(e.step_legal(0), Err(EnvError::Finished));
    }
}

#[test]
fn masks_match_the_legal_list() {
    for seed in 0..3u64 {
        let mut e = env(bot(BotKind::Rush, 0.0), AgentSeat::SeedParity, 8);
        e.reset(seed);
        let mut rng = Rng::new(seed);
        while !e.is_terminal() {
            let fresh = ActionMasks::new(e.state(), e.legal_actions());
            assert_eq!(e.masks().actor(), fresh.actor());
            for a in 0..num_actors(11) {
                assert_eq!(e.masks().kind(a), fresh.kind(a));
                for k in ActionType::ALL {
                    assert_eq!(e.masks().arg(a, k), fresh.arg(a, k));
                }
            }
            let i = random_index(e.legal_actions(), &mut rng);
            e.step_legal(i).unwrap();
        }
    }
}
