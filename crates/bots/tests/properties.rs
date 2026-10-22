//! Milestone 8: bot properties (docs/MILESTONE_8 "Random bot and epsilon wrapper", "Progress and
//! termination"; "Tests" item 3). Games are driven by `common::my_play_game` (its own loop and
//! rngs); legality is checked with `sim_core::actions::legal_actions`.

mod common;

use bots::{bot_rng, make_bot, Bot, BotKind, EconomyBot, RandomBot, RushBot};
use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::UnitType;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Phase, State, UnitId};
use sim_core::view::PlayerView;

#[test]
fn bot_kind_names() {
    assert_eq!(BotKind::Random.name(), "random");
    assert_eq!(BotKind::Rush.name(), "rush");
    assert_eq!(BotKind::Economy.name(), "economy");
    // Assumption: a bare bot's name is its kind's name.
    for k in KINDS {
        assert_eq!(make_bot(k, 0.0).name(), k.name());
    }
    assert_eq!(RandomBot.name(), "random");
    assert_eq!(RushBot.name(), "rush");
    assert_eq!(EconomyBot.name(), "economy");
}

#[test]
fn bot_rng_is_the_salted_seed() {
    for seed in [0u64, 1, 42, 0x0123_4567_89ab_cdef, u64::MAX] {
        for player in 0..2u8 {
            let mut got = bot_rng(seed, player);
            let mut want = Rng::new(seed ^ SALT[player as usize]);
            assert_eq!(got, want, "seed {seed} player {player}");
            for _ in 0..4 {
                assert_eq!(got.next_u32(), want.next_u32());
            }
        }
    }
    // The published salts, spelled out once more.
    assert_eq!(bot_rng(0, 0), Rng::new(0x9e37_79b9_7f4a_7c15));
    assert_eq!(bot_rng(0, 1), Rng::new(0xc2b2_ae3d_27d4_eb4f));
}

/// Plays one game with `my_play_game`'s loop, checking every chosen action is legal and that a
/// Rush or Economy player reaches EndTurn within 200 of its own actions per turn.
fn check_game(seed: u64, kinds: [BotKind; 2], eps: f32, turn_cap: u32) {
    let bots = [make_bot(kinds[0], eps), make_bot(kinds[1], eps)];
    let mut rngs = [Rng::new(seed ^ SALT[0]), Rng::new(seed ^ SALT[1])];
    let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg(turn_cap));
    let mut in_turn = 0usize;
    while !s.is_terminal() {
        let cur = s.current as usize;
        let a = bots[cur].choose(&PlayerView::new(&s, s.current), &mut rngs[cur]);
        assert!(
            legal_actions(&s).contains(&a),
            "seed {seed} {kinds:?} eps {eps}: illegal {a:?}"
        );
        apply(&mut s, &a).unwrap();
        in_turn += 1;
        if kinds[cur] != BotKind::Random {
            assert!(
                in_turn <= 200,
                "seed {seed} {kinds:?} eps {eps}: player {cur} took over 200 actions in turn {}",
                s.turn
            );
        }
        if a == Action::EndTurn {
            in_turn = 0;
        }
    }
}

#[test]
fn bare_bots_always_choose_legal_actions_and_end_their_turns() {
    for a in KINDS {
        for b in KINDS {
            for seed in 0..3 {
                check_game(seed, [a, b], 0.0, 30);
            }
        }
    }
}

#[test]
fn epsilon_bots_always_choose_legal_actions_and_end_their_turns() {
    for a in KINDS {
        for b in KINDS {
            for (seed, eps) in [(10, 0.1), (11, 0.3), (12, 0.05)] {
                check_game(seed, [a, b], eps, 25);
            }
        }
    }
}

#[test]
fn rush_and_economy_long_games_stay_legal() {
    for seed in 20..24 {
        check_game(seed, [BotKind::Rush, BotKind::Economy], 0.0, 50);
        check_game(seed, [BotKind::Economy, BotKind::Rush], 0.0, 50);
    }
}

#[test]
fn same_seed_same_game() {
    for (kinds, eps) in [
        ([BotKind::Rush, BotKind::Economy], [0.0, 0.0]),
        ([BotKind::Economy, BotKind::Random], [0.2, 0.0]),
        ([BotKind::Rush, BotKind::Rush], [0.1, 0.4]),
    ] {
        let a = my_play_game(5, kinds, eps, &MapParams::default(), &cfg(20));
        let b = my_play_game(5, kinds, eps, &MapParams::default(), &cfg(20));
        assert_eq!(a.actions, b.actions, "{kinds:?}");
        assert_eq!(a.state, b.state, "{kinds:?}");
    }
}

#[test]
fn bots_are_pure_functions_of_view_and_rng() {
    let mut states = random_playout_states(3, &cfg(20), 99, 7);
    states.extend(bot_playout_states(
        4,
        [BotKind::Rush, BotKind::Economy],
        0.2,
        &cfg(20),
        5,
    ));
    for s in &states {
        for k in KINDS {
            for eps in [0.0, 0.25] {
                let bot = make_bot(k, eps);
                let view = PlayerView::new(s, s.current);
                let (mut r1, mut r2) = (Rng::new(s.turn as u64), Rng::new(s.turn as u64));
                let a1 = bot.choose(&view, &mut r1);
                let a2 = bot.choose(&view, &mut r2);
                assert_eq!(a1, a2, "{k:?} eps {eps}");
                assert_eq!(r1, r2, "{k:?} eps {eps}: same draws");
            }
        }
    }
}

#[test]
fn rush_and_economy_draw_nothing_from_the_rng() {
    // Their rules have no randomness, and with epsilon 0 no wrapper draw is made.
    let mut states = random_playout_states(6, &cfg(15), 7, 5);
    states.extend(bot_playout_states(
        8,
        [BotKind::Economy, BotKind::Rush],
        0.0,
        &cfg(15),
        5,
    ));
    for s in &states {
        for k in [BotKind::Rush, BotKind::Economy] {
            let mut rng = Rng::new(1234);
            make_bot(k, 0.0).choose(&PlayerView::new(s, s.current), &mut rng);
            assert_eq!(rng, Rng::new(1234), "{k:?} drew from the rng");
        }
    }
}

#[test]
fn random_bot_replicates_the_random_policy_on_the_view() {
    let mut states = random_playout_states(9, &cfg(20), 5, 3);
    states.extend(bot_playout_states(
        10,
        [BotKind::Rush, BotKind::Rush],
        0.0,
        &cfg(20),
        4,
    ));
    for (i, s) in states.iter().enumerate() {
        let legal = legal_actions(s);
        let mut mine = Rng::new(i as u64);
        let want = my_random_from(&legal, &mut mine);
        let mut theirs = Rng::new(i as u64);
        let got = RandomBot.choose(&PlayerView::new(s, s.current), &mut theirs);
        assert_eq!(got, want, "state {i}");
        assert_eq!(theirs, mine, "state {i}: draw order");
        let mut bare = Rng::new(i as u64);
        assert_eq!(
            make_bot(BotKind::Random, 0.0).choose(&PlayerView::new(s, s.current), &mut bare),
            want
        );
        assert_eq!(bare, mine);
    }
}

/// Rush and Economy both choose `Promote { unit: 1 }` here (rule 2); a random action is one of
/// the many legal actions.
fn promote_board() -> State {
    open(7)
        .village(p(1, 1))
        .unit(0, UnitType::Warrior, p(1, 1))
        .unit_with(0, UnitType::Warrior, p(5, 5), |u| {
            u.promotion_available = true;
            u.phase = Phase::Done;
        })
        .unit_with(1, UnitType::Warrior, p(1, 2), |u| u.health = 2)
        .build()
}

#[test]
fn epsilon_wrapper_draws_gen_f32_first() {
    let s = promote_board();
    let legal = legal_actions(&s);
    let promote = Action::Promote { unit: UnitId(1) };
    for k in [BotKind::Rush, BotKind::Economy] {
        let (mut random_branch, mut bot_branch) = (0, 0);
        for seed in 0..200u64 {
            let mut mine = Rng::new(seed);
            let u = mine.gen_f32();
            let want = if u < 0.5 {
                random_branch += 1;
                my_random_from(&legal, &mut mine)
            } else {
                bot_branch += 1;
                promote
            };
            let mut theirs = Rng::new(seed);
            let got = make_bot(k, 0.5).choose(&PlayerView::new(&s, 0), &mut theirs);
            assert_eq!(got, want, "{k:?} seed {seed}");
            assert_eq!(theirs, mine, "{k:?} seed {seed}: draw order");
        }
        assert!(random_branch > 50 && bot_branch > 50);
    }
}

#[test]
fn epsilon_one_is_always_random_and_epsilon_zero_never() {
    let s = promote_board();
    let legal = legal_actions(&s);
    for k in [BotKind::Rush, BotKind::Economy] {
        for seed in 0..50u64 {
            let mut mine = Rng::new(seed);
            mine.gen_f32();
            let want = my_random_from(&legal, &mut mine);
            let mut theirs = Rng::new(seed);
            let got = make_bot(k, 1.0).choose(&PlayerView::new(&s, 0), &mut theirs);
            assert_eq!(got, want, "{k:?} seed {seed}");
            assert_eq!(theirs, mine);

            let mut untouched = Rng::new(seed);
            let got = make_bot(k, 0.0).choose(&PlayerView::new(&s, 0), &mut untouched);
            assert_eq!(got, Action::Promote { unit: UnitId(1) });
            assert_eq!(untouched, Rng::new(seed), "epsilon 0 makes no draw");
        }
    }
}

#[test]
fn epsilon_wrapped_random_bot_draws_twice_in_order() {
    // gen_f32 for the wrapper, then the random policy from the same rng either way.
    let s = promote_board();
    let legal = legal_actions(&s);
    for seed in 0..50u64 {
        let mut mine = Rng::new(seed);
        mine.gen_f32();
        let want = my_random_from(&legal, &mut mine);
        let mut theirs = Rng::new(seed);
        let got = make_bot(BotKind::Random, 0.3).choose(&PlayerView::new(&s, 0), &mut theirs);
        assert_eq!(got, want, "seed {seed}");
        assert_eq!(theirs, mine);
    }
}

#[test]
fn bots_need_only_the_trait_object() {
    // `make_bot` returns a usable `Box<dyn Bot>` for every kind and epsilon.
    let s = promote_board();
    for k in KINDS {
        for eps in [0.0f32, 0.5, 1.0] {
            let bot: Box<dyn Bot> = make_bot(k, eps);
            let a = bot.choose(&PlayerView::new(&s, 0), &mut Rng::new(3));
            assert!(legal_actions(&s).contains(&a));
        }
    }
}
