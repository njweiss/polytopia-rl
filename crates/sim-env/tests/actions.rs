//! Milestone 9 "Tests" item 5: golden `Factored` values for each of the 10 kinds (the global
//! slot and the last tile included), `num_actors` / `arg_len`, the encode/decode round trip on
//! every legal action of thousands of playout states, and each decode failure case. Goldens are
//! hand-computed from "Actor index": unit `tile`, city `N + tile`, site `2N + tile`, global `3N`,
//! with `tile = row * size + col`.

mod common;

use common::*;
use sim_core::actions::{legal_actions, Action};
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::state::{CityId, State, UnitId};
use sim_core::testing::StateBuilder;
use sim_env::actions::{
    arg_len, decode_action, encode_action, num_actors, ActionType, DecodeError, Factored,
    NUM_ACTION_TYPES,
};

fn f(actor: usize, kind: ActionType, arg: usize) -> Factored {
    Factored { actor, kind, arg }
}

#[test]
fn num_actors_is_three_per_tile_plus_one() {
    assert_eq!(num_actors(11), 364);
    assert_eq!(num_actors(8), 193);
    assert_eq!(num_actors(6), 109);
}

#[test]
fn action_types_and_arg_lens() {
    assert_eq!(NUM_ACTION_TYPES, 10);
    assert_eq!(ActionType::ALL, KIND_ORDER);
    for (i, k) in ActionType::ALL.iter().enumerate() {
        assert_eq!(*k as u8 as usize, i, "{k:?}");
    }
    use ActionType::*;
    for (size, n) in [(11u8, 121usize), (8, 64)] {
        assert_eq!(arg_len(size, Move), n);
        assert_eq!(arg_len(size, Attack), n);
        assert_eq!(arg_len(size, Train), 5);
        assert_eq!(arg_len(size, ChooseReward), 8);
        assert_eq!(arg_len(size, Site), 6);
        assert_eq!(arg_len(size, Research), 9);
        for k in [Capture, Recover, Promote, EndTurn] {
            assert_eq!(arg_len(size, k), 0, "{k:?}");
        }
    }
}

/// 8x8 (N = 64). Player 0: Warrior u0 at (2,3), Rider u1 at (5,1) on a village, Archer u2 at
/// (7,7) (last tile) with a promotion, city c0 at (0,1), city c1 at (7,6). Player 1: Warrior u3 at
/// (3,4), city c2 at (5,6).
fn board8() -> State {
    StateBuilder::new(8)
        .explore_all()
        .stars(0, 20)
        .city(0, p(0, 1))
        .city(0, p(7, 6))
        .city(1, p(5, 6))
        .village(p(5, 1))
        .unit(0, UnitType::Warrior, p(2, 3))
        .unit(0, UnitType::Rider, p(5, 1))
        .unit_with(0, UnitType::Archer, p(7, 7), |u| {
            u.promotion_available = true
        })
        .unit(1, UnitType::Warrior, p(3, 4))
        .build()
}

#[test]
fn golden_factored_values_on_an_8x8_board() {
    let s = board8();
    let (u0, u1, u2) = (UnitId(0), UnitId(1), UnitId(2));
    let cases = [
        // Move: unit actor = tile of (2,3) = 19; destination (1,3) = 11.
        (
            Action::Move {
                unit: u0,
                to: p(1, 3),
            },
            f(19, ActionType::Move, 11),
        ),
        // Attack (3,4) = 28.
        (
            Action::Attack {
                unit: u0,
                target: p(3, 4),
            },
            f(19, ActionType::Attack, 28),
        ),
        // Capture by the Rider at (5,1) = 41.
        (Action::Capture { unit: u1 }, f(41, ActionType::Capture, 0)),
        (Action::Recover { unit: u0 }, f(19, ActionType::Recover, 0)),
        // Promote the Archer on the last tile (7,7) = 63.
        (Action::Promote { unit: u2 }, f(63, ActionType::Promote, 0)),
        // Train at c1 (7,6) = 62: city actor 64 + 62 = 126; Archer is UnitType index 2.
        (
            Action::Train {
                city: CityId(1),
                kind: UnitType::Archer,
            },
            f(126, ActionType::Train, 2),
        ),
        // Train at c0 (0,1) = 1: actor 65; Warrior index 0.
        (
            Action::Train {
                city: CityId(0),
                kind: UnitType::Warrior,
            },
            f(65, ActionType::Train, 0),
        ),
        // Site at (7,7): 128 + 63 = 191 (the last site slot); Mine is SiteAction index 5.
        (
            Action::Site {
                at: p(7, 7),
                action: SiteAction::Mine,
            },
            f(191, ActionType::Site, 5),
        ),
        // Site at (0,0): 128; HarvestAnimal index 0.
        (
            Action::Site {
                at: p(0, 0),
                action: SiteAction::HarvestAnimal,
            },
            f(128, ActionType::Site, 0),
        ),
        // Site at (2,3) = 19: 147; ClearForest index 3.
        (
            Action::Site {
                at: p(2, 3),
                action: SiteAction::ClearForest,
            },
            f(147, ActionType::Site, 3),
        ),
        // Research: global actor 3 x 64 = 192; Mining is Tech index 8, Riding 3.
        (
            Action::Research { tech: Tech::Mining },
            f(192, ActionType::Research, 8),
        ),
        (
            Action::Research { tech: Tech::Riding },
            f(192, ActionType::Research, 3),
        ),
        (Action::EndTurn, f(192, ActionType::EndTurn, 0)),
    ];
    for (action, want) in cases {
        assert_eq!(encode_action(&s, &action), want, "encode {action:?}");
        assert_eq!(decode_action(&s, &want), Ok(action), "decode {want:?}");
    }
}

#[test]
fn golden_choose_reward() {
    // A pending reward at a level-3 city on (6,2) = 50 of an 8x8 board: actor 64 + 50 = 114.
    let s = StateBuilder::new(8)
        .explore_all()
        .city_with(0, p(6, 2), |c| {
            c.level = 3;
            c.population = 5;
            c.pending_reward = true;
        })
        .build();
    assert_eq!(
        legal_actions(&s),
        vec![
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::CityWall
            },
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::Resources
            }
        ],
        "test board"
    );
    // Reward indices: Workshop 0, Explorer 1, CityWall 2, Resources 3, ... SuperUnit 7.
    for (reward, arg) in [(Reward::CityWall, 2), (Reward::Resources, 3)] {
        let a = Action::ChooseReward {
            city: CityId(0),
            reward,
        };
        let want = f(114, ActionType::ChooseReward, arg);
        assert_eq!(encode_action(&s, &a), want);
        assert_eq!(decode_action(&s, &want), Ok(a));
    }
    // Every reward index decodes (identification only, not legality).
    let a = decode_action(&s, &f(114, ActionType::ChooseReward, 7));
    assert_eq!(
        a,
        Ok(Action::ChooseReward {
            city: CityId(0),
            reward: Reward::SuperUnit
        })
    );
}

#[test]
fn golden_values_on_the_11x11_map() {
    // N = 121: last tile (10,10) = 120; city actor 241, site actor 362, global 363.
    let s = StateBuilder::new(11)
        .explore_all()
        .city(0, p(10, 10))
        .unit(0, UnitType::Warrior, p(10, 9))
        .unit(1, UnitType::Warrior, p(0, 0))
        .build();
    let u = UnitId(0);
    let cases = [
        (
            Action::Move {
                unit: u,
                to: p(9, 10),
            },
            f(119, ActionType::Move, 109),
        ),
        (
            Action::Train {
                city: CityId(0),
                kind: UnitType::Defender,
            },
            f(241, ActionType::Train, 3),
        ),
        (
            Action::Site {
                at: p(10, 10),
                action: SiteAction::HarvestFruit,
            },
            f(362, ActionType::Site, 1),
        ),
        (
            Action::Research {
                tech: Tech::Climbing,
            },
            f(363, ActionType::Research, 0),
        ),
        (Action::EndTurn, f(363, ActionType::EndTurn, 0)),
    ];
    for (action, want) in cases {
        assert_eq!(encode_action(&s, &action), want, "encode {action:?}");
        assert_eq!(decode_action(&s, &want), Ok(action), "decode {want:?}");
    }
}

#[test]
fn decode_errors() {
    let s = board8(); // player 0 to move
    use ActionType::*;
    use DecodeError::*;
    let cases = [
        // Actor past the global slot.
        (f(193, EndTurn, 0), ActorOutOfRange),
        (f(10_000, Move, 0), ActorOutOfRange),
        // Unit actor slot with no unit at (0,0).
        (f(0, Move, 1), NoActor),
        (f(0, Recover, 0), NoActor),
        // Unit actor slot holding an enemy unit (3,4) = 28.
        (f(28, Move, 27), NoActor),
        (f(28, Promote, 0), NoActor),
        // City actor slot with no city: (0,0) -> 64.
        (f(64, Train, 0), NoActor),
        // City actor slot holding an enemy city (5,6) = 46 -> 110.
        (f(110, Train, 0), NoActor),
        (f(110, ChooseReward, 0), NoActor),
        // Kind not of the actor's family.
        (f(19, Train, 0), WrongKind),
        (f(19, Site, 0), WrongKind),
        (f(19, EndTurn, 0), WrongKind),
        (f(126, Move, 0), WrongKind),
        (f(126, Research, 0), WrongKind),
        (f(147, Move, 0), WrongKind),
        (f(147, Train, 0), WrongKind),
        (f(192, Move, 0), WrongKind),
        (f(192, Site, 0), WrongKind),
        (f(192, Train, 0), WrongKind),
        // Argument out of range, or nonzero for a kind without one.
        (f(19, Move, 64), BadArg),
        (f(19, Attack, 64), BadArg),
        (f(126, Train, 5), BadArg),
        (f(126, ChooseReward, 8), BadArg),
        (f(147, Site, 6), BadArg),
        (f(192, Research, 9), BadArg),
        (f(19, Capture, 1), BadArg),
        (f(19, Recover, 1), BadArg),
        (f(63, Promote, 2), BadArg),
        (f(192, EndTurn, 1), BadArg),
    ];
    for (fac, want) in cases {
        assert_eq!(decode_action(&s, &fac), Err(want), "{fac:?}");
    }
    // Identification only: decodable but illegal actions are not errors.
    assert_eq!(
        decode_action(&s, &f(19, Move, 63)),
        Ok(Action::Move {
            unit: UnitId(0),
            to: p(7, 7)
        })
    );
    assert_eq!(
        decode_action(&s, &f(192, Research, 8)),
        Ok(Action::Research { tech: Tech::Mining })
    );
}

#[test]
fn ownership_follows_the_player_to_move() {
    // With player 1 to move, its unit and city decode and player 0's do not.
    let mut s = board8();
    s.current = 1;
    assert_eq!(
        decode_action(&s, &f(28, ActionType::Recover, 0)),
        Ok(Action::Recover { unit: UnitId(3) })
    );
    assert_eq!(
        decode_action(&s, &f(110, ActionType::Train, 0)),
        Ok(Action::Train {
            city: CityId(2),
            kind: UnitType::Warrior
        })
    );
    assert_eq!(
        decode_action(&s, &f(19, ActionType::Recover, 0)),
        Err(DecodeError::NoActor)
    );
    assert_eq!(
        decode_action(&s, &f(126, ActionType::Train, 0)),
        Err(DecodeError::NoActor)
    );
}

#[test]
fn round_trip_on_every_legal_action_of_playout_states() {
    let mut states = 0;
    let mut actions = 0;
    for s in corpus(0..40, 2) {
        for a in legal_actions(&s) {
            let fac = encode_action(&s, &a);
            let (actor, kind, arg) = my_encode(&s, &a);
            assert_eq!(fac, f(actor, kind, arg), "encode {a:?}");
            assert_eq!(decode_action(&s, &fac), Ok(a), "decode {fac:?}");
            actions += 1;
        }
        states += 1;
    }
    assert!(states > 2000, "too few states: {states}");
    assert!(actions > 50_000, "too few actions: {actions}");
}
