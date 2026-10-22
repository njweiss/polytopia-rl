//! Milestone 9 "Tests" item 2: board goldens. One scripted 8x8 board exercises every row of the
//! brief's channel table; the expected tensor is written out by hand below (every nonzero entry
//! at its exact `(channel, row, col)`) and compared in full, so every unexplored tile must be all
//! zeros. Harvest/build flags are cross-checked against `sim_core::actions::legal_actions`, the
//! source the brief names. A last test compares `encode_board` with the independent replica in
//! `common` on a playout corpus.
//!
//! The board (viewer = player 0, explored rows 0..=4 x cols 0..=6 except (0,4)):
//!
//! ```text
//!       c0        c1        c2        c3        c4        c5        c6        c7
//! r0  forest    .         crop+farm .         [mtn,metal, village   P0 archer .
//!     +animal                                  P1 archer]           (vet 7/15,
//!                                                                    Attacked)
//! r1  .         P0 cap L3 fruit     .         .         .         forest    .
//!               +P0 warr
//! r2  forest    crop      fruit     .         P0 rider  .         .         .
//!     +lumber             +P1 warr            (6/10, Moved)
//!                         (vet 3/15)
//! r3  P0 def    .         .         .         .         P0 city   mtn+metal .
//!     (promo)                                           L12+super
//! r4  .         .         P1 cap    .         forest    .         mtn+metal .
//!                         L10+rider                               +mine
//! rows 5..7 and col 7: unexplored (a forest+animal, a mountain, a P1 city with a defender).
//! ```

mod common;

use common::*;
use sim_core::actions::{legal_actions, Action};
use sim_core::data::{SiteAction, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::{Resource, Terrain};
use sim_core::state::{Config, Improvement, Phase, State};
use sim_core::testing::StateBuilder;
use sim_core::view::PlayerView;
use sim_env::obs::{encode_board, obs_spec};

const N: u8 = 8;

fn explored_p0() -> Vec<Pos> {
    (0..=4u8)
        .flat_map(|r| (0..=6u8).map(move |c| p(r, c)))
        .filter(|&q| q != p(0, 4))
        .collect()
}

fn board(techs: &[Tech]) -> StateBuilder {
    StateBuilder::new(N)
        .turn(7)
        .stars(0, 10)
        .techs(0, techs)
        .techs(1, &[Tech::Organization])
        .explore(0, &explored_p0())
        // Cities in id order: c0, c1 (player 0), c2, c3 (player 1).
        .city_with(0, p(1, 1), |c| {
            c.level = 3;
            c.population = 5;
            c.is_capital = true;
        })
        .city_with(0, p(3, 5), |c| c.level = 12)
        .city_with(1, p(4, 2), |c| {
            c.level = 10;
            c.is_capital = true;
        })
        .city_with(1, p(6, 6), |c| c.level = 2)
        .terrain(p(0, 0), Terrain::Forest)
        .resource(p(0, 0), Resource::Animal)
        .terrain(p(2, 0), Terrain::Forest)
        .improvement(p(2, 0), Improvement::LumberHut)
        .resource(p(0, 2), Resource::Crop)
        .improvement(p(0, 2), Improvement::Farm)
        .resource(p(2, 1), Resource::Crop)
        .resource(p(1, 2), Resource::Fruit)
        .resource(p(2, 2), Resource::Fruit)
        .terrain(p(4, 6), Terrain::Mountain)
        .resource(p(4, 6), Resource::Metal)
        .improvement(p(4, 6), Improvement::Mine)
        .terrain(p(3, 6), Terrain::Mountain)
        .resource(p(3, 6), Resource::Metal)
        .village(p(0, 5))
        .terrain(p(1, 6), Terrain::Forest)
        .terrain(p(4, 4), Terrain::Forest)
        // Hidden from player 0.
        .terrain(p(0, 4), Terrain::Mountain)
        .resource(p(0, 4), Resource::Metal)
        .village(p(0, 4))
        .terrain(p(5, 2), Terrain::Forest)
        .resource(p(5, 2), Resource::Animal)
        .terrain(p(7, 0), Terrain::Mountain)
        .improvement(p(6, 7), Improvement::Farm)
        // Units in id order.
        .unit(0, UnitType::Warrior, p(1, 1))
        .unit_with(0, UnitType::Rider, p(2, 4), |u| {
            u.health = 6;
            u.phase = Phase::Moved;
        })
        .unit_with(0, UnitType::Archer, p(0, 6), |u| {
            u.max_health = 15;
            u.health = 7;
            u.veteran = true;
            u.phase = Phase::Attacked;
        })
        .unit_with(0, UnitType::Defender, p(3, 0), |u| {
            u.phase = Phase::Done;
            u.kills = 3;
            u.promotion_available = true;
        })
        .unit_with(0, UnitType::SuperUnit, p(3, 5), |u| u.phase = Phase::Done)
        .unit_with(1, UnitType::Warrior, p(2, 2), |u| {
            u.max_health = 15;
            u.health = 3;
            u.veteran = true;
            u.phase = Phase::Done;
        })
        .unit(1, UnitType::Rider, p(4, 2))
        .unit_with(1, UnitType::Archer, p(0, 4), |u| u.health = 4)
        .unit_with(1, UnitType::Defender, p(6, 6), |u| u.veteran = true)
}

/// Variant A: Organization (crops shown), no Climbing (metal hidden), Hunting and Forestry.
const TECHS_A: [Tech; 3] = [Tech::Organization, Tech::Hunting, Tech::Forestry];
/// Variant B: Climbing (metal shown), no Organization (crops hidden), Hunting and Mining.
const TECHS_B: [Tech; 3] = [Tech::Climbing, Tech::Hunting, Tech::Mining];

/// Everything that does not depend on the techs or on legal actions, written out by hand.
fn expected_common() -> Vec<f32> {
    let mut e = vec![0.0f32; ch::COUNT * 64];
    let mut set = |c: usize, q: Pos, x: f32| e[bi(N, c, q)] = x;
    for q in explored_p0() {
        set(ch::EXPLORED, q, 1.0);
        let forest = [p(0, 0), p(2, 0), p(1, 6), p(4, 4)];
        let mountain = [p(3, 6), p(4, 6)];
        let t = if forest.contains(&q) {
            1
        } else if mountain.contains(&q) {
            2
        } else {
            0
        };
        set(ch::TERRAIN + t, q, 1.0);
    }
    // Resources that need no tech.
    set(ch::RESOURCE + 2, p(0, 0), 1.0); // animal
    set(ch::RESOURCE, p(1, 2), 1.0); // fruit
    set(ch::RESOURCE, p(2, 2), 1.0); // fruit
    set(ch::IMPROVEMENT, p(2, 0), 1.0); // lumber_hut
    set(ch::IMPROVEMENT + 1, p(0, 2), 1.0); // farm
    set(ch::IMPROVEMENT + 2, p(4, 6), 1.0); // mine
    set(ch::VILLAGE, p(0, 5), 1.0);
    // Territory: c0 rows 0-2 x cols 0-2, c1 rows 2-4 x cols 4-6 (mine); c2 rows 3-5 x cols
    // 1-3, of which rows 3-4 are explored (enemy).
    for r in 0..=2 {
        for c in 0..=2 {
            set(ch::TERRITORY_MINE, p(r, c), 1.0);
        }
    }
    for r in 2..=4 {
        for c in 4..=6 {
            set(ch::TERRITORY_MINE, p(r, c), 1.0);
        }
    }
    for r in 3..=4 {
        for c in 1..=3 {
            set(ch::TERRITORY_ENEMY, p(r, c), 1.0);
        }
    }
    set(ch::CITY_MINE, p(1, 1), 1.0);
    set(ch::CITY_LEVEL, p(1, 1), 0.3);
    set(ch::CITY_CAPITAL, p(1, 1), 1.0);
    set(ch::CITY_MINE, p(3, 5), 1.0);
    set(ch::CITY_LEVEL, p(3, 5), 1.0); // level 12, clipped at 10
    set(ch::CITY_ENEMY, p(4, 2), 1.0);
    set(ch::CITY_LEVEL, p(4, 2), 1.0); // level 10, exactly the cap
    set(ch::CITY_CAPITAL, p(4, 2), 1.0);
    // Own units: type, health fraction, phase.
    set(ch::MY_UNIT, p(1, 1), 1.0); // warrior
    set(ch::HEALTH, p(1, 1), 1.0);
    set(ch::PHASE, p(1, 1), 1.0); // fresh
    set(ch::MY_UNIT + 1, p(2, 4), 1.0); // rider
    set(ch::HEALTH, p(2, 4), 0.6);
    set(ch::PHASE + 1, p(2, 4), 1.0); // moved
    set(ch::MY_UNIT + 2, p(0, 6), 1.0); // archer
    set(ch::HEALTH, p(0, 6), 7.0 / 15.0);
    set(ch::PHASE + 2, p(0, 6), 1.0); // attacked
    set(ch::VETERAN, p(0, 6), 1.0);
    set(ch::MY_UNIT + 3, p(3, 0), 1.0); // defender
    set(ch::HEALTH, p(3, 0), 1.0);
    set(ch::PHASE + 3, p(3, 0), 1.0); // done
    set(ch::MY_UNIT + 4, p(3, 5), 1.0); // super_unit
    set(ch::HEALTH, p(3, 5), 1.0);
    set(ch::PHASE + 3, p(3, 5), 1.0); // done
                                      // Enemy units: type, health, veteran; no phase.
    set(ch::ENEMY_UNIT, p(2, 2), 1.0); // warrior
    set(ch::HEALTH, p(2, 2), 0.2); // 3 / 15
    set(ch::VETERAN, p(2, 2), 1.0);
    set(ch::ENEMY_UNIT + 1, p(4, 2), 1.0); // rider
    set(ch::HEALTH, p(4, 2), 1.0);
    e
}

fn expected_a() -> Vec<f32> {
    let mut e = expected_common();
    // Organization: crops shown. No Climbing: metal at (3,6) and (4,6) hidden.
    e[bi(N, ch::RESOURCE + 1, p(0, 2))] = 1.0;
    e[bi(N, ch::RESOURCE + 1, p(2, 1))] = 1.0;
    e[bi(N, ch::CAN_PROMOTE, p(3, 0))] = 1.0;
    // Harvest animal (0,0), harvest fruit (1,2); (2,2) has an enemy unit on it.
    e[bi(N, ch::CAN_HARVEST, p(0, 0))] = 1.0;
    e[bi(N, ch::CAN_HARVEST, p(1, 2))] = 1.0;
    // Lumber hut / clear forest on unimproved own forests (0,0) and (4,4).
    e[bi(N, ch::CAN_BUILD, p(0, 0))] = 1.0;
    e[bi(N, ch::CAN_BUILD, p(4, 4))] = 1.0;
    e
}

fn expected_b() -> Vec<f32> {
    let mut e = expected_common();
    // Climbing: metal shown. No Organization: crops hidden.
    e[bi(N, ch::RESOURCE + 3, p(3, 6))] = 1.0;
    e[bi(N, ch::RESOURCE + 3, p(4, 6))] = 1.0;
    e[bi(N, ch::CAN_PROMOTE, p(3, 0))] = 1.0;
    // Harvest animal (0,0) only (fruit needs Organization; no Forestry).
    e[bi(N, ch::CAN_HARVEST, p(0, 0))] = 1.0;
    // Mine on the unimproved metal at (3,6).
    e[bi(N, ch::CAN_BUILD, p(3, 6))] = 1.0;
    e
}

fn sites(s: &State) -> Vec<(Pos, SiteAction)> {
    legal_actions(s)
        .into_iter()
        .filter_map(|a| match a {
            Action::Site { at, action } => Some((at, action)),
            _ => None,
        })
        .collect()
}

fn encode(s: &State, v: u8) -> Vec<f32> {
    let spec = obs_spec(s.size);
    let mut out = vec![0.0f32; spec.channels * spec.height * spec.width];
    encode_board(&PlayerView::new(s, v), &mut out);
    out
}

/// Names the first mismatch by `(channel name, row, col)`.
fn assert_board(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "board length");
    let names = obs_spec(N).channel_names;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let (c, r, col) = (i / 64, (i / 8) % 8, i % 8);
        assert!(
            approx(*a, *e),
            "({}, row {r}, col {col}) = ({c}, {r}, {col}): got {a}, expected {e}",
            names.get(c).map_or("?", |s| s.as_str())
        );
    }
}

/// Checks the scripted board against `legal_actions` (a check on the test, not the code).
fn assert_board_premises() {
    let a = board(&TECHS_A).build();
    use SiteAction::*;
    assert_eq!(
        sites(&a),
        vec![
            (p(0, 0), HarvestAnimal),
            (p(0, 0), LumberHut),
            (p(0, 0), ClearForest),
            (p(1, 2), HarvestFruit),
            (p(4, 4), LumberHut),
            (p(4, 4), ClearForest),
        ]
    );
    let b = board(&TECHS_B).build();
    assert_eq!(sites(&b), vec![(p(0, 0), HarvestAnimal), (p(3, 6), Mine)]);
    for s in [&a, &b] {
        let promotes: Vec<Action> = legal_actions(s)
            .into_iter()
            .filter(|a| matches!(a, Action::Promote { .. }))
            .collect();
        assert_eq!(promotes.len(), 1);
        assert_eq!(
            s.unit(match promotes[0] {
                Action::Promote { unit } => unit,
                _ => unreachable!(),
            })
            .unwrap()
            .pos,
            p(3, 0)
        );
    }
}

#[test]
fn golden_variant_a_crops_shown_metal_hidden() {
    assert_board_premises();
    let s = board(&TECHS_A).build();
    assert_board(&encode(&s, 0), &expected_a());
}

#[test]
fn golden_variant_b_metal_shown_crops_hidden() {
    assert_board_premises();
    let s = board(&TECHS_B).build();
    assert_board(&encode(&s, 0), &expected_b());
}

#[test]
fn spot_checks_at_exact_flat_indexes() {
    // (C, H, W): flat = (c * 8 + row) * 8 + col.
    let e = encode(&board(&TECHS_A).build(), 0);
    assert_eq!(e[(14 * 8 + 1) * 8 + 1], 1.0, "city.mine at (1,1)");
    assert_eq!(e[(17 * 8 + 4) * 8 + 2], 1.0, "city.capital at (4,2)");
    assert_eq!(e[(17 * 8 + 2) * 8 + 4], 0.0, "transposed city.capital");
    assert_eq!(e[(19 * 8 + 2) * 8 + 4], 1.0, "my_unit.rider at (2,4)");
    assert_eq!(e[(19 * 8 + 4) * 8 + 2], 0.0, "my_unit.rider transposed");
    assert_eq!(e[(36 * 8 + 4) * 8 + 4], 1.0, "can_build at (4,4)");
    assert!(approx(e[(28 * 8) * 8 + 6], 7.0 / 15.0), "health at (0,6)");
    assert_eq!(e[(2 * 8) * 8 + 4], 0.0, "unexplored (0,4) has no terrain");
    assert_eq!(e[(3 * 8) * 8 + 4], 0.0, "unexplored (0,4) has no terrain");
}

#[test]
fn unexplored_tiles_are_all_zero_including_explored() {
    let s = board(&TECHS_B).build();
    let e = encode(&s, 0);
    for q in all_pos(N) {
        if s.is_explored(0, q) {
            continue;
        }
        for c in 0..ch::COUNT {
            assert_eq!(e[bi(N, c, q)], 0.0, "unexplored {q:?} channel {c}");
        }
    }
}

#[test]
fn legal_flags_are_zero_when_it_is_not_the_viewers_turn() {
    // Same board with player 1 to move: the viewer has no legal actions.
    let s = board(&TECHS_A).current(1).build();
    assert_board(&encode(&s, 0), &expected_common_with_crops());
}

#[test]
fn legal_flags_are_zero_with_a_pending_reward() {
    // A pending reward makes ChooseReward the only legal action (spec "Forced rewards").
    let s = board(&TECHS_A).build();
    let mut s2 = s.clone();
    s2.cities[0].pending_reward = true;
    assert!(legal_actions(&s2)
        .iter()
        .all(|a| matches!(a, Action::ChooseReward { .. })));
    assert_board(&encode(&s2, 0), &expected_common_with_crops());
}

fn expected_common_with_crops() -> Vec<f32> {
    let mut e = expected_common();
    e[bi(N, ch::RESOURCE + 1, p(0, 2))] = 1.0;
    e[bi(N, ch::RESOURCE + 1, p(2, 1))] = 1.0;
    e
}

#[test]
fn the_other_seat_sees_itself_as_mine() {
    // Player 1 has explored everything; "mine" is always the observer.
    let s = board(&TECHS_A).explore(1, &all_pos(N)).build();
    let e = encode(&s, 1);
    let at = |c: usize, q: Pos| e[bi(N, c, q)];
    assert_eq!(at(ch::CITY_MINE, p(4, 2)), 1.0);
    assert_eq!(at(ch::CITY_ENEMY, p(4, 2)), 0.0);
    assert_eq!(at(ch::CITY_ENEMY, p(1, 1)), 1.0);
    assert_eq!(at(ch::CITY_MINE, p(6, 6)), 1.0);
    assert!(approx(at(ch::CITY_LEVEL, p(6, 6)), 0.2));
    assert_eq!(at(ch::TERRITORY_MINE, p(5, 3)), 1.0);
    assert_eq!(at(ch::TERRITORY_ENEMY, p(0, 0)), 1.0);
    assert_eq!(at(ch::TERRITORY_MINE, p(0, 0)), 0.0);
    // Own units: player 1's rider (Fresh) and warrior (Done) show phases.
    assert_eq!(at(ch::MY_UNIT + 1, p(4, 2)), 1.0);
    assert_eq!(at(ch::PHASE, p(4, 2)), 1.0);
    assert_eq!(at(ch::MY_UNIT, p(2, 2)), 1.0);
    assert_eq!(at(ch::PHASE + 3, p(2, 2)), 1.0);
    assert_eq!(at(ch::MY_UNIT + 2, p(0, 4)), 1.0);
    assert!(approx(at(ch::HEALTH, p(0, 4)), 0.4));
    // Player 0's units are enemies now: type shown, phase hidden.
    assert_eq!(at(ch::ENEMY_UNIT + 1, p(2, 4)), 1.0);
    assert_eq!(at(ch::ENEMY_UNIT + 4, p(3, 5)), 1.0);
    for k in 0..4 {
        assert_eq!(at(ch::PHASE + k, p(2, 4)), 0.0, "enemy phase {k}");
        assert_eq!(at(ch::PHASE + k, p(0, 6)), 0.0, "enemy phase {k}");
    }
    assert_eq!(at(ch::VETERAN, p(0, 6)), 1.0);
    assert_eq!(at(ch::VETERAN, p(6, 6)), 1.0);
    // Player 1 has Organization but not Climbing.
    assert_eq!(at(ch::RESOURCE + 1, p(0, 2)), 1.0);
    assert_eq!(at(ch::RESOURCE + 3, p(0, 4)), 0.0);
    assert_eq!(at(ch::RESOURCE + 2, p(5, 2)), 1.0);
    // Not player 1's turn, so no legal flags.
    for q in all_pos(N) {
        for c in [ch::CAN_PROMOTE, ch::CAN_HARVEST, ch::CAN_BUILD] {
            assert_eq!(at(c, q), 0.0);
        }
    }
}

#[test]
fn full_observability_uses_the_same_encoding_with_everything_explored() {
    let config = Config {
        full_observability: true,
        ..Config::default()
    };
    let s = board(&TECHS_B).config(config).build();
    let e = encode(&s, 0);
    let at = |c: usize, q: Pos| e[bi(N, c, q)];
    for q in all_pos(N) {
        assert_eq!(at(ch::EXPLORED, q), 1.0, "explored {q:?}");
        let terrain: f32 = (0..3).map(|k| at(ch::TERRAIN + k, q)).sum();
        assert_eq!(terrain, 1.0, "terrain one-hot at {q:?}");
    }
    assert_eq!(at(ch::TERRAIN + 2, p(0, 4)), 1.0);
    assert_eq!(at(ch::RESOURCE + 3, p(0, 4)), 1.0); // Climbing in variant B
    assert_eq!(at(ch::VILLAGE, p(0, 4)), 1.0);
    assert_eq!(at(ch::ENEMY_UNIT + 2, p(0, 4)), 1.0);
    assert!(approx(at(ch::HEALTH, p(0, 4)), 0.4));
    assert_eq!(at(ch::ENEMY_UNIT + 3, p(6, 6)), 1.0);
    assert_eq!(at(ch::CITY_ENEMY, p(6, 6)), 1.0);
    assert!(approx(at(ch::CITY_LEVEL, p(6, 6)), 0.2));
    assert_eq!(at(ch::CITY_CAPITAL, p(6, 6)), 0.0);
    assert_eq!(at(ch::TERRITORY_ENEMY, p(7, 7)), 1.0);
    assert_eq!(at(ch::IMPROVEMENT + 1, p(6, 7)), 1.0);
    assert_eq!(at(ch::RESOURCE + 2, p(5, 2)), 1.0);
    assert_eq!(at(ch::TERRAIN + 1, p(5, 2)), 1.0);
    assert_eq!(at(ch::TERRAIN + 2, p(7, 0)), 1.0);
    // Everything that was already explored is unchanged.
    let fog = encode(&board(&TECHS_B).build(), 0);
    for q in explored_p0() {
        for c in 0..ch::COUNT {
            assert_eq!(at(c, q), fog[bi(N, c, q)], "explored {q:?} channel {c}");
        }
    }
}

#[test]
fn hidden_crop_and_metal_follow_organization_and_climbing() {
    // A 6x6 board, everything explored, a crop and a metal in plain sight.
    let make = |techs: &[Tech]| {
        StateBuilder::new(6)
            .explore_all()
            .techs(0, techs)
            .resource(p(1, 1), Resource::Crop)
            .terrain(p(4, 4), Terrain::Mountain)
            .resource(p(4, 4), Resource::Metal)
            .build()
    };
    for (techs, crop, metal) in [
        (vec![], 0.0, 0.0),
        (vec![Tech::Organization], 1.0, 0.0),
        (vec![Tech::Climbing], 0.0, 1.0),
        (vec![Tech::Organization, Tech::Climbing], 1.0, 1.0),
    ] {
        let s = make(&techs);
        let mut e = vec![0.0f32; ch::COUNT * 36];
        encode_board(&PlayerView::new(&s, 0), &mut e);
        assert_eq!(
            e[bi(6, ch::RESOURCE + 1, p(1, 1))],
            crop,
            "crop with {techs:?}"
        );
        assert_eq!(
            e[bi(6, ch::RESOURCE + 3, p(4, 4))],
            metal,
            "metal with {techs:?}"
        );
        // The terrain stays visible either way.
        assert_eq!(e[bi(6, ch::TERRAIN + 2, p(4, 4))], 1.0);
    }
}

#[test]
fn matches_the_independent_replica_on_playout_states() {
    let mut n = 0;
    for s in corpus(0..8, 7) {
        for v in [0u8, 1] {
            let got = encode(&s, v);
            let want = my_encode_board(&s, v);
            assert_slices_approx(&got, &want, &format!("turn {} viewer {v}", s.turn));
            n += 1;
        }
    }
    assert!(n > 500, "corpus too small: {n}");
}
