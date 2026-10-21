//! Milestone 6: the Capture action for villages and enemy cities, its legality, and the win by
//! capital capture (docs/MILESTONE_6 "Capture", F1 to F4, "Tests" items 2 to 5; spec "Ownership
//! rules", "Capture notes"). Every expected state is built by hand from the brief on a drawn
//! board; combat numbers are hand arithmetic with the spec's formula (f32, half-up rounding).

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Reward, Tech, UnitType};
use sim_core::economy::supported_units;
use sim_core::grid::Pos;
use sim_core::score::score_breakdown;
use sim_core::state::{City, CityId, Dir, Outcome, Phase, State, UnitId};
use sim_core::testing::StateBuilder;
use UnitType::*;

// ---------- helpers ----------

const S: Dir = Dir { dr: 1, dc: 0 };
const SE: Dir = Dir { dr: 1, dc: 1 };

fn capture(unit: u32) -> Action {
    Action::Capture { unit: UnitId(unit) }
}

fn ok(s: &mut State, a: Action) {
    assert!(
        legal_actions(s).contains(&a),
        "{a:?} should be legal; legal = {:?}",
        legal_actions(s)
    );
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

fn illegal(s: &mut State, a: Action) {
    assert!(!legal_actions(s).contains(&a), "{a:?} should not be legal");
    let before = s.clone();
    assert_eq!(apply(s, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(*s, before, "illegal {a:?} changed the state");
}

fn unit_mut(s: &mut State, id: u32) -> &mut sim_core::state::Unit {
    s.units.iter_mut().find(|u| u.id == UnitId(id)).unwrap()
}

fn city_mut(s: &mut State, id: u32) -> &mut City {
    s.cities.iter_mut().find(|c| c.id == CityId(id)).unwrap()
}

fn explore(s: &mut State, player: u8, tiles: &[Pos]) {
    for &q in tiles {
        let i = idx(s, q);
        s.players[player as usize].explored[i] = true;
    }
}

/// The city the brief's village capture founds.
fn new_city(id: u32, owner: u8, pos: Pos) -> City {
    City {
        id: CityId(id),
        owner,
        pos,
        level: 1,
        population: 0,
        is_capital: false,
        workshop: false,
        parks: 0,
        wall: false,
        border_growth_used: false,
        pending_reward: false,
    }
}

// ---------- village capture ----------

/// ```text
///      0 1 2 3 4 5 6
///   0  a a a . 1 . v      A = city 0 (player 0's capital), a = its territory
///   1  a A a . . . .      B = city 1 (player 1's capital), b = its territory
///   2  a a a . . . .      v = village; unit 0 (player 0 Warrior, origin city 0, last moved S)
///   3  0 . . . . . .          stands on (3,0), unit 3 (player 0 Warrior, no origin) on (0,6)
///   4  . . . . b b b      1 = unit 1, player 0 Warrior with origin city 0
///   5  . . . . b B b      unit 2: player 1 Warrior on B, origin city 1
///   6  . . . . b b b
/// ```
/// Player 0 has 6 stars and Organization, and has explored only (2,0) and (3,0).
fn village_board() -> State {
    StateBuilder::new(7)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .village(p(3, 0))
        .village(p(0, 6))
        .unit_with(0, Warrior, p(3, 0), |u| {
            u.origin = Some(CityId(0));
            u.last_dir = Some(S);
        })
        .unit_with(0, Warrior, p(0, 4), |u| u.origin = Some(CityId(0)))
        .unit_with(1, Warrior, p(5, 5), |u| u.origin = Some(CityId(1)))
        .unit(0, Warrior, p(0, 6))
        .techs(0, &[Tech::Organization])
        .stars(0, 6)
        .explore(0, &[p(2, 0), p(3, 0)])
        .build()
}

/// The village tile and its unclaimed in-bounds neighbors: (2,0) and (2,1) already belong to
/// city 0, and the column left of (3,0) is off the map.
#[test]
fn capturing_a_village_founds_a_level_1_city() {
    let before = village_board();
    let mut s = before.clone();
    ok(&mut s, capture(0));

    let mut want = before.clone();
    want.cities.push(new_city(2, 0, p(3, 0)));
    let t = tile_mut(&mut want, p(3, 0));
    (t.village, t.city) = (false, Some(CityId(2)));
    let claimed = [p(3, 0), p(3, 1), p(4, 0), p(4, 1)];
    for q in claimed {
        tile_mut(&mut want, q).owner_city = Some(CityId(2));
    }
    let u = unit_mut(&mut want, 0);
    (u.origin, u.phase) = (Some(CityId(2)), Phase::Done);
    explore(&mut want, 0, &claimed);
    assert_eq!(s, want);

    assert_eq!(territory_of(&s, CityId(2)), claimed);
    assert_eq!(s.tile(p(2, 0)).owner_city, Some(CityId(0)));
    assert_eq!(s.tile(p(2, 1)).owner_city, Some(CityId(0)));
    assert_eq!(
        s.unit(UnitId(0)).unwrap().last_dir,
        Some(S),
        "last_dir kept"
    );
    assert_eq!(s.outcome, None);
}

#[test]
fn the_capturer_migrates_its_origin() {
    let mut s = village_board();
    assert_eq!(supported_units(&s, CityId(0)), 2);
    ok(&mut s, capture(0));
    assert_eq!(supported_units(&s, CityId(0)), 1, "old city drops");
    assert_eq!(supported_units(&s, CityId(2)), 1, "new city rises");
    // A capturer without an origin gains one.
    ok(&mut s, capture(3));
    assert_eq!(s.unit(UnitId(3)).unwrap().origin, Some(CityId(3)));
    assert_eq!(supported_units(&s, CityId(3)), 1);
}

#[test]
fn a_second_village_capture_gets_the_next_id() {
    let mut s = village_board();
    ok(&mut s, capture(0));
    ok(&mut s, capture(3));
    assert_eq!(s.cities.len(), 4);
    assert_eq!(s.cities[3], new_city(3, 0, p(0, 6)));
    assert_eq!(s.tile(p(0, 6)).city, Some(CityId(3)));
    assert!(!s.tile(p(0, 6)).village);
    // (0,6) sits in the corner of the board: four tiles, all unclaimed.
    assert_eq!(
        territory_of(&s, CityId(3)),
        [p(0, 5), p(0, 6), p(1, 5), p(1, 6)]
    );
}

/// A Super Unit captures too; its origin moves but it never counts toward capacity.
#[test]
fn a_super_unit_captures_without_using_capacity() {
    let mut s = StateBuilder::new(5)
        .city(0, p(0, 0))
        .village(p(3, 3))
        .unit_with(0, SuperUnit, p(3, 3), |u| u.origin = Some(CityId(0)))
        .build();
    ok(&mut s, capture(0));
    let u = s.unit(UnitId(0)).unwrap();
    assert_eq!((u.origin, u.phase), (Some(CityId(1)), Phase::Done));
    assert_eq!(supported_units(&s, CityId(1)), 0);
    assert_eq!(supported_units(&s, CityId(0)), 0);
    assert_eq!(s.cities[1], new_city(1, 0, p(3, 3)));
}

/// Research cost is `4 + cities * tier`: Hunting costs 5 with one city and 6 with two.
#[test]
fn a_new_city_raises_research_costs() {
    let mut control = village_board();
    ok(
        &mut control,
        Action::Research {
            tech: Tech::Hunting,
        },
    );
    assert_eq!(control.players[0].stars, 1);

    let mut s = village_board();
    ok(&mut s, capture(0));
    ok(
        &mut s,
        Action::Research {
            tech: Tech::Hunting,
        },
    );
    assert_eq!(s.players[0].stars, 0);
}

// ---------- enemy city capture ----------

/// ```text
///      0 1 2 3 4 5 6 7
///   0  a a a 2 . c c c      A = city 0 (player 0's capital), a = its territory
///   1  a A a . . c C c      B = city 1 (player 1, not capital): level 3, population 6,
///   2  a a 4 . b c c c          workshop, wall, one park, border growth used; b = its
///   3  . . . 5 b b . .          territory, including (2,4) and (6,6)
///   4  . . . b 0 b . .      C = city 2 (player 1's capital), c = its territory
///   5  . . . 6 b b . .      units: 0 player 0 Warrior on B (origin A, last moved SE);
///   6  . 1 . . . . b .          1 player 1 Defender (origin B); 2 player 1 Warrior (origin C);
///   7  . . . . . . . 3          3 player 1 Super Unit (origin B); 4 player 0 Rider (origin A);
///                               5 player 0 Warrior at 3 hp; 6 player 1 Warrior at 3 hp
/// ```
/// Player 0 has explored only (4,4) and (5,5).
fn city_board() -> State {
    let mut s = StateBuilder::new(8)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| {
            (c.level, c.population, c.parks) = (3, 6, 1);
            (c.workshop, c.wall, c.border_growth_used) = (true, true, true);
        })
        .city_with(1, p(1, 6), |c| c.is_capital = true)
        .unit_with(0, Warrior, p(4, 4), |u| {
            u.origin = Some(CityId(0));
            u.last_dir = Some(SE);
        })
        .unit_with(1, Defender, p(6, 1), |u| u.origin = Some(CityId(1)))
        .unit_with(1, Warrior, p(0, 3), |u| u.origin = Some(CityId(2)))
        .unit_with(1, SuperUnit, p(7, 7), |u| u.origin = Some(CityId(1)))
        .unit_with(0, Rider, p(2, 2), |u| u.origin = Some(CityId(0)))
        .unit_with(0, Warrior, p(3, 3), |u| u.health = 3)
        .unit_with(1, Warrior, p(5, 3), |u| u.health = 3)
        .explore(0, &[p(4, 4), p(5, 5)])
        .build();
    tile_mut(&mut s, p(2, 4)).owner_city = Some(CityId(1));
    tile_mut(&mut s, p(6, 6)).owner_city = Some(CityId(1));
    s
}

/// The 11 tiles of city 1 on `city_board`, row-major.
fn city_1_tiles() -> Vec<Pos> {
    let mut t = vec![p(2, 4)];
    for r in 3..=5 {
        for c in 3..=5 {
            t.push(p(r, c));
        }
    }
    t.push(p(6, 6));
    t
}

#[test]
fn capturing_an_enemy_city_changes_its_owner_and_orphans_its_units() {
    let before = city_board();
    assert_eq!(territory_of(&before, CityId(1)), city_1_tiles());
    let mut s = before.clone();
    ok(&mut s, capture(0));

    // F2: only the owner changes; level, population, buildings and flags are kept. Tiles keep
    // their owning city. The previous owner's units of city 1 (the Defender and the Super Unit)
    // are orphaned; the Warrior of city 2 keeps its origin. Player 0 sees all of city 1's tiles.
    let mut want = before.clone();
    city_mut(&mut want, 1).owner = 0;
    unit_mut(&mut want, 1).origin = None;
    unit_mut(&mut want, 3).origin = None;
    let u = unit_mut(&mut want, 0);
    (u.origin, u.phase) = (Some(CityId(1)), Phase::Done);
    explore(&mut want, 0, &city_1_tiles());
    assert_eq!(s, want);
    assert_eq!(
        s.unit(UnitId(0)).unwrap().last_dir,
        Some(SE),
        "last_dir kept"
    );

    assert_eq!(supported_units(&s, CityId(0)), 1, "only the Rider is left");
    assert_eq!(supported_units(&s, CityId(1)), 1, "only the capturer");
    assert_eq!(supported_units(&s, CityId(2)), 1);
}

/// Orphans never count again, even after their old city changes hands back.
#[test]
fn orphans_stay_orphaned_after_a_recapture() {
    let mut s = city_board();
    ok(&mut s, capture(0));
    // Test edit, not a rule: unit 0 steps to (4,3) and the Defender to (4,4). Then player 1
    // retakes city 1 with the Defender: unit 0 is orphaned in turn.
    ok(&mut s, Action::EndTurn);
    for (id, from, to) in [(0, p(4, 4), p(4, 3)), (1, p(6, 1), p(4, 4))] {
        tile_mut(&mut s, from).unit = None;
        tile_mut(&mut s, to).unit = Some(UnitId(id));
        unit_mut(&mut s, id).pos = to;
    }
    ok(&mut s, capture(1));
    assert_eq!(s.city(CityId(1)).owner, 1);
    assert_eq!(s.unit(UnitId(0)).unwrap().origin, None);
    assert_eq!(s.unit(UnitId(1)).unwrap().origin, Some(CityId(1)));
    assert_eq!(
        s.unit(UnitId(3)).unwrap().origin,
        None,
        "Super Unit stays orphaned"
    );
    assert_eq!(supported_units(&s, CityId(1)), 1);
}

/// Spec: a captured city moves its city, tile and park points to the new owner; orphaned units
/// keep scoring for their owner. City 1: 100 + 50 * 2 + 5 * (6 - 5) = 205, one park 250,
/// 11 tiles = 220.
#[test]
fn captured_territory_scores_for_the_capturer() {
    let before = city_board();
    let mut s = before.clone();
    ok(&mut s, capture(0));
    let (b0, b1) = (score_breakdown(&before, 0), score_breakdown(&before, 1));
    let (a0, a1) = (score_breakdown(&s, 0), score_breakdown(&s, 1));
    assert_eq!((b0.territory, a0.territory), (180, 180 + 220));
    assert_eq!((b1.territory, a1.territory), (220 + 180, 180));
    assert_eq!((b0.cities, a0.cities), (100, 100 + 205));
    assert_eq!((b1.cities, a1.cities), (205 + 100, 100));
    assert_eq!((b0.parks, a0.parks), (0, 250));
    assert_eq!((b1.parks, a1.parks), (250, 0));
    // Player 1's units: Defender 15, two Warriors 20, Super Unit 50, before and after.
    assert_eq!((b1.units, a1.units), (85, 85));
    // Player 0 explored 2 tiles, then all 11 of city 1's.
    assert_eq!((b0.exploration, a0.exploration), (10, 55));
}

/// Recover heals 4 in the player's own territory and 2 elsewhere; city 1's tiles switch sides.
#[test]
fn recover_follows_the_new_owner_of_the_territory() {
    let mut control = city_board();
    ok(&mut control, Action::Recover { unit: UnitId(5) });
    assert_eq!(
        control.unit(UnitId(5)).unwrap().health,
        5,
        "enemy territory: +2"
    );
    ok(&mut control, Action::EndTurn);
    ok(&mut control, Action::Recover { unit: UnitId(6) });
    assert_eq!(
        control.unit(UnitId(6)).unwrap().health,
        7,
        "own territory: +4"
    );

    let mut s = city_board();
    ok(&mut s, capture(0));
    ok(&mut s, Action::Recover { unit: UnitId(5) });
    assert_eq!(
        s.unit(UnitId(5)).unwrap().health,
        7,
        "now own territory: +4"
    );
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::Recover { unit: UnitId(6) });
    assert_eq!(
        s.unit(UnitId(6)).unwrap().health,
        5,
        "now enemy territory: +2"
    );
}

/// ```text
///      0 1 2 3 4
///   0  . . 1 . .      city 0 on (2,2): player 1's walled city (not a capital)
///   1  . 3 . . .      units: 0 player 0 Warrior (on the city), 1 player 1 Archer,
///   2  . . 0 . .      2 player 1 Warrior, 3 player 0 Warrior. Everything explored.
///   3  . . 2 . .
///   4  . . . . .
/// ```
///
/// Archer (attack 2, 10/10) on a Warrior (defense 2, 10/10), two tiles away so no retaliation:
/// bonus 1 gives round(2/4 * 2 * 4.5) = round(4.5) = 5 damage; the wall's 4 gives
/// round(2/10 * 9) = round(1.8) = 2. Warrior on Warrior at full health with bonus 1: 5 and 5
/// (combat golden 1); with the city's 1.5 the defender would keep 6 (round(2/5 * 9) = 4).
#[test]
fn city_defense_follows_the_new_owner() {
    let board = || {
        StateBuilder::new(5)
            .city_with(1, p(2, 2), |c| c.wall = true)
            .unit(0, Warrior, p(2, 2))
            .unit(1, Archer, p(0, 2))
            .unit(1, Warrior, p(3, 2))
            .unit(0, Warrior, p(1, 1))
            .explore_all()
            .build()
    };
    let shoot = Action::Attack {
        unit: UnitId(1),
        target: p(2, 2),
    };

    // Before the capture the city is the enemy's: no bonus for unit 0.
    let mut control = board();
    ok(&mut control, Action::EndTurn);
    ok(&mut control, shoot);
    assert_eq!(control.unit(UnitId(0)).unwrap().health, 5);

    // After the capture the wall protects unit 0 (Warriors have Fortify).
    let mut s = board();
    ok(&mut s, capture(0));
    ok(&mut s, Action::EndTurn);
    ok(&mut s, shoot);
    assert_eq!(s.unit(UnitId(0)).unwrap().health, 8);
    ok(&mut s, Action::EndTurn);

    // Unit 0 steps off, player 1's Warrior walks in, and gets no bonus in the lost city.
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(2, 3),
        },
    );
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(2),
            to: p(2, 2),
        },
    );
    ok(&mut s, Action::EndTurn);
    ok(
        &mut s,
        Action::Attack {
            unit: UnitId(3),
            target: p(2, 2),
        },
    );
    assert_eq!(s.unit(UnitId(2)).unwrap().health, 5, "defender, no bonus");
    assert_eq!(s.unit(UnitId(3)).unwrap().health, 5, "attacker");
}

// ---------- legality ----------

/// ```text
///      0 1 2
///   0  E . .      E = player 1 Warrior; v = village under player 0's Warrior (unit 0),
///   1  . v .          which has 3 kills and a promotion available
///   2  . . .
/// ```
/// Per unit: Move (row-major), Attack, Capture, Recover, Promote; no cities, techs or stars.
#[test]
fn legal_actions_list_capture_between_attack_and_recover() {
    let s = StateBuilder::new(3)
        .village(p(1, 1))
        .unit_with(0, Warrior, p(1, 1), |u| {
            u.kills = 3;
            u.promotion_available = true;
        })
        .unit(1, Warrior, p(0, 0))
        .explore_all()
        .build();
    let id = UnitId(0);
    let mut want: Vec<Action> = [
        p(0, 1),
        p(0, 2),
        p(1, 0),
        p(1, 2),
        p(2, 0),
        p(2, 1),
        p(2, 2),
    ]
    .into_iter()
    .map(|to| Action::Move { unit: id, to })
    .collect();
    want.extend([
        Action::Attack {
            unit: id,
            target: p(0, 0),
        },
        Action::Capture { unit: id },
        Action::Recover { unit: id },
        Action::Promote { unit: id },
        Action::EndTurn,
    ]);
    assert_eq!(legal_actions(&s), want);
}

#[test]
fn capture_needs_a_village_or_an_enemy_city_under_the_unit() {
    // Units 0..3 of player 0: on a plain tile, on its own city, on a village, on an enemy city.
    let s = StateBuilder::new(6)
        .city(0, p(0, 3))
        .city(1, p(4, 4))
        .village(p(2, 0))
        .unit(0, Warrior, p(0, 0))
        .unit(0, Warrior, p(0, 3))
        .unit(0, Warrior, p(2, 0))
        .unit(0, Warrior, p(4, 4))
        .explore_all()
        .build();
    let legal = legal_actions(&s);
    let mut t = s.clone();
    illegal(&mut t, capture(0));
    illegal(&mut t, capture(1));
    assert!(legal.contains(&capture(2)));
    assert!(legal.contains(&capture(3)));
    // An enemy city's territory is not its city tile.
    let mut t = StateBuilder::new(6)
        .city(1, p(4, 4))
        .unit(0, Warrior, p(3, 3))
        .explore_all()
        .build();
    illegal(&mut t, capture(0));
}

#[test]
fn capture_needs_a_fresh_unit() {
    let board = || {
        StateBuilder::new(5)
            .village(p(2, 2))
            .unit(0, Warrior, p(2, 1))
            .explore_all()
    };
    // A unit that moved onto the village this turn must wait for its next turn.
    let mut s = board().build();
    ok(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(2, 2),
        },
    );
    illegal(&mut s, capture(0));
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::EndTurn);
    ok(&mut s, capture(0));
    // Moved, Attacked and Done units on a village cannot capture.
    for phase in [Phase::Moved, Phase::Attacked, Phase::Done] {
        let mut s = StateBuilder::new(5)
            .village(p(2, 2))
            .unit_with(0, Rider, p(2, 2), |u| u.phase = phase)
            .explore_all()
            .build();
        illegal(&mut s, capture(0));
    }
    // Capture spends the turn: nothing else for that unit afterwards.
    let mut s = StateBuilder::new(5)
        .village(p(2, 2))
        .unit(0, Rider, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, capture(0));
    assert_eq!(legal_actions(&s), [Action::EndTurn]);
}

#[test]
fn only_the_current_players_units_capture() {
    let mut s = StateBuilder::new(5)
        .village(p(2, 2))
        .unit(1, Warrior, p(2, 2))
        .explore_all()
        .build();
    illegal(&mut s, capture(0));
    ok(&mut s, Action::EndTurn);
    ok(&mut s, capture(0));
    assert_eq!(s.cities[0].owner, 1);
    // A unit id that does not exist.
    illegal(&mut s, capture(7));
}

#[test]
fn a_pending_reward_blocks_capture() {
    let mut s = StateBuilder::new(5)
        .city_with(0, p(0, 0), |c| {
            (c.level, c.population, c.pending_reward) = (2, 2, true);
        })
        .village(p(3, 3))
        .unit(0, Warrior, p(3, 3))
        .explore_all()
        .build();
    illegal(&mut s, capture(0));
    ok(
        &mut s,
        Action::ChooseReward {
            city: CityId(0),
            reward: Reward::Workshop,
        },
    );
    ok(&mut s, capture(0));
}

/// F4: every unit type may capture, a village or an enemy city.
#[test]
fn any_unit_type_can_capture() {
    for kind in [Warrior, Rider, Archer, Defender, SuperUnit] {
        let mut s = StateBuilder::new(6)
            .city(1, p(4, 4))
            .village(p(1, 1))
            .unit(0, kind, p(1, 1))
            .unit(0, kind, p(4, 4))
            .explore_all()
            .build();
        ok(&mut s, capture(0));
        ok(&mut s, capture(1));
        assert_eq!(s.cities[1].owner, 0, "{kind:?}");
        assert_eq!(s.cities[0].owner, 0, "{kind:?}");
    }
}

/// The defender must die first: an occupied city cannot be entered, the kill-move leaves the
/// attacker Attacked, and the capture comes on its next turn. Warrior (10/10) on a Warrior at
/// 1/10 in its own city (bonus 1.5): attack force 2, defense force 2 * 0.1 * 1.5 = 0.3,
/// round(2 / 2.3 * 2 * 4.5) = 8 damage, a kill.
#[test]
fn an_enemy_city_is_captured_only_after_its_defender_dies() {
    let mut s = StateBuilder::new(5)
        .city(1, p(2, 2))
        .unit(0, Warrior, p(2, 1))
        .unit_with(1, Warrior, p(2, 2), |u| u.health = 1)
        .explore_all()
        .build();
    illegal(
        &mut s,
        Action::Move {
            unit: UnitId(0),
            to: p(2, 2),
        },
    );
    ok(
        &mut s,
        Action::Attack {
            unit: UnitId(0),
            target: p(2, 2),
        },
    );
    let u = s.unit(UnitId(0)).unwrap();
    assert_eq!((u.pos, u.phase), (p(2, 2), Phase::Attacked), "kill-move");
    illegal(&mut s, capture(0));
    ok(&mut s, Action::EndTurn);
    ok(&mut s, Action::EndTurn);
    ok(&mut s, capture(0));
    assert_eq!(s.cities[0].owner, 0);
}

// ---------- win by capital capture ----------

/// ```text
///      0 1 2 3 4
///   0  A a . . .      A = city 0, player 0's capital; B = city 1, player 1's capital
///   1  a a . . .      unit 0: player 0 Warrior on B; unit 1: player 1 Warrior (origin B)
///   2  . . 1 . .      on (2,2). Player 0 has explored only (4,4).
///   3  . . . b b
///   4  . . . b 0
/// ```
fn capital_board() -> State {
    StateBuilder::new(5)
        .city_with(0, p(0, 0), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit(0, Warrior, p(4, 4))
        .unit_with(1, Warrior, p(2, 2), |u| u.origin = Some(CityId(1)))
        .explore(0, &[p(4, 4)])
        .build()
}

/// After the end: no legal actions, and `apply` rejects every action kind without a change.
fn assert_final(s: &State) {
    assert!(s.is_terminal());
    assert!(legal_actions(s).is_empty(), "{:?}", legal_actions(s));
    for a in every_action_kind(s) {
        let mut t = s.clone();
        assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{a:?}");
        assert_eq!(t, *s, "{a:?} changed a final state");
    }
}

#[test]
fn capturing_the_enemy_capital_wins() {
    let before = capital_board();
    let mut s = before.clone();
    ok(&mut s, capture(0));

    // The capture itself applies in full, then the game is over.
    let mut want = before.clone();
    city_mut(&mut want, 1).owner = 0;
    unit_mut(&mut want, 1).origin = None;
    let u = unit_mut(&mut want, 0);
    (u.origin, u.phase) = (Some(CityId(1)), Phase::Done);
    explore(&mut want, 0, &[p(3, 3), p(3, 4), p(4, 3), p(4, 4)]);
    want.outcome = Some(Outcome::Win(0));
    assert_eq!(s, want);
    assert_final(&s);
    // The player-1 unit and the capturer get nothing either.
    let mut t = s.clone();
    assert_eq!(apply(&mut t, &capture(1)), Err(ActionError::Illegal));
    assert_eq!(
        apply(
            &mut t,
            &Action::Move {
                unit: UnitId(1),
                to: p(2, 3)
            }
        ),
        Err(ActionError::Illegal)
    );
    assert_eq!(t, s);
}

#[test]
fn player_1_capturing_player_0s_capital_wins_for_player_1() {
    let mut s = StateBuilder::new(5)
        .city_with(0, p(0, 0), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .unit(1, Defender, p(0, 0))
        .unit(0, Warrior, p(2, 2))
        .current(1)
        .turn(3)
        .explore_all()
        .build();
    ok(&mut s, capture(0));
    assert_eq!(s.outcome, Some(Outcome::Win(1)));
    assert_eq!((s.turn, s.current), (3, 1), "no turn change");
    assert_eq!(s.cities[0].owner, 1);
    assert_final(&s);
}

#[test]
fn capturing_a_non_capital_city_or_a_village_does_not_end_the_game() {
    let mut s = StateBuilder::new(6)
        .city_with(1, p(0, 0), |c| c.is_capital = true)
        .city(1, p(4, 4))
        .village(p(2, 2))
        .unit(0, Warrior, p(4, 4))
        .unit(0, Warrior, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, capture(0));
    assert_eq!(s.outcome, None);
    ok(&mut s, capture(1));
    assert_eq!(s.outcome, None);
    assert!(!s.is_terminal());
    assert_eq!(legal_actions(&s), [Action::EndTurn]);
    ok(&mut s, Action::EndTurn);
}
