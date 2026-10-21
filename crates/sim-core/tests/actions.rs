//! Unit actions through `legal_actions` / `apply` on scripted boards (docs/MILESTONE_3 "Move",
//! "Attack", "Recover", "Promote", "EndTurn", "legal_actions and apply", "Vision and the explored
//! mask"). Combat numbers are the spec's five wiki-calculator rows (milestone 1 goldens) or are
//! worked by hand from the spec formula in the comment next to them.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::Terrain;
use sim_core::state::{Dir, Outcome, Phase, State, UnitId};
use sim_core::testing::StateBuilder;
use UnitType::*;

fn mv(unit: u32, to: Pos) -> Action {
    Action::Move {
        unit: UnitId(unit),
        to,
    }
}
fn atk(unit: u32, target: Pos) -> Action {
    Action::Attack {
        unit: UnitId(unit),
        target,
    }
}
fn recover(unit: u32) -> Action {
    Action::Recover { unit: UnitId(unit) }
}
fn promote(unit: u32) -> Action {
    Action::Promote { unit: UnitId(unit) }
}

/// Applies an action that must be legal (and checks it is listed).
fn ok(s: &mut State, a: Action) {
    assert!(
        legal_actions(s).contains(&a),
        "{a:?} should be legal; legal = {:?}",
        legal_actions(s)
    );
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

/// Asserts an action is illegal: not listed, `apply` errors and changes nothing.
fn illegal(s: &mut State, a: Action) {
    assert!(!legal_actions(s).contains(&a), "{a:?} should not be legal");
    let before = s.clone();
    assert_eq!(apply(s, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(*s, before, "illegal {a:?} changed the state");
}

fn alive(s: &State, id: u32) -> bool {
    s.units.iter().any(|u| u.id == UnitId(id))
}

fn dir(dr: i8, dc: i8) -> Option<Dir> {
    Some(Dir { dr, dc })
}

fn open(n: u8) -> StateBuilder {
    StateBuilder::new(n)
}

fn rect(r0: u8, r1: u8, c0: u8, c1: u8) -> Vec<Pos> {
    (r0..=r1)
        .flat_map(|r| (c0..=c1).map(move |c| p(r, c)))
        .collect()
}

// ---------- combat goldens through apply ----------

#[test]
fn golden_1_warrior_vs_warrior() {
    // Spec row 1: Warrior 10 vs Warrior 10, bonus 1 -> 5 / 5.
    let mut s = open(3)
        .unit(0, Warrior, p(1, 0))
        .unit(1, Warrior, p(1, 1))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 0).health, 5);
    assert_eq!(unit(&s, 1).health, 5);
    assert_eq!(unit(&s, 0).pos, p(1, 0), "no kill, no move");
    assert_eq!(unit(&s, 0).phase, Phase::Attacked);
    assert_eq!(unit(&s, 0).last_dir, None, "melee attack without kill-move");
    assert_eq!(unit(&s, 1).phase, Phase::Fresh);
    assert_eq!((unit(&s, 0).kills, unit(&s, 1).kills), (0, 0));
}

#[test]
fn golden_2_retaliation_kill() {
    // Spec row 2: Warrior 6 vs Warrior 10 -> attacker 0 (dies), defender 7.
    let mut s = open(3)
        .unit_with(0, Warrior, p(1, 0), |u| u.health = 6)
        .unit(1, Warrior, p(1, 1))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert!(!alive(&s, 0), "attacker removed");
    assert_eq!(s.tile(p(1, 0)).unit, None, "its tile is freed");
    assert_eq!(unit(&s, 1).health, 7);
    assert_eq!(unit(&s, 1).pos, p(1, 1));
    assert_eq!(unit(&s, 1).kills, 1, "retaliation kill is credited");
    assert_eq!(s.units.len(), 1);
    assert_eq!(s.next_unit_id, 2, "ids are never reused");
}

#[test]
fn golden_3_warrior_vs_defender() {
    // Spec row 3: Warrior 10 vs Defender 15, bonus 1 -> 2 / 11.
    let mut s = open(3)
        .unit(0, Warrior, p(1, 0))
        .unit(1, Defender, p(1, 1))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 0).health, 2);
    assert_eq!(unit(&s, 1).health, 11);
}

#[test]
fn golden_4_veteran_rider_vs_defender_in_city() {
    // Spec row 4: veteran Rider 15/15 vs Defender 11, bonus 1.5 (own city, Fortify) -> 7 / 8.
    let mut s = open(3)
        .city(1, p(1, 1))
        .unit_with(0, Rider, p(1, 0), |u| {
            u.max_health = 15;
            u.health = 15;
            u.veteran = true;
        })
        .unit_with(1, Defender, p(1, 1), |u| u.health = 11)
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 0).health, 7);
    assert_eq!(unit(&s, 1).health, 8);
}

#[test]
fn golden_5_defender_vs_archer_behind_wall() {
    // Spec row 5: Defender 15 vs Archer 10, bonus 4 (own walled city) -> 11 / 9.
    let mut s = open(3)
        .city_with(1, p(1, 1), |c| c.wall = true)
        .unit(0, Defender, p(1, 0))
        .unit(1, Archer, p(1, 1))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 0).health, 11);
    assert_eq!(unit(&s, 1).health, 9);
}

#[test]
fn warrior_vs_defender_in_own_city() {
    // aF = 2, dF = 3 * 1 * 1.5 = 4.5, total 6.5.
    // to defender: round(2/6.5 * 2 * 4.5) = round(2.77) = 3 -> 12.
    // to attacker: round(4.5/6.5 * 3 * 4.5) = round(9.35) = 9 -> 1.
    let mut s = open(3)
        .city(1, p(1, 1))
        .unit(0, Warrior, p(1, 0))
        .unit(1, Defender, p(1, 1))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 1).health, 12);
    assert_eq!(unit(&s, 0).health, 1);
}

#[test]
fn city_bonus_does_not_apply_in_an_enemy_city() {
    // Player 1's Defender stands in player 0's city: bonus 1, like golden 3 -> 2 / 11.
    let mut s = open(3)
        .city(0, p(1, 1))
        .unit(0, Warrior, p(1, 0))
        .unit(1, Defender, p(1, 1))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 0).health, 2);
    assert_eq!(unit(&s, 1).health, 11);
}

// ---------- range and retaliation ----------

#[test]
fn archer_at_range_two_takes_no_retaliation_from_melee() {
    // aF = 2, dF = 2: to defender round(4.5) = 5. Warrior range 1 < 2: no retaliation.
    let mut s = open(3)
        .unit(0, Archer, p(0, 0))
        .unit(1, Warrior, p(0, 2))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(0, 2)));
    assert_eq!(unit(&s, 1).health, 5);
    assert_eq!(unit(&s, 0).health, 10);
}

#[test]
fn archer_adjacent_takes_retaliation() {
    // Same forces at distance 1: retaliation round(2/4 * 2 * 4.5) = 5.
    let mut s = open(3)
        .unit(0, Archer, p(0, 1))
        .unit(1, Warrior, p(0, 2))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(0, 2)));
    assert_eq!(unit(&s, 1).health, 5);
    assert_eq!(unit(&s, 0).health, 5);
}

#[test]
fn archer_vs_archer_at_range_two_retaliates() {
    // The defender's range decides. aF = 2, dF = 1, total 3.
    // to defender: round(2/3 * 2 * 4.5) = 6 -> 4. to attacker: round(1/3 * 1 * 4.5) = round(1.5)
    // = 2 (half up) -> 8.
    let mut s = open(3)
        .unit(0, Archer, p(0, 0))
        .unit(1, Archer, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(2, 2)));
    assert_eq!(unit(&s, 1).health, 4);
    assert_eq!(unit(&s, 0).health, 8);
}

#[test]
fn attack_range_limits() {
    let mut s = open(5)
        .unit(0, Warrior, p(0, 0))
        .unit(0, Archer, p(4, 4))
        .unit(1, Warrior, p(0, 2)) // Warrior distance 2
        .unit(1, Warrior, p(4, 1)) // Archer distance 3
        .unit(1, Warrior, p(2, 2)) // Archer distance 2, Warrior distance 2
        .explore_all()
        .build();
    illegal(&mut s, atk(0, p(0, 2)));
    illegal(&mut s, atk(0, p(2, 2)));
    illegal(&mut s, atk(1, p(4, 1)));
    ok(&mut s, atk(1, p(2, 2)));
}

#[test]
fn attack_needs_an_enemy_on_an_explored_tile() {
    let mut s = open(3)
        .unit(0, Warrior, p(1, 1))
        .unit(0, Warrior, p(0, 0))
        .unit(1, Warrior, p(2, 2))
        .explore_all()
        .hide(0, &[p(2, 2)])
        .build();
    illegal(&mut s, atk(0, p(0, 0))); // friendly
    illegal(&mut s, atk(0, p(1, 2))); // empty
    illegal(&mut s, atk(0, p(2, 2))); // enemy on an unexplored tile (C3)
    illegal(&mut s, atk(0, p(1, 1))); // itself
}

#[test]
fn explored_target_needs_no_line_of_sight_or_path() {
    // Archer behind a mountain wall without Climbing still shoots over it.
    let mut s = board(&[". m .", ". m .", ". m ."])
        .unit(0, Archer, p(1, 0))
        .unit(1, Warrior, p(1, 2))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 2)));
}

// ---------- kill-move ----------

#[test]
fn melee_kill_moves_onto_the_target_and_reveals() {
    // Defender at 1 HP: aF = 2, dF = 0.2: to defender round(2/2.2 * 9) = 8 >= 1, so it dies;
    // the attacker takes no retaliation.
    let mut s = open(7)
        .unit(0, Warrior, p(3, 2))
        .unit_with(1, Warrior, p(3, 3), |u| u.health = 1)
        .explore(0, &rect(2, 4, 1, 3))
        .build();
    ok(&mut s, atk(0, p(3, 3)));
    assert!(!alive(&s, 1));
    let w = unit(&s, 0);
    assert_eq!(w.pos, p(3, 3));
    assert_eq!(w.health, 10, "no retaliation from a dead defender");
    assert_eq!(w.kills, 1);
    assert_eq!(w.phase, Phase::Attacked);
    assert_eq!(w.last_dir, dir(0, 1));
    assert_eq!(s.tile(p(3, 3)).unit, Some(UnitId(0)));
    assert_eq!(s.tile(p(3, 2)).unit, None);
    assert_eq!(
        explored_set(&s, 0),
        rect(2, 4, 1, 4),
        "vision radius 1 from (3,3)"
    );
    assert_eq!(
        explored_set(&s, 1),
        Vec::<Pos>::new(),
        "opponent mask untouched"
    );
}

#[test]
fn kill_move_diagonal_sets_direction() {
    let mut s = open(3)
        .unit(0, Warrior, p(2, 0))
        .unit_with(1, Warrior, p(1, 1), |u| u.health = 1)
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 0).pos, p(1, 1));
    assert_eq!(unit(&s, 0).last_dir, dir(-1, 1));
}

#[test]
fn no_kill_move_onto_a_mountain_without_climbing() {
    let mut s = open(7)
        .terrain(p(3, 3), Terrain::Mountain)
        .unit(0, Warrior, p(3, 2))
        .unit_with(1, Warrior, p(3, 3), |u| u.health = 1)
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(3, 3)));
    assert!(!alive(&s, 1));
    let w = unit(&s, 0);
    assert_eq!(w.pos, p(3, 2));
    assert_eq!(w.last_dir, None);
    assert_eq!(w.kills, 1);
    assert_eq!(w.phase, Phase::Attacked);
    assert_eq!(s.tile(p(3, 3)).unit, None);
    assert_eq!(s.tile(p(3, 2)).unit, Some(UnitId(0)));
}

#[test]
fn kill_move_onto_a_mountain_with_climbing_sees_radius_two() {
    let mut s = open(7)
        .terrain(p(3, 3), Terrain::Mountain)
        .techs(0, &[Tech::Climbing])
        .unit(0, Warrior, p(3, 2))
        .unit_with(1, Warrior, p(3, 3), |u| u.health = 1)
        .explore(0, &rect(2, 4, 1, 3))
        .build();
    ok(&mut s, atk(0, p(3, 3)));
    assert_eq!(unit(&s, 0).pos, p(3, 3));
    let mut want = rect(1, 5, 1, 5);
    want.extend(rect(2, 4, 1, 1));
    want.sort();
    want.dedup();
    assert_eq!(explored_set(&s, 0), want);
}

#[test]
fn ranged_kill_does_not_move() {
    // Archer at distance 2 and at distance 1; neither moves. last_dir points at the target.
    for from in [p(3, 1), p(3, 2)] {
        let mut s = open(7)
            .unit(0, Archer, from)
            .unit_with(1, Warrior, p(3, 3), |u| u.health = 1)
            .explore_all()
            .build();
        ok(&mut s, atk(0, p(3, 3)));
        assert!(!alive(&s, 1));
        let a = unit(&s, 0);
        assert_eq!(a.pos, from);
        assert_eq!(a.kills, 1);
        assert_eq!(a.last_dir, dir(0, 1));
        assert_eq!(a.phase, Phase::Attacked);
        assert_eq!(s.tile(p(3, 3)).unit, None);
    }
}

#[test]
fn ranged_attack_without_kill_sets_direction() {
    let mut s = open(7)
        .unit_with(0, Archer, p(3, 3), |u| u.last_dir = dir(1, 1))
        .unit(1, Defender, p(1, 4))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 4)));
    assert_eq!(unit(&s, 0).last_dir, dir(-1, 1));
    assert_eq!(unit(&s, 0).pos, p(3, 3));
}

#[test]
fn melee_attack_without_kill_keeps_direction() {
    let mut s = open(3)
        .unit_with(0, Warrior, p(1, 0), |u| u.last_dir = dir(1, 0))
        .unit(1, Warrior, p(1, 1))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert_eq!(unit(&s, 0).last_dir, dir(1, 0));
}

// ---------- phases: Dash and Escape ----------

#[test]
fn rider_move_attack_move() {
    // Rider 10 vs Warrior 10: 5 / 5 (as golden 1; Rider attack 2, Warrior defense 2).
    let mut s = open(7)
        .unit(0, Rider, p(3, 0))
        .unit(1, Warrior, p(3, 3))
        .explore_all()
        .build();
    ok(&mut s, mv(0, p(3, 2)));
    assert_eq!(unit(&s, 0).phase, Phase::Moved);
    illegal(&mut s, mv(0, p(3, 1)));
    illegal(&mut s, recover(0));
    ok(&mut s, atk(0, p(3, 3)));
    assert_eq!(unit(&s, 0).phase, Phase::Attacked);
    assert_eq!((unit(&s, 0).health, unit(&s, 1).health), (5, 5));
    illegal(&mut s, atk(0, p(3, 3)));
    illegal(&mut s, recover(0));
    // Escape: full budget of 2 from (3,2).
    ok(&mut s, mv(0, p(3, 0)));
    assert_eq!(unit(&s, 0).phase, Phase::Done);
    assert_eq!(unit(&s, 0).last_dir, dir(0, -1));
    illegal(&mut s, mv(0, p(3, 1)));
    illegal(&mut s, atk(0, p(3, 3)));
    illegal(&mut s, recover(0));
    assert!(
        legal_actions(&s).iter().all(|a| *a == Action::EndTurn),
        "only EndTurn is left: {:?}",
        legal_actions(&s)
    );
}

#[test]
fn rider_attack_then_escape_from_fresh() {
    let mut s = open(5)
        .unit(0, Rider, p(2, 1))
        .unit(1, Warrior, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(2, 2)));
    ok(&mut s, mv(0, p(0, 0)));
    assert_eq!(unit(&s, 0).phase, Phase::Done);
}

#[test]
fn rider_kill_move_keeps_the_escape_move() {
    // C11: after the kill-move the phase is Attacked; Escape still has a full budget.
    let mut s = open(7)
        .unit(0, Rider, p(3, 2))
        .unit_with(1, Warrior, p(3, 3), |u| u.health = 1)
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(3, 3)));
    assert_eq!(unit(&s, 0).pos, p(3, 3));
    assert_eq!(unit(&s, 0).phase, Phase::Attacked);
    ok(&mut s, mv(0, p(3, 5)));
    assert_eq!(unit(&s, 0).phase, Phase::Done);
}

#[test]
fn warrior_can_move_then_attack_but_not_attack_then_move() {
    // Dash: move, then attack.
    let mut s = open(5)
        .unit(0, Warrior, p(2, 0))
        .unit(1, Defender, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, mv(0, p(2, 1)));
    ok(&mut s, atk(0, p(2, 2)));
    // No Escape: attack, then no move.
    let mut t = open(5)
        .unit(0, Warrior, p(0, 1))
        .unit(1, Defender, p(0, 2))
        .explore_all()
        .build();
    ok(&mut t, atk(0, p(0, 2)));
    assert_eq!(unit(&t, 0).phase, Phase::Attacked);
    illegal(&mut t, mv(0, p(0, 0)));
    illegal(&mut t, mv(0, p(1, 1)));
}

#[test]
fn defender_without_dash_cannot_attack_after_moving() {
    let mut s = open(5)
        .unit(0, Defender, p(2, 0))
        .unit(1, Warrior, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, mv(0, p(2, 1)));
    assert_eq!(unit(&s, 0).phase, Phase::Moved);
    illegal(&mut s, atk(0, p(2, 2)));
    illegal(&mut s, mv(0, p(1, 0)));
}

#[test]
fn defender_attack_ends_its_actions() {
    let mut s = open(5)
        .unit(0, Defender, p(2, 1))
        .unit(1, Warrior, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(2, 2)));
    illegal(&mut s, mv(0, p(2, 0)));
    illegal(&mut s, atk(0, p(2, 2)));
}

#[test]
fn super_unit_moves_or_attacks() {
    let mut s = open(5)
        .unit(0, SuperUnit, p(2, 0))
        .unit(1, Defender, p(2, 2))
        .explore_all()
        .build();
    ok(&mut s, mv(0, p(2, 1)));
    illegal(&mut s, atk(0, p(2, 2)));
}

// ---------- move effects ----------

#[test]
fn move_updates_position_direction_phase_and_explored() {
    let mut s = open(7)
        .unit(0, Rider, p(3, 3))
        .explore(0, &rect(1, 5, 1, 5))
        .build();
    ok(&mut s, mv(0, p(1, 5)));
    let r = unit(&s, 0);
    assert_eq!(r.pos, p(1, 5));
    assert_eq!(r.phase, Phase::Moved);
    assert_eq!(r.last_dir, dir(-1, 1));
    assert_eq!(s.tile(p(1, 5)).unit, Some(UnitId(0)));
    assert_eq!(s.tile(p(3, 3)).unit, None);
    let mut want = rect(1, 5, 1, 5);
    want.extend(rect(0, 2, 4, 6));
    want.sort();
    want.dedup();
    assert_eq!(explored_set(&s, 0), want);
    assert!(explored_set(&s, 1).is_empty());
}

#[test]
fn move_direction_is_the_sign_of_net_displacement() {
    for (to, d) in [
        (p(3, 5), dir(0, 1)),
        (p(5, 3), dir(1, 0)),
        (p(1, 1), dir(-1, -1)),
        (p(2, 5), dir(-1, 1)),
        (p(4, 3), dir(1, 0)),
    ] {
        let mut s = open(7)
            .unit_with(0, Rider, p(3, 3), |u| u.last_dir = dir(1, 1))
            .explore_all()
            .build();
        ok(&mut s, mv(0, to));
        assert_eq!(unit(&s, 0).last_dir, d, "to {to:?}");
    }
}

#[test]
fn moving_onto_a_mountain_sees_radius_two() {
    let mut s = open(7)
        .terrain(p(3, 3), Terrain::Mountain)
        .techs(0, &[Tech::Climbing])
        .unit(0, Warrior, p(3, 2))
        .explore(0, &rect(2, 4, 1, 3))
        .build();
    ok(&mut s, mv(0, p(3, 3)));
    let mut want = rect(1, 5, 1, 5);
    want.extend(rect(2, 4, 1, 1));
    want.sort();
    want.dedup();
    assert_eq!(explored_set(&s, 0), want);
}

#[test]
fn move_to_unreachable_tiles_is_illegal() {
    let mut s = open(5)
        .terrain(p(2, 3), Terrain::Mountain)
        .unit(0, Warrior, p(2, 2))
        .unit(0, Warrior, p(1, 1))
        .explore_all()
        .build();
    illegal(&mut s, mv(0, p(2, 2))); // own tile
    illegal(&mut s, mv(0, p(1, 1))); // friendly
    illegal(&mut s, mv(0, p(2, 3))); // mountain, no Climbing
    illegal(&mut s, mv(0, p(0, 0))); // too far
    illegal(&mut s, mv(0, p(5, 5))); // off the map
    illegal(&mut s, mv(0, p(200, 2))); // far off the map
}

// ---------- kill credit and promotion ----------

/// Player 0's `kind` at (1,0) with `kills`/`veteran`, killing a 1 HP Warrior at (1,1).
fn kill_with(kind: UnitType, kills: u32, veteran: bool) -> State {
    let mut s = open(3)
        .unit_with(0, kind, p(1, 0), |u| {
            u.kills = kills;
            u.veteran = veteran;
            if veteran {
                u.max_health += 5;
                u.health += 5;
            }
        })
        .unit_with(1, Warrior, p(1, 1), |u| u.health = 1)
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert!(!alive(&s, 1));
    s
}

#[test]
fn promotion_becomes_available_at_exactly_three_kills() {
    let s = kill_with(Warrior, 1, false);
    assert_eq!(unit(&s, 0).kills, 2);
    assert!(!unit(&s, 0).promotion_available);
    let s = kill_with(Warrior, 2, false);
    assert_eq!(unit(&s, 0).kills, 3);
    assert!(unit(&s, 0).promotion_available);
    assert!(legal_actions(&s).contains(&promote(0)));
}

#[test]
fn veterans_keep_counting_kills_but_never_promote_again() {
    let mut s = kill_with(Warrior, 3, true);
    assert_eq!(unit(&s, 0).kills, 4);
    assert!(!unit(&s, 0).promotion_available);
    illegal(&mut s, promote(0));
}

#[test]
fn super_unit_never_gets_promotion() {
    let mut s = kill_with(SuperUnit, 2, false);
    assert_eq!(unit(&s, 0).kills, 3);
    assert!(!unit(&s, 0).promotion_available);
    illegal(&mut s, promote(0));
}

#[test]
fn retaliation_kill_can_make_the_defender_eligible() {
    // Golden 2 setup with the defender on 2 kills.
    let mut s = open(3)
        .unit_with(0, Warrior, p(1, 0), |u| u.health = 6)
        .unit_with(1, Warrior, p(1, 1), |u| u.kills = 2)
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    assert!(!alive(&s, 0));
    assert_eq!(unit(&s, 1).kills, 3);
    assert!(unit(&s, 1).promotion_available);
    assert_eq!(unit(&s, 1).pos, p(1, 1), "defenders never move");
}

#[test]
fn promote_effects_and_any_phase() {
    for phase in [Phase::Fresh, Phase::Moved, Phase::Attacked, Phase::Done] {
        let mut s = open(3)
            .unit_with(0, Warrior, p(1, 1), |u| {
                u.health = 3;
                u.kills = 3;
                u.promotion_available = true;
                u.phase = phase;
                u.last_dir = dir(0, -1);
            })
            .explore_all()
            .build();
        ok(&mut s, promote(0));
        let w = unit(&s, 0);
        assert_eq!((w.max_health, w.health), (15, 15), "{phase:?}");
        assert!(w.veteran);
        assert!(!w.promotion_available);
        assert_eq!(w.phase, phase, "promotion does not use the turn");
        assert_eq!(w.last_dir, dir(0, -1));
        assert_eq!(w.kills, 3);
        illegal(&mut s, promote(0));
    }
}

#[test]
fn promote_needs_availability_and_the_current_player() {
    let mut s = open(3)
        .unit_with(0, Warrior, p(0, 0), |u| u.kills = 3)
        .unit_with(1, Warrior, p(2, 2), |u| {
            u.kills = 3;
            u.promotion_available = true;
        })
        .explore_all()
        .build();
    illegal(&mut s, promote(0));
    illegal(&mut s, promote(1));
}

// ---------- recover ----------

/// Recover for player 0's Warrior at (2,2) of a 5x5 board shaped by `setup`.
fn recovered(health: u32, max: u32, setup: impl FnOnce(StateBuilder) -> StateBuilder) -> u32 {
    let mut s = setup(open(5))
        .unit_with(0, Warrior, p(2, 2), |u| {
            u.health = health;
            u.max_health = max;
        })
        .explore_all()
        .build();
    ok(&mut s, recover(0));
    assert_eq!(unit(&s, 0).phase, Phase::Done);
    unit(&s, 0).health
}

#[test]
fn recover_heals_four_in_own_territory_and_two_elsewhere() {
    assert_eq!(recovered(3, 10, |b| b.city(0, p(2, 2))), 7, "own city tile");
    assert_eq!(recovered(3, 10, |b| b.city(0, p(1, 1))), 7, "own territory");
    assert_eq!(
        recovered(3, 10, |b| b.city(1, p(1, 1))),
        5,
        "enemy territory"
    );
    assert_eq!(
        recovered(3, 10, |b| b.city(1, p(2, 2))),
        5,
        "enemy city tile"
    );
    assert_eq!(recovered(3, 10, |b| b), 5, "unowned");
    assert_eq!(
        recovered(3, 10, |b| b.city(0, p(4, 4))),
        5,
        "outside own border"
    );
}

#[test]
fn recover_is_capped_and_legal_at_full_health() {
    assert_eq!(recovered(9, 10, |b| b.city(0, p(2, 2))), 10);
    assert_eq!(recovered(9, 10, |b| b), 10);
    assert_eq!(recovered(13, 15, |b| b.city(0, p(2, 2))), 15);
    assert_eq!(recovered(10, 10, |b| b), 10, "full health: a pass");
}

#[test]
fn recover_only_from_fresh() {
    for phase in [Phase::Moved, Phase::Attacked, Phase::Done] {
        let mut s = open(3)
            .unit_with(0, Rider, p(1, 1), |u| {
                u.health = 3;
                u.phase = phase;
            })
            .explore_all()
            .build();
        illegal(&mut s, recover(0));
    }
}

// ---------- end turn ----------

#[test]
fn end_turn_alternates_players_and_counts_rounds_from_zero() {
    let mut s = open(3)
        .unit(0, Warrior, p(0, 0))
        .unit(1, Warrior, p(2, 2))
        .explore_all()
        .build();
    let mut seen = vec![(s.current, s.turn)];
    for _ in 0..5 {
        ok(&mut s, Action::EndTurn);
        seen.push((s.current, s.turn));
    }
    assert_eq!(seen, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]);
}

#[test]
fn end_turn_resets_only_the_new_current_players_units() {
    let mut s = open(5)
        .unit_with(0, Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .unit_with(0, Rider, p(0, 4), |u| u.phase = Phase::Moved)
        .unit_with(1, Warrior, p(4, 0), |u| {
            u.phase = Phase::Done;
            u.last_dir = dir(1, 0);
        })
        .unit_with(1, Archer, p(4, 4), |u| u.phase = Phase::Attacked)
        .explore_all()
        .build();
    ok(&mut s, Action::EndTurn);
    assert_eq!(unit(&s, 0).phase, Phase::Done);
    assert_eq!(unit(&s, 1).phase, Phase::Moved);
    assert_eq!(unit(&s, 2).phase, Phase::Fresh);
    assert_eq!(unit(&s, 3).phase, Phase::Fresh);
    assert_eq!(unit(&s, 2).last_dir, dir(1, 0), "last_dir is never reset");
    ok(&mut s, Action::EndTurn);
    assert_eq!(unit(&s, 0).phase, Phase::Fresh);
    assert_eq!(unit(&s, 1).phase, Phase::Fresh);
}

#[test]
fn end_turn_blocked_by_own_pending_reward_only() {
    let mut s = open(3)
        .city_with(0, p(0, 0), |c| c.pending_reward = true)
        .explore_all()
        .build();
    illegal(&mut s, Action::EndTurn);
    let mut s = open(3)
        .city_with(1, p(0, 0), |c| c.pending_reward = true)
        .explore_all()
        .build();
    ok(&mut s, Action::EndTurn);
}

#[test]
fn nothing_is_legal_once_there_is_an_outcome() {
    let mut s = open(3)
        .unit(0, Warrior, p(0, 0))
        .unit(1, Warrior, p(0, 1))
        .explore_all()
        .build();
    s.outcome = Some(Outcome::Draw);
    assert!(legal_actions(&s).is_empty());
    illegal(&mut s, Action::EndTurn);
    illegal(&mut s, recover(0));
    illegal(&mut s, atk(0, p(0, 1)));
}

// ---------- legal_actions ----------

#[test]
fn legal_actions_canonical_order() {
    // Unit 0 (P0 Warrior, Fresh) at (2,2), next to P1's unit 1 at (2,3).
    // Unit 2 (P0 Archer, Done, promotion available) at (0,4).
    // Unit 3 (P1 Warrior) at (4,0) has nothing (not current).
    // Unit 4 (P0 Rider, Moved) at (4,4): may only attack (Dash), but no enemy is in range 1.
    let s = open(5)
        .unit(0, Warrior, p(2, 2))
        .unit(1, Warrior, p(2, 3))
        .unit_with(0, Archer, p(0, 4), |u| {
            u.phase = Phase::Done;
            u.kills = 3;
            u.promotion_available = true;
        })
        .unit(1, Warrior, p(4, 0))
        .unit_with(0, Rider, p(4, 4), |u| u.phase = Phase::Moved)
        .explore_all()
        .build();
    let want = vec![
        mv(0, p(1, 1)),
        mv(0, p(1, 2)),
        mv(0, p(1, 3)),
        mv(0, p(2, 1)),
        mv(0, p(3, 1)),
        mv(0, p(3, 2)),
        mv(0, p(3, 3)),
        atk(0, p(2, 3)),
        recover(0),
        promote(2),
        Action::EndTurn,
    ];
    assert_eq!(bf_legal(&s), want, "test bug: brute force disagrees");
    assert_eq!(legal_actions(&s), want);
}

#[test]
fn legal_actions_attack_targets_row_major_and_moves_before_attacks() {
    // Archer (range 2) at (2,2) with enemies at (4,1), (0,3), (2,0), (1,2) (all within 2).
    let s = open(5)
        .unit(0, Archer, p(2, 2))
        .unit(1, Warrior, p(4, 1))
        .unit(1, Warrior, p(0, 3))
        .unit(1, Warrior, p(2, 0))
        .unit(1, Warrior, p(1, 2))
        .explore_all()
        .build();
    let got = legal_actions(&s);
    let attacks: Vec<Action> = got
        .iter()
        .copied()
        .filter(|a| matches!(a, Action::Attack { .. }))
        .collect();
    assert_eq!(
        attacks,
        vec![
            atk(0, p(0, 3)),
            atk(0, p(1, 2)),
            atk(0, p(2, 0)),
            atk(0, p(4, 1))
        ]
    );
    assert_eq!(got, bf_legal(&s));
}

#[test]
fn only_the_current_player_acts() {
    let mut s = open(3)
        .unit(0, Warrior, p(0, 0))
        .unit(1, Warrior, p(2, 2))
        .current(1)
        .explore_all()
        .build();
    illegal(&mut s, mv(0, p(0, 1)));
    illegal(&mut s, recover(0));
    ok(&mut s, mv(1, p(1, 1)));
    assert!(legal_actions(&s).iter().all(|a| match a {
        Action::Move { unit, .. }
        | Action::Attack { unit, .. }
        | Action::Recover { unit }
        | Action::Promote { unit } => *unit == UnitId(1),
        Action::EndTurn => true,
    }));
}

#[test]
fn apply_rejects_junk_without_changing_anything() {
    let mut s = open(3)
        .unit(0, Warrior, p(0, 0))
        .unit(1, Warrior, p(2, 2))
        .explore_all()
        .build();
    illegal(&mut s, mv(7, p(0, 1))); // no such unit
    illegal(&mut s, recover(7));
    illegal(&mut s, promote(7));
    illegal(&mut s, atk(7, p(2, 2)));
    illegal(&mut s, atk(0, p(3, 3))); // off the map
    illegal(&mut s, atk(0, p(255, 255)));
}

#[test]
fn dead_units_cannot_act() {
    let mut s = open(3)
        .unit_with(0, Warrior, p(1, 0), |u| u.health = 6)
        .unit(1, Warrior, p(1, 1))
        .unit(0, Warrior, p(0, 0))
        .explore_all()
        .build();
    ok(&mut s, atk(0, p(1, 1)));
    illegal(&mut s, recover(0));
    illegal(&mut s, mv(0, p(2, 0)));
    ok(&mut s, recover(2));
}
