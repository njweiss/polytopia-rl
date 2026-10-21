//! Milestone 5: the Super Unit reward, the forced push of a city tile's occupant, capacity, and
//! the new reward menus through `legal_actions`/`apply` (docs/MILESTONE_5 "Super Unit spawn",
//! "Menus", E6 to E9; spec "Forced unit spawn"). Every expected state is built by hand from the
//! brief: the ring N, NE, E, SE, S, SW, W, NW (clockwise as drawn, rows grow downward), probes
//! intended, 1 ccw, 1 cw, 2 ccw, 2 cw, 3 ccw, 3 cw, opposite; Super Unit health 40 (spec table).

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Reward, Tech, UnitType};
use sim_core::economy::supported_units;
use sim_core::grid::Pos;
use sim_core::map::Terrain;
use sim_core::spawn::{push_direction, spawn_super_unit};
use sim_core::state::{CityId, Dir, Phase, PlayerId, State, Unit, UnitId};
use sim_core::testing::StateBuilder;
use UnitType::*;

// ---------- helpers ----------

const N: Dir = Dir { dr: -1, dc: 0 };
const NE: Dir = Dir { dr: -1, dc: 1 };
const E: Dir = Dir { dr: 0, dc: 1 };
const SE: Dir = Dir { dr: 1, dc: 1 };
const S: Dir = Dir { dr: 1, dc: 0 };
const SW: Dir = Dir { dr: 1, dc: -1 };
const W: Dir = Dir { dr: 0, dc: -1 };
const NW: Dir = Dir { dr: -1, dc: -1 };

fn choose(city: u32, reward: Reward) -> Action {
    Action::ChooseReward {
        city: CityId(city),
        reward,
    }
}

/// The Super Unit the brief's rule 2 creates.
fn super_unit(id: u32, owner: PlayerId, pos: Pos, city: u32) -> Unit {
    Unit {
        id: UnitId(id),
        kind: SuperUnit,
        owner,
        pos,
        health: 40,
        max_health: 40,
        origin: Some(CityId(city)),
        kills: 0,
        veteran: false,
        promotion_available: false,
        phase: Phase::Done,
        last_dir: None,
    }
}

/// `before` with unit `id` moved to `to` (occupancy and its owner's mask: radius 1, or 2 on a
/// mountain) and the Super Unit for city `city` added on the city tile with the next id.
fn pushed(before: &State, id: u32, to: Pos, city: u32) -> State {
    let mut s = before.clone();
    let c = s.cities[city as usize].clone();
    let u = s.units.iter_mut().find(|u| u.id == UnitId(id)).unwrap();
    let owner = u.owner;
    u.pos = to;
    tile_mut(&mut s, to).unit = Some(UnitId(id));
    for q in vision(before, to) {
        let i = idx(&s, q);
        s.players[owner as usize].explored[i] = true;
    }
    add_super_unit(&mut s, &c.pos, c.owner, city);
    s
}

fn add_super_unit(s: &mut State, at: &Pos, owner: PlayerId, city: u32) {
    let id = s.next_unit_id;
    s.next_unit_id += 1;
    s.units.push(super_unit(id, owner, *at, city));
    tile_mut(s, *at).unit = Some(UnitId(id));
}

fn spawned(before: &State, city: u32) -> (State, UnitId) {
    let mut s = before.clone();
    let id = spawn_super_unit(&mut s, CityId(city));
    (s, id)
}

/// 11x11, P0 city 0 at `city` (level 5), one occupant (unit 0) owned by `owner` with `last_dir`.
fn occupied(city: Pos, owner: PlayerId, last_dir: Option<Dir>) -> StateBuilder {
    StateBuilder::new(11)
        .city_with(0, city, |c| c.level = 5)
        .unit_with(owner, Warrior, city, |u| u.last_dir = last_dir)
}

// ---------- push direction ----------

#[test]
fn push_direction_follows_or_reverses_last_dir() {
    for (owner, last, want) in [
        (0, E, E),
        (1, E, W),
        (0, SW, SW),
        (1, SW, NE),
        (1, N, S),
        (0, NW, NW),
        (1, NW, SE),
    ] {
        let s = occupied(p(5, 5), owner, Some(last)).build();
        assert_eq!(
            push_direction(&s, UnitId(0), 0),
            want,
            "owner {owner}, last {last:?}"
        );
    }
    // The city owner is an argument: the same unit is friendly to P1's city.
    let s = occupied(p(5, 5), 1, Some(E)).build();
    assert_eq!(push_direction(&s, UnitId(0), 1), E);
}

#[test]
fn push_direction_without_last_dir_points_to_the_center() {
    // 11x11: center (5,5). Sign of (5 - row, 5 - col) per axis; the exact center goes south.
    let cases = [
        (p(2, 2), SE),
        (p(2, 5), S),
        (p(2, 8), SW),
        (p(5, 8), W),
        (p(8, 8), NW),
        (p(8, 5), N),
        (p(8, 2), NE),
        (p(5, 2), E),
        (p(5, 5), S),
        // Sign, not angle: (4,0) is mostly west of the center but still goes SE.
        (p(4, 0), SE),
        (p(1, 3), SE),
        (p(0, 10), SW),
        (p(10, 4), NE),
        (p(6, 5), N),
        (p(5, 0), E),
        (p(4, 5), S),
    ];
    for (at, want) in cases {
        for owner in 0..2u8 {
            let s = occupied(at, owner, None).build();
            assert_eq!(
                push_direction(&s, UnitId(0), 0),
                want,
                "{at:?} owner {owner}"
            );
        }
    }
    // 9x9: center (4,4).
    for (at, want) in [(p(4, 4), S), (p(4, 5), W), (p(0, 0), SE), (p(8, 3), NE)] {
        let s = StateBuilder::new(9).unit(0, Warrior, at).build();
        assert_eq!(push_direction(&s, UnitId(0), 0), want, "9x9 {at:?}");
    }
}

// ---------- spawn goldens ----------

#[test]
fn spawn_on_an_empty_city_tile() {
    // 7x7, P1 city 0 at (3,3), two units elsewhere (ids 0 and 1).
    let before = StateBuilder::new(7)
        .city_with(1, p(3, 3), |c| c.level = 5)
        .unit(0, Warrior, p(0, 0))
        .unit(1, Archer, p(6, 6))
        .current(1)
        .build();
    let (s, id) = spawned(&before, 0);
    assert_eq!(id, UnitId(2));
    let mut want = before.clone();
    add_super_unit(&mut want, &p(3, 3), 1, 0);
    assert_eq!(s, want);
    assert_eq!(unit(&s, 2).phase, Phase::Done, "spawns Done (E6)");
}

#[test]
fn friendly_occupant_is_pushed_along_its_last_dir() {
    let before = occupied(p(5, 5), 0, Some(E)).build();
    let (s, id) = spawned(&before, 0);
    assert_eq!(id, UnitId(1));
    assert_eq!(s, pushed(&before, 0, p(5, 6), 0));
}

#[test]
fn enemy_occupant_is_pushed_opposite_its_last_dir() {
    let before = occupied(p(5, 5), 1, Some(E)).build();
    let (s, _) = spawned(&before, 0);
    assert_eq!(s, pushed(&before, 0, p(5, 4), 0));
    // The enemy's mask grows, not the city owner's.
    assert!(s.is_explored(1, p(4, 3)) && !s.is_explored(0, p(4, 3)));
}

#[test]
fn occupant_without_last_dir_is_pushed_toward_the_center() {
    for (city, to) in [
        (p(2, 2), p(3, 3)),
        (p(8, 5), p(7, 5)),
        (p(5, 8), p(5, 7)),
        (p(4, 0), p(5, 1)),
        (p(5, 5), p(6, 5)), // exact center: south
    ] {
        let before = occupied(city, 0, None).build();
        let (s, _) = spawned(&before, 0);
        assert_eq!(s, pushed(&before, 0, to, 0), "city at {city:?}");
    }
}

#[test]
fn probe_order_counterclockwise_first() {
    // City (5,5), friendly occupant moving E. Probe tiles in order: E (5,6), NE (4,6), SE (6,6),
    // N (4,5), S (6,5), NW (4,4), SW (6,4), W (5,4). Block the first k with a unit (either
    // owner) or a mountain (P0 lacks Climbing), alternately; the occupant lands on probe k, and
    // with all 8 blocked it is removed.
    let probes = [
        p(5, 6),
        p(4, 6),
        p(6, 6),
        p(4, 5),
        p(6, 5),
        p(4, 4),
        p(6, 4),
        p(5, 4),
    ];
    for k in 0..=8 {
        let mut b = occupied(p(5, 5), 0, Some(E));
        for (j, &q) in probes.iter().enumerate().take(k) {
            b = match j % 4 {
                0 => b.unit(1, Warrior, q),
                1 => b.terrain(q, Terrain::Mountain),
                2 => b.unit(0, Defender, q),
                _ => b.terrain(q, Terrain::Mountain),
            };
        }
        let before = b.build();
        let (s, _) = spawned(&before, 0);
        if k < 8 {
            assert_eq!(s, pushed(&before, 0, probes[k], 0), "{k} blocked");
        } else {
            let mut want = before.clone();
            want.units.retain(|u| u.id != UnitId(0));
            add_super_unit(&mut want, &p(5, 5), 0, 0);
            assert_eq!(s, want, "all blocked: removed");
        }
    }
}

#[test]
fn mountains_block_only_without_the_occupants_climbing() {
    // City (5,5) of P0. Enemy occupant (P1) last moved E, so the intended direction is W (5,4),
    // a mountain. Ring from W: ccw is SW (6,4).
    let base = |p0: &[Tech], p1: &[Tech]| {
        occupied(p(5, 5), 1, Some(E))
            .terrain(p(5, 4), Terrain::Mountain)
            .techs(0, p0)
            .techs(1, p1)
            .build()
    };
    // The city owner's Climbing does not help the occupant.
    let before = base(&[Tech::Climbing], &[]);
    assert_eq!(spawned(&before, 0).0, pushed(&before, 0, p(6, 4), 0));
    // The occupant's Climbing lets it onto the mountain; on a mountain it sees radius 2.
    let before = base(&[], &[Tech::Climbing]);
    let (s, _) = spawned(&before, 0);
    assert_eq!(s, pushed(&before, 0, p(5, 4), 0));
    assert!(s.is_explored(1, p(3, 2)), "radius 2 from (5,4)");
}

#[test]
fn the_map_edge_blocks_like_anything_else() {
    // Top edge, city (0,5), friendly occupant moving N: N, NW and NE are off the map, so the next
    // probe, W (0,4), takes it.
    let before = occupied(p(0, 5), 0, Some(N)).build();
    assert_eq!(spawned(&before, 0).0, pushed(&before, 0, p(0, 4), 0));
    // Corner (0,0), moving NW: NW, W, N, SW, NE are off the map; S (1,0) is the sixth probe.
    let before = occupied(p(0, 0), 0, Some(NW)).build();
    assert_eq!(spawned(&before, 0).0, pushed(&before, 0, p(1, 0), 0));
    // Bottom-right corner (10,10) without last_dir: toward the center is NW (9,9).
    let before = occupied(p(10, 10), 0, None).build();
    assert_eq!(spawned(&before, 0).0, pushed(&before, 0, p(9, 9), 0));
}

#[test]
fn cities_villages_and_fog_are_fine_destinations() {
    // Intended E (5,6) is a village; the occupant has explored nothing there.
    let before = occupied(p(5, 5), 0, Some(E)).village(p(5, 6)).build();
    assert_eq!(spawned(&before, 0).0, pushed(&before, 0, p(5, 6), 0));
    // Intended E (5,6) is an empty city of the other player.
    let before = occupied(p(5, 5), 0, Some(E)).city(1, p(5, 6)).build();
    assert_eq!(spawned(&before, 0).0, pushed(&before, 0, p(5, 6), 0));
}

#[test]
fn a_pushed_unit_keeps_everything_but_its_position() {
    for (phase, owner) in [
        (Phase::Fresh, 0),
        (Phase::Moved, 1),
        (Phase::Attacked, 0),
        (Phase::Done, 1),
    ] {
        let before = StateBuilder::new(11)
            .city_with(0, p(5, 5), |c| c.level = 6)
            .unit_with(owner, Rider, p(5, 5), |u| {
                u.phase = phase;
                u.last_dir = Some(S);
                u.kills = 4;
                u.veteran = true;
                u.max_health = 15;
                u.health = 6;
                u.origin = Some(CityId(0));
            })
            .build();
        // Friendly: along S to (6,5); enemy: opposite, N to (4,5).
        let to = if owner == 0 { p(6, 5) } else { p(4, 5) };
        let (s, _) = spawned(&before, 0);
        assert_eq!(s, pushed(&before, 0, to, 0), "{phase:?}");
        let mut moved = unit(&before, 0).clone();
        moved.pos = to;
        assert_eq!(*unit(&s, 0), moved);
    }
}

#[test]
fn a_removed_occupant_is_not_a_kill() {
    // City (1,1) on 3x3, enemy occupant with 2 kills; every neighbor holds a unit.
    let mut b = StateBuilder::new(3)
        .city_with(0, p(1, 1), |c| c.level = 5)
        .unit_with(1, Warrior, p(1, 1), |u| u.kills = 2);
    for q in all_pos(3) {
        if q != p(1, 1) {
            let owner = (q.row + q.col) % 2;
            b = b.unit_with(owner, Archer, q, |u| u.kills = 1);
        }
    }
    let before = b.build();
    let (s, id) = spawned(&before, 0);
    assert_eq!(id, UnitId(9));
    let mut want = before.clone();
    want.units.retain(|u| u.id != UnitId(0));
    add_super_unit(&mut want, &p(1, 1), 0, 0);
    assert_eq!(s, want);
    assert!(s.units.iter().all(|u| u.kind == SuperUnit || u.kills == 1));
    assert_eq!(unit(&s, 9).kills, 0);
}

// ---------- capacity ----------

#[test]
fn super_units_never_count_toward_capacity() {
    // Level 1 city 0 at (2,2), capacity 2. One Warrior and two Super Units from city 0.
    let b = StateBuilder::new(5)
        .city(0, p(2, 2))
        .unit_with(0, Warrior, p(0, 0), |u| u.origin = Some(CityId(0)))
        .unit_with(0, SuperUnit, p(0, 1), |u| u.origin = Some(CityId(0)))
        .unit_with(0, SuperUnit, p(0, 2), |u| u.origin = Some(CityId(0)))
        .stars(0, 10);
    let s = b.build();
    assert_eq!(supported_units(&s, CityId(0)), 1);
    let train = Action::Train {
        city: CityId(0),
        kind: Warrior,
    };
    assert!(legal_actions(&s).contains(&train), "1 of 2 used");
    let s = StateBuilder::new(5)
        .city(0, p(2, 2))
        .unit_with(0, Warrior, p(0, 0), |u| u.origin = Some(CityId(0)))
        .unit_with(0, Warrior, p(4, 4), |u| u.origin = Some(CityId(0)))
        .unit_with(0, SuperUnit, p(0, 1), |u| u.origin = Some(CityId(0)))
        .stars(0, 10)
        .build();
    assert_eq!(supported_units(&s, CityId(0)), 2);
    assert!(!legal_actions(&s).contains(&train), "2 of 2 used");
}

#[test]
fn a_spawned_super_unit_leaves_supported_units_unchanged() {
    let before = occupied(p(5, 5), 0, Some(E)).build();
    let mut s = before.clone();
    s.units[0].origin = Some(CityId(0));
    let n = supported_units(&s, CityId(0));
    assert_eq!(n, 1);
    spawn_super_unit(&mut s, CityId(0));
    assert_eq!(supported_units(&s, CityId(0)), n);
}

// ---------- menus and ChooseReward ----------

/// 11x11, P0 city 0 at (5,5) pending at `level` with `pop`, a friendly Warrior on it moving E,
/// 10 stars.
fn pending_with_occupant(level: u32, pop: u32) -> State {
    StateBuilder::new(11)
        .city_with(0, p(5, 5), |c| {
            c.level = level;
            c.population = pop;
            c.pending_reward = true;
        })
        .unit_with(0, Warrior, p(5, 5), |u| u.last_dir = Some(E))
        .stars(0, 10)
        .build()
}

#[test]
fn choose_super_unit_through_apply() {
    let before = pending_with_occupant(5, 14);
    assert_eq!(
        legal_actions(&before),
        [choose(0, Reward::Park), choose(0, Reward::SuperUnit)]
    );
    let mut s = before.clone();
    assert_eq!(apply(&mut s, &choose(0, Reward::SuperUnit)), Ok(()));
    let mut want = pushed(&before, 0, p(5, 6), 0);
    want.cities[0].pending_reward = false;
    assert_eq!(s, want);
    // Level 6 and up offer the same menu.
    let s = pending_with_occupant(9, 44);
    assert_eq!(
        legal_actions(&s),
        [choose(0, Reward::Park), choose(0, Reward::SuperUnit)]
    );
}

#[test]
fn the_level_up_check_runs_after_the_spawn() {
    // Level 5, pending, population 20 = cum(6).
    let mut s = pending_with_occupant(5, 20);
    assert_eq!(apply(&mut s, &choose(0, Reward::SuperUnit)), Ok(()));
    let c = &s.cities[0];
    assert_eq!((c.level, c.pending_reward), (6, true));
    assert_eq!(s.tile(p(5, 5)).unit, Some(UnitId(1)), "the spawn happened");
}

#[test]
fn level_two_offers_workshop_or_explorer() {
    let s = pending_with_occupant(2, 2);
    assert_eq!(
        legal_actions(&s),
        [choose(0, Reward::Workshop), choose(0, Reward::Explorer)]
    );
}

#[test]
fn illegal_reward_choices_change_nothing() {
    let rejected = |s: &State, a: Action| {
        let mut t = s.clone();
        assert!(!legal_actions(s).contains(&a), "{a:?} listed");
        assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{a:?}");
        assert_eq!(t, *s, "{a:?} changed the state");
    };
    // Controls: the same choices are legal at the right level.
    let s = pending_with_occupant(5, 14);
    assert!(legal_actions(&s).contains(&choose(0, Reward::SuperUnit)));
    let s = pending_with_occupant(2, 2);
    assert!(legal_actions(&s).contains(&choose(0, Reward::Explorer)));
    for level in [2, 3, 4] {
        let s = pending_with_occupant(level, exp_cum(level));
        rejected(&s, choose(0, Reward::SuperUnit));
    }
    for level in [3, 4, 5, 8] {
        let s = pending_with_occupant(level, exp_cum(level));
        rejected(&s, choose(0, Reward::Explorer));
    }
    // Not pending.
    let mut s = pending_with_occupant(5, 14);
    s.cities[0].pending_reward = false;
    rejected(&s, choose(0, Reward::SuperUnit));
    let mut s = pending_with_occupant(2, 2);
    s.cities[0].pending_reward = false;
    rejected(&s, choose(0, Reward::Explorer));
    // The other player's pending city.
    let mut s = pending_with_occupant(5, 14);
    s.current = 1;
    rejected(&s, choose(0, Reward::SuperUnit));
    let mut s = pending_with_occupant(2, 2);
    s.current = 1;
    rejected(&s, choose(0, Reward::Explorer));
}
