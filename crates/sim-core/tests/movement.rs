//! Movement goldens and the defense bonus table (docs/MILESTONE_3 "reachable_tiles",
//! "defense_bonus"; spec "Movement"). Boards are drawn by hand; every expected set is drawn by
//! hand too and is also checked against the test's own brute-force path enumerator
//! (`common::bf_reach`), never against `reachable_tiles` itself.
//!
//! Board tokens: `.` field, `f` forest, `m` mountain, `v` village. Expected grids: `*` marks a
//! reachable tile; `U` is the mover, `e` an enemy unit, `u` a friendly unit, `.` not reachable.

mod common;

use common::*;
use sim_core::data::{Tech, UnitType};
use sim_core::map::Terrain;
use sim_core::movement::{defense_bonus, reachable_tiles};
use sim_core::state::{Phase, State, UnitId};
use sim_core::testing::StateBuilder;
use UnitType::*;

fn assert_reach(s: &State, id: u32, expected: &[&str]) {
    let want = marked(expected, "*");
    assert_eq!(
        bf_reach(s, UnitId(id)),
        want,
        "test bug: brute force disagrees with the drawn golden"
    );
    assert_eq!(reachable_tiles(s, UnitId(id)), want);
}

/// Row 2 is an open field corridor; everything else is mountain (impassable without Climbing).
const CORRIDOR: [&str; 5] = [
    "m m m m m",
    "m m m m m",
    ". . . . .",
    "m m m m m",
    "m m m m m",
];

fn corridor() -> StateBuilder {
    board(&CORRIDOR)
}

const OPEN5: [&str; 5] = [
    ". . . . .",
    ". . . . .",
    ". . . . .",
    ". . . . .",
    ". . . . .",
];

// ---------- open field ----------

#[test]
fn warrior_open_field_reaches_its_eight_neighbors() {
    let s = board(&OPEN5)
        .unit(0, Warrior, p(2, 2))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". * * * .",
            ". * U * .",
            ". * * * .",
            ". . . . .",
        ],
    );
}

#[test]
fn warrior_in_corner_is_bounded_by_the_map() {
    let s = board(&OPEN5)
        .unit(0, Warrior, p(0, 0))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            "U * . . .",
            "* * . . .",
            ". . . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn rider_open_field_reaches_chebyshev_two() {
    let s = board(&[
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
    ])
    .unit(0, Rider, p(3, 3))
    .explore_all()
    .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . . . .",
            ". * * * * * .",
            ". * * * * * .",
            ". * * U * * .",
            ". * * * * * .",
            ". * * * * * .",
            ". . . . . . .",
        ],
    );
}

#[test]
fn rider_in_corridor_baseline() {
    let s = corridor().unit(0, Rider, p(2, 0)).explore_all().build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U * * . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn reach_ignores_phase_and_whose_turn_it_is() {
    // Pure geometry for the unit's owner: a Done unit of the non-current player still has reach.
    let s = corridor()
        .unit_with(1, Rider, p(2, 0), |u| u.phase = Phase::Done)
        .explore_all()
        .current(0)
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U * * . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

// ---------- occupancy ----------

#[test]
fn friendly_unit_blocks_the_corridor() {
    let s = corridor()
        .unit(0, Rider, p(2, 0))
        .unit(0, Warrior, p(2, 1))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U u . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn enemy_unit_blocks_the_corridor() {
    let s = corridor()
        .unit(0, Rider, p(2, 0))
        .unit(1, Warrior, p(2, 1))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U e . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn detour_around_a_friendly_unit() {
    let s = board(&[
        "m m m m m",
        "m . m m m",
        ". . . . .",
        "m m m m m",
        "m m m m m",
    ])
    .unit(0, Rider, p(2, 0))
    .unit(0, Warrior, p(2, 1))
    .explore_all()
    .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". * . . .",
            "U u * . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn empty_village_and_cities_are_enterable_and_passable() {
    // Village at (2,1), enemy city at (2,2): the Rider passes through the village into the city.
    let s = corridor()
        .village(p(2, 1))
        .city(1, p(2, 2))
        .unit(0, Rider, p(2, 0))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U * * . .",
            ". . . . .",
            ". . . . .",
        ],
    );
    // Own city too.
    let s = board(&OPEN5)
        .village(p(1, 1))
        .city(1, p(1, 3))
        .city(0, p(3, 3))
        .unit(0, Warrior, p(2, 2))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". * * * .",
            ". * U * .",
            ". * * * .",
            ". . . . .",
        ],
    );
}

#[test]
fn occupied_village_and_city_are_not_enterable() {
    let s = board(&OPEN5)
        .village(p(1, 1))
        .city(1, p(1, 3))
        .unit(0, Warrior, p(2, 2))
        .unit(0, Warrior, p(1, 1))
        .unit(1, Warrior, p(1, 3))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". u * e .",
            ". * U * .",
            ". * * * .",
            ". . . . .",
        ],
    );
}

// ---------- rough terrain ----------

#[test]
fn forest_stops_on_arrival() {
    let s = corridor()
        .terrain(p(2, 1), Terrain::Forest)
        .unit(0, Rider, p(2, 0))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U * . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn starting_in_forest_does_not_stop() {
    let s = corridor()
        .terrain(p(2, 0), Terrain::Forest)
        .unit(0, Rider, p(2, 0))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U * * . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn mountain_without_climbing_is_impassable() {
    let s = corridor()
        .terrain(p(2, 1), Terrain::Mountain)
        .unit(0, Rider, p(2, 0))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U . . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn mountain_with_climbing_is_entered_and_stops() {
    // With Climbing every corridor wall becomes enterable, but each one stops the unit.
    let s = corridor()
        .terrain(p(2, 1), Terrain::Mountain)
        .unit(0, Rider, p(2, 0))
        .techs(0, &[Tech::Climbing])
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            "* * . . .",
            "U * . . .",
            "* * . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn climbing_is_the_owners_tech() {
    // The opponent having Climbing does not help.
    let s = corridor()
        .terrain(p(2, 1), Terrain::Mountain)
        .unit(0, Rider, p(2, 0))
        .techs(1, &[Tech::Climbing])
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U . . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn starting_on_a_mountain_does_not_stop() {
    let s = corridor()
        .terrain(p(2, 0), Terrain::Mountain)
        .unit(0, Rider, p(2, 0))
        .techs(0, &[Tech::Climbing])
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            "* * * . .",
            "U * * . .",
            "* * * . .",
            ". . . . .",
        ],
    );
}

// ---------- zone of control ----------

#[test]
fn zoc_stops_on_arrival_diagonal_adjacency() {
    // Enemy at (1,2) is diagonally adjacent to (2,1).
    let s = corridor()
        .unit(0, Rider, p(2, 0))
        .unit(1, Warrior, p(1, 2))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . e . .",
            "U * . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn zoc_stops_on_arrival_orthogonal_adjacency() {
    // Enemy at (1,1) is directly above (2,1). The start tile is also adjacent (exempt).
    let s = corridor()
        .unit(0, Rider, p(2, 0))
        .unit(1, Warrior, p(1, 1))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". e . . .",
            "U * . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn leaving_zoc_uses_the_full_budget() {
    // Starts adjacent to the enemy at (1,0); (2,2) is not adjacent to it, so (2,3) is reachable.
    let s = corridor()
        .unit(0, Rider, p(2, 1))
        .unit(1, Warrior, p(1, 0))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            "e . . . .",
            "* U * * .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn zoc_open_field_leave_and_reenter() {
    // Rider at (2,2) starts in the ZoC of the enemy at (1,3). ZoC tiles: rows 0-2, cols 2-4.
    // (2,4) is reached by leaving to (3,3) and re-entering. (0,3), (1,4) and (0,4) are only
    // reachable through ZoC tiles or the enemy's tile, so they are not reachable.
    let s = board(&OPEN5)
        .unit(0, Rider, p(2, 2))
        .unit(1, Warrior, p(1, 3))
        .explore_all()
        .build();
    assert_reach(
        &s,
        0,
        &[
            "* * * . .",
            "* * * e .",
            "* * U * *",
            "* * * * *",
            "* * * * *",
        ],
    );
}

#[test]
fn zoc_comes_only_from_enemy_units() {
    // Empty enemy city at (1,2) and a friendly unit at (3,1) are both adjacent to (2,1);
    // neither stops the Rider. The empty enemy city is itself enterable.
    let s = board(&[
        "m m m m m",
        "m m . m m",
        ". . . . .",
        "m m m m m",
        "m m m m m",
    ])
    .city(1, p(1, 2))
    .unit(0, Rider, p(2, 0))
    .unit(0, Warrior, p(3, 1))
    .explore_all()
    .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . * . .",
            "U * * . .",
            ". u . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn zoc_is_relative_to_the_movers_owner() {
    // Player 1's Rider: the player 0 Warrior is the enemy now.
    let s = corridor()
        .unit(1, Rider, p(2, 0))
        .unit(0, Warrior, p(1, 2))
        .explore_all()
        .current(1)
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . e . .",
            "U * . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

// ---------- fog ----------

#[test]
fn unexplored_destination_is_not_reachable() {
    // (2,2) would be revealed by stepping to (2,1), but reach uses the mask before the move.
    let s = corridor()
        .unit(0, Rider, p(2, 0))
        .explore_all()
        .hide(0, &[p(2, 2)])
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U * . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn unexplored_tile_cannot_be_passed_through() {
    let s = corridor()
        .unit(0, Rider, p(2, 0))
        .explore_all()
        .hide(0, &[p(2, 1)])
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U . . . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

#[test]
fn fog_uses_the_owners_mask() {
    // Hidden only from player 1; player 0's Rider is unaffected.
    let s = corridor()
        .unit(0, Rider, p(2, 0))
        .explore_all()
        .hide(1, &[p(2, 1), p(2, 2)])
        .build();
    assert_reach(
        &s,
        0,
        &[
            ". . . . .",
            ". . . . .",
            "U * * . .",
            ". . . . .",
            ". . . . .",
        ],
    );
}

// ---------- randomized cross-check ----------

#[test]
fn reachable_tiles_match_brute_force_on_random_boards() {
    for seed in 0..600 {
        let s = random_board(seed);
        for u in &s.units {
            assert_eq!(
                reachable_tiles(&s, u.id),
                bf_reach(&s, u.id),
                "seed {seed} unit {:?} {:?} at {:?}",
                u.id,
                u.kind,
                u.pos
            );
        }
    }
}

// ---------- defense bonus ----------

/// One unit of `kind` for player 0 at (1,1) of a 3x3 board shaped by `setup`.
fn bonus_of(kind: UnitType, setup: impl FnOnce(StateBuilder) -> StateBuilder) -> f32 {
    let s = setup(StateBuilder::new(3))
        .unit(0, kind, p(1, 1))
        .explore_all()
        .build();
    defense_bonus(&s, UnitId(0))
}

#[test]
fn defense_bonus_terrain() {
    let c = p(1, 1);
    assert_eq!(bonus_of(Warrior, |b| b), 1.0, "field");
    assert_eq!(
        bonus_of(Warrior, |b| b.terrain(c, Terrain::Forest)),
        1.0,
        "forest without Archery"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b
            .terrain(c, Terrain::Forest)
            .techs(0, &[Tech::Archery])),
        1.5,
        "forest with Archery"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b.terrain(c, Terrain::Mountain)),
        1.0,
        "mountain without Climbing"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b
            .terrain(c, Terrain::Mountain)
            .techs(0, &[Tech::Climbing])),
        1.5,
        "mountain with Climbing"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b
            .terrain(c, Terrain::Forest)
            .techs(0, &[Tech::Climbing])),
        1.0,
        "forest needs Archery, not Climbing"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b
            .terrain(c, Terrain::Mountain)
            .techs(0, &[Tech::Archery])),
        1.0,
        "mountain needs Climbing, not Archery"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b
            .terrain(c, Terrain::Forest)
            .techs(1, &[Tech::Archery])),
        1.0,
        "the defender's techs decide, not the opponent's"
    );
    assert_eq!(
        bonus_of(SuperUnit, |b| b
            .terrain(c, Terrain::Mountain)
            .techs(0, &[Tech::Climbing])),
        1.5,
        "terrain bonus does not need Fortify"
    );
}

#[test]
fn defense_bonus_cities() {
    let c = p(1, 1);
    assert_eq!(bonus_of(Warrior, |b| b.city(0, c)), 1.5, "own city");
    assert_eq!(
        bonus_of(Defender, |b| b.city(0, c)),
        1.5,
        "Defender in own city"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b.city_with(0, c, |x| x.wall = true)),
        4.0,
        "own walled city"
    );
    assert_eq!(bonus_of(SuperUnit, |b| b.city(0, c)), 1.0, "no Fortify");
    assert_eq!(
        bonus_of(SuperUnit, |b| b.city_with(0, c, |x| x.wall = true)),
        1.0,
        "no Fortify, wall"
    );
    assert_eq!(bonus_of(Warrior, |b| b.city(1, c)), 1.0, "enemy city");
    assert_eq!(
        bonus_of(Warrior, |b| b.city_with(1, c, |x| x.wall = true)),
        1.0,
        "enemy walled city"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b.city(0, p(0, 0))),
        1.0,
        "own territory but not the city tile"
    );
}

#[test]
fn defense_bonus_takes_the_largest_value_without_stacking() {
    let c = p(1, 1);
    assert_eq!(
        bonus_of(Warrior, |b| b
            .terrain(c, Terrain::Mountain)
            .techs(0, &[Tech::Climbing])
            .city(0, c)),
        1.5,
        "mountain + own city: 1.5, not 2.25"
    );
    assert_eq!(
        bonus_of(Warrior, |b| b
            .terrain(c, Terrain::Forest)
            .techs(0, &[Tech::Archery])
            .city_with(0, c, |x| x.wall = true)),
        4.0,
        "forest + walled city: 4.0, not 6.0"
    );
}

#[test]
fn defense_bonus_matches_the_rule_on_random_boards() {
    for seed in 0..600 {
        let s = random_board(seed);
        for u in &s.units {
            assert_eq!(
                defense_bonus(&s, u.id),
                expected_bonus(&s, u.id),
                "seed {seed} unit {:?}",
                u.id
            );
        }
    }
}
