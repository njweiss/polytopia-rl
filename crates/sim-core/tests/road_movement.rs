//! Milestone 16 "Tests" item 4: road movement in `reachable_tiles` under v2 (docs/MILESTONE_16
//! "Road movement"; spec "Roads and city connections").
//!
//! `oracle_reach` is an independent brute-force search over every step sequence, written from the
//! brief: budget `2 x movement` half-points; a step is allowed with at least 1 half-point left; a
//! road step (both tiles road-usable for the mover) costs 1, every other step 2, floored at 0;
//! the target must be explored, empty and allowed by terrain; arriving next to an enemy unit, or
//! on a forest or mountain by a non-road step, stops the unit (the start tile is exempt). Each
//! scripted board also states its expected tiles, worked out by hand in the comment above it.

mod common;

use std::sync::Arc;

use common::{all_pos, cheb, idx, marked, p, random_board, tile_mut};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Tables, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::Terrain;
use sim_core::movement::reachable_tiles;
use sim_core::rng::Rng;
use sim_core::state::{Config, Phase, PlayerId, Ruleset, State, UnitId};
use sim_core::testing::StateBuilder;

fn v2_config() -> Config {
    Config {
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

/// A v2 board from a drawing: `.` field, `f` forest, `m` mountain, `=` road on field, `F` road on
/// forest. Everything explored. Units and cities are added by `extra` before the roads are laid.
fn scene(rows: &[&str], extra: impl FnOnce(StateBuilder) -> StateBuilder) -> State {
    let n = rows.len() as u8;
    let mut b = StateBuilder::new(n);
    let mut roads = Vec::new();
    for (r, line) in rows.iter().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(toks.len(), rows.len(), "square board: {line:?}");
        for (c, t) in toks.iter().enumerate() {
            let q = p(r as u8, c as u8);
            match *t {
                "." => {}
                "f" => b = b.terrain(q, Terrain::Forest),
                "m" => b = b.terrain(q, Terrain::Mountain),
                "=" => roads.push(q),
                "F" => {
                    b = b.terrain(q, Terrain::Forest);
                    roads.push(q);
                }
                other => panic!("unknown token {other:?}"),
            }
        }
    }
    let mut s = extra(b.explore_all()).config(v2_config()).build();
    for q in roads {
        tile_mut(&mut s, q).road = true;
    }
    s
}

fn owner_of_tile(s: &State, q: Pos) -> Option<PlayerId> {
    s.tile(q)
        .owner_city
        .map(|c| s.cities.iter().find(|x| x.id == c).unwrap().owner)
}

/// The brief's "Usable": a road or a city, not in the other player's territory.
fn usable(s: &State, player: PlayerId, q: Pos) -> bool {
    let t = s.tile(q);
    (t.road || t.city.is_some()) && owner_of_tile(s, q).is_none_or(|o| o == player)
}

/// Every tile some allowed step sequence reaches, row-major, without the start tile.
fn oracle_reach(s: &State, id: UnitId) -> Vec<Pos> {
    let u = s.units.iter().find(|u| u.id == id).unwrap().clone();
    let budget = 2 * s.tables.unit(u.kind).movement;
    let climbing = s.players[u.owner as usize].techs.contains(Tech::Climbing);
    let enemies: Vec<Pos> = s
        .units
        .iter()
        .filter(|e| e.owner != u.owner)
        .map(|e| e.pos)
        .collect();
    let enterable = |q: Pos| {
        s.players[u.owner as usize].explored[idx(s, q)]
            && s.units.iter().all(|o| o.pos != q)
            && (s.tile(q).terrain != Terrain::Mountain || climbing)
    };
    let mut found = vec![false; s.tiles.len()];
    let mut stack = vec![(u.pos, budget)];
    // Depth-first over all step sequences: every step spends at least 1, so this ends.
    while let Some((at, left)) = stack.pop() {
        if left < 1 {
            continue;
        }
        for q in all_pos(s.size) {
            if cheb(q, at) != 1 || !enterable(q) {
                continue;
            }
            let road_step = usable(s, u.owner, at) && usable(s, u.owner, q);
            let rest = left.saturating_sub(if road_step { 1 } else { 2 });
            found[idx(s, q)] = true;
            let zoc = enemies.iter().any(|&e| cheb(e, q) == 1);
            let rough = matches!(s.tile(q).terrain, Terrain::Forest | Terrain::Mountain);
            if !(zoc || (rough && !road_step)) {
                stack.push((q, rest));
            }
        }
    }
    all_pos(s.size)
        .into_iter()
        .filter(|&q| q != u.pos && found[idx(s, q)])
        .collect()
}

fn unit_at(s: &State, q: Pos) -> UnitId {
    s.units.iter().find(|u| u.pos == q).unwrap().id
}

/// `reachable_tiles` of the unit on `at` equals both the hand-drawn `expected` (`x` marks) and the
/// oracle.
fn check(s: &State, at: Pos, expected: &[&str]) {
    let id = unit_at(s, at);
    let want = marked(expected, "x");
    assert_eq!(
        oracle_reach(s, id),
        want,
        "oracle disagrees with the drawing"
    );
    assert_eq!(reachable_tiles(s, id), want);
}

// ---------------------------------------------------------------- scripted boards

/// Warrior (movement 1, budget 2 half-points) on (3,1), roads (3,1) to (3,4). Two road steps
/// reach (3,3) with nothing left; (3,4) would need a third. After one road step (1 left) a normal
/// step still goes anywhere (costs round up), so the neighbors of (3,2) are reached too.
#[test]
fn a_warrior_takes_two_road_steps_but_not_three() {
    let s = scene(
        &[
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". = = = = . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
        |b| b.unit(0, UnitType::Warrior, p(3, 1)),
    );
    check(
        &s,
        p(3, 1),
        &[
            ". . . . . . .",
            ". . . . . . .",
            "x x x x . . .",
            "x . x x . . .",
            "x x x x . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
    );
}

/// Rider (movement 2, budget 4) on (3,1), roads along all of row 3. Four road steps reach (3,5)
/// with 0 left, so (3,6) is out. Normal steps: from (3,1) (4 left) two of them cover rows 1-5,
/// cols 0-3; from (3,2) (3 left) a normal step leaves 1, so another reaches col 4 in rows 1-5;
/// from (3,4) (1 left) a normal step reaches (2,5) and (4,5). Nothing reaches col 4 with 1 left
/// except via (3,4) itself, so (1,5) and (5,5) are out.
#[test]
fn a_rider_takes_four_road_steps() {
    let s = scene(
        &[
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            "= = = = = = =",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
        |b| b.unit(0, UnitType::Rider, p(3, 1)),
    );
    check(
        &s,
        p(3, 1),
        &[
            ". . . . . . .",
            "x x x x x . .",
            "x x x x x x .",
            "x . x x x x .",
            "x x x x x x .",
            "x x x x x . .",
            ". . . . . . .",
        ],
    );
}

/// Warrior on (3,1), roads (3,1) and (3,2) only. The road step leaves 1 half-point (0.5
/// movement), which still pays for a normal step (costs round up): (3,2)'s neighbors in col 3 are
/// reached, and the final tile needs no road.
#[test]
fn a_normal_step_after_a_road_step_with_half_a_point_left() {
    let s = scene(
        &[
            ". . . . . .",
            ". . . . . .",
            ". . . . . .",
            ". = = . . .",
            ". . . . . .",
            ". . . . . .",
        ],
        |b| b.unit(0, UnitType::Warrior, p(3, 1)),
    );
    check(
        &s,
        p(3, 1),
        &[
            ". . . . . .",
            ". . . . . .",
            "x x x x . .",
            "x . x x . .",
            "x x x x . .",
            ". . . . . .",
        ],
    );
}

/// Warrior on (3,1), roads on (3,1), the forest (3,2) and (3,3). The road step onto the forest
/// does not stop the unit, so it continues from (3,2) with 1 left.
#[test]
fn a_road_step_onto_a_forest_does_not_stop() {
    let s = scene(
        &[
            ". . . . . .",
            ". . . . . .",
            ". . . . . .",
            ". = F = . .",
            ". . . . . .",
            ". . . . . .",
        ],
        |b| b.unit(0, UnitType::Warrior, p(3, 1)),
    );
    check(
        &s,
        p(3, 1),
        &[
            ". . . . . .",
            ". . . . . .",
            "x x x x . .",
            "x . x x . .",
            "x x x x . .",
            ". . . . . .",
        ],
    );
}

/// A corridor (mountains, no Climbing) along row 3. Rider on (3,1). In the first board (3,1) has a
/// road but the forest (3,2) has none; in the second the forest has a road but (3,1) has none.
/// Either way the step onto the forest is a normal step and stops there, so (3,3) is out even
/// with budget left. Control: with both roads, the road steps (3,2), (3,3), (3,4) leave 1, and a
/// normal step reaches (3,5).
#[test]
fn a_normal_step_onto_a_forest_stops() {
    let expected = [
        "m m m m m m",
        "m m m m m m",
        "m m m m m m",
        "x . x . . .",
        "m m m m m m",
        "m m m m m m",
    ];
    for row in [". = f = = .", ". . F = = ."] {
        let rows = [
            "m m m m m m",
            "m m m m m m",
            "m m m m m m",
            row,
            "m m m m m m",
            "m m m m m m",
        ];
        let s = scene(&rows, |b| b.unit(0, UnitType::Rider, p(3, 1)));
        check(&s, p(3, 1), &expected);
    }
    let mut rows = expected;
    rows[3] = ". = F = = .";
    let s = scene(&rows, |b| b.unit(0, UnitType::Rider, p(3, 1)));
    let mut control = expected;
    control[3] = "x . x x x x";
    check(&s, p(3, 1), &control);
}

/// Warrior on (3,1), roads (3,1) to (3,4), an enemy Warrior on (2,3). The road step onto (3,2),
/// next to the enemy, stops the unit (Zone of Control applies to road steps), so only the 8
/// neighbors of (3,1) are reached. Control: with the enemy on (0,6) instead, far away, the two road
/// steps of `a_warrior_takes_two_road_steps_but_not_three` come back.
#[test]
fn zone_of_control_stops_a_road_step() {
    let rows = [
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". = = = = . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
    ];
    let far = scene(&rows, |b| {
        b.unit(0, UnitType::Warrior, p(3, 1))
            .unit(1, UnitType::Warrior, p(0, 6))
    });
    check(
        &far,
        p(3, 1),
        &[
            ". . . . . . .",
            ". . . . . . .",
            "x x x x . . .",
            "x . x x . . .",
            "x x x x . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
    );
    let s = scene(&rows, |b| {
        b.unit(0, UnitType::Warrior, p(3, 1))
            .unit(1, UnitType::Warrior, p(2, 3))
    });
    check(
        &s,
        p(3, 1),
        &[
            ". . . . . . .",
            ". . . . . . .",
            "x x x . . . .",
            "x . x . . . .",
            "x x x . . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
    );
}

/// Player 0's city on (1,1) owns rows 0-2 x cols 0-2; player 1's on (5,5) owns rows 4-6 x cols
/// 4-6. Roads on row 0 cols 0-2 (player 0's land) and row 6 cols 4-6 (player 1's land).
fn territory_board(units: impl FnOnce(StateBuilder) -> StateBuilder) -> State {
    scene(
        &[
            "= = = . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . = = =",
        ],
        |b| units(b.city(0, p(1, 1)).city(1, p(5, 5))),
    )
}

/// A Warrior on a road in enemy territory gets no discount: player 1's Warrior on (0,0) and
/// player 0's Warrior on (6,4) reach only their plain neighbors (an empty enemy city tile can be
/// entered). Control: with row 6's road tiles made neutral, player 0's Warrior takes the road step
/// to (6,5) (1 left) and reaches (5,6) and (6,6); the step onto player 1's city (5,5) stays normal.
#[test]
fn roads_in_enemy_territory_work_for_neither_player() {
    let s = territory_board(|b| {
        b.unit(1, UnitType::Warrior, p(0, 0))
            .unit(0, UnitType::Warrior, p(6, 4))
    });
    check(
        &s,
        p(0, 0),
        &[
            ". x . . . . .",
            "x x . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
    );
    check(
        &s,
        p(6, 4),
        &[
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . x x x .",
            ". . . x . x .",
        ],
    );
    let mut t = s.clone();
    for q in [p(6, 4), p(6, 5), p(6, 6)] {
        tile_mut(&mut t, q).owner_city = None;
    }
    check(
        &t,
        p(6, 4),
        &[
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . x x x x",
            ". . . x . x x",
        ],
    );
}

/// The same roads for their owners: player 0's Warrior on (0,0) takes road steps to (0,1) and to
/// its city (1,1) (a city counts as a road), 1 left on each; player 1's on (6,4) likewise to
/// (6,5) and (5,5).
#[test]
fn own_roads_and_own_cities_work_for_their_owner() {
    let s = territory_board(|b| {
        b.unit(0, UnitType::Warrior, p(0, 0))
            .unit(1, UnitType::Warrior, p(6, 4))
    });
    check(
        &s,
        p(0, 0),
        &[
            ". x x . . . .",
            "x x x . . . .",
            "x x x . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
    );
    check(
        &s,
        p(6, 4),
        &[
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . x x x",
            ". . . x x x x",
            ". . . x . x x",
        ],
    );
}

/// Neutral roads (3,1) to (3,3), no cities: a Warrior of either player on (3,1) gets the same
/// two road steps.
#[test]
fn neutral_roads_work_for_both_players() {
    let rows = [
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". = = = . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
    ];
    let expected = [
        ". . . . . . .",
        ". . . . . . .",
        "x x x x . . .",
        "x . x x . . .",
        "x x x x . . .",
        ". . . . . . .",
        ". . . . . . .",
    ];
    for owner in 0..2u8 {
        let s = scene(&rows, |b| b.unit(owner, UnitType::Warrior, p(3, 1)));
        check(&s, p(3, 1), &expected);
    }
}

/// A Warrior on its own city (3,3) with a road on (3,4): the step city -> road is a road step
/// (1 left), so (3,4)'s far neighbors in col 5 are reached. Second board: a second own city on
/// (3,4) instead of the road (a city -> city step also costs 0.5, R2).
#[test]
fn a_city_counts_as_a_road() {
    let expected = [
        ". . . . . . .",
        ". . . . . . .",
        ". . x x x x .",
        ". . x . x x .",
        ". . x x x x .",
        ". . . . . . .",
        ". . . . . . .",
    ];
    let plain = [
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
        ". . . . . . .",
    ];
    let mut road_rows = plain;
    road_rows[3] = ". . . . = . .";
    let s = scene(&road_rows, |b| {
        b.city(0, p(3, 3)).unit(0, UnitType::Warrior, p(3, 3))
    });
    check(&s, p(3, 3), &expected);

    let t = scene(&plain, |b| {
        b.city(0, p(3, 3))
            .city(0, p(3, 4))
            .unit(0, UnitType::Warrior, p(3, 3))
    });
    check(&t, p(3, 3), &expected);
}

/// Unexplored and occupied road tiles cannot be entered; a mountain road needs Climbing (roads
/// are never built on mountains, but the movement rule does not depend on that).
#[test]
fn road_tiles_must_still_be_enterable() {
    let s = {
        let mut s = scene(
            &[
                ". . . . . . .",
                ". . . . . . .",
                ". . . . . . .",
                ". = = = = . .",
                ". . . . . . .",
                ". . . . . . .",
                ". . . . . . .",
            ],
            |b| {
                b.unit(0, UnitType::Rider, p(3, 1))
                    .unit(0, UnitType::Warrior, p(3, 3))
            },
        );
        let i = idx(&s, p(2, 2));
        s.players[0].explored[i] = false;
        s
    };
    // (3,3) is occupied, (2,2) unexplored: the oracle settles the rest.
    let id = unit_at(&s, p(3, 1));
    let got = reachable_tiles(&s, id);
    assert!(!got.contains(&p(3, 3)) && !got.contains(&p(2, 2)));
    assert_eq!(got, oracle_reach(&s, id));
}

// ---------------------------------------------------------------- Escape

/// Rider on (3,1), roads (3,1) to (3,6), an enemy Warrior on (2,0). The Rider attacks (both end
/// on 5 health: 2 vs 2 at full health gives round(4.5) = 5 each way), then its Escape move uses
/// the roads: (3,5) is four road steps away.
#[test]
fn the_escape_move_after_an_attack_uses_roads() {
    let mut s = scene(
        &[
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
            ". = = = = = =",
            ". . . . . . .",
            ". . . . . . .",
            ". . . . . . .",
        ],
        |b| {
            b.unit(0, UnitType::Rider, p(3, 1))
                .unit(1, UnitType::Warrior, p(2, 0))
        },
    );
    let rider = unit_at(&s, p(3, 1));
    apply(
        &mut s,
        &Action::Attack {
            unit: rider,
            target: p(2, 0),
        },
    )
    .unwrap();
    let r = s.units.iter().find(|u| u.id == rider).unwrap();
    assert_eq!((r.phase, r.health, r.pos), (Phase::Attacked, 5, p(3, 1)));
    let moves: Vec<Pos> = legal_actions(&s)
        .into_iter()
        .filter_map(|a| match a {
            Action::Move { unit, to } if unit == rider => Some(to),
            _ => None,
        })
        .collect();
    assert!(moves.contains(&p(3, 5)), "{moves:?}");
    assert!(!moves.contains(&p(3, 6)));
    assert_eq!(moves, oracle_reach(&s, rider));
}

// ---------------------------------------------------------------- property

/// Random v2 boards (the milestone 3 generator: terrain, cities of both players, villages, units
/// of every kind, partial fog, random techs) with roads on a random third of the non-mountain
/// tiles: `reachable_tiles` equals the oracle for every unit of both players.
#[test]
fn reachable_tiles_matches_the_oracle_on_random_v2_boards() {
    let mut longer = 0;
    for seed in 0..300u64 {
        let mut s = random_board(seed);
        s.config.ruleset = Ruleset::V2;
        s.tables = Arc::new(Tables::load_for(Ruleset::V2).unwrap());
        let mut rng = Rng::new(seed ^ 0x16_4004);
        for q in all_pos(s.size) {
            let t = s.tile(q);
            if t.terrain != Terrain::Mountain && t.city.is_none() && rng.gen_range(3) != 0 {
                tile_mut(&mut s, q).road = true;
            }
        }
        for u in s.units.clone() {
            let want = oracle_reach(&s, u.id);
            let movement = s.tables.unit(u.kind).movement;
            if want.iter().any(|&q| cheb(q, u.pos) > movement) {
                longer += 1;
            }
            assert_eq!(
                reachable_tiles(&s, u.id),
                want,
                "seed {seed}, unit {:?}",
                u.id
            );
        }
    }
    // The boards exercise roads: many units reach farther than their v1 range.
    assert!(longer > 100, "only {longer} units used roads");
}
