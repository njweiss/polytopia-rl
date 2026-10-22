//! Milestone 8: `PlayerView` per-field goldens on a scripted board (docs/MILESTONE_8 "PlayerView
//! visibility"; "Tests" item 1). Expected values are written out by hand from the board; only
//! `legal_actions`, `reachable_tiles` and `score`, which the view must pass through unchanged,
//! are compared to the existing `sim_core` functions.

use std::panic::AssertUnwindSafe;

use sim_core::actions::legal_actions;
use sim_core::data::{Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::{Resource, Terrain};
use sim_core::movement::reachable_tiles;
use sim_core::score::score;
use sim_core::state::{CityId, Config, Dir, Improvement, Outcome, Phase, State, TechSet, UnitId};
use sim_core::testing::StateBuilder;
use sim_core::view::{CityView, PlayerView, UnitView, VisibleTile};

fn p(row: u8, col: u8) -> Pos {
    Pos::new(row, col)
}

fn rect(r0: u8, r1: u8, c0: u8, c1: u8) -> Vec<Pos> {
    (r0..=r1)
        .flat_map(|r| (c0..=c1).map(move |c| Pos::new(r, c)))
        .collect()
}

/// Player 0 has explored everything except rows 0..=2 x columns 4..=6 and (5,1); player 1 has
/// explored everything.
///
/// Cities: 0 = P0 capital (1,1), level 2, workshop; 1 = P1 (5,5), level 3, wall, 1 park,
/// population 7; 2 = P1 (1,5) (unexplored for P0); 3 = P0 (5,1), level 2 (unexplored for P0).
/// Units: 0 = P0 Warrior (1,1) 7/10 Moved; 1 = P1 veteran Rider (4,4) 12/15 Done, 3 kills;
/// 2 = P1 Warrior (0,6) (unexplored for P0); 3 = P0 Archer (3,0); 4 = P1 Warrior (2,2).
/// Tiles: (0,0) fruit; (0,1) forest + animal; (2,1) crop + farm; (3,2) mountain + metal;
/// (3,3) village; (0,5) village (unexplored for P0).
fn board(techs0: &[Tech]) -> StateBuilder {
    StateBuilder::new(7)
        .config(Config {
            turn_cap: 30,
            ..Config::default()
        })
        .turn(4)
        .resource(p(0, 0), Resource::Fruit)
        .terrain(p(0, 1), Terrain::Forest)
        .resource(p(0, 1), Resource::Animal)
        .resource(p(2, 1), Resource::Crop)
        .improvement(p(2, 1), Improvement::Farm)
        .terrain(p(3, 2), Terrain::Mountain)
        .resource(p(3, 2), Resource::Metal)
        .village(p(3, 3))
        .village(p(0, 5))
        .city_with(0, p(1, 1), |c| {
            c.level = 2;
            c.population = 3;
            c.is_capital = true;
            c.workshop = true;
        })
        .city_with(1, p(5, 5), |c| {
            c.level = 3;
            c.population = 7;
            c.wall = true;
            c.parks = 1;
        })
        .city(1, p(1, 5))
        .city_with(0, p(5, 1), |c| c.level = 2)
        .unit_with(0, UnitType::Warrior, p(1, 1), |u| {
            u.health = 7;
            u.phase = Phase::Moved;
            u.origin = Some(CityId(0));
        })
        .unit_with(1, UnitType::Rider, p(4, 4), |u| {
            u.health = 12;
            u.max_health = 15;
            u.veteran = true;
            u.kills = 3;
            u.phase = Phase::Done;
            u.origin = Some(CityId(1));
            u.last_dir = Some(Dir { dr: 1, dc: 0 });
        })
        .unit(1, UnitType::Warrior, p(0, 6))
        .unit(0, UnitType::Archer, p(3, 0))
        .unit(1, UnitType::Warrior, p(2, 2))
        .explore_all()
        .hide(0, &rect(0, 2, 4, 6))
        .hide(0, &[p(5, 1)])
        .techs(0, techs0)
        .techs(1, &[Tech::Organization, Tech::Climbing, Tech::Riding])
        .stars(0, 7)
        .stars(1, 13)
}

fn city_view(id: u32, owner: u8, pos: Pos, level: u32, is_capital: bool) -> CityView {
    CityView {
        id: CityId(id),
        owner,
        pos,
        level,
        is_capital,
    }
}

fn own_warrior() -> UnitView {
    UnitView {
        id: Some(UnitId(0)),
        kind: UnitType::Warrior,
        owner: 0,
        pos: p(1, 1),
        health: 7,
        max_health: 10,
        veteran: false,
        phase: Phase::Moved,
    }
}

fn enemy_rider() -> UnitView {
    UnitView {
        id: None,
        kind: UnitType::Rider,
        owner: 1,
        pos: p(4, 4),
        health: 12,
        max_health: 15,
        veteran: true,
        phase: Phase::Done,
    }
}

fn plain(unit: Option<UnitView>, territory: Option<u8>) -> VisibleTile {
    VisibleTile {
        terrain: Terrain::Field,
        resource: None,
        improvement: None,
        village: false,
        city: None,
        territory,
        unit,
    }
}

#[test]
fn unexplored_tiles_are_none() {
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    for q in [p(0, 4), p(0, 5), p(0, 6), p(1, 5), p(2, 6), p(5, 1)] {
        assert!(!v.is_explored(q), "{q:?}");
        assert_eq!(v.tile(q), None, "{q:?}");
    }
    for q in [p(0, 0), p(0, 3), p(3, 3), p(4, 4), p(6, 6), p(5, 0)] {
        assert!(v.is_explored(q), "{q:?}");
        assert!(v.tile(q).is_some(), "{q:?}");
    }
}

#[test]
fn crops_need_organization_and_metal_needs_climbing() {
    let farm_tile = |resource| VisibleTile {
        resource,
        improvement: Some(Improvement::Farm),
        ..plain(None, Some(0))
    };
    let metal_tile = |resource| VisibleTile {
        terrain: Terrain::Mountain,
        resource,
        ..plain(None, None)
    };

    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(v.tile(p(2, 1)), Some(farm_tile(None)));
    assert_eq!(v.tile(p(3, 2)), Some(metal_tile(None)));

    let s = board(&[Tech::Organization]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(v.tile(p(2, 1)), Some(farm_tile(Some(Resource::Crop))));
    assert_eq!(v.tile(p(3, 2)), Some(metal_tile(None)));

    let s = board(&[Tech::Climbing]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(v.tile(p(2, 1)), Some(farm_tile(None)));
    assert_eq!(v.tile(p(3, 2)), Some(metal_tile(Some(Resource::Metal))));

    // The opponent's techs do not matter: player 1 has Organization and Climbing.
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(v.tile(p(3, 2)).unwrap().resource, None);
    let v1 = PlayerView::new(&s, 1);
    assert_eq!(v1.tile(p(3, 2)).unwrap().resource, Some(Resource::Metal));
}

#[test]
fn fruit_and_animals_are_always_visible() {
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(
        v.tile(p(0, 0)),
        Some(VisibleTile {
            resource: Some(Resource::Fruit),
            ..plain(None, Some(0))
        })
    );
    assert_eq!(
        v.tile(p(0, 1)),
        Some(VisibleTile {
            terrain: Terrain::Forest,
            resource: Some(Resource::Animal),
            ..plain(None, Some(0))
        })
    );
}

#[test]
fn units_cities_villages_and_territory_on_explored_tiles() {
    let s = board(&[Tech::Organization]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(
        v.tile(p(1, 1)),
        Some(VisibleTile {
            city: Some(city_view(0, 0, p(1, 1), 2, true)),
            ..plain(Some(own_warrior()), Some(0))
        })
    );
    assert_eq!(v.tile(p(4, 4)), Some(plain(Some(enemy_rider()), Some(1))));
    assert_eq!(
        v.tile(p(5, 5)),
        Some(VisibleTile {
            city: Some(city_view(1, 1, p(5, 5), 3, false)),
            ..plain(None, Some(1))
        })
    );
    assert_eq!(
        v.tile(p(3, 3)),
        Some(VisibleTile {
            village: true,
            ..plain(None, None)
        })
    );
    // (4,0) belongs to the unexplored own city 3.
    assert_eq!(v.tile(p(4, 0)), Some(plain(None, Some(0))));
}

#[test]
fn visible_units_row_major_with_ids_only_for_own_units() {
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    let enemy_warrior = UnitView {
        id: None,
        kind: UnitType::Warrior,
        owner: 1,
        pos: p(2, 2),
        health: 10,
        max_health: 10,
        veteran: false,
        phase: Phase::Fresh,
    };
    let own_archer = UnitView {
        id: Some(UnitId(3)),
        kind: UnitType::Archer,
        owner: 0,
        pos: p(3, 0),
        health: 10,
        max_health: 10,
        veteran: false,
        phase: Phase::Fresh,
    };
    assert_eq!(
        v.visible_units(),
        vec![
            own_warrior(),
            enemy_warrior.clone(),
            own_archer.clone(),
            enemy_rider()
        ]
    );

    // Player 1 sees everything; its own units carry ids.
    let v1 = PlayerView::new(&s, 1);
    let hidden_warrior = UnitView {
        id: Some(UnitId(2)),
        pos: p(0, 6),
        ..enemy_warrior.clone()
    };
    assert_eq!(
        v1.visible_units(),
        vec![
            hidden_warrior,
            UnitView {
                id: None,
                ..own_warrior()
            },
            UnitView {
                id: Some(UnitId(4)),
                ..enemy_warrior
            },
            UnitView {
                id: None,
                ..own_archer
            },
            UnitView {
                id: Some(UnitId(1)),
                ..enemy_rider()
            },
        ]
    );
}

#[test]
fn visible_cities_own_always_enemy_when_explored() {
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(
        v.visible_cities(),
        vec![
            city_view(0, 0, p(1, 1), 2, true),
            city_view(1, 1, p(5, 5), 3, false),
            city_view(3, 0, p(5, 1), 2, false),
        ]
    );
    let v1 = PlayerView::new(&s, 1);
    assert_eq!(
        v1.visible_cities(),
        vec![
            city_view(0, 0, p(1, 1), 2, true),
            city_view(1, 1, p(5, 5), 3, false),
            city_view(2, 1, p(1, 5), 1, false),
            city_view(3, 0, p(5, 1), 2, false),
        ]
    );
}

#[test]
fn own_units_and_cities_in_full() {
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    let units: Vec<_> = v.my_units().into_iter().cloned().collect();
    assert_eq!(units, vec![s.units[0].clone(), s.units[3].clone()]);
    let cities: Vec<_> = v.my_cities().into_iter().cloned().collect();
    assert_eq!(cities, vec![s.cities[0].clone(), s.cities[3].clone()]);
    let v1 = PlayerView::new(&s, 1);
    let ids: Vec<UnitId> = v1.my_units().iter().map(|u| u.id).collect();
    assert_eq!(ids, vec![UnitId(1), UnitId(2), UnitId(4)]);
    let ids: Vec<CityId> = v1.my_cities().iter().map(|c| c.id).collect();
    assert_eq!(ids, vec![CityId(1), CityId(2)]);
}

#[test]
fn globals_stars_techs_income_and_scores() {
    let s = board(&[Tech::Organization, Tech::Hunting]).build();
    let v = PlayerView::new(&s, 0);
    assert_eq!(v.player(), 0);
    assert_eq!(v.size(), 7);
    assert_eq!(v.turn(), 4);
    assert_eq!(v.turn_cap(), 30);
    assert_eq!(v.current(), 0);
    assert!(!v.is_terminal());
    assert_eq!(v.stars(), 7);
    assert_eq!(
        v.techs(),
        TechSet::from_slice(&[Tech::Organization, Tech::Hunting])
    );
    // City 0: level 2 + capital 1 + workshop 1; city 3: level 2.
    assert_eq!(v.income(), 6);
    assert_eq!(v.score(0), score(&s, 0));
    assert_eq!(v.score(1), score(&s, 1));
    assert_eq!(*v.tables(), *s.tables);

    let v1 = PlayerView::new(&s, 1);
    assert_eq!(v1.player(), 1);
    assert_eq!(v1.current(), 0);
    assert_eq!(v1.stars(), 13);
    assert_eq!(
        v1.techs(),
        TechSet::from_slice(&[Tech::Organization, Tech::Climbing, Tech::Riding])
    );
    // City 1: level 3 + 1 park; city 2: level 1.
    assert_eq!(v1.income(), 5);
    assert_eq!(v1.score(0), score(&s, 0));
    assert_eq!(v1.score(1), score(&s, 1));
}

#[test]
fn legal_actions_only_on_the_viewers_turn() {
    let s = board(&[Tech::Organization]).build();
    let want = legal_actions(&s);
    assert!(want.len() > 1);
    assert_eq!(PlayerView::new(&s, 0).legal_actions(), want);
    assert!(PlayerView::new(&s, 1).legal_actions().is_empty());

    let s1 = board(&[Tech::Organization]).current(1).build();
    assert!(PlayerView::new(&s1, 0).legal_actions().is_empty());
    assert_eq!(PlayerView::new(&s1, 1).legal_actions(), legal_actions(&s1));

    let mut over: State = board(&[]).build();
    over.outcome = Some(Outcome::Draw);
    let v = PlayerView::new(&over, 0);
    assert!(v.is_terminal());
    assert!(v.legal_actions().is_empty());
    assert!(PlayerView::new(&over, 1).legal_actions().is_empty());
}

#[test]
fn reachable_tiles_of_own_units() {
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    for id in [UnitId(0), UnitId(3)] {
        assert_eq!(v.reachable_tiles(id), reachable_tiles(&s, id), "{id:?}");
    }
    let v1 = PlayerView::new(&s, 1);
    assert_eq!(
        v1.reachable_tiles(UnitId(1)),
        reachable_tiles(&s, UnitId(1))
    );
}

#[test]
fn reachable_tiles_panics_for_units_that_are_not_the_viewers() {
    let s = board(&[]).build();
    let v = PlayerView::new(&s, 0);
    // Own units work (so a panic below is the ownership check, not a missing method).
    assert_eq!(v.reachable_tiles(UnitId(3)), reachable_tiles(&s, UnitId(3)));
    // Unit 1 is a visible enemy, unit 2 a hidden enemy.
    for id in [UnitId(1), UnitId(2)] {
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| v.reachable_tiles(id)));
        assert!(r.is_err(), "{id:?} is not the viewer's unit");
    }
}

#[test]
fn pending_reward_city_is_own_only() {
    let mut s = board(&[]).build();
    assert_eq!(PlayerView::new(&s, 0).pending_reward_city(), None);
    s.cities[3].pending_reward = true;
    assert_eq!(
        PlayerView::new(&s, 0).pending_reward_city(),
        Some(CityId(3))
    );
    assert_eq!(PlayerView::new(&s, 1).pending_reward_city(), None);
    s.cities[3].pending_reward = false;
    s.cities[1].pending_reward = true;
    assert_eq!(PlayerView::new(&s, 0).pending_reward_city(), None);
    assert_eq!(
        PlayerView::new(&s, 1).pending_reward_city(),
        Some(CityId(1))
    );
}
