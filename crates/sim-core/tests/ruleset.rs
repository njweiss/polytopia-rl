//! Milestone 15 "Tests" items 3 to 5: the ruleset flag in `Config`, v2 tech research, the 16-tech
//! set, the score of v2 techs, nothing else unlocked under v2, and v1 never seeing a v2 variant
//! (docs/MILESTONE_15 "Ruleset flag", "v2 tech research").
//!
//! The research rule is replayed independently from the brief's table (`V2_TECH_TABLE`, copied
//! from docs/MILESTONE_15 "Data files" and the spec's v1 tech table) and its cost formula
//! `4 + cities x tier`; nothing here calls `sim_core::economy` to compute an expectation.

mod common;

use std::sync::Arc;

use common::{p, pick};
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{SiteAction, Tables, Tech, UnitType, ALL_TECHS};
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::score_breakdown;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{CityId, Config, Ruleset, State, TechSet};
use sim_core::testing::StateBuilder;

/// (tech, tier, requires), in declaration order: the spec's v1 table, then the brief's v2 rows.
const V2_TECH_TABLE: [(Tech, u32, Option<Tech>); 16] = [
    (Tech::Climbing, 1, None),
    (Tech::Hunting, 1, None),
    (Tech::Organization, 1, None),
    (Tech::Riding, 1, None),
    (Tech::Archery, 2, Some(Tech::Hunting)),
    (Tech::Forestry, 2, Some(Tech::Hunting)),
    (Tech::Farming, 2, Some(Tech::Organization)),
    (Tech::Strategy, 2, Some(Tech::Organization)),
    (Tech::Mining, 2, Some(Tech::Climbing)),
    (Tech::Roads, 2, Some(Tech::Riding)),
    (Tech::FreeSpirit, 2, Some(Tech::Riding)),
    (Tech::Trade, 3, Some(Tech::Roads)),
    (Tech::Chivalry, 3, Some(Tech::FreeSpirit)),
    (Tech::Mathematics, 3, Some(Tech::Forestry)),
    (Tech::Construction, 3, Some(Tech::Farming)),
    (Tech::Smithery, 3, Some(Tech::Mining)),
];

const NEW_TECHS: [Tech; 7] = [
    Tech::Roads,
    Tech::FreeSpirit,
    Tech::Trade,
    Tech::Chivalry,
    Tech::Mathematics,
    Tech::Construction,
    Tech::Smithery,
];

const NEW_UNITS: [UnitType; 3] = [UnitType::Swordsman, UnitType::Catapult, UnitType::Knight];

fn v2_config() -> Config {
    Config {
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn tier(t: Tech) -> u32 {
    V2_TECH_TABLE.iter().find(|r| r.0 == t).unwrap().1
}

/// A scripted v2 state: the builder's board with the v2 config and the v2 tables.
fn v2_state(b: StateBuilder) -> State {
    let mut s = b.config(v2_config()).build();
    s.tables = Arc::new(Tables::load_for(Ruleset::V2).expect("v2 tables"));
    s
}

fn is_v2_only(a: &Action) -> bool {
    match *a {
        Action::Research { tech } => NEW_TECHS.contains(&tech),
        Action::Train { kind, .. } => NEW_UNITS.contains(&kind),
        Action::Site { action, .. } => action == SiteAction::BurnForest,
        _ => false,
    }
}

/// The brief's rule: every tech of the table not yet researched whose prerequisite is researched
/// and whose cost `4 + cities x tier` the current player can pay, in table order.
fn expected_research(s: &State) -> Vec<Action> {
    let me = s.current;
    let pl = &s.players[me as usize];
    let cities = s.cities.iter().filter(|c| c.owner == me).count() as u32;
    V2_TECH_TABLE
        .iter()
        .filter(|(t, tier, req)| {
            !pl.techs.contains(*t)
                && req.is_none_or(|r| pl.techs.contains(r))
                && pl.stars >= 4 + cities * tier
        })
        .map(|&(tech, _, _)| Action::Research { tech })
        .collect()
}

fn research_in(legal: &[Action]) -> Vec<Action> {
    legal
        .iter()
        .copied()
        .filter(|a| matches!(a, Action::Research { .. }))
        .collect()
}

fn assert_illegal(s: &State, a: Action) {
    let mut t = s.clone();
    assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{a:?}");
    assert_eq!(&t, s, "{a:?}: a refused action leaves the state unchanged");
}

/// A v2 board: player 0's city at (2,2) (empty tile) with forest, crop and metal in its territory.
fn v2_board(stars: u32, techs: &[Tech]) -> StateBuilder {
    StateBuilder::new(5)
        .terrain(p(1, 1), Terrain::Forest)
        .terrain(p(1, 2), Terrain::Forest)
        .resource(p(1, 2), Resource::Animal)
        .resource(p(2, 3), Resource::Crop)
        .terrain(p(3, 3), Terrain::Mountain)
        .resource(p(3, 3), Resource::Metal)
        .city_with(0, p(2, 2), |c| {
            c.is_capital = true;
            c.level = 5;
            c.population = 14;
        })
        .city(1, p(0, 4))
        .stars(0, stars)
        .techs(0, techs)
        .explore_all()
}

// ---------------------------------------------------------------- Config and new_game

#[test]
fn config_json_carries_the_ruleset_only_under_v2() {
    let v1 = serde_json::to_string(&Config::default()).unwrap();
    assert!(!v1.contains("ruleset"), "{v1}");
    let c = Config {
        turn_cap: 12,
        ..v2_config()
    };
    let json = serde_json::to_string(&c).unwrap();
    assert!(json.contains(r#""ruleset":"v2""#), "{json}");
    assert_eq!(serde_json::from_str::<Config>(&json).unwrap(), c);
    // Without the field: V1. Spelled out: also accepted.
    let back: Config = serde_json::from_str(&v1).unwrap();
    assert_eq!(back.ruleset, Ruleset::V1);
    assert_eq!(back, Config::default());
    let explicit = v1.replacen('{', r#"{"ruleset":"v1","#, 1);
    assert_eq!(
        serde_json::from_str::<Config>(&explicit).unwrap(),
        Config::default()
    );
    let v3 = v1.replacen('{', r#"{"ruleset":"v3","#, 1);
    assert!(serde_json::from_str::<Config>(&v3).is_err());
    assert_eq!(Ruleset::default(), Ruleset::V1);
    assert_eq!(serde_json::to_string(&Ruleset::V2).unwrap(), r#""v2""#);
}

#[test]
fn new_game_reads_the_tables_of_the_configs_ruleset() {
    for seed in 0..4 {
        let a = new_game_from_seed(seed, &MapParams::default(), &Config::default());
        let b = new_game_from_seed(seed, &MapParams::default(), &v2_config());
        assert_eq!(a.tables.ruleset(), Ruleset::V1);
        assert_eq!(b.tables.ruleset(), Ruleset::V2);
        assert_eq!(*b.tables, Tables::load_for(Ruleset::V2).unwrap());
        assert_eq!(b.config.ruleset, Ruleset::V2);
        // No other v2 rule is active in milestone 15: the same map, seats and start.
        let mut b1 = b.clone();
        b1.config.ruleset = Ruleset::V1;
        b1.tables = a.tables.clone();
        assert_eq!(b1, a, "seed {seed}");
    }
    // Shared once per ruleset per process.
    let x = new_game_from_seed(1, &MapParams::default(), &v2_config());
    let y = new_game_from_seed(2, &MapParams::default(), &v2_config());
    let z = new_game_from_seed(1, &MapParams::default(), &Config::default());
    assert!(Arc::ptr_eq(&x.tables, &y.tables));
    assert!(!Arc::ptr_eq(&x.tables, &z.tables));
}

// ---------------------------------------------------------------- research under v2

#[test]
fn roads_needs_riding() {
    let s = v2_state(v2_board(30, &[Tech::Organization]));
    let legal = legal_actions(&s);
    assert!(!legal.contains(&Action::Research { tech: Tech::Roads }));
    assert_eq!(research_in(&legal), expected_research(&s));
    assert_illegal(&s, Action::Research { tech: Tech::Roads });

    let mut s = s;
    apply(&mut s, &Action::Research { tech: Tech::Riding }).unwrap();
    assert_eq!(s.players[0].stars, 30 - 5);
    let legal = legal_actions(&s);
    assert!(legal.contains(&Action::Research { tech: Tech::Roads }));
    assert!(legal.contains(&Action::Research {
        tech: Tech::FreeSpirit
    }));
    assert_eq!(research_in(&legal), expected_research(&s));
    // One city of player 0: Roads costs 4 + 1 x 2.
    apply(&mut s, &Action::Research { tech: Tech::Roads }).unwrap();
    assert_eq!(s.players[0].stars, 25 - 6);
    assert!(s.players[0].techs.contains(Tech::Roads));
    assert!(s.players[0].techs.contains(Tech::Riding));
}

#[test]
fn trade_needs_roads_and_its_cost_follows_the_city_count() {
    // Free Spirit is not Trade's prerequisite.
    let s = v2_state(v2_board(99, &[Tech::Riding, Tech::FreeSpirit]));
    let legal = legal_actions(&s);
    assert!(!legal.contains(&Action::Research { tech: Tech::Trade }));
    assert!(legal.contains(&Action::Research {
        tech: Tech::Chivalry
    }));
    assert_illegal(&s, Action::Research { tech: Tech::Trade });

    // Two cities: Trade costs 4 + 2 x 3 = 10.
    let two = |stars| {
        v2_state(
            v2_board(stars, &[Tech::Riding, Tech::Roads]).city_with(0, p(4, 0), |c| c.level = 1),
        )
    };
    let s9 = two(9);
    assert!(!legal_actions(&s9).contains(&Action::Research { tech: Tech::Trade }));
    assert_illegal(&s9, Action::Research { tech: Tech::Trade });
    let mut s10 = two(10);
    assert!(legal_actions(&s10).contains(&Action::Research { tech: Tech::Trade }));
    assert_eq!(research_in(&legal_actions(&s10)), expected_research(&s10));
    apply(&mut s10, &Action::Research { tech: Tech::Trade }).unwrap();
    assert_eq!(s10.players[0].stars, 0);
    assert!(s10.players[0].techs.contains(Tech::Trade));
}

#[test]
fn every_v2_tech_needs_exactly_its_prerequisite() {
    for &(tech, t, req) in &V2_TECH_TABLE[9..] {
        let req = req.unwrap();
        let with = v2_state(v2_board(50, &[req]));
        assert!(
            legal_actions(&with).contains(&Action::Research { tech }),
            "{tech:?} after {req:?}"
        );
        let mut after = with.clone();
        apply(&mut after, &Action::Research { tech }).unwrap();
        assert_eq!(after.players[0].stars, 50 - (4 + t), "{tech:?} cost");
        // Every other tech researched, but not the prerequisite.
        let others: Vec<Tech> = V2_TECH_TABLE
            .iter()
            .map(|r| r.0)
            .filter(|&k| k != tech && k != req)
            .collect();
        let without = v2_state(v2_board(50, &others));
        assert!(
            !legal_actions(&without).contains(&Action::Research { tech }),
            "{tech:?} without {req:?}"
        );
        assert_illegal(&without, Action::Research { tech });
        // Already researched: not again.
        let done = v2_state(v2_board(50, &[req, tech]));
        assert!(!legal_actions(&done).contains(&Action::Research { tech }));
    }
}

#[test]
fn research_list_matches_the_rule_on_scripted_v2_boards() {
    let mut rng = Rng::new(15);
    for case in 0..200 {
        let techs: Vec<Tech> = V2_TECH_TABLE
            .iter()
            .map(|r| r.0)
            .filter(|_| rng.gen_range(3) == 0)
            .collect();
        let stars = rng.gen_range(14);
        let mut b = v2_board(stars, &techs);
        if case % 2 == 0 {
            b = b.city(0, p(4, 0));
        }
        let s = v2_state(b);
        assert_eq!(
            research_in(&legal_actions(&s)),
            expected_research(&s),
            "case {case}: techs {techs:?}, stars {stars}"
        );
    }
}

#[test]
fn the_tech_set_holds_all_16_techs_without_collisions() {
    let all: Vec<Tech> = V2_TECH_TABLE.iter().map(|r| r.0).collect();
    for &t in &all {
        let one = TechSet::from_slice(&[t]);
        for &k in &all {
            assert_eq!(one.contains(k), k == t, "{t:?} set, {k:?} checked");
        }
    }
    let full = TechSet::from_slice(&all);
    assert!(all.iter().all(|&t| full.contains(t)));
    // Distinct subsets are distinct sets.
    for &t in &all {
        let without: Vec<Tech> = all.iter().copied().filter(|&k| k != t).collect();
        assert_ne!(TechSet::from_slice(&without), full, "{t:?}");
    }
    // v1 sets are unchanged: a v1 set never contains a v2 tech.
    let v1 = TechSet::from_slice(&ALL_TECHS);
    assert!(NEW_TECHS.iter().all(|&t| !v1.contains(t)));
}

#[test]
fn researching_every_v2_tech_records_each_one() {
    let mut s = v2_state(v2_board(200, &[Tech::Organization]));
    let order = [
        Tech::Climbing,
        Tech::Hunting,
        Tech::Riding,
        Tech::Forestry,
        Tech::Farming,
        Tech::Mining,
        Tech::Roads,
        Tech::FreeSpirit,
        Tech::Trade,
        Tech::Chivalry,
        Tech::Mathematics,
        Tech::Construction,
        Tech::Smithery,
    ];
    let mut have = vec![Tech::Organization];
    let mut stars = 200;
    for t in order {
        apply(&mut s, &Action::Research { tech: t }).unwrap();
        stars -= 4 + tier(t);
        have.push(t);
        assert_eq!(s.players[0].stars, stars, "{t:?}");
        for &(k, _, _) in &V2_TECH_TABLE {
            assert_eq!(s.players[0].techs.contains(k), have.contains(&k), "{k:?}");
        }
    }
}

#[test]
fn v2_techs_score_100_per_tier() {
    // Tiers: Organization 1, Roads 2, Free Spirit 2, then five tier-3 techs: 1 + 2 + 2 + 15 = 20.
    let techs = [
        Tech::Organization,
        Tech::Roads,
        Tech::FreeSpirit,
        Tech::Trade,
        Tech::Chivalry,
        Tech::Mathematics,
        Tech::Construction,
        Tech::Smithery,
    ];
    let s = v2_state(v2_board(0, &techs));
    assert_eq!(score_breakdown(&s, 0).techs, 2000);
    let only_v1 = v2_state(v2_board(0, &[Tech::Organization]));
    assert_eq!(score_breakdown(&only_v1, 0).techs, 100);
    assert_eq!(
        score_breakdown(&s, 0).total() - score_breakdown(&only_v1, 0).total(),
        1900
    );
    // The weight applies as in v1.
    let mut w = s.clone();
    w.config.score_weights.tech_per_tier = 7;
    assert_eq!(score_breakdown(&w, 0).techs, 140);
}

// ---------------------------------------------------------------- nothing else unlocked

#[test]
fn under_v2_with_every_tech_no_new_unit_or_burn_forest_is_legal() {
    let all: Vec<Tech> = V2_TECH_TABLE.iter().map(|r| r.0).collect();
    let s = v2_state(v2_board(99, &all));
    let legal = legal_actions(&s);
    // The board is rich enough that v1 content is legal ...
    assert!(legal.contains(&Action::Train {
        city: CityId(0),
        kind: UnitType::Rider
    }));
    assert!(legal.contains(&Action::Site {
        at: p(1, 1),
        action: SiteAction::LumberHut
    }));
    assert!(legal.contains(&Action::Site {
        at: p(1, 2),
        action: SiteAction::ClearForest
    }));
    // ... but nothing from v2 beyond research.
    for a in &legal {
        assert!(
            !matches!(a, Action::Train { kind, .. } if NEW_UNITS.contains(kind)),
            "{a:?}"
        );
        assert!(
            !matches!(
                a,
                Action::Site {
                    action: SiteAction::BurnForest,
                    ..
                }
            ),
            "{a:?}"
        );
    }
    for kind in NEW_UNITS {
        assert_illegal(
            &s,
            Action::Train {
                city: CityId(0),
                kind,
            },
        );
    }
    for at in [p(1, 1), p(1, 2)] {
        assert_illegal(
            &s,
            Action::Site {
                at,
                action: SiteAction::BurnForest,
            },
        );
    }
}

/// Random v2 games: research follows the rule, v2 techs get researched, nothing else from v2.
#[test]
fn random_v2_playouts_research_v2_techs_and_nothing_else() {
    let mut researched_v2 = 0;
    for seed in 0..8u64 {
        let cfg = Config {
            turn_cap: 15,
            ..v2_config()
        };
        let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg);
        let mut rng = Rng::new(seed ^ 0x15);
        let mut steps = 0;
        while !s.is_terminal() && steps < 4000 {
            let legal = legal_actions(&s);
            if !legal
                .iter()
                .any(|a| matches!(a, Action::ChooseReward { .. }))
            {
                assert_eq!(research_in(&legal), expected_research(&s), "seed {seed}");
            }
            for a in &legal {
                assert!(
                    !is_v2_only(a) || matches!(a, Action::Research { .. }),
                    "{a:?}"
                );
            }
            let a = pick(&mut rng, &legal);
            if matches!(a, Action::Research { tech } if NEW_TECHS.contains(&tech)) {
                researched_v2 += 1;
            }
            apply(&mut s, &a).unwrap();
            steps += 1;
        }
    }
    assert!(researched_v2 >= 5, "only {researched_v2} v2 researches");
}

// ---------------------------------------------------------------- v1 never sees v2

#[test]
fn under_v1_no_v2_action_is_ever_legal_or_applicable() {
    for seed in 0..10u64 {
        let cfg = Config {
            turn_cap: 12,
            full_observability: seed % 2 == 0,
            ..Config::default()
        };
        let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg);
        let mut rng = Rng::new(seed ^ 0x51);
        let mut steps = 0;
        while !s.is_terminal() && steps < 3000 {
            let legal = legal_actions(&s);
            for a in &legal {
                assert!(!is_v2_only(a), "seed {seed}: {a:?} under v1");
            }
            if steps % 37 == 0 {
                for tech in NEW_TECHS {
                    assert_illegal(&s, Action::Research { tech });
                }
                let me = s.current;
                if let Some(c) = s.cities.iter().find(|c| c.owner == me) {
                    for kind in NEW_UNITS {
                        assert_illegal(&s, Action::Train { city: c.id, kind });
                    }
                    assert_illegal(
                        &s,
                        Action::Site {
                            at: c.pos,
                            action: SiteAction::BurnForest,
                        },
                    );
                }
            }
            let a = pick(&mut rng, &legal);
            apply(&mut s, &a).unwrap();
            steps += 1;
        }
    }
}

#[test]
fn under_v1_a_rich_board_refuses_every_v2_action() {
    let s = v2_board(99, &ALL_TECHS).build();
    assert_eq!(s.config.ruleset, Ruleset::V1);
    let legal = legal_actions(&s);
    assert!(legal.iter().all(|a| !is_v2_only(a)));
    for tech in NEW_TECHS {
        assert_illegal(&s, Action::Research { tech });
    }
    for kind in NEW_UNITS {
        assert_illegal(
            &s,
            Action::Train {
                city: CityId(0),
                kind,
            },
        );
    }
    assert_illegal(
        &s,
        Action::Site {
            at: p(1, 1),
            action: SiteAction::BurnForest,
        },
    );
}
