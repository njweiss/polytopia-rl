//! Milestone 19 bot tests (docs/MILESTONE_19 "Bots", "Tests" item 4):
//! - `PlayerView::ruleset()`
//! - Scripted `RushBot` and `EconomyBot` v2 rule goldens + `attack_score` `Stiff` awareness
//! - V1 identity check
//! - V2 property tests (legality, determinism, turn termination <= 200 actions)
//! - V2 behavior floor (>= 75% win rate over 40 side-swapped games vs RandomBot)
//! - V2 no-cheating metamorphic property (including hidden v2 player & tile fields)

mod common;

use std::sync::Arc;

use bots::{make_bot, run_match, Bot, BotKind, EconomyBot, RushBot};
use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Building, Monument, SiteAction, Tables, Tech, UnitType, ALL_TECHS};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{
    City, CityId, Config, Dir, Improvement, MonumentSet, Phase, PlayerId, Ruleset, State, TechSet,
    Unit, UnitId,
};
use sim_core::testing::StateBuilder;
use sim_core::view::PlayerView;

const ALL_V2_TECHS: [Tech; 16] = [
    Tech::Climbing,
    Tech::Hunting,
    Tech::Organization,
    Tech::Riding,
    Tech::Archery,
    Tech::Forestry,
    Tech::Farming,
    Tech::Strategy,
    Tech::Mining,
    Tech::Roads,
    Tech::FreeSpirit,
    Tech::Trade,
    Tech::Chivalry,
    Tech::Mathematics,
    Tech::Construction,
    Tech::Smithery,
];

fn v2_cfg(turn_cap: u32) -> Config {
    Config {
        turn_cap,
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn v2_open(n: u8) -> StateBuilder {
    StateBuilder::new(n).config(v2_cfg(30)).explore_all()
}

// ---------------------------------------------------------------- PlayerView::ruleset

#[test]
fn player_view_exposes_ruleset_for_both_players() {
    let v1 = open(5).build();
    assert_eq!(PlayerView::new(&v1, 0).ruleset(), Ruleset::V1);
    assert_eq!(PlayerView::new(&v1, 1).ruleset(), Ruleset::V1);

    let v2 = v2_open(5).build();
    assert_eq!(PlayerView::new(&v2, 0).ruleset(), Ruleset::V2);
    assert_eq!(PlayerView::new(&v2, 1).ruleset(), Ruleset::V2);
}

// ---------------------------------------------------------------- attack_score Stiff awareness

#[test]
fn attack_score_treats_stiff_defenders_as_zero_retaliation() {
    // Custom v2 table: Catapult has defense 3, health 15, range 3, skills ["stiff"].
    // Warrior 10/10 vs this Catapult 15/15 at distance 1:
    //   af = 2.0, df = 3.0, td = 5.0 -> dealt = 4; retaliation WOULD be 8 if not Stiff.
    //   Without Stiff awareness: score = 4 - 8 = -4 (refused by both Rush >= 0 and Economy >= 3).
    //   With Stiff awareness: taken = 0 -> score = 4 (accepted by both Rush and Economy!).
    let custom_units = r#"[
      { "unit": "swordsman", "cost": 5, "health": 15, "attack": 3,   "defense": 3, "movement": 1, "range": 1, "skills": ["dash", "fortify"],            "unlocked_by": "smithery" },
      { "unit": "catapult",  "cost": 8, "health": 15, "attack": 4,   "defense": 3, "movement": 1, "range": 3, "skills": ["stiff"],                      "unlocked_by": "mathematics" },
      { "unit": "knight",    "cost": 8, "health": 10, "attack": 3.5, "defense": 1, "movement": 3, "range": 1, "skills": ["dash", "fortify", "persist"], "unlocked_by": "chivalry" }
    ]"#;
    let tables = Tables::from_json_v2(
        custom_units,
        include_str!("../../../data/v2/techs.json"),
        include_str!("../../../data/v2/site_actions.json"),
        include_str!("../../../data/v2/buildings.json"),
        include_str!("../../../data/v2/monuments.json"),
        include_str!("../../../data/v2/rules.json"),
    )
    .expect("custom v2 tables");

    let mut s = v2_open(7)
        .unit(0, UnitType::Warrior, p(3, 3))
        .unit_with(1, UnitType::Catapult, p(3, 4), |u| {
            u.health = 15;
            u.max_health = 15;
        })
        .build();
    s.tables = Arc::new(tables);

    let atk = Action::Attack {
        unit: UnitId(0),
        target: p(3, 4),
    };
    assert_choice(&EconomyBot, &s, atk);
    assert_choice(&RushBot, &s, atk);
}

// ---------------------------------------------------------------- RushBot v2 scripted goldens

#[test]
fn rush_v2_places_monument_after_attack_and_before_train() {
    let base = || {
        v2_open(7)
            .city(0, p(3, 3))
            .techs(
                0,
                &[
                    Tech::Organization,
                    Tech::Riding,
                    Tech::FreeSpirit,
                    Tech::Chivalry,
                ],
            )
            .stars(0, 8)
            .player_with(0, |pl| {
                pl.monuments_unlocked.insert(Monument::GateOfPower);
            })
    };
    // No enemy -> BuildMonument on first legal tile in row-major order: (2,2).
    let s = base().build();
    assert_choice(
        &RushBot,
        &s,
        Action::BuildMonument {
            at: p(2, 2),
            monument: Monument::GateOfPower,
        },
    );

    // With a favorable attack available -> Attack beats BuildMonument.
    let s_atk = base()
        .unit(0, UnitType::Warrior, p(0, 0))
        .unit_with(1, UnitType::Warrior, p(0, 1), |u| u.health = 8)
        .build();
    assert_choice(
        &RushBot,
        &s_atk,
        Action::Attack {
            unit: UnitId(0),
            target: p(0, 1),
        },
    );
}

#[test]
fn rush_v2_trains_knight_over_rider_over_warrior() {
    let techs = &[
        Tech::Organization,
        Tech::Riding,
        Tech::FreeSpirit,
        Tech::Chivalry,
    ];
    let s8 = v2_open(7)
        .city(0, p(3, 3))
        .techs(0, techs)
        .stars(0, 8)
        .build();
    assert_choice(
        &RushBot,
        &s8,
        Action::Train {
            city: CityId(0),
            kind: UnitType::Knight,
        },
    );

    let s3 = v2_open(7)
        .city(0, p(3, 3))
        .techs(0, techs)
        .stars(0, 3)
        .build();
    assert_choice(
        &RushBot,
        &s3,
        Action::Train {
            city: CityId(0),
            kind: UnitType::Rider,
        },
    );

    let s2 = v2_open(7)
        .city(0, p(3, 3))
        .techs(0, techs)
        .stars(0, 2)
        .build();
    assert_choice(
        &RushBot,
        &s2,
        Action::Train {
            city: CityId(0),
            kind: UnitType::Warrior,
        },
    );

    // City 0 occupied, city 1 open: trains Knight in city 1.
    let s_two = v2_open(7)
        .city(0, p(1, 1))
        .city(0, p(5, 5))
        .unit_with(0, UnitType::Warrior, p(1, 1), |u| u.phase = Phase::Done)
        .techs(0, techs)
        .stars(0, 8)
        .build();
    assert_choice(
        &RushBot,
        &s_two,
        Action::Train {
            city: CityId(1),
            kind: UnitType::Knight,
        },
    );
}

#[test]
fn rush_v2_researches_riding_free_spirit_chivalry_then_harvests() {
    let board = |techs: &[Tech], stars: u32| {
        v2_open(7)
            .city(0, p(3, 3))
            .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
            .resource(p(2, 2), Resource::Fruit)
            .techs(0, techs)
            .stars(0, stars)
            .build()
    };
    assert_choice(
        &RushBot,
        &board(&[Tech::Organization], 10),
        Action::Research { tech: Tech::Riding },
    );
    assert_choice(
        &RushBot,
        &board(&[Tech::Organization, Tech::Riding], 10),
        Action::Research {
            tech: Tech::FreeSpirit,
        },
    );
    assert_choice(
        &RushBot,
        &board(&[Tech::Organization, Tech::Riding, Tech::FreeSpirit], 10),
        Action::Research {
            tech: Tech::Chivalry,
        },
    );
    // Once Chivalry is researched, Rush does not research other techs; it harvests fruit.
    assert_choice(
        &RushBot,
        &board(
            &[
                Tech::Organization,
                Tech::Riding,
                Tech::FreeSpirit,
                Tech::Chivalry,
            ],
            10,
        ),
        Action::Site {
            at: p(2, 2),
            action: SiteAction::HarvestFruit,
        },
    );
}

#[test]
fn rush_v2_chains_knight_persist_kills_and_promotes_mid_chain() {
    let mut s = v2_open(9)
        .city(0, p(7, 7))
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 10)
        .unit(0, UnitType::Knight, p(3, 0))
        .unit(1, UnitType::Warrior, p(3, 1))
        .unit(1, UnitType::Warrior, p(3, 2))
        .unit(1, UnitType::Warrior, p(3, 3))
        .unit(1, UnitType::Warrior, p(3, 4))
        .build();

    let mut rng = Rng::new(0);
    let expected = [
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 1),
        },
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 2),
        },
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 3),
        },
        Action::Promote { unit: UnitId(0) },
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 4),
        },
    ];
    for (step, want) in expected.into_iter().enumerate() {
        let got = RushBot.choose(&PlayerView::new(&s, 0), &mut rng);
        assert_eq!(got, want, "step {step}");
        apply(&mut s, &got).unwrap();
    }
    assert_eq!(s.unit(UnitId(0)).unwrap().pos, p(3, 4));
    assert!(s.unit(UnitId(0)).unwrap().veteran);
    assert_eq!(s.unit(UnitId(0)).unwrap().kills, 4);
}

// ---------------------------------------------------------------- EconomyBot v2 scripted goldens

#[test]
fn economy_v2_places_monument_after_attack_and_before_sites() {
    let base = || {
        v2_open(7)
            .city(0, p(3, 3))
            .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
            .resource(p(4, 4), Resource::Crop)
            .techs(0, &[Tech::Organization, Tech::Farming])
            .stars(0, 10)
            .player_with(0, |pl| {
                pl.monuments_unlocked.insert(Monument::ParkOfFortune);
            })
    };
    let s = base().build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::BuildMonument {
            at: p(2, 2),
            monument: Monument::ParkOfFortune,
        },
    );

    let s_atk = base()
        .unit(0, UnitType::Warrior, p(0, 0))
        .unit_with(1, UnitType::Warrior, p(0, 1), |u| u.health = 2)
        .build();
    assert_choice(
        &EconomyBot,
        &s_atk,
        Action::Attack {
            unit: UnitId(1),
            target: p(0, 1),
        },
    );
}

#[test]
fn economy_v2_builds_economy_buildings_in_priority_order_after_sites() {
    // 7x7 board with city 0 at (3,3) (owns 2..=4 x 2..=4).
    // Improvements placed on corners of territory:
    //   (2,2) Forest + LumberHut -> enables Sawmill on adjacent field (2,3) & (3,2)
    //   (2,4) Crop + Farm        -> enables Windmill on adjacent field (2,3) & (3,4)
    //   (4,2) Mountain + Mine    -> enables Forge on adjacent field (3,2) & (4,3)
    // If (2,3) already has a Sawmill, (3,4) can take a Market (adjacent to Sawmill at (2,3))!
    // Wait: let's test Sites > Market > Forge > Sawmill > Windmill cleanly!
    let base_sites = || {
        v2_open(7)
            .city_with(0, p(3, 3), |c| c.border_growth_used = true)
            .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
            .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
            .terrain(p(2, 2), Terrain::Forest)
            .improvement(p(2, 2), Improvement::LumberHut)
            .resource(p(2, 4), Resource::Crop)
            .improvement(p(2, 4), Improvement::Farm)
            .terrain(p(4, 2), Terrain::Mountain)
            .resource(p(4, 2), Resource::Metal)
            .improvement(p(4, 2), Improvement::Mine)
            .techs(0, &ALL_V2_TECHS)
            .stars(0, 20)
    };

    // 1. Sites beat BuildBuilding: add an unharvested Fruit at (4,4) -> HarvestFruit wins.
    let s_site = base_sites().resource(p(4, 4), Resource::Fruit).build();
    assert_choice(
        &EconomyBot,
        &s_site,
        Action::Site {
            at: p(4, 4),
            action: SiteAction::HarvestFruit,
        },
    );

    // 2. Market beats Forge, Sawmill, Windmill (even when Market's only tile is later row-major):
    //    Place a Sawmill at (3,2) (adjacent to LumberHut (2,2)), wait — one Sawmill per city!
    //    So to have Market + Forge + Windmill all legal simultaneously:
    //    Sawmill at (2,3) (next to LumberHut (2,2)) -> Market legal at (3,2), (3,4);
    //    Forge legal at (3,2), (4,3) (next to Mine (4,2));
    //    Windmill legal at (3,4) (next to Farm (2,4)).
    //    Wait: let's have Market ONLY at (4,4) by placing Sawmill at (3,4) in city 1 or owning tiles:
    //    Or simply: Sawmill at (2,3) -> Market is legal at (3,2) and (3,4); Forge at (3,2),(4,3);
    //    Windmill at (3,4). Market wins at (3,2)!
    let s_market = base_sites().building(p(2, 3), Building::Sawmill).build();
    assert_choice(
        &EconomyBot,
        &s_market,
        Action::BuildBuilding {
            at: p(3, 2),
            building: Building::Market,
        },
    );

    // 3. Without Market (no existing Sawmill/Windmill/Forge), Forge beats Sawmill and Windmill
    //    even though Sawmill/Windmill are legal on earlier row-major tile (2,3)!
    let s_forge = base_sites().build();
    assert!(legal_actions(&s_forge).contains(&Action::BuildBuilding {
        at: p(2, 3),
        building: Building::Sawmill,
    }));
    assert!(legal_actions(&s_forge).contains(&Action::BuildBuilding {
        at: p(2, 3),
        building: Building::Windmill,
    }));
    assert_choice(
        &EconomyBot,
        &s_forge,
        Action::BuildBuilding {
            at: p(3, 2),
            building: Building::Forge,
        },
    );

    // 4. Without Forge (already built or no Mine), Sawmill beats Windmill; row-major within Sawmill.
    let s_sawmill = v2_open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .terrain(p(4, 2), Terrain::Forest)
        .improvement(p(4, 2), Improvement::LumberHut)
        .resource(p(2, 4), Resource::Crop)
        .improvement(p(2, 4), Improvement::Farm)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 20)
        .build();
    // Windmill is legal at (2,3) (earlier row-major), Sawmill at (3,2) & (4,3) -> Sawmill at (3,2)!
    assert!(legal_actions(&s_sawmill).contains(&Action::BuildBuilding {
        at: p(2, 3),
        building: Building::Windmill,
    }));
    assert_choice(
        &EconomyBot,
        &s_sawmill,
        Action::BuildBuilding {
            at: p(3, 2),
            building: Building::Sawmill,
        },
    );

    // 5. Only Windmill legal (and research also legal: omit Trade/Chivalry so research is legal):
    //    Windmill beats Research, row-major first tile (2,3) over (3,4).
    let s_windmill = v2_open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .resource(p(2, 4), Resource::Crop)
        .improvement(p(2, 4), Improvement::Farm)
        .techs(0, &[Tech::Organization, Tech::Farming, Tech::Construction])
        .stars(0, 20)
        .build();
    assert_choice(
        &EconomyBot,
        &s_windmill,
        Action::BuildBuilding {
            at: p(2, 3),
            building: Building::Windmill,
        },
    );
}

#[test]
fn economy_v2_research_order_follows_v1_then_v2_list() {
    use Tech::*;
    // First, verify a remaining v1 tech (Archery) is picked before v2 tech (Roads).
    let v1_first = v2_open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .techs(
            0,
            &[
                Organization,
                Hunting,
                Farming,
                Climbing,
                Mining,
                Forestry,
                Riding,
                Strategy,
            ],
        )
        .stars(0, 10)
        .build();
    assert_choice(&EconomyBot, &v1_first, Action::Research { tech: Archery });

    // Once all 9 v1 techs are researched, follow exact v2 order:
    // Roads, Mathematics, Construction, Smithery, Trade, FreeSpirit, Chivalry.
    let v2_order = [
        Roads,
        Mathematics,
        Construction,
        Smithery,
        Trade,
        FreeSpirit,
        Chivalry,
    ];
    let mut have: Vec<Tech> = ALL_TECHS.to_vec();
    for &expected in &v2_order {
        let s = v2_open(7)
            .city(0, p(3, 3))
            .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
            .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
            .techs(0, &have)
            .stars(0, 20)
            .build();
        assert_choice(&EconomyBot, &s, Action::Research { tech: expected });
        have.push(expected);
    }

    // With 6 stars and Roads already researched: tier-3 techs (cost 7) are unaffordable,
    // so FreeSpirit (tier 2, cost 6) is selected next.
    let mut have_roads = ALL_TECHS.to_vec();
    have_roads.push(Roads);
    let s6 = v2_open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .techs(0, &have_roads)
        .stars(0, 6)
        .build();
    assert_choice(&EconomyBot, &s6, Action::Research { tech: FreeSpirit });
}

#[test]
fn economy_v2_trains_swordsman_catapult_warrior_respecting_unit_limit() {
    // All techs so research doesn't preempt train.
    let s_sw = v2_open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 8)
        .build();
    assert_choice(
        &EconomyBot,
        &s_sw,
        Action::Train {
            city: CityId(0),
            kind: UnitType::Swordsman,
        },
    );

    // With 2 cities and 8 stars, Research Smithery costs 4 + 2*3 = 10 > 8, so Train Catapult (8)
    // runs without Research preempting it.
    let all_but_smithery: Vec<Tech> = ALL_V2_TECHS
        .iter()
        .copied()
        .filter(|&t| t != Tech::Smithery)
        .collect();
    let s_cat = v2_open(7)
        .city(0, p(1, 1))
        .city(0, p(5, 5))
        .unit_with(0, UnitType::Warrior, p(1, 1), |u| u.phase = Phase::Done)
        .techs(0, &all_but_smithery)
        .stars(0, 8)
        .build();
    assert_choice(
        &EconomyBot,
        &s_cat,
        Action::Train {
            city: CityId(1),
            kind: UnitType::Catapult,
        },
    );

    // With 5 stars and no Smithery (Catapult costs 8): trains Warrior.
    let s_war = v2_open(7)
        .city(0, p(1, 1))
        .city(0, p(5, 5))
        .unit_with(0, UnitType::Warrior, p(1, 1), |u| u.phase = Phase::Done)
        .techs(0, &all_but_smithery)
        .stars(0, 5)
        .build();
    assert_choice(
        &EconomyBot,
        &s_war,
        Action::Train {
            city: CityId(1),
            kind: UnitType::Warrior,
        },
    );

    // At 2 units per city: refuses training even with 99 stars.
    let s_cap = v2_open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .unit_with(0, UnitType::Warrior, p(0, 6), |u| u.phase = Phase::Done)
        .techs(0, &ALL_V2_TECHS)
        .stars(0, 99)
        .build();
    assert_choice(&EconomyBot, &s_cap, Action::EndTurn);
}

// ---------------------------------------------------------------- V1 identity

#[test]
fn bots_remain_identical_under_v1() {
    // Under V1, RushBot still ignores non-Riding research and EconomyBot still only trains Warrior.
    let s_rush = open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(3, 3), |u| u.phase = Phase::Done)
        .resource(p(2, 2), Resource::Fruit)
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(0, 20)
        .build();
    assert_choice(
        &RushBot,
        &s_rush,
        Action::Site {
            at: p(2, 2),
            action: SiteAction::HarvestFruit,
        },
    );

    let s_econ = open(7)
        .city(0, p(3, 3))
        .unit_with(0, UnitType::Warrior, p(0, 0), |u| u.phase = Phase::Done)
        .techs(0, &ALL_TECHS)
        .stars(0, 20)
        .build();
    assert_choice(
        &EconomyBot,
        &s_econ,
        Action::Train {
            city: CityId(0),
            kind: UnitType::Warrior,
        },
    );
}

// ---------------------------------------------------------------- V2 properties

fn check_v2_game(seed: u64, kinds: [BotKind; 2], eps: f32, turn_cap: u32) {
    let bots = [make_bot(kinds[0], eps), make_bot(kinds[1], eps)];
    let mut rngs = [Rng::new(seed ^ SALT[0]), Rng::new(seed ^ SALT[1])];
    let mut s = new_game_from_seed(seed, &MapParams::default(), &v2_cfg(turn_cap));
    let mut in_turn = 0usize;
    while !s.is_terminal() {
        let cur = s.current as usize;
        let rng_before = rngs[cur].clone();
        let a = bots[cur].choose(&PlayerView::new(&s, s.current), &mut rngs[cur]);
        if eps == 0.0 && kinds[cur] != BotKind::Random {
            assert_eq!(rngs[cur], rng_before, "bare {:?} drew from rng", kinds[cur]);
        }
        assert!(
            legal_actions(&s).contains(&a),
            "v2 seed {seed} {kinds:?} eps {eps}: illegal {a:?}"
        );
        apply(&mut s, &a).unwrap();
        in_turn += 1;
        if kinds[cur] != BotKind::Random {
            assert!(
                in_turn <= 200,
                "v2 seed {seed} {kinds:?} eps {eps}: player {cur} took >200 actions in turn {}",
                s.turn
            );
        }
        if a == Action::EndTurn {
            in_turn = 0;
        }
    }
}

#[test]
fn v2_all_bot_pairs_bare_and_epsilon_play_legally_and_deterministically() {
    for a in KINDS {
        for b in KINDS {
            for seed in 0..3 {
                check_v2_game(seed, [a, b], 0.0, 30);
            }
            for (seed, eps) in [(10, 0.1), (11, 0.25)] {
                check_v2_game(seed, [a, b], eps, 25);
            }
        }
    }
    // Determinism check under V2.
    for (kinds, eps) in [
        ([BotKind::Rush, BotKind::Economy], [0.0, 0.0]),
        ([BotKind::Economy, BotKind::Random], [0.2, 0.0]),
        ([BotKind::Rush, BotKind::Rush], [0.1, 0.3]),
    ] {
        let a = my_play_game(5, kinds, eps, &MapParams::default(), &v2_cfg(20));
        let b = my_play_game(5, kinds, eps, &MapParams::default(), &v2_cfg(20));
        assert_eq!(a.actions, b.actions, "v2 {kinds:?}");
        assert_eq!(a.state, b.state, "v2 {kinds:?}");
    }
}

// ---------------------------------------------------------------- V2 behavior floor

fn v2_floor(kind: BotKind) {
    let m = run_match(
        kind,
        BotKind::Random,
        0..20,
        &MapParams::default(),
        &v2_cfg(30),
    );
    assert_eq!(m.games, 40);
    assert!(
        m.a_wins >= 30,
        "v2 {kind:?} won {} of 40 against random ({} draws)",
        m.a_wins,
        m.draws
    );
}

#[test]
fn v2_rush_beats_random() {
    v2_floor(BotKind::Rush);
}

#[test]
fn v2_economy_beats_random() {
    v2_floor(BotKind::Economy);
}

// ---------------------------------------------------------------- V2 no-cheating

fn idx(s: &State, q: Pos) -> usize {
    q.row as usize * s.size as usize + q.col as usize
}

fn coin(rng: &mut Rng, p: f32) -> bool {
    rng.gen_f32() < p
}

fn pick<T: Copy>(rng: &mut Rng, xs: &[T]) -> T {
    xs[rng.gen_range(xs.len() as u32) as usize]
}

fn deep(s: &State, v: PlayerId, q: Pos) -> bool {
    all_pos(s.size)
        .into_iter()
        .filter(|&n| cheb(n, q) <= 1)
        .all(|n| !s.is_explored(v, n))
}

/// Pre-buckets all 2^16 v2 tech masks by their total tier sum.
fn build_v2_tier_buckets(tables: &Tables) -> Vec<Vec<u16>> {
    let tiers: [u32; 16] = ALL_V2_TECHS.map(|t| tables.tech(t).tier);
    let max_sum: usize = tiers.iter().sum::<u32>() as usize;
    let mut buckets = vec![Vec::new(); max_sum + 1];
    for mask in 0u32..65536 {
        let mut sum = 0usize;
        for (i, &t) in tiers.iter().enumerate() {
            if mask & (1 << i) != 0 {
                sum += t as usize;
            }
        }
        buckets[sum].push(mask as u16);
    }
    buckets
}

fn techset_from_mask(mask: u16) -> TechSet {
    let mut set = TechSet::new();
    for (i, &t) in ALL_V2_TECHS.iter().enumerate() {
        if mask & (1 << i) != 0 {
            set.insert(t);
        }
    }
    set
}

fn try_v2_twin(s: &State, v: PlayerId, buckets: &[Vec<u16>], rng: &mut Rng) -> Option<State> {
    let o = 1 - v;
    let mut t = s.clone();
    t.rng = Rng::new(rng.next_u32() as u64 + 1);
    t.next_unit_id += 1 + rng.gen_range(5);

    // Mutate opponent hidden v1 & v2 player fields.
    t.players[o as usize].stars = rng.gen_range(80);
    let tier_sum: usize = ALL_V2_TECHS
        .iter()
        .filter(|&&x| s.players[o as usize].techs.contains(x))
        .map(|&x| s.tables.tech(x).tier as usize)
        .sum();
    let mask = pick(rng, &buckets[tier_sum]);
    t.players[o as usize].techs = techset_from_mask(mask);
    t.players[o as usize].kills = rng.gen_range(20);
    t.players[o as usize].lighthouses_discovered = (rng.gen_range(16)) as u8;
    t.players[o as usize].monuments_unlocked = MonumentSet::from_slice(
        &[
            Monument::ParkOfFortune,
            Monument::GateOfPower,
            Monument::EyeOfGod,
        ]
        .into_iter()
        .filter(|_| coin(rng, 0.5))
        .collect::<Vec<_>>(),
    );
    t.players[o as usize].monuments_placed = MonumentSet::from_slice(
        &[
            Monument::ParkOfFortune,
            Monument::GateOfPower,
            Monument::EyeOfGod,
        ]
        .into_iter()
        .filter(|&m| t.players[o as usize].monuments_unlocked.contains(m) && coin(rng, 0.5))
        .collect::<Vec<_>>(),
    );

    let o_cities: Vec<CityId> = s
        .cities
        .iter()
        .filter(|c| c.owner == o)
        .map(|c| c.id)
        .collect();
    let dirs = [
        None,
        Some(Dir { dr: -1, dc: 0 }),
        Some(Dir { dr: 0, dc: 1 }),
        Some(Dir { dr: 1, dc: -1 }),
    ];
    for u in t.units.iter_mut().filter(|u| u.owner == o) {
        u.kills = rng.gen_range(6);
        u.origin = if o_cities.is_empty() || coin(rng, 0.3) {
            None
        } else {
            Some(pick(rng, &o_cities))
        };
        u.last_dir = pick(rng, &dirs);
    }
    for c in t.cities.iter_mut().filter(|c| c.owner == o) {
        c.population += rng.gen_range(4) as i32;
        c.workshop = coin(rng, 0.5);
        c.wall = coin(rng, 0.5);
        c.border_growth_used = coin(rng, 0.5);
        c.connected_to = if coin(rng, 0.5) {
            o_cities.first().copied()
        } else {
            None
        };
    }

    let terrains = [Terrain::Field, Terrain::Forest, Terrain::Mountain];
    let resources = [
        None,
        Some(Resource::Fruit),
        Some(Resource::Crop),
        Some(Resource::Animal),
        Some(Resource::Metal),
    ];
    let improvements = [None, None, Some(Improvement::Farm), Some(Improvement::Mine)];
    let buildings = [
        None,
        None,
        Some(Building::Sawmill),
        Some(Building::Windmill),
        Some(Building::Forge),
        Some(Building::Market),
    ];
    let monuments = [
        None,
        None,
        Some(Monument::ParkOfFortune),
        Some(Monument::GateOfPower),
        Some(Monument::EyeOfGod),
    ];
    let v2_unit_kinds = [
        UnitType::Warrior,
        UnitType::Swordsman,
        UnitType::Catapult,
        UnitType::Knight,
    ];
    let mut new_city = false;
    let mut added_units = 0;

    for q in all_pos(s.size) {
        if s.is_explored(v, q) {
            continue;
        }
        let i = idx(s, q);
        let own_territory = s.tiles[i].owner_city.is_some_and(|c| s.city(c).owner == v);
        if own_territory {
            continue;
        }
        let tile = s.tiles[i].clone();
        if tile.unit.is_none() && tile.city.is_none() {
            t.tiles[i].terrain = pick(rng, &terrains);
            t.tiles[i].resource = pick(rng, &resources);
            t.tiles[i].improvement = pick(rng, &improvements);
            if coin(rng, 0.3) {
                t.tiles[i].village = !tile.village;
            }
            if coin(rng, 0.4) {
                t.tiles[i].road = !tile.road;
            }
            let b = pick(rng, &buildings);
            t.tiles[i].building = b;
            t.tiles[i].building_level = if b.is_some() {
                1 + rng.gen_range(4) as u8
            } else {
                0
            };
            // Placing a monument on an unowned unexplored tile tests monument visibility without
            // blowing out the 400-point exploration score delta (opponent-owned monument tested
            // separately below).
            if tile.owner_city.is_none() {
                t.tiles[i].monument = pick(rng, &monuments);
            }
        }
        if !deep(s, v, q) {
            continue;
        }
        match tile.unit {
            Some(id) if s.unit(id).is_some_and(|u| u.owner == o) && coin(rng, 0.4) => {
                t.units.retain(|u| u.id != id);
                t.tiles[i].unit = None;
            }
            None if added_units < 2 && coin(rng, 0.12) => {
                added_units += 1;
                let kind = pick(rng, &v2_unit_kinds);
                let hp = t.tables.unit(kind).health;
                let id = UnitId(t.next_unit_id);
                t.next_unit_id += 1;
                t.units.push(Unit {
                    id,
                    kind,
                    owner: o,
                    pos: q,
                    health: hp,
                    max_health: hp,
                    origin: None,
                    kills: 0,
                    veteran: false,
                    promotion_available: false,
                    phase: Phase::Fresh,
                    last_dir: None,
                });
                t.tiles[i].unit = Some(id);
            }
            _ => {}
        }
        if !new_city && tile.city.is_none() && t.tiles[i].unit.is_none() && coin(rng, 0.05) {
            new_city = true;
            let id = CityId(t.cities.len() as u32);
            t.cities.push(City {
                id,
                owner: o,
                pos: q,
                level: 1 + rng.gen_range(2),
                population: 2,
                is_capital: false,
                workshop: false,
                parks: 0,
                wall: false,
                border_growth_used: false,
                pending_reward: false,
                connected_to: None,
            });
            t.tiles[i].city = Some(id);
            t.tiles[i].village = false;
        }
    }

    // Restore opponent's public score through opponent's hidden explored mask.
    let delta = score(&t, o) as i64 - score(s, o) as i64;
    let weight = s.config.score_weights.exploration_per_tile as i64;
    if delta % weight != 0 {
        return None;
    }
    let mut k = delta / weight;
    let mask = &mut t.players[o as usize].explored;
    for e in mask.iter_mut().rev() {
        if k > 0 && *e {
            *e = false;
            k -= 1;
        } else if k < 0 && !*e {
            *e = true;
            k += 1;
        }
    }
    if k != 0 {
        return None;
    }
    assert_eq!(score(&t, o), score(s, o));
    assert_eq!(score(&t, v), score(s, v));
    Some(t)
}

fn assert_views_equal(a: &State, b: &State, v: PlayerId, what: &str) {
    let (x, y) = (PlayerView::new(a, v), PlayerView::new(b, v));
    assert_eq!(x.player(), y.player(), "{what}");
    assert_eq!(x.size(), y.size(), "{what}");
    assert_eq!(x.turn(), y.turn(), "{what}");
    assert_eq!(x.turn_cap(), y.turn_cap(), "{what}");
    assert_eq!(x.ruleset(), y.ruleset(), "{what}: ruleset");
    assert_eq!(x.current(), y.current(), "{what}");
    assert_eq!(x.is_terminal(), y.is_terminal(), "{what}");
    assert_eq!(x.stars(), y.stars(), "{what}: stars");
    assert_eq!(x.techs(), y.techs(), "{what}: techs");
    assert_eq!(x.income(), y.income(), "{what}: income");
    assert_eq!(x.score(0), y.score(0), "{what}: score 0");
    assert_eq!(x.score(1), y.score(1), "{what}: score 1");
    let units = |w: &PlayerView| w.my_units().into_iter().cloned().collect::<Vec<_>>();
    assert_eq!(units(&x), units(&y), "{what}: my_units");
    let cities = |w: &PlayerView| w.my_cities().into_iter().cloned().collect::<Vec<_>>();
    assert_eq!(cities(&x), cities(&y), "{what}: my_cities");
    assert_eq!(
        x.visible_units(),
        y.visible_units(),
        "{what}: visible_units"
    );
    assert_eq!(
        x.visible_cities(),
        y.visible_cities(),
        "{what}: visible_cities"
    );
    assert_eq!(*x.tables(), *y.tables(), "{what}: tables");
    assert_eq!(
        x.legal_actions(),
        y.legal_actions(),
        "{what}: legal_actions"
    );
    assert_eq!(
        x.pending_reward_city(),
        y.pending_reward_city(),
        "{what}: pending_reward_city"
    );
    for q in all_pos(a.size) {
        assert_eq!(
            x.is_explored(q),
            y.is_explored(q),
            "{what}: is_explored {q:?}"
        );
        assert_eq!(x.tile(q), y.tile(q), "{what}: tile {q:?}");
    }
    for u in a.units.iter().filter(|u| u.owner == v) {
        assert_eq!(
            x.reachable_tiles(u.id),
            y.reachable_tiles(u.id),
            "{what}: reachable_tiles {:?}",
            u.id
        );
    }
}

fn assert_bots_equal(a: &State, b: &State, what: &str) {
    let v = a.current;
    for k in KINDS {
        for (i, eps) in [0.0f32, 0.15, 0.6].into_iter().enumerate() {
            let bot = make_bot(k, eps);
            let seed = (a.turn as u64) << 8 | i as u64;
            let (mut ra, mut rb) = (Rng::new(seed), Rng::new(seed));
            let x = bot.choose(&PlayerView::new(a, v), &mut ra);
            let y = bot.choose(&PlayerView::new(b, v), &mut rb);
            assert_eq!(x, y, "{what}: {k:?} eps {eps}");
            assert_eq!(ra, rb, "{what}: {k:?} eps {eps} draws");
        }
    }
}

#[test]
fn v2_views_and_bots_ignore_hidden_v2_information() {
    let mut states = Vec::new();
    for seed in 0..5 {
        states.extend(random_playout_states(seed, &v2_cfg(20), 2000 + seed, 7));
    }
    for (seed, kinds, eps) in [
        (5, [BotKind::Rush, BotKind::Economy], 0.1),
        (6, [BotKind::Economy, BotKind::Rush], 0.1),
        (7, [BotKind::Rush, BotKind::Random], 0.0),
        (8, [BotKind::Random, BotKind::Economy], 0.0),
    ] {
        states.extend(bot_playout_states(seed, kinds, eps, &v2_cfg(25), 5));
    }
    assert!(states.len() > 150, "only {} states", states.len());

    let buckets = build_v2_tier_buckets(&states[0].tables);
    let mut rng = Rng::new(0x19_5eed);
    let (mut twins, mut skipped, mut v2_tile_edits) = (0, 0, 0);
    for (n, s) in states.iter().enumerate() {
        for v in 0..2u8 {
            let Some(t) = (0..8).find_map(|_| try_v2_twin(s, v, &buckets, &mut rng)) else {
                skipped += 1;
                continue;
            };
            twins += 1;
            v2_tile_edits += s
                .tiles
                .iter()
                .zip(&t.tiles)
                .filter(|(a, b)| {
                    a.road != b.road || a.building != b.building || a.monument != b.monument
                })
                .count();
            let what = format!("v2 state {n} viewer {v}");
            assert_views_equal(s, &t, v, &what);
            if v == s.current {
                assert_bots_equal(s, &t, &what);
            }
        }
    }
    assert!(
        skipped * 10 <= twins,
        "too many v2 twins skipped: {skipped} of {twins}"
    );
    assert!(
        v2_tile_edits > 500,
        "only {v2_tile_edits} hidden v2 tile fields mutated"
    );
}

#[test]
fn v2_hidden_opponent_monument_and_player_fields_never_leak() {
    // Explicit targeted twin where opponent owns an unexplored monument (+400 score), balanced by
    // removing 4 tier-1 techs (-400 score), plus differing kills, lighthouses_discovered,
    // monuments_unlocked, monuments_placed, unexplored road/building, and deep Knight.
    let base = StateBuilder::new(9)
        .config(v2_cfg(30))
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(7, 7), |c| c.is_capital = true)
        .unit(0, UnitType::Warrior, p(1, 1))
        .explore(0, &rect(0, 3, 0, 3))
        .explore(1, &all_pos(9))
        .techs(
            1,
            &[
                Tech::Climbing,
                Tech::Hunting,
                Tech::Organization,
                Tech::Riding,
            ],
        )
        .stars(0, 10)
        .build();

    let mut twin = base.clone();
    // Replace opponent's 400 tech score with a 400-point monument on unexplored opponent territory
    // (6,6), plus road, building, hidden player counters, and a deep hidden Knight.
    twin.players[1].techs = TechSet::new();
    twin.players[1].kills = 12;
    twin.players[1].lighthouses_discovered = 0b1111;
    twin.players[1]
        .monuments_unlocked
        .insert(Monument::ParkOfFortune);
    twin.players[1]
        .monuments_unlocked
        .insert(Monument::GateOfPower);
    twin.players[1]
        .monuments_placed
        .insert(Monument::ParkOfFortune);
    let m_idx = idx(&twin, p(6, 6));
    twin.tiles[m_idx].monument = Some(Monument::ParkOfFortune);
    twin.tiles[m_idx].road = true;
    let b_idx = idx(&twin, p(6, 7));
    twin.tiles[b_idx].building = Some(Building::Market);
    twin.tiles[b_idx].building_level = 4;
    assert_eq!(score(&base, 1), score(&twin, 1));
    assert_eq!(score(&base, 0), score(&twin, 0));

    assert_views_equal(&base, &twin, 0, "targeted hidden v2 monument");
    assert_bots_equal(&base, &twin, "targeted hidden v2 monument");
}
