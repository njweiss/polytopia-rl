//! Milestone 8: Economy bot rule goldens (docs/MILESTONE_8 "Common bot machinery", "Economy
//! bot"; "Tests" item 2). Each board is scripted so that the rule under test is the first that
//! yields a legal action; the expected `Action` is derived by hand from the rule text. Combat
//! numbers are the spec formula worked by hand (defense bonus 1.0, f32, half-up):
//! - Warrior 10/10 vs Warrior 10/10: dealt 5, taken 5, score 0
//! - Warrior 10/10 vs Warrior 8/10: dealt 5, taken 4, score 1
//! - Warrior 10/10 vs veteran Warrior 7/15: dealt 6.14 -> 6 < 7, taken 2.86 -> 3, score 3
//! - Warrior 10/10 vs Rider 10/10: dealt 6, taken 2, score 4 (with the real wall bonus 4 it
//!   would be dealt 3, taken 3, score 0)
//! - Archer 10/10 vs Warrior 10/10: dealt 5; taken 5 at distance 1 (score 0), 0 at distance 2
//!   (outside the Warrior's range 1, score 5)
//! - Warrior 10/10 vs Warrior 2/10: dealt 8, a kill, score 1008

mod common;

use bots::EconomyBot;
use common::*;
use sim_core::actions::Action;
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::map::{Resource, Terrain};
use sim_core::state::{CityId, Phase, UnitId};
use sim_core::testing::StateBuilder;

const W: UnitType = UnitType::Warrior;

#[test]
fn reward_preferences_per_level() {
    for (level, r) in [
        (2, Reward::Workshop),
        (3, Reward::Resources),
        (4, Reward::PopulationGrowth),
        (5, Reward::Park),
        (7, Reward::Park),
    ] {
        let s = open(5)
            .city_with(0, p(2, 2), |c| {
                c.level = level;
                c.pending_reward = true;
            })
            .build();
        assert_choice(
            &EconomyBot,
            &s,
            Action::ChooseReward {
                city: CityId(0),
                reward: r,
            },
        );
    }
}

#[test]
fn promote_then_capture_before_attack() {
    let promote = open(7)
        .village(p(1, 1))
        .unit(0, W, p(1, 1))
        .unit_with(0, W, p(5, 5), |u| {
            u.promotion_available = true;
            u.phase = Phase::Done;
        })
        .unit_with(1, W, p(1, 2), |u| u.health = 2)
        .build();
    assert_choice(&EconomyBot, &promote, Action::Promote { unit: UnitId(1) });
    let capture = open(7)
        .village(p(5, 1))
        .city(1, p(5, 5))
        .unit(0, W, p(1, 1))
        .unit(0, W, p(5, 5))
        .unit(0, W, p(5, 1))
        .unit_with(1, W, p(1, 2), |u| u.health = 2)
        .build();
    assert_choice(&EconomyBot, &capture, Action::Capture { unit: UnitId(1) });
}

#[test]
fn kills_are_always_acceptable() {
    let s = open(7)
        .unit(0, W, p(3, 3))
        .unit_with(1, W, p(3, 4), |u| u.health = 2)
        .build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 4),
        },
    );
}

#[test]
fn threshold_three_refuses_scores_0_and_1() {
    // Nothing else applies (no cities, nothing unexplored, no village), so the turn ends.
    for health in [10, 8] {
        let s = open(7)
            .unit(0, W, p(3, 3))
            .unit_with(1, W, p(3, 4), |u| u.health = health)
            .build();
        assert_choice(&EconomyBot, &s, Action::EndTurn);
    }
}

#[test]
fn threshold_three_accepts_score_3() {
    let s = open(7)
        .unit(0, W, p(3, 3))
        .unit_with(1, W, p(3, 4), |u| {
            u.health = 7;
            u.max_health = 15;
            u.veteran = true;
        })
        .build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 4),
        },
    );
}

#[test]
fn defense_bonus_is_estimated_as_1() {
    // The enemy Rider stands in its walled city (real bonus 4); the bot estimates score 4.
    let s = open(7)
        .city_with(1, p(3, 4), |c| c.wall = true)
        .unit(0, W, p(3, 3))
        .unit(1, UnitType::Rider, p(3, 4))
        .build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 4),
        },
    );
}

#[test]
fn no_retaliation_from_outside_the_defender_range() {
    let s = open(7)
        .unit(0, UnitType::Archer, p(3, 3))
        .unit(1, W, p(3, 5))
        .build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(3, 5),
        },
    );
    let s = open(7)
        .unit(0, UnitType::Archer, p(3, 3))
        .unit(1, W, p(3, 4))
        .build();
    assert_choice(&EconomyBot, &s, Action::EndTurn);
}

/// City (3,3) with an own Done unit on it, every relevant tech and 5 stars.
fn site_board() -> StateBuilder {
    open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
        .techs(
            0,
            &[
                Tech::Organization,
                Tech::Hunting,
                Tech::Climbing,
                Tech::Farming,
                Tech::Mining,
                Tech::Forestry,
            ],
        )
        .stars(0, 5)
}

fn site(at: sim_core::grid::Pos, action: SiteAction) -> Action {
    Action::Site { at, action }
}

#[test]
fn site_kind_order() {
    let metal = |b: StateBuilder| {
        b.terrain(p(2, 2), Terrain::Mountain)
            .resource(p(2, 2), Resource::Metal)
    };
    let fruit = |b: StateBuilder| b.resource(p(2, 3), Resource::Fruit);
    let animal = |b: StateBuilder| {
        b.terrain(p(2, 4), Terrain::Forest)
            .resource(p(2, 4), Resource::Animal)
    };
    let forest = |b: StateBuilder| b.terrain(p(3, 2), Terrain::Forest);
    let crop = |b: StateBuilder| b.resource(p(4, 4), Resource::Crop);

    let s = crop(forest(animal(fruit(metal(site_board()))))).build();
    assert_choice(&EconomyBot, &s, site(p(4, 4), SiteAction::Farm));
    let s = forest(animal(fruit(metal(site_board())))).build();
    assert_choice(&EconomyBot, &s, site(p(2, 2), SiteAction::Mine));
    let s = forest(animal(fruit(site_board()))).build();
    assert_choice(&EconomyBot, &s, site(p(2, 3), SiteAction::HarvestFruit));
    let s = forest(animal(site_board())).build();
    assert_choice(&EconomyBot, &s, site(p(2, 4), SiteAction::HarvestAnimal));
    // Forests at (2,4) and (3,2): lumber hut on the first in row-major order.
    let s = forest(site_board())
        .terrain(p(2, 4), Terrain::Forest)
        .build();
    assert_choice(&EconomyBot, &s, site(p(2, 4), SiteAction::LumberHut));
}

#[test]
fn site_tiles_row_major_within_a_kind() {
    let s = site_board()
        .resource(p(4, 2), Resource::Fruit)
        .resource(p(2, 3), Resource::Fruit)
        .build();
    assert_choice(&EconomyBot, &s, site(p(2, 3), SiteAction::HarvestFruit));
}

#[test]
fn economy_never_clears_forest() {
    // Clear forest (cost 0) is legal; lumber hut (3) and research (5+) are not affordable; two
    // units for one city means no training.
    let s = open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(0, 0), |u| u.phase = Phase::Done)
        .techs(0, &[Tech::Organization, Tech::Hunting, Tech::Forestry])
        .stars(0, 2)
        .terrain(p(2, 2), Terrain::Forest)
        .build();
    assert!(sim_core::actions::legal_actions(&s).contains(&site(p(2, 2), SiteAction::ClearForest)));
    assert_choice(&EconomyBot, &s, Action::EndTurn);
}

#[test]
fn research_order() {
    // One city: tier 1 costs 5, tier 2 costs 6. Two own units block training.
    use Tech::*;
    let cases: &[(&[Tech], u32, Tech)] = &[
        (&[Organization], 5, Hunting),
        (&[Organization], 6, Hunting),
        (&[Organization, Hunting], 5, Climbing),
        (&[Organization, Hunting], 6, Farming),
        (&[Organization, Hunting, Farming], 6, Climbing),
        (&[Organization, Hunting, Farming, Climbing], 6, Mining),
        (
            &[Organization, Hunting, Farming, Climbing, Mining],
            6,
            Forestry,
        ),
        (
            &[Organization, Hunting, Farming, Climbing, Mining, Forestry],
            6,
            Riding,
        ),
        (
            &[
                Organization,
                Hunting,
                Farming,
                Climbing,
                Mining,
                Forestry,
                Riding,
            ],
            6,
            Strategy,
        ),
        (
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
            6,
            Archery,
        ),
    ];
    for &(techs, stars, expected) in cases {
        let s = open(7)
            .city(0, p(3, 3))
            .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
            .unit_with(0, W, p(0, 0), |u| u.phase = Phase::Done)
            .techs(0, techs)
            .stars(0, stars)
            .build();
        assert_choice(&EconomyBot, &s, Action::Research { tech: expected });
    }
}

#[test]
fn train_warriors_up_to_two_units_per_city() {
    // One city, one unit, 3 stars and Riding: a Warrior, not a Rider.
    let one = open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(0, 0), |u| u.phase = Phase::Done)
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(0, 3)
        .build();
    assert_choice(
        &EconomyBot,
        &one,
        Action::Train {
            city: CityId(0),
            kind: W,
        },
    );
    // Two units for one city: no training even though it is legal.
    let two = open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(0, 0), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(0, 6), |u| u.phase = Phase::Done)
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(0, 3)
        .build();
    assert_choice(&EconomyBot, &two, Action::EndTurn);
    // Two cities, three units: the first city that can train (city 0 is occupied).
    let s = open(7)
        .city(0, p(1, 1))
        .city(0, p(5, 5))
        .unit_with(0, W, p(1, 1), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(0, 6), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(6, 0), |u| u.phase = Phase::Done)
        .techs(0, &[Tech::Organization])
        .stars(0, 2)
        .build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Train {
            city: CityId(1),
            kind: W,
        },
    );
}

#[test]
fn recover_lowest_hurt_unit_with_no_enemy_within_2() {
    let s = open(7)
        .unit_with(0, W, p(0, 0), |u| {
            u.health = 4;
            u.phase = Phase::Done;
        })
        .unit_with(0, W, p(0, 6), |u| u.health = 5)
        .unit_with(0, W, p(2, 2), |u| u.health = 4)
        .unit_with(0, W, p(6, 0), |u| u.health = 4)
        .unit(1, W, p(4, 4))
        .build();
    assert_choice(&EconomyBot, &s, Action::Recover { unit: UnitId(3) });
}

fn mv(unit: u32, row: u8, col: u8) -> Action {
    Action::Move {
        unit: UnitId(unit),
        to: p(row, col),
    }
}

#[test]
fn move_toward_the_nearest_village() {
    // (8,8) at 4: the only strictly closer neighbor of (4,4) is (5,5).
    let s = open(9).village(p(8, 8)).unit(0, W, p(4, 4)).build();
    assert_choice(&EconomyBot, &s, mv(0, 5, 5));
}

#[test]
fn equally_near_villages_break_ties_row_major() {
    // (4,8) and (4,0) both at 4; (4,0) first -> (3,3), (4,3), (5,3) at 3 -> (3,3).
    let s = open(9)
        .village(p(4, 8))
        .village(p(4, 0))
        .unit(0, W, p(4, 4))
        .build();
    assert_choice(&EconomyBot, &s, mv(0, 3, 3));
}

#[test]
fn occupied_and_unexplored_villages_are_not_targets() {
    // (2,4) holds an own unit, (7,4) an enemy unit, (3,1) is unexplored; (8,8) is the target.
    let s = open(9)
        .village(p(2, 4))
        .village(p(7, 4))
        .village(p(3, 1))
        .village(p(8, 8))
        .unit(0, W, p(4, 4))
        .unit_with(0, W, p(2, 4), |u| u.phase = Phase::Done)
        .unit(1, W, p(7, 4))
        .hide(0, &[p(3, 1)])
        .build();
    assert_choice(&EconomyBot, &s, mv(0, 5, 5));
}

#[test]
fn explore_without_a_village() {
    // Same fog as the Rush test: explored rows 2..=5, columns 3..=5 -> (5,3). Enemy cities and
    // units are not Economy targets.
    let s = StateBuilder::new(9)
        .explore(0, &rect(2, 5, 3, 5))
        .city(1, p(2, 3))
        .unit(0, W, p(4, 4))
        .build();
    assert_choice(&EconomyBot, &s, mv(0, 5, 3));
}

#[test]
fn explore_fallback_when_greedy_step_fails() {
    // Village (4,6) at 2 behind own Done units; (3,4) and (5,4) are only as close, so
    // greedy_step is None. Rows 0..=2 are unexplored: of the reachable (3,3), (3,4), (4,3),
    // (5,3), (5,4) only the row-3 tiles see row 2 (gain 3 each) -> (3,3).
    let s = StateBuilder::new(9)
        .explore(0, &rect(3, 8, 0, 8))
        .village(p(4, 6))
        .unit(0, W, p(4, 4))
        .unit_with(0, W, p(3, 5), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(4, 5), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(5, 5), |u| u.phase = Phase::Done)
        .build();
    assert_eq!(my_vision_gain(&s, 0, p(3, 3)), 3);
    assert_eq!(my_vision_gain(&s, 0, p(3, 4)), 3);
    assert_choice(&EconomyBot, &s, mv(0, 3, 3));
}

#[test]
fn economy_ignores_enemy_cities_and_units() {
    let s = open(9)
        .city(1, p(6, 4))
        .unit(0, W, p(4, 4))
        .unit(1, W, p(0, 4))
        .build();
    assert_choice(&EconomyBot, &s, Action::EndTurn);
}

#[test]
fn economy_plays_player_1_too() {
    let s = open(9)
        .village(p(8, 8))
        .unit(1, W, p(4, 4))
        .current(1)
        .build();
    assert_choice(&EconomyBot, &s, mv(0, 5, 5));
}

#[test]
fn end_turn_when_nothing_applies() {
    let s = open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(0, 0), |u| u.phase = Phase::Done)
        .build();
    assert_choice(&EconomyBot, &s, Action::EndTurn);
}

#[test]
fn attack_comes_before_sites_research_and_training() {
    // A kill (score 1008) while a fruit harvest, research (5 stars) and a Warrior are all legal.
    let s = open(7)
        .city(0, p(3, 3))
        .resource(p(2, 2), Resource::Fruit)
        .unit(0, W, p(0, 0))
        .unit_with(1, W, p(0, 1), |u| u.health = 2)
        .techs(0, &[Tech::Organization])
        .stars(0, 5)
        .build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(0, 1),
        },
    );
}

#[test]
fn research_comes_before_training() {
    // One unit for one city and 5 stars: Hunting (5) is researched before a Warrior (2).
    let s = open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(0, 0), |u| u.phase = Phase::Done)
        .techs(0, &[Tech::Organization])
        .stars(0, 5)
        .build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Research {
            tech: Tech::Hunting,
        },
    );
}

#[test]
fn training_comes_before_recover_and_recover_before_move() {
    // A hurt unit with a village to head for; with 2 stars a Warrior is legal first.
    let base = || {
        open(9)
            .city(0, p(1, 1))
            .village(p(8, 8))
            .unit_with(0, W, p(4, 4), |u| u.health = 4)
            .techs(0, &[Tech::Organization])
    };
    let s = base().stars(0, 2).build();
    assert_choice(
        &EconomyBot,
        &s,
        Action::Train {
            city: CityId(0),
            kind: W,
        },
    );
    let s = base().build();
    assert_choice(&EconomyBot, &s, Action::Recover { unit: UnitId(0) });
}
