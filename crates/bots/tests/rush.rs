//! Milestone 8: Rush bot rule goldens (docs/MILESTONE_8 "Common bot machinery", "Rush bot";
//! "Tests" item 2). Each board is scripted so that the rule under test is the first that yields a
//! legal action, and the expected `Action` is derived by hand from the rule text. Combat numbers
//! are the spec formula worked by hand (defense bonus 1.0, f32, half-up):
//! - Warrior 10/10 vs Warrior 10/10: dealt 4.5 -> 5, taken 4.5 -> 5, score 0
//! - Warrior 10/10 vs Warrior 8/10: dealt 5, taken 4.0000005 -> 4, score 1
//! - Warrior 10/10 vs Warrior 6/10: dealt 5.625 -> 6 >= 6, a kill, score 1006
//! - Warrior 10/10 vs Warrior 4/10: dealt 6.43 -> 6, a kill, score 1006
//! - Warrior 10/10 vs Warrior 2/10: dealt 7.5 -> 8, a kill, score 1008
//! - Warrior 10/10 vs Rider 10/10: dealt 6, taken 1.5 -> 2, score 4
//! - Warrior 4/10 vs Warrior 10/10: dealt 2.57 -> 3, taken 6.43 -> 6, score -3

mod common;

use bots::RushBot;
use common::*;
use sim_core::actions::Action;
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::map::{Resource, Terrain};
use sim_core::state::{CityId, Phase, UnitId};

const W: UnitType = UnitType::Warrior;

fn reward(level: u32) -> Action {
    let s = open(5)
        .city_with(0, p(2, 2), |c| {
            c.level = level;
            c.pending_reward = true;
        })
        .unit(0, W, p(0, 0))
        .build();
    choice(&RushBot, &s)
}

#[test]
fn reward_preferences_per_level() {
    let city = CityId(0);
    for (level, r) in [
        (2, Reward::Explorer),
        (3, Reward::Resources),
        (4, Reward::PopulationGrowth),
        (5, Reward::SuperUnit),
        (6, Reward::SuperUnit),
    ] {
        assert_eq!(
            reward(level),
            Action::ChooseReward { city, reward: r },
            "level {level}"
        );
    }
}

#[test]
fn pending_reward_of_a_second_city() {
    let s = open(7)
        .city(0, p(1, 1))
        .city_with(0, p(5, 5), |c| {
            c.level = 3;
            c.pending_reward = true;
        })
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::ChooseReward {
            city: CityId(1),
            reward: Reward::Resources,
        },
    );
}

#[test]
fn promote_lowest_id_before_capture_and_attack() {
    // Unit 0 stands on a village (Capture legal) next to a killable enemy; units 1 and 2 can
    // promote (unit 1 is Done: promotion is legal in any phase).
    let s = open(7)
        .village(p(1, 1))
        .unit(0, W, p(1, 1))
        .unit_with(0, W, p(5, 5), |u| {
            u.promotion_available = true;
            u.phase = Phase::Done;
        })
        .unit_with(0, W, p(3, 5), |u| u.promotion_available = true)
        .unit_with(1, W, p(1, 2), |u| u.health = 2)
        .build();
    assert_choice(&RushBot, &s, Action::Promote { unit: UnitId(1) });
}

#[test]
fn capture_lowest_id_before_attack() {
    // Unit 0 can kill (score 1008) but not capture; unit 1 stands on a village, unit 2 on an
    // enemy city.
    let s = open(7)
        .village(p(5, 1))
        .city(1, p(5, 5))
        .unit(0, W, p(1, 1))
        .unit(0, W, p(5, 1))
        .unit(0, W, p(5, 5))
        .unit_with(1, W, p(1, 2), |u| u.health = 2)
        .build();
    assert_choice(&RushBot, &s, Action::Capture { unit: UnitId(1) });
}

#[test]
fn kill_beats_any_non_kill() {
    // Unit 0 vs Rider 10/10 (score 4), unit 1 vs Warrior 6/10 (dealt 6 >= 6: kill, 1006).
    let s = open(7)
        .unit(0, W, p(1, 1))
        .unit(0, W, p(5, 5))
        .unit(1, UnitType::Rider, p(1, 2))
        .unit_with(1, W, p(6, 6), |u| u.health = 6)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Attack {
            unit: UnitId(1),
            target: p(6, 6),
        },
    );
}

#[test]
fn larger_kill_damage_wins() {
    // Unit 0 kills a 4-health Warrior (1006), unit 1 a 2-health Warrior (1008).
    let s = open(7)
        .unit(0, W, p(1, 1))
        .unit(0, W, p(5, 5))
        .unit_with(1, W, p(1, 2), |u| u.health = 4)
        .unit_with(1, W, p(6, 6), |u| u.health = 2)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Attack {
            unit: UnitId(1),
            target: p(6, 6),
        },
    );
}

#[test]
fn equal_scores_go_to_the_lowest_attacker_id() {
    // Both attacks score 1; unit 1's target is earlier in row-major order.
    let s = open(7)
        .unit(0, W, p(4, 4))
        .unit(0, W, p(1, 1))
        .unit_with(1, W, p(0, 0), |u| u.health = 8)
        .unit_with(1, W, p(5, 5), |u| u.health = 8)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(5, 5),
        },
    );
}

#[test]
fn equal_scores_for_one_attacker_go_to_the_row_major_target() {
    // The (4,2) enemy has the lower id; (2,4) comes first in row-major order.
    let s = open(7)
        .unit(0, W, p(3, 3))
        .unit_with(1, W, p(4, 2), |u| u.health = 8)
        .unit_with(1, W, p(2, 4), |u| u.health = 8)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(2, 4),
        },
    );
}

#[test]
fn threshold_zero_accepts_scores_0_and_1() {
    for health in [10, 8] {
        let s = open(7)
            .unit(0, W, p(3, 3))
            .unit_with(1, W, p(3, 4), |u| u.health = health)
            .build();
        assert_choice(
            &RushBot,
            &s,
            Action::Attack {
                unit: UnitId(0),
                target: p(3, 4),
            },
        );
    }
}

#[test]
fn negative_score_is_not_taken() {
    // Score -3: no attack. The hurt unit cannot recover with the enemy adjacent, greedy_step has
    // no strictly closer tile and nothing is unexplored, so the turn ends.
    let s = open(7)
        .unit_with(0, W, p(3, 3), |u| u.health = 4)
        .unit(1, W, p(3, 4))
        .build();
    assert_choice(&RushBot, &s, Action::EndTurn);
}

#[test]
fn train_rider_when_affordable() {
    let s = open(7)
        .city(0, p(3, 3))
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(0, 3)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Train {
            city: CityId(0),
            kind: UnitType::Rider,
        },
    );
}

#[test]
fn train_warrior_when_no_rider_is_legal() {
    // 2 stars with Riding (a Rider costs 3), and a harvestable fruit that must wait.
    let s = open(7)
        .city(0, p(3, 3))
        .resource(p(2, 2), Resource::Fruit)
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(0, 2)
        .build();
    let warrior = Action::Train {
        city: CityId(0),
        kind: W,
    };
    assert_choice(&RushBot, &s, warrior);
    // 5 stars without Riding: Train comes before Research Riding (cost 5); Archery does not lead
    // to an Archer.
    let s = open(7)
        .city(0, p(3, 3))
        .techs(0, &[Tech::Organization, Tech::Hunting, Tech::Archery])
        .stars(0, 5)
        .build();
    assert_choice(&RushBot, &s, warrior);
}

#[test]
fn train_in_the_first_city_that_can() {
    let s = open(7)
        .city(0, p(1, 1))
        .city(0, p(5, 5))
        .unit_with(0, W, p(1, 1), |u| u.phase = Phase::Done)
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(0, 3)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Train {
            city: CityId(1),
            kind: UnitType::Rider,
        },
    );
}

#[test]
fn research_riding_before_harvesting_and_no_other_research() {
    let base = || {
        open(7)
            .city(0, p(3, 3))
            .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
            .resource(p(2, 2), Resource::Fruit)
            .stars(0, 5)
    };
    // Riding costs 4 + 1 city x tier 1 = 5.
    let s = base().techs(0, &[Tech::Organization]).build();
    assert_choice(&RushBot, &s, Action::Research { tech: Tech::Riding });
    // With Riding, other research (Climbing, Hunting: 5) is never chosen: harvest instead.
    let s = base().techs(0, &[Tech::Organization, Tech::Riding]).build();
    assert_choice(
        &RushBot,
        &s,
        Action::Site {
            at: p(2, 2),
            action: SiteAction::HarvestFruit,
        },
    );
}

fn site_board() -> sim_core::testing::StateBuilder {
    open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
        .techs(0, &[Tech::Organization, Tech::Hunting, Tech::Riding])
        .stars(0, 4)
}

#[test]
fn harvest_fruit_before_animal() {
    let s = site_board()
        .terrain(p(2, 2), Terrain::Forest)
        .resource(p(2, 2), Resource::Animal)
        .resource(p(4, 4), Resource::Fruit)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Site {
            at: p(4, 4),
            action: SiteAction::HarvestFruit,
        },
    );
    let s = site_board()
        .terrain(p(2, 2), Terrain::Forest)
        .resource(p(2, 2), Resource::Animal)
        .resource(p(4, 2), Resource::Fruit)
        .resource(p(2, 4), Resource::Fruit)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Site {
            at: p(2, 4),
            action: SiteAction::HarvestFruit,
        },
    );
}

#[test]
fn harvest_animal_row_major() {
    let s = site_board()
        .terrain(p(4, 4), Terrain::Forest)
        .resource(p(4, 4), Resource::Animal)
        .terrain(p(2, 3), Terrain::Forest)
        .resource(p(2, 3), Resource::Animal)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Site {
            at: p(2, 3),
            action: SiteAction::HarvestAnimal,
        },
    );
}

#[test]
fn rush_ignores_other_site_actions() {
    // Lumber hut, clear forest and farm are legal; Rush takes none of them.
    let s = open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
        .techs(
            0,
            &[
                Tech::Organization,
                Tech::Hunting,
                Tech::Riding,
                Tech::Forestry,
                Tech::Farming,
            ],
        )
        .stars(0, 5)
        .terrain(p(2, 2), Terrain::Forest)
        .resource(p(4, 4), Resource::Crop)
        .build();
    assert_choice(&RushBot, &s, Action::EndTurn);
}

#[test]
fn recover_lowest_hurt_unit_with_no_enemy_within_2() {
    // 0: hurt but Done; 1: health 5 of 10 is not hurt (10 < 10 is false); 2: hurt, enemy at
    // distance 2; 3: hurt, enemy at distance 4.
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
    assert_choice(&RushBot, &s, Action::Recover { unit: UnitId(3) });
}

#[test]
fn recover_ignores_enemies_on_unexplored_tiles() {
    let hidden = open(7)
        .unit_with(0, W, p(2, 2), |u| u.health = 4)
        .unit(1, W, p(4, 4))
        .hide(0, &[p(4, 4)])
        .build();
    assert_choice(&RushBot, &hidden, Action::Recover { unit: UnitId(0) });
    // Visible: no recover; the enemy unit is the move target, (3,3) the only closer tile.
    let visible = open(7)
        .unit_with(0, W, p(2, 2), |u| u.health = 4)
        .unit(1, W, p(4, 4))
        .build();
    assert_choice(
        &RushBot,
        &visible,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 3),
        },
    );
}

#[test]
fn move_targets_the_enemy_capital_first() {
    // Capital (0,8): the only strictly closer neighbor of (4,4) is (3,5). The nearer city (6,4)
    // would give (5,3).
    let s = open(9)
        .city_with(1, p(0, 8), |c| c.is_capital = true)
        .city(1, p(6, 4))
        .unit(0, W, p(4, 4))
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 5),
        },
    );
}

#[test]
fn move_ignores_a_capital_on_an_unexplored_tile() {
    // Hidden capital (8,0) would give (5,3). Visible: enemy unit (4,7) at 3 beats city (0,4)
    // at 4; closer tiles (3,5), (4,5), (5,5) at 2 -> (3,5).
    let s = open(9)
        .city_with(1, p(8, 0), |c| c.is_capital = true)
        .city(1, p(0, 4))
        .unit(0, W, p(4, 4))
        .unit(1, W, p(4, 7))
        .hide(0, &[p(8, 0)])
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 5),
        },
    );
}

#[test]
fn move_toward_the_nearest_city_or_unit() {
    // City (6,4) at 2 beats unit (0,8) at 4: (5,3), (5,4), (5,5) are at 1 -> (5,3).
    let s = open(9)
        .city(1, p(6, 4))
        .unit(0, W, p(4, 4))
        .unit(1, W, p(0, 8))
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(5, 3),
        },
    );
}

#[test]
fn equally_near_targets_break_ties_row_major() {
    // Units at (7,4) (lower id) and (1,4), both at 3. Target (1,4): (3,3), (3,4), (3,5) at 2 ->
    // (3,3). Target (7,4) would give (5,3).
    let s = open(9)
        .unit(0, W, p(4, 4))
        .unit(1, W, p(7, 4))
        .unit(1, W, p(1, 4))
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 3),
        },
    );
}

#[test]
fn greedy_step_takes_the_smallest_distance() {
    // A Rider reaches radius 2. Distance to (4,8) is smallest (2) in column 6, rows 2..=6 ->
    // (2,6). The first strictly closer tile in row-major order would be (2,5).
    let s = open(9)
        .unit(0, UnitType::Rider, p(4, 4))
        .unit(1, W, p(4, 8))
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(2, 6),
        },
    );
}

#[test]
fn greedy_step_must_be_strictly_closer() {
    // Target (4,6) at 2; (3,5), (4,5), (5,5) hold own Done units; (3,4) and (5,4) are only as
    // close, never closer. Everything is explored, so explore_step has nothing either.
    let s = open(9)
        .unit(0, W, p(4, 4))
        .unit_with(0, W, p(3, 5), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(4, 5), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(5, 5), |u| u.phase = Phase::Done)
        .unit(1, W, p(4, 6))
        .build();
    assert_choice(&RushBot, &s, Action::EndTurn);
}

/// Player 0 has explored rows 2..=5, columns 3..=5 of a 9x9 board. Vision gains of the
/// neighbors of (4,4): (3,3) 3, (3,4) 0, (3,5) 3, (4,3) 3, (5,3) 5, (5,4) 3, (5,5) 5.
fn fog_board() -> sim_core::testing::StateBuilder {
    sim_core::testing::StateBuilder::new(9)
        .explore(0, &rect(2, 5, 3, 5))
        .explore(1, &all_pos(9))
}

#[test]
fn explore_fallback_when_greedy_step_fails() {
    // Score -3 attack refused, hurt unit next to a visible enemy cannot recover, the target is
    // adjacent so greedy_step is None: explore_step -> (5,3) (gain 5, ties (5,5) row-major).
    let s = fog_board()
        .unit_with(0, W, p(4, 4), |u| u.health = 4)
        .unit(1, W, p(4, 5))
        .build();
    assert_eq!(my_vision_gain(&s, 0, p(5, 3)), 5);
    assert_eq!(my_vision_gain(&s, 0, p(5, 5)), 5);
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(5, 3),
        },
    );
}

#[test]
fn explore_without_a_target() {
    // An enemy unit on an unexplored tile is not a target.
    let s = fog_board().unit(0, W, p(4, 4)).unit(1, W, p(0, 0)).build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(5, 3),
        },
    );
}

#[test]
fn explore_counts_radius_2_from_a_mountain() {
    // Explored rows 3..=5, columns 3..=5. Field neighbors of (4,4) gain at most 5; the mountain
    // (5,5) gains 25 - 9 = 16 with Climbing (it cannot be entered without).
    let board = || {
        sim_core::testing::StateBuilder::new(9)
            .explore(0, &rect(3, 5, 3, 5))
            .terrain(p(5, 5), Terrain::Mountain)
            .unit(0, W, p(4, 4))
    };
    let s = board().techs(0, &[Tech::Climbing]).build();
    assert_eq!(my_vision_gain(&s, 0, p(5, 5)), 16);
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(5, 5),
        },
    );
    // Without Climbing: (3,3), (3,5), (5,3) gain 5 each -> (3,3).
    let s = board().build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 3),
        },
    );
}

#[test]
fn only_fresh_units_move() {
    // Unit 0 has Moved (it still has reachable tiles, but no legal move). Unit 1 heads for the
    // enemy unit (8,8): (2,2) is the only strictly closer tile.
    let s = open(9)
        .unit_with(0, W, p(4, 4), |u| u.phase = Phase::Moved)
        .unit(0, W, p(1, 1))
        .unit(1, W, p(8, 8))
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(1),
            to: p(2, 2),
        },
    );
}

#[test]
fn rush_plays_player_1_too() {
    let s = open(9)
        .city_with(0, p(0, 8), |c| c.is_capital = true)
        .city(0, p(6, 4))
        .unit(1, W, p(4, 4))
        .current(1)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 5),
        },
    );
}

#[test]
fn end_turn_when_nothing_applies() {
    let s = open(7)
        .city(0, p(3, 3))
        .unit_with(0, W, p(3, 3), |u| u.phase = Phase::Done)
        .unit_with(0, W, p(0, 0), |u| u.phase = Phase::Done)
        .unit(1, W, p(6, 6))
        .build();
    assert_choice(&RushBot, &s, Action::EndTurn);
}

#[test]
fn attack_comes_before_training() {
    // Score 1 attack (Warrior 10 vs Warrior 8) and a legal Rider train.
    let s = open(7)
        .city(0, p(3, 3))
        .unit(0, W, p(0, 0))
        .unit_with(1, W, p(0, 1), |u| u.health = 8)
        .techs(0, &[Tech::Organization, Tech::Riding])
        .stars(0, 3)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(0, 1),
        },
    );
}

#[test]
fn harvest_comes_before_recover_and_recover_before_move() {
    // A hurt unit far from enemies, a fruit to harvest and a visible enemy to move toward.
    let base = || {
        open(9)
            .city(0, p(1, 1))
            .unit_with(0, W, p(1, 1), |u| u.phase = Phase::Done)
            .unit_with(0, W, p(4, 4), |u| u.health = 4)
            .unit(1, W, p(8, 8))
            .techs(0, &[Tech::Organization, Tech::Riding])
    };
    let s = base()
        .resource(p(0, 0), Resource::Fruit)
        .stars(0, 2)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Site {
            at: p(0, 0),
            action: SiteAction::HarvestFruit,
        },
    );
    let s = base().build();
    assert_choice(&RushBot, &s, Action::Recover { unit: UnitId(1) });
}

#[test]
fn dealt_equal_to_health_is_a_kill() {
    // Unit 0: Warrior 10 vs Warrior 6, dealt 6 = health: a kill, 1006. Unit 1: Warrior 7/10 vs
    // Warrior 4/10: attack force 1.4, defense force 0.8, dealt 1.4/2.2*2*4.5 = 5.73 -> 6, a
    // kill, 1006. The tie goes to unit 0; were an exact hit not a kill, unit 0 would score 6.
    let s = open(7)
        .unit(0, W, p(1, 1))
        .unit_with(0, W, p(5, 5), |u| u.health = 7)
        .unit_with(1, W, p(1, 2), |u| u.health = 6)
        .unit_with(1, W, p(6, 6), |u| u.health = 4)
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Attack {
            unit: UnitId(0),
            target: p(1, 2),
        },
    );
}

#[test]
fn explore_heads_for_the_nearest_fog_when_no_reachable_tile_sees_any() {
    // Q10. Only (8,8) is unexplored: no neighbor of (4,4) sees it. Tiles with gain: (7,7) at 3,
    // (7,8) and (8,7) at 4 -> f = (7,7); greedy_step from (4,4): (5,5) is the only tile at 2.
    let s = open(9).hide(0, &[p(8, 8)]).unit(0, W, p(4, 4)).build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(5, 5),
        },
    );
    // (0,8) and (8,0) unexplored: (1,7) and (7,1) both at 3 -> (1,7) row-major -> (3,5).
    let s = open(9)
        .hide(0, &[p(0, 8), p(8, 0)])
        .unit(0, W, p(4, 4))
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 5),
        },
    );
}

#[test]
fn the_fog_fallback_also_follows_a_failed_greedy_step() {
    // Score -3 attack refused, no recover next to the enemy, the target is adjacent: explore,
    // and with only (0,0) unexplored f = (1,1) at 3 -> greedy_step to (3,3).
    let s = open(9)
        .hide(0, &[p(0, 0)])
        .unit_with(0, W, p(4, 4), |u| u.health = 4)
        .unit(1, W, p(4, 5))
        .build();
    assert_choice(
        &RushBot,
        &s,
        Action::Move {
            unit: UnitId(0),
            to: p(3, 3),
        },
    );
}
