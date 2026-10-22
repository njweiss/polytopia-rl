//! Milestone 7: text renderer goldens (docs/MILESTONE_7 "Text renderer", G4, G5, "Tests" item 3).
//! Every board, panel line and description below was drawn by hand from the brief's formats.
//! Income and score are computed by hand in the comments from the spec's rules (income: level +
//! capital + workshop + parks; score: units 5 per star of cost / Super Unit 50, territory 20 per
//! tile, exploration 5 per explored tile, cities 100 + 50 per level above 1 + 5 per surplus
//! population over `L(L+1)/2 - 1`, parks 250, techs 100 per tier).

mod common;

use common::*;
use sim_core::actions::Action;
use sim_core::data::{Reward, SiteAction, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::{Resource, Terrain};
use sim_core::state::{CityId, Config, Improvement, Outcome, Phase, State, UnitId};
use sim_core::testing::StateBuilder;
use sim_tools::render::{action_label, describe_tile, render, View};

fn lines(ls: &[&str]) -> String {
    let mut s = ls.join("\n");
    s.push('\n');
    s
}

fn tiles(rows: std::ops::RangeInclusive<u8>, cols: std::ops::RangeInclusive<u8>) -> Vec<Pos> {
    rows.flat_map(|r| cols.clone().map(move |c| p(r, c)))
        .collect()
}

/// Board A (5x5). Cities: 0 = P0 capital at (1,1), level 3, population 6, workshop; 1 = P1 city
/// at (3,3), level 3, population 5, one park. City 0 claims rows 0-2 x cols 0-2; city 1 claims
/// rows 2-4 x cols 2-4 except (2,2). Village at (0,4).
/// Units: 0 P0 Warrior (1,1); 1 P1 Rider (2,3) veteran 12/15 Moved kills 3; 2 P0 Archer (4,0)
/// 7/10 Attacked kills 1; 3 P1 Defender (0,4); 4 P1 Super Unit (3,3); 5 P0 Rider (0,0)
/// veteran 15/15 Done.
/// P0: Organization, 7 stars, explored rows 0-3 x cols 0-3. P1: Climbing + Hunting, 4 stars,
/// explored rows 1-4 x cols 1-4. Turn 3, P1 to move.
fn board_a() -> State {
    StateBuilder::new(5)
        .terrain(p(0, 1), Terrain::Forest)
        .resource(p(0, 1), Resource::Animal)
        .terrain(p(0, 2), Terrain::Forest)
        .improvement(p(0, 2), Improvement::LumberHut)
        .resource(p(1, 0), Resource::Fruit)
        .resource(p(1, 2), Resource::Crop)
        .resource(p(2, 0), Resource::Crop)
        .improvement(p(2, 0), Improvement::Farm)
        .terrain(p(2, 1), Terrain::Mountain)
        .resource(p(2, 1), Resource::Metal)
        .terrain(p(2, 2), Terrain::Mountain)
        .resource(p(2, 2), Resource::Metal)
        .improvement(p(2, 2), Improvement::Mine)
        .terrain(p(3, 0), Terrain::Forest)
        .terrain(p(3, 1), Terrain::Mountain)
        .resource(p(3, 1), Resource::Metal)
        .resource(p(4, 2), Resource::Crop)
        .terrain(p(4, 4), Terrain::Mountain)
        .village(p(0, 4))
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 6;
            c.workshop = true;
        })
        .city_with(1, p(3, 3), |c| {
            c.level = 3;
            c.population = 5;
            c.parks = 1;
        })
        .unit(0, UnitType::Warrior, p(1, 1))
        .unit_with(1, UnitType::Rider, p(2, 3), |u| {
            u.veteran = true;
            u.max_health = 15;
            u.health = 12;
            u.kills = 3;
            u.phase = Phase::Moved;
        })
        .unit_with(0, UnitType::Archer, p(4, 0), |u| {
            u.health = 7;
            u.kills = 1;
            u.phase = Phase::Attacked;
        })
        .unit(1, UnitType::Defender, p(0, 4))
        .unit(1, UnitType::SuperUnit, p(3, 3))
        .unit_with(0, UnitType::Rider, p(0, 0), |u| {
            u.veteran = true;
            u.max_health = 15;
            u.health = 15;
            u.phase = Phase::Done;
        })
        .techs(0, &[Tech::Organization])
        .techs(1, &[Tech::Climbing, Tech::Hunting])
        .stars(0, 7)
        .stars(1, 4)
        .explore(0, &tiles(0..=3, 0..=3))
        .explore(1, &tiles(1..=4, 1..=4))
        .turn(3)
        .current(1)
        .build()
}

// P0 income: level 3 + capital 1 + workshop 1 = 5.
// P0 score: units 10 + 15 + 15 = 40; territory 9 x 20 = 180; exploration 16 x 5 = 80;
//   city 100 + 50 x 2 + 5 x (6 - 5) = 205; parks 0; techs Organization 100. Total 605.
// P1 income: level 3 + park 1 = 4.
// P1 score: units 15 + 15 + 50 = 80; territory 8 x 20 = 160; exploration 16 x 5 = 80;
//   city 100 + 100 + 0 = 200; parks 250; techs 100 + 100 = 200. Total 970.
const A_P0: &str = "P0: stars 7 | income 5 | score 605 | cities 1 | units 3 | techs organization";
const A_P1: &str =
    "P1: stars 4 | income 4 | score 970 | cities 1 | units 3 | techs climbing, hunting";

#[test]
fn board_a_omniscient() {
    let expected = lines(&[
        "   0    1    2    3    4",
        " 0 .0.R a0.. H0.. .... ..Vd",
        " 1 o0.. .0KW +0.. .... ....",
        " 2 F0.. $0.. M0.. .1.r .1..",
        " 3 f... $... .1.. .1Cs .1..",
        " 4 ...A .... +1.. .1.. ^1..",
        "",
        "turn 3 | P1 to move",
        A_P0,
        A_P1,
    ]);
    assert_eq!(render(&board_a(), View::Omniscient), expected);
}

/// P0 lacks Climbing: metal shows as its mountain; the mine still shows (improvement first).
/// Column 4 and row 4 are fog. The opponent line is reduced to the score.
#[test]
fn board_a_player0_view() {
    let expected = lines(&[
        "   0    1    2    3    4",
        " 0 .0.R a0.. H0.. .... ????",
        " 1 o0.. .0KW +0.. .... ????",
        " 2 F0.. ^0.. M0.. .1.r ????",
        " 3 f... ^... .1.. .1Cs ????",
        " 4 ???? ???? ???? ???? ????",
        "",
        "turn 3 | P1 to move",
        A_P0,
        "P1: score 970",
    ]);
    assert_eq!(render(&board_a(), View::Player(0)), expected);
}

/// P1 lacks Organization: crops show as field; it has Climbing, so metal shows.
/// Row 0 and column 0 are fog.
#[test]
fn board_a_player1_view() {
    let expected = lines(&[
        "   0    1    2    3    4",
        " 0 ???? ???? ???? ???? ????",
        " 1 ???? .0KW .0.. .... ....",
        " 2 ???? $0.. M0.. .1.r .1..",
        " 3 ???? $... .1.. .1Cs .1..",
        " 4 ???? .... .1.. .1.. ^1..",
        "",
        "turn 3 | P1 to move",
        "P0: score 605",
        A_P1,
    ]);
    assert_eq!(render(&board_a(), View::Player(1)), expected);
}

/// The view follows live state on explored tiles and the explored mask decides fog, also when
/// `full_observability` was used to build the state.
#[test]
fn full_observability_shows_every_tile_in_a_player_view() {
    let s = StateBuilder::new(5)
        .config(Config {
            full_observability: true,
            ..Config::default()
        })
        .resource(p(0, 0), Resource::Crop)
        .terrain(p(4, 4), Terrain::Mountain)
        .resource(p(4, 4), Resource::Metal)
        .unit(1, UnitType::Warrior, p(2, 2))
        .build();
    // Player 0 has no techs: crop and metal hidden. 25 tiles explored = 125 points each.
    // P1 score: Warrior 10 + 125 = 135.
    let expected = lines(&[
        "   0    1    2    3    4",
        " 0 .... .... .... .... ....",
        " 1 .... .... .... .... ....",
        " 2 .... .... ...w .... ....",
        " 3 .... .... .... .... ....",
        " 4 .... .... .... .... ^...",
        "",
        "turn 0 | P0 to move",
        "P0: stars 0 | income 0 | score 125 | cities 0 | units 0 | techs -",
        "P1: score 135",
    ]);
    assert_eq!(render(&s, View::Player(0)), expected);
}

/// Board B (5x5): only a P0 level 1 capital at (2,2), nothing explored, turn 50, P0 to move.
/// P0 techs given out of order to check the `ALL_TECHS` ordering.
fn board_b(outcome: Option<Outcome>) -> State {
    let mut s = StateBuilder::new(5)
        .city_with(0, p(2, 2), |c| c.is_capital = true)
        .techs(
            0,
            &[
                Tech::Mining,
                Tech::Farming,
                Tech::Organization,
                Tech::Climbing,
            ],
        )
        .stars(0, 9)
        .turn(50)
        .build();
    s.outcome = outcome;
    s
}

// P0 income: level 1 + capital 1 = 2. P0 score: territory 180 + city 100 + techs
// (1 + 1 + 2 + 2) x 100 = 600 -> 880. P1 score 0.
const B_BOARD: [&str; 6] = [
    "   0    1    2    3    4",
    " 0 .... .... .... .... ....",
    " 1 .... .0.. .0.. .0.. ....",
    " 2 .... .0.. .0K. .0.. ....",
    " 3 .... .0.. .0.. .0.. ....",
    " 4 .... .... .... .... ....",
];
const B_P0: &str =
    "P0: stars 9 | income 2 | score 880 | cities 1 | units 0 | techs climbing, organization, farming, mining";
const B_P1: &str = "P1: stars 0 | income 0 | score 0 | cities 0 | units 0 | techs -";

fn board_b_expected(first_panel_line: &str) -> String {
    let mut ls: Vec<&str> = B_BOARD.to_vec();
    ls.extend(["", first_panel_line, B_P0, B_P1]);
    lines(&ls)
}

#[test]
fn panel_lines_for_each_outcome() {
    assert_eq!(
        render(&board_b(None), View::Omniscient),
        board_b_expected("turn 50 | P0 to move")
    );
    assert_eq!(
        render(&board_b(Some(Outcome::Win(1))), View::Omniscient),
        board_b_expected("turn 50 | P0 to move | game over: P1 wins")
    );
    assert_eq!(
        render(&board_b(Some(Outcome::Win(0))), View::Omniscient),
        board_b_expected("turn 50 | P0 to move | game over: P0 wins")
    );
    assert_eq!(
        render(&board_b(Some(Outcome::Draw)), View::Omniscient),
        board_b_expected("turn 50 | P0 to move | game over: draw")
    );
}

/// Board C (7x7). City 0 = P1 capital at (1,5) (rows 0-2 x cols 4-6); city 1 = P0 level 2 city at
/// (5,1), population 3, workshop (rows 4-6 x cols 0-2). Village at (3,3).
/// Units: 0 P1 Warrior (1,5); 1 P1 Archer (3,3); 2 P0 Defender (5,1); 3 P0 Super Unit (4,4).
/// Both players explored everything, then (0,6) and (6,6) hidden for P0.
/// P0: Organization + Strategy, 3 stars. P1: no techs, 5 stars. Turn 0, P0 to move.
fn board_c() -> State {
    StateBuilder::new(7)
        .terrain(p(0, 0), Terrain::Mountain)
        .terrain(p(0, 1), Terrain::Forest)
        .resource(p(0, 1), Resource::Animal)
        .improvement(p(0, 1), Improvement::LumberHut)
        .resource(p(3, 0), Resource::Fruit)
        .terrain(p(6, 6), Terrain::Forest)
        .terrain(p(2, 4), Terrain::Mountain)
        .resource(p(2, 4), Resource::Metal)
        .resource(p(4, 2), Resource::Crop)
        .improvement(p(4, 2), Improvement::Farm)
        .village(p(3, 3))
        .city_with(1, p(1, 5), |c| c.is_capital = true)
        .city_with(0, p(5, 1), |c| {
            c.level = 2;
            c.population = 3;
            c.workshop = true;
        })
        .unit(1, UnitType::Warrior, p(1, 5))
        .unit(1, UnitType::Archer, p(3, 3))
        .unit(0, UnitType::Defender, p(5, 1))
        .unit(0, UnitType::SuperUnit, p(4, 4))
        .techs(0, &[Tech::Strategy, Tech::Organization])
        .stars(0, 3)
        .stars(1, 5)
        .explore_all()
        .hide(0, &[p(0, 6), p(6, 6)])
        .build()
}

// P0 income: level 2 + workshop 1 = 3. P0 score: units 15 + 50 = 65; territory 180;
//   exploration 47 x 5 = 235; city 100 + 50 + 5 x (3 - 2) = 155; techs 100 + 200 = 300 -> 935.
// P1 income: level 1 + capital 1 = 2. P1 score: units 10 + 15 = 25; territory 180;
//   exploration 49 x 5 = 245; city 100; techs 0 -> 550.
const C_P0: &str =
    "P0: stars 3 | income 3 | score 935 | cities 1 | units 2 | techs organization, strategy";

#[test]
fn board_c_size_7_omniscient() {
    let expected = lines(&[
        "   0    1    2    3    4    5    6",
        " 0 ^... H... .... .... .1.. .1.. .1..",
        " 1 .... .... .... .... .1.. .1Kw .1..",
        " 2 .... .... .... .... $1.. .1.. .1..",
        " 3 o... .... .... ..Va .... .... ....",
        " 4 .0.. .0.. F0.. .... ...S .... ....",
        " 5 .0.. .0CD .0.. .... .... .... ....",
        " 6 .0.. .0.. .0.. .... .... .... f...",
        "",
        "turn 0 | P0 to move",
        C_P0,
        "P1: stars 5 | income 2 | score 550 | cities 1 | units 2 | techs -",
    ]);
    assert_eq!(render(&board_c(), View::Omniscient), expected);
}

#[test]
fn board_c_size_7_player0_view() {
    let expected = lines(&[
        "   0    1    2    3    4    5    6",
        " 0 ^... H... .... .... .1.. .1.. ????",
        " 1 .... .... .... .... .1.. .1Kw .1..",
        " 2 .... .... .... .... ^1.. .1.. .1..",
        " 3 o... .... .... ..Va .... .... ....",
        " 4 .0.. .0.. F0.. .... ...S .... ....",
        " 5 .0.. .0CD .0.. .... .... .... ....",
        " 6 .0.. .0.. .0.. .... .... .... ????",
        "",
        "turn 0 | P0 to move",
        C_P0,
        "P1: score 550",
    ]);
    assert_eq!(render(&board_c(), View::Player(0)), expected);
}

/// Two-digit column and row numbers on an 11x11 board.
#[test]
fn header_and_row_labels_on_an_11_board() {
    let s = StateBuilder::new(11).build();
    let row = ".... .... .... .... .... .... .... .... .... .... ....";
    let labels = [
        " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10",
    ];
    let mut ls = vec!["   0    1    2    3    4    5    6    7    8    9    10".to_string()];
    for l in labels {
        ls.push(format!("{l} {row}"));
    }
    ls.push(String::new());
    ls.push("turn 0 | P0 to move".into());
    ls.push("P0: stars 0 | income 0 | score 0 | cities 0 | units 0 | techs -".into());
    ls.push("P1: stars 0 | income 0 | score 0 | cities 0 | units 0 | techs -".into());
    let refs: Vec<&str> = ls.iter().map(|s| s.as_str()).collect();
    assert_eq!(render(&s, View::Omniscient), lines(&refs));
}

#[test]
fn no_line_has_trailing_whitespace_and_output_ends_with_one_newline() {
    for (s, views) in [
        (
            board_a(),
            [View::Omniscient, View::Player(0), View::Player(1)],
        ),
        (
            board_c(),
            [View::Omniscient, View::Player(0), View::Player(1)],
        ),
    ] {
        for v in views {
            let out = render(&s, v);
            assert!(out.ends_with('\n') && !out.ends_with("\n\n"), "{v:?}");
            for l in out.lines() {
                assert_eq!(l, l.trim_end(), "{v:?}: trailing whitespace in {l:?}");
            }
        }
    }
}

// --- describe_tile -------------------------------------------------------------------------------

#[test]
fn describe_tile_omniscient() {
    let s = board_a();
    let d = |r, c| describe_tile(&s, View::Omniscient, p(r, c));
    assert_eq!(
        d(1, 1),
        "(1,1) field res=- imp=- territory=P0 site=capital#0(P0) unit=#0 P0 warrior hp 10/10 Fresh"
    );
    assert_eq!(
        d(3, 3),
        "(3,3) field res=- imp=- territory=P1 site=city#1(P1) unit=#4 P1 super_unit hp 40/40 Fresh"
    );
    assert_eq!(
        d(0, 4),
        "(0,4) field res=- imp=- territory=- site=village unit=#3 P1 defender hp 15/15 Fresh"
    );
    assert_eq!(
        d(2, 3),
        "(2,3) field res=- imp=- territory=P1 site=- unit=#1 P1 rider hp 12/15 Moved veteran kills 3"
    );
    assert_eq!(
        d(4, 0),
        "(4,0) field res=- imp=- territory=- site=- unit=#2 P0 archer hp 7/10 Attacked kills 1"
    );
    assert_eq!(
        d(0, 0),
        "(0,0) field res=- imp=- territory=P0 site=- unit=#5 P0 rider hp 15/15 Done veteran"
    );
    assert_eq!(
        d(0, 1),
        "(0,1) forest res=animal imp=- territory=P0 site=- unit=-"
    );
    assert_eq!(
        d(0, 2),
        "(0,2) forest res=- imp=lumber_hut territory=P0 site=- unit=-"
    );
    assert_eq!(
        d(1, 0),
        "(1,0) field res=fruit imp=- territory=P0 site=- unit=-"
    );
    assert_eq!(
        d(1, 2),
        "(1,2) field res=crop imp=- territory=P0 site=- unit=-"
    );
    assert_eq!(
        d(2, 0),
        "(2,0) field res=crop imp=farm territory=P0 site=- unit=-"
    );
    assert_eq!(
        d(2, 1),
        "(2,1) mountain res=metal imp=- territory=P0 site=- unit=-"
    );
    assert_eq!(
        d(2, 2),
        "(2,2) mountain res=metal imp=mine territory=P0 site=- unit=-"
    );
    assert_eq!(
        d(4, 4),
        "(4,4) mountain res=- imp=- territory=P1 site=- unit=-"
    );
    assert_eq!(d(1, 4), "(1,4) field res=- imp=- territory=- site=- unit=-");
}

#[test]
fn describe_tile_player_views() {
    let s = board_a();
    let d0 = |r, c| describe_tile(&s, View::Player(0), p(r, c));
    let d1 = |r, c| describe_tile(&s, View::Player(1), p(r, c));
    // Fog.
    assert_eq!(d0(0, 4), "(0,4) unexplored");
    assert_eq!(d0(4, 4), "(4,4) unexplored");
    assert_eq!(d1(0, 0), "(0,0) unexplored");
    assert_eq!(d1(2, 0), "(2,0) unexplored");
    // P0 lacks Climbing: metal hidden, the mine is still shown.
    assert_eq!(
        d0(2, 1),
        "(2,1) mountain res=- imp=- territory=P0 site=- unit=-"
    );
    assert_eq!(
        d0(2, 2),
        "(2,2) mountain res=- imp=mine territory=P0 site=- unit=-"
    );
    // P0 has Organization: crop shown.
    assert_eq!(
        d0(1, 2),
        "(1,2) field res=crop imp=- territory=P0 site=- unit=-"
    );
    assert_eq!(
        d0(2, 0),
        "(2,0) field res=crop imp=farm territory=P0 site=- unit=-"
    );
    // P1 lacks Organization: crop hidden; has Climbing: metal shown.
    assert_eq!(
        d1(1, 2),
        "(1,2) field res=- imp=- territory=P0 site=- unit=-"
    );
    assert_eq!(
        d1(4, 2),
        "(4,2) field res=- imp=- territory=P1 site=- unit=-"
    );
    assert_eq!(
        d1(2, 2),
        "(2,2) mountain res=metal imp=mine territory=P0 site=- unit=-"
    );
    // Live enemy units and cities on explored tiles.
    assert_eq!(
        d0(3, 3),
        "(3,3) field res=- imp=- territory=P1 site=city#1(P1) unit=#4 P1 super_unit hp 40/40 Fresh"
    );
    assert_eq!(
        d1(1, 1),
        "(1,1) field res=- imp=- territory=P0 site=capital#0(P0) unit=#0 P0 warrior hp 10/10 Fresh"
    );
    assert_eq!(
        d0(2, 3),
        "(2,3) field res=- imp=- territory=P1 site=- unit=#1 P1 rider hp 12/15 Moved veteran kills 3"
    );
}

#[test]
fn describe_tile_on_board_c() {
    let s = board_c();
    let d = |v, r, c| describe_tile(&s, v, p(r, c));
    assert_eq!(
        d(View::Omniscient, 1, 5),
        "(1,5) field res=- imp=- territory=P1 site=capital#0(P1) unit=#0 P1 warrior hp 10/10 Fresh"
    );
    assert_eq!(
        d(View::Player(0), 5, 1),
        "(5,1) field res=- imp=- territory=P0 site=city#1(P0) unit=#2 P0 defender hp 15/15 Fresh"
    );
    assert_eq!(
        d(View::Omniscient, 0, 1),
        "(0,1) forest res=animal imp=lumber_hut territory=- site=- unit=-"
    );
    assert_eq!(
        d(View::Player(1), 3, 3),
        "(3,3) field res=- imp=- territory=- site=village unit=#1 P1 archer hp 10/10 Fresh"
    );
    assert_eq!(
        d(View::Player(1), 4, 2),
        "(4,2) field res=- imp=farm territory=P0 site=- unit=-"
    );
    assert_eq!(d(View::Player(0), 6, 6), "(6,6) unexplored");
    assert_eq!(
        d(View::Player(1), 6, 6),
        "(6,6) forest res=- imp=- territory=- site=- unit=-"
    );
}

// --- action_label --------------------------------------------------------------------------------

#[test]
fn action_label_for_every_variant() {
    let cases: Vec<(Action, &str)> = vec![
        (
            Action::Move {
                unit: UnitId(3),
                to: p(2, 10),
            },
            "Move u3 -> (2,10)",
        ),
        (
            Action::Attack {
                unit: UnitId(12),
                target: p(0, 1),
            },
            "Attack u12 -> (0,1)",
        ),
        (Action::Capture { unit: UnitId(0) }, "Capture u0"),
        (Action::Recover { unit: UnitId(5) }, "Recover u5"),
        (Action::Promote { unit: UnitId(7) }, "Promote u7"),
        (
            Action::Train {
                city: CityId(1),
                kind: UnitType::Warrior,
            },
            "Train c1 warrior",
        ),
        (
            Action::Train {
                city: CityId(0),
                kind: UnitType::Defender,
            },
            "Train c0 defender",
        ),
        (
            Action::Site {
                at: p(4, 3),
                action: SiteAction::HarvestFruit,
            },
            "Site (4,3) harvest_fruit",
        ),
        (
            Action::Site {
                at: p(0, 0),
                action: SiteAction::LumberHut,
            },
            "Site (0,0) lumber_hut",
        ),
        (
            Action::Site {
                at: p(1, 9),
                action: SiteAction::ClearForest,
            },
            "Site (1,9) clear_forest",
        ),
        (
            Action::Research {
                tech: Tech::Organization,
            },
            "Research organization",
        ),
        (
            Action::Research {
                tech: Tech::Climbing,
            },
            "Research climbing",
        ),
        (
            Action::ChooseReward {
                city: CityId(2),
                reward: Reward::SuperUnit,
            },
            "Reward c2 super_unit",
        ),
        (
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::PopulationGrowth,
            },
            "Reward c0 population_growth",
        ),
        (
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::CityWall,
            },
            "Reward c0 city_wall",
        ),
        (Action::EndTurn, "End turn"),
    ];
    for (a, label) in cases {
        assert_eq!(action_label(&a), label, "{a:?}");
    }
}
