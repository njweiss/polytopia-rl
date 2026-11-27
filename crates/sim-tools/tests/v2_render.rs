//! Milestone 19 tooling tests (docs/MILESTONE_19 "Renderer, HTML viewer, and CLI", U3-U5,
//! "Tests" item 5):
//! - Text `render`: v2 unit glyphs (`X`/`x`, `C`/`c`, `K`/`k`) and v2 techs in `panel_line`
//! - `describe_tile`: v2 optional tags in fixed order (`road`, `bldg=<name>(<level>)`,
//!   `monument=<name>`, `lighthouse`) under `V2` only; `V1` unchanged
//! - `html_page` and `frames_json`: v2 units in `KIND` + legend, optional v2 tile keys in
//!   `tile_json` under `V2`, omitted in `V1`
//! - `run_cli`: `--ruleset v1|v2` on `play-random`, `play`, and `tournament`, plus exit 2 on
//!   invalid values

mod common;

use bots::{run_match, BotKind};
use common::*;
use serde_json::Value;
use sim_core::actions::{legal_actions, Action};
use sim_core::data::{Building, Monument, SiteAction, Tech, UnitType};
use sim_core::map::{MapParams, Terrain};
use sim_core::state::{Config, Improvement, Ruleset, State};
use sim_core::testing::StateBuilder;
use sim_tools::html::{frames_json, html_page};
use sim_tools::render::{describe_tile, render, View};
use sim_tools::replay::{record_game, replay, ReplayLog};

const V1_FIXTURE: &str = include_str!("fixtures/v1_seed5_cap10.json");

fn v2_cfg(turn_cap: u32) -> Config {
    Config {
        turn_cap,
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn lines(ls: &[&str]) -> String {
    let mut s = ls.join("\n");
    s.push('\n');
    s
}

fn run(args: &[&str]) -> (i32, String, String) {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = sim_tools::cli::run_cli(&args, &mut out, &mut err);
    (
        code,
        String::from_utf8(out).expect("utf-8 out"),
        String::from_utf8(err).expect("utf-8 err"),
    )
}

fn read_log(path: &std::path::Path) -> ReplayLog {
    let text = std::fs::read_to_string(path).expect("log file written");
    serde_json::from_str(&text).expect("valid ReplayLog")
}

// ---------------------------------------------------------------- Text render

#[test]
fn render_shows_v2_unit_glyphs_and_v2_techs_in_declaration_order() {
    // 5x5 v2 board with P0 Swordsman (X), Catapult (C), Knight (K) and P1 x, c, k.
    // P0 techs given out of order to verify `techs(Ruleset::V2)` declaration order.
    let s = StateBuilder::new(5)
        .config(v2_cfg(30))
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(3, 3), |c| c.is_capital = true)
        .unit(0, UnitType::Swordsman, p(0, 0))
        .unit(0, UnitType::Catapult, p(0, 1))
        .unit(0, UnitType::Knight, p(0, 2))
        .unit(1, UnitType::Swordsman, p(4, 0))
        .unit(1, UnitType::Catapult, p(4, 1))
        .unit(1, UnitType::Knight, p(4, 2))
        .techs(
            0,
            &[
                Tech::Smithery,
                Tech::Chivalry,
                Tech::Roads,
                Tech::Organization,
                Tech::Mathematics,
                Tech::FreeSpirit,
                Tech::Trade,
                Tech::Construction,
            ],
        )
        .techs(1, &[Tech::Chivalry, Tech::FreeSpirit])
        .stars(0, 12)
        .stars(1, 7)
        .explore_all()
        .build();

    // P0 income: level 1 + capital 1 = 2.
    // P0 score: units 25+40+40=105, territory 9*20=180, exploration 25*5=125, city 100,
    //   techs (org 1 + roads 2 + free_spirit 2 + trade 3 + chivalry 3 + math 3 + constr 3 + smithery 3) * 100 = 2000.
    //   Total = 2510.
    // P1 income: 2.
    // P1 score: units 105, territory 8*20=160 ((2,2) belongs to city 0), exploration 125, city 100,
    //   techs (free_spirit 2 + chivalry 3) * 100 = 500. Total = 990.
    let expected_omni = lines(&[
        "   0    1    2    3    4",
        " 0 .0.X .0.C .0.K .... ....",
        " 1 .0.. .0K. .0.. .... ....",
        " 2 .0.. .0.. .0.. .1.. .1..",
        " 3 .... .... .1.. .1K. .1..",
        " 4 ...x ...c .1.k .1.. .1..",
        "",
        "turn 0 | P0 to move",
        "P0: stars 12 | income 2 | score 2510 | cities 1 | units 3 | techs organization, roads, free_spirit, trade, chivalry, mathematics, construction, smithery",
        "P1: stars 7 | income 2 | score 990 | cities 1 | units 3 | techs free_spirit, chivalry",
    ]);
    assert_eq!(render(&s, View::Omniscient), expected_omni);

    // Player 0 view hides P1's v2 techs and reduces P1's panel line to score only.
    let out_p0 = render(&s, View::Player(0));
    assert!(out_p0.contains("P1: score 990\n"), "{out_p0}");
    assert!(!out_p0.contains("P1: stars"), "{out_p0}");
}

// ---------------------------------------------------------------- describe_tile

#[test]
fn describe_tile_appends_v2_tags_in_fixed_order_under_v2_only() {
    let mut s = StateBuilder::new(5)
        .config(v2_cfg(30))
        .lighthouse(p(0, 0))
        .terrain(p(0, 1), Terrain::Forest)
        .improvement(p(0, 1), Improvement::LumberHut)
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .building(p(0, 2), Building::Sawmill) // adjacent to LumberHut (0,1) -> level 1
        .monument(p(2, 0), Monument::ParkOfFortune)
        .unit(0, UnitType::Knight, p(1, 2))
        .unit(1, UnitType::Catapult, p(2, 2))
        .unit(0, UnitType::Swordsman, p(2, 1))
        .explore_all()
        .hide(0, &[p(0, 0), p(2, 0)])
        .build();
    // Put roads on (0,2) (with Sawmill), (2,0) (with Monument), and (1,2) (with Knight).
    for q in [p(0, 2), p(2, 0), p(1, 2)] {
        let i = q.row as usize * 5 + q.col as usize;
        s.tiles[i].road = true;
    }

    let d = |r, c| describe_tile(&s, View::Omniscient, p(r, c));
    // Lighthouse only.
    assert_eq!(
        d(0, 0),
        "(0,0) field res=- imp=- territory=P0 site=- unit=- lighthouse"
    );
    // Road + building with level.
    assert_eq!(
        d(0, 2),
        "(0,2) field res=- imp=- territory=P0 site=- unit=- road bldg=sawmill(1)"
    );
    // Road + monument.
    assert_eq!(
        d(2, 0),
        "(2,0) field res=- imp=- territory=P0 site=- unit=- road monument=park_of_fortune"
    );
    // Unit + road.
    assert_eq!(
        d(1, 2),
        "(1,2) field res=- imp=- territory=P0 site=- unit=#0 P0 knight hp 10/10 Fresh road"
    );
    // Plain v2 tile with no v2 features: NO trailing whitespace or extra tags.
    assert_eq!(d(4, 4), "(4,4) field res=- imp=- territory=- site=- unit=-");
    // v2 unit names.
    assert_eq!(
        d(2, 2),
        "(2,2) field res=- imp=- territory=P0 site=- unit=#1 P1 catapult hp 10/10 Fresh"
    );
    assert_eq!(
        d(2, 1),
        "(2,1) field res=- imp=- territory=P0 site=- unit=#2 P0 swordsman hp 15/15 Fresh"
    );
    // Fogged tiles in player view must still report unexplored (no v2 leak).
    assert_eq!(
        describe_tile(&s, View::Player(0), p(0, 0)),
        "(0,0) unexplored"
    );
    assert_eq!(
        describe_tile(&s, View::Player(0), p(2, 0)),
        "(2,0) unexplored"
    );

    // Verify all 4 tags together if present on one tile: road, bldg, monument, lighthouse.
    let mut all_tags = s.clone();
    let i44 = 4 * 5 + 4;
    all_tags.tiles[i44].road = true;
    all_tags.tiles[i44].building = Some(Building::Market);
    all_tags.tiles[i44].building_level = 3;
    all_tags.tiles[i44].monument = Some(Monument::EyeOfGod);
    all_tags.tiles[i44].lighthouse = true;
    assert_eq!(
        describe_tile(&all_tags, View::Omniscient, p(4, 4)),
        "(4,4) field res=- imp=- territory=- site=- unit=- road bldg=market(3) monument=eye_of_god lighthouse"
    );

    // Under V1, describe_tile omits v2 tags even on a plain tile (exact v1 format).
    let v1 = StateBuilder::new(5).explore_all().build();
    assert_eq!(
        describe_tile(&v1, View::Omniscient, p(0, 0)),
        "(0,0) field res=- imp=- territory=- site=- unit=-"
    );
}

// ---------------------------------------------------------------- HTML & frames_json

/// Records a real, verifiable v2 replay where P0 builds a road, a sawmill (with level >= 1),
/// a Park of Fortune monument, and trains Swordsman, Catapult, and Knight while P1 ends turns.
fn record_rich_v2_replay() -> ReplayLog {
    // Pick a seed where P0's capital ring has at least one forest and one empty field next to it
    // (almost every seed qualifies; scan 0..20 deterministically).
    let target_techs = [
        Tech::Hunting,
        Tech::Forestry,
        Tech::Mathematics,
        Tech::Climbing,
        Tech::Mining,
        Tech::Smithery,
        Tech::Riding,
        Tech::Roads,
        Tech::FreeSpirit,
        Tech::Chivalry,
        Tech::Farming,
        Tech::Construction,
    ];
    for seed in 0..20u64 {
        let mut trained = [false; 3];
        let mut built_road = false;
        let mut built_bldg = false;
        let mut built_mon = false;
        let log = record_game(
            seed,
            &MapParams::default(),
            &v2_cfg(60),
            2000,
            &mut |s: &State| {
                let legal = legal_actions(s);
                if s.current == 1 {
                    return if legal.contains(&Action::EndTurn) {
                        Action::EndTurn
                    } else {
                        legal[0]
                    };
                }
                // Pending reward: pick PopulationGrowth / Resources / Workshop / Park to reach lvl 5.
                if let Some(a) = legal.iter().copied().find(|a| {
                    matches!(
                        a,
                        Action::ChooseReward {
                            reward: sim_core::data::Reward::PopulationGrowth
                                | sim_core::data::Reward::Workshop
                                | sim_core::data::Reward::Resources
                                | sim_core::data::Reward::Park,
                            ..
                        }
                    )
                }) {
                    return a;
                }
                if let Some(a) = legal
                    .iter()
                    .copied()
                    .find(|a| matches!(a, Action::ChooseReward { .. }))
                {
                    return a;
                }
                // Move units off capital so we can train!
                let cap_pos = s
                    .cities
                    .iter()
                    .find(|c| c.owner == 0 && c.is_capital)
                    .map(|c| c.pos);
                if let Some(cp) = cap_pos {
                    if let Some(uid) = s.tile(cp).unit {
                        if let Some(a) = legal
                            .iter()
                            .copied()
                            .find(|a| matches!(a, Action::Move { unit, .. } if *unit == uid))
                        {
                            return a;
                        }
                    }
                }
                if !built_mon {
                    if let Some(a) = legal
                        .iter()
                        .copied()
                        .find(|a| matches!(a, Action::BuildMonument { .. }))
                    {
                        built_mon = true;
                        return a;
                    }
                }
                if !built_road {
                    if let Some(a) = legal
                        .iter()
                        .copied()
                        .find(|a| matches!(a, Action::BuildRoad { .. }))
                    {
                        built_road = true;
                        return a;
                    }
                }
                if !built_bldg {
                    if let Some(a) = legal
                        .iter()
                        .copied()
                        .find(|a| matches!(a, Action::BuildBuilding { .. }))
                    {
                        built_bldg = true;
                        return a;
                    }
                }
                // Train each of the 3 v2 units at least once.
                for (idx, kind) in [UnitType::Swordsman, UnitType::Catapult, UnitType::Knight]
                    .into_iter()
                    .enumerate()
                {
                    if !trained[idx] {
                        if let Some(a) = legal
                            .iter()
                            .copied()
                            .find(|a| matches!(a, Action::Train { kind: k, .. } if *k == kind))
                        {
                            trained[idx] = true;
                            return a;
                        }
                    }
                }
                // Grow capital population to unlock ParkOfFortune (level 5) and enable buildings.
                for sa in [
                    SiteAction::Farm,
                    SiteAction::Mine,
                    SiteAction::LumberHut,
                    SiteAction::HarvestFruit,
                    SiteAction::HarvestAnimal,
                ] {
                    if let Some(a) = legal
                        .iter()
                        .copied()
                        .find(|a| matches!(a, Action::Site { action, .. } if *action == sa))
                    {
                        return a;
                    }
                }
                // Research target techs.
                for tech in target_techs {
                    let a = Action::Research { tech };
                    if legal.contains(&a) {
                        return a;
                    }
                }
                if built_bldg {
                    if let Some(a) = legal.iter().copied().find(|a| {
                        matches!(
                            a,
                            Action::Site {
                                action: SiteAction::BurnForest,
                                ..
                            }
                        )
                    }) {
                        return a;
                    }
                }
                // Capture villages if any unit stands on one, or move to explore lighthouses.
                if let Some(a) = legal
                    .iter()
                    .copied()
                    .find(|a| matches!(a, Action::Capture { .. }))
                {
                    return a;
                }
                if built_mon && built_road && built_bldg && trained.iter().all(|&x| x) {
                    return Action::EndTurn;
                }
                if let Some(a) = legal
                    .iter()
                    .copied()
                    .find(|a| matches!(a, Action::Move { .. }))
                {
                    return a;
                }
                Action::EndTurn
            },
        );
        if built_mon && built_road && built_bldg && trained.iter().all(|&x| x) {
            return log;
        }
    }
    panic!("failed to record a rich v2 replay with all v2 features in seeds 0..20");
}

#[test]
fn html_and_frames_json_support_v2_units_and_v2_tile_fields() {
    let log = record_rich_v2_replay();
    assert!(replay(&log).is_ok());

    let frames_str = frames_json(&log).unwrap();
    let page = html_page(&log).unwrap();
    assert!(page.contains(&format!("const REPLAY = {frames_str};")));
    assert!(!page.contains("http://") && !page.contains("https://"));

    // Verify KIND map in inline JS maps swordsman -> X, catapult -> C, knight -> K.
    let kind_line = page
        .lines()
        .find(|l| l.contains("const KIND"))
        .expect("KIND map in html_page");
    assert!(
        kind_line.contains("swordsman") && kind_line.contains("\"X\""),
        "{kind_line}"
    );
    assert!(
        kind_line.contains("catapult") && kind_line.contains("\"C\""),
        "{kind_line}"
    );
    assert!(
        kind_line.contains("knight") && kind_line.contains("\"K\""),
        "{kind_line}"
    );
    // Legend should also mention the v2 unit letters X, C, K.
    let legend_start = page.find("id=\"legend\"").expect("legend in html_page");
    let legend_end = page[legend_start..]
        .find("</p>")
        .expect("legend closing tag");
    let legend = &page[legend_start..legend_start + legend_end];
    for glyph in ["X", "C", "K"] {
        assert!(
            legend.contains(glyph),
            "legend missing v2 glyph {glyph}: {legend}"
        );
    }

    // Check every frame's tiles and units against independently re-simulated states.
    let states = resimulate(&log);
    let data: Value = serde_json::from_str(&frames_str).unwrap();
    let frames = data["frames"].as_array().unwrap();
    assert_eq!(frames.len(), states.len());

    let mut saw_road = false;
    let mut saw_building = false;
    let mut saw_monument = false;
    let mut saw_lighthouse = false;
    let mut saw_swordsman = false;
    let mut saw_catapult = false;
    let mut saw_knight = false;

    for (fi, (frame, s)) in frames.iter().zip(&states).enumerate() {
        let tiles = frame["tiles"].as_array().unwrap();
        assert_eq!(tiles.len(), s.tiles.len());
        for (ti, (tj, ts)) in tiles.iter().zip(&s.tiles).enumerate() {
            let obj = tj.as_object().unwrap();
            if ts.road {
                assert_eq!(obj.get("road"), Some(&Value::Bool(true)), "f{fi} t{ti}");
                saw_road = true;
            } else {
                assert!(!obj.contains_key("road"), "f{fi} t{ti}: omit false road");
            }
            if let Some(b) = ts.building {
                let bname = serde_json::to_value(b).unwrap();
                assert_eq!(obj.get("building"), Some(&bname), "f{fi} t{ti}");
                assert_eq!(
                    obj.get("building_level"),
                    Some(&Value::from(ts.building_level)),
                    "f{fi} t{ti}"
                );
                saw_building = true;
            } else {
                assert!(!obj.contains_key("building"), "f{fi} t{ti}");
                assert!(!obj.contains_key("building_level"), "f{fi} t{ti}");
            }
            if let Some(m) = ts.monument {
                let mname = serde_json::to_value(m).unwrap();
                assert_eq!(obj.get("monument"), Some(&mname), "f{fi} t{ti}");
                saw_monument = true;
            } else {
                assert!(!obj.contains_key("monument"), "f{fi} t{ti}");
            }
            if ts.lighthouse {
                assert_eq!(
                    obj.get("lighthouse"),
                    Some(&Value::Bool(true)),
                    "f{fi} t{ti}"
                );
                saw_lighthouse = true;
            } else {
                assert!(!obj.contains_key("lighthouse"), "f{fi} t{ti}");
            }
        }
        for u in frame["units"].as_array().unwrap() {
            match u["kind"].as_str() {
                Some("swordsman") => saw_swordsman = true,
                Some("catapult") => saw_catapult = true,
                Some("knight") => saw_knight = true,
                _ => {}
            }
        }
    }

    assert!(saw_road, "vacuity: replay must include road");
    assert!(saw_building, "vacuity: replay must include building");
    assert!(saw_monument, "vacuity: replay must include monument");
    assert!(saw_lighthouse, "vacuity: replay must include lighthouse");
    assert!(saw_swordsman, "vacuity: replay must include swordsman");
    assert!(saw_catapult, "vacuity: replay must include catapult");
    assert!(saw_knight, "vacuity: replay must include knight");
}

#[test]
fn v1_frames_json_omits_all_v2_tile_keys() {
    let log = ReplayLog::from_json(V1_FIXTURE).unwrap();
    let v: Value = serde_json::from_str(&frames_json(&log).unwrap()).unwrap();
    for frame in v["frames"].as_array().unwrap() {
        for tile in frame["tiles"].as_array().unwrap() {
            let obj = tile.as_object().unwrap();
            for key in [
                "road",
                "building",
                "building_level",
                "monument",
                "lighthouse",
            ] {
                assert!(
                    !obj.contains_key(key),
                    "v1 tile_json must omit {key}: {obj:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------- CLI --ruleset

#[test]
fn cli_play_random_supports_ruleset_v1_and_v2() {
    let f_v2 = TempFile::new("cli_m19_rand_v2.json");
    let (code, out, err) = run(&[
        "play-random",
        "--seed",
        "3",
        "--out",
        &f_v2.arg(),
        "--turn-cap",
        "3",
        "--ruleset",
        "v2",
    ]);
    assert_eq!(code, 0, "err: {err}");
    assert!(err.is_empty());
    assert!(out.starts_with("wrote "), "{out}");
    let log_v2 = read_log(&f_v2.0);
    assert_eq!(log_v2.config.ruleset, Ruleset::V2);
    let (vc, _, verr) = run(&["verify", &f_v2.arg()]);
    assert_eq!(vc, 0, "verify v2 failed: {verr}");
    let (view_c, _, view_err) = run(&["view", &f_v2.arg()]);
    assert_eq!(view_c, 0, "view v2 failed: {view_err}");
    let html_out = TempFile::new("cli_m19_rand_v2.html");
    let (hc, _, herr) = run(&["html", &f_v2.arg(), &html_out.arg()]);
    assert_eq!(hc, 0, "html v2 failed: {herr}");

    // Explicit --ruleset v1 matches default.
    let f_v1 = TempFile::new("cli_m19_rand_v1.json");
    let f_def = TempFile::new("cli_m19_rand_def.json");
    let (c1, _, e1) = run(&[
        "play-random",
        "--seed",
        "3",
        "--out",
        &f_v1.arg(),
        "--turn-cap",
        "3",
        "--ruleset",
        "v1",
    ]);
    let (c2, _, e2) = run(&[
        "play-random",
        "--seed",
        "3",
        "--out",
        &f_def.arg(),
        "--turn-cap",
        "3",
    ]);
    assert_eq!(c1, 0, "{e1}");
    assert_eq!(c2, 0, "{e2}");
    assert_eq!(
        std::fs::read_to_string(&f_v1.0).unwrap(),
        std::fs::read_to_string(&f_def.0).unwrap()
    );
}

#[test]
fn cli_play_supports_ruleset_v2() {
    let f = TempFile::new("cli_m19_play_v2.json");
    let (code, out, err) = run(&[
        "play",
        "--bots",
        "rush,economy",
        "--seed",
        "4",
        "--out",
        &f.arg(),
        "--turn-cap",
        "6",
        "--ruleset",
        "v2",
    ]);
    assert_eq!(code, 0, "err: {err}");
    assert!(err.is_empty());
    assert!(out.starts_with("wrote "), "{out}");
    let log = read_log(&f.0);
    assert_eq!(log.config.ruleset, Ruleset::V2);
    let (vc, _, verr) = run(&["verify", &f.arg()]);
    assert_eq!(vc, 0, "verify failed: {verr}");

    let f_v1 = TempFile::new("cli_m19_play_v1.json");
    let (c1, _, e1) = run(&[
        "play",
        "--bots",
        "rush,economy",
        "--seed",
        "4",
        "--out",
        &f_v1.arg(),
        "--turn-cap",
        "6",
        "--ruleset",
        "v1",
    ]);
    assert_eq!(c1, 0, "{e1}");
    assert_eq!(read_log(&f_v1.0).config.ruleset, Ruleset::V1);
}

#[test]
fn cli_tournament_supports_ruleset_v2() {
    let dir_file = TempFile::new("cli_m19_tourn_v2");
    let dir_path = dir_file.0.clone();
    drop(dir_file);
    let _ = std::fs::remove_dir_all(&dir_path);
    std::fs::create_dir_all(&dir_path).unwrap();

    let (code, out, err) = run(&[
        "tournament",
        "--bots",
        "rush,random",
        "--games",
        "1",
        "--turn-cap",
        "5",
        "--seed",
        "2",
        "--ruleset",
        "v2",
        "--save-replays",
        dir_path.to_str().unwrap(),
        "--save-count",
        "1",
    ]);
    assert_eq!(code, 0, "err: {err}");
    assert!(err.is_empty());
    let m = run_match(
        BotKind::Rush,
        BotKind::Random,
        2..3,
        &MapParams::default(),
        &v2_cfg(5),
    );
    let expected_line = format!(
        "rush vs random: rush {} win, random {} win, {} draw | by capture {}/{} | mean score diff {:+.1}",
        m.a_wins, m.b_wins, m.draws, m.a_capture_wins, m.b_capture_wins, m.mean_score_diff
    );
    assert!(out.contains(&expected_line), "out:\n{out}");

    let saved = dir_path.join("rush-vs-random-0.json");
    let log = read_log(&saved);
    assert_eq!(log.config.ruleset, Ruleset::V2);
    assert!(replay(&log).is_ok());

    let _ = std::fs::remove_dir_all(&dir_path);
}

#[test]
fn cli_invalid_ruleset_exits_2_with_usage_error() {
    let missing = TempFile::new("cli_m19_never_written.json");
    let out_arg = missing.arg();
    let cases: &[&[&str]] = &[
        &[
            "play-random",
            "--seed",
            "1",
            "--out",
            &out_arg,
            "--ruleset",
            "v3",
        ],
        &["play-random", "--seed", "1", "--out", &out_arg, "--ruleset"],
        &[
            "play",
            "--bots",
            "rush,economy",
            "--seed",
            "1",
            "--out",
            &out_arg,
            "--ruleset",
            "bogus",
        ],
        &[
            "play",
            "--bots",
            "rush,economy",
            "--seed",
            "1",
            "--out",
            &out_arg,
            "--ruleset",
        ],
        &[
            "tournament",
            "--bots",
            "rush,economy",
            "--games",
            "1",
            "--ruleset",
            "v3",
        ],
        &["tournament", "--bots", "rush,economy", "--ruleset"],
    ];
    for args in cases {
        let (code, out, err) = run(args);
        assert_eq!(code, 2, "{args:?}");
        assert!(out.is_empty(), "{args:?}: unexpected stdout {out:?}");
        assert!(!err.is_empty(), "{args:?}: expected usage on stderr");
    }
    assert!(!missing.0.exists(), "invalid ruleset must not write output");
}
