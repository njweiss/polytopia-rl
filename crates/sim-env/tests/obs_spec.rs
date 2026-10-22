//! Milestone 9 "Tests" item 1: `obs_spec` counts, names and order, and `encode_*` overwriting the
//! whole buffer. Expected names come from the brief's tables (groups as `"group.variant"`, enum
//! variants by their `snake_case` data names, in declaration order); see docs/QUESTIONS.md Q15
//! for the names the brief does not spell out (`phase.*`, `tech.*`).

mod common;

use common::*;
use sim_core::data::{ALL_TECHS, ALL_UNITS};
use sim_core::map::MapParams;
use sim_core::setup::new_game_from_seed;
use sim_core::view::PlayerView;
use sim_env::obs::{encode_board, encode_globals, obs_spec};

fn expected_channel_names() -> Vec<String> {
    let mut v: Vec<String> = vec!["explored".into()];
    for t in ["field", "forest", "mountain"] {
        v.push(format!("terrain.{t}"));
    }
    for r in ["fruit", "crop", "animal", "metal"] {
        v.push(format!("resource.{r}"));
    }
    for i in ["lumber_hut", "farm", "mine"] {
        v.push(format!("improvement.{i}"));
    }
    v.push("village".into());
    v.push("territory.mine".into());
    v.push("territory.enemy".into());
    for c in ["mine", "enemy", "level", "capital"] {
        v.push(format!("city.{c}"));
    }
    let units = ["warrior", "rider", "archer", "defender", "super_unit"];
    for u in units {
        v.push(format!("my_unit.{u}"));
    }
    for u in units {
        v.push(format!("enemy_unit.{u}"));
    }
    v.push("health".into());
    for ph in ["fresh", "moved", "attacked", "done"] {
        v.push(format!("phase.{ph}"));
    }
    for name in ["veteran", "can_promote", "can_harvest", "can_build"] {
        v.push(name.into());
    }
    v
}

fn expected_global_names() -> Vec<String> {
    let mut v: Vec<String> = [
        "stars",
        "income",
        "turn",
        "pending_reward",
        "my_score",
        "opp_score",
        "score_diff",
        "my_cities",
        "my_units",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for t in [
        "climbing",
        "hunting",
        "organization",
        "riding",
        "archery",
        "forestry",
        "farming",
        "strategy",
        "mining",
    ] {
        v.push(format!("tech.{t}"));
    }
    v
}

#[test]
fn counts_equal_the_sums_implied_by_the_enum_sizes() {
    // explored 1, terrain 3, resource 4, improvement 3, village 1, territory 2, city 4,
    // my_unit |UnitType|, enemy_unit |UnitType|, health 1, phase 4, veteran 1, can_* 3.
    let channels = 1 + 3 + 4 + 3 + 1 + 2 + 4 + ALL_UNITS.len() * 2 + 1 + 4 + 1 + 3;
    // 9 scalars, then one per Tech.
    let globals = 9 + ALL_TECHS.len();
    assert_eq!(channels, ch::COUNT, "test layout constants");
    assert_eq!(globals, gl::COUNT, "test layout constants");
    for size in [11u8, 8] {
        let spec = obs_spec(size);
        assert_eq!(spec.height, size as usize);
        assert_eq!(spec.width, size as usize);
        assert_eq!(spec.channels, channels);
        assert_eq!(spec.globals, globals);
        assert_eq!(spec.channel_names.len(), channels);
        assert_eq!(spec.global_names.len(), globals);
    }
}

#[test]
fn channel_names_in_documented_order() {
    assert_eq!(obs_spec(11).channel_names, expected_channel_names());
}

#[test]
fn global_names_in_documented_order() {
    assert_eq!(obs_spec(11).global_names, expected_global_names());
}

#[test]
fn names_are_unique() {
    let spec = obs_spec(11);
    let mut all: Vec<&String> = spec
        .channel_names
        .iter()
        .chain(&spec.global_names)
        .collect();
    let n = all.len();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), n, "duplicate names");
}

#[test]
fn encode_overwrites_the_whole_buffer() {
    let s = new_game_from_seed(5, &MapParams::default(), &cfg(30));
    for v in [0u8, 1] {
        let view = PlayerView::new(&s, v);
        let spec = obs_spec(11);
        let len = spec.channels * spec.height * spec.width;

        let mut clean = vec![0.0f32; len];
        encode_board(&view, &mut clean);
        for garbage in [7.5f32, -3.0, f32::NAN] {
            let mut dirty = vec![garbage; len];
            encode_board(&view, &mut dirty);
            assert_eq!(
                dirty.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                clean.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "board: pre-filled {garbage} was not fully overwritten"
            );
        }

        let mut clean = vec![0.0f32; spec.globals];
        encode_globals(&view, &mut clean);
        for garbage in [7.5f32, -3.0, f32::NAN] {
            let mut dirty = vec![garbage; spec.globals];
            encode_globals(&view, &mut dirty);
            assert_eq!(
                dirty.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                clean.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "globals: pre-filled {garbage} was not fully overwritten"
            );
        }
    }
}
