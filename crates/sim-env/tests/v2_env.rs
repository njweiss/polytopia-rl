//! Milestone 20 Phase A: v2 observation (`obs_spec_for`, 64-channel board with damage preview,
//! 33 globals), 15-kind factored action space & masks, `obs_hash_for`, `Env` under `Ruleset::V2`,
//! and v2 no-cheating (including the documented damage-preview exception on attackable enemies).

mod common;

use bots::{make_bot, BotKind};
use common::*;
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{
    techs, Building, Monument, Reward, SiteAction, Tech, UnitType, ALL_MONUMENTS,
};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{
    CityId, Config, Dir, Improvement, MonumentSet, Outcome, Phase, PlayerId, Ruleset, State,
    TechSet, UnitId,
};
use sim_core::testing::StateBuilder;
use sim_core::view::PlayerView;
use sim_env::actions::{
    action_types, arg_len, arg_len_for, decode_action, encode_action, num_actors, ActionMasks,
    ActionType, DecodeError, Factored, NUM_ACTION_TYPES, V2_NUM_ACTION_TYPES,
};
use sim_env::env::{AgentSeat, Env, EnvConfig, Opponent, RewardConfig};
use sim_env::hash::{obs_canonical, obs_canonical_for, obs_hash, obs_hash_for};
use sim_env::obs::{
    encode_board, encode_globals, obs_spec, obs_spec_for, CHANNELS, GLOBALS, V2_CHANNELS,
    V2_GLOBALS,
};
use sim_tools::replay::{replay, ReplayLog};

/// Independent v2 channel offsets (docs/MILESTONE_20 "Appended V2 Board Channels").
mod ch2 {
    pub const MY_SWORDSMAN: usize = 37;
    pub const MY_CATAPULT: usize = 38;
    pub const MY_KNIGHT: usize = 39;
    pub const ENEMY_SWORDSMAN: usize = 40;
    pub const ENEMY_CATAPULT: usize = 41;
    pub const ENEMY_KNIGHT: usize = 42;
    pub const ROAD: usize = 43;
    pub const CITY_CONNECTED: usize = 44;
    pub const BLD_SAWMILL: usize = 45;
    pub const BLD_WINDMILL: usize = 46;
    pub const BLD_FORGE: usize = 47;
    pub const BLD_MARKET: usize = 48;
    pub const BLD_LEVEL: usize = 49;
    pub const LIGHTHOUSE: usize = 50;
    pub const MON_PARK: usize = 51;
    pub const MON_GATE: usize = 52;
    pub const MON_EYE: usize = 53;
    pub const CAN_BUILD_ROAD: usize = 54;
    pub const CAN_BUILD_BUILDING: usize = 55;
    pub const CAN_BUILD_MONUMENT: usize = 56;
    pub const CAN_DESTROY: usize = 57;
    pub const PREV_TARGET_CAN_KILL: usize = 58;
    pub const PREV_TARGET_MAX_DMG: usize = 59;
    pub const PREV_TARGET_MIN_RETAL: usize = 60;
    pub const PREV_TARGET_DEF_BONUS: usize = 61;
    pub const PREV_ATTACKER_CAN_KILL: usize = 62;
    pub const PREV_ATTACKER_MAX_DMG: usize = 63;
    pub const COUNT: usize = 64;
}

/// Independent v2 global offsets (docs/MILESTONE_20 "Appended V2 Global Features").
mod gl2 {
    pub const V2_TECH: usize = 18; // roads, free_spirit, trade, chivalry, mathematics, construction, smithery
    pub const MY_KILLS: usize = 25;
    pub const MY_LIGHTHOUSES: usize = 26;
    pub const MON_UNLOCKED: usize = 27; // park_of_fortune, gate_of_power, eye_of_god
    pub const MON_PLACED: usize = 30; // park_of_fortune, gate_of_power, eye_of_god
    pub const COUNT: usize = 33;
}

const V2_KIND_ORDER: [ActionType; 15] = [
    ActionType::Move,
    ActionType::Attack,
    ActionType::Capture,
    ActionType::Recover,
    ActionType::Promote,
    ActionType::Train,
    ActionType::ChooseReward,
    ActionType::Site,
    ActionType::Research,
    ActionType::EndTurn,
    ActionType::Disband,
    ActionType::BuildRoad,
    ActionType::BuildBuilding,
    ActionType::BuildMonument,
    ActionType::Destroy,
];

fn v2_cfg(turn_cap: u32) -> Config {
    Config {
        turn_cap,
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn v2_env_cfg(size: u8, opponent: Opponent, seat: AgentSeat, turn_cap: u32) -> EnvConfig {
    EnvConfig {
        map: MapParams {
            size,
            ..MapParams::default()
        },
        config: v2_cfg(turn_cap),
        opponent,
        agent_seat: seat,
        reward: RewardConfig::default(),
        max_agent_actions: 5000,
    }
}

fn set_road(s: &mut State, q: Pos) {
    let i = ti(s.size, q);
    s.tiles[i].road = true;
}

fn encode_v2_board(s: &State, v: PlayerId) -> Vec<f32> {
    let n = s.size as usize;
    let mut out = vec![0.0f32; ch2::COUNT * n * n];
    encode_board(&PlayerView::new(s, v), &mut out);
    out
}

fn encode_v2_globals(s: &State, v: PlayerId) -> Vec<f32> {
    let mut out = vec![0.0f32; gl2::COUNT];
    encode_globals(&PlayerView::new(s, v), &mut out);
    out
}

fn my_fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn expected_v2_channel_names() -> Vec<String> {
    let mut v = obs_spec(11).channel_names;
    assert_eq!(v.len(), 37);
    let extra = [
        "my_unit.swordsman",
        "my_unit.catapult",
        "my_unit.knight",
        "enemy_unit.swordsman",
        "enemy_unit.catapult",
        "enemy_unit.knight",
        "road",
        "city.connected",
        "building.sawmill",
        "building.windmill",
        "building.forge",
        "building.market",
        "building.level",
        "lighthouse",
        "monument.park_of_fortune",
        "monument.gate_of_power",
        "monument.eye_of_god",
        "can_build_road",
        "can_build_building",
        "can_build_monument",
        "can_destroy",
        "preview.target_can_kill",
        "preview.target_max_dmg",
        "preview.target_min_retal",
        "preview.target_def_bonus",
        "preview.attacker_can_kill",
        "preview.attacker_max_dmg",
    ];
    v.extend(extra.iter().map(|s| s.to_string()));
    assert_eq!(v.len(), 64);
    v
}

fn expected_v2_global_names() -> Vec<String> {
    let mut v = obs_spec(11).global_names;
    assert_eq!(v.len(), 18);
    let extra = [
        "tech.roads",
        "tech.free_spirit",
        "tech.trade",
        "tech.chivalry",
        "tech.mathematics",
        "tech.construction",
        "tech.smithery",
        "my_kills",
        "my_lighthouses",
        "monument_unlocked.park_of_fortune",
        "monument_unlocked.gate_of_power",
        "monument_unlocked.eye_of_god",
        "monument_placed.park_of_fortune",
        "monument_placed.gate_of_power",
        "monument_placed.eye_of_god",
    ];
    v.extend(extra.iter().map(|s| s.to_string()));
    assert_eq!(v.len(), 33);
    v
}

// ============================================================================
// 1. ObsSpec under V1 and V2
// ============================================================================

#[test]
fn obs_spec_for_v2_counts_names_and_exact_v1_prefix() {
    assert_eq!(CHANNELS, 37);
    assert_eq!(GLOBALS, 18);
    assert_eq!(V2_CHANNELS, 64);
    assert_eq!(V2_GLOBALS, 33);

    for size in [6u8, 8, 11] {
        let v1 = obs_spec(size);
        assert_eq!(obs_spec_for(size, Ruleset::V1), v1);

        let v2 = obs_spec_for(size, Ruleset::V2);
        assert_eq!(v2.height, size as usize);
        assert_eq!(v2.width, size as usize);
        assert_eq!(v2.channels, 64);
        assert_eq!(v2.globals, 33);
        assert_eq!(v2.channel_names, expected_v2_channel_names());
        assert_eq!(v2.global_names, expected_v2_global_names());
        assert_eq!(&v2.channel_names[..37], v1.channel_names.as_slice());
        assert_eq!(&v2.global_names[..18], v1.global_names.as_slice());

        // All channel + global names are unique.
        let mut all: Vec<&String> = v2.channel_names.iter().chain(&v2.global_names).collect();
        let n = all.len();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n, "duplicate names in v2 obs_spec");
    }
}

// ============================================================================
// 2. V2 Board Encoding (all 27 new channels + v1 prefix + overwrite)
// ============================================================================

#[test]
fn encode_board_v2_covers_all_27_new_channels_and_hides_fog_and_enemy_connections() {
    const N: u8 = 8;
    // All v2 techs so every v2 site/build/destroy/burn_forest action can fire.
    let all_v2_techs = techs(Ruleset::V2);
    let mut s = StateBuilder::new(N)
        .config(v2_cfg(30))
        .stars(0, 40)
        .techs(0, all_v2_techs)
        .techs(1, &[Tech::Organization])
        .player_with(0, |pl| {
            pl.monuments_unlocked = MonumentSet::from_slice(&[
                Monument::ParkOfFortune,
                Monument::GateOfPower,
                Monument::EyeOfGod,
            ]);
            pl.monuments_placed = MonumentSet::from_slice(&[Monument::ParkOfFortune]);
        })
        // Own capital c0 at (1,1); own connected city c1 at (1,4); own unconnected city c2 at (4,1).
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(0, p(1, 4), |c| c.connected_to = Some(CityId(0)))
        .city(0, p(4, 1))
        // Enemy capital c3 at (5,5) and enemy connected city c4 at (5,2) (explored, MUST NOT leak city.connected!).
        .city_with(1, p(5, 5), |c| c.is_capital = true)
        .city_with(1, p(5, 2), |c| c.connected_to = Some(CityId(3)))
        // Own v2 units: Swordsman at (1,1) Fresh, Catapult at (1,4) Moved, Knight at (4,1) veteran Attacked 7/15.
        .unit(0, UnitType::Swordsman, p(1, 1))
        .unit_with(0, UnitType::Catapult, p(1, 4), |u| {
            u.health = 6;
            u.phase = Phase::Moved;
        })
        .unit_with(0, UnitType::Knight, p(4, 1), |u| {
            u.max_health = 15;
            u.health = 7;
            u.veteran = true;
            u.phase = Phase::Attacked;
        })
        // Enemy v2 units (kept >3 tiles away from own units so preview channels are isolated in the next test):
        // Swordsman at (5,5) Done, Catapult at (6,6) veteran 8/15, Knight at (6,4).
        .unit_with(1, UnitType::Swordsman, p(5, 5), |u| u.phase = Phase::Done)
        .unit_with(1, UnitType::Catapult, p(6, 6), |u| {
            u.max_health = 15;
            u.health = 8;
            u.veteran = true;
            u.phase = Phase::Moved;
        })
        .unit(1, UnitType::Knight, p(6, 4))
        // Buildings in c0/c1/c2 territories:
        // Sawmill at (0,2) adjacent to 2 lumber huts at (0,1) and (1,2) -> level 2.
        .terrain(p(0, 1), Terrain::Forest)
        .improvement(p(0, 1), Improvement::LumberHut)
        .terrain(p(1, 2), Terrain::Forest)
        .improvement(p(1, 2), Improvement::LumberHut)
        .building(p(0, 2), Building::Sawmill)
        // Windmill at (2,0) adjacent to 1 farm at (1,0) -> level 1.
        .resource(p(1, 0), Resource::Crop)
        .improvement(p(1, 0), Improvement::Farm)
        .building(p(2, 0), Building::Windmill)
        // Forge at (2,4) adjacent to 1 mine at (2,5) -> level 1.
        .terrain(p(2, 5), Terrain::Mountain)
        .resource(p(2, 5), Resource::Metal)
        .improvement(p(2, 5), Improvement::Mine)
        .building(p(2, 4), Building::Forge)
        // Market at (2,2) adjacent to Sawmill (1,2 is lumber hut, wait: let's put Market at (1,3) adjacent to Sawmill (0,2) L2 and Forge (2,4) L1 -> L3).
        .building(p(1, 3), Building::Market)
        // Own unimproved forest at (3,0) -> BurnForest sets can_build (channel 36), NOT can_build_building (55).
        .terrain(p(3, 0), Terrain::Forest)
        // Lighthouses on corners: (0,0) explored, (7,7) unexplored!
        .lighthouse(p(0, 0))
        .lighthouse(p(7, 7))
        // Explored monuments: ParkOfFortune at (2,1), GateOfPower at (0,5), EyeOfGod at (5,0).
        .monument(p(2, 1), Monument::ParkOfFortune)
        .monument(p(0, 5), Monument::GateOfPower)
        .monument(p(5, 0), Monument::EyeOfGod)
        // Unexplored v2 tiles in row 7 (7,0..=7): road, building, monument, lighthouse.
        .building(p(7, 2), Building::Sawmill)
        .monument(p(7, 4), Monument::EyeOfGod)
        .explore_all()
        .hide(
            0,
            &[
                p(7, 0),
                p(7, 1),
                p(7, 2),
                p(7, 3),
                p(7, 4),
                p(7, 5),
                p(7, 6),
                p(7, 7),
            ],
        )
        .build();
    // Roads: own tile (2,2), neutral tile (3,4), unexplored tile (7,1).
    set_road(&mut s, p(2, 2));
    set_road(&mut s, p(3, 4));
    set_road(&mut s, p(7, 1));

    // Sanity-check building levels on the scripted board before encoding.
    assert_eq!(s.tile(p(0, 2)).building_level, 2);
    assert_eq!(s.tile(p(2, 0)).building_level, 1);
    assert_eq!(s.tile(p(2, 4)).building_level, 1);
    assert_eq!(s.tile(p(1, 3)).building_level, 3);

    let e = encode_v2_board(&s, 0);
    let at = |c: usize, q: Pos| e[bi(N, c, q)];

    // Own v2 units: encoded at 37..40, NOT 18..23; health/phase/veteran shared.
    assert_eq!(at(ch2::MY_SWORDSMAN, p(1, 1)), 1.0);
    assert_eq!(at(ch::HEALTH, p(1, 1)), 1.0);
    assert_eq!(at(ch::PHASE, p(1, 1)), 1.0); // Fresh
    assert_eq!(at(ch2::MY_CATAPULT, p(1, 4)), 1.0);
    assert_approx(at(ch::HEALTH, p(1, 4)), 0.6, "catapult health 6/10");
    assert_eq!(at(ch::PHASE + 1, p(1, 4)), 1.0); // Moved
    assert_eq!(at(ch2::MY_KNIGHT, p(4, 1)), 1.0);
    assert_approx(at(ch::HEALTH, p(4, 1)), 7.0 / 15.0, "knight health 7/15");
    assert_eq!(at(ch::PHASE + 2, p(4, 1)), 1.0); // Attacked
    assert_eq!(at(ch::VETERAN, p(4, 1)), 1.0);
    for q in [p(1, 1), p(1, 4), p(4, 1)] {
        for c in ch::MY_UNIT..ch::ENEMY_UNIT {
            assert_eq!(
                at(c, q),
                0.0,
                "v2 own unit at {q:?} leaked to v1 channel {c}"
            );
        }
    }

    // Enemy v2 units: encoded at 40..43, health/veteran visible, phase NEVER visible.
    assert_eq!(at(ch2::ENEMY_SWORDSMAN, p(5, 5)), 1.0);
    assert_eq!(at(ch2::ENEMY_CATAPULT, p(6, 6)), 1.0);
    assert_approx(at(ch::HEALTH, p(6, 6)), 8.0 / 15.0, "enemy cat health");
    assert_eq!(at(ch::VETERAN, p(6, 6)), 1.0);
    assert_eq!(at(ch2::ENEMY_KNIGHT, p(6, 4)), 1.0);
    for q in [p(5, 5), p(6, 6), p(6, 4)] {
        for ph in 0..4 {
            assert_eq!(at(ch::PHASE + ph, q), 0.0, "enemy phase leaked at {q:?}");
        }
        for c in ch::ENEMY_UNIT..ch::HEALTH {
            assert_eq!(at(c, q), 0.0, "v2 enemy unit at {q:?} leaked to v1 ch {c}");
        }
    }

    // Roads.
    assert_eq!(at(ch2::ROAD, p(2, 2)), 1.0);
    assert_eq!(at(ch2::ROAD, p(3, 4)), 1.0);
    assert_eq!(at(ch2::ROAD, p(3, 5)), 0.0);

    // city.connected: ONLY own connected city (1,4); NOT own capital (1,1), NOT own unconnected (4,1),
    // and NEVER enemy connected city (5,2)!
    assert_eq!(at(ch2::CITY_CONNECTED, p(1, 4)), 1.0);
    assert_eq!(at(ch2::CITY_CONNECTED, p(1, 1)), 0.0);
    assert_eq!(at(ch2::CITY_CONNECTED, p(4, 1)), 0.0);
    assert_eq!(
        at(ch2::CITY_CONNECTED, p(5, 2)),
        0.0,
        "enemy city.connected leaked!"
    );
    assert_eq!(at(ch2::CITY_CONNECTED, p(5, 5)), 0.0);

    // Buildings + building.level.
    assert_eq!(at(ch2::BLD_SAWMILL, p(0, 2)), 1.0);
    assert_approx(at(ch2::BLD_LEVEL, p(0, 2)), 2.0 / 8.0, "sawmill L2");
    assert_eq!(at(ch2::BLD_WINDMILL, p(2, 0)), 1.0);
    assert_approx(at(ch2::BLD_LEVEL, p(2, 0)), 1.0 / 8.0, "windmill L1");
    assert_eq!(at(ch2::BLD_FORGE, p(2, 4)), 1.0);
    assert_approx(at(ch2::BLD_LEVEL, p(2, 4)), 1.0 / 8.0, "forge L1");
    assert_eq!(at(ch2::BLD_MARKET, p(1, 3)), 1.0);
    assert_approx(at(ch2::BLD_LEVEL, p(1, 3)), 3.0 / 8.0, "market L3");
    assert_eq!(at(ch2::BLD_LEVEL, p(2, 2)), 0.0, "no building -> level 0");

    // Lighthouse: explored corner is 1.0; unexplored corner (7,7) is 0.0.
    assert_eq!(at(ch2::LIGHTHOUSE, p(0, 0)), 1.0);
    assert_eq!(
        at(ch2::LIGHTHOUSE, p(7, 7)),
        0.0,
        "unexplored lighthouse leaked!"
    );

    // Monuments.
    assert_eq!(at(ch2::MON_PARK, p(2, 1)), 1.0);
    assert_eq!(at(ch2::MON_GATE, p(0, 5)), 1.0);
    assert_eq!(at(ch2::MON_EYE, p(5, 0)), 1.0);

    // Legal action hint channels cross-checked against `legal_actions(&s)`.
    let legal = legal_actions(&s);
    assert!(
        legal.contains(&Action::Site {
            at: p(3, 0),
            action: SiteAction::BurnForest
        }),
        "test premise: BurnForest legal at (3,0)"
    );
    assert_eq!(at(ch::CAN_BUILD, p(3, 0)), 1.0, "BurnForest sets can_build");
    assert_eq!(
        at(ch2::CAN_BUILD_BUILDING, p(3, 0)),
        0.0,
        "BurnForest is Site, not BuildBuilding"
    );

    // Neutral BuildRoad at (3,5) (neutral, explored, field, empty).
    assert!(
        legal.contains(&Action::BuildRoad { at: p(3, 5) }),
        "test premise: neutral BuildRoad legal at (3,5)"
    );
    assert_eq!(at(ch2::CAN_BUILD_ROAD, p(3, 5)), 1.0);
    // Own BuildRoad at (3,1) (own field, no road).
    assert!(legal.contains(&Action::BuildRoad { at: p(3, 1) }));
    assert_eq!(at(ch2::CAN_BUILD_ROAD, p(3, 1)), 1.0);
    // Already has road at (2,2) -> can_build_road == 0.
    assert_eq!(at(ch2::CAN_BUILD_ROAD, p(2, 2)), 0.0);

    // BuildBuilding: e.g. Windmill or Forge or Sawmill on eligible own field (e.g. (0,3) adjacent to nothing,
    // while (0,4) or (2,3) adjacent to Forge/LumberHut; let's check every tile matches `legal`!).
    for q in all_pos(N) {
        let want_road = legal.contains(&Action::BuildRoad { at: q });
        let want_bld = legal
            .iter()
            .any(|a| matches!(a, Action::BuildBuilding { at, .. } if *at == q));
        let want_mon = legal
            .iter()
            .any(|a| matches!(a, Action::BuildMonument { at, .. } if *at == q));
        let want_dest = legal.contains(&Action::Destroy { at: q });
        assert_eq!(
            at(ch2::CAN_BUILD_ROAD, q) == 1.0,
            want_road,
            "can_build_road at {q:?}"
        );
        assert_eq!(
            at(ch2::CAN_BUILD_BUILDING, q) == 1.0,
            want_bld,
            "can_build_building at {q:?}"
        );
        assert_eq!(
            at(ch2::CAN_BUILD_MONUMENT, q) == 1.0,
            want_mon,
            "can_build_monument at {q:?}"
        );
        assert_eq!(
            at(ch2::CAN_DESTROY, q) == 1.0,
            want_dest,
            "can_destroy at {q:?}"
        );
    }
    // Sanity check that each of the 4 new `can_*` channels actually fired on at least one tile!
    assert!(all_pos(N)
        .iter()
        .any(|&q| at(ch2::CAN_BUILD_ROAD, q) == 1.0));
    assert!(all_pos(N)
        .iter()
        .any(|&q| at(ch2::CAN_BUILD_BUILDING, q) == 1.0));
    assert!(all_pos(N)
        .iter()
        .any(|&q| at(ch2::CAN_BUILD_MONUMENT, q) == 1.0));
    assert!(all_pos(N).iter().any(|&q| at(ch2::CAN_DESTROY, q) == 1.0));

    // Unexplored row 7 MUST be all zeros across all 64 channels.
    for col in 0..N {
        let q = p(7, col);
        for c in 0..ch2::COUNT {
            assert_eq!(at(c, q), 0.0, "unexplored {q:?} channel {c}");
        }
    }

    // Overwrites pre-filled garbage completely.
    let view = PlayerView::new(&s, 0);
    for garbage in [7.5f32, -3.0, f32::NAN] {
        let mut dirty = vec![garbage; ch2::COUNT * (N as usize) * (N as usize)];
        encode_board(&view, &mut dirty);
        assert_eq!(
            dirty.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            e.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "board v2 did not overwrite {garbage}"
        );
    }
}

#[test]
fn encode_board_v2_building_level_caps_at_8() {
    // 7x7 board: cities placed so all improvements and buildings belong to player 0.
    // Market at (2,2) adjacent to Sawmill at (1,2) (5 adjacent lumber huts -> L5) and Windmill at
    // (3,2) (4 adjacent farms -> L4) => Market level 9, encoded as min(9, 8) / 8 = 1.0.
    let s = StateBuilder::new(7)
        .config(v2_cfg(30))
        .explore_all()
        .city_with(0, p(0, 0), |c| c.is_capital = true)
        .city(0, p(0, 3))
        .city(0, p(2, 3))
        .city(0, p(3, 0))
        .city(0, p(5, 2))
        // 5 lumber huts adjacent to Sawmill (1,2).
        .terrain(p(0, 1), Terrain::Forest)
        .improvement(p(0, 1), Improvement::LumberHut)
        .terrain(p(0, 2), Terrain::Forest)
        .improvement(p(0, 2), Improvement::LumberHut)
        .terrain(p(1, 1), Terrain::Forest)
        .improvement(p(1, 1), Improvement::LumberHut)
        .terrain(p(1, 3), Terrain::Forest)
        .improvement(p(1, 3), Improvement::LumberHut)
        .terrain(p(2, 1), Terrain::Forest)
        .improvement(p(2, 1), Improvement::LumberHut)
        .building(p(1, 2), Building::Sawmill)
        // 4 farms adjacent to Windmill (3,2).
        .improvement(p(3, 1), Improvement::Farm)
        .improvement(p(3, 3), Improvement::Farm)
        .improvement(p(4, 2), Improvement::Farm)
        .improvement(p(4, 3), Improvement::Farm)
        .building(p(3, 2), Building::Windmill)
        .building(p(2, 2), Building::Market)
        .build();
    assert_eq!(
        s.tile(p(1, 2)).building_level,
        5,
        "test premise: sawmill L5"
    );
    assert_eq!(
        s.tile(p(3, 2)).building_level,
        4,
        "test premise: windmill L4"
    );
    assert_eq!(s.tile(p(2, 2)).building_level, 9, "test premise: market L9");
    let e = encode_v2_board(&s, 0);
    assert_eq!(e[bi(7, ch2::BLD_MARKET, p(2, 2))], 1.0);
    assert_approx(e[bi(7, ch2::BLD_LEVEL, p(2, 2))], 1.0, "capped at 8/8");
    assert_approx(e[bi(7, ch2::BLD_LEVEL, p(1, 2))], 5.0 / 8.0, "sawmill L5");
    assert_approx(e[bi(7, ch2::BLD_LEVEL, p(3, 2))], 4.0 / 8.0, "windmill L4");
}

#[test]
fn encode_board_v2_damage_preview_aggregation_and_zero_cases() {
    const N: u8 = 8;
    // Target 1 at (2,2): wounded enemy Defender (9/15) on forest with hidden enemy Archery (1.5x).
    //   - Attacked by own wounded Warrior (9/10) at (2,1) AND own full Swordsman (15/15) at (1,2).
    //   - Neither kills; Swordsman deals more damage (7/9) and takes smaller retal fraction (6/15 vs 8/9).
    // Target 2 at (5,5): wounded enemy Warrior (3/10) in enemy walled city (4.0x).
    //   - Attacked by own Archer (10/10) at (5,3) (range 2 -> 0 retal; dist 3 from (2,2), Q57 A)
    //     AND own SuperUnit (40/40) at (5,4) (kills!).
    //   - Target 2 has can_kill=1.0, max_dmg=1.0, min_retal=0.0, def_bonus=4.0/4.0=1.0.
    // Target 3 at (7,7): enemy Warrior (10/10) on forest, OUT OF RANGE of all own units -> ALL preview 0.0!
    // Attacker (own Archer at (5,3)) reaches both (5,5) and Target 4 at (3,5) (enemy Warrior 10/10, plain).
    let s = StateBuilder::new(N)
        .config(v2_cfg(30))
        .explore_all()
        .techs(1, &[Tech::Archery])
        .terrain(p(2, 2), Terrain::Forest)
        .terrain(p(7, 7), Terrain::Forest)
        .city_with(1, p(5, 5), |c| c.wall = true)
        // Own units:
        .unit_with(0, UnitType::Warrior, p(2, 1), |u| u.health = 9) // u0
        .unit(0, UnitType::Swordsman, p(1, 2)) // u1
        .unit(0, UnitType::Archer, p(5, 3)) // u2 (hits (5,5) and (3,5); out of range of (2,2))
        .unit(0, UnitType::SuperUnit, p(5, 4)) // u3 (hits (5,5) -> lethal!)
        // Enemy units:
        .unit_with(1, UnitType::Defender, p(2, 2), |u| u.health = 9) // u4
        .unit_with(1, UnitType::Warrior, p(5, 5), |u| u.health = 3) // u5
        .unit(1, UnitType::Warrior, p(7, 7)) // u6 (untargetable)
        .unit(1, UnitType::Warrior, p(3, 5)) // u7 (targetable by u2 only, non-lethal)
        .build();

    let view = PlayerView::new(&s, 0);
    let p_u0_t1 = view.attack_preview(UnitId(0), p(2, 2)).unwrap();
    let p_u1_t1 = view.attack_preview(UnitId(1), p(2, 2)).unwrap();
    let p_u2_t2 = view.attack_preview(UnitId(2), p(5, 5)).unwrap();
    let p_u3_t2 = view.attack_preview(UnitId(3), p(5, 5)).unwrap();
    let p_u2_t4 = view.attack_preview(UnitId(2), p(3, 5)).unwrap();

    assert!(!p_u0_t1.defender_dies && !p_u1_t1.defender_dies);
    assert!(p_u0_t1.retaliation > 0 && p_u1_t1.retaliation > 0);
    assert!(
        p_u3_t2.defender_dies,
        "SuperUnit kills 3 HP warrior in wall"
    );
    assert!(!p_u2_t4.defender_dies);

    let e = encode_v2_board(&s, 0);
    let at = |c: usize, q: Pos| e[bi(N, c, q)];

    // Target 1 (2,2): non-lethal multi-attacker, wounded defender (health=9, max=15), wounded attacker (health=9, max=10).
    let t1_max_dmg = (p_u0_t1.dealt as f32 / 9.0).max(p_u1_t1.dealt as f32 / 9.0);
    let t1_min_retal = (p_u0_t1.retaliation as f32 / 9.0).min(p_u1_t1.retaliation as f32 / 15.0);
    assert_eq!(at(ch2::PREV_TARGET_CAN_KILL, p(2, 2)), 0.0);
    assert_approx(
        at(ch2::PREV_TARGET_MAX_DMG, p(2, 2)),
        t1_max_dmg,
        "t1 max_dmg",
    );
    assert_approx(
        at(ch2::PREV_TARGET_MIN_RETAL, p(2, 2)),
        t1_min_retal,
        "t1 min_retal",
    );
    assert_approx(
        at(ch2::PREV_TARGET_DEF_BONUS, p(2, 2)),
        1.5 / 4.0,
        "t1 def_bonus (hidden Archery)",
    );

    // Target 2 (5,5): lethal by SuperUnit, wall 4.0x.
    assert_eq!(at(ch2::PREV_TARGET_CAN_KILL, p(5, 5)), 1.0);
    assert_approx(at(ch2::PREV_TARGET_MAX_DMG, p(5, 5)), 1.0, "t2 max_dmg");
    assert_approx(at(ch2::PREV_TARGET_MIN_RETAL, p(5, 5)), 0.0, "t2 min_retal");
    assert_approx(at(ch2::PREV_TARGET_DEF_BONUS, p(5, 5)), 1.0, "t2 wall 4/4");

    // Attacker channels on u2 (Archer at (5,3)): max across (5,5) and (3,5).
    let u2_max_dmg = (p_u2_t2.dealt as f32 / 3.0).max(p_u2_t4.dealt as f32 / 10.0);
    assert_eq!(
        at(ch2::PREV_ATTACKER_CAN_KILL, p(5, 3)),
        if p_u2_t2.defender_dies || p_u2_t4.defender_dies {
            1.0
        } else {
            0.0
        }
    );
    assert_approx(
        at(ch2::PREV_ATTACKER_MAX_DMG, p(5, 3)),
        u2_max_dmg,
        "u2 max_dmg",
    );
    // Attacker u3 (SuperUnit at (5,4)): lethal on (5,5).
    assert_eq!(at(ch2::PREV_ATTACKER_CAN_KILL, p(5, 4)), 1.0);
    assert_approx(at(ch2::PREV_ATTACKER_MAX_DMG, p(5, 4)), 1.0, "u3 max_dmg");
    // Attacker u1 (Swordsman at (1,2)): non-lethal on (2,2).
    assert_eq!(at(ch2::PREV_ATTACKER_CAN_KILL, p(1, 2)), 0.0);
    assert_approx(
        at(ch2::PREV_ATTACKER_MAX_DMG, p(1, 2)),
        p_u1_t1.dealt as f32 / 9.0,
        "u1 max_dmg",
    );

    // Untargetable enemy at (7,7) and empty tiles MUST have ALL 6 preview channels == 0.0
    // (catches target_min_retal defaulting to 1.0/+inf or def_bonus leaking on non-attackable enemies).
    for q in [p(7, 7), p(0, 0), p(7, 0), p(3, 3)] {
        for c in ch2::PREV_TARGET_CAN_KILL..=ch2::PREV_ATTACKER_MAX_DMG {
            assert_eq!(at(c, q), 0.0, "non-targeted {q:?} preview ch {c}");
        }
    }

    // When every attacker is Done, or on opponent's turn, or terminal, all 6 preview channels are 0.0.
    let mut done_s = s.clone();
    for u in &mut done_s.units {
        if u.owner == 0 {
            u.phase = Phase::Done;
        }
    }
    let mut opp_turn = s.clone();
    opp_turn.current = 1;
    let mut term = s.clone();
    term.outcome = Some(Outcome::Draw);
    for (label, st) in [
        ("all done", &done_s),
        ("opp turn", &opp_turn),
        ("terminal", &term),
    ] {
        let buf = encode_v2_board(st, 0);
        for q in all_pos(N) {
            for c in ch2::PREV_TARGET_CAN_KILL..=ch2::PREV_ATTACKER_MAX_DMG {
                assert_eq!(buf[bi(N, c, q)], 0.0, "{label}: {q:?} ch {c}");
            }
        }
    }
}

// ============================================================================
// 3. V2 Global Encoding (all 33 features)
// ============================================================================

#[test]
fn encode_globals_v2_covers_all_33_features_and_caps() {
    let s = StateBuilder::new(6)
        .config(v2_cfg(20))
        .turn(5)
        .stars(0, 15)
        .techs(
            0,
            &[
                Tech::Organization,
                Tech::Riding,
                Tech::Roads,
                Tech::Chivalry,
                Tech::Smithery,
            ],
        )
        .techs(1, &[Tech::Organization, Tech::Trade, Tech::Mathematics])
        .city_with(0, p(1, 1), |c| c.is_capital = true)
        .city_with(1, p(4, 4), |c| c.is_capital = true)
        .player_with(0, |pl| {
            pl.kills = 4;
            pl.lighthouses_discovered = 0b0101; // 2 lighthouses -> 0.5
            pl.monuments_unlocked =
                MonumentSet::from_slice(&[Monument::ParkOfFortune, Monument::EyeOfGod]);
            pl.monuments_placed = MonumentSet::from_slice(&[Monument::ParkOfFortune]);
        })
        .player_with(1, |pl| {
            pl.kills = 15; // capped at 10 -> 1.0
            pl.lighthouses_discovered = 0b1111; // 4 -> 1.0
            pl.monuments_unlocked = MonumentSet::from_slice(&[Monument::GateOfPower]);
            pl.monuments_placed = MonumentSet::from_slice(&[Monument::GateOfPower]);
        })
        .build();

    let g0 = encode_v2_globals(&s, 0);
    assert_eq!(g0.len(), 33);
    // Prefix 0..18 matches exact v1 formula (with v2 score/income).
    let (sc0, sc1) = (score(&s, 0), score(&s, 1));
    assert_approx(g0[gl::STARS], 15.0 / 50.0, "stars");
    assert_approx(g0[gl::INCOME], 2.0 / 20.0, "income");
    assert_approx(g0[gl::TURN], 5.0 / 20.0, "turn");
    assert_eq!(g0[gl::PENDING], 0.0);
    assert_approx(g0[gl::MY_SCORE], sc0.min(2000) as f32 / 2000.0, "my_score");
    assert_approx(
        g0[gl::OPP_SCORE],
        sc1.min(2000) as f32 / 2000.0,
        "opp_score",
    );
    assert_approx(
        g0[gl::SCORE_DIFF],
        ((sc0 as f32 - sc1 as f32) / 1000.0).clamp(-1.0, 1.0),
        "score_diff",
    );
    assert_approx(g0[gl::MY_CITIES], 1.0 / 8.0, "my_cities");
    assert_approx(g0[gl::MY_UNITS], 0.0, "my_units");
    // v1 techs: Organization (2), Riding (3).
    for (i, t) in TECHS.iter().enumerate() {
        let want = if matches!(t, Tech::Organization | Tech::Riding) {
            1.0
        } else {
            0.0
        };
        assert_eq!(g0[gl::TECH + i], want, "v1 tech {t:?}");
    }
    // v2 techs (18..25): Roads(0), FreeSpirit(1), Trade(2), Chivalry(3), Mathematics(4), Construction(5), Smithery(6).
    assert_eq!(
        &g0[gl2::V2_TECH..gl2::V2_TECH + 7],
        &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    );
    assert_approx(g0[gl2::MY_KILLS], 0.4, "kills 4/10");
    assert_approx(g0[gl2::MY_LIGHTHOUSES], 0.5, "lighthouses 2/4");
    assert_eq!(
        &g0[gl2::MON_UNLOCKED..gl2::MON_UNLOCKED + 3],
        &[1.0, 0.0, 1.0]
    );
    assert_eq!(&g0[gl2::MON_PLACED..gl2::MON_PLACED + 3], &[1.0, 0.0, 0.0]);

    // Player 1 checks caps (kills 15 -> 1.0, lighthouses 4 -> 1.0).
    let g1 = encode_v2_globals(&s, 1);
    assert_eq!(
        &g1[gl2::V2_TECH..gl2::V2_TECH + 7],
        &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    );
    assert_approx(g1[gl2::MY_KILLS], 1.0, "kills 15 capped at 10/10");
    assert_approx(g1[gl2::MY_LIGHTHOUSES], 1.0, "lighthouses 4/4");
    assert_eq!(
        &g1[gl2::MON_UNLOCKED..gl2::MON_UNLOCKED + 3],
        &[0.0, 1.0, 0.0]
    );
    assert_eq!(&g1[gl2::MON_PLACED..gl2::MON_PLACED + 3], &[0.0, 1.0, 0.0]);

    // Overwrites pre-filled NaN/garbage.
    let mut dirty = vec![f32::NAN; gl2::COUNT];
    encode_globals(&PlayerView::new(&s, 0), &mut dirty);
    assert_eq!(
        dirty.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        g0.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

// ============================================================================
// 4. V2 Factored Action Space & Masks + V1 Strict Rejection
// ============================================================================

#[test]
fn v2_action_types_arg_lens_and_round_trip_all_15_kinds() {
    assert_eq!(NUM_ACTION_TYPES, 10);
    assert_eq!(V2_NUM_ACTION_TYPES, 15);
    assert_eq!(action_types(Ruleset::V1), &ActionType::ALL);
    assert_eq!(action_types(Ruleset::V2), &V2_KIND_ORDER);
    for (i, &k) in V2_KIND_ORDER.iter().enumerate() {
        assert_eq!(k as u8 as usize, i, "{k:?}");
    }

    for (size, n) in [(6u8, 36usize), (8, 64), (11, 121)] {
        for &k in &ActionType::ALL {
            assert_eq!(arg_len_for(size, k, Ruleset::V1), arg_len(size, k));
        }
        let want_v2 = [n, n, 0, 0, 0, 8, 8, 7, 16, 0, 0, 0, 4, 3, 0];
        for (&k, &w) in V2_KIND_ORDER.iter().zip(&want_v2) {
            assert_eq!(arg_len_for(size, k, Ruleset::V2), w, "size {size} {k:?}");
        }
    }

    // 8x8 v2 board exercising all 15 ActionTypes (including neutral BuildRoad).
    let s = StateBuilder::new(8)
        .config(v2_cfg(30))
        .explore_all()
        .city(0, p(1, 1)) // city 0 at tile 9 -> actor 64 + 9 = 73
        .city(1, p(6, 6)) // enemy city 1 at tile 54 -> actor 64 + 54 = 118
        .unit(0, UnitType::Knight, p(2, 3)) // u0 at tile 19
        .unit(1, UnitType::Warrior, p(2, 4)) // u1 at tile 20
        .build();

    let cases: Vec<(Action, Factored)> = vec![
        (
            Action::Move {
                unit: UnitId(0),
                to: p(3, 3),
            },
            Factored {
                actor: 19,
                kind: ActionType::Move,
                arg: 27,
            },
        ),
        (
            Action::Attack {
                unit: UnitId(0),
                target: p(2, 4),
            },
            Factored {
                actor: 19,
                kind: ActionType::Attack,
                arg: 20,
            },
        ),
        (
            Action::Capture { unit: UnitId(0) },
            Factored {
                actor: 19,
                kind: ActionType::Capture,
                arg: 0,
            },
        ),
        (
            Action::Recover { unit: UnitId(0) },
            Factored {
                actor: 19,
                kind: ActionType::Recover,
                arg: 0,
            },
        ),
        (
            Action::Promote { unit: UnitId(0) },
            Factored {
                actor: 19,
                kind: ActionType::Promote,
                arg: 0,
            },
        ),
        (
            Action::Disband { unit: UnitId(0) },
            Factored {
                actor: 19,
                kind: ActionType::Disband,
                arg: 0,
            },
        ),
        (
            Action::Train {
                city: CityId(0),
                kind: UnitType::Swordsman,
            },
            Factored {
                actor: 73,
                kind: ActionType::Train,
                arg: 5,
            },
        ),
        (
            Action::Train {
                city: CityId(0),
                kind: UnitType::Catapult,
            },
            Factored {
                actor: 73,
                kind: ActionType::Train,
                arg: 6,
            },
        ),
        (
            Action::Train {
                city: CityId(0),
                kind: UnitType::Knight,
            },
            Factored {
                actor: 73,
                kind: ActionType::Train,
                arg: 7,
            },
        ),
        (
            Action::ChooseReward {
                city: CityId(0),
                reward: Reward::Park,
            },
            Factored {
                actor: 73,
                kind: ActionType::ChooseReward,
                arg: 6,
            },
        ),
        (
            Action::Site {
                at: p(0, 0),
                action: SiteAction::BurnForest,
            },
            Factored {
                actor: 128,
                kind: ActionType::Site,
                arg: 6,
            },
        ),
        // Neutral BuildRoad at (4,5) = tile 37 -> site actor 128 + 37 = 165.
        (
            Action::BuildRoad { at: p(4, 5) },
            Factored {
                actor: 165,
                kind: ActionType::BuildRoad,
                arg: 0,
            },
        ),
        (
            Action::BuildBuilding {
                at: p(0, 1),
                building: Building::Sawmill,
            },
            Factored {
                actor: 129,
                kind: ActionType::BuildBuilding,
                arg: 0,
            },
        ),
        (
            Action::BuildBuilding {
                at: p(0, 1),
                building: Building::Windmill,
            },
            Factored {
                actor: 129,
                kind: ActionType::BuildBuilding,
                arg: 1,
            },
        ),
        (
            Action::BuildBuilding {
                at: p(0, 1),
                building: Building::Forge,
            },
            Factored {
                actor: 129,
                kind: ActionType::BuildBuilding,
                arg: 2,
            },
        ),
        (
            Action::BuildBuilding {
                at: p(0, 1),
                building: Building::Market,
            },
            Factored {
                actor: 129,
                kind: ActionType::BuildBuilding,
                arg: 3,
            },
        ),
        (
            Action::BuildMonument {
                at: p(0, 2),
                monument: Monument::ParkOfFortune,
            },
            Factored {
                actor: 130,
                kind: ActionType::BuildMonument,
                arg: 0,
            },
        ),
        (
            Action::BuildMonument {
                at: p(0, 2),
                monument: Monument::GateOfPower,
            },
            Factored {
                actor: 130,
                kind: ActionType::BuildMonument,
                arg: 1,
            },
        ),
        (
            Action::BuildMonument {
                at: p(0, 2),
                monument: Monument::EyeOfGod,
            },
            Factored {
                actor: 130,
                kind: ActionType::BuildMonument,
                arg: 2,
            },
        ),
        (
            Action::Destroy { at: p(1, 0) },
            Factored {
                actor: 136,
                kind: ActionType::Destroy,
                arg: 0,
            },
        ),
        (
            Action::Research { tech: Tech::Roads },
            Factored {
                actor: 192,
                kind: ActionType::Research,
                arg: 9,
            },
        ),
        (
            Action::Research {
                tech: Tech::Smithery,
            },
            Factored {
                actor: 192,
                kind: ActionType::Research,
                arg: 15,
            },
        ),
        (
            Action::EndTurn,
            Factored {
                actor: 192,
                kind: ActionType::EndTurn,
                arg: 0,
            },
        ),
    ];
    for (action, want) in cases {
        assert_eq!(encode_action(&s, &action), want, "encode {action:?}");
        assert_eq!(decode_action(&s, &want), Ok(action), "decode {want:?}");
    }

    // V2 decode errors.
    let v2_errs = [
        (
            Factored {
                actor: 0,
                kind: ActionType::Disband,
                arg: 0,
            },
            DecodeError::NoActor,
        ),
        (
            Factored {
                actor: 20,
                kind: ActionType::Disband,
                arg: 0,
            },
            DecodeError::NoActor,
        ),
        (
            Factored {
                actor: 19,
                kind: ActionType::Disband,
                arg: 1,
            },
            DecodeError::BadArg,
        ),
        (
            Factored {
                actor: 73,
                kind: ActionType::Disband,
                arg: 0,
            },
            DecodeError::WrongKind,
        ),
        (
            Factored {
                actor: 19,
                kind: ActionType::BuildRoad,
                arg: 0,
            },
            DecodeError::WrongKind,
        ),
        (
            Factored {
                actor: 128,
                kind: ActionType::BuildRoad,
                arg: 1,
            },
            DecodeError::BadArg,
        ),
        (
            Factored {
                actor: 128,
                kind: ActionType::BuildBuilding,
                arg: 4,
            },
            DecodeError::BadArg,
        ),
        (
            Factored {
                actor: 128,
                kind: ActionType::BuildMonument,
                arg: 3,
            },
            DecodeError::BadArg,
        ),
        (
            Factored {
                actor: 128,
                kind: ActionType::Destroy,
                arg: 1,
            },
            DecodeError::BadArg,
        ),
        (
            Factored {
                actor: 73,
                kind: ActionType::Train,
                arg: 8,
            },
            DecodeError::BadArg,
        ),
        (
            Factored {
                actor: 128,
                kind: ActionType::Site,
                arg: 7,
            },
            DecodeError::BadArg,
        ),
        (
            Factored {
                actor: 192,
                kind: ActionType::Research,
                arg: 16,
            },
            DecodeError::BadArg,
        ),
    ];
    for (fac, err) in v2_errs {
        assert_eq!(decode_action(&s, &fac), Err(err), "v2 err {fac:?}");
    }
}

#[test]
fn v1_decode_action_rejects_all_v2_kinds_and_v2_arg_indices() {
    let s = StateBuilder::new(8)
        .explore_all()
        .city(0, p(1, 1)) // city actor 64 + 9 = 73
        .unit(0, UnitType::Warrior, p(2, 3)) // unit actor 19
        .build();
    assert_eq!(s.config.ruleset, Ruleset::V1);

    // Every v2 ActionType must be rejected with WrongKind under V1, even on its valid family!
    for (actor, kind) in [
        (19, ActionType::Disband),
        (128, ActionType::BuildRoad),
        (128, ActionType::BuildBuilding),
        (128, ActionType::BuildMonument),
        (128, ActionType::Destroy),
    ] {
        assert_eq!(
            decode_action(
                &s,
                &Factored {
                    actor,
                    kind,
                    arg: 0
                }
            ),
            Err(DecodeError::WrongKind),
            "V1 must reject {kind:?}"
        );
    }

    // V2 arg indices for Train (5..8), Site (6), Research (9..16) must be BadArg under V1!
    for arg in 5..8 {
        assert_eq!(
            decode_action(
                &s,
                &Factored {
                    actor: 73,
                    kind: ActionType::Train,
                    arg
                }
            ),
            Err(DecodeError::BadArg),
            "V1 Train arg {arg}"
        );
    }
    assert_eq!(
        decode_action(
            &s,
            &Factored {
                actor: 128,
                kind: ActionType::Site,
                arg: 6
            }
        ),
        Err(DecodeError::BadArg),
        "V1 Site BurnForest arg 6"
    );
    for arg in 9..16 {
        assert_eq!(
            decode_action(
                &s,
                &Factored {
                    actor: 192,
                    kind: ActionType::Research,
                    arg
                }
            ),
            Err(DecodeError::BadArg),
            "V1 Research arg {arg}"
        );
    }
}

fn check_mask_bijection(s: &State, ctx: &str) {
    let legal = legal_actions(s);
    let m = ActionMasks::new(s, &legal);
    let ruleset = s.config.ruleset;
    let kinds = action_types(ruleset);
    assert_eq!(m.actor().len(), num_actors(s.size), "{ctx}: actor len");

    // 1. Every legal action has true actor, kind_slice, and arg mask entries.
    let mut expected_paths: Vec<Factored> = legal.iter().map(|a| encode_action(s, a)).collect();
    for f in &expected_paths {
        assert!(m.actor()[f.actor], "{ctx}: actor false for {f:?}");
        let ks = m.kind_slice(f.actor);
        assert_eq!(ks.len(), kinds.len(), "{ctx}: kind_slice len");
        assert!(ks[f.kind as usize], "{ctx}: kind false for {f:?}");
        let am = m.arg(f.actor, f.kind);
        assert_eq!(
            am.len(),
            arg_len_for(s.size, f.kind, ruleset),
            "{ctx}: arg len for {:?}",
            f.kind
        );
        if !am.is_empty() {
            assert!(am[f.arg], "{ctx}: arg false for {f:?}");
        } else {
            assert_eq!(f.arg, 0);
        }
    }

    // 2. Every true path through actor -> kind_slice -> arg decodes to a legal action (1-to-1).
    let mut actual_paths = Vec::new();
    for (actor, &actor_ok) in m.actor().iter().enumerate() {
        let ks = m.kind_slice(actor);
        assert_eq!(ks.len(), kinds.len());
        if !actor_ok {
            assert!(
                ks.iter().all(|&x| !x),
                "{ctx}: false actor {actor} has true kind"
            );
            continue;
        }
        assert!(
            ks.iter().any(|&x| x),
            "{ctx}: true actor {actor} has no true kind"
        );
        for &kind in kinds {
            let am = m.arg(actor, kind);
            assert_eq!(am.len(), arg_len_for(s.size, kind, ruleset));
            if !ks[kind as usize] {
                assert!(
                    am.iter().all(|&x| !x),
                    "{ctx}: false kind {kind:?} has true arg"
                );
                continue;
            }
            if am.is_empty() {
                actual_paths.push(Factored {
                    actor,
                    kind,
                    arg: 0,
                });
            } else {
                let mut any = false;
                for (arg, &ok) in am.iter().enumerate() {
                    if ok {
                        any = true;
                        actual_paths.push(Factored { actor, kind, arg });
                    }
                }
                assert!(any, "{ctx}: true kind {kind:?} has no true arg");
            }
        }
    }

    let key = |f: &Factored| (f.actor, f.kind as u8, f.arg);
    expected_paths.sort_by_key(key);
    actual_paths.sort_by_key(key);
    assert_eq!(actual_paths, expected_paths, "{ctx}: bijection mismatch");

    for f in &actual_paths {
        let a = decode_action(s, f).unwrap_or_else(|e| panic!("{ctx}: decode {f:?} failed: {e:?}"));
        assert!(legal.contains(&a), "{ctx}: decoded {a:?} not in legal");
    }
}

#[test]
fn v2_action_masks_bijection_on_scripted_and_playout_states() {
    // Scripted v2 board where Disband, neutral BuildRoad, BuildBuilding, BuildMonument, Destroy,
    // BurnForest, v2 Train, and v2 Research are ALL legal simultaneously.
    let s = StateBuilder::new(8)
        .config(v2_cfg(30))
        .explore_all()
        .stars(0, 50)
        .techs(
            0,
            &[
                Tech::Organization,
                Tech::Climbing,
                Tech::Hunting,
                Tech::Riding,
                Tech::Forestry,
                Tech::Farming,
                Tech::Mining,
                Tech::Roads,
                Tech::FreeSpirit,
                Tech::Chivalry,
                Tech::Mathematics,
                Tech::Construction,
                // Omit Trade and Smithery so Research has v2 techs legal too!
            ],
        )
        .player_with(0, |pl| {
            pl.monuments_unlocked = MonumentSet::from_slice(&[
                Monument::ParkOfFortune,
                Monument::GateOfPower,
                Monument::EyeOfGod,
            ]);
        })
        .city_with(0, p(1, 1), |c| {
            c.level = 3;
            c.is_capital = true;
        })
        .unit(0, UnitType::Warrior, p(2, 2))
        .terrain(p(0, 1), Terrain::Forest)
        .improvement(p(0, 1), Improvement::LumberHut)
        .terrain(p(0, 2), Terrain::Forest) // BurnForest legal
        .resource(p(1, 0), Resource::Crop)
        .improvement(p(1, 0), Improvement::Farm) // Windmill + Destroy legal
        .build();

    let legal = legal_actions(&s);
    assert!(legal.iter().any(|a| matches!(a, Action::Disband { .. })));
    assert!(legal
        .iter()
        .any(|a| matches!(a, Action::BuildRoad { at } if s.tile(*at).owner_city.is_none())));
    assert!(legal
        .iter()
        .any(|a| matches!(a, Action::BuildBuilding { .. })));
    assert!(legal
        .iter()
        .any(|a| matches!(a, Action::BuildMonument { .. })));
    assert!(legal.iter().any(|a| matches!(a, Action::Destroy { .. })));
    assert!(legal.iter().any(|a| matches!(
        a,
        Action::Site {
            action: SiteAction::BurnForest,
            ..
        }
    )));
    assert!(legal.iter().any(|a| matches!(
        a,
        Action::Train {
            kind: UnitType::Knight | UnitType::Catapult,
            ..
        }
    )));
    assert!(legal.iter().any(|a| matches!(
        a,
        Action::Research {
            tech: Tech::Trade | Tech::Smithery
        }
    )));

    check_mask_bijection(&s, "scripted all-v2");

    // V1 kind_slice compatibility check.
    let v1 = new_game_from_seed(0, &MapParams::default(), &cfg(10));
    let m1 = ActionMasks::new(&v1, &legal_actions(&v1));
    for a in 0..num_actors(v1.size) {
        assert_eq!(m1.kind_slice(a).len(), 10);
        assert_eq!(m1.kind_slice(a), &m1.kind(a)[..]);
    }

    // Multi-seed v2 random + bot playouts.
    let mut checked = 0usize;
    for seed in 0..6u64 {
        let mut st = new_game_from_seed(seed, &MapParams::default(), &v2_cfg(10));
        let mut rng = Rng::new(seed + 500);
        let mut step = 0;
        while !st.is_terminal() {
            if step % 3 == 0 {
                check_mask_bijection(&st, &format!("v2 random seed {seed} step {step}"));
                checked += 1;
            }
            let legal = legal_actions(&st);
            let a = legal[random_index(&legal, &mut rng)];
            apply(&mut st, &a).unwrap();
            step += 1;
        }
        check_mask_bijection(&st, &format!("v2 random seed {seed} terminal"));
        checked += 1;

        let bot0 = make_bot(KINDS[seed as usize % 3], 0.1);
        let bot1 = make_bot(KINDS[(seed as usize + 1) % 3], 0.1);
        let mut rngs = [Rng::new(seed ^ SALT[0]), Rng::new(seed ^ SALT[1])];
        let mut st = new_game_from_seed(seed, &MapParams::default(), &v2_cfg(12));
        step = 0;
        while !st.is_terminal() {
            if step % 4 == 0 {
                check_mask_bijection(&st, &format!("v2 bot seed {seed} step {step}"));
                checked += 1;
            }
            let cur = st.current as usize;
            let a = if cur == 0 {
                bot0.choose(&PlayerView::new(&st, 0), &mut rngs[0])
            } else {
                bot1.choose(&PlayerView::new(&st, 1), &mut rngs[1])
            };
            apply(&mut st, &a).unwrap();
            step += 1;
        }
    }
    assert!(checked > 200, "too few states checked: {checked}");
}

// ============================================================================
// 5. obs_hash_for(size, Ruleset::V2) & V1 immutability
// ============================================================================

fn expected_v2_canonical(size: u8) -> String {
    let n = size as usize * size as usize;
    let mut items: Vec<String> = vec![size.to_string()];
    items.extend(expected_v2_channel_names());
    items.extend(expected_v2_global_names());
    let action_names = [
        "move",
        "attack",
        "capture",
        "recover",
        "promote",
        "train",
        "choose_reward",
        "site",
        "research",
        "end_turn",
        "disband",
        "build_road",
        "build_building",
        "build_monument",
        "destroy",
    ];
    items.extend(action_names.iter().map(|s| s.to_string()));
    let arg_lens = [n, n, 0, 0, 0, 8, 8, 7, 16, 0, 0, 0, 4, 3, 0];
    items.extend(arg_lens.iter().map(|x| x.to_string()));
    let unit_names = [
        "warrior",
        "rider",
        "archer",
        "defender",
        "super_unit",
        "swordsman",
        "catapult",
        "knight",
    ];
    items.extend(unit_names.iter().map(|s| s.to_string()));
    let reward_names = [
        "workshop",
        "explorer",
        "city_wall",
        "resources",
        "population_growth",
        "border_growth",
        "park",
        "super_unit",
    ];
    items.extend(reward_names.iter().map(|s| s.to_string()));
    let tech_names = [
        "climbing",
        "hunting",
        "organization",
        "riding",
        "archery",
        "forestry",
        "farming",
        "strategy",
        "mining",
        "roads",
        "free_spirit",
        "trade",
        "chivalry",
        "mathematics",
        "construction",
        "smithery",
    ];
    items.extend(tech_names.iter().map(|s| s.to_string()));
    let site_names = [
        "harvest_animal",
        "harvest_fruit",
        "lumber_hut",
        "clear_forest",
        "farm",
        "mine",
        "burn_forest",
    ];
    items.extend(site_names.iter().map(|s| s.to_string()));
    let building_names = ["sawmill", "windmill", "forge", "market"];
    items.extend(building_names.iter().map(|s| s.to_string()));
    let monument_names = ["park_of_fortune", "gate_of_power", "eye_of_god"];
    items.extend(monument_names.iter().map(|s| s.to_string()));
    assert_eq!(items.len(), 174);
    items.join("\n")
}

#[test]
fn obs_hash_for_v2_matches_independent_canonical_and_v1_stays_pinned() {
    // V1 identity.
    assert_eq!(obs_hash(11), "d5f9b1033cfd55ca");
    assert_eq!(obs_hash_for(11, Ruleset::V1), "d5f9b1033cfd55ca");
    for size in [3u8, 6, 8, 11] {
        assert_eq!(obs_canonical_for(size, Ruleset::V1), obs_canonical(size));
        assert_eq!(obs_hash_for(size, Ruleset::V1), obs_hash(size));
    }

    // V2 canonical & FNV-1a 64.
    for size in [3u8, 6, 8, 11] {
        let want_canon = expected_v2_canonical(size);
        assert!(!want_canon.ends_with('\n'));
        assert_eq!(obs_canonical_for(size, Ruleset::V2), want_canon);
        let want_hash = format!("{:016x}", my_fnv1a64(want_canon.as_bytes()));
        let got_hash = obs_hash_for(size, Ruleset::V2);
        assert_eq!(got_hash, want_hash);
        assert_eq!(got_hash, obs_hash_for(size, Ruleset::V2), "deterministic");
        assert_ne!(got_hash, obs_hash_for(size, Ruleset::V1));
    }
    assert_ne!(obs_hash_for(11, Ruleset::V2), obs_hash_for(6, Ruleset::V2));

    // Mutating any ingredient (including buildings & monuments) changes the hash.
    let base_items: Vec<String> = expected_v2_canonical(6)
        .split('\n')
        .map(str::to_string)
        .collect();
    let base_hash = my_fnv1a64(base_items.join("\n").as_bytes());
    for idx in [
        0,   // size
        1,   // v1 channel
        64,  // last v2 preview channel
        97,  // last v2 global
        112, // last action type (destroy)
        125, // BuildBuilding arg_len
        135, // Knight unit
        143, // SuperUnit reward
        159, // Smithery tech
        166, // BurnForest site action
        167, // Sawmill building
        171, // ParkOfFortune monument
        173, // EyeOfGod monument
    ] {
        let mut mutated = base_items.clone();
        mutated[idx].push('!');
        assert_ne!(
            my_fnv1a64(mutated.join("\n").as_bytes()),
            base_hash,
            "mutating item {idx} ({}) did not change hash",
            base_items[idx]
        );
    }
}

// ============================================================================
// 6. Env under Ruleset::V2 (TwoSeat, Bot, step_factored, bot_action, replay, determinism)
// ============================================================================

fn sample_v2_factored(env: &Env, rng: &mut Rng) -> Factored {
    let m = env.masks();
    let actors: Vec<usize> = m
        .actor()
        .iter()
        .enumerate()
        .filter_map(|(i, &ok)| ok.then_some(i))
        .collect();
    let actor = actors[rng.gen_range(actors.len() as u32) as usize];
    let ks = m.kind_slice(actor);
    assert_eq!(ks.len(), V2_NUM_ACTION_TYPES);
    let kinds: Vec<ActionType> = V2_KIND_ORDER
        .iter()
        .copied()
        .filter(|&k| ks[k as usize])
        .collect();
    let kind = kinds[rng.gen_range(kinds.len() as u32) as usize];
    let am = m.arg(actor, kind);
    let arg = if am.is_empty() {
        0
    } else {
        let args: Vec<usize> = am
            .iter()
            .enumerate()
            .filter_map(|(i, &ok)| ok.then_some(i))
            .collect();
        args[rng.gen_range(args.len() as u32) as usize]
    };
    Factored { actor, kind, arg }
}

#[test]
fn env_v2_playouts_bot_action_replay_and_determinism() {
    let v1_env = Env::new(env_cfg(Opponent::TwoSeat, AgentSeat::Fixed(0), 5));
    assert_eq!(v1_env.ruleset(), Ruleset::V1);

    // Multi-seed factored playouts in TwoSeat and all 3 Bot modes.
    let opponents = [
        Opponent::TwoSeat,
        Opponent::Bot {
            kind: BotKind::Random,
            epsilon: 0.0,
        },
        Opponent::Bot {
            kind: BotKind::Rush,
            epsilon: 0.1,
        },
        Opponent::Bot {
            kind: BotKind::Economy,
            epsilon: 0.1,
        },
    ];
    for (idx, &opp) in opponents.iter().enumerate() {
        for seed in 0..3u64 {
            let mut env = Env::new(v2_env_cfg(11, opp, AgentSeat::SeedParity, 10));
            assert_eq!(env.ruleset(), Ruleset::V2);
            env.reset(seed);
            let mut rng = Rng::new(seed + 100 * idx as u64);
            let mut last = None;
            while !env.is_terminal() {
                let f = sample_v2_factored(&env, &mut rng);
                last = Some(env.step_factored(&f).expect("masked factored action legal"));
            }
            let last = last.unwrap();
            assert!(last.terminated && !last.truncated);
            assert!(env.legal_actions().is_empty());
            assert!(env.masks().actor().iter().all(|&x| !x));

            // Replay JSON must verify cleanly with sim_tools::replay::replay.
            let json = env.replay_json();
            assert!(json.contains("\"v2\""), "v2 replay carries ruleset v2");
            let log = ReplayLog::from_json(&json).expect("parse v2 replay");
            let replayed = replay(&log).expect("v2 replay verifies");
            assert_eq!(&replayed, env.state(), "replayed state matches env");
        }
    }

    // bot_action works under V2 and determinism holds across two identical runs.
    let run_bot = |kind: BotKind, seed: u64| {
        let mut env = Env::new(v2_env_cfg(
            8,
            Opponent::Bot {
                kind: BotKind::Economy,
                epsilon: 0.0,
            },
            AgentSeat::Fixed(0),
            8,
        ));
        env.reset(seed);
        let mut trace = Vec::new();
        while !env.is_terminal() {
            let b = encode_v2_board(env.state(), env.observer());
            let g = encode_v2_globals(env.state(), env.observer());
            let f = env.bot_action(kind).expect("bot_action in v2");
            let out = env.step_factored(&f).expect("bot_action steps");
            trace.push((
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                g.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                f,
                out.reward.to_bits(),
                out.terminated,
            ));
        }
        (trace, env.replay_json())
    };
    for kind in KINDS {
        let a = run_bot(kind, 7);
        let b = run_bot(kind, 7);
        assert_eq!(a, b, "v2 determinism for {kind:?}");
    }
}

// ============================================================================
// 7. V2 No-Cheating + Documented Damage-Preview Exception
// ============================================================================

fn v2_twin(s: &State, v: PlayerId, rng: &mut Rng) -> State {
    let o = 1 - v;
    let mut t = s.clone();
    t.rng = Rng::new(rng.next_u32() as u64 + 1);
    t.next_unit_id += 1 + rng.gen_range(5);
    t.players[o as usize].stars = rng.gen_range(60);
    t.players[o as usize].kills = rng.gen_range(25);
    t.players[o as usize].lighthouses_discovered = (rng.gen_range(16)) as u8;
    t.players[o as usize].monuments_unlocked = MonumentSet::from_slice(
        &ALL_MONUMENTS
            .iter()
            .copied()
            .filter(|_| rng.gen_range(2) == 1)
            .collect::<Vec<_>>(),
    );
    t.players[o as usize].monuments_placed = MonumentSet::from_slice(
        &ALL_MONUMENTS
            .iter()
            .copied()
            .filter(|_| rng.gen_range(2) == 1)
            .collect::<Vec<_>>(),
    );

    // Swap opponent techs that do NOT change defense bonus (avoid Archery/Climbing) and preserve tier sum.
    let mut o_techs = t.players[o as usize].techs;
    let swap_pairs = [
        (Tech::Hunting, Tech::Riding),
        (Tech::Forestry, Tech::Farming),
        (Tech::Strategy, Tech::Mining),
        (Tech::Roads, Tech::FreeSpirit),
        (Tech::Trade, Tech::Chivalry),
        (Tech::Mathematics, Tech::Construction),
    ];
    for (a, b) in swap_pairs {
        let has_a = o_techs.contains(a);
        let has_b = o_techs.contains(b);
        if has_a != has_b && rng.gen_range(2) == 1 {
            // Rebuild TechSet swapping a and b (same tier, so opponent's public score is unchanged).
            let mut next = TechSet::new();
            for &tech in techs(Ruleset::V2) {
                let keep = if tech == a {
                    has_b
                } else if tech == b {
                    has_a
                } else {
                    o_techs.contains(tech)
                };
                if keep {
                    next.insert(tech);
                }
            }
            o_techs = next;
        }
    }
    t.players[o as usize].techs = o_techs;

    // Enemy city hidden fields: workshop, border_growth_used, connected_to (and wall only if no
    // enemy unit sits on the city, since wall affects attack_preview when attackable!).
    let o_cap = s
        .cities
        .iter()
        .find(|c| c.owner == o && c.is_capital)
        .map(|c| c.id);
    for c in t.cities.iter_mut().filter(|c| c.owner == o) {
        c.workshop = rng.gen_range(2) == 1;
        c.border_growth_used = rng.gen_range(2) == 1;
        if !c.is_capital {
            c.connected_to = if rng.gen_range(2) == 1 { o_cap } else { None };
        }
        if s.tile(c.pos).unit.is_none() {
            c.wall = rng.gen_range(2) == 1;
        }
    }

    // Enemy unit hidden internals: kills, origin, last_dir, promotion_available, phase.
    let phases = [Phase::Fresh, Phase::Moved, Phase::Attacked, Phase::Done];
    let dirs = [
        None,
        Some(Dir { dr: -1, dc: 0 }),
        Some(Dir { dr: 1, dc: 1 }),
    ];
    for u in t.units.iter_mut().filter(|u| u.owner == o) {
        u.kills = rng.gen_range(5);
        u.origin = None;
        u.last_dir = dirs[rng.gen_range(dirs.len() as u32) as usize];
        u.promotion_available =
            rng.gen_range(2) == 1 && !u.veteran && u.kind != UnitType::SuperUnit;
        u.phase = phases[rng.gen_range(phases.len() as u32) as usize];
    }

    // Unexplored empty tiles: mutate v2 road, building, building_level, lighthouse (keep monument
    // unchanged if owned by opponent because monuments add 400 score! Mutate monument only on
    // unowned unexplored tiles where it does not affect either player's score).
    for q in all_pos(s.size) {
        if s.is_explored(v, q) {
            continue;
        }
        let i = ti(s.size, q);
        let tile = &mut t.tiles[i];
        if tile.unit.is_some() || tile.city.is_some() {
            continue;
        }
        tile.road = rng.gen_range(2) == 1;
        tile.lighthouse = rng.gen_range(2) == 1;
        tile.building = match rng.gen_range(5) {
            0 => None,
            1 => Some(Building::Sawmill),
            2 => Some(Building::Windmill),
            3 => Some(Building::Forge),
            _ => Some(Building::Market),
        };
        tile.building_level = if tile.building.is_some() {
            rng.gen_range(9) as u8
        } else {
            0
        };
        if tile.owner_city.is_none() {
            tile.monument = match rng.gen_range(4) {
                0 => None,
                1 => Some(Monument::ParkOfFortune),
                2 => Some(Monument::GateOfPower),
                _ => Some(Monument::EyeOfGod),
            };
        }
    }

    assert_eq!(
        score(&t, o),
        score(s, o),
        "v2 twin changed o's public score"
    );
    assert_eq!(score(&t, v), score(s, v), "v2 twin changed v's score");
    t
}

fn flatten_v2_masks(s: &State) -> Vec<bool> {
    let m = ActionMasks::new(s, &legal_actions(s));
    let mut out = m.actor().to_vec();
    for a in 0..num_actors(s.size) {
        out.extend_from_slice(m.kind_slice(a));
        for &k in action_types(Ruleset::V2) {
            out.extend_from_slice(m.arg(a, k));
        }
    }
    out
}

#[test]
fn v2_no_cheating_twins_encode_identically() {
    let mut rng = Rng::new(2026);
    let mut checked = 0usize;
    for seed in 0..8u64 {
        let mut s = new_game_from_seed(seed, &MapParams::default(), &v2_cfg(10));
        let mut prng = Rng::new(seed + 77);
        let mut step = 0;
        while !s.is_terminal() {
            if step % 5 == 0 {
                for v in [0u8, 1] {
                    let b: Vec<u32> = encode_v2_board(&s, v).iter().map(|x| x.to_bits()).collect();
                    let g: Vec<u32> = encode_v2_globals(&s, v)
                        .iter()
                        .map(|x| x.to_bits())
                        .collect();
                    let m = (s.current == v && !s.is_terminal()).then(|| flatten_v2_masks(&s));
                    for _ in 0..2 {
                        let tw = v2_twin(&s, v, &mut rng);
                        let tb: Vec<u32> = encode_v2_board(&tw, v)
                            .iter()
                            .map(|x| x.to_bits())
                            .collect();
                        let tg: Vec<u32> = encode_v2_globals(&tw, v)
                            .iter()
                            .map(|x| x.to_bits())
                            .collect();
                        assert_eq!(tb, b, "v2 board leaked seed {seed} step {step} viewer {v}");
                        assert_eq!(
                            tg, g,
                            "v2 globals leaked seed {seed} step {step} viewer {v}"
                        );
                        if let Some(m) = &m {
                            assert_eq!(
                                legal_actions(&tw),
                                legal_actions(&s),
                                "v2 twin changed legal_actions"
                            );
                            assert_eq!(
                                &flatten_v2_masks(&tw),
                                m,
                                "v2 masks leaked seed {seed} step {step} viewer {v}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
            let legal = legal_actions(&s);
            let a = legal[random_index(&legal, &mut prng)];
            apply(&mut s, &a).unwrap();
            step += 1;
        }
    }
    assert!(checked > 200, "too few v2 twins checked: {checked}");
}

#[test]
fn hidden_enemy_archery_changes_preview_only_when_attackable() {
    // Compare two v2 states with equal public scores: enemy has Forestry (tier 2) vs Archery (tier 2)
    // and stands on an explored forest tile.
    let make = |unit_pos: Pos, enemy_tech: Tech| {
        StateBuilder::new(6)
            .config(v2_cfg(30))
            .explore_all()
            .terrain(p(2, 3), Terrain::Forest)
            .techs(1, &[enemy_tech])
            .unit(0, UnitType::Swordsman, unit_pos)
            .unit(1, UnitType::Warrior, p(2, 3))
            .build()
    };

    // Case A: adjacent -> attackable! Preview channels MUST reflect hidden Archery.
    let a_no = make(p(2, 2), Tech::Forestry);
    let a_yes = make(p(2, 2), Tech::Archery);
    assert_eq!(score(&a_no, 1), score(&a_yes, 1));
    let ba_no = encode_v2_board(&a_no, 0);
    let ba_yes = encode_v2_board(&a_yes, 0);
    assert_approx(
        ba_no[bi(6, ch2::PREV_TARGET_DEF_BONUS, p(2, 3))],
        0.25,
        "1.0/4",
    );
    assert_approx(
        ba_yes[bi(6, ch2::PREV_TARGET_DEF_BONUS, p(2, 3))],
        0.375,
        "1.5/4",
    );
    assert!(
        ba_yes[bi(6, ch2::PREV_TARGET_MAX_DMG, p(2, 3))]
            < ba_no[bi(6, ch2::PREV_TARGET_MAX_DMG, p(2, 3))],
        "Archery reduces target_max_dmg"
    );
    assert!(
        ba_yes[bi(6, ch2::PREV_TARGET_MIN_RETAL, p(2, 3))]
            > ba_no[bi(6, ch2::PREV_TARGET_MIN_RETAL, p(2, 3))],
        "Archery increases target_min_retal"
    );
    assert!(
        ba_yes[bi(6, ch2::PREV_ATTACKER_MAX_DMG, p(2, 2))]
            < ba_no[bi(6, ch2::PREV_ATTACKER_MAX_DMG, p(2, 2))],
        "Archery reduces attacker_max_dmg"
    );
    // Globals remain identical.
    assert_eq!(
        encode_v2_globals(&a_no, 0)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        encode_v2_globals(&a_yes, 0)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );

    // Case B: distance 2 -> NOT attackable by Warrior! ALL 64 channels and 33 globals MUST be bit-identical.
    let b_no = make(p(2, 1), Tech::Forestry);
    let b_yes = make(p(2, 1), Tech::Archery);
    assert_eq!(score(&b_no, 1), score(&b_yes, 1));
    assert_eq!(
        encode_v2_board(&b_no, 0)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        encode_v2_board(&b_yes, 0)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "hidden enemy Archery must NOT leak when enemy is out of attack range"
    );
    assert_eq!(
        encode_v2_globals(&b_no, 0)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        encode_v2_globals(&b_yes, 0)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
}
