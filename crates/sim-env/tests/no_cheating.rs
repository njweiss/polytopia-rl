//! Milestone 9 "Tests" item 4: no-cheating metamorphic property (the milestone 8 idea, reused).
//! For playout states and each viewer `v`, build twins that differ only in information `v` must
//! not have, then require bit-identical `encode_board` and `encode_globals`, and, when `v` is to
//! move, identical legal lists and masks.
//!
//! Hidden edits (opponent `o = 1 - v`):
//! - the state's rng and `next_unit_id`;
//! - `o`'s stars, and `o`'s techs (a random set with the same total tier, so `o`'s public score
//!   is unchanged);
//! - every `o` unit's kills, origin, last direction, promotion flag and phase (phase is hidden
//!   for enemy units, J2);
//! - every `o` city's workshop, wall and border-growth flags (they change `o`'s income, which
//!   is hidden);
//! - on tiles `v` has not explored: terrain, resource, improvement and village flag (tiles
//!   without a unit or city), and the health and kind (among the 3-star units, so `o`'s score
//!   is unchanged) of `o` units standing there.
//!
//! Units are not added, removed or moved: Zone of Control from a hidden unit can reach `v`'s
//! legal moves (docs/QUESTIONS.md Q11), and unit counts are public through the score.

mod common;

use common::*;
use sim_core::actions::legal_actions;
use sim_core::data::{UnitType, ALL_TECHS};
use sim_core::map::{Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::state::{CityId, Dir, Improvement, Phase, PlayerId, State, TechSet};
use sim_core::view::PlayerView;
use sim_env::actions::{arg_len, num_actors, ActionMasks, ActionType};
use sim_env::obs::{encode_board, encode_globals, obs_spec};

fn coin(rng: &mut Rng) -> bool {
    rng.gen_range(2) == 1
}

fn pick<T: Copy>(rng: &mut Rng, xs: &[T]) -> T {
    xs[rng.gen_range(xs.len() as u32) as usize]
}

fn tier_sum(s: &State, techs: TechSet) -> u32 {
    ALL_TECHS
        .iter()
        .filter(|&&t| techs.contains(t))
        .map(|&t| s.tables.tech(t).tier)
        .sum()
}

fn techs_with_tier_sum(s: &State, sum: u32, rng: &mut Rng) -> TechSet {
    let options: Vec<TechSet> = (0u32..512)
        .map(|mask| {
            let techs: Vec<_> = ALL_TECHS
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, &t)| t)
                .collect();
            TechSet::from_slice(&techs)
        })
        .filter(|&t| tier_sum(s, t) == sum)
        .collect();
    pick(rng, &options)
}

fn twin(s: &State, v: PlayerId, rng: &mut Rng) -> State {
    let o = 1 - v;
    let mut t = s.clone();
    t.rng = Rng::new(rng.next_u32() as u64 + 1);
    t.next_unit_id += 1 + rng.gen_range(5);
    t.players[o as usize].stars = rng.gen_range(60);
    let sum = tier_sum(s, s.players[o as usize].techs);
    t.players[o as usize].techs = techs_with_tier_sum(s, sum, rng);

    let o_cities: Vec<Option<CityId>> = std::iter::once(None)
        .chain(s.cities.iter().filter(|c| c.owner == o).map(|c| Some(c.id)))
        .collect();
    let dirs = [
        None,
        Some(Dir { dr: -1, dc: 0 }),
        Some(Dir { dr: 1, dc: 1 }),
        Some(Dir { dr: 0, dc: -1 }),
    ];
    let phases = [Phase::Fresh, Phase::Moved, Phase::Attacked, Phase::Done];
    let three_star = [UnitType::Rider, UnitType::Archer, UnitType::Defender];
    for i in 0..t.units.len() {
        if t.units[i].owner != o {
            continue;
        }
        let hidden = !s.is_explored(v, t.units[i].pos);
        let u = &mut t.units[i];
        u.kills = rng.gen_range(5);
        u.origin = pick(rng, &o_cities);
        u.last_dir = pick(rng, &dirs);
        u.promotion_available = coin(rng) && !u.veteran && u.kind != UnitType::SuperUnit;
        u.phase = pick(rng, &phases);
        if hidden {
            if three_star.contains(&u.kind) {
                u.kind = pick(rng, &three_star);
                let base = s.tables.unit(u.kind).health;
                let bonus = s.tables.rules().veteran_health_bonus;
                u.max_health = if u.veteran { base + bonus } else { base };
            }
            u.health = 1 + rng.gen_range(u.max_health);
        }
    }
    for c in t.cities.iter_mut().filter(|c| c.owner == o) {
        c.workshop = coin(rng);
        c.wall = coin(rng);
        c.border_growth_used = coin(rng);
    }
    let terrains = [Terrain::Field, Terrain::Forest, Terrain::Mountain];
    let resources = [
        None,
        Some(Resource::Fruit),
        Some(Resource::Crop),
        Some(Resource::Animal),
        Some(Resource::Metal),
    ];
    let improvements = [
        None,
        Some(Improvement::LumberHut),
        Some(Improvement::Farm),
        Some(Improvement::Mine),
    ];
    for q in all_pos(s.size) {
        if s.is_explored(v, q) {
            continue;
        }
        let i = ti(s.size, q);
        let tile = &mut t.tiles[i];
        if tile.unit.is_some() || tile.city.is_some() {
            continue;
        }
        tile.terrain = pick(rng, &terrains);
        tile.resource = pick(rng, &resources);
        tile.improvement = pick(rng, &improvements);
        tile.village = coin(rng);
    }
    assert_eq!(score(&t, o), score(s, o), "twin changed o's public score");
    assert_eq!(score(&t, v), score(s, v), "twin changed v's score");
    t
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn board(s: &State, v: PlayerId) -> Vec<u32> {
    let spec = obs_spec(s.size);
    let mut out = vec![0.0f32; spec.channels * spec.height * spec.width];
    encode_board(&PlayerView::new(s, v), &mut out);
    bits(&out)
}

fn globals(s: &State, v: PlayerId) -> Vec<u32> {
    let mut out = vec![0.0f32; obs_spec(s.size).globals];
    encode_globals(&PlayerView::new(s, v), &mut out);
    bits(&out)
}

/// Every mask entry, flattened.
fn masks(s: &State) -> Vec<bool> {
    let m = ActionMasks::new(s, &legal_actions(s));
    let mut out = m.actor().to_vec();
    for a in 0..num_actors(s.size) {
        let kinds = m.kind(a);
        out.extend_from_slice(&kinds);
        for k in ActionType::ALL {
            let arg = m.arg(a, k);
            assert_eq!(arg.len(), arg_len(s.size, k));
            out.extend_from_slice(arg);
        }
    }
    out
}

fn check(states: &[State], twins_per_state: usize, rng: &mut Rng) -> usize {
    let mut checked = 0;
    for s in states {
        for v in [0u8, 1] {
            let (b, g) = (board(s, v), globals(s, v));
            let viewer_moves = s.current == v && !s.is_terminal();
            let m = viewer_moves.then(|| masks(s));
            for _ in 0..twins_per_state {
                let t = twin(s, v, rng);
                let ctx = format!("turn {} current {} viewer {v}", s.turn, s.current);
                assert!(board(&t, v) == b, "encode_board leaked: {ctx}");
                assert!(globals(&t, v) == g, "encode_globals leaked: {ctx}");
                if let Some(m) = &m {
                    assert_eq!(legal_actions(&t), legal_actions(s), "test twin: {ctx}");
                    assert!(masks(&t) == *m, "masks leaked: {ctx}");
                }
                checked += 1;
            }
        }
    }
    checked
}

#[test]
fn twins_differing_in_hidden_information_encode_identically() {
    let mut rng = Rng::new(99);
    let mut states = Vec::new();
    for seed in 0..12 {
        states.extend(random_playout_states(seed, &cfg(12), seed + 50, 6));
        let a = KINDS[seed as usize % 3];
        states.extend(bot_playout_states(
            seed,
            [a, KINDS[(seed as usize + 2) % 3]],
            0.1,
            &cfg(30),
            6,
        ));
    }
    let n = check(&states, 2, &mut rng);
    assert!(n > 2000, "too few twins: {n}");
}

#[test]
fn full_observability_twins_differing_in_internals_encode_identically() {
    let mut rng = Rng::new(7);
    let full = sim_core::state::Config {
        full_observability: true,
        ..cfg(10)
    };
    let mut states = Vec::new();
    for seed in 0..8 {
        states.extend(random_playout_states(seed, &full, seed + 70, 5));
    }
    let n = check(&states, 2, &mut rng);
    assert!(n > 400, "too few twins: {n}");
}
