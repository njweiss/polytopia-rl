//! Milestone 8: no-cheating metamorphic property (docs/MILESTONE_8 "PlayerView visibility";
//! "Tests" item 1). For many playout states and each viewer, build twins that differ only in
//! information the viewer must not have, then require every `PlayerView` method and every bot
//! (equal bot rngs, several epsilons) to give equal results on the original and the twin.
//!
//! Hidden edits (all relative to the viewer `v`, opponent `o`):
//! - the state's rng and `next_unit_id`;
//! - `o`'s stars, and `o`'s techs (replaced by a random set with the same total tier, so `o`'s
//!   public score is unchanged);
//! - every `o` unit's kills, origin (none or an `o` city) and last direction;
//! - every `o` city's population, workshop, wall and border-growth flag;
//! - unexplored tiles: terrain, resource, improvement and village flag (tiles without unit or
//!   city, and not in `v`'s territory); enemy units removed from or added to "deep" unexplored
//!   tiles; sometimes a new `o` city on a deep unexplored tile.
//!
//! Two kinds of hidden information are public through other channels and are handled so the
//! twin stays indistinguishable:
//! - Score is public (spec "Score"), and `o`'s units, cities and population count in it. After
//!   the edits `o`'s explored mask (itself hidden) is grown or shrunk so `o`'s score is the same.
//! - Zone of Control: an enemy unit next to an explored tile changes `reachable_tiles` and the
//!   legal moves even when its own tile is unexplored (rules of milestone 3). Units are therefore
//!   only added or removed on "deep" tiles, whose eight neighbors are all unexplored too
//!   (docs/QUESTIONS.md).

mod common;

use bots::{make_bot, BotKind};
use common::*;
use sim_core::data::{UnitType, ALL_TECHS};
use sim_core::grid::Pos;
use sim_core::map::{Resource, Terrain};
use sim_core::rng::Rng;
use sim_core::score::score;
use sim_core::state::{
    City, CityId, Dir, Improvement, Phase, PlayerId, State, TechSet, Unit, UnitId,
};
use sim_core::view::PlayerView;

fn idx(s: &State, q: Pos) -> usize {
    q.row as usize * s.size as usize + q.col as usize
}

fn coin(rng: &mut Rng, p: f32) -> bool {
    rng.gen_f32() < p
}

fn pick<T: Copy>(rng: &mut Rng, xs: &[T]) -> T {
    xs[rng.gen_range(xs.len() as u32) as usize]
}

/// Unexplored for `v`, and so are all of its in-bounds neighbors.
fn deep(s: &State, v: PlayerId, q: Pos) -> bool {
    all_pos(s.size)
        .into_iter()
        .filter(|&n| cheb(n, q) <= 1)
        .all(|n| !s.is_explored(v, n))
}

fn techs_with_tier_sum(s: &State, sum: u32, rng: &mut Rng) -> TechSet {
    let options: Vec<u32> = (0u32..512)
        .filter(|mask| {
            ALL_TECHS
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, &t)| s.tables.tech(t).tier)
                .sum::<u32>()
                == sum
        })
        .collect();
    let mask = pick(rng, &options);
    let techs: Vec<_> = ALL_TECHS
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1 << i) != 0)
        .map(|(_, &t)| t)
        .collect();
    TechSet::from_slice(&techs)
}

/// A twin of `s` that differs only in what `v` must not know. Draws fresh edits up to 8 times
/// when `o`'s score cannot be restored through its explored mask; `None` if none worked.
fn twin(s: &State, v: PlayerId, rng: &mut Rng) -> Option<State> {
    (0..8).find_map(|_| try_twin(s, v, rng))
}

fn try_twin(s: &State, v: PlayerId, rng: &mut Rng) -> Option<State> {
    let o = 1 - v;
    let mut t = s.clone();
    t.rng = Rng::new(rng.next_u32() as u64 + 1);
    t.next_unit_id += 1 + rng.gen_range(5);

    t.players[o as usize].stars = rng.gen_range(60);
    let tier_sum: u32 = ALL_TECHS
        .iter()
        .filter(|&&x| s.players[o as usize].techs.contains(x))
        .map(|&x| s.tables.tech(x).tier)
        .sum();
    t.players[o as usize].techs = techs_with_tier_sum(s, tier_sum, rng);

    let o_cities: Vec<CityId> = s
        .cities
        .iter()
        .filter(|c| c.owner == o)
        .map(|c| c.id)
        .collect();
    let dirs: Vec<Option<Dir>> = std::iter::once(None)
        .chain(
            [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
            .into_iter()
            .map(|(dr, dc)| Some(Dir { dr, dc })),
        )
        .collect();
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
        c.population += rng.gen_range(4);
        c.workshop = coin(rng, 0.5);
        c.wall = coin(rng, 0.5);
        c.border_growth_used = coin(rng, 0.5);
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
        None,
        None,
        Some(Improvement::Farm),
        Some(Improvement::Mine),
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
            // Not reachable in play: a player's territory is always explored.
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
        }
        if !deep(s, v, q) {
            continue;
        }
        match tile.unit {
            Some(id) if s.unit(id).is_some_and(|u| u.owner == o) && coin(rng, 0.4) => {
                t.units.retain(|u| u.id != id);
                t.tiles[i].unit = None;
            }
            None if added_units < 2 && coin(rng, 0.1) => {
                added_units += 1;
                let id = UnitId(t.next_unit_id);
                t.next_unit_id += 1;
                t.units.push(Unit {
                    id,
                    kind: UnitType::Warrior,
                    owner: o,
                    pos: q,
                    health: 10,
                    max_health: 10,
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
            });
            t.tiles[i].city = Some(id);
            t.tiles[i].village = false;
        }
    }

    // Restore o's public score through o's hidden explored mask.
    let delta = score(&t, o) as i64 - score(s, o) as i64;
    assert_eq!(delta % 5, 0, "every score weight is a multiple of 5");
    let weight = s.config.score_weights.exploration_per_tile as i64;
    assert_eq!(delta % weight, 0);
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
    assert_eq!(
        score(&t, v),
        score(s, v),
        "the viewer's score never depends on hidden data"
    );
    Some(t)
}

/// Every `PlayerView` method gives the same answer on both states.
fn assert_views_equal(a: &State, b: &State, v: PlayerId, what: &str) {
    let (x, y) = (PlayerView::new(a, v), PlayerView::new(b, v));
    assert_eq!(x.player(), y.player(), "{what}");
    assert_eq!(x.size(), y.size(), "{what}");
    assert_eq!(x.turn(), y.turn(), "{what}");
    assert_eq!(x.turn_cap(), y.turn_cap(), "{what}");
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

fn states() -> Vec<State> {
    let mut out = Vec::new();
    for seed in 0..5 {
        out.extend(random_playout_states(seed, &cfg(20), 1000 + seed, 9));
    }
    for (seed, kinds, eps) in [
        (5, [BotKind::Rush, BotKind::Economy], 0.1),
        (6, [BotKind::Economy, BotKind::Rush], 0.1),
        (7, [BotKind::Rush, BotKind::Random], 0.0),
        (8, [BotKind::Random, BotKind::Economy], 0.0),
        (9, [BotKind::Rush, BotKind::Rush], 0.2),
    ] {
        out.extend(bot_playout_states(seed, kinds, eps, &cfg(30), 6));
    }
    out
}

#[test]
fn views_and_bots_ignore_hidden_information() {
    let all = states();
    assert!(all.len() > 200, "only {} states", all.len());
    let mut rng = Rng::new(0x5eed);
    let (mut twins, mut skipped, mut unit_edits) = (0, 0, 0);
    for (n, s) in all.iter().enumerate() {
        for v in 0..2u8 {
            for k in 0..2 {
                let Some(t) = twin(s, v, &mut rng) else {
                    skipped += 1;
                    continue;
                };
                twins += 1;
                if t.units.len() != s.units.len() || t.cities.len() != s.cities.len() {
                    unit_edits += 1;
                }
                let what = format!("state {n} viewer {v} twin {k}");
                assert_views_equal(s, &t, v, &what);
                if v == s.current {
                    assert_bots_equal(s, &t, &what);
                }
            }
        }
    }
    assert!(
        skipped * 10 <= twins,
        "too many twins skipped: {skipped} of {twins}"
    );
    assert!(
        unit_edits > 20,
        "only {unit_edits} twins added or removed units or cities"
    );
}

#[test]
fn twins_really_differ_in_hidden_tile_contents() {
    // Guard for the property above: unexplored tiles do change in the twins.
    let all = states();
    let mut rng = Rng::new(7);
    let mut changed = 0;
    for s in all.iter().take(60) {
        if let Some(t) = twin(s, s.current, &mut rng) {
            changed += s.tiles.iter().zip(&t.tiles).filter(|(a, b)| a != b).count();
        }
    }
    assert!(changed > 500, "only {changed} hidden tiles changed");
}
