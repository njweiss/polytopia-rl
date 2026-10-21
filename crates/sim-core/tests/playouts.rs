//! Property tests and random playouts (docs/MILESTONE_3 "Tests" items 5-7, docs/MILESTONE_4
//! "Tests" item 6, docs/MILESTONE_5 "Tests" item 6, docs/MILESTONE_6 "Tests" items 7-8; spec
//! "Invariants and test plan"). Games start from `new_game_from_seed`, from
//! boosted new games (extra stars, techs and pre-grown capitals so levels 3 to 5 are visited), from
//! random scripted boards (`common::random_board`: every unit type, cities, walls, partial fog)
//! and from random economy boards (`common::random_economy_board`: resources, improvements,
//! levels, pending rewards, unit origins), and for milestone 6 from capture scenarios (new games
//! with a small turn cap and extra units on villages and next to the enemy capital). Actions are
//! chosen by the test's own RNG. Every check is an independent replay of the briefs' rules.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Reward, Skill, Tech, ALL_TECHS, ALL_UNITS};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Terrain};
use sim_core::rng::Rng;
use sim_core::score::{score, score_breakdown};
use sim_core::setup::new_game_from_seed;
use sim_core::state::{CityId, Config, Dir, Outcome, Phase, State, Unit, UnitId};

fn seeded_game(seed: u64) -> State {
    let cfg = Config {
        full_observability: seed % 5 == 4,
        ..Config::default()
    };
    new_game_from_seed(seed, &MapParams::default(), &cfg)
}

/// A new game where both capitals start one population short of the next level (level 1 to 5,
/// so the Explorer and Super Unit menus come up often) and both players have extra stars; every
/// other game also has every tech.
fn boosted_game(seed: u64) -> State {
    let mut s = seeded_game(seed);
    let level = 1 + (seed % 5) as u32;
    for c in &mut s.cities {
        c.level = level;
        c.population = exp_cum(level + 1) - 1;
    }
    for pl in &mut s.players {
        pl.stars = 10 + (seed % 20) as u32;
        if seed.is_multiple_of(2) {
            pl.techs = sim_core::state::TechSet::from_slice(&ALL_TECHS);
        }
    }
    s
}

/// Milestone 6 scenario: a new game with `turn_cap` 10 to 30 (by seed) where every free village
/// holds a Fresh Warrior (players alternate in row-major village order, starting with
/// `seed % 2`) and each player has a Fresh Rider on the first free non-mountain neighbor
/// (row-major) of the enemy capital. When `seed` is a multiple of 3 the Rider stands on the enemy capital
/// itself instead, and the capital's own Warrior is moved to that neighbor. Added units have no
/// origin; their owner explores what they see (radius 1, or 2 on a mountain).
fn capture_scenario(seed: u64) -> State {
    let cfg = Config {
        full_observability: seed % 5 == 4,
        turn_cap: 10 + (seed % 21) as u32,
        ..Config::default()
    };
    let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg);
    let place = |s: &mut State, owner: u8, kind, at: Pos| {
        let health = s.tables.unit(kind).health;
        let id = UnitId(s.next_unit_id);
        s.next_unit_id += 1;
        s.units.push(Unit {
            id,
            kind,
            owner,
            pos: at,
            health,
            max_health: health,
            origin: None,
            kills: 0,
            veteran: false,
            promotion_available: false,
            phase: Phase::Fresh,
            last_dir: None,
        });
        tile_mut(s, at).unit = Some(id);
        for q in vision(s, at) {
            let i = idx(s, q);
            s.players[owner as usize].explored[i] = true;
        }
    };
    let villages: Vec<Pos> = all_pos(s.size)
        .into_iter()
        .filter(|&q| s.tile(q).village && s.tile(q).unit.is_none())
        .collect();
    for (k, &v) in villages.iter().enumerate() {
        let owner = ((seed as usize + k) % 2) as u8;
        place(&mut s, owner, sim_core::data::UnitType::Warrior, v);
    }
    for pl in 0..2u8 {
        let capital = s.cities[1 - pl as usize].pos;
        let spot = nbrs(s.size, capital)
            .into_iter()
            .find(|&q| s.tile(q).unit.is_none() && s.tile(q).terrain != Terrain::Mountain);
        let Some(q) = spot else { continue };
        if seed.is_multiple_of(3) {
            let home = s.tile(capital).unit.unwrap();
            s.units.iter_mut().find(|u| u.id == home).unwrap().pos = q;
            tile_mut(&mut s, q).unit = Some(home);
            tile_mut(&mut s, capital).unit = None;
            place(&mut s, pl, sim_core::data::UnitType::Rider, capital);
        } else {
            place(&mut s, pl, sim_core::data::UnitType::Rider, q);
        }
    }
    s
}

fn actor(a: &Action) -> Option<UnitId> {
    match *a {
        Action::Move { unit, .. }
        | Action::Attack { unit, .. }
        | Action::Capture { unit }
        | Action::Recover { unit }
        | Action::Promote { unit } => Some(unit),
        _ => None,
    }
}

/// In `Reward` declaration order (the brief's enum), so `reward as usize` indexes it.
const REWARDS: [Reward; 8] = [
    Reward::Workshop,
    Reward::Explorer,
    Reward::CityWall,
    Reward::Resources,
    Reward::PopulationGrowth,
    Reward::BorderGrowth,
    Reward::Park,
    Reward::SuperUnit,
];

/// Random actions, mostly not legal: unit and city ids up to two past the last one, positions up
/// to two past the board edge, any unit type, site action, tech or reward.
fn junk(rng: &mut Rng, s: &State) -> Action {
    let unit = UnitId(rng.gen_range(s.next_unit_id + 2));
    let city = CityId(rng.gen_range(s.cities.len() as u32 + 2));
    let n = s.size as u32 + 2;
    let pos = Pos::new(rng.gen_range(n) as u8, rng.gen_range(n) as u8);
    match rng.gen_range(10) {
        0 => Action::Move { unit, to: pos },
        1 => Action::Attack { unit, target: pos },
        2 => Action::Recover { unit },
        3 => Action::Promote { unit },
        4 => Action::Train {
            city,
            kind: ALL_UNITS[rng.gen_range(5) as usize],
        },
        5 => Action::Site {
            at: pos,
            action: EXP_SITE_ACTIONS[rng.gen_range(6) as usize],
        },
        6 => Action::Research {
            tech: ALL_TECHS[rng.gen_range(9) as usize],
        },
        7 => Action::ChooseReward {
            city,
            reward: REWARDS[rng.gen_range(8) as usize],
        },
        8 => Action::Capture { unit },
        _ => Action::EndTurn,
    }
}

// ---------- the step checker ----------

/// Milestone 6 invariants on `next` (brief "Tests" item 7), and what may change from `prev`.
fn check_m6_invariants(prev: &State, a: &Action, next: &State, ctx: &str) {
    // Cities are dense by id, and `tile.city` agrees with `cities[].pos`.
    for (i, c) in next.cities.iter().enumerate() {
        assert_eq!(c.id, CityId(i as u32), "{ctx}: city ids not dense");
        assert_eq!(
            next.tile(c.pos).city,
            Some(c.id),
            "{ctx}: tile.city of {c:?}"
        );
    }
    let city_tiles = next.tiles.iter().filter(|t| t.city.is_some()).count();
    assert_eq!(city_tiles, next.cities.len(), "{ctx}: stray tile.city");
    for t in &next.tiles {
        assert!(
            !(t.village && t.city.is_some()),
            "{ctx}: village with a city"
        );
    }
    // Cities are never removed; a city's owner changes only by capturing it, `is_capital` never
    // changes, and new cities (village captures) are not capitals.
    assert!(
        next.cities.len() >= prev.cities.len(),
        "{ctx}: city removed"
    );
    let captured = match *a {
        Action::Capture { unit } => prev.tile(prev.unit(unit).unwrap().pos).city,
        _ => None,
    };
    for (c0, c1) in prev.cities.iter().zip(&next.cities) {
        assert_eq!(c1.is_capital, c0.is_capital, "{ctx}: is_capital changed");
        if Some(c0.id) != captured {
            assert_eq!(c1.owner, c0.owner, "{ctx}: owner changed without capture");
        }
    }
    for c in &next.cities[prev.cities.len()..] {
        assert!(!c.is_capital, "{ctx}: new capital {c:?}");
    }
    let capitals = |s: &State| s.cities.iter().filter(|c| c.is_capital).count();
    assert_eq!(capitals(next), capitals(prev), "{ctx}: capitals");
    // Every origin names an existing city owned by the unit's owner (orphans have `None`).
    for u in &next.units {
        if let Some(o) = u.origin {
            let c = next.cities.iter().find(|c| c.id == o);
            assert!(c.is_some(), "{ctx}: {u:?} names a missing city");
            assert_eq!(c.unwrap().owner, u.owner, "{ctx}: {u:?} not orphaned");
        }
    }
    // Score is the breakdown total and the brief's formula.
    for pl in 0..2u8 {
        let b = score_breakdown(next, pl);
        let terms = [
            b.units,
            b.territory,
            b.exploration,
            b.cities,
            b.parks,
            b.techs,
        ];
        assert_eq!(terms, exp_score_terms(next, pl), "{ctx}: breakdown of {pl}");
        assert_eq!(score(next, pl), b.total(), "{ctx}: score of {pl}");
    }
    // The outcome, once set, never changes; the turn never passes the cap.
    if prev.outcome.is_some() {
        assert_eq!(next.outcome, prev.outcome, "{ctx}: outcome changed");
    }
    assert!(
        next.turn <= next.config.turn_cap,
        "{ctx}: turn past the cap"
    );
    if next.outcome.is_none() {
        assert!(
            next.turn < next.config.turn_cap,
            "{ctx}: cap reached without an outcome"
        );
    }
}

/// Checks `next = apply(prev, a)` against the briefs' rules and the spec invariants.
fn check_step(prev: &State, a: &Action, next: &State, ctx: &str) {
    // Invariants for every action: ownership and improvements are written at most once.
    for (t0, t1) in prev.tiles.iter().zip(&next.tiles) {
        if t0.owner_city.is_some() {
            assert_eq!(t1.owner_city, t0.owner_city, "{ctx}: owner_city changed");
        }
        if t0.improvement.is_some() {
            assert_eq!(t1.improvement, t0.improvement, "{ctx}: improvement changed");
        }
    }
    for pl in 0..2 {
        for (e0, e1) in prev.players[pl]
            .explored
            .iter()
            .zip(&next.players[pl].explored)
        {
            assert!(!e0 || *e1, "{ctx}: explored mask of {pl} shrank");
        }
    }
    check_economy_invariants(next, ctx);
    check_m6_invariants(prev, a, next, ctx);

    // Economy actions, Capture and EndTurn: the whole next state is the briefs' replay.
    if actor(a).is_none() || matches!(a, Action::Capture { .. }) {
        assert_eq!(*next, exp_after(prev, a), "{ctx}: state after {a:?}");
        if *a == Action::EndTurn && next.outcome.is_none() {
            assert_eq!(next.current, 1 - prev.current, "{ctx}");
            assert!(
                next.units
                    .iter()
                    .filter(|u| u.owner == next.current)
                    .all(|u| u.phase == Phase::Fresh),
                "{ctx}"
            );
        }
        return;
    }

    // Unit actions (milestone 3). Things no unit action writes.
    assert_eq!(next.size, prev.size, "{ctx}");
    assert_eq!(next.config, prev.config, "{ctx}");
    assert_eq!(next.rng, prev.rng, "{ctx}: nothing draws from the rng");
    assert_eq!(next.cities, prev.cities, "{ctx}: cities untouched");
    assert_eq!(next.next_unit_id, prev.next_unit_id, "{ctx}");
    assert_eq!(next.outcome, None, "{ctx}");
    for i in 0..2 {
        assert_eq!(next.players[i].stars, prev.players[i].stars, "{ctx}: stars");
        assert_eq!(next.players[i].techs, prev.players[i].techs, "{ctx}: techs");
    }
    for (t0, t1) in prev.tiles.iter().zip(&next.tiles) {
        assert_eq!(t1.owner_city, t0.owner_city, "{ctx}: owner_city changed");
        assert_eq!(
            (t1.terrain, t1.resource, t1.improvement, t1.village, t1.city),
            (t0.terrain, t0.resource, t0.improvement, t0.village, t0.city),
            "{ctx}: tile changed"
        );
    }

    // Structure: ascending ids, one unit per tile, occupancy matches positions, health bounds.
    for w in next.units.windows(2) {
        assert!(w[0].id < w[1].id, "{ctx}: units not ascending by id");
    }
    for u in &next.units {
        assert!(u.id.0 < next.next_unit_id, "{ctx}");
        assert_eq!(
            next.tile(u.pos).unit,
            Some(u.id),
            "{ctx}: occupancy of {:?}",
            u.id
        );
        assert!(
            u.health > 0 && u.health <= u.max_health,
            "{ctx}: health {:?}",
            u
        );
    }
    let occupied = next.tiles.iter().filter(|t| t.unit.is_some()).count();
    assert_eq!(occupied, next.units.len(), "{ctx}: stray tile occupancy");

    // Explored masks: exactly the old mask plus what each moved unit sees from its new tile.
    for pl in 0..2u8 {
        let mut want = prev.players[pl as usize].explored.clone();
        if next.config.full_observability {
            want.iter_mut().for_each(|e| *e = true);
        }
        for u in next.units.iter().filter(|u| u.owner == pl) {
            let old = prev.units.iter().find(|o| o.id == u.id).unwrap();
            if old.pos != u.pos {
                for q in vision(next, u.pos) {
                    want[idx(next, q)] = true;
                }
            }
        }
        assert_eq!(
            next.players[pl as usize].explored, want,
            "{ctx}: explored mask of {pl}"
        );
    }

    let find = |s: &State, id: UnitId| s.units.iter().find(|u| u.id == id).cloned();

    assert_eq!(
        (next.turn, next.current),
        (prev.turn, prev.current),
        "{ctx}"
    );
    let id = actor(a).unwrap();
    let me = find(prev, id).unwrap();
    assert_eq!(
        me.owner, prev.current,
        "{ctx}: acted with a non-current unit"
    );

    // Units other than the actor (and an attack's defender) are unchanged.
    let defender = match *a {
        Action::Attack { target, .. } => Some(unit_at(prev, target).unwrap().clone()),
        _ => None,
    };
    for u0 in &prev.units {
        if u0.id == id || Some(u0.id) == defender.as_ref().map(|d| d.id) {
            continue;
        }
        assert_eq!(
            find(next, u0.id).as_ref(),
            Some(u0),
            "{ctx}: bystander changed"
        );
    }
    if let Some(after) = find(next, id) {
        assert!(
            phase_rank(after.phase) >= phase_rank(me.phase),
            "{ctx}: phase regressed {:?} -> {:?}",
            me.phase,
            after.phase
        );
    }
    let stats = prev.tables.unit(me.kind);

    match *a {
        Action::Move { to, .. } => {
            let u = find(next, id).unwrap();
            assert_eq!(u.pos, to, "{ctx}");
            let want_phase = match me.phase {
                Phase::Fresh => Phase::Moved,
                Phase::Attacked => Phase::Done,
                other => panic!("{ctx}: move from {other:?} was accepted"),
            };
            assert_eq!(u.phase, want_phase, "{ctx}");
            let d = Dir {
                dr: sign(to.row as i32 - me.pos.row as i32),
                dc: sign(to.col as i32 - me.pos.col as i32),
            };
            assert_eq!(u.last_dir, Some(d), "{ctx}");
            let mut rest = u.clone();
            (rest.pos, rest.phase, rest.last_dir) = (me.pos, me.phase, me.last_dir);
            assert_eq!(rest, me, "{ctx}: move changed other fields");
        }
        Action::Attack { target, .. } => {
            let d0 = defender.unwrap();
            let (ha, hd) = expected_combat(prev, id, d0.id);
            let att = find(next, id);
            let def = find(next, d0.id);
            assert_eq!(
                att.as_ref().map_or(0, |u| u.health),
                ha,
                "{ctx}: attacker hp"
            );
            assert_eq!(
                def.as_ref().map_or(0, |u| u.health),
                hd,
                "{ctx}: defender hp"
            );
            let promo = |kills: u32, veteran: bool, k| {
                kills >= 3 && !veteran && !prev.tables.unit(k).skills.contains(&Skill::Static)
            };
            if let Some(u) = att {
                let kills = me.kills + u32::from(hd == 0);
                assert_eq!(u.kills, kills, "{ctx}: attacker kills");
                assert_eq!(
                    u.promotion_available,
                    me.promotion_available || promo(kills, me.veteran, me.kind),
                    "{ctx}: attacker promotion"
                );
                assert_eq!(u.phase, Phase::Attacked, "{ctx}");
                let melee = stats.range == 1;
                let blocked = prev.tile(target).terrain == Terrain::Mountain
                    && !has_tech(prev, me.owner, Tech::Climbing);
                let toward = Some(Dir {
                    dr: sign(target.row as i32 - me.pos.row as i32),
                    dc: sign(target.col as i32 - me.pos.col as i32),
                });
                if hd == 0 && melee && !blocked {
                    assert_eq!(u.pos, target, "{ctx}: kill-move");
                    assert_eq!(u.last_dir, toward, "{ctx}");
                } else {
                    assert_eq!(u.pos, me.pos, "{ctx}: no move");
                    let want = if melee { me.last_dir } else { toward };
                    assert_eq!(u.last_dir, want, "{ctx}: last_dir");
                }
            }
            if let Some(u) = def {
                let kills = d0.kills + u32::from(ha == 0);
                assert_eq!(u.kills, kills, "{ctx}: defender kills");
                assert_eq!(
                    u.promotion_available,
                    d0.promotion_available || promo(kills, d0.veteran, d0.kind),
                    "{ctx}: defender promotion"
                );
                assert_eq!(u.pos, d0.pos, "{ctx}");
                assert_eq!(u.phase, d0.phase, "{ctx}");
            }
        }
        Action::Recover { .. } => {
            let u = find(next, id).unwrap();
            let own = prev
                .tile(me.pos)
                .owner_city
                .is_some_and(|c| prev.city(c).owner == me.owner);
            let heal = if own { 4 } else { 2 };
            assert_eq!(u.health, (me.health + heal).min(me.max_health), "{ctx}");
            assert_eq!(u.phase, Phase::Done, "{ctx}");
            assert_eq!(u.pos, me.pos, "{ctx}");
        }
        Action::Promote { .. } => {
            let u = find(next, id).unwrap();
            assert_eq!(u.max_health, me.max_health + 5, "{ctx}");
            assert_eq!(u.health, u.max_health, "{ctx}");
            assert!(u.veteran && !u.promotion_available, "{ctx}");
            assert_eq!(
                (u.phase, u.last_dir, u.pos),
                (me.phase, me.last_dir, me.pos)
            );
        }
        _ => unreachable!(),
    }
}

/// What the playouts exercised (vacuity guards, not rules).
#[derive(Default)]
struct Seen {
    attacks: usize,
    kills: usize,
    trains: usize,
    research: usize,
    sites: [usize; 6],
    rewards: [usize; 8],
    max_level: u32,
    cascades: usize,
    /// Super Unit spawns onto an occupied city tile, and those where the occupant was removed.
    pushes: usize,
    removals: usize,
    /// Milestone 6: village and city captures, and how games ended.
    village_captures: usize,
    city_captures: usize,
    capital_wins: usize,
    cap_endings: usize,
}

/// Plays `steps` actions with the test's RNG; at each step compares `legal_actions` with the
/// brute force, optionally probes `apply` on every legal action and on junk, and checks the step.
fn play(
    s: State,
    rng: &mut Rng,
    steps: usize,
    probe: bool,
    ctx: &str,
    seen: &mut Seen,
) -> (State, Vec<Action>) {
    play_with(s, rng, steps, probe, ctx, seen, pick_weighted)
}

/// Milestone 6 picker: a legal Capture (uniformly among them) with probability 1/2, otherwise
/// `pick_weighted`. Makes captures common enough for the vacuity guards.
fn pick_capture(rng: &mut Rng, legal: &[Action]) -> Action {
    let captures: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| matches!(a, Action::Capture { .. }))
        .collect();
    if !captures.is_empty() && rng.gen_range(2) == 0 {
        return captures[rng.gen_range(captures.len() as u32) as usize];
    }
    pick_weighted(rng, legal)
}

/// `play` with a given action picker.
fn play_with(
    mut s: State,
    rng: &mut Rng,
    steps: usize,
    probe: bool,
    ctx: &str,
    seen: &mut Seen,
    pick: fn(&mut Rng, &[Action]) -> Action,
) -> (State, Vec<Action>) {
    let mut log = Vec::new();
    check_economy_invariants(&s, &format!("{ctx} start"));
    for step in 0..steps {
        let ctx = format!("{ctx} step {step}");
        if s.is_terminal() {
            check_terminal(&s, rng, &ctx);
            break;
        }
        let legal = legal_actions(&s);
        assert_eq!(legal, bf_legal(&s), "{ctx}: legal_actions vs brute force");
        for a in &legal {
            if let Some(id) = actor(a) {
                assert_eq!(
                    s.unit(id).map(|u| u.owner),
                    Some(s.current),
                    "{ctx}: {a:?} for a non-current unit"
                );
            }
        }
        if probe {
            for a in &legal {
                let mut t = s.clone();
                assert_eq!(apply(&mut t, a), Ok(()), "{ctx}: listed {a:?} rejected");
            }
            for _ in 0..8 {
                let a = junk(rng, &s);
                if legal.contains(&a) {
                    continue;
                }
                let mut t = s.clone();
                assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{ctx}: {a:?}");
                assert_eq!(t, s, "{ctx}: rejected {a:?} changed the state");
            }
        }
        let a = pick(rng, &legal);
        let prev = s.clone();
        assert_eq!(apply(&mut s, &a), Ok(()), "{ctx}: {a:?}");
        check_step(&prev, &a, &s, &format!("{ctx} {a:?}"));
        match a {
            Action::Attack { .. } => {
                seen.attacks += 1;
                seen.kills += prev.units.len() - s.units.len();
            }
            Action::Train { .. } => seen.trains += 1,
            Action::Capture { unit } => {
                if prev.tile(prev.unit(unit).unwrap().pos).village {
                    seen.village_captures += 1;
                } else {
                    seen.city_captures += 1;
                }
                if s.is_terminal() {
                    seen.capital_wins += 1;
                }
            }
            Action::EndTurn if s.is_terminal() => seen.cap_endings += 1,
            Action::Research { .. } => seen.research += 1,
            Action::Site { action, .. } => seen.sites[action as usize] += 1,
            Action::ChooseReward { city, reward } => {
                seen.rewards[reward as usize] += 1;
                if reward == Reward::SuperUnit {
                    let at = prev.city(city).pos;
                    if let Some(occ) = prev.tile(at).unit {
                        seen.pushes += 1;
                        seen.removals += usize::from(s.unit(occ).is_none());
                    }
                }
                if s.cities.iter().any(|c| c.id == city && c.pending_reward) {
                    seen.cascades += 1;
                }
            }
            _ => {}
        }
        for c in &s.cities {
            seen.max_level = seen.max_level.max(c.level);
        }
        log.push(a);
    }
    (s, log)
}

/// A terminal state: `legal_actions` is empty, and `apply` rejects every action kind and junk
/// without changing anything.
fn check_terminal(s: &State, rng: &mut Rng, ctx: &str) {
    assert!(
        legal_actions(s).is_empty(),
        "{ctx}: legal actions after the end"
    );
    let mut probes = every_action_kind(s);
    probes.extend((0..8).map(|_| junk(rng, s)));
    for a in probes {
        let mut t = s.clone();
        assert_eq!(apply(&mut t, &a), Err(ActionError::Illegal), "{ctx}: {a:?}");
        assert_eq!(t, *s, "{ctx}: {a:?} after the end changed the state");
    }
}

// ---------- tests ----------

#[test]
fn legality_matches_brute_force_and_apply_agrees() {
    let mut rng = Rng::new(0xA11);
    let seen = &mut Seen::default();
    for seed in 0..20 {
        play(
            seeded_game(seed),
            &mut rng,
            150,
            true,
            &format!("game {seed}"),
            seen,
        );
    }
    for seed in 0..20 {
        let ctx = format!("boosted {seed}");
        play(boosted_game(seed), &mut rng, 150, true, &ctx, seen);
    }
    for seed in 0..50 {
        play(
            random_board(seed),
            &mut rng,
            150,
            true,
            &format!("board {seed}"),
            seen,
        );
    }
    for seed in 0..75 {
        let ctx = format!("economy {seed}");
        play(random_economy_board(seed), &mut rng, 150, true, &ctx, seen);
    }
}

#[test]
fn random_playouts_keep_every_invariant() {
    let mut rng = Rng::new(0xB0B);
    let seen = &mut Seen::default();
    for seed in 0..100 {
        play(
            seeded_game(seed),
            &mut rng,
            200,
            false,
            &format!("game {seed}"),
            seen,
        );
    }
    for seed in 0..100 {
        let ctx = format!("boosted {seed}");
        play(boosted_game(seed), &mut rng, 200, false, &ctx, seen);
    }
    for seed in 1000..1150 {
        play(
            random_board(seed),
            &mut rng,
            200,
            false,
            &format!("board {seed}"),
            seen,
        );
    }
    for seed in 2000..2150 {
        let ctx = format!("economy {seed}");
        play(random_economy_board(seed), &mut rng, 200, false, &ctx, seen);
    }
    // Sanity: the playouts actually fight and build (not rules; guards against a vacuous run).
    assert!(seen.attacks > 2000, "only {} attacks", seen.attacks);
    assert!(seen.kills > 1000, "only {} kills", seen.kills);
    assert!(seen.trains > 1500, "only {} trains", seen.trains);
    assert!(seen.research > 1500, "only {} research", seen.research);
    for (i, n) in seen.sites.iter().enumerate() {
        assert!(
            *n > 150,
            "site action {:?} seen {n} times",
            EXP_SITE_ACTIONS[i]
        );
    }
    for (i, n) in seen.rewards.iter().enumerate() {
        assert!(*n > 30, "reward {:?} chosen {n} times", REWARDS[i]);
    }
    assert!(seen.max_level >= 6, "max level {}", seen.max_level);
    assert!(seen.cascades > 5, "only {} cascades", seen.cascades);
    assert!(seen.pushes > 30, "only {} Super Unit pushes", seen.pushes);
}

/// Milestone 6: from capture scenarios and plain new games with a small cap, every game ends
/// with a valid outcome within `2 * turn_cap * 100` actions, keeping every invariant on the way.
#[test]
fn random_games_always_end_with_a_valid_outcome() {
    let mut rng = Rng::new(0xE2D);
    let seen = &mut Seen::default();
    let mut outcomes = [0usize; 3]; // Win(0), Win(1), Draw
    for seed in 0..90 {
        let s = if seed % 3 == 2 {
            let mut s = seeded_game(seed);
            s.config.turn_cap = 10 + (seed % 11) as u32;
            s
        } else {
            capture_scenario(seed)
        };
        let cap = s.config.turn_cap;
        let bound = 2 * cap as usize * 100;
        let ctx = format!("ending {seed}");
        let probe = seed % 10 == 0;
        let (end, log) = play_with(s, &mut rng, bound + 1, probe, &ctx, seen, pick_capture);
        assert!(
            end.is_terminal(),
            "{ctx}: no outcome after {} actions",
            log.len()
        );
        assert!(log.len() <= bound, "{ctx}: {} actions", log.len());
        match end.outcome.unwrap() {
            Outcome::Win(0) => outcomes[0] += 1,
            Outcome::Win(_) => outcomes[1] += 1,
            Outcome::Draw => outcomes[2] += 1,
        }
        // A game ends either by a capital capture or at the cap (turn == cap, player 0 to move).
        if *log.last().unwrap() == Action::EndTurn {
            assert_eq!((end.turn, end.current), (cap, 0), "{ctx}");
        } else {
            assert!(matches!(log.last(), Some(Action::Capture { .. })), "{ctx}");
            assert!(end.turn < cap, "{ctx}");
        }
    }
    // Sanity: both endings and both winners occur (not rules; guards against a vacuous run).
    assert!(
        seen.capital_wins >= 10,
        "only {} capital wins",
        seen.capital_wins
    );
    assert!(
        seen.cap_endings >= 10,
        "only {} cap endings",
        seen.cap_endings
    );
    assert!(
        seen.village_captures >= 100,
        "{} village captures",
        seen.village_captures
    );
    assert!(
        seen.city_captures >= 15,
        "{} city captures",
        seen.city_captures
    );
    assert!(
        outcomes[0] >= 10 && outcomes[1] >= 10,
        "outcomes {outcomes:?}"
    );
}

/// Milestone 6: same seed + actions give identical states through captures and the cap, and a
/// clone taken mid-game evolves like the original to the end.
#[test]
fn games_replay_identically_through_captures_and_the_cap() {
    let mut rng = Rng::new(0xF1A);
    let mut endings = [0usize; 2]; // by capture, at the cap
    for seed in 0..30 {
        let mut s = capture_scenario(seed);
        let mut states = vec![s.clone()];
        let mut log = Vec::new();
        let bound = 2 * s.config.turn_cap as usize * 100;
        while !s.is_terminal() {
            assert!(
                log.len() < bound,
                "seed {seed}: no outcome after {bound} actions"
            );
            let a = pick_capture(&mut rng, &legal_actions(&s));
            apply(&mut s, &a).unwrap();
            states.push(s.clone());
            log.push(a);
        }
        endings[usize::from(*log.last().unwrap() == Action::EndTurn)] += 1;
        let mut r = capture_scenario(seed);
        assert_eq!(r, states[0]);
        for (i, a) in log.iter().enumerate() {
            apply(&mut r, a).unwrap();
            assert_eq!(r, states[i + 1], "seed {seed} step {i}");
        }
        let mid = log.len() / 2;
        let mut c = states[mid].clone();
        for (i, a) in log[mid..].iter().enumerate() {
            assert_eq!(apply(&mut c, a), Ok(()), "seed {seed} clone step {i}");
        }
        assert_eq!(c, s, "seed {seed}: clone ended elsewhere");
    }
    assert!(endings[0] >= 3 && endings[1] >= 3, "endings {endings:?}");
}

/// Seeded, boosted, scripted or economy start, by `seed % 4`.
fn any_start(seed: u64) -> State {
    match seed % 4 {
        0 => seeded_game(seed),
        1 => boosted_game(seed),
        2 => random_board(seed),
        _ => random_economy_board(seed),
    }
}

#[test]
fn same_seed_and_actions_replay_identically() {
    let mut rng = Rng::new(0xC0DE);
    for seed in 0..40 {
        let mut states = vec![any_start(seed)];
        let (_, log) = {
            let mut s = any_start(seed);
            let mut log = Vec::new();
            for _ in 0..200 {
                if s.is_terminal() {
                    break;
                }
                let a = pick_weighted(&mut rng, &legal_actions(&s));
                apply(&mut s, &a).unwrap();
                states.push(s.clone());
                log.push(a);
            }
            (s, log)
        };
        let mut r = any_start(seed);
        assert_eq!(r, states[0]);
        for (i, a) in log.iter().enumerate() {
            apply(&mut r, a).unwrap();
            assert_eq!(r, states[i + 1], "seed {seed} step {i}");
        }
    }
}

#[test]
fn a_clone_evolves_like_the_original() {
    let mut rng = Rng::new(0xD00D);
    for seed in 0..40 {
        let mut s = any_start(seed);
        for _ in 0..60 {
            if s.is_terminal() {
                break;
            }
            let a = pick_weighted(&mut rng, &legal_actions(&s));
            apply(&mut s, &a).unwrap();
        }
        let mut c = s.clone();
        for step in 0..120 {
            if s.is_terminal() {
                break;
            }
            let legal = legal_actions(&s);
            assert_eq!(legal, legal_actions(&c), "seed {seed} step {step}");
            let a = pick_weighted(&mut rng, &legal);
            assert_eq!(apply(&mut s, &a), apply(&mut c, &a));
            assert_eq!(s, c, "seed {seed} step {step}");
        }
    }
}
