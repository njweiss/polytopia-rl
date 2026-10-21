//! Property tests and random playouts (docs/MILESTONE_3 "Tests" items 5-7; spec "Invariants and
//! test plan"). Games start from `new_game_from_seed` and from random scripted boards
//! (`common::random_board`, which has every unit type, cities, walls and partial fog). Actions are
//! chosen by the test's own RNG. Every check is an independent replay of the brief's rules.

mod common;

use common::*;
use sim_core::actions::{apply, legal_actions, Action, ActionError};
use sim_core::data::{Skill, Tech};
use sim_core::grid::Pos;
use sim_core::map::{MapParams, Terrain};
use sim_core::rng::Rng;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Dir, Phase, State, UnitId};

fn seeded_game(seed: u64) -> State {
    let cfg = Config {
        full_observability: seed % 5 == 4,
        turn_cap: 50,
    };
    new_game_from_seed(seed, &MapParams::default(), &cfg)
}

fn actor(a: &Action) -> Option<UnitId> {
    match *a {
        Action::Move { unit, .. }
        | Action::Attack { unit, .. }
        | Action::Recover { unit }
        | Action::Promote { unit } => Some(unit),
        Action::EndTurn => None,
    }
}

/// Random actions, mostly not legal: unit ids up to two past the last one, positions up to two
/// past the board edge.
fn junk(rng: &mut Rng, s: &State) -> Action {
    let unit = UnitId(rng.gen_range(s.next_unit_id + 2));
    let n = s.size as u32 + 2;
    let pos = Pos::new(rng.gen_range(n) as u8, rng.gen_range(n) as u8);
    match rng.gen_range(5) {
        0 => Action::Move { unit, to: pos },
        1 => Action::Attack { unit, target: pos },
        2 => Action::Recover { unit },
        3 => Action::Promote { unit },
        _ => Action::EndTurn,
    }
}

// ---------- the step checker ----------

/// Checks `next = apply(prev, a)` against the brief's rules and the spec invariants.
fn check_step(prev: &State, a: &Action, next: &State, ctx: &str) {
    // Things nothing in this milestone writes.
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

    if *a == Action::EndTurn {
        assert_eq!(next.current, 1 - prev.current, "{ctx}");
        let turn = prev.turn + u32::from(next.current == 0);
        assert_eq!(next.turn, turn, "{ctx}: turn counter");
        assert_eq!(next.units.len(), prev.units.len(), "{ctx}");
        for (u0, u1) in prev.units.iter().zip(&next.units) {
            let mut expect = u0.clone();
            if u0.owner == next.current {
                expect.phase = Phase::Fresh;
            }
            assert_eq!(
                *u1, expect,
                "{ctx}: EndTurn changes only the new player's phases"
            );
        }
        assert_eq!(next.tiles, prev.tiles, "{ctx}");
        assert_eq!(next.players, prev.players, "{ctx}");
        return;
    }

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
        Action::EndTurn => unreachable!(),
    }
}

/// Plays `steps` actions with the test's RNG; at each step compares `legal_actions` with the
/// brute force, optionally probes `apply` on every legal action and on junk, and checks the step.
fn play(mut s: State, rng: &mut Rng, steps: usize, probe: bool, ctx: &str) -> (State, Vec<Action>) {
    let mut log = Vec::new();
    for step in 0..steps {
        let ctx = format!("{ctx} step {step}");
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
        if a == Action::EndTurn {
            assert!(
                s.units
                    .iter()
                    .filter(|u| u.owner == s.current)
                    .all(|u| u.phase == Phase::Fresh),
                "{ctx}"
            );
        }
        log.push(a);
    }
    (s, log)
}

// ---------- tests ----------

#[test]
fn legality_matches_brute_force_and_apply_agrees() {
    let mut rng = Rng::new(0xA11);
    for seed in 0..60 {
        play(
            seeded_game(seed),
            &mut rng,
            150,
            true,
            &format!("game {seed}"),
        );
    }
    for seed in 0..150 {
        play(
            random_board(seed),
            &mut rng,
            150,
            true,
            &format!("board {seed}"),
        );
    }
}

#[test]
fn random_playouts_keep_every_invariant() {
    let mut rng = Rng::new(0xB0B);
    let mut attacks_seen = 0;
    for seed in 0..300 {
        let (_, log) = play(
            seeded_game(seed),
            &mut rng,
            200,
            false,
            &format!("game {seed}"),
        );
        attacks_seen += log
            .iter()
            .filter(|a| matches!(a, Action::Attack { .. }))
            .count();
    }
    let mut kills_seen = 0;
    for seed in 1000..1300 {
        let (s, log) = play(
            random_board(seed),
            &mut rng,
            200,
            false,
            &format!("board {seed}"),
        );
        attacks_seen += log
            .iter()
            .filter(|a| matches!(a, Action::Attack { .. }))
            .count();
        kills_seen += (random_board(seed).units.len() - s.units.len()) as u32;
    }
    // Sanity: the playouts actually fight (not a rule; guards against a vacuous run).
    assert!(attacks_seen > 300, "only {attacks_seen} attacks");
    assert!(kills_seen > 100, "only {kills_seen} kills");
}

#[test]
fn same_seed_and_actions_replay_identically() {
    let mut rng = Rng::new(0xC0DE);
    for seed in 0..40 {
        let game = || {
            if seed % 2 == 0 {
                seeded_game(seed)
            } else {
                random_board(seed)
            }
        };
        let mut states = vec![game()];
        let (_, log) = {
            let mut s = game();
            let mut log = Vec::new();
            for _ in 0..200 {
                let a = pick(&mut rng, &legal_actions(&s));
                apply(&mut s, &a).unwrap();
                states.push(s.clone());
                log.push(a);
            }
            (s, log)
        };
        let mut r = game();
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
        let mut s = if seed % 2 == 0 {
            seeded_game(seed)
        } else {
            random_board(seed)
        };
        for _ in 0..60 {
            let a = pick(&mut rng, &legal_actions(&s));
            apply(&mut s, &a).unwrap();
        }
        let mut c = s.clone();
        for step in 0..120 {
            let legal = legal_actions(&s);
            assert_eq!(legal, legal_actions(&c), "seed {seed} step {step}");
            let a = pick(&mut rng, &legal);
            assert_eq!(apply(&mut s, &a), apply(&mut c, &a));
            assert_eq!(s, c, "seed {seed} step {step}");
        }
    }
}
