//! Milestone 19 playout tests (docs/MILESTONE_19 "Tests" item 5): multi-seed randomized v2
//! playouts exercising training of all 3 new units (`Swordsman`, `Catapult`, `Knight`), `Stiff`,
//! `Persist` chains, invariants, and exact replay determinism.

mod common;

use common::{all_pos, exp_cum, exp_supported};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Skill, Tech, UnitType};
use sim_core::grid::chebyshev;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::score::{score, score_breakdown};
use sim_core::setup::new_game_from_seed;
use sim_core::state::{Config, Phase, Ruleset, State};

const ALL_V2_TECHS: [Tech; 16] = [
    Tech::Climbing,
    Tech::Hunting,
    Tech::Organization,
    Tech::Riding,
    Tech::Archery,
    Tech::Forestry,
    Tech::Farming,
    Tech::Strategy,
    Tech::Mining,
    Tech::Roads,
    Tech::FreeSpirit,
    Tech::Trade,
    Tech::Chivalry,
    Tech::Mathematics,
    Tech::Construction,
    Tech::Smithery,
];

fn v2_config(turn_cap: u32) -> Config {
    Config {
        turn_cap,
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

/// Biased random policy that favors v2 unit training, enemy-directed movement, and Persist
/// follow-up attacks so multi-seed playouts thoroughly exercise M19 mechanics (Q55 A).
fn pick_m19(s: &State, legal: &[Action], rng: &mut Rng) -> Action {
    // Always prioritize follow-up attacks from a Persist unit currently in Phase::Moved.
    let persist_attacks: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| {
            let Action::Attack { unit, .. } = *a else {
                return false;
            };
            s.unit(unit).is_some_and(|u| {
                u.phase == Phase::Moved && s.tables.unit(u.kind).skills.contains(&Skill::Persist)
            })
        })
        .collect();
    if !persist_attacks.is_empty() {
        return persist_attacks[rng.gen_range(persist_attacks.len() as u32) as usize];
    }

    let priority: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| {
            matches!(
                a,
                Action::Train {
                    kind: UnitType::Swordsman | UnitType::Catapult | UnitType::Knight,
                    ..
                } | Action::Attack { .. }
                    | Action::Promote { .. }
                    | Action::Capture { .. }
            )
        })
        .collect();
    if !priority.is_empty() && rng.gen_range(4) != 0 {
        return priority[rng.gen_range(priority.len() as u32) as usize];
    }

    // Bias moves toward enemy units/cities so v2 armies engage and trigger Persist/Stiff combat.
    let targets: Vec<_> = s
        .units
        .iter()
        .filter(|u| u.owner != s.current)
        .map(|u| u.pos)
        .chain(
            s.cities
                .iter()
                .filter(|c| c.owner != s.current)
                .map(|c| c.pos),
        )
        .collect();
    let closing_moves: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| {
            let Action::Move { unit, to } = *a else {
                return false;
            };
            let Some(u) = s.unit(unit) else {
                return false;
            };
            let d_now = targets.iter().map(|&t| chebyshev(u.pos, t)).min();
            let d_new = targets.iter().map(|&t| chebyshev(to, t)).min();
            matches!((d_now, d_new), (Some(a), Some(b)) if b < a)
        })
        .collect();
    if !closing_moves.is_empty() && rng.gen_range(4) != 0 {
        return closing_moves[rng.gen_range(closing_moves.len() as u32) as usize];
    }

    let others: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| !matches!(a, Action::EndTurn | Action::Disband { .. }))
        .collect();
    if others.is_empty() || (legal.contains(&Action::EndTurn) && rng.gen_range(6) == 0) {
        return Action::EndTurn;
    }
    others[rng.gen_range(others.len() as u32) as usize]
}

fn check_m19_invariants(prev: &State, a: &Action, s: &State, ctx: &str) {
    // Unit-tile bijection, health bounds, and support caps.
    let mut seen_tiles = vec![false; s.tiles.len()];
    for u in &s.units {
        assert!(u.health >= 1 && u.health <= u.max_health, "{ctx}: {u:?}");
        let i = u.pos.row as usize * s.size as usize + u.pos.col as usize;
        assert!(!seen_tiles[i], "{ctx}: duplicate unit on {:?}", u.pos);
        seen_tiles[i] = true;
        assert_eq!(s.tiles[i].unit, Some(u.id), "{ctx}: tile unit mismatch");
    }
    for q in all_pos(s.size) {
        let i = q.row as usize * s.size as usize + q.col as usize;
        if !seen_tiles[i] {
            assert_eq!(s.tiles[i].unit, None, "{ctx}: dangling tile unit at {q:?}");
        }
    }
    for c in &s.cities {
        assert!(
            exp_supported(s, c.id) <= c.level + 1,
            "{ctx}: city {:?} over cap",
            c.id
        );
        if !c.pending_reward {
            assert!(
                c.population < exp_cum(c.level + 1) as i32,
                "{ctx}: missed level up {c:?}"
            );
        }
    }
    for pl in 0..2u8 {
        assert!(
            s.players[pl as usize].stars < 1_000_000,
            "{ctx}: stars wrap"
        );
        assert_eq!(
            score(s, pl),
            score_breakdown(s, pl).total(),
            "{ctx}: score breakdown"
        );
        for (a_exp, b_exp) in prev.players[pl as usize]
            .explored
            .iter()
            .zip(&s.players[pl as usize].explored)
        {
            if *a_exp {
                assert!(*b_exp, "{ctx}: explored mask shrank");
            }
        }
    }

    // M19 attack skill invariants.
    if let Action::Attack { unit, target } = *a {
        let prev_a = prev.unit(unit).unwrap();
        let prev_d = prev.tile(target).unit.and_then(|id| prev.unit(id)).unwrap();
        let sa = prev.tables.unit(prev_a.kind);
        let sd = prev.tables.unit(prev_d.kind);
        let defender_died = s.unit(prev_d.id).is_none();

        if let Some(after_a) = s.unit(unit) {
            if sd.skills.contains(&Skill::Stiff) {
                assert_eq!(
                    after_a.health, prev_a.health,
                    "{ctx}: Stiff defender dealt retaliation"
                );
            }
            if sa.range > 1 {
                assert_eq!(
                    after_a.pos, prev_a.pos,
                    "{ctx}: ranged attacker must not kill-move"
                );
            }
            let expected_phase = if defender_died && sa.skills.contains(&Skill::Persist) {
                Phase::Moved
            } else {
                Phase::Attacked
            };
            assert_eq!(
                after_a.phase, expected_phase,
                "{ctx}: attacker post-attack phase"
            );
        }
    }
}

#[test]
fn random_v2_playouts_train_all_new_units_trigger_persist_and_replay_identically() {
    let mut trained = [0usize; 3]; // Swordsman, Catapult, Knight
    let mut persist_chains = 0usize;
    let mut stiff_defenses = 0usize;

    for seed in 0..24u64 {
        let mut s = new_game_from_seed(seed, &MapParams::default(), &v2_config(18));
        for pl in &mut s.players {
            for t in ALL_V2_TECHS {
                pl.techs.insert(t);
            }
            pl.stars += 60;
        }
        for c in &mut s.cities {
            c.level = 3;
            c.population = 5;
        }
        let start = s.clone();
        let mut rng = Rng::new(seed ^ 0x19_beef);
        let mut log = Vec::new();
        let mut states = Vec::new();

        // Track units that already scored a Persist kill in the current turn.
        let mut persist_primed: Vec<sim_core::state::UnitId> = Vec::new();

        while !s.is_terminal() && log.len() < 1500 {
            let legal = legal_actions(&s);
            let a = pick_m19(&s, &legal, &mut rng);
            let prev = s.clone();
            apply(&mut s, &a).unwrap();
            let ctx = format!("seed {seed} step {}", log.len());
            check_m19_invariants(&prev, &a, &s, &ctx);

            match a {
                Action::Train {
                    kind: UnitType::Swordsman,
                    ..
                } => trained[0] += 1,
                Action::Train {
                    kind: UnitType::Catapult,
                    ..
                } => trained[1] += 1,
                Action::Train {
                    kind: UnitType::Knight,
                    ..
                } => trained[2] += 1,
                Action::Attack { unit, target } => {
                    let prev_d = prev.tile(target).unit.and_then(|id| prev.unit(id)).unwrap();
                    if prev.tables.unit(prev_d.kind).skills.contains(&Skill::Stiff) {
                        stiff_defenses += 1;
                    }
                    if persist_primed.contains(&unit) {
                        persist_chains += 1;
                    }
                    let prev_a = prev.unit(unit).unwrap();
                    let defender_died = s.unit(prev_d.id).is_none();
                    if defender_died
                        && s.unit(unit).is_some()
                        && prev
                            .tables
                            .unit(prev_a.kind)
                            .skills
                            .contains(&Skill::Persist)
                    {
                        if !persist_primed.contains(&unit) {
                            persist_primed.push(unit);
                        }
                    } else {
                        persist_primed.retain(|&id| id != unit);
                    }
                }
                Action::EndTurn => persist_primed.clear(),
                _ => {}
            }

            log.push(a);
            states.push(s.clone());
        }

        // Exact replay determinism check.
        let mut r = start;
        for (i, a) in log.iter().enumerate() {
            apply(&mut r, a).unwrap();
            assert_eq!(r, states[i], "seed {seed}: replay mismatch at step {i}");
        }
    }

    assert!(trained[0] >= 10, "only {} Swordsmen trained", trained[0]);
    assert!(trained[1] >= 10, "only {} Catapults trained", trained[1]);
    assert!(trained[2] >= 10, "only {} Knights trained", trained[2]);
    assert!(
        persist_chains >= 3,
        "only {persist_chains} Persist chain follow-up attacks occurred"
    );
    assert!(
        stiff_defenses >= 3,
        "only {stiff_defenses} attacks against Stiff defenders occurred"
    );
}
