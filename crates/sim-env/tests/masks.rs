//! Milestone 9 "Tests" item 6: mask correctness (spec "Mask correctness (property tests)").
//! Across playout states of every opponent kind (pending rewards and terminal states included):
//! every true path `(actor, kind, arg)` decodes to a legal action; every legal action is reached
//! by exactly one true path; the decoded paths equal `legal_actions` as a set; with a pending
//! reward only that city's actor is true; terminal states have all masks false; masks for
//! no-argument kinds have length 0. Paths are decoded twice: by the independent `my_decode` and
//! by `decode_action`.

mod common;

use common::*;
use sim_core::actions::{legal_actions, Action};
use sim_core::data::UnitType;
use sim_core::state::{CityId, State};
use sim_core::testing::StateBuilder;
use sim_env::actions::{arg_len, decode_action, num_actors, ActionMasks, ActionType, Factored};

/// Every true path of the masks, checking the per-level consistency on the way: an actor is true
/// iff it has a true kind, and an argument kind is true iff it has a true argument.
fn true_paths(s: &State, m: &ActionMasks) -> Vec<(usize, ActionType, usize)> {
    let actors = num_actors(s.size);
    assert_eq!(m.actor().len(), actors, "actor mask length");
    let mut out = Vec::new();
    for a in 0..actors {
        let kinds = m.kind(a);
        assert_eq!(
            m.actor()[a],
            kinds.iter().any(|&k| k),
            "actor {a} vs its kinds"
        );
        for k in ActionType::ALL {
            let args = m.arg(a, k);
            assert_eq!(
                args.len(),
                my_arg_len(s.size, k),
                "arg mask length {a} {k:?}"
            );
            if my_arg_len(s.size, k) == 0 {
                if kinds[kind_i(k)] {
                    out.push((a, k, 0));
                }
            } else {
                assert_eq!(
                    kinds[kind_i(k)],
                    args.iter().any(|&x| x),
                    "kind {k:?} of actor {a} vs its args"
                );
                for (x, &on) in args.iter().enumerate() {
                    if on {
                        assert!(kinds[kind_i(k)], "arg true under a false kind");
                        out.push((a, k, x));
                    }
                }
            }
        }
    }
    out
}

fn check_state(s: &State) {
    let legal = legal_actions(s);
    let m = ActionMasks::new(s, &legal);
    let paths = true_paths(s, &m);
    let mut decoded: Vec<Action> = Vec::new();
    for &(a, k, x) in &paths {
        let mine = my_decode(s, a, k, x)
            .unwrap_or_else(|| panic!("true path ({a}, {k:?}, {x}) does not decode"));
        assert!(
            legal.contains(&mine),
            "true path decodes to illegal {mine:?}"
        );
        let theirs = decode_action(
            s,
            &Factored {
                actor: a,
                kind: k,
                arg: x,
            },
        );
        assert_eq!(theirs, Ok(mine), "decode_action on a true path");
        decoded.push(mine);
    }
    // Exactly one path per legal action, and no duplicates in the legal list.
    let mut l = legal.clone();
    l.sort_by_key(|a| format!("{a:?}"));
    l.dedup();
    assert_eq!(l.len(), legal.len(), "legal list has duplicates");
    assert_eq!(paths.len(), legal.len(), "paths vs legal actions");
    for a in &legal {
        assert_eq!(
            decoded.iter().filter(|d| *d == a).count(),
            1,
            "{a:?} not reached by exactly one path"
        );
    }
}

fn pending_city(s: &State) -> Option<CityId> {
    s.cities
        .iter()
        .find(|c| c.owner == s.current && c.pending_reward)
        .map(|c| c.id)
}

#[test]
fn masks_are_exactly_the_legal_list_on_playout_states() {
    let (mut states, mut pending, mut terminal) = (0, 0, 0);
    for s in corpus(0..20, 2) {
        check_state(&s);
        states += 1;
        pending += usize::from(pending_city(&s).is_some());
        terminal += usize::from(s.is_terminal());
    }
    assert!(states > 3000, "too few states: {states}");
    assert!(pending > 40, "too few pending-reward states: {pending}");
    assert!(terminal >= 60, "too few terminal states: {terminal}");
}

#[test]
fn pending_reward_allows_only_that_city_actor() {
    let mut seen = 0;
    for s in corpus(0..10, 2) {
        let Some(city) = pending_city(&s) else {
            continue;
        };
        let m = ActionMasks::new(&s, &legal_actions(&s));
        let n = s.size as usize * s.size as usize;
        let slot = n + ti(s.size, s.city(city).pos);
        for (a, &on) in m.actor().iter().enumerate() {
            assert_eq!(
                on,
                a == slot,
                "actor {a} with a pending reward at slot {slot}"
            );
        }
        let kinds = m.kind(slot);
        for k in ActionType::ALL {
            assert_eq!(kinds[kind_i(k)], k == ActionType::ChooseReward, "{k:?}");
        }
        seen += 1;
    }
    assert!(seen > 20, "too few pending-reward states: {seen}");
}

#[test]
fn terminal_and_empty_masks_are_all_false() {
    let mut seen = 0;
    for s in corpus(0..10, 50) {
        if !s.is_terminal() {
            continue;
        }
        assert!(legal_actions(&s).is_empty());
        let m = ActionMasks::new(&s, &[]);
        assert!(m.actor().iter().all(|&x| !x));
        for a in 0..num_actors(s.size) {
            assert!(m.kind(a).iter().all(|&x| !x));
            for k in ActionType::ALL {
                assert!(m.arg(a, k).iter().all(|&x| !x));
                assert_eq!(m.arg(a, k).len(), arg_len(s.size, k));
            }
        }
        seen += 1;
    }
    assert!(seen >= 30, "too few terminal states: {seen}");
}

#[test]
fn scripted_masks_at_exact_slots() {
    // 6x6 (N = 36). Player 0: no techs, 5 stars, Warrior at (1,1) = 7 next to an enemy Warrior
    // at (2,2) = 14, city at (4,4) = 28. A tier-1 tech costs 4 + 1 x 1 = 5.
    let s = StateBuilder::new(6)
        .explore_all()
        .stars(0, 5)
        .city(0, p(4, 4))
        .unit(0, UnitType::Warrior, p(1, 1))
        .unit(1, UnitType::Warrior, p(2, 2))
        .build();
    let legal = legal_actions(&s);
    let m = ActionMasks::new(&s, &legal);
    let on: Vec<usize> = (0..num_actors(6)).filter(|&a| m.actor()[a]).collect();
    // Unit actor 7, city actor 36 + 28 = 64, global 108 (no site actions on this board).
    assert_eq!(on, vec![7, 64, 108]);
    let k7 = m.kind(7);
    assert!(k7[kind_i(ActionType::Move)]);
    assert!(k7[kind_i(ActionType::Attack)]);
    assert!(k7[kind_i(ActionType::Recover)]);
    assert!(!k7[kind_i(ActionType::Capture)]);
    assert!(!k7[kind_i(ActionType::Train)]);
    let attack = m.arg(7, ActionType::Attack);
    assert_eq!((0..36).filter(|&x| attack[x]).collect::<Vec<_>>(), vec![14]);
    assert_eq!(m.arg(7, ActionType::Recover).len(), 0);
    let train = m.arg(64, ActionType::Train);
    assert_eq!(train, &[true, false, false, false, false]);
    let global = m.kind(108);
    assert!(global[kind_i(ActionType::EndTurn)]);
    assert!(global[kind_i(ActionType::Research)]);
    // Only the tier-1 techs without prerequisites cost 5; Organization is not researched yet.
    let research = m.arg(108, ActionType::Research);
    assert_eq!(
        research,
        &[true, true, true, true, false, false, false, false, false]
    );
    // Slots of other families stay empty: no unit/site at the city tile, no city at the unit.
    assert!(m.kind(28).iter().all(|&x| !x));
    assert!(m.kind(36 + 7).iter().all(|&x| !x));
    assert!(m.kind(72 + 28).iter().all(|&x| !x));
    assert!(m.arg(108, ActionType::Move).iter().all(|&x| !x));
    assert_eq!(m.arg(108, ActionType::Move).len(), 36);
}
