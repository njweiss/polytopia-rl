//! Milestone 16 "Tests" items 5 and 8: city connections and their population, the road under a
//! captured village, and v2 playouts with roads that replay identically (docs/MILESTONE_16 "City
//! connections", "Capture of a village with a road"; clarifications R2, R3, R4, R6).
//!
//! The path rule and the update are replayed in this file from the brief (`exp_connected`,
//! `check_update`); nothing here calls `is_connected`, `road_usable`, `legal_actions` or `apply` to
//! compute an expectation. Numbers from the brief: `connection_population` 1, `road_cost` 3; level
//! thresholds from the spec (`L(L+1)/2 - 1` cumulative population for level L).

mod common;

use common::{all_pos, cheb, exp_cum, p, pick, tile_mut};
use sim_core::actions::{apply, legal_actions, Action};
use sim_core::data::{Reward, Tech, UnitType};
use sim_core::grid::Pos;
use sim_core::map::MapParams;
use sim_core::rng::Rng;
use sim_core::roads::is_connected;
use sim_core::setup::new_game_from_seed;
use sim_core::state::{City, CityId, Config, PlayerId, Ruleset, State, UnitId};
use sim_core::testing::StateBuilder;

fn v2_config() -> Config {
    Config {
        ruleset: Ruleset::V2,
        ..Config::default()
    }
}

fn city(s: &State, id: u32) -> &City {
    s.cities.iter().find(|c| c.id == CityId(id)).unwrap()
}

fn pops(s: &State) -> Vec<u32> {
    s.cities.iter().map(|c| c.population).collect()
}

fn unit_on(s: &State, q: Pos) -> UnitId {
    s.units.iter().find(|u| u.pos == q).unwrap().id
}

fn owner_of_tile(s: &State, q: Pos) -> Option<PlayerId> {
    s.tile(q)
        .owner_city
        .map(|c| s.cities.iter().find(|x| x.id == c).unwrap().owner)
}

/// The brief's "Usable".
fn usable(s: &State, player: PlayerId, q: Pos) -> bool {
    let t = s.tile(q);
    (t.road || t.city.is_some()) && owner_of_tile(s, q).is_none_or(|o| o == player)
}

/// The owner's capital: the city with `is_capital` that `player` owns (only one outside terminal
/// states, which the checks below skip).
fn capital_of(s: &State, player: PlayerId) -> Option<CityId> {
    s.cities
        .iter()
        .find(|c| c.owner == player && c.is_capital)
        .map(|c| c.id)
}

/// The brief's path rule: a non-capital city reaches its owner's capital through 8-adjacent tiles
/// that are all usable for its owner (the end tiles excepted).
fn exp_connected(s: &State, id: CityId) -> bool {
    let c = s.cities.iter().find(|c| c.id == id).unwrap();
    if c.is_capital {
        return false;
    }
    let Some(cap) = capital_of(s, c.owner) else {
        return false;
    };
    let goal = s.cities.iter().find(|k| k.id == cap).unwrap().pos;
    let mut seen = vec![c.pos];
    let mut frontier = vec![c.pos];
    while let Some(at) = frontier.pop() {
        for q in all_pos(s.size) {
            if cheb(q, at) != 1 || seen.contains(&q) {
                continue;
            }
            if q == goal {
                return true;
            }
            if usable(s, c.owner, q) {
                seen.push(q);
                frontier.push(q);
            }
        }
    }
    false
}

/// 9x9, everything explored, player 0 to move with Riding, Roads and 20 stars. City 0: player 0's
/// capital on (1,1) at level 3 with 5 population (the next level needs 9, so +1s never level it
/// up). City 1: player 1's capital on `p1_capital`, level 3, 5 population. `extra` adds the rest.
fn board(p1_capital: Pos, extra: impl FnOnce(StateBuilder) -> StateBuilder) -> State {
    let b = StateBuilder::new(9)
        .city_with(0, p(1, 1), |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .city_with(1, p1_capital, |c| {
            c.is_capital = true;
            c.level = 3;
            c.population = 5;
        })
        .techs(0, &[Tech::Riding, Tech::Roads])
        .stars(0, 20)
        .explore_all();
    extra(b).config(v2_config()).build()
}

fn with_roads(mut s: State, roads: &[Pos]) -> State {
    for &q in roads {
        tile_mut(&mut s, q).road = true;
    }
    s
}

fn act(s: &mut State, a: Action) {
    assert_eq!(apply(s, &a), Ok(()), "{a:?}");
}

fn road(at: Pos) -> Action {
    Action::BuildRoad { at }
}

// ---------------------------------------------------------------- gains

/// City 2 (player 0) on (1,5). Roads on (1,2) (capital's land) and (1,4) (city 2's land); the
/// last link (1,3) is neutral. Building it connects city 2: it and the capital gain 1 each, once.
#[test]
fn a_connected_city_and_its_capital_each_gain_one_once() {
    let mut s = with_roads(board(p(7, 7), |b| b.city(0, p(1, 5))), &[p(1, 2), p(1, 4)]);
    assert_eq!(s.tile(p(1, 3)).owner_city, None, "the last link is neutral");
    assert!(!exp_connected(&s, CityId(2)));
    assert!(!is_connected(&s, CityId(2)));

    act(&mut s, road(p(1, 3)));
    assert!(exp_connected(&s, CityId(2)));
    assert!(is_connected(&s, CityId(2)));
    assert_eq!(pops(&s), vec![6, 5, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));
    assert_eq!(
        city(&s, 0).connected_to,
        None,
        "capitals hold no connection"
    );
    assert_eq!(city(&s, 1).connected_to, None);
    assert_eq!(s.players[0].stars, 17);

    // More actions of both players: nothing more.
    act(&mut s, road(p(3, 3)));
    act(&mut s, Action::EndTurn);
    act(&mut s, Action::EndTurn);
    act(&mut s, road(p(4, 4)));
    assert_eq!(pops(&s), vec![6, 5, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));
}

/// Any action triggers the update: roads laid on the board connect on the first action, here
/// EndTurn. City 2 (player 0) on (1,4) and city 3 (player 0) on (1,7); roads (1,2), (1,3), (1,5),
/// (1,6). City 3's path runs through city 2's tile (a city is a link). The capital gains one per
/// connected city.
#[test]
fn a_path_may_pass_through_an_own_city() {
    let mut s = with_roads(
        board(p(7, 7), |b| b.city(0, p(1, 4)).city(0, p(1, 7))),
        &[p(1, 2), p(1, 3), p(1, 5), p(1, 6)],
    );
    assert!(s.tile(p(1, 4)).city.is_some() && !s.tile(p(1, 4)).road);
    act(&mut s, Action::EndTurn);
    assert_eq!(pops(&s), vec![7, 5, 1, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));
    assert_eq!(city(&s, 3).connected_to, Some(CityId(0)));
    assert!(is_connected(&s, CityId(3)));
}

#[test]
fn an_adjacent_city_and_capital_are_connected_without_roads() {
    let mut s = board(p(7, 7), |b| b.city(0, p(1, 2)));
    assert!(s.tiles.iter().all(|t| !t.road));
    assert!(is_connected(&s, CityId(2)));
    act(&mut s, Action::EndTurn);
    assert_eq!(pops(&s), vec![6, 5, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));
}

/// A neutral village under a road is a usable link (it is a road on a neutral tile).
#[test]
fn neutral_roads_and_a_neutral_village_road_count() {
    let mut s = with_roads(
        board(p(7, 7), |b| b.city(0, p(1, 5)).village(p(1, 3))),
        &[p(1, 2), p(1, 4)],
    );
    act(&mut s, road(p(1, 3)));
    assert!(s.tile(p(1, 3)).village && s.tile(p(1, 3)).road);
    assert_eq!(pops(&s), vec![6, 5, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));
}

/// Player 1's city 2 on (1,4) owns rows 0-2 x cols 3-5, which separates the capital from player
/// 0's city 3 on (1,7). Roads everywhere on rows 0-2 between them (some in player 1's land) do
/// not connect it.
#[test]
fn a_path_through_enemy_territory_does_not_connect() {
    let mut roads = Vec::new();
    for r in 0..3u8 {
        for c in 2..7u8 {
            if (r, c) != (1, 4) {
                roads.push(p(r, c));
            }
        }
    }
    let mut s = with_roads(
        board(p(7, 7), |b| b.city(1, p(1, 4)).city(0, p(1, 7))),
        &roads,
    );
    assert_eq!(owner_of_tile(&s, p(0, 3)), Some(1));
    assert!(!exp_connected(&s, CityId(3)));
    assert!(!is_connected(&s, CityId(3)));
    act(&mut s, Action::EndTurn);
    act(&mut s, Action::EndTurn);
    assert_eq!(pops(&s), vec![5, 5, 0, 0]);
    assert!(s.cities.iter().all(|c| c.connected_to.is_none()));
}

#[test]
fn is_connected_ignores_connected_to() {
    // Linked by roads, `connected_to` still None (no action applied yet).
    let s = with_roads(
        board(p(7, 7), |b| b.city(0, p(1, 5))),
        &[p(1, 2), p(1, 3), p(1, 4)],
    );
    assert_eq!(city(&s, 2).connected_to, None);
    assert!(is_connected(&s, CityId(2)));
    // Not linked, `connected_to` set by hand.
    let t = board(p(7, 7), |b| {
        b.city_with(0, p(1, 5), |c| {
            c.connected_to = Some(CityId(0));
            c.population = 1;
        })
    });
    assert!(!is_connected(&t, CityId(2)));
}

// ---------------------------------------------------------------- losses

/// The board of `a_path_may_pass_through_an_own_city`, plus a player 1 Warrior on city 2's tile.
/// Player 0 ends the turn (both cities connect: 7, 5, 1, 1); player 1 captures city 2. City 2
/// loses its connection (it and city 0 lose 1; player 1's capital is not linked to it), and city
/// 3's path through city 2 breaks (it and city 0 lose 1). The capital's level stays 3.
#[test]
fn capturing_a_city_on_the_path_breaks_the_connection() {
    let mut s = with_roads(
        board(p(7, 7), |b| {
            b.city(0, p(1, 4))
                .city(0, p(1, 7))
                .unit(1, UnitType::Warrior, p(1, 4))
        }),
        &[p(1, 2), p(1, 3), p(1, 5), p(1, 6)],
    );
    act(&mut s, Action::EndTurn);
    assert_eq!(pops(&s), vec![7, 5, 1, 1]);
    let w = unit_on(&s, p(1, 4));
    act(&mut s, Action::Capture { unit: w });
    assert_eq!(city(&s, 2).owner, 1);
    assert_eq!(pops(&s), vec![5, 5, 0, 0]);
    assert!(s.cities.iter().all(|c| c.connected_to.is_none()));
    assert_eq!(city(&s, 0).level, 3);
    assert!(!is_connected(&s, CityId(3)));
}

/// R6. Player 1's capital (city 1) on (4,7) owns rows 3-5 x cols 6-8. City 2 (player 0) on (1,5)
/// is linked to city 0 by roads (1,2) to (1,4), and to city 1 by roads (2,6) and (3,7). Player 0
/// ends the turn (city 2 connects to city 0: 6, 5, 1); player 1 captures city 2. Its old
/// connection is lost first (city 2 and city 0 lose 1), then it connects to its new owner's
/// capital (city 2 and city 1 gain 1).
#[test]
fn capturing_the_connected_city_moves_its_connection_to_the_new_owner() {
    let mut s = with_roads(
        board(p(4, 7), |b| {
            b.city(0, p(1, 5)).unit(1, UnitType::Warrior, p(1, 5))
        }),
        &[p(1, 2), p(1, 3), p(1, 4), p(2, 6), p(3, 7)],
    );
    act(&mut s, Action::EndTurn);
    assert_eq!(pops(&s), vec![6, 5, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));
    let w = unit_on(&s, p(1, 5));
    act(&mut s, Action::Capture { unit: w });
    assert_eq!(city(&s, 2).owner, 1);
    assert_eq!(pops(&s), vec![5, 6, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(1)));
    assert!(is_connected(&s, CityId(2)));
}

// ---------------------------------------------------------------- village capture

/// City 2 (player 0) on (1,5); a neutral village on (1,3) with a road; roads (1,2) and (1,4).
fn village_board(capturer: PlayerId) -> State {
    with_roads(
        board(p(7, 7), |b| {
            b.city(0, p(1, 5))
                .village(p(1, 3))
                .unit(capturer, UnitType::Warrior, p(1, 3))
        }),
        &[p(1, 2), p(1, 3), p(1, 4)],
    )
}

/// Player 0 captures the village: its road disappears, but the new city (3) is a link for its
/// owner, so city 2 stays connected, and city 3 connects through (1,2) (it and city 0 gain 1).
#[test]
fn capturing_a_village_removes_its_road_and_the_new_city_still_links() {
    let mut s = village_board(0);
    act(&mut s, Action::EndTurn);
    act(&mut s, Action::EndTurn);
    assert_eq!(pops(&s), vec![6, 5, 1]);
    let w = unit_on(&s, p(1, 3));
    act(&mut s, Action::Capture { unit: w });
    let t = s.tile(p(1, 3));
    assert_eq!(t.city, Some(CityId(3)));
    assert!(!t.road, "the road under a captured village disappears");
    assert!(!t.village);
    assert_eq!(pops(&s), vec![7, 5, 1, 1]);
    assert_eq!(city(&s, 2).connected_to, Some(CityId(0)));
    assert_eq!(city(&s, 3).connected_to, Some(CityId(0)));
}

/// Player 1 captures the village on player 0's path: the road disappears and city 2 loses its
/// connection (it and city 0 lose 1).
#[test]
fn an_enemy_village_capture_removes_the_road_and_breaks_the_path() {
    let mut s = village_board(1);
    act(&mut s, Action::EndTurn);
    assert_eq!(pops(&s), vec![6, 5, 1]);
    let w = unit_on(&s, p(1, 3));
    act(&mut s, Action::Capture { unit: w });
    let t = s.tile(p(1, 3));
    assert_eq!(t.city, Some(CityId(3)));
    assert!(!t.road);
    assert_eq!(city(&s, 3).owner, 1);
    assert_eq!(pops(&s), vec![5, 5, 0, 0]);
    assert!(s.cities.iter().all(|c| c.connected_to.is_none()));
}

// ---------------------------------------------------------------- levels

/// The capital and city 2 both start at level 1 with 1 population (level 2 needs 2). Connecting
/// them (the last road, (1,4)) brings both to 2: each levels up with a pending reward, and the
/// capital (lowest id) must choose first. After both choose, player 1 captures the village on the
/// path: both drop back to 1 population and stay level 2.
#[test]
fn a_gain_levels_up_with_a_pending_reward_and_a_loss_never_lowers_the_level() {
    let mut s = with_roads(
        StateBuilder::new(9)
            .city_with(0, p(1, 1), |c| {
                c.is_capital = true;
                c.population = 1;
            })
            .city_with(1, p(7, 7), |c| c.is_capital = true)
            .city_with(0, p(1, 5), |c| c.population = 1)
            .village(p(1, 3))
            .unit(1, UnitType::Warrior, p(1, 3))
            .techs(0, &[Tech::Riding, Tech::Roads])
            .stars(0, 20)
            .explore_all()
            .config(v2_config())
            .build(),
        &[p(1, 2), p(1, 3)],
    );
    assert_eq!(exp_cum(2), 2);
    act(&mut s, road(p(1, 4)));
    for id in [0, 2] {
        let c = city(&s, id);
        assert_eq!(
            (c.population, c.level, c.pending_reward),
            (2, 2, true),
            "{id}"
        );
    }
    let legal = legal_actions(&s);
    assert!(legal
        .iter()
        .all(|a| matches!(a, Action::ChooseReward { city, .. } if *city == CityId(0))));
    for id in [0, 2] {
        act(
            &mut s,
            Action::ChooseReward {
                city: CityId(id),
                reward: Reward::Workshop,
            },
        );
    }
    act(&mut s, Action::EndTurn);
    let w = unit_on(&s, p(1, 3));
    act(&mut s, Action::Capture { unit: w });
    for id in [0, 2] {
        let c = city(&s, id);
        assert_eq!(
            (c.population, c.level, c.pending_reward),
            (1, 2, false),
            "{id}"
        );
        assert!(c.workshop);
    }
}

// ---------------------------------------------------------------- playouts

/// The brief's update, checked on `post` (the state after `a` from `prev`): for every city in id
/// order, a held connection is dropped when the path rule fails or its capital is no longer the
/// owner's capital (both lose 1); then an unconnected non-capital city on a path connects (it and
/// the capital gain 1). Returns the number of (gains, losses). Population and level are checked
/// for every action that changes population only through connections.
fn check_update(prev: &State, a: &Action, post: &State, ctx: &str) -> (u32, u32) {
    let mut delta = vec![0i64; post.cities.len()];
    let (mut gains, mut losses) = (0, 0);
    for c in &post.cities {
        let i = c.id.0 as usize;
        let mut conn = prev.cities.get(i).and_then(|x| x.connected_to);
        if let Some(k) = conn {
            let kc = &post.cities[k.0 as usize];
            let kept = exp_connected(post, c.id) && kc.is_capital && kc.owner == c.owner;
            if !kept {
                conn = None;
                delta[i] -= 1;
                delta[k.0 as usize] -= 1;
                losses += 1;
            }
        }
        if conn.is_none() && !c.is_capital && exp_connected(post, c.id) {
            let cap = capital_of(post, c.owner).unwrap();
            conn = Some(cap);
            delta[i] += 1;
            delta[cap.0 as usize] += 1;
            gains += 1;
        }
        assert_eq!(c.connected_to, conn, "{ctx}: city {i} after {a:?}");
    }
    if !matches!(a, Action::Site { .. } | Action::ChooseReward { .. }) {
        for c in &post.cities {
            let i = c.id.0 as usize;
            let (pop0, level0, pending0) = prev
                .cities
                .get(i)
                .map_or((0, 1, false), |x| (x.population, x.level, x.pending_reward));
            assert_eq!(
                c.population as i64,
                pop0 as i64 + delta[i],
                "{ctx}: city {i} population after {a:?}"
            );
            let (level, pending) =
                if delta[i] > 0 && !pending0 && c.population >= exp_cum(level0 + 1) {
                    (level0 + 1, true)
                } else {
                    (level0, pending0)
                };
            assert_eq!(
                (c.level, c.pending_reward),
                (level, pending),
                "{ctx}: city {i} level after {a:?}"
            );
        }
    }
    (gains, losses)
}

/// A random policy that grows road networks: a legal Capture half the time; else a third of the
/// time a road, preferring tiles next to one the player can already use; else `pick`.
fn pick_growing(s: &State, legal: &[Action], rng: &mut Rng) -> Action {
    let of = |f: &dyn Fn(&Action) -> bool| -> Vec<Action> {
        legal.iter().copied().filter(|a| f(a)).collect()
    };
    let captures = of(&|a| matches!(a, Action::Capture { .. }));
    if !captures.is_empty() && rng.gen_range(2) == 0 {
        return captures[rng.gen_range(captures.len() as u32) as usize];
    }
    let roads = of(&|a| matches!(a, Action::BuildRoad { .. }));
    if !roads.is_empty() && rng.gen_range(3) == 0 {
        let me = s.current;
        let growing = of(&|a| match *a {
            Action::BuildRoad { at } => all_pos(s.size)
                .into_iter()
                .any(|q| cheb(q, at) == 1 && usable(s, me, q)),
            _ => false,
        });
        let from = if growing.is_empty() { roads } else { growing };
        return from[rng.gen_range(from.len() as u32) as usize];
    }
    pick(rng, legal)
}

/// Random v2 games where both players start with Roads and extra stars and build roads often.
/// Every step: `connected_to`, population and level follow the replayed update (terminal states
/// skipped: a capital capture leaves one player with two capitals, see docs/QUESTIONS.md Q47), and
/// `road` only ever goes from false to true except under a captured village. Then the action log
/// replays to the identical sequence of states.
#[test]
fn random_v2_playouts_follow_the_update_and_replay_identically() {
    let (mut gains, mut losses, mut built) = (0, 0, 0);
    for seed in 0..30u64 {
        let cfg = Config {
            turn_cap: 30,
            ..v2_config()
        };
        let mut s = new_game_from_seed(seed, &MapParams::default(), &cfg);
        for pl in &mut s.players {
            pl.techs.insert(Tech::Riding);
            pl.techs.insert(Tech::Roads);
            pl.stars += 30;
        }
        let start = s.clone();
        let mut rng = Rng::new(seed ^ 0x16_c0);
        let mut log = Vec::new();
        let mut states = Vec::new();
        while !s.is_terminal() && log.len() < 3000 {
            let legal = legal_actions(&s);
            let a = pick_growing(&s, &legal, &mut rng);
            let prev = s.clone();
            act(&mut s, a);
            let ctx = format!("seed {seed}, step {}", log.len());
            if matches!(a, Action::BuildRoad { .. }) {
                built += 1;
            }
            for q in all_pos(s.size) {
                let (r0, r1) = (prev.tile(q).road, s.tile(q).road);
                let captured_village = prev.tile(q).village && !s.tile(q).village;
                let built_here = a == road(q);
                assert_eq!(
                    r1,
                    (r0 && !captured_village) || built_here,
                    "{ctx}: road at {q:?} after {a:?}"
                );
            }
            if !s.is_terminal() {
                let (g, l) = check_update(&prev, &a, &s, &ctx);
                gains += g;
                losses += l;
            }
            log.push(a);
            states.push(s.clone());
        }
        let mut r = start;
        for (i, a) in log.iter().enumerate() {
            act(&mut r, *a);
            assert_eq!(r, states[i], "seed {seed}: replay differs at step {i}");
        }
    }
    // The reference run of this policy built 2261 roads, made 77 connections and lost 8.
    assert!(built > 1000, "only {built} roads built");
    assert!(gains > 30, "only {gains} connections made");
    assert!(losses > 2, "only {losses} connections lost");
}
