//! Milestone 8: behavior floor (docs/MILESTONE_8 "Tests" item 5). Rush and Economy each win at
//! least 75% of 40 games (seeds 0..20, both seats, turn cap 30) against `random`. The numbers
//! come from the brief; the full 200-game 80% gate is `scripts/bots_gate.sh`.

use bots::{run_match, BotKind};
use sim_core::map::MapParams;
use sim_core::state::Config;

fn floor(kind: BotKind) {
    let config = Config {
        turn_cap: 30,
        ..Config::default()
    };
    let m = run_match(kind, BotKind::Random, 0..20, &MapParams::default(), &config);
    assert_eq!(m.games, 40);
    assert!(
        m.a_wins >= 30,
        "{kind:?} won {} of 40 against random ({} draws)",
        m.a_wins,
        m.draws
    );
}

#[test]
fn rush_beats_random() {
    floor(BotKind::Rush);
}

#[test]
fn economy_beats_random() {
    floor(BotKind::Economy);
}
