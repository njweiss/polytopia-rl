//! Unit actions, legality and `apply` (docs/MILESTONE_3 "Move" .. "legal_actions and apply").

use crate::combat::{resolve_attack, Combatant};
use crate::data::{Skill, Tech};
use crate::grid::{chebyshev, Pos};
use crate::map::Terrain;
use crate::movement::{defense_bonus, reachable_tiles};
use crate::state::{Dir, Phase, PlayerId, State, Unit, UnitId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Move { unit: UnitId, to: Pos },
    Attack { unit: UnitId, target: Pos },
    Recover { unit: UnitId },
    Promote { unit: UnitId },
    EndTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionError {
    Illegal,
}

/// The current player's legal actions in canonical order: units by ascending id; per unit Move
/// (row-major), Attack (row-major), Recover, Promote; then EndTurn. Empty once there is an outcome.
pub fn legal_actions(state: &State) -> Vec<Action> {
    let mut out = Vec::new();
    if state.outcome.is_some() {
        return out;
    }
    // `state.units` is kept in ascending id order.
    for u in state.units.iter().filter(|u| u.owner == state.current) {
        push_unit_actions(state, u, &mut out);
    }
    if end_turn_legal(state) {
        out.push(Action::EndTurn);
    }
    out
}

/// Applies a legal action. Any action not in `legal_actions(state)` returns `Err(Illegal)` and
/// leaves `state` completely unchanged.
pub fn apply(state: &mut State, action: &Action) -> Result<(), ActionError> {
    if !is_legal(state, action) {
        return Err(ActionError::Illegal);
    }
    match *action {
        Action::Move { unit, to } => apply_move(state, unit, to),
        Action::Attack { unit, target } => apply_attack(state, unit, target),
        Action::Recover { unit } => apply_recover(state, unit),
        Action::Promote { unit } => apply_promote(state, unit),
        Action::EndTurn => apply_end_turn(state),
    }
    Ok(())
}

/// Same answer as `legal_actions(state).contains(action)`, enumerating only the actor's actions.
fn is_legal(state: &State, action: &Action) -> bool {
    if state.outcome.is_some() {
        return false;
    }
    let id = match *action {
        Action::Move { unit, .. }
        | Action::Attack { unit, .. }
        | Action::Recover { unit }
        | Action::Promote { unit } => unit,
        Action::EndTurn => return end_turn_legal(state),
    };
    let Some(u) = state.unit(id) else {
        return false;
    };
    if u.owner != state.current {
        return false;
    }
    let mut own = Vec::new();
    push_unit_actions(state, u, &mut own);
    own.contains(action)
}

fn end_turn_legal(state: &State) -> bool {
    state.outcome.is_none()
        && !state
            .cities
            .iter()
            .any(|c| c.owner == state.current && c.pending_reward)
}

/// One unit's legal actions in canonical order (the caller checks owner and outcome).
fn push_unit_actions(state: &State, u: &Unit, out: &mut Vec<Action>) {
    let stats = state.tables.unit(u.kind);
    let dash = stats.skills.contains(&Skill::Dash);
    let escape = stats.skills.contains(&Skill::Escape);

    if u.phase == Phase::Fresh || (u.phase == Phase::Attacked && escape) {
        for to in reachable_tiles(state, u.id) {
            out.push(Action::Move { unit: u.id, to });
        }
    }
    if u.phase == Phase::Fresh || (u.phase == Phase::Moved && dash) {
        // Row-major over the square of radius `range`, clipped to the board.
        let r = stats.range as i32;
        for dr in -r..=r {
            for dc in -r..=r {
                let (row, col) = (u.pos.row as i32 + dr, u.pos.col as i32 + dc);
                let n = state.size as i32;
                if (dr, dc) == (0, 0) || row < 0 || col < 0 || row >= n || col >= n {
                    continue;
                }
                let q = Pos::new(row as u8, col as u8);
                let enemy_there = state
                    .tile(q)
                    .unit
                    .and_then(|id| state.unit(id))
                    .is_some_and(|e| e.owner != u.owner);
                if state.is_explored(u.owner, q) && enemy_there {
                    out.push(Action::Attack {
                        unit: u.id,
                        target: q,
                    });
                }
            }
        }
    }
    if u.phase == Phase::Fresh {
        out.push(Action::Recover { unit: u.id });
    }
    if u.promotion_available && !u.veteran {
        out.push(Action::Promote { unit: u.id });
    }
}

fn unit_mut(state: &mut State, id: UnitId) -> &mut Unit {
    state
        .units
        .iter_mut()
        .find(|u| u.id == id)
        .expect("legal actions name living units")
}

/// Sign of `to - from` per axis.
fn direction(from: Pos, to: Pos) -> Dir {
    Dir {
        dr: (to.row as i32 - from.row as i32).signum() as i8,
        dc: (to.col as i32 - from.col as i32).signum() as i8,
    }
}

/// Moves a unit (occupancy, position, `last_dir`) and grows its owner's explored mask from the
/// new tile. Does not touch the phase.
fn relocate(state: &mut State, id: UnitId, to: Pos) {
    let from = state.unit(id).expect("living unit").pos;
    let (fi, ti) = (state.index(from), state.index(to));
    state.tiles[fi].unit = None;
    state.tiles[ti].unit = Some(id);
    let u = unit_mut(state, id);
    u.pos = to;
    u.last_dir = Some(direction(from, to));
    let owner = u.owner;
    reveal(state, owner, to);
}

/// Unit vision: Chebyshev 1, or 2 when standing on a mountain. The mask never shrinks.
fn reveal(state: &mut State, player: PlayerId, at: Pos) {
    let radius = if state.tile(at).terrain == Terrain::Mountain {
        2
    } else {
        1
    };
    let n = state.size;
    for r in 0..n {
        for c in 0..n {
            let q = Pos::new(r, c);
            if chebyshev(at, q) <= radius {
                let i = state.index(q);
                state.players[player as usize].explored[i] = true;
            }
        }
    }
}

fn apply_move(state: &mut State, id: UnitId, to: Pos) {
    relocate(state, id, to);
    let u = unit_mut(state, id);
    u.phase = match u.phase {
        Phase::Fresh => Phase::Moved,
        _ => Phase::Done,
    };
}

fn apply_attack(state: &mut State, attacker_id: UnitId, target: Pos) {
    let defender_id = state
        .tile(target)
        .unit
        .expect("legal attacks target a unit");
    let bonus = defense_bonus(state, defender_id);
    let a = state.unit(attacker_id).expect("living attacker").clone();
    let d = state.unit(defender_id).expect("living defender").clone();
    let (sa, sd) = (state.tables.unit(a.kind), state.tables.unit(d.kind));
    let combatant = |u: &Unit, attack: u32, defense: u32| Combatant {
        attack,
        defense,
        health: u.health,
        max_health: u.max_health,
    };
    let in_range = chebyshev(a.pos, d.pos) <= sd.range;
    let outcome = resolve_attack(
        &combatant(&a, sa.attack, sa.defense),
        &combatant(&d, sd.attack, sd.defense),
        bonus,
        in_range,
    );
    let attacker_ranged = sa.range > 1;

    unit_mut(state, attacker_id).health = outcome.attacker_health;
    unit_mut(state, defender_id).health = outcome.defender_health;
    let attacker_died = outcome.attacker_health == 0;
    let defender_died = outcome.defender_health == 0;
    for (dead, pos) in [(attacker_died, a.pos), (defender_died, d.pos)] {
        if dead {
            let i = state.index(pos);
            state.tiles[i].unit = None;
        }
    }
    state.units.retain(|u| u.health > 0);

    // Kill credit, then promotion eligibility for the surviving killer.
    let killer = match (attacker_died, defender_died) {
        (_, true) => Some(attacker_id),
        (true, false) => Some(defender_id),
        _ => None,
    };
    if let Some(k) = killer {
        let tables = state.tables.clone();
        let u = unit_mut(state, k);
        u.kills += 1;
        if u.kills >= 3 && !u.veteran && !tables.unit(u.kind).skills.contains(&Skill::Static) {
            u.promotion_available = true;
        }
    }

    if attacker_died {
        return;
    }
    // Kill-move for melee attackers, unless the tile is a mountain the owner cannot climb.
    if defender_died && !attacker_ranged {
        let climbing = state.players[a.owner as usize]
            .techs
            .contains(Tech::Climbing);
        if state.tile(d.pos).terrain != Terrain::Mountain || climbing {
            relocate(state, attacker_id, d.pos);
        }
    }
    let u = unit_mut(state, attacker_id);
    u.phase = Phase::Attacked;
    if attacker_ranged {
        u.last_dir = Some(direction(a.pos, d.pos));
    }
}

fn apply_recover(state: &mut State, id: UnitId) {
    let u = state.unit(id).expect("living unit");
    let own_territory = state
        .tile(u.pos)
        .owner_city
        .is_some_and(|c| state.city(c).owner == u.owner);
    let heal = if own_territory { 4 } else { 2 };
    let u = unit_mut(state, id);
    u.health = (u.health + heal).min(u.max_health);
    u.phase = Phase::Done;
}

fn apply_promote(state: &mut State, id: UnitId) {
    let u = unit_mut(state, id);
    u.max_health += 5;
    u.health = u.max_health;
    u.veteran = true;
    u.promotion_available = false;
}

fn apply_end_turn(state: &mut State) {
    state.current = 1 - state.current;
    if state.current == 0 {
        state.turn += 1;
    }
    let current = state.current;
    for u in state.units.iter_mut().filter(|u| u.owner == current) {
        u.phase = Phase::Fresh;
    }
}
