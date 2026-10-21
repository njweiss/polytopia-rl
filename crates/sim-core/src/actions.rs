//! Actions, legality and `apply` (docs/MILESTONE_3 "Move" .. "legal_actions and apply";
//! docs/MILESTONE_4 "Research" .. "legal_actions order").

use crate::combat::{resolve_attack, Combatant};
use crate::data::{
    Reward, SiteAction, Skill, Tech, UnitType, ALL_SITE_ACTIONS, ALL_TECHS, ALL_UNITS,
};
use crate::economy::{
    cumulative_population_for_level, income, research_cost, reward_menu, supported_units,
};
use crate::explorer::explorer_walk;
use crate::grid::{chebyshev, Pos};
use crate::map::{Resource, Terrain};
use crate::movement::{defense_bonus, reachable_tiles};
use crate::spawn::spawn_super_unit;
use crate::state::{City, CityId, Dir, Improvement, Phase, PlayerId, State, Unit, UnitId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Move { unit: UnitId, to: Pos },
    Attack { unit: UnitId, target: Pos },
    Capture { unit: UnitId },
    Recover { unit: UnitId },
    Promote { unit: UnitId },
    Train { city: CityId, kind: UnitType },
    Site { at: Pos, action: SiteAction },
    Research { tech: Tech },
    ChooseReward { city: CityId, reward: Reward },
    EndTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionError {
    Illegal,
}

/// The current player's legal actions in canonical order. With a pending reward: only the
/// lowest-id pending city's ChooseReward, in menu order. Otherwise: units by ascending id (per unit
/// Move (row-major), Attack (row-major), Recover, Promote); Train per city by ascending id in
/// `ALL_UNITS` order; site actions row-major in `ALL_SITE_ACTIONS` order; Research in `ALL_TECHS`
/// order; then EndTurn. Empty once there is an outcome.
pub fn legal_actions(state: &State) -> Vec<Action> {
    let mut out = Vec::new();
    if state.outcome.is_some() {
        return out;
    }
    if let Some(city) = pending_city(state) {
        for reward in reward_menu(state, city) {
            out.push(Action::ChooseReward { city, reward });
        }
        return out;
    }
    // `state.units` and `state.cities` are kept in ascending id order.
    for u in state.units.iter().filter(|u| u.owner == state.current) {
        push_unit_actions(state, u, &mut out);
    }
    for c in state.cities.iter().filter(|c| c.owner == state.current) {
        for kind in ALL_UNITS {
            if train_legal(state, c.id, kind) {
                out.push(Action::Train { city: c.id, kind });
            }
        }
    }
    for row in 0..state.size {
        for col in 0..state.size {
            let at = Pos::new(row, col);
            for action in ALL_SITE_ACTIONS {
                if site_legal(state, at, action) {
                    out.push(Action::Site { at, action });
                }
            }
        }
    }
    for tech in ALL_TECHS {
        if research_legal(state, tech) {
            out.push(Action::Research { tech });
        }
    }
    out.push(Action::EndTurn);
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
        Action::Capture { unit } => apply_capture(state, unit),
        Action::Recover { unit } => apply_recover(state, unit),
        Action::Promote { unit } => apply_promote(state, unit),
        Action::Train { city, kind } => apply_train(state, city, kind),
        Action::Site { at, action } => apply_site(state, at, action),
        Action::Research { tech } => apply_research(state, tech),
        Action::ChooseReward { city, reward } => apply_choose_reward(state, city, reward),
        Action::EndTurn => apply_end_turn(state),
    }
    Ok(())
}

/// Same answer as `legal_actions(state).contains(action)`, enumerating only the actor's actions.
fn is_legal(state: &State, action: &Action) -> bool {
    if state.outcome.is_some() {
        return false;
    }
    // Forced rewards (D9): nothing else is legal while one is pending.
    if let Some(pending) = pending_city(state) {
        return matches!(*action, Action::ChooseReward { city, reward }
            if city == pending && reward_menu(state, city).contains(&reward));
    }
    let id = match *action {
        Action::Move { unit, .. }
        | Action::Attack { unit, .. }
        | Action::Capture { unit }
        | Action::Recover { unit }
        | Action::Promote { unit } => unit,
        Action::Train { city, kind } => return train_legal(state, city, kind),
        Action::Site { at, action } => return site_legal(state, at, action),
        Action::Research { tech } => return research_legal(state, tech),
        Action::ChooseReward { .. } => return false,
        Action::EndTurn => return true,
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

/// The current player's lowest-id city with a pending reward, if any.
fn pending_city(state: &State) -> Option<CityId> {
    state
        .cities
        .iter()
        .find(|c| c.owner == state.current && c.pending_reward)
        .map(|c| c.id)
}

/// Train legality apart from the pending-reward block (checked by the callers).
fn train_legal(state: &State, city: CityId, kind: UnitType) -> bool {
    let Some(c) = state.cities.iter().find(|c| c.id == city) else {
        return false;
    };
    let stats = state.tables.unit(kind);
    let Some(cost) = stats.cost else {
        return false;
    };
    let me = &state.players[state.current as usize];
    c.owner == state.current
        && state.tile(c.pos).unit.is_none()
        && stats.unlocked_by.is_none_or(|t| me.techs.contains(t))
        && me.stars >= cost
        && supported_units(state, city) < c.level + 1
}

/// Site-action legality apart from the pending-reward block (checked by the callers).
fn site_legal(state: &State, at: Pos, action: SiteAction) -> bool {
    if at.row >= state.size || at.col >= state.size {
        return false;
    }
    let t = state.tile(at);
    let Some(owner_city) = t.owner_city else {
        return false;
    };
    let enemy_here = t
        .unit
        .and_then(|id| state.unit(id))
        .is_some_and(|u| u.owner != state.current);
    let info = state.tables.site_action(action);
    let me = &state.players[state.current as usize];
    let condition = match action {
        SiteAction::HarvestAnimal => t.resource == Some(Resource::Animal),
        SiteAction::HarvestFruit => t.resource == Some(Resource::Fruit),
        SiteAction::LumberHut | SiteAction::ClearForest => {
            t.terrain == Terrain::Forest && t.improvement.is_none()
        }
        SiteAction::Farm => t.resource == Some(Resource::Crop) && t.improvement.is_none(),
        SiteAction::Mine => t.resource == Some(Resource::Metal) && t.improvement.is_none(),
    };
    state.city(owner_city).owner == state.current
        && !enemy_here
        && me.techs.contains(info.tech)
        && me.stars >= info.cost
        && condition
}

/// Research legality apart from the pending-reward block (checked by the callers).
fn research_legal(state: &State, tech: Tech) -> bool {
    let me = &state.players[state.current as usize];
    !me.techs.contains(tech)
        && state
            .tables
            .tech(tech)
            .requires
            .is_none_or(|r| me.techs.contains(r))
        && me.stars >= research_cost(state, state.current, tech)
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
pub(crate) fn reveal(state: &mut State, player: PlayerId, at: Pos) {
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
        if u.kills >= tables.rules().promotion_kills
            && !u.veteran
            && !tables.unit(u.kind).skills.contains(&Skill::Static)
        {
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

fn apply_capture(_state: &mut State, _id: UnitId) {
    unimplemented!("milestone 6: Capture")
}

fn apply_recover(state: &mut State, id: UnitId) {
    let u = state.unit(id).expect("living unit");
    let own_territory = state
        .tile(u.pos)
        .owner_city
        .is_some_and(|c| state.city(c).owner == u.owner);
    let rules = state.tables.rules();
    let heal = if own_territory {
        rules.recover_own_territory
    } else {
        rules.recover_elsewhere
    };
    let u = unit_mut(state, id);
    u.health = (u.health + heal).min(u.max_health);
    u.phase = Phase::Done;
}

fn apply_promote(state: &mut State, id: UnitId) {
    let bonus = state.tables.rules().veteran_health_bonus;
    let u = unit_mut(state, id);
    u.max_health += bonus;
    u.health = u.max_health;
    u.veteran = true;
    u.promotion_available = false;
}

fn city_mut(state: &mut State, id: CityId) -> &mut City {
    state
        .cities
        .iter_mut()
        .find(|c| c.id == id)
        .expect("city ids are never removed")
}

/// One level at a time, and only while no reward is pending; resolving a reward re-runs it.
fn level_up_check(state: &mut State, id: CityId) {
    let c = city_mut(state, id);
    if !c.pending_reward && c.population >= cumulative_population_for_level(c.level + 1) {
        c.level += 1;
        c.pending_reward = true;
    }
}

fn apply_train(state: &mut State, city: CityId, kind: UnitType) {
    let stats = state.tables.unit(kind);
    let (cost, health) = (stats.cost.expect("legal Train is trainable"), stats.health);
    let c = state.city(city);
    let (pos, owner) = (c.pos, c.owner);
    state.players[owner as usize].stars -= cost;
    let id = UnitId(state.next_unit_id);
    state.next_unit_id += 1;
    // Ids only grow, so pushing keeps `state.units` in ascending id order.
    state.units.push(Unit {
        id,
        kind,
        owner,
        pos,
        health,
        max_health: health,
        origin: Some(city),
        kills: 0,
        veteran: false,
        promotion_available: false,
        phase: Phase::Done,
        last_dir: None,
    });
    let i = state.index(pos);
    state.tiles[i].unit = Some(id);
}

fn apply_site(state: &mut State, at: Pos, action: SiteAction) {
    let info = *state.tables.site_action(action);
    let player = &mut state.players[state.current as usize];
    player.stars = player.stars - info.cost + info.stars;
    let i = state.index(at);
    let t = &mut state.tiles[i];
    match action {
        SiteAction::HarvestAnimal | SiteAction::HarvestFruit => t.resource = None,
        SiteAction::LumberHut => t.improvement = Some(Improvement::LumberHut),
        SiteAction::ClearForest => {
            t.terrain = Terrain::Field;
            t.resource = None;
        }
        SiteAction::Farm => t.improvement = Some(Improvement::Farm),
        SiteAction::Mine => t.improvement = Some(Improvement::Mine),
    }
    let owner_city = t.owner_city.expect("legal site actions are on owned tiles");
    city_mut(state, owner_city).population += info.population;
    level_up_check(state, owner_city);
}

fn apply_research(state: &mut State, tech: Tech) {
    let cost = research_cost(state, state.current, tech);
    let player = &mut state.players[state.current as usize];
    player.stars -= cost;
    player.techs.insert(tech);
}

fn apply_choose_reward(state: &mut State, city: CityId, reward: Reward) {
    let tables = state.tables.clone();
    let owner = state.city(city).owner;
    match reward {
        Reward::Workshop => city_mut(state, city).workshop = true,
        Reward::CityWall => city_mut(state, city).wall = true,
        Reward::Park => city_mut(state, city).parks += 1,
        Reward::Resources => state.players[owner as usize].stars += tables.resources_reward_stars(),
        Reward::PopulationGrowth => {
            city_mut(state, city).population += tables.population_growth_amount()
        }
        Reward::BorderGrowth => border_growth(state, city),
        Reward::Explorer => explorer_walk(state, city),
        Reward::SuperUnit => {
            spawn_super_unit(state, city);
        }
    }
    city_mut(state, city).pending_reward = false;
    level_up_check(state, city);
}

/// Claims every in-bounds tile within Chebyshev 2 of the city that has no owning city, is not a
/// village and holds no city (D7), and reveals the claimed tiles to the owner.
fn border_growth(state: &mut State, city: CityId) {
    let c = city_mut(state, city);
    c.border_growth_used = true;
    let (at, owner) = (c.pos, c.owner);
    let n = state.size;
    for r in 0..n {
        for col in 0..n {
            let q = Pos::new(r, col);
            let i = state.index(q);
            let t = &mut state.tiles[i];
            if chebyshev(at, q) <= 2 && t.owner_city.is_none() && !t.village && t.city.is_none() {
                t.owner_city = Some(city);
                state.players[owner as usize].explored[i] = true;
            }
        }
    }
}

fn apply_end_turn(state: &mut State) {
    state.current = 1 - state.current;
    if state.current == 0 {
        state.turn += 1;
    }
    let current = state.current;
    // Income first, then the unit reset (spec "Turn start"); no income on turn 0 (D1).
    if state.turn >= 1 {
        let gain = income(state, current);
        state.players[current as usize].stars += gain;
    }
    for u in state.units.iter_mut().filter(|u| u.owner == current) {
        u.phase = Phase::Fresh;
    }
}
