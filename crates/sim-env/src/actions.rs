//! The factored action space (docs/MILESTONE_9 "Factored action space").

use sim_core::actions::Action;
use sim_core::data::{
    site_actions, techs, units, ALL_BUILDINGS, ALL_MONUMENTS, ALL_REWARDS, ALL_SITE_ACTIONS,
    ALL_TECHS, ALL_UNITS,
};
use sim_core::grid::Pos;
use sim_core::state::{Ruleset, State};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Move = 0,
    Attack,
    Capture,
    Recover,
    Promote,
    Train,
    ChooseReward,
    Site,
    Research,
    EndTurn,
    Disband,
    BuildRoad,
    BuildBuilding,
    BuildMonument,
    Destroy,
}

pub const NUM_ACTION_TYPES: usize = 10;
pub const V2_NUM_ACTION_TYPES: usize = 15;

const V2_ALL: [ActionType; V2_NUM_ACTION_TYPES] = [
    ActionType::Move,
    ActionType::Attack,
    ActionType::Capture,
    ActionType::Recover,
    ActionType::Promote,
    ActionType::Train,
    ActionType::ChooseReward,
    ActionType::Site,
    ActionType::Research,
    ActionType::EndTurn,
    ActionType::Disband,
    ActionType::BuildRoad,
    ActionType::BuildBuilding,
    ActionType::BuildMonument,
    ActionType::Destroy,
];

pub fn action_types(ruleset: Ruleset) -> &'static [ActionType] {
    match ruleset {
        Ruleset::V1 => &ActionType::ALL,
        Ruleset::V2 => &V2_ALL,
    }
}

impl ActionType {
    pub const ALL: [ActionType; NUM_ACTION_TYPES] = [
        ActionType::Move,
        ActionType::Attack,
        ActionType::Capture,
        ActionType::Recover,
        ActionType::Promote,
        ActionType::Train,
        ActionType::ChooseReward,
        ActionType::Site,
        ActionType::Research,
        ActionType::EndTurn,
    ];

    /// The actor slot family: 0 unit, 1 city, 2 site, 3 global.
    fn family(self) -> usize {
        match self {
            ActionType::Move
            | ActionType::Attack
            | ActionType::Capture
            | ActionType::Recover
            | ActionType::Promote
            | ActionType::Disband => 0,
            ActionType::Train | ActionType::ChooseReward => 1,
            ActionType::Site
            | ActionType::BuildRoad
            | ActionType::BuildBuilding
            | ActionType::BuildMonument
            | ActionType::Destroy => 2,
            ActionType::Research | ActionType::EndTurn => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factored {
    pub actor: usize,
    pub kind: ActionType,
    pub arg: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// `actor >= num_actors(size)`.
    ActorOutOfRange,
    /// A unit or city actor slot that holds no unit or city of `state.current`.
    NoActor,
    /// The kind does not belong to the actor's slot family.
    WrongKind,
    /// `arg >= arg_len(kind)`, or nonzero for a kind without an argument.
    BadArg,
}

fn tiles(size: u8) -> usize {
    size as usize * size as usize
}

fn tile_index(size: u8, p: Pos) -> usize {
    p.row as usize * size as usize + p.col as usize
}

fn tile_pos(size: u8, i: usize) -> Pos {
    Pos::new((i / size as usize) as u8, (i % size as usize) as u8)
}

/// `3 * size * size + 1`.
pub fn num_actors(size: u8) -> usize {
    3 * tiles(size) + 1
}

pub fn arg_len(size: u8, kind: ActionType) -> usize {
    match kind {
        ActionType::Move | ActionType::Attack => tiles(size),
        ActionType::Train => ALL_UNITS.len(),
        ActionType::ChooseReward => ALL_REWARDS.len(),
        ActionType::Site => ALL_SITE_ACTIONS.len(),
        ActionType::Research => ALL_TECHS.len(),
        ActionType::BuildBuilding => ALL_BUILDINGS.len(),
        ActionType::BuildMonument => ALL_MONUMENTS.len(),
        ActionType::Capture
        | ActionType::Recover
        | ActionType::Promote
        | ActionType::EndTurn
        | ActionType::Disband
        | ActionType::BuildRoad
        | ActionType::Destroy => 0,
    }
}

pub fn arg_len_for(size: u8, kind: ActionType, ruleset: Ruleset) -> usize {
    match kind {
        ActionType::Move | ActionType::Attack => tiles(size),
        ActionType::Train => units(ruleset).len(),
        ActionType::ChooseReward => ALL_REWARDS.len(),
        ActionType::Site => site_actions(ruleset).len(),
        ActionType::Research => techs(ruleset).len(),
        ActionType::BuildBuilding => ALL_BUILDINGS.len(),
        ActionType::BuildMonument => ALL_MONUMENTS.len(),
        ActionType::Capture
        | ActionType::Recover
        | ActionType::Promote
        | ActionType::EndTurn
        | ActionType::Disband
        | ActionType::BuildRoad
        | ActionType::Destroy => 0,
    }
}

/// `state` is the state before the action.
pub fn encode_action(state: &State, action: &Action) -> Factored {
    let size = state.size;
    let n = tiles(size);
    let unit = |id| tile_index(size, state.unit(id).expect("living unit").pos);
    let city = |id| n + tile_index(size, state.city(id).pos);
    let site = |at| 2 * n + tile_index(size, at);
    let (actor, kind, arg) = match *action {
        Action::Move { unit: u, to } => (unit(u), ActionType::Move, tile_index(size, to)),
        Action::Attack { unit: u, target } => {
            (unit(u), ActionType::Attack, tile_index(size, target))
        }
        Action::Capture { unit: u } => (unit(u), ActionType::Capture, 0),
        Action::Recover { unit: u } => (unit(u), ActionType::Recover, 0),
        Action::Promote { unit: u } => (unit(u), ActionType::Promote, 0),
        Action::Disband { unit: u } => (unit(u), ActionType::Disband, 0),
        Action::Train { city: c, kind } => (city(c), ActionType::Train, kind as usize),
        Action::ChooseReward { city: c, reward } => {
            (city(c), ActionType::ChooseReward, reward as usize)
        }
        Action::Site { at, action } => (site(at), ActionType::Site, action as usize),
        Action::BuildRoad { at } => (site(at), ActionType::BuildRoad, 0),
        Action::BuildBuilding { at, building } => {
            (site(at), ActionType::BuildBuilding, building as usize)
        }
        Action::BuildMonument { at, monument } => {
            (site(at), ActionType::BuildMonument, monument as usize)
        }
        Action::Destroy { at } => (site(at), ActionType::Destroy, 0),
        Action::Research { tech } => (3 * n, ActionType::Research, tech as usize),
        Action::EndTurn => (3 * n, ActionType::EndTurn, 0),
    };
    Factored { actor, kind, arg }
}

/// Identification only; legality is checked by `Env::step`.
pub fn decode_action(state: &State, f: &Factored) -> Result<Action, DecodeError> {
    let size = state.size;
    let ruleset = state.config.ruleset;
    let n = tiles(size);
    if f.actor >= num_actors(size) {
        return Err(DecodeError::ActorOutOfRange);
    }
    if (f.kind as usize) >= action_types(ruleset).len() || f.kind.family() != f.actor / n {
        return Err(DecodeError::WrongKind);
    }
    if f.arg >= arg_len_for(size, f.kind, ruleset).max(1) {
        return Err(DecodeError::BadArg);
    }
    let at = tile_pos(size, f.actor % n);
    let arg_pos = || tile_pos(size, f.arg);
    match f.kind.family() {
        0 => {
            let unit = state
                .tile(at)
                .unit
                .and_then(|id| state.unit(id))
                .filter(|u| u.owner == state.current)
                .ok_or(DecodeError::NoActor)?
                .id;
            Ok(match f.kind {
                ActionType::Move => Action::Move {
                    unit,
                    to: arg_pos(),
                },
                ActionType::Attack => Action::Attack {
                    unit,
                    target: arg_pos(),
                },
                ActionType::Capture => Action::Capture { unit },
                ActionType::Recover => Action::Recover { unit },
                ActionType::Promote => Action::Promote { unit },
                ActionType::Disband => Action::Disband { unit },
                _ => unreachable!(),
            })
        }
        1 => {
            let city = state
                .tile(at)
                .city
                .map(|id| state.city(id))
                .filter(|c| c.owner == state.current)
                .ok_or(DecodeError::NoActor)?
                .id;
            Ok(match f.kind {
                ActionType::Train => Action::Train {
                    city,
                    kind: units(ruleset)[f.arg],
                },
                ActionType::ChooseReward => Action::ChooseReward {
                    city,
                    reward: ALL_REWARDS[f.arg],
                },
                _ => unreachable!(),
            })
        }
        2 => Ok(match f.kind {
            ActionType::Site => Action::Site {
                at,
                action: site_actions(ruleset)[f.arg],
            },
            ActionType::BuildRoad => Action::BuildRoad { at },
            ActionType::BuildBuilding => Action::BuildBuilding {
                at,
                building: ALL_BUILDINGS[f.arg],
            },
            ActionType::BuildMonument => Action::BuildMonument {
                at,
                monument: ALL_MONUMENTS[f.arg],
            },
            ActionType::Destroy => Action::Destroy { at },
            _ => unreachable!(),
        }),
        _ => Ok(match f.kind {
            ActionType::Research => Action::Research {
                tech: techs(ruleset)[f.arg],
            },
            ActionType::EndTurn => Action::EndTurn,
            _ => unreachable!(),
        }),
    }
}

/// Built once from a legal-action list.
#[derive(Debug, Clone)]
pub struct ActionMasks {
    size: u8,
    ruleset: Ruleset,
    num_kinds: usize,
    actor: Vec<bool>,
    /// Flat `[actor * num_kinds .. (actor + 1) * num_kinds]`.
    kind: Vec<bool>,
    /// Argument masks of every actor's own kinds, packed by `arg_offset`.
    args: Vec<bool>,
    /// All false; served for kinds outside the actor's family.
    zeros: Vec<bool>,
}

/// Per-actor argument width of each slot family: unit (Move, Attack), city (Train,
/// ChooseReward), site (Site [, BuildBuilding, BuildMonument]), global (Research).
fn family_width(size: u8, family: usize, ruleset: Ruleset) -> usize {
    use ActionType::*;
    match family {
        0 => arg_len_for(size, Move, ruleset) + arg_len_for(size, Attack, ruleset),
        1 => arg_len_for(size, Train, ruleset) + arg_len_for(size, ChooseReward, ruleset),
        2 => {
            let site = arg_len_for(size, Site, ruleset);
            match ruleset {
                Ruleset::V1 => site,
                Ruleset::V2 => {
                    site + arg_len_for(size, BuildBuilding, ruleset)
                        + arg_len_for(size, BuildMonument, ruleset)
                }
            }
        }
        _ => arg_len_for(size, Research, ruleset),
    }
}

/// Start of the argument mask of `(actor, kind)` in `ActionMasks::args`, if `kind` takes an
/// argument and belongs to the actor's family.
fn arg_offset(size: u8, actor: usize, kind: ActionType, ruleset: Ruleset) -> Option<usize> {
    if (kind as usize) >= action_types(ruleset).len() {
        return None;
    }
    let n = tiles(size);
    let family = actor / n;
    if kind.family() != family || arg_len_for(size, kind, ruleset) == 0 {
        return None;
    }
    let base = (0..family)
        .map(|f| n * family_width(size, f, ruleset))
        .sum::<usize>()
        + (actor % n) * family_width(size, family, ruleset);
    let within = match kind {
        ActionType::Attack => arg_len_for(size, ActionType::Move, ruleset),
        ActionType::ChooseReward => arg_len_for(size, ActionType::Train, ruleset),
        ActionType::BuildBuilding => arg_len_for(size, ActionType::Site, ruleset),
        ActionType::BuildMonument => {
            arg_len_for(size, ActionType::Site, ruleset)
                + arg_len_for(size, ActionType::BuildBuilding, ruleset)
        }
        _ => 0,
    };
    Some(base + within)
}

impl ActionMasks {
    pub fn new(state: &State, legal: &[Action]) -> ActionMasks {
        let size = state.size;
        let ruleset = state.config.ruleset;
        let num_kinds = action_types(ruleset).len();
        let n = tiles(size);
        let actors = num_actors(size);
        let total = n
            * (0..3)
                .map(|f| family_width(size, f, ruleset))
                .sum::<usize>()
            + family_width(size, 3, ruleset);
        let mut m = ActionMasks {
            size,
            ruleset,
            num_kinds,
            actor: vec![false; actors],
            kind: vec![false; actors * num_kinds],
            args: vec![false; total],
            zeros: vec![
                false;
                arg_len_for(size, ActionType::Move, ruleset).max(techs(ruleset).len())
            ],
        };
        for a in legal {
            let f = encode_action(state, a);
            m.actor[f.actor] = true;
            m.kind[f.actor * num_kinds + f.kind as usize] = true;
            if let Some(off) = arg_offset(size, f.actor, f.kind, ruleset) {
                m.args[off + f.arg] = true;
            }
        }
        m
    }

    /// Length `num_actors`.
    pub fn actor(&self) -> &[bool] {
        &self.actor
    }

    pub fn kind(&self, actor: usize) -> [bool; NUM_ACTION_TYPES] {
        let mut out = [false; NUM_ACTION_TYPES];
        let start = actor * self.num_kinds;
        out.copy_from_slice(&self.kind[start..start + NUM_ACTION_TYPES]);
        out
    }

    pub fn kind_slice(&self, actor: usize) -> &[bool] {
        let start = actor * self.num_kinds;
        &self.kind[start..start + self.num_kinds]
    }

    /// Length `arg_len_for(size, kind, ruleset)`; empty for kinds without an argument.
    pub fn arg(&self, actor: usize, kind: ActionType) -> &[bool] {
        let len = arg_len_for(self.size, kind, self.ruleset);
        match arg_offset(self.size, actor, kind, self.ruleset) {
            Some(off) => &self.args[off..off + len],
            None => &self.zeros[..len],
        }
    }
}
