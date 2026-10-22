//! Game state: plain public data holding ground truth (spec: "State definition";
//! docs/MILESTONE_3 "Module layout and public API").

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::data::{Tables, Tech, UnitType};
use crate::grid::Pos;
use crate::map::{Resource, Terrain};
use crate::rng::Rng;

/// 0 or 1. Player 0 always moves first.
pub type PlayerId = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UnitId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CityId(pub u32);

/// Per-unit progress within a turn (spec: "Unit phases").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Fresh,
    Moved,
    Attacked,
    Done,
}

/// A unit's last direction: each component in -1..=1, not both 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dir {
    pub dr: i8,
    pub dc: i8,
}

/// Set of researched techs (bitset indexed by `Tech as usize`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TechSet(u16);

impl TechSet {
    pub fn new() -> TechSet {
        TechSet(0)
    }

    pub fn from_slice(techs: &[Tech]) -> TechSet {
        let mut set = TechSet::new();
        for &t in techs {
            set.insert(t);
        }
        set
    }

    pub fn contains(&self, tech: Tech) -> bool {
        self.0 & (1 << tech as u16) != 0
    }

    pub fn insert(&mut self, tech: Tech) {
        self.0 |= 1 << tech as u16;
    }
}

/// Score weights (spec: "Score"; docs/MILESTONE_6 "Score").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreWeights {
    /// Times the unit's cost in stars.
    pub unit_per_star: u32,
    /// Flat; Super Units have no cost.
    pub super_unit: u32,
    pub territory_per_tile: u32,
    pub exploration_per_tile: u32,
    pub city_base: u32,
    /// Times `level - 1`.
    pub city_per_level: u32,
    /// Times the population above the current level's threshold.
    pub population_surplus: u32,
    pub park: u32,
    /// Times the tier, per researched tech.
    pub tech_per_tier: u32,
}

impl Default for ScoreWeights {
    fn default() -> ScoreWeights {
        ScoreWeights {
            unit_per_star: 5,
            super_unit: 50,
            territory_per_tile: 20,
            exploration_per_tile: 5,
            city_base: 100,
            city_per_level: 50,
            population_surplus: 5,
            park: 250,
            tech_per_tier: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub full_observability: bool,
    pub turn_cap: u32,
    pub score_weights: ScoreWeights,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            full_observability: false,
            turn_cap: 50,
            score_weights: ScoreWeights::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub id: UnitId,
    pub kind: UnitType,
    pub owner: PlayerId,
    pub pos: Pos,
    pub health: u32,
    pub max_health: u32,
    pub origin: Option<CityId>,
    pub kills: u32,
    pub veteran: bool,
    pub promotion_available: bool,
    pub phase: Phase,
    /// One direction per unit (milestone 3, C6).
    pub last_dir: Option<Dir>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct City {
    pub id: CityId,
    pub owner: PlayerId,
    pub pos: Pos,
    pub level: u32,
    pub population: u32,
    pub is_capital: bool,
    pub workshop: bool,
    pub parks: u32,
    pub wall: bool,
    pub border_growth_used: bool,
    /// A level-up reward the owner must choose before anything else (D9).
    pub pending_reward: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Improvement {
    LumberHut,
    Farm,
    Mine,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileState {
    pub terrain: Terrain,
    pub resource: Option<Resource>,
    pub improvement: Option<Improvement>,
    /// Unclaimed village.
    pub village: bool,
    /// A city stands here.
    pub city: Option<CityId>,
    /// Territory; written at most once.
    pub owner_city: Option<CityId>,
    /// Occupant.
    pub unit: Option<UnitId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerState {
    pub stars: u32,
    pub techs: TechSet,
    /// Row-major, length `size * size`. Never shrinks.
    pub explored: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Outcome {
    Win(PlayerId),
    Draw,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    pub size: u8,
    pub turn: u32,
    pub current: PlayerId,
    pub rng: Rng,
    pub config: Config,
    pub tables: Arc<Tables>,
    /// Row-major.
    pub tiles: Vec<TileState>,
    /// Ascending id, dead units removed.
    pub units: Vec<Unit>,
    /// Ascending id, never removed.
    pub cities: Vec<City>,
    pub players: [PlayerState; 2],
    pub next_unit_id: u32,
    pub outcome: Option<Outcome>,
}

impl State {
    pub(crate) fn index(&self, p: Pos) -> usize {
        p.row as usize * self.size as usize + p.col as usize
    }

    pub fn tile(&self, p: Pos) -> &TileState {
        &self.tiles[self.index(p)]
    }

    pub fn unit(&self, id: UnitId) -> Option<&Unit> {
        self.units.iter().find(|u| u.id == id)
    }

    pub fn city(&self, id: CityId) -> &City {
        self.cities
            .iter()
            .find(|c| c.id == id)
            .expect("city ids are never removed")
    }

    pub(crate) fn unit_mut(&mut self, id: UnitId) -> &mut Unit {
        self.units
            .iter_mut()
            .find(|u| u.id == id)
            .expect("living unit")
    }

    pub(crate) fn city_mut(&mut self, id: CityId) -> &mut City {
        self.cities
            .iter_mut()
            .find(|c| c.id == id)
            .expect("city ids are never removed")
    }

    /// Creates a unit at full table health, phase Done, with the next id, on the empty tile `pos`.
    pub(crate) fn add_unit(
        &mut self,
        kind: UnitType,
        owner: PlayerId,
        pos: Pos,
        origin: Option<CityId>,
    ) -> UnitId {
        let health = self.tables.unit(kind).health;
        let id = UnitId(self.next_unit_id);
        self.next_unit_id += 1;
        // Ids only grow, so pushing keeps `units` in ascending id order.
        self.units.push(Unit {
            id,
            kind,
            owner,
            pos,
            health,
            max_health: health,
            origin,
            kills: 0,
            veteran: false,
            promotion_available: false,
            phase: Phase::Done,
            last_dir: None,
        });
        let i = self.index(pos);
        self.tiles[i].unit = Some(id);
        id
    }

    pub fn is_explored(&self, player: PlayerId, p: Pos) -> bool {
        self.players[player as usize].explored[self.index(p)]
    }

    pub fn is_terminal(&self) -> bool {
        self.outcome.is_some()
    }
}
