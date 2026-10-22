//! A fog-restricted, read-only view of the game for one player (docs/MILESTONE_8 "PlayerView
//! visibility"). Bots read the game only through this.

use crate::actions::Action;
use crate::data::{Tables, UnitType};
use crate::grid::Pos;
use crate::map::{Resource, Terrain};
use crate::state::{City, CityId, Improvement, Phase, PlayerId, State, TechSet, Unit, UnitId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CityView {
    pub id: CityId,
    pub owner: PlayerId,
    pub pos: Pos,
    pub level: u32,
    pub is_capital: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitView {
    /// `Some` only for the viewer's own units.
    pub id: Option<UnitId>,
    pub kind: UnitType,
    pub owner: PlayerId,
    pub pos: Pos,
    pub health: u32,
    pub max_health: u32,
    pub veteran: bool,
    pub phase: Phase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleTile {
    pub terrain: Terrain,
    /// A crop is hidden without Organization, a metal without Climbing.
    pub resource: Option<Resource>,
    pub improvement: Option<Improvement>,
    pub village: bool,
    pub city: Option<CityView>,
    /// The owner of the tile's owning city.
    pub territory: Option<PlayerId>,
    pub unit: Option<UnitView>,
}

/// Never hands out `&State`, unexplored contents, the opponent's hidden data or the rng.
pub struct PlayerView<'a> {
    state: &'a State,
    player: PlayerId,
}

impl<'a> PlayerView<'a> {
    pub fn new(state: &'a State, player: PlayerId) -> PlayerView<'a> {
        PlayerView { state, player }
    }

    pub fn player(&self) -> PlayerId {
        self.player
    }

    pub fn size(&self) -> u8 {
        unimplemented!()
    }

    pub fn turn(&self) -> u32 {
        unimplemented!()
    }

    pub fn turn_cap(&self) -> u32 {
        unimplemented!()
    }

    pub fn current(&self) -> PlayerId {
        unimplemented!()
    }

    pub fn is_terminal(&self) -> bool {
        unimplemented!()
    }

    pub fn is_explored(&self, _p: Pos) -> bool {
        let _ = self.state;
        unimplemented!()
    }

    /// `None` if unexplored.
    pub fn tile(&self, _p: Pos) -> Option<VisibleTile> {
        unimplemented!()
    }

    pub fn stars(&self) -> u32 {
        unimplemented!()
    }

    pub fn techs(&self) -> TechSet {
        unimplemented!()
    }

    pub fn income(&self) -> u32 {
        unimplemented!()
    }

    /// Both players' scores are public.
    pub fn score(&self, _player: PlayerId) -> u32 {
        unimplemented!()
    }

    /// Ascending id.
    pub fn my_units(&self) -> Vec<&'a Unit> {
        unimplemented!()
    }

    /// Ascending id.
    pub fn my_cities(&self) -> Vec<&'a City> {
        unimplemented!()
    }

    /// Every unit on an explored tile, row-major.
    pub fn visible_units(&self) -> Vec<UnitView> {
        unimplemented!()
    }

    /// Own cities and enemy cities on explored tiles, by id.
    pub fn visible_cities(&self) -> Vec<CityView> {
        unimplemented!()
    }

    pub fn tables(&self) -> &'a Tables {
        unimplemented!()
    }

    /// `legal_actions(state)` if it is this player's turn, else empty.
    pub fn legal_actions(&self) -> Vec<Action> {
        unimplemented!()
    }

    /// Own units only; panics otherwise.
    pub fn reachable_tiles(&self, _unit: UnitId) -> Vec<Pos> {
        unimplemented!()
    }

    /// The viewer's city with a pending reward.
    pub fn pending_reward_city(&self) -> Option<CityId> {
        unimplemented!()
    }
}
