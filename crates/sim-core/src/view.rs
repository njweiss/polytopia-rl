//! A fog-restricted, read-only view of the game for one player (docs/MILESTONE_8 "PlayerView
//! visibility"). Bots read the game only through this.

use crate::actions::{legal_actions, Action};
use crate::data::{Tables, Tech, UnitType};
use crate::economy::income;
use crate::grid::{all_tiles, Pos};
use crate::map::{Resource, Terrain};
use crate::movement::reachable_tiles;
use crate::score::score;
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
        self.state.size
    }

    pub fn turn(&self) -> u32 {
        self.state.turn
    }

    pub fn turn_cap(&self) -> u32 {
        self.state.config.turn_cap
    }

    pub fn current(&self) -> PlayerId {
        self.state.current
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    pub fn is_explored(&self, p: Pos) -> bool {
        self.state.is_explored(self.player, p)
    }

    /// `None` if unexplored.
    pub fn tile(&self, p: Pos) -> Option<VisibleTile> {
        if !self.is_explored(p) {
            return None;
        }
        let t = self.state.tile(p);
        let techs = self.techs();
        let resource = t.resource.filter(|&r| resource_visible(&techs, r));
        Some(VisibleTile {
            terrain: t.terrain,
            resource,
            improvement: t.improvement,
            village: t.village,
            city: t.city.map(|c| self.city_view(self.state.city(c))),
            territory: t.owner_city.map(|c| self.state.city(c).owner),
            unit: t
                .unit
                .map(|u| self.unit_view(self.state.unit(u).expect("occupant is alive"))),
        })
    }

    pub fn stars(&self) -> u32 {
        self.state.players[self.player as usize].stars
    }

    pub fn techs(&self) -> TechSet {
        self.state.players[self.player as usize].techs
    }

    pub fn income(&self) -> u32 {
        income(self.state, self.player)
    }

    /// Both players' scores are public.
    pub fn score(&self, player: PlayerId) -> u32 {
        score(self.state, player)
    }

    /// Ascending id.
    pub fn my_units(&self) -> Vec<&'a Unit> {
        self.state
            .units
            .iter()
            .filter(|u| u.owner == self.player)
            .collect()
    }

    /// Ascending id.
    pub fn my_cities(&self) -> Vec<&'a City> {
        self.state
            .cities
            .iter()
            .filter(|c| c.owner == self.player)
            .collect()
    }

    /// Every unit on an explored tile, row-major.
    pub fn visible_units(&self) -> Vec<UnitView> {
        all_tiles(self.state.size)
            .filter_map(|p| self.tile(p).and_then(|t| t.unit))
            .collect()
    }

    /// Own cities and enemy cities on explored tiles, by id.
    pub fn visible_cities(&self) -> Vec<CityView> {
        self.state
            .cities
            .iter()
            .filter(|c| c.owner == self.player || self.is_explored(c.pos))
            .map(|c| self.city_view(c))
            .collect()
    }

    pub fn tables(&self) -> &'a Tables {
        &self.state.tables
    }

    /// `legal_actions(state)` if it is this player's turn, else empty.
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.state.current != self.player {
            return Vec::new();
        }
        legal_actions(self.state)
    }

    /// Own units only; panics otherwise.
    pub fn reachable_tiles(&self, unit: UnitId) -> Vec<Pos> {
        let own = self
            .state
            .unit(unit)
            .is_some_and(|u| u.owner == self.player);
        assert!(own, "reachable_tiles: {unit:?} is not the viewer's unit");
        reachable_tiles(self.state, unit)
    }

    /// The viewer's city with a pending reward.
    pub fn pending_reward_city(&self) -> Option<CityId> {
        self.state
            .cities
            .iter()
            .find(|c| c.owner == self.player && c.pending_reward)
            .map(|c| c.id)
    }

    fn city_view(&self, c: &City) -> CityView {
        CityView {
            id: c.id,
            owner: c.owner,
            pos: c.pos,
            level: c.level,
            is_capital: c.is_capital,
        }
    }

    fn unit_view(&self, u: &Unit) -> UnitView {
        UnitView {
            id: (u.owner == self.player).then_some(u.id),
            kind: u.kind,
            owner: u.owner,
            pos: u.pos,
            health: u.health,
            max_health: u.max_health,
            veteran: u.veteran,
            phase: u.phase,
        }
    }
}

/// A crop needs Organization and a metal Climbing to be seen; other resources are always seen.
pub fn resource_visible(techs: &TechSet, r: Resource) -> bool {
    match r {
        Resource::Crop => techs.contains(Tech::Organization),
        Resource::Metal => techs.contains(Tech::Climbing),
        Resource::Fruit | Resource::Animal => true,
    }
}
