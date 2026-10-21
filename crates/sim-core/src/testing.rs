//! Test support: `StateBuilder` assembles scripted boards as plain data. It applies no game rules
//! beyond what each method says (no vision, no legality checks).

use std::sync::Arc;

use crate::data::{Tables, Tech, UnitType};
use crate::grid::{chebyshev, Pos};
use crate::map::{Resource, Terrain};
use crate::rng::Rng;
use crate::state::{
    City, CityId, Config, Phase, PlayerId, PlayerState, State, TechSet, TileState, Unit, UnitId,
};

/// Defaults: every tile Field with nothing on it, nothing explored, no techs, 0 stars, turn 0,
/// player 0 to move, `Config::default()`, `Rng::new(0)`.
pub struct StateBuilder {
    state: State,
}

impl StateBuilder {
    pub fn new(size: u8) -> StateBuilder {
        let n = size as usize * size as usize;
        let player = PlayerState {
            stars: 0,
            techs: TechSet::new(),
            explored: vec![false; n],
        };
        let tile = TileState {
            terrain: Terrain::Field,
            resource: None,
            improvement: None,
            village: false,
            city: None,
            owner_city: None,
            unit: None,
        };
        StateBuilder {
            state: State {
                size,
                turn: 0,
                current: 0,
                rng: Rng::new(0),
                config: Config::default(),
                tables: Arc::new(Tables::load().expect("embedded tables load")),
                tiles: vec![tile; n],
                units: Vec::new(),
                cities: Vec::new(),
                players: [player.clone(), player],
                next_unit_id: 0,
                outcome: None,
            },
        }
    }

    fn tile_mut(&mut self, p: Pos) -> &mut TileState {
        let i = p.row as usize * self.state.size as usize + p.col as usize;
        &mut self.state.tiles[i]
    }

    pub fn terrain(mut self, p: Pos, terrain: Terrain) -> StateBuilder {
        self.tile_mut(p).terrain = terrain;
        self
    }

    pub fn resource(mut self, p: Pos, resource: Resource) -> StateBuilder {
        self.tile_mut(p).resource = Some(resource);
        self
    }

    pub fn village(mut self, p: Pos) -> StateBuilder {
        self.tile_mut(p).village = true;
        self
    }

    /// A level 1 city (next city id) that claims every still-unclaimed tile within Chebyshev 1.
    pub fn city(self, owner: PlayerId, p: Pos) -> StateBuilder {
        self.city_with(owner, p, |_| {})
    }

    /// Like `city`, then `edit` may change any field except `id` and `pos`.
    pub fn city_with(
        mut self,
        owner: PlayerId,
        p: Pos,
        edit: impl FnOnce(&mut City),
    ) -> StateBuilder {
        let id = CityId(self.state.cities.len() as u32);
        let mut city = City {
            id,
            owner,
            pos: p,
            level: 1,
            population: 0,
            is_capital: false,
            workshop: false,
            parks: 0,
            wall: false,
            border_growth_used: false,
            pending_reward: false,
        };
        edit(&mut city);
        self.state.cities.push(city);
        self.tile_mut(p).city = Some(id);
        let size = self.state.size;
        for r in 0..size {
            for c in 0..size {
                let q = Pos::new(r, c);
                if chebyshev(p, q) <= 1 && self.tile_mut(q).owner_city.is_none() {
                    self.tile_mut(q).owner_city = Some(id);
                }
            }
        }
        self
    }

    /// A unit (next unit id) at full table health, phase Fresh, no origin, kills or direction.
    pub fn unit(self, owner: PlayerId, kind: UnitType, p: Pos) -> StateBuilder {
        self.unit_with(owner, kind, p, |_| {})
    }

    /// Like `unit`, then `edit` may change any field except `id`, `owner` and `pos`.
    pub fn unit_with(
        mut self,
        owner: PlayerId,
        kind: UnitType,
        p: Pos,
        edit: impl FnOnce(&mut Unit),
    ) -> StateBuilder {
        assert!(self.tile_mut(p).unit.is_none(), "two units on {p:?}");
        let id = UnitId(self.state.next_unit_id);
        let health = self.state.tables.unit(kind).health;
        let mut unit = Unit {
            id,
            kind,
            owner,
            pos: p,
            health,
            max_health: health,
            origin: None,
            kills: 0,
            veteran: false,
            promotion_available: false,
            phase: Phase::Fresh,
            last_dir: None,
        };
        edit(&mut unit);
        self.state.units.push(unit);
        self.state.next_unit_id += 1;
        self.tile_mut(p).unit = Some(id);
        self
    }

    pub fn techs(mut self, player: PlayerId, techs: &[Tech]) -> StateBuilder {
        self.state.players[player as usize].techs = TechSet::from_slice(techs);
        self
    }

    pub fn stars(mut self, player: PlayerId, stars: u32) -> StateBuilder {
        self.state.players[player as usize].stars = stars;
        self
    }

    /// Both players have explored every tile.
    pub fn explore_all(mut self) -> StateBuilder {
        for p in &mut self.state.players {
            p.explored.iter_mut().for_each(|e| *e = true);
        }
        self
    }

    pub fn explore(mut self, player: PlayerId, tiles: &[Pos]) -> StateBuilder {
        let size = self.state.size as usize;
        for p in tiles {
            self.state.players[player as usize].explored[p.row as usize * size + p.col as usize] =
                true;
        }
        self
    }

    /// Marks tiles unexplored for `player` (useful after `explore_all`).
    pub fn hide(mut self, player: PlayerId, tiles: &[Pos]) -> StateBuilder {
        let size = self.state.size as usize;
        for p in tiles {
            self.state.players[player as usize].explored[p.row as usize * size + p.col as usize] =
                false;
        }
        self
    }

    pub fn current(mut self, player: PlayerId) -> StateBuilder {
        self.state.current = player;
        self
    }

    pub fn turn(mut self, turn: u32) -> StateBuilder {
        self.state.turn = turn;
        self
    }

    /// With `full_observability`, `build` marks every tile explored for both players.
    pub fn config(mut self, config: Config) -> StateBuilder {
        self.state.config = config;
        self
    }

    pub fn rng(mut self, rng: Rng) -> StateBuilder {
        self.state.rng = rng;
        self
    }

    pub fn build(self) -> State {
        if self.state.config.full_observability {
            return self.explore_all().state;
        }
        self.state
    }
}
