//! Creating a game from a generated map (docs/MILESTONE_3 "new_game").

use std::sync::Arc;

use crate::data::{Tables, Tech, UnitType};
use crate::grid::{chebyshev, Pos};
use crate::map::{generate_map, GeneratedMap, MapParams, Site};
use crate::rng::Rng;
use crate::state::{
    City, CityId, Config, Phase, PlayerState, State, TechSet, TileState, Unit, UnitId,
};

/// Builds the starting state. Draws exactly one value from `rng` (the seat coin flip).
pub fn new_game(map: &GeneratedMap, config: &Config, mut rng: Rng) -> State {
    let flip = rng.gen_range(2) as usize;
    let seats = [map.capitals[flip], map.capitals[1 - flip]];
    let tables = Arc::new(Tables::load().expect("embedded tables load"));

    let mut tiles: Vec<TileState> = map
        .tiles
        .iter()
        .map(|t| TileState {
            terrain: t.terrain,
            resource: t.resource,
            improvement: None,
            village: t.site == Site::Village,
            city: None,
            owner_city: None,
            unit: None,
        })
        .collect();

    let n = map.size as usize;
    let mut cities = Vec::new();
    let mut units = Vec::new();
    let mut players = Vec::new();
    for (i, &capital) in seats.iter().enumerate() {
        let city_id = CityId(i as u32);
        let unit_id = UnitId(i as u32);
        cities.push(City {
            id: city_id,
            owner: i as u8,
            pos: capital,
            level: 1,
            population: 0,
            is_capital: true,
            workshop: false,
            parks: 0,
            wall: false,
            border_growth_used: false,
            pending_reward: false,
        });
        let health = tables.unit(UnitType::Warrior).health;
        units.push(Unit {
            id: unit_id,
            kind: UnitType::Warrior,
            owner: i as u8,
            pos: capital,
            health,
            max_health: health,
            origin: Some(city_id),
            kills: 0,
            veteran: false,
            promotion_available: false,
            phase: Phase::Fresh,
            last_dir: None,
        });
        let mut explored = vec![config.full_observability; n * n];
        for (k, tile) in tiles.iter_mut().enumerate() {
            let q = Pos::new((k / n) as u8, (k % n) as u8);
            let d = chebyshev(capital, q);
            // Border radius 1; city 0 claims first, so a shared tile is never rewritten.
            if d <= 1 && tile.owner_city.is_none() {
                tile.owner_city = Some(city_id);
            }
            if d <= 2 {
                explored[k] = true;
            }
        }
        let t = &mut tiles[capital.row as usize * n + capital.col as usize];
        t.city = Some(city_id);
        t.unit = Some(unit_id);
        players.push(PlayerState {
            stars: 5,
            techs: TechSet::from_slice(&[Tech::Organization]),
            explored,
        });
    }
    let players: [PlayerState; 2] = players.try_into().expect("two players");

    State {
        size: map.size,
        turn: 0,
        current: 0,
        rng,
        config: config.clone(),
        tables,
        tiles,
        units,
        cities,
        players,
        next_unit_id: 2,
        outcome: None,
    }
}

pub fn new_game_from_seed(seed: u64, params: &MapParams, config: &Config) -> State {
    let mut rng = Rng::new(seed);
    let map = generate_map(&mut rng, params);
    new_game(&map, config, rng)
}
