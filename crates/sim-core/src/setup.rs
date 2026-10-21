//! Creating a game from a generated map (docs/MILESTONE_3 "new_game").

use crate::map::{generate_map, GeneratedMap, MapParams};
use crate::rng::Rng;
use crate::state::{Config, State};

/// Builds the starting state. Draws exactly one value from `rng` (the seat coin flip).
pub fn new_game(map: &GeneratedMap, config: &Config, rng: Rng) -> State {
    let _ = (map, config, rng);
    unimplemented!()
}

pub fn new_game_from_seed(seed: u64, params: &MapParams, config: &Config) -> State {
    let mut rng = Rng::new(seed);
    let map = generate_map(&mut rng, params);
    new_game(&map, config, rng)
}
