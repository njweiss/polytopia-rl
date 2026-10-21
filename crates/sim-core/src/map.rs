//! Seeded map generation (spec: "Map generation"). Copies the Tribes `LevelGenerator` pipeline
//! with water, ruins and lighthouses skipped.

use crate::grid::Pos;
use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Terrain {
    Field,
    Forest,
    Mountain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resource {
    Fruit,
    Crop,
    Animal,
    Metal,
}

/// What stands on the tile at game start. Capital and village tiles are always `Terrain::Field`
/// with no resource (see docs/MILESTONE_2.md, clarification C2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Site {
    None,
    Village,
    Capital,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile {
    pub terrain: Terrain,
    pub resource: Option<Resource>,
    pub site: Site,
}

/// Generation parameters. `Default` is the v1 spec / terrainProbs.json (Imperius) values.
#[derive(Debug, Clone, PartialEq)]
pub struct MapParams {
    pub size: u8,
    pub forest_prob: f32,
    pub mountain_prob: f32,
    pub fruit_prob: f32,
    pub crop_prob: f32,
    pub animal_prob: f32,
    pub metal_prob: f32,
    /// Multiplier applied to resource probabilities on outer-ring (distance 2) tiles.
    pub outer_ring_factor: f32,
    /// Each capital's 8 neighbors must hold at least this many fruit.
    pub min_capital_fruit: u32,
}

impl Default for MapParams {
    fn default() -> MapParams {
        MapParams {
            size: 11,
            forest_prob: 0.40,
            mountain_prob: 0.15,
            fruit_prob: 0.5,
            crop_prob: 0.5,
            animal_prob: 0.5,
            metal_prob: 0.5,
            outer_ring_factor: 1.0 / 3.0,
            min_capital_fruit: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedMap {
    pub size: u8,
    /// Row-major, length `size * size`.
    pub tiles: Vec<Tile>,
    /// In generation order: `[first capital, second capital]`. NOT a seat assignment.
    pub capitals: [Pos; 2],
}

impl GeneratedMap {
    pub fn tile(&self, p: Pos) -> &Tile {
        &self.tiles[p.row as usize * self.size as usize + p.col as usize]
    }

    /// All village positions in row-major order.
    pub fn villages(&self) -> Vec<Pos> {
        let n = self.size;
        (0..n)
            .flat_map(|r| (0..n).map(move |c| Pos::new(r, c)))
            .filter(|&p| self.tile(p).site == Site::Village)
            .collect()
    }
}

/// Generate a map, drawing all randomness from `rng` (so the caller's RNG state advances and
/// replays are exact). Pipeline and draw order: docs/SPEC.md "Map generation" and
/// docs/MILESTONE_2.md.
pub fn generate_map(_rng: &mut Rng, _params: &MapParams) -> GeneratedMap {
    unimplemented!("milestone 2")
}
