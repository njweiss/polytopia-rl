//! Seeded map generation (spec: "Map generation"). Copies the Tribes `LevelGenerator` pipeline
//! with water, ruins and lighthouses skipped.

use serde::{Deserialize, Serialize};

use crate::grid::{all_tiles, chebyshev, neighbors, Pos};
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        all_tiles(self.size)
            .filter(|&p| self.tile(p).site == Site::Village)
            .collect()
    }
}

/// Generate a map, drawing all randomness from `rng` (so the caller's RNG state advances and
/// replays are exact). Pipeline and draw order: docs/SPEC.md "Map generation" and
/// docs/MILESTONE_2.md.
pub fn generate_map(rng: &mut Rng, params: &MapParams) -> GeneratedMap {
    let n = params.size;
    let idx = |p: Pos| p.row as usize * n as usize + p.col as usize;
    let all: Vec<Pos> = all_tiles(n).collect();
    let mut tiles = vec![
        Tile {
            terrain: Terrain::Field,
            resource: None,
            site: Site::None,
        };
        all.len()
    ];

    // Capitals: first uniform over inland tiles, second uniform over the inland tiles farthest
    // from the first.
    let inland: Vec<Pos> = all
        .iter()
        .copied()
        .filter(|p| (2..=n - 3).contains(&p.row) && (2..=n - 3).contains(&p.col))
        .collect();
    let cap1 = inland[rng.gen_range(inland.len() as u32) as usize];
    let max_d = inland.iter().map(|&t| chebyshev(t, cap1)).max().unwrap();
    let far: Vec<Pos> = inland
        .iter()
        .copied()
        .filter(|&t| chebyshev(t, cap1) == max_d)
        .collect();
    let cap2 = far[rng.gen_range(far.len() as u32) as usize];
    tiles[idx(cap1)].site = Site::Capital;
    tiles[idx(cap2)].site = Site::Capital;

    // Terrain: one draw per non-capital tile.
    for &p in &all {
        let t = &mut tiles[idx(p)];
        if t.site != Site::None {
            continue;
        }
        let u = rng.gen_f32();
        if u < params.forest_prob {
            t.terrain = Terrain::Forest;
        } else if u > 1.0 - params.mountain_prob {
            t.terrain = Terrain::Mountain;
        }
    }

    // Villages: greedy row-major pass, no randomness.
    let mut sites = vec![cap1, cap2];
    for &p in &all {
        let on_edge = p.row == 0 || p.col == 0 || p.row == n - 1 || p.col == n - 1;
        let t = &mut tiles[idx(p)];
        if !on_edge && t.terrain != Terrain::Mountain && sites.iter().all(|&s| chebyshev(s, p) > 2)
        {
            t.site = Site::Village;
            t.terrain = Terrain::Field;
            sites.push(p);
        }
    }

    // Resources: only within 2 of a site; the outer ring (distance 2) is scaled down.
    for &p in &all {
        let t = &mut tiles[idx(p)];
        if t.site != Site::None {
            continue;
        }
        let d = sites.iter().map(|&s| chebyshev(s, p)).min().unwrap();
        if d > 2 {
            continue;
        }
        let factor = if d == 1 {
            1.0
        } else {
            params.outer_ring_factor
        };
        t.resource = match t.terrain {
            Terrain::Field => {
                let (f, c) = (params.fruit_prob, params.crop_prob);
                if rng.gen_f32() < f * (1.0 - c / 2.0) * factor {
                    Some(Resource::Fruit)
                } else if rng.gen_f32() < c * (1.0 - f / 2.0) * factor {
                    Some(Resource::Crop)
                } else {
                    None
                }
            }
            Terrain::Forest => {
                (rng.gen_f32() < params.animal_prob * factor).then_some(Resource::Animal)
            }
            Terrain::Mountain => {
                (rng.gen_f32() < params.metal_prob * factor).then_some(Resource::Metal)
            }
        };
    }

    // Imperius guarantee: overwrite random capital neighbors with fruit until there are enough.
    for cap in [cap1, cap2] {
        let neighbors: Vec<usize> = neighbors(n, cap).map(idx).collect();
        let fruit_count = |tiles: &[Tile]| {
            neighbors
                .iter()
                .filter(|&&i| tiles[i].resource == Some(Resource::Fruit))
                .count() as u32
        };
        while fruit_count(&tiles) < params.min_capital_fruit {
            let i = neighbors[rng.gen_range(neighbors.len() as u32) as usize];
            tiles[i].terrain = Terrain::Field;
            tiles[i].resource = Some(Resource::Fruit);
        }
    }

    GeneratedMap {
        size: n,
        tiles,
        capitals: [cap1, cap2],
    }
}
