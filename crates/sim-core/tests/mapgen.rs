//! Map generation tests (spec "Map generation > Tests"). Structural checks run on every seed;
//! statistical checks run on a fixed seed range with tolerances of ~4-5 sigma.
//! The checks are written independently of the implementation (own distance, own mark replay).

use std::collections::HashSet;

use sim_core::grid::Pos;
use sim_core::map::{generate_map, GeneratedMap, MapParams, Resource, Site, Terrain};
use sim_core::rng::Rng;

const STRUCT_SEEDS: u64 = 2_000;
const STAT_SEEDS: u64 = 4_000;

fn cheb(a: Pos, b: Pos) -> i32 {
    (a.row as i32 - b.row as i32)
        .abs()
        .max((a.col as i32 - b.col as i32).abs())
}

fn gen(seed: u64) -> GeneratedMap {
    generate_map(&mut Rng::new(seed), &MapParams::default())
}

fn all_pos(n: u8) -> Vec<Pos> {
    (0..n)
        .flat_map(|r| (0..n).map(move |c| Pos::new(r, c)))
        .collect()
}

fn sites(m: &GeneratedMap) -> Vec<Pos> {
    all_pos(m.size)
        .into_iter()
        .filter(|&p| m.tile(p).site != Site::None)
        .collect()
}

fn on_edge(m: &GeneratedMap, p: Pos) -> bool {
    p.row == 0 || p.col == 0 || p.row == m.size - 1 || p.col == m.size - 1
}

// ---------- determinism ----------

#[test]
fn same_seed_same_map_and_same_rng_state() {
    let p = MapParams::default();
    for seed in 0..50 {
        let (mut r1, mut r2) = (Rng::new(seed), Rng::new(seed));
        let (m1, m2) = (generate_map(&mut r1, &p), generate_map(&mut r2, &p));
        assert_eq!(m1, m2);
        assert_eq!(r1, r2, "rng state after generation must match");
        assert_ne!(
            r1,
            Rng::new(seed),
            "generation must consume the caller's rng"
        );
    }
}

#[test]
fn different_seeds_give_different_maps() {
    let distinct: HashSet<Vec<_>> = (0..200).map(|s| gen(s).tiles).collect();
    assert!(
        distinct.len() >= 195,
        "only {} distinct of 200",
        distinct.len()
    );
}

// ---------- structure, every seed ----------

#[test]
fn shape_and_site_counts() {
    for seed in 0..STRUCT_SEEDS {
        let m = gen(seed);
        assert_eq!(m.size, 11);
        assert_eq!(m.tiles.len(), 121);
        let caps = all_pos(11)
            .into_iter()
            .filter(|&p| m.tile(p).site == Site::Capital)
            .count();
        assert_eq!(caps, 2, "seed {seed}");
        assert_eq!(m.tile(m.capitals[0]).site, Site::Capital);
        assert_eq!(m.tile(m.capitals[1]).site, Site::Capital);
        assert_ne!(m.capitals[0], m.capitals[1]);
    }
}

#[test]
fn capitals_are_inland_and_second_is_farthest_from_first() {
    for seed in 0..STRUCT_SEEDS {
        let m = gen(seed);
        for c in m.capitals {
            assert!(
                (2..=8).contains(&c.row) && (2..=8).contains(&c.col),
                "seed {seed} {c:?}"
            );
        }
        let candidates: Vec<Pos> = all_pos(11)
            .into_iter()
            .filter(|p| (2..=8).contains(&p.row) && (2..=8).contains(&p.col))
            .collect();
        let max_d = candidates
            .iter()
            .map(|&t| cheb(t, m.capitals[0]))
            .max()
            .unwrap();
        assert_eq!(cheb(m.capitals[0], m.capitals[1]), max_d, "seed {seed}");
    }
}

#[test]
fn site_tiles_are_plain_field_without_resources() {
    for seed in 0..STRUCT_SEEDS {
        let m = gen(seed);
        for p in sites(&m) {
            let t = m.tile(p);
            assert_eq!(t.terrain, Terrain::Field, "seed {seed} {p:?}");
            assert_eq!(t.resource, None, "seed {seed} {p:?}");
        }
    }
}

#[test]
fn villages_follow_the_row_major_greedy_rule_exactly() {
    for seed in 0..STRUCT_SEEDS {
        let m = gen(seed);
        // Replay: a tile becomes a village iff it is non-edge, not a mountain, and farther than 2
        // (Chebyshev) from every city/village placed so far. Village tiles are Field, so the
        // final terrain is a valid stand-in for "not a mountain".
        let mut placed: Vec<Pos> = m.capitals.to_vec();
        for p in all_pos(11) {
            let eligible = !on_edge(&m, p)
                && m.tile(p).terrain != Terrain::Mountain
                && placed.iter().all(|&s| cheb(s, p) > 2);
            let is_village = m.tile(p).site == Site::Village;
            assert_eq!(is_village, eligible, "seed {seed} {p:?}");
            if eligible {
                placed.push(p);
            }
        }
    }
}

#[test]
fn villages_are_far_from_everything_and_never_on_the_edge() {
    for seed in 0..STRUCT_SEEDS {
        let m = gen(seed);
        let s = sites(&m);
        for (i, &a) in s.iter().enumerate() {
            for &b in &s[i + 1..] {
                assert!(cheb(a, b) >= 3, "seed {seed}: {a:?} {b:?}");
            }
            if m.tile(a).site == Site::Village {
                assert!(!on_edge(&m, a), "seed {seed} {a:?}");
            }
        }
    }
}

#[test]
fn resources_match_terrain_and_stay_near_sites() {
    for seed in 0..STRUCT_SEEDS {
        let m = gen(seed);
        let s = sites(&m);
        for p in all_pos(11) {
            let t = m.tile(p);
            let Some(res) = t.resource else { continue };
            let want = match res {
                Resource::Fruit | Resource::Crop => Terrain::Field,
                Resource::Animal => Terrain::Forest,
                Resource::Metal => Terrain::Mountain,
            };
            assert_eq!(t.terrain, want, "seed {seed} {p:?} {res:?}");
            assert!(
                s.iter().any(|&q| cheb(q, p) <= 2),
                "seed {seed}: resource far from sites at {p:?}"
            );
        }
    }
}

#[test]
fn imperius_guarantee_holds_on_every_map() {
    for seed in 0..STRUCT_SEEDS {
        let m = gen(seed);
        for c in m.capitals {
            let fruit = all_pos(11)
                .into_iter()
                .filter(|&p| cheb(p, c) == 1 && m.tile(p).resource == Some(Resource::Fruit))
                .count();
            assert!(fruit >= 2, "seed {seed}: capital {c:?} has {fruit} fruit");
        }
    }
}

// ---------- parameters are honored ----------

#[test]
fn zero_probabilities_give_plain_land_except_the_guaranteed_fruit() {
    let p = MapParams {
        forest_prob: 0.0,
        mountain_prob: 0.0,
        fruit_prob: 0.0,
        crop_prob: 0.0,
        animal_prob: 0.0,
        metal_prob: 0.0,
        ..MapParams::default()
    };
    for seed in 0..200 {
        let m = generate_map(&mut Rng::new(seed), &p);
        for q in all_pos(11) {
            let t = m.tile(q);
            assert_eq!(t.terrain, Terrain::Field);
            if let Some(r) = t.resource {
                assert_eq!(r, Resource::Fruit);
                assert!(
                    m.capitals.iter().any(|&c| cheb(c, q) == 1),
                    "fruit only from the guarantee"
                );
            }
        }
    }
}

#[test]
fn smaller_map_size_is_respected() {
    let p = MapParams {
        size: 9,
        ..MapParams::default()
    };
    for seed in 0..300 {
        let m = generate_map(&mut Rng::new(seed), &p);
        assert_eq!((m.size, m.tiles.len()), (9, 81));
        for c in m.capitals {
            assert!(
                (2..=6).contains(&c.row) && (2..=6).contains(&c.col),
                "seed {seed} {c:?}"
            );
        }
        for v in m.villages() {
            assert!(!on_edge(&m, v));
        }
    }
}

// ---------- statistics (spec "Known behavior") ----------

#[test]
fn capital_separation_matches_the_exact_distribution() {
    // Exact expectation by enumeration: capital 1 uniform on the 49 inland tiles, capital 2
    // uniform among the tiles at maximum distance from it.
    let cand: Vec<Pos> = all_pos(11)
        .into_iter()
        .filter(|p| (2..=8).contains(&p.row) && (2..=8).contains(&p.col))
        .collect();
    let mut expect = [0f64; 12];
    for &c1 in &cand {
        let max_d = cand.iter().map(|&t| cheb(t, c1)).max().unwrap();
        expect[max_d as usize] += 1.0 / cand.len() as f64;
    }
    let mut got = [0f64; 12];
    for seed in 0..STAT_SEEDS {
        let m = gen(seed);
        got[cheb(m.capitals[0], m.capitals[1]) as usize] += 1.0 / STAT_SEEDS as f64;
    }
    for d in 0..12 {
        assert!(
            (got[d] - expect[d]).abs() < 0.035,
            "distance {d}: got {} want {}",
            got[d],
            expect[d]
        );
    }
    // Sanity against the spec's quoted figures: 3/4/5/6 tiles ~ 2/16/33/49 percent.
    assert!((expect[6] - 0.49).abs() < 0.02 && (expect[3] - 0.02).abs() < 0.01);
}

#[test]
fn village_counts_match_the_spec() {
    let (mut sum, mut min, mut max) = (0usize, usize::MAX, 0usize);
    for seed in 0..STAT_SEEDS {
        let n = gen(seed).villages().len();
        sum += n;
        min = min.min(n);
        max = max.max(n);
    }
    let mean = sum as f64 / STAT_SEEDS as f64;
    assert!((6.5..6.9).contains(&mean), "mean villages {mean}"); // spec: about 6.7
    assert!(min >= 4 && max <= 7, "range {min}..={max}"); // spec: 4 to 7
}

#[test]
fn terrain_shares_match_the_reference_pipeline() {
    // Measured on the reference pipeline over 20k seeds: forest .361, mountain .145, field .494.
    // (Below the raw 0.40 forest probability because villages replace forests and the Imperius
    // guarantee overwrites ring tiles with field.)
    let (mut f, mut mt, mut fi, mut total) = (0u64, 0u64, 0u64, 0u64);
    for seed in 0..STAT_SEEDS {
        let m = gen(seed);
        for t in &m.tiles {
            total += 1;
            match t.terrain {
                Terrain::Forest => f += 1,
                Terrain::Mountain => mt += 1,
                Terrain::Field => fi += 1,
            }
        }
    }
    let share = |x: u64| x as f64 / total as f64;
    assert!((share(f) - 0.361).abs() < 0.015, "forest {}", share(f));
    assert!((share(mt) - 0.145).abs() < 0.010, "mountain {}", share(mt));
    assert!((share(fi) - 0.494).abs() < 0.015, "field {}", share(fi));
}

#[test]
fn resource_rates_match_the_spec_table() {
    // Spec "Resulting rates". Counted over tiles that are not site tiles, are within 2 of a
    // site, and are NOT within 1 of a capital (the Imperius guarantee rewrites those).
    // ring 0 = within 1 of a village ("inner"), ring 1 = otherwise within 2 ("outer").
    let mut field_n = [0u64; 2];
    let (mut fruit, mut crop) = ([0u64; 2], [0u64; 2]);
    let (mut forest_n, mut animal) = ([0u64; 2], [0u64; 2]);
    let (mut mtn_n, mut metal) = ([0u64; 2], [0u64; 2]);
    for seed in 0..STAT_SEEDS {
        let m = gen(seed);
        let s = sites(&m);
        for p in all_pos(11) {
            let t = m.tile(p);
            if t.site != Site::None || m.capitals.iter().any(|&c| cheb(c, p) <= 1) {
                continue;
            }
            let d = s.iter().map(|&q| cheb(q, p)).min().unwrap();
            if d > 2 {
                continue;
            }
            let ring = usize::from(d == 2);
            match t.terrain {
                Terrain::Field => {
                    field_n[ring] += 1;
                    fruit[ring] += u64::from(t.resource == Some(Resource::Fruit));
                    crop[ring] += u64::from(t.resource == Some(Resource::Crop));
                }
                Terrain::Forest => {
                    forest_n[ring] += 1;
                    animal[ring] += u64::from(t.resource == Some(Resource::Animal));
                }
                Terrain::Mountain => {
                    mtn_n[ring] += 1;
                    metal[ring] += u64::from(t.resource == Some(Resource::Metal));
                }
            }
        }
    }
    let rate = |hits: u64, n: u64| hits as f64 / n as f64;
    // (observed, expected) per spec table: inner then outer.
    let checks = [
        ("inner fruit", rate(fruit[0], field_n[0]), 0.375),
        ("inner crop", rate(crop[0], field_n[0]), 0.234),
        ("inner animal", rate(animal[0], forest_n[0]), 0.50),
        ("inner metal", rate(metal[0], mtn_n[0]), 0.50),
        ("outer fruit", rate(fruit[1], field_n[1]), 0.125),
        ("outer crop", rate(crop[1], field_n[1]), 0.109),
        ("outer animal", rate(animal[1], forest_n[1]), 0.167),
        ("outer metal", rate(metal[1], mtn_n[1]), 0.167),
    ];
    for (name, got, want) in checks {
        assert!(
            (got - want).abs() < 0.02,
            "{name}: got {got:.4} want {want}"
        );
    }
}
