//! The game's own deterministic RNG (spec: "RNG state is part of the game state").
//!
//! Algorithm: PCG-XSH-RR 64/32 (O'Neill), the reference "pcg32" with a fixed stream.
//! `Rng::new(seed)` must behave exactly like the reference `pcg32_srandom_r(&rng, seed, 54)`,
//! so the published test vector (seed 42) pins the implementation. No external crates.

/// Cloning an `Rng` clones its future. It must stay `Clone + PartialEq + Eq + Debug`
/// because it lives inside the game state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng {
    // Implementation detail left to the implementer (replace this placeholder).
    _private: (),
}

impl Rng {
    /// Seed with stream (sequence) 54, as in the pcg32 reference demo.
    pub fn new(_seed: u64) -> Rng {
        unimplemented!("milestone 2: pcg32 seeding")
    }

    /// Next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        unimplemented!("milestone 2: pcg32 step")
    }

    /// Uniform in `[0, 1)`: `(next_u32() >> 8) as f32 / 16_777_216.0`. Consumes exactly one
    /// `next_u32`. Exact formula matters for replay stability.
    pub fn gen_f32(&mut self) -> f32 {
        unimplemented!("milestone 2")
    }

    /// Uniform integer in `[0, bound)` with no modulo bias (rejection sampling, as in the pcg32
    /// reference `pcg32_boundedrand_r`). Panics if `bound == 0`.
    pub fn gen_range(&mut self, _bound: u32) -> u32 {
        unimplemented!("milestone 2")
    }
}
