//! The game's own deterministic RNG (spec: "RNG state is part of the game state").
//!
//! Algorithm: PCG-XSH-RR 64/32 (O'Neill), the reference "pcg32" with a fixed stream.
//! `Rng::new(seed)` must behave exactly like the reference `pcg32_srandom_r(&rng, seed, 54)`,
//! so the published test vector (seed 42) pins the implementation. No external crates.

/// Cloning an `Rng` clones its future. It must stay `Clone + PartialEq + Eq + Debug`
/// because it lives inside the game state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng {
    state: u64,
    inc: u64,
}

const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

impl Rng {
    /// Seed with stream (sequence) 54, as in the pcg32 reference demo.
    pub fn new(seed: u64) -> Rng {
        let mut rng = Rng {
            state: 0,
            inc: (54 << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)`: `(next_u32() >> 8) as f32 / 16_777_216.0`. Consumes exactly one
    /// `next_u32`. Exact formula matters for replay stability.
    pub fn gen_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Uniform integer in `[0, bound)` with no modulo bias (rejection sampling, as in the pcg32
    /// reference `pcg32_boundedrand_r`). Panics if `bound == 0`.
    pub fn gen_range(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "gen_range: bound must be positive");
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % bound;
            }
        }
    }
}
