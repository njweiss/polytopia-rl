//! RNG tests. The known-answer vector is the published pcg32 demo output for
//! `pcg32_srandom_r(&rng, 42, 54)` (pcg-random.org), so it is independent of this project.

use sim_core::rng::Rng;

#[test]
fn pcg32_known_answer_vector() {
    let mut r = Rng::new(42);
    let got: Vec<u32> = (0..6).map(|_| r.next_u32()).collect();
    assert_eq!(
        got,
        vec![
            0xa15c_02b7,
            0x7b47_f409,
            0xba1d_3330,
            0x83d2_f293,
            0xbfa4_784b,
            0xcbed_606e
        ]
    );
}

#[test]
fn same_seed_same_stream_different_seed_different_stream() {
    let mut a = Rng::new(7);
    let mut b = Rng::new(7);
    let mut c = Rng::new(8);
    let sa: Vec<u32> = (0..32).map(|_| a.next_u32()).collect();
    let sb: Vec<u32> = (0..32).map(|_| b.next_u32()).collect();
    let sc: Vec<u32> = (0..32).map(|_| c.next_u32()).collect();
    assert_eq!(sa, sb);
    assert_ne!(sa, sc);
}

#[test]
fn clone_clones_the_future_and_equality_tracks_state() {
    let mut a = Rng::new(123);
    a.next_u32();
    let mut b = a.clone();
    assert_eq!(a, b);
    assert_eq!(a.next_u32(), b.next_u32());
    assert_eq!(a, b);
    a.next_u32();
    assert_ne!(a, b);
}

#[test]
fn gen_f32_is_in_unit_interval_and_uses_top_24_bits() {
    let mut a = Rng::new(99);
    let mut shadow = a.clone();
    for _ in 0..10_000 {
        let x = a.gen_f32();
        assert!((0.0..1.0).contains(&x));
        let expect = (shadow.next_u32() >> 8) as f32 / 16_777_216.0;
        assert_eq!(x, expect);
    }
    assert_eq!(a, shadow, "gen_f32 must consume exactly one next_u32");
}

#[test]
fn gen_range_matches_pcg32_bounded_rand() {
    // Reference: threshold = (-bound) % bound; loop { r = next(); if r >= threshold { r % bound } }
    for &bound in &[1u32, 2, 3, 8, 49, 1_000_000, 3_000_000_000] {
        let mut a = Rng::new(2024);
        let mut shadow = a.clone();
        for _ in 0..2_000 {
            let got = a.gen_range(bound);
            let threshold = bound.wrapping_neg() % bound;
            let expect = loop {
                let r = shadow.next_u32();
                if r >= threshold {
                    break r % bound;
                }
            };
            assert_eq!(got, expect, "bound {bound}");
            assert!(got < bound);
        }
        assert_eq!(a, shadow);
    }
}

#[test]
fn gen_range_is_roughly_uniform() {
    let mut r = Rng::new(5);
    let mut counts = [0u32; 6];
    let n = 60_000;
    for _ in 0..n {
        counts[r.gen_range(6) as usize] += 1;
    }
    let expect = n as f64 / 6.0; // 10_000, sigma ~ 91
    for c in counts {
        assert!((c as f64 - expect).abs() < 500.0, "counts {counts:?}");
    }
}

#[test]
#[should_panic]
fn gen_range_zero_bound_panics() {
    Rng::new(1).gen_range(0);
}
