//! State digest (docs/MILESTONE_7 "State digest").

use sim_core::data::techs;
use sim_core::grid::Pos;
use sim_core::state::{Outcome, Ruleset, State};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 fed directly with the little-endian byte stream.
struct Fnv(u64);

impl Fnv {
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }
    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }
    fn u16(&mut self, v: u16) {
        self.bytes(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.u8(v as u8);
    }
    fn len(&mut self, n: usize) {
        self.u32(n as u32);
    }
    fn pos(&mut self, p: Pos) {
        self.u8(p.row);
        self.u8(p.col);
    }
    fn opt_u8(&mut self, v: Option<u8>) {
        match v {
            None => self.u8(0),
            Some(x) => {
                self.u8(1);
                self.u8(x);
            }
        }
    }
    fn opt_u32(&mut self, v: Option<u32>) {
        match v {
            None => self.u8(0),
            Some(x) => {
                self.u8(1);
                self.u32(x);
            }
        }
    }
}

/// Every field of `State` in declaration order except `tables`, with `rng` replaced by the next two
/// outputs of a clone. Fixed-size arrays have no length prefix, `TechSet` is its `u16` bitset and
/// `Outcome` is its variant index plus the `Win` payload (docs/QUESTIONS.md Q6).
pub fn state_digest(state: &State) -> u64 {
    let mut h = Fnv(FNV_OFFSET);
    h.u8(state.size);
    h.u32(state.turn);
    h.u8(state.current);
    let mut rng = state.rng.clone();
    h.u32(rng.next_u32());
    h.u32(rng.next_u32());

    let c = &state.config;
    h.bool(c.full_observability);
    h.u32(c.turn_cap);
    let w = &c.score_weights;
    for v in [
        w.unit_per_star,
        w.super_unit,
        w.territory_per_tile,
        w.exploration_per_tile,
        w.city_base,
        w.city_per_level,
        w.population_surplus,
        w.park,
        w.tech_per_tier,
    ] {
        h.u32(v);
    }
    // Only under v2, so v1 digests are unchanged (docs/MILESTONE_15 "Ruleset flag").
    if c.ruleset == Ruleset::V2 {
        h.u8(1);
    }

    h.len(state.tiles.len());
    for t in &state.tiles {
        h.u8(t.terrain as u8);
        h.opt_u8(t.resource.map(|r| r as u8));
        h.opt_u8(t.improvement.map(|i| i as u8));
        h.bool(t.village);
        h.opt_u32(t.city.map(|c| c.0));
        h.opt_u32(t.owner_city.map(|c| c.0));
        h.opt_u32(t.unit.map(|u| u.0));
    }

    h.len(state.units.len());
    for u in &state.units {
        h.u32(u.id.0);
        h.u8(u.kind as u8);
        h.u8(u.owner);
        h.pos(u.pos);
        h.u32(u.health);
        h.u32(u.max_health);
        h.opt_u32(u.origin.map(|c| c.0));
        h.u32(u.kills);
        h.bool(u.veteran);
        h.bool(u.promotion_available);
        h.u8(u.phase as u8);
        match u.last_dir {
            None => h.u8(0),
            Some(d) => {
                h.u8(1);
                h.bytes(&d.dr.to_le_bytes());
                h.bytes(&d.dc.to_le_bytes());
            }
        }
    }

    h.len(state.cities.len());
    for c in &state.cities {
        h.u32(c.id.0);
        h.u8(c.owner);
        h.pos(c.pos);
        h.u32(c.level);
        h.u32(c.population);
        h.bool(c.is_capital);
        h.bool(c.workshop);
        h.u32(c.parks);
        h.bool(c.wall);
        h.bool(c.border_growth_used);
        h.bool(c.pending_reward);
    }

    for pl in &state.players {
        h.u32(pl.stars);
        // All 16 techs, so v2 research is covered (docs/QUESTIONS.md Q42); a v1 set has no v2 bits.
        let bits = techs(Ruleset::V2)
            .iter()
            .filter(|&&t| pl.techs.contains(t))
            .fold(0u16, |acc, &t| acc | 1 << t as u16);
        h.u16(bits);
        h.len(pl.explored.len());
        for &e in &pl.explored {
            h.bool(e);
        }
    }

    h.u32(state.next_unit_id);
    match state.outcome {
        None => h.u8(0),
        Some(Outcome::Win(p)) => {
            h.u8(1);
            h.u8(0);
            h.u8(p);
        }
        Some(Outcome::Draw) => {
            h.u8(1);
            h.u8(1);
        }
    }
    h.0
}
