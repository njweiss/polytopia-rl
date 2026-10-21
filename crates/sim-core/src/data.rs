//! Rules data tables (spec: "Data tables"). Rules live in `data/*.json`, not in code branches.

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitType {
    Warrior,
    Rider,
    Archer,
    Defender,
    SuperUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Skill {
    Dash,
    Escape,
    Fortify,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tech {
    Climbing,
    Hunting,
    Organization,
    Riding,
    Archery,
    Forestry,
    Farming,
    Strategy,
    Mining,
}

pub const ALL_UNITS: [UnitType; 5] = [
    UnitType::Warrior,
    UnitType::Rider,
    UnitType::Archer,
    UnitType::Defender,
    UnitType::SuperUnit,
];

pub const ALL_TECHS: [Tech; 9] = [
    Tech::Climbing,
    Tech::Hunting,
    Tech::Organization,
    Tech::Riding,
    Tech::Archery,
    Tech::Forestry,
    Tech::Farming,
    Tech::Strategy,
    Tech::Mining,
];

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct UnitStats {
    /// `None` for reward-only units (Super Unit).
    pub cost: Option<u32>,
    pub health: u32,
    pub attack: u32,
    pub defense: u32,
    pub movement: u32,
    pub range: u32,
    pub skills: Vec<Skill>,
    /// `None` = available without a tech (Warrior) or not trainable (Super Unit).
    pub unlocked_by: Option<Tech>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct TechInfo {
    pub tier: u32,
    pub requires: Option<Tech>,
}

/// All rules tables, loaded once at startup.
#[derive(Debug, Clone)]
pub struct Tables {
    // Implementation detail left to the implementer.
}

impl Tables {
    /// Load from the embedded `data/units.json` and `data/techs.json`
    /// (use `include_str!`). Must error (not panic) on malformed or incomplete data:
    /// every `UnitType` and every `Tech` must be present exactly once.
    pub fn load() -> Result<Tables, String> {
        unimplemented!("milestone 1: parse data/units.json and data/techs.json")
    }

    /// Same as `load` but from caller-provided JSON (used to test error handling).
    pub fn from_json(_units_json: &str, _techs_json: &str) -> Result<Tables, String> {
        unimplemented!("milestone 1")
    }

    pub fn unit(&self, _unit: UnitType) -> &UnitStats {
        unimplemented!("milestone 1")
    }

    pub fn tech(&self, _tech: Tech) -> &TechInfo {
        unimplemented!("milestone 1")
    }

    /// Spec: cost = 4 + (city_count x tier).
    pub fn tech_cost(&self, _tech: Tech, _city_count: u32) -> u32 {
        unimplemented!("milestone 1")
    }
}
