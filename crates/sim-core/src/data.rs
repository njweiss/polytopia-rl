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

#[derive(Deserialize)]
struct UnitRow {
    unit: UnitType,
    #[serde(flatten)]
    stats: UnitStats,
}

#[derive(Deserialize)]
struct TechRow {
    tech: Tech,
    #[serde(flatten)]
    info: TechInfo,
}

/// All rules tables, loaded once at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tables {
    /// Indexed by `UnitType as usize` (same order as `ALL_UNITS`).
    units: Vec<UnitStats>,
    /// Indexed by `Tech as usize` (same order as `ALL_TECHS`).
    techs: Vec<TechInfo>,
}

impl Tables {
    /// Load from the embedded `data/units.json` and `data/techs.json`
    /// (use `include_str!`). Must error (not panic) on malformed or incomplete data:
    /// every `UnitType` and every `Tech` must be present exactly once.
    pub fn load() -> Result<Tables, String> {
        Tables::from_json(
            include_str!("../../../data/units.json"),
            include_str!("../../../data/techs.json"),
        )
    }

    /// Same as `load` but from caller-provided JSON (used to test error handling).
    pub fn from_json(units_json: &str, techs_json: &str) -> Result<Tables, String> {
        let unit_rows: Vec<UnitRow> =
            serde_json::from_str(units_json).map_err(|e| format!("units.json: {e}"))?;
        let tech_rows: Vec<TechRow> =
            serde_json::from_str(techs_json).map_err(|e| format!("techs.json: {e}"))?;

        let mut units: Vec<Option<UnitStats>> = vec![None; ALL_UNITS.len()];
        for row in unit_rows {
            let slot = &mut units[row.unit as usize];
            if slot.is_some() {
                return Err(format!("units.json: duplicate {:?}", row.unit));
            }
            *slot = Some(row.stats);
        }
        let mut techs: Vec<Option<TechInfo>> = vec![None; ALL_TECHS.len()];
        for row in tech_rows {
            let slot = &mut techs[row.tech as usize];
            if slot.is_some() {
                return Err(format!("techs.json: duplicate {:?}", row.tech));
            }
            *slot = Some(row.info);
        }

        let units = ALL_UNITS
            .iter()
            .zip(units)
            .map(|(u, s)| s.ok_or_else(|| format!("units.json: missing {u:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        let techs = ALL_TECHS
            .iter()
            .zip(techs)
            .map(|(t, s)| s.ok_or_else(|| format!("techs.json: missing {t:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tables { units, techs })
    }

    pub fn unit(&self, unit: UnitType) -> &UnitStats {
        &self.units[unit as usize]
    }

    pub fn tech(&self, tech: Tech) -> &TechInfo {
        &self.techs[tech as usize]
    }

    /// Spec: cost = 4 + (city_count x tier).
    pub fn tech_cost(&self, tech: Tech, city_count: u32) -> u32 {
        4 + city_count * self.tech(tech).tier
    }
}
