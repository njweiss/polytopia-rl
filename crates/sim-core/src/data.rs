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

/// Tile (site) actions (docs/MILESTONE_4 "Site actions").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteAction {
    HarvestAnimal,
    HarvestFruit,
    LumberHut,
    ClearForest,
    Farm,
    Mine,
}

/// Level-up rewards (docs/MILESTONE_4 "Reward menus").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reward {
    Workshop,
    CityWall,
    Resources,
    PopulationGrowth,
    BorderGrowth,
    Park,
}

pub const ALL_SITE_ACTIONS: [SiteAction; 6] = [
    SiteAction::HarvestAnimal,
    SiteAction::HarvestFruit,
    SiteAction::LumberHut,
    SiteAction::ClearForest,
    SiteAction::Farm,
    SiteAction::Mine,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct SiteActionInfo {
    pub cost: u32,
    pub tech: Tech,
    pub population: u32,
    pub stars: u32,
}

/// Scalar rules constants (`data/rules.json`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rules {
    pub start_stars: u32,
    pub start_techs: Vec<Tech>,
    pub terrain_defense_bonus: f32,
    pub city_defense_bonus: f32,
    pub wall_defense_bonus: f32,
    pub recover_own_territory: u32,
    pub recover_elsewhere: u32,
    pub promotion_kills: u32,
    pub veteran_health_bonus: u32,
}

// Sound because the floats come from JSON, which cannot encode NaN.
impl Eq for Rules {}

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

#[derive(Deserialize)]
struct SiteActionRow {
    action: SiteAction,
    #[serde(flatten)]
    info: SiteActionInfo,
}

/// One reward menu, offered at levels `min_level..=max_level` (`None` = no upper bound).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RewardMenu {
    min_level: u32,
    max_level: Option<u32>,
    options: Vec<Reward>,
}

/// `data/rewards.json`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RewardsFile {
    menus: Vec<RewardMenu>,
    resources_stars: u32,
    population_growth_amount: u32,
}

/// All rules tables, loaded once at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tables {
    /// Indexed by `UnitType as usize` (same order as `ALL_UNITS`).
    units: Vec<UnitStats>,
    /// Indexed by `Tech as usize` (same order as `ALL_TECHS`).
    techs: Vec<TechInfo>,
    /// Indexed by `SiteAction as usize` (same order as `ALL_SITE_ACTIONS`).
    site_actions: Vec<SiteActionInfo>,
    /// Ascending `min_level`, covering every level from 2 up exactly once; the last is open-ended.
    reward_menus: Vec<RewardMenu>,
    resources_reward_stars: u32,
    population_growth_amount: u32,
    rules: Rules,
}

impl Tables {
    /// Load from the embedded `data/units.json`, `data/techs.json`, `data/site_actions.json`,
    /// `data/rewards.json` and `data/rules.json` (use `include_str!`). Must error (not panic) on
    /// malformed or incomplete data: every `UnitType`, `Tech` and `SiteAction` must be present
    /// exactly once, and the reward menus must cover every level from 2 up exactly once.
    pub fn load() -> Result<Tables, String> {
        Tables::from_json(
            include_str!("../../../data/units.json"),
            include_str!("../../../data/techs.json"),
            include_str!("../../../data/site_actions.json"),
            include_str!("../../../data/rewards.json"),
        )
    }

    /// Same as `load` but from caller-provided JSON (used to test error handling).
    /// Rules always come from the embedded `data/rules.json`.
    pub fn from_json(
        units_json: &str,
        techs_json: &str,
        site_actions_json: &str,
        rewards_json: &str,
    ) -> Result<Tables, String> {
        let rules: Rules = serde_json::from_str(include_str!("../../../data/rules.json"))
            .map_err(|e| format!("rules.json: {e}"))?;
        let unit_rows: Vec<UnitRow> =
            serde_json::from_str(units_json).map_err(|e| format!("units.json: {e}"))?;
        let tech_rows: Vec<TechRow> =
            serde_json::from_str(techs_json).map_err(|e| format!("techs.json: {e}"))?;
        let site_rows: Vec<SiteActionRow> = serde_json::from_str(site_actions_json)
            .map_err(|e| format!("site_actions.json: {e}"))?;
        let rewards: RewardsFile =
            serde_json::from_str(rewards_json).map_err(|e| format!("rewards.json: {e}"))?;

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
        let mut site_actions: Vec<Option<SiteActionInfo>> = vec![None; ALL_SITE_ACTIONS.len()];
        for row in site_rows {
            let slot = &mut site_actions[row.action as usize];
            if slot.is_some() {
                return Err(format!("site_actions.json: duplicate {:?}", row.action));
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
        let site_actions = ALL_SITE_ACTIONS
            .iter()
            .zip(site_actions)
            .map(|(a, s)| s.ok_or_else(|| format!("site_actions.json: missing {a:?}")))
            .collect::<Result<Vec<_>, _>>()?;

        // Menus must tile levels 2, 3, ... with no gap or overlap, ending open-ended.
        let mut reward_menus = rewards.menus;
        reward_menus.sort_by_key(|m| m.min_level);
        let mut next = Some(2);
        for m in &reward_menus {
            if next != Some(m.min_level) {
                return Err(format!(
                    "rewards.json: menus must cover every level from 2 up exactly once \
                     (expected a menu starting at {next:?}, found {})",
                    m.min_level
                ));
            }
            if m.max_level.is_some_and(|max| max < m.min_level) {
                return Err(format!(
                    "rewards.json: empty level range at {}",
                    m.min_level
                ));
            }
            next = m.max_level.map(|max| max + 1);
        }
        if next.is_some() {
            return Err("rewards.json: the last menu must be open-ended (max_level null)".into());
        }

        Ok(Tables {
            units,
            techs,
            site_actions,
            reward_menus,
            resources_reward_stars: rewards.resources_stars,
            population_growth_amount: rewards.population_growth_amount,
            rules,
        })
    }

    pub fn unit(&self, unit: UnitType) -> &UnitStats {
        &self.units[unit as usize]
    }

    pub fn tech(&self, tech: Tech) -> &TechInfo {
        &self.techs[tech as usize]
    }

    pub fn rules(&self) -> &Rules {
        &self.rules
    }

    pub fn site_action(&self, a: SiteAction) -> &SiteActionInfo {
        &self.site_actions[a as usize]
    }

    /// The menu offered when a city reaches `level` (2 and up). Empty below level 2.
    pub fn reward_menu(&self, level: u32) -> &[Reward] {
        self.reward_menus
            .iter()
            .find(|m| m.min_level <= level && m.max_level.is_none_or(|max| level <= max))
            .map_or(&[], |m| &m.options)
    }

    pub fn resources_reward_stars(&self) -> u32 {
        self.resources_reward_stars
    }

    pub fn population_growth_amount(&self) -> u32 {
        self.population_growth_amount
    }

    /// Spec: cost = 4 + (city_count x tier).
    pub fn tech_cost(&self, tech: Tech, city_count: u32) -> u32 {
        4 + city_count * self.tech(tech).tier
    }
}
