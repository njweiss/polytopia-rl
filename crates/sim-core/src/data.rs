//! Rules data tables (spec: "Data tables"). Rules live in `data/*.json`, not in code branches.

use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use crate::state::Ruleset;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitType {
    Warrior,
    Rider,
    Archer,
    Defender,
    SuperUnit,
    Swordsman,
    Catapult,
    Knight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Skill {
    Dash,
    Escape,
    Fortify,
    Static,
    Stiff,
    Persist,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Roads,
    FreeSpirit,
    Trade,
    Chivalry,
    Mathematics,
    Construction,
    Smithery,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteAction {
    HarvestAnimal,
    HarvestFruit,
    LumberHut,
    ClearForest,
    Farm,
    Mine,
    BurnForest,
}

/// v2 economy buildings (docs/MILESTONE_15 "Data files").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Building {
    Sawmill,
    Windmill,
    Forge,
    Market,
}

/// v2 monuments (docs/MILESTONE_15 "Data files").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Monument {
    ParkOfFortune,
    GateOfPower,
    EyeOfGod,
}

/// Level-up rewards (docs/MILESTONE_4 "Reward menus", docs/MILESTONE_5 "Menus").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reward {
    Workshop,
    Explorer,
    CityWall,
    Resources,
    PopulationGrowth,
    BorderGrowth,
    Park,
    SuperUnit,
}

pub const ALL_SITE_ACTIONS: [SiteAction; 6] = [
    SiteAction::HarvestAnimal,
    SiteAction::HarvestFruit,
    SiteAction::LumberHut,
    SiteAction::ClearForest,
    SiteAction::Farm,
    SiteAction::Mine,
];

pub const ALL_REWARDS: [Reward; 8] = [
    Reward::Workshop,
    Reward::Explorer,
    Reward::CityWall,
    Reward::Resources,
    Reward::PopulationGrowth,
    Reward::BorderGrowth,
    Reward::Park,
    Reward::SuperUnit,
];

const V2_UNITS: [UnitType; 8] = [
    UnitType::Warrior,
    UnitType::Rider,
    UnitType::Archer,
    UnitType::Defender,
    UnitType::SuperUnit,
    UnitType::Swordsman,
    UnitType::Catapult,
    UnitType::Knight,
];

const V2_TECHS: [Tech; 16] = [
    Tech::Climbing,
    Tech::Hunting,
    Tech::Organization,
    Tech::Riding,
    Tech::Archery,
    Tech::Forestry,
    Tech::Farming,
    Tech::Strategy,
    Tech::Mining,
    Tech::Roads,
    Tech::FreeSpirit,
    Tech::Trade,
    Tech::Chivalry,
    Tech::Mathematics,
    Tech::Construction,
    Tech::Smithery,
];

const V2_SITE_ACTIONS: [SiteAction; 7] = [
    SiteAction::HarvestAnimal,
    SiteAction::HarvestFruit,
    SiteAction::LumberHut,
    SiteAction::ClearForest,
    SiteAction::Farm,
    SiteAction::Mine,
    SiteAction::BurnForest,
];

/// v1: `ALL_UNITS`; v2: all 8 (declaration order).
pub fn units(ruleset: Ruleset) -> &'static [UnitType] {
    match ruleset {
        Ruleset::V1 => &ALL_UNITS,
        Ruleset::V2 => &V2_UNITS,
    }
}

/// v1: `ALL_TECHS`; v2: all 16 (declaration order).
pub fn techs(ruleset: Ruleset) -> &'static [Tech] {
    match ruleset {
        Ruleset::V1 => &ALL_TECHS,
        Ruleset::V2 => &V2_TECHS,
    }
}

/// v1: `ALL_SITE_ACTIONS`; v2: all 7 (declaration order).
pub fn site_actions(ruleset: Ruleset) -> &'static [SiteAction] {
    match ruleset {
        Ruleset::V1 => &ALL_SITE_ACTIONS,
        Ruleset::V2 => &V2_SITE_ACTIONS,
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct UnitStats {
    /// `None` for reward-only units (Super Unit).
    pub cost: Option<u32>,
    pub health: u32,
    /// `f32` for the Knight's 3.5 (docs/MILESTONE_15 V2); v1's integer attacks convert exactly.
    pub attack: f32,
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
impl Eq for UnitStats {}

/// What a building's level counts (`data/v2/buildings.json` `level_from`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelFrom {
    /// The number of adjacent matches.
    Count,
    /// The sum of the adjacent buildings' levels.
    Levels,
}

/// One row of `data/v2/buildings.json` (without the `building` key).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BuildingInfo {
    pub cost: u32,
    pub tech: Tech,
    /// Snake-case names of improvements (`lumber_hut`, `farm`, `mine`) or buildings.
    pub adjacent: Vec<String>,
    pub level_from: LevelFrom,
    pub population_per_level: u32,
    pub stars_per_level: u32,
    /// Caps the output (stars for the market); `None` = no cap.
    pub max_output: Option<u32>,
    pub one_per_city: bool,
}

/// A monument's task (`data/v2/monuments.json` `task`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MonumentTask {
    CityLevel,
    Kills,
    Lighthouses,
}

/// One row of `data/v2/monuments.json` (without the `monument` key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct MonumentInfo {
    pub task: MonumentTask,
    pub threshold: u32,
}

/// v2 scalar rules constants (`data/v2/rules.json`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulesV2 {
    pub road_cost: u32,
    pub road_tech: Tech,
    pub road_step_cost: f32,
    pub connection_population: u32,
    pub disband_tech: Tech,
    pub destroy_tech: Tech,
    pub disband_refund_divisor: u32,
    pub super_unit_value: u32,
    pub lighthouse_population: u32,
    pub explorer_lighthouse_bonus: f32,
    pub monument_population: u32,
    pub monument_score: u32,
}

// Sound because the floats come from JSON, which cannot encode NaN.
impl Eq for RulesV2 {}

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
    ruleset: Ruleset,
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

    /// `V1`: the same as `load()`. `V2`: the v1 files plus `data/v2/*.json`, validated (docs/MILESTONE_15
    /// "Data files"): every variant of `units(V2)`, `techs(V2)`, `site_actions(V2)`, `Building` and
    /// `Monument` exactly once, every tech prerequisite a tech of a lower tier, every `adjacent` name
    /// known. Errors (never panics) on malformed or incomplete data.
    pub fn load_for(ruleset: Ruleset) -> Result<Tables, String> {
        match ruleset {
            Ruleset::V1 => Tables::load(),
            Ruleset::V2 => unimplemented!("milestone 15: Tables::load_for(V2)"),
        }
    }

    /// Same as `load_for(V2)` but with caller-provided `data/v2/*.json` contents (used to test
    /// error handling). The v1 files always come from the embedded `data/*.json`.
    pub fn from_json_v2(
        units_json: &str,
        techs_json: &str,
        site_actions_json: &str,
        buildings_json: &str,
        monuments_json: &str,
        rules_json: &str,
    ) -> Result<Tables, String> {
        let _ = (
            units_json,
            techs_json,
            site_actions_json,
            buildings_json,
            monuments_json,
            rules_json,
        );
        unimplemented!("milestone 15: Tables::from_json_v2")
    }

    pub fn ruleset(&self) -> Ruleset {
        self.ruleset
    }

    /// Panics under `V1`.
    pub fn building(&self, b: Building) -> &BuildingInfo {
        let _ = b;
        unimplemented!("milestone 15: Tables::building")
    }

    /// Panics under `V1`.
    pub fn monument(&self, m: Monument) -> &MonumentInfo {
        let _ = m;
        unimplemented!("milestone 15: Tables::monument")
    }

    /// Panics under `V1`.
    pub fn v2_rules(&self) -> &RulesV2 {
        unimplemented!("milestone 15: Tables::v2_rules")
    }

    /// The embedded tables, parsed once per process and shared by every new game.
    pub(crate) fn shared() -> Arc<Tables> {
        static SHARED: OnceLock<Arc<Tables>> = OnceLock::new();
        SHARED
            .get_or_init(|| Arc::new(Tables::load().expect("embedded tables load")))
            .clone()
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
            ruleset: Ruleset::V1,
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
