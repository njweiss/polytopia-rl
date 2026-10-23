//! Heuristic bots and the match runner (docs/MILESTONE_8). Bots read the game only through
//! `sim_core::view::PlayerView` and are pure functions of that view and their rng stream.

mod economy;
mod random;
mod runner;
mod rush;

use sim_core::actions::{vision_radius, Action};
use sim_core::combat::{compute_damage, Combatant};
use sim_core::data::{Reward, SiteAction, UnitType};
use sim_core::grid::{all_tiles, chebyshev, within, Pos};
use sim_core::rng::Rng;
use sim_core::state::{Phase, PlayerId, Unit, UnitId};
use sim_core::view::{PlayerView, UnitView};

pub use economy::EconomyBot;
pub use random::{random_choice, RandomBot};
pub use runner::{
    bot_policy, play_game, run_match, CheckpointStats, GameResult, MatchStats, PlayerMeans,
    Snapshot,
};
pub use rush::RushBot;

/// Bots are stateless: a pure function of `(view, rng)`.
pub trait Bot {
    fn name(&self) -> &'static str;
    /// Must be in `view.legal_actions()`.
    fn choose(&self, view: &PlayerView, rng: &mut Rng) -> Action;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotKind {
    Random,
    Rush,
    Economy,
}

impl BotKind {
    /// `"random"`, `"rush"` or `"economy"`.
    pub fn name(self) -> &'static str {
        match self {
            BotKind::Random => "random",
            BotKind::Rush => "rush",
            BotKind::Economy => "economy",
        }
    }

    /// The inverse of `name`.
    pub fn from_name(name: &str) -> Option<BotKind> {
        [BotKind::Random, BotKind::Rush, BotKind::Economy]
            .into_iter()
            .find(|k| k.name() == name)
    }
}

/// `epsilon` in `[0, 1]`; with `epsilon > 0` the bot is wrapped (docs/MILESTONE_8 "Random bot and
/// epsilon wrapper"), with `epsilon == 0` it is the bare bot.
pub fn make_bot(kind: BotKind, epsilon: f32) -> Box<dyn Bot> {
    let bot: Box<dyn Bot> = match kind {
        BotKind::Random => Box::new(RandomBot),
        BotKind::Rush => Box::new(RushBot),
        BotKind::Economy => Box::new(EconomyBot),
    };
    if epsilon > 0.0 {
        Box::new(Epsilon { bot, epsilon })
    } else {
        bot
    }
}

/// With probability `epsilon` (one `gen_f32` first) a `RandomBot` action from the same rng, else
/// the wrapped bot's action.
struct Epsilon {
    bot: Box<dyn Bot>,
    epsilon: f32,
}

impl Bot for Epsilon {
    fn name(&self) -> &'static str {
        self.bot.name()
    }

    fn choose(&self, view: &PlayerView, rng: &mut Rng) -> Action {
        if rng.gen_f32() < self.epsilon {
            RandomBot.choose(view, rng)
        } else {
            self.bot.choose(view, rng)
        }
    }
}

const SALT: [u64; 2] = [0x9e37_79b9_7f4a_7c15, 0xc2b2_ae3d_27d4_eb4f];

/// `Rng::new(seed ^ SALT[player])`, `SALT = [0x9e3779b97f4a7c15, 0xc2b2ae3d27d4eb4f]`.
pub fn bot_rng(seed: u64, player: PlayerId) -> Rng {
    Rng::new(seed ^ SALT[player as usize])
}

// Common bot machinery (docs/MILESTONE_8 "Common bot machinery"), shared by Rush and Economy.

/// The first candidate (in order) that is legal.
fn first_legal(legal: &[Action], candidates: impl IntoIterator<Item = Action>) -> Option<Action> {
    candidates.into_iter().find(|a| legal.contains(a))
}

/// The pending city's reward: the preference for its level (`prefs[0]` is level 2, the last entry
/// covers every higher level) if legal, else the first legal `ChooseReward`.
fn choose_reward(view: &PlayerView, legal: &[Action], prefs: &[Reward]) -> Option<Action> {
    let city = view.pending_reward_city()?;
    let level = view
        .my_cities()
        .into_iter()
        .find(|c| c.id == city)
        .expect("own pending city")
        .level;
    let i = (level.saturating_sub(2) as usize).min(prefs.len() - 1);
    first_legal(
        legal,
        [Action::ChooseReward {
            city,
            reward: prefs[i],
        }],
    )
    .or_else(|| {
        legal
            .iter()
            .copied()
            .find(|a| matches!(a, Action::ChooseReward { city: c, .. } if *c == city))
    })
}

/// The legal `Promote` of the lowest unit id (legal actions are ordered by unit id).
fn promote(legal: &[Action]) -> Option<Action> {
    legal
        .iter()
        .copied()
        .find(|a| matches!(a, Action::Promote { .. }))
}

/// The legal `Capture` of the lowest unit id.
fn capture(legal: &[Action]) -> Option<Action> {
    legal
        .iter()
        .copied()
        .find(|a| matches!(a, Action::Capture { .. }))
}

fn own_unit<'a>(view: &PlayerView<'a>, id: UnitId) -> &'a Unit {
    view.my_units()
        .into_iter()
        .find(|u| u.id == id)
        .expect("own unit")
}

fn combatant(view: &PlayerView, kind: UnitType, health: u32, max_health: u32) -> Combatant {
    let stats = view.tables().unit(kind);
    Combatant {
        attack: stats.attack,
        defense: stats.defense,
        health,
        max_health,
    }
}

/// Attack score with defense bonus 1.0: `1000 + dealt` for a kill, else `dealt - taken`.
fn attack_score(view: &PlayerView, unit: UnitId, target: Pos) -> (i64, bool) {
    let a = own_unit(view, unit);
    let d: UnitView = view
        .tile(target)
        .and_then(|t| t.unit)
        .expect("a legal attack has a visible defender");
    let damage = compute_damage(
        &combatant(view, a.kind, a.health, a.max_health),
        &combatant(view, d.kind, d.health, d.max_health),
        1.0,
    );
    let dealt = damage.to_defender;
    if dealt >= d.health {
        return (1000 + dealt as i64, true);
    }
    let in_range = chebyshev(a.pos, target) <= view.tables().unit(d.kind).range;
    let taken = if in_range { damage.to_attacker } else { 0 };
    (dealt as i64 - taken as i64, false)
}

/// The best acceptable attack (a kill, or score `>= threshold`). Legal actions list attacks by
/// attacker id, then target row-major, so the first of the highest score wins ties.
fn best_attack(view: &PlayerView, legal: &[Action], threshold: i64) -> Option<Action> {
    let mut best: Option<(i64, Action)> = None;
    for &a in legal {
        let Action::Attack { unit, target } = a else {
            continue;
        };
        let (score, kill) = attack_score(view, unit, target);
        if !(kill || score >= threshold) {
            continue;
        }
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, a));
        }
    }
    best.map(|(_, a)| a)
}

/// The first legal `Site` of `kind`, tile row-major.
fn site_of_kind(legal: &[Action], kind: SiteAction) -> Option<Action> {
    legal
        .iter()
        .copied()
        .find(|a| matches!(a, Action::Site { action, .. } if *action == kind))
}

fn is_hurt(u: &Unit) -> bool {
    u.health * 2 < u.max_health
}

/// The lowest-id own unit that is hurt, has a legal `Recover`, and has no visible enemy unit
/// within distance 2.
fn recover(view: &PlayerView, legal: &[Action]) -> Option<Action> {
    let enemies: Vec<Pos> = view
        .visible_units()
        .into_iter()
        .filter(|u| u.owner != view.player())
        .map(|u| u.pos)
        .collect();
    view.my_units()
        .into_iter()
        .filter(|u| is_hurt(u))
        .filter(|u| enemies.iter().all(|&e| chebyshev(e, u.pos) > 2))
        .map(|u| Action::Recover { unit: u.id })
        .find(|a| legal.contains(a))
}

/// Unexplored in-bounds tiles within radius 2 of `t` if `t` is a mountain, 1 otherwise. `t` must
/// be explored.
fn vision_gain(view: &PlayerView, t: Pos) -> u32 {
    let r = view.tile(t).map_or(1, |x| vision_radius(x.terrain));
    within(view.size(), t, r)
        .filter(|&q| !view.is_explored(q))
        .count() as u32
}

/// The reachable tile strictly closer to `target` with the smallest distance, ties row-major
/// (`reachable_tiles` is row-major, so the first minimum wins).
fn greedy_step(view: &PlayerView, unit: &Unit, target: Pos) -> Option<Pos> {
    let here = chebyshev(unit.pos, target);
    view.reachable_tiles(unit.id)
        .into_iter()
        .filter(|&t| chebyshev(t, target) < here)
        .min_by_key(|&t| (chebyshev(t, target), t))
}

/// The reachable tile with the largest vision gain > 0, ties row-major; failing that (Q10),
/// `greedy_step` toward the nearest explored tile with vision gain > 0, ties row-major.
fn explore_step(view: &PlayerView, unit: &Unit) -> Option<Pos> {
    let best = view
        .reachable_tiles(unit.id)
        .into_iter()
        .map(|t| (vision_gain(view, t), t))
        .filter(|&(g, _)| g > 0)
        .min_by_key(|&(g, t)| (std::cmp::Reverse(g), t));
    if let Some((_, t)) = best {
        return Some(t);
    }
    let fog = all_tiles(view.size())
        .filter(|&q| view.is_explored(q) && vision_gain(view, q) > 0)
        .min_by_key(|&q| (chebyshev(unit.pos, q), q))?;
    greedy_step(view, unit, fog)
}

/// The nearest of `targets` to `from`, ties row-major.
fn nearest(from: Pos, targets: impl IntoIterator<Item = Pos>) -> Option<Pos> {
    targets.into_iter().min_by_key(|&t| (chebyshev(from, t), t))
}

/// For own Fresh units in ascending id, the first legal `Move` given by `greedy_step` toward
/// `target(unit)` or, if that is `None` or there is no target, `explore_step`.
fn move_toward(
    view: &PlayerView,
    legal: &[Action],
    target: impl Fn(&Unit) -> Option<Pos>,
) -> Option<Action> {
    view.my_units()
        .into_iter()
        .filter(|u| u.phase == Phase::Fresh)
        .find_map(|u| {
            let to = target(u)
                .and_then(|t| greedy_step(view, u, t))
                .or_else(|| explore_step(view, u))?;
            first_legal(legal, [Action::Move { unit: u.id, to }])
        })
}
