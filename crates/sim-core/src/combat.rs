//! Combat (spec: "Combat"). All arithmetic is f32, evaluated in exactly the order the spec
//! writes the formulas, with half-up rounding.

/// The numbers combat needs about one unit at the moment of the attack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Combatant {
    /// `f32` like `UnitStats.attack` (docs/MILESTONE_15 V2).
    pub attack: f32,
    pub defense: u32,
    pub health: u32,
    pub max_health: u32,
}

/// Raw damage numbers, before health is applied and before the retaliation rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CombatDamage {
    /// `attackResult`: damage dealt to the defender.
    pub to_defender: u32,
    /// `defenseResult`: retaliation damage dealt to the attacker (if retaliation happens).
    pub to_attacker: u32,
}

/// Health of both units after one attack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CombatOutcome {
    pub attacker_health: u32,
    pub defender_health: u32,
}

/// Round to nearest integer, with exact halves going UP. Only called on non-negative values.
pub fn round_half_up(x: f32) -> u32 {
    // f32::round sends halves away from zero, which is half-up for x >= 0.
    x.round() as u32
}

/// Implements the spec formulas literally. Both forces use pre-attack health.
pub fn compute_damage(
    attacker: &Combatant,
    defender: &Combatant,
    defense_bonus: f32,
) -> CombatDamage {
    let attack_force = attacker.attack * (attacker.health as f32 / attacker.max_health as f32);
    let defense_force = defender.defense as f32
        * (defender.health as f32 / defender.max_health as f32)
        * defense_bonus;
    let total_damage = attack_force + defense_force;
    let attack_result = attack_force / total_damage * attacker.attack * 4.5;
    let defense_result = defense_force / total_damage * defender.defense as f32 * 4.5;
    CombatDamage {
        to_defender: round_half_up(attack_result),
        to_attacker: round_half_up(defense_result),
    }
}

/// Applies one attack. The defender always takes `to_defender` (health floors at 0).
/// Retaliation applies only if the defender survives AND `attacker_in_defender_range`.
/// This function does not move units, award kills or promote; it only computes health.
pub fn resolve_attack(
    attacker: &Combatant,
    defender: &Combatant,
    defense_bonus: f32,
    attacker_in_defender_range: bool,
) -> CombatOutcome {
    let damage = compute_damage(attacker, defender, defense_bonus);
    let defender_health = defender.health.saturating_sub(damage.to_defender);
    let attacker_health = if defender_health > 0 && attacker_in_defender_range {
        attacker.health.saturating_sub(damage.to_attacker)
    } else {
        attacker.health
    };
    CombatOutcome {
        attacker_health,
        defender_health,
    }
}
