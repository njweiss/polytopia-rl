//! Combat (spec: "Combat"). All arithmetic is f32, evaluated in exactly the order the spec
//! writes the formulas, with half-up rounding.

/// The numbers combat needs about one unit at the moment of the attack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Combatant {
    pub attack: u32,
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
pub fn round_half_up(_x: f32) -> u32 {
    unimplemented!("milestone 1")
}

/// Implements the spec formulas literally. Both forces use pre-attack health.
pub fn compute_damage(
    _attacker: &Combatant,
    _defender: &Combatant,
    _defense_bonus: f32,
) -> CombatDamage {
    unimplemented!("milestone 1")
}

/// Applies one attack. The defender always takes `to_defender` (health floors at 0).
/// Retaliation applies only if the defender survives AND `attacker_in_defender_range`.
/// This function does not move units, award kills or promote; it only computes health.
pub fn resolve_attack(
    _attacker: &Combatant,
    _defender: &Combatant,
    _defense_bonus: f32,
    _attacker_in_defender_range: bool,
) -> CombatOutcome {
    unimplemented!("milestone 1")
}
