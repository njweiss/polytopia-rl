//! Golden combat tests. Source: wiki combat calculator (spec "Combat golden tests").
//! DO NOT edit expected values to make code pass; if one looks wrong, record it in docs/QUESTIONS.md.

use sim_core::combat::{compute_damage, resolve_attack, round_half_up, Combatant};
use sim_core::data::{Tables, UnitType};

fn from_table(t: &Tables, u: UnitType, health: u32, max_health: u32) -> Combatant {
    let s = t.unit(u);
    Combatant {
        attack: s.attack,
        defense: s.defense,
        health,
        max_health,
    }
}

struct Case {
    n: u32,
    attacker: UnitType,
    atk_hp: u32,
    atk_max: u32,
    defender: UnitType,
    def_hp: u32,
    def_max: u32,
    bonus: f32,
    atk_final: u32,
    def_final: u32,
}

#[test]
fn wiki_calculator_cases() {
    use UnitType::*;
    let t = Tables::load().unwrap();
    let cases = [
        Case {
            n: 1,
            attacker: Warrior,
            atk_hp: 10,
            atk_max: 10,
            defender: Warrior,
            def_hp: 10,
            def_max: 10,
            bonus: 1.0,
            atk_final: 5,
            def_final: 5,
        },
        Case {
            n: 2,
            attacker: Warrior,
            atk_hp: 6,
            atk_max: 10,
            defender: Warrior,
            def_hp: 10,
            def_max: 10,
            bonus: 1.0,
            atk_final: 0,
            def_final: 7,
        },
        Case {
            n: 3,
            attacker: Warrior,
            atk_hp: 10,
            atk_max: 10,
            defender: Defender,
            def_hp: 15,
            def_max: 15,
            bonus: 1.0,
            atk_final: 2,
            def_final: 11,
        },
        // Case 4: Rider is a veteran, so max HP is 15 (10 + 5). Defender is damaged: 11/15.
        Case {
            n: 4,
            attacker: Rider,
            atk_hp: 15,
            atk_max: 15,
            defender: Defender,
            def_hp: 11,
            def_max: 15,
            bonus: 1.5,
            atk_final: 7,
            def_final: 8,
        },
        Case {
            n: 5,
            attacker: Defender,
            atk_hp: 15,
            atk_max: 15,
            defender: Archer,
            def_hp: 10,
            def_max: 10,
            bonus: 4.0,
            atk_final: 11,
            def_final: 9,
        },
    ];
    for c in cases {
        let a = from_table(&t, c.attacker, c.atk_hp, c.atk_max);
        let d = from_table(&t, c.defender, c.def_hp, c.def_max);
        // All five golden cases are adjacent fights, so the attacker is in the defender's range.
        let out = resolve_attack(&a, &d, c.bonus, true);
        assert_eq!(out.attacker_health, c.atk_final, "case {} attacker", c.n);
        assert_eq!(out.defender_health, c.def_final, "case {} defender", c.n);
    }
}

#[test]
fn case_1_is_an_exact_half_tie_and_rounds_up() {
    // attackForce = defenseForce = 2, so damage is exactly 2/4 * 2 * 4.5 = 4.5 -> 5 (not 4).
    let w = Combatant {
        attack: 2,
        defense: 2,
        health: 10,
        max_health: 10,
    };
    let dmg = compute_damage(&w, &w, 1.0);
    assert_eq!(dmg.to_defender, 5);
    assert_eq!(dmg.to_attacker, 5);
}

#[test]
fn compute_damage_matches_hand_calculation() {
    // Case 3 by hand: aF=2, dF=3, total=5 -> round(3.6)=4, round(8.1)=8.
    let w = Combatant {
        attack: 2,
        defense: 2,
        health: 10,
        max_health: 10,
    };
    let d = Combatant {
        attack: 1,
        defense: 3,
        health: 15,
        max_health: 15,
    };
    let dmg = compute_damage(&w, &d, 1.0);
    assert_eq!((dmg.to_defender, dmg.to_attacker), (4, 8));
}

#[test]
fn round_half_up_pins_the_tie_rule() {
    assert_eq!(round_half_up(0.0), 0);
    assert_eq!(round_half_up(0.49), 0);
    assert_eq!(round_half_up(0.5), 1);
    assert_eq!(round_half_up(1.5), 2);
    assert_eq!(round_half_up(2.5), 3); // round-half-to-even would give 2
    assert_eq!(round_half_up(4.5), 5);
    assert_eq!(round_half_up(8.1), 8);
    assert_eq!(round_half_up(5.806), 6);
}

#[test]
fn no_retaliation_when_attacker_is_out_of_defender_range() {
    // Archer shoots a Warrior from 2 tiles away: Warrior (range 1) cannot hit back.
    let archer = Combatant {
        attack: 2,
        defense: 1,
        health: 10,
        max_health: 10,
    };
    let warrior = Combatant {
        attack: 2,
        defense: 2,
        health: 10,
        max_health: 10,
    };
    let out = resolve_attack(&archer, &warrior, 1.0, false);
    assert_eq!(out.attacker_health, 10);
    assert!(out.defender_health < 10);
}

#[test]
fn no_retaliation_when_defender_dies() {
    let w = Combatant {
        attack: 2,
        defense: 2,
        health: 10,
        max_health: 10,
    };
    let nearly_dead = Combatant {
        attack: 2,
        defense: 2,
        health: 1,
        max_health: 10,
    };
    let out = resolve_attack(&w, &nearly_dead, 1.0, true);
    assert_eq!(out.defender_health, 0);
    assert_eq!(out.attacker_health, 10);
}

#[test]
fn attacker_health_floors_at_zero_on_retaliation_kill() {
    // Golden case 2 shape: a 6/10 Warrior takes 6 retaliation and must floor at 0 (no underflow).
    let w6 = Combatant {
        attack: 2,
        defense: 2,
        health: 6,
        max_health: 10,
    };
    let w10 = Combatant {
        attack: 2,
        defense: 2,
        health: 10,
        max_health: 10,
    };
    let out = resolve_attack(&w6, &w10, 1.0, true);
    assert_eq!(out.attacker_health, 0);
}

#[test]
fn damage_uses_pre_attack_health_for_both_forces() {
    // Same defender at 15/15 vs 11/15 must change the numbers via health ratio only.
    let rider = Combatant {
        attack: 2,
        defense: 1,
        health: 15,
        max_health: 15,
    };
    let full = Combatant {
        attack: 1,
        defense: 3,
        health: 15,
        max_health: 15,
    };
    let hurt = Combatant {
        attack: 1,
        defense: 3,
        health: 11,
        max_health: 15,
    };
    let d_full = compute_damage(&rider, &full, 1.5);
    let d_hurt = compute_damage(&rider, &hurt, 1.5);
    assert!(d_hurt.to_defender >= d_full.to_defender);
    assert!(d_hurt.to_attacker <= d_full.to_attacker);
}
