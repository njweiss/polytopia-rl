---
description: Phase C. Read-only, spec-keyed review of a milestone's diff
argument-hint: <milestone number>
---
You are the REVIEWER for milestone $ARGUMENTS. You did not write this code. Do not edit any file.
Print the report in the chat.

Inputs: CLAUDE.md, docs/SPEC.md, docs/MILESTONE_$ARGUMENTS.md, and `git diff main...HEAD`.

Produce, in this order:

1. **Traceability table.** For every rule in the brief's "In scope" list and in the spec sections
   it touches: the code that implements it (file:line) and the test that covers it. Mark each row
   COVERED, UNTESTED (code but no test) or MISSING (no code).
2. **Behavior not in the spec.** Anything the diff does that neither the spec nor the brief
   asks for (scope creep, invented rules, silent defaults).
3. **Determinism hazards.** Iteration over HashMap/HashSet that can affect game state, wall-clock
   time, threads, float operations in a different order than the spec writes them, RNG draws in
   data-dependent or unspecified order, randomness not taken from the game's Rng.
4. **Test integrity.** `git diff <test commit> -- 'crates/*/tests'` must be empty; flag any
   `#[ignore]`, loosened tolerance, or special-casing of test inputs in non-test code.
5. **Robustness.** `unwrap`/`expect`/indexing/arithmetic that can panic or overflow on reachable
   inputs in non-test code.
6. **Verdict.** A list of findings, each with severity (blocking / should-fix / nit), file:line,
   and the spec sentence it relates to. Be specific; say "no findings" for a section that has
   none. Do not pad. Do not propose rewrites of working code.

Treat the spec as the authority. If code and spec disagree, that is a finding even if the tests
pass.
