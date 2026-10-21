---
description: Phase A. Write stubs and tests for a milestone (no implementation)
argument-hint: <milestone number>
---
You are the TEST AUTHOR for milestone $ARGUMENTS. You do not implement the milestone.

1. Read CLAUDE.md, docs/WORKFLOW.md, docs/SPEC.md and docs/MILESTONE_$ARGUMENTS.md. Work on a branch
   named `m$ARGUMENTS-<short-name>` (create it if needed).
2. Create the stubs: the exact public signatures from the brief, bodies `unimplemented!()`.
   Types, trivial accessors and data files the brief calls "already done" may be real.
3. Write the tests under `crates/*/tests/`. Rules:
   - Expected values come from the sources the brief or spec names (wiki, hand calculation, a
     published test vector, an exact enumeration in the test). Never from running the code under
     test, and never copied from your own reference implementation's output.
   - Checkers in tests are written independently of the implementation (own distance function,
     own replay of the rule), not by calling the function being tested to compute its expectation.
   - Statistical tests: fixed seed ranges, tolerances of about 4-5 sigma, and a comment saying
     where each expected number comes from.
   - Every rule in the brief's "In scope" list is covered by at least one test.
4. Validate the tests. Outside the repo (a scratch directory or `git worktree` you delete after),
   write a throwaway reference implementation and confirm every test passes. Then make at least
   four mutants of the reference, each breaking a different rule from the brief, and confirm each
   is caught by at least one test. If a mutant survives, add or tighten a test. Do not commit the
   reference or the mutants.
5. In the repo, confirm: the stubs compile; every new test FAILS (by panic, not by compile error);
   `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings` are clean.
6. Commit with a message starting `tests(m$ARGUMENTS):`. The implementer session locates the
   test commit by that prefix, so it must be the commit that contains the final tests.
7. Report: test counts per file; the mutation table (mutant, test that caught it); every
   assumption you made; anything ambiguous goes in docs/QUESTIONS.md. Then stop. Do not start the
   implementation.
