---
description: Phase B. Implement a milestone against its locked tests
argument-hint: <milestone number>
---
You are the IMPLEMENTER for milestone $ARGUMENTS. The tests already exist and are locked.

1. Find the test commit and lock the tests:
   `git log --grep='^tests(m$ARGUMENTS)' -1 --format=%H > "$(git rev-parse --git-dir)/tests-lock"`
   If that prints nothing, stop and tell the owner: the test phase has not been run.
   From now on `./scripts/check.sh` fails if anything under `crates/*/tests/` differs from that
   commit. You must not edit, add, delete or ignore tests to get green.
2. Read CLAUDE.md, docs/WORKFLOW.md, docs/SPEC.md and docs/MILESTONE_$ARGUMENTS.md, then read the
   tests. Give a plan of at most ten lines, then start.
3. Implement only what the brief's "In scope" lists. No extra features, files or dependencies.
   Follow its draw orders and clarifications exactly.
4. Run `./scripts/check.sh` after each meaningful chunk and iterate until it exits 0. Commit at
   green points.
5. If a test seems wrong or contradicts the spec, STOP. Add an entry to docs/QUESTIONS.md (the
   test, the spec text, what you think is right) and report. Do not work around it.
6. Final report: what you built, the last `check.sh` output summary, anything deferred or unsure
   (with QUESTIONS.md numbers), and any behavior you added that the brief did not ask for (there
   should be none).
