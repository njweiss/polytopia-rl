# Agent workflow (per milestone)

The owner writes the milestone brief (`docs/MILESTONE_N.md`) with the orchestrator session. Then
three separate Claude Code sessions run from the repo root (or a git worktree), using the slash
commands in `.claude/commands/`:

| Phase | Command | Writes | Owner checks |
| --- | --- | --- | --- |
| A. Tests | `/write-tests N` | stubs + tests (`crates/*/tests/` and/or `python/tests/rl/`), committed as `tests(mN): ...` | read the tests and assumptions; answer `docs/QUESTIONS.md` |
| B. Code | `/implement N` | the implementation | read the diff against the spec |
| C. Review (complex milestones) | `/review N` | nothing (report in chat) | decide which findings are real |

Always `/clear` or start a new session between phases. Each phase must work from the repo alone.

## Launch lines

- **Phase A:** `claude` (default permissions), then `/write-tests N`.
- **Phase B:** `claude --disallowedTools "Edit(crates/*/tests/**)" "Write(crates/*/tests/**)" "Edit(python/tests/rl/**)" "Write(python/tests/rl/**)"`,
  then `/implement N`. Check `/permissions` once to confirm the deny rules loaded. Even if they did
  not, `pixi run check` (`scripts/check.sh`) fails on any change under `crates/*/tests/` or
  `python/tests/rl/` after step 1 of `/implement` writes `.git/tests-lock`.
- **Phase C:** start `claude` in plan mode (Shift+Tab until it says plan), then `/review N`.

## Worktrees and the check gate

`pixi run check` (`scripts/check.sh`) is worktree-safe: in a git worktree, `scripts/py_check.sh`
builds an isolated wheel into `target/pyext` and puts it on `PYTHONPATH` for `py_check.sh` and
`rl_check.sh`, leaving the main checkout's `polytopia_sim` installation untouched.

## When to skip phases

- Small, mechanical milestones: A and B in one session is fine (still tests first).
- Phase C (`/review N`): use for rules-heavy or architecturally complex milestones.

## Rules that never change

- The owner approves the tests before phase B starts. Tests are the spec in executable form.
- Tests never change during phase B. A wrong test is a `docs/QUESTIONS.md` entry, then an owner
  decision, then a new phase A commit.
- Branch per milestone, merge when `pixi run check` is green and the owner has reviewed the diff.
