# Agent workflow (per milestone)

The owner writes the milestone brief (`docs/MILESTONE_N.md`) with Claude in chat. Then three
separate Claude Code sessions, each started fresh from the repo root:

| Phase | Command | Writes | Owner checks |
| --- | --- | --- | --- |
| A. Tests | `/write-tests N` | stubs + tests, committed as `tests(mN): ...` | read the tests and the assumptions list; answer QUESTIONS.md |
| B. Code | `/implement N` | the implementation | read the diff against the spec |
| C. Review (complex milestones only) | `/review N` | nothing (report in chat) | decide which findings are real |

Always `/clear` or start a new session between phases. A phase must work from the repo alone.

## Launch lines
- Phase A: `claude` (default permissions), then `/write-tests N`.
- Phase B: `claude --disallowedTools "Edit(crates/*/tests/**)" "Write(crates/*/tests/**)"`, then
  `/implement N`. Check `/permissions` once to confirm the deny rules loaded. Even if they did not,
  `./scripts/check.sh` fails on any change under `crates/*/tests/` after step 1 of `/implement`.
- Phase C: start `claude` in plan mode (Shift+Tab until it says plan), then `/review N`.

## When to skip phases
- Small, mechanical milestones: A and B in one session is fine (still tests first).
- C only for rules-heavy milestones: movement/legality, economy, capture/win/score, and one full
  pass before RL starts.

## Rules that never change
- The owner approves the tests before phase B starts. Tests are the spec in executable form.
- Tests never change during phase B. A wrong test is a QUESTIONS.md entry, then an owner decision,
  then a new phase A commit.
- Branch per milestone, merge when `check.sh` is green and the owner has read the diff.
