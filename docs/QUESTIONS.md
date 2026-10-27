# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q34 are there with pointers to their full text). Number new entries from **Q35**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q35: where and when the `lr_anneal_steps` check applies

- Needed for: docs/MILESTONE_12c "Rules" (`lr_anneal_steps` below `total_steps` raises `ValueError` when the config is loaded)
- Ambiguity: (a) whether the check also applies when `anneal_lr` is false (the horizon is then unused);
  (b) whether a direct `PPOConfig(...)` construction must raise too, or only `PPOConfig.from_json` and
  `load_config` (the two the brief's test 4 names).
- Options: (a) raise regardless of `anneal_lr` / skip when `anneal_lr` is false; (b) check in
  `from_json` (which `load_config` calls after its overrides) / in `__post_init__`.
- Agent's pick and why: the tests pin neither. They check only `from_json` and `load_config`, the latter
  after overrides (a file with `lr_anneal_steps == total_steps` plus `total_steps=2000` raises; plus
  `total_steps=500` loads), and every check case has `anneal_lr` true. I would raise regardless of
  `anneal_lr` (the brief's rule has no condition) and check in `from_json`, so tests building configs
  directly stay free to construct anything.
- Owner answer:
