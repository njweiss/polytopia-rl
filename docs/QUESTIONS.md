# Open questions from agents

Only open questions live here, plus the answered ones of the milestone in progress (the implementer
needs their full text). At the end of each milestone, answered entries move to `docs/DECISIONS.md` as one
line (Q1-Q58 are there with pointers to their full text). Number new entries from **Q59**.

Append entries below. Format:

## Q<n>: <short title>

- Needed for: <milestone / spec section>
- Ambiguity: <what the spec does not say>
- Options: <A / B>
- Agent's pick and why: <...>
- Owner answer: (blank until answered)

## Q59: Batch-size mismatch in `test_transfer.py:693` (`teacher.arg_logits(tf, actors[:5], kinds[:5])`)

- Needed for: Milestone 22 (`python/tests/rl/test_transfer.py::test_distill_v1_to_v2_arch_v2_mixed_batch_and_all_edge_cases`, line 693)
- Ambiguity: Line 690 computes `tf = teacher.trunk(board[:, : v1_spec["channels"]], glob[:, : v1_spec["globals"]])` with batch size `b = 14`, and line 693 passes the full 14-row `tf` alongside 5-row slices `actors[:5], kinds[:5]` into `teacher.arg_logits(tf, actors[:5], kinds[:5])`. In `PolicyNet` (`python/polytopia_rl/model.py`), `ActorContext.forward` (`torch.cat([at_tile, feats.g, self.family(family)], dim=1)`) and `ArgHead.forward` (`einsum("bwn,bw->bn", keys, c)`) require `feats` and `actors`/`kinds` to have the same batch size, so line 693 raises `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 5 but got size 14` before `assert_close` is even reached. (Every other assertion in `test_transfer.py`, 27/28 tests, passes.)
- Options:
  - A (recommended): Fix line 693 in `python/tests/rl/test_transfer.py` (and re-lock `.git/tests-lock`) to slice `tf` to the first 5 rows (`tf5 = Features(tf.fmap[:5], tf.g[:5])` or `teacher.trunk(board[:5, : v1_spec["channels"]], glob[:5, : v1_spec["globals"]])`) before calling `teacher.arg_logits(tf5, actors[:5], kinds[:5])`.
  - B: Change `ActorContext.forward` and `ArgHead.forward` in `model.py` to silently slice `feats.fmap[:B]` / `feats.g[:B]` to `actors.shape[0]`. (Not recommended: silently hides real batch-size mismatches in production code.)
- Agent's pick and why: Option A. The test clearly intended to compare `t_logits[2][:5]` against the teacher's direct `arg_logits` on the 5 v1 rows (`0..4`), and `Features` has never supported implicit prefix slicing.
- Owner answer: A (owner, 2026-11-28) — slice `tf` to the first 5 rows (`tf5 = Features(tf.fmap[:5], tf.g[:5])`) on line 693 before calling `teacher.arg_logits`.
