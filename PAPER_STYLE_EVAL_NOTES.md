# Paper-Style TwoRoom Eval Notes

The local transformer "paper-style" eval result was 34.0% success, but this
should not be treated as a faithful paper reproduction.

## Key Findings

- The run used `wm.history_size=1`, while the repo's default LeWM training
  config uses `wm.history_size=3`.
- With `history_size=1`, training samples only contain
  `num_preds + history_size = 2` frames. Default LeWM uses 4 frames, giving the
  predictor more temporal context.
- The earlier strong local benchmark used `history_size=3`, not 1.
- Git history does not explain the low result: the local repo is current
  `origin/main`, and the train/eval configs plus core scripts are essentially
  unchanged from the initial commit.
- The eval was also much harder than repo defaults:
  - Ran: `goal_offset_steps=100`, `eval_budget=150`, `solver.n_steps=10`
  - Repo default: `goal_offset_steps=25`, `eval_budget=50`,
    `solver.n_steps=30`
- The local `tworoom.h5` dataset has average episode length around 92 and max
  101. With `goal_offset_steps=100`, only 6,056 valid starting points remain,
  so the eval distribution is narrow and different from the default.
- Installed dependencies may differ from the paper/released checkpoint
  environment:
  - `stable-worldmodel 0.0.6`
  - `stable-pretraining 0.1.6`
  - `torch 2.11.0`
  - `lightning 2.6.1`
  - `transformers 5.7.0`

## Recommendation

Rerun the transformer and RWKV comparison using `wm.history_size=3` for
training, then evaluate both models under the same chosen eval settings. The
34.0% transformer result should be considered a mismatched local run, not the
paper baseline.
