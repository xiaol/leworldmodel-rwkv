# RWKV-7 vs Transformer Comparison

## Final default TwoRoom eval

This is the current apples-to-apples local comparison using the repo default
TwoRoom evaluation, excluding the mismatched `history_size=1` comparison run.

Training setup for the local transformer and RWKV rows:

- Full TwoRoom training data
- 10 epochs
- Batch size 128
- AdamW, learning rate `5e-5`, weight decay `1e-3`
- `wm.history_size=3`
- ViT-tiny encoder trained from scratch

Evaluation setup:

- `eval_budget=50`
- `goal_offset_steps=25`
- `solver.num_samples=300`
- `solver.n_steps=30`
- `solver.topk=30`
- `eval.num_eval=50`

| Model | Depth | Total params | Predictor params | TwoRoom success | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper/released LeWM checkpoint | 6 | 18.03M | 10.79M | 84.0% | 170.3s |
| Transformer | 6 | 18.03M | 10.79M | 90.0% | 196.2s |
| RWKV-7 CUDA | 11 | 18.22M | 10.98M | 92.0% | 240.2s |

Artifacts:

```text
/home/xiaol/.stable_worldmodel/checkpoints/tworoom_results.txt
/home/xiaol/.stable_worldmodel/datasets/quentinll--lewm-tworooms/compare_rwkv7_repo_history3_tworoom_cuda/summary.json
```

Conclusion: under the default local TwoRoom evaluation, the parameter-matched
RWKV-7 model is slightly better than the locally trained transformer on success
rate, but it is slower in evaluation wall time. The released LeWM checkpoint is
included as a reference baseline.

## LeWM resource breakdown

For the locally trained transformer LeWM checkpoint:

| Component | Params | Share |
| --- | ---: | ---: |
| Predictor | 10.79M | 59.8% |
| ViT encoder | 5.50M | 30.5% |
| Projector | 0.79M | 4.4% |
| Prediction projector | 0.79M | 4.4% |
| Action encoder | 0.16M | 0.9% |

The predictor is the largest component by parameter count. During evaluation,
the biggest wall-time cost is CEM planning, because it repeatedly calls the
world model over many action samples. The ViT encoder is also a major runtime
cost because it encodes pixels during train and eval.

## Long-horizon eval caveat

The attempted long-horizon setting used:

```text
goal_offset_steps=100
eval_budget=150
solver.n_steps=10
```

On the local `tworoom.h5`, `goal_offset_steps=100` leaves only 6,056 valid
starts, all at step 0 of 101-step episodes. This produces a much narrower and
harder local distribution. The released LeWM checkpoint also performs poorly
under this local long-horizon setting, so those numbers should not be treated
as the paper/released-checkpoint baseline.

## Short 2ep100b run

This run is training-budget matched, but not parameter matched:

```bash
python compare_predictors.py \
  --run-name compare_rwkv7_transformer_2ep100b \
  --rwkv-config rwkv7 \
  --rwkv-name rwkv7 \
  --train-epochs 2 \
  --train-batches 100 \
  --val-batches 10 \
  --batch-size 64 \
  --num-workers 4
```

Results are saved in:

```text
/home/xiaol/.stable_worldmodel/datasets/quentinll--lewm-tworooms/compare_rwkv7_transformer_2ep100b/summary.json
```

| Model | Total params | Predictor params | TwoRoom success | Eval time |
| --- | ---: | ---: | ---: | ---: |
| Transformer, depth 6 | 18.03M | 10.79M | 52.0% | 185.9s |
| RWKV-7, depth 6 | 13.23M | 5.98M | 48.0% | 202.9s |

## Paper/repo-matched setup

The repo config uses the LeWM paper training hyperparameters: 100 epochs, batch
size 128, AdamW with lr `5e-5`, weight decay `1e-3`, bf16, history size 3,
one-step prediction, ViT-tiny encoder, and the standard TwoRoom evaluation
config with CEM.

The current dependency stack reports the repo transformer LeWM as 18.03M
parameters. For a repo-size-matched RWKV-7 model, use:

```bash
python compare_predictors.py \
  --run-name compare_rwkv7_transformer_paper_repo_matched \
  --rwkv-config rwkv7_repo_matched \
  --rwkv-name rwkv7_repo_matched
```

This trains:

| Model | Config | Total params | Predictor params |
| --- | --- | ---: | ---: |
| Transformer | `lewm` | 18.03M | 10.79M |
| RWKV-7 repo matched | `rwkv7_repo_matched` | 18.22M | 10.98M |

The paper text describes LeWM as approximately 15M parameters. If you want an
RWKV-7 model closer to that paper-level size instead of the exact repo
transformer size, use:

```bash
python compare_predictors.py \
  --run-name compare_rwkv7_transformer_paper_approx \
  --rwkv-config rwkv7_paper_approx \
  --rwkv-name rwkv7_paper_approx
```

That RWKV-7 model is 15.22M total parameters.
