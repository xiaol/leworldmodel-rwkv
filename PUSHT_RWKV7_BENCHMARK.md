# PushT RWKV-7 vs Transformer Benchmark

## Current short comparison

This is the first local PushT run using the same short "old comparison"
budget style as the earlier TwoRoom 2ep100b benchmark.

Training setup for the local transformer and RWKV rows:

- PushT expert training data: `pusht_expert_train.h5`
- 2 epochs
- 100 train batches per epoch
- 10 validation batches per epoch
- Batch size 64
- AdamW, learning rate `5e-5`, weight decay `1e-3`
- `wm.history_size=3`
- ViT-tiny encoder trained from scratch
- RWKV-7 uses native CUDA backend

Evaluation setup:

- `eval.num_eval=50`
- `eval_budget=50`
- `goal_offset_steps=25`
- `solver.num_samples=300`
- `solver.n_steps=30`
- `solver.topk=30`

| Model | Depth | Total params | Predictor params | PushT success | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper/released LeWM checkpoint | 6 | 18.03M | 10.79M | 96.0% | 186.2s |
| Transformer, 2ep100b | 6 | 18.03M | 10.79M | 2.0% | 191.4s |
| RWKV-7 CUDA, 2ep100b | 11 | 18.22M | 10.98M | 2.0% | 234.2s |

Artifacts:

```text
/home/xiaol/.stable_worldmodel/datasets/pusht_expert_train.h5
/home/xiaol/.stable_worldmodel/checkpoints/pusht_results.txt
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_2ep100b_cuda/summary.json
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_2ep100b_cuda/transformer/pusht_results.txt
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_2ep100b_cuda/rwkv7_repo_matched/pusht_results.txt
```

## Interpretation

The 2ep100b PushT run is too short to compare final model quality. Both local
models are effectively untrained for PushT under this budget, while the
released checkpoint reaches 96.0% with the same default PushT eval settings.

Under this short run, RWKV-7 CUDA is not faster than the transformer in
evaluation. It has slightly more matched parameters and took 234.2s versus
191.4s for the transformer.

For a fair quality comparison on PushT, continue with the paper-style training
budget: full data, batch size 128, 100 epochs, same optimizer, same eval.

## Full epoch-9 comparison

Started: 2026-05-01 16:47 Asia/Shanghai.

Command:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='8.9' \
STABLEWM_HOME=/home/xiaol/.stable_worldmodel/datasets \
.venv/bin/python compare_predictors.py \
  --cache-dir /home/xiaol/.stable_worldmodel/datasets \
  --run-name compare_rwkv7_pusht_full_cuda \
  --data pusht \
  --eval-config pusht \
  --rwkv-config rwkv7_repo_matched \
  --rwkv-name rwkv7_repo_matched \
  --rwkv-backend cuda \
  --train-epochs 100 \
  --batch-size 128 \
  --num-workers 4 \
  --eval-num 50 \
  --eval-budget 50 \
  --goal-offset 25 \
  --solver-samples 300 \
  --solver-steps 30 \
  --solver-topk 30
```

The original full 100-epoch run was stopped cleanly after the Transformer
reached epoch 9. To make a matched comparison, RWKV-7 CUDA was resumed and
trained to epoch 9 with the same data, optimizer, batch size, and eval setup.

Training setup for the epoch-9 comparison:

- Full PushT expert training data
- Batch size 128
- AdamW, learning rate `5e-5`, weight decay `1e-3`
- `wm.history_size=3`
- ViT-tiny encoder trained from scratch
- Transformer depth 6
- RWKV-7 CUDA depth 11

Evaluation setup:

- `eval.num_eval=50`
- `eval_budget=50`
- `goal_offset_steps=25`
- `solver.num_samples=300`
- `solver.n_steps=30`
- `solver.topk=30`

| Model | Epoch | Depth | Total params | Predictor params | PushT success | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper/released LeWM checkpoint | - | 6 | 18.03M | 10.79M | 96.0% | 186.2s |
| Transformer | 9 | 6 | 18.03M | 10.79M | 98.0% | 177.2s |
| RWKV-7 CUDA | 9 | 11 | 18.22M | 10.98M | 98.0% | 217.2s |

Notes:

- Transformer epoch 9 and RWKV-7 CUDA epoch 9 reached the same 98.0% success
  rate on this 50-episode PushT eval.
- RWKV-7 CUDA remained slower in eval wall time: 217.2s versus 177.2s for the
  Transformer.
- The first full RWKV eval attempt segfaulted after dataset sampling. A tiny
  one-episode RWKV eval succeeded, and the full eval retry completed normally.

Epoch-9 artifacts:

```text
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/transformer/lewm_transformer_compare_epoch_9_object.ckpt
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/transformer/pusht_results.txt
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/rwkv7_repo_matched/lewm_rwkv7_repo_matched_compare_epoch_9_object.ckpt
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/rwkv7_repo_matched/pusht_results.txt
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/logs/transformer_train.log
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/logs/transformer_epoch9_eval.log
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/logs/rwkv7_repo_matched_train_epoch9_resume2.log
/home/xiaol/.stable_worldmodel/datasets/compare_rwkv7_pusht_full_cuda/logs/rwkv7_repo_matched_epoch9_eval_retry.log
```
