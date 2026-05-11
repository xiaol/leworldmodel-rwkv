
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

### Quick start for this RWKV fork

Clone and install:

```bash
git clone git@github.com:xiaol/leworldmodel-rwkv.git
cd leworldmodel-rwkv
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

Choose a storage directory outside the git repo for datasets and checkpoints:

```bash
export STABLEWM_HOME=/path/to/stable_worldmodel_storage
mkdir -p "$STABLEWM_HOME"
```

Download the `.h5` datasets from the LeWM HuggingFace collection and place them
under `$STABLEWM_HOME`. For example, PushT training expects:

```text
$STABLEWM_HOME/pusht_expert_train.h5
```

This repository intentionally does not include datasets, checkpoints, videos,
or training logs. Large local artifacts are ignored by `.gitignore`.

Train the original Transformer predictor:

```bash
python train.py --config-name=lewm data=pusht wandb.enabled=False
```

Train the parameter-matched RWKV-7 predictor:

```bash
python train.py --config-name=rwkv7_repo_matched data=pusht wandb.enabled=False
```

Use the native RWKV CUDA recurrence for RWKV speed experiments:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='8.9' \
python train.py --config-name=rwkv7_repo_matched \
  data=pusht \
  wandb.enabled=False \
  predictor.backend=cuda
```

If the CUDA extension cannot be built, use `predictor.backend=auto` or
`predictor.backend=torch`. The torch backend is useful for portability, but it
is not a fair speed benchmark for RWKV-7.

Evaluate a saved checkpoint by passing the checkpoint path without the
`_object.ckpt` suffix:

```bash
python eval.py --config-name=pusht \
  policy=$STABLEWM_HOME/my_run/lewm_rwkv7_repo_matched_epoch_9
```

Run a matched Transformer-vs-RWKV comparison:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='8.9' \
python compare_predictors.py \
  --cache-dir "$STABLEWM_HOME" \
  --run-name compare_rwkv7_pusht \
  --data pusht \
  --eval-config pusht \
  --rwkv-config rwkv7_repo_matched \
  --rwkv-name rwkv7_repo_matched \
  --rwkv-backend cuda \
  --train-epochs 9 \
  --batch-size 128 \
  --num-workers 4 \
  --eval-num 50 \
  --eval-budget 50 \
  --goal-offset 25 \
  --solver-samples 300 \
  --solver-steps 30 \
  --solver-topk 30
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

This fork also includes an RWKV-7 predictor variant. The base config is
`config/train/rwkv7.yaml`; `config/train/rwkv7_repo_matched.yaml` increases
RWKV depth to 11 to roughly match the repo Transformer parameter count.

```bash
# Transformer LeWM
python train.py --config-name=lewm data=pusht

# RWKV-7 LeWM, portable backend selection
python train.py --config-name=rwkv7_repo_matched data=pusht

# RWKV-7 LeWM, force the native CUDA recurrence backend
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST='8.9' \
python train.py --config-name=rwkv7_repo_matched data=pusht predictor.backend=cuda
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## RWKV-7 benchmark summary

The RWKV-7 variant replaces the autoregressive Transformer predictor with an
RWKV-7 x070-style recurrent predictor while keeping the same ViT-tiny encoder,
action encoder, projector heads, optimizer, training loss, and CEM planner.
Parameter-matched comparisons use `rwkv7_repo_matched` at depth 11:

| Model | Depth | Total params | Predictor params |
| --- | ---: | ---: | ---: |
| Transformer LeWM | 6 | 18.03M | 10.79M |
| RWKV-7 LeWM | 11 | 18.22M | 10.98M |

Default eval settings used below:

- `eval.num_eval=50`
- `eval_budget=50`
- `goal_offset_steps=25`
- `solver.num_samples=300`
- `solver.n_steps=30`
- `solver.topk=30`

### TwoRoom

Full TwoRoom data, 10 epochs, batch size 128, AdamW lr `5e-5`, weight decay
`1e-3`, `wm.history_size=3`, ViT-tiny trained from scratch.

| Model | Depth | TwoRoom success | Eval time |
| --- | ---: | ---: | ---: |
| Released LeWM checkpoint | 6 | 84.0% | 170.3s |
| Transformer LeWM | 6 | 90.0% | 196.2s |
| RWKV-7 LeWM, CUDA | 11 | 92.0% | 240.2s |

### PushT

Full PushT data, batch size 128, AdamW lr `5e-5`, weight decay `1e-3`,
`wm.history_size=3`, ViT-tiny trained from scratch. The local run was stopped at
epoch 9 and evaluated both architectures at the same epoch.

| Model | Epoch | Depth | PushT success | Eval time |
| --- | ---: | ---: | ---: | ---: |
| Released LeWM checkpoint | - | 6 | 96.0% | 186.2s |
| Transformer LeWM | 9 | 6 | 98.0% | 177.2s |
| RWKV-7 LeWM, CUDA | 9 | 11 | 98.0% | 217.2s |

Interpretation: RWKV-7 matched or slightly exceeded the Transformer success
rate in these local runs, but it was slower during CEM evaluation. PushT at
epoch 9 is saturated on this 50-episode eval, so it is not a strong quality
separator; both local models solved 49/50 starts but failed different episodes.

Detailed notes and artifacts are in:

- `RWKV7_COMPARISON.md`
- `PUSHT_RWKV7_BENCHMARK.md`
- `PAPER_STYLE_EVAL_NOTES.md`

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
