import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def optional_int(value):
    if value is None or value.lower() in {"none", "null", "all", "full"}:
        return None
    return int(value)


def run_command(cmd, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"Log: {log_path}\n\n{tail}"
        )


def parse_results(path):
    text = path.read_text(errors="replace")
    matches = list(
        re.finditer(
            r"success_rate':\s*([0-9.]+).*?evaluation_time:\s*([0-9.]+)\s*seconds",
            text,
            flags=re.DOTALL,
        )
    )
    if not matches:
        return {}
    match = matches[-1]
    return {
        "success_rate": float(match.group(1)),
        "evaluation_time_seconds": float(match.group(2)),
    }


def default_result_file(eval_config):
    return {
        "pusht": "pusht_results.txt",
        "tworoom": "tworoom_results.txt",
        "cube": "ogb_cube_results.txt",
        "reacher": "dmc_results.txt",
    }.get(eval_config, f"{eval_config}_results.txt")


def model_jobs(args):
    common_train = [
        f"data={args.data}",
        "wandb.enabled=False",
        f"trainer.max_epochs={args.train_epochs}",
        f"loader.batch_size={args.batch_size}",
        f"loader.num_workers={args.num_workers}",
    ]
    common_train.extend(args.train_override)

    if args.train_batches is not None:
        common_train.append(f"+trainer.limit_train_batches={args.train_batches}")
    if args.val_batches is not None:
        common_train.append(f"+trainer.limit_val_batches={args.val_batches}")
    if args.num_workers == 0:
        common_train.extend(
            ["loader.persistent_workers=False", "loader.prefetch_factor=null"]
        )

    return [
        {
            "name": "transformer",
            "config": "lewm",
            "output_model_name": "lewm_transformer_compare",
            "extra_train": [],
            "train": common_train
            + [
                f"subdir={args.run_name}/transformer",
                "output_model_name=lewm_transformer_compare",
            ],
        },
        {
            "name": args.rwkv_name,
            "config": args.rwkv_config,
            "output_model_name": f"lewm_{args.rwkv_name}_compare",
            "extra_train": [],
            "train": common_train
            + [
                f"subdir={args.run_name}/{args.rwkv_name}",
                f"output_model_name=lewm_{args.rwkv_name}_compare",
                f"predictor.backend={args.rwkv_backend}",
            ],
        },
    ]


def count_checkpoint_params(ckpt_base):
    ckpt_path = Path(f"{ckpt_base}_object.ckpt")
    if not ckpt_path.exists():
        return {}

    import torch

    model = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    result = {"total_params": sum(p.numel() for p in model.parameters())}
    if hasattr(model, "predictor"):
        result["predictor_params"] = sum(p.numel() for p in model.predictor.parameters())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate comparable LeWM transformer vs RWKV-7 predictors."
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "STABLEWM_HOME",
            "/home/xiaol/.stable_worldmodel/datasets/quentinll--lewm-tworooms",
        ),
        help="StableWM cache containing the dataset and receiving checkpoints.",
    )
    parser.add_argument("--run-name", default="compare_rwkv7_transformer")
    parser.add_argument("--data", default="tworoom")
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Hydra config under config/eval. Defaults to --data.",
    )
    parser.add_argument(
        "--result-file",
        default=None,
        help="Eval result filename. Defaults to the eval config's output filename convention.",
    )
    parser.add_argument("--rwkv-config", default="rwkv7_repo_matched")
    parser.add_argument("--rwkv-name", default="rwkv7_repo_matched")
    parser.add_argument(
        "--rwkv-backend",
        choices=["cuda", "auto", "torch"],
        default="cuda",
        help="RWKV-7 recurrence backend for the RWKV job. Defaults to cuda so benchmarks do not silently fall back.",
    )
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--train-batches", type=optional_int, default=None)
    parser.add_argument("--val-batches", type=optional_int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-num", type=int, default=50)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--solver-samples", type=int, default=300)
    parser.add_argument("--solver-steps", type=int, default=30)
    parser.add_argument("--solver-topk", type=int, default=30)
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Extra Hydra override passed to both train jobs. Can be repeated.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    if args.eval_config is None:
        args.eval_config = args.data
    if args.result_file is None:
        args.result_file = default_result_file(args.eval_config)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    run_dir = cache_dir / args.run_name
    log_dir = run_dir / "logs"
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(cache_dir)

    jobs = model_jobs(args)
    summary = {
        "cache_dir": str(cache_dir),
        "run_name": args.run_name,
        "train": {
            "epochs": args.train_epochs,
            "limit_train_batches": args.train_batches,
            "limit_val_batches": args.val_batches,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": 3072,
            "rwkv_backend": args.rwkv_backend,
            "overrides": args.train_override,
        },
        "eval": {
            "config": args.eval_config,
            "result_file": args.result_file,
            "num_eval": args.eval_num,
            "eval_budget": args.eval_budget,
            "goal_offset_steps": args.goal_offset,
            "solver_samples": args.solver_samples,
            "solver_steps": args.solver_steps,
            "solver_topk": args.solver_topk,
        },
        "models": {},
    }

    for job in jobs:
        ckpt_base = (
            run_dir
            / job["name"]
            / f"{job['output_model_name']}_epoch_{args.train_epochs}"
        )
        if not args.skip_train:
            train_cmd = [
                sys.executable,
                "train.py",
                f"--config-name={job['config']}",
                *job["train"],
            ]
            run_command(train_cmd, env, log_dir / f"{job['name']}_train.log")

        if not args.skip_eval:
            eval_cmd = [
                sys.executable,
                "eval.py",
                f"--config-name={args.eval_config}",
                f"policy={ckpt_base}",
                f"eval.num_eval={args.eval_num}",
                f"eval.eval_budget={args.eval_budget}",
                f"eval.goal_offset_steps={args.goal_offset}",
                f"solver.num_samples={args.solver_samples}",
                f"solver.n_steps={args.solver_steps}",
                f"solver.topk={args.solver_topk}",
            ]
            run_command(eval_cmd, env, log_dir / f"{job['name']}_eval.log")

        result_path = ckpt_base.parent / args.result_file
        summary["models"][job["name"]] = {
            "checkpoint": str(ckpt_base),
            "result_file": str(result_path),
            **count_checkpoint_params(ckpt_base),
            **(parse_results(result_path) if result_path.exists() else {}),
        }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
