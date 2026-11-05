#!/usr/bin/env python
"""Reproduce a final TOFU unlearning run from the paper.

"Learning What to Forget: Improving LLM Unlearning via Learned Token-Level
Importance" (https://arxiv.org/abs/2606.06320).

Takes a method, a model, and a forget split, unlearns with the selected
hyperparameters from `scripts/selected_tofu.json` (Table 6 of the paper, full
precision), then evaluates. By default the evaluation is the TOFU
forget/retain extraction strengths + model utility, plus repetitiveness and
MMLU; pass --with-paraphrase to additionally run
the paraphrase quality (LLM-judged) and win-rate evals (two extra stages:
GPU generation, then CPU/API judging — needs OPENAI_API_KEY).

Examples:
    python scripts/reproduce_tofu.py --method NPO  --model 1B --forget-split forget10
    python scripts/reproduce_tofu.py --method ATWU --model 8B --forget-split forget05 --gpu 1
    python scripts/reproduce_tofu.py --method ATWU --model 1B --forget-split forget10 --with-paraphrase

Method availability follows the paper: RMU was evaluated only on 8B, ETW only
on 1B; all other methods run on both models.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_JSON = REPO_ROOT / "scripts" / "selected_tofu.json"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); does not override existing vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

MODELS = {
    "1B": "Llama-3.2-1B-Instruct",
    "8B": "Llama-3.1-8B-Instruct",
}
INPUT_DIM = {"1B": 2048, "8B": 4096}  # hidden size, for the ATWU scorer

SPLITS = {
    "forget01": ("retain99", "holdout01"),
    "forget05": ("retain95", "holdout05"),
    "forget10": ("retain90", "holdout10"),
}

METHODS = [
    "GradDiff", "DPO", "NPO", "SimNPO", "JensUn",
    "RMU", "WGA", "SatImp", "ETW", "ATWU",
]

NUM_EPOCHS = 10
BATCH_SIZE = 8
GRAD_ACCUM = 4

# The paper's 8B runs of these methods were trained with gradient checkpointing
# enabled (it changes backward-pass numerics, so it is required for bit-exact
# reproduction). ATWU/RMU/ETW runs never used it, nor did any 1B run.
GRAD_CKPT_8B_METHODS = {"GradDiff", "DPO", "NPO", "SimNPO", "JensUn", "WGA", "SatImp"}


def method_overrides(method: str, p: dict, model_key: str) -> list[str]:
    """Method-specific hydra overrides from the selected-params entry."""
    ma = "trainer.method_args"
    ov = [f"{ma}.gamma={p['gamma']}", f"{ma}.alpha={p['alpha']}"]

    if method == "GradDiff":
        ov += [f"{ma}.retain_loss_type={p['retain_loss_type']}"]
    elif method in ("NPO", "DPO"):
        ov += [f"{ma}.beta={p['beta']}"]
    elif method == "SimNPO":
        ov += [f"{ma}.beta={p['beta']}", f"{ma}.delta={p['delta']}"]
    elif method == "JensUn":
        pass  # gamma/alpha only; retain_loss_type=JensUn is the config default
    elif method == "WGA":
        ov += [f"{ma}.beta={p['beta']}"]
    elif method == "SatImp":
        ov += [f"{ma}.beta1={p['beta1']}", f"{ma}.beta2={p['beta2']}"]
    elif method == "ETW":
        ov += [f"{ma}.scorer.temperature={p['temperature']}"]
    elif method == "RMU":
        layer = int(p["layer"])
        window = "|".join(str(x) for x in (layer - 2, layer - 1, layer))
        ov += [
            f"{ma}.steering_coeff={p['steering_coeff']}",
            rf"{ma}.module_regex=model\.layers\.{layer}",
            rf'{ma}.trainable_params_regex=["model\.layers\.({window})\.mlp\.down_proj\.weight"]',
        ]
    elif method == "ATWU":
        ov += [
            f"{ma}.beta={p['beta']}",
            f"{ma}.scorer.cfg.input_dimension={INPUT_DIM[model_key]}",
            f"{ma}.scorer_trainer.optim_cfg.lr={p['scorer_lr']}",
            f"{ma}.scorer_trainer.optim_cfg.update_every_n_steps=5",
            f"+{ma}.scorer_trainer.optim_cfg.scheduler=linear",
            f"{ma}.scorer_trainer.lambda_entropy=1",
            f"{ma}.scorer_trainer.lambda_population=15",
            f"{ma}.scorer_trainer.budget=0.2",
            f"{ma}.scorer_trainer.lambda_l2=1",
        ]
    return ov


def hf_token_overrides() -> list[str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return []
    return [
        f"+model.model_args.token={token}",
        f"+model.tokenizer_args.token={token}",
    ]


def run(cmd: list[str], gpu: str | None, dry: bool) -> None:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if "MASTER_PORT" not in env:
        s = socket.socket()
        s.bind(("", 0))
        env["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    shown = " ".join(
        a.split("=")[0] + "=***" if ".token=" in a else a for a in cmd
    )
    print(f"\n[reproduce_tofu] CUDA_VISIBLE_DEVICES={gpu} {shown}\n", flush=True)
    if dry:
        return
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f"[reproduce_tofu] command failed (exit {result.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--model", required=True, choices=["1B", "8B"],
                    help="1B = Llama-3.2-1B-Instruct, 8B = Llama-3.1-8B-Instruct")
    ap.add_argument("--forget-split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--with-paraphrase", action="store_true",
                    help="Additionally run the paraphrase quality + win-rate evals "
                         "(default: skip them; run TOFU suite + repetitiveness + MMLU)")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (default 0)")
    ap.add_argument("--params", default=str(PARAMS_JSON),
                    help="Selected-hyperparameters JSON (default scripts/selected_tofu.json)")
    ap.add_argument("--task-root", default="TOFU_repro",
                    help="Runs land in saves/unlearn/<task-root>/<model>/<split>/<method>")
    ap.add_argument("--quality-samples", type=int, default=400,
                    help="max_samples for the forget/retain quality judging "
                         "(only with --with-paraphrase; default 400 = paper setting; "
                         "use a small value for cheap smoke tests)")
    ap.add_argument("--winrate-baseline", default=None,
                    help="Baseline repetitiveness .jsonl for the win-rate judge "
                         "(only used with --with-paraphrase). Default: "
                         "saves/eval/paraphrase_baselines/<model>/<retain>_<forget>/"
                         "paraphrase_evals/repetitiveness/model.jsonl")
    ap.add_argument("--skip-unlearn", action="store_true",
                    help="Skip training; evaluate the existing checkpoint")
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if a completed checkpoint exists "
                         "(default: auto-resume by skipping finished training)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")  # HF_TOKEN, OPENAI_API_KEY for the subprocesses
    params = json.loads(Path(args.params).read_text())["methods"]
    p = params[args.method]

    lr = p[f"lr_{args.model}"]
    if lr is None:
        sys.exit(
            f"[reproduce_tofu] {args.method} was not evaluated on the {args.model} "
            f"model in the paper (RMU is 8B-only, ETW is 1B-only)."
        )

    model = MODELS[args.model]
    retain_split, holdout_split = SPLITS[args.forget_split]
    task_name = f"{args.task_root}/{model}/{args.forget_split}/{args.method}"
    out_dir = f"saves/unlearn/{task_name}"
    experiment = "unlearn/tofu/idk" if args.method == "DPO" else "unlearn/tofu/default"
    token_ov = hf_token_overrides()

    # Auto-resume: training saves the checkpoint + trainer_state only on
    # completion, so their presence means training finished — skip it on
    # restarts (job preemptions/crashes) unless --force is given.
    out_path = REPO_ROOT / out_dir
    training_done = (out_path / "trainer_state.json").exists() and (
        (out_path / "model.safetensors").exists()
        or any(out_path.glob("model-*.safetensors"))
    )
    if training_done and not args.force and not args.skip_unlearn:
        print(f"[reproduce_tofu] completed checkpoint found at {out_dir} — "
              "skipping training (pass --force to retrain from scratch)")
        args.skip_unlearn = True
    elif args.force and out_path.exists():
        print(f"[reproduce_tofu] --force: retraining from scratch over {out_dir}")

    # ── 1. Unlearn ──
    train_cmd = [
        "python", "src/train.py", "--config-name=unlearn.yaml",
        f"experiment={experiment}",
        f"trainer={args.method if args.method != 'ETW' else 'ETWGradDiff'}",
        f"model={model}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{model}_full",
        *token_ov,
        "++model.model_args.device_map=auto",
        f"forget_split={args.forget_split}",
        f"retain_split={retain_split}",
        f"trainer.args.num_train_epochs={NUM_EPOCHS}",
        f"trainer.args.per_device_train_batch_size={BATCH_SIZE}",
        f"trainer.args.gradient_accumulation_steps={GRAD_ACCUM}",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.do_eval=false",
        f"trainer.args.learning_rate={lr}",
        *method_overrides(args.method, p, args.model),
    ]
    if args.model == "8B" and args.method in GRAD_CKPT_8B_METHODS:
        train_cmd.append("trainer.args.gradient_checkpointing=True")
    if not args.skip_unlearn:
        run(train_cmd, args.gpu, args.dry_run)

    # ── 2. TOFU eval: forget/retain extraction strength + model utility ──
    tofu_eval_cmd = [
        "python", "src/eval.py", "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        "eval=[tofu_repro]",
        f"model={model}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={out_dir}",
        *token_ov,
        "++model.model_args.device_map=auto",
        f"forget_split={args.forget_split}",
        f"holdout_split={holdout_split}",
        f"paths.output_dir={out_dir}",
        "eval.tofu.overwrite=true",
    ]
    run(tofu_eval_cmd, args.gpu, args.dry_run)

    para_out = f"{out_dir}/paraphrase_evals"
    common_eval = [
        "python", "src/eval.py",
        "experiment=eval/paraphrase/default.yaml",
        f"model={model}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={out_dir}",
        *token_ov,
        "++model.model_args.device_map=auto",
        f"paths.output_dir={para_out}",
    ]
    repet_gen = "++eval.paraphrase.metrics.repetitiveness.generation={max_new_tokens:128,do_sample:false,temperature:0.0}"

    if not args.with_paraphrase:
        # ── 3a. Utility evals only: repetitiveness + MMLU ──
        run(common_eval + [
            "eval=[paraphrase]",
            repet_gen,
            "~eval.paraphrase.metrics.winrate",
        ], args.gpu, args.dry_run)
        return

    # ── 3b. Full paraphrase eval: generation (GPU), then judging (CPU/API) ──
    quality_ov = [
        f"eval.paraphrase.forget_split={args.forget_split}",
        f"eval.paraphrase.retain_split={retain_split}",
        f"eval.paraphrase.metrics.forget_quality.max_samples={args.quality_samples}",
        f"eval.paraphrase.metrics.retain_quality.max_samples={args.quality_samples}",
        "eval.paraphrase.metrics.forget_quality.datasets.TOFU_para_forget_eval.args.num_train_paraphrases=10",
        "eval.paraphrase.metrics.retain_quality.datasets.TOFU_para_retain_eval.args.num_train_paraphrases=10",
    ]
    run(common_eval + [
        "eval=[paraphrase_tofu]",
        *quality_ov,
        repet_gen,
        "+eval.paraphrase.metrics.forget_quality.generation_only=true",
        "+eval.paraphrase.metrics.retain_quality.generation_only=true",
        "~eval.paraphrase.metrics.winrate",
    ], args.gpu, args.dry_run)

    winrate_baseline = args.winrate_baseline or (
        f"saves/eval/paraphrase_baselines/{model}/{retain_split}_{args.forget_split}"
        "/paraphrase_evals/repetitiveness/model.jsonl"
    )
    run(common_eval + [
        "eval=[paraphrase_tofu]",
        *quality_ov,
        f"+eval.paraphrase.metrics.winrate.baseline_path={winrate_baseline}",
        "+judge_only=true",
    ], "", args.dry_run)  # judging is API-only; hide GPUs


if __name__ == "__main__":
    main()
