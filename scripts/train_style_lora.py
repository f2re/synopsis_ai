#!/usr/bin/env python3
"""Run the official Diffusers Z-Image LoRA trainer on reviewed project images.

The old standalone MSE network did not train a diffusion model and its files
must not be treated as Z-Image adapters. This launcher never activates a LoRA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from core.lora_trainer import TRIGGER_WORD, build_lora_dataset  # noqa: E402

TRAINER_SHA256 = "a3c801c3586983f41262253185f78c95d84791759f5faf4160a85738497bd042"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed training manifest")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Path to repository root")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="New empty imagefolder directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory for the candidate LoRA")
    parser.add_argument("--base-model", type=Path, required=True, help="Local Tongyi-MAI/Z-Image base model")
    parser.add_argument("--trainer", type=Path, required=True, help="Official Diffusers v0.41.0 Z-Image trainer")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=26063)
    args = parser.parse_args()

    if args.steps < 1 or args.resolution < 256 or args.resolution % 16:
        parser.error("steps must be positive; resolution must be at least 256 and divisible by 16")
    if "turbo" in args.base_model.name.lower() or not (args.base_model / "model_index.json").is_file():
        parser.error("a complete local Z-Image base model (not Turbo) is required")
    if not args.trainer.is_file() or hashlib.sha256(args.trainer.read_bytes()).hexdigest() != TRAINER_SHA256:
        parser.error("trainer must match the pinned official Diffusers v0.41.0 script")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")

    dataset = build_lora_dataset(args.repo_root, args.manifest, args.dataset_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "accelerate.commands.launch", str(args.trainer),
        "--pretrained_model_name_or_path", str(args.base_model),
        "--dataset_name", str(args.dataset_dir.resolve()), "--caption_column", "text",
        "--instance_prompt", TRIGGER_WORD, "--output_dir", str(args.output_dir),
        "--mixed_precision", "bf16", "--gradient_checkpointing", "--cache_latents", "--offload",
        "--resolution", str(args.resolution), "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4", "--learning_rate", "1e-4",
        "--max_train_steps", str(args.steps), "--rank", "16", "--seed", str(args.seed),
        "--report_to", "tensorboard",
    ]
    run = {"status": "running", "base_model": str(args.base_model), "trainer_sha256": TRAINER_SHA256,
           "dataset": dataset, "steps": args.steps, "resolution": args.resolution, "seed": args.seed}
    record = args.output_dir / "training-run.json"
    record.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(command, check=False)
    weights = args.output_dir / "pytorch_lora_weights.safetensors"
    run["exit_code"] = result.returncode
    run["status"] = "trained_candidate" if result.returncode == 0 and weights.is_file() else "failed"
    if weights.is_file():
        run["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    record.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if run["status"] == "trained_candidate" else (result.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
