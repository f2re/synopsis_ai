#!/usr/bin/env python3
"""Downloads foundational model weights for synopsis-assets-mcp:
- Z-Image Turbo / Z-Image Base (Tongyi-MAI/Z-Image)
- FLUX.2 Klein 4B (black-forest-labs/flux2)
- SDXL + ControlNet (stabilityai, lllyasviel)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS_DIR = Path("/home/user/synopsis-ai/models")

TARGETS = {
    "sdxl_turbo": {
        "repo_id": "stabilityai/sdxl-turbo",
        "filename": "sd_xl_turbo_1.0_fp16.safetensors",
        "dest": MODELS_DIR / "checkpoints/sd_xl_turbo_1.0_fp16.safetensors",
        "alias": "z_image_turbo.safetensors",
        "description": "Fast 1-4 step SDXL Turbo single safetensors checkpoint for ComfyUI"
    },
    "z_image_turbo": {
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "filename": "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "dest": MODELS_DIR / "checkpoints/z_image_turbo/diffusion_pytorch_model-00001-of-00003.safetensors",
        "description": "Tongyi-MAI 6B Z-Image Turbo diffusers transformer"
    },
    "z_image_base": {
        "repo_id": "Tongyi-MAI/Z-Image",
        "filename": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "dest": MODELS_DIR / "checkpoints/z_image_base/diffusion_pytorch_model-00001-of-00002.safetensors",
        "description": "6B foundation model for LoRA style fine-tuning"
    },
    "flux2_klein_4b": {
        "repo_id": "black-forest-labs/FLUX.2-Klein-4B",
        "filename": "flux2_klein_4b.safetensors",
        "dest": MODELS_DIR / "diffusion_models/flux2_klein_4b.safetensors",
        "description": "Primary high-fidelity editor and multi-reference model"
    },
    "sdxl_controlnet_canny": {
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
        "filename": "diffusion_pytorch_model.safetensors",
        "dest": MODELS_DIR / "controlnet/controlnet_canny_sdxl.safetensors",
        "description": "ControlNet for pixel-matched day/night silhouettes"
    }
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights for synopsis-assets-mcp")
    parser.add_argument("--model", choices=list(TARGETS.keys()) + ["all"], default="all",
                        help="Model to download")
    parser.add_argument("--token", help="HuggingFace API token if required for gated weights")
    args = parser.parse_args()

    selected = list(TARGETS.keys()) if args.model == "all" else [args.model]

    print("=== Model Weights Download Helper ===")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return 1

    for key in selected:
        info = TARGETS[key]
        dest = info["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nModel [{key}]: {info['description']}")
        print(f"Target: {dest}")

        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            print(f"✅ Already present ({dest.stat().st_size // (1024*1024)} MB). Skipping.")
            continue

        print(f"Downloading {info['filename']} from {info['repo_id']}...")
        try:
            downloaded = hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                token=args.token,
                local_dir=str(dest.parent)
            )
            print(f"✅ Downloaded: {downloaded}")

            # If alias is specified, create symlinks for ComfyUI
            alias = info.get("alias")
            if alias:
                alias_path = dest.parent / alias
                if not alias_path.exists():
                    alias_path.symlink_to(Path(downloaded).name)
                    print(f"🔗 Linked alias: {alias_path} -> {Path(downloaded).name}")
                
                # Also link into ComfyUI checkpoints folder
                comfy_ckpt_dir = Path("/home/user/synopsis-ai/comfyui/models/checkpoints")
                if comfy_ckpt_dir.is_dir():
                    target_comfy = comfy_ckpt_dir / alias
                    if not target_comfy.exists():
                        target_comfy.symlink_to(dest)
                        print(f"🔗 Linked to ComfyUI: {target_comfy} -> {dest}")
        except Exception as exc:
            print(f"⚠️ Could not download {key}: {exc}")
            print(f"   (Place weights manually at {dest} when offline/authenticated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
