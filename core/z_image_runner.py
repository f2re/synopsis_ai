from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_DIR = Path("/home/user/synopsis-ai/models/diffusers/Z-Image-Turbo")
COMFY_PYTHON = Path("/home/user/synopsis-ai/comfy-venv/bin/python3")

_pipeline = None


def is_z_image_available() -> bool:
    """Checks whether Z-Image-Turbo diffusers weights are present on disk."""
    if not MODEL_DIR.is_dir():
        return False
    transformer_dir = MODEL_DIR / "transformer"
    return transformer_dir.is_dir() and any(transformer_dir.glob("*.safetensors"))


def _run_in_comfy_env(
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    seed: int,
    steps: int,
    negative_prompt: str = ""
) -> bool:
    """Delegates inference to comfy-venv which contains PyTorch CUDA 12.8 and diffusers."""
    script_path = Path(__file__).resolve()
    cmd = [
        str(COMFY_PYTHON),
        str(script_path),
        "--prompt", prompt,
        "--negative-prompt", negative_prompt,
        "--output", str(output_path),
        "--width", str(width),
        "--height", str(height),
        "--seed", str(seed),
        "--steps", str(steps)
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path.is_file()
    except subprocess.CalledProcessError as err:
        print(f"Subprocess generation failed: {err.stderr}", file=sys.stderr)
        return False


def generate_z_image_candidate(
    prompt: str,
    output_path: Path,
    width: int = 256,
    height: int = 256,
    seed: int | None = None,
    steps: int = 8,
    negative_prompt: str = ""
) -> bool:
    """Generates an image candidate using Z-Image-Turbo on GPU 0 with aspect ratio preservation."""
    if not is_z_image_available():
        return False

    seed_val = seed if seed is not None else 26063

    try:
        import torch
        from diffusers import ZImagePipeline
    except ImportError:
        # If in an environment without torch (e.g. mcp-venv), delegate to comfy-venv
        return _run_in_comfy_env(prompt, output_path, width, height, seed_val, steps, negative_prompt)

    global _pipeline
    if _pipeline is None:
        _pipeline = ZImagePipeline.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.bfloat16
        ).to("cuda:0")

    generator = torch.Generator(device="cuda:0").manual_seed(seed_val)

    # Compute high-res generator scale keeping exact aspect ratio, aligned to 16px multiples.
    # For small items (e.g. 16x16, 32x32), generate raw at at least 256x256 so the diffusion
    # model has sufficient latent capacity to render intricate semantic details.
    raw_w = max(256, min(width * 4, 1024))
    raw_h = max(256, min(height * 4, 1024))

    # Preserve exact target aspect ratio
    target_aspect = width / float(height)
    if target_aspect > 1.0:
        raw_h = max(128, int(raw_w / target_aspect))
    elif target_aspect < 1.0:
        raw_w = max(128, int(raw_h * target_aspect))

    gen_w = max(128, (raw_w // 16) * 16)
    gen_h = max(128, (raw_h // 16) * 16)

    try:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "generator": generator,
            "height": gen_h,
            "width": gen_w,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
            kwargs["guidance_scale"] = 3.5
        else:
            kwargs["guidance_scale"] = 0.0

        image = _pipeline(**kwargs).images[0]

        # Ensure saved raw candidate matches aligned generation dimensions
        if image.size != (gen_w, gen_h):
            from PIL import Image
            image = image.resize((gen_w, gen_h), Image.LANCZOS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return True
    except Exception as exc:
        print(f"Error during Z-Image-Turbo generation: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Z-Image-Turbo inference worker")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--width", type=int, default=256, help="Logical width")
    parser.add_argument("--height", type=int, default=256, help="Logical height")
    parser.add_argument("--seed", type=int, default=26063, help="Seed")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps")
    args = parser.parse_args()

    success = generate_z_image_candidate(
        prompt=args.prompt,
        output_path=Path(args.output),
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        negative_prompt=args.negative_prompt
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
