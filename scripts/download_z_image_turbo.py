#!/usr/bin/env python3
"""Downloads Tongyi-MAI/Z-Image-Turbo diffusers pipeline for synopsis-assets-mcp."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEST_DIR = Path("/home/user/synopsis-ai/models/diffusers/Z-Image-Turbo")
COMFY_DIR = Path("/home/user/synopsis-ai/comfyui/models/diffusers/Z-Image-Turbo")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Z-Image-Turbo diffusers weights")
    parser.add_argument("--dest", default=str(DEST_DIR), help="Destination directory")
    parser.add_argument("--token", help="Optional HuggingFace token for higher bandwidth limits")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    COMFY_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not COMFY_DIR.exists():
        COMFY_DIR.symlink_to(dest)

    print(f"=== Downloading Tongyi-MAI/Z-Image-Turbo ===")
    print(f"Target: {dest}")

    from huggingface_hub import snapshot_download

    t0 = time.time()
    try:
        downloaded = snapshot_download(
            repo_id="Tongyi-MAI/Z-Image-Turbo",
            local_dir=str(dest),
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ignore_patterns=["assets/*", "*.webp", "*.jpg", "*.png", "*.pdf"],
            token=args.token,
            max_workers=4
        )
        dt = time.time() - t0
        print(f"✅ Successfully downloaded Z-Image-Turbo to {downloaded} in {round(dt, 1)}s")
        return 0
    except Exception as exc:
        print(f"❌ Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
