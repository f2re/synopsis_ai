from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

TRIGGER_WORD = "<26063_pixel_style>"
BASE_STYLE_CAPTION = (
    f"{TRIGGER_WORD}, modern high-quality crisp pixel art, hard-edged square pixels, "
    "clean clusters, cozy scientific meteorological campus on Gulf of Finland, St Petersburg, "
    "north-west Russia, birch and pine, granite embankment, ochre and terracotta facade, "
    "strict orthographic projection, no anti-aliasing blur, no 3D render look"
)


def collect_approved_project_assets(repo_root: Path, spec_path: Path) -> list[dict[str, Any]]:
    """Gathers approved/existing source images from the repository that conform to project standards."""
    approved_list = []
    
    # 1. Check spec assets that already exist
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        for asset in spec.get("assets", []):
            img_path = repo_root / asset["path"]
            if img_path.is_file():
                approved_list.append({
                    "id": asset["id"],
                    "path": img_path,
                    "title": asset.get("title", ""),
                    "prompt": asset.get("prompt", ""),
                    "group": asset.get("group", ""),
                    "size": asset.get("size", [256, 256])
                })

    # 2. Check candidate and reference assets under assets/source
    source_dir = repo_root / "assets/source"
    if source_dir.is_dir():
        for png in sorted(source_dir.rglob("*.png")):
            if png.name.startswith("._") or png.name.startswith("."):
                continue
            if any(item["path"] == png for item in approved_list):
                continue
            name = png.stem.replace("_", " ")
            approved_list.append({
                "id": f"SRC-{png.stem}",
                "path": png,
                "title": name,
                "prompt": f"pixel art illustration of {name}",
                "group": png.parent.name,
                "size": [256, 256]
            })
            
    return approved_list


def build_lora_dataset(
    repo_root: Path,
    spec_path: Path,
    dataset_dir: Path
) -> dict[str, Any]:
    """Extracts approved project images and creates paired (image.png, image.txt) dataset for fine-tuning."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images = collect_approved_project_assets(repo_root, spec_path)
    
    copied = 0
    categories: dict[str, int] = {}
    
    for item in images:
        src_path = item["path"]
        if not src_path.is_file():
            continue
            
        dst_img = dataset_dir / f"{item['id']}.png"
        dst_txt = dataset_dir / f"{item['id']}.txt"
        
        shutil.copy2(src_path, dst_img)
        
        caption_parts = [TRIGGER_WORD]
        if item.get("title"):
            caption_parts.append(item["title"])
        if item.get("prompt"):
            caption_parts.append(item["prompt"])
        caption_parts.append(BASE_STYLE_CAPTION)
        
        dst_txt.write_text(", ".join(caption_parts), encoding="utf-8")
        copied += 1
        
        group = item.get("group", "general")
        categories[group] = categories.get(group, 0) + 1

    return {
        "dataset_path": str(dataset_dir),
        "total_images": copied,
        "categories": categories,
        "trigger_word": TRIGGER_WORD
    }
