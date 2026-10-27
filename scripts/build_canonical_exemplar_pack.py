#!/usr/bin/env python3
"""Validate and materialize the 26063 canonical exemplar training pack.

The source images stay in datasets/master_style_lora. This script selects only the
references explicitly admitted by datasets/canonical_exemplars/manifest.v1.json,
uses corrected canonical captions, and prevents legacy mirrored/excluded examples
from silently entering a training run.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "datasets/canonical_exemplars/manifest.v1.json"
DEFAULT_OUT = REPO_ROOT / "datasets/canonical_exemplars/_build/v1"
ALLOWED_USES = {"canonical", "style_only", "excluded"}


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"Unsupported manifest version: {data.get('version')!r}")
    return data


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()

    contract = data.get("global_contract", {})
    if contract.get("world_logical_grid") != [480, 270]:
        errors.append("global_contract.world_logical_grid must be [480, 270]")
    if contract.get("world_character_grid") != [32, 48]:
        errors.append("global_contract.world_character_grid must be [32, 48]")
    if contract.get("composition_ratio") != "60/30/10":
        errors.append("global_contract.composition_ratio must be '60/30/10'")

    for item in data.get("examples", []):
        item_id = item.get("id")
        if not item_id:
            errors.append("example without id")
            continue
        if item_id in seen_ids:
            errors.append(f"duplicate id: {item_id}")
        seen_ids.add(item_id)

        use = item.get("training_use")
        if use not in ALLOWED_USES:
            errors.append(f"{item_id}: invalid training_use={use!r}")

        image_rel = item.get("image")
        caption_rel = item.get("source_caption")
        if not image_rel or not (REPO_ROOT / image_rel).is_file():
            errors.append(f"{item_id}: missing image {image_rel!r}")
        if not caption_rel or not (REPO_ROOT / caption_rel).is_file():
            errors.append(f"{item_id}: missing source caption {caption_rel!r}")

        if image_rel in seen_images:
            errors.append(f"{item_id}: duplicate image reference {image_rel}")
        if image_rel:
            seen_images.add(image_rel)

        caption = item.get("caption_override", "").strip()
        if use != "excluded" and "<26063_pixel_style>" not in caption:
            errors.append(f"{item_id}: active example caption lacks <26063_pixel_style>")
        if use == "canonical" and item.get("weight", 0) <= 0:
            errors.append(f"{item_id}: canonical example must have positive weight")
        if use == "excluded" and item.get("weight", 0) != 0:
            errors.append(f"{item_id}: excluded example must have zero weight")
        if "_flip." in str(image_rel) or str(image_rel).endswith("_flip.png"):
            errors.append(f"{item_id}: manifest must never point at a legacy mirrored file")

    return errors


def selected_examples(data: dict[str, Any], include_style_only: bool) -> list[dict[str, Any]]:
    allowed = {"canonical"}
    if include_style_only:
        allowed.add("style_only")
    return [item for item in data["examples"] if item.get("training_use") in allowed]


def build_pack(data: dict[str, Any], out_dir: Path, include_style_only: bool) -> int:
    examples = selected_examples(data, include_style_only=include_style_only)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    for index, item in enumerate(examples, start=1):
        source_image = REPO_ROOT / item["image"]
        suffix = source_image.suffix.lower()
        stem = f"{index:03d}_{item['id'].lower()}"
        target_image = out_dir / f"{stem}{suffix}"
        target_caption = out_dir / f"{stem}.txt"

        shutil.copy2(source_image, target_image)
        target_caption.write_text(item["caption_override"].strip() + "\n", encoding="utf-8")

        metadata_rows.append(
            {
                "file_name": target_image.name,
                "text": item["caption_override"].strip(),
                "example_id": item["id"],
                "family": item["family"],
                "training_use": item["training_use"],
                "weight": item.get("weight", 1.0),
                "flip_allowed": bool(item.get("flip_allowed", False)),
                "anchors": item.get("anchors", []),
            }
        )

    metadata_path = out_dir / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )

    contract_path = out_dir / "canonical_contract.json"
    contract_path.write_text(
        json.dumps(data["global_contract"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(examples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-style-only",
        action="store_true",
        help="Also include style-only references. Canonical mode excludes them by default.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest and source paths without copying files.",
    )
    args = parser.parse_args()

    data = load_manifest(args.manifest)
    errors = validate_manifest(data)
    if errors:
        print("Canonical exemplar validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    canonical_count = sum(1 for item in data["examples"] if item["training_use"] == "canonical")
    style_count = sum(1 for item in data["examples"] if item["training_use"] == "style_only")
    excluded_count = sum(1 for item in data["examples"] if item["training_use"] == "excluded")
    print(
        f"Validation OK: canonical={canonical_count}, style_only={style_count}, excluded={excluded_count}"
    )

    if args.validate_only:
        return 0

    count = build_pack(data, args.out, include_style_only=args.include_style_only)
    print(f"Built {count} training exemplars in {args.out}")
    print("No mirrored legacy *_flip files were admitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
