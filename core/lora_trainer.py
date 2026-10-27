"""Prepare an explicitly reviewed and provenance-tracked LoRA dataset."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

TRIGGER_WORD = "synopsis_pixel_art"
REQUIRED_REVIEW_ROLES = {"pixel-art-quality-reviewer", "art-director"}
SUPPORTED_LICENSES = {"project-internal-production", "CC0-1.0", "CC-BY-4.0", "Apache-2.0", "public-domain"}


def _project_file(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-project file: {relative}")
    return path


def collect_approved_project_assets(repo_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Require explicit rights, checksums and two independent review records.

    Existing PNGs and files in candidate folders are not evidence of approval.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reviewed training manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("assets", [])
    if not isinstance(entries, list) or len(entries) < 3:
        raise ValueError("Training manifest must contain at least three reviewed assets")

    images: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        asset_id = entry.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", asset_id) or asset_id in seen:
            raise ValueError(f"Invalid or duplicate asset ID: {asset_id}")
        seen.add(asset_id)
        relative = entry.get("path", "")
        source = _project_file(repo_root, relative)
        parts = Path(relative).parts
        if (parts[:2] != ("assets", "source") or ".." in parts or
                not source.is_relative_to((repo_root / "assets/source").resolve()) or
                any(part.lower() in {"candidates", "rejected", "generated"} for part in parts) or
                source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}):
            raise ValueError(f"Training source is not an approved source image: {relative}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if entry.get("sha256") != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        if (entry.get("license") not in SUPPORTED_LICENSES or not entry.get("origin")
                or not entry.get("caption") or not entry.get("author_id")):
            raise ValueError(f"Supported license, origin, caption and author_id required: {asset_id}")
        rights = json.loads(_project_file(repo_root, entry.get("rights_evidence", "")).read_text(encoding="utf-8"))
        if any(rights.get(key) != value for key, value in {
            "asset_id": asset_id, "sha256": digest, "license": entry["license"], "origin": entry["origin"]
        }.items()):
            raise ValueError(f"Rights evidence does not match asset: {asset_id}")
        reviews = entry.get("reviews", [])
        by_role = {review.get("role"): review for review in reviews}
        if not REQUIRED_REVIEW_ROLES.issubset(by_role):
            raise ValueError(f"Both independent reviews required: {asset_id}")
        reviewers: set[str] = set()
        for role in REQUIRED_REVIEW_ROLES:
            review = by_role[role]
            if review.get("status", "").lower() != "pass" or not review.get("reviewer_id"):
                raise ValueError(f"Missing PASS or reviewer for {asset_id}: {role}")
            evidence = json.loads(_project_file(repo_root, review.get("evidence", "")).read_text(encoding="utf-8"))
            expected = {"asset_id": asset_id, "sha256": digest, "role": role,
                        "reviewer_id": review["reviewer_id"], "status": "pass"}
            if any(evidence.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Review evidence does not match asset and reviewer: {asset_id}: {role}")
            reviewers.add(review["reviewer_id"])
        if len(reviewers) != 2 or entry["author_id"] in reviewers:
            raise ValueError(f"Reviews must be independent: {asset_id}")
        images.append({**entry, "source": source})
    return images


def build_lora_dataset(repo_root: Path, manifest_path: Path, dataset_dir: Path) -> dict[str, Any]:
    """Build an imagefolder dataset with per-image captions for Diffusers."""
    images = collect_approved_project_assets(repo_root, manifest_path)
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise ValueError(f"Dataset directory must be empty: {dataset_dir}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for entry in images:
        filename = f"{entry['id']}{entry['source'].suffix.lower()}"
        shutil.copy2(entry["source"], dataset_dir / filename)
        metadata.append({"file_name": filename, "text": f"{TRIGGER_WORD}, {entry['caption']}"})
    (dataset_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata), encoding="utf-8"
    )
    return {"dataset_path": str(dataset_dir), "total_images": len(images), "trigger_word": TRIGGER_WORD,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
