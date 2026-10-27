from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mcp.server import MCPServer

from core.asset_classes import (
    ALL_CANONICAL_CLASSES,
    GENERATION_POLICY,
    GENERATION_PROFILES,
    OCCUPANCY_CONTRACTS,
    classify_asset,
    get_generation_profile,
    get_profile_for_asset,
)
from core.character_synthesizer import generate_character_sprite_frames
from core.comfy_client import ComfyClient
from core.day_night_matcher import compare_silhouettes, enforce_geometry_lock
from core.lora_trainer import build_lora_dataset
from core.normalizer import normalize_image
from core.pixel_refiner import refine_pixel_candidate
from core.pixellab_client import PixelLabClient
from core.prompt_compiler import compile_generation_prompts
from core.quality_review import evaluate_asset_candidate
from core.spec_loader import (
    build_generation_prompts,
    get_asset_by_id,
    get_palette_for_asset,
    load_gpl,
    load_spec,
)
from core.sprite_builder import assemble_sprite_sheet
from core.work_storage import WorkStorage
from core.z_image_runner import is_z_image_available, generate_z_image_candidate

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
CONFIG_PATH = ROOT_DIR / "config/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SERVER_INSTRUCTIONS = """
Specialized generative AI sprite generator, pixel refiner, and QA pipeline for «26063: Позывной «Синоптик»».

STANDARD ASSET PRODUCTION WORKFLOW:
1. SPEC & RECON:
   Call `asset_spec(asset_id)` to inspect dimensions, target class (one of 14 profiles), art priorities, framing contracts, and palette.
   Call `list_art_profiles()` to view all 14 visual generation classes and their guidelines.

2. GENERATION:
   Call `generate_candidates(asset_id, count=4)` to produce raw structured candidates in `assets/work/<asset_id>/raw/`.
   For animated characters, call `generate_sprite_frames(asset_id)`.
   For paired buildings (day & night), call `generate_day_night_pair(day_id, night_id)`.

3. EVALUATION & SELECTION:
   Call `rank_candidates(asset_id)` to evaluate candidate occupancy, silhouette clarity, and composition.
   Call `select_candidate(asset_id, candidate_id)` to choose the best candidate.

4. PIXEL REFINEMENT:
   Call `refine_candidate(asset_id)` to convert candidate to strict logical pixel grid, edge-aware cluster refinement, and palette mapping without destructive blur. Returns 1x, 2x, 4x preview paths and QA metrics.
   Call `preview_asset(asset_id)` at any time to inspect generated scales.

5. ART DIRECTOR REVIEW:
   Call `art_review(asset_id, status='pass'|'fail', reasons=[...])`.
   Review checks: adult proportions, age readability, material textures, Baltic atmospheric lighting, no chibi/babyface.

6. PRODUCTION APPROVAL:
   Call `approve_asset(asset_id)` to promote the refined asset to `assets/source/` (only succeeds if both technical contract and art review pass).
   If rejected, call `reject_candidate(asset_id, candidate_id, reasons=[...])` or `regenerate_asset(asset_id, correction="...")`.
"""

# Initialize MCP Server (v2 Python SDK)
mcp = MCPServer(
    name=CONFIG["server"]["name"],
    version="2.0.0",
    instructions=SERVER_INSTRUCTIONS.strip()
)

# Fallback paths allowing standalone execution or execution within synopsis_game monorepo
SPEC_PATH = (
    REPO_ROOT / CONFIG["paths"]["spec_path"]
    if (REPO_ROOT / CONFIG["paths"]["spec_path"]).is_file()
    else ROOT_DIR / "specs/kn01_level1.image-spec.json"
)
PALETTES_DIR = (
    REPO_ROOT / "assets/source/palettes"
    if (REPO_ROOT / "assets/source/palettes").is_dir()
    else ROOT_DIR / "palettes"
)
CANDIDATES_DIR = Path(CONFIG["paths"].get("candidates_dir", str(ROOT_DIR / "output/candidates")))
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = (
    REPO_ROOT / "assets/work"
    if (REPO_ROOT / "assets").is_dir()
    else ROOT_DIR / "assets/work"
)
WORK_DIR.mkdir(parents=True, exist_ok=True)

comfy = ComfyClient(base_url=CONFIG["comfyui"]["base_url"])
pixellab = PixelLabClient(api_url=CONFIG["pixellab"]["api_url"])
GPU_GENERATION_LOCK = asyncio.Lock()


def _candidate_rejection(storage: WorkStorage, source_path: Path) -> dict[str, Any] | None:
    """Return a rejection record only when it matches this exact source image."""
    source_name = source_path.name.removeprefix("selected_")
    marker = storage.review_dir / f"{Path(source_name).stem}.rejected.json"
    if not marker.is_file():
        return None
    try:
        rejection = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "rejected", "reasons": ["Invalid rejection record"]}
    if rejection.get("sha256") == hashlib.sha256(source_path.read_bytes()).hexdigest():
        return rejection
    return None


def _refined_output(storage: WorkStorage, report: dict[str, Any]) -> Path:
    """Resolve the current report image while refusing paths outside this asset."""
    path = Path(report.get("candidate_path") or storage.get_refined_path(suffix="refined"))
    if path.resolve().parent != storage.refined_dir.resolve():
        raise HTTPException(status_code=400, detail="Review report points outside the asset refined directory")
    return path


def get_active_palette(asset: dict[str, Any]) -> list[tuple[int, int, int]]:
    return get_palette_for_asset(asset, PALETTES_DIR)


# --------------------------------------------------------------------------
# MCP TOOLS: SPEC & ASSET METADATA
# --------------------------------------------------------------------------

@mcp.tool()
def list_art_profiles() -> dict[str, Any]:
    """Lists all 14 specialized generation profiles, their workflows, color limits, and art priorities."""
    return {
        "version": 3,
        "policy": GENERATION_POLICY,
        "classes": ALL_CANONICAL_CLASSES,
        "profiles": GENERATION_PROFILES,
    }


@mcp.tool()
def asset_spec(asset_id: str) -> dict[str, Any]:
    """[Lifecycle Step 1 - Recon] Returns canonical metadata, target class (one of 14 profiles), dimensions, palette family, occupancy bounds, art priorities, and compiled positive/negative prompts."""
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    pos_prompt, neg_prompt, meta = compile_generation_prompts(asset_id, asset)
    asset_class = classify_asset(asset_id, asset)
    profile = get_profile_for_asset(asset_id, asset)
    return {
        "asset_id": asset_id,
        "asset_class": asset_class,
        "generation_profile": profile,
        "size": asset.get("size"),
        "alpha": asset.get("alpha"),
        "occupancy_contract": OCCUPANCY_CONTRACTS.get(asset_class, {}),
        "compiled_positive_prompt": pos_prompt,
        "compiled_negative_prompt": neg_prompt,
        "palette_family": get_active_palette(asset)[:5],
        "palette_size": len(get_active_palette(asset)),
        "metadata": meta,
    }


# --------------------------------------------------------------------------
# MCP TOOLS: STRUCTURED GENERATION & CANDIDATES
# --------------------------------------------------------------------------

@mcp.tool()
async def generate_candidates(
    asset_id: str,
    count: int = 4,
    seed: int | None = None,
    backend: str = "auto",
    correction: str | None = None,
    creator_id: str | None = None,
) -> dict[str, Any]:
    """[Lifecycle Step 2 - Generation] Generates raw high-detail candidate images for an asset into assets/work/<asset_id>/raw/ using class-specific workflow and prompt."""
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    if not 1 <= count <= 8:
        raise HTTPException(status_code=400, detail="count must be between 1 and 8")
    pos_prompt, neg_prompt, meta = compile_generation_prompts(asset_id, asset, correction=correction)
    if creator_id:
        meta["creator_id"] = creator_id

    storage = WorkStorage(asset_id, base_dir=WORK_DIR)

    target_w, target_h = asset.get("size", [256, 256])
    palette = get_active_palette(asset)
    raw_paths: list[str] = []
    chosen_backend = backend

    base_seed = seed if seed is not None else 26063
    seeds: list[int] = []

    # Reserve filenames while holding the same lock as generation: concurrent
    # requests must never overwrite one another's candidates or run metadata.
    async with GPU_GENERATION_LOCK:
        storage.save_spec(asset)
        storage.save_prompt(pos_prompt, neg_prompt, meta)
        start_index = storage.next_candidate_index()
        if backend == "procedural" and asset.get("people") and asset.get("frame"):
            for c_idx in range(count):
                c_seed = base_seed + c_idx * 17
                frames, (fw, fh), cols = generate_character_sprite_frames(
                    asset, seed=c_seed, frames=None, palette=palette, repo_root=REPO_ROOT
                )
                sheet_raw = storage.get_raw_path(start_index + c_idx)
                frame_paths = []
                for f_idx, frame_img in enumerate(frames):
                    fn = storage.raw_dir / f"f_{f_idx:02d}_c{start_index + c_idx}.png"
                    frame_img.save(fn)
                    frame_paths.append(fn)
                assemble_sprite_sheet(frame_paths, sheet_raw, (fw, fh), columns=cols)
                raw_paths.append(str(sheet_raw))
                seeds.append(c_seed)
            chosen_backend = "character_synthesizer"
        elif is_z_image_available() and backend in ("auto", "z_image", "z_image_turbo"):
            for c_idx in range(count):
                c_seed = base_seed + c_idx * 17
                raw_target = storage.get_raw_path(start_index + c_idx)
                if not await asyncio.to_thread(
                    generate_z_image_candidate, pos_prompt, raw_target, target_w, target_h,
                    seed=c_seed, steps=9, negative_prompt=neg_prompt
                ):
                    break
                raw_paths.append(str(raw_target))
                seeds.append(c_seed)
            if raw_paths:
                chosen_backend = "z_image_turbo"

        status = "generated_raw" if len(raw_paths) == count else "failed"
        storage.record_generation(asset, pos_prompt, neg_prompt, meta, [Path(p) for p in raw_paths],
                                  seeds, chosen_backend, status)
    if status != "generated_raw":
        raise HTTPException(status_code=503, detail=(
            f"Generation incomplete: {len(raw_paths)}/{count} candidates; "
            "check Z-Image weights and GPU logs. No mock fallback was created."
        ))

    return {
        "asset_id": asset_id,
        "backend": chosen_backend,
        "count_generated": len(raw_paths),
        "raw_candidates": raw_paths,
        "status": status,
        "next_step": f"Use rank_candidates('{asset_id}') to choose a candidate"
    }


# Backwards compatibility alias
@mcp.tool()
async def generate_asset(
    asset_id: str,
    backend: str = "auto",
    candidates: int = 4,
    seed: int | None = None,
    creator_id: str | None = None,
) -> dict[str, Any]:
    """Compatibility alias: generate raw candidates; select one before refinement."""
    return await generate_candidates(asset_id, count=candidates, seed=seed, backend=backend, creator_id=creator_id)


# --------------------------------------------------------------------------
# MCP TOOLS: RANKING & SELECTION
# --------------------------------------------------------------------------

@mcp.tool()
def rank_candidates(asset_id: str) -> dict[str, Any]:
    """[Lifecycle Step 3 - Evaluation] Evaluates and scores generated candidates in assets/work/<asset_id>/raw/ on framing, occupancy, silhouette clarity, and composition. Returns ranked list and recommended candidate."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    raw_files = [p for p in storage.latest_raw_candidates()
                 if not (storage.review_dir / f"{p.stem}.rejected.json").is_file()]
    if not raw_files:
        return {"error": f"No raw candidates found for {asset_id}. Run generate_candidates first."}

    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)

    rankings = []
    for c_file in raw_files:
        report = evaluate_asset_candidate(c_file, asset, palette=palette, is_raw=True)
        # Compute quality score based on metrics
        occ_x = report["metrics"].get("occupancy_x", 0.5)
        occ_y = report["metrics"].get("occupancy_y", 0.5)
        score = 1.0
        if report["art_status"]["reasons"]:
            score -= 0.15 * len(report["art_status"]["reasons"])
        if report["technical_status"]["reasons"]:
            score -= 0.20 * len(report["technical_status"]["reasons"])
        rankings.append({
            "candidate": c_file.name,
            "path": str(c_file),
            "score": round(max(0.0, score), 3),
            "occupancy": [occ_x, occ_y],
            "art_status": report["art_status"],
            "technical_status": report["technical_status"]
        })

    rankings.sort(key=lambda r: r["score"], reverse=True)
    best_cand = rankings[0]["candidate"] if rankings else None

    return {
        "asset_id": asset_id,
        "ranked_candidates": rankings,
        "recommended_candidate": best_cand,
        "selection_command": f"select_candidate('{asset_id}', '{best_cand}')"
    }


@mcp.tool()
def select_candidate(asset_id: str, candidate_id: str | int) -> dict[str, Any]:
    """[Lifecycle Step 4a - Selection] Promotes the chosen candidate from raw/ to selected/ for pixel refinement."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    try:
        cand_path = storage.resolve_candidate(candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not cand_path.is_file():
        raise HTTPException(status_code=404, detail=f"Candidate {cand_path} does not exist")
    if (storage.review_dir / f"{cand_path.stem}.rejected.json").is_file():
        raise HTTPException(status_code=400, detail="Candidate has been rejected")

    try:
        selected = storage.select_candidate(cand_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    storage.save_review_report({
        "asset_id": asset_id, "production_status": "candidate",
        "technical_status": {"value": "pending", "reasons": ["Selection changed; refine again"]},
        "art_status": {"value": "pending", "reasons": ["Selection changed; review again"]},
    })
    return {
        "asset_id": asset_id,
        "selected_candidate": str(selected),
        "status": "selected",
        "next_step": f"refine_candidate('{asset_id}')"
    }


# --------------------------------------------------------------------------
# MCP TOOLS: PIXEL REFINEMENT & PREVIEW
# --------------------------------------------------------------------------

@mcp.tool()
async def refine_candidate(
    asset_id: str,
    candidate_id: str | int | None = None
) -> dict[str, Any]:
    """[Lifecycle Step 4b - Refinement] Transforms raw candidate into crisp logical pixel art with edge-aware downsampling, noise cleanup, and palette mapping. Produces 1x, 2x, 4x previews."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)

    # Find source candidate
    src_file: Path | None = None
    if candidate_id is not None:
        try:
            src_file = storage.resolve_candidate(candidate_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        src_file = storage.selected_candidate()

    if not src_file or not src_file.is_file():
        raise HTTPException(status_code=404, detail=f"No candidate found for {asset_id}")
    raw_stem = src_file.stem.removeprefix("selected_")
    if (storage.review_dir / f"{raw_stem}.rejected.json").is_file():
        raise HTTPException(status_code=400, detail="Candidate has been rejected")

    while True:
        refinement_id = uuid.uuid4().hex
        refined_path = storage.get_refined_path(suffix=f"refined-{raw_stem}-{refinement_id}")
        if not refined_path.exists():
            break
    refine_metrics = refine_pixel_candidate(src_file, asset, palette, refined_path)
    report = evaluate_asset_candidate(refined_path, asset, palette=palette)
    report["source_candidate"] = str(src_file)
    report["source_sha256"] = hashlib.sha256(src_file.read_bytes()).hexdigest()
    report["creator_id"] = storage.creator_for_candidate(src_file)
    storage.save_review_report(report)

    return {
        "asset_id": asset_id,
        "source_candidate": str(src_file),
        "refinement_id": refinement_id,
        "refined_output": refine_metrics["output_path"],
        "previews": refine_metrics["previews"],
        "metrics": refine_metrics,
        "qa_report": report,
        "production_status": report["production_status"]
    }


@mcp.tool()
def preview_asset(asset_id: str, scales: list[int] = [1, 2, 4]) -> dict[str, Any]:
    """Returns absolute paths for 1x, 2x, and 4x previews of the refined asset."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    report_path = storage.review_dir / "report.json"
    qa_report = json_load(report_path) if report_path.is_file() else {}
    refined_base = _refined_output(storage, qa_report)
    previews = {}
    for s in scales:
        p = refined_base.with_name(f"{refined_base.stem}_{s}x.png")
        if p.is_file():
            previews[f"{s}x"] = str(p)

    return {
        "asset_id": asset_id,
        "previews": previews,
        "qa_report": qa_report,
        "production_status": qa_report.get("production_status", "unknown")
    }


# --------------------------------------------------------------------------
# MCP TOOLS: ART REVIEW & APPROVAL GATE
# --------------------------------------------------------------------------

@mcp.tool()
def pixel_art_review(
    asset_id: str,
    status: str,
    reviewer_id: str,
    reasons: list[str] = [],
) -> dict[str, Any]:
    """Record the independent pixel-art quality verdict for the current refined image."""
    if status not in ("pass", "fail") or not reviewer_id.strip():
        raise HTTPException(status_code=400, detail="status and reviewer_id are required")
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    report_path = storage.review_dir / "report.json"
    report = json_load(report_path) if report_path.is_file() else {}
    refined_path = _refined_output(storage, report)
    source = Path(report.get("source_candidate", ""))
    if source.is_file() and _candidate_rejection(storage, source):
        raise HTTPException(status_code=400, detail="Rejected source candidate cannot be reviewed")
    if (not refined_path.is_file() or not source.is_file()
            or report.get("sha256") != hashlib.sha256(refined_path.read_bytes()).hexdigest()
            or report.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()):
        raise HTTPException(status_code=400, detail="Selection or refined image changed; refine before review")
    if status == "pass" and (report.get("technical_status", {}).get("value") != "pass"
                             or not report.get("creator_id") or reviewer_id == report["creator_id"]):
        raise HTTPException(status_code=400, detail="Technical PASS and an independent reviewer are required")
    report["pixel_review"] = {"value": status, "reviewer_id": reviewer_id,
                              "identity_verified": False, "sha256": report["sha256"], "reasons": reasons}
    report["art_status"] = {"value": "pending", "reasons": ["Art Director review required"]}
    report["production_status"] = "candidate"
    storage.save_review_report(report)
    return {"asset_id": asset_id, "pixel_review": report["pixel_review"], "production_status": "candidate"}


@mcp.tool()
def art_review(
    asset_id: str,
    status: str,
    reasons: list[str] = [],
    reviewer_id: str | None = None,
) -> dict[str, Any]:
    """[Lifecycle Step 5 - Art Gate] Records Art Director verdict ('pass' or 'fail' with reasons). Evaluates silhouette, adult proportions, age readability, material textures, Baltic atmosphere, no chibi/babyface."""
    if status not in ("pass", "fail"):
        raise HTTPException(status_code=400, detail="status must be 'pass' or 'fail'")

    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)
    report_path = storage.review_dir / "report.json"
    previous = json_load(report_path) if report_path.is_file() else {}
    refined_path = _refined_output(storage, previous)
    source = Path(previous.get("source_candidate", ""))
    if source.is_file() and _candidate_rejection(storage, source):
        raise HTTPException(status_code=400, detail="Rejected source candidate cannot be reviewed")
    if (not refined_path.is_file() or not source.is_file()
            or previous.get("sha256") != hashlib.sha256(refined_path.read_bytes()).hexdigest()
            or previous.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()):
        raise HTTPException(status_code=400, detail="Selection or refined image changed; refine before review")
    pixel_review = previous.get("pixel_review", {})
    if status == "pass" and (
        pixel_review.get("value") != "pass" or pixel_review.get("sha256") != previous.get("sha256")
        or not reviewer_id or reviewer_id in (previous.get("creator_id"), pixel_review.get("reviewer_id"))
    ):
        raise HTTPException(status_code=400, detail="Independent pixel-art and Art Director reviews are required")

    override = {"status": status, "reasons": reasons}
    report = evaluate_asset_candidate(refined_path, asset, palette=palette, art_verdict_override=override)
    report["source_candidate"] = str(source)
    report["source_sha256"] = previous["source_sha256"]
    report["creator_id"] = previous.get("creator_id")
    report["pixel_review"] = pixel_review
    report["art_reviewer_id"] = reviewer_id
    report["art_reviewer_identity_verified"] = False
    storage.save_review_report(report)

    return {
        "asset_id": asset_id,
        "art_status": report["art_status"],
        "technical_status": report["technical_status"],
        "production_status": report["production_status"],
        "can_be_approved": False,
        "approval_blocker": "MCP review IDs are unverified assertions; production also requires editable source and project manifest integration."
    }


@mcp.tool()
def check_asset_contract(asset_id: str) -> dict[str, Any]:
    """Verifies that an asset strictly meets objective engine runtime contracts (size, binary alpha, frame grid)."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)

    # Prefer approved, then refined
    target = storage.approved_dir / f"{asset_id}.png"
    if not target.is_file():
        report_path = storage.review_dir / "report.json"
        report = json_load(report_path) if report_path.is_file() else {}
        target = _refined_output(storage, report)

    report = evaluate_asset_candidate(target, asset, palette=palette)
    return {
        "asset_id": asset_id,
        "target_file": str(target),
        "technical_contract_pass": report["technical_status"]["value"] == "pass",
        "technical_reasons": report["technical_status"]["reasons"],
        "metrics": report["metrics"]
    }


@mcp.tool()
def approve_asset(asset_id: str) -> dict[str, Any]:
    """[Lifecycle Step 6 - Production Approval] Promotes the refined asset to production assets/source/ ONLY if both technical_status and art_status are PASS."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    report_path = storage.review_dir / "report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=400, detail="Cannot approve asset without QA review report.")

    report = json_load(report_path)
    if report.get("art_status", {}).get("value") != "pass":
        return {
            "approved": False,
            "error": "Art review has not passed (art_status != pass). Cannot promote to production source.",
            "art_reasons": report.get("art_status", {}).get("reasons", [])
        }

    if report.get("technical_status", {}).get("value") != "pass":
        return {
            "approved": False,
            "error": "Technical contract checks failed (technical_status != pass).",
            "technical_reasons": report.get("technical_status", {}).get("reasons", [])
        }

    pixel_review = report.get("pixel_review", {})
    creator = report.get("creator_id")
    art_reviewer = report.get("art_reviewer_id")
    pixel_reviewer = pixel_review.get("reviewer_id")
    if (pixel_review.get("value") != "pass" or pixel_review.get("sha256") != report.get("sha256")
            or not all((creator, art_reviewer, pixel_reviewer))
            or len({creator, art_reviewer, pixel_reviewer}) != 3):
        return {"approved": False, "error": "Independent author, pixel reviewer and Art Director required."}

    refined_path = _refined_output(storage, report)
    source = Path(report.get("source_candidate", ""))
    if source.is_file() and _candidate_rejection(storage, source):
        return {"approved": False, "error": "Rejected source candidate cannot be approved."}
    if (not refined_path.is_file() or not source.is_file()
            or report.get("sha256") != hashlib.sha256(refined_path.read_bytes()).hexdigest()
            or report.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()):
        return {"approved": False, "error": "Reviewed files changed; refine and review again."}
    # A reviewed PNG remains a raster candidate. The project requires editable
    # source, manifest and deterministic build before canonical promotion.
    return {
        "approved": False,
        "asset_id": asset_id,
        "candidate_path": str(refined_path),
        "error": "Review gates passed, but production promotion requires editable source and project manifest integration.",
        "production_status": "candidate"
    }


@mcp.tool()
def reject_candidate(
    asset_id: str,
    candidate_id: str | int,
    reasons: list[str] = []
) -> dict[str, Any]:
    """Rejects a candidate, recording reasons."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    try:
        candidate = storage.resolve_candidate(candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Candidate not found")
    rejection = {"asset_id": asset_id, "candidate_id": candidate.name,
                 "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                 "status": "rejected", "reasons": reasons}
    rejection_path = storage.review_dir / f"{candidate.stem}.rejected.json"
    if rejection_path.exists():
        raise HTTPException(status_code=400, detail="Candidate already rejected")
    rejection_path.write_text(json.dumps(rejection, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = storage.review_dir / "report.json"
    if report_path.is_file():
        report = json_load(report_path)
        source_name = Path(report.get("source_candidate", "")).name.removeprefix("selected_")
        if source_name == candidate.name:
            report["production_status"] = "candidate"
            report["art_status"] = {"value": "fail", "reasons": reasons}
            report["pixel_review"] = {"value": "fail", "reasons": ["Source candidate rejected"]}
            storage.save_review_report(report)
    return {
        "asset_id": asset_id,
        "candidate_id": candidate_id,
        "status": "rejected",
        "reasons": reasons
    }


@mcp.tool()
async def regenerate_asset(
    asset_id: str,
    correction: str | None = None,
    seed: int | None = None,
    creator_id: str | None = None,
) -> dict[str, Any]:
    """Re-generates raw candidates applying optional corrections to the positive prompt."""
    return await generate_candidates(asset_id, count=4, seed=seed, correction=correction, creator_id=creator_id)


# --------------------------------------------------------------------------
# MCP TOOLS: SPRITE SHEETS & DAY/NIGHT
# --------------------------------------------------------------------------

@mcp.tool()
async def generate_sprite_frames(
    asset_id: str,
    backend: str = "auto",
    frames: int | None = None,
    seed: int | None = None
) -> dict[str, Any]:
    """Builds procedural character animation frames, normalizes them, and prepares a sprite sheet."""
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)

    import uuid
    run_id = uuid.uuid4().hex
    char_frames, (fw, fh), cols = generate_character_sprite_frames(
        asset, seed=seed if seed is not None else 26063, frames=frames, palette=palette, repo_root=REPO_ROOT
    )

    run_dir = storage.raw_dir / f"sprite-run-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_paths = []
    for f_idx, frame_img in enumerate(char_frames):
        fn = run_dir / f"frame_{f_idx:02d}.png"
        frame_img.save(fn)
        norm_dir = storage.refined_dir / f"sprite-run-{run_id}"
        norm_dir.mkdir(parents=True, exist_ok=True)
        norm_fn = norm_dir / f"frame_{f_idx:02d}_norm.png"
        refine_pixel_candidate(fn, {"size": [fw, fh], "alpha": asset.get("alpha", "transparent")}, palette, norm_fn)
        frame_paths.append(norm_fn)

    sheet_out = storage.get_refined_path(suffix=f"sheet-{run_id}")
    sheet_info = assemble_sprite_sheet(frame_paths, sheet_out, (fw, fh), columns=cols)
    report = evaluate_asset_candidate(sheet_out, asset, palette=palette)
    storage.save_review_report(report)

    return {
        "asset_id": asset_id,
        "run_id": run_id,
        "backend": "procedural_character_synthesizer",
        "source_frames": [str(path) for path in frame_paths],
        "frames_generated": len(frame_paths),
        "sheet": sheet_info,
        "qa_report": report
    }


@mcp.tool()
async def generate_day_night_pair(
    day_asset_id: str,
    night_asset_id: str,
    backend: str = "auto",
    seed: int | None = None
) -> dict[str, Any]:
    """Generates day and night building variants with locked geometry silhouette."""
    raise HTTPException(
        status_code=501,
        detail="Day/night generation is temporarily disabled until both outputs use immutable, provenance-linked candidate IDs.",
    )


# --------------------------------------------------------------------------
# MCP TOOLS: SYSTEM STATUS
# --------------------------------------------------------------------------

@mcp.tool()
def gpu_status() -> dict[str, Any]:
    """Reports status, VRAM, and availability of GPU 0 and local AI models."""
    vram_info = "unknown"
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        vram_info = res.stdout.strip()
    except Exception:
        pass

    return {
        "gpu": vram_info,
        "z_image_available": is_z_image_available(),
        "comfyui_online": comfy.is_online(),
        "work_dir": str(WORK_DIR),
        "spec_path": str(SPEC_PATH)
    }


@mcp.tool()
def queue_status() -> dict[str, Any]:
    """Reports active asset generation tasks and work directory assets."""
    assets_in_work = [d.name for d in WORK_DIR.iterdir() if d.is_dir()]
    return {
        "assets_in_work_count": len(assets_in_work),
        "assets_in_work": sorted(assets_in_work)
    }


def json_load(p: Path) -> dict[str, Any]:
    import json
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# HTTP REST & SSE RUNNER
# --------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request):
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "online",
        "service": "synopsis-assets-mcp",
        "version": "2.0.0",
        "z_image_available": is_z_image_available(),
    })


@mcp.custom_route("/api/assets/profiles", methods=["GET"])
async def api_get_profiles(request: Request):
    from starlette.responses import JSONResponse
    return JSONResponse(list_art_profiles())


@mcp.custom_route("/api/assets/spec/{asset_id}", methods=["GET"])
async def api_get_asset_spec(request: Request):
    from starlette.responses import JSONResponse
    asset_id = request.path_params.get("asset_id", "")
    return JSONResponse(asset_spec(asset_id))


app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=CONFIG["server"]["host"],
        port=CONFIG["server"]["port"],
        log_level="info"
    )
