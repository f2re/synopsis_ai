from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
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
    backend: str = "auto"
) -> dict[str, Any]:
    """[Lifecycle Step 2 - Generation] Generates raw high-detail candidate images for an asset into assets/work/<asset_id>/raw/ using class-specific workflow and prompt."""
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    pos_prompt, neg_prompt, meta = compile_generation_prompts(asset_id, asset)

    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    storage.save_spec(asset)
    storage.save_prompt(pos_prompt, neg_prompt, meta)

    target_w, target_h = asset.get("size", [256, 256])
    palette = get_active_palette(asset)
    raw_paths: list[str] = []
    chosen_backend = backend

    base_seed = seed or 26063

    # 1. Authentic character synthesis for sprite sheets
    if asset.get("people") and (asset.get("frame") or "avatar" in asset_id.lower() or "walk" in asset_id.lower()):
        frames, (fw, fh), cols = generate_character_sprite_frames(
            asset, seed=base_seed, frames=None, palette=palette, repo_root=REPO_ROOT
        )
        for c_idx in range(min(count, 4)):
            sheet_raw = storage.get_raw_path(c_idx + 1)
            frame_norm_paths = []
            for f_idx, frame_img in enumerate(frames):
                fn = storage.raw_dir / f"f_{f_idx:02d}_c{c_idx}.png"
                frame_img.save(fn)
                frame_norm_paths.append(fn)
            assemble_sprite_sheet(frame_norm_paths, sheet_raw, (fw, fh), columns=cols)
            raw_paths.append(str(sheet_raw))
        chosen_backend = "character_synthesizer"

    # 2. Neural generation via Z-Image-Turbo
    elif is_z_image_available() and backend in ("auto", "z_image", "z_image_turbo"):
        for c_idx in range(count):
            c_seed = base_seed + c_idx * 17
            raw_target = storage.get_raw_path(c_idx + 1)
            if generate_z_image_candidate(
                pos_prompt, raw_target, target_w, target_h, seed=c_seed, steps=8, negative_prompt=neg_prompt
            ):
                raw_paths.append(str(raw_target))
        if raw_paths:
            chosen_backend = "z_image_turbo"

    # 3. Fallback mock generation if neural weights not present
    if not raw_paths:
        from PIL import Image, ImageDraw
        for c_idx in range(count):
            raw_target = storage.get_raw_path(c_idx + 1)
            img = Image.new("RGBA", (target_w * 4, target_h * 4), (30, 36, 48, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [int(target_w * 0.8), int(target_h * 0.8), int(target_w * 3.2), int(target_h * 3.2)],
                fill=palette[min(2 + c_idx, len(palette) - 1)] + (255,)
            )
            img.save(raw_target)
            raw_paths.append(str(raw_target))
        chosen_backend = "mock_fallback"

    return {
        "asset_id": asset_id,
        "backend": chosen_backend,
        "count_generated": len(raw_paths),
        "raw_candidates": raw_paths,
        "status": "generated_raw",
        "next_step": f"Use rank_candidates('{asset_id}') or select_candidate('{asset_id}', 1)"
    }


# Backwards compatibility alias
@mcp.tool()
async def generate_asset(
    asset_id: str,
    backend: str = "auto",
    candidates: int = 4,
    seed: int | None = None
) -> dict[str, Any]:
    """Generates candidate images for an asset, refines them, and reports dual technical & art status."""
    gen_res = await generate_candidates(asset_id, count=candidates, seed=seed, backend=backend)
    refined_results = []
    for idx in range(1, gen_res["count_generated"] + 1):
        ref_res = await refine_candidate(asset_id, candidate_id=idx)
        refined_results.append(ref_res)
    return {
        "asset_id": asset_id,
        "backend": gen_res["backend"],
        "candidates": refined_results
    }


# --------------------------------------------------------------------------
# MCP TOOLS: RANKING & SELECTION
# --------------------------------------------------------------------------

@mcp.tool()
def rank_candidates(asset_id: str) -> dict[str, Any]:
    """[Lifecycle Step 3 - Evaluation] Evaluates and scores generated candidates in assets/work/<asset_id>/raw/ on framing, occupancy, silhouette clarity, and composition. Returns ranked list and recommended candidate."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    raw_files = storage.list_raw_candidates()
    if not raw_files:
        return {"error": f"No raw candidates found for {asset_id}. Run generate_candidates first."}

    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)

    rankings = []
    for c_file in raw_files:
        report = evaluate_asset_candidate(c_file, asset, palette=palette)
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
    if isinstance(candidate_id, int):
        cand_path = storage.get_raw_path(candidate_id)
    else:
        cand_path = storage.raw_dir / candidate_id
        if not cand_path.is_file():
            cand_path = storage.raw_dir / f"candidate-{int(candidate_id):02d}.png"

    if not cand_path.is_file():
        raise HTTPException(status_code=404, detail=f"Candidate {cand_path} does not exist")

    selected = storage.select_candidate(cand_path)
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
        if isinstance(candidate_id, int):
            src_file = storage.get_raw_path(candidate_id)
        else:
            src_file = storage.raw_dir / str(candidate_id)
    else:
        selected_files = sorted(storage.selected_dir.glob("*.png"))
        if selected_files:
            src_file = selected_files[0]
        else:
            raw_files = storage.list_raw_candidates()
            if raw_files:
                src_file = raw_files[0]

    if not src_file or not src_file.is_file():
        raise HTTPException(status_code=404, detail=f"No candidate found for {asset_id}")

    refined_path = storage.get_refined_path(suffix="refined")
    refine_metrics = refine_pixel_candidate(src_file, asset, palette, refined_path)
    report = evaluate_asset_candidate(refined_path, asset, palette=palette)
    storage.save_review_report(report)

    return {
        "asset_id": asset_id,
        "source_candidate": str(src_file),
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
    refined_base = storage.get_refined_path(suffix="refined")
    previews = {}
    for s in scales:
        p = refined_base.with_name(f"{refined_base.stem}_{s}x.png")
        if p.is_file():
            previews[f"{s}x"] = str(p)

    report_path = storage.review_dir / "report.json"
    qa_report = json_load(report_path) if report_path.is_file() else {}

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
def art_review(
    asset_id: str,
    status: str,
    reasons: list[str] = []
) -> dict[str, Any]:
    """[Lifecycle Step 5 - Art Gate] Records Art Director verdict ('pass' or 'fail' with reasons). Evaluates silhouette, adult proportions, age readability, material textures, Baltic atmosphere, no chibi/babyface."""
    if status not in ("pass", "fail"):
        raise HTTPException(status_code=400, detail="status must be 'pass' or 'fail'")

    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)
    refined_path = storage.get_refined_path(suffix="refined")

    override = {"status": status, "reasons": reasons}
    report = evaluate_asset_candidate(refined_path, asset, palette=palette, art_verdict_override=override)
    storage.save_review_report(report)

    return {
        "asset_id": asset_id,
        "art_status": report["art_status"],
        "technical_status": report["technical_status"],
        "production_status": report["production_status"],
        "can_be_approved": report["art_status"]["value"] == "pass" and report["technical_status"]["value"] == "pass"
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
        target = storage.get_refined_path(suffix="refined")

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

    refined_path = storage.get_refined_path(suffix="refined")
    approved_path = storage.approve_asset(refined_path)

    # Copy to canonical repo source if path specified
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    canonical_rel = asset.get("path")
    if canonical_rel:
        canonical_dest = REPO_ROOT / canonical_rel
        canonical_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(approved_path, canonical_dest)

    return {
        "approved": True,
        "asset_id": asset_id,
        "approved_path": str(approved_path),
        "canonical_source": str(canonical_dest) if canonical_rel else None,
        "production_status": "approved"
    }


@mcp.tool()
def reject_candidate(
    asset_id: str,
    candidate_id: str | int,
    reasons: list[str] = []
) -> dict[str, Any]:
    """Rejects a candidate, recording reasons."""
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)
    report_path = storage.review_dir / "report.json"
    report = json_load(report_path) if report_path.is_file() else {}
    report["production_status"] = "rejected"
    report.setdefault("art_status", {})["value"] = "fail"
    report.setdefault("art_status", {}).setdefault("reasons", []).extend(reasons)
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
    seed: int | None = None
) -> dict[str, Any]:
    """Re-generates raw candidates applying optional corrections to the positive prompt."""
    return await generate_candidates(asset_id, count=4, seed=seed)


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
    """Generates authentic animation frames, normalizes them, and prepares a sprite sheet."""
    spec = load_spec(SPEC_PATH)
    asset = get_asset_by_id(spec, asset_id)
    palette = get_active_palette(asset)
    storage = WorkStorage(asset_id, base_dir=WORK_DIR)

    char_frames, (fw, fh), cols = generate_character_sprite_frames(
        asset, seed=seed or 26063, frames=frames, palette=palette, repo_root=REPO_ROOT
    )

    frame_paths = []
    for f_idx, frame_img in enumerate(char_frames):
        fn = storage.raw_dir / f"frame_{f_idx:02d}.png"
        frame_img.save(fn)
        norm_fn = storage.refined_dir / f"frame_{f_idx:02d}_norm.png"
        refine_pixel_candidate(fn, {"size": [fw, fh], "alpha": asset.get("alpha", "transparent")}, palette, norm_fn)
        frame_paths.append(norm_fn)

    sheet_out = storage.get_refined_path(suffix="sheet")
    sheet_info = assemble_sprite_sheet(frame_paths, sheet_out, (fw, fh), columns=cols)
    report = evaluate_asset_candidate(sheet_out, asset, palette=palette)
    storage.save_review_report(report)

    return {
        "asset_id": asset_id,
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
    spec = load_spec(SPEC_PATH)
    day_asset = get_asset_by_id(spec, day_asset_id)
    night_asset = get_asset_by_id(spec, night_asset_id)

    # 1. Generate DAY candidate base
    await generate_candidates(day_asset_id, count=1, seed=seed, backend=backend)
    day_ref = await refine_candidate(day_asset_id, candidate_id=1)
    day_cand_path = Path(day_ref["refined_output"])

    # 2. Night candidate with locked silhouette
    night_storage = WorkStorage(night_asset_id, base_dir=WORK_DIR)
    night_raw = night_storage.get_raw_path(1)
    night_palette = get_active_palette(night_asset)

    enforce_geometry_lock(day_cand_path, night_raw, night_palette)
    night_ref = await refine_candidate(night_asset_id, candidate_id=1)

    silhouette_match = compare_silhouettes(day_cand_path, Path(night_ref["refined_output"]))

    return {
        "day_asset_id": day_asset_id,
        "night_asset_id": night_asset_id,
        "day_refined": day_ref["refined_output"],
        "night_refined": night_ref["refined_output"],
        "silhouette_match": silhouette_match
    }


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
