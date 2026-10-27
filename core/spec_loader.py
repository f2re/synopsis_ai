from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from core.asset_classes import PALETTE_ROLE_MAP, classify_asset
from core.prompt_compiler import compile_generation_prompts

STYLE_LOCK = (
    "STYLE LOCK: modern high-quality pixel art (reference feel: Eastward, Sea of Stars, A Short Hike), "
    "crisp hard-edged square pixels on a strict logical pixel grid, deliberate clusters of 2+ pixels, "
    "no anti-aliasing blur, no soft gradients (only ordered 2x2 dithering, max 3 bands), no noise, "
    "no photographic texture, no painterly brush strokes, no 3D-render look. Light from upper-left. "
    "Outlines are a darker shade of the local colour, never pure black. "
    "World: fictional cozy scientific meteorological campus on the shore of the Gulf of Finland near "
    "St Petersburg, North-West Russia: flat low horizon, water, birch and pine, granite embankments, "
    "ochre and terracotta facades, warm amber windows against a cool blue night. "
    "Absolutely no mountains or hills, no palms, no text, letters, digits or logos anywhere in the image, "
    "no UI, no watermark, no signature, no frame around the picture."
)

NEG_COMMON = (
    "blurry, anti-aliased, smooth gradient, painterly, oil painting, watercolor, 3d render, photo, "
    "noise, jpeg artifacts, text, letters, numbers, logo, watermark, signature, mountains, hills, "
    "alps, palm trees, chromatic aberration, bloom, lens flare, isometric perspective drift"
)


def load_spec(spec_path: Path) -> dict[str, Any]:
    if not spec_path.is_file():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def load_gpl(palette_path: Path) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    if not palette_path.is_file():
        return colors
    for line in palette_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            colors.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return colors


def get_asset_by_id(spec: dict[str, Any], asset_id: str) -> dict[str, Any]:
    # 1. Exact match
    for asset in spec.get("assets", []):
        if asset["id"] == asset_id:
            return asset
            
    # 2. Normalized match (dashes, underscores, dots, uppercase)
    norm_id = asset_id.replace("_", "-").replace(".", "-").upper()
    for asset in spec.get("assets", []):
        if asset["id"].upper() == norm_id:
            return asset

    # 3. Known aliases for avatars & characters
    if "AVATAR" in norm_id or "CHAR" in norm_id or "PORTRAIT" in norm_id or "MENTOR" in norm_id or "LADOZH" in norm_id:
        if "M2" in norm_id:
            for asset in spec.get("assets", []):
                if asset["id"] == "AVATAR-BASE-M2":
                    return asset
        elif "M1" in norm_id:
            for asset in spec.get("assets", []):
                if asset["id"] == "AVATAR-BASE-M1":
                    return asset
        elif "MENTOR" in norm_id or "LADOZH" in norm_id or "VERA" in norm_id:
            if "PORTRAIT" in norm_id and "SHEET" not in norm_id and norm_id != "MENTOR-PORTRAIT":
                return {
                    "id": asset_id,
                    "path": "assets/source/characters/npc/portrait_vera_ladozhskaya.png",
                    "size": [128, 128],
                    "frame": None,
                    "alpha": "transparent",
                    "max_colors": 24,
                    "people": True,
                    "class": "portrait",
                    "palette_role": "people"
                }
            for asset in spec.get("assets", []):
                if asset["id"] == "MENTOR-PORTRAIT":
                    return asset

    # 4. Fallback search in assets/manifests
    manifest_dirs = [
        Path(__file__).resolve().parents[3] / "assets/manifests",
        Path(__file__).resolve().parents[1] / "specs/manifests",
        Path(__file__).resolve().parents[1] / "manifests",
        Path("/home/user/synopsis-ai/synopsis_game/assets/manifests"),
        Path("/home/user/synopsis-ai/specs/manifests"),
        Path("/home/user/synopsis-ai/manifests"),
    ]
    for mdir in manifest_dirs:
        if mdir.is_dir():
            for mf in mdir.glob("*.json"):
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                    if data.get("id") == asset_id or mf.stem == asset_id or mf.stem in asset_id:
                        size = data.get("logicalSize", {})
                        w = size.get("width", 128)
                        h = size.get("height", 128)
                        fw = 32 if w % 32 == 0 else 16
                        fh = 48 if h % 48 == 0 else 24
                        return {
                            "id": asset_id,
                            "path": data.get("source", f"assets/source/{asset_id}.png"),
                            "size": [w, h],
                            "frame": [fw, fh],
                            "alpha": "transparent",
                            "max_colors": 24,
                            "people": "npc" in asset_id or "char" in asset_id or "avatar" in asset_id
                        }
                except Exception:
                    continue

    raise KeyError(f"Asset ID '{asset_id}' not found in registry")


def get_palette_for_asset(asset: dict[str, Any], palette_dir: Path) -> list[tuple[int, int, int]]:
    """Loads the specialized palette family for this asset based on palette_role or class."""
    role = asset.get("palette_role")
    if not role:
        aclass = classify_asset(asset.get("id", ""), asset)
        role = PALETTE_ROLE_MAP.get(aclass, "master")

    # Map role to filename
    file_map = {
        "architecture": "architecture_26063_v1.gpl",
        "people": "people_26063_v2.gpl",
        "props": "props_26063_v1.gpl",
        "terrain": "terrain_26063_v1.gpl",
        "environment": "environment_26063_v1.gpl",
        "ui": "ui_26063_v1.gpl",
        "effects": "environment_26063_v1.gpl",
        "master": "master_26063_v3.gpl",
    }
    fname = file_map.get(role, "master_26063_v3.gpl")
    pal_path = palette_dir / fname
    if not pal_path.is_file():
        pal_path = palette_dir / "master_26063_v2.gpl"

    palette = load_gpl(pal_path)
    if not palette:
        palette = load_gpl(palette_dir / "master_26063_v2.gpl")
    return palette


def build_generation_prompts(asset: dict[str, Any], style_lora_trigger: str = "<26063_pixel_style>") -> tuple[str, str]:
    pos, neg, _ = compile_generation_prompts(asset.get("id", ""), asset, style_trigger=style_lora_trigger)
    return pos, neg
