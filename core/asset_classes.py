from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Canonical 14 asset generation classes from config/art-generation-profiles.json
CLASS_DIALOGUE_PORTRAIT = "dialogue_portrait"
CLASS_PROFILE_PORTRAIT = "profile_portrait"
CLASS_WORLD_CHARACTER = "world_character"
CLASS_COMPANION_SPRITE = "companion_sprite"
CLASS_ARCHITECTURE_LANDMARK = "architecture_landmark"
CLASS_ARCHITECTURE_SMALL = "architecture_small"
CLASS_INTERACTIVE_OBJECT = "interactive_object"
CLASS_PROP_SPRITE = "prop_sprite"
CLASS_TERRAIN_TILE = "terrain_tile"
CLASS_VEGETATION_SPRITE = "vegetation_sprite"
CLASS_UI_ICON = "ui_icon"
CLASS_UI_FRAME = "ui_frame"
CLASS_VFX_SMALL = "vfx_small"
CLASS_LARGE_SCENE = "large_scene"

# Backwards-compatible aliases
CLASS_PORTRAIT = "portrait"
CLASS_CHARACTER_SPRITE = "character_sprite"
CLASS_ARCHITECTURE = "architecture"
CLASS_PROP = "prop"
CLASS_TERRAIN = "terrain"
CLASS_VEGETATION = "vegetation"
CLASS_UI = "ui"

ALL_CANONICAL_CLASSES = [
    CLASS_DIALOGUE_PORTRAIT,
    CLASS_PROFILE_PORTRAIT,
    CLASS_WORLD_CHARACTER,
    CLASS_COMPANION_SPRITE,
    CLASS_ARCHITECTURE_LANDMARK,
    CLASS_ARCHITECTURE_SMALL,
    CLASS_INTERACTIVE_OBJECT,
    CLASS_PROP_SPRITE,
    CLASS_TERRAIN_TILE,
    CLASS_VEGETATION_SPRITE,
    CLASS_UI_ICON,
    CLASS_UI_FRAME,
    CLASS_VFX_SMALL,
    CLASS_LARGE_SCENE,
]

# Default occupancy contracts
OCCUPANCY_CONTRACTS: dict[str, dict[str, Any]] = {
    CLASS_DIALOGUE_PORTRAIT: {
        "min_x": 0.50,
        "max_x": 0.95,
        "min_y": 0.65,
        "max_y": 0.98,
        "head_height_ratio": (0.28, 0.45),
        "eye_line_y": (0.30, 0.45),
    },
    CLASS_PROFILE_PORTRAIT: {
        "min_x": 0.55,
        "max_x": 0.95,
        "min_y": 0.70,
        "max_y": 0.98,
        "head_height_ratio": (0.40, 0.58),
        "eye_line_y": (0.35, 0.50),
    },
    CLASS_WORLD_CHARACTER: {
        "min_x": 0.35,
        "max_x": 0.85,
        "min_y": 0.65,
        "max_y": 0.98,
    },
    CLASS_COMPANION_SPRITE: {
        "min_x": 0.35,
        "max_x": 0.90,
        "min_y": 0.40,
        "max_y": 0.95,
    },
    CLASS_ARCHITECTURE_LANDMARK: {
        "min_x": 0.55,
        "max_x": 0.95,
        "min_y": 0.50,
        "max_y": 0.95,
    },
    CLASS_ARCHITECTURE_SMALL: {
        "min_x": 0.45,
        "max_x": 0.90,
        "min_y": 0.45,
        "max_y": 0.90,
    },
    CLASS_INTERACTIVE_OBJECT: {
        "min_x": 0.40,
        "max_x": 0.90,
        "min_y": 0.40,
        "max_y": 0.90,
    },
    CLASS_PROP_SPRITE: {
        "min_x": 0.35,
        "max_x": 0.85,
        "min_y": 0.35,
        "max_y": 0.85,
    },
    CLASS_TERRAIN_TILE: {
        "min_x": 0.90,
        "max_x": 1.00,
        "min_y": 0.90,
        "max_y": 1.00,
    },
    CLASS_VEGETATION_SPRITE: {
        "min_x": 0.40,
        "max_x": 0.90,
        "min_y": 0.50,
        "max_y": 0.95,
    },
    CLASS_UI_ICON: {
        "min_x": 0.35,
        "max_x": 0.95,
        "min_y": 0.35,
        "max_y": 0.95,
    },
    CLASS_UI_FRAME: {
        "min_x": 0.60,
        "max_x": 1.00,
        "min_y": 0.60,
        "max_y": 1.00,
    },
    CLASS_VFX_SMALL: {
        "min_x": 0.20,
        "max_x": 0.95,
        "min_y": 0.20,
        "max_y": 0.95,
    },
    CLASS_LARGE_SCENE: {
        "min_x": 0.90,
        "max_x": 1.00,
        "min_y": 0.90,
        "max_y": 1.00,
    },
    # Backwards-compatible aliases
    CLASS_PORTRAIT: {
        "min_x": 0.55,
        "max_x": 0.95,
        "min_y": 0.70,
        "max_y": 0.98,
        "head_height_ratio": (0.35, 0.55),
        "eye_line_y": (0.35, 0.50),
    },
    CLASS_CHARACTER_SPRITE: {
        "min_x": 0.35,
        "max_x": 0.85,
        "min_y": 0.65,
        "max_y": 0.98,
    },
    CLASS_ARCHITECTURE: {
        "min_x": 0.55,
        "max_x": 0.95,
        "min_y": 0.50,
        "max_y": 0.95,
    },
    CLASS_PROP: {
        "min_x": 0.35,
        "max_x": 0.85,
        "min_y": 0.35,
        "max_y": 0.85,
    },
    CLASS_TERRAIN: {
        "min_x": 0.90,
        "max_x": 1.00,
        "min_y": 0.90,
        "max_y": 1.00,
    },
    CLASS_VEGETATION: {
        "min_x": 0.40,
        "max_x": 0.90,
        "min_y": 0.50,
        "max_y": 0.95,
    },
    CLASS_UI: {
        "min_x": 0.35,
        "max_x": 0.95,
        "min_y": 0.35,
        "max_y": 0.95,
    },
}

PALETTE_ROLE_MAP: dict[str, str] = {
    CLASS_DIALOGUE_PORTRAIT: "people",
    CLASS_PROFILE_PORTRAIT: "people",
    CLASS_WORLD_CHARACTER: "people",
    CLASS_COMPANION_SPRITE: "people",
    CLASS_ARCHITECTURE_LANDMARK: "architecture",
    CLASS_ARCHITECTURE_SMALL: "architecture",
    CLASS_INTERACTIVE_OBJECT: "props",
    CLASS_PROP_SPRITE: "props",
    CLASS_TERRAIN_TILE: "terrain",
    CLASS_VEGETATION_SPRITE: "environment",
    CLASS_UI_ICON: "ui",
    CLASS_UI_FRAME: "ui",
    CLASS_VFX_SMALL: "effects",
    CLASS_LARGE_SCENE: "environment",
    # Legacy aliases
    CLASS_PORTRAIT: "people",
    CLASS_CHARACTER_SPRITE: "people",
    CLASS_ARCHITECTURE: "architecture",
    CLASS_PROP: "props",
    CLASS_TERRAIN: "terrain",
    CLASS_VEGETATION: "environment",
    CLASS_UI: "ui",
}

PALETTE_CONTRACTS: dict[str, dict[str, Any]] = {
    CLASS_DIALOGUE_PORTRAIT: {
        "role": "people",
        "target_range": (48, 96),
        "hard_max": 128,
        "strict_family": False,
    },
    CLASS_PROFILE_PORTRAIT: {
        "role": "people",
        "target_range": (36, 72),
        "hard_max": 96,
        "strict_family": False,
    },
    CLASS_WORLD_CHARACTER: {
        "role": "people",
        "target_range": (24, 48),
        "hard_max": 64,
        "strict_family": False,
    },
    CLASS_COMPANION_SPRITE: {
        "role": "people",
        "target_range": (24, 48),
        "hard_max": 64,
        "strict_family": False,
    },
    CLASS_ARCHITECTURE_LANDMARK: {
        "role": "architecture",
        "target_range": (48, 96),
        "hard_max": 128,
        "strict_family": False,
    },
    CLASS_ARCHITECTURE_SMALL: {
        "role": "architecture",
        "target_range": (32, 72),
        "hard_max": 96,
        "strict_family": False,
    },
    CLASS_INTERACTIVE_OBJECT: {
        "role": "props",
        "target_range": (28, 64),
        "hard_max": 80,
        "strict_family": False,
    },
    CLASS_PROP_SPRITE: {
        "role": "props",
        "target_range": (24, 56),
        "hard_max": 72,
        "strict_family": False,
    },
    CLASS_TERRAIN_TILE: {
        "role": "terrain",
        "target_range": (24, 48),
        "hard_max": 64,
        "strict_family": False,
    },
    CLASS_VEGETATION_SPRITE: {
        "role": "environment",
        "target_range": (28, 64),
        "hard_max": 80,
        "strict_family": False,
    },
    CLASS_UI_ICON: {
        "role": "ui",
        "target_range": (12, 32),
        "hard_max": 40,
        "strict_family": False,
    },
    CLASS_UI_FRAME: {
        "role": "ui",
        "target_range": (24, 56),
        "hard_max": 72,
        "strict_family": False,
    },
    CLASS_VFX_SMALL: {
        "role": "effects",
        "target_range": (16, 40),
        "hard_max": 56,
        "strict_family": False,
    },
    CLASS_LARGE_SCENE: {
        "role": "environment",
        "target_range": (96, 192),
        "hard_max": 256,
        "strict_family": False,
    },
    # Legacy aliases
    CLASS_PORTRAIT: {
        "role": "people",
        "target_range": (36, 72),
        "hard_max": 96,
        "strict_family": False,
    },
    CLASS_CHARACTER_SPRITE: {
        "role": "people",
        "target_range": (24, 48),
        "hard_max": 64,
        "strict_family": False,
    },
    CLASS_ARCHITECTURE: {
        "role": "architecture",
        "target_range": (48, 96),
        "hard_max": 128,
        "strict_family": False,
    },
    CLASS_PROP: {
        "role": "props",
        "target_range": (24, 56),
        "hard_max": 72,
        "strict_family": False,
    },
    CLASS_TERRAIN: {
        "role": "terrain",
        "target_range": (24, 48),
        "hard_max": 64,
        "strict_family": False,
    },
    CLASS_VEGETATION: {
        "role": "environment",
        "target_range": (28, 64),
        "hard_max": 80,
        "strict_family": False,
    },
    CLASS_UI: {
        "role": "ui",
        "target_range": (12, 32),
        "hard_max": 40,
        "strict_family": False,
    },
}

# Dynamically load profiles from config/art-generation-profiles.json when available
GENERATION_PROFILES: dict[str, dict[str, Any]] = {}
GENERATION_POLICY: dict[str, Any] = {}


def _load_profiles_config() -> None:
    search_paths = [
        Path(__file__).resolve().parents[3] / "config" / "art-generation-profiles.json",
        Path(__file__).resolve().parents[2] / "config" / "art-generation-profiles.json",
        Path("/home/user/synopsis-ai/synopsis_game/config/art-generation-profiles.json"),
        Path.cwd() / "config" / "art-generation-profiles.json",
    ]
    for p in search_paths:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                profiles = data.get("profiles", {})
                policy = data.get("policy", {})
                GENERATION_PROFILES.update(profiles)
                GENERATION_POLICY.update(policy)

                for name, prof in profiles.items():
                    pal = prof.get("palette", {})
                    target_colors = pal.get("targetColors", [24, 48])
                    soft_max = pal.get("softMaxColors", 64)
                    role = pal.get("role", "master")
                    PALETTE_ROLE_MAP[name] = role
                    PALETTE_CONTRACTS[name] = {
                        "role": role,
                        "target_range": tuple(target_colors),
                        "hard_max": soft_max,
                        "strict_family": False,
                    }
                break
            except Exception:
                continue


_load_profiles_config()


def get_generation_profile(class_name: str) -> dict[str, Any]:
    """Returns the art-generation-profile dict for a given asset class."""
    if class_name in GENERATION_PROFILES:
        return GENERATION_PROFILES[class_name]
    # Check legacy aliases
    alias_map = {
        CLASS_PORTRAIT: CLASS_DIALOGUE_PORTRAIT,
        CLASS_CHARACTER_SPRITE: CLASS_WORLD_CHARACTER,
        CLASS_ARCHITECTURE: CLASS_ARCHITECTURE_LANDMARK,
        CLASS_PROP: CLASS_PROP_SPRITE,
        CLASS_TERRAIN: CLASS_TERRAIN_TILE,
        CLASS_VEGETATION: CLASS_VEGETATION_SPRITE,
        CLASS_UI: CLASS_UI_ICON,
    }
    canon = alias_map.get(class_name, class_name)
    return GENERATION_PROFILES.get(canon, {})


def classify_asset(asset_id: str, asset_spec: dict[str, Any] | None = None) -> str:
    """Classifies an asset into its target class based on ID and metadata."""
    aid = asset_id.upper()
    spec = asset_spec or {}

    spec_class = spec.get("class")
    if spec_class:
        if spec_class in ALL_CANONICAL_CLASSES:
            return spec_class
        if spec_class in (CLASS_PORTRAIT, "portrait"):
            size = spec.get("size", [128, 128])
            return CLASS_DIALOGUE_PORTRAIT if size[1] >= 200 else CLASS_PROFILE_PORTRAIT
        if spec_class in (CLASS_CHARACTER_SPRITE, "character_sprite"):
            return CLASS_WORLD_CHARACTER
        if spec_class in (CLASS_ARCHITECTURE, "architecture"):
            size = spec.get("size", [96, 96])
            return CLASS_ARCHITECTURE_LANDMARK if max(size) >= 80 else CLASS_ARCHITECTURE_SMALL
        if spec_class in (CLASS_PROP, "prop"):
            return CLASS_INTERACTIVE_OBJECT if "INTERACTIVE" in aid else CLASS_PROP_SPRITE
        if spec_class in (CLASS_TERRAIN, "terrain"):
            return CLASS_TERRAIN_TILE
        if spec_class in (CLASS_VEGETATION, "vegetation"):
            return CLASS_VEGETATION_SPRITE
        if spec_class in (CLASS_UI, "ui"):
            return CLASS_UI_FRAME if ("FRAME" in aid or "WINDOW" in aid or "PANEL" in aid) else CLASS_UI_ICON

    size = spec.get("size", [0, 0])

    # 1. UI Elements (HIGHEST PRIORITY: prevent ICON-* with people:True from becoming world_character)
    if aid.startswith("ICON-") or aid.startswith("UI-ICON") or "-ICON-" in aid or aid.endswith("-ICON"):
        return CLASS_UI_ICON

    if (
        aid.startswith("UI-FRAME")
        or aid.startswith("UI-PANEL")
        or aid.startswith("UI-WINDOW")
        or aid.startswith("FRAME-")
        or "UI-BOX" in aid
        or "CARD-PANEL" in aid
    ):
        return CLASS_UI_FRAME

    if aid.startswith("UI-") or aid.startswith("BADGE-") or "BADGE" in aid:
        if "FRAME" in aid or "WINDOW" in aid or "PANEL" in aid or size[0] >= 48:
            return CLASS_UI_FRAME
        return CLASS_UI_ICON

    # 2. Large Scenes / Background Panoramas
    if (
        aid.startswith("HUB-L0")
        or aid.startswith("HUB-L1")
        or aid.startswith("HUB-BACKGROUND")
        or "SCENE" in aid
        or "BG" in aid
        or "BACKGROUND" in aid
        or "PANORAMA" in aid
        or size == [480, 270]
    ):
        return CLASS_LARGE_SCENE

    # 3. Portraits (Dialogue and Profile)
    if "DIALOGUE" in aid or (("PORTRAIT" in aid or "VERA" in aid) and size[1] >= 200):
        return CLASS_DIALOGUE_PORTRAIT

    if (
        "PROFILE" in aid
        or ("PORTRAIT" in aid and "SHEET" not in aid and "WALK" not in aid)
        or aid == "MENTOR-PORTRAIT"
        or "MENTOR-PORTRAIT" in aid
    ):
        return CLASS_PROFILE_PORTRAIT

    # 4. Companions and Fauna
    if "COMPANION" in aid or "CAT" in aid or "DOG" in aid or "BIRD" in aid or "GULL" in aid or "SQUIRREL" in aid:
        return CLASS_COMPANION_SPRITE

    # 5. World Characters and NPCs
    if (
        spec.get("people")
        or "AVATAR" in aid
        or "CHAR" in aid
        or "NPC" in aid
        or "MENTOR-M1" in aid
        or "MENTOR-M2" in aid
    ):
        if "SHEET" in aid or "WALK" in aid or (size[0] > 0 and size[0] <= 64):
            return CLASS_WORLD_CHARACTER
        if size[1] >= 200:
            return CLASS_DIALOGUE_PORTRAIT
        return CLASS_PROFILE_PORTRAIT

    # 6. Architecture
    if aid.startswith("BLD-") or "BUILDING" in aid or "RADAR" in aid or "FACADE" in aid or "STATION" in aid:
        if max(size) >= 80 or not size or size == [0, 0]:
            return CLASS_ARCHITECTURE_LANDMARK
        return CLASS_ARCHITECTURE_SMALL

    # 7. Interactive Objects
    if "INTERACTIVE" in aid or "CONSOLE" in aid or "ANEMOMETER" in aid or "DESK" in aid or "TERMINAL" in aid:
        return CLASS_INTERACTIVE_OBJECT

    # 8. Vegetation
    if "VEGETATION" in aid or "TREE" in aid or "BUSH" in aid or "PINE" in aid or "BIRCH" in aid:
        return CLASS_VEGETATION_SPRITE

    # 9. Terrain
    if aid.startswith("KF-") or aid.startswith("HUB-GROUND") or "TILE" in aid or "TERRAIN" in aid:
        return CLASS_TERRAIN_TILE

    # 10. VFX
    if "VFX" in aid or "EFFECT" in aid or "SMOKE" in aid or "RAIN" in aid or "SNOW" in aid:
        return CLASS_VFX_SMALL

    # 11. Props & Items
    if aid.startswith("PROP-") or "PROP" in aid or "FURNITURE" in aid or "ITEM" in aid:
        return CLASS_PROP_SPRITE

    return CLASS_PROP_SPRITE


def get_profile_for_asset(asset_id: str, asset_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Returns the full generation profile for an asset ID and spec."""
    aclass = classify_asset(asset_id, asset_spec)
    return get_generation_profile(aclass)
