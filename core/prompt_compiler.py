from __future__ import annotations

from typing import Any
from core.asset_classes import (
    CLASS_ARCHITECTURE,
    CLASS_CHARACTER_SPRITE,
    CLASS_PORTRAIT,
    CLASS_PROP,
    CLASS_TERRAIN,
    CLASS_UI,
    CLASS_VEGETATION,
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
    classify_asset,
    get_generation_profile,
)
from core.lore_registry import WORLD_LORE_CONTEXT, get_lore_for_asset

STYLE_LOCK_BASE = (
    "STYLE LOCK: modern high-quality master pixel art (reference feel: Eastward, Sea of Stars, A Short Hike). "
    "Crisp hard-edged square pixels on a strict logical pixel grid, deliberate clusters of 2+ pixels, "
    "no anti-aliasing blur, no photographic texture, no 3D-render look. Ordered 2x2 dithering where needed. "
    "Light from upper-left. Outlines are a darker shade of the local colour, never pure black."
)

NEG_BASE = (
    "blurry, anti-aliased, smooth airbrushed gradient, painterly oil painting, 3d render, photo, "
    "noise, jpeg artifacts, text, letters, numbers, logo, watermark, signature, "
    "chromatic aberration, bloom, lens flare, distortion"
)

NEG_PORTRAIT_SPECIFIC = (
    "chibi, kawaii, super-deformed, baby face, toddler, young girl, anime eyes, oversized head, "
    "toy-like proportions, sticker style, emoji face, cartoonish caricature, childish, cute mascot, "
    "deformed hands, extra fingers, plastic skin, airbrush, wide background landscape"
)

NEG_ARCHITECTURE_SPECIFIC = (
    "generic residential cottage, country house, suburban villa, fantasy observatory, church dome, "
    "sci-fi antenna, palace, alpine mountains, palm trees, wide landscape, forest clutter, "
    "messy trees blocking structure, multiple scattered buildings"
)

NEG_PROP_SPECIFIC = (
    "multiple unrelated objects, person holding object, busy background, wide room interior, "
    "low contrast, messy clutter"
)


def compile_generation_prompts(
    asset_id: str,
    asset_spec: dict[str, Any],
    style_trigger: str = "<26063_pixel_style>"
) -> tuple[str, str, dict[str, Any]]:
    """Compiles class-specific positive and negative prompts adhering to 26063 visual contracts."""
    asset_class = classify_asset(asset_id, asset_spec)
    lore_info = get_lore_for_asset(asset_id)
    profile = get_generation_profile(asset_class)

    pos_parts = []
    if style_trigger:
        pos_parts.append(style_trigger)
    pos_parts.append(STYLE_LOCK_BASE)
    pos_parts.append(WORLD_LORE_CONTEXT)

    neg_parts = [NEG_BASE]

    target_w, target_h = asset_spec.get("size", [256, 256])

    metadata: dict[str, Any] = {
        "asset_class": asset_class,
        "lore_key": lore_info.get("key"),
        "target_size": [target_w, target_h],
        "workflow": profile.get("workflow", "default"),
        "art_priorities": profile.get("artPriorities", []),
    }

    if asset_class in (CLASS_PORTRAIT, CLASS_DIALOGUE_PORTRAIT, CLASS_PROFILE_PORTRAIT):
        neg_parts.append(NEG_PORTRAIT_SPECIFIC)
        char_data = lore_info.get("data", {})
        appearance = char_data.get("appearance", "Adult professional meteorological staff member, intelligent calm face.")

        pos_parts.append(f"Subject: {appearance}")
        if asset_class == CLASS_DIALOGUE_PORTRAIT or target_h >= 200:
            pos_parts.append(
                f"Framing contract: Expressive narrative dialogue portrait bust (head, neck and upper shoulders). "
                f"Head height ratio 28% to 42% of canvas height, face width at least 88 logical pixels. "
                f"Adult realistic facial anatomy, calm demanding expression, focused gaze, age-appropriate features. "
                f"Subtle Baltic lighting, neutral dark slate studio background (#1E2430), "
                f"clean silhouette isolation. Absolutely no chibi, no baby-face, no toy-like proportions."
            )
            metadata["framing"] = "dialogue_portrait_bust"
        else:
            pos_parts.append(
                f"Framing contract: {target_w}x{target_h} pixel profile portrait bust (head, neck and upper shoulders). "
                f"Head height occupies 40% to 55% of canvas height. Eyes level at 35% to 50% from top. "
                f"Adult realistic facial anatomy, calm demanding expression, focused gaze. "
                f"Isolated character against a neutral uniform dark slate studio background (#1E2430), "
                f"clean contrast for silhouette isolation. Absolutely no chibi or baby-face aesthetics."
            )
            metadata["framing"] = "profile_portrait_bust"

    elif asset_class in (CLASS_ARCHITECTURE, CLASS_ARCHITECTURE_LANDMARK, CLASS_ARCHITECTURE_SMALL):
        neg_parts.append(NEG_ARCHITECTURE_SPECIFIC)
        arch_data = lore_info.get("data", {})
        description = arch_data.get("description", "Single architectural facility building, Baltic meteorological campus.")

        pos_parts.append(f"Subject: {description}")
        pos_parts.append(
            f"Framing contract: Single isolated architectural structure landmark, orthographic frontal/isometric view. "
            f"Canvas resolution {target_w}x{target_h} pixels. Structure occupies 70% to 85% of canvas width. "
            f"Strong crisp technical silhouette, high contrast against a clean neutral dark slate background (#1E2430). "
            f"Baltic scientific campus architecture, brick/concrete/glass, warm amber windows, weather-responsive materials. "
            f"No wide landscape, no distracting foliage blocking the building structure."
        )
        metadata["framing"] = "architecture_structure"

    elif asset_class in (CLASS_CHARACTER_SPRITE, CLASS_WORLD_CHARACTER):
        neg_parts.append(NEG_PORTRAIT_SPECIFIC)
        char_data = lore_info.get("data", {})
        appearance = char_data.get("appearance", "Adult staff member, meteorological campus.")
        pos_parts.append(f"Subject: {appearance}")
        pos_parts.append(
            f"Framing contract: Full-body gameplay sprite, logical size {target_w}x{target_h} pixels. "
            f"Clear silhouette, adult proportions, grounded feet, centered on neutral dark matte (#1E2430). "
            f"Clean pixel clusters, recognizable clothing and equipment."
        )
        metadata["framing"] = "world_sprite_character"

    elif asset_class == CLASS_COMPANION_SPRITE:
        pos_parts.append(
            f"Subject: Small animal companion, Baltic campus fauna (cat, gull, or dog). "
            f"Logical size {target_w}x{target_h} pixels. Clean pixel clusters, fur/feather texture ramps, "
            f"lively posture, centered on neutral dark matte (#1E2430)."
        )
        metadata["framing"] = "companion_sprite"

    elif asset_class == CLASS_LARGE_SCENE:
        pos_parts.append(
            f"Framing contract: Expansive {target_w}x{target_h} panoramic scene of the Gulf of Finland meteorological campus. "
            f"Atmospheric depth, low Baltic horizon, water surface, weather layers, warm amber indoor lights contrasting cold exterior. "
            f"Human traces, scientific activity, North-West Russian maritime climate."
        )
        metadata["framing"] = "large_scene_480x270"

    elif asset_class in (CLASS_PROP, CLASS_PROP_SPRITE, CLASS_INTERACTIVE_OBJECT):
        neg_parts.append(NEG_PROP_SPECIFIC)
        pos_parts.append(
            f"Subject: Centered isolated meteorological scientific equipment instrument. "
            f"Canvas resolution {target_w}x{target_h} pixels. Clear mechanical silhouette, "
            f"centered on neutral contrasting dark slate matte (#1E2430)."
        )
        metadata["framing"] = "prop_centered"

    else:
        # Terrain / Vegetation / UI / VFX
        if asset_spec.get("prompt"):
            pos_parts.append(f"Subject: {asset_spec['prompt']}")
        pos_parts.append(
            f"Orthographic projection, canvas {target_w}x{target_h} pixels, crisp pixel art on neutral background."
        )

    pos_prompt = " ".join(pos_parts)
    neg_prompt = ", ".join(neg_parts)

    return pos_prompt, neg_prompt, metadata
