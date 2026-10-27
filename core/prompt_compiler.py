from __future__ import annotations

from typing import Any
from core.asset_classes import (
    CLASS_ARCHITECTURE,
    CLASS_ARCHITECTURE_LANDMARK,
    CLASS_ARCHITECTURE_SMALL,
    CLASS_CHARACTER_SPRITE,
    CLASS_COMPANION_SPRITE,
    CLASS_DIALOGUE_PORTRAIT,
    CLASS_INTERACTIVE_OBJECT,
    CLASS_LARGE_SCENE,
    CLASS_PORTRAIT,
    CLASS_PROFILE_PORTRAIT,
    CLASS_PROP,
    CLASS_PROP_SPRITE,
    CLASS_TERRAIN,
    CLASS_TERRAIN_TILE,
    CLASS_UI,
    CLASS_UI_FRAME,
    CLASS_UI_ICON,
    CLASS_VEGETATION,
    CLASS_VEGETATION_SPRITE,
    CLASS_VFX_SMALL,
    CLASS_WORLD_CHARACTER,
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

STYLE_LOCK_UI = (
    "STYLE LOCK: modern retro-game pixel art UI icon glyph (reference feel: Eastward, Celeste, Owlboy inventory/menu UI). "
    "Clean high-contrast isolated icon glyph centered on uniform solid dark slate matte background (#1E2430), "
    "crisp readable silhouette, bold readable contours, deliberate pixel clusters, no anti-aliasing blur, "
    "no photographic texture, no 3D-render look. Absolutely no background scenery, no buildings, no outdoor landscape."
)

NEG_BASE = (
    "blurry, anti-aliased, smooth airbrushed gradient, painterly oil painting, 3d render, photo, "
    "noise, jpeg artifacts, text, letters, numbers, logo, watermark, signature, "
    "chromatic aberration, bloom, lens flare, distortion"
)

NEG_UI_SPECIFIC = (
    "building, facade, house, campus, cottage, room, outdoor scenery, landscape, trees, mountains, "
    "sky, horizon, wide shot, person, human face, 3d render, photo, blurry, anti-aliased, messy clutter, noise"
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

UI_ICON_SUBJECTS: dict[str, str] = {
    "JOURNAL": "An open meteorological observation field notebook logbook journal with lined paper pages, leather binding cover, and ribbon bookmark.",
    "BOOK": "A closed hardcover weather observer reference handbook book with bookmark ribbon.",
    "DISCOVERIES": "A brass magnifying glass exploring a sparkling star discovery symbol.",
    "PERSON": "A clean minimalist user profile head and shoulders silhouette icon.",
    "GEAR": "A mechanical brass cog gear settings wheel symbol.",
    "SOUND-ON": "An audio speaker megaphone broadcasting sound waves.",
    "SOUND-OFF": "An audio speaker with a mute cancellation slash cross.",
    "MOON": "A luminous crescent nocturnal moon with soft rim glow.",
    "SUN": "A radiant stylized daytime golden sun icon glyph.",
    "DAWN": "A sun rising above flat ocean water horizon line, dawn twilight.",
    "LOCK": "A solid brass padlock closed shackle security lock.",
    "UNLOCK": "An open brass padlock unlocked security symbol.",
    "PAUSE": "Two vertical pause bars button symbol.",
    "CHECK": "A crisp green verification checkmark tick symbol.",
    "CROSS": "A crisp red rejection cross X symbol.",
    "STAR": "A five-pointed golden award star achievement badge symbol.",
    "CLOCK": "An analog stopwatch clock face dial with minute and hour hands.",
    "TAPE": "A magnetic cassette audio tape spool reel.",
    "STATION": "A minimalist meteorological weather mast tower station symbol.",
    "THERMOMETER": "A vertical glass mercury thermometer tube with Celsius temperature scale markings.",
    "BAROMETER": "An aneroid barometer dial instrument with brass bezel and pressure pointer arrow.",
    "VANE": "A meteorological wind vane anemometer weathercock with cardinal direction arrow.",
    "CLOUD": "A stylized weather cumulus cloud icon symbol.",
    "MODE-MENTOR": "An academic graduation mentor cap and star crest icon badge symbol.",
    "MODE-SOLO": "A solitary observer compass silhouette badge icon symbol.",
    "MODE-BLITZ": "A fast lightning bolt blitz speed timer challenge badge icon symbol.",
    "ARROW-LEFT": "A crisp pixel navigation arrow pointing left.",
    "ARROW-RIGHT": "A crisp pixel navigation arrow pointing right.",
}


def _infer_ui_subject(asset_id: str, prompt_override: str | None = None) -> str:
    if prompt_override:
        return prompt_override
    aid = asset_id.upper()
    for key, desc in UI_ICON_SUBJECTS.items():
        if key in aid:
            return desc
    slug = aid.replace("ICON-", "").replace("UI-", "").replace("-", " ").strip().lower()
    return f"A clean graphic pixel UI icon representing {slug}."


def compile_generation_prompts(
    asset_id: str,
    asset_spec: dict[str, Any],
    style_trigger: str = "<26063_pixel_style>",
    correction: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Compiles class-specific positive and negative prompts adhering to 26063 visual contracts."""
    asset_class = classify_asset(asset_id, asset_spec)
    lore_info = get_lore_for_asset(asset_id)
    profile = get_generation_profile(asset_class)

    target_w, target_h = asset_spec.get("size", [256, 256])

    metadata: dict[str, Any] = {
        "asset_class": asset_class,
        "lore_key": lore_info.get("key"),
        "target_size": [target_w, target_h],
        "workflow": profile.get("workflow", "default"),
        "art_priorities": profile.get("artPriorities", []),
    }
    if correction:
        metadata["correction"] = correction

    pos_parts: list[str] = []
    if style_trigger:
        pos_parts.append(style_trigger)

    neg_parts: list[str] = [NEG_BASE]

    # 1. UI Icons & Frames: NEVER include campus building lore context
    if asset_class in (CLASS_UI, CLASS_UI_ICON, CLASS_UI_FRAME):
        pos_parts.append(STYLE_LOCK_UI)
        neg_parts.append(NEG_UI_SPECIFIC)

        raw_prompt = asset_spec.get("prompt")
        subject = _infer_ui_subject(asset_id, raw_prompt)
        pos_parts.append(f"Subject: {subject}")
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")

        if asset_class == CLASS_UI_FRAME:
            pos_parts.append(
                f"Framing contract: Rectangular UI frame container border, resolution {target_w}x{target_h} pixels, "
                f"ornate Baltic meteorological brass/steel border trim, centered dark translucent center."
            )
            metadata["framing"] = "ui_frame"
        else:
            pos_parts.append(
                f"Framing contract: Single centered isolated icon glyph, canvas resolution {target_w}x{target_h} pixels. "
                f"Bold graphic silhouette, high contrast against solid dark matte (#1E2430), absolutely no buildings or landscape."
            )
            metadata["framing"] = "ui_icon_centered"

    # 2. Portraits (Dialogue and Profile)
    elif asset_class in (CLASS_PORTRAIT, CLASS_DIALOGUE_PORTRAIT, CLASS_PROFILE_PORTRAIT):
        pos_parts.append(STYLE_LOCK_BASE)
        pos_parts.append(WORLD_LORE_CONTEXT)
        neg_parts.append(NEG_PORTRAIT_SPECIFIC)

        char_data = lore_info.get("data", {})
        appearance = char_data.get("appearance", "Adult professional meteorological staff member, intelligent calm face.")
        pos_parts.append(f"Subject: {appearance}")
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")

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

    # 3. Architecture
    elif asset_class in (CLASS_ARCHITECTURE, CLASS_ARCHITECTURE_LANDMARK, CLASS_ARCHITECTURE_SMALL):
        pos_parts.append(STYLE_LOCK_BASE)
        pos_parts.append(WORLD_LORE_CONTEXT)
        neg_parts.append(NEG_ARCHITECTURE_SPECIFIC)

        arch_data = lore_info.get("data", {})
        description = arch_data.get("description", "Single architectural facility building, Baltic meteorological campus.")
        pos_parts.append(f"Subject: {description}")
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")

        pos_parts.append(
            f"Framing contract: Single isolated architectural structure landmark, orthographic frontal/isometric view. "
            f"Canvas resolution {target_w}x{target_h} pixels. Structure occupies 70% to 85% of canvas width. "
            f"Strong crisp technical silhouette, high contrast against a clean neutral dark slate background (#1E2430). "
            f"Baltic scientific campus architecture, brick/concrete/glass, warm amber windows, weather-responsive materials. "
            f"No wide landscape, no distracting foliage blocking the building structure."
        )
        metadata["framing"] = "architecture_structure"

    # 4. Large Scenes / Backgrounds
    elif asset_class == CLASS_LARGE_SCENE:
        pos_parts.append(STYLE_LOCK_BASE)
        pos_parts.append(WORLD_LORE_CONTEXT)

        aid_u = asset_id.upper()
        if "SKY" in aid_u:
            scene_subject = "Atmospheric Baltic daytime sky layer over the Gulf of Finland, subtle layered stratocumulus clouds, pale cool blue northern sky gradient, gentle maritime lighting."
        elif "WATER" in aid_u or "SEA" in aid_u:
            scene_subject = "Baltic sea water surface layer, subtle cold waves, ripples, maritime reflections."
        elif "GROUND" in aid_u:
            scene_subject = "Granite embankment ground layer, paved stone pathway, coastal North-West Russian vegetation."
        else:
            scene_subject = f"Expansive {target_w}x{target_h} panoramic scene of the Gulf of Finland meteorological campus."

        pos_parts.append(f"Subject: {scene_subject}")
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")

        pos_parts.append(
            f"Framing contract: Expansive {target_w}x{target_h} panoramic scene layer of the Gulf of Finland meteorological campus. "
            f"Atmospheric depth, low Baltic horizon, weather layers, warm amber indoor lights contrasting cold exterior. "
            f"Human traces, scientific activity, North-West Russian maritime climate."
        )
        metadata["framing"] = "large_scene_480x270"

    # 5. World Characters and Companions
    elif asset_class in (CLASS_CHARACTER_SPRITE, CLASS_WORLD_CHARACTER):
        pos_parts.append(STYLE_LOCK_BASE)
        pos_parts.append(WORLD_LORE_CONTEXT)
        neg_parts.append(NEG_PORTRAIT_SPECIFIC)

        char_data = lore_info.get("data", {})
        appearance = char_data.get("appearance", "Adult staff member, meteorological campus.")
        pos_parts.append(f"Subject: {appearance}")
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")

        pos_parts.append(
            f"Framing contract: Full-body gameplay sprite, logical size {target_w}x{target_h} pixels. "
            f"Clear silhouette, adult proportions, grounded feet, centered on neutral dark matte (#1E2430). "
            f"Clean pixel clusters, recognizable clothing and equipment."
        )
        metadata["framing"] = "world_sprite_character"

    elif asset_class == CLASS_COMPANION_SPRITE:
        pos_parts.append(STYLE_LOCK_BASE)
        pos_parts.append(WORLD_LORE_CONTEXT)
        pos_parts.append(
            f"Subject: Small animal companion, Baltic campus fauna (cat, gull, squirrel, or tit bird). "
            f"Logical size {target_w}x{target_h} pixels. Clean pixel clusters, fur/feather texture ramps, "
            f"lively posture, centered on neutral dark matte (#1E2430)."
        )
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")
        metadata["framing"] = "companion_sprite"

    # 6. Props & Interactive Equipment
    elif asset_class in (CLASS_PROP, CLASS_PROP_SPRITE, CLASS_INTERACTIVE_OBJECT):
        pos_parts.append(STYLE_LOCK_BASE)
        pos_parts.append(WORLD_LORE_CONTEXT)
        neg_parts.append(NEG_PROP_SPECIFIC)
        pos_parts.append(
            f"Subject: Centered isolated meteorological scientific equipment instrument. "
            f"Canvas resolution {target_w}x{target_h} pixels. Clear mechanical silhouette, "
            f"centered on neutral contrasting dark slate matte (#1E2430)."
        )
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")
        metadata["framing"] = "prop_centered"

    else:
        # Terrain / Vegetation / VFX
        pos_parts.append(STYLE_LOCK_BASE)
        pos_parts.append(WORLD_LORE_CONTEXT)
        if asset_spec.get("prompt"):
            pos_parts.append(f"Subject: {asset_spec['prompt']}")
        if correction:
            pos_parts.append(f"Correction emphasis: {correction}")
        pos_parts.append(
            f"Orthographic projection, canvas {target_w}x{target_h} pixels, crisp pixel art on neutral background."
        )

    pos_prompt = " ".join(pos_parts)
    neg_prompt = ", ".join(neg_parts)

    return pos_prompt, neg_prompt, metadata
