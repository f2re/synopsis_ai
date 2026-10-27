from __future__ import annotations

from typing import Any

# This module intentionally keeps only compact generation-ready summaries.
# The canonical source of truth is f2re/synopsis_game/docs/lore/lore-registry.yaml
# together with docs/lore/03_CHARACTERS.md and docs/lore/04_LOCATIONS.md.
WORLD_LORE_CONTEXT = (
    "Setting: 26063: Позывной «Синоптик». A fictional educational and research meteorological campus on the shore "
    "of the Gulf of Finland, North-West Russia. Core theme: continuity of observation across generations. "
    "Low north-western horizon, Baltic water, granite terraces, pine, birch, rowan, rosehip, lilac and mossy lawn. "
    "Compact low-rise scientific buildings from different decades, functional meteorological instruments, paper charts, "
    "analogue equipment and modern screens coexist naturally. Warm amber human light contrasts with cool Baltic weather. "
    "Scenes should feel lived-in and normally contain at least three subtle traces of recent human work. "
    "Strictly no alpine mountains, palm trees, palace-scale architecture, giant ceremonial domes, military-bunker mood, "
    "post-apocalypse, futuristic sci-fi or fantasy. The campus is fictional and must not be presented as a literal copy "
    "of a real institution. Background scenes should not bake the full game logo into buildings."
)

CHARACTER_LORE: dict[str, dict[str, Any]] = {
    "vera_ladozhskaya": {
        "full_name": "Вера Павловна Ладожская",
        "role": "Ведущий синоптик кампуса 26063, руководитель учебной практики",
        "age": 58,
        "appearance": (
            "58-year-old senior female synoptic meteorologist. Realistic adult proportions, intelligent calm face, "
            "quiet professional confidence and a subtle demanding squint. Neat greying hair gathered into a restrained bun, "
            "thin metal-frame glasses. Dark navy or lingonberry knitted cardigan over a light blouse. Carries a classic "
            "two-colour red-and-blue synoptic pencil or a folded working chart. No uniformed or militarised styling."
        ),
        "forbidden_traits": [
            "chibi", "kawaii", "baby face", "toddler", "oversized head", "anime girl", "cute mascot",
            "sticker character", "emoji face", "childish", "military uniform", "deformed hands", "extra fingers"
        ],
        "required_traits": [
            "adult realistic proportions", "age readable ~55-65", "thin metal-frame glasses",
            "greying hair in bun", "red-blue synoptic pencil or folded chart", "navy or lingonberry cardigan"
        ]
    },
    "semyon_kruglov": {
        "full_name": "Семён Петрович Круглов",
        "role": "Старший техник-метеоролог и мастер по метеоприборам",
        "age": 51,
        "appearance": (
            "51-year-old senior meteorological technician. Realistic adult proportions and a practical weather-beaten "
            "working appearance. Dark olive or mustard work windbreaker with many pockets, warm high-neck sweater, knitted "
            "watch cap, canvas instrument bag over the shoulder. He belongs to a civilian scientific campus, not a military unit."
        ),
        "forbidden_traits": [
            "chibi", "kawaii", "baby face", "cartoon mascot", "military hero", "combat gear", "anime"
        ],
        "required_traits": [
            "adult male age ~50", "practical instrument technician", "olive or mustard work jacket",
            "warm high-neck sweater", "knitted cap", "canvas instrument bag"
        ]
    },
    "pascal_cat": {
        "full_name": "Кот Гектопаскаль (Паскаль)",
        "role": "Живой обитатель кампуса 26063",
        "appearance": (
            "A fluffy ginger-and-white station cat with thick fur and a green eye, usually resting on a warm archive windowsill, "
            "near the operational-room radiator, or beside Semyon's workshop and old teletype. Calm, slightly lazy, lived-in campus companion."
        ),
        "forbidden_traits": ["fantasy familiar", "magical aura", "anime mascot", "human clothing", "logo costume"],
        "required_traits": ["ginger and white fur", "thick coat", "green eye", "natural cat anatomy", "relaxed station-cat posture"]
    }
}

ARCHITECTURE_LORE: dict[str, dict[str, Any]] = {
    "radar": {
        "name": "Радиолокационный комплекс",
        "description": (
            "Doppler meteorological radar facility: a clear technical structure with a white spherical radome, maintenance "
            "elements and a compact equipment/service base. Baltic scientific-campus materials, practical proportions, "
            "warm occupied windows where appropriate. It must read as working meteorological infrastructure, not a monument."
        ),
        "required_features": ["white radar dome", "technical radar structure", "service equipment base", "clear silhouette"],
        "forbidden_features": ["generic residential cottage", "fantasy observatory", "church dome", "sci-fi antenna", "palace"]
    },
    "synoptic": {
        "name": "Оперативный синоптический зал",
        "description": (
            "Modest two-story operational building with a 1950s scientific-campus character and later practical upgrades: "
            "ochre facade, granite base, large readable windows with warm amber light, simple roof/lightwell and weather instruments. "
            "Inside: broad synoptic tables, paper charts, a green desk lamp, analogue instruments, UTC clock and modern monitors."
        ),
        "required_features": ["modest low-rise scale", "ochre facade", "warm amber windows", "granite base", "weather instruments"],
        "forbidden_features": ["palace", "monumental columnade", "modern skyscraper", "log cabin", "military headquarters"]
    },
    "archive": {
        "name": "Архив метеоданных и отчётов",
        "description": (
            "Warm one-and-a-half-story archive building with stone/granite base, ochre plaster, wooden window frames and a lived-in "
            "working interior: tall wooden shelves, boxes marked by the 26063 series, paper folders, a warm desk lamp and an old barograph."
        ),
        "required_features": ["granite or stone base", "ochre plaster", "wooden frames", "archive shelving and paper traces"],
        "forbidden_features": ["generic cottage", "futuristic building", "palace", "museum display hall"]
    }
}


def get_lore_for_asset(asset_id: str) -> dict[str, Any]:
    aid = asset_id.lower()

    for key, data in CHARACTER_LORE.items():
        aliases = {
            "vera_ladozhskaya": ("mentor", "ladozh", "vera", "teacher"),
            "semyon_kruglov": ("semyon", "semen", "kruglov", "technician"),
            "pascal_cat": ("pascal", "cat", "companion.cat", "hectopascal"),
        }.get(key, ())
        if key in aid or any(alias in aid for alias in aliases):
            return {"type": "character", "key": key, "data": data}

    for key, data in ARCHITECTURE_LORE.items():
        if key in aid:
            return {"type": "architecture", "key": key, "data": data}

    return {"type": "general", "data": {"context": WORLD_LORE_CONTEXT}}
