from __future__ import annotations

from typing import Any

WORLD_LORE_CONTEXT = (
    "Setting: 26063: Позывной «Синоптик». A fictional cozy scientific meteorological campus on the shore "
    "of the Gulf of Finland near St. Petersburg, North-West Russia (59.9° N). Late 20th century academic "
    "atmosphere: flat horizon, Baltic water, birch and pine, granite embankments, weathered ochre and terracotta "
    "brick facades, warm amber windows, authentic Russian synoptic meteorological instruments. "
    "Strictly no alpine mountains, no palm trees, no futuristic sci-fi or fantasy elements."
)

CHARACTER_LORE: dict[str, dict[str, Any]] = {
    "vera_ladozhskaya": {
        "full_name": "Вера Павловна Ладожская",
        "role": "Ведущий синоптик станции 26063, наставник",
        "age": 58,
        "appearance": (
            "58-year-old senior female synoptic meteorologist. Realistic adult proportions, intelligent focused "
            "face, quiet professional confidence, subtle polite demanding squint. Silver-grey hair with darker blond "
            "streaks styled in a restrained bun. Classic thin metal rectangular reading glasses on the nose bridge. "
            "Wearing an authentic Soviet/Russian hydrometeorological service uniform or warm burgundy/navy knitted "
            "wool cardigan with a subtle brass lapel pin badge. Holding or carrying in breast pocket a classic "
            "two-tone red-and-blue synoptic pencil (сине-красный карандаш синоптика)."
        ),
        "forbidden_traits": [
            "chibi", "kawaii", "baby face", "toddler", "oversized head", "anime girl", "cute mascot",
            "sticker character", "emoji face", "childish", "deformed hands", "extra fingers"
        ],
        "required_traits": [
            "adult realistic proportions", "age readable ~55-65", "thin wireframe glasses",
            "grey hair in bun", "red-blue synoptic pencil", "cozy academic sweater or uniform"
        ]
    },
    "semyon_kruglov": {
        "full_name": "Семён Круглов",
        "role": "Техник-метеоролог метеоплощадки",
        "age": 46,
        "appearance": (
            "46-year-old meteorologist observer technician. Weather-beaten face, observant squint, short brown hair "
            "with grey temples, wool watch cap beanie, mustard canvas work jacket with fleece collar, clipboard with "
            "КМ-1 observation logbook."
        ),
        "forbidden_traits": ["chibi", "kawaii", "baby face", "cartoonish"],
        "required_traits": ["adult male ~45", "weathered observer", "beanie and work jacket", "clipboard"]
    }
}

ARCHITECTURE_LORE: dict[str, dict[str, Any]] = {
    "radar": {
        "name": "Радиолокационная вышка ДМРЛ (BLD-RADAR)",
        "description": (
            "Doppler Meteorological Radar (ДМРЛ) tower complex. A tall technical meteorological structure: a large "
            "spherical white geodesic radome dome housing the radar scanner mounted atop a sturdy technical lattice/steel "
            "observation tower with external maintenance ladders and railings, sitting on a weathered single-story ochre "
            "brick/concrete equipment pavilion annex at the base. Single isolated structure, high contrast clear silhouette, "
            "Baltic scientific campus context."
        ),
        "required_features": ["white radar dome", "technical radar tower", "service equipment annex base", "clear silhouette"],
        "forbidden_features": ["generic residential cottage", "fantasy observatory", "church dome", "sci-fi antenna", "palace"]
    },
    "synoptic": {
        "name": "Главный синоптический корпус (BLD-SYNOPTIC)",
        "description": (
            "Operational synoptic hall building. St. Petersburg classical academic architecture: two-story ochre "
            "facade, rectangular proportioned windows with warm interior amber glow, granite base socle, white cornice, "
            "roof weather observation deck with anemometer mast."
        ),
        "required_features": ["ochre facade", "warm amber windows", "granite socle", "roof anemometer mast"],
        "forbidden_features": ["residential house", "modern skyscraper", "log cabin"]
    },
    "archive": {
        "name": "Архив гидрометеорологических наблюдений (BLD-ARCHIVE)",
        "description": (
            "Hydrometeorological archive building. Sturdy historical brick pavilion with high narrow windows, heavy oak "
            "entrance door, granite steps, brass sign plate on wall."
        ),
        "required_features": ["brick walls", "granite foundation", "high archive windows"],
        "forbidden_features": ["generic cottage", "futuristic building"]
    }
}


def get_lore_for_asset(asset_id: str) -> dict[str, Any]:
    aid = asset_id.lower()
    for key, data in CHARACTER_LORE.items():
        if key in aid or (key == "vera_ladozhskaya" and ("mentor" in aid or "ladozh" in aid or "portrait" in aid)):
            return {"type": "character", "key": key, "data": data}

    for key, data in ARCHITECTURE_LORE.items():
        if key in aid:
            return {"type": "architecture", "key": key, "data": data}

    return {"type": "general", "data": {"context": WORLD_LORE_CONTEXT}}
