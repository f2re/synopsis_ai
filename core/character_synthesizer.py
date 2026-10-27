from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CHARACTER_PATH = REPO_ROOT / "assets/source/characters/base/candidates/avatar_base_sheet_v2.png"

# Authentic character archetypes conforming to Russian synoptic meteorology visual lore and people_26063_v1.gpl
ARCHETYPES = [
    # 0: Senior Professor / Chief Synoptician (burgundy cardigan, gray hair, glasses, fair northern skin)
    {
        "name": "chief_synoptician_professor",
        "map": {
            (201, 139, 99): (240, 200, 180), (169, 111, 83): (210, 160, 140), # skin
            (94, 152, 189): (180, 50, 60), (53, 110, 154): (130, 30, 45), (20, 56, 66): (80, 15, 30), # coat
            (85, 59, 54): (210, 215, 220), (77, 57, 50): (160, 165, 170), # hair
            (38, 51, 58): (30, 40, 60) # pants
        }
    },
    # 1: Senior Marine Observer (weathered tan, mustard yellow storm parka, dark blue watch cap beanie)
    {
        "name": "marine_weather_observer",
        "map": {
            (201, 139, 99): (215, 155, 115), (169, 111, 83): (175, 120, 85),
            (94, 152, 189): (242, 159, 69), (53, 110, 154): (204, 116, 32), (20, 56, 66): (140, 70, 18),
            (85, 59, 54): (35, 50, 80), (77, 57, 50): (20, 30, 55),
            (38, 51, 58): (40, 45, 55)
        }
    },
    # 2: Field Forecaster (olive green storm coat, warm brown hair)
    {
        "name": "field_forecaster",
        "map": {
            (201, 139, 99): (225, 175, 140), (169, 111, 83): (185, 135, 105),
            (94, 152, 189): (110, 130, 70), (53, 110, 154): (75, 95, 45), (20, 56, 66): (45, 60, 25),
            (85, 59, 54): (110, 75, 50), (77, 57, 50): (75, 50, 35),
            (38, 51, 58): (50, 55, 45)
        }
    },
    # 3: Station Technician / Aerologist (station navy blue windbreaker, dark brown hair)
    {
        "name": "station_technician",
        "map": {
            (201, 139, 99): (220, 165, 125), (169, 111, 83): (180, 125, 90),
            (94, 152, 189): (94, 152, 189), (53, 110, 154): (53, 110, 154), (20, 56, 66): (20, 56, 66),
            (85, 59, 54): (60, 45, 35), (77, 57, 50): (40, 30, 25),
            (38, 51, 58): (35, 40, 45)
        }
    },
    # 4: Junior Synoptician Trainee (teal knit sweater, sandy hair, blue denim)
    {
        "name": "junior_trainee",
        "map": {
            (201, 139, 99): (235, 190, 160), (169, 111, 83): (195, 145, 115),
            (94, 152, 189): (70, 140, 150), (53, 110, 154): (45, 100, 110), (20, 56, 66): (25, 65, 75),
            (85, 59, 54): (220, 190, 120), (77, 57, 50): (170, 140, 80),
            (38, 51, 58): (40, 55, 85)
        }
    }
]


def load_base_character(repo_root: Path | None = None) -> Image.Image:
    """Loads base character sprite sheet."""
    root = repo_root or REPO_ROOT
    candidates = [
        root / "assets/source/characters/base/candidates/avatar_base_sheet_v2.png",
        Path("/home/user/synopsis-ai/synopsis_game/assets/source/characters/base/candidates/avatar_base_sheet_v2.png")
    ]
    for c in candidates:
        if c.is_file():
            return Image.open(c).convert("RGBA")
    # Fallback synthetic base if file is somehow missing
    img = Image.new("RGBA", (384, 48), (0, 0, 0, 0))
    return img


def nearest_color(rgb: tuple[int, int, int], palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    r, g, b = rgb
    return min(palette, key=lambda p: 2 * (p[0] - r) ** 2 + 4 * (p[1] - g) ** 2 + 3 * (p[2] - b) ** 2)


def generate_character_frame(
    base_frame: Image.Image,
    target_size: tuple[int, int],
    seed: int,
    frame_idx: int,
    row_idx: int = 0,
    palette: list[tuple[int, int, int]] | None = None
) -> Image.Image:
    """Renders a single character animation frame with palette variation and subtle secondary motion."""
    tw, th = target_size
    arch = ARCHETYPES[seed % len(ARCHETYPES)]["map"]

    # 1. Palette remap
    recolored = Image.new("RGBA", base_frame.size, (0, 0, 0, 0))
    src = base_frame.load()
    dst = recolored.load()
    for y in range(base_frame.height):
        for x in range(base_frame.width):
            r, g, b, a = src[x, y]
            if a > 0:
                key = (r, g, b)
                new_rgb = arch.get(key, key)
                dst[x, y] = (*new_rgb, a)

    # 2. Add subtle accessories based on seed
    # If professor: add round spectacles and red-blue pencil
    if (seed % len(ARCHETYPES)) == 0 and row_idx in (0, 4, 5):
        if dst[14, 16][3] > 0:
            dst[14, 16] = (255, 255, 255, 255)
            dst[17, 16] = (255, 255, 255, 255)
        if dst[13, 27][3] > 0:
            dst[13, 27] = (184, 66, 58, 255) # Red pencil
            dst[14, 27] = (53, 110, 154, 255) # Blue pencil

    # 3. Resize if target frame size differs (e.g. M1 12x24 vs M2 32x48)
    if recolored.size != target_size:
        scaled = recolored.resize(target_size, Image.Resampling.NEAREST)
        pix = scaled.load()
        for y in range(th):
            for x in range(tw):
                r, g, b, a = pix[x, y]
                alpha = 255 if a >= 128 else 0
                c = (r, g, b)
                if alpha > 0 and palette:
                    c = nearest_color(c, palette)
                pix[x, y] = (*c, alpha)
        return scaled

    # Snap colors to active palette if provided
    if palette:
        pix = recolored.load()
        for y in range(recolored.height):
            for x in range(recolored.width):
                if pix[x, y][3] > 0:
                    c = nearest_color(pix[x, y][:3], palette)
                    pix[x, y] = (*c, 255)

    return recolored


def generate_character_sprite_frames(
    asset: dict[str, Any],
    seed: int = 26063,
    frames: int | None = None,
    palette: list[tuple[int, int, int]] | None = None,
    repo_root: Path | None = None
) -> tuple[list[Image.Image], tuple[int, int], int]:
    """Generates all animation frames required to assemble a compliant character sprite sheet."""
    base_img = load_base_character(repo_root)
    
    fw, fh = asset.get("frame", (32, 48))
    sheet_w, sheet_h = asset.get("size", (192, 288))
    
    cols = sheet_w // fw
    rows = sheet_h // fh
    total_needed = cols * rows

    base_row_starts = [0, 3, 6, 9, 0, 0]
    generated_frames: list[Image.Image] = []
    
    for r in range(rows):
        start_idx = base_row_starts[r % len(base_row_starts)]
        for c in range(cols):
            sub_step = c % 3 if c < 3 else (2 - (c % 3))
            base_col = start_idx + sub_step
            
            base_frame = base_img.crop((base_col * 32, 0, (base_col + 1) * 32, 48))
            frame_img = generate_character_frame(base_frame, (fw, fh), seed, c, row_idx=r, palette=palette)
            
            if r == 4 and c in (2, 3):
                offset_frame = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
                offset_frame.paste(frame_img, (0, 1), frame_img)
                frame_img = offset_frame
                
            generated_frames.append(frame_img)
            
    return generated_frames, (fw, fh), cols
