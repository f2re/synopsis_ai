from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from PIL import Image


def assemble_sprite_sheet(
    frame_paths: list[Path],
    output_path: Path,
    frame_size: tuple[int, int],
    columns: int | None = None
) -> dict[str, Any]:
    """Combines individual normalized frames into an animation sheet with strict grid alignment."""
    if not frame_paths:
        raise ValueError("No frame paths provided for sprite sheet assembly")

    fw, fh = frame_size
    n_frames = len(frame_paths)
    
    if columns is None or columns <= 0:
        columns = n_frames
    rows = math.ceil(n_frames / columns)
    
    sheet_width = columns * fw
    sheet_height = rows * fh
    
    sheet = Image.new("RGBA", (sheet_width, sheet_height), (0, 0, 0, 0))
    
    for idx, path in enumerate(frame_paths):
        col = idx % columns
        row = idx // columns
        frame_img = Image.open(path).convert("RGBA")
        
        # Verify frame matches target frame size exactly
        if (frame_img.width, frame_img.height) != (fw, fh):
            frame_img = frame_img.resize((fw, fh), Image.Resampling.NEAREST)
            
        sheet.paste(frame_img, (col * fw, row * fh), frame_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")

    return {
        "output_path": str(output_path),
        "sheet_size": [sheet_width, sheet_height],
        "frame_size": [fw, fh],
        "frames_count": n_frames,
        "grid": [columns, rows]
    }
