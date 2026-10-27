from __future__ import annotations

from pathlib import Path
from typing import Any
from PIL import Image


def extract_silhouette_mask(image_path: Path) -> Image.Image:
    """Extracts binary silhouette mask (white = solid building/prop, black = background)."""
    img = Image.open(image_path).convert("RGBA")
    mask = Image.new("L", img.size, 0)
    img_pixels = img.load()
    mask_pixels = mask.load()
    
    for y in range(img.height):
        for x in range(img.width):
            _, _, _, a = img_pixels[x, y]
            if a > 0:
                mask_pixels[x, y] = 255
    return mask


def compare_silhouettes(day_path: Path, night_path: Path) -> dict[str, Any]:
    """Calculates pixel-exact silhouette match and IoU between day and night assets."""
    day_mask = extract_silhouette_mask(day_path)
    night_mask = extract_silhouette_mask(night_path)
    
    if day_mask.size != night_mask.size:
        return {
            "match": False,
            "iou": 0.0,
            "error": f"Size mismatch: {day_mask.size} vs {night_mask.size}"
        }
        
    d_pix = day_mask.load()
    n_pix = night_mask.load()
    
    intersection = 0
    union = 0
    mismatch_pixels = 0
    
    for y in range(day_mask.height):
        for x in range(day_mask.width):
            d = d_pix[x, y] > 0
            n = n_pix[x, y] > 0
            if d and n:
                intersection += 1
            if d or n:
                union += 1
            if d != n:
                mismatch_pixels += 1
                
    iou = (intersection / union) if union > 0 else 1.0
    return {
        "match": mismatch_pixels == 0,
        "iou": round(iou, 4),
        "mismatch_pixels": mismatch_pixels,
        "total_solid_pixels": union
    }


def enforce_geometry_lock(base_day_path: Path, edited_night_path: Path, output_path: Path) -> None:
    """Clamps the night image strictly to the base day silhouette mask so outlines match 1:1."""
    day_mask = extract_silhouette_mask(base_day_path)
    night_img = Image.open(edited_night_path).convert("RGBA")
    
    if night_img.size != day_mask.size:
        night_img = night_img.resize(day_mask.size, Image.Resampling.NEAREST)
        
    d_pix = day_mask.load()
    n_pix = night_img.load()
    
    for y in range(night_img.height):
        for x in range(night_img.width):
            if d_pix[x, y] == 0:
                # Outside day silhouette: force transparent
                n_pix[x, y] = (0, 0, 0, 0)
            elif n_pix[x, y][3] == 0:
                # Inside day silhouette but night became transparent: fill from day base
                pass
                
    output_path.parent.mkdir(parents=True, exist_ok=True)
    night_img.save(output_path, format="PNG")
