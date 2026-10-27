from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from PIL import Image

CHROMA = (255, 0, 255)
CHROMA_TOLERANCE = 60


def is_chroma(rgb: tuple[int, int, int]) -> bool:
    return sum(abs(a - b) for a, b in zip(rgb, CHROMA)) <= CHROMA_TOLERANCE


def nearest_palette_color(
    rgb: tuple[int, int, int],
    palette: list[tuple[int, int, int]]
) -> tuple[int, int, int]:
    r, g, b = rgb
    # Weighted Euclidean distance matching human perception (CIE approximation)
    return min(palette, key=lambda p: 2 * (p[0] - r) ** 2 + 4 * (p[1] - g) ** 2 + 3 * (p[2] - b) ** 2)


def block_majority_downsample(image: Image.Image, width: int, height: int) -> Image.Image:
    src = image.load()
    sx, sy = image.width / width, image.height / height
    out = Image.new("RGBA", (width, height))
    dst = out.load()
    
    for y in range(height):
        for x in range(width):
            x0, x1 = int(x * sx), max(int((x + 1) * sx), int(x * sx) + 1)
            y0, y1 = int(y * sy), max(int((y + 1) * sy), int(y * sy) + 1)
            # Sample central area of each block to avoid generator edge anti-aliasing
            mx, my = (x1 - x0) // 4, (y1 - y0) // 4
            candidates = [src[i, j] for i in range(x0 + mx, x1 - mx) for j in range(y0 + my, y1 - my)]
            if not candidates:
                candidates = [src[int((x0 + x1) / 2), int((y0 + y1) / 2)]]
            votes = Counter(candidates)
            dst[x, y] = votes.most_common(1)[0][0]
    return out


def normalize_image(
    raw_path: Path | str | Image.Image,
    target_path: Path,
    asset_spec: dict[str, Any],
    palette: list[tuple[int, int, int]]
) -> dict[str, Any]:
    width, height = asset_spec["size"]
    if isinstance(raw_path, (str, Path)):
        image = Image.open(raw_path).convert("RGBA")
    else:
        image = raw_path.convert("RGBA")
    
    # If the raw image is at higher resolution (e.g. 4x or 8x scale), downsample with block majority
    if (image.width, image.height) != (width, height):
        image = block_majority_downsample(image, width, height)
        
    pixels = image.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    is_transparent = asset_spec.get("alpha") == "transparent"
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            transparent = a < 128 or is_chroma((r, g, b))
            if is_transparent and transparent:
                pixels[x, y] = (0, 0, 0, 0)
                continue
                
            key = (r, g, b)
            if key not in cache:
                cache[key] = nearest_palette_color(key, palette)
            pixels[x, y] = (*cache[key], 255)

    # If colors exceed asset limit, reduce to the most frequent top-N palette colors
    max_colors = asset_spec.get("max_colors", 16)
    opaque_pixels = [pixels[x, y][:3] for y in range(height) for x in range(width) if pixels[x, y][3] == 255]
    color_counts = Counter(opaque_pixels)
    if len(color_counts) > max_colors:
        allowed_palette = [c for c, _ in color_counts.most_common(max_colors)]
        remap_cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for y in range(height):
            for x in range(width):
                if pixels[x, y][3] == 255:
                    c = pixels[x, y][:3]
                    if c not in remap_cache:
                        remap_cache[c] = nearest_palette_color(c, allowed_palette)
                    pixels[x, y] = (*remap_cache[c], 255)
            
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_path, format="PNG")
    
    used_colors = {px[:3] for px in image.getdata() if px[3] == 255}
    alphas = sorted({px[3] for px in image.getdata()})
    
    return {
        "output_path": str(target_path),
        "size": [image.width, image.height],
        "colors_used": len(used_colors),
        "max_colors": max_colors,
        "within_color_limit": len(used_colors) <= max_colors,
        "alphas": alphas,
        "is_binary_alpha": set(alphas).issubset({0, 255})
    }
