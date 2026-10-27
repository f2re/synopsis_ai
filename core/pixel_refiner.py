from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from PIL import Image

from core.asset_classes import PALETTE_CONTRACTS, classify_asset

CHROMA = (255, 0, 255)
CHROMA_TOLERANCE = 65


def is_chroma(rgb: tuple[int, int, int]) -> bool:
    return sum(abs(a - b) for a, b in zip(rgb, CHROMA)) <= CHROMA_TOLERANCE


def nearest_palette_color(
    rgb: tuple[int, int, int],
    palette: list[tuple[int, int, int]]
) -> tuple[int, int, int]:
    r, g, b = rgb
    # Weighted Euclidean distance matching human visual sensitivity (CIE approximation)
    return min(palette, key=lambda p: 2 * (p[0] - r) ** 2 + 4 * (p[1] - g) ** 2 + 3 * (p[2] - b) ** 2)


def remove_neutral_background(image: Image.Image, tolerance: int = 35) -> Image.Image:
    """Removes solid or near-uniform backdrop by flood-filling from outer image borders."""
    img = image.convert("RGBA")
    w, h = img.size
    pixels = img.load()

    # Sample corners to determine background color
    corners = [pixels[0, 0][:3], pixels[w - 1, 0][:3], pixels[0, h - 1][:3], pixels[w - 1, h - 1][:3]]
    bg_color = Counter(corners).most_common(1)[0][0]

    # BFS flood-fill from border pixels
    visited = set()
    queue = []

    for x in range(w):
        for y in (0, h - 1):
            if (x, y) not in visited:
                queue.append((x, y))
                visited.add((x, y))

    for y in range(h):
        for x in (0, w - 1):
            if (x, y) not in visited:
                queue.append((x, y))
                visited.add((x, y))

    bg_mask = set()
    while queue:
        cx, cy = queue.pop(0)
        r, g, b, a = pixels[cx, cy]
        dist = sum(abs(c - bc) for c, bc in zip((r, g, b), bg_color))
        # If matches background or is already transparent / chroma
        if dist <= tolerance or is_chroma((r, g, b)) or a < 128:
            bg_mask.add((cx, cy))
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    out = img.copy()
    out_pixels = out.load()
    for (x, y) in bg_mask:
        out_pixels[x, y] = (0, 0, 0, 0)

    return out


def edge_aware_downsample(
    source_img: Image.Image,
    target_w: int,
    target_h: int
) -> Image.Image:
    """Resamples high-res candidate to target logical grid preserving fine edges and landmarks."""
    src = source_img.convert("RGBA")
    sw, sh = src.size
    out = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    dst = out.load()
    src_pixels = src.load()

    step_x = sw / target_w
    step_y = sh / target_h

    for ty in range(target_h):
        for tx in range(target_w):
            x0 = int(tx * step_x)
            x1 = max(int((tx + 1) * step_x), x0 + 1)
            y0 = int(ty * step_y)
            y1 = max(int((ty + 1) * step_y), y0 + 1)

            cell_pixels = [
                src_pixels[ix, iy]
                for ix in range(x0, min(x1, sw))
                for iy in range(y0, min(y1, sh))
            ]

            if not cell_pixels:
                continue

            # Alpha vote: if more than 50% transparent, treat cell as transparent
            opaque_count = sum(1 for px in cell_pixels if px[3] >= 128)
            if opaque_count < len(cell_pixels) * 0.45:
                dst[tx, ty] = (0, 0, 0, 0)
                continue

            opaque_pixels = [px[:3] for px in cell_pixels if px[3] >= 128]
            if not opaque_pixels:
                dst[tx, ty] = (0, 0, 0, 0)
                continue

            # Compute luminance of each pixel to detect high-contrast outlines and accents
            # Standard Rec. 601 luma: Y = 0.299 R + 0.587 G + 0.114 B
            lumas = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in opaque_pixels]
            avg_luma = sum(lumas) / len(lumas)

            # Score candidates: frequency boosted by contrast deviation
            scores: Counter = Counter()
            for rgb, luma in zip(opaque_pixels, lumas):
                contrast_weight = 1.0 + (abs(luma - avg_luma) / 128.0)
                # Boost dark outlines (pupils, glasses, seams)
                if luma < 50:
                    contrast_weight += 0.8
                scores[rgb] += contrast_weight

            best_color = scores.most_common(1)[0][0]
            dst[tx, ty] = (*best_color, 255)

    return out


def cleanup_isolated_noise_pixels(img: Image.Image) -> Image.Image:
    """Eliminates orphaned 1x1 noise pixels without eroding intentional outlines."""
    w, h = img.size
    px = img.load()
    out = img.copy()
    out_px = out.load()

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            cur = px[x, y]
            if cur[3] == 0:
                continue

            # 8-neighbors
            neighbors = [
                px[x + dx, y + dy]
                for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
            ]
            matching = [n for n in neighbors if n == cur]
            if len(matching) == 0:
                # Completely orphaned 1-pixel: replace with dominant neighbor color
                opaque_neighbors = [n for n in neighbors if n[3] == 255]
                if opaque_neighbors:
                    out_px[x, y] = Counter(opaque_neighbors).most_common(1)[0][0]
                else:
                    out_px[x, y] = (0, 0, 0, 0)

    return out


def refine_pixel_candidate(
    raw_image: Image.Image | Path | str,
    target_spec: dict[str, Any],
    palette: list[tuple[int, int, int]],
    output_path: Path
) -> dict[str, Any]:
    """Applies pixel_refiner_v2 pipeline to produce a production-quality pixel asset."""
    if isinstance(raw_image, (str, Path)):
        img = Image.open(raw_image).convert("RGBA")
    else:
        img = raw_image.convert("RGBA")

    target_w, target_h = target_spec.get("size", [128, 128])
    is_transparent = target_spec.get("alpha") == "transparent"

    # 1. Background removal if transparent
    if is_transparent:
        img = remove_neutral_background(img)

    # 2. Edge-aware resampling
    refined = edge_aware_downsample(img, target_w, target_h)

    # 3. Clean stray single-pixel noise
    refined = cleanup_isolated_noise_pixels(refined)

    # 4. Map colors to palette with feature protection
    r_pixels = refined.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for y in range(target_h):
        for x in range(target_w):
            r, g, b, a = r_pixels[x, y]
            if a < 128:
                r_pixels[x, y] = (0, 0, 0, 0)
                continue
            key = (r, g, b)
            if key not in cache:
                cache[key] = nearest_palette_color(key, palette)
            r_pixels[x, y] = (*cache[key], 255)

    # 5. Respect palette contract: target range and hard_max safeguard against noise
    asset_class = classify_asset(target_spec.get("id", ""), target_spec)
    contract = PALETTE_CONTRACTS.get(asset_class, {})
    hard_max = contract.get("hard_max", target_spec.get("max_colors", 48))
    max_colors = max(target_spec.get("max_colors", 32), hard_max)

    opaque_list = [r_pixels[x, y][:3] for y in range(target_h) for x in range(target_w) if r_pixels[x, y][3] == 255]
    color_counts = Counter(opaque_list)

    if len(color_counts) > max_colors:
        # Protect darkest ink outline and brightest accent
        sorted_by_luma = sorted(color_counts.keys(), key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        protected = {sorted_by_luma[0], sorted_by_luma[-1]}
        remaining_slots = max_colors - len(protected)
        most_common = [c for c, _ in color_counts.most_common() if c not in protected][:remaining_slots]
        allowed_palette = list(protected) + most_common

        remap_cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for y in range(target_h):
            for x in range(target_w):
                if r_pixels[x, y][3] == 255:
                    c = r_pixels[x, y][:3]
                    if c not in remap_cache:
                        remap_cache[c] = nearest_palette_color(c, allowed_palette)
                    r_pixels[x, y] = (*remap_cache[c], 255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined.save(output_path, format="PNG")

    # Generate 1x, 2x, 4x previews
    p1 = output_path.with_name(f"{output_path.stem}_1x.png")
    p2 = output_path.with_name(f"{output_path.stem}_2x.png")
    p4 = output_path.with_name(f"{output_path.stem}_4x.png")

    refined.save(p1)
    refined.resize((target_w * 2, target_h * 2), Image.NEAREST).save(p2)
    refined.resize((target_w * 4, target_h * 4), Image.NEAREST).save(p4)

    used_colors = {px[:3] for px in refined.getdata() if px[3] == 255}
    alphas = sorted({px[3] for px in refined.getdata()})

    return {
        "output_path": str(output_path),
        "previews": {
            "1x": str(p1),
            "2x": str(p2),
            "4x": str(p4),
        },
        "size": [target_w, target_h],
        "colors_used": len(used_colors),
        "max_colors": max_colors,
        "within_color_limit": len(used_colors) <= max_colors,
        "alphas": alphas,
        "is_binary_alpha": set(alphas).issubset({0, 255}),
    }
