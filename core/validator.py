from __future__ import annotations

import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path: Path) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    raw = path.read_bytes()
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError(f"File {path.name} is not a valid PNG")

    pos = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = None
    idat = bytearray()

    while pos < len(raw):
        if pos + 8 > len(raw):
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        chunk_type = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("Only non-interlaced 8-bit PNG supported")
            if color_type not in (2, 6):
                raise ValueError(f"Unsupported PNG color type {color_type}")
        elif chunk_type == b"IDAT":
            idat.extend(data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or color_type is None:
        raise ValueError("Missing IHDR chunk")

    channels = 3 if color_type == 2 else 4
    stride = width * channels
    decompressed = zlib.decompress(bytes(idat))
    expected = height * (stride + 1)
    if len(decompressed) != expected:
        raise ValueError(f"Unexpected decompressed size {len(decompressed)} != {expected}")

    rows: list[bytearray] = []
    offset = 0
    previous = bytearray(stride)

    for _ in range(height):
        filter_type = decompressed[offset]
        scan = bytearray(decompressed[offset + 1:offset + 1 + stride])
        offset += stride + 1

        recon = bytearray(stride)
        for i, value in enumerate(scan):
            left = recon[i - channels] if i >= channels else 0
            up = previous[i]
            up_left = previous[i - channels] if i >= channels else 0
            if filter_type == 0:
                result = value
            elif filter_type == 1:
                result = (value + left) & 0xFF
            elif filter_type == 2:
                result = (value + up) & 0xFF
            elif filter_type == 3:
                result = (value + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                result = (value + paeth(left, up, up_left)) & 0xFF
            else:
                raise ValueError(f"Unknown PNG filter {filter_type}")
            recon[i] = result
        rows.append(recon)
        previous = recon

    pixels: list[tuple[int, int, int, int]] = []
    for row in rows:
        for x in range(width):
            i = x * channels
            if channels == 3:
                pixels.append((row[i], row[i + 1], row[i + 2], 255))
            else:
                pixels.append((row[i], row[i + 1], row[i + 2], row[i + 3]))
    return width, height, pixels


def validate_pixel_asset(
    image_path: Path,
    asset_spec: dict[str, Any],
    palette: list[tuple[int, int, int]] | set[tuple[int, int, int]] | None = None
) -> dict[str, Any]:
    if not image_path.exists():
        return {
            "valid": False,
            "errors": [f"File not found: {image_path}"],
            "metrics": {}
        }

    errors: list[str] = []
    try:
        width, height, pixels = read_png(image_path)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"PNG parse error: {exc}"],
            "metrics": {}
        }

    colors = Counter(pixels)
    alphas = sorted({rgba[3] for rgba in colors})
    opaque_colors = sum(1 for rgba in colors if rgba[3] == 255)

    expected_size = tuple(asset_spec["size"])
    if (width, height) != expected_size:
        errors.append(f"Size mismatch: {(width, height)} != {expected_size}")

    if any(alpha not in (0, 255) for alpha in alphas):
        errors.append(f"Fractional alpha forbidden (only 0 or 255 allowed): found {alphas}")

    max_colors = int(asset_spec.get("max_colors", 64))
    if opaque_colors > max_colors:
        errors.append(f"Too many colors: {opaque_colors} > {max_colors}")

    if asset_spec.get("alpha") == "opaque" and alphas != [255]:
        errors.append("Asset must be fully opaque (alpha == 255 everywhere)")

    frame = asset_spec.get("frame")
    if frame:
        fw, fh = frame
        if width % fw or height % fh:
            errors.append(f"Sheet dimensions {(width, height)} not aligned to frame grid ({fw}, {fh})")

    palette_set = set(palette) if palette else None
    if palette_set:
        outside = sorted({rgba[:3] for rgba in colors if rgba[3] == 255 and rgba[:3] not in palette_set})
        if outside:
            sample = ", ".join("#%02X%02X%02X" % rgb for rgb in outside[:5])
            errors.append(f"{len(outside)} colors outside palette, e.g. {sample}")

    metrics = {
        "width": width,
        "height": height,
        "expected_size": list(expected_size),
        "opaque_colors": opaque_colors,
        "max_colors": max_colors,
        "alphas": alphas,
        "frame": frame,
        "valid": len(errors) == 0
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "metrics": metrics
    }
