from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import httpx


class PixelLabClient:
    """External fallback backend for specialized tilesets, 4/8-way characters, and complex animations.
    Note: Output from PixelLab is strictly candidate material and subject to PixelLab ToS (no fine-tuning models on output).
    """

    def __init__(self, api_url: str = "https://api.pixellab.ai/mcp", api_key: str | None = None):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or os.getenv("PIXELLAB_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def generate_tileset(
        self,
        prompt: str,
        tile_size: int = 16,
        grid_width: int = 4,
        grid_height: int = 4
    ) -> dict[str, Any]:
        if not self.is_configured():
            return {"success": False, "error": "PixelLab API key not configured"}
            
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "method": "create_topdown_tileset",
            "params": {
                "prompt": prompt,
                "tile_size": tile_size,
                "columns": grid_width,
                "rows": grid_height
            }
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{self.api_url}/rpc", headers=headers, json=payload)
            res.raise_for_status()
            return res.json()

    async def generate_character_directions(
        self,
        character_description: str,
        directions: int = 4,
        size: tuple[int, int] = (32, 48)
    ) -> dict[str, Any]:
        if not self.is_configured():
            return {"success": False, "error": "PixelLab API key not configured"}
            
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "method": "create_character_directions",
            "params": {
                "prompt": character_description,
                "directions": directions,
                "width": size[0],
                "height": size[1]
            }
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{self.api_url}/rpc", headers=headers, json=payload)
            res.raise_for_status()
            return res.json()
