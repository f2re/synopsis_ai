from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
import httpx


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", output_dir: Path | None = None):
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir or Path("/home/user/synopsis-ai/comfyui/output")

    async def is_online(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/system_stats")
                return res.status_code == 200
        except Exception:
            return False

    async def get_system_stats(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{self.base_url}/system_stats")
            res.raise_for_status()
            return res.json()

    async def get_queue(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{self.base_url}/queue")
            res.raise_for_status()
            return res.json()

    async def queue_prompt(self, workflow_graph: dict[str, Any], client_id: str = "synopsis-mcp") -> str:
        payload = {
            "prompt": workflow_graph,
            "client_id": client_id
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(f"{self.base_url}/prompt", json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"ComfyUI prompt error {res.status_code}: {res.text}")
            data = res.json()
            return data["prompt_id"]

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{self.base_url}/history/{prompt_id}")
            res.raise_for_status()
            return res.json()

    async def wait_for_completion(self, prompt_id: str, timeout: float = 300.0) -> list[str]:
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            history = await self.get_history(prompt_id)
            if prompt_id in history:
                prompt_info = history[prompt_id]
                outputs = prompt_info.get("outputs", {})
                image_filenames: list[str] = []
                for _, node_output in outputs.items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            filename = img.get("filename")
                            if filename:
                                image_filenames.append(filename)
                return image_filenames
            await asyncio.sleep(1.0)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout} seconds")

    async def interrupt(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(f"{self.base_url}/interrupt")
                return res.status_code == 200
        except Exception:
            return False
