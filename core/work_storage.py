from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT_DIR.parents[1] / "assets/work"


class WorkStorage:
    def __init__(self, asset_id: str, base_dir: Path | None = None):
        self.asset_id = asset_id
        self.root = (base_dir or WORK_DIR) / asset_id
        self.raw_dir = self.root / "raw"
        self.selected_dir = self.root / "selected"
        self.refined_dir = self.root / "refined"
        self.review_dir = self.root / "review"
        self.approved_dir = self.root / "approved"

        for d in (self.raw_dir, self.selected_dir, self.refined_dir, self.review_dir, self.approved_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_spec(self, spec: dict[str, Any]) -> Path:
        p = self.root / "spec.json"
        p.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def save_prompt(self, pos_prompt: str, neg_prompt: str, meta: dict[str, Any] | None = None) -> Path:
        p = self.root / "prompt.json"
        data = {
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "metadata": meta or {},
        }
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def get_raw_path(self, candidate_index: int) -> Path:
        return self.raw_dir / f"candidate-{candidate_index:02d}.png"

    def list_raw_candidates(self) -> list[Path]:
        return sorted(self.raw_dir.glob("candidate-*.png"))

    def select_candidate(self, candidate_path: Path) -> Path:
        for old in self.selected_dir.glob("*.png"):
            try:
                old.unlink()
            except Exception:
                pass
        target = self.selected_dir / f"selected_{candidate_path.name}"
        shutil.copy2(candidate_path, target)
        return target

    def get_refined_path(self, suffix: str = "refined") -> Path:
        return self.refined_dir / f"{self.asset_id}_{suffix}.png"

    def save_review_report(self, report: dict[str, Any]) -> Path:
        p = self.review_dir / "report.json"
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def approve_asset(self, source_path: Path) -> Path:
        target = self.approved_dir / f"{self.asset_id}.png"
        shutil.copy2(source_path, target)
        return target
