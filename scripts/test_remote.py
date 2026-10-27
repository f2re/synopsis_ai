#!/usr/bin/env python3
"""E2E Test Suite for synopsis-assets-mcp on the remote GPU server (10.7.88.12:9200)."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "synopsis-tester/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, data: dict) -> dict:
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "User-Agent": "synopsis-tester/1.0"
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test remote synopsis-assets-mcp")
    parser.add_argument("--base-url", default="http://10.7.88.12:9200", help="Base URL of MCP server")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print(f"=== Testing synopsis-assets-mcp at {base} ===")
    
    passed = 0
    total = 0

    # 1. Health check
    total += 1
    try:
        health = get_json(f"{base}/health")
        assert health.get("status") == "ok"
        print(f"✅ [1/{total}] /health: OK ({health.get('service')})")
        passed += 1
    except Exception as exc:
        print(f"❌ [1/{total}] /health failed: {exc}")

    # 2. GPU Status
    total += 1
    try:
        gpu = get_json(f"{base}/api/v1/gpu/status")
        assert gpu.get("success") is True
        gpus = gpu.get("gpus", [])
        print(f"✅ [2/{total}] /api/v1/gpu/status: Found {len(gpus)} GPU(s):")
        for g in gpus:
            print(f"    - GPU {g['index']}: {g['name']} ({g['memory_used_mb']} / {g['memory_total_mb']} MB, util: {g['gpu_utilization_pct']}%)")
        passed += 1
    except Exception as exc:
        print(f"❌ [2/{total}] /api/v1/gpu/status failed: {exc}")

    # 3. Assets list & missing assets
    total += 1
    try:
        missing = get_json(f"{base}/api/v1/assets/missing")
        print(f"✅ [3/{total}] /api/v1/assets/missing: {len(missing)} assets pending generation")
        passed += 1
    except Exception as exc:
        print(f"❌ [3/{total}] /api/v1/assets/missing failed: {exc}")

    # 4. Asset spec retrieval
    total += 1
    test_asset_id = "BLD-RADAR-DAY"
    try:
        spec = get_json(f"{base}/api/v1/assets/{test_asset_id}")
        assert spec.get("id") == test_asset_id
        print(f"✅ [4/{total}] /api/v1/assets/{test_asset_id}: '{spec.get('title')}' ({spec.get('size')}, alpha={spec.get('alpha')})")
        passed += 1
    except Exception as exc:
        print(f"❌ [4/{total}] /api/v1/assets/{test_asset_id} failed: {exc}")

    # 5. Candidate Generation, Normalization and Validation
    total += 1
    try:
        gen = post_json(f"{base}/api/v1/assets/{test_asset_id}/generate", {"candidates": 1, "backend": "auto"})
        assert gen.get("candidates_count", 0) > 0
        cand = gen["candidates"][0]
        norm = cand["normalization"]
        val = cand["validation"]
        print(f"✅ [5/{total}] /api/v1/assets/{test_asset_id}/generate:")
        print(f"    - Candidate: {cand['candidate_path']}")
        print(f"    - Normalization: size {norm['size']}, colors {norm['colors_used']}/{norm['max_colors']}, binary_alpha: {norm['is_binary_alpha']}")
        print(f"    - Validation valid: {val['valid']} (errors: {val['errors']})")
        passed += 1
    except Exception as exc:
        print(f"❌ [5/{total}] Candidate generation pipeline failed: {exc}")

    print("\n" + "=" * 50)
    print(f"Test summary: {passed}/{total} tests passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
