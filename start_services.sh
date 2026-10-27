#!/usr/bin/env bash
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$BASE/logs" "$BASE/output" "$BASE/models"

# Start ComfyUI on GPU 0
if ! lsof -i :8188 >/dev/null 2>&1; then
    echo "Starting ComfyUI on 127.0.0.1:8188 (GPU 0)..."
    CUDA_VISIBLE_DEVICES=0 nohup "$BASE/comfy-venv/bin/python" "$BASE/comfyui/main.py" \
        --listen 127.0.0.1 --port 8188 --cuda-device 0 \
        --models-directory "$BASE/models" \
        --output-directory "$BASE/output" \
        < /dev/null >> "$BASE/logs/comfyui.log" 2>&1 &
    disown
else
    echo "ComfyUI is already listening on :8188"
fi

# Determine MCP server entry point
if [ -f "$BASE/server.py" ]; then
    MCP_SCRIPT="$BASE/server.py"
elif [ -f "$BASE/synopsis_game/services/asset-generator/server.py" ]; then
    MCP_SCRIPT="$BASE/synopsis_game/services/asset-generator/server.py"
else
    MCP_SCRIPT="$BASE/server.py"
fi

# Start synopsis-assets-mcp on :9200
if ! lsof -i :9200 >/dev/null 2>&1; then
    echo "Starting synopsis-assets-mcp on 0.0.0.0:9200 using $MCP_SCRIPT..."
    nohup "$BASE/mcp-venv/bin/python" "$MCP_SCRIPT" \
        < /dev/null >> "$BASE/logs/mcp.log" 2>&1 &
    disown
else
    echo "synopsis-assets-mcp is already listening on :9200"
fi

echo "Services started successfully!"
