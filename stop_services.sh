#!/usr/bin/env bash
echo "Stopping synopsis-assets-mcp and ComfyUI..."
pkill -f "server.py" || true
pkill -f "comfyui/main.py" || true
echo "Stopped."
