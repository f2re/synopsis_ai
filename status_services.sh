#!/usr/bin/env bash
echo "=== Service Ports (8188 ComfyUI, 9200 MCP) ==="
ss -tulpn | grep -E '8188|9200' || echo "None listening on 8188/9200"
echo ""
echo "=== Running Processes ==="
ps aux | grep -E 'server.py|comfyui/main.py' | grep -v grep || echo "No processes found."
echo ""
echo "=== MCP Health Check ==="
python3 -c "
import urllib.request
try:
    res = urllib.request.urlopen('http://127.0.0.1:9200/health', timeout=3)
    print('MCP Server /health:', res.read().decode())
except Exception as e:
    print('MCP Server offline or unreachable:', e)
" 2>/dev/null || true
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
