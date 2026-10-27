#!/usr/bin/env bash
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "========================================================"
echo "Initializing Synopsis AI Generation & MCP Infrastructure"
echo "Target Base: $BASE_DIR"
echo "========================================================"

# 1. Create runtime directories
echo "[1/6] Creating runtime directories..."
mkdir -p "$BASE_DIR/logs" \
         "$BASE_DIR/output/candidates" \
         "$BASE_DIR/assets/work" \
         "$BASE_DIR/models/checkpoints" \
         "$BASE_DIR/models/diffusers" \
         "$BASE_DIR/models/loras" \
         "$BASE_DIR/models/vae" \
         "$BASE_DIR/models/controlnet"

# 2. Setup mcp-venv
echo "[2/6] Setting up mcp-venv..."
if [ ! -d "$BASE_DIR/mcp-venv" ]; then
    python3 -m venv "$BASE_DIR/mcp-venv"
fi
"$BASE_DIR/mcp-venv/bin/pip" install --upgrade pip
"$BASE_DIR/mcp-venv/bin/pip" install -r "$BASE_DIR/requirements.txt"

# 3. Setup ComfyUI repository
echo "[3/6] Checking ComfyUI core..."
if [ ! -d "$BASE_DIR/comfyui/.git" ]; then
    echo "Cloning ComfyUI (v0.43.0)..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$BASE_DIR/comfyui"
    (cd "$BASE_DIR/comfyui" && git checkout 426f5eea 2>/dev/null || true)
fi

# 4. Setup comfy-venv
echo "[4/6] Setting up comfy-venv..."
if [ ! -d "$BASE_DIR/comfy-venv" ]; then
    python3 -m venv "$BASE_DIR/comfy-venv"
fi
"$BASE_DIR/comfy-venv/bin/pip" install --upgrade pip
if [ -f "$BASE_DIR/comfyui/requirements.txt" ]; then
    echo "Installing ComfyUI requirements..."
    "$BASE_DIR/comfy-venv/bin/pip" install -r "$BASE_DIR/comfyui/requirements.txt" || true
fi

# 5. Check neural weights
echo "[5/6] Checking AI models..."
if [ -d "$BASE_DIR/models/diffusers/Z-Image-Turbo" ] || [ -f "$BASE_DIR/models/checkpoints/z_image_turbo.safetensors" ]; then
    echo "✓ Z-Image-Turbo weights found in models/."
else
    echo "Notice: Z-Image-Turbo weights not detected in models/."
    echo "Run '$BASE_DIR/mcp-venv/bin/python scripts/download_z_image_turbo.py' to download local neural generator weights."
fi

# 6. Make scripts executable
echo "[6/6] Setting permissions..."
chmod +x "$BASE_DIR"/*.sh "$BASE_DIR/systemd"/*.sh 2>/dev/null || true

echo "========================================================"
echo "Initialization Complete!"
echo "To start services manually: ./start_services.sh"
echo "To check status:           ./status_services.sh"
echo "To stop services:            ./stop_services.sh"
echo "To install systemd units:    ./systemd/install_services.sh"
echo "========================================================"
