#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="user@10.7.88.12"
REMOTE_BASE="/home/user/synopsis-ai"
SSH_OPTS="-o ConnectTimeout=30"

echo "=== Deploying synopsis-assets-mcp to $REMOTE_HOST ==="

# 1. Create remote base layout
ssh $SSH_OPTS "$REMOTE_HOST" "mkdir -p \
  $REMOTE_BASE/models/diffusion_models \
  $REMOTE_BASE/models/checkpoints \
  $REMOTE_BASE/models/loras \
  $REMOTE_BASE/models/controlnet \
  $REMOTE_BASE/models/clip \
  $REMOTE_BASE/models/vae \
  $REMOTE_BASE/workflows \
  $REMOTE_BASE/output/candidates \
  $REMOTE_BASE/datasets/style_lora \
  $REMOTE_BASE/logs \
  $REMOTE_BASE/synopsis_game/assets/source/palettes \
  $REMOTE_BASE/synopsis_game/services/asset-generator/core \
  $REMOTE_BASE/synopsis_game/services/asset-generator/workflows \
  $REMOTE_BASE/synopsis_game/services/asset-generator/config \
  $REMOTE_BASE/synopsis_game/services/asset-generator/scripts"

# 2. Copy image-spec and palettes
echo "Copying specs and palettes..."
scp $SSH_OPTS assets/source/kn01_level1.image-spec.json "$REMOTE_HOST:$REMOTE_BASE/synopsis_game/assets/source/"
scp $SSH_OPTS assets/source/palettes/*.gpl "$REMOTE_HOST:$REMOTE_BASE/synopsis_game/assets/source/palettes/"

# 3. Copy asset-generator code
echo "Copying services/asset-generator..."
scp $SSH_OPTS services/asset-generator/*.py services/asset-generator/requirements.txt "$REMOTE_HOST:$REMOTE_BASE/synopsis_game/services/asset-generator/"
scp $SSH_OPTS services/asset-generator/config/*.yaml "$REMOTE_HOST:$REMOTE_BASE/synopsis_game/services/asset-generator/config/"
scp $SSH_OPTS services/asset-generator/core/*.py "$REMOTE_HOST:$REMOTE_BASE/synopsis_game/services/asset-generator/core/"
scp $SSH_OPTS services/asset-generator/workflows/*.json "$REMOTE_HOST:$REMOTE_BASE/synopsis_game/services/asset-generator/workflows/"
scp $SSH_OPTS services/asset-generator/workflows/*.json "$REMOTE_HOST:$REMOTE_BASE/workflows/"
scp $SSH_OPTS services/asset-generator/scripts/*.py "$REMOTE_HOST:$REMOTE_BASE/synopsis_game/services/asset-generator/scripts/"

# 4. Install remote runner scripts
echo "Installing runner scripts on remote..."
ssh $SSH_OPTS "$REMOTE_HOST" "cat << 'EOF' > $REMOTE_BASE/start_services.sh
#!/usr/bin/env bash
set -e
BASE=\"/home/user/synopsis-ai\"
mkdir -p \$BASE/logs

# Start ComfyUI on GPU 0
if ! lsof -i :8188 >/dev/null 2>&1; then
    echo \"Starting ComfyUI on 127.0.0.1:8188 (GPU 0)...\"
    CUDA_VISIBLE_DEVICES=0 nohup \$BASE/comfy-venv/bin/python \$BASE/comfyui/main.py \
        --listen 127.0.0.1 --port 8188 --cuda-device 0 \
        --models-directory \$BASE/models \
        --output-directory \$BASE/output \
        > \$BASE/logs/comfyui.log 2>&1 &
    sleep 3
else
    echo \"ComfyUI is already listening on :8188\"
fi

# Start synopsis-assets-mcp on :9200
if ! lsof -i :9200 >/dev/null 2>&1; then
    echo \"Starting synopsis-assets-mcp on 0.0.0.0:9200...\"
    nohup \$BASE/mcp-venv/bin/python \$BASE/synopsis_game/services/asset-generator/server.py \
        > \$BASE/logs/mcp.log 2>&1 &
    sleep 2
else
    echo \"synopsis-assets-mcp is already listening on :9200\"
fi

echo \"Services started! Check \$BASE/logs/\"
EOF
chmod +x $REMOTE_BASE/start_services.sh

cat << 'EOF' > $REMOTE_BASE/stop_services.sh
#!/usr/bin/env bash
echo \"Stopping synopsis-assets-mcp and ComfyUI...\"
pkill -f \"services/asset-generator/server.py\" || true
pkill -f \"comfyui/main.py\" || true
echo \"Stopped.\"
EOF
chmod +x $REMOTE_BASE/stop_services.sh

cat << 'EOF' > $REMOTE_BASE/status_services.sh
#!/usr/bin/env bash
echo \"=== Service Ports ===\"
ss -tulpn | grep -E '8188|9200' || echo \"None listening on 8188/9200\"
echo \"=== Processes ===\"
ps aux | grep -E 'server.py|comfyui/main.py' | grep -v grep || true
EOF
chmod +x $REMOTE_BASE/status_services.sh
"

echo "=== Deployment finished successfully ==="
