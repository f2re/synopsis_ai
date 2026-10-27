#!/usr/bin/env bash
set -e
DEST="$HOME/.config/systemd/user"
mkdir -p "$DEST"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing systemd user service units to $DEST..."
cp "$DIR/synopsis-assets-mcp.service" "$DEST/"
cp "$DIR/comfyui.service" "$DEST/"

systemctl --user daemon-reload
systemctl --user enable synopsis-assets-mcp.service comfyui.service
echo "Services installed and enabled!"
echo "To start services: systemctl --user start comfyui.service synopsis-assets-mcp.service"
echo "To check status:   systemctl --user status synopsis-assets-mcp.service"
