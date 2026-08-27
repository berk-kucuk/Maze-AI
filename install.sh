#!/usr/bin/env bash
# Install Maze AI for the current user (pip --user + desktop entry + icon).
set -euo pipefail
cd "$(dirname "$0")"

echo ">> Installing Maze AI…"
python -m pip install --user .

# Desktop entry
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/512x512/apps"
mkdir -p "$APPS_DIR" "$ICON_DIR"

install -m644 maze_ai/resources/logo.png "$ICON_DIR/maze-ai.png"
install -m644 maze-ai.desktop "$APPS_DIR/maze-ai.desktop"

command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$APPS_DIR" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && \
    gtk-update-icon-cache -f "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" || true

echo ">> Done. Launch it from your app menu or run: maze-ai"
echo "   (make sure ~/.local/bin is on your PATH)"
