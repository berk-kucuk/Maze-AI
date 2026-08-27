#!/usr/bin/env bash
# Dev launcher: runs Maze AI straight from the source tree.
set -euo pipefail
cd "$(dirname "$0")"
exec python -m maze_ai "$@"
