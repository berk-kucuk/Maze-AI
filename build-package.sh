#!/usr/bin/env bash
#
# build-package.sh — build the Maze AI pacman package (.pkg.tar.zst) from the
# current source tree, ready to drop into a repository.
#
# Usage:
#   ./build-package.sh                 # build from the working tree (local)
#   REPO_DB=/path/mazelinux.db.tar.gz ./build-package.sh
#                                      # …and add it to a pacman repo database
#
# For a tagged release you'd instead run `updpkgsums && makepkg` against the
# committed PKGBUILD (its source points at the GitHub release tarball). This
# script is the convenience path that packages exactly what's in the tree now.

set -euo pipefail
cd "$(dirname "$0")"

pkgname="maze-ai"
# Version comes straight from pyproject.toml so it can never drift.
pkgver="$(grep -m1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')"
prefix="Maze-AI-${pkgver}"          # must match the dir name PKGBUILD cd's into
tarball="${pkgname}-${pkgver}.tar.gz"

workdir="build/pkg"
outdir="dist-pkg"
rm -rf "$workdir"
mkdir -p "$workdir" "$outdir"

echo ">> Packaging ${pkgname} ${pkgver}"

# ── 1. staged source tarball (prefixed so makepkg extracts to $prefix/) ──────
staging="$(mktemp -d)"
mkdir -p "${staging}/${prefix}"
cp -r \
    maze_ai \
    pyproject.toml \
    README.md \
    requirements.txt \
    maze-ai.desktop \
    install.sh \
    run.sh \
    "${staging}/${prefix}/"
# Drop caches that may have crept into the source tree.
find "${staging}/${prefix}" -name '__pycache__' -type d -prune -exec rm -rf {} +
tar -C "$staging" -czf "${workdir}/${tarball}" "${prefix}"
rm -rf "$staging"

# ── 2. local PKGBUILD pointing at the staged tarball ─────────────────────────
sed -E \
    -e "s#^source=.*#source=(\"${tarball}\")#" \
    -e "s#^sha256sums=.*#sha256sums=('SKIP')#" \
    PKGBUILD > "${workdir}/PKGBUILD"

# ── 3. build ─────────────────────────────────────────────────────────────────
(
    cd "$workdir"
    makepkg -f --noconfirm
)

# ── 4. collect artifacts ─────────────────────────────────────────────────────
# Drop older builds of this package so the output dir only holds the current
# version (otherwise a glob install would try to install several versions).
rm -f "$outdir/${pkgname}-"*.pkg.tar.* 2>/dev/null || true
cp "$workdir"/*.pkg.tar.* "$outdir"/ 2>/dev/null || true

pkgfile=$(ls -1 "$outdir/${pkgname}-${pkgver}-"*.pkg.tar.* 2>/dev/null | head -1)
[[ -n "$pkgfile" ]] || { echo ">> ERROR: no package produced"; exit 1; }
echo ">> Built: $pkgfile"

# ── 5. optional: add to a pacman repository database ─────────────────────────
if [[ -n "${REPO_DB:-}" ]]; then
    echo ">> Adding to repo database: ${REPO_DB}"
    repo-add "${REPO_DB}" "$pkgfile"
fi

echo ">> Done. Install locally with:  sudo pacman -U ${pkgfile}"
