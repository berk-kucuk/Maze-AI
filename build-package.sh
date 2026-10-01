#!/usr/bin/env bash
#
# build-package.sh — build the Maze AI pacman package (.pkg.tar.zst) from the
# current source tree, ready to drop into a repository.
#
# Usage:
#   ./build-package.sh                 # build from the working tree (local)
#   ./build-package.sh --install       # …then install it, with every dependency
#   REPO_DB=/path/mazelinux.db.tar.gz ./build-package.sh
#                                      # …and add it to a pacman repo database
#
# Missing build tools (python-build, python-installer, …) are installed first,
# so a fresh machine needs nothing but this script. Runtime dependencies come
# with the package itself: `pacman -U` pulls them from the repositories.
#
# For a tagged release you'd instead run `updpkgsums && makepkg` against the
# committed PKGBUILD (its source points at the GitHub release tarball). This
# script is the convenience path that packages exactly what's in the tree now.

set -euo pipefail
cd "$(dirname "$0")"

install_after=0
[[ "${1:-}" == "--install" ]] && install_after=1

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

# ── 0. build tools ───────────────────────────────────────────────────────────
# Read makedepends from the PKGBUILD itself so the two never drift, and install
# whatever is missing in one go (pacman -T prints exactly the unsatisfied ones).
mapfile -t build_deps < <(bash -c 'source ./PKGBUILD >/dev/null 2>&1; printf "%s\n" base-devel "${makedepends[@]}"')
mapfile -t missing < <(pacman -T "${build_deps[@]}" 2>/dev/null || true)
if (( ${#missing[@]} )); then
    echo ">> Installing build tools: ${missing[*]}"
    sudo pacman -S --needed --noconfirm --asdeps "${missing[@]}"
fi

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
# The .install script sits next to the PKGBUILD, not inside the source.
# Drop caches that may have crept into the source tree.
find "${staging}/${prefix}" -name '__pycache__' -type d -prune -exec rm -rf {} +
tar -C "$staging" -czf "${workdir}/${tarball}" "${prefix}"
rm -rf "$staging"

# ── 2. local PKGBUILD pointing at the staged tarball ─────────────────────────
sed -E \
    -e "s#^source=.*#source=(\"${tarball}\")#" \
    -e "s#^sha256sums=.*#sha256sums=('SKIP')#" \
    PKGBUILD > "${workdir}/PKGBUILD"
cp maze-ai.install "${workdir}/"

# ── 3. build ─────────────────────────────────────────────────────────────────
(
    cd "$workdir"
    # --nodeps: building needs only the tools above; the runtime dependencies
    # are pulled in when the package is installed, not to build it.
    makepkg -f --noconfirm --nodeps
)

# ── 4. collect artifacts ─────────────────────────────────────────────────────
# Drop older builds of this package so the output dir only holds the current
# version (otherwise a glob install would try to install several versions).
rm -f "$outdir/${pkgname}-"*.pkg.tar.* 2>/dev/null || true
cp "$workdir"/*.pkg.tar.* "$outdir"/ 2>/dev/null || true

built=("$outdir/${pkgname}-${pkgver}-"*.pkg.tar.*)
pkgfile=""
[[ -e "${built[0]}" ]] && pkgfile="${built[0]}"
[[ -n "$pkgfile" ]] || { echo ">> ERROR: no package produced"; exit 1; }
echo ">> Built: $pkgfile"

# ── 5. optional: add to a pacman repository database ─────────────────────────
if [[ -n "${REPO_DB:-}" ]]; then
    echo ">> Adding to repo database: ${REPO_DB}"
    repo-add "${REPO_DB}" "$pkgfile"
fi

if (( install_after )); then
    echo ">> Installing ${pkgfile} and its dependencies"
    sudo pacman -U --needed --noconfirm "$pkgfile"
else
    echo ">> Done. Install locally with:  sudo pacman -U ${pkgfile}"
    echo "   (pacman installs every dependency it lists automatically)"
fi
