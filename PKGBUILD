# Maintainer: Berk Küçük <dev.berkkucukk@gmail.com>
pkgname=maze-ai
pkgver=1.23.0
pkgrel=1
pkgdesc="Agentic AI assistant for Maze Linux — local models via Ollama, native tool calling, monochrome UI"
arch=('any')
url="https://github.com/berkkucukk/Maze-AI"
license=('GPL-3.0-or-later')
depends=(
    'python'
    'pyside6'
    'python-requests'
    'python-numpy'          # folder chat: fast similarity search
    # Local models
    'ollama'
    # Knowledge: Arch Wiki offline, man pages, Arch news, project maps
    'arch-wiki-docs'
    'man-db'
    'git'
    # Checking code it writes (statically — it never runs it)
    'ruff'
    'shellcheck'
    # Seeing the screen and reading text in images
    'tesseract'
    'tesseract-data-eng'
    'tesseract-data-tur'
    'grim'
    'slurp'
    'maim'
    # Clipboard, notifications and single-window capture (Wayland and X11)
    'wl-clipboard'
    'xclip'
    'libnotify'
    'xdotool'
)
optdepends=(
    'ollama-cuda: run models on an NVIDIA GPU (much faster)'
    'ollama-rocm: run models on an AMD GPU (much faster)'
    'ollama-vulkan: run models on other GPUs through Vulkan'
    'spectacle: KDE screenshots and area selection (part of Plasma)'
    'kdotool: read a single window on KDE Wayland (AUR)'
    'nodejs: syntax-check JavaScript the assistant writes'
    'gcc: syntax-check C and C++ the assistant writes'
)
makedepends=(
    'python-build'
    'python-installer'
    'python-wheel'
    'python-setuptools'
)
install=maze-ai.install
source=("$pkgname-$pkgver.tar.gz::$url/archive/refs/tags/v$pkgver.tar.gz")
sha256sums=('SKIP')  # run `updpkgsums` after tagging the release to pin this

build() {
    cd "$srcdir/Maze-AI-$pkgver"
    python -m build --wheel --no-isolation
}

package() {
    cd "$srcdir/Maze-AI-$pkgver"
    python -m installer --destdir="$pkgdir" dist/*.whl

    # Desktop entry + icons
    install -Dm644 maze-ai.desktop \
        "$pkgdir/usr/share/applications/maze-ai.desktop"
    # Same file again where kglobalacceld looks for default shortcuts
    # (X-KDE-Shortcuts on the Quick Ask action → Meta+M).
    install -Dm644 maze-ai.desktop \
        "$pkgdir/usr/share/kglobalaccel/maze-ai.desktop"
    install -Dm644 maze_ai/resources/logo.png \
        "$pkgdir/usr/share/icons/hicolor/512x512/apps/maze-ai.png"
    install -Dm644 maze_ai/resources/logo.svg \
        "$pkgdir/usr/share/icons/hicolor/scalable/apps/maze-ai.svg"
    # The simplified mark where panels and menus draw it small.
    for size in 16 22 24 32; do
        install -Dm644 maze_ai/resources/logo-small.svg \
            "$pkgdir/usr/share/icons/hicolor/${size}x${size}/apps/maze-ai.svg"
    done
    install -Dm644 maze_ai/resources/logo.png \
        "$pkgdir/usr/share/pixmaps/maze-ai.png"
}
