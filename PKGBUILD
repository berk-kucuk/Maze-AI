# Maintainer: Berk Küçük <dev.berkkucukk@gmail.com>
pkgname=maze-ai
pkgver=1.17.1
pkgrel=1
pkgdesc="Agentic AI assistant for Maze Linux — local models via Ollama, native tool calling, monochrome UI"
arch=('any')
url="https://github.com/berkkucukk/Maze-AI"
license=('GPL-3.0-or-later')
depends=(
    'python'
    'pyside6'
    'python-requests'
)
optdepends=(
    'ollama: run local open-source models'
    'tesseract: OCR — read text out of images (ocr_image tool)'
    'tesseract-data-eng: English OCR language data'
    'tesseract-data-tur: Turkish OCR language data'
    'wl-clipboard: clipboard support on Wayland (clipboard_copy tool)'
    'xclip: clipboard support on X11 (clipboard_copy tool)'
    'grim: screenshots on Wayland (screenshot tool)'
    'maim: screenshots on X11 (screenshot tool)'
    'libnotify: desktop notifications and reminders (notify-send)'
    'kdotool: read a single application window on Wayland (read_window tool)'
    'xdotool: read a single application window on X11 (read_window tool)'
)
makedepends=(
    'python-build'
    'python-installer'
    'python-wheel'
    'python-setuptools'
)
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
    install -Dm644 maze_ai/resources/logo.png \
        "$pkgdir/usr/share/pixmaps/maze-ai.png"
}
