# Maze AI

**Agentic AI assistant for Maze Linux** — a native desktop application that brings large language models to your terminal, without cloud dependencies. Uses local models via [Ollama](https://ollama.ai), runs fully offline, and speaks Turkish out of the box.

<div align="center">

![Version](https://img.shields.io/badge/version-1.18.0-blue)
![License](https://img.shields.io/badge/license-GPL--3.0--or--later-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Qt](https://img.shields.io/badge/Qt-6.6+-teal)

</div>

## Features

- **Local-first**: All processing on your machine via Ollama. No account, no tracking, no cloud.
- **Native GUI**: Fast PySide6 interface in the Maze monochrome design — history grouped by date, a centred reading column, crisp vector icons, and a frameless window you can resize, maximize and drag.
- **Agentic**: Tools for reading the clipboard, files, directories, your shell history, and capturing the screen.
- **Turkish native**: Full support for Turkish language detection (morphological suffix analysis), UI translation, and prompt responses in Turkish.
- **60 FPS streaming**: Token-by-token response animation at frame-perfect cadence — no visible lumps or jumps in text reveal.
- **"Quick Ask"**: Copy text → hit the hotkey → get an answer. One-button clipboard actions: Explain, Fix, Translate, Summarize.
- **Vision ready**: Supports vision models (e.g., `llava`) when you read the screen or paste images.
- **Keyboard-first**: every action has a shortcut — press **Ctrl+/** in the app for the full list. See [Keyboard shortcuts](#keyboard-shortcuts).
- **Safe by design**: model output never renders as HTML, links open only after you see the real address, and chats, backups and pasted screenshots are stored owner-only (0600/0700).

## Install

### From the Maze repository

**On Maze Linux** the repository is already configured:

```bash
sudo pacman -S maze-ai
```

**On Arch Linux and Arch-based distributions**, add the repository once:

1. Import and trust the Maze signing key:

   ```bash
   curl -O https://mazerepo.berkkucukk.com.tr/packages/mazelinux.gpg
   gpg --show-keys --with-fingerprint mazelinux.gpg
   sudo pacman-key --add mazelinux.gpg
   sudo pacman-key --lsign-key 7C4D515A6B930CB04794CEF6147C8159B3E2EE5F
   ```

   The fingerprint `gpg` prints must be `7C4D 515A 6B93 0CB0 4794  CEF6 147C 8159 B3E2 EE5F`.

2. Add the repository to the end of `/etc/pacman.conf`:

   ```ini
   [mazelinux]
   SigLevel = Required DatabaseOptional
   Server = https://mazerepo.berkkucukk.com.tr/packages
   ```

3. Sync and install:

   ```bash
   sudo pacman -Syu maze-ai
   ```

Optionally install `mazelinux-keyring` as well; it keeps the signing key up to date through pacman.

Remove with `sudo pacman -Rns maze-ai`.

### Build from source

The package is built from this working tree by `build-package.sh` and installed with pacman:

```bash
sudo pacman -S --needed base-devel git
git clone https://github.com/berk-kucuk/Maze-AI.git
cd Maze-AI
sudo pacman -S --needed $(bash -c 'source PKGBUILD; echo "${depends[@]}" "${makedepends[@]}"')
./build-package.sh
sudo pacman -U dist-pkg/maze-ai-*.pkg.tar.zst
```

Maze AI needs **Ollama** running on `localhost:11434` (see Quick Start).

For development, install it editable into a virtual environment instead: `pip install -e ".[dev]"`.

## Quick Start

### 1. Start Ollama

```bash
# Pull a model (first time only)
ollama pull mistral

# Run the server (default: localhost:11434)
ollama serve
```

Or pick another model: `llama2`, `neural-chat`, `orca-mini`, etc. Smaller models (7B) are fast; larger ones (13B+) are smarter.

### 2. Launch Maze AI

```bash
maze-ai
```

A window appears. Type your question and press **Enter**, or use **Quick Ask**:

- Copy some text
- Press the global hotkey (default: `Meta+M`, changeable in System Settings → Shortcuts)
- A small window opens with the clipboard text
- Pick an action or refine the question
- See the answer stream in real-time

Press **Ctrl+/** (or **F1**) at any time to see every keyboard shortcut.

### Keyboard shortcuts

| Chats | | Messages | |
|---|---|---|---|
| New chat | `Ctrl+N` | Send / new line | `Enter` / `Shift+Enter` |
| Search chats | `Ctrl+F` | Stop generating | `Esc` |
| Previous / next chat | `Alt+↑` / `Alt+↓` | Regenerate the last answer | `Ctrl+R` |
| Rename chat | `F2` | Copy the last answer | `Ctrl+Shift+C` |
| Export chat | `Ctrl+E` | Edit the last message | `↑` in an empty box |
| Delete chat | `Ctrl+Shift+Backspace` | Attach an image | `Ctrl+O` |
| Show / hide history | `Ctrl+B` | Focus the message box | `Ctrl+L` |

| Window | | Quick Ask & approvals | |
|---|---|---|---|
| Settings | `Ctrl+,` | Quick Ask (anywhere) | `Meta+M` |
| Reminders | `Ctrl+Shift+R` | Copy answer / continue in chat | `Ctrl+Shift+C` / `Ctrl+Shift+Enter` |
| Keyboard shortcuts | `Ctrl+/`, `F1` | Clipboard actions | `Alt+1` … `Alt+4` |
| Maximize / restore | `F11` | Approve a command | `Ctrl+Enter` |
| Hide the window | `Ctrl+W` | Deny / always allow | `Esc` / `Alt+A` |
| Quit | `Ctrl+Q` | | |

A plain **Enter** never approves a command: the approval dialog appears on its own, often while you are typing, so approving takes **Ctrl+Enter** or a click — and only after the dialog has been on screen for a moment.

### 3. Settings

Open **⚙ Settings** (inside the chat window) to:
- Choose your model (must be running in Ollama)
- Pick your language (Turkish or English)
- Set the global hotkey
- Adjust token reveal speed and other animations
- Configure tool approval gating (for sensitive operations)

## Architecture

```
maze_ai/
├── app.py              # Main window, event loop
├── config.py           # Settings, hotkey binding
├── i18n.py             # Turkish/English translations (100+ strings)
├── agent/
│   ├── agent.py        # Agentic loop: parse tool calls, execute, loop
│   ├── tools.py        # read_clipboard, read_file, read_dir, recent_commands, read_screen, capture_region, read_window
│   └── prompts.py      # System prompt, language detection, morphological Turkish analysis
├── llm/
│   ├── backend.py      # Ollama HTTP API client
│   └── tokenizer.py    # Token counting (for context window awareness)
└── ui/
    ├── theme.py        # Design tokens (colours, radii, type) and the global QSS
    ├── icons.py        # Vector line icons rendered from inline SVG
    ├── richtext.py     # Safe rendering: plain-text labels, HTML-free Markdown, link checks
    ├── dialogs.py      # Shared frameless dialogs, shortcuts sheet, toasts
    ├── effects.py      # AuroraCard: glass background (animates only while focused)
    ├── quick_ask.py    # Clipboard mode, one-click actions, pill-shaped input field
    ├── chat_view.py    # Message bubbles, 60 FPS token reveal queue, auto-scroll
    ├── input_bar.py    # Main question input, auto-sizing
    ├── approval.py     # Approval dialog for sensitive tools
    └── worker.py       # Qt Worker thread, LLM/agent execution
```

### Data Flow

1. **User types** → `input_bar` captures text
2. **Agent loop** → `prompts.py` assembles system + user message
3. **LLM call** → `backend.py` streams tokens from Ollama
4. **Token reveal** → `chat_view._reveal_timer` drains queue at 60 FPS (24 chars/frame max)
5. **Tool calls** → Agent detects tool use, executes (e.g., read clipboard), loops
6. **Final response** → Rendered as markdown with syntax highlighting

## Turkish Language

The app detects whether you're typing Turkish based on agglutinative morpheme endings: `-yor`, `-ebilir`, `-ıyor`, `-ısı`, `-mak`, `-lik`, etc. (40+ suffixes in the detection heuristic). If detected, responses come in Turkish automatically.

If auto-detection fails, switch the language in **Settings** → **Language**.

### Morphological Suffix Matching

Because Turkish speakers often type without diacritics (e.g., "gorebiliyor" not "görebiliyor"), the system analyzes the morphological structure of words rather than relying on dictionary lookups. This works offline and handles novel word forms.

## Technology

- **PySide6**: Modern Qt 6 bindings for Python
- **Ollama**: Open-source LLM server with quantized model support (CPU-friendly)
- **QPropertyAnimation**: Smooth 60 FPS animations (token reveal, window resize, scroll easing)
- **Tokenizer**: BPE-style token counting to stay within Ollama's context window
- **OCR** (optional): Tesseract integration for `read_screen` and `capture_region` tools (requires `tesseract` package)

## Performance

- **Startup**: ~500ms (Qt + Ollama connection check)
- **Token streaming**: 60 FPS, 24 chars/frame (prevents visible text lumps)
- **Memory**: ~50–100 MB (UI + agent state); LLM runs in Ollama process
- **Latency to first token**: Depends on model size and CPU; typical: 1–3s for 7B models

## Tools (Agentic Capabilities)

The agent can call these tools *with your approval*:

| Tool | Purpose | Requires Approval |
|------|---------|---|
| `read_clipboard` | Get clipboard text | No |
| `read_file(path)` | Read a file (text only) | No |
| `read_dir(path)` | List directory contents | No |
| `recent_commands(count)` | Read shell history (zsh/bash/fish) | **Yes** |
| `read_screen()` | Capture + OCR the screen (requires tesseract) | **Yes** |
| `capture_region()` | Select a region to capture + OCR | **Yes** |
| `read_window(name, lang)` | Focus a named window, capture + OCR it | **Yes** |

Sensitive tools require user approval before execution. You can disable approval gating in **Settings** → **Tool Approval** if you trust the LLM.

## Configuration

Settings are stored in `~/.config/maze-ai/` (XDG-compliant):

```
~/.config/maze-ai/
├── config.json         # Model, hotkey, language, animation speeds
```

Chats, reminders and undo backups live in `~/.local/share/maze-ai/`; pasted images in `~/.cache/maze-ai/pasted/`. All of it is created owner-only (directories 0700, files 0600), and files left world-readable by older versions are tightened on first start.

## Development

### Running Tests

```bash
pytest -v
```

All 415 tests pass (UI behavior, streaming, language detection, tool execution).

### Linting

```bash
ruff check maze_ai tests
```

### Building the Package

```bash
./build-package.sh
```

Creates `dist-pkg/maze-ai-*.pkg.tar.zst` ready for `pacman -U`.

### Key Design Decisions

1. **Monochrome theme**: No color = no distraction; focus is on content.
2. **60 FPS token reveal**: Using a queue + proportional drain prevents the "word explosion" effect where all tokens arrive at once visually.
3. **No external API calls**: Everything runs offline. Network traffic is only to Ollama on localhost.
4. **Frame-perfect animations**: Qt's `QPropertyAnimation` with `EasingCurve.OutQuad` for natural motion.
5. **Turkish morphology over dictionaries**: Suffix-based detection works for typos, neologisms, and offline-only environments.

## Troubleshooting

**"Connection refused to localhost:11434"**
- Start Ollama: `ollama serve`
- Check it's running: `curl http://localhost:11434`

**"Model not found"**
- List available models: `ollama list`
- Pull a model: `ollama pull mistral`

**"Screen capture shows black window"**
- Requires X11 or Wayland with XWayland. Wayland-native capture coming soon.
- Tesseract OCR must be installed: `sudo pacman -S tesseract` or `brew install tesseract`

**"Turkish detection not working"**
- Switch manually in **Settings** → **Language** → **Turkish**
- The heuristic works for real Turkish; heavy code-switching or English names may confuse it

**"Response is too slow"**
- Use a smaller model: `ollama pull orca-mini` (3B)
- More VRAM or a GPU accelerator helps; most 7B models run on CPU in 2–5s/token

## Contributing

Bug reports, feature requests, and pull requests are welcome. Please open an issue first to discuss major changes.

## License

GPL-3.0 or later. See [LICENSE](LICENSE) for details.

## Author

[Berk Küçük](https://github.com/berk-kucuk) — built with a focus on Turkish users and offline-first AI workflows.

---

**Get started**: `ollama pull mistral && maze-ai` 🚀
