# Maze AI

**Agentic AI assistant for Maze Linux** — a native desktop application that brings large language models to your terminal, without cloud dependencies. Uses local models via [Ollama](https://ollama.ai), runs fully offline, and speaks Turkish out of the box.

<div align="center">

![Version](https://img.shields.io/badge/version-1.23.0-blue)
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
- **Fits your GPU**: the context window is sized from each model's real KV-cache cost (read from Ollama, hybrid and sliding-window models included) and your free VRAM, then kept fixed so the model isn't reloaded whenever another app takes some VRAM.
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

The package is built from this working tree by `build-package.sh`. It installs any missing build tools itself, and `--install` installs the result together with every dependency:

```bash
git clone https://github.com/berk-kucuk/Maze-AI.git
cd Maze-AI
./build-package.sh --install
```

Everything Maze AI uses comes with the package — Ollama, OCR (Tesseract with English and Turkish data), the offline Arch Wiki, screenshot and clipboard tools, and the linters that check the code it writes. Only the GPU backend for Ollama depends on your hardware; the installer tells you which one to add (`ollama-cuda` for NVIDIA, `ollama-rocm` or `ollama-vulkan` for AMD).

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
| Keyboard shortcuts | `Ctrl+/`, `F1` | Copy answer / continue in chat | `Ctrl+Shift+C` / `Ctrl+Shift+Enter` |
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
- Pick the assistant's **personality** — Balanced, Concise, Detailed teacher, Friendly, Professional or Witty hacker — and its **creativity** (Precise / Balanced / Creative)
- Keep answers emoji-free: emoji are swapped for text emoticons like `:)` `:P` `:/` (on by default, even while an answer streams)

## Architecture

```
maze_ai/
├── app.py              # Entry point, single-instance socket, tray wiring
├── cli.py              # Headless --fix / --ask-cli / --shell-init
├── config.py           # Settings (~/.config/maze-ai/config.json, 0600)
├── style.py            # Personality presets, creativity, emoji → emoticon filter
├── history.py          # Saved chats; memory.py — notes it keeps about you
├── i18n.py             # Turkish / English interface strings
├── agent/
│   ├── agent.py        # Agent loop: native tool calling or JSON protocol, approvals
│   ├── tools.py        # Every tool, its schema and argument coercion
│   ├── safety.py       # Read-only / dangerous / secret / exfiltration checks
│   └── prompts.py      # System prompt assembly, language detection
├── llm/
│   ├── ollama_backend.py   # Ollama: capabilities, context sizing, keep-alive, pulls
│   ├── hardware.py         # GPU/VRAM detection, KV-cache and fit estimates
│   ├── gemini_backend.py   # Google Gemini REST
│   ├── openai_backend.py   # Any OpenAI-compatible /v1 endpoint
│   └── library.py          # ollama.com model browser
└── ui/                 # PySide6 windows: main_window, quick_ask, settings_dialog,
                        # chat_view, approval, sidebar, tray, …
```

### Data Flow

1. **User types** → the composer hands the text (and any images) to an `AgentWorker` thread
2. **System prompt** → `prompts.py` describes only the enabled tools, the personality and the reply language
3. **LLM call** → the backend streams tokens; Ollama uses native tool calling when the model supports it, schema-constrained JSON otherwise
4. **Tool calls** → arguments are type-checked, relative paths anchored at the session directory, sensitive actions confirmed, results fenced as untrusted data
5. **Final answer** → emoji are swapped for emoticons and the Markdown is rendered without HTML

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

Tools are grouped, and each group can be switched off in **Settings → Behaviour & safety** (fewer tools = less context used on small local models).

| Group | Tools |
|------|------|
| Shell | `run_command`, `launch_app`, `recent_commands` |
| Files | `read_file`, `write_file`, `edit_file`, `append_file`, `list_dir`, `search_files`, `create_dir`, `delete_path`, `move_path`, `copy_path`, `undo_file_change` |
| Web | `web_search`, `fetch_url` |
| Docs | `arch_wiki`, `arch_news`, `man_page`, `python_doc` |
| Desktop | `screenshot`, `read_screen`, `read_window`, `ocr_image`, `clipboard_copy`, `notify` |
| Memory | `remember`, `forget` |

**Checked answers.** For questions about running or fixing the system (pacman, systemd, drivers, audio, Bluetooth…), Maze AI looks the topic up in the Arch Wiki *before* the model answers and hands it the relevant section, so commands come from the wiki rather than the model's memory. It uses the offline copy from `arch-wiki-docs` when installed, wiki.archlinux.org otherwise. `man_page` reads the manual of the program actually installed, focused on the option in question (`-Qdt` is looked up as `-Q`, `-d`, `-t`).

**Memory.** Ask "remember that I use the fish shell" and Maze AI keeps a short note, added to every chat as background (Settings → Personality, where notes can be edited). Notes stay on this computer, secrets are refused, and a note the model wants to save right after reading a web page or file needs your approval. Long chats keep their thread: messages that no longer fit the context window are condensed into a summary saved with the chat.

**Arch news before upgrades.** When you talk about updating the system, Maze AI reads the Arch news feed, compares it with your last full upgrade in `/var/log/pacman.log`, and warns you first about anything that needs manual intervention — with the steps exactly as the news gives them.

**Know what you approve.** The approval dialog explains a command part by part from the manual pages installed on your machine (`pacman -Rns` → remove · ignore backup files · remove unneeded dependencies), never from the model, and updates as you edit the command.

**Pictures with any model.** If the current model can't see images, an installed model that can (one that fits your GPU, preferably) answers that one message, and the next turn is back on your model.

**Chat with a folder.** Attach a project or documents folder with the folder button: Maze AI indexes its text files locally (SQLite full-text search; an Ollama embedding model such as `embeddinggemma` adds meaning-based search), finds the relevant passages for every question and answers with `path:line` citations. Secrets, binaries and `node_modules`-style folders are never indexed; the index stays owner-only in `~/.cache/maze-ai/folders/`, and only changed files are re-read.

**Writes code, never runs it.** Maze AI can write and edit code, but code a model wrote can do anything once it runs — so running it is always your call. Inline code (`python -c`, `bash -c`, `eval`, piping into an interpreter), any file Maze AI wrote, downloaded or you applied from an answer, and tests or builds after it changed code are blocked in every mode, even after an approval. Instead, every save is checked *statically*: a file that doesn't parse is refused, and installed linters (`ruff`, `shellcheck`, `bash -n`, `node --check`, `gcc -fsyntax-only`) report problems back so the model fixes them. `python_doc` reads the real documentation of installed libraries (in an isolated interpreter that can't import your project), a project map (file outline, conventions, `git status` with repository hooks disabled) keeps changes in the project's style, coding questions go to an installed coding model such as `qwen2.5-coder`, and code blocks in answers have an **Apply to file** button that shows the diff first.

**The right model for your hardware.** Settings recommends the biggest tool-calling model that stays entirely in your VRAM (installed ones first), and the status bar offers a one-click switch when the current model spills onto the CPU.

Every shell command (and every app `launch_app` would start) goes through a three-tier rule set (`maze_ai/agent/rules.py`, listed in **Settings → Command rules**):

- **Never run**, in any mode, even after an approval: gaining root (`sudo`, `su`, `doas`, `pkexec`), deleting or moving `/` or the home folder, formatting or overwriting disks, piping a download into a shell, reverse shells, killing every process, fork bombs, recursive permission changes on `/` or home.
- **Always ask**, even in Autonomous mode, and never remembered: recursive or forced deletes, discarding git work (`reset --hard`, `push --force`…), killing processes, stopping services, powering off, uninstalling software.
- **Run without asking** in Ask mode: read-only inspections such as `ls`, `cd`, `cat`, `grep`, `find`, `git status`, `pacman -Q…`, `systemctl status`, `journalctl`.

You can add your own blocked and safe commands (a command prefix like `git push`, or `re:` plus a regular expression); your safe entries never outrank a built-in block or confirmation. Reading keys, tokens or shell history, and requests that carry data out of the machine, are always confirmed. Overwritten or deleted files are backed up and can be restored with `undo_file_change`.


## Configuration

Settings are stored in `~/.config/maze-ai/` (XDG-compliant):

```
~/.config/maze-ai/
├── config.json         # Model, hotkey, language, animation speeds
```

Chats, memory notes and undo backups live in `~/.local/share/maze-ai/`; pasted images in `~/.cache/maze-ai/pasted/`. All of it is created owner-only (directories 0700, files 0600), and files left world-readable by older versions are tightened on first start.

## Development

### Running Tests

```bash
pytest -v
```

All 893 tests pass (UI behavior, streaming, language detection, tool execution). Tests that talk to a real Ollama server run with `MAZE_AI_LIVE=1` (optionally `MAZE_AI_LIVE_MODEL=<name>`).

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

**Get started**: `ollama pull mistral && maze-ai`
