"""How the assistant talks: personality presets, creativity and emoji policy.

The personality is a short block of the system prompt, chosen in Settings, so
the user can make the assistant terse, chatty or formal without writing custom
instructions. Creativity maps to the sampling temperature.

Emoji are replaced by plain-text emoticons (``:)``, ``:P``, ``:/``). Local
models scatter pictographs through their answers whatever the prompt says, and
in the monochrome UI they render as loud colour blobs or tofu boxes. So the
prompt asks for emoticons, and :class:`EmojiFilter` cleans whatever slips
through — chunk by chunk, so it works on a live stream as well.
"""

from __future__ import annotations

# ── personality ──────────────────────────────────────────────────────────
PERSONA_BALANCED = "balanced"

#: id -> (settings label, short description, prompt block)
PERSONAS: dict[str, tuple[str, str, str]] = {
    "balanced": (
        "Balanced",
        "Friendly and to the point. The default.",
        "Be friendly, clear and to the point. Give enough detail to be useful, "
        "and no filler.",
    ),
    "concise": (
        "Concise",
        "Short, direct answers. Commands and results first, little prose.",
        "Be extremely concise. Lead with the answer, command or result; use as "
        "few words as possible. No introductions, no recaps, no small talk. "
        "Expand only when the user asks for more.",
    ),
    "detailed": (
        "Detailed teacher",
        "Explains the why behind every step. Good for learning.",
        "Be thorough and educational. Explain the reasoning behind each step, "
        "what every command and flag does, and what to watch out for. Use "
        "short examples. Stay structured so long answers are easy to scan.",
    ),
    "friendly": (
        "Friendly",
        "Warm and casual, with a light sense of humour.",
        "Be warm, casual and encouraging, like a helpful friend who knows "
        "Linux well. A little light humour is welcome, but never at the "
        "expense of correctness or getting the task done.",
    ),
    "professional": (
        "Professional",
        "Formal, neutral and precise. No jokes, no emoticons.",
        "Be formal, neutral and precise, like a senior engineer writing for a "
        "colleague. No jokes, no slang, no emoticons.",
    ),
    "hacker": (
        "Witty hacker",
        "Dry, nerdy humour with terminal-veteran attitude.",
        "Talk like a sharp, slightly sarcastic terminal veteran: dry wit, nerdy "
        "references, confident opinions. Keep the jokes short and never let "
        "the attitude get in the way of a correct, complete answer.",
    ),
}

#: Personas that should not use emoticons at all.
_NO_EMOTICONS = {"professional", "concise"}


def persona_ids() -> list[str]:
    return list(PERSONAS)


def persona_block(persona: str, no_emoji: bool = True) -> str:
    """The system-prompt section describing tone and emoji policy."""
    persona = persona if persona in PERSONAS else PERSONA_BALANCED
    lines = ["# Personality and tone", PERSONAS[persona][2]]
    if no_emoji:
        if persona in _NO_EMOTICONS:
            lines.append(
                "Never use emoji or emoticons. Write plain text only."
            )
        else:
            lines.append(
                "Never use emoji (pictographs such as smileys, rockets, check "
                "marks, folders, hearts). To show emotion, use plain-text "
                "emoticons instead, e.g. :) :D ;) :P :/ :( ^^ xD — sparingly, "
                "at most one or two per answer, and never inside code, "
                "commands or file contents."
            )
    return "\n".join(lines)


# ── creativity → temperature ─────────────────────────────────────────────
CREATIVITY: dict[str, tuple[str, float]] = {
    "precise": ("Precise  ·  predictable, best for commands", 0.15),
    "balanced": ("Balanced", 0.4),
    "creative": ("Creative  ·  more varied wording and ideas", 0.8),
}


def temperature_for(creativity: str | None) -> float:
    return CREATIVITY.get(creativity or "", CREATIVITY["balanced"])[1]


# ── emoji → emoticon filter ──────────────────────────────────────────────
#: Emoji that have a natural text emoticon. Everything else is dropped.
EMOTICONS: dict[str, str] = {
    "\U0001F600": ":D", "\U0001F603": ":D", "\U0001F604": ":D", "\U0001F601": ":D",
    "\U0001F606": "xD", "\U0001F602": "xD", "\U0001F923": "xD",
    "\U0001F605": "^^'", "\U0001F642": ":)", "\U0001F60A": ":)", "☺": ":)",
    "\U0001F917": ":)", "\U0001F60C": ":)", "\U0001F609": ";)", "\U0001F60F": ";)",
    "\U0001F61B": ":P", "\U0001F61C": ";P", "\U0001F61D": "xP", "\U0001F92A": ":P",
    "\U0001F60B": ":P", "\U0001F615": ":/", "\U0001FAE4": ":/", "\U0001F610": ":|",
    "\U0001F611": ":|", "\U0001F636": ":|", "\U0001F641": ":(", "☹": ":(",
    "\U0001F61E": ":(", "\U0001F61F": ":(", "\U0001F614": ":(", "\U0001F622": ":'(",
    "\U0001F62D": ":'(", "\U0001F62E": ":O", "\U0001F62F": ":O", "\U0001F632": ":O",
    "\U0001F631": "D:", "\U0001F60E": "B)", "\U0001F618": ":*", "\U0001F60D": "<3",
    "\U0001F970": "<3", "❤": "<3", "♥": "<3", "\U0001F495": "<3",
    "\U0001F496": "<3", "\U0001F499": "<3", "\U0001F49A": "<3", "\U0001F49B": "<3",
    "\U0001F49C": "<3", "\U0001F5A4": "<3", "\U0001F9E1": "<3", "\U0001F90D": "<3",
    "\U0001F620": ">:(", "\U0001F621": ">:(", "\U0001F607": "O:)", "\U0001F643": "(:",
    "\U0001F62C": ":S", "\U0001F634": "-_-", "\U0001F644": "-_-", "\U0001F612": "-_-",
    "\U0001F633": "O_O", "\U0001F92F": "O_O", "\U0001F44B": "o/", "\U0001F389": ":D",
    "\U0001F973": ":D",
}

# BMP characters that render as emoji by default (Emoji_Presentation=Yes).
_BMP_EMOJI = set(
    "⌚⌛⏩⏪⏫⏬⏰⏳◽◾☔☕"
    "♿⚓⚡⚪⚫⚽⚾⛄⛅⛎⛔⛪"
    "⛲⛳⛵⛺⛽✅✊✋✨❌❎❓"
    "❔❕❗➕➖➗➰➿⬛⬜⭐⭕"
    + "".join(chr(c) for c in range(0x2648, 0x2654))      # zodiac signs
)
_VS16 = "️"      # "show the previous character as emoji"
_VS15 = "︎"
_ZWJ = "‍"
_KEYCAP = "⃣"


def _is_pictograph(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF            # emoticons, pictographs, flags, …
        or 0xE0020 <= cp <= 0xE007F         # tag sequences (subdivision flags)
        or ch in _BMP_EMOJI
    )


def _can_take_vs16(ch: str) -> bool:
    """Text symbols that turn into emoji when followed by U+FE0F."""
    cp = ord(ch)
    return 0x2000 <= cp <= 0x2BFF or ch in "©®〰〽㊗㊙"


class EmojiFilter:
    """Streams text through, swapping emoji for emoticons or dropping them.

    Feed chunks as they arrive and call :meth:`flush` at the end. A symbol
    that becomes an emoji only when the *next* character is U+FE0F (⚠ + FE0F
    is ⚠️) is held back one chunk, so a split across chunks is handled the
    same as a whole string.
    """

    def __init__(self) -> None:
        self._held = ""           # one text symbol waiting to see a VS16
        self._last = "\n"         # last character emitted
        self._dropped = False     # the previous character was a removed emoji
        self._eat_space = False   # skip one space left behind by a removal

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in (text or ""):
            self._step(ch, out)
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        if self._held:
            held, self._held = self._held, ""
            self._emit(held, out)
        return "".join(out)

    # ── internals ────────────────────────────────────────────────────────
    def _emit(self, text: str, out: list[str]) -> None:
        if not text:
            return
        if self._eat_space and text[0] == " ":
            text = text[1:]
        self._eat_space = False
        if text:
            out.append(text)
            self._last = text[-1]
        self._dropped = False

    def _replace(self, ch: str, out: list[str]) -> None:
        """An emoji: write its emoticon, or remove it cleanly."""
        face = EMOTICONS.get(ch)
        if face:
            # Keep words and emoticons apart: "done:)" reads badly.
            prefix = "" if self._last in " \n\t([" else " "
            self._emit(prefix + face, out)
            self._dropped = True
            return
        self._dropped = True
        # "## 🚀 Title" and "- ✅ done" would keep a double space otherwise.
        self._eat_space = self._last in " \n\t"

    def _step(self, ch: str, out: list[str]) -> None:
        if self._held:
            held, self._held = self._held, ""
            if ch == _VS16:
                self._replace(held, out)
                return
            self._emit(held, out)
        if ch in (_VS16, _VS15, _KEYCAP):
            return                       # orphan presentation selector
        if ch == _ZWJ:
            if self._dropped:
                return                   # joins two pictographs: drop it
            self._emit(ch, out)          # legitimate in Indic/Arabic text
            return
        if _is_pictograph(ch) or ch in EMOTICONS:
            if 0x1F3FB <= ord(ch) <= 0x1F3FF and self._dropped:
                return                   # skin tone of an emoji just handled
            if self._dropped and ch not in EMOTICONS:
                return                   # rest of a ZWJ/flag sequence
            self._replace(ch, out)
            return
        if _can_take_vs16(ch):
            self._held = ch
            return
        if self._eat_space and ch == " ":
            self._eat_space = False
            return
        self._emit(ch, out)


def strip_emoji(text: str) -> str:
    """Whole-string version of :class:`EmojiFilter`."""
    flt = EmojiFilter()
    return flt.feed(text) + flt.flush()
