"""Deciding when to look something up before the model answers.

Telling a small model "check the Arch Wiki first" is not enough — in testing it
answered "how do I remove orphans?" from memory with the wrong command
(``pacman -Sc``, which cleans the cache). So for questions that are clearly
about running or fixing an Arch system, the agent fetches the relevant wiki
section itself and hands it to the model as reference.

The detection is deliberately conservative: a question shape (how / why / error
/ install / remove…) *and* a system topic (pacman, systemd, Bluetooth, drivers…),
in English or Turkish. The wiki is English, so Turkish topic words are mapped to
English search terms.
"""

from __future__ import annotations

import re

from .prompts import _STRONG_HINTS, _TR_SUFFIXES, _WEAK_HINTS, strip_quoted

_TOPICS = {
    "pacman", "yay", "paru", "aur", "makepkg", "pkgbuild", "systemd", "systemctl",
    "journalctl", "grub", "mkinitcpio", "initramfs", "nvidia", "amdgpu", "mesa",
    "vulkan", "wayland", "xorg", "x11", "kde", "plasma", "kwin", "gnome", "sddm",
    "bluetooth", "bluez", "pipewire", "pulseaudio", "wireplumber", "alsa",
    "networkmanager", "nmcli", "wifi", "iwd", "ufw", "nftables", "iptables",
    "firewall", "btrfs", "snapper", "ext4", "fstab", "swap", "zram", "hibernate",
    "suspend", "locale", "ssh", "sshd", "flatpak", "docker", "podman", "cups",
    "printer", "fonts", "font", "mirrorlist", "reflector", "kernel", "dkms",
    "sbctl", "luks", "cryptsetup", "tlp", "orphan", "orphans", "package",
    "packages", "driver", "drivers", "boot", "bootloader", "audio", "microphone",
    "headset", "keyring", "gpg", "timer", "cron", "service", "daemon", "dns",
    "vpn", "wireguard", "openvpn", "steam", "proton", "wine", "gamemode",
    "secure", "partition", "mount", "encryption", "resolution", "monitor",
    "touchpad", "keyboard", "screen", "tearing", "freeze", "crash",
}

# Turkish word starts → English search terms. Matched at the start of a word,
# so "paketleri" hits "paket" and "temizlerim" hits "temizle".
_TR_TERMS = {
    "paket": "package", "yetim": "orphan", "temizle": "remove", "kaldır": "remove",
    "kaldir": "remove", "silme": "remove", "silin": "remove", "sil": "remove",
    "kurulum": "install", "kurul": "install", "kurmak": "install", "kur": "install",
    "yükle": "install", "yukle": "install", "güncelle": "update", "guncelle": "update",
    "önbellek": "cache", "onbellek": "cache", "sürücü": "driver", "surucu": "driver",
    "ekran kart": "graphics driver", "ses": "audio", "mikrofon": "microphone",
    "kulaklık": "headset", "kulaklik": "headset", "kulaklı": "headset", "kulakl": "headset",
    "hoparlör": "speaker", "hoparlor": "speaker", "kablosuz": "wireless",
    "açılış": "boot", "acilis": "boot", "önyükle": "boot", "onyukle": "boot",
    "uyku": "suspend", "askıya": "suspend", "yazıcı": "printer", "yazici": "printer",
    "yazı tip": "fonts", "servis": "service", "hizmet": "service",
    "güvenlik duvar": "firewall", "bölüm": "partition", "şifrele": "encryption",
    "takas": "swap", "çekirdek": "kernel", "cekirdek": "kernel", "klavye": "keyboard",
    "çözünürlük": "resolution", "titre": "tearing", "donuyor": "freeze",
    "çöküyor": "crash", "zamanlanmış": "timer", "bağlan": "connect",
}
_TR_EXACT = {"ağ": "network", "ağı": "network", "ağa": "network", "wifi": "wifi"}

_QUESTION = re.compile(
    r"\?|\b(how|why|what|which|fix|error|fail|failed|broken|install|remove|enable|"
    r"disable|configure|setup|set up|update|upgrade|clean|not working|doesn't work)\b"
    r"|\b(nasıl|nasil|neden|niye|niçin|nicin|hata|sorun|düzelt|duzelt|çalışmıyor|"
    r"calismiyor|olmuyor|açılmıyor|acilmiyor|kur|kaldır|kaldir|temizle|etkinleştir|"
    r"etkinlestir|ayarla|güncelle|guncelle|yapabilirim|yaparım|yaparim|edebilirim|"
    r"ederim|gerekiyor|kopuyor|donuyor|çöküyor|cokuyor)\w*"
    # Turkish complaints are negative verbs: "gelmiyor", "bağlanmıyor".
    r"|\w+(?:mıyor|miyor|muyor|müyor)\w*",
    re.IGNORECASE,
)
_STOP = {"the", "and", "for", "with", "how", "why", "what", "can", "does", "this", "that",
         "arch", "linux", "maze", "my", "on", "in", "to", "is", "it", "do", "use",
         "when", "from", "after", "before", "not", "working", "doesn't", "error", "get",
         "gets", "keeps", "still", "there", "have", "has", "want"}
#: Search terms that name a subject (a lookup needs at least one); verbs like
#: install/remove only sharpen a query, as in "hatırlatıcı kur" they mean nothing.
_VERBS = {"install", "remove", "update", "connect", "clean", "upgrade", "enable",
          "disable", "fix", "error", "mount"}


def lookup_query(message: str) -> str:
    """An English wiki query for this message, or "" when no lookup is due."""
    text = strip_quoted(message or "").strip()
    if not text or len(text) > 600 or not _QUESTION.search(text):
        return ""
    low = text.lower()
    words = re.findall(r"[\w'+.-]+", low)
    terms: list[str] = []

    def add(term: str) -> None:
        for part in term.split():
            if part not in terms:
                terms.append(part)

    for word in words:
        bare = word.strip("'.")
        if bare in _TOPICS or (bare.isascii() and bare.endswith(("ctl", "d")) and bare in _TOPICS):
            add(bare)
    for word in words:
        if word in _TR_EXACT:
            add(_TR_EXACT[word])
            continue
        for stem, english in _TR_TERMS.items():
            if " " in stem:
                if stem in low:
                    add(english)
            elif len(stem) <= 3:
                if word == stem or (word.startswith(stem) and len(word) <= len(stem) + 4):
                    add(english)
            elif word.startswith(stem):
                add(english)
    # English verbs and nouns that sharpen a query once a topic is known.
    for word in words:
        if word in ("install", "remove", "clean", "enable", "disable", "update", "cache",
                    "error", "fix", "orphan", "orphans", "upgrade", "mount", "suspend"):
            add(word)
    topical = [t for t in terms if t not in _VERBS and (
        t in _TOPICS or t in " ".join(_TR_TERMS.values()).split()
        or t in _TR_EXACT.values())]
    if not topical:
        return ""
    # Other English words in the question ("conflicting files") are usually
    # exactly what the wiki calls the problem.
    for word in words:
        bare = word.strip("'.:,;")
        if (bare.isascii() and bare.isalpha() and len(bare) >= 4 and bare not in _STOP
                and not _looks_turkish(bare) and len(terms) < 7):
            add(bare)
    query = " ".join(t for t in terms if t not in _STOP)
    return query[:80]


_TR_WORDS = set(_STRONG_HINTS["tr"]) | set(_WEAK_HINTS["tr"]) | {
    "neden", "niye", "yetim", "sorun", "kart", "karti", "veriyor", "surekli", "bunu",
    "bana", "beni", "artik", "hala", "acmiyor", "olmuyor", "kopuyor",
}


def _looks_turkish(word: str) -> bool:
    """Turkish typed without its letters is still ASCII: catch it by word or suffix."""
    return word in _TR_WORDS or word.endswith(_TR_SUFFIXES) or any(
        word.startswith(stem) for stem in _TR_TERMS if " " not in stem and len(stem) > 3
    )


_CODING = re.compile(
    r"\b(code|coding|function|method|class|script|refactor|debug|bug|compile|"
    r"implement|algorithm|regex|unit test|api|endpoint|python|javascript|typescript|"
    r"rust|golang|c\+\+|java|kotlin|swift|sql|bash script|shell script|html|css|react|"
    r"django|flask|fastapi|pytest|traceback|stack trace|exception|"
    r"kod\w*|fonksiyon\w*|metod\w*|sınıf\w*|betik\w*|script\w*|yazılım\w*|"
    r"programla\w*|algoritma\w*|derle\w*|hata ayıkla\w*|refaktör\w*|"
    r"test yaz\w*|uygulama yaz\w*|program yaz\w*)\b",
    re.IGNORECASE,
)
_CODE_FILES = re.compile(r"\.(py|js|ts|tsx|jsx|rs|go|c|h|cpp|hpp|java|kt|rb|php|sh|sql|lua)\b")


def is_coding_request(message: str) -> bool:
    """Is this a programming task (write, fix or explain code)?"""
    text = strip_quoted(message or "")
    return bool(_CODING.search(text) or _CODE_FILES.search(text) or "```" in (message or ""))
