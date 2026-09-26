"""Safe text rendering for anything the model, a tool or a web page wrote.

Qt labels are quietly dangerous with untrusted text:

* A ``QLabel`` in its default ``AutoText`` mode renders anything that *looks*
  like HTML as HTML. A command such as ``echo <span style="display:none">…``
  could make the approval dialog show one thing while another runs.
* Markdown in Qt accepts raw HTML blocks too, and ``setOpenExternalLinks``
  hands every clicked link — ``file://``, ``smb://``, a desktop-file handler —
  straight to the desktop, where the visible link text need not match the
  target at all.

So: labels holding outside text are plain text; Markdown is rendered with raw
HTML switched off and images reduced to their alt text; and links go through
:func:`open_link`, which allows only web and mail links and shows the user
the real address before anything opens.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase, QTextDocument
from PySide6.QtWidgets import QLabel, QWidget

from .theme import FONT_MONO, LINK, PANEL_HI

#: Link schemes the chat may open (after confirmation). Everything else —
#: file:, smb:, ftp:, data:, javascript:, custom handlers — is refused.
SAFE_SCHEMES = frozenset({"http", "https", "mailto"})

_IMG_RE = re.compile(r"<img\b[^>]*?>", re.IGNORECASE)
_ALT_RE = re.compile(r'\balt="([^"]*)"', re.IGNORECASE)
_BODY_RE = re.compile(r"<body[^>]*>", re.IGNORECASE)
_ANCHOR_SPAN_RE = re.compile(r'(<a href="[^"]*">)<span style=" color:#[0-9a-fA-F]{6};">')

#: Qt's Markdown headings are sized for documents, not chat messages.
_HEADING_SIZES = (
    ("font-size:xx-large;", "font-size:15pt;"),
    ("font-size:x-large;", "font-size:13.5pt;"),
    ("font-size:large;", "font-size:12pt;"),
)


def harden_labels(root: QWidget) -> None:
    """Switch every auto-detecting label under ``root`` to plain text.

    Labels that deliberately render Markdown or HTML set their format
    explicitly and are left alone; ``AutoText`` is only ever the default, and
    the default is what lets outside text become markup.
    """
    for label in root.findChildren(QLabel):
        if label.textFormat() == Qt.TextFormat.AutoText:
            label.setTextFormat(Qt.TextFormat.PlainText)


def plain_label(text: str = "", parent: QWidget | None = None) -> QLabel:
    """A label that can never interpret its text as markup."""
    label = QLabel(parent)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setText(text)
    return label


def plain_tooltip(text: str) -> str:
    """Tool tips detect rich text too; this one always shows the literal text."""
    if not text:
        return ""
    escaped = html.escape(text).replace("\n", "<br>")
    return f"<qt>{escaped}</qt>"


@lru_cache(maxsize=1)
def _fixed_family() -> str:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()


def _strip_image(match: re.Match) -> str:
    alt = _ALT_RE.search(match.group(0))
    label = alt.group(1) if alt else ""
    # The alt text is already escaped inside the generated HTML.
    return f"[{label}]" if label else ""


def markdown_to_html(text: str) -> str:
    """Render Markdown to Qt rich text with raw HTML and images disabled.

    Anything that looks like a tag in the source is escaped and shown as-is;
    ``![alt](url)`` becomes ``[alt]`` so no resource is ever loaded from a
    path the model chose.
    """
    doc = QTextDocument()
    features = QTextDocument.MarkdownFeature(
        QTextDocument.MarkdownFeature.MarkdownDialectGitHub.value
        | QTextDocument.MarkdownFeature.MarkdownNoHTML.value
    )
    doc.setMarkdown(text or "", features)
    out = doc.toHtml()
    out = _IMG_RE.sub(_strip_image, out)
    # Let the label's own font and colour apply rather than the document's.
    out = _BODY_RE.sub("<body>", out, count=1)
    out = _ANCHOR_SPAN_RE.sub(
        lambda m: m.group(1) + f'<span style=" color:{LINK}; text-decoration:none;">', out
    )
    fixed = _fixed_family()
    if fixed:
        out = out.replace(
            f"font-family:'{fixed}';",
            f"font-family:{FONT_MONO}; background-color:{PANEL_HI};",
        )
    for old, new in _HEADING_SIZES:
        out = out.replace(old, new)
    return out


def is_safe_link(url: str) -> bool:
    parsed = QUrl(url or "")
    return parsed.isValid() and parsed.scheme().lower() in SAFE_SCHEMES and bool(
        parsed.host() or parsed.scheme().lower() == "mailto"
    )


def open_link(url: str, parent: QWidget | None = None) -> bool:
    """Open a link from untrusted text — only after the user sees where it goes.

    Returns True if the link was opened.
    """
    from ..i18n import tr
    from .dialogs import ConfirmDialog, notify_toast

    if not is_safe_link(url):
        notify_toast(parent, tr("Blocked a link that isn't a web address."), kind="danger")
        return False
    target = QUrl(url)
    shown = target.toDisplayString()
    dialog = ConfirmDialog(
        tr("Open this link?"),
        tr("It will open in your browser. Make sure you trust the address:"),
        detail=shown,
        confirm=tr("Open link"),
        icon_name="link",
        parent=parent,
    )
    if not dialog.exec():
        return False
    return QDesktopServices.openUrl(target)


def wire_links(label: QLabel) -> None:
    """Route a label's link clicks through :func:`open_link` and show targets."""
    label.setOpenExternalLinks(False)
    label.linkActivated.connect(lambda url, lab=label: open_link(url, lab.window()))
    label.linkHovered.connect(
        lambda url, lab=label: lab.setToolTip(plain_tooltip(url) if url else "")
    )
