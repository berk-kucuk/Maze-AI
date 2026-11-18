"""Small hardening (audit MA5–MA7): every request has a timeout, outside XML
cannot declare entities, the folder-index id is not a security hash."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from maze_ai import xmlsafe
from maze_ai.agent import news
from maze_ai.llm import http


def test_a_request_without_timeout_gets_one(monkeypatch):
    seen = {}

    class R:
        status_code = 200

    def fake(method, url, **kw):
        seen.update(kw)
        return R()
    monkeypatch.setattr(http.requests, "request", fake)
    http.request_with_retry("GET", "http://example.invalid/")
    assert seen["timeout"] == http.DEFAULT_TIMEOUT
    http.request_with_retry("GET", "http://example.invalid/", timeout=5)
    assert seen["timeout"] == 5


LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>
<rss><channel><item><title>&lol2;</title></item></channel></rss>"""

EXTERNAL = """<?xml version="1.0"?>
<!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><rss>&x;</rss>"""

PLAIN = """<?xml version="1.0"?>
<!DOCTYPE rss PUBLIC "-//Netscape Communications//DTD RSS 0.91//EN" "http://my.netscape.com/publish/formats/rss-0.91.dtd">
<rss version="0.91"><channel><title>t</title>
<item><title>Arch news</title><link>https://archlinux.org/news/x/</link><pubDate>Mon, 10 Nov 2026 12:00:00 +0000</pubDate></item></channel></rss>"""


@pytest.mark.parametrize("doc", [LAUGHS, EXTERNAL, LAUGHS.replace("ENTITY", "entity")])
def test_entity_declarations_are_refused(doc):
    with pytest.raises(ET.ParseError):
        xmlsafe.fromstring(doc)
    with pytest.raises(ET.ParseError):
        news.parse_feed(doc)


def test_ordinary_feeds_still_parse():
    assert xmlsafe.fromstring(PLAIN).tag == "rss"
    items = news.parse_feed(PLAIN)
    assert [i.title for i in items] == ["Arch news"]


def test_parse_file(tmp_path):
    p = tmp_path / "uca.xml"
    p.write_text("<actions><action><name>x</name></action></actions>")
    assert xmlsafe.parse(p).getroot().tag == "actions"
    p.write_text(EXTERNAL)
    with pytest.raises(ET.ParseError):
        xmlsafe.parse(p)
