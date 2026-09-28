"""Tests for the ollama.com library parser behind the model download picker."""

from __future__ import annotations

from maze_ai.llm import library

SEARCH = """
<ul>
<li  class="flex items-baseline border-b border-neutral-200 py-6">
  <a href="/library/qwen3.6" class="group w-full">
    <div class="flex flex-col mb-1" title="qwen3.6">
      <h2><span >qwen3.6</span></h2>
      <p class="max-w-lg break-words text-neutral-800 text-md">Qwen3.6 &amp; agentic coding.</p>
    </div>
    <span  class="inline-flex my-1 items-center rounded-md bg-indigo-50 px-2 text-indigo-600">vision</span>
    <span  class="inline-flex my-1 items-center rounded-md bg-indigo-50 px-2 text-indigo-600">tools</span>
    <span  class="inline-flex my-1 items-center rounded-md bg-[#ddf4ff] px-2 text-blue-600">27b</span>
    <span  class="inline-flex my-1 items-center rounded-md bg-[#ddf4ff] px-2 text-blue-600">35b</span>
    <p class="my-1 flex">
      <span class="flex items-center"><svg></svg><span >6.8M</span>
        <span class="hidden sm:flex">&nbsp;Pulls</span></span>
      <span class="flex items-center"><svg></svg><span >35</span>
        <span class="hidden sm:flex">&nbsp;Tags</span></span>
      <span class="flex items-center" title="x"><svg></svg>
        <span class="hidden sm:flex">Updated&nbsp;</span>
        <span >3 weeks ago</span></span>
    </p>
  </a>
</li>
<li  class="flex items-baseline border-b border-neutral-200 py-6">
  <a href="/library/glm-5.3" class="group w-full">
    <p class="max-w-lg break-words">Cloud flagship.</p>
    <span  class="inline-flex my-1 items-center rounded-md bg-indigo-50 px-2">tools</span>
    <span  class="inline-flex my-1 items-center rounded-md bg-cyan-50 px-2">cloud</span>
  </a>
</li>
</ul>
"""

TAGS = """
<div class="group px-4 py-3">
  <a href="/library/qwen3.6:27b" class="md:hidden flex flex-col space-y-[6px] group">
    <span class="group-hover:underline">qwen3.6:27b</span>
    <div class="flex flex-col text-neutral-500 text-[13px]">
      <span><span class="font-mono">9d5803d493a9</span> • 18GB • 256K context window  •
      <span class="hidden sm:inline">Text, Image input • 1 month ago</span></span>
    </div>
  </a>
</div>
<div class="group px-4 py-3">
  <a href="/library/qwen3.6:cloud" class="md:hidden flex flex-col">
    <div class="flex flex-col text-neutral-500 text-[13px]">
      <span><span class="font-mono">72434d5a621f</span> • Medium Usage • 1M context window •
      Text input • 2 weeks ago</span>
    </div>
  </a>
</div>
"""


def test_parse_search_reads_every_field():
    models = library.parse_search(SEARCH)
    assert [m.name for m in models] == ["qwen3.6", "glm-5.3"]
    qwen = models[0]
    assert qwen.description == "Qwen3.6 & agentic coding."
    assert qwen.capabilities == ["vision", "tools"]
    assert qwen.sizes == ["27b", "35b"]
    assert qwen.pulls == "6.8M"
    assert qwen.tag_count == 35
    assert qwen.updated == "3 weeks ago"
    assert not qwen.cloud_only


def test_cloud_only_models_are_recognised():
    glm = library.parse_search(SEARCH)[1]
    assert glm.cloud_only


def test_parse_tags_reads_size_and_context():
    tags = library.parse_tags(TAGS, "qwen3.6")
    assert [t.tag for t in tags] == ["qwen3.6:27b", "qwen3.6:cloud"]
    local, cloud = tags
    assert local.size_bytes == 18_000_000_000
    assert local.size_label == "18GB"
    assert local.context == "256K"
    assert local.inputs == "Text, Image input"
    assert not local.cloud
    assert cloud.cloud and cloud.size_bytes == 0


def test_parse_size():
    assert library.parse_size("523MB") == 523_000_000
    assert library.parse_size("5.2GB") == 5_200_000_000
    assert library.parse_size("Medium Usage") == 0


def test_garbage_parses_to_nothing():
    assert library.parse_search("<html>maintenance</html>") == []
    assert library.parse_tags("<html></html>", "x") == []


def test_fallback_filters_like_a_search():
    assert library.fallback()
    assert all("vision" in m.capabilities for m in library.fallback(capability="vision"))
    assert [m.name for m in library.fallback("gemma 4")] == ["gemma4"]


def test_search_falls_through_to_the_request(monkeypatch):
    seen = {}

    class Resp:
        text = SEARCH

        def raise_for_status(self):
            pass

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers)
        return Resp()

    monkeypatch.setattr(library.requests, "get", fake_get)
    models = library.search("qwen", order=library.ORDER_NEWEST, capability="tools", page=2)
    assert len(models) == 2
    assert seen["params"] == {"q": "qwen", "o": "newest", "c": "tools", "page": 2}
    assert seen["headers"]["HX-Request"] == "true"
