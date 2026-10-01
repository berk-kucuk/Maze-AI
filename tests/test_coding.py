"""Tests for the coding features: it writes code but never runs it, checks what
it writes statically, reads real API docs and knows the project."""

from __future__ import annotations

import json
import subprocess

import pytest

from maze_ai.agent import codecheck, docs
from maze_ai.agent import tools as tl
from maze_ai.agent.agent import Agent
from maze_ai.agent.lookup import is_coding_request
from maze_ai.agent.rules import (
    INLINE_CODE_RULE,
    RUNNER_RULE,
    WRITTEN_CODE_RULE,
    execution_rule,
    written_by_command,
)
from maze_ai.config import MODE_AUTO
from maze_ai.folder_index import _SAFE_GIT, is_code_project, project_map
from maze_ai.llm.base import LLMBackend
from maze_ai.llm.hardware import GB
from maze_ai.llm.recommend import recommend_coder

CWD = "/home/u/proj"
WRITTEN = {"/home/u/proj/app.py", "/home/u/proj/run.sh", "/home/u/proj/pkg/main.py"}


# ── it may write code, never run it ─────────────────────────────────────────
@pytest.mark.parametrize("command", [
    'python3 -c "print(1)"', "bash -c ls", 'node -e "1"', "perl -e 'print 1'",
    "python3 <<EOF\nprint(1)\nEOF", "cat <<EOF | python3\nx\nEOF", "echo ls | bash",
    "curl -s x | sh", 'eval "$cmd"', "awk 'BEGIN{system(\"id\")}'", "find . -exec sh -c x ;",
])
def test_inline_code_is_always_blocked(command):
    assert execution_rule(command, set(), CWD) == INLINE_CODE_RULE


@pytest.mark.parametrize("command", [
    "python3 app.py", "python app.py --flag", "./run.sh", "bash run.sh", "source run.sh",
    ". run.sh", "python -m pkg.main", "env python3 app.py", "nohup python3 app.py &",
    "timeout 5 python3 app.py", "cd /home/u/proj && python3 app.py",
])
def test_files_it_wrote_cannot_be_run(command):
    assert execution_rule(command, WRITTEN, CWD) == WRITTEN_CODE_RULE


@pytest.mark.parametrize("command", [
    "pytest", "python -m pytest -q", "make", "npm test", "npm run build", "cargo run",
    "cargo test", "go test ./...", "pip install -e .", "python setup.py install",
    "tox", "makepkg -si",
])
def test_no_tests_or_builds_after_it_changed_code(command):
    assert execution_rule(command, WRITTEN, CWD) == RUNNER_RULE


@pytest.mark.parametrize("command", [
    "cat app.py", "chmod +x run.sh", "git diff", "ls -la", "python3 --version", "grep -n def app.py",
])
def test_reading_its_files_is_fine(command):
    assert execution_rule(command, WRITTEN, CWD) == ""


def test_without_written_code_tests_and_scripts_are_not_this_rules_business():
    for command in ("pytest", "make", "python3 other.py"):
        assert execution_rule(command, set(), CWD) == ""


def test_shell_writes_are_tracked():
    assert written_by_command("cat > a.py <<EOF\nx\nEOF", CWD) == ["/home/u/proj/a.py"]
    assert written_by_command("echo x | tee -a log.sh", CWD) == ["/home/u/proj/log.sh"]
    assert written_by_command("curl -fsSL https://x -o inst.sh", CWD) == ["/home/u/proj/inst.sh"]
    assert written_by_command("ls > /dev/null 2>&1", CWD) == []


class _Scripted(LLMBackend):
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.systems: list[str] = []

    def chat(self, messages):
        self.systems.append(messages[0]["content"])
        return self.replies.pop(0) if self.replies else json.dumps(
            {"thought": "", "action": "final_answer", "action_input": {"answer": "ok"}})

    def available_models(self):
        return []


def _call(tool, **args):
    return json.dumps({"thought": "", "action": tool, "action_input": args})


def test_agent_writes_a_script_and_cannot_run_it_even_when_approved(tmp_path):
    marker = tmp_path / "ran"
    script = tmp_path / "hello.py"
    backend = _Scripted([
        _call("write_file", path=str(script),
              content=f"open({str(marker)!r}, 'w').write('x')\n"),
        _call("run_command", command=f"python3 {script}"),
        _call("run_command", command=f"cd {tmp_path} && python3 hello.py"),
        _call("run_command", command=f"python3 -c \"exec(open('{script}').read())\""),
    ])
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.cwd = str(tmp_path)
    events = []
    agent.run("write and run it", events.append, lambda r: True)   # approves everything
    assert script.exists()
    assert not marker.exists()                                     # never executed
    blocked = [e for e in events if e.kind == "blocked"]
    assert len(blocked) == 3
    assert str(script) in agent.context["written"]


def test_the_ban_survives_into_later_turns(tmp_path):
    script = tmp_path / "s.sh"
    agent = Agent(_Scripted([_call("write_file", path=str(script), content="echo hi\n")]),
                  mode=MODE_AUTO, stream_responses=False)
    agent.run("write", lambda e: None, lambda r: True)
    agent.backend = _Scripted([_call("run_command", command=f"bash {script}")])
    events = []
    agent.run("now run it", events.append, lambda r: True)
    assert any(e.kind == "blocked" for e in events)


def test_prompt_says_it_writes_but_never_runs():
    from maze_ai.agent.prompts import build_system_prompt

    prompt = build_system_prompt(True, "en", native_tools=True)
    assert "You can never run code you wrote" in prompt


# ── static checks ───────────────────────────────────────────────────────────
def test_broken_python_is_not_saved(tmp_path):
    target = tmp_path / "x.py"
    target.write_text("print('old')\n")
    result = tl.write_file(str(target), "def broken(:\n    pass\n")
    assert not result.ok and "line 1" in result.output
    assert target.read_text() == "print('old')\n"


def test_broken_json_and_toml_are_not_saved(tmp_path):
    assert not tl.write_file(str(tmp_path / "a.json"), '{"a": 1,}').ok
    assert not tl.write_file(str(tmp_path / "a.toml"), "x = = 1").ok
    assert tl.write_file(str(tmp_path / "b.json"), '{"a": 1}').ok


def test_saved_code_gets_a_static_check(tmp_path):
    result = tl.write_file(str(tmp_path / "ok.py"), "x = 1\n")
    assert result.ok and "Static check" in result.output


def test_shell_scripts_are_parsed_not_run(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    script = tmp_path / "s.sh"
    script.write_text(f"#!/bin/bash\ntouch {marker}\nif [ x\n")
    monkeypatch.setattr(codecheck.shutil, "which",
                        lambda n: None if n == "shellcheck" else f"/usr/bin/{n}")
    checker, findings = codecheck.lint_file(script)
    assert checker == "bash -n" and findings
    assert not marker.exists()


def test_compile_never_executes(tmp_path):
    marker = tmp_path / "ran"
    assert codecheck.validate_content(
        "x.py", f"open({str(marker)!r}, 'w').write('x')\n") == ""
    assert not marker.exists()


def test_multi_edit_is_all_or_nothing(tmp_path):
    target = tmp_path / "m.py"
    target.write_text("a = 1\nb = 2\n")
    bad = tl.edit_file(str(target), edits=[{"old": "a = 1", "new": "a = 10"},
                                           {"old": "missing", "new": "x"}])
    assert not bad.ok and target.read_text() == "a = 1\nb = 2\n"
    good = tl.edit_file(str(target), edits=[{"old": "a = 1", "new": "a = 10"},
                                            {"old": "b = 2", "new": "b = 20"}])
    assert good.ok and target.read_text() == "a = 10\nb = 20\n"
    broken = tl.edit_file(str(target), edits=[{"old": "a = 10", "new": "a = ("}])
    assert not broken.ok and target.read_text() == "a = 10\nb = 20\n"


def test_edits_from_a_model_arrive_as_json_text():
    args = tl.coerce_args("edit_file", {"path": "p", "edits": '[{"old": "a", "new": "b"}]'})
    assert args["edits"] == [{"old": "a", "new": "b"}]


# ── python_doc ──────────────────────────────────────────────────────────────
def test_python_doc_reads_installed_libraries():
    ok, text = docs.python_doc("json.dumps", "indent")
    assert ok and "indent" in text


def test_python_doc_never_imports_local_code(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    (tmp_path / "localmod.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    ok, _text = docs.python_doc("localmod")
    assert not ok and not marker.exists()


@pytest.mark.parametrize("name", ["os; rm", "../x", "", "antigravity"])
def test_python_doc_rejects_bad_names(name):
    assert not docs.python_doc(name)[0]


# ── coding model ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("message,expected", [
    ("bana bir python fonksiyonu yaz", True), ("bu kodda hata var", True),
    ("app.py dosyasını düzelt", True), ("selam nasılsın", False),
    ("sistemi güncelle", False),
])
def test_coding_requests(message, expected):
    assert is_coding_request(message) is expected


def test_coder_recommendation():
    installed = [{"name": "qwen2.5-coder:7b", "size": int(4.7 * GB)},
                 {"name": "llama3:8b", "size": int(4.7 * GB)}]
    rec = recommend_coder(installed, int(8 * GB))
    assert rec.model == "qwen2.5-coder:7b" and rec.installed
    assert recommend_coder([], int(5.6 * GB)).model == "qwen2.5-coder:7b"


def test_coding_question_goes_to_the_coder(tmp_path):
    used = []

    class Ollamaish(_Scripted):
        supports_vision = True

        def __init__(self):
            super().__init__([])
            self.model = "general:8b"

        def chat(self, messages):
            used.append(self.model)
            return super().chat(messages)

        def installed_models(self):
            return [{"name": "general:8b", "size": 5 * GB},
                    {"name": "qwen2.5-coder:3b", "size": 2 * GB}]

        def usable_vram(self):
            return (8 * GB, 0)

    backend = Ollamaish()
    agent = Agent(backend, mode=MODE_AUTO, stream_responses=False)
    agent.run("bana bir python fonksiyonu yaz", lambda e: None, lambda r: False)
    agent.run("selam", lambda e: None, lambda r: False)
    assert used == ["qwen2.5-coder:3b", "general:8b"]
    assert backend.model == "general:8b"


# ── project map ─────────────────────────────────────────────────────────────
def test_project_map_outlines_code_without_running_it(tmp_path):
    marker = tmp_path / "ran"
    (tmp_path / "pyproject.toml").write_text(
        'dependencies = ["requests"]\n[tool.ruff]\nline-length = 100\n')
    (tmp_path / "app.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\n"
        "class Store:\n    def save(self): ...\n\ndef main(): ...\n")
    assert is_code_project(tmp_path)
    text = project_map(tmp_path, ["app.py"])
    assert "class Store: save" in text and "def main()" in text
    assert "line-length = 100" in text
    assert not marker.exists()


def test_git_is_called_with_its_command_hooks_off(tmp_path):
    marker = tmp_path / "ran"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.fsmonitor",
                    f"touch {marker}; false"], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    text = project_map(tmp_path, ["a.py"])
    assert "git status" in text
    assert not marker.exists()                  # the repo's fsmonitor never ran
    assert "core.fsmonitor=false" in _SAFE_GIT
